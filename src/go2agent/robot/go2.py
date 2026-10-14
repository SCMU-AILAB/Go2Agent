"""Go2 driver for the Linux unitree_sdk2_cpp bindings; imports lazily."""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from go2agent.core.types import ActionSpec, RobotState, Velocity
from go2agent.robot.base import Robot
from go2agent.robot.go2_actions import GO2_ACTIONS

logger = logging.getLogger(__name__)

# Python binding names, verified against unitree_sdk2_bindings' SportClient stub.
ACTION_METHODS = {
    "stand_up": "stand_up",
    "stand_down": "stand_down",
    "balance_stand": "balance_stand",
    "sit": "sit",
    "rise_sit": "rise_sit",
    "hello": "hello",
    "stretch": "stretch",
    "content": "content",
    "heart": "heart",
    "scrape": "scrape",
    "dance1": "dance1",
    "dance2": "dance2",
    "front_jump": "front_jump",
    "front_pounce": "front_pounce",
    "front_flip": "front_flip",
    "back_flip": "back_flip",
    "left_flip": "left_flip",
}
_ACTIONS = {spec.name: spec for spec in GO2_ACTIONS}


@dataclass(frozen=True, slots=True)
class Go2Config:
    network_interface: str = "eth0"
    domain_id: int = 0
    timeout_s: float = 0.2
    command_timeout_s: float = 0.5
    telemetry_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if type(self.domain_id) is not int or self.domain_id < 0:
            raise ValueError("domain_id must be a nonnegative integer")
        for value in (self.timeout_s, self.command_timeout_s, self.telemetry_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be positive and finite")


@dataclass(frozen=True, slots=True)
class Go2Bindings:
    """Inject SDK factories for tests; one driver owns the process-wide DDS channel."""

    channel: Any
    create_sport_client: Callable[[], Any]
    create_state_subscriber: Callable[[Callable[[Any], None]], Any]


class Go2Robot(Robot):
    """Serialize SDK calls on one worker, keeping control-loop methods non-blocking.

    Stop discards pending commands and runs immediately after any in-flight RPC;
    native calls cannot be interrupted safely. Its latency is bounded by the SDK
    timeout when the SDK honors that timeout. Busy uses estimated action duration,
    not physical completion feedback. New actions are refused while busy/stopping.
    Velocity waits behind a pending stop, but is dropped while busy or not ready.
    Operational refusals are recorded in state.details["command_refusal"], which
    clears when a command is accepted; they do not raise in the control loop.
    """

    def __init__(
        self,
        config: Go2Config | None = None,
        *,
        bindings: Go2Bindings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or Go2Config()
        self._bindings = bindings
        self._clock = clock
        self._condition = threading.Condition()
        self._client: Any | None = None
        self._subscriber: Any | None = None
        self._worker: threading.Thread | None = None
        self._closing = False
        self._telemetry: dict[str, Any] = {}
        self._telemetry_stamp: float | None = None
        self._pending_velocity: tuple[Velocity, float] | None = None
        self._pending_action: str | None = None
        self._stop_pending = False
        self._stopped = False
        self._zero_velocity = False
        self._moving = False
        self._velocity_stamp = 0.0
        self._action_running = False
        self._busy_until = 0.0
        self._last_error: str | None = None
        self._command_refusal: str | None = None

    def connect(self) -> None:
        if self._client is not None:
            return
        bindings = self._bindings or self._load_bindings()
        subscriber = None
        initialized = False
        try:
            bindings.channel.initialize(self.config.domain_id, self.config.network_interface)
            initialized = True
            client = bindings.create_sport_client()
            client.set_timeout(self.config.timeout_s)
            client.init()
            with self._condition:
                self._telemetry = {}
                self._telemetry_stamp = None
            subscriber = bindings.create_state_subscriber(self._on_state)
            subscriber.init_channel()
        except Exception:
            try:
                if subscriber is not None:
                    subscriber.close_channel()
            finally:
                if initialized:
                    bindings.channel.release()
            raise
        with self._condition:
            self._bindings, self._client, self._subscriber = bindings, client, subscriber
            self._closing = False
            self._stopped = self._zero_velocity = False
            self._stop_pending = self._moving = self._action_running = False
            self._pending_action = self._pending_velocity = None
            self._busy_until = 0.0
            self._last_error = None
            self._command_refusal = None
        self._worker = threading.Thread(target=self._run, name="go2-sport", daemon=True)
        self._worker.start()

    def close(self) -> None:
        if self._client is None:
            return
        self.stop()
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=self.config.timeout_s * 2 + 2)
            if self._worker.is_alive():
                raise TimeoutError("Go2 SDK call did not finish; DDS remains open until the worker stops")
        try:
            if self._subscriber is not None:
                self._subscriber.close_channel()
        finally:
            try:
                if self._bindings is not None:
                    self._bindings.channel.release()
            finally:
                with self._condition:
                    self._client = self._subscriber = self._worker = None

    def velocity(self, cmd: Velocity) -> None:
        if not all(math.isfinite(v) for v in (cmd.vx, cmd.vy, cmd.wz)):
            raise ValueError("Velocity must be finite")
        with self._condition:
            reason = self._not_ready_reason()
            if reason is not None:
                self._refuse(reason)
                return
            if self._busy():
                self._refuse("Go2 is busy with a native action")
                return
            self._command_refusal = None
            if cmd.is_zero:
                if not self._zero_velocity:
                    self.stop()
                return
            # Keep the latest velocity while stop_move is pending/in flight.
            # The worker dispatches it only after that stop succeeds.
            self._zero_velocity = self._stopped = False
            self._velocity_stamp = self._clock()
            self._pending_velocity = (cmd, self._velocity_stamp)
            self._condition.notify_all()

    def perform(self, action: str) -> str | None:
        if action not in ACTION_METHODS:
            raise ValueError(f"Unknown Go2 action {action!r}")
        with self._condition:
            reason = self._not_ready_reason()
            if reason is not None:
                return self._refuse(reason)
            if self._busy():
                return self._refuse("Go2 is busy with a native action")
            if self._stop_pending:
                return self._refuse("Go2 is stopping")
            if not callable(getattr(self._client, ACTION_METHODS[action], None)):
                return self._refuse(f"Go2 bindings do not provide {ACTION_METHODS[action]}")
            self._command_refusal = None
            if not self._zero_velocity and (self._moving or self._pending_velocity is not None):
                self._stop_pending = True
            self._pending_velocity = None
            self._pending_action = action
            self._zero_velocity = True  # Idle ticks must not cancel a native action.
            self._stopped = False
            self._condition.notify_all()
        return None

    def stop(self) -> None:
        with self._condition:
            self._pending_action = self._pending_velocity = None
            self._busy_until = 0.0
            self._zero_velocity = True
            if self._client is not None and not self._stopped:
                self._stop_pending = True
            self._condition.notify_all()

    def actions(self) -> tuple[ActionSpec, ...]:
        return GO2_ACTIONS

    def state(self) -> RobotState:
        with self._condition:
            now = self._clock()
            age = None if self._telemetry_stamp is None else max(0.0, now - self._telemetry_stamp)
            return RobotState(
                stamp=now,
                connected=self._client is not None and not self._closing and self._fresh_telemetry(),
                busy=self._busy(),
                details={
                    **self._telemetry,
                    "telemetry_age": age,
                    "last_error": self._last_error,
                    "command_refusal": self._command_refusal,
                    "stopping": self._stop_pending,
                },
            )

    def _busy(self) -> bool:
        return self._pending_action is not None or self._action_running or self._clock() < self._busy_until

    def _fresh_telemetry(self) -> bool:
        return (
            self._telemetry_stamp is not None
            and self._clock() - self._telemetry_stamp <= self.config.telemetry_timeout_s
        )

    def _not_ready_reason(self) -> str | None:
        if self._client is None or self._closing:
            return "Go2 is not connected"
        if not self._fresh_telemetry():
            return "Go2 telemetry is missing or stale"
        return None

    def _refuse(self, reason: str) -> str:
        if reason != self._command_refusal:
            logger.debug("Go2 command refused: %s", reason)
        self._command_refusal = reason
        return reason

    def _on_state(self, message: Any) -> None:
        try:
            snapshot = {
                "mode": int(message.mode),
                "gait_type": int(message.gait_type),
                "position": tuple(message.position),
                "velocity": tuple(message.velocity),
                "rpy": tuple(message.imu_state.rpy),
                "error_code": int(message.error_code),
            }
            with self._condition:
                self._telemetry = snapshot
                self._telemetry_stamp = self._clock()
        except Exception:
            logger.exception("Could not copy Go2 telemetry")

    def _run(self) -> None:
        while True:
            with self._condition:
                now = self._clock()
                if self._moving and (
                    now - self._velocity_stamp > self.config.command_timeout_s or not self._fresh_telemetry()
                ):
                    self._pending_velocity = None
                    self._stop_pending = True
                    self._zero_velocity = True
                if self._stop_pending:
                    method, args = "stop_move", ()
                elif self._closing:
                    return
                elif self._pending_action is not None:
                    action, self._pending_action = self._pending_action, None
                    if not self._fresh_telemetry():
                        continue
                    method, args = ACTION_METHODS[action], ()
                    self._action_running = True
                    self._stopped = False
                    self._busy_until = now + _ACTIONS[action].duration
                elif self._pending_velocity is not None:
                    (cmd, stamp), self._pending_velocity = self._pending_velocity, None
                    if now - stamp > self.config.command_timeout_s or not self._fresh_telemetry():
                        continue
                    method, args = "move", (cmd.vx, cmd.vy, cmd.wz)
                    self._stopped = False  # A queued stop may have completed since submission.
                    self._moving = True  # A timed-out RPC may still reach the robot.
                else:
                    self._condition.wait(timeout=0.02)
                    continue
            error = None
            try:
                status = getattr(self._client, method)(*args)
                if type(status) is not int or status != 0:
                    raise RuntimeError(f"Go2 {method} returned SDK status {status!r}")
            except Exception as exc:
                error = str(exc)
                logger.exception("Go2 %s failed", method)
            with self._condition:
                if method == "stop_move":
                    self._stop_pending = False
                    self._stopped = error is None
                    self._moving = False
                elif method != "move":
                    self._action_running = False
                if error is not None:
                    self._last_error = error
                    self._pending_action = self._pending_velocity = None
                    self._busy_until = 0.0
                    self._zero_velocity = True
                    if method != "stop_move":
                        self._stop_pending = True
                self._condition.notify_all()

    @staticmethod
    def _load_bindings() -> Go2Bindings:
        try:
            channel = importlib.import_module("unitree_sdk2_cpp.channel")
            go2 = importlib.import_module("unitree_sdk2_cpp.robot.go2")
            idl = importlib.import_module("unitree_sdk2_cpp.idl.go2")
        except ImportError as exc:
            raise RuntimeError(
                "Install unitree_sdk2_bindings on Linux for the robot's Python interpreter"
            ) from exc
        return Go2Bindings(
            channel,
            go2.SportClient,
            lambda callback: channel.ChannelSubscriber(
                "rt/sportmodestate", idl.SportModeState, callback, queue_length=1
            ),
        )
