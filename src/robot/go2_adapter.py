"""Go2 SportClient adapter for the local unitree_sdk2_cpp bindings.

This adapter owns the process-wide DDS channel; do not share ownership with
another adapter. SDK success means command acceptance, not physical completion.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from .base import ActionVerification, RobotCommandError, RobotState
from .unitree_adapter import ChannelApi

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Go2SportClientApi(Protocol):
    def set_timeout(self, seconds: float) -> None: ...

    def init(self) -> None: ...

    def move(self, vx: float, vy: float, vyaw: float) -> int: ...

    def stop_move(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Go2Bindings:
    channel: ChannelApi
    create_sport_client: Callable[[], Go2SportClientApi]


@dataclass(frozen=True, slots=True)
class UnitreeGo2Config:
    network_interface: str = ""
    domain_id: int = 0
    timeout_s: float = 1.0
    # Application limits, not the hardware's maximum capabilities.
    max_forward_m_s: float = 0.3
    max_lateral_m_s: float = 0.3
    max_yaw_rad_s: float = 0.6

    def __post_init__(self) -> None:
        if (
            isinstance(self.domain_id, bool)
            or not isinstance(self.domain_id, int)
            or self.domain_id < 0
        ):
            raise ValueError("domain_id must be a nonnegative integer")
        for name in (
            "timeout_s",
            "max_forward_m_s",
            "max_lateral_m_s",
            "max_yaw_rad_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and greater than zero")


# Only verified Go2 method names; no G1 FSM IDs or arbitrary getattr dispatch.
_LOCO_ACTIONS = frozenset(
    {
        "stand_up",
        "stand_down",
        "balance_stand",
        "sit",
        "rise_sit",
        "stop_move",
        "damp",
        "recovery_stand",
        "hello",
        "stretch",
    }
)


class UnitreeGo2Adapter:
    """Implement RobotAdapter with explicit rejection of G1 arm operations.

    connect/close only initialize/release communication, never command motion.
    connected reports local initialization, not remote liveness. move_velocity
    sends one velocity sample; callers must refresh it for sustained movement
    and issue stop in their cleanup. No background velocity loop is started.
    """

    def __init__(
        self,
        config: UnitreeGo2Config | None = None,
        *,
        bindings: Go2Bindings | None = None,
    ) -> None:
        self.config = config or UnitreeGo2Config()
        self._bindings = bindings
        self._sport: Go2SportClientApi | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._sport is not None

    async def _run_native(self, operation: str, function: Callable[[], T]) -> T:
        async with self._lock:
            pending = asyncio.create_task(asyncio.to_thread(function))
            try:
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A cancelled to_thread call still runs. Keep serialization until
                # it finishes so a later stop/close cannot overtake the command.
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not pending.cancelled():
                    pending.exception()
                raise
            except RobotCommandError:
                raise
            except Exception as exc:
                raise RobotCommandError(f"Go2 {operation} failed: {exc}") from exc

    async def connect(self) -> None:
        await self._run_native("connect", self._connect_sync)

    def _connect_sync(self) -> None:
        if self.connected:
            return
        bindings = self._bindings or self._load_bindings()
        initialized = False
        try:
            bindings.channel.initialize(
                self.config.domain_id, self.config.network_interface
            )
            initialized = True
            client = bindings.create_sport_client()
            client.set_timeout(self.config.timeout_s)
            client.init()
        except Exception:
            if initialized:
                try:
                    bindings.channel.release()
                except Exception:
                    logger.exception("failed to release Go2 DDS after connect failure")
            raise
        self._bindings = bindings
        self._sport = client

    async def close(self) -> None:
        await self._run_native("close", self._close_sync)

    def _close_sync(self) -> None:
        if self._sport is None:
            return
        self._sport = None
        assert self._bindings is not None
        self._bindings.channel.release()

    async def get_state(self) -> RobotState:
        async with self._lock:
            return RobotState(
                hardware=True,
                connected=self.connected,
                details={
                    "robot_model": "go2",
                    "state_source": "local_client",
                    "telemetry_available": False,
                    "completion_feedback_available": False,
                    "supported_loco_actions": sorted(_LOCO_ACTIONS),
                    "arm_action_presets": False,
                },
            )

    def _require_sport(self) -> Go2SportClientApi:
        if self._sport is None:
            raise RobotCommandError("Go2 is not connected")
        return self._sport

    @staticmethod
    def _require_success(operation: str, status: object) -> None:
        if isinstance(status, bool) or not isinstance(status, int):
            raise RobotCommandError(
                f"Go2 {operation} returned invalid SDK status: {status!r}"
            )
        if status != 0:
            raise RobotCommandError(f"Go2 {operation} failed with SDK status {status}")

    async def stop(self) -> None:
        """Send StopMove; does not disable torque or act as a physical e-stop."""
        await self._run_native(
            "stop_move",
            lambda: self._require_success(
                "stop_move", self._require_sport().stop_move()
            ),
        )

    async def move_velocity(
        self, forward_m_s: float, lateral_m_s: float, yaw_rad_s: float
    ) -> None:
        for name, value, limit in (
            ("forward_m_s", forward_m_s, self.config.max_forward_m_s),
            ("lateral_m_s", lateral_m_s, self.config.max_lateral_m_s),
            ("yaw_rad_s", yaw_rad_s, self.config.max_yaw_rad_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or abs(value) > limit
            ):
                raise RobotCommandError(
                    f"{name} must be finite and within [-{limit}, {limit}]"
                )
        await self._run_native(
            "move",
            lambda: self._require_success(
                "move",
                self._require_sport().move(
                    float(forward_m_s), float(lateral_m_s), float(yaw_rad_s)
                ),
            ),
        )

    async def execute_loco_action(
        self, action: str, arguments: Mapping[str, object] | None = None
    ) -> None:
        if action not in _LOCO_ACTIONS:
            raise RobotCommandError(f"unsupported Go2 loco action: {action}")
        if arguments:
            raise RobotCommandError(f"Go2 {action} does not accept arguments")

        def execute() -> None:
            method = getattr(self._require_sport(), action, None)
            if not callable(method):
                raise RobotCommandError(f"Go2 bindings do not provide {action}")
            self._require_success(action, method())

        await self._run_native(action, execute)

    @staticmethod
    def _unsupported_arm() -> RobotCommandError:
        return RobotCommandError(
            "Go2 does not support G1 arm actions; use Go2 loco action 'hello' "
            "for its native greeting, not G1 wave/handshake/high_five"
        )

    async def wave(self, arm: str) -> None:
        raise self._unsupported_arm()

    async def wait_for_wave_completion(
        self, arm: str, timeout_s: float
    ) -> ActionVerification:
        raise self._unsupported_arm()

    async def execute_arm_action(self, action_id: int, action_name: str) -> None:
        raise self._unsupported_arm()

    async def execute_custom_arm_action(self, action_name: str) -> None:
        raise self._unsupported_arm()

    async def stop_custom_arm_action(self) -> None:
        raise self._unsupported_arm()

    async def wait_for_arm_action_completion(
        self, action_id: int, action_name: str, timeout_s: float
    ) -> ActionVerification:
        raise self._unsupported_arm()

    async def release_arm(self) -> None:
        # No arm exists to release. This permits the shared StopSkill to finish
        # after StopMove without issuing an unrelated quadruped posture command.
        return None

    @staticmethod
    def _load_bindings() -> Go2Bindings:
        try:
            channel = importlib.import_module("unitree_sdk2_cpp.channel")
            go2 = importlib.import_module("unitree_sdk2_cpp.robot.go2")
            return Go2Bindings(
                channel=cast(ChannelApi, channel),
                create_sport_client=go2.SportClient,
            )
        except (ImportError, AttributeError) as exc:
            raise RobotCommandError(
                "Go2 requires unitree_sdk2_cpp bindings with robot.go2.SportClient; "
                "install the adjacent unitree_sdk2_bindings project on the robot host"
            ) from exc
