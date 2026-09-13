"""Direct adapter for the Unitree SDK2 Python bindings."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from .base import ActionVerification, RobotCommandError, RobotState
from .g1_actions import G1_ARM_ACTION_NAMES

logger = logging.getLogger(__name__)

_ARM_ACTION_FEEDBACK_PROBE_S = 1.0
_LOCO_STATE_NOT_AVAILABLE_STATUS = 7301


class ChannelApi(Protocol):
    def initialize(self, domain_id: int = 0, network_interface: str = "") -> None: ...

    def release(self) -> None: ...


class G1LocoClientApi(Protocol):
    def set_timeout(self, seconds: float) -> None: ...

    def init(self) -> None: ...

    def get_fsm_id(self) -> tuple[int, int]: ...

    def get_fsm_mode(self) -> tuple[int, int]: ...

    def get_balance_mode(self) -> tuple[int, int]: ...

    def set_fsm_id(self, fsm_id: int) -> int: ...

    def set_balance_mode(self, balance_mode: int) -> int: ...

    def set_swing_height(self, swing_height: float) -> int: ...

    def set_stand_height(self, stand_height: float) -> int: ...

    def set_velocity(
        self,
        vx: float,
        vy: float,
        omega: float,
        duration: float = 1.0,
    ) -> int: ...

    def set_task_id(self, task_id: int) -> int: ...

    def switch_to_user_ctrl(self) -> int: ...

    def switch_to_internal_ctrl(self, mode: object) -> int: ...

    def _fsm_api(self, parameter: str) -> tuple[int, str] | int: ...

    def damp(self) -> int: ...

    def start(self) -> int: ...

    def squat(self) -> int: ...

    def sit(self) -> int: ...

    def stand_up(self) -> int: ...

    def zero_torque(self) -> int: ...

    def high_stand(self) -> int: ...

    def low_stand(self) -> int: ...

    def balance_stand(self) -> int: ...

    def continuous_gait(self, flag: bool) -> int: ...

    def switch_move_mode(self, flag: bool) -> int: ...

    def wave_hand(self, turn_flag: bool = False) -> int: ...

    def shake_hand(self, stage: int = -1) -> int: ...

    def set_speed_mode(self, mode: int) -> int: ...

    def stop_move(self) -> int: ...

    def move(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        continous_move: bool,
    ) -> int: ...


class G1ArmActionClientApi(Protocol):
    def set_timeout(self, seconds: float) -> None: ...

    def init(self) -> None: ...

    def execute_action(self, action_id: int | str) -> int: ...

    def stop_custom_action(self) -> int: ...


class ArmActionMonitorApi(Protocol):
    def init_channel(self) -> None: ...

    def close_channel(self) -> None: ...


class _StringMessageApi(Protocol):
    @property
    def data(self) -> str: ...


class _ChannelSubscriberFactoryApi(Protocol):
    def __call__(
        self,
        topic: str,
        message_type: object,
        callback: Callable[[object], None],
        queue_length: int = 0,
    ) -> ArmActionMonitorApi: ...


@dataclass(frozen=True, slots=True)
class UnitreeBindings:
    channel: ChannelApi
    create_loco_client: Callable[[], G1LocoClientApi]
    create_arm_action_client: Callable[[], G1ArmActionClientApi] | None = None
    create_arm_action_monitor: (
        Callable[[Callable[[str], None]], ArmActionMonitorApi] | None
    ) = None


@dataclass(frozen=True, slots=True)
class _ArmActionSample:
    sequence: int
    action_id: int
    action_name: str
    holding: bool


@dataclass(frozen=True, slots=True)
class UnitreeG1Config:
    network_interface: str = ""
    domain_id: int = 0
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.domain_id < 0:
            raise ValueError("domain_id must not be negative")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")


class UnitreeG1Adapter:
    """Own the DDS channel and G1 locomotion client used by robot skills.

    Connection only initializes communication. It deliberately does not call
    ``start()`` or any other motion command. Likewise, ``close()`` only releases
    DDS resources; it is not a mechanical stop or physical emergency stop.
    """

    def __init__(
        self,
        config: UnitreeG1Config | None = None,
        *,
        bindings: UnitreeBindings | None = None,
    ) -> None:
        self.config = config or UnitreeG1Config()
        self._bindings = bindings
        self._loco: G1LocoClientApi | None = None
        self._arm_action: G1ArmActionClientApi | None = None
        self._arm_action_monitor: ArmActionMonitorApi | None = None
        self._channel_ready = False
        self._lock = asyncio.Lock()
        self._native_lock = threading.Lock()
        self._arm_action_condition = threading.Condition()
        self._arm_action_sequence = 0
        self._arm_action_samples: deque[_ArmActionSample] = deque(maxlen=32)
        self._arm_action_markers: dict[int, int] = {}
        self._legacy_handshake_active = False

    @property
    def connected(self) -> bool:
        return self._channel_ready and self._loco is not None

    async def connect(self) -> None:
        async with self._lock:
            if self.connected:
                return
            await asyncio.to_thread(self._connect_sync)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_sync)

    async def get_state(self) -> RobotState:
        async with self._lock:
            if not self.connected:
                return RobotState(hardware=True, connected=False)
            details = await self._run_native(
                "get_robot_state",
                self._get_robot_state_sync,
            )
            return RobotState(
                hardware=True,
                connected=True,
                details=details,
            )

    async def stop(self) -> None:
        async with self._lock:
            await self._run_native("stop_move", self._stop_sync)

    async def wave(self, arm: str) -> None:
        if arm != "right":
            raise RobotCommandError(f"unsupported wave arm: {arm}")
        async with self._lock:
            await self._run_native("wave_hand", self._wave_sync)

    async def wait_for_wave_completion(
        self,
        arm: str,
        timeout_s: float,
    ) -> ActionVerification:
        if arm != "right":
            raise RobotCommandError(f"unsupported wave arm: {arm}")
        if timeout_s <= 0:
            raise ValueError("wave verification timeout must be greater than zero")
        verification = await self.wait_for_arm_action_completion(
            25,
            "wave",
            timeout_s,
        )
        return ActionVerification(
            completed=verification.completed,
            observable=verification.observable,
            message=verification.message,
            details={
                **verification.details,
                "wave_observed": verification.details.get(
                    "action_observed",
                    False,
                ),
            },
        )

    async def execute_arm_action(
        self,
        action_id: int,
        action_name: str,
    ) -> None:
        async with self._lock:
            await self._run_native(
                action_name,
                lambda: self._execute_arm_action_sync(action_id, action_name),
            )

    async def execute_custom_arm_action(self, action_name: str) -> None:
        if not action_name.strip():
            raise ValueError("custom arm action name must not be empty")
        async with self._lock:
            await self._run_native(
                action_name,
                lambda: self._execute_custom_arm_action_sync(action_name),
            )

    async def stop_custom_arm_action(self) -> None:
        async with self._lock:
            await self._run_native(
                "stop custom arm action",
                self._stop_custom_arm_action_sync,
            )

    async def wait_for_arm_action_completion(
        self,
        action_id: int,
        action_name: str,
        timeout_s: float,
    ) -> ActionVerification:
        if timeout_s <= 0:
            raise ValueError("arm action timeout must be greater than zero")
        return await asyncio.to_thread(
            self._wait_for_arm_action_completion_sync,
            action_id,
            action_name,
            timeout_s,
        )

    async def release_arm(self) -> None:
        async with self._lock:
            await self._run_native("release arm", self._release_arm_sync)

    async def execute_loco_action(
        self,
        action: str,
        arguments: Mapping[str, object] | None = None,
    ) -> None:
        action_arguments = dict(arguments or {})
        async with self._lock:
            await self._run_native(
                action,
                lambda: self._execute_loco_action_sync(action, action_arguments),
            )

    async def move_velocity(
        self,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
    ) -> None:
        async with self._lock:
            await self._run_native(
                "move",
                lambda: self._move_sync(
                    forward_m_s,
                    lateral_m_s,
                    yaw_rad_s,
                ),
            )

    @staticmethod
    async def _run_native[ResultT](
        operation: str,
        function: Callable[[], ResultT],
    ) -> ResultT:
        try:
            return await asyncio.to_thread(function)
        except RobotCommandError:
            raise
        except Exception as exc:
            raise RobotCommandError(f"{operation} failed: {exc}") from exc

    def _connect_sync(self) -> None:
        with self._native_lock:
            if self.connected:
                return
            bindings = self._bindings or self._load_bindings()
            channel_ready = False
            arm_action_monitor: ArmActionMonitorApi | None = None
            try:
                bindings.channel.initialize(
                    self.config.domain_id,
                    self.config.network_interface,
                )
                channel_ready = True
                client = bindings.create_loco_client()
                client.set_timeout(self.config.timeout_s)
                client.init()
                arm_action = None
                if bindings.create_arm_action_client is not None:
                    arm_action = bindings.create_arm_action_client()
                    arm_action.set_timeout(self.config.timeout_s)
                    arm_action.init()
                if (
                    arm_action is not None
                    and bindings.create_arm_action_monitor is not None
                ):
                    arm_action_monitor = bindings.create_arm_action_monitor(
                        self._on_arm_action_state
                    )
                    arm_action_monitor.init_channel()
            except Exception as exc:
                if arm_action_monitor is not None:
                    try:
                        arm_action_monitor.close_channel()
                    except Exception:
                        logger.exception(
                            "failed to close arm action monitor after connect failure"
                        )
                if channel_ready:
                    try:
                        bindings.channel.release()
                    except Exception:
                        logger.exception(
                            "failed to release Unitree DDS after connect failure"
                        )
                raise RobotCommandError(f"failed to connect to G1: {exc}") from exc

            self._bindings = bindings
            self._loco = client
            self._arm_action = arm_action
            self._arm_action_monitor = arm_action_monitor
            self._channel_ready = True

    def _close_sync(self) -> None:
        with self._native_lock:
            bindings = self._bindings
            arm_action_monitor = self._arm_action_monitor
            self._loco = None
            self._arm_action = None
            self._arm_action_monitor = None
            if not self._channel_ready or bindings is None:
                return
            self._channel_ready = False
            with self._arm_action_condition:
                self._arm_action_markers.clear()
                self._legacy_handshake_active = False
                self._arm_action_condition.notify_all()
            monitor_error: Exception | None = None
            if arm_action_monitor is not None:
                try:
                    arm_action_monitor.close_channel()
                except Exception as exc:  # noqa: BLE001 - release DDS regardless
                    monitor_error = exc
            try:
                bindings.channel.release()
            except Exception as exc:
                raise RobotCommandError(
                    f"failed to release Unitree DDS: {exc}"
                ) from exc
            if monitor_error is not None:
                raise RobotCommandError(
                    f"failed to close arm action monitor: {monitor_error}"
                ) from monitor_error

    def _get_robot_state_sync(self) -> dict[str, object]:
        with self._native_lock:
            client = self._require_loco()
            status, fsm_id = client.get_fsm_id()
            self._require_success("get_fsm_id", status)
            details: dict[str, object] = {"fsm_id": fsm_id}
            unavailable_fields: dict[str, dict[str, object]] = {}
            for key, method_name in (
                ("fsm_mode", "get_fsm_mode"),
                ("balance_mode", "get_balance_mode"),
            ):
                method = getattr(client, method_name, None)
                if not callable(method):
                    continue
                value_status, value = method()
                if value_status != 0:
                    unavailable_fields[key] = {
                        "status": value_status,
                        "reason": self._sdk_status_detail(value_status)
                        or f"{method_name} query failed",
                    }
                    continue
                details[key] = value
            if unavailable_fields:
                details["unavailable_state_fields"] = unavailable_fields
            if self._arm_action is not None:
                details["arm_action_presets"] = True
            return details

    def _stop_sync(self) -> None:
        with self._native_lock:
            status = self._require_loco().stop_move()
            self._require_success("stop_move", status)

    def _wave_sync(self) -> None:
        self._execute_arm_action_sync(25, "face wave")

    def _on_arm_action_state(self, payload: str) -> None:
        try:
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                return
            action_id = decoded.get("id")
            action_name = decoded.get("name")
            holding = decoded.get("holding")
            if (
                not isinstance(action_id, int)
                or isinstance(action_id, bool)
                or not isinstance(action_name, str)
                or not isinstance(holding, bool)
            ):
                return
        except (TypeError, ValueError):
            logger.warning("ignored invalid G1 arm action state: %r", payload)
            return

        with self._arm_action_condition:
            self._arm_action_sequence += 1
            self._arm_action_samples.append(
                _ArmActionSample(
                    sequence=self._arm_action_sequence,
                    action_id=action_id,
                    action_name=action_name,
                    holding=holding,
                )
            )
            self._arm_action_condition.notify_all()

    def _execute_arm_action_sync(
        self,
        action_id: int,
        action_name: str,
    ) -> None:
        if action_id not in G1_ARM_ACTION_NAMES:
            raise RobotCommandError(f"unsupported G1 arm action id: {action_id}")
        if action_id == 99:
            self._release_arm_sync()
            return

        with self._native_lock:
            if self._arm_action is not None:
                with self._arm_action_condition:
                    self._arm_action_markers[action_id] = self._arm_action_sequence
                status = self._arm_action.execute_action(action_id)
                try:
                    self._require_success(action_name, status)
                except RobotCommandError:
                    with self._arm_action_condition:
                        self._arm_action_markers.pop(action_id, None)
                    raise
                return

            with self._arm_action_condition:
                self._arm_action_markers.pop(action_id, None)
            loco = self._require_loco()
            if action_id == 25:
                status = loco.wave_hand(False)
                self._require_success("wave_hand", status)
                return
            if action_id == 27:
                status = loco.shake_hand(0)
                self._require_success("shake_hand stage 0", status)
                self._legacy_handshake_active = True
                return
            raise RobotCommandError(f"{action_name} requires G1ArmActionClient support")

    def _execute_custom_arm_action_sync(self, action_name: str) -> None:
        with self._native_lock:
            if self._arm_action is None:
                raise RobotCommandError(
                    "custom arm actions require G1ArmActionClient support"
                )
            status = self._arm_action.execute_action(action_name)
            self._require_success(action_name, status)

    def _stop_custom_arm_action_sync(self) -> None:
        with self._native_lock:
            if self._arm_action is None:
                raise RobotCommandError(
                    "custom arm actions require G1ArmActionClient support"
                )
            stop_custom_action = getattr(self._arm_action, "stop_custom_action", None)
            if not callable(stop_custom_action):
                raise RobotCommandError("G1 bindings do not provide stop_custom_action")
            self._require_success("stop custom arm action", stop_custom_action())

    def _release_arm_sync(self) -> None:
        with self._native_lock:
            if self._arm_action is not None:
                status = self._arm_action.execute_action(99)
                self._require_success("release arm", status)
                return
            if not self._legacy_handshake_active:
                return
            status = self._require_loco().shake_hand(1)
            self._require_success("shake_hand stage 1", status)
            self._legacy_handshake_active = False

    def _execute_loco_action_sync(
        self,
        action: str,
        arguments: Mapping[str, object],
    ) -> None:
        with self._native_lock:
            loco = self._require_loco()
            no_argument_actions = {
                "damp": "damp",
                "start": "start",
                "squat": "squat",
                "sit": "sit",
                "stand_up": "stand_up",
                "zero_torque": "zero_torque",
                "high_stand": "high_stand",
                "low_stand": "low_stand",
                "balance_stand": "balance_stand",
                "stop_move": "stop_move",
            }
            method_name = no_argument_actions.get(action)
            if method_name is not None:
                if arguments:
                    raise RobotCommandError(
                        f"{action} does not accept arguments: {sorted(arguments)}"
                    )
                function = getattr(loco, method_name, None)
                if not callable(function):
                    raise RobotCommandError(
                        f"G1 bindings do not provide loco action {action}"
                    )
                self._require_success(action, function())
                return

            if action == "wave_with_turn":
                if arguments:
                    raise RobotCommandError(
                        f"{action} does not accept arguments: {sorted(arguments)}"
                    )
                function = getattr(loco, "wave_hand", None)
                if not callable(function):
                    raise RobotCommandError(
                        "G1 bindings do not provide loco action wave_with_turn"
                    )
                self._require_success(action, function(True))
                return

            if action in {"continuous_gait", "switch_move_mode"}:
                enabled = arguments.get("enabled")
                if not isinstance(enabled, bool) or len(arguments) != 1:
                    raise RobotCommandError(
                        f"{action} requires exactly one boolean 'enabled' argument"
                    )
                method = (
                    loco.continuous_gait
                    if action == "continuous_gait"
                    else loco.switch_move_mode
                )
                self._require_success(action, method(enabled))
                return

            if action == "wave_hand":
                turn_flag = arguments.get("turn_flag", False)
                if not isinstance(turn_flag, bool) or len(arguments) > 1:
                    raise RobotCommandError(
                        "wave_hand accepts one optional boolean 'turn_flag' argument"
                    )
                if set(arguments) - {"turn_flag"}:
                    raise RobotCommandError(
                        "wave_hand accepts one optional boolean 'turn_flag' argument"
                    )
                self._require_success(action, loco.wave_hand(turn_flag))
                return

            if action == "shake_hand":
                stage = arguments.get("stage", -1)
                if (
                    not isinstance(stage, int)
                    or isinstance(stage, bool)
                    or stage not in {-1, 0, 1}
                    or len(arguments) > 1
                    or set(arguments) - {"stage"}
                ):
                    raise RobotCommandError(
                        "shake_hand accepts one integer 'stage' argument (-1, 0, or 1)"
                    )
                self._require_success(action, loco.shake_hand(stage))
                return

            if action == "set_speed_mode":
                mode = arguments.get("mode")
                if (
                    not isinstance(mode, int)
                    or isinstance(mode, bool)
                    or len(arguments) != 1
                ):
                    raise RobotCommandError(
                        "set_speed_mode requires exactly one integer 'mode' argument"
                    )
                self._require_success(action, loco.set_speed_mode(mode))
                return

            if action in {
                "set_fsm_id",
                "set_balance_mode",
                "set_swing_height",
                "set_stand_height",
                "set_velocity",
                "move_sdk",
                "set_task_id",
                "switch_to_user_ctrl",
                "switch_to_internal_ctrl",
                "fsm_api",
            }:
                self._execute_low_level_loco_action_sync(action, arguments, loco)
                return

            raise RobotCommandError(f"unsupported G1 loco action: {action}")

    def _execute_low_level_loco_action_sync(
        self,
        action: str,
        arguments: Mapping[str, object],
        loco: G1LocoClientApi,
    ) -> None:
        if action == "set_fsm_id":
            value = self._require_int_argument(action, arguments, "fsm_id")
            self._require_success(action, self._call_loco_method(loco, action, value))
            return
        if action == "set_balance_mode":
            value = self._require_int_argument(action, arguments, "balance_mode")
            self._require_success(action, self._call_loco_method(loco, action, value))
            return
        if action == "set_swing_height":
            value = self._require_float_argument(action, arguments, "swing_height")
            self._require_success(action, self._call_loco_method(loco, action, value))
            return
        if action == "set_stand_height":
            value = self._require_float_argument(action, arguments, "stand_height")
            self._require_success(action, self._call_loco_method(loco, action, value))
            return
        if action == "set_velocity":
            expected = {"vx", "vy", "omega"}
            if set(arguments) not in (expected, expected | {"duration"}):
                raise RobotCommandError(
                    "set_velocity requires vx, vy, omega and optional duration"
                )
            vx = self._require_number_argument(action, arguments, "vx")
            vy = self._require_number_argument(action, arguments, "vy")
            omega = self._require_number_argument(action, arguments, "omega")
            duration = arguments.get("duration", 1.0)
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise RobotCommandError("set_velocity duration must be numeric")
            function = getattr(loco, "set_velocity", None)
            if not callable(function):
                raise RobotCommandError("G1 bindings do not provide set_velocity")
            self._require_success(
                action,
                function(float(vx), float(vy), float(omega), float(duration)),
            )
            return
        if action == "move_sdk":
            expected = {"vx", "vy", "vyaw", "continuous_move"}
            if set(arguments) != expected:
                raise RobotCommandError(
                    "move_sdk requires vx, vy, vyaw, and continuous_move"
                )
            vx = self._require_number_argument(action, arguments, "vx")
            vy = self._require_number_argument(action, arguments, "vy")
            vyaw = self._require_number_argument(action, arguments, "vyaw")
            continuous_move = arguments["continuous_move"]
            if not isinstance(continuous_move, bool):
                raise RobotCommandError(
                    "move_sdk argument 'continuous_move' must be boolean"
                )
            function = getattr(loco, "move", None)
            if not callable(function):
                raise RobotCommandError("G1 bindings do not provide move")
            self._require_success(
                action,
                function(float(vx), float(vy), float(vyaw), continuous_move),
            )
            return
        if action == "set_task_id":
            value = self._require_int_argument(action, arguments, "task_id")
            self._require_success(action, self._call_loco_method(loco, action, value))
            return
        if action == "switch_to_user_ctrl":
            if arguments:
                raise RobotCommandError(f"{action} does not accept arguments")
            self._require_success(
                action,
                self._call_loco_method(loco, "switch_to_user_ctrl"),
            )
            return
        if action == "switch_to_internal_ctrl":
            if set(arguments) != {"mode"} or not isinstance(arguments["mode"], str):
                raise RobotCommandError(
                    "switch_to_internal_ctrl requires mode: last, passive, or walkrun"
                )
            mode = arguments["mode"].lower()
            if mode not in {"last", "passive", "walkrun"}:
                raise RobotCommandError(
                    "switch_to_internal_ctrl requires mode: last, passive, or walkrun"
                )
            self._require_success(
                action,
                self._call_loco_method(
                    loco,
                    action,
                    self._resolve_internal_fsm_mode(mode),
                ),
            )
            return
        if action == "fsm_api":
            if set(arguments) != {"parameter"} or not isinstance(
                arguments["parameter"], str
            ):
                raise RobotCommandError("fsm_api requires one string 'parameter'")
            result = self._call_loco_method(loco, "_fsm_api", arguments["parameter"])
            status = result[0] if isinstance(result, tuple) else result
            if not isinstance(status, int) or isinstance(status, bool):
                raise RobotCommandError("fsm_api returned an invalid SDK status")
            self._require_success(action, status)
            return

    @staticmethod
    def _call_loco_method(
        loco: G1LocoClientApi,
        action: str,
        *args: object,
    ) -> object:
        function = getattr(loco, action, None)
        if not callable(function):
            raise RobotCommandError(f"G1 bindings do not provide {action}")
        return function(*args)

    @staticmethod
    def _require_int_argument(
        action: str,
        arguments: Mapping[str, object],
        name: str,
    ) -> int:
        if set(arguments) != {name}:
            raise RobotCommandError(
                f"{action} requires exactly one integer '{name}' argument"
            )
        value = arguments[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise RobotCommandError(f"{action} argument '{name}' must be an integer")
        return value

    @staticmethod
    def _require_float_argument(
        action: str,
        arguments: Mapping[str, object],
        name: str,
    ) -> float:
        if set(arguments) != {name}:
            raise RobotCommandError(
                f"{action} requires exactly one numeric '{name}' argument"
            )
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RobotCommandError(f"{action} argument '{name}' must be numeric")
        return float(value)

    @staticmethod
    def _require_number_argument(
        action: str,
        arguments: Mapping[str, object],
        name: str,
    ) -> float:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RobotCommandError(f"{action} argument '{name}' must be numeric")
        return float(value)

    @staticmethod
    def _resolve_internal_fsm_mode(mode: str) -> object:
        try:
            module = importlib.import_module("unitree_sdk2_cpp.robot.g1")
            enum_type = module.InternalFsmMode
            return getattr(enum_type, mode.upper())
        except (ImportError, AttributeError):
            # Fake adapters and older bindings may accept the textual enum.
            return mode

    def _wait_for_arm_action_completion_sync(
        self,
        action_id: int,
        action_name: str,
        timeout_s: float,
    ) -> ActionVerification:
        with self._arm_action_condition:
            marker = self._arm_action_markers.get(action_id)
            if self._arm_action_monitor is None or marker is None:
                return ActionVerification(
                    completed=False,
                    observable=False,
                    message=(
                        f"{action_name} command was accepted, but completion "
                        "feedback is unavailable"
                    ),
                    details={
                        "method": "rt/arm/action/state",
                        "action_observed": False,
                    },
                )

            started = time.monotonic()
            deadline = started + timeout_s
            probe_deadline = min(deadline, started + _ARM_ACTION_FEEDBACK_PROBE_S)
            action_sample: _ArmActionSample | None = None
            while True:
                for sample in self._arm_action_samples:
                    if sample.sequence <= marker:
                        continue
                    if action_sample is None and sample.action_id == action_id:
                        action_sample = sample
                        continue
                    if action_sample is not None and sample.action_id == 99:
                        return ActionVerification(
                            completed=True,
                            observable=True,
                            message=f"{action_name} completion verified",
                            details={
                                "method": "rt/arm/action/state",
                                "action_observed": True,
                                "action_id": action_sample.action_id,
                                "action_name": action_sample.action_name,
                                "release_action_id": sample.action_id,
                            },
                        )
                    if (
                        action_sample is not None
                        and sample.sequence > action_sample.sequence
                        and sample.action_id != action_id
                    ):
                        return ActionVerification(
                            completed=False,
                            observable=True,
                            message=(
                                f"{action_name} was interrupted by another arm "
                                "action before completion"
                            ),
                            details={
                                "method": "rt/arm/action/state",
                                "action_observed": True,
                                "interrupting_action_id": sample.action_id,
                                "interrupting_action_name": sample.action_name,
                            },
                        )

                if not self.connected:
                    return ActionVerification(
                        completed=False,
                        observable=True,
                        message=(
                            "robot disconnected before "
                            f"{action_name} completion was verified"
                        ),
                        details={
                            "method": "rt/arm/action/state",
                            "action_observed": action_sample is not None,
                        },
                    )
                active_deadline = (
                    deadline if action_sample is not None else probe_deadline
                )
                remaining = active_deadline - time.monotonic()
                if remaining <= 0:
                    return ActionVerification(
                        completed=False,
                        observable=action_sample is not None,
                        message=f"{action_name} completion feedback timed out",
                        details={
                            "method": "rt/arm/action/state",
                            "action_observed": action_sample is not None,
                            "holding": (
                                action_sample.holding
                                if action_sample is not None
                                else None
                            ),
                        },
                    )
                self._arm_action_condition.wait(timeout=remaining)

    # Kept for callers written against the original wave-only adapter.
    def _wait_for_wave_completion_sync(
        self,
        timeout_s: float,
    ) -> ActionVerification:
        verification = self._wait_for_arm_action_completion_sync(25, "wave", timeout_s)
        return ActionVerification(
            completed=verification.completed,
            observable=verification.observable,
            message=verification.message,
            details={
                **verification.details,
                "wave_observed": verification.details.get("action_observed", False),
            },
        )

    def _move_sync(
        self,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
    ) -> None:
        with self._native_lock:
            status = self._require_loco().move(
                forward_m_s,
                lateral_m_s,
                yaw_rad_s,
                False,
            )
            self._require_success("move", status)

    def _require_loco(self) -> G1LocoClientApi:
        if self._loco is None or not self._channel_ready:
            raise RobotCommandError("G1 adapter is not connected")
        return self._loco

    @staticmethod
    def _require_success(operation: str, status: int) -> None:
        if status != 0:
            detail = UnitreeG1Adapter._sdk_status_detail(status)
            suffix = f": {detail}" if detail else ""
            raise RobotCommandError(
                f"{operation} failed with SDK status {status}{suffix}"
            )

    @staticmethod
    def _sdk_status_detail(status: int) -> str | None:
        return {
            _LOCO_STATE_NOT_AVAILABLE_STATUS: "LocoState is not available",
            7302: "invalid locomotion FSM id",
            7303: "invalid locomotion task id",
            7400: "the rt/armsdk topic is occupied",
            7401: "the arm is holding; release action 99 first",
            7402: "invalid arm action id",
            7404: ("arm actions require FSM 500, 501, or 801 (FSM 801 modes 0 or 3)"),
        }.get(status)

    @staticmethod
    def _load_bindings() -> UnitreeBindings:
        try:
            channel_module = importlib.import_module("unitree_sdk2_cpp.channel")
            g1_module = importlib.import_module("unitree_sdk2_cpp.robot.g1")
            channel = cast(ChannelApi, cast(object, channel_module))
            loco_factory = cast(
                Callable[[], G1LocoClientApi],
                cast(object, getattr(g1_module, "LocoClient")),  # noqa: B009
            )
            arm_factory_object = getattr(g1_module, "G1ArmActionClient", None)
            arm_factory = (
                cast(Callable[[], G1ArmActionClientApi], arm_factory_object)
                if callable(arm_factory_object)
                else None
            )
            monitor_factory = UnitreeG1Adapter._load_arm_action_monitor_factory(
                channel_module
            )
            return UnitreeBindings(
                channel=channel,
                create_loco_client=loco_factory,
                create_arm_action_client=arm_factory,
                create_arm_action_monitor=monitor_factory,
            )
        except (ImportError, AttributeError) as exc:
            raise RobotCommandError(
                "Unitree SDK2 Python bindings are unavailable; install "
                "~/unitree_sdk2/unitree_sdk2_bindings on the Linux robot host"
            ) from exc

    @staticmethod
    def _load_arm_action_monitor_factory(
        channel_module: object,
    ) -> Callable[[Callable[[str], None]], ArmActionMonitorApi] | None:
        try:
            ros2_module = importlib.import_module("unitree_sdk2_cpp.idl.ros2")
            string_type = getattr(ros2_module, "String")  # noqa: B009
            subscriber_factory = cast(
                _ChannelSubscriberFactoryApi,
                cast(object, getattr(channel_module, "ChannelSubscriber")),  # noqa: B009
            )
        except (ImportError, AttributeError):
            return None

        def create_monitor(callback: Callable[[str], None]) -> ArmActionMonitorApi:
            def on_message(message: object) -> None:
                callback(cast(_StringMessageApi, message).data)

            return subscriber_factory(
                "rt/arm/action/state",
                string_type,
                on_message,
                8,
            )

        return create_monitor
