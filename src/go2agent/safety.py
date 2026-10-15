"""SafetyGate: the only path from behaviors to robot motion.

Enforces speed limits, depth-based obstacle stop, stale-frame stop, an e-stop latch, a dangerous-action
allowlist, and a watchdog thread that stops the robot if the control loop stops refreshing commands.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from go2agent.core.types import ActionSpec, Event, EventKind, Frame, Obstacles, Velocity
from go2agent.robot.base import Robot

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    max_vx: float = 0.5
    max_back_vx: float = 0.3
    max_vy: float = 0.3
    max_wz: float = 0.8
    obstacle_distance: float = 0.4  # m; camera depth: forward motion blocked below this
    # Obstacle clearances from the robot (LiDAR height map), measured from the body edge:
    stop_front: float = 0.25  # m; no forward motion below this
    stop_back: float = 0.20
    stop_side: float = 0.15
    slow_gain: float = 1.0  # allowed speed toward an obstacle = gain * (clearance - stop), m/s per m
    obstacle_timeout: float = (
        0.5  # s; older clearances block all translation (turning in place stays allowed)
    )
    frame_timeout: float = 0.5  # s; all motion stops when the newest frame is older
    watchdog_timeout: float = 0.2  # s; robot stopped if the loop stops calling apply()
    allow_dangerous: bool = False


@dataclass(frozen=True, slots=True)
class SafetyResult:
    applied: Velocity
    reasons: tuple[str, ...] = ()  # why the command was modified, empty if passed through
    events: tuple[Event, ...] = ()


def front_distance(depth: np.ndarray) -> float | None:
    """Robust nearest distance in the central region in front of the robot, or None if no valid depth."""
    h, w = depth.shape
    region = depth[int(h * 0.3) : int(h * 0.85), int(w * 0.3) : int(w * 0.7)]
    valid = region[np.isfinite(region) & (region > 0.1)]
    if valid.size < 50:
        return None
    return float(np.percentile(valid, 5))


def _stop(direction: str, c: SafetyConfig) -> float:
    return {"front": c.stop_front, "back": c.stop_back}.get(direction, c.stop_side)


def limit_by_clearance(
    vx: float, vy: float, obs: Obstacles, c: SafetyConfig
) -> tuple[float, float, dict[str, float]]:
    """Slow, then stop, each velocity component that heads toward a close obstacle.

    Returns the limited (vx, vy) and {direction: clearance} for every component that was reduced.
    """
    limited: dict[str, float] = {}

    def cap(speed: float, direction: str, clearance: float) -> float:
        allowed = max(0.0, c.slow_gain * (clearance - _stop(direction, c)))
        if speed > allowed:
            limited[direction] = clearance
            return allowed
        return speed

    if vx > 0:
        vx = cap(vx, "front", obs.front)
    elif vx < 0:
        vx = -cap(-vx, "back", obs.back)
    if vy > 0:
        vy = cap(vy, "left", obs.left)
    elif vy < 0:
        vy = -cap(-vy, "right", obs.right)
    return vx, vy, limited


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SafetyGate:
    def __init__(
        self, robot: Robot, config: SafetyConfig | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.robot = robot
        self.config = config or SafetyConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._estopped = False
        self._last_apply = clock()
        self._moving = False
        self._blocked: set[str] = set()  # directions currently blocked by an obstacle
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # --- e-stop -----------------------------------------------------------------------------------

    @property
    def estopped(self) -> bool:
        return self._estopped

    def estop(self) -> None:
        """Latch: stop now and refuse motion until release()."""
        with self._lock:
            self._estopped = True
            self._moving = False
        self.robot.stop()

    def release(self) -> None:
        with self._lock:
            self._estopped = False

    # --- commands ---------------------------------------------------------------------------------

    def apply(self, cmd: Velocity | None, frame: Frame | None) -> SafetyResult:
        """Filter and send a velocity. Call every control tick; `None` means hold still."""
        now = self._clock()
        c = self.config
        reasons: list[str] = []
        events: list[Event] = []
        with self._lock:
            self._last_apply = now
            if cmd is None or cmd.is_zero:
                out = Velocity()
            elif self._estopped:
                out, reasons = Velocity(), ["estop"]
            elif frame is None or now - frame.stamp > c.frame_timeout:
                out, reasons = Velocity(), ["stale_frame"]
            else:
                vx = _clamp(cmd.vx, -c.max_back_vx, c.max_vx)
                vy = _clamp(cmd.vy, -c.max_vy, c.max_vy)
                wz = _clamp(cmd.wz, -c.max_wz, c.max_wz)
                if (vx, vy, wz) != (cmd.vx, cmd.vy, cmd.wz):
                    reasons.append("speed_limit")
                blocked: dict[str, float] = {}
                if frame.depth is not None and vx > 0:
                    dist = front_distance(frame.depth)
                    if dist is not None and dist < c.obstacle_distance:
                        vx = 0.0
                        blocked["front"] = dist
                obs = self.robot.obstacles()
                if obs is not None and (vx or vy):
                    if now - obs.stamp > c.obstacle_timeout:
                        vx = vy = 0.0
                        reasons.append("obstacle_sensor_stale")
                    else:
                        vx, vy, limited = limit_by_clearance(vx, vy, obs, c)
                        if limited:
                            reasons.append("obstacle_slowdown")
                        blocked.update({k: v for k, v in limited.items() if v <= _stop(k, c)})
                if blocked:
                    reasons.append("obstacle")
                for direction, dist in blocked.items():
                    if direction not in self._blocked:
                        data = {"direction": direction, "distance": round(dist, 2)}
                        events.append(Event(EventKind.OBSTACLE_CLOSE, now, data))
                self._blocked = set(blocked)
                out = Velocity(vx, vy, wz)
            send = not out.is_zero or self._moving
            self._moving = not out.is_zero
        if send:  # avoid spamming zero commands while idle (would fight native actions)
            self.robot.velocity(out)
        return SafetyResult(out, tuple(reasons), tuple(events))

    def perform(self, action: str) -> str | None:
        """Start a native action. Returns a refusal reason, or None if started."""
        spec = next((a for a in self.robot.actions() if a.name == action), None)
        if spec is None:
            return f"unknown action {action!r}"
        if "dangerous" in spec.tags and not self.config.allow_dangerous:
            return f"action {action!r} is dangerous and not enabled"
        with self._lock:
            if self._estopped:
                return "estop"
        st = self.robot.state()
        if not st.connected:
            return "robot_not_connected"
        if st.busy:
            return "robot_busy"
        with self._lock:
            self._moving = False
        return self.robot.perform(action)

    def allowed_actions(self) -> tuple[ActionSpec, ...]:
        return tuple(
            a for a in self.robot.actions() if self.config.allow_dangerous or "dangerous" not in a.tags
        )

    # --- watchdog ---------------------------------------------------------------------------------

    def start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=self._watch, name="safety-watchdog", daemon=True)
        self._watchdog.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog:
            self._watchdog.join(timeout=1)

    def check_watchdog(self) -> bool:
        """Stop the robot if the control loop stalled while moving. Returns True if it fired."""
        with self._lock:
            stalled = self._moving and self._clock() - self._last_apply > self.config.watchdog_timeout
            if stalled:
                self._moving = False
        if stalled:
            log.warning("control loop stalled; stopping robot")
            self.robot.stop()
        return stalled

    def _watch(self) -> None:
        period = self.config.watchdog_timeout / 4
        while not self._watchdog_stop.wait(period):
            self.check_watchdog()
