"""Conservative, camera-feedback person following for a single visible target."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.skill import RobotSkill
from core.types import FailureCode, SkillStatus
from perception import CameraFrame


@dataclass(frozen=True, slots=True)
class _Target:
    observed_at_s: float
    count: int
    distance_m: float | None
    center_x: float | None
    obstacle_m: float | None


class FollowPersonSkill(RobotSkill[SkillArgs]):
    """Follow one person using fresh local RGB/depth detections, not VLM timing."""

    metadata = SkillMetadata(
        name="follow_person",
        description=(
            "Continuously follow one visible person (跟着人走/跟随我) using local "
            "D435i depth and image position; keep about 1.5 m away. "
            "Use for a persistent following goal, not a one-step move. "
            "Automatically stops on lost/ambiguous target, stale camera or obstacle."
        ),
        tags=("motion", "go2", "closed_loop"),
        required_resources=("mobile_base",),
        timeout_s=300.0,
    )
    args_model = SkillArgs
    TARGET_M = 1.5
    DISTANCE_HYSTERESIS_M = 0.25
    MAX_FORWARD_M_S = 0.15
    MAX_YAW_RAD_S = 0.3
    FRAME_MAX_AGE_S = 0.5
    # Keep this close to the console depth safety gate.  The previous 1.2 m
    # value treated ordinary floor/background depth as an obstacle and made a
    # valid person target permanently ineligible on many D435i mounts.
    OBSTACLE_STOP_M = 0.55
    TARGET_MISS_GRACE_S = 0.30
    CONTROL_PERIOD_S = 0.02

    def __init__(self) -> None:
        self._target: _Target | None = None
        self._last_valid_target: _Target | None = None

    def observe_frame(self, frame: CameraFrame) -> None:
        observation = frame.observation
        if self._target is not None and frame.observed_at_s < self._target.observed_at_s:
            return
        target = _Target(
            observed_at_s=frame.observed_at_s,
            count=observation.person_count,
            distance_m=observation.nearest_person_distance_m,
            center_x=observation.person_center_x,
            obstacle_m=frame.nearest_obstacle_distance_m,
        )
        self._target = target
        if self._is_valid_target(target):
            self._last_valid_target = target

    @classmethod
    def _is_valid_target(cls, target: _Target | None) -> bool:
        if target is None or target.count != 1:
            return False
        if target.distance_m is None or target.center_x is None:
            return False
        if target.obstacle_m is None or target.obstacle_m <= cls.OBSTACLE_STOP_M:
            return False
        return all(
            math.isfinite(value)
            for value in (target.distance_m, target.center_x, target.obstacle_m)
        )

    def _fresh_target(self) -> _Target | None:
        target = self._target
        now = time.monotonic()
        if target is not None and self._is_valid_target(target):
            if now - target.observed_at_s <= self.FRAME_MAX_AGE_S:
                return target
            return None

        # HOG occasionally misses one frame while the person is still plainly
        # visible.  Keep the last depth-confirmed target for a short, bounded
        # grace period; ambiguity, a close obstacle, and stale data never use
        # this fallback.
        if (
            target is not None
            and target.count == 0
            and self._last_valid_target is not None
            and now - target.observed_at_s <= self.TARGET_MISS_GRACE_S
            and now - self._last_valid_target.observed_at_s <= self.FRAME_MAX_AGE_S
        ):
            return self._last_valid_target
        if target is None or now - target.observed_at_s > self.FRAME_MAX_AGE_S:
            return None
        return None

    async def check_preconditions(self, ctx: SkillContext, args: SkillArgs) -> tuple[bool, str]:
        state = await ctx.robot.get_state()
        if state.hardware and not state.connected:
            return False, "robot is not connected"
        if state.hardware and state.details.get("telemetry_available") is not True:
            return False, "following requires fresh Go2 telemetry"
        if self._fresh_target() is None:
            return False, "following requires one fresh person with depth and clear space"
        return True, ""

    async def execute(self, ctx: SkillContext, args: SkillArgs) -> SkillResult:
        previous: _Target | None = None
        moving = False
        last_state_check_s = 0.0
        try:
            while True:
                now = time.monotonic()
                if now - last_state_check_s >= 0.5:
                    state = await ctx.robot.get_state()
                    last_state_check_s = now
                    if state.hardware and (
                        not state.connected
                        or state.details.get("telemetry_available") is not True
                    ):
                        return SkillResult.fail(
                            SkillStatus.BLOCKED,
                            "following stopped: Go2 telemetry unavailable or error",
                            failure_code=FailureCode.SAFETY_REJECTED,
                        )
                target = self._fresh_target()
                if target is None:
                    return SkillResult.fail(
                        SkillStatus.BLOCKED,
                        "following stopped: person lost, ambiguous, camera stale, or obstacle close",
                        failure_code=FailureCode.SAFETY_REJECTED,
                        recoverable=True,
                    )
                if (
                    previous is not None
                    and target.observed_at_s != previous.observed_at_s
                    and (
                        abs(target.center_x - previous.center_x) > 0.2
                        or abs(target.distance_m - previous.distance_m) > 0.75
                    )
                ):
                    return SkillResult.fail(
                        SkillStatus.BLOCKED,
                        "following stopped: target changed abruptly",
                        failure_code=FailureCode.SAFETY_REJECTED,
                        recoverable=True,
                    )
                previous = target
                offset = target.center_x - 0.5
                yaw = max(-self.MAX_YAW_RAD_S, min(self.MAX_YAW_RAD_S, -offset * 0.8))
                if abs(offset) < 0.1 or target.distance_m < self.TARGET_M - self.DISTANCE_HYSTERESIS_M:
                    yaw = 0.0
                forward = (
                    self.MAX_FORWARD_M_S
                    if target.distance_m > self.TARGET_M + self.DISTANCE_HYSTERESIS_M
                    and abs(offset) < 0.2
                    else 0.0
                )
                # At the desired distance, hold position instead of creeping forward.
                if forward or yaw:
                    await ctx.robot.move_velocity(forward, 0.0, yaw)
                    moving = True
                elif moving:
                    await ctx.robot.stop()
                    moving = False
                await asyncio.sleep(self.CONTROL_PERIOD_S)
        finally:
            # Includes cancellation, timeout, sensor loss, and SDK errors.
            await ctx.robot.stop()


__all__ = ["FollowPersonSkill"]
