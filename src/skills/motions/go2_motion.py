"""Bounded Go2 locomotion skills that refresh velocity for SportClient.

Go2 SportClient.move is a single velocity sample. Sustained motion requires a
periodic refresh (~20 ms in the official SDK examples) and an explicit stop in
cleanup. close() only releases DDS and must not be treated as a stop.
"""

from __future__ import annotations

import asyncio
import math
import time

from pydantic import Field

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.skill import RobotSkill

GO2_VELOCITY_REFRESH_S = 0.02


class Go2LinearMoveArgs(SkillArgs):
    distance_m: float = Field(default=0.2, ge=0.05, le=0.3)


class Go2TurnArgs(SkillArgs):
    angle_deg: float = Field(default=15.0, ge=5.0, le=45.0)


class Go2MoveArgs(SkillArgs):
    forward_m_s: float = Field(default=0.1, ge=-0.3, le=0.3)
    lateral_m_s: float = Field(default=0.0, ge=-0.3, le=0.3)
    yaw_rad_s: float = Field(default=0.0, ge=-0.6, le=0.6)
    duration_s: float = Field(default=0.5, ge=0.1, le=2.0)


class _Go2BoundedMotionSkill[ArgsT: SkillArgs](RobotSkill[ArgsT]):
    async def check_preconditions(
        self,
        ctx: SkillContext,
        args: ArgsT,
    ) -> tuple[bool, str]:
        state = await ctx.robot.get_state()
        if state.hardware and not state.connected:
            return False, "robot is not connected"
        return True, ""

    async def cleanup(self, ctx: SkillContext, args: ArgsT) -> None:
        if (
            ctx.runtime_data.get("motion_attempted") is True
            and ctx.runtime_data.get("motion_stopped") is not True
        ):
            await ctx.robot.stop()

    async def _run_refreshed_velocity(
        self,
        ctx: SkillContext,
        *,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
        duration_s: float,
    ) -> None:
        ctx.runtime_data["motion_attempted"] = True
        deadline = time.monotonic() + duration_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await ctx.robot.move_velocity(forward_m_s, lateral_m_s, yaw_rad_s)
                await asyncio.sleep(min(GO2_VELOCITY_REFRESH_S, remaining))
        finally:
            await ctx.robot.stop()
            ctx.runtime_data["motion_stopped"] = True


class _Go2LinearMoveSkill(_Go2BoundedMotionSkill[Go2LinearMoveArgs]):
    args_model = Go2LinearMoveArgs
    forward_sign = 0.0
    lateral_sign = 0.0
    result_label = "movement"

    async def execute(
        self,
        ctx: SkillContext,
        args: Go2LinearMoveArgs,
    ) -> SkillResult:
        speed_m_s = min(max(args.distance_m, 0.1), 0.3)
        duration_s = args.distance_m / speed_m_s
        await self._run_refreshed_velocity(
            ctx,
            forward_m_s=self.forward_sign * speed_m_s,
            lateral_m_s=self.lateral_sign * speed_m_s,
            yaw_rad_s=0.0,
            duration_s=duration_s,
        )
        return SkillResult.ok(
            f"{self.result_label} completed",
            distance_m=args.distance_m,
            speed_m_s=speed_m_s,
            duration_s=duration_s,
            velocity_refresh_s=GO2_VELOCITY_REFRESH_S,
        )


class Go2MoveForwardSkill(_Go2LinearMoveSkill):
    metadata = SkillMetadata(
        name="move_forward",
        description=(
            "Move the Go2 forward a short bounded open-loop distance "
            "(向前走/前进), then stop. Max 0.3 m. Call stand_up first if "
            "not already standing. Not obstacle-aware navigation."
        ),
        tags=("motion", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    forward_sign = 1.0
    result_label = "forward movement"


class Go2MoveBackwardSkill(_Go2LinearMoveSkill):
    metadata = SkillMetadata(
        name="move_backward",
        description=(
            "Move the Go2 backward a short bounded open-loop distance "
            "(后退/向后走), then stop. Max 0.3 m. Prefer stand_up first. "
            "Safety-focused; not reverse navigation."
        ),
        tags=("motion", "safety", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    forward_sign = -1.0
    result_label = "backward movement"


class Go2MoveLeftSkill(_Go2LinearMoveSkill):
    metadata = SkillMetadata(
        name="move_left",
        description=(
            "Move the Go2 left (向左走/左侧移) a short bounded distance, "
            "then stop. Max 0.3 m. Lateral step, not a turn."
        ),
        tags=("motion", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    lateral_sign = 1.0
    result_label = "left movement"


class Go2MoveRightSkill(_Go2LinearMoveSkill):
    metadata = SkillMetadata(
        name="move_right",
        description=(
            "Move the Go2 right (向右走/右侧移) a short bounded distance, "
            "then stop. Max 0.3 m. Lateral step, not a turn."
        ),
        tags=("motion", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    lateral_sign = -1.0
    result_label = "right movement"


class _Go2TurnSkill(_Go2BoundedMotionSkill[Go2TurnArgs]):
    args_model = Go2TurnArgs
    yaw_sign = 1.0
    result_label = "turn"

    async def execute(
        self,
        ctx: SkillContext,
        args: Go2TurnArgs,
    ) -> SkillResult:
        angle_rad = math.radians(args.angle_deg)
        yaw_rad_s = min(max(angle_rad, 0.25), 0.6)
        duration_s = angle_rad / yaw_rad_s
        await self._run_refreshed_velocity(
            ctx,
            forward_m_s=0.0,
            lateral_m_s=0.0,
            yaw_rad_s=self.yaw_sign * yaw_rad_s,
            duration_s=duration_s,
        )
        return SkillResult.ok(
            f"{self.result_label} completed",
            angle_deg=args.angle_deg,
            yaw_rad_s=yaw_rad_s,
            duration_s=duration_s,
            velocity_refresh_s=GO2_VELOCITY_REFRESH_S,
        )


class Go2TurnLeftSkill(_Go2TurnSkill):
    metadata = SkillMetadata(
        name="turn_left",
        description=(
            "Yaw the Go2 left by a small bounded angle (左转/向左转), "
            "then stop. In-place turn, not lateral move_left."
        ),
        tags=("motion", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    yaw_sign = 1.0
    result_label = "left turn"


class Go2TurnRightSkill(_Go2TurnSkill):
    metadata = SkillMetadata(
        name="turn_right",
        description=(
            "Yaw the Go2 right by a small bounded angle (右转/向右转), "
            "then stop. In-place turn, not lateral move_right."
        ),
        tags=("motion", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    yaw_sign = -1.0
    result_label = "right turn"


class Go2MoveSkill(_Go2BoundedMotionSkill[Go2MoveArgs]):
    """Bounded Go2 SportClient.move with the official ~20 ms refresh cadence."""

    metadata = SkillMetadata(
        name="move",
        description=(
            "Low-level bounded velocity move with optional yaw (带速度移动), "
            "refreshed until duration_s ends. Prefer named move_forward/"
            "turn_left skills unless combined velocity is required. "
            "Not navigation; max ~0.3 m/s."
        ),
        tags=("motion", "sdk_loco", "go2"),
        required_resources=("mobile_base",),
        timeout_s=6.0,
    )
    args_model = Go2MoveArgs

    async def execute(self, ctx: SkillContext, args: Go2MoveArgs) -> SkillResult:
        await self._run_refreshed_velocity(
            ctx,
            forward_m_s=args.forward_m_s,
            lateral_m_s=args.lateral_m_s,
            yaw_rad_s=args.yaw_rad_s,
            duration_s=args.duration_s,
        )
        return SkillResult.ok(
            "bounded Go2 movement completed",
            forward_m_s=args.forward_m_s,
            lateral_m_s=args.lateral_m_s,
            yaw_rad_s=args.yaw_rad_s,
            duration_s=args.duration_s,
            velocity_refresh_s=GO2_VELOCITY_REFRESH_S,
        )


__all__ = [
    "GO2_VELOCITY_REFRESH_S",
    "Go2LinearMoveArgs",
    "Go2MoveArgs",
    "Go2MoveBackwardSkill",
    "Go2MoveForwardSkill",
    "Go2MoveLeftSkill",
    "Go2MoveRightSkill",
    "Go2MoveSkill",
    "Go2TurnArgs",
    "Go2TurnLeftSkill",
    "Go2TurnRightSkill",
]
