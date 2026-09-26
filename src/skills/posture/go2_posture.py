"""Go2 high-level posture skills exposed by SportClient."""

from __future__ import annotations

from dataclasses import dataclass

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.skill import RobotSkill


class PostureArgs(SkillArgs):
    pass


@dataclass(frozen=True, slots=True)
class PostureSpec:
    skill_name: str
    sdk_action: str
    description: str
    operator_only: bool = False
    dangerous: bool = False


class PostureSkill(RobotSkill[PostureArgs]):
    args_model = PostureArgs

    def __init__(self, spec: PostureSpec) -> None:
        self.spec = spec
        tags = ["posture", "sdk_loco"]
        if spec.operator_only:
            tags.append("operator_only")
        if spec.dangerous:
            tags.append("dangerous")
        self.metadata = SkillMetadata(
            name=spec.skill_name,
            description=spec.description,
            tags=tuple(tags),
            required_resources=("mobile_base",),
            timeout_s=12.0,
            interruptible=not spec.dangerous,
        )

    async def check_preconditions(
        self,
        ctx: SkillContext,
        args: PostureArgs,
    ) -> tuple[bool, str]:
        state = await ctx.robot.get_state()
        if state.hardware and not state.connected:
            return False, "robot is not connected"
        return True, ""

    async def execute(
        self,
        ctx: SkillContext,
        args: PostureArgs,
    ) -> SkillResult:
        await ctx.robot.execute_loco_action(self.spec.sdk_action)
        return SkillResult.ok(
            f"{self.spec.sdk_action} command accepted",
            sdk_action=self.spec.sdk_action,
        )


__all__ = [
    "PostureArgs",
    "PostureSkill",
    "PostureSpec",
]
