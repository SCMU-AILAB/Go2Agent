"""Go2 stop controls shared by text and vision agents."""

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
from core.skill import RobotSkill


class StopArgs(SkillArgs):
    pass


class StopSkill(RobotSkill[StopArgs]):
    metadata = SkillMetadata(
        name="stop",
        description=(
            "Stop now (停/停下/停止/别动): cancels any running skill, including "
            "follow_person, and halts movement. A software stop, not a power-off."
        ),
        tags=("motion", "safety"),
        required_resources=("mobile_base",),
        timeout_s=12.0,
        interruptible=False,
    )
    args_model = StopArgs

    async def execute(self, ctx: SkillContext, args: StopArgs) -> SkillResult:
        await ctx.robot.stop()
        return SkillResult.ok("Go2 software stop command accepted")


__all__ = ["StopArgs", "StopSkill"]
