"""Go2 skill catalog: only SportClient capabilities with matching semantics.

Do not reuse register_g1_skills() for Go2. Arm presets, FSM controls, and G1-only
postures are intentionally omitted. damp and recovery_stand stay operator-only.
"""

from __future__ import annotations

from typing import cast

from pydantic import Field, StrictBool

from core.context import SkillContext
from core.models import SkillArgs, SkillResult
from core.runtime import SkillRuntime
from core.skill import RobotSkill

from .motions import (
    Go2MoveBackwardSkill,
    Go2MoveForwardSkill,
    Go2MoveLeftSkill,
    Go2MoveRightSkill,
    Go2MoveSkill,
    Go2TurnLeftSkill,
    Go2TurnRightSkill,
    StopMoveSkill,
    StopSkill,
)
from .posture import PostureSkill, PostureSpec

# Skill names map 1:1 to SportClient methods documented in docs/go2-adapter.md.
GO2_AUTONOMY_POSTURES = (
    PostureSpec(
        "stand_up",
        "stand_up",
        "Stand the Go2 up using the sport controller.",
    ),
    PostureSpec(
        "stand_down",
        "stand_down",
        "Put the Go2 into the lying down posture.",
    ),
    PostureSpec(
        "balance_stand",
        "balance_stand",
        "Enter Go2 balance standing mode.",
    ),
    PostureSpec(
        "sit",
        "sit",
        "Make the Go2 sit.",
    ),
    PostureSpec(
        "rise_sit",
        "rise_sit",
        "Rise from the Go2 sitting posture.",
    ),
    PostureSpec(
        "hello",
        "hello",
        "Play the Go2 native greeting action (hello).",
    ),
    PostureSpec("stretch", "stretch", "Play Go2's native stretch action (伸懒腰)."),
    PostureSpec("content", "content", "Play Go2's native content action."),
    PostureSpec("heart", "heart", "Play Go2's native heart gesture (比心)."),
    PostureSpec("scrape", "scrape", "Play Go2's native scrape action."),
    PostureSpec("dance1", "dance1", "Play Go2's first native dance (舞蹈一)."),
    PostureSpec("dance2", "dance2", "Play Go2's second native dance (舞蹈二)."),
)

GO2_OPERATOR_POSTURES = (
    PostureSpec(
        "damp",
        "damp",
        "Switch the Go2 to damping mode.",
        operator_only=True,
        dangerous=True,
    ),
    PostureSpec(
        "recovery_stand",
        "recovery_stand",
        "Request Go2 recovery stand.",
        operator_only=True,
    ),
    *(
        PostureSpec(name, name, description, operator_only=True, dangerous=True)
        for name, description in (
            ("front_flip", "Perform Go2's front flip (前空翻)."),
            ("front_jump", "Perform Go2's front jump (前跳)."),
            ("front_pounce", "Perform Go2's front pounce (前扑)."),
            ("left_flip", "Perform Go2's left flip (左侧翻)."),
            ("back_flip", "Perform Go2's back flip (后空翻)."),
        )
    ),
    *(
        PostureSpec(name, name, description, operator_only=True)
        for name, description in (
            ("free_walk", "Select Go2's free walk mode."),
            ("static_walk", "Select Go2's static walk gait."),
            ("trot_run", "Select Go2's trot/run gait."),
            ("economic_gait", "Select Go2's economic gait."),
            (
                "switch_avoid_mode",
                "Invoke Go2's native obstacle-avoidance mode switch.",
            ),
        )
    ),
)

GO2_FLAG_SPECS = (
    PostureSpec("pose", "pose", "Enable/disable Go2 pose mode with flag."),
    *(
        PostureSpec(name, name, description, operator_only=True, dangerous=dangerous)
        for name, description, dangerous in (
            ("hand_stand", "Enable/disable Go2 hand stand (倒立).", True),
            ("free_bound", "Enable/disable Go2 free bound mode.", True),
            ("free_jump", "Enable/disable Go2 free jump mode.", True),
            ("free_avoid", "Enable/disable Go2 free avoidance mode.", False),
            ("classic_walk", "Enable/disable Go2 classic walk mode.", False),
            ("walk_upright", "Enable/disable Go2 upright walking.", True),
            ("cross_step", "Enable/disable Go2 cross step mode.", False),
        )
    ),
)


class Go2FlagArgs(SkillArgs):
    flag: StrictBool = Field(description="Explicit true to enable, false to disable.")


class Go2FlagSkill(RobotSkill[Go2FlagArgs]):
    args_model = Go2FlagArgs

    def __init__(self, spec: PostureSpec) -> None:
        self.spec = spec
        self.metadata = PostureSkill(spec).metadata

    async def check_preconditions(
        self,
        ctx: SkillContext,
        args: Go2FlagArgs,
    ) -> tuple[bool, str]:
        state = await ctx.robot.get_state()
        if state.hardware and not state.connected:
            return False, "robot is not connected"
        return True, ""

    async def execute(self, ctx: SkillContext, args: Go2FlagArgs) -> SkillResult:
        await ctx.robot.execute_loco_action(self.spec.sdk_action, {"flag": args.flag})
        return SkillResult.ok(
            f"{self.spec.sdk_action} command accepted",
            sdk_action=self.spec.sdk_action,
            flag=args.flag,
            completion_verified=False,
        )


def _go2_wave_hello_skill() -> PostureSkill:
    """Social-vision skill name 'wave' backed by the native Go2 hello action.

    The vision policy looks up skills by gesture name. Go2 has no G1 arm wave;
    this skill keeps the wave gesture actionable while documenting the real
    SportClient call. handshake and high_five remain unregistered on purpose.
    """
    return PostureSkill(
        PostureSpec(
            "wave",
            "hello",
            (
                "Greet with the Go2 native hello action when a person "
                "visibly waves. Not a G1 arm wave."
            ),
        )
    )


def build_go2_autonomy_skills() -> tuple[RobotSkill[SkillArgs], ...]:
    skills = (
        *(PostureSkill(spec) for spec in GO2_AUTONOMY_POSTURES),
        *(Go2FlagSkill(spec) for spec in GO2_FLAG_SPECS if not spec.operator_only),
        _go2_wave_hello_skill(),
        Go2MoveForwardSkill(),
        Go2MoveBackwardSkill(),
        Go2MoveLeftSkill(),
        Go2MoveRightSkill(),
        Go2TurnLeftSkill(),
        Go2TurnRightSkill(),
        StopSkill(),
        StopMoveSkill(),
        Go2MoveSkill(),
    )
    return cast(tuple[RobotSkill[SkillArgs], ...], skills)


def build_go2_operator_skills() -> tuple[RobotSkill[SkillArgs], ...]:
    skills = (
        *(PostureSkill(spec) for spec in GO2_OPERATOR_POSTURES),
        *(Go2FlagSkill(spec) for spec in GO2_FLAG_SPECS if spec.operator_only),
    )
    return cast(tuple[RobotSkill[SkillArgs], ...], tuple(skills))


def build_go2_all_skills() -> tuple[RobotSkill[SkillArgs], ...]:
    return build_go2_autonomy_skills() + build_go2_operator_skills()


def register_go2_skills(
    runtime: SkillRuntime,
    *,
    include_operator_only: bool = False,
) -> None:
    skills = (
        build_go2_all_skills() if include_operator_only else build_go2_autonomy_skills()
    )
    for skill in skills:
        runtime.register(skill)


__all__ = [
    "GO2_AUTONOMY_POSTURES",
    "GO2_OPERATOR_POSTURES",
    "build_go2_all_skills",
    "build_go2_autonomy_skills",
    "build_go2_operator_skills",
    "register_go2_skills",
]
