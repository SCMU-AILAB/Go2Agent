"""Go2 skill catalog: only SportClient capabilities with matching semantics.

Do not reuse register_g1_skills() for Go2. Arm presets, FSM controls, and G1-only
postures are intentionally omitted. damp and recovery_stand stay operator-only.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import cast

from pydantic import Field, StrictBool

from core.context import SkillContext
from core.models import SkillArgs, SkillMetadata, SkillResult
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
from .motions.go2_follow import FollowPersonSkill
from .posture import PostureSkill, PostureSpec

# Safe random-dance pool only. Never operator-only or dangerous actions.
GO2_RANDOM_DANCE_POOL: tuple[str, ...] = ("dance1", "dance2")

# Skill names map 1:1 to SportClient methods documented in docs/go2-adapter.md.
# Descriptions are tool-selection text for the cloud LLM: include Chinese
# trigger phrases, when to use, and when not to substitute another skill.
GO2_AUTONOMY_POSTURES = (
    PostureSpec(
        "stand_up",
        "stand_up",
        (
            "Stand the Go2 up on all fours (站起来/起身/立正). "
            "Use before any move_forward/turn skill if the dog may be lying "
            "or sitting. Not a dance or greeting."
        ),
    ),
    PostureSpec(
        "stand_down",
        "stand_down",
        (
            "Lie the Go2 down (趴下/躺下/休息). "
            "Use when the user asks to lie down or rest. Not sit."
        ),
    ),
    PostureSpec(
        "balance_stand",
        "balance_stand",
        (
            "Enter Go2 balance standing mode (平衡站立/站好). "
            "Use for balanced idle stand only; not a greeting or dance."
        ),
    ),
    PostureSpec(
        "sit",
        "sit",
        (
            "Make the Go2 sit (坐下/坐好). "
            "Use for sitting posture only; not lie down or stand up."
        ),
    ),
    PostureSpec(
        "rise_sit",
        "rise_sit",
        (
            "Rise from the Go2 sitting posture (从坐姿起身). "
            "Use only when currently sitting and the user asks to get up."
        ),
    ),
    PostureSpec(
        "hello",
        "hello",
        (
            "Play the Go2 native greeting (打招呼/你好/问好). "
            "Use only for greeting. Never substitute for 比心/heart, "
            "跳舞/dance, 坐下/sit, or locomotion."
        ),
    ),
    PostureSpec(
        "stretch",
        "stretch",
        (
            "Play Go2's native stretch (伸懒腰/拉伸/活动一下). "
            "Use for stretch only; not dance or greeting."
        ),
    ),
    PostureSpec(
        "content",
        "content",
        (
            "Play Go2's native content action (开心/满足). "
            "Use when the user asks the dog to look pleased; not heart or dance."
        ),
    ),
    PostureSpec(
        "heart",
        "heart",
        (
            "Play Go2's native heart gesture (比心/送心/爱心). "
            "Use when the user asks for a heart. Not hello/wave and not dance."
        ),
    ),
    PostureSpec(
        "scrape",
        "scrape",
        (
            "Play Go2's native scrape action (刮地示意). "
            "Use only when the user names scrape/作揖-like demo; not sit or stand_down."
        ),
    ),
    PostureSpec(
        "dance1",
        "dance1",
        (
            "Play Go2's first native dance (跳舞/舞蹈一/跳一支舞). "
            "Prefer dance1 unless the user asks for the second dance."
        ),
    ),
    PostureSpec(
        "dance2",
        "dance2",
        (
            "Play Go2's second native dance (舞蹈二/再来一段舞). "
            "Use when the user wants a different dance from dance1."
        ),
    ),
)

GO2_OPERATOR_POSTURES = (
    PostureSpec(
        "damp",
        "damp",
        (
            "OPERATOR-ONLY dangerous: switch Go2 to damping mode (阻尼). "
            "Never call from casual chat; operator control only."
        ),
        operator_only=True,
        dangerous=True,
    ),
    PostureSpec(
        "recovery_stand",
        "recovery_stand",
        (
            "OPERATOR-ONLY: Go2 recovery stand (恢复站立). "
            "Use only for explicit recovery by an operator, not normal 站起来."
        ),
        operator_only=True,
    ),
    *(
        PostureSpec(name, name, description, operator_only=True, dangerous=True)
        for name, description in (
            (
                "front_flip",
                "OPERATOR-ONLY dangerous: Go2 front flip (前空翻). Not for chat agent.",
            ),
            (
                "front_jump",
                "OPERATOR-ONLY dangerous: Go2 front jump (前跳). Not for chat agent.",
            ),
            (
                "front_pounce",
                "OPERATOR-ONLY dangerous: Go2 front pounce (前扑). Not for chat agent.",
            ),
            (
                "left_flip",
                "OPERATOR-ONLY dangerous: Go2 left flip (左侧翻). Not for chat agent.",
            ),
            (
                "back_flip",
                "OPERATOR-ONLY dangerous: Go2 back flip (后空翻). Not for chat agent.",
            ),
        )
    ),
    *(
        PostureSpec(name, name, description, operator_only=True)
        for name, description in (
            (
                "free_walk",
                "OPERATOR-ONLY: select Go2 free walk gait (自由行走). Restricted area only.",
            ),
            (
                "static_walk",
                "OPERATOR-ONLY: select Go2 static walk gait (静态行走). Restricted area only.",
            ),
            (
                "trot_run",
                "OPERATOR-ONLY: select Go2 trot/run gait (小跑/跑步). Restricted area only.",
            ),
            (
                "economic_gait",
                "OPERATOR-ONLY: select Go2 economic gait (省电步态). Restricted area only.",
            ),
            (
                "switch_avoid_mode",
                "OPERATOR-ONLY: Go2 obstacle-avoidance mode switch (避障模式). Not free navigation.",
            ),
        )
    ),
)

GO2_FLAG_SPECS = (
    PostureSpec(
        "pose",
        "pose",
        (
            "Enable/disable Go2 pose mode with boolean flag (姿态模式). "
            "Pass flag=true to enable, flag=false to disable. "
            "Not a greeting, dance, or sit."
        ),
    ),
    *(
        PostureSpec(name, name, description, operator_only=True, dangerous=dangerous)
        for name, description, dangerous in (
            (
                "hand_stand",
                "OPERATOR-ONLY dangerous: Go2 hand stand 倒立 (flag true/false).",
                True,
            ),
            (
                "free_bound",
                "OPERATOR-ONLY dangerous: Go2 free bound mode (flag true/false).",
                True,
            ),
            (
                "free_jump",
                "OPERATOR-ONLY dangerous: Go2 free jump mode (flag true/false).",
                True,
            ),
            (
                "free_avoid",
                "OPERATOR-ONLY: Go2 free avoidance mode (flag true/false).",
                False,
            ),
            (
                "classic_walk",
                "OPERATOR-ONLY: Go2 classic walk mode (flag true/false).",
                False,
            ),
            (
                "walk_upright",
                "OPERATOR-ONLY dangerous: Go2 upright walking 直立行走 (flag true/false).",
                True,
            ),
            (
                "cross_step",
                "OPERATOR-ONLY: Go2 cross step mode (flag true/false).",
                False,
            ),
            (
                "switch_joystick",
                (
                    "OPERATOR-ONLY dangerous: switch Go2 joystick ownership "
                    "(flag true/false). Can break app/SDK control handoff."
                ),
                True,
            ),
            (
                "auto_recover_set",
                (
                    "OPERATOR-ONLY: enable/disable Go2 auto recovery "
                    "(flag true/false). Not recovery_stand."
                ),
                False,
            ),
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
                "Gesture-mode alias for Go2 native hello when a person waves. "
                "Not a humanoid arm wave. Text agent should use hello or heart "
                "directly when the user names them."
            ),
        )
    )


class RandomGo2DanceSkill(RobotSkill[SkillArgs]):
    """Pick one safe native dance from a fixed pool after Runtime checks."""

    args_model = SkillArgs

    def __init__(
        self,
        chooser: Callable[[Sequence[str]], str] | None = None,
    ) -> None:
        self._chooser = chooser or random.choice
        self.metadata = SkillMetadata(
            name="random_dance",
            description=(
                "用户竖起大拇指、点赞或明确要求随机跳舞时使用；从 dance1/dance2 "
                "中随机选择一个。不是跳跃、空翻或危险特技。 "
                "Use when the user gives a thumbs-up / like or explicitly asks "
                "for a random dance; randomly choose dance1 or dance2. "
                "Not a jump, flip, or dangerous stunt."
            ),
            tags=("motion", "dance", "go2"),
            required_resources=("mobile_base",),
            timeout_s=20.0,
            interruptible=True,
        )

    async def check_preconditions(
        self,
        ctx: SkillContext,
        args: SkillArgs,
    ) -> tuple[bool, str]:
        state = await ctx.robot.get_state()
        if state.hardware and not state.connected:
            return False, "robot is not connected"
        return True, ""

    async def execute(self, ctx: SkillContext, args: SkillArgs) -> SkillResult:
        chosen = self._chooser(GO2_RANDOM_DANCE_POOL)
        if chosen not in GO2_RANDOM_DANCE_POOL:
            raise ValueError(
                f"random dance chooser returned unsafe action: {chosen!r}"
            )
        await ctx.robot.execute_loco_action(chosen)
        return SkillResult.ok(
            f"{chosen} command accepted from random_dance pool",
            chosen_action=chosen,
            pool=list(GO2_RANDOM_DANCE_POOL),
            completion_verified=False,
        )


def build_go2_autonomy_skills() -> tuple[RobotSkill[SkillArgs], ...]:
    skills = (
        *(PostureSkill(spec) for spec in GO2_AUTONOMY_POSTURES),
        *(Go2FlagSkill(spec) for spec in GO2_FLAG_SPECS if not spec.operator_only),
        _go2_wave_hello_skill(),
        RandomGo2DanceSkill(),
        Go2MoveForwardSkill(),
        Go2MoveBackwardSkill(),
        Go2MoveLeftSkill(),
        Go2MoveRightSkill(),
        Go2TurnLeftSkill(),
        Go2TurnRightSkill(),
        StopSkill(),
        StopMoveSkill(),
        Go2MoveSkill(),
        FollowPersonSkill(),
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
    "GO2_RANDOM_DANCE_POOL",
    "RandomGo2DanceSkill",
    "build_go2_all_skills",
    "build_go2_autonomy_skills",
    "build_go2_operator_skills",
    "register_go2_skills",
]
