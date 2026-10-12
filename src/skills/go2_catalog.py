"""Complete Go2 SportClient skill catalog."""

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
)
from .motions.go2_controls import StopSkill
from .motions.go2_follow import FollowPersonSkill
from .posture import PostureSkill, PostureSpec

# Safe random-dance pool only. Never operator-only or dangerous actions.
GO2_RANDOM_DANCE_POOL: tuple[str, ...] = ("dance1", "dance2")

# Skill names map 1:1 to SportClient methods documented in docs/go2-adapter.md.
# Descriptions are the model-facing contract shared by the text and vision
# Agents. Each states what the action physically does, its Chinese names, and
# how it differs from similar skills. They describe the action, not which
# instruction or visual cue should trigger it; that is the task's job.
_OPERATOR_RULE = (
    "Only when the operator's instruction explicitly names this exact action."
)

GO2_AUTONOMY_POSTURES = (
    PostureSpec(
        "stand_up",
        "stand_up",
        (
            "Stand up on all four legs from lying down (站起来/起身/起立). "
            "Needed before walking if the dog is lying down. To get up from "
            "sitting use rise_sit. Not a greeting or a dance."
        ),
    ),
    PostureSpec(
        "stand_down",
        "stand_down",
        (
            "Lie down flat on the floor (趴下/躺下/卧倒/休息). A posture change, "
            "not sit. Not a way to stay still: to remain in place, take no action."
        ),
    ),
    PostureSpec(
        "balance_stand",
        "balance_stand",
        (
            "Switch to balanced standing, the normal ready stance (平衡站立/站稳). "
            "Not needed merely to stay still; not a greeting or a dance."
        ),
    ),
    PostureSpec(
        "sit",
        "sit",
        (
            "Sit down on the hind legs (坐下/坐好). Not lie down (stand_down). "
            "Get up again with rise_sit."
        ),
    ),
    PostureSpec(
        "rise_sit",
        "rise_sit",
        (
            "Get up from the sitting posture back to standing (从坐姿起身). "
            "Only meaningful while sitting; from lying down use stand_up."
        ),
    ),
    PostureSpec(
        "hello",
        "hello",
        (
            "Native greeting: lifts a front paw and waves (打招呼/问好/挥手/你好). "
            "The only greeting action, also used to return a wave. Not heart, "
            "not a dance, not a posture change."
        ),
    ),
    PostureSpec(
        "stretch",
        "stretch",
        (
            "Native stretch, like waking up (伸懒腰/拉伸/活动筋骨). "
            "Not a dance or a greeting."
        ),
    ),
    PostureSpec(
        "content",
        "content",
        (
            "Native happy/content expression (开心/高兴/满足). Shows the dog is "
            "pleased; not heart, not a greeting, not a dance."
        ),
    ),
    PostureSpec(
        "heart",
        "heart",
        (
            "Native heart gesture made with the front paws (比心/送心/爱心). "
            "Not a greeting (hello) and not a dance."
        ),
    ),
    PostureSpec(
        "scrape",
        "scrape",
        (
            "Native 作揖 / 拜年 gesture with the front paws (作揖/拜年/恭喜). "
            "Not sit, not lie down, not a greeting wave."
        ),
    ),
    PostureSpec(
        "dance1",
        "dance1",
        (
            "Native dance routine number one (舞蹈一/第一支舞). Use when this "
            "specific dance is named; for an unspecified dance use random_dance."
        ),
    ),
    PostureSpec(
        "dance2",
        "dance2",
        (
            "Native dance routine number two (舞蹈二/第二支舞). Use when this "
            "specific dance is named; for an unspecified dance use random_dance."
        ),
    ),
)

GO2_OPERATOR_POSTURES = (
    PostureSpec(
        "damp",
        "damp",
        (
            "OPERATOR-ONLY, dangerous: damping mode, all joints go limp and the "
            f"dog collapses (阻尼/软化). {_OPERATOR_RULE}"
        ),
        operator_only=True,
        dangerous=True,
    ),
    PostureSpec(
        "recovery_stand",
        "recovery_stand",
        (
            "OPERATOR-ONLY: recover to standing after a fall or roll-over "
            f"(恢复站立/翻身起来). Not the normal stand_up. {_OPERATOR_RULE}"
        ),
        operator_only=True,
    ),
    *(
        PostureSpec(
            name,
            name,
            f"OPERATOR-ONLY, dangerous: {text}. {_OPERATOR_RULE}",
            operator_only=True,
            dangerous=True,
        )
        for name, text in (
            ("front_flip", "front flip (前空翻)"),
            ("front_jump", "jump forward (前跳/向前跳)"),
            ("front_pounce", "pounce forward (前扑)"),
            ("left_flip", "sideways flip to the left (左侧翻)"),
            ("back_flip", "back flip (后空翻)"),
        )
    ),
    *(
        PostureSpec(
            name,
            name,
            f"OPERATOR-ONLY: {text}. {_OPERATOR_RULE}",
            operator_only=True,
        )
        for name, text in (
            ("free_walk", "select the free walk gait (自由行走步态)"),
            ("static_walk", "select the static walk gait (静态行走步态)"),
            ("trot_run", "select the trot/run gait (小跑/跑步步态)"),
            ("economic_gait", "select the power-saving gait (省电步态)"),
            (
                "switch_avoid_mode",
                (
                    "toggle the built-in obstacle-avoidance mode (避障模式); "
                    "not navigation"
                ),
            ),
        )
    ),
)

GO2_FLAG_SPECS = (
    PostureSpec(
        "pose",
        "pose",
        (
            "Pose mode (姿态模式/摆姿势): flag=true enters it, the body tilts and "
            "turns in place without walking; flag=false exits. "
            "Not a greeting, dance or sit."
        ),
    ),
    *(
        PostureSpec(
            name,
            name,
            (
                f"OPERATOR-ONLY{', dangerous' if dangerous else ''}: {text}; "
                f"flag=true enables, flag=false disables. {_OPERATOR_RULE}"
            ),
            operator_only=True,
            dangerous=dangerous,
        )
        for name, text, dangerous in (
            ("hand_stand", "handstand on the front legs (倒立)", True),
            ("free_bound", "free bound gait (自由跳跃步态)", True),
            ("free_jump", "free jump mode (自由跳)", True),
            ("free_avoid", "free obstacle-avoidance mode (自由避障)", False),
            ("classic_walk", "classic walk mode (经典步态)", False),
            ("walk_upright", "walk upright on the hind legs (直立行走)", True),
            ("cross_step", "cross-step gait (交叉步)", False),
            (
                "switch_joystick",
                (
                    "hand control to or from the remote joystick (遥控器控制权); "
                    "can break app/SDK control"
                ),
                True,
            ),
            (
                "auto_recover_set",
                (
                    "automatic recovery after a fall (自动翻身恢复); "
                    "not recovery_stand"
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
                "Play one native dance chosen at random from dance1/dance2 "
                "(跳舞/跳支舞/随机跳舞). Use for a dance request that does not "
                "name a specific routine. Not a jump, flip or stunt."
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
        RandomGo2DanceSkill(),
        Go2MoveForwardSkill(),
        Go2MoveBackwardSkill(),
        Go2MoveLeftSkill(),
        Go2MoveRightSkill(),
        Go2TurnLeftSkill(),
        Go2TurnRightSkill(),
        StopSkill(),
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
    include_operator_only: bool = True,
) -> None:
    """Register every Go2 skill for both text and vision agents.

    ``include_operator_only`` remains as a compatibility keyword for old
    launchers, but the Go2-only runtime intentionally exposes the full catalog.
    """
    del include_operator_only
    for skill in build_go2_all_skills():
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
