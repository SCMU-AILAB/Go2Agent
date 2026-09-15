"""Go2 skill catalog: only SportClient capabilities with matching semantics.

Do not reuse register_g1_skills() for Go2. Arm presets, FSM controls, and G1-only
postures are intentionally omitted. damp and recovery_stand stay operator-only.
"""

from __future__ import annotations

from typing import cast

from core.models import SkillArgs
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
    skills = (PostureSkill(spec) for spec in GO2_OPERATOR_POSTURES)
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
