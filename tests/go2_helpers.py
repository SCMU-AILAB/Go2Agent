"""Construct individual Go2 skills from the production catalog for tests."""

from core.models import SkillArgs
from core.skill import RobotSkill
from skills import build_go2_all_skills


def go2_skill(name: str) -> RobotSkill[SkillArgs]:
    return next(
        skill for skill in build_go2_all_skills() if skill.metadata.name == name
    )
