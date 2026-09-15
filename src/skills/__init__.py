"""Robot skills grouped by capability."""

from .catalog import (
    build_g1_all_skills,
    build_g1_autonomy_skills,
    build_g1_operator_skills,
    register_g1_skills,
)
from .go2_catalog import (
    build_go2_all_skills,
    build_go2_autonomy_skills,
    build_go2_operator_skills,
    register_go2_skills,
)

__all__ = [
    "build_g1_all_skills",
    "build_g1_autonomy_skills",
    "build_g1_operator_skills",
    "build_go2_all_skills",
    "build_go2_autonomy_skills",
    "build_go2_operator_skills",
    "register_g1_skills",
    "register_go2_skills",
]
