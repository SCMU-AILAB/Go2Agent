"""Go2 robot skills grouped by capability."""

from .go2_catalog import (
    build_go2_all_skills,
    build_go2_autonomy_skills,
    build_go2_operator_skills,
    register_go2_skills,
)

__all__ = [
    "build_go2_all_skills",
    "build_go2_autonomy_skills",
    "build_go2_operator_skills",
    "register_go2_skills",
]
