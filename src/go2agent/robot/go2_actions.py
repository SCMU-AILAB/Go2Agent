"""Go2 native action catalog (SportClient). Pure data shared by the Go2 and Sim robots.

Descriptions are model-facing: what the action physically does, its Chinese names, and how it differs
from similar actions. Durations are rough estimates of when the dog is ready again.
"""

from __future__ import annotations

from go2agent.core.types import ActionSpec

_D = frozenset({"dangerous"})
_POSTURE = frozenset({"posture"})

GO2_ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "stand_up",
        "Stand up on four legs from lying down (站起来/起立). From sitting use rise_sit.",
        2.0,
        _POSTURE,
    ),
    ActionSpec("stand_down", "Lie down flat on the floor (趴下/卧倒/休息). Not sit.", 2.0, _POSTURE),
    ActionSpec("balance_stand", "Return to the normal balanced standing stance (站稳).", 1.0, _POSTURE),
    ActionSpec("sit", "Sit on the hind legs (坐下). Get up again with rise_sit.", 2.0, _POSTURE),
    ActionSpec("rise_sit", "Get up from sitting back to standing (从坐姿起身).", 2.0, _POSTURE),
    ActionSpec(
        "hello", "Lift a front paw and wave (打招呼/挥手/你好). The greeting action.", 5.0
    ),  # measured 4.7 s
    ActionSpec("stretch", "Stretch like waking up (伸懒腰).", 4.0),
    ActionSpec("content", "Happy/content expression (开心/高兴).", 3.0),
    ActionSpec("heart", "Make a heart gesture with the front paws (比心).", 3.0),
    ActionSpec("scrape", "Chinese bow / 作揖 gesture with the front paws (作揖/拜年).", 3.0),
    ActionSpec("dance1", "Dance routine one (跳舞).", 10.0),
    ActionSpec("dance2", "Dance routine two (跳舞).", 10.0),
    ActionSpec("front_jump", "Jump forward (向前跳). Needs clear space.", 2.0, _D),
    ActionSpec("front_pounce", "Pounce forward (前扑). Needs clear space.", 2.0, _D),
    ActionSpec("front_flip", "Front flip (前空翻). Needs clear space.", 3.0, _D),
    ActionSpec("back_flip", "Back flip (后空翻). Needs clear space.", 3.0, _D),
    ActionSpec("left_flip", "Side flip to the left (侧空翻).", 3.0, _D),
)
