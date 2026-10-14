"""Brain output format: a pydantic model for parsing and a JSON schema for constrained decoding."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field

from go2agent.behaviors.primitives import BEHAVIORS, TARGET_BEHAVIORS
from go2agent.core.types import ActionSpec


class BoxTarget(BaseModel):
    """A target the tracker has not marked: the brain points at it with a box on the newest image."""

    box: tuple[float, float, float, float] = Field(description="normalized x1,y1,x2,y2 on the newest image")
    label: str


class PlanStep(BaseModel):
    behavior: str
    args: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    thought: str = ""
    say: str = ""
    new_task: str = ""  # non-empty only when the user's speech changed the task: the new full task
    mode: Literal["keep", "replace", "append"] = "keep"
    plan: list[PlanStep] = Field(default_factory=list)


_BOX_TARGET = {
    "type": "object",
    "properties": {
        "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        "label": {"type": "string"},
    },
    "required": ["box", "label"],
    "additionalProperties": False,
}


def _step_schema(name: str, actions: tuple[ActionSpec, ...]) -> dict[str, Any]:
    args = copy.deepcopy(BEHAVIORS[name].args_model.model_json_schema())
    args.pop("title", None)
    props = args.get("properties", {})
    for p in props.values():
        p.pop("title", None)
    if name in TARGET_BEHAVIORS:
        props["target"] = {"anyOf": [{"type": "integer"}, _BOX_TARGET]}
    if name == "perform":
        props["action"] = {"type": "string", "enum": [a.name for a in actions]}
    args["additionalProperties"] = False
    return {
        "type": "object",
        "properties": {"behavior": {"const": name}, "args": args},
        "required": ["behavior", "args"],
        "additionalProperties": False,
    }


def decision_json_schema(actions: tuple[ActionSpec, ...]) -> dict[str, Any]:
    """JSON schema for constrained decoding. `actions` are the actions the brain may use."""
    names = [n for n in BEHAVIORS if n != "perform" or actions]
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "maxLength": 60},
            "say": {"type": "string", "maxLength": 120},
            "new_task": {"type": "string", "maxLength": 200},
            "mode": {"type": "string", "enum": ["keep", "replace", "append"]},
            "plan": {
                "type": "array",
                "maxItems": 4,
                "items": {"anyOf": [_step_schema(n, actions) for n in names]},
            },
        },
        "required": ["thought", "say", "new_task", "mode", "plan"],
        "additionalProperties": False,
    }
