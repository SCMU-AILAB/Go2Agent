"""Compile natural-language visual goals into validated task contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.models import SkillArgs
from core.skill import RobotSkill

from .decision import DecisionAgentError
from .vision_policy import VisionModelInvoker, _skill_catalog_payload


class VisualTaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=1000)
    target: str = Field(default="task-defined visible target", max_length=200)
    trigger_kind: Literal["presence", "gesture", "condition"] = "condition"
    trigger: str = Field(
        default="when current evidence satisfies the task", max_length=300
    )
    completion: str = Field(default="until cancelled", max_length=300)
    stop_conditions: list[str] = Field(
        default_factory=lambda: ["operator cancellation", "invalid motion perception"]
    )
    required_skills: list[str] = Field(default_factory=list)
    skill_arguments: dict[str, dict[str, object]] = Field(default_factory=dict)
    repeat_policy: Literal["once_per_event", "while_needed", "persistent"] = (
        "once_per_event"
    )
    supported: bool = True
    capability_gap: str | None = Field(default=None, max_length=500)
    clarification: str | None = Field(default=None, max_length=500)


class VisualTaskPlanner:
    def __init__(self, invoker: VisionModelInvoker) -> None:
        self.invoker = invoker

    async def close(self) -> None:
        close = getattr(self.invoker, "close", None)
        if callable(close):
            await close()

    async def plan(
        self, instruction: str, catalog: Sequence[RobotSkill[SkillArgs]]
    ) -> VisualTaskSpec:
        allowed = {
            s.metadata.name: s
            for s in catalog
            if not {"operator_only", "dangerous"}.intersection(s.metadata.tags)
        }
        prompt = (
            "Compile the operator's visual task, not a current camera decision. Return ONLY JSON matching "
            "the supplied task schema. Preserve the operator's meaning; never replace an unsupported goal "
            "with a greeting. Use only autonomous skills in the catalog. Fill required_skills and "
            "skill_arguments with actual validated parameter values, not schemas. Choose repeat_policy=persistent for continuous following, once_per_event for greetings and gesture responses, or while_needed for repeated bounded steps toward a goal. Conditions must be "
            "observable. Use the skill's defaults when the operator omits a parameter. For following, "
            "the capability is single visible person distance following, not identity recognition, "
            "route replay or navigation behind a person around corners. Multiple ambiguous targets "
            "need clarification. If the goal requires absent capabilities, set supported=false and "
            "explain capability_gap. Use clarification only for essential missing choices. "
            "An immediate greeting on seeing a person differs from responding to waving. "
            "Do not add hand-gesture prerequisites to unrelated goals. Reply with short Chinese "
            "goal, target, trigger, completion, stop_conditions and gap descriptions.\n"
            f"Task schema: {json.dumps(VisualTaskSpec.model_json_schema(), ensure_ascii=False)}\n"
            f"Autonomous skills: {json.dumps(_skill_catalog_payload(tuple(allowed.values())), ensure_ascii=False)}\n"
            f"Operator instruction: {json.dumps(instruction, ensure_ascii=False)}"
        )
        output = await self.invoker.ainvoke((), prompt)
        try:
            spec = (
                VisualTaskSpec.model_validate_json(output)
                if isinstance(output, str)
                else VisualTaskSpec.model_validate(output)
            )
        except ValueError as exc:
            raise DecisionAgentError(
                f"invalid visual task specification: {exc}"
            ) from exc
        for name in set(spec.required_skills) | set(spec.skill_arguments):
            if name not in allowed:
                return spec.model_copy(
                    update={
                        "supported": False,
                        "capability_gap": f"未提供可自主执行的技能：{name}",
                    }
                )
            try:
                allowed[name].args_model.model_validate(
                    spec.skill_arguments.get(name, {})
                )
            except ValueError as exc:
                return spec.model_copy(
                    update={
                        "supported": False,
                        "capability_gap": f"技能参数不可用：{name}: {exc}",
                    }
                )
        if not spec.supported and not spec.capability_gap:
            spec.capability_gap = "任务超出当前感知或技能能力"
        return spec
