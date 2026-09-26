"""Expose registered robot skills as LangChain structured tools."""

from __future__ import annotations

import json
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from core.models import SkillArgs
from core.runtime import SkillRuntime
from core.skill import RobotSkill


class SkillToolObserver(Protocol):
    """Observe Agent tool calls without bypassing the SkillRuntime boundary."""

    async def before_skill(
        self,
        skill_name: str,
        arguments: dict[str, object],
    ) -> None: ...

    async def after_skill(
        self,
        skill_name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> None: ...


class _ValidatedStructuredTool(StructuredTool):
    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        # LangChain skips _parse_input entirely for zero-field Pydantic models.
        if isinstance(tool_input, dict) and self.args_schema is not None:
            schema = self.args_schema
            if isinstance(schema, type) and issubclass(schema, SkillArgs):
                schema.model_validate(tool_input)
        return super()._to_args_and_kwargs(tool_input, tool_call_id)


def build_langchain_tools(
    runtime: SkillRuntime,
    *,
    observer: SkillToolObserver | None = None,
) -> tuple[BaseTool, ...]:
    return tuple(
        _build_skill_tool(runtime, skill, observer=observer)
        for skill in runtime.registry.list()
    )


def _build_skill_tool(
    runtime: SkillRuntime,
    skill: RobotSkill[SkillArgs],
    *,
    observer: SkillToolObserver | None = None,
) -> BaseTool:
    skill_name = skill.metadata.name

    async def invoke_skill(**arguments: object) -> str:
        if observer is not None:
            await observer.before_skill(skill_name, dict(arguments))
        result = await runtime.execute(skill_name, **arguments)
        payload = result.to_dict()
        if observer is not None:
            await observer.after_skill(skill_name, dict(arguments), payload)
        return json.dumps(payload, ensure_ascii=False, default=str)

    return _ValidatedStructuredTool.from_function(
        coroutine=invoke_skill,
        name=skill_name,
        description=skill.metadata.description,
        args_schema=skill.args_model,
        infer_schema=False,
    )
