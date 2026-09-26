"""Structured LangChain decision Agent for sparse world events."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, Self, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from core.models import SkillArgs
from core.skill import RobotSkill
from perception.events import WorldEvent, WorldEventType

_DECISION_SYSTEM_PROMPT = """Select one Unitree Go2 response for the supplied event.
Output only a JSON object. Its top-level keys are action, skill, arguments,
speech, and optionally reason. Never wrap the object in a decision key.
The action field must be exactly one of execute_skill, speak,
execute_and_speak, or ignore. Put the robot skill name in skill, never in action.
If both skill and speech are present, use execute_and_speak.

For person_entered, greet only when the supplied world state says the person has
not been greeted. For person_too_close, select a bounded safety response when
one is available. For person_left or no behavior, ignore the event.

Use only skills and argument shapes from the registry catalog below. Keep
Chinese speech brief.
"""


def build_decision_system_prompt(
    skill_catalog: Sequence[RobotSkill[SkillArgs]],
) -> str:
    if not skill_catalog:
        catalog = "- (no skills registered)"
    else:
        entries: list[str] = []
        for skill in skill_catalog:
            schema = skill.args_model.model_json_schema()
            entries.append(
                "- "
                f"{skill.metadata.name}: {skill.metadata.description}; "
                f"arguments schema: {json.dumps(schema, ensure_ascii=True, default=str)}"
            )
        catalog = "\n".join(entries)
    return f"{_DECISION_SYSTEM_PROMPT}\nRegistered skill catalog:\n{catalog}\n"


# Kept as a stable import for callers that do not need a dynamic catalog.
DECISION_SYSTEM_PROMPT = build_decision_system_prompt(())

DecisionAction = Literal[
    "execute_skill",
    "speak",
    "execute_and_speak",
    "continue",
    "interrupt",
    "ignore",
]


def _has_nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    skill: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    speech: str | None = None
    reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_skill_action_alias(cls, value: object) -> object:
        """Accept the compact ``{action: skill, ...}`` shape some local models emit."""
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        action = payload.get("action")
        if not isinstance(action, str):
            return payload
        action = action.strip()
        payload["action"] = action
        skill_value = payload.get("skill")
        # Some multimodal models put the selected tool in arguments.name.
        # Normalize that shape at the protocol boundary.
        raw_arguments = payload.get("arguments")
        if (
            action in {"execute_skill", "execute_and_speak"}
            and not _has_nonempty_text(skill_value)
            and isinstance(raw_arguments, Mapping)
            and _has_nonempty_text(raw_arguments.get("name"))
        ):
            payload["skill"] = str(raw_arguments["name"]).strip()
            payload["arguments"] = {
                key: item for key, item in raw_arguments.items() if key != "name"
            }
        has_speech = _has_nonempty_text(payload.get("speech"))

        if action == "execute_skill":
            if has_speech:
                payload["action"] = "execute_and_speak"
            return payload
        if action == "speak":
            payload["skill"] = None
            payload["arguments"] = {}
            return payload
        if action == "execute_and_speak":
            if not has_speech:
                payload["action"] = "execute_skill"
            return payload
        if action in {"continue", "interrupt", "ignore"}:
            # An explicit no-op or interrupt must never turn into a physical
            # action because the model attached an inconsistent skill field.
            payload["skill"] = None
            payload["arguments"] = {}
            payload["speech"] = None
            return payload

        if action in {"stop", "cancel"}:
            payload["action"] = "interrupt"
            payload["skill"] = None
            payload["arguments"] = {}
            payload.pop("speech", None)
            return payload
        if action in {"keep", "keep_going"}:
            payload["action"] = "continue"
            payload["skill"] = None
            payload["arguments"] = {}
            payload.pop("speech", None)
            return payload
        if action in {"none", "no_action", "noop"}:
            payload["action"] = "speak" if has_speech else "ignore"
            payload["skill"] = None
            payload["arguments"] = {}
            return payload

        skill = payload.get("skill")
        if not isinstance(skill, str) or not skill.strip():
            payload["skill"] = action
        payload["action"] = (
            "execute_and_speak"
            if has_speech
            else "execute_skill"
        )
        return payload

    @field_validator("skill", "speech", "reason", mode="before")
    @classmethod
    def normalize_empty_optional_string(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_action_contract(self) -> Self:
        executes = self.action in {"execute_skill", "execute_and_speak"}
        speaks = self.action in {"speak", "execute_and_speak"}
        if executes and self.skill is None:
            raise ValueError(f"{self.action} requires a skill")
        if not executes and (self.skill is not None or self.arguments):
            raise ValueError(f"{self.action} must not include a skill or arguments")
        if speaks and (self.speech is None or not self.speech.strip()):
            raise ValueError(f"{self.action} requires non-empty speech")
        if not speaks and self.speech is not None:
            raise ValueError(f"{self.action} must not include speech")
        return self

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class DecisionAgentError(RuntimeError):
    """Raised when the decision Agent does not return a valid decision."""


class DecisionAgent(Protocol):
    async def decide(
        self,
        event: WorldEvent,
        world_state: Mapping[str, object],
    ) -> AgentDecision: ...


class DecisionInvoker(Protocol):
    async def ainvoke(self, input_state: dict[str, object]) -> object: ...


class _StructuredDecisionModel(Protocol):
    async def ainvoke(self, messages: list[BaseMessage]) -> object: ...


class _StructuredDecisionInvoker:
    """Adapt one structured model call to the testable invoker contract."""

    def __init__(self, model: _StructuredDecisionModel, system_prompt: str) -> None:
        self._model = model
        self._system_prompt = system_prompt

    async def ainvoke(self, input_state: dict[str, object]) -> object:
        raw_messages = input_state.get("messages")
        if not isinstance(raw_messages, list) or not all(
            isinstance(message, BaseMessage) for message in raw_messages
        ):
            raise TypeError("Decision Agent input must contain messages")
        result = await self._model.ainvoke(
            [SystemMessage(content=self._system_prompt), *raw_messages]
        )
        return {"structured_response": result}


class EventDecisionAgent:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        invoker: DecisionInvoker | None = None,
        timeout_s: float = 8.0,
        skill_catalog: Sequence[RobotSkill[SkillArgs]] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("decision timeout must be greater than zero")
        self.timeout_s = timeout_s
        self._skill_names = (
            frozenset(skill.metadata.name for skill in skill_catalog)
            if skill_catalog is not None
            else None
        )
        if invoker is not None:
            self._invoker = invoker
            return

        system_prompt = build_decision_system_prompt(skill_catalog or ())
        model = ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            base_url=base_url or os.getenv("OLLAMA_HOST"),
            temperature=0.1,
            reasoning=False,
            num_ctx=8192,
            num_predict=64,
            keep_alive="30m",
            client_kwargs={"trust_env": False, "timeout": timeout_s},
        )
        structured_model = model.with_structured_output(
            AgentDecision,
            method="json_mode",
        )
        self._invoker = _StructuredDecisionInvoker(
            cast(_StructuredDecisionModel, cast(object, structured_model)),
            system_prompt,
        )

    async def decide(
        self,
        event: WorldEvent,
        world_state: Mapping[str, object],
    ) -> AgentDecision:
        if event.type == WorldEventType.PERSON_LEFT:
            return AgentDecision(action="ignore")

        payload = {
            "instruction": (
                f"Select only the response mapped to {event.type.value!r}; "
                "do not select an example for another event type."
            ),
            "event": event.to_dict(),
            "world_state": dict(world_state),
        }
        input_state: dict[str, object] = {
            "messages": [
                HumanMessage(
                    content=json.dumps(payload, ensure_ascii=False, default=str)
                )
            ]
        }
        try:
            async with asyncio.timeout(self.timeout_s):
                output = await self._invoker.ainvoke(input_state)
        except TimeoutError as exc:
            raise DecisionAgentError(
                f"Decision Agent timed out after {self.timeout_s:g} seconds"
            ) from exc
        except Exception as exc:
            raise DecisionAgentError(f"Decision Agent invocation failed: {exc}") from exc
        if not isinstance(output, Mapping):
            raise DecisionAgentError("Decision Agent returned an invalid state")
        try:
            decision = AgentDecision.model_validate(output.get("structured_response"))
        except ValidationError as exc:
            raise DecisionAgentError(
                f"Decision Agent returned an invalid decision: {exc}"
            ) from exc
        if (
            self._skill_names is not None
            and decision.skill is not None
            and decision.skill not in self._skill_names
        ):
            raise DecisionAgentError(
                f"Decision Agent selected unregistered skill: {decision.skill}"
            )
        return decision
