"""Conversational Agent that can invoke registered robot skills as tools."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

from adapters.langchain import SkillToolObserver, build_langchain_tools
from core.runtime import SkillRuntime

SYSTEM_PROMPT = """You are the conversational controller for a Unitree G1 robot.

Reply in the user's language and keep spoken responses concise.
Use a robot skill tool only when the user explicitly asks the robot to perform
that capability. Never claim that a physical action succeeded before the tool
returns success. If a tool fails or rejects its arguments, explain the failure
briefly. Do not invent robot capabilities or emit action JSON.
"""

GO2_SYSTEM_PROMPT = """You are the conversational controller for a Unitree Go2 quadruped.

Reply in the user's language and keep spoken responses concise.
Use the live registered tool descriptions and argument schemas to choose actions
for the user's goal. You may call more than one tool in sequence when needed.
After each result, decide whether another step is needed. Do not invent tools,
and never claim physical completion merely because an SDK command was accepted.
For a persistent goal like following a person, choose the registered feedback
skill rather than repeating short open-loop movement commands.

Hard rules:
- Choose only skills that actually match the requested behavior. If none exists,
  explain the limitation instead of substituting a greeting or another action.
- This text Agent receives no camera images, even when preview is on. A camera-
  dependent skill must pass its own local perception and safety preconditions;
  do not promise tracking when the camera is unavailable.
- Short move/turn skills are bounded open-loop steps, not navigation.
- Operator-only tools (flips, gaits, damp, recovery_stand, switch_joystick,
  auto_recover_set, dangerous flags) are present only when enabled at startup;
  do not call them for casual chat.
- Respect each tool's parameter schema. If a tool fails or is blocked, explain
  why and do not repeatedly retry unsafe motion.
Do not invent robot capabilities or emit action JSON.
"""


def system_prompt_for(robot_model: str) -> str:
    if robot_model == "go2":
        return GO2_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_runtime_system_prompt(
    runtime: SkillRuntime,
    system_prompt: str,
) -> str:
    """Append the live SkillRegistry contract to the model's system prompt.

    LangChain also exposes the skills as structured tools, but a compact catalog
    in the prompt makes the available capabilities and argument schemas visible
    to models that are weak at tool discovery. The registry remains the only
    source of truth; this is generated at Agent construction time.
    """
    entries: list[dict[str, object]] = []
    for skill in runtime.registry.list():
        entries.append(
            {
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "tags": list(skill.metadata.tags),
                "required_resources": list(skill.metadata.required_resources),
                "timeout_s": skill.metadata.timeout_s,
                "interruptible": skill.metadata.interruptible,
                "arguments_schema": skill.args_model.model_json_schema(),
            }
        )
    catalog = json.dumps(entries, ensure_ascii=False, default=str)
    return (
        f"{system_prompt.rstrip()}\n\n"
        "Live registered robot skill catalog (the only callable capabilities):\n"
        f"{catalog}\n"
        "Choose tools by matching the user's goal to these descriptions and "
        "schemas. You may call multiple tools in sequence when the goal needs "
        "multiple steps; inspect each SkillResult before choosing the next step. "
        "Never invent a skill name or arguments.\n"
    )


class AgentError(RuntimeError):
    """Raised when the Agent fails to produce a usable final response."""


class AgentInvoker(Protocol):
    """Narrow boundary around the compiled LangChain graph."""

    async def ainvoke(self, input_state: dict[str, object]) -> object: ...


class RobotAgent:
    def __init__(
        self,
        runtime: SkillRuntime,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        tool_observer: SkillToolObserver | None = None,
        invoker: AgentInvoker | None = None,
    ) -> None:
        self._history: list[BaseMessage] = []
        if invoker is not None:
            self._invoker = invoker
            return

        effective_system_prompt = build_runtime_system_prompt(runtime, system_prompt)

        model = ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=base_url or os.getenv("OLLAMA_HOST"),
            temperature=0.2,
            client_kwargs={"trust_env": False},
        )
        graph = create_agent(
            model=model,
            tools=build_langchain_tools(runtime, observer=tool_observer),
            system_prompt=effective_system_prompt,
        )
        self._invoker = cast(AgentInvoker, cast(object, graph))

    async def chat(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("agent input must not be empty")

        input_messages = [*self._history, HumanMessage(content=text)]
        input_state: dict[str, object] = {"messages": input_messages}
        try:
            output = await self._invoker.ainvoke(input_state)
        except Exception as exc:
            raise AgentError(f"Agent invocation failed: {exc}") from exc

        messages = self._extract_messages(output)
        reply = next(
            (
                str(message.text).strip()
                for message in reversed(messages)
                if isinstance(message, AIMessage) and str(message.text).strip()
            ),
            "",
        )
        if not reply:
            raise AgentError("Agent did not return a final text response")

        self._history = messages
        return reply

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def _extract_messages(output: object) -> list[BaseMessage]:
        if not isinstance(output, Mapping):
            raise AgentError("Agent returned an invalid state")
        raw_messages = output.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages, (str, bytes)
        ):
            raise AgentError("Agent state does not contain messages")
        messages = [
            message for message in raw_messages if isinstance(message, BaseMessage)
        ]
        if not messages:
            raise AgentError("Agent state contains no valid messages")
        return messages
