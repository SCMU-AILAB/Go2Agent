"""Conversational Agent that can invoke registered robot skills as tools."""

from __future__ import annotations

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
Only use registered skill tools when the user requests the corresponding action.
Go2 has native heart (比心), hello, stretch, and dance1/dance2 actions.
Match the requested action: 比心 -> heart; 坐下 -> sit; 站起来 -> stand_up;
跳舞 -> dance1 or dance2; 向前移动 -> move_forward; 打招呼 -> hello.
Never substitute hello/wave for a different requested skill. If unavailable,
explain the missing capability instead of greeting. Before a requested move,
ensure standing posture via stand_up; abort movement if standing fails.
Moves are short bounded open-loop motions, not obstacle-aware navigation.
This text agent receives no camera images, even when the live preview is on.
Do not claim to see people or obstacles or guarantee collision avoidance.
The registered wave alias uses Go2 hello, not a humanoid arm wave.
Never invent handshake, high-five, or custom humanoid arm actions.
For boolean flag tools, pass true to enable or false to disable.
Operator-only tools are available only when explicitly enabled at startup.
SDK command acceptance does not prove physical completion; describe unverified
results as commands sent. If a tool fails, explain the failure briefly.
Do not invent robot capabilities or emit action JSON.
"""


def system_prompt_for(robot_model: str) -> str:
    if robot_model == "go2":
        return GO2_SYSTEM_PROMPT
    return SYSTEM_PROMPT


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

        model = ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=base_url or os.getenv("OLLAMA_HOST"),
            temperature=0.2,
            client_kwargs={"trust_env": False},
        )
        graph = create_agent(
            model=model,
            tools=build_langchain_tools(runtime, observer=tool_observer),
            system_prompt=system_prompt,
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
