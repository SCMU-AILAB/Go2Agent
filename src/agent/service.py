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
Match Chinese (and English) requests to the exact tool; never invent tools.

Canonical mapping:
- 站起来 / 起身 / stand up -> stand_up
- 趴下 / 躺下 / lie down -> stand_down
- 坐下 / sit -> sit
- 从坐姿起身 -> rise_sit
- 平衡站立 / 站好 -> balance_stand
- 打招呼 / 你好 / say hi -> hello (or wave only if the user literally says wave)
- 比心 / 送心 / heart -> heart
- 伸懒腰 / stretch -> stretch
- 得意 / 开心 / 满足 -> content
- 作揖 / scrape -> scrape
- 跳舞 / 跳一段舞 -> dance1 (dance2 only if they want the other dance)
- 向前走 / 前进 -> stand_up first if needed, then move_forward
- 后退 -> move_backward
- 向左走 -> move_left; 向右走 -> move_right
- 左转 -> turn_left; 右转 -> turn_right
- 停下 / 停止 -> stop (or stop_move if they name StopMove)

Hard rules:
- Never substitute hello/wave for 比心, 跳舞, 坐下, 趴下, or locomotion.
  If a skill is unavailable, say so instead of greeting.
- Before any move_* or turn_* skill, ensure stand_up first; if stand_up fails,
  abort the move and report the failure.
- Moves are short bounded open-loop steps (about 0.05-0.3 m), not obstacle-aware
  navigation. Do not promise path following or collision avoidance.
- This text agent receives no camera images, even when the live preview is on.
- Operator-only tools (flips, gaits, damp, recovery_stand, switch_joystick,
  auto_recover_set, dangerous flags) are present only when enabled at startup;
  do not call them for casual chat.
- For boolean flag tools such as pose, pass true to enable or false to disable.
- The registered wave alias uses Go2 hello, not a humanoid arm wave.
- Never invent handshake, high-five, or custom humanoid arm actions.
- SDK status 0 means the command was accepted, not that the motion finished.
  Describe unverified results as commands sent. If a tool fails, explain briefly.
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
