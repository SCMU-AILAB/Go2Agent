from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence
from unittest.mock import AsyncMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import ValidationError

from adapters.langchain import build_langchain_tools
from agent import AgentError, RobotAgent
from app.go2_action import run_action
from app.main import _run_turn
from core.runtime import SkillRuntime
from robot import (
    RobotCommandError,
    RobotState,
    SimulatedRobotAdapter,
)
from tests.go2_helpers import go2_skill


class FakeAgentInvoker:
    def __init__(self, reply: str = "好的，我已经挥手了。") -> None:
        self.reply = reply
        self.inputs: list[dict[str, object]] = []

    async def ainvoke(self, input_state: dict[str, object]) -> object:
        self.inputs.append(input_state)
        raw_messages = input_state.get("messages")
        if not isinstance(raw_messages, Sequence):
            raise TypeError("messages missing")
        messages = [
            message for message in raw_messages if isinstance(message, BaseMessage)
        ]
        return {"messages": [*messages, AIMessage(content=self.reply)]}


class InvalidAgentInvoker:
    async def ainvoke(self, input_state: dict[str, object]) -> object:
        return {"messages": []}


class ToolCapableFakeChatModel(FakeMessagesListChatModel):
    """Allow LangChain's create_agent to bind tools during the integration test."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self


class FailingRobotAdapter:
    async def get_state(self) -> RobotState:
        return RobotState(hardware=False, connected=True)

    async def stop(self) -> None:
        pass

    async def execute_loco_action(self, action: str, arguments=None) -> None:
        raise RobotCommandError("wave rejected")

    async def move_velocity(
        self,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
    ) -> None:
        raise RobotCommandError("move rejected")


class AgentAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_reply_uses_host_speech_output(self) -> None:
        runtime = SkillRuntime(SimulatedRobotAdapter())
        agent = RobotAgent(runtime, invoker=FakeAgentInvoker("你好，我是 Go2。"))
        speech = AsyncMock()

        await _run_turn("你好", agent, speech)

        speech.speak.assert_awaited_once_with("你好，我是 Go2。")

    async def test_agent_accepts_text_and_returns_final_text(self) -> None:
        runtime = SkillRuntime(SimulatedRobotAdapter())
        invoker = FakeAgentInvoker()
        agent = RobotAgent(runtime, invoker=invoker)

        reply = await agent.chat("请挥手")

        self.assertEqual(reply, "好的，我已经挥手了。")
        messages = invoker.inputs[0]["messages"]
        if not isinstance(messages, list):
            self.fail("Agent input messages must be a list")
        self.assertEqual(len(messages), 1)

    async def test_agent_rejects_invalid_graph_output(self) -> None:
        runtime = SkillRuntime(SimulatedRobotAdapter())
        agent = RobotAgent(runtime, invoker=InvalidAgentInvoker())

        with self.assertRaises(AgentError):
            await agent.chat("你好")

    async def test_agent_calls_skill_then_reads_result_before_final_reply(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))
        model = ToolCapableFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "wave",
                            "args": {},
                            "id": "wave-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="已根据工具结果完成挥手。"),
            ]
        )
        agent = RobotAgent(runtime, chat_model=model)

        reply = await agent.chat("请挥手，确认动作结果后告诉我")

        self.assertEqual(reply, "已根据工具结果完成挥手。")
        self.assertEqual(robot.events, [("loco_action", ("hello", {}))])

    async def test_langchain_tool_invokes_skill_runtime(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))
        tools = build_langchain_tools(runtime)

        raw_result = await tools[0].ainvoke({})

        self.assertIsInstance(raw_result, str)
        payload = json.loads(str(raw_result))
        self.assertIsInstance(payload, Mapping)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(robot.events, [("loco_action", ("hello", {}))])

    async def test_invalid_tool_arguments_never_reach_robot(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("pose"))
        tools = build_langchain_tools(runtime)

        with self.assertRaises(ValidationError):
            await tools[0].ainvoke({"flag": "left"})

        self.assertEqual(robot.events, [])

    async def test_zero_argument_tool_rejects_extra_fields(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(go2_skill("wave"))

        with self.assertRaises(ValidationError):
            await build_langchain_tools(runtime)[0].ainvoke({"arm": "left"})

        self.assertEqual(robot.events, [])

    async def test_failed_skill_result_returns_through_tool(self) -> None:
        runtime = SkillRuntime(FailingRobotAdapter())
        runtime.register(go2_skill("wave"))
        tools = build_langchain_tools(runtime)

        raw_result = await tools[0].ainvoke({})

        payload = json.loads(str(raw_result))
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["failure_code"], "robot_error")
        self.assertEqual(payload["message"], "wave rejected")

    async def test_wave_runner_executes_without_agent(self) -> None:
        result = await run_action("wave", hardware=False)

        self.assertEqual(
            {key: result.to_dict()[key] for key in ("success", "status")},
            {"success": True, "status": "succeeded"},
        )
