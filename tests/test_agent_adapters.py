from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import ValidationError

from adapters.langchain import build_langchain_tools
from adapters.unitree_audio import (
    AudioOutputError,
    UnitreeAudioBindings,
    UnitreeAudioOutput,
)
from agent import AgentError, RobotAgent
from app.main import _run_turn
from app.wave import run_wave
from core.runtime import SkillRuntime
from robot import (
    ActionVerification,
    RobotCommandError,
    RobotState,
    SimulatedRobotAdapter,
)
from skills.motions import WaveSkill


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

    async def wave(self, arm: str) -> None:
        raise RobotCommandError("wave rejected")

    async def wait_for_wave_completion(
        self,
        arm: str,
        timeout_s: float,
    ) -> ActionVerification:
        return ActionVerification(
            completed=False,
            observable=False,
            message="wave feedback unavailable",
        )

    async def move_velocity(
        self,
        forward_m_s: float,
        lateral_m_s: float,
        yaw_rad_s: float,
    ) -> None:
        raise RobotCommandError("move rejected")


class AgentAdapterTests(unittest.IsolatedAsyncioTestCase):
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
        runtime.register(WaveSkill())
        model = ToolCapableFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "wave",
                            "args": {"arm": "right"},
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
        self.assertEqual(robot.events, [("wave", "right")])

    async def test_langchain_tool_invokes_skill_runtime(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        tools = build_langchain_tools(runtime)

        raw_result = await tools[0].ainvoke({"arm": "right"})

        self.assertIsInstance(raw_result, str)
        payload = json.loads(str(raw_result))
        self.assertIsInstance(payload, Mapping)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(robot.events, [("wave", "right")])

    async def test_invalid_tool_arguments_never_reach_robot(self) -> None:
        robot = SimulatedRobotAdapter()
        runtime = SkillRuntime(robot)
        runtime.register(WaveSkill())
        tools = build_langchain_tools(runtime)

        with self.assertRaises(ValidationError):
            await tools[0].ainvoke({"arm": "left"})

        self.assertEqual(robot.events, [])

    async def test_failed_skill_result_returns_through_tool(self) -> None:
        runtime = SkillRuntime(FailingRobotAdapter())
        runtime.register(WaveSkill())
        tools = build_langchain_tools(runtime)

        raw_result = await tools[0].ainvoke({"arm": "right"})

        payload = json.loads(str(raw_result))
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["failure_code"], "robot_error")
        self.assertEqual(payload["message"], "wave rejected")

    async def test_wave_runner_executes_without_agent(self) -> None:
        result = await run_wave(hardware=False)

        self.assertEqual(
            {key: result.to_dict()[key] for key in ("success", "status")},
            {"success": True, "status": "succeeded"},
        )


class FakeConnection:
    def __init__(self, connected: bool = False) -> None:
        self.connected = connected


class FakeAudioClient:
    def __init__(self) -> None:
        self.timeout_s: float | None = None
        self.init_count = 0
        self.tts_status = 0
        self.speech: list[tuple[str, int]] = []

    def set_timeout(self, seconds: float) -> None:
        self.timeout_s = seconds

    def init(self) -> None:
        self.init_count += 1

    def tts_maker(self, text: str, speaker_id: int) -> int:
        self.speech.append((text, speaker_id))
        return self.tts_status


class UnitreeAudioOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_client_requires_existing_dds_connection(self) -> None:
        connection = FakeConnection()
        client = FakeAudioClient()
        audio = UnitreeAudioOutput(
            connection,
            bindings=UnitreeAudioBindings(lambda: client),
        )

        with self.assertRaisesRegex(AudioOutputError, "DDS"):
            await audio.connect()

        self.assertEqual(client.init_count, 0)

    async def test_agent_reply_is_spoken_by_audio_client(self) -> None:
        connection = FakeConnection(connected=True)
        client = FakeAudioClient()
        audio = UnitreeAudioOutput(
            connection,
            speaker_id=3,
            timeout_s=4.0,
            bindings=UnitreeAudioBindings(lambda: client),
        )
        runtime = SkillRuntime(SimulatedRobotAdapter())
        agent = RobotAgent(
            runtime,
            invoker=FakeAgentInvoker("你好，我是 G1。"),
        )

        await audio.connect()
        await _run_turn("你好", agent, audio)
        await audio.close()

        self.assertEqual(client.timeout_s, 4.0)
        self.assertEqual(client.init_count, 1)
        self.assertEqual(client.speech, [("你好，我是 G1。", 3)])
        self.assertFalse(audio.connected)

    async def test_nonzero_tts_status_is_reported(self) -> None:
        connection = FakeConnection(connected=True)
        client = FakeAudioClient()
        client.tts_status = 12
        audio = UnitreeAudioOutput(
            connection,
            bindings=UnitreeAudioBindings(lambda: client),
        )
        await audio.connect()

        with self.assertRaisesRegex(AudioOutputError, "SDK status 12"):
            await audio.speak("测试")
