from __future__ import annotations

import asyncio
import time
import unittest
from typing import cast

from fastapi.testclient import TestClient

from adapters import SpeechOutput, SpeechRecognizer
from adapters.langchain import SkillToolObserver
from app.api import _build_parser, create_app
from app.backend import AgentFactory, BackendConfig, ConsoleBackend
from core.runtime import SkillRuntime
from robot import RobotCommandError, SimulatedRobotAdapter
from skills import build_go2_all_skills


class FakeAgent:
    def __init__(self, reply: str = "好的，任务已完成。") -> None:
        self.reply = reply
        self.inputs: list[str] = []

    async def chat(self, text: str) -> str:
        self.inputs.append(text)
        await asyncio.sleep(0)
        return self.reply

    def reset(self) -> None:
        self.inputs.clear()


class SlowAgent(FakeAgent):
    async def chat(self, text: str) -> str:
        self.inputs.append(text)
        await asyncio.sleep(60)
        return self.reply


class FakeRecognizer(SpeechRecognizer):
    def __init__(self, transcript: str = "给我比个心") -> None:
        self.transcript = transcript
        self.calls = 0
        self.closed = False
        self._after_first = asyncio.Event()

    async def warmup(self) -> None:
        return None

    async def transcribe_once(self) -> str:
        self.calls += 1
        if self.calls == 1:
            return self.transcript
        await self._after_first.wait()
        return ""

    async def close(self) -> None:
        self.closed = True
        self._after_first.set()


class FakeSpeechOutput(SpeechOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def speak(self, text: str) -> None:
        self.messages.append(text)

    async def close(self) -> None:
        self.closed = True


def fake_agent_factory(
    runtime: SkillRuntime,
    system_prompt: str,
    observer: SkillToolObserver,
) -> FakeAgent:
    del runtime, system_prompt, observer
    return FakeAgent()


def slow_agent_factory(
    runtime: SkillRuntime,
    system_prompt: str,
    observer: SkillToolObserver,
) -> SlowAgent:
    del runtime, system_prompt, observer
    return SlowAgent()


class ApiTests(unittest.TestCase):
    def build_backend(
        self,
        *,
        agent_factory: AgentFactory = fake_agent_factory,
    ) -> ConsoleBackend:
        return ConsoleBackend(
            BackendConfig(audio_enabled=False),
            agent_factory=agent_factory,
        )

    def test_lifespan_starts_simulation_without_hardware(self) -> None:
        backend = self.build_backend()
        self.assertIsInstance(backend.robot, SimulatedRobotAdapter)
        self.assertIsNone(backend.hardware_robot)

        with TestClient(create_app(backend=backend)) as client:
            health = client.get("/api/v1/health")
            console = client.get("/api/v1/console")

            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ready")
            self.assertFalse(health.json()["hardware"])
            self.assertTrue(console.json()["backend"])
            self.assertEqual(console.json()["robot"]["mode"], "simulation")

        self.assertFalse(backend.backend)

    def test_go2_config_registers_complete_catalog(self) -> None:
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False, robot_model="go2"),
            agent_factory=fake_agent_factory,
        )
        registered = {skill.metadata.name for skill in backend.runtime.registry.list()}
        expected = {skill.metadata.name for skill in build_go2_all_skills()}
        self.assertEqual(registered, expected)
        self.assertIn("hello", registered)
        self.assertIn("damp", registered)
        self.assertTrue(
            backend.system_prompt.startswith(
                "You are the conversational controller for a Unitree Go2"
            )
        )
        self.assertEqual(backend.config.robot_model, "go2")

    def test_config_rejects_other_robot_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "Go2 only"):
            BackendConfig(robot_model="unsupported")  # type: ignore[arg-type]

    def test_vision_window_cli_defaults_and_overrides(self) -> None:
        parser = _build_parser()
        defaults = parser.parse_args([])
        overridden = parser.parse_args(
            ["--vision-window-s", "2.0", "--vision-frame-count", "8"]
        )

        self.assertEqual(defaults.vision_window_s, 0.8)
        self.assertEqual(defaults.vision_frame_count, 3)
        self.assertEqual(overridden.vision_window_s, 2.0)
        self.assertEqual(overridden.vision_frame_count, 8)

    def test_unifolm_cli_selects_unitree_model_by_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--vision-backend", "unifolm"])

        self.assertEqual(args.vision_backend, "unifolm")
        self.assertIsNone(args.vision_model)
        self.assertIsNone(args.vision_url)

    def test_llamacpp_cli_selects_local_quantized_backend(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--vision-backend", "llamacpp"])

        self.assertEqual(args.vision_backend, "llamacpp")
        self.assertIsNone(args.vision_model)
        self.assertIsNone(args.vision_url)

    def test_console_schema_matches_flutter_field_names(self) -> None:
        with TestClient(create_app(backend=self.build_backend())) as client:
            payload = client.get("/api/v1/console").json()

        expected = {
            "backend",
            "starting",
            "busy",
            "promptSaved",
            "sessionId",
            "cameraSource",
            "modelStatus",
            "skillStatus",
            "skillName",
            "progress",
            "progressText",
            "activeStep",
            "currentTask",
            "modelOutput",
            "modelDuration",
            "latency",
            "taskCount",
            "tools",
            "logs",
        }
        self.assertTrue(expected.issubset(payload))
        self.assertNotIn("model_duration", payload)

    def test_skill_catalog_and_direct_wave_execution(self) -> None:
        backend = self.build_backend()
        with TestClient(create_app(backend=backend)) as client:
            catalog = client.get("/api/v1/skills")
            execution = client.post(
                "/api/v1/skills/wave/execute",
                json={"arguments": {}},
            )

            self.assertEqual(catalog.status_code, 200)
            self.assertIn(
                "wave",
                {skill["name"] for skill in catalog.json()["skills"]},
            )
            self.assertEqual(execution.status_code, 200)
            self.assertTrue(execution.json()["result"]["success"])
            self.assertEqual(execution.json()["result"]["status"], "succeeded")

        robot = backend.robot
        self.assertIsInstance(robot, SimulatedRobotAdapter)
        simulated = cast(SimulatedRobotAdapter, robot)
        self.assertIn(("loco_action", ("hello", {})), simulated.events)

    def test_task_runs_in_background_and_updates_console(self) -> None:
        with TestClient(create_app(backend=self.build_backend())) as client:
            accepted = client.post(
                "/api/v1/tasks",
                json={"instruction": "跟我打个招呼"},
            )
            self.assertEqual(accepted.status_code, 202)
            self.assertTrue(accepted.json()["busy"])

            deadline = time.monotonic() + 2
            payload = accepted.json()
            while payload["busy"] and time.monotonic() < deadline:
                time.sleep(0.01)
                payload = client.get("/api/v1/console").json()

            self.assertFalse(payload["busy"])
            self.assertEqual(payload["modelStatus"], "已完成")
            self.assertIn("好的，任务已完成。", payload["modelOutput"])
            self.assertEqual(payload["taskCount"], 1)

    def test_task_conflict_and_cancel(self) -> None:
        backend = self.build_backend(agent_factory=slow_agent_factory)
        with TestClient(create_app(backend=backend)) as client:
            first = client.post("/api/v1/tasks", json={"instruction": "持续动作"})
            second = client.post("/api/v1/tasks", json={"instruction": "另一个任务"})
            cancelled = client.post(
                "/api/v1/tasks/current/cancel",
                json={"reason": "测试取消"},
            )

            self.assertEqual(first.status_code, 202)
            self.assertEqual(second.status_code, 409)
            self.assertEqual(cancelled.status_code, 200)
            self.assertFalse(cancelled.json()["busy"])
            self.assertEqual(cancelled.json()["skillStatus"], "STOPPED")

    def test_websocket_starts_with_current_state(self) -> None:
        with TestClient(create_app(backend=self.build_backend())) as client:
            with client.websocket_connect("/api/v1/events") as websocket:
                event = websocket.receive_json()

            self.assertEqual(event["type"], "state")
            self.assertTrue(event["data"]["backend"])
            self.assertIn("sessionId", event["data"])

    def test_local_voice_routes_transcript_through_agent_and_speaker(self) -> None:
        recognizer = FakeRecognizer()
        speaker = FakeSpeechOutput()
        backend = ConsoleBackend(
            BackendConfig(robot_model="go2", audio_enabled=True),
            agent_factory=fake_agent_factory,
            asr_factory=lambda: recognizer,
            speech_factory=lambda lock: speaker,
        )
        with TestClient(create_app(backend=backend)) as client:
            started = client.post("/api/v1/voice/start")
            self.assertEqual(started.status_code, 200)
            deadline = time.monotonic() + 2
            payload = started.json()
            while not payload["voice"]["reply"] and time.monotonic() < deadline:
                time.sleep(0.01)
                payload = client.get("/api/v1/console").json()

            self.assertEqual(payload["voice"]["transcript"], "给我比个心")
            self.assertEqual(payload["voice"]["reply"], "好的，任务已完成。")
            self.assertEqual(speaker.messages, ["好的，任务已完成。"])

            spoken = client.post(
                "/api/v1/voice/speak",
                json={"text": "扬声器测试"},
            )
            stopped = client.post("/api/v1/voice/stop")
            self.assertEqual(spoken.status_code, 200)
            self.assertEqual(speaker.messages[-1], "扬声器测试")
            self.assertFalse(stopped.json()["voice"]["enabled"])

        self.assertTrue(recognizer.closed)
        self.assertTrue(speaker.closed)

    def test_go2_offline_voice_commands_do_not_use_text_llm(self) -> None:
        recognizer = FakeRecognizer("给我比个心")
        speaker = FakeSpeechOutput()
        backend = ConsoleBackend(
            BackendConfig(
                robot_model="go2",
                audio_enabled=True,
                voice_agent_backend="local_commands",
            ),
            agent_factory=slow_agent_factory,
            asr_factory=lambda: recognizer,
            speech_factory=lambda lock: speaker,
        )
        with TestClient(create_app(backend=backend)) as client:
            client.post("/api/v1/voice/start")
            deadline = time.monotonic() + 2
            payload = client.get("/api/v1/console").json()
            while not payload["voice"]["reply"] and time.monotonic() < deadline:
                time.sleep(0.01)
                payload = client.get("/api/v1/console").json()

            self.assertEqual(payload["voice"]["transcript"], "给我比个心")
            self.assertEqual(payload["voice"]["reply"], "好的，比心指令已发送。")
            self.assertEqual(speaker.messages, ["好的，比心指令已发送。"])

        robot = cast(SimulatedRobotAdapter, backend.robot)
        self.assertIn(("loco_action", ("heart", {})), robot.events)


class EmergencyStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_spoken_stop_preempts_text_agent(self) -> None:
        entered = asyncio.Event()

        class WaitingAgent:
            async def chat(self, text: str) -> str:
                entered.set()
                await asyncio.Event().wait()
                return "done"

            def reset(self) -> None:
                pass

        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False),
            agent_factory=lambda *args: WaitingAgent(),
            asr_factory=lambda: FakeRecognizer("停止"),
        )
        await backend.start()
        try:
            await backend.submit_task("持续动作")
            await asyncio.wait_for(entered.wait(), 1.0)
            await backend.start_voice()
            async with asyncio.timeout(1.0):
                while backend.voice_reply != "好的，已停止当前任务。":
                    await asyncio.sleep(0.01)
            self.assertFalse(backend.busy)
            self.assertIn(("stop", None), backend.robot.events)
        finally:
            await backend.stop()

    async def test_emergency_stop_cancels_independent_voice_turn(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingVoiceAgent:
            async def chat(self, text: str) -> str:
                entered.set()
                await release.wait()
                await backend.runtime.execute("hello")
                return "done"

            def reset(self) -> None:
                pass

        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False, voice_enabled=True),
            agent_factory=lambda *args: WaitingVoiceAgent(),
            asr_factory=FakeRecognizer,
        )
        await backend.start()
        try:
            await asyncio.wait_for(entered.wait(), timeout=1.0)
            await backend.emergency_stop()
            release.set()
            await asyncio.sleep(0.03)
            self.assertFalse(backend.voice_enabled)
            self.assertNotIn(("loco_action", ("hello", {})), backend.robot.events)
        finally:
            await backend.stop()

    async def test_emergency_stop_cancels_manual_motion_and_latches(self) -> None:
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False), agent_factory=fake_agent_factory
        )
        await backend.start()
        try:
            request = asyncio.create_task(
                backend.execute_skill("move", {"duration_s": 1.0})
            )
            await asyncio.sleep(0.04)
            await backend.emergency_stop()
            mark = len(backend.robot.events)
            await asyncio.sleep(0.06)
            self.assertTrue(request.cancelled())
            self.assertFalse(any(
                name == "move_velocity" for name, _ in backend.robot.events[mark:]
            ))
            with self.assertRaisesRegex(Exception, "latched"):
                await backend.execute_skill("move", {})
        finally:
            await backend.stop()

    async def test_failed_stop_is_not_reported_as_success(self) -> None:
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False), agent_factory=fake_agent_factory
        )
        await backend.start()
        original_stop = backend.robot.stop

        async def fail_stop() -> None:
            raise RobotCommandError("SDK rejected stop")

        backend.robot.stop = fail_stop  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(RobotCommandError, "SDK rejected stop"):
                await backend.emergency_stop()
            self.assertEqual(backend.skill_status, "FAILED")
            self.assertEqual(backend.model_status, "急停失败")
        finally:
            backend.robot.stop = original_stop  # type: ignore[method-assign]
            await backend.stop()


if __name__ == "__main__":
    unittest.main()
