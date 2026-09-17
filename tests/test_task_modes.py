"""Task routing exercised with real tools/runtime and fake models/camera."""

import unittest

from fastapi.testclient import TestClient

from adapters.langchain import build_langchain_tools
from app.api import create_app
from app.backend import BackendConfig, ConsoleBackend
from tests.test_console_vision import Camera, wait_for


class TaskModeTests(unittest.TestCase):
    def build(self, source="local"):
        self.inputs = []
        owner = self

        def factory(runtime, prompt, observer):
            tools = {
                t.name: t for t in build_langchain_tools(runtime, observer=observer)
            }

            class Agent:
                async def chat(self, instruction):
                    owner.inputs.append(instruction)
                    names = {
                        "比心": "heart",
                        "坐下": "sit",
                        "站起来": "stand_up",
                        "跳舞": "dance1",
                        "打招呼": "hello",
                    }
                    await tools[names[instruction]].ainvoke({})
                    return "命令已发送"

                def reset(self):
                    pass

            return Agent()

        return ConsoleBackend(
            BackendConfig(robot_model="go2", camera_source=source, audio_enabled=False),
            agent_factory=factory,
            camera_factory=Camera,
            vision_agent_factory=lambda _: self.fail("text task entered vision agent"),
        )

    def test_live_preview_text_tasks_reach_matching_tool_not_hello(self):
        backend = self.build()
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            for instruction, skill in [
                ("比心", "heart"),
                ("坐下", "sit"),
                ("站起来", "stand_up"),
                ("跳舞", "dance1"),
                ("打招呼", "hello"),
            ]:
                with self.subTest(instruction=instruction):
                    response = client.post(
                        "/api/v1/tasks",
                        json={
                            "instruction": instruction,
                            "taskMode": "text",
                            "cameraSource": "local",
                        },
                    )
                    self.assertEqual(response.status_code, 202)
                    state = wait_for(client, lambda s: not s["busy"])
                    self.assertEqual([t["name"] for t in state["tools"]], [skill])
                    self.assertTrue(state["tools"][0]["result"]["success"])
                    self.assertEqual(state["cameraSource"], "local")
                    self.assertTrue(state["camera"]["frameAvailable"])
                    self.assertIsNone(backend._vision_worker)
            self.assertEqual(self.inputs, ["比心", "坐下", "站起来", "跳舞", "打招呼"])

    def test_gesture_requires_real_camera_and_rejects_invalid_mode(self):
        backend = self.build("demo")
        with TestClient(create_app(backend=backend)) as client:
            for mode in ["gesture", "unrestricted"]:
                response = client.post(
                    "/api/v1/tasks", json={"instruction": "比心", "taskMode": mode}
                )
                self.assertEqual(response.status_code, 422)
            self.assertFalse(backend.busy)
            self.assertEqual(self.inputs, [])

    def test_demo_legacy_text_path_is_preserved(self):
        backend = self.build("demo")
        with TestClient(create_app(backend=backend)) as client:
            self.assertEqual(
                client.post("/api/v1/tasks", json={"instruction": "比心"}).status_code,
                202,
            )
            state = wait_for(client, lambda s: not s["busy"])
            self.assertEqual(state["tools"][0]["name"], "heart")
