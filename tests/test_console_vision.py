"""HTTP/WebSocket -> real vision worker -> simulated skills; no hardware/model."""

import asyncio
import io
import time
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from agent.social_vision import SocialVisionAgent
from app.api import create_app
from app.backend import BackendConfig, ConsoleBackend
from perception import CameraFrame, PerceptionResult
from tests.test_api import fake_agent_factory


class Camera:
    def __init__(self):
        image = Image.new("RGB", (80, 40), "red")
        image.paste("blue", (40, 0, 80, 40))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        self.rgb = buf.getvalue()
        self.opened = False
        self.fail = False
        self.stale = False
        self.distance_m = 2.0
        self.capture_count = 0

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def capture_frame(self):
        time.sleep(0.02)
        self.capture_count += 1
        if self.fail:
            raise RuntimeError("test camera disconnected")
        now = time.monotonic() - (20 if self.stale else 0)
        return CameraFrame(
            observed_at_s=now,
            rgb=self.rgb,
            depth=None,
            observation=PerceptionResult(observed_at_s=now, source="test"),
            nearest_obstacle_distance_m=self.distance_m,
        )


class VisionInvoker:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.bad = False
        self.slow = False
        self.cancelled = False

    async def warmup(self):
        pass

    async def close(self):
        self.closed = True

    async def ainvoke(self, frames, prompt):
        self.calls.append((frames, prompt))
        if self.slow:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.bad:
            return "bad json"
        return {
            "gesture": "wave",
            "hand_visible": True,
            "directed_at_robot": True,
            "present_in_latest": True,
            "evidence": "side_to_side",
            "speech": "你好呀！",
        }


class Audio:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)

    async def close(self):
        pass


class BlockingSafetyWorker:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def observe_frame(self, frame):
        pass

    def set_safety_latched(self, latched):
        pass

    async def stop_locomotion_for_safety(self, reason):
        self.started.set()
        await self.release.wait()
        return False


def wait_for(client, predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        snapshot = client.get("/api/v1/console").json()
        if predicate(snapshot):
            return snapshot
        time.sleep(0.02)
    raise AssertionError(snapshot)


class ConsoleVisionTests(unittest.TestCase):
    def test_cancel_pending_inference_prevents_later_actions(self):
        backend, _camera, invoker = self.build()
        invoker.slow = True
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.post("/api/v1/tasks", json={"instruction": "回应挥手"})
            wait_for(client, lambda s: bool(invoker.calls))
            client.post("/api/v1/tasks/current/cancel", json={})
            self.assertTrue(invoker.cancelled)
            self.assertTrue(invoker.closed)
            self.assertNotIn(("wave", "right"), backend.robot.events)
            self.assertIsNone(backend._vision_worker)

    def test_default_vision_factory_uses_console_preferences(self):
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False), agent_factory=fake_agent_factory
        )
        backend.system_prompt = "简短回应"
        agent = backend._build_vision_agent("观察挥手")
        self.assertEqual(agent.model_name, "qwen3.5:9b")
        self.assertTrue(agent.generate_speech)
        self.assertIn("简短回应", agent.task_context)
        self.assertIn("观察挥手", agent.task_context)
        self.assertIs(agent._invoker.think, False)
        self.assertEqual(backend.config.vision_frame_count, 3)
        self.assertEqual(backend.config.vision_window_s, 0.8)

    def test_default_camera_applies_rotation_before_jpeg_encoding(self):
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False, vision_rotation_deg=180),
            agent_factory=fake_agent_factory,
        )

        camera = backend._build_camera()

        self.assertEqual(camera.rgb_rotation_deg, 180)

    def build(self):
        camera = Camera()
        invoker = VisionInvoker()
        backend = ConsoleBackend(
            BackendConfig(
                camera_source="local", audio_enabled=False, vision_rotation_deg=180
            ),
            agent_factory=fake_agent_factory,
            camera_factory=lambda: camera,
            vision_agent_factory=lambda text: SocialVisionAgent(
                invoker=invoker,
                prompt_profile="egocentric",
                generate_speech=True,
                task_context=backend.system_prompt + "\n" + text,
            ),
        )
        return backend, camera, invoker

    def test_local_task_transmits_images_executes_reports_and_cancels(self):
        backend, camera, invoker = self.build()
        audio = Audio()
        with TestClient(create_app(backend=backend)) as client:
            backend._audio = audio
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.put(
                "/api/v1/config/system-prompt", json={"systemPrompt": "请用中文"}
            )
            result = client.post(
                "/api/v1/tasks",
                json={"instruction": "回应挥手", "cameraSource": "local"},
            )
            self.assertEqual(result.status_code, 202)
            snap = wait_for(
                client, lambda s: any(t["name"] == "vision.outcome" for t in s["tools"])
            )
            self.assertTrue(snap["busy"])
            self.assertEqual(audio.spoken, ["你好呀！"])
            self.assertIn(("wave", "right"), backend.robot.events)
            self.assertIn("回应挥手", invoker.calls[0][1])
            self.assertIn("请用中文", invoker.calls[0][1])
            self.assertGreaterEqual(len(invoker.calls[0][0]), 2)
            preview = client.get("/api/v1/camera/frame.jpg").content
            self.assertEqual(invoker.calls[0][0][-1], preview)
            image = Image.open(io.BytesIO(preview))
            self.assertGreater(image.getpixel((10, 20))[2], 240)
            outcome = next(
                t["result"] for t in snap["tools"] if t["name"] == "vision.outcome"
            )
            self.assertTrue(outcome["speech_spoken"])
            with client.websocket_connect("/api/v1/events") as websocket:
                state = websocket.receive_json()
                self.assertEqual(state["type"], "state")
                self.assertIn(
                    "vision.decide", [t["name"] for t in state["data"]["tools"]]
                )
            self.assertEqual(
                client.put(
                    "/api/v1/camera/source", json={"source": "demo"}
                ).status_code,
                409,
            )
            self.assertEqual(
                client.post(
                    "/api/v1/skills/wave/execute", json={"arguments": {}}
                ).status_code,
                409,
            )
            stopped = client.post("/api/v1/tasks/current/cancel", json={}).json()
            self.assertFalse(stopped["busy"])
            self.assertIsNone(backend._vision_worker)
            self.assertTrue(invoker.closed)
            count = len(invoker.calls)
            time.sleep(0.15)
            self.assertEqual(len(invoker.calls), count)
        self.assertFalse(camera.opened)

    def test_stale_frames_and_bad_output_never_execute(self):
        for fault in ("stale", "bad"):
            with self.subTest(fault=fault):
                backend, camera, invoker = self.build()
                camera.stale = fault == "stale"
                invoker.bad = fault == "bad"
                with TestClient(create_app(backend=backend)) as client:
                    wait_for(client, lambda s: s["camera"]["frameAvailable"])
                    client.post("/api/v1/tasks", json={"instruction": "回应挥手"})
                    wait_for(
                        client, lambda s: not s["busy"] and s["modelStatus"] == "失败"
                    )
                    self.assertNotIn(("wave", "right"), backend.robot.events)
                    self.assertIsNone(backend._vision_worker)

    def test_camera_disconnect_stops_inference_and_clears_preview_on_switch(self):
        backend, camera, _invoker = self.build()
        with TestClient(create_app(backend=backend)) as client:
            wait_for(client, lambda s: s["camera"]["frameAvailable"])
            client.post("/api/v1/tasks", json={"instruction": "回应挥手"})
            wait_for(
                client, lambda s: any(t["name"] == "vision.decide" for t in s["tools"])
            )
            camera.fail = True
            wait_for(
                client, lambda s: not s["busy"] and s["camera"]["status"] == "error"
            )
            self.assertIsNone(backend._vision_worker)
            self.assertEqual(client.get("/api/v1/camera/frame.jpg").status_code, 404)
            self.assertEqual(
                client.post("/api/v1/tasks", json={"instruction": "重试"}).status_code,
                502,
            )
            client.put("/api/v1/camera/source", json={"source": "demo"})
            self.assertEqual(len(backend._video_buffer), 0)


class ConsoleCameraConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_depth_safety_stop_does_not_block_camera_capture(self):
        camera = Camera()
        camera.distance_m = 0.2
        backend = ConsoleBackend(
            BackendConfig(audio_enabled=False, vision_rotation_deg=0),
            agent_factory=fake_agent_factory,
            camera_factory=lambda: camera,
        )
        worker = BlockingSafetyWorker()
        backend._vision_worker = worker
        camera_task = asyncio.create_task(backend._camera_loop(camera))

        try:
            await asyncio.wait_for(worker.started.wait(), timeout=1.0)
            count_while_stop_is_blocked = camera.capture_count
            await asyncio.sleep(0.1)
            self.assertGreater(camera.capture_count, count_while_stop_is_blocked)
        finally:
            worker.release.set()
            safety_task = backend._safety_stop_task
            if safety_task is not None:
                await asyncio.wait_for(safety_task, timeout=1.0)
            camera_task.cancel()
            await asyncio.gather(camera_task, return_exceptions=True)
