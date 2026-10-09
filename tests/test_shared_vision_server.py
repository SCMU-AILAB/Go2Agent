"""Exercise the real HTTP queue with a fake inference runtime; no GPU or robot."""

import asyncio
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import httpx


def load_server():
    spec = importlib.util.spec_from_file_location(
        "normality_vision_server",
        Path(__file__).parents[1] / "scripts/unifolm-vision-server.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRuntime:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = SimpleNamespace(device="fake", dtype="fake")
        self.load_s = 0
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def invoke(self, body):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.06)
            return {
                "request_id": body.request_id,
                "output": "test evidence",
                "metrics": {},
            }
        finally:
            with self.lock:
                self.active -= 1


class SharedVisionServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_timeout_returns_429_and_recovers(self):
        server = load_server()
        original_wait = asyncio.wait_for
        with patch.object(server, "UnifolmRuntime", FakeRuntime):
            app = server.create_app("fake")
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client,
            ):
                body = {
                    "request_id": 1,
                    "model": "fake",
                    "frames": ["fake"],
                    "prompt": "read evidence",
                }
                first = asyncio.create_task(client.post("/v1/vision/invoke", json=body))
                while app.state.runtime.active == 0:
                    await asyncio.sleep(0.001)

                async def fast_wait(awaitable, timeout):
                    return await original_wait(awaitable, timeout=0.01)

                with patch.object(server.asyncio, "wait_for", side_effect=fast_wait):
                    rejected = await client.post(
                        "/v1/vision/invoke", json={**body, "request_id": 2}
                    )
                self.assertEqual(rejected.status_code, 429)
                self.assertEqual((await first).status_code, 200)
                self.assertEqual(
                    (
                        await client.post(
                            "/v1/vision/invoke", json={**body, "request_id": 3}
                        )
                    ).status_code,
                    200,
                )
                self.assertEqual(app.state.pending_requests, 0)

    async def test_two_clients_queue_successfully_and_overload_is_bounded(self):
        server = load_server()
        with patch.object(server, "UnifolmRuntime", FakeRuntime):
            app = server.create_app("fake")
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client,
            ):

                def request(number):
                    return client.post(
                        "/v1/vision/invoke",
                        json={
                            "request_id": number,
                            "model": "fake",
                            "frames": ["fake"],
                            "prompt": "read evidence",
                        },
                    )

                responses = await asyncio.gather(request(1), request(2))
                self.assertEqual([r.status_code for r in responses], [200, 200])
                self.assertGreater(responses[1].json()["metrics"]["queue_wait_s"], 0.03)
                self.assertEqual(app.state.runtime.max_active, 1)
                app.state.pending_requests = 8
                rejected = await request(3)
                self.assertEqual(rejected.status_code, 429)
                self.assertEqual(rejected.headers["Retry-After"], "1")
                app.state.pending_requests = 0
                self.assertEqual((await request(4)).status_code, 200)

    async def test_cancelled_client_does_not_release_running_inference_slot(self):
        server = load_server()
        with patch.object(server, "UnifolmRuntime", FakeRuntime):
            app = server.create_app("fake")
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://test"
                ) as client,
            ):
                body = {
                    "request_id": 1,
                    "model": "fake",
                    "frames": ["fake"],
                    "prompt": "read evidence",
                }
                first = asyncio.create_task(client.post("/v1/vision/invoke", json=body))
                while app.state.runtime.active == 0:
                    await asyncio.sleep(0.001)
                first.cancel()
                second = asyncio.create_task(
                    client.post("/v1/vision/invoke", json={**body, "request_id": 2})
                )
                await asyncio.gather(first, return_exceptions=True)
                self.assertEqual((await second).status_code, 200)
                self.assertEqual(app.state.runtime.max_active, 1)
                self.assertEqual(app.state.pending_requests, 0)
