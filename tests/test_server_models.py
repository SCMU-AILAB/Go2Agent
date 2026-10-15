import asyncio
import socket
import sys
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from go2agent.server.models import ModelState, VLLMConfig, VLLMManager, create_router

FAKE_VLLM = r"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model = int(sys.argv[1]), sys.argv[2]
if model == "broken":
    sys.exit(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self._send({"ok": True, "model": model})
    def do_POST(self):
        n = int(self.headers["content-length"]); req = json.loads(self.rfile.read(n))
        if req.get("stream"):
            import time
            self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
            for part in ("a", "b", "c"):
                self.wfile.write(f"data: {part}\n\n".encode()); self.wfile.flush(); time.sleep(0.3)
            return
        self._send({"echo": req, "model": model})
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def manager(tmp_path):
    port = free_port()
    m = VLLMManager(
        VLLMConfig(model="m1", port=port, load_timeout_s=20),
        tmp_path / "vllm.log",
        command=lambda model: [sys.executable, "-c", FAKE_VLLM, str(port), model],
    )
    yield m
    await m.close()


def client(m):
    app = FastAPI()
    app.include_router(create_router(m))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc")


async def test_load_proxy_unload(manager):
    async with client(manager) as c:
        r = await c.post("/v1/chat/completions", json={"x": 1})
        assert r.status_code == 503 and "not loaded" in r.json()["error"]
        st = (await c.post("/models/brain/load")).json()
        assert st["state"] == ModelState.READY and st["model"] == "m1" and st["pid"]
        r = await c.post("/v1/chat/completions", json={"x": 1})
        assert r.status_code == 200 and r.json() == {"echo": {"x": 1}, "model": "m1"}
        pid = st["pid"]
        assert (await c.post("/models/brain/load")).json()["pid"] == pid  # already loaded: no restart
        st = (await c.post("/models/brain/load", json={"model": "m2"})).json()
        assert st["model"] == "m2" and st["pid"] != pid  # different model: restart
        st = (await c.post("/models/brain/unload")).json()
        assert st["state"] == ModelState.UNLOADED
        assert (await c.post("/v1/chat/completions", json={})).status_code == 503
        assert (await c.get("/models")).json()["brain"]["state"] == "unloaded"


async def test_failed_load_reports_error(manager):
    st = await manager.load("broken")
    assert st["state"] == ModelState.FAILED and "exited with code 3" in st["error"]
    assert not manager.ready


async def test_loopback_ignores_proxy_env(manager, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # unreachable proxy
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    st = await manager.load()
    assert st["state"] == ModelState.READY


async def test_proxy_streams_chunks_as_they_arrive(manager):
    await manager.load()
    app = FastAPI()
    app.include_router(create_router(manager))
    config = uvicorn.Config(app, host="127.0.0.1", port=free_port(), log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        url = f"http://127.0.0.1:{config.port}/v1/chat/completions"
        arrivals = []
        async with httpx.AsyncClient(trust_env=False) as c, c.stream("POST", url, json={"stream": True}) as r:
            t0 = time.monotonic()
            async for line in r.aiter_lines():
                if line:
                    arrivals.append((line, time.monotonic() - t0))
        assert [a[0] for a in arrivals] == ["data: a", "data: b", "data: c"]
        assert arrivals[0][1] < 0.25 < arrivals[-1][1]  # first chunk did not wait for the last
    finally:
        server.should_exit = True
        await task
