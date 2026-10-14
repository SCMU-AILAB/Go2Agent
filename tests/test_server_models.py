import socket
import sys

import httpx
import pytest
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
