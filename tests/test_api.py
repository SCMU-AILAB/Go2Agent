from fastapi.testclient import TestClient
from test_runtime import make_runtime

from go2agent.api.server import create_app


def client(clock):
    rt, *_ = make_runtime(clock, [])
    rt.control_step(clock())
    return rt, TestClient(create_app(rt))


def test_task_estop_resume(clock):
    rt, c = client(clock)
    assert c.post("/task", json={"text": "跟着我"}).json()["task"] == "跟着我"
    assert c.post("/estop").json()["estopped"] is True
    assert c.post("/resume").json()["estopped"] is False
    c.post("/hear", json={"text": "你好"})
    assert rt._pending[-1].data == {"text": "你好"}


def test_status_and_frame(clock):
    _, c = client(clock)
    s = c.get("/status").json()
    assert s["tracks"][0]["id"] == 3 and s["tracker_healthy"] is True
    r = c.get("/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"


def test_events_sends_status_first(clock):
    _, c = client(clock)
    with c.websocket_connect("/events") as ws:
        assert ws.receive_json()["kind"] == "status"


def test_video_streams_jpeg(clock):
    _, c = client(clock)
    with c.websocket_connect("/video?fps=30") as ws:
        assert ws.receive_bytes()[:2] == b"\xff\xd8"
