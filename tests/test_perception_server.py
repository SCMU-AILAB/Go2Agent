"""Protocol and tracking tests without YOLO weights or GPU dependencies."""

import asyncio
import json
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from go2agent.perception.protocol import AddBox, AddBoxResult, FrameHeader, TrackMsg, TracksMsg
from go2agent.server import perception


class FakeDetector:
    def track(self, bgr):
        return [
            TrackMsg(id=43, label="person", box=(0.1, 0.2, 0.4, 0.8), score=0.9),
            TrackMsg(id=70, label="cup", box=(0.6, 0.2, 0.8, 0.4), score=0.8),
        ]


class EmptyDetector:
    def track(self, bgr: np.ndarray) -> list[TrackMsg]:
        return []


def send_frame(socket, seq, stamp=10.0):
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), seq, dtype=np.uint8))
    assert ok
    socket.send_text(FrameHeader(seq=seq, stamp=stamp).model_dump_json())
    socket.send_bytes(jpeg.tobytes())


def test_health_tracking_ids_and_per_connection_reset():
    instances = []

    def factory():
        instance = FakeDetector()
        instances.append(instance)
        return instance

    with TestClient(perception.create_app(detector_factory=factory)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        with client.websocket_connect("/track") as ws:
            send_frame(ws, 1, 123.5)
            first = TracksMsg.model_validate_json(ws.receive_text())
            assert (first.seq, first.stamp) == (1, 123.5)
            assert first.infer_ms >= 0
            assert [(t.id, t.label) for t in first.items] == [(1, "person")]
            send_frame(ws, 2)
            assert TracksMsg.model_validate_json(ws.receive_text()).items == first.items
        with client.websocket_connect("/track") as ws:
            send_frame(ws, 3)
            assert TracksMsg.model_validate_json(ws.receive_text()).items[0].id == 1
    assert len(instances) == 2
    assert instances[0] is not instances[1]


def test_add_box_matches_any_class_and_recent_cache(monkeypatch):
    monkeypatch.setattr(perception, "_opencv_tracker", lambda *_: None)
    with (
        TestClient(perception.create_app(detector_factory=FakeDetector)) as client,
        client.websocket_connect("/track") as ws,
    ):
        for seq in range(1, 12):
            send_frame(ws, seq)
            ws.receive_text()
        request = AddBox(request_id=1, frame_seq=1, box=(0.6, 0.2, 0.8, 0.4), label="my mug")
        ws.send_text(request.model_dump_json())
        error = AddBoxResult.model_validate_json(ws.receive_text())
        assert error.track_id is None and error.error
        # Frame 2 is the oldest of the ten retained frames.
        ws.send_text(request.model_copy(update={"request_id": 2, "frame_seq": 2}).model_dump_json())
        added = AddBoxResult.model_validate_json(ws.receive_text())
        assert added.request_id == 2 and added.track_id == 2
        send_frame(ws, 12)
        tracks = TracksMsg.model_validate_json(ws.receive_text())
        assert {(t.id, t.label) for t in tracks.items} == {(1, "person"), (2, "my mug")}
        assert next(t for t in tracks.items if t.id == 2).score == 0.8
        invalid = request.model_copy(update={"request_id": 3, "frame_seq": 12, "box": (0.8, 0.1, 0.2, 0.3)})
        ws.send_text(invalid.model_dump_json())
        assert AddBoxResult.model_validate_json(ws.receive_text()).error


def test_inference_drops_stale_frames_without_blocking_socket(monkeypatch):
    entered, release, decoded_latest = threading.Event(), threading.Event(), threading.Event()
    processed = []
    decode = perception._decode_jpeg

    class SlowDetector:
        def track(self, bgr):
            value = int(bgr[0, 0, 0])
            processed.append(value)
            if value == 1:
                entered.set()
                assert release.wait(5)
            return []

    def decode_frame(data):
        bgr = decode(data)
        if bgr[0, 0, 0] == 10:
            decoded_latest.set()
        return bgr

    monkeypatch.setattr(perception, "_decode_jpeg", decode_frame)
    with (
        TestClient(perception.create_app(detector_factory=SlowDetector)) as client,
        client.websocket_connect("/track") as ws,
    ):
        try:
            send_frame(ws, 1)
            assert entered.wait(2)
            for seq in range(2, 11):
                send_frame(ws, seq)
            assert decoded_latest.wait(2), "Socket stopped reading while inference ran"
            # Let the receive coroutine store the decoded last frame.
            assert client.portal is not None
            client.portal.call(asyncio.sleep, 0.02)
            release.set()
            assert TracksMsg.model_validate_json(ws.receive_text()).seq == 1
            assert TracksMsg.model_validate_json(ws.receive_text()).seq == 10
        finally:
            release.set()
    assert processed == [1, 10]


def test_manual_track_expiry_and_opencv_fallback(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(perception, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(perception, "_opencv_tracker", lambda *_: None)
    state = perception._TrackingState(EmptyDetector(), ("person",))
    bgr = np.zeros((100, 100, 3), np.uint8)
    request = AddBox(request_id=1, frame_seq=1, box=(0.1, 0.2, 0.4, 0.8), label="object")
    state.add_box(request, bgr)
    now[0] = 10.9
    assert len(state.process(FrameHeader(seq=2, stamp=10.9), bgr).items) == 1
    now[0] = 11.1
    assert state.process(FrameHeader(seq=3, stamp=11.1), bgr).items == []
    tracker = SimpleNamespace(update=lambda _: (True, (20, 30, 20, 40)))
    monkeypatch.setattr(perception, "_opencv_tracker", lambda *_: tracker)
    state.add_box(request, bgr)
    now[0] = 13.0
    assert state.process(FrameHeader(seq=4, stamp=13.0), bgr).items[0].box == (0.2, 0.3, 0.4, 0.7)


@pytest.mark.parametrize("payload", [b"not a JPEG", json.dumps({"type": "other"}), "[]"])
def test_invalid_messages_close_connection(payload):
    with (
        TestClient(perception.create_app(detector_factory=FakeDetector)) as client,
        client.websocket_connect("/track") as ws,
    ):
        if isinstance(payload, bytes):
            ws.send_text(FrameHeader(seq=1, stamp=0).model_dump_json())
            ws.send_bytes(payload)
        else:
            ws.send_text(payload)
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_text()
        assert error.value.code == 1008


def test_yolo_adapter_uses_persistent_bytetrack(monkeypatch):
    import sys

    calls = []

    def tensor(values):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))

    boxes = SimpleNamespace(
        id=tensor([7]), cls=tensor([0]), xyxyn=tensor([[0.1, 0.2, 0.3, 0.4]]), conf=tensor([0.9])
    )
    base_track = SimpleNamespace(_count=42)

    class Model:
        names = {0: "person", 1: "bicycle"}

        def __init__(self, model):
            assert model == "fake.pt"
            self.started = False

        def track(self, image, **kwargs):
            if not self.started:
                base_track._count = 0
                self.started = True
            base_track._count += 1
            boxes.id = tensor([base_track._count])
            calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes, names=self.names)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(sys.modules, "ultralytics.trackers.basetrack", SimpleNamespace(BaseTrack=base_track))
    detector = perception.YoloDetector("fake.pt", ["person"], "cpu")
    bgr = np.zeros((10, 10, 3), np.uint8)
    assert detector.track(bgr)[0].id == 1
    assert calls[0] == {"persist": True, "tracker": "bytetrack.yaml", "device": "cpu", "verbose": False}
    other = perception.YoloDetector("fake.pt", ["person"], "cpu")
    assert other.track(bgr)[0].id == 1
    assert detector.track(bgr)[0].id == 2
    assert base_track._count == 42
    with pytest.raises(ValueError, match="Unknown model classes"):
        perception.YoloDetector("fake.pt", ["dragon"], "cpu")


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_preference(monkeypatch, cuda, mps, expected):
    import sys

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert perception.select_device() == expected


def test_configurable_class_filter():
    with (
        TestClient(perception.create_app(classes=["cup"], detector_factory=FakeDetector)) as client,
        client.websocket_connect("/track") as ws,
    ):
        send_frame(ws, 1)
        tracks = TracksMsg.model_validate_json(ws.receive_text())
        assert [item.label for item in tracks.items] == ["cup"]
