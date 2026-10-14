"""Per-connection YOLO/ByteTrack service with a latest-frame-only input queue."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from go2agent.core.types import Box
from go2agent.perception.protocol import AddBox, AddBoxResult, FrameHeader, TrackMsg, TracksMsg

logger = logging.getLogger(__name__)
_yolo_lock = threading.Lock()


class Detector(Protocol):
    """One instance per socket; return normalized boxes for all detected classes."""

    def track(self, bgr: np.ndarray) -> Sequence[TrackMsg]: ...


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    """Own a model and ByteTrack state; serialize YOLO's shared ID allocator."""

    def __init__(self, model: str, classes: Sequence[str], device: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model)
        unknown = set(classes) - set(self._model.names.values())
        if unknown:
            raise ValueError(f"Unknown model classes: {', '.join(sorted(unknown))}")
        self._device = device or select_device()
        self._track_counter = 0

    def track(self, bgr: np.ndarray) -> Sequence[TrackMsg]:
        from ultralytics.trackers.basetrack import BaseTrack

        # Keep all classes available for AddBox matching; filter only the public output.
        # ByteTrack resets a process-global counter when a new tracker starts.
        # Preserve it per connection so another client's first frame cannot reuse
        # IDs already assigned in this connection.
        with _yolo_lock:
            previous_counter = BaseTrack._count
            BaseTrack._count = self._track_counter
            try:
                result = list(
                    self._model.track(
                        bgr, persist=True, tracker="bytetrack.yaml", device=self._device, verbose=False
                    )
                )[0]
            finally:
                self._track_counter = BaseTrack._count
                BaseTrack._count = previous_counter
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        return [
            TrackMsg(
                id=int(track_id),
                label=result.names[int(class_id)],
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(score),
            )
            for track_id, class_id, box, score in zip(
                _to_list(boxes.id),
                _to_list(boxes.cls),
                _to_list(boxes.xyxyn),
                _to_list(boxes.conf),
                strict=True,
            )
        ]


def _to_list(values: Any) -> list[Any]:
    """Ultralytics box fields are torch tensors (or numpy arrays on some backends)."""
    return values.tolist() if isinstance(values, np.ndarray) else values.cpu().tolist()


def _iou(a: Box, b: Box) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def _opencv_tracker(bgr: np.ndarray, box: Box) -> Any | None:
    height, width = bgr.shape[:2]
    x, y = int(box[0] * width), int(box[1] * height)
    bounds = (x, y, max(1, int(box[2] * width) - x), max(1, int(box[3] * height) - y))
    for namespace in (cv2, getattr(cv2, "legacy", None)):
        for name in ("TrackerKCF_create", "TrackerCSRT_create"):
            factory = getattr(namespace, name, None)
            if factory is not None:
                try:
                    tracker = factory()
                    if tracker.init(bgr, bounds) is not False:
                        return tracker
                except cv2.error:
                    logger.debug("OpenCV tracker initialization failed", exc_info=True)
    return None


@dataclass
class _ManualTrack:
    item: TrackMsg
    last_seen: float
    tracker: Any | None


class _TrackingState:
    """Simple AddBox tracking: greedy IoU, then optional KCF/CSRT, then 1s grace.

    This is not ReID: occlusion, fast motion, overlapping objects, or boxes on old
    frames can lose or switch identity. Without OpenCV contrib, an unmatched box
    stays fixed for at most one second. Manual labels survive detector matches.
    """

    def __init__(self, detector: Detector, classes: Sequence[str]) -> None:
        self.detector = detector
        self.classes = set(classes)
        self._next_id = 1
        self._detector_ids: dict[int, int] = {}
        self._manual: dict[int, _ManualTrack] = {}

    def _allocate_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def add_box(self, request: AddBox, bgr: np.ndarray | None) -> AddBoxResult:
        box = request.box
        if bgr is None:
            return self._refuse(request, "Frame is no longer in the recent-frame cache")
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or not (
            box[0] < box[2] and box[1] < box[3]
        ):
            return self._refuse(request, "Box must be normalized xyxy with positive width and height")
        track_id = self._allocate_id()
        self._manual[track_id] = _ManualTrack(
            TrackMsg(id=track_id, label=request.label, box=box, score=1.0),
            time.monotonic(),
            _opencv_tracker(bgr, box),
        )
        return AddBoxResult(request_id=request.request_id, track_id=track_id)

    @staticmethod
    def _refuse(request: AddBox, error: str) -> AddBoxResult:
        return AddBoxResult(request_id=request.request_id, track_id=None, error=error)

    def process(self, header: FrameHeader, bgr: np.ndarray) -> TracksMsg:
        started = time.monotonic()
        detections = list(self.detector.track(bgr))
        now = time.monotonic()
        used: set[int] = set()
        items: list[TrackMsg] = []
        for track_id, manual in list(self._manual.items()):
            candidates = [i for i in range(len(detections)) if i not in used]
            match = max(candidates, key=lambda i: _iou(manual.item.box, detections[i].box), default=None)
            if match is not None and _iou(manual.item.box, detections[match].box) >= 0.3:
                detected = detections[match]
                used.add(match)
                manual.item = manual.item.model_copy(update={"box": detected.box, "score": detected.score})
                manual.last_seen = now
                manual.tracker = _opencv_tracker(bgr, detected.box)
            elif manual.tracker is not None:
                try:
                    ok, (x, y, width, height) = manual.tracker.update(bgr)
                    if ok and width > 0 and height > 0:
                        h, w = bgr.shape[:2]
                        box = tuple(
                            max(0.0, min(1.0, v)) for v in (x / w, y / h, (x + width) / w, (y + height) / h)
                        )
                        manual.item = manual.item.model_copy(update={"box": box, "score": 0.5})
                        manual.last_seen = now
                except cv2.error:
                    manual.tracker = None
            if now - manual.last_seen >= 1.0:
                del self._manual[track_id]
            else:
                items.append(manual.item)
        for index, detected in enumerate(detections):
            if index in used or detected.label not in self.classes:
                continue
            if detected.id not in self._detector_ids:
                self._detector_ids[detected.id] = self._allocate_id()
            items.append(detected.model_copy(update={"id": self._detector_ids[detected.id]}))
        return TracksMsg(
            seq=header.seq,
            stamp=header.stamp,
            items=items,
            infer_ms=(time.monotonic() - started) * 1000,
        )


def _decode_jpeg(payload: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Invalid JPEG frame")
    return bgr


class _Session:
    def __init__(self, socket: WebSocket, factory: Callable[[], Detector], classes: Sequence[str]) -> None:
        self.socket = socket
        self.factory = factory
        self.classes = classes
        self.recent: deque[tuple[FrameHeader, np.ndarray]] = deque(maxlen=10)
        self.pending: tuple[FrameHeader, np.ndarray] | None = None
        self.commands: deque[tuple[AddBox, np.ndarray | None]] = deque()
        self.ready = asyncio.Event()

    async def receive(self) -> None:
        header: FrameHeader | None = None
        while True:
            message = await self.socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("text") is not None:
                payload = json.loads(message["text"])
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if payload.get("type") == "frame" and header is None:
                    header = FrameHeader.model_validate(payload)
                elif payload.get("type") == "add_box":
                    request = AddBox.model_validate(payload)
                    image = next(
                        (bgr for h, bgr in reversed(self.recent) if h.seq == request.frame_seq), None
                    )
                    self.commands.append((request, image))
                    self.ready.set()
                else:
                    raise ValueError("Expected a frame header followed by JPEG, or AddBox")
            elif message.get("bytes") is not None and header is not None:
                bgr = await asyncio.to_thread(_decode_jpeg, message["bytes"])
                self.pending = (header, bgr)
                self.recent.append(self.pending)
                header = None
                self.ready.set()
            else:
                raise ValueError("JPEG requires a preceding frame header")

    async def process(self) -> None:
        detector = await asyncio.to_thread(self.factory)
        state = _TrackingState(detector, self.classes)
        while True:
            await self.ready.wait()
            self.ready.clear()
            while self.commands:
                request, bgr = self.commands.popleft()
                result = await asyncio.to_thread(state.add_box, request, bgr)
                await self.socket.send_text(result.model_dump_json())
            if self.pending is not None:
                header, bgr = self.pending
                self.pending = None
                tracks = await asyncio.to_thread(state.process, header, bgr)
                await self.socket.send_text(tracks.model_dump_json())

    async def run(self) -> None:
        tasks = [asyncio.create_task(self.receive()), asyncio.create_task(self.process())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except (ValueError, cv2.error) as exc:
            logger.info("Invalid perception message: %s", exc)
            await self.socket.close(code=1008, reason="Invalid perception message")
        except Exception:
            logger.exception("Perception connection failed")
            await self.socket.close(code=1011, reason="Perception failed")
        finally:
            for task in tasks:
                task.cancel()
            # Starlette may cancel the entire request scope during disconnect.
            with anyio.CancelScope(shield=True):
                await asyncio.gather(*tasks, return_exceptions=True)


def create_app(
    model: str = "yolo11s.pt",
    classes: Sequence[str] = ("person",),
    *,
    device: str | None = None,
    detector_factory: Callable[[], Detector] | None = None,
) -> FastAPI:
    """Create the service; inject a detector factory to test without model weights."""
    app = FastAPI(title="Go2Agent perception")
    selected_classes = tuple(classes)
    factory = detector_factory or (lambda: YoloDetector(model, selected_classes, device))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/track")
    async def track(socket: WebSocket) -> None:
        await socket.accept()
        await _Session(socket, factory, selected_classes).run()

    return app


app = create_app()
