"""Non-blocking latest-frame client for the remote perception service."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import suppress

import cv2
from websockets.asyncio.client import ClientConnection, connect

from go2agent.core.types import Box, Frame, Track, Tracks
from go2agent.perception.base import Tracker
from go2agent.perception.protocol import AddBox, AddBoxResult, FrameHeader, TracksMsg

logger = logging.getLogger(__name__)


class RemoteTracker(Tracker):
    """Submit from any thread; start, close, and add_box run on one asyncio loop.

    Health uses the original frame's local monotonic stamp so a late response
    cannot make an old observation look fresh. Reconnection clears old track IDs.
    """

    def __init__(self, url: str, *, jpeg_quality: int = 80, max_side: int | None = 640) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if max_side is not None and max_side <= 0:
            raise ValueError("max_side must be positive or None")
        self.url = url
        self.jpeg_quality = jpeg_quality
        self.max_side = max_side
        self._pending: Frame | None = None
        self._pending_lock = threading.Lock()
        self._pending_ready = asyncio.Event()
        self._ack = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: ClientConnection | None = None
        self._inflight: tuple[int, float] | None = None
        self._latest: Tracks | None = None
        self._received_at = 0.0
        self._request_id = 0
        self._requests: dict[int, asyncio.Future[AddBoxResult]] = {}

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._pending_ready = asyncio.Event()
        self._ack = asyncio.Event()
        self._send_lock = asyncio.Lock()
        with self._pending_lock:
            if self._pending is not None:
                self._pending_ready.set()
        self._task = asyncio.create_task(self._run(), name="remote-tracker")

    async def close(self) -> None:
        task, self._task = self._task, None
        self._loop = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._disconnected()
        with self._pending_lock:
            self._pending = None

    def submit(self, frame: Frame) -> None:
        with self._pending_lock:
            self._pending = frame
        loop = self._loop
        if loop is not None and not loop.is_closed():
            # The loop can close concurrently with a camera's final submit.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._pending_ready.set)

    def latest(self) -> Tracks | None:
        return self._latest

    @property
    def healthy(self) -> bool:
        latest = self._latest
        now = time.monotonic()
        return (
            self._socket is not None
            and latest is not None
            and now - self._received_at < 0.5
            and now - latest.stamp < 0.5
        )

    async def add_box(self, frame_seq: int, box: Box, label: str) -> int:
        socket = self._socket
        if socket is None:
            raise ConnectionError("Perception service is disconnected")
        self._request_id += 1
        request = AddBox(request_id=self._request_id, frame_seq=frame_seq, box=box, label=label)
        future: asyncio.Future[AddBoxResult] = asyncio.get_running_loop().create_future()
        self._requests[request.request_id] = future
        try:
            async with asyncio.timeout(2.0):
                async with self._send_lock:
                    if socket is not self._socket:
                        raise ConnectionError("Perception connection changed")
                    await socket.send(request.model_dump_json())
                result = await future
            if result.error is not None or result.track_id is None:
                raise ValueError(result.error or "Perception service did not return a track id")
            return result.track_id
        finally:
            self._requests.pop(request.request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()  # Consume a disconnect racing a failed send.

    def _encode(self, frame: Frame) -> bytes:
        rgb = frame.rgb
        height, width = rgb.shape[:2]
        if self.max_side is not None and max(height, width) > self.max_side:
            scale = self.max_side / max(height, width)
            rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise ValueError("Could not JPEG-encode frame")
        return jpeg.tobytes()

    async def _send_frames(self, socket: ClientConnection) -> None:
        while True:
            await self._pending_ready.wait()
            if self._inflight is not None:
                remaining = max(0.0, 1.0 - (time.monotonic() - self._inflight[1]))
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._ack.wait(), timeout=remaining)
            with self._pending_lock:
                frame, self._pending = self._pending, None
                self._pending_ready.clear()
            if frame is None:
                continue
            try:
                jpeg = await asyncio.to_thread(self._encode, frame)
            except (ValueError, cv2.error):
                logger.exception("Could not encode perception frame %s", frame.seq)
                continue
            async with self._send_lock:
                self._ack.clear()
                self._inflight = (frame.seq, time.monotonic())
                await socket.send(FrameHeader(seq=frame.seq, stamp=frame.stamp).model_dump_json())
                await socket.send(jpeg)

    async def _receive(self, socket: ClientConnection) -> None:
        async for payload in socket:
            message = json.loads(payload)
            if message.get("type") == "tracks":
                result = TracksMsg.model_validate(message)
                if self._inflight is not None and result.seq == self._inflight[0]:
                    self._inflight = None
                    self._ack.set()
                if self._latest is None or result.stamp >= self._latest.stamp:
                    self._latest = Tracks(
                        frame_seq=result.seq,
                        stamp=result.stamp,
                        items=tuple(
                            Track(id=item.id, label=item.label, box=item.box, score=item.score)
                            for item in result.items
                        ),
                    )
                    self._received_at = time.monotonic()
            elif message.get("type") == "add_box_result":
                result = AddBoxResult.model_validate(message)
                future = self._requests.get(result.request_id)
                if future is not None and not future.done():
                    future.set_result(result)
            else:
                raise ValueError("Unknown perception response")

    def _disconnected(self) -> None:
        self._socket = None
        self._latest = None
        self._inflight = None
        for future in self._requests.values():
            if not future.done():
                future.set_exception(ConnectionError("Perception service disconnected"))

    async def _run(self) -> None:
        backoff = 0.5
        while True:
            connected_at: float | None = None
            try:
                async with connect(self.url, open_timeout=5, close_timeout=1, proxy=None) as socket:
                    connected_at = time.monotonic()
                    self._socket = socket
                    tasks = [
                        asyncio.create_task(self._send_frames(socket)),
                        asyncio.create_task(self._receive(socket)),
                    ]
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            task.result()
                    finally:
                        self._disconnected()
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Perception connection failed: %s", exc)
            finally:
                self._disconnected()
            if connected_at is not None and time.monotonic() - connected_at >= 5.0:
                backoff = 0.5
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
