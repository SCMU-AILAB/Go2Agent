"""Client integration tests over real localhost WebSockets."""

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytest
import uvicorn
from websockets.asyncio.server import serve

from go2agent.core.types import Frame, Tracks
from go2agent.perception.protocol import AddBoxResult, FrameHeader, TrackMsg, TracksMsg
from go2agent.perception.remote import RemoteTracker
from go2agent.server.perception import create_app


def tracks_of(tracker: RemoteTracker) -> Tracks:
    tracks = tracker.latest()
    assert tracks is not None
    return tracks


def frame(seq, *, stamp=None, width=800, height=400):
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[..., 0] = 255
    return Frame(seq=seq, stamp=time.monotonic() if stamp is None else stamp, rgb=rgb)


async def eventually(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def receive_frame(ws):
    header = FrameHeader.model_validate_json(await ws.recv())
    data = await ws.recv()
    assert isinstance(data, bytes)
    return header, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


async def acknowledge(ws, header):
    await ws.send(TracksMsg(seq=header.seq, stamp=header.stamp, items=[], infer_ms=1).model_dump_json())


@asynccontextmanager
async def ws_server(handler):
    async with serve(handler, "127.0.0.1", 0) as server:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/track"


@asynccontextmanager
async def fastapi_server(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", ws="websockets-sansio"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await eventually(lambda: server.started or task.done())
        if task.done():
            task.result()
        yield f"ws://127.0.0.1:{listener.getsockname()[1]}/track"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


async def test_client_with_real_app_and_add_box():
    seen = []

    class Detector:
        def track(self, bgr):
            seen.append(bgr)
            return [TrackMsg(id=10, label="person", box=(0.1, 0.1, 0.5, 0.9), score=0.9)]

    async with fastapi_server(create_app(detector_factory=Detector)) as url:
        tracker = RemoteTracker(url, stale_s=0.5)
        assert not tracker.healthy and tracker.latest() is None
        tracker.submit(frame(1))
        await tracker.start()
        try:
            await eventually(lambda: tracker.latest() is not None)
            result = tracks_of(tracker)
            assert result.frame_seq == 1 and result.items[0].label == "person"
            assert result.items[0].id == 1 and tracker.healthy
            assert seen[0].shape == (320, 640, 3)
            assert seen[0][0, 0, 2] > 250 and seen[0][0, 0, 0] < 5  # RGB -> BGR -> JPEG
            track_id = await tracker.add_box(1, (0.6, 0.2, 0.8, 0.6), "box")
            assert track_id != 1
            tracker.submit(frame(2))
            await eventually(lambda: (r := tracker.latest()) is not None and r.frame_seq == 2)
            added = tracks_of(tracker).get(track_id)
            assert added is not None and added.label == "box"
            with pytest.raises(ValueError, match="recent-frame cache"):
                await tracker.add_box(999, (0, 0, 1, 1), "missing")
            await asyncio.sleep(0.51)
            assert not tracker.healthy
        finally:
            await tracker.close()
        assert tracker.latest() is None and not tracker.healthy


async def test_newest_only_and_matching_acknowledgements():
    headers = asyncio.Queue()
    connections = asyncio.Queue()

    async def handler(ws):
        await connections.put(ws)
        try:
            while True:
                header, _ = await receive_frame(ws)
                await headers.put(header)
        except Exception:
            return

    async with ws_server(handler) as url:
        tracker = RemoteTracker(url)
        await tracker.start()
        try:
            ws = await asyncio.wait_for(connections.get(), 2)
            tracker.submit(frame(1))
            first = await asyncio.wait_for(headers.get(), 2)
            await asyncio.to_thread(lambda: [tracker.submit(frame(seq)) for seq in range(2, 11)])
            await asyncio.sleep(0.1)
            assert headers.empty()
            await acknowledge(ws, first.model_copy(update={"seq": 999}))
            await asyncio.sleep(0.1)
            assert headers.empty(), "Unrelated results must not acknowledge the pending frame"
            await acknowledge(ws, first)
            second = await asyncio.wait_for(headers.get(), 0.5)
            assert second.seq == 10
        finally:
            await tracker.close()


async def test_frame_timeout_and_late_result_does_not_release_new_frame():
    headers = asyncio.Queue()
    connections = asyncio.Queue()

    async def handler(ws):
        await connections.put(ws)
        try:
            while True:
                header, _ = await receive_frame(ws)
                await headers.put((header, time.monotonic()))
        except Exception:
            return

    async with ws_server(handler) as url:
        tracker = RemoteTracker(url)
        await tracker.start()
        try:
            ws = await asyncio.wait_for(connections.get(), 2)
            tracker.submit(frame(1))
            first, first_time = await asyncio.wait_for(headers.get(), 2)
            tracker.submit(frame(2))
            second, second_time = await asyncio.wait_for(headers.get(), 1.5)
            assert second.seq == 2 and second_time - first_time >= 0.9
            tracker.submit(frame(3))
            await acknowledge(ws, first)
            await asyncio.sleep(0.1)
            assert headers.empty()
            assert not tracker.healthy, "Old capture must remain stale even after a recent response"
            await acknowledge(ws, second)
            third, _ = await asyncio.wait_for(headers.get(), 0.5)
            assert third.seq == 3
        finally:
            await tracker.close()


async def test_add_box_timeout_errors_and_out_of_order_replies():
    requests = asyncio.Queue()
    connections = asyncio.Queue()

    async def handler(ws):
        await connections.put(ws)
        async for payload in ws:
            await requests.put(json.loads(payload))

    async with ws_server(handler) as url:
        tracker = RemoteTracker(url)
        await tracker.start()
        try:
            ws = await asyncio.wait_for(connections.get(), 2)
            await eventually(lambda: tracker._socket is not None)
            tasks = [asyncio.create_task(tracker.add_box(1, (0, 0, 1, 1), str(i))) for i in range(2)]
            first, second = await requests.get(), await requests.get()
            for request, track_id in ((second, 20), (first, 10)):
                await ws.send(
                    AddBoxResult(request_id=request["request_id"], track_id=track_id).model_dump_json()
                )
            assert await asyncio.gather(*tasks) == [10, 20]
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                await tracker.add_box(1, (0, 0, 1, 1), "timeout")
            assert 1.9 <= time.monotonic() - started < 2.5
            assert not tracker._requests
        finally:
            await tracker.close()


async def test_reconnect_clears_ids_and_fails_pending_requests():
    connections = asyncio.Queue()
    requests = asyncio.Queue()

    async def handler(ws):
        await connections.put(ws)
        async for payload in ws:
            if isinstance(payload, str) and json.loads(payload)["type"] == "add_box":
                await requests.put(json.loads(payload))

    async with ws_server(handler) as url:
        tracker = RemoteTracker(url)
        await tracker.start()
        try:
            first = await asyncio.wait_for(connections.get(), 2)
            await first.send(TracksMsg(seq=1, stamp=time.monotonic(), items=[], infer_ms=1).model_dump_json())
            await eventually(lambda: tracker.healthy)
            pending = asyncio.create_task(tracker.add_box(1, (0, 0, 1, 1), "object"))
            await requests.get()
            await first.close()
            with pytest.raises(ConnectionError):
                await pending
            assert not tracker.healthy and tracker.latest() is None
            second = await asyncio.wait_for(connections.get(), 2)
            await second.send(
                TracksMsg(seq=2, stamp=time.monotonic(), items=[], infer_ms=1).model_dump_json()
            )
            await eventually(lambda: tracker.healthy)
            assert tracks_of(tracker).frame_seq == 2
        finally:
            await tracker.close()
