"""HTTP + WebSocket API for the Swift console app. Trusted LAN only: there is no authentication.

REST
  GET  /status            runtime snapshot (task, executor, tracks, last decision, ...)
  POST /task   {"text"}   set the task (empty text clears it); releases an e-stop
  POST /hear   {"text"}   inject an utterance as if heard by the microphone
  POST /estop             latch e-stop
  POST /resume            release e-stop without changing the task
  GET  /frame.jpg         newest camera frame with track marks
WebSocket
  /events                 first message {"kind":"status","data":<status>}, then every timeline record
  /video?fps=10           binary JPEG frames with track marks
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from go2agent.brain.marks import draw_marks, jpeg_bytes, resize_long_side
from go2agent.runtime import Runtime


class TextBody(BaseModel):
    text: str


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="go2agent")

    def marked_jpeg(long_side: int = 640, quality: int = 75) -> bytes | None:
        frame = rt.latest_frame()
        if frame is None:
            return None
        rgb = resize_long_side(draw_marks(frame.rgb, rt.tracker.latest()), long_side)
        return jpeg_bytes(rgb, quality)

    @app.get("/status")
    def status() -> dict:
        return rt.status()

    @app.post("/task")
    def task(body: TextBody) -> dict:
        rt.set_task(body.text)
        return rt.status()

    @app.post("/hear")
    def hear(body: TextBody) -> dict:
        rt.hear(body.text)
        return {"ok": True}

    @app.post("/estop")
    def estop() -> dict:
        rt.estop("app")
        return rt.status()

    @app.post("/resume")
    def resume() -> dict:
        rt.safety.release()
        return rt.status()

    @app.get("/frame.jpg")
    def frame() -> Response:
        jpg = marked_jpeg()
        if jpg is None:
            return Response(status_code=404)
        return Response(jpg, media_type="image/jpeg")

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        q = rt.subscribe()
        try:
            await ws.send_json({"kind": "status", "data": rt.status()})
            while True:
                rec = await q.get()
                await ws.send_json(rec)
        except WebSocketDisconnect:
            pass
        finally:
            rt.unsubscribe(q)

    @app.websocket("/video")
    async def video(ws: WebSocket, fps: float = 10.0) -> None:
        await ws.accept()
        period = 1.0 / max(1.0, min(fps, 30.0))
        last_seq = -1
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                t0 = time.monotonic()
                frame = rt.latest_frame()
                if frame is not None and frame.seq != last_seq:
                    last_seq = frame.seq
                    jpg = await asyncio.to_thread(marked_jpeg)
                    if jpg:
                        await ws.send_bytes(jpg)
                await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    return app
