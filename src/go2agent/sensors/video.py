"""Threaded video-file and OpenCV-device camera."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import cv2

from go2agent.core.types import Frame
from go2agent.sensors.base import Camera

logger = logging.getLogger(__name__)


class VideoCamera(Camera):
    def __init__(self, source: str | Path | int, *, fps: float | None = None, loop: bool = False) -> None:
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps must be positive and finite")
        self.source = str(source) if isinstance(source, Path) else source
        self.fps = fps
        self.loop = loop
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video source {self.source!r}")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = self.fps or (native_fps if math.isfinite(native_fps) and native_fps > 0 else 30.0)
        self._stop.clear()
        self._latest = None
        self._thread = threading.Thread(
            target=self._capture, args=(capture, fps), name="video-camera", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Video camera did not stop; the OpenCV device may be blocked")
            self._thread = None

    def latest(self) -> Frame | None:
        return self._latest

    def _capture(self, capture: cv2.VideoCapture, fps: float) -> None:
        deadline = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, bgr = capture.read()
                if not ok and self.loop and not isinstance(self.source, int):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = capture.read()
                if not ok:
                    return
                self._seq += 1
                self._latest = Frame(
                    seq=self._seq, stamp=time.monotonic(), rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )
                deadline += 1.0 / fps
                now = time.monotonic()
                if deadline < now:
                    deadline = now
                if self._stop.wait(max(0.0, deadline - now)):
                    return
        except Exception:
            logger.exception("Video camera capture failed")
        finally:
            capture.release()
