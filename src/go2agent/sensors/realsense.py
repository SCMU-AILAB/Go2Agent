"""Aligned D435i capture, with a system-Python bridge for aarch64 bindings."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from go2agent.core.types import Frame
from go2agent.sensors.base import Camera

logger = logging.getLogger(__name__)


class RealSenseCamera(Camera):
    """Capture RGB and aligned float32 depth; rotate both together if requested.

    If native bindings cannot import, run the standalone Python 3.8-compatible
    worker with GO2_REALSENSE_PYTHON (default /usr/bin/python3). An explicit
    bridge_python always selects the worker. That interpreter needs numpy and
    pyrealsense2; raw local pipes preserve the complete depth image.
    """

    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        frame_timeout_ms: int = 1000,
        rgb_rotation_deg: int = 0,
        bridge_python: str | None = None,
    ) -> None:
        if min(width, height, fps, frame_timeout_ms) <= 0:
            raise ValueError("Camera dimensions, fps, and timeout must be positive")
        if rgb_rotation_deg not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_timeout_ms = frame_timeout_ms
        self.rgb_rotation_deg = rgb_rotation_deg
        self._bridge_python = bridge_python or os.getenv("GO2_REALSENSE_PYTHON")
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.close()
        self._stop.clear()
        self._latest = None
        self._error = None
        if self._bridge_python is not None:
            self._start_worker(self._bridge_python)
            return
        try:
            import pyrealsense2 as rs
        except (ImportError, OSError) as exc:
            logger.warning("Native RealSense bindings unavailable (%s); using system Python", exc)
            self._start_worker("/usr/bin/python3")
            return
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        started = False
        try:
            profile = pipeline.start(config)
            started = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
        except Exception:
            if started:
                pipeline.stop()
            raise
        self._thread = threading.Thread(
            target=self._capture_native,
            args=(pipeline, align, scale),
            name="realsense-camera",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            else:
                process.wait()
        if self._thread is not None:
            self._thread.join(timeout=self.frame_timeout_ms / 1000 + 2)
            if self._thread.is_alive():
                raise RuntimeError("RealSense capture thread did not stop")
            self._thread = None
        if process is not None and process.stdout is not None:
            process.stdout.close()
        self._process = None

    def latest(self) -> Frame | None:
        return self._latest

    def _publish(self, rgb: np.ndarray, depth: np.ndarray, stamp: float) -> None:
        if self.rgb_rotation_deg:
            turns = -(self.rgb_rotation_deg // 90)
            rgb = np.rot90(rgb, turns).copy()
            depth = np.rot90(depth, turns).copy()
        self._seq += 1
        self._latest = Frame(seq=self._seq, stamp=stamp, rgb=rgb, depth=depth)

    def _capture_native(self, pipeline: Any, align: Any, scale: float) -> None:
        try:
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(self.frame_timeout_ms)
                stamp = time.monotonic()
                aligned = align.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not color or not depth:
                    continue
                rgb = np.asanyarray(color.get_data())[..., ::-1].copy()
                meters = np.asanyarray(depth.get_data()).astype(np.float32) * np.float32(scale)
                self._publish(rgb, meters, stamp)
        except Exception as exc:
            self._error = exc
            if not self._stop.is_set():
                logger.exception("RealSense capture failed")
        finally:
            try:
                pipeline.stop()
            except Exception:
                logger.exception("Could not stop RealSense pipeline")

    def _start_worker(self, python: str) -> None:
        command = [
            python,
            "-u",
            str(Path(__file__).with_name("_realsense_worker.py")),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            str(self.fps),
            "--frame-timeout-ms",
            str(self.frame_timeout_ms),
        ]
        if self.serial:
            command.extend(("--serial", self.serial))
        self._ready.clear()
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        self._thread = threading.Thread(target=self._capture_worker, name="realsense-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self.close()
            raise TimeoutError("RealSense worker did not initialize within 10 seconds")
        if self._error is not None:
            error = self._error
            self.close()
            raise RuntimeError(f"RealSense worker failed using {python}: {error}") from error

    def _capture_worker(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stream = self._process.stdout
        try:
            while not self._stop.is_set():
                line = stream.readline()
                if not line:
                    raise RuntimeError("RealSense worker exited")
                message = json.loads(line)
                if message["type"] == "ready":
                    self._ready.set()
                elif message["type"] == "error":
                    raise RuntimeError(message["message"])
                elif message["type"] == "frame":
                    h, w = int(message["height"]), int(message["width"])
                    if (w, h) != (self.width, self.height):
                        raise ValueError("RealSense worker returned unexpected frame dimensions")
                    rgb = (
                        np.frombuffer(_read_exact(stream, h * w * 3), dtype=np.uint8).reshape(h, w, 3).copy()
                    )
                    depth = np.frombuffer(_read_exact(stream, h * w * 4), dtype="<f4").reshape(h, w).copy()
                    self._publish(rgb, depth, float(message["stamp"]))
                else:
                    raise ValueError("Unknown RealSense worker message")
        except Exception as exc:
            self._error = exc
            if not self._stop.is_set():
                logger.exception("RealSense worker capture failed")
        finally:
            self._ready.set()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise RuntimeError("RealSense worker closed during a frame")
        parts.extend(part)
    return bytes(parts)
