"""Go2 built-in front camera (VideoClient JPEG snapshots): level, wide (~116 x 84 deg), no depth.

The SDK returns the newest JPEG on every call (~11 ms); a thread polls it at `fps`, skips repeats and
decodes at half resolution (960 x 540) straight from the JPEG, cheaper than decoding 1080p and resizing.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from go2agent.behaviors.geometry import CameraModel
from go2agent.core.types import Frame
from go2agent.sensors.base import Camera

log = logging.getLogger(__name__)

# Estimated on the robot from a LiDAR-located cone (45 deg right at 81% of the image width), pinhole model.
GO2_FRONT_MODEL = CameraModel(hfov=math.radians(116.0), vfov=math.radians(84.0))


class Go2FrontCamera(Camera):
    def __init__(
        self,
        *,
        fps: float = 15.0,
        network_interface: str = "eth0",
        owns_channel: bool = False,
        client_factory: Callable[[], Any] | None = None,
        channel: Any | None = None,
    ) -> None:
        """`owns_channel=False` reuses the DDS channel the Go2 driver initialized (the usual case);
        True initializes and releases it here (front camera with a simulated robot)."""
        self.period = 1.0 / fps
        self.network_interface = network_interface
        self.owns_channel = owns_channel
        self._client_factory = client_factory
        self._channel = channel
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._client_factory is None:
            self._channel = importlib.import_module("unitree_sdk2_cpp.channel")
            self._client_factory = importlib.import_module("unitree_sdk2_cpp.robot.go2").VideoClient
        if self.owns_channel and self._channel is not None:
            self._channel.initialize(0, self.network_interface)
        client = self._client_factory()
        client.set_timeout(1.0)
        client.init()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(client,), name="go2-front-camera", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.owns_channel and self._channel is not None:
            self._channel.release()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def _run(self, client: Any) -> None:
        last: bytes | None = None
        failures = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                status, data = client.get_image_sample()
                if status != 0 or not data:
                    raise RuntimeError(f"get_image_sample status {status}")
                jpeg = bytes(data)
                if jpeg != last:
                    last = jpeg
                    self._publish(jpeg, t0)
                failures = 0
            except Exception as e:
                failures += 1
                if failures == 1 or failures % 50 == 0:
                    log.warning("Go2 front camera: %s (%d consecutive failures)", e, failures)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def _publish(self, jpeg: bytes, stamp: float) -> None:
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None:
            log.warning("Go2 front camera: undecodable JPEG (%d bytes)", len(jpeg))
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._seq += 1
            self._latest = Frame(seq=self._seq, stamp=stamp, rgb=rgb, depth=None)
