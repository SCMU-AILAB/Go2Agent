import threading
import time

import cv2
import numpy as np

from go2agent.sensors.go2_front import Go2FrontCamera


def jpeg(color: tuple[int, int, int], w: int = 64, h: int = 36) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), color, np.uint8))
    assert ok
    return buf.tobytes()


class FakeVideoClient:
    def __init__(self, frames: list[bytes | None]) -> None:
        self.frames = frames
        self.i = 0
        self.lock = threading.Lock()

    def set_timeout(self, timeout: float) -> None: ...
    def init(self) -> None: ...

    def get_image_sample(self) -> tuple[int, list[int]]:
        with self.lock:
            data = self.frames[min(self.i, len(self.frames) - 1)]
            self.i += 1
        return (3104, []) if data is None else (0, list(data))


class FakeChannel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, domain: int, interface: str) -> None:
        self.calls.append(f"init {interface}")

    def release(self) -> None:
        self.calls.append("release")


def wait_for(predicate, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end
        time.sleep(0.01)


def test_decodes_half_resolution_rgb_and_skips_repeats_and_errors():
    red, blue = jpeg((0, 0, 255)), jpeg((255, 0, 0))  # BGR
    client = FakeVideoClient([red, red, None, red, blue])
    cam = Go2FrontCamera(fps=200, client_factory=lambda: client)
    cam.start()
    try:
        wait_for(lambda: (f := cam.latest()) is not None and f.seq == 2)
        frame = cam.latest()
        assert frame is not None and frame.depth is None
        assert frame.rgb.shape == (18, 32, 3)  # decoded at half resolution
        assert frame.rgb[0, 0, 2] > 200 and frame.rgb[0, 0, 0] < 50  # blue, in RGB order
    finally:
        cam.close()


def test_channel_is_only_touched_when_owned():
    shared, owned = FakeChannel(), FakeChannel()
    a = Go2FrontCamera(client_factory=lambda: FakeVideoClient([jpeg((0, 0, 0))]), channel=shared)
    b = Go2FrontCamera(
        client_factory=lambda: FakeVideoClient([jpeg((0, 0, 0))]),
        channel=owned,
        owns_channel=True,
        network_interface="eth9",
    )
    for cam in (a, b):
        cam.start()
        cam.close()
    assert shared.calls == [] and owned.calls == ["init eth9", "release"]
