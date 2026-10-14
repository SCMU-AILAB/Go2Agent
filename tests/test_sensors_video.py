"""Threaded camera tests using real video files and simulated RealSense bindings."""

import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from go2agent.sensors.realsense import RealSenseCamera
from go2agent.sensors.video import VideoCamera


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Camera did not produce the expected frame")
        time.sleep(0.005)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "colors.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for bgr in ((0, 0, 255), (0, 255, 0), (255, 0, 0)):
        writer.write(np.full((48, 64, 3), bgr, dtype=np.uint8))
    writer.release()
    return path


def test_video_native_fps_rgb_depth_and_eof(video):
    camera = VideoCamera(video)
    assert camera.latest() is None
    camera.start()
    try:
        wait_for(lambda: camera.latest() is not None)
        first = camera.latest()
        assert first.seq == 1 and first.size == (64, 48)
        assert first.rgb.dtype == np.uint8 and first.depth is None
        assert first.rgb[0, 0, 0] > 250 and first.rgb[0, 0, 2] < 5
        wait_for(lambda: camera.latest().seq == 2)
        second = camera.latest()
        assert 0.07 < second.stamp - first.stamp < 0.3
        wait_for(lambda: not camera._thread.is_alive())
        last = camera.latest()
        assert last.seq == 3 and last.rgb[0, 0, 2] > 250
    finally:
        camera.close()
    assert camera.latest() is last
    camera.close()


def test_video_loop_and_restart(video):
    camera = VideoCamera(video, fps=30, loop=True)
    camera.start()
    thread = camera._thread
    camera.start()
    assert camera._thread is thread
    try:
        wait_for(lambda: camera.latest() is not None and camera.latest().seq >= 5)
        previous = camera.latest().seq
    finally:
        camera.close()
    camera.start()
    try:
        wait_for(lambda: camera.latest() is not None)
        assert camera.latest().seq > previous
    finally:
        camera.close()


def test_video_fps_override_and_interruptible_close(video):
    camera = VideoCamera(video, fps=0.1, loop=True)
    camera.start()
    try:
        wait_for(lambda: camera.latest() is not None)
        time.sleep(0.05)
        assert camera.latest().seq == 1
    finally:
        started = time.monotonic()
        camera.close()
        assert time.monotonic() - started < 0.5


def test_video_device_index_and_unknown_fps(monkeypatch):
    released = []

    class Device:
        def __init__(self, source):
            assert source == 3

        def isOpened(self):
            return True

        def get(self, prop):
            return 0

        def read(self):
            return True, np.zeros((20, 30, 3), np.uint8)

        def release(self):
            released.append(True)

    monkeypatch.setattr(cv2, "VideoCapture", Device)
    camera = VideoCamera(3)
    camera.start()
    try:
        wait_for(lambda: camera.latest() is not None and camera.latest().seq >= 2)
        assert camera.latest().size == (30, 20)
    finally:
        camera.close()
    assert released == [True]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps(fps):
    with pytest.raises(ValueError):
        VideoCamera(0, fps=fps)


def test_missing_video(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        VideoCamera(tmp_path / "missing.avi").start()


def test_realsense_native_alignment_scale_rotation_and_owned_buffers(monkeypatch):
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 2] = 255
    raw_depth = np.array([[0, 100, 200], [300, 400, 500]], np.uint16)
    stopped, configured, aligned = [], [], []
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: bgr),
        get_depth_frame=lambda: SimpleNamespace(get_data=lambda: raw_depth),
    )

    class Pipeline:
        def start(self, config):
            return SimpleNamespace(
                get_device=lambda: SimpleNamespace(
                    first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.002)
                )
            )

        def wait_for_frames(self, timeout):
            time.sleep(0.01)
            return frames

        def stop(self):
            stopped.append(True)

    def align_to_color(stream):
        assert stream == "color"

        def process(value):
            aligned.append(value)
            return value

        return SimpleNamespace(process=process)

    rs = SimpleNamespace(
        pipeline=Pipeline,
        config=lambda: SimpleNamespace(enable_stream=lambda *args: configured.append(args)),
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", bgr8="bgr8"),
        align=align_to_color,
    )
    monkeypatch.delenv("GO2_REALSENSE_PYTHON", raising=False)
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = RealSenseCamera(width=3, height=2, rgb_rotation_deg=90)
    camera.start()
    try:
        wait_for(lambda: camera.latest() is not None)
        result = camera.latest()
    finally:
        camera.close()
    assert aligned and stopped == [True]
    assert configured == [("depth", 3, 2, "z16", 30), ("color", 3, 2, "bgr8", 30)]
    assert result.rgb.shape == (3, 2, 3) and result.depth.shape == (3, 2)
    assert result.depth.dtype == np.float32
    np.testing.assert_allclose(result.depth, np.rot90(raw_depth.astype(np.float32) * 0.002, -1))
    bgr.fill(0)
    raw_depth.fill(0)
    assert result.rgb[0, 0, 0] == 255 and result.depth.max() == 1.0


def test_realsense_worker_transports_full_depth(tmp_path, monkeypatch):
    # Run the real standalone worker in another interpreter with fake hardware.
    (tmp_path / "pyrealsense2.py").write_text(
        """
import time
from types import SimpleNamespace
import numpy as np
stream = SimpleNamespace(depth=0, color=1)
format = SimpleNamespace(z16=0, bgr8=1)
class config:
    def enable_stream(self, *args): pass
class pipeline:
    def start(self, cfg):
        return SimpleNamespace(get_device=lambda: SimpleNamespace(
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)))
    def wait_for_frames(self, timeout):
        time.sleep(0.01)
        bgr = np.zeros((2, 3, 3), np.uint8)
        bgr[..., 2] = 255
        depth = np.full((2, 3), 1234, np.uint16)
        return SimpleNamespace(
            get_color_frame=lambda: SimpleNamespace(get_data=lambda: bgr),
            get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth))
    def stop(self): pass
def align(stream):
    return SimpleNamespace(process=lambda frames: frames)
"""
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    camera = RealSenseCamera(width=3, height=2, bridge_python=sys.executable)
    camera.start()
    process = camera._process
    try:
        wait_for(lambda: camera.latest() is not None)
        result = camera.latest()
        assert result.rgb.shape == (2, 3, 3) and result.rgb[0, 0, 0] == 255
        assert result.depth.dtype == np.float32
        np.testing.assert_allclose(result.depth, 1.234)
        assert time.monotonic() - result.stamp < 0.5
    finally:
        camera.close()
    assert process.poll() is not None and camera._thread is None
