from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from perception._realsense_worker import rotate_bgr_image
from perception.realsense_bridge import RealSenseBridge

FAKE_WORKER = """
import json
import sys

print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "capture":
        print(json.dumps({
            "type": "observation",
            "result": {
                "observed_at_s": 12.5,
                "person_count": 1,
                "nearest_person_distance_m": 1.25,
                "confidence": 0.9,
                "source": "realsense:test",
                "nearest_obstacle_distance_m": 0.75,
                "rgb_jpeg_base64": "anBlZw==",
            },
        }), flush=True)
    elif request.get("command") == "close":
        print(json.dumps({"type": "closed"}), flush=True)
        break
"""


class RealSenseBridgeTests(unittest.TestCase):
    def test_worker_rotates_before_jpeg_encoding(self) -> None:
        calls: list[tuple[object, object]] = []
        cv2 = SimpleNamespace(
            ROTATE_90_CLOCKWISE="cw",
            ROTATE_180="half",
            ROTATE_90_COUNTERCLOCKWISE="ccw",
            rotate=lambda image, code: calls.append((image, code)) or code,
        )
        image = object()

        self.assertIs(rotate_bgr_image(cv2, image, 0), image)
        self.assertEqual(rotate_bgr_image(cv2, image, 90), "cw")
        self.assertEqual(rotate_bgr_image(cv2, image, 180), "half")
        self.assertEqual(rotate_bgr_image(cv2, image, 270), "ccw")
        self.assertEqual(
            calls,
            [(image, "cw"), (image, "half"), (image, "ccw")],
        )

    def test_persistent_worker_returns_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory, "fake_worker.py")
            worker.write_text(FAKE_WORKER, encoding="utf-8")
            bridge = RealSenseBridge(
                serial=None,
                width=640,
                height=480,
                fps=30,
                frame_timeout_ms=1000,
                min_score=0.5,
                max_distance_m=4.0,
                rgb_rotation_deg=180,
                python_executable=sys.executable,
                worker_path=worker,
            )

            bridge.open()
            first = bridge.capture()
            second = bridge.capture()
            frame = bridge.capture_frame()
            bridge.close()

        self.assertEqual(first.person_count, 1)
        self.assertEqual(first.nearest_person_distance_m, 1.25)
        self.assertEqual(first.source, "realsense:test")
        self.assertEqual(second, first)
        self.assertEqual(frame.rgb, b"jpeg")
        self.assertEqual(frame.nearest_obstacle_distance_m, 0.75)
        self.assertEqual(frame.observation, first)
        self.assertEqual(bridge.rgb_rotation_deg, 180)
        self.assertFalse(bridge.opened)


if __name__ == "__main__":
    unittest.main()
