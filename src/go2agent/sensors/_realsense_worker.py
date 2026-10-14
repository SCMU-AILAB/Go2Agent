"""Standalone Python 3.8-compatible capture worker; no go2agent imports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)


def _send(stream: BinaryIO, message: dict[str, Any]) -> None:
    stream.write(json.dumps(message).encode("utf-8") + b"\n")
    stream.flush()


def _terminate(signum: int, frame: Any) -> None:
    raise SystemExit(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    args = parser.parse_args()
    # SDK diagnostics must not corrupt the binary stdout protocol.
    transport = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    signal.signal(signal.SIGTERM, _terminate)
    pipeline = None
    started = False
    try:
        import numpy as np
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        profile = pipeline.start(config)
        started = True
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        _send(transport, {"type": "ready"})
        while True:
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            stamp = time.monotonic()
            aligned = align.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth:
                continue
            rgb = np.asanyarray(color.get_data())[..., ::-1].copy()
            meters = np.asanyarray(depth.get_data()).astype(np.float32) * np.float32(scale)
            h, w = rgb.shape[:2]
            _send(transport, {"type": "frame", "stamp": stamp, "width": w, "height": h})
            transport.write(rgb.tobytes())
            transport.write(meters.astype("<f4", copy=False).tobytes())
            transport.flush()
    except Exception as exc:
        logger.exception("RealSense worker failed")
        with suppress(BrokenPipeError, OSError):
            _send(transport, {"type": "error", "message": str(exc)})
        return 1
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                logger.exception("Could not stop RealSense pipeline")
        transport.close()


if __name__ == "__main__":
    sys.exit(main())
