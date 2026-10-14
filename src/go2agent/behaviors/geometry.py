"""Image-space geometry for target-relative control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from go2agent.core.types import Box, Frame, Track

PERSON_HEIGHT_M = 1.7


@dataclass(frozen=True, slots=True)
class CameraModel:
    hfov: float = math.radians(69.0)  # D435i color
    vfov: float = math.radians(42.0)


def bearing(box: Box, cam: CameraModel) -> float:
    """Horizontal angle to the box center in radians; positive = target is to the left."""
    cx = (box[0] + box[2]) / 2
    return math.atan((0.5 - cx) * 2 * math.tan(cam.hfov / 2))


def distance(track: Track, frame: Frame | None, cam: CameraModel) -> float | None:
    """Distance to a target in meters: depth median over the box core, else a person-height estimate."""
    x1, y1, x2, y2 = track.box
    if frame is not None and frame.depth is not None:
        h, w = frame.depth.shape
        bw, bh = x2 - x1, y2 - y1
        # Core of the box (torso for a person) avoids background bleeding in at the edges.
        cx1, cx2 = int((x1 + bw * 0.3) * w), int((x2 - bw * 0.3) * w)
        cy1, cy2 = int((y1 + bh * 0.2) * h), int((y1 + bh * 0.6) * h)
        core = frame.depth[cy1 : max(cy2, cy1 + 1), cx1 : max(cx2, cx1 + 1)]
        valid = core[np.isfinite(core) & (core > 0.1)]
        if valid.size >= 10:
            return float(np.median(valid))
    if track.label == "person" and y2 - y1 > 0.05:
        return PERSON_HEIGHT_M * 0.5 / math.tan(cam.vfov / 2) / (y2 - y1)
    return None
