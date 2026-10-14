"""Set-of-marks rendering: draw numbered track boxes so the VLM can refer to targets by id."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from go2agent.core.types import Tracks

_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (240, 50, 230)]


def resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def draw_marks(rgb: np.ndarray, tracks: Tracks | None) -> np.ndarray:
    """Copy of `rgb` with each track's box and id drawn on it."""
    img = rgb.copy()
    if tracks is None:
        return img
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 320))
    for t in tracks.items:
        color = _COLORS[t.id % len(_COLORS)]
        x1, y1, x2, y2 = (round(t.box[0] * w), round(t.box[1] * h), round(t.box[2] * w), round(t.box[3] * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = str(t.id)
        scale = max(0.6, max(h, w) / 640)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    return img


def jpeg_bytes(rgb: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
