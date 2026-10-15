"""Obstacle clearances from a 2.5D height map around the robot (e.g. the Go2 L1 LiDAR height map).

The map is a grid of terrain heights in an odometry frame; `data[row, col]` covers
x = origin_x + (row + 0.5) * resolution, y = origin_y + (col + 0.5) * resolution (verified on a Go2 against
the front camera). Cells are obstacles when they rise above, or drop below, the local ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from go2agent.core.types import Obstacles

UNKNOWN_ABOVE = 1e6  # the Go2 marks unobserved cells with 1e9


@dataclass(frozen=True, slots=True)
class HeightMapConfig:
    body_half_length: float = 0.35  # Go2 body is ~0.70 x 0.31 m
    body_half_width: float = 0.16
    corridor_margin: float = 0.10  # widen each corridor beyond the body by this much per side
    max_range: float = 2.5  # ignore cells farther than this from the body center
    rise: float = 0.08  # higher than ground by this much = obstacle (curbs, boxes, legs)
    drop: float = 0.10  # lower than ground by this much = obstacle (steps down, holes)
    ground_ring: tuple[float, float] = (0.4, 1.2)  # where the local ground height is estimated


def clearances(
    data: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    pose: tuple[float, float, float],
    stamp: float,
    config: HeightMapConfig | None = None,
) -> Obstacles:
    """`pose` = (x, y, yaw) of the robot in the map frame."""
    c = config or HeightMapConfig()
    rows, cols = np.meshgrid(
        np.arange(data.shape[0], dtype=np.float64), np.arange(data.shape[1], dtype=np.float64), indexing="ij"
    )
    dx: np.ndarray = (rows + 0.5) * resolution + (origin[0] - pose[0])
    dy: np.ndarray = (cols + 0.5) * resolution + (origin[1] - pose[1])
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    fwd: np.ndarray = dx * cos + dy * sin
    left: np.ndarray = dy * cos - dx * sin
    dist = np.hypot(fwd, left)
    known = np.isfinite(data) & (data < UNKNOWN_ABOVE) & (dist <= c.max_range)

    ring = known & (dist >= c.ground_ring[0]) & (dist <= c.ground_ring[1])
    if not ring.any():
        return Obstacles(stamp, math.inf, math.inf, math.inf, math.inf)
    ground = float(np.percentile(data[ring], 20))  # robust to obstacles inside the ring
    body = (np.abs(fwd) <= c.body_half_length) & (np.abs(left) <= c.body_half_width)
    blocked = known & ~body & ((data > ground + c.rise) | (data < ground - c.drop))

    half_w = c.body_half_width + c.corridor_margin
    half_l = c.body_half_length + c.corridor_margin
    in_width = np.abs(left) <= half_w
    in_length = np.abs(fwd) <= half_l

    def nearest(mask: np.ndarray, extent: np.ndarray, body_half: float) -> float:
        sel = blocked & mask
        if not sel.any():
            return math.inf
        # Cell edge nearest to the robot, measured from the body edge.
        return max(0.0, float(extent[sel].min()) - resolution / 2 - body_half)

    return Obstacles(
        stamp=stamp,
        front=nearest(in_width & (fwd > 0), fwd, c.body_half_length),
        back=nearest(in_width & (fwd < 0), -fwd, c.body_half_length),
        left=nearest(in_length & (left > 0), left, c.body_half_width),
        right=nearest(in_length & (left < 0), -left, c.body_half_width),
    )
