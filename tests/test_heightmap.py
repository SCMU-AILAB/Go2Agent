import math

import numpy as np

from go2agent.robot.heightmap import clearances

RES = 0.06
N = 100  # 6 m x 6 m
ORIGIN = (-3.0, -3.0)


def floor(level: float = -0.03) -> np.ndarray:
    return np.full((N, N), level, np.float32)


def put(grid: np.ndarray, x: float, y: float, h: float, size: float = 0.2) -> None:
    """A square block centered at map (x, y); rows are x, columns are y."""
    r0, r1 = int((x - size / 2 - ORIGIN[0]) / RES), int((x + size / 2 - ORIGIN[0]) / RES)
    c0, c1 = int((y - size / 2 - ORIGIN[1]) / RES), int((y + size / 2 - ORIGIN[1]) / RES)
    grid[r0 : r1 + 1, c0 : c1 + 1] = h


def test_open_floor_is_clear():
    o = clearances(floor(), RES, ORIGIN, (0.0, 0.0, 0.0), stamp=5.0)
    assert (o.front, o.left, o.back, o.right) == (math.inf,) * 4 and o.stamp == 5.0


def test_box_ahead_gives_front_clearance_from_body_edge():
    g = floor()
    put(g, 1.2, 0.0, 0.4)  # box face at ~1.1 m from the center; body half length 0.35
    o = clearances(g, RES, ORIGIN, (0.0, 0.0, 0.0), 0.0)
    assert 0.65 < o.front < 0.8
    assert math.isinf(o.back) and math.isinf(o.left) and math.isinf(o.right)


def test_directions_follow_robot_yaw():
    g = floor()
    put(g, 0.0, 1.0, 0.4)  # +y in the map
    facing_y = clearances(g, RES, ORIGIN, (0.0, 0.0, math.pi / 2), 0.0)
    assert facing_y.front < 1.0 and math.isinf(facing_y.left)
    facing_x = clearances(g, RES, ORIGIN, (0.0, 0.0, 0.0), 0.0)
    assert facing_x.left < 1.0 and math.isinf(facing_x.front)


def test_drop_off_counts_as_obstacle_and_unknown_cells_do_not():
    g = floor()
    put(g, -1.0, 0.0, -0.4, size=0.6)  # a step down behind
    g[:, int((1.5 - ORIGIN[1]) / RES) :] = 1e9  # unobserved area on the left
    o = clearances(g, RES, ORIGIN, (0.0, 0.0, 0.0), 0.0)
    assert o.back < 0.5 and math.isinf(o.left)


def test_off_corridor_obstacles_do_not_block_that_direction():
    g = floor()
    put(g, 1.0, 1.0, 0.5)  # front-left diagonal, outside both corridors
    o = clearances(g, RES, ORIGIN, (0.0, 0.0, 0.0), 0.0)
    assert math.isinf(o.front) and math.isinf(o.left)


def test_ground_estimate_tolerates_a_raised_floor():
    g = floor(level=0.25)  # odom origin not at floor level
    put(g, 1.2, 0.0, 0.7)
    o = clearances(g, RES, ORIGIN, (0.0, 0.0, 0.0), 0.0)
    assert o.front < 1.0 and math.isinf(o.left)
