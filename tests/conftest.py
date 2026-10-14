import numpy as np
import pytest

from go2agent.core.types import Frame


class FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_frame(stamp: float, seq: int = 0, depth: float | None = None, w: int = 64, h: int = 48) -> Frame:
    rgb = np.zeros((h, w, 3), np.uint8)
    d = None if depth is None else np.full((h, w), depth, np.float32)
    return Frame(seq=seq, stamp=stamp, rgb=rgb, depth=d)
