"""Data types shared by every layer. Pure data, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# Normalized image box (x1, y1, x2, y2), each in [0, 1], origin top-left.
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Velocity:
    """Body-frame velocity command. vx forward m/s, vy left m/s, wz counter-clockwise rad/s."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    @property
    def is_zero(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0 and self.wz == 0.0


Velocity.ZERO = Velocity()  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class Frame:
    """One camera frame. `stamp` is time.monotonic() seconds at capture."""

    seq: int
    stamp: float
    rgb: np.ndarray  # (H, W, 3) uint8, RGB order
    depth: np.ndarray | None = None  # (H, W) float32 meters aligned to rgb; 0 or nan = invalid

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.rgb.shape[1], self.rgb.shape[0]


@dataclass(frozen=True, slots=True)
class Track:
    """A tracked object. `id` is stable across frames while the tracker holds it."""

    id: int
    label: str  # e.g. "person"
    box: Box
    score: float


@dataclass(frozen=True, slots=True)
class Tracks:
    """Tracker output for one frame."""

    frame_seq: int
    stamp: float  # stamp of the frame these tracks belong to
    items: tuple[Track, ...] = ()

    def get(self, track_id: int) -> Track | None:
        return next((t for t in self.items if t.id == track_id), None)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """A native action the robot can perform (e.g. Go2 'hello', 'heart')."""

    name: str
    description: str
    duration: float  # expected seconds until the robot is ready again
    tags: frozenset[str] = frozenset()  # e.g. {"dangerous"}


@dataclass(frozen=True, slots=True)
class Obstacles:
    """Clearance in meters from the body edge to the nearest obstacle, per direction of travel.

    Each value covers a corridor as wide (front/back) or as long (left/right) as the body plus a margin,
    so it answers "how far can the robot move this way". `math.inf` = nothing seen within range.
    `stamp` is time.monotonic() of the measurement.
    """

    stamp: float
    front: float
    left: float
    back: float
    right: float


@dataclass(frozen=True, slots=True)
class RobotState:
    stamp: float
    connected: bool
    busy: bool = False  # a native action is running
    details: dict[str, Any] = field(default_factory=dict)


class EventKind(StrEnum):
    TASK = "task"  # new task text; data: {"text"}
    USER_SPEECH = "user_speech"  # data: {"text"}
    BEHAVIOR_DONE = "behavior_done"  # data: {"behavior", "args"}
    BEHAVIOR_FAILED = "behavior_failed"  # data: {"behavior", "args", "reason"}
    TRACK_APPEARED = "track_appeared"  # data: {"id", "label"}
    TRACK_LOST = "track_lost"  # data: {"id", "label"}
    OBSTACLE_CLOSE = "obstacle_close"  # data: {"distance"}
    PERIODIC = "periodic"
    ESTOP = "estop"  # data: {"source"}
    BRAIN_DECISION = "brain_decision"  # data: Decision dict + timings
    BRAIN_ERROR = "brain_error"  # data: {"reason"}
    SAY = "say"  # data: {"text"}
    SERVICE = "service"  # data: {"service", "ok", "detail"}


# Events that make the brain think. Others are recorded only.
TRIGGER_EVENTS = frozenset(
    {
        EventKind.TASK,
        EventKind.USER_SPEECH,
        EventKind.BEHAVIOR_DONE,
        EventKind.BEHAVIOR_FAILED,
        EventKind.TRACK_APPEARED,
        EventKind.TRACK_LOST,
        EventKind.OBSTACLE_CLOSE,
        EventKind.PERIODIC,
    }
)


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    stamp: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "stamp": self.stamp, "data": self.data}
