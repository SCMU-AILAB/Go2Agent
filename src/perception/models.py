"""Structured output produced by a person detector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    observed_at_s: float
    person_count: int = 0
    nearest_person_distance_m: float | None = None
    person_center_x: float | None = None
    confidence: float | None = None
    source: str = "camera"

    def __post_init__(self) -> None:
        if self.observed_at_s < 0:
            raise ValueError("observed_at_s must not be negative")
        if self.person_count < 0:
            raise ValueError("person_count must not be negative")
        if (
            self.nearest_person_distance_m is not None
            and self.nearest_person_distance_m <= 0
        ):
            raise ValueError("nearest person distance must be greater than zero")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if self.person_center_x is not None and not 0 <= self.person_center_x <= 1:
            raise ValueError("person_center_x must be between zero and one")
        if not self.source.strip():
            raise ValueError("perception source must not be empty")

    @property
    def person_detected(self) -> bool:
        return self.person_count > 0

    @classmethod
    def now(
        cls,
        *,
        person_count: int = 0,
        nearest_person_distance_m: float | None = None,
        person_center_x: float | None = None,
        confidence: float | None = None,
        source: str = "camera",
    ) -> PerceptionResult:
        return cls(
            observed_at_s=time.monotonic(),
            person_count=person_count,
            nearest_person_distance_m=nearest_person_distance_m,
            person_center_x=person_center_x,
            confidence=confidence,
            source=source,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at_s": self.observed_at_s,
            "person_detected": self.person_detected,
            "person_count": self.person_count,
            "nearest_person_distance_m": self.nearest_person_distance_m,
            "person_center_x": self.person_center_x,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One aligned camera sample shared by video policy and safety perception."""

    observed_at_s: float
    rgb: Any
    depth: Any | None
    observation: PerceptionResult
    nearest_obstacle_distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.observed_at_s < 0:
            raise ValueError("frame timestamp must not be negative")
        if self.observation.observed_at_s != self.observed_at_s:
            raise ValueError("frame and observation timestamps must match")
        if (
            self.nearest_obstacle_distance_m is not None
            and self.nearest_obstacle_distance_m <= 0
        ):
            raise ValueError("nearest obstacle distance must be greater than zero")

    @property
    def source(self) -> str:
        return self.observation.source
