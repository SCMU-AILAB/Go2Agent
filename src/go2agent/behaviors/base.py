"""Behavior primitives: closed-loop controllers ticked at the control rate by the Executor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from go2agent.behaviors.geometry import CameraModel
from go2agent.core.types import ActionSpec, Frame, RobotState, Tracks, Velocity


class Status(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Tick:
    now: float
    frame: Frame | None
    tracks: Tracks | None
    tracker_healthy: bool
    robot: RobotState


@dataclass(frozen=True, slots=True)
class Step:
    status: Status = Status.RUNNING
    velocity: Velocity | None = None  # None = hold still
    perform: str | None = None  # start this native action on this tick
    reason: str = ""  # for FAILED: machine-readable cause, e.g. "target_lost"

    @classmethod
    def ok(cls, reason: str = "") -> Step:
        return cls(Status.SUCCEEDED, reason=reason)

    @classmethod
    def fail(cls, reason: str) -> Step:
        return cls(Status.FAILED, reason=reason)


@dataclass(frozen=True, slots=True)
class BehaviorEnv:
    camera: CameraModel
    actions: dict[str, ActionSpec]  # actions the brain may use


class BehaviorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Behavior[A: BehaviorArgs](ABC):
    name: ClassVar[str]
    description: ClassVar[str]  # model-facing
    args_model: ClassVar[type[BehaviorArgs]]  # the concrete A, used to validate and describe args

    def __init__(self, args: A, env: BehaviorEnv) -> None:
        self.args = args
        self.env = env
        self.started: float | None = None

    def tick(self, t: Tick) -> Step:
        if self.started is None:
            self.started = t.now
        return self.step(t)

    @abstractmethod
    def step(self, t: Tick) -> Step: ...

    def elapsed(self, t: Tick) -> float:
        return 0.0 if self.started is None else t.now - self.started

    def summary(self) -> dict[str, Any]:
        return {"behavior": self.name, "args": self.args.model_dump()}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.model_dump().items())
        return f"{self.name}({args})"
