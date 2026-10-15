"""Executor: runs the behavior queue and turns behavior outcomes into events."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from go2agent.behaviors.base import Behavior, Status, Tick
from go2agent.core.types import Event, EventKind, Velocity

Mode = Literal["keep", "replace", "append"]


@dataclass(frozen=True, slots=True)
class ExecOutput:
    velocity: Velocity | None
    events: tuple[Event, ...] = ()


class Executor:
    def __init__(self, perform: Callable[[str], str | None]) -> None:
        """`perform` starts a native action and returns a refusal reason or None (SafetyGate.perform)."""
        self._perform = perform
        self.current: Behavior | None = None
        self.queue: deque[Behavior] = deque()

    @property
    def idle(self) -> bool:
        return self.current is None and not self.queue

    def submit(self, plan: Sequence[Behavior], mode: Mode) -> None:
        if mode == "keep" or not plan:
            return
        if plan[0].name == "stop":
            self.clear()
            return
        if mode == "replace":
            self.queue.clear()
            if self.current is not None and same_behavior(self.current, plan[0]):
                # Re-issuing what is already running must not restart it (a restart is a visible jerk).
                plan = plan[1:]
            else:
                self.current = None
        self.queue.extend(plan)

    def clear(self) -> None:
        self.queue.clear()
        self.current = None

    def tick(self, t: Tick) -> ExecOutput:
        if self.current is None:
            if not self.queue:
                return ExecOutput(None)
            self.current = self.queue.popleft()
        b = self.current
        step = b.tick(t)
        if step.perform is not None and step.status is Status.RUNNING:
            refusal = self._perform(step.perform)
            if refusal is not None:
                return self._finish(t, b, Status.FAILED, refusal)
        if step.status is Status.RUNNING:
            return ExecOutput(step.velocity)
        return self._finish(t, b, step.status, step.reason)

    def _finish(self, t: Tick, b: Behavior, status: Status, reason: str) -> ExecOutput:
        self.current = None
        if status is Status.FAILED:
            self.queue.clear()  # later steps of a plan assume earlier ones worked
            kind = EventKind.BEHAVIOR_FAILED
        else:
            kind = EventKind.BEHAVIOR_DONE
        data = {**b.summary(), "reason": reason, "queue_empty": not self.queue}
        return ExecOutput(None, (Event(kind, t.now, data),))

    def snapshot(self) -> dict:
        return {
            "current": repr(self.current) if self.current else None,
            "queue": [repr(b) for b in self.queue],
        }


def same_behavior(a: Behavior, b: Behavior) -> bool:
    return a.name == b.name and a.args.model_dump() == b.args.model_dump()
