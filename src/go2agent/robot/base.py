"""Robot interface. Swapping hardware means writing a new implementation of `Robot`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from go2agent.core.types import ActionSpec, Obstacles, RobotState, Velocity


class Robot(ABC):
    """All methods must be cheap and non-blocking; the runtime calls them from a 20Hz loop.

    Only `go2agent.safety.SafetyGate` may call `velocity` / `perform` on a real robot.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def velocity(self, cmd: Velocity) -> None:
        """Apply a body-frame velocity. Implementations must stop on their own if not refreshed
        within a short window when the hardware supports it."""

    @abstractmethod
    def perform(self, action: str) -> str | None:
        """Start a native action from `actions()`. Returns immediately: None if started, else the reason
        it was refused (busy, not connected, ...). Raises only for an unknown action."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all motion now (zero velocity, abort action if possible). Must be idempotent."""

    @abstractmethod
    def state(self) -> RobotState: ...

    @abstractmethod
    def actions(self) -> tuple[ActionSpec, ...]:
        """Native actions this robot supports."""

    def obstacles(self) -> Obstacles | None:
        """Latest obstacle clearances, or None if this robot has no obstacle sensing (e.g. no LiDAR).

        A robot that has sensing must keep returning its last measurement even when it gets old; the
        SafetyGate decides what a stale measurement means."""
        return None
