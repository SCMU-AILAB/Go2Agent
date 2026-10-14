"""Robot interface. Swapping hardware means writing a new implementation of `Robot`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from go2agent.core.types import ActionSpec, RobotState, Velocity


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
    def perform(self, action: str) -> None:
        """Start a native action from `actions()`. Returns immediately."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all motion now (zero velocity, abort action if possible). Must be idempotent."""

    @abstractmethod
    def state(self) -> RobotState: ...

    @abstractmethod
    def actions(self) -> tuple[ActionSpec, ...]:
        """Native actions this robot supports."""
