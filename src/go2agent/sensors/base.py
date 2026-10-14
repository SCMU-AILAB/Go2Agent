"""Camera interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from go2agent.core.types import Frame


class Camera(ABC):
    """Captures on its own thread; `latest()` returns the newest frame without blocking."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def latest(self) -> Frame | None: ...
