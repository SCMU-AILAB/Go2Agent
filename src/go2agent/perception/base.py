"""Tracker interface. The default implementation talks to the GPU-server perception service."""

from __future__ import annotations

from abc import ABC, abstractmethod

from go2agent.core.types import Box, Frame, Tracks


class Tracker(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def submit(self, frame: Frame) -> None:
        """Queue a frame for tracking. Non-blocking; drops older pending frames."""

    @abstractmethod
    def latest(self) -> Tracks | None:
        """Newest tracking result, or None if none yet."""

    @abstractmethod
    async def add_box(self, frame_seq: int, box: Box, label: str) -> int:
        """Start tracking an object the detector missed, given a box on a recent frame.
        Returns the new track id."""

    @property
    @abstractmethod
    def healthy(self) -> bool:
        """False when the service is unreachable or results are stale."""
