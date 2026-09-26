"""Exclusive resource acquisition for concurrently running skills."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ResourceManager:
    """Serialize skills that declare the same hardware resource."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, resource: str) -> asyncio.Lock:
        lock = self._locks.get(resource)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[resource] = lock
        return lock

    @asynccontextmanager
    async def acquire(self, resources: tuple[str, ...]) -> AsyncIterator[None]:
        locks = [self._lock_for(resource) for resource in sorted(set(resources))]
        acquired: list[asyncio.Lock] = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
