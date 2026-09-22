"""Concurrency scheduler: queue, semaphore, backpressure.

Parallel sub-agents are the point of the fan-out, but an unbounded fan-out is
how you get rate-limited. Everything that runs many-at-once goes through here.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

__all__ = ["ConcurrencyScheduler"]

T = TypeVar("T")


class ConcurrencyScheduler:
    """A semaphore with bookkeeping, plus a gather that never runs away."""

    def __init__(self, max_concurrency: int = 8, *, max_queue: int = 0) -> None:
        self._limit = max(1, max_concurrency)
        self._sem = asyncio.Semaphore(self._limit)
        self._queue_cap = max_queue
        self.running = 0
        self.peak = 0
        self.completed = 0
        self.failed = 0
        self.queued = 0
        self._depth: contextvars.ContextVar[int] = contextvars.ContextVar(
            f"ah_sched_{id(self)}", default=0
        )

    @property
    def max_concurrency(self) -> int:
        return self._limit

    @max_concurrency.setter
    def max_concurrency(self, value: int) -> None:
        """Change the cap. Set it while the scheduler is idle."""
        self._limit = max(1, value)
        self._sem = asyncio.Semaphore(self._limit)

    async def run(self, coro: Awaitable[T]) -> T:
        """Await `coro` under the concurrency cap."""
        # Work fanned out from inside a slot does not take a second one. Without
        # this, a manager that delegates N tasks in parallel holds every slot
        # while its sub-agents wait for one — a deadlock, not slow progress.
        if self._depth.get() > 0:
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                result = await coro
            except BaseException:
                self.failed += 1
                raise
            else:
                self.completed += 1
                return result
            finally:
                self.running -= 1

        if self._queue_cap and self.queued >= self._queue_cap:
            raise RuntimeError(f"scheduler queue is full ({self._queue_cap})")
        self.queued += 1
        try:
            await self._sem.acquire()
        finally:
            self.queued -= 1
        token = self._depth.set(self._depth.get() + 1)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            result = await coro
        except BaseException:
            self.failed += 1
            raise
        else:
            self.completed += 1
            return result
        finally:
            self.running -= 1
            self._depth.reset(token)
            self._sem.release()

    async def map(self, fn: Callable[[Any], Awaitable[T]], items: Iterable[Any], *,
                  return_exceptions: bool = True) -> list[T | BaseException]:
        """Run `fn` over `items` in parallel, capped. Failures come back as values."""
        tasks = [asyncio.create_task(self.run(fn(item))) for item in items]
        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def gather(self, *coros: Awaitable[T],
                     return_exceptions: bool = True) -> list[T | BaseException]:
        tasks = [asyncio.create_task(self.run(c)) for c in coros]
        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def stats(self) -> dict[str, int]:
        return {"running": self.running, "queued": self.queued, "peak": self.peak,
                "completed": self.completed, "failed": self.failed,
                "limit": self.max_concurrency}
