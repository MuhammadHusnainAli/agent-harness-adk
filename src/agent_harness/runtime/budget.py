"""Budget and rate guard: a spend ceiling and throughput limits the run respects."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from pydantic import BaseModel

from ..errors import BudgetExceeded
from ..types import Usage

__all__ = ["Budget", "BudgetGuard", "RateLimit", "RateGuard"]


class Budget(BaseModel):
    """Ceilings for one run. `None` means no limit on that axis."""

    max_usd: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    max_seconds: float | None = None
    max_tool_calls: int | None = None
    max_subagents: int | None = None

    @classmethod
    def unlimited(cls) -> Budget:
        return cls()


class BudgetGuard:
    """Tracks spend as the run goes and stops it at the ceiling.

    Attribution is per sub-agent and per task, so you can charge it back.
    """

    def __init__(self, budget: Budget | None = None, *, parent: BudgetGuard | None = None):
        self.budget = budget or Budget()
        self.parent = parent
        self.usage = Usage()
        self.started = time.monotonic()
        self.steps = 0
        self.tool_calls = 0
        self.subagents = 0
        self.by_agent: dict[str, Usage] = defaultdict(Usage)
        self.by_task: dict[str, float] = defaultdict(float)

    # ---- recording ----------------------------------------------------
    def record(self, usage: Usage, *, agent: str = "", task: str = "") -> None:
        self.usage += usage
        if agent:
            self.by_agent[agent] += usage
        if task:
            self.by_task[task] += usage.cost_usd
        if self.parent:
            self.parent.record(usage, agent=agent, task=task)
        self.check()

    def step(self) -> None:
        self.steps += 1
        if self.budget.max_steps is not None and self.steps > self.budget.max_steps:
            raise BudgetExceeded(f"step cap reached ({self.budget.max_steps})",
                                 kind="steps", limit=self.budget.max_steps,
                                 spent=self.steps)
        self.check()

    def tool_call(self) -> None:
        self.tool_calls += 1
        if (self.budget.max_tool_calls is not None
                and self.tool_calls > self.budget.max_tool_calls):
            raise BudgetExceeded(f"tool-call cap reached ({self.budget.max_tool_calls})",
                                 kind="tool_calls", limit=self.budget.max_tool_calls,
                                 spent=self.tool_calls)

    def subagent(self) -> None:
        self.subagents += 1
        if (self.budget.max_subagents is not None
                and self.subagents > self.budget.max_subagents):
            raise BudgetExceeded(f"sub-agent cap reached ({self.budget.max_subagents})",
                                 kind="subagents", limit=self.budget.max_subagents,
                                 spent=self.subagents)
        if self.parent:
            self.parent.subagent()

    # ---- checking -----------------------------------------------------
    def check(self) -> None:
        b = self.budget
        if b.max_usd is not None and self.usage.cost_usd > b.max_usd:
            raise BudgetExceeded(
                f"spend ceiling reached: ${self.usage.cost_usd:.4f} of ${b.max_usd:.4f}",
                kind="usd", limit=b.max_usd, spent=self.usage.cost_usd,
            )
        if b.max_tokens is not None and self.usage.total_tokens > b.max_tokens:
            raise BudgetExceeded(
                f"token cap reached: {self.usage.total_tokens} of {b.max_tokens}",
                kind="tokens", limit=b.max_tokens, spent=self.usage.total_tokens,
            )
        if b.max_seconds is not None and self.elapsed > b.max_seconds:
            raise BudgetExceeded(f"deadline passed after {self.elapsed:.1f}s",
                                 kind="seconds", limit=b.max_seconds, spent=self.elapsed)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_usd(self) -> float | None:
        if self.budget.max_usd is None:
            return None
        return max(self.budget.max_usd - self.usage.cost_usd, 0.0)

    def child(self, budget: Budget | None = None) -> BudgetGuard:
        """A sub-agent's guard: its own caps, but spend rolls up to the parent."""
        return BudgetGuard(budget or Budget(), parent=self)

    def report(self) -> dict[str, object]:
        return {
            "cost_usd": round(self.usage.cost_usd, 6),
            "tokens": self.usage.total_tokens,
            "calls": self.usage.calls,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "subagents": self.subagents,
            "elapsed_s": round(self.elapsed, 3),
            "by_agent": {k: round(v.cost_usd, 6) for k, v in self.by_agent.items()},
            "by_task": {k: round(v, 6) for k, v in self.by_task.items()},
        }


class RateLimit(BaseModel):
    """Throughput ceilings. `None` means no limit on that axis."""

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    max_concurrent: int | None = None

    @classmethod
    def unlimited(cls) -> RateLimit:
        return cls()


class RateGuard:
    """A sliding-window throughput limiter that waits rather than failing.

    Being rate-limited by a provider costs a round trip and a retry; pacing
    ourselves costs nothing, so `acquire()` sleeps until there is room.
    """

    def __init__(self, limit: RateLimit | None = None, *, window: float = 60.0) -> None:
        self.limit = limit or RateLimit()
        self.window = window
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self._sem = (asyncio.Semaphore(self.limit.max_concurrent)
                     if self.limit.max_concurrent else None)
        self.waits = 0
        self.total_wait_s = 0.0

    @property
    def active(self) -> bool:
        return (self.limit.requests_per_minute is not None
                or self.limit.tokens_per_minute is not None)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] < cutoff:
            self._tokens.popleft()

    def _delay(self, now: float, estimated_tokens: int) -> float:
        """How long until this call fits inside both windows."""
        self._prune(now)
        waits: list[float] = []
        rpm = self.limit.requests_per_minute
        if rpm is not None and len(self._requests) >= rpm:
            waits.append(self._requests[0] + self.window - now)
        tpm = self.limit.tokens_per_minute
        if tpm is not None and self._tokens:
            spent = sum(t for _, t in self._tokens)
            if spent + estimated_tokens > tpm:
                waits.append(self._tokens[0][0] + self.window - now)
        return max([w for w in waits if w > 0], default=0.0)

    async def acquire(self, estimated_tokens: int = 0) -> float:
        """Wait until the call fits the limits. Returns how long it waited."""
        if not self.active:
            return 0.0
        waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                delay = self._delay(now, estimated_tokens)
                if delay <= 0:
                    self._requests.append(now)
                    if estimated_tokens:
                        self._tokens.append((now, estimated_tokens))
                    return waited
            self.waits += 1
            await asyncio.sleep(min(delay, self.window))
            waited = round(waited + delay, 3)
            self.total_wait_s = round(self.total_wait_s + delay, 3)

    def record(self, usage: Usage) -> None:
        """Correct the token window with what the call actually used."""
        if self.limit.tokens_per_minute is not None and usage.total_tokens:
            self._tokens.append((time.monotonic(), usage.total_tokens))

    async def __aenter__(self) -> RateGuard:
        if self._sem:
            await self._sem.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._sem:
            self._sem.release()

    def stats(self) -> dict[str, float | int]:
        self._prune(time.monotonic())
        return {
            "requests_in_window": len(self._requests),
            "tokens_in_window": sum(t for _, t in self._tokens),
            "waits": self.waits,
            "total_wait_s": self.total_wait_s,
        }
