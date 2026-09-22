"""Budget and rate guard: a spend ceiling and throughput limits the run respects."""

from __future__ import annotations

import time
from collections import defaultdict

from pydantic import BaseModel

from ..errors import BudgetExceeded
from ..types import Usage

__all__ = ["Budget", "BudgetGuard"]


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
