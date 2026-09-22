"""The harness core: the shared services every agent in a run depends on.

One `Harness` is created per application (or per run) and passed down to every
sub-agent, so tracing, spend, permissions, caching and workspaces are consistent
across the whole tree instead of being re-invented per agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory.base import InMemoryStore, MemoryStore
from .providers.base import Provider
from .runtime.budget import Budget, BudgetGuard
from .runtime.cache import ResultCache
from .runtime.checkpoints import Checkpointer
from .runtime.guardrails import Guardrails
from .runtime.hooks import HookEngine
from .runtime.journal import RunJournal
from .runtime.permissions import PolicyGate
from .runtime.router import ModelRouter
from .runtime.scheduler import ConcurrencyScheduler
from .runtime.session import InMemorySessionStore, SessionStore
from .runtime.tracing import Tracer, jsonl_exporter
from .runtime.workspace import WorkspaceBroker

__all__ = ["Harness"]


@dataclass
class Harness:
    """Shared runtime services. Everything has a working default."""

    provider: Provider | str | None = None
    router: ModelRouter = field(default_factory=ModelRouter)
    hooks: HookEngine = field(default_factory=HookEngine)
    policy: PolicyGate = field(default_factory=PolicyGate)
    guardrails: Guardrails = field(default_factory=Guardrails)
    tracer: Tracer = field(default_factory=Tracer)
    journal: RunJournal = field(default_factory=RunJournal)
    cache: ResultCache = field(default_factory=ResultCache)
    scheduler: ConcurrencyScheduler = field(default_factory=ConcurrencyScheduler)
    sessions: SessionStore = field(default_factory=InMemorySessionStore)
    checkpoints: Checkpointer = field(default_factory=lambda: Checkpointer(every=0))
    memory_store: MemoryStore = field(default_factory=InMemoryStore)
    workspaces: WorkspaceBroker = field(default_factory=WorkspaceBroker)
    budget: Budget = field(default_factory=Budget)
    _guard: BudgetGuard | None = field(default=None, repr=False)

    @property
    def guard(self) -> BudgetGuard:
        """The run-wide budget guard. Sub-agents get children of this."""
        if self._guard is None:
            self._guard = BudgetGuard(self.budget)
        return self._guard

    def reset_budget(self, budget: Budget | None = None) -> BudgetGuard:
        self._guard = BudgetGuard(budget or self.budget)
        return self._guard

    # ---- ready-made configurations ------------------------------------
    @classmethod
    def local(cls, root: str | Path = ".harness", *, trace: bool = True,
              **kwargs: Any) -> Harness:
        """Everything persisted under one directory — the usual production shape."""
        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
        from .memory.base import FileStore
        from .runtime.session import FileSessionStore

        tracer = Tracer([jsonl_exporter(base / "traces.jsonl")] if trace else [])
        return cls(
            tracer=tracer,
            journal=RunJournal(base / "journal.jsonl"),
            sessions=FileSessionStore(base / "sessions"),
            checkpoints=Checkpointer(base / "checkpoints", every=1),
            memory_store=FileStore(base / "memory"),
            cache=ResultCache(path=base / "cache"),
            workspaces=WorkspaceBroker(base / "workspaces"),
            **kwargs,
        )

    @classmethod
    def testing(cls, provider: Provider | None = None, **kwargs: Any) -> Harness:
        """No disk, no network, no tracing noise."""
        from .providers.fake import FakeProvider

        return cls(provider=provider or FakeProvider(), tracer=Tracer(enabled=False),
                   **kwargs)

    def report(self) -> dict[str, Any]:
        """One glance at the run: spend, cache, concurrency, guardrails."""
        return {
            "budget": self.guard.report(),
            "cache": self.cache.stats(),
            "scheduler": self.scheduler.stats(),
            "guardrails": self.guardrails.report(),
            "journal_entries": len(self.journal.entries),
            "spans": len(self.tracer.spans),
        }

    async def aclose(self) -> None:
        from .providers import close_all

        if isinstance(self.provider, Provider):
            await self.provider.aclose()
        await close_all()
        self.workspaces.cleanup()
