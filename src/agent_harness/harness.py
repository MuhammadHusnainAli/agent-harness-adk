"""The harness core: the shared services every agent in a run depends on.

One `Harness` is created per application (or per run) and passed down to every
sub-agent, so tracing, spend, permissions, caching and workspaces are consistent
across the whole tree instead of being re-invented per agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .guardrails import Guardrails
from .llm_providers.base import Provider
from .memory.base import InMemoryStore, MemoryStore
from .runtime.audit import AuditTrail
from .runtime.budget import Budget, BudgetGuard, RateGuard, RateLimit
from .runtime.cache import ResultCache
from .runtime.checkpoints import Checkpointer
from .runtime.control import StopController
from .runtime.deliverables import DeliverableStore
from .runtime.health import ServiceHealth
from .runtime.hooks import HookEngine
from .runtime.journal import RunJournal
from .runtime.permissions import PolicyGate
from .runtime.replay import Replayer
from .runtime.router import ModelRouter
from .runtime.scheduler import ConcurrencyScheduler
from .runtime.session import InMemorySessionStore, SessionStore
from .runtime.tracing import Tracer, jsonl_exporter
from .runtime.workspace import WorkspaceBroker
from .spec import SpecCompiler

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
    deliverables: DeliverableStore = field(default_factory=DeliverableStore)
    audit: AuditTrail = field(default_factory=AuditTrail)
    health: ServiceHealth = field(default_factory=ServiceHealth)
    control: StopController = field(default_factory=StopController)
    budget: Budget = field(default_factory=Budget)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    #: An `agent_harness.governance.Governance`, or None. Typed loosely so the
    #: governance package is only imported by those who use it.
    governance: Any = None
    _guard: BudgetGuard | None = field(default=None, repr=False)
    _rate: RateGuard | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.governance is not None:
            self.governance.attach(self)

    @property
    def guard(self) -> BudgetGuard:
        """The run-wide budget guard. Sub-agents get children of this."""
        if self._guard is None:
            self._guard = BudgetGuard(self.budget)
        return self._guard

    def reset_budget(self, budget: Budget | None = None) -> BudgetGuard:
        self._guard = BudgetGuard(budget or self.budget)
        return self._guard

    @property
    def rate(self) -> RateGuard:
        """Throughput limiter shared by every agent on this harness."""
        if self._rate is None:
            self._rate = RateGuard(self.rate_limit)
        return self._rate

    @property
    def replayer(self) -> Replayer:
        """Re-run a recorded run, whole or from any prior step."""
        return Replayer(self.checkpoints)

    @property
    def compiler(self) -> SpecCompiler:
        """Turn a sub-agent blueprint into the payload a provider accepts."""
        return SpecCompiler(router=self.router)

    def stop(self, reason: str = "", *, mode: str = "drain",
             requested_by: str = "human") -> Any:
        """Stop every run on this harness. A human is always in charge."""
        state = self.control.stop(reason, mode=mode, requested_by=requested_by)
        self.audit.record(requested_by, "stop", decision=mode, reason=reason)
        return state

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
            deliverables=DeliverableStore(base / "deliverables"),
            audit=AuditTrail(base / "audit.jsonl"),
            **kwargs,
        )

    @classmethod
    def testing(cls, provider: Provider | None = None, **kwargs: Any) -> Harness:
        """No disk, no network, no tracing noise."""
        from .llm_providers.fake import FakeProvider

        return cls(provider=provider or FakeProvider(), tracer=Tracer(enabled=False),
                   **kwargs)

    def report(self) -> dict[str, Any]:
        """One glance at the run: spend, cache, concurrency, guardrails."""
        return {
            "budget": self.guard.report(),
            "rate": self.rate.stats(),
            "cache": self.cache.stats(),
            "scheduler": self.scheduler.stats(),
            "guardrails": self.guardrails.report(),
            "health": self.health.snapshot(self.scheduler),
            "control": self.control.report(),
            "deliverables": self.deliverables.manifest(),
            "audit_entries": len(self.audit),
            "journal_entries": len(self.journal.entries),
            "spans": len(self.tracer.spans),
            **({"governance": self.governance.summary()}
               if self.governance is not None else {}),
        }

    async def aclose(self) -> None:
        from .llm_providers import close_all

        if isinstance(self.provider, Provider):
            await self.provider.aclose()
        await close_all()
        self.workspaces.cleanup()
