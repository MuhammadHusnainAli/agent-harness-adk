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
    #: Something that turns speech into text and text into speech — an
    #: `agent_harness.voice.OpenAISpeech`, say. A voice agent speaks through it,
    #: and a recording attached for a model that cannot listen is transcribed
    #: by it.
    speech: Any = None
    _guard: BudgetGuard | None = field(default=None, repr=False)
    _rate: RateGuard | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # A connection string is as good as a store: "postgresql://…",
        # "redis://…", "azure://container". Resolved here, connected on first use.
        if isinstance(self.sessions, str):
            from .sessions import session_provider

            self.sessions = session_provider(self.sessions)
        if isinstance(self.memory_store, str):
            from .memory import memory_provider

            self.memory_store = memory_provider(self.memory_store)
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
        # Anything named here replaces its file-backed default — chats in
        # Postgres and everything else on disk is `Harness.local(sessions=url)`.
        parts: dict[str, Any] = {
            "tracer": tracer,
            "journal": lambda: RunJournal(base / "journal.jsonl"),
            "sessions": lambda: FileSessionStore(base / "sessions"),
            "checkpoints": lambda: Checkpointer(base / "checkpoints", every=1),
            "memory_store": lambda: FileStore(base / "memory"),
            "cache": lambda: ResultCache(path=base / "cache"),
            "workspaces": lambda: WorkspaceBroker(base / "workspaces"),
            "deliverables": lambda: DeliverableStore(base / "deliverables"),
            "audit": lambda: AuditTrail(base / "audit.jsonl"),
        }
        built = {name: (make() if callable(make) and name != "tracer" else make)
                 for name, make in parts.items() if name not in kwargs}
        return cls(**built, **kwargs)

    @classmethod
    def on(cls, url: str, **kwargs: Any) -> Harness:
        """Memory and chat sessions in one database, on one connection pool.

            Harness.on("postgresql://user:pass@host/agents")
            Harness.on("mongodb://host:27017/agents")

        The usual shape for a service with more than one replica: nothing that
        has to be shared lives on a local disk.
        """
        from .memory import memory_provider
        from .sessions import session_provider

        store = memory_provider(url)
        return cls(memory_store=store, sessions=session_provider(store), **kwargs)

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
        await self.workspaces.aclose()
        await self.sessions.aclose()
        if self.speech is not None and hasattr(self.speech, "aclose"):
            await self.speech.aclose()
