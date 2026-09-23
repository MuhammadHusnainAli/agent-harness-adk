"""The four memory scopes, and the manager that wires them to an agent.

    user         preferences, standards, settled decisions — loaded in full,
                 permanent, rewritten at the end of every session
    session      the whole conversation plus the artefacts it produced —
                 in full, this session only
    orchestrator plans, staffing decisions, spend, findings — a *digest* only,
                 never the whole history; distilled into user memory at the end
    sub-agent    only the resources its task produced — nothing carried in,
                 every sub-agent starts clean, and it dies with the task
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..prompts import Prompt
from ..tools import Tool, tool
from ..types import Artifact, Message, new_id
from .base import InMemoryStore, MemoryRecord, MemoryStore
from .semantic import Embedder, SemanticMemory, VectorStore
from .trace import Trace

__all__ = ["UserMemory", "SessionMemory", "OrchestratorMemory", "SubAgentMemory",
           "MemoryManager"]

Summarizer = Callable[[str], Awaitable[str]]

DISTILL_PROMPT = Prompt(
    "memory.distill",
    """You maintain a long-lived memory file about one user.

Rewrite the file below so the next session starts better informed than this one did.

Rules:
- Keep preferences, standards, and decisions that are already settled.
- Fold in anything new from this session's summary; drop what it contradicts.
- Keep it under {max_chars} characters. Bullets, no preamble, no commentary.
- Do not record one-off task details, secrets, or anything that will be stale tomorrow.

CURRENT FILE
{current}

THIS SESSION
{summary}

Return the complete new file and nothing else.""",
)


class UserMemory:
    """`user.md` — what is true across sessions. Loaded in full, every message."""

    def __init__(self, store: MemoryStore, *, doc: str = "user.md",
                 max_chars: int = 8000, trace: Trace | None = None) -> None:
        self.store = store
        self.doc = doc
        self.max_chars = max_chars
        self.trace = trace

    async def load(self) -> str:
        return (await self.store.read_doc(self.doc, trace=self.trace)).strip()

    async def write(self, text: str) -> None:
        await self.store.write_doc(self.doc, text.strip()[: self.max_chars],
                                   trace=self.trace)

    async def remember(self, text: str, *, kind: str = "preference",
                       tags: list[str] | None = None) -> MemoryRecord:
        """Record a durable fact and append it to the document."""
        record = MemoryRecord(scope="user", kind=kind, text=text.strip(),
                              tags=tags or [])
        if self.trace is not None:
            self.trace.stamp(record)
        record = await self.store.append(record)
        current = await self.load()
        if record.line() not in current:
            await self.write(f"{current}\n{record.line()}" if current else record.line())
        return record

    async def block(self) -> str:
        text = await self.load()
        return f"## What you already know about this user\n{text}" if text else ""

    async def distill(self, summary: str, *, summarize: Summarizer | None = None) -> str:
        """Session close → user memory update. Falls back to appending if no model."""
        current = await self.load()
        if not summary.strip():
            return current
        if summarize is None:
            merged = f"{current}\n\n## Session {time.strftime('%Y-%m-%d')}\n{summary}"
            await self.write(merged)
            return (await self.load())
        rewritten = await summarize(DISTILL_PROMPT.render(
            current=current or "(empty)", summary=summary, max_chars=self.max_chars
        ))
        if rewritten.strip():
            await self.write(rewritten)
        return await self.load()


class SessionMemory:
    """The live conversation and everything it produced. Dies with the session."""

    def __init__(self, session_id: str | None = None) -> None:
        self.id = session_id or new_id("ses")
        self.messages: list[Message] = []
        self.artifacts: list[Artifact] = []
        self.started = time.time()
        self.facts: list[str] = []

    def add_message(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def add_artifact(self, artifact: Artifact) -> Artifact:
        self.artifacts.append(artifact)
        return artifact

    def pin(self, fact: str) -> None:
        """A fact that must survive compaction."""
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def transcript(self, limit: int | None = None) -> str:
        rows = self.messages[-limit:] if limit else self.messages
        return "\n".join(f"{m.role}: {m.text}" for m in rows if m.text)

    def digest(self, limit: int = 12) -> str:
        parts = []
        if self.facts:
            parts.append("Pinned facts:\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.artifacts:
            parts.append("Artefacts produced:\n" + "\n".join(
                f"- {a.name} (by {a.produced_by or 'agent'})" for a in self.artifacts
            ))
        tail = self.transcript(limit)
        if tail:
            parts.append(f"Recent exchange:\n{tail}")
        return "\n\n".join(parts)

    def reset(self) -> None:
        self.messages.clear()
        self.artifacts.clear()


class OrchestratorMemory:
    """Plans, staffing calls, spend, findings — read back as a digest, never raw."""

    def __init__(self, store: MemoryStore, *, job_id: str | None = None,
                 trace: Trace | None = None) -> None:
        self.store = store
        self.job_id = job_id or new_id("job")
        self.trace = trace
        self._local: list[MemoryRecord] = []

    async def note(self, kind: str, text: str, **data: Any) -> MemoryRecord:
        record = MemoryRecord(scope="orchestrator", kind=kind, text=text, data=data,
                              source=self.job_id)
        if self.trace is not None:
            self.trace.stamp(record)
        self._local.append(record)
        await self.store.append(record)
        return record

    async def plan(self, text: str, **data: Any) -> MemoryRecord:
        return await self.note("plan", text, **data)

    async def staffing(self, task: str, agent: str, reused: bool) -> MemoryRecord:
        verb = "reused" if reused else "built"
        return await self.note("staffing", f"{task} → {verb} {agent}",
                               task=task, agent=agent, reused=reused)

    async def finding(self, text: str, **data: Any) -> MemoryRecord:
        return await self.note("finding", text, **data)

    async def spend(self, agent: str, usd: float, tokens: int = 0) -> MemoryRecord:
        return await self.note("spend", f"{agent}: ${usd:.4f}", agent=agent, usd=usd,
                               tokens=tokens)

    def digest(self, *, limit: int = 20) -> str:
        """A compact read-back. This is the only thing that reaches the model."""
        if not self._local:
            return ""
        buckets: dict[str, list[str]] = {}
        for rec in self._local:
            buckets.setdefault(rec.kind, []).append(rec.text)
        lines: list[str] = []
        for kind in ("plan", "staffing", "finding", "spend", "note"):
            rows = buckets.get(kind)
            if not rows:
                continue
            if kind == "spend":
                total = sum(r.data.get("usd", 0.0) for r in self._local
                            if r.kind == "spend")
                lines.append(f"spend so far: ${total:.4f} across {len(rows)} sub-agents")
                continue
            head = rows[-limit:]
            lines.append(f"{kind}:\n" + "\n".join(f"  - {r}" for r in head))
        return "## Job so far\n" + "\n".join(lines) if lines else ""

    def total_spend(self) -> float:
        return round(sum(r.data.get("usd", 0.0) for r in self._local
                         if r.kind == "spend"), 6)


class SubAgentMemory:
    """Nothing carried in; only the resources the task produced carried out."""

    def __init__(self, task: str, *, agent: str = "") -> None:
        self.task = task
        self.agent = agent
        self.artifacts: list[Artifact] = []
        self.notes: list[str] = []

    def produce(self, artifact: Artifact) -> Artifact:
        artifact.produced_by = artifact.produced_by or self.agent
        self.artifacts.append(artifact)
        return artifact

    def note(self, text: str) -> None:
        self.notes.append(text)

    def resources(self) -> list[Artifact]:
        """The hand-back: typed artefacts, not a transcript."""
        return list(self.artifacts)

    def close(self) -> list[Artifact]:
        produced = self.resources()
        self.notes.clear()
        self.artifacts = []
        return produced


class MemoryManager:
    """One object an agent holds that knows about all four scopes."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        session: SessionMemory | None = None,
        semantic: bool | SemanticMemory = True,
        embedder: Embedder | None = None,
        index: VectorStore | None = None,
        user_doc: str = "user.md",
        summarize: Summarizer | None = None,
        recall_limit: int = 4,
        trace: Trace | str | dict[str, Any] | None = None,
    ) -> None:
        base = store or InMemoryStore()
        if isinstance(semantic, SemanticMemory):
            self.store: MemoryStore = semantic
        elif semantic:
            self.store = SemanticMemory(base, embedder=embedder, index=index)
        else:
            self.store = base
        self.trace = Trace.of(trace)
        # A session id on the trace and the live session should be the same thing.
        if self.trace.session_id is None and session is not None:
            self.trace.session_id = session.id
        self.user = UserMemory(self.store, doc=user_doc, trace=self.trace)
        self.session = session or SessionMemory()
        if self.trace.session_id is None:
            self.trace.session_id = self.session.id
        self.orchestrator = OrchestratorMemory(self.store, trace=self.trace)
        self.summarize = summarize
        self.recall_limit = recall_limit

    # ---- scopes -------------------------------------------------------
    def subagent(self, task: str, agent: str = "") -> SubAgentMemory:
        return SubAgentMemory(task, agent=agent)

    async def remember(self, text: str, *, kind: str = "fact", scope: str = "user",
                       tags: list[str] | None = None) -> MemoryRecord:
        if scope == "user":
            return await self.user.remember(text, kind=kind, tags=tags)
        record = MemoryRecord(scope=scope, kind=kind, text=text, tags=tags or [])
        self.trace.stamp(record)
        return await self.store.append(record)

    async def recall(self, query: str, *, limit: int | None = None,
                     scope: str | None = None) -> list[MemoryRecord]:
        return await self.store.search(query, scope=scope,
                                       limit=limit or self.recall_limit,
                                       trace=self.trace)

    def for_trace(self, trace: Trace | str | dict[str, Any], **kw: Any) -> MemoryManager:
        """The same backend, scoped to somebody else. One store, many users.

        The store is reused as it is — including its vector index — so serving a
        request per user costs a small object, not a rebuilt index.
        """
        return MemoryManager(
            self.store,
            semantic=self.store if isinstance(self.store, SemanticMemory) else False,
            trace=Trace.of(trace, **kw), summarize=self.summarize,
            recall_limit=self.recall_limit, user_doc=self.user.doc,
        )

    async def aclose(self) -> None:
        await self.store.aclose()

    # ---- what reaches the model ---------------------------------------
    async def prompt_blocks(self, query: str = "") -> list[str]:
        """User memory in full, orchestrator as a digest, precedent if relevant."""
        blocks: list[str] = []
        user_block = await self.user.block()
        if user_block:
            blocks.append(user_block)
        digest = self.orchestrator.digest()
        if digest:
            blocks.append(digest)
        if query:
            hits = await self.recall(query)
            recalled = [h for h in hits if h.scope != "user"]
            if recalled:
                blocks.append("## Relevant precedent\n" +
                              "\n".join(h.line() for h in recalled))
        return blocks

    async def close_session(self, summary: str | None = None) -> str:
        """Session close → user memory update. Returns the new `user.md`."""
        text = summary if summary is not None else self.session.digest()
        return await self.user.distill(text, summarize=self.summarize)

    # ---- tools the agent can call itself -------------------------------
    def tools(self) -> list[Tool]:
        manager = self

        @tool(name="remember", tags=["builtin", "memory"])
        async def remember_tool(fact: str, kind: str = "fact") -> str:
            """Save something worth knowing in future sessions.

            Args:
                fact: one durable sentence — a preference, standard or decision.
                kind: preference, standard, decision or fact.
            """
            record = await manager.remember(fact, kind=kind)
            return f"remembered: {record.text}"

        @tool(name="recall", tags=["builtin", "memory"])
        async def recall_tool(query: str, limit: int = 5) -> str:
            """Search memory for past decisions, findings and preferences.

            Args:
                query: what you are trying to remember.
                limit: how many hits to return.
            """
            hits = await manager.recall(query, limit=limit)
            return "\n".join(h.line() for h in hits) or "nothing on record"

        return [remember_tool, recall_tool]
