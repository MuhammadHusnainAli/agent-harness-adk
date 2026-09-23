"""Storage behind every memory scope: records in, records out."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..types import new_id
from .trace import Trace

__all__ = ["MemoryRecord", "MemoryStore", "InMemoryStore", "FileStore"]


class MemoryRecord(BaseModel):
    """One remembered thing. `kind` is free-form: fact, decision, finding, plan..."""

    id: str = Field(default_factory=lambda: new_id("mem"))
    scope: str = "user"
    kind: str = "note"
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = ""
    # Whose memory this is. Flat rather than nested so every backend can index
    # them — a column in SQL, a field in Mongo, a prefix in an object store.
    user_id: str | None = None
    session_id: str | None = None
    tenant_id: str | None = None
    ts: float = Field(default_factory=time.time)
    embedding: list[float] | None = Field(default=None, exclude=True)

    def line(self) -> str:
        return f"- [{self.kind}] {self.text}" if self.kind != "note" else f"- {self.text}"


class MemoryStore(ABC):
    """Append-only records plus a handful of named documents (like `user.md`)."""

    @abstractmethod
    async def append(self, record: MemoryRecord) -> MemoryRecord: ...

    @abstractmethod
    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]: ...

    @abstractmethod
    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None: ...

    @abstractmethod
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str: ...

    @abstractmethod
    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None: ...

    async def search(self, query: str, *, scope: str | None = None,
                     limit: int = 5,
                     trace: Trace | None = None) -> list[MemoryRecord]:
        """Keyword fallback. SemanticMemory overrides this with real retrieval."""
        terms = [t for t in query.lower().split() if len(t) > 2]
        records = await self.all(scope, trace=trace)
        scored: list[tuple[int, MemoryRecord]] = []
        for rec in records:
            hay = f"{rec.text} {rec.kind} {' '.join(rec.tags)}".lower()
            hits = sum(1 for t in terms if t in hay)
            if hits:
                scored.append((hits, rec))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].ts))
        return [rec for _, rec in scored[:limit]]

    async def extend(self, records: list[MemoryRecord]) -> None:
        for record in records:
            await self.append(record)

    @staticmethod
    def doc_key(name: str, trace: Trace | None) -> str:
        """Where a document lives for this trace — `u-alice/user.md`, say."""
        return f"{trace.slug}/{name}" if trace is not None and trace else name

    async def aclose(self) -> None:
        """Release whatever the backend holds open. Safe to call twice."""
        return None


class InMemoryStore(MemoryStore):
    """The default. Fast, and gone when the process exits."""

    def __init__(self) -> None:
        self._records: list[MemoryRecord] = []
        self._docs: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def append(self, record: MemoryRecord) -> MemoryRecord:
        async with self._lock:
            self._records.append(record)
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        rows = [r for r in self._records
                if (scope is None or r.scope == scope)
                and (kind is None or r.kind == kind)
                and (trace is None or trace.matches(r))]
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        async with self._lock:
            keep = [r for r in self._records
                    if (scope is not None and r.scope != scope)
                    or (trace is not None and not trace.matches(r))]
            self._records = keep
            if scope is None and trace is None:
                self._docs.clear()
            elif trace is not None:
                prefix = f"{trace.slug}/"
                self._docs = {k: v for k, v in self._docs.items()
                              if not k.startswith(prefix)}

    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        return self._docs.get(self.doc_key(name, trace), "")

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        self._docs[self.doc_key(name, trace)] = text


class FileStore(MemoryStore):
    """Durable memory on disk: one JSONL per scope, plain files for documents.

    Writes are small and append-only, so this stays cheap even on a hot loop.
    """

    def __init__(self, root: str | Path = ".harness/memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, scope: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in scope)
        return self.root / f"{safe}.jsonl"

    async def append(self, record: MemoryRecord) -> MemoryRecord:
        line = record.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_sync, self._path(record.scope), line)
        return record

    @staticmethod
    def _append_sync(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        files = [self._path(scope)] if scope else sorted(self.root.glob("*.jsonl"))
        rows: list[MemoryRecord] = []
        for file in files:
            if not file.exists():
                continue
            text = await asyncio.to_thread(file.read_text, "utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(MemoryRecord(**json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue  # a half-written line never kills a run
        if kind:
            rows = [r for r in rows if r.kind == kind]
        if trace is not None:
            rows = [r for r in rows if trace.matches(r)]
        rows.sort(key=lambda r: r.ts)
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        if trace is not None:
            # Selective delete: rewrite each file without this trace's rows.
            async with self._lock:
                files = ([self._path(scope)] if scope
                         else list(self.root.glob("*.jsonl")))
                for file in files:
                    if not file.exists():
                        continue
                    kept = [line for line in
                            file.read_text(encoding="utf-8").splitlines()
                            if line.strip() and not self._belongs(line, trace)]
                    file.write_text("\n".join(kept) + ("\n" if kept else ""),
                                    encoding="utf-8")
                folder = self.root / trace.slug
                if folder.is_dir():
                    for doc in folder.iterdir():
                        doc.unlink()
            return
        async with self._lock:
            files = [self._path(scope)] if scope else list(self.root.glob("*.jsonl"))
            for file in files:
                if file.exists():
                    file.unlink()

    @staticmethod
    def _belongs(line: str, trace: Trace) -> bool:
        try:
            blob = json.loads(line)
        except json.JSONDecodeError:
            return False
        return all(blob.get(field) in (None, value)
                   for field, value in trace.filters().items()) and any(
            blob.get(field) == value for field, value in trace.filters().items())

    def _doc_path(self, name: str, trace: Trace | None = None) -> Path:
        return self.root / self.doc_key(name, trace)

    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        path = self._doc_path(name, trace)
        if not path.exists():
            return ""
        return await asyncio.to_thread(path.read_text, "utf-8")

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        path = self._doc_path(name, trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, text, "utf-8")
