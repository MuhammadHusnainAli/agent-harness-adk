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
                  kind: str | None = None) -> list[MemoryRecord]: ...

    @abstractmethod
    async def clear(self, scope: str | None = None) -> None: ...

    @abstractmethod
    async def read_doc(self, name: str) -> str: ...

    @abstractmethod
    async def write_doc(self, name: str, text: str) -> None: ...

    async def search(self, query: str, *, scope: str | None = None,
                     limit: int = 5) -> list[MemoryRecord]:
        """Keyword fallback. SemanticMemory overrides this with real retrieval."""
        terms = [t for t in query.lower().split() if len(t) > 2]
        records = await self.all(scope)
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
                  kind: str | None = None) -> list[MemoryRecord]:
        rows = [r for r in self._records
                if (scope is None or r.scope == scope) and (kind is None or r.kind == kind)]
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None) -> None:
        async with self._lock:
            if scope is None:
                self._records.clear()
                self._docs.clear()
            else:
                self._records = [r for r in self._records if r.scope != scope]

    async def read_doc(self, name: str) -> str:
        return self._docs.get(name, "")

    async def write_doc(self, name: str, text: str) -> None:
        self._docs[name] = text


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
                  kind: str | None = None) -> list[MemoryRecord]:
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
        rows.sort(key=lambda r: r.ts)
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None) -> None:
        async with self._lock:
            files = [self._path(scope)] if scope else list(self.root.glob("*.jsonl"))
            for file in files:
                if file.exists():
                    file.unlink()

    def _doc_path(self, name: str) -> Path:
        return self.root / name

    async def read_doc(self, name: str) -> str:
        path = self._doc_path(name)
        if not path.exists():
            return ""
        return await asyncio.to_thread(path.read_text, "utf-8")

    async def write_doc(self, name: str, text: str) -> None:
        path = self._doc_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, text, "utf-8")
