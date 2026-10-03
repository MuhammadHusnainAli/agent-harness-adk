"""Semantic recall: answer from precedent instead of from scratch.

Embeddings come from whatever you give it. With no embedder configured it falls
back to a deterministic hashing embedder — offline, free, no extra dependency,
and good enough to surface records that share vocabulary with the query. Swap in
`ProviderEmbedder(OpenAIProvider())` when you want real semantics.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..llm_providers.fake import hash_embedding
from .base import MemoryRecord, MemoryStore
from .trace import Trace

__all__ = ["Embedder", "HashEmbedder", "ProviderEmbedder", "VectorStore", "SemanticMemory",
           "DatabaseIndex", "cosine"]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Zero-dependency, deterministic, offline. The default."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(t, self.dims) for t in texts]


class ProviderEmbedder:
    """Real embeddings from a provider that has an embeddings endpoint."""

    def __init__(self, provider: Any, model: str | None = None, *, batch: int = 64) -> None:
        self.provider = provider
        self.model = model
        self.batch = batch

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            out.extend(await self.provider.embed(chunk, self.model))
        return out


class VectorStore:
    """A brute-force cosine index. Exact, dependency-free, fine to ~50k rows."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._rows: list[tuple[list[float], MemoryRecord]] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            vector = blob.pop("_vector", [])
            self._rows.append((vector, MemoryRecord(**blob)))

    def add(self, vector: list[float], record: MemoryRecord) -> None:
        self._rows.append((vector, record))
        if self.path:
            blob = record.model_dump(mode="json")
            blob["_vector"] = vector
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(blob) + "\n")

    def search(self, vector: list[float], *, limit: int = 5, scope: str | None = None,
               min_score: float = 0.0,
               trace: Trace | None = None) -> list[tuple[float, MemoryRecord]]:
        scored = [
            (cosine(vector, vec), rec)
            for vec, rec in self._rows
            if (scope is None or rec.scope == scope)
            and (trace is None or trace.matches(rec))
        ]
        scored = [pair for pair in scored if pair[0] > min_score]
        scored.sort(key=lambda pair: -pair[0])
        return scored[:limit]

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()
        if self.path and self.path.exists():
            self.path.unlink()


class DatabaseIndex:
    """Memory's index in a vector database — past what one process can scan.

        from agent_harness.knowledge import vector_store

        memory = SemanticMemory(store, embedder=embedder,
                                index=vector_store("qdrant://host:6333/memory"))

    Each record is stored with its scope and its owner, and a search is held to
    them in the database, not after the fact — so a tenant's memory is never
    even fetched for another. A record with no owner on an axis is shared, as
    it is everywhere else.
    """

    def __init__(self, store: Any) -> None:
        self.store = store
        self._ready = False

    async def add(self, vector: list[float], record: MemoryRecord) -> None:
        from ..knowledge.base import VectorRecord

        if not self._ready:
            await self.store.ensure(len(vector))
            self._ready = True
        await self.store.upsert([VectorRecord(record.id, vector, record.text, {
            "scope": record.scope, "kind": record.kind,
            # "" for "nobody in particular", so it can be asked for by name.
            "user_id": record.user_id or "", "tenant_id": record.tenant_id or "",
            "session_id": record.session_id or "",
            "record": record.model_dump_json()})])

    async def search(self, vector: list[float], *, limit: int = 5,
                     scope: str | None = None, min_score: float = 0.0,
                     trace: Trace | None = None) -> list[tuple[float, MemoryRecord]]:
        if not self._ready:
            await self.store.ensure(len(vector))
            self._ready = True
        wanted: dict[str, Any] = {"scope": scope} if scope is not None else {}
        for axis, value in (trace.filters() if trace is not None else {}).items():
            wanted[axis] = {"$in": [value, ""]}
        out = []
        for hit in await self.store.query(vector, k=limit, filter=wanted or None):
            if hit.score <= min_score:
                continue
            try:
                out.append((hit.score, MemoryRecord(**json.loads(hit.metadata["record"]))))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    async def clear(self) -> None:
        await self.store.clear()
        self._ready = False


class SemanticMemory(MemoryStore):
    """A MemoryStore that also retrieves by meaning. Wraps any other store.

    `index` is where the vectors are kept: by default in this process, exact
    and good to about fifty thousand records. Give it a vector database —
    `index=vector_store("pgvector://…")`, or any `agent_harness.knowledge`
    store — and it is good to as many as the database is.
    """

    def __init__(self, store: MemoryStore, *, embedder: Embedder | None = None,
                 index: Any = None, min_score: float = 0.05) -> None:
        self.store = store
        self.embedder: Embedder = embedder if embedder is not None else HashEmbedder()
        if index is not None and hasattr(index, "upsert") and hasattr(index, "query"):
            index = DatabaseIndex(index)
        self.index = index if index is not None else VectorStore()
        #: The index is a database: it is asked, not scanned.
        self.remote = isinstance(self.index, DatabaseIndex)
        self.min_score = min_score
        self._lock = asyncio.Lock()

    async def append(self, record: MemoryRecord) -> MemoryRecord:
        await self.store.append(record)
        if record.text.strip():
            vector = record.embedding or (await self.embedder.embed([record.text]))[0]
            if self.remote:
                await self.index.add(vector, record)
            else:
                async with self._lock:
                    self.index.add(vector, record)
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        return await self.store.all(scope, limit=limit, kind=kind, trace=trace)

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        await self.store.clear(scope, trace=trace)
        if scope is None and trace is None:
            if self.remote:
                await self.index.clear()
            else:
                self.index.clear()

    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        return await self.store.read_doc(name, trace=trace)

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        await self.store.write_doc(name, text, trace=trace)

    async def aclose(self) -> None:
        await self.store.aclose()

    async def search(self, query: str, *, scope: str | None = None, limit: int = 5,
                     trace: Trace | None = None) -> list[MemoryRecord]:
        if not query.strip() or (not self.remote and len(self.index) == 0):
            return await self.store.search(query, scope=scope, limit=limit,
                                           trace=trace)
        vector = (await self.embedder.embed([query]))[0]
        hits = self.index.search(vector, limit=limit, scope=scope,
                                 min_score=self.min_score, trace=trace)
        if self.remote:
            hits = await hits
        if not hits:
            return await self.store.search(query, scope=scope, limit=limit,
                                           trace=trace)
        return [rec for _, rec in hits]

    async def search_scored(self, query: str, *, scope: str | None = None,
                            limit: int = 5, trace: Trace | None = None
                            ) -> list[tuple[float, MemoryRecord]]:
        vector = (await self.embedder.embed([query]))[0]
        hits = self.index.search(vector, limit=limit, scope=scope,
                                 min_score=self.min_score, trace=trace)
        return await hits if self.remote else hits
