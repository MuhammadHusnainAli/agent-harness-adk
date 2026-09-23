"""Shared object-store implementation: S3, Azure Blob, Google Cloud Storage.

Keys are laid out so the trace and the scope are *prefixes*, which is the only
filter an object store can push down:

    {prefix}/records/{trace}/{scope}/{ts}-{id}.json
    {prefix}/docs/{trace}/{name}

That makes reading one user's memory a single prefix listing. Anything narrower
— a `kind`, a keyword — is filtered after the fetch, so treat an object store as
durable archival for memory rather than a hot query path. For heavy recall put
SQL or Mongo in front of it.
"""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from typing import Any, ClassVar

from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["ObjectStoreMemory"]


class ObjectStoreMemory(MemoryStore):
    """Records as one small object each, documents as one object per name."""

    driver_hint: ClassVar[str] = ""

    def __init__(self, bucket: str, *, prefix: str = "agent-memory",
                 concurrency: int = 16, **options: Any) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.concurrency = concurrency
        self.options = options

    # ---- what a backend must provide ---------------------------------------
    @abstractmethod
    async def _put(self, key: str, body: bytes, content_type: str) -> None: ...

    @abstractmethod
    async def _get(self, key: str) -> bytes | None: ...

    @abstractmethod
    async def _list(self, prefix: str) -> list[str]: ...

    @abstractmethod
    async def _delete(self, keys: list[str]) -> None: ...

    # ---- key layout ---------------------------------------------------------
    def _records_prefix(self, trace: Trace | None, scope: str | None) -> str:
        parts = [self.prefix, "records", trace.slug if trace is not None else ""]
        if scope:
            parts.append(scope)
        return "/".join(p for p in parts if p) + "/"

    def _record_key(self, record: MemoryRecord, trace: Trace | None) -> str:
        namespace = trace.slug if trace is not None else _slug_of(record)
        # The timestamp leads so a prefix listing comes back in time order.
        return (f"{self.prefix}/records/{namespace}/{record.scope}/"
                f"{record.ts:.6f}-{record.id}.json")

    def _doc_object(self, name: str, trace: Trace | None) -> str:
        namespace = trace.slug if trace is not None else "_shared"
        return f"{self.prefix}/docs/{namespace}/{name}"

    # ---- records -------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        body = record.model_dump_json().encode()
        await self._put(self._record_key(record, None), body, "application/json")
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        keys = await self._list(self._records_prefix(trace, scope))
        keys.sort()                                  # the key starts with the time
        if limit and not kind:
            keys = keys[-limit:]                     # newest, before fetching

        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch(key: str) -> MemoryRecord | None:
            async with semaphore:
                raw = await self._get(key)
            if not raw:
                return None
            try:
                return MemoryRecord(**json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                return None

        fetched = await asyncio.gather(*(fetch(k) for k in keys))
        rows = [r for r in fetched if r is not None]
        if kind:
            rows = [r for r in rows if r.kind == kind]
        if trace is not None:
            rows = [r for r in rows if trace.matches(r)]
        rows.sort(key=lambda r: r.ts)
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        keys = await self._list(self._records_prefix(trace, scope))
        if trace is not None:
            keys += await self._list(
                f"{self.prefix}/docs/{trace.slug}/")
        elif scope is None:
            keys += await self._list(f"{self.prefix}/docs/")
        if keys:
            await self._delete(keys)

    # ---- documents -----------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        raw = await self._get(self._doc_object(name, trace))
        return raw.decode("utf-8", errors="replace") if raw else ""

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        await self._put(self._doc_object(name, trace), text.encode(),
                        "text/markdown" if name.endswith(".md") else "text/plain")


def _slug_of(record: MemoryRecord) -> str:
    """The namespace a record belongs to, from the identity stamped on it."""
    return Trace(user_id=record.user_id, session_id=record.session_id,
                 tenant_id=record.tenant_id).slug
