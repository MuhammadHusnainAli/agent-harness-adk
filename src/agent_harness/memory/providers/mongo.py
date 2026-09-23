"""MongoDB memory. Uses motor if it is installed, otherwise pymongo in a thread."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["MongoMemory"]


class MongoMemory(MemoryStore):
    """Records and documents in two collections, indexed on the trace.

        MongoMemory("mongodb://localhost:27017", database="agents")

    Needs `motor` (preferred) or `pymongo`.
    """

    driver_hint: ClassVar[str] = "pip install motor   # or: pip install pymongo"

    def __init__(self, dsn: str = "mongodb://localhost:27017", *,
                 database: str = "agent_harness", collection: str = "memory",
                 docs_collection: str = "", client: Any = None,
                 **options: Any) -> None:
        self.dsn = dsn
        self.database = database
        self.collection = collection
        self.docs_collection = docs_collection or f"{collection}_docs"
        self.options = options
        self._client = client
        self._async_driver = client is not None
        self._ready = False
        self._lock = asyncio.Lock()

    # ---- connection ---------------------------------------------------------
    async def _ensure(self) -> Any:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = self._make_client()
        if not self._ready:
            await self._index()
            self._ready = True
        return self._client

    def _make_client(self) -> Any:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            AsyncIOMotorClient = None
        if AsyncIOMotorClient is not None:
            self._async_driver = True
            return AsyncIOMotorClient(self.dsn, **self.options)
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ConfigurationError(
                "MongoMemory needs a driver — " + self.driver_hint) from exc
        self._async_driver = False
        return MongoClient(self.dsn, **self.options)

    def _records(self) -> Any:
        return self._client[self.database][self.collection]

    def _docs(self) -> Any:
        return self._client[self.database][self.docs_collection]

    async def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """One call path for both drivers: await motor, thread out pymongo."""
        if self._async_driver:
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))

    async def _index(self) -> None:
        # Matches how the harness reads: trace first, then scope, newest first.
        await self._call(self._records().create_index,
                         [("user_id", 1), ("session_id", 1), ("tenant_id", 1),
                          ("scope", 1), ("ts", -1)])
        await self._call(self._docs().create_index, [("doc_key", 1)], unique=True)

    @staticmethod
    def _query(scope: str | None, kind: str | None,
               trace: Trace | None) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if scope is not None:
            query["scope"] = scope
        if kind is not None:
            query["kind"] = kind
        if trace is not None:
            # A record with no value on an axis is shared, not hidden.
            for field, value in trace.filters().items():
                query[field] = {"$in": [value, None]}
        return query

    # ---- records --------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        await self._ensure()
        blob = record.model_dump(mode="json")
        blob["_id"] = record.id
        await self._call(self._records().insert_one, blob)
        return record

    async def extend(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        await self._ensure()
        blobs = []
        for record in records:
            blob = record.model_dump(mode="json")
            blob["_id"] = record.id
            blobs.append(blob)
        await self._call(self._records().insert_many, blobs)

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        await self._ensure()
        cursor = self._records().find(self._query(scope, kind, trace)).sort("ts", -1)
        if limit:
            cursor = cursor.limit(int(limit))
        rows = (await cursor.to_list(length=limit or 10_000) if self._async_driver
                else await asyncio.to_thread(list, cursor))
        out = [MemoryRecord(**{k: v for k, v in row.items() if k != "_id"})
               for row in rows]
        out.reverse()                       # back to chronological order
        return out

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        await self._ensure()
        await self._call(self._records().delete_many,
                         self._query(scope, None, trace))
        if trace is not None:
            await self._call(self._docs().delete_many, {"namespace": trace.slug})
        elif scope is None:
            await self._call(self._docs().delete_many, {})

    # ---- documents -------------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        await self._ensure()
        found = await self._call(self._docs().find_one,
                                 {"doc_key": self.doc_key(name, trace)})
        return (found or {}).get("text", "")

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        import time

        await self._ensure()
        key = self.doc_key(name, trace)
        await self._call(
            self._docs().update_one, {"doc_key": key},
            {"$set": {"doc_key": key, "name": name,
                      "namespace": trace.slug if trace is not None else "_shared",
                      "text": text, "updated": time.time()}},
            upsert=True)

    async def aclose(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            self._client = None
            self._ready = False
