"""Elasticsearch and OpenSearch memory — full-text recall without an embedder."""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["ElasticsearchMemory"]


class ElasticsearchMemory(MemoryStore):
    """Records in one index, documents in another.

        ElasticsearchMemory("https://localhost:9200", api_key="...")

    `search()` uses the engine's own text scoring rather than the keyword
    fallback, so this is the one backend where recall is good without an
    embedder. Needs `elasticsearch` (the OpenSearch client works too).
    """

    driver_hint: ClassVar[str] = "pip install elasticsearch"

    def __init__(self, hosts: str | list[str] = "http://localhost:9200", *,
                 index: str = "agent-memory", docs_index: str = "",
                 client: Any = None, refresh: bool = False, **options: Any) -> None:
        self.hosts = hosts
        self.index = index
        self.docs_index = docs_index or f"{index}-docs"
        self.refresh = refresh          # True in tests; costly in production
        self._client = client
        self._options = options
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        from elasticsearch import AsyncElasticsearch
                    except ImportError as exc:
                        raise ConfigurationError(
                            "ElasticsearchMemory needs a driver — " + self.driver_hint
                        ) from exc
                    self._client = AsyncElasticsearch(self.hosts, **self._options)
        if not self._ready:
            await self._create_index()
            self._ready = True
        return self._client

    async def _create_index(self) -> None:
        mapping = {
            "mappings": {"properties": {
                "scope": {"type": "keyword"}, "kind": {"type": "keyword"},
                "text": {"type": "text"}, "source": {"type": "keyword"},
                "user_id": {"type": "keyword"}, "session_id": {"type": "keyword"},
                "tenant_id": {"type": "keyword"}, "ts": {"type": "double"},
            }}
        }
        for name, body in ((self.index, mapping), (self.docs_index, None)):
            if not await self._client.indices.exists(index=name):
                await self._client.indices.create(index=name, **(body or {}))

    @staticmethod
    def _query(scope: str | None, kind: str | None, trace: Trace | None,
               text: str = "") -> dict[str, Any]:
        must: list[dict[str, Any]] = []
        if text:
            must.append({"match": {"text": text}})
        filters: list[dict[str, Any]] = []
        if scope is not None:
            filters.append({"term": {"scope": scope}})
        if kind is not None:
            filters.append({"term": {"kind": kind}})
        if trace is not None:
            for field, value in trace.filters().items():
                # Either it is this trace's, or it predates traces entirely.
                filters.append({"bool": {"should": [
                    {"term": {field: value}},
                    {"bool": {"must_not": {"exists": {"field": field}}}},
                ], "minimum_should_match": 1}})
        return {"bool": {"must": must or [{"match_all": {}}], "filter": filters}}

    # ---- records ---------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        client = await self._ensure()
        await client.index(index=self.index, id=record.id,
                           document=record.model_dump(mode="json"),
                           refresh=self.refresh)
        return record

    async def extend(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        client = await self._ensure()
        operations: list[dict[str, Any]] = []
        for record in records:
            operations.append({"index": {"_index": self.index, "_id": record.id}})
            operations.append(record.model_dump(mode="json"))
        await client.bulk(operations=operations, refresh=self.refresh)

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        client = await self._ensure()
        found = await client.search(
            index=self.index, query=self._query(scope, kind, trace),
            sort=[{"ts": "desc"}], size=int(limit or 1000))
        rows = [MemoryRecord(**hit["_source"])
                for hit in found["hits"]["hits"]]
        rows.reverse()
        return rows

    async def search(self, query: str, *, scope: str | None = None, limit: int = 5,
                     trace: Trace | None = None) -> list[MemoryRecord]:
        """Ranked by the engine, which is why you would choose this backend."""
        if not query.strip():
            return await self.all(scope, limit=limit, trace=trace)
        client = await self._ensure()
        found = await client.search(
            index=self.index, query=self._query(scope, None, trace, query),
            size=int(limit))
        return [MemoryRecord(**hit["_source"]) for hit in found["hits"]["hits"]]

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        client = await self._ensure()
        await client.delete_by_query(index=self.index,
                                     query=self._query(scope, None, trace),
                                     refresh=self.refresh)
        if trace is not None or scope is None:
            docs_query = ({"term": {"namespace": trace.slug}} if trace is not None
                          else {"match_all": {}})
            await client.delete_by_query(index=self.docs_index, query=docs_query,
                                         refresh=self.refresh)

    # ---- documents ---------------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        client = await self._ensure()
        try:
            found = await client.get(index=self.docs_index,
                                     id=self.doc_key(name, trace))
        except Exception as exc:
            if type(exc).__name__ == "NotFoundError":
                return ""
            raise
        return found["_source"].get("text", "")

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        client = await self._ensure()
        await client.index(
            index=self.docs_index, id=self.doc_key(name, trace),
            document={"name": name, "text": text, "updated": time.time(),
                      "namespace": trace.slug if trace is not None else "_shared"},
            refresh=self.refresh)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._ready = False
