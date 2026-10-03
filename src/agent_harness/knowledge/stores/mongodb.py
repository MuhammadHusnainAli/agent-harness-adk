"""MongoDB Atlas Vector Search — Atlas, or the Atlas local deployment."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ...errors import ConfigurationError
from ..base import (
    ALWAYS_FILTERABLE,
    VectorHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)

__all__ = ["MongoVectorStore"]


class MongoVectorStore(VectorStore):
    """
        MongoVectorStore("mongodb+srv://user:pass@cluster.mongodb.net", database="kb",
                         collection="handbook", filterable=["source", "year"])

    Needs `pymongo` 4.9 or later (`pip install pymongo`), and a deployment with
    Atlas Search: Atlas itself, or `mongodb/mongodb-atlas-local`. A plain
    community server has no vector search.

    Atlas filters on the fields its index was told about: name them in
    `filterable`. A new index takes a few seconds to become queryable; `ensure`
    waits for it.
    """

    name = "mongodb"
    batch_size = 500

    def __init__(self, url: str, *, database: str = "knowledge",
                 collection: str = "vectors", index: str = "vector_index",
                 filterable: list[str] | None = None, ready_timeout: float = 120.0,
                 **options: Any) -> None:
        self.url = url
        self.database, self.collection, self.index = database, collection, index
        self.filterable = list(dict.fromkeys([*ALWAYS_FILTERABLE, *(filterable or [])]))
        self.ready_timeout = ready_timeout
        self.options = options
        self._client: Any = None
        self._loop: Any = None

    async def _coll(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            try:
                from pymongo import AsyncMongoClient
            except ImportError as exc:
                raise ConfigurationError(
                    "MongoVectorStore needs pymongo 4.9 or later — pip install pymongo"
                ) from exc
            self._client = AsyncMongoClient(self.url, **self.options)
            self._loop = loop
        return self._client[self.database][self.collection]

    async def _indexes(self, coll: Any) -> list[dict[str, Any]]:
        cursor = await coll.list_search_indexes(self.index)
        return [row async for row in cursor]

    async def ensure(self, dimension: int) -> None:
        coll = await self._coll()
        try:
            found = await self._indexes(coll)
        except Exception:
            found = []
        if not found:
            from pymongo.operations import SearchIndexModel

            if self.collection not in await coll.database.list_collection_names():
                await coll.database.create_collection(self.collection)
            fields: list[dict[str, Any]] = [
                {"type": "vector", "path": "vector", "numDimensions": dimension,
                 "similarity": "cosine"}]
            fields += [{"type": "filter", "path": f"meta.{name}"}
                       for name in self.filterable]
            try:
                await coll.create_search_index(SearchIndexModel(
                    definition={"fields": fields}, name=self.index, type="vectorSearch"))
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise VectorStoreError(
                        f"mongodb could not create the vector index: {exc}"[:400],
                        store=self.name) from None
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            found = await self._indexes(coll)
            if found and (found[0].get("queryable") or found[0].get("status") == "READY"):
                return
            await asyncio.sleep(1.0)
        raise VectorStoreError(
            f"the vector index {self.index!r} was not ready after "
            f"{self.ready_timeout:.0f}s", store=self.name)

    def _filter(self, wanted: list[tuple[str, str, Any]]) -> dict[str, Any]:
        parts = []
        for name, op, value in wanted:
            if name not in self.filterable:
                raise VectorStoreError(
                    f"{name!r} is not filterable in this Atlas index — name it in "
                    f"filterable=[...]. Filterable now: "
                    f"{', '.join(self.filterable) or 'nothing'}", store=self.name)
            parts.append({f"meta.{name}": {op: value}})
        if not parts:
            return {}
        return parts[0] if len(parts) == 1 else {"$and": parts}

    async def _upsert(self, records: list[VectorRecord]) -> None:
        from pymongo import ReplaceOne

        coll = await self._coll()
        await coll.bulk_write([
            ReplaceOne({"_id": r.id}, {"_id": r.id, "vector": [float(x) for x in r.vector],
                                       "text": r.text, "meta": r.metadata}, upsert=True)
            for r in records], ordered=False)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        coll = await self._coll()
        stage: dict[str, Any] = {"index": self.index, "path": "vector",
                                 "queryVector": [float(x) for x in vector],
                                 "numCandidates": min(10_000, max(100, k * 15)),
                                 "limit": k}
        found = self._filter(wanted)
        if found:
            stage["filter"] = found
        try:
            cursor = await coll.aggregate([
                {"$vectorSearch": stage},
                {"$project": {"text": 1, "meta": 1,
                              "score": {"$meta": "vectorSearchScore"}}}])
            rows = [row async for row in cursor]
        except Exception as exc:
            raise VectorStoreError(f"mongodb: {exc}"[:400], store=self.name) from None
        # Atlas scores cosine as (1 + cosine) / 2.
        return [VectorHit(str(row["_id"]), 2.0 * float(row.get("score") or 0.0) - 1.0,
                          row.get("text") or "", row.get("meta") or {}) for row in rows]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        coll = await self._coll()
        # A plain delete filters on any field, declared or not.
        query: dict[str, Any] = {}
        if ids is not None:
            query["_id"] = {"$in": ids}
        for name, op, value in wanted:
            query.setdefault(f"meta.{name}", {})[op] = value
        await coll.delete_many(query)

    async def count(self) -> int:
        return int(await (await self._coll()).count_documents({}))

    async def clear(self) -> None:
        await (await self._coll()).delete_many({})

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._loop is asyncio.get_running_loop():
            await client.close()
