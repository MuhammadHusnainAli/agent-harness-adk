"""Qdrant — self-hosted or Qdrant Cloud."""

from __future__ import annotations

import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord, as_uuid

__all__ = ["QdrantStore"]

_RANGE = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


class QdrantStore(HTTPVectorStore):
    """
        QdrantStore("http://localhost:6333", collection="handbook")
        QdrantStore("https://xyz.cloud.qdrant.io", api_key="…", collection="handbook")

    Qdrant takes UUIDs or integers as ids; yours is kept in the payload and is
    what comes back.
    """

    name = "qdrant"
    batch_size = 256

    def __init__(self, url: str = "http://localhost:6333", *, collection: str = "knowledge",
                 api_key: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.url = url.rstrip("/")
        self.collection = collection
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY", "")

    def _secrets(self) -> list[str]:
        return [self.api_key]

    async def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key} if self.api_key else {}

    @property
    def _base(self) -> str:
        return f"{self.url}/collections/{self.collection}"

    async def ensure(self, dimension: int) -> None:
        if await self._request("GET", self._base, ok=(404,)) == 404:
            # 409: another replica created it between the look and the leap.
            await self._request("PUT", self._base, ok=(409,), json_body={
                "vectors": {"size": dimension, "distance": "Cosine"}})

    @staticmethod
    def _filter(wanted: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        for name, op, value in wanted:
            key = f"meta.{name}"
            if op == "$eq":
                must.append({"key": key, "match": {"value": value}})
            elif op == "$ne":
                must_not.append({"key": key, "match": {"value": value}})
            elif op == "$in":
                must.append({"key": key, "match": {"any": value}})
            else:
                must.append({"key": key, "range": {_RANGE[op]: value}})
        if not must and not must_not:
            return None
        return {**({"must": must} if must else {}),
                **({"must_not": must_not} if must_not else {})}

    async def _upsert(self, records: list[VectorRecord]) -> None:
        await self._request("PUT", f"{self._base}/points", params={"wait": "true"},
                            json_body={"points": [
                                {"id": as_uuid(r.id), "vector": r.vector,
                                 "payload": {"ref": r.id, "text": r.text, "meta": r.metadata}}
                                for r in records]})

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"query": vector, "limit": k, "with_payload": True}
        found = self._filter(wanted)
        if found:
            body["filter"] = found
        answer = await self._request("POST", f"{self._base}/points/query",
                                     json_body=body, ok=(404,))
        if answer == 404:
            # Before 1.10 the same question was asked at /points/search.
            body["vector"] = body.pop("query")
            answer = await self._request("POST", f"{self._base}/points/search",
                                         json_body=body)
        result = answer.get("result") or []
        points = result.get("points", []) if isinstance(result, dict) else result
        return [VectorHit((p.get("payload") or {}).get("ref") or str(p["id"]),
                          float(p.get("score") or 0.0),
                          (p.get("payload") or {}).get("text") or "",
                          (p.get("payload") or {}).get("meta") or {}) for p in points]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        found = self._filter(wanted) or {}
        if ids is not None:
            found.setdefault("must", []).append({"has_id": [as_uuid(i) for i in ids]})
        await self._request("POST", f"{self._base}/points/delete",
                            params={"wait": "true"}, json_body={"filter": found})

    async def count(self) -> int:
        answer = await self._request("POST", f"{self._base}/points/count",
                                     json_body={"exact": True})
        return int((answer.get("result") or {}).get("count") or 0)

    async def clear(self) -> None:
        await self._request("POST", f"{self._base}/points/delete",
                            params={"wait": "true"}, json_body={"filter": {}})
