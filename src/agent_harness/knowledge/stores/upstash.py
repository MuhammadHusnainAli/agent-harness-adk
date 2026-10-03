"""Upstash Vector."""

from __future__ import annotations

import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord, holds, literal

__all__ = ["UpstashStore"]

_OPS = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class UpstashStore(HTTPVectorStore):
    """
        UpstashStore("https://x-y-z-vector.upstash.io", token="…", namespace="handbook")

    The index is created in the Upstash console, with the dimension of your
    embedding model and the cosine metric.
    """

    name = "upstash"
    batch_size = 500

    def __init__(self, url: str | None = None, *, token: str | None = None,
                 namespace: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self.url = (url or os.environ.get("UPSTASH_VECTOR_REST_URL", "")).rstrip("/")
        self.token = token or os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "")
        self.namespace = namespace

    def _secrets(self) -> list[str]:
        return [self.token]

    async def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token}"}

    def _path(self, action: str) -> str:
        return f"{self.url}/{action}" + (f"/{self.namespace}" if self.namespace else "")

    async def ensure(self, dimension: int) -> None:
        await self._request("GET", f"{self.url}/info")      # there, and the key works

    @staticmethod
    def _expr(wanted: list[tuple[str, str, Any]]) -> str:
        parts = []
        for name, op, value in wanted:
            if op == "$in":
                parts.append(f"{name} IN ({', '.join(literal(v).replace(chr(34), chr(39)) for v in value)})")
            else:
                written = literal(value)
                if isinstance(value, str):
                    written = "'" + value.replace("'", "\\'") + "'"
                parts.append(f"{name} {_OPS[op]} {written}")
        return " AND ".join(parts)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        await self._request("POST", self._path("upsert"), json_body=[
            {"id": r.id, "vector": r.vector, "metadata": r.metadata, "data": r.text}
            for r in records])

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"vector": vector, "topK": k, "includeMetadata": True,
                                "includeData": True}
        expr = self._expr(wanted)
        if expr:
            body["filter"] = expr
        answer = await self._request("POST", self._path("query"), json_body=body)
        # Upstash scores cosine as (1 + cosine) / 2.
        return [VectorHit(str(row["id"]), 2.0 * float(row.get("score") or 0.0) - 1.0,
                          row.get("data") or "", row.get("metadata") or {})
                for row in answer.get("result") or []]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        expr = self._expr(wanted)
        if ids is not None and expr:
            # Ids and a filter together: fetch the ids, keep those that match.
            fetched = await self._request("POST", self._path("fetch"), json_body={
                "ids": ids, "includeMetadata": True})
            ids = [str(row["id"]) for row in fetched.get("result") or []
                   if row and holds(row.get("metadata") or {}, wanted)]
            expr = ""
        body: dict[str, Any] = {"filter": expr} if expr else {"ids": ids or []}
        await self._request("POST", self._path("delete"), json_body=body)

    async def count(self) -> int:
        info = (await self._request("GET", f"{self.url}/info")).get("result") or {}
        if self.namespace:
            space = (info.get("namespaces") or {}).get(self.namespace) or {}
            return int(space.get("vectorCount") or 0)
        return int(info.get("vectorCount") or 0)

    async def clear(self) -> None:
        await self._request("POST", self._path("reset"))
