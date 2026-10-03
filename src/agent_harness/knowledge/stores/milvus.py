"""Milvus — self-hosted, or Zilliz Cloud."""

from __future__ import annotations

import json
import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord, VectorStoreError, literal

__all__ = ["MilvusStore"]

_OPS = {"$eq": "==", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class MilvusStore(HTTPVectorStore):
    """
        MilvusStore("http://localhost:19530", collection="handbook")
        MilvusStore("https://in03-x.api.gcp-us-west1.zillizcloud.com", token="…",
                    collection="handbook")

    Spoken to over Milvus's REST API (v2), so no client library is needed.
    """

    name = "milvus"
    batch_size = 500

    def __init__(self, url: str = "http://localhost:19530", *, collection: str = "knowledge",
                 token: str | None = None, database: str = "", **kw: Any) -> None:
        super().__init__(**kw)
        self.url = url.rstrip("/")
        self.collection = collection
        self.database = database
        self.token = token or os.environ.get("MILVUS_TOKEN", "")

    def _secrets(self) -> list[str]:
        return [self.token]

    async def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token}"} if self.token else {}

    async def _call(self, path: str, **body: Any) -> Any:
        payload = {"collectionName": self.collection, **body}
        if self.database:
            payload["dbName"] = self.database
        answer = await self._request("POST", f"{self.url}/v2/vectordb/{path}",
                                     json_body=payload)
        # Milvus answers 200 to everything and says how it went in the body.
        if answer.get("code") not in (0, 200, None):
            raise VectorStoreError(
                f"milvus: {path} → {answer.get('code')}: "
                f"{self._redact(str(answer.get('message')))[:300]}", store=self.name)
        return answer.get("data")

    async def ensure(self, dimension: int) -> None:
        if (await self._call("collections/has") or {}).get("has"):
            return
        await self._call(
            "collections/create",
            schema={"autoId": False, "enableDynamicField": True, "fields": [
                {"fieldName": "id", "dataType": "VarChar", "isPrimary": True,
                 "elementTypeParams": {"max_length": "512"}},
                {"fieldName": "vector", "dataType": "FloatVector",
                 "elementTypeParams": {"dim": str(dimension)}},
                {"fieldName": "text", "dataType": "VarChar",
                 "elementTypeParams": {"max_length": "65535"}},
                {"fieldName": "meta", "dataType": "JSON"}]},
            indexParams=[{"fieldName": "vector", "indexName": "vector",
                          "metricType": "COSINE", "indexType": "AUTOINDEX"}])

    @staticmethod
    def _expr(wanted: list[tuple[str, str, Any]], ids: list[str] | None = None) -> str:
        parts = []
        if ids is not None:
            parts.append(f"id in [{', '.join(literal(i) for i in ids)}]")
        for name, op, value in wanted:
            field = f'meta["{name}"]'
            if op == "$in":
                parts.append(f"{field} in [{', '.join(literal(v) for v in value)}]")
            else:
                parts.append(f"{field} {_OPS[op]} {literal(value)}")
        return " and ".join(parts)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        await self._call("entities/upsert", data=[
            {"id": r.id, "vector": r.vector, "text": r.text[:65000], "meta": r.metadata}
            for r in records])

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"data": [vector], "annsField": "vector", "limit": k,
                                "outputFields": ["id", "text", "meta"],
                                # Read what was just written, not a moment ago.
                                "consistencyLevel": "Strong"}
        expr = self._expr(wanted)
        if expr:
            body["filter"] = expr
        out = []
        for row in await self._call("entities/search", **body) or []:
            meta = row.get("meta") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except ValueError:
                    meta = {}
            out.append(VectorHit(str(row.get("id", "")), float(row.get("distance") or 0.0),
                                 row.get("text") or "", meta))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        await self._call("entities/delete", filter=self._expr(wanted, ids))

    async def count(self) -> int:
        rows = await self._call("entities/query", filter="", outputFields=["count(*)"],
                                consistencyLevel="Strong") or [{}]
        return int(rows[0].get("count(*)") or 0)

    async def clear(self) -> None:
        await self._call("collections/drop")
