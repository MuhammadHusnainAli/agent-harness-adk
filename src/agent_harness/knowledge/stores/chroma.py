"""Chroma — a Chroma server, or Chroma Cloud."""

from __future__ import annotations

import json
import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord

__all__ = ["ChromaStore"]


class ChromaStore(HTTPVectorStore):
    """
        ChromaStore("http://localhost:8000", collection="handbook")
        ChromaStore("https://api.trychroma.com", api_key="…", tenant="…", database="…")

    Chroma's metadata is flat and holds no lists, so a list is kept as its
    members joined by a separator and is not filterable there.
    """

    name = "chroma"
    batch_size = 500

    def __init__(self, url: str = "http://localhost:8000", *, collection: str = "knowledge",
                 tenant: str = "default_tenant", database: str = "default_database",
                 api_key: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.url = url.rstrip("/")
        self.collection = collection
        self.tenant, self.database = tenant, database
        self.api_key = api_key or os.environ.get("CHROMA_API_KEY", "")
        self._id = ""

    def _secrets(self) -> list[str]:
        return [self.api_key]

    async def _headers(self) -> dict[str, str]:
        return {"x-chroma-token": self.api_key} if self.api_key else {}

    @property
    def _collections(self) -> str:
        return (f"{self.url}/api/v2/tenants/{self.tenant}/databases/{self.database}"
                "/collections")

    async def _collection(self) -> str:
        if not self._id:
            answer = await self._request("POST", self._collections, json_body={
                "name": self.collection, "get_or_create": True,
                "configuration": {"hnsw": {"space": "cosine"}}})
            self._id = answer["id"]
        return f"{self._collections}/{self._id}"

    async def ensure(self, dimension: int) -> None:
        await self._collection()

    @staticmethod
    def _where(wanted: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
        parts = [{name: {op: value}} for name, op, value in wanted]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"$and": parts}

    @staticmethod
    def _flat(metadata: dict[str, Any]) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, dict)):
                out[f"{key}__json"] = json.dumps(value, default=str, ensure_ascii=False)
            else:
                out[key] = value
        return out or None

    @staticmethod
    def _unflat(metadata: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if key.endswith("__json"):
                try:
                    out[key[:-6]] = json.loads(value)
                    continue
                except (ValueError, TypeError):
                    pass
            out[key] = value
        return out

    async def _upsert(self, records: list[VectorRecord]) -> None:
        await self._request("POST", f"{await self._collection()}/upsert", json_body={
            "ids": [r.id for r in records], "embeddings": [r.vector for r in records],
            "documents": [r.text for r in records],
            "metadatas": [self._flat(r.metadata) for r in records]})

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"query_embeddings": [vector], "n_results": k,
                                "include": ["documents", "metadatas", "distances"]}
        where = self._where(wanted)
        if where:
            body["where"] = where
        answer = await self._request("POST", f"{await self._collection()}/query",
                                     json_body=body)
        ids = (answer.get("ids") or [[]])[0]
        distances = (answer.get("distances") or [[]])[0]
        texts = (answer.get("documents") or [[]])[0]
        metas = (answer.get("metadatas") or [[]])[0]
        return [VectorHit(key, 1.0 - float(distances[n]), texts[n] or "",
                          self._unflat(metas[n])) for n, key in enumerate(ids)]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        body: dict[str, Any] = {}
        if ids is not None:
            body["ids"] = ids
        where = self._where(wanted)
        if where:
            body["where"] = where
        await self._request("POST", f"{await self._collection()}/delete", json_body=body)

    async def count(self) -> int:
        return int(await self._request("GET", f"{await self._collection()}/count"))

    async def clear(self) -> None:
        await self._request("DELETE", f"{self._collections}/{self.collection}", ok=(404,))
        self._id = ""
        await self._collection()
