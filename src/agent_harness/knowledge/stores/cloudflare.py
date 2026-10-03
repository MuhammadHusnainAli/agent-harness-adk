"""Cloudflare Vectorize."""

from __future__ import annotations

import json
import os
from typing import Any

from ..base import (
    ALWAYS_FILTERABLE,
    HTTPVectorStore,
    VectorHit,
    VectorRecord,
    VectorStoreError,
)

__all__ = ["VectorizeStore"]


class VectorizeStore(HTTPVectorStore):
    """
        VectorizeStore(account_id="…", index="handbook", api_token="…",
                       filterable={"source": "string", "year": "number"})

    Vectorize filters only on metadata it has an index for: `filterable` names
    them (`string`, `number` or `boolean`) and they are created with the index.
    It applies writes a few seconds after accepting them, returns at most 20
    results with their metadata, and cannot delete by filter — so removing a
    document means removing its ids, which `KnowledgeBase` does.
    """

    name = "cloudflare-vectorize"
    batch_size = 500
    #: With metadata returned, Vectorize answers with at most this many.
    max_k = 20

    def __init__(self, *, account_id: str | None = None, index: str = "knowledge",
                 api_token: str | None = None, namespace: str = "",
                 filterable: dict[str, str] | None = None,
                 url: str = "https://api.cloudflare.com/client/v4", **kw: Any) -> None:
        super().__init__(**kw)
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not self.account_id:
            raise VectorStoreError("Vectorize needs account_id= or $CLOUDFLARE_ACCOUNT_ID",
                                   store=self.name)
        self.index = index
        self.namespace = namespace
        self.filterable = {**dict.fromkeys(ALWAYS_FILTERABLE[:2], "string"),
                           **{k: v.lower() for k, v in (filterable or {}).items()}}
        self._root = f"{url.rstrip('/')}/accounts/{self.account_id}/vectorize/v2/indexes"

    def _secrets(self) -> list[str]:
        return [self.api_token]

    async def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_token}"}

    async def _call(self, method: str, path: str, **kw: Any) -> Any:
        answer = await self._request(method, f"{self._root}{path}", **kw)
        if isinstance(answer, dict) and answer.get("success") is False:
            raise VectorStoreError(
                f"vectorize: {json.dumps(answer.get('errors'))[:300]}", store=self.name)
        return answer

    async def ensure(self, dimension: int) -> None:
        if await self._call("GET", f"/{self.index}", ok=(404, 410)) not in (404, 410):
            return
        await self._call("POST", "", ok=(409,), json_body={
            "name": self.index, "config": {"dimensions": dimension, "metric": "cosine"}})
        for name, kind in self.filterable.items():
            await self._call("POST", f"/{self.index}/metadata_index/create", ok=(409,),
                             json_body={"propertyName": name, "indexType": kind})

    def _filter(self, wanted: list[tuple[str, str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, op, value in wanted:
            if name not in self.filterable:
                raise VectorStoreError(
                    f"{name!r} has no metadata index in Vectorize — declare it: "
                    f"filterable={{{name!r}: 'string'}}. Filterable now: "
                    f"{', '.join(self.filterable) or 'nothing'}", store=self.name)
            out.setdefault(name, {})[op] = value
        return out

    async def _upsert(self, records: list[VectorRecord]) -> None:
        lines = []
        for r in records:
            row: dict[str, Any] = {"id": r.id, "values": r.vector,
                                   "metadata": {**r.metadata, "_text": r.text}}
            if self.namespace:
                row["namespace"] = self.namespace
            lines.append(json.dumps(row, ensure_ascii=False))
        await self._call("POST", f"/{self.index}/upsert",
                         content=("\n".join(lines) + "\n").encode(),
                         headers={"content-type": "application/x-ndjson"})

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"vector": vector, "topK": k, "returnMetadata": "all",
                                "returnValues": False}
        found = self._filter(wanted)
        if found:
            body["filter"] = found
        if self.namespace:
            body["namespace"] = self.namespace
        answer = await self._call("POST", f"/{self.index}/query", json_body=body)
        out = []
        for match in (answer.get("result") or {}).get("matches") or []:
            meta = dict(match.get("metadata") or {})
            text = meta.pop("_text", "")
            out.append(VectorHit(match["id"], float(match.get("score") or 0.0), text, meta))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if ids is None or wanted:
            raise VectorStoreError(
                "Vectorize deletes by id only — it has no delete-by-filter",
                store=self.name)
        for start in range(0, len(ids), 1000):
            await self._call("POST", f"/{self.index}/delete_by_ids",
                             json_body={"ids": ids[start:start + 1000]})

    async def count(self) -> int:
        answer = await self._call("GET", f"/{self.index}/info")
        result = answer.get("result") or {}
        return int(result.get("vectorCount") or result.get("vectorsCount") or 0)
