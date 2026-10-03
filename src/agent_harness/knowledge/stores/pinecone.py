"""Pinecone."""

from __future__ import annotations

import os
from typing import Any

from ..base import HTTPVectorStore, VectorHit, VectorRecord, VectorStoreError

__all__ = ["PineconeStore"]


class PineconeStore(HTTPVectorStore):
    """
        PineconeStore(index="handbook")                        # PINECONE_API_KEY
        PineconeStore(index="handbook", namespace="acme", cloud="aws", region="us-east-1")
        PineconeStore(host="https://handbook-abc123.svc.aped-4627-b74a.pinecone.io")

    Named by `index`, its address is looked up — and the index created, as a
    serverless one, if it is not there. Given a `host`, that lookup is skipped.

    Pinecone shows a write a moment after it was made, not at once.
    """

    name = "pinecone"
    batch_size = 100
    max_k = 10_000
    api_version = "2025-01"

    def __init__(self, index: str = "", *, host: str = "", api_key: str | None = None,
                 namespace: str = "", cloud: str = "aws", region: str = "us-east-1",
                 control_url: str = "https://api.pinecone.io", **kw: Any) -> None:
        super().__init__(**kw)
        if not index and not host:
            raise VectorStoreError("Pinecone needs the index's name, or its host",
                                   store=self.name)
        self.index = index
        self.host = host.rstrip("/")
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        self.namespace = namespace
        self.cloud, self.region = cloud, region
        self.control_url = control_url.rstrip("/")
        self._dimension = 0

    def _secrets(self) -> list[str]:
        return [self.api_key]

    async def _headers(self) -> dict[str, str]:
        return {"api-key": self.api_key, "x-pinecone-api-version": self.api_version}

    async def _data(self) -> str:
        if not self.host:
            found = await self._request("GET", f"{self.control_url}/indexes/{self.index}")
            self.host = "https://" + str(found["host"]).removeprefix("https://")
        return self.host

    async def ensure(self, dimension: int) -> None:
        if self.host:
            return
        found = await self._request("GET", f"{self.control_url}/indexes/{self.index}",
                                    ok=(404,))
        if found == 404:
            found = await self._request(
                "POST", f"{self.control_url}/indexes", ok=(409,), json_body={
                    "name": self.index, "dimension": dimension, "metric": "cosine",
                    "spec": {"serverless": {"cloud": self.cloud, "region": self.region}}})
            if found == 409:
                found = await self._request(
                    "GET", f"{self.control_url}/indexes/{self.index}")
        elif found.get("dimension") not in (None, dimension):
            raise VectorStoreError(
                f"the Pinecone index {self.index!r} holds {found['dimension']}-dimension "
                f"vectors; {dimension} was asked for", store=self.name)
        self.host = "https://" + str(found["host"]).removeprefix("https://")

    @staticmethod
    def _filter(wanted: list[tuple[str, str, Any]]) -> dict[str, Any] | None:
        parts = [{name: {op: value}} for name, op, value in wanted]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else {"$and": parts}

    def _scoped(self, body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "namespace": self.namespace} if self.namespace else body

    async def _upsert(self, records: list[VectorRecord]) -> None:
        self._dimension = len(records[0].vector)
        await self._request("POST", f"{await self._data()}/vectors/upsert",
                            json_body=self._scoped({"vectors": [
                                {"id": r.id, "values": r.vector,
                                 "metadata": {**{k: v for k, v in r.metadata.items()
                                                 if v is not None}, "_text": r.text}}
                                for r in records]}))

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {"vector": vector, "topK": k, "includeMetadata": True}
        found = self._filter(wanted)
        if found:
            body["filter"] = found
        answer = await self._request("POST", f"{await self._data()}/query",
                                     json_body=self._scoped(body))
        out = []
        for match in answer.get("matches") or []:
            meta = dict(match.get("metadata") or {})
            text = meta.pop("_text", "")
            out.append(VectorHit(match["id"], float(match.get("score") or 0.0), text, meta))
        return out

    async def _matching(self, found: dict[str, Any], limit: int = 1000) -> list[str]:
        """The ids a filter matches, asked for with a vector that favours none."""
        if not self._dimension:
            stats = await self._request(
                "POST", f"{await self._data()}/describe_index_stats", json_body={})
            self._dimension = int(stats.get("dimension") or 0)
        if not self._dimension:
            raise VectorStoreError(
                "the index's dimension is not known yet, so it cannot be searched "
                "by filter alone", store=self.name)
        answer = await self._request(
            "POST", f"{await self._data()}/query", json_body=self._scoped({
                "vector": [1.0] * self._dimension, "topK": limit, "filter": found,
                "includeMetadata": False}))
        return [match["id"] for match in answer.get("matches") or []]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        found = self._filter(wanted)
        url = f"{await self._data()}/vectors/delete"
        if found and ids is None:
            try:
                await self._request("POST", url, json_body=self._scoped({"filter": found}))
                return
            except VectorStoreError as exc:
                # Serverless indexes do not delete by filter: find what matches,
                # and remove it by id, until nothing matches.
                if exc.status != 400:
                    raise
            for _ in range(10_000):
                matched = await self._matching(found)
                if not matched:
                    return
                await self._request("POST", url, json_body=self._scoped({"ids": matched}))
                if len(matched) < 1000:
                    return
            return
        if found:
            # Ids and a filter: only the ids the filter also matches.
            keep = set(await self._matching(found, 10_000))
            ids = [i for i in ids or [] if i in keep]
        for start in range(0, len(ids or []), 1000):
            await self._request("POST", url, ok=(404,), json_body=self._scoped(
                {"ids": (ids or [])[start:start + 1000]}))

    async def count(self) -> int:
        stats = await self._request("POST", f"{await self._data()}/describe_index_stats",
                                    json_body={})
        if self.namespace:
            space = (stats.get("namespaces") or {}).get(self.namespace) or {}
            return int(space.get("vectorCount") or 0)
        return int(stats.get("totalVectorCount") or 0)

    async def clear(self) -> None:
        await self._request("POST", f"{await self._data()}/vectors/delete",
                            json_body=self._scoped({"deleteAll": True}), ok=(404,))
