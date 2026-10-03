"""Vertex AI Vector Search."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..base import (
    HTTPVectorStore,
    VectorHit,
    VectorRecord,
    VectorStoreError,
    holds,
    normalise,
)

__all__ = ["VertexVectorStore", "SQLitePayloads"]

_NUMERIC = {"$eq": "EQUAL", "$ne": "NOT_EQUAL", "$gt": "GREATER", "$gte": "GREATER_EQUAL",
            "$lt": "LESS", "$lte": "LESS_EQUAL"}


class SQLitePayloads:
    """The text and metadata of each record, in a SQLite file.

    Vertex AI Vector Search keeps vectors and the values they are filtered on,
    and nothing else — so what a record *says* has to live somewhere. This is
    the default place. For several replicas, give the store an object with the
    same three methods over a database they share.
    """

    def __init__(self, path: str | Path = "vertex-payloads.db") -> None:
        self.path = str(path)
        self._db: sqlite3.Connection | None = None

    def _open(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._db.execute("CREATE TABLE IF NOT EXISTS payloads "
                             "(id TEXT PRIMARY KEY, text TEXT, metadata TEXT)")
            self._db.commit()
        return self._db

    async def put(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        def write() -> None:
            db = self._open()
            db.executemany(
                "INSERT INTO payloads VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "text=excluded.text, metadata=excluded.metadata",
                [(key, text, json.dumps(meta, default=str)) for key, text, meta in rows])
            db.commit()

        await asyncio.to_thread(write)

    async def get(self, ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        def read() -> dict[str, tuple[str, dict[str, Any]]]:
            out = {}
            for start in range(0, len(ids), 500):
                part = ids[start:start + 500]
                marks = ",".join("?" * len(part))
                for key, text, meta in self._open().execute(
                        f"SELECT id, text, metadata FROM payloads WHERE id IN ({marks})",  # noqa: S608
                        part):
                    out[key] = (text or "", json.loads(meta or "{}"))
            return out

        return await asyncio.to_thread(read)

    async def delete(self, ids: list[str]) -> None:
        def remove() -> None:
            db = self._open()
            db.executemany("DELETE FROM payloads WHERE id = ?", [(i,) for i in ids])
            db.commit()

        await asyncio.to_thread(remove)

    async def all(self) -> dict[str, tuple[str, dict[str, Any]]]:
        def read() -> dict[str, tuple[str, dict[str, Any]]]:
            return {key: (text or "", json.loads(meta or "{}")) for key, text, meta in
                    self._open().execute("SELECT id, text, metadata FROM payloads")}

        return await asyncio.to_thread(read)


class VertexVectorStore(HTTPVectorStore):
    """
        VertexVectorStore(project="my-project", region="us-central1",
                          index="1234567890", endpoint="9876543210",
                          deployed_index="handbook_v1",
                          endpoint_domain="1234.us-central1-56789.vdb.vertexai.goog")

    The index is one you created for *streaming updates* and deployed to an
    endpoint — that takes the better part of an hour and is done in the Cloud
    console or with `gcloud`, not here. Create it with the dot-product distance:
    vectors are sent at unit length, so a dot product is a cosine.

    Vector Search stores no text. Each record's text and metadata go to
    `payloads` (a SQLite file by default); what is filtered on is also sent to
    Vertex as restricts.
    """

    name = "vertex-vector-search"
    batch_size = 100

    def __init__(self, *, project: str, region: str, index: str, endpoint: str,
                 deployed_index: str, endpoint_domain: str, payloads: Any = None,
                 token: str | None = None, auth: Any = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.project, self.region = project, region
        self.index, self.endpoint = index, endpoint
        self.deployed_index = deployed_index
        self.endpoint_domain = endpoint_domain.removeprefix("https://").rstrip("/")
        self.payloads = payloads if payloads is not None else SQLitePayloads()
        self._auth = auth
        self._token = token

    async def _headers(self) -> dict[str, str]:
        if self._auth is None:
            from ...llm_providers.vertex import GoogleAuth

            self._auth = GoogleAuth(self._token)
        return {"authorization": f"Bearer {await self._auth.token()}"}

    @property
    def _index_url(self) -> str:
        return (f"https://{self.region}-aiplatform.googleapis.com/v1/projects/"
                f"{self.project}/locations/{self.region}/indexes/{self.index}")

    async def ensure(self, dimension: int) -> None:
        found = await self._request("GET", self._index_url)
        held = ((found.get("metadata") or {}).get("config") or {}).get("dimensions")
        if held and int(held) != dimension:
            raise VectorStoreError(
                f"the Vertex index holds {held}-dimension vectors; {dimension} was "
                "asked for", store=self.name)

    @staticmethod
    def _restricts(metadata: dict[str, Any]) -> dict[str, Any]:
        tokens, numbers = [], []
        for key, value in metadata.items():
            if isinstance(value, bool):
                tokens.append({"namespace": key, "allowList": [str(value).lower()]})
            elif isinstance(value, int):
                numbers.append({"namespace": key, "valueInt": str(value)})
            elif isinstance(value, float):
                numbers.append({"namespace": key, "valueDouble": value})
            elif isinstance(value, str):
                tokens.append({"namespace": key, "allowList": [value]})
            elif isinstance(value, list) and value:
                tokens.append({"namespace": key, "allowList": [str(v) for v in value]})
        return {**({"restricts": tokens} if tokens else {}),
                **({"numericRestricts": numbers} if numbers else {})}

    def _wanted(self, wanted: list[tuple[str, str, Any]]) -> dict[str, Any]:
        tokens, numbers = [], []
        for name, op, value in wanted:
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if op == "$in":
                tokens.append({"namespace": name, "allowList": [
                    str(v).lower() if isinstance(v, bool) else str(v) for v in value]})
            elif numeric:
                key = "valueInt" if isinstance(value, int) else "valueDouble"
                numbers.append({"namespace": name, "op": _NUMERIC[op],
                                key: str(value) if isinstance(value, int) else value})
            elif op in ("$eq", "$ne"):
                written = str(value).lower() if isinstance(value, bool) else str(value)
                tokens.append({"namespace": name,
                               "allowList" if op == "$eq" else "denyList": [written]})
            else:
                raise VectorStoreError(
                    f"a range on {name!r} needs a number, not {value!r}", store=self.name)
        return {**({"restricts": tokens} if tokens else {}),
                **({"numericRestricts": numbers} if numbers else {})}

    async def _upsert(self, records: list[VectorRecord]) -> None:
        # The text first: a vector with nothing to say is worse than text with
        # no vector yet.
        await self.payloads.put([(r.id, r.text, r.metadata) for r in records])
        await self._request("POST", f"{self._index_url}:upsertDatapoints", json_body={
            "datapoints": [{"datapointId": r.id, "featureVector": normalise(r.vector),
                            **self._restricts(r.metadata)} for r in records]})

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        answer = await self._request(
            "POST",
            f"https://{self.endpoint_domain}/v1/projects/{self.project}/locations/"
            f"{self.region}/indexEndpoints/{self.endpoint}:findNeighbors",
            json_body={"deployedIndexId": self.deployed_index, "queries": [{
                "neighborCount": k,
                "datapoint": {"featureVector": normalise(vector), **self._wanted(wanted)}}]})
        nearest = (answer.get("nearestNeighbors") or [{}])[0].get("neighbors") or []
        ids = [(n.get("datapoint") or {}).get("datapointId", "") for n in nearest]
        held = await self.payloads.get([i for i in ids if i])
        return [VectorHit(key, float(n.get("distance") or 0.0), *held.get(key, ("", {})))
                for key, n in zip(ids, nearest, strict=False) if key]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if wanted:
            # Vertex removes by id; the payloads know which ids a filter means.
            held = await self.payloads.all()
            ids = [key for key, (_, meta) in held.items()
                   if (ids is None or key in ids) and holds(meta, wanted)]
        for start in range(0, len(ids or []), 1000):
            part = (ids or [])[start:start + 1000]
            await self._request("POST", f"{self._index_url}:removeDatapoints",
                                json_body={"datapointIds": part})
            await self.payloads.delete(part)

    async def count(self) -> int:
        found = await self._request("GET", self._index_url)
        return int((found.get("indexStats") or {}).get("vectorsCount") or 0)
