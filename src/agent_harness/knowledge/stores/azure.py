"""Azure AI Search."""

from __future__ import annotations

import base64
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

__all__ = ["AzureAISearchStore"]

_OPS = {"$eq": "eq", "$ne": "ne", "$gt": "gt", "$gte": "ge", "$lt": "lt", "$lte": "le"}
_TYPES = {"string": "Edm.String", "str": "Edm.String", "int": "Edm.Int64",
          "integer": "Edm.Int64", "number": "Edm.Double", "float": "Edm.Double",
          "bool": "Edm.Boolean", "boolean": "Edm.Boolean",
          "strings": "Collection(Edm.String)", "list": "Collection(Edm.String)"}


class AzureAISearchStore(HTTPVectorStore):
    """
        AzureAISearchStore("https://my-service.search.windows.net", index="handbook",
                           api_key="…", filterable={"source": "string", "year": "int"})

    An Azure index has a fixed schema, so the metadata that is filtered on is
    declared: `filterable` maps each field to `string`, `int`, `number`, `bool`
    or `strings` (a list). All metadata is kept and returned either way.

    `token=` (a string, or a function returning one) signs in with Microsoft
    Entra ID instead of a key.
    """

    name = "azure-ai-search"
    batch_size = 500
    api_version = "2024-07-01"

    def __init__(self, endpoint: str, *, index: str = "knowledge",
                 api_key: str | None = None, token: Any = None,
                 filterable: dict[str, str] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.endpoint = endpoint.rstrip("/")
        self.index = index
        self.api_key = api_key or os.environ.get("AZURE_SEARCH_API_KEY", "")
        self.token = token
        self.filterable: dict[str, str] = dict.fromkeys(ALWAYS_FILTERABLE, "Edm.String")
        for key, kind in (filterable or {}).items():
            if kind.lower() not in _TYPES:
                raise VectorStoreError(
                    f"{key!r} is {kind!r}; a filterable field is one of "
                    f"{', '.join(sorted(set(_TYPES)))}", store=self.name)
            self.filterable[key] = _TYPES[kind.lower()]

    def _secrets(self) -> list[str]:
        return [self.api_key, self.token if isinstance(self.token, str) else ""]

    async def _headers(self) -> dict[str, str]:
        if self.token is not None:
            import inspect

            token = self.token() if callable(self.token) else self.token
            if inspect.isawaitable(token):
                token = await token
            return {"authorization": f"Bearer {token}"}
        return {"api-key": self.api_key}

    def _url(self, path: str = "") -> str:
        return f"{self.endpoint}/indexes/{self.index}{path}"

    @property
    def _version(self) -> dict[str, str]:
        return {"api-version": self.api_version}

    @staticmethod
    def _key(record_id: str) -> str:
        """A document key: letters, digits, dashes, underscores and equals signs."""
        return base64.urlsafe_b64encode(record_id.encode()).decode()

    async def ensure(self, dimension: int) -> None:
        if await self._request("GET", self._url(), params=self._version, ok=(404,)) != 404:
            return
        fields: list[dict[str, Any]] = [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            {"name": "ref", "type": "Edm.String", "filterable": True},
            {"name": "text", "type": "Edm.String", "searchable": True},
            {"name": "meta", "type": "Edm.String"},
            {"name": "vector", "type": "Collection(Edm.Single)", "searchable": True,
             "dimensions": dimension, "vectorSearchProfile": "default"}]
        fields += [{"name": f"m_{name}", "type": kind, "filterable": True}
                   for name, kind in self.filterable.items()]
        await self._request("PUT", self._url(), params=self._version, json_body={
            "name": self.index, "fields": fields,
            "vectorSearch": {
                "algorithms": [{"name": "hnsw", "kind": "hnsw",
                                "hnswParameters": {"metric": "cosine"}}],
                "profiles": [{"name": "default", "algorithm": "hnsw"}]}})

    def _odata(self, wanted: list[tuple[str, str, Any]], ids: list[str] | None = None) -> str:
        def lit(value: Any) -> str:
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, (int, float)):
                return repr(value)
            return "'" + str(value).replace("'", "''") + "'"

        parts = []
        if ids is not None:
            parts.append("search.in(id, '" + ",".join(self._key(i) for i in ids) + "', ',')")
        for name, op, value in wanted:
            kind = self.filterable.get(name)
            if kind is None:
                raise VectorStoreError(
                    f"{name!r} is not filterable in this Azure index — declare it: "
                    f"filterable={{{name!r}: 'string'}}. Filterable now: "
                    f"{', '.join(self.filterable) or 'nothing'}", store=self.name)
            field = f"m_{name}"
            if kind.startswith("Collection"):
                values = value if op == "$in" else [value]
                test = " or ".join(f"t eq {lit(v)}" for v in values)
                clause = f"{field}/any(t: {test})"
                parts.append(f"not {clause}" if op == "$ne" else clause)
            elif op == "$in":
                parts.append("(" + " or ".join(f"{field} eq {lit(v)}" for v in value) + ")")
            else:
                parts.append(f"{field} {_OPS[op]} {lit(value)}")
        return " and ".join(parts)

    async def _index(self, actions: list[dict[str, Any]]) -> None:
        answer = await self._request("POST", self._url("/docs/index"),
                                     params=self._version, json_body={"value": actions})
        failed = [row for row in answer.get("value") or []
                  if not row.get("status") and row.get("statusCode") != 404]
        if failed:
            raise VectorStoreError(
                f"azure refused {len(failed)} records: "
                f"{json.dumps(failed[0])[:300]}", store=self.name)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        actions = []
        for r in records:
            row: dict[str, Any] = {
                "@search.action": "mergeOrUpload", "id": self._key(r.id), "ref": r.id,
                "text": r.text, "vector": r.vector,
                "meta": json.dumps(r.metadata, default=str, ensure_ascii=False)}
            for name, kind in self.filterable.items():
                value = r.metadata.get(name)
                if kind.startswith("Collection"):
                    value = [str(v) for v in (value if isinstance(value, list)
                                              else [] if value is None else [value])]
                row[f"m_{name}"] = value
            actions.append(row)
        await self._index(actions)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        body: dict[str, Any] = {
            "select": "ref,text,meta", "top": k,
            "vectorQueries": [{"kind": "vector", "vector": vector, "fields": "vector",
                               "k": k}]}
        found = self._odata(wanted)
        if found:
            body["filter"] = found
        answer = await self._request("POST", self._url("/docs/search"),
                                     params=self._version, json_body=body)
        out = []
        for row in answer.get("value") or []:
            try:
                meta = json.loads(row.get("meta") or "{}")
            except ValueError:
                meta = {}
            score = float(row.get("@search.score") or 0.0)
            # Azure scores cosine as 1 / (2 - cosine).
            out.append(VectorHit(row.get("ref") or "", 2.0 - 1.0 / score if score else 0.0,
                                 row.get("text") or "", meta))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if ids is not None and not wanted:
            for start in range(0, len(ids), 500):
                await self._index([{"@search.action": "delete", "id": self._key(i)}
                                   for i in ids[start:start + 500]])
            return
        found = self._odata(wanted, ids)
        # There is no delete-by-filter: find what matches, then remove it by key.
        for _ in range(1000):
            answer = await self._request(
                "POST", self._url("/docs/search"), params=self._version,
                json_body={"select": "id", "top": 500, "filter": found, "search": "*"})
            keys = [row["id"] for row in answer.get("value") or []]
            if not keys:
                return
            await self._index([{"@search.action": "delete", "id": key} for key in keys])
            if len(keys) < 500:
                return

    async def count(self) -> int:
        answer = await self._request("GET", self._url("/docs/$count"),
                                     params=self._version)
        return int(answer if isinstance(answer, int) else answer.get("text") or 0)

    async def clear(self) -> None:
        await self._request("DELETE", self._url(), params=self._version, ok=(404,))
