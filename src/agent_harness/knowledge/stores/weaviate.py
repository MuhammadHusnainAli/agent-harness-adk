"""Weaviate — self-hosted or Weaviate Cloud."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..base import (
    HTTPVectorStore,
    VectorHit,
    VectorRecord,
    VectorStoreError,
    as_uuid,
)

__all__ = ["WeaviateStore"]

_OPS = {"$eq": "Equal", "$ne": "NotEqual", "$gt": "GreaterThan", "$gte": "GreaterThanEqual",
        "$lt": "LessThan", "$lte": "LessThanEqual"}
_NAME = re.compile(r"[^0-9A-Za-z_]")


class WeaviateStore(HTTPVectorStore):
    """
        WeaviateStore("http://localhost:8080", collection="Handbook")
        WeaviateStore("https://xyz.weaviate.cloud", api_key="…", collection="Handbook")

    Metadata is kept twice: whole, as it was given, and field by field (as
    `m_<name>`) so it can be filtered on.
    """

    name = "weaviate"
    batch_size = 200

    def __init__(self, url: str = "http://localhost:8080", *, collection: str = "Knowledge",
                 api_key: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.url = url.rstrip("/")
        # Weaviate's class names start with a capital.
        cleaned = _NAME.sub("_", collection) or "Knowledge"
        self.collection = cleaned[0].upper() + cleaned[1:]
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY", "")

    def _secrets(self) -> list[str]:
        return [self.api_key]

    async def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def ensure(self, dimension: int) -> None:
        if await self._request("GET", f"{self.url}/v1/schema/{self.collection}",
                               ok=(404,)) != 404:
            return
        await self._request("POST", f"{self.url}/v1/schema", ok=(422,), json_body={
            "class": self.collection, "vectorizer": "none",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [{"name": "ref", "dataType": ["text"]},
                           {"name": "text", "dataType": ["text"]},
                           {"name": "meta", "dataType": ["text"],
                            "indexFilterable": False, "indexSearchable": False}]})

    @staticmethod
    def _field(name: str) -> str:
        return "m_" + _NAME.sub("_", name)

    def _where(self, wanted: list[tuple[str, str, Any]]) -> str:
        """The filter as GraphQL, which is how Weaviate is asked."""
        def value(v: Any) -> str:
            if isinstance(v, bool):
                return f"valueBoolean: {'true' if v else 'false'}"
            # Weaviate's own schema calls every number it is given a `number`.
            if isinstance(v, (int, float)):
                return f"valueNumber: {v}"
            return f"valueText: {json.dumps(str(v))}"

        parts = []
        for name, op, operand in wanted:
            path = f'path: ["{self._field(name)}"]'
            if op == "$in":
                if not operand:
                    raise VectorStoreError("$in needs at least one value", store=self.name)
                kind = ("valueNumber" if all(isinstance(v, (int, float))
                                             and not isinstance(v, bool)
                                             for v in operand) else "valueText")
                items = ", ".join(json.dumps(v if kind != "valueText" else str(v))
                                  for v in operand)
                parts.append(f"{{{path}, operator: ContainsAny, {kind}: [{items}]}}")
            else:
                parts.append(f"{{{path}, operator: {_OPS[op]}, {value(operand)}}}")
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else (
            f"{{operator: And, operands: [{', '.join(parts)}]}}")

    def _rest_where(self, wanted: list[tuple[str, str, Any]]) -> dict[str, Any]:
        """The same filter as JSON, which is how a batch delete takes it."""
        def one(name: str, op: str, operand: Any) -> dict[str, Any]:
            path = [self._field(name)]
            if op == "$in":
                kind = ("valueNumber" if all(isinstance(v, (int, float))
                                             and not isinstance(v, bool)
                                             for v in operand) else "valueText")
                # In JSON, a list of values goes under the …Array spelling.
                return {"path": path, "operator": "ContainsAny",
                        f"{kind}Array": [v if kind != "valueText" else str(v)
                                         for v in operand]}
            kind = ("valueBoolean" if isinstance(operand, bool)
                    else "valueNumber" if isinstance(operand, (int, float))
                    else "valueText")
            return {"path": path, "operator": _OPS[op],
                    kind: operand if kind != "valueText" else str(operand)}

        parts = [one(*item) for item in wanted]
        return parts[0] if len(parts) == 1 else {"operator": "And", "operands": parts}

    async def _graphql(self, query: str) -> dict[str, Any]:
        answer = await self._request("POST", f"{self.url}/v1/graphql",
                                     json_body={"query": query})
        if answer.get("errors"):
            raise VectorStoreError(
                f"weaviate: {json.dumps(answer['errors'])[:400]}", store=self.name)
        return answer.get("data") or {}

    async def _upsert(self, records: list[VectorRecord]) -> None:
        objects = []
        for r in records:
            fields = {self._field(k): v for k, v in r.metadata.items()
                      if isinstance(v, (str, int, float, bool))
                      or (isinstance(v, list) and all(isinstance(i, str) for i in v))}
            objects.append({"class": self.collection, "id": as_uuid(r.id), "vector": r.vector,
                            "properties": {**fields, "ref": r.id, "text": r.text,
                                           "meta": json.dumps(r.metadata, default=str,
                                                              ensure_ascii=False)}})
        answer = await self._request("POST", f"{self.url}/v1/batch/objects",
                                     json_body={"objects": objects})
        for item in answer if isinstance(answer, list) else []:
            errors = ((item.get("result") or {}).get("errors") or {}).get("error")
            if errors:
                raise VectorStoreError(
                    f"weaviate refused a record: {json.dumps(errors)[:300]}",
                    store=self.name)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        where = self._where(wanted)
        data = await self._graphql(
            f"{{ Get {{ {self.collection}(nearVector: {{vector: {json.dumps(vector)}}}, "
            f"limit: {k}{f', where: {where}' if where else ''}) "
            "{ ref text meta _additional { id distance } } } }")
        out = []
        for row in (data.get("Get") or {}).get(self.collection) or []:
            extra = row.get("_additional") or {}
            try:
                meta = json.loads(row.get("meta") or "{}")
            except ValueError:
                meta = {}
            out.append(VectorHit(row.get("ref") or extra.get("id", ""),
                                 1.0 - float(extra.get("distance") or 0.0),
                                 row.get("text") or "", meta))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        if ids is not None and not wanted:
            where: dict[str, Any] = {"path": ["id"], "operator": "ContainsAny",
                                     "valueTextArray": [as_uuid(i) for i in ids]}
        else:
            where = self._rest_where(wanted)
            if ids is not None:
                where = {"operator": "And", "operands": [
                    where, {"path": ["id"], "operator": "ContainsAny",
                            "valueTextArray": [as_uuid(i) for i in ids]}]}
        # A batch delete removes at most 10,000 at a time; go round until none match.
        for _ in range(1000):
            answer = await self._request(
                "DELETE", f"{self.url}/v1/batch/objects",
                json_body={"match": {"class": self.collection, "where": where}})
            results = answer.get("results") or {}
            if not results.get("matches") or results.get("matches") <= results.get(
                    "successful", 0) + results.get("failed", 0):
                return

    async def count(self) -> int:
        data = await self._graphql(
            f"{{ Aggregate {{ {self.collection} {{ meta {{ count }} }} }} }}")
        rows = (data.get("Aggregate") or {}).get(self.collection) or [{}]
        return int((rows[0].get("meta") or {}).get("count") or 0)

    async def clear(self) -> None:
        await self._request("DELETE", f"{self.url}/v1/schema/{self.collection}", ok=(404,))
