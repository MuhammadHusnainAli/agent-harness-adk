"""Redis — Redis 8, Redis Stack, Redis Cloud, Azure Managed Redis, MemoryDB."""

from __future__ import annotations

import asyncio
import json
import re
from array import array
from typing import Any

from ...errors import ConfigurationError
from ..base import (
    ALWAYS_FILTERABLE,
    VectorHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)

__all__ = ["RedisVectorStore"]

_ESCAPE = re.compile(r"([^0-9A-Za-z_])")
_NAME = re.compile(r"[^0-9A-Za-z_]")


class RedisVectorStore(VectorStore):
    """
        RedisVectorStore("redis://localhost:6379", index="handbook",
                         filterable={"source": "tag", "year": "numeric"})

    Needs `redis` (`pip install redis`) and a Redis with the query engine —
    Redis 8, or Redis Stack before it.

    Redis indexes the fields it is told about, so say which metadata is filtered
    on, and how: `"tag"` for exact values, `"numeric"` for numbers and ranges.
    Everything else in the metadata is kept and returned, but not filterable.
    """

    name = "redis"
    batch_size = 500

    def __init__(self, url: str = "redis://localhost:6379", *, index: str = "knowledge",
                 filterable: dict[str, str] | None = None, **options: Any) -> None:
        self.url = url
        self.index = index
        self.prefix = f"{index}:"
        self.filterable = {**dict.fromkeys(ALWAYS_FILTERABLE, "tag"),
                           **{k: v.lower() for k, v in (filterable or {}).items()}}
        for key, kind in self.filterable.items():
            if kind not in ("tag", "numeric"):
                raise VectorStoreError(
                    f"{key!r} is {kind!r}; a filterable field is 'tag' or 'numeric'",
                    store=self.name)
        self.options = options
        self._client: Any = None
        self._loop: Any = None

    async def _redis(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            try:
                from redis.asyncio import from_url
            except ImportError as exc:
                raise ConfigurationError(
                    "RedisVectorStore needs a driver — pip install redis") from exc
            # Vectors are bytes, so nothing is decoded on the way back.
            # And the older reply format, whose shape does not vary by server.
            self._client = from_url(self.url, decode_responses=False,
                                    **{"protocol": 2, **self.options})
            self._loop = loop
        return self._client

    @staticmethod
    def _field(name: str) -> str:
        return "m_" + _NAME.sub("_", name)

    async def ensure(self, dimension: int) -> None:
        client = await self._redis()
        try:
            await client.execute_command("FT.INFO", self.index)
            return
        except Exception as exc:
            if "unknown command" in str(exc).lower():
                raise VectorStoreError(
                    "this Redis has no query engine — it needs Redis 8 or Redis Stack",
                    store=self.name) from None
        schema: list[Any] = ["vector", "VECTOR", "HNSW", 6, "TYPE", "FLOAT32",
                             "DIM", dimension, "DISTANCE_METRIC", "COSINE"]
        for key, kind in self.filterable.items():
            schema += [self._field(key), "TAG" if kind == "tag" else "NUMERIC"]
        try:
            await client.execute_command(
                "FT.CREATE", self.index, "ON", "HASH", "PREFIX", 1, self.prefix,
                "SCHEMA", *schema)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise VectorStoreError(f"redis: {exc}", store=self.name) from None

    def _expression(self, wanted: list[tuple[str, str, Any]]) -> str:
        parts = []
        for name, op, value in wanted:
            kind = self.filterable.get(name)
            if kind is None:
                raise VectorStoreError(
                    f"{name!r} is not filterable in this Redis index — declare it: "
                    f"filterable={{{name!r}: 'tag'}} (or 'numeric'). Filterable now: "
                    f"{', '.join(self.filterable) or 'nothing'}", store=self.name)
            field = f"@{self._field(name)}"
            if kind == "numeric":
                low, high = "-inf", "+inf"
                if op in ("$eq", "$ne"):
                    low = high = repr(float(value))
                elif op == "$in":
                    parts.append("(" + " | ".join(
                        f"{field}:[{float(v)!r} {float(v)!r}]" for v in value) + ")")
                    continue
                elif op == "$gt":
                    low = f"({float(value)!r}"
                elif op == "$gte":
                    low = repr(float(value))
                elif op == "$lt":
                    high = f"({float(value)!r}"
                else:
                    high = repr(float(value))
                clause = f"{field}:[{low} {high}]"
                parts.append(f"-{clause}" if op == "$ne" else clause)
                continue
            if op not in ("$eq", "$ne", "$in"):
                raise VectorStoreError(
                    f"{name!r} is a tag; a range needs it declared 'numeric'",
                    store=self.name)
            values = value if op == "$in" else [value]
            tags = "|".join(_ESCAPE.sub(r"\\\1", self._tag(v)) for v in values)
            clause = f"{field}:{{{tags}}}"
            parts.append(f"-{clause}" if op == "$ne" else clause)
        return " ".join(parts) or "*"

    @staticmethod
    def _tag(value: Any) -> str:
        return ("true" if value else "false") if isinstance(value, bool) else str(value)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        client = await self._redis()
        pipe = client.pipeline(transaction=False)
        for r in records:
            fields: dict[str, Any] = {
                "vector": array("f", r.vector).tobytes(), "ref": r.id, "text": r.text,
                "meta": json.dumps(r.metadata, default=str, ensure_ascii=False)}
            for key, kind in self.filterable.items():
                value = r.metadata.get(key)
                if value is None:
                    continue
                if kind == "numeric":
                    fields[self._field(key)] = float(value)
                else:
                    fields[self._field(key)] = ",".join(
                        self._tag(v) for v in (value if isinstance(value, list)
                                               else [value]))
            pipe.delete(self.prefix + r.id)         # no fields left over from before
            pipe.hset(self.prefix + r.id, mapping=fields)
        await pipe.execute()

    @staticmethod
    def _rows(answer: Any) -> list[dict[str, Any]]:
        """`[total, key, [field, value, …], key, …]` as a list of mappings."""
        out = []
        for index in range(1, len(answer or []), 2):
            fields = answer[index + 1] if index + 1 < len(answer) else []
            row = {"_key": answer[index].decode()}
            for n in range(0, len(fields) - 1, 2):
                row[fields[n].decode()] = fields[n + 1]
            out.append(row)
        return out

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        client = await self._redis()
        expression = self._expression(wanted)
        try:
            answer = await client.execute_command(
                "FT.SEARCH", self.index,
                f"({expression})=>[KNN {k} @vector $vec AS dist]",
                "PARAMS", 2, "vec", array("f", vector).tobytes(), "SORTBY", "dist",
                "RETURN", 4, "ref", "text", "meta", "dist", "LIMIT", 0, k, "DIALECT", 2)
        except Exception as exc:
            raise VectorStoreError(f"redis: {exc}", store=self.name) from None
        out = []
        for row in self._rows(answer):
            try:
                meta = json.loads(row.get("meta") or b"{}")
            except ValueError:
                meta = {}
            out.append(VectorHit(
                (row.get("ref") or b"").decode() or row["_key"][len(self.prefix):],
                1.0 - float(row.get("dist") or 0.0),
                (row.get("text") or b"").decode(errors="replace"), meta))
        return out

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        client = await self._redis()
        if not wanted:
            keys = [self.prefix + i for i in ids or []]
            if keys:
                await client.delete(*keys)
            return
        expression = self._expression(wanted)
        for _ in range(10_000):
            answer = await client.execute_command(
                "FT.SEARCH", self.index, expression, "NOCONTENT", "LIMIT", 0, 500,
                "DIALECT", 2)
            keys = [key for key in (answer or [])[1:]
                    if ids is None or key.decode()[len(self.prefix):] in ids]
            if not keys:
                return
            await client.delete(*keys)
            if len(answer) - 1 < 500:
                return
            if ids is not None:
                return

    async def count(self) -> int:
        client = await self._redis()
        info = await client.execute_command("FT.INFO", self.index)
        pairs = {info[n].decode() if isinstance(info[n], bytes) else info[n]: info[n + 1]
                 for n in range(0, len(info) - 1, 2)}
        return int(float(pairs.get("num_docs") or 0))

    async def clear(self) -> None:
        client = await self._redis()
        try:
            await client.execute_command("FT.DROPINDEX", self.index, "DD")
        except Exception:  # noqa: S110 - nothing to drop
            pass

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._loop is asyncio.get_running_loop():
            await client.aclose()
