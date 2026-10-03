"""PostgreSQL with pgvector — your own, or Supabase, Neon, RDS, Aurora,
Cloud SQL, AlloyDB, Azure Database for PostgreSQL."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ...errors import ConfigurationError
from ..base import VectorHit, VectorRecord, VectorStore, VectorStoreError

__all__ = ["PgVectorStore"]

_OPS = {"$eq": "=", "$ne": "<>", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


class PgVectorStore(VectorStore):
    """
        PgVectorStore("postgresql://user:pass@host/db", table="handbook")

    Needs `asyncpg` (`pip install asyncpg`) and the `vector` extension, which is
    created if the role may create it. An HNSW index on cosine distance is built
    with the table.
    """

    name = "pgvector"
    batch_size = 500

    def __init__(self, dsn: str, *, table: str = "knowledge", schema: str = "public",
                 min_size: int = 1, max_size: int = 10, index: bool = True) -> None:
        for part in (table, schema):
            if not part.replace("_", "").isalnum():
                raise VectorStoreError(f"{part!r} is not a usable name", store=self.name)
        self.dsn = dsn.replace("pgvector://", "postgresql://", 1)
        self.table = f'"{schema}"."{table}"'
        self._bare = table
        self.min_size, self.max_size = min_size, max_size
        self.index = index
        self._pool: Any = None
        self._loop: Any = None

    async def _connect(self) -> Any:
        loop = asyncio.get_running_loop()
        if self._pool is None or self._loop is not loop:
            try:
                import asyncpg
            except ImportError as exc:
                raise ConfigurationError(
                    "PgVectorStore needs a driver — pip install asyncpg") from exc
            self._pool = await asyncpg.create_pool(self.dsn, min_size=self.min_size,
                                                   max_size=self.max_size)
            self._loop = loop
        return self._pool

    @staticmethod
    def _vector(vector: list[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    async def ensure(self, dimension: int) -> None:
        pool = await self._connect()
        async with pool.acquire() as db:
            try:
                await db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as exc:
                held = await db.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if not held:
                    raise VectorStoreError(
                        "the pgvector extension is not installed in this database, "
                        f"and this role may not create it ({type(exc).__name__})",
                        store=self.name) from None
            await db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} (id TEXT PRIMARY KEY, "
                f"vector vector({int(dimension)}) NOT NULL, text TEXT NOT NULL DEFAULT '', "
                "metadata JSONB NOT NULL DEFAULT '{}'::jsonb)")
            if self.index:
                await db.execute(
                    f'CREATE INDEX IF NOT EXISTS "{self._bare}_vector_hnsw" ON '
                    f"{self.table} USING hnsw (vector vector_cosine_ops)")
                await db.execute(
                    f'CREATE INDEX IF NOT EXISTS "{self._bare}_metadata" ON '
                    f"{self.table} USING gin (metadata jsonb_path_ops)")

    @staticmethod
    def _where(wanted: list[tuple[str, str, Any]], args: list[Any],
               ids: list[str] | None = None) -> str:
        parts = []
        if ids is not None:
            args.append(ids)
            parts.append(f"id = ANY(${len(args)}::text[])")
        for name, op, value in wanted:
            if op == "$in":
                # Any of these values — or, for a list in the metadata, any shared.
                args.append([json.dumps(v) for v in value])
                args.append(name)
                parts.append(
                    f"(metadata -> ${len(args)}) @> ANY(${len(args) - 1}::jsonb[]) ")
                continue
            if op in ("$eq", "$ne"):
                args.append(json.dumps({name: value}))
                contains = f"metadata @> ${len(args)}::jsonb"
                args.append(json.dumps({name: [value]}))
                held = f"({contains} OR metadata @> ${len(args)}::jsonb)"
                parts.append(held if op == "$eq" else f"NOT {held}")
                continue
            args.append(name)
            field = f"(metadata ->> ${len(args)})"
            args.append(value if isinstance(value, str) else float(value))
            cast = "" if isinstance(value, str) else "::float8"
            parts.append(f"{field}{cast} {_OPS[op]} ${len(args)}")
        return " AND ".join(parts)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        pool = await self._connect()
        rows = [(r.id, self._vector(r.vector), r.text.replace("\x00", ""),
                 json.dumps(r.metadata, default=str)) for r in records]
        async with pool.acquire() as db:
            await db.executemany(
                f"INSERT INTO {self.table} (id, vector, text, metadata) "  # noqa: S608
                "VALUES ($1, $2::vector, $3, $4::jsonb) ON CONFLICT (id) DO UPDATE SET "
                "vector = EXCLUDED.vector, text = EXCLUDED.text, "
                "metadata = EXCLUDED.metadata", rows)

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        args: list[Any] = [self._vector(vector)]
        where = self._where(wanted, args)
        args.append(k)
        pool = await self._connect()
        async with pool.acquire() as db:
            rows = await db.fetch(
                f"SELECT id, text, metadata, 1 - (vector <=> $1::vector) AS score "  # noqa: S608
                f"FROM {self.table} {'WHERE ' + where if where else ''} "
                f"ORDER BY vector <=> $1::vector LIMIT ${len(args)}", *args)
        return [VectorHit(row["id"], float(row["score"]), row["text"],
                          json.loads(row["metadata"]) if isinstance(row["metadata"], str)
                          else dict(row["metadata"])) for row in rows]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        args: list[Any] = []
        where = self._where(wanted, args, ids)
        pool = await self._connect()
        async with pool.acquire() as db:
            await db.execute(f"DELETE FROM {self.table} WHERE {where}", *args)  # noqa: S608

    async def count(self) -> int:
        pool = await self._connect()
        async with pool.acquire() as db:
            return int(await db.fetchval(f"SELECT COUNT(*) FROM {self.table}"))  # noqa: S608

    async def clear(self) -> None:
        pool = await self._connect()
        async with pool.acquire() as db:
            await db.execute(f"TRUNCATE {self.table}")

    async def aclose(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None and self._loop is asyncio.get_running_loop():
            await pool.close()
