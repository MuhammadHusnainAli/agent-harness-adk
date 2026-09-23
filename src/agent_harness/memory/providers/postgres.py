"""PostgreSQL memory. Uses asyncpg if it is installed, otherwise psycopg 3."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ._sql import SQLMemoryStore

__all__ = ["PostgresMemory"]


class PostgresMemory(SQLMemoryStore):
    """A connection pool over one Postgres database.

        PostgresMemory("postgresql://user:pass@host/db")

    Needs `asyncpg` (preferred) or `psycopg[binary,pool]`.
    """

    text_type: ClassVar[str] = "TEXT"
    float_type: ClassVar[str] = "DOUBLE PRECISION"
    key_type: ClassVar[str] = "TEXT"
    driver_hint: ClassVar[str] = "pip install asyncpg   # or: pip install 'psycopg[binary,pool]'"

    def __init__(self, dsn: str, *, table: str = "agent_memory",
                 min_size: int = 1, max_size: int = 10, **options: Any) -> None:
        super().__init__(dsn, table=table, **options)
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Any = None
        self._driver = ""
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                import asyncpg
            except ImportError:
                asyncpg = None
            if asyncpg is not None:
                self._driver = "asyncpg"
                self.paramstyle = "numeric"
                self._pool = await asyncpg.create_pool(
                    self.dsn, min_size=self.min_size, max_size=self.max_size,
                    **self.options)
                return self._pool
            try:
                from psycopg_pool import AsyncConnectionPool
            except ImportError as exc:
                raise ConfigurationError(
                    "PostgresMemory needs a driver — " + self.driver_hint
                ) from exc
            self._driver = "psycopg"
            self.paramstyle = "format"
            self._pool = AsyncConnectionPool(self.dsn, min_size=self.min_size,
                                             max_size=self.max_size, open=False)
            await self._pool.open()
            return self._pool

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        pool = await self._ensure_pool()
        if self._driver == "asyncpg":
            async with pool.acquire() as conn:
                await conn.execute(sql, *params)
            return
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)

    async def _fetch(self, sql: str,
                     params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        if self._driver == "asyncpg":
            async with pool.acquire() as conn:
                return [dict(r) for r in await conn.fetch(sql, *params)]
        from psycopg.rows import dict_row

        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    async def _upsert_doc(self, key: str, name: str, namespace: str, text: str,
                          updated: float) -> None:
        await self._execute(
            self._q(f"INSERT INTO {self.docs_table} "
                    f"(doc_key, name, namespace, text, updated) "
                    f"VALUES (?, ?, ?, ?, ?) "
                    f"ON CONFLICT (doc_key) DO UPDATE SET text = EXCLUDED.text, "
                    f"updated = EXCLUDED.updated"),
            (key, name, namespace, text, updated))

    async def aclose(self) -> None:
        if self._pool is None:
            return
        await (self._pool.close() if self._driver == "asyncpg"
               else self._pool.close())
        self._pool = None
        self._ready = False
