"""MySQL and MariaDB memory, over aiomysql."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from ...errors import ConfigurationError
from ._sql import SQLMemoryStore

__all__ = ["MySQLMemory"]


class MySQLMemory(SQLMemoryStore):
    """A connection pool over one MySQL database.

        MySQLMemory("mysql://user:pass@host:3306/agents")

    Needs `aiomysql`. Note the 191-character key columns: that is the largest
    index a utf8mb4 column can take on older MySQL, and it is plenty for an id.
    """

    paramstyle: ClassVar[str] = "format"
    text_type: ClassVar[str] = "LONGTEXT"
    float_type: ClassVar[str] = "DOUBLE"
    key_type: ClassVar[str] = "VARCHAR(191)"
    driver_hint: ClassVar[str] = "pip install aiomysql"

    def __init__(self, dsn: str = "", *, table: str = "agent_memory",
                 host: str = "localhost", port: int = 3306, user: str = "root",
                 password: str = "", database: str = "agents",
                 pool_size: int = 10, **options: Any) -> None:
        super().__init__(dsn, table=table, **options)
        if dsn:
            parsed = urlparse(dsn)
            host = parsed.hostname or host
            port = parsed.port or port
            user = unquote(parsed.username or user)
            password = unquote(parsed.password or password)
            database = (parsed.path or "/").lstrip("/") or database
        self.connect_args = {"host": host, "port": port, "user": user,
                             "password": password, "db": database}
        self.pool_size = pool_size
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            try:
                import aiomysql
            except ImportError as exc:
                raise ConfigurationError(
                    "MySQLMemory needs a driver — " + self.driver_hint) from exc
            self._pool = await aiomysql.create_pool(
                minsize=1, maxsize=self.pool_size, autocommit=True,
                **self.connect_args, **self.options)
            return self._pool

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)

    async def _fetch(self, sql: str,
                     params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        import aiomysql

        pool = await self._ensure_pool()
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params)
            return list(await cur.fetchall())

    async def _upsert_doc(self, key: str, name: str, namespace: str, text: str,
                          updated: float) -> None:
        await self._execute(
            f"INSERT INTO {self.docs_table} "
            f"(doc_key, name, namespace, text, updated) VALUES (%s, %s, %s, %s, %s) "
            f"ON DUPLICATE KEY UPDATE text = VALUES(text), updated = VALUES(updated)",
            (key, name, namespace, text, updated))

    async def aclose(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            self._ready = False
