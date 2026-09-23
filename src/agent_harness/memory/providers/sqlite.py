"""SQLite memory. No driver to install — it is in the standard library.

The right default when you want memory that survives a restart but do not want
to run a database. One file, indexed, and it handles millions of records before
you need anything bigger.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, ClassVar

from ._sql import SQLMemoryStore

__all__ = ["SQLiteMemory"]


class SQLiteMemory(SQLMemoryStore):
    """A single SQLite file. `:memory:` gives a throwaway in-process database."""

    paramstyle: ClassVar[str] = "qmark"
    float_type: ClassVar[str] = "REAL"
    key_type: ClassVar[str] = "TEXT"

    def __init__(self, path: str | Path = ".harness/memory.db", *,
                 table: str = "agent_memory", timeout: float = 10.0,
                 **options: Any) -> None:
        super().__init__(str(path), table=table, **options)
        self.path = str(path)
        self.timeout = timeout
        self._conn: sqlite3.Connection | None = None
        # One connection, one lock: sqlite3 objects are not safe to share across
        # threads, and every call here hops to a worker thread.
        self._lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=self.timeout,
                                         check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # WAL lets readers and a writer work at once; NORMAL is the usual
            # durability/throughput trade for an append-heavy workload.
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _run(self, sql: str, params: tuple[Any, ...], fetch: bool) -> list[dict]:
        conn = self._connect()
        cursor = conn.execute(sql, params)
        rows = [dict(r) for r in cursor.fetchall()] if fetch else []
        conn.commit()
        cursor.close()
        return rows

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        async with self._lock:
            await asyncio.to_thread(self._run, sql, params, False)

    async def _fetch(self, sql: str,
                     params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(self._run, sql, params, True)

    async def _upsert_doc(self, key: str, name: str, namespace: str, text: str,
                          updated: float) -> None:
        # SQLite has a real upsert, so skip the portable two-step.
        await self._execute(
            f"INSERT INTO {self.docs_table} "
            f"(doc_key, name, namespace, text, updated) VALUES (?, ?, ?, ?, ?) "
            f"ON CONFLICT(doc_key) DO UPDATE SET text = excluded.text, "
            f"updated = excluded.updated",
            (key, name, namespace, text, updated))

    async def aclose(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
                self._ready = False
