"""Vector stores that need nothing: one in memory, one in a SQLite file.

Both compare the query with every record — exact, and fine into the tens of
thousands of records. Past that, use a database built for it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from array import array
from pathlib import Path
from typing import Any

from ..base import VectorHit, VectorRecord, VectorStore, VectorStoreError, cosine, holds

__all__ = ["MemoryVectorStore", "SQLiteVectorStore"]


class MemoryVectorStore(VectorStore):
    """In this process, gone with it. For tests, and for trying things out."""

    name = "memory"
    batch_size = 10_000

    def __init__(self) -> None:
        self._rows: dict[str, VectorRecord] = {}
        self.dimension = 0

    async def ensure(self, dimension: int) -> None:
        if self.dimension and self._rows and dimension != self.dimension:
            raise VectorStoreError(
                f"this store holds {self.dimension}-dimension vectors; {dimension} "
                "was asked for — a different embedding model needs a different store",
                store=self.name)
        self.dimension = dimension

    async def _upsert(self, records: list[VectorRecord]) -> None:
        if not self._rows:
            self.dimension = 0              # empty: whatever comes next sets it
        for record in records:
            if self.dimension and len(record.vector) != self.dimension:
                raise VectorStoreError(
                    f"record {record.id!r} has {len(record.vector)} dimensions; this "
                    f"store holds {self.dimension}", store=self.name)
            self.dimension = self.dimension or len(record.vector)
            self._rows[record.id] = VectorRecord(
                record.id, list(record.vector), record.text, dict(record.metadata))

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        scored = [VectorHit(r.id, cosine(vector, r.vector), r.text, dict(r.metadata))
                  for r in self._rows.values() if holds(r.metadata, wanted)]
        scored.sort(key=lambda hit: -hit.score)
        return scored[:k]

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        for key in list(self._rows):
            if (ids is None or key in ids) and holds(self._rows[key].metadata, wanted):
                del self._rows[key]

    async def count(self) -> int:
        return len(self._rows)

    async def clear(self) -> None:
        self._rows.clear()


class SQLiteVectorStore(VectorStore):
    """Vectors in one SQLite file. Survives a restart; needs no server.

        SQLiteVectorStore("knowledge.db", table="handbook")
    """

    name = "sqlite"
    batch_size = 1000

    def __init__(self, path: str | Path = "knowledge.db", *, table: str = "vectors") -> None:
        if not table.replace("_", "").isalnum():
            raise VectorStoreError(f"{table!r} is not a usable table name", store="sqlite")
        self.path = str(path)
        self.table = table
        self._db: sqlite3.Connection | None = None
        self._locks: dict[int, asyncio.Lock] = {}
        self._cache: list[tuple[str, list[float], str, dict[str, Any]]] | None = None

    def _mutex(self) -> asyncio.Lock:
        loop = id(asyncio.get_running_loop())
        if loop not in self._locks:
            self._locks.clear()
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    def _open(self) -> sqlite3.Connection:
        if self._db is None:
            if self.path != ":memory:":
                Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} (id TEXT PRIMARY KEY, "
                "dimension INTEGER NOT NULL, vector BLOB NOT NULL, text TEXT NOT NULL, "
                "metadata TEXT NOT NULL)")
            self._db.commit()
        return self._db

    async def ensure(self, dimension: int) -> None:
        def look() -> int | None:
            row = self._open().execute(
                f"SELECT dimension FROM {self.table} LIMIT 1").fetchone()  # noqa: S608
            return row[0] if row else None

        async with self._mutex():
            held = await asyncio.to_thread(look)
        if held is not None and held != dimension:
            raise VectorStoreError(
                f"{self.path} holds {held}-dimension vectors; {dimension} was asked "
                "for — a different embedding model needs a different table",
                store=self.name)

    async def _upsert(self, records: list[VectorRecord]) -> None:
        rows = [(r.id, len(r.vector), array("f", r.vector).tobytes(), r.text,
                 json.dumps(r.metadata, default=str, ensure_ascii=False)) for r in records]

        def write() -> None:
            db = self._open()
            db.executemany(
                f"INSERT INTO {self.table} (id, dimension, vector, text, metadata) "  # noqa: S608
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "dimension=excluded.dimension, vector=excluded.vector, "
                "text=excluded.text, metadata=excluded.metadata", rows)
            db.commit()

        async with self._mutex():
            await asyncio.to_thread(write)
            self._cache = None

    def _load(self) -> list[tuple[str, list[float], str, dict[str, Any]]]:
        if self._cache is None:
            out = []
            for key, blob, text, meta in self._open().execute(
                    f"SELECT id, vector, text, metadata FROM {self.table}"):  # noqa: S608
                vector = array("f")
                vector.frombytes(blob)
                out.append((key, vector.tolist(), text, json.loads(meta)))
            self._cache = out
        return self._cache

    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]:
        def scan() -> list[VectorHit]:
            scored = [VectorHit(key, cosine(vector, held), text, dict(meta))
                      for key, held, text, meta in self._load() if holds(meta, wanted)]
            scored.sort(key=lambda hit: -hit.score)
            return scored[:k]

        async with self._mutex():
            return await asyncio.to_thread(scan)

    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None:
        def remove() -> None:
            doomed = [key for key, _, _, meta in self._load()
                      if (ids is None or key in ids) and holds(meta, wanted)]
            db = self._open()
            db.executemany(f"DELETE FROM {self.table} WHERE id = ?",  # noqa: S608
                           [(key,) for key in doomed])
            db.commit()

        async with self._mutex():
            await asyncio.to_thread(remove)
            self._cache = None

    async def count(self) -> int:
        def tally() -> int:
            return self._open().execute(
                f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]  # noqa: S608

        async with self._mutex():
            return await asyncio.to_thread(tally)

    async def clear(self) -> None:
        def empty() -> None:
            db = self._open()
            db.execute(f"DELETE FROM {self.table}")  # noqa: S608
            db.commit()

        async with self._mutex():
            await asyncio.to_thread(empty)
            self._cache = None

    async def aclose(self) -> None:
        db, self._db = self._db, None
        if db is not None:
            db.close()
