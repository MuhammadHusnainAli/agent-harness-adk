"""Shared SQL implementation. Every relational backend is a thin dialect on top.

One table for records, one for documents, and an index that matches how the
harness actually reads: narrowed by trace first, then scope, then time. Queries
are built once with `?` placeholders and converted per dialect, so there is a
single place where the SQL can be wrong.
"""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from typing import Any, ClassVar

from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["SQLMemoryStore"]

COLUMNS = ("id", "scope", "kind", "text", "data", "tags", "source",
           "user_id", "session_id", "tenant_id", "ts")


class SQLMemoryStore(MemoryStore):
    """Records and documents in two tables, with the trace as indexed columns."""

    #: How this dialect writes a bound parameter.
    paramstyle: ClassVar[str] = "qmark"      # qmark | format | numeric
    text_type: ClassVar[str] = "TEXT"
    float_type: ClassVar[str] = "DOUBLE PRECISION"
    driver_hint: ClassVar[str] = ""

    def __init__(self, dsn: str = "", *, table: str = "agent_memory",
                 docs_table: str = "", **options: Any) -> None:
        self.dsn = dsn
        self.table = table
        self.docs_table = docs_table or f"{table}_docs"
        self.options = options
        self._ready = False

    # ---- what a dialect must provide --------------------------------------
    @abstractmethod
    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None: ...

    @abstractmethod
    async def _fetch(self, sql: str,
                     params: tuple[Any, ...] = ()) -> list[dict[str, Any]]: ...

    # ---- SQL construction ---------------------------------------------------
    def _q(self, sql: str) -> str:
        """Convert `?` placeholders to whatever this driver expects."""
        if self.paramstyle == "qmark":
            return sql
        if self.paramstyle == "format":
            return sql.replace("?", "%s")
        out, index = [], 0
        for char in sql:
            if char == "?":
                index += 1
                out.append(f"${index}")
            else:
                out.append(char)
        return "".join(out)

    def _schema(self) -> list[str]:
        t, d, txt, flt = self.table, self.docs_table, self.text_type, self.float_type
        return [
            f"""CREATE TABLE IF NOT EXISTS {t} (
                id {self.key_type} PRIMARY KEY,
                scope {txt} NOT NULL,
                kind {txt} NOT NULL,
                text {txt},
                data {txt},
                tags {txt},
                source {txt},
                user_id {self.key_type},
                session_id {self.key_type},
                tenant_id {self.key_type},
                ts {flt} NOT NULL
            )""",
            # The harness reads narrowed by trace, then scope, then newest first.
            f"CREATE INDEX IF NOT EXISTS {t}_trace_idx "
            f"ON {t} (user_id, session_id, tenant_id, scope, ts)",
            f"""CREATE TABLE IF NOT EXISTS {d} (
                doc_key {self.key_type} PRIMARY KEY,
                name {txt} NOT NULL,
                namespace {txt} NOT NULL,
                text {txt},
                updated {flt} NOT NULL
            )""",
        ]

    key_type: ClassVar[str] = "VARCHAR(191)"   # indexable on MySQL too

    async def _ensure(self) -> None:
        if self._ready:
            return
        for statement in self._schema():
            await self._execute(statement)
        self._ready = True

    @staticmethod
    def _where(scope: str | None, kind: str | None,
               trace: Trace | None) -> tuple[str, list[Any]]:
        """Build the filter once — a record with no value on an axis is shared."""
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if trace is not None:
            for field, value in trace.filters().items():
                clauses.append(f"({field} = ? OR {field} IS NULL)")
                params.append(value)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    # ---- records -------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        await self._ensure()
        columns = ", ".join(COLUMNS)
        marks = ", ".join("?" for _ in COLUMNS)
        await self._execute(
            self._q(f"INSERT INTO {self.table} ({columns}) VALUES ({marks})"),
            self._row(record),
        )
        return record

    @staticmethod
    def _row(record: MemoryRecord) -> tuple[Any, ...]:
        return (
            record.id, record.scope, record.kind, record.text,
            json.dumps(record.data, default=str), json.dumps(record.tags),
            record.source, record.user_id, record.session_id, record.tenant_id,
            record.ts,
        )

    @staticmethod
    def _record(row: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], scope=row["scope"], kind=row["kind"],
            text=row.get("text") or "",
            data=_loads(row.get("data"), {}), tags=_loads(row.get("tags"), []),
            source=row.get("source") or "", user_id=row.get("user_id"),
            session_id=row.get("session_id"), tenant_id=row.get("tenant_id"),
            ts=row.get("ts") or 0.0,
        )

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        await self._ensure()
        where, params = self._where(scope, kind, trace)
        # Newest-first with LIMIT so the database does the work, then flip back
        # to chronological order, which is what every caller expects.
        sql = f"SELECT {', '.join(COLUMNS)} FROM {self.table}{where} ORDER BY ts DESC"
        if limit:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        rows = await self._fetch(self._q(sql), tuple(params))
        return [self._record(r) for r in reversed(rows)]

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        await self._ensure()
        where, params = self._where(scope, None, trace)
        await self._execute(self._q(f"DELETE FROM {self.table}{where}"),
                            tuple(params))
        if trace is not None:
            await self._execute(
                self._q(f"DELETE FROM {self.docs_table} WHERE namespace = ?"),
                (trace.slug,))
        elif scope is None:
            await self._execute(self._q(f"DELETE FROM {self.docs_table}"))

    # ---- documents -----------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        await self._ensure()
        rows = await self._fetch(
            self._q(f"SELECT text FROM {self.docs_table} WHERE doc_key = ?"),
            (self.doc_key(name, trace),))
        return (rows[0]["text"] or "") if rows else ""

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        await self._ensure()
        key = self.doc_key(name, trace)
        namespace = trace.slug if trace is not None else "_shared"
        await self._upsert_doc(key, name, namespace, text, time.time())

    async def _upsert_doc(self, key: str, name: str, namespace: str, text: str,
                          updated: float) -> None:
        """Dialects differ on upsert; this is the portable two-step."""
        await self._execute(
            self._q(f"UPDATE {self.docs_table} SET text = ?, updated = ? "
                    f"WHERE doc_key = ?"), (text, updated, key))
        rows = await self._fetch(
            self._q(f"SELECT doc_key FROM {self.docs_table} WHERE doc_key = ?"),
            (key,))
        if not rows:
            await self._execute(
                self._q(f"INSERT INTO {self.docs_table} "
                        f"(doc_key, name, namespace, text, updated) "
                        f"VALUES (?, ?, ?, ?, ?)"),
                (key, name, namespace, text, updated))


def _loads(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback
