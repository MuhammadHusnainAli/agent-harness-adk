"""Sessions in a relational database: SQLite, PostgreSQL, MySQL / MariaDB.

One row per conversation. Who it belongs to and when it last changed are real,
indexed columns, so "this user's chats, newest first" is one query the database
answers from an index; the conversation itself is one JSON document beside them.

A save is a single statement that only matches the version the session was
loaded at. Each save also leaves its own mark on the row, and the save is
confirmed by reading that mark back — which works the same on every driver,
whatever it reports about rows affected.
"""

from __future__ import annotations

import json
from typing import Any

from ..runtime.session import DurableSessionStore, Session
from ..types import new_id

__all__ = ["SQLSessionStore"]

_COLUMNS = ("agent", "user_id", "tenant_id", "title", "created", "updated",
            "version", "rev", "payload")


class SQLSessionStore(DurableSessionStore):
    """Sessions in a table, over any of the SQL memory backends.

        SQLSessionStore(PostgresMemory("postgresql://user:pass@host/agents"))

    It borrows the backend's connection pool and dialect, so a harness whose
    memory is already in Postgres keeps its chats there on the same connections.
    """

    def __init__(self, database: Any, *, table: str = "agent_sessions") -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError(f"not a table name: {table!r}")
        self.db = database
        self.table = table
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        db, t = self.db, self.table
        await db._execute(f"""CREATE TABLE IF NOT EXISTS {t} (
            id {db.key_type} PRIMARY KEY,
            agent {db.key_type},
            user_id {db.key_type},
            tenant_id {db.key_type},
            title {db.text_type},
            created {db.float_type} NOT NULL,
            updated {db.float_type} NOT NULL,
            version INTEGER NOT NULL,
            rev {db.key_type} NOT NULL,
            payload {db.text_type} NOT NULL
        )""")
        # Listing is by owner, newest first. MySQL has no IF NOT EXISTS for an
        # index, so the plain form is tried next, and "already there" is fine.
        columns = f"{t} (tenant_id, user_id, updated)"
        for statement in (f"CREATE INDEX IF NOT EXISTS {t}_owner_idx ON {columns}",
                          f"CREATE INDEX {t}_owner_idx ON {columns}"):
            try:
                await db._execute(statement)
                break
            except Exception:  # noqa: S112 - the next form, or it already exists
                continue
        self._ready = True

    def _values(self, session: Session, rev: str) -> tuple[Any, ...]:
        return (session.agent, session.user_id, session.tenant_id, session.title,
                session.created, session.updated, session.version, rev,
                session.model_dump_json())

    async def _mark(self, session_id: str) -> str | None:
        rows = await self.db._fetch(
            self.db._q(f"SELECT rev FROM {self.table} WHERE id = ?"), (session_id,))
        return rows[0]["rev"] if rows else None

    async def _write(self, session: Session, expect: int) -> bool:
        await self._ensure()
        db, t = self.db, self.table
        rev = new_id("rev")
        values = self._values(session, rev)
        if expect == 0:
            marks = ", ".join("?" for _ in range(len(_COLUMNS) + 1))
            try:
                await db._execute(
                    db._q(f"INSERT INTO {t} (id, {', '.join(_COLUMNS)}) "
                          f"VALUES ({marks})"), (session.id, *values))
            except Exception:
                # Already there — somebody created it first — or a real failure.
                if await self._mark(session.id) is None:
                    raise
                return False
        else:
            assignments = ", ".join(f"{column} = ?" for column in _COLUMNS)
            await db._execute(
                db._q(f"UPDATE {t} SET {assignments} WHERE id = ? AND version = ?"),
                (*values, session.id, expect))
        return await self._mark(session.id) == rev

    @staticmethod
    def _session(row: dict[str, Any]) -> Session | None:
        try:
            found = Session(**json.loads(row["payload"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        found.version = int(row.get("version") or found.version)   # the column rules
        return found

    async def _read(self, session_id: str) -> Session | None:
        await self._ensure()
        rows = await self.db._fetch(
            self.db._q(f"SELECT payload, version FROM {self.table} WHERE id = ?"),
            (session_id,))
        return self._session(rows[0]) if rows else None

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        await self._ensure()
        clauses, params = [], []
        for column, value in (("tenant_id", tenant_id), ("user_id", user_id),
                              ("agent", agent)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await self.db._fetch(
            self.db._q(f"SELECT payload, version FROM {self.table}{where} "
                       f"ORDER BY updated DESC LIMIT ?"), (*params, int(limit)))
        return [s for s in map(self._session, rows) if s is not None]

    async def _delete(self, session_id: str) -> None:
        await self._ensure()
        await self.db._execute(
            self.db._q(f"DELETE FROM {self.table} WHERE id = ?"), (session_id,))

    async def aclose(self) -> None:
        await self.db.aclose()
        self._ready = False
