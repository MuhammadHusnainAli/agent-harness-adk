"""Session store: resume, fork or branch a run. A long job survives a restart.

A `Session` is one conversation: its messages, what it cost, who it belongs to.
The two stores here keep it in memory or in a file each. For a database — SQL,
MongoDB, Redis, DynamoDB, a storage account — see `agent_harness.sessions`:

    Harness(sessions=session_provider("postgresql://user:pass@host/agents"))

Every store that outlives the process holds a session under three promises.
It knows **whose** it is, so a chat can be listed for its owner and refused to
anyone else. It is **versioned**, so two requests on one chat cannot silently
overwrite each other — the second is told (`SessionConflict`). And saving it is
**all or nothing**: a reader never sees half a conversation.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ConfigurationError, SessionConflict
from ..types import Artifact, Message, Usage, new_id

__all__ = ["Session", "SessionStore", "DurableSessionStore",
           "InMemorySessionStore", "FileSessionStore"]


class Session(BaseModel):
    """Everything needed to pick a conversation back up where it stopped."""

    id: str = Field(default_factory=lambda: new_id("ses"))
    agent: str = ""
    title: str = ""
    #: Whose conversation this is. Set from the agent's trace on the first run;
    #: a store can list by them, and an agent acting for someone else is refused.
    user_id: str | None = None
    tenant_id: str | None = None
    #: How many times it has been saved. A store accepts a save only from the
    #: version it holds.
    version: int = 0
    messages: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)
    parent_id: str | None = None
    forked_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add(self, message: Message) -> Message:
        self.messages.append(message)
        self.updated = time.time()
        return message

    def fork(self, *, at: int | None = None) -> Session:
        """Branch a run: same history up to `at`, new id, parent recorded."""
        cut = len(self.messages) if at is None else max(0, min(at, len(self.messages)))
        return Session(
            agent=self.agent, title=f"{self.title} (fork)" if self.title else "",
            user_id=self.user_id, tenant_id=self.tenant_id,
            messages=[m.model_copy(deep=True) for m in self.messages[:cut]],
            artifacts=list(self.artifacts), parent_id=self.id, forked_at=cut,
            metadata=dict(self.metadata),
        )

    def owned_by(self, user_id: str | None, tenant_id: str | None) -> bool:
        """May someone acting as this user, in this tenant, have it?

        An axis nobody set is not a wall: a session with no owner is anyone's,
        and a caller with no identity is trusted — that is a single-user setup.
        """
        return ((not self.user_id or not user_id or self.user_id == user_id)
                and (not self.tenant_id or not tenant_id
                     or self.tenant_id == tenant_id))

    def summary(self) -> dict[str, Any]:
        """One row for a list of chats — everything but the messages."""
        return {"id": self.id, "agent": self.agent, "title": self.title,
                "user_id": self.user_id, "tenant_id": self.tenant_id,
                "messages": len(self.messages), "created": self.created,
                "updated": self.updated, "cost_usd": self.usage.cost_usd,
                "version": self.version}

    def replay(self) -> list[Message]:
        """The messages a resumed run should start from."""
        return [m.model_copy(deep=True) for m in self.messages]


class SessionStore(ABC):
    """Where conversations are kept. Four methods make a store."""

    @abstractmethod
    async def save(self, session: Session) -> Session: ...

    @abstractmethod
    async def load(self, session_id: str) -> Session: ...

    @abstractmethod
    async def list(self, *, limit: int = 50, user_id: str | None = None,
                   tenant_id: str | None = None,
                   agent: str | None = None) -> list[Session]:
        """The newest sessions first, narrowed to an owner or an agent."""

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

    async def fork(self, session_id: str, *, at: int | None = None) -> Session:
        forked = (await self.load(session_id)).fork(at=at)
        return await self.save(forked)

    async def resume(self, session_id: str) -> Session:
        return await self.load(session_id)

    async def aclose(self) -> None:
        """Release whatever the store holds open. Safe to call twice."""
        return None

    async def check(self) -> dict[str, Any]:
        """Prove the store does what a chat needs of it, against the real backend.

        Saves a session, reads it back, lists it for its owner, shows that a
        stale save is refused, and deletes it. Run it once when you wire a
        database in, rather than finding out on the first real conversation.
        """
        steps: list[dict[str, Any]] = []
        probe = Session(agent="_check", title="store check", user_id="_check_user",
                        tenant_id="_check_tenant",
                        messages=[Message.user("hello"), Message.assistant("héllo ✓")])

        async def step(name: str, action: Any) -> bool:
            began = time.monotonic()
            try:
                detail = await action()
                steps.append({"step": name, "ok": True, "detail": str(detail or ""),
                              "seconds": round(time.monotonic() - began, 3)})
                return True
            except Exception as exc:
                steps.append({"step": name, "ok": False,
                              "detail": f"{type(exc).__name__}: {exc}",
                              "seconds": round(time.monotonic() - began, 3)})
                return False

        async def save() -> str:
            await self.save(probe)
            return f"version {probe.version}"

        async def load() -> str:
            back = await self.load(probe.id)
            if [m.text for m in back.messages] != ["hello", "héllo ✓"]:
                raise ConfigurationError("the messages did not come back as saved")
            if (back.user_id, back.tenant_id) != ("_check_user", "_check_tenant"):
                raise ConfigurationError("the owner did not come back as saved")
            return f"{len(back.messages)} messages"

        async def listing() -> str:
            mine = await self.list(limit=5, user_id="_check_user",
                                   tenant_id="_check_tenant")
            if probe.id not in [s.id for s in mine]:
                raise ConfigurationError("the session is not listed for its owner")
            others = await self.list(limit=5, user_id="_somebody_else",
                                     tenant_id="_check_tenant")
            if probe.id in [s.id for s in others]:
                raise ConfigurationError("the session is listed for someone else")
            return "listed for its owner only"

        async def conflict() -> str:
            stale = (await self.load(probe.id)).model_copy(deep=True)
            fresh = (await self.load(probe.id)).model_copy(deep=True)
            fresh.title = "saved first"
            await self.save(fresh)
            try:
                await self.save(stale)
            except SessionConflict:
                return "a stale save is refused"
            raise ConfigurationError(
                "a save from an old version was accepted — two requests on one "
                "chat would overwrite each other")

        async def delete() -> str:
            await self.delete(probe.id)
            try:
                await self.load(probe.id)
            except ConfigurationError:
                return ""
            raise ConfigurationError("the session is still there after delete")

        if await step("save", save):
            for name, action in (("load", load), ("list by owner", listing),
                                 ("refuse a stale save", conflict)):
                await step(name, action)
        await step("delete", delete)
        return {"store": type(self).__name__, "ok": all(s["ok"] for s in steps),
                "steps": steps}


class DurableSessionStore(SessionStore):
    """The shape every store that outlives the process shares.

    A backend says how to read one, write one *if it is still at the version
    expected*, list and delete. Versioning, the conflict, titles and timestamps
    are handled here, once.
    """

    @abstractmethod
    async def _read(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def _write(self, session: Session, expect: int) -> bool:
        """Store `session` if what is held is at version `expect` (0: not held
        at all). False means somebody else got there first."""

    @abstractmethod
    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]: ...

    @abstractmethod
    async def _delete(self, session_id: str) -> None: ...

    async def save(self, session: Session) -> Session:
        expect = session.version
        was = session.updated
        session.version, session.updated = expect + 1, time.time()
        try:
            written = await self._write(session, expect)
        except BaseException:
            session.version, session.updated = expect, was
            raise
        if not written:
            session.version, session.updated = expect, was
            raise SessionConflict(
                f"session {session.id} was saved by someone else since it was "
                f"loaded (it is no longer at version {expect})",
                session_id=session.id)
        return session

    async def load(self, session_id: str) -> Session:
        found = await self._read(session_id)
        if found is None:
            raise ConfigurationError(f"no session {session_id!r}")
        return found

    async def list(self, *, limit: int = 50, user_id: str | None = None,
                   tenant_id: str | None = None,
                   agent: str | None = None) -> list[Session]:
        return await self._list(limit=max(1, int(limit)), user_id=user_id,
                                tenant_id=tenant_id, agent=agent)

    async def delete(self, session_id: str) -> None:
        await self._delete(session_id)


def _matches(session: Session, user_id: str | None, tenant_id: str | None,
             agent: str | None) -> bool:
    return ((user_id is None or session.user_id == user_id)
            and (tenant_id is None or session.tenant_id == tenant_id)
            and (agent is None or session.agent == agent))


class InMemorySessionStore(SessionStore):
    """The default. Gone when the process exits."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def save(self, session: Session) -> Session:
        held = self._sessions.get(session.id)
        if held is not None and held is not session and held.version != session.version:
            raise SessionConflict(
                f"session {session.id} was saved by someone else since it was "
                f"loaded (it is no longer at version {session.version})",
                session_id=session.id)
        session.version += 1
        session.updated = time.time()
        self._sessions[session.id] = session
        return session

    async def load(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise ConfigurationError(f"no session {session_id!r}")
        return self._sessions[session_id]

    async def list(self, *, limit: int = 50, user_id: str | None = None,
                   tenant_id: str | None = None,
                   agent: str | None = None) -> list[Session]:
        rows = sorted((s for s in self._sessions.values()
                       if _matches(s, user_id, tenant_id, agent)),
                      key=lambda s: -s.updated)
        return rows[:limit]

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class FileSessionStore(DurableSessionStore):
    """One JSON file per session under `root`. For one machine.

    A save replaces the file in one step, so a reader never finds half of one,
    and it is refused if the file has moved on since the session was loaded.
    """

    def __init__(self, root: str | Path = ".harness/sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.root / f"{safe}.json"

    @staticmethod
    def _parse(text: str) -> Session | None:
        try:
            return Session(**json.loads(text))
        except (json.JSONDecodeError, ValueError):
            return None

    def _held_version(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("version") or 0)
        except (json.JSONDecodeError, ValueError, OSError):
            return 0

    def _replace(self, path: Path, payload: str, expect: int) -> bool:
        # A file written before sessions were versioned reads as 0, and so does
        # a session loaded from it: the first save after an upgrade goes through.
        if self._held_version(path) != expect:
            return False
        scratch = path.with_suffix(f".{new_id()}.tmp")
        scratch.write_text(payload, encoding="utf-8")
        scratch.replace(path)
        return True

    async def _write(self, session: Session, expect: int) -> bool:
        payload = session.model_dump_json(indent=2)
        async with self._lock:
            return await asyncio.to_thread(self._replace, self._path(session.id),
                                           payload, expect)

    async def _read(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return self._parse(await asyncio.to_thread(path.read_text, "utf-8"))

    async def load(self, session_id: str) -> Session:
        found = await self._read(session_id)
        if found is None:
            raise ConfigurationError(f"no session {session_id!r} in {self.root}")
        return found

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        def scan() -> list[Session]:
            files = sorted(self.root.glob("*.json"), key=lambda f: -f.stat().st_mtime)
            out: list[Session] = []
            for file in files:
                found = self._parse(file.read_text(encoding="utf-8"))
                if found is not None and _matches(found, user_id, tenant_id, agent):
                    out.append(found)
                    if len(out) >= limit:
                        break
            return out

        return await asyncio.to_thread(scan)

    async def _delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
