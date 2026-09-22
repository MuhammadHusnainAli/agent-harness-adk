"""Session store: resume, fork or branch a run. A long job survives a restart."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ConfigurationError
from ..types import Artifact, Message, Usage, new_id

__all__ = ["Session", "SessionStore", "InMemorySessionStore", "FileSessionStore"]


class Session(BaseModel):
    """Everything needed to pick a conversation back up where it stopped."""

    id: str = Field(default_factory=lambda: new_id("ses"))
    agent: str = ""
    title: str = ""
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
            messages=[m.model_copy(deep=True) for m in self.messages[:cut]],
            artifacts=list(self.artifacts), parent_id=self.id, forked_at=cut,
            metadata=dict(self.metadata),
        )

    def replay(self) -> list[Message]:
        """The messages a resumed run should start from."""
        return [m.model_copy(deep=True) for m in self.messages]


class SessionStore(ABC):
    @abstractmethod
    async def save(self, session: Session) -> Session: ...

    @abstractmethod
    async def load(self, session_id: str) -> Session: ...

    @abstractmethod
    async def list(self, *, limit: int = 50) -> list[Session]: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

    async def fork(self, session_id: str, *, at: int | None = None) -> Session:
        forked = (await self.load(session_id)).fork(at=at)
        return await self.save(forked)

    async def resume(self, session_id: str) -> Session:
        return await self.load(session_id)


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def save(self, session: Session) -> Session:
        session.updated = time.time()
        self._sessions[session.id] = session
        return session

    async def load(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise ConfigurationError(f"no session {session_id!r}")
        return self._sessions[session_id]

    async def list(self, *, limit: int = 50) -> list[Session]:
        rows = sorted(self._sessions.values(), key=lambda s: -s.updated)
        return rows[:limit]

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class FileSessionStore(SessionStore):
    """One JSON file per session under `root`."""

    def __init__(self, root: str | Path = ".harness/sessions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.root / f"{safe}.json"

    async def save(self, session: Session) -> Session:
        session.updated = time.time()
        payload = session.model_dump_json(indent=2)
        async with self._lock:
            await asyncio.to_thread(self._path(session.id).write_text, payload, "utf-8")
        return session

    async def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            raise ConfigurationError(f"no session {session_id!r} in {self.root}")
        text = await asyncio.to_thread(path.read_text, "utf-8")
        return Session(**json.loads(text))

    async def list(self, *, limit: int = 50) -> list[Session]:
        files = sorted(self.root.glob("*.json"), key=lambda f: -f.stat().st_mtime)
        out: list[Session] = []
        for file in files[:limit]:
            try:
                out.append(Session(**json.loads(file.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    async def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
