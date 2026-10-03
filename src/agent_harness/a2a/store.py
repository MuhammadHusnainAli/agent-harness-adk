"""Where A2A tasks are kept — so any replica can answer for any task.

A task outlives the request that started it: a client sends a message to one
replica and asks another how it went. So a task's record lives in a store, not
in the process that is working on it.

By default that store is the harness's own session store — the one chats are
kept in. Name a database there and tasks are in it too, versioned the same way:

    harness = Harness(sessions="postgresql://user:pass@host/agents")
    A2AServer(agent)                      # tasks in PostgreSQL, shared by every replica
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ConfigurationError, SessionConflict
from ..runtime.session import Session, SessionStore

__all__ = ["TaskRecord", "TaskStore", "MemoryTaskStore", "SessionTaskStore"]


class TaskRecord(BaseModel):
    """One A2A task, and what the server needs to know about it besides."""

    id: str
    #: Which of the server's agents it was sent to.
    agent: str = ""
    #: Whose it is. Another caller asking for it is told there is no such task.
    user_id: str | None = None
    tenant_id: str | None = None
    #: The task as the protocol sends it.
    task: dict[str, Any] = Field(default_factory=dict)
    #: Where to send word when it is over.
    push: list[dict[str, Any]] = Field(default_factory=list)
    #: Set by a replica that was asked to cancel a task it is not running; the
    #: one that is running it stops at its next step.
    cancel_requested: bool = False
    created: float = Field(default_factory=time.time)
    #: When it last changed state — what a lost worker is detected by.
    updated: float = Field(default_factory=time.time)
    version: int = 0

    @property
    def state(self) -> str:
        return (self.task.get("status") or {}).get("state", "unknown")

    def owned_by(self, user_id: str | None, tenant_id: str | None) -> bool:
        return self.user_id == user_id and self.tenant_id == tenant_id


class TaskStore(ABC):
    """Three methods keep tasks. `save` must refuse a stale write."""

    @abstractmethod
    async def load(self, task_id: str) -> TaskRecord | None: ...

    @abstractmethod
    async def create(self, record: TaskRecord) -> bool:
        """Store a new task. False if one with this id is already there."""

    @abstractmethod
    async def save(self, record: TaskRecord) -> TaskRecord:
        """Store a changed task. Raises `SessionConflict` if it has been saved
        by someone else since it was loaded."""

    async def delete(self, task_id: str) -> None:
        return None

    async def change(self, task_id: str,
                     edit: Callable[[TaskRecord], Any]) -> TaskRecord | None:
        """Load, edit, save — again if someone else saved in between. `edit`
        returns False to leave the task as it is."""
        for _ in range(8):
            record = await self.load(task_id)
            if record is None:
                return None
            if edit(record) is False:
                return record
            try:
                return await self.save(record)
            except SessionConflict:
                continue
        raise SessionConflict(f"task {task_id} could not be saved: it kept changing",
                              session_id=task_id)


class MemoryTaskStore(TaskStore):
    """In this process, gone with it. For one replica, and for tests."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    async def load(self, task_id: str) -> TaskRecord | None:
        held = self._tasks.get(task_id)
        return held.model_copy(deep=True) if held is not None else None

    async def create(self, record: TaskRecord) -> bool:
        if record.id in self._tasks:
            return False
        record.version = 1
        self._tasks[record.id] = record.model_copy(deep=True)
        return True

    async def save(self, record: TaskRecord) -> TaskRecord:
        held = self._tasks.get(record.id)
        if held is not None and held.version != record.version:
            raise SessionConflict(f"task {record.id} was saved by someone else",
                                  session_id=record.id)
        record.version += 1
        self._tasks[record.id] = record.model_copy(deep=True)
        return record

    async def delete(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)


class SessionTaskStore(TaskStore):
    """Tasks kept in a session store: any database the harness keeps chats in.

    Each task is one session, under an id and an agent name of its own, so it
    is never listed among an agent's chats.
    """

    def __init__(self, sessions: SessionStore, *, prefix: str = "a2a_task_") -> None:
        self.sessions = sessions
        self.prefix = prefix

    def _session(self, record: TaskRecord) -> Session:
        return Session(
            id=f"{self.prefix}{record.id}", agent=f"{record.agent}#a2a",
            title=f"A2A task {record.id}", user_id=record.user_id,
            tenant_id=record.tenant_id, version=record.version,
            created=record.created,
            metadata={"a2a": record.model_dump(mode="json", exclude={"version"})})

    async def load(self, task_id: str) -> TaskRecord | None:
        try:
            session = await self.sessions.load(f"{self.prefix}{task_id}")
        except ConfigurationError:
            return None
        held = session.metadata.get("a2a")
        if not isinstance(held, dict):
            return None
        return TaskRecord(**held, version=session.version)

    async def create(self, record: TaskRecord) -> bool:
        record.version = 0
        try:
            saved = await self.sessions.save(self._session(record))
        except SessionConflict:
            return False
        record.version = saved.version
        return True

    async def save(self, record: TaskRecord) -> TaskRecord:
        saved = await self.sessions.save(self._session(record))
        record.version = saved.version
        return record

    async def delete(self, task_id: str) -> None:
        await self.sessions.delete(f"{self.prefix}{task_id}")
