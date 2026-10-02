"""Sessions in object storage: an S3 bucket, an Azure storage account, GCS.

One object per conversation, and a tiny marker per owner so that listing one
person's chats is a prefix listing rather than a read of every conversation:

    {prefix}/sessions/{id}.json
    {prefix}/session-index/{tenant}/{user}/{id}

Cheap, durable, and the right place for chats kept for years. Two limits, both
because an object store is not a database. Listing reads the markers and sorts
them here, so it is for a person's history, not for querying millions. And the
version check reads before it writes: it stops the ordinary case of two requests
on one chat, but it is not atomic — two saves landing in the same instant can
both pass. Where that matters, keep live chats in SQL, MongoDB, Redis or
DynamoDB, and archive to a bucket.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..runtime.session import DurableSessionStore, Session

__all__ = ["ObjectSessionStore"]


class ObjectSessionStore(DurableSessionStore):
    """Sessions as objects, over any of the object-store memory backends.

        ObjectSessionStore(AzureBlobMemory("chats", connection_string="..."))
    """

    def __init__(self, objects: Any, *, concurrency: int = 16) -> None:
        self.objects = objects
        self.concurrency = concurrency

    def _key(self, session_id: str) -> str:
        return f"{self.objects.prefix}/sessions/{session_id}.json"

    def _marker(self, session: Session) -> str:
        return (f"{self.objects.prefix}/session-index/{session.tenant_id or '_'}/"
                f"{session.user_id or '_'}/{session.id}")

    @staticmethod
    def _parse(raw: bytes | None) -> Session | None:
        if not raw:
            return None
        try:
            return Session(**json.loads(raw))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    async def _read(self, session_id: str) -> Session | None:
        return self._parse(await self.objects._get(self._key(session_id)))

    async def _write(self, session: Session, expect: int) -> bool:
        held = await self._read(session.id)
        if (held.version if held is not None else 0) != expect:
            return False
        await self.objects._put(self._key(session.id),
                                session.model_dump_json().encode(), "application/json")
        if held is not None and self._marker(held) != self._marker(session):
            await self.objects._delete([self._marker(held)])
        await self.objects._put(
            self._marker(session),
            json.dumps({"updated": session.updated, "agent": session.agent}).encode(),
            "application/json")
        return True

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        base = f"{self.objects.prefix}/session-index/"
        if tenant_id is not None:
            base += f"{tenant_id}/"
            if user_id is not None:
                base += f"{user_id}/"
        markers = await self.objects._list(base)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def stamp(key: str) -> tuple[float, str, str]:
            async with semaphore:
                raw = await self.objects._get(key)
            try:
                blob = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                blob = {}
            return float(blob.get("updated") or 0), key.rsplit("/", 1)[-1], str(
                blob.get("agent") or "")

        stamped = sorted(await asyncio.gather(*(stamp(k) for k in markers)),
                         reverse=True)
        out: list[Session] = []
        for _, session_id, owner_agent in stamped:
            if agent is not None and owner_agent != agent:
                continue
            found = await self._read(session_id)
            if (found is not None
                    and (user_id is None or found.user_id == user_id)
                    and (tenant_id is None or found.tenant_id == tenant_id)):
                out.append(found)
                if len(out) >= limit:
                    break
        return out

    async def _delete(self, session_id: str) -> None:
        held = await self._read(session_id)
        if held is None:
            return      # some stores (Azure) object to deleting what is not there
        await self.objects._delete([self._key(session_id), self._marker(held)])

    async def aclose(self) -> None:
        await self.objects.aclose()
