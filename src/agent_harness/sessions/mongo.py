"""Sessions in MongoDB (and what speaks its protocol: Cosmos DB, DocumentDB).

One document per conversation, keyed by its id. A save replaces the document
only where the version still matches — the database makes that comparison and
the write one atomic step.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..runtime.session import DurableSessionStore, Session

__all__ = ["MongoSessionStore"]


class MongoSessionStore(DurableSessionStore):
    """Sessions in a collection, over the MongoDB memory backend's client.

        MongoSessionStore(MongoMemory("mongodb://host:27017", database="agents"))

    The conversation is stored as one JSON string, not as nested fields: tool
    arguments are arbitrary and may hold keys Mongo will not take. A document
    is limited to 16 MB, which is a very long conversation.
    """

    def __init__(self, mongo: Any, *, collection: str = "sessions") -> None:
        self.mongo = mongo
        self.collection = collection
        self._ready = False

    async def _coll(self) -> Any:
        await self.mongo._ensure()
        coll = self.mongo._client[self.mongo.database][self.collection]
        if not self._ready:
            await self.mongo._call(coll.create_index,
                                   [("tenant_id", 1), ("user_id", 1), ("updated", -1)])
            self._ready = True
        return coll

    @staticmethod
    def _document(session: Session) -> dict[str, Any]:
        return {"_id": session.id, "agent": session.agent,
                "user_id": session.user_id, "tenant_id": session.tenant_id,
                "title": session.title, "created": session.created,
                "updated": session.updated, "version": session.version,
                "payload": session.model_dump_json()}

    @staticmethod
    def _session(document: dict[str, Any] | None) -> Session | None:
        if not document:
            return None
        try:
            found = Session(**json.loads(document["payload"]))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return None
        found.version = int(document.get("version") or found.version)
        return found

    async def _write(self, session: Session, expect: int) -> bool:
        coll = await self._coll()
        document = self._document(session)
        if expect == 0:
            try:
                await self.mongo._call(coll.insert_one, document)
            except Exception as exc:
                if type(exc).__name__ == "DuplicateKeyError":
                    return False
                raise
            return True
        result = await self.mongo._call(
            coll.replace_one, {"_id": session.id, "version": expect}, document)
        return getattr(result, "matched_count", 0) == 1

    async def _read(self, session_id: str) -> Session | None:
        coll = await self._coll()
        return self._session(await self.mongo._call(coll.find_one, {"_id": session_id}))

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        coll = await self._coll()
        query = {key: value for key, value in (("tenant_id", tenant_id),
                                               ("user_id", user_id), ("agent", agent))
                 if value is not None}
        cursor = coll.find(query).sort("updated", -1).limit(int(limit))
        rows = (await cursor.to_list(length=int(limit)) if self.mongo._async_driver
                else await asyncio.to_thread(list, cursor))
        return [s for s in map(self._session, rows) if s is not None]

    async def _delete(self, session_id: str) -> None:
        coll = await self._coll()
        await self.mongo._call(coll.delete_one, {"_id": session_id})

    async def aclose(self) -> None:
        await self.mongo.aclose()
        self._ready = False
