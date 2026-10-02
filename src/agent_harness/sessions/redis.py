"""Sessions in Redis. Fast, and the natural home for chats with a lifetime.

A conversation is a hash; a sorted set per owner, scored by when it last
changed, is what makes "newest first" a single call. The version check and the
write happen inside one Lua script, so they are atomic on the server.
"""

from __future__ import annotations

import json
from typing import Any

from ..runtime.session import DurableSessionStore, Session

__all__ = ["RedisSessionStore"]

# KEYS: the session, the index of every session, the owner's index.
# ARGV: expected version, new version, payload, updated, id, ttl, agent.
_SAVE = """
local held = redis.call('HGET', KEYS[1], 'version')
if (held == false and ARGV[1] == '0') or (held == ARGV[1]) then
  redis.call('HSET', KEYS[1], 'version', ARGV[2], 'payload', ARGV[3],
             'owner', KEYS[3], 'agent', ARGV[7])
  redis.call('ZADD', KEYS[2], ARGV[4], ARGV[5])
  redis.call('ZADD', KEYS[3], ARGV[4], ARGV[5])
  if tonumber(ARGV[6]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[6]) end
  return 1
end
return 0
"""


class RedisSessionStore(DurableSessionStore):
    """Sessions in Redis, over the Redis memory backend's client.

        RedisSessionStore(RedisMemory("redis://host:6379/0"), ttl=30 * 86_400)

    `ttl` lets a conversation expire that long after it was last saved; None
    keeps it until it is deleted. The keys of one save live under one prefix but
    not one hash slot, so on Redis Cluster wrap the prefix in braces —
    `prefix="{chats}"` — to keep them together.
    """

    def __init__(self, redis: Any, *, prefix: str | None = None,
                 ttl: int | None = None) -> None:
        self.redis = redis
        self.prefix = (prefix or redis.prefix).rstrip(":")
        self.ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:session:{session_id}"

    def _index(self, tenant_id: str | None = None, user_id: str | None = None,
               *, everything: bool = False) -> str:
        if everything:
            return f"{self.prefix}:sessions:_all"
        return f"{self.prefix}:sessions:{tenant_id or '_'}:{user_id or '_'}"

    async def _write(self, session: Session, expect: int) -> bool:
        client = await self.redis._ensure()
        done = await client.eval(
            _SAVE, 3, self._key(session.id), self._index(everything=True),
            self._index(session.tenant_id, session.user_id),
            str(expect), str(session.version), session.model_dump_json(),
            repr(session.updated), session.id, str(int(self.ttl or 0)),
            session.agent)
        return int(done) == 1

    async def _read(self, session_id: str) -> Session | None:
        client = await self.redis._ensure()
        row = await client.hgetall(self._key(session_id))
        if not row or "payload" not in row:
            return None
        try:
            found = Session(**json.loads(row["payload"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        found.version = int(row.get("version") or found.version)
        return found

    async def _list(self, *, limit: int, user_id: str | None,
                    tenant_id: str | None, agent: str | None) -> list[Session]:
        client = await self.redis._ensure()
        # One owner's index when both are named; otherwise everything, filtered.
        exact = user_id is not None and tenant_id is not None
        index = (self._index(tenant_id, user_id) if exact
                 else self._index(everything=True))
        out: list[Session] = []
        start, page = 0, max(limit * 2, 50)
        while len(out) < limit:
            ids = await client.zrevrange(index, start, start + page - 1)
            if not ids:
                break
            for session_id in ids:
                found = await self._read(session_id)
                if found is None:
                    await client.zrem(index, session_id)     # expired: tidy up
                    continue
                if ((user_id is None or found.user_id == user_id)
                        and (tenant_id is None or found.tenant_id == tenant_id)
                        and (agent is None or found.agent == agent)):
                    out.append(found)
                    if len(out) >= limit:
                        break
            start += page
        return out

    async def _delete(self, session_id: str) -> None:
        client = await self.redis._ensure()
        owner = await client.hget(self._key(session_id), "owner")
        pipe = client.pipeline()
        pipe.delete(self._key(session_id))
        pipe.zrem(self._index(everything=True), session_id)
        if owner:
            pipe.zrem(owner, session_id)
        await pipe.execute()

    async def aclose(self) -> None:
        await self.redis.aclose()
