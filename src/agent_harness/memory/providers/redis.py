"""Redis memory. Fast, and the right fit for session-scoped memory with a TTL."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, ClassVar

from ...errors import ConfigurationError
from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["RedisMemory"]


class RedisMemory(MemoryStore):
    """Records in a sorted set per namespace, documents in plain keys.

        RedisMemory("redis://localhost:6379/0", ttl=86_400)

    Scored by timestamp, so a range read is ordered and a `limit` is served by
    Redis rather than by us. Needs `redis`.
    """

    driver_hint: ClassVar[str] = "pip install redis"

    def __init__(self, dsn: str = "redis://localhost:6379/0", *,
                 prefix: str = "agent-memory", ttl: int | None = None,
                 client: Any = None, **options: Any) -> None:
        self.dsn = dsn
        self.prefix = prefix.rstrip(":")
        self.ttl = ttl
        self.options = options
        self._client = client
        self._lock = asyncio.Lock()

    async def _ensure(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is None:
                try:
                    from redis.asyncio import from_url
                except ImportError as exc:
                    raise ConfigurationError(
                        "RedisMemory needs a driver — " + self.driver_hint) from exc
                self._client = from_url(self.dsn, decode_responses=True,
                                        **self.options)
        return self._client

    # ---- keys -----------------------------------------------------------------
    def _records_key(self, namespace: str, scope: str) -> str:
        return f"{self.prefix}:rec:{namespace}:{scope}"

    def _doc_key(self, name: str, trace: Trace | None) -> str:
        return f"{self.prefix}:doc:{self.doc_key(name, trace)}"

    @staticmethod
    def _namespace(trace: Trace | None, record: MemoryRecord | None = None) -> str:
        if trace is not None:
            return trace.slug
        if record is not None:
            return Trace(user_id=record.user_id, session_id=record.session_id,
                         tenant_id=record.tenant_id).slug
        return "_shared"

    async def _scopes(self, namespace: str) -> list[str]:
        client = await self._ensure()
        pattern = f"{self.prefix}:rec:{namespace}:*"
        return [key.rsplit(":", 1)[-1] async for key in client.scan_iter(pattern)]

    # ---- records ---------------------------------------------------------------
    async def append(self, record: MemoryRecord) -> MemoryRecord:
        client = await self._ensure()
        key = self._records_key(self._namespace(None, record), record.scope)
        pipe = client.pipeline()
        pipe.zadd(key, {record.model_dump_json(): record.ts})
        if self.ttl:
            pipe.expire(key, self.ttl)
        await pipe.execute()
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        client = await self._ensure()
        namespace = self._namespace(trace)
        scopes = [scope] if scope else await self._scopes(namespace)

        rows: list[MemoryRecord] = []
        for one in scopes:
            key = self._records_key(namespace, one)
            # Newest first from Redis, so a limit is applied server-side.
            raw = await client.zrevrange(key, 0, (limit - 1) if limit else -1)
            for blob in raw:
                try:
                    rows.append(MemoryRecord(**json.loads(blob)))
                except (json.JSONDecodeError, ValueError):
                    continue
        if kind:
            rows = [r for r in rows if r.kind == kind]
        if trace is not None:
            rows = [r for r in rows if trace.matches(r)]
        rows.sort(key=lambda r: r.ts)
        return rows[-limit:] if limit else rows

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        client = await self._ensure()
        namespace = self._namespace(trace)
        scopes = [scope] if scope else await self._scopes(namespace)
        keys = [self._records_key(namespace, s) for s in scopes]
        if trace is not None or scope is None:
            pattern = (f"{self.prefix}:doc:{trace.slug}/*" if trace is not None
                       else f"{self.prefix}:doc:*")
            keys += [key async for key in client.scan_iter(pattern)]
        if keys:
            await client.delete(*keys)

    # ---- documents ---------------------------------------------------------------
    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        client = await self._ensure()
        return await client.get(self._doc_key(name, trace)) or ""

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        client = await self._ensure()
        key = self._doc_key(name, trace)
        # Documents outlive records by design, so the TTL does not apply here.
        await client.set(key, text)
        await client.hset(f"{self.prefix}:doc_meta", key, str(time.time()))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
