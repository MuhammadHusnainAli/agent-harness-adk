"""Where conversations are kept.

    from agent_harness import Harness, session_provider

    Harness(sessions=session_provider("postgresql://user:pass@host/agents"))
    Harness(sessions="sqlite:///./chats.db")                 # a URL works too
    Harness(sessions=session_provider("azure://chats", connection_string="..."))

| store | URL | install |
|---|---|---|
| in process | `memory://` | — |
| files | `file:///path` or a bare path | — |
| SQLite | `sqlite:///chats.db` | — (standard library) |
| PostgreSQL | `postgresql://…` | `asyncpg` or `psycopg` |
| MySQL / MariaDB | `mysql://…` | `aiomysql` |
| MongoDB, Cosmos DB (Mongo API) | `mongodb://…` | `motor` or `pymongo` |
| Redis | `redis://…` | `redis` |
| DynamoDB | `dynamodb://table` | `boto3` |
| Amazon S3 | `s3://bucket/prefix` | `boto3` |
| Azure Blob (a storage account) | `azure://container` | `azure-storage-blob` |
| Google Cloud Storage | `gs://bucket/prefix` | `google-cloud-storage` |

The URLs, the drivers and the connection handling are the memory backends' own
(`agent_harness.memory.providers`), so there is one place where a database is
configured — and `session_provider(harness.memory_store)` keeps chats on the
connections memory already holds.

Every one of them keeps a session with its owner and its version, refuses a
stale save, and can prove it: `await store.check()`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ..errors import ConfigurationError
from ..runtime.session import (
    DurableSessionStore,
    FileSessionStore,
    InMemorySessionStore,
    Session,
    SessionStore,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from .dynamodb import DynamoDBSessionStore
    from .mongo import MongoSessionStore
    from .objects import ObjectSessionStore
    from .redis import RedisSessionStore
    from .sql import SQLSessionStore

__all__ = [
    "Session",
    "SessionStore",
    "DurableSessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
    "SQLSessionStore",
    "MongoSessionStore",
    "RedisSessionStore",
    "DynamoDBSessionStore",
    "ObjectSessionStore",
    "session_provider",
    "session_backends",
]

#: memory backend name -> (module, class) of the session store that sits on it.
STORES: dict[str, tuple[str, str]] = {
    "sqlite": ("agent_harness.sessions.sql", "SQLSessionStore"),
    "postgres": ("agent_harness.sessions.sql", "SQLSessionStore"),
    "mysql": ("agent_harness.sessions.sql", "SQLSessionStore"),
    "mongo": ("agent_harness.sessions.mongo", "MongoSessionStore"),
    "redis": ("agent_harness.sessions.redis", "RedisSessionStore"),
    "dynamodb": ("agent_harness.sessions.dynamodb", "DynamoDBSessionStore"),
    "s3": ("agent_harness.sessions.objects", "ObjectSessionStore"),
    "azure": ("agent_harness.sessions.objects", "ObjectSessionStore"),
    "gcs": ("agent_harness.sessions.objects", "ObjectSessionStore"),
}

_EXPORTS = {cls: module for module, cls in STORES.values()}

# Options that belong to the session store rather than to the connection.
_STORE_OPTIONS = {"sql": ("table",), "mongo": ("collection",),
                  "redis": ("ttl",), "objects": ("concurrency",)}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


def session_backends() -> dict[str, bool]:
    """Which stores can be used right now, without installing anything."""
    from ..memory.providers import available

    ready = available()
    return {"memory": True, "file": True,
            **{name: ready.get(name, False) for name in STORES}}


def _family(memory_store: Any) -> str | None:
    """Which kind of backend this is, by what it can do rather than its name —
    so a backend of your own that follows one of the shapes is taken too."""
    if hasattr(memory_store, "_execute") and hasattr(memory_store, "_fetch"):
        return "sql"
    if hasattr(memory_store, "_put") and hasattr(memory_store, "_list"):
        return "objects"
    return {"MongoMemory": "mongo", "RedisMemory": "redis",
            "DynamoDBMemory": "dynamodb"}.get(type(memory_store).__name__)


def _wrap(memory_store: Any, options: dict[str, Any]) -> SessionStore:
    # A semantic layer wraps the real store; the sessions go underneath it.
    inner = getattr(memory_store, "store", None)
    if inner is not None and _family(memory_store) is None:
        memory_store = inner
    family = _family(memory_store)
    if family is None:
        raise ConfigurationError(
            f"{type(memory_store).__name__} cannot hold sessions; use SQL, MongoDB, "
            "Redis, DynamoDB or an object store — or write a SessionStore")
    module = {"sql": "sql", "objects": "objects", "mongo": "mongo", "redis": "redis",
              "dynamodb": "dynamodb"}[family]
    cls = {"sql": "SQLSessionStore", "objects": "ObjectSessionStore",
           "mongo": "MongoSessionStore", "redis": "RedisSessionStore",
           "dynamodb": "DynamoDBSessionStore"}[family]
    try:
        return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(
            memory_store, **options)
    except TypeError as exc:
        raise ConfigurationError(f"{cls}: {exc}") from None


def session_provider(target: Any, **options: Any) -> SessionStore:
    """A session store from a connection string, or on a memory backend you
    already have.

        session_provider("postgresql://user:pass@host/agents", table="chats")
        session_provider("redis://host:6379/0", ttl=30 * 86_400)
        session_provider("s3://my-bucket/chats")
        session_provider(harness.memory_store)        # the same pool as memory
    """
    if isinstance(target, SessionStore):
        return target
    if not isinstance(target, str):
        return _wrap(target, options)

    scheme = urlparse(target).scheme.lower()
    if scheme == "memory":
        return InMemorySessionStore()
    if scheme in ("", "file"):
        return FileSessionStore(urlparse(target).path or target or ".harness/sessions")

    from ..memory.providers import SCHEMES, memory_provider

    name = SCHEMES.get(scheme)
    if name not in STORES:
        known = sorted(s for s, n in SCHEMES.items() if n in STORES and s)
        raise ConfigurationError(
            f"no session store for {scheme!r}; known: memory, file, "
            + ", ".join(known))
    family = {"sqlite": "sql", "postgres": "sql", "mysql": "sql", "s3": "objects",
              "azure": "objects", "gcs": "objects"}.get(name, name)
    mine = {k: options.pop(k) for k in _STORE_OPTIONS.get(family, ()) if k in options}
    return _wrap(memory_provider(target, **options), mine)
