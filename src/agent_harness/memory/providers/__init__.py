"""Where memory is stored.

Every backend implements `MemoryStore`, so they are interchangeable — the agent
loop never learns which one it is talking to. Each is scoped by `Trace`, so one
store serves many users without them seeing each other.

    from agent_harness.memory.providers import memory_provider

    store = memory_provider("postgresql://user:pass@host/agents")
    store = memory_provider("s3://my-bucket/memory")
    store = memory_provider("sqlite:///./memory.db")

| backend | class | install |
|---|---|---|
| in-process | `InMemoryStore` | — |
| files | `FileStore` | — |
| SQLite | `SQLiteMemory` | — (standard library) |
| PostgreSQL | `PostgresMemory` | `asyncpg` or `psycopg` |
| MySQL / MariaDB | `MySQLMemory` | `aiomysql` |
| MongoDB | `MongoMemory` | `motor` or `pymongo` |
| Redis | `RedisMemory` | `redis` |
| DynamoDB | `DynamoDBMemory` | `boto3` |
| Elasticsearch | `ElasticsearchMemory` | `elasticsearch` |
| Amazon S3 | `S3Memory` | `boto3` |
| Azure Blob | `AzureBlobMemory` | `azure-storage-blob` |
| Google Cloud Storage | `GCSMemory` | `google-cloud-storage` |
| your own API | `HTTPMemory` | — (httpx ships with the harness) |

Nothing is imported until you ask for it: a driver you do not use costs nothing,
and one you have not installed says exactly which `pip install` fixes it.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ...errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from ..base import MemoryStore
    from .azure_blob import AzureBlobMemory
    from .dynamodb import DynamoDBMemory
    from .elasticsearch import ElasticsearchMemory
    from .gcs import GCSMemory
    from .http import HTTPMemory
    from .mongo import MongoMemory
    from .mysql import MySQLMemory
    from .postgres import PostgresMemory
    from .redis import RedisMemory
    from .s3 import S3Memory
    from .sqlite import SQLiteMemory

__all__ = [
    "SQLiteMemory",
    "PostgresMemory",
    "MySQLMemory",
    "MongoMemory",
    "RedisMemory",
    "DynamoDBMemory",
    "ElasticsearchMemory",
    "S3Memory",
    "AzureBlobMemory",
    "GCSMemory",
    "HTTPMemory",
    "memory_provider",
    "register_backend",
    "available",
    "BACKENDS",
]

#: name -> (module, class). The module is imported on first use, never before.
BACKENDS: dict[str, tuple[str, str]] = {
    "memory": ("agent_harness.memory.base", "InMemoryStore"),
    "file": ("agent_harness.memory.base", "FileStore"),
    "sqlite": ("agent_harness.memory.providers.sqlite", "SQLiteMemory"),
    "postgres": ("agent_harness.memory.providers.postgres", "PostgresMemory"),
    "mysql": ("agent_harness.memory.providers.mysql", "MySQLMemory"),
    "mongo": ("agent_harness.memory.providers.mongo", "MongoMemory"),
    "redis": ("agent_harness.memory.providers.redis", "RedisMemory"),
    "dynamodb": ("agent_harness.memory.providers.dynamodb", "DynamoDBMemory"),
    "elasticsearch": ("agent_harness.memory.providers.elasticsearch",
                      "ElasticsearchMemory"),
    "s3": ("agent_harness.memory.providers.s3", "S3Memory"),
    "azure": ("agent_harness.memory.providers.azure_blob", "AzureBlobMemory"),
    "gcs": ("agent_harness.memory.providers.gcs", "GCSMemory"),
    "http": ("agent_harness.memory.providers.http", "HTTPMemory"),
}

#: URL scheme -> backend name.
SCHEMES: dict[str, str] = {
    "memory": "memory", "": "file", "file": "file",
    "sqlite": "sqlite", "sqlite3": "sqlite",
    "postgres": "postgres", "postgresql": "postgres",
    "mysql": "mysql", "mariadb": "mysql",
    "mongodb": "mongo", "mongodb+srv": "mongo", "mongo": "mongo",
    "redis": "redis", "rediss": "redis",
    "dynamodb": "dynamodb",
    "es": "elasticsearch", "elasticsearch": "elasticsearch",
    "opensearch": "elasticsearch",
    "s3": "s3", "minio": "s3",
    "azure": "azure", "abfs": "azure", "blob": "azure",
    "gs": "gcs", "gcs": "gcs",
    "http": "http", "https": "http",
}

#: backend -> the driver modules it can use. Any one of them is enough; an empty
#: list means the backend needs nothing beyond what the harness already ships.
DRIVERS: dict[str, list[str]] = {
    "memory": [], "file": [], "sqlite": [], "http": [],
    "postgres": ["asyncpg", "psycopg_pool"],
    "mysql": ["aiomysql"],
    "mongo": ["motor", "pymongo"],
    "redis": ["redis"],
    "dynamodb": ["boto3"],
    "elasticsearch": ["elasticsearch"],
    "s3": ["boto3"],
    "azure": ["azure.storage.blob"],
    "gcs": ["google.cloud.storage"],
}

#: driver module -> what you actually type into pip. The module name is not
#: always the package name, and a wrong hint is worse than none.
PIP_NAMES: dict[str, str] = {
    "asyncpg": "asyncpg",
    "psycopg_pool": "'psycopg[binary,pool]'",
    "aiomysql": "aiomysql",
    "motor": "motor",
    "pymongo": "pymongo",
    "redis": "redis",
    "boto3": "boto3",
    "elasticsearch": "elasticsearch",
    "azure.storage.blob": "azure-storage-blob aiohttp",
    "google.cloud.storage": "google-cloud-storage",
}

#: Classes are exported lazily, so `from ... import S3Memory` still works.
_EXPORTS = {cls: module for module, cls in BACKENDS.values()}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


def register_backend(name: str, module: str, cls: str, *,
                     schemes: list[str] | None = None,
                     drivers: list[str] | None = None) -> None:
    """Add your own backend: `register_backend("cassandra", "myapp.mem", "Cassandra")`."""
    BACKENDS[name] = (module, cls)
    DRIVERS[name] = list(drivers or [])
    _EXPORTS[cls] = module
    for scheme in schemes or [name]:
        SCHEMES[scheme] = name


def get_backend(name: str) -> type:
    """The class for a backend name, imported on demand."""
    if name not in BACKENDS:
        raise ConfigurationError(
            f"unknown memory backend {name!r}; known: {', '.join(sorted(BACKENDS))}")
    module, cls = BACKENDS[name]
    return getattr(importlib.import_module(module), cls)


def _installed(module: str) -> bool:
    """Is this driver importable? Checked without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def available() -> dict[str, bool]:
    """Which backends can be used right now, without installing anything.

    Checks the *driver*, not our wrapper — every wrapper imports fine, which is
    the point of loading drivers lazily.
    """
    return {
        name: (not drivers) or any(_installed(d) for d in drivers)
        for name, drivers in DRIVERS.items()
    }


def missing_driver(name: str) -> str:
    """What to install for a backend that is not ready. Empty if it is."""
    drivers = DRIVERS.get(name, [])
    if not drivers or any(_installed(d) for d in drivers):
        return ""
    return " or ".join(f"pip install {PIP_NAMES.get(d, d)}" for d in drivers)


def memory_provider(url: str, **options: Any) -> MemoryStore:
    """Build a store from a connection string.

        memory_provider("sqlite:///./memory.db")
        memory_provider("postgresql://user:pass@host/agents")
        memory_provider("s3://my-bucket/agent-memory")
        memory_provider("mongodb://localhost:27017", database="agents")
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    name = SCHEMES.get(scheme)
    if name is None:
        raise ConfigurationError(
            f"no memory backend for {scheme or 'a bare path'!r}; known schemes: "
            + ", ".join(sorted(s for s in SCHEMES if s))
        )
    cls = get_backend(name)

    # Each family reads a URL differently; do the translation in one place.
    if name in {"memory"}:
        return cls()
    if name == "file":
        return cls(parsed.path or url or ".harness/memory")
    if name == "sqlite":
        path = (parsed.netloc + parsed.path) or ":memory:"
        return cls(path.lstrip("/") if path.startswith("//") else path, **options)
    if name in {"s3", "azure", "gcs"}:
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
        if prefix:
            options.setdefault("prefix", prefix)
        return cls(bucket, **options)
    if name == "dynamodb":
        return cls(parsed.netloc or parsed.path.lstrip("/") or "agent_memory",
                   **options)
    if name == "elasticsearch":
        inner = url.split("://", 1)[1]
        secure = scheme in {"es", "elasticsearch"} and inner.startswith("https")
        host = inner if "://" in inner else f"http{'s' if secure else ''}://{inner}"
        return cls(host, **options)
    if name == "http":
        return cls(url, **options)
    if name == "mongo":
        database = parsed.path.strip("/")
        if database:
            options.setdefault("database", database)
        return cls(url.split("?")[0].rsplit("/", 1)[0] if database else url,
                   **options)
    return cls(url, **options)
