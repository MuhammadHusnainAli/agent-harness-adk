"""The vector databases that ship. Each is imported when it is first asked for."""

from __future__ import annotations

import importlib
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from ...errors import ConfigurationError
from ..base import VectorStore

__all__ = ["STORES", "vector_store", "vector_stores"]

#: name → (module, class, what it needs beyond the three dependencies)
STORES: dict[str, tuple[str, str, str]] = {
    "memory": ("local", "MemoryVectorStore", ""),
    "sqlite": ("local", "SQLiteVectorStore", ""),
    "qdrant": ("qdrant", "QdrantStore", ""),
    "chroma": ("chroma", "ChromaStore", ""),
    "weaviate": ("weaviate", "WeaviateStore", ""),
    "milvus": ("milvus", "MilvusStore", ""),
    "pinecone": ("pinecone", "PineconeStore", ""),
    "opensearch": ("opensearch", "OpenSearchStore", ""),
    "elasticsearch": ("opensearch", "ElasticsearchStore", ""),
    "pgvector": ("pgvector", "PgVectorStore", "asyncpg"),
    "redis": ("redis", "RedisVectorStore", "redis"),
    "mongodb": ("mongodb", "MongoVectorStore", "pymongo"),
    "azure-ai-search": ("azure", "AzureAISearchStore", ""),
    "cloudflare-vectorize": ("cloudflare", "VectorizeStore", ""),
    "upstash": ("upstash", "UpstashStore", ""),
    "vertex-vector-search": ("vertex", "VertexVectorStore", ""),
    "s3-vectors": ("aws", "S3VectorsStore", ""),
}
_ALIASES = {
    "zilliz": "milvus", "postgres": "pgvector", "postgresql": "pgvector",
    "supabase": "pgvector", "neon": "pgvector", "elastic": "elasticsearch",
    "es": "elasticsearch", "aoss": "opensearch", "atlas": "mongodb", "mongo": "mongodb",
    "mongodb+srv": "mongodb", "azure": "azure-ai-search", "azure-search": "azure-ai-search",
    "vectorize": "cloudflare-vectorize", "cloudflare": "cloudflare-vectorize",
    "vertex": "vertex-vector-search", "vertex-ai": "vertex-vector-search",
    "s3vectors": "s3-vectors", "s3": "s3-vectors", "rediss": "redis", "file": "sqlite",
}


def _class(name: str) -> type[VectorStore]:
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in STORES:
        raise ConfigurationError(
            f"no vector store named {name!r}; there are: {', '.join(STORES)}")
    module, attribute, _ = STORES[key]
    return getattr(importlib.import_module(f".{module}", __name__), attribute)


def vector_stores() -> list[dict[str, str]]:
    """Every store that ships, and the driver it needs, if it needs one."""
    return [{"name": name, "class": attribute, "needs": needs}
            for name, (_, attribute, needs) in STORES.items()]


def vector_store(spec: Any = "memory", **options: Any) -> VectorStore:
    """A vector store from its name, or from a URL.

        vector_store("qdrant", url="http://localhost:6333", collection="handbook")
        vector_store("qdrant://localhost:6333/handbook")
        vector_store("pgvector://user:pass@host/db?table=handbook")
        vector_store("opensearch+https://host:9200/handbook")
        vector_store("sqlite:///knowledge.db")

    In a URL the path names the collection, `+https` asks for TLS, and anything
    after `?` is passed to the store as an option.
    """
    if isinstance(spec, VectorStore):
        return spec
    if not isinstance(spec, str) or not spec.strip():
        raise ConfigurationError(
            "a vector store is a name, a URL, or a VectorStore — got "
            f"{type(spec).__name__}")
    if "://" not in spec:
        return _class(spec)(**options)

    parts = urlsplit(spec)
    scheme, _, secure = parts.scheme.partition("+")
    kind = _class(scheme)
    key = _ALIASES.get(scheme.lower(), scheme.lower())
    given = {k: v for k, v in parse_qsl(parts.query)}
    path = unquote(parts.path.strip("/"))
    web = ("https" if secure == "https" or given.pop("tls", "") in ("1", "true")
           else "http") + f"://{parts.netloc}"

    if key == "memory":
        return kind(**options)
    if key == "sqlite":
        return kind(unquote(parts.path[1:] if parts.netloc == "" else parts.netloc
                            + parts.path) or "knowledge.db", **given, **options)
    if key in ("qdrant", "chroma", "weaviate", "milvus"):
        return kind(web, **({"collection": path} if path else {}), **given, **options)
    if key in ("opensearch", "elasticsearch"):
        return kind(web, **({"index": path} if path else {}), **given, **options)
    if key == "pgvector":
        table = given.pop("table", "knowledge")
        query = "&".join(f"{k}={v}" for k, v in given.items())
        return kind(f"postgresql://{parts.netloc}{parts.path}"
                    + (f"?{query}" if query else ""), table=table, **options)
    if key == "redis":
        index = given.pop("index", "knowledge")
        return kind(f"{parts.scheme}://{parts.netloc}{parts.path}", index=index, **options)
    if key == "mongodb":
        names = {k: given.pop(k) for k in ("database", "collection", "index")
                 if k in given}
        query = "&".join(f"{k}={v}" for k, v in given.items())
        return kind(f"{parts.scheme}://{parts.netloc}{parts.path}"
                    + (f"?{query}" if query else ""), **names, **options)
    if key == "pinecone":
        return kind(parts.netloc, **given, **options)
    if key == "azure-ai-search":
        return kind(f"https://{parts.netloc}", **({"index": path} if path else {}),
                    **given, **options)
    if key == "cloudflare-vectorize":
        return kind(account_id=parts.netloc, **({"index": path} if path else {}),
                    **given, **options)
    if key == "upstash":
        return kind(f"https://{parts.netloc}", **given, **options)
    if key == "s3-vectors":
        return kind(bucket=parts.netloc, **({"index": path} if path else {}),
                    **given, **options)
    if key == "vertex-vector-search":
        region, _, index = path.partition("/")
        return kind(project=parts.netloc, region=region, index=index, **given, **options)
    raise ConfigurationError(f"{scheme!r} cannot be given as a URL — build it directly")
