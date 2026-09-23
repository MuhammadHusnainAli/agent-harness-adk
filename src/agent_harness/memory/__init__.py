"""Memory: four scopes, one manager, and pluggable storage underneath."""

from .base import FileStore, InMemoryStore, MemoryRecord, MemoryStore
from .manager import (
    MemoryManager,
    OrchestratorMemory,
    SessionMemory,
    SubAgentMemory,
    UserMemory,
)
from .providers import available as available_backends
from .providers import memory_provider, register_backend
from .semantic import (
    Embedder,
    HashEmbedder,
    ProviderEmbedder,
    SemanticMemory,
    VectorStore,
    cosine,
)
from .trace import Scope, Trace

__all__ = [
    "MemoryRecord",
    "MemoryStore",
    "InMemoryStore",
    "FileStore",
    "MemoryManager",
    "UserMemory",
    "SessionMemory",
    "OrchestratorMemory",
    "SubAgentMemory",
    "SemanticMemory",
    "VectorStore",
    "Embedder",
    "HashEmbedder",
    "ProviderEmbedder",
    "cosine",
    "Trace",
    "Scope",
    "memory_provider",
    "register_backend",
    "available_backends",
]
