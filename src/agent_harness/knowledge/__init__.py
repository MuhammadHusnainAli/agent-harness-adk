"""Knowledge bases, over any vector database.

    from agent_harness import KnowledgeBase

    kb = KnowledgeBase("qdrant://localhost:6333/handbook", embedder=...)
    await kb.add(path="handbook.md")
    agent = Agent("support", tools=[kb.as_tool()])

`KnowledgeBase` is the documents-in, passages-out layer; `vector_store()` is
the database under it, and `vector_stores()` lists the ones that ship.
"""

from .base import (
    HTTPVectorStore,
    VectorHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
    conditions,
    cosine,
    normalise,
)
from .kb import Document, KnowledgeBase, Passage, chunk_text
from .stores import STORES, vector_store, vector_stores

__all__ = [
    "KnowledgeBase",
    "Passage",
    "Document",
    "chunk_text",
    "vector_store",
    "vector_stores",
    "STORES",
    "VectorStore",
    "HTTPVectorStore",
    "VectorRecord",
    "VectorHit",
    "VectorStoreError",
    "conditions",
    "cosine",
    "normalise",
]
