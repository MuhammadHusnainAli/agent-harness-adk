"""A knowledge base: documents in, the passages that answer a question out.

    from agent_harness import Agent, KnowledgeBase, OpenAIProvider
    from agent_harness.memory import ProviderEmbedder

    kb = KnowledgeBase("qdrant://localhost:6333/handbook",
                       embedder=ProviderEmbedder(OpenAIProvider(), "text-embedding-3-small"))
    await kb.add(path="handbook/refunds.md")
    await kb.add("Orders ship within two working days.", id="shipping", title="Shipping")

    passages = await kb.search("how long do refunds take?")
    agent = Agent("support", tools=[kb.as_tool()])

Adding a document reads it, cuts it into overlapping passages at paragraph and
sentence boundaries, embeds them in batches, and stores them under the
document's id. Adding it again replaces it — and when nothing in it has
changed, nothing is embedded or written. Removing it removes every passage.

A search embeds the question, takes more candidates than were asked for, and
orders them by meaning and by the words they share with the question, so an
exact term — a product code, a name — is not lost to a near-synonym.

The store is any vector database: `vector_store()` lists them. With no
embedder a hashing one is used: free and offline, and blind to meaning — fine
for tests, not for answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError, ToolError
from ..tools import Tool, tool
from .base import VectorHit, VectorRecord, VectorStore, VectorStoreError, conditions
from .stores import vector_store

__all__ = ["KnowledgeBase", "Passage", "Document", "chunk_text"]

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_WORD = re.compile(r"[0-9A-Za-zÀ-￿]+")
#: Metadata the knowledge base writes itself. A document's own may not use them.
RESERVED = ("doc", "chunk", "title", "source", "doc_hash", "doc_chunks", "namespace")


def chunk_text(text: str, *, size: int = 1200, overlap: int = 150) -> list[str]:
    """Cut text into passages of about `size` characters.

    Cuts fall between paragraphs where they can, between sentences where a
    paragraph is too long, and inside a sentence only when the sentence itself
    is. Each passage starts with the tail of the one before, so a fact that
    straddles a cut is whole in one of them — and under the headings it sits
    beneath, so "it takes five days" still says *what* takes five days.
    """
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    pieces: list[tuple[str, str]] = []          # (heading trail, text)
    trail: list[tuple[int, str]] = []
    for block in _PARAGRAPH.split(text.replace("\r\n", "\n")):
        block = block.strip()
        if not block:
            continue
        first = block.split("\n", 1)[0]
        heading = _HEADING.match(first)
        if heading:
            level = len(heading.group(1))
            trail = [(lv, t) for lv, t in trail if lv < level] + [(level, heading.group(2))]
        context = " > ".join(t for _, t in trail)
        if len(block) <= size:
            pieces.append((context, block))
            continue
        for sentence in _SENTENCE.split(block):
            while len(sentence) > size:           # one sentence longer than a passage
                cut = sentence.rfind(" ", size // 2, size)
                cut = cut if cut > 0 else size
                pieces.append((context, sentence[:cut].strip()))
                sentence = sentence[cut:].strip()
            if sentence:
                pieces.append((context, sentence))

    chunks: list[str] = []
    current, where, tail = "", "", ""
    for context, piece in pieces:
        if current and (len(current) + len(piece) + 2 > size or context != where):
            chunks.append(current)
            tail = current[-overlap:] if overlap and context == where else ""
            # Start the overlap at a word, not in the middle of one.
            tail = tail[tail.find(" ") + 1:] if " " in tail else ""
            current = ""
        if not current:
            # Every passage says where in the document it is from.
            label = (f"[{context}]\n" if context and not piece.lstrip().startswith("#")
                     else "")
            current = f"{label}{tail} {piece}".strip() if tail else f"{label}{piece}"
            tail = ""
        else:
            current = f"{current} {piece}"
        where = context
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


@dataclass
class Passage:
    """One stretch of a document that a search found."""

    text: str
    #: Cosine similarity between the question and this passage.
    score: float
    document: str = ""
    title: str = ""
    source: str = ""
    chunk: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text, "score": round(self.score, 3)}
        for key in ("title", "source", "document"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        return out


@dataclass
class Document:
    """What adding a document came to."""

    id: str
    chunks: int
    title: str = ""
    source: str = ""
    #: True when it was already there, unchanged, and nothing was written.
    unchanged: bool = False


Reranker = Callable[[str, list[Passage]], Any]


class KnowledgeBase:
    """Documents, searchable by meaning, in any vector database.

    ``store``          a `VectorStore`, a name, or a URL — `"qdrant://host/handbook"`
    ``embedder``       what turns text into vectors: `ProviderEmbedder(provider, model)`
    ``namespace``      keeps one knowledge base apart from another in the same store;
                       every write is marked with it and every search held to it
    ``chunk_size`` / ``chunk_overlap``  the size of a passage, and how much of the
                       last one each begins with, in characters
    ``hybrid``         order results by shared words as well as by meaning
    ``reranker``       `(query, passages) -> scores`, sync or async — a
                       cross-encoder, a rerank API — applied to the candidates
    ``min_score``      drop passages less similar than this
    ``retriever``      instead of a store: something with `retrieve(query, k=, filter=)`
                       that does its own embedding — a Bedrock knowledge base, say
    """

    def __init__(self, store: Any = None, *, embedder: Any = None, retriever: Any = None,
                 namespace: str = "", chunk_size: int = 1200, chunk_overlap: int = 150,
                 hybrid: bool = True, reranker: Reranker | None = None,
                 min_score: float = 0.0, candidates: int = 4, batch: int = 64,
                 concurrency: int = 4) -> None:
        if retriever is not None and store is not None:
            raise ConfigurationError("a knowledge base has a store or a retriever, not both")
        self.retriever = retriever
        self.store: VectorStore | None = (
            None if retriever is not None else vector_store(store or "memory"))
        if embedder is None:
            from ..memory.semantic import HashEmbedder

            embedder = HashEmbedder()
        self.embedder = embedder
        self.namespace = namespace
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.hybrid = hybrid
        self.reranker = reranker
        self.min_score = min_score
        self.candidates = max(1, int(candidates))
        self.batch = max(1, int(batch))
        self.concurrency = max(1, int(concurrency))
        self._ready = False
        self._dimension = 0

    # ---- embedding ---------------------------------------------------------
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            part = texts[start:start + self.batch]
            vectors = await self.embedder.embed(part)
            if len(vectors) != len(part):
                raise VectorStoreError(
                    f"the embedder returned {len(vectors)} vectors for {len(part)} texts")
            out.extend(vectors)
        if out:
            sizes = {len(v) for v in out}
            if len(sizes) != 1 or (self._dimension and sizes != {self._dimension}):
                raise VectorStoreError(
                    f"the embedder returned vectors of {sorted(sizes)} dimensions"
                    + (f"; this knowledge base holds {self._dimension}"
                       if self._dimension else ""))
            self._dimension = len(out[0])
        return out

    async def _prepare(self) -> VectorStore:
        if self.store is None:
            raise VectorStoreError(
                "this knowledge base reads from a retriever; documents are added "
                "where the retriever keeps them")
        if not self._ready:
            if not self._dimension:
                await self._embed(["dimension probe"])
            await self.store.ensure(self._dimension)
            self._ready = True
        return self.store

    def _scope(self, filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not self.namespace:
            return filter or None
        return {**(filter or {}), "namespace": self.namespace}

    def _chunk_id(self, document: str, number: int) -> str:
        return f"{self.namespace}/{document}#{number}" if self.namespace else (
            f"{document}#{number}")

    # ---- documents -----------------------------------------------------------
    async def _existing(self, store: VectorStore, document: str,
                        vector: list[float]) -> dict[str, Any]:
        """What is stored of a document already: its hash and how many passages."""
        try:
            hits = await store.query(vector, k=1, filter=self._scope({"doc": document}))
        except VectorStoreError:
            return {}
        return hits[0].metadata if hits else {}

    async def add(self, text: str | None = None, *, path: str | Path | None = None,
                  url: str | None = None, id: str | None = None, title: str = "",
                  source: str = "", metadata: dict[str, Any] | None = None) -> Document:
        """Add a document — or replace the one with the same `id`.

        Give it the text, or a `path` (anything `parse_document` reads: text,
        Markdown, HTML, CSV, JSON, and PDF or DOCX with their optional
        installs), or a `url` to fetch.
        """
        if sum(x is not None for x in (text, path, url)) != 1:
            raise ToolError("add takes one of: text, path=, url=", tool="knowledge")
        if path is not None:
            from ..toolkits.documents import parse_document

            file = Path(path).expanduser()
            parsed = await asyncio.to_thread(parse_document, file, max_chars=10_000_000,
                                             max_pages=5_000)
            text = parsed.get("text") or ""
            source = source or str(file)
            title = title or file.name
        elif url is not None:
            from ..toolkits.web import make_fetch_tool

            text = await make_fetch_tool(max_chars=10_000_000).invoke({"url": url})
            source = source or url
            title = title or url
        text = (text or "").replace("\x00", "").strip()
        if not text:
            raise ToolError("there is no text in that document", tool="knowledge")
        extra = dict(metadata or {})
        clash = [key for key in extra if key in RESERVED]
        if clash:
            raise ToolError(f"metadata may not use the names {', '.join(clash)} — the "
                            "knowledge base writes those itself", tool="knowledge")
        document = id or source or hashlib.sha256(text.encode()).hexdigest()[:16]

        chunks = chunk_text(text, size=self.chunk_size, overlap=self.chunk_overlap)
        digest = hashlib.sha256(repr((text, sorted(extra.items(), key=str), title, source,
                                      self.chunk_size, self.chunk_overlap)).encode()
                                ).hexdigest()[:24]
        store = await self._prepare()
        first = (await self._embed(chunks[:1]))[0]
        held = await self._existing(store, document, first)
        if held.get("doc_hash") == digest and held.get("doc_chunks") == len(chunks):
            return Document(document, len(chunks), title, source, unchanged=True)

        vectors = [first, *(await self._embed(chunks[1:]))] if len(chunks) > 1 else [first]
        base = {**extra, "doc": document, "title": title, "source": source,
                "doc_hash": digest, "doc_chunks": len(chunks),
                **({"namespace": self.namespace} if self.namespace else {})}
        await store.upsert(VectorRecord(self._chunk_id(document, n), vectors[n], chunk,
                                        {**base, "chunk": n})
                           for n, chunk in enumerate(chunks))
        # It was longer before: the passages past its new end are no longer its.
        before = int(held.get("doc_chunks") or 0)
        if before > len(chunks):
            await store.delete([self._chunk_id(document, n)
                                for n in range(len(chunks), before)])
        return Document(document, len(chunks), title, source)

    async def add_many(self, documents: Iterable[dict[str, Any] | str | Path]
                       ) -> list[Document]:
        """Add several — each a mapping of `add`'s arguments, or a path."""
        gate = asyncio.Semaphore(self.concurrency)

        async def one(item: Any) -> Document:
            async with gate:
                if isinstance(item, dict):
                    return await self.add(**item)
                return await self.add(path=item)

        await self._prepare()          # once, not once per document
        return list(await asyncio.gather(*(one(item) for item in documents)))

    async def delete(self, document: str) -> None:
        """Remove a document and every passage of it."""
        store = await self._prepare()
        try:
            await store.delete(filter=self._scope({"doc": document}))
            return
        except VectorStoreError:
            pass
        # A store that deletes by id only: ask how many passages there are.
        probe = (await self._embed([document]))[0]
        held = await self._existing(store, document, probe)
        count = int(held.get("doc_chunks") or 0)
        if count:
            await store.delete([self._chunk_id(document, n) for n in range(count)])

    async def count(self) -> int:
        """How many passages are stored (in the whole store, not this namespace)."""
        return await (await self._prepare()).count()

    # ---- searching ------------------------------------------------------------
    @staticmethod
    def _lexical(query: str, hits: list[VectorHit]) -> list[float]:
        """How much each candidate shares the question's words — rare ones most."""
        terms = [w.lower() for w in _WORD.findall(query) if len(w) > 1]
        if not terms or not hits:
            return [0.0] * len(hits)
        bags = [Counter(w.lower() for w in _WORD.findall(hit.text)) for hit in hits]
        average = sum(sum(bag.values()) for bag in bags) / len(bags) or 1.0
        scores = []
        for bag in bags:
            length = sum(bag.values()) or 1
            total = 0.0
            for term in set(terms):
                seen = bag.get(term, 0)
                if not seen:
                    continue
                having = sum(1 for other in bags if term in other)
                idf = math.log(1 + (len(bags) - having + 0.5) / (having + 0.5))
                total += idf * seen * 2.2 / (seen + 1.2 * (0.25 + 0.75 * length / average))
            scores.append(total)
        return scores

    async def search(self, query: str, *, k: int = 5,
                     filter: dict[str, Any] | None = None) -> list[Passage]:
        """The passages that best answer `query`, best first."""
        query = (query or "").strip()
        if not query:
            raise ToolError("the query is empty — say what to look for", tool="knowledge")
        k = max(1, int(k))
        conditions(filter)                      # a bad filter fails before any call
        if self.retriever is not None:
            hits = list(await self.retriever.retrieve(query, k=k * self.candidates
                                                      if self.reranker else k,
                                                      filter=self._scope(filter)))
        else:
            store = await self._prepare()
            vector = (await self._embed([query]))[0]
            wide = self.hybrid or self.reranker is not None
            hits = await store.query(vector, k=min(store.max_k, k * self.candidates)
                                     if wide else k, filter=self._scope(filter))
        hits = [hit for hit in hits if hit.score >= self.min_score and hit.text.strip()]

        order = list(range(len(hits)))
        if self.hybrid and len(hits) > 1:
            lexical = self._lexical(query, hits)
            by_words = sorted(order, key=lambda n: -lexical[n])
            rank = {n: place for place, n in enumerate(by_words)}
            # Reciprocal rank fusion: a passage near the top of either list rises.
            fused = {n: 1.0 / (60 + n) + (1.0 / (60 + rank[n]) if lexical[n] > 0 else 0.0)
                     for n in order}
            order.sort(key=lambda n: -fused[n])

        seen: set[str] = set()
        passages: list[Passage] = []
        for n in order:
            hit = hits[n]
            key = " ".join(hit.text.split())[:400]
            if key in seen:                     # the same words twice add nothing
                continue
            seen.add(key)
            meta = dict(hit.metadata)
            passages.append(Passage(
                text=hit.text, score=hit.score, id=hit.id,
                document=str(meta.pop("doc", "") or ""), title=str(meta.pop("title", "") or ""),
                source=str(meta.pop("source", "") or ""),
                chunk=int(meta.pop("chunk", 0) or 0),
                metadata={key: value for key, value in meta.items()
                          if key not in RESERVED}))

        if self.reranker is not None and len(passages) > 1:
            scores = self.reranker(query, passages)
            if inspect.isawaitable(scores):
                scores = await scores
            scores = list(scores)
            if len(scores) != len(passages):
                raise VectorStoreError(
                    f"the reranker returned {len(scores)} scores for "
                    f"{len(passages)} passages")
            passages = [p for _, p in sorted(zip(scores, passages, strict=True),
                                             key=lambda pair: -float(pair[0]))]
        return passages[:k]

    # ---- as a tool -------------------------------------------------------------
    def as_tool(self, name: str = "search_knowledge", *, description: str | None = None,
                k: int = 5, max_k: int = 10, filter: dict[str, Any] | None = None,
                max_chars: int = 1500) -> Tool:
        """The knowledge base as a tool an agent can search.

        `filter` is applied to every search the agent makes and cannot be
        removed by it — the way to give one agent one tenant's documents.
        """
        kb = self
        fixed = dict(filter or {})

        @tool(name=name, description=description, tags=["builtin", "knowledge", "search"])
        async def search_knowledge(query: str, limit: int = k) -> Any:
            """Search the knowledge base and return the passages that best answer the query.

            Each passage comes with the document it is from: say which you used.
            Search again with different words when the passages miss.

            Args:
                query: what you want to know, as a question or a few key terms.
                limit: how many passages to return.
            """
            found = await kb.search(query, k=max(1, min(int(limit or k), max_k)),
                                    filter=fixed or None)
            if not found:
                return (f"Nothing in the knowledge base matches {query!r}. Try other "
                        "words, or say that it is not covered.")
            out = []
            for number, passage in enumerate(found, 1):
                row = passage.to_dict()
                if len(row["text"]) > max_chars:
                    row["text"] = row["text"][:max_chars].rsplit(" ", 1)[0] + "…"
                out.append({"n": number, **row})
            return out

        return search_knowledge

    async def aclose(self) -> None:
        if self.store is not None:
            await self.store.aclose()

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<KnowledgeBase {self.store or self.retriever}>"
