"""One interface over every vector database.

A vector store keeps records — an id, a vector, the text it came from, some
metadata — and finds the ones nearest a query vector. Every store here does
those four things the same way:

    await store.ensure(1536)                         # the collection, if it is not there
    await store.upsert([VectorRecord("a", vector, "the text", {"source": "faq.md"})])
    hits = await store.query(vector, k=5, filter={"source": "faq.md"})
    await store.delete(filter={"source": "faq.md"})

**Scores** are cosine similarity, higher is nearer, whatever the database
reports natively — so a threshold means the same thing on every backend.

**Filters** are one small language, translated for each database:

    {"source": "faq.md"}                             # equals
    {"lang": {"$in": ["en", "fr"]}}                  # any of
    {"year": {"$gte": 2024}, "draft": {"$ne": True}} # ranges, not-equal; keys are ANDed

**Ids** are yours: any string. A database that only takes UUIDs or integers is
given one derived from yours, and yours is what comes back.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..errors import HarnessError
from ..llm_providers.resilience import RetryPolicy, retry_after

__all__ = [
    "VectorRecord",
    "VectorHit",
    "VectorStore",
    "VectorStoreError",
    "HTTPVectorStore",
    "conditions",
    "cosine",
    "normalise",
]

_Timeout = (TimeoutError, asyncio.TimeoutError)
OPS = ("$eq", "$ne", "$in", "$gt", "$gte", "$lt", "$lte")
#: What `KnowledgeBase` and `SemanticMemory` filter on. A store that has to be
#: told which fields are filterable is told these without being asked.
ALWAYS_FILTERABLE = ("doc", "namespace", "scope", "user_id", "tenant_id", "session_id")
#: The namespace ids are derived in, for stores that will not take a string.
_NAMESPACE = uuid.UUID("6f1d6c1e-1a4e-4c0b-9a51-7c0f3a2b9d10")


class VectorStoreError(HarnessError):
    """A vector database refused something, or could not be reached."""

    def __init__(self, message: str, *, store: str = "", status: int | None = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.store = store
        self.status = status
        self.retryable = retryable


@dataclass
class VectorRecord:
    """One thing to remember: where it sits, what it says, what it is about."""

    id: str
    vector: list[float]
    text: str = ""
    #: Flat: strings, numbers, booleans, and lists of strings.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    """One record a query found."""

    id: str
    #: Cosine similarity: 1.0 is the same direction, 0 is unrelated.
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def normalise(vector: list[float]) -> list[float]:
    """The same direction, length one — so a dot product is a cosine."""
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else list(vector)


def conditions(filter: dict[str, Any] | None) -> list[tuple[str, str, Any]]:
    """A filter as a list of `(field, operator, value)`, all of which must hold."""
    out: list[tuple[str, str, Any]] = []
    for key, value in (filter or {}).items():
        if key == "$and":
            for part in value or []:
                out.extend(conditions(part))
            continue
        if key.startswith("$"):
            raise VectorStoreError(
                f"filters take $and at the top level, and {', '.join(OPS)} on a "
                f"field — not {key!r}")
        if isinstance(value, dict):
            for op, operand in value.items():
                if op not in OPS:
                    raise VectorStoreError(
                        f"unknown filter operator {op!r} on {key!r}; there are: "
                        f"{', '.join(OPS)}")
                if op == "$in" and not isinstance(operand, (list, tuple)):
                    raise VectorStoreError(f"$in on {key!r} takes a list")
                out.append((key, op, list(operand) if op == "$in" else operand))
        else:
            out.append((key, "$eq", value))
    return out


def holds(metadata: dict[str, Any], wanted: list[tuple[str, str, Any]]) -> bool:
    """Does this metadata satisfy every condition? For stores that filter here."""
    for name, op, value in wanted:
        found = metadata.get(name)
        if op == "$eq":
            ok = (value in found) if isinstance(found, list) else found == value
        elif op == "$ne":
            ok = (value not in found) if isinstance(found, list) else found != value
        elif op == "$in":
            ok = (bool(set(found) & set(value)) if isinstance(found, list)
                  else found in value)
        else:
            try:
                ok = found is not None and {
                    "$gt": found > value, "$gte": found >= value,
                    "$lt": found < value, "$lte": found <= value}[op]
            except TypeError:
                ok = False
        if not ok:
            return False
    return True


def as_uuid(record_id: str) -> str:
    """A UUID for a store that insists on one: the id itself if it is one
    already, otherwise one derived from it — the same every time."""
    try:
        return str(uuid.UUID(record_id))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(_NAMESPACE, str(record_id)))


def literal(value: Any) -> str:
    """A value as it is written inside a filter expression."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


class VectorStore(ABC):
    """A vector database. Five methods make one."""

    #: What this store is called in errors and listings.
    name = "vector-store"
    #: The most records one upsert request carries.
    batch_size = 100
    #: The most results one query may ask for.
    max_k = 1000

    @abstractmethod
    async def ensure(self, dimension: int) -> None:
        """Create the collection if it is not there. Safe to call every time."""

    @abstractmethod
    async def _upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    async def _query(self, vector: list[float], k: int,
                     wanted: list[tuple[str, str, Any]]) -> list[VectorHit]: ...

    @abstractmethod
    async def _delete(self, ids: list[str] | None,
                      wanted: list[tuple[str, str, Any]]) -> None: ...

    @abstractmethod
    async def count(self) -> int:
        """How many records it holds."""

    async def aclose(self) -> None:  # noqa: B027 - optional, not abstract
        """Let go of connections."""

    # ---- what callers use ---------------------------------------------------
    async def upsert(self, records: Iterable[VectorRecord]) -> int:
        """Add records, replacing any that have the same id. Returns how many."""
        rows = list(records)
        for row in rows:
            if not row.id or not isinstance(row.id, str):
                raise VectorStoreError("every record needs an id that is a string",
                                       store=self.name)
            if not row.vector or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or x != x or x in (float("inf"), float("-inf"))
                                     for x in row.vector):
                raise VectorStoreError(
                    f"record {row.id!r} has no usable vector: it is empty, or holds "
                    "something that is not a finite number", store=self.name)
        for start in range(0, len(rows), self.batch_size):
            await self._upsert(rows[start:start + self.batch_size])
        return len(rows)

    async def query(self, vector: list[float], *, k: int = 5,
                    filter: dict[str, Any] | None = None) -> list[VectorHit]:
        """The `k` records nearest `vector`, nearest first."""
        if not vector:
            raise VectorStoreError("a query needs a vector", store=self.name)
        hits = await self._query(list(vector), max(1, min(int(k), self.max_k)),
                                 conditions(filter))
        hits.sort(key=lambda hit: -hit.score)
        return hits[:k]

    async def delete(self, ids: Iterable[str] | None = None, *,
                     filter: dict[str, Any] | None = None) -> None:
        """Remove records: these ids, or everything a filter matches."""
        wanted = conditions(filter)
        listed = None if ids is None else [i for i in ids if i]
        if listed is None and not wanted:
            raise VectorStoreError(
                "delete takes ids or a filter — to empty the store, use clear()",
                store=self.name)
        if listed is not None and not listed and not wanted:
            return
        await self._delete(listed, wanted)

    async def clear(self) -> None:
        """Remove every record. Stores that can do it cheaply override this."""
        raise VectorStoreError(f"{self.name} cannot be emptied from here — drop the "
                               "collection in the database", store=self.name)

    async def check(self, dimension: int = 8) -> dict[str, Any]:
        """Prove the store works: create, upsert, find, filter, delete.

        Writes three records under ids beginning `__check__` and removes them.
        Never raises: each step's outcome is in the answer.
        """
        steps: list[dict[str, Any]] = []
        prefix = f"__check__{uuid.uuid4().hex[:8]}"
        rows = [VectorRecord(f"{prefix}-{n}", normalise([1.0 if i == n else 0.05
                                                         for i in range(dimension)]),
                             f"check {n}", {"doc": prefix, "n": n, "scope": k})
                for n, k in enumerate(["a", "b", "b"])]
        # A store that filters only on fields it was told about is asked about
        # the ones it always knows; a number is tried where it can be.
        declared = getattr(self, "filterable", None)
        numbers = declared is None or "n" in declared

        async def step(name: str, action: Any) -> bool:
            started = time.perf_counter()
            try:
                detail = await action()
                steps.append({"step": name, "ok": True, "detail": detail or "",
                              "seconds": time.perf_counter() - started})
                return True
            except Exception as exc:
                steps.append({"step": name, "ok": False,
                              "detail": f"{type(exc).__name__}: {exc}"[:300],
                              "seconds": time.perf_counter() - started})
                return False

        async def find(want: str, **kw: Any) -> list[VectorHit]:
            # A store may take a moment to show what it was just given.
            for _ in range(40):
                hits = await self.query(rows[0].vector, k=3, **kw)
                if hits and hits[0].id == want:
                    return hits
                await asyncio.sleep(0.25)
            raise VectorStoreError(f"{want} was not the nearest record: "
                                   f"{[(h.id, round(h.score, 3)) for h in hits]}")

        async def nearest() -> str:
            hits = await find(rows[0].id, filter={"doc": prefix})
            if not 0.98 <= hits[0].score <= 1.001:
                raise VectorStoreError(f"an identical vector scored {hits[0].score:.4f}, "
                                       "not 1.0 — the score is not cosine similarity")
            if hits[0].text != "check 0" or hits[0].metadata.get("n") != 0:
                raise VectorStoreError(f"text or metadata did not come back: {hits[0]}")
            return f"score {hits[0].score:.4f}"

        async def filtered() -> str:
            hits = await self.query(rows[0].vector, k=3,
                                    filter={"doc": prefix, "scope": "b"})
            if sorted(h.id for h in hits) != [rows[1].id, rows[2].id]:
                raise VectorStoreError(f"the filter returned {[h.id for h in hits]}")
            either = await self.query(rows[0].vector, k=3, filter={
                "doc": prefix, "scope": {"$in": ["b", "c"]}})
            if sorted(h.id for h in either) != [rows[1].id, rows[2].id]:
                raise VectorStoreError(f"$in returned {[h.id for h in either]}")
            if not numbers:
                return "equals, $in"
            ranged = await self.query(rows[0].vector, k=3, filter={
                "doc": prefix, "n": {"$gte": 1}})
            if sorted(h.id for h in ranged) != [rows[1].id, rows[2].id]:
                raise VectorStoreError(f"$gte returned {[h.id for h in ranged]}")
            return "equals, $in, $gte"

        async def replaced() -> str:
            await self.upsert([VectorRecord(rows[1].id, rows[1].vector, "check 1 again",
                                            {"doc": prefix, "n": 1, "scope": "c"})])
            for _ in range(40):
                hits = await self.query(rows[1].vector, k=3, filter={"doc": prefix})
                if (len(hits) == 3 and hits[0].id == rows[1].id
                        and hits[0].text == "check 1 again"):
                    return "same id, new content, still three records"
                await asyncio.sleep(0.25)
            raise VectorStoreError(f"upserting an existing id left {hits}")

        async def gone(what: str, **kw: Any) -> str:
            await self.delete(**kw)
            for _ in range(40):
                left = await self.query(rows[0].vector, k=3, filter={"doc": prefix})
                if what == "id" and rows[0].id not in [h.id for h in left]:
                    return f"{len(left)} left"
                if what == "filter" and not left:
                    return "none left"
                await asyncio.sleep(0.25)
            raise VectorStoreError(f"still there after deleting by {what}: "
                                   f"{[h.id for h in left]}")

        ok = await step("ensure", lambda: self.ensure(dimension))
        ok = ok and await step("upsert", lambda: self.upsert(rows))
        ok = ok and await step("query", nearest)
        ok = ok and await step("filter", filtered)
        ok = ok and await step("replace", replaced)
        ok = ok and await step("delete by id", lambda: gone("id", ids=[rows[0].id]))
        if ok:
            ok = await step("delete by filter",
                            lambda: gone("filter", filter={"doc": prefix}))
        else:
            try:                                # leave nothing behind either way
                await self.delete([r.id for r in rows])
            except Exception:  # noqa: S110 - the failure is already in the report
                pass
        return {"store": self.name, "ok": ok, "steps": steps}

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} {self.name}>"


class HTTPVectorStore(VectorStore):
    """A vector database spoken to over HTTP: the client, the retries, the errors.

    A call that may succeed in a moment — 429, 5xx, a timeout, a dropped
    connection — is tried again with back-off. Anything else is a
    `VectorStoreError` that says what the database said, with keys removed.
    """

    def __init__(self, *, timeout: float = 30.0, retries: int = 3,
                 retry: RetryPolicy | None = None, headers: dict[str, str] | None = None,
                 transport: Any = None, client: Any = None) -> None:
        self.timeout = float(timeout)
        self.retry = retry or RetryPolicy(max_retries=max(0, int(retries)),
                                          initial_delay=0.4, max_delay=8.0,
                                          max_retry_after=20.0)
        self.extra_headers = dict(headers or {})
        self._transport = transport
        self._given = client
        self._client: Any = None
        self._loop: Any = None
        self._ready: set[int] = set()

    def _secrets(self) -> list[str]:
        return []

    async def _headers(self) -> dict[str, str]:
        """The headers every request carries — the key, mostly."""
        return {}

    def _sign(self, method: str, url: str, body: bytes,
              headers: dict[str, str]) -> dict[str, str]:
        """A hook for stores whose requests are signed rather than keyed."""
        return headers

    def _http(self) -> Any:
        if self._given is not None:
            return self._given
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self.timeout, transport=self._transport,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16))
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None and self._loop is asyncio.get_running_loop():
            await client.aclose()

    def _redact(self, text: str) -> str:
        for secret in self._secrets():
            if secret and len(secret) >= 4:
                text = text.replace(secret, "***")
        return text

    async def _request(self, method: str, url: str, *, json_body: Any = None,
                       content: bytes | None = None, params: dict[str, Any] | None = None,
                       headers: dict[str, str] | None = None,
                       ok: tuple[int, ...] = (), raw: bool = False) -> Any:
        """One request, retried when that can help. Returns the JSON body.

        `ok` lists statuses besides 2xx that are an answer rather than a
        failure — a 404 when asking whether something exists, say; for those
        the status code is returned instead.
        """
        body = content if content is not None else (
            json.dumps(json_body).encode() if json_body is not None else b"")
        base = {**self.extra_headers, **(await self._headers()), **(headers or {})}
        if json_body is not None:
            base.setdefault("content-type", "application/json")
        attempt = 0
        started = time.monotonic()
        while True:
            problem: VectorStoreError
            try:
                import httpx

                target = str(httpx.URL(url, params=params)) if params else url
                response = await self._http().request(
                    method, target, content=body or None,
                    headers=self._sign(method, target, body, dict(base)))
                status = response.status_code
                if 200 <= status < 300:
                    if raw:
                        return response
                    if not response.content:
                        return {}
                    try:
                        return response.json()
                    except ValueError:
                        return {"text": response.text}
                if status in ok:
                    return status
                detail = self._redact(" ".join(response.text[:800].split()))[:400]
                again = status in (408, 425, 429) or status >= 500
                problem = VectorStoreError(
                    f"{self.name}: {method} {self._redact(url.split('?')[0])} → "
                    f"{status}: {detail}", store=self.name, status=status,
                    retryable=again)
                wait = retry_after(response.headers, rate_limited=status == 429)
            except VectorStoreError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                kind = type(exc).__name__
                module = type(exc).__module__ or ""
                if not module.startswith(("httpx", "httpcore")) and not isinstance(
                        exc, (OSError, *_Timeout)):
                    raise
                # Only the kind: the message of a transport error can carry the URL.
                problem = VectorStoreError(
                    f"{self.name} could not be reached ({kind})", store=self.name,
                    retryable=True)
                wait = None
            if not problem.retryable or attempt >= self.retry.max_retries:
                raise problem
            delay = wait if wait is not None else self.retry.backoff(attempt)
            if delay > self.retry.max_retry_after or (
                    self.retry.max_elapsed is not None
                    and time.monotonic() - started + delay > self.retry.max_elapsed):
                raise problem
            await asyncio.sleep(delay)
            attempt += 1
