"""Memory behind your own HTTP API — no extra driver, httpx is already here.

Use it when memory has to live in a service you already run, or behind an
internal API that enforces its own access rules. The contract is small:

    POST   {base}/records            {record}
    GET    {base}/records            ?scope=&kind=&limit=&user_id=&...
    DELETE {base}/records            ?scope=&user_id=&...
    GET    {base}/docs/{key}         -> {"text": "..."}
    PUT    {base}/docs/{key}         {"text": "...", "name": ..., "namespace": ...}
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import quote

from ...errors import ProviderError
from ..base import MemoryRecord, MemoryStore
from ..trace import Trace

__all__ = ["HTTPMemory"]


class HTTPMemory(MemoryStore):
    """A REST backend. Bring your own service; this speaks to it."""

    driver_hint: ClassVar[str] = ""      # httpx is a runtime dependency already

    def __init__(self, base_url: str, *, headers: dict[str, str] | None = None,
                 timeout: float = 30.0, client: Any = None, **options: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._options = options

    @property
    def http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout,
                                             headers=self.headers, **self._options)
        return self._client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.http.request(method, f"{self.base_url}{path}",
                                           **kwargs)
        if response.status_code == 404:
            return None
        if response.status_code >= 300:
            raise ProviderError(
                f"memory API {method} {path} returned {response.status_code}: "
                f"{response.text[:300]}", provider="http-memory",
                status=response.status_code)
        return response.json() if response.content else None

    @staticmethod
    def _params(scope: str | None, kind: str | None, limit: int | None,
                trace: Trace | None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if scope:
            params["scope"] = scope
        if kind:
            params["kind"] = kind
        if limit:
            params["limit"] = int(limit)
        if trace is not None:
            params.update(trace.filters())
        return params

    async def append(self, record: MemoryRecord) -> MemoryRecord:
        await self._request("POST", "/records",
                            json=record.model_dump(mode="json"))
        return record

    async def all(self, scope: str | None = None, *, limit: int | None = None,
                  kind: str | None = None,
                  trace: Trace | None = None) -> list[MemoryRecord]:
        rows = await self._request(
            "GET", "/records", params=self._params(scope, kind, limit, trace))
        if not rows:
            return []
        if isinstance(rows, dict):
            rows = rows.get("records", [])
        out = []
        for row in rows:
            try:
                out.append(MemoryRecord(**row))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda r: r.ts)
        return out

    async def clear(self, scope: str | None = None, *,
                    trace: Trace | None = None) -> None:
        await self._request("DELETE", "/records",
                            params=self._params(scope, None, None, trace))

    async def read_doc(self, name: str, *, trace: Trace | None = None) -> str:
        key = quote(self.doc_key(name, trace), safe="")
        found = await self._request("GET", f"/docs/{key}")
        if not found:
            return ""
        return found.get("text", "") if isinstance(found, dict) else str(found)

    async def write_doc(self, name: str, text: str, *,
                        trace: Trace | None = None) -> None:
        key = quote(self.doc_key(name, trace), safe="")
        await self._request("PUT", f"/docs/{key}", json={
            "name": name, "text": text,
            "namespace": trace.slug if trace is not None else "_shared"})

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
