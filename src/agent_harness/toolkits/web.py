"""Web fetch, with a domain policy and a size cap. Native tool, not a connector."""

from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Any

import httpx

from ..errors import ToolError
from ..tools import Tool, tool

__all__ = ["http_fetch", "make_fetch_tool"]

_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_MARKUP = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")

_PRIVATE = ("localhost", "127.", "0.0.0.0", "10.", "192.168.", "169.254.", "[::1]")


def _to_text(body: str, content_type: str) -> str:
    if "html" not in content_type:
        return body
    stripped = _MARKUP.sub(" ", _TAGS.sub(" ", body))
    return _BLANKS.sub("\n\n", stripped.replace("&nbsp;", " ")).strip()


def make_fetch_tool(
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_chars: int = 40_000,
    timeout: float = 30.0,
    allow_private: bool = False,
    name: str = "web_fetch",
) -> Tool:
    """Build a fetch tool with your own domain policy."""

    @tool(name=name, tags=["builtin", "web"], cacheable=True)
    async def web_fetch(url: str, as_text: bool = True) -> str:
        """Fetch a URL and return its content.

        Args:
            url: the absolute http(s) URL to fetch.
            as_text: strip HTML markup and return readable text.
        """
        if not url.startswith(("http://", "https://")):
            raise ToolError("only http and https URLs can be fetched", tool=name)
        host = url.split("//", 1)[1].split("/", 1)[0].split(":")[0].lower()
        if not allow_private and host.startswith(_PRIVATE):
            raise ToolError("refusing to fetch a private or loopback address", tool=name)
        if blocked_domains and any(fnmatch(host, p) for p in blocked_domains):
            raise ToolError(f"{host} is on the blocked list", tool=name)
        if allowed_domains and not any(fnmatch(host, p) for p in allowed_domains):
            raise ToolError(f"{host} is not on the allowed list", tool=name)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers={"user-agent": "agent-harness/0.1"})
        if response.status_code >= 400:
            raise ToolError(f"{url} returned {response.status_code}", tool=name)
        content_type = response.headers.get("content-type", "")
        body = response.text
        text = _to_text(body, content_type) if as_text else body
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} characters]"
        return text

    return web_fetch


http_fetch: Tool = make_fetch_tool()


def make_http_tool(*, base_url: str = "", headers: dict[str, str] | None = None,
                   name: str = "http_request", timeout: float = 30.0) -> Tool:
    """A general HTTP tool for an internal API the agent is allowed to call."""

    @tool(name=name, tags=["builtin", "web", "api"], permission="ask")
    async def http_request(path: str, method: str = "GET", body: dict[str, Any] | None = None,
                           query: dict[str, Any] | None = None) -> Any:
        """Call an HTTP API.

        Args:
            path: path appended to the configured base URL, or a full URL.
            method: GET, POST, PUT, PATCH or DELETE.
            body: JSON body for write methods.
            query: query-string parameters.
        """
        url = path if path.startswith("http") else f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method.upper(), url, json=body, params=query,
                                            headers=headers or {})
        if response.status_code >= 400:
            raise ToolError(f"{method} {url} → {response.status_code}: "
                            f"{response.text[:500]}", tool=name)
        try:
            return response.json()
        except ValueError:
            return response.text

    return http_request
