"""Web search, as a tool: a query in, a ranked list of pages out.

    from agent_harness import Agent
    from agent_harness.toolkits import http_fetch, web_search

    agent = Agent("analyst", mode="research", tools=[web_search, http_fetch])

`web_search` uses whichever search engine has a key in the environment —
Tavily, Brave, Exa, Serper, Google, or a SearXNG instance of your own — and
DuckDuckGo, which needs no key, when none does. Name one, or several to fall
back through, when the choice should be yours:

    search = WebSearch(["brave", "duckduckgo"], allowed_domains=["*.gov"])
    await search.search("heat pump subsidy 2026", recency="year")   # call it yourself
    agent = Agent("analyst", tools=[search.as_tool()])

Every engine is spoken to over plain HTTP, and every one is answered the same
way: title, URL, snippet, and the date when the engine knows it.

A search is something that fails — a key runs out, an engine throttles, a page
layout changes — so the failure is handled here rather than handed to the
model. A call that may succeed later is retried with back-off, honouring
`Retry-After`; an engine that cannot answer is passed over for the next one; an
engine that keeps failing is left alone for a minute; the whole search has a
deadline. Results are de-duplicated, stripped of markup and tracking
parameters, held to the domain policy, and kept for a few minutes so the same
question is not asked twice. When nothing works the model is told which engine
failed and why — never with a key in the message.
"""

from __future__ import annotations

import asyncio
import html
import inspect
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatch
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from ..errors import ConfigurationError, ToolError
from ..llm_providers.resilience import CircuitBreaker, RetryPolicy, retry_after
from ..tools import Tool, tool

__all__ = [
    "WebSearch",
    "SearchEngine",
    "SearchQuery",
    "SearchResult",
    "SearchFailure",
    "make_search_tool",
    "search_engines",
    "web_search",
]

RECENCY = ("day", "week", "month", "year")
SAFE = ("off", "moderate", "strict")
#: Set to an engine name, or several separated by commas, to choose the engines
#: of a search that was not given any.
ENGINE_ENV = "AGENT_HARNESS_SEARCH"

_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}
_MAX_QUERY_CHARS = 400
_MAX_QUERY_WORDS = 50
_MAX_SITES = 8          # more `site:` operators than this and engines ignore the query
_TITLE_CHARS = 200

_MARKUP = re.compile(r"<[^>]+>")
#: Emphasis an engine puts around the matched words; it leaves no gap behind.
_INLINE = re.compile(r"</?(?:b|strong|em|i|mark|span|u)\b[^>]*>", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_GLOB = re.compile(r"[*?\[\]]")
#: Query parameters that say where a click came from, and nothing about the page.
_TRACKING = re.compile(
    r"^(utm_\w+|fbclid|gclid|gclsrc|dclid|msclkid|yclid|mc_cid|mc_eid|igshid|"
    r"_hsenc|_hsmi|oly_anon_id|oly_enc_id|srsltid|ref_src|ref_url)$", re.IGNORECASE)


# ----------------------------------------------------------------------
# what goes in, what comes out
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SearchQuery:
    """One search, as every engine is asked it."""

    text: str
    limit: int = 5
    #: "day", "week", "month" or "year"; empty for any time.
    recency: str = ""
    #: Only these domains. Plain names or globs.
    domains: tuple[str, ...] = ()
    #: Never these.
    exclude: tuple[str, ...] = ()
    region: str = ""
    language: str = ""
    safe: str = "moderate"
    snippet_chars: int = 500

    @property
    def include_domains(self) -> list[str]:
        """The wanted domains an engine can be told about — globs it cannot."""
        return [d for d in self.domains if not _GLOB.search(d)]

    @property
    def exclude_domains(self) -> list[str]:
        return [d for d in self.exclude if not _GLOB.search(d)]

    def with_sites(self) -> str:
        """The text with `site:` operators, for an engine with no domain filter."""
        text = self.text
        wanted = self.include_domains[:_MAX_SITES]
        # Only when every wanted domain fits: half a filter loses results.
        if wanted and len(wanted) == len(self.domains):
            sites = " OR ".join(f"site:{d}" for d in wanted)
            text = f"{text} {sites}" if len(wanted) == 1 else f"{text} ({sites})"
        for domain in self.exclude_domains[:_MAX_SITES]:
            text = f"{text} -site:{domain}"
        return text


@dataclass
class SearchResult:
    """One page a search found."""

    title: str
    url: str
    snippet: str = ""
    #: When the page was published, as the engine reported it. Often empty.
    published: str = ""
    #: The engine that found it.
    engine: str = ""
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """What the model reads: the fields that have something in them."""
        out: dict[str, Any] = {"title": self.title, "url": self.url}
        if self.snippet:
            out["snippet"] = self.snippet
        if self.published:
            out["published"] = self.published
        return out


class SearchFailure(Exception):
    """One engine could not answer. Raised by engines, read by `WebSearch`.

    ``retryable``    the same request may succeed in a moment (429, 5xx, timeout)
    ``retry_after``  seconds the engine asked us to wait, when it said
    ``unhealthy``    counts towards leaving the engine alone for a while; a
                     query the engine refused says nothing about its health
    """

    def __init__(self, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, unhealthy: bool = True,
                 status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.unhealthy = unhealthy
        self.status = status


# ----------------------------------------------------------------------
# tidying what an engine returned
# ----------------------------------------------------------------------
def _text(value: Any, limit: int) -> str:
    """Markup, entities and stray whitespace out; cut at a word when too long."""
    if not isinstance(value, str):
        if isinstance(value, (list, tuple)):
            value = " … ".join(str(v) for v in value if v)
        elif value is None:
            return ""
        else:
            value = str(value)
    text = html.unescape(_MARKUP.sub(" ", _INLINE.sub("", value)))
    text = _SPACE.sub(" ", _CONTROL.sub(" ", text)).strip()
    if len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0] if " " in text[limit // 2:limit] else text[:limit]
        text = cut.rstrip(" ,;:.") + "…"
    return text


def _published(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    value = value.strip()
    match = _ISO_DATE.match(value)
    return match.group(0) if match else _text(value, 40)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _domain(value: str) -> str:
    """`https://Docs.Example.com/page` → `docs.example.com`. Globs are kept."""
    value = str(value or "").strip().lower()
    if not value:
        return ""
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split("?", 1)[0].strip()
    if value.startswith("site:"):
        value = value[5:]
    if not value.startswith("["):
        value = value.rsplit(":", 1)[0] if value.rsplit(":", 1)[-1].isdigit() else value
    return value.lstrip(".").rstrip(".")


def _matches(host: str, pattern: str) -> bool:
    """A plain domain covers its subdomains; a glob is matched as written."""
    if not host or not pattern:
        return False
    if _GLOB.search(pattern):
        return fnmatch(host, pattern)
    if pattern.startswith("www."):
        pattern = pattern[4:]
    return host == pattern or host.endswith("." + pattern)


def _canonical(url: str) -> str:
    """The URL without tracking parameters or a text-fragment anchor."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not _TRACKING.match(k)]
    fragment = "" if parts.fragment.startswith(":~:") else parts.fragment
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path,
                       urlencode(query), fragment))


def _identity(url: str) -> str:
    """Two URLs with the same identity are the same page."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return f"{host}{parts.path.rstrip('/')}?{query}"


def _redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return text


# ----------------------------------------------------------------------
# engines
# ----------------------------------------------------------------------
class SearchEngine:
    """One search service, spoken to over HTTP.

    Subclass it for an engine that is not here: say how the request is built
    (`request`) and how the answer is read (`parse`), and `WebSearch` brings the
    retries, the fallback, the domain policy and the cache.

        class Internal(SearchEngine):
            name, url, env = "internal", "https://search.corp.example", ("CORP_KEY",)

            def request(self, query):
                return {"method": "GET", "url": f"{self.base_url}/q",
                        "params": {"q": query.text, "n": query.limit},
                        "headers": {"authorization": f"Bearer {self.api_key}"}}

            def parse(self, data, query):
                return [{"title": h["name"], "url": h["link"], "snippet": h["text"]}
                        for h in data["hits"]]
    """

    name = ""
    url = ""
    #: Environment variables read for the key, the first that is set winning.
    env: tuple[str, ...] = ()
    needs_key = True
    #: The most results one request may ask for.
    max_results = 20

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self.headers = dict(headers or {})

    # The environment is read when the engine is used, not when it is built, so
    # a tool created at import time sees a key that is set afterwards.
    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return next((os.environ[v] for v in self.env if os.environ.get(v)), "")

    @property
    def base_url(self) -> str:
        return (self._base_url or self.url).rstrip("/")

    def ready(self) -> str:
        """Empty when the engine can be called; otherwise what is missing."""
        if self.needs_key and not self.api_key:
            return f"no API key — set {' or '.join(self.env) or 'api_key='}"
        if not self.base_url:
            return "no URL to call"
        return ""

    def secrets(self) -> list[str]:
        return [self.api_key]

    def needs(self) -> list[str]:
        """The environment variables that set this engine up."""
        return [" or ".join(self.env)] if self.needs_key and self.env else []

    def request(self, query: SearchQuery) -> dict[str, Any]:
        """The arguments of the HTTP request: method, url, params, json, headers."""
        raise NotImplementedError

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        """The engine's JSON, as a list of results or of mappings."""
        raise NotImplementedError

    def check(self, response: Any) -> None:
        """Raise `SearchFailure` for an answer that is not a result."""
        status = response.status_code
        if 200 <= status < 300:
            return
        detail = _text(response.text[:300], 200)
        if status in (401, 403):
            raise SearchFailure(
                "the API key was refused" if self.needs_key else
                f"the request was refused ({status})", status=status)
        if status == 402 or (status == 429 and re.search(
                r"quota|credit|billing|plan limit|usage limit", detail, re.IGNORECASE)):
            raise SearchFailure("out of quota", status=status)
        if status == 429:
            raise SearchFailure("rate limited", retryable=True, status=status,
                                retry_after=retry_after(response.headers))
        if status == 408 or status >= 500:
            raise SearchFailure(f"the engine answered {status}", retryable=True,
                                status=status, retry_after=retry_after(
                                    response.headers, rate_limited=False))
        if 300 <= status < 400:
            raise SearchFailure(f"redirected ({status}) — check the engine's URL",
                                status=status)
        raise SearchFailure(f"refused the query ({status}): {detail}" if detail else
                            f"refused the query ({status})", unhealthy=False, status=status)

    def read(self, response: Any, query: SearchQuery) -> list[Any]:
        try:
            data = response.json()
        except ValueError:
            raise SearchFailure("answered with something that is not JSON") from None
        try:
            return list(self.parse(data, query) or [])
        except SearchFailure:
            raise
        except Exception as exc:
            raise SearchFailure(
                f"answered in a shape that could not be read ({type(exc).__name__})"
            ) from None

    async def search(self, query: SearchQuery, client: Any) -> list[Any]:
        request = self.request(query)
        headers = {**self.headers, **(request.pop("headers", None) or {})}
        response = await client.request(headers=headers, **request)
        self.check(response)
        return self.read(response, query)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<SearchEngine {self.name}>"


def _items(data: Any, *path: str) -> list[dict[str, Any]]:
    """`data[path[0]][path[1]]…` as a list of mappings, whatever was there."""
    for key in path:
        data = data.get(key) if isinstance(data, dict) else None
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


class Tavily(SearchEngine):
    name = "tavily"
    url = "https://api.tavily.com"
    env = ("TAVILY_API_KEY",)

    def request(self, query: SearchQuery) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query.text, "max_results": query.limit,
                                "search_depth": "basic"}
        if query.recency:
            body["time_range"] = query.recency
        if query.include_domains:
            body["include_domains"] = query.include_domains
        if query.exclude_domains:
            body["exclude_domains"] = query.exclude_domains
        return {"method": "POST", "url": f"{self.base_url}/search", "json": body,
                "headers": {"Authorization": f"Bearer {self.api_key}"}}

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        return [{"title": r.get("title"), "url": r.get("url"),
                 "snippet": r.get("content"), "published": r.get("published_date"),
                 "score": r.get("score")} for r in _items(data, "results")]


class Brave(SearchEngine):
    name = "brave"
    url = "https://api.search.brave.com"
    env = ("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")

    def request(self, query: SearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {"q": query.with_sites(), "count": query.limit,
                                  "safesearch": query.safe, "text_decorations": "false"}
        if query.recency:
            params["freshness"] = "p" + query.recency[0]
        if query.region:
            params["country"] = query.region.upper()
        if query.language:
            params["search_lang"] = query.language
        return {"method": "GET", "url": f"{self.base_url}/res/v1/web/search",
                "params": params,
                "headers": {"X-Subscription-Token": self.api_key,
                            "Accept": "application/json"}}

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        return [{"title": r.get("title"), "url": r.get("url"),
                 "snippet": r.get("description") or r.get("extra_snippets"),
                 "published": r.get("page_age") or r.get("age")}
                for r in _items(data, "web", "results")]


class Exa(SearchEngine):
    name = "exa"
    url = "https://api.exa.ai"
    env = ("EXA_API_KEY",)
    max_results = 25

    def request(self, query: SearchQuery) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query.text, "numResults": query.limit, "type": "auto",
            "contents": {"text": {"maxCharacters": max(200, query.snippet_chars)}}}
        if query.recency:
            since = datetime.now(timezone.utc) - timedelta(days=_DAYS[query.recency])
            body["startPublishedDate"] = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if query.include_domains:
            body["includeDomains"] = query.include_domains
        if query.exclude_domains:
            body["excludeDomains"] = query.exclude_domains
        return {"method": "POST", "url": f"{self.base_url}/search", "json": body,
                "headers": {"x-api-key": self.api_key}}

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        return [{"title": r.get("title"), "url": r.get("url"),
                 "snippet": r.get("summary") or r.get("highlights") or r.get("text"),
                 "published": r.get("publishedDate"), "score": r.get("score")}
                for r in _items(data, "results")]


class Serper(SearchEngine):
    """Google results, through serper.dev."""

    name = "serper"
    url = "https://google.serper.dev"
    env = ("SERPER_API_KEY",)

    def request(self, query: SearchQuery) -> dict[str, Any]:
        body: dict[str, Any] = {"q": query.with_sites(), "num": query.limit}
        if query.recency:
            body["tbs"] = "qdr:" + query.recency[0]
        if query.region:
            body["gl"] = query.region.lower()
        if query.language:
            body["hl"] = query.language
        return {"method": "POST", "url": f"{self.base_url}/search", "json": body,
                "headers": {"X-API-KEY": self.api_key}}

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        return [{"title": r.get("title"), "url": r.get("link"),
                 "snippet": r.get("snippet"), "published": r.get("date")}
                for r in _items(data, "organic")]


class Google(SearchEngine):
    """Google's Programmable Search Engine: a key, and the id of an engine."""

    name = "google"
    url = "https://www.googleapis.com"
    env = ("GOOGLE_SEARCH_API_KEY", "GOOGLE_API_KEY")
    cx_env = ("GOOGLE_CSE_ID", "GOOGLE_SEARCH_ENGINE_ID")
    max_results = 10

    def __init__(self, api_key: str | None = None, *, cx: str | None = None,
                 base_url: str | None = None, headers: dict[str, str] | None = None) -> None:
        super().__init__(api_key, base_url=base_url, headers=headers)
        self._cx = cx

    @property
    def cx(self) -> str:
        return self._cx or next(
            (os.environ[v] for v in self.cx_env if os.environ.get(v)), "")

    def ready(self) -> str:
        if not self.cx:
            return f"no search engine id — set {self.cx_env[0]}"
        return super().ready()

    def needs(self) -> list[str]:
        return [self.env[0], self.cx_env[0]]

    def request(self, query: SearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {"key": self.api_key, "cx": self.cx,
                                  "q": query.with_sites(), "num": query.limit,
                                  "safe": "off" if query.safe == "off" else "active"}
        if query.recency:
            params["dateRestrict"] = query.recency[0] + "1"
        if query.region:
            params["gl"] = query.region.lower()
        if query.language:
            params["lr"] = f"lang_{query.language}"
        return {"method": "GET", "url": f"{self.base_url}/customsearch/v1",
                "params": params}

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        out = []
        for r in _items(data, "items"):
            tags = _items(r, "pagemap", "metatags")
            out.append({"title": r.get("title"), "url": r.get("link"),
                        "snippet": r.get("snippet"),
                        "published": (tags[0].get("article:published_time")
                                      if tags else "")})
        return out


class SearXNG(SearchEngine):
    """A SearXNG instance of your own, with the JSON format switched on."""

    name = "searxng"
    env = ("SEARXNG_API_KEY",)
    url_env = "SEARXNG_URL"
    needs_key = False

    @property
    def base_url(self) -> str:
        return (self._base_url or os.environ.get(self.url_env, "")).rstrip("/")

    def ready(self) -> str:
        return "" if self.base_url else f"no instance to call — set {self.url_env}"

    def needs(self) -> list[str]:
        return [self.url_env]

    def request(self, query: SearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {"q": query.with_sites(), "format": "json",
                                  "safesearch": SAFE.index(query.safe)}
        if query.recency:
            params["time_range"] = query.recency
        if query.language:
            params["language"] = query.language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return {"method": "GET", "url": f"{self.base_url}/search", "params": params,
                "headers": headers}

    def check(self, response: Any) -> None:
        if response.status_code == 403:
            raise SearchFailure("the instance refused the request (403) — the JSON "
                                "format must be enabled in its settings", status=403)
        super().check(response)

    def parse(self, data: Any, query: SearchQuery) -> list[Any]:
        return [{"title": r.get("title"), "url": r.get("url"),
                 "snippet": r.get("content"), "published": r.get("publishedDate"),
                 "score": r.get("score")} for r in _items(data, "results")]


_DDG_TITLE = re.compile(
    r"<a\b(?=[^>]*\bclass=\"[^\"]*\bresult__a\b)([^>]*)>(.*?)</a>", re.DOTALL)
_DDG_SNIPPET = re.compile(
    r"<(a|div|td)\b(?=[^>]*\bclass=\"[^\"]*\bresult__snippet\b)[^>]*>(.*?)</\1>", re.DOTALL)
_DDG_DATE = re.compile(r"<span>(?:\s|&nbsp;)*(\d{4}-\d{2}-\d{2})T[\d:.]+\s*</span>")
_HREF = re.compile(r"\bhref=\"([^\"]*)\"")
_DDG_BLOCKED = ("anomaly-modal", "challenge-form", "bots use DuckDuckGo too")


class DuckDuckGo(SearchEngine):
    """DuckDuckGo's HTML results page. No key — and no promise of availability.

    This is a page meant for browsers, read here because it is the one search
    that works with nothing set up. It throttles heavy use and its layout can
    change; both are reported as failures, so a fallback engine takes over. For
    anything that matters, give the search a key.
    """

    name = "duckduckgo"
    url = "https://html.duckduckgo.com"
    needs_key = False
    max_results = 30

    def request(self, query: SearchQuery) -> dict[str, Any]:
        data = {"q": query.with_sites(),
                "kp": {"strict": "1", "moderate": "-1", "off": "-2"}[query.safe]}
        if query.recency:
            data["df"] = query.recency[0]
        if query.region:
            data["kl"] = f"{query.region.lower()}-{query.language or 'en'}"
        return {"method": "POST", "url": f"{self.base_url}/html/", "data": data}

    def read(self, response: Any, query: SearchQuery) -> list[Any]:
        page = response.text
        if response.status_code == 202 or any(mark in page for mark in _DDG_BLOCKED):
            raise SearchFailure("DuckDuckGo is throttling this address — give the "
                                "search an API key, or try again later")
        titles = list(_DDG_TITLE.finditer(page))
        out: list[dict[str, Any]] = []
        for index, match in enumerate(titles):
            href = _HREF.search(match.group(1))
            if not href:
                continue
            end = titles[index + 1].start() if index + 1 < len(titles) else len(page)
            block = page[match.end():end]
            snippet = _DDG_SNIPPET.search(block)
            date = _DDG_DATE.search(block)
            out.append({"title": match.group(2), "url": self._target(href.group(1)),
                        "snippet": snippet.group(2) if snippet else "",
                        "published": date.group(1) if date else ""})
        if not out and "no-results" not in page and "No results" not in page:
            raise SearchFailure("the results page could not be read — its layout "
                                "may have changed")
        return out

    @staticmethod
    def _target(href: str) -> str:
        """The page behind DuckDuckGo's redirect link; nothing for an advert."""
        href = html.unescape(href)
        if href.startswith("//"):
            href = "https:" + href
        parts = urlsplit(href)
        if (parts.hostname or "").endswith("duckduckgo.com"):
            if parts.path.startswith("/y.js"):
                return ""
            target = dict(parse_qsl(parts.query)).get("uddg", "")
            return unquote(target) if "%3A" in target[:12].upper() else target
        return href


class _Function(SearchEngine):
    """A function of your own as an engine: `(query) -> results`, sync or async."""

    needs_key = False

    def __init__(self, fn: Callable[..., Any], name: str = "") -> None:
        super().__init__()
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "") or "custom"

    def ready(self) -> str:
        return ""

    async def search(self, query: SearchQuery, client: Any) -> list[Any]:
        found = self.fn(query)
        if inspect.isawaitable(found):
            found = await found
        return list(found or [])


#: In the order they are preferred when the choice is left to the environment.
ENGINES: dict[str, type[SearchEngine]] = {
    "tavily": Tavily, "brave": Brave, "exa": Exa, "serper": Serper,
    "google": Google, "searxng": SearXNG, "duckduckgo": DuckDuckGo,
}
_ALIASES = {"ddg": "duckduckgo", "duck": "duckduckgo", "searx": "searxng",
            "google_cse": "google", "google-cse": "google", "serper.dev": "serper"}


def search_engines() -> list[dict[str, Any]]:
    """Every engine that ships, and whether this environment can call it."""
    out = []
    for name, kind in ENGINES.items():
        engine = kind()
        missing = engine.ready()
        out.append({"name": name, "ready": not missing, "missing": missing,
                    "needs": engine.needs()})
    return out


def _engine(spec: Any, api_key: str | None = None) -> SearchEngine:
    if isinstance(spec, SearchEngine):
        return spec
    if isinstance(spec, type) and issubclass(spec, SearchEngine):
        return spec(api_key) if api_key else spec()
    if isinstance(spec, str):
        name = spec.strip().lower()
        name = _ALIASES.get(name, name)
        if name not in ENGINES:
            raise ConfigurationError(
                f"no search engine named {spec!r}; there are: {', '.join(ENGINES)}")
        return ENGINES[name](api_key) if api_key else ENGINES[name]()
    if callable(spec):
        return _Function(spec)
    raise ConfigurationError(
        "a search engine is a name, a SearchEngine, or a function — got "
        f"{type(spec).__name__}")


# ----------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------
class WebSearch:
    """A web search with retries, fallback engines, a domain policy and a cache.

        search = WebSearch()                              # from the environment
        search = WebSearch("brave", api_key="BSA…")
        search = WebSearch(["tavily", "duckduckgo"], allowed_domains=["*.gov"])

        hits = await search.search("heat pump subsidy", limit=5, recency="year")
        agent = Agent("analyst", tools=[search.as_tool()])

    ``engine``            a name, a `SearchEngine`, a function, or a list to fall
                          back through in order. None: `$AGENT_HARNESS_SEARCH`,
                          else every engine with a key set, then DuckDuckGo.
    ``limit`` / ``max_limit``  results when the model does not say, and the most
                          it may ask for
    ``allowed_domains`` / ``blocked_domains``  the domain policy. A plain name
                          covers its subdomains; globs work. The model can narrow
                          a search to some domains, never widen it past these.
    ``timeout``           seconds for one request
    ``deadline``          seconds for the whole search, retries and fallbacks included
    ``retries``           tries after the first, per engine (or pass `retry=`)
    ``cache_ttl``         seconds an answer is kept; 0 turns the cache off
    ``min_interval``      seconds between two requests to one engine
    ``fallback_on_empty`` ask the next engine when one finds nothing
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        api_key: str | None = None,
        limit: int = 5,
        max_limit: int = 10,
        allowed_domains: Iterable[str] | None = None,
        blocked_domains: Iterable[str] | None = None,
        region: str = "",
        language: str = "",
        safe_search: str = "moderate",
        snippet_chars: int = 500,
        timeout: float = 15.0,
        deadline: float = 45.0,
        retries: int = 2,
        retry: RetryPolicy | None = None,
        cache_ttl: float = 300.0,
        cache_size: int = 256,
        min_interval: float = 0.0,
        fallback_on_empty: bool = True,
        failure_threshold: int = 3,
        cooldown: float = 60.0,
        transport: Any = None,
        client: Any = None,
        name: str = "web_search",
    ) -> None:
        if safe_search not in SAFE:
            raise ConfigurationError(
                f"safe_search is one of {', '.join(SAFE)} — got {safe_search!r}")
        if isinstance(allowed_domains, str) or isinstance(blocked_domains, str):
            raise ConfigurationError("allowed_domains and blocked_domains are lists")
        specs = list(engine) if isinstance(engine, (list, tuple)) else (
            [] if engine is None else [engine])
        if api_key and len(specs) != 1:
            raise ConfigurationError(
                "api_key= goes with one named engine; for several, pass each as "
                "an engine of its own — Brave(api_key=...)")
        # Names are checked now, so a typo fails where it was written. Keys are
        # read from the environment when a search is made.
        self._engines: list[SearchEngine] | None = (
            [_engine(spec, api_key) for spec in specs] if specs else None)
        self.name = name
        self.max_limit = max(1, int(max_limit))
        self.limit = max(1, min(int(limit), self.max_limit))
        self.allowed = [d for d in map(_domain, allowed_domains or ()) if d]
        self.blocked = [d for d in map(_domain, blocked_domains or ()) if d]
        self.region = region.strip()
        self.language = language.strip()
        self.safe_search = safe_search
        self.snippet_chars = max(80, int(snippet_chars))
        self.timeout = float(timeout)
        self.deadline = float(deadline)
        self.retry = retry or RetryPolicy(max_retries=max(0, int(retries)),
                                          initial_delay=0.5, max_delay=8.0,
                                          max_retry_after=10.0)
        self.cache_ttl = float(cache_ttl)
        self.cache_size = max(1, int(cache_size))
        self.min_interval = float(min_interval)
        self.fallback_on_empty = fallback_on_empty
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._transport = transport
        self._client = client
        self._breakers: dict[str, CircuitBreaker] = {}
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, list[SearchResult]]] = (
            OrderedDict())
        self._inflight: dict[tuple[Any, ...], asyncio.Future[list[SearchResult]]] = {}
        self._next_call: dict[str, float] = {}
        #: The engine that answered the last search.
        self.last_engine = ""
        #: Per engine: how many searches it answered and how many it failed.
        self.stats: dict[str, dict[str, Any]] = {}

    # ---- engines -------------------------------------------------------
    def _resolve(self) -> list[SearchEngine]:
        if self._engines is None:
            named = [n.strip() for n in os.environ.get(ENGINE_ENV, "").split(",")
                     if n.strip()]
            if named:
                self._engines = [_engine(n) for n in named]
            else:
                found = [kind() for name, kind in ENGINES.items() if name != "duckduckgo"]
                self._engines = [e for e in found if not e.ready()] + [DuckDuckGo()]
        return self._engines

    @property
    def engines(self) -> list[str]:
        """The engines a search goes through, in order."""
        return [engine.name for engine in self._resolve()]

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(self._failure_threshold, self._cooldown)
        return self._breakers[name]

    def _count(self, name: str, outcome: str, error: str = "") -> None:
        entry = self.stats.setdefault(name, {"answered": 0, "failed": 0, "last_error": ""})
        entry[outcome] += 1
        if error:
            entry["last_error"] = error

    # ---- the question ---------------------------------------------------
    def _query(self, text: Any, limit: int | None, recency: str | None,
               domains: Iterable[str] | None) -> SearchQuery:
        if not isinstance(text, str):
            raise ToolError("the query must be text", tool=self.name)
        text = _SPACE.sub(" ", _CONTROL.sub(" ", text)).strip()
        if not text:
            raise ToolError("the query is empty — say what to search for", tool=self.name)
        if len(text) > _MAX_QUERY_CHARS:
            text = text[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        text = " ".join(text.split(" ")[:_MAX_QUERY_WORDS])

        try:
            count = int(limit) if limit else self.limit
        except (TypeError, ValueError):
            count = self.limit
        count = max(1, min(count, self.max_limit))

        when = str(recency or "").strip().lower()
        when = {"d": "day", "w": "week", "m": "month", "y": "year", "any": "",
                "all": "", "none": "", "today": "day", "24h": "day"}.get(when, when)
        if when and when not in RECENCY:
            raise ToolError(f"recency is one of {', '.join(RECENCY)}, or empty for any "
                            f"time — got {recency!r}", tool=self.name)

        if isinstance(domains, str):
            domains = [d for d in re.split(r"[,\s]+", domains) if d]
        wanted = list(dict.fromkeys(d for d in map(_domain, domains or ()) if d))
        for domain in wanted:
            if any(_matches(domain, b) for b in self.blocked):
                raise ToolError(f"{domain} is on the blocked list", tool=self.name)
            if self.allowed and not _GLOB.search(domain) and not any(
                    _matches(domain, a) for a in self.allowed):
                raise ToolError(
                    f"{domain} is not on the allowed list — this search covers "
                    f"{', '.join(self.allowed)}", tool=self.name)
        return SearchQuery(
            text=text, limit=count, recency=when,
            domains=tuple(wanted or self.allowed), exclude=tuple(self.blocked),
            region=self.region, language=self.language, safe=self.safe_search,
            snippet_chars=self.snippet_chars)

    def _permitted(self, url: str, query: SearchQuery) -> bool:
        host = _host(url)
        if not host:
            return False
        if any(_matches(host, b) for b in self.blocked):
            return False
        if self.allowed and not any(_matches(host, a) for a in self.allowed):
            return False
        return not query.domains or any(_matches(host, d) for d in query.domains)

    def _tidy(self, raw: Iterable[Any], query: SearchQuery, engine: str) -> list[SearchResult]:
        """An engine's results, cleaned, filtered and without repeats."""
        seen: set[str] = set()
        out: list[SearchResult] = []
        for item in raw:
            if isinstance(item, SearchResult):
                item = {"title": item.title, "url": item.url, "snippet": item.snippet,
                        "published": item.published, "score": item.score}
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link") or ""
            if not isinstance(url, str) or not url.strip().lower().startswith(
                    ("http://", "https://")):
                continue
            try:
                url = _canonical(url.strip())
                identity = _identity(url)
            except ValueError:
                continue
            if identity in seen or not self._permitted(url, query):
                continue
            seen.add(identity)
            score = item.get("score")
            out.append(SearchResult(
                title=_text(item.get("title"), _TITLE_CHARS) or _host(url),
                url=url,
                snippet=_text(item.get("snippet") or item.get("content")
                              or item.get("description"), query.snippet_chars),
                published=_published(item.get("published")),
                engine=engine,
                score=float(score) if isinstance(score, (int, float))
                and not isinstance(score, bool) else None))
            if len(out) >= query.limit:
                break
        return out

    # ---- asking ----------------------------------------------------------
    @asynccontextmanager
    async def _http(self):
        if self._client is not None:
            yield self._client
            return
        import httpx

        from .. import __version__

        # A client per search: nothing is left open, and nothing is tied to an
        # event loop that has since closed. Redirects are not followed — a key
        # in a header must not travel to wherever an engine points.
        async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False, transport=self._transport,
                headers={"user-agent": f"agent-harness/{__version__}"}) as client:
            yield client

    async def _pace(self, name: str) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        slot = max(now, self._next_call.get(name, 0.0))
        self._next_call[name] = slot + self.min_interval
        if slot > now:
            await asyncio.sleep(slot - now)

    async def _ask(self, engine: SearchEngine, query: SearchQuery, client: Any,
                   started: float) -> list[Any]:
        """One engine, tried until it answers, cannot, or the time is up."""
        missing = engine.ready()
        if missing:
            raise SearchFailure(missing)
        # Ask for a few more than wanted when some will be filtered away.
        want = query.limit + (5 if (query.domains or query.exclude) else 2)
        asked = replace(query, limit=max(1, min(want, engine.max_results)))
        attempt = 0
        while True:
            await self._pace(engine.name)
            left = self.deadline - (time.monotonic() - started)
            if left <= 0:
                raise SearchFailure("the search ran out of time", unhealthy=False)
            try:
                return await asyncio.wait_for(engine.search(asked, client),
                                              timeout=min(self.timeout, left))
            except SearchFailure as exc:
                failure = exc
            except asyncio.CancelledError:
                raise
            # On 3.10 asyncio.TimeoutError is not the builtin; catch both.
            except (asyncio.TimeoutError, TimeoutError):
                failure = SearchFailure(
                    f"no answer within {min(self.timeout, left):g}s", retryable=True)
            except Exception as exc:
                failure = self._classify(exc, engine)
            policy = self.retry
            if not failure.retryable or attempt >= policy.max_retries:
                raise failure
            wait = failure.retry_after
            if wait is None:
                wait = policy.backoff(attempt)
            elif wait > policy.max_retry_after:
                raise failure
            if time.monotonic() - started + wait >= self.deadline:
                raise failure
            await asyncio.sleep(wait)
            attempt += 1

    @staticmethod
    def _classify(exc: Exception, engine: SearchEngine) -> SearchFailure:
        kind = type(exc).__name__
        module = type(exc).__module__ or ""
        if module.startswith(("httpx", "httpcore")):
            if "Timeout" in kind:
                return SearchFailure("timed out", retryable=True)
            # The message of a transport error can carry the URL, and for one
            # engine the URL carries the key — so only the kind is reported.
            return SearchFailure(f"could not be reached ({kind})", retryable=True)
        return SearchFailure(_redact(f"{kind}: {exc}", engine.secrets())[:300])

    async def _run(self, query: SearchQuery) -> list[SearchResult]:
        engines = self._resolve()
        started = time.monotonic()
        problems: list[str] = []
        answered = False
        async with self._http() as client:
            for engine in engines:
                breaker = self._breaker(engine.name)
                if not breaker.allow():
                    problems.append(f"{engine.name}: left alone after repeated failures, "
                                    f"for another {breaker.retry_in():.0f}s")
                    continue
                try:
                    raw = await self._ask(engine, query, client, started)
                except asyncio.CancelledError:
                    # A cancelled probe must not hold the probe slot for ever.
                    if breaker.state == "half_open":
                        breaker.record_failure()
                    raise
                except SearchFailure as exc:
                    if exc.unhealthy:
                        breaker.record_failure()
                    else:
                        breaker.record_success()
                    message = _redact(str(exc), engine.secrets())
                    self._count(engine.name, "failed", message)
                    problems.append(f"{engine.name}: {message}")
                    continue
                breaker.record_success()
                self._count(engine.name, "answered")
                self.last_engine = engine.name
                answered = True
                results = self._tidy(raw, query, engine.name)
                if results or not self.fallback_on_empty:
                    return results
        if answered:
            return []
        raise ToolError("web search failed — " + "; ".join(problems or ["no engine to ask"]),
                        tool=self.name)

    async def search(self, query: str, *, limit: int | None = None,
                     recency: str | None = None,
                     domains: Iterable[str] | None = None) -> list[SearchResult]:
        """Search, and return what was found. Raises `ToolError` when no engine answers."""
        asked = self._query(query, limit, recency, domains)
        key = (asked.text.casefold(), asked.limit, asked.recency, asked.domains)
        now = time.monotonic()
        if self.cache_ttl > 0:
            kept = self._cache.get(key)
            if kept and kept[0] > now:
                self._cache.move_to_end(key)
                return [replace(r) for r in kept[1]]
            self._cache.pop(key, None)

        loop = asyncio.get_running_loop()
        # The model often asks the same thing twice in one step: ask once.
        pending = self._inflight.get(key)
        if pending is not None and not pending.done() and pending.get_loop() is loop:
            return [replace(r) for r in await asyncio.shield(pending)]

        future: asyncio.Future[list[SearchResult]] = loop.create_future()
        self._inflight[key] = future
        try:
            results = await self._run(asked)
        except BaseException as exc:
            failure = exc if isinstance(exc, Exception) else ToolError(
                "the search was cancelled", tool=self.name)
            future.set_exception(failure)
            future.exception()          # read it, so nobody is warned it was not
            raise
        else:
            future.set_result(results)
        finally:
            if self._inflight.get(key) is future:
                del self._inflight[key]
        if self.cache_ttl > 0:
            self._cache[key] = (time.monotonic() + self.cache_ttl, results)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return [replace(r) for r in results]

    def clear_cache(self) -> None:
        self._cache.clear()

    # ---- as a tool --------------------------------------------------------
    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """The search as a tool an agent can call."""
        search = self
        if name:
            self.name = name
        default_limit, most = self.limit, self.max_limit

        @tool(name=self.name, description=description, tags=["builtin", "web", "search"])
        async def web_search(query: str, limit: int = default_limit, recency: str = "",
                             domains: list[str] | None = None) -> Any:
            """Search the web. Returns the best-matching pages: title, URL and a short snippet.

            A snippet is an extract, not the page: fetch the URL to read it in
            full. Search again with different words when the results miss.

            Args:
                query: what to look for, as you would type it into a search engine.
                limit: how many results to return.
                recency: only pages from the last "day", "week", "month" or "year"; leave empty for any time.
                domains: only search these sites, e.g. ["docs.python.org"].
            """
            results = await search.search(query, limit=limit, recency=recency,
                                          domains=domains)
            if not results:
                scope = f" on {', '.join(domains)}" if domains else ""
                return (f"No results for {query!r}{scope}. Try fewer or different "
                        "words" + (", or a wider time range." if recency else "."))
            return [r.to_dict() for r in results]

        if most != default_limit:
            web_search.parameters["properties"]["limit"]["description"] = (
                f"how many results to return, at most {most}.")
        return web_search

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<WebSearch {', '.join(e.name for e in self._engines or []) or 'auto'}>"


def make_search_tool(engine: Any = None, *, name: str = "web_search",
                     description: str | None = None, **options: Any) -> Tool:
    """Build a search tool with your own engines and domain policy.

    Takes everything `WebSearch` takes.
    """
    return WebSearch(engine, name=name, **options).as_tool(description=description)


#: The search tool, set up from the environment the first time it is called.
web_search: Tool = make_search_tool()
