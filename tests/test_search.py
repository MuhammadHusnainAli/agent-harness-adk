"""Web search: every engine's wire format, and everything that can go wrong — over MockTransport."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs

import httpx
import pytest

from agent_harness import Agent, FakeProvider, Harness, cli, tool_call
from agent_harness.errors import ConfigurationError, ToolError
from agent_harness.llm_providers.resilience import RetryPolicy
from agent_harness.toolkits import WebSearch, make_search_tool, search_engines, web_search
from agent_harness.toolkits.search import (
    ENGINES,
    Brave,
    Google,
    SearchEngine,
    SearchFailure,
    SearchQuery,
    SearchResult,
)

FAST = RetryPolicy(max_retries=2, initial_delay=0.0, max_delay=0.0, jitter=0.0,
                   max_retry_after=1.0)
KEYS = ("TAVILY_API_KEY", "BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY", "EXA_API_KEY",
        "SERPER_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID",
        "GOOGLE_SEARCH_ENGINE_ID", "SEARXNG_URL", "SEARXNG_API_KEY", "AGENT_HARNESS_SEARCH")

PAGES = [{"title": "Heat <strong>pumps</strong> &amp; grants",
          "url": "https://energy.gov/pumps?utm_source=x&id=7",
          "description": "A <b>grant</b> of up to   5,000.", "page_age": "2026-03-01T09:00:00"},
         {"title": "Subsidy guide", "url": "https://example.org/guide",
          "description": "How to apply."}]

DDG_PAGE = """
<div class="result results_links result--ad"><h2 class="result__title">
 <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=shop.example">Buy now</a></h2>
 <a class="result__snippet" href="https://duckduckgo.com/y.js?x=1">An advert.</a></div>
<div class="result results_links web-result"><h2 class="result__title">
 <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fenergy.gov%2Fpumps%3Fid%3D7&amp;rut=abc">Heat pumps &amp; grants</a></h2>
 <span>&nbsp; &nbsp; 2026-03-01T00:00:00.0000000</span>
 <a class="result__snippet" href="x">A <b>grant</b> of up to 5,000.</a></div>
<div class="result results_links web-result"><h2 class="result__title">
 <a class="result__a" rel="nofollow" href="https://example.org/guide">Subsidy guide</a></h2>
 <a class="result__snippet" href="x">How to apply.</a></div>
"""


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    """No test reads a key from the machine it runs on."""
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)


def served(handler, engine="brave", **kw):
    """A search whose engine is answered by `handler`, and the requests it saw."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    kw.setdefault("retry", FAST)
    kw.setdefault("api_key", "sk-secret-key")
    return WebSearch(engine, transport=httpx.MockTransport(record), **kw), seen


def brave_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"web": {"results": PAGES}})


# --- engines, on the wire -----------------------------------------------------------

async def test_brave_sends_the_key_in_a_header_and_reads_the_results():
    search, seen = served(brave_ok, region="us", language="en")
    hits = await search.search("heat pump subsidy", limit=2, recency="year")
    request = seen[0]
    assert request.url.path == "/res/v1/web/search"
    assert request.headers["x-subscription-token"] == "sk-secret-key"
    assert dict(request.url.params) == {
        "q": "heat pump subsidy", "count": "4", "safesearch": "moderate",
        "text_decorations": "false", "freshness": "py", "country": "US", "search_lang": "en"}
    assert [h.to_dict() for h in hits] == [
        {"title": "Heat pumps & grants", "url": "https://energy.gov/pumps?id=7",
         "snippet": "A grant of up to 5,000.", "published": "2026-03-01"},
        {"title": "Subsidy guide", "url": "https://example.org/guide",
         "snippet": "How to apply."}]
    assert hits[0].engine == "brave" and search.last_engine == "brave"


async def test_tavily_and_exa_are_told_the_domains_themselves():
    def tavily(request):
        return httpx.Response(200, json={"results": [
            {"title": "T", "url": "https://docs.python.org/3/", "content": "c",
             "score": 0.9, "published_date": "2026-01-02"}]})

    search, seen = served(tavily, "tavily", blocked_domains=["spam.example"])
    hits = await search.search("asyncio", domains=["https://docs.python.org/3/"],
                               recency="week")
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer sk-secret-key"
    assert body["query"] == "asyncio" and body["time_range"] == "week"
    assert body["include_domains"] == ["docs.python.org"]
    assert body["exclude_domains"] == ["spam.example"]
    assert hits[0].score == 0.9 and hits[0].published == "2026-01-02"

    def exa(request):
        return httpx.Response(200, json={"results": [
            {"title": "E", "url": "https://docs.python.org/3/library/",
             "highlights": ["first part", "second part"], "publishedDate": "2025-12-01T00:00:00Z"}]})

    search, seen = served(exa, "exa")
    hits = await search.search("asyncio", domains=["docs.python.org"], recency="month")
    body = json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == "sk-secret-key"
    assert body["includeDomains"] == ["docs.python.org"] and "startPublishedDate" in body
    assert hits[0].snippet == "first part … second part"


async def test_serper_google_and_searxng_wire_formats():
    def serper(request):
        return httpx.Response(200, json={"organic": [
            {"title": "S", "link": "https://a.example/1", "snippet": "s", "date": "3 days ago"}]})

    search, seen = served(serper, "serper", region="GB")
    hits = await search.search("q", recency="day")
    assert seen[0].headers["x-api-key"] == "sk-secret-key"
    assert json.loads(seen[0].content) == {"q": "q", "num": 7, "tbs": "qdr:d", "gl": "gb"}
    assert hits[0].published == "3 days ago"

    def google(request):
        return httpx.Response(200, json={"items": [
            {"title": "G", "link": "https://a.example/2", "snippet": "g"}]})

    search, seen = served(google, Google("g-key", cx="cx-1"), api_key=None)
    hits = await search.search("q", recency="month")
    params = dict(seen[0].url.params)
    assert params["key"] == "g-key" and params["cx"] == "cx-1"
    assert params["dateRestrict"] == "m1" and params["num"] == "7"
    assert hits[0].url == "https://a.example/2"

    def searx(request):
        return httpx.Response(200, json={"results": [
            {"title": "X", "url": "https://a.example/3", "content": "x"}]})

    search, seen = served(searx, ENGINES["searxng"](base_url="https://searx.corp/"),
                          api_key=None, safe_search="strict")
    hits = await search.search("q")
    assert str(seen[0].url).startswith("https://searx.corp/search?")
    assert dict(seen[0].url.params) == {"q": "q", "format": "json", "safesearch": "2"}
    assert hits[0].title == "X"


async def test_duckduckgo_reads_the_page_unwraps_links_and_drops_adverts():
    def page(request):
        return httpx.Response(200, text=DDG_PAGE)

    search, seen = served(page, "duckduckgo", api_key=None)
    hits = await search.search("heat pump subsidy", recency="week")
    form = parse_qs(seen[0].content.decode())
    assert seen[0].method == "POST" and seen[0].url.path == "/html/"
    assert form == {"q": ["heat pump subsidy"], "kp": ["-1"], "df": ["w"]}
    assert [h.to_dict() for h in hits] == [
        {"title": "Heat pumps & grants", "url": "https://energy.gov/pumps?id=7",
         "snippet": "A grant of up to 5,000.", "published": "2026-03-01"},
        {"title": "Subsidy guide", "url": "https://example.org/guide",
         "snippet": "How to apply."}]


async def test_duckduckgo_says_when_it_is_throttled_or_unreadable():
    for response, why in [
            (httpx.Response(202, text="<div class='anomaly-modal'></div>"), "throttling"),
            (httpx.Response(200, text="<html><body>new layout</body></html>"), "layout")]:
        search, _ = served(lambda r, response=response: response, "duckduckgo", api_key=None)
        with pytest.raises(ToolError, match=why):
            await search.search("q")
    search, _ = served(lambda r: httpx.Response(
        200, text='<div class="no-results">No results.</div>'), "duckduckgo", api_key=None)
    assert await search.search("q") == []


# --- failing, retrying, falling back -------------------------------------------------

async def test_a_throttled_or_failing_call_is_retried():
    answers = [httpx.Response(429, headers={"retry-after": "0"}),
               httpx.Response(503), httpx.Response(200, json={"web": {"results": PAGES}})]
    search, seen = served(lambda r: answers[len(seen) - 1])
    assert len(await search.search("q")) == 2
    assert len(seen) == 3
    assert search.stats["brave"] == {"answered": 1, "failed": 0, "last_error": ""}


async def test_a_refused_key_is_not_retried_and_the_next_engine_answers():
    def handler(request):
        if "brave" in request.url.host:
            return httpx.Response(401, json={"error": "bad key sk-secret-key"})
        return httpx.Response(200, text=DDG_PAGE)

    search, seen = served(handler, [Brave("sk-secret-key"), "duckduckgo"], api_key=None)
    hits = await search.search("q")
    assert [r.url.host for r in seen] == ["api.search.brave.com", "html.duckduckgo.com"]
    assert hits[0].engine == "duckduckgo" and search.engines == ["brave", "duckduckgo"]
    assert search.stats["brave"]["last_error"] == "the API key was refused"


async def test_when_every_engine_fails_the_model_is_told_why_without_the_key():
    def handler(request):
        if "googleapis" in request.url.host:
            raise httpx.ConnectError(f"cannot connect to {request.url}")
        if "brave" in request.url.host:
            return httpx.Response(400, text="bad query for token sk-secret-key")
        return httpx.Response(429, text="monthly quota exceeded")

    search, seen = served(handler, [Google("g-secret-key", cx="cx"), Brave("sk-secret-key"),
                                    ENGINES["serper"]("sp-key")], api_key=None)
    with pytest.raises(ToolError) as caught:
        await search.search("q")
    message = str(caught.value)
    assert "google: could not be reached (ConnectError)" in message
    assert "brave: refused the query (400)" in message
    assert "serper: out of quota" in message
    assert "secret-key" not in message
    # Google was retried, the other two were not worth asking again.
    assert [r.url.host for r in seen].count("www.googleapis.com") == 3
    assert len(seen) == 5


async def test_a_missing_key_says_which_variable_to_set():
    search = WebSearch("brave")
    with pytest.raises(ToolError, match="brave: no API key — set BRAVE_API_KEY"):
        await search.search("q")
    with pytest.raises(ToolError, match="google: no search engine id — set GOOGLE_CSE_ID"):
        await WebSearch("google").search("q")


async def test_an_engine_that_keeps_failing_is_left_alone_for_a_while():
    search, seen = served(lambda r: httpx.Response(500), retry=RetryPolicy.none(),
                          failure_threshold=2, cache_ttl=0)
    for _ in range(2):
        with pytest.raises(ToolError, match="answered 500"):
            await search.search("q")
    with pytest.raises(ToolError, match="left alone after repeated failures"):
        await search.search("q")
    assert len(seen) == 2


async def test_a_cancelled_probe_does_not_lock_the_engine_out(monkeypatch):
    state = {"mode": "fail"}

    async def engine(query):
        if state["mode"] == "fail":
            raise SearchFailure("offline")
        if state["mode"] == "hang":
            await asyncio.sleep(5)
        return [{"title": "Back", "url": "https://a.example/"}]

    search = WebSearch(engine, failure_threshold=1, cooldown=0.0, cache_ttl=0,
                       retry=RetryPolicy.none())
    with pytest.raises(ToolError):
        await search.search("q")
    state["mode"] = "hang"                                   # the probe is cancelled
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await asyncio.wait_for(search.search("q"), 0.02)
    state["mode"] = "ok"
    assert (await search.search("q"))[0].title == "Back"


async def test_a_slow_engine_times_out_and_the_search_has_a_deadline():
    async def slow(query):
        await asyncio.sleep(5)

    def quick(query):
        return [SearchResult("Found", "https://a.example/")]

    search = WebSearch([slow, quick], timeout=0.05, retry=RetryPolicy.none())
    hits = await search.search("q")
    assert hits[0].title == "Found" and hits[0].engine == "quick"
    assert "no answer within 0.05s" in search.stats["slow"]["last_error"]

    stuck = WebSearch(slow, timeout=0.05, deadline=0.12,
                      retry=RetryPolicy(max_retries=50, initial_delay=0.0, jitter=0.0))
    started = asyncio.get_running_loop().time()
    with pytest.raises(ToolError, match="slow: "):
        await stuck.search("q")
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_an_engine_that_raises_or_answers_nonsense_is_a_failure_not_a_crash():
    def broken(query):
        raise RuntimeError("boom")

    with pytest.raises(ToolError, match="broken: RuntimeError: boom"):
        await WebSearch(broken).search("q")
    search, _ = served(lambda r: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ToolError, match="not JSON"):
        await search.search("q")
    # A shape nobody expected is no results, never an exception.
    search, _ = served(lambda r: httpx.Response(200, json={"web": "nope"}))
    assert await search.search("q") == []
    search, _ = served(lambda r: httpx.Response(200, json=["a list"]))
    assert await search.search("q") == []


async def test_an_engine_that_finds_nothing_passes_the_question_on():
    asked: list[str] = []

    def empty(query):
        asked.append("empty")
        return []

    def full(query):
        asked.append("full")
        return [{"title": "Found", "url": "https://a.example/"}]

    assert len(await WebSearch([empty, full]).search("q")) == 1
    assert asked == ["empty", "full"]
    asked.clear()
    assert await WebSearch([empty, full], fallback_on_empty=False).search("q") == []
    assert asked == ["empty"]


# --- results ---------------------------------------------------------------------

async def test_results_are_cleaned_deduplicated_and_cut_to_the_limit():
    def engine(query):
        return [
            {"title": "", "url": "https://www.a.example/page/?utm_campaign=x&b=2&a=1#:~:text=hi"},
            {"title": "Same page", "url": "http://a.example/page?a=1&b=2"},
            {"title": "Not a page", "url": "javascript:alert(1)"},
            {"title": "No url"}, "junk", None,
            {"title": "Long", "url": "https://b.example/", "snippet": "word " * 400},
            {"title": "Third", "url": "https://c.example/"},
            {"title": "Fourth", "url": "https://d.example/"}]

    hits = await WebSearch(engine, snippet_chars=120).search("q", limit=3)
    assert [h.url for h in hits] == ["https://www.a.example/page/?b=2&a=1",
                                     "https://b.example/", "https://c.example/"]
    assert hits[0].title == "www.a.example"
    assert len(hits[1].snippet) <= 121 and hits[1].snippet.endswith("…")


async def test_the_domain_policy_holds_whatever_the_engine_returns():
    def engine(query):
        return [{"title": t, "url": u} for t, u in [
            ("gov", "https://www.energy.gov/a"), ("sub", "https://data.energy.gov/b"),
            ("lookalike", "https://notenergy.gov.evil.example/c"),
            ("blocked", "https://ads.energy.gov/d"), ("other", "https://example.org/e")]]

    search = WebSearch(engine, allowed_domains=["energy.gov", "*.europa.eu"],
                       blocked_domains=["ads.energy.gov"])
    assert [h.title for h in await search.search("q")] == ["gov", "sub"]
    assert [h.title for h in await search.search("q", domains=["data.energy.gov"])] == ["sub"]
    with pytest.raises(ToolError, match="not on the allowed list"):
        await search.search("q", domains=["example.org"])
    with pytest.raises(ToolError, match="blocked list"):
        await search.search("q", domains=["ads.energy.gov"])


async def test_domains_become_site_operators_for_engines_without_a_filter():
    search, seen = served(brave_ok, blocked_domains=["pinterest.com"])
    await search.search("grants", domains=["energy.gov", "example.org"])
    assert seen[0].url.params["q"] == ("grants (site:energy.gov OR site:example.org) "
                                       "-site:pinterest.com")
    assert SearchQuery("x", domains=("a.com",)).with_sites() == "x site:a.com"
    assert SearchQuery("x", domains=("*.gov",)).with_sites() == "x"


# --- the question ----------------------------------------------------------------

async def test_the_query_is_checked_before_anything_is_sent():
    search, seen = served(brave_ok, max_limit=4)
    with pytest.raises(ToolError, match="query is empty"):
        await search.search("  \n ")
    with pytest.raises(ToolError, match="recency is one of"):
        await search.search("q", recency="fortnight")
    assert not seen
    await search.search("q", limit=99)                      # clamped, not refused
    assert seen[-1].url.params["count"] == "6"
    await search.search("word " * 300, recency="W")
    assert len(seen[-1].url.params["q"].split()) == 50
    assert seen[-1].url.params["freshness"] == "pw"


async def test_the_same_question_is_asked_once():
    search, seen = served(brave_ok)
    first, second = await asyncio.gather(search.search("Heat  pumps"),
                                         search.search("heat pumps"))
    assert len(seen) == 1 and first == second
    first[0].title = "changed by the caller"
    assert (await search.search("heat pumps"))[0].title == "Heat pumps & grants"
    assert len(seen) == 1
    await search.search("heat pumps", recency="day")         # a different question
    search.clear_cache()
    await search.search("heat pumps")
    assert len(seen) == 3

    uncached, seen = served(brave_ok, cache_ttl=0)
    await uncached.search("q")
    await uncached.search("q")
    assert len(seen) == 2


async def test_questions_in_flight_are_shared_and_so_is_their_failure():
    calls: list[str] = []

    async def engine(query):
        calls.append(query.text)
        await asyncio.sleep(0.01)
        if "fail" in query.text:
            raise SearchFailure("offline")
        return [{"title": "Found", "url": "https://a.example/"}]

    search = WebSearch(engine, cache_ttl=0)
    first, second = await asyncio.gather(search.search("q"), search.search("q"))
    assert calls == ["q"] and first == second and first[0] is not second[0]
    outcomes = await asyncio.gather(search.search("fail"), search.search("fail"),
                                    return_exceptions=True)
    assert all(isinstance(o, ToolError) for o in outcomes) and calls == ["q", "fail"]
    await search.search("q")                                 # nothing is left behind
    assert calls == ["q", "fail", "q"] and not search._inflight


# --- choosing engines ----------------------------------------------------------------

def test_engines_come_from_the_environment_with_duckduckgo_last(monkeypatch):
    assert WebSearch().engines == ["duckduckgo"]
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "b")
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("GOOGLE_API_KEY", "a Gemini key, with no search engine id")
    assert WebSearch().engines == ["tavily", "brave", "duckduckgo"]
    monkeypatch.setenv("AGENT_HARNESS_SEARCH", "ddg, brave")
    assert WebSearch().engines == ["duckduckgo", "brave"]
    ready = {e["name"]: e["ready"] for e in search_engines()}
    assert ready["brave"] and ready["duckduckgo"] and not ready["google"]


def test_a_search_that_is_set_up_wrong_fails_where_it_was_written():
    with pytest.raises(ConfigurationError, match="no search engine named 'bing'"):
        WebSearch("bing")
    with pytest.raises(ConfigurationError, match="api_key= goes with one"):
        WebSearch(["brave", "tavily"], api_key="k")
    with pytest.raises(ConfigurationError, match="safe_search"):
        WebSearch(safe_search="maybe")
    with pytest.raises(ConfigurationError, match="are lists"):
        WebSearch(allowed_domains="example.com")
    with pytest.raises(ConfigurationError, match="a name, a SearchEngine"):
        WebSearch(42)


async def test_an_engine_of_your_own_gets_the_retries_and_the_policy():
    class Internal(SearchEngine):
        name, url, env = "internal", "https://search.corp.example", ("CORP_KEY",)

        def request(self, query):
            return {"method": "GET", "url": f"{self.base_url}/q",
                    "params": {"q": query.text, "n": query.limit},
                    "headers": {"authorization": f"Bearer {self.api_key}"}}

        def parse(self, data, query):
            if "hits" not in data:
                raise SearchFailure("index is rebuilding", retryable=True)
            return [{"title": h["name"], "url": h["link"]} for h in data["hits"]]

    answers = [{"status": "warming"}, {"hits": [{"name": "Wiki", "link": "https://wiki.corp.example/x"},
                                                {"name": "Out", "link": "https://elsewhere.example/"}]}]
    search, seen = served(lambda r: httpx.Response(200, json=answers[len(seen) - 1]),
                          Internal("corp-key"), api_key=None,
                          allowed_domains=["corp.example"])
    hits = await search.search("holiday policy")
    assert len(seen) == 2 and seen[0].headers["authorization"] == "Bearer corp-key"
    assert [h.title for h in hits] == ["Wiki"]


# --- as a tool --------------------------------------------------------------------

async def test_the_tool_has_a_small_schema_and_readable_answers():
    assert web_search.name == "web_search" and {"web", "search"} <= web_search.tags
    assert web_search.parameters["required"] == ["query"]
    assert set(web_search.parameters["properties"]) == {"query", "limit", "recency", "domains"}

    def engine(query):
        return [] if "nothing" in query.text else [
            {"title": "Found", "url": "https://a.example/", "snippet": "text"}]

    search = make_search_tool(engine, name="lookup", limit=3, max_limit=8)
    assert search.name == "lookup"
    assert search.parameters["properties"]["limit"]["default"] == 3
    assert await search.invoke({"query": "something"}) == [
        {"title": "Found", "url": "https://a.example/", "snippet": "text"}]
    empty = await search.invoke({"query": "nothing", "recency": "day",
                                 "domains": ["a.example"]})
    assert empty.startswith("No results for 'nothing' on a.example")
    outcome = await search.run("c1", {"query": ""})
    assert outcome.is_error and "query is empty" in outcome.content


async def test_an_agent_searches_and_a_failed_search_is_something_it_reads():
    def engine(query):
        if "down" in query.text:
            raise SearchFailure("the index is offline")
        return [{"title": "Subsidy guide", "url": "https://example.org/guide",
                 "snippet": "Up to 5,000."}]

    harness = Harness.testing(FakeProvider([
        tool_call("web_search", query="is it down"),
        tool_call("web_search", query="heat pump subsidy", limit=1),
        "Up to 5,000 — https://example.org/guide"]))
    agent = Agent("analyst", tools=[make_search_tool(engine)], harness=harness,
                  memory=False)
    read: list[str] = []
    output = ""
    async for event in agent.stream("What is the heat pump subsidy?"):
        if event.type == "tool_result":
            read.append(event.text)
        elif event.type == "run_end":
            output = event.data["result"].output
    assert output == "Up to 5,000 — https://example.org/guide"
    assert "web search failed — engine: the index is offline" in read[0]
    assert "https://example.org/guide" in read[1]


def test_the_cli_lists_the_engines(capsys, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "k")
    assert cli.main(["search"]) == 0
    out = capsys.readouterr().out
    assert "exa          ready" in out and "tavily       not set    TAVILY_API_KEY" in out
    assert "a search uses: exa, duckduckgo" in out
