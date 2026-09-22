from __future__ import annotations

import httpx
import pytest

from agent_harness.errors import ToolError
from agent_harness.toolkits import calculate, make_corpus_search, now
from agent_harness.toolkits.web import make_fetch_tool


async def test_calculate_is_exact_and_refuses_anything_else():
    assert await calculate.invoke({"expression": "1249 * 0.175 + 32"}) == pytest.approx(250.575)
    with pytest.raises(ToolError):
        await calculate.invoke({"expression": "__import__('os').system('ls')"})
    with pytest.raises(ToolError):
        await calculate.invoke({"expression": "2 ** 999"})


async def test_now_returns_an_iso_timestamp():
    assert "T" in await now.invoke({})
    with pytest.raises(ToolError):
        await now.invoke({"timezone": "Mars/Olympus"})


async def test_corpus_search_ranks_by_hit_count():
    search = make_corpus_search({
        "policy.md": "Refunds are issued within 30 days of purchase.",
        "hours.md": "The office is open 9 to 5.",
    })
    hits = await search.invoke({"query": "refunds days"})
    assert hits[0]["document"] == "policy.md"
    assert "Refunds" in hits[0]["snippet"]
    assert await search.invoke({"query": "zebra"}) == []


async def test_web_fetch_strips_markup_and_honours_the_domain_policy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body><h1>Title</h1>"
                                        "<script>bad()</script><p>Body text</p></body></html>",
                              headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    fetch = make_fetch_tool(allowed_domains=["example.com"])

    # Patch the client the tool builds so the test never touches the network.
    original = httpx.AsyncClient

    class Patched(original):
        def __init__(self, *a, **kw):
            kw["transport"] = transport
            super().__init__(*a, **kw)

    httpx.AsyncClient = Patched
    try:
        text = await fetch.invoke({"url": "https://example.com/page"})
        assert "Title" in text and "Body text" in text
        assert "bad()" not in text and "<p>" not in text
        with pytest.raises(ToolError, match="not on the allowed list"):
            await fetch.invoke({"url": "https://elsewhere.com/page"})
        with pytest.raises(ToolError, match="private or loopback"):
            await fetch.invoke({"url": "http://127.0.0.1/admin"})
        with pytest.raises(ToolError, match="http and https"):
            await fetch.invoke({"url": "file:///etc/passwd"})
    finally:
        httpx.AsyncClient = original
