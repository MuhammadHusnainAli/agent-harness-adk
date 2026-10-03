"""Web search: an engine that fails, a fallback that answers, and an agent that cites.

With a search key in the environment (`TAVILY_API_KEY`, `BRAVE_API_KEY`,
`EXA_API_KEY`, `SERPER_API_KEY`, …) this searches the real web. Without one it
searches a small index kept here, so the example runs anywhere — offline too.
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, Harness, WebSearch, tool_call
from agent_harness.toolkits import search_engines
from agent_harness.toolkits.search import SearchFailure, SearchQuery

INDEX = [
    {"title": "Heat pump grants 2026", "published": "2026-03-01",
     "url": "https://energy.example.gov/grants/heat-pumps?utm_source=newsletter",
     "snippet": "Households can claim up to <b>7,500</b> towards an air-source heat pump."},
    {"title": "Heat pump grants 2026 (mirror)",
     "url": "https://www.energy.example.gov/grants/heat-pumps/",
     "snippet": "The same page, linked a second way."},
    {"title": "Ten heat pumps you MUST buy", "url": "https://ads.example.com/top-ten",
     "snippet": "Sponsored."},
    {"title": "Installer register", "url": "https://register.example.org/installers",
     "snippet": "Only a registered installer can apply for the heat pump grant."},
]


def overloaded(query: SearchQuery):
    """The first engine: down, the way a search engine sometimes is."""
    raise SearchFailure("the engine answered 503", retryable=True)


def local_index(query: SearchQuery):
    words = query.text.lower().split()
    return [page for page in INDEX
            if any(w in (page["title"] + page["snippet"]).lower() for w in words)]


async def main() -> None:
    print("engines   ", ", ".join(f"{e['name']}{'' if e['ready'] else ' (not set)'}"
                                 for e in search_engines()), "\n")
    live = any(e["ready"] and e["needs"] for e in search_engines())
    search = (WebSearch(blocked_domains=["ads.example.com"]) if live else
              WebSearch([overloaded, local_index], blocked_domains=["ads.example.com"]))

    # --- call it yourself ------------------------------------------------------
    for hit in await search.search("heat pump grant", limit=3):
        print(f"result     {hit.title} — {hit.url}")
    print(f"answered   {search.last_engine}; {search.stats}\n")
    await search.search("heat pump grant", limit=3)        # the same question: not asked again

    # --- and hand it to an agent --------------------------------------------------
    provider, model = pick_provider([
        tool_call("web_search", query="heat pump grant amount", limit=3),
        "Households can claim up to 7,500 towards a heat pump, through a registered "
        "installer — https://energy.example.gov/grants/heat-pumps",
    ])
    harness = Harness(provider=provider)
    agent = Agent("analyst", "Answer from what you find on the web, and give the URL.",
                  tools=[search.as_tool()], model=model, harness=harness, memory=False)
    async for event in agent.stream("How much is the heat pump grant?"):
        if event.type == "tool_result":
            print(f"tool       {event.data['tool']} → {' '.join(event.text.split())[:100]}")
        elif event.type == "run_end":
            print(f"\nanalyst    {event.data['result'].output}")
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
