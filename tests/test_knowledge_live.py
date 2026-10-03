"""The vector stores, against real databases — when it is pointed at some.

Skipped unless `AGENT_HARNESS_VECTOR_STORES` names them, separated by spaces:

    AGENT_HARNESS_VECTOR_STORES="qdrant://localhost:6333/check \\
        pgvector://postgres:pw@localhost/postgres?table=check" \\
        uv run --with asyncpg pytest tests/test_knowledge_live.py

Each store is put through `store.check()` — create, upsert, find, filter,
replace, delete by id, delete by filter — and through a knowledge base: add,
search, re-add unchanged, shrink, remove. It writes under names of its own and
removes what it wrote.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from agent_harness.knowledge import KnowledgeBase, vector_store

STORES = os.environ.get("AGENT_HARNESS_VECTOR_STORES", "").split()
pytestmark = pytest.mark.skipif(not STORES, reason="no live vector stores were named")


async def eventually(ask, good, tries: int = 60):
    for _ in range(tries):
        value = await ask()
        if good(value):
            return value
        await asyncio.sleep(0.25)
    return value


@pytest.mark.parametrize("url", STORES)
async def test_a_live_store_passes_the_check(url):
    store = vector_store(url)
    try:
        # The dimension the knowledge base test below will use, so the two can
        # share one collection.
        report = await store.check(dimension=256)
        assert report["ok"], [(s["step"], s["detail"]) for s in report["steps"] if not s["ok"]]
    finally:
        await store.aclose()


@pytest.mark.parametrize("url", STORES)
async def test_a_knowledge_base_works_on_a_live_store(url):
    namespace = "live" + uuid.uuid4().hex[:8]
    kb = KnowledgeBase(vector_store(url), namespace=namespace, chunk_size=400)
    text = ("# Refunds\n\nA refund takes five working days. " + "Banks vary. " * 80
            + "\n\n# Shipping\n\nThe code XK-42 means express delivery.")
    try:
        added = await kb.add(text, id="handbook", title="Handbook")
        found = await eventually(lambda: kb.search("what does XK-42 mean", k=3),
                                 lambda hits: hits and "XK-42" in hits[0].text)
        assert found and "XK-42" in found[0].text and found[0].document == "handbook"
        assert (await kb.add(text, id="handbook", title="Handbook")).unchanged
        assert added.chunks > 1 and (await kb.add("Two days now.", id="handbook")).chunks == 1
        left = await eventually(lambda: kb.search("refund", k=10), lambda hits: len(hits) == 1)
        assert [p.text for p in left] == ["Two days now."]
    finally:
        await kb.delete("handbook")
        assert await eventually(lambda: kb.search("refund", k=10), lambda hits: not hits) == []
        await kb.aclose()
