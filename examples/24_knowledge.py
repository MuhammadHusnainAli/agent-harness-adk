"""A knowledge base: documents in, the passages that answer a question out.

The store here is a SQLite file and the embedder is the offline hashing one, so
the example runs anywhere with no key and no server. Change one line —
`KnowledgeBase("qdrant://localhost:6333/handbook", embedder=...)` — and the
same code runs on Qdrant, pgvector, OpenSearch, Pinecone, Azure AI Search or
any other store `vector_stores()` lists.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _common import pick_provider

from agent_harness import Agent, Harness, KnowledgeBase, tool_call
from agent_harness.knowledge import vector_stores

HANDBOOK = """# Refunds

Refunds are issued within 30 days of purchase and go back to the original card.

## How long a refund takes

A refund takes five working days to arrive. Banks vary in how fast they post it.

# Shipping

Orders ship within two working days. The code XK-42 on a label means express
delivery; XK-10 means standard.
"""


async def main() -> None:
    print("stores    ", ", ".join(row["name"] for row in vector_stores()), "\n")
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "handbook.md").write_text(HANDBOOK)
        kb = KnowledgeBase(f"sqlite:///{folder}/knowledge.db", namespace="acme",
                           chunk_size=300, chunk_overlap=40)

        # --- documents in -----------------------------------------------------------
        added = await kb.add(path=Path(folder) / "handbook.md", id="handbook",
                             title="Customer handbook", metadata={"team": "support"})
        print(f"added      {added.id}: {added.chunks} passages")
        await kb.add("Staff discount is 20% and is not to be mentioned to customers.",
                     id="staff-notes", title="Staff notes", metadata={"team": "internal"})
        again = await kb.add(path=Path(folder) / "handbook.md", id="handbook",
                             title="Customer handbook", metadata={"team": "support"})
        print(f"added      {again.id} again: {'unchanged, nothing written' if again.unchanged else again.chunks}\n")

        # --- passages out ----------------------------------------------------------------
        for passage in await kb.search("what does XK-42 mean?", k=2):
            print(f"found      [{passage.score:.2f}] {passage.title}: "
                  f"{' '.join(passage.text.split())[:70]}")

        # --- and an agent, held to the documents it may see --------------------------------
        provider, model = pick_provider([
            tool_call("search_knowledge", query="how long does a refund take"),
            "A refund takes five working days to arrive (Customer handbook).",
        ])
        harness = Harness(provider=provider)
        search = kb.as_tool(filter={"team": "support"})      # never the staff notes
        agent = Agent("support", "Answer from the knowledge base, and say which "
                      "document you used.", tools=[search], model=model,
                      harness=harness, memory=False)
        result = await agent.run("How long does a refund take?")
        print(f"\nsupport    {result.output}")
        hidden = await search.invoke({"query": "staff discount"})
        print("held to    support documents only:",
              all("Staff" not in row.get("title", "") for row in hidden))
        await kb.aclose()
        await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
