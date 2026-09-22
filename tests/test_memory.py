from __future__ import annotations

from agent_harness import (
    FileStore,
    HashEmbedder,
    InMemoryStore,
    MemoryManager,
    MemoryRecord,
    SemanticMemory,
    VectorStore,
)
from agent_harness.memory.manager import SubAgentMemory
from agent_harness.types import Artifact, Message


async def test_user_memory_is_loaded_in_full_and_survives():
    manager = MemoryManager(InMemoryStore(), semantic=False)
    await manager.user.remember("Prefers metric units", kind="preference")
    await manager.user.remember("Ships on Thursdays", kind="standard")
    block = await manager.user.block()
    assert "Prefers metric units" in block and "Ships on Thursdays" in block


async def test_remembering_the_same_fact_twice_does_not_duplicate():
    manager = MemoryManager(InMemoryStore(), semantic=False)
    await manager.user.remember("Prefers metric units")
    await manager.user.remember("Prefers metric units")
    assert (await manager.user.load()).count("Prefers metric units") == 1


async def test_session_close_distills_into_user_memory():
    async def summarize(prompt: str) -> str:
        return "- The user works in EUR."

    manager = MemoryManager(InMemoryStore(), semantic=False, summarize=summarize)
    manager.session.add_message(Message.user("Invoice in euros please"))
    updated = await manager.close_session()
    assert "EUR" in updated


async def test_session_close_without_a_model_still_records():
    manager = MemoryManager(InMemoryStore(), semantic=False)
    manager.session.pin("Deadline is Friday")
    updated = await manager.close_session()
    assert "Deadline is Friday" in updated


async def test_orchestrator_memory_returns_a_digest_not_the_history():
    manager = MemoryManager(InMemoryStore(), semantic=False)
    await manager.orchestrator.plan("Ship the report")
    for i in range(30):
        await manager.orchestrator.finding(f"finding number {i}")
    await manager.orchestrator.spend("research", 0.25)
    await manager.orchestrator.spend("writer", 0.75)
    digest = manager.orchestrator.digest(limit=5)
    assert "Ship the report" in digest
    assert digest.count("finding number") == 5   # a digest, never the whole history
    assert "$1.0000" in digest
    assert manager.orchestrator.total_spend() == 1.0


async def test_subagent_memory_carries_nothing_in_and_resources_out():
    sub = SubAgentMemory("extract the totals", agent="extractor")
    sub.produce(Artifact(name="totals.csv", content="1,2,3"))
    resources = sub.close()
    assert [a.name for a in resources] == ["totals.csv"]
    assert resources[0].produced_by == "extractor"
    assert sub.artifacts == []          # the task's memory dies with the task


async def test_semantic_recall_beats_keyword_overlap():
    store = SemanticMemory(InMemoryStore(), embedder=HashEmbedder(),
                           index=VectorStore())
    await store.append(MemoryRecord(scope="job", text="Refund policy is 30 days"))
    await store.append(MemoryRecord(scope="job", text="Office is closed on Friday"))
    hits = await store.search("what is the refund policy", limit=1)
    assert "Refund" in hits[0].text


async def test_semantic_index_persists_to_disk(tmp_path):
    index = VectorStore(tmp_path / "vectors.jsonl")
    store = SemanticMemory(InMemoryStore(), index=index)
    await store.append(MemoryRecord(scope="job", text="Invoices are paid net 30"))
    reopened = VectorStore(tmp_path / "vectors.jsonl")
    assert len(reopened) == 1
    hits = reopened.search((await HashEmbedder().embed(["invoices net 30"]))[0])
    assert hits and "net 30" in hits[0][1].text


async def test_file_store_round_trips(tmp_path):
    store = FileStore(tmp_path)
    await store.append(MemoryRecord(scope="user", text="likes short answers"))
    await store.write_doc("user.md", "- likes short answers")
    reopened = FileStore(tmp_path)
    rows = await reopened.all("user")
    assert rows[0].text == "likes short answers"
    assert await reopened.read_doc("user.md") == "- likes short answers"


async def test_memory_tools_are_exposed_to_the_agent():
    manager = MemoryManager(InMemoryStore())
    names = {t.name for t in manager.tools()}
    assert names == {"remember", "recall"}
    remember = next(t for t in manager.tools() if t.name == "remember")
    await remember.invoke({"fact": "Deploys happen on Tuesday"})
    recall = next(t for t in manager.tools() if t.name == "recall")
    assert "Tuesday" in await recall.invoke({"query": "when do deploys happen"})


async def test_prompt_blocks_include_user_memory_and_job_digest():
    manager = MemoryManager(InMemoryStore(), semantic=False)
    await manager.user.remember("Writes in British English")
    await manager.orchestrator.plan("Draft the brief")
    blocks = await manager.prompt_blocks("anything")
    joined = "\n".join(blocks)
    assert "British English" in joined and "Draft the brief" in joined
