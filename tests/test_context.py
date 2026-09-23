from __future__ import annotations

import pytest

from agent_harness import ContextAssembler, ContextCompactor, MemoryManager, SkillRegistry
from agent_harness.context import conversation_tokens, estimate_tokens
from agent_harness.memory import InMemoryStore
from agent_harness.types import Message, TextBlock, ToolResultBlock, ToolUseBlock


def conversation(turns: int, *, tool_result_chars: int = 3000) -> list[Message]:
    messages: list[Message] = []
    for i in range(turns):
        messages.append(Message.user(f"question {i}"))
        messages.append(Message(role="assistant", content=[
            ToolUseBlock(id=f"c{i}", name="search", input={"q": f"q{i}"})
        ]))
        messages.append(Message.tool_results([
            ToolResultBlock(tool_use_id=f"c{i}", content="x" * tool_result_chars)
        ]))
        messages.append(Message.assistant(f"answer {i}"))
    return messages


def test_token_estimates_scale_with_content():
    assert estimate_tokens("hello world") < estimate_tokens("hello world" * 10)
    assert conversation_tokens(conversation(2)) > 1000


async def test_compaction_evicts_fat_tool_results_first():
    messages = conversation(6)
    compactor = ContextCompactor(max_tokens=3000, keep_last=4)
    compacted = await compactor.compact(messages)

    # Nothing was dropped — the bulky results were hollowed out instead.
    assert len(compacted) == len(messages)
    evicted = [b for m in compacted for b in m.content
               if isinstance(b, ToolResultBlock) and "evicted" in b.content]
    assert evicted
    assert conversation_tokens(compacted) < conversation_tokens(messages)


async def test_compaction_never_orphans_a_tool_result():
    messages = conversation(20, tool_result_chars=200)
    compactor = ContextCompactor(max_tokens=400, keep_last=3)
    compacted = await compactor.compact(messages)

    # Any tool_result left must still have its tool_use earlier in the list.
    pending = set()
    for message in compacted:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                pending.add(block.id)
            if isinstance(block, ToolResultBlock):
                assert block.tool_use_id in pending, "a tool result lost its call"


async def test_compaction_summarises_the_head_when_a_model_is_available():
    async def summarize(prompt: str) -> str:
        return "EARLIER: the user asked twenty questions."

    messages = conversation(20, tool_result_chars=200)
    compactor = ContextCompactor(max_tokens=400, keep_last=3, summarize=summarize)
    compacted = await compactor.compact(messages)
    assert "EARLIER" in compacted[0].text
    assert len(compacted) < len(messages)


async def test_compaction_keeps_pinned_facts():
    messages = conversation(20, tool_result_chars=200)
    compactor = ContextCompactor(max_tokens=400, keep_last=3)
    compacted = await compactor.compact(messages, pinned=["The deadline is Friday"])
    assert "The deadline is Friday" in compacted[0].text


async def test_a_short_conversation_is_left_alone():
    messages = conversation(1, tool_result_chars=50)
    compactor = ContextCompactor(max_tokens=100_000)
    assert await compactor.compact(messages) is messages


async def test_the_assembler_layers_identity_instructions_skills_and_memory(tmp_path):
    folder = tmp_path / "refunds"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: refunds\ndescription: How refunds work.\n---\n\nThe long body.\n"
    )
    memory = MemoryManager(InMemoryStore(), semantic=False)
    await memory.user.remember("Writes in British English")

    assembler = ContextAssembler(
        identity="You are support.", instructions="Answer billing questions.",
        skills=SkillRegistry.from_dir(tmp_path), memory=memory,
    )
    system = await assembler.build(query="refund", tool_names=["lookup", "load_skill"])

    assert system.index("You are support.") < system.index("Answer billing questions.")
    assert "refunds: How refunds work." in system
    assert "The long body." not in system      # skills stay closed until loaded
    assert "lookup" in system
    assert "British English" in system


async def test_the_assembler_adds_the_output_contract_when_there_is_one():
    assembler = ContextAssembler(identity="You are a classifier.")
    system = await assembler.build(output_contract="Return JSON with `label`.")
    assert "## Output contract" in system
    assert "Return JSON with `label`." in system


def test_text_blocks_round_trip_through_messages():
    message = Message(role="assistant", content=[TextBlock(text="a"),
                                                 TextBlock(text="b")])
    assert message.text == "a\nb"


# --- where the compaction threshold comes from --------------------------------

def _agent(**kw):
    from agent_harness import Agent, FakeProvider, Harness

    provider = FakeProvider(["ok"], loop=True)
    kw.setdefault("model", "claude-sonnet-5")   # a 1M-token window
    return Agent("t", provider=provider, harness=Harness.testing(provider),
                 memory=False, **kw), provider


def test_an_absolute_threshold_is_used_as_given():
    agent, _ = _agent(compact_at=10_000)
    assert agent.compact_at == 10_000
    assert agent.compactor.max_tokens == 10_000

    agent, _ = _agent(compact_at=50_000)
    assert agent.compactor.max_tokens == 50_000


def test_a_fraction_is_read_against_the_model_window():
    agent, _ = _agent(compact_at=0.5)            # half of a 1M window
    assert agent.compact_at == pytest.approx(500_000, rel=0.01)

    agent, _ = _agent(compact_at=1.0)
    assert agent.compact_at == pytest.approx(1_000_000, rel=0.01)


def test_the_default_leaves_room_for_the_answer():
    agent, _ = _agent()
    assert agent.compact_at == pytest.approx(660_000, rel=0.01)   # two thirds

    small, _ = _agent(model="claude-haiku-4-5")                   # a 200k window
    assert small.compact_at == pytest.approx(132_000, rel=0.01)


def test_max_context_tokens_still_works():
    agent, _ = _agent(max_context_tokens=25_000)
    assert agent.compact_at == 25_000


def test_a_nonsense_threshold_is_refused():
    from agent_harness.errors import ConfigurationError

    for bad in (0, -1, -10_000):
        with pytest.raises(ConfigurationError, match="positive token count"):
            _agent(compact_at=bad)


def test_the_strategy_knobs_reach_the_compactor():
    agent, _ = _agent(compact_at=10_000, compact_keep_last=3, compact_target=0.4)
    assert agent.compactor.keep_last == 3
    assert agent.compactor.target_ratio == 0.4


def test_your_own_compactor_is_used_as_given():
    mine = ContextCompactor(max_tokens=1234, keep_last=2)
    agent, _ = _agent(compactor=mine)
    assert agent.compactor is mine


async def test_the_threshold_actually_triggers_compaction_mid_run():
    """Below the threshold the history goes through untouched; above it, it shrinks."""
    from agent_harness import Agent, FakeProvider, Harness
    from agent_harness.types import Message

    history = conversation(12, tool_result_chars=4_000)   # a few thousand tokens
    before = conversation_tokens(history)
    assert before > 10_000

    provider = FakeProvider(["done"], loop=True)
    roomy = Agent("roomy", provider=provider, harness=Harness.testing(provider),
                  model="claude-sonnet-5", memory=False, compact_at=1_000_000)
    await roomy.run("carry on", messages=list(history))
    assert conversation_tokens(provider.requests[-1].messages) >= before

    provider2 = FakeProvider(["done"], loop=True)
    tight = Agent("tight", provider=provider2, harness=Harness.testing(provider2),
                  model="claude-sonnet-5", memory=False, compact_at=2_000)
    await tight.run("carry on", messages=list(history))
    sent = provider2.requests[-1].messages
    assert conversation_tokens(sent) < before
    assert isinstance(sent[0], Message)


async def test_compaction_at_the_threshold_keeps_the_conversation_valid():
    """Whatever the threshold, the provider must never see an orphaned tool result."""
    from agent_harness import Agent, FakeProvider, Harness

    for threshold in (1_000, 2_500, 10_000):
        provider = FakeProvider(["done"], loop=True)
        agent = Agent("t", provider=provider, harness=Harness.testing(provider),
                      model="claude-sonnet-5", memory=False, compact_at=threshold)
        await agent.run("carry on", messages=conversation(20, tool_result_chars=900))

        seen: set[str] = set()
        for message in provider.requests[-1].messages:
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    seen.add(block.id)
                if isinstance(block, ToolResultBlock):
                    assert block.tool_use_id in seen, (
                        f"orphaned tool result at threshold {threshold}")
