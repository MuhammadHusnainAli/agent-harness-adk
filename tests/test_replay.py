"""Replay and time travel: reproduce before you fix."""

from __future__ import annotations

import pytest

from agent_harness import (
    Agent,
    Checkpointer,
    FakeProvider,
    Harness,
    RecordingProvider,
    Replayer,
    ReplayProvider,
    tool,
    tool_call,
)
from agent_harness.errors import ConfigurationError, ProviderError
from agent_harness.llm_providers.base import CompletionRequest
from agent_harness.runtime.replay import request_key
from agent_harness.types import Message

MODEL = "claude-sonnet-5"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return f"the answer about {topic}"


def build(script, tmp_path=None, **kw):
    provider = FakeProvider(script)
    harness = Harness.testing(provider)
    if tmp_path is not None:
        harness.checkpoints = Checkpointer(tmp_path, every=1)
    agent = Agent("worker", "You do work.", provider=provider, model=MODEL,
                  harness=harness, tools=[lookup], memory=False, **kw)
    return agent, harness, provider


# --- fingerprinting --------------------------------------------------------

def test_the_same_request_fingerprints_the_same_way():
    a = CompletionRequest(model=MODEL, messages=[Message.user("hello")], system="be brief")
    b = CompletionRequest(model=MODEL, messages=[Message.user("hello")], system="be brief")
    c = CompletionRequest(model=MODEL, messages=[Message.user("goodbye")], system="be brief")
    assert request_key(a) == request_key(b)
    assert request_key(a) != request_key(c)


def test_a_changed_system_prompt_changes_the_fingerprint():
    a = CompletionRequest(model=MODEL, messages=[Message.user("x")], system="one")
    b = CompletionRequest(model=MODEL, messages=[Message.user("x")], system="two")
    assert request_key(a) != request_key(b)


# --- record and replay -----------------------------------------------------

async def test_a_recorded_run_replays_identically_with_no_network(tmp_path):
    recording = tmp_path / "run.jsonl"

    live = FakeProvider([tool_call("lookup", topic="margins"), "Margins are 24%."])
    recorder = RecordingProvider(live, recording)
    harness = Harness.testing(recorder)
    agent = Agent("worker", "You do work.", provider=recorder, model=MODEL,
                  harness=harness, tools=[lookup], memory=False)

    first = await agent.run("what are the margins?")
    assert first.output == "Margins are 24%."
    assert recorder.count == 2
    assert recording.exists()

    # Replay: same agent, same input, no live provider at all.
    replay = ReplayProvider(recording)
    harness2 = Harness.testing(replay)
    agent2 = Agent("worker", "You do work.", provider=replay, model=MODEL,
                   harness=harness2, tools=[lookup], memory=False)
    second = await agent2.run("what are the margins?")

    assert second.output == first.output
    assert [c.name for c in second.tool_calls] == [c.name for c in first.tool_calls]
    assert replay.divergences == []       # every call matched its recording
    assert replay.exhausted


async def test_replay_reports_where_a_run_diverged(tmp_path):
    recording = tmp_path / "run.jsonl"
    live = FakeProvider(["the original answer"])
    recorder = RecordingProvider(live, recording)
    harness = Harness.testing(recorder)
    agent = Agent("worker", "You do work.", provider=recorder, model=MODEL,
                  harness=harness, memory=False)
    await agent.run("the original question")

    # Ask something else: the fingerprint no longer matches the recording.
    replay = ReplayProvider(recording)
    harness2 = Harness.testing(replay)
    agent2 = Agent("worker", "You do work.", provider=replay, model=MODEL,
                   harness=harness2, memory=False)
    result = await agent2.run("a completely different question")

    assert result.output == "the original answer"   # fell back, in order
    assert replay.divergences and "call 1" in replay.divergences[0]   # 1-based


async def test_strict_replay_refuses_to_guess(tmp_path):
    recording = tmp_path / "run.jsonl"
    recorder = RecordingProvider(FakeProvider(["recorded"]), recording)
    harness = Harness.testing(recorder)
    agent = Agent("worker", "You do work.", provider=recorder, model=MODEL,
                  harness=harness, memory=False)
    await agent.run("question one")

    replay = ReplayProvider(recording, strict=True)
    harness2 = Harness.testing(replay)
    agent2 = Agent("worker", "You do work.", provider=replay, model=MODEL,
                   harness=harness2, memory=False)
    result = await agent2.run("question two")
    assert not result.ok and "diverged" in result.error


async def test_running_out_of_recording_is_a_clear_error(tmp_path):
    recording = tmp_path / "run.jsonl"
    recorder = RecordingProvider(FakeProvider(["one"]), recording)
    harness = Harness.testing(recorder)
    agent = Agent("w", provider=recorder, model=MODEL, harness=harness, memory=False)
    await agent.run("q")

    replay = ReplayProvider(recording)
    await replay.complete(CompletionRequest(model=MODEL, messages=[Message.user("a")]))
    with pytest.raises(ProviderError, match="ran out of responses"):
        await replay.complete(CompletionRequest(model=MODEL,
                                                messages=[Message.user("b")]))


def test_a_missing_recording_says_so(tmp_path):
    with pytest.raises(ConfigurationError, match="no recording"):
        ReplayProvider(tmp_path / "nope.jsonl")


# --- time travel -----------------------------------------------------------

async def test_the_timeline_shows_what_happened_at_each_step(tmp_path):
    agent, harness, _ = build([tool_call("lookup", topic="q3"), "Q3 was fine."],
                              tmp_path)
    result = await agent.run("how was q3?")

    timeline = await Replayer(harness.checkpoints).timeline(result.run_id)
    assert [row["step"] for row in timeline] == [1, 2]
    assert timeline[0]["tools_called"] == ["lookup"]
    assert "Q3 was fine." in timeline[1]["last"]


async def test_resuming_from_a_step_replays_the_state_it_had_then(tmp_path):
    agent, harness, provider = build(
        [tool_call("lookup", topic="q3"), "Q3 was fine."], tmp_path)
    first = await agent.run("how was q3?")

    # Travel back to step 1 — after the tool call, before the wrong answer — and
    # ask a different question with the same context.
    provider.queue("Actually Q3 revenue was 4.2M.")
    replayer = Replayer(harness.checkpoints)
    second = await replayer.resume(agent, first.run_id, step=1,
                                   task="Give the revenue figure, not a summary.")

    assert second.output == "Actually Q3 revenue was 4.2M."
    resumed_request = provider.requests[-1]
    transcript = "\n".join(m.text for m in resumed_request.messages)
    assert "Give the revenue figure" in transcript
    assert "how was q3?" in transcript            # the earlier context came along


async def test_resume_never_leaves_an_unanswered_tool_call(tmp_path):
    """A checkpoint taken right after a tool call must not resume mid-pair."""
    agent, harness, provider = build(
        [tool_call("lookup", topic="q3"), "done"], tmp_path)
    result = await agent.run("go")

    checkpoint = await harness.checkpoints.load(result.run_id, step=1)
    assert checkpoint.messages[-1].tool_uses      # the raw checkpoint ends mid-pair

    provider.queue("resumed cleanly")
    await Replayer(harness.checkpoints).resume(agent, result.run_id, step=1,
                                               task="carry on")
    sent = provider.requests[-1].messages
    assert not (sent[-1].role == "assistant" and sent[-1].tool_uses)
    for index, message in enumerate(sent):
        for block in message.content:
            if getattr(block, "type", "") == "tool_result":
                earlier = [b.id for m in sent[:index] for b in m.tool_uses]
                assert block.tool_use_id in earlier


async def test_resuming_without_a_task_repeats_the_original_question(tmp_path):
    agent, harness, provider = build(["first answer"], tmp_path)
    result = await agent.run("the original question")

    provider.queue("second answer")
    again = await Replayer(harness.checkpoints).resume(agent, result.run_id)
    assert again.output == "second answer"
    assert "the original question" in provider.requests[-1].messages[-1].text


async def test_inspecting_a_run_that_was_never_checkpointed_says_so(tmp_path):
    replayer = Replayer(Checkpointer(tmp_path, every=1))
    with pytest.raises(ConfigurationError, match="no checkpoints"):
        await replayer.inspect("run_that_never_happened")


async def test_the_harness_exposes_the_replayer(tmp_path):
    agent, harness, _ = build(["done"], tmp_path)
    result = await agent.run("go")
    assert await harness.replayer.history(result.run_id)
