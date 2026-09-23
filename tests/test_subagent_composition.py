"""Building sub-agents separately, attaching them later, and running them at once."""

from __future__ import annotations

import asyncio
import time

from agent_harness import (
    Agent,
    Budget,
    FakeProvider,
    Harness,
    SubAgentSpec,
    tool,
    tool_call,
)
from agent_harness.subagents import build_agent
from agent_harness.types import Message, ToolUseBlock

MODEL = "claude-sonnet-5"


@tool
async def slow_lookup(topic: str) -> str:
    """Look something up, slowly.

    Args:
        topic: what to look up
    """
    await asyncio.sleep(0.05)
    return f"the answer about {topic}"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return f"about {topic}"


# --- built separately, attached later -------------------------------------------

def test_a_spec_is_a_standalone_object():
    """A sub-agent is declared on its own, with everything it needs."""
    researcher = SubAgentSpec(
        name="researcher",
        description="Finds things out, read-only.",
        instructions="Cite every claim.",
        tools=["lookup"],
        tier="fast",
        budget=Budget(max_input_tokens=10_000, max_output_tokens=2_000),
        guardrails={"require_citation": True},
    )
    assert researcher.name == "researcher"
    assert researcher.budget.max_output_tokens == 2_000
    # It is serialisable, so it can live in a file or a database.
    assert SubAgentSpec(**researcher.model_dump()).name == "researcher"


def test_a_spec_is_attached_after_the_agent_exists():
    provider = FakeProvider(["ok"])
    manager = Agent("manager", provider=provider, harness=Harness.testing(provider),
                    tools=[lookup], memory=False)
    assert manager.subagents == {}
    assert "delegate" not in manager.tools

    researcher = SubAgentSpec(name="researcher", description="Finds things.",
                              tools=["lookup"])
    built = manager.add_subagent(researcher)

    assert built.name == "researcher"
    assert "researcher" in manager.subagents
    assert "delegate" in manager.tools
    assert built.tools.names == ["lookup"]


def test_several_are_attached_and_all_are_reachable():
    provider = FakeProvider(["ok"])
    manager = Agent("manager", provider=provider, harness=Harness.testing(provider),
                    tools=[lookup], memory=False)
    for name in ("research", "analysis", "writing"):
        manager.add_subagent(SubAgentSpec(name=name, description=f"Does {name}."))

    assert sorted(manager.subagents) == ["analysis", "research", "writing"]
    # The delegate tool's schema lists every one of them, not just the first.
    options = manager.tools.get("delegate").parameters["properties"]["agent_name"]
    assert options["enum"] == ["analysis", "research", "writing"]


def test_a_fully_built_agent_can_be_attached_as_a_sub_agent():
    provider = FakeProvider(["ok"])
    harness = Harness.testing(provider)

    # Built on its own, with its own configuration.
    specialist = Agent("specialist", "You are a specialist.", provider=provider,
                       harness=harness, tools=[lookup], memory=False,
                       max_steps=3)

    manager = Agent("manager", provider=provider, harness=harness, memory=False)
    manager.add_subagent(specialist)

    assert manager.subagents["specialist"] is specialist
    assert specialist.max_steps == 3          # its own configuration is kept


def test_a_spec_can_be_built_into_an_agent_without_a_parent_attaching_it():
    provider = FakeProvider(["ok"])
    parent = Agent("parent", provider=provider, harness=Harness.testing(provider),
                   tools=[lookup], memory=False)

    standalone = build_agent(
        SubAgentSpec(name="worker", description="Works.", tools=["lookup"]), parent)

    assert standalone.name == "worker"
    assert standalone.harness is parent.harness      # shares the rails
    assert "worker" not in parent.subagents          # but is not attached


async def test_an_attached_sub_agent_runs():
    provider = FakeProvider([
        tool_call("delegate", agent_name="researcher", task="find it"),
        "found it", "All done.",
    ])
    harness = Harness.testing(provider)
    manager = Agent("manager", provider=provider, harness=harness, tools=[lookup],
                    memory=False)
    manager.add_subagent(SubAgentSpec(name="researcher", description="Finds."))

    result = await manager.run("go")
    assert result.output == "All done."
    assert [c.agent for c in result.children] == ["researcher"]


# --- running them at once ----------------------------------------------------------

async def test_sub_agents_asked_for_together_run_together():
    """Three delegations in one turn should take one delay, not three."""
    provider = FakeProvider([
        Message(role="assistant", content=[
            ToolUseBlock(id=f"c{i}", name="delegate",
                         input={"agent_name": f"worker{i}", "task": f"part {i}"})
            for i in range(3)
        ]),
        tool_call("slow_lookup", topic="a"), "part 0 done",
        tool_call("slow_lookup", topic="b"), "part 1 done",
        tool_call("slow_lookup", topic="c"), "part 2 done",
        "All three parts are done.",
    ])
    harness = Harness.testing(provider)
    manager = Agent("manager", provider=provider, harness=harness,
                    tools=[slow_lookup], memory=False,
                    subagents=[SubAgentSpec(name=f"worker{i}",
                                            description=f"Does part {i}.",
                                            tools=["slow_lookup"])
                               for i in range(3)])

    started = time.perf_counter()
    result = await manager.run("split it three ways")
    elapsed = time.perf_counter() - started

    assert result.output == "All three parts are done."
    assert len(result.children) == 3
    assert {c.agent for c in result.children} == {"worker0", "worker1", "worker2"}
    # Each worker sleeps 50ms. Serially that is 150ms; together it is about 50.
    assert elapsed < 0.12, f"they ran one after another ({elapsed:.3f}s)"


async def test_one_sub_agent_failing_does_not_take_the_others_down():
    provider = FakeProvider([
        Message(role="assistant", content=[
            ToolUseBlock(id="c0", name="delegate",
                         input={"agent_name": "good", "task": "work"}),
            ToolUseBlock(id="c1", name="delegate",
                         input={"agent_name": "bad", "task": "work"}),
        ]),
        "the good one's answer",
        tool_call("lookup", topic="loop"),      # the bad one burns its only step
        "One worked, one did not.",
    ])
    harness = Harness.testing(provider)
    manager = Agent("manager", provider=provider, harness=harness, tools=[lookup],
                    memory=False, subagents=[
                        SubAgentSpec(name="good", description="Works."),
                        SubAgentSpec(name="bad", description="Fails.", max_steps=1),
                    ])

    result = await manager.run("try both")
    assert result.ok
    outcomes = [b.content for m in result.messages for b in m.content
                if getattr(b, "type", "") == "tool_result"]
    assert any("the good one's answer" in o for o in outcomes)
    assert any("failed" in o for o in outcomes)


async def test_parallel_sub_agents_are_capped_by_the_scheduler():
    harness = Harness.testing(FakeProvider(["ok"], loop=True))
    harness.scheduler.max_concurrency = 2
    assert harness.scheduler.max_concurrency == 2

    async def work(n: int) -> int:
        await asyncio.sleep(0.02)
        return n

    await harness.scheduler.map(work, range(6))
    assert harness.scheduler.peak <= 2
