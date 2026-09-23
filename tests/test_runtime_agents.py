"""Run-time agents: the core agent spinning up its own specialists, within a budget."""

from __future__ import annotations

import json

import pytest

from agent_harness import Agent, FakeProvider, Harness, tool, tool_call
from agent_harness.errors import ConfigurationError

MODEL = "claude-sonnet-5"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return f"the answer about {topic}"


@tool(tags=["write"])
def publish(text: str) -> str:
    """Publish something.

    Args:
        text: what to publish
    """
    return "published"


def spec_json(name: str = "ledger_reconciler", tools: list[str] | None = None) -> str:
    return json.dumps({
        "name": name, "description": "Reconciles the ledger.",
        "instructions": "Reconcile it and report.",
        "tools": tools if tools is not None else ["lookup"],
        "tier": "fast", "max_steps": 4, "workspace": "none",
    })


def build(script, **kw) -> tuple[Agent, FakeProvider, Harness]:
    provider = FakeProvider(script)
    harness = Harness.testing(provider)
    kw.setdefault("runtime_agents", "enable")
    kw.setdefault("memory", False)
    agent = Agent("core", "You do work.", provider=provider, model=MODEL,
                  harness=harness, tools=[lookup, publish], **kw)
    return agent, provider, harness


# --- the switch ---------------------------------------------------------------

def test_the_switch_takes_words_or_booleans():
    for value in ("enable", "enabled", "ON", "yes", True):
        agent, _, _ = build(["x"], runtime_agents=value)
        assert agent.runtime_agents is True, value
    for value in ("disable", "disabled", "off", "no", False, ""):
        agent, _, _ = build(["x"], runtime_agents=value, max_runtime_agents=5)
        assert agent.runtime_agents is False, value


def test_an_unknown_switch_value_is_refused():
    with pytest.raises(ConfigurationError, match="enable/disable"):
        build(["x"], runtime_agents="maybe")


def test_the_spawn_tool_appears_only_when_enabled():
    enabled, _, _ = build(["x"], runtime_agents="enable")
    assert "spawn_agent" in enabled.tools
    assert "delegate" in enabled.tools      # spawned agents are reusable by name

    disabled, _, _ = build(["x"], runtime_agents="disable")
    assert "spawn_agent" not in disabled.tools


def test_the_count_is_bounded_between_zero_and_a_hundred():
    agent, _, _ = build(["x"], max_runtime_agents=100)
    assert agent.max_runtime_agents == 100

    for bad in (101, -1, 1000):
        with pytest.raises(ConfigurationError, match="between 0 and 100"):
            build(["x"], max_runtime_agents=bad)

    with pytest.raises(ConfigurationError, match="whole number"):
        build(["x"], max_runtime_agents="five")


def test_enabling_with_a_budget_of_zero_is_a_contradiction():
    with pytest.raises(ConfigurationError, match="max_runtime_agents is 0"):
        build(["x"], runtime_agents="enable", max_runtime_agents=0)


def test_a_disabled_agent_reports_no_budget():
    agent, _, _ = build(["x"], runtime_agents="disable")
    assert agent.runtime_agents_remaining == 0


def test_the_budget_is_stated_in_the_tool_description():
    agent, _, _ = build(["x"], max_runtime_agents=5)
    assert "5 of these" in agent.tools.get("spawn_agent").description


# --- spinning one up ------------------------------------------------------------

async def test_the_core_agent_writes_a_specialist_and_runs_it():
    agent, provider, harness = build([
        tool_call("spawn_agent", task="reconcile the widget ledger",
                  purpose="Reconciles ledgers."),
        spec_json(),                       # the factory writes the spec
        "the ledger reconciles to 4.2M",   # the specialist's answer
        "The ledger reconciles to 4.2M.",  # the core agent's answer
    ], max_runtime_agents=5)

    result = await agent.run("sort out the ledger")

    assert result.output == "The ledger reconciles to 4.2M."
    assert [c.agent for c in result.children] == ["ledger_reconciler"]
    assert agent.total_spawned == 1
    assert agent.runtime_agents_remaining == 4
    # It exists for the rest of the run, addressable by name.
    assert "ledger_reconciler" in agent.subagents


async def test_the_specialist_gets_only_the_tools_it_asked_for():
    agent, provider, _ = build([
        tool_call("spawn_agent", task="reconcile it", tools=["lookup"]),
        spec_json(tools=["lookup"]),
        "done", "all sorted",
    ])
    await agent.run("go")
    child = agent.subagents["ledger_reconciler"]
    assert child.tools.names == ["lookup"]     # no publish, no spawn, no delegate


async def test_a_spawned_specialist_cannot_spawn_its_own():
    agent, _, _ = build([
        tool_call("spawn_agent", task="do it"),
        spec_json(), "done", "finished",
    ])
    await agent.run("go")
    child = agent.subagents["ledger_reconciler"]
    assert child.runtime_agents is False       # delegation does not cascade
    assert "spawn_agent" not in child.tools


async def test_the_specialist_starts_with_no_memory_of_the_conversation():
    agent, provider, _ = build([
        tool_call("spawn_agent", task="find the totals"),
        spec_json(), "found them", "done",
    ])
    await agent.run("a long conversation the specialist must not inherit")
    child_request = provider.requests[-2]
    transcript = "\n".join(m.text for m in child_request.messages)
    assert "find the totals" in transcript
    assert "long conversation" not in transcript


# --- the budget -------------------------------------------------------------------

async def test_the_budget_is_enforced_and_says_what_is_left():
    script = [tool_call("spawn_agent", task="one"), spec_json("agent_one"), "a",
              tool_call("spawn_agent", task="two"), spec_json("agent_two"), "b",
              tool_call("spawn_agent", task="three"),   # over the budget of 2
              "I have no specialists left, so I finished it myself."]
    agent, _, _ = build(script, max_runtime_agents=2)

    result = await agent.run("do three things")

    assert agent.total_spawned == 2
    assert agent.runtime_agents_remaining == 0
    refusals = [b.content for m in result.messages for b in m.content
                if getattr(b, "type", "") == "tool_result"
                and "No run-time agents left" in b.content]
    assert refusals and "2 of 2" in refusals[0]
    assert result.ok


async def test_the_budget_refreshes_for_the_next_run():
    agent, provider, _ = build([
        tool_call("spawn_agent", task="one"), spec_json("agent_one"), "a", "done",
    ], max_runtime_agents=1)

    await agent.run("first job")
    assert agent.runtime_agents_remaining == 0

    provider.queue(tool_call("spawn_agent", task="two"), spec_json("agent_two"),
                   "b", "done again")
    await agent.run("second job")

    assert agent.total_spawned == 2          # two across the two runs
    assert "agent_two" in agent.subagents


async def test_five_specialists_can_be_spun_up_at_once():
    """The stated case: the core agent needs five, so it spins five up in one turn."""
    from agent_harness.types import Message, ToolUseBlock

    five = Message(role="assistant", content=[
        ToolUseBlock(id=f"c{i}", name="spawn_agent",
                     input={"task": f"handle part {i}"})
        for i in range(5)
    ])
    script = [five]
    for i in range(5):
        script += [spec_json(f"worker_{i}"), f"part {i} done"]
    script.append("All five parts are done.")

    agent, _, harness = build(script, max_runtime_agents=5)
    result = await agent.run("split this five ways")

    assert result.output == "All five parts are done."
    assert agent.total_spawned == 5
    assert len(result.children) == 5
    assert sorted(agent.subagents) == [f"worker_{i}" for i in range(5)]
    assert harness.guard.subagents == 5       # each one counted against the budget


async def test_going_over_the_budget_in_one_turn_is_partly_refused():
    from agent_harness.types import Message, ToolUseBlock

    three = Message(role="assistant", content=[
        ToolUseBlock(id=f"c{i}", name="spawn_agent", input={"task": f"part {i}"})
        for i in range(3)
    ])
    script = [three, spec_json("worker_0"), "zero done",
              spec_json("worker_1"), "one done",
              "Two ran, the third had no budget."]
    agent, _, _ = build(script, max_runtime_agents=2)
    result = await agent.run("try three")

    assert agent.total_spawned == 2
    results = [b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result"]
    assert sum("No run-time agents left" in r for r in results) == 1


# --- the rails still apply ---------------------------------------------------------

async def test_a_stopped_run_spins_nothing_up():
    agent, provider, harness = build([
        tool_call("spawn_agent", task="do it"), "I could not start it.",
    ])
    harness.control.stop("budget review", requested_by="finance")
    result = await agent.run("go")

    assert agent.total_spawned == 0
    assert not result.ok and "StopRequested" in result.error


async def test_spinning_one_up_is_audited():
    agent, _, harness = build([
        tool_call("spawn_agent", task="reconcile it"), spec_json(), "done", "finished",
    ])
    await agent.run("go")

    spawns = [e for e in harness.audit.entries if e.action == "spawn_agent"]
    assert len(spawns) == 1
    assert spawns[0].target == "ledger_reconciler"
    assert spawns[0].detail["remaining"] == 4
    assert harness.audit.verify()[0]


async def test_a_spawned_specialist_is_recorded_as_built_not_reused():
    agent, _, _ = build([
        tool_call("spawn_agent", task="reconcile it"), spec_json(), "done", "finished",
    ], memory=True)
    await agent.run("go")
    digest = agent.memory.orchestrator.digest()
    assert "built ledger_reconciler" in digest


async def test_a_spawned_specialist_can_then_be_delegated_to_by_name():
    agent, provider, _ = build([
        tool_call("spawn_agent", task="first task"), spec_json(), "first answer",
        tool_call("delegate", agent_name="ledger_reconciler", task="second task"),
        "second answer",
        "Both done.",
    ], max_runtime_agents=1)

    result = await agent.run("two things")
    assert result.output == "Both done."
    assert agent.total_spawned == 1            # the second call reused it
    assert len(result.children) == 2


async def test_delegating_to_a_name_that_does_not_exist_suggests_spawning():
    agent, _, _ = build([
        tool_call("delegate", agent_name="ghost", task="x"),
        "I will build one instead.",
    ])
    result = await agent.run("go")
    hint = [b.content for m in result.messages for b in m.content
            if getattr(b, "type", "") == "tool_result"][0]
    assert "spawn_agent" in hint and "5 left" in hint


async def test_the_factory_falling_back_still_produces_a_working_specialist():
    agent, _, _ = build([
        tool_call("spawn_agent", task="do the thing"),
        "sorry, I cannot write that spec",     # unusable factory reply
        "the fallback specialist's answer",
        "Done.",
    ])
    result = await agent.run("go")
    assert result.output == "Done."
    assert agent.total_spawned == 1
    assert len(result.children) == 1


async def test_a_disabled_agent_refuses_if_the_tool_is_called_anyway():
    agent, _, _ = build(["x"], runtime_agents="disable")
    assert await agent._spawn("anything") == "run-time agents are disabled for this agent."
