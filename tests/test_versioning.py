"""Agent versions: several configurations of one agent, switchable by name."""

from __future__ import annotations

import pytest

from agent_harness import (
    Agent,
    AgentVersion,
    Budget,
    FakeProvider,
    Harness,
    SubAgentSpec,
    tool,
)
from agent_harness.errors import ConfigurationError


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return "shipped Thursday"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return f"about {topic}"


@tool
def issue_refund(amount: float) -> str:
    """Refund a customer.

    Args:
        amount: how much
    """
    return "refunded"


VERSIONS = {
    "v1": {"instructions": "Answer order questions.",
           "tools": ["order_status"],
           "model": "claude-sonnet-5",
           "max_steps": 4},
    "v2": {"instructions": "Answer order questions. Always cite the order.",
           "tools": ["order_status", "lookup"],
           "model": "claude-opus-5",
           "guardrails": {"require_tools": ["order_status"]},
           "notes": "adds the citation requirement"},
}


def build(script=("ok",), version="v2", versions=None, **kw):
    provider = FakeProvider(list(script), loop=True)
    harness = Harness.testing(provider)
    agent = Agent("support", "Base instructions.", provider=provider,
                  harness=harness, memory=False,
                  tools=[order_status, lookup, issue_refund],
                  version=version, versions=versions or VERSIONS, **kw)
    return agent, provider, harness


# --- declaring them ---------------------------------------------------------------

def test_the_active_version_configures_the_agent():
    agent, _, _ = build(version="v2")
    assert agent.version == "v2"
    assert agent.available_versions == ["v1", "v2"]
    assert agent.instructions == "Answer order questions. Always cite the order."
    assert agent.model == "claude-opus-5"
    assert set(agent.tools.names) >= {"order_status", "lookup"}
    assert "issue_refund" not in agent.tools        # not in this version's list


def test_a_different_version_is_a_different_configuration():
    agent, _, _ = build(version="v1")
    assert agent.instructions == "Answer order questions."
    assert agent.model == "claude-sonnet-5"
    assert agent.tools.names == ["order_status"]
    assert agent.max_steps == 4
    assert agent.guardrails is None                 # v1 declares none


def test_anything_a_version_leaves_out_falls_through():
    agent, _, _ = build(version="v1")
    # v1 says nothing about these, so the constructor's values stand.
    assert agent.max_tokens == 8192
    assert agent.parallel_tools is True


def test_an_undefined_version_is_refused_with_the_known_ones():
    with pytest.raises(ConfigurationError, match="version 'v9' is not defined"):
        build(version="v9")


def test_an_agent_with_no_versions_still_reports_one():
    provider = FakeProvider(["ok"])
    agent = Agent("plain", provider=provider, harness=Harness.testing(provider),
                  memory=False)
    assert agent.version == "v1" and agent.available_versions == []


def test_a_version_can_be_an_object_or_a_dict():
    agent, _, _ = build(versions={
        "v1": AgentVersion(instructions="typed", model="claude-sonnet-5"),
        "v2": {"instructions": "dict-shaped"},
    }, version="v1")
    assert agent.instructions == "typed"
    assert agent.version_spec("v2").instructions == "dict-shaped"


def test_notes_travel_with_the_version():
    agent, _, _ = build()
    assert agent.version_spec("v2").notes == "adds the citation requirement"


# --- switching between them ---------------------------------------------------------

def test_use_returns_the_agent_for_that_version():
    agent, _, _ = build(version="v2")
    old = agent.use("v1")

    assert old is not agent
    assert old.version == "v1" and old.model == "claude-sonnet-5"
    assert agent.version == "v2"                    # the original is untouched


def test_the_sibling_is_built_once_and_reused():
    agent, _, _ = build()
    assert agent.use("v1") is agent.use("v1")
    assert agent.use("v2") is agent                 # the active one is itself


def test_versions_share_the_harness_so_they_are_comparable():
    agent, _, harness = build()
    old = agent.use("v1")
    assert old.harness is harness
    assert old.provider is agent.provider


def test_switching_to_an_unknown_version_is_refused():
    agent, _, _ = build()
    with pytest.raises(ConfigurationError, match="not defined"):
        agent.use("v3")


async def test_running_a_specific_version_without_switching():
    agent, provider, _ = build(script=["the v2 answer"])
    provider.responses = ["the v1 answer"]

    result = await agent.run("where is my order?", version="v1")

    assert result.output == "the v1 answer"
    assert result.agent == "support"
    assert agent.version == "v2"                    # still the active one
    # It really ran as v1: that version's model was used.
    assert provider.requests[-1].model == "claude-sonnet-5"


async def test_the_active_version_runs_by_default():
    agent, provider, _ = build(script=["the v2 answer"])
    result = await agent.run("where is my order?")
    assert result.output == "the v2 answer"
    assert provider.requests[-1].model == "claude-opus-5"


async def test_each_version_sends_its_own_prompt_and_tools():
    agent, provider, _ = build(script=["done"])

    await agent.run("q", version="v1")
    v1 = provider.requests[-1]
    await agent.run("q", version="v2")
    v2 = provider.requests[-1]

    assert "Always cite the order" in v2.system
    assert "Always cite the order" not in v1.system
    assert {t.name for t in v1.tools} == {"order_status"}
    assert {t.name for t in v2.tools} >= {"order_status", "lookup"}


async def test_a_version_can_bring_its_own_guardrails():
    agent, _, _ = build()
    assert agent.guardrails is not None
    assert len(agent.guardrails) == 1
    assert agent.use("v1").guardrails is None


async def test_a_version_can_bring_its_own_sub_agents_and_budget():
    agent, provider, _ = build(versions={
        "v1": {"instructions": "plain"},
        "v2": {"instructions": "delegating",
               "subagents": [SubAgentSpec(name="researcher", description="Finds.")],
               "budget": Budget(max_steps=3)},
    }, version="v2")

    assert "researcher" in agent.subagents
    assert "delegate" in agent.tools
    assert agent.budget.max_steps == 3
    assert agent.use("v1").subagents == {}


async def test_streaming_a_specific_version():
    agent, provider, _ = build(script=["streamed v1"])
    provider.responses = ["streamed v1"]
    events = [e async for e in agent.stream("q", version="v1")]
    assert events[-1].type == "run_end"
    assert events[-1].data["result"].output == "streamed v1"


async def test_two_versions_can_be_evaluated_against_the_same_tasks():
    """The point of versions: prove on the same cases that v2 behaves differently.

    Here the model guesses instead of looking the order up. v1 accepts that;
    v2's guardrail does not — which is exactly the change v2 was made for.
    """
    from agent_harness import Evaluator, Expect, GoldenTask

    agent, provider, _ = build()
    provider.responses = ["order 4182 shipped, I think"]
    provider.loop = True

    suite = Evaluator([GoldenTask(id="mentions-order", input="where is it?",
                                  expect=Expect(contains=["order"]))],
                      concurrency=1)

    v1 = await suite.run(agent.use("v1"), label="v1")
    v2 = await suite.run(agent.use("v2"), label="v2")

    assert v1.score == 1.0       # the old version was happy with a guess
    assert v2.score == 0.0       # the new one insists the order is looked up
    assert "order_status" in v2.failed[0].detail

    comparison = v2.compare(v1)
    assert comparison.verdict == "regressed"
    assert comparison.regressed == ["mentions-order"]
