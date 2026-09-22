from __future__ import annotations

from agent_harness import Agent, Bench, FakeProvider, Harness, SubAgentSpec, tool, tool_call
from agent_harness.subagent import SubAgentFactory

MODEL = "claude-sonnet-5"


@tool
def lookup(q: str) -> str:
    """Look something up.

    Args:
        q: the query
    """
    return f"result for {q}"


@tool(tags=["write"])
def publish(text: str) -> str:
    """Publish text.

    Args:
        text: what to publish
    """
    return "published"


def parent_with(specs, script, **kw) -> Agent:
    provider = FakeProvider(script)
    harness = Harness.testing(provider)
    return Agent("manager", "You delegate.", provider=provider, model=MODEL,
                 harness=harness, tools=[lookup, publish], subagents=specs,
                 memory=True, **kw)


async def test_delegation_runs_the_sub_agent_and_hands_back_only_the_result():
    spec = SubAgentSpec(name="researcher", description="Finds things out.",
                        instructions="Find it.", tools=["lookup"])
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="researcher", task="find the total"),
        "the total is 42",          # the sub-agent's own answer
        "The total is 42.",         # the manager's answer
    ])
    result = await agent.run("what is the total?")
    assert result.output == "The total is 42."
    assert [c.agent for c in result.children] == ["researcher"]
    assert result.children[0].output == "the total is 42"


async def test_a_sub_agent_starts_clean_and_sees_only_its_brief():
    spec = SubAgentSpec(name="researcher", description="Finds things out.",
                        instructions="Find it.")
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="researcher", task="find X",
                  context="X lives in the ledger"),
        "found it", "done",
    ])
    await agent.run("a long conversation the sub-agent must not inherit")
    child_request = agent.harness.provider.requests[1]
    transcript = "\n".join(m.text for m in child_request.messages)
    assert "find X" in transcript
    assert "X lives in the ledger" in transcript
    assert "long conversation" not in transcript


async def test_the_tool_allowlist_is_enforced_when_the_sub_agent_is_built():
    spec = SubAgentSpec(name="reader", description="Reads only.", tools=["lookup"])
    parent = parent_with([spec], ["done"])
    child = parent.subagents["reader"]
    assert child.tools.names == ["lookup"]     # no publish, no delegate


async def test_sub_agents_inherit_the_parents_tools_by_default():
    spec = SubAgentSpec(name="worker", description="Does work.")
    parent = parent_with([spec], ["done"])
    child = parent.subagents["worker"]
    assert "lookup" in child.tools.names and "publish" in child.tools.names
    assert "delegate" not in child.tools.names      # delegation does not cascade
    assert "remember" not in child.tools.names      # nor does the parent's memory


async def test_a_sub_agent_failure_is_reported_not_swallowed():
    spec = SubAgentSpec(name="flaky", description="Fails.", max_steps=1)
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="flaky", task="do it"),
        tool_call("lookup", q="loop"),   # the child burns its only step on a tool
        "I could not get it done.",
    ])
    result = await agent.run("try")
    assert "could not" in result.output
    assert result.children[0].error is not None


async def test_delegating_to_an_unknown_sub_agent_is_answered_not_fatal():
    spec = SubAgentSpec(name="researcher", description="Finds things.")
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="ghost", task="x"),
        "There is no such sub-agent.",
    ])
    result = await agent.run("go")
    assert result.ok and "no such" in result.output


async def test_sub_agent_cost_rolls_up_to_the_parent():
    spec = SubAgentSpec(name="researcher", description="Finds things.")
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="researcher", task="find"),
        "found", "done",
    ])
    result = await agent.run("go")
    assert result.cost_usd >= result.usage.cost_usd
    assert result.children[0].usage.calls == 1
    digest = agent.memory.orchestrator.digest()
    assert "researcher" in digest


def test_the_bench_reuses_a_matching_sub_agent():
    bench = Bench.standard()
    assert bench.find("please validate this invoice against the rules").name == "validator"
    assert bench.find("extract the fields from this document").name == "document_extractor"
    assert bench.find("xyzzy plugh") is None      # nothing fits → the factory is next


async def test_the_factory_writes_a_new_spec_when_the_bench_has_nothing():
    provider = FakeProvider(['{"name": "ledger_reconciler", "description": "Reconciles.",'
                             ' "instructions": "Reconcile the ledger.",'
                             ' "tools": ["lookup"], "tier": "fast", "max_steps": 6,'
                             ' "workspace": "none"}'])
    harness = Harness.testing(provider)
    builder = Agent("factory", provider=provider, model=MODEL, harness=harness,
                    memory=False)
    factory = SubAgentFactory(builder)
    spec = await factory.create("reconcile the ledger", tools=["lookup", "publish"])
    assert spec.name == "ledger_reconciler"
    assert spec.tools == ["lookup"]       # least privilege: only what it asked for
    assert spec.origin == "factory"


async def test_the_factory_falls_back_when_the_model_returns_nothing_usable():
    provider = FakeProvider(["sorry, I cannot"])
    harness = Harness.testing(provider)
    builder = Agent("factory", provider=provider, model=MODEL, harness=harness,
                    memory=False)
    spec = await SubAgentFactory(builder).create("do a thing", tools=["lookup"])
    assert spec.origin == "factory" and spec.tools == ["lookup"]
    assert spec.instructions


async def test_an_output_contract_can_be_pinned_to_a_sub_agent_spec():
    spec = SubAgentSpec(
        name="extractor", description="Extracts totals.",
        output_schema={"type": "object", "properties": {"total": {"type": "number"}},
                       "required": ["total"]},
    )
    agent = parent_with([spec], [
        tool_call("delegate", agent_name="extractor", task="get the total"),
        '{"total": 42}',
        "The total is 42.",
    ])
    result = await agent.run("get it")
    assert result.children[0].data.total == 42


async def test_sub_agents_inherit_an_explicitly_configured_provider():
    """A provider passed to the parent must reach its sub-agents.

    Without this a sub-agent silently resolves its own backend from the model id
    and fails on a missing key — while the parent works fine.
    """
    from agent_harness import Harness

    provider = FakeProvider([
        tool_call("delegate", agent_name="worker", task="do it"),
        "the worker's answer",
        "all done",
    ])
    # A bare Harness: nothing on it points at the fake provider.
    agent = Agent("manager", provider=provider, model=MODEL, harness=Harness(),
                  memory=False,
                  subagents=[SubAgentSpec(name="worker", description="Works.")])
    assert agent.subagents["worker"].provider is provider
    result = await agent.run("go")
    assert result.children[0].output == "the worker's answer"
    assert result.output == "all done"
