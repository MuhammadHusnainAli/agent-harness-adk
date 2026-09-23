"""Declaring a whole agent tree in a YAML or JSON file, then building it."""

from __future__ import annotations

import json

import pytest

from agent_harness import Blueprint, FakeProvider, Harness, tool
from agent_harness.errors import ConfigurationError

YAML = """
version: v2

defaults:
  model: claude-sonnet-5
  memory: false

prompts:
  house_style: Answer in plain sentences and cite the order.

guardrails:
  strict:
    require_tools: [order_status]
    no_placeholders: true

subagents:
  researcher:
    description: Finds things out, read-only.
    instructions: "{house_style} Cite every claim."
    tools: [lookup]
    tier: fast
    budget:
      max_input_tokens: 10000
      max_output_tokens: 2000
    guardrails:
      require_citation: true

agents:
  support:
    description: Answers order questions.
    instructions: "{house_style}"
    tools: [order_status, lookup]
    subagents: [researcher]
    guardrails: strict
    max_steps: 6
    versions:
      v1:
        instructions: Answer order questions.
        tools: [order_status]
        model: claude-haiku-4-5
      v2:
        instructions: "{house_style}"
  triage:
    instructions: Sort the ticket.
    tools: [order_status]
    model: claude-haiku-4-5
"""


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return "shipped"


@tool
def lookup(topic: str) -> str:
    """Look something up.

    Args:
        topic: what to look up
    """
    return topic


TOOLS = [order_status, lookup]


def harness() -> Harness:
    return Harness.testing(FakeProvider(["ok"], loop=True))


# --- loading -----------------------------------------------------------------

def test_a_yaml_file_loads(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text(YAML)
    blueprint = Blueprint.from_file(path)

    assert blueprint.version == "v2"
    assert blueprint.names == ["support", "triage"]
    assert "researcher" in blueprint.subagents
    assert blueprint.path == str(path)


def test_the_same_file_as_json(tmp_path):
    import yaml as pyyaml

    path = tmp_path / "agents.json"
    path.write_text(json.dumps(pyyaml.safe_load(YAML)))
    blueprint = Blueprint.from_file(path)
    assert blueprint.names == ["support", "triage"]
    assert blueprint.subagents["researcher"].tier == "fast"


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(ConfigurationError, match="no blueprint at"):
        Blueprint.from_file(tmp_path / "nope.yaml")


def test_a_file_that_is_not_a_mapping_is_refused():
    with pytest.raises(ConfigurationError, match="mapping at the top level"):
        Blueprint.from_text("- just\n- a list\n")


def test_it_round_trips_through_a_file(tmp_path):
    blueprint = Blueprint.from_text(YAML)
    for suffix in (".yaml", ".json"):
        path = blueprint.save(tmp_path / f"out{suffix}")
        again = Blueprint.from_file(path)
        assert again.names == blueprint.names
        assert again.subagents["researcher"].tools == ["lookup"]


# --- building -----------------------------------------------------------------

def test_an_agent_is_built_from_its_declaration():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    assert agent.name == "support"
    assert agent.description == "Answers order questions."
    assert agent.model == "claude-sonnet-5"        # from defaults
    assert agent.max_steps == 6
    assert set(agent.tools.names) >= {"order_status", "lookup"}


def test_prompt_references_are_substituted():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    assert agent.instructions == "Answer in plain sentences and cite the order."
    assert "{house_style}" not in agent.instructions


def test_sub_agents_are_looked_up_and_attached():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    assert "researcher" in agent.subagents
    researcher = agent.subagents["researcher"]
    assert researcher.tools.names == ["lookup"]
    assert "Cite every claim" in researcher.instructions
    assert "Answer in plain sentences" in researcher.instructions   # rendered too


def test_a_sub_agent_budget_and_guardrails_come_through():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    researcher = agent.subagents["researcher"]
    assert researcher.budget.max_input_tokens == 10_000
    assert researcher.budget.max_output_tokens == 2_000
    assert researcher.guardrails is not None and len(researcher.guardrails) == 1


def test_named_guardrails_are_resolved():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    assert agent.guardrails is not None
    assert agent.guardrails.checks                    # require_tools + no_placeholders
    assert len(agent.guardrails) == 2


def test_guardrails_that_are_not_declared_are_refused():
    blueprint = Blueprint.from_text(YAML)
    blueprint.agents["support"].guardrails = "nonexistent"
    with pytest.raises(ConfigurationError, match="are not defined"):
        blueprint.build("support", tools=TOOLS, harness=harness())


def test_versions_declared_in_the_file_work():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness())
    assert agent.version == "v2"
    assert agent.available_versions == ["v1", "v2"]

    old = agent.use("v1")
    assert old.instructions == "Answer order questions."
    assert old.model == "claude-haiku-4-5"
    # v1 narrows the tools but says nothing about sub-agents, so it keeps them —
    # and `delegate` comes with them.
    assert "order_status" in old.tools.names
    assert "lookup" not in old.tools.names
    assert "delegate" in old.tools.names


def test_defaults_lose_to_anything_the_agent_states():
    agent = Blueprint.from_text(YAML).build("triage", tools=TOOLS,
                                            harness=harness())
    assert agent.model == "claude-haiku-4-5"          # its own, not the default
    assert agent.memory is None                       # memory: false from defaults


def test_the_caller_can_override_anything():
    agent = Blueprint.from_text(YAML).build("support", tools=TOOLS,
                                            harness=harness(),
                                            model="gpt-4.1", max_steps=99)
    assert agent.model == "gpt-4.1" and agent.max_steps == 99


def test_building_them_all_shares_one_harness():
    shared = harness()
    agents = Blueprint.from_text(YAML).build_all(tools=TOOLS, harness=shared)
    assert set(agents) == {"support", "triage"}
    assert all(a.harness is shared for a in agents.values())


def test_an_undeclared_agent_is_refused():
    with pytest.raises(ConfigurationError, match="no agent 'ghost'"):
        Blueprint.from_text(YAML).build("ghost", tools=TOOLS)


def test_a_missing_tool_explains_how_to_supply_it():
    with pytest.raises(ConfigurationError, match="were not supplied"):
        Blueprint.from_text(YAML).build("support", tools=[order_status],
                                        harness=harness())


def test_an_undeclared_sub_agent_is_refused():
    blueprint = Blueprint.from_text(YAML)
    blueprint.agents["support"].subagents = ["ghost"]
    with pytest.raises(ConfigurationError, match="sub-agent 'ghost' is not declared"):
        blueprint.build("support", tools=TOOLS, harness=harness())


def test_a_tool_can_be_named_as_an_import_path():
    blueprint = Blueprint.from_text(YAML)
    blueprint.agents["triage"].tools = ["agent_harness.toolkits:calculate"]
    agent = blueprint.build("triage", tools=[], harness=harness())
    assert agent.tools.names == ["calculate"]


def test_a_bad_import_path_says_what_failed():
    blueprint = Blueprint.from_text(YAML)
    blueprint.agents["triage"].tools = ["nope.module:thing"]
    with pytest.raises(ConfigurationError, match="cannot import"):
        blueprint.build("triage", tools=[], harness=harness())


def test_a_declared_memory_backend_is_built():
    blueprint = Blueprint.from_text(YAML + '\nmemory: "sqlite://:memory:"\n')
    store = blueprint.memory_store()
    assert type(store).__name__ == "SQLiteMemory"

    with_options = Blueprint.from_dict(
        {"memory": {"url": "sqlite://:memory:", "table": "custom"}})
    assert with_options.memory_store().table == "custom"

    assert Blueprint.from_text(YAML).memory_store() is None


async def test_an_agent_built_from_a_file_actually_runs():
    provider = FakeProvider(["Order 4182 shipped Thursday."])
    shared = Harness.testing(provider)
    agent = Blueprint.from_text(YAML).build(
        "triage", tools=TOOLS, harness=shared, provider=provider)

    result = await agent.run("where is order 4182?")
    assert result.output == "Order 4182 shipped Thursday."
    assert provider.requests[0].model == "claude-haiku-4-5"
