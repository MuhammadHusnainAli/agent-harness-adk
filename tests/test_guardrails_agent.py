"""Per-agent guardrails: what an agent must do before it may call itself done."""

from __future__ import annotations

import pytest

from agent_harness import (
    Agent,
    AgentGuardrails,
    FakeProvider,
    ForbidTools,
    Harness,
    MaxCost,
    MaxSteps,
    MinLength,
    MustInclude,
    MustMatch,
    MustNotInclude,
    NoPlaceholders,
    RequireCitation,
    RequireJSON,
    RequireStructured,
    RequireTools,
    SubAgentSpec,
    tool,
    tool_call,
)
from agent_harness.errors import ConfigurationError
from agent_harness.guardrails import Custom
from agent_harness.guardrails.checks import CompletionContext

MODEL = "claude-sonnet-5"


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return "shipped Thursday"


@tool
def issue_refund(amount: float) -> str:
    """Refund a customer.

    Args:
        amount: how much
    """
    return f"refunded {amount}"


def build(script, guardrails=None, **kw):
    provider = FakeProvider(script)
    harness = Harness.testing(provider)
    agent = Agent("support", "Answer order questions.", provider=provider,
                  model=MODEL, harness=harness, tools=[order_status, issue_refund],
                  memory=False, guardrails=guardrails, **kw)
    return agent, provider, harness


def ctx(**kw) -> CompletionContext:
    return CompletionContext(**{"agent": "a", "output": "", **kw})


# --- the checks, on their own ---------------------------------------------------

def test_require_tools_objects_when_the_agent_guessed():
    check = RequireTools("order_status", "lookup")
    assert check(ctx(tools_called=["order_status", "lookup"])) is None
    violation = check(ctx(tools_called=["lookup"]))
    assert violation and "order_status" in violation.detail
    assert "call order_status" in violation.fix


def test_forbid_tools_objects_when_one_was_used():
    check = ForbidTools("issue_refund")
    assert check(ctx(tools_called=["order_status"])) is None
    assert check(ctx(tools_called=["issue_refund"])) is not None


def test_phrase_checks_are_case_insensitive_unless_asked():
    assert MustInclude("Order")(ctx(output="your order shipped")) is None
    assert MustInclude("Order", case_sensitive=True)(
        ctx(output="your order shipped")) is not None
    assert MustNotInclude("guarantee")(ctx(output="we guarantee it")) is not None


def test_pattern_length_and_placeholder_checks():
    assert MustMatch(r"\b\d{4}\b")(ctx(output="order 4182")) is None
    assert MustMatch(r"\b\d{4}\b", "needs an order number")(
        ctx(output="no number")).detail == "needs an order number"
    assert MinLength(20)(ctx(output="short")) is not None
    assert NoPlaceholders()(ctx(output="Dear [insert name], TODO")) is not None
    assert NoPlaceholders()(ctx(output="A complete answer.")) is None


def test_citation_json_and_structured_checks():
    assert RequireCitation()(ctx(output="see https://x.test/doc")) is None
    assert RequireCitation()(ctx(output="trust me")) is not None
    assert RequireJSON()(ctx(output='{"a": 1}')) is None
    assert RequireJSON()(ctx(output='```json\n{"a": 1}\n```')) is None
    assert RequireJSON()(ctx(output="not json")) is not None
    assert RequireStructured()(ctx(data={"a": 1})) is None
    assert RequireStructured()(ctx(data=None)) is not None


def test_budget_checks_are_soft_and_checked_at_the_end():
    assert MaxSteps(3)(ctx(steps=3)) is None
    assert MaxSteps(3)(ctx(steps=4)) is not None
    assert MaxCost(0.5)(ctx(cost_usd=0.75)) is not None


def test_a_custom_check_can_explain_itself():
    check = Custom(lambda c: (False, "the total is missing"), name="has_total")
    violation = check(ctx(output="x"))
    assert violation.check == "has_total" and violation.detail == "the total is missing"
    assert Custom(lambda c: True)(ctx()) is None


def test_a_check_that_itself_explodes_is_reported_not_raised():
    def broken(context):
        raise RuntimeError("the check is buggy")

    rails = AgentGuardrails(Custom(broken, name="broken"))
    violations = rails.check(ctx(output="anything"))
    assert violations and "the check itself failed" in violations[0].detail


# --- assembling them -------------------------------------------------------------

def test_keyword_form_builds_the_same_checks():
    rails = AgentGuardrails(require_tools=["order_status"],
                            forbid_tools=["issue_refund"],
                            must_include=["order"], no_placeholders=True,
                            max_cost_usd=0.25, min_output_chars=10)
    # forbid_tools is enforced before the call, so it is not also a completion
    # check — five checks, not six.
    assert len(rails) == 5
    assert rails.tool_allowed("issue_refund") == (
        False, "issue_refund is not permitted for this agent")
    assert not any(c.name == "forbid_tools" for c in rails.checks)


def test_an_allowlist_refuses_everything_else():
    rails = AgentGuardrails(allow_tools=["order_status"])
    assert rails.tool_allowed("order_status")[0] is True
    permitted, reason = rails.tool_allowed("issue_refund")
    assert not permitted and "allowlist" in reason


def test_bad_configuration_is_refused():
    with pytest.raises(ConfigurationError, match="retry, fail or warn"):
        AgentGuardrails(on_violation="explode")
    with pytest.raises(ConfigurationError, match="negative"):
        AgentGuardrails(max_retries=-1)
    with pytest.raises(TypeError):
        AgentGuardrails("not a check")


def test_the_feedback_reads_like_something_an_agent_can_act_on():
    rails = AgentGuardrails(require_tools=["order_status"], must_include=["order"])
    violations = rails.check(ctx(output="dunno", tools_called=[]))
    text = rails.feedback(violations)
    assert "does not meet" in text
    assert "call order_status" in text
    assert text.rstrip().endswith("Put it right and answer again.")


# --- in a run --------------------------------------------------------------------

async def test_a_forbidden_tool_never_actually_runs():
    agent, _, harness = build(
        [tool_call("issue_refund", amount=60.0), "I am not allowed to refund."],
        guardrails=AgentGuardrails(forbid_tools=["issue_refund"]),
    )
    result = await agent.run("refund 60")

    assert result.ok and "not allowed" in result.output
    blocked = [b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result"]
    assert "Not permitted" in blocked[0]
    assert [e.target for e in harness.audit.denials()] == ["issue_refund"]


async def test_an_unmet_requirement_sends_the_agent_back_round():
    agent, provider, _ = build(
        ["Your order shipped Thursday.",                     # answered without looking
         tool_call("order_status", order_id="4182"),         # told off, looks it up
         "Order 4182 shipped Thursday."],
        guardrails=AgentGuardrails(require_tools=["order_status"]),
    )
    result = await agent.run("Where is order 4182?")

    assert result.ok
    assert result.output == "Order 4182 shipped Thursday."
    assert result.violations == []          # the final answer met them
    nudge = [m.text for m in result.messages
             if m.role == "user" and "does not meet" in m.text]
    assert nudge and "call order_status" in nudge[0]


async def test_it_gives_up_after_the_retries_are_used():
    agent, _, _ = build(
        ["nope", "still nope", "nope again", "and again"],
        guardrails=AgentGuardrails(must_include=["order"], on_violation="retry",
                                   max_retries=2),
    )
    result = await agent.run("Where is my order?")

    assert not result.ok
    assert "GuardrailTripped" in result.error
    assert result.violations and "does not mention" in result.violations[0]


async def test_fail_stops_at_the_first_unmet_requirement():
    agent, _, _ = build(["an answer with no sources"],
                        guardrails=AgentGuardrails(require_citation=True,
                                                   on_violation="fail"))
    result = await agent.run("what happened?")
    assert not result.ok and "cites nothing" in result.error
    assert result.steps == 1            # no retry was attempted


async def test_warn_records_the_problem_and_still_delivers():
    agent, _, harness = build(["an answer with no sources"],
                              guardrails=AgentGuardrails(require_citation=True,
                                                         on_violation="warn"))
    result = await agent.run("what happened?")

    assert result.ok
    assert result.output == "an answer with no sources"
    assert result.violations and "cites nothing" in result.violations[0]
    assert any(e.kind == "guardrail" for e in harness.journal.entries)
    assert harness.audit.verify()[0]


async def test_several_unmet_requirements_are_reported_together():
    agent, _, _ = build(["TODO", "A proper answer about the order, source: the ledger."],
                        guardrails=AgentGuardrails(
                            must_include=["order"], no_placeholders=True,
                            require_citation=True, min_output_chars=20))
    result = await agent.run("what happened?")

    assert result.ok
    nudge = [m.text for m in result.messages
             if m.role == "user" and "does not meet" in m.text][0]
    assert nudge.count("\n- ") == 4      # all four, in one message


async def test_guardrails_see_what_sub_agents_called_too():
    provider = FakeProvider([
        tool_call("delegate", agent_name="looker", task="look it up"),
        "shipped Thursday",
        "Your order shipped Thursday.",
    ])
    harness = Harness.testing(provider)
    agent = Agent("manager", provider=provider, model=MODEL, harness=harness,
                  tools=[order_status], memory=False,
                  guardrails=AgentGuardrails(require_tools=["order_status"]),
                  subagents=[SubAgentSpec(name="looker", description="Looks up.",
                                          tools=["order_status"])])
    # The sub-agent does the lookup; the parent's requirement is still satisfied.
    provider.responses.insert(1, tool_call("order_status", order_id="4182"))
    result = await agent.run("Where is order 4182?")
    assert result.ok and result.violations == []


async def test_a_spec_can_carry_its_own_guardrails():
    provider = FakeProvider([
        tool_call("delegate", agent_name="reader", task="read it"),
        "a bare answer",                 # fails the sub-agent's own guardrails
        "A proper answer, source: the ledger.",
        "Done.",
    ])
    harness = Harness.testing(provider)
    parent = Agent("manager", provider=provider, model=MODEL, harness=harness,
                   tools=[order_status], memory=False,
                   subagents=[SubAgentSpec(
                       name="reader", description="Reads.",
                       guardrails={"require_citation": True, "max_retries": 1})])

    child = parent.subagents["reader"]
    assert child.guardrails is not None and len(child.guardrails) == 1

    result = await parent.run("go")
    assert result.ok
    assert result.children[0].output == "A proper answer, source: the ledger."


async def test_an_agent_can_carry_its_own_content_rules():
    from agent_harness import Guardrails
    from agent_harness.guardrails import Rule

    strict = Guardrails([Rule("no_ids", r"\b\d{4}\b", "redact", "[id]")],
                        secrets=False, injection=False)
    agent, _, harness = build(["order 4182 shipped"],
                              guardrails=AgentGuardrails(content=strict))
    result = await agent.run("where is it?")

    assert "4182" not in result.output and "[id]" in result.output
    assert agent.content_guardrails is strict
    assert harness.guardrails is not strict      # the harness default is untouched


def test_an_agent_without_guardrails_falls_back_to_the_harness():
    agent, _, harness = build(["x"])
    assert agent.guardrails is None
    assert agent.content_guardrails is harness.guardrails


def test_the_report_counts_what_tripped():
    rails = AgentGuardrails(must_include=["order"], require_citation=True)
    rails.check(ctx(output="nothing here"))
    report = rails.report()
    assert report["checks"] == 2
    assert report["violations"] == {"must_include": 1, "require_citation": 1}
    assert report["on_violation"] == "retry"


def test_checks_can_be_passed_as_a_bare_list():
    agent, _, _ = build(["x"], guardrails=[RequireTools("order_status"),
                                           NoPlaceholders()])
    assert isinstance(agent.guardrails, AgentGuardrails)
    assert len(agent.guardrails) == 2
