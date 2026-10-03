"""Handoff: another agent takes over the conversation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_harness import (
    Agent,
    AgentGuardrails,
    Blueprint,
    ConfigurationError,
    FakeProvider,
    Handoff,
    Harness,
    Message,
    Trace,
    tool,
    tool_call,
)
from agent_harness.governance import Governance
from agent_harness.types import ThinkingBlock, ToolResultBlock, ToolUseBlock
from agent_harness.voice import VoiceAgent


@tool
def lookup(order: str) -> str:
    """Look an order up.

    Args:
        order: the order number.
    """
    return f"order {order}: charged twice, 40 EUR each"


def to(name: str, reason: str = "") -> ToolUseBlock:
    return tool_call("handoff", agent_name=name, reason=reason)


def desk(triage: list, billing: list, *, harness: Harness | None = None,
         handoff: dict[str, Any] | None = None, mode: str | None = "chat",
         **kw: Any) -> tuple[Agent, Agent]:
    """A front desk and the specialist it hands billing questions to, each
    with its own script."""
    harness = harness or Harness.testing()
    specialist = Agent("billing", "Handle refunds.", description="Refunds and invoices.",
                       provider=FakeProvider(billing), harness=harness, memory=False,
                       tools=[lookup])
    front = Agent("triage", "Work out what the customer needs.", mode=mode,
                  provider=FakeProvider(triage), harness=harness, memory=False,
                  handoffs=[Handoff(specialist, **(handoff or {}))], **kw)
    return front, specialist


def results_of(messages: list[Message]) -> list[ToolResultBlock]:
    return [b for m in messages for b in m.content if isinstance(b, ToolResultBlock)]


# ----------------------------------------------------------------------
# the handoff itself
# ----------------------------------------------------------------------
async def test_the_other_agent_takes_over_and_answers():
    triage, billing = desk([to("billing", "a refund")], ["Refunded 40 EUR."])

    result = await triage.run("I was charged twice.")

    assert result.ok and result.output == "Refunded 40 EUR."
    assert result.agent == "billing" and result.active_agent == "billing"
    assert [h.line() for h in result.handoffs] == ["triage → billing: a refund"]
    assert result.stop_reason == "end_turn"
    # billing was given the conversation, and told in it how it got there.
    seen = billing.provider.requests[0].messages
    assert seen[0].text == "I was charged twice."
    assert "triage handed this conversation to billing: a refund" in (
        results_of(seen)[-1].content)
    # Its own instructions, not the front desk's.
    assert "Handle refunds." in billing.provider.requests[0].system
    assert len(triage.provider.requests) == 1


async def test_the_tool_says_who_can_be_handed_to():
    triage, _ = desk([], [])
    schema = triage.tools.get("handoff").parameters["properties"]["agent_name"]
    assert schema["enum"] == ["billing"]
    assert "Refunds and invoices." in schema["description"]
    assert "handoff" not in Agent("alone", memory=False).tools


async def test_what_both_agents_did_is_on_the_one_result():
    triage, billing = desk(
        [to("billing")], [tool_call("lookup", order="4182"), "Refunded."])

    result = await triage.run("Refund order 4182.")

    assert [(c.agent, c.name) for c in result.tool_calls] == [
        ("triage", "handoff"), ("billing", "lookup")]
    assert result.steps == 3
    assert result.usage.calls == 3
    assert result.messages[-1].text == "Refunded."


async def test_a_stream_is_one_run_with_a_handoff_in_it():
    triage, _ = desk([to("billing", "refund")], ["Refunded."])

    events = [e async for e in triage.stream("I was charged twice.")]

    kinds = [e.type for e in events]
    assert kinds.count("run_start") == 1 and kinds.count("run_end") == 1
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    hop = next(e for e in events if e.type == "handoff")
    assert (hop.agent, hop.text, hop.data["reason"]) == ("triage", "billing", "refund")
    said = [e for e in events if e.type == "text"]
    assert {e.agent for e in said} == {"billing"}
    assert events[-1].data["result"].agent == "billing"


# ----------------------------------------------------------------------
# who has the conversation on the next turn
# ----------------------------------------------------------------------
async def test_whoever_was_handed_the_conversation_keeps_it():
    triage, billing = desk([to("billing")], ["Refunded.", "It was sent today."])

    await triage.run("I was charged twice.")
    again = await triage.run("And the invoice?")

    assert again.agent == "billing" and again.output == "It was sent today."
    assert not again.handoffs and again.active_agent == "billing"
    assert len(triage.provider.requests) == 1          # not asked a second time
    texts = [m.text for m in billing.provider.requests[-1].messages if m.text]
    assert texts == ["I was charged twice.", "Refunded.", "And the invoice?"]

    triage.new_session()
    triage.provider.queue("Hello again.")
    assert (await triage.run("Hi")).agent == "triage"


async def test_another_process_finds_who_has_it_from_the_session(tmp_path):
    def process(triage: list, billing: list) -> Agent:
        harness = Harness.local(tmp_path / "state", trace=False)
        return desk(triage, billing, harness=harness, mode=None)[0]

    first = await process([to("billing")], ["Refunded."]).run("I was charged twice.")

    front = process(["never asked"], ["It was sent today."])
    second = await front.run("And the invoice?", session=first.session_id)

    assert second.agent == "billing" and second.output == "It was sent today."
    assert not front.provider.requests
    kept = await front.harness.sessions.load(first.session_id)
    assert kept.metadata["handoff"]["active"] == "billing"
    assert kept.metadata["handoff"]["entry"] == "triage"
    assert [t["target"] for t in kept.metadata["handoff"]["trail"]] == ["billing"]
    assert [m.text for m in kept.messages if m.text] == [
        "I was charged twice.", "Refunded.", "And the invoice?", "It was sent today."]


async def test_a_handoff_for_one_turn_gives_the_conversation_back():
    triage, billing = desk([to("billing"), "Anything else?"], ["Refunded."],
                           handoff={"sticky": False})

    first = await triage.run("I was charged twice.")
    second = await triage.run("No, thanks.")

    assert (first.agent, first.active_agent) == ("billing", "triage")
    assert second.agent == "triage" and second.output == "Anything else?"


async def test_it_can_be_handed_back():
    triage, billing = desk([to("billing"), "What else can I do?"],
                           ["Refunded.", to("triage", "not a billing question")])
    billing.add_handoff(triage)

    await triage.run("I was charged twice.")
    back = await triage.run("What are your opening hours?")

    assert back.agent == "triage" and back.output == "What else can I do?"
    assert [h.line() for h in back.handoffs] == [
        "billing → triage: not a billing question"]
    assert back.active_agent == "triage"
    assert triage._thread.metadata["handoff"]["active"] == "triage"
    assert len(triage._thread.metadata["handoff"]["trail"]) == 2


async def test_an_agent_reached_through_another_is_still_found():
    harness = Harness.testing()
    legal = Agent("legal", provider=FakeProvider(["Clause 4.", "Clause 5."]),
                  harness=harness, memory=False)
    billing = Agent("billing", provider=FakeProvider([to("legal")]), harness=harness,
                    memory=False, handoffs=[legal])
    triage = Agent("triage", mode="chat", provider=FakeProvider([to("billing")]),
                   harness=harness, memory=False, handoffs=[billing])

    first = await triage.run("Can I dispute this charge?")
    second = await triage.run("Which clause?")

    assert [h.target for h in first.handoffs] == ["billing", "legal"]
    assert first.output == "Clause 4." and second.output == "Clause 5."
    assert second.agent == "legal"
    assert triage.handoff_agent("legal") is legal
    assert triage.handoff_agent("nobody") is None


async def test_an_agent_that_is_gone_gives_the_conversation_back(tmp_path):
    harness = Harness.local(tmp_path / "state", trace=False)
    triage, _ = desk([to("billing")], ["Refunded."], harness=harness, mode=None)
    first = await triage.run("I was charged twice.")

    # A later deployment: the front desk hands to somebody else now.
    harness = Harness.local(tmp_path / "state", trace=False)
    other = Agent("orders", provider=FakeProvider([]), harness=harness, memory=False)
    front = Agent("triage", provider=FakeProvider(["I can help with that."]),
                  harness=harness, memory=False, handoffs=[other])
    result = await front.run("And the invoice?", session=first.session_id)

    assert result.agent == "triage" and result.output == "I can help with that."
    assert result.active_agent == "triage"
    assert any(e.action == "handoff" and e.decision == "reclaim"
               for e in harness.audit.entries)
    kept = await harness.sessions.load(first.session_id)
    assert kept.metadata["handoff"]["active"] == "triage"


async def test_a_run_that_starts_clean_starts_with_the_agent_you_called():
    """No mode, no session: nothing is being continued, so nobody is holding it
    — and the specialist sees this run, not the ones before it."""
    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider(["Refunded.", "Sent."]),
                    harness=harness, memory=False, tools=[lookup])
    triage = Agent("triage", provider=FakeProvider([to("billing"), to("billing")]),
                   harness=harness, handoffs=[billing])

    first = await triage.run("I was charged twice.")
    second = await triage.run("Send the invoice.")

    assert first.session_id == second.session_id
    assert len(triage.provider.requests) == 2
    assert [m.text for m in billing.provider.requests[-1].messages if m.text] == [
        "Send the invoice."]
    # The record still holds both runs, whole.
    kept = await harness.sessions.load(second.session_id)
    assert [m.text for m in kept.messages if m.text] == [
        "I was charged twice.", "Refunded.", "Send the invoice.", "Sent."]


async def test_a_one_off_run_hands_off_and_says_who_has_it():
    triage, billing = desk([to("billing")], ["Refunded."], mode=None)
    history = [Message.user("Hello"), Message.assistant("Hi, how can I help?")]

    result = await triage.run("I was charged twice.", messages=history)

    assert result.agent == "billing" and result.active_agent == "billing"
    assert [m.text for m in result.messages if m.text][:2] == [
        "Hello", "Hi, how can I help?"]
    assert len(history) == 2


# ----------------------------------------------------------------------
# what the agent taking over sees
# ----------------------------------------------------------------------
async def test_text_history_leaves_out_what_was_looked_up():
    triage, billing = desk(
        [tool_call("lookup", order="4182"), [Message.assistant("One moment.").content[0],
                                             to("billing", "a refund")]],
        ["Refunded."], handoff={"history": "text"}, tools=[lookup])

    result = await triage.run("Refund order 4182.")

    seen = billing.provider.requests[0].messages
    assert not any(isinstance(b, (ToolUseBlock, ToolResultBlock))
                   for m in seen for b in m.content)
    assert [m.role for m in seen] == ["user", "assistant", "user"]
    assert seen[1].text == "One moment."
    assert seen[-1].text.startswith("[triage handed this conversation to billing: "
                                    "a refund.")
    # And that is the conversation from here on.
    assert not results_of(result.messages)
    assert [m.text for m in triage._thread.messages] == [m.text for m in result.messages]


async def test_fresh_history_is_only_what_the_user_last_said():
    triage, billing = desk(["Hello!", to("billing")], ["Refunded."],
                           handoff={"history": "fresh"})

    await triage.run("Hi")
    await triage.run("I was charged twice.")

    seen = billing.provider.requests[0].messages
    assert [m.text for m in seen][0] == "I was charged twice."
    assert len(seen) == 2 and seen[1].role == "user"


async def test_a_filter_of_your_own_decides():
    def last_two(messages: list[Message]) -> list[Message]:
        return [m for m in messages if m.text][-2:]

    triage, billing = desk(["Hello!", to("billing")], ["Refunded."],
                           handoff={"history": last_two})
    await triage.run("Hi")
    await triage.run("I was charged twice.")

    seen = billing.provider.requests[0].messages
    assert [m.text for m in seen] == ["Hello!", "I was charged twice."]

    with pytest.raises(ConfigurationError, match="must return a list of Message"):
        Handoff(billing, history=lambda messages: "nope").view([], source=triage)


async def test_an_agent_with_no_tools_is_not_sent_tool_calls():
    harness = Harness.testing()
    plain = Agent("plain", provider=FakeProvider(["Here you go."]), harness=harness,
                  memory=False)
    triage = Agent("triage", provider=FakeProvider([to("plain")]), harness=harness,
                   memory=False, handoffs=[plain])

    result = await triage.run("Help")

    assert result.output == "Here you go."
    request = plain.provider.requests[0]
    assert not request.tools
    assert not any(isinstance(b, (ToolUseBlock, ToolResultBlock))
                   for m in request.messages for b in m.content)
    assert request.messages[-1].role == "user"


async def test_reasoning_does_not_follow_the_conversation_to_another_model():
    harness = Harness.testing()
    thought = [ThinkingBlock(thinking="a billing matter", signature="sig"),
               to("billing")]
    billing = Agent("billing", model="gpt-4.1", provider=FakeProvider(["Refunded."]),
                    harness=harness, memory=False, tools=[lookup])
    same = Agent("same", model="claude-opus-5", provider=FakeProvider(["Done."]),
                 harness=harness, memory=False, tools=[lookup])
    triage = Agent("triage", model="claude-opus-5", harness=harness, memory=False,
                   provider=FakeProvider([thought, [thought[0], to("same")]]),
                   handoffs=[billing, same])

    await triage.run("I was charged twice.")
    await triage.run("Again.")

    def thinking(agent: Agent) -> int:
        return sum(isinstance(b, ThinkingBlock)
                   for m in agent.provider.requests[0].messages for b in m.content)

    assert thinking(billing) == 0 and thinking(same) == 1
    # The handoff itself is still in what the other model was given.
    assert results_of(billing.provider.requests[0].messages)


# ----------------------------------------------------------------------
# when a handoff does not happen
# ----------------------------------------------------------------------
async def test_a_conversation_cannot_be_passed_round_for_ever():
    harness = Harness.testing()
    b = Agent("b", provider=FakeProvider([to("a"), to("a"), "I will take it."]),
              harness=harness, memory=False)
    a = Agent("a", provider=FakeProvider([to("b"), to("b")]),
              harness=harness, memory=False, handoffs=[b], max_handoffs=3)
    b.add_handoff(a)

    result = await a.run("Help")

    assert [h.line() for h in result.handoffs] == ["a → b", "b → a", "a → b"]
    assert result.ok and result.agent == "b"
    refused = [r for r in results_of(result.messages) if r.is_error]
    assert len(refused) == 1 and "which is the limit (3)" in refused[0].content


async def test_no_handoffs_at_all_means_the_agent_answers():
    triage, billing = desk([to("billing"), "I will handle it."], [], max_handoffs=0)
    result = await triage.run("Help")
    assert result.agent == "triage" and result.output == "I will handle it."
    assert not result.handoffs and not billing.provider.requests

    with pytest.raises(ConfigurationError, match="max_handoffs"):
        Agent("x", max_handoffs=-1)


async def test_only_one_of_two_handoffs_asked_for_at_once_is_granted():
    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider(["Refunded."]), harness=harness,
                    memory=False, tools=[lookup])
    orders = Agent("orders", provider=FakeProvider(["Shipped."]), harness=harness,
                   memory=False)
    triage = Agent("triage", provider=FakeProvider([[to("billing"), to("orders")]]),
                   harness=harness, memory=False, handoffs=[billing, orders],
                   parallel_tools=False)

    result = await triage.run("Refund it and tell me where it is.")

    assert result.agent == "billing" and len(result.handoffs) == 1
    assert not orders.provider.requests
    refused = [r for r in results_of(result.messages) if r.is_error]
    assert "already being handed to billing" in refused[0].content


async def test_a_hook_can_keep_the_conversation_where_it_is():
    triage, billing = desk([to("billing"), "Let me help you myself."], [])

    @triage.hooks.on("handoff")
    def office_hours(ctx):
        assert (ctx.agent, ctx.data["source"]) == ("billing", "triage")
        ctx.block("billing is closed")

    result = await triage.run("I was charged twice.")

    assert result.agent == "triage" and result.output == "Let me help you myself."
    assert not result.handoffs and result.active_agent == "triage"
    assert "billing is closed" in results_of(result.messages)[0].content
    assert "handoff" not in triage._thread.metadata
    denied = [e for e in triage.harness.audit.entries
              if e.action == "handoff" and e.decision == "deny"]
    assert denied and denied[0].target == "billing"


async def test_on_handoff_is_told_and_can_refuse():
    told: list[str] = []

    async def note(record):
        told.append(record.line())

    triage, _ = desk([to("billing", "refund")], ["Refunded."],
                     handoff={"on_handoff": note})
    assert (await triage.run("Help")).agent == "billing"
    assert told == ["triage → billing: refund"]

    def refuse(record):
        raise RuntimeError("the queue is full")

    triage, _ = desk([to("billing"), "Staying with you."], [],
                     handoff={"on_handoff": refuse})
    result = await triage.run("Help")
    assert result.agent == "triage" and not result.handoffs
    assert "the queue is full" in results_of(result.messages)[0].content


async def test_the_permission_gate_and_guardrails_cover_the_handoff_tool():
    triage, billing = desk(
        [to("billing"), "Answering myself."], [],
        guardrails=AgentGuardrails(forbid_tools=["handoff"]))
    result = await triage.run("Help")
    assert result.agent == "triage" and not billing.provider.requests
    assert results_of(result.messages)[0].content.startswith("Not permitted")


async def test_an_unknown_name_and_a_tool_called_outside_a_run_are_refused():
    triage, _ = desk([to("nobody"), "Sorry."], [])
    result = await triage.run("Help")
    assert "no agent named 'nobody'" in results_of(result.messages)[0].content
    assert result.agent == "triage"

    outcome = await triage.call_tool("handoff", {"agent_name": "billing"})
    assert outcome.is_error and "no conversation to hand over" in outcome.content


async def test_an_agent_that_cannot_start_does_not_keep_the_conversation():
    triage, billing = desk([to("billing"), "I can take that."], [])

    @billing.hooks.on("run_start")
    def closed(ctx):
        if ctx.agent == "billing":
            ctx.block("billing is offline")

    failed = await triage.run("I was charged twice.")
    assert failed.agent == "billing" and "billing is offline" in failed.error
    assert failed.active_agent == "triage"

    again = await triage.run("Hello?")
    assert again.agent == "triage" and again.output == "I can take that."


async def test_a_conversation_is_not_handed_to_an_agent_acting_for_someone_else():
    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider([]), harness=harness,
                    trace=Trace(user_id="bob"))
    triage = Agent("triage", mode="chat", harness=harness, trace=Trace(user_id="ada"),
                   provider=FakeProvider([to("billing"), "I will help."]),
                   handoffs=[billing])

    result = await triage.run("I was charged twice.")

    assert result.agent == "triage" and not result.handoffs
    assert "acting for a different user" in results_of(result.messages)[0].content


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def test_what_cannot_be_a_handoff_is_refused_when_the_agent_is_built():
    agent = Agent("solo", memory=False)
    with pytest.raises(ConfigurationError, match="cannot hand the conversation to itself"):
        agent.add_handoff(agent)
    with pytest.raises(ConfigurationError, match="needs an Agent"):
        Agent("a", handoffs=["billing"])
    with pytest.raises(ConfigurationError, match="history must be one of"):
        Handoff(agent, history="everything")

    @tool
    def handoff(note: str) -> str:
        """Write a shift handoff note."""
        return note

    with pytest.raises(ConfigurationError, match="already has a tool called 'handoff'"):
        Agent("b", tools=[handoff], handoffs=[agent])


def test_helpers_are_not_given_the_conversation_to_hand_away():
    harness = Harness.testing()
    billing = Agent("billing", harness=harness, memory=False)
    triage = Agent("triage", harness=harness, memory=False, tools=[lookup],
                   handoffs=[billing],
                   subagents=[{"name": "reader", "description": "Reads orders."}])
    assert triage.subagents["reader"].tools.names == ["lookup"]


async def test_a_sub_agent_can_hand_its_task_to_a_peer():
    harness = Harness.testing()
    senior = Agent("senior", provider=FakeProvider(["The answer is 42."]),
                   harness=harness, memory=False)
    junior = Agent("junior", provider=FakeProvider([to("senior", "beyond me")]),
                   harness=harness, memory=False, handoffs=[senior])
    boss = Agent("boss", harness=harness, memory=False, subagents=[junior],
                 provider=FakeProvider(
                     [tool_call("delegate", agent_name="junior", task="Work it out."),
                      "It is 42."]))

    result = await boss.run("What is the answer?")

    assert result.output == "It is 42." and result.agent == "boss"
    assert result.children[0].agent == "senior"
    assert results_of(result.messages)[0].content == "The answer is 42."


async def test_every_version_of_an_agent_hands_off_to_the_same_places():
    harness = Harness.testing(FakeProvider([to("billing"), "Refunded."]))
    billing = Agent("billing", harness=harness, memory=False)
    orders = Agent("orders", harness=harness, memory=False)
    triage = Agent("triage", harness=harness, memory=False, version="v1",
                   versions={"v1": {}, "v2": {"handoffs": [orders]}})
    triage.add_handoff(billing)

    assert sorted(triage.use("v2").handoffs) == ["orders"]
    assert sorted(Agent("t", harness=harness, memory=False, handoffs=[billing],
                        version="v1", versions={"v1": {}, "v2": {}})
                  .use("v2").handoffs) == ["billing"]
    assert (await triage.run("Help")).agent == "billing"


async def test_a_blueprint_declares_handoffs_between_its_agents():
    blueprint = Blueprint.from_text("""
agents:
  triage:
    mode: chat
    instructions: Work out what the customer needs.
    handoffs:
      - billing
      - {agent: orders, history: fresh, sticky: false, description: Where an order is.}
  billing:
    description: Refunds and invoices.
    handoffs: [triage]
  orders:
    memory: false
""")
    harness = Harness.testing(FakeProvider([to("billing"), "Refunded."]))
    agents = blueprint.build_all(harness=harness)

    triage = agents["triage"]
    assert triage.handoffs["billing"].agent is agents["billing"]
    assert agents["billing"].handoffs["triage"].agent is triage
    orders = triage.handoffs["orders"]
    assert (orders.history, orders.sticky, orders.description) == (
        "fresh", False, "Where an order is.")
    assert (await triage.run("I was charged twice.")).agent == "billing"

    # Built alone, it still comes with the agents it hands off to — on its harness.
    alone = blueprint.build("triage", harness=Harness.testing())
    assert alone.handoffs["billing"].agent.harness is alone.harness
    assert alone.handoffs["billing"].agent.handoffs["triage"].agent is alone

    with pytest.raises(ConfigurationError, match="not an agent in this blueprint"):
        Blueprint.from_dict({"agents": {"a": {"handoffs": ["ghost"]}}}).build("a")


# ----------------------------------------------------------------------
# the rest of the harness still applies
# ----------------------------------------------------------------------
async def test_the_agent_handing_off_is_not_held_to_what_an_answer_must_be():
    class Refund(BaseModel):
        amount: int

    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider(['{"amount": 40}']),
                    harness=harness, memory=False, output_type=Refund)
    triage = Agent("triage", provider=FakeProvider([to("billing")]), harness=harness,
                   memory=False, handoffs=[billing],
                   guardrails=AgentGuardrails(require_tools=["lookup"],
                                              min_output_chars=200))

    result = await triage.run("Refund me.")

    assert result.ok and result.data == Refund(amount=40)
    assert not result.violations


async def test_the_sources_and_todos_follow_the_work():
    @tool
    def search(query: str) -> str:
        """Search the archive.

        Args:
            query: what to look for.
        """
        return "https://example.com/q3 — revenue rose 12% in Q3"

    harness = Harness.testing()
    writer = Agent("writer", mode="research", depth="fast", harness=harness,
                   tools=[search], memory=False,
                   provider=FakeProvider(["Revenue rose 12% in Q3 [1]."]))
    lead = Agent("lead", mode="research", depth="fast", harness=harness,
                 tools=[search], memory=False, handoffs=[writer],
                 provider=FakeProvider([
                     tool_call("search", query="q3 revenue"),
                     tool_call("record_source", ref="https://example.com/q3",
                               title="Q3 results", finding="revenue rose 12%"),
                     to("writer", "write it up")]))

    result = await lead.run("How did Q3 go?")

    assert result.agent == "writer"
    assert [s.ref for s in result.sources] == ["https://example.com/q3"]
    assert "[1] Q3 results" in result.output
    assert not any("citation" in v or "cites" in v for v in result.violations)


async def test_agents_on_different_harnesses_keep_one_record(tmp_path):
    kept = Harness.local(tmp_path / "state", trace=False)
    billing = Agent("billing", provider=FakeProvider(["Refunded."]),
                    harness=Harness.testing(), memory=False)
    triage = Agent("triage", provider=FakeProvider([to("billing")]), harness=kept,
                   memory=False, handoffs=[billing])

    result = await triage.run("I was charged twice.")

    saved = await kept.sessions.load(result.session_id)
    assert saved.messages[-1].text == "Refunded."
    assert saved.metadata["handoff"]["active"] == "billing"
    assert saved.usage.calls == 2


async def test_an_entry_agent_that_keeps_no_sessions_keeps_none_of_the_chain():
    triage, billing = desk([to("billing")], ["Refunded."], mode=None,
                           persist_session=False)
    result = await triage.run("I was charged twice.")
    assert result.agent == "billing"
    with pytest.raises(ConfigurationError, match="no session"):
        await triage.harness.sessions.load(result.session_id)


async def test_artifacts_from_both_agents_stay_on_the_session():
    harness = Harness.testing()
    billing = Agent("billing", provider=FakeProvider(["Refunded."]), harness=harness)
    triage = Agent("triage", provider=FakeProvider([to("billing")]), harness=harness,
                   handoffs=[billing])
    triage.produce("intake.md", "charged twice")
    billing.produce("refund.md", "40 EUR")

    result = await triage.run("I was charged twice.")

    assert {a.name for a in result.artifacts} == {"intake.md", "refund.md"}
    saved = await harness.sessions.load(result.session_id)
    assert {a.name for a in saved.artifacts} == {"intake.md", "refund.md"}


async def test_governance_is_asked_before_a_conversation_changes_hands():
    harness = Harness.testing(governance=Governance())
    billing = Agent("billing", provider=FakeProvider(["Refunded."]), harness=harness,
                    memory=False, tools=[lookup])
    rogue = Agent("rogue", provider=FakeProvider(["never"]), harness=Harness.testing(),
                  memory=False)
    triage = Agent("triage", harness=harness, memory=False, handoffs=[billing, rogue],
                   provider=FakeProvider([to("rogue"), to("billing")]))

    result = await triage.run("I was charged twice.")

    assert result.agent == "billing" and [h.target for h in result.handoffs] == [
        "billing"]
    assert "not registered with governance" in results_of(result.messages)[0].content
    assert not rogue.provider.requests
    decisions = [e.decision for e in harness.audit.entries
                 if e.action == "governance.delegate"]
    assert decisions == ["deny", "allow"]
    # The specialist's answer is a top-level answer: it carries its own record.
    assert result.governance["delegation_depth"] == 0


async def test_a_stop_is_honoured_by_whoever_has_the_conversation():
    triage, billing = desk([to("billing")], ["never said"])

    @triage.hooks.on("handoff")
    def stop(ctx):
        triage.harness.control.stop("operator stop")

    result = await triage.run("I was charged twice.")

    assert result.agent == "billing" and result.stop_reason == "stopped"
    assert not billing.provider.requests


# ----------------------------------------------------------------------
# voice
# ----------------------------------------------------------------------
class Speech:
    rate = 16_000

    async def transcribe(self, audio: bytes, **kw: Any) -> str:
        return ""

    async def synthesize(self, text: str):
        yield b"\x00\x00" * 160


async def test_a_voice_conversation_stays_with_whoever_it_was_handed_to():
    triage, billing = desk([to("billing")], ["Refunded.", "It was sent today."],
                           mode=None)
    voice = VoiceAgent(triage, speech=Speech())

    first = [e async for e in voice.respond("I was charged twice.")]
    second = [e async for e in voice.respond("And the invoice?")]

    assert first[-1].text == "Refunded." and second[-1].text == "It was sent today."
    assert len(triage.provider.requests) == 1
    saved = await triage.harness.sessions.load(voice.session_id)
    assert saved.metadata["handoff"]["active"] == "billing"
