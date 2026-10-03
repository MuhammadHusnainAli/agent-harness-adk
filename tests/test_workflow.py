"""Declared workflows: sequence, parallel, loop, branch, graph, shared state."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from agent_harness import (
    Agent,
    Blueprint,
    Budget,
    ConfigurationError,
    FakeProvider,
    Harness,
    PolicyGate,
    Workflow,
    tool,
    tool_call,
)
from agent_harness.cli import main
from agent_harness.workflow import render

CALLS: list[str] = []


@tool
def find_charges(order: str) -> dict:
    """List what an order was charged.

    Args:
        order: the order number.
    """
    CALLS.append(f"find:{order}")
    return {"order": order, "charges": [40, 40], "duplicate": True}


@tool
def issue_refund(order: str, amount: float) -> str:
    """Refund part of an order.

    Args:
        order: the order number.
        amount: how much.
    """
    CALLS.append(f"refund:{order}:{amount}")
    return f"refunded {amount} on {order}"


@tool
def flaky(times: int) -> str:
    """Fails until it has been called `times` times.

    Args:
        times: how many calls it takes.
    """
    CALLS.append("flaky")
    if CALLS.count("flaky") < times:
        raise RuntimeError("not yet")
    return "ok"


@tool
async def slow(seconds: float, label: str = "") -> str:
    """Take a while.

    Args:
        seconds: how long.
        label: what to say when done.
    """
    CALLS.append(f"start:{label}")
    await asyncio.sleep(seconds)
    CALLS.append(f"end:{label}")
    return label


TOOLS = [find_charges, issue_refund, flaky, slow]


@pytest.fixture(autouse=True)
def _fresh():
    CALLS.clear()
    yield


def agent(name: str, *script: Any, harness: Harness, **kw: Any) -> Agent:
    return Agent(name, provider=FakeProvider(list(script)), harness=harness,
                 memory=False, **kw)


def flow(text: str, *, agents: list[Agent] | None = None,
         harness: Harness | None = None, **kw: Any) -> Workflow:
    return Workflow.from_text(text, agents=agents, tools=TOOLS,
                              harness=harness or Harness.testing(), **kw)


# ----------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------
def test_a_template_that_is_one_expression_keeps_its_type():
    context = {"state": {"total": 80, "items": [1, 2]}, "inputs": {"name": "Ada"}}
    assert render("{{ state.total }}", context) == 80
    assert render("{{ state.items }}", context) == [1, 2]
    assert render("Total: {{ state.total }} for {{ inputs.name }}", context) == (
        "Total: 80 for Ada")
    assert render({"a": ["{{ state.total / 2 }}", "plain"]}, context) == {
        "a": [40.0, "plain"]}
    assert render("{{ state.missing }}", context) is None
    assert render("got {{ state.items }}", context) == "got [1, 2]"
    assert render("{{ join(state.items, '-') }} {{ json(inputs) }}", context) == (
        '1-2 {"name": "Ada"}')


# ----------------------------------------------------------------------
# sequence and state
# ----------------------------------------------------------------------
async def test_steps_run_in_order_and_share_what_they_found():
    harness = Harness.testing()
    classifier = agent("classifier", '{"refund": true, "amount": 40}', harness=harness)
    writer = agent("writer", "We refunded 40 EUR.", harness=harness)
    workflow = flow("""
name: refunds
inputs:
  order: {required: true}
  message: {default: "I was charged twice."}
state: {attempts: 0}
steps:
  - id: charges
    tool: find_charges
    args: {order: "{{ inputs.order }}"}
  - id: classify
    agent: classifier
    input: "Charges {{ steps.charges.output.charges }}. Customer: {{ inputs.message }}"
    parse: json
    save: verdict
  - set: {attempts: "{{ state.attempts + 1 }}"}
  - id: refund
    when: state.verdict.refund
    tool: issue_refund
    args: {order: "{{ inputs.order }}", amount: "{{ state.verdict.amount }}"}
  - id: reply
    agent: writer
output: "{{ steps.reply.output }}"
""", agents=[classifier, writer], harness=harness)

    result = await workflow.run({"order": "4182"})

    assert result.ok and result.output == "We refunded 40 EUR."
    assert result.state == {"attempts": 1, "verdict": {"refund": True, "amount": 40}}
    assert CALLS == ["find:4182", "refund:4182:40.0"]
    assert [s for s in result.steps] == ["charges", "classify", "set_1", "refund",
                                         "reply"]
    asked = classifier.provider.requests[0].messages[0].text
    assert asked == "Charges [40, 40]. Customer: I was charged twice."
    # With no `input`, a step is handed what the one before it produced.
    assert writer.provider.requests[0].messages[0].text == "refunded 40.0 on 4182"
    assert result.steps["classify"].data == {"refund": True, "amount": 40}
    assert result.steps_run == 5 and result.usage.calls == 2
    assert workflow.spec.state == {"attempts": 0}       # the declaration is untouched


async def test_a_step_is_skipped_unless_its_condition_holds():
    workflow = flow("""
steps:
  - id: charges
    tool: find_charges
    args: {order: "1"}
  - id: refund
    when: "{{ not steps.charges.output.duplicate }}"
    tool: issue_refund
    args: {order: "1", amount: 40}
""")
    result = await workflow.run()
    assert result.steps["refund"].status == "skipped"
    # A skipped step produces nothing: the output is still the last real one.
    assert result.output["order"] == "1" and CALLS == ["find:1"]


async def test_an_output_contract_becomes_the_steps_data():
    class Verdict(BaseModel):
        refund: bool

    harness = Harness.testing()
    judge = agent("judge", '{"refund": true}', harness=harness, output_type=Verdict)
    result = await flow("""
steps:
  - {id: judge, agent: judge, input: "Should we?"}
  - return: "{{ steps.judge.data.refund }}"
  - tool: issue_refund
    args: {order: "never", amount: 1}
""", agents=[judge], harness=harness).run()
    assert result.output is True and result.ok and not CALLS


# ----------------------------------------------------------------------
# branching
# ----------------------------------------------------------------------
BRANCHES = """
inputs: {amount: 0}
steps:
  - id: route
    if: inputs.amount > 100
    then:
      - set: {route: manager}
    else:
      - set: {route: auto}
  - id: tier
    switch:
      - when: inputs.amount > 1000
        steps: [{set: {tier: gold}}]
      - when: inputs.amount > 100
        steps: [{set: {tier: silver}}]
      - steps: [{set: {tier: none}}]
"""


@pytest.mark.parametrize("amount, route, tier, taken", [
    (5000, "manager", "gold", "case 1"), (500, "manager", "silver", "case 2"),
    (5, "auto", "none", "default")])
async def test_one_branch_is_taken(amount, route, tier, taken):
    result = await flow(BRANCHES).run({"amount": amount})
    assert result.state == {"route": route, "tier": tier}
    assert result.steps["route"].branch == ("then" if amount > 100 else "else")
    assert result.steps["tier"].branch == taken


# ----------------------------------------------------------------------
# parallel, foreach, loop
# ----------------------------------------------------------------------
async def test_parallel_branches_run_at_the_same_time():
    result = await flow("""
steps:
  - id: both
    parallel:
      - {id: a, tool: slow, args: {seconds: 0.05, label: a}}
      - id: b
        steps:
          - {tool: slow, args: {seconds: 0.01, label: b1}}
          - {tool: slow, args: {seconds: 0.01, label: b2}}
  - return: "{{ steps.both.output }}"
""").run()
    assert result.output == {"a": "a", "b": "b2"}
    assert CALLS[:2] == ["start:a", "start:b1"] and CALLS[-1] == "end:a"


async def test_parallel_can_be_capped():
    await flow("""
steps:
  - parallel:
      - {tool: slow, args: {seconds: 0.01, label: a}}
      - {tool: slow, args: {seconds: 0.01, label: b}}
    concurrency: 1
""").run()
    assert CALLS == ["start:a", "end:a", "start:b", "end:b"]


async def test_foreach_runs_the_body_once_per_item():
    result = await flow("""
inputs: {orders: {type: array}}
steps:
  - id: each
    foreach: "{{ inputs.orders }}"
    as: order
    concurrency: 3
    steps:
      - id: found
        tool: find_charges
        args: {order: "{{ order.id }}"}
      - return_total: ignored
""".replace("      - return_total: ignored\n", """      - id: line
        set: {last: "{{ index }}:{{ steps.found.output.order }}"}
"""), ).run({"orders": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    assert result.ok and result.steps["each"].iterations == 3
    assert [o["last"] for o in result.output] == ["0:a", "1:b", "2:c"]
    assert sorted(CALLS) == ["find:a", "find:b", "find:c"]


async def test_foreach_takes_a_count_a_mapping_or_nothing():
    text = """
inputs: {over: null}
steps:
  - id: each
    foreach: "{{ inputs.over }}"
    steps: [{set: {seen: "{{ item }}"}}]
"""
    assert (await flow(text).run({"over": 3})).state == {"seen": 2}
    assert (await flow(text).run({"over": {"k": 1}})).state == {
        "seen": {"key": "k", "value": 1}}
    assert (await flow(text).run()).steps["each"].iterations == 0
    failed = await flow(text).run({"over": "abc"})
    assert not failed.ok and "foreach needs a list" in failed.error


async def test_a_loop_goes_round_until_its_condition_holds():
    harness = Harness.testing()
    writer = agent("writer", "draft one", "draft two", "draft three", harness=harness)
    critic = agent("critic", '{"ok": false}', '{"ok": false}', '{"ok": true}',
                   harness=harness)
    result = await flow("""
steps:
  - id: refine
    loop: {max: 5, until: state.review.ok}
    steps:
      - {id: draft, agent: writer, input: "Write it. Pass {{ loop.iteration }}."}
      - {id: review, agent: critic, parse: json, save: review}
  - return: "{{ steps.draft.output }} after {{ steps.refine.iterations }}"
""", agents=[writer, critic], harness=harness).run()
    assert result.output == "draft three after 3"
    assert not result.steps["refine"].exhausted
    assert critic.provider.requests[1].messages[0].text == "draft two"


async def test_a_loop_that_never_gets_there_stops_at_its_ceiling():
    result = await flow("""
state: {n: 0}
steps:
  - id: spin
    loop: {max: 4, until: state.n > 100}
    steps: [{set: {n: "{{ state.n + 1 }}"}}]
  - id: counted
    loop: 2
    steps: [{set: {n: "{{ state.n + 10 }}"}}]
  - id: guarded
    loop: {while: state.n < 0}
    steps: [{set: {n: -1}}]
""").run()
    assert result.state == {"n": 24}
    assert result.steps["spin"].exhausted and result.steps["spin"].iterations == 4
    assert not result.steps["counted"].exhausted
    assert result.steps["guarded"].iterations == 0


async def test_a_workflow_that_does_not_end_is_ended():
    result = await flow("""
max_steps: 12
steps:
  - loop: 1000
    steps: [{set: {n: 1}}]
""").run()
    assert not result.ok and "ran 12 steps without finishing" in result.error


# ----------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------
async def test_a_graph_runs_each_node_as_soon_as_what_it_needs_is_done():
    result = await flow("""
graph:
  - {id: fetch, tool: slow, args: {seconds: 0.01, label: fetch}}
  - {id: a, needs: [fetch], tool: slow, args: {seconds: 0.03, label: "a+{{ previous }}"}}
  - {id: b, needs: [fetch], tool: slow, args: {seconds: 0.01, label: b}}
  - id: merge
    needs: [a, b]
    set: {merged: "{{ previous.a }} & {{ previous.b }}"}
""").run()
    assert result.ok and result.state == {"merged": "a+fetch & b"}
    order = [c for c in CALLS if c.startswith("start")]
    assert order == ["start:fetch", "start:a+fetch", "start:b"]
    assert CALLS.index("end:b") < CALLS.index("end:a+fetch")


async def test_a_node_whose_needs_did_not_arrive_does_not_run():
    result = await flow("""
inputs: {vip: false}
graph:
  - {id: start, set: {seen: start}}
  - {id: vip, needs: [start], when: inputs.vip, set: {lane: vip}}
  - {id: normal, needs: [start], when: not inputs.vip, set: {lane: normal}}
  - {id: gift, needs: [vip], set: {gift: true}}
  - {id: done, needs: [vip, normal], join: any, set: {finished: true}}
  - {id: strict, needs: [vip, normal], set: {never: true}}
""").run()
    assert result.state == {"seen": "start", "lane": "normal", "finished": True}
    assert {k: v.status for k, v in result.steps.items()} == {
        "start": "done", "vip": "skipped", "normal": "done", "gift": "skipped",
        "done": "done", "strict": "skipped"}


async def test_a_graph_can_be_a_step_inside_a_sequence():
    result = await flow("""
steps:
  - id: fan
    graph:
      - {id: a, set: {a: 1}}
      - {id: b, needs: [a], set: {b: "{{ state.a + 1 }}"}}
  - return: "{{ state.b }}"
""").run()
    assert result.output == 2


# ----------------------------------------------------------------------
# failure
# ----------------------------------------------------------------------
async def test_a_failed_step_ends_the_run_and_says_where():
    result = await flow("""
steps:
  - {id: first, set: {a: 1}}
  - id: group
    steps:
      - {id: boom, tool: flaky, args: {times: 99}}
  - {id: never, set: {b: 1}}
""").run()
    assert result.status == "failed" and result.failed_step == "boom"
    assert result.error == "step 'boom' failed: Error: flaky failed: not yet"
    assert result.state == {"a": 1} and "never" not in result.steps
    assert result.steps["boom"].status == result.steps["group"].status == "failed"


async def test_a_step_can_be_tried_again():
    result = await flow("""
steps:
  - {id: shaky, tool: flaky, args: {times: 3}, retry: {max: 3, delay: 0}}
""").run()
    assert result.ok and result.steps["shaky"].attempts == 3
    assert result.failed_step == ""

    CALLS.clear()
    result = await flow("""
steps:
  - {id: shaky, tool: flaky, args: {times: 9}, retry: 1}
""").run()
    assert not result.ok and CALLS.count("flaky") == 2


async def test_a_failure_can_be_carried_past():
    result = await flow("""
steps:
  - {id: boom, tool: flaky, args: {times: 99}, on_error: continue}
  - if: steps.boom.status == 'failed'
    then: [{set: {fallback: "{{ steps.boom.error }}"}}]
""").run()
    assert result.ok and result.failed_step == ""
    assert "not yet" in result.state["fallback"]


async def test_a_step_that_takes_too_long_is_cut_off():
    result = await flow("""
steps:
  - {id: wait, tool: slow, args: {seconds: 5, label: x}, timeout: 0.02}
""").run()
    assert not result.ok and "step 'wait' timed out after 0.02s" in result.error

    result = await flow("""
timeout: 0.02
steps: [{wait: 5}]
""").run()
    assert not result.ok and "timed out" in result.error


async def test_one_failing_branch_stops_the_others():
    result = await flow("""
steps:
  - parallel:
      - {id: long, tool: slow, args: {seconds: 5, label: long}}
      - {id: boom, fail: "no stock for {{ inputs.input }}"}
""").run("order 9")
    assert not result.ok and result.error == "step 'boom' failed: no stock for order 9"
    assert "end:long" not in CALLS


async def test_an_agent_that_fails_fails_its_step():
    harness = Harness.testing()
    bad = agent("bad", tool_call("nope"), tool_call("nope"), harness=harness,
                max_steps=2)
    result = await flow("steps: [{id: ask, agent: bad, input: go}]",
                        agents=[bad], harness=harness).run()
    assert not result.ok and "MaxStepsExceeded" in result.error

    plain = agent("plain", "not json at all", harness=harness)
    result = await flow("steps: [{id: ask, agent: plain, input: go, parse: json}]",
                        agents=[plain], harness=harness).run()
    assert "was to answer in JSON and did not" in result.error


# ----------------------------------------------------------------------
# the rails
# ----------------------------------------------------------------------
async def test_tool_steps_pass_the_permission_gate_and_the_audit_trail():
    harness = Harness.testing()
    harness.policy = PolicyGate("allow", deny=["issue_refund"])
    result = await flow("""
steps:
  - {tool: find_charges, args: {order: "1"}}
  - {id: refund, tool: issue_refund, args: {order: "1", amount: 40}}
""", harness=harness).run()
    assert not result.ok and "Not permitted" in result.error
    assert CALLS == ["find:1"]
    actions = [(e.action, e.target, e.decision) for e in harness.audit.entries]
    assert ("tool_call", "issue_refund", "deny") in actions
    assert ("workflow_step", "refund", "error") in actions
    assert actions[0][0] == "workflow_start" and actions[-1][0] == "workflow_end"


async def test_a_hook_can_refuse_a_run_or_a_step():
    harness = Harness.testing()
    workflow = flow("steps: [{id: a, set: {x: 1}}, {id: b, set: {y: 1}}]",
                    harness=harness)

    @harness.hooks.on("workflow_step")
    def no_b(ctx):
        if ctx.data["step_id"] == "b":
            ctx.block("b is frozen")

    result = await workflow.run()
    assert result.error == "step 'b' failed: not permitted: b is frozen"
    assert result.state == {"x": 1}

    harness.hooks.add("workflow_start", lambda ctx: ctx.block("maintenance"))
    assert "may not run: maintenance" in (await workflow.run()).error


async def test_a_stop_ends_the_workflow_at_the_next_step():
    harness = Harness.testing()
    workflow = flow("""
steps:
  - {id: a, set: {x: 1}}
  - {id: b, set: {y: 1}}
""", harness=harness)
    harness.hooks.add("workflow_step", lambda ctx: harness.control.stop("enough"))
    result = await workflow.run()
    assert result.status == "stopped" and result.state == {"x": 1}


async def test_a_workflow_budget_covers_every_agent_in_it():
    harness = Harness.testing()
    talker = agent("talker", "one two three four five six",
                   "seven eight nine ten eleven twelve", harness=harness)
    result = await flow("""
budget: {max_output_tokens: 8}
steps:
  - {id: first, agent: talker, input: go}
  - {id: second, agent: talker, input: again}
""", agents=[talker], harness=harness).run()
    assert not result.ok and result.failed_step == "second"
    assert "stopped at its budget" in result.error
    assert isinstance(flow("steps: [{set: {a: 1}}]").spec.budget, type(None))
    assert Budget(max_output_tokens=8).max_output_tokens == 8


async def test_an_agent_step_can_keep_its_conversation_through_the_run():
    harness = Harness.testing()
    chatty = agent("chatty", "Noted.", "It was 4182.", harness=harness)
    await flow("""
steps:
  - {agent: chatty, input: "The order is 4182.", thread: true}
  - {agent: chatty, input: "Which order?", thread: true}
""", agents=[chatty], harness=harness).run()
    assert [m.text for m in chatty.provider.requests[-1].messages] == [
        "The order is 4182.", "Noted.", "Which order?"]
    # Without it, each step is its own task.
    chatty.provider.queue("a", "b")
    await flow("steps: [{agent: chatty, input: one}, {agent: chatty, input: two}]",
               agents=[chatty], harness=harness).run()
    assert [m.text for m in chatty.provider.requests[-1].messages] == ["two"]


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------
async def test_a_stream_says_what_is_happening():
    workflow = flow("""
steps:
  - {id: a, set: {x: 1}}
  - {id: b, when: "false", set: {y: 1}}
  - {id: c, tool: flaky, args: {times: 2}, retry: 2}
""")
    events = [e async for e in workflow.stream()]
    assert [(e.type, e.step) for e in events] == [
        ("workflow_start", ""), ("step_start", "a"), ("step_end", "a"),
        ("step_skipped", "b"), ("step_start", "c"), ("step_retry", "c"),
        ("step_end", "c"), ("workflow_end", "")]
    assert events[-1].data["result"].ok

    with pytest.raises(ConfigurationError, match="does not take"):
        [e async for e in flow("inputs: {a: 1}\nsteps: [{set: {x: 1}}]").stream({"b": 1})]


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
async def test_inputs_are_checked_before_anything_runs():
    workflow = flow("""
inputs:
  order: {required: true, type: string}
  amount: {default: 10, type: number}
  note: hello
steps: [{return: "{{ inputs.order }}/{{ inputs.amount }}/{{ inputs.note }}"}]
""")
    assert (await workflow.run({"order": "7"})).output == "7/10/hello"
    with pytest.raises(ConfigurationError, match="needs the input 'order'"):
        await workflow.run({})
    with pytest.raises(ConfigurationError, match="does not take colour"):
        await workflow.run({"order": "7", "colour": "red"})
    with pytest.raises(ConfigurationError, match="must be number"):
        await workflow.run({"order": "7", "amount": "lots"})
    with pytest.raises(ConfigurationError, match="pass a mapping"):
        await workflow.run("7")
    # One input: a bare value is that input.
    assert (await flow("inputs: {q: {required: true}}\nsteps: [{return: '{{ inputs.q }}'}]")
            .run("hi")).output == "hi"


# ----------------------------------------------------------------------
# a declaration that is wrong says so before it runs
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text, message", [
    ("steps: [{id: a, set: {x: 1}}, {id: a, set: {y: 1}}]", "two steps are called 'a'"),
    ("steps: [{id: a}]", "does nothing"),
    ("steps: [{id: a, tool: slow, set: {x: 1}}]", "is tool and set at once"),
    ("steps: [{id: a, tool: slow, colour: red}]", "Extra inputs are not permitted"),
    ("steps: [{id: a-b, set: {x: 1}}]", "must be a plain name"),
    ("steps: [{set: {x: '{{ steps.ghost.output }}'}}]", "no step is called that"),
    ("steps: [{set: {x: '{{ stat.total }}'}}]", "uses 'stat', which is not known"),
    ("steps: [{set: {x: '{{ item }}'}}]", "uses 'item'"),
    ("inputs: {a: 1}\nsteps: [{set: {x: '{{ inputs.b }}'}}]", "does not declare"),
    ("steps: [{set: {x: '{{ __import__(1) }}'}}]", "not an allowed function"),
    ("steps: [{when: 'a ==', set: {x: 1}}]", "invalid syntax"),
    ("steps: [{tool: nothing_like_it}]", "no tool named 'nothing_like_it'"),
    ("steps: [{agent: ghost}]", "no agent named 'ghost'"),
    ("steps: [{foreach: [1]}]", "a foreach needs `steps`"),
    ("steps: [{id: a, set: {x: 1}, needs: [b]}]", "only a node of a `graph`"),
    ("graph: [{id: a, needs: [b], set: {x: 1}}, {id: b, needs: [a], set: {y: 1}}]",
     "wait on each other"),
    ("graph: [{id: a, needs: [zz], set: {x: 1}}]", "not a node of the same graph"),
    ("name: x", "needs `steps` or `graph`"),
    ("steps: [{if: 'true'}]", "an `if` needs a `then`"),
    ("steps: [{switch: [{steps: [{set: {a: 1}}]}, {when: 'true', steps: []}]}]",
     "must come last"),
    ("steps: [{set: {a: 1}, retry: {max: 99}}]", "less than or equal to 20"),
    ("- not a mapping", "must be a mapping"),
])
def test_what_is_wrong_is_said_when_the_file_is_loaded(text, message):
    with pytest.raises(ConfigurationError, match=message):
        flow(text)


# ----------------------------------------------------------------------
# where agents and tools come from
# ----------------------------------------------------------------------
async def test_a_file_can_declare_its_own_agents(tmp_path):
    file = tmp_path / "flow.json"
    file.write_text(json.dumps({
        "name": "desk",
        "prompts": {"style": "Answer in one line."},
        # Braces that are not a prompt's name are left as they are.
        "agents": {"writer": {"instructions": '{style} Like {"ok": true}.',
                              "memory": False}},
        "steps": [
            {"id": "write", "agent": "writer", "input": "Say hello."},
            {"id": "check", "input": "Is this polite? {{ previous }}",
             "agent": {"instructions": "You check tone.", "memory": False}},
        ]}))
    harness = Harness.testing(FakeProvider(["Hello.", "Yes."]))
    workflow = Workflow.from_file(file, harness=harness)

    result = await workflow.run()

    assert result.output == "Yes."
    assert workflow.agents["writer"].instructions == (
        'Answer in one line. Like {"ok": true}.')
    assert workflow.agents["check"].harness is harness
    assert harness.provider.requests[1].messages[0].text == "Is this polite? Hello."


async def test_a_blueprint_declares_workflows_over_its_agents():
    blueprint = Blueprint.from_text("""
agents:
  writer: {instructions: Write., memory: false, tools: []}
workflows:
  publish:
    inputs: {topic: {required: true}}
    steps:
      - {id: draft, agent: writer, input: "Write about {{ inputs.topic }}."}
      - {tool: "MODULE:issue_refund", args: {order: x, amount: 1}}
""".replace("MODULE", __name__))
    harness = Harness.testing(FakeProvider(["A draft."]))
    workflow = blueprint.workflow("publish", harness=harness)
    result = await workflow.run({"topic": "tea"})
    assert result.ok and result.steps["draft"].output == "A draft."
    assert CALLS == ["refund:x:1.0"]
    with pytest.raises(ConfigurationError, match="no workflow 'nope'"):
        blueprint.workflow("nope")


async def test_an_agent_can_call_a_workflow_as_a_tool():
    harness = Harness.testing()
    workflow = flow("""
name: refund_order
description: Refund an order in full.
inputs: {order: {required: true, type: string, description: The order number.}}
steps:
  - {id: found, tool: find_charges, args: {order: "{{ inputs.order }}"}}
  - tool: issue_refund
    args: {order: "{{ inputs.order }}", amount: "{{ sum(steps.found.output.charges) }}"}
""", harness=harness)
    desk = agent("desk", tool_call("refund_order", order="4182"), "Done.",
                 harness=harness, tools=[workflow.as_tool()])

    result = await desk.run("Refund order 4182.")

    assert result.output == "Done." and CALLS == ["find:4182", "refund:4182:80.0"]
    schema = desk.tools.get("refund_order").parameters
    assert schema["required"] == ["order"]
    assert schema["properties"]["order"] == {"type": "string",
                                             "description": "The order number."}


def test_the_outline_shows_the_shape():
    outline = flow(BRANCHES.replace("inputs: {amount: 0}",
                                    "name: tiers\ninputs: {amount: 0}")).describe()
    assert outline.splitlines()[:4] == [
        "tiers", "inputs: amount?", "  - route: if inputs.amount > 100", "    then:"]
    assert "      - set_1: set" in outline and "    default:" in outline


# ----------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------
def test_the_command_runs_a_file_and_checks_one(tmp_path, capsys):
    file = tmp_path / "flow.yaml"
    file.write_text("""
name: refund
inputs: {order: {required: true, type: string}, amount: {default: 1}}
steps:
  - tool: issue_refund
    args: {order: "{{ inputs.order }}", amount: "{{ inputs.amount }}"}
""")
    tool_path = f"{__name__}:issue_refund"

    assert main(["workflow", str(file), "--check", "--tool", tool_path]) == 0
    assert "issue_refund" in capsys.readouterr().out

    code = main(["workflow", str(file), "--tool", tool_path, "--input", "order=4182",
                 "--input", "amount=40"])
    printed = capsys.readouterr()
    assert code == 0 and printed.out.strip() == "refunded 40.0 on 4182"
    assert "[done · 1 steps" in printed.err

    assert main(["workflow", str(file), "--tool", tool_path]) == 1
    assert "needs the input 'order'" in capsys.readouterr().err
