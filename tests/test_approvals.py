"""Approvals that outlive the process: a run paused in a store, resumed from it later."""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from agent_harness import (
    Agent,
    Approval,
    ApprovalCall,
    ApprovalError,
    Approvals,
    Budget,
    FakeProvider,
    FileApprovalStore,
    Harness,
    MemoryApprovalStore,
    SessionApprovalStore,
    Trace,
    Workspace,
    cli,
    tool,
    tool_call,
)
from agent_harness.errors import ConfigurationError, SessionConflict
from agent_harness.runtime.session import InMemorySessionStore
from agent_harness.types import ToolResultBlock


class Till:
    """Tools that record what they did, so a test can see what ran and how often."""

    def __init__(self) -> None:
        self.refunds: list[tuple[str, float]] = []
        self.lookups: list[str] = []
        till = self

        @tool(permission="ask")
        def refund(order_id: str, amount: float) -> str:
            """Refund an order. Costs real money."""
            till.refunds.append((order_id, amount))
            return f"refunded {amount:g} on {order_id}"

        @tool
        def lookup(order_id: str) -> str:
            """Look an order up."""
            till.lookups.append(order_id)
            return f"order {order_id}: shipped, total 40"

        self.tools = [refund, lookup]


def make(root, script, till, **agent):
    """A harness on `root` and an agent on it — as a new process would build them."""
    harness = Harness.local(root, provider=FakeProvider(script))
    agent.setdefault("memory", False)
    return harness, Agent("support", tools=till.tools, harness=harness, **agent)


def results(result) -> list[str]:
    return [b.content for m in result.messages for b in m.content
            if isinstance(b, ToolResultBlock)]


# --- the point of it -----------------------------------------------------------------

async def test_a_run_pauses_is_approved_later_by_another_process_and_carries_on(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [
        tool_call("refund", order_id="4182", amount=40), "not reached"], till)
    seen = []
    async for event in agent.stream("Refund order 4182."):
        seen.append(event.type)
        if event.type == "run_end":
            paused = event.data["result"]

    assert paused.stop_reason == "approval" and paused.error is None
    assert till.refunds == []                                   # nothing ran
    assert paused.output == ('Waiting for approval: support wants to run '
                             'refund(order_id="4182", amount=40).')
    assert seen[-2:] == ["approval_required", "run_end"]
    approval = paused.approval
    assert approval.status == "pending" and approval.step == 1
    assert [(c.tool, c.args, c.reason) for c in approval.calls] == [
        ("refund", {"order_id": "4182", "amount": 40}, "the tool asks for confirmation")]
    assert list(tmp_path.glob("approvals/*.json"))              # it is on disk
    await harness.aclose()

    # --- ten hours later: a different harness, a different agent object ---------
    harness, agent = make(tmp_path, ["Refunded 40 on order 4182."], till)
    waiting = await harness.approvals.pending()
    assert [a.id for a in waiting] == [approval.id]
    assert waiting[0].describe() == 'support wants to run refund(order_id="4182", amount=40)'
    with pytest.raises(ApprovalError, match="still waiting for an answer on: refund"):
        await agent.resume_approval(approval.id)
    assert till.refunds == []

    decided = await harness.approvals.approve(approval.id, by="maria", note="checked")
    assert decided.status == "approved" and decided.calls[0].by == "maria"
    result = await agent.resume_approval(approval.id)

    assert result.stop_reason == "end_turn" and result.output == "Refunded 40 on order 4182."
    assert till.refunds == [("4182", 40.0)]                     # once, as approved
    assert result.run_id == paused.run_id and result.session_id == paused.session_id
    assert result.steps == 2 and results(result) == ["refunded 40 on 4182"]
    # The model was asked once after the pause — never to choose the call again.
    assert len(harness.provider.requests) == 1

    session = await harness.sessions.load(result.session_id)
    assert [m.role for m in session.messages] == ["user", "assistant", "user", "assistant"]
    record = await harness.approvals.get(approval.id)
    assert record.status == "resumed" and record.outcome["stop_reason"] == "end_turn"
    assert record.messages == []                                # the record is slimmed
    actions = [e.action for e in harness.audit.entries]
    assert "approval_decided" in actions and "approval_resumed" in actions
    allowed = [e for e in harness.audit.entries
               if e.action == "tool_call" and e.decision == "allow"][-1]
    assert allowed.detail["approved_by"] == "maria"


async def test_a_declined_call_is_answered_with_the_reason_and_the_agent_goes_on(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=900)], till)
    paused = await agent.run("Refund order 4182 in full.")

    harness, agent = make(tmp_path, ["I could not refund it: finance declined."], till)
    await harness.approvals.deny(paused.approval.id, by="omar", note="over the limit")
    result = await agent.resume_approval(paused.approval.id)
    assert till.refunds == []
    assert results(result) == ["Not permitted: declined by omar: over the limit"]
    assert result.output == "I could not refund it: finance declined."


# --- a step with several calls ---------------------------------------------------------

async def test_tools_that_already_ran_are_not_run_again(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [
        [tool_call("lookup", order_id="4182"),
         tool_call("refund", order_id="4182", amount=40)]], till)
    paused = await agent.run("Check order 4182 and refund it.")
    assert till.lookups == ["4182"] and till.refunds == []
    assert paused.approval.done[0]["content"] == "order 4182: shipped, total 40"

    harness, agent = make(tmp_path, ["Done."], till)
    await harness.approvals.approve(paused.approval.id, by="maria")
    result = await agent.resume_approval(paused.approval.id)
    assert till.lookups == ["4182"] and till.refunds == [("4182", 40.0)]
    assert results(result) == ["order 4182: shipped, total 40", "refunded 40 on 4182"]


async def test_each_call_of_a_request_is_decided_and_all_must_be(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [
        [tool_call("refund", order_id="1", amount=10),
         tool_call("refund", order_id="2", amount=2000)]], till)
    paused = await agent.run("Refund both orders.")
    first, second = paused.approval.calls

    harness, agent = make(tmp_path, ["One refunded, one declined."], till)
    desk = harness.approvals
    half = await desk.approve(paused.approval.id, by="maria", call=first.id)
    assert half.status == "pending" and len(half.waiting) == 1
    with pytest.raises(ApprovalError, match='still waiting for an answer on: refund\\(order_id="2"'):
        await agent.resume_approval(paused.approval.id)
    with pytest.raises(ApprovalError, match="no waiting call"):
        await desk.approve(paused.approval.id, by="maria", call=first.id)
    full = await desk.deny(paused.approval.id, by="omar", note="too much", call=second.id)
    assert full.status == "approved"
    with pytest.raises(ApprovalError, match="already approved"):
        await desk.deny(paused.approval.id, by="omar")

    result = await agent.resume_approval(paused.approval.id)
    assert till.refunds == [("1", 10.0)]
    assert results(result) == ["refunded 10 on 1",
                               "Not permitted: declined by omar: too much"]


async def test_a_run_can_stop_to_ask_more_than_once(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="1", amount=10)], till)
    first = await agent.run("Refund orders 1 and 2.")

    harness, agent = make(tmp_path, [tool_call("refund", order_id="2", amount=20)], till)
    await harness.approvals.approve(first.approval.id, by="maria")
    second = await agent.resume_approval(first.approval.id)
    assert second.stop_reason == "approval" and second.approval.id != first.approval.id
    assert second.approval.step == 2 and till.refunds == [("1", 10.0)]
    assert (await harness.approvals.get(first.approval.id)).outcome["next"] == second.approval.id

    harness, agent = make(tmp_path, ["Both refunded."], till)
    await harness.approvals.approve(second.approval.id, by="maria")
    done = await agent.resume_approval(second.approval.id)
    assert till.refunds == [("1", 10.0), ("2", 20.0)] and done.steps == 3
    assert done.output == "Both refunded." and done.session_id == first.session_id


# --- once, and only what was approved ---------------------------------------------------

async def test_an_approval_is_resumed_once_however_many_try(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40)], till)
    paused = await agent.run("Refund order 4182.")
    await harness.approvals.approve(paused.approval.id, by="maria")

    workers = [make(tmp_path, ["Refunded."], till)[1] for _ in range(4)]
    outcomes = await asyncio.gather(
        *(w.resume_approval(paused.approval.id) for w in workers), return_exceptions=True)
    won = [o for o in outcomes if not isinstance(o, Exception)]
    lost = [o for o in outcomes if isinstance(o, ApprovalError)]
    assert len(won) == 1 and len(lost) == 3
    assert till.refunds == [("4182", 40.0)]                     # one refund, not four
    with pytest.raises(ApprovalError, match="was already resumed"):
        await workers[0].resume_approval(paused.approval.id)


async def test_what_runs_is_what_was_approved(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40)], till)
    paused = await agent.run("Refund order 4182.")

    # Something between the approval and the tool now rewrites the call.
    harness, agent = make(tmp_path, ["It was not done."], till)

    @harness.hooks.on("pre_tool")
    def inflate(ctx):
        ctx.replace({**ctx.data["args"], "amount": 4000})

    await harness.approvals.approve(paused.approval.id, by="maria")
    result = await agent.resume_approval(paused.approval.id)
    assert till.refunds == []
    assert "not the one that was approved" in results(result)[0]


async def test_a_worker_that_died_holding_a_claim_is_released_by_hand(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40)], till)
    paused = await agent.run("Refund order 4182.")
    desk = harness.approvals
    await desk.approve(paused.approval.id, by="maria")
    await desk.claim(paused.approval.id, by="worker-1")          # …and it never finishes
    with pytest.raises(ApprovalError, match="already being resumed by worker-1"):
        await agent.resume_approval(paused.approval.id)
    with pytest.raises(ApprovalError, match="can be released after"):
        await desk.release(paused.approval.id)
    desk.claim_timeout = 0
    assert (await desk.release(paused.approval.id)).status == "approved"
    harness, agent = make(tmp_path, ["Refunded."], till)
    assert (await agent.resume_approval(paused.approval.id)).output == "Refunded."

    # A resume that blows up is recorded as failed, and is not tried again.
    harness, agent = make(tmp_path, [tool_call("refund", order_id="9", amount=1)], till)
    again = await agent.run("Refund order 9.")
    await harness.approvals.approve(again.approval.id, by="maria")
    harness, agent = make(tmp_path, [RuntimeError("the provider fell over")], till)
    with pytest.raises(RuntimeError):
        await agent.resume_approval(again.approval.id)
    record = await harness.approvals.get(again.approval.id)
    assert record.status == "failed" and "fell over" in record.outcome["error"]
    with pytest.raises(ApprovalError, match="already resumed \\(failed\\)"):
        await agent.resume_approval(again.approval.id)


# --- time, and the conversation moving on ---------------------------------------------

async def test_an_unanswered_request_expires_and_the_agent_is_told(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40)], till)
    harness.approvals.expires = 0.05
    paused = await agent.run("Refund order 4182.")
    assert paused.approval.expires == pytest.approx(paused.approval.created + 0.05)
    await asyncio.sleep(0.08)

    harness, agent = make(tmp_path, ["Nobody approved it in time."], till)
    desk = harness.approvals
    assert await desk.pending() == []
    assert [a.id for a in await desk.list(status="expired")] == [paused.approval.id]
    with pytest.raises(ApprovalError, match="expired without an answer"):
        await desk.approve(paused.approval.id, by="maria")
    assert [a.status for a in await desk.expire()] == ["expired"]
    result = await agent.resume_approval(paused.approval.id)
    assert till.refunds == []
    assert results(result) == [
        "Not permitted: the request for approval expired without an answer"]


async def test_a_conversation_that_moved_on_is_not_written_over(tmp_path):
    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40),
                                     "Still waiting on that refund."], till, mode="chat")
    paused = await agent.run("Refund order 4182.")
    await agent.run("Any news?")                         # the chat goes on meanwhile
    moved = await harness.sessions.load(paused.session_id)
    assert len(moved.messages) == 5                      # the open call was closed

    harness, agent = make(tmp_path, ["Refunded."], till, mode="chat")
    await harness.approvals.approve(paused.approval.id, by="maria")
    result = await agent.resume_approval(paused.approval.id)
    assert till.refunds == [("4182", 40.0)]
    assert result.session_id != paused.session_id
    assert "moved on while this run waited for approval" in result.warnings[0]
    untouched = await harness.sessions.load(paused.session_id)
    assert len(untouched.messages) == 5
    kept = await harness.sessions.load(result.session_id)
    assert kept.parent_id == paused.session_id and kept.messages[-1].text == "Refunded."


async def test_what_a_mode_was_keeping_and_what_was_spent_are_carried_over(tmp_path):
    till = Till()
    todo = tool_call("todo_write", todos=[
        {"content": "Refund the order", "status": "in_progress"},
        {"content": "Tell the customer", "status": "pending"}])
    harness, agent = make(tmp_path, [todo, tool_call("refund", order_id="4182", amount=40)],
                          till, mode="cowork", workspace=Workspace(tmp_path / "work"),
                          budget=Budget(max_steps=10))
    paused = await agent.run("Refund order 4182 and tell the customer.")
    assert paused.stop_reason == "approval" and paused.approval.step == 2
    assert [t["content"] for t in paused.approval.notebook["todos"]] == [
        "Refund the order", "Tell the customer"]

    done = tool_call("todo_write", todos=[
        {"content": "Refund the order", "status": "done"},
        {"content": "Tell the customer", "status": "done"}])
    harness, agent = make(tmp_path, [done, "Refunded, and the customer knows."], till,
                          mode="cowork", workspace=Workspace(tmp_path / "work"),
                          budget=Budget(max_steps=10))
    await harness.approvals.approve(paused.approval.id, by="maria")
    result = await agent.resume_approval(paused.approval.id)
    assert result.stop_reason == "end_turn" and result.steps == 4
    assert [t.status for t in result.todos] == ["done", "done"]
    assert result.usage.calls == paused.usage.calls + 2       # the whole run's


# --- who may ---------------------------------------------------------------------------

async def test_an_approval_belongs_to_whoever_the_run_was_for(tmp_path):
    till = Till()
    harness = Harness.local(tmp_path, provider=FakeProvider(
        [tool_call("refund", order_id="4182", amount=40)]))
    harness.approvals.self_approval = False
    ada = Agent("support", tools=till.tools, harness=harness, memory=False,
                trace=Trace(user_id="ada", tenant_id="acme"))
    paused = await ada.run("Refund my order.")
    assert (paused.approval.user_id, paused.approval.tenant_id) == ("ada", "acme")

    desk = harness.approvals
    assert [a.id for a in await desk.pending(user_id="ada")] == [paused.approval.id]
    assert await desk.pending(user_id="bob") == []
    with pytest.raises(ApprovalError, match="separation of duties"):
        await desk.approve(paused.approval.id, by="ada")
    with pytest.raises(ApprovalError, match="pass by="):
        await desk.approve(paused.approval.id, by="")
    await desk.approve(paused.approval.id, by="maria")

    bob = Agent("support", tools=till.tools, harness=harness, memory=False,
                trace=Trace(user_id="bob", tenant_id="acme"))
    with pytest.raises(ApprovalError, match="no approval"):       # as if it were not there
        await bob.resume_approval(paused.approval.id)
    other = Agent("billing", tools=till.tools, harness=harness, memory=False,
                  trace=Trace(user_id="ada", tenant_id="acme"))
    with pytest.raises(ApprovalError, match="is for the agent 'support'"):
        await other.resume_approval(paused.approval.id)
    with pytest.raises(ApprovalError, match="no approval 'apr_nope'"):
        await ada.resume_approval("apr_nope")
    assert till.refunds == []


# --- when it does not pause --------------------------------------------------------------

async def test_without_a_store_or_with_someone_to_ask_nothing_changes(tmp_path):
    till = Till()
    # No store: asking with nobody to ask is a refusal, as it always was.
    plain = Agent("support", tools=till.tools, memory=False, harness=Harness.testing(
        FakeProvider([tool_call("refund", order_id="1", amount=1), "Could not."])))
    result = await plain.run("Refund order 1.")
    assert result.stop_reason == "end_turn" and "no approver is configured" in results(result)[0]
    with pytest.raises(ConfigurationError, match="keeps no approvals"):
        await plain.resume_approval("apr_x")

    # A store, and an approver who is here: asked on the spot, no pause.
    harness, agent = make(tmp_path, [tool_call("refund", order_id="1", amount=1), "Done."], till)
    asked = []
    harness.policy.approver = lambda tool, args, reason: asked.append(tool) or True
    result = await agent.run("Refund order 1.")
    assert result.stop_reason == "end_turn" and asked == ["refund"]
    assert till.refunds == [("1", 1.0)] and await harness.approvals.list() == []

    # A sub-agent cannot be left waiting inside its lead's tool call: refused there.
    harness = Harness.local(tmp_path / "team", provider=FakeProvider([
        tool_call("delegate", agent_name="clerk", task="Refund order 2."),
        tool_call("refund", order_id="2", amount=2), "I could not.", "The clerk could not."]))
    clerk = Agent("clerk", "Refunds.", tools=till.tools, harness=harness, memory=False)
    lead = Agent("lead", subagents=[clerk], harness=harness, memory=False)
    result = await lead.run("Have order 2 refunded.")
    assert result.stop_reason == "end_turn" and await harness.approvals.list() == []
    assert till.refunds == [("1", 1.0)]


# --- stores -----------------------------------------------------------------------------

def a_record(**kw) -> Approval:
    return Approval(agent="support", run_id="run_1", task="Refund it.",
                    calls=[ApprovalCall(id="c1", tool="refund", args={"amount": 40})], **kw)


@pytest.mark.parametrize("kind", ["memory", "file", "session"])
async def test_every_store_keeps_lists_and_refuses_a_stale_write(kind, tmp_path):
    store = {"memory": MemoryApprovalStore,
             "file": lambda: FileApprovalStore(tmp_path / "approvals"),
             "session": lambda: SessionApprovalStore(InMemorySessionStore())}[kind]()
    assert await store.load("apr_missing") is None
    first = await store.create(a_record())
    second = await store.create(a_record(created=time.time() + 1))
    assert [a.id for a in await store.list()] == [second.id, first.id]

    mine, theirs = await store.load(first.id), await store.load(first.id)
    assert mine.calls[0].args == {"amount": 40}
    mine.status = "approved"
    await store.save(mine)
    theirs.status = "denied"
    with pytest.raises(SessionConflict):
        await store.save(theirs)                          # the first to save wins
    assert (await store.load(first.id)).status == "approved"

    desk = Approvals(store)
    with pytest.raises(ApprovalError, match="no approval"):
        await desk.get("apr_missing")
    with pytest.raises(ApprovalError, match="already approved"):
        await desk.deny(first.id, by="omar")
    await store.delete(first.id)
    assert await store.load(first.id) is None


CLAIM = """
import asyncio, sys
from agent_harness import Approvals, ApprovalError, FileApprovalStore, Harness

async def main():
    desk = {desk}
    try:
        await desk.claim(sys.argv[1], by=sys.argv[2])
        print("won")
    except ApprovalError:
        print("lost")

asyncio.run(main())
"""


@pytest.mark.parametrize("kind", ["files", "sqlite"])
async def test_separate_processes_racing_for_one_approval_get_it_once(kind, tmp_path):
    if kind == "files":
        desk = Approvals(FileApprovalStore(tmp_path / "approvals"))
        there = f"Approvals(FileApprovalStore({str(tmp_path / 'approvals')!r}))"
    else:
        url = f"sqlite:///{tmp_path / 'agents.db'}"
        desk = Harness(sessions=url, approvals=True).approvals
        there = f"Harness(sessions={url!r}, approvals=True).approvals"
    record = await desk.open(a_record())
    await desk.approve(record.id, by="maria")

    workers = [await asyncio.create_subprocess_exec(
        sys.executable, "-c", CLAIM.format(desk=there), record.id, f"worker-{n}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        for n in range(6)]
    said = [(await w.communicate()) for w in workers]
    answers = [out.decode().strip() for out, _ in said]
    assert sorted(answers) == ["lost"] * 5 + ["won"], [err.decode()[-300:] for _, err in said]
    held = await desk.get(record.id)
    assert held.status == "resuming" and held.claimed_by.startswith("worker-")


async def test_approvals_live_in_the_database_the_chats_are_in(tmp_path):
    till = Till()
    url = f"sqlite:///{tmp_path / 'agents.db'}"
    notified = []

    def build(script):
        harness = Harness(sessions=url, approvals=True, provider=FakeProvider(script))
        return harness, Agent("support", tools=till.tools, harness=harness, memory=False)

    harness, agent = build([tool_call("refund", order_id="4182", amount=40)])
    harness.approvals.notify = notified.append
    assert isinstance(harness.approvals.store, SessionApprovalStore)
    paused = await agent.run("Refund order 4182.")
    assert [a.id for a in notified] == [paused.approval.id]
    # Kept beside the chats, and never listed among them.
    assert [s.id for s in await harness.sessions.list(agent="support")] == [paused.session_id]
    await harness.aclose()

    harness, agent = build(["Refunded."])
    assert [a.id for a in await harness.approvals.pending()] == [paused.approval.id]
    await harness.approvals.approve(paused.approval.id, by="maria")
    result = await agent.resume_approval(paused.approval.id)
    assert result.output == "Refunded." and till.refunds == [("4182", 40.0)]
    await harness.aclose()

    # A notifier that breaks does not lose the request.
    harness, agent = build([tool_call("refund", order_id="9", amount=1)])
    harness.approvals.notify = lambda approval: 1 / 0
    again = await agent.run("Refund order 9.")
    assert again.stop_reason == "approval"
    assert (await harness.approvals.get(again.approval.id)).status == "pending"
    await harness.aclose()


def test_the_cli_lists_approves_and_resumes(tmp_path, capsys, monkeypatch):
    from agent_harness.llm_providers import PROVIDERS

    till = Till()
    harness, agent = make(tmp_path, [tool_call("refund", order_id="4182", amount=40)], till)
    paused = asyncio.run(agent.run("Refund order 4182."))
    state = ["--state", str(tmp_path)]

    assert cli.main(["approvals", *state]) == 0
    listing = capsys.readouterr().out
    assert paused.approval.id in listing and "pending" in listing and "refund(" in listing
    assert cli.main(["approvals", "show", paused.approval.id, *state]) == 0
    assert "[pending] refund(" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="needs --by"):
        cli.main(["approvals", "approve", paused.approval.id, *state])
    assert cli.main(["approvals", "approve", paused.approval.id, "--by", "maria",
                     "--note", "fine", *state]) == 0
    assert "is approved" in capsys.readouterr().out
    assert cli.main(["approvals", "show", paused.approval.id, *state]) == 0
    assert "by maria: fine" in capsys.readouterr().out
    assert cli.main(["approvals", "--status", "pending", *state]) == 0
    assert "no approvals" in capsys.readouterr().out

    provider = FakeProvider(["Refunded 40."])
    monkeypatch.setitem(PROVIDERS, "fake", lambda **kw: provider)
    monkeypatch.setattr(cli, "_agent", lambda args, harness: Agent(
        "support", tools=till.tools, harness=harness, memory=False, provider=provider))
    assert cli.main(["approvals", "resume", paused.approval.id, *state]) == 0
    assert "Refunded 40." in capsys.readouterr().out and till.refunds == [("4182", 40.0)]
