"""Stop control: a human is always in charge."""

from __future__ import annotations

import asyncio

import pytest

from agent_harness import (
    Agent,
    FakeProvider,
    Harness,
    Orchestrator,
    StopController,
    SubAgentSpec,
    tool,
    tool_call,
)
from agent_harness.errors import StopRequested

MODEL = "claude-sonnet-5"


@tool
def ping(label: str = "x") -> str:
    """Do a small thing.

    Args:
        label: which call this is
    """
    return f"pong {label}"


def test_stop_is_recorded_and_idempotent():
    control = StopController()
    assert not control.stopped and control.may_start()

    first = control.stop("customer withdrew the request", requested_by="ops")
    control.stop("a different reason")          # the first reason is the one that sticks

    assert control.stopped and not control.may_start()
    assert first.reason == "customer withdrew the request"
    assert control.state.reason == "customer withdrew the request"
    assert control.state.requested_by == "ops"
    assert control.state.mode == "drain" and control.draining


def test_check_raises_only_once_stopped():
    control = StopController()
    control.check("step 1")                     # no stop in force: silent
    control.stop("enough")
    with pytest.raises(StopRequested, match="enough"):
        control.check("step 2")


def test_resume_clears_the_stop():
    control = StopController()
    control.stop("pause")
    control.resume()
    assert not control.stopped
    control.check()


async def test_listeners_are_told_and_a_bad_one_cannot_block_the_stop():
    control = StopController()
    seen: list[str] = []

    control.on_stop(lambda state: (_ for _ in ()).throw(RuntimeError("boom")))
    control.on_stop(lambda state: seen.append(state.reason))

    control.stop("shutting down")
    assert seen == ["shutting down"]             # the failing listener did not stop it


async def test_waiting_for_a_stop():
    control = StopController()
    assert await control.wait(timeout=0.01) is False

    async def later():
        await asyncio.sleep(0.01)
        control.stop("later")

    asyncio.create_task(later())
    assert await control.wait(timeout=1.0) is True


async def test_abort_cancels_what_is_in_flight():
    control = StopController()
    started = asyncio.Event()

    async def long_job():
        started.set()
        await asyncio.sleep(5)

    task = asyncio.create_task(long_job())
    control.track("job-1", task)
    await started.wait()

    control.abort("pull the plug")
    await asyncio.sleep(0)
    assert task.cancelled() or task.cancelling()
    assert control.state.mode == "abort"


async def test_drain_waits_for_work_already_running():
    control = StopController()

    async def short_job():
        await asyncio.sleep(0.02)
        return "finished"

    task = asyncio.create_task(short_job())
    control.track("job-1", task)
    control.stop("no new work please")

    assert await control.drain(timeout=1.0) is True
    assert task.result() == "finished"           # in-flight work was allowed to land


async def test_a_running_agent_stops_at_the_next_step_boundary():
    """A stop lands between steps, not in the middle of one."""
    provider = FakeProvider([tool_call("ping", label="1"),
                             tool_call("ping", label="2"),
                             "all done"])
    harness = Harness.testing(provider)
    agent = Agent("worker", provider=provider, model=MODEL, harness=harness,
                  tools=[ping], memory=False)

    @harness.hooks.on("post_tool")
    def stop_after_the_first_tool(ctx) -> None:
        harness.control.stop("a human pulled the switch")

    result = await agent.run("do the work")
    assert not result.ok
    assert "StopRequested" in result.error
    assert "a human pulled the switch" in result.error
    # The first step completed and its work is kept — this is a stop, not a crash.
    assert result.steps == 1
    assert any("pong 1" in b.content for m in result.messages for b in m.content
               if getattr(b, "type", "") == "tool_result")


async def test_a_stopped_run_does_not_start_new_sub_agents():
    provider = FakeProvider([
        tool_call("delegate", agent_name="worker", task="go"),
        "I could not start it.",
    ])
    harness = Harness.testing(provider)
    agent = Agent("manager", provider=provider, model=MODEL, harness=harness,
                  memory=False,
                  subagents=[SubAgentSpec(name="worker", description="Works.")])

    harness.control.stop("budget review", requested_by="finance")
    result = await agent.run("delegate it")

    # The loop refuses at the first boundary; no sub-agent was ever started.
    assert not result.ok and "StopRequested" in result.error
    assert result.children == []
    assert provider.requests == []


async def test_a_stop_mid_job_skips_the_tasks_that_have_not_started():
    import json

    plan = {
        "goal": "Two things",
        "definition_of_done": ["both done"],
        "tasks": [
            {"id": "t1", "statement": "Research the first thing", "depends_on": []},
            {"id": "t2", "statement": "Write the report", "depends_on": ["t1"]},
        ],
    }

    def respond(request):
        text = request.messages[-1].text
        if "costed plan" in text:
            return json.dumps(plan)
        if "Review this deliverable" in text:
            return json.dumps({"accepted": True, "score": 1.0, "summary": "ok"})
        if "Consolidate these sub-agent results" in text:
            return "the consolidated answer"
        return "task finished"

    provider = FakeProvider([respond], loop=True)
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, model=MODEL)

    calls = {"n": 0}

    @harness.hooks.on("run_end")
    def stop_after_the_first_task(ctx) -> None:
        calls["n"] += 1
        if calls["n"] == 2:            # the planner, then the first worker
            harness.control.stop("stop after the first wave")

    result = await boss.run("do two things")
    from agent_harness import Plan

    done = Plan(**result.data["plan"])
    statuses = {t.id: t.status for t in done.tasks}
    assert statuses["t2"] == "skipped"
    assert done.by_id("t2").error == "the run was stopped"


def test_the_harness_reports_who_stopped_it_and_why():
    harness = Harness.testing(FakeProvider())
    harness.stop("month-end freeze", requested_by="platform")

    report = harness.report()["control"]
    assert report["stopped"] is True
    assert report["reason"] == "month-end freeze"
    assert report["requested_by"] == "platform"
    # The stop itself is an audited action.
    assert [e.action for e in harness.audit.entries] == ["stop"]
