from __future__ import annotations

import json

import pytest

from agent_harness import (
    Budget,
    FakeProvider,
    Harness,
    Orchestrator,
    SubAgentSpec,
)
from agent_harness.errors import ConfigurationError
from agent_harness.orchestrator import Plan, Task

PLAN = {
    "goal": "Produce the quarterly summary",
    "definition_of_done": ["Every number is sourced", "It fits on one page"],
    "constraints": ["British English"],
    "tasks": [
        {"id": "t1", "statement": "Research the revenue figures", "depends_on": [],
         "done_when": "the figures are cited"},
        {"id": "t2", "statement": "Research the cost figures", "depends_on": [],
         "done_when": "the figures are cited"},
        {"id": "t3", "statement": "Write the report", "depends_on": ["t1", "t2"],
         "done_when": "one page, sourced"},
    ],
}


def router(accept: bool = True, *, record: list | None = None):
    """A fake model that answers according to which prompt it was handed."""

    def respond(request):
        text = request.messages[-1].text
        if record is not None:
            record.append(text[:60])
        if "costed plan" in text:
            return json.dumps(PLAN)
        if "Review this deliverable" in text:
            return json.dumps({
                "accepted": accept, "score": 0.9 if accept else 0.4,
                "problems": [] if accept else ["The cost figures are not sourced"],
                "gaps": [] if accept else ["Source the cost figures"],
                "summary": "checked",
            })
        if "Consolidate these sub-agent results" in text:
            return "THE QUARTERLY SUMMARY"
        if "specification for a brand-new sub-agent" in text:
            return json.dumps({"name": "writer_x", "description": "Writes.",
                               "instructions": "Write it.", "tools": [],
                               "tier": "fast", "max_steps": 4, "workspace": "none"})
        lines = text.splitlines()
        return f"done: {lines[1] if len(lines) > 1 else text}"[:120]

    return respond


def build(accept: bool = True, **kw) -> Orchestrator:
    provider = FakeProvider([router(accept)], loop=True)
    harness = Harness.testing(provider)
    return Orchestrator("boss", harness=harness, model="claude-sonnet-5", **kw)


async def test_planning_writes_the_acceptance_tests_first():
    plan = await build().plan("write the quarterly summary")
    assert plan.goal == "Produce the quarterly summary"
    assert plan.definition_of_done[0] == "Every number is sourced"
    assert [t.id for t in plan.tasks] == ["t1", "t2", "t3"]
    assert plan.estimate_usd > 0


def test_waves_group_independent_tasks_together():
    plan = Plan(goal="g", tasks=[
        Task(id="t1", statement="a"), Task(id="t2", statement="b"),
        Task(id="t3", statement="c", depends_on=["t1", "t2"]),
    ])
    waves = plan.waves()
    assert [[t.id for t in wave] for wave in waves] == [["t1", "t2"], ["t3"]]


def test_a_dependency_cycle_does_not_hang_the_run():
    plan = Plan(goal="g", tasks=[
        Task(id="t1", statement="a", depends_on=["t2"]),
        Task(id="t2", statement="b", depends_on=["t1"]),
    ])
    assert len(plan.waves()) == 1   # the deadlocked remainder runs, it is not dropped


async def test_the_full_job_plans_staffs_runs_consolidates_and_reviews():
    boss = build()
    result = await boss.run("write the quarterly summary")
    assert result.output == "THE QUARTERLY SUMMARY"
    assert result.ok
    plan = Plan(**result.data["plan"])
    assert all(t.status == "done" for t in plan.tasks)
    assert result.data["review"]["accepted"] is True
    # Every task was staffed from the bench where one fit.
    assert {t.agent for t in plan.tasks} >= {"research"}
    assert any(a.name == "plan.json" for a in result.artifacts)


async def test_a_rejected_deliverable_triggers_one_rework_round():
    boss = build(accept=False, max_rework=1)
    result = await boss.run("write the quarterly summary")
    plan = Plan(**result.data["plan"])
    # The reviewer's gap became a new task, and it ran.
    assert any(t.statement == "Source the cost figures" for t in plan.tasks)
    assert result.data["review"]["accepted"] is False
    assert result.stop_reason == "stopped"


async def test_staffing_reuses_the_bench_before_building_anything():
    boss = build()
    task = Task(id="t1", statement="validate the invoice against the rules")
    worker = await boss.staff(task)
    assert worker.name == "validator"
    assert task.reused is True


async def test_staffing_falls_through_to_the_factory():
    boss = build()
    task = Task(id="t9", statement="reconcile the widget ledger with the vault")
    worker = await boss.staff(task)
    assert task.reused is False
    assert worker.name == "writer_x"      # written by the factory during the run


async def test_a_registered_spec_joins_the_bench():
    spec = SubAgentSpec(name="ledger", description="Reconciles ledgers.",
                        tags=["ledger", "reconcile"])
    boss = build(subagents=[spec])
    task = Task(id="t1", statement="reconcile the ledger please")
    worker = await boss.staff(task)
    assert worker.name == "ledger" and task.reused is True


async def test_dependent_tasks_receive_what_their_dependencies_produced():
    record: list[str] = []
    provider = FakeProvider([router(True, record=record)], loop=True)
    boss = Orchestrator("boss", harness=Harness.testing(provider),
                        model="claude-sonnet-5")
    await boss.run("write the quarterly summary")
    briefs = [r for r in record if "Your task" in r or "done:" in r]
    assert briefs  # the workers were briefed, not handed the whole conversation


async def test_the_job_budget_is_enforced_across_sub_agents():
    provider = FakeProvider([router(True)], loop=True)
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, model="claude-sonnet-5",
                        budget=Budget(max_subagents=1))
    result = await boss.run("write the quarterly summary")
    report = harness.report()
    assert report["budget"]["subagents"] <= 2
    assert result is not None


async def test_the_report_shows_spend_and_concurrency():
    boss = build()
    await boss.run("write the quarterly summary")
    report = boss.report()
    assert report["budget"]["calls"] > 0
    assert report["scheduler"]["completed"] >= 3   # every task went through the pool


async def test_a_failure_while_planning_is_reported_not_raised():
    """The manager reports; it never crashes the caller — even on the first step."""
    class Exploding(FakeProvider):
        async def complete(self, req):
            raise RuntimeError("the planner exploded")

    provider = Exploding()
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, provider=provider,
                        model="claude-sonnet-5")

    result = await boss.run("do something")
    assert not result.ok
    assert "exploded" in result.error
    assert result.output == ""
    # The plan artefact is still written, so there is something to debug from.
    assert any(a.name == "plan.json" for a in result.artifacts)


async def test_the_deliverable_is_stored_as_an_artefact():
    boss = build()
    result = await boss.run("write the quarterly summary")
    names = {a.name for a in result.artifacts}
    assert {"plan.json", "deliverable.md"} <= names
    stored = boss.harness.deliverables
    assert stored.get("deliverable.md").content == "THE QUARTERLY SUMMARY"


async def test_a_task_that_misses_its_deadline_is_retried_then_given_up_on():
    import asyncio
    import json as _json

    class Slow(FakeProvider):
        async def complete(self, req):
            text = req.messages[-1].text
            if "costed plan" in text:
                return await super().complete(req)
            await asyncio.sleep(2)
            return await super().complete(req)

    one_task = {"goal": "g", "definition_of_done": [],
                "tasks": [{"id": "t1", "statement": "Research it", "depends_on": []}]}
    provider = Slow([lambda req: _json.dumps(one_task)
                     if "costed plan" in req.messages[-1].text else "too late"],
                    loop=True)
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, provider=provider,
                        model="claude-sonnet-5", task_timeout=0.1, task_retries=1,
                        review=False, max_replans=0)     # this test is about the deadline

    result = await boss.run("do the slow thing")
    plan = Plan(**result.data["plan"])
    task = plan.by_id("t1")
    assert task.status == "failed"
    assert "deadline" in task.error
    assert task.attempts == 2          # tried once, retried once, then gave up


def test_a_task_falls_back_to_its_partial_output():
    task = Task(id="t1", statement="s")
    assert task.usable_output == ""
    task.partial = "half an answer"
    assert task.usable_output == "half an answer"
    task.output = "the whole answer"
    assert task.usable_output == "the whole answer"


def test_a_dependent_task_is_told_when_its_input_is_partial():
    plan = Plan(goal="g", tasks=[
        Task(id="t1", statement="Research it", status="failed",
             partial="found two of the three figures"),
        Task(id="t2", statement="Write it up", depends_on=["t1"]),
    ])
    context = build()._context_for(plan.by_id("t2"), plan)
    assert "found two of the three figures" in context
    assert "partial, this task did not finish" in context


async def test_partial_work_from_a_failed_task_is_kept_and_consolidated():
    """A task that produced something before failing still contributes."""
    import json as _json

    one = {"goal": "g", "definition_of_done": [],
           "tasks": [{"id": "t1", "statement": "Extract the totals",
                      "depends_on": []}]}

    def respond(request):
        text = request.messages[-1].text
        if "costed plan" in text:
            return _json.dumps(one)
        if "Consolidate these sub-agent results" in text:
            assert "PARTIAL" in text          # the manager was told it is partial
            assert "I found 2 of 3 totals" in text
            return "consolidated from partial work"
        return "I found 2 of 3 totals but could not finish."

    provider = FakeProvider([respond], loop=True)
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, provider=provider,
                        model="claude-sonnet-5", review=False, task_retries=0)
    # A worker with an output contract it cannot satisfy: it produces prose,
    # fails the contract, and the prose is what survives.
    boss.bench.register(SubAgentSpec(
        name="extractor", description="Extracts totals from documents.",
        tags=["extract", "totals"],
        output_schema={"type": "object", "properties": {"total": {"type": "number"}},
                       "required": ["total"]},
        max_steps=3,
    ))

    result = await boss.run("get the totals")
    task = Plan(**result.data["plan"]).by_id("t1")

    assert task.status == "failed"
    assert "OutputContract" in task.error
    assert "2 of 3 totals" in task.partial      # partial delivery kept
    assert result.output == "consolidated from partial work"


async def test_an_unexpected_exception_marks_the_task_failed_not_running():
    """A crash inside a worker must surface as a failed task, never as silence."""
    import json as _json

    one = {"goal": "g", "definition_of_done": [],
           "tasks": [{"id": "t1", "statement": "Research it", "depends_on": []}]}

    class Exploding(FakeProvider):
        async def complete(self, req):
            if "costed plan" in req.messages[-1].text:
                return await super().complete(req)
            raise RuntimeError("something nobody catches")

    provider = Exploding([lambda req: _json.dumps(one)], loop=True)
    harness = Harness.testing(provider)
    boss = Orchestrator("boss", harness=harness, provider=provider,
                        model="claude-sonnet-5", review=False, task_retries=0)

    result = await boss.run("do it")
    task = Plan(**result.data["plan"]).by_id("t1")
    assert task.status == "failed"
    assert "RuntimeError" in task.error


async def test_a_job_reports_everything_it_cost_not_only_the_sub_agents():
    boss = build()
    result = await boss.run("Write the Q3 summary")
    spent = boss.harness.guard.usage.model_copy()       # as it stood after this job
    work = sum(c.cost_usd for c in result.children)

    # The whole job: planning, the factory, consolidation and review included.
    assert result.cost_usd == round(spent.cost_usd, 8) > work > 0
    # `usage` is the manager's own; with the sub-agents' it is everything.
    own, children = result.usage, [c.usage for c in result.children]
    assert own.calls + sum(u.calls for u in children) == spent.calls
    assert own.total_tokens + sum(u.total_tokens for u in children) == spent.total_tokens
    assert own.calls >= 3 and own.cost_usd > 0             # plan, consolidate, review

    spend = result.spend
    assert spend["total_usd"] == round(spent.cost_usd, 6) == round(result.cost_usd, 6)
    assert spend["work_usd"] == round(work, 6)
    assert spend["overhead_usd"] == round(spend["total_usd"] - spend["work_usd"], 6) > 0
    assert (spend["calls"], spend["tokens"]) == (spent.calls, spent.total_tokens)
    assert {"boss", "boss.planner", "boss.reviewer"} <= set(spend["by_agent"])
    assert sum(a["calls"] for a in spend["by_agent"].values()) == spent.calls
    assert round(sum(a["cost_usd"] for a in spend["by_agent"].values()), 5) == round(
        spend["total_usd"], 5)

    # A second job on the same harness reports its own spend, not the running total.
    again = await boss.run("Write the Q4 summary")
    assert again.spend["calls"] == boss.harness.guard.usage.calls - spend["calls"]
    assert again.cost_usd == round(boss.harness.guard.usage.cost_usd - spent.cost_usd, 8)


async def test_a_rework_round_and_a_retried_task_are_both_counted():
    boss = build(accept=False, max_rework=1)
    result = await boss.run("Write the Q3 summary")
    assert result.spend["calls"] == boss.harness.guard.usage.calls
    assert result.cost_usd == round(boss.harness.guard.usage.cost_usd, 8)
    assert result.spend["by_agent"]["boss.reviewer"]["calls"] == 2      # reviewed twice


async def test_a_sub_agents_own_budget_applies_inside_a_job():
    capped = SubAgentSpec(name="research", description="Research the revenue and cost figures",
                          instructions="Research.", budget=Budget(max_output_tokens=1))
    boss = build(subagents=[capped])
    result = await boss.run("Write the Q3 summary")
    stopped = [c for c in result.children if c.stop_reason == "budget"]
    assert stopped and all(c.agent == "research" for c in stopped)
    assert result.spend["calls"] == boss.harness.guard.usage.calls     # still all counted


# --- hand-backs checked, and tasks planned again when they come back short ----------

def scripted(*, plan=None, check=None, replan=None, work=None, seen=None):
    """A fake model for every prompt the orchestrator writes. `check`, `replan`
    and `work` are functions of the prompt text; `seen` collects each prompt."""

    def respond(request):
        text = request.messages[-1].text
        if seen is not None:
            seen.append(text)
        if "costed plan" in text:
            return json.dumps(plan or PLAN)
        if "Check this hand-back" in text:
            return check(text) if check else json.dumps({"met": True})
        if "came back short. Plan what to do instead" in text:
            return replan(text) if replan else json.dumps({"tasks": []})
        if "Review this deliverable" in text:
            return json.dumps({"accepted": True, "score": 0.9, "summary": "ok"})
        if "Consolidate these sub-agent results" in text:
            return "CONSOLIDATED:\n" + text.split("RESULTS", 1)[1]
        if "specification for a brand-new sub-agent" in text:
            return json.dumps({"name": "writer_x", "description": "Writes.",
                               "instructions": "Write it.", "tools": [],
                               "tier": "fast", "max_steps": 4, "workspace": "none"})
        return work(text) if work else "the work"

    return respond


def orchestrator(**kw) -> Orchestrator:
    options = {k: kw.pop(k) for k in list(kw) if k in (
        "check_tasks", "max_replans", "task_retries", "max_rework", "review")}
    harness = Harness.testing(FakeProvider([scripted(**kw)], loop=True))
    return Orchestrator("boss", harness=harness, model="claude-sonnet-5", **options)


ONE = {"goal": "Report the figures", "definition_of_done": ["Figures are cited"],
       "tasks": [{"id": "t1", "statement": "Research the revenue figures",
                  "depends_on": [], "done_when": "every figure has a source"},
                 {"id": "t2", "statement": "Write the report", "depends_on": ["t1"],
                  "done_when": ""}]}


async def test_a_hand_back_is_checked_and_a_short_one_is_sent_back_with_what_it_lacked():
    seen: list[str] = []
    verdicts = iter([{"met": False, "missing": "the Q3 figure has no source"},
                     {"met": True}])
    boss = orchestrator(plan=ONE, seen=seen, check=lambda t: json.dumps(next(verdicts)),
                        work=lambda t: "revenue 4.2M (source: ledger)"
                        if "last attempt was missing" in t else "revenue 4.2M")
    result = await boss.run("Report the figures")
    plan = Plan(**result.data["plan"])
    first = plan.by_id("t1")

    assert first.status == "done" and first.attempts == 2 and first.checked is True
    assert first.output == "revenue 4.2M (source: ledger)" and first.missing == ""
    checks = [t for t in seen if "Check this hand-back" in t]
    assert len(checks) == 2 and "every figure has a source" in checks[0]
    assert "HAND-BACK\nrevenue 4.2M" in checks[0]
    retry = [t for t in seen if "What the last attempt was missing" in t]
    assert len(retry) == 1 and "the Q3 figure has no source" in retry[0]
    # A task with nothing to be checked against is not checked.
    assert plan.by_id("t2").checked is None and plan.by_id("t2").status == "done"
    assert result.spend["by_agent"]["boss.checker"]["calls"] == 2


async def test_a_task_that_stays_short_is_planned_again_by_another_route():
    seen: list[str] = []

    def check(text):
        return json.dumps({"met": "ledger extract" in text,
                           "missing": "no figure has a source"})

    def replan(text):
        return json.dumps({"tasks": [
            {"id": "a", "statement": "Pull the ledger extract", "depends_on": []},
            {"id": "b", "statement": "Cite each figure from the extract",
             "depends_on": ["a"], "done_when": "every figure has a source"}]})

    def work(text):
        if "Pull the ledger extract" in text or "Cite each figure" in text:
            return "from the ledger extract: revenue 4.2M"
        return "revenue is about 4M, I think"

    boss = orchestrator(plan=ONE, seen=seen, check=check, replan=replan, work=work,
                        task_retries=1)
    result = await boss.run("Report the figures")
    plan = Plan(**result.data["plan"])
    by = {t.id: t for t in plan.tasks}

    # The first route failed its check twice, and was planned again.
    assert by["t1"].status == "replaced" and by["t1"].attempts == 2
    assert by["t1"].error == "short of done-when: no figure has a source"
    assert by["t1"].partial == "revenue is about 4M, I think"          # kept
    assert [t.id for t in plan.tasks] == ["t1", "t2", "t1.1", "t1.2"]
    assert by["t1.1"].replaces == by["t1.2"].replaces == "t1"
    assert by["t1.2"].depends_on == ["t1.1"]
    # What waited for the failed task now waits for what replaced it.
    assert by["t2"].depends_on == ["t1.2"] and by["t2"].status == "done"
    assert all(by[i].status == "done" for i in ("t1.1", "t1.2", "t2"))

    asked = next(t for t in seen if "came back short" in t)
    assert "Research the revenue figures" in asked and "no figure has a source" in asked
    assert "revenue is about 4M, I think" in asked
    # The new route is told what the old one produced, and the report builds on it.
    brief = next(t for t in seen if t.startswith("## Your task\nPull the ledger extract"))
    assert "An earlier attempt at this" in brief and "revenue is about 4M" in brief
    report = next(t for t in seen if t.startswith("## Your task\nWrite the report"))
    assert "Result of t1.2" in report and "from the ledger extract" in report
    # The abandoned attempt is not consolidated as if it were a finding.
    assert "revenue is about 4M" not in result.output
    assert "from the ledger extract" in result.output
    assert result.stop_reason == "end_turn"
    assert result.spend["by_agent"]["boss.replanner"]["calls"] == 1
    assert result.spend["calls"] == boss.harness.guard.usage.calls


async def test_second_routes_are_limited_and_a_failure_without_one_blocks_what_follows():
    def check(text):
        return json.dumps({"met": False, "missing": "still no source"})

    def replan(text):
        return json.dumps({"tasks": [{"id": "a", "statement": "Try the archive"}]})

    # No second routes allowed: the failure stands and its dependants are blocked.
    boss = orchestrator(plan=ONE, check=check, replan=replan, max_replans=0, task_retries=0)
    plan = Plan(**(await boss.run("Report the figures")).data["plan"])
    assert [(t.id, t.status) for t in plan.tasks] == [("t1", "failed"), ("t2", "blocked")]

    # One allowed. The replacement fails its check too — and is not planned a third way.
    seen: list[str] = []
    boss = orchestrator(plan=ONE, check=check, replan=replan, seen=seen, task_retries=0)
    result = await boss.run("Report the figures")
    plan = Plan(**result.data["plan"])
    assert [(t.id, t.status) for t in plan.tasks] == [
        ("t1", "replaced"), ("t2", "blocked"), ("t1.1", "failed")]
    assert sum("Plan what to do instead" in t for t in seen) == 1
    assert "[PARTIAL — this task did not finish]" in result.output      # said to be partial

    # A planner with no better route leaves the failure as it was.
    boss = orchestrator(plan=ONE, check=check, task_retries=0)
    plan = Plan(**(await boss.run("Report the figures")).data["plan"])
    assert [(t.id, t.status) for t in plan.tasks] == [("t1", "failed"), ("t2", "blocked")]


async def test_a_task_that_errors_or_times_out_is_planned_again_too():
    def replan(text):
        return json.dumps({"tasks": [{"id": "a", "statement": "Use the cached figures"}]})

    def work(text):
        if "Research the revenue figures" in text:
            raise RuntimeError("the ledger API is down")
        return "figures from the cache"

    boss = orchestrator(plan=ONE, replan=replan, work=work, task_retries=0)
    result = await boss.run("Report the figures")
    by = {t.id: t for t in Plan(**result.data["plan"]).tasks}
    assert by["t1"].status == "replaced" and "ledger API is down" in by["t1"].error
    assert by["t1.1"].status == "done" and by["t2"].status == "done"
    assert by["t2"].depends_on == ["t1.1"]


async def test_checking_can_be_turned_off_and_an_unusable_check_does_not_hold_work_back():
    seen: list[str] = []
    boss = orchestrator(plan=ONE, seen=seen, check_tasks=False)
    plan = Plan(**(await boss.run("Report the figures")).data["plan"])
    assert not any("Check this hand-back" in t for t in seen)
    assert plan.by_id("t1").checked is None and plan.by_id("t1").status == "done"

    boss = orchestrator(plan=ONE, check=lambda t: "looks fine to me")     # not a verdict
    plan = Plan(**(await boss.run("Report the figures")).data["plan"])
    assert plan.by_id("t1").status == "done" and plan.by_id("t1").checked is False


# --- the estimate, and the budget it is held against -----------------------------------

async def test_the_plan_says_how_wide_it_goes_and_learns_what_tasks_cost():
    boss = orchestrator()
    plan = await boss.plan("Write the Q3 summary")
    assert plan.parallelism == 2                       # t1 and t2 together, then t3
    assert plan.estimate_usd > 0
    narrow = Orchestrator("boss", model="claude-sonnet-5", max_concurrency=1,
                          harness=Harness.testing(FakeProvider([scripted()], loop=True)))
    assert (await narrow.plan("Write the Q3 summary")).parallelism == 1
    assert boss.estimate(plan, tokens_per_task=12_000) == pytest.approx(
        2 * boss.estimate(plan, tokens_per_task=6_000), rel=0.01)

    # Before any task has run the estimate is a guess from the model's prices…
    guessed = plan.estimate_usd
    result = await boss.run("Write the Q3 summary")
    paid = [c.cost_usd for c in result.children]
    # …and after three have, it is what tasks here actually cost.
    learned = (await boss.plan("Write the Q4 summary")).estimate_usd
    assert learned < guessed / 10
    assert learned == pytest.approx(sum(paid) / len(paid) * (3 + 2 + 0.2 * 3), rel=0.05)


async def test_work_that_will_not_fit_the_budget_is_warned_about_or_not_started():
    def harness_with(budget):
        return Harness.testing(FakeProvider([scripted()], loop=True), budget=budget)

    # By default it says so and carries on.
    boss = Orchestrator("boss", model="claude-sonnet-5",
                        harness=harness_with(Budget(max_usd=0.01)))
    result = await boss.run("Write the Q3 summary")
    assert result.stop_reason == "end_turn" and len(result.children) == 3
    assert "would cost about $" in result.warnings[0] and "the estimate" in result.warnings[0]
    assert "the budget has $0.00" in result.warnings[0]

    # Told to stop, it starts nothing it cannot pay for — and says what it would have done.
    seen: list[str] = []
    strict = Orchestrator(
        "boss", model="claude-sonnet-5", on_over_estimate="stop",
        harness=Harness.testing(FakeProvider([scripted(seen=seen)], loop=True),
                                budget=Budget(max_usd=0.01)))
    result = await strict.run("Write the Q3 summary")
    plan = Plan(**result.data["plan"])
    assert result.stop_reason == "budget" and result.budget_exceeded == "estimate"
    assert result.error is None and result.children == []
    assert [t.status for t in plan.tasks] == ["skipped"] * 3
    assert plan.tasks[0].error.startswith("not started: the 3 tasks still to run")
    assert not any(t.startswith("## Your task") for t in seen)      # no work was begun
    assert result.spend["by_agent"].keys() == {"boss.planner"}      # only the plan was paid for

    # No spend ceiling, or told not to look: no gate.
    free = Orchestrator("boss", model="claude-sonnet-5", on_over_estimate="stop",
                        harness=harness_with(Budget(max_steps=500)))
    assert (await free.run("Write the Q3 summary")).warnings == []
    blind = Orchestrator("boss", model="claude-sonnet-5", on_over_estimate="ignore",
                         harness=harness_with(Budget(max_usd=5.0)))
    assert (await blind.run("Write the Q3 summary")).warnings == []
    with pytest.raises(ValueError, match="on_over_estimate"):
        Orchestrator("boss", on_over_estimate="maybe")


async def test_a_job_stops_part_way_when_its_own_tasks_show_the_rest_will_not_fit():
    def work(text):
        return "figures " * 4000                      # every task is expensive

    # What the first two tasks cost, measured on a job with no ceiling.
    free = orchestrator(work=work, check_tasks=False)
    measured = await free.run("Write the Q3 summary")
    each = measured.children[0].cost_usd
    planning = measured.spend["by_agent"]["boss.planner"]["cost_usd"]

    # Enough for the plan and the first two tasks, not for the third.
    seen: list[str] = []
    harness = Harness.testing(FakeProvider([scripted(work=work, seen=seen)], loop=True),
                              budget=Budget(max_usd=planning + each * 2.6))
    boss = Orchestrator("boss", harness=harness, model="claude-sonnet-5",
                        on_over_estimate="stop", check_tasks=False, tokens_per_task=1)
    result = await boss.run("Write the Q3 summary")
    plan = Plan(**result.data["plan"])
    assert [(t.id, t.status) for t in plan.tasks] == [
        ("t1", "done"), ("t2", "done"), ("t3", "skipped")]
    assert "its tasks are costing $" in plan.by_id("t3").error
    assert result.stop_reason == "budget" and result.budget_exceeded == "estimate"
    # What was finished is still handed back.
    assert len(result.children) == 2 and "figures" in result.output
    assert not any(t.startswith("## Your task\nWrite the report") for t in seen)
    assert "its tasks are costing" in result.warnings[0]


# --- the review ----------------------------------------------------------------------

def reviewed_by(answers, **kw):
    """An orchestrator whose critics answer, in order, with `answers`."""
    queue = list(answers)
    seen: list[str] = []

    def respond(request):
        text = request.messages[-1].text
        if "Review this deliverable" in text:
            seen.append(text)
            answer = queue.pop(0) if queue else {"accepted": True, "score": 1.0}
            return answer if isinstance(answer, str) else json.dumps(answer)
        return scripted()(request)

    harness = Harness.testing(FakeProvider([respond], loop=True))
    return Orchestrator("boss", harness=harness, model="claude-sonnet-5",
                        check_tasks=False, **kw), seen


async def test_several_critics_review_on_their_own_and_all_must_accept():
    boss, seen = reviewed_by([
        {"accepted": True, "score": 0.9, "summary": "figures check out"},
        {"accepted": False, "score": 0.4, "summary": "too long",
         "problems": ["It runs to two pages"], "gaps": ["Cut it to one page"]},
        {"accepted": False, "score": 0.5, "summary": "length",
         "problems": ["it runs to  two pages"], "gaps": ["Cut it to one page",
                                                         "Add the source for Q3"]},
    ], critics=["the accuracy of every figure", "length and format", "sourcing"],
        max_rework=0)
    result = await boss.run("Write the Q3 summary")
    review = result.data["review"]

    assert len(seen) == 3
    assert "one thing in particular: the accuracy of every figure" in seen[0]
    assert "one thing in particular: sourcing" in seen[2]
    assert review["accepted"] is False and result.stop_reason == "stopped"
    assert review["score"] == 0.6
    # What two critics both said is said once.
    assert review["problems"] == ["It runs to two pages"]
    assert review["gaps"] == ["Cut it to one page", "Add the source for Q3"]
    assert [(v["lens"], v["accepted"]) for v in review["verdicts"]] == [
        ("the accuracy of every figure", True), ("length and format", False),
        ("sourcing", False)]
    assert result.spend["by_agent"]["boss.reviewer"]["calls"] == 3

    # With a majority rule, two of three is enough.
    boss, _ = reviewed_by([{"accepted": True, "score": 1.0}, {"accepted": True, "score": 1.0},
                           {"accepted": False, "score": 0.2, "gaps": ["Nitpick"]}],
                          critics=3, accept="majority")
    result = await boss.run("Write the Q3 summary")
    assert result.data["review"]["accepted"] is True and result.stop_reason == "end_turn"
    assert result.data["review"]["gaps"] == []

    # Every gap any critic named becomes a task in the rework round.
    boss, _ = reviewed_by([{"accepted": False, "gaps": ["Cut it to one page"]},
                           {"accepted": False, "gaps": ["Add the source for Q3"]}],
                          critics=2, max_rework=1)
    result = await boss.run("Write the Q3 summary")
    statements = [t["statement"] for t in result.data["plan"]["tasks"]]
    assert statements[-2:] == ["Cut it to one page", "Add the source for Q3"]
    assert result.data["review"]["accepted"] is True                  # the second pass
    with pytest.raises(ValueError, match="accept is"):
        Orchestrator("boss", accept="most")


async def test_a_review_that_says_nothing_is_not_an_approval():
    # A critic that returns no verdict is asked once more, and then left out.
    boss, seen = reviewed_by(["I think it is probably fine?", "Yes, fine.",
                              {"accepted": True, "score": 0.8, "summary": "good"}],
                             critics=2)
    result = await boss.run("Write the Q3 summary")
    review = result.data["review"]
    assert len(seen) == 3 and review["accepted"] is True and review["reviewed"] is True
    assert [v["critic"] for v in review["verdicts"]] == [2]

    # None of them gave a verdict: delivered, unreviewed, and said to be.
    boss, seen = reviewed_by(["fine", "fine", "fine", "fine"], critics=1, max_rework=2)
    result = await boss.run("Write the Q3 summary")
    review = result.data["review"]
    assert review["accepted"] is False and review["reviewed"] is False
    assert result.output and result.error is None and result.stop_reason == "stopped"
    assert result.warnings == ["the deliverable was not reviewed: no critic returned a "
                               "usable verdict"]
    assert len(seen) == 2                    # asked twice, and no rework on no findings

    # Review switched off is a choice, and reads as one.
    boss, seen = reviewed_by([], review=False)
    result = await boss.run("Write the Q3 summary")
    assert seen == [] and result.stop_reason == "end_turn" and result.warnings == []
    assert result.data["review"]["summary"] == "review disabled"


# --- a job that outlives its process -----------------------------------------------------

def process(database, *, hold=None, seen=None, **kw):
    """What one process builds: a harness on the shared database, and the manager.
    `hold` names a prompt at which this process's model stops answering."""
    import asyncio

    stuck = asyncio.Event()
    inner = scripted(seen=seen)

    class Model(FakeProvider):
        async def complete(self, req):
            if hold and hold in req.messages[-1].text:
                stuck.set()
                await asyncio.Event().wait()             # until the process is killed
            return await super().complete(req)

    harness = Harness.testing(Model([inner], loop=True), sessions=database)
    boss = Orchestrator("boss", harness=harness, model="claude-sonnet-5",
                        check_tasks=False, **kw)
    return boss, stuck


async def killed(boss, stuck, request, job_id):
    """Run a job until its model stops answering, then kill the process."""
    import asyncio

    running = asyncio.ensure_future(boss.run(request, job_id=job_id))
    await asyncio.wait_for(stuck.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await boss.harness.aclose()


async def test_a_job_cut_short_is_resumed_from_the_last_task_that_finished(tmp_path):
    database = f"sqlite:///{tmp_path / 'jobs.db'}"
    first: list[str] = []
    boss, stuck = process(database, hold="## Your task\nWrite the report", seen=first)
    await killed(boss, stuck, "Write the Q3 summary", "job_q3")
    spent_before = boss.harness.guard.usage.model_copy()

    # --- another process, later -----------------------------------------------------
    second: list[str] = []
    boss, _ = process(database, seen=second)
    standing = await boss.job("job_q3")
    assert standing["status"] == "running" and standing["stage"] == "executing"
    assert [(t["id"], t["status"]) for t in standing["plan"]["tasks"]] == [
        ("t1", "done"), ("t2", "done"), ("t3", "pending")]

    result = await boss.resume("job_q3")
    assert result.run_id == "job_q3" and result.stop_reason == "end_turn"
    assert result.output.startswith("CONSOLIDATED:")
    # The two finished tasks were not done again, and the plan was not made again.
    briefs = [t.split("\n")[1] for t in second if t.startswith("## Your task")]
    assert briefs == ["Write the report"]
    assert not any("costed plan" in t for t in second)
    # The third task was given what the first two produced before the crash.
    report = next(t for t in second if t.startswith("## Your task\nWrite the report"))
    assert "Result of t1" in report and "Result of t2" in report
    assert [c.agent for c in result.children] and len(result.children) == 3

    # What it cost is what both processes spent — minus the call that was cut off.
    spent_after = boss.harness.guard.usage
    assert result.spend["calls"] == spent_before.calls + spent_after.calls
    assert result.cost_usd == pytest.approx(spent_before.cost_usd + spent_after.cost_usd)
    assert {"boss.planner", "boss.reviewer", "boss"} <= set(result.spend["by_agent"])

    done = await boss.job("job_q3")
    assert done["status"] == "done" and done["stage"] == "finished"
    # Resumed again, a finished job returns what it produced and runs nothing.
    before = len(second)
    again = await boss.resume("job_q3")
    assert again.output == result.output and len(second) == before
    assert again.spend["total_usd"] == result.spend["total_usd"]
    with pytest.raises(ConfigurationError, match="no job 'job_nope'"):
        await boss.resume("job_nope")
    await boss.harness.aclose()


async def test_a_job_cut_short_before_its_review_is_not_consolidated_twice(tmp_path):
    database = f"sqlite:///{tmp_path / 'jobs.db'}"
    boss, stuck = process(database, hold="Review this deliverable")
    await killed(boss, stuck, "Write the Q3 summary", "job_q3")

    seen: list[str] = []
    boss, _ = process(database, seen=seen)
    assert (await boss.job("job_q3"))["stage"] == "consolidated"
    result = await boss.resume("job_q3")
    assert result.stop_reason == "end_turn" and result.output.startswith("CONSOLIDATED:")
    assert [t[:24] for t in seen] == ["Review this deliverable "]      # only the review
    await boss.harness.aclose()


async def test_two_processes_cannot_both_run_one_job(tmp_path):
    import asyncio

    database = f"sqlite:///{tmp_path / 'jobs.db'}"
    release = asyncio.Event()
    inner = scripted()

    class Slow(FakeProvider):
        async def complete(self, req):
            if "## Your task\nWrite the report" in req.messages[-1].text:
                await release.wait()
            return await super().complete(req)

    harness = Harness.testing(Slow([inner], loop=True), sessions=database)
    first = Orchestrator("boss", harness=harness, model="claude-sonnet-5", check_tasks=False)
    running = asyncio.ensure_future(first.run("Write the Q3 summary", job_id="job_q3"))
    for _ in range(200):                                   # until it is on its last task
        await asyncio.sleep(0.01)
        try:
            if (await first.job("job_q3"))["stage"] == "executing":
                break
        except ConfigurationError:
            pass

    # A second process takes the job, thinking the first is dead.
    second, _ = process(database)
    taken = await second.resume("job_q3")
    assert taken.stop_reason == "end_turn"

    # The first finds out at its next save, and stops rather than finish it twice.
    release.set()
    lost = await running
    assert lost.stop_reason == "error"
    assert "is being run by another process" in lost.error
    assert (await second.job("job_q3"))["status"] == "done"
    await harness.aclose()
    await second.harness.aclose()


async def test_a_job_can_be_left_unsaved_and_a_store_that_fails_does_not_stop_it(tmp_path):
    boss = orchestrator(check_tasks=False)
    boss.persist = False
    result = await boss.run("Write the Q3 summary")
    with pytest.raises(ConfigurationError, match="no job"):
        await boss.job(result.run_id)

    boss = orchestrator(check_tasks=False)

    async def broken(session):
        raise OSError("disk full")

    boss.harness.sessions.save = broken
    result = await boss.run("Write the Q3 summary")
    assert result.stop_reason == "end_turn" and result.output
    assert result.warnings == ["the job could not be saved (OSError); it will not be "
                               "resumable from here"]
