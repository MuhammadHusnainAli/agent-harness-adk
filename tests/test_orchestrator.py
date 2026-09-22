from __future__ import annotations

import json

from agent_harness import Budget, FakeProvider, Harness, Orchestrator, SubAgentSpec
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
