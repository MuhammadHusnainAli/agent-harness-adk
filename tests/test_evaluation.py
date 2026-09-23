"""Quality evaluation: golden tasks, regression scoring, proof a change helps."""

from __future__ import annotations

from agent_harness import (
    Agent,
    EvalReport,
    Evaluator,
    Expect,
    FakeProvider,
    GoldenTask,
    Harness,
    llm_judge,
    tool,
    tool_call,
)
from agent_harness.types import RunResult, ToolCall

MODEL = "claude-sonnet-5"


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return "shipped Thursday"


def agent_for(script, **kw) -> Agent:
    provider = FakeProvider(script, loop=kw.pop("loop", False))
    return Agent("support", provider=provider, model=MODEL,
                 harness=Harness.testing(provider), tools=[order_status],
                 memory=False, **kw)


def result(output="", *, tools=(), steps=1, cost=0.0, error=None, data=None):
    return RunResult(output=output, steps=steps, error=error, data=data,
                     usage={"cost_usd": cost},
                     tool_calls=[ToolCall(name=t) for t in tools])


# --- the expectations themselves ------------------------------------------

def test_contains_and_not_contains():
    expect = Expect(contains=["30-day"], not_contains=["yes, of course"])
    assert expect.check(result("Our 30-day window has passed."))[0]
    passed, why = expect.check(result("Yes, of course — refunded."))
    assert not passed and "30-day" in why


def test_matching_is_case_insensitive_unless_you_ask():
    assert Expect(contains=["SHIPPED"]).check(result("it shipped"))[0]
    assert not Expect(contains=["SHIPPED"], case_sensitive=True).check(
        result("it shipped"))[0]


def test_regex_and_exact_equality():
    assert Expect(regex=r"\b\d{4}\b").check(result("order 4182 shipped"))[0]
    assert not Expect(regex=r"\b\d{4}\b").check(result("no number here"))[0]
    assert Expect(equals="yes").check(result("Yes"))[0]
    assert not Expect(equals="yes").check(result("yes indeed"))[0]


def test_tool_expectations_look_at_the_run_not_the_text():
    assert Expect(tool_called="order_status").check(
        result("shipped", tools=["order_status"]))[0]
    passed, why = Expect(tool_called="order_status").check(result("shipped"))
    assert not passed and "never called" in why
    assert Expect(tool_not_called="wire_money").check(
        result("done", tools=["order_status"]))[0]


def test_tool_expectations_see_sub_agent_calls_too():
    parent = result("done")
    parent.children.append(result("child", tools=["order_status"]))
    assert Expect(tool_called="order_status").check(parent)[0]


def test_budgets_are_part_of_quality():
    assert not Expect(max_steps=2).check(result("fine", steps=5))[0]
    assert not Expect(max_cost_usd=0.01).check(result("fine", cost=0.5))[0]
    assert Expect(max_steps=5, max_cost_usd=1.0).check(
        result("fine", steps=3, cost=0.2))[0]


def test_an_errored_run_fails_by_default():
    passed, why = Expect().check(result("", error="BudgetExceeded: no money"))
    assert not passed and "errored" in why
    assert Expect(no_error=False).check(result("", error="x"))[0]


def test_structured_output_can_be_compared():
    from pydantic import BaseModel

    class Ticket(BaseModel):
        id: str
        priority: int

    good = result("{}", data=Ticket(id="T-1", priority=2))
    assert Expect(json_equals={"id": "T-1", "priority": 2}).check(good)[0]
    assert not Expect(json_equals={"id": "T-2", "priority": 2}).check(good)[0]


# --- running a suite -------------------------------------------------------

async def test_a_suite_runs_and_scores():
    agent = agent_for([tool_call("order_status", order_id="4182"),
                       "Order 4182 shipped Thursday.",
                       "I cannot help with that."])
    report = await Evaluator([
        GoldenTask(id="lookup", input="Where is order 4182?",
                   expect=Expect(contains=["Thursday"], tool_called="order_status"),
                   tags=["orders"]),
        GoldenTask(id="refusal", input="What is the meaning of life?",
                   expect=Expect(contains=["cannot"]), tags=["scope"]),
    ], concurrency=1).run(agent, label="v1")

    assert report.total == 2
    assert len(report.passed) == 2
    assert report.score == 1.0
    assert report.by_tag() == {"orders": 1.0, "scope": 1.0}
    assert "2/2 passed" in report.render()


async def test_a_failing_task_records_why_without_failing_the_suite():
    agent = agent_for(["the wrong answer"], loop=True)
    report = await Evaluator([
        GoldenTask(id="a", input="q", expect=Expect(contains=["the right answer"])),
        GoldenTask(id="b", input="q", expect=Expect(contains=["wrong"])),
    ], concurrency=1).run(agent)

    assert [o.id for o in report.failed] == ["a"]
    assert "missing" in report.failed[0].detail
    assert report.score == 0.5


async def test_a_crashing_agent_is_a_failed_task_not_a_crashed_suite():
    class Exploding(Agent):
        async def run(self, task, **kwargs):
            raise RuntimeError("the agent exploded")

    provider = FakeProvider(["x"])
    agent = Exploding("boom", provider=provider, model=MODEL,
                      harness=Harness.testing(provider), memory=False)
    report = await Evaluator([GoldenTask(id="t", input="go")]).run(agent)

    assert report.failed[0].error and "exploded" in report.failed[0].error
    assert report.score == 0.0


async def test_weights_move_the_headline_score():
    agent = agent_for(["right", "wrong"], loop=False)
    report = await Evaluator([
        GoldenTask(id="important", input="q1", expect=Expect(contains=["right"]),
                   weight=9.0),
        GoldenTask(id="minor", input="q2", expect=Expect(contains=["right"]),
                   weight=1.0),
    ], concurrency=1).run(agent)

    assert [o.id for o in report.passed] == ["important"]
    assert report.score > 0.85          # the heavy task carries the score


async def test_a_custom_grader_runs_after_the_expectations():
    def grader(run, task):
        return ("42" in run.output, "the number is missing")

    agent = agent_for(["the answer is 42", "no number here"], loop=False)
    report = await Evaluator([
        GoldenTask(id="has-number", input="q1", grader=grader),
        GoldenTask(id="no-number", input="q2", grader=grader),
    ], concurrency=1).run(agent)

    assert [o.id for o in report.passed] == ["has-number"]
    assert report.failed[0].detail == "the number is missing"


async def test_an_llm_judge_can_grade_open_ended_answers():
    judge_provider = FakeProvider(["PASS\nit answers the question"])
    judge = Agent("judge", provider=judge_provider, model=MODEL,
                  harness=Harness.testing(judge_provider), memory=False)

    agent = agent_for(["a thoughtful open-ended answer"])
    report = await Evaluator([
        GoldenTask(id="open", input="explain it",
                   grader=llm_judge(judge, "must actually explain the thing")),
    ]).run(agent)
    assert report.score == 1.0

    judge_provider.queue("FAIL\nit dodged the question")
    agent2 = agent_for(["waffle"])
    report2 = await Evaluator([
        GoldenTask(id="open", input="explain it",
                   grader=llm_judge(judge, "must actually explain the thing")),
    ]).run(agent2)
    assert report2.score == 0.0
    assert "dodged" in report2.failed[0].detail


async def test_tasks_can_be_selected_by_tag():
    suite = Evaluator([
        GoldenTask(id="a", input="q", tags=["fast"]),
        GoldenTask(id="b", input="q", tags=["slow"]),
    ])
    assert [t.id for t in suite.select(["fast"]).tasks] == ["a"]


def test_a_suite_loads_from_a_file(tmp_path):
    import json

    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"tasks": [
        {"id": "one", "input": "q", "expect": {"contains": ["x"]}, "tags": ["t"]},
    ]}))
    suite = Evaluator.from_file(path)
    assert len(suite) == 1 and suite.tasks[0].expect.contains == ["x"]


# --- regression scoring ----------------------------------------------------

async def test_comparing_against_a_baseline_names_what_broke(tmp_path):
    tasks = [
        GoldenTask(id="a", input="q1", expect=Expect(contains=["good"])),
        GoldenTask(id="b", input="q2", expect=Expect(contains=["good"])),
    ]
    before_agent = agent_for(["good", "bad"], loop=False)
    baseline = await Evaluator(tasks, concurrency=1).run(before_agent, label="before")
    baseline.save(tmp_path / "baseline.json")

    # A change fixes b but breaks a.
    after_agent = agent_for(["bad", "good"], loop=False)
    candidate = await Evaluator(tasks, concurrency=1).run(after_agent, label="after")

    comparison = candidate.compare(EvalReport.load(tmp_path / "baseline.json"))
    assert comparison.fixed == ["b"]
    assert comparison.regressed == ["a"]
    assert comparison.verdict == "regressed"
    assert not comparison.improved         # a regression is never an improvement
    assert "REGRESSED" in comparison.render()


async def test_a_clean_improvement_is_reported_as_one():
    tasks = [GoldenTask(id="a", input="q", expect=Expect(contains=["good"]))]
    baseline = await Evaluator(tasks).run(agent_for(["bad"]), label="before")
    candidate = await Evaluator(tasks).run(agent_for(["good"]), label="after")

    comparison = candidate.compare(baseline)
    assert comparison.verdict == "improved" and comparison.improved
    assert comparison.fixed == ["a"] and comparison.regressed == []
    assert comparison.delta > 0


async def test_added_and_removed_tasks_are_called_out():
    old = await Evaluator([GoldenTask(id="a", input="q")]).run(agent_for(["x"]))
    new = await Evaluator([GoldenTask(id="b", input="q")]).run(agent_for(["x"]))
    comparison = new.compare(old)
    assert comparison.new_tasks == ["b"] and comparison.missing_tasks == ["a"]


def test_an_unchanged_score_is_not_an_improvement():
    same = EvalReport(outcomes=[])
    assert same.compare(same).verdict == "unchanged"
