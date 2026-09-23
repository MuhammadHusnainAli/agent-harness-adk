"""Quality evaluation: golden tasks, regression scoring, proof a change is an improvement.

    tasks = [
        GoldenTask(id="refund-window", input="Can I refund a 40-day-old order?",
                   expect=Expect(contains=["30-day", "no"], not_contains=["yes, "])),
        GoldenTask(id="uses-lookup", input="Where is order 4182?",
                   expect=Expect(tool_called="order_status")),
    ]

    report = await Evaluator(tasks).run(agent)
    report.save("baseline.json")

    # after a change:
    after = await Evaluator(tasks).run(agent)
    print(after.compare(EvalReport.load("baseline.json")).render())

Graders are ordinary predicates over the run result, so a check can look at the
text, the tools that were called, the structured output, the cost or the number
of steps. An LLM judge is available when the answer is genuinely open-ended, but
reach for it last: it costs money and it is the least reliable grader you have.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .types import RunResult

__all__ = ["Expect", "GoldenTask", "TaskOutcome", "EvalReport", "Comparison",
           "Evaluator", "llm_judge"]

Grader = Callable[[RunResult, "GoldenTask"], "bool | tuple[bool, str] | Awaitable[Any]"]


class Expect(BaseModel):
    """What a good answer looks like. Every field set must hold."""

    model_config = ConfigDict(extra="allow")

    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: str | None = None
    equals: str | None = None
    json_equals: dict[str, Any] | None = None
    tool_called: str | None = None
    tools_called: list[str] = Field(default_factory=list)
    tool_not_called: str | None = None
    max_steps: int | None = None
    max_cost_usd: float | None = None
    no_error: bool = True
    case_sensitive: bool = False

    def check(self, result: RunResult) -> tuple[bool, str]:
        """Returns (passed, why it failed)."""
        text = result.output if self.case_sensitive else result.output.lower()

        def norm(value: str) -> str:
            return value if self.case_sensitive else value.lower()

        if self.no_error and result.error:
            return False, f"the run errored: {result.error}"
        for needle in self.contains:
            if norm(needle) not in text:
                return False, f"missing {needle!r}"
        for needle in self.not_contains:
            if norm(needle) in text:
                return False, f"should not contain {needle!r}"
        if self.regex and not re.search(self.regex, result.output,
                                        0 if self.case_sensitive else re.IGNORECASE):
            return False, f"does not match /{self.regex}/"
        if self.equals is not None and norm(self.equals) != text.strip():
            return False, f"expected exactly {self.equals!r}"
        if self.json_equals is not None:
            actual = result.data
            if hasattr(actual, "model_dump"):
                actual = actual.model_dump()
            if actual != self.json_equals:
                return False, f"structured output was {actual!r}"

        called = [c.name for c in result.tool_calls]
        called += [c.name for child in result.children for c in child.tool_calls]
        if self.tool_called and self.tool_called not in called:
            return False, f"never called {self.tool_called!r} (called: {called or 'none'})"
        for name in self.tools_called:
            if name not in called:
                return False, f"never called {name!r}"
        if self.tool_not_called and self.tool_not_called in called:
            return False, f"should not have called {self.tool_not_called!r}"

        if self.max_steps is not None and result.steps > self.max_steps:
            return False, f"took {result.steps} steps, budget was {self.max_steps}"
        if self.max_cost_usd is not None and result.cost_usd > self.max_cost_usd:
            return False, (f"cost ${result.cost_usd:.4f}, budget was "
                           f"${self.max_cost_usd:.4f}")
        return True, ""


class GoldenTask(BaseModel):
    """One case in the suite: an input, and what a good answer looks like."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    id: str
    input: str
    expect: Expect = Field(default_factory=Expect)
    weight: float = 1.0
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    grader: Grader | None = Field(default=None, exclude=True)


class TaskOutcome(BaseModel):
    """How one task went."""

    model_config = ConfigDict(extra="allow")

    id: str
    passed: bool
    score: float = 0.0
    detail: str = ""
    output: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


class EvalReport(BaseModel):
    """The suite's result. Save it and you have a baseline to score against."""

    model_config = ConfigDict(extra="allow")

    agent: str = ""
    model: str = ""
    label: str = ""
    ts: float = Field(default_factory=time.time)
    outcomes: list[TaskOutcome] = Field(default_factory=list)
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if o.passed]

    @property
    def failed(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if not o.passed]

    @property
    def score(self) -> float:
        """Weighted pass rate, 0.0 to 1.0."""
        if not self.outcomes:
            return 0.0
        return round(sum(o.score for o in self.outcomes) / len(self.outcomes), 4)

    @property
    def cost_usd(self) -> float:
        return round(sum(o.cost_usd for o in self.outcomes), 6)

    def by_tag(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            for tag in outcome.tags:
                buckets.setdefault(tag, []).append(outcome.score)
        return {tag: round(sum(v) / len(v), 4) for tag, v in buckets.items()}

    def render(self) -> str:
        lines = [
            f"{self.label or self.agent}: {len(self.passed)}/{self.total} passed "
            f"· score {self.score:.2%} · ${self.cost_usd:.4f} · {self.duration_s:.1f}s"
        ]
        for outcome in self.outcomes:
            mark = "PASS" if outcome.passed else "FAIL"
            lines.append(f"  [{mark}] {outcome.id}"
                         + (f" — {outcome.detail}" if outcome.detail else ""))
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return file

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def compare(self, baseline: EvalReport) -> Comparison:
        """Score this run against a saved baseline."""
        before = {o.id: o for o in baseline.outcomes}
        fixed, broke, still_failing = [], [], []
        for outcome in self.outcomes:
            was = before.get(outcome.id)
            if was is None:
                continue
            if outcome.passed and not was.passed:
                fixed.append(outcome.id)
            elif not outcome.passed and was.passed:
                broke.append(outcome.id)
            elif not outcome.passed:
                still_failing.append(outcome.id)
        return Comparison(
            baseline_label=baseline.label or "baseline",
            candidate_label=self.label or "candidate",
            baseline_score=baseline.score, candidate_score=self.score,
            fixed=fixed, regressed=broke, still_failing=still_failing,
            new_tasks=[o.id for o in self.outcomes if o.id not in before],
            missing_tasks=[i for i in before if i not in {o.id for o in self.outcomes}],
            baseline_cost=baseline.cost_usd, candidate_cost=self.cost_usd,
        )


class Comparison(BaseModel):
    """Did the change help? The only question an eval exists to answer."""

    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    fixed: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    still_failing: list[str] = Field(default_factory=list)
    new_tasks: list[str] = Field(default_factory=list)
    missing_tasks: list[str] = Field(default_factory=list)
    baseline_cost: float = 0.0
    candidate_cost: float = 0.0

    @property
    def delta(self) -> float:
        return round(self.candidate_score - self.baseline_score, 4)

    @property
    def improved(self) -> bool:
        """Better overall, and nothing that used to work is broken."""
        return self.delta > 0 and not self.regressed

    @property
    def verdict(self) -> str:
        if self.regressed:
            return "regressed"
        if self.delta > 0:
            return "improved"
        if self.delta < 0:
            return "worse"
        return "unchanged"

    def render(self) -> str:
        lines = [
            f"{self.verdict.upper()}: {self.baseline_score:.2%} → "
            f"{self.candidate_score:.2%} ({self.delta:+.2%})",
            f"cost: ${self.baseline_cost:.4f} → ${self.candidate_cost:.4f}",
        ]
        if self.fixed:
            lines.append(f"  fixed:        {', '.join(self.fixed)}")
        if self.regressed:
            lines.append(f"  REGRESSED:    {', '.join(self.regressed)}")
        if self.still_failing:
            lines.append(f"  still failing: {', '.join(self.still_failing)}")
        if self.new_tasks:
            lines.append(f"  new:          {', '.join(self.new_tasks)}")
        if self.missing_tasks:
            lines.append(f"  missing:      {', '.join(self.missing_tasks)}")
        return "\n".join(lines)


class Evaluator:
    """Runs a suite of golden tasks against an agent."""

    def __init__(self, tasks: Sequence[GoldenTask] | None = None, *,
                 concurrency: int = 4, clean_run: bool = True) -> None:
        self.tasks: list[GoldenTask] = list(tasks or [])
        self.concurrency = concurrency
        self.clean_run = clean_run  # each task starts with no conversation history

    # ---- loading -------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> Evaluator:
        """Load tasks from JSON or JSONL."""
        file = Path(path)
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            blob = json.loads(text)
            rows = blob["tasks"] if isinstance(blob, dict) else blob
        return cls([GoldenTask(**row) for row in rows], **kwargs)

    def add(self, task: GoldenTask) -> GoldenTask:
        self.tasks.append(task)
        return task

    def select(self, tags: Iterable[str]) -> Evaluator:
        wanted = set(tags)
        return Evaluator([t for t in self.tasks if wanted & set(t.tags)],
                         concurrency=self.concurrency, clean_run=self.clean_run)

    # ---- running -------------------------------------------------------
    async def run(self, agent: Any, *, label: str = "",
                  scheduler: Any = None) -> EvalReport:
        """Run every task. Failures are recorded, never raised."""
        import asyncio
        import inspect

        started = time.perf_counter()
        semaphore = asyncio.Semaphore(max(1, self.concurrency))

        async def one(task: GoldenTask) -> TaskOutcome:
            async with semaphore:
                task_started = time.perf_counter()
                kwargs: dict[str, Any] = {"messages": []} if self.clean_run else {}
                try:
                    result = await agent.run(task.input, **kwargs)
                except Exception as exc:  # a crash is a failed task, not a failed suite
                    return TaskOutcome(
                        id=task.id, passed=False, score=0.0, tags=task.tags,
                        detail=f"{type(exc).__name__}: {exc}",
                        error=f"{type(exc).__name__}: {exc}",
                        duration_s=round(time.perf_counter() - task_started, 3),
                    )

                passed, detail = task.expect.check(result)
                if passed and task.grader is not None:
                    verdict = task.grader(result, task)
                    if inspect.isawaitable(verdict):
                        verdict = await verdict
                    if isinstance(verdict, tuple):
                        passed, detail = verdict
                    else:
                        passed = bool(verdict)
                        detail = "" if passed else "the custom grader said no"

                return TaskOutcome(
                    id=task.id, passed=passed, score=task.weight if passed else 0.0,
                    detail=detail, output=result.output[:2000], steps=result.steps,
                    cost_usd=result.cost_usd, tags=task.tags, error=result.error,
                    duration_s=round(time.perf_counter() - task_started, 3),
                )

        outcomes = await asyncio.gather(*(one(t) for t in self.tasks))
        by_id = {t.id: t.weight for t in self.tasks}
        total_weight = sum(by_id.values()) or 1.0
        # Normalise so weights actually matter in the headline score.
        for outcome in outcomes:
            if outcome.passed:
                outcome.score = round(by_id.get(outcome.id, 1.0) * len(outcomes)
                                      / total_weight, 4)

        return EvalReport(
            agent=getattr(agent, "name", ""), model=getattr(agent, "model", ""),
            label=label, outcomes=list(outcomes),
            duration_s=round(time.perf_counter() - started, 3),
        )

    def __len__(self) -> int:
        return len(self.tasks)


def llm_judge(judge_agent: Any, rubric: str) -> Grader:
    """A grader that asks a model. Use it last — it costs money and it can be wrong.

    Args:
        judge_agent: an Agent that will score the answer.
        rubric: what a passing answer must do.
    """

    async def grade(result: RunResult, task: GoldenTask) -> tuple[bool, str]:
        verdict = await judge_agent.run(
            "You are grading another agent's answer. Be strict and brief.\n\n"
            f"TASK\n{task.input}\n\nRUBRIC\n{rubric}\n\n"
            f"ANSWER\n{result.output}\n\n"
            "Reply with PASS or FAIL on the first line, then one line of reason.",
            messages=[],
        )
        text = verdict.output.strip()
        passed = text.upper().startswith("PASS")
        reason = "\n".join(text.splitlines()[1:]).strip()
        return passed, "" if passed else (reason or "the judge said FAIL")

    return grade
