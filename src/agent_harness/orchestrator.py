"""The accountable manager: plan → staff → run → consolidate → review.

One orchestrator owns the job. It turns a request into a costed plan, decides
for every task whether to reuse a sub-agent from the bench or have the factory
write a new one, runs what can run in parallel, consolidates the hand-backs, and
only then accepts the work against the definition of done it wrote up front.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from .agent import Agent, _agent_count
from .harness import Harness
from .prompts import Prompt
from .providers.base import model_info
from .runtime.budget import Budget
from .subagents import Bench, SubAgentFactory, SubAgentSpec, build_agent
from .tools import Tool
from .types import Artifact, RunResult, Usage, new_id

__all__ = ["Task", "Plan", "Review", "Orchestrator"]

TaskStatus = Literal["pending", "running", "done", "failed", "blocked", "skipped"]


class Task(BaseModel):
    """One unit of work a single sub-agent can finish alone."""

    id: str = Field(default_factory=lambda: new_id("task"))
    statement: str
    depends_on: list[str] = Field(default_factory=list)
    done_when: str = ""
    agent: str = ""
    reused: bool = True
    status: TaskStatus = "pending"
    output: str = ""
    partial: str = ""          # kept when a task fails or runs out of time
    error: str | None = None
    cost_usd: float = 0.0
    attempts: int = 0

    @property
    def usable_output(self) -> str:
        """What consolidation can work with — the result, or what got done."""
        return self.output or self.partial


class Plan(BaseModel):
    """A costed plan: the goal, how we will know it is done, and the task graph."""

    goal: str
    definition_of_done: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    estimate_usd: float = 0.0
    notes: str = ""

    def by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def waves(self) -> list[list[Task]]:
        """Tasks grouped into rounds that can each run in parallel."""
        done: set[str] = set()
        remaining = list(self.tasks)
        out: list[list[Task]] = []
        while remaining:
            ready = [t for t in remaining if all(d in done for d in t.depends_on)]
            if not ready:  # a cycle or a dangling dependency — run the rest serially
                out.append(remaining)
                break
            out.append(ready)
            ready_ids = {t.id for t in ready}
            done |= ready_ids
            remaining = [t for t in remaining if t.id not in ready_ids]
        return out

    def render(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.definition_of_done:
            lines.append("Definition of done:\n" +
                         "\n".join(f"  - {d}" for d in self.definition_of_done))
        for task in self.tasks:
            deps = f" (after {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"  [{task.id}] {task.statement}{deps}")
        if self.estimate_usd:
            lines.append(f"Estimated cost: ${self.estimate_usd:.3f}")
        return "\n".join(lines)


class Review(BaseModel):
    """An independent critic's verdict against the definition of done."""

    accepted: bool = False
    score: float = 0.0
    problems: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""


# --- drafting models the planner and reviewer must return --------------------

class _TaskDraft(BaseModel):
    id: str
    statement: str
    depends_on: list[str] = Field(default_factory=list)
    done_when: str = ""


class _PlanDraft(BaseModel):
    goal: str
    definition_of_done: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tasks: list[_TaskDraft] = Field(default_factory=list)


class _ReviewDraft(BaseModel):
    accepted: bool
    score: float = 0.0
    problems: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""


PLANNER_PROMPT = Prompt(
    "orchestrator.plan",
    """Turn this request into a costed plan.

REQUEST
{request}

SUB-AGENTS ON THE BENCH
{bench}

Rules:
- Write the definition of done FIRST: the acceptance tests, written before any
  work starts. Each one must be checkable by someone who was not involved.
- Break the goal into tasks one worker could finish alone. Give every task a
  short id (t1, t2, ...) and list what it depends on. Independent tasks must
  have empty dependencies so they can run at the same time.
- As few tasks as the goal allows. Do not invent work the request did not ask for.
- Note any constraint the request imposes (deadline, format, tone, sources).""",
)

CONSOLIDATE_PROMPT = Prompt(
    "orchestrator.consolidate",
    """Consolidate these sub-agent results into the answer the request asked for.

REQUEST
{request}

DEFINITION OF DONE
{dod}

RESULTS
{results}

Merge what agrees, de-duplicate what repeats, and rank what matters most first.
Attribute anything contested to the sub-agent that said it. Do not add findings
nobody reported. Write the deliverable itself — not a description of it.""",
)

REVIEW_PROMPT = Prompt(
    "orchestrator.review",
    """Review this deliverable against the definition of done. You did not do the work.

REQUEST
{request}

DEFINITION OF DONE
{dod}

DELIVERABLE
{deliverable}

Check every acceptance test. Accept only if all of them hold. If you reject,
list the specific gaps — each one must be actionable as a task on its own.""",
)


class Orchestrator:
    """Owns the goal, the plan, the budget and the answer."""

    def __init__(
        self,
        name: str = "orchestrator",
        *,
        harness: Harness | None = None,
        bench: Bench | None = None,
        subagents: Sequence[SubAgentSpec | Agent] = (),
        tools: Iterable[Tool] = (),
        model: str | None = None,
        provider: Any = None,
        tier: str = "deep",
        budget: Budget | None = None,
        max_concurrency: int = 4,
        max_rework: int = 1,
        review: bool = True,
        use_factory: bool = True,
        task_retries: int = 1,
        task_timeout: float | None = None,
        runtime_agents: bool | str = True,
        max_runtime_agents: int = 20,
        compact_at: int | float | None = None,
        instructions: str = "",
    ) -> None:
        self.harness = harness or Harness()
        if budget is not None:
            self.harness.budget = budget
            self.harness.reset_budget(budget)
        self.harness.scheduler.max_concurrency = max_concurrency
        self.bench = bench if bench is not None else Bench.standard()
        for entry in subagents:
            if isinstance(entry, SubAgentSpec):
                self.bench.register(entry)
        self.max_rework = max_rework
        self.review_enabled = review
        self.task_retries = task_retries
        self.task_timeout = task_timeout

        # The manager owns the staffing decision through `staff()`, so it does not
        # need the spawn tool as well — `use_factory` is that switch. The knobs are
        # here so an orchestrator built by hand can be tuned like any other agent.
        self.manager = Agent(
            name, instructions or _MANAGER_INSTRUCTIONS, description=(
                "Owns the goal, the plan, the budget and the answer for one job."
            ),
            model=model, provider=provider, tier=tier, harness=self.harness,
            tools=tools, memory=True, max_steps=4, compact_at=compact_at,
            runtime_agents=runtime_agents and use_factory,
            max_runtime_agents=max_runtime_agents,
        )
        self.max_runtime_agents = _agent_count(max_runtime_agents)
        self.planner = Agent(
            f"{name}.planner", "You plan work. You do not do the work.",
            model=model, provider=provider, tier=tier, harness=self.harness,
            memory=False, output_type=_PlanDraft, max_steps=3, persist_session=False,
        )
        self.reviewer = Agent(
            f"{name}.reviewer",
            "You are an independent critic. You did not do the work and you have no "
            "stake in accepting it.",
            model=model, provider=provider, tier=tier, harness=self.harness,
            memory=False, output_type=_ReviewDraft, max_steps=3, persist_session=False,
        )
        self.factory = SubAgentFactory(
            Agent(f"{name}.factory", "You write specifications for sub-agents.",
                  model=model, provider=provider, tier="balanced",
                  harness=self.harness, memory=False, max_steps=2,
                  persist_session=False),
            bench=self.bench,
        ) if use_factory else None

        self._live: dict[str, Agent] = {}
        for entry in subagents:
            if isinstance(entry, Agent):
                self._live[entry.name] = entry

    # ------------------------------------------------------------------
    # 1 — plan and scope
    # ------------------------------------------------------------------
    async def plan(self, request: str) -> Plan:
        """Turn a request into a costed plan with acceptance tests written first."""
        with self.harness.tracer.span("orchestrator:plan", kind="step"):
            result = await self.planner.run(
                PLANNER_PROMPT.render(request=request, bench=self.bench.catalogue()),
                messages=[],
            )
        draft = result.data if isinstance(result.data, _PlanDraft) else None
        if draft is None or not draft.tasks:
            plan = Plan(goal=request, definition_of_done=["The request is answered."],
                        tasks=[Task(id="t1", statement=request)])
        else:
            plan = Plan(
                goal=draft.goal or request,
                definition_of_done=draft.definition_of_done,
                constraints=draft.constraints,
                tasks=[Task(id=t.id, statement=t.statement, depends_on=t.depends_on,
                            done_when=t.done_when) for t in draft.tasks],
            )
        plan.estimate_usd = self.estimate(plan)
        if self.manager.memory is not None:
            await self.manager.memory.orchestrator.plan(
                plan.goal, tasks=len(plan.tasks), estimate_usd=plan.estimate_usd
            )
        return plan

    def estimate(self, plan: Plan, *, tokens_per_task: int = 6000) -> float:
        """A cost and parallelism estimate: how wide to go, and what it will cost."""
        info = model_info(self.manager.model)
        if info is None:
            return 0.0
        per_task = (tokens_per_task * 0.75 * info.input_cost
                    + tokens_per_task * 0.25 * info.output_cost) / 1_000_000
        overhead = per_task * 2  # planning, consolidation, review
        return round(per_task * max(len(plan.tasks), 1) + overhead, 4)

    # ------------------------------------------------------------------
    # 2 — the staffing decision
    # ------------------------------------------------------------------
    async def staff(self, task: Task, *, available_tools: Iterable[str] = ()) -> Agent:
        """Reuse a pre-defined sub-agent, or have the factory write a new one."""
        if task.agent and task.agent in self._live:
            return self._live[task.agent]

        spec = self.bench.find(task.statement)
        reused = spec is not None
        if spec is None:
            if self.factory is None:
                spec = SubAgentSpec(name=f"worker_{task.id}",
                                    description=task.statement[:80],
                                    instructions="Finish the task you were given.")
            else:
                spec = await self.factory.create(
                    task.statement, tools=available_tools or self.manager.tools.names
                )

        task.agent = spec.name
        task.reused = reused
        if spec.name not in self._live:
            self._live[spec.name] = build_agent(spec, self.manager)
        if self.manager.memory:
            await self.manager.memory.orchestrator.staffing(
                task.statement[:80], spec.name, reused
            )
        await self.harness.journal.write(
            "decision", f"{task.id}: {'reused' if reused else 'built'} {spec.name}",
            agent=self.manager.name,
        )
        return self._live[spec.name]

    # ------------------------------------------------------------------
    # 3 — work assignment
    # ------------------------------------------------------------------
    async def _run_task(self, task: Task, plan: Plan, request: str) -> Task:
        worker = await self.staff(task)
        context = self._context_for(task, plan)
        brief = self._brief(task, request, context)

        for attempt in range(1, self.task_retries + 2):
            if not self.harness.control.may_start():
                task.status = "skipped"
                task.error = "the run was stopped"
                return task

            task.attempts = attempt
            task.status = "running"
            try:
                coro = worker.run(brief, messages=[],
                                  guard=self.harness.guard.child())
                result: RunResult = (
                    await asyncio.wait_for(coro, self.task_timeout)
                    if self.task_timeout else await coro
                )
            except (TimeoutError, asyncio.TimeoutError):
                task.error = f"missed its {self.task_timeout}s deadline"
                await self.harness.journal.write("timeout", task.error,
                                                 agent=worker.name)
                if attempt > self.task_retries:
                    break
                continue

            task.cost_usd = round(task.cost_usd + result.cost_usd, 6)
            # Partial delivery is kept: a task that failed halfway still leaves
            # something consolidation can use, and something you can debug from.
            if result.output:
                task.partial = result.output
            if result.artifacts:
                self.harness.deliverables.extend(result.artifacts, run_id=task.id)
            if self.manager.memory:
                await self.manager.memory.orchestrator.spend(
                    worker.name, result.cost_usd, result.usage.total_tokens
                )
            if not result.error:
                task.output = result.output
                task.status = "done"
                self._results[task.id] = result
                return task
            task.error = result.error
            self._results.setdefault(task.id, result)
            if attempt > self.task_retries:
                break
        task.status = "failed"
        return task

    def _context_for(self, task: Task, plan: Plan) -> str:
        """Only what this task depends on — never the whole job's history."""
        parts: list[str] = []
        for dep_id in task.depends_on:
            dep = plan.by_id(dep_id)
            if dep and dep.usable_output:
                note = "" if dep.output else " — partial, this task did not finish"
                parts.append(f"### Result of {dep_id} ({dep.statement}){note}\n"
                             f"{dep.usable_output}")
        return "\n\n".join(parts)

    @staticmethod
    def _brief(task: Task, request: str, context: str) -> str:
        parts = [
            f"## Your task\n{task.statement}",
            f"## Why it matters\nIt is part of: {request}",
        ]
        if task.done_when:
            parts.append(f"## Done when\n{task.done_when}")
        if context:
            parts.append(f"## What earlier tasks produced\n{context}")
        return "\n\n".join(parts)

    async def execute(self, plan: Plan, request: str) -> Plan:
        """Run the task graph, many at once where the dependencies allow it."""
        self._results: dict[str, RunResult] = getattr(self, "_results", {})
        for wave in plan.waves():
            if not self.harness.control.may_start():
                for task in plan.tasks:
                    if task.status == "pending":
                        task.status = "skipped"
                        task.error = "the run was stopped"
                break
            runnable = [t for t in wave if t.status == "pending"]
            if not runnable:
                continue
            with self.harness.tracer.span(f"wave:{len(runnable)}", kind="step"):
                outcomes = await self.harness.scheduler.map(
                    lambda t: self._run_task(t, plan, request), runnable
                )
            # An unexpected exception must not leave a task stuck in "running"
            # with the job reporting itself finished.
            for task, outcome in zip(runnable, outcomes, strict=False):
                if isinstance(outcome, BaseException):
                    task.status = "failed"
                    task.error = f"{type(outcome).__name__}: {outcome}"
                    await self.harness.journal.write("error", task.error,
                                                     agent=task.agent or "unstaffed")
                elif task.status == "running":  # pragma: no cover - belt and braces
                    task.status = "failed"
                    task.error = task.error or "the task ended without a result"
            for task in plan.tasks:
                if task.status == "pending" and any(
                    (plan.by_id(d) or Task(id=d, statement="")).status == "failed"
                    for d in task.depends_on
                ):
                    task.status = "blocked"
                    task.error = "a task it depends on failed"
        return plan

    # ------------------------------------------------------------------
    # 4 — consolidate, review, accept or rework
    # ------------------------------------------------------------------
    async def consolidate(self, plan: Plan, request: str) -> str:
        usable = [t for t in plan.tasks if t.usable_output]
        if not usable:
            return ""
        # A single finished task is the answer. A single *partial* one is not —
        # it goes through consolidation so it is framed as what it actually is.
        if len(usable) == 1 and len(plan.tasks) == 1 and usable[0].status == "done":
            return usable[0].usable_output
        body = "\n\n".join(
            f"### {t.id} — {t.agent} ({'reused' if t.reused else 'purpose-built'})"
            + ("" if t.status == "done" else " [PARTIAL — this task did not finish]")
            + f"\nTask: {t.statement}\n{t.usable_output}"
            for t in usable
        )
        with self.harness.tracer.span("orchestrator:consolidate", kind="step"):
            result = await self.manager.run(
                CONSOLIDATE_PROMPT.render(
                    request=request, results=body,
                    dod="\n".join(f"- {d}" for d in plan.definition_of_done) or "(none)",
                ),
                messages=[],
            )
        return result.output

    async def review(self, plan: Plan, deliverable: str, request: str) -> Review:
        if not self.review_enabled or not deliverable:
            return Review(accepted=True, score=1.0, summary="review disabled")
        with self.harness.tracer.span("orchestrator:review", kind="step"):
            result = await self.reviewer.run(
                REVIEW_PROMPT.render(
                    request=request, deliverable=deliverable[:40_000],
                    dod="\n".join(f"- {d}" for d in plan.definition_of_done) or "(none)",
                ),
                messages=[],
            )
        draft = result.data if isinstance(result.data, _ReviewDraft) else None
        if draft is None:
            return Review(accepted=True, score=0.5,
                          summary="the reviewer returned nothing usable")
        return Review(**draft.model_dump())

    # ------------------------------------------------------------------
    # the whole job
    # ------------------------------------------------------------------
    async def run(self, request: str, *, plan: Plan | None = None) -> RunResult:
        """Plan, staff, run, consolidate and review — one costed job."""
        started = time.time()
        run_id = new_id("job")
        self._results = {}
        result = RunResult(agent=self.manager.name, run_id=run_id)

        # Bound before the try: a failure while planning must still leave the
        # artefact and reporting path below with something to work with.
        deliverable = ""
        review = Review()

        with self.harness.tracer.span(f"job:{self.manager.name}", kind="run") as span:
            result.trace_id = span.trace_id
            await self.harness.journal.assignment(self.manager.name, request[:500],
                                                  run_id=run_id)
            try:
                plan = plan or await self.plan(request)

                for round_no in range(self.max_rework + 1):
                    await self.execute(plan, request)
                    deliverable = await self.consolidate(plan, request)
                    review = await self.review(plan, deliverable, request)
                    if review.accepted or round_no == self.max_rework:
                        break
                    # Rejected: re-plan only the gaps, then run those.
                    for gap in review.gaps or review.problems:
                        plan.tasks.append(Task(id=new_id("gap")[:8], statement=gap,
                                               done_when="the gap the reviewer named "
                                                         "is closed"))

                result.output = deliverable
                result.data = {"plan": plan.model_dump(mode="json"),
                               "review": review.model_dump()}
                result.stop_reason = "end_turn" if review.accepted else "stopped"
                if not review.accepted:
                    result.error = None  # delivered, but flagged as short of the bar
            except Exception as exc:  # the manager reports, it does not crash the caller
                result.error = f"{type(exc).__name__}: {exc}"
                result.stop_reason = "error"

        for child in self._results.values():
            result.children.append(child)
            result.artifacts.extend(child.artifacts)
        result.usage = _sum_usage(self._results.values())
        result.steps = len(self._results)
        result.artifacts.append(Artifact(
            name="plan.json", media_type="application/json",
            content=(plan.model_dump_json(indent=2) if plan else "{}"),
            produced_by=self.manager.name,
        ))
        if deliverable:
            result.artifacts.append(Artifact(
                name="deliverable.md", content=deliverable,
                media_type="text/markdown", produced_by=self.manager.name,
            ))
        # The output of the run belongs in the deliverable store, not only in the
        # result object the caller happens to be holding.
        self.harness.deliverables.extend(result.artifacts, run_id=run_id)
        await self.harness.journal.handback(
            self.manager.name, (result.output or result.error or "")[:500],
            run_id=run_id, cost_usd=result.cost_usd,
            elapsed_s=round(time.time() - started, 2),
        )
        return result

    def run_sync(self, request: str, **kw: Any) -> RunResult:
        return asyncio.run(self.run(request, **kw))

    def report(self) -> dict[str, Any]:
        return self.harness.report()


def _sum_usage(results: Iterable[RunResult]) -> Usage:
    total = Usage()
    for result in results:
        total += result.usage
    return total


_MANAGER_INSTRUCTIONS = (
    "You own this job: the goal, the plan, the budget and the answer.\n"
    "- Decide reuse-or-create for every task; never hand a sub-agent more access "
    "than its task needs.\n"
    "- Delegate work that can run in parallel; never wait on one sub-agent when "
    "two could run.\n"
    "- Re-plan when a result comes back short.\n"
    "- You are the only component that speaks to the requester."
)
