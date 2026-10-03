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
from .errors import ConfigurationError, HarnessError, SessionConflict
from .harness import Harness
from .llm_providers.base import model_info
from .prompts import Prompt
from .runtime.budget import Budget
from .runtime.session import Session
from .subagents import Bench, SubAgentFactory, SubAgentSpec, build_agent
from .tools import Tool
from .types import Artifact, RunResult, Usage, new_id

__all__ = ["Task", "Plan", "Review", "Orchestrator"]

TaskStatus = Literal["pending", "running", "done", "failed", "blocked", "skipped",
                     "replaced"]


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
    #: Was the hand-back checked against `done_when`? True: checked and met.
    #: False: it could not be judged. None: there was nothing to check it against.
    checked: bool | None = None
    #: What the check found missing, the last time it found something.
    missing: str = ""
    #: The task this one was planned in place of, when one came back short.
    replaces: str = ""

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
    #: How wide the plan goes: the most tasks that will run at one time.
    parallelism: int = 1
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
    #: Did any critic actually return a verdict? False means the deliverable is
    #: unreviewed — which is not the same as accepted, and not reported as it.
    reviewed: bool = True
    #: Each critic's own verdict: who, what they looked for, and what they said.
    verdicts: list[dict[str, Any]] = Field(default_factory=list)


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


class _CheckDraft(BaseModel):
    met: bool
    missing: str = ""


class _ReplanDraft(BaseModel):
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

CHECK_PROMPT = Prompt(
    "orchestrator.check",
    """Check this hand-back against what the task was to be done by. You did not do the work.

TASK
{task}

DONE WHEN
{done_when}

HAND-BACK
{output}

Say whether the hand-back meets "done when" — judged on what is written, not on
what is promised. If it does not, say exactly what is missing, in one or two
sentences the worker could act on.""",
)

REPLAN_PROMPT = Prompt(
    "orchestrator.replan",
    """A task in this plan came back short. Plan what to do instead.

GOAL
{goal}

THE TASK THAT CAME BACK SHORT
{task}

DONE WHEN
{done_when}

WHAT WENT WRONG
{error}

WHAT IT DID PRODUCE
{partial}

SUB-AGENTS ON THE BENCH
{bench}

Write the task or tasks that will get this done by a different route: split it
into smaller steps, narrow it, or come at it another way. Do not repeat the task
as it was. Give each a short id and say what it depends on among the others you
write. If there is no better route, return no tasks.""",
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

{lens}Check every acceptance test. Accept only if all of them hold. If you reject,
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
        critics: int | Sequence[str] = 1,
        accept: Literal["all", "majority"] = "all",
        use_factory: bool = True,
        task_retries: int = 1,
        task_timeout: float | None = None,
        check_tasks: bool = True,
        max_replans: int = 2,
        on_over_estimate: Literal["warn", "stop", "ignore"] = "warn",
        tokens_per_task: int = 6000,
        persist: bool = True,
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
        # Who reviews: a number of independent critics, or what each is to look
        # for — `critics=["accuracy of every figure", "tone and length"]`.
        if isinstance(critics, int):
            self.critics: list[str] = [""] * max(1, critics)
        else:
            self.critics = [str(c).strip() for c in critics] or [""]
        if accept not in ("all", "majority"):
            raise ValueError("accept is 'all' or 'majority'")
        #: How many of the critics who gave a verdict have to accept.
        self.accept = accept
        self.task_retries = task_retries
        self.task_timeout = task_timeout
        #: Check each hand-back against its task's `done_when` before taking it.
        self.check_tasks = check_tasks
        #: How many tasks one job may plan again after they came back short.
        self.max_replans = max(0, int(max_replans))
        self._replans = 0
        if on_over_estimate not in ("warn", "stop", "ignore"):
            raise ValueError("on_over_estimate is 'warn', 'stop' or 'ignore'")
        #: What to do when the work is expected to cost more than the budget has
        #: left: say so and carry on, refuse to start what cannot be paid for,
        #: or not look.
        self.on_over_estimate = on_over_estimate
        #: What a task is assumed to use before any has been seen to finish.
        self.tokens_per_task = max(1, int(tokens_per_task))
        # What tasks have actually cost on this orchestrator: (count, total USD).
        self._seen_tasks: tuple[int, float] = (0, 0.0)
        self._warnings: list[str] = []
        #: Keep the job in the harness's session store as it goes, so one cut
        #: short — a crash, a deploy — is picked up by `resume(job_id)` from the
        #: last task that finished. It lasts as long as that store does.
        self.persist = persist
        self._record: dict[str, Any] | None = None
        self._record_version = 0
        self._prior = Usage()

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
        self.checker = Agent(
            f"{name}.checker",
            "You check a piece of work against what it was to be done by. You did "
            "not do the work and you have no stake in passing it.",
            model=model, provider=provider, tier="balanced", harness=self.harness,
            memory=False, output_type=_CheckDraft, max_steps=2, persist_session=False,
        )
        self.replanner = Agent(
            f"{name}.replanner", "You plan work. You do not do the work.",
            model=model, provider=provider, tier=tier, harness=self.harness,
            memory=False, output_type=_ReplanDraft, max_steps=3, persist_session=False,
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
        #: The guard everything in the job under way is charged to: planning,
        #: staffing, the work, consolidation and review alike.
        self._job: Any = None

    def _charged(self) -> dict[str, Any]:
        """Run under the job's guard — or, outside a job, as an agent normally does."""
        return {"guard": self._job} if self._job is not None else {}

    # ------------------------------------------------------------------
    # 1 — plan and scope
    # ------------------------------------------------------------------
    async def plan(self, request: str) -> Plan:
        """Turn a request into a costed plan with acceptance tests written first."""
        with self.harness.tracer.span("orchestrator:plan", kind="step"):
            result = await self.planner.run(
                PLANNER_PROMPT.render(request=request, bench=self.bench.catalogue()),
                messages=[], **self._charged(),
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

    def _per_task(self, tokens_per_task: int | None = None) -> float:
        """What one task is expected to cost: what tasks here have actually been
        costing, once a few have finished; a guess from the model's prices until
        then."""
        count, spent = self._seen_tasks
        if tokens_per_task is None and count >= 3:
            return spent / count
        info = model_info(self.manager.model)
        if info is None:
            return 0.0
        tokens = tokens_per_task or self.tokens_per_task
        return (tokens * 0.75 * info.input_cost + tokens * 0.25 * info.output_cost) / 1e6

    def estimate(self, plan: Plan, *, tokens_per_task: int | None = None) -> float:
        """A cost and parallelism estimate: how wide to go, and what it will cost.

        Sets `plan.parallelism`. The cost is per-task spend times the tasks,
        plus the manager's own share: the plan, the consolidation, the review,
        and a check of each hand-back that has something to be checked against.
        """
        per_task = self._per_task(tokens_per_task)
        widest = max((len(wave) for wave in plan.waves()), default=1)
        plan.parallelism = max(1, min(widest, self.harness.scheduler.max_concurrency))
        checks = sum(1 for t in plan.tasks if t.done_when) if self.check_tasks else 0
        overhead = per_task * (2 + 0.2 * checks)
        return round(per_task * max(len(plan.tasks), 1) + overhead, 4)

    def _affordable(self, plan: Plan) -> str:
        """Why the work still to do will not fit in the budget; empty when it will,
        or when there is no spend ceiling to fit it in."""
        left = self.harness.guard.remaining_usd
        if left is None or self.on_over_estimate == "ignore":
            return ""
        waiting = [t for t in plan.tasks if t.status == "pending"]
        if not waiting:
            return ""
        finished = [t for t in plan.tasks if t.status in ("done", "failed", "replaced")
                    and t.cost_usd > 0]
        if finished:
            # Going by what this job's tasks have cost so far, not by the guess.
            each = sum(t.cost_usd for t in finished) / len(finished)
            needed = each * len(waiting)
            basis = f"its tasks are costing ${each:.4f} each"
        else:
            needed = plan.estimate_usd
            basis = "the estimate"
        if needed <= left:
            return ""
        return (f"the {len(waiting)} task{'s' * (len(waiting) != 1)} still to run "
                f"would cost about ${needed:.4f} going by {basis}, and the budget "
                f"has ${left:.4f} left")

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
                    task.statement, tools=available_tools or self.manager.tools.names,
                    guard=self._job,
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

        for attempt in range(1, self.task_retries + 2):
            # A second attempt is told what the first was missing.
            brief = self._brief(task, request, context)
            if not self.harness.control.may_start():
                task.status = "skipped"
                task.error = "the run was stopped"
                return task

            task.attempts = attempt
            task.status = "running"
            try:
                # Its own ceiling if its spec set one; its spend counts to the job.
                parent = self._job if self._job is not None else self.harness.guard
                coro = worker.run(brief, messages=[],
                                  guard=parent.child(worker.budget))
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
                missing = await self._short_of(task, result.output)
                if not missing:
                    task.output = result.output
                    task.status = "done"
                    task.error = None
                    self._results[task.id] = result
                    return task
                # It answered, but not with what the task was to be done by.
                task.missing = missing
                task.error = f"short of done-when: {missing}"
                await self.harness.journal.write(
                    "decision", f"{task.id}: hand-back refused — {missing[:200]}",
                    agent=self.manager.name)
                self._results[task.id] = result
            else:
                task.error = result.error
                self._results.setdefault(task.id, result)
            if attempt > self.task_retries:
                break
        task.status = "failed"
        return task

    async def _short_of(self, task: Task, output: str) -> str:
        """What a hand-back is missing against `done_when`; empty when it meets it.

        A check that cannot be made — no criterion, or a checker that answered
        with nothing usable — does not hold the work back: it is recorded as
        unchecked, not as failed.
        """
        if not self.check_tasks or not task.done_when.strip() or not output.strip():
            task.checked = None
            return ""
        with self.harness.tracer.span("orchestrator:check", kind="step"):
            verdict = await self.checker.run(
                CHECK_PROMPT.render(task=task.statement, done_when=task.done_when,
                                    output=output[:30_000]),
                messages=[], **self._charged())
        draft = verdict.data if isinstance(verdict.data, _CheckDraft) else None
        if draft is None:
            task.checked = False
            return ""
        task.checked = True
        if draft.met:
            task.missing = ""
            return ""
        return draft.missing.strip() or "it does not meet the criterion"

    async def _replan(self, task: Task, plan: Plan) -> list[Task]:
        """Plan another route for a task that came back short.

        The new tasks take the failed one's place in the graph: they wait for
        what it waited for, and what waited for it now waits for them.
        """
        if self._replans >= self.max_replans or task.replaces:
            return []                 # the budget for second routes is spent
        self._replans += 1
        with self.harness.tracer.span("orchestrator:replan", kind="step"):
            answer = await self.replanner.run(
                REPLAN_PROMPT.render(
                    goal=plan.goal, task=task.statement,
                    done_when=task.done_when or "(not stated)",
                    error=task.error or "it did not finish",
                    partial=(task.partial or "(nothing)")[:8_000],
                    bench=self.bench.catalogue()),
                messages=[], **self._charged())
        draft = answer.data if isinstance(answer.data, _ReplanDraft) else None
        if draft is None or not draft.tasks:
            return []
        taken = {t.id for t in plan.tasks}
        renamed: dict[str, str] = {}
        for item in draft.tasks:
            fresh = f"{task.id}.{len(renamed) + 1}"
            while fresh in taken:
                fresh += "x"
            renamed[item.id] = fresh
            taken.add(fresh)
        added = [Task(
            id=renamed[item.id], statement=item.statement, replaces=task.id,
            done_when=item.done_when or task.done_when,
            # Among themselves as planned; and all of them after what the
            # failed task was waiting for.
            depends_on=[*task.depends_on,
                        *(renamed[d] for d in item.depends_on if d in renamed)])
            for item in draft.tasks]
        ends = [t.id for t in added
                if not any(t.id in other.depends_on for other in added)]
        for other in plan.tasks:
            if task.id in other.depends_on:
                other.depends_on = [d for d in other.depends_on if d != task.id] + ends
        task.status = "replaced"
        plan.tasks.extend(added)
        await self.harness.journal.write(
            "decision", f"{task.id} came back short ({(task.error or '')[:120]}); "
            f"re-planned as {', '.join(t.id for t in added)}", agent=self.manager.name)
        return added

    def _context_for(self, task: Task, plan: Plan) -> str:
        """Only what this task depends on — never the whole job's history."""
        parts: list[str] = []
        earlier = plan.by_id(task.replaces) if task.replaces else None
        if earlier is not None and earlier.partial:
            parts.append(f"### An earlier attempt at this ({earlier.statement}) came back "
                         f"short — {earlier.error or 'unfinished'}. What it produced:\n"
                         f"{earlier.partial}")
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
        if task.missing:
            parts.append("## What the last attempt was missing\n"
                         f"{task.missing}\nPut that right in this one.")
        if context:
            parts.append(f"## What earlier tasks produced\n{context}")
        return "\n\n".join(parts)

    def _ready(self, plan: Plan) -> list[Task]:
        """The pending tasks whose dependencies are all in."""
        known = {t.id: t for t in plan.tasks}
        waiting = [t for t in plan.tasks if t.status == "pending"]
        ready = [t for t in waiting
                 if all(known[d].status == "done" for d in t.depends_on if d in known)]
        if ready:
            return ready
        # Nothing is ready, but something waits on a task that is itself still
        # waiting: a cycle. Run those together rather than hang.
        return [t for t in waiting
                if all(known[d].status in ("done", "pending")
                       for d in t.depends_on if d in known)]

    async def execute(self, plan: Plan, request: str) -> Plan:
        """Run the task graph, many at once where the dependencies allow it.

        A task that comes back short — it failed, missed its deadline, or its
        hand-back did not meet its `done_when` — is planned again by another
        route, up to `max_replans` times a job. What depended on it then waits
        for the new tasks instead.
        """
        self._results: dict[str, RunResult] = getattr(self, "_results", {})
        for _ in range(len(plan.tasks) * 4 + 50):         # every round finishes a task
            if not self.harness.control.may_start():
                for task in plan.tasks:
                    if task.status == "pending":
                        task.status = "skipped"
                        task.error = "the run was stopped"
                break
            runnable = self._ready(plan)
            if not runnable:
                break
            short = self._affordable(plan)
            if short and short not in self._warnings:
                self._warnings.append(short)
                await self.harness.journal.write("budget", short, agent=self.manager.name)
                if self.on_over_estimate == "stop":
                    # Better to hand back what is finished than to start work
                    # that will be cut off half-way through.
                    for task in plan.tasks:
                        if task.status == "pending":
                            task.status = "skipped"
                            task.error = f"not started: {short}"
                    break
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
            for task in runnable:
                if task.status == "failed" and self.harness.control.may_start():
                    try:
                        await self._replan(task, plan)
                    except Exception as exc:
                        # A second route that cannot be planned leaves the first
                        # failure standing; it does not become a second one.
                        await self.harness.journal.write(
                            "error", f"{task.id} could not be re-planned: {exc}",
                            agent=self.manager.name)
            await self._checkpoint(plan, "executing")
            blocked = True
            while blocked:                               # down the whole chain
                blocked = False
                for task in plan.tasks:
                    if task.status == "pending" and any(
                        (plan.by_id(d) or Task(id=d, statement="", status="done")).status
                        in ("failed", "blocked", "skipped") for d in task.depends_on
                    ):
                        task.status = "blocked"
                        task.error = "a task it depends on failed"
                        blocked = True
        return plan

    # ------------------------------------------------------------------
    # 4 — consolidate, review, accept or rework
    # ------------------------------------------------------------------
    async def consolidate(self, plan: Plan, request: str) -> str:
        # What was planned again is spoken for by the tasks that replaced it.
        usable = [t for t in plan.tasks if t.usable_output and t.status != "replaced"]
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
                messages=[], **self._charged(),
            )
        return result.output

    async def _critic(self, number: int, lens: str, plan: Plan, deliverable: str,
                      request: str) -> dict[str, Any] | None:
        """One critic's verdict; None when it gave nothing that can be read as one."""
        prompt = REVIEW_PROMPT.render(
            request=request, deliverable=deliverable[:40_000],
            dod="\n".join(f"- {d}" for d in plan.definition_of_done) or "(none)",
            lens=(f"You are reviewing it for one thing in particular: {lens}.\n"
                  if lens else ""))
        for _ in range(2):                       # asked again once if it says nothing
            result = await self.reviewer.run(prompt, messages=[], **self._charged())
            draft = result.data if isinstance(result.data, _ReviewDraft) else None
            if draft is not None:
                return {"critic": number, "lens": lens, **draft.model_dump()}
        return None

    async def review(self, plan: Plan, deliverable: str, request: str) -> Review:
        """The critics' verdict against the definition of done.

        Each critic reviews on its own, without seeing the others. The
        deliverable is accepted when all of them accept (or most, with
        `accept="majority"`); what any of them found missing is kept, said
        once. A critic that returns nothing usable is asked again, and then
        left out — and if none gives a verdict the deliverable is *unreviewed*,
        not accepted.
        """
        if not self.review_enabled or not deliverable:
            return Review(accepted=True, score=1.0, summary="review disabled",
                          reviewed=False)
        with self.harness.tracer.span("orchestrator:review", kind="step"):
            answers = await asyncio.gather(
                *(self._critic(n, lens, plan, deliverable, request)
                  for n, lens in enumerate(self.critics, 1)))
        verdicts = [a for a in answers if a is not None]
        if not verdicts:
            return Review(accepted=False, score=0.0, reviewed=False,
                          summary="no critic returned a usable verdict; the "
                                  "deliverable is unreviewed")
        yes = sum(1 for v in verdicts if v["accepted"])
        accepted = yes == len(verdicts) if self.accept == "all" else yes * 2 > len(verdicts)

        def merged(key: str) -> list[str]:
            seen: dict[str, str] = {}
            for verdict in verdicts:
                for item in verdict[key]:
                    seen.setdefault(" ".join(item.lower().split()), item)
            return list(seen.values())

        return Review(
            accepted=accepted,
            score=round(sum(v["score"] for v in verdicts) / len(verdicts), 3),
            problems=[] if accepted else merged("problems"),
            gaps=[] if accepted else merged("gaps"),
            summary=" | ".join(v["summary"] for v in verdicts if v["summary"]),
            verdicts=[{k: v[k] for k in ("critic", "lens", "accepted", "score", "summary")}
                      for v in verdicts])

    # ------------------------------------------------------------------
    # the whole job
    # ------------------------------------------------------------------
    # ---- a job that outlives the process ---------------------------------------
    @staticmethod
    def _key(job_id: str) -> str:
        return f"orchestrator_{job_id}"

    async def _checkpoint(self, plan: Plan | None, stage: str, **more: Any) -> None:
        """Write the job down as it stands: the plan, what each task produced,
        what has been spent. Called after every wave and every stage."""
        record = self._record
        if record is None or not self.persist:
            return
        job = self._job
        spent = self._prior + job.usage if job is not None else self._prior
        by_agent = {k: dict(v) for k, v in record.get("by_agent_prior", {}).items()}
        for name, used in (job.by_agent.items() if job is not None else ()):
            held = by_agent.setdefault(name, {"cost_usd": 0.0, "tokens": 0, "calls": 0})
            held["cost_usd"] = round(held["cost_usd"] + used.cost_usd, 6)
            held["tokens"] += used.total_tokens
            held["calls"] += used.calls
        record.update(
            stage=stage, updated=time.time(), replans=self._replans,
            plan=plan.model_dump(mode="json") if plan is not None else record.get("plan"),
            usage=spent.model_dump(), by_agent=by_agent,
            results={key: {"agent": r.agent, "output": r.output, "error": r.error,
                           "usage": r.usage.model_dump()}
                     for key, r in self._results.items()},
            **more)
        try:
            saved = await self.harness.sessions.save(Session(
                id=self._key(record["id"]), agent="#jobs",
                title=str(record["request"])[:80], version=self._record_version,
                created=record["created"],
                metadata={"job": {k: v for k, v in record.items()
                                  if k != "by_agent_prior"}}))
            self._record_version = saved.version
        except SessionConflict:
            # Someone else has written to this job since we last did: it is
            # being run elsewhere. Two managers on one job would do the work twice.
            raise HarnessError(
                f"job {record['id']} is being run by another process; this one "
                "has stopped") from None
        except Exception as exc:
            warning = (f"the job could not be saved ({type(exc).__name__}); it will "
                       "not be resumable from here")
            if warning not in self._warnings:
                self._warnings.append(warning)

    async def job(self, job_id: str) -> dict[str, Any]:
        """Where a job stands: its stage, each task's status, what it has cost."""
        try:
            session = await self.harness.sessions.load(self._key(job_id))
        except ConfigurationError:
            raise ConfigurationError(f"no job {job_id!r}") from None
        record = session.metadata.get("job")
        if not isinstance(record, dict):
            raise ConfigurationError(f"no job {job_id!r}")
        return {**record, "_version": session.version}

    async def resume(self, job_id: str) -> RunResult:
        """Carry on a job that was cut short — a crash, a deploy, a stop.

        Tasks that had finished are not run again: their results were saved.
        A task that was running when the process went is started afresh, so
        anything it had already done to the outside world may happen twice.
        A job that had finished returns what it produced.
        """
        record = await self.job(job_id)
        if not record.get("plan"):
            return await self.run(record["request"], job_id=job_id, _resume=record)
        plan = Plan(**record["plan"])
        for task in plan.tasks:
            if task.status == "running":
                task.status = "pending"
        if record.get("status") in ("done", "failed"):
            review = Review(**(record.get("review") or {}))
            result = RunResult(
                agent=self.manager.name, run_id=job_id, output=record.get("deliverable", ""),
                error=record.get("error"), stop_reason=record.get("stop_reason", "end_turn"),
                data={"plan": plan.model_dump(mode="json"), "review": review.model_dump()})
            result.spend = record.get("spend") or {}
            return result
        return await self.run(record["request"], plan=plan, job_id=job_id, _resume=record)

    async def run(self, request: str, *, plan: Plan | None = None,
                  job_id: str | None = None,
                  _resume: dict[str, Any] | None = None) -> RunResult:
        """Plan, staff, run, consolidate and review — one costed job.

        `result.run_id` is the job's id. The job is saved as it goes; if this
        process does not live to finish it, `resume(result.run_id)` does.
        """
        started = time.time()
        run_id = job_id or new_id("job")
        prior = _resume or {}
        self._results = {
            key: RunResult(agent=held.get("agent", ""), output=held.get("output", ""),
                           error=held.get("error"), run_id=key,
                           usage=Usage(**(held.get("usage") or {})))
            for key, held in (prior.get("results") or {}).items()}
        self._replans = int(prior.get("replans") or 0)
        self._warnings = []
        self._prior = Usage(**(prior.get("usage") or {}))
        self._record_version = int(prior.get("_version") or 0)
        self._record = {
            "id": run_id, "request": request, "status": "running",
            "created": prior.get("created") or time.time(),
            "round": int(prior.get("round") or 0),
            "by_agent_prior": prior.get("by_agent") or {},
        }
        result = RunResult(agent=self.manager.name, run_id=run_id)
        # One guard for the whole job. The sub-agents' work is not all a job
        # costs: the plan, the specs the factory wrote, the consolidation and
        # the review were paid for too, and the result says so.
        self._job = job = self.harness.guard.child()

        # Bound before the try: a failure while planning must still leave the
        # artefact and reporting path below with something to work with.
        deliverable = ""
        review = Review()

        with self.harness.tracer.span(f"job:{self.manager.name}", kind="run") as span:
            result.trace_id = span.trace_id
            await self.harness.journal.assignment(self.manager.name, request[:500],
                                                  run_id=run_id)
            try:
                # Claimed before any work: a job someone else is running is
                # found out here, not after the work has been done twice.
                await self._checkpoint(plan, "resumed" if prior else "planning")
                plan = plan or await self.plan(request)
                await self._checkpoint(plan, "planned")

                # Cut short between consolidating and reviewing: what was
                # consolidated is still good.
                held = (prior.get("deliverable", "")
                        if prior.get("stage") == "consolidated" else "")
                for round_no in range(self._record["round"], self.max_rework + 1):
                    if held:
                        deliverable, held = held, ""
                    else:
                        await self.execute(plan, request)
                        deliverable = await self.consolidate(plan, request)
                        await self._checkpoint(plan, "consolidated",
                                               deliverable=deliverable, round=round_no)
                    review = await self.review(plan, deliverable, request)
                    await self._checkpoint(plan, "reviewed", review=review.model_dump(),
                                           deliverable=deliverable, round=round_no)
                    if review.accepted or round_no == self.max_rework:
                        break
                    if not review.reviewed:
                        break            # nothing was found wrong; nothing to rework
                    if any((t.error or "").startswith("not started:") for t in plan.tasks):
                        break            # no money for the work; none for rework
                    # Rejected: re-plan only the gaps, then run those.
                    for gap in review.gaps or review.problems:
                        plan.tasks.append(Task(id=new_id("gap")[:8], statement=gap,
                                               done_when="the gap the reviewer named "
                                                         "is closed"))
                    await self._checkpoint(plan, "rework", round=round_no + 1)

                result.output = deliverable
                result.data = {"plan": plan.model_dump(mode="json"),
                               "review": review.model_dump()}
                result.stop_reason = "end_turn" if review.accepted else "stopped"
                if self.review_enabled and deliverable and not review.reviewed:
                    self._warnings.append(
                        "the deliverable was not reviewed: no critic returned a "
                        "usable verdict")
                if self.on_over_estimate == "stop" and any(
                        (t.error or "").startswith("not started:") for t in plan.tasks):
                    result.stop_reason = "budget"
                    result.budget_exceeded = "estimate"
                if not review.accepted:
                    result.error = None  # delivered, but flagged as short of the bar
            except Exception as exc:  # the manager reports, it does not crash the caller
                result.error = f"{type(exc).__name__}: {exc}"
                result.stop_reason = "error"

        for child in self._results.values():
            result.children.append(child)
            result.artifacts.extend(child.artifacts)
        self._job = None
        result.steps = len(self._results)
        result.warnings.extend(self._warnings)
        if plan is not None:
            # What tasks cost here, for the next plan's estimate.
            paid = [t.cost_usd for t in plan.tasks if t.status == "done" and t.cost_usd > 0]
            count, spent = self._seen_tasks
            self._seen_tasks = (count + len(paid), spent + sum(paid))
        # A result's `usage` is its own, and `cost_usd` adds its children's. The
        # children are the sub-agents' work; the manager's own is everything
        # else the job paid for — the plan, the specs the factory wrote, the
        # consolidation, the review, and attempts that were thrown away.
        total, work = self._prior + job.usage, _sum_usage(self._results.values())
        result.usage = Usage(
            input_tokens=max(total.input_tokens - work.input_tokens, 0),
            output_tokens=max(total.output_tokens - work.output_tokens, 0),
            cache_read_tokens=max(total.cache_read_tokens - work.cache_read_tokens, 0),
            cache_write_tokens=max(total.cache_write_tokens - work.cache_write_tokens, 0),
            cost_usd=max(total.cost_usd - work.cost_usd, 0.0),
            calls=max(total.calls - work.calls, 0))
        result.spend = {
            "total_usd": round(total.cost_usd, 6), "work_usd": round(work.cost_usd, 6),
            "overhead_usd": round(result.usage.cost_usd, 6),
            "tokens": total.total_tokens, "calls": total.calls,
            "by_agent": {},
        }
        merged = {k: dict(v) for k, v in self._record.get("by_agent_prior", {}).items()}
        for name, used in job.by_agent.items():
            held = merged.setdefault(name, {"cost_usd": 0.0, "tokens": 0, "calls": 0})
            held["cost_usd"] = round(held["cost_usd"] + used.cost_usd, 6)
            held["tokens"] += used.total_tokens
            held["calls"] += used.calls
        result.spend["by_agent"] = dict(sorted(merged.items()))
        # The last word on the job, for whoever asks after it — or resumes it.
        if "is being run by another process" not in (result.error or ""):
            self._job = job
            await self._checkpoint(
                plan, "finished", status="failed" if result.error else "done",
                deliverable=deliverable, review=review.model_dump(),
                stop_reason=result.stop_reason, error=result.error, spend=result.spend)
            self._job = None
            result.warnings.extend(w for w in self._warnings if w not in result.warnings)
        self._record = None
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
