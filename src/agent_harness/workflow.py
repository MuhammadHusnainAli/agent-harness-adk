"""Declared workflows: the steps are written down, and run as written.

    # refunds.yaml
    name: refunds
    inputs:
      order: {required: true}
      message: {required: true}
    steps:
      - id: charges
        tool: find_charges
        args: {order: "{{ inputs.order }}"}
      - id: classify
        agent: classifier
        input: "Charges: {{ steps.charges.output }}\\nCustomer: {{ inputs.message }}"
        parse: json
        save: verdict
      - if: "state.verdict.duplicate"
        then:
          - tool: issue_refund
            args: {order: "{{ inputs.order }}", amount: "{{ state.verdict.amount }}"}
        else:
          - agent: support
            input: "Explain why order {{ inputs.order }} is not refunded."
    output: "{{ previous }}"

```python
workflow = Workflow.from_file("refunds.yaml", agents=[classifier, support],
                              tools=[find_charges, issue_refund])
result = await workflow.run({"order": "4182", "message": "I was charged twice."})
result.output, result.state, result.steps["classify"].data
```

An agent decides what to do next; a workflow already knows. Use one when the
order of the work is yours to fix — and put agents in the steps where judgement
is needed. A step is one of:

    agent      run an agent on an input          tool      call a tool
    set        write to the shared state         steps     a sequence, as one step
    parallel   branches at the same time         foreach   a body once per item
    loop       a body until a condition holds    if/switch take one branch
    graph      nodes that say what they `needs`, run as soon as they can
    wait       pause        fail   stop with an error        return   finish now

Every step may carry `when` (skip unless), `retry`, `timeout`, `on_error:
continue`, and `save` (keep its result in the state under a name).

Values are templates. `{{ ... }}` holds an expression over `inputs`, `state`,
`steps.<id>` (`.output`, `.data`, `.status`, `.error`) and `previous` — the
output of the step before. A value that is one expression keeps its type; inside
other text it is written out. Expressions are parsed, never evaluated as code.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import time
from collections import ChainMap
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import ConfigurationError, HarnessError, StopRequested, WorkflowError
from .governance.expr import _NAMES, FUNCTIONS, compile_expr
from .harness import Harness
from .runtime.budget import Budget, BudgetGuard
from .runtime.session import Session
from .tools import Tool, ToolRegistry
from .types import Usage, new_id

__all__ = ["Workflow", "WorkflowSpec", "Step", "StepResult", "WorkflowResult",
           "WorkflowEvent", "render"]

#: What a step can be. Exactly one of these keys says which.
KINDS: tuple[str, ...] = ("agent", "tool", "set", "parallel", "foreach", "loop",
                          "if", "switch", "graph", "wait", "fail", "return")

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEMPLATE = re.compile(r"\{\{((?:(?!\}\}).)*)\}\}", re.DOTALL)
_INPUT_KEYS = {"default", "required", "description", "type"}
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "number": (int, float), "integer": (int,), "boolean": (bool,),
    "array": (list, tuple), "object": (dict,),
}

# A few things a workflow says that a policy never needed to.
FUNCTIONS.update({
    "round": round,
    "sum": sum,
    "sorted": sorted,
    "bool": bool,
    "keys": lambda v: list(v) if isinstance(v, Mapping) else [],
    "values": lambda v: list(v.values()) if isinstance(v, Mapping) else [],
    "join": lambda items, sep=", ": sep.join(_text(i) for i in (items or [])),
    "json": lambda v: json.dumps(_plain(v), default=str, ensure_ascii=False),
    "default": lambda v, fallback: fallback if v is None or v == "" else v,
    "range": lambda *a: list(range(*[min(int(n), 10_000) for n in a])),
})


# ----------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------
def _plain(value: Any) -> Any:
    """Something JSON can carry: models become mappings."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return value


def _text(value: Any) -> str:
    """A value as it reads inside other text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (BaseModel, Mapping, list, tuple)):
        return json.dumps(_plain(value), default=str, ensure_ascii=False)
    return str(value)


def _bare(source: str) -> str:
    """A condition may be written with or without the braces."""
    text = source.strip()
    whole = _TEMPLATE.fullmatch(text)
    return whole.group(1).strip() if whole else text


def render(value: Any, context: Mapping[str, Any]) -> Any:
    """Fill the templates in a value — a string, or anything holding strings.

    `"{{ state.total }}"` is the total itself, a number if it is one;
    `"Total: {{ state.total }}"` is text.
    """
    if isinstance(value, str):
        if "{{" not in value:
            return value
        whole = _TEMPLATE.fullmatch(value.strip())
        if whole:
            return compile_expr(whole.group(1).strip())(context)
        return _TEMPLATE.sub(
            lambda m: _text(compile_expr(m.group(1).strip())(context)), value)
    if isinstance(value, Mapping):
        return {key: render(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, context) for item in value]
    return value


def _expressions(value: Any) -> list[str]:
    """Every expression inside a templated value."""
    if isinstance(value, str):
        return [m.strip() for m in _TEMPLATE.findall(value)]
    if isinstance(value, Mapping):
        return [e for item in value.values() for e in _expressions(item)]
    if isinstance(value, list):
        return [e for item in value for e in _expressions(item)]
    return []


# ----------------------------------------------------------------------
# the declaration
# ----------------------------------------------------------------------
class Retry(BaseModel):
    """Try a step again when it fails: `retry: 2`, or the mapping."""

    model_config = ConfigDict(extra="forbid")

    max: int = Field(1, ge=0, le=20)
    delay: float = Field(0.0, ge=0)
    backoff: float = Field(2.0, ge=1)


class Loop(BaseModel):
    """How long a loop goes on: `loop: 5`, or the mapping."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    max: int = Field(10, ge=1, le=10_000)
    until: str | None = None                       # checked after each pass
    while_: str | None = Field(None, alias="while")  # checked before each pass


class Case(BaseModel):
    """One branch of a `switch`. Without `when`, it is the default."""

    model_config = ConfigDict(extra="forbid")

    when: str | None = None
    steps: list[Step] = Field(default_factory=list)


class Step(BaseModel):
    """One step, as declared. Which kind it is, is the key it carries."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = ""
    description: str = ""
    # ---- what it is: exactly one -------------------------------------
    agent: str | dict[str, Any] | None = None
    tool: str | None = None
    assign: dict[str, Any] | None = Field(None, alias="set")
    parallel: list[Step] | None = None
    foreach: Any = None
    loop: Loop | None = None
    if_: str | None = Field(None, alias="if")
    switch: list[Case] | None = None
    graph: list[Step] | None = None
    wait: float | str | None = None
    fail: str | None = None
    return_: Any = Field(None, alias="return")
    #: A body for `foreach` and `loop`; on its own, a sequence run as one step.
    steps: list[Step] | None = None
    # ---- what goes with a kind ---------------------------------------
    input: Any = None                 # agent: what it is asked
    attachments: Any = None           # agent: files to send with it
    version: str | None = None        # agent: which version of it
    thread: bool = False              # agent: carry its conversation through the run
    parse: Literal["json"] | None = None
    args: dict[str, Any] = Field(default_factory=dict)        # tool
    as_: str = Field("item", alias="as")                      # foreach
    concurrency: int | None = Field(None, ge=1)               # parallel, foreach, graph
    then: list[Step] | None = None                            # if
    else_: list[Step] | None = Field(None, alias="else")      # if
    needs: list[str] = Field(default_factory=list)            # a node in a graph
    join: Literal["all", "any"] = "all"
    # ---- what any step may carry -------------------------------------
    when: str | None = None
    save: str | None = None
    retry: Retry | None = None
    timeout: float | None = Field(None, gt=0)
    on_error: Literal["fail", "continue"] = "fail"

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if isinstance(data.get("retry"), int) and not isinstance(data["retry"], bool):
            data["retry"] = {"max": data["retry"]}
        loop = data.get("loop")
        if isinstance(loop, int) and not isinstance(loop, bool):
            data["loop"] = {"max": loop}
        elif loop is True:
            data["loop"] = {}
        return data

    @property
    def kind(self) -> str:
        return self._kinds()[0]

    def _kinds(self) -> list[str]:
        present = {"agent": self.agent, "tool": self.tool, "set": self.assign,
                   "parallel": self.parallel, "foreach": self.foreach,
                   "loop": self.loop, "if": self.if_, "switch": self.switch,
                   "graph": self.graph, "wait": self.wait, "fail": self.fail}
        found = [kind for kind, value in present.items() if value is not None]
        if "return_" in self.model_fields_set:
            found.append("return")
        if not found and self.steps is not None:
            found.append("steps")
        return found or [""]

    def children(self) -> list[tuple[str, list[Step]]]:
        """Its bodies, each with a word for where it hangs."""
        out: list[tuple[str, list[Step]]] = []
        for label, body in (("parallel", self.parallel), ("graph", self.graph),
                            ("steps", self.steps), ("then", self.then),
                            ("else", self.else_)):
            if body:
                out.append((label, body))
        for index, case in enumerate(self.switch or []):
            out.append((f"case {index + 1}" if case.when else "default", case.steps))
        return out


Case.model_rebuild()
Step.model_rebuild()


class WorkflowSpec(BaseModel):
    """A whole workflow, as a file declares it."""

    model_config = ConfigDict(extra="forbid")

    name: str = "workflow"
    description: str = ""
    version: str = "1"
    #: What it takes: a name to its default, or to
    #: {default, required, type, description}.
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: What the shared state starts as.
    state: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] | None = None
    #: Instead of `steps`: nodes that say what they need, run as a graph.
    graph: list[Step] | None = None
    #: What the run hands back. Left out, it is the last step's output.
    output: Any = None
    budget: Budget | None = None
    timeout: float | None = Field(None, gt=0)
    #: How many steps one run may execute — a loop that never ends, ends here.
    max_steps: int = Field(1000, ge=1)
    #: How many branches run at once, where a step does not say.
    concurrency: int = Field(8, ge=1)
    # ---- agents declared with it, exactly as in a blueprint ----------
    agents: dict[str, Any] = Field(default_factory=dict)
    subagents: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    guardrails: dict[str, Any] = Field(default_factory=dict)

    def root(self) -> Step:
        """The whole of it as one step."""
        if (self.steps is None) == (self.graph is None):
            raise ConfigurationError(
                f"workflow {self.name!r} needs `steps` or `graph` — one of them")
        if self.graph is not None:
            return Step(id="_root", graph=self.graph)
        return Step(id="_root", steps=self.steps)

    def input_specs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, value in self.inputs.items():
            if isinstance(value, dict) and set(value) <= _INPUT_KEYS:
                out[name] = dict(value)
            else:
                out[name] = {"default": value}
        return out


# ----------------------------------------------------------------------
# what a run produces
# ----------------------------------------------------------------------
StepStatus = Literal["done", "skipped", "failed"]


class StepResult(BaseModel):
    """What one step came to. `steps.<id>` in a template is one of these."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    kind: str = ""
    status: StepStatus = "done"
    output: Any = None
    #: The structured form of the output, when there is one: an agent's output
    #: contract, parsed JSON, or what a tool returned.
    data: Any = None
    error: str | None = None
    attempts: int = 1
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    #: The agent that answered, for an agent step (after any handoff).
    agent: str = ""
    #: How many passes a loop or a foreach made, and whether a loop ran out.
    iterations: int = 0
    exhausted: bool = False
    #: The branch an `if` or a `switch` took.
    branch: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "failed"


class WorkflowEvent(BaseModel):
    """One beat of a running workflow."""

    type: Literal["workflow_start", "step_start", "step_end", "step_skipped",
                  "step_failed", "step_retry", "workflow_end"]
    step: str = ""
    kind: str = ""
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    """The one object a caller gets back from `workflow.run()`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow: str = ""
    run_id: str = Field(default_factory=lambda: new_id("wf"))
    status: Literal["done", "failed", "stopped"] = "done"
    output: Any = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    steps: dict[str, StepResult] = Field(default_factory=dict)
    error: str | None = None
    #: The step the failure started in.
    failed_step: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    steps_run: int = 0
    duration_ms: float = 0.0
    artifacts: list[Any] = Field(default_factory=list)
    trace_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return _text(self.output)


# ----------------------------------------------------------------------
# running
# ----------------------------------------------------------------------
class _Return(Exception):
    """A `return` step: the workflow is finished, with this."""

    def __init__(self, value: Any) -> None:
        super().__init__("return")
        self.value = value


class _StepFailed(WorkflowError):
    """A step failed and was not told to carry on."""


@dataclass
class _Scope:
    """What a template can see from where a step stands."""

    inputs: dict[str, Any]
    state: dict[str, Any]
    steps: Any                         # id → StepResult
    names: dict[str, Any] = field(default_factory=dict)

    def context(self) -> dict[str, Any]:
        return {"previous": None, **self.names, "inputs": self.inputs,
                "state": self.state, "steps": self.steps}

    def child(self, *, own_steps: bool = False, **names: Any) -> _Scope:
        """A place for a branch to stand: the same state, its own `previous`.
        With `own_steps`, what its steps come to is not seen outside it — one
        pass of a `foreach` does not read another's."""
        steps = ChainMap({}, self.steps) if own_steps else self.steps
        return _Scope(self.inputs, self.state, steps, {**self.names, **names})


@dataclass
class _Run:
    id: str
    guard: BudgetGuard | None
    emit: Callable[[WorkflowEvent], None]
    results: dict[str, StepResult] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    count: int = 0
    failed: str = ""
    threads: dict[str, Session] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)


class Workflow:
    """A declared workflow, with the agents and tools its steps name.

    Args:
        spec: the declaration — a `WorkflowSpec`, or the mapping a file holds.
        agents: the agents its `agent` steps name: a list, or a name → agent
            mapping. Agents the file declares itself are built for you.
        tools: the tools its `tool` steps call, and that its declared agents
            may be given.
        harness: the harness it runs on. Left out, the one its agents share.
        blueprint: a `Blueprint` to find agents in that were not handed over.
    """

    def __init__(self, spec: WorkflowSpec | Mapping[str, Any], *,
                 agents: Iterable[Any] | Mapping[str, Any] | None = None,
                 tools: Iterable[Any] | ToolRegistry = (),
                 harness: Harness | None = None, blueprint: Any = None) -> None:
        if not isinstance(spec, WorkflowSpec):
            try:
                spec = WorkflowSpec(**spec)
            except ValidationError as exc:
                raise ConfigurationError(_explain(exc)) from None
        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self.root = spec.root()
        self.ids: dict[str, Step] = {}
        _number(self.root, self.ids, counter=[0])
        _check(self.root, self.ids, spec)

        given: dict[str, Any] = (dict(agents) if isinstance(agents, Mapping)
                                 else {a.name: a for a in agents or ()})
        if harness is None:
            harness = next((a.harness for a in given.values()
                            if getattr(a, "harness", None) is not None), None) or Harness()
        self.harness = harness
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.agents = self._agents(given, blueprint)
        for step in self.ids.values():
            if step.tool is not None and step.tool not in self.tools:
                self.tools.add(self._import(step))
        self._runner: Any = None

    # ---- loading ---------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> Workflow:
        """Load a `.yaml`, `.yml` or `.json` file."""
        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"no workflow at {file}")
        return cls.from_text(file.read_text(encoding="utf-8"),
                             fmt=file.suffix.lstrip("."), **kwargs)

    @classmethod
    def from_text(cls, text: str, *, fmt: str = "yaml", **kwargs: Any) -> Workflow:
        try:
            if fmt == "json":
                data = json.loads(text)
            else:
                import yaml

                data = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigurationError(f"the workflow could not be read: {exc}") from None
        if not isinstance(data, dict):
            raise ConfigurationError("a workflow must be a mapping at the top level")
        return cls(data, **kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **kwargs: Any) -> Workflow:
        return cls(data, **kwargs)

    def _agents(self, given: dict[str, Any], blueprint: Any) -> dict[str, Any]:
        """Every agent a step names: handed in, or built from a declaration."""
        spec = self.spec
        wanted: dict[str, Step] = {}
        inline: dict[str, Any] = {}
        for step in self.ids.values():
            if isinstance(step.agent, dict):
                inline[step.id] = step.agent      # declared where it is used
                wanted[step.id] = step
            elif step.agent is not None:
                wanted.setdefault(step.agent, step)
        missing = [name for name in wanted if name not in given]
        if not missing:
            return {name: given[name] for name in wanted}

        from .blueprint import Blueprint

        try:
            declared = Blueprint(
                agents={**spec.agents, **inline}, subagents=spec.subagents,
                prompts=spec.prompts, defaults=spec.defaults,
                guardrails=spec.guardrails)
        except ValidationError as exc:
            raise ConfigurationError(_explain(exc)) from None
        if blueprint is not None:
            declared = blueprint.model_copy(update={
                "agents": {**blueprint.agents, **declared.agents},
                "subagents": {**blueprint.subagents, **declared.subagents},
                "prompts": {**blueprint.prompts, **declared.prompts},
                "guardrails": {**blueprint.guardrails, **declared.guardrails},
                "defaults": {**blueprint.defaults, **declared.defaults}})
        built: dict[str, Any] = {}
        out: dict[str, Any] = {}
        for name, step in wanted.items():
            if name in given:
                out[name] = given[name]
            elif name in declared.agents:
                out[name] = declared._build(name, built, tools=self.tools,
                                            harness=self.harness)
            else:
                known = ", ".join(sorted({*given, *declared.agents})) or "none"
                raise ConfigurationError(
                    f"workflow {self.name!r}, step {step.id!r}: no agent named "
                    f"{name!r}. Pass it in agents=[...], or declare it under "
                    f"`agents:`. Known: {known}")
        return out

    def _import(self, step: Step) -> Tool:
        name = step.tool or ""
        if ":" in name or "." in name:
            from .blueprint import _import_tool

            found = _import_tool(name)
            step.tool = found.name
            return found
        known = ", ".join(self.tools.names) or "none"
        raise ConfigurationError(
            f"workflow {self.name!r}, step {step.id!r}: no tool named {name!r}. "
            f"Pass it in tools=[...], or name it by import path "
            f"('myapp.tools:{name}'). Known: {known}")

    @property
    def runner(self) -> Any:
        """The agent a `tool` step is called through — so the call passes the
        same guardrails, hooks, permission gate and audit trail as any other."""
        if self._runner is None:
            from .agent import Agent

            self._runner = Agent(self.name, description=self.description
                                 or f"the {self.name} workflow",
                                 tools=list(self.tools), harness=self.harness,
                                 memory=False, persist_session=False)
        return self._runner

    # ---- describing --------------------------------------------------------
    def describe(self) -> str:
        """The workflow as an outline: what runs, in what shape."""
        lines = [f"{self.name}" + (f" — {self.description}" if self.description else "")]
        specs = self.spec.input_specs()
        if specs:
            lines.append("inputs: " + ", ".join(
                name + ("" if spec.get("required") else "?") for name, spec in specs.items()))

        def walk(steps: list[Step], depth: int) -> None:
            for step in steps:
                detail = {"agent": step.agent if isinstance(step.agent, str) else step.id,
                          "tool": step.tool, "foreach": f"as {step.as_}",
                          "if": step.if_, "wait": step.wait}.get(step.kind)
                extras = [f"needs {', '.join(step.needs)}" if step.needs else "",
                          f"when {step.when}" if step.when else "",
                          f"until {step.loop.until}" if step.loop and step.loop.until
                          else "",
                          f"→ state.{step.save}" if step.save else ""]
                tail = " · ".join(e for e in extras if e)
                lines.append("  " * depth + f"- {step.id}: {step.kind}"
                             + (f" {detail}" if detail else "")
                             + (f"  ({tail})" if tail else ""))
                bodies = step.children()
                for label, body in bodies:
                    if len(bodies) > 1 or label in ("then", "else", "default"):
                        lines.append("  " * (depth + 1) + f"{label}:")
                        walk(body, depth + 2)
                    else:
                        walk(body, depth + 1)

        walk(self.root.graph or self.root.steps or [], 1)
        return "\n".join(lines)

    # ---- inputs ------------------------------------------------------------
    def _inputs(self, given: Any) -> dict[str, Any]:
        specs = self.spec.input_specs()
        if given is None:
            given = {}
        elif not isinstance(given, Mapping):
            # One thing handed over: it is the input, whatever it is called.
            names = list(specs) or ["input"]
            if len(names) != 1:
                raise ConfigurationError(
                    f"workflow {self.name!r} takes {', '.join(names)} — pass a mapping")
            given = {names[0]: given}
        if not specs:
            return dict(given)
        unknown = sorted(set(given) - set(specs))
        if unknown:
            raise ConfigurationError(
                f"workflow {self.name!r} does not take {', '.join(unknown)}; it takes "
                f"{', '.join(specs)}")
        out: dict[str, Any] = {}
        for name, spec in specs.items():
            if name in given:
                value = given[name]
            elif spec.get("required"):
                raise ConfigurationError(
                    f"workflow {self.name!r} needs the input {name!r}")
            else:
                value = spec.get("default")
            kinds = _TYPES.get(str(spec.get("type") or ""))
            if kinds and value is not None and (
                    not isinstance(value, kinds)
                    or isinstance(value, bool) and bool not in kinds):
                raise ConfigurationError(
                    f"workflow {self.name!r}: input {name!r} must be "
                    f"{spec['type']} — got {type(value).__name__}")
            out[name] = value
        return out

    # ---- running -----------------------------------------------------------
    async def run(self, inputs: Any = None, *, state: Mapping[str, Any] | None = None,
                  on_event: Callable[[WorkflowEvent], Any] | None = None,
                  run_id: str | None = None) -> WorkflowResult:
        """Run it to the end and return what it came to.

        A step that fails ends the run with `result.error` set — it is not
        raised. What is raised is a mistake in the call: an input that is
        missing, or one the workflow does not take.
        """
        harness = self.harness
        values = self._inputs(inputs)
        result = WorkflowResult(workflow=self.name, run_id=run_id or new_id("wf"),
                                inputs=values)
        shared: dict[str, Any] = json.loads(json.dumps(_plain(self.spec.state)))
        shared.update(state or {})

        def emit(event: WorkflowEvent) -> None:
            if on_event is not None:
                on_event(event)

        guard = (BudgetGuard(self.spec.budget, parent=harness.guard)
                 if self.spec.budget is not None else None)
        run = _Run(id=result.run_id, guard=guard, emit=emit)
        scope = _Scope(values, shared, {})
        began = time.perf_counter()

        with harness.tracer.span(f"workflow:{self.name}", kind="workflow") as span:
            result.trace_id = span.trace_id
            harness.audit.record(self.name, "workflow_start", target=self.name,
                                 run_id=run.id, inputs=sorted(values))
            await harness.journal.write("workflow", f"started {self.name}",
                                        agent=self.name, run_id=run.id)
            emit(WorkflowEvent(type="workflow_start", text=self.name,
                               data={"run_id": run.id, "inputs": values}))
            try:
                start = await harness.hooks.emit(
                    "workflow_start", agent=self.name, run_id=run.id,
                    workflow=self.name, inputs=values)
                if start.blocked:
                    raise WorkflowError(f"{self.name} may not run: {start.reason}")
                last = await self._within(self._body(self.root, scope, run),
                                          self.spec.timeout, self.name)
                context = {**scope.context(), "previous": last}
                result.output = (render(self.spec.output, context)
                                 if self.spec.output is not None else last)
            except _Return as done:
                result.output = done.value
            except StopRequested as exc:
                result.status, result.error = "stopped", str(exc)
            except Exception as exc:
                result.status = "failed"
                result.error = (str(exc) if isinstance(exc, HarnessError)
                                else f"{type(exc).__name__}: {exc}")
                result.failed_step = run.failed
            span.set(status=result.status, steps=run.count, cost_usd=run.cost)

        result.state = shared
        result.steps = dict(run.results)
        result.usage, result.cost_usd = run.usage, round(run.cost, 8)
        result.steps_run = run.count
        result.artifacts = run.artifacts
        result.duration_ms = (time.perf_counter() - began) * 1000
        harness.audit.record(self.name, "workflow_end", target=self.name,
                             run_id=run.id, steps=run.count, cost_usd=result.cost_usd,
                             decision="ok" if result.ok else result.status,
                             failed_step=result.failed_step)
        await harness.journal.write(
            "workflow", f"{self.name} {result.status}"
            + (f": {result.error}" if result.error else ""), agent=self.name,
            run_id=run.id, steps=run.count, cost_usd=result.cost_usd)
        await harness.hooks.emit("workflow_end", agent=self.name, run_id=run.id,
                                 workflow=self.name, result=result)
        emit(WorkflowEvent(type="workflow_end", text=result.status,
                           data={"result": result}))
        return result

    def run_sync(self, inputs: Any = None, **kwargs: Any) -> WorkflowResult:
        """Blocking wrapper for scripts and notebooks."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(inputs, **kwargs))
        raise RuntimeError(
            "run_sync() cannot be called from inside a running event loop — await "
            "workflow.run() instead")

    async def stream(self, inputs: Any = None,
                     **kwargs: Any) -> AsyncIterator[WorkflowEvent]:
        """The run as it happens: a `step_start` and a `step_end` for every
        step, ending with `workflow_end`, whose `data["result"]` is the result.

        Stop listening and the run is cancelled.
        """
        events: asyncio.Queue[WorkflowEvent | None] = asyncio.Queue()

        async def work() -> None:
            try:
                await self.run(inputs, on_event=events.put_nowait, **kwargs)
            finally:
                events.put_nowait(None)

        task = asyncio.create_task(work())
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield event
            await task                 # a mistake in the call is raised here
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose the workflow as a tool an agent can call."""
        workflow = self
        specs = self.spec.input_specs() or {"input": {"required": True}}

        async def call_workflow(**inputs: Any) -> str:
            result = await workflow.run(inputs)
            if not result.ok:
                return f"[{result.status}] {result.error}"
            return _text(result.output)

        call_workflow.__name__ = name or self.name
        return Tool(
            call_workflow, name=name or self.name,
            description=description or self.description or f"Run the {self.name} "
            "workflow.",
            parameters={
                "type": "object",
                "properties": {
                    key: {**({"type": spec["type"]} if spec.get("type") in _TYPES
                             else {}),
                          "description": str(spec.get("description") or key)}
                    for key, spec in specs.items()},
                "required": [k for k, s in specs.items() if s.get("required")],
            },
            tags=["workflow"],
        )

    # ---- the engine ----------------------------------------------------------
    @staticmethod
    async def _within(work: Awaitable[Any], seconds: float | None, what: str) -> Any:
        if seconds is None:
            return await work
        try:
            return await asyncio.wait_for(work, seconds)
        except (TimeoutError, asyncio.TimeoutError):
            raise WorkflowError(f"{what} timed out after {seconds:g}s") from None

    async def _fan(self, jobs: list[Callable[[], Awaitable[Any]]],
                   limit: int | None) -> list[Any]:
        """Run branches at once, capped. The first failure stops the rest."""
        gate = asyncio.Semaphore(limit or self.spec.concurrency)

        async def one(job: Callable[[], Awaitable[Any]]) -> Any:
            async with gate:
                return await job()

        tasks = [asyncio.create_task(one(job)) for job in jobs]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _sequence(self, steps: list[Step], scope: _Scope, run: _Run) -> Any:
        """One after another. Each sees the one before as `previous`."""
        last = scope.names.get("previous")
        for step in steps:
            outcome = await self._step(step, scope, run)
            if outcome.status == "done":
                last = scope.names["previous"] = outcome.output
        return last

    async def _body(self, step: Step, scope: _Scope, run: _Run) -> Any:
        """The root: a sequence or a graph, with no step of its own around it."""
        if step.graph is not None:
            output, _ = await self._graph(step, scope, run)
            return output
        return await self._sequence(step.steps or [], scope, run)

    async def _step(self, step: Step, scope: _Scope, run: _Run) -> StepResult:
        """One step, with everything any step may carry around it."""
        harness = self.harness
        harness.control.check(f"{self.name} step {step.id}")
        outcome = StepResult(id=step.id, kind=step.kind)

        def keep() -> StepResult:
            run.results[step.id] = outcome
            scope.steps[step.id] = outcome
            return outcome

        if step.when is not None and not compile_expr(_bare(step.when))(scope.context()):
            outcome.status = "skipped"
            run.emit(WorkflowEvent(type="step_skipped", step=step.id, kind=step.kind,
                                   text=step.when))
            return keep()

        run.count += 1
        if run.count > self.spec.max_steps:
            run.failed = run.failed or step.id
            raise WorkflowError(
                f"{self.name} ran {self.spec.max_steps} steps without finishing — "
                "a loop that does not end? Raise `max_steps` if it is meant to")
        run.emit(WorkflowEvent(type="step_start", step=step.id, kind=step.kind,
                               text=step.description))
        began = time.perf_counter()
        retry = step.retry or Retry(max=0)
        problem, nested = "", False
        with harness.tracer.span(f"step:{step.id}", kind="workflow_step",
                                 step_kind=step.kind) as span:
            for attempt in range(1, retry.max + 2):
                outcome.attempts = attempt
                try:
                    asked = await harness.hooks.emit(
                        "workflow_step", agent=self.name, run_id=run.id,
                        workflow=self.name, step_id=step.id, kind=step.kind,
                        attempt=attempt)
                    if asked.blocked:
                        raise WorkflowError(f"not permitted: {asked.reason}")
                    await self._within(self._execute(step, scope, run, outcome),
                                       step.timeout, f"step {step.id!r}")
                    problem = ""
                    break
                except (_Return, StopRequested, asyncio.CancelledError):
                    raise
                except Exception as exc:
                    run.failed = run.failed or step.id
                    # A step inside this one failed: its message says it all.
                    nested = isinstance(exc, _StepFailed)
                    problem = (str(exc) if isinstance(exc, HarnessError)
                               else f"{type(exc).__name__}: {exc}")
                    if attempt <= retry.max and harness.control.may_start():
                        run.emit(WorkflowEvent(type="step_retry", step=step.id,
                                               kind=step.kind, text=problem,
                                               data={"attempt": attempt}))
                        await asyncio.sleep(retry.delay * retry.backoff ** (attempt - 1))
                        continue
                    break
            outcome.duration_ms = (time.perf_counter() - began) * 1000
            span.set(status="failed" if problem else "done", attempts=outcome.attempts,
                     cost_usd=outcome.cost_usd)

        if problem:
            outcome.status, outcome.error = "failed", problem
            keep()
            harness.audit.record(self.name, "workflow_step", target=step.id,
                                 decision="error", run_id=run.id, kind=step.kind,
                                 reason=problem[:300])
            await harness.journal.write("error", f"step {step.id} failed: {problem}",
                                        agent=self.name, run_id=run.id)
            run.emit(WorkflowEvent(type="step_failed", step=step.id, kind=step.kind,
                                   text=problem,
                                   data={"continues": step.on_error == "continue"}))
            if step.on_error == "continue":
                run.failed = ""
                return outcome
            raise _StepFailed(problem if nested
                              else f"step {step.id!r} failed: {problem}")

        if outcome.attempts > 1:
            run.failed = ""                      # it got there in the end
        if step.save:
            scope.state[step.save] = (outcome.data if outcome.data is not None
                                      else outcome.output)
        harness.audit.record(self.name, "workflow_step", target=step.id,
                             decision="ok", run_id=run.id, kind=step.kind,
                             attempts=outcome.attempts)
        run.emit(WorkflowEvent(
            type="step_end", step=step.id, kind=step.kind,
            text=_text(outcome.output)[:400],
            data={"duration_ms": round(outcome.duration_ms, 1),
                  "cost_usd": outcome.cost_usd, "attempts": outcome.attempts}))
        return keep()

    async def _execute(self, step: Step, scope: _Scope, run: _Run,
                       outcome: StepResult) -> None:
        """Do what the step is, and write what it came to on `outcome`."""
        kind = step.kind
        context = scope.context()

        if kind == "agent":
            await self._agent(step, context, run, outcome)

        elif kind == "tool":
            args = render(step.args, context)
            called = await self.runner.call_tool(step.tool, args, run_id=run.id,
                                                 guard=run.guard)
            if called.is_error:
                raise WorkflowError(called.content)
            outcome.output = called.value if called.value is not None else called.content
            if not isinstance(outcome.output, str):
                outcome.output = _plain(outcome.output)
                outcome.data = outcome.output

        elif kind == "set":
            values = render(step.assign, context)
            scope.state.update(values)
            outcome.output = values

        elif kind == "wait":
            seconds = render(step.wait, context)
            try:
                await asyncio.sleep(max(0.0, float(seconds)))
            except (TypeError, ValueError):
                raise WorkflowError(f"cannot wait {seconds!r} seconds") from None
            outcome.output = scope.names.get("previous")

        elif kind == "fail":
            raise WorkflowError(_text(render(step.fail, context)) or "failed")

        elif kind == "return":
            raise _Return(render(step.return_, context))

        elif kind == "steps":
            outcome.output = await self._sequence(step.steps or [], scope.child(), run)

        elif kind == "parallel":
            branches = step.parallel or []
            results = await self._fan(
                [lambda b=branch: self._step(b, scope.child(), run)
                 for branch in branches], step.concurrency)
            outcome.output = {r.id: r.output for r in results if r.status == "done"}

        elif kind == "foreach":
            items = render(step.foreach, context)
            if isinstance(items, Mapping):
                items = [{"key": k, "value": v} for k, v in items.items()]
            elif isinstance(items, int) and not isinstance(items, bool):
                items = list(range(items))
            elif items is None:
                items = []
            if isinstance(items, str) or not isinstance(items, (list, tuple)):
                raise WorkflowError(
                    f"foreach needs a list — got {type(items).__name__}")

            async def one(index: int, item: Any) -> Any:
                inner = scope.child(own_steps=True, **{step.as_: item, "index": index})
                return await self._sequence(step.steps or [], inner, run)

            outcome.output = await self._fan(
                [lambda i=i, item=item: one(i, item) for i, item in enumerate(items)],
                step.concurrency or 1)
            outcome.iterations = len(items)

        elif kind == "loop":
            loop = step.loop or Loop()
            inner = scope.child()
            last: Any = None
            outcome.exhausted = True
            for index in range(loop.max):
                inner.names["loop"] = {"index": index, "iteration": index + 1,
                                       "max": loop.max}
                if loop.while_ is not None and not compile_expr(
                        _bare(loop.while_))(inner.context()):
                    outcome.exhausted = False
                    break
                self.harness.control.check(f"{self.name} loop {step.id}")
                last = await self._sequence(step.steps or [], inner, run)
                outcome.iterations = index + 1
                if loop.until is not None and compile_expr(
                        _bare(loop.until))(inner.context()):
                    outcome.exhausted = False
                    break
            if loop.until is None and loop.while_ is None:
                outcome.exhausted = False        # a counted loop simply finished
            outcome.output = last

        elif kind == "if":
            taken = bool(compile_expr(_bare(step.if_ or ""))(context))
            outcome.branch = "then" if taken else "else"
            body = step.then if taken else step.else_
            outcome.output = (await self._sequence(body, scope.child(), run)
                              if body else None)

        elif kind == "switch":
            outcome.output = None
            for index, case in enumerate(step.switch or []):
                if case.when is None or compile_expr(_bare(case.when))(context):
                    outcome.branch = (f"case {index + 1}" if case.when is not None
                                      else "default")
                    outcome.output = await self._sequence(case.steps, scope.child(),
                                                          run)
                    break

        elif kind == "graph":
            outcome.output, _ = await self._graph(step, scope, run)

    async def _agent(self, step: Step, context: dict[str, Any], run: _Run,
                     outcome: StepResult) -> None:
        name = step.agent if isinstance(step.agent, str) else step.id
        agent = self.agents[name]
        if step.input is not None:
            asked = render(step.input, context)
        elif context.get("previous") is not None:
            asked = context["previous"]          # the step before feeds this one
        else:
            inputs = context["inputs"]
            asked = next(iter(inputs.values())) if len(inputs) == 1 else inputs
        kwargs: dict[str, Any] = {}
        if step.thread:
            # One conversation for this agent through the whole run.
            kwargs["session"] = run.threads.setdefault(name, Session(agent=agent.name))
        else:
            kwargs["messages"] = []              # each step is its own task
        if step.version:
            kwargs["version"] = step.version
        if step.attachments is not None:
            files = render(step.attachments, context)
            kwargs["attachments"] = files if isinstance(files, list) else [files]
        if run.guard is not None:
            kwargs["guard"] = run.guard.child(agent.budget)

        result = await agent.run(_text(asked), **kwargs)
        run.usage += result.usage
        for child in result.children:
            run.usage += child.usage
        run.cost += result.cost_usd
        run.artifacts.extend(a for a in result.artifacts if a not in run.artifacts)
        outcome.cost_usd = round(outcome.cost_usd + result.cost_usd, 8)
        outcome.agent = result.agent
        if result.error:
            raise WorkflowError(f"{result.agent or name}: {result.error}")
        if result.budget_exceeded:
            raise WorkflowError(
                f"{result.agent or name} stopped at its budget "
                f"({result.budget_exceeded})")
        outcome.output = result.output
        if result.data is not None:
            outcome.data = _plain(result.data)
        elif step.parse == "json":
            from .agent import _extract_json

            outcome.data = _extract_json(result.output)
            if outcome.data is None:
                raise WorkflowError(
                    f"{name} was to answer in JSON and did not: "
                    f"{result.output[:200]!r}")

    async def _graph(self, step: Step, scope: _Scope,
                     run: _Run) -> tuple[dict[str, Any], dict[str, StepResult]]:
        """Nodes that say what they need. Each starts the moment its needs are
        met, so everything that can run at once does."""
        nodes = {node.id: node for node in step.graph or []}
        settled: dict[str, StepResult] = {}
        running: dict[asyncio.Task[StepResult], str] = {}
        waiting = dict(nodes)
        gate = asyncio.Semaphore(step.concurrency or self.spec.concurrency)

        async def one(node: Step) -> StepResult:
            met = [settled[n] for n in node.needs]
            arrived = [r for r in met if r.status == "done"]
            enough = (len(arrived) == len(met) if node.join == "all" else bool(arrived))
            if met and not enough:
                # What it was waiting for never came: it does not run either.
                skipped = StepResult(id=node.id, kind=node.kind, status="skipped")
                run.results[node.id] = scope.steps[node.id] = skipped
                run.emit(WorkflowEvent(
                    type="step_skipped", step=node.id, kind=node.kind,
                    text="needs " + ", ".join(
                        r.id for r in met if r.status != "done")))
                return skipped
            previous = (arrived[0].output if len(node.needs) == 1 and arrived
                        else {r.id: r.output for r in arrived} if arrived
                        else scope.names.get("previous"))
            async with gate:
                return await self._step(node, scope.child(previous=previous), run)

        try:
            while waiting or running:
                for name in [n for n, node in waiting.items()
                             if all(need in settled for need in node.needs)]:
                    running[asyncio.create_task(one(waiting.pop(name)))] = name
                finished, _ = await asyncio.wait(
                    list(running), return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    settled[running.pop(task)] = task.result()
        except BaseException:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            raise
        return ({name: r.output for name, r in settled.items() if r.status == "done"},
                settled)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<Workflow {self.name} steps={len(self.ids)}>"


# ----------------------------------------------------------------------
# checking a declaration before anything runs
# ----------------------------------------------------------------------
def _explain(exc: ValidationError) -> str:
    """A pydantic error as one line a person who wrote the YAML can act on."""
    problems = []
    for error in exc.errors()[:6]:
        where = ".".join(str(p) for p in error["loc"])
        problems.append(f"{where}: {error['msg']}")
    return "the workflow is not valid — " + "; ".join(problems)


def _number(step: Step, ids: dict[str, Step], *, counter: list[int]) -> None:
    """Give every step a name, and refuse two with the same one."""
    for _, body in step.children():
        for child in body:
            if not child.id:
                counter[0] += 1
                child.id = f"{child.kind or 'step'}_{counter[0]}"
            elif not _ID.match(child.id):
                raise ConfigurationError(
                    f"step id {child.id!r} must be a plain name — letters, digits "
                    "and underscores — so templates can say steps." + "name")
            if child.id in ids:
                raise ConfigurationError(f"two steps are called {child.id!r}")
            ids[child.id] = child
            _number(child, ids, counter=counter)


def _check(root: Step, ids: dict[str, Step], spec: WorkflowSpec) -> None:
    """Everything that can be known to be wrong without running it."""
    inputs = set(spec.input_specs())
    base = {"inputs", "state", "steps", "previous"}

    def expression(source: str, names: set[str], where: str) -> None:
        source = _bare(source)
        compile_expr(source)                         # syntax, and what is allowed
        tree = ast.parse(source, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in (
                    names | set(FUNCTIONS) | set(_NAMES)):
                raise ConfigurationError(
                    f"step {where!r}: {source!r} uses {node.id!r}, which is not "
                    f"known here. Known: {', '.join(sorted(names))}")
            target = key = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                target, key = node.value.id, node.attr
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                  and isinstance(node.slice, ast.Constant)):
                target, key = node.value.id, node.slice.value
            if target == "steps" and key not in ids:
                raise ConfigurationError(
                    f"step {where!r}: {source!r} refers to steps.{key}, and no "
                    f"step is called that. Steps: {', '.join(sorted(ids))}")
            if target == "inputs" and inputs and key not in inputs:
                raise ConfigurationError(
                    f"step {where!r}: {source!r} refers to inputs.{key}, which "
                    f"the workflow does not declare. Inputs: "
                    f"{', '.join(sorted(inputs))}")

    def templates(value: Any, names: set[str], where: str) -> None:
        for source in _expressions(value):
            expression(source, names, where)

    def walk(step: Step, names: set[str], *, node: bool = False) -> None:
        kinds = step._kinds()
        if kinds == [""]:
            raise ConfigurationError(
                f"step {step.id!r} does nothing — give it one of: "
                f"{', '.join(KINDS)}, or `steps`")
        if len(kinds) > 1:
            raise ConfigurationError(
                f"step {step.id!r} is {' and '.join(kinds)} at once — a step is "
                "one thing; nest the other inside it")
        kind = kinds[0]
        if step.needs and not node:
            raise ConfigurationError(
                f"step {step.id!r} has `needs`, which only a node of a `graph` "
                "can have")
        if kind in ("foreach", "loop") and not step.steps:
            raise ConfigurationError(f"step {step.id!r}: a {kind} needs `steps` to run")
        if kind not in ("foreach", "loop", "steps") and step.steps is not None:
            raise ConfigurationError(
                f"step {step.id!r}: `steps` belongs to a foreach, a loop, or a "
                f"sequence — not to {kind}")
        if kind != "if" and (step.then is not None or step.else_ is not None):
            raise ConfigurationError(
                f"step {step.id!r}: `then` and `else` belong to an `if`")
        if kind == "if" and not (step.then or step.else_):
            raise ConfigurationError(f"step {step.id!r}: an `if` needs a `then`")
        if kind == "switch":
            cases = step.switch or []
            if any(c.when is None for c in cases[:-1]):
                raise ConfigurationError(
                    f"step {step.id!r}: a case with no `when` is the default, and "
                    "must come last")
        if step.save is not None and not _ID.match(step.save):
            raise ConfigurationError(
                f"step {step.id!r}: `save` names a key in the state — a plain name")
        if kind == "foreach" and (not _ID.match(step.as_) or step.as_ in base):
            raise ConfigurationError(
                f"step {step.id!r}: `as: {step.as_}` is not a name an item can have")

        for source in (step.when, step.if_):
            if source is not None:
                expression(source, names, step.id)
        templates([step.input, step.args, step.assign, step.wait, step.fail,
                   step.return_, step.attachments, step.foreach],
                  names, step.id)
        for case in step.switch or []:
            if case.when is not None:
                expression(case.when, names, step.id)

        inner = set(names)
        if kind == "foreach":
            inner |= {step.as_, "index"}
        if kind == "loop":
            inner |= {"loop"}
            for source in (step.loop.until, step.loop.while_):
                if source is not None:
                    expression(source, inner, step.id)
        if kind == "graph":
            _acyclic(step)
        for _, body in step.children():
            for child in body:
                walk(child, inner, node=kind == "graph")

    for _, body in root.children():
        for child in body:
            walk(child, base, node=root.graph is not None)
    if root.graph is not None:
        _acyclic(root)
    templates(spec.output, base, "output")


def _acyclic(step: Step) -> None:
    """A graph's `needs` must name its own nodes and never come back round."""
    nodes = {node.id: node for node in step.graph or []}
    for node in nodes.values():
        for need in node.needs:
            if need not in nodes:
                raise ConfigurationError(
                    f"step {node.id!r} needs {need!r}, which is not a node of the "
                    f"same graph. Nodes: {', '.join(sorted(nodes))}")
    done: set[str] = set()
    while len(done) < len(nodes):
        ready = [n for n, node in nodes.items()
                 if n not in done and all(need in done for need in node.needs)]
        if not ready:
            stuck = ", ".join(sorted(set(nodes) - done))
            raise ConfigurationError(
                f"these nodes wait on each other and can never start: {stuck}. "
                "A graph runs forwards — use a `loop` to go round again")
        done.update(ready)
