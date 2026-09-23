"""The agent, and the loop that drives it: think → act → observe → repeat.

    agent = Agent("support", "Answer billing questions.", tools=[lookup_order])
    result = await agent.run("Where is order 4182?")

Everything else in this library exists to serve this loop: the context
assembler decides what it sees, the rails decide what it may do, and memory
decides what it remembers afterwards.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from .context import ContextAssembler, ContextCompactor
from .errors import (
    BudgetExceeded,
    ConfigurationError,
    GuardrailTripped,
    HarnessError,
    MaxStepsExceeded,
    OutputContractError,
    PermissionDenied,
    ProviderError,
    StopRequested,
    ToolNotFound,
)
from .guardrails import AgentGuardrails
from .guardrails.checks import CompletionContext
from .harness import Harness
from .memory.manager import MemoryManager
from .memory.trace import Trace
from .prompts import Prompt
from .providers import resolve_provider
from .providers.base import CompletionRequest, Provider
from .runtime.budget import Budget, BudgetGuard
from .runtime.checkpoints import Checkpoint
from .runtime.hooks import HookEngine
from .runtime.permissions import PolicyGate
from .runtime.session import Session
from .runtime.workspace import Workspace
from .skills import Skill, SkillRegistry
from .tools import Tool, ToolContext, ToolRegistry
from .types import (
    Artifact,
    Message,
    ModelResponse,
    RunResult,
    StreamEvent,
    ToolCall,
    ToolOutcome,
    ToolUseBlock,
    new_id,
)

__all__ = ["Agent"]

MAX_RUNTIME_AGENTS = 100


def _enabled(value: bool | str) -> bool:
    """Accept `True`, `"enable"`, `"on"`, `"yes"` — and their opposites."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"enable", "enabled", "on", "yes", "true", "1"}:
        return True
    if text in {"disable", "disabled", "off", "no", "false", "0", ""}:
        return False
    raise ConfigurationError(
        f"runtime_agents must be enable/disable (or a bool) — got {value!r}"
    )


def _agent_count(value: int) -> int:
    """How many run-time agents may be spun up: 0 to 100."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"max_runtime_agents must be a whole number — got {value!r}"
        ) from None
    if not 0 <= count <= MAX_RUNTIME_AGENTS:
        raise ConfigurationError(
            f"max_runtime_agents must be between 0 and {MAX_RUNTIME_AGENTS} — "
            f"got {count}"
        )
    return count


IDENTITY = Prompt(
    "agent.identity",
    "You are {name}, an AI agent.{purpose}\n"
    "Work to the point. Say what you did and what you found; do not narrate what "
    "you are about to do. If you cannot complete something, say so plainly and "
    "explain what is missing.",
)

CONTRACT = (
    "Your final message must be a single JSON object matching this schema, with "
    "no prose and no code fence around it:\n{schema}"
)


class Agent:
    """One accountable agent: a model, a prompt, its tools, and its sub-agents."""

    def __init__(
        self,
        name: str = "agent",
        instructions: str | Prompt = "",
        *,
        description: str = "",
        model: str | None = None,
        provider: Provider | str | None = None,
        tier: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 8192,
        thinking: bool | None = None,
        tools: Iterable[Tool | Callable[..., Any]] = (),
        skills: SkillRegistry | Iterable[Skill | str] | str | None = None,
        subagents: Sequence[Any] = (),
        runtime_agents: bool | str = False,
        max_runtime_agents: int = 5,
        runtime_agent_tools: Iterable[str] | None = None,
        memory: MemoryManager | bool = True,
        trace: Trace | str | dict[str, Any] | None = None,
        harness: Harness | None = None,
        hooks: HookEngine | None = None,
        policy: PolicyGate | None = None,
        guardrails: AgentGuardrails | Iterable[Any] | None = None,
        budget: Budget | None = None,
        max_steps: int = 20,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | bool = False,
        allow_shell: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        max_context_tokens: int | None = None,
        compact_at: int | float | None = None,
        compact_keep_last: int = 8,
        compact_target: float = 0.6,
        compactor: ContextCompactor | None = None,
        parallel_tools: bool = True,
        contract_retries: int = 2,
        stop: Iterable[str] = (),
        persist_session: bool = True,
    ) -> None:
        self.name = name
        self.description = description or f"{name} agent"
        self.instructions = (instructions.render() if isinstance(instructions, Prompt)
                             else instructions)
        self.harness = harness or Harness()
        self.model, routed_effort = self.harness.router.pick(
            model=model, tier=tier, task=f"{name} {description}"
        )
        self._provider = provider
        self.effort = effort or routed_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_steps = max_steps
        self.output_type = output_type
        self.tool_choice = tool_choice
        self.parallel_tools = parallel_tools
        self.contract_retries = contract_retries
        self.stop = list(stop)
        self.persist_session = persist_session
        self.budget = budget
        self.hooks = hooks.merge(self.harness.hooks) if hooks else self.harness.hooks
        self.policy = policy or self.harness.policy
        if guardrails is None or isinstance(guardrails, AgentGuardrails):
            self.guardrails: AgentGuardrails | None = guardrails
        else:
            self.guardrails = AgentGuardrails(*guardrails)

        # --- skills -----------------------------------------------------
        if isinstance(skills, SkillRegistry):
            self.skills: SkillRegistry | None = skills
        elif isinstance(skills, str):
            self.skills = SkillRegistry.from_dir(skills)
        elif skills:
            self.skills = SkillRegistry(
                s if isinstance(s, Skill) else Skill.from_dir(s) for s in skills
            )
        else:
            self.skills = None

        # --- memory -----------------------------------------------------
        self.trace = Trace.of(trace, agent=name) if trace is not None else None
        if isinstance(memory, MemoryManager):
            self.memory: MemoryManager | None = memory
            if self.trace is not None:
                self.memory = memory.for_trace(self.trace)
        elif memory:
            self.memory = MemoryManager(self.harness.memory_store,
                                        summarize=self._summarize,
                                        trace=self.trace)
        else:
            self.memory = None
        if self.memory is not None:
            self.trace = self.memory.trace

        # --- workspace --------------------------------------------------
        if isinstance(workspace, Workspace):
            self.workspace: Workspace | None = workspace
        elif workspace:
            self.workspace = self.harness.workspaces.acquire(name)
            self.workspace.allow_shell = allow_shell
        else:
            self.workspace = None

        # --- tools ------------------------------------------------------
        self.tools = ToolRegistry(tools)
        if self.skills is not None:
            self.tools.extend(self.skills.tools())
        if self.memory is not None:
            self.tools.extend(self.memory.tools())
        if self.workspace is not None:
            self.tools.extend(self.workspace.tools())

        # --- sub-agents -------------------------------------------------
        self._subagents: dict[str, Agent] = {}
        for entry in subagents:
            self.add_subagent(entry)

        # --- run-time agents ----------------------------------------------
        self.runtime_agents = _enabled(runtime_agents)
        self.max_runtime_agents = _agent_count(max_runtime_agents)
        self.runtime_agent_tools = (list(runtime_agent_tools)
                                    if runtime_agent_tools is not None else None)
        if self.runtime_agents and self.max_runtime_agents == 0:
            raise ConfigurationError(
                f"{name}: runtime_agents is enabled but max_runtime_agents is 0 — "
                "give it a budget between 1 and 100, or disable it"
            )
        self._spawned = 0          # this run
        self.total_spawned = 0     # the lifetime of this agent
        self._factory: Any = None
        if self.runtime_agents:
            self.tools.add(self._spawn_tool())
            if "delegate" not in self.tools:
                self.tools.add(self._delegate_tool())

        self.assembler = ContextAssembler(
            identity=IDENTITY.render(
                name=name, purpose=f" {description}" if description else ""
            ),
            instructions=self.instructions,
            skills=self.skills,
            memory=self.memory,
        )
        self.compact_at = self._compaction_threshold(compact_at, max_context_tokens)
        self.compactor = compactor or ContextCompactor(
            max_tokens=self.compact_at,
            keep_last=compact_keep_last,
            target_ratio=compact_target,
            summarize=self._summarize,
        )

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def provider(self) -> Provider:
        if not isinstance(self._provider, Provider):
            self._provider = resolve_provider(self._provider or self.harness.provider,
                                              self.model)
        return self._provider

    @property
    def content_guardrails(self) -> Any:
        """The text rules for this agent: its own if it has them, else the harness's."""
        if self.guardrails is not None and self.guardrails.content is not None:
            return self.guardrails.content
        return self.harness.guardrails

    def _default_window(self) -> int:
        from .providers.base import model_info

        info = model_info(self.model)
        # Leave a third of the window for the answer and the next tool result.
        return int((info.context_window if info else 200_000) * 0.66)

    def _compaction_threshold(self, compact_at: int | float | None,
                              max_context_tokens: int | None) -> int:
        """How many tokens of conversation before the compactor runs.

        `compact_at=10_000` is an absolute token count; `compact_at=0.5` is a
        fraction of the model's context window. With neither, two thirds of the
        window, leaving room for the answer and the next tool result.
        """
        if compact_at is None:
            return int(max_context_tokens or self._default_window())
        if isinstance(compact_at, float) and 0 < compact_at <= 1:
            return max(1, int(self._default_window() / 0.66 * compact_at))
        threshold = int(compact_at)
        if threshold < 1:
            raise ConfigurationError(
                f"{self.name}: compact_at must be a positive token count, or a "
                f"fraction of the context window between 0 and 1 — got {compact_at!r}"
            )
        return threshold

    def add_tool(self, item: Tool | Callable[..., Any]) -> Tool:
        return self.tools.add(item)

    def add_skill(self, skill: Skill | str) -> Skill:
        if self.skills is None:
            self.skills = SkillRegistry()
            self.tools.extend(self.skills.tools())
            self.assembler.skills = self.skills
        return self.skills.add(skill)

    def add_subagent(self, entry: Any) -> Agent:
        """Attach a sub-agent, given an Agent or a SubAgentSpec."""
        from .subagents import SubAgentSpec, build_agent

        child = entry if isinstance(entry, Agent) else build_agent(
            entry if isinstance(entry, SubAgentSpec) else SubAgentSpec(**entry), parent=self
        )
        self._subagents[child.name] = child
        # Rebuilt, not just added: the tool's schema carries the roster, and a
        # stale enum would hide every sub-agent attached after the first.
        self.tools.add(self._delegate_tool())
        return child

    @property
    def subagents(self) -> dict[str, Agent]:
        return dict(self._subagents)

    async def _summarize(self, prompt: str) -> str:
        """A single cheap model call used for compaction and memory distillation."""
        model, _ = self.harness.router.pick(tier="fast")
        try:
            response = await self.provider.complete(CompletionRequest(
                model=model if self.provider.name != "fake" else self.model,
                messages=[Message.user(prompt)], max_tokens=1500,
            ))
        except (ProviderError, HarnessError):
            return ""
        return response.text

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    async def run(self, task: str | Message, **kwargs: Any) -> RunResult:
        """Run to completion and return the result."""
        result: RunResult | None = None
        async for event in self._drive(task, token_stream=False, **kwargs):
            if event.type == "run_end":
                result = event.data.get("result")
        if result is None:  # pragma: no cover - the loop always emits run_end
            raise HarnessError("the agent loop produced no result")
        return result

    def run_sync(self, task: str | Message, **kwargs: Any) -> RunResult:
        """Blocking wrapper for scripts and notebooks."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(task, **kwargs))
        raise RuntimeError(
            "run_sync() cannot be called from inside a running event loop — await "
            "agent.run() instead"
        )

    def stream(self, task: str | Message, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Token-by-token events, ending with a `run_end` event carrying the result."""
        return self._drive(task, token_stream=True, **kwargs)

    async def _drive(
        self,
        task: str | Message,
        *,
        token_stream: bool = False,
        session: Session | str | None = None,
        messages: list[Message] | None = None,
        run_id: str | None = None,
        guard: BudgetGuard | None = None,
        max_steps: int | None = None,
        memory: MemoryManager | None = None,
        subagent_memory: Any = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        harness = self.harness
        run_id = run_id or new_id("run")
        guard = guard or (BudgetGuard(self.budget, parent=harness.guard)
                          if self.budget else harness.guard)
        memory = memory if memory is not None else self.memory
        steps_allowed = max_steps or self.max_steps
        model = model or self.model
        session_obj = await self._session(session)

        task_message = task if isinstance(task, Message) else Message.user(str(task))
        task_text = task_message.text

        result = RunResult(agent=self.name, run_id=run_id, session_id=session_obj.id)

        with harness.tracer.span(f"agent:{self.name}", kind="run", task=task_text[:120],
                                 model=model) as span:
            result.trace_id = span.trace_id
            history = list(messages) if messages is not None else list(session_obj.messages)

            try:
                task_text = self.content_guardrails.check(task_text, where="input",
                                                          label="task")
            except GuardrailTripped as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.stop_reason = "error"
                yield StreamEvent(type="run_end", agent=self.name,
                                  data={"result": result})
                return

            task_message = Message.user(task_text)
            history.append(task_message)
            if memory is not None:
                memory.session.add_message(task_message)

            if messages is None or guard is harness.guard:
                self._spawned = 0        # a fresh run gets a fresh agent budget
            harness.control.enter(self.name, run_id)
            harness.audit.record(self.name, "run_start", target=task_text[:120],
                                 run_id=run_id, model=model)
            await self.hooks.emit("run_start", agent=self.name, run_id=run_id,
                                     task=task_text)
            await harness.journal.assignment(self.name, task_text[:500], run_id=run_id,
                                             trace_id=result.trace_id)
            yield StreamEvent(type="run_start", agent=self.name,
                              data={"task": task_text, "run_id": run_id})

            contract = self._contract_text()
            retries_left = self.contract_retries
            guard_retries = self.guardrails.max_retries if self.guardrails else 0
            final_text = ""

            try:
                for step in range(1, steps_allowed + 1):
                    harness.control.check(f"{self.name} step {step}")
                    guard.step()
                    result.steps = step
                    yield StreamEvent(type="step_start", agent=self.name, step=step)
                    await self.hooks.emit("step_start", agent=self.name,
                                             run_id=run_id, step=step)

                    history = await self.compactor.compact(
                        history, pinned=memory.session.facts if memory else ()
                    )
                    system = await self.assembler.build(
                        query=task_text, tool_names=self.tools.names,
                        output_contract=contract,
                    )
                    request = CompletionRequest(
                        model=model,
                        messages=history,
                        system=system,
                        tools=self.tools.schemas(),
                        tool_choice=self.tool_choice,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        thinking=self.thinking,
                        effort=self.effort,
                        stop=self.stop,
                        response_schema=(self.output_type.model_json_schema()
                                         if self.output_type else None),
                    )

                    hook = await self.hooks.emit("pre_model", agent=self.name,
                                                    run_id=run_id, step=step,
                                                    request=request)
                    if hook.blocked:
                        raise StopRequested(hook.reason)
                    if hook.replaced:
                        request = hook.replacement

                    response = None
                    async for event in self._model_events(request, step, token_stream):
                        if event.type == "step_end" and "response" in event.data:
                            response = event.data["response"]
                        else:
                            yield event
                    if response is None:  # a provider that yielded no response
                        raise ProviderError("no response from the model",
                                            provider=self.provider.name)

                    guard.record(response.usage, agent=self.name, task=task_text[:60])
                    result.usage += response.usage
                    span.set(cost_usd=result.usage.cost_usd)

                    post = await self.hooks.emit("post_model", agent=self.name,
                                                    run_id=run_id, step=step,
                                                    response=response)
                    if post.replaced:
                        response = post.replacement

                    history.append(response.message)
                    if memory is not None:
                        memory.session.add_message(response.message)

                    if harness.checkpoints.should_save(step):
                        await harness.checkpoints.save(Checkpoint(
                            run_id=run_id, step=step, agent=self.name,
                            session_id=session_obj.id, messages=history,
                            usage=result.usage,
                        ))

                    calls = response.tool_uses
                    if calls:
                        outcomes = await self._execute_tools(
                            calls, run_id=run_id, step=step, guard=guard,
                            memory=memory, subagent_memory=subagent_memory,
                            result=result,
                        )
                        for call, outcome in zip(calls, outcomes, strict=False):
                            result.tool_calls.append(ToolCall(
                                id=call.id, name=call.name, args=call.input,
                                agent=self.name, step=step,
                            ))
                            yield StreamEvent(type="tool_result", agent=self.name,
                                              step=step, text=outcome.content[:400],
                                              data={"tool": outcome.name,
                                                    "error": outcome.is_error})
                        tool_message = Message.tool_results(
                            [o.as_block() for o in outcomes]
                        )
                        history.append(tool_message)
                        if memory is not None:
                            memory.session.add_message(tool_message)
                        yield StreamEvent(type="step_end", agent=self.name, step=step)
                        continue

                    # No tool calls — this is the answer.
                    final_text = response.text
                    if self.output_type is not None:
                        parsed, problem = self._parse_output(final_text)
                        if problem and retries_left > 0:
                            retries_left -= 1
                            history.append(Message.user(
                                f"That did not satisfy the output contract ({problem}). "
                                "Reply with only the JSON object."
                            ))
                            yield StreamEvent(type="step_end", agent=self.name, step=step)
                            continue
                        if problem:
                            raise OutputContractError(problem)
                        result.data = parsed

                    violations = self._check_completion(final_text, result)
                    # Always reassigned, so a successful retry clears what the
                    # previous attempt failed on.
                    result.violations = [v.line() for v in violations]
                    if violations:
                        rails = self.guardrails
                        if rails is None:  # pragma: no cover - defensive
                            raise GuardrailTripped("; ".join(result.violations))
                        await harness.journal.write(
                            "guardrail", "; ".join(result.violations),
                            agent=self.name, run_id=run_id,
                        )
                        harness.audit.record(
                            self.name, "guardrail", target="completion",
                            decision=rails.on_violation, run_id=run_id,
                            unmet=[v.check for v in violations],
                        )
                        if rails.on_violation == "retry" and guard_retries > 0:
                            guard_retries -= 1
                            history.append(Message.user(rails.feedback(violations)))
                            yield StreamEvent(type="step_end", agent=self.name,
                                              step=step)
                            continue
                        if rails.on_violation != "warn":
                            raise GuardrailTripped(
                                "the answer did not meet this agent's guardrails: "
                                + "; ".join(result.violations),
                                rule=violations[0].check, where="completion",
                            )

                    result.stop_reason = response.stop_reason
                    yield StreamEvent(type="step_end", agent=self.name, step=step)
                    break
                else:
                    raise MaxStepsExceeded(
                        f"{self.name} did not finish within {steps_allowed} steps"
                    )

                final_text = self.content_guardrails.check(final_text, where="output",
                                                           label=self.name)
                result.output = final_text

            except (BudgetExceeded, PermissionDenied, GuardrailTripped, ProviderError,
                    MaxStepsExceeded, OutputContractError, StopRequested,
                    ConfigurationError) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.stop_reason = "stopped" if isinstance(exc, StopRequested) else "error"
                result.output = result.output or final_text
                await self.hooks.emit("error", agent=self.name, run_id=run_id,
                                         error=exc)
                await harness.journal.write("error", result.error, agent=self.name,
                                            run_id=run_id)
                yield StreamEvent(type="error", agent=self.name, text=result.error)

            harness.control.leave(run_id)
            harness.audit.record(
                self.name, "run_end", target=task_text[:120], run_id=run_id,
                decision="error" if result.error else "ok",
                steps=result.steps, cost_usd=result.cost_usd,
                stop_reason=result.stop_reason,
            )
            result.messages = history
            session_obj.messages = history
            session_obj.usage += result.usage
            if memory is not None:
                result.artifacts.extend(memory.session.artifacts)
                session_obj.artifacts = list(memory.session.artifacts)
            if result.artifacts:
                harness.deliverables.extend(result.artifacts, run_id=run_id)
            if self.persist_session:
                await harness.sessions.save(session_obj)

            await harness.journal.handback(
                self.name, (result.output or result.error or "")[:500], run_id=run_id,
                cost_usd=result.cost_usd, steps=result.steps,
            )
            await self.hooks.emit("run_end", agent=self.name, run_id=run_id,
                                     result=result)
            yield StreamEvent(type="run_end", agent=self.name, step=result.steps,
                              text=result.output, data={"result": result})

    # ------------------------------------------------------------------
    # model + tools
    # ------------------------------------------------------------------
    async def _model_events(self, request: CompletionRequest, step: int,
                            token_stream: bool) -> AsyncIterator[StreamEvent]:
        """One model call. Streams tokens when asked, and ends with the response.

        Throughput is paced before the call and health recorded after it, so a
        provider that goes slow or starts failing shows up in `harness.report()`
        rather than only in the wall clock.
        """
        harness = self.harness
        estimated = sum(len(m.text) for m in request.messages) // 4
        await harness.rate.acquire(estimated)

        started = time.perf_counter()
        with harness.tracer.span(f"model:{request.model}", kind="model",
                                 step=step) as span:
            response: ModelResponse | None = None
            try:
                if not token_stream:
                    response = await self.provider.complete(request)
                else:
                    async for event in self.provider.stream(request):
                        if event.type in ("text", "thinking"):
                            yield StreamEvent(type=event.type, text=event.text,
                                              agent=self.name, step=step)
                        elif event.type == "tool_call":
                            yield StreamEvent(type="tool_call", agent=self.name,
                                              step=step,
                                              text=event.data.get("name", ""),
                                              data=event.data)
                        elif event.type == "step_end" and "response" in event.data:
                            response = ModelResponse(**event.data["response"])
                    if response is None:
                        raise ProviderError(
                            "the stream ended without a final response",
                            provider=self.provider.name,
                        )
            except Exception as exc:
                harness.health.record(
                    request.model, (time.perf_counter() - started) * 1000,
                    kind="model", ok=False, error=f"{type(exc).__name__}: {exc}",
                )
                raise

            harness.health.record(request.model,
                                  (time.perf_counter() - started) * 1000, kind="model")
            harness.rate.record(response.usage)
            span.set(tokens=response.usage.total_tokens,
                     cost_usd=response.usage.cost_usd,
                     stop_reason=response.stop_reason)
            # The caller picks this out of the stream — it is not a public event.
            yield StreamEvent(type="step_end", step=step, data={"response": response})

    async def _execute_tools(
        self,
        calls: list[ToolUseBlock],
        *,
        run_id: str,
        step: int,
        guard: BudgetGuard,
        memory: MemoryManager | None,
        subagent_memory: Any,
        result: RunResult,
    ) -> list[ToolOutcome]:
        """Run every tool the model asked for, in parallel, under the rails."""
        ctx = ToolContext(
            agent=self.name, run_id=run_id, step=step, workspace=self.workspace,
            memory=memory, harness=self.harness,
            state={"result": result, "guard": guard, "subagent_memory": subagent_memory},
        )

        async def one(call: ToolUseBlock) -> ToolOutcome:
            return await self._run_tool(call, ctx=ctx, guard=guard, run_id=run_id,
                                        step=step)

        if len(calls) == 1 or not self.parallel_tools:
            return [await one(call) for call in calls]

        raw = await self.harness.scheduler.map(one, calls)
        outcomes: list[ToolOutcome] = []
        for call, item in zip(calls, raw, strict=False):
            if isinstance(item, BaseException):
                if isinstance(item, (BudgetExceeded, StopRequested)):
                    raise item
                outcomes.append(ToolOutcome(call_id=call.id, name=call.name,
                                            content=f"Error: {item}", is_error=True))
            else:
                outcomes.append(item)
        return outcomes

    async def _run_tool(self, call: ToolUseBlock, *, ctx: ToolContext,
                        guard: BudgetGuard, run_id: str, step: int) -> ToolOutcome:
        harness = self.harness
        with harness.tracer.span(f"tool:{call.name}", kind="tool", step=step) as span:
            try:
                entry = self.tools.get(call.name)
            except ToolNotFound as exc:
                span.status = "error"
                return ToolOutcome(call_id=call.id, name=call.name, content=str(exc),
                                   is_error=True)

            if self.guardrails is not None:
                permitted, reason = self.guardrails.tool_allowed(call.name)
                if not permitted:
                    span.status = "error"
                    harness.audit.record(self.name, "tool_call", target=call.name,
                                         decision="deny", run_id=run_id, reason=reason)
                    return ToolOutcome(call_id=call.id, name=call.name,
                                       content=f"Not permitted: {reason}",
                                       is_error=True)

            args = dict(call.input)
            hook = await self.hooks.emit("pre_tool", agent=self.name, run_id=run_id,
                                            step=step, tool=call.name, args=args)
            if hook.blocked:
                return ToolOutcome(call_id=call.id, name=call.name,
                                   content=f"Blocked: {hook.reason}", is_error=True)
            if hook.replaced and isinstance(hook.replacement, dict):
                args = hook.replacement

            try:
                await self.policy.check(call.name, args,
                                        tool_permission=entry.permission)
            except PermissionDenied as exc:
                span.status = "error"
                harness.audit.record(self.name, "tool_call", target=call.name,
                                     decision="deny", run_id=run_id, reason=str(exc))
                await harness.journal.write("denied", str(exc), agent=self.name,
                                            run_id=run_id, tool=call.name)
                return ToolOutcome(call_id=call.id, name=call.name,
                                   content=f"Not permitted: {exc}", is_error=True)
            harness.audit.record(self.name, "tool_call", target=call.name,
                                 decision="allow", run_id=run_id, args=args)

            key = harness.cache.key("tool", call.name, args) if entry.cacheable else ""
            if key:
                hit = harness.cache.get(key)
                if hit is not None:
                    span.set(cached=True)
                    return ToolOutcome(call_id=call.id, name=call.name, content=hit,
                                       cached=True)

            harness.control.check(f"{self.name} tool {call.name}")
            guard.tool_call()
            started = time.perf_counter()
            outcome = await entry.run(call.id, args, ctx)
            harness.health.record(call.name, (time.perf_counter() - started) * 1000,
                                  kind="tool", ok=not outcome.is_error,
                                  error=outcome.content if outcome.is_error else "")

            if not outcome.is_error:
                try:
                    outcome.content = self.content_guardrails.check(
                        outcome.content, where="input", label=call.name
                    )
                except GuardrailTripped as exc:
                    outcome = ToolOutcome(call_id=call.id, name=call.name,
                                          content=f"Blocked: {exc}", is_error=True)

            post = await self.hooks.emit("post_tool", agent=self.name, run_id=run_id,
                                            step=step, tool=call.name, args=args,
                                            outcome=outcome)
            if post.replaced:
                outcome.content = str(post.replacement)

            if key and not outcome.is_error:
                harness.cache.set(key, outcome.content)

            span.set(error=outcome.is_error, duration_ms=round(outcome.duration_ms, 1))
            await harness.journal.write(
                "tool", f"{call.name} → {outcome.content[:200]}", agent=self.name,
                run_id=run_id, tool=call.name, error=outcome.is_error,
            )
            return outcome

    # ------------------------------------------------------------------
    # delegation
    # ------------------------------------------------------------------
    def _delegate_tool(self) -> Tool:
        agent = self

        async def delegate(agent_name: str, task: str, context: str = "",
                           ctx: ToolContext | None = None) -> str:
            return await agent._delegate(agent_name, task, context, ctx)

        delegate.__name__ = "delegate"
        return Tool(
            delegate,
            name="delegate",
            description=(
                "Hand one self-contained task to a sub-agent and get its result back. "
                "Sub-agents start with no memory of this conversation, so put "
                "everything they need in `task` and `context`. Delegate several at "
                "once when the tasks do not depend on each other."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": sorted(agent._subagents),
                        "description": agent._subagent_catalogue(),
                    },
                    "task": {"type": "string",
                             "description": "The complete task, stated on its own."},
                    "context": {"type": "string",
                                "description": "Facts the sub-agent needs and cannot "
                                               "look up itself."},
                },
                "required": ["agent_name", "task"],
            },
            tags=["builtin", "delegation"],
        )

    @property
    def factory(self) -> Any:
        """The sub-agent factory this agent writes new specialists with."""
        if self._factory is None:
            from .subagents import SubAgentFactory

            builder = Agent(
                f"{self.name}.factory",
                "You write specifications for sub-agents.",
                model=self.harness.router.pick(tier="balanced")[0],
                provider=self._provider if isinstance(self._provider, Provider) else None,
                harness=self.harness, memory=False, max_steps=2,
                persist_session=False, runtime_agents=False,
            )
            self._factory = SubAgentFactory(builder)
        return self._factory

    @property
    def runtime_agents_remaining(self) -> int:
        """How many more specialists this agent may spin up in this run."""
        if not self.runtime_agents:
            return 0
        return max(0, self.max_runtime_agents - self._spawned)

    def _spawn_tool(self) -> Tool:
        agent = self

        async def spawn_agent(task: str, purpose: str = "",
                              tools: list[str] | None = None,
                              ctx: ToolContext | None = None) -> str:
            return await agent._spawn(task, purpose=purpose, tools=tools, ctx=ctx)

        spawn_agent.__name__ = "spawn_agent"
        return Tool(
            spawn_agent,
            name="spawn_agent",
            description=(
                "Build a specialist for one task that no existing sub-agent covers, "
                f"and run it. You may spin up {agent.max_runtime_agents} of these "
                "in this run; spend them on work that genuinely needs its own "
                "worker, and call this several times in one turn when the tasks "
                "are independent — they run at the same time. The specialist "
                "starts with no memory of this conversation, so put everything it "
                "needs in `task`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string",
                             "description": "The complete task, stated on its own."},
                    "purpose": {"type": "string",
                                "description": "One line: what this specialist is "
                                               "accountable for."},
                    "tools": {"type": "array", "items": {"type": "string"},
                              "description": "The smallest set of tool names it "
                                             "needs. Omit to let the factory "
                                             "choose."},
                },
                "required": ["task"],
            },
            tags=["builtin", "delegation", "runtime"],
        )

    async def _spawn(self, task: str, *, purpose: str = "",
                     tools: list[str] | None = None,
                     ctx: ToolContext | None = None) -> str:
        """Write a new specialist for `task`, run it, and hand back its result."""
        if not self.runtime_agents:
            return "run-time agents are disabled for this agent."
        if self._spawned >= self.max_runtime_agents:
            return (f"No run-time agents left: {self.max_runtime_agents} of "
                    f"{self.max_runtime_agents} already spun up in this run. "
                    "Finish the work with the tools and sub-agents you have.")
        if not self.harness.control.may_start():
            return (f"Not started — the run was stopped: "
                    f"{self.harness.control.state.reason or 'no reason given'}")

        self._spawned += 1
        self.total_spawned += 1

        allowed = self.runtime_agent_tools
        if allowed is None:
            allowed = [t.name for t in self.tools
                       if "delegation" not in t.tags and "memory" not in t.tags]
        if tools:
            wanted = set(tools)
            allowed = [name for name in allowed if name in wanted] or allowed

        with self.harness.tracer.span("factory:spec", kind="subagent") as span:
            spec = await self.factory.create(task, tools=allowed)
            if purpose:
                spec.description = purpose
            span.set(spec=spec.name, tools=spec.tools)

        self.harness.audit.record(self.name, "spawn_agent", target=spec.name,
                                  decision="allow", task=task[:200],
                                  remaining=self.runtime_agents_remaining)
        await self.harness.journal.write(
            "decision", f"spun up {spec.name} for: {task[:120]}", agent=self.name,
        )
        if self.memory is not None:
            await self.memory.orchestrator.staffing(task[:80], spec.name,
                                                    reused=False)

        from .subagents import build_agent

        child = build_agent(spec, parent=self)
        self._subagents[child.name] = child
        self.tools.add(self._delegate_tool())   # it can be re-used by name now
        return await self._run_child(child, task, "", ctx)

    def _subagent_catalogue(self) -> str:
        return "; ".join(f"{n}: {a.description}" for n, a in self._subagents.items())

    async def _delegate(self, agent_name: str, task: str, context: str = "",
                        ctx: ToolContext | None = None) -> str:
        """Run an existing sub-agent on one task and hand back only its result."""
        child = self._subagents.get(agent_name)
        if child is None:
            known = ", ".join(sorted(self._subagents)) or "none"
            suffix = ""
            if self.runtime_agents and self.runtime_agents_remaining:
                suffix = (" Use `spawn_agent` to build one for this task "
                          f"({self.runtime_agents_remaining} left).")
            return f"No sub-agent named {agent_name!r}. Available: {known}.{suffix}"

        if not self.harness.control.may_start():
            return (f"[{agent_name} not started] the run was stopped: "
                    f"{self.harness.control.state.reason or 'no reason given'}")

        if self.memory is not None:
            await self.memory.orchestrator.staffing(task[:80], child.name, reused=True)
        return await self._run_child(child, task, context, ctx)

    async def _run_child(self, child: Agent, task: str, context: str = "",
                         ctx: ToolContext | None = None) -> str:
        """Run one sub-agent under this agent's budget and hand back its result.

        Shared by `delegate` (reuse) and `spawn_agent` (build one), so both go
        through the same budget, tracing, memory and hand-back path.
        """
        parent_result: RunResult | None = (ctx.state.get("result") if ctx else None)
        guard: BudgetGuard = (ctx.state.get("guard") if ctx else None) or self.harness.guard
        guard.subagent()

        sub_memory = (self.memory.subagent(task, child.name) if self.memory
                      else None)
        brief = f"{task}\n\n## Context you were given\n{context}" if context else task

        await self.hooks.emit("subagent_start", agent=child.name,
                              run_id=ctx.run_id if ctx else "", task=task)

        with self.harness.tracer.span(f"subagent:{child.name}", kind="subagent") as span:
            child_result = await child.run(
                brief,
                messages=[],  # every sub-agent starts clean
                guard=guard.child(child.budget),
                subagent_memory=sub_memory,
            )
            span.set(cost_usd=child_result.cost_usd, steps=child_result.steps,
                     error=child_result.error)

        if parent_result is not None:
            parent_result.children.append(child_result)
            parent_result.artifacts.extend(child_result.artifacts)
        if self.memory is not None:
            await self.memory.orchestrator.spend(child.name, child_result.cost_usd,
                                                 child_result.usage.total_tokens)
        await self.hooks.emit("subagent_end", agent=child.name, result=child_result)

        if child_result.error:
            return f"[{child.name} failed] {child_result.error}"
        handback = child_result.output
        if child_result.artifacts:
            names = ", ".join(a.name for a in child_result.artifacts)
            handback += f"\n\nArtefacts produced: {names}"
        return handback

    def as_tool(self, name: str | None = None, description: str | None = None,
                *, cacheable: bool = False) -> Tool:
        """Expose this agent as a tool another agent can call."""
        agent = self

        async def call_agent(task: str, ctx: ToolContext | None = None) -> str:
            result = await agent.run(task, messages=[])
            return f"[failed] {result.error}" if result.error else result.output

        call_agent.__name__ = name or agent.name
        return Tool(
            call_agent,
            name=name or agent.name,
            description=description or agent.description,
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string",
                                        "description": "The task, stated in full."}},
                "required": ["task"],
            },
            cacheable=cacheable,
            tags=["agent"],
        )

    # ------------------------------------------------------------------
    # sessions, artefacts, output contract
    # ------------------------------------------------------------------
    async def _session(self, session: Session | str | None) -> Session:
        if isinstance(session, Session):
            return session
        if isinstance(session, str):
            return await self.harness.sessions.load(session)
        if self.memory is not None:
            return Session(id=self.memory.session.id, agent=self.name)
        return Session(agent=self.name)

    def produce(self, name: str, content: str, **kw: Any) -> Artifact:
        """Record an artefact this agent produced, and store it."""
        artifact = Artifact(name=name, content=content, produced_by=self.name, **kw)
        stored = self.harness.deliverables.put(artifact)
        if self.memory is not None:
            self.memory.session.add_artifact(stored)
        return stored

    def _check_completion(self, output: str, result: RunResult) -> list[Any]:
        """Has this agent done what it was required to do before answering?"""
        if self.guardrails is None or not self.guardrails.checks:
            return []
        called = [c.name for c in result.tool_calls]
        called += [c.name for child in result.children for c in child.tool_calls]
        return self.guardrails.check(CompletionContext(
            agent=self.name, output=output, steps=result.steps,
            cost_usd=result.cost_usd, tools_called=called,
            artifacts=[a.name for a in result.artifacts],
            data=result.data, result=result,
        ))

    def _contract_text(self) -> str:
        if self.output_type is None:
            return ""
        schema = json.dumps(self.output_type.model_json_schema(), indent=2)
        return CONTRACT.format(schema=schema)

    def _parse_output(self, text: str) -> tuple[Any, str]:
        """Validate the final answer against `output_type`. Returns (value, problem)."""
        if self.output_type is None:
            return None, ""
        blob = _extract_json(text)
        if blob is None:
            return None, "no JSON object found in the reply"
        try:
            return self.output_type.model_validate(blob), ""
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return None, problems

    async def close_session(self, summary: str | None = None) -> str:
        """Session close → user memory update. Returns the new `user.md`."""
        if self.memory is None:
            return ""
        return await self.memory.close_session(summary)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<Agent {self.name} model={self.model} tools={len(self.tools)}>"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Any | None:
    """Pull a JSON object out of a reply, fenced or not."""
    if not text:
        return None
    candidates = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
