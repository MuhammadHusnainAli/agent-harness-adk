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

from .attachments import prepare_attachments
from .context import ContextAssembler, ContextCompactor, close_open_tool_calls
from .errors import (
    AuthenticationError,
    BudgetExceeded,
    ConfigurationError,
    GuardrailTripped,
    HarnessError,
    InvalidRequestError,
    MaxStepsExceeded,
    OutputContractError,
    PermissionDenied,
    ProviderError,
    QuotaExceededError,
    SessionConflict,
    StopRequested,
    ToolError,
    ToolNotFound,
)
from .guardrails import AgentGuardrails
from .guardrails.checks import CompletionContext
from .harness import Harness
from .llm_providers import resolve_provider
from .llm_providers.base import CompletionRequest, Provider
from .llm_providers.parameters import GENERATION_PARAMETERS, validate_parameters
from .memory.manager import MemoryManager
from .memory.trace import Trace
from .modes import LEAD, Mode, Notebook, resolve_mode
from .prompts import Prompt
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
    TextBlock,
    ToolCall,
    ToolOutcome,
    ToolUseBlock,
    attach,
    new_id,
)
from .versioning import AgentVersion

__all__ = ["Agent"]

MAX_RUNTIME_AGENTS = 100

# How much of what a sandboxed conversation wrote is kept with its session.
_SESSION_FILE_CHARS = 2_000_000


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


class _Leaving:
    """Takes a run off the stop controller's list on the way out, whatever the
    way out is. A run whose caller cancels it must not be left "running"."""

    def __init__(self, control: Any, run_id: str) -> None:
        self.control, self.run_id = control, run_id

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        self.control.leave(self.run_id)


_YES = {"true", "yes", "on", "1", "enable", "enabled", "local", "isolated"}
_NO = {"false", "no", "off", "0", "disable", "disabled", "none", ""}


def _is_sandbox(value: Any) -> bool:
    """A bare `Sandbox`, told apart without importing the package to ask."""
    return (not isinstance(value, (bool, type(None)))
            and callable(getattr(value, "_exec", None))
            and callable(getattr(value, "stop", None)))


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
        mode: str | Mode | dict[str, Any] | None = None,
        depth: str | None = None,
        tier: str | None = None,
        effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 8192,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
        cache: bool | None = None,
        user: str | None = None,
        model_options: dict[str, Any] | None = None,
        tools: Iterable[Tool | Callable[..., Any]] = (),
        skills: SkillRegistry | Iterable[Skill | str] | str | None = None,
        subagents: Sequence[Any] = (),
        runtime_agents: bool | str | None = None,
        max_runtime_agents: int | None = None,
        runtime_agent_tools: Iterable[str] | None = None,
        memory: MemoryManager | bool = True,
        trace: Trace | str | dict[str, Any] | None = None,
        harness: Harness | None = None,
        hooks: HookEngine | None = None,
        policy: PolicyGate | None = None,
        guardrails: AgentGuardrails | Iterable[Any] | None = None,
        identity: Any = None,
        budget: Budget | None = None,
        max_steps: int | None = None,
        output_type: type[BaseModel] | None = None,
        workspace: Workspace | bool | str | dict[str, Any] | None = None,
        allow_shell: bool | None = None,
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
        version: str = "v1",
        versions: dict[str, AgentVersion | dict[str, Any]] | None = None,
    ) -> None:
        # --- versions ----------------------------------------------------
        # A version says what is *different*; everything it leaves unset falls
        # through to how the agent was constructed. Applied here, before any of
        # it is assembled, so a version is a real configuration rather than a
        # patch on a half-built object.
        self._versions: dict[str, AgentVersion] = {
            key: AgentVersion.of(spec) for key, spec in (versions or {}).items()
        }
        self._version = version
        self._siblings: dict[str, Agent] = {}
        self._base_kwargs: dict[str, Any] = {
            "instructions": instructions, "description": description,
            "model": model, "provider": provider, "mode": mode, "depth": depth,
            "tier": tier, "effort": effort,
            "temperature": temperature, "max_tokens": max_tokens,
            "thinking": thinking, "thinking_budget": thinking_budget,
            "top_p": top_p, "top_k": top_k, "min_p": min_p,
            "frequency_penalty": frequency_penalty, "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty, "seed": seed, "cache": cache,
            "user": user, "model_options": model_options,
            "tools": list(tools), "skills": skills,
            "subagents": list(subagents), "runtime_agents": runtime_agents,
            "max_runtime_agents": max_runtime_agents,
            "runtime_agent_tools": runtime_agent_tools, "memory": memory,
            "trace": trace, "harness": harness, "hooks": hooks, "policy": policy,
            "guardrails": guardrails, "identity": identity, "budget": budget,
            "max_steps": max_steps,
            "output_type": output_type, "workspace": workspace,
            "allow_shell": allow_shell, "tool_choice": tool_choice,
            "max_context_tokens": max_context_tokens, "compact_at": compact_at,
            "compact_keep_last": compact_keep_last,
            "compact_target": compact_target, "compactor": compactor,
            "parallel_tools": parallel_tools,
            "contract_retries": contract_retries, "stop": list(stop),
            "persist_session": persist_session,
        }
        if self._versions:
            if version not in self._versions:
                raise ConfigurationError(
                    f"{name}: version {version!r} is not defined; known: "
                    f"{', '.join(sorted(self._versions)) or 'none'}")
            active = self._versions[version]
            instructions = (active.instructions if active.instructions is not None
                            else instructions)
            description = active.description or description
            model = active.model or model
            if active.mode is not None:
                # A version that changes the mode takes its depth with it.
                mode, depth = active.mode, active.depth
            elif active.depth is not None:
                depth = active.depth
            tier = active.tier or tier
            effort = active.effort or effort
            temperature = (active.temperature if active.temperature is not None
                           else temperature)
            max_tokens = active.max_tokens or max_tokens
            max_steps = active.max_steps or max_steps
            thinking = active.thinking if active.thinking is not None else thinking
            # The rest of the sampling vocabulary: a version sets what it names.
            sampling = {"thinking_budget": thinking_budget, "top_p": top_p,
                        "top_k": top_k, "min_p": min_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                        "repetition_penalty": repetition_penalty, "seed": seed}
            for key in sampling:
                if getattr(active, key, None) is not None:
                    sampling[key] = getattr(active, key)
            thinking_budget, top_p, top_k, min_p = (
                sampling["thinking_budget"], sampling["top_p"], sampling["top_k"],
                sampling["min_p"])
            frequency_penalty, presence_penalty = (sampling["frequency_penalty"],
                                                   sampling["presence_penalty"])
            repetition_penalty, seed = sampling["repetition_penalty"], sampling["seed"]
            compact_at = (active.compact_at if active.compact_at is not None
                          else compact_at)
            contract_retries = (active.contract_retries
                                if active.contract_retries is not None
                                else contract_retries)
            runtime_agents = (active.runtime_agents
                              if active.runtime_agents is not None
                              else runtime_agents)
            max_runtime_agents = (active.max_runtime_agents
                                  if active.max_runtime_agents is not None
                                  else max_runtime_agents)
            budget = active.budget if active.budget is not None else budget
            guardrails = (active.guardrails if active.guardrails is not None
                          else guardrails)
            if active.tools is not None:
                wanted = set(active.tools)
                tools = [t for t in tools
                         if getattr(t, "name", getattr(t, "__name__", "")) in wanted]
            if active.subagents is not None:
                subagents = list(active.subagents)
        # --- mode ---------------------------------------------------------
        # A mode fills in only what was left unset, so anything said explicitly
        # — here, or by the active version — still wins.
        try:
            self.mode: Mode | None = resolve_mode(mode, depth)
        except ConfigurationError as exc:
            raise ConfigurationError(f"{name}: {exc}") from None
        profile = self.mode
        if max_steps is None:
            max_steps = profile.max_steps if profile else 20
        if workspace is None:
            workspace = profile.workspace if profile else False
        if runtime_agents is None:
            runtime_agents = bool(profile and profile.helpers)
        if max_runtime_agents is None:
            max_runtime_agents = (profile.helpers if profile and profile.helpers
                                  else 5)
        if tier is None and profile is not None:
            tier = profile.tier
        self.name = name
        self.description = description or f"{name} agent"
        self.instructions = (instructions.render() if isinstance(instructions, Prompt)
                             else instructions)
        self.harness = harness or Harness()
        # Checked now, so a bad value fails when the agent is built — not mid-run.
        options = model_options or {}
        validate_parameters(**{
            "effort": effort, "temperature": temperature, "max_tokens": max_tokens,
            "thinking": thinking, "thinking_budget": thinking_budget, "top_p": top_p,
            "top_k": top_k, "min_p": min_p, "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty, "seed": seed, "stop": list(stop),
            **{k: v for k, v in options.items() if k in GENERATION_PARAMETERS},
        })
        self.model, routed_effort = self.harness.router.pick(
            model=model, tier=tier, task=f"{name} {description}"
        )
        self._provider = provider
        self.effort = effort or routed_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        # Everything else a provider will take. `model_options` is the escape
        # hatch for anything these do not name.
        self.model_options: dict[str, Any] = {
            "thinking_budget": thinking_budget, "top_p": top_p, "top_k": top_k,
            "min_p": min_p, "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "seed": seed, "cache": cache, "user": user,
            **options,
        }
        self.model_options = {k: v for k, v in self.model_options.items()
                              if v is not None}
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
        elif isinstance(guardrails, dict):
            self.guardrails = AgentGuardrails(**guardrails)
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
        # True is a folder (or whatever the harness's broker hands out). A name,
        # a URL or a mapping is a sandbox: "docker", "e2b://template",
        # {"sandbox": "daytona", "image": ...}.
        if isinstance(workspace, str):
            word = workspace.strip().lower()
            if word in _YES:
                workspace = True
            elif word in _NO:
                workspace = False
        if isinstance(workspace, Workspace):
            self.workspace: Workspace | None = workspace
        elif isinstance(workspace, (str, dict)) or _is_sandbox(workspace):
            from .sandboxes import sandbox as _sandbox

            self.workspace = _sandbox(workspace)
            # Every version of this agent works in the one sandbox.
            self._base_kwargs["workspace"] = self.workspace
        elif workspace:
            self.workspace = self.harness.workspaces.acquire(name)
            if allow_shell is None and not self.workspace.isolated:
                # On this machine a shell is asked for, never assumed.
                self.workspace.allow_shell = False
        else:
            self.workspace = None
        if self.workspace is not None:
            # The harness closes it, so a sandbox is not left running.
            self.harness.workspaces.adopt(self.workspace)
            if allow_shell is not None:
                self.workspace.allow_shell = allow_shell

        # --- tools ------------------------------------------------------
        self.tools = ToolRegistry(tools)
        if self.skills is not None:
            self.tools.extend(self.skills.tools())
        if self.memory is not None:
            self.tools.extend(self.memory.tools())
        if self.workspace is not None:
            self.tools.extend(self.workspace.tools())
        if profile is not None:
            self.tools.extend(profile.tools(self.workspace))
        #: The conversation a chat or cowork agent carries from one run to the next.
        self._thread: Session | None = None
        #: Set by `resume()`: follow that conversation, whatever the mode.
        self._following = False
        #: The stored record of this agent's own runs, when it is not following
        #: a conversation: each run starts clean, and is added to this.
        self._log: Session | None = None

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
        self._fallback_providers: dict[str, Any] = {}
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
            mode=profile.prompt(self.tools.names) if profile else "",
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

        # --- governance ---------------------------------------------------
        #: Who this agent is, who answers for it, and what it may do. A
        #: `governance.AgentIdentity` (or a dict of its fields); governance
        #: fills in a default, and reports the gap, when it is left out.
        self.identity = identity
        if self.memory is not None:
            self.memory.on_write = self._memory_write
        if self.harness.governance is not None:
            self.harness.governance.register_agent(self)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # versions
    # ------------------------------------------------------------------
    @property
    def version(self) -> str:
        """Which version of this agent is running."""
        return self._version

    @property
    def available_versions(self) -> list[str]:
        return sorted(self._versions)

    def use(self, version: str) -> Agent:
        """This agent, configured as `version`. Built once, then reused.

        The harness, provider and memory are shared, so switching versions is
        cheap and two versions are directly comparable.
        """
        if version == self._version:
            return self
        if version not in self._versions:
            raise ConfigurationError(
                f"{self.name}: version {version!r} is not defined; known: "
                f"{', '.join(self.available_versions) or 'none'}")
        if version not in self._siblings:
            sibling = Agent(self.name, version=version, versions=self._versions,
                            **self._base_kwargs)
            sibling._siblings = self._siblings
            self._siblings[version] = sibling
        return self._siblings[version]

    def version_spec(self, version: str | None = None) -> AgentVersion | None:
        return self._versions.get(version or self._version)

    async def resume(self, session: str, *, sandbox_id: str | None = None) -> Session:
        """Pick a conversation back up: every run from here continues it.

            agent = Agent("colleague", mode="cowork", workspace="e2b", harness=harness)
            await agent.resume("ses_4f1c")          # the chat, and the sandbox it used
            await agent.run("Now add the tests.")

        The session remembers which sandbox it was working in, so the id alone
        is enough; `sandbox_id=` says a different one. If that sandbox is gone,
        the run starts a new one, puts back the files the conversation is known
        to have written, and tells the model what it could not put back.

        It needs a session store that outlives the process — `Harness.local()`,
        or your own `SessionStore` — for there to be anything to resume.
        """
        loaded = self._may_have(await self.harness.sessions.load(session))
        if sandbox_id:
            box = getattr(self.workspace, "sandbox", None)
            if box is None:
                raise ConfigurationError(
                    f"{self.name}: sandbox_id={sandbox_id!r} was given, but this "
                    "agent's workspace is not a sandbox")
            loaded.metadata["sandbox"] = {"sandbox": box.name, "id": sandbox_id}
        self._thread = loaded
        self._following = True
        return loaded

    async def _rejoin(self, session: Session, sandbox_id: str | None) -> str:
        """Put this agent back in the sandbox its conversation was using.

        Returns what the model has to be told when that was not possible: it is
        in a new sandbox, and which of its files came back.
        """
        box = getattr(self.workspace, "sandbox", None)
        if box is None:
            if sandbox_id:
                raise ConfigurationError(
                    f"{self.name}: sandbox_id={sandbox_id!r} was given, but this "
                    "agent's workspace is not a sandbox")
            return ""
        ref = dict(session.metadata.get("sandbox") or {})
        if not ref and not sandbox_id:
            return ""                  # a conversation with no sandbox behind it yet
        # What was asked for now beats what the sandbox was built with, which
        # beats what the conversation remembers.
        elsewhere = bool(ref) and ref.get("sandbox") != box.name
        wanted = sandbox_id or box.attach_id or ("" if elsewhere else ref.get("id", ""))
        if box.running:
            if wanted and box.id != wanted:
                raise ConfigurationError(
                    f"{self.name} is already working in {box.name} sandbox "
                    f"{box.id}, but this conversation's files are in {wanted} — "
                    "build an agent for each conversation that has its own sandbox")
            return ""
        if wanted and box.attach_id != wanted:
            box.attach(wanted)
        try:
            await self.workspace.start()
        except ToolError as exc:
            raise ConfigurationError(
                f"{self.name}'s sandbox could not be picked back up: {exc}") from None
        gone = box.replaced or (ref.get("id", "") if elsewhere else "")
        if not gone:
            return ""

        # A new sandbox: put back what the conversation is known to have written.
        restored: list[str] = []
        lost: list[str] = []
        for name, content in (session.metadata.get("sandbox_files") or {}).items():
            try:
                if content is None:
                    lost.append(name)
                elif not await self.workspace.aexists(name):
                    await self.workspace.awrite(name, content)
                    restored.append(name)
            except ToolError:
                lost.append(name)
        self.harness.audit.record(self.name, "sandbox_replaced", target=box.id,
                                  decision="ok", was=gone, restored=len(restored),
                                  lost=len(lost), reason=box.missing_reason)
        await self.harness.journal.write(
            "decision", f"sandbox {gone} was gone; continuing in {box.id} with "
            f"{len(restored)} files restored", agent=self.name)
        lines = [f"[Workspace notice: the sandbox this conversation was working in "
                 f"({gone}) is no longer available, so you are in a new one."]
        if restored:
            lines.append("Put back from the conversation's record: "
                         + ", ".join(restored) + ".")
        if lost:
            lines.append("Could not be put back (not text, or too large): "
                         + ", ".join(lost) + ".")
        lines.append("Anything else that was installed, built or written there is "
                     "gone — check before you rely on it.]")
        return " ".join(lines)

    async def _remember_sandbox(self, session: Session, result: RunResult,
                                produced: list[Artifact]) -> None:
        """Note on the session which sandbox it is in and what it wrote there,
        so the conversation can be picked up — in it, or without it."""
        box = getattr(self.workspace, "sandbox", None)
        if box is None or not box.running:
            return
        result.sandbox_id = box.id
        session.metadata["sandbox"] = self.workspace.ref()
        files: dict[str, str | None] = dict(session.metadata.get("sandbox_files") or {})
        for artifact in produced:
            if artifact.produced_by == self.name and artifact.name != getattr(
                    self.mode, "report", ""):
                files[artifact.name] = artifact.content or None
        if files:
            try:
                present = await self.workspace.asnapshot()
                files = {name: text for name, text in files.items() if name in present}
            except ToolError:
                pass
            # The record is for getting a conversation going again, not a backup:
            # past the cap, the largest files are remembered by name only.
            budget = _SESSION_FILE_CHARS
            for name in sorted(files, key=lambda n: len(files[n] or "")):
                size = len(files[name] or "")
                if size > budget:
                    files[name] = None
                else:
                    budget -= size
        session.metadata["sandbox_files"] = files

    # ------------------------------------------------------------------
    # mode
    # ------------------------------------------------------------------
    @property
    def depth(self) -> str:
        """How hard this agent's mode works: fast, balanced or deep. "" without one."""
        return self.mode.depth if self.mode else ""

    @property
    def conversational(self) -> bool:
        """Does one run pick up where the last one left off?"""
        return bool(self.mode and self.mode.conversational)

    def new_session(self) -> None:
        """Start a fresh conversation: the next run does not see the last one.

        Only the thread is dropped. What the agent remembered about the user
        stays, and so does everything in its workspace.
        """
        self._thread = None
        self._following = False
        if self.memory is not None:
            from .memory.manager import SessionMemory

            old, fresh = self.memory.session, SessionMemory()
            self.memory.session = fresh
            # A trace that was following the old conversation follows the new
            # one; a session id you set yourself is left alone.
            if self.memory.trace.session_id == old.id:
                self.memory.trace.session_id = fresh.id

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
        from .llm_providers.base import model_info

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
        request = CompletionRequest(
            model=model if self.provider.name != "fake" else self.model,
            messages=[Message.user(prompt)], max_tokens=1500,
        )
        # The conversation leaves here too, so it passes the same egress check
        # as the loop's own calls. A refused summary is simply not made.
        egress = await self.hooks.emit("model_egress", agent=self.name, request=request,
                                       provider=self.provider, model=request.model,
                                       purpose="summarize")
        if egress.blocked:
            self.harness.audit.record(self.name, "model_egress", target=request.model,
                                      decision="deny", reason=egress.reason,
                                      purpose="summarize")
            return ""
        try:
            response = await self.provider.complete(
                egress.replacement if egress.replaced else request)
        except (ProviderError, HarnessError):
            return ""
        return response.text

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    async def run(self, task: str | Message, *, version: str | None = None,
                  **kwargs: Any) -> RunResult:
        """Run to completion and return the result.

        `version="v1"` runs that version of this agent instead of the active one.
        """
        if version is not None and version != self._version:
            return await self.use(version).run(task, **kwargs)
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

    def stream(self, task: str | Message, *, version: str | None = None,
               **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Token-by-token events, ending with a `run_end` event carrying the result."""
        if version is not None and version != self._version:
            return self.use(version).stream(task, **kwargs)
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
        notebook: Notebook | None = None,
        sandbox_id: str | None = None,
        attachments: Iterable[Any] = (),
    ) -> AsyncIterator[StreamEvent]:
        harness = self.harness
        run_id = run_id or new_id("run")
        guard = guard or (BudgetGuard(self.budget, parent=harness.guard)
                          if self.budget else harness.guard)
        memory = memory if memory is not None else self.memory
        steps_allowed = max_steps or self.max_steps
        model = model or self.model
        # Handed a history, this is a one-off (a sub-agent's task, a replay) and
        # has no business reading or moving the agent's own conversation.
        threaded = (self.conversational or self._following) and messages is None
        # Handed its messages and no session, a run is somebody else's turn —
        # a sub-agent's task, a tool call, a replay. It gets a session of its
        # own, never this agent's.
        one_off = messages is not None and session is None
        session_obj = (self._may_have(Session(agent=self.name)) if one_off
                       else await self._session(session, threaded=threaded))

        task_message = task if isinstance(task, Message) else Message.user(str(task))
        task_text = task_message.text
        # Whatever came with the task that is not text: files, images, audio.
        attached = [*task_message.media, *attachments]

        result = RunResult(agent=self.name, run_id=run_id, session_id=session_obj.id)
        profile = self.mode
        if profile is not None:
            result.mode, result.depth = profile.name, profile.depth
            if notebook is None and profile.keeps_notebook:
                notebook = Notebook(ledger=profile.sources,
                                    tracking=profile.sources
                                    and profile.verify_sources)
        if notebook is not None:
            # The same lists, not copies: what a tool writes shows on the result.
            result.todos, result.sources = notebook.todos, notebook.sources
        files_before: dict[str, tuple[int, int]] | None = None

        # However the run ends — finished, failed, or abandoned by a caller who
        # stopped listening — it stops counting as running.
        with _Leaving(harness.control, run_id), harness.tracer.span(
                f"agent:{self.name}", kind="run", task=task_text[:120],
                model=model) as span:
            result.trace_id = span.trace_id
            # A conversation being continued starts from what was said. A run
            # that is not continuing one starts clean, and is added to the
            # session's record afterwards rather than replacing it.
            continuing = threaded or session is not None
            if messages is not None:
                history = list(messages)
            elif continuing:
                history = close_open_tool_calls(list(session_obj.messages))
            else:
                history = []
            earlier = [] if continuing else list(session_obj.messages)
            began_at = len(history)
            rewritten = False

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
            if notebook is not None:
                notebook.saw(task_text)

            if messages is None or guard is harness.guard:
                self._spawned = 0        # a fresh run gets a fresh agent budget
            harness.control.enter(self.name, run_id)
            # The hook first: governance learns whose run this is, so the task
            # text below reaches the audit trail only as that person's tokens.
            started = await self.hooks.emit("run_start", agent=self.name, run_id=run_id,
                                             task=task_text, trace=self.trace,
                                             model=model)
            harness.audit.record(self.name, "run_start", target=task_text[:120],
                                 run_id=run_id, model=model)
            await harness.journal.assignment(self.name, task_text[:500], run_id=run_id,
                                             trace_id=result.trace_id)
            yield StreamEvent(type="run_start", agent=self.name,
                              data={"task": task_text, "run_id": run_id})

            contract = self._contract_text()
            # The last thing the model actually said. `final_text` is only set on
            # a turn with no tool calls, but a run cut short mid-way still has
            # work worth handing back.
            last_text = ""
            retries_left = self.contract_retries
            guard_retries = self.guardrails.max_retries if self.guardrails else 0
            mode_retries = profile.retries if profile else 0
            reported = notebook.revision if notebook is not None else 0
            final_text = ""

            try:
                if started.blocked:
                    raise PermissionDenied(f"{self.name} may not run: {started.reason}",
                                           reason=started.reason)
                self._check_mode()
                if attached:
                    # Sent as they are where the model can take them; read or
                    # transcribed into text where it cannot. Text made that way
                    # passes the same input checks as anything else coming in.
                    ready = await prepare_attachments(
                        attached, self.provider, model, speech=harness.speech)
                    for block in ready:
                        if isinstance(block, TextBlock):
                            block.text = self.content_guardrails.check(
                                block.text, where="input", label="attachment")
                    task_message.content[:0] = ready
                    harness.audit.record(
                        self.name, "attachments", target=str(len(ready)),
                        decision="ok", run_id=run_id,
                        kinds=[getattr(b, "type", "text") for b in ready],
                        names=[attach(a).label for a in attached])
                if messages is None:
                    notice = await self._rejoin(session_obj, sandbox_id)
                    if notice:
                        # Said in the conversation itself, so it is still there
                        # the next time this history is read.
                        task_message.content[-1].text = f"{notice}\n\n{task_text}"
                if profile and profile.collect_files and self.workspace is not None:
                    # Also what starts a sandbox — so one that cannot start
                    # ends the run here, before a model call is paid for.
                    try:
                        files_before = await self.workspace.asnapshot()
                    except ToolError as exc:
                        raise ConfigurationError(
                            f"{self.name}'s workspace is not usable: {exc}"
                        ) from None
                for step in range(1, steps_allowed + 1):
                    harness.control.check(f"{self.name} step {step}")
                    guard.step()
                    result.steps = step
                    yield StreamEvent(type="step_start", agent=self.name, step=step)
                    await self.hooks.emit("step_start", agent=self.name,
                                             run_id=run_id, step=step)

                    compacted = await self.compactor.compact(
                        history, pinned=[*(memory.session.facts if memory else ()),
                                         *(notebook.pins() if notebook else ())]
                    )
                    # The compactor hands back the same list when it did nothing.
                    rewritten = rewritten or compacted is not history
                    history = compacted
                    system = await self.assembler.build(
                        query=task_text, tool_names=self.tools.names,
                        output_contract=contract,
                    )
                    tool_choice = self.tool_choice
                    # A mode's last step is for handing over, not for one more
                    # tool call whose result nobody would ever read.
                    last_step = (profile is not None and step == steps_allowed
                                 and steps_allowed > 1)
                    if last_step:
                        system = f"{system}\n\n{profile.wrap_up}"
                        tool_choice = "none" if len(self.tools) else tool_choice
                    request = CompletionRequest(
                        model=model,
                        messages=history,
                        system=system,
                        tools=self.tools.schemas(),
                        tool_choice=tool_choice,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        thinking=self.thinking,
                        effort=self.effort,
                        stop=self.stop,
                        response_schema=(self.output_type.model_json_schema()
                                         if self.output_type else None),
                        **self.model_options,
                    )

                    hook = await self.hooks.emit("pre_model", agent=self.name,
                                                    run_id=run_id, step=step,
                                                    request=request)
                    if hook.blocked:
                        raise StopRequested(hook.reason)
                    if hook.replaced:
                        request = hook.replacement

                    response = None
                    async for event in self._model_events(request, step, token_stream,
                                                          run_id=run_id):
                        if event.type == "step_end" and "response" in event.data:
                            response = event.data["response"]
                        else:
                            yield event
                    if response is None:  # a provider that yielded no response
                        raise ProviderError("no response from the model",
                                            provider=self.provider.name)

                    # Capture what it said *before* charging for it: recording the
                    # usage is what trips a budget, and work already done should
                    # still be handed back.
                    last_text = response.text or last_text
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
                            result=result, notebook=notebook,
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
                        if notebook is not None and notebook.revision != reported:
                            reported = notebook.revision
                            await harness.journal.write(
                                "progress", notebook.todo_summary(), agent=self.name,
                                run_id=run_id, sources=len(notebook.sources))
                            yield StreamEvent(type="progress", agent=self.name,
                                              step=step, text=notebook.todo_summary(),
                                              data=notebook.progress())
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

                    # What the mode itself requires: nothing left open on the
                    # todo list, every citation leading to a recorded source. It
                    # is sent back to be finished; if it still is not, the answer
                    # is delivered with `violations` saying what is missing —
                    # work that fell short is still worth more than an error.
                    unmet = profile.unmet(final_text, notebook) if profile else []
                    if unmet and mode_retries > 0 and step < steps_allowed:
                        mode_retries -= 1
                        await harness.journal.write(
                            "mode", "; ".join(unmet), agent=self.name, run_id=run_id)
                        history.append(Message.user(profile.feedback(unmet)))
                        yield StreamEvent(type="step_end", agent=self.name, step=step)
                        continue
                    if unmet:
                        harness.audit.record(self.name, "mode", target=profile.name,
                                             decision="warn", run_id=run_id,
                                             unmet=unmet)

                    violations = await self._check_completion(final_text, result)
                    # Always reassigned, so a successful retry clears what the
                    # previous attempt failed on.
                    result.violations = [*unmet, *(v.line() for v in violations)]
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

                if profile is not None and self.output_type is None:
                    final_text = profile.close(final_text, notebook)
                final_text = self.content_guardrails.check(final_text, where="output",
                                                           label=self.name)
                result.output = final_text

            except BudgetExceeded as exc:
                # A budget is a ceiling, not a failure. By default the run ends
                # cleanly: what the agent produced is kept, and a note says why
                # it stopped, so the caller (often a parent agent) gets an answer
                # rather than an exception.
                if guard.budget.on_exceed == "stop":
                    partial = final_text or last_text or result.output
                    if profile is not None and self.output_type is None:
                        partial = profile.close(partial, notebook)
                    result.output = self.content_guardrails.check(
                        partial, where="output", label=self.name)
                    result.output = (result.output + self._budget_note(exc)).strip()
                    result.stop_reason = "budget"
                    result.budget_exceeded = exc.kind or "budget"
                    await harness.journal.write(
                        "budget", f"{self.name} stopped: {exc}", agent=self.name,
                        run_id=run_id)
                    harness.audit.record(self.name, "budget", target=exc.kind,
                                         decision="stop", run_id=run_id,
                                         limit=exc.limit, spent=exc.spent)
                    yield StreamEvent(type="step_end", agent=self.name,
                                      step=result.steps,
                                      data={"budget_exceeded": exc.kind})
                else:
                    result.error = f"{type(exc).__name__}: {exc}"
                    result.stop_reason = "error"
                    result.output = result.output or final_text or last_text
                    await harness.hooks.emit("error", agent=self.name,
                                             run_id=run_id, error=exc)
                    yield StreamEvent(type="error", agent=self.name,
                                      text=result.error)

            except (PermissionDenied, GuardrailTripped, ProviderError,
                    MaxStepsExceeded, OutputContractError, StopRequested,
                    ConfigurationError) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                result.stop_reason = "stopped" if isinstance(exc, StopRequested) else "error"
                result.output = result.output or final_text or last_text
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
            session_obj.messages = [*earlier, *history]
            session_obj.usage += result.usage
            if not session_obj.title:
                session_obj.title = " ".join(task_text.split())[:80]
            produced: list[Artifact] = []
            if notebook is not None:
                result.todos, result.sources = list(notebook.todos), list(notebook.sources)
            if profile is not None:
                # A report is only worth keeping if the run finished one; files
                # were written whether it finished or not.
                have = {(a.name, a.path) for a in result.artifacts}
                try:
                    produced = await profile.artifacts(
                        result.output if result.error is None
                        and result.budget_exceeded is None else "",
                        agent=self.name, workspace=self.workspace,
                        before=files_before)
                except Exception as exc:
                    # A sandbox that died must not take the answer down with it.
                    await harness.journal.write(
                        "error", f"could not collect the files this run wrote: {exc}",
                        agent=self.name, run_id=run_id)
                result.artifacts.extend(a for a in produced
                                        if (a.name, a.path) not in have)
            if messages is None:
                await self._remember_sandbox(session_obj, result, produced)
            if memory is not None:
                result.artifacts.extend(memory.session.artifacts)
                session_obj.artifacts = list(memory.session.artifacts)
            if result.artifacts:
                harness.deliverables.extend(result.artifacts, run_id=run_id)
            if self.persist_session and not one_off:
                session_obj = await self._save_session(
                    session_obj, result, added=history[began_at:],
                    rewritten=rewritten, run_id=run_id)
            if threaded:
                self._thread = session_obj
            elif session is None and not one_off:
                self._log = session_obj

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
                            token_stream: bool, *,
                            run_id: str = "") -> AsyncIterator[StreamEvent]:
        """One model call. Streams tokens when asked, and ends with the response.

        Throughput is paced before the call and health recorded after it, so a
        provider that goes slow or starts failing shows up in `harness.report()`
        rather than only in the wall clock.
        """
        harness = self.harness
        estimated = sum(len(m.text) for m in request.messages) // 4
        await harness.rate.acquire(estimated)

        chain = harness.router.chain(request.model)
        refused: list[str] = []
        for attempt, model in enumerate(chain):
            request.model = model
            provider = (self.provider if model == self.model
                        else self._provider_for(model))
            last = attempt == len(chain) - 1
            # Fired per backend actually tried, so a handler sees where the data
            # is really going. A refusal moves on down the chain.
            egress = await self.hooks.emit("model_egress", agent=self.name,
                                           run_id=run_id, step=step, request=request,
                                           provider=provider, model=model)
            if egress.blocked:
                refused.append(f"{model}: {egress.reason}")
                harness.audit.record(self.name, "model_egress", target=model,
                                     decision="deny", run_id=run_id,
                                     reason=egress.reason)
                if last:
                    raise PermissionDenied(
                        "no model in the chain may receive this request — "
                        + "; ".join(refused), tool=model, reason=egress.reason)
                continue
            sent = egress.replacement if egress.replaced else request
            try:
                async for event in self._one_call(sent, provider, step,
                                                  token_stream):
                    yield event
                return
            except (ProviderError, OSError) as exc:
                if last or not _is_reachability_problem(exc):
                    raise
                await harness.journal.write(
                    "fallback", f"{model} unreachable ({exc}); trying "
                    f"{chain[attempt + 1]}", agent=self.name, run_id="")
                harness.audit.record(self.name, "model_fallback", target=model,
                                     decision="retry", to=chain[attempt + 1],
                                     reason=str(exc)[:200])

    def _provider_for(self, model: str) -> Any:
        """The provider for a fallback model, cached per agent."""
        if isinstance(self._provider, Provider):
            return self._provider          # an explicit backend serves everything
        cached = self._fallback_providers.get(model)
        if cached is None:
            cached = resolve_provider(None, model)
            self._fallback_providers[model] = cached
        return cached

    async def _one_call(self, request: CompletionRequest, provider: Any, step: int,
                        token_stream: bool) -> AsyncIterator[StreamEvent]:
        harness = self.harness
        started = time.perf_counter()
        with harness.tracer.span(f"model:{request.model}", kind="model",
                                 step=step) as span:
            response: ModelResponse | None = None
            try:
                if not token_stream:
                    response = await provider.complete(request)
                else:
                    async for event in provider.stream(request):
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
                            provider=provider.name,
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
        notebook: Notebook | None = None,
    ) -> list[ToolOutcome]:
        """Run every tool the model asked for, in parallel, under the rails."""
        ctx = ToolContext(
            agent=self.name, run_id=run_id, step=step, workspace=self.workspace,
            memory=memory, harness=self.harness,
            state={"result": result, "guard": guard, "subagent_memory": subagent_memory,
                   "notebook": notebook},
        )

        async def one(call: ToolUseBlock) -> ToolOutcome:
            return await self._run_tool(call, ctx=ctx, guard=guard, run_id=run_id,
                                        step=step)

        if len(calls) == 1 or not self.parallel_tools:
            outcomes = [await one(call) for call in calls]
        else:
            raw = await self.harness.scheduler.map(one, calls)
            outcomes = []
            for call, item in zip(calls, raw, strict=False):
                if isinstance(item, BaseException):
                    if isinstance(item, (BudgetExceeded, StopRequested)):
                        raise item
                    outcomes.append(ToolOutcome(call_id=call.id, name=call.name,
                                                content=f"Error: {item}",
                                                is_error=True))
                else:
                    outcomes.append(item)

        # What the run has now read — the evidence a recorded source is checked
        # against. Here rather than around the call, so a cached result counts.
        if notebook is not None and notebook.tracking:
            for call, outcome in zip(calls, outcomes, strict=False):
                entry = self.tools.get(call.name) if call.name in self.tools else None
                if outcome.is_error or entry is None or "mode" in entry.tags:
                    continue
                notebook.saw(call.name,
                             json.dumps(call.input, default=str, ensure_ascii=False),
                             outcome.content)
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
                                            step=step, tool=call.name, args=args,
                                            tags=sorted(entry.tags),
                                            permission=entry.permission)
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

    async def call_tool(self, name: str, args: dict[str, Any] | None = None, *,
                        run_id: str = "", step: int = 0,
                        guard: BudgetGuard | None = None,
                        result: RunResult | None = None) -> ToolOutcome:
        """Run one of this agent's tools under all of its rails, outside a run.

        For when something other than this agent's own loop decides which tool
        to call — a realtime voice model, a workflow of your own. The call still
        passes the guardrails, the hooks, the permission gate, the audit trail
        and the budget, exactly as it would have inside a run. Never raises: a
        refusal or a failure comes back as an outcome with `is_error` set.
        """
        guard = guard or self.harness.guard
        call = ToolUseBlock(name=name, input=dict(args or {}))
        ctx = ToolContext(
            agent=self.name, run_id=run_id, step=step, workspace=self.workspace,
            memory=self.memory, harness=self.harness,
            state={"result": result or RunResult(agent=self.name, run_id=run_id),
                   "guard": guard, "subagent_memory": None, "notebook": None})
        try:
            return await self._run_tool(call, ctx=ctx, guard=guard, run_id=run_id,
                                        step=step)
        except (BudgetExceeded, StopRequested) as exc:
            return ToolOutcome(call_id=call.id, name=name, is_error=True,
                               content=f"Not run: {exc}")

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
                       if not t.tags & {"delegation", "memory", LEAD}]
        if tools:
            wanted = set(tools)
            allowed = [name for name in allowed if name in wanted] or allowed

        with self.harness.tracer.span("factory:spec", kind="subagent") as span:
            spec = await self.factory.create(task, tools=allowed)
            if purpose:
                spec.description = purpose
            if (self.mode is not None and self.mode.sources
                    and spec.tools is not None and "record_source" not in spec.tools):
                spec.tools = [*spec.tools, "record_source"]
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

        # Asked before any budget is spent on it: a refused hand-off costs nothing.
        start = await self.hooks.emit("subagent_start", agent=child.name,
                                      run_id=ctx.run_id if ctx else "", task=task,
                                      parent=self.name)
        if start.blocked:
            self.harness.audit.record(self.name, "delegate", target=child.name,
                                      decision="deny",
                                      run_id=ctx.run_id if ctx else "",
                                      reason=start.reason)
            return f"[{child.name} not permitted] {start.reason}"
        guard.subagent()

        sub_memory = (self.memory.subagent(task, child.name) if self.memory
                      else None)
        brief = f"{task}\n\n## Context you were given\n{context}" if context else task
        # A helper keeps its own todo list but writes in the run's one source
        # ledger, so the number it cites is the number the lead can cite.
        notebook: Notebook | None = ctx.state.get("notebook") if ctx else None
        shared = notebook.child() if notebook is not None and notebook.ledger else None
        if shared is not None and "record_source" in child.tools:
            brief += ("\n\n## Sources\nRecord every source you rely on with "
                      "`record_source` and cite it by the number that returns — "
                      "[n] after the claim it supports.")

        with self.harness.tracer.span(f"subagent:{child.name}", kind="subagent") as span:
            child_result = await child.run(
                brief,
                messages=[],  # every sub-agent starts clean
                guard=guard.child(child.budget),
                subagent_memory=sub_memory,
                notebook=shared,
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
    async def _session(self, session: Session | str | None, *,
                       threaded: bool = False) -> Session:
        if isinstance(session, Session):
            return self._may_have(session)
        if isinstance(session, str):
            return self._may_have(await self.harness.sessions.load(session))
        if threaded and self._thread is not None:
            return self._thread
        if self.memory is not None:
            if self._log is not None and self._log.id == self.memory.session.id:
                return self._log
            return self._may_have(Session(id=self.memory.session.id, agent=self.name))
        return self._may_have(Session(agent=self.name))

    def _may_have(self, session: Session) -> Session:
        """This agent's trace says who it is acting for. A session that belongs
        to someone else is refused in the same words as one that does not exist,
        so an id cannot be probed; one that belongs to nobody yet becomes theirs.
        """
        user = self.trace.user_id if self.trace is not None else None
        tenant = self.trace.tenant_id if self.trace is not None else None
        if not session.owned_by(user, tenant):
            self.harness.audit.record(self.name, "session_denied", target=session.id,
                                      decision="deny", user=user, tenant=tenant)
            raise ConfigurationError(f"no session {session.id!r}")
        session.user_id = session.user_id or user
        session.tenant_id = session.tenant_id or tenant
        return session

    async def _save_session(self, session: Session, result: RunResult, *,
                            added: list[Message], rewritten: bool,
                            run_id: str) -> Session:
        """Save the conversation without losing anyone's turn — or the answer.

        If another request saved this session while this run was working, the
        store refuses the save. This run's own messages are then added to what
        is there now. When they cannot be — the history was compacted along the
        way, so "this run's messages" no longer lines up — the run is kept as a
        fork, and the result says which session it ended up in.
        """
        harness = self.harness
        try:
            try:
                return await harness.sessions.save(session)
            except SessionConflict:
                pass
            for _ in range(3):
                if rewritten:
                    break
                latest = self._may_have(await harness.sessions.load(session.id))
                if latest is session:
                    break
                latest.messages = [*close_open_tool_calls(list(latest.messages)), *added]
                latest.usage += result.usage
                latest.metadata.update(session.metadata)
                latest.title = latest.title or session.title
                names = {a.name for a in session.artifacts}
                latest.artifacts = [*(a for a in latest.artifacts if a.name not in names),
                                    *session.artifacts]
                try:
                    saved = await harness.sessions.save(latest)
                except SessionConflict:
                    continue
                harness.audit.record(self.name, "session_merged", target=session.id,
                                     decision="ok", run_id=run_id, added=len(added))
                await harness.journal.write(
                    "decision", f"session {session.id} had moved on; this run's "
                    f"{len(added)} messages were added to it", agent=self.name,
                    run_id=run_id)
                return saved
            forked = session.model_copy(update={
                "id": new_id("ses"), "parent_id": session.id, "version": 0,
                "title": f"{session.title} (fork)" if session.title else ""})
            saved = await harness.sessions.save(forked)
            result.session_id = saved.id
            result.warnings.append(
                f"session {session.id} was changed by another request while this "
                f"run was working; this run is kept as {saved.id}")
            harness.audit.record(self.name, "session_forked", target=session.id,
                                 decision="ok", run_id=run_id, fork=saved.id)
            return saved
        except Exception as exc:
            # The work is done and paid for. A store that is down loses the
            # record of it, not the answer.
            result.warnings.append(f"the session could not be saved: "
                                   f"{type(exc).__name__}: {exc}")
            harness.audit.record(self.name, "session_save", target=session.id,
                                 decision="error", run_id=run_id,
                                 reason=str(exc)[:300])
            await harness.journal.write(
                "error", f"session {session.id} could not be saved: {exc}",
                agent=self.name, run_id=run_id)
            return session

    def _check_mode(self) -> None:
        """Can this mode do its job with what the agent has? Asked at the start
        of a run rather than at construction, because tools — an MCP server's,
        say — are often attached after the agent is built."""
        profile = self.mode
        if profile is None or not profile.sources:
            return
        # Its own notebook, its memory and a factory with nothing to hand down
        # are not ways of finding anything out; a sub-agent might be.
        if self._subagents or any(
                not t.tags & {"mode", "memory", "delegation"} for t in self.tools):
            return
        raise ConfigurationError(
            f"{self.name} is in {profile.name} mode but has nothing to read with — give "
            "it a search, fetch or lookup tool (tools=[...]), a sub-agent, or an "
            "MCP server")

    def produce(self, name: str, content: str, **kw: Any) -> Artifact:
        """Record an artefact this agent produced, and store it."""
        artifact = Artifact(name=name, content=content, produced_by=self.name, **kw)
        stored = self.harness.deliverables.put(artifact)
        if self.memory is not None:
            self.memory.session.add_artifact(stored)
        return stored

    async def _check_completion(self, output: str, result: RunResult) -> list[Any]:
        """Has this agent done what it was required to do before answering?"""
        if self.guardrails is None or not self.guardrails.checks:
            return []
        called = [c.name for c in result.tool_calls]
        called += [c.name for child in result.children for c in child.tool_calls]
        return await self.guardrails.check_async(CompletionContext(
            agent=self.name, output=output, steps=result.steps,
            cost_usd=result.cost_usd, tools_called=called,
            artifacts=[a.name for a in result.artifacts],
            data=result.data, result=result,
        ))

    @staticmethod
    def _budget_note(exc: BudgetExceeded) -> str:
        """The line appended to partial work when a ceiling is reached."""
        spent = (f"{exc.spent:,.0f}" if exc.spent >= 1 else f"{exc.spent}")
        limit = (f"{exc.limit:,.0f}" if exc.limit >= 1 else f"{exc.limit}")
        axis = (exc.kind or "budget").replace("_", " ")
        return (f"\n\n[The budget for this agent is exceeded — {axis} "
                f"{spent} of {limit}. The answer above is what it completed "
                f"before stopping.]")

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

    async def _memory_write(self, text: str, *, scope: str, kind: str) -> str:
        """Every durable memory write passes the `memory_write` hook first.

        A handler may block it (raises `PermissionDenied`) or rewrite the text —
        to strip personal data before it is kept, say.
        """
        hook = await self.hooks.emit("memory_write", agent=self.name, text=text,
                                     scope=scope, kind=kind, trace=self.trace)
        if hook.blocked:
            self.harness.audit.record(self.name, "memory_write", target=scope,
                                      decision="deny", reason=hook.reason)
            raise PermissionDenied(f"memory write refused: {hook.reason}",
                                   tool="remember", reason=hook.reason)
        return str(hook.replacement) if hook.replaced else text

    async def close_session(self, summary: str | None = None) -> str:
        """Session close → user memory update. Returns the new `user.md`."""
        if self.memory is None:
            return ""
        return await self.memory.close_session(summary)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        mode = f" mode={self.mode.name}/{self.mode.depth}" if self.mode else ""
        return f"<Agent {self.name} model={self.model}{mode} tools={len(self.tools)}>"


#: A model we could not reach is worth retrying elsewhere. A 400 is not — the
#: request is wrong, and the next model will reject it just the same.
_REACHABILITY = (408, 429, 500, 502, 503, 504, 529)


def _is_reachability_problem(exc: Exception) -> bool:
    if isinstance(exc, OSError):
        return True
    # The provider layer has already decided: a timeout, an open circuit, a 5xx
    # or a rate limit it gave up waiting on. An exhausted quota is not retryable
    # here, but another vendor's model has its own quota — so it still moves on.
    if getattr(exc, "retryable", False) or isinstance(exc, QuotaExceededError):
        return True
    status = getattr(exc, "status", None)
    if status is None:
        return not isinstance(exc, (AuthenticationError, InvalidRequestError))
    return status in _REACHABILITY


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
