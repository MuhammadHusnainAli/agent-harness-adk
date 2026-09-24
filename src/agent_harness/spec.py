"""Spec compiler: an agent blueprint becomes the payload a provider accepts.

`SubAgentSpec` is a declaration — a name, some instructions, a tool allowlist, a
tier. This turns it into the concrete thing that goes over the wire: a resolved
model, a system prompt, the tool schemas, the output contract and the sampling
settings.

Having it as its own step means you can inspect exactly what a sub-agent will be
sent *before* you spend anything on it:

    compiled = SpecCompiler().compile(spec, tools=registry)
    print(compiled.explain())
    request = compiled.to_request([Message.user("do the thing")])
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .context import ContextAssembler
from .llm_providers.base import CompletionRequest, ToolSchema, model_info
from .runtime.router import ModelRouter
from .tools import ToolRegistry
from .types import Message

__all__ = ["CompiledSpec", "SpecCompiler"]


class CompiledSpec(BaseModel):
    """A blueprint, resolved. Everything needed to make the call and nothing else."""

    model_config = ConfigDict(extra="allow")

    name: str
    model: str
    system: str = ""
    tools: list[ToolSchema] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    effort: str | None = None
    thinking: bool | None = None
    thinking_budget: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    response_schema: dict[str, Any] | None = None
    max_steps: int = 12
    workspace: str = "none"
    allow_shell: bool = False
    permissions: dict[str, list[str]] = Field(default_factory=dict)
    estimated_input_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_request(self, messages: list[Message] | None = None,
                   **overrides: Any) -> CompletionRequest:
        """The provider payload. This is the end of the compilation."""
        request = CompletionRequest(
            model=self.model,
            messages=list(messages or []),
            system=self.system or None,
            tools=list(self.tools),
            tool_choice=self.tool_choice,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            effort=self.effort,  # type: ignore[arg-type]
            thinking=self.thinking,
            thinking_budget=self.thinking_budget,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            response_schema=self.response_schema,
        )
        for key, value in overrides.items():
            setattr(request, key, value)
        return request

    def explain(self) -> str:
        """What this sub-agent is, in the terms that matter: access and cost."""
        tools = ", ".join(t.name for t in self.tools) or "none"
        lines = [
            f"{self.name}",
            f"  model        {self.model}"
            + (f" (effort {self.effort})" if self.effort else ""),
            f"  steps        up to {self.max_steps}, {self.max_tokens} tokens per reply",
            f"  tools        {tools}",
            f"  workspace    {self.workspace}"
            + (" + shell" if self.allow_shell else ""),
        ]
        for kind in ("allow", "ask", "deny"):
            entries = self.permissions.get(kind)
            if entries:
                lines.append(f"  {kind:<12} {', '.join(entries)}")
        if self.response_schema:
            required = ", ".join(self.response_schema.get("required", [])) or "-"
            lines.append(f"  must return  JSON with {required}")
        lines.append(f"  system       {len(self.system)} chars, "
                     f"~{self.estimated_input_tokens} tokens")
        if self.estimated_cost_usd:
            lines.append(f"  first call   ~${self.estimated_cost_usd:.5f}")
        return "\n".join(lines)


class SpecCompiler:
    """Blueprint in, provider payload out. No network, no side effects."""

    def __init__(self, *, router: ModelRouter | None = None,
                 assembler: ContextAssembler | None = None) -> None:
        self.router = router or ModelRouter()
        self.assembler = assembler

    def compile(
        self,
        spec: Any,
        *,
        tools: ToolRegistry | Iterable[Any] | None = None,
        skills: Any = None,
        memory_blocks: Iterable[str] = (),
        extra_instructions: str = "",
    ) -> CompiledSpec:
        """Resolve a `SubAgentSpec` into something you can send."""
        registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or ())
        allowed = (registry.select(spec.tools) if getattr(spec, "tools", None) is not None
                   else registry)

        model, effort = self.router.pick(
            model=getattr(spec, "model", None), tier=getattr(spec, "tier", None),
            task=f"{spec.name} {getattr(spec, 'description', '')}",
        )
        if getattr(spec, "effort", None):
            effort = spec.effort

        system = self._system(spec, allowed, skills, memory_blocks, extra_instructions)
        schema = getattr(spec, "output_schema", None)
        if schema:
            system += "\n\n" + (
                "Your final message must be a single JSON object matching this "
                "schema, with no prose and no code fence around it:\n"
                + json.dumps(schema, indent=2)
            )

        estimated = max(1, len(system) // 4) + sum(
            len(t.description) // 4 + 20 for t in allowed.schemas()
        )
        info = model_info(model)
        cost = round(estimated * info.input_cost / 1_000_000, 8) if info else 0.0

        return CompiledSpec(
            name=spec.name,
            model=model,
            system=system,
            tools=allowed.schemas(),
            max_tokens=getattr(spec, "max_tokens", 8192),
            temperature=getattr(spec, "temperature", None),
            effort=effort,
            **{name: getattr(spec, name, None) for name in (
                "thinking", "thinking_budget", "top_p", "top_k", "min_p",
                "frequency_penalty", "presence_penalty", "repetition_penalty", "seed")},
            response_schema=schema,
            max_steps=getattr(spec, "max_steps", 12),
            workspace=getattr(spec, "workspace", "none"),
            allow_shell=getattr(spec, "allow_shell", False),
            permissions={
                "allow": list(getattr(spec, "allow", []) or []),
                "ask": list(getattr(spec, "ask", []) or []),
                "deny": list(getattr(spec, "deny", []) or []),
            },
            estimated_input_tokens=estimated,
            estimated_cost_usd=cost,
        )

    def payload(self, spec: Any, messages: list[Message] | None = None,
                **kwargs: Any) -> CompletionRequest:
        """Compile and hand back the request in one step."""
        return self.compile(spec, **kwargs).to_request(messages)

    def _system(self, spec: Any, tools: ToolRegistry, skills: Any,
                memory_blocks: Iterable[str], extra: str) -> str:
        from .prompts import sections

        description = getattr(spec, "description", "")
        identity = (f"You are {spec.name}, an AI agent."
                    + (f" {description}" if description else ""))
        tool_note = ""
        if tools.names:
            tool_note = (
                f"Tools you can call: {', '.join(tools.names)}.\n"
                "Call a tool when it gets you a fact you do not have. Do not guess "
                "what a tool would have returned."
            )
        return sections(
            identity,
            getattr(spec, "instructions", ""),
            skills.index() if skills is not None else "",
            tool_note,
            extra,
            *memory_blocks,
        )
