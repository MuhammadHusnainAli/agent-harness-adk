"""Turning a blueprint into a live agent that shares its parent's harness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..guardrails import AgentGuardrails
from ..llm_providers.base import Provider
from ..runtime.permissions import PolicyGate
from ..types import new_id
from .spec import SubAgentSpec

__all__ = ["build_agent"]


def build_agent(spec: SubAgentSpec, parent: Any) -> Any:
    """Turn a spec into a live Agent that shares the parent's harness."""
    from ..agent import Agent

    # Tools: an explicit allowlist, or the parent's own tools minus the
    # orchestration-only ones. Delegation never cascades by default.
    if spec.tools is None:
        inherited = [t for t in parent.tools
                     if "delegation" not in t.tags and "memory" not in t.tags]
    else:
        inherited = list(parent.tools.select(spec.tools))

    policy = parent.policy
    if spec.allow or spec.ask or spec.deny:
        policy = PolicyGate(policy.default, rules=list(policy.rules), allow=spec.allow,
                            ask=spec.ask, deny=spec.deny, approver=policy.approver,
                            audit=policy.audit)

    workspace: Any = False
    if spec.workspace == "isolated":
        workspace = parent.harness.workspaces.acquire(f"{spec.name}-{new_id()}")
        workspace.allow_shell = spec.allow_shell
    elif spec.workspace == "shared":
        workspace = parent.harness.workspaces.shared()
        workspace.allow_shell = spec.allow_shell

    guardrails = AgentGuardrails.from_dict(spec.guardrails)

    output_type = None
    if spec.output_schema:
        output_type = _model_from_schema(spec.name, spec.output_schema)

    skills = parent.skills.select(spec.skills) if (parent.skills and spec.skills) else None

    # A provider handed to the parent explicitly is the configured backend for the
    # whole tree; otherwise the child resolves one from its own model.
    provider = parent._provider if isinstance(parent._provider, Provider) else None

    return Agent(
        name=spec.name,
        provider=provider,
        instructions=spec.instructions,
        description=spec.description,
        model=spec.model,
        tier=spec.tier,
        effort=spec.effort,
        temperature=spec.temperature,
        thinking=spec.thinking,
        thinking_budget=spec.thinking_budget,
        top_p=spec.top_p,
        top_k=spec.top_k,
        min_p=spec.min_p,
        frequency_penalty=spec.frequency_penalty,
        presence_penalty=spec.presence_penalty,
        repetition_penalty=spec.repetition_penalty,
        seed=spec.seed,
        max_tokens=spec.max_tokens,
        max_steps=spec.max_steps,
        tools=inherited,
        skills=skills,
        memory=spec.memory,
        harness=parent.harness,
        policy=policy,
        guardrails=guardrails,
        budget=spec.budget,
        workspace=workspace,
        allow_shell=spec.allow_shell,
        output_type=output_type,
        persist_session=False,
    )


def _model_from_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """A pydantic model from a JSON-Schema object, for the output contract."""
    from pydantic import create_model

    type_map: dict[str, Any] = {
        "string": str, "integer": int, "number": float, "boolean": bool,
        "array": list, "object": dict,
    }
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for field, spec in (schema.get("properties") or {}).items():
        annotation = type_map.get(spec.get("type", "string"), Any)
        default = ... if field in required else spec.get("default", None)
        if field not in required:
            annotation = annotation | None
        fields[field] = (annotation, default)
    if not fields:
        fields = {"result": (str, ...)}
    title = "".join(p.title() for p in name.replace("-", "_").split("_")) or "Output"
    return create_model(f"{title}Output", **fields)
