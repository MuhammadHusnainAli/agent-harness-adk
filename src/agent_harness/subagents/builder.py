"""Turning a blueprint into a live agent that shares its parent's harness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..guardrails import AgentGuardrails
from ..llm_providers.base import Provider
from ..modes import LEAD
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
                     if not t.tags & {"delegation", "memory", LEAD}]
    else:
        inherited = list(parent.tools.select(spec.tools))

    policy = parent.policy
    if spec.allow or spec.ask or spec.deny:
        policy = PolicyGate(policy.default, rules=list(policy.rules), allow=spec.allow,
                            ask=spec.ask, deny=spec.deny, approver=policy.approver,
                            audit=policy.audit)

    # A mode decides the two things a spec has no way to leave unsaid — the step
    # ceiling and the workspace — unless the spec said them itself.
    said = spec.model_fields_set
    moded = spec.mode is not None
    workspace: Any = None if moded and "workspace" not in said else False
    if spec.workspace == "isolated":
        workspace = parent.harness.workspaces.acquire(f"{spec.name}-{new_id()}")
    elif spec.workspace == "shared":
        workspace = parent.harness.workspaces.shared()
    # A sandbox comes with its shell; a folder on this machine has one only if
    # the spec asked.
    allow_shell: bool | None = spec.allow_shell
    if "allow_shell" not in said and (workspace is None
                                      or getattr(workspace, "isolated", False)):
        allow_shell = None

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
        mode=spec.mode,
        depth=spec.depth,
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
        max_steps=None if moded and "max_steps" not in said else spec.max_steps,
        tools=inherited,
        skills=skills,
        memory=spec.memory,
        harness=parent.harness,
        policy=policy,
        guardrails=guardrails,
        identity=spec.identity,
        budget=spec.budget,
        workspace=workspace,
        allow_shell=allow_shell,
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
