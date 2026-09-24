"""Agent versions: several configurations of one agent, switchable by name.

    agent = Agent(
        "support",
        tools=[order_status, issue_refund, lookup],
        version="v2",
        versions={
            "v1": {"instructions": "Answer order questions.",
                   "tools": ["order_status"],
                   "model": "claude-sonnet-5"},
            "v2": {"instructions": "Answer order questions. Cite the order.",
                   "tools": ["order_status", "lookup"],
                   "subagents": [researcher],
                   "guardrails": {"require_tools": ["order_status"]},
                   "model": "claude-opus-5"},
        },
    )

    await agent.run(task)                  # v2, the active version
    await agent.run(task, version="v1")    # the old one, for comparison

Versions are how you change a prompt without losing the ability to answer "what
did it do before?" — evaluate v2 against v1 on the same golden tasks, and roll
back by changing one string.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError

__all__ = ["AgentVersion"]

#: The constructor arguments a version is allowed to change. Anything else —
#: the harness, the provider, the memory backend — is infrastructure, shared by
#: every version so switching one is cheap and comparable.
OVERRIDABLE = {
    "instructions", "description", "model", "tier", "effort", "temperature",
    "max_tokens", "thinking", "thinking_budget", "top_p", "top_k", "min_p",
    "frequency_penalty", "presence_penalty", "repetition_penalty", "seed",
    "max_steps", "tool_choice", "stop",
    "compact_at", "compact_keep_last", "compact_target",
    "runtime_agents", "max_runtime_agents", "contract_retries",
}


class AgentVersion(BaseModel):
    """One version of an agent's configuration. Every field is optional.

    Anything left unset falls through to how the agent was constructed, so a
    version says what is *different*, not everything.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    instructions: str | None = None
    description: str | None = None
    model: str | None = None
    tier: str | None = None
    effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    thinking: bool | None = None
    thinking_budget: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    #: Tool names, selected from the tools the agent was given. `None` keeps
    #: them all; `[]` takes them all away.
    tools: list[str] | None = None
    subagents: list[Any] | None = None
    prompts: dict[str, str] = Field(default_factory=dict)
    guardrails: dict[str, Any] | Any = None
    budget: Any = None
    runtime_agents: bool | str | None = None
    max_runtime_agents: int | None = None
    compact_at: int | float | None = None
    contract_retries: int | None = None
    notes: str = ""

    def overrides(self) -> dict[str, Any]:
        """The constructor arguments this version changes."""
        out: dict[str, Any] = {}
        for field in type(self).model_fields:
            value = getattr(self, field)
            if value is None or field in {"notes", "prompts", "tools", "subagents"}:
                continue
            if field in OVERRIDABLE or field in {"guardrails", "budget"}:
                out[field] = value
        return out

    @classmethod
    def of(cls, value: AgentVersion | dict[str, Any]) -> AgentVersion:
        if isinstance(value, AgentVersion):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ConfigurationError(
            f"a version must be an AgentVersion or a dict — got {type(value).__name__}")
