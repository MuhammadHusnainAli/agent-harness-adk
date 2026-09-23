"""Declare a whole agent tree in one file, then build it.

    # agents.yaml
    version: v2
    defaults:
      model: claude-opus-5
      memory: true

    prompts:
      house_style: Answer in plain sentences. Cite the order.

    subagents:
      researcher:
        description: Finds things out, read-only.
        instructions: Cite every claim.
        tools: [lookup]
        tier: fast
        budget: {max_input_tokens: 10000, max_output_tokens: 2000}
        guardrails: {require_citation: true}

    agents:
      support:
        instructions: "{house_style}"
        tools: [order_status, lookup]
        subagents: [researcher]
        guardrails: strict
        versions:
          v1: {instructions: Answer order questions., tools: [order_status]}
          v2: {instructions: "{house_style}"}

    guardrails:
      strict:
        require_tools: [order_status]
        no_placeholders: true

```python
blueprint = Blueprint.from_file("agents.yaml")
agent = blueprint.build("support", tools=[order_status, lookup])
```

Tools cannot come from a file — they are code. Either hand them in, or let the
file name them as import paths (`myapp.tools:order_status`) and they are
imported. Everything else — prompts, sub-agents, budgets, guardrails, versions —
is declaration, and belongs in the file where it can be reviewed and diffed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigurationError
from .runtime.budget import Budget
from .subagents import SubAgentSpec
from .tools import Tool, ToolRegistry
from .versioning import AgentVersion

__all__ = ["Blueprint", "AgentEntry"]


class AgentEntry(BaseModel):
    """One agent, as declared in the file."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    instructions: str = ""
    description: str = ""
    model: str | None = None
    tier: str | None = None
    effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    tools: list[str] | None = None
    skills: str | list[str] | None = None
    subagents: list[str] = Field(default_factory=list)
    memory: bool | None = None
    trace: dict[str, Any] | str | None = None
    budget: dict[str, Any] | Budget | None = None
    guardrails: str | dict[str, Any] | None = None
    runtime_agents: bool | str | None = None
    max_runtime_agents: int | None = None
    compact_at: int | float | None = None
    workspace: bool | str | None = None
    allow_shell: bool = False
    output_schema: dict[str, Any] | None = None
    version: str | None = None
    versions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class Blueprint(BaseModel):
    """A file's worth of declarations, and the agents it can build."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    version: str = "v1"
    defaults: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    guardrails: dict[str, dict[str, Any]] = Field(default_factory=dict)
    subagents: dict[str, SubAgentSpec] = Field(default_factory=dict)
    agents: dict[str, AgentEntry] = Field(default_factory=dict)
    memory: str | dict[str, Any] | None = None
    path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _name_from_key(cls, data: Any) -> Any:
        """In a file the name is the mapping key, so fill it in before parsing."""
        if not isinstance(data, dict):
            return data
        subagents = data.get("subagents")
        if isinstance(subagents, dict):
            data = {**data, "subagents": {
                key: ({"name": key, **spec} if isinstance(spec, dict) and
                      "name" not in spec else spec)
                for key, spec in subagents.items()
            }}
        return data

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> Blueprint:
        """Load a `.yaml`, `.yml` or `.json` file."""
        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"no blueprint at {file}")
        return cls.from_text(file.read_text(encoding="utf-8"),
                             fmt=file.suffix.lstrip("."), path=str(file))

    @classmethod
    def from_text(cls, text: str, *, fmt: str = "yaml",
                  path: str | None = None) -> Blueprint:
        if fmt in {"json"}:
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigurationError("a blueprint must be a mapping at the top level")
        return cls(**data, path=path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blueprint:
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        """Write it back out, in whichever format the extension asks for."""
        file = Path(path)
        blob = self.model_dump(mode="json", exclude_none=True, exclude={"path"})
        if file.suffix == ".json":
            file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        else:
            import yaml

            file.write_text(yaml.safe_dump(blob, sort_keys=False), encoding="utf-8")
        return file

    # ---- resolving ----------------------------------------------------------
    def render(self, text: str) -> str:
        """Substitute `{prompt_name}` references from the prompts section."""
        if not text or "{" not in text:
            return text
        from .prompts import Prompt

        return Prompt("_inline", text).render(**self.prompts)

    def _guardrails(self, value: str | dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if value not in self.guardrails:
            raise ConfigurationError(
                f"guardrails {value!r} are not defined; known: "
                f"{', '.join(sorted(self.guardrails)) or 'none'}")
        return self.guardrails[value]

    def _tools(self, wanted: list[str] | None,
               registry: ToolRegistry) -> list[Tool]:
        """Named tools, from the registry you passed or imported by path."""
        if wanted is None:
            return list(registry)
        out: list[Tool] = []
        missing: list[str] = []
        for name in wanted:
            if name in registry:
                out.append(registry.get(name))
            elif ":" in name or "." in name:
                out.append(_import_tool(name))
            else:
                missing.append(name)
        if missing:
            known = ", ".join(registry.names) or "none"
            raise ConfigurationError(
                f"the blueprint asks for tools that were not supplied: "
                f"{', '.join(missing)}. Pass them to build(tools=[...]), or give "
                f"an import path like 'myapp.tools:{missing[0]}'. Available: {known}")
        return out

    def _subagent(self, name: str) -> SubAgentSpec:
        if name not in self.subagents:
            raise ConfigurationError(
                f"sub-agent {name!r} is not declared; known: "
                f"{', '.join(sorted(self.subagents)) or 'none'}")
        spec = self.subagents[name].model_copy(deep=True)
        spec.instructions = self.render(spec.instructions)
        return spec

    # ---- building ------------------------------------------------------------
    def build(self, name: str, *, tools: Any = (), harness: Any = None,
              **overrides: Any) -> Any:
        """Build one declared agent."""
        from .agent import Agent

        if name not in self.agents:
            raise ConfigurationError(
                f"no agent {name!r} in this blueprint; declared: "
                f"{', '.join(sorted(self.agents)) or 'none'}")
        entry = self.agents[name]
        registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)

        kwargs: dict[str, Any] = {
            "instructions": self.render(entry.instructions),
            "description": entry.description,
            "tools": self._tools(entry.tools, registry),
            "subagents": [self._subagent(n) for n in entry.subagents],
        }
        for field in ("model", "tier", "effort", "temperature", "max_tokens",
                      "max_steps", "runtime_agents", "max_runtime_agents",
                      "compact_at", "trace", "skills", "allow_shell"):
            value = getattr(entry, field)
            if value is not None:
                kwargs[field] = value
        if entry.memory is not None:
            kwargs["memory"] = entry.memory
        if entry.workspace is not None:
            kwargs["workspace"] = entry.workspace
        if entry.budget is not None:
            kwargs["budget"] = (entry.budget if isinstance(entry.budget, Budget)
                                else Budget(**entry.budget))
        rails = self._guardrails(entry.guardrails)
        if rails is not None:
            kwargs["guardrails"] = rails
        if entry.output_schema:
            from .subagents.builder import _model_from_schema

            kwargs["output_type"] = _model_from_schema(name, entry.output_schema)

        if entry.versions:
            kwargs["versions"] = {
                key: self._version(spec) for key, spec in entry.versions.items()
            }
            kwargs["version"] = entry.version or self.version

        # Defaults apply where neither the agent nor the caller said otherwise.
        for key, value in self.defaults.items():
            kwargs.setdefault(key, value)
        kwargs.update(overrides)
        if harness is not None:
            kwargs["harness"] = harness
        return Agent(name, **kwargs)

    def _version(self, spec: dict[str, Any]) -> AgentVersion:
        version = AgentVersion.of(dict(spec))
        if version.instructions:
            version.instructions = self.render(version.instructions)
        if version.subagents:
            version.subagents = [
                self._subagent(s) if isinstance(s, str) else s
                for s in version.subagents
            ]
        return version

    def build_all(self, *, tools: Any = (), harness: Any = None,
                  **overrides: Any) -> dict[str, Any]:
        """Every declared agent, sharing one harness."""
        from .harness import Harness

        shared = harness if harness is not None else Harness()
        return {name: self.build(name, tools=tools, harness=shared, **overrides)
                for name in self.agents}

    def memory_store(self) -> Any:
        """The store the file declares, if it declares one."""
        if self.memory is None:
            return None
        from .memory import memory_provider

        if isinstance(self.memory, str):
            return memory_provider(self.memory)
        options = dict(self.memory)
        url = options.pop("url", None)
        if not url:
            raise ConfigurationError("a memory section needs a `url`")
        return memory_provider(url, **options)

    @property
    def names(self) -> list[str]:
        return sorted(self.agents)


def _import_tool(path: str) -> Tool:
    """Import a tool named as `package.module:name` or `package.module.name`."""
    module_name, _, attribute = path.partition(":")
    if not attribute:
        module_name, _, attribute = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigurationError(f"cannot import {path!r}: {exc}") from exc
    found = getattr(module, attribute, None)
    if found is None:
        raise ConfigurationError(f"{module_name} has no {attribute!r}")
    return found if isinstance(found, Tool) else Tool(found)
