"""Sub-agents: the bench you reuse, and the factory that writes a new one.

Two supplies, one decision. Before staffing a task the orchestrator asks: is
there a pre-defined sub-agent that already covers this? If yes it is reused as
it is. If no, the factory writes a brand-new specialist during the run —
blueprint, prompt, tool allowlist, model and effort, workspace isolation and an
input/output contract — and that specialist exists only for this job.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .prompts import Prompt
from .providers.base import Provider
from .runtime.budget import Budget
from .runtime.permissions import PolicyGate
from .types import new_id

__all__ = ["SubAgentSpec", "build_agent", "Bench", "BENCH", "SubAgentFactory"]

_STOPWORDS = {
    "with", "from", "this", "that", "into", "your", "their", "them", "please",
    "would", "should", "about", "when", "what", "which", "where", "have", "been",
    "make", "made", "will", "must", "such", "than", "then", "they", "also", "each",
    "only", "very", "just", "over", "more", "most", "some", "here", "there", "were",
    "does", "done", "need", "needs", "want", "wants", "give", "take", "using", "use",
}


def _tokens(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {t for t in cleaned.split() if t}


def _matches_any(word: str, hay: set[str], *, prefix: int = 5) -> bool:
    if word in hay:
        return True
    head = word[:prefix]
    return any(token.startswith(head) or word.startswith(token[:prefix])
               for token in hay if len(token) >= 4)


class SubAgentSpec(BaseModel):
    """The blueprint a sub-agent is built from. Serialisable, versionable."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    instructions: str = ""
    tools: list[str] | None = None          # glob allowlist; None inherits the parent's
    skills: list[str] | None = None
    model: str | None = None
    tier: str | None = None
    effort: str | None = None
    max_steps: int = 12
    max_tokens: int = 8192
    temperature: float | None = None
    output_schema: dict[str, Any] | None = None
    workspace: Literal["isolated", "shared", "none"] = "none"
    allow_shell: bool = False
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    budget: Budget | None = None
    memory: bool = False                     # sub-agents start clean by design
    version: str = "1"
    origin: Literal["bench", "factory"] = "bench"
    tags: list[str] = Field(default_factory=list)

    def matches(self, task: str) -> float:
        """A cheap capability score used when no embedder is configured.

        Token-level with a shared-prefix rule, so "validate" finds the validator
        but a stopword like "with" never staffs anybody.
        """
        hay = _tokens(f"{self.name} {self.description} {' '.join(self.tags)}")
        words = _tokens(task) - _STOPWORDS
        words = {w for w in words if len(w) > 3}
        if not words:
            return 0.0
        hits = sum(1 for w in words if _matches_any(w, hay))
        return hits / len(words)


def build_agent(spec: SubAgentSpec, parent: Any) -> Any:
    """Turn a spec into a live Agent that shares the parent's harness."""
    from .agent import Agent

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
        max_tokens=spec.max_tokens,
        max_steps=spec.max_steps,
        tools=inherited,
        skills=skills,
        memory=spec.memory,
        harness=parent.harness,
        policy=policy,
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


# ---------------------------------------------------------------------------
# The bench — proven, versioned, reused as they are
# ---------------------------------------------------------------------------

BENCH_SPECS: list[SubAgentSpec] = [
    SubAgentSpec(
        name="research",
        description="Maps the source material and reports what it says, read-only.",
        instructions=(
            "Find what is actually there and report it. Quote or cite the source for "
            "every claim. If the sources disagree, say so rather than picking one. "
            "Never guess at a fact you could not find — list it as an open question."
        ),
        tags=["research", "discovery", "search", "sources", "investigate"],
        tier="balanced",
    ),
    SubAgentSpec(
        name="planner",
        description="Turns a goal into an ordered task graph with parallelism marked.",
        instructions=(
            "Break the goal into tasks that one worker could finish alone. For each "
            "task give: a one-line statement, what it depends on, and how we will "
            "know it is done. Mark which tasks can run at the same time. Keep the "
            "plan as short as the goal allows."
        ),
        tags=["plan", "planning", "breakdown", "scope", "schedule"],
        tier="deep",
    ),
    SubAgentSpec(
        name="document_extractor",
        description="Pulls structured fields out of documents, read-only.",
        instructions=(
            "Extract exactly the fields you were asked for. Use null when a field is "
            "genuinely absent — never invent a plausible value. Note the location in "
            "the source for anything you extracted."
        ),
        tags=["extract", "document", "pdf", "fields", "parse", "invoice"],
        tier="fast",
    ),
    SubAgentSpec(
        name="data_analyst",
        description="Aggregates data and explains the variance behind the numbers.",
        instructions=(
            "State the number, then what drives it. Show the calculation you used. "
            "Call out the caveats in the data before someone acts on it."
        ),
        tags=["data", "analysis", "numbers", "metrics", "variance", "statistics"],
        workspace="shared",
        tier="balanced",
    ),
    SubAgentSpec(
        name="validator",
        description="Re-checks work against the rules and reports pass or fail.",
        instructions=(
            "Check the work against the stated rules, one by one. For each: pass or "
            "fail, and the evidence. Do not fix anything — report. End with an "
            "overall verdict and the single most important problem."
        ),
        tags=["validate", "check", "qa", "verify", "review", "test"],
        tier="balanced",
    ),
    SubAgentSpec(
        name="compliance_checker",
        description="Checks policy and data-handling rules, read-only.",
        instructions=(
            "Check the work against the policies you were given. Quote the clause "
            "behind every finding. Flag anything that handles personal or regulated "
            "data. When a rule is ambiguous, say it is ambiguous — do not resolve it."
        ),
        tags=["compliance", "policy", "legal", "regulation", "privacy", "risk"],
        tier="deep",
    ),
    SubAgentSpec(
        name="drafting",
        description="Writes correspondence and short-form copy to a brief.",
        instructions=(
            "Write to the brief and the audience you were given. Plain sentences, no "
            "filler, no hedging. Match the register you were asked for; if none was "
            "given, write like a competent colleague."
        ),
        tags=["draft", "write", "email", "letter", "copy", "correspondence"],
        tier="balanced",
    ),
    SubAgentSpec(
        name="report_writer",
        description="Consolidates findings into one structured report.",
        instructions=(
            "Lead with the answer, then the evidence. Merge duplicate findings and "
            "attribute each one to where it came from. Keep every number traceable "
            "to its source. No filler sections."
        ),
        tags=["report", "summary", "consolidate", "write-up", "document"],
        workspace="shared",
        tier="balanced",
    ),
]


class Bench:
    """The pre-defined sub-agents: proven, versioned, evaluated, reused as they are."""

    def __init__(self, specs: Iterable[SubAgentSpec] = ()) -> None:
        self._specs: dict[str, SubAgentSpec] = {s.name: s for s in specs}

    @classmethod
    def standard(cls) -> Bench:
        return cls(s.model_copy(deep=True) for s in BENCH_SPECS)

    def register(self, spec: SubAgentSpec) -> SubAgentSpec:
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> SubAgentSpec | None:
        return self._specs.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def find(self, task: str, *, threshold: float = 0.3) -> SubAgentSpec | None:
        """The staffing decision: is there already a sub-agent that covers this?"""
        best: tuple[float, SubAgentSpec | None] = (0.0, None)
        for spec in self._specs.values():
            score = spec.matches(task)
            if score > best[0]:
                best = (score, spec)
        return best[1] if best[0] >= threshold else None

    def catalogue(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in self._specs.values())


BENCH = Bench.standard


# ---------------------------------------------------------------------------
# The factory — a brand-new specialist, written during the run
# ---------------------------------------------------------------------------

FACTORY_PROMPT = Prompt(
    "subagent.factory",
    """Write the specification for a brand-new sub-agent that will do exactly one task.

TASK
{task}

TOOLS IT COULD BE GIVEN
{tools}

Decide, in order:
1. name — lower_snake_case, specific to the task, not a generic title.
2. description — one line, what it is accountable for.
3. instructions — its operating instructions. Address it as "you". Say what done
   looks like and what it must never do. No preamble, no flattery.
4. tools — the smallest set from the list above that can finish the task. Least
   privilege: if it only reads, give it no writing tools. Use [] for none.
5. tier — "fast" for mechanical work, "balanced" for normal judgement, "deep" for
   hard reasoning.
6. max_steps — a realistic ceiling for this task.
7. workspace — "isolated" if it writes files, "shared" if it must hand files to
   another sub-agent, otherwise "none".

Return only this JSON object:
{{"name": "...", "description": "...", "instructions": "...", "tools": ["..."],
  "tier": "fast|balanced|deep", "max_steps": 8, "workspace": "none"}}""",
)


class SubAgentFactory:
    """Writes a new sub-agent spec at run time when the bench has nothing that fits."""

    def __init__(self, builder: Any, *, bench: Bench | None = None) -> None:
        self.builder = builder  # an Agent used purely to author the spec
        self.bench = bench if bench is not None else Bench.standard()

    async def create(self, task: str, *, tools: Iterable[str] = (),
                     keep: bool = False) -> SubAgentSpec:
        """Build a spec for `task`. Falls back to a safe generic spec on any problem."""
        catalogue = "\n".join(f"- {t}" for t in tools) or "(no tools available)"
        prompt = FACTORY_PROMPT.render(task=task, tools=catalogue)
        spec = self._fallback(task, tools)

        result = await self.builder.run(prompt, messages=[])
        if not result.error:
            drafted = _parse_spec(result.output)
            if drafted:
                drafted.setdefault("description", f"Purpose-built for: {task[:80]}")
                drafted["origin"] = "factory"
                allowed = set(tools)
                if allowed and drafted.get("tools"):
                    drafted["tools"] = [t for t in drafted["tools"] if t in allowed]
                try:
                    spec = SubAgentSpec(**drafted)
                except (TypeError, ValueError):
                    pass
        if keep:
            self.bench.register(spec)
        return spec

    def _fallback(self, task: str, tools: Iterable[str]) -> SubAgentSpec:
        return SubAgentSpec(
            name=f"task_{new_id()[:6]}",
            description=f"Purpose-built for: {task[:80]}",
            instructions=(
                "You have one task, stated below by the agent that assigned it. "
                "Finish it and report the result. Do not expand the scope. If you "
                "cannot finish it, say exactly what stopped you."
            ),
            tools=list(tools) or None,
            origin="factory",
            max_steps=10,
        )


def _parse_spec(text: str) -> dict[str, Any] | None:
    from .agent import _extract_json

    blob = _extract_json(text)
    if isinstance(blob, dict):
        return blob
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
