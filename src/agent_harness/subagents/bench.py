"""The bench: pre-defined sub-agents — proven, versioned, reused as they are."""

from __future__ import annotations

from collections.abc import Iterable

from .spec import SubAgentSpec

__all__ = ["Bench", "BENCH", "BENCH_SPECS"]


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
