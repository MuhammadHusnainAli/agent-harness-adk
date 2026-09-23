"""The factory: a brand-new specialist, written during the run."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..prompts import Prompt
from ..types import new_id
from .bench import Bench
from .spec import SubAgentSpec

__all__ = ["SubAgentFactory", "FACTORY_PROMPT"]


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
    from ..agent import _extract_json

    blob = _extract_json(text)
    if isinstance(blob, dict):
        return blob
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
