"""Skills: packaged know-how an agent can pull in when the task calls for it.

A skill is a folder (or a single Markdown file) with YAML frontmatter:

    skills/refunds/SKILL.md
    ---
    name: refunds
    description: How we process a refund, including the approval thresholds.
    ---
    1. Check the order is inside the 30-day window...

Only the *name and description* of each skill go into the system prompt. The
body is loaded on demand through the `load_skill` tool, so twenty skills cost
twenty lines of context instead of twenty documents.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError
from .prompts import parse_frontmatter
from .tools import Tool, tool

__all__ = ["Skill", "SkillRegistry"]

_SKILL_FILES = ("SKILL.md", "skill.md", "README.md")


class Skill(BaseModel):
    """Instructions plus optional tools and reference files, under one name."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str
    description: str = ""
    instructions: str = ""
    version: str = "1"
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    resources: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    tools: list[Tool] = Field(default_factory=list, exclude=True)

    # ---- loading ------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> Skill:
        file = Path(path)
        meta, body = parse_frontmatter(file.read_text(encoding="utf-8"))
        meta.setdefault("name", file.parent.name if file.stem.lower() == "skill"
                        else file.stem)
        meta.setdefault("description", _first_line(body))
        return cls(instructions=body.strip(), path=str(file.parent), **meta)

    @classmethod
    def from_dir(cls, path: str | Path, *, load_tools: bool = True) -> Skill:
        root = Path(path)
        doc = next((root / n for n in _SKILL_FILES if (root / n).exists()), None)
        if doc is None:
            raise ConfigurationError(f"{root} has no SKILL.md")
        skill = cls.from_file(doc)
        skill.path = str(root)
        for extra in sorted(root.rglob("*")):
            if extra.is_file() and extra != doc and extra.suffix != ".pyc":
                skill.resources[str(extra.relative_to(root))] = extra.suffix.lstrip(".")
        if load_tools:
            skill.tools.extend(_load_tools(root))
        return skill

    # ---- prompt surfaces ----------------------------------------------
    def index_entry(self) -> str:
        """The one line that goes into the system prompt."""
        return f"- {self.name}: {self.description}"

    def body(self) -> str:
        """The full text, handed over when the agent loads the skill."""
        out = [f"# Skill: {self.name}", self.instructions.strip()]
        if self.resources:
            listing = "\n".join(
                f"- {Path(self.path or '.') / name}" for name in sorted(self.resources)
            )
            out.append(f"## Files bundled with this skill\n{listing}")
        if self.tools:
            names = ", ".join(t.name for t in self.tools)
            out.append(f"## Tools this skill brings\n{names}")
        return "\n\n".join(p for p in out if p.strip())


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            return line.strip()
    return ""


def _load_tools(root: Path) -> list[Tool]:
    """Import `tools.py` from a skill folder and collect its Tool objects."""
    module_file = root / "tools.py"
    if not module_file.exists():
        return []
    spec = importlib.util.spec_from_file_location(f"_skill_{root.name}_tools", module_file)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared = getattr(module, "TOOLS", None)
    if declared:
        return [t if isinstance(t, Tool) else Tool(t) for t in declared]
    return [v for v in vars(module).values() if isinstance(v, Tool)]


class SkillRegistry:
    """The skills an agent knows about, plus the tool that loads one."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            self.add(skill)

    @classmethod
    def from_dir(cls, path: str | Path) -> SkillRegistry:
        """Load every skill folder (or loose .md file) under `path`."""
        root = Path(path)
        if not root.exists():
            raise ConfigurationError(f"skills directory not found: {root}")
        found: list[Skill] = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and any((child / n).exists() for n in _SKILL_FILES):
                found.append(Skill.from_dir(child))
            elif child.is_file() and child.suffix == ".md":
                found.append(Skill.from_file(child))
        return cls(found)

    def add(self, skill: Skill | str | Path) -> Skill:
        if not isinstance(skill, Skill):
            target = Path(skill)
            skill = Skill.from_dir(target) if target.is_dir() else Skill.from_file(target)
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            known = ", ".join(sorted(self._skills)) or "none"
            raise ConfigurationError(f"no skill {name!r}; available: {known}")
        return self._skills[name]

    def select(self, names: Iterable[str] | None) -> SkillRegistry:
        if names is None:
            return self
        return SkillRegistry(self._skills[n] for n in names if n in self._skills)

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def index(self) -> str:
        """The catalogue block for the system prompt — names and one-liners only."""
        if not self._skills:
            return ""
        lines = [s.index_entry() for s in self._skills.values()]
        return (
            "Skills available to you (call `load_skill` to read one in full before "
            "you rely on it):\n" + "\n".join(lines)
        )

    def tools(self) -> list[Tool]:
        """Tools contributed by the loaded skills, plus `load_skill` itself."""
        bundled = [t for s in self._skills.values() for t in s.tools]
        return [self.load_tool(), *bundled]

    def load_tool(self) -> Tool:
        registry = self

        @tool(name="load_skill", tags=["builtin", "skills"],
              description="Read a skill's full instructions before doing work it covers.")
        def load_skill(name: str) -> str:
            """Load one skill's instructions.

            Args:
                name: the skill's name, exactly as listed in the skills catalogue.
            """
            return registry.get(name).body()

        return load_skill
