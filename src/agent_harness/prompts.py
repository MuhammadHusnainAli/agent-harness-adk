"""Prompts as first-class, versioned objects — not string literals buried in code.

    greet = Prompt("greet", "Say hello to {name} in {language}.")
    greet.render(name="Ada", language="French")

A directory of Markdown files with YAML frontmatter becomes a PromptLibrary, so
prompts can be reviewed and versioned like any other asset.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError

__all__ = [
    "Prompt",
    "PromptLibrary",
    "parse_frontmatter",
    "sections",
]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split `---\\nkey: value\\n---\\nbody` into (metadata, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    import yaml

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"bad frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        return {}, text
    return meta, parts[2].lstrip("\n")


def sections(*parts: tuple[str, str] | str | None, sep: str = "\n\n") -> str:
    """Join non-empty prompt sections, titling the ones given as (heading, body)."""
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, tuple):
            heading, body = part
            if body and body.strip():
                out.append(f"## {heading}\n{body.strip()}")
        elif part.strip():
            out.append(part.strip())
    return sep.join(out)


class _SafeFormatter(string.Formatter):
    """Leave unknown placeholders alone instead of raising mid-render."""

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> tuple[Any, str]:
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, IndexError, AttributeError, TypeError):
            return "{" + field_name + "}", field_name


_SAFE = _SafeFormatter()
_STRICT = string.Formatter()


class Prompt(BaseModel):
    """A named, versioned template with default variables."""

    model_config = ConfigDict(extra="allow")

    name: str
    template: str = ""
    description: str = ""
    version: str = "1"
    engine: Literal["auto", "format", "jinja"] = "auto"
    variables: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None

    def __init__(self, name: str | None = None, template: str | None = None, **data: Any):
        if name is not None:
            data["name"] = name
        if template is not None:
            data["template"] = template
        super().__init__(**data)

    # ---- rendering ----------------------------------------------------
    def render(self, _strict: bool = False, **variables: Any) -> str:
        """Fill the template. Unknown placeholders survive unless `_strict`."""
        merged: dict[str, Any] = {**self.variables, **variables}
        text = self.template
        if self.engine == "jinja" or (self.engine == "auto" and _has_jinja(text)):
            return _render_jinja(text, merged)
        try:
            if _strict:
                return _STRICT.vformat(text, (), merged)
            return _SAFE.vformat(text, (), _Defaults(merged))
        except (KeyError, IndexError) as exc:
            raise ConfigurationError(
                f"prompt {self.name!r} is missing variable {exc}"
            ) from exc

    def partial(self, **variables: Any) -> Prompt:
        """A copy with some variables already bound."""
        return self.model_copy(update={"variables": {**self.variables, **variables}})

    def __add__(self, other: Prompt | str) -> Prompt:
        tail = other.template if isinstance(other, Prompt) else str(other)
        return self.model_copy(update={"template": f"{self.template}\n\n{tail}".strip()})

    def __str__(self) -> str:
        return self.render()

    # ---- io -----------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> Prompt:
        file = Path(path)
        meta, body = parse_frontmatter(file.read_text(encoding="utf-8"))
        meta.setdefault("name", file.stem)
        return cls(template=body.strip(), path=str(file), **meta)

    @classmethod
    def from_text(cls, text: str, *, name: str = "prompt") -> Prompt:
        meta, body = parse_frontmatter(text)
        meta.setdefault("name", name)
        return cls(template=body.strip(), **meta)

    def save(self, path: str | Path) -> Path:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        meta = {"name": self.name, "description": self.description,
                "version": self.version}
        if self.tags:
            meta["tags"] = self.tags
        front = yaml.safe_dump(meta, sort_keys=False).strip()
        file.write_text(f"---\n{front}\n---\n\n{self.template}\n", encoding="utf-8")
        return file


class _Defaults(dict):
    """Mapping that never KeyErrors during safe formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _has_jinja(text: str) -> bool:
    # Only a Jinja *statement* is unambiguous: `{{ }}` collides with the `{{`
    # escape that str.format templates use for a literal brace.
    return "{%" in text


def _render_jinja(text: str, variables: Mapping[str, Any]) -> str:
    try:
        from jinja2 import Template, select_autoescape
    except ImportError as exc:  # pragma: no cover - optional path
        raise ConfigurationError(
            "this prompt uses Jinja syntax; `pip install jinja2` to render it"
        ) from exc
    # A prompt is plain text on its way to a model, so HTML-escaping it would
    # corrupt it — `&amp;` in a prompt is a bug, not a defence. `select_autoescape`
    # says that explicitly and still escapes if a prompt is ever loaded from an
    # .html or .xml template, where the escaping would actually matter.
    template = Template(
        text,
        keep_trailing_newline=True,
        autoescape=select_autoescape(
            enabled_extensions=("html", "xml"),
            default_for_string=False,
            default=False,
        ),
    )
    return template.render(**variables)


class PromptLibrary:
    """A folder of prompts, addressable by name."""

    def __init__(self, prompts: Iterable[Prompt] = ()) -> None:
        self._prompts: dict[str, Prompt] = {p.name: p for p in prompts}

    @classmethod
    def from_dir(cls, path: str | Path, *, pattern: str = "*.md",
                 recursive: bool = True) -> PromptLibrary:
        root = Path(path)
        if not root.exists():
            raise ConfigurationError(f"prompt directory not found: {root}")
        glob = root.rglob(pattern) if recursive else root.glob(pattern)
        return cls(Prompt.from_file(f) for f in sorted(glob) if f.is_file())

    def add(self, prompt: Prompt) -> Prompt:
        self._prompts[prompt.name] = prompt
        return prompt

    def get(self, name: str) -> Prompt:
        if name not in self._prompts:
            known = ", ".join(sorted(self._prompts)) or "none"
            raise ConfigurationError(f"no prompt {name!r}; available: {known}")
        return self._prompts[name]

    def render(self, name: str, **variables: Any) -> str:
        return self.get(name).render(**variables)

    @property
    def names(self) -> list[str]:
        return sorted(self._prompts)

    def __contains__(self, name: object) -> bool:
        return name in self._prompts

    def __iter__(self):
        return iter(self._prompts.values())

    def __len__(self) -> int:
        return len(self._prompts)
