"""Jurisdiction and framework packs: the law, as data.

A pack says three things about one law, regulation or standard:

* **requirements** — its articles, each mapped to catalog controls (or marked
  ``manual`` where only people can meet it: appointing a DPO, registering in
  the EU database, signing a DPIA);
* **policy** — the rules, residency limits, retention and transparency
  settings it implies, merged into the governance policy;
* **incidents** — who must be told of what, and how fast.

Packs ship in ``governance/frameworks/packs/*.yaml``; your own go anywhere and
are loaded by path. Each carries the date its content was checked (``as_of``)
and its sources. Laws change; review a pack before relying on it.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import ConfigurationError
from .policy import Policy

__all__ = ["Requirement", "Pack", "load_pack", "list_packs", "PACK_DIR"]

PACK_DIR = "frameworks/packs"


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    title: str
    controls: list[str] = Field(default_factory=list)
    manual: bool = False
    note: str = ""


class Pack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    jurisdiction: str = "global"
    kind: Literal["law", "regulation", "guidance", "standard", "framework"] = "law"
    binding: bool = True
    status: str = ""
    as_of: str = ""
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    policy: Policy = Field(default_factory=Policy)
    #: incident kind → [{notify, within (hours), basis, min_severity}]
    incidents: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @field_validator("as_of", mode="before")
    @classmethod
    def _date(cls, value: Any) -> str:
        return str(value) if value is not None else ""

    def model_post_init(self, __context: Any) -> None:
        # Stamp the rules so every decision can say which law asked for it.
        self.policy.packs = [self.id]
        for rule in self.policy.rules:
            rule.source = rule.source or self.id
            if not rule.id.startswith(self.id + "/"):
                rule.id = f"{self.id}/{rule.id}"

    @property
    def controls(self) -> set[str]:
        return {c for r in self.requirements for c in r.controls}


def _builtin_dir() -> Any:
    return resources.files("agent_harness.governance").joinpath(PACK_DIR)


@lru_cache(maxsize=64)
def _load_builtin(pack_id: str) -> Pack:
    entry = _builtin_dir().joinpath(f"{pack_id}.yaml")
    if not entry.is_file():
        raise ConfigurationError(
            f"no governance pack named {pack_id!r}; available: {', '.join(list_packs())}")
    return _parse(entry.read_text(encoding="utf-8"), pack_id)


def _parse(text: str, where: str) -> Pack:
    import yaml

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigurationError(f"pack {where}: expected a mapping")
    try:
        return Pack.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(f"pack {where} is invalid: {exc}") from None


def load_pack(ref: str | Path) -> Pack:
    """A built-in pack by id (``"gdpr"``), or a YAML file by path."""
    text = str(ref)
    if text.endswith((".yaml", ".yml")) or "/" in text:
        path = Path(text)
        if not path.exists():
            raise ConfigurationError(f"pack file not found: {path}")
        return _parse(path.read_text(encoding="utf-8"), str(path))
    return _load_builtin(text).model_copy(deep=True)


def list_packs() -> list[str]:
    """The ids of every built-in pack."""
    return sorted(entry.name[:-5] for entry in _builtin_dir().iterdir()
                  if entry.name.endswith(".yaml"))
