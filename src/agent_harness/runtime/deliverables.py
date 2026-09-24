"""Deliverable store: the documents, reports and artefacts a run produced.

The output of the run, kept apart from the transcript that produced it. Versioned
by name, so a second pass at `report.md` does not destroy the first.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..types import Artifact

__all__ = ["DeliverableStore"]


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "artifact"


class DeliverableStore:
    """Versioned artefacts, in memory or on disk."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._items: dict[str, list[Artifact]] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            self._reload()

    def _reload(self) -> None:
        assert self.root is not None
        index = self.root / "index.jsonl"
        if not index.exists():
            return
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            artifact = Artifact(**blob)
            self._items.setdefault(artifact.name, []).append(artifact)

    # ---- writing ----------------------------------------------------------
    def put(self, artifact: Artifact, *, run_id: str = "") -> Artifact:
        """Store a new version of an artefact and return it with its path set."""
        versions = self._items.setdefault(artifact.name, [])
        stored = artifact.model_copy(deep=True)
        stored.ts = stored.ts or time.time()
        if run_id:
            stored.run_id = run_id
        stored.version = len(versions) + 1
        stored.digest = hashlib.sha256(stored.content.encode()).hexdigest()[:16]

        if self.root:
            folder = self.root / _safe(stored.name)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"v{len(versions) + 1}{Path(stored.name).suffix or '.txt'}"
            path.write_text(stored.content, encoding="utf-8")
            stored.path = str(path)
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(stored.model_dump_json() + "\n")

        versions.append(stored)
        return stored

    def extend(self, artifacts: list[Artifact], *, run_id: str = "") -> list[Artifact]:
        return [self.put(a, run_id=run_id) for a in artifacts]

    # ---- reading ----------------------------------------------------------
    def get(self, name: str, version: int | None = None) -> Artifact | None:
        versions = self._items.get(name)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        return versions[version - 1] if 0 < version <= len(versions) else None

    def versions(self, name: str) -> list[Artifact]:
        return list(self._items.get(name, []))

    def latest(self) -> list[Artifact]:
        """The newest version of every artefact."""
        return [v[-1] for v in self._items.values() if v]

    def for_run(self, run_id: str) -> list[Artifact]:
        return [a for versions in self._items.values() for a in versions
                if getattr(a, "run_id", "") == run_id]

    @property
    def names(self) -> list[str]:
        return sorted(self._items)

    def __len__(self) -> int:
        return sum(len(v) for v in self._items.values())

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def manifest(self) -> list[dict[str, Any]]:
        """What the run produced, one row per artefact."""
        return [
            {
                "name": a.name, "version": getattr(a, "version", 1),
                "produced_by": a.produced_by, "media_type": a.media_type,
                "chars": len(a.content), "digest": getattr(a, "digest", ""),
                "path": a.path,
            }
            for a in self.latest()
        ]

    def clear(self) -> None:
        self._items.clear()

    def forget_runs(self, run_ids: set[str] | list[str]) -> list[str]:
        """Delete every version produced by these runs, files and index included.

        Returns the names of what was removed. Used when a person is erased.
        """
        doomed = set(run_ids)
        removed: list[str] = []
        for name in list(self._items):
            kept = []
            for artifact in self._items[name]:
                if getattr(artifact, "run_id", "") in doomed:
                    removed.append(name)
                    if artifact.path:
                        Path(artifact.path).unlink(missing_ok=True)
                else:
                    kept.append(artifact)
            if kept:
                self._items[name] = kept
            else:
                del self._items[name]
        if removed and self.root:
            index = self.root / "index.jsonl"
            index.write_text("".join(a.model_dump_json() + "\n"
                                     for versions in self._items.values()
                                     for a in versions), encoding="utf-8")
        return sorted(set(removed))
