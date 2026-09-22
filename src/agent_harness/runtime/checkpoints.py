"""Checkpoints and time travel: durable state at every step, re-runnable.

No work is lost on failure, and you can reproduce a run from any prior step
instead of guessing what the agent was holding at the time.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ConfigurationError
from ..types import Message, Usage

__all__ = ["Checkpoint", "Checkpointer"]


class Checkpoint(BaseModel):
    """A run frozen mid-flight."""

    run_id: str
    step: int = 0
    ts: float = Field(default_factory=time.time)
    agent: str = ""
    session_id: str = ""
    label: str = ""
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    state: dict[str, Any] = Field(default_factory=dict)


class Checkpointer:
    """Step snapshots, in memory or on disk. `every=0` disables snapshotting."""

    def __init__(self, path: str | Path | None = None, *, every: int = 1,
                 keep: int = 100) -> None:
        self.path = Path(path) if path else None
        self.every = every
        self.keep = keep
        self._mem: dict[str, list[Checkpoint]] = {}
        self._lock = asyncio.Lock()
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.every > 0

    def should_save(self, step: int) -> bool:
        return self.enabled and (step % self.every == 0)

    def _file(self, run_id: str) -> Path:
        assert self.path is not None
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
        return self.path / f"{safe}.jsonl"

    async def save(self, checkpoint: Checkpoint) -> Checkpoint:
        rows = self._mem.setdefault(checkpoint.run_id, [])
        rows.append(checkpoint)
        if len(rows) > self.keep:
            del rows[: len(rows) - self.keep]
        if self.path:
            line = checkpoint.model_dump_json() + "\n"
            async with self._lock:
                await asyncio.to_thread(self._append, self._file(checkpoint.run_id), line)
        return checkpoint

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    async def history(self, run_id: str) -> list[Checkpoint]:
        if run_id in self._mem:
            return list(self._mem[run_id])
        if not self.path:
            return []
        file = self._file(run_id)
        if not file.exists():
            return []
        text = await asyncio.to_thread(file.read_text, "utf-8")
        rows: list[Checkpoint] = []
        for line in text.splitlines():
            if line.strip():
                try:
                    rows.append(Checkpoint(**json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
        return rows

    async def load(self, run_id: str, step: int | None = None) -> Checkpoint:
        """The checkpoint at `step`, or the latest one."""
        rows = await self.history(run_id)
        if not rows:
            raise ConfigurationError(f"no checkpoints for run {run_id!r}")
        if step is None:
            return rows[-1]
        for row in reversed(rows):
            if row.step <= step:
                return row
        raise ConfigurationError(f"run {run_id!r} has no checkpoint at or before step {step}")

    async def runs(self) -> list[str]:
        known = set(self._mem)
        if self.path:
            known |= {f.stem for f in self.path.glob("*.jsonl")}
        return sorted(known)

    async def delete(self, run_id: str) -> None:
        self._mem.pop(run_id, None)
        if self.path:
            self._file(run_id).unlink(missing_ok=True)
