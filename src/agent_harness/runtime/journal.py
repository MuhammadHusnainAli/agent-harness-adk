"""Run journal: what each agent was asked, and what it returned.

The append-only record you debug a multi-agent run from — and the audit trail
compliance will ask for.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..types import new_id

__all__ = ["JournalEntry", "RunJournal"]


class JournalEntry(BaseModel):
    id: str = Field(default_factory=lambda: new_id("jrn"))
    ts: float = Field(default_factory=time.time)
    run_id: str = ""
    trace_id: str = ""
    agent: str = ""
    kind: str = "event"  # assignment | handback | tool | model | decision | error
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.ts))
        return f"[{stamp}] {self.agent or '-'} {self.kind}: {self.text}"


class RunJournal:
    """In-memory by default; give it a path and it survives the process."""

    def __init__(self, path: str | Path | None = None, *, echo: bool = False) -> None:
        self.path = Path(path) if path else None
        self.entries: list[JournalEntry] = []
        self.echo = echo
        self._lock = asyncio.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, kind: str, text: str, *, agent: str = "", run_id: str = "",
                    trace_id: str = "", **data: Any) -> JournalEntry:
        entry = JournalEntry(kind=kind, text=text, agent=agent, run_id=run_id,
                             trace_id=trace_id, data=data)
        self.entries.append(entry)
        if self.echo:
            print(entry.line())
        if self.path:
            async with self._lock:
                await asyncio.to_thread(self._append, entry)
        return entry

    def _append(self, entry: JournalEntry) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def forget(self, run_ids: set[str] | list[str]) -> int:
        """Drop every entry for these runs, on disk too. Returns how many went.

        The journal is a debugging record, not an immutable one, so erasing a
        person's runs from it is a rewrite.
        """
        doomed = set(run_ids)
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.run_id not in doomed]
        if self.path and self.path.exists():
            kept = []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("run_id") in doomed:
                        continue
                except json.JSONDecodeError:
                    pass
                kept.append(line)
            self.path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        return before - len(self.entries)

    async def assignment(self, agent: str, task: str, **data: Any) -> JournalEntry:
        return await self.write("assignment", task, agent=agent, **data)

    async def handback(self, agent: str, summary: str, **data: Any) -> JournalEntry:
        return await self.write("handback", summary, agent=agent, **data)

    def for_agent(self, agent: str) -> list[JournalEntry]:
        return [e for e in self.entries if e.agent == agent]

    def render(self, limit: int | None = None) -> str:
        rows = self.entries[-limit:] if limit else self.entries
        return "\n".join(e.line() for e in rows)

    @classmethod
    def load(cls, path: str | Path) -> RunJournal:
        journal = cls(path)
        file = Path(path)
        if file.exists():
            for line in file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        journal.entries.append(JournalEntry(**json.loads(line)))
                    except (json.JSONDecodeError, ValueError):
                        continue
        return journal
