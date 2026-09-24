"""End-to-end tracing: one span per run, step, tool call and sub-agent.

Spans nest automatically through a context variable, so a sub-agent three levels
down still lands under the run that started it.
"""

from __future__ import annotations

import contextvars
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ..types import new_id

__all__ = ["Span", "Tracer", "console_exporter", "jsonl_exporter", "current_span"]

_CURRENT: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "ah_current_span", default=None
)


def current_span() -> Span | None:
    return _CURRENT.get()


@dataclass
class Span:
    """One timed unit of work."""

    name: str
    kind: str = "run"  # run | step | tool | model | subagent | memory
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    id: str = field(default_factory=lambda: new_id("span"))
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    status: str = "ok"
    error: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000

    def set(self, **attrs: Any) -> Span:
        self.attrs.update(attrs)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.id, "parent_id": self.parent_id,
            "name": self.name, "kind": self.kind, "status": self.status,
            "duration_ms": round(self.duration_ms, 2), "start": self.start,
            "error": self.error, "attrs": self.attrs,
        }


Exporter = Callable[[Span], Any]


def console_exporter(stream: TextIO | None = None, *, indent: bool = True) -> Exporter:
    """Human-readable spans, indented by depth."""
    out = stream or sys.stderr
    depths: dict[str, int] = {}

    def export(span: Span) -> None:
        depth = depths.get(span.parent_id or "", 0) if indent else 0
        depths[span.id] = depth + 1
        pad = "  " * depth
        status = "" if span.status == "ok" else f" [{span.status}: {span.error}]"
        extra = ""
        if span.attrs.get("cost_usd"):
            extra = f" ${span.attrs['cost_usd']:.4f}"
        print(f"{pad}{span.kind}:{span.name} {span.duration_ms:.0f}ms{extra}{status}",
              file=out)

    return export


def jsonl_exporter(path: str | Path) -> Exporter:
    """One JSON object per span — the record you debug from."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    def export(span: Span) -> None:
        with file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(span.to_dict(), default=str) + "\n")

    export.path = file  # type: ignore[attr-defined]  # so `Tracer.forget` can rewrite it
    return export


class Tracer:
    """Creates spans and hands finished ones to its exporters."""

    def __init__(self, exporters: list[Exporter] | None = None, *,
                 enabled: bool = True, keep: int = 1000) -> None:
        self.exporters = list(exporters or [])
        self.enabled = enabled
        self.spans: list[Span] = []
        self.keep = keep

    def add_exporter(self, exporter: Exporter) -> None:
        self.exporters.append(exporter)

    def forget(self, trace_ids: set[str] | list[str]) -> int:
        """Drop these traces from memory and from every JSONL export.

        Spans carry the start of each task, so erasing a person includes them.
        Returns how many spans were dropped from memory.
        """
        doomed = set(trace_ids)
        before = len(self.spans)
        self.spans = [s for s in self.spans if s.trace_id not in doomed]
        for exporter in self.exporters:
            path = getattr(exporter, "path", None)
            if path is None or not Path(path).exists():
                continue
            kept = []
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line).get("trace_id") in doomed:
                        continue
                except json.JSONDecodeError:
                    pass
                kept.append(line)
            Path(path).write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        return before - len(self.spans)

    @contextmanager
    def span(self, name: str, kind: str = "run", **attrs: Any) -> Iterator[Span]:
        """Time a block. Exceptions mark the span failed and re-raise."""
        parent = _CURRENT.get()
        span = Span(
            name=name, kind=kind, attrs=dict(attrs),
            trace_id=parent.trace_id if parent else new_id("trace"),
            parent_id=parent.id if parent else None,
        )
        token = _CURRENT.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT.reset(token)
            span.end = time.time()
            self._finish(span)

    def _finish(self, span: Span) -> None:
        if len(self.spans) < self.keep:
            self.spans.append(span)
        if not self.enabled:
            return
        for exporter in self.exporters:
            try:
                exporter(span)
            except Exception:  # an exporter must never break a run
                continue

    def tree(self) -> str:
        """The collected spans as an indented tree — handy in tests and the CLI."""
        by_parent: dict[str | None, list[Span]] = {}
        for span in self.spans:
            by_parent.setdefault(span.parent_id, []).append(span)

        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for span in by_parent.get(parent, []):
                lines.append(f"{'  ' * depth}{span.kind}:{span.name} "
                             f"{span.duration_ms:.0f}ms")
                walk(span.id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)
