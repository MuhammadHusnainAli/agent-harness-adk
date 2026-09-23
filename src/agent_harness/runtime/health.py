"""Service health: latency, failure rate, saturation. Is the platform itself healthy?

Every model call and tool call is timed and counted here, so when a run gets
slow you can tell whether it is the model, one tool, or your own concurrency cap.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["ServiceHealth", "ComponentHealth"]

Status = Literal["healthy", "degraded", "unhealthy", "unknown"]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return round(ordered[index], 2)


@dataclass
class ComponentHealth:
    """Rolling stats for one component — a provider, a tool, an MCP server."""

    name: str
    kind: str = "component"
    window: int = 200
    calls: int = 0
    failures: int = 0
    inflight: int = 0
    last_error: str = ""
    last_ts: float = 0.0
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, duration_ms: float, *, ok: bool = True, error: str = "") -> None:
        self.calls += 1
        self.last_ts = time.time()
        self.durations.append(duration_ms)
        if not ok:
            self.failures += 1
            self.last_error = error[:300]

    @property
    def failure_rate(self) -> float:
        return round(self.failures / self.calls, 4) if self.calls else 0.0

    @property
    def p50(self) -> float:
        return _percentile(list(self.durations), 0.50)

    @property
    def p95(self) -> float:
        return _percentile(list(self.durations), 0.95)

    def status(self, *, degraded_rate: float = 0.1,
               unhealthy_rate: float = 0.5) -> Status:
        if not self.calls:
            return "unknown"
        if self.failure_rate >= unhealthy_rate:
            return "unhealthy"
        if self.failure_rate >= degraded_rate:
            return "degraded"
        return "healthy"

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "status": self.status(),
            "calls": self.calls, "failures": self.failures,
            "failure_rate": self.failure_rate, "inflight": self.inflight,
            "p50_ms": self.p50, "p95_ms": self.p95,
            "last_error": self.last_error or None,
        }


class ServiceHealth:
    """The health of every component the harness talks to, in one place."""

    def __init__(self, *, window: int = 200, degraded_rate: float = 0.1,
                 unhealthy_rate: float = 0.5, saturation_warn: float = 0.9) -> None:
        self.window = window
        self.degraded_rate = degraded_rate
        self.unhealthy_rate = unhealthy_rate
        self.saturation_warn = saturation_warn
        self.components: dict[str, ComponentHealth] = {}
        self.started = time.time()

    def component(self, name: str, kind: str = "component") -> ComponentHealth:
        key = f"{kind}:{name}"
        if key not in self.components:
            entry = ComponentHealth(name=name, kind=kind, window=self.window)
            entry.durations = deque(maxlen=self.window)
            self.components[key] = entry
        return self.components[key]

    def record(self, name: str, duration_ms: float, *, kind: str = "component",
               ok: bool = True, error: str = "") -> None:
        self.component(name, kind).record(duration_ms, ok=ok, error=error)

    def begin(self, name: str, kind: str = "component") -> None:
        self.component(name, kind).inflight += 1

    def end(self, name: str, duration_ms: float, *, kind: str = "component",
            ok: bool = True, error: str = "") -> None:
        entry = self.component(name, kind)
        entry.inflight = max(0, entry.inflight - 1)
        entry.record(duration_ms, ok=ok, error=error)

    def saturation(self, scheduler: Any) -> float:
        """How full the concurrency pool is, 0.0 to 1.0."""
        try:
            limit = max(1, scheduler.max_concurrency)
            return round(min((scheduler.running + scheduler.queued) / limit, 1.0), 3)
        except AttributeError:
            return 0.0

    def status(self) -> Status:
        """The worst status across every component."""
        seen = [c.status(degraded_rate=self.degraded_rate,
                         unhealthy_rate=self.unhealthy_rate)
                for c in self.components.values()]
        if not seen or all(s == "unknown" for s in seen):
            return "unknown"
        if "unhealthy" in seen:
            return "unhealthy"
        if "degraded" in seen:
            return "degraded"
        return "healthy"

    def snapshot(self, scheduler: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status(),
            "uptime_s": round(time.time() - self.started, 1),
            "components": [c.snapshot() for c in self.components.values()],
        }
        if scheduler is not None:
            saturation = self.saturation(scheduler)
            out["saturation"] = saturation
            out["saturated"] = saturation >= self.saturation_warn
        return out

    def reset(self) -> None:
        self.components.clear()
        self.started = time.time()
