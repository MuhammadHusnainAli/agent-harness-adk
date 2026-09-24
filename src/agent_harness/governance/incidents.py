"""Incidents, and the regulatory clocks that start when one is opened.

    incident = gov.incidents.open("personal_data_breach", "high",
                                  "support agent emailed a card number to the wrong customer")
    for d in incident.deadlines:
        print(d.framework, d.notify, d.due_iso)
    # gdpr        supervisory authority   2026-09-27T14:02:11Z   (72 h)
    # ksa-pdpl    SDAIA                   2026-09-27T14:02:11Z   (72 h)
    # dora        competent authority     2026-09-24T18:02:11Z   (4 h initial)

Which clocks apply comes from the active packs; each pack lists its incident
kinds and deadlines. The desk opens incidents on its own for what the runtime
can see (a runaway loop, a denied cross-border transfer, a prompt-injection
signal followed by a sensitive tool call) and calls your `notify` hooks — the
reporting itself stays a human decision.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import new_id

__all__ = ["Deadline", "Incident", "IncidentDesk", "INCIDENT_KINDS"]

Severity = Literal["low", "medium", "high", "critical"]

INCIDENT_KINDS: dict[str, str] = {
    "personal_data_breach": "personal data disclosed, lost or altered without authority",
    "serious_incident": "an AI system caused or nearly caused serious harm (AI Act Art 73)",
    "ict_incident": "an ICT disruption affecting a financial service (DORA)",
    "security_incident": "a cyber-security incident (NIS2, national CERT rules)",
    "policy_violation": "the agent attempted something policy forbids",
    "runaway": "the agent looped, stormed a tool or ran away with spend",
}


@dataclass
class Deadline:
    framework: str
    notify: str
    within_hours: float
    due: float
    basis: str = ""

    @property
    def due_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.due))

    @property
    def overdue(self) -> bool:
        return time.time() > self.due


@dataclass
class Incident:
    kind: str
    severity: Severity
    title: str
    detail: dict[str, Any] = field(default_factory=dict)
    agent: str = ""
    run_id: str = ""
    id: str = field(default_factory=lambda: new_id("inc"))
    opened: float = field(default_factory=time.time)
    status: Literal["open", "reported", "closed"] = "open"
    deadlines: list[Deadline] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "severity": self.severity,
                "title": self.title, "agent": self.agent, "run_id": self.run_id,
                "opened": self.opened, "status": self.status, "detail": self.detail,
                "deadlines": [{"framework": d.framework, "notify": d.notify,
                               "within_hours": d.within_hours, "due": d.due_iso,
                               "basis": d.basis} for d in self.deadlines],
                "notes": self.notes}


class IncidentDesk:
    """Holds incidents and works out their deadlines from the active packs."""

    def __init__(self, clocks: dict[str, dict[str, list[dict[str, Any]]]] | None = None, *,
                 notify: Iterable[Callable[[Incident], Any]] = (),
                 record: Callable[..., Any] | None = None,
                 min_severity: Severity = "medium") -> None:
        #: pack id → incident kind → [{notify, within (hours), basis, min_severity}]
        self.clocks = clocks or {}
        self.notify = list(notify)
        self.record = record
        self.min_severity = min_severity
        self.incidents: list[Incident] = []

    def deadlines_for(self, kind: str, severity: str, opened: float) -> list[Deadline]:
        order = ["low", "medium", "high", "critical"]
        out: list[Deadline] = []
        for framework, kinds in self.clocks.items():
            for clock in kinds.get(kind, []):
                floor = clock.get("min_severity", "medium")
                if order.index(severity) < order.index(floor):
                    continue
                hours = float(clock["within"])
                out.append(Deadline(framework, clock.get("notify", "authority"), hours,
                                    opened + hours * 3600, clock.get("basis", "")))
        return sorted(out, key=lambda d: d.due)

    async def open(self, kind: str, severity: Severity, title: str, *,
                   agent: str = "", run_id: str = "", **detail: Any) -> Incident:
        incident = Incident(kind=kind, severity=severity, title=title, detail=detail,
                            agent=agent, run_id=run_id)
        incident.deadlines = self.deadlines_for(kind, severity, incident.opened)
        self.incidents.append(incident)
        if self.record is not None:
            self.record(incident)
        order = ["low", "medium", "high", "critical"]
        if order.index(severity) >= order.index(self.min_severity):
            for hook in self.notify:
                # A pager that is down must not take the agent down with it.
                try:
                    result = hook(incident)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:
                    incident.notes.append(f"notify failed: {type(exc).__name__}: {exc}")
        return incident

    def open_incidents(self) -> list[Incident]:
        return [i for i in self.incidents if i.status == "open"]

    def overdue(self) -> list[tuple[Incident, Deadline]]:
        return [(i, d) for i in self.open_incidents() for d in i.deadlines if d.overdue]

    def close(self, incident_id: str, note: str = "") -> Incident:
        incident = next(i for i in self.incidents if i.id == incident_id)
        incident.status = "closed"
        if note:
            incident.notes.append(note)
        return incident
