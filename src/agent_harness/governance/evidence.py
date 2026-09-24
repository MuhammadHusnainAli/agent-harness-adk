"""The control catalog, and the report that proves each control is working.

Sixteen controls (C1–C16) are what every pack's requirements reduce to. For
each one the report says whether it is **met**, **partial** or a **gap** on
*this* deployment — worked out from the live configuration and the audit
trail, not from a questionnaire — and gives the evidence and what to do next.

Requirements a machine cannot meet (appoint a DPO, register in the EU
database, get a DPIA signed) are listed as **manual**. The report never claims
compliance; it shows what the technical controls cover and what is left to
people.

    print(gov.report("eu-ai-act").markdown())
    gov.report().json()          # every active pack
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from .core import Governance

__all__ = ["CONTROLS", "Control", "ControlStatus", "ComplianceReport"]

Status = Literal["met", "partial", "gap", "manual"]
_ORDER = {"met": 0, "manual": 1, "partial": 2, "gap": 3}


@dataclass
class ControlStatus:
    id: str
    title: str
    status: Status
    evidence: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Control:
    id: str
    title: str
    description: str
    check: Callable[[Governance], ControlStatus]


def _status(cid: str, ok: int, total: int, evidence: list[str],
            actions: list[str]) -> ControlStatus:
    title = CONTROLS[cid].title if cid in CONTROLS else cid
    if total and ok == total:
        state: Status = "met"
    elif ok:
        state = "partial"
    else:
        state = "gap"
    return ControlStatus(cid, title, state, evidence, actions if state != "met" else [])


def _decisions(gov: Governance, action: str | None = None) -> dict[str, int]:
    out: dict[str, int] = {}
    for (act, effect), n in gov.stats.items():
        if action is None or act == action:
            out[effect] = out.get(effect, 0) + n
    return out


def _c1(gov: Governance) -> ControlStatus:
    ids = gov.identities
    ok = [n for n, i in ids.items() if i.accountable]
    missing = sorted(set(ids) - set(ok))
    return _status("C1", len(ok), len(ids),
                   [f"{len(ids)} agents registered, {len(ok)} with an owner and purpose"],
                   [f"give {', '.join(missing)} an owner and a purpose"] if missing
                   else ["register agents on a governed harness"])


def _c2(gov: Governance) -> ControlStatus:
    ids = gov.identities
    scoped = [n for n, i in ids.items() if i.tools is not None]
    depth = gov.policy.max_delegation_depth is not None or any(
        i.max_delegation_depth is not None or not i.may_delegate for i in ids.values())
    denied = _decisions(gov, "delegate").get("deny", 0)
    evidence = [f"{len(scoped)}/{len(ids)} agents have a tool allowlist",
                f"delegation depth limited: {'yes' if depth else 'no'}",
                f"{denied} delegations refused"]
    actions = []
    if len(scoped) < len(ids):
        actions.append("declare `tools=` on every agent identity")
    if not depth:
        actions.append("set `max_delegation_depth` (the owasp-agentic pack sets 3)")
    return _status("C2", len(scoped) + int(depth), len(ids) + 1, evidence, actions)


def _c3(gov: Governance) -> ControlStatus:
    total = sum(gov.stats.values())
    rules = len(gov.policy.rules)
    return _status("C3", int(rules > 0) + int(total > 0), 2,
                   [f"policy {gov.policy.name}@{gov.policy.version} "
                    f"sha256:{gov.policy.hash[:16]}", f"{rules} rules",
                    f"{total} decisions recorded"],
                   ["add rules or packs" if not rules else "run agents to record decisions"])


def _c4(gov: Governance) -> ControlStatus:
    seen = ", ".join(f"{k}×{v}" for k, v in sorted(gov.data_seen.items())) or "none yet"
    return ControlStatus("C4", CONTROLS["C4"].title, "met",
                         [f"deterministic classifier on every egress, tool call and "
                          f"memory write; classes observed: {seen}"])


def _c5(gov: Governance) -> ControlStatus:
    rules = [r.id for r in gov.policy.rules if r.effect == "redact" or r.redact]
    purposes = bool(gov.policy.purposes)
    return _status("C5", int(bool(rules)) + int(purposes), 2,
                   [f"{len(rules)} redaction rules", f"purpose-based minimisation: "
                    f"{'on' if purposes else 'off'}",
                    "credentials are always stripped from memory"],
                   ["add a redact rule for egress or declare `purposes:` with the data "
                    "classes each purpose may use"])


def _c6(gov: Governance) -> ControlStatus:
    residency = gov.policy.residency
    if residency is None or not residency.transfers:
        return ControlStatus("C6", CONTROLS["C6"].title, "gap",
                             ["no residency limits configured"],
                             ["add a jurisdiction pack (gdpr, ksa-pdpl, ...) or `residency:`"])
    transfers = [e for e in gov.harness.audit.entries
                 if e.action == "governance.transfer"] if gov.harness else []
    egress = _decisions(gov, "egress")
    evidence = [f"home: {residency.home or '(per principal)'}",
                "limits: " + "; ".join(f"{o} → {', '.join(d)}"
                                       for o, d in sorted(residency.transfers.items())),
                f"unknown regions: {residency.unknown}",
                f"egress decisions: {egress}",
                f"{len(transfers)} cross-border transfers recorded"]
    restricted = [o for o, d in residency.transfers.items() if "*" not in d]
    if residency.home is None and len(restricted) > 1:
        return ControlStatus("C6", CONTROLS["C6"].title, "partial", evidence,
                             ["set `home=` — a principal with no `jurisdiction` tag is "
                              f"otherwise unrestricted ({', '.join(sorted(restricted))} "
                              "all restrict transfers)"])
    unknown = [a["name"] for a in gov.inventory.agents.values() if a["region"] == "unknown"]
    if unknown and residency.unknown == "allow":
        return ControlStatus("C6", CONTROLS["C6"].title, "partial", evidence,
                             [f"declare regions for {', '.join(unknown)} (`regions=`)"])
    return ControlStatus("C6", CONTROLS["C6"].title, "met", evidence)


def _c7(gov: Governance) -> ControlStatus:
    purposes = gov.policy.purposes
    ids = gov.identities
    ok = [n for n, i in ids.items() if i.purpose and (not purposes or i.purpose in purposes)]
    return _status("C7", len(ok) + int(bool(purposes)), len(ids) + 1,
                   [f"declared purposes: {', '.join(sorted(purposes)) or 'none'}",
                    f"{len(ok)}/{len(ids)} agents run under a declared purpose"],
                   ["declare `purposes:` in the policy and give every agent one of them"])


def _c8(gov: Governance) -> ControlStatus:
    rules = [r.id for r in gov.policy.rules if r.effect == "require_approval"]
    history = gov.oversight.history()
    high = [n for n, i in gov.identities.items() if i.risk == "high"]
    evidence = [f"{len(rules)} approval rules", f"{len(history)} approval requests, "
                f"{sum(1 for h in history if h['status'] == 'approved')} approved",
                "fail-closed timeouts, quorum of distinct approvers, separation of duties",
                "stop control: harness.stop() halts every run"]
    if not rules:
        return ControlStatus("C8", CONTROLS["C8"].title, "gap" if high else "partial",
                             evidence, ["add `require_approval` rules for consequential "
                                        "or irreversible tools"])
    return ControlStatus("C8", CONTROLS["C8"].title, "met", evidence)


def _c9(gov: Governance) -> ControlStatus:
    t = gov.policy.transparency
    return _status("C9", int(t.disclose) + int(t.provenance), 2,
                   [f"disclosure: {'on' if t.disclose else 'off'} ({', '.join(t.languages)})",
                    f"provenance manifests: {'on' if t.provenance else 'off'}"
                    f"{' (signed)' if t.provenance and gov.signer else ''}",
                    f"visible labels: {'on' if t.label_output else 'off'}"],
                   ["enable transparency.disclose and transparency.provenance "
                    "(the eu-ai-act pack does)"])


def _c10(gov: Governance) -> ControlStatus:
    audit = gov.harness.audit if gov.harness else None
    if audit is None:
        return ControlStatus("C10", CONTROLS["C10"].title, "gap", ["not attached"],
                             ["attach governance to a harness"])
    ok, why = audit.verify()
    checks = [ok, audit.signer is not None, audit.path is not None]
    evidence = [f"{len(audit)} entries, chain: {why}",
                f"signed: {'yes (' + audit.signer.key_id + ')' if audit.signer else 'no'}",
                f"persisted: {audit.path or 'no (in memory)'}",
                f"minimum retention: {gov.policy.log_retention_days or 'not set'} days"]
    actions = []
    if not ok:
        actions.append(f"the audit chain is broken — {why}; investigate before anything else")
    if audit.signer is None:
        actions.append("pass `signing_key=` (HMAC) or an Ed25519Signer")
    if audit.path is None:
        actions.append("persist the audit trail (Harness.local does)")
    status = _status("C10", sum(checks), 3, evidence, actions)
    if not ok:
        status.status = "gap"
    return status


def _c11(gov: Governance) -> ControlStatus:
    durable = gov.vault.path is not None
    return _status("C11", 1 + int(durable), 2,
                   ["access and erasure across memory, sessions and the audit trail "
                    "(crypto-shredding)", f"subjects indexed: {len(gov.vault.subjects)}",
                    f"subject index persisted: {'yes' if durable else 'no'}"],
                   ["give the vault a path so the subject index survives restarts"])


def _c12(gov: Governance) -> ControlStatus:
    tiers: dict[str, int] = {}
    for a in gov.assessments.values():
        tiers[a.tier] = tiers.get(a.tier, 0) + 1
    unowned_high = [n for n, i in gov.identities.items()
                    if i.risk == "high" and not i.accountable]
    return _status("C12", len(gov.assessments) - len(unowned_high), len(gov.assessments),
                   [f"tiers: {tiers or 'no agents'}",
                    "prohibited uses are refused at registration"],
                   [f"high-risk agents without an accountable owner: "
                    f"{', '.join(unowned_high)}" if unowned_high else
                    "register agents to classify them"])


def _c13(gov: Governance) -> ControlStatus:
    inv = gov.inventory
    drift = inv.drift() if inv.pins else None
    changed = bool(drift and any(drift.values()))
    evidence = [f"{len(inv.agents)} agents, {len(inv.fingerprints())} tools fingerprinted",
                f"pinned: {'yes' if inv.pins else 'no'}", f"drift: {drift or 'n/a'}",
                f"third parties: {len(inv.third_parties())}"]
    actions = []
    if not inv.pins:
        actions.append("review the inventory and call `inventory.pin()`")
    if changed:
        actions.append("review the drifted tools, then re-pin")
    return _status("C13", int(bool(inv.agents)) + int(bool(inv.pins) and not changed), 2,
                   evidence, actions)


def _c14(gov: Governance) -> ControlStatus:
    m = gov.monitor
    return ControlStatus("C14", CONTROLS["C14"].title, "met", [
        f"loop detection at {m.max_identical_calls} identical calls, "
        f"{m.max_calls_per_tool or '∞'} calls per tool, "
        f"{m.max_delegations or '∞'} delegations per run",
        "prompt-injection signals in tool results mark the run untrusted",
        "health and tracing: harness.health, harness.tracer"])


def _c15(gov: Governance) -> ControlStatus:
    clocks = sum(len(v) for p in gov.packs for v in p.incidents.values())
    hooked = bool(gov.incidents.notify)
    return _status("C15", int(clocks > 0) + int(hooked), 2,
                   [f"{clocks} regulatory clocks from active packs",
                    f"notification hooks: {len(gov.incidents.notify)}",
                    f"open incidents: {len(gov.incidents.open_incidents())}, "
                    f"overdue: {len(gov.incidents.overdue())}"],
                   ["pass `notify=[...]` so a person hears about incidents"
                    if not hooked else "activate packs with incident clocks"])


def _c16(gov: Governance) -> ControlStatus:
    return ControlStatus("C16", CONTROLS["C16"].title, "met",
                         ["this report, generated from live configuration and the audit trail"])


CONTROLS: dict[str, Control] = {c.id: c for c in (
    Control("C1", "Agent identity & accountability",
            "every agent has an owner, a purpose and a risk tier", _c1),
    Control("C2", "Least privilege & delegation",
            "sub-agents act under a narrowed copy of their parent's authority", _c2),
    Control("C3", "Policy as code",
            "versioned, hashed policy; every decision names the policy that made it", _c3),
    Control("C4", "Data classification",
            "personal, special-category, credential and national-ID data detected", _c4),
    Control("C5", "Minimisation & pseudonymisation",
            "data is redacted or tokenised before it leaves", _c5),
    Control("C6", "Residency & cross-border transfer",
            "each model call's real region is checked against transfer limits", _c6),
    Control("C7", "Purpose limitation",
            "a run carries a purpose and may use only the data it allows", _c7),
    Control("C8", "Human oversight",
            "approvals with quorum and fail-closed timeouts; a stop switch", _c8),
    Control("C9", "Transparency & provenance",
            "AI disclosure and machine-readable marking of output", _c9),
    Control("C10", "Tamper-evident records",
            "signed, hash-chained, retained decision records", _c10),
    Control("C11", "Data-subject rights",
            "access and erasure without breaking the audit trail", _c11),
    Control("C12", "Risk classification",
            "per-framework risk tiers; prohibited uses refused", _c12),
    Control("C13", "AI inventory & supply chain",
            "models, providers, tools and MCP servers recorded and pinned", _c13),
    Control("C14", "Runtime monitoring",
            "loops, storms, cascades and injection signals detected", _c14),
    Control("C15", "Incident management",
            "incidents opened with every applicable regulatory deadline", _c15),
    Control("C16", "Evidence & reporting",
            "control status and evidence, per framework, on demand", _c16),
)}


@dataclass
class RequirementStatus:
    pack: str
    ref: str
    title: str
    status: Status
    controls: list[str]
    note: str = ""


@dataclass
class ComplianceReport:
    generated: float
    policy: str
    mode: str
    controls: dict[str, ControlStatus]
    requirements: list[RequirementStatus]
    packs: list[dict[str, Any]]

    @classmethod
    def build(cls, gov: Governance, pack: str | None = None) -> ComplianceReport:
        packs = [p for p in gov.packs if pack is None or p.id == pack]
        if pack is not None and not packs:
            from .packs import load_pack

            packs = [load_pack(pack)]     # report against a pack that is not active
        controls = {cid: control.check(gov) for cid, control in CONTROLS.items()}
        requirements: list[RequirementStatus] = []
        for p in packs:
            for r in p.requirements:
                if r.manual or not r.controls:
                    state: Status = "manual"
                else:
                    state = max((controls[c].status for c in r.controls if c in controls),
                                key=_ORDER.__getitem__, default="gap")
                requirements.append(RequirementStatus(p.id, r.ref, r.title, state,
                                                      r.controls, r.note))
        return cls(time.time(), f"{gov.policy.name}@{gov.policy.version} "
                   f"sha256:{gov.policy.hash[:16]}", gov.mode, controls, requirements,
                   [{"id": p.id, "name": p.name, "jurisdiction": p.jurisdiction,
                     "binding": p.binding, "status": p.status, "as_of": p.as_of,
                     "active": p in gov.packs, "sources": p.sources} for p in packs])

    # ---- views ------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        out = {"met": 0, "partial": 0, "gap": 0, "manual": 0}
        for r in self.requirements:
            out[r.status] += 1
        return out

    def gaps(self) -> list[RequirementStatus]:
        return [r for r in self.requirements if r.status in ("gap", "partial")]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.generated)),
            "policy": self.policy, "mode": self.mode, "packs": self.packs,
            "summary": self.counts(),
            "controls": {k: {"title": v.title, "status": v.status, "evidence": v.evidence,
                             "actions": v.actions} for k, v in self.controls.items()},
            "requirements": [r.__dict__ for r in self.requirements],
            "disclaimer": DISCLAIMER,
        }

    def json(self, **kw: Any) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False, **kw)

    def markdown(self) -> str:
        lines = ["# Governance evidence report", "",
                 f"Policy `{self.policy}` · mode **{self.mode}** · generated "
                 f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.generated))}", ""]
        counts = self.counts()
        lines += [f"**{counts['met']} met · {counts['partial']} partial · {counts['gap']} gap"
                  f" · {counts['manual']} manual**", ""]
        for p in self.packs:
            reqs = [r for r in self.requirements if r.pack == p["id"]]
            lines += [f"## {p['name']} (`{p['id']}`)", "",
                      f"{p['jurisdiction']} · {'binding' if p['binding'] else 'voluntary'}"
                      f" · {p['status']} · checked {p['as_of']}", "",
                      "| Ref | Requirement | Controls | Status |", "|---|---|---|---|"]
            lines += [f"| {r.ref} | {r.title} | {', '.join(r.controls) or '—'} | "
                      f"{_badge(r.status)} |" for r in reqs]
            lines.append("")
        lines += ["## Controls", ""]
        for c in self.controls.values():
            lines.append(f"### {c.id} {c.title} — {_badge(c.status)}")
            lines += [f"- {e}" for e in c.evidence]
            lines += [f"- **Next:** {a}" for a in c.actions]
            lines.append("")
        lines += ["---", f"_{DISCLAIMER}_", ""]
        return "\n".join(lines)

    def render(self) -> str:
        """The Markdown report — what a terminal or a pull request shows best."""
        return self.markdown()

    def html(self) -> str:
        body = []
        for p in self.packs:
            rows = "".join(
                f"<tr><td>{html.escape(r.ref)}</td><td>{html.escape(r.title)}</td>"
                f"<td>{html.escape(', '.join(r.controls))}</td>"
                f"<td class='{r.status}'>{r.status}</td></tr>"
                for r in self.requirements if r.pack == p["id"])
            body.append(f"<h2>{html.escape(p['name'])}</h2><table><tr><th>Ref</th>"
                        f"<th>Requirement</th><th>Controls</th><th>Status</th></tr>"
                        f"{rows}</table>")
        controls = "".join(
            f"<h3>{c.id} {html.escape(c.title)} <span class='{c.status}'>{c.status}</span>"
            f"</h3><ul>{''.join(f'<li>{html.escape(e)}</li>' for e in c.evidence)}"
            f"{''.join(f'<li><b>Next:</b> {html.escape(a)}</li>' for a in c.actions)}</ul>"
            for c in self.controls.values())
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Governance report</title><style>"
            ":root{color-scheme:light dark;--fg:#1b1b1f;--bg:#fff;--line:#d7d7de}"
            "@media (prefers-color-scheme:dark){:root{--fg:#e8e8ee;--bg:#15151a;--line:#33333d}}"
            "body{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:960px;"
            "padding:16px;color:var(--fg);background:var(--bg)}"
            "table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}"
            "td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}"
            ".met{color:#1a7f37}.partial{color:#9a6700}.gap{color:#cf222e}"
            ".manual{color:#6e7781}</style></head><body>"
            f"<h1>Governance evidence report</h1><p>Policy <code>{html.escape(self.policy)}"
            f"</code> · mode {self.mode}</p>{''.join(body)}<h2>Controls</h2>{controls}"
            f"<p><em>{html.escape(DISCLAIMER)}</em></p></body></html>")


def _badge(status: str) -> str:
    return {"met": "✅ met", "partial": "🟡 partial", "gap": "❌ gap",
            "manual": "📝 manual"}[status]


DISCLAIMER = ("This report shows which technical controls are in place and the evidence "
              "for them. It is not legal advice and does not by itself establish "
              "compliance with any law; organisational measures and legal review are "
              "still required.")
