"""Audit trail: an immutable who-did-what record. What compliance will ask for.

Every entry carries the hash of the one before it, so a tampered or deleted
record breaks the chain and `verify()` says exactly where.

A hash chain proves order and integrity, but anyone able to rewrite the file
can rebuild the chain. Give the trail a `signer` (see
`agent_harness.governance.records`) and every entry's hash is also signed, so a
rebuilt chain fails verification without the key.

This is deliberately separate from the run journal. The journal is for debugging
and can be summarised, trimmed or thrown away; the audit trail is append-only
and answers "who did what, when, and was it allowed".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["AuditEntry", "AuditTrail"]

GENESIS = "0" * 64


class AuditEntry(BaseModel):
    """One immutable record. `hash` covers every field above it plus `prev_hash`."""

    seq: int = 0
    ts: float = Field(default_factory=time.time)
    actor: str = ""          # which agent, or "human"
    action: str = ""         # tool_call | permission | run_start | run_end | stop | ...
    target: str = ""         # the tool, sub-agent or resource acted on
    decision: str = ""       # allow | deny | ask | ok | error
    run_id: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""
    #: A signature over `hash`, and which key made it. Outside the hashed
    #: payload, so an unsigned trail and a signed one chain identically.
    signature: str = ""
    key_id: str = ""

    def payload(self) -> str:
        """The exact bytes the hash covers. Sorted, so it is reproducible."""
        return json.dumps(
            {
                "seq": self.seq, "ts": self.ts, "actor": self.actor,
                "action": self.action, "target": self.target,
                "decision": self.decision, "run_id": self.run_id,
                "detail": self.detail, "prev_hash": self.prev_hash,
            },
            sort_keys=True, default=str, separators=(",", ":"),
        )

    def compute_hash(self) -> str:
        return hashlib.sha256(self.payload().encode()).hexdigest()

    def line(self) -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts))
        target = f" {self.target}" if self.target else ""
        return f"[{stamp}] {self.actor} {self.action}{target} → {self.decision}"


class AuditTrail:
    """Append-only, hash-chained. In memory by default; give it a path to persist."""

    def __init__(self, path: str | Path | None = None, *, redact: bool = True,
                 signer: Any = None) -> None:
        self.path = Path(path) if path else None
        self.redact = redact
        #: Anything with ``key_id``, ``sign(bytes) -> str`` and
        #: ``verify(bytes, str) -> bool``.
        self.signer = signer
        self.entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.entries = self._load(self.path)

    # ---- writing ---------------------------------------------------------
    def record(self, actor: str, action: str, *, target: str = "",
               decision: str = "ok", run_id: str = "", **detail: Any) -> AuditEntry:
        """Append one record. Synchronous on purpose — an audit write never waits."""
        entry = AuditEntry(
            seq=len(self.entries) + 1,
            actor=actor, action=action, target=target, decision=decision,
            run_id=run_id, detail=self._clean(detail),
            prev_hash=self.entries[-1].hash if self.entries else GENESIS,
        )
        entry.hash = entry.compute_hash()
        if self.signer is not None:
            entry.signature = self.signer.sign(entry.hash.encode())
            entry.key_id = self.signer.key_id
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        return entry

    def _clean(self, detail: dict[str, Any]) -> dict[str, Any]:
        """Keep the record small and free of anything that should not be kept."""
        if not self.redact:
            return detail
        out: dict[str, Any] = {}
        for key, value in detail.items():
            if any(word in key.lower() for word in
                   ("key", "token", "secret", "password", "authorization")):
                out[key] = "[redacted]"
            elif isinstance(value, str) and len(value) > 500:
                out[key] = value[:500] + f"... [{len(value)} chars]"
            else:
                out[key] = value
        return out

    # ---- reading and proving ---------------------------------------------
    def verify(self) -> tuple[bool, str]:
        """Walk the chain. Returns (ok, reason) and names the first bad entry."""
        previous = GENESIS
        for index, entry in enumerate(self.entries):
            if entry.seq != index + 1:
                return False, f"entry {index + 1}: sequence is {entry.seq}"
            if entry.prev_hash != previous:
                return False, f"entry {entry.seq}: does not follow the previous entry"
            if entry.hash != entry.compute_hash():
                return False, f"entry {entry.seq}: contents do not match its hash"
            if self.signer is not None:
                if not entry.signature:
                    return False, f"entry {entry.seq}: is not signed"
                if not self.signer.verify(entry.hash.encode(), entry.signature):
                    return False, f"entry {entry.seq}: signature does not verify"
            previous = entry.hash
        return True, "chain intact"

    @staticmethod
    def _load(path: Path) -> list[AuditEntry]:
        rows: list[AuditEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(AuditEntry(**json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
        return rows

    @classmethod
    def load(cls, path: str | Path, *, signer: Any = None) -> AuditTrail:
        return cls(path, signer=signer)

    def for_run(self, run_id: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.run_id == run_id]

    def by_actor(self, actor: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.actor == actor]

    def denials(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.decision == "deny"]

    def render(self, limit: int | None = None) -> str:
        rows = self.entries[-limit:] if limit else self.entries
        return "\n".join(e.line() for e in rows)

    def export(self) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)
