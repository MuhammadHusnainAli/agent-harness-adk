"""Data-subject requests: access and erasure, across everything the harness keeps.

    export = await gov.rights.access(Trace(tenant_id="acme", user_id="alice"))
    receipt = await gov.rights.erase(Trace(tenant_id="acme", user_id="alice"))

Erasure covers the memory store (records and `user.md`), every session the
person's runs were saved in, pseudonym tokens held in memory, and the audit
trail — by crypto-shredding: the subject's key is destroyed, so the tokens
standing in for their data in the trail can never be linked back to them,
while the chain itself still verifies. The receipt says what was removed and
is signed when governance has a signer. The request and its outcome are
themselves recorded (without the personal data).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..memory.trace import Trace

if TYPE_CHECKING:  # pragma: no cover
    from .core import Governance

__all__ = ["SubjectRights"]


def _subject(trace: Trace | str | dict[str, Any]) -> tuple[Trace, str]:
    trace = Trace.of(trace)
    if not trace.user_id:
        raise ValueError("a data-subject request needs a user_id on the trace")
    return trace, f"{trace.tenant_id or ''}/{trace.user_id}"


class SubjectRights:
    def __init__(self, gov: Governance) -> None:
        self.gov = gov

    @property
    def _harness(self) -> Any:
        if self.gov.harness is None:
            raise RuntimeError("attach governance to a harness first")
        return self.gov.harness

    async def access(self, trace: Trace | str | dict[str, Any]) -> dict[str, Any]:
        """Everything held about one person, as a JSON-ready dict (GDPR Art 15)."""
        trace, subject = _subject(trace)
        harness = self._harness
        store = harness.memory_store
        user_trace = trace.model_copy(update={"scope": "user"})
        records = await store.all(trace=user_trace)
        sessions = []
        for session_id in self.gov.vault.locations(subject).get("sessions", []):
            try:
                session = await harness.sessions.load(session_id)
            except Exception:
                continue
            sessions.append({"id": session.id, "agent": session.agent,
                             "created": session.created,
                             "messages": [{"role": m.role, "text": m.text}
                                          for m in session.messages]})
        ref = self.gov.vault.subject_ref(subject)
        decisions = [e.model_dump(mode="json") for e in harness.audit.entries
                     if e.detail.get("subject") == ref] if ref else []
        harness.audit.record("governance", "governance.subject_access", decision="ok",
                             subject=ref, records=len(records), sessions=len(sessions))
        return {
            "subject": {"tenant": trace.tenant_id, "user": trace.user_id},
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": await store.read_doc("user.md", trace=user_trace),
            "memory": [r.model_dump(mode="json") for r in records],
            "sessions": sessions,
            "decisions": decisions,
        }

    async def erase(self, trace: Trace | str | dict[str, Any], *,
                    requested_by: str = "data subject") -> dict[str, Any]:
        """Erase one person everywhere (GDPR Art 17, DPDP s12, PDPL equivalents)."""
        trace, subject = _subject(trace)
        harness = self._harness
        store = harness.memory_store
        user_trace = trace.model_copy(update={"scope": "user"})
        ref = self.gov.vault.subject_ref(subject)

        records = len(await store.all(trace=user_trace))
        await store.clear(trace=user_trace)
        await store.write_doc("user.md", "", trace=user_trace)

        deleted_sessions = []
        for session_id in self.gov.vault.locations(subject).get("sessions", []):
            try:
                await harness.sessions.delete(session_id)
                deleted_sessions.append(session_id)
            except Exception:
                continue
        tokens = self.gov.pseudonymizer.forget(subject)
        shredded = self.gov.vault.erase(subject)

        receipt: dict[str, Any] = {
            "subject": ref, "erased_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requested_by": requested_by, "memory_records": records,
            "sessions": deleted_sessions, "pseudonyms_forgotten": tokens,
            "audit_key_destroyed": shredded["key_destroyed"],
            "audit_chain": harness.audit.verify()[1],
        }
        if self.gov.signer is not None:
            import json

            payload = json.dumps(receipt, sort_keys=True).encode()
            receipt["signature"] = self.gov.signer.sign(payload)
            receipt["key_id"] = self.gov.signer.key_id
        harness.audit.record("governance", "governance.erasure", decision="ok",
                             subject=ref, requested_by=requested_by,
                             memory_records=records, sessions=len(deleted_sessions))
        return receipt
