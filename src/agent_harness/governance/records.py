"""Records that hold up: signatures, per-subject keys, retention.

**Signing.** `AuditTrail` already hash-chains its entries; a signer makes the
chain unforgeable by anyone without the key.

    HMACSigner(os.environ["AUDIT_KEY"])         # stdlib, shared secret
    Ed25519Signer.generate()                     # public-key: auditors verify
                                                 # with the public key alone
                                                 # (needs `cryptography`)

**Crypto-shredding.** The audit trail must be immutable; people have a right
to erasure. Both hold if personal data never enters the trail in the clear:
values are replaced by tokens keyed per data subject, and erasing someone
destroys their key. The chain still verifies — every byte it covers is
unchanged — but nothing in it can be linked back to that person any more.

**Retention.** Decision records are kept at least as long as the strictest
pack asks (EU AI Act deployers: six months). Conversations are kept no longer
than the shortest data-retention limit. `RetentionSweeper` enforces the second
and reports on the first.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

__all__ = ["HMACSigner", "Ed25519Signer", "SubjectVault", "ErasureReceipt",
           "RetentionSweeper"]


class HMACSigner:
    """HMAC-SHA256. Anyone who can verify can also sign — keep the key close."""

    algorithm = "hmac-sha256"

    def __init__(self, key: bytes | str, *, key_id: str | None = None) -> None:
        if isinstance(key, str):
            key = key.encode()
        if len(key) < 16:
            raise ConfigurationError("an audit signing key needs at least 16 bytes")
        self._key = key
        self.key_id = key_id or "hmac:" + hashlib.sha256(key).hexdigest()[:12]

    def sign(self, data: bytes) -> str:
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(data), signature)


class Ed25519Signer:
    """Ed25519 via the optional `cryptography` package.

    `pip install agent-harness-adk[governance]`. Hand auditors
    `public_key_pem()`; `Ed25519Signer.verifier(pem)` checks a trail with it
    and cannot sign anything.
    """

    algorithm = "ed25519"

    def __init__(self, private_key: Any = None, *, public_key: Any = None) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError:  # pragma: no cover - exercised only without the extra
            raise ConfigurationError(
                "Ed25519 signing needs the `cryptography` package: "
                "pip install agent-harness-adk[governance]") from None
        self._ser = serialization
        if isinstance(private_key, (bytes, str)) and private_key:
            data = private_key.encode() if isinstance(private_key, str) else private_key
            private_key = serialization.load_pem_private_key(data, password=None)
        if isinstance(public_key, (bytes, str)) and public_key:
            data = public_key.encode() if isinstance(public_key, str) else public_key
            public_key = serialization.load_pem_public_key(data)
        if private_key is None and public_key is None:
            private_key = ed25519.Ed25519PrivateKey.generate()
        self._private = private_key
        self._public = public_key or private_key.public_key()
        raw = self._public.public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
        self.key_id = "ed25519:" + hashlib.sha256(raw).hexdigest()[:12]

    @classmethod
    def generate(cls) -> Ed25519Signer:
        return cls()

    @classmethod
    def verifier(cls, public_key_pem: bytes | str) -> Ed25519Signer:
        return cls(public_key=public_key_pem)

    def sign(self, data: bytes) -> str:
        if self._private is None:
            raise ConfigurationError("this Ed25519Signer holds only a public key")
        return base64.b64encode(self._private.sign(data)).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        try:
            self._public.verify(base64.b64decode(signature), data)
        except Exception:
            return False
        return True

    def public_key_pem(self) -> str:
        return self._public.public_bytes(
            self._ser.Encoding.PEM, self._ser.PublicFormat.SubjectPublicKeyInfo).decode()

    def private_key_pem(self) -> str:
        if self._private is None:
            raise ConfigurationError("this Ed25519Signer holds only a public key")
        return self._private.private_bytes(
            self._ser.Encoding.PEM, self._ser.PrivateFormat.PKCS8,
            self._ser.NoEncryption()).decode()


class ErasureReceipt(dict):
    """What was erased, when, for whom — signed when a signer is available."""


class SubjectVault:
    """Per-subject keys and the index of where each subject's data lives.

    Kept apart from the audit trail on purpose: the trail is immutable, this is
    not. A file path makes it durable; without one it lives in memory.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._keys: dict[str, str] = {}
        self._index: dict[str, dict[str, list[str]]] = {}
        self._erased: dict[str, float] = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._keys = data.get("keys", {})
            self._index = data.get("index", {})
            self._erased = data.get("erased", {})

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"keys": self._keys, "index": self._index,
                                   "erased": self._erased}), encoding="utf-8")
        tmp.replace(self.path)

    def _key(self, subject: str) -> bytes:
        if subject not in self._keys:
            self._keys[subject] = secrets.token_hex(32)
            self._save()
        return bytes.fromhex(self._keys[subject])

    def token(self, subject: str, value: str, kind: str = "value") -> str:
        """A stable, unlinkable-after-erasure stand-in for one personal value."""
        if subject in self._erased:
            return f"⟨{kind}:erased⟩"
        digest = hmac.new(self._key(subject), value.encode(), hashlib.sha256).hexdigest()
        return f"⟨{kind}:{digest[:12]}⟩"

    def subject_ref(self, subject: str) -> str:
        """How a subject is named in records: a keyed hash, never the id itself."""
        if not subject:
            return ""
        return "subj:" + hmac.new(self._key(subject), b"subject-ref",
                                  hashlib.sha256).hexdigest()[:16]

    def note(self, subject: str, kind: str, ref: str) -> None:
        """Remember that `subject` has data at `ref` (a session id, a document)."""
        if not subject or subject in self._erased:
            return
        refs = self._index.setdefault(subject, {}).setdefault(kind, [])
        if ref not in refs:
            refs.append(ref)
            self._save()

    def locations(self, subject: str) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._index.get(subject, {}).items()}

    def erase(self, subject: str) -> dict[str, Any]:
        """Destroy the subject's key and index. Returns what was there."""
        had_key = self._keys.pop(subject, None) is not None
        locations = self._index.pop(subject, {})
        self._erased[subject] = time.time()
        self._save()
        return {"key_destroyed": had_key, "locations": locations}

    def is_erased(self, subject: str) -> bool:
        return subject in self._erased

    @property
    def subjects(self) -> list[str]:
        return sorted(set(self._keys) | set(self._index))


class RetentionSweeper:
    """Deletes conversations older than the data-retention limit."""

    def __init__(self, days: int | None) -> None:
        self.days = days

    def cutoff(self, now: float | None = None) -> float | None:
        if self.days is None:
            return None
        return (now or time.time()) - self.days * 86400

    async def sweep(self, harness: Any, *, now: float | None = None) -> dict[str, Any]:
        cutoff = self.cutoff(now)
        report: dict[str, Any] = {"days": self.days, "sessions_deleted": []}
        if cutoff is None:
            return report
        sessions = await harness.sessions.list(limit=100_000)
        for session in sessions:
            if session.updated < cutoff:
                await harness.sessions.delete(session.id)
                report["sessions_deleted"].append(session.id)
        if harness.audit is not None:
            harness.audit.record("governance", "retention_sweep", decision="ok",
                                 deleted=len(report["sessions_deleted"]),
                                 days=self.days)
        return report
