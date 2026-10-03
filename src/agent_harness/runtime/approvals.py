"""Approvals that outlive the process: a run paused for a person, kept in a database.

    harness = Harness(sessions="postgresql://…", approvals=True)
    agent = Agent("support", tools=[refund], harness=harness)     # refund asks first

    result = await agent.run("Refund order 4182.")
    result.stop_reason            # "approval"
    result.approval.id            # "apr_9f2c…" — and the run is in the database

Ten hours later, in another process, on another machine:

    await harness.approvals.approve("apr_9f2c…", by="maria", note="checked the order")
    result = await agent.resume_approval("apr_9f2c…")     # carries on from that step

When a tool needs approval and nobody is there to give it, the run does not
wait and does not fail. It stops at that step and is written down: the
conversation so far, the call that is waiting with its exact arguments, and
what the other tools of the same step returned. Nothing has to stay running.

When the answer comes, the run is picked up from the record. The approved call
is executed with the arguments that were approved — the model is not asked
again, so it cannot ask for something else — and the tools that had already run
are not run a second time. A call that was declined is answered with the
reviewer's reason, and the agent carries on from there.

A record is resumed once. The claim is taken before the tool runs, in a write
the store refuses to a second taker, so two workers that both see an approval
cannot both make the refund.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import ApprovalError, ConfigurationError, SessionConflict
from ..types import Message, Usage, new_id
from .session import Session, SessionStore

__all__ = [
    "Approval",
    "ApprovalCall",
    "ApprovalStore",
    "Approvals",
    "MemoryApprovalStore",
    "FileApprovalStore",
    "SessionApprovalStore",
]

ApprovalStatus = Literal["pending", "approved", "denied", "expired", "resuming",
                         "resumed", "failed"]


class ApprovalCall(BaseModel):
    """One tool call waiting for a person."""

    #: The id the model gave the call. What `approve(..., call=…)` names.
    id: str
    tool: str
    #: Exactly what will be run. An approval is for these arguments and no others.
    args: dict[str, Any] = Field(default_factory=dict)
    #: Why it needs asking.
    reason: str = ""
    status: Literal["pending", "approved", "denied"] = "pending"
    by: str = ""
    note: str = ""
    decided: float | None = None

    def describe(self) -> str:
        args = ", ".join(f"{k}={json.dumps(v, default=str, ensure_ascii=False)}"
                         for k, v in self.args.items())
        return f"{self.tool}({args})"


class Approval(BaseModel):
    """A run that stopped to ask, and everything needed to carry it on."""

    id: str = Field(default_factory=lambda: new_id("apr"))
    run_id: str = ""
    agent: str = ""
    #: Whose run it is. Listing and resuming are held to it.
    user_id: str | None = None
    tenant_id: str | None = None
    #: The conversation it belongs to, and the version it was at when it paused.
    session_id: str = ""
    session_version: int = 0
    task: str = ""
    model: str = ""
    #: The step it stopped in. Resuming finishes this step, then goes on.
    step: int = 0
    status: ApprovalStatus = "pending"
    calls: list[ApprovalCall] = Field(default_factory=list)
    #: What the other tools of that step returned. They are not run again.
    done: list[dict[str, Any]] = Field(default_factory=list)
    #: The conversation as the model had it, ending with its request for the tools.
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    #: A mode's todo list and source ledger, when the run kept them.
    notebook: dict[str, Any] | None = None
    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)
    #: After this, an unanswered request counts as declined. None: never.
    expires: float | None = None
    #: When a worker took it to resume, and which.
    claimed: float | None = None
    claimed_by: str = ""
    #: How the resumed run ended: its stop reason, the start of its answer, and
    #: the next approval's id if it stopped to ask again.
    outcome: dict[str, Any] = Field(default_factory=dict)
    version: int = 0

    @property
    def waiting(self) -> list[ApprovalCall]:
        return [c for c in self.calls if c.status == "pending"]

    @property
    def expired(self) -> bool:
        return (self.status == "pending" and self.expires is not None
                and time.time() >= self.expires)

    def owned_by(self, user_id: str | None, tenant_id: str | None) -> bool:
        """The same rule as a session: an axis nobody set is not a wall."""
        return ((not self.user_id or not user_id or self.user_id == user_id)
                and (not self.tenant_id or not tenant_id
                     or self.tenant_id == tenant_id))

    def describe(self) -> str:
        """One line for a person deciding: who wants to do what."""
        what = "; ".join(c.describe() for c in self.calls)
        return f"{self.agent} wants to run {what}"

    def summary(self) -> dict[str, Any]:
        """Everything but the conversation — what a list of approvals shows."""
        return {
            "id": self.id, "status": "expired" if self.expired else self.status,
            "agent": self.agent, "run_id": self.run_id, "session_id": self.session_id,
            "user_id": self.user_id, "tenant_id": self.tenant_id, "task": self.task[:200],
            "step": self.step, "created": self.created, "expires": self.expires,
            "calls": [c.model_dump(mode="json") for c in self.calls],
            "outcome": self.outcome}

    def _settle(self) -> None:
        """The request's status, from its calls'."""
        if self.status != "pending" or self.waiting:
            return
        self.status = ("approved" if any(c.status == "approved" for c in self.calls)
                       else "denied")


# ----------------------------------------------------------------------
# stores
# ----------------------------------------------------------------------
class ApprovalStore(ABC):
    """Where paused runs are kept. `save` must refuse a stale write — that
    refusal is what makes a run resume once and a decision count once."""

    @abstractmethod
    async def load(self, approval_id: str) -> Approval | None: ...

    @abstractmethod
    async def create(self, approval: Approval) -> Approval: ...

    @abstractmethod
    async def save(self, approval: Approval) -> Approval:
        """Store a changed record. Raises `SessionConflict` if someone else
        saved it since it was loaded."""

    @abstractmethod
    async def list(self, *, limit: int = 200) -> list[Approval]:
        """The newest first."""

    async def delete(self, approval_id: str) -> None:  # noqa: B027 - optional
        return None

    async def change(self, approval_id: str,
                     edit: Callable[[Approval], Any]) -> Approval:
        """Load, edit, save — again if someone else saved in between. `edit`
        raises to refuse, or returns False to leave the record as it is."""
        for _ in range(8):
            record = await self.load(approval_id)
            if record is None:
                raise ApprovalError(f"no approval {approval_id!r}")
            if edit(record) is False:
                return record
            record.updated = time.time()
            try:
                return await self.save(record)
            except SessionConflict:
                continue
        raise ApprovalError(f"approval {approval_id} could not be saved: it kept changing")


class MemoryApprovalStore(ApprovalStore):
    """In this process, gone with it. For tests, and for trying it out."""

    def __init__(self) -> None:
        self._held: dict[str, Approval] = {}

    async def load(self, approval_id: str) -> Approval | None:
        held = self._held.get(approval_id)
        return held.model_copy(deep=True) if held is not None else None

    async def create(self, approval: Approval) -> Approval:
        approval.version = 1
        self._held[approval.id] = approval.model_copy(deep=True)
        return approval

    async def save(self, approval: Approval) -> Approval:
        held = self._held.get(approval.id)
        if held is not None and held.version != approval.version:
            raise SessionConflict(f"approval {approval.id} was saved by someone else",
                                  session_id=approval.id)
        approval.version += 1
        self._held[approval.id] = approval.model_copy(deep=True)
        return approval

    async def list(self, *, limit: int = 200) -> list[Approval]:
        rows = sorted(self._held.values(), key=lambda a: -a.created)[:limit]
        return [a.model_copy(deep=True) for a in rows]

    async def delete(self, approval_id: str) -> None:
        self._held.pop(approval_id, None)


class FileApprovalStore(ApprovalStore):
    """One JSON file per paused run, in a directory. What `Harness.local()` uses.

    Good for one machine. For several replicas, keep approvals where the
    sessions are: `SessionApprovalStore`.
    """

    def __init__(self, root: str | Path = ".harness/approvals") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, approval_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in approval_id)
        return self.root / f"{safe}.json"

    @staticmethod
    def _parse(text: str) -> Approval | None:
        try:
            return Approval(**json.loads(text))
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _read(self, path: Path) -> Approval | None:
        try:
            return self._parse(path.read_text(encoding="utf-8"))
        except OSError:
            return None

    def _write(self, approval: Approval, expect: int | None) -> bool:
        path = self._path(approval.id)
        # Check-then-write has to be one step for every process that shares
        # this directory, not only for this one: two workers resuming the same
        # approval are two processes. A lock file is the one thing a filesystem
        # will give to exactly one of them.
        lock = path.with_suffix(".lock")
        deadline = time.monotonic() + 10
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 30:
                        lock.unlink(missing_ok=True)     # its holder died mid-write
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.005)
        try:
            held = self._read(path)
            if expect is None:
                if held is not None:
                    return False
            elif held is None or held.version != expect:
                return False
            scratch = path.with_suffix(f".{new_id()}.tmp")
            scratch.write_text(approval.model_dump_json(indent=2), encoding="utf-8")
            scratch.replace(path)
            return True
        finally:
            lock.unlink(missing_ok=True)

    async def load(self, approval_id: str) -> Approval | None:
        return await asyncio.to_thread(self._read, self._path(approval_id))

    async def create(self, approval: Approval) -> Approval:
        approval.version = 1
        if not await asyncio.to_thread(self._write, approval, None):
            raise SessionConflict(f"approval {approval.id} already exists",
                                  session_id=approval.id)
        return approval

    async def save(self, approval: Approval) -> Approval:
        expect = approval.version
        approval.version += 1
        written = await asyncio.to_thread(self._write, approval, expect)
        if not written:
            approval.version = expect
            raise SessionConflict(f"approval {approval.id} was saved by someone else",
                                  session_id=approval.id)
        return approval

    async def list(self, *, limit: int = 200) -> list[Approval]:
        def scan() -> list[Approval]:
            files = sorted(self.root.glob("*.json"), key=lambda f: -f.stat().st_mtime)
            found = (self._read(f) for f in files[:max(limit * 2, limit)])
            return [a for a in found if a is not None][:limit]

        return await asyncio.to_thread(scan)

    async def delete(self, approval_id: str) -> None:
        self._path(approval_id).unlink(missing_ok=True)


class SessionApprovalStore(ApprovalStore):
    """Approvals kept in a session store: any database the harness keeps chats in.

    Each paused run is one row there, under an id and an agent name of its own,
    so it is never listed among an agent's chats — and it is versioned the same
    way a chat is, which is what a claim relies on.
    """

    #: The agent name every approval row is filed under.
    AGENT = "#approvals"

    def __init__(self, sessions: SessionStore, *, prefix: str = "approval_") -> None:
        self.sessions = sessions
        self.prefix = prefix

    def _session(self, approval: Approval) -> Session:
        return Session(
            id=f"{self.prefix}{approval.id}", agent=self.AGENT,
            title=approval.describe()[:120], user_id=approval.user_id,
            tenant_id=approval.tenant_id, version=approval.version,
            created=approval.created,
            metadata={"approval": approval.model_dump(mode="json", exclude={"version"})})

    @staticmethod
    def _record(session: Session) -> Approval | None:
        held = session.metadata.get("approval")
        if not isinstance(held, dict):
            return None
        try:
            return Approval(**held, version=session.version)
        except (ValueError, TypeError):
            return None

    async def load(self, approval_id: str) -> Approval | None:
        try:
            session = await self.sessions.load(f"{self.prefix}{approval_id}")
        except ConfigurationError:
            return None
        return self._record(session)

    async def create(self, approval: Approval) -> Approval:
        approval.version = 0
        saved = await self.sessions.save(self._session(approval))
        approval.version = saved.version
        return approval

    async def save(self, approval: Approval) -> Approval:
        saved = await self.sessions.save(self._session(approval))
        approval.version = saved.version
        return approval

    async def list(self, *, limit: int = 200) -> list[Approval]:
        rows = await self.sessions.list(limit=limit, agent=self.AGENT)
        return [a for a in map(self._record, rows) if a is not None]

    async def delete(self, approval_id: str) -> None:
        await self.sessions.delete(f"{self.prefix}{approval_id}")


# ----------------------------------------------------------------------
# the desk
# ----------------------------------------------------------------------
class Approvals:
    """Where paused runs wait for people, and where people answer.

    ``store``          where the records are kept
    ``expires``        seconds an unanswered request stays open; after that it
                       counts as declined, and resuming tells the agent so.
                       None: it waits for ever.
    ``notify``         called with each new `Approval` — post it to Slack, a
                       ticket queue, a web page — so a person knows there is
                       something to answer. One that fails does not lose the
                       request: it is stored first.
    ``self_approval``  may the person the agent is acting for approve their own
                       request? False enforces a second pair of eyes.
    ``claim_timeout``  seconds after which a resume that never finished — its
                       worker died — may be released by hand with `release`
    """

    def __init__(self, store: ApprovalStore | None = None, *,
                 expires: float | None = 7 * 24 * 3600.0,
                 notify: Callable[[Approval], Any] | None = None,
                 self_approval: bool = True, claim_timeout: float = 900.0,
                 audit: Callable[..., Any] | None = None) -> None:
        self.store = store or MemoryApprovalStore()
        self.expires = expires
        self.notify = notify
        self.self_approval = self_approval
        self.claim_timeout = claim_timeout
        #: `audit(actor, action, target=…, decision=…, **detail)` — the harness's
        #: audit trail, set when this is given to a harness.
        self.audit = audit

    def _note(self, actor: str, action: str, approval: Approval, decision: str,
              **detail: Any) -> None:
        if self.audit is None:
            return
        try:
            self.audit(actor, action, target=approval.id, decision=decision,
                       run_id=approval.run_id, agent=approval.agent, **detail)
        except Exception:  # noqa: S110 - an audit sink must not lose an approval
            pass

    # ---- the harness side ---------------------------------------------------
    async def open(self, approval: Approval) -> Approval:
        """Store a paused run and tell whoever answers these."""
        if approval.expires is None and self.expires is not None:
            approval.expires = approval.created + self.expires
        saved = await self.store.create(approval)
        self._note(approval.agent, "approval_requested", saved, "pending",
                   calls=[c.describe() for c in saved.calls])
        if self.notify is not None:
            try:
                sent = self.notify(saved)
                if inspect.isawaitable(sent):
                    await sent
            except Exception as exc:
                self._note("system", "approval_notify_failed", saved, "error",
                           error=str(exc)[:200])
        return saved

    async def claim(self, approval_id: str, *, by: str = "",
                    user_id: str | None = None, tenant_id: str | None = None,
                    agent: str | None = None) -> Approval:
        """Take a decided request to resume it. Only one taker gets it."""
        def take(record: Approval) -> None:
            if not record.owned_by(user_id, tenant_id):
                # In the words of one that does not exist: an id cannot be probed.
                raise ApprovalError(f"no approval {approval_id!r}")
            if agent is not None and record.agent != agent:
                raise ApprovalError(
                    f"approval {record.id} is for the agent {record.agent!r}, "
                    f"not {agent!r}")
            if record.expired:
                record.status = "expired"
            if record.status == "pending":
                waiting = ", ".join(c.describe() for c in record.waiting)
                raise ApprovalError(
                    f"approval {record.id} is still waiting for an answer on: {waiting}")
            if record.status == "resuming":
                raise ApprovalError(
                    f"approval {record.id} is already being resumed"
                    + (f" by {record.claimed_by}" if record.claimed_by else "")
                    + " — if that worker is gone, release it first")
            if record.status in ("resumed", "failed"):
                raise ApprovalError(f"approval {record.id} was already resumed "
                                    f"({record.status})")
            record.status = "resuming"
            record.claimed, record.claimed_by = time.time(), by

        claimed = await self.store.change(approval_id, take)
        self._note(by or claimed.agent, "approval_resumed", claimed, "ok")
        return claimed

    async def finish(self, approval_id: str, *, outcome: dict[str, Any],
                     failed: bool = False) -> Approval:
        def close(record: Approval) -> None:
            record.status = "failed" if failed else "resumed"
            record.outcome = outcome
            # The conversation is in the session now; the record keeps the
            # decision and how it ended.
            record.messages, record.done = [], []

        return await self.store.change(approval_id, close)

    async def release(self, approval_id: str) -> Approval:
        """Give back a claim whose worker died before finishing, so the run can
        be resumed again. The approved tool may already have run — check first."""
        def back(record: Approval) -> None:
            if record.status != "resuming":
                raise ApprovalError(f"approval {record.id} is not being resumed "
                                    f"({record.status})")
            held = time.time() - (record.claimed or 0)
            if held < self.claim_timeout:
                raise ApprovalError(
                    f"approval {record.id} was claimed {held:.0f}s ago; it can be "
                    f"released after {self.claim_timeout:.0f}s")
            record.status = ("approved" if any(c.status == "approved"
                                               for c in record.calls) else "denied")
            record.claimed, record.claimed_by = None, ""

        return await self.store.change(approval_id, back)

    # ---- the human side -------------------------------------------------------
    async def get(self, approval_id: str) -> Approval:
        record = await self.store.load(approval_id)
        if record is None:
            raise ApprovalError(f"no approval {approval_id!r}")
        return record

    async def pending(self, *, user_id: str | None = None, tenant_id: str | None = None,
                      agent: str | None = None, limit: int = 200) -> list[Approval]:
        """The requests still waiting for an answer, newest first."""
        return [a for a in await self.list(user_id=user_id, tenant_id=tenant_id,
                                           agent=agent, limit=limit)
                if a.status == "pending" and not a.expired]

    async def list(self, *, status: str | None = None, user_id: str | None = None,
                   tenant_id: str | None = None, agent: str | None = None,
                   limit: int = 200) -> list[Approval]:
        rows = await self.store.list(limit=limit)
        return [a for a in rows
                if a.owned_by(user_id, tenant_id)
                and (agent is None or a.agent == agent)
                and (status is None or status == ("expired" if a.expired else a.status))]

    async def approve(self, approval_id: str, *, by: str, note: str = "",
                      call: str | None = None) -> Approval:
        """Say yes — to every waiting call, or to the one named."""
        return await self._decide(approval_id, True, by, note, call)

    async def deny(self, approval_id: str, *, by: str, note: str = "",
                   call: str | None = None) -> Approval:
        """Say no. The agent is told who declined and why, and carries on."""
        return await self._decide(approval_id, False, by, note, call)

    async def _decide(self, approval_id: str, yes: bool, by: str, note: str,
                      call: str | None) -> Approval:
        if not str(by or "").strip():
            raise ApprovalError("an approval is given by someone — pass by=")

        def decide(record: Approval) -> Any:
            if record.expired:
                raise ApprovalError(f"approval {record.id} expired without an answer")
            if record.status != "pending":
                raise ApprovalError(
                    f"approval {record.id} is already {record.status}")
            if yes and not self.self_approval and record.user_id and by == record.user_id:
                raise ApprovalError(
                    "separation of duties: a request cannot be approved by the "
                    "person it was made for")
            targets = [c for c in record.waiting if call in (None, c.id)]
            if not targets:
                raise ApprovalError(
                    f"approval {record.id} has no waiting call {call!r}")
            for item in targets:
                item.status = "approved" if yes else "denied"
                item.by, item.note, item.decided = by, note, time.time()
            record._settle()
            return None

        decided = await self.store.change(approval_id, decide)
        self._note(by, "approval_decided", decided, "approve" if yes else "deny",
                   note=note[:200], call=call or "all")
        return decided

    async def expire(self) -> list[Approval]:
        """Mark every unanswered request that is past its time. Returns them —
        resume each to let its agent wrap up."""
        out = []
        for record in await self.store.list(limit=1000):
            if record.expired:
                def lapse(held: Approval) -> Any:
                    if not held.expired:
                        return False
                    held.status = "expired"
                    return None

                out.append(await self.store.change(record.id, lapse))
        return out
