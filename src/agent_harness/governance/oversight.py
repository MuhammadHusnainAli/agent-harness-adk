"""Human oversight: approvals that fail closed.

    desk = OversightDesk()                       # a queue humans answer through your API
    ...
    for request in desk.pending():
        desk.approve(request.id, by="maria@acme.com", note="checked the invoice")

or give it a callable and let it ask:

    desk = OversightDesk(approver=lambda request: slack_ask(request))

The rules every framework lands on, built in:

* **Fail closed.** No approver, no answer before the timeout, an approver
  that raises — all of those are a *no*.
* **A quorum of distinct people.** ``approvers: 2`` needs two different
  names. An anonymous ``True`` counts as one vote, however many times it is
  given, so four-eyes cannot be satisfied by one rubber stamp.
* **Separation of duties.** The person the agent is acting for cannot approve
  their own request.
* **Any denial ends it.** One *no* outweighs any number of *yes*.

Every request and every vote goes into the audit trail.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import new_id

__all__ = ["ApprovalVote", "ApprovalRequest", "OversightDesk"]

Status = Literal["pending", "approved", "denied", "expired"]


@dataclass
class ApprovalVote:
    approve: bool
    by: str = ""
    note: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class ApprovalRequest:
    agent: str
    action: str
    target: str
    reason: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    approvers: int = 1
    timeout: float | None = None
    requester: str = ""
    run_id: str = ""
    id: str = field(default_factory=lambda: new_id("apr"))
    created: float = field(default_factory=time.time)
    status: Status = "pending"
    votes: list[ApprovalVote] = field(default_factory=list)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def approved_by(self) -> set[str]:
        return {v.by or "(anonymous)" for v in self.votes if v.approve}

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "agent": self.agent, "action": self.action,
                "target": self.target, "reason": self.reason, "args": self.args,
                "approvers": self.approvers, "status": self.status,
                "votes": [{"approve": v.approve, "by": v.by, "note": v.note}
                          for v in self.votes]}


Approver = Callable[[ApprovalRequest], Any]


class OversightDesk:
    """Where approval requests wait for people."""

    def __init__(self, approver: Approver | None = None, *,
                 default_timeout: float = 900.0,
                 record: Callable[..., Any] | None = None) -> None:
        #: Called once per vote still needed. May return a bool or an
        #: `ApprovalVote`; sync or async. Leave it out to answer via
        #: `approve` / `deny` from elsewhere (a web handler, a Slack action).
        self.approver = approver
        self.default_timeout = default_timeout
        self.record = record
        self._requests: dict[str, ApprovalRequest] = {}

    # ---- the harness side ----------------------------------------------------
    async def request(self, request: ApprovalRequest) -> ApprovalRequest:
        """Wait for a decision. Returns the request with its final status."""
        self._requests[request.id] = request
        self._note(request, "approval_requested")
        timeout = request.timeout if request.timeout is not None else self.default_timeout
        try:
            if self.approver is not None:
                await asyncio.wait_for(self._ask(request), timeout)
            else:
                await asyncio.wait_for(request._done.wait(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            if request.status == "pending":
                request.status = "expired"
        except Exception as exc:          # an approver that breaks is a no
            request.votes.append(ApprovalVote(False, "system", f"approver failed: {exc}"))
            request.status = "denied"
        if request.status == "pending":
            request.status = "denied"
        self._note(request, "approval_decided")
        return request

    async def _ask(self, request: ApprovalRequest) -> None:
        assert self.approver is not None
        # Ask at most `approvers` times plus a little slack for anonymous repeats.
        for _ in range(request.approvers * 2):
            answer = self.approver(request)
            if inspect.isawaitable(answer):
                answer = await answer
            vote = answer if isinstance(answer, ApprovalVote) else ApprovalVote(bool(answer))
            self._vote(request, vote)
            if request.status != "pending":
                return
        request.status = "denied"

    # ---- the human side --------------------------------------------------------
    def pending(self) -> list[ApprovalRequest]:
        return [r for r in self._requests.values() if r.status == "pending"]

    def get(self, request_id: str) -> ApprovalRequest:
        return self._requests[request_id]

    def approve(self, request_id: str, *, by: str, note: str = "") -> ApprovalRequest:
        return self._vote(self._requests[request_id], ApprovalVote(True, by, note))

    def deny(self, request_id: str, *, by: str, note: str = "") -> ApprovalRequest:
        return self._vote(self._requests[request_id], ApprovalVote(False, by, note))

    def _vote(self, request: ApprovalRequest, vote: ApprovalVote) -> ApprovalRequest:
        if request.status != "pending":
            return request
        if vote.approve and request.requester and vote.by == request.requester:
            vote = ApprovalVote(False, vote.by,
                                "separation of duties: cannot approve your own request")
        request.votes.append(vote)
        if not vote.approve:
            request.status = "denied"
        elif len(request.approved_by) >= request.approvers:
            request.status = "approved"
        if request.status != "pending":
            request._done.set()
        return request

    def _note(self, request: ApprovalRequest, action: str) -> None:
        if self.record is not None:
            self.record(request, action)

    def history(self) -> list[dict[str, Any]]:
        return [r.summary() for r in self._requests.values()]
