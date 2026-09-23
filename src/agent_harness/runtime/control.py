"""Stop control: abort a run and drain the sub-agents. A human is always in charge.

    control = harness.control
    control.stop("the customer withdrew the request")

The loop checks the controller before every step and every tool call, so a stop
takes effect at the next safe boundary rather than tearing a run in half. Work
already in flight is allowed to finish and hand back what it has (`drain`), or
cancelled outright (`abort`).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

from ..errors import StopRequested

__all__ = ["StopController", "StopState"]

Mode = Literal["drain", "abort"]


@dataclass
class StopState:
    """Why a run was stopped, by whom, and how."""

    stopped: bool = False
    mode: Mode = "drain"
    reason: str = ""
    requested_by: str = "human"
    ts: float = 0.0


class StopController:
    """One switch, shared by every agent in a run.

    `drain` lets in-flight work finish and hand back partial results; `abort`
    cancels the tasks it is tracking. Either way nothing *new* starts.
    """

    def __init__(self) -> None:
        self.state = StopState()
        self._event = asyncio.Event()
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._agents: dict[str, str] = {}
        self._listeners: list[Any] = []

    # ---- asking ---------------------------------------------------------
    @property
    def stopped(self) -> bool:
        return self.state.stopped

    @property
    def draining(self) -> bool:
        return self.state.stopped and self.state.mode == "drain"

    def check(self, where: str = "") -> None:
        """Raise if a stop is in force. The loop calls this at safe boundaries."""
        if not self.state.stopped:
            return
        detail = f" at {where}" if where else ""
        raise StopRequested(
            f"run stopped by {self.state.requested_by}{detail}: "
            f"{self.state.reason or 'no reason given'}"
        )

    def may_start(self) -> bool:
        """False once a stop is in force — nothing new should be started."""
        return not self.state.stopped

    # ---- pulling the switch ---------------------------------------------
    def stop(self, reason: str = "", *, mode: Mode = "drain",
             requested_by: str = "human") -> StopState:
        """Stop the run. Idempotent — the first reason is the one that sticks."""
        if self.state.stopped:
            return self.state
        self.state = StopState(stopped=True, mode=mode, reason=reason,
                               requested_by=requested_by, ts=time.time())
        self._event.set()
        for listener in list(self._listeners):
            try:
                listener(self.state)
            except Exception:  # a listener must never break the stop
                continue
        if mode == "abort":
            self.cancel_inflight()
        return self.state

    def abort(self, reason: str = "", *, requested_by: str = "human") -> StopState:
        """Stop and cancel everything in flight."""
        return self.stop(reason, mode="abort", requested_by=requested_by)

    def resume(self) -> None:
        """Clear the stop so the controller can be reused for the next run."""
        self.state = StopState()
        self._event = asyncio.Event()

    def on_stop(self, listener: Any) -> Any:
        self._listeners.append(listener)
        return listener

    async def wait(self, timeout: float | None = None) -> bool:
        """Block until someone stops the run. Returns False on timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except (TimeoutError, asyncio.TimeoutError):
            return False

    # ---- tracking what is in flight --------------------------------------
    def track(self, key: str, task: asyncio.Task[Any]) -> None:
        self._inflight[key] = task
        task.add_done_callback(lambda _: self._inflight.pop(key, None))

    def enter(self, agent: str, run_id: str) -> None:
        self._agents[run_id] = agent

    def leave(self, run_id: str) -> None:
        self._agents.pop(run_id, None)

    @property
    def running_agents(self) -> list[str]:
        return sorted(self._agents.values())

    def cancel_inflight(self) -> int:
        cancelled = 0
        for task in list(self._inflight.values()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    async def drain(self, timeout: float = 30.0) -> bool:
        """Wait for in-flight work to finish. True if everything landed in time."""
        tasks = [t for t in self._inflight.values() if not t.done()]
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        return not pending

    def report(self) -> dict[str, Any]:
        return {
            "stopped": self.state.stopped,
            "mode": self.state.mode,
            "reason": self.state.reason,
            "requested_by": self.state.requested_by,
            "inflight": len(self._inflight),
            "agents_running": self.running_agents,
        }
