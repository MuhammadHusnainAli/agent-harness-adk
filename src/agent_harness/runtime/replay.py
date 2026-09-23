"""Replay and time travel: re-run from any prior step. Reproduce before you fix.

Two different things, both useful:

**Time travel** — `Replayer.resume()` loads a checkpoint and continues the run
from that step with a live model. Use it when a run went wrong at step 7 and you
want to try step 7 again with a changed prompt or tool.

**Deterministic reproduction** — wrap your provider in `RecordingProvider` once,
then replay the recording with `ReplayProvider`. The same model responses come
back in the same order, with no network and no spend, so a bug reproduces
exactly. This is what turns "it failed in production" into a test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

from ..errors import ConfigurationError, ProviderError
from ..providers.base import CompletionRequest, Provider
from ..types import Message, ModelResponse, RunResult, StreamEvent
from .checkpoints import Checkpoint, Checkpointer

__all__ = ["RecordingProvider", "ReplayProvider", "Replayer", "request_key"]


def request_key(request: CompletionRequest) -> str:
    """A stable fingerprint of a model call, used to match a recording."""
    blob = json.dumps(
        {
            "model": request.model,
            "system": request.system or "",
            "messages": [
                {"role": m.role, "content": [b.model_dump(mode="json")
                                             for b in m.content]}
                for m in request.messages
            ],
            "tools": sorted(t.name for t in request.tools),
            "tool_choice": str(request.tool_choice),
        },
        sort_keys=True, default=str, separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class RecordingProvider(Provider):
    """Wraps a real provider and writes every exchange to a JSONL file."""

    name: ClassVar[str] = "recording"
    BASE_URL: ClassVar[str] = ""

    def __init__(self, inner: Provider, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.name = f"recording:{inner.name}"  # type: ignore[misc]
        self.api_key = "recorded"
        self.base_url = ""
        self.timeout = getattr(inner, "timeout", 600.0)
        self.max_retries = 0
        self.extra_headers: dict[str, str] = {}
        self._client = None
        self._owns_client = False
        self.count = 0

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        response = await self.inner.complete(req)
        self.count += 1
        record = {
            "seq": self.count,
            "key": request_key(req),
            "model": req.model,
            "response": response.model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return response

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        response = await self.complete(req)
        if response.text:
            yield StreamEvent(type="text", text=response.text)
        yield StreamEvent(type="step_end",
                          data={"response": response.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def aclose(self) -> None:
        await self.inner.aclose()


class ReplayProvider(Provider):
    """Replays a recording. No network, no keys, no spend.

    Matches each call by fingerprint. If the run diverges from the recording —
    because you changed a prompt — it falls back to the next unused response in
    order, and `divergences` tells you where that happened.
    """

    name: ClassVar[str] = "replay"
    BASE_URL: ClassVar[str] = ""

    def __init__(self, path: str | Path, *, strict: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ConfigurationError(f"no recording at {self.path}")
        self.strict = strict
        self.api_key = "replay"
        self.base_url = ""
        self.timeout = 0.0
        self.max_retries = 0
        self.extra_headers: dict[str, str] = {}
        self._client = None
        self._owns_client = False

        self.records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._by_key: dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            self._by_key.setdefault(record.get("key", ""), []).append(index)
        self._used: set[int] = set()
        self.divergences: list[str] = []
        self.requests: list[CompletionRequest] = []

    def _take(self, req: CompletionRequest) -> dict[str, Any]:
        key = request_key(req)
        for index in self._by_key.get(key, []):
            if index not in self._used:
                self._used.add(index)
                return self.records[index]

        self.divergences.append(
            f"call {len(self.requests)}: no recording matches this request"
        )
        if self.strict:
            raise ProviderError(
                f"replay diverged at call {len(self.requests)} — the request does "
                "not match the recording (re-record, or use strict=False)",
                provider=self.name,
            )
        for index, record in enumerate(self.records):
            if index not in self._used:
                self._used.add(index)
                return record
        raise ProviderError("the recording ran out of responses", provider=self.name)

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        self.requests.append(req)
        return ModelResponse(**self._take(req)["response"])

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        response = await self.complete(req)
        if response.text:
            yield StreamEvent(type="text", text=response.text)
        yield StreamEvent(type="step_end",
                          data={"response": response.model_dump(mode="json")})

    @property
    def exhausted(self) -> bool:
        return len(self._used) >= len(self.records)

    async def aclose(self) -> None:
        return None


class Replayer:
    """Re-run a recorded run, whole or from a given step."""

    def __init__(self, checkpoints: Checkpointer) -> None:
        self.checkpoints = checkpoints

    async def history(self, run_id: str) -> list[Checkpoint]:
        return await self.checkpoints.history(run_id)

    async def inspect(self, run_id: str, step: int | None = None) -> Checkpoint:
        """The state the agent was holding at that step."""
        return await self.checkpoints.load(run_id, step)

    async def timeline(self, run_id: str) -> list[dict[str, Any]]:
        """A step-by-step summary of the run — what to read before you re-run it."""
        rows: list[dict[str, Any]] = []
        for checkpoint in await self.history(run_id):
            last = checkpoint.messages[-1] if checkpoint.messages else None
            rows.append({
                "step": checkpoint.step,
                "agent": checkpoint.agent,
                "messages": len(checkpoint.messages),
                "cost_usd": round(checkpoint.usage.cost_usd, 6),
                "tools_called": [b.name for b in (last.tool_uses if last else [])],
                "last": (last.text[:160] if last else ""),
            })
        return rows

    async def resume(self, agent: Any, run_id: str, *, step: int | None = None,
                     task: str | None = None, **kwargs: Any) -> RunResult:
        """Continue the run from `step` (default: the last checkpoint).

        The agent keeps the conversation it had at that point. Pass `task` to
        change what it is asked to do next — that is the point of time travel.
        """
        checkpoint = await self.checkpoints.load(run_id, step)
        history = [m.model_copy(deep=True) for m in checkpoint.messages]

        # Continue from a clean boundary: a trailing assistant turn with unanswered
        # tool calls would leave the next request with orphaned tool_use blocks.
        while history and history[-1].role == "assistant" and history[-1].tool_uses:
            history.pop()

        prompt = task or self._last_user_message(history) or "Continue."
        if task is None and history and history[-1].role == "user":
            history.pop()
        return await agent.run(prompt, messages=history, **kwargs)

    @staticmethod
    def _last_user_message(history: list[Message]) -> str:
        for message in reversed(history):
            if message.role == "user" and message.text:
                return message.text
        return ""
