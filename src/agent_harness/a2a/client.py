"""Call an agent that is served over A2A — whatever it was built with.

    client = A2AClient("https://agents.example.com/pricing", token="sk-…")
    task = await client.send("What does the gold plan cost?")
    task.text, task.state, task.context_id

    async for event in client.stream("And for 40 seats?", context_id=task.context_id):
        print(event.text, end="")

And as one of your own agents' sub-agents, or a tool, or a workflow step:

    pricing = await RemoteAgent.connect("https://agents.example.com/pricing")
    manager = Agent("manager", subagents=[pricing], tools=[pricing.as_tool("ask_pricing")])
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigurationError
from ..tools import Tool, ToolContext, ToolRegistry
from ..types import Artifact, Message, RunResult
from . import protocol as wire
from .protocol import A2AError

__all__ = ["A2AClient", "RemoteAgent", "RemoteTask", "A2AEvent"]

_RETRY_STATUS = {429, 502, 503, 504}


@dataclass
class RemoteTask:
    """A task on somebody else's agent, as it stood when it was last heard of."""

    id: str = ""
    context_id: str = ""
    state: str = "unknown"
    #: What it came to, as text.
    text: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    #: The task exactly as the server sent it.
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, result: dict[str, Any]) -> RemoteTask:
        if result.get("kind") == "message" or "parts" in result:
            # An agent may answer a simple request with a message and no task.
            return cls(id=result.get("taskId", ""), state="completed",
                       context_id=result.get("contextId", ""),
                       text=wire.text_of(result.get("parts")), raw=result)
        return cls(id=result.get("id", ""), context_id=result.get("contextId", ""),
                   state=(result.get("status") or {}).get("state", "unknown"),
                   text=wire.answer_of(result),
                   artifacts=list(result.get("artifacts") or []), raw=result)

    @property
    def done(self) -> bool:
        return self.state in wire.TERMINAL

    @property
    def ok(self) -> bool:
        return self.state == "completed"

    @property
    def error(self) -> str | None:
        """Why it did not complete, in the server's words."""
        if self.ok or not self.done:
            return None
        said = wire.text_of(
            ((self.raw.get("status") or {}).get("message") or {}).get("parts"))
        return said or f"the task was {self.state}"


@dataclass
class A2AEvent:
    """One event of a streamed task. `text` is what was just written."""

    kind: str                       # task · message · status-update · artifact-update
    text: str = ""
    state: str = ""
    final: bool = False
    task_id: str = ""
    context_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class A2AClient:
    """A client for one A2A agent.

    Args:
        url: the agent's base URL — where its card is — or its endpoint.
        token: a bearer token, if it asks for one.
        headers: anything else every request should carry.
        timeout: how long one request may take. A blocking `send` waits for the
            whole task, so this is the longest a task may run.
        retries: how many times a request that could not be delivered, or was
            told to come back later, is sent again. A resent message carries the
            same message id, so the server runs it once.
        transport, http: an `httpx` transport or client of your own.
    """

    def __init__(self, url: str, *, token: str | None = None,
                 headers: dict[str, str] | None = None, timeout: float = 600.0,
                 retries: int = 2, transport: Any = None, http: Any = None) -> None:
        self.url = url.rstrip("/")
        for suffix in (wire.CARD_PATH, wire.LEGACY_CARD_PATH):
            if self.url.endswith(suffix):
                self.url = self.url[: -len(suffix)]
        self.headers = dict(headers or {})
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.timeout = timeout
        self.retries = max(0, retries)
        self._transport = transport
        self._http = http
        self._owns_http = http is None
        self._card: dict[str, Any] | None = None
        self._endpoint: str | None = None
        self._ids = 0

    # ---- plumbing ----------------------------------------------------------
    @property
    def http(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self.timeout,
                                           transport=self._transport)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None

    async def __aenter__(self) -> A2AClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def card(self, *, refresh: bool = False) -> dict[str, Any]:
        """The agent's card: who it is, what it can do, where to send requests."""
        if self._card is not None and not refresh:
            return self._card
        problem = ""
        for path in (wire.CARD_PATH, wire.LEGACY_CARD_PATH):
            try:
                response = await self.http.get(f"{self.url}{path}", headers=self.headers)
            except Exception as exc:
                raise A2AError(f"{self.url} could not be reached: {exc}",
                               status=503) from None
            if response.status_code == 200:
                try:
                    self._card = response.json()
                except ValueError:
                    problem = "its agent card is not JSON"
                    continue
                return self._card
            problem = f"HTTP {response.status_code}"
        raise A2AError(f"no A2A agent card at {self.url} ({problem})", status=404)

    async def _where(self) -> str:
        """The endpoint requests go to: what the card says, else the URL given."""
        if self._endpoint is None:
            try:
                self._endpoint = (await self.card()).get("url") or self.url
            except A2AError:
                self._endpoint = self.url       # no card: treat the URL as the endpoint
        return self._endpoint

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ids += 1
        return {"jsonrpc": "2.0", "id": self._ids, "method": method, "params": params}

    async def call(self, method: str, params: dict[str, Any]) -> Any:
        """One JSON-RPC call. Returns its result; raises `A2AError` for an error."""
        payload = self._request(method, params)
        endpoint = await self._where()
        attempt = 0
        while True:
            retry_after: float | None = None
            try:
                response = await self.http.post(endpoint, json=payload,
                                                headers=self.headers)
            except Exception as exc:
                if attempt >= self.retries:
                    raise A2AError(f"{endpoint} could not be reached: {exc}",
                                   status=503) from None
            else:
                if response.status_code not in _RETRY_STATUS or attempt >= self.retries:
                    return _result(response.status_code, response.text)
                try:
                    retry_after = float(response.headers.get("retry-after", ""))
                except ValueError:
                    retry_after = None
            await asyncio.sleep(min(retry_after if retry_after is not None
                                    else 0.5 * 2 ** attempt, 30.0))
            attempt += 1

    def _send_params(self, text: str | dict[str, Any], *, context_id: str | None,
                     task_id: str | None, attachments: Any, message_id: str | None,
                     metadata: dict[str, Any] | None, blocking: bool | None,
                     history_length: int | None,
                     push: dict[str, Any] | str | None) -> dict[str, Any]:
        if isinstance(text, dict):
            message = {"kind": "message", "role": "user", **text}
            message.setdefault("messageId", message_id or wire.new_uuid())
        else:
            message = wire.message("user", wire.parts_of(text, attachments),
                                   message_id=message_id or "")
        if context_id:
            message["contextId"] = context_id
        if task_id:
            message["taskId"] = task_id
        params: dict[str, Any] = {"message": message}
        configuration: dict[str, Any] = {}
        if blocking is not None:
            configuration["blocking"] = blocking
        if history_length is not None:
            configuration["historyLength"] = history_length
        if push:
            configuration["pushNotificationConfig"] = (
                {"url": push} if isinstance(push, str) else push)
        if configuration:
            params["configuration"] = configuration
        if metadata:
            params["metadata"] = metadata
        return params

    # ---- the protocol --------------------------------------------------------
    async def send(self, text: str | dict[str, Any], *, context_id: str | None = None,
                   task_id: str | None = None, attachments: Any = (),
                   message_id: str | None = None,
                   metadata: dict[str, Any] | None = None, blocking: bool = True,
                   history_length: int | None = None,
                   push: dict[str, Any] | str | None = None) -> RemoteTask:
        """Send a message. Blocking, the task comes back finished; otherwise as
        soon as it is accepted — ask after it with `get` or `wait`.

        `context_id` carries a conversation on: pass the one a task came back
        with.
        """
        params = self._send_params(
            text, context_id=context_id, task_id=task_id, attachments=attachments,
            message_id=message_id, metadata=metadata,
            blocking=None if blocking else False, history_length=history_length,
            push=push)
        return RemoteTask.of(await self.call("message/send", params))

    async def stream(self, text: str | dict[str, Any], *,
                     context_id: str | None = None, attachments: Any = (),
                     message_id: str | None = None,
                     metadata: dict[str, Any] | None = None
                     ) -> AsyncIterator[A2AEvent]:
        """Send a message and hear the task as it happens."""
        params = self._send_params(
            text, context_id=context_id, task_id=None, attachments=attachments,
            message_id=message_id, metadata=metadata, blocking=None,
            history_length=None, push=None)
        async for event in self._events("message/stream", params):
            yield event

    async def resubscribe(self, task_id: str) -> AsyncIterator[A2AEvent]:
        """Pick the events of a running task back up after a dropped connection."""
        async for event in self._events("tasks/resubscribe", {"id": task_id}):
            yield event

    async def _events(self, method: str,
                      params: dict[str, Any]) -> AsyncIterator[A2AEvent]:
        endpoint = await self._where()
        headers = {**self.headers, "Accept": "text/event-stream"}
        async with self.http.stream("POST", endpoint, headers=headers,
                                    json=self._request(method, params)) as response:
            if "text/event-stream" not in response.headers.get("content-type", ""):
                # Refused before any stream began: the body is the error.
                await response.aread()
                result = _result(response.status_code, response.text)
                yield _event(result)        # a server that answered without streaming
                return
            data: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    yield _event(_unwrap(json.loads("\n".join(data))))
                    data = []
            if data:
                yield _event(_unwrap(json.loads("\n".join(data))))

    async def get(self, task_id: str, *,
                  history_length: int | None = None) -> RemoteTask:
        params: dict[str, Any] = {"id": task_id}
        if history_length is not None:
            params["historyLength"] = history_length
        return RemoteTask.of(await self.call("tasks/get", params))

    async def cancel(self, task_id: str) -> RemoteTask:
        return RemoteTask.of(await self.call("tasks/cancel", {"id": task_id}))

    async def wait(self, task_id: str, *, timeout: float = 600.0,
                   interval: float = 0.5) -> RemoteTask:
        """Ask after a task until it is over. Raises `A2AError` if it is not by
        `timeout`."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            task = await self.get(task_id)
            if task.done or task.state in ("input-required", "auth-required"):
                return task
            if loop.time() >= deadline:
                raise A2AError(f"task {task_id} was still {task.state} after "
                               f"{timeout:g}s", status=504)
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 5.0)

    async def set_push(self, task_id: str, url: str, *,
                       token: str | None = None) -> dict[str, Any]:
        """Ask to be told at `url` when a task is over."""
        config: dict[str, Any] = {"url": url}
        if token:
            config["token"] = token
        return await self.call("tasks/pushNotificationConfig/set",
                               {"taskId": task_id, "pushNotificationConfig": config})


def _unwrap(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise A2AError("the agent's reply is not a JSON-RPC response",
                       code=wire.INVALID_AGENT_RESPONSE)
    error = response.get("error")
    if error:
        raise A2AError(str(error.get("message") or "the agent returned an error"),
                       code=int(error.get("code") or wire.INTERNAL_ERROR),
                       data=error.get("data"))
    return response.get("result")


def _result(status: int, body: str) -> Any:
    try:
        parsed = json.loads(body)
    except ValueError:
        raise A2AError(f"the agent answered HTTP {status} with something that is "
                       f"not JSON: {body[:200]!r}",
                       code=wire.INVALID_AGENT_RESPONSE, status=status) from None
    try:
        return _unwrap(parsed)
    except A2AError as exc:
        exc.status = status
        raise


def _event(result: Any) -> A2AEvent:
    if not isinstance(result, dict):
        raise A2AError("the agent sent an event that is not an object",
                       code=wire.INVALID_AGENT_RESPONSE)
    kind = result.get("kind") or ("task" if "status" in result and "id" in result
                                  else "message")
    if kind == "artifact-update":
        return A2AEvent(kind, text=wire.text_of((result.get("artifact") or {}).get("parts")),
                        task_id=result.get("taskId", ""),
                        context_id=result.get("contextId", ""), raw=result)
    if kind == "status-update":
        status = result.get("status") or {}
        return A2AEvent(kind, state=status.get("state", ""),
                        final=bool(result.get("final")),
                        task_id=result.get("taskId", ""),
                        context_id=result.get("contextId", ""), raw=result)
    if kind == "message":
        return A2AEvent(kind, text=wire.text_of(result.get("parts")), state="completed",
                        final=True, task_id=result.get("taskId", ""),
                        context_id=result.get("contextId", ""), raw=result)
    task = RemoteTask.of(result)
    return A2AEvent("task", state=task.state, final=task.done, task_id=task.id,
                    context_id=task.context_id, raw=result)


class RemoteAgent:
    """An agent somewhere else, used as if it were one of yours.

    It runs a task and hands back a `RunResult`, which is all the harness asks
    of a sub-agent, a tool, or a workflow step:

        Agent("manager", subagents=[remote])
        Agent("manager", tools=[remote.as_tool()])
        Workflow.from_file("flow.yaml", agents=[remote])

    Run directly, it keeps the conversation going between runs
    (`new_session()` starts another). As a sub-agent it starts clean each time,
    like any other.

    Args:
        url: the remote agent's base URL.
        name, description: what your agents know it as. `RemoteAgent.connect`
            reads both from its card.
        token, headers, timeout, transport: as for `A2AClient`.
        harness: a harness to record the calls on — and, if it is governed, to
            register this agent with.
        identity: its governance identity, as for an `Agent`.
    """

    def __init__(self, url: str, *, name: str = "", description: str = "",
                 token: str | None = None, headers: dict[str, str] | None = None,
                 timeout: float = 600.0, transport: Any = None,
                 client: A2AClient | None = None, harness: Any = None,
                 identity: Any = None, budget: Any = None) -> None:
        self.client = client or A2AClient(url, token=token, headers=headers,
                                          timeout=timeout, transport=transport)
        self.url = self.client.url
        self.name = name or self.url.rstrip("/").rsplit("/", 1)[-1].split(":")[0] \
            or "remote"
        self.description = description or f"A remote agent at {self.url}."
        self.harness = harness
        self.identity = identity
        self.budget = budget
        self.timeout = timeout
        # What the harness reads off any agent it is handed.
        self.tools = ToolRegistry()
        self.skills = None
        self.subagents: dict[str, Any] = {}
        self.model = "a2a"
        self._provider = "a2a"
        self.version = ""
        self.mode = None
        self._context_id: str | None = None
        self._register()

    def _register(self) -> None:
        """On a governed harness, an agent has to be known before anything is
        delegated to it — a remote one as much as a local one."""
        harness = self.harness
        if harness is not None and getattr(harness, "governance", None) is not None:
            harness.governance.register_agent(self)

    @classmethod
    async def connect(cls, url: str, **kwargs: Any) -> RemoteAgent:
        """Read the agent's card, and call it by its own name and description."""
        # Registered once it has its own name, not the one guessed from the URL.
        harness = kwargs.pop("harness", None)
        agent = cls(url, **kwargs)
        card = await agent.client.card()
        if not kwargs.get("name"):
            agent.name = _identifier(card.get("name") or agent.name)
        if not kwargs.get("description"):
            skills = "; ".join(s.get("description") or s.get("name", "")
                               for s in card.get("skills") or [])
            agent.description = card.get("description") or skills or agent.description
        agent.version = str(card.get("version") or "")
        agent.harness = harness
        agent._register()
        return agent

    def new_session(self) -> None:
        """Start a fresh conversation: the next run does not continue the last."""
        self._context_id = None

    async def run(self, task: str | Message, *, messages: Any = None,
                  attachments: Any = (), guard: Any = None,
                  **_: Any) -> RunResult:
        """Send the task and wait for it. Never raises for a failed task: like
        an agent's own run, the failure is `result.error`."""
        text = task.text if isinstance(task, Message) else str(task)
        files = [*(task.media if isinstance(task, Message) else ()), *attachments]
        one_off = messages is not None       # a sub-agent's task: no thread to keep
        result = RunResult(agent=self.name)
        try:
            remote = await self.client.send(
                text, attachments=files,
                context_id=None if one_off else self._context_id)
            if not remote.done and remote.id:
                remote = await self.client.wait(remote.id, timeout=self.timeout)
        except A2AError as exc:
            result.error = f"A2AError: {exc}"
            result.stop_reason = "error"
            self._record("error", str(exc))
            return result
        if not one_off:
            self._context_id = remote.context_id or self._context_id
        result.output = remote.text
        result.session_id = remote.context_id
        result.run_id = remote.id or result.run_id
        result.steps = 1
        if not remote.ok:
            result.error = f"A2AError: {remote.error or remote.state}"
            result.stop_reason = "stopped" if remote.state == "canceled" else "error"
        for artifact in remote.artifacts:
            if artifact.get("name") and artifact.get("name") != wire.ANSWER:
                result.artifacts.append(Artifact(
                    name=str(artifact["name"]), produced_by=self.name,
                    content=wire.text_of(artifact.get("parts"))))
        result.messages = [Message.user(text), Message.assistant(remote.text)]
        self._record(remote.state, remote.id)
        return result

    def run_sync(self, task: str | Message, **kwargs: Any) -> RunResult:
        return asyncio.run(self.run(task, **kwargs))

    def _record(self, decision: str, target: str) -> None:
        if self.harness is not None:
            self.harness.audit.record(self.name, "a2a_call", target=target,
                                      decision=decision, url=self.url)

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose the remote agent as a tool another agent can call."""
        agent = self

        async def call_agent(task: str, ctx: ToolContext | None = None) -> str:
            result = await agent.run(task, messages=[])
            return f"[failed] {result.error}" if result.error else result.output

        call_agent.__name__ = name or agent.name
        return Tool(
            call_agent, name=name or agent.name,
            description=description or agent.description,
            parameters={
                "type": "object",
                "properties": {"task": {"type": "string",
                                        "description": "The task, stated in full."}},
                "required": ["task"],
            },
            tags=["agent", "a2a"],
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<RemoteAgent {self.name} {self.url}>"


def _identifier(name: str) -> str:
    """A card's display name as something a model can name in a tool call."""
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        raise ConfigurationError(f"the agent card's name {name!r} cannot be used")
    return cleaned
