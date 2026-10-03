"""Serve an agent over A2A, so any other framework's agent can call it.

    server = A2AServer(agent, auth={"sk-live-1": {"user_id": "ada", "tenant_id": "acme"}})
    app = server                      # an ASGI app:  uvicorn myapp:app --workers 8

    await server.serve(port=8000)     # or the built-in server, for development

It speaks the JSON-RPC binding of A2A 0.3: the agent card at
`/.well-known/agent-card.json`, `message/send`, `message/stream` over SSE,
`tasks/get`, `tasks/cancel`, `tasks/resubscribe`, and push notifications.

Built to run as many replicas behind a load balancer:

* a replica keeps nothing another one needs. Tasks and conversations live in
  the harness's session store — name a database there and every replica shares
  them (`Harness(sessions="postgresql://…")`);
* a message sent twice (a client retrying) is one task, not two;
* a replica that is full says so (`429`, `Retry-After`) rather than queueing
  without end;
* a task can be asked about, cancelled or re-subscribed to on any replica,
  whichever one is running it; and one whose worker died is reported failed
  rather than working for ever.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigurationError, HarnessError, SessionConflict
from ..runtime.session import Session
from ..types import MediaBlock, RunResult
from . import protocol as wire
from .protocol import A2AError
from .store import SessionTaskStore, TaskRecord, TaskStore

__all__ = ["A2AServer", "Principal"]

_SSE_HEADERS = [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"),
                (b"x-accel-buffering", b"no")]
_INPUT_MODES = ["text/plain", "application/json", "application/pdf", "image/png",
                "image/jpeg"]


@dataclass
class Principal:
    """Who a request is from. Their tasks and conversations are theirs."""

    user_id: str | None = None
    tenant_id: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, value: Any) -> Principal:
        if isinstance(value, Principal):
            return value
        if isinstance(value, str):
            return cls(user_id=value)
        if isinstance(value, dict):
            return cls(user_id=value.get("user_id"), tenant_id=value.get("tenant_id"),
                       claims={k: v for k, v in value.items()
                               if k not in ("user_id", "tenant_id")})
        return cls()

    @property
    def key(self) -> str:
        return f"{self.tenant_id or ''}/{self.user_id or ''}"


class _Bus:
    """The events of one running task, for everyone listening to it here."""

    def __init__(self) -> None:
        self.listeners: list[asyncio.Queue[dict[str, Any] | None]] = []

    def listen(self) -> asyncio.Queue[dict[str, Any] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=2000)
        self.listeners.append(queue)
        return queue

    def leave(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        if queue in self.listeners:
            self.listeners.remove(queue)

    def publish(self, event: dict[str, Any] | None) -> None:
        for queue in list(self.listeners):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A listener that cannot keep up is dropped, not waited for.
                self.listeners.remove(queue)


@dataclass
class _Mount:
    name: str
    prefix: str
    agent: Any = None
    factory: Callable[[Principal], Any] | None = None


class A2AServer:
    """One or more agents, served over A2A. It is itself an ASGI application.

    Args:
        agent: the agent to serve; or a function from the caller (`Principal`)
            to the agent that serves them; or a mapping of name → either, each
            served under `/<name>`.
        url: the public base URL, for the agent card. Left out, it is read from
            each request — which is right behind most proxies.
        auth: who may call. A token, or several; a mapping of token → who it
            is (`{"user_id": ..., "tenant_id": ...}`); or a function from the
            request headers to a `Principal` (or None to refuse). Left out, the
            server is open.
        store: where tasks are kept. Left out, the harness's session store.
        max_concurrency: how many tasks run at once on this replica.
        max_queue: how many more may wait. Past that, callers are told to come
            back (`429`).
        task_timeout: the longest one task may run, in seconds.
        push_notifications: let callers register a webhook to be told when a
            task is over. True allows any public https URL; a function from the
            URL to a bool decides for itself.
        isolate: give every conversation its own instance of the agent, acting
            for its caller. Turn it off only for an agent that keeps nothing
            between runs.
        version, provider, documentation_url, skills, security_schemes: what
            the agent card says.
    """

    def __init__(self, agent: Any, *, url: str | None = None, auth: Any = None,
                 store: TaskStore | None = None, max_concurrency: int = 64,
                 max_queue: int = 256, task_timeout: float = 600.0,
                 push_notifications: bool | Callable[[str], bool] = False,
                 isolate: bool = True, version: str = "1.0.0",
                 provider: dict[str, str] | None = None,
                 documentation_url: str | None = None,
                 skills: list[dict[str, Any]] | None = None,
                 security_schemes: dict[str, Any] | None = None,
                 max_body_bytes: int = 10_000_000, max_agents: int = 1024,
                 poll_interval: float = 1.0, keepalive: float = 15.0) -> None:
        self.mounts: dict[str, _Mount] = {}
        if isinstance(agent, dict):
            for name, entry in agent.items():
                self._mount(name, entry, prefix=f"/{name}")
        else:
            self._mount(getattr(agent, "name", "agent"), agent, prefix="")
        if not self.mounts:
            raise ConfigurationError("an A2A server needs at least one agent")
        first = next(iter(self.mounts.values()))
        self.harness = (first.agent.harness if first.agent is not None else None)
        if store is None:
            if self.harness is None:
                raise ConfigurationError(
                    "an A2A server built from a factory needs `store=` — "
                    "SessionTaskStore(harness.sessions), say")
            store = SessionTaskStore(self.harness.sessions)
        self.store = store
        self.url = url.rstrip("/") if url else None
        self.auth = auth
        self.max_concurrency = max(1, max_concurrency)
        self.max_queue = max(0, max_queue)
        self.task_timeout = task_timeout
        self.push_notifications = push_notifications
        self.isolate = isolate
        self.version = version
        self.provider = provider
        self.documentation_url = documentation_url
        self.skills = skills
        self.security_schemes = security_schemes
        self.max_body_bytes = max_body_bytes
        self.max_agents = max_agents
        self.poll_interval = poll_interval
        self.keepalive = keepalive

        self._gate = asyncio.Semaphore(self.max_concurrency)
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._buses: dict[str, _Bus] = {}
        self._agents: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._pushes: set[asyncio.Task[Any]] = set()
        self._http: Any = None
        self._closing = False
        self.stats = {"accepted": 0, "completed": 0, "failed": 0, "canceled": 0,
                      "rejected": 0, "deduplicated": 0}

    def _mount(self, name: str, entry: Any, *, prefix: str) -> None:
        if hasattr(entry, "stream") and hasattr(entry, "harness"):
            self.mounts[name] = _Mount(name, prefix, agent=entry)
        elif callable(entry):
            self.mounts[name] = _Mount(name, prefix, factory=entry)
        else:
            raise ConfigurationError(
                f"{name}: an A2A server serves an Agent, or a function that "
                f"returns one — got {type(entry).__name__}")

    # ------------------------------------------------------------------
    # the agent card
    # ------------------------------------------------------------------
    def card(self, name: str | None = None, base_url: str = "") -> dict[str, Any]:
        """The agent card: what this agent is, where it is, how to call it."""
        mount = self.mounts[name] if name else next(iter(self.mounts.values()))
        agent = mount.agent
        described = getattr(agent, "description", "") or f"{mount.name} agent"
        skills = self.skills
        if skills is None:
            registry = getattr(agent, "skills", None)
            skills = [{"id": s.name, "name": s.name,
                       "description": s.description or s.name,
                       "tags": list(s.tags) or ["skill"]} for s in registry or ()]
            if not skills:
                mode = getattr(getattr(agent, "mode", None), "name", "")
                skills = [{"id": mount.name, "name": mount.name,
                           "description": described, "tags": [mode or "agent"]}]
        outputs = ["text/plain"]
        if getattr(agent, "output_type", None) is not None:
            outputs.append("application/json")
        card: dict[str, Any] = {
            "protocolVersion": wire.PROTOCOL_VERSION,
            "name": mount.name,
            "description": described,
            "url": f"{self.url or base_url}{mount.prefix}/",
            "preferredTransport": "JSONRPC",
            "version": self.version,
            "capabilities": {"streaming": True,
                             "pushNotifications": bool(self.push_notifications),
                             "stateTransitionHistory": False},
            "defaultInputModes": list(_INPUT_MODES),
            "defaultOutputModes": outputs,
            "skills": skills,
            "supportsAuthenticatedExtendedCard": False,
        }
        if self.provider:
            card["provider"] = self.provider
        if self.documentation_url:
            card["documentationUrl"] = self.documentation_url
        if self.security_schemes:
            card["securitySchemes"] = self.security_schemes
            card["security"] = [{name: []} for name in self.security_schemes]
        elif self.auth is not None:
            card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
            card["security"] = [{"bearer": []}]
        return card

    # ------------------------------------------------------------------
    # who is calling
    # ------------------------------------------------------------------
    async def authenticate(self, headers: dict[str, str]) -> Principal:
        """Who this request is from. Raises `A2AError` (401) if nobody we know."""
        auth = self.auth
        if auth is None:
            return Principal()
        if callable(auth):
            found = auth(headers)
            if inspect.isawaitable(found):
                found = await found
            if not found:
                raise A2AError("not authorised", code=wire.INVALID_REQUEST, status=401)
            return Principal.of(None if found is True else found)
        scheme, _, token = headers.get("authorization", "").partition(" ")
        token = token.strip()
        known: dict[str, Any] = (auth if isinstance(auth, dict)
                                 else {str(t): None for t in
                                       ([auth] if isinstance(auth, str) else auth)})
        if scheme.lower() == "bearer" and token:
            for candidate, who in known.items():
                # Every token is compared, in constant time, so how long this
                # takes says nothing about which one nearly matched.
                if hmac.compare_digest(candidate.encode(), token.encode()):
                    return Principal.of(who)
        raise A2AError("not authorised", code=wire.INVALID_REQUEST, status=401)

    # ------------------------------------------------------------------
    # the agent for a conversation
    # ------------------------------------------------------------------
    def _agent_for(self, mount: _Mount, context_id: str, principal: Principal) -> Any:
        if mount.factory is None and not self.isolate:
            return mount.agent
        key = (mount.name, f"{principal.key}/{context_id}")
        held = self._agents.get(key)
        if held is not None:
            self._agents.move_to_end(key)
            return held
        if mount.factory is not None:
            built = mount.factory(principal)
        else:
            built = _acting_for(mount.agent, principal)
        self._agents[key] = built
        while len(self._agents) > self.max_agents:
            self._agents.popitem(last=False)
        return built

    def _sessions(self, mount: _Mount) -> Any:
        agent = mount.agent
        return agent.harness.sessions if agent is not None else None

    # ------------------------------------------------------------------
    # JSON-RPC
    # ------------------------------------------------------------------
    async def handle(self, payload: Any, *, principal: Principal | None = None,
                     agent: str | None = None) -> Any:
        """Answer one JSON-RPC request.

        Returns the response object — or, for the two streaming methods, an
        async iterator of response objects. This is the whole protocol without
        HTTP around it: mount it in a framework of your own, or call it in a
        test.
        """
        principal = principal or Principal()
        mount = self.mounts[agent] if agent else next(iter(self.mounts.values()))
        request_id = payload.get("id") if isinstance(payload, dict) else None
        try:
            if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" \
                    or not isinstance(payload.get("method"), str):
                raise A2AError("not a JSON-RPC 2.0 request", code=wire.INVALID_REQUEST)
            method = payload["method"]
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise A2AError("params must be an object", code=wire.INVALID_PARAMS)

            if method == "message/stream":
                return self._framed(request_id,
                                    await self._send_stream(mount, params, principal))
            if method == "tasks/resubscribe":
                record = await self._task(params.get("id"), principal)
                return self._framed(request_id, self._follow(record, first=True))

            handler = {
                "message/send": self._send,
                "tasks/get": self._get,
                "tasks/cancel": self._cancel,
                "tasks/pushNotificationConfig/set": self._push_set,
                "tasks/pushNotificationConfig/get": self._push_get,
                "tasks/pushNotificationConfig/list": self._push_list,
                "tasks/pushNotificationConfig/delete": self._push_delete,
            }.get(method)
            if method == "agent/getAuthenticatedExtendedCard":
                raise A2AError("this agent has no extended card",
                               code=wire.NO_EXTENDED_CARD)
            if handler is None:
                raise A2AError(f"method {method!r} is not supported",
                               code=wire.METHOD_NOT_FOUND)
            result = await handler(mount, params, principal)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except A2AError as exc:
            if exc.status != 200:
                raise                       # the transport's to answer: 401, 429
            return {"jsonrpc": "2.0", "id": request_id, "error": exc.body()}
        except SessionConflict as exc:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": wire.INTERNAL_ERROR, "message": str(exc)}}
        except HarnessError as exc:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": wire.INVALID_PARAMS, "message": str(exc)}}

    @staticmethod
    async def _framed(request_id: Any,
                      events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        async for event in events:
            yield {"jsonrpc": "2.0", "id": request_id, "result": event}

    # ---- tasks: finding one -------------------------------------------------
    async def _task(self, task_id: Any, principal: Principal) -> TaskRecord:
        """The caller's task. Somebody else's is "not found", like one that
        never existed, so an id cannot be probed."""
        if not isinstance(task_id, str) or not task_id:
            raise A2AError("a task id is required", code=wire.INVALID_PARAMS)
        record = await self.store.load(task_id)
        if record is None or not record.owned_by(principal.user_id, principal.tenant_id):
            raise A2AError(f"no task {task_id!r}", code=wire.TASK_NOT_FOUND)
        if (record.state not in wire.TERMINAL and task_id not in self._running
                and time.time() - record.updated > self.task_timeout + 60):
            # Nobody here is working on it, and whoever was has had longer than
            # any task is allowed: its worker is gone.
            changed = await self.store.change(task_id, lambda r: self._lose(r))
            record = changed or record
        return record

    def _lose(self, record: TaskRecord) -> bool | None:
        if record.state in wire.TERMINAL:
            return False
        record.task["status"] = wire.status(
            "failed", "The worker handling this task was lost before it finished.",
            task_id=record.id, context_id=record.task.get("contextId", ""))
        record.updated = time.time()
        return None

    # ---- message/send, message/stream ---------------------------------------
    async def _accept(self, mount: _Mount, params: dict[str, Any],
                      principal: Principal) -> tuple[TaskRecord, bool, str,
                                                     list[MediaBlock]]:
        """Turn a message into a task — or find the task it already became."""
        incoming = params.get("message")
        if not isinstance(incoming, dict):
            raise A2AError("`message` is required", code=wire.INVALID_PARAMS)
        if incoming.get("role", "user") != "user":
            raise A2AError("a message sent to an agent has the role `user`",
                           code=wire.INVALID_PARAMS)
        text, files = wire.read_parts(incoming.get("parts"))
        if not text and not files:
            raise A2AError("the message is empty", code=wire.INVALID_PARAMS)

        if incoming.get("taskId"):
            earlier = await self._task(incoming["taskId"], principal)
            if earlier.state in wire.TERMINAL:
                raise A2AError(
                    f"task {earlier.id} is {earlier.state} and takes no more "
                    "messages; send a new message with its contextId to carry on "
                    "the conversation", code=wire.INVALID_PARAMS)
            raise A2AError(f"task {earlier.id} is still {earlier.state}; this agent "
                           "does not take input part-way through a task",
                           code=wire.UNSUPPORTED_OPERATION)

        message_id = str(incoming.get("messageId") or wire.new_uuid())
        context_id = str(incoming.get("contextId") or wire.new_uuid())
        # The same message from the same caller is the same task: a client that
        # retries a request it never saw the answer to does not pay twice.
        digest = hashlib.sha256(
            f"{mount.name}\0{principal.key}\0{message_id}".encode()).hexdigest()
        task_id = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-" \
                  f"{digest[20:32]}"
        existing = await self.store.load(task_id)
        if existing is not None:
            self.stats["deduplicated"] += 1
            return existing, False, text, files

        sessions = self._sessions(mount)
        if sessions is not None and incoming.get("contextId"):
            # A conversation that is somebody else's is refused before any work
            # is started on it — in the words used for one that does not exist.
            try:
                held = await sessions.load(_context_session(mount, context_id))
            except ConfigurationError:
                held = None
            if held is not None and not held.owned_by(principal.user_id,
                                                      principal.tenant_id):
                raise A2AError(f"no conversation {context_id!r}",
                               code=wire.INVALID_PARAMS)

        if self._closing:
            raise A2AError("the server is shutting down", code=wire.SERVER_BUSY,
                           status=503, retry_after=5)
        if len(self._running) >= self.max_concurrency + self.max_queue:
            self.stats["rejected"] += 1
            raise A2AError("the server is at capacity; try again shortly",
                           code=wire.SERVER_BUSY, status=429, retry_after=2)

        configuration = params.get("configuration") or {}
        push: list[dict[str, Any]] = []
        if configuration.get("pushNotificationConfig"):
            push.append(self._push_checked(configuration["pushNotificationConfig"]))
        # Kept without the bytes of what was attached: the record is read far
        # more often than a file would ever be wanted back.
        said = {**incoming, "kind": "message", "role": "user", "messageId": message_id,
                "taskId": task_id, "contextId": context_id,
                "parts": [_without_bytes(part) for part in incoming["parts"]]}
        task: dict[str, Any] = {
            "kind": "task", "id": task_id, "contextId": context_id,
            "status": wire.status("submitted"), "history": [said], "artifacts": [],
        }
        if isinstance(params.get("metadata"), dict):
            task["metadata"] = params["metadata"]
        record = TaskRecord(id=task_id, agent=mount.name, user_id=principal.user_id,
                            tenant_id=principal.tenant_id, task=task, push=push)
        if not await self.store.create(record):
            # Another replica got the same message a moment sooner.
            self.stats["deduplicated"] += 1
            return (await self.store.load(task_id)) or record, False, text, files
        self.stats["accepted"] += 1
        return record, True, text, files

    def _start(self, mount: _Mount, record: TaskRecord, principal: Principal,
               text: str, files: list[MediaBlock]) -> None:
        self._buses[record.id] = _Bus()
        worker = asyncio.create_task(
            self._work(mount, record, principal, text, files))
        self._running[record.id] = worker
        worker.add_done_callback(lambda _: self._settle(record.id))

    def _settle(self, task_id: str) -> None:
        self._running.pop(task_id, None)
        bus = self._buses.pop(task_id, None)
        if bus is not None:
            bus.publish(None)

    async def _send(self, mount: _Mount, params: dict[str, Any],
                    principal: Principal) -> dict[str, Any]:
        record, created, text, files = await self._accept(mount, params, principal)
        configuration = params.get("configuration") or {}
        if created:
            self._start(mount, record, principal, text, files)
        if configuration.get("blocking", True) is not False:
            record = await self._settled(record)
        return wire.trimmed(record.task, configuration.get("historyLength"))

    async def _settled(self, record: TaskRecord) -> TaskRecord:
        """Wait for a task to be over — here, or on whichever replica has it."""
        worker = self._running.get(record.id)
        deadline = time.monotonic() + self.task_timeout + 5
        if worker is not None:
            await asyncio.wait({worker}, timeout=self.task_timeout + 5)
        while True:
            latest = await self.store.load(record.id) or record
            if latest.state in wire.TERMINAL or time.monotonic() >= deadline \
                    or record.id in self._running:
                return latest
            await asyncio.sleep(self.poll_interval)

    async def _send_stream(self, mount: _Mount, params: dict[str, Any],
                           principal: Principal) -> AsyncIterator[dict[str, Any]]:
        record, created, text, files = await self._accept(mount, params, principal)
        if not created:
            return self._follow(record, first=True)
        # Listening before the work starts, so not one event is missed.
        self._buses[record.id] = bus = _Bus()
        queue = bus.listen()
        worker = asyncio.create_task(self._work(mount, record, principal, text, files))
        self._running[record.id] = worker
        worker.add_done_callback(lambda _: self._settle(record.id))
        return self._listen(record, bus, queue, first=True)

    async def _listen(self, record: TaskRecord, bus: _Bus,
                      queue: asyncio.Queue[dict[str, Any] | None], *,
                      first: bool) -> AsyncIterator[dict[str, Any]]:
        try:
            if first:
                yield record.task
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
                if event.get("final"):
                    return
        finally:
            bus.leave(queue)

    async def _follow(self, record: TaskRecord, *,
                      first: bool) -> AsyncIterator[dict[str, Any]]:
        """The events of a task that is already under way: from the worker if
        it is here, from the store if it is on another replica."""
        bus = self._buses.get(record.id)
        if bus is not None and record.state not in wire.TERMINAL:
            queue = bus.listen()
            latest = await self.store.load(record.id) or record
            async for event in self._listen(latest, bus, queue, first=first):
                yield event
            return
        if first:
            yield record.task
        state = record.state
        deadline = time.monotonic() + self.task_timeout + 60
        while state not in wire.TERMINAL and time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            latest = await self.store.load(record.id)
            if latest is None:
                return
            if latest.state != state:
                state = latest.state
                if state in wire.TERMINAL:
                    for artifact in latest.task.get("artifacts") or []:
                        yield wire.artifact_event(latest.task, artifact, last=True)
                yield wire.status_event(latest.task, final=state in wire.TERMINAL)

    # ---- the work ---------------------------------------------------------------
    async def _work(self, mount: _Mount, record: TaskRecord, principal: Principal,
                    text: str, files: list[MediaBlock]) -> None:
        """Run the agent on one task, and write down how it went."""
        task_id, context_id = record.id, record.task["contextId"]
        bus = self._buses.get(task_id) or _Bus()
        answer_id = wire.new_uuid()
        result: RunResult | None = None
        state, note, streamed = "failed", "", ""
        try:
            async with self._gate:
                working = await self.store.change(task_id, _to("working"))
                if working is None or working.cancel_requested:
                    raise asyncio.CancelledError
                bus.publish(wire.status_event(working.task))
                result, streamed = await asyncio.wait_for(
                    self._run(mount, working, principal, text, files, bus, answer_id),
                    self.task_timeout)
            if result.stop_reason == "stopped":
                state, note = "canceled", result.error or "stopped"
            elif result.error:
                state, note = "failed", result.error
            else:
                state = "completed"
        except asyncio.CancelledError:
            state = "canceled"
            note = ("The server shut down before this task finished."
                    if self._closing else "Canceled.")
        except (TimeoutError, asyncio.TimeoutError):
            note = f"The task ran longer than {self.task_timeout:g}s and was stopped."
        except Exception as exc:
            note = str(exc) if isinstance(exc, HarnessError) else (
                f"{type(exc).__name__}: {exc}")

        artifacts = (wire.artifacts_of(result, answer_id=answer_id)
                     if result is not None and state == "completed" else [])

        def finish(held: TaskRecord) -> bool | None:
            if held.state in wire.TERMINAL:
                return False                # already settled, by a cancel say
            held.task["artifacts"] = artifacts
            held.task["status"] = wire.status(state, note, task_id=task_id,
                                              context_id=context_id)
            if result is not None and result.output and state == "completed":
                held.task["history"].append(wire.message(
                    "agent", result.output, task_id=task_id, context_id=context_id))
            if result is not None and result.agent:
                held.task.setdefault("metadata", {})["agent"] = result.agent
            held.updated = time.time()
            return None

        try:
            # Shielded: a task that is being cancelled still writes down that it was.
            final = await asyncio.shield(self.store.change(task_id, finish))
        except Exception as exc:
            final = None
            self._audit(mount, "a2a_task", task_id, "error", reason=str(exc)[:300])
        if final is None:
            return
        self.stats[final.state] = self.stats.get(final.state, 0) + 1
        self._audit(mount, "a2a_task", task_id, final.state,
                    user=principal.user_id, tenant=principal.tenant_id,
                    context=context_id)
        if final.state == "completed":
            answer = next((a for a in final.task["artifacts"]
                           if a.get("name") == wire.ANSWER), None)
            for artifact in final.task["artifacts"]:
                if artifact is answer and streamed:
                    # Already sent as it was written. If what was kept differs —
                    # a guardrail redacted it, a mode added its sources — the
                    # whole of it replaces what was streamed.
                    same = wire.text_of(artifact["parts"][:1]) == streamed
                    closing = ({**artifact, "parts": [wire.text_part("")]}
                               if same else artifact)
                    bus.publish(wire.artifact_event(final.task, closing, append=same,
                                                    last=True))
                else:
                    bus.publish(wire.artifact_event(final.task, artifact, last=True))
        bus.publish(wire.status_event(final.task, final=True))
        if final.push and final.state in wire.TERMINAL:
            self._notify(final)

    async def _run(self, mount: _Mount, record: TaskRecord, principal: Principal,
                   text: str, files: list[MediaBlock], bus: _Bus,
                   answer_id: str) -> tuple[RunResult, str]:
        agent = self._agent_for(mount, record.task["contextId"], principal)
        # The conversation is a session of the agent's own, found by the
        # context id — on whichever replica the last turn ran.
        session_id = _context_session(mount, record.task["contextId"])
        try:
            session: Any = await agent.harness.sessions.load(session_id)
        except ConfigurationError:
            session = Session(id=session_id, agent=agent.name)
        result: RunResult | None = None
        streamed: list[str] = []
        async for event in agent.stream(text or "(see the attached files)",
                                        session=session, attachments=files):
            if event.type == "text" and event.text:
                bus.publish(wire.artifact_event(
                    record.task, {"artifactId": answer_id, "name": wire.ANSWER,
                                  "parts": [wire.text_part(event.text)]},
                    append=bool(streamed)))
                streamed.append(event.text)
            elif event.type == "handoff":
                streamed.clear()             # another agent answers from here
            elif event.type == "step_start" and event.step > 1:
                # Between steps: has another replica been asked to cancel this?
                held = await self.store.load(record.id)
                if held is not None and held.cancel_requested:
                    raise asyncio.CancelledError
            elif event.type == "run_end":
                result = event.data.get("result")
        if result is None:
            raise HarnessError("the agent produced no result")
        return result, "".join(streamed)

    def _audit(self, mount: _Mount, action: str, target: str, decision: str,
               **detail: Any) -> None:
        harness = mount.agent.harness if mount.agent is not None else self.harness
        if harness is not None:
            harness.audit.record(mount.name, action, target=target, decision=decision,
                                 **detail)

    # ---- tasks/get, tasks/cancel ----------------------------------------------
    async def _get(self, mount: _Mount, params: dict[str, Any],
                   principal: Principal) -> dict[str, Any]:
        record = await self._task(params.get("id"), principal)
        return wire.trimmed(record.task, params.get("historyLength"))

    async def _cancel(self, mount: _Mount, params: dict[str, Any],
                      principal: Principal) -> dict[str, Any]:
        record = await self._task(params.get("id"), principal)
        if record.state in wire.TERMINAL:
            raise A2AError(f"task {record.id} is already {record.state}",
                           code=wire.TASK_NOT_CANCELABLE)
        worker = self._running.get(record.id)
        if worker is not None:
            worker.cancel()
            await asyncio.wait({worker}, timeout=10)
            return (await self.store.load(record.id) or record).task

        # Another replica has it. Mark it; that replica stops at its next step.
        def ask(held: TaskRecord) -> bool | None:
            if held.state in wire.TERMINAL:
                return False
            held.cancel_requested = True
            if held.state == "submitted":
                held.task["status"] = wire.status("canceled", "Canceled.")
                held.updated = time.time()
            return None

        changed = await self.store.change(record.id, ask)
        return (changed or record).task

    # ---- push notifications ---------------------------------------------------
    def _push_checked(self, config: Any) -> dict[str, Any]:
        allowed = self.push_notifications
        if not allowed:
            raise A2AError("this agent does not send push notifications",
                           code=wire.PUSH_NOT_SUPPORTED)
        if not isinstance(config, dict) or not isinstance(config.get("url"), str):
            raise A2AError("a push notification config needs a `url`",
                           code=wire.INVALID_PARAMS)
        url = config["url"]
        if not (allowed(url) if callable(allowed) else _public_https(url)):
            raise A2AError(f"push notifications may not be sent to {url!r}",
                           code=wire.INVALID_PARAMS)
        return {**config, "id": config.get("id") or wire.new_uuid()}

    async def _push_set(self, mount: _Mount, params: dict[str, Any],
                        principal: Principal) -> dict[str, Any]:
        record = await self._task(params.get("taskId"), principal)
        config = self._push_checked(params.get("pushNotificationConfig"))

        def add(held: TaskRecord) -> None:
            held.push = [c for c in held.push if c.get("id") != config["id"]] + [config]

        await self.store.change(record.id, add)
        return {"taskId": record.id, "pushNotificationConfig": config}

    async def _push_get(self, mount: _Mount, params: dict[str, Any],
                        principal: Principal) -> dict[str, Any]:
        record = await self._task(params.get("id"), principal)
        wanted = params.get("pushNotificationConfigId")
        for config in record.push:
            if wanted is None or config.get("id") == wanted:
                return {"taskId": record.id, "pushNotificationConfig": config}
        raise A2AError("no push notification config for this task",
                       code=wire.TASK_NOT_FOUND)

    async def _push_list(self, mount: _Mount, params: dict[str, Any],
                         principal: Principal) -> list[dict[str, Any]]:
        record = await self._task(params.get("id"), principal)
        return [{"taskId": record.id, "pushNotificationConfig": c} for c in record.push]

    async def _push_delete(self, mount: _Mount, params: dict[str, Any],
                           principal: Principal) -> None:
        record = await self._task(params.get("id"), principal)
        wanted = params.get("pushNotificationConfigId")

        def drop(held: TaskRecord) -> None:
            held.push = [c for c in held.push if c.get("id") != wanted]

        await self.store.change(record.id, drop)

    def _notify(self, record: TaskRecord) -> None:
        job = asyncio.create_task(self._deliver(record))
        self._pushes.add(job)
        job.add_done_callback(self._pushes.discard)

    async def _deliver(self, record: TaskRecord) -> None:
        """Tell each webhook the task is over. Tried three times, then dropped:
        the task is still there to be asked about."""
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=10.0, follow_redirects=False)
        for config in record.push:
            headers = {"content-type": "application/json"}
            if config.get("token"):
                headers["X-A2A-Notification-Token"] = str(config["token"])
            credentials = (config.get("authentication") or {}).get("credentials")
            if credentials:
                headers["Authorization"] = f"Bearer {credentials}"
            for attempt in range(3):
                try:
                    response = await self._http.post(config["url"], json=record.task,
                                                     headers=headers)
                    if response.status_code < 500:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5 * 2 ** attempt)

    # ------------------------------------------------------------------
    # health, and going away
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        running = len(self._running)
        return {"status": "draining" if self._closing else "ok",
                "agents": sorted(self.mounts),
                "running": min(running, self.max_concurrency),
                "queued": max(0, running - self.max_concurrency),
                "capacity": self.max_concurrency + self.max_queue, **self.stats}

    async def aclose(self, *, drain: float = 0.0) -> None:
        """Stop taking work. Give what is running `drain` seconds to finish,
        then cancel it — each task records that it was cancelled."""
        self._closing = True
        workers = list(self._running.values())
        if workers and drain > 0:
            await asyncio.wait(workers, timeout=drain)
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self._pushes:
            await asyncio.gather(*self._pushes, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # HTTP: the server is an ASGI application
    # ------------------------------------------------------------------
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.aclose(drain=5.0)
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        path = scope.get("path", "/").rstrip("/") or "/"
        method = scope.get("method", "GET").upper()

        async def reply(status: int, body: Any, extra: Iterable[tuple[bytes, bytes]] = ()
                        ) -> None:
            data = json.dumps(body, ensure_ascii=False, default=str).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode()), *extra]})
            await send({"type": "http.response.body", "body": data})

        if method == "GET" and path == "/healthz":
            await reply(503 if self._closing else 200, self.health())
            return
        mount, rest = self._route(path)
        if mount is None:
            if method == "GET" and path == "/" and len(self.mounts) > 1:
                base = self.url or _base_url(scope, headers)
                await reply(200, {"agents": {
                    name: f"{base}{m.prefix}{wire.CARD_PATH}"
                    for name, m in self.mounts.items()}})
                return
            await reply(404, {"error": f"nothing at {path}"})
            return
        if rest in (wire.CARD_PATH, wire.LEGACY_CARD_PATH):
            if method != "GET":
                await reply(405, {"error": "the agent card is read with GET"})
                return
            await reply(200, self.card(mount.name, _base_url(scope, headers)),
                        [(b"cache-control", b"public, max-age=300")])
            return
        if rest != "/":
            await reply(404, {"error": f"nothing at {path}"})
            return
        if method != "POST":
            await reply(405, {"error": "A2A requests are sent with POST"},
                        [(b"allow", b"POST")])
            return

        try:
            principal = await self.authenticate(headers)
            body = await _read_body(receive, self.max_body_bytes)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                await reply(200, {"jsonrpc": "2.0", "id": None, "error": {
                    "code": wire.PARSE_ERROR, "message": "the body is not JSON"}})
                return
            answer = await self.handle(payload, principal=principal, agent=mount.name)
        except A2AError as exc:
            extra = []
            if exc.status == 401:
                extra.append((b"www-authenticate", b'Bearer realm="a2a"'))
            if exc.retry_after is not None:
                extra.append((b"retry-after", f"{exc.retry_after:g}".encode()))
            await reply(exc.status if exc.status != 200 else 400,
                        {"jsonrpc": "2.0", "id": None, "error": exc.body()}, extra)
            return
        except Exception as exc:
            self._audit(mount, "a2a_request", path, "error",
                        reason=f"{type(exc).__name__}: {exc}"[:300])
            await reply(500, {"jsonrpc": "2.0", "id": None, "error": {
                "code": wire.INTERNAL_ERROR, "message": "internal error"}})
            return
        if isinstance(answer, dict):
            await reply(200, answer)
            return
        await self._sse(answer, receive, send)

    def _route(self, path: str) -> tuple[_Mount | None, str]:
        """Which agent a path belongs to, and what is left of the path."""
        best: _Mount | None = None
        for mount in self.mounts.values():
            if (path == mount.prefix or path.startswith(f"{mount.prefix}/")) and (
                    best is None or len(mount.prefix) > len(best.prefix)):
                best = mount
        if best is None or (not best.prefix and len(self.mounts) > 1):
            return None, path
        return best, path[len(best.prefix):] or "/"

    async def _sse(self, events: AsyncIterator[dict[str, Any]], receive: Any,
                   send: Any) -> None:
        """Send events as they come; stop sending when the caller hangs up.
        The task itself carries on — it can be asked about, or re-subscribed to."""
        await send({"type": "http.response.start", "status": 200,
                    "headers": list(_SSE_HEADERS)})

        async def gone() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        hung_up = asyncio.create_task(gone())
        iterator = events.__aiter__()
        pending: asyncio.Task[Any] | None = None
        try:
            while True:
                pending = pending or asyncio.ensure_future(iterator.__anext__())
                done, _ = await asyncio.wait({pending, hung_up}, timeout=self.keepalive,
                                             return_when=asyncio.FIRST_COMPLETED)
                if hung_up in done:
                    return
                if not done:
                    # Nothing to say yet: a comment keeps a proxy from closing it.
                    await send({"type": "http.response.body", "body": b": keep-alive\n\n",
                                "more_body": True})
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    break
                finally:
                    pending = None
                data = json.dumps(event, ensure_ascii=False, default=str)
                await send({"type": "http.response.body",
                            "body": f"data: {data}\n\n".encode(), "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        finally:
            hung_up.cancel()
            if pending is not None:
                pending.cancel()
            await asyncio.gather(hung_up, *([pending] if pending else []),
                                 return_exceptions=True)
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()

    # ------------------------------------------------------------------
    # a server of its own, for development
    # ------------------------------------------------------------------
    async def serve(self, host: str = "127.0.0.1", port: int = 8000, *,
                    ready: Callable[[str], Any] | None = None) -> None:
        """Serve until cancelled, on a small HTTP server of its own.

        Enough to develop against and to put behind a proxy on one machine. For
        production run the server as the ASGI app it is, under a server built
        for it: `uvicorn myapp:server --workers 8`.
        """
        listener = await asyncio.start_server(
            lambda r, w: _connection(self, r, w, host, port), host, port)
        bound = listener.sockets[0].getsockname()
        if ready is not None:
            ready(f"http://{bound[0]}:{bound[1]}")
        try:
            async with listener:
                await listener.serve_forever()
        finally:
            await self.aclose()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _to(state: str) -> Callable[[TaskRecord], Any]:
    def edit(record: TaskRecord) -> bool | None:
        if record.state in wire.TERMINAL:
            return False
        record.task["status"] = wire.status(state)
        record.updated = time.time()
        return None
    return edit


def _without_bytes(part: dict[str, Any]) -> dict[str, Any]:
    file = part.get("file")
    if isinstance(file, dict) and "bytes" in file:
        return {**part, "file": {k: v for k, v in file.items() if k != "bytes"}}
    return part


def _context_session(mount: _Mount, context_id: str) -> str:
    return f"a2a_{mount.name}_{context_id}"


def _acting_for(agent: Any, principal: Principal) -> Any:
    """This agent, built again for one conversation and the caller it is for.

    Its harness, provider, tools and sub-agents are the same; what it keeps
    between runs — its thread, its session memory — is its own.
    """
    from ..agent import Agent
    from ..memory.trace import Trace

    kwargs = dict(agent._base_kwargs)
    if principal.user_id or principal.tenant_id:
        base = kwargs.get("trace")
        held = Trace.of(base).model_dump(exclude_none=True) if base is not None else {}
        kwargs["trace"] = {**held, "user_id": principal.user_id,
                           "tenant_id": principal.tenant_id}
    if not isinstance(kwargs.get("workspace"), (bool, type(None))):
        kwargs["workspace"] = agent.workspace        # one sandbox, not one each
    return Agent(agent.name, version=agent.version,
                 versions=dict(agent._versions) or None, **kwargs)


def _public_https(url: str) -> bool:
    """A webhook the server will call unasked: https, and not this network."""
    import ipaddress
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host or host == "localhost" \
            or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


def _base_url(scope: dict[str, Any], headers: dict[str, str]) -> str:
    scheme = headers.get("x-forwarded-proto", scope.get("scheme", "http")).split(",")[0]
    host = headers.get("x-forwarded-host") or headers.get("host")
    if not host:
        server = scope.get("server") or ("localhost", 80)
        host = f"{server[0]}:{server[1]}"
    return f"{scheme.strip()}://{host.split(',')[0].strip()}{scope.get('root_path', '')}"


async def _read_body(receive: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise A2AError("the request was cut off", code=wire.INVALID_REQUEST,
                           status=400)
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            raise A2AError(f"the request is larger than {limit} bytes",
                           code=wire.INVALID_REQUEST, status=413)
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)


_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
            405: "Method Not Allowed", 413: "Payload Too Large",
            429: "Too Many Requests", 500: "Internal Server Error",
            503: "Service Unavailable"}


async def _connection(app: Any, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter, host: str, port: int) -> None:
    """One HTTP/1.1 request on one connection, handed to the ASGI app."""
    try:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                TimeoutError, asyncio.TimeoutError):
            return
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, _ = lines[0].split(" ", 2)
        except ValueError:
            return
        headers = [(name.strip().lower().encode("latin-1"), value.strip().encode("latin-1"))
                   for name, _, value in (line.partition(":") for line in lines[1:] if line)]
        length = int(dict(headers).get(b"content-length", b"0") or 0)
        body = await reader.readexactly(length) if length else b""
        path, _, query = target.partition("?")
        scope = {"type": "http", "http_version": "1.1", "method": method, "path": path,
                 "query_string": query.encode(), "headers": headers, "scheme": "http",
                 "server": (host, port), "root_path": ""}
        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            # Nothing more is coming; a read that ends means they hung up.
            await reader.read(1)
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                out = [f"HTTP/1.1 {status} {_REASONS.get(status, 'OK')}"]
                out += [f"{k.decode('latin-1')}: {v.decode('latin-1')}"
                        for k, v in message.get("headers") or []]
                out.append("connection: close")
                writer.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1"))
            elif message["type"] == "http.response.body":
                writer.write(message.get("body", b""))
            await writer.drain()

        await app(scope, receive, send)
    except (ConnectionError, asyncio.CancelledError):
        pass
    except Exception:
        try:
            writer.write(b"HTTP/1.1 500 Internal Server Error\r\ncontent-length: 0\r\n"
                         b"connection: close\r\n\r\n")
            await writer.drain()
        except ConnectionError:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

