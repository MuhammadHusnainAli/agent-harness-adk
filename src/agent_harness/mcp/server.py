"""Expose agents as an MCP server, so any MCP client can use them as tools.

    from agent_harness.mcp import MCPAgentServer

    server = MCPAgentServer(
        [billing, orders],
        api_keys={"billing": "sk-billing-…", "orders": "sk-orders-…"},
    )
    app = server                        # an ASGI app:  uvicorn myapp:server
    await server.serve(port=8000)       # or the built-in one, to develop against
    await server.serve_stdio()          # or over stdio, for a desktop client

Each agent is one tool, named after it: the client sends a `task`, the agent
runs — with its own model, tools, guardrails and budget — and its answer comes
back. Claude Desktop, Cursor, another framework's agent: whatever speaks MCP.

**Every agent can have its own key.** A key opens only the agents it was issued
for: a caller holding the billing key sees one tool, and is told the orders
agent does not exist. `api_key=` is a key that opens all of them. Keys are sent
as `Authorization: Bearer …` or `X-API-Key: …`, and are compared in constant
time. An agent is also served alone at `/<name>/mcp`, for a client that should
be pointed at exactly one.

The transport is MCP's streamable HTTP, kept stateless — no session lives in
the process, so any replica answers any request.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import re
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

from ..errors import ConfigurationError, HarnessError
from ..runtime.session import Session
from ..types import RunResult

__all__ = ["MCPAgentServer", "PROTOCOL_VERSIONS"]

#: The MCP revisions this server speaks, newest first.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_BUSY = -32000

_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_HIDDEN_TAGS = {"delegation", "memory", "mode", "handoff"}


class _Refused(Exception):
    """A request refused before any JSON-RPC was read: 401, 403, 404."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class _Key:
    """One API key: which agents it opens, and whose calls they are."""

    id: str                                   # what is logged — never the key
    agents: set[str] | None = None            # None: every agent
    user_id: str | None = None
    tenant_id: str | None = None


@dataclass
class _Access:
    """What one request may reach."""

    agents: list[str]
    key_id: str = ""
    user_id: str | None = None
    tenant_id: str | None = None

    @property
    def owner(self) -> str:
        return f"{self.tenant_id or ''}/{self.user_id or ''}"


@dataclass
class _Exposed:
    """One MCP tool: an agent, or one of an agent's own tools."""

    name: str
    agent: str
    tool: str = ""                            # "" — the agent itself
    definition: dict[str, Any] = field(default_factory=dict)


def _key_id(key: str) -> str:
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]


class MCPAgentServer:
    """Agents, served as MCP tools. It is itself an ASGI application.

    Args:
        agents: an agent, several, or a mapping of name → agent.
        api_key: a key — or several — that opens every agent.
        api_keys: agent name → its own key, or keys. A key opens only the
            agents it is listed under. `None` for an agent leaves that one open
            to callers with no key at all.
        auth: instead of keys, a function from the request headers to who is
            calling: None to refuse, True to allow, or a mapping with `user_id`,
            `tenant_id` and optionally `agents` (the names they may use).
        expose_tools: also serve the agents' own tools — True for all of them,
            or globs of tool names. They are called through the agent, so its
            permission gate, hooks and audit trail still apply.
        conversations: let a caller pass `conversation_id` to carry a
            conversation on from one call to the next.
        name, version, instructions: what the server tells a client it is.
        max_concurrency, max_queue: how many calls run at once, and how many
            more may wait before callers are told the server is busy.
        timeout: the longest one call may run, in seconds.
        isolate: give every caller's conversation its own instance of the agent.
        allowed_origins: if set, a browser request from any other `Origin` is
            refused — the protection MCP asks of a server on localhost.

    A key may be written as a mapping to say whose calls it makes:
    `{"key": "sk-…", "user_id": "ada", "tenant_id": "acme"}`. Conversations
    belong to that user; without one, to the key itself.
    """

    def __init__(self, agents: Any, *, api_key: Any = None,
                 api_keys: dict[str, Any] | None = None, auth: Any = None,
                 expose_tools: bool | Iterable[str] = False,
                 conversations: bool = True, name: str = "agent-harness",
                 version: str = "1.0.0", instructions: str | None = None,
                 max_concurrency: int = 32, max_queue: int = 128,
                 timeout: float = 600.0, isolate: bool = True,
                 allowed_origins: Iterable[str] | None = None,
                 max_body_bytes: int = 4_000_000, max_agents: int = 1024,
                 keepalive: float = 15.0) -> None:
        if isinstance(agents, dict):
            self.agents: dict[str, Any] = dict(agents)
        else:
            listed = agents if isinstance(agents, (list, tuple)) else [agents]
            self.agents = {a.name: a for a in listed}
        if not self.agents:
            raise ConfigurationError("an MCP server needs at least one agent")
        for label, agent in self.agents.items():
            if not (hasattr(agent, "stream") and hasattr(agent, "harness")):
                raise ConfigurationError(
                    f"{label}: an MCP server serves Agents — got "
                    f"{type(agent).__name__}")
        self.name, self.version, self.instructions = name, version, instructions
        self.conversations = conversations
        self.timeout = timeout
        self.isolate = isolate
        self.max_concurrency, self.max_queue = max(1, max_concurrency), max(0, max_queue)
        self.max_body_bytes, self.max_agents = max_body_bytes, max_agents
        self.keepalive = keepalive
        self.allowed_origins = (set(allowed_origins) if allowed_origins is not None
                                else None)
        self.auth = auth

        # ---- keys ---------------------------------------------------------
        self._keys: dict[str, _Key] = {}
        #: Agents anyone may call, key or no key.
        self.public: set[str] = set()
        for entry in _several(api_key):
            self.add_key(entry)
        for agent_name, entries in (api_keys or {}).items():
            if agent_name not in self.agents:
                raise ConfigurationError(
                    f"api_keys names {agent_name!r}, which is not served; the "
                    f"agents are: {', '.join(self.agents)}")
            if entries is None:
                self.public.add(agent_name)
            for entry in _several(entries):
                self.add_key(entry, agent=agent_name)
        self.open = not self._keys and auth is None and api_keys is None
        if not self.open and auth is None:
            master = any(k.agents is None for k in self._keys.values())
            locked = [n for n in self.agents if n not in self.public and not master
                      and not any(n in (k.agents or ()) for k in self._keys.values())]
            if locked:
                raise ConfigurationError(
                    f"{', '.join(locked)} would be served with no key that opens "
                    "it. Give each a key in api_keys={…}, pass api_key= for one "
                    "that opens every agent, or list it with None to leave it "
                    "open")

        # ---- tools ----------------------------------------------------------
        self._tools: dict[str, _Exposed] = {}
        patterns = (None if expose_tools is True
                    else list(expose_tools) if expose_tools else [])
        for agent_name, agent in self.agents.items():
            self._expose(_Exposed(self._unique(_slug(agent_name)), agent_name))
            for tool in agent.tools:
                if patterns == [] or tool.tags & _HIDDEN_TAGS:
                    continue
                if patterns is not None and not any(fnmatch(tool.name, p)
                                                    for p in patterns):
                    continue
                self._expose(_Exposed(
                    self._unique(_slug(f"{agent_name}_{tool.name}")), agent_name,
                    tool=tool.name))

        self._gate = asyncio.Semaphore(self.max_concurrency)
        self._waiting = 0
        self._running = 0
        self._instances: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._calls: dict[tuple[str, Any], asyncio.Task[Any]] = {}
        self._closing = False
        self.stats = {"calls": 0, "errors": 0, "refused": 0, "busy": 0}

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------
    def add_key(self, key: Any, *, agent: str | None = None) -> str:
        """Issue a key: for one agent, or (with no agent) for all of them.
        Returns the id it is logged under. Safe to call while serving."""
        entry = {"key": key} if isinstance(key, str) else dict(key or {})
        secret = entry.get("key")
        if not isinstance(secret, str) or len(secret) < 8:
            raise ConfigurationError(
                "an API key must be a string of at least 8 characters"
                + (f" (for {agent})" if agent else ""))
        if agent is not None and agent not in self.agents:
            raise ConfigurationError(f"no agent {agent!r} is served")
        held = self._keys.get(secret)
        if held is None:
            held = self._keys[secret] = _Key(
                id=_key_id(secret), agents=set() if agent else None,
                user_id=entry.get("user_id"), tenant_id=entry.get("tenant_id"))
        if agent is None:
            held.agents = None
        elif held.agents is not None:
            held.agents.add(agent)
        self.open = False
        return held.id

    def revoke_key(self, key: str) -> bool:
        """Stop a key working, now. True if there was such a key."""
        return self._keys.pop(key, None) is not None

    async def _who(self, headers: dict[str, str]) -> _Access:
        """Which agents this request may reach. Raises `_Refused` (401)."""
        everything = list(self.agents)
        if self.auth is not None:
            found = self.auth(headers)
            if inspect.isawaitable(found):
                found = await found
            if not found:
                raise _Refused(401, "not authorised")
            claims = found if isinstance(found, dict) else {}
            allowed = claims.get("agents")
            return _Access(
                agents=[a for a in everything if allowed is None or a in allowed],
                key_id="auth", user_id=claims.get("user_id"),
                tenant_id=claims.get("tenant_id"))
        if self.open:
            return _Access(agents=everything)
        scheme, _, bearer = headers.get("authorization", "").partition(" ")
        offered = (bearer.strip() if scheme.lower() == "bearer" else "") \
            or headers.get("x-api-key", "").strip()
        if not offered:
            if self.public:
                return _Access(agents=[a for a in everything if a in self.public])
            raise _Refused(401, "an API key is required")
        match: _Key | None = None
        for secret, key in self._keys.items():
            # Every key is compared, in constant time, so the time this takes
            # says nothing about which one nearly matched.
            if hmac.compare_digest(secret.encode(), offered.encode()):
                match = key
        if match is None:
            raise _Refused(401, "that API key is not valid")
        return _Access(
            agents=[a for a in everything
                    if match.agents is None or a in match.agents or a in self.public],
            key_id=match.id, user_id=match.user_id or match.id,
            tenant_id=match.tenant_id)

    # ------------------------------------------------------------------
    # what is offered
    # ------------------------------------------------------------------
    def _unique(self, name: str) -> str:
        candidate, n = name[:64], 2
        while candidate in self._tools:
            suffix = f"_{n}"
            candidate, n = f"{name[:64 - len(suffix)]}{suffix}", n + 1
        return candidate

    def _expose(self, exposed: _Exposed) -> None:
        agent = self.agents[exposed.agent]
        if exposed.tool:
            tool = agent.tools.get(exposed.tool)
            exposed.definition = {
                "name": exposed.name, "description": tool.description,
                "inputSchema": tool.parameters or {"type": "object", "properties": {}},
            }
        else:
            properties: dict[str, Any] = {"task": {
                "type": "string",
                "description": "What you want done, stated in full. The agent "
                               "does not see your conversation — include "
                               "everything it needs."}}
            said = (f"Ask the {exposed.agent} agent. {agent.description.rstrip('.')}. "
                    "It runs the task with its own tools and answers in text.")
            if self.conversations:
                properties["conversation_id"] = {
                    "type": "string",
                    "description": "Optional. Any id of your choosing: calls that "
                                   "share it are one conversation, and the agent "
                                   "remembers the earlier ones."}
            exposed.definition = {
                "name": exposed.name, "title": exposed.agent, "description": said,
                "inputSchema": {"type": "object", "properties": properties,
                                "required": ["task"]},
            }
        self._tools[exposed.name] = exposed

    def tools(self, agents: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """The tool definitions a caller who may reach `agents` is shown."""
        allowed = set(self.agents if agents is None else agents)
        return [t.definition for t in self._tools.values() if t.agent in allowed]

    # ------------------------------------------------------------------
    # JSON-RPC
    # ------------------------------------------------------------------
    async def handle(self, payload: Any, *, access: _Access | None = None,
                     notify: Callable[[dict[str, Any]], Any] | None = None
                     ) -> dict[str, Any] | None:
        """Answer one JSON-RPC message. None for a notification, which has no
        answer. This is the protocol with no transport around it."""
        access = access or _Access(agents=list(self.agents))
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        request_id, method = payload.get("id"), payload.get("method")
        if not isinstance(method, str):
            return None                       # a response to something we never asked
        params = payload.get("params") or {}
        if "id" not in payload:
            if method == "notifications/cancelled" and isinstance(params, dict):
                running = self._calls.get((access.owner, params.get("requestId")))
                if running is not None:
                    running.cancel()
            return None
        try:
            if not isinstance(params, dict):
                raise _RpcError(INVALID_PARAMS, "params must be an object")
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tools(access.agents)}
            elif method == "tools/call":
                result = await self._tracked(request_id, params, access, notify)
            elif method in ("resources/list", "prompts/list",
                            "resources/templates/list"):
                key = method.split("/")[0] if "templates" not in method \
                    else "resourceTemplates"
                result = {key: []}
            else:
                raise _RpcError(METHOD_NOT_FOUND, f"method {method!r} is not supported")
        except _RpcError as exc:
            return _error(request_id, exc.code, str(exc))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        result: dict[str, Any] = {
            "protocolVersion": asked if asked in PROTOCOL_VERSIONS
            else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
        }
        if self.instructions:
            result["instructions"] = self.instructions
        return result

    async def _tracked(self, request_id: Any, params: dict[str, Any], access: _Access,
                       notify: Callable[[dict[str, Any]], Any] | None) -> dict[str, Any]:
        """A call that can be cancelled by the `notifications/cancelled` that
        names it."""
        key = (access.owner, request_id)
        current = asyncio.current_task()
        if current is not None:
            self._calls[key] = current
        try:
            return await self._call(params, access, notify)
        except asyncio.CancelledError:
            if self._calls.get(key) is not current:
                raise
            return _text("Cancelled.", error=True)
        finally:
            if self._calls.get(key) is current:
                self._calls.pop(key, None)

    async def _call(self, params: dict[str, Any], access: _Access,
                    notify: Callable[[dict[str, Any]], Any] | None) -> dict[str, Any]:
        name = params.get("name")
        exposed = self._tools.get(name) if isinstance(name, str) else None
        # A tool the caller's key does not open is, to them, not there.
        if exposed is None or exposed.agent not in access.agents:
            raise _RpcError(INVALID_PARAMS, f"Unknown tool: {name!r}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _RpcError(INVALID_PARAMS, "arguments must be an object")
        if self._closing:
            raise _RpcError(SERVER_BUSY, "the server is shutting down")
        if self._gate.locked() and self._waiting >= self.max_queue:
            self.stats["busy"] += 1
            raise _RpcError(SERVER_BUSY, "the server is at capacity; try again shortly")

        token = (params.get("_meta") or {}).get("progressToken")
        self.stats["calls"] += 1
        agent = self.agents[exposed.agent]
        self._waiting += 1
        try:
            await self._gate.acquire()
        finally:
            self._waiting -= 1
        self._running += 1
        try:
            if exposed.tool:
                outcome = await asyncio.wait_for(
                    self._instance(exposed.agent, access, "").call_tool(
                        exposed.tool, arguments), self.timeout)
                answer = _text(outcome.content, error=outcome.is_error)
            else:
                answer = await asyncio.wait_for(
                    self._ask(exposed, arguments, access, token, notify), self.timeout)
        except (TimeoutError, asyncio.TimeoutError):
            answer = _text(f"{exposed.name} ran longer than {self.timeout:g}s and "
                           "was stopped.", error=True)
        except _RpcError:
            raise
        except Exception as exc:
            answer = _text(str(exc) if isinstance(exc, HarnessError)
                           else f"{type(exc).__name__}: {exc}", error=True)
        finally:
            self._running -= 1
            self._gate.release()
        failed = bool(answer.get("isError"))
        self.stats["errors"] += failed
        agent.harness.audit.record(
            exposed.agent, "mcp_call", target=exposed.name,
            decision="error" if failed else "ok", key=access.key_id,
            user=access.user_id, tenant=access.tenant_id)
        return answer

    async def _ask(self, exposed: _Exposed, arguments: dict[str, Any], access: _Access,
                   token: Any, notify: Callable[[dict[str, Any]], Any] | None
                   ) -> dict[str, Any]:
        """Run the agent on one task and hand back what it said."""
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            raise _RpcError(INVALID_PARAMS, "`task` is required: what should the "
                                            "agent do?")
        conversation = arguments.get("conversation_id") if self.conversations else None
        if conversation is not None and (not isinstance(conversation, str)
                                         or not 0 < len(conversation) <= 128):
            raise _RpcError(INVALID_PARAMS, "`conversation_id` is a short string")
        agent = self._instance(exposed.agent, access, conversation or "")
        kwargs: dict[str, Any] = {}
        if conversation:
            # A session of the agent's own, found by the caller and the id they
            # chose — on whichever replica the last call ran.
            digest = hashlib.sha256(
                f"{access.owner}\0{conversation}".encode()).hexdigest()[:24]
            session_id = f"mcp_{_slug(exposed.agent)}_{digest}"
            try:
                kwargs["session"] = await agent.harness.sessions.load(session_id)
            except ConfigurationError:
                kwargs["session"] = Session(id=session_id, agent=agent.name)
        else:
            kwargs["messages"] = []            # one task, nothing carried over

        result: RunResult | None = None
        done = 0
        async for event in agent.stream(task, **kwargs):
            if event.type == "run_end":
                result = event.data.get("result")
            elif token is not None and notify is not None and event.type in (
                    "step_start", "tool_call", "handoff"):
                done += 1
                said = {"step_start": f"step {event.step}",
                        "tool_call": f"using {event.text}",
                        "handoff": f"handed to {event.text}"}[event.type]
                notify({"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": token, "progress": done,
                                   "message": said}})
        if result is None:
            raise HarnessError("the agent produced no result")
        if result.error:
            answer = _text(result.error, error=True)
        else:
            answer = _text(result.output)
            if result.data is not None:
                data = (result.data.model_dump(mode="json")
                        if hasattr(result.data, "model_dump") else result.data)
                if isinstance(data, dict):
                    answer["structuredContent"] = data
            for produced in result.artifacts:
                if produced.content and len(produced.content) <= 200_000:
                    answer["content"].append({"type": "resource", "resource": {
                        "uri": f"artifact:///{produced.name}",
                        "mimeType": produced.media_type or "text/plain",
                        "text": produced.content}})
        answer["_meta"] = {"agent": result.agent, "steps": result.steps,
                           **({"conversation_id": conversation} if conversation else {})}
        return answer

    def _instance(self, name: str, access: _Access, conversation: str) -> Any:
        """The agent for this caller's conversation: its own instance, acting
        for them, so nothing one caller said is in another's memory."""
        agent = self.agents[name]
        if not self.isolate:
            return agent
        key = (name, f"{access.owner}/{conversation}")
        held = self._instances.get(key)
        if held is not None:
            self._instances.move_to_end(key)
            return held
        from ..a2a.server import Principal, _acting_for

        built = _acting_for(agent, Principal(access.user_id, access.tenant_id))
        self._instances[key] = built
        while len(self._instances) > self.max_agents:
            self._instances.popitem(last=False)
        return built

    # ------------------------------------------------------------------
    # health, and going away
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {"status": "draining" if self._closing else "ok",
                "agents": sorted(self.agents), "tools": len(self._tools),
                "running": self._running,
                "waiting": self._waiting, "keys": len(self._keys), **self.stats}

    async def aclose(self) -> None:
        """Stop taking calls, and cancel the ones running."""
        self._closing = True
        running = list(self._calls.values())
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    # ------------------------------------------------------------------
    # HTTP: streamable, stateless
    # ------------------------------------------------------------------
    def _scope_of(self, path: str) -> str | None:
        """Which agent a path is about. "" for the endpoint that serves all;
        None for a path that is nothing."""
        path = path.rstrip("/") or "/"
        if path in ("/mcp", "/"):
            return ""
        if path.endswith("/mcp"):
            name = path[1:-len("/mcp")]
            return name if name in self.agents else None
        return None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.aclose()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        path, method = scope.get("path", "/"), scope.get("method", "GET").upper()

        async def reply(status: int, body: Any = None,
                        extra: Iterable[tuple[bytes, bytes]] = ()) -> None:
            data = b"" if body is None else json.dumps(
                body, ensure_ascii=False, default=str).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode()), *extra]})
            await send({"type": "http.response.body", "body": data})

        if method == "GET" and path.rstrip("/") == "/healthz":
            await reply(503 if self._closing else 200, self.health())
            return
        only = self._scope_of(path)
        if only is None:
            await reply(404, {"error": f"nothing at {path}"})
            return
        if method != "POST":
            # No stream of the server's own to open, and no session to end.
            await reply(405, {"error": "this MCP endpoint takes POST"},
                        [(b"allow", b"POST")])
            return
        origin = headers.get("origin")
        if self.allowed_origins is not None and origin and \
                origin not in self.allowed_origins:
            await reply(403, {"error": f"requests from {origin} are not accepted"})
            return
        try:
            access = await self._who(headers)
            if only:
                if only not in access.agents:
                    raise _Refused(403, f"this key does not open {only}")
                access.agents = [only]
        except _Refused as exc:
            self.stats["refused"] += 1
            await reply(exc.status, _error(None, INVALID_REQUEST, str(exc)),
                        [(b"www-authenticate", b'Bearer realm="mcp"')]
                        if exc.status == 401 else ())
            return

        body = await _read(receive, self.max_body_bytes)
        if body is None:
            await reply(413, _error(None, INVALID_REQUEST,
                                    f"the request is larger than "
                                    f"{self.max_body_bytes} bytes"))
            return
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            await reply(400, _error(None, PARSE_ERROR, "the body is not JSON"))
            return

        messages = payload if isinstance(payload, list) else [payload]
        streaming = ("text/event-stream" in headers.get("accept", "")
                     and any(isinstance(m, dict) and m.get("method") == "tools/call"
                             and ((m.get("params") or {}).get("_meta") or {})
                             .get("progressToken") is not None for m in messages))
        notes: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def answer_all() -> list[dict[str, Any]]:
            answers = []
            for message in messages:
                answer = await self.handle(
                    message, access=access,
                    notify=notes.put_nowait if streaming else None)
                if answer is not None:
                    answers.append(answer)
            return answers

        async def gone() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        work = asyncio.create_task(answer_all())
        hung_up = asyncio.create_task(gone())
        try:
            if streaming:
                await send({"type": "http.response.start", "status": 200, "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"), (b"x-accel-buffering", b"no")]})
            while True:
                waiting: set[asyncio.Future[Any]] = {work, hung_up}
                note = asyncio.create_task(notes.get()) if streaming else None
                if note is not None:
                    waiting.add(note)
                done, _ = await asyncio.wait(
                    waiting, timeout=self.keepalive if streaming else None,
                    return_when=asyncio.FIRST_COMPLETED)
                if note is not None and note in done:
                    await send(_sse(note.result()))
                    continue
                if note is not None:
                    note.cancel()
                if hung_up in done and work not in done:
                    work.cancel()             # nobody is waiting for the answer
                    await asyncio.gather(work, return_exceptions=True)
                    return
                if work in done:
                    break
                await send({"type": "http.response.body", "body": b": keep-alive\n\n",
                            "more_body": True})
            answers = work.result()
            if streaming:
                while not notes.empty():
                    await send(_sse(notes.get_nowait()))
                for answer in answers:
                    await send(_sse(answer))
                await send({"type": "http.response.body", "body": b""})
            elif not answers:
                await reply(202)              # only notifications: nothing to say
            else:
                await reply(200, answers if isinstance(payload, list) else answers[0])
        except Exception as exc:
            if not streaming:
                await reply(500, _error(None, INTERNAL_ERROR, "internal error"))
            next(iter(self.agents.values())).harness.audit.record(
                self.name, "mcp_request", target=path, decision="error",
                reason=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            hung_up.cancel()
            if not work.done():
                work.cancel()
            await asyncio.gather(hung_up, work, return_exceptions=True)

    async def serve(self, host: str = "127.0.0.1", port: int = 8000, *,
                    ready: Callable[[str], Any] | None = None) -> None:
        """Serve until cancelled, on a small HTTP server of its own.

        Enough to develop against. For production run the server as the ASGI
        app it is: `uvicorn myapp:server --workers 8`.
        """
        from ..a2a.server import _connection

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

    # ------------------------------------------------------------------
    # stdio: for a client that starts the server itself
    # ------------------------------------------------------------------
    async def serve_stdio(self, stdin: Any = None, stdout: Any = None, *,
                          agents: Iterable[str] | None = None) -> None:
        """Serve one client over stdin and stdout, a JSON message to a line.

        This is how a desktop client runs a server: it starts the process and
        owns both ends, so there is no key to check — the client is whoever
        could start it. `agents` narrows what it is offered.
        """
        source = stdin if stdin is not None else sys.stdin
        sink = stdout if stdout is not None else sys.stdout
        access = _Access(agents=[a for a in self.agents
                                 if agents is None or a in set(agents)],
                         key_id="stdio")
        lock = asyncio.Lock()
        running: set[asyncio.Task[Any]] = set()

        async def write(message: dict[str, Any]) -> None:
            line = json.dumps(message, ensure_ascii=False, default=str) + "\n"
            async with lock:
                await asyncio.to_thread(_put, sink, line)

        def notify(message: dict[str, Any]) -> None:
            job = asyncio.create_task(write(message))
            running.add(job)
            job.add_done_callback(running.discard)

        async def one(payload: Any) -> None:
            answer = await self.handle(payload, access=access, notify=notify)
            if answer is not None:
                await write(answer)

        try:
            while True:
                line = await asyncio.to_thread(source.readline)
                if not line:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    await write(_error(None, PARSE_ERROR, "the line is not JSON"))
                    continue
                # Each on its own, so a long task does not hold up a ping.
                job = asyncio.create_task(one(payload))
                running.add(job)
                job.add_done_callback(running.discard)
        finally:
            for job in list(running):
                job.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            await self.aclose()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _several(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    return list(value)


def _slug(name: str) -> str:
    return _NAME.sub("_", name).strip("_") or "agent"


def _text(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text or ""}], "isError": error}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _sse(message: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(message, ensure_ascii=False, default=str)
    return {"type": "http.response.body", "body": f"data: {data}\n\n".encode(),
            "more_body": True}


def _put(sink: Any, line: str) -> None:
    sink.write(line)
    sink.flush()


async def _read(receive: Any, limit: int) -> bytes | None:
    """The request body, or None if it is larger than allowed."""
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"".join(chunks)
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)
