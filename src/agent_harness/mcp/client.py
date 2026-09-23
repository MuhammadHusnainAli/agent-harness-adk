"""MCP client: connect to Model Context Protocol servers and use their tools.

    server = MCPServer(name="files", command="npx",
                       args=["-y", "@modelcontextprotocol/server-filesystem", "/data"])
    async with MCPManager([server]) as mcp:
        agent = Agent("assistant", tools=mcp.tools())

Both transports are supported: `stdio` (a subprocess speaking line-delimited
JSON-RPC) and `http` (streamable HTTP). No SDK required.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..errors import MCPError
from ..tools import Tool, render_result

__all__ = ["MCPServer", "MCPClient", "MCPManager"]

PROTOCOL_VERSION = "2025-06-18"

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)


class MCPServer(BaseModel):
    """How to reach one MCP server."""

    model_config = ConfigDict(extra="allow")

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 60.0
    prefix: bool = True          # expose tools as `<server>_<tool>`
    allowed_tools: list[str] | None = None

    def model_post_init(self, _: Any) -> None:
        if self.url and self.transport == "stdio" and not self.command:
            self.transport = "http"


class MCPClient:
    """One connection, JSON-RPC 2.0 either way."""

    def __init__(self, server: MCPServer, *,
                 http_client: httpx.AsyncClient | None = None) -> None:
        self.server = server
        self._proc: asyncio.subprocess.Process | None = None
        self._http: httpx.AsyncClient | None = http_client
        self._owns_http = http_client is None
        self._session_id: str | None = None
        self._id = 0
        self._lock = asyncio.Lock()
        self.tools_cache: list[dict[str, Any]] = []
        self.connected = False

    # ---- transport ----------------------------------------------------
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def connect(self) -> MCPClient:
        if self.connected:
            return self
        if self.server.transport == "stdio":
            if not self.server.command:
                raise MCPError(f"{self.server.name}: stdio transport needs a command")
            import os
            self._proc = await asyncio.create_subprocess_exec(
                self.server.command, *self.server.args,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **self.server.env},
            )
        else:
            if not self.server.url:
                raise MCPError(f"{self.server.name}: http transport needs a url")
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=self.server.timeout)

        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "clientInfo": {"name": "agent-harness", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized")
        self.connected = True
        return self

    async def _send_stdio(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        assert self._proc and self._proc.stdin and self._proc.stdout
        line = json.dumps(payload) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()
        if "id" not in payload:
            return None
        while True:
            raw = await asyncio.wait_for(self._proc.stdout.readline(),
                                         self.server.timeout)
            if not raw:
                err = b""
                if self._proc.stderr:
                    try:
                        err = await asyncio.wait_for(self._proc.stderr.read(2000), 0.5)
                    except _Timeout:
                        err = b""
                raise MCPError(f"{self.server.name} closed the connection: "
                               f"{err.decode(errors='replace')[:500]}")
            try:
                message = json.loads(raw.decode())
            except json.JSONDecodeError:
                continue  # servers sometimes log to stdout; skip the noise
            if message.get("id") == payload["id"]:
                return message
            # Anything else is a notification or a server-initiated request: ignore.

    async def _send_http(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        assert self._http and self.server.url
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **self.server.headers,
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        resp = await self._http.post(self.server.url, json=payload, headers=headers)
        if resp.status_code >= 300:
            raise MCPError(f"{self.server.name} returned {resp.status_code}: "
                           f"{resp.text[:500]}")
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]
        if "id" not in payload:
            return None
        body = resp.text.strip()
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            for line in body.splitlines():
                if line.startswith("data:"):
                    message = json.loads(line[5:].strip())
                    if message.get("id") == payload["id"]:
                        return message
            raise MCPError(f"{self.server.name}: no response in the event stream")
        return json.loads(body) if body else None

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method,
                   "params": params or {}}
        async with self._lock:
            if self.server.transport == "stdio":
                message = await self._send_stdio(payload)
            else:
                message = await self._send_http(payload)
        if message is None:
            raise MCPError(f"{self.server.name}: no reply to {method}")
        if "error" in message:
            err = message["error"]
            raise MCPError(f"{self.server.name} {method}: "
                           f"{err.get('message', err)} ({err.get('code')})")
        return message.get("result", {})

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        async with self._lock:
            if self.server.transport == "stdio":
                await self._send_stdio(payload)
            else:
                await self._send_http(payload)

    # ---- the MCP surface ----------------------------------------------
    async def list_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self.tools_cache and not refresh:
            return self.tools_cache
        result = await self._request("tools/list")
        tools = result.get("tools", [])
        if self.server.allowed_tools is not None:
            allowed = set(self.server.allowed_tools)
            tools = [t for t in tools if t.get("name") in allowed]
        self.tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self._request("tools/call",
                                     {"name": name, "arguments": arguments or {}})
        return self._render(result)

    async def list_resources(self) -> list[dict[str, Any]]:
        return (await self._request("resources/list")).get("resources", [])

    async def read_resource(self, uri: str) -> str:
        result = await self._request("resources/read", {"uri": uri})
        parts = [c.get("text", "") for c in result.get("contents", [])]
        return "\n".join(p for p in parts if p)

    async def list_prompts(self) -> list[dict[str, Any]]:
        return (await self._request("prompts/list")).get("prompts", [])

    async def get_prompt(self, name: str,
                         arguments: dict[str, Any] | None = None) -> str:
        result = await self._request("prompts/get",
                                     {"name": name, "arguments": arguments or {}})
        out: list[str] = []
        for message in result.get("messages", []):
            content = message.get("content")
            if isinstance(content, dict):
                out.append(content.get("text", ""))
            elif isinstance(content, list):
                out.extend(c.get("text", "") for c in content if isinstance(c, dict))
        return "\n".join(p for p in out if p)

    @staticmethod
    def _render(result: dict[str, Any]) -> str:
        """MCP content blocks → the string the model reads."""
        if result.get("isError"):
            prefix = "Error: "
        else:
            prefix = ""
        chunks: list[str] = []
        for block in result.get("content", []) or []:
            kind = block.get("type")
            if kind == "text":
                chunks.append(block.get("text", ""))
            elif kind == "resource":
                resource = block.get("resource") or {}
                chunks.append(resource.get("text") or resource.get("uri", ""))
            elif kind in {"image", "audio"}:
                chunks.append(f"[{kind}: {block.get('mimeType', 'binary')}]")
        if not chunks and result.get("structuredContent"):
            chunks.append(render_result(result["structuredContent"]))
        return prefix + "\n".join(c for c in chunks if c)

    # ---- tools ---------------------------------------------------------
    def as_tools(self) -> list[Tool]:
        """Expose the server's tools as harness Tools. Call after `list_tools()`."""
        built: list[Tool] = []
        for spec in self.tools_cache:
            built.append(self._make_tool(spec))
        return built

    def _make_tool(self, spec: dict[str, Any]) -> Tool:
        client = self
        remote_name = spec.get("name", "tool")
        local_name = (f"{self.server.name}_{remote_name}" if self.server.prefix
                      else remote_name)
        schema = spec.get("inputSchema") or {"type": "object", "properties": {}}

        async def call(**kwargs: Any) -> str:
            return await client.call_tool(remote_name, kwargs)

        call.__name__ = local_name
        return Tool(
            call,
            name=local_name,
            description=spec.get("description") or f"MCP tool {remote_name}",
            parameters=schema,
            tags=["mcp", self.server.name],
        )

    async def close(self) -> None:
        self.connected = False
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), 5)
            except (ProcessLookupError, *_Timeout):
                self._proc.kill()
            finally:
                # Release the transport here rather than leaving it to __del__,
                # which on 3.10 runs after the loop that owns it has closed and
                # raises "Event loop is closed" where nothing can catch it.
                transport = getattr(self._proc, "_transport", None)
                if transport is not None:
                    try:
                        transport.close()
                    except (RuntimeError, AttributeError):
                        pass
            self._proc = None
        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None


class MCPManager:
    """Several servers at once, with one tool list across all of them."""

    def __init__(self, servers: list[MCPServer | dict[str, Any]] | None = None) -> None:
        self.clients: dict[str, MCPClient] = {}
        for entry in servers or []:
            server = entry if isinstance(entry, MCPServer) else MCPServer(**entry)
            self.clients[server.name] = MCPClient(server)

    def add(self, server: MCPServer, **kw: Any) -> MCPClient:
        client = MCPClient(server, **kw)
        self.clients[server.name] = client
        return client

    async def connect(self) -> MCPManager:
        """Connect every server in parallel; a server that fails is reported, not fatal."""
        async def one(client: MCPClient) -> None:
            await client.connect()
            await client.list_tools()

        results = await asyncio.gather(
            *(one(c) for c in self.clients.values()), return_exceptions=True
        )
        self.errors = [
            f"{name}: {res}" for name, res in
            zip(self.clients, results, strict=False) if isinstance(res, BaseException)
        ]
        return self

    def tools(self) -> list[Tool]:
        return [t for client in self.clients.values() if client.connected
                for t in client.as_tools()]

    def get(self, name: str) -> MCPClient:
        if name not in self.clients:
            raise MCPError(f"no MCP server named {name!r}")
        return self.clients[name]

    async def close(self) -> None:
        await asyncio.gather(*(c.close() for c in self.clients.values()),
                             return_exceptions=True)

    async def __aenter__(self) -> MCPManager:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
