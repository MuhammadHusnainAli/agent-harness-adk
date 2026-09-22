from __future__ import annotations

import json

import httpx
import pytest

from agent_harness import MCPClient, MCPManager, MCPServer
from agent_harness.errors import MCPError

TOOLS = [{
    "name": "read_file",
    "description": "Read a file from the server.",
    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]},
}]


def fake_server(*, fail_call: bool = False):
    """A minimal MCP server over the streamable-HTTP transport."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        method = body.get("method")
        if "id" not in body:
            return httpx.Response(202)
        result: dict = {}
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18",
                      "serverInfo": {"name": "files", "version": "1"},
                      "capabilities": {"tools": {}}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            if fail_call:
                result = {"isError": True,
                          "content": [{"type": "text", "text": "no such file"}]}
            else:
                path = body["params"]["arguments"]["path"]
                result = {"content": [{"type": "text", "text": f"contents of {path}"}]}
        elif method == "resources/list":
            result = {"resources": [{"uri": "file:///a.txt", "name": "a.txt"}]}
        elif method == "resources/read":
            result = {"contents": [{"uri": "file:///a.txt", "text": "hello"}]}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "summarise"}]}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user",
                                    "content": {"type": "text", "text": "Summarise:"}}]}
        else:
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "error": {"code": -32601, "message": "method not found"},
            })
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                         "result": result},
                              headers={"mcp-session-id": "s-1"})

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handler))


def client_for(**kw) -> tuple[list[dict], MCPClient]:
    seen, http = fake_server(**kw)
    server = MCPServer(name="files", transport="http", url="http://mcp.test/rpc")
    return seen, MCPClient(server, http_client=http)


async def test_the_handshake_runs_before_anything_else():
    seen, client = client_for()
    await client.connect()
    assert seen[0]["method"] == "initialize"
    assert seen[1]["method"] == "notifications/initialized"
    assert client.connected


async def test_tools_are_discovered_and_become_harness_tools():
    _, client = client_for()
    await client.connect()
    await client.list_tools()
    tools = client.as_tools()
    assert [t.name for t in tools] == ["files_read_file"]
    assert tools[0].description == "Read a file from the server."
    assert tools[0].parameters["properties"]["path"]["type"] == "string"


async def test_calling_a_remote_tool_passes_the_arguments_through():
    _, client = client_for()
    await client.connect()
    await client.list_tools()
    entry = client.as_tools()[0]
    assert await entry.invoke({"path": "/etc/hosts"}) == "contents of /etc/hosts"


async def test_a_server_side_tool_error_comes_back_as_an_error_string():
    _, client = client_for(fail_call=True)
    await client.connect()
    await client.list_tools()
    entry = client.as_tools()[0]
    assert (await entry.invoke({"path": "/nope"})).startswith("Error: ")


async def test_resources_and_prompts_are_reachable():
    _, client = client_for()
    await client.connect()
    assert (await client.list_resources())[0]["uri"] == "file:///a.txt"
    assert await client.read_resource("file:///a.txt") == "hello"
    assert (await client.list_prompts())[0]["name"] == "summarise"
    assert await client.get_prompt("summarise") == "Summarise:"


async def test_a_json_rpc_error_is_raised_with_context():
    _, client = client_for()
    await client.connect()
    with pytest.raises(MCPError, match="method not found"):
        await client._request("nonsense/method")


async def test_an_allowlist_hides_the_rest_of_the_servers_tools():
    seen, http = fake_server()
    server = MCPServer(name="files", transport="http", url="http://mcp.test/rpc",
                       allowed_tools=["something_else"])
    client = MCPClient(server, http_client=http)
    await client.connect()
    assert await client.list_tools() == []


async def test_tool_names_can_go_unprefixed():
    seen, http = fake_server()
    server = MCPServer(name="files", transport="http", url="http://mcp.test/rpc",
                       prefix=False)
    client = MCPClient(server, http_client=http)
    await client.connect()
    await client.list_tools()
    assert client.as_tools()[0].name == "read_file"


async def test_the_manager_reports_a_server_that_will_not_connect():
    manager = MCPManager([MCPServer(name="broken", transport="stdio", command=None)])
    await manager.connect()
    assert manager.errors and "broken" in manager.errors[0]
    assert manager.tools() == []
    await manager.close()


async def test_a_url_without_a_command_selects_the_http_transport():
    server = MCPServer(name="remote", url="http://mcp.test/rpc")
    assert server.transport == "http"
