"""Agents served as an MCP server, each behind a key of its own."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from agent_harness import (
    Agent,
    Blueprint,
    ConfigurationError,
    FakeProvider,
    Harness,
    MCPError,
    PolicyGate,
    tool,
    tool_call,
)
from agent_harness.mcp import MCPAgentServer, MCPClient, MCPServer

BASE = "http://mcp.test"
BILLING_KEY, ORDERS_KEY, MASTER_KEY = "sk-billing-0001", "sk-orders-0001", "sk-master-0001"
GATE: dict[str, asyncio.Event] = {}


@tool
def find_charges(order: str) -> str:
    """List what an order was charged.

    Args:
        order: the order number.
    """
    return f"order {order}: 40 EUR, 40 EUR"


@tool
async def slow() -> str:
    """Wait until the test lets it go."""
    GATE.setdefault("started", asyncio.Event()).set()
    await GATE.setdefault("go", asyncio.Event()).wait()
    return "waited"


@pytest.fixture(autouse=True)
def _gates():
    GATE.clear()
    yield
    GATE.clear()


def agents(billing: list | None = None, orders: list | None = None,
           harness: Harness | None = None, **kw: Any) -> tuple[Agent, Agent]:
    harness = harness or Harness.testing()
    return (
        Agent("billing", "Handle refunds.", description="Refunds and invoices.",
              provider=FakeProvider(billing or ["Refunded 40 EUR."]), harness=harness,
              memory=False, tools=[find_charges, slow], **kw),
        Agent("orders", "Track orders.", description="Where an order is.",
              provider=FakeProvider(orders or ["It shipped on Thursday."]),
              harness=harness, memory=False),
    )


def keyed(**kw: Any) -> tuple[MCPAgentServer, Agent, Agent]:
    billing, orders = agents(**{k: kw.pop(k) for k in ("billing", "orders", "harness")
                                if k in kw})
    kw.setdefault("api_keys", {"billing": BILLING_KEY, "orders": ORDERS_KEY})
    kw.setdefault("api_key", MASTER_KEY)
    return MCPAgentServer([billing, orders], **kw), billing, orders


def http(server: MCPAgentServer, key: str | None = None, **headers: str) -> httpx.AsyncClient:
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url=BASE,
                             headers=headers)


def rpc(method: str, params: dict[str, Any] | None = None, id: Any = 1) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}


async def connect(server: MCPAgentServer, key: str | None = None, path: str = "/mcp",
                  **options: Any) -> MCPClient:
    """The harness's own MCP client, as any other MCP client would connect."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = MCPClient(
        MCPServer(name="remote", url=f"{BASE}{path}", headers=headers, **options),
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=server)))
    await client.connect()
    await client.list_tools()
    return client


def names(client: MCPClient) -> list[str]:
    return [t["name"] for t in client.tools_cache]


# ----------------------------------------------------------------------
# an agent is a tool
# ----------------------------------------------------------------------
async def test_an_agent_is_served_as_a_tool_any_mcp_client_can_call():
    billing, _ = agents()
    server = MCPAgentServer(billing, name="acme-agents", version="2.0.0",
                            instructions="Ask billing about money.")

    hello = await server.handle(rpc("initialize", {"protocolVersion": "2025-03-26"}))
    assert hello["result"] == {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "acme-agents", "version": "2.0.0"},
        "instructions": "Ask billing about money."}
    assert (await server.handle(rpc("initialize", {"protocolVersion": "1999-01-01"}))
            )["result"]["protocolVersion"] == "2025-06-18"
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) \
        is None

    client = await connect(server)
    assert names(client) == ["billing"]
    spec = client.tools_cache[0]
    assert spec["description"].startswith(
        "Ask the billing agent. Refunds and invoices. It runs the task")
    assert spec["inputSchema"]["required"] == ["task"]
    assert set(spec["inputSchema"]["properties"]) == {"task", "conversation_id"}

    assert await client.call_tool("billing", {"task": "Refund order 4182."}) == (
        "Refunded 40 EUR.")
    assert billing.provider.requests[0].messages[0].text == "Refund order 4182."
    # Its own instructions and tools: the caller got an agent, not a model.
    assert "Handle refunds." in billing.provider.requests[0].system
    await client.close()


async def test_one_of_your_agents_uses_another_over_mcp():
    billing, _ = agents(billing=[tool_call("find_charges", order="4182"),
                                 "Order 4182 was charged twice; refunded 40 EUR."])
    server = MCPAgentServer(billing, api_key=MASTER_KEY)
    client = await connect(server, MASTER_KEY)

    manager = Agent("manager", tools=client.as_tools(), harness=Harness.testing(),
                    memory=False, provider=FakeProvider([
                        tool_call("remote_billing", task="Check and refund order 4182."),
                        "Done: 40 EUR is on its way back."]))
    result = await manager.run("I was charged twice for 4182.")

    assert result.output == "Done: 40 EUR is on its way back."
    handed = result.messages[2].content[0].content
    assert handed == "Order 4182 was charged twice; refunded 40 EUR."
    await client.close()


# ----------------------------------------------------------------------
# every agent has its own key
# ----------------------------------------------------------------------
async def test_a_key_opens_only_the_agents_it_was_issued_for():
    server, billing, orders = keyed()

    with_billing = await connect(server, BILLING_KEY)
    assert names(with_billing) == ["billing"]
    assert await with_billing.call_tool("billing", {"task": "Refund."}) == "Refunded 40 EUR."
    # The orders agent is not refused to this caller: it is not there.
    with pytest.raises(MCPError, match="Unknown tool: 'orders'"):
        await with_billing.call_tool("orders", {"task": "Where?"})
    assert not orders.provider.requests

    with_orders = await connect(server, ORDERS_KEY)
    assert names(with_orders) == ["orders"]
    assert await with_orders.call_tool("orders", {"task": "Where?"}) == (
        "It shipped on Thursday.")

    with_master = await connect(server, MASTER_KEY)
    assert names(with_master) == ["billing", "orders"]
    for client in (with_billing, with_orders, with_master):
        await client.close()


async def test_no_key_and_a_wrong_key_are_refused():
    server, billing, _ = keyed()
    async with http(server) as nobody:
        refused = await nobody.post("/mcp", json=rpc("tools/list"))
        assert refused.status_code == 401
        assert refused.headers["www-authenticate"] == 'Bearer realm="mcp"'
        assert refused.json()["error"]["message"] == "an API key is required"
    async with http(server, "sk-not-a-real-key") as wrong:
        refused = await wrong.post("/mcp", json=rpc("tools/call", {
            "name": "billing", "arguments": {"task": "Refund."}}))
        assert refused.status_code == 401
        assert refused.json()["error"]["message"] == "that API key is not valid"
    assert not billing.provider.requests and server.health()["refused"] == 2

    # The other header people send a key in.
    async with http(server, **{"X-API-Key": BILLING_KEY}) as other:
        listed = (await other.post("/mcp", json=rpc("tools/list"))).json()
        assert [t["name"] for t in listed["result"]["tools"]] == ["billing"]


async def test_an_agent_is_served_alone_at_its_own_path():
    server, _, _ = keyed()
    alone = await connect(server, MASTER_KEY, "/billing/mcp")
    assert names(alone) == ["billing"]              # even the master key: one agent here
    await alone.close()

    async with http(server, ORDERS_KEY) as client:
        wrong = await client.post("/billing/mcp", json=rpc("tools/list"))
        assert wrong.status_code == 403
        assert wrong.json()["error"]["message"] == "this key does not open billing"
        assert (await client.post("/ghost/mcp", json=rpc("tools/list"))).status_code == 404
        assert (await client.post("/orders/mcp", json=rpc("tools/list"))).status_code == 200


async def test_keys_can_be_several_shared_public_issued_and_revoked():
    billing, orders = agents()
    server = MCPAgentServer(
        [billing, orders],
        api_keys={"billing": [BILLING_KEY, "sk-shared-0001"],
                  "orders": ["sk-shared-0001", {"key": ORDERS_KEY, "user_id": "ada"}]})
    assert names(await connect(server, "sk-shared-0001")) == ["billing", "orders"]
    assert names(await connect(server, BILLING_KEY)) == ["billing"]

    issued = server.add_key("sk-new-partner-1", agent="orders")
    assert issued.startswith("key_") and "partner" not in issued
    assert names(await connect(server, "sk-new-partner-1")) == ["orders"]
    assert server.revoke_key("sk-new-partner-1") and not server.revoke_key("sk-gone-00001")
    with pytest.raises(MCPError, match="returned 401"):
        await connect(server, "sk-new-partner-1")

    # An agent listed with None is open to a caller with no key at all.
    billing, orders = agents()
    mixed = MCPAgentServer([billing, orders],
                           api_keys={"billing": BILLING_KEY, "orders": None})
    assert names(await connect(mixed)) == ["orders"]
    assert names(await connect(mixed, BILLING_KEY)) == ["billing", "orders"]
    with pytest.raises(MCPError, match="returned 401"):
        await connect(mixed, "sk-not-a-real-key")


def test_a_server_that_would_leave_an_agent_unreachable_is_not_built():
    billing, orders = agents()
    with pytest.raises(ConfigurationError, match="orders would be served with no key"):
        MCPAgentServer([billing, orders], api_keys={"billing": BILLING_KEY})
    with pytest.raises(ConfigurationError, match="names 'ghost', which is not served"):
        MCPAgentServer([billing], api_keys={"ghost": BILLING_KEY})
    with pytest.raises(ConfigurationError, match="at least 8 characters"):
        MCPAgentServer([billing], api_key="short")
    with pytest.raises(ConfigurationError, match="serves Agents"):
        MCPAgentServer({"x": object()})
    with pytest.raises(ConfigurationError, match="at least one agent"):
        MCPAgentServer([])
    # No keys at all is an open server — for development, and it says so.
    assert MCPAgentServer([billing, orders]).open


async def test_a_function_can_decide_who_is_calling():
    def gate(headers: dict[str, str]) -> Any:
        tenant = headers.get("x-tenant")
        return {"user_id": "svc", "tenant_id": tenant, "agents": ["orders"]} \
            if tenant == "acme" else None

    billing, orders = agents()
    server = MCPAgentServer([billing, orders], auth=gate)
    async with http(server, **{"X-Tenant": "acme"}) as client:
        listed = (await client.post("/mcp", json=rpc("tools/list"))).json()
        assert [t["name"] for t in listed["result"]["tools"]] == ["orders"]
    async with http(server) as client:
        assert (await client.post("/mcp", json=rpc("tools/list"))).status_code == 401


# ----------------------------------------------------------------------
# conversations
# ----------------------------------------------------------------------
async def test_a_conversation_id_carries_a_conversation_on_and_is_the_callers_own():
    harness = Harness.testing()
    server, billing, _ = keyed(
        harness=harness,
        billing=["Noted: order 4182.", "It was 4182.", "Which order?", "No idea."])
    ada = await connect(server, BILLING_KEY)
    bob = await connect(server, MASTER_KEY)

    await ada.call_tool("billing", {"task": "My order is 4182.", "conversation_id": "c1"})
    assert await ada.call_tool("billing", {"task": "Which order?",
                                           "conversation_id": "c1"}) == "It was 4182."
    seen = [[m.text for m in r.messages] for r in billing.provider.requests]
    assert seen[1] == ["My order is 4182.", "Noted: order 4182.", "Which order?"]

    # Without an id a call is one task, with nothing carried over.
    await ada.call_tool("billing", {"task": "Which order?"})
    # The same id under another key is another conversation.
    await bob.call_tool("billing", {"task": "Which order?", "conversation_id": "c1"})
    seen = [[m.text for m in r.messages] for r in billing.provider.requests]
    assert seen[2] == ["Which order?"] and seen[3] == ["Which order?"]

    chats = [s for s in await harness.sessions.list() if s.id.startswith("mcp_billing_")]
    assert len(chats) == 2 and len({s.user_id for s in chats}) == 2

    off, _, _ = keyed(conversations=False)
    assert "conversation_id" not in off.tools()[0]["inputSchema"]["properties"]


# ----------------------------------------------------------------------
# what comes back
# ----------------------------------------------------------------------
async def test_failures_data_and_files_come_back_as_mcp_says():
    class Refund(BaseModel):
        amount: int

    harness = Harness.testing()
    broken = Agent("broken", provider=FakeProvider([tool_call("nope"), tool_call("nope")]),
                   harness=harness, memory=False, max_steps=2)
    typed = Agent("typed", provider=FakeProvider(['{"amount": 40}']), harness=harness,
                  memory=False, output_type=Refund)
    writer = Agent("writer", provider=FakeProvider(["Report written."]), harness=harness)
    writer.produce("report.md", "# Q3\nRevenue rose 12%.", media_type="text/markdown")
    server = MCPAgentServer([broken, typed, writer], isolate=False)

    async def call(name: str, **arguments: Any) -> dict[str, Any]:
        return (await server.handle(rpc("tools/call", {
            "name": name, "arguments": arguments})))["result"]

    failed = await call("broken", task="Go.")
    assert failed["isError"] and "MaxStepsExceeded" in failed["content"][0]["text"]

    structured = await call("typed", task="Refund.")
    assert structured["structuredContent"] == {"amount": 40}
    assert structured["_meta"] == {"agent": "typed", "steps": 1}

    written = await call("writer", task="Write it.")
    assert written["content"][1] == {"type": "resource", "resource": {
        "uri": "artifact:///report.md", "mimeType": "text/markdown",
        "text": "# Q3\nRevenue rose 12%."}}

    assert (await server.handle(rpc("tools/call", {"name": "typed", "arguments": {}})))[
        "error"]["code"] == -32602
    assert (await server.handle(rpc("tools/call", {"name": "typed", "arguments": {
        "task": "x", "conversation_id": 7}})))["error"]["code"] == -32602
    assert (await server.handle(rpc("sampling/createMessage")))["error"]["code"] == -32601
    assert (await server.handle(rpc("resources/list")))["result"] == {"resources": []}
    assert (await server.handle(rpc("ping")))["result"] == {}
    assert (await server.handle("nonsense"))["error"]["code"] == -32600
    assert server.health()["errors"] == 1


async def test_a_call_that_runs_too_long_is_stopped_and_one_can_be_cancelled():
    billing, _ = agents(billing=[tool_call("slow"), "never"])
    server = MCPAgentServer(billing, timeout=0.05, isolate=False)
    late = (await server.handle(rpc("tools/call", {
        "name": "billing", "arguments": {"task": "Wait."}})))["result"]
    assert late["isError"] and "ran longer than 0.05s" in late["content"][0]["text"]

    GATE.clear()
    billing, _ = agents(billing=[tool_call("slow"), "never"])
    server = MCPAgentServer(billing, isolate=False)
    running = asyncio.create_task(server.handle(rpc("tools/call", {
        "name": "billing", "arguments": {"task": "Wait."}}, id="req-7")))
    await GATE.setdefault("started", asyncio.Event()).wait()
    await server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled",
                         "params": {"requestId": "req-7"}})
    cancelled = (await running)["result"]
    assert cancelled["isError"] and cancelled["content"][0]["text"] == "Cancelled."
    assert len(billing.provider.requests) == 1 and server.health()["running"] == 0


async def test_a_full_server_says_so():
    billing, _ = agents(billing=[tool_call("slow"), "One.", "Two."])
    server = MCPAgentServer(billing, max_concurrency=1, max_queue=0, isolate=False)
    call = rpc("tools/call", {"name": "billing", "arguments": {"task": "Go."}})
    first = asyncio.create_task(server.handle(call))
    await GATE.setdefault("started", asyncio.Event()).wait()
    busy = await server.handle(call)
    assert busy["error"]["code"] == -32000 and "at capacity" in busy["error"]["message"]
    GATE.setdefault("go", asyncio.Event()).set()
    assert (await first)["result"]["content"][0]["text"] == "One."
    assert (await server.handle(call))["result"]["content"][0]["text"] == "Two."
    assert server.health()["busy"] == 1


# ----------------------------------------------------------------------
# the agents' own tools
# ----------------------------------------------------------------------
async def test_an_agents_tools_can_be_served_too_and_keep_their_rails():
    harness = Harness.testing()
    server, billing, _ = keyed(harness=harness, expose_tools=["find_*"])
    client = await connect(server, BILLING_KEY)
    assert names(client) == ["billing", "billing_find_charges"]
    assert client.tools_cache[1]["inputSchema"]["required"] == ["order"]
    assert await client.call_tool("billing_find_charges", {"order": "4182"}) == (
        "order 4182: 40 EUR, 40 EUR")
    assert not billing.provider.requests                 # no model was asked

    everything, _, _ = keyed(expose_tools=True)
    assert [t["name"] for t in everything.tools()] == [
        "billing", "billing_find_charges", "billing_slow", "orders"]

    billing, _ = agents(policy=PolicyGate("allow", deny=["find_charges"]))
    locked = MCPAgentServer(billing, expose_tools=True)
    refused = (await locked.handle(rpc("tools/call", {
        "name": "billing_find_charges", "arguments": {"order": "1"}})))["result"]
    assert refused["isError"] and "Not permitted" in refused["content"][0]["text"]
    await client.close()


async def test_every_call_is_on_the_audit_trail_without_the_key():
    harness = Harness.testing()
    server, _, _ = keyed(harness=harness)
    client = await connect(server, BILLING_KEY)
    await client.call_tool("billing", {"task": "Refund."})
    entry = next(e for e in harness.audit.entries if e.action == "mcp_call")
    assert (entry.actor, entry.target, entry.decision) == ("billing", "billing", "ok")
    assert BILLING_KEY not in json.dumps(entry.model_dump(mode="json"), default=str)
    assert "key_" in json.dumps(entry.model_dump(mode="json"), default=str)
    await client.close()


# ----------------------------------------------------------------------
# the transport
# ----------------------------------------------------------------------
async def test_http_is_answered_as_streamable_http_says():
    server, _, _ = keyed(allowed_origins=["https://app.example"], max_body_bytes=2000)
    async with http(server, MASTER_KEY) as client:
        assert (await client.get("/mcp")).status_code == 405
        assert (await client.delete("/mcp")).status_code == 405
        noted = await client.post("/mcp", json={
            "jsonrpc": "2.0", "method": "notifications/initialized"})
        assert noted.status_code == 202 and noted.content == b""
        bad = await client.post("/mcp", content=b"{not json")
        assert bad.status_code == 400 and bad.json()["error"]["code"] == -32700
        big = await client.post("/mcp", json=rpc("tools/call", {
            "name": "billing", "arguments": {"task": "x" * 3000}}))
        assert big.status_code == 413
        several = (await client.post("/mcp", json=[
            rpc("ping", id=1), {"jsonrpc": "2.0", "method": "notifications/initialized"},
            rpc("tools/list", id=2)])).json()
        assert [a["id"] for a in several] == [1, 2]
        assert (await client.post("/", json=rpc("ping"))).status_code == 200
        assert (await client.post("/mcp", json=rpc("ping"), headers={
            "Origin": "https://evil.example"})).status_code == 403
        assert (await client.post("/mcp", json=rpc("ping"), headers={
            "Origin": "https://app.example"})).status_code == 200
        health = (await client.get("/healthz")).json()
        assert health["status"] == "ok" and health["agents"] == ["billing", "orders"]
        assert health["keys"] == 3 and MASTER_KEY not in json.dumps(health)


async def test_progress_is_streamed_to_a_caller_that_asks_for_it():
    billing, _ = agents(billing=[tool_call("find_charges", order="1"), "Refunded."])
    server = MCPAgentServer(billing, api_key=MASTER_KEY)
    async with http(server, MASTER_KEY) as client:
        response = await client.post(
            "/mcp", headers={"Accept": "application/json, text/event-stream"},
            json=rpc("tools/call", {"name": "billing", "arguments": {"task": "Refund."},
                                    "_meta": {"progressToken": "p-1"}}))
    assert response.headers["content-type"] == "text/event-stream"
    events = [json.loads(line[5:]) for line in response.text.splitlines()
              if line.startswith("data:")]
    progress = [e["params"] for e in events if e.get("method") == "notifications/progress"]
    assert [p["message"] for p in progress] == ["step 1", "using find_charges", "step 2"]
    assert [p["progress"] for p in progress] == [1, 2, 3]
    assert all(p["progressToken"] == "p-1" for p in progress)
    assert events[-1]["id"] == 1
    assert events[-1]["result"]["content"][0]["text"] == "Refunded."


async def test_the_built_in_server_serves_real_http():
    server, _, _ = keyed()
    started: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    serving = asyncio.create_task(server.serve("127.0.0.1", 0, ready=started.set_result))
    url = await started
    try:
        client = MCPClient(MCPServer(name="live", url=f"{url}/orders/mcp", prefix=False,
                                     headers={"X-API-Key": ORDERS_KEY}))
        await client.connect()
        await client.list_tools()
        assert names(client) == ["orders"]
        assert await client.call_tool("orders", {"task": "Where is it?"}) == (
            "It shipped on Thursday.")
        await client.close()
        async with httpx.AsyncClient() as plain:
            assert (await plain.post(f"{url}/mcp", json=rpc("ping"))).status_code == 401
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


STDIO = '''
import asyncio
from agent_harness import Agent, FakeProvider, Harness
from agent_harness.mcp import MCPAgentServer

agent = Agent("helper", description="Answers questions.", memory=False,
              provider=FakeProvider(["Noted.", "You said 42."]), harness=Harness.testing())
asyncio.run(MCPAgentServer(agent).serve_stdio())
'''


async def test_a_client_can_start_the_server_itself_over_stdio():
    client = MCPClient(MCPServer(name="local", command=sys.executable,
                                 args=["-c", STDIO], prefix=False, timeout=30))
    await client.connect()
    await client.list_tools()
    assert names(client) == ["helper"]
    await client.call_tool("helper", {"task": "The number is 42.", "conversation_id": "c"})
    assert await client.call_tool("helper", {"task": "Which number?",
                                             "conversation_id": "c"}) == "You said 42."
    await client.close()


# ----------------------------------------------------------------------
# declared in a file
# ----------------------------------------------------------------------
async def test_a_blueprint_serves_its_agents_each_behind_the_key_it_names(monkeypatch):
    monkeypatch.setenv("BILLING_MCP_KEY", BILLING_KEY)
    monkeypatch.setenv("ORDERS_MCP_KEY", ORDERS_KEY)
    blueprint = Blueprint.from_text("""
agents:
  billing: {description: Refunds., memory: false, mcp_key_env: BILLING_MCP_KEY}
  orders:  {description: Tracking., memory: false, mcp_key_env: ORDERS_MCP_KEY}
  internal: {description: Not for callers., memory: false}
""")
    harness = Harness.testing(FakeProvider(["Refunded."]))
    server = blueprint.mcp_server(harness=harness)
    assert sorted(server.agents) == ["billing", "orders"]       # only those with a key
    client = await connect(server, BILLING_KEY)
    assert names(client) == ["billing"]
    assert await client.call_tool("billing", {"task": "Refund."}) == "Refunded."
    await client.close()

    everything = blueprint.mcp_server(harness=harness, api_key=MASTER_KEY)
    assert sorted(everything.agents) == ["billing", "internal", "orders"]
    assert names(await connect(everything, ORDERS_KEY)) == ["orders"]

    monkeypatch.delenv("ORDERS_MCP_KEY")
    with pytest.raises(ConfigurationError, match="ORDERS_MCP_KEY is not set"):
        blueprint.mcp_server(harness=harness)
    with pytest.raises(ConfigurationError, match="no agent 'ghost'"):
        blueprint.mcp_server(["ghost"], harness=harness)
