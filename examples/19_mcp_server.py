"""Agents served as an MCP server, each behind a key of its own.

Two agents are served over MCP on a real socket. Each has its own API key: a
caller holding the billing key is offered one tool, and the orders agent is —
as far as that caller can tell — not there. Then a manager agent connects, as
any MCP client would, and uses what its key opens.
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, FakeProvider, Harness, MCPError, tool, tool_call
from agent_harness.mcp import MCPAgentServer, MCPClient, MCPServer

BILLING_KEY = "sk-billing-demo-key"
ORDERS_KEY = "sk-orders-demo-key"


@tool
def find_charges(order: str) -> str:
    """List what an order was charged.

    Args:
        order: the order number.
    """
    return f"order {order}: 40.00 EUR on 2 May, 40.00 EUR on 2 May (duplicate)"


async def connect(url: str, key: str | None, name: str = "desk") -> MCPClient:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    client = MCPClient(MCPServer(name=name, url=url, headers=headers))
    await client.connect()
    await client.list_tools()
    return client


async def main() -> None:
    provider, model = pick_provider()
    scripted = provider is not None
    harness = Harness()

    def scripts(*turns: object) -> FakeProvider | None:
        return FakeProvider(list(turns)) if scripted else None

    billing = Agent(
        "billing", "You handle charges and refunds. Look the charges up first.",
        description="Charges, refunds and invoices.", tools=[find_charges],
        model=model, harness=harness, memory=False,
        provider=scripts(tool_call("find_charges", order="4182"),
                         "Order 4182 was charged twice; the duplicate 40 EUR is refunded.",
                         "It reaches the card in three to five working days."))
    orders = Agent(
        "orders", "You track where orders are.", description="Where an order is.",
        model=model, harness=harness, memory=False,
        provider=scripts("Order 4182 shipped on Thursday."))

    # --- serve them, each behind its own key -----------------------------------
    server = MCPAgentServer([billing, orders],
                            api_keys={"billing": BILLING_KEY, "orders": ORDERS_KEY})
    started: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    serving = asyncio.create_task(server.serve("127.0.0.1", 0, ready=started.set_result))
    url = await started
    print(f"serving at {url}/mcp   (and {url}/billing/mcp, {url}/orders/mcp)\n")

    # --- what each key opens ------------------------------------------------------
    for label, key in (("billing key", BILLING_KEY), ("orders key", ORDERS_KEY)):
        client = await connect(f"{url}/mcp", key)
        print(f"{label:<12} sees {[t['name'] for t in client.tools_cache]}")
        await client.close()
    for label, key in (("no key", None), ("wrong key", "sk-not-a-real-key")):
        try:
            await connect(f"{url}/mcp", key)
        except MCPError as refused:
            print(f"{label:<12} {str(refused).split(':')[0]}")

    # --- an agent of yours, using one through MCP ----------------------------------
    client = await connect(f"{url}/mcp", BILLING_KEY)
    manager = Agent(
        "manager", "Help the customer. Ask the billing agent about anything to do "
        "with money, and keep to one conversation with it.",
        tools=client.as_tools(), model=model, harness=Harness(), memory=False,
        provider=scripts(
            tool_call("desk_billing", task="Customer was charged twice for order "
                      "4182. Check and refund the duplicate.", conversation_id="c-1"),
            tool_call("desk_billing", task="When will they see the money?",
                      conversation_id="c-1"),
            "The duplicate 40 EUR is refunded and arrives in three to five working days."))
    result = await manager.run("I was charged twice for order 4182 — when do I get "
                               "my money back?")
    print(f"\nmanager      {result.output}")
    for call in result.tool_calls:
        print(f"             called {call.name}({call.args['task'][:48]}…)")

    health = server.health()
    print(f"\nserver       {health['calls']} calls, {health['refused']} refused, "
          f"{health['keys']} keys")
    await client.close()
    serving.cancel()
    await asyncio.gather(serving, return_exceptions=True)
    await harness.aclose()
    await manager.harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
