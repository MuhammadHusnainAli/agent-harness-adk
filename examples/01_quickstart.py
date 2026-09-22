"""An agent with one tool. The smallest thing worth running."""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, tool, tool_call

ORDERS = {"4182": "shipped 2 days ago, arriving Thursday", "5011": "awaiting payment"}


@tool
def order_status(order_id: str) -> str:
    """Look up the status of a customer order.

    Args:
        order_id: the order number, digits only.
    """
    return ORDERS.get(order_id, "no such order")


async def main() -> None:
    provider, model = pick_provider([
        tool_call("order_status", order_id="4182"),
        "Order 4182 shipped two days ago and should arrive on Thursday.",
    ])

    agent = Agent(
        "support",
        "Answer customer questions about orders. Look the order up before answering.",
        model=model,
        provider=provider,
        tools=[order_status],
    )

    result = await agent.run("Where is order 4182?")
    print(result.output)
    print(f"\n{result.steps} steps · ${result.cost_usd:.4f} · "
          f"tools: {[c.name for c in result.tool_calls]}")


if __name__ == "__main__":
    asyncio.run(main())
