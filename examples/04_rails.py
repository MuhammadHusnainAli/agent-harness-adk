"""The rails: permissions, budget, hooks, guardrails, tracing and the journal."""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import (
    Agent,
    Budget,
    Harness,
    HookContext,
    HookEngine,
    PolicyGate,
    console_exporter,
    tool,
    tool_call,
)


@tool(permission="ask")
def issue_refund(order_id: str, amount: float) -> str:
    """Refund a customer. Costs real money.

    Args:
        order_id: the order to refund.
        amount: how much, in EUR.
    """
    return f"refunded {amount} EUR on {order_id}"


async def main() -> None:
    provider, model = pick_provider([
        tool_call("issue_refund", order_id="4182", amount=60.0),
        "I have issued the 60 EUR refund on order 4182.",
    ])

    hooks = HookEngine()

    @hooks.on("pre_tool")
    def cap_refunds(ctx: HookContext) -> None:
        if ctx.data["tool"] == "issue_refund" and ctx.data["args"]["amount"] > 100:
            ctx.block("refunds over 100 EUR need a manager")

    async def approve(tool_name: str, args: dict, reason: str) -> bool:
        print(f"  [approval] {tool_name}({args}) — {reason} → allowed")
        return True

    harness = Harness()
    harness.policy = PolicyGate("allow", ask=["issue_refund"], deny=["shell"],
                                approver=approve)
    harness.tracer.add_exporter(console_exporter())
    harness.reset_budget(Budget(max_usd=0.50, max_steps=8))

    agent = Agent("refunds", "Handle refund requests.", model=model, provider=provider,
                  harness=harness, hooks=hooks, tools=[issue_refund])

    result = await agent.run("Refund 60 EUR on order 4182.")
    print("\n" + result.output)

    print("\n--- journal ---")
    print(harness.journal.render())
    print("\n--- run report ---")
    for key, value in harness.report().items():
        print(f"  {key}: {value}")
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
