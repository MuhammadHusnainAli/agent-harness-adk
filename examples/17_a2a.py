"""A2A: serve an agent to anyone, and call one as if it were yours.

One process here plays both sides. A pricing agent is served over the
agent-to-agent protocol on a real socket; a manager — which could just as well
be a different framework on a different continent — finds it by its card and
uses it as a sub-agent.
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, FakeProvider, Harness, tool, tool_call
from agent_harness.a2a import A2AClient, A2AServer, RemoteAgent

TOKEN = "sk-demo-token"


@tool
def price_list(plan: str) -> str:
    """The monthly price of a plan.

    Args:
        plan: the plan's name.
    """
    return {"gold": "40 EUR per seat", "silver": "15 EUR per seat"}.get(
        plan.lower(), "no such plan")


async def main() -> None:
    provider, model = pick_provider()
    scripted = provider is not None

    def scripts(*turns: object) -> FakeProvider | None:
        return FakeProvider(list(turns), stream_words=True) if scripted else None

    # --- the side that serves ----------------------------------------------
    pricing = Agent(
        "pricing", "You know the price list. Look a price up before you quote it.",
        description="Quotes plan prices from the price list.",
        tools=[price_list], model=model, harness=Harness(), memory=False,
        provider=scripts(
            tool_call("price_list", plan="gold"),
            "The gold plan is 40 EUR per seat, per month.",
            "For 12 seats that is 480 EUR per month.",
            tool_call("price_list", plan="silver"),
            "Silver is 15 EUR per seat, per month."),
    )
    server = A2AServer(pricing, auth={TOKEN: {"user_id": "acme-bot", "tenant_id": "acme"}})
    started: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    serving = asyncio.create_task(
        server.serve("127.0.0.1", 0, ready=started.set_result))
    url = await started
    print(f"pricing is served at {url}\n")

    # --- any A2A client can call it -------------------------------------------
    async with A2AClient(url, token=TOKEN) as client:
        card = await client.card()
        print(f"card       {card['name']} v{card['version']} — {card['description']}")
        print(f"           A2A {card['protocolVersion']}, streaming: "
              f"{card['capabilities']['streaming']}, auth: "
              f"{', '.join(card.get('securitySchemes', {'none': 0}))}")

        task = await client.send("What does the gold plan cost?")
        print(f"\nsend       {task.state}: {task.text}")

        print("stream     ", end="")
        async for event in client.stream("And for 12 seats?",
                                         context_id=task.context_id):
            if event.kind == "artifact-update":
                print(event.text, end="", flush=True)
            elif event.final:
                print(f"   [{event.state}]")

    # --- and your own agents can use it as one of theirs ------------------------
    remote = await RemoteAgent.connect(url, token=TOKEN)
    manager = Agent(
        "manager", "Answer the customer. Ask the pricing agent for any price.",
        subagents=[remote], model=model, harness=Harness(), memory=False,
        provider=scripts(
            tool_call("delegate", agent_name="pricing",
                      task="What does the silver plan cost per seat?"),
            "Silver is 15 EUR per seat each month."),
    )
    result = await manager.run("How much is silver?")
    print(f"\nmanager    {result.output}")
    print(f"           delegated to: {[c.agent for c in result.children]} over A2A")

    health = server.health()
    print(f"\nserver     {health['completed']} tasks completed, "
          f"{health['running']} running, capacity {health['capacity']}")

    await remote.aclose()
    serving.cancel()
    await asyncio.gather(serving, return_exceptions=True)
    await pricing.harness.aclose()
    await manager.harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
