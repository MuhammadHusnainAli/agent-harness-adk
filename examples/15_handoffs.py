"""Handoffs: another agent takes over the conversation, and keeps it.

A front desk works out what the customer needs and hands the conversation to
the specialist whose job it is. The specialist answers the customer directly,
is still there on the next turn, and hands back when the subject changes.
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, FakeProvider, Handoff, Harness, tool, tool_call


@tool
def find_charges(order: str) -> str:
    """List what an order was charged.

    Args:
        order: the order number.
    """
    return f"order {order}: 40.00 EUR on 2 May, 40.00 EUR on 2 May (duplicate)"


@tool
def issue_refund(order: str, amount: float) -> str:
    """Refund part of an order.

    Args:
        order: the order number.
        amount: how much to give back, in EUR.
    """
    return f"refunded {amount:.2f} EUR on order {order}"


async def main() -> None:
    provider, model = pick_provider()
    scripted = provider is not None
    harness = Harness()

    def scripts(*turns: object) -> FakeProvider | None:
        return FakeProvider(list(turns)) if scripted else None

    billing = Agent(
        "billing",
        "You handle charges, refunds and invoices. Look the charges up before you "
        "refund anything. Anything else is not yours: hand it back to triage.",
        description="Charges, refunds and invoices.",
        tools=[find_charges, issue_refund], model=model, harness=harness,
        provider=scripts(
            tool_call("find_charges", order="4182"),
            tool_call("issue_refund", order="4182", amount=40.0),
            "You were charged twice for order 4182. I have refunded 40.00 EUR.",
            "It reaches your card in three to five working days.",
            tool_call("handoff", agent_name="triage", reason="asks about opening hours"),
        ),
    )
    triage = Agent(
        "triage",
        "You are the front desk. Answer general questions yourself. Hand anything "
        "about money to billing — do not try to answer it.",
        mode="chat", model=model, harness=harness,
        # `history="text"`: billing gets what was said, not the front desk's
        # own tool calls. `"full"` (the default) hands over everything.
        handoffs=[Handoff(billing, history="text")],
        provider=scripts(
            tool_call("handoff", agent_name="billing", reason="charged twice, order 4182"),
            "We are open 9 to 5, Monday to Friday.",
        ),
    )
    billing.add_handoff(triage)          # and back again

    for said in ("I was charged twice for order 4182.",
                 "When will I see the money?",
                 "Thanks. What are your opening hours?"):
        print(f"you      › {said}")
        async for event in triage.stream(said):
            if event.type == "handoff":
                print(f"           ({event.data['from']} → {event.data['to']}: "
                      f"{event.data['reason']})")
            elif event.type == "tool_result" and event.data["tool"] != "handoff":
                print(f"           · {event.agent} {event.data['tool']} → {event.text}")
            elif event.type == "run_end":
                result = event.data["result"]
        print(f"{result.agent:<8} › {result.output}")
        print(f"           [{result.steps} steps · next turn goes to "
              f"{result.active_agent}]\n")

    session = await harness.sessions.load(result.session_id)
    print("the one session holds it all:",
          " · ".join(f"{hop['source']} → {hop['target']}"
                     for hop in session.metadata["handoff"]["trail"]))
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
