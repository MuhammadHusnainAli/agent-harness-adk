"""Human-in-the-loop approval that survives a restart.

An agent wants to make a refund. The refund needs a person's yes, and nobody is
there — so the run stops at that step and is written to a database. Later, a
different process (here: a second harness and a second agent, built from
nothing but the database) records the approval and carries the run on from
exactly where it stopped: the approved call runs with the approved arguments,
the lookup that had already run is not run again, and the model is never asked
to choose the call a second time.

Runs with no API key, against a scripted provider and a SQLite file.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agent_harness import Agent, ApprovalError, FakeProvider, Harness, tool, tool_call

LEDGER: list[str] = []          # what actually happened to money


@tool
def find_order(order_id: str) -> str:
    """Look an order up.

    Args:
        order_id: the order to find.
    """
    LEDGER.append(f"looked up {order_id}")
    return f"order {order_id}: delivered 28 September, charged twice, 40 EUR each"


@tool(permission="ask")
def refund(order_id: str, amount: float, reason: str) -> str:
    """Refund a customer. Costs real money, so a person approves it first.

    Args:
        order_id: the order to refund.
        amount: how much, in EUR.
        reason: why.
    """
    LEDGER.append(f"refunded {amount:g} EUR on {order_id}")
    return f"refunded {amount:g} EUR on order {order_id}"


def process(database: str, script: list) -> tuple[Harness, Agent]:
    """What each process builds: a harness on the database, and the agent."""
    harness = Harness(sessions=database, approvals=True, provider=FakeProvider(script))
    harness.approvals.notify = lambda a: print(f"notify     → #refunds: {a.describe()}")
    agent = Agent("support", "Help customers with their orders.",
                  tools=[find_order, refund], harness=harness, memory=False)
    return harness, agent


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        database = f"sqlite:///{Path(folder) / 'agents.db'}"

        # --- Monday, 09:00 — the request comes in ------------------------------
        harness, agent = process(database, [
            [tool_call("find_order", order_id="4182"),
             tool_call("refund", order_id="4182", amount=40, reason="charged twice")]])
        result = await agent.run("Order 4182 was charged twice. Please fix it.")
        print(f"run        stopped: {result.stop_reason} at step {result.steps}")
        print(f"approval   {result.approval.id}")
        print(f"ledger     {LEDGER}\n")
        approval_id = result.approval.id
        await harness.aclose()                    # the process ends; nothing is waiting

        # --- Monday, 19:00 — another process, ten hours later ---------------------
        harness, agent = process(database, ["I refunded the duplicate 40 EUR charge "
                                            "on order 4182."])
        for waiting in await harness.approvals.pending():
            print(f"waiting    {waiting.id}: {waiting.describe()}")
        try:
            await agent.resume_approval(approval_id)
        except ApprovalError as exc:
            print(f"too soon   {exc}")

        await harness.approvals.approve(approval_id, by="maria",
                                        note="duplicate charge confirmed")
        result = await agent.resume_approval(approval_id)
        print(f"\nsupport    {result.output}")
        print(f"ledger     {LEDGER}")              # one lookup, one refund
        print(f"run        {result.steps} steps in all, same session: "
              f"{result.session_id}")

        try:
            await agent.resume_approval(approval_id)
        except ApprovalError as exc:
            print(f"once only  {exc}")
        await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
