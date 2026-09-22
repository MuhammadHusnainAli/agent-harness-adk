"""Skills loaded from disk, and memory that outlives the session."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _common import pick_provider

from agent_harness import Agent, Harness, tool_call


def write_skills(root: Path) -> Path:
    """Two skills. Only their descriptions reach the prompt until one is loaded."""
    skills = root / "skills"
    (skills / "refunds").mkdir(parents=True)
    (skills / "refunds" / "SKILL.md").write_text(
        "---\nname: refunds\ndescription: How we process a refund, including the "
        "approval thresholds.\n---\n\n"
        "1. Check the order is inside the 30-day window.\n"
        "2. Under 100 EUR: refund immediately.\n"
        "3. Over 100 EUR: a manager approves first.\n"
    )
    (skills / "escalation").mkdir(parents=True)
    (skills / "escalation" / "SKILL.md").write_text(
        "---\nname: escalation\ndescription: When and how to escalate a ticket.\n---\n\n"
        "Escalate anything involving a chargeback within one hour.\n"
    )
    return skills


async def main() -> None:
    provider, model = pick_provider([
        tool_call("load_skill", name="refunds"),
        tool_call("remember", fact="This user's customers pay in EUR"),
        "A 60 EUR refund inside the 30-day window can be issued immediately.",
        "You told me your customers pay in EUR.",
    ])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        harness = Harness.local(root / "state")   # memory and sessions persist here

        agent = Agent(
            "support",
            "Answer support questions. Load the relevant skill before you rely on it.",
            model=model, provider=provider, harness=harness,
            skills=str(write_skills(root)),
        )

        first = await agent.run("Customer wants 60 EUR back on an order from last week.")
        print(first.output)

        # A new run, no transcript carried over — but memory still knows the user.
        second = await agent.run("What currency do my customers use?", messages=[])
        print("\n" + second.output)

        print("\n--- user.md after the session ---")
        print(await agent.close_session())
        await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
