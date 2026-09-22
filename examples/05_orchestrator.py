"""The whole pipeline: plan → staff → run in parallel → consolidate → review."""

from __future__ import annotations

import asyncio
import json

from _common import pick_provider

from agent_harness import Budget, Harness, Orchestrator
from agent_harness.orchestrator import Plan

PLAN = {
    "goal": "Summarise how Q3 went",
    "definition_of_done": ["Revenue and costs are both cited", "It fits in a paragraph"],
    "tasks": [
        {"id": "t1", "statement": "Research the Q3 revenue figures", "depends_on": []},
        {"id": "t2", "statement": "Research the Q3 cost figures", "depends_on": []},
        {"id": "t3", "statement": "Write the summary", "depends_on": ["t1", "t2"]},
    ],
}


def script(request):
    """Only used when there is no API key — routes on which prompt arrived."""
    text = request.messages[-1].text
    if "costed plan" in text:
        return json.dumps(PLAN)
    if "Review this deliverable" in text:
        return json.dumps({"accepted": True, "score": 0.9, "summary": "meets the bar"})
    if "Consolidate these sub-agent results" in text:
        return "Q3: revenue 4.2M (up 12%), costs 3.1M. Margin 1.1M."
    return "figure found and cited"


async def main() -> None:
    provider, model = pick_provider([script])

    harness = Harness()
    if provider is not None:
        harness.provider = provider

    boss = Orchestrator(
        "boss", harness=harness, model=model, max_concurrency=4, review=True,
        max_rework=1, budget=Budget(max_usd=2.00),
    )

    result = await boss.run("Summarise how Q3 went, with the numbers cited.")

    plan = Plan(**result.data["plan"])
    print(plan.render())
    print("\n--- staffing ---")
    for task in plan.tasks:
        how = "reused from the bench" if task.reused else "purpose-built"
        print(f"  {task.id}: {task.agent:<20} {how:<22} {task.status}  "
              f"${task.cost_usd:.4f}")
    print(f"\n--- deliverable ---\n{result.output}")
    print(f"\n--- review ---\n{result.data['review']}")
    print(f"\ntotal ${result.cost_usd:.4f} against an estimate of "
          f"${plan.estimate_usd:.4f}")
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
