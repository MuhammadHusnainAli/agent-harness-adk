"""A manager that delegates to sub-agents — some from the bench, some built for the job."""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Agent, Budget, Harness, SubAgentSpec, tool, tool_call

CORPUS = {
    "q3-revenue": "Q3 revenue was 4.2M, up 12% on Q2.",
    "q3-costs": "Q3 costs were 3.1M, of which 1.4M was payroll.",
}


@tool
def lookup(topic: str) -> str:
    """Look a topic up in the company corpus.

    Args:
        topic: the topic key, e.g. "q3-revenue".
    """
    return CORPUS.get(topic, "nothing on file")


async def main() -> None:
    provider, model = pick_provider([
        tool_call("delegate", agent_name="revenue_reader", task="find Q3 revenue"),
        tool_call("lookup", topic="q3-revenue"),
        "Q3 revenue was 4.2M, up 12%.",
        tool_call("delegate", agent_name="cost_reader", task="find Q3 costs"),
        tool_call("lookup", topic="q3-costs"),
        "Q3 costs were 3.1M.",
        "Q3: revenue 4.2M (up 12%), costs 3.1M — margin 1.1M.",
    ])

    harness = Harness()
    harness.reset_budget(Budget(max_usd=1.00, max_subagents=6))

    manager = Agent(
        "manager",
        "Answer the question by delegating the lookups, then consolidate what comes "
        "back into one answer.",
        model=model,
        provider=provider,
        harness=harness,
        tools=[lookup],
        subagents=[
            SubAgentSpec(name="revenue_reader", description="Finds revenue figures.",
                         instructions="Look up the figure and report it with its source.",
                         tools=["lookup"], tier="fast"),
            SubAgentSpec(name="cost_reader", description="Finds cost figures.",
                         instructions="Look up the figure and report it with its source.",
                         tools=["lookup"], tier="fast"),
        ],
    )

    result = await manager.run("How did Q3 go?")
    print(result.output)
    print("\n--- what each sub-agent cost ---")
    for child in result.children:
        print(f"  {child.agent:<16} {child.steps} steps  ${child.cost_usd:.4f}")
    print(f"  {'TOTAL':<16} {result.steps} steps  ${result.cost_usd:.4f}")
    await harness.aclose()




# ---------------------------------------------------------------------------
# The same job, but the manager writes its own specialists at run time instead
# of being handed a roster. Run this file with RUNTIME=1 to see it.
# ---------------------------------------------------------------------------

async def runtime_agents_demo() -> None:
    import json

    provider, model = pick_provider([
        tool_call("spawn_agent", task="find the Q3 revenue figure"),
        json.dumps({"name": "revenue_finder", "description": "Finds revenue.",
                    "instructions": "Look it up and cite the source.",
                    "tools": ["lookup"], "tier": "fast", "max_steps": 3,
                    "workspace": "none"}),
        tool_call("lookup", topic="q3-revenue"),
        "Q3 revenue was 4.2M, up 12%.",
        tool_call("spawn_agent", task="find the Q3 cost figure"),
        json.dumps({"name": "cost_finder", "description": "Finds costs.",
                    "instructions": "Look it up and cite the source.",
                    "tools": ["lookup"], "tier": "fast", "max_steps": 3,
                    "workspace": "none"}),
        tool_call("lookup", topic="q3-costs"),
        "Q3 costs were 3.1M.",
        "Q3: revenue 4.2M (up 12%), costs 3.1M — margin 1.1M.",
    ])

    harness = Harness()
    harness.reset_budget(Budget(max_usd=1.00, max_subagents=10))

    manager = Agent(
        "core",
        "Break the question into parts and give each part its own specialist.",
        model=model, provider=provider, harness=harness, tools=[lookup],
        runtime_agents="enable",   # it writes its own workers
        max_runtime_agents=5,      # up to five of them, this run
    )

    result = await manager.run("How did Q3 go?")
    print(result.output)
    print(f"\nspun up {manager.total_spawned} specialists, "
          f"{manager.runtime_agents_remaining} of the budget left")
    for child in result.children:
        print(f"  {child.agent:<16} {child.steps} steps  ${child.cost_usd:.4f}")
    await harness.aclose()


if __name__ == "__main__":
    import os

    asyncio.run(runtime_agents_demo() if os.environ.get("RUNTIME") else main())
