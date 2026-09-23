"""The assurance rails: stop control, audit trail, health, replay and evaluation."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from _common import pick_provider

from agent_harness import (
    Agent,
    Evaluator,
    Expect,
    GoldenTask,
    Harness,
    RecordingProvider,
    ReplayProvider,
    tool,
    tool_call,
)


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return "shipped Thursday"


async def main() -> None:
    provider, model = pick_provider([
        tool_call("order_status", order_id="4182"),
        "Order 4182 shipped Thursday.",
    ])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        harness = Harness.local(root / "state", trace=False)
        recording = root / "recording.jsonl"

        # Wrap the provider so the run can be reproduced later, exactly. (With a
        # real API key `pick_provider` returns None and the model id resolves the
        # provider, so there is nothing to wrap.)
        recorder = RecordingProvider(provider, recording) if provider else None
        live = recorder or provider

        agent = Agent("support", "Answer order questions.", model=model,
                      provider=live, harness=harness, tools=[order_status])

        result = await agent.run("Where is order 4182?")
        print(f"answer: {result.output}\n")

        # --- audit trail: who did what, and was it allowed -------------------
        print("--- audit trail ---")
        print(harness.audit.render())
        ok, reason = harness.audit.verify()
        print(f"chain: {reason}\n")

        # --- service health --------------------------------------------------
        print("--- health ---")
        snapshot = harness.health.snapshot(harness.scheduler)
        print(f"status: {snapshot['status']}")
        for component in snapshot["components"]:
            print(f"  {component['kind']}:{component['name']} "
                  f"{component['calls']} calls · p95 {component['p95_ms']}ms "
                  f"· {component['status']}")

        # --- time travel: what was it holding at step 1? ----------------------
        print("\n--- timeline ---")
        for row in await harness.replayer.timeline(result.run_id):
            print(f"  step {row['step']}: {row['messages']} messages, "
                  f"tools {row['tools_called'] or '-'}, ${row['cost_usd']:.5f}")

        # --- stop control -----------------------------------------------------
        harness.stop("that is enough for the demo", requested_by="the example")
        print(f"\nstopped: {harness.control.report()}")

        # --- deterministic reproduction, with no network and no spend ---------
        if recording.exists():
            replay = ReplayProvider(recording)
            replay_harness = Harness.testing(replay)
            twin = Agent("support", "Answer order questions.", model=model,
                         provider=replay, harness=replay_harness,
                         tools=[order_status], memory=False)
            again = await twin.run("Where is order 4182?")
            print(f"\nreplayed identically: {again.output == result.output}")

        await harness.aclose()

    # --- quality evaluation: does a change help or hurt? ----------------------
    print("\n--- evaluation ---")
    eval_provider, eval_model = pick_provider([
        tool_call("order_status", order_id="4182"),
        "Order 4182 shipped Thursday.",
    ])
    eval_harness = Harness.testing(eval_provider) if eval_provider else Harness()
    graded = Agent("support", "Answer order questions.", model=eval_model,
                   provider=eval_provider, harness=eval_harness,
                   tools=[order_status], memory=False)

    report = await Evaluator([
        GoldenTask(id="looks-it-up", input="Where is order 4182?",
                   expect=Expect(contains=["Thursday"],
                                 tool_called="order_status",
                                 max_steps=4),
                   tags=["orders"]),
    ], concurrency=1).run(graded, label="current")
    print(report.render())


if __name__ == "__main__":
    asyncio.run(main())
