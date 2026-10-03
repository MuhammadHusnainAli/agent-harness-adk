"""A declared workflow: sequence, parallel, a branch, a foreach and a loop.

The whole thing is the YAML below — the agents, the steps, and how data moves
between them. Nothing decides the order at run time: it runs as written, and
the agents are in the steps where judgement is needed.
"""

from __future__ import annotations

import asyncio

from _common import pick_provider

from agent_harness import Harness, Workflow, tool

WORKFLOW = """
name: refund_desk
description: Check a customer's orders, refund the duplicates, and write to them.

inputs:
  customer: {required: true, type: string}
  orders: {required: true, type: array}

state: {refunded: 0}

agents:
  classifier:
    instructions: >
      You are given the charges on some orders. Reply with only a JSON object:
      {"duplicates": [the order numbers charged more than once]}.
    memory: false
  writer:
    instructions: Write a short, plain email to the customer. No placeholders.
    memory: false
  reviewer:
    instructions: >
      Review the email. Reply with only a JSON object:
      {"approved": true or false, "fix": "what to change"}.
    memory: false

steps:
  # --- two lookups at once -------------------------------------------------
  - id: gather
    parallel:
      - id: charges
        foreach: "{{ inputs.orders }}"
        as: order
        concurrency: 4
        steps:
          - tool: find_charges
            args: {order: "{{ order }}"}
      - id: profile
        tool: customer_profile
        args: {customer: "{{ inputs.customer }}"}

  # --- an agent reads what came back ---------------------------------------
  - id: classify
    agent: classifier
    input: "Charges: {{ steps.charges.output }}"
    parse: json
    save: verdict

  # --- a branch, and a refund for each duplicate ----------------------------
  - id: refunds
    if: len(state.verdict.duplicates) > 0
    then:
      - foreach: "{{ state.verdict.duplicates }}"
        as: order
        steps:
          - tool: issue_refund
            args: {order: "{{ order }}", amount: 40}
            retry: 2
          - set: {refunded: "{{ state.refunded + 40 }}"}
    else:
      - return: "Nothing to refund for {{ inputs.customer }}."

  # --- write, review, and go round until it is approved ---------------------
  - id: email
    loop: {max: 3, until: state.review.approved}
    steps:
      - id: draft
        agent: writer
        input: |
          Customer: {{ steps.profile.output.name }} ({{ steps.profile.output.tier }})
          Refunded: {{ state.refunded }} EUR on orders {{ join(state.verdict.duplicates) }}
          Reviewer's note on the last draft: {{ default(state.review.fix, "none yet") }}
      - id: review
        agent: reviewer
        parse: json
        save: review

output: "{{ steps.draft.output }}"
"""


@tool
def find_charges(order: str) -> dict:
    """List what an order was charged.

    Args:
        order: the order number.
    """
    charges = {"4182": [40, 40], "4190": [15], "4201": [40, 40]}
    return {"order": order, "charges": charges.get(order, [])}


@tool
def customer_profile(customer: str) -> dict:
    """Who a customer is.

    Args:
        customer: the customer id.
    """
    return {"id": customer, "name": "Ada Lovelace", "tier": "gold"}


@tool
def issue_refund(order: str, amount: float) -> str:
    """Refund part of an order.

    Args:
        order: the order number.
        amount: how much to give back, in EUR.
    """
    return f"refunded {amount:.2f} EUR on order {order}"


async def main() -> None:
    provider, _ = pick_provider([
        '{"duplicates": ["4182", "4201"]}',
        "Hi, we refunded you.",
        '{"approved": false, "fix": "Say how much, and on which orders."}',
        "Dear Ada, we have refunded 80 EUR for the duplicate charges on orders "
        "4182 and 4201. It reaches your card in three to five working days.",
        '{"approved": true, "fix": ""}',
    ])
    harness = Harness(provider=provider)
    workflow = Workflow.from_text(
        WORKFLOW, tools=[find_charges, customer_profile, issue_refund],
        harness=harness)

    print(workflow.describe(), "\n")

    async for event in workflow.stream({"customer": "c_17",
                                        "orders": ["4182", "4190", "4201"]}):
        if event.type == "step_end":
            print(f"  ✓ {event.step:<10} {event.kind:<9} {event.text[:70]!r}")
        elif event.type in ("step_skipped", "step_failed", "step_retry"):
            print(f"  · {event.step:<10} {event.type[5:]} {event.text[:70]}")
        elif event.type == "workflow_end":
            result = event.data["result"]

    print(f"\n{result.status}: {result.output}")
    print(f"state: refunded={result.state['refunded']} · "
          f"drafts={result.steps['email'].iterations} · "
          f"{result.steps_run} steps · ${result.cost_usd:.4f}")
    await harness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
