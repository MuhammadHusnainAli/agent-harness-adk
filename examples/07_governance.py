"""Governance: one harness serving EU and Saudi customers under their own laws.

Turn on the packs for the markets you operate in and every run is governed:
each model call is checked against where the customer's data may go, refunds
over a threshold wait for a person, every decision is signed into the audit
trail, a customer can be erased without breaking it, and the evidence report
writes itself.

This example always uses a scripted provider and *declares* where its models
run (``regions=``) — residency is about real endpoints, so with a real key you
would point `regions` at your actual Bedrock/Vertex/Azure deployments instead.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agent_harness import Agent, FakeProvider, Harness, ModelRouter, Trace, tool, tool_call
from agent_harness.governance import AgentIdentity, ApprovalVote, Governance


@tool
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: the order number
    """
    return f"order {order_id}: delivered, damaged in transit"


@tool(tags=["payments", "irreversible"])
def issue_refund(order_id: str, amount: float) -> str:
    """Refund a customer. Costs real money.

    Args:
        order_id: the order to refund
        amount: how much, in EUR
    """
    return f"refunded {amount:.2f} EUR on order {order_id}"


def supervisor(request) -> ApprovalVote:
    """Stands in for a person answering in your back office."""
    amount = request.args["amount"]
    print(f"  ↳ approval asked: {request.target}({amount}) — {request.reason}")
    return ApprovalVote(amount <= 1000, by="maria@bank.example", note="within limits")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gov = Governance.from_packs(
            ["eu-ai-act", "gdpr", "ksa-pdpl", "owasp-agentic"],
            policy={
                "name": "retail-support", "version": "3",
                "purposes": {"customer_support": ["contact", "financial"]},
                "rules": [{"id": "refunds-over-100", "on": "tool",
                           "match": {"tags": ["payments"]}, "when": "args.amount > 100",
                           "effect": "require_approval",
                           "reason": "refunds over 100 EUR need a supervisor"}],
            },
            home="eu",
            # Where each model really runs. A us-hosted and an eu-hosted deployment.
            regions={"model:us-large": "us", "model:eu-large": "eu"},
            signing_key="example-key-change-me-0123456789",
            approver=supervisor,
            vault=Path(tmp) / "vault.json",
            service_provider="Example Bank",
        )
        harness = Harness.local(Path(tmp) / "state", trace=False, governance=gov,
                                router=ModelRouter(fallbacks=["eu-large"]))

        provider = FakeProvider([
            # Anna (EU): look up the order, refund 250 EUR, answer.
            tool_call("order_status", order_id="7731"),
            tool_call("issue_refund", order_id="7731", amount=250),
            "I've refunded 250 EUR for order 7731 — sorry it arrived damaged.",
        ])

        def support(trace: Trace) -> Agent:
            return Agent(
                "support", "Resolve order problems. Refund damaged goods.",
                provider=provider, model="us-large", harness=harness,
                tools=[order_status, issue_refund], trace=trace,
                identity=AgentIdentity(owner="cx-lead@bank.example",
                                       purpose="customer_support",
                                       tools=["order_status", "issue_refund"]),
            )

        print("1. An EU customer — personal data may not go to the US-hosted model")
        anna = Trace(tenant_id="bank", user_id="anna", tags={"jurisdiction": "eu"})
        result = await support(anna).run(
            "Order 7731 arrived broken. I'm anna@example.eu, please refund me.")
        print(f"  answer:  {result.output}")
        print(f"  served from: {result.governance['regions']}  "
              f"(data seen: {', '.join(result.governance['data_classes'])})")
        print(f"  disclosure: {result.disclosure['en']}")

        print("\n2. A Saudi customer — no model inside the Kingdom is configured")
        omar = Trace(tenant_id="bank", user_id="omar", tags={"jurisdiction": "sa"})
        refused = await support(omar).run("I'm omar@example.sa, where is order 9120?")
        print(f"  refused: {refused.error.split(' — ')[0]}")
        print("  (add a model in me-central2 / me-central-1 and it would be served there)")

        print("\n3. Anna asks to be forgotten")
        receipt = await gov.rights.erase(anna)
        print(f"  erased: {receipt['memory_records']} memory records, "
              f"{len(receipt['sessions'])} session(s); audit key destroyed: "
              f"{receipt['audit_key_destroyed']}")
        print(f"  audit chain after erasure: {harness.audit.verify()[1]}")

        print("\n4. Evidence, per framework")
        report = gov.report("eu-ai-act")
        counts = report.counts()
        print(f"  EU AI Act: {counts['met']} met, {counts['partial']} partial, "
              f"{counts['gap']} gap, {counts['manual']} for people to do")
        for cid in ("C1", "C6", "C8", "C10"):
            control = report.controls[cid]
            print(f"  {cid} {control.title}: {control.status} — {control.evidence[0]}")


if __name__ == "__main__":
    asyncio.run(main())
