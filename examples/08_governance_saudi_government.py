"""Governance for a Saudi government service: data stays in the Kingdom.

A citizen-services agent runs under the Saudi PDPL, SDAIA's AI principles and
the OWASP agentic list:

* a citizen's national ID is recognised (by its check digit) and their data is
  only ever sent to the in-Kingdom model — the overseas fallback is skipped;
* issuing a permit is irreversible, so two officials must approve it, through
  the approval queue your back office answers;
* the disclosure is in Arabic and English, a breach gets SDAIA's 72-hour clock,
  and a DPIA draft is filled in from what the harness knows.

Always runs on a scripted provider, with the regions of its two models declared.
"""

from __future__ import annotations

import asyncio

from agent_harness import Agent, FakeProvider, Harness, ModelRouter, Trace, tool, tool_call
from agent_harness.governance import AgentIdentity, Governance
from agent_harness.governance.data import luhn


def national_id() -> str:
    """A well-formed Saudi national ID for the demo (valid check digit)."""
    return next(f"109876543{d}" for d in "0123456789" if luhn(f"109876543{d}"))


@tool(tags=["irreversible", "consequential"])
def issue_permit(national_id: str, permit: str) -> str:
    """Issue a permit to a citizen. Cannot be undone.

    Args:
        national_id: the citizen's national ID
        permit: which permit
    """
    return f"permit {permit} issued to {national_id}"


async def officials(gov: Governance) -> None:
    """Two officials answering the queue, as your back-office UI would."""
    for name in ("khalid@gov.example", "noura@gov.example"):
        while not gov.oversight.pending():
            await asyncio.sleep(0.01)
        request = gov.oversight.pending()[0]
        print(f"  ↳ {name} approves {request.target} ({request.reason})")
        gov.oversight.approve(request.id, by=name)


async def main() -> None:
    gov = Governance.from_packs(
        ["ksa-pdpl", "sdaia", "owasp-agentic"],
        policy={"rules": [{"id": "permits-four-eyes", "on": "tool",
                           "match": {"tags": ["irreversible"]}, "effect": "require_approval",
                           "approvers": 2, "timeout": "10m",
                           "reason": "permits need two officials"}]},
        regions={"model:gov-llm-riyadh": "sa", "model:global-llm": "us"},
        signing_key="example-key-change-me-0123456789",
        approval_notify=lambda r: print(f"  ↳ queued for approval: {r.target}"),
    )
    # The overseas model is first in line; residency makes the harness skip it.
    harness = Harness.testing(governance=gov, router=ModelRouter(fallbacks=["gov-llm-riyadh"]))
    nid = national_id()
    agent = Agent(
        "citizen-services", "Help citizens apply for permits.",
        provider=FakeProvider([tool_call("issue_permit", national_id=nid, permit="events"),
                               "Your events permit has been issued."]),
        model="global-llm", harness=harness, tools=[issue_permit],
        trace=Trace(tenant_id="ministry", user_id="citizen-1", tags={"jurisdiction": "sa"}),
        identity=AgentIdentity(owner="digital-services@gov.example",
                               purpose="permit_issuance", domains=["essential_services"],
                               tools=["issue_permit"]),
    )

    print("1. A citizen applies — their data stays in the Kingdom, two officials sign off")
    answer, _ = await asyncio.gather(
        agent.run(f"I need an events permit. My national ID is {nid}."), officials(gov))
    print(f"  answer: {answer.output}")
    print(f"  served from: {answer.governance['regions']} · data: "
          f"{', '.join(answer.governance['data_classes'])}")
    print(f"  disclosure (ar): {answer.disclosure['ar']}")

    print("\n2. How this service is classified")
    assessment = gov.assessments["citizen-services"]
    print(f"  risk tier: {assessment.tier} (declared domain: essential services)")
    print("  under ksa-pdpl and sdaia: a DPIA, human oversight, records (see below)")

    print("\n3. A breach — the clock starts")
    breach = await gov.incidents.open("personal_data_breach", "high",
                                      "permit letters sent to the wrong addresses")
    for deadline in breach.deadlines:
        print(f"  {deadline.framework}: notify {deadline.notify} within "
              f"{deadline.within_hours:.0f} h")

    print("\n4. The DPIA, drafted from what the harness knows")
    print("  " + "\n  ".join(gov.impact_assessment("citizen-services").splitlines()[:9]))


if __name__ == "__main__":
    asyncio.run(main())
