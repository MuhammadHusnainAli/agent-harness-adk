"""`Governance`: one object that turns the controls on.

    gov = Governance.from_packs(["eu-ai-act", "gdpr", "ksa-pdpl", "owasp-agentic"],
                                policy="governance.yaml", home="eu",
                                signing_key=os.environ["AUDIT_KEY"])
    harness = Harness.local(".harness", governance=gov)

From then on every run on that harness is governed. It works entirely through
the hook points the loop already has — `run_start`, `model_egress`,
`pre_tool`, `post_tool`, `subagent_start`, `memory_write`, `run_end` — so it
adds nothing to a harness that does not use it, and it sees exactly what the
loop does: the real provider and region of each model call, the final
arguments of each tool call, each hand-off to a sub-agent.

Every decision is recorded in the audit trail with the policy hash that made
it. `mode="monitor"` makes the same decisions and records them without
enforcing any — the way to roll governance out on a live system.
"""

from __future__ import annotations

import json
import time
import weakref
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from ..errors import ConfigurationError
from ..types import ModelResponse, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from .data import (
    PERSONAL,
    Classification,
    DataClassifier,
    Pseudonymizer,
    expand_classes,
    known_classes,
)
from .identity import AgentIdentity, DelegationChain, Principal
from .incidents import Incident, IncidentDesk
from .inventory import AIInventory
from .monitor import RunState, RuntimeMonitor, current_run
from .oversight import ApprovalRequest, OversightDesk
from .packs import Pack, load_pack
from .policy import Decision, Policy, PolicyEngine, ResidencyConfig
from .records import HMACSigner, RetentionSweeper, SubjectVault
from .residency import Region, RegionResolver
from .risk import RiskAssessment, assess, impact_assessment
from .transparency import JURISDICTION_LANGUAGE, ProvenanceManifest, disclosure, label

__all__ = ["Governance", "Mode"]

Mode = Literal["enforce", "monitor"]


class Governance:
    """Policy, identity, data, residency, oversight, records — attached to a harness."""

    def __init__(
        self,
        policy: str | Path | Mapping[str, Any] | Policy | None = None,
        *,
        packs: Iterable[str | Pack] = (),
        mode: Mode = "enforce",
        home: str | None = None,
        home_region: str | None = None,
        regions: Mapping[str, str] | None = None,
        signer: Any = None,
        signing_key: bytes | str | None = None,
        approver: Any = None,
        approval_timeout: float = 900.0,
        approval_notify: Any = None,
        vault: SubjectVault | str | Path | None = None,
        inventory: AIInventory | str | Path | None = None,
        classifier: DataClassifier | None = None,
        monitor: RuntimeMonitor | None = None,
        notify: Iterable[Any] = (),
        service_provider: str = "",
        pseudonym_key: bytes | str | None = None,
        audit_args: Literal["scrubbed", "tokens"] = "scrubbed",
    ) -> None:
        if mode not in ("enforce", "monitor"):
            raise ConfigurationError(f"mode must be 'enforce' or 'monitor', not {mode!r}")
        self.mode: Mode = mode
        self.packs: list[Pack] = [p if isinstance(p, Pack) else load_pack(p) for p in packs]
        merged = Policy.load(policy).merge(*(p.policy for p in self.packs))
        home = home or home_region
        if home:
            merged.residency = (merged.residency or ResidencyConfig()).model_copy(
                update={"home": home})
        elif merged.residency is not None and merged.residency.home is None:
            # One jurisdiction restricting transfers is the obvious home. With
            # several, a principal without a jurisdiction would be unrestricted
            # — the evidence report flags that until `home=` is set.
            restricted = [o for o, d in merged.residency.transfers.items() if "*" not in d]
            if len(restricted) == 1:
                merged.residency.home = restricted[0]
        self.policy = merged
        self.engine = PolicyEngine(merged)
        self.classifier = classifier or DataClassifier()
        # A misspelt class in a deny rule would never match — fail open. Refuse it.
        known = known_classes([*merged.data_classes,
                               *(cls for cls, _ in self.classifier.extra.values())])
        unknown = sorted(merged.class_names() - known)
        if unknown:
            raise ConfigurationError(
                f"unknown data classes in the policy: {', '.join(unknown)} — known: "
                f"{', '.join(sorted(known))}; declare your own under `data_classes:`")

        if signer is None and signing_key:
            signer = HMACSigner(signing_key)
        self.signer = signer
        self.resolver = RegionResolver(regions)
        self.pseudonymizer = Pseudonymizer(pseudonym_key)
        self.vault = vault if isinstance(vault, SubjectVault) else SubjectVault(vault)
        self.inventory = (inventory if isinstance(inventory, AIInventory)
                          else AIInventory(inventory))
        self.monitor = monitor or RuntimeMonitor()
        self.oversight = OversightDesk(approver, default_timeout=approval_timeout,
                                       record=self._record_approval,
                                       notify=approval_notify)
        self.incidents = IncidentDesk({p.id: p.incidents for p in self.packs},
                                      notify=notify, record=self._record_incident)
        self.retention = RetentionSweeper(merged.data_retention_days)
        self.service_provider = service_provider
        #: How tool arguments enter the audit trail: "scrubbed" replaces the
        #: personal values the classifier can find; "tokens" replaces *every*
        #: string argument with the person's token — for when arguments may hold
        #: names or free text no pattern can recognise.
        if audit_args not in ("scrubbed", "tokens"):
            raise ConfigurationError("audit_args is 'scrubbed' or 'tokens'")
        self.audit_args = audit_args

        self.harness: Any = None
        self.identities: dict[str, AgentIdentity] = {}
        self.assessments: dict[str, RiskAssessment] = {}
        self.stats: Counter[tuple[str, str]] = Counter()
        self.data_seen: Counter[str] = Counter()
        self._agents: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._states: dict[str, RunState] = {}
        self._rights: Any = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_packs(cls, packs: Iterable[str | Pack], policy: Any = None,
                   **kwargs: Any) -> Governance:
        return cls(policy, packs=packs, **kwargs)

    @classmethod
    def from_file(cls, path: str | Path, **overrides: Any) -> Governance:
        """A governance config file: ``packs``, ``policy``, ``home``, ``regions``, ``mode``.

        The signing key is never read from the file — pass `signing_key=` or set
        ``signing_key_env`` to the name of an environment variable holding it.
        """
        import os

        import yaml

        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(config, Mapping):
            raise ConfigurationError(f"{path}: a governance config must be a mapping")
        known = {"packs", "policy", "home", "regions", "mode", "vault", "inventory",
                 "service_provider", "approval_timeout", "signing_key_env"}
        unknown = sorted(set(config) - known)
        if unknown:
            raise ConfigurationError(f"{path}: unknown keys {', '.join(unknown)}")
        kwargs: dict[str, Any] = {k: v for k, v in config.items()
                                  if k in known - {"packs", "policy", "signing_key_env"}}
        env = config.get("signing_key_env")
        if env and "signing_key" not in overrides:
            if not os.environ.get(env):
                raise ConfigurationError(f"{path}: ${env} is not set")
            kwargs["signing_key"] = os.environ[env]
        kwargs.update(overrides)
        return cls(config.get("policy"), packs=config.get("packs", []), **kwargs)

    def attach(self, harness: Any) -> Governance:
        """Wire into a harness. Called for you by ``Harness(governance=...)``."""
        if self.harness is harness:
            return self
        if self.harness is not None:
            raise ConfigurationError("this Governance is already attached to a harness")
        if self.signer is not None:
            if harness.audit.signer is None and any(not e.signature
                                                    for e in harness.audit.entries):
                raise ConfigurationError(
                    "the audit trail already holds unsigned entries; a signed trail "
                    "must be signed from its first entry — start a new audit file")
            harness.audit.signer = harness.audit.signer or self.signer
        self.harness = harness
        # Personal data reaches the trail only as the current person's tokens —
        # including what the loop itself records (task text, tool arguments).
        harness.audit.scrub = self._scrub_entry
        for event, handler in (
            ("run_start", self._on_run_start), ("run_end", self._on_run_end),
            ("model_egress", self._on_egress), ("post_model", self._on_post_model),
            ("pre_tool", self._on_pre_tool), ("post_tool", self._on_post_tool),
            ("subagent_start", self._on_subagent_start),
            ("memory_write", self._on_memory_write),
        ):
            harness.hooks.add(event, self._guarded(event, handler))
        harness.audit.record("governance", "governance.attach", decision="ok",
                             policy=self.policy.hash[:16], mode=self.mode,
                             packs=[p.id for p in self.packs])
        return self

    def _guarded(self, event: str, handler: Any) -> Any:
        """A governance failure is a refusal, never a crash and never a pass.

        In enforce mode the action is blocked and the error recorded; in
        monitor mode it is recorded and the action goes ahead.
        """
        async def run(hook: Any) -> None:
            try:
                await handler(hook)
            except Exception as exc:
                self.stats[("error", event)] += 1
                self._audit(hook.agent or "governance", "governance.error", event,
                            "deny" if self.mode == "enforce" else "log",
                            run_id=hook.run_id, error=f"{type(exc).__name__}: {exc}")
                if self.mode == "enforce" and event != "run_end":
                    hook.block(f"governance could not decide ({type(exc).__name__}); "
                               "refused")
        run.__name__ = getattr(handler, "__name__", event)
        return run

    #: Loop records whose target is the task's free text.
    _FREE_TEXT_TARGETS = frozenset({"run_start", "run_end"})

    def _scrub_entry(self, action: str, target: str,
                     detail: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """What the audit trail may keep of one entry.

        A task's text can name anyone, and no pattern finds every name, so it
        enters the immutable trail only as the person's token (the readable
        text stays in the run journal, which erasure rewrites). Everything else
        is kept, with the personal values the classifier finds tokenised.
        """
        state = current_run.get()
        subject = (state.principal.get("subject", "") if state else "") or "_anonymous"
        try:
            if action in self._FREE_TEXT_TARGETS and target:
                target = self.vault.token(subject, target, "task")
            else:
                target = self._scrub(target, subject)
            clean: dict[str, Any] = {}
            for key, value in detail.items():
                if key == "args" and self.audit_args == "tokens":
                    clean[key] = _map_strings(value, lambda s: self.vault.token(
                        subject, s, "arg") if s else s)
                else:
                    clean[key] = self._scrub(value, subject)
            return target, clean
        except Exception:
            # Unable to tell what is personal: keep nothing rather than everything.
            self.stats[("error", "audit_scrub")] += 1
            return _withhold(target), _withhold(detail)

    def register_agent(self, agent: Any) -> AgentIdentity:
        """Called by `Agent.__init__` on a governed harness."""
        identity = AgentIdentity.of(agent.identity, name=agent.name)
        assessment = assess(identity)
        if assessment.prohibited:
            self._audit("governance", "governance.register", agent.name, "deny",
                        reasons=assessment.reasons)
            if self.mode == "enforce":
                raise ConfigurationError(
                    f"{agent.name}: prohibited use — " + "; ".join(assessment.reasons))
        if self.packs:
            # Report the classification under the frameworks actually in force.
            assessment = assessment.only({p.id for p in self.packs})
        if identity.risk is None or identity.risk != assessment.tier:
            identity = identity.model_copy(update={"risk": assessment.tier})
        self.identities[agent.name] = identity
        self.assessments[agent.name] = assessment
        self._agents[agent.name] = agent
        provider = agent._provider if not isinstance(agent._provider, (str, type(None))) \
            else None
        region = self.resolver.resolve(provider, agent.model) if provider else None
        self.inventory.register(agent, identity=identity, region=region)
        self._audit("governance", "governance.register", agent.name, "ok",
                    risk=assessment.tier, owner=identity.owner, purpose=identity.purpose,
                    frameworks=assessment.frameworks)
        return identity

    # ------------------------------------------------------------------
    # deciding and recording
    # ------------------------------------------------------------------
    def decide(self, action: str, ctx: Mapping[str, Any], *, target: str = "") -> Decision:
        decision = self.engine.decide(action, ctx, target=target)
        decision.enforced = self.mode == "enforce"
        return decision

    def _audit(self, actor: str, action: str, target: str, decision: str, *,
               run_id: str = "", **detail: Any) -> None:
        if self.harness is not None:
            self.harness.audit.record(actor, action, target=target, decision=decision,
                                      run_id=run_id, **detail)

    def _record(self, decision: Decision, state: RunState | None, *,
                classes: Iterable[str] = (), **detail: Any) -> None:
        self.stats[(decision.action, decision.effect)] += 1
        if state is not None and not decision.allowed:
            state.denials += 1
        subject = (state.principal.get("subject", "") if state else "")
        self._audit(
            state.agent if state else "governance", f"governance.{decision.action}",
            decision.target, decision.effect, run_id=state.run_id if state else "",
            policy=decision.policy, rules=decision.rules, reasons=decision.reasons,
            controls=decision.controls, enforced=decision.enforced,
            data=sorted(classes), subject=self.vault.subject_ref(subject),
            **{k: self._scrub(v, subject) for k, v in detail.items()},
        )

    def _scrub(self, value: Any, subject: str) -> Any:
        """Personal values → per-subject tokens, so erasure works on the record."""
        if isinstance(value, str):
            found = [f for f in self.classifier.scan(value).findings
                     if f.data_class in PERSONAL and f.kind != "keyword"]
            if not found:
                return value
            out, cursor = [], 0
            for f in sorted(found, key=lambda f: f.span[0]):
                if f.span[0] < cursor:
                    continue
                out += [value[cursor:f.span[0]],
                        self.vault.token(subject or "_anonymous", f.value(value), f.kind)]
                cursor = f.span[1]
            return "".join(out) + value[cursor:]
        if isinstance(value, Mapping):
            return {k: self._scrub(v, subject) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub(v, subject) for v in value]
        return value

    def _record_approval(self, request: ApprovalRequest, action: str) -> None:
        self._audit(request.agent, f"governance.{action}", request.target,
                    request.status, run_id=request.run_id, approval=request.id,
                    approvers=request.approvers, rules=request.rules,
                    votes=[{"by": v.by, "approve": v.approve, "note": v.note}
                           for v in request.votes])

    def _record_incident(self, incident: Incident) -> None:
        self._audit(incident.agent or "governance", "governance.incident", incident.kind,
                    incident.severity, run_id=incident.run_id, incident=incident.id,
                    title=incident.title,
                    deadlines=[f"{d.framework}: {d.notify} by {d.due_iso}"
                               for d in incident.deadlines])

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def _state(self, run_id: str = "") -> RunState | None:
        return self._states.get(run_id) or current_run.get()

    def _principal(self, trace: Any, parent: RunState | None) -> dict[str, Any]:
        if parent is not None:
            return parent.principal       # a sub-agent acts for the same person
        residency = self.policy.residency
        return Principal.from_trace(trace, home=residency.home if residency else None
                                    ).context()

    def _context(self, state: RunState | None, **extra: Any) -> dict[str, Any]:
        identity = state.chain.head if state else None
        ctx: dict[str, Any] = {
            "agent": {**(identity.model_dump() if identity else {}),
                      "name": state.agent if state else ""},
            "principal": state.principal if state else {},
            "purpose": state.purpose if state else "",
            "run": state.context() if state else {},
            "delegation": {"depth": state.chain.depth if state else 0,
                           "chain": state.chain.names if state else []},
            "data": self._data_context(state.classes if state else set()),
        }
        ctx.update(extra)
        return ctx

    @staticmethod
    def _data_context(classes: Iterable[str]) -> dict[str, Any]:
        present = set(classes)
        return {"classes": sorted(present), "personal": "personal" in present,
                "special_category": "special_category" in present,
                **{c: True for c in present}}

    def _classify(self, state: RunState | None, *texts: str,
                  declared: Iterable[str] = ()) -> Classification:
        result = Classification(declared=frozenset(declared))
        for text in texts:
            if text:
                result = result | self.classifier.scan(text)
        if state is not None:
            state.taint(result.classes)
        return result

    def _enforce(self, decision: Decision) -> bool:
        """True when this decision should actually stop something."""
        return self.mode == "enforce" and not decision.allowed

    # ------------------------------------------------------------------
    # hook handlers
    # ------------------------------------------------------------------
    async def _on_run_start(self, hook: Any) -> None:
        parent = current_run.get()
        if parent is not None and parent.run_id not in self._states:
            parent = None     # left behind by a run that ended without `run_end`
        identity = self.identities.get(hook.agent) or AgentIdentity(name=hook.agent)
        chain = parent.chain.child(identity) if parent else DelegationChain((identity,))
        principal = self._principal(hook.data.get("trace"), parent)
        purpose = (principal.get("purpose") or identity.purpose
                   or (parent.purpose if parent else ""))
        state = RunState(run_id=hook.run_id, agent=hook.agent, chain=chain,
                         principal=principal, purpose=purpose, parent=parent)
        self._states[hook.run_id] = state
        current_run.set(state)
        found = self._classify(state, hook.data.get("task", ""))

        decision = self.decide("run", self._context(state), target=hook.agent)
        if identity.risk == "prohibited":
            decision.tighten("deny", f"{hook.agent} is a prohibited use", rule="risk",
                             controls=["C12"])
        if parent is not None:
            ok, why = parent.chain.may_delegate(self.policy.max_delegation_depth)
            if not ok:
                decision.tighten("deny", why, rule="delegation", controls=["C2"])
        if self.policy.purposes and purpose not in self.policy.purposes:
            decision.tighten("deny", f"purpose {purpose or '(none)'!r} is not declared "
                             f"in the policy", rule="purpose", controls=["C7"])
        self._record(decision, state, classes=found.classes,
                     depth=chain.depth, purpose=purpose)
        if self._enforce(decision):
            hook.block(decision.reason)

    async def _on_run_end(self, hook: Any) -> None:
        state = self._states.pop(hook.run_id, None)
        result = hook.data.get("result")
        if state is None or result is None:
            return
        current_run.set(state.parent)
        output = result.output or ""
        found = self._classify(state, output)
        decision = self.decide("output", self._context(state), target=state.agent)
        if not self._within_limits(state, found):
            decision.tighten("redact", "the output carries data this run may not hand out",
                             rule="purpose", controls=["C5", "C7"])
            decision.redact = sorted(self._disallowed(state, found))
        if decision.effect == "deny" and decision.enforced:
            result.output = f"[withheld by policy: {decision.reason}]"
        elif decision.effect == "redact" and decision.enforced and output:
            result.output = self._redact_text(output, decision, state)
        self._record(decision, state, classes=found.classes)

        transparency = self.policy.transparency
        if state.top_level:
            if transparency.disclose:
                result.disclosure = disclosure(transparency.languages)
            reader = state.principal.get("jurisdiction") or ""
            if transparency.label_output and result.output and (
                    "*" in transparency.label_for or reader in transparency.label_for):
                language = JURISDICTION_LANGUAGE.get(reader, "en")
                result.output = label(result.output, [language])
            if transparency.provenance and result.output:
                manifest = ProvenanceManifest.for_output(
                    result.output, agent=state.agent, run_id=state.run_id,
                    models=sorted(state.models), providers=sorted(state.providers),
                    service_provider=self.service_provider, policy=self.engine.hash[:16],
                    labels={"explicit": transparency.label_output,
                            "implicit": True, "languages": transparency.languages})
                if self.signer is not None:
                    manifest.sign(self.signer)
                result.provenance = manifest.model_dump(mode="json")
        result.governance = {
            "policy": self.engine.hash[:16], "mode": self.mode,
            "data_classes": sorted(state.classes), "regions": sorted(state.regions),
            "denials": state.denials, "untrusted": state.untrusted,
            "delegation_depth": state.chain.depth,
        }
        for cls in state.classes:
            self.data_seen[cls] += 1
        subject = state.principal.get("subject", "")
        if subject:
            # Where this person's data now lives, so erasure can find all of it.
            if getattr(result, "session_id", ""):
                self.vault.note(subject, "sessions", result.session_id)
            self.vault.note(subject, "runs", state.run_id)
            if getattr(result, "trace_id", ""):
                self.vault.note(subject, "traces", result.trace_id)

    async def _on_egress(self, hook: Any) -> None:
        state = self._state(hook.run_id)
        request = hook.replacement if hook.replaced else hook.data["request"]
        provider, model = hook.data.get("provider"), hook.data.get("model", "")
        region = self.resolver.resolve(provider, model)
        found = self._classify(state, *_request_texts(request))
        classes = state.classes if state else set(found.classes)
        pname = getattr(provider, "name", "") or ""
        ctx = self._context(state, provider={"name": pname, "model": model,
                                             "region": region.jurisdiction,
                                             "location": region.location})
        ctx["data"] = self._data_context(classes)
        decision = self.decide("egress", ctx, target=f"{pname}:{model}")
        origin = self._residency_check(decision, state, region, classes)
        if state is not None and not self._within_limits(state, found, classes):
            decision.tighten("redact", "purpose and delegation limits: remove data this run "
                             "may not use", rule="purpose", controls=["C5", "C7"])
            decision.redact = sorted({*decision.redact,
                                      *self._disallowed(state, found, classes)})

        if decision.effect == "require_approval":
            await self._settle_approval(decision, state, hook.agent,
                                        {"provider": pname, "model": model,
                                         "region": region.jurisdiction})
        self._record(decision, state, classes=classes, region=region.jurisdiction,
                     location=region.location, origin=origin or "")
        if self._enforce(decision):
            hook.block(decision.reason)
            return
        if state is not None:
            state.models.add(model)
            state.providers.add(pname)
            state.regions.add(region.jurisdiction)
        self.inventory.saw_region(hook.agent, region.jurisdiction)
        if origin and region.jurisdiction not in (origin, "on_prem") and \
                classes & expand_classes(self._residency.applies_to):
            self._audit(hook.agent, "governance.transfer", region.jurisdiction, "allow",
                        run_id=hook.run_id, origin=origin, destination=region.jurisdiction,
                        location=region.location, provider=pname, model=model,
                        data=sorted(classes))
        if decision.effect == "redact" and decision.enforced:
            hook.replace(self._redact_request(request, decision, state))

    @property
    def _residency(self) -> ResidencyConfig:
        return self.policy.residency or ResidencyConfig()

    def _residency_check(self, decision: Decision, state: RunState | None, region: Region,
                         classes: set[str]) -> str | None:
        residency = self.policy.residency
        origin = (state.principal.get("jurisdiction") if state else None) or \
            (residency.home if residency else None)
        chain_regions = state.chain.allowed_regions() if state else None
        if chain_regions is not None and region.jurisdiction not in chain_regions \
                and region.jurisdiction != "on_prem":
            decision.tighten("reroute", f"{state.agent if state else 'agent'} may only use "
                             f"models in {', '.join(sorted(chain_regions))}",
                             rule="identity.regions", controls=["C2", "C6"])
        if residency is None or not (classes & expand_classes(residency.applies_to)):
            return origin
        allowed = residency.destinations(origin)
        if allowed is None:
            return origin
        dest = region.jurisdiction
        if dest == "on_prem" and residency.on_prem_always_allowed:
            return origin
        if not region.known:
            if residency.unknown == "deny":
                decision.tighten("reroute", f"cannot tell where {region.location or 'this model'}"
                                 " processes data, and personal data is present — "
                                 "declare its region", rule="residency", controls=["C6"])
            return origin
        if dest not in allowed:
            decision.tighten("reroute", f"personal data of a subject in {origin} may not "
                             f"be sent to {dest} (allowed: {', '.join(allowed)})",
                             rule="residency", controls=["C6"])
        return origin

    def _disallowed(self, state: RunState, found: Classification,
                    classes: set[str] | None = None) -> set[str]:
        present = (classes if classes is not None else set(found.classes)) & PERSONAL
        allowed: set[str] | None = None
        if self.policy.purposes and state.purpose in self.policy.purposes:
            allowed = expand_classes(self.policy.purposes[state.purpose])
        chain = state.chain.allowed_data()
        if chain is not None:
            chain = expand_classes(chain)
            allowed = chain if allowed is None else allowed & chain
        if allowed is None:
            return set()
        return present - allowed

    def _within_limits(self, state: RunState, found: Classification,
                        classes: set[str] | None = None) -> bool:
        return not self._disallowed(state, found, classes)

    def _redact_text(self, text: str, decision: Decision, state: RunState | None) -> str:
        classes = decision.redact or ["personal"]
        subject = state.principal.get("subject", "") if state else ""
        if decision.redact_mode == "pseudonymize":
            if state is not None:
                state.pseudonymized = True
            return self.classifier.redact(text, classes, pseudonymizer=self.pseudonymizer,
                                          subject=subject)
        return self.classifier.redact(text, classes)

    def _redact_request(self, request: Any, decision: Decision,
                        state: RunState | None) -> Any:
        sent = request.model_copy(deep=True)
        if sent.system:
            sent.system = self._redact_text(sent.system, decision, state)
        for message in sent.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    block.text = self._redact_text(block.text, decision, state)
                elif isinstance(block, ToolResultBlock):
                    block.content = self._redact_text(block.content, decision, state)
                elif isinstance(block, ToolUseBlock):
                    block.input = _map_strings(block.input, lambda s: self._redact_text(
                        s, decision, state))
        return sent

    async def _on_post_model(self, hook: Any) -> None:
        state = self._state(hook.run_id)
        if state is None or not state.pseudonymized:
            return
        response: ModelResponse = hook.replacement if hook.replaced else hook.data["response"]
        restore = self.pseudonymizer.restore
        message = response.message.model_copy(deep=True)
        changed = False
        for block in message.content:
            if isinstance(block, TextBlock) and "⟨" in block.text:
                block.text, changed = restore(block.text), True
            elif isinstance(block, ToolUseBlock):
                restored = self.pseudonymizer.restore_value(block.input)
                if restored != block.input:
                    block.input, changed = restored, True
        if changed:
            hook.replace(response.model_copy(update={"message": message}))

    async def _on_pre_tool(self, hook: Any) -> None:
        state = self._state(hook.run_id)
        name = hook.data.get("tool", "")
        args = hook.replacement if hook.replaced and isinstance(hook.replacement, dict) \
            else hook.data.get("args", {})
        tags = list(hook.data.get("tags", []))
        found = self._classify(state, json.dumps(args, default=str))
        ctx = self._context(state, tool={"name": name, "tags": tags,
                                         "permission": hook.data.get("permission")},
                            args=args)
        ctx["data"] = self._data_context((state.classes if state else set())
                                         | found.classes)
        decision = self.decide("tool", ctx, target=name)

        if state is not None:
            ok, why = state.chain.allows_tool(name, tags)
            if not ok:
                decision.tighten("deny", why, rule="identity.tools", controls=["C2"])
            loop = self.monitor.on_tool(state, name, args)
            if loop:
                decision.tighten("deny", loop, rule="monitor", controls=["C14"])
                key = f"breaker:{name}" if "breaker" in loop else f"loop:{name}"
                await self._incident_once(state, "runaway", key, loop,
                                          agent=hook.agent, tool=name)
        drift = self._drift(hook.agent, name)
        if drift and self.policy.tool_drift != "allow":
            decision.tighten("deny" if self.policy.tool_drift == "deny" else "log", drift,
                             rule="supply_chain", controls=["C13"])

        if decision.effect == "require_approval":
            await self._settle_approval(decision, state, hook.agent, dict(args))
        self._record(decision, state, classes=found.classes, args=args, tags=tags)
        if self._enforce(decision):
            hook.block(decision.reason)
        elif decision.effect == "redact" and decision.enforced:
            hook.replace(_map_strings(args, lambda s: self._redact_text(s, decision, state)))

    def _drift(self, agent_name: str, tool_name: str) -> str | None:
        agent = self._agents.get(agent_name)
        if agent is None or tool_name not in agent.tools:
            return None
        return self.inventory.check_tool(agent_name, agent.tools.get(tool_name))

    async def _on_post_tool(self, hook: Any) -> None:
        state = self._state(hook.run_id)
        outcome = hook.data.get("outcome")
        if outcome is None:
            return
        name = hook.data.get("tool", "")
        if state is not None and self.monitor.on_tool_outcome(state, name, outcome.is_error):
            await self._incident_once(
                state, "runaway", f"breaker:{name}",
                f"{name} failed {state.failures[name]} times in a row; breaker open",
                agent=hook.agent, tool=name)
        if outcome.is_error:
            return
        content = str(hook.replacement) if hook.replaced else outcome.content
        declared = [t[5:] for t in hook.data.get("tags", ()) if t.startswith("data:")]
        agent = self._agents.get(hook.agent)
        if not declared and agent is not None and hook.data.get("tool") in agent.tools:
            declared = [t[5:] for t in agent.tools.get(hook.data["tool"]).tags
                        if t.startswith("data:")]
        found = self._classify(state, content, declared=declared)
        score = self.monitor.on_tool_result(state, content) if state else 0.0
        ctx = self._context(state, tool={"name": name}, result={"injection": score})
        ctx["data"] = self._data_context((state.classes if state else set())
                                         | found.classes)
        decision = self.decide("tool_result", ctx, target=name)
        if state is not None and not self._within_limits(state, found):
            decision.tighten("redact", "tool returned data this run may not use",
                             rule="purpose", controls=["C5", "C7"])
            decision.redact = sorted(self._disallowed(state, found))
        if decision.effect != "allow" or found.classes or score:
            self._record(decision, state, classes=found.classes, injection=round(score, 2))
        if self._enforce(decision):
            hook.replace(f"[withheld by policy: {decision.reason}]")
        elif decision.effect == "redact" and decision.enforced:
            hook.replace(self._redact_text(content, decision, state))

    async def _on_subagent_start(self, hook: Any) -> None:
        state = current_run.get()
        child = hook.agent
        chain = state.chain if state else DelegationChain()
        ctx = self._context(state, child={"name": child},
                            delegation={"depth": chain.depth + 1, "chain": chain.names,
                                        "child": child})
        decision = self.decide("delegate", ctx, target=child)
        ok, why = chain.may_delegate(self.policy.max_delegation_depth)
        if not ok:
            decision.tighten("deny", why, rule="delegation", controls=["C2"])
        if state is not None:
            storm = self.monitor.on_delegate(state)
            if storm:
                decision.tighten("deny", storm, rule="monitor", controls=["C14"])
        child_identity = self.identities.get(child)
        if child_identity is None:
            decision.tighten("deny", f"{child} is not registered with governance — it was "
                             "built on a different harness, so its runs would go "
                             "unchecked", rule="registration", controls=["C1", "C2"])
        elif child_identity.risk == "prohibited":
            decision.tighten("deny", f"{child} is a prohibited use", rule="risk")
        self._record(decision, state, depth=chain.depth + 1)
        if self._enforce(decision):
            hook.block(decision.reason)

    async def _on_memory_write(self, hook: Any) -> None:
        state = current_run.get()
        text = str(hook.replacement) if hook.replaced else hook.data.get("text", "")
        found = self._classify(None, text)       # what is kept, not what the run saw
        ctx = self._context(state, memory={"scope": hook.data.get("scope"),
                                           "kind": hook.data.get("kind")})
        if state is None:                        # written outside a run
            ctx["principal"] = self._principal(hook.data.get("trace"), None)
        ctx["data"] = self._data_context(found.classes)
        decision = self.decide("memory", ctx, target=str(hook.data.get("scope", "")))
        if "credentials" in found.classes:
            decision.tighten("redact", "credentials are never kept in memory",
                             rule="builtin.credentials", controls=["C5"])
            decision.redact = sorted({*decision.redact, "credentials"})
        self._record(decision, state, classes=found.classes)
        if self._enforce(decision):
            hook.block(decision.reason)
        elif decision.effect == "redact" and decision.enforced:
            hook.replace(self._redact_text(text, decision, None))

    async def _settle_approval(self, decision: Decision, state: RunState | None,
                               agent: str, args: dict[str, Any]) -> None:
        """Ask the people. A yes clears the gate (other obligations stay); anything
        else is a deny. In monitor mode nobody is asked and the decision is
        recorded as `require_approval` — what would have happened."""
        if self.mode == "monitor":
            return
        if await self._approve(decision, state, agent, args):
            decision.effect = "redact" if decision.redact else "log"
            decision.reasons.append("approved")
        else:
            decision.tighten("deny", "approval was not given", rule="oversight",
                             controls=["C8"])

    async def _incident_once(self, state: RunState, kind: str, key: str, title: str,
                             **detail: Any) -> None:
        """One incident per cause per run, however often the cause repeats."""
        if key in state.incidents:
            return
        state.incidents.add(key)
        await self.incidents.open(kind, "medium", title, run_id=state.run_id, **detail)

    async def _approve(self, decision: Decision, state: RunState | None, agent: str,
                       args: dict[str, Any]) -> bool:
        if self.mode == "monitor":
            return True
        request = ApprovalRequest(
            agent=agent, action=decision.action, target=decision.target,
            reason=decision.reason, args=args, rules=list(decision.rules),
            approvers=decision.approvers, timeout=decision.timeout,
            requester=(state.principal.get("user", "") if state else ""),
            run_id=state.run_id if state else "")
        result = await self.oversight.request(request)
        return result.status == "approved"

    # ------------------------------------------------------------------
    # the things people ask for
    # ------------------------------------------------------------------
    @property
    def rights(self) -> Any:
        """Data-subject requests: `access(trace)` and `erase(trace)`."""
        if self._rights is None:
            from .rights import SubjectRights

            self._rights = SubjectRights(self)
        return self._rights

    def report(self, pack: str | None = None) -> Any:
        """The evidence report — for one pack, or every active one."""
        from .evidence import ComplianceReport

        return ComplianceReport.build(self, pack)

    def impact_assessment(self, agent: str, kind: str = "dpia") -> str:
        identity = self.identities.get(agent)
        if identity is None:
            raise ConfigurationError(f"{agent} is not registered with governance")
        return impact_assessment(identity, inventory=self.inventory.agents.get(agent),
                                 policy=self.policy, kind=kind,
                                 data_classes=[c for c, n in self.data_seen.items() if n])

    async def sweep(self) -> dict[str, Any]:
        """Apply the data-retention limit now."""
        if self.harness is None:
            raise ConfigurationError("attach governance to a harness first")
        return await self.retention.sweep(self.harness, vault=self.vault)

    def summary(self) -> dict[str, Any]:
        decisions: dict[str, dict[str, int]] = {}
        for (action, effect), count in self.stats.items():
            decisions.setdefault(action, {})[effect] = count
        return {
            "policy": f"{self.policy.name}@{self.policy.version}",
            "policy_hash": self.engine.hash[:16], "mode": self.mode,
            "packs": [p.id for p in self.packs], "agents": len(self.identities),
            "decisions": decisions,
            "approvals_pending": len(self.oversight.pending()),
            "incidents_open": len(self.incidents.open_incidents()),
            "data_seen": dict(self.data_seen),
            "signed": self.signer is not None,
            "checked_at": time.time(),
        }


def _request_texts(request: Any) -> list[str]:
    texts: list[str] = [request.system or ""]
    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ToolResultBlock):
                texts.append(block.content)
            elif isinstance(block, ToolUseBlock):
                texts.append(json.dumps(block.input, default=str))
            elif isinstance(block, ThinkingBlock):
                continue
    return texts


def _withhold(value: Any) -> Any:
    if isinstance(value, str):
        return "[withheld]" if value else value
    if isinstance(value, Mapping):
        return {k: _withhold(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_withhold(v) for v in value]
    return value


def _map_strings(value: Any, fn: Any) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value

