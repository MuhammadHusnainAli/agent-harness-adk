"""The `agent-harness` command: run an agent, chat with one, inspect what happened."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent
from .harness import Harness
from .llm_providers.base import MODELS
from .runtime.permissions import console_approver
from .runtime.tracing import console_exporter

__all__ = ["main"]


def _harness(args: argparse.Namespace) -> Harness:
    harness = (Harness.local(args.state) if args.state else Harness())
    if args.trace:
        harness.tracer.add_exporter(console_exporter())
    if args.approve:
        harness.policy.approver = console_approver
    return harness


def _agent(args: argparse.Namespace, harness: Harness) -> Agent:
    tools: list[Any] = []
    if args.tools:
        from .toolkits import basic_tools, http_fetch
        tools = [*basic_tools(), http_fetch]
    return Agent(
        args.name,
        args.instructions or "",
        model=args.model,
        provider=args.provider,
        harness=harness,
        tools=tools,
        skills=args.skills,
        memory=not args.no_memory,
        max_steps=args.max_steps,
    )


async def _run(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    result = None
    try:
        if args.stream:
            async for event in agent.stream(args.task, session=args.session):
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "tool_result":
                    print(f"\n  · {event.data.get('tool')} → {event.text[:80]}",
                          file=sys.stderr)
                elif event.type == "run_end":
                    result = event.data["result"]
            print()
        else:
            result = await agent.run(args.task, session=args.session)
            print(result.output)

        if result is None:
            print("the run produced no result", file=sys.stderr)
            return 1
        if result.error:
            print(f"\nerror: {result.error}", file=sys.stderr)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        if args.report:
            print(json.dumps(harness.report(), indent=2), file=sys.stderr)
        print(f"\n[{result.steps} steps · ${result.cost_usd:.4f} · "
              f"session {result.session_id}]", file=sys.stderr)
        return 1 if result.error else 0
    finally:
        await harness.aclose()


async def _chat(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    session = args.session
    print(f"agent-harness {__version__} · {agent.name} on {agent.model}")
    print("Type /exit to leave, /new for a fresh session, /cost for the bill.\n")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "you › ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/new":
                session = None
                print("(new session)")
                continue
            if line == "/cost":
                print(json.dumps(harness.report()["budget"], indent=2))
                continue
            result = await agent.run(line, session=session)
            session = result.session_id
            print(f"\n{agent.name} › {result.output or result.error}\n")
        summary = await agent.close_session()
        if summary:
            print("(memory updated)", file=sys.stderr)
        return 0
    finally:
        await harness.aclose()


def _models(args: argparse.Namespace) -> int:
    rows = sorted(MODELS.values(), key=lambda m: (m.provider, m.id))
    print(f"{'model':<26} {'provider':<10} {'tier':<9} {'in $/M':>8} {'out $/M':>8} "
          f"{'context':>10}")
    for info in rows:
        print(f"{info.id:<26} {info.provider:<10} {info.tier:<9} "
              f"{info.input_cost:>8.2f} {info.output_cost:>8.2f} "
              f"{info.context_window:>10,}")
    return 0


async def _providers(args: argparse.Namespace) -> int:
    from .llm_providers import (
        describe_llm_provider,
        list_llm_providers,
        ping_llm_provider,
    )

    if args.name:
        spec = describe_llm_provider(args.name)
        result = await ping_llm_provider(args.name) if args.ping else None
        if args.json:
            print(json.dumps({**spec.model_dump(mode="json"),
                              **({"ping": result} if result else {})}, indent=2))
        else:
            print(spec.render())
            if result:
                state = "ok" if result["ok"] else f"failed — {result.get('error')}"
                print(f"  ping     {state} ({result.get('latency_ms', 0)} ms)")
        return 0 if result is None or result["ok"] else 1

    specs = list_llm_providers(configured_only=args.ready)
    if args.json:
        print(json.dumps([s.model_dump(mode="json") for s in specs], indent=2))
        return 0
    print(f"{'provider':<18} {'status':<12} needs")
    for spec in specs:
        needs = []
        groups: dict[str, list[str]] = {}
        for field in spec.required_fields:
            if field.one_of:
                groups.setdefault(field.one_of, []).append(field.name)
            else:
                needs.append(field.name)
        needs += [" | ".join(names) for names in groups.values()]
        status = "ready" if spec.configured else "needs setup"
        print(f"{spec.name:<18} {status:<12} {', '.join(needs) or '—'}")
    print("\nagent-harness providers <name> for every field; --ping to test the "
          "credentials.")
    return 0


async def _sessions(args: argparse.Namespace) -> int:
    harness = Harness.local(args.state or ".harness")
    rows = await harness.sessions.list(limit=args.limit)
    if args.show:
        session = await harness.sessions.load(args.show)
        for message in session.messages:
            print(f"{message.role}: {message.text[:2000]}")
        return 0
    if not rows:
        print("no sessions yet")
        return 0
    for session in rows:
        print(f"{session.id}  {session.agent:<16} {len(session.messages):>3} messages  "
              f"${session.usage.cost_usd:.4f}")
    return 0


async def _mcp(args: argparse.Namespace) -> int:
    from .mcp import MCPManager, MCPServer

    if args.url:
        server = MCPServer(name=args.server_name, url=args.url, transport="http")
    else:
        server = MCPServer(name=args.server_name, command=args.command[0],
                           args=args.command[1:])
    async with MCPManager([server]) as manager:
        if manager.errors:
            print("\n".join(manager.errors), file=sys.stderr)
            return 1
        for entry in manager.tools():
            print(f"{entry.name}\n    {entry.description}")
    return 0


def _journal(args: argparse.Namespace) -> int:
    from .runtime.journal import RunJournal

    path = Path(args.state or ".harness") / "journal.jsonl"
    if not path.exists():
        print(f"no journal at {path}", file=sys.stderr)
        return 1
    print(RunJournal.load(path).render(limit=args.limit))
    return 0


def _governance(args: argparse.Namespace) -> int:
    import os

    from .governance import Governance, HMACSigner, list_packs, load_pack
    from .runtime.audit import AuditTrail

    if args.action == "packs":
        rows = [load_pack(p) for p in list_packs()]
        if args.json:
            print(json.dumps([{"id": p.id, "name": p.name, "jurisdiction": p.jurisdiction,
                               "kind": p.kind, "binding": p.binding, "as_of": p.as_of}
                              for p in rows], indent=2))
            return 0
        for p in rows:
            kind = p.kind if p.binding else f"{p.kind}, voluntary"
            print(f"{p.id:<19} {p.jurisdiction:<8} {kind:<22} {p.name}")
        return 0

    if args.action == "pack":
        if not args.target:
            print("usage: agent-harness governance pack <id>", file=sys.stderr)
            return 2
        p = load_pack(args.target)
        print(f"{p.name}\n  {p.jurisdiction} · {p.kind} · "
              f"{'binding' if p.binding else 'voluntary'} · checked {p.as_of}")
        print(f"  status   {' '.join(p.status.split())}")
        print("\nrequirements")
        for r in p.requirements:
            where = "manual" if r.manual else ", ".join(r.controls)
            print(f"  {r.ref:<28} {r.title}  [{where}]")
        if p.policy.rules:
            print("\nrules")
            for rule in p.policy.rules:
                print(f"  {rule.id}: {rule.effect} on {', '.join(rule.on)} — {rule.reason}")
        if p.policy.residency and p.policy.residency.transfers:
            print("\nresidency")
            for origin, dest in p.policy.residency.transfers.items():
                print(f"  {origin} → {', '.join(dest)}")
        if p.incidents:
            print("\nincident clocks")
            for kind, clocks in p.incidents.items():
                for c in clocks:
                    print(f"  {kind}: {c.get('notify')} within {c.get('within')} h "
                          f"({c.get('basis', '')})")
        print("\nsources\n" + "\n".join(f"  {u}" for u in p.sources))
        return 0

    if args.action == "verify":
        path = Path(args.target or Path(args.state or ".harness") / "audit.jsonl")
        if not path.exists():
            print(f"no audit trail at {path}", file=sys.stderr)
            return 1
        signer = None
        if args.key_env:
            if not os.environ.get(args.key_env):
                print(f"${args.key_env} is not set", file=sys.stderr)
                return 1
            signer = HMACSigner(os.environ[args.key_env])
        trail = AuditTrail.load(path, signer=signer)
        ok, why = trail.verify()
        signed = "signed, signatures checked" if signer else "signatures not checked"
        print(f"{path}: {len(trail)} entries, {why} ({signed})")
        return 0 if ok else 1

    if args.action == "check":
        if not args.config:
            print("usage: agent-harness governance check --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)     # raises on anything invalid
        policy = gov.policy
        print(f"ok   {args.config}: policy {policy.name}@{policy.version} "
              f"sha256:{gov.engine.hash[:16]} · mode {gov.mode}")
        print(f"     packs: {', '.join(p.id for p in gov.packs) or 'none'}")
        print(f"     rules: {len(policy.rules)} · purposes: "
              f"{', '.join(sorted(policy.purposes)) or 'none'}")
        residency = policy.residency
        if residency and residency.transfers:
            print(f"     residency: home {residency.home or '(per principal)'}; "
                  + "; ".join(f"{o} → {', '.join(d)}"
                              for o, d in sorted(residency.transfers.items())))
        warnings = []
        if residency and residency.home is None and sum(
                "*" not in d for d in residency.transfers.values()) > 1:
            warnings.append("no `home:` — a person with no jurisdiction tag is unrestricted")
        if gov.signer is None:
            warnings.append("no signing key — the audit trail is hash-chained but unsigned")
        if gov.vault.path is None:
            warnings.append("no `vault:` path — the subject index is lost on restart, "
                            "so erasure cannot find earlier sessions")
        for warning in warnings:
            print(f"warn {warning}")
        return 0

    if args.action in ("inventory", "dsar"):
        if not args.config:
            print(f"usage: agent-harness governance {args.action} --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)

    if args.action == "inventory":
        if not args.agents:
            print("inventory needs --agents agents.yaml (a blueprint)", file=sys.stderr)
            return 2
        from .blueprint import Blueprint

        harness = Harness.testing(governance=gov)
        Blueprint.from_file(args.agents).build_all(harness=harness)
        if args.format == "json":
            print(json.dumps(gov.inventory.to_cyclonedx(), indent=2))
            return 0
        for entry in gov.inventory.agents.values():
            identity = entry["identity"]
            print(f"{entry['name']}  model={entry['model']}  provider={entry['provider']}"
                  f"  risk={identity.get('risk')}  owner={identity.get('owner') or '-'}")
            for tool_row in entry["tools"]:
                print(f"    {tool_row['name']:<24} {tool_row['fingerprint'][:12]}  "
                      f"{','.join(tool_row['tags'])}")
            if entry["mcp_servers"]:
                print(f"    mcp: {', '.join(entry['mcp_servers'])}")
        return 0

    if args.action == "dsar":
        if args.target not in ("access", "erase") or not args.user:
            print("usage: agent-harness governance dsar access|erase --user ID "
                  "[--tenant ID] --state DIR --config governance.yaml", file=sys.stderr)
            return 2
        from .memory.trace import Trace

        harness = Harness.local(args.state or ".harness", trace=False, governance=gov)
        trace = Trace(user_id=args.user, tenant_id=args.tenant)
        if args.target == "access":
            result = asyncio.run(gov.rights.access(trace))
        else:
            result = asyncio.run(gov.rights.erase(trace, requested_by=args.requested_by))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.action == "report":
        if not args.config:
            print("usage: agent-harness governance report --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)
        gov.attach(Harness.testing())
        report = gov.report(args.pack)
        print({"json": report.json, "html": report.html}.get(args.format,
                                                             report.markdown)())
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="Run and inspect agents built with agent-harness.",
    )
    parser.add_argument("--version", action="version", version=f"agent-harness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", default="assistant", help="what to call the agent")
        p.add_argument("--model", default=None, help="model id (routed by default)")
        p.add_argument("--provider", default=None,
                       help="any name from `agent-harness providers` "
                            "(inferred from the model)")
        p.add_argument("--instructions", default="", help="the agent's instructions")
        p.add_argument("--skills", default=None, help="a directory of skills to load")
        p.add_argument("--tools", action="store_true",
                       help="give it the built-in tools (time, maths, web fetch)")
        p.add_argument("--no-memory", action="store_true", help="run without memory")
        p.add_argument("--max-steps", type=int, default=20)
        p.add_argument("--state", default=None,
                       help="persist sessions, memory and traces under this directory")
        p.add_argument("--trace", action="store_true", help="print spans as they close")
        p.add_argument("--approve", action="store_true",
                       help="ask before any tool that needs approval")
        p.add_argument("--session", default=None, help="resume a session by id")

    run = sub.add_parser("run", help="run one task and print the answer")
    run.add_argument("task")
    run.add_argument("--stream", action="store_true", help="stream the answer")
    run.add_argument("--json", action="store_true", help="also print the full result")
    run.add_argument("--report", action="store_true", help="print the run report")
    common(run)

    chat = sub.add_parser("chat", help="an interactive session")
    common(chat)

    sub.add_parser("models", help="list the models the harness knows and their prices")

    providers = sub.add_parser(
        "providers", help="list the LLM providers and what each needs to connect")
    providers.add_argument("name", nargs="?", default=None,
                           help="describe one provider in full")
    providers.add_argument("--ping", action="store_true",
                           help="with a name: connect and check the credentials work")
    providers.add_argument("--ready", action="store_true",
                           help="only the providers already configured")
    providers.add_argument("--json", action="store_true", help="machine-readable output")

    sessions = sub.add_parser("sessions", help="list or show stored sessions")
    sessions.add_argument("--state", default=None)
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--show", default=None, help="print one session's transcript")

    mcp = sub.add_parser("mcp", help="list the tools an MCP server offers")
    mcp.add_argument("command", nargs="*", help="the server command and its arguments")
    mcp.add_argument("--url", default=None, help="an HTTP MCP endpoint instead")
    mcp.add_argument("--server-name", default="server")

    journal = sub.add_parser("journal", help="print the run journal")
    journal.add_argument("--state", default=None)
    journal.add_argument("--limit", type=int, default=50)

    gov = sub.add_parser("governance",
                         help="jurisdiction packs, evidence reports, audit verification")
    gov.add_argument("action",
                     choices=["packs", "pack", "check", "report", "inventory", "verify",
                              "dsar"],
                     help="packs: list them · pack <id>: one in full · check: validate a "
                          "config · report: evidence for a config · inventory: the AI "
                          "inventory of a blueprint · verify: check an audit trail · "
                          "dsar access|erase: a data-subject request")
    gov.add_argument("target", nargs="?", default=None,
                     help="the pack id (pack), an audit file (verify), or access|erase (dsar)")
    gov.add_argument("--agents", default=None, help="a blueprint agents.yaml (inventory)")
    gov.add_argument("--user", default=None, help="the data subject's user id (dsar)")
    gov.add_argument("--tenant", default=None, help="the data subject's tenant id (dsar)")
    gov.add_argument("--requested-by", default="data subject",
                     help="who asked for the erasure, for the record (dsar)")
    gov.add_argument("--config", default=None, help="a governance.yaml (report)")
    gov.add_argument("--pack", default=None, help="report on one pack only")
    gov.add_argument("--format", choices=["md", "json", "html"], default="md")
    gov.add_argument("--state", default=None, help="the harness directory (verify, dsar)")
    gov.add_argument("--key-env", default=None,
                     help="environment variable holding the HMAC audit key (verify)")
    gov.add_argument("--json", action="store_true", help="machine-readable (packs)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "models":
            return _models(args)
        if args.command == "providers":
            return asyncio.run(_providers(args))
        if args.command == "sessions":
            return asyncio.run(_sessions(args))
        if args.command == "mcp":
            return asyncio.run(_mcp(args))
        if args.command == "journal":
            return _journal(args)
        if args.command == "governance":
            return _governance(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
