#!/usr/bin/env python3
"""Generate `llms.txt` and `llms-full.txt` from the repository.

`llms.txt` is the index an AI assistant reads first: what this library is, and
where to find the rest. `llms-full.txt` is everything in one file, so a model
that can take the whole thing does not need to follow links at all.

Both are generated, never hand-edited — CI regenerates them and fails if the
committed copies have drifted:

    python scripts/build_llms_txt.py --check
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/MuhammadHusnainAli/agent-harness-adk/main"
REPO = "https://github.com/MuhammadHusnainAli/agent-harness-adk"

# The order the index presents things in: what a reader needs first, first.
DOCS: list[tuple[str, str, str]] = [
    ("README.md", "README",
     "The full API tour with runnable snippets: agents, sub-agents, skills, "
     "prompts, tools, MCP, memory and the runtime rails."),
    ("CHANGELOG.md", "Changelog", "What changed in each release."),
    ("CONTRIBUTING.md", "Contributing",
     "Development setup, the testing pattern, and the constraints this project "
     "holds to (three dependencies, ~100 ms import, Python 3.10 floor)."),
    ("SECURITY.md", "Security policy",
     "How to report a vulnerability, what counts as one for an agent library, "
     "and how to run agents safely."),
    ("SUPPORT.md", "Support", "Where questions, bugs and feature requests go."),
    ("RELEASING.md", "Releasing", "How a release is cut and published to PyPI."),
    ("CODE_OF_CONDUCT.md", "Code of conduct", "Contributor Covenant 2.1."),
]

EXAMPLES: list[tuple[str, str]] = [
    ("examples/01_quickstart.py", "An agent with one tool — the smallest useful thing."),
    ("examples/02_subagents.py",
     "A manager delegating to sub-agents in parallel, with per-sub-agent cost."),
    ("examples/03_skills_and_memory.py",
     "Skills loaded from disk, and memory that outlives the session."),
    ("examples/04_rails.py",
     "Permissions, budget, hooks, guardrails, tracing and the run journal."),
    ("examples/05_orchestrator.py",
     "The whole pipeline: plan, staff, run in parallel, consolidate, review."),
    ("examples/06_assurance.py",
     "Stop control, audit trail, health, replay and quality evaluation."),
    ("examples/07_governance.py",
     "Governance: EU and Saudi customers on one harness — residency, approvals, "
     "erasure and the evidence report."),
    ("examples/08_governance_saudi_government.py",
     "Governance for a Saudi government service: in-Kingdom residency, four-eyes "
     "approvals through the queue, SDAIA breach clock, DPIA draft."),
    ("examples/09_governance_singapore_fintech.py",
     "Governance under Singapore's agentic framework: monitor then enforce, prompt "
     "injection, inherited authority, tool drift."),
]


def project() -> dict[str, str]:
    """name, version and description from pyproject, without needing tomllib.

    `tomllib` is 3.11+, and this script has to run on the project's 3.10 floor.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    head = text.split("[project]", 1)[-1].split("\n[", 1)[0]
    out: dict[str, str] = {}
    for field in ("name", "version", "description"):
        match = re.search(rf'^{field}\s*=\s*"(.*)"$', head, re.MULTILINE)
        if not match:
            raise SystemExit(f"pyproject.toml has no [project] {field}")
        out[field] = match.group(1)
    return out


def read(path: str) -> str:
    file = ROOT / path
    return file.read_text(encoding="utf-8").rstrip() if file.exists() else ""


# --- API reference ------------------------------------------------------------

def _summary(obj: Any) -> str:
    """The first sentence of a docstring, on one line."""
    doc = inspect.getdoc(obj) or ""
    first = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    return re.sub(r"\s+", " ", first)


def _signature(obj: Any, limit: int = 220) -> str:
    try:
        text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    text = text.replace("'", "")
    return text if len(text) <= limit else text[:limit] + "...)"


def _inherited_names() -> set[str]:
    """Everything pydantic and object contribute — noise in an API reference."""
    from pydantic import BaseModel

    return set(dir(BaseModel)) | set(dir(object)) | {"model_config"}


_INHERITED = _inherited_names()


def _own_methods(cls: type) -> list[tuple[str, Any]]:
    """Methods this class actually defines, not the ones it inherits from a base."""
    out: list[tuple[str, Any]] = []
    for name, member in inspect.getmembers(cls, callable):
        if name.startswith("_") or name in _INHERITED:
            continue
        if not (inspect.isfunction(member) or inspect.ismethod(member)):
            continue
        out.append((name, member))
    return out


# Grouped the way the library is actually used, not alphabetically.
GROUPS: list[tuple[str, list[str]]] = [
    ("Core", ["Agent", "Harness", "Orchestrator", "Plan", "Task", "Review",
              "AgentVersion", "Blueprint", "AgentEntry"]),
    ("Sub-agents", ["SubAgentSpec", "Bench", "SubAgentFactory", "SpecCompiler",
                    "CompiledSpec"]),
    ("Tools and skills", ["tool", "Tool", "ToolRegistry", "ToolContext", "Skill",
                          "SkillRegistry"]),
    ("Guardrails", ["Guardrails", "AgentGuardrails", "CompletionContext",
                    "Violation", "Check", "RequireTools", "ForbidTools",
                    "MustInclude", "MustNotInclude", "MustMatch", "MinLength",
                    "MaxSteps", "MaxCost", "RequireCitation", "RequireJSON",
                    "RequireStructured", "NoPlaceholders", "Custom",
                    "NoPII", "NoSecrets", "NoInjection", "NotToxic",
                    "NoRepetition", "Grounded", "DetectorCheck",
                    "PIIDetector", "SecretDetector", "InjectionDetector",
                    "ToxicityDetector", "GroundednessDetector",
                    "RepetitionDetector", "Finding", "LLMGuard", "LLMVerdict",
                    "POLICIES"]),
    ("Prompts", ["Prompt", "PromptLibrary"]),
    ("Memory", ["MemoryManager", "Trace", "memory_provider", "available_backends",
                "MemoryStore", "MemoryRecord", "InMemoryStore",
                "FileStore", "UserMemory", "SessionMemory", "OrchestratorMemory",
                "SubAgentMemory", "SemanticMemory", "VectorStore", "Embedder",
                "HashEmbedder", "ProviderEmbedder"]),
    ("Providers", ["Provider", "AnthropicProvider", "OpenAIProvider",
                   "GeminiProvider", "FakeProvider", "BedrockProvider",
                   "VertexProvider", "VertexGeminiProvider",
                   "AzureOpenAIProvider", "AzureFoundryProvider", "GoogleAuth",
                   "CompletionRequest",
                   "ToolSchema", "ModelInfo", "get_provider", "register_provider",
                   "register_model", "tool_call"]),
    ("MCP", ["MCPServer", "MCPClient", "MCPManager"]),
    ("Runtime rails", ["Budget", "BudgetGuard", "RateLimit", "RateGuard",
                       "PolicyGate", "PermissionRule", "HookEngine", "HookContext",
                       "Guardrails", "StopController", "StopState", "AuditTrail",
                       "AuditEntry", "ServiceHealth", "ComponentHealth", "Tracer",
                       "Span", "RunJournal", "ResultCache", "ConcurrencyScheduler",
                       "ModelRouter", "RouteRule", "Session", "SessionStore",
                       "PermissionRule",
                       "InMemorySessionStore", "FileSessionStore", "Checkpoint",
                       "Checkpointer", "Replayer", "RecordingProvider",
                       "ReplayProvider", "DeliverableStore", "Workspace",
                       "WorkspaceBroker", "console_exporter", "jsonl_exporter"]),
    ("Evaluation", ["Evaluator", "GoldenTask", "Expect", "EvalReport",
                    "TaskOutcome", "Comparison", "llm_judge"]),
    ("Context", ["ContextAssembler", "ContextCompactor", "estimate_tokens"]),
    ("Types", ["Message", "ModelResponse", "RunResult", "StreamEvent", "Usage",
               "Artifact", "TextBlock", "ToolUseBlock", "ToolResultBlock"]),
    ("Errors", ["HarnessError", "ConfigurationError", "ProviderError",
                "RateLimitError", "ToolError", "ToolNotFound", "PermissionDenied",
                "BudgetExceeded", "GuardrailTripped", "MaxStepsExceeded",
                "OutputContractError", "StopRequested", "MCPError"]),
]


def api_reference() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    import agent_harness as ah

    lines = ["# API reference", ""]
    covered: set[str] = set()

    for heading, names in GROUPS:
        rows: list[str] = []
        for name in names:
            obj = getattr(ah, name, None)
            if obj is None:
                continue
            covered.add(name)
            kind = "class" if inspect.isclass(obj) else "function"
            signature = _signature(obj) if kind == "function" else ""
            rows.append(f"### `{name}{signature}`  ({kind})")
            summary = _summary(obj)
            if summary:
                rows.append(f"{summary}")
            if inspect.isclass(obj):
                members = [f"`{n}{_signature(m)}`" for n, m in _own_methods(obj)]
                if members:
                    rows.append(f"Methods: {', '.join(members)}")
                properties = [n for n, m in inspect.getmembers(
                    obj, lambda m: isinstance(m, property))
                    if not n.startswith("_") and n not in _INHERITED]
                if properties:
                    rows.append(f"Properties: {', '.join(properties)}")
                fields = getattr(obj, "model_fields", None)
                if fields:
                    rows.append(f"Fields: {', '.join(list(fields)[:24])}")
            rows.append("")
        if rows:
            lines.append(f"## {heading}")
            lines.append("")
            lines.extend(rows)

    missing = [n for n in ah.__all__ if n not in covered and n != "__version__"]
    if missing:
        lines.append("## Also exported")
        lines.append("")
        lines.append(", ".join(f"`{n}`" for n in missing))
        lines.append("")
    return "\n".join(lines).rstrip()


# --- the two files ------------------------------------------------------------

def build_index() -> str:
    meta = project()
    out = [
        f"# {meta['name']}",
        "",
        f"> {meta['description']}",
        "",
        "Installed as `agent-harness-adk`, imported as `agent_harness`. Python "
        "3.10-3.14. Three runtime dependencies (pydantic, httpx, pyyaml), no vendor "
        "SDKs: the Anthropic, OpenAI and Gemini adapters speak HTTP directly, so all "
        "three share one retry, cost-accounting and tracing path.",
        "",
        "```python",
        "from agent_harness import Agent, tool",
        "",
        "@tool",
        "def order_status(order_id: str) -> str:",
        '    """Look up a customer order.',
        "",
        "    Args:",
        "        order_id: the order number.",
        '    """',
        "    return db.lookup(order_id)",
        "",
        'agent = Agent("support", "Answer order questions.", tools=[order_status])',
        'print(agent.run_sync("Where is order 4182?").output)',
        "```",
        "",
        "What it covers: the agent loop (think, act, observe), sub-agents from a "
        "reusable bench or written at run time by a factory, skills with progressive "
        "disclosure, versioned prompts, four memory scopes, MCP over stdio and HTTP, "
        "and the runtime rails underneath all of it - permissions, budgets, rate "
        "limits, hooks, guardrails, tracing, an immutable audit trail, service "
        "health, checkpoints, deterministic replay, isolated workspaces and quality "
        "evaluation.",
        "",
        "## Docs",
        "",
    ]
    for path, title, description in DOCS:
        if (ROOT / path).exists():
            out.append(f"- [{title}]({RAW}/{path}): {description}")

    out += ["", "## Examples", "",
            "Every example runs without an API key - they fall back to a scripted "
            "`FakeProvider`.", ""]
    for path, description in EXAMPLES:
        if (ROOT / path).exists():
            out.append(f"- [{Path(path).stem}]({RAW}/{path}): {description}")

    out += [
        "", "## Optional", "",
        f"- [Full documentation in one file]({RAW}/llms-full.txt): this index, every "
        "document above, the complete API reference and every example, concatenated.",
        f"- [PyPI](https://pypi.org/project/{meta['name']}/): released versions.",
        f"- [Source]({REPO}): the repository.",
        f"- [Issues]({REPO}/issues): bugs and feature requests.",
        "",
    ]
    return "\n".join(out)


def build_full() -> str:
    meta = project()
    parts = [
        f"# {meta['name']} - complete documentation",
        "",
        f"> {meta['description']}",
        "",
        f"Version {meta['version']}. Generated by `scripts/build_llms_txt.py` - "
        "do not edit by hand.",
        "",
        "This file contains: the index, every document in the repository, the full "
        "API reference, and the source of every example.",
        "",
        "---",
        "",
    ]
    for path, title, _ in DOCS:
        body = read(path)
        if body:
            parts += [f"# {title} ({path})", "", body, "", "---", ""]

    parts += [api_reference(), "", "---", "", "# Examples", ""]
    for path, description in EXAMPLES:
        body = read(path)
        if body:
            parts += [f"## {path}", "", description, "", "```python", body,
                      "```", ""]
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed files are out of date")
    args = parser.parse_args()

    files = {ROOT / "llms.txt": build_index(), ROOT / "llms-full.txt": build_full()}

    if args.check:
        stale = [p.name for p, text in files.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print(f"out of date: {', '.join(stale)}\n"
                  "run `python scripts/build_llms_txt.py` and commit the result",
                  file=sys.stderr)
            return 1
        print("llms.txt and llms-full.txt are up to date")
        return 0

    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.name} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
