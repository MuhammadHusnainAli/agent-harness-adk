# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-23

The first release. Published as `agent-harness-adk`, imported as `agent_harness`.

### Added

**Agents**
- `Agent` with the full loop: think → act → observe → repeat, parallel tool
  calls, token streaming, `run_sync` for scripts, and `as_tool()` so any agent
  can be called by another.
- Structured output via `output_type=<pydantic model>`, validated and retried
  when the model gets it wrong.
- `Harness` — the shared runtime services every agent in a run depends on, with
  `Harness.local()` for persistence and `Harness.testing()` for offline tests.

**Sub-agents**
- `SubAgentSpec` blueprints, a `Bench` of eight pre-defined sub-agents
  (research, planner, document_extractor, data_analyst, validator,
  compliance_checker, drafting, report_writer), and a `SubAgentFactory` that
  writes a new specialist during the run when nothing on the bench fits.
- Automatic `delegate` tool. Sub-agents start clean, take a hard tool allowlist,
  and delegation does not cascade by default.
- `Orchestrator` — plan (acceptance tests written first) → staff → run in
  dependency-ordered parallel waves → consolidate → independent review, with
  bounded rework rounds.

**Tools, skills, prompts, MCP**
- `@tool` decorator building JSON Schema from type hints and docstrings, with
  validation, context injection, permissions, tags and caching.
- `Skill` / `SkillRegistry` with progressive disclosure: only names and
  descriptions reach the prompt; bodies load through `load_skill`.
- `Prompt` / `PromptLibrary` — versioned templates with YAML frontmatter.
- MCP client over stdio and streamable HTTP: tools, resources and prompts, no
  vendor SDK.
- Built-in tools: `now`, `calculate`, `make_corpus_search`, `make_fetch_tool`,
  `make_http_tool`, plus workspace filesystem and (opt-in) shell tools.

**Memory**
- Four scopes with distinct loading rules: user (`user.md`, in full, rewritten
  at session close), session, orchestrator (digest only) and sub-agent (nothing
  carried in, resources carried out).
- Semantic recall with a pluggable embedder; the default hashing embedder works
  offline with no extra dependency.
- `InMemoryStore` and `FileStore` backends.

**Runtime rails**
- `PolicyGate` (allow / ask / deny), `BudgetGuard` (spend, tokens, steps, tool
  calls, sub-agents; child guards roll up), `HookEngine` (12 events),
  `Guardrails` (secret redaction, injection warnings, size caps), `Tracer`,
  `RunJournal`, `ResultCache`, `Checkpointer`, `SessionStore` (resume and fork),
  `WorkspaceBroker` (jailed local or Docker workspaces), `ConcurrencyScheduler`
  and `ModelRouter`.
- `ContextAssembler` and `ContextCompactor`; compaction evicts oversized tool
  results first and never orphans a `tool_use`/`tool_result` pair.

**Providers**
- Anthropic, OpenAI and Gemini adapters over raw HTTP — one retry, cost and
  tracing path for all three — plus `FakeProvider` for tests.
- Built-in price table with per-call cost accounting; `register_model` and
  `register_provider` to extend.

**Tooling**
- `agent-harness` CLI: `run`, `chat`, `models`, `sessions`, `journal`, `mcp`.
- Typed (`py.typed`), three runtime dependencies, ~100 ms import.
- 150 tests, green on Python 3.10 through 3.14.

[0.1.0]: https://github.com/MuhammadHusnainAli/agent-harness-adk/releases/tag/v0.1.0
