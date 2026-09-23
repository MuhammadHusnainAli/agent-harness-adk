# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.1] — 2026-09-23

Completes the architecture in `preview-01.png` — every box in the diagram now has
a working implementation, and `tests/test_harness_completeness.py` fails if one
goes missing.

### Added

**Agent**
- `runtime_agents` (`"enable"` / `"disable"` / bool) and `max_runtime_agents`
  (0-100) — an agent can write and run its own specialists mid-run through the
  factory, via a `spawn_agent` tool. Call it several times in one turn and they
  run in parallel. The budget is per run, does not cascade to spawned agents,
  is audited, counts against `Budget(max_subagents=...)`, and is refused once
  the run is stopped. `runtime_agent_tools` caps what any of them may be given.
- `compact_at` — where context compaction kicks in, as an absolute token count
  (`10_000`) or a fraction of the model's window (`0.5`). With
  `compact_keep_last`, `compact_target`, and `compactor=` to replace the
  strategy outright.
- `Orchestrator` takes `runtime_agents`, `max_runtime_agents` and `compact_at`
  and passes them to its manager.

**Memory knows whose it is, and where it lives**
- `Trace` — a user id, a session id, a tenant, or any combination. Pass it to an
  `Agent` or a `MemoryManager` and every record is stamped with it on write and
  filtered by it on read, so one store serves many users without them seeing
  each other. `scope` decides what documents like `user.md` are namespaced by:
  `"user"` (default), `"session"`, `"tenant"` or `"global"`. A record written
  without a trace stays shared rather than orphaned.
- `MemoryManager.for_trace(user_id)` — the same backend scoped to somebody else,
  reusing the store and its vector index rather than rebuilding them.
- New `agent_harness.memory.providers` package: thirteen interchangeable
  backends — `SQLiteMemory` (standard library), `PostgresMemory`, `MySQLMemory`,
  `MongoMemory`, `RedisMemory`, `DynamoDBMemory`, `ElasticsearchMemory`,
  `S3Memory`, `AzureBlobMemory`, `GCSMemory`, `HTTPMemory`, plus the existing
  in-process and file stores.
- `memory_provider("postgresql://...")` builds one from a connection string;
  `available_backends()` says which are ready; `register_backend()` adds yours.
- Every driver is imported on first use, so `import agent_harness` still touches
  none of them, and a driver you have not installed names its own `pip install`
  instead of raising `ImportError` mid-run.

**Guardrails you can put on one agent** — new `agent_harness.guardrails` package
- `AgentGuardrails` — what an agent may touch and what must be true before its
  answer is accepted. A forbidden tool is refused before it runs; everything
  else is checked at completion, and an unmet requirement sends the agent back
  round with a plain-English note about what is missing (`on_violation` is
  `"retry"`, `"fail"` or `"warn"`).
- Checks: `RequireTools`, `ForbidTools`, `MustInclude`, `MustNotInclude`,
  `MustMatch`, `MinLength`, `RequireCitation`, `RequireJSON`,
  `RequireStructured`, `NoPlaceholders`, `MaxSteps`, `MaxCost`, `Custom`.
- `SubAgentSpec.guardrails` carries them in serialisable form, so a sub-agent's
  requirements travel with its blueprint.
- `AgentGuardrails(content=Guardrails(...))` gives one agent stricter text rules
  than the harness default; `RunResult.violations` records what was unmet.

**Packages**
- `agent_harness.subagents` (spec, bench, factory, builder) and
  `agent_harness.guardrails` (rules, engine, checks, per-agent) are now packages
  rather than single modules. `agent_harness.subagent` and
  `agent_harness.runtime.guardrails` still re-export everything, so existing
  imports keep working.

**Assurance & control**
- `StopController` — abort a run and drain the sub-agents. The loop checks
  between steps and before every tool, so a stop lands at a safe boundary and
  finished work is kept. `harness.stop(reason)` is audited.
- `AuditTrail` — immutable, hash-chained who-did-what. Every permission
  decision, run start and run end is recorded; `verify()` names the first
  tampered or deleted entry. Secrets are redacted before anything is written.
- `ServiceHealth` — latency (p50/p95), failure rate and in-flight count per
  model, tool and MCP server, plus scheduler saturation.
- `Replayer`, `RecordingProvider`, `ReplayProvider` — record a run once, then
  reproduce it exactly with no network and no spend; or travel back to any step
  and continue from there with a changed prompt.
- `Evaluator`, `GoldenTask`, `Expect`, `EvalReport`, `Comparison`, `llm_judge` —
  golden tasks, weighted scoring and regression comparison against a saved
  baseline. Expectations cover text, tools called, structured output, step count
  and cost.

**Runtime**
- `RateGuard` / `RateLimit` — requests- and tokens-per-minute pacing that waits
  rather than failing.
- `DeliverableStore` — versioned, digested artefacts with a manifest; wired into
  `Agent.produce()`, sub-agent hand-backs and the orchestrator's output.
- `SpecCompiler` / `CompiledSpec` — a sub-agent blueprint resolved into the exact
  provider payload, with a cost estimate and an `explain()` you can read before
  spending anything.

**Tooling**
- `parse_document` — text, Markdown, CSV, TSV, JSON, JSONL, HTML and XML with no
  dependencies; PDF, DOCX and OCR through an optional install each, each naming
  the exact `pip install` when absent.
- `bar_chart`, `line_chart`, `markdown_table`, `render_report` — dependency-free
  inline SVG that works in light and dark, with a colour-vision-validated
  categorical palette, direct value labels and a legend.
- `make_python_tool` — sandboxed compute inside the workspace; asks for approval
  every time.

**Orchestrator**
- `task_timeout` — a deadline per sub-agent, with retry.
- Partial delivery is kept: a task that produced something before failing still
  contributes to consolidation, marked as partial, and dependent tasks are told
  their input is incomplete.

### Fixed

- A worker crashing with an unexpected exception left its task stuck in
  `running` while the job reported itself finished; it is now marked failed with
  the error.
- A failure during planning raised `NameError` from the artefact path instead of
  being reported as a failed run.
- A single *partial* task was presented as the finished answer; it now goes
  through consolidation so it is framed as what it is.
- The orchestrator's own artefacts (`plan.json`, `deliverable.md`) never reached
  the deliverable store.
- Bar charts: the value label on a negative bar collided with the category label
  when the bar reached the plot floor.
- The `delegate` tool's schema listed only the sub-agents present when it was
  first built, hiding every one attached afterwards.
- Subprocess transports (workspace `shell`, the sandboxed Python tool, and MCP
  stdio servers) are now released when the process finishes instead of being
  left to the garbage collector. On Python 3.10 the collector runs after the
  event loop has closed, so the destructor raised "Event loop is closed" from
  somewhere nothing could catch it. The test suite now fails on an unraisable
  exception rather than warning about it.
- `MCPClient.close()` did not catch `asyncio.TimeoutError` on Python 3.10,
  where it is a different class from the builtin `TimeoutError`, so a server
  that ignored `terminate()` was never killed.

### Security

- Jinja prompt rendering now goes through `select_autoescape` rather than
  leaving autoescaping off outright. Prompts still render as plain text — HTML
  escaping a prompt would corrupt it — but a template loaded from `.html` or
  `.xml` is escaped, and the intent is explicit rather than implied.
  (CodeQL `py/jinja2/autoescape-false`.)
- Every GitHub Action in the pipeline is pinned to a commit SHA instead of a
  moving tag, so a retagged upstream action cannot change what runs in the job
  that publishes to PyPI. (CodeQL `actions/unpinned-tag`.)

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

[Unreleased]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MuhammadHusnainAli/agent-harness-adk/releases/tag/v0.1.0
