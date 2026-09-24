# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Governance — `agent_harness.governance`**
- `Governance`, attached with `Harness(governance=...)`: policy, agent identity,
  data classification, residency, human oversight, transparency, signed records,
  data-subject rights, risk classification, AI inventory, runtime monitoring,
  incidents and evidence reports — enforced through the loop's own hook points.
- 26 jurisdiction and framework packs, each dated and sourced: EU AI Act, GDPR,
  DORA, NIS2, UK GDPR; UAE PDPL, DIFC Regulation 10, ADGM, Saudi PDPL, SDAIA,
  Qatar, Bahrain, Oman; Singapore (agentic framework and PDPA), India DPDP,
  China, Korea AI Basic Act, Japan APPI, Vietnam AI law; NIST AI RMF, Colorado,
  Texas TRAIGA, CCPA ADMT; ISO/IEC 42001 and the OWASP Top 10 for Agentic
  Applications.
- Policy as code: YAML rules with a safe expression language, strictest-effect
  combination, fail-closed evaluation, and a sha256 on every decision.
- Residency per model call and per data subject, with rerouting down the model
  chain; purpose-based redaction and reversible pseudonymisation.
- National-ID detection validated by check digit: Emirates ID, Saudi ID/Iqama,
  Aadhaar, Singapore NRIC/FIN, Chinese resident ID, Korean RRN, UK NINO.
- Approvals with quorum, separation of duties and fail-closed timeouts.
- `AuditTrail(signer=...)`: HMAC (stdlib) or Ed25519 signatures; erasure by
  crypto-shredding keeps the chain verifiable.
- Erasure covers memory, sessions, deliverables, run-journal entries and trace
  spans; the audit trail holds task text and (optionally, `audit_args="tokens"`)
  tool arguments only as per-person tokens.
- A breaker for tools that keep failing, one incident per cause, approval
  notifications (`approval_notify=`), and fail-closed handling of governance's
  own errors.
- `agent-harness governance packs | pack | check | report | inventory | verify | dsar`.
- Optional extra `governance` (`cryptography`) for Ed25519 signing.
- `examples/07_governance.py`, `08_governance_saudi_government.py`,
  `09_governance_singapore_fintech.py`.

### Changed

- New hook event `model_egress`, fired per backend actually tried, with the
  provider and model; blocking it moves on to the next model in the chain.
- `subagent_start` and `run_start` hooks can now block; `memory_write` is now
  emitted (it was declared but never fired); `pre_tool` carries the tool's
  `tags` and declared `permission`.
- `Agent(identity=...)`, and `identity:` in blueprints and `SubAgentSpec`.
- `AuditTrail(signer=..., scrub=...)`; `RunJournal.forget`, `Tracer.forget` and
  `DeliverableStore.forget_runs` remove a person's runs.
- The audit trail records `run_start` after the hook, and scrubs before it trims.

## [0.1.3] — 2026-09-24

### Changed — breaking

- **`agent_harness.providers` is now `agent_harness.llm_providers`.** Imports
  from the package root (`from agent_harness import AnthropicProvider`) are
  unchanged; only code importing the submodule directly needs the new path.
- A missing key raises `AuthenticationError` (still a `ProviderError`).
- `max_retries` now defaults to 3, from 2.

### Added

**Know what a provider needs before connecting**
- `list_llm_providers()` — every supported provider as a `ProviderSpec`: its
  fields (required, secret, either-or, and the environment variables each is
  read from), capabilities, known models, and whether it is configured now.
  Filter with `configured_only=True` or `capability="embeddings"`.
- `describe_llm_provider(name)` — one provider in full, with `.render()` and a
  ready-made `.example()` line.
- `check_llm_provider(name, **settings)` — an offline `ProviderCheck`: what is
  missing, where each setting was found (never its value), unknown arguments.
- `ping_llm_provider(name)` and `Provider.ping()` — a live credentials check
  that returns a result instead of raising. `Provider.list_models()` for
  Anthropic, OpenAI, Gemini and every OpenAI-compatible preset.
- `agent-harness providers [name] [--ping] [--ready] [--json]`.
- `get_provider` with a misspelt setting names the settings it accepts.

**Twelve more providers**
- OpenAI-compatible presets: `OpenRouterProvider`, `GroqProvider`,
  `TogetherProvider`, `DeepSeekProvider`, `MistralProvider`, `XAIProvider`,
  `FireworksProvider`, `CerebrasProvider`, `OllamaProvider`, `LMStudioProvider`,
  `VLLMProvider`, and `OpenAICompatibleProvider(base_url=...)` for anything else.
  `deepseek-*`, `grok-*` and Mistral model ids route to their vendor.
- Bedrock API keys (`AWS_BEARER_TOKEN_BEDROCK`) as an alternative to SigV4.
- OpenAI `organization` / `project`; reasoning returned by DeepSeek, vLLM and
  OpenRouter is kept as thinking.

**Generation parameters, configured properly everywhere**
- `min_p`, `frequency_penalty`, `presence_penalty` and `repetition_penalty` join
  `temperature`, `top_p`, `top_k`, `seed`, `effort`, `thinking` and
  `thinking_budget` as first-class settings on `Agent`, `AgentVersion`,
  `SubAgentSpec`, blueprints and `CompiledSpec`.
- Values are range-checked when the agent is built (`validate_parameters`);
  `CompletionRequest` enforces the same ranges.
- `effort` gains `none` and `minimal`, and is mapped per model: Claude's
  `output_config.effort`, OpenAI's `reasoning_effort`, Gemini 2.5 budgets fitted
  to each model's range, Gemini 3 `thinkingLevel`, OpenRouter's `reasoning`
  object, and `reasoning_effort` for gpt-oss and grok-3-mini on every host.
- Each provider declares the sampling `parameters` it accepts (shown in the
  catalog). A `ParameterPlan` records what is sent, dropped and adjusted, and
  why; `provider.explain(request)` shows it with the exact payload, and each
  decision is logged once.

**Resilience, for every provider alike**
- `RetryPolicy`: back-off, jitter, a retry count, a wall-clock ceiling, and
  `max_retry_after` — a server asking for a longer wait fails the call at once
  so a fallback model takes over.
- Retries on 408/409/425/429/5xx/529, timeouts and dropped connections. Every
  hint is honoured: `Retry-After` (seconds or HTTP date), `retry-after-ms`,
  `x-ratelimit-reset-*`, Gemini's `RetryInfo`, and `x-should-retry`.
- A shared cool-down: a rate limit on one call holds back its siblings.
- `CircuitBreaker` (on by default: 5 failures, 30 s), `max_concurrency`,
  `connect_timeout`, `on_retry` callbacks with a `RetryEvent`, and
  `provider.stats` / `provider.health()`.
- Typed errors: `QuotaExceededError`, `AuthenticationError`,
  `InvalidRequestError`, `ModelNotFoundError`, `ContextWindowExceededError`,
  `ProviderTimeoutError`, `ProviderConnectionError`, `ProviderUnavailableError`.
  Each carries `retryable`, `retry_after`, `attempts` and the vendor `request_id`.
  Model fallback now follows `retryable`, and moves on from an exhausted quota.
- Streams are restarted if they fail before the first token.
- A 401 refreshes Vertex and Entra ID tokens, and chain-sourced AWS
  credentials, once.

### Fixed

- Claude requests the API rejects are no longer sent: `temperature`/`top_k`
  with extended thinking, `top_p` below 0.95 with thinking, `temperature` above
  1, both `temperature` and `top_p`, a thinking budget at or above
  `max_tokens` or below 1024, thinking with a forced `tool_choice`.
- Thinking-only Claude models were not recognised under Bedrock/Vertex ids
  (`us.anthropic.claude-opus-5`), so they were sent budgets and sampling.
- OpenAI reasoning models were sent `stop`; Grok reasoning models were sent
  penalties and `stop`.
- Gemini 2.5 Pro was asked to switch thinking off; budgets outside a model's
  range were sent unchanged; Gemini 3 was sent a budget instead of a level;
  Gemini 2.0 was sent a thinking config.
- Streaming never retried — not even a 429 on connect.
- Vertex never retried anything.
- Bedrock and Vertex streaming posted to `api.anthropic.com`'s path on the
  wrong host; Azure streaming ignored the deployment URL and Entra ID. All
  three now stream properly — Bedrock by decoding its binary event stream.
- Bedrock retried with the signature from the first attempt, did not retry
  network errors, and did not escape `:` and `/` in ARN model ids.
- `CompletionRequest.timeout` was accepted and ignored.
- Streamed Anthropic thinking lost its signature, so the next turn of a tool
  loop with thinking on was rejected; redacted thinking was dropped; streamed
  output tokens were double-counted.
- Thinking from another vendor was sent to Anthropic unsigned (a 400).
- Gemini thought signatures were dropped; images given by URL were dropped.
- OpenAI-compatible servers that omit tool-call ids, or send arguments already
  parsed, or say `stop` after a tool call, are handled.
- An Azure/Vertex token lock created at construction could fail when a cached
  provider was reused under a new event loop.

## [0.1.2] — 2026-09-23

### Added

**Budgets that stop rather than fail**
- `Budget(max_input_tokens=..., max_output_tokens=...)` — per-axis token
  ceilings, so a sub-agent can be given "10k in, 2k out" and held to it.
- `on_exceed` decides what reaching one means. The default, `"stop"`, ends the
  run cleanly: the work the agent had done is kept, a line saying the budget is
  exceeded is appended, and the parent receives an answer rather than an
  exception. `"raise"` restores the old behaviour.
- `BudgetGuard.remaining()` and `.would_exceed()` report what is left and what
  the next call would cross.

**Model fallback**
- `ModelRouter(fallbacks=["claude-sonnet-5", "gpt-4.1"])`. When a model cannot
  be reached the loop moves to the next one, resolving its provider as it goes.
  A 4xx is not retried — the request is wrong and the next model will reject it
  too. Every switch is journalled and audited.

**Versions**
- `Agent(version="v2", versions={...})` — several configurations of one agent,
  each with its own instructions, model, tools, sub-agents, guardrails and
  budget. `agent.use("v1")` returns that version, `run(task, version="v1")`
  runs it, and both share the harness so they are directly comparable — run the
  same golden tasks against each and see what changed.

**Blueprints**
- `Blueprint.from_file("agents.yaml")` — declare prompts, sub-agents, agents,
  guardrail sets, versions and a memory backend in YAML or JSON, then
  `build("support", tools=[...])`. Tools stay in code; everything else is
  declaration. Tools may also be named as import paths.

**Cloud platforms**
- `BedrockProvider` — Claude on AWS Bedrock, with SigV4 signing implemented from
  the standard library and checked against AWS's published test vectors
  (signing-key derivation and the `get-vanilla` signature both match exactly).
  Credentials come from arguments, `AWS_*`, or the botocore chain if boto3 is
  installed.
- `VertexProvider` and `VertexGeminiProvider` — Claude and Gemini on Google
  Vertex AI, authenticating through `google-auth`, `gcloud`, or a token you pass.
- `AzureOpenAIProvider` and `AzureFoundryProvider` — deployments, api-version
  routing, `api-key` or Entra ID via `credential=`, with the Foundry route
  overridable because those routes move.
- Platform model ids are normalised, so `anthropic.claude-opus-5`,
  `us.anthropic.claude-opus-5` and `claude-opus-5@20260401` resolve to the same
  model and are priced identically — cost attribution survives the move to a
  cloud platform.

**Every connection parameter**
- `CompletionRequest` now carries `effort` (low/medium/high/xhigh/max),
  `thinking_budget`, `top_k`, `seed`, `frequency_penalty`, `presence_penalty`,
  `parallel_tool_calls`, `cache`, `speed`, `user`, `metadata`,
  `response_mime_type`, `safety_settings` and `timeout`, alongside what was
  there before. `Agent` takes the common ones directly and `model_options={}`
  for the rest.
- Each adapter maps what its provider supports and drops what it does not,
  rather than inventing an equivalent. Sampling is withheld from thinking-only
  models that reject it, `effort` maps onto OpenAI's three levels, and becomes
  a token budget on Gemini.

**Guardrails at industry scale**
- Algorithmic detectors that are exact where they can be: `PIIDetector`
  (Luhn-checked cards, mod-97 IBANs, precedence-ordered so a card is not also
  reported as a phone number), `SecretDetector` (known formats plus Shannon
  entropy for keys nobody has seen), `InjectionDetector` (weighted signals
  scored 0-1 rather than a single regex hit), `ToxicityDetector`,
  `GroundednessDetector` and `RepetitionDetector`.
- Ready-made checks: `NoPII`, `NoSecrets`, `NoInjection`, `NotToxic`,
  `NoRepetition`, `Grounded`, and `DetectorCheck` to wrap your own.
- `LLMGuard` — a model judging against a policy, with a structured verdict, a
  severity threshold, per-content caching, and an `on_error` that decides
  whether a broken judge blocks or lets work through. Seven ready policies.
- `AgentGuardrails.check_async()` runs the cheap deterministic checks first and
  only pays for a judge if they are all happy.

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

[Unreleased]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MuhammadHusnainAli/agent-harness-adk/releases/tag/v0.1.0
