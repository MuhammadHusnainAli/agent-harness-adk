# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Approvals that outlive the process — `Harness(approvals=...)`, `agent.resume_approval`**
- With an approval store on the harness, a tool call that needs a person and
  has no approver to ask no longer ends in a refusal: the run stops at that
  step with `stop_reason == "approval"` and `result.approval`, and is stored —
  the conversation, the waiting call with its exact arguments, and what the
  other tools of the step returned. An `approval_required` stream event.
- `harness.approvals`: `pending()`, `list()`, `get()`, `approve(id, by=, note=,
  call=)`, `deny(...)`, `expire()`, `release()`. `Approvals(store, notify=,
  expires=, self_approval=, claim_timeout=)`.
- `agent.resume_approval(id)` and `stream_approval(id)` carry the run on from
  the step it stopped in, in any process: the approved call runs with the
  approved arguments and the model is not asked again; a declined or expired
  one is answered with the reason; tools that had already run are not re-run.
  The session, its sandbox, a mode's todo list and sources, and the usage so
  far are carried over. A run may pause again.
- A record is resumed once: the claim is a write the store refuses to a second
  taker, across processes. A resume that fails is marked `failed` and is not
  retried by itself. Arguments that changed since approval are refused.
- Requests are owned by the run's user and tenant. If the conversation was
  continued while a request waited, the resumed run is kept as a fork.
- Stores: `SessionApprovalStore` (any session database — `approvals=True`),
  `FileApprovalStore` (`Harness.local()`), `MemoryApprovalStore`;
  `ApprovalStore` to write one. `Harness.on(url)` keeps approvals in the same
  database. `ApprovalError`.
- `PolicyGate.needs()`; `PolicyGate.check(approved_by=)`.
- **Changed:** on `Harness.local()` and `Harness.on()`, a tool that asks when
  no approver is configured now pauses the run instead of being refused. A
  plain `Harness()` and any harness with an `approver` behave as before.
- Not covered: sub-agents, workflow tool steps, voice turns and handoff chains
  do not pause; governance `require_approval` still waits in-process.
- `agent-harness approvals [list|show|approve|deny|resume|release] [ID]`, and
  `run` prints how to approve a run that stopped to ask.
- `examples/23_approvals.py`.

**Image generation — `ImageGenerator`**
- `ImageGenerator()` makes images with the engine whose key is set: Gemini
  (`gemini-2.5-flash-image`, `gemini-3-pro-image-preview`, Imagen), OpenAI
  (`gpt-image-1`, or any server with the same API) and Replicate (any hosted
  model). All over HTTP; no dependency is added. A list of engines is a
  fallback chain.
- Reference images: a path, a URL, bytes, an `ImageBlock`, or an image made
  earlier, sent with the prompt to edit, restyle or keep a likeness. An agent
  names a file in its workspace, an earlier image, an attachment
  (`attachment:1`) or an http(s) address — never a path on the machine. Each is
  verified to be an image from its bytes; URLs are fetched from public
  addresses only, with a size cap.
- Every generation is an `ImageJob`: `submit` returns it at once, with
  `status`, `fraction`, `describe()`, `wait()`, `cancel()` and `result()`;
  `generate` waits for it. `on_progress` receives an `ImageProgress` at every
  step. Replicate predictions are polled and report the model's own progress.
- `images.tools()`: `generate_image(prompt, references, aspect_ratio, count,
  name)` and `image_status(job_id, wait, cancel)`. The image is saved to the
  workspace under `images/`, added to `result.artifacts`, and shown to the
  model. A job that outlives `wait` is collected with `image_status`.
- Retries with back-off on 429, 5xx, timeouts and an answer with no image; the
  next engine on a refused key, an exhausted quota or unsupported references; a
  circuit breaker per engine. A prompt the provider refuses is not retried and
  not taken to another provider. `timeout`, `deadline`, `max_count`,
  `max_images`, `max_concurrency`; a cancelled job is cancelled at the provider;
  what comes back is checked to be an image; `cost_per_image` is charged to the
  run's budget; keys are redacted from errors.
- `FakeImages` for tests; `ImageEngine` to subclass, or a function, for a
  service of your own; `image_engines()`; `AGENT_HARNESS_IMAGES`.
- A tool's context now carries what came with the task:
  `ctx.state["attachments"]`.
- `agent-harness image [PROMPT] [--engine] [--model] [--reference] [--aspect]
  [--count] [--name] [--out]`, and `--images` on `run` and `chat`.
- `examples/22_images.py`.

**Browser and computer use — `Browser`, `computer_tool`**
- `Browser()` starts Chrome, Chromium or Edge headless and drives it over the
  DevTools protocol with the harness's own WebSocket client: no dependency is
  added. `cdp_url=` drives a browser that is already running; `headless=False`
  shows it; `user_data_dir=` keeps a profile.
- `browser.tools()`: `browser_navigate`, `browser_click`, `browser_type`,
  `browser_select`, `browser_press`, `browser_scroll`, `browser_back`,
  `browser_snapshot`, `browser_read`, `browser_wait`, `browser_tabs`,
  `browser_screenshot`. After each action the model reads the page as numbered
  elements — role, name, value, state — with what is on screen first, and acts
  by number, so a model without vision can browse. `vision=True` adds a
  screenshot to every answer; `permission="ask"` has each action approved.
- `allowed_domains`, `blocked_domains` and `allow_private` are applied to every
  document a tab loads: typed, clicked, redirected or framed. Only `http` and
  `https` open; downloads are refused.
- New tabs are followed, dialogs are answered and reported, covered elements
  are clicked directly, shadow roots are read, password input is not echoed. A
  browser that died is started again and a crashed tab reloaded; the browser
  never outlives its process.
- `computer_tool(computer)`: one `computer` tool — `screenshot`, `click`,
  `double_click`, `right_click`, `move`, `drag`, `type`, `key`, `scroll`,
  `wait`, `open` — answering every action with a screenshot. It is an ordinary
  tool, so it works with any model that sees. `browser.computer()` is a browser
  page; `DesktopComputer(sandbox)` is an X11 desktop driven with `xdotool`, and
  asks before each action when the desktop is this machine's own; `Computer` is
  the class to subclass for any other screen.
- A tool may return an image — `return [text, ImageBlock.from_bytes(png)]` —
  and the model is shown it, on every provider that takes images. A
  conversation keeps the newest three (`agent.tool_images_kept`).
  `ToolOutcome.media`; `media` on the `tool_result` stream event.
- `agent-harness run|chat … --browser` (and `--show-browser`).
- `examples/21_browser.py`.

**Web search — `web_search`, `WebSearch`**
- `web_search` in `agent_harness.toolkits`: a search tool that works with any
  model. It uses the engine whose key is in the environment — Tavily, Brave,
  Exa, Serper, Google Programmable Search, or a SearXNG instance — and
  DuckDuckGo, which needs no key, when none is set. Every engine is spoken to
  over HTTP; no dependency is added.
- The model passes `query`, `limit`, `recency` (`day`, `week`, `month`, `year`)
  and `domains`, and reads back title, URL, snippet and publication date.
- `WebSearch(engine, ...)` to choose: one engine or a list to fall back through,
  `allowed_domains`, `blocked_domains`, `limit`, `max_limit`, `region`,
  `language`, `safe_search`, `snippet_chars`. `search.search(...)` to call it
  yourself, `search.as_tool()`, `search.stats`, `search.last_engine`;
  `make_search_tool(...)` for the tool in one call.
- Retries with back-off on 429, 5xx, timeouts and dropped connections,
  honouring `Retry-After`; a refused key, an exhausted quota or a rejected query
  is not retried. An engine that fails, or finds nothing, is passed over for the
  next; one that keeps failing is left alone for a minute. `timeout` bounds a
  request and `deadline` the whole search.
- Results are stripped of markup and tracking parameters, de-duplicated, and
  held to the domain policy whatever the engine returned. Answers are cached for
  `cache_ttl` seconds and identical concurrent questions are one request.
- A failed search is a tool error naming each engine and why, with keys
  redacted.
- An engine of your own: a function of a `SearchQuery`, or a `SearchEngine`
  subclass. `search_engines()` lists what ships and what is set up;
  `AGENT_HARNESS_SEARCH` names the engines for a search that was given none.
- `agent-harness search [QUERY] [--engine NAME] [--limit N] [--recency R]
  [--domain D] [--json]`, and `--tools` now includes `web_search`.
- `examples/20_web_search.py`.

**Agents as an MCP server — `MCPAgentServer`**
- `MCPAgentServer(agents)` serves agents as MCP tools: each agent is one tool
  taking a `task`. An ASGI application over MCP's streamable HTTP, a built-in
  server for development (`await server.serve()`), and stdio
  (`await server.serve_stdio()`).
- A key per agent: `api_keys={"billing": "sk-…", "orders": "sk-…"}`. A key opens
  only the agents it was issued for, and the others are not listed or callable.
  `api_key=` opens every agent; several keys per agent; `None` leaves an agent
  open; a key can name its `user_id` and `tenant_id`; `auth=` for a check of
  your own; `add_key` and `revoke_key` while serving. Keys are accepted as
  `Authorization: Bearer` or `X-API-Key`, compared in constant time, and logged
  only by id. Each agent is also served alone at `/<name>/mcp`.
- `conversation_id` carries a conversation across calls, owned by the key.
  `expose_tools=` serves an agent's own tools through its rails. Output
  contracts come back as `structuredContent`, artefacts as embedded resources,
  failures as `isError`.
- Progress notifications for a caller that sends a progress token; calls end at
  `timeout`, on `notifications/cancelled`, or when the caller disconnects;
  `max_concurrency`, `max_queue`, `allowed_origins`, `/healthz`.
- `mcp_key_env` on a blueprint agent and `Blueprint.mcp_server()`.
- `agent-harness mcp-serve [--blueprint FILE] [--api-key K] [--key-env AGENT=VAR]
  [--expose-tools] [--stdio]`.
- `examples/19_mcp_server.py`.

**OpenAPI → tools — `openapi_tools`, `OpenAPIToolkit`**
- `openapi_tools(spec)` turns every operation of an OpenAPI document into a tool.
  The document is a file (JSON or YAML), a URL, its text, or a mapping; OpenAPI
  3.0, 3.1 and Swagger 2.0.
- Arguments come from the document: path, query, header and cookie parameters,
  and the fields of a JSON or form body; `$ref` followed, `allOf` merged,
  `readOnly` fields left out, recursive schemas ended. Query `style`/`explode`,
  `deepObject`, JSON-encoded parameters, form, multipart and text bodies.
- `token=` (a string or a function), `api_key=`, `basic=`, `credentials=` by
  security-scheme name, `headers=`, `params=`. Credentials are added to the
  request and never appear in a tool's schema. Redirects are not followed.
- `include`, `exclude`, `tags`, `methods`, `deprecated`, `prefix`, `base_url`,
  `writes="allow"|"ask"|"deny"`, `cache_reads`, `retries`, `timeout`.
- `toolkit.call(name, **args)`, `.describe()`, `.names`, `.skipped`, `.aclose()`.
  An HTTP error is a tool error the model reads.
- `openapi:` in blueprints and workflow files (`spec`, `token_env`,
  `api_key_env`, …); `Blueprint.api_toolkits()`.
- `agent-harness openapi SPEC [--call TOOL --arg name=value] [--json]`, and
  `--openapi SPEC` on `run`, `chat`, `voice` and `a2a serve`.
- `examples/18_openapi.py`.

**Agent-to-agent protocol — `agent_harness.a2a`**
- `A2AServer(agent)` serves an agent over the JSON-RPC binding of A2A 0.3: the
  agent card, `message/send`, `message/stream` (SSE), `tasks/get`,
  `tasks/cancel`, `tasks/resubscribe` and the push-notification methods. It is
  an ASGI application (`uvicorn myapp:server`), with a small server of its own
  for development (`await server.serve()`); a mapping serves several agents.
- Built for many replicas: tasks and conversations are kept in the harness's
  session store (`TaskStore`, `SessionTaskStore`, `MemoryTaskStore`); a message
  retried with the same `messageId` is one task; `max_concurrency` and
  `max_queue` answer `429` with `Retry-After` when full; `task_timeout`; a task
  whose worker was lost is reported failed; cancel and re-subscribe work from
  any replica; `/healthz`; drain on shutdown.
- `auth=`: a token, several, token → identity, or a function of the headers.
  Tasks and conversations belong to their caller, and each conversation runs on
  its own instance of the agent.
- Push notifications to https webhooks, off unless `push_notifications=` allows.
- `A2AClient`: `card`, `send`, `stream`, `get`, `cancel`, `wait`, `resubscribe`,
  with retries that keep the message id.
- `RemoteAgent` — an A2A agent as a sub-agent, a tool (`as_tool`), or a workflow
  step; `a2a:` on an agent in a blueprint or workflow file. A remote failure is
  `result.error`. Registers with governance when given a harness.
- `agent-harness a2a serve | card URL | send URL TEXT [--stream]`.
- `examples/17_a2a.py`.

**Declared workflows — `agent_harness.Workflow`**
- A workflow written in YAML or JSON and run as written:
  `Workflow.from_file(path, agents=, tools=, harness=)`, `.run(inputs)`,
  `.stream(inputs)`, `.describe()`, `.as_tool()`.
- Step kinds: `agent`, `tool`, `set`, `steps` (sequence), `parallel`, `foreach`
  (with `as`, `index`, `concurrency`), `loop` (`max`, `until`, `while`),
  `if`/`then`/`else`, `switch`, `graph` (nodes with `needs`, `join: all|any`),
  `wait`, `fail`, `return`.
- On any step: `when`, `save`, `retry`, `timeout`, `on_error: continue`.
- Shared `state`, declared `inputs` (required, default, type), and `{{ }}`
  templates over `inputs`, `state`, `steps.<id>` and `previous`, compiled with
  the policy expression language — never `eval`. A template that is one
  expression keeps its type.
- The file is validated when loaded: unknown keys, duplicate ids, templates that
  name a step or input that does not exist, missing agents and tools, cyclic
  graphs.
- Agents may be passed in, declared in the file (`agents:`, as in a blueprint),
  or declared inline on a step. `workflows:` in a blueprint and
  `Blueprint.workflow(name)`.
- `WorkflowResult` (`output`, `state`, `steps`, `status`, `error`,
  `failed_step`, `usage`, `cost_usd`), `StepResult`, `WorkflowEvent`,
  `WorkflowError`.
- Tool steps pass the permission gate, hooks, guardrails and audit trail. A
  stop is honoured before every step; `budget:`, `timeout:` and `max_steps:`
  bound a run; `workflow_start`, `workflow_step` and `workflow_end` hook events.
- `agent-harness workflow FILE [--check] [--input name=value] [--tool path]`.
- `examples/16_workflows.py`.

**Handoffs — `Agent(handoffs=[...])`**
- Another agent takes over the conversation: it sees what was said, answers the
  user itself, and keeps the conversation on the turns that follow. A `handoff`
  tool appears on any agent with somewhere to hand off to; `Agent.add_handoff`
  introduces two agents that hand to each other.
- `Handoff(agent, description=, history=, sticky=, on_handoff=)`. `history` is
  `"full"`, `"text"` (no tool calls or results), `"fresh"` (the user's last
  message) or a function; `sticky=False` hands over one turn only.
- `RunResult.handoffs` (`HandoffRecord`: source, target, reason), `.agent` for
  who answered and `.active_agent` for who has the next turn; usage, steps, tool
  calls and artefacts of every agent are merged onto the one result. A `handoff`
  stream event, and a `"handoff"` stop reason on the handing agent's own record.
- The session records who holds the conversation
  (`session.metadata["handoff"]`), so any process continuing it by id reaches
  the same agent. Every agent in a chain saves to the entry agent's store.
- `Agent(max_handoffs=5)` caps handoffs per run. A `handoff` hook event can
  block one; the permission gate, `forbid_tools` and governance (as a
  delegation) apply. A refused handoff is a tool error the model reads.
- An agent that cannot start, is gone from the configuration, or acts for a
  different user gives the conversation back to the agent that was called.
- `handoffs:` in blueprints (by agent name, cycles allowed) and in
  `AgentVersion`; `Agent.handoff_agent(name)`; the CLI chat prints who answered.
- A voice pipeline keeps the call with whoever was handed it.
- `examples/15_handoffs.py`.

**Modes — `Agent(mode=..., depth=...)`**
- `mode="chat" | "research" | "cowork"`, chosen where the agent is created, and
  `depth="fast" | "balanced" | "deep"` for how hard it works. An agent with no
  mode is unchanged. A mode only fills in what was left unset; `agent_harness.modes`
  builds a tuned one (`modes.research("deep", min_sources=12)`), and
  `modes.register_mode` adds your own.
- **chat** keeps the conversation between runs; `Agent.new_session()` starts a
  new one.
- **research** keeps a source ledger: `record_source` returns the number to cite,
  refuses a source nothing in the run returned, and the report is sent back for
  a citation to nothing, too few sources, or none cited. The source list is
  appended from the ledger and the report kept as `report.md`. Helpers share the
  lead's ledger.
- **cowork** keeps a binding todo list (`todo_write`), works in a workspace with
  `parse_document`, can put questions to a handler (`ask_user`), spins up helpers
  at `balanced` and `deep`, and hands back every file it created or changed as
  an artefact.
- `RunResult.mode`, `.depth`, `.todos` and `.sources`; `Todo` and `Source` types;
  a `progress` stream event and journal entry when the list or ledger changes.
- `mode:` / `depth:` in blueprints, `SubAgentSpec` and `AgentVersion`; the
  governance inventory records them.
- `agent-harness run|chat --mode --depth --workspace`.
- `Workspace.snapshot()` and `Workspace.changed(since)`.
- `examples/10_modes.py`.

**Sandboxes — `agent_harness.sandboxes`**
- `sandbox(name_or_url, **options)` returns a workspace that lives in a sandbox:
  `docker` and `podman` (a long-lived container, `runtime=` for gVisor or Kata,
  `mount=` to work in a host folder), `kubernetes` (a pod, created or attached
  to), `ssh`, `e2b`, `daytona`, `modal`, and `command` for anything that can be
  put in front of `sh -c`. `Agent(workspace=...)` takes one, or its name, URL or
  mapping; so do blueprints.
- `Sandbox`: a backend implements `_exec`, and reading, writing, listing and
  change detection are derived from it in shell that runs on busybox and GNU.
  `register_sandbox` names your own.
- `WorkspaceBroker(sandbox=..., **options)` gives every agent and isolated
  sub-agent its own sandbox; `Harness.aclose()` stops them all.
- `SandboxWorkspace.check()`, `available_sandboxes()`, and
  `agent-harness sandboxes [name] [--check]`; `--sandbox` on `run` and `chat`.
- Every `Workspace` operation has an async twin (`aread`, `awrite`, `alistdir`,
  `aremove`, `aexists`, `asnapshot`, `achanged`, `materialize`, `aclose`), which
  is what the tools now call.
- `examples/11_sandboxes.py`.

**Picking a conversation back up**
- `RunResult.sandbox_id`, alongside `session_id`. A session remembers the
  sandbox it was working in, so `agent.run(task, session=...)` puts the agent
  back in it; `sandbox_id=` on a run, or `id=` on any sandbox, names one by hand.
- `await agent.resume(session_id)` follows a conversation from then on, in any
  mode.
- `sandbox(..., keep=True)` leaves a sandbox for later; `workspace.destroy()`
  ends a kept one. A kept Docker container is started again if it has stopped.
- When the sandbox is gone, a new one is started, the text files the
  conversation wrote are restored from the session, and the model is told what
  was lost. `on_missing="error"` refuses instead.
- `agent-harness run|chat --sandbox-id --keep-sandbox`; the run footer prints
  the sandbox id.

**Sessions in a database — `agent_harness.sessions`**
- `session_provider(url)`: chats in SQLite, PostgreSQL, MySQL/MariaDB, MongoDB,
  Redis, DynamoDB, S3, Azure Blob or GCS, on the memory backends' own URLs and
  drivers. `Harness(sessions=url)`, `Harness.on(url)` for memory and chats on
  one pool, `sessions:` in a blueprint, `--sessions` on the CLI.
- A session has an owner (`user_id`, `tenant_id`, from the agent's trace) and a
  `version`. `SessionStore.list(user_id=, tenant_id=, agent=)`,
  `Session.summary()`.
- An agent acting for someone else is refused the session as if it did not
  exist, and the refusal is audited.
- A stale save raises `SessionConflict`. The agent then adds its own turn to
  what is there, or — if its history was compacted — keeps the run as a fork and
  says so in the new `RunResult.warnings`.
- A failed save no longer raises out of `run()`: the answer is returned with a
  warning.
- `SessionStore.check()` and `agent-harness sessions --store URL --check`;
  `--user`, `--tenant`, `--agent`, `--delete`, `--json`, `--backends`.
- `DurableSessionStore` for writing your own. `examples/12_sessions.py`.

**Attachments — images, files, audio, video**
- `agent.run(task, attachments=[...])` and `Message.user(text, attachments=[...])`
  take paths, URLs, bytes or blocks. `attach()`, `AudioBlock`, `VideoBlock`,
  `DocumentBlock`; `ImageBlock` gains `path` and `name`.
- Sent natively where the model takes them — PDFs to Claude, GPT and Gemini;
  audio to Gemini and OpenAI's audio models; video to Gemini — and as text where
  it does not: files are read with `parse_document`, recordings transcribed with
  `Harness(speech=...)`. `Provider.modalities` and `Provider.accepts()`.
- What cannot be sent ends the run before any model call. Text made from an
  attachment passes the input guardrails. A file is held by path, so saved
  sessions stay small. `agent-harness run --attach`.
- `examples/13_multimodal.py`.

**Voice — `agent_harness.voice`**
- `VoiceAgent`: speech in, the agent's answer spoken back, on any provider.
  Sentence-by-sentence synthesis that starts before the model has finished,
  barge-in that keeps only what was heard, turns joined when a pause was
  mid-sentence, and per-turn latency (`stt_ms`, `first_token_ms`,
  `first_audio_ms`).
- `RealtimeAgent`: speech-to-speech over OpenAI Realtime or Gemini Live, with
  the agent's instructions, its tools run under every rail (`Agent.call_tool`),
  its budget, and the transcript kept as a session.
- `EnergyVAD` (adaptive voice-activity detection), `SpeechChunker`,
  `OpenAISpeech` (any OpenAI-compatible STT/TTS), `FakeSpeech`, a `Resampler`
  that does not click at chunk joins, µ-law, and a dependency-free `WebSocket`
  client.
- `mode="voice"`; `agent-harness voice [--input WAV] [--realtime openai|gemini]`.
- Audio leaving for transcription, speech or a realtime model passes the
  `model_egress` hook. `examples/14_voice.py`.

### Changed

- In a mode, the final allowed step tells the model it is the last and disables
  tool calls, so the run ends with a hand-over rather than `MaxStepsExceeded`.
- A todo list and source ledger are pinned through context compaction.
- `agent-harness chat`: `/new` starts a session with its own id instead of
  overwriting the previous one.
- `DeliverableStore.put` keeps an artefact that has a file but no text (an
  image, a large file) by its bytes, rather than storing it empty.
- Inside a sandbox that is its own machine, `shell` is present and neither it
  nor `run_python` asks for approval. On a local workspace nothing changes: the
  shell is opt-in and both ask every time.
- `Agent(allow_shell=...)` and a blueprint's `allow_shell:` default to unset
  rather than `False`, so a sandbox keeps its shell unless told otherwise.
- `WorkspaceBroker` no longer creates a temporary directory until a workspace is
  asked for. `Harness.aclose()` awaits `WorkspaceBroker.aclose()`.
- The workspace tools, `run_python`, `parse_document`, `render_chart` and
  `write_report` are now async, and go through the workspace's async methods.

- A run that is not continuing a conversation is now *added* to its session's
  record instead of replacing it; it still starts with a clean context.
- A run handed its own `messages` and no session (an agent used as a tool, a
  replay) is no longer saved over the agent's session.
- `FileSessionStore` replaces a session file in one step, and refuses a save
  from a stale version.
- `Harness.local(...)` accepts replacements for any of its file-backed parts.

- A run whose caller cancels it, or stops reading its stream, is taken off the
  stop controller's list of running agents.
- A user `Message` carrying non-text blocks no longer loses them when it is the
  task of a run.
- `FakeProvider(stream_words=True, stream_delay=...)` streams a reply a word at
  a time.

### Fixed

- A blueprint agent whose instructions contain braces that are not a prompt
  reference — a JSON example, say — no longer fails to build.
- A sqlite URL such as `sqlite:///./memory.db` pointed at the filesystem root,
  and `sqlite:////abs/path` at a relative path.
- An attachment a provider could not encode was sent as its printed form — the
  raw base64 — instead of being left out.
- `AzureBlobMemory` raised when asked to delete a blob that was already gone.
- Resuming a session whose last run was stopped mid-step (a budget, a stop
  request) sent a tool call with no result, which providers reject. Unanswered
  tool calls are now closed when the history is picked back up.

## [0.1.4] — 2026-09-24

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

[Unreleased]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MuhammadHusnainAli/agent-harness-adk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MuhammadHusnainAli/agent-harness-adk/releases/tag/v0.1.0
