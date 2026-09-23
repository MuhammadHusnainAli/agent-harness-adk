# agent-harness-adk

[![CI](https://github.com/MuhammadHusnainAli/agent-harness-adk/actions/workflows/ci.yml/badge.svg)](https://github.com/MuhammadHusnainAli/agent-harness-adk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-harness-adk.svg)](https://pypi.org/project/agent-harness-adk/)
[![Python](https://img.shields.io/pypi/pyversions/agent-harness-adk.svg)](https://pypi.org/project/agent-harness-adk/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fast, lightweight harness for building production AI agents in Python.

Agents, sub-agents, skills, prompts, tools, MCP servers, memory — and the runtime
rails underneath them: permissions, budgets, hooks, guardrails, tracing,
checkpoints and isolated workspaces. Three model providers, one loop, no
framework lock-in.

```bash
pip install agent-harness-adk     # or: uv add agent-harness-adk
```

```python
import agent_harness              # installed as agent-harness-adk, imported as agent_harness
```

Python 3.10 – 3.14. Three dependencies (`pydantic`, `httpx`, `pyyaml`), ~100 ms
to import, and no vendor SDKs — the provider adapters speak HTTP directly so
Anthropic, OpenAI and Gemini all travel the same retry, cost and tracing path.

---

## 60 seconds

```python
from agent_harness import Agent, tool

@tool
def order_status(order_id: str) -> str:
    """Look up the status of a customer order.

    Args:
        order_id: the order number, digits only.
    """
    return db.lookup(order_id)

agent = Agent(
    "support",
    "Answer customer questions about orders. Look the order up before answering.",
    tools=[order_status],
)

result = agent.run_sync("Where is order 4182?")
print(result.output, result.cost_usd, result.steps)
```

The decorator reads your signature and docstring and builds the JSON Schema the
model needs. Arguments coming back from the model are validated before your
function is called. `await agent.run(...)` is the real implementation;
`run_sync` is the wrapper for scripts and notebooks.

---

## The shape of the system

```
         ┌──────────────────────────────────────────────────────────┐
         │  Orchestrator — plan · staff · run · consolidate · review │
         └───────────────┬──────────────────────────────────────────┘
                         │ staffing decision: reuse or create?
          ┌──────────────┴───────────────┐
   ┌──────▼──────┐               ┌───────▼────────┐
   │  The bench  │               │  The factory   │
   │ pre-defined │               │ a new spec     │
   │ sub-agents  │               │ written at run │
   └──────┬──────┘               └───────┬────────┘
          └──────────────┬───────────────┘
                  ┌──────▼───────┐
                  │  Agent loop  │  think → act → observe → repeat
                  └──────┬───────┘
   ┌─────────────────────┼─────────────────────────┐
   │ context assembler   │ tools · skills · MCP    │  memory: user · session
   │ context compactor   │ workspace · providers   │  orchestrator · sub-agent
   └─────────────────────┴─────────────────────────┘
   rails: permissions · budget · hooks · guardrails · tracing · journal ·
          cache · checkpoints · sessions · scheduler · router
```

---

## Tools

```python
from agent_harness import tool, ToolContext

@tool(permission="ask", cacheable=True, tags=["billing"])
async def issue_refund(order_id: str, amount: float, ctx: ToolContext) -> str:
    """Refund a customer. Costs real money.

    Args:
        order_id: the order to refund.
        amount: how much, in EUR.
    """
    ctx.log("refunding", order=order_id)
    return await billing.refund(order_id, amount)
```

- Sync or async, it makes no difference.
- A parameter named `ctx` (or annotated `ToolContext`) is injected and hidden
  from the model.
- A pydantic model as a parameter type is validated and passed through as a
  model, not a dict.
- `permission` can tighten the policy for one tool. It can never loosen it.
- Tools run in parallel when the model asks for several at once.

Built-ins in `agent_harness.toolkits`:

| Tool | Notes |
|---|---|
| `now`, `calculate` | exact arithmetic, no `eval` |
| `make_corpus_search` | keyword search over documents you hand it |
| `make_fetch_tool`, `make_http_tool` | domain allowlist, private-address refusal, HTML stripping |
| `parse_document` | text, Markdown, CSV, TSV, JSON, JSONL, HTML, XML with no dependencies; PDF, DOCX and OCR with an optional install each |
| `bar_chart`, `line_chart`, `render_report` | inline SVG that works in light and dark, plus markdown reports |
| `make_python_tool` | run code in the workspace — asks for approval every time |

A workspace brings `fs_read`, `fs_write`, `fs_list`, `fs_delete` and — only when
you ask for it — `shell`.

## Skills

A skill is packaged know-how: a folder with `SKILL.md` and, optionally, its own
tools and reference files.

```
skills/refunds/SKILL.md
---
name: refunds
description: How we process a refund, including the approval thresholds.
---
1. Check the order is inside the 30-day window...
```

```python
agent = Agent("support", "Answer support questions.", skills="./skills")
```

Only each skill's **name and description** go into the system prompt. The body
is loaded on demand through the `load_skill` tool, so twenty skills cost twenty
lines of context instead of twenty documents. A `tools.py` in the skill folder is
imported and its tools come along with it.

## Prompts

```python
from agent_harness import Prompt, PromptLibrary

triage = Prompt("triage", "Sort {ticket} into {buckets}.", version="2")
triage.render(ticket="T-1", buckets="p1/p2/p3")

library = PromptLibrary.from_dir("./prompts")   # .md files with YAML frontmatter
library.render("triage", ticket="T-1")
```

Versioned, reviewable, `.partial()`-able, composable with `+`. Jinja is used
only when a template contains a `{% %}` statement and `jinja2` is installed.

## Memory — four scopes

| Scope | Stored | Loaded | Lifetime |
|---|---|---|---|
| **user** (`user.md`) | preferences, standards, settled decisions | in full, every message | permanent, rewritten at session close |
| **session** | the whole conversation plus its artefacts | in full | this session |
| **orchestrator** | plans, staffing decisions, spend, findings | **a digest only** | this job, then distilled into user memory |
| **sub-agent** | only the resources its task produced | nothing carried in | the task |

```python
agent = Agent("assistant", memory=True)          # the default
await agent.run("I bill my customers in EUR")
await agent.run("What currency do I use?", messages=[])   # clean run, still knows

print(await agent.close_session())   # session close → user.md rewritten
```

### Whose memory is it? — `trace`

A trace says who a memory belongs to. Pass one and every record is stamped with
it on the way in and filtered by it on the way out, so one store serves any
number of users without them ever seeing each other.

```python
agent = Agent("support", trace="alice")                      # a bare user id
agent = Agent("support", trace=Trace(user_id="alice", session_id="s-42"))
agent = Agent("support", trace=Trace(tenant_id="acme", user_id="alice"))
```

```python
alice = MemoryManager(store, trace="alice")
bob   = MemoryManager(store, trace="bob")

await alice.user.remember("bills in EUR")
await bob.user.load()        # "" — bob never sees it
```

One manager, many users, one backend:

```python
shared = MemoryManager(store)
await shared.for_trace(request.user_id).user.remember(fact)
```

`for_trace` reuses the store and its vector index, so serving a request per user
costs a small object rather than a rebuilt index.

`scope` decides what documents like `user.md` are namespaced by — `"user"` (the
default: preferences follow the person across sessions), `"session"`, `"tenant"`
or `"global"`. A record written without a trace stays visible to everyone: it is
shared, not orphaned.

### Where is it stored? — `memory/providers`

Thirteen backends, one contract. The agent loop never learns which is behind it.

```python
from agent_harness import memory_provider

store = memory_provider("postgresql://user:pass@host/agents")
store = memory_provider("mongodb://localhost:27017", database="agents")
store = memory_provider("s3://my-bucket/agent-memory")
store = memory_provider("sqlite:///./memory.db")

agent = Agent("support", memory=MemoryManager(store, trace="alice"))
```

| Backend | Class | Needs |
|---|---|---|
| in-process | `InMemoryStore` | — |
| files | `FileStore` | — |
| **SQLite** | `SQLiteMemory` | — (standard library) |
| **PostgreSQL** | `PostgresMemory` | `asyncpg` or `psycopg` |
| **MySQL / MariaDB** | `MySQLMemory` | `aiomysql` |
| **MongoDB** | `MongoMemory` | `motor` or `pymongo` |
| **Redis** | `RedisMemory` | `redis` |
| **DynamoDB** | `DynamoDBMemory` | `boto3` |
| **Elasticsearch / OpenSearch** | `ElasticsearchMemory` | `elasticsearch` |
| **Amazon S3** | `S3Memory` | `boto3` |
| **Azure Blob Storage** | `AzureBlobMemory` | `azure-storage-blob` |
| **Google Cloud Storage** | `GCSMemory` | `google-cloud-storage` |
| **your own API** | `HTTPMemory` | — (httpx already ships) |

Nothing is imported until you ask for it — a driver you do not use costs nothing
at import, and one you have not installed names its own `pip install` rather than
raising `ImportError` somewhere deep in a run:

```python
from agent_harness import available_backends
available_backends()
# {'sqlite': True, 'postgres': False, 's3': False, ...}
```

Choosing between them:

- **SQLite** is the right default for a single service — durable, indexed, no
  server to run.
- **Postgres, MySQL and Mongo** index on the trace, so reading one user's memory
  is one query however many users you have.
- **Redis** suits session-scoped memory: pass `ttl=` and it expires itself.
- **Elasticsearch** is the only backend where `search()` is ranked by the engine,
  so recall is good without an embedder.
- **S3, Azure Blob and GCS** lay keys out so a trace is a prefix. That makes one
  user's memory a single listing, but anything narrower is filtered after the
  fetch — treat them as durable archival rather than a hot query path.
- **HTTPMemory** is for when memory must live behind a service you already run.

`register_backend("cassandra", "myapp.memory", "CassandraMemory")` adds your own.

The agent gets `remember` and `recall` tools. Recall is semantic: embeddings
come from whatever you configure, and the default is a deterministic offline
hashing embedder so semantic recall works with no extra dependency and no
network. Swap it for the real thing when you want to:

```python
from agent_harness import MemoryManager, ProviderEmbedder, OpenAIProvider, FileStore

memory = MemoryManager(FileStore(".harness/memory"),
                       embedder=ProviderEmbedder(OpenAIProvider()))
```

## Context: when it compresses

Every step, the conversation is measured and compacted if it is over the
threshold. You choose where that is:

```python
Agent("a", compact_at=10_000)        # an absolute token count
Agent("b", compact_at=50_000)
Agent("c", compact_at=0.5)           # or a fraction of the model's context window
Agent("d")                           # default: two thirds of the window
```

Compaction happens in two stages, so the cheap thing is tried first:

1. **Evict** — oversized tool results are hollowed out, keeping their first 400
   characters. Cheapest tokens to lose, and it cannot break a `tool_use`/`tool_result`
   pair because nothing is removed.
2. **Summarise** — if it is still over, the head of the conversation is summarised
   by a cheap model call and the tail kept verbatim. The cut moves forward until
   no tool result is left without its call.

```python
Agent("a",
      compact_at=10_000,        # start compacting here
      compact_target=0.6,       # compress down to 60% of that
      compact_keep_last=8)      # never touch the last 8 messages
```

Pass `compactor=ContextCompactor(...)` to replace the strategy wholesale, and
`memory.session.pin("the deadline is Friday")` for facts that must survive it.

## Sub-agents: the bench and the factory

Before staffing a task, the orchestrator asks one question: **is there already a
sub-agent that covers this?**

```python
from agent_harness import Agent, SubAgentSpec

manager = Agent(
    "manager",
    "Delegate the lookups, then consolidate what comes back.",
    tools=[lookup],
    subagents=[
        SubAgentSpec(name="revenue_reader", description="Finds revenue figures.",
                     instructions="Look up the figure and report it with its source.",
                     tools=["lookup"], tier="fast"),
        SubAgentSpec(name="cost_reader", description="Finds cost figures.",
                     tools=["lookup"], tier="fast"),
    ],
)
result = await manager.run("How did Q3 go?")
for child in result.children:
    print(child.agent, child.steps, child.cost_usd)
```

A `delegate` tool appears automatically. Sub-agents **start clean** — no parent
transcript, no parent memory — and hand back a result, not a conversation.
Delegation does not cascade by default, and a spec's `tools` list is a hard
allowlist. Ask for several delegations in one turn and they run in parallel
under the concurrency cap.

### Agents it writes for itself, at run time

An ordinary agent can build its own specialists mid-run, within a budget you set:

```python
agent = Agent(
    "core",
    "Break the work up and give each part its own specialist.",
    tools=[lookup, publish],
    runtime_agents="enable",     # or "disable", or a plain bool
    max_runtime_agents=5,        # 0-100; the ceiling for one run
)
```

That adds a `spawn_agent` tool. When the agent decides it needs five workers, it
calls it five times in one turn and they run in parallel — each one written for
its task by the factory (name, instructions, tool allowlist, model tier, step
ceiling), then run, with only its result handed back.

```python
result = await agent.run("Reconcile these five ledgers.")
print(agent.total_spawned, [c.agent for c in result.children])
# 5 ['ledger_2024', 'ledger_2025', ...]
```

The rules around it:

- **The budget is per run** and refreshes on the next one. `agent.runtime_agents_remaining`
  is what is left; past the ceiling the tool says so and the agent finishes with
  what it has rather than failing.
- **It does not cascade.** A spawned specialist cannot spawn its own.
- **Least privilege.** A specialist gets the tools its spec asked for, narrowed to
  what the parent holds; `runtime_agent_tools=[...]` caps that further.
- **Every spin-up is audited**, counted against `Budget(max_subagents=...)`, and
  refused once the run is stopped.
- **Spawned specialists stay addressable** by name through `delegate` for the rest
  of the run, so the second task for the same worker costs nothing extra to set up.

Nothing on the bench fits? The factory writes a new specialist during the run —
name, instructions, tool allowlist, model tier, step ceiling and workspace
isolation — and that specialist exists only for this job.

```python
from agent_harness import Bench
Bench.standard().names
# ['compliance_checker', 'data_analyst', 'document_extractor', 'drafting',
#  'planner', 'report_writer', 'research', 'validator']
```

## The orchestrator

```python
from agent_harness import Orchestrator, Budget

boss = Orchestrator("boss", max_concurrency=4, review=True, max_rework=1,
                    budget=Budget(max_usd=2.00))
result = await boss.run("Summarise how Q3 went, with the numbers cited.")
```

1. **Plan** — acceptance tests are written *before* any work starts, then the
   task graph, then a cost estimate.
2. **Staff** — reuse from the bench, else build with the factory.
3. **Run** — dependency-ordered waves, parallel inside each wave, per-task
   retries, dependent tasks receive only what they depend on.
4. **Consolidate** — merge, de-duplicate, rank, attribute.
5. **Review** — an independent critic checks the deliverable against the
   definition of done; a rejection becomes new tasks and one rework round.

`result.data["plan"]` and `result.data["review"]` carry the full record.

## MCP

```python
from agent_harness import Agent, MCPManager, MCPServer

servers = [
    MCPServer(name="files", command="npx",
              args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]),
    MCPServer(name="api", url="https://mcp.internal/rpc",
              headers={"authorization": "Bearer ..."}),
]

async with MCPManager(servers) as mcp:
    agent = Agent("analyst", "Answer from the files.", tools=mcp.tools())
    print((await agent.run("What is in /data/report.md?")).output)
```

Both transports (stdio and streamable HTTP), tools, resources and prompts. A
server that will not connect is reported in `mcp.errors`, not raised into your
run. `allowed_tools` trims what a server may expose.

## Guardrails: what an agent must do to be done

The content engine (`Guardrails`) polices *text* — secrets, injection, size, on
every path in and out. `AgentGuardrails` polices *behaviour*: which tools an
agent may touch, and what has to be true of its answer before that answer is
accepted.

```python
from agent_harness import Agent, AgentGuardrails

support = Agent(
    "support",
    "Answer order questions.",
    tools=[order_status, issue_refund],
    guardrails=AgentGuardrails(
        require_tools=["order_status"],   # look it up, never guess
        forbid_tools=["issue_refund"],    # not this agent's job
        must_include=["order"],
        require_citation=True,
        no_placeholders=True,             # no "TODO", no "[insert name]"
        max_cost_usd=0.25,
        on_violation="retry",             # tell it what is missing, let it fix it
    ),
)
```

A forbidden tool is refused **before it runs**. Everything else is checked when
the agent tries to finish: if something is unmet the agent is told, in words,
and gets another turn —

```
That answer does not meet this task's requirements yet:
- you answered without calling order_status — call order_status and answer from what it returns
- your answer cites nothing — give the source for each claim, or say you could not find one

Put it right and answer again.
```

which is usually all it needs. `on_violation` decides what happens when it does
not: `"retry"` (the default, up to `max_retries`), `"fail"` (stop the run), or
`"warn"` (deliver it, record the problem in `result.violations`).

The checks ship as objects, so you can compose them directly or write your own:

| Check | Fails when |
|---|---|
| `RequireTools(*names)` | it answered without calling them |
| `ForbidTools(*names)` | it called one anyway (post-hoc audit) |
| `MustInclude` / `MustNotInclude` | the answer misses, or contains, a phrase |
| `MustMatch(pattern)` | the answer is not in the shape asked for |
| `MinLength(chars)` | a one-word answer to a question that needed working through |
| `RequireCitation()` | nothing in the answer points at a source |
| `RequireJSON()` / `RequireStructured()` | the output contract was not met |
| `NoPlaceholders()` | it handed back `TODO`, `[insert x]`, `lorem ipsum` |
| `MaxSteps(n)` / `MaxCost(usd)` | it got there, but not within budget |
| `Custom(fn)` | your own rule — return `False` or `(False, "why")` |

Sub-agents carry their own, declared in the spec so it stays serialisable:

```python
SubAgentSpec(
    name="researcher",
    description="Finds things out.",
    guardrails={"require_citation": True, "forbid_tools": ["publish"],
                "max_retries": 1},
)
```

And `AgentGuardrails(content=Guardrails(...))` gives one agent stricter text
rules than the rest of the harness.

## The rails

```python
from agent_harness import (Harness, Budget, PolicyGate, HookEngine, Guardrails,
                           console_exporter)

harness = Harness.local(".harness")          # sessions, memory, traces, checkpoints
harness.policy = PolicyGate("allow", ask=["issue_refund"], deny=["shell"],
                            approver=my_approver)
harness.guardrails = Guardrails(strict=True)
harness.tracer.add_exporter(console_exporter())
harness.reset_budget(Budget(max_usd=0.50, max_steps=8, max_subagents=4))

hooks = HookEngine()

@hooks.on("pre_tool")
def cap_refunds(ctx):
    if ctx.data["tool"] == "issue_refund" and ctx.data["args"]["amount"] > 100:
        ctx.block("refunds over 100 EUR need a manager")

agent = Agent("refunds", harness=harness, hooks=hooks, tools=[issue_refund])
print(harness.report())   # spend by agent and task, cache hit rate, concurrency
```

| Rail | What it does |
|---|---|
| `PolicyGate` | allow / ask / deny per action, glob rules, conditional on arguments, approver callback |
| `BudgetGuard` | spend, token, step, tool-call and sub-agent ceilings; child guards roll up to the parent |
| `RateGuard` | requests- and tokens-per-minute pacing, so you are not rate-limited by the provider |
| `HookEngine` | 12 events; `pre_tool` can block or rewrite arguments, `post_tool` can rewrite the result |
| `Guardrails` | secret redaction, private-key blocking, injection warnings, size caps — on tool output *and* final answers |
| `StopController` | abort a run and drain the sub-agents; a human is always in charge |
| `Tracer` | one span per run, step, model call, tool and sub-agent; console and JSONL exporters |
| `AuditTrail` | immutable, hash-chained who-did-what; `verify()` names the first tampered entry |
| `ServiceHealth` | latency, failure rate and saturation per model, tool and MCP server |
| `RunJournal` | what each agent was asked and what it returned, append-only |
| `ResultCache` | identical task + identical input served from cache, memory and disk tiers |
| `Checkpointer` + `Replayer` | step snapshots, a timeline, and resume-from-any-step |
| `RecordingProvider` / `ReplayProvider` | record a run once, reproduce it exactly with no network and no spend |
| `DeliverableStore` | the documents and reports a run produced, versioned and digested |
| `SessionStore` | resume, fork or branch a run; a long job survives a restart |
| `WorkspaceBroker` | a jailed directory per sub-agent (or a shared one for handovers), local or Docker |
| `ConcurrencyScheduler` | semaphore, queue, backpressure, peak tracking |
| `ModelRouter` | per-task model and effort tier instead of one model for everything |
| `SpecCompiler` | a sub-agent blueprint → the exact provider payload, inspectable before you spend |

Path safety is enforced, not clamped: a workspace tool given `../../etc/passwd`
refuses rather than resolving it. `shell` is absent unless the workspace was
created with `allow_shell=True`, and even then it asks for approval.

## Providers

```python
Agent("a", model="claude-opus-5")      # → Anthropic
Agent("b", model="gpt-4.1")            # → OpenAI
Agent("c", model="gemini-2.5-pro")     # → Gemini
Agent("d", provider=OpenAIProvider(base_url="http://localhost:11434/v1"))
```

The provider is inferred from the model id. Keys come from `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`. Anything that speaks the OpenAI wire format
(Azure, Groq, Together, Ollama, vLLM) works through `OpenAIProvider(base_url=...)`,
and `register_provider("name", MyProvider)` adds your own.

Adapters normalise everything the loop depends on: tool calls, tool results,
thinking blocks, cache tokens, stop reasons and refusals. Cost is computed per
call from a built-in price table (`register_model` to extend it), so
`result.cost_usd` is real money, not an estimate.

## Stopping, reproducing, and proving it got better

A human is always in charge:

```python
harness.stop("the customer withdrew the request")   # drains; nothing new starts
harness.control.abort("pull the plug")              # cancels what is in flight
```

The loop checks between steps and before every tool, so a stop lands at a safe
boundary and the work already done is kept.

Reproduce a failure before you fix it:

```python
from agent_harness import RecordingProvider, ReplayProvider

agent = Agent("support", provider=RecordingProvider(AnthropicProvider(), "run.jsonl"))
await agent.run("...")                # once, against the real model

replay = ReplayProvider("run.jsonl")  # then as often as you like: no network, no spend
twin = Agent("support", provider=replay)
assert (await twin.run("...")).output == original.output
```

Or travel back to any step and try it differently:

```python
print(await harness.replayer.timeline(result.run_id))
again = await harness.replayer.resume(agent, result.run_id, step=3,
                                      task="Give the figure, not a summary.")
```

And prove a change actually helped:

```python
from agent_harness import Evaluator, Expect, GoldenTask

suite = Evaluator([
    GoldenTask(id="refund-window", input="Can I refund a 40-day-old order?",
               expect=Expect(contains=["30-day"], not_contains=["yes, of course"])),
    GoldenTask(id="uses-lookup", input="Where is order 4182?",
               expect=Expect(tool_called="order_status", max_steps=4)),
])

baseline = await suite.run(agent, label="before"); baseline.save("baseline.json")
# ... change the prompt ...
after = await suite.run(agent, label="after")
print(after.compare(baseline).render())
# REGRESSED: 100.00% → 50.00% (-50.00%)
#   REGRESSED:    uses-lookup
```

Expectations can check the text, the tools that were called, the structured
output, the step count or the cost. `llm_judge` is there for genuinely
open-ended answers — reach for it last; it costs money and it can be wrong.

## Streaming and structured output

```python
async for event in agent.stream("Summarise the incident"):
    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "tool_result":
        print(f"\n· {event.data['tool']}")
    elif event.type == "run_end":
        result = event.data["result"]
```

```python
from pydantic import BaseModel

class Ticket(BaseModel):
    id: str
    priority: int
    summary: str

agent = Agent("triage", output_type=Ticket)
result = await agent.run("Customer cannot log in since the deploy")
result.data.priority     # a validated Ticket, retried if the model got it wrong
```

## Testing your agents

```python
from agent_harness import Agent, FakeProvider, Harness, tool_call

provider = FakeProvider([tool_call("order_status", order_id="4182"),
                         "It ships Thursday."])
agent = Agent("support", provider=provider, harness=Harness.testing(provider),
              tools=[order_status])

result = await agent.run("Where is order 4182?")
assert result.output == "It ships Thursday."
assert provider.requests[0].system.startswith("You are support")
```

No network, no keys, no recorded cassettes. Script strings, tool calls, whole
messages, exceptions, or a callable that inspects the request and answers
accordingly. The harness's own suite is 296 tests and runs in half a second.

## CLI

```bash
agent-harness run "summarise this incident" --tools --stream --state .harness
agent-harness chat --skills ./skills --state .harness --approve
agent-harness models
agent-harness sessions --state .harness
agent-harness journal --state .harness
agent-harness mcp npx -y @modelcontextprotocol/server-filesystem /data
```

## Design notes

- **Async core, sync wrapper.** Parallel sub-agents, MCP and the concurrency cap
  all need it. `run_sync` covers scripts.
- **Compaction never orphans a tool call.** Fat tool results are hollowed out
  first, and the summarise-the-head fallback moves its cut forward until no
  `tool_result` is left without its `tool_use`. Naive trimming corrupts a
  conversation; this does not.
- **Least privilege by default.** Sub-agents get an explicit tool allowlist,
  delegation does not cascade, `shell` is opt-in, and a tool's own permission can
  only tighten the policy.
- **Everything is optional.** An `Agent` with no memory, no skills and no
  sub-agents is a tight `while` loop around one model call.

## Contributing

```bash
git clone https://github.com/MuhammadHusnainAli/agent-harness-adk
cd agent-harness-adk
uv sync --extra dev
uv run pytest -q                        # no API key needed — everything runs on FakeProvider
uv run ruff check src tests examples
```

Every push to `main` runs the suite on Python 3.10, 3.11, 3.12, 3.13 and 3.14,
lints, builds the wheel, installs it into a clean environment and smoke-tests
it, and runs every example without an API key.

[CONTRIBUTING.md](CONTRIBUTING.md) has the details — including the three things
this project is picky about: the dependency count, the import time, and the
3.10 floor.

| | |
|---|---|
| Report a bug or ask for a feature | [Issues](https://github.com/MuhammadHusnainAli/agent-harness-adk/issues) |
| Ask how to do something | [Discussions](https://github.com/MuhammadHusnainAli/agent-harness-adk/discussions) · [SUPPORT.md](SUPPORT.md) |
| Report a vulnerability | **Privately** — [SECURITY.md](SECURITY.md) |
| Community standards | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Cut a release (maintainers) | [RELEASING.md](RELEASING.md) |

Running agents safely — tool allowlists, policy gates, workspace isolation and
what this library does *not* defend against — is covered in
[SECURITY.md](SECURITY.md). Worth reading before you give an agent a tool that
writes, spends or sends.

## Status

0.1.0 — the first release. The public API above is what we intend to keep.
Changes are recorded in [CHANGELOG.md](CHANGELOG.md).

Not in this release: a vector-database backend (the built-in index is exact
brute force, fine to ~50k records) and provider-side batch APIs. OCR, PDF and
DOCX parsing work through an optional install each rather than shipping in the
default dependency set.

## Licence

MIT — see [LICENSE](LICENSE).
