# agent-harness

A fast, lightweight harness for building production AI agents in Python.

Agents, sub-agents, skills, prompts, tools, MCP servers, memory — and the runtime
rails underneath them: permissions, budgets, hooks, guardrails, tracing,
checkpoints and isolated workspaces. Three model providers, one loop, no
framework lock-in.

```
pip install agent-harness        # or: uv add agent-harness
```

Three dependencies (`pydantic`, `httpx`, `pyyaml`), ~100 ms to import, and no
vendor SDKs — the provider adapters speak HTTP directly so Anthropic, OpenAI and
Gemini all travel the same retry, cost and tracing path.

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

Built-ins: `agent_harness.toolkits` has `now`, `calculate`, `make_corpus_search`,
`make_fetch_tool` (domain allowlist, private-address refusal, HTML stripping) and
`make_http_tool`. A workspace brings `fs_read`, `fs_write`, `fs_list`,
`fs_delete` and — only when you ask for it — `shell`.

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

The agent gets `remember` and `recall` tools. Recall is semantic: embeddings
come from whatever you configure, and the default is a deterministic offline
hashing embedder so semantic recall works with no extra dependency and no
network. Swap it for the real thing when you want to:

```python
from agent_harness import MemoryManager, ProviderEmbedder, OpenAIProvider, FileStore

memory = MemoryManager(FileStore(".harness/memory"),
                       embedder=ProviderEmbedder(OpenAIProvider()))
```

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
| `HookEngine` | 12 events; `pre_tool` can block or rewrite arguments, `post_tool` can rewrite the result |
| `Guardrails` | secret redaction, private-key blocking, injection warnings, size caps — on tool output *and* final answers |
| `Tracer` | one span per run, step, model call, tool and sub-agent; console and JSONL exporters |
| `RunJournal` | what each agent was asked and what it returned, append-only |
| `ResultCache` | identical task + identical input served from cache, memory and disk tiers |
| `Checkpointer` | step-level snapshots; resume or replay from any prior step |
| `SessionStore` | resume, fork or branch a run; a long job survives a restart |
| `WorkspaceBroker` | a jailed directory per sub-agent (or a shared one for handovers), local or Docker |
| `ConcurrencyScheduler` | semaphore, queue, backpressure, peak tracking |
| `ModelRouter` | per-task model and effort tier instead of one model for everything |

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
accordingly. The harness's own suite is 150 tests and runs in half a second.

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

## Status

0.1.0 — the first release. The public API above is what we intend to keep.

Not in this release: a vector-database backend (the built-in index is exact
brute force, fine to ~50k records), OCR and document parsing, and provider-side
batch APIs.

## Licence

MIT.
