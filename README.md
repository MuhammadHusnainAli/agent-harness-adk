# agent-harness-adk

[![CI](https://github.com/MuhammadHusnainAli/agent-harness-adk/actions/workflows/ci.yml/badge.svg)](https://github.com/MuhammadHusnainAli/agent-harness-adk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-harness-adk.svg)](https://pypi.org/project/agent-harness-adk/)
[![Python](https://img.shields.io/pypi/pyversions/agent-harness-adk.svg)](https://pypi.org/project/agent-harness-adk/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fast, lightweight harness for building production AI agents in Python.

Agents, sub-agents, skills, prompts, tools, MCP servers, memory — and the runtime
rails underneath them: permissions, budgets, hooks, guardrails, tracing,
checkpoints and isolated workspaces. Twenty LLM providers, one loop, no
framework lock-in.

```bash
pip install agent-harness-adk     # or: uv add agent-harness-adk
```

```python
import agent_harness              # installed as agent-harness-adk, imported as agent_harness
```

Python 3.10 – 3.14. Three dependencies (`pydantic`, `httpx`, `pyyaml`), ~100 ms
to import, and no vendor SDKs — the provider adapters speak HTTP directly so
every backend travels the same retry, circuit-breaker, cost and tracing path.

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
   governance: policy · identity · residency · oversight · evidence
```

---

## Modes: chat, research, cowork

An agent with no mode is the plain loop, exactly as you configured it. A mode is
a way of working laid over that loop, chosen where the agent is created:

```python
from agent_harness import Agent, Workspace, modes

pal       = Agent("pal", mode="chat")
analyst   = Agent("analyst", mode="research", depth="deep", tools=[search, http_fetch])
colleague = Agent("colleague", mode="cowork", workspace=Workspace("./project"))
```

| mode | what it is | what the harness adds |
|---|---|---|
| `chat` | a conversation | the thread is kept between runs — the second `run()` is the second turn |
| `research` | a report someone can check | a todo list of the questions, a **source ledger**, citations checked against it, the source list appended, `report.md` kept as an artefact |
| `cowork` | a task handed over and carried through | a todo list it may not leave open, a workspace, `parse_document`, questions to you, helpers in parallel, every file it wrote handed back |

`depth` is the second dial. The name is also the model tier it asks the router
for — leave it out and the agent works at `balanced` on whatever model the
router would have picked anyway.

| | `fast` | `balanced` | `deep` |
|---|---|---|---|
| `chat` steps | 6 | 12 | 20 |
| `research` steps · sources needed · helpers | 12 · 2 · 0 | 30 · 4 · 3 | 60 · 8 · 6 |
| `cowork` steps · helpers | 25 · 0 | 60 · 3 | 120 · 8 |

A mode only fills in what you left unset — `Agent(mode="cowork", max_steps=10)`
takes ten steps, and `model=` is never overridden. To tune a mode itself, build it:

```python
Agent("analyst", mode=modes.research("deep", min_sources=12, helpers=4))
Agent("colleague", mode=modes.cowork("balanced", ask=ask_me, python=True))
```

**Chat** keeps the conversation on the agent. `agent.new_session()` starts a
fresh one; `session="..."` picks an old one back up, and `await
agent.resume("...")` follows it from then on (see [picking a conversation back
up](#picking-a-conversation-back-up) for the sandbox that goes with it). A run that was cut off
mid-step — a budget, a stop — no longer poisons the thread: the tool calls it
left unanswered are closed before the next turn is sent.

**Research** makes the report checkable rather than asking the model to be
careful:

```python
result = await analyst.run("How did Q3 go, and what is expected for Q4?")

result.output       # "...revenue was 4.2M [1]...\n\n## Sources\n- [1] Q3 results — https://..."
result.sources      # [Source(id=1, ref="https://...", finding="Q3 revenue 4.2M", agent="analyst")]
result.violations   # [] — or what the report still lacks
```

- The agent calls `record_source(ref, finding)` for everything it relies on and
  gets back the number to cite. A `ref` that nothing in the run actually
  returned — no tool result, no tool argument, not the task — is refused, so a
  source cannot be invented after the fact (`verify_sources=False` turns that off).
- A citation to a number that was never recorded sends the report back. So does
  having fewer sources than the depth requires, or citing none of them.
- Helpers spun up for parallel reading write in the same ledger, so `[4]` means
  one thing across the whole run.
- With nothing to read with — no tool, no sub-agent — the run fails before the
  first model call, not after a report written from memory.

**Cowork** is the long one. Give it a folder and a task:

```python
async def ask_me(question: str, options: list[str]) -> str:
    return await my_ui.prompt(question, options)       # sync works too

colleague = Agent("colleague", mode=modes.cowork("deep", ask=ask_me),
                  workspace=Workspace("./project"))
result = await colleague.run("Turn the interview notes into a findings deck outline.")

result.todos        # [Todo(content="Read the notes", status="done"), ...]
result.artifacts    # every file created or changed in the workspace, with its path
```

- `todo_write` is the plan, and it is binding: the run may not finish while an
  item is pending or in progress. Dropping one means marking it `skipped` with a
  note saying why.
- Without an `ask` handler there is no `ask_user` tool, and the agent is told to
  make the reasonable assumption and say what it assumed.
- `python=True` adds `run_python` in the workspace. On this machine it asks for
  approval on every call, like the shell — a mode never loosens the permission
  gate. Give it a [sandbox](#sandboxes-somewhere-else-to-work) and it has a shell
  it can simply use.
- The conversation is kept, so "now make it shorter" works on the same files.

Three things hold in every mode:

- **Unfinished work is sent back, then delivered honestly.** What a mode requires
  is retried up to `retries` times (two by default). If it still falls short the
  answer is returned with `result.violations` naming what is missing — not turned
  into an error.
- **The last step is for handing over.** On its final allowed step the model is
  told so and cannot call a tool, so a long run ends with a hand-over instead of
  `MaxStepsExceeded` and a tool result nobody read.
- **The plan survives compaction.** The todo list and the ledger are pinned, so a
  run that compresses its context still knows what is left and which number means
  which source.

Modes are declarable everywhere an agent is: `mode:` and `depth:` in a blueprint
or a `SubAgentSpec`, and in a version — `versions={"v2": {"mode": "research",
"depth": "deep"}}` — so a way of working can be evaluated against the old one.
Under governance, an `identity.tools` allowlist needs `"tag:mode"` for the todo
list and the ledger. `modes.register_mode(name, factory)` adds one of your own.

## Sandboxes: somewhere else to work

A workspace is a folder on this machine. A sandbox is a workspace that is its
own machine — and it goes wherever a workspace does:

```python
from agent_harness import Agent, sandbox

Agent("colleague", mode="cowork", workspace=sandbox("docker"))
Agent("colleague", mode="cowork", workspace=sandbox("e2b", template="base"))
Agent("colleague", mode="cowork", workspace="docker://node:22")      # by URL
```

| name | what it is | needs |
|---|---|---|
| `docker`, `podman` | a long-lived container; `runtime="runsc"` for gVisor | the CLI |
| `kubernetes` | a pod, created or attached to | `kubectl` |
| `ssh` | a machine you can already reach | `ssh` |
| `e2b` | a Firecracker micro-VM | `pip install e2b`, `E2B_API_KEY` |
| `daytona` | a sandbox from an image or snapshot | `pip install daytona`, `DAYTONA_API_KEY` |
| `modal` | a container on Modal | `pip install modal`, a Modal token |
| `command` | anything you can put in front of `sh -c` — nsjail, bubblewrap, firejail, `lxc exec` | — |

```python
sandbox("docker", image="python:3.12-slim")                 # no network, 2 CPUs, 2 GB
sandbox("docker", image="node:22", mount="./project", network=True)
sandbox("kubernetes", image="python:3.12-slim", namespace="agents")
sandbox("ssh", host="agent@build-7", key="~/.ssh/agent")
sandbox("daytona", image="python:3.12-slim", network=False)
sandbox("modal", image="python:3.12-slim", cpu=2, memory=4096)
sandbox("e2b", sandbox_id="i1a2b3")                         # one already running
```

Everything that works on a folder works in one, unchanged: the file tools,
`shell`, `run_python`, `parse_document`, the chart and report tools, and cowork
handing back the files it wrote — which are brought down to this machine, so
they outlive the sandbox they were written in.

- **It starts on first use and is stopped for you.** `await harness.aclose()`
  stops every sandbox an agent was given; `async with sandbox(...) as ws:` does
  it sooner. Each one also has a lifetime (`ttl`) enforced by the platform
  itself, so one orphaned by a crash does not run — or bill — for ever. A
  sandbox you attached to by id is left running.
- **Inside one, the shell does not ask.** On this machine `shell` and
  `run_python` are opt-in and ask for approval on every call. In a sandbox they
  are there and they run: a command can only hurt the sandbox, and asking
  before each one would make it pointless. `allow_shell=False` takes the shell
  away; `PolicyGate(ask=["shell"])` puts the question back. `ssh` and `command`
  do not claim to be isolated unless you say `isolated=True`.
- **Docker starts closed.** No network, a CPU, memory and process limit, no
  privilege escalation, and the container removes itself. `mount=` works in a
  folder of yours, as you, so the files it leaves are yours and not root's. A
  Kubernetes pod is created with no service-account token.
- **A sandbox that is not ready says what is missing** — the package to
  install, the key to set. A cowork run finds out at its start, before any model
  call is paid for; any other agent finds out at its first file or shell call.
  To find out ahead of time:

```python
from agent_harness import available_sandboxes

available_sandboxes()      # {'docker': True, 'e2b': False, 'ssh': True, ...}
report = await sandbox("docker").check()
report["ok"], report["steps"]     # start · exec · write and read · list · detect changes · delete
```

```bash
agent-harness sandboxes                              # what is ready, what each needs
agent-harness sandboxes docker://alpine --check      # start one and prove it works
agent-harness run "build it" --mode cowork --sandbox docker --workspace ./project
```

### Picking a conversation back up

A run hands back the two ids that continue it — the chat, and the sandbox it
was working in:

```python
harness = Harness.local(".harness")              # sessions that outlive the process
# or a database: Harness(sessions="postgresql://…") — see "Sessions" below
agent = Agent("colleague", mode="cowork", harness=harness,
              workspace=sandbox("e2b", keep=True))      # keep: do not kill it at the end

result = await agent.run("Build the importer.")
result.session_id, result.sandbox_id             # ("ses_4f1c...", "i1a2b3...")
```

Later — another request, another process — the chat id is enough, because the
session remembers its sandbox:

```python
agent = Agent("colleague", mode="cowork", workspace="e2b",
              harness=Harness.local(".harness"))

await agent.run("Now add the tests.", session="ses_4f1c...")

# or pass both by hand, or follow the conversation from here on:
await agent.run("Now add the tests.", session="ses_4f1c...", sandbox_id="i1a2b3...")
await agent.resume("ses_4f1c...")                # every run after this continues it
```

`sandbox("e2b", id="i1a2b3...")` says the same thing where the sandbox is built;
every backend takes `id=`.

- **`keep=True` is what leaves a sandbox to come back to.** Without it the
  sandbox is destroyed when the harness closes. One that was picked up is left
  as it was found; `await agent.workspace.destroy()` ends it when the
  conversation is over. A kept Docker container that has since stopped is
  started again with its files in it; a Daytona sandbox stopped for idleness
  likewise; E2B and Modal sandboxes last until their `ttl`, which picking one
  up renews on E2B.
- **If the sandbox is gone, the conversation still continues.** A new one is
  started, the text files the conversation is known to have written are put
  back, and the model is told — in the conversation — which sandbox was lost,
  what came back and what did not. `on_missing="error"` refuses instead.
  The record kept with the session is for this, not a backup: it holds what
  cowork handed back, up to 2 MB of text, and names anything larger or binary.
- **One agent, one sandbox.** An agent already working in one sandbox refuses a
  conversation whose files are in another, rather than mixing two people's
  work. In a server, build an agent per conversation — it is cheap — on a shared
  harness.
- A local `Workspace("./project")` needs none of this: the files never left.

### One for every agent

To give every agent and sub-agent its own, set it once on the harness. A
sub-agent whose spec says `workspace: isolated` then gets a sandbox to itself:

```python
harness = Harness(workspaces=WorkspaceBroker(sandbox="e2b", template="base"))
```

In a blueprint it is `workspace: docker`, or
`workspace: {sandbox: daytona, image: python:3.12-slim}`.

**Your own sandbox is one method.** Reading, writing, listing and noticing what
changed are all derived from `exec`, in shell that runs on busybox as well as
GNU, so a new backend works the day it can run a command:

```python
from agent_harness import ExecResult, Sandbox, register_sandbox

class MySandbox(Sandbox):
    name = "mine"

    async def _start(self):                       # optional
        self.vm = await my_platform.create()

    async def _exec(self, command, *, cwd, env, timeout):
        done = await self.vm.run(self.script(command, cwd=cwd, env=env), timeout)
        return ExecResult(done.code, done.stdout, done.stderr)

    async def _stop(self):                        # optional
        await self.vm.destroy()

register_sandbox("mine", "myapp.sandboxes", "MySandbox")
Agent("colleague", mode="cowork", workspace="mine")
```

Docker is tested against a real container. E2B, Daytona and Modal are driven
through their own SDKs and tested against stand-ins for them, and Kubernetes and
SSH against a recorded command line — so before relying on one, run
`agent-harness sandboxes <name> --check` with your credentials.

## Sessions: where chats are kept

A chat id is a row in a store. By default that store is in memory and gone with
the process; `Harness.local()` keeps a JSON file per chat. For anything with more
than one replica, name a database:

```python
from agent_harness import Agent, Harness, session_provider

harness = Harness(sessions="postgresql://user:pass@host/agents")
harness = Harness(sessions=session_provider("redis://host:6379/0", ttl=30 * 86_400))
harness = Harness.on("postgresql://user:pass@host/agents")    # memory and chats, one pool
harness = Harness.local(".harness", sessions="azure://chats")  # chats in a storage account
```

| store | URL | install |
|---|---|---|
| SQLite | `sqlite:///chats.db` | — |
| PostgreSQL | `postgresql://…` | `asyncpg` or `psycopg` |
| MySQL / MariaDB | `mysql://…` | `aiomysql` |
| MongoDB (and Cosmos DB's Mongo API) | `mongodb://…` | `motor` or `pymongo` |
| Redis | `redis://…` | `redis` |
| DynamoDB | `dynamodb://table` | `boto3` |
| Amazon S3 | `s3://bucket/prefix` | `boto3` |
| Azure Blob — a storage account | `azure://container` | `azure-storage-blob` |
| Google Cloud Storage | `gs://bucket/prefix` | `google-cloud-storage` |

The URLs and drivers are the memory backends' own, so a database is configured
in one place. Every store keeps a session under the same three promises:

**It knows whose it is.** A session is stamped with the user and tenant the agent
was acting for (`Agent(trace={"user_id": ..., "tenant_id": ...})`), and those
are indexed columns, not buried in a blob:

```python
chats = await harness.sessions.list(user_id="ada", tenant_id="acme", limit=20)
[c.summary() for c in chats]        # id, title, messages, updated, cost — for a sidebar
```

An agent acting for someone else is refused the session in the same words as for
an id that never existed — `no session 'ses_…'` — so ids cannot be probed, and
the refusal is in the audit trail.

**Two requests cannot overwrite each other.** A session carries a version, and a
store accepts a save only from the version it holds — one atomic statement in
SQL, a conditional write in MongoDB and DynamoDB, a Lua script in Redis. When a
second request on the same chat finishes after the first, its turn is *added* to
what is there; nobody's message is lost. If that is not possible — the run
compacted its history along the way — the run is kept as a fork and
`result.warnings` says which session it ended up in.

**A store that is down loses the record, not the answer.** If the save fails,
the result still comes back, with the failure in `result.warnings` and the audit
trail.

To find out whether a database is wired up correctly before the first real
conversation:

```python
report = await harness.sessions.check()
report["ok"], report["steps"]   # save · load · list by owner · refuse a stale save · delete
```

```bash
agent-harness sessions --store postgresql://… --check
agent-harness sessions --store postgresql://… --user ada --tenant acme
agent-harness run "carry on" --sessions postgresql://… --user ada --session ses_4f1c
```

Three things worth knowing:

- **DynamoDB** limits an item to 400 KB, so a conversation is stored compressed
  and in pieces behind one small head item; the conditional write on that head
  is what makes a save all-or-nothing.
- **Object storage** (S3, Azure Blob, GCS) is the cheap, durable place for chats
  kept for years. Its version check reads before it writes, which stops the
  ordinary double-submit but is not atomic; keep live, concurrent chats in a
  database and archive to a bucket.
- **Your own store** is a `SessionStore` — `save`, `load`, `list`, `delete` — or
  a `DurableSessionStore`, which gives you the versioning for four smaller
  methods. `check()` will tell you if it holds the line.

PostgreSQL, MariaDB, MongoDB, Redis, DynamoDB (DynamoDB Local) and Azure Blob
(Azurite) have each been run against a real server through the whole contract,
including a race of eight saves from one version. S3 and GCS share the Azure
code path and have not; `tests/test_sessions.py` lists the environment variables
that run the suite against yours.

## Attachments: images, files, audio, video

Hand an agent whatever you have. Each thing reaches the model in the form the
model can take:

```python
result = await agent.run(
    "What does the contract say about notice, and does the call agree?",
    attachments=["contract.pdf", "call.mp3", "whiteboard.jpg", "figures.xlsx"],
)
```

| | Claude | GPT | Gemini | others (Groq, Ollama, …) |
|---|---|---|---|---|
| image | as an image | as an image | as an image | as an image |
| PDF | as the PDF | as the PDF | as the PDF | read here, sent as text |
| Word, CSV, Markdown, JSON, … | read here, sent as text | same | same | same |
| audio | transcribed, sent as text | audio models hear it; others get a transcript | hears it | transcribed |
| video | — | — | watches it | — |

- **Native where the model can, text where it cannot.** A file the model does
  not read is read here with `parse_document`; a recording it cannot hear is
  transcribed with whatever you gave the harness — `Harness(speech=OpenAISpeech())`.
  So the same line works on every provider.
- **It says so before it spends anything.** A video for a model that cannot
  watch, a recording with nothing to transcribe it, a file that is not there or
  is over 20 MB: the run ends with the reason, and no model call is made.
- **Text made from an attachment is input like any other.** It passes the same
  guardrails as the task — a document is the most common place for an injected
  instruction to hide.
- **A file is pointed at, not copied.** `attach("report.pdf")` keeps the path
  and reads it when the message is sent, so a saved session holds a reference
  rather than megabytes of base64. Bytes that exist only in memory —
  `attach(data, "audio/wav")` — do travel with it.
- **Nothing a model cannot take is ever sent as its bytes.** If the file has
  gone, or the model changed, the model is told in a line that an attachment was
  not sent.

```python
from agent_harness import Message, attach

Message.user("Compare these.", attachments=["before.png", "after.png"])
attach("https://example.com/clip.mp4")           # a URL the provider fetches
provider.accepts(attach("call.mp3"), "gpt-4.1")  # False: it will be transcribed
```

Not yet: tools that *return* images, and uploads past 20 MB through a
provider's file API.

## Voice: agents you talk to

```python
from agent_harness import Agent
from agent_harness.voice import OpenAISpeech, RealtimeAgent, VoiceAgent

agent = Agent("concierge", "Help callers with their bookings.", mode="voice",
              tools=[find_booking])

voice = VoiceAgent(agent, speech=OpenAISpeech())        # listen · think · speak
voice = RealtimeAgent(agent, provider="openai")         # one speech-to-speech model

async for event in voice.run(microphone()):             # 16-bit PCM in
    if event.type == "audio":
        speaker.write(event.audio)                      # 16-bit PCM out
    elif event.type == "interrupted":
        speaker.flush()                                 # they spoke over it
```

Two ways to build it, used the same way:

| | `VoiceAgent` | `RealtimeAgent` |
|---|---|---|
| how | speech → text → your agent → speech | audio straight into a realtime model |
| models | any of the twenty providers | OpenAI Realtime, Gemini Live |
| time to first sound | a transcription and a first sentence | the lowest there is |
| budgets, memory, audit, tools | all of them, unchanged | tools run under every rail |
| guardrails on what is said | before it is spoken | on the transcript, after |
| turn-taking | here: `EnergyVAD`, or your own | the model's own |

`mode="voice"` makes the agent write for the ear — short sentences, no markdown,
a few words before a tool call so a lookup is not a silence.

What makes it a conversation rather than a queue:

- **It starts speaking before it has finished thinking.** The answer is cut into
  sentences as the model writes them — the first at a clause, sooner still — and
  each is synthesised while the next is being written.
- **It stops when you speak.** Speech over an answer cancels the rest, and the
  conversation keeps only what was actually said aloud: the agent is not left
  believing it told you something you never heard.
- **It waits for you to finish.** If you pause and carry on before anything has
  been said back, the two halves are transcribed and answered as one turn.
- **It knows when someone is speaking.** `EnergyVAD` measures speech against the
  room — the quietest tenth of the last few seconds, so a fan or a hum is the
  room, not a voice. `silence_ms` is the dial between cutting people off and
  awkward pauses. A neural detector (Silero, WebRTC) plugs in where it goes.
- **Every turn is timed.** `turn_end` carries `stt_ms`, `first_token_ms` and
  `first_audio_ms`; `voice.latency` and `harness.health` keep them.

```python
VoiceAgent(agent, speech=OpenAISpeech(voice="marin", language="en"),
           vad=EnergyVAD(16_000, silence_ms=400),
           greeting="Hello, how can I help?", tool_filler="One moment.",
           output_rate=8000)                          # a phone line

RealtimeAgent(agent, OpenAIRealtime(voice="cedar", turn_detection="semantic_vad"))
RealtimeAgent(agent, "gemini", rate=16_000, output_rate=8000)
```

Speech itself is a small contract — `transcribe(audio)` and `synthesize(text)` —
so Deepgram, ElevenLabs or a model of your own goes where `OpenAISpeech` does.
`OpenAISpeech(base_url=...)` covers Groq's Whisper, Azure OpenAI and local
servers. `voice.ulaw_decode` / `ulaw_encode` carry a phone line's audio.

It is the same agent underneath. The conversation is a session — owned, stored
and resumable like any other (`VoiceAgent(agent, session="ses_…")`). The budget
still stops it, and says so aloud. And where the audio goes is put to the same
`model_egress` check as a model call, so a residency policy covers the voice too.

```bash
agent-harness voice                                   # the microphone (pip install sounddevice)
agent-harness voice --input question.wav --output answer.wav
agent-harness voice --realtime openai
```

Tested without a network: the turn-taking, interruption and latency paths run
against scripted speech; the WebSocket client against a server written for the
tests and against the `websockets` library; OpenAI Realtime and Gemini Live
against scripted sockets that speak their documented protocols. **None of it has
been run against the live OpenAI or Google services** — do that with your own
key before you put a caller on it. Echo cancellation is the client's job: played
through an open speaker, an agent will hear itself.

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
| `web_search`, `make_search_tool`, `WebSearch` | web search over Tavily, Brave, Exa, Serper, Google, SearXNG or keyless DuckDuckGo — retries, fallback engines, domain policy, cache. [More below](#web-search) |
| `make_fetch_tool`, `make_http_tool` | domain allowlist, private-address refusal, HTML stripping |
| `KnowledgeBase(...).as_tool()` | documents searched by meaning, in any of seventeen vector stores. [More below](#knowledge-bases-and-vector-stores) |
| `ImageGenerator().tools()` | image generation with reference images — Gemini, OpenAI, Replicate — as jobs with progress, retries and fallback. [More below](#image-generation) |
| `Browser().tools()`, `computer_tool` | a real browser the agent drives by numbered elements, and a mouse and keyboard for a model that sees. [More below](#browser-and-computer-use) |
| `parse_document` | text, Markdown, CSV, TSV, JSON, JSONL, HTML, XML with no dependencies; PDF, DOCX and OCR with an optional install each |
| `bar_chart`, `line_chart`, `render_report` | inline SVG that works in light and dark, plus markdown reports |
| `make_python_tool` | run code in the workspace — asks for approval every time, unless the workspace is a sandbox |

A workspace brings `fs_read`, `fs_write`, `fs_list`, `fs_delete` and — only when
you ask for it — `shell`. A sandbox is a workspace too, and comes with its shell.

## Web search

```python
from agent_harness import Agent
from agent_harness.toolkits import http_fetch, web_search

analyst = Agent("analyst", mode="research", tools=[web_search, http_fetch])
```

`web_search` works with any model, because the search is made here and not by
the model's vendor. It uses whichever engine has a key in the environment, and
DuckDuckGo — which needs none — when no key is set:

| Engine | Set | Notes |
|---|---|---|
| `tavily` | `TAVILY_API_KEY` | domain filters applied by the engine |
| `brave` | `BRAVE_API_KEY` | |
| `exa` | `EXA_API_KEY` | domain filters applied by the engine |
| `serper` | `SERPER_API_KEY` | Google results |
| `google` | `GOOGLE_SEARCH_API_KEY` + `GOOGLE_CSE_ID` | Programmable Search Engine |
| `searxng` | `SEARXNG_URL` | an instance of your own, with the JSON format on |
| `duckduckgo` | nothing | the HTML results page: fine to start with, throttled under load |

The model gets four arguments — `query`, `limit`, `recency` (`day`, `week`,
`month`, `year`) and `domains` — and reads back title, URL, snippet and the
date when the engine knows it. To choose the engines and the policy yourself:

```python
from agent_harness import WebSearch

search = WebSearch(
    ["brave", "duckduckgo"],              # asked in order; the next when one fails
    allowed_domains=["*.gov", "europa.eu"],   # a plain name covers its subdomains
    blocked_domains=["pinterest.com"],
    limit=5, max_limit=10, region="gb", language="en", safe_search="strict",
)
hits = await search.search("heat pump subsidy", recency="year")   # call it yourself
agent = Agent("analyst", tools=[search.as_tool()])

search.last_engine      # 'brave'
search.stats            # {'brave': {'answered': 1, 'failed': 0, 'last_error': ''}}
```

What happens when a search goes wrong is decided here, not left to the model:

- **Retries.** A 429, a 5xx, a timeout or a dropped connection is tried again
  with back-off, and `Retry-After` is honoured up to ten seconds. A refused key,
  an exhausted quota or a rejected query is not retried.
- **Fallback.** An engine that cannot answer — or, unless
  `fallback_on_empty=False`, finds nothing — is passed over for the next one.
- **Circuit breaker.** After `failure_threshold=3` failures in a row an engine
  is left alone for `cooldown=60` seconds, then probed once.
- **Time.** `timeout=15` seconds a request, `deadline=45` seconds the whole
  search, retries and fallbacks included.
- **Results.** Markup and entities removed, tracking parameters stripped, the
  same page listed once, snippets cut at `snippet_chars`. The domain policy is
  applied to what comes back, whatever the engine was asked — and the model can
  narrow a search to some domains but not reach past `allowed_domains`.
- **Cache.** The same question within `cache_ttl=300` seconds is answered from
  memory, and two identical questions asked at once are one request.
- **Errors.** When every engine fails the model reads which and why —
  `web search failed — brave: out of quota; duckduckgo: timed out` — and no key
  ever appears in that message.

An engine that is not on the list is a function, sync or async, or a
`SearchEngine` subclass when it is an HTTP service:

```python
from agent_harness.toolkits import SearchQuery

async def intranet(query: SearchQuery):
    rows = await wiki.find(query.text, top=query.limit)
    return [{"title": r.name, "url": r.link, "snippet": r.summary} for r in rows]

search = WebSearch([intranet, "tavily"])
```

In a blueprint the tool is `agent_harness.toolkits:web_search`, with
`AGENT_HARNESS_SEARCH=brave,duckduckgo` choosing the engines. `agent-harness
search` lists the engines and which are set up; `agent-harness search "heat
pump subsidy" --recency year` runs one. Snippets are text from the open web:
treat them as you treat any fetched page, as something the agent reads and not
something it obeys. `examples/20_web_search.py` runs all of this offline.

## Browser and computer use

```python
from agent_harness import Agent, Browser

async with Browser() as browser:
    agent = Agent("shopper", "Find things on the web.", tools=browser.tools())
    await agent.run("What is the top story on Hacker News right now?")
```

The browser is Chrome, Chromium or Edge — whichever is installed — started
headless and driven over the DevTools protocol by the harness itself. Nothing
is added to install: no Playwright, no driver. `Browser(cdp_url="http://host:9222")`
drives one that is already running somewhere else — a container, another
machine, a hosted browser — and `headless=False` shows the window.

The agent is not handed pixels to guess at. After every action it reads the
page as the things that can be acted on, each with a number, and acts by number:

```text
Page: Sign in — https://shop.example/login

[1] textbox "Email"
[2] textbox "Password"
[3] combobox "Country" value="France"  options: France | Germany
[4] checkbox "Remember me"
[5] button "Sign in"
[6] link "Forgot your password?" → /reset

Text on screen:
Sign in to see your orders. …
```

So any model can browse, with or without vision. The tools:

| Tool | Does |
|---|---|
| `browser_navigate(url)` | open a page |
| `browser_click(ref)` · `browser_type(ref, text, submit)` · `browser_select(ref, option)` | act on an element by its number |
| `browser_press(key)` · `browser_scroll(direction, amount)` · `browser_back()` | keys (`Enter`, `ctrl+a`), scrolling, history |
| `browser_snapshot()` · `browser_read(offset, max_chars)` | read the page again; read all of its text |
| `browser_wait(seconds, text)` | wait, or wait until some text appears |
| `browser_tabs(action, index, url)` | list, switch, open, close |
| `browser_screenshot()` | a picture of the page, for a model that sees |

`browser.tools(vision=True)` sends a screenshot back with every action;
`permission="ask"` has each one approved first. Every method is yours to call
too — `await browser.goto(url)`, `.click(3)`, `.type(1, "ada@example.com")`,
`.screenshot()`.

```python
Browser(
    allowed_domains=["shop.example", "*.gov"],   # a plain name covers its subdomains
    blocked_domains=["ads.example"],
    allow_private=False,          # no localhost, no 10.x, no cloud metadata address
    viewport=(1280, 800), timeout=30, accept_dialogs=False,
    user_data_dir="./profile",    # keep cookies and logins; a fresh profile otherwise
)
```

What is taken care of, so the model does not have to:

- **Where it may go.** The policy is applied to every document a tab asks for —
  a typed address, a clicked link, a redirect, a frame — and names are resolved
  before a private address is ruled out. Only `http` and `https` open: no
  `file:`, no `chrome:`. Downloads are refused.
- **Pages that move.** After each action the page is given time to load and is
  read again, so the answer to a click is what the click led to. Numbers stay
  with their elements for as long as the page lives; a number from a page that
  has gone is refused with "read the page again", not guessed at.
- **What gets in the way.** A link that opens a new tab is followed there; an
  `alert` is accepted and a `confirm` dismissed (or accepted, with
  `accept_dialogs=True`), and the model is told what it said; an element under
  an overlay is clicked directly; elements inside shadow roots are listed.
- **What breaks.** A browser that has died is started again on the next call, a
  crashed tab is reloaded, and both are reported. The browser is closed with
  its `Browser`, and never outlives the process that started it.
- **Secrets.** What is typed into a password field is not echoed back.

### Computer use

For what a list of elements cannot reach — a canvas, a map, a game, a desktop
application — there is one tool, `computer`, with a mouse, a keyboard and a
screenshot after every action. It needs a model that sees; any such model will
do, because it is an ordinary tool and not a vendor's.

```python
from agent_harness import Browser, DesktopComputer, computer_tool

agent = Agent("operator", tools=[computer_tool(browser.computer())])       # a browser page
agent = Agent("operator", tools=[computer_tool(DesktopComputer(sandbox))])  # a Linux desktop
```

Its actions are `screenshot`, `click`, `double_click`, `right_click`, `move`,
`drag`, `type`, `key`, `scroll`, `wait` — and `open` on a browser. Coordinates
are pixels of the screenshot, and a point off the screen is refused.

`DesktopComputer` drives an X11 desktop with `xdotool` and takes its pictures
with `scrot`: give it a sandbox whose image has a display (`Xvfb`), those two
and the applications you want used. With no sandbox it drives *this* machine's
desktop, and the tool then asks before every action. A screen that is neither —
VNC, a phone — is a subclass of `Computer`.

Any tool can return an image this way: `return ["what happened",
ImageBlock.from_bytes(png)]` and the model is shown it. A conversation keeps the
newest three (`agent.tool_images_kept`); older ones leave a note in their place,
so a long session does not fill with screenshots.

A web page is text from strangers. An agent that browses will read instructions
that are not yours; keep it to the domains it needs, and put `permission="ask"`,
a guardrail or a human in front of anything it can change.
`examples/21_browser.py` runs a browser against a shop served on the spot.

## Image generation

```python
from agent_harness import Agent, ImageGenerator

images = ImageGenerator()                      # the engine whose key is set
agent = Agent("designer", "Make what is asked for, look at it, improve it.",
              tools=images.tools(), workspace="./studio")
await agent.run("A poster for the autumn sale, in the style of this photo.",
                attachments=["storefront.jpg"])
```

| Engine | Set | Default model | References |
|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash-image` — or `gemini-3-pro-image-preview`, or an `imagen-…` model | up to 14 (none for Imagen) |
| `openai` | `OPENAI_API_KEY` | `gpt-image-1`; `base_url=` for anything that speaks the same API | up to 16 |
| `replicate` | `REPLICATE_API_TOKEN` | `black-forest-labs/flux-kontext-pro`; any hosted model by name | as the model takes them |

All three are spoken to over HTTP. Name one, or several to fall back through:

```python
from agent_harness.toolkits import GeminiImages, ReplicateImages

images = ImageGenerator("gemini", model="gemini-3-pro-image-preview")
images = ImageGenerator([GeminiImages(), "openai"], aspect_ratio="16:9",
                        cost_per_image=0.04, max_images=50)
images = ImageGenerator(ReplicateImages(model="google/nano-banana",
                                        reference_key="image_input", reference_list=True))

made = await images.generate("a red fox in snow, 35mm photograph",
                             references=["fox.jpg"], count=2)
made[0].save("fox.png")           # .data, .media_type, .width, .height, .engine, .model
```

**Reference images** are what a picture is made from, or made to look like: a
photo to edit, a style to follow, a character to keep the likeness of. In your
own code a reference is a path, a URL, bytes, an `ImageBlock`, or an image made
earlier. The agent names one by what it can see — a file in its workspace, an
image it generated earlier (`"images/poster.png"`), an attachment that came
with the task (`"attachment:1"`), an http(s) address — and never by a path on
this machine. `images.tools(references=[...])` gives every generation the same
ones: a brand's style, a character sheet. Each reference is checked to be an
image, by its bytes and not its name, before it is sent anywhere.

**Progress.** Every generation is a job:

```python
job = await images.submit("a city at dusk", count=4)
while not job.done:
    print(job.describe())         # running on replicate — 40%, 12s (processing)
    await asyncio.sleep(2)
made = job.result()               # or: await job.wait(60), job.cancel()

ImageGenerator(on_progress=lambda p: print(p.status, p.fraction, p.message))
```

The agent has the same. `generate_image` waits up to `wait=120` seconds; a job
that outlives the wait comes back as a job id, and `image_status` checks on it,
waits for it, collects it or cancels it — so a slow model does not hold a run
hostage, and the agent is told not to start the same image twice. Replicate
reports how far along it is; Gemini and OpenAI answer in one piece, so their
progress is the stage and the time.

What the agent gets back is where the image went, its size, and — with
`show=True`, the default — the image itself, so a model that sees can judge it
and try again. Images are written to the agent's workspace under `images/`
(or `output_dir=` when there is none) and listed in `result.artifacts`.

When it goes wrong:

- **Retried**: 429, 5xx, timeouts, dropped connections — with back-off, honouring
  `Retry-After` — and a model that answered with words and no picture.
- **Passed to the next engine**: a refused key, an exhausted quota, an engine
  that cannot take the references given. One that keeps failing is left alone
  for a minute.
- **Not retried and not passed on**: a prompt the provider *refused*. The agent
  is told it was refused and why; another provider is not a way round that.
- **Bounded**: `timeout=180` seconds an attempt, `deadline=600` the whole job,
  `max_count=4` images a request, `max_images=` in total, `max_concurrency=3` at
  once. A job that is cancelled or runs out of time is cancelled at the provider.
- **Checked**: what comes back must be an image; an error page is a failure.
- **Counted**: `cost_per_image=` is charged to the run's budget, and stops a run
  like any other spend. Failures name the engine and never the key.

`FakeImages` draws without a key or a network, for tests; an engine of your own
is a function of an `ImageRequest` or an `ImageEngine` subclass.
`agent-harness image` lists the engines, `agent-harness image "a red fox"
--reference fox.jpg --aspect 16:9` makes one, and `--images` gives the tools to
`run` and `chat`. `examples/22_images.py` runs all of it offline.

## OpenAPI → tools

Hand an agent an API by handing it the API's document. Every operation in an
`openapi.json` becomes a tool, with the arguments the document describes.

```python
from agent_harness import Agent, openapi_tools

api = openapi_tools("openapi.json", token=os.environ["SHOP_TOKEN"])
agent = Agent("support", "Help the customer with their orders.", tools=api)

api.names                                     # ['listOrders', 'getOrder', 'refundOrder']
print(api.describe())                         # each tool, its request, its arguments
await api.call("getOrder", orderId="4182")    # call one yourself, to see it is wired up
```

The document can be a file (JSON or YAML), a URL, the text itself, or a mapping
you already loaded. OpenAPI 3.0 and 3.1 are read as they are, and Swagger 2.0 too.

**What a tool looks like.** Its name is the `operationId` (or the method and
path, when there is none). Its arguments are the operation's own: path, query and
header parameters, and — for a JSON or form body that is an object — one argument
per field, which is what a model fills in best. Any other body is a single `body`
argument. Types, enums, defaults and descriptions come from the document with
`$ref` followed and `allOf` merged; fields the server sets (`readOnly`) are left
out. A schema that refers to itself ends rather than going round for ever.

**What is sent.** Each argument goes where the document says — into the path
(escaped, so a value cannot become another path), the query (with the document's
`style` and `explode`), a header, or the body as JSON, a form, or text. JSON
comes back as JSON; a `4xx` or `5xx` comes back to the model as an error it can
read and act on, with the status and the body's message.

**Credentials are yours, never the model's.** They are added to the request and
appear in no tool's schema:

```python
openapi_tools(spec, token="…")                 # Authorization: Bearer …
openapi_tools(spec, token=fetch_token)         # a function: called per request, to renew
openapi_tools(spec, api_key="…")               # wherever the document's apiKey scheme says
openapi_tools(spec, basic=("user", "pass"))
openapi_tools(spec, credentials={"partnerKey": "…"})   # by security-scheme name
openapi_tools(spec, headers={"X-Tenant": "acme"}, params={"version": "2"})
```

A parameter you fix with `headers=` or `params=` is not offered to the model.
Redirects are not followed, so a credential never goes to wherever a response
points.

**Choosing what the agent gets.** A large API is hundreds of tools; give an
agent the ones its job needs.

```python
openapi_tools(spec, include=["getOrder", "list*", "POST /orders/*/refunds"])
openapi_tools(spec, exclude=["delete*"], tags=["orders"], methods=["get"])
openapi_tools(spec, writes="ask")       # anything but GET/HEAD/OPTIONS needs an approver
openapi_tools(spec, prefix="shop_")     # keep two APIs' tools apart
openapi_tools(spec, base_url="https://staging.shop.example/v1", cache_reads=True)
```

Tools are tagged `openapi`, the API's name, and `read` or `write`. They pass the
same permission gate, hooks, guardrails, audit trail and budget as any other
tool. Reads are retried on a dropped connection or a `429`/`502`/`503`/`504`;
writes never are. `api.skipped` lists what could not be made a tool, and why —
deprecated operations (`deprecated=True` includes them) and file uploads.

In a blueprint or a workflow file an API is declared by name, with its secret
named rather than written:

```yaml
openapi:
  shop:
    spec: ./shop.openapi.json        # or a URL
    token_env: SHOP_TOKEN
    include: [getOrder, listOrders, refundOrder]
    writes: ask
agents:
  support: {instructions: Help with orders., tools: [getOrder, listOrders]}
```

```bash
agent-harness openapi openapi.json                       # the tools it becomes
agent-harness openapi openapi.json --call getOrder --arg orderId=4182 --token …
agent-harness run "Where is order 4182?" --openapi openapi.json
```

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

Past about fifty thousand records, keep memory's index in a vector database
instead of in the process — any of the stores below:

```python
from agent_harness.knowledge import vector_store

memory_store = SemanticMemory(PostgresMemory(dsn), embedder=embedder,
                              index=vector_store("pgvector://user:pw@host/db?table=memory"))
```

Each record is stored with its scope and its owner, and a search is held to
them in the database — one tenant's memory is never fetched for another.

## Knowledge bases and vector stores

```python
from agent_harness import Agent, KnowledgeBase, OpenAIProvider
from agent_harness.memory import ProviderEmbedder

kb = KnowledgeBase("qdrant://localhost:6333/handbook",
                   embedder=ProviderEmbedder(OpenAIProvider(), "text-embedding-3-small"))
await kb.add(path="handbook/refunds.md", metadata={"team": "support"})
await kb.add(url="https://example.com/terms")
await kb.add("Orders ship within two working days.", id="shipping", title="Shipping")

passages = await kb.search("how long do refunds take?", k=5, filter={"team": "support"})
agent = Agent("support", tools=[kb.as_tool(filter={"team": "support"})])
```

**Documents in.** A document is read (`parse_document`: text, Markdown, HTML,
CSV, JSON, and PDF or DOCX with their optional installs), cut into overlapping
passages at paragraph and sentence boundaries — each one labelled with the
headings it sits under — embedded in batches, and stored under its id. Adding it
again replaces it; when nothing has changed, one embedding is spent finding
that out and nothing is written. A shorter version leaves no stale passages
behind, and `kb.delete(id)` removes all of them.

**Passages out.** A search embeds the question, takes more candidates than
were asked for, and orders them by meaning *and* by the words they share with
the question, so a product code or a name is not lost to a near-synonym.
`reranker=` puts a cross-encoder or a rerank API over the candidates;
`min_score=` drops what is not close; `namespace=` keeps several knowledge
bases apart in one store. The tool an agent gets returns each passage with the
document it came from, and `as_tool(filter=...)` holds that agent to the
documents it may see — a filter the model cannot remove.

**The store** is a name or a URL. One interface, one filter language, and
scores that are cosine similarity on every one of them:

| Store | Name / URL | Needs |
|---|---|---|
| In memory · SQLite file | `memory` · `sqlite:///knowledge.db` | — |
| Qdrant (and Qdrant Cloud) | `qdrant://host:6333/collection` | — |
| Chroma (and Chroma Cloud) | `chroma://host:8000/collection` | — |
| Weaviate (and Weaviate Cloud) | `weaviate://host:8080/Collection` | — |
| Milvus · Zilliz Cloud | `milvus://host:19530/collection` | — |
| Pinecone | `pinecone://index-name` | — |
| OpenSearch · Amazon OpenSearch Service and Serverless | `opensearch://host:9200/index` | — (SigV4 built in) |
| Elasticsearch · Elastic Cloud | `elasticsearch://host:9200/index` | — |
| PostgreSQL + pgvector · Supabase, Neon, Aurora, AlloyDB, Azure Postgres | `pgvector://user:pw@host/db?table=t` | `asyncpg` |
| Redis 8 / Redis Stack · MemoryDB, Azure Managed Redis | `redis://host:6379?index=name` | `redis` |
| MongoDB Atlas Vector Search | `mongodb+srv://…?database=d&collection=c` | `pymongo` |
| Azure AI Search | `azure-search://service.search.windows.net/index` | — |
| Cloudflare Vectorize | `vectorize://account-id/index` | — |
| Upstash Vector | `upstash://host` | — |
| Vertex AI Vector Search | `VertexVectorStore(project=…, index=…, endpoint=…)` | — |
| Amazon S3 Vectors | `s3vectors://bucket/index?region=…` | — |
| Amazon Bedrock Knowledge Bases | `KnowledgeBase(retriever=BedrockKnowledgeBase(id))` | — |

Add `+https` for TLS (`qdrant+https://…`), and pass keys as arguments or let
them come from the environment (`QDRANT_API_KEY`, `PINECONE_API_KEY`,
`AZURE_SEARCH_API_KEY`, `CLOUDFLARE_API_TOKEN`, the usual AWS and Google
variables). All but three are spoken to over plain HTTP, so they add nothing to
install.

```python
from agent_harness.knowledge import VectorRecord, vector_store

store = vector_store("opensearch+https://search-x.eu-west-1.es.amazonaws.com/handbook",
                     aws_region="eu-west-1")
await store.ensure(1536)                                  # the index, if it is not there
await store.upsert([VectorRecord("a", vector, "the text", {"source": "faq.md", "year": 2026})])
hits = await store.query(vector, k=5, filter={"source": "faq.md", "year": {"$gte": 2025}})
await store.delete(filter={"source": "faq.md"})
print(await store.check())                                # prove it works, end to end
```

Filters are `{"field": value}`, `$in`, `$ne`, `$gt`/`$gte`/`$lt`/`$lte`, and
keys are ANDed; each is translated into the database's own. Ids are any string:
a store that only takes UUIDs is given one derived from yours. Requests that
may succeed later — 429, 5xx, timeouts — are retried with back-off, and keys
never appear in an error.

What differs between stores is said where it matters:

- Redis, MongoDB Atlas, Azure AI Search and Vectorize filter only on fields
  their index was told about: `filterable={"team": "tag"}`. The fields a
  knowledge base and memory use themselves are declared for you.
- Vertex AI Vector Search stores no text, so text and metadata go to a
  `payloads` side store (a SQLite file by default); its index is created and
  deployed in Google Cloud, not here.
- Pinecone, Vectorize, S3 Vectors and OpenSearch Serverless show a write a
  moment after it is made. Vectorize returns at most 20 results, S3 Vectors 30.
- A Bedrock knowledge base is a retriever: Bedrock does the chunking and the
  embedding, and it is asked in words.
- SageMaker hosts models and is not a vector store; on AWS the stores are
  OpenSearch, S3 Vectors, Aurora/RDS (pgvector) and MemoryDB (Redis).

`agent-harness knowledge stores` lists them, `knowledge check --store URL`
proves one works, `knowledge add FILE… --store URL` and `knowledge search`
do what they say, and `run … --knowledge URL` gives an agent the search tool.
`tests/test_knowledge_live.py` runs every store you point it at through the
same checks. `examples/24_knowledge.py` runs on a SQLite file with no key.

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

## Handoffs: another agent takes over

A sub-agent is given a task and reports back to the agent that asked. A handoff
gives the **conversation** away: the other agent sees what was said, answers the
user itself, and is still the one answering on the next turn.

```python
from agent_harness import Agent, Handoff

billing = Agent("billing", "Handle charges, refunds and invoices.",
                description="Charges, refunds and invoices.", tools=[issue_refund])
triage = Agent("triage", "Work out what the customer needs.", mode="chat",
               handoffs=[billing])
billing.add_handoff(triage)                    # and back again

result = await triage.run("I was charged twice for order 4182.")
result.agent           # "billing" — who answered
result.handoffs        # [triage → billing: "charged twice, order 4182"]
result.active_agent    # "billing" — who gets the next turn

await triage.run("When will I see the money?")     # billing answers; triage is not asked
```

A `handoff` tool appears on any agent with somewhere to hand off to, listing
each agent and its `description` — that is what the model chooses by. You keep
calling the agent you started with; the session remembers who has the
conversation, so another process picking the chat up by its id finds the same
agent holding it.

What the agent taking over sees is up to the `Handoff`:

```python
Handoff(billing)                         # everything, tool calls and results included
Handoff(billing, history="text")         # what was said, not what was looked up
Handoff(billing, history="fresh")        # only the user's last message
Handoff(billing, history=my_filter)      # your own: list[Message] in, list[Message] out
Handoff(billing, sticky=False)           # answers this turn; the next goes back
Handoff(billing, description="Anything about money.", on_handoff=notify)
```

A narrowed history is the conversation from then on — it is what gets saved —
which is the point of narrowing it. `on_handoff` is called with the
`HandoffRecord` before the other agent starts, and may raise to refuse.

The rules around it:

- **One run, one result.** A stream has one `run_start`, one `run_end`, and a
  `handoff` event where the conversation changed hands; events carry the name of
  the agent they came from. Usage, steps, tool calls and artefacts from every
  agent are on the one `RunResult`.
- **One session.** Every agent in the chain writes to the same session, in the
  store of the agent you called — even if they were built on different harnesses.
- **It cannot go round for ever.** `Agent(max_handoffs=5)` is the ceiling for one
  run; past it the tool says so and the agent holding the conversation answers.
  Two handoffs asked for in one turn: the first is granted.
- **A refusal is something the model reads.** A `handoff` hook can block it
  (`ctx.block("billing is closed")`), and so can the permission gate, an agent's
  `forbid_tools`, and governance — which asks of a handoff what it asks of a
  delegation, and refuses an agent it has never registered. The agent then
  answers the user itself.
- **The agent handing off gives no answer**, so its output contract, its mode's
  requirements and its completion guardrails are not asked of it. The agent that
  answers is held to its own.
- **Nobody is left holding a conversation they cannot have.** An agent that
  could not start, is no longer in the configuration, or is acting for a
  different user gives the conversation back to the agent you called.
- **The details that would break a provider are handled.** An agent with no tools
  is given the conversation as text; reasoning blocks are dropped when the model
  changes; a research or cowork agent taking over from one in the same mode
  carries on its todo list and its numbered sources.
- **Each agent keeps its own** instructions, model, tools, memory, budget and
  workspace. Give two agents the same `Workspace` if they should share files.

In a voice pipeline the agent that was handed the call keeps it between turns. A
realtime voice model drives its own loop, so it is not offered the tool.

## Workflows: when the order is yours to fix

An agent decides what to do next. A workflow already knows: the steps are
written down, in YAML or JSON, and run as written — in sequence, in parallel,
in a loop, down one branch, or as a graph. Put agents in the steps where
judgement is needed, and tools where it is not.

```yaml
# refunds.yaml
name: refund_desk
inputs:
  customer: {required: true, type: string}
  orders:   {required: true, type: array}
state: {refunded: 0}

agents:                       # declared here, as in a blueprint — or passed in
  classifier: {instructions: 'Reply with only JSON: {"duplicates": [...]}'}
  writer:     {instructions: Write a short, plain email to the customer.}
  reviewer:   {instructions: 'Reply with only JSON: {"approved": true|false, "fix": "..."}'}

steps:
  - id: gather                # two lookups at once
    parallel:
      - id: charges
        foreach: "{{ inputs.orders }}"
        as: order
        concurrency: 4
        steps:
          - {tool: find_charges, args: {order: "{{ order }}"}}
      - {id: profile, tool: customer_profile, args: {customer: "{{ inputs.customer }}"}}

  - id: classify              # an agent reads what came back
    agent: classifier
    input: "Charges: {{ steps.charges.output }}"
    parse: json
    save: verdict             # → state.verdict

  - if: len(state.verdict.duplicates) > 0
    then:
      - foreach: "{{ state.verdict.duplicates }}"
        as: order
        steps:
          - {tool: issue_refund, args: {order: "{{ order }}", amount: 40}, retry: 2}
          - set: {refunded: "{{ state.refunded + 40 }}"}
    else:
      - return: "Nothing to refund."

  - id: email                 # write, review, go round until approved
    loop: {max: 3, until: state.review.approved}
    steps:
      - {id: draft, agent: writer, input: "Refunded {{ state.refunded }} EUR. Fix: {{ state.review.fix }}"}
      - {id: review, agent: reviewer, parse: json, save: review}

output: "{{ steps.draft.output }}"
```

```python
from agent_harness import Workflow

workflow = Workflow.from_file("refunds.yaml",
                              tools=[find_charges, customer_profile, issue_refund])
result = await workflow.run({"customer": "c_17", "orders": ["4182", "4190"]})

result.output                     # the approved email
result.state                      # {"refunded": 40, "verdict": {...}, "review": {...}}
result.steps["email"].iterations  # how many drafts it took
result.status, result.error, result.failed_step, result.cost_usd
```

**A step is one thing**, said by the key it carries:

| step | what it does |
|---|---|
| `agent: name` | Runs an agent on `input`. With no `input` it is handed what the step before produced. `parse: json` or the agent's output contract fills `.data`; `thread: true` keeps its conversation through the run. |
| `tool: name` | Calls a tool with `args` — through the permission gate, hooks, guardrails and audit trail, like any other call. |
| `set: {...}` | Writes to the shared state. |
| `steps: [...]` | A sequence, as one step. |
| `parallel: [...]` | Branches at the same time; `concurrency` caps them. The first failure stops the rest. |
| `foreach: <list>` | The `steps` body once per item (`as: order`, plus `index`), `concurrency` at a time. Its output is the list of what each pass came to. |
| `loop: {max, until, while}` | The `steps` body until a condition holds. `loop: 5` is a counted loop. `max` is a ceiling, always. |
| `if:` / `then:` / `else:` | One of two branches. |
| `switch: [{when, steps}, …]` | The first case that holds; a case with no `when` is the default. |
| `graph: [...]` | Nodes that say what they `needs`. Each starts the moment its needs are done, so everything that can run at once does. A node whose needs were skipped or failed is skipped; `join: any` runs it if any one arrived. |
| `wait: 2` · `fail: "why"` · `return: value` | Pause; stop with an error; finish now with this output. |

**Any step may carry** `when:` (skip unless it holds), `save:` (keep its result
in the state), `retry: 2` or `{max, delay, backoff}`, `timeout:` in seconds, and
`on_error: continue` — the failure is recorded on `steps.<id>` and the run goes on.

**Values are templates.** `{{ ... }}` holds an expression over `inputs`, `state`,
`steps.<id>` (`.output`, `.data`, `.status`, `.error`, `.iterations`) and
`previous`. A value that is one expression keeps its type —
`amount: "{{ state.total }}"` is a number — and inside other text it is written
out. Expressions are parsed and checked against a short list of what is allowed
(comparisons, arithmetic, `and`/`or`/`not`, `a if b else c`, and functions such
as `len`, `sum`, `join`, `json`, `default`, `lower`, `matches`); nothing is ever
passed to `eval`.

**A file that is wrong says so when it is loaded**, not half-way through a run:
an unknown key, two steps with one id, a template naming a step or an input that
does not exist, a tool or agent nobody supplied, a graph whose nodes wait on each
other.

```python
async for event in workflow.stream(inputs):       # step_start, step_end, step_skipped,
    print(event.type, event.step, event.text)     # step_retry, step_failed, workflow_end

print(workflow.describe())                         # the outline, without running it
agent = Agent("desk", tools=[workflow.as_tool()])  # a workflow an agent can call
blueprint.workflow("refund_desk", tools=[...])     # `workflows:` in an agents.yaml
```

```bash
agent-harness workflow refunds.yaml --check
agent-harness workflow refunds.yaml --tool myapp.tools:find_charges \
    --input customer=c_17 --input 'orders=["4182","4190"]'
```

The rest of the harness applies. A stop is honoured before every step; `budget:`
in the file is a ceiling for every agent in it; `max_steps` (1000) ends a loop
that does not; every step is a span, an audit record and — through the
`workflow_start`, `workflow_step` and `workflow_end` hooks — something you can
refuse. A failed step ends the run with `result.error` set rather than raising.

Not there yet: a run is not checkpointed, so a failed workflow starts again from
the top; and a graph runs forwards — to go round again, use a `loop`.

## The orchestrator

```python
from agent_harness import Orchestrator, Budget

boss = Orchestrator("boss", max_concurrency=4, review=True, max_rework=1,
                    budget=Budget(max_usd=2.00))
result = await boss.run("Summarise how Q3 went, with the numbers cited.")
```

1. **Plan** — acceptance tests are written *before* any work starts, then the
   task graph, then an estimate: what it will cost and how wide it goes
   (`plan.estimate_usd`, `plan.parallelism`).
2. **Staff** — reuse from the bench, else build with the factory.
3. **Run** — tasks run as their dependencies finish, many at once, with
   per-task retries and deadlines; a task receives only what it depends on.
4. **Check** — each hand-back is checked against its task's `done_when`. One
   that falls short is sent back with what it lacked, and when it stays short —
   or fails, or misses its deadline — the task is **planned again by another
   route** (`max_replans=2` a job). What waited for it then waits for the new
   tasks, which are told what the first attempt produced.
5. **Consolidate** — merge, de-duplicate, rank, attribute. Work that was
   re-planned is spoken for by what replaced it.
6. **Review** — independent critics check the deliverable against the
   definition of done; a rejection becomes new tasks and a rework round.

```python
Orchestrator(
    "boss",
    critics=["the accuracy of every figure", "length and tone"],   # or critics=3
    accept="all",                  # or "majority"
    check_tasks=True, max_replans=2,
    on_over_estimate="stop",       # "warn" (the default) · "stop" · "ignore"
)
```

- **Critics** review on their own, without seeing each other's verdict; what
  two of them both found is said once. A critic that returns nothing usable is
  asked again and then left out — and if none gives a verdict the deliverable is
  reported **unreviewed** (`review.reviewed is False`, a warning on the result),
  never silently accepted.
- **The estimate is held against the budget.** Before each wave the work still
  to run is priced — by what this job's own tasks have been costing once some
  have finished, by the estimate until then — and compared with what the budget
  has left. `"warn"` says so and carries on; `"stop"` starts nothing it cannot
  pay for, and hands back what was finished with `stop_reason == "budget"`.
  After three finished tasks the estimate is what tasks here actually cost.
- **What it cost is all of it.** `result.cost_usd` is the whole job — the plan,
  the specs the factory wrote, every check, the consolidation and the review as
  well as the sub-agents' work — and `result.spend` breaks it down: `work_usd`,
  `overhead_usd`, and each agent's share.
- **A job outlives its process.** It is saved to the harness's session store
  after every wave and every stage. If the process dies,
  `await boss.resume(result.run_id)` — in any process on the same store —
  carries on from the last task that finished: finished tasks are not run
  again, and neither is the plan or a consolidation already made.
  `boss.job(id)` says where a job stands. Two processes cannot both run one
  job: the one that finds the other has written to it stops.

`result.data["plan"]` and `result.data["review"]` carry the full record.

## A2A: agents that talk to other frameworks' agents

The [agent-to-agent protocol](https://a2a-protocol.org) is how an agent built
with one framework calls an agent built with another. Both sides are here: serve
any of your agents over A2A, and use anyone else's as if it were one of yours.
There is no SDK — the server is a plain ASGI application and the client is
`httpx`.

### Serving an agent

```python
# myapp.py
from agent_harness import Agent, Harness
from agent_harness.a2a import A2AServer

harness = Harness(sessions="postgresql://user:pass@host/agents")
agent = Agent("pricing", "Quote prices from the price list.",
              description="Quotes plan prices.", tools=[price_list], harness=harness)

server = A2AServer(agent, auth={"sk-live-1": {"user_id": "acme-bot", "tenant_id": "acme"}})
```

```bash
uvicorn myapp:server --workers 8          # it is an ASGI app: any ASGI server runs it
agent-harness a2a serve --name pricing --token sk-live-1   # or the built-in one, to develop
```

That serves the agent card at `/.well-known/agent-card.json` and the JSON-RPC
binding of A2A 0.3: `message/send`, `message/stream` (SSE, token by token),
`tasks/get`, `tasks/cancel`, `tasks/resubscribe` and the push-notification
methods. A message becomes a task; the answer is its `response` artefact; a
`contextId` carries the conversation on; files and structured data in a message
reach the agent as attachments.

`A2AServer({"billing": billing, "orders": orders})` serves several agents, each
under its own name.

### Calling one

```python
from agent_harness.a2a import A2AClient, RemoteAgent

async with A2AClient("https://agents.example.com/pricing", token="sk-…") as client:
    task = await client.send("What does the gold plan cost?")
    task.text, task.state, task.context_id

    async for event in client.stream("And for 12 seats?", context_id=task.context_id):
        print(event.text, end="")

    task = await client.send("Price the whole catalogue.", blocking=False)
    task = await client.wait(task.id)                  # or client.get / client.cancel
```

A `RemoteAgent` is that agent wearing the shape the harness expects, so it goes
wherever one of your own would:

```python
pricing = await RemoteAgent.connect("https://agents.example.com/pricing", token="sk-…")

Agent("manager", subagents=[pricing])                   # delegated to, like any sub-agent
Agent("manager", tools=[pricing.as_tool("ask_pricing")])
Workflow.from_file("quote.yaml", agents=[pricing])      # an `agent: pricing` step
```

```yaml
# or declared, in a blueprint or a workflow file
agents:
  pricing:
    description: Quotes plan prices.
    a2a: {url: "https://agents.example.com/pricing", token_env: PRICING_TOKEN}
```

A failed or unreachable remote agent is a `result.error`, never an exception in
the middle of your run. On a governed harness a remote agent must be registered
before anything is delegated to it — pass it the harness
(`RemoteAgent.connect(url, harness=harness, identity={...})`).

### Running it at scale

The server is written to be one of many replicas behind a load balancer.

- **A replica keeps nothing another one needs.** Tasks and conversations live in
  the harness's session store. Name a database there
  (`Harness(sessions="postgresql://…")`, or Redis, MongoDB, DynamoDB, …) and any
  replica can answer `tasks/get`, carry on a `contextId`, cancel a task or
  re-subscribe to it — whichever replica is running it. `store=` takes a
  `TaskStore` of your own.
- **A message sent twice is one task.** The task id is derived from the caller
  and the `messageId`, so a client that retries a request it never saw the
  answer to gets the first task back — even from a different replica — and the
  model is paid for once. `A2AClient` retries with the same id for that reason.
- **A full replica says so.** `max_concurrency` tasks run, `max_queue` wait, and
  past that the answer is `429` with `Retry-After` rather than a queue that
  grows without end. `GET /healthz` reports what is running and is `503` while
  draining.
- **Nothing waits for ever.** `task_timeout` ends a task that runs too long. A
  task whose worker died is reported `failed` the next time anyone asks, not
  `working` for ever. A streaming connection gets a keep-alive comment so a
  proxy does not close it, and a caller who hangs up stops the stream, not the
  task.
- **Callers are told, not polled, if you allow it.**
  `push_notifications=True` lets a caller register an https webhook (never a
  private or loopback address) that is sent the task when it is over.
- **Every caller has their own.** `auth` is a token, several, a token → identity
  mapping, or a function of the request headers. A caller's tasks and
  conversations are stamped with who they are; anyone else asking is told there
  is no such task. Each conversation runs on its own instance of the agent,
  acting for its caller, so nothing one caller said is in another's memory.
- **Shutdown is orderly.** On the ASGI lifespan shutdown the server stops taking
  work, gives what is running a moment, and records the rest as cancelled.

Tasks are kept until you delete them — give the store a TTL
(`session_provider("redis://…", ttl=7 * 86_400)`) or clear them yourself.

Not there yet: the gRPC and REST bindings, `input-required` (a task that stops to
ask the caller something), and signed or authenticated-extended agent cards.

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

### Serving your agents as an MCP server

The other direction: any MCP client — Claude Desktop, Cursor, another
framework's agent — uses your agents as tools.

```python
# myapp.py
from agent_harness.mcp import MCPAgentServer

server = MCPAgentServer(
    [billing, orders],
    api_keys={"billing": os.environ["BILLING_MCP_KEY"],     # each agent, its own key
              "orders": os.environ["ORDERS_MCP_KEY"]},
)
```

```bash
uvicorn myapp:server --workers 8                     # it is an ASGI app
agent-harness mcp-serve --name helper --api-key …    # the built-in server, to develop
agent-harness mcp-serve --stdio                      # for a client that starts it itself
```

Each agent is one tool, named after it. The client sends a `task`; the agent
runs with its own model, tools, guardrails and budget; its answer comes back.

**Every agent has its own key.** A key opens only the agents it was issued for:

| the caller sends | they are offered |
|---|---|
| the billing key | `billing` — and `orders` is, to them, not there |
| the orders key | `orders` |
| a key from `api_key=` | every agent |
| no key, or a wrong one | `401` |

```python
MCPAgentServer(agents, api_key="sk-…")                          # one key for everything
MCPAgentServer(agents, api_keys={"billing": ["sk-a", "sk-b"]})  # several keys, to rotate
MCPAgentServer(agents, api_keys={"docs": None, "billing": "sk-…"})   # `docs` needs no key
MCPAgentServer(agents, api_keys={"billing": {"key": "sk-…", "user_id": "ada",
                                             "tenant_id": "acme"}})   # whose calls they are
MCPAgentServer(agents, auth=lambda headers: {...})               # your own check

server.add_key("sk-new-partner", agent="orders")     # issue one while serving
server.revoke_key("sk-old-partner")                  # and stop one working, now
```

Keys arrive as `Authorization: Bearer …` or `X-API-Key: …`, are compared in
constant time, and are never logged — the audit trail records a key's id. With
no keys at all the server is open, which is for development. Each agent is also
served alone at `/<name>/mcp`, for a client that should be pointed at exactly
one; `/mcp` serves everything the key opens.

- **Conversations.** A caller may pass `conversation_id` — any id of their
  choosing — and calls that share it are one conversation the agent remembers.
  It belongs to the key that started it: the same id under another key is
  another conversation. Without one, each call is a single task.
- **The agents' own tools.** `expose_tools=True` (or globs) serves them too, as
  `<agent>_<tool>`, called through the agent so its permission gate, hooks and
  audit trail still apply.
- **What comes back.** The answer as text; an output contract as
  `structuredContent`; files the agent produced as embedded resources; a failed
  run as `isError`, which the calling model can read and act on.
- **Long runs.** A caller that sends a progress token is streamed
  `notifications/progress` for each step and tool. A call is stopped at
  `timeout`, by `notifications/cancelled`, or when the caller hangs up.
- **Stateless.** The transport is MCP's streamable HTTP with no session kept in
  the process, so any replica answers any request; conversations live in the
  harness's session store. `max_concurrency` and `max_queue` bound the work, and
  `/healthz` reports it.

In a blueprint each agent names the environment variable its key is in, and
`blueprint.mcp_server()` (or `agent-harness mcp-serve --blueprint agents.yaml`)
serves them:

```yaml
agents:
  billing: {instructions: Handle refunds., mcp_key_env: BILLING_MCP_KEY}
  orders:  {instructions: Track orders.,   mcp_key_env: ORDERS_MCP_KEY}
```

## Budgets that stop instead of failing

```python
SubAgentSpec(name="researcher", description="Finds things out.",
             budget=Budget(max_input_tokens=10_000, max_output_tokens=2_000))
```

Reaching a ceiling is not an error. The run ends cleanly, whatever the agent
produced is kept, and a line is appended saying why it stopped:

```
I got through three of the five documents...

[The budget for this agent is exceeded — output tokens 2,048 of 2,000.
 The answer above is what it completed before stopping.]
```

The parent gets that as the sub-agent's result and carries on. `result.stop_reason`
is `"budget"` and `result.budget_exceeded` names the axis. Pass
`Budget(..., on_exceed="raise")` if you would rather it were an error.

## When a model cannot be reached

```python
harness.router = ModelRouter(fallbacks=["claude-sonnet-5", "gpt-4.1"])
```

The loop walks the chain, resolving each model's provider as it goes. A 4xx is
*not* retried elsewhere — the request is wrong and the next model will reject it
the same way. Every switch lands in the journal and the audit trail.

## Versions

One agent, several configurations:

```python
agent = Agent(
    "support",
    tools=[order_status, issue_refund, lookup],
    version="v2",
    versions={
        "v1": {"instructions": "Answer order questions.",
               "tools": ["order_status"], "model": "claude-sonnet-5"},
        "v2": {"instructions": "Answer order questions. Cite the order.",
               "tools": ["order_status", "lookup"],
               "guardrails": {"require_tools": ["order_status"]},
               "model": "claude-opus-5"},
    },
)

await agent.run(task)                  # v2
await agent.run(task, version="v1")    # the old one, unchanged
```

A version says what is *different*; everything it leaves out falls through. The
harness, provider and memory are shared, so switching is cheap and the two are
comparable — run the same golden tasks against each:

```python
v1 = await suite.run(agent.use("v1"), label="v1")
v2 = await suite.run(agent.use("v2"), label="v2")
print(v2.compare(v1).render())
```

## Declaring it all in a file

```yaml
# agents.yaml
defaults: {model: claude-opus-5}

prompts:
  house_style: Answer in plain sentences and cite the order.

guardrails:
  strict: {require_tools: [order_status], no_pii: true, no_placeholders: true}

subagents:
  researcher:
    description: Finds things out, read-only.
    instructions: "{house_style} Cite every claim."
    tools: [lookup]
    tier: fast
    budget: {max_input_tokens: 10000, max_output_tokens: 2000}
    guardrails: {require_citation: true}

agents:
  support:
    instructions: "{house_style}"
    tools: [order_status, lookup]
    subagents: [researcher]
    handoffs: [billing]          # or {agent: billing, history: text, sticky: false}
    guardrails: strict
    versions:
      v1: {instructions: Answer order questions., tools: [order_status]}
      v2: {instructions: "{house_style}"}
  billing:
    description: Refunds and invoices.
    tools: [lookup]
    handoffs: [support]
```

```python
blueprint = Blueprint.from_file("agents.yaml")
agent = blueprint.build("support", tools=[order_status, lookup])
everything = blueprint.build_all(tools=[order_status, lookup])   # one harness
```

JSON works the same way. Tools stay in code — they *are* code — so you either
hand them in or let the file name them as import paths
(`myapp.tools:order_status`). Everything else is declaration, and belongs
somewhere it can be reviewed and diffed.

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

### Deterministic detectors

Exact where they can be, so you can leave them switched on:

```python
AgentGuardrails(no_pii=True, no_secrets=True, no_injection=True,
                grounded=0.6, not_toxic=True, no_repetition=True)
```

| Detector | What makes it usable |
|---|---|
| `PIIDetector` | cards are Luhn-checked, IBANs mod-97-checked, and matches are precedence-ordered — a card is never also reported as a phone number |
| `SecretDetector` | known key formats, plus Shannon entropy for keys nobody has published a pattern for |
| `InjectionDetector` | weighted signals scored 0-1, because one suspicious phrase is weak evidence and three together are not |
| `GroundednessDetector` | which content words in the answer appear nowhere in the sources |
| `ToxicityDetector` | a screen, including character substitution — not a classifier |
| `RepetitionDetector` | n-gram repetition, for a model looping on itself |

They report *findings* with a severity, a confidence and the spans they matched,
so `PIIDetector().redact(text)` removes exactly the value and leaves the sentence.

### LLM judges

For what an algorithm cannot decide:

```python
from agent_harness import LLMGuard, POLICIES

rails = AgentGuardrails(
    LLMGuard(cheap_agent, POLICIES["safety"]),
    LLMGuard(cheap_agent, "never name a competitor", block_at="high"),
    no_pii=True,            # the deterministic checks run first
)
```

Three things this gets right:

- **A structured verdict.** The judge returns JSON with a severity, not a mood,
  and a judge that will not answer in JSON has failed rather than passed.
- **It fails the way you choose.** `on_error="block"` (the default), `"allow"`
  or `"raise"`. A guard that silently passes when it breaks is not a guard.
- **Cheap checks first.** `check_async` runs the deterministic checks and only
  pays for a judge if they are all happy — no reason to spend a model call
  confirming what a regex just proved.

Ready policies: `safety`, `pii`, `relevance`, `groundedness`, `jailbreak`,
`tone`, `compliance`. Or pass your own sentence.

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
| `Approvals` | a run that needs a person's yes is stored and stops; approved hours later, in any process, it carries on from that step. [More below](#approvals-that-outlive-the-process) |
| `BudgetGuard` | spend, token, step, tool-call and sub-agent ceilings; child guards roll up to the parent |
| `RateGuard` | requests- and tokens-per-minute pacing, so you are not rate-limited by the provider |
| `HookEngine` | 13 events; `pre_tool` can block or rewrite arguments, `post_tool` can rewrite the result, `model_egress` sees the real provider and region of every model call |
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
| `SessionStore` | resume, fork or branch a run; a long job survives a restart — in files, SQL, MongoDB, Redis, DynamoDB or a storage account, owned and versioned |
| `WorkspaceBroker` | a jailed directory per sub-agent (or a shared one for handovers) — or a sandbox each: Docker, E2B, Daytona, Modal, a pod |
| `ConcurrencyScheduler` | semaphore, queue, backpressure, peak tracking |
| `ModelRouter` | per-task model and effort tier instead of one model for everything |
| `SpecCompiler` | a sub-agent blueprint → the exact provider payload, inspectable before you spend |

Path safety is enforced, not clamped: a workspace tool given `../../etc/passwd`
refuses rather than resolving it. On this machine `shell` is absent unless the
workspace was created with `allow_shell=True`, and even then it asks for
approval. Only a sandbox — its own machine — runs commands without asking.

## Approvals that outlive the process

A tool that asks first — `@tool(permission="ask")`, or a `PolicyGate` rule —
needs a person. When that person is at the terminal, pass an `approver` and they
are asked on the spot. When they are not — it is a ticket, a Slack message, a
manager who is asleep — the run should not wait and should not fail:

```python
harness = Harness(sessions="postgresql://user:pass@host/agents", approvals=True)
agent = Agent("support", tools=[find_order, refund], harness=harness)

result = await agent.run("Order 4182 was charged twice. Please fix it.")
result.stop_reason        # "approval"
result.approval.id        # "apr_9f2c…" — the run is in the database; nothing is waiting
```

Ten hours later, in another process, on another machine:

```python
for waiting in await harness.approvals.pending():
    print(waiting.id, waiting.describe())   # support wants to run refund(order_id="4182", amount=40)

await harness.approvals.approve("apr_9f2c…", by="maria", note="duplicate confirmed")
result = await agent.resume_approval("apr_9f2c…")     # or stream_approval(...)
```

The run picks up in the step it stopped in. What that means, exactly:

- **The approved call runs with the approved arguments.** They were stored; the
  model is not asked again, so it cannot ask for something else. If anything
  between the approval and the tool changes them — a hook, say — the call is
  refused.
- **Tools that had already run are not run again.** A step often asks for
  several: the lookup ran, the refund waited. The lookup's answer is kept with
  the record and reused.
- **A no is an answer.** `approvals.deny(id, by=, note=)` and the agent reads
  "declined by omar: over the limit" as the tool's result, and carries on.
  Several calls waiting in one step are decided together or one at a time
  (`call=`); the run resumes when all are.
- **Once.** Resuming takes a claim in a write the store refuses to a second
  taker. Four workers that see the same approval make one refund. A resume that
  crashes is marked `failed` and is not retried by itself — the tool may have
  run; `approvals.release(id)` hands it back once you have checked.
- **The rest of the run comes with it**: the conversation, the session and its
  sandbox, a mode's todo list and source ledger, what was spent. It may stop
  again further on; `result.approval` is then the next one.
- **Time.** An unanswered request expires (`expires=`, a week by default) and
  counts as declined; resuming it tells the agent so. If the conversation was
  continued while the request waited, the resumed run is saved beside it as a
  fork rather than over it.
- **Whose.** A request belongs to the user and tenant the run was for: others
  neither list it nor resume it. `self_approval=False` stops that user
  approving their own.

Where it is kept is where the chats are: `approvals=True` uses the session
database (PostgreSQL, MySQL, SQLite, MongoDB, Redis, DynamoDB, a storage
account), `Harness.local()` a directory, `Harness.on(url)` turns it on by
itself, and an `ApprovalStore` of your own is four methods.
`Approvals(store, notify=post_to_slack, expires=86_400)` says who is told and
for how long. From the shell:

```bash
agent-harness approvals --state .harness                     # what is waiting
agent-harness approvals approve apr_9f2c --by maria --note "checked"
agent-harness approvals resume apr_9f2c --tools              # with the run's own options
```

A run pauses only where it can be picked up: at the top of a conversation. A
sub-agent, a workflow's tool step, a voice turn and an agent mid-handoff are
still refused when nobody is there to ask, as before — and governance's
`require_approval` rules still wait in the process that asked.
`examples/23_approvals.py` plays the whole of it over a SQLite file.

## Governance: the laws your agents run under

```python
import os
from agent_harness import Agent, Harness
from agent_harness.governance import Governance, AgentIdentity

gov = Governance.from_packs(
    ["eu-ai-act", "gdpr", "ksa-pdpl", "owasp-agentic"],
    policy="governance.yaml",                  # your rules on top of the packs
    home="eu",
    regions={"azure": "eu"},                   # where endpoints that do not say, are
    signing_key=os.environ["AUDIT_KEY"],
)
harness = Harness.local(".harness", governance=gov)

support = Agent("support", "Resolve order problems.", harness=harness,
                tools=[order_status, issue_refund],
                identity=AgentIdentity(owner="cx-lead@acme.com", purpose="customer_support",
                                       tools=["order_status", "issue_refund"]))

print(gov.report("eu-ai-act").markdown())      # control → status → evidence
```

Packs switch on what a law or framework asks for. Your policy adds rules on top.
Combining them can only make things stricter: the strictest effect wins, so
adding a pack never loosens anything. Twenty-six packs ship, each dated and
sourced:

| Region | Packs |
|---|---|
| EU & UK | `eu-ai-act` · `gdpr` · `dora` · `nis2` · `uk-gdpr` |
| Gulf | `uae-pdpl` · `difc-reg10` · `adgm` · `ksa-pdpl` · `sdaia` · `qatar-pdppl` · `bahrain-pdpl` · `oman-pdpl` |
| Asia-Pacific | `singapore-agentic` · `singapore-pdpa` · `india-dpdp` · `china-genai` · `korea-ai-basic` · `japan-appi` · `vietnam-ai` |
| Americas | `us-nist-rmf` · `colorado-adm` · `texas-traiga` · `ccpa-admt` |
| Standards | `iso-42001` · `owasp-agentic` |

What it enforces, in the loop, before anything happens:

- **Residency, per call and per person.** Each model call is checked
  *after* the fallback chain has picked the backend, so the check sees the real
  destination: the Bedrock region, the `eu.` inference profile, the Vertex
  location, the endpoint host. An EU customer's data never reaches a US model.
  The call moves on to the next model in the chain instead. A Saudi customer
  on the same harness is held to Saudi rules (`Trace(tags={"jurisdiction": "sa"})`).
- **Least privilege that survives delegation.** A sub-agent acts under the
  intersection of its own identity and every identity above it. It can never
  reach a tool its parent could not.
- **Purpose limitation and minimisation.** A run carries a purpose. Data classes
  that purpose does not allow are redacted before the call leaves. They can also
  be pseudonymised: the model sees `⟨email:3f9a1c0b2d4e⟩`, and the tool gets the
  real address back.
- **Exact data classification.** Card numbers are Luhn-checked. National IDs
  are validated by their own check digits: Emirates ID, Saudi ID and Iqama,
  Aadhaar (Verhoeff), Singapore NRIC, the Chinese resident ID; EU VAT numbers
  by country prefix. Special-category data (health, religion and so on) is
  detected by keyword screens, reported at lower confidence, and labelled as
  such; redacting it removes the whole sentence that carries it. A data-class
  name a policy misspells is refused at load time instead of never matching.
- **Human oversight that fails closed.** `require_approval` rules need a quorum
  of *distinct* approvers. The person the agent acts for cannot approve their
  own request. A timeout, an error or a missing approver all count as *no*.
  Pass an `approver=` callable, or leave it out and answer the queue from your
  own UI (`gov.oversight.pending()`, `.approve(id, by=...)`); `approval_notify=`
  tells your team when something is waiting.
- **Agent-specific threats.** A tool result carrying prompt-injection signals
  marks the run *untrusted* (the flag spreads to the parent run). After that,
  payments and other sensitive tools wait for a person, and nothing is written
  to memory. Loops, tool storms and runaway delegation are stopped, a tool that
  keeps failing has its breaker opened for the rest of the run, and an incident
  is opened once per cause.
- **No way round it.** Context summaries go through the same egress check as
  the loop's own calls. A sub-agent on a harness without governance cannot be
  delegated to, and `Agent.as_tool()` nesting obeys the same depth limit. A
  governance failure refuses the action rather than crashing the run — or
  letting it through.
- **Supply chain.** Every tool's schema is fingerprinted. `inventory.pin()`
  freezes them. A tool that changes afterwards (an MCP server swapping what a
  tool does) is reported, or refused with `tool_drift: deny`.
- **Prohibited uses are refused at build time.** Declaring `domains=["social_scoring"]`
  raises an error instead of producing an agent.

And what it records:

- **Every decision, signed.** Each decision names the policy version (sha256)
  that made it. `signing_key=` adds HMAC signatures. `Ed25519Signer`
  (`pip install agent-harness-adk[governance]`) adds public-key signatures, so
  auditors can verify the trail without being able to write it.
- **Erasure that keeps the audit intact.** Nothing personal enters the trail
  in the clear: detected values are tokens under a per-person key, and task
  text — which can name anyone — is kept only as a token
  (`audit_args="tokens"` does the same for every tool argument). `await
  gov.rights.erase(trace)` deletes their memory, sessions, deliverables, journal
  entries and trace spans, and destroys that key. The chain still verifies, but
  nothing in it can be linked back to them. You get a signed receipt;
  `gov.rights.access(trace)` exports everything held about the person.
- **Transparency.** `result.disclosure` holds the AI notice in the languages the
  packs need. `result.provenance` is a signed, machine-readable manifest (EU AI
  Act Art 50, China GB 45438). Visible labels are applied only where the law
  asks for them.
- **Incidents with the regulator's deadlines.** Opening one works out every
  clock that applies: GDPR 72 h, DORA 4 h / 72 h / 1 month, EU AI Act Art 73,
  NIS2, SDAIA, PDPC and the rest.
- **Evidence.** `gov.report(pack)` shows each requirement as met, partial,
  gap or manual. It is worked out from the live configuration and the audit
  trail, with the next action for every gap. It also produces an AI inventory
  (`to_cyclonedx()`), a DORA third-party register, and draft DPIA / FRIA text
  (`gov.impact_assessment(agent)`).

```yaml
# governance.yaml
packs: [eu-ai-act, gdpr, owasp-agentic]
home: eu
signing_key_env: AUDIT_KEY
policy:
  purposes:
    customer_support: [contact, financial]      # nothing else reaches the model
  rules:
    - id: refunds-need-a-human
      on: tool
      match: {tags: [payments]}
      when: "args.amount > 100"                 # a small, safe expression language
      effect: require_approval
      approvers: 2
      timeout: 15m
```

Runnable end to end with no key: `examples/07_governance.py` (an EU bank also
serving Saudi customers), `08_governance_saudi_government.py` (in-Kingdom
residency, four-eyes permits, SDAIA breach clock, DPIA draft) and
`09_governance_singapore_fintech.py` (monitor-then-enforce rollout, prompt
injection, inherited authority, tool drift).

`mode="monitor"` makes and records every decision without enforcing any of them.
That is the way to roll governance out on a live system. Without governance
attached the loop pays nothing, and `import agent_harness` does not load it.

```bash
agent-harness governance packs                  # what ships
agent-harness governance pack ksa-pdpl          # requirements, rules, residency, clocks
agent-harness governance check --config governance.yaml       # validate, with warnings
agent-harness governance report --config governance.yaml --pack gdpr
agent-harness governance inventory --config governance.yaml --agents agents.yaml
agent-harness governance verify --state .harness --key-env AUDIT_KEY
agent-harness governance dsar erase --user alice --tenant acme --state .harness \
    --config governance.yaml
```

This layer supplies technical controls and the evidence for them. It is not
legal advice and does not make anyone compliant by itself. The reports say so,
and list what is left for people: the DPO, the EU database registration, the
signed DPIA.

## Providers

```python
Agent("a", model="claude-opus-5")      # → Anthropic
Agent("b", model="gpt-4.1")            # → OpenAI
Agent("c", model="gemini-2.5-pro")     # → Gemini
Agent("d", model="grok-4")             # → xAI
Agent("e", model="llama3.2", provider="ollama")
```

The provider is inferred from the model id, or named. They live in
`agent_harness.llm_providers`: the three direct APIs, Bedrock, Vertex (Claude
and Gemini), Azure OpenAI and Azure AI Foundry, and every OpenAI-compatible
vendor as a preset — `OpenRouterProvider`, `GroqProvider`, `TogetherProvider`,
`DeepSeekProvider`, `MistralProvider`, `XAIProvider`, `FireworksProvider`,
`CerebrasProvider`, `OllamaProvider`, `LMStudioProvider`, `VLLMProvider` — plus
`OpenAICompatibleProvider(base_url=...)` for any gateway or proxy.
`register_provider("name", MyProvider)` adds your own.

### What does each provider need?

```python
from agent_harness import list_llm_providers, describe_llm_provider, check_llm_provider

for spec in list_llm_providers():                 # every provider, one ProviderSpec each
    print(spec.name, spec.configured, [f.name for f in spec.required_fields])

print(describe_llm_provider("azure").render())
```

```
Azure OpenAI — provider='azure' (aliases: azure-openai)
  OpenAI models deployed in your own Azure OpenAI resource.
  status   needs setup — missing endpoint (or set $AZURE_OPENAI_ENDPOINT), one of: api_key ($AZURE_OPENAI_API_KEY) | credential
  auth     api_key or Entra ID
  can      embeddings, json_schema, streaming, thinking, tools, vision
  fields
    endpoint         required           url     ← $AZURE_OPENAI_ENDPOINT
                     The resource endpoint from the Azure portal.
    deployment       optional           str     ← $AZURE_OPENAI_DEPLOYMENT
                     The deployment name. Defaults to the model id.
    ...
  example  get_provider('azure', endpoint='https://my-resource.openai.azure.com', deployment='gpt-4.1-prod')
```

Every provider declares its connection fields as `ProviderField`s — which are
required, which are secret, which are alternatives to each other, and which
environment variables they are read from — so you can see what a backend needs
before you touch it. `list_llm_providers(configured_only=True)` shows what is
ready to use now; `capability="embeddings"` filters by what a provider can do.

`check_llm_provider("bedrock", region="eu-west-1")` answers "would this
connect?" offline, as a `ProviderCheck`: what is missing, where each setting
was found (`argument` or `$ENV_VAR` — never the value), and any argument that is
not a setting at all. `await ping_llm_provider("openai")` connects for real,
using the free model-listing endpoint where there is one, and returns a result
rather than raising. From a terminal:

```bash
agent-harness providers                 # the table: status and what each needs
agent-harness providers bedrock         # every field, env var, capability, an example
agent-harness providers openai --ping   # are these credentials any good?
```

### When a call fails

Every provider sends through one transport, so every one gets the same
behaviour:

- **Retries** on 408, 409, 425, 429, 5xx and Anthropic's 529, on timeouts, and
  on dropped connections — exponential back-off with jitter, so a hundred
  throttled sub-agents do not retry in lockstep.
- **Every Retry-After hint is honoured**: `Retry-After` in seconds or as a date,
  `retry-after-ms`, OpenAI's `x-ratelimit-reset-*`, Gemini's `RetryInfo`, and
  the `x-should-retry` verdict OpenAI and Anthropic send. A server asking for
  longer than `max_retry_after` fails the call at once, so a fallback model can
  take over instead of the run sleeping for ten minutes.
- **A shared cool-down**: when one call is told to wait, the calls running
  alongside it on the same provider wait too.
- **Streams recover** — a stream that fails before its first token (Anthropic's
  mid-stream `overloaded_error`, a throttled Bedrock stream) is restarted; one
  that fails after is raised, because replaying it would repeat text you have
  already shown.
- **A circuit breaker**: after five consecutive failures a provider is not
  called for 30 seconds — calls fail at once with `ProviderUnavailableError`,
  and the model router moves to the next model immediately. A 400 is the
  caller's fault and does not count.
- **Stale credentials are refreshed** once on a 401 — Vertex and Entra ID tokens,
  and Bedrock credentials from the AWS chain — and every Bedrock retry is
  signed afresh.

```python
from agent_harness import AnthropicProvider, RetryPolicy, CircuitBreaker

provider = AnthropicProvider(
    retry=RetryPolicy(max_retries=5, initial_delay=1.0, max_delay=30,
                      max_retry_after=60, max_elapsed=180),
    circuit_breaker=CircuitBreaker(failure_threshold=3, reset_timeout=60),
    max_concurrency=16,               # cap on in-flight requests
    timeout=120, connect_timeout=5,   # per attempt; CompletionRequest.timeout per call
    on_retry=lambda e: print(f"{e.provider} retry {e.attempt} in {e.delay:.1f}s: {e.reason}"),
)
provider.health()   # requests, retries, rate_limited, timeouts, last_request_id, circuit state
```

`on_retry` receives a `RetryEvent`; retries are also logged on the
`agent_harness.llm_providers` logger. What finally fails is typed, so you can
handle the cases differently — all are `ProviderError`s, and each carries
`status`, `retryable`, `retry_after`, `attempts` and the vendor's `request_id`:

| Error | Meaning | Retried |
|---|---|---|
| `RateLimitError` | 429 or throttling | yes |
| `QuotaExceededError` | out of credit or quota — waiting won't help | no (falls back) |
| `ProviderTimeoutError` | no answer in time | yes |
| `ProviderConnectionError` | DNS, TLS, a dropped connection | yes |
| `ProviderUnavailableError` | 5xx, 529 overloaded, or the circuit is open | yes |
| `AuthenticationError` | key, token or signature refused | no |
| `ContextWindowExceededError` | the prompt is longer than the model reads | no |
| `ModelNotFoundError` | no such model for this account | no |
| `InvalidRequestError` | anything else the request got wrong | no |

### The same models, on your cloud

```python
from agent_harness import BedrockProvider, VertexProvider, AzureOpenAIProvider

# AWS Bedrock — SigV4 signed, no API key
Agent("support", model="anthropic.claude-opus-5",
      provider=BedrockProvider(region="eu-west-1"))

# Google Vertex AI
Agent("support", model="claude-opus-5",
      provider=VertexProvider(project="my-project", region="europe-west1"))
Agent("support", model="gemini-2.5-pro",
      provider=VertexGeminiProvider(project="my-project"))

# Azure OpenAI, and Azure AI Foundry
Agent("support", provider=AzureOpenAIProvider(
    endpoint="https://my-resource.openai.azure.com",
    deployment="gpt-4.1-prod", api_version="2024-10-21"))
Agent("support", provider=AzureFoundryProvider(
    endpoint="https://my-project.services.ai.azure.com",
    deployment="claude-opus-5"))
```

Credentials follow each platform's own conventions: `AWS_*` environment
variables, a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`), or the botocore chain
(instance roles, SSO) if boto3 happens to be installed; `google-auth` or `gcloud auth print-access-token`; an `api-key` or
`credential=` from `azure-identity` for Entra ID. None of those libraries are
required — SigV4 is implemented against AWS's published test vectors using only
the standard library.

Platform model ids are normalised, so `anthropic.claude-opus-5` on Bedrock,
`us.anthropic.claude-opus-5` on a cross-region profile and `claude-opus-5@20260401`
on Vertex are all recognised as the same model and **priced the same** — cost
attribution keeps working wherever a model is served from.

### Every generation parameter

```python
Agent("deep",
      model="claude-opus-5",
      effort="high",             # none · minimal · low · medium · high · xhigh · max
      thinking=True,
      thinking_budget=16_000,    # for models that take a budget, not a level
      temperature=0.2, top_p=0.9, top_k=40, min_p=0.05,
      frequency_penalty=0.3, presence_penalty=0.1, repetition_penalty=1.05,
      seed=7, max_tokens=4096, stop=["</answer>"],
      cache=True,                # prompt caching where the provider has it
      user="customer-42",        # for abuse tracing
      model_options={"speed": "fast", "safety_settings": [...]})
```

The same names work on `AgentVersion`, `SubAgentSpec`, blueprint YAML and
`CompletionRequest`. They are **validated when the agent is built** —
`temperature=5` or `top_p=1.5` raises a `ConfigurationError` naming every bad
value and its allowed range, instead of a 400 halfway through a run.
`validate_parameters(...)` runs the same check on its own.

Each provider then turns them into what *it* accepts:

- **A parameter a provider does not have is dropped, not translated.** Anthropic
  has no `seed` or penalties; OpenAI has no `top_k`; `min_p` and
  `repetition_penalty` reach only the hosts that take them (vLLM, Together,
  Fireworks, OpenRouter). `describe_llm_provider(name).parameters` lists them.
- **A value a model would reject is fitted, not sent to fail.** Claude takes
  `temperature` 0–1 and one of `temperature`/`top_p`; with extended thinking it
  refuses `temperature` and `top_k` and needs `top_p` ≥ 0.95, a budget of at
  least 1024, and `max_tokens` above the budget. Thinking-only models (Opus 5,
  Sonnet 5 — under any Bedrock or Vertex id) get no sampling at all. OpenAI's
  reasoning models refuse sampling and `stop`. Grok's reasoning models refuse
  penalties and `stop`. Gemini 2.5 Pro cannot switch thinking off.
- **`effort` is the portable way to ask for reasoning**, mapped to the nearest
  level each model has:

| Provider | `effort` becomes |
|---|---|
| Anthropic, Bedrock, Vertex | `output_config.effort` (low–max); `none` switches thinking off |
| OpenAI o-series / GPT-5 | `reasoning_effort` (low–high; GPT-5 also `minimal`) |
| Gemini 2.5 | a thinking budget: 0 / 512 / 1024 / 8192 / 16384 / 24576 / 32768, fitted to the model's range |
| Gemini 3 | `thinkingLevel` (Pro: low, high; Flash: minimal–high) |
| OpenRouter | `reasoning.effort`, or `reasoning.max_tokens` for a budget |
| gpt-oss on Groq, Together, Fireworks, Cerebras, Ollama, vLLM | `reasoning_effort` |
| grok-3-mini | `reasoning_effort` (low, high) |

Nothing is dropped or changed silently. Every decision is recorded in a
`ParameterPlan`, logged once on `agent_harness.llm_providers`, and shown by
`explain` — without sending anything:

```python
OpenAIProvider().explain(CompletionRequest(model="o3", messages=[...],
                                           temperature=0.3, effort="max"))
# {"sent":     {"effort": "high", "max_tokens": 8192},
#  "dropped":  {"temperature": "o3 is a reasoning model and rejects sampling settings"},
#  "adjusted": {"effort": "'max' → 'high': o3 takes low, medium, high"},
#  "payload":  {...exactly what would go on the wire...}}
```

`extra={...}` is merged into the payload verbatim for anything not covered —
beta headers, new fields, a provider feature that shipped this morning.

Adapters normalise everything the loop depends on: tool calls, tool results,
thinking blocks (with their signatures, so extended thinking survives a tool
loop — including Gemini's thought signatures and Anthropic's redacted
thinking), cache tokens, stop reasons and refusals. All of them stream,
including Bedrock (its binary event stream is decoded and CRC-checked) and
Vertex. Cost is computed per
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
    elif event.type == "progress":       # a mode's todo list or ledger changed
        print(f"\n· {event.text}", event.data["todos"])
    elif event.type == "handoff":        # another agent has the conversation now
        print(f"\n· {event.data['from']} → {event.data['to']}")
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
agent-harness run "how did Q3 go?" --mode research --depth deep --tools
agent-harness chat --mode cowork --workspace ./project --approve
agent-harness run "build it" --mode cowork --sandbox docker://node:22
agent-harness sandboxes docker --check
agent-harness run "what does this say?" --attach contract.pdf --attach call.mp3
agent-harness voice --input question.wav --output answer.wav
agent-harness run "start it" --mode cowork --sandbox docker --state .harness --keep-sandbox
agent-harness run "carry on" --mode cowork --sandbox docker --state .harness --session ses_4f1c
agent-harness run "what is on the front page of example.com?" --browser
agent-harness knowledge add handbook/*.md --store qdrant://localhost:6333/handbook
agent-harness run "how long do refunds take?" --knowledge qdrant://localhost:6333/handbook
agent-harness knowledge check --store pgvector://user:pw@host/db   # prove a store works
agent-harness approvals --state .harness              # runs waiting for a person
agent-harness approvals approve apr_9f2c --by maria && agent-harness approvals resume apr_9f2c
agent-harness image "a red fox in snow" --reference fox.jpg --aspect 16:9
agent-harness run "design a logo for a bakery" --images --workspace ./studio
agent-harness search "heat pump subsidy" --recency year --engine brave
agent-harness search                    # the search engines, and which are set up
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

Not in this release: provider-side batch APIs. OCR, PDF and
DOCX parsing work through an optional install each rather than shipping in the
default dependency set.

## Licence

MIT — see [LICENSE](LICENSE).
