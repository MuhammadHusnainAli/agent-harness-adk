# Contributing

Thanks for taking the time. This is a small, opinionated library — the bar is
less about volume and more about not making it heavier or slower.

## Getting set up

```bash
git clone https://github.com/MuhammadHusnainAli/agent-harness-adk
cd agent-harness-adk
uv sync --extra dev
```

No API key is needed to develop or test. Everything runs against
`FakeProvider`.

```bash
uv run pytest -q                        # the suite: ~150 tests, under a second
uv run ruff check src tests examples    # lint
uv run ruff check --fix src tests       # and fix what it can
```

Against another Python version:

```bash
UV_PYTHON=3.10 uv sync --extra dev && UV_PYTHON=3.10 uv run pytest -q
```

CI runs 3.10, 3.11, 3.12, 3.13 and 3.14. If your change touches anything
version-sensitive, check 3.10 yourself before pushing — it is the one that
catches things.

## Before you open a pull request

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check src tests examples` is clean
- [ ] New behaviour has a test. Bug fixes have a test that fails without the fix.
- [ ] Every example still runs without an API key: `cd examples && uv run python 01_quickstart.py`
- [ ] `uv lock` was re-run if you touched dependencies, and the lockfile is committed
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]` if it is user-visible

Open an issue first for anything large or architectural. It saves you writing
code that gets turned down on the shape rather than the substance.

## What this project is picky about

**Dependencies.** There are three: `pydantic`, `httpx`, `pyyaml`. A fourth needs
a real argument. Optional extras are fine if the import is lazy and the feature
degrades cleanly without it — `SemanticMemory` falling back to a hashing
embedder is the pattern to copy.

**Import time.** Around 100 ms, and that is a feature. `httpx`, `yaml` and the
MCP module are imported lazily. Check before and after:

```bash
uv run python -X importtime -c "import agent_harness" 2>&1 | tail -20
```

**Python 3.10.** The floor. `datetime.UTC` is 3.11+. On 3.10
`asyncio.TimeoutError` is a *different class* from the builtin `TimeoutError`.
Ruff is set to `target-version = "py310"` and will catch most of it.

**Async first.** `async def` is the real implementation; sync wrappers are thin.
Never block the event loop — `asyncio.to_thread` for file and CPU work.

**No vendor SDKs.** The provider adapters speak raw HTTP on purpose, so all
three share one retry, cost and tracing path. If a provider gains a feature the
loop needs, add it to `CompletionRequest` and implement it in each adapter.

## Tests

Use `FakeProvider` — no network, no keys, no recorded cassettes:

```python
provider = FakeProvider([tool_call("order_status", order_id="4182"),
                         "It ships Thursday."])
agent = Agent("support", provider=provider, harness=Harness.testing(provider),
              tools=[order_status])
```

Script it with strings, tool calls, whole `Message` objects, exceptions, or a
callable that inspects the request and answers accordingly. For provider wire
formats, use `httpx.MockTransport` and assert on the request body — see
`tests/test_providers.py`.

Name tests after the behaviour, not the function:
`test_compaction_never_orphans_a_tool_result`, not `test_compact_2`.

## Style

Ruff is the arbiter (`line-length = 100`). Beyond that:

- Docstrings say what a thing is for and when to reach for it, not what the code
  obviously does.
- Comments explain *why*, especially where the code looks odd — those are the
  ones that survive.
- Type annotations everywhere; the package ships `py.typed`.
- Errors carry context: which tool, which provider, what limit, what was spent.

## Commits and pull requests

Conventional prefixes, because the changelog is written from them:
`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`, `chore:`.

Write the body for someone reading `git log` in a year: what changed, and why
it needed to. A pull request that fixes a bug should say how the bug showed up.

## Releasing

Maintainers only — see [RELEASING.md](RELEASING.md).
