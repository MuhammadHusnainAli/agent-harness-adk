# Getting help

## Read first

- [README](README.md) — the full API tour, with runnable snippets for every
  piece: agents, sub-agents, skills, prompts, memory, MCP and the rails.
- [`examples/`](examples/) — five programs that run end to end **without an API
  key**. Start with `01_quickstart.py`.
- [CHANGELOG](CHANGELOG.md) — what changed and when.

## Then

| What you have | Where it goes |
|---|---|
| "How do I ...?" | [Discussions → Q&A](https://github.com/MuhammadHusnainAli/agent-harness-adk/discussions) |
| Something is broken | [Open a bug report](https://github.com/MuhammadHusnainAli/agent-harness-adk/issues/new?template=bug_report.yml) |
| Something is missing | [Open a feature request](https://github.com/MuhammadHusnainAli/agent-harness-adk/issues/new?template=feature_request.yml) |
| A security problem | **Do not open an issue** — see [SECURITY.md](SECURITY.md) |
| You want to contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Helping us help you

A bug report that gets fixed quickly usually has:

- the output of `pip show agent-harness-adk` and `python --version`
- which provider and model you were using
- a reproduction using `FakeProvider` — no key needed, and it proves the problem
  is in the harness rather than in a model response:

  ```python
  from agent_harness import Agent, FakeProvider, Harness

  provider = FakeProvider(["the response that triggers it"])
  agent = Agent("repro", provider=provider, harness=Harness.testing(provider))
  ```

- the full traceback, not just the last line

If a run misbehaved and you still have it, `harness.report()` and
`harness.journal.render()` say a great deal in very little space.

## What this project does not support

- Debugging your prompts or your model's answers. That is model behaviour, not
  harness behaviour — unless the harness sent the model something wrong, which
  `provider.requests` will show you.
- Provider outages, quotas and billing. Those belong to Anthropic, OpenAI or
  Google.
- MCP servers written by other people.
