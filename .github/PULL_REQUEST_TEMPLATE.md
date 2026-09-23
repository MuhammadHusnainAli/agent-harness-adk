## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!-- The problem being solved. Link the issue: Fixes #123 -->

## How

<!-- Only if the approach is not obvious from the diff — trade-offs, what you
     tried first, anything a reviewer would otherwise have to guess. -->

## Checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check src tests examples` is clean
- [ ] Tests cover the change — a bug fix has a test that fails without it
- [ ] Every example still runs without an API key
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` if this is user-visible
- [ ] `uv lock` re-run and committed, if dependencies changed
- [ ] Docstrings and README updated, if the public API changed

## Things this project is picky about

<!-- Tick what applies, or delete the section. -->

- [ ] This adds a **runtime dependency** — explained above why the three we have
      are not enough
- [ ] This affects **import time** — before/after from
      `uv run python -X importtime -c "import agent_harness" 2>&1 | tail -5`
- [ ] This is a **breaking API change** — noted in the changelog
- [ ] This touches a **provider wire format** — tested with `httpx.MockTransport`
- [ ] This uses syntax newer than **Python 3.10** — checked with `UV_PYTHON=3.10`
