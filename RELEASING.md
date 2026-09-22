# Releasing

`agent-harness-adk` on PyPI, imported as `agent_harness`. Releases are driven by
a git tag: push `vX.Y.Z` and the `Release` workflow runs the full test matrix,
builds, publishes to PyPI and cuts the GitHub Release — in that order, so a
failing test stops the release before anything is published.

---

## One-time setup

Do these once, **before the first tag is pushed**. The release workflow will
fail at the publish step otherwise.

### 1. Create the PyPI Trusted Publisher

Trusted Publishing lets GitHub Actions prove its identity to PyPI over OIDC, so
no API token is stored anywhere. Because `agent-harness-adk` does not exist on
PyPI yet, register it as a *pending* publisher:

1. Sign in at <https://pypi.org> and go to
   <https://pypi.org/manage/account/publishing/>.
2. Under **Add a new pending publisher**, fill in exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `agent-harness-adk` |
   | Owner | `MuhammadHusnainAli` |
   | Repository name | `agent-harness-adk` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Save. The project is created on PyPI the first time the workflow publishes.

The environment name must be `pypi` — it is what `release.yml` declares. If you
change one, change both.

### 2. Create the GitHub environment

In the repository: **Settings → Environments → New environment → `pypi`**.

Optional but worth it: add yourself under **Required reviewers**, so every
publish waits for you to click approve. Nothing reaches PyPI until you do.

### 3. Check Actions permissions

**Settings → Actions → General → Workflow permissions** must allow
`GITHUB_TOKEN` to write contents, or the GitHub Release step cannot create the
release. "Read and write permissions" is the simple setting; the workflow itself
only requests `contents: write` on that one job.

---

## Cutting a release

1. **Update the version** in `pyproject.toml`:

   ```toml
   version = "0.1.0"
   ```

2. **Add the section to `CHANGELOG.md`** under a `## [0.1.0] — YYYY-MM-DD`
   heading. The workflow lifts exactly that section into the GitHub Release
   notes, so write it for the reader.

3. **Refresh the lockfile and check it locally**:

   ```bash
   uv lock
   uv sync --extra dev
   uv run pytest -q
   uv run ruff check src tests examples
   uv build && uvx twine check --strict dist/*
   ```

4. **Commit and push to `main`**, and let CI go green.

   ```bash
   git add -A
   git commit -m "release: 0.1.0"
   git push origin main
   ```

5. **Tag and push the tag.** This is the step that publishes.

   ```bash
   git tag -a v0.1.0 -m "agent-harness-adk 0.1.0"
   git push origin v0.1.0
   ```

6. **Watch it**: <https://github.com/MuhammadHusnainAli/agent-harness-adk/actions>.
   If you set required reviewers, approve the `pypi` environment when it asks.

7. **Verify** the published package installs from a clean environment:

   ```bash
   uv run --isolated --no-project --with agent-harness-adk python -c \
     "import agent_harness; print(agent_harness.__version__)"
   ```

The tag must match `version` in `pyproject.toml`. The `guard` job checks this
first and fails the release immediately if they disagree.

---

## Publishing by hand

If you would rather not use Trusted Publishing, or need to push a release from
your machine:

```bash
uv build
uvx twine check --strict dist/*

# Test it on TestPyPI first (a separate account and token):
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<test-token>

# Then the real thing:
uv publish --token pypi-<token>
```

Create the token at <https://pypi.org/manage/account/token/>. Scope it to this
project once the project exists; the first upload needs an account-wide token.

To use a token from CI instead of OIDC, replace the `publish` job's publish step
with:

```yaml
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

and drop the `permissions: id-token: write` block.

---

## If something goes wrong

**A version number on PyPI can never be reused**, even after deleting the
release. If a bad `0.1.0` is published, yank it
(**Manage project → Releases → Yank**) and ship `0.1.1`. Yanking hides it from
new installs while leaving existing pins working.

Before the tag is pushed, everything is reversible:

```bash
git tag -d v0.1.0                 # local only
git push origin :refs/tags/v0.1.0 # if it was already pushed and the run has not published
```

If the run already published to PyPI, deleting the tag changes nothing on PyPI —
bump the version instead.
