# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

While the project is pre-1.0, security fixes land on the latest release only.
Upgrade before reporting a problem you found on an older version.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report it privately through GitHub:

1. Go to the [Security tab](https://github.com/MuhammadHusnainAli/agent-harness-adk/security/advisories/new).
2. Click **Report a vulnerability**.

If you cannot use GitHub advisories, email
**muhammad.husnain.ali.738@gmail.com** with `SECURITY` in the subject.

Please include:

- what the problem is, and what an attacker gets out of it
- the version of `agent-harness-adk` and Python you used
- the smallest reproduction you can manage
- whether you have already disclosed it anywhere

### What to expect

| | |
|---|---|
| First response | within 5 days |
| Assessment and plan | within 14 days |
| Fix released | as fast as the severity warrants |

You will be credited in the advisory and the changelog unless you ask not to
be. Please give us a reasonable window to ship a fix before disclosing
publicly.

## What counts as a vulnerability here

This is a library for running AI agents, which means it executes tools, reaches
networks and handles credentials on behalf of a model. Things we want to hear
about:

- **Escaping the workspace jail** — any path, symlink or argument that lets a
  workspace tool read or write outside its root.
- **Bypassing the policy gate** — getting a tool to run that `PolicyGate`
  should have denied, or turning an `ask` into an `allow`.
- **Bypassing guardrails** — getting a secret pattern past redaction, or
  defeating the blocking rules with encoding or chunking.
- **Credential leakage** — an API key reaching a log, a trace, the journal, a
  memory file, a tool result, or a provider it was not meant for.
- **Budget bypass** — spending past a `Budget` ceiling that should have stopped
  the run.
- **MCP trust boundary** — a malicious MCP server reaching beyond the tools it
  declared, or its responses being executed rather than treated as data.
- **Deserialisation or injection** in session, checkpoint, memory or cache
  files that an attacker could have written.
- Dependency vulnerabilities that we actually reach in our code paths.

## What does not count

These are known, documented behaviours, not vulnerabilities:

- **The model deciding to call a tool you gave it.** If you hand an agent
  `shell` and allow it in the policy, it can run shell commands. That is the
  feature. Constrain it with `PolicyGate`, a tool allowlist and a workspace.
- **Prompt injection changing what the model does.** No prompt defence is
  complete. `Guardrails` warns on known patterns; it does not claim to stop a
  determined injection. Put the security boundary in the policy gate and the
  tool allowlist, not in the prompt.
- **`shell` or `fs_write` doing what they say.** They are gated (`shell` needs
  `allow_shell=True` *and* approval), and inside a workspace, by design.
- **A provider or MCP server you configured behaving badly.** Those are your
  trust decisions.

## Using this library safely

- Give every agent the smallest tool allowlist that can finish its task. Default
  `PolicyGate` to `deny` for anything that writes, spends or sends.
- Run untrusted work in an isolated workspace — `WorkspaceBroker(backend="docker")`
  where the command itself must be contained.
- Set a `Budget`. An agent loop with no spend ceiling is an open tap.
- Treat everything coming back from a tool, a document or an MCP server as
  untrusted data, never as instructions.
- Keep `Guardrails` on. It redacts known key formats from anything entering or
  leaving the context — cheap insurance against a key in a log.
- Do not put secrets in prompts, skills or memory. Keep them in your own code
  behind a tool.
