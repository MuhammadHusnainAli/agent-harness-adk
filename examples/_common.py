"""Every example runs with or without an API key.

With a key in the environment it calls the real model. Without one it falls back
to a scripted FakeProvider so the example still runs end to end.
"""

from __future__ import annotations

import os

from agent_harness import FakeProvider


def pick_provider(script: list | None = None):
    """The real provider when a key is present, a scripted fake otherwise."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return None, "claude-opus-5"           # None → resolved from the model id
    if os.environ.get("OPENAI_API_KEY"):
        return None, "gpt-4.1"
    if os.environ.get("GEMINI_API_KEY"):
        return None, "gemini-2.5-pro"
    print("(no API key found — running against a scripted fake provider)\n")
    return FakeProvider(script or [], loop=True), "claude-opus-5"
