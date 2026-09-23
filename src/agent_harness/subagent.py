"""Moved to `agent_harness.subagents`. Kept so existing imports keep working."""

from __future__ import annotations

from .subagents import (
    BENCH,
    BENCH_SPECS,
    FACTORY_PROMPT,
    Bench,
    SubAgentFactory,
    SubAgentSpec,
    build_agent,
)

__all__ = [
    "SubAgentSpec",
    "build_agent",
    "Bench",
    "BENCH",
    "BENCH_SPECS",
    "SubAgentFactory",
    "FACTORY_PROMPT",
]
