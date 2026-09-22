"""Small native tools worth having on day one."""

from __future__ import annotations

import ast
import operator
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from datetime import timezone as _tz
from typing import Any

from ..errors import ToolError
from ..tools import Tool, tool

__all__ = ["now", "calculate", "make_corpus_search", "basic_tools"]


@tool(name="now", tags=["builtin", "time"])
def now(timezone: str = "UTC") -> str:
    """The current date and time.

    Args:
        timezone: an IANA timezone name, or UTC.
    """
    if timezone.upper() == "UTC":
        return datetime.now(_tz.utc).isoformat(timespec="seconds")
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
    except Exception as exc:
        raise ToolError(f"unknown timezone {timezone!r}: {exc}", tool="now") from exc


_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right)
            if abs(exponent) > 64:
                raise ToolError("exponent too large", tool="calculate")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("only arithmetic is allowed here", tool="calculate")


@tool(name="calculate", tags=["builtin", "math"])
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression exactly.

    Args:
        expression: e.g. "1249 * 0.175 + 32".
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"cannot parse {expression!r}", tool="calculate") from exc
    return _eval(tree.body)


def make_corpus_search(
    documents: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    name: str = "search_corpus",
    snippet_chars: int = 600,
) -> Tool:
    """A keyword search tool over documents you hand it.

    Enough for a small, fixed corpus. For anything larger, put the documents in
    `SemanticMemory` and give the agent the `recall` tool instead.
    """
    corpus: dict[str, str] = dict(documents)

    @tool(name=name, tags=["builtin", "search"], cacheable=True)
    def search_corpus(query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the document corpus.

        Args:
            query: the words to look for.
            limit: how many documents to return.
        """
        terms = [t for t in query.lower().split() if len(t) > 2]
        hits: list[tuple[int, str, str]] = []
        for title, text in corpus.items():
            low = text.lower()
            score = sum(low.count(t) for t in terms)
            if not score:
                continue
            first = min((low.find(t) for t in terms if low.find(t) != -1), default=0)
            start = max(0, first - 100)
            hits.append((score, title, text[start:start + snippet_chars]))
        hits.sort(key=lambda h: -h[0])
        return [{"document": title, "score": score, "snippet": snippet}
                for score, title, snippet in hits[:limit]]

    return search_corpus


@tool(name="sleep", tags=["builtin"])
async def wait(seconds: float) -> str:
    """Pause before the next step, for polling an external job.

    Args:
        seconds: how long to wait, capped at 60.
    """
    import asyncio
    capped = max(0.0, min(seconds, 60.0))
    started = time.time()
    await asyncio.sleep(capped)
    return f"waited {time.time() - started:.1f}s"


def basic_tools() -> list[Tool]:
    """The safe, dependency-free defaults."""
    return [now, calculate]
