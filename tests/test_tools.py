from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_harness import Tool, ToolContext, ToolRegistry, tool
from agent_harness.errors import ToolError, ToolNotFound


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone.

    Args:
        name: who to greet
        excited: add an exclamation mark
    """
    return f"hello {name}{'!' if excited else ''}"


async def test_schema_comes_from_signature_and_docstring():
    schema = greet.to_schema()
    assert schema.name == "greet"
    assert schema.description == "Greet someone."
    props = schema.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["name"]["description"] == "who to greet"
    assert schema.parameters["required"] == ["name"]


async def test_invoke_validates_and_coerces():
    assert await greet.invoke({"name": "Ada"}) == "hello Ada"
    assert await greet.invoke({"name": "Ada", "excited": "true"}) == "hello Ada!"
    with pytest.raises(ToolError):
        await greet.invoke({})


async def test_async_tools_and_context_injection():
    seen: dict[str, str] = {}

    @tool
    async def remember(text: str, ctx: ToolContext) -> str:
        """Store text.

        Args:
            text: what to store
        """
        seen["agent"] = ctx.agent
        return text.upper()

    assert "ctx" not in remember.parameters["properties"]
    out = await remember.invoke({"text": "hi"}, ToolContext(agent="tester"))
    assert out == "HI"
    assert seen["agent"] == "tester"


class Order(BaseModel):
    id: str
    qty: int = 1


async def test_pydantic_arguments_are_supported():
    @tool
    def place(order: Order) -> str:
        """Place an order.

        Args:
            order: the order
        """
        return f"{order.qty}x{order.id}"

    assert await place.invoke({"order": {"id": "A1", "qty": 3}}) == "3xA1"


async def test_run_never_raises_and_reports_errors():
    @tool
    def boom() -> str:
        """Always fails."""
        raise ValueError("nope")

    outcome = await boom.run("call_1")
    assert outcome.is_error
    assert "nope" in outcome.content


async def test_registry_select_and_lookup():
    registry = ToolRegistry([greet])

    @tool(tags=["fs"])
    def fs_read(path: str) -> str:
        """Read a file.

        Args:
            path: where
        """
        return path

    registry.add(fs_read)
    assert registry.names == ["fs_read", "greet"]
    assert registry.select(["fs_*"]).names == ["fs_read"]
    assert registry.select(["fs"]).names == ["fs_read"]  # tags match too
    assert registry.select([]).names == []
    with pytest.raises(ToolNotFound):
        registry.get("missing")


async def test_explicit_schema_passes_kwargs_through():
    async def remote(**kwargs):
        return kwargs

    entry = Tool(remote, name="remote",
                 parameters={"type": "object", "properties": {"q": {"type": "string"}}})
    assert await entry.invoke({"q": "hi"}) == {"q": "hi"}


async def test_large_output_is_truncated():
    @tool(max_output_chars=50)
    def big() -> str:
        """Returns a lot."""
        return "x" * 500

    outcome = await big.run("c1")
    assert "truncated" in outcome.content
    assert len(outcome.content) < 200
