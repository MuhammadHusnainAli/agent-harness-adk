from __future__ import annotations

import pytest

from agent_harness import Prompt, PromptLibrary, Skill, SkillRegistry
from agent_harness.errors import ConfigurationError, ToolError
from agent_harness.prompts import parse_frontmatter, sections


def test_render_fills_variables_and_keeps_unknown_ones():
    prompt = Prompt("greet", "Hello {name}, welcome to {place}.")
    assert prompt.render(name="Ada", place="Cairo") == "Hello Ada, welcome to Cairo."
    assert "{place}" in prompt.render(name="Ada")


def test_partial_and_composition():
    prompt = Prompt("greet", "Hi {name}").partial(name="Ada")
    assert prompt.render() == "Hi Ada"
    combined = prompt + Prompt("bye", "Goodbye.")
    assert combined.render().endswith("Goodbye.")


def test_strict_render_raises_on_missing():
    with pytest.raises(ConfigurationError):
        Prompt("x", "Hi {name}").render(_strict=True)


def test_frontmatter_round_trip(tmp_path):
    path = tmp_path / "p.md"
    Prompt("triage", "Sort {ticket}.", description="Ticket triage").save(path)
    loaded = Prompt.from_file(path)
    assert loaded.name == "triage"
    assert loaded.description == "Ticket triage"
    assert loaded.render(ticket="T1") == "Sort T1."


def test_prompt_library_from_dir(tmp_path):
    Prompt("a", "A {x}").save(tmp_path / "a.md")
    Prompt("b", "B").save(tmp_path / "b.md")
    library = PromptLibrary.from_dir(tmp_path)
    assert library.names == ["a", "b"]
    assert library.render("a", x="1") == "A 1"


def test_sections_skips_empties():
    assert sections("head", None, ("Title", "body"), "") == "head\n\n## Title\nbody"


def test_parse_frontmatter_without_frontmatter():
    meta, body = parse_frontmatter("just text")
    assert meta == {} and body == "just text"


def _write_skill(root, name, description, body="Do the thing.", tools_py=None):
    folder = root / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    )
    if tools_py:
        (folder / "tools.py").write_text(tools_py)
    return folder


def test_skill_registry_loads_a_directory(tmp_path):
    _write_skill(tmp_path, "refunds", "How refunds work.")
    _write_skill(tmp_path, "escalation", "When to escalate.")
    registry = SkillRegistry.from_dir(tmp_path)
    assert registry.names == ["escalation", "refunds"]
    index = registry.index()
    assert "refunds: How refunds work." in index
    # The catalogue holds one line per skill, not the bodies.
    assert "Do the thing." not in index


async def test_load_skill_tool_returns_the_body(tmp_path):
    _write_skill(tmp_path, "refunds", "How refunds work.", body="Step one. Step two.")
    registry = SkillRegistry.from_dir(tmp_path)
    load = registry.load_tool()
    body = await load.invoke({"name": "refunds"})
    assert "Step one." in body
    # Asking for a skill that does not exist is a tool failure the agent can see.
    with pytest.raises(ToolError, match="no skill 'nope'"):
        await load.invoke({"name": "nope"})


def test_skill_bundles_its_own_tools(tmp_path):
    _write_skill(
        tmp_path, "pricing", "Price lookups.",
        tools_py=(
            "from agent_harness import tool\n\n"
            "@tool\n"
            "def price(sku: str) -> str:\n"
            '    """Look up a price.\n\n    Args:\n        sku: the sku\n    """\n'
            "    return '42'\n"
        ),
    )
    skill = Skill.from_dir(tmp_path / "pricing")
    assert [t.name for t in skill.tools] == ["price"]
    assert "tools.py" in skill.resources


def test_jinja_prompts_render_as_text_not_escaped_html():
    """A prompt goes to a model, so `&amp;` in it would be a bug, not a defence."""
    pytest.importorskip("jinja2")

    prompt = Prompt("report",
                    "{{ company }} profit {{ profit }}{% if urgent %} — URGENT{% endif %}")
    rendered = prompt.render(company="Ada & Co", profit="<1M>", urgent=True)

    assert rendered == "Ada & Co profit <1M> — URGENT"
    assert "&amp;" not in rendered and "&lt;" not in rendered


def test_a_plain_template_never_goes_near_jinja():
    prompt = Prompt("greet", "Hello {name} & welcome")
    assert prompt.render(name="Ada") == "Hello Ada & welcome"
