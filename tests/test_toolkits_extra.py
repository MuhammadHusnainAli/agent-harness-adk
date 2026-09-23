"""Document parsing, chart rendering and sandboxed compute."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from agent_harness import Workspace
from agent_harness.errors import ToolError
from agent_harness.toolkits import (
    bar_chart,
    line_chart,
    make_chart_tool,
    make_document_tool,
    make_python_tool,
    make_report_tool,
    markdown_table,
    parse_document,
    parse_text,
    render_report,
    supported_formats,
)
from agent_harness.toolkits.charts import SERIES_DARK, SERIES_LIGHT

# --- document parsing ---------------------------------------------------------

def test_plain_text_and_markdown_come_back_whole(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nSome body text.")
    parsed = parse_document(path)
    assert "Some body text." in parsed["text"]
    assert parsed["format"] == ".md" and parsed["characters"] > 0


def test_csv_becomes_a_readable_table(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,amount\nada,10\ngrace,20\n")
    text = parse_document(path)["text"]
    assert "name | amount" in text
    assert "ada | 10" in text
    assert "(2 rows, 2 columns)" in text


def test_a_big_csv_is_capped_and_says_so():
    body = "a,b\n" + "\n".join(f"{i},{i}" for i in range(2000))
    text = parse_text(body, ".csv")
    assert "more rows" in text


def test_tsv_uses_tabs():
    assert "a | b" in parse_text("x\ty\na\tb\n", ".tsv")


def test_json_is_pretty_printed_and_bad_json_is_reported(tmp_path):
    path = tmp_path / "d.json"
    path.write_text('{"b":2,"a":1}')
    assert '"b": 2' in parse_document(path)["text"]

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ToolError, match="invalid JSON"):
        parse_document(bad)


def test_jsonl_survives_one_bad_line():
    text = parse_text('{"a":1}\nnot json\n{"b":2}', ".jsonl")
    assert '{"a": 1}' in text and "line 2" in text and '{"b": 2}' in text


def test_html_loses_its_markup_and_its_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><body><h1>Title</h1><script>evil()</script>"
                    "<p>Body&nbsp;text</p></body></html>")
    text = parse_document(path)["text"]
    assert "Title" in text and "Body text" in text
    assert "evil()" not in text and "<p>" not in text


def test_a_missing_file_and_an_unknown_format_are_clear(tmp_path):
    with pytest.raises(ToolError, match="no such file"):
        parse_document(tmp_path / "nope.txt")

    odd = tmp_path / "thing.xyz"
    odd.write_text("x")
    with pytest.raises(ToolError, match="cannot read"):
        parse_document(odd)


def test_an_optional_format_says_which_install_unlocks_it(tmp_path):
    pytest.importorskip  # noqa: B018 - the point is what happens when it is absent
    try:
        import pypdf  # noqa: F401
        pytest.skip("pypdf is installed, so this path cannot be exercised")
    except ImportError:
        pass
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(ToolError, match="pip install pypdf"):
        parse_document(path)


def test_the_format_list_separates_native_from_needs_install():
    formats = supported_formats()
    assert ".csv" in formats["native"] and ".json" in formats["native"]
    assert isinstance(formats["available"], list)
    assert all("pip install" in row for row in formats["needs_install"])


def test_truncation_is_flagged(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 5000)
    parsed = parse_document(path, max_chars=1000)
    assert parsed["truncated"] is True and "truncated" in parsed["text"]


async def test_the_document_tool_resolves_inside_the_workspace(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    workspace.write("report.md", "the contents")
    entry = make_document_tool(workspace)

    assert "the contents" in (await entry.invoke({"path": "report.md"}))["text"]
    with pytest.raises(ToolError, match="outside the workspace"):
        await entry.invoke({"path": "../../etc/passwd"})


# --- charts -------------------------------------------------------------------

def test_a_bar_chart_is_valid_svg_with_one_bar_per_value():
    svg = bar_chart({"jan": 10, "feb": 25, "mar": 18}, title="Revenue by month")
    root = ET.fromstring(svg)                       # parses, so it is well-formed
    assert root.tag.endswith("svg")
    # one surface rect plus one rect per bar
    rects = [e for e in root.iter() if e.tag.endswith("rect")]
    assert len(rects) == 4
    assert "Revenue by month" in svg


def test_bar_values_are_direct_labelled():
    svg = bar_chart({"a": 10, "b": 20})
    assert ">10<" in svg and ">20<" in svg          # the relief rule, in practice


def test_a_bar_chart_baseline_sits_at_zero_even_for_a_narrow_range():
    svg = bar_chart({"a": 100, "b": 104})
    root = ET.fromstring(svg)
    heights = [float(e.get("height")) for e in root.iter()
               if e.tag.endswith("rect") and e.get("rx") != "8"]
    # If the axis were truncated the two bars would look wildly different.
    assert min(heights) / max(heights) > 0.9


def test_negative_values_render_below_the_baseline():
    svg = bar_chart({"gain": 30, "loss": -20})
    root = ET.fromstring(svg)
    bars = [e for e in root.iter() if e.tag.endswith("rect") and e.get("rx") != "8"]
    assert len(bars) == 2
    assert float(bars[1].get("y")) > float(bars[0].get("y"))


def test_a_line_chart_draws_one_polyline_per_series_and_a_legend():
    svg = line_chart({"revenue": [1, 2, 3], "costs": [3, 2, 1]},
                     labels=["jan", "feb", "mar"], title="Trend")
    root = ET.fromstring(svg)
    lines = [e for e in root.iter() if e.tag.endswith("polyline")]
    assert len(lines) == 2
    assert lines[0].get("points").count(",") == 3
    assert "revenue" in svg and "costs" in svg      # identity is never colour alone


def test_series_colours_are_assigned_in_fixed_order_never_cycled():
    svg = line_chart({"a": [1, 2], "b": [2, 1], "c": [1, 1]})
    assert 'stroke="var(--s1)"' in svg
    assert 'stroke="var(--s2)"' in svg
    assert 'stroke="var(--s3)"' in svg
    # Dropping a series must not repaint the survivors.
    two = line_chart({"a": [1, 2], "b": [2, 1]})
    assert two.count("var(--s1)") >= 1 and "var(--s3)" not in two


def test_both_themes_are_defined_and_text_never_wears_a_series_colour():
    svg = bar_chart({"a": 1}, title="t")
    assert "prefers-color-scheme: dark" in svg
    for light, dark in zip(SERIES_LIGHT, SERIES_DARK, strict=True):
        assert light in svg and dark in svg
    assert "--ink: #0b0b0b" in svg and "--ink: #ffffff" in svg
    assert "fill: var(--ink)" in svg


def test_labels_are_escaped_not_injected():
    svg = bar_chart({"<script>alert(1)</script>": 5})
    assert "<script>" not in svg
    ET.fromstring(svg)


def test_too_many_series_is_refused_rather_than_rendered_unreadably():
    with pytest.raises(ToolError, match="cannot be told apart"):
        line_chart({str(i): [1, 2] for i in range(9)})


def test_empty_input_is_refused():
    with pytest.raises(ToolError):
        bar_chart({})
    with pytest.raises(ToolError):
        line_chart({})


def test_non_numeric_values_are_caught_by_name():
    with pytest.raises(ToolError, match="'b' is not a number"):
        bar_chart({"a": 1, "b": "twelve"})


def test_markdown_table_aligns_and_handles_both_shapes():
    from_dicts = markdown_table([{"name": "ada", "n": 1}, {"name": "grace", "n": 22}])
    assert "| name  | n  |" in from_dicts
    assert "|-------|----|" in from_dicts
    from_rows = markdown_table([["a", "b"]], headers=["x", "y"])
    assert "| x | y |" in from_rows
    assert markdown_table([]) == "(no rows)"


def test_a_report_leads_with_the_answer():
    text = render_report("Q3", [("Revenue", "4.2M"), ("Empty", "")],
                         summary="Q3 went well.")
    assert text.startswith("# Q3")
    assert text.index("Q3 went well.") < text.index("## Revenue")
    assert "## Empty" not in text          # empty sections are dropped


async def test_the_chart_tool_writes_into_the_workspace(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    entry = make_chart_tool(workspace)

    out = await entry.invoke({"kind": "bar", "title": "Sales",
                              "data": {"a": 1, "b": 2}, "path": "sales.svg"})
    assert "wrote sales.svg" in out
    assert "<svg" in workspace.read("sales.svg")

    svg = await entry.invoke({"kind": "line", "title": "Trend",
                              "data": {"s": [1, 2, 3]}})
    assert svg.startswith("<svg")

    with pytest.raises(ToolError, match="unknown chart kind"):
        await entry.invoke({"kind": "pie", "title": "x", "data": {"a": 1}})


async def test_the_report_tool_writes_markdown(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    entry = make_report_tool(workspace)
    await entry.invoke({
        "title": "Findings", "summary": "It works.",
        "sections": [{"heading": "Detail", "body": "The evidence."}],
        "path": "report.md",
    })
    written = workspace.read("report.md")
    assert written.startswith("# Findings") and "## Detail" in written


# --- sandboxed compute ---------------------------------------------------------

async def test_python_runs_in_the_workspace_and_returns_what_it_printed(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    entry = make_python_tool(workspace)

    result = await entry.invoke({"code": "print(sum(range(10)))"})
    assert result["ok"] is True
    assert result["stdout"].strip() == "45"


async def test_the_preamble_gives_the_snippet_the_usual_imports(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    entry = make_python_tool(workspace)
    result = await entry.invoke(
        {"code": "print(json.dumps({'root': math.sqrt(16)}))"})
    assert json.loads(result["stdout"])["root"] == 4.0


async def test_a_snippet_that_raises_reports_the_error_without_killing_the_run(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    entry = make_python_tool(workspace)
    result = await entry.invoke({"code": "raise ValueError('nope')"})
    assert result["ok"] is False
    assert "ValueError" in result["error"] and "nope" in result["error"]


async def test_the_snippet_starts_inside_the_workspace(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    workspace.write("data.txt", "hello from the workspace")
    entry = make_python_tool(workspace)
    result = await entry.invoke(
        {"code": "print(Path('data.txt').read_text())"})
    assert "hello from the workspace" in result["stdout"]


async def test_a_runaway_snippet_is_killed(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    entry = make_python_tool(workspace)
    with pytest.raises(ToolError, match="timed out"):
        await entry.invoke({"code": "import time; time.sleep(30)",
                            "timeout_s": 0.3})


async def test_running_code_always_asks_first(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    assert make_python_tool(workspace).permission == "ask"


def test_compute_without_a_workspace_is_refused():
    with pytest.raises(ToolError, match="needs a workspace"):
        make_python_tool(None)


async def test_empty_code_is_refused(tmp_path):
    workspace = Workspace(tmp_path / "ws", allow_shell=True)
    with pytest.raises(ToolError, match="no code"):
        await make_python_tool(workspace).invoke({"code": "   "})


def _texts(svg: str) -> list[tuple[float, float, str]]:
    """Every text element as (x, y, content)."""
    root = ET.fromstring(svg)
    return [(float(e.get("x", 0)), float(e.get("y", 0)), (e.text or ""))
            for e in root.iter() if e.tag.endswith("text")]


def test_a_negative_bar_label_does_not_land_on_the_category_label():
    """Regression: the lowest bar used to reach the plot floor and collide."""
    svg = bar_chart({"Jan": 42000, "Feb": 51500, "May": -8200}, title="Revenue")
    labels = {text: (x, y) for x, y, text in _texts(svg)}
    value_y = labels["-8200"][1]
    category_y = labels["May"][1]
    assert value_y < category_y - 8, "the value label overlaps the category label"


def test_no_mark_or_label_escapes_the_canvas():
    svg = bar_chart({"a": 100, "b": -100, "c": 5}, title="Mixed")
    root = ET.fromstring(svg)
    width, height = 640, 360
    for x, y, _ in _texts(svg):
        assert 0 <= x <= width and 0 <= y <= height
    for rect in (e for e in root.iter() if e.tag.endswith("rect")):
        top = float(rect.get("y", 0))
        assert top >= 0 and top + float(rect.get("height", 0)) <= height


def test_line_marks_stay_off_the_plot_edges():
    svg = line_chart({"s": [0, 50, 100]})
    root = ET.fromstring(svg)
    ys = [float(c.get("cy")) for c in root.iter() if c.tag.endswith("circle")]
    assert min(ys) > 24, "the highest point sits on the plot ceiling"
    assert max(ys) < 360 - 24, "the lowest point sits on the plot floor"
