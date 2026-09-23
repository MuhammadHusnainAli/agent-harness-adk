"""Chart and report rendering — inline SVG, no dependencies.

An agent that has done the analysis usually needs to *show* it. These render to
a standalone SVG string an agent can write to a file, drop in a report or hand
back as an artefact.

The palette is a validated categorical set: hues assigned in fixed order (never
cycled), stepped separately for the light and dark surfaces, and checked for
colour-vision separation on adjacent pairs. Text never wears a series colour, a
legend is always present once there are two or more series, and values are
direct-labelled — which is also what keeps the lighter light-mode hues legible.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ToolError
from ..tools import Tool, tool

__all__ = ["bar_chart", "line_chart", "markdown_table", "render_report",
           "make_chart_tool", "make_report_tool", "SERIES_LIGHT", "SERIES_DARK"]

# Fixed order — slot 1 is always blue, slot 2 always orange, and so on. Cycling
# these or reordering them per chart is what makes a set of charts unreadable.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

_STYLE = """
.viz {{ font: 13px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
        --ink: #0b0b0b; --ink-2: #52514e; --grid: #e6e5e1; --surface: #fcfcfb;
{light} }}
@media (prefers-color-scheme: dark) {{
  .viz {{ --ink: #ffffff; --ink-2: #c3c2b7; --grid: #333330; --surface: #1a1a19;
{dark} }}
}}
.viz-surface {{ fill: var(--surface); }}
.viz-title {{ font-size: 15px; font-weight: 600; fill: var(--ink); }}
.viz-sub {{ font-size: 12px; fill: var(--ink-2); }}
.viz-label {{ font-size: 11px; fill: var(--ink-2); }}
.viz-value {{ font-size: 11px; font-weight: 600; fill: var(--ink); }}
.viz-grid {{ stroke: var(--grid); stroke-width: 1; }}
.viz-axis {{ stroke: var(--grid); stroke-width: 1.5; }}
.viz-line {{ fill: none; stroke-width: 2; stroke-linecap: round;
             stroke-linejoin: round; }}
.viz-dot {{ stroke: var(--surface); stroke-width: 2; }}
"""


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _nice(value: float) -> str:
    """Short, readable numbers — 1.2k rather than 1200.0."""
    if value == int(value) and abs(value) < 10_000:
        return str(int(value))
    for limit, suffix in ((1e9, "b"), (1e6, "m"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Round tick values covering the range."""
    if high <= low:
        return [low]
    span = high - low
    raw = span / count
    magnitude = 10 ** (len(str(int(abs(raw)))) - 1) if abs(raw) >= 1 else 0.1
    step = max(round(raw / magnitude) * magnitude, magnitude)
    ticks, value = [], (int(low / step)) * step
    while value <= high + step / 2 and len(ticks) < count + 2:
        if value >= low - step / 2:
            ticks.append(round(value, 6))
        value += step
    return ticks or [low, high]


def _palette_vars() -> tuple[str, str]:
    light = "\n".join(f"        --s{i + 1}: {c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "\n".join(f"          --s{i + 1}: {c};" for i, c in enumerate(SERIES_DARK))
    return light, dark


def _frame(width: int, height: int, body: str) -> str:
    light, dark = _palette_vars()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" class="viz">'
        f"<style>{_STYLE.format(light=light, dark=dark)}</style>"
        f'<rect class="viz-surface" width="{width}" height="{height}" rx="8"/>'
        f"{body}</svg>"
    )


def _legend(names: Sequence[str], x: float, y: float) -> str:
    """Always present for two or more series — identity is never colour alone."""
    if len(names) < 2:
        return ""
    out, offset = [], 0.0
    for index, name in enumerate(names):
        colour = f"var(--s{(index % 8) + 1})"
        out.append(
            f'<rect x="{x + offset:.1f}" y="{y - 8}" width="10" height="10" rx="2" '
            f'fill="{colour}"/>'
            f'<text class="viz-label" x="{x + offset + 15:.1f}" y="{y + 1}">'
            f"{_esc(name)}</text>"
        )
        offset += 25 + len(str(name)) * 6.5
    return "".join(out)


def bar_chart(
    data: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    title: str = "",
    subtitle: str = "",
    width: int = 640,
    height: int = 360,
    series_index: int = 0,
) -> str:
    """A bar chart as a standalone SVG string.

    Bars are for comparing magnitude across categories, so the scale starts at
    zero — a truncated bar axis misstates the comparison it exists to make.
    """
    items = list(data.items()) if isinstance(data, Mapping) else list(data)
    if not items:
        raise ToolError("bar_chart needs at least one value", tool="bar_chart")
    for label, value in items:
        if not isinstance(value, (int, float)):
            raise ToolError(f"{label!r} is not a number: {value!r}", tool="bar_chart")

    top = 56 if title else 24
    left, right, bottom = 52, 20, 52
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [float(v) for _, v in items]
    high = max(max(values), 0.0)
    low = min(min(values), 0.0)
    if high == low:
        high = low + 1
    ticks = _ticks(low, high, 4)
    floor, ceiling = min(ticks + [low]), max(ticks + [high])
    # Leave headroom so a bar never touches the plot edge — without it the value
    # label on the lowest bar lands on top of the category labels.
    pad = (ceiling - floor) * 0.09 or 1.0
    base = floor - (pad if low < 0 else 0.0)
    span = (ceiling + pad * 0.6 - base) or 1

    def y_of(value: float) -> float:
        return top + plot_h - (value - base) / span * plot_h

    parts: list[str] = []
    if title:
        parts.append(f'<text class="viz-title" x="{left}" y="26">{_esc(title)}</text>')
    if subtitle:
        parts.append(f'<text class="viz-sub" x="{left}" y="43">{_esc(subtitle)}</text>')

    for tick in ticks:
        y = y_of(tick)
        parts.append(f'<line class="viz-grid" x1="{left}" y1="{y:.1f}" '
                     f'x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="viz-label" x="{left - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_esc(_nice(tick))}</text>')

    colour = f"var(--s{(series_index % 8) + 1})"
    slot = plot_w / len(items)
    gap = min(slot * 0.3, 24)
    bar_w = max(slot - gap, 2)          # the gap doubles as the 2px surface spacer
    zero_y = y_of(0.0)

    for index, (label, value) in enumerate(items):
        value = float(value)
        x = left + index * slot + gap / 2
        y = y_of(value)
        bar_h = abs(zero_y - y)
        radius = min(4, bar_w / 2, max(bar_h, 1))
        parts.append(
            f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bar_w:.1f}" '
            f'height="{max(bar_h, 1):.1f}" rx="{radius:.1f}" fill="{colour}"/>'
        )
        # Direct value labels: the relief rule for lighter hues, and they save a
        # reader from measuring against the gridlines. Clamped inside the plot so
        # they never collide with the title above or the category labels below.
        label_y = (y - 6) if value >= 0 else (y + 14)
        label_y = min(max(label_y, top + 10), top + plot_h + 2)
        parts.append(f'<text class="viz-value" x="{x + bar_w / 2:.1f}" '
                     f'y="{label_y:.1f}" text-anchor="middle">'
                     f"{_esc(_nice(value))}</text>")
        text = str(label)
        if len(text) * 6 > slot:
            text = text[: max(int(slot / 6) - 1, 3)] + "…"
        parts.append(f'<text class="viz-label" x="{x + bar_w / 2:.1f}" '
                     f'y="{top + plot_h + 18:.1f}" text-anchor="middle">'
                     f"{_esc(text)}</text>")

    parts.append(f'<line class="viz-axis" x1="{left}" y1="{zero_y:.1f}" '
                 f'x2="{left + plot_w}" y2="{zero_y:.1f}"/>')
    return _frame(width, height, "".join(parts))


def line_chart(
    series: Mapping[str, Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    title: str = "",
    subtitle: str = "",
    width: int = 640,
    height: int = 360,
) -> str:
    """One or more lines over a shared x axis, as a standalone SVG string.

    Every series shares one y scale on purpose. Two measures of different
    magnitude belong in two charts, not on two axes.
    """
    rows = {str(k): [float(v) for v in vals] for k, vals in series.items() if len(vals)}
    if not rows:
        raise ToolError("line_chart needs at least one non-empty series",
                        tool="line_chart")
    if len(rows) > 8:
        raise ToolError("more than 8 series cannot be told apart — fold the tail "
                        "into 'Other' or split the chart", tool="line_chart")

    length = max(len(v) for v in rows.values())
    top = 56 if title else 24
    left, right, bottom = 52, 72, 56
    plot_w = width - left - right
    plot_h = height - top - bottom

    flat = [v for vals in rows.values() for v in vals]
    high, low = max(flat), min(flat)
    if high == low:
        high, low = high + 1, low - 1
    ticks = _ticks(low, high, 4)
    floor, ceiling = min(ticks + [low]), max(ticks + [high])
    pad = (ceiling - floor) * 0.08 or 1.0   # keep marks off the plot edges
    base = floor - pad
    span = (ceiling + pad - base) or 1

    def x_of(index: int) -> float:
        return left + (index / max(length - 1, 1)) * plot_w

    def y_of(value: float) -> float:
        return top + plot_h - (value - base) / span * plot_h

    parts: list[str] = []
    if title:
        parts.append(f'<text class="viz-title" x="{left}" y="26">{_esc(title)}</text>')
    if subtitle:
        parts.append(f'<text class="viz-sub" x="{left}" y="43">{_esc(subtitle)}</text>')

    for tick in ticks:
        y = y_of(tick)
        parts.append(f'<line class="viz-grid" x1="{left}" y1="{y:.1f}" '
                     f'x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="viz-label" x="{left - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{_esc(_nice(tick))}</text>')

    for index, (name, values) in enumerate(rows.items()):
        colour = f"var(--s{(index % 8) + 1})"
        points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline class="viz-line" stroke="{colour}" points="{points}"/>')
        if len(values) <= 24:
            for i, value in enumerate(values):
                parts.append(f'<circle class="viz-dot" cx="{x_of(i):.1f}" '
                             f'cy="{y_of(value):.1f}" r="4" fill="{colour}"/>')
        # Direct-label the end of each line when there are few enough to fit.
        if len(rows) <= 4:
            end_x, end_y = x_of(len(values) - 1), y_of(values[-1])
            parts.append(f'<text class="viz-value" x="{end_x + 8:.1f}" '
                         f'y="{end_y + 4:.1f}">{_esc(name)}</text>')

    if labels:
        stride = max(1, len(labels) // 8)
        for index, label in enumerate(labels[:length]):
            if index % stride:
                continue
            parts.append(f'<text class="viz-label" x="{x_of(index):.1f}" '
                         f'y="{top + plot_h + 20:.1f}" text-anchor="middle">'
                         f"{_esc(label)}</text>")

    parts.append(f'<line class="viz-axis" x1="{left}" y1="{top + plot_h:.1f}" '
                 f'x2="{left + plot_w}" y2="{top + plot_h:.1f}"/>')
    parts.append(_legend(list(rows), left, height - 14))
    return _frame(width, height, "".join(parts))


def markdown_table(rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[Any]],
                   headers: Sequence[str] | None = None) -> str:
    """A markdown table — the text view every chart should have alongside it."""
    if not rows:
        return "(no rows)"
    if isinstance(rows[0], Mapping):
        headers = list(headers or rows[0].keys())
        body = [[str(row.get(h, "")) for h in headers] for row in rows]  # type: ignore[union-attr]
    else:
        body = [[str(cell) for cell in row] for row in rows]  # type: ignore[union-attr]
        headers = list(headers or [f"col{i + 1}" for i in range(len(body[0]))])
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    rule = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    out = [line, rule]
    out += ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |"
            for r in body]
    return "\n".join(out)


def render_report(title: str, sections: Sequence[tuple[str, str]], *,
                  summary: str = "") -> str:
    """Assemble a markdown report: heading, lead, then each section."""
    out = [f"# {title}"]
    if summary:
        out.append(summary.strip())
    for heading, body in sections:
        if body and body.strip():
            out.append(f"## {heading}\n\n{body.strip()}")
    return "\n\n".join(out) + "\n"


def make_chart_tool(workspace: Any = None, *, name: str = "render_chart") -> Tool:
    """A chart tool for an agent. With a workspace, it writes the file too."""

    @tool(name=name, tags=["builtin", "chart"])
    def render_chart(kind: str, title: str, data: dict[str, Any],
                     path: str = "", labels: list[str] | None = None,
                     subtitle: str = "") -> str:
        """Render a chart as SVG.

        Args:
            kind: "bar" for comparing categories, "line" for change over time.
            title: what the chart shows — a sentence, not a label.
            data: for "bar", {category: number}. For "line", {series: [numbers]}.
            path: where to save it in the workspace. Omit to get the SVG back.
            labels: x-axis labels, for "line" only.
            subtitle: an optional second line under the title.
        """
        if kind == "bar":
            svg = bar_chart(data, title=title, subtitle=subtitle)
        elif kind == "line":
            svg = line_chart({k: list(v) for k, v in data.items()}, labels=labels,
                             title=title, subtitle=subtitle)
        else:
            raise ToolError(f"unknown chart kind {kind!r}; use 'bar' or 'line'",
                            tool=name)
        if path and workspace is not None:
            workspace.write(path, svg)
            return f"wrote {path} ({len(svg)} bytes)"
        return svg

    return render_chart


def make_report_tool(workspace: Any = None, *, name: str = "write_report") -> Tool:
    """A report tool for an agent. With a workspace, it writes the file too."""

    @tool(name=name, tags=["builtin", "report"])
    def write_report(title: str, sections: list[dict[str, str]], summary: str = "",
                     path: str = "") -> str:
        """Assemble a markdown report.

        Args:
            title: the report's title.
            sections: [{"heading": ..., "body": ...}] in the order they appear.
            summary: the lead paragraph — the answer, before the evidence.
            path: where to save it in the workspace. Omit to get the text back.
        """
        pairs = [(s.get("heading", ""), s.get("body", "")) for s in sections]
        text = render_report(title, pairs, summary=summary)
        if path and workspace is not None:
            workspace.write(path, text)
            return f"wrote {path} ({len(text)} characters)"
        return text

    return write_report
