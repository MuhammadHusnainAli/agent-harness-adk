//! Chart and report rendering — inline SVG, no dependencies.
//!
//! An agent that has done the analysis usually needs to *show* it. These
//! render to a standalone SVG string an agent can write to a file, drop in a
//! report or hand back as an artefact.
//!
//! The palette is a validated categorical set: hues assigned in fixed order
//! (never cycled), stepped separately for the light and dark surfaces, and
//! checked for colour-vision separation on adjacent pairs. Text never wears a
//! series colour, a legend is always present once there are two or more
//! series, and values are direct-labelled — which is also what keeps the
//! lighter light-mode hues legible.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::governance::{py_repr, py_str, py_str_repr};
use crate::runtime::workspace::Workspace;
use crate::tools::{Tool, ToolContext};

/// Fixed order — slot 1 is always blue, slot 2 always orange, and so on.
/// Cycling these or reordering them per chart is what makes a set of charts unreadable.
pub const SERIES_LIGHT: [&str; 8] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
pub const SERIES_DARK: [&str; 8] = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

const STYLE_HEAD: &str = "\n.viz { font: 13px ui-sans-serif, system-ui, -apple-system, \"Segoe UI\", sans-serif;\n        --ink: #0b0b0b; --ink-2: #52514e; --grid: #e6e5e1; --surface: #fcfcfb;\n";
const STYLE_DARK: &str = " }\n@media (prefers-color-scheme: dark) {\n  .viz { --ink: #ffffff; --ink-2: #c3c2b7; --grid: #333330; --surface: #1a1a19;\n";
const STYLE_TAIL: &str = " }\n}\n.viz-surface { fill: var(--surface); }\n.viz-title { font-size: 15px; font-weight: 600; fill: var(--ink); }\n.viz-sub { font-size: 12px; fill: var(--ink-2); }\n.viz-label { font-size: 11px; fill: var(--ink-2); }\n.viz-value { font-size: 11px; font-weight: 600; fill: var(--ink); }\n.viz-grid { stroke: var(--grid); stroke-width: 1; }\n.viz-axis { stroke: var(--grid); stroke-width: 1.5; }\n.viz-line { fill: none; stroke-width: 2; stroke-linecap: round;\n             stroke-linejoin: round; }\n.viz-dot { stroke: var(--surface); stroke-width: 2; }\n";

/// `html.escape(str(text), quote=True)`.
fn esc(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

fn strip_zeros(s: String) -> String {
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Short, readable numbers — 1.2k rather than 1200.0.
fn nice(value: f64) -> String {
    if value == value.trunc() && value.abs() < 10_000.0 {
        return format!("{}", value as i64);
    }
    for (limit, suffix) in [(1e9, "b"), (1e6, "m"), (1e3, "k")] {
        if value.abs() >= limit {
            return strip_zeros(format!("{:.1}", value / limit)) + suffix;
        }
    }
    strip_zeros(format!("{value:.2}"))
}

/// Python's `round(value, 6)`, near enough for tick values.
fn round6(value: f64) -> f64 {
    (value * 1e6).round_ties_even() / 1e6
}

/// Round tick values covering the range.
fn ticks(low: f64, high: f64, count: usize) -> Vec<f64> {
    if high <= low {
        return vec![low];
    }
    let raw = (high - low) / count as f64;
    let magnitude = if raw.abs() >= 1.0 { 10f64.powi(format!("{}", raw.abs().trunc() as i64).len() as i32 - 1) } else { 0.1 };
    let step = ((raw / magnitude).round_ties_even() * magnitude).max(magnitude);
    let mut out = Vec::new();
    let mut value = (low / step).trunc() * step;
    while value <= high + step / 2.0 && out.len() < count + 2 {
        if value >= low - step / 2.0 {
            out.push(round6(value));
        }
        value += step;
    }
    if out.is_empty() { vec![low, high] } else { out }
}

fn palette_vars() -> (String, String) {
    let light = SERIES_LIGHT.iter().enumerate().map(|(i, c)| format!("        --s{}: {c};", i + 1)).collect::<Vec<_>>().join("\n");
    let dark = SERIES_DARK.iter().enumerate().map(|(i, c)| format!("          --s{}: {c};", i + 1)).collect::<Vec<_>>().join("\n");
    (light, dark)
}

fn frame(width: u32, height: u32, body: &str) -> String {
    let (light, dark) = palette_vars();
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" width=\"{width}\" height=\"{height}\" role=\"img\" class=\"viz\"><style>{STYLE_HEAD}{light}{STYLE_DARK}{dark}{STYLE_TAIL}</style><rect class=\"viz-surface\" width=\"{width}\" height=\"{height}\" rx=\"8\"/>{body}</svg>"
    )
}

/// Always present for two or more series — identity is never colour alone.
fn legend(names: &[String], x: f64, y: i64) -> String {
    if names.len() < 2 {
        return String::new();
    }
    let mut out = String::new();
    let mut offset = 0.0;
    for (index, name) in names.iter().enumerate() {
        let colour = format!("var(--s{})", (index % 8) + 1);
        out.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{}\" width=\"10\" height=\"10\" rx=\"2\" fill=\"{colour}\"/><text class=\"viz-label\" x=\"{:.1}\" y=\"{}\">{}</text>",
            x + offset,
            y - 8,
            x + offset + 15.0,
            y + 1,
            esc(name)
        ));
        offset += 25.0 + name.chars().count() as f64 * 6.5;
    }
    out
}

/// Title, subtitle, size and the rest of a chart's dressing.
#[derive(Debug, Clone)]
pub struct ChartOptions {
    pub title: String,
    pub subtitle: String,
    pub width: u32,
    pub height: u32,
    /// For a bar chart: which palette slot its bars wear.
    pub series_index: usize,
    /// For a line chart: the x-axis labels.
    pub labels: Vec<String>,
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self { title: String::new(), subtitle: String::new(), width: 640, height: 360, series_index: 0, labels: Vec::new() }
    }
}

fn header(parts: &mut Vec<String>, opts: &ChartOptions, left: i64) {
    if !opts.title.is_empty() {
        parts.push(format!("<text class=\"viz-title\" x=\"{left}\" y=\"26\">{}</text>", esc(&opts.title)));
    }
    if !opts.subtitle.is_empty() {
        parts.push(format!("<text class=\"viz-sub\" x=\"{left}\" y=\"43\">{}</text>", esc(&opts.subtitle)));
    }
}

fn grid(parts: &mut Vec<String>, tick_values: &[f64], left: i64, plot_w: f64, y_of: impl Fn(f64) -> f64) {
    for tick in tick_values {
        let y = y_of(*tick);
        parts.push(format!("<line class=\"viz-grid\" x1=\"{left}\" y1=\"{y:.1}\" x2=\"{}\" y2=\"{y:.1}\"/>", fmt_num(left as f64 + plot_w)));
        parts.push(format!("<text class=\"viz-label\" x=\"{}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>", left - 8, y + 4.0, esc(&nice(*tick))));
    }
}

/// How Python's f-string prints `left + plot_w`: an int stays an int.
fn fmt_num(v: f64) -> String {
    if v == v.trunc() { format!("{}", v as i64) } else { crate::pyjson::float_repr(v) }
}

/// A bar chart as a standalone SVG string.
///
/// Bars are for comparing magnitude across categories, so the scale starts at
/// zero — a truncated bar axis misstates the comparison it exists to make.
pub fn bar_chart<S: AsRef<str>>(data: &[(S, f64)], opts: &ChartOptions) -> Result<String> {
    if data.is_empty() {
        return Err(Error::tool("bar_chart needs at least one value", "bar_chart"));
    }
    let (width, height) = (opts.width as i64, opts.height as i64);
    let top: i64 = if opts.title.is_empty() { 24 } else { 56 };
    let (left, right, bottom) = (52i64, 20i64, 52i64);
    let plot_w = (width - left - right) as f64;
    let plot_h = (height - top - bottom) as f64;

    let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
    let mut high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(0.0);
    let low = values.iter().copied().fold(f64::INFINITY, f64::min).min(0.0);
    if high == low {
        high = low + 1.0;
    }
    let tick_values = ticks(low, high, 4);
    let floor = tick_values.iter().copied().fold(low, f64::min);
    let ceiling = tick_values.iter().copied().fold(high, f64::max);
    // Leave headroom so a bar never touches the plot edge — without it the
    // value label on the lowest bar lands on top of the category labels.
    let pad = match (ceiling - floor) * 0.09 {
        0.0 => 1.0,
        p => p,
    };
    let base = floor - if low < 0.0 { pad } else { 0.0 };
    let span = match ceiling + pad * 0.6 - base {
        0.0 => 1.0,
        s => s,
    };
    let y_of = |value: f64| top as f64 + plot_h - (value - base) / span * plot_h;

    let mut parts = Vec::new();
    header(&mut parts, opts, left);
    grid(&mut parts, &tick_values, left, plot_w, y_of);

    let colour = format!("var(--s{})", (opts.series_index % 8) + 1);
    let slot = plot_w / data.len() as f64;
    let gap = (slot * 0.3).min(24.0);
    let bar_w = (slot - gap).max(2.0); // the gap doubles as the 2px surface spacer
    let zero_y = y_of(0.0);

    for (index, (label, value)) in data.iter().enumerate() {
        let value = *value;
        let x = left as f64 + index as f64 * slot + gap / 2.0;
        let y = y_of(value);
        let bar_h = (zero_y - y).abs();
        let radius = 4f64.min(bar_w / 2.0).min(bar_h.max(1.0));
        parts.push(format!(
            "<rect x=\"{x:.1}\" y=\"{:.1}\" width=\"{bar_w:.1}\" height=\"{:.1}\" rx=\"{radius:.1}\" fill=\"{colour}\"/>",
            y.min(zero_y),
            bar_h.max(1.0)
        ));
        // Direct value labels, clamped inside the plot so they never collide
        // with the title above or the category labels below.
        let label_y = if value >= 0.0 { y - 6.0 } else { y + 14.0 };
        let label_y = label_y.max(top as f64 + 10.0).min(top as f64 + plot_h + 2.0);
        parts.push(format!("<text class=\"viz-value\" x=\"{:.1}\" y=\"{label_y:.1}\" text-anchor=\"middle\">{}</text>", x + bar_w / 2.0, esc(&nice(value))));
        let mut text = label.as_ref().to_string();
        if text.chars().count() as f64 * 6.0 > slot {
            let keep = ((slot / 6.0) as i64 - 1).max(3) as usize;
            text = text.chars().take(keep).collect::<String>() + "\u{2026}";
        }
        parts.push(format!(
            "<text class=\"viz-label\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            x + bar_w / 2.0,
            top as f64 + plot_h + 18.0,
            esc(&text)
        ));
    }
    parts.push(format!("<line class=\"viz-axis\" x1=\"{left}\" y1=\"{zero_y:.1}\" x2=\"{}\" y2=\"{zero_y:.1}\"/>", fmt_num(left as f64 + plot_w)));
    Ok(frame(opts.width, opts.height, &parts.concat()))
}

/// One or more lines over a shared x axis, as a standalone SVG string.
///
/// Every series shares one y scale on purpose. Two measures of different
/// magnitude belong in two charts, not on two axes.
pub fn line_chart<S: AsRef<str>>(series: &[(S, Vec<f64>)], opts: &ChartOptions) -> Result<String> {
    let rows: Vec<(String, &Vec<f64>)> = series.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| (k.as_ref().to_string(), v)).collect();
    if rows.is_empty() {
        return Err(Error::tool("line_chart needs at least one non-empty series", "line_chart"));
    }
    if rows.len() > 8 {
        return Err(Error::tool("more than 8 series cannot be told apart \u{2014} fold the tail into 'Other' or split the chart", "line_chart"));
    }
    let length = rows.iter().map(|(_, v)| v.len()).max().unwrap_or(1);
    let (width, height) = (opts.width as i64, opts.height as i64);
    let top: i64 = if opts.title.is_empty() { 24 } else { 56 };
    let (left, right, bottom) = (52i64, 72i64, 56i64);
    let plot_w = (width - left - right) as f64;
    let plot_h = (height - top - bottom) as f64;

    let flat: Vec<f64> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let mut high = flat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut low = flat.iter().copied().fold(f64::INFINITY, f64::min);
    if high == low {
        (high, low) = (high + 1.0, low - 1.0);
    }
    let tick_values = ticks(low, high, 4);
    let floor = tick_values.iter().copied().fold(low, f64::min);
    let ceiling = tick_values.iter().copied().fold(high, f64::max);
    let pad = match (ceiling - floor) * 0.08 {
        0.0 => 1.0,
        p => p,
    }; // keep marks off the plot edges
    let base = floor - pad;
    let span = match ceiling + pad - base {
        0.0 => 1.0,
        s => s,
    };
    let x_of = |index: usize| left as f64 + (index as f64 / (length.max(2) - 1) as f64) * plot_w;
    let y_of = |value: f64| top as f64 + plot_h - (value - base) / span * plot_h;

    let mut parts = Vec::new();
    header(&mut parts, opts, left);
    grid(&mut parts, &tick_values, left, plot_w, y_of);

    for (index, (name, values)) in rows.iter().enumerate() {
        let colour = format!("var(--s{})", (index % 8) + 1);
        let points = values.iter().enumerate().map(|(i, v)| format!("{:.1},{:.1}", x_of(i), y_of(*v))).collect::<Vec<_>>().join(" ");
        parts.push(format!("<polyline class=\"viz-line\" stroke=\"{colour}\" points=\"{points}\"/>"));
        if values.len() <= 24 {
            for (i, value) in values.iter().enumerate() {
                parts.push(format!("<circle class=\"viz-dot\" cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"{colour}\"/>", x_of(i), y_of(*value)));
            }
        }
        // Direct-label the end of each line when there are few enough to fit.
        if rows.len() <= 4 {
            let (end_x, end_y) = (x_of(values.len() - 1), y_of(values[values.len() - 1]));
            parts.push(format!("<text class=\"viz-value\" x=\"{:.1}\" y=\"{:.1}\">{}</text>", end_x + 8.0, end_y + 4.0, esc(name)));
        }
    }

    if !opts.labels.is_empty() {
        let stride = (opts.labels.len() / 8).max(1);
        for (index, label) in opts.labels.iter().take(length).enumerate() {
            if index % stride != 0 {
                continue;
            }
            parts.push(format!(
                "<text class=\"viz-label\" x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
                x_of(index),
                top as f64 + plot_h + 20.0,
                esc(label)
            ));
        }
    }

    parts.push(format!(
        "<line class=\"viz-axis\" x1=\"{left}\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\"/>",
        top as f64 + plot_h,
        fmt_num(left as f64 + plot_w),
        top as f64 + plot_h
    ));
    let names: Vec<String> = rows.iter().map(|(n, _)| n.clone()).collect();
    parts.push(legend(&names, left as f64, height - 14));
    Ok(frame(opts.width, opts.height, &parts.concat()))
}

/// A markdown table — the text view every chart should have alongside it.
///
/// `rows` are JSON objects (headers from the first row's keys unless given)
/// or arrays (headers `col1`, `col2`, ... unless given).
pub fn markdown_table(rows: &[Value], headers: Option<&[String]>) -> String {
    if rows.is_empty() {
        return "(no rows)".into();
    }
    let (headers, body): (Vec<String>, Vec<Vec<String>>) = if let Some(first) = rows[0].as_object() {
        let headers: Vec<String> = headers.map(<[String]>::to_vec).unwrap_or_else(|| first.keys().cloned().collect());
        let body = rows.iter().map(|r| headers.iter().map(|h| r.get(h).map(py_str).unwrap_or_default()).collect()).collect();
        (headers, body)
    } else {
        let body: Vec<Vec<String>> = rows.iter().map(|r| r.as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_else(|| vec![py_str(r)])).collect();
        let headers = headers.map(<[String]>::to_vec).unwrap_or_else(|| (1..=body[0].len()).map(|i| format!("col{i}")).collect());
        (headers, body)
    };
    let len = |s: &str| s.chars().count();
    let cell = |r: &Vec<String>, i: usize| r.get(i).cloned().unwrap_or_default();
    let widths: Vec<usize> = headers.iter().enumerate().map(|(i, h)| body.iter().map(|r| len(&cell(r, i))).max().unwrap_or(0).max(len(h))).collect();
    let pad = |s: &str, w: usize| format!("{s}{}", " ".repeat(w.saturating_sub(len(s))));
    let mut out = vec![
        format!("| {} |", headers.iter().enumerate().map(|(i, h)| pad(h, widths[i])).collect::<Vec<_>>().join(" | ")),
        format!("|-{}-|", widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>().join("-|-")),
    ];
    for r in &body {
        out.push(format!("| {} |", (0..headers.len()).map(|i| pad(&cell(r, i), widths[i])).collect::<Vec<_>>().join(" | ")));
    }
    out.join("\n")
}

/// Assemble a markdown report: heading, lead, then each non-empty section.
pub fn render_report<H: AsRef<str>, B: AsRef<str>>(title: &str, sections: &[(H, B)], summary: &str) -> String {
    let mut out = vec![format!("# {title}")];
    if !summary.is_empty() {
        out.push(summary.trim().to_string());
    }
    for (heading, body) in sections {
        let body = body.as_ref().trim();
        if !body.is_empty() {
            out.push(format!("## {}\n\n{body}", heading.as_ref()));
        }
    }
    out.join("\n\n") + "\n"
}

// ---------------------------------------------------------------------------
// tools
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct ChartArgs {
    /// "bar" for comparing categories, "line" for change over time.
    kind: String,
    /// what the chart shows — a sentence, not a label.
    title: String,
    /// for "bar", {category: number}. For "line", {series: [numbers]}.
    data: Map<String, Value>,
    /// where to save it in the workspace. Omit to get the SVG back.
    #[serde(default)]
    path: String,
    /// x-axis labels, for "line" only.
    #[serde(default)]
    labels: Option<Vec<String>>,
    /// an optional second line under the title.
    #[serde(default)]
    subtitle: String,
}

/// A JSON value as a number, as Python accepts one (a bool is an int).
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// A chart tool for an agent, named `render_chart`. With a workspace, it
/// writes the file too.
pub fn make_chart_tool(workspace: Option<Arc<dyn Workspace>>, name: &str) -> Tool {
    let tool_name = name.to_string();
    Tool::new(name, move |args: ChartArgs, _ctx: ToolContext| {
        let (workspace, name) = (workspace.clone(), tool_name.clone());
        async move {
            let opts = ChartOptions { title: args.title, subtitle: args.subtitle, labels: args.labels.unwrap_or_default(), ..Default::default() };
            let svg = match args.kind.as_str() {
                "bar" => {
                    let mut data = Vec::new();
                    for (label, value) in &args.data {
                        let n = number(value).ok_or_else(|| Error::tool(format!("{} is not a number: {}", py_str_repr(label), py_repr(value)), "bar_chart"))?;
                        data.push((label.clone(), n));
                    }
                    bar_chart(&data, &opts)?
                }
                "line" => {
                    let mut series = Vec::new();
                    for (label, values) in &args.data {
                        let numbers: Option<Vec<f64>> = values.as_array().and_then(|a| a.iter().map(number).collect());
                        let numbers =
                            numbers.ok_or_else(|| Error::tool(format!("{name} failed: series {} is not a list of numbers", py_str_repr(label)), &name))?;
                        series.push((label.clone(), numbers));
                    }
                    line_chart(&series, &opts)?
                }
                other => return Err(Error::tool(format!("unknown chart kind {}; use 'bar' or 'line'", py_str_repr(other)), &name)),
            };
            if let (false, Some(ws)) = (args.path.is_empty(), &workspace) {
                ws.write(&args.path, svg.as_bytes(), false).await?;
                return Ok(format!("wrote {} ({} bytes)", args.path, svg.chars().count()));
            }
            Ok(svg)
        }
    })
    .describe("Render a chart as SVG.")
    .tags(["builtin", "chart"])
}

#[derive(Deserialize, JsonSchema)]
struct Section {
    #[serde(default)]
    heading: String,
    #[serde(default)]
    body: String,
}

#[derive(Deserialize, JsonSchema)]
struct ReportArgs {
    /// the report's title.
    title: String,
    /// [{"heading": ..., "body": ...}] in the order they appear.
    sections: Vec<Section>,
    /// the lead paragraph — the answer, before the evidence.
    #[serde(default)]
    summary: String,
    /// where to save it in the workspace. Omit to get the text back.
    #[serde(default)]
    path: String,
}

/// A report tool for an agent, named `write_report`. With a workspace, it
/// writes the file too.
pub fn make_report_tool(workspace: Option<Arc<dyn Workspace>>, name: &str) -> Tool {
    Tool::new(name, move |args: ReportArgs, _ctx: ToolContext| {
        let workspace = workspace.clone();
        async move {
            let pairs: Vec<(String, String)> = args.sections.into_iter().map(|s| (s.heading, s.body)).collect();
            let text = render_report(&args.title, &pairs, &args.summary);
            if let (false, Some(ws)) = (args.path.is_empty(), &workspace) {
                ws.write(&args.path, text.as_bytes(), false).await?;
                return Ok(format!("wrote {} ({} characters)", args.path, text.chars().count()));
            }
            Ok(text)
        }
    })
    .describe("Assemble a markdown report.")
    .tags(["builtin", "report"])
}

#[cfg(test)]
mod tests {
    use regex::Regex;
    use serde_json::json;

    use super::*;
    use crate::runtime::workspace::{LocalWorkspace, WorkspaceOptions};

    fn opts(title: &str) -> ChartOptions {
        ChartOptions { title: title.into(), ..Default::default() }
    }

    /// Every `<tag ...>` element's attributes, in order.
    fn elements(svg: &str, tag: &str) -> Vec<Map<String, Value>> {
        let el = Regex::new(&format!(r"<{tag}\b([^>]*)>")).unwrap();
        let attr = Regex::new(r#"([\w-]+)="([^"]*)""#).unwrap();
        el.captures_iter(svg).map(|c| attr.captures_iter(&c[1]).map(|a| (a[1].to_string(), json!(a[2].to_string()))).collect()).collect()
    }

    fn f(m: &Map<String, Value>, k: &str) -> f64 {
        m.get(k).and_then(Value::as_str).unwrap_or("0").parse().unwrap()
    }

    /// Every text element as (x, y, content).
    fn texts(svg: &str) -> Vec<(f64, f64, String)> {
        let re = Regex::new(r#"<text[^>]*\bx="([^"]*)" y="([^"]*)"[^>]*>([^<]*)</text>"#).unwrap();
        re.captures_iter(svg).map(|c| (c[1].parse().unwrap(), c[2].parse().unwrap(), c[3].to_string())).collect()
    }

    #[test]
    fn a_bar_chart_is_svg_with_one_bar_per_value() {
        let svg = bar_chart(&[("jan", 10.0), ("feb", 25.0), ("mar", 18.0)], &opts("Revenue by month")).unwrap();
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert_eq!(elements(&svg, "rect").len(), 4); // one surface plus one per bar
        assert!(svg.contains("Revenue by month"));
        let two = bar_chart(&[("a", 10.0), ("b", 20.0)], &ChartOptions::default()).unwrap();
        assert!(two.contains(">10<") && two.contains(">20<"));
    }

    #[test]
    fn the_bar_baseline_sits_at_zero_and_negatives_go_below_it() {
        let svg = bar_chart(&[("a", 100.0), ("b", 104.0)], &ChartOptions::default()).unwrap();
        let heights: Vec<f64> = elements(&svg, "rect").iter().filter(|r| r["rx"] != "8").map(|r| f(r, "height")).collect();
        let (lo, hi) = heights.iter().fold((f64::MAX, 0.0f64), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(lo / hi > 0.9);
        let svg = bar_chart(&[("gain", 30.0), ("loss", -20.0)], &ChartOptions::default()).unwrap();
        let bars: Vec<_> = elements(&svg, "rect").into_iter().filter(|r| r["rx"] != "8").collect();
        assert_eq!(bars.len(), 2);
        assert!(f(&bars[1], "y") > f(&bars[0], "y"));
    }

    #[test]
    fn a_line_chart_draws_one_polyline_per_series_and_a_legend() {
        let o = ChartOptions { title: "Trend".into(), labels: vec!["jan".into(), "feb".into(), "mar".into()], ..Default::default() };
        let svg = line_chart(&[("revenue", vec![1.0, 2.0, 3.0]), ("costs", vec![3.0, 2.0, 1.0])], &o).unwrap();
        let lines = elements(&svg, "polyline");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["points"].as_str().unwrap().matches(',').count(), 3);
        assert!(svg.contains("revenue") && svg.contains("costs"));
        let three = line_chart(&[("a", vec![1.0, 2.0]), ("b", vec![2.0, 1.0]), ("c", vec![1.0, 1.0])], &ChartOptions::default()).unwrap();
        for s in ["stroke=\"var(--s1)\"", "stroke=\"var(--s2)\"", "stroke=\"var(--s3)\""] {
            assert!(three.contains(s));
        }
        let two = line_chart(&[("a", vec![1.0, 2.0]), ("b", vec![2.0, 1.0])], &ChartOptions::default()).unwrap();
        assert!(!two.contains("var(--s3)\"") && two.contains("var(--s1)"));
    }

    #[test]
    fn both_themes_are_defined_and_labels_are_escaped() {
        let svg = bar_chart(&[("a", 1.0)], &opts("t")).unwrap();
        assert!(svg.contains("prefers-color-scheme: dark"));
        for (light, dark) in SERIES_LIGHT.iter().zip(SERIES_DARK) {
            assert!(svg.contains(light) && svg.contains(dark));
        }
        assert!(svg.contains("--ink: #0b0b0b") && svg.contains("--ink: #ffffff") && svg.contains("fill: var(--ink)"));
        let svg = bar_chart(&[("<script>alert(1)</script>", 5.0)], &ChartOptions::default()).unwrap();
        assert!(!svg.contains("<script>"));
    }

    #[test]
    fn bad_input_is_refused() {
        let many: Vec<(String, Vec<f64>)> = (0..9).map(|i| (i.to_string(), vec![1.0, 2.0])).collect();
        assert!(line_chart(&many, &ChartOptions::default()).unwrap_err().to_string().contains("cannot be told apart"));
        assert!(bar_chart::<&str>(&[], &ChartOptions::default()).is_err());
        assert!(line_chart::<&str>(&[], &ChartOptions::default()).is_err());
    }

    #[test]
    fn labels_and_marks_stay_on_the_canvas() {
        let svg = bar_chart(&[("Jan", 42000.0), ("Feb", 51500.0), ("May", -8200.0)], &opts("Revenue")).unwrap();
        let found: std::collections::HashMap<String, (f64, f64)> = texts(&svg).into_iter().map(|(x, y, t)| (t, (x, y))).collect();
        assert!(found.contains_key("42k") && found.contains_key("51.5k"));
        assert!(found["-8200"].1 < found["May"].1 - 8.0, "the value label overlaps the category label");

        let svg = bar_chart(&[("a", 100.0), ("b", -100.0), ("c", 5.0)], &opts("Mixed")).unwrap();
        for (x, y, _) in texts(&svg) {
            assert!((0.0..=640.0).contains(&x) && (0.0..=360.0).contains(&y));
        }
        for r in elements(&svg, "rect") {
            assert!(f(&r, "y") >= 0.0 && f(&r, "y") + f(&r, "height") <= 360.0);
        }
        let svg = line_chart(&[("s", vec![0.0, 50.0, 100.0])], &ChartOptions::default()).unwrap();
        let ys: Vec<f64> = elements(&svg, "circle").iter().map(|c| f(c, "cy")).collect();
        assert!(ys.iter().all(|y| *y > 24.0 && *y < 336.0));
    }

    #[test]
    fn numbers_ticks_tables_and_reports() {
        assert_eq!(
            (nice(1200.0), nice(12000.0), nice(2_500_000.0), nice(0.125), nice(-8200.0), nice(3.0)),
            ("1200".into(), "12k".into(), "2.5m".into(), "0.12".into(), "-8200".into(), "3".into())
        );
        assert_eq!(ticks(0.0, 25.0, 4), vec![0.0, 6.0, 12.0, 18.0, 24.0]);
        let table = markdown_table(&[json!({"name": "ada", "n": 1}), json!({"name": "grace", "n": 22})], None);
        assert!(table.contains("| name  | n  |") && table.contains("|-------|----|"));
        assert!(markdown_table(&[json!(["a", "b"])], Some(&["x".to_string(), "y".to_string()])).contains("| x | y |"));
        assert_eq!(markdown_table(&[], None), "(no rows)");
        let text = render_report("Q3", &[("Revenue", "4.2M"), ("Empty", "")], "Q3 went well.");
        assert!(text.starts_with("# Q3") && text.find("Q3 went well.") < text.find("## Revenue") && !text.contains("## Empty"));
    }

    #[tokio::test]
    async fn the_chart_and_report_tools_write_into_the_workspace() {
        let root = std::env::temp_dir().join(format!("ah-charts-{}", uuid::Uuid::new_v4()));
        let ws: Arc<dyn Workspace> = Arc::new(LocalWorkspace::new(root, WorkspaceOptions { ephemeral: true, ..Default::default() }).unwrap());
        let chart = make_chart_tool(Some(ws.clone()), "render_chart");
        let a = |v: Value| v.as_object().cloned().unwrap();
        let out =
            chart.invoke(a(json!({"kind": "bar", "title": "Sales", "data": {"a": 1, "b": 2}, "path": "sales.svg"})), ToolContext::default()).await.unwrap();
        assert!(out.as_str().unwrap().starts_with("wrote sales.svg ("));
        assert!(ws.read("sales.svg").await.unwrap().contains("<svg"));
        let svg = chart.invoke(a(json!({"kind": "line", "title": "Trend", "data": {"s": [1, 2, 3]}})), ToolContext::default()).await.unwrap();
        assert!(svg.as_str().unwrap().starts_with("<svg"));
        let e = chart.invoke(a(json!({"kind": "pie", "title": "x", "data": {"a": 1}})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "unknown chart kind 'pie'; use 'bar' or 'line'");
        let e = chart.invoke(a(json!({"kind": "bar", "title": "x", "data": {"a": 1, "b": "twelve"}})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "'b' is not a number: 'twelve'");

        let report = make_report_tool(Some(ws.clone()), "write_report");
        report
            .invoke(
                a(json!({"title": "Findings", "summary": "It works.", "sections": [{"heading": "Detail", "body": "The evidence."}], "path": "report.md"})),
                ToolContext::default(),
            )
            .await
            .unwrap();
        let written = ws.read("report.md").await.unwrap();
        assert!(written.starts_with("# Findings") && written.contains("## Detail"));
        ws.close().await;
    }
}
