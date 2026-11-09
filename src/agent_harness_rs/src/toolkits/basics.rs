//! Small native tools worth having on day one.

use std::time::{Duration, Instant};

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::governance::py_str_repr;
use crate::tools::{Tool, ToolContext};

// ---------------------------------------------------------------------------
// now
// ---------------------------------------------------------------------------

fn utc() -> String {
    "UTC".into()
}

#[derive(Deserialize, JsonSchema)]
struct NowArgs {
    /// an IANA timezone name, or UTC.
    #[serde(default = "utc")]
    timezone: String,
}

/// The current date and time in `timezone` (an IANA name, or UTC), as
/// `2026-11-06T09:30:00+00:00`.
pub fn current_time(timezone: &str) -> Result<String> {
    use jiff::tz::TimeZone;
    let zone = if timezone.eq_ignore_ascii_case("UTC") {
        TimeZone::UTC
    } else {
        TimeZone::get(timezone)
            .map_err(|_| Error::tool(format!("unknown timezone {}: 'No time zone found with key {timezone}'", py_str_repr(timezone)), "now"))?
    };
    Ok(jiff::Zoned::now().with_time_zone(zone).strftime("%Y-%m-%dT%H:%M:%S%:z").to_string())
}

/// The `now` tool: the current date and time.
pub fn now() -> Tool {
    Tool::new("now", |args: NowArgs, _ctx: ToolContext| async move { current_time(&args.timezone) })
        .describe("The current date and time.")
        .tags(["builtin", "time"])
}

// ---------------------------------------------------------------------------
// calculate
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct CalculateArgs {
    /// e.g. "1249 * 0.175 + 32".
    expression: String,
}

/// The `calculate` tool: arithmetic, exactly, and nothing else.
///
/// `+ - * / // % **`, unary signs and parentheses over integer and float
/// literals, with Python's meaning for each (`7 // -2` is `-4`, `-7 % 3` is
/// `2`, `10 / 5` is `2.0`). Exponents above 64 are refused.
pub fn calculate() -> Tool {
    Tool::new("calculate", |args: CalculateArgs, _ctx: ToolContext| async move { evaluate(&args.expression) })
        .describe("Evaluate an arithmetic expression exactly.")
        .tags(["builtin", "math"])
}

/// A number as Python holds it: an int stays an int until something makes it
/// a float. Integers are 128-bit here; one that outgrows that becomes a float.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    fn float(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }

    fn to_json(self) -> Result<Value> {
        match self {
            Num::Int(i) => Ok(i64::try_from(i).map(Value::from).or_else(|_| u64::try_from(i).map(Value::from)).unwrap_or_else(|_| json!(i as f64))),
            Num::Float(f) if f.is_finite() => Ok(json!(f)),
            Num::Float(_) => Err(calc_error("calculate failed: the result is too large")),
        }
    }
}

fn calc_error(message: impl Into<String>) -> Error {
    Error::tool(message, "calculate")
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(Num),
    Op(&'static str),
    Open,
    Close,
    /// Valid Python, but not arithmetic: a name, a string, a comparison...
    Other,
}

fn tokenize(src: &str) -> std::result::Result<Vec<Tok>, ()> {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) {
            let start = i;
            if c == '0' && matches!(chars.get(i + 1), Some('x' | 'X' | 'o' | 'O' | 'b' | 'B')) {
                let radix = match chars[i + 1].to_ascii_lowercase() {
                    'x' => 16,
                    'o' => 8,
                    _ => 2,
                };
                i += 2;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let digits: String = chars[start + 2..i].iter().filter(|c| **c != '_').collect();
                let value = i128::from_str_radix(&digits, radix).map_err(|_| ())?;
                out.push(Tok::Num(Num::Int(value)));
                continue;
            }
            let mut float = false;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '_') {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                float = true;
                i += 1;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '_') {
                    i += 1;
                }
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    float = true;
                    i = j;
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '_') {
                        i += 1;
                    }
                }
            }
            let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            // `1j` is a complex literal: Python, but not a real number.
            if i < chars.len() && (chars[i] == 'j' || chars[i] == 'J') {
                i += 1;
                out.push(Tok::Other);
                continue;
            }
            if i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                return Err(());
            }
            let num = if float {
                Num::Float(text.parse().map_err(|_| ())?)
            } else {
                if text.len() > 1 && text.starts_with('0') && text.chars().any(|c| c != '0') {
                    return Err(()); // `012` is a syntax error in Python 3
                }
                text.parse::<i128>().map(Num::Int).or_else(|_| text.parse::<f64>().map(Num::Float)).map_err(|_| ())?
            };
            out.push(Tok::Num(num));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Other);
            continue;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            if i >= chars.len() {
                return Err(());
            }
            i += 1;
            out.push(Tok::Other);
            continue;
        }
        let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        match two.as_str() {
            "**" | "//" => {
                out.push(Tok::Op(if two == "**" { "**" } else { "//" }));
                i += 2;
                continue;
            }
            "<<" | ">>" | "<=" | ">=" | "==" | "!=" => {
                out.push(Tok::Other);
                i += 2;
                continue;
            }
            _ => {}
        }
        match c {
            '+' => out.push(Tok::Op("+")),
            '-' => out.push(Tok::Op("-")),
            '*' => out.push(Tok::Op("*")),
            '/' => out.push(Tok::Op("/")),
            '%' => out.push(Tok::Op("%")),
            '(' => out.push(Tok::Open),
            ')' => out.push(Tok::Close),
            '&' | '|' | '^' | '~' | '<' | '>' | '@' | ',' | '[' | ']' | '{' | '}' | '.' | ':' => out.push(Tok::Other),
            _ => return Err(()),
        }
        i += 1;
    }
    Ok(out)
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

type Step = std::result::Result<Num, Error>;

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn op(&self) -> Option<&'static str> {
        match self.peek() {
            Some(Tok::Op(o)) => Some(o),
            _ => None,
        }
    }

    fn expr(&mut self) -> Step {
        let mut left = self.term()?;
        while let Some(op @ ("+" | "-")) = self.op() {
            self.pos += 1;
            let right = self.term()?;
            left = binary(op, left, right)?;
        }
        Ok(left)
    }

    fn term(&mut self) -> Step {
        let mut left = self.factor()?;
        while let Some(op @ ("*" | "/" | "//" | "%")) = self.op() {
            self.pos += 1;
            let right = self.factor()?;
            left = binary(op, left, right)?;
        }
        Ok(left)
    }

    fn factor(&mut self) -> Step {
        match self.op() {
            Some("-") => {
                self.pos += 1;
                Ok(match self.factor()? {
                    Num::Int(i) => i.checked_neg().map_or(Num::Float(-(i as f64)), Num::Int),
                    Num::Float(f) => Num::Float(-f),
                })
            }
            Some("+") => {
                self.pos += 1;
                self.factor()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Step {
        let base = self.atom()?;
        if self.op() == Some("**") {
            self.pos += 1;
            let exponent = self.factor()?;
            if exponent.float().abs() > 64.0 {
                return Err(calc_error("exponent too large"));
            }
            return binary("**", base, exponent);
        }
        Ok(base)
    }

    fn atom(&mut self) -> Step {
        match self.toks.get(self.pos).cloned() {
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(n)
            }
            Some(Tok::Open) => {
                self.pos += 1;
                let value = self.expr()?;
                if self.peek() != Some(&Tok::Close) {
                    return Err(syntax());
                }
                self.pos += 1;
                Ok(value)
            }
            _ => Err(syntax()),
        }
    }
}

/// A marker the caller turns into "cannot parse ...".
fn syntax() -> Error {
    Error::tool("\0syntax", "calculate")
}

fn floor_div_int(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) { q - 1 } else { q }
}

fn py_fmod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) { r + b } else { r }
}

fn binary(op: &str, left: Num, right: Num) -> Step {
    use Num::{Float, Int};
    let fail = |m: &str| Err(calc_error(format!("calculate failed: {m}")));
    Ok(match (op, left, right) {
        ("+", Int(a), Int(b)) => a.checked_add(b).map_or(Float(a as f64 + b as f64), Int),
        ("-", Int(a), Int(b)) => a.checked_sub(b).map_or(Float(a as f64 - b as f64), Int),
        ("*", Int(a), Int(b)) => a.checked_mul(b).map_or(Float(a as f64 * b as f64), Int),
        ("+", a, b) => Float(a.float() + b.float()),
        ("-", a, b) => Float(a.float() - b.float()),
        ("*", a, b) => Float(a.float() * b.float()),
        ("/", Int(_), Int(0)) => return fail("division by zero"),
        ("/", _, b) if b.float() == 0.0 => return fail("float division by zero"),
        ("/", a, b) => Float(a.float() / b.float()),
        ("//", Int(_), Int(0)) => return fail("integer division or modulo by zero"),
        ("//", Int(a), Int(b)) => a.checked_div(b).map_or(Float((a as f64 / b as f64).floor()), |_| Int(floor_div_int(a, b))),
        ("//", _, b) if b.float() == 0.0 => return fail("float floor division by zero"),
        ("//", a, b) => Float(((a.float() - py_fmod(a.float(), b.float())) / b.float()).round()),
        ("%", Int(_), Int(0)) => return fail("integer modulo by zero"),
        ("%", Int(a), Int(b)) => Int(a.checked_rem(b).map_or(0, |r| if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r })),
        ("%", _, b) if b.float() == 0.0 => return fail("float modulo by zero"),
        ("%", a, b) => Float(py_fmod(a.float(), b.float())),
        ("**", Int(a), Int(b)) if b >= 0 => match u32::try_from(b).ok().and_then(|e| a.checked_pow(e)) {
            Some(v) => Int(v),
            None => Float((a as f64).powf(b as f64)),
        },
        ("**", a, b) => {
            let (x, y) = (a.float(), b.float());
            if x == 0.0 && y < 0.0 {
                return fail("0.0 cannot be raised to a negative power");
            }
            if x < 0.0 && y.fract() != 0.0 {
                return fail("the result is not a real number");
            }
            let v = x.powf(y);
            if !v.is_finite() {
                return fail("(34, 'Result too large')");
            }
            Float(v)
        }
        _ => return Err(syntax()),
    })
}

/// Evaluate an arithmetic expression as Python would, refusing anything else.
pub fn evaluate(expression: &str) -> Result<Value> {
    let cannot = || calc_error(format!("cannot parse {}", py_str_repr(expression)));
    let toks = tokenize(expression).map_err(|_| cannot())?;
    if toks.contains(&Tok::Other) {
        return Err(calc_error("only arithmetic is allowed here"));
    }
    let mut parser = Parser { toks, pos: 0 };
    let value = parser.expr().and_then(|v| if parser.pos == parser.toks.len() { Ok(v) } else { Err(syntax()) });
    match value {
        Ok(v) => v.to_json(),
        Err(Error::Tool { message, .. }) if message == "\0syntax" => Err(cannot()),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// corpus search
// ---------------------------------------------------------------------------

fn five() -> usize {
    5
}

#[derive(Deserialize, JsonSchema)]
struct CorpusArgs {
    /// the words to look for.
    query: String,
    /// how many documents to return.
    #[serde(default = "five")]
    limit: usize,
}

/// A keyword search tool over documents you hand it, named `search_corpus`
/// with 600-character snippets. See [`make_corpus_search_with`].
pub fn make_corpus_search<I, K, V>(documents: I) -> Tool
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    make_corpus_search_with(documents, "search_corpus", 600)
}

/// A keyword search tool over documents you hand it.
///
/// Enough for a small, fixed corpus. For anything larger, put the documents in
/// semantic memory and give the agent the `recall` tool instead.
pub fn make_corpus_search_with<I, K, V>(documents: I, name: &str, snippet_chars: usize) -> Tool
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let corpus: IndexMap<String, String> = documents.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
    let corpus = std::sync::Arc::new(corpus);
    Tool::new(name, move |args: CorpusArgs, _ctx: ToolContext| {
        let corpus = corpus.clone();
        async move { Ok(search_corpus(&corpus, &args.query, args.limit, snippet_chars)) }
    })
    .describe("Search the document corpus.")
    .tags(["builtin", "search"])
    .cacheable(true)
}

fn search_corpus(corpus: &IndexMap<String, String>, query: &str, limit: usize, snippet_chars: usize) -> Vec<Value> {
    let lowered = query.to_lowercase();
    let terms: Vec<&str> = lowered.split_whitespace().filter(|t| t.chars().count() > 2).collect();
    let mut hits: Vec<(usize, &str, String)> = Vec::new();
    for (title, text) in corpus {
        let low = text.to_lowercase();
        let score: usize = terms.iter().map(|t| low.matches(t).count()).sum();
        if score == 0 {
            continue;
        }
        // Character offsets, as Python's `str.find` gives them.
        let first = terms.iter().filter_map(|t| low.find(t).map(|b| low[..b].chars().count())).min().unwrap_or(0);
        let start = first.saturating_sub(100);
        let snippet: String = text.chars().skip(start).take(snippet_chars).collect();
        hits.push((score, title, snippet));
    }
    hits.sort_by_key(|h| std::cmp::Reverse(h.0));
    hits.into_iter().take(limit).map(|(score, title, snippet)| json!({"document": title, "score": score, "snippet": snippet})).collect()
}

// ---------------------------------------------------------------------------
// sleep
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
struct WaitArgs {
    /// how long to wait, capped at 60.
    seconds: f64,
}

/// The `sleep` tool: pause before the next step, for polling an external job.
pub fn wait() -> Tool {
    Tool::new("sleep", |args: WaitArgs, _ctx: ToolContext| async move {
        let capped = if args.seconds.is_nan() { 0.0 } else { args.seconds.clamp(0.0, 60.0) };
        let started = Instant::now();
        tokio::time::sleep(Duration::from_secs_f64(capped)).await;
        Ok(format!("waited {:.1}s", started.elapsed().as_secs_f64()))
    })
    .describe("Pause before the next step, for polling an external job.")
    .tags(["builtin"])
}

/// The safe, dependency-free defaults: `now` and `calculate`.
pub fn basic_tools() -> Vec<Tool> {
    vec![now(), calculate()]
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    async fn calc(e: &str) -> Result<Value> {
        calculate().invoke(args(json!({"expression": e})), ToolContext::default()).await
    }

    #[tokio::test]
    async fn calculate_is_exact_and_refuses_anything_else() {
        let v = calc("1249 * 0.175 + 32").await.unwrap().as_f64().unwrap();
        assert!((v - 250.575).abs() < 1e-9);
        let e = calc("__import__('os').system('ls')").await.unwrap_err().to_string();
        assert_eq!(e, "only arithmetic is allowed here");
        assert_eq!(calc("2 ** 999").await.unwrap_err().to_string(), "exponent too large");
        assert_eq!(calc("1 +").await.unwrap_err().to_string(), "cannot parse '1 +'");
    }

    #[tokio::test]
    async fn calculate_follows_python_arithmetic() {
        for (expr, want) in [
            ("7 // -2", json!(-4)),
            ("-7 % 3", json!(2)),
            ("7.5 % -2", json!(-0.5)),
            ("10 / 4", json!(2.5)),
            ("10 / 5", json!(2.0)),
            ("2 ** -2", json!(0.25)),
            ("-2 ** 2", json!(-4)),
            ("2 ** 3 ** 2", json!(512)),
            ("(1 + 2) * 3", json!(9)),
            ("1_000 + 0x10", json!(1016)),
            ("2 ** 64", json!(18446744073709551616.0)),
            ("1e3", json!(1000.0)),
        ] {
            assert_eq!(calc(expr).await.unwrap(), want, "{expr}");
        }
        assert_eq!(calc("1/0").await.unwrap_err().to_string(), "calculate failed: division by zero");
        assert_eq!(calc("1 % 0").await.unwrap_err().to_string(), "calculate failed: integer modulo by zero");
        assert_eq!(calc("1 < 2").await.unwrap_err().to_string(), "only arithmetic is allowed here");
        assert_eq!(calc("1j").await.unwrap_err().to_string(), "only arithmetic is allowed here");
    }

    #[tokio::test]
    async fn now_returns_an_iso_timestamp() {
        let t = now().invoke(Map::new(), ToolContext::default()).await.unwrap();
        let t = t.as_str().unwrap();
        assert!(t.contains('T') && t.ends_with("+00:00"), "{t}");
        let tokyo = now().invoke(args(json!({"timezone": "Asia/Tokyo"})), ToolContext::default()).await.unwrap();
        assert!(tokyo.as_str().unwrap().ends_with("+09:00"));
        let e = now().invoke(args(json!({"timezone": "Mars/Olympus"})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "unknown timezone 'Mars/Olympus': 'No time zone found with key Mars/Olympus'");
        assert_eq!(now().description, "The current date and time.");
        assert_eq!(now().parameters["properties"]["timezone"]["default"], "UTC");
    }

    #[tokio::test]
    async fn corpus_search_ranks_by_hit_count() {
        let search = make_corpus_search([("policy.md", "Refunds are issued within 30 days of purchase."), ("hours.md", "The office is open 9 to 5.")]);
        assert!(search.cacheable && search.name == "search_corpus");
        let hits = search.invoke(args(json!({"query": "refunds days"})), ToolContext::default()).await.unwrap();
        assert_eq!(hits[0]["document"], "policy.md");
        assert!(hits[0]["snippet"].as_str().unwrap().contains("Refunds"));
        assert_eq!(hits[0]["score"], 2);
        let none = search.invoke(args(json!({"query": "zebra"})), ToolContext::default()).await.unwrap();
        assert_eq!(none, json!([]));
    }

    #[tokio::test]
    async fn sleep_is_capped_and_reports_the_wait() {
        let out = wait().invoke(args(json!({"seconds": -5})), ToolContext::default()).await.unwrap();
        assert_eq!(out, "waited 0.0s");
        assert_eq!(wait().name, "sleep");
        assert_eq!(basic_tools().iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["now", "calculate"]);
    }
}
