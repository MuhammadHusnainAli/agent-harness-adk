//! A small, safe expression language — for workflow conditions and templates,
//! and for policy conditions.
//!
//! ```
//! use agent_harness::expr::{compile, Ctx};
//! use serde_json::json;
//!
//! let e = compile("args.amount > 100 and 'payments' in tool.tags").unwrap();
//! let ctx: Ctx = json!({"args": {"amount": 250}, "tool": {"tags": ["payments"]}}).as_object().unwrap().clone();
//! assert_eq!(e.eval(&ctx), json!(true));
//! ```
//!
//! It looks like Python and behaves like it, but it is parsed here, checked,
//! and evaluated over JSON values. Nothing is ever executed as code, there is
//! no access to a name or field starting with an underscore, and no function
//! can be called except the listed ones. Two rules make it safe against loose
//! data: a missing name or field is `null`, never an error; and a comparison
//! that cannot be made (`null > 100`) is `false`.
//!
//! Ported from the Python harness's `governance/expr.py`, with the functions
//! `workflow.py` adds.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use regex::Regex;
use serde_json::{Map, Number, Value, json};

use crate::errors::{Error, Result};

/// What an expression is evaluated against: name → value.
pub type Ctx = Map<String, Value>;

/// Every function an expression may call.
pub const FUNCTIONS: [&str; 25] = [
    "len",
    "lower",
    "upper",
    "str",
    "int",
    "float",
    "abs",
    "min",
    "max",
    "any",
    "all",
    "matches",
    "contains",
    "startswith",
    "endswith",
    "round",
    "sum",
    "sorted",
    "bool",
    "keys",
    "values",
    "join",
    "json",
    "default",
    "range",
];

/// Names that are constants, not lookups.
pub const CONSTANTS: [&str; 9] = ["true", "false", "null", "none", "True", "False", "None", "nil", "NULL"];

fn constant(name: &str) -> Option<Value> {
    match name {
        "true" | "True" => Some(json!(true)),
        "false" | "False" => Some(json!(false)),
        "null" | "none" | "None" => Some(Value::Null),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// the tree
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
    Is,
    IsNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// A parsed expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(Value),
    Name(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Compare(Box<Expr>, Vec<(Cmp, Expr)>),
    Binary(Bin, Box<Expr>, Box<Expr>),
    IfElse(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

// ---------------------------------------------------------------------------
// tokens
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(Value),
    Str(String),
    Ident(String),
    Op(&'static str),
}

fn err(source: &str, what: impl std::fmt::Display) -> Error {
    Error::config(format!("'{source}': {what}"))
}

fn tokenize(source: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == '_') {
                i += 1;
            }
            let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            let num = if text.contains('.') {
                text.parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number)
            } else {
                text.parse::<i64>().ok().map(|n| json!(n))
            };
            out.push(Tok::Num(num.ok_or_else(|| err(source, format!("bad number {text}")))?));
            continue;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut s = String::new();
            loop {
                let Some(&ch) = chars.get(i) else { return Err(err(source, "unterminated string literal")) };
                i += 1;
                if ch == quote {
                    break;
                }
                if ch == '\\' {
                    let Some(&esc) = chars.get(i) else { return Err(err(source, "unterminated string literal")) };
                    i += 1;
                    s.push(match esc {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        other => other,
                    });
                } else {
                    s.push(ch);
                }
            }
            out.push(Tok::Str(s));
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
            continue;
        }
        let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
        let op = match two.as_str() {
            "==" => Some("=="),
            "!=" => Some("!="),
            "<=" => Some("<="),
            ">=" => Some(">="),
            _ => None,
        };
        if let Some(op) = op {
            out.push(Tok::Op(op));
            i += 2;
            continue;
        }
        let op = match c {
            '<' => "<",
            '>' => ">",
            '+' => "+",
            '-' => "-",
            '*' => "*",
            '/' => "/",
            '%' => "%",
            '(' => "(",
            ')' => ")",
            '[' => "[",
            ']' => "]",
            '{' => "{",
            '}' => "}",
            ',' => ",",
            '.' => ".",
            ':' => ":",
            other => return Err(err(source, format!("invalid syntax at '{other}'"))),
        };
        out.push(Tok::Op(op));
        i += 1;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// parsing — Python's precedence, lowest first
// ---------------------------------------------------------------------------
struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    source: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n)
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn is_word(&self, w: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(i)) if i == w)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if self.is_op(op) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_word(&mut self, w: &str) -> bool {
        if self.is_word(w) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<()> {
        if self.eat_op(op) { Ok(()) } else { Err(err(self.source, format!("expected '{op}'"))) }
    }

    fn test(&mut self) -> Result<Expr> {
        let body = self.or_test()?;
        if self.eat_word("if") {
            let cond = self.or_test()?;
            if !self.eat_word("else") {
                return Err(err(self.source, "expected 'else'"));
            }
            let other = self.test()?;
            return Ok(Expr::IfElse(Box::new(cond), Box::new(body), Box::new(other)));
        }
        Ok(body)
    }

    fn or_test(&mut self) -> Result<Expr> {
        let mut parts = vec![self.and_test()?];
        while self.eat_word("or") {
            parts.push(self.and_test()?);
        }
        Ok(if parts.len() == 1 { parts.pop().expect("one") } else { Expr::Or(parts) })
    }

    fn and_test(&mut self) -> Result<Expr> {
        let mut parts = vec![self.not_test()?];
        while self.eat_word("and") {
            parts.push(self.not_test()?);
        }
        Ok(if parts.len() == 1 { parts.pop().expect("one") } else { Expr::And(parts) })
    }

    fn not_test(&mut self) -> Result<Expr> {
        if self.eat_word("not") {
            return Ok(Expr::Not(Box::new(self.not_test()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Expr> {
        let left = self.arith()?;
        let mut ops = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Op("==")) => Cmp::Eq,
                Some(Tok::Op("!=")) => Cmp::Ne,
                Some(Tok::Op("<")) => Cmp::Lt,
                Some(Tok::Op("<=")) => Cmp::Le,
                Some(Tok::Op(">")) => Cmp::Gt,
                Some(Tok::Op(">=")) => Cmp::Ge,
                Some(Tok::Ident(w)) if w == "in" => Cmp::In,
                Some(Tok::Ident(w)) if w == "not" && matches!(self.peek_at(1), Some(Tok::Ident(n)) if n == "in") => {
                    self.pos += 1;
                    Cmp::NotIn
                }
                Some(Tok::Ident(w)) if w == "is" => {
                    if matches!(self.peek_at(1), Some(Tok::Ident(n)) if n == "not") {
                        self.pos += 1;
                        Cmp::IsNot
                    } else {
                        Cmp::Is
                    }
                }
                _ => break,
            };
            self.pos += 1;
            ops.push((op, self.arith()?));
        }
        Ok(if ops.is_empty() { left } else { Expr::Compare(Box::new(left), ops) })
    }

    fn arith(&mut self) -> Result<Expr> {
        let mut left = self.term()?;
        loop {
            let op = if self.eat_op("+") {
                Bin::Add
            } else if self.eat_op("-") {
                Bin::Sub
            } else {
                break;
            };
            left = Expr::Binary(op, Box::new(left), Box::new(self.term()?));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Expr> {
        let mut left = self.factor()?;
        loop {
            let op = if self.eat_op("*") {
                Bin::Mul
            } else if self.eat_op("/") {
                Bin::Div
            } else if self.eat_op("%") {
                Bin::Mod
            } else {
                break;
            };
            left = Expr::Binary(op, Box::new(left), Box::new(self.factor()?));
        }
        Ok(left)
    }

    fn factor(&mut self) -> Result<Expr> {
        if self.eat_op("-") {
            return Ok(Expr::Neg(Box::new(self.factor()?)));
        }
        if self.eat_op("+") {
            return self.factor();
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr> {
        let mut e = self.atom()?;
        loop {
            if self.eat_op(".") {
                let Some(Tok::Ident(name)) = self.peek().cloned() else { return Err(err(self.source, "expected a name after '.'")) };
                self.pos += 1;
                if name.starts_with('_') {
                    return Err(err(self.source, format!("private attribute '{name}'")));
                }
                e = Expr::Attr(Box::new(e), name);
            } else if self.eat_op("[") {
                let key = self.test()?;
                self.expect_op("]")?;
                e = Expr::Index(Box::new(e), Box::new(key));
            } else if self.is_op("(") {
                let Expr::Name(name) = &e else { return Err(err(self.source, "only a named function can be called")) };
                if !FUNCTIONS.contains(&name.as_str()) {
                    let mut allowed = FUNCTIONS.to_vec();
                    allowed.sort();
                    return Err(err(self.source, format!("{name}() is not an allowed function; allowed: {}", allowed.join(", "))));
                }
                let name = name.clone();
                self.pos += 1;
                let mut args = Vec::new();
                while !self.is_op(")") {
                    if matches!(self.peek(), Some(Tok::Ident(_))) && matches!(self.peek_at(1), Some(Tok::Op(o)) if *o == "=") {
                        return Err(err(self.source, "keyword arguments are not allowed"));
                    }
                    args.push(self.test()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op(")")?;
                e = Expr::Call(name, args);
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr> {
        let Some(tok) = self.peek().cloned() else { return Err(err(self.source, "unexpected end of expression")) };
        self.pos += 1;
        match tok {
            Tok::Num(n) => Ok(Expr::Const(n)),
            Tok::Str(s) => Ok(Expr::Const(Value::String(s))),
            Tok::Ident(name) => {
                if ["and", "or", "not", "in", "is", "if", "else", "lambda", "import"].contains(&name.as_str()) {
                    return Err(err(self.source, format!("invalid syntax at '{name}'")));
                }
                if name.starts_with('_') {
                    return Err(err(self.source, format!("private name '{name}'")));
                }
                Ok(constant(&name).map(Expr::Const).unwrap_or(Expr::Name(name)))
            }
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Ok(Expr::List(vec![]));
                }
                let first = self.test()?;
                if self.eat_op(",") {
                    let mut items = vec![first];
                    while !self.is_op(")") {
                        items.push(self.test()?);
                        if !self.eat_op(",") {
                            break;
                        }
                    }
                    self.expect_op(")")?;
                    return Ok(Expr::List(items));
                }
                self.expect_op(")")?;
                Ok(first)
            }
            Tok::Op(open @ ("[" | "{")) => {
                let close = if open == "[" { "]" } else { "}" };
                let mut items = Vec::new();
                while !self.is_op(close) {
                    items.push(self.test()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op(close)?;
                Ok(Expr::List(items))
            }
            Tok::Op(op) => Err(err(self.source, format!("invalid syntax at '{op}'"))),
        }
    }
}

static CACHE: LazyLock<Mutex<HashMap<String, Arc<Expr>>>> = LazyLock::new(Default::default);

/// Parse and compile once; the result is cached by its source.
pub fn compile(source: &str) -> Result<Arc<Expr>> {
    let key = source.trim().to_string();
    if let Some(e) = CACHE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(e.clone());
    }
    let toks = tokenize(&key)?;
    if toks.is_empty() {
        return Err(err(&key, "empty expression"));
    }
    let mut p = Parser { toks, pos: 0, source: &key };
    let e = p.test()?;
    if p.pos != p.toks.len() {
        return Err(err(&key, "invalid syntax"));
    }
    let e = Arc::new(e);
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if cache.len() > 1024 {
        cache.clear();
    }
    cache.insert(key, e.clone());
    Ok(e)
}

// ---------------------------------------------------------------------------
// evaluating
// ---------------------------------------------------------------------------
/// Python truthiness.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A value as it reads inside other text — strings as they are, the rest as JSON.
pub fn text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

fn num(f: f64) -> Value {
    if f.fract() == 0.0 && f.abs() < 9.0e15 { json!(f as i64) } else { Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null) }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn both_int(a: &Value, b: &Value) -> Option<(i64, i64)> {
    Some((a.as_i64()?, b.as_i64()?))
}

/// `a.b` and `a['b']`: objects by key, lists by (negative) index, else null.
fn field(v: &Value, key: &Value) -> Value {
    match (v, key) {
        (Value::Object(m), Value::String(k)) if !k.starts_with('_') => m.get(k).cloned().unwrap_or(Value::Null),
        (Value::Array(a), Value::Number(n)) => {
            let Some(i) = n.as_i64() else { return Value::Null };
            let len = a.len() as i64;
            if -len <= i && i < len { a[if i < 0 { (len + i) as usize } else { i as usize }].clone() } else { Value::Null }
        }
        (Value::String(s), Value::Number(n)) => {
            let chars: Vec<char> = s.chars().collect();
            let len = chars.len() as i64;
            n.as_i64()
                .filter(|i| -len <= *i && *i < len)
                .map(|i| Value::String(chars[if i < 0 { (len + i) as usize } else { i as usize }].to_string()))
                .unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

fn eq(a: &Value, b: &Value) -> bool {
    match (as_f64(a), as_f64(b)) {
        (Some(x), Some(y)) if !a.is_boolean() || !b.is_boolean() => x == y,
        _ => a == b,
    }
}

fn contains(container: &Value, item: &Value) -> bool {
    match container {
        Value::String(s) => item.as_str().is_some_and(|i| s.contains(i)),
        Value::Array(a) => a.iter().any(|x| eq(x, item)),
        Value::Object(m) => item.as_str().is_some_and(|k| m.contains_key(k)),
        _ => false,
    }
}

/// `<`, or `None` when the two cannot be ordered (Python's TypeError → false).
fn order(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        (Value::Array(x), Value::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                match order(p, q)? {
                    std::cmp::Ordering::Equal => continue,
                    other => return Some(other),
                }
            }
            Some(x.len().cmp(&y.len()))
        }
        _ => as_f64(a)?.partial_cmp(&as_f64(b)?),
    }
}

fn compare(op: Cmp, a: &Value, b: &Value) -> bool {
    use std::cmp::Ordering::*;
    match op {
        Cmp::Eq => eq(a, b),
        Cmp::Ne => !eq(a, b),
        Cmp::Lt => order(a, b) == Some(Less),
        Cmp::Le => matches!(order(a, b), Some(Less | Equal)),
        Cmp::Gt => order(a, b) == Some(Greater),
        Cmp::Ge => matches!(order(a, b), Some(Greater | Equal)),
        Cmp::In => contains(b, a),
        Cmp::NotIn => !contains(b, a),
        // Only `x is None` is meaningful over JSON values.
        Cmp::Is => (a.is_null() && b.is_null()) || (a.is_boolean() && a == b),
        Cmp::IsNot => !((a.is_null() && b.is_null()) || (a.is_boolean() && a == b)),
    }
}

fn binary(op: Bin, a: &Value, b: &Value) -> Value {
    match (op, a, b) {
        (Bin::Add, Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
        (Bin::Add, Value::Array(x), Value::Array(y)) => Value::Array([x.clone(), y.clone()].concat()),
        (Bin::Mul, Value::String(s), n) | (Bin::Mul, n, Value::String(s)) if n.is_i64() => {
            Value::String(s.repeat(n.as_i64().unwrap_or(0).clamp(0, 10_000) as usize))
        }
        _ => {
            if a.is_string() || b.is_string() || a.is_null() || b.is_null() {
                return Value::Null;
            }
            if let Some((x, y)) = both_int(a, b) {
                return match op {
                    Bin::Add => x.checked_add(y).map(|n| json!(n)).unwrap_or(Value::Null),
                    Bin::Sub => x.checked_sub(y).map(|n| json!(n)).unwrap_or(Value::Null),
                    Bin::Mul => x.checked_mul(y).map(|n| json!(n)).unwrap_or(Value::Null),
                    Bin::Div if y == 0 => Value::Null,
                    Bin::Div => Number::from_f64(x as f64 / y as f64).map(Value::Number).unwrap_or(Value::Null),
                    Bin::Mod if y == 0 => Value::Null,
                    // Python: the remainder takes the divisor's sign.
                    Bin::Mod => {
                        let r = x % y;
                        json!(if r != 0 && (r < 0) != (y < 0) { r + y } else { r })
                    }
                };
            }
            let (Some(x), Some(y)) = (as_f64(a), as_f64(b)) else { return Value::Null };
            let r = match op {
                Bin::Add => x + y,
                Bin::Sub => x - y,
                Bin::Mul => x * y,
                Bin::Div if y == 0.0 => return Value::Null,
                Bin::Div => x / y,
                Bin::Mod if y == 0.0 => return Value::Null,
                Bin::Mod => x - y * (x / y).floor(),
            };
            Number::from_f64(r).map(Value::Number).unwrap_or(Value::Null)
        }
    }
}

fn items(v: &Value) -> Vec<Value> {
    match v {
        Value::Array(a) => a.clone(),
        Value::Object(m) => m.keys().map(|k| json!(k)).collect(),
        Value::String(s) => s.chars().map(|c| json!(c.to_string())).collect(),
        _ => vec![],
    }
}

fn call(name: &str, args: &[Value]) -> Value {
    let a0 = args.first().unwrap_or(&Value::Null);
    let a1 = args.get(1);
    let extreme = |want: std::cmp::Ordering| {
        let pool: Vec<Value> = if args.len() == 1 { items(a0) } else { args.to_vec() };
        let mut best: Option<Value> = None;
        for v in pool {
            best = match best {
                None => Some(v),
                Some(b) => match order(&v, &b) {
                    Some(o) if o == want => Some(v),
                    Some(_) => Some(b),
                    None => return Value::Null,
                },
            };
        }
        best.unwrap_or(Value::Null)
    };
    match name {
        "len" => match a0 {
            Value::Null => json!(0),
            Value::String(s) => json!(s.chars().count()),
            Value::Array(a) => json!(a.len()),
            Value::Object(m) => json!(m.len()),
            _ => Value::Null,
        },
        "lower" => json!(if a0.is_null() { String::new() } else { text(a0).to_lowercase() }),
        "upper" => json!(if a0.is_null() { String::new() } else { text(a0).to_uppercase() }),
        "str" => json!(text(a0)),
        "int" => match a0 {
            Value::Number(n) => n.as_i64().map(|i| json!(i)).or_else(|| n.as_f64().map(|f| json!(f.trunc() as i64))).unwrap_or(Value::Null),
            Value::String(s) => s.trim().parse::<i64>().map(|i| json!(i)).unwrap_or(Value::Null),
            Value::Bool(b) => json!(*b as i64),
            _ => Value::Null,
        },
        "float" => match a0 {
            Value::String(s) => s.trim().parse::<f64>().ok().and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null),
            other => as_f64(other).and_then(Number::from_f64).map(Value::Number).unwrap_or(Value::Null),
        },
        "abs" => match a0 {
            v if v.is_i64() => json!(v.as_i64().unwrap_or(0).abs()),
            v => as_f64(v).and_then(|f| Number::from_f64(f.abs())).map(Value::Number).unwrap_or(Value::Null),
        },
        "min" => extreme(std::cmp::Ordering::Less),
        "max" => extreme(std::cmp::Ordering::Greater),
        "any" => json!(items(a0).iter().any(truthy)),
        "all" => json!(items(a0).iter().all(truthy)),
        "matches" => match (a0, a1.and_then(Value::as_str)) {
            (Value::Null, _) | (_, None) => json!(false),
            (v, Some(p)) => json!(Regex::new(p).is_ok_and(|re| re.is_match(&text(v)))),
        },
        "contains" => json!(a1.is_some_and(|i| contains(a0, i))),
        "startswith" => json!(!a0.is_null() && a1.and_then(Value::as_str).is_some_and(|p| text(a0).starts_with(p))),
        "endswith" => json!(!a0.is_null() && a1.and_then(Value::as_str).is_some_and(|p| text(a0).ends_with(p))),
        "round" => {
            let Some(x) = as_f64(a0) else { return Value::Null };
            match a1.and_then(Value::as_i64) {
                None => {
                    // Python rounds half to even.
                    let r = x.round();
                    let r = if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r };
                    json!(r as i64)
                }
                Some(d) => {
                    let m = 10f64.powi(d as i32);
                    Number::from_f64((x * m).round() / m).map(Value::Number).unwrap_or(Value::Null)
                }
            }
        }
        "sum" => {
            let vals = items(a0);
            if vals.iter().all(|v| v.is_i64()) {
                json!(vals.iter().filter_map(Value::as_i64).sum::<i64>())
            } else if vals.iter().all(|v| as_f64(v).is_some()) {
                num(vals.iter().filter_map(as_f64).sum())
            } else {
                Value::Null
            }
        }
        "sorted" => {
            let mut v = items(a0);
            let mut ok = true;
            v.sort_by(|x, y| {
                order(x, y).unwrap_or_else(|| {
                    ok = false;
                    std::cmp::Ordering::Equal
                })
            });
            if ok { Value::Array(v) } else { Value::Null }
        }
        "bool" => json!(truthy(a0)),
        "keys" => match a0 {
            Value::Object(m) => Value::Array(m.keys().map(|k| json!(k)).collect()),
            _ => json!([]),
        },
        "values" => match a0 {
            Value::Object(m) => Value::Array(m.values().cloned().collect()),
            _ => json!([]),
        },
        "join" => {
            let sep = a1.and_then(Value::as_str).unwrap_or(", ");
            json!(items(a0).iter().map(text).collect::<Vec<_>>().join(sep))
        }
        "json" => json!(a0.to_string()),
        "default" => {
            if a0.is_null() || a0.as_str() == Some("") {
                a1.cloned().unwrap_or(Value::Null)
            } else {
                a0.clone()
            }
        }
        "range" => {
            let n: Vec<i64> = args.iter().filter_map(Value::as_i64).map(|n| n.min(10_000)).collect();
            let (start, stop, step) = match n.as_slice() {
                [stop] => (0, *stop, 1),
                [start, stop] => (*start, *stop, 1),
                [start, stop, step] if *step != 0 => (*start, *stop, *step),
                _ => return Value::Null,
            };
            let mut out = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(json!(i));
                i += step;
                if out.len() >= 10_000 {
                    break;
                }
            }
            Value::Array(out)
        }
        _ => Value::Null,
    }
}

impl Expr {
    /// Evaluate against a context. Never fails: what cannot be done is `null`
    /// (or `false`, for a comparison).
    pub fn eval(&self, ctx: &Ctx) -> Value {
        match self {
            Expr::Const(v) => v.clone(),
            Expr::Name(n) => ctx.get(n).cloned().unwrap_or(Value::Null),
            Expr::Attr(e, k) => field(&e.eval(ctx), &Value::String(k.clone())),
            Expr::Index(e, k) => field(&e.eval(ctx), &k.eval(ctx)),
            Expr::List(items) => Value::Array(items.iter().map(|i| i.eval(ctx)).collect()),
            Expr::And(parts) => json!(parts.iter().all(|p| truthy(&p.eval(ctx)))),
            Expr::Or(parts) => json!(parts.iter().any(|p| truthy(&p.eval(ctx)))),
            Expr::Not(e) => json!(!truthy(&e.eval(ctx))),
            Expr::Neg(e) => match e.eval(ctx) {
                v if v.is_i64() => json!(-v.as_i64().unwrap_or(0)),
                v => as_f64(&v).and_then(|f| Number::from_f64(-f)).map(Value::Number).unwrap_or(Value::Null),
            },
            Expr::Compare(left, ops) => {
                let mut current = left.eval(ctx);
                for (op, right) in ops {
                    let other = right.eval(ctx);
                    if !compare(*op, &current, &other) {
                        return json!(false);
                    }
                    current = other;
                }
                json!(true)
            }
            Expr::Binary(op, a, b) => binary(*op, &a.eval(ctx), &b.eval(ctx)),
            Expr::IfElse(cond, body, other) => {
                if truthy(&cond.eval(ctx)) {
                    body.eval(ctx)
                } else {
                    other.eval(ctx)
                }
            }
            Expr::Call(name, args) => {
                let values: Vec<Value> = args.iter().map(|a| a.eval(ctx)).collect();
                call(name, &values)
            }
        }
    }

    /// Every name it reads, and every `name.key` / `name['key']` it reaches into.
    pub fn references(&self) -> (Vec<String>, Vec<(String, String)>) {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        self.walk(&mut |e| match e {
            Expr::Name(n) => names.push(n.clone()),
            Expr::Attr(inner, k) => {
                if let Expr::Name(n) = inner.as_ref() {
                    fields.push((n.clone(), k.clone()));
                }
            }
            Expr::Index(inner, k) => {
                if let (Expr::Name(n), Expr::Const(Value::String(k))) = (inner.as_ref(), k.as_ref()) {
                    fields.push((n.clone(), k.clone()));
                }
            }
            _ => {}
        });
        (names, fields)
    }

    fn walk(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Attr(e, _) | Expr::Not(e) | Expr::Neg(e) => e.walk(f),
            Expr::Index(e, k) => {
                e.walk(f);
                k.walk(f);
            }
            Expr::List(v) | Expr::And(v) | Expr::Or(v) | Expr::Call(_, v) => v.iter().for_each(|x| x.walk(f)),
            Expr::Compare(l, ops) => {
                l.walk(f);
                ops.iter().for_each(|(_, r)| r.walk(f));
            }
            Expr::Binary(_, a, b) => {
                a.walk(f);
                b.walk(f);
            }
            Expr::IfElse(c, b, o) => {
                c.walk(f);
                b.walk(f);
                o.walk(f);
            }
            Expr::Const(_) | Expr::Name(_) => {}
        }
    }
}

/// Compile and evaluate in one step.
pub fn eval(source: &str, ctx: &Ctx) -> Result<Value> {
    Ok(compile(source)?.eval(ctx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(v: Value) -> Ctx {
        v.as_object().cloned().unwrap_or_default()
    }

    fn e(src: &str, c: &Value) -> Value {
        eval(src, &ctx(c.clone())).unwrap()
    }

    #[test]
    fn python_semantics() {
        let c = json!({"args": {"amount": 250, "items": [1, 2, 3]}, "tool": {"tags": ["payments"], "name": "Refund"}, "s": "abc"});
        assert_eq!(e("args.amount > 100 and 'payments' in tool.tags", &c), json!(true));
        assert_eq!(e("missing.deep.field", &c), Value::Null);
        assert_eq!(e("missing > 100", &c), json!(false), "an impossible comparison is false");
        assert_eq!(e("1 < args.amount <= 250", &c), json!(true));
        assert_eq!(e("args.items[-1] + 2 * 3", &c), json!(9));
        assert_eq!(e("7 / 2", &c), json!(3.5));
        assert_eq!(e("-7 % 3", &c), json!(2));
        assert_eq!(e("7 % -3", &c), json!(-2));
        assert_eq!(e("'yes' if not args.missing else 'no'", &c), json!("yes"));
        assert_eq!(e("lower(tool.name) == 'refund'", &c), json!(true));
        assert_eq!(e("x is None and 4 not in args.items", &c), json!(true));
        assert_eq!(e("len(args.items) == 3 and sum(args.items) == 6", &c), json!(true));
        assert_eq!(e("max(args.items)", &c), json!(3));
        assert_eq!(e("min(4, 2, 9)", &c), json!(2));
        assert_eq!(e("join(sorted([3, 1, 2]), '-')", &c), json!("1-2-3"));
        assert_eq!(e("default(missing, 'fallback')", &c), json!("fallback"));
        assert_eq!(e("range(3)", &c), json!([0, 1, 2]));
        assert_eq!(e("round(2.5)", &c), json!(2));
        assert_eq!(e("round(2.567, 2)", &c), json!(2.57));
        assert_eq!(e("matches(s, '^a.c$')", &c), json!(true));
        assert_eq!(e("'b' in s and s[0] == 'a'", &c), json!(true));
        assert_eq!(e("1 == 1.0 and True", &c), json!(true));
        assert_eq!(e("int('42') + float('0.5')", &c), json!(42.5));
        assert_eq!(e("1 / 0", &c), Value::Null);
        assert_eq!(e("keys(args)", &c), json!(["amount", "items"]));
    }

    #[test]
    fn refuses_what_is_not_allowed() {
        for bad in ["__import__('os')", "open('x')", "a._secret", "x(1)", "lambda: 1", "1 +", "f(a=1)", "'unclosed"] {
            assert!(compile(bad).is_err(), "{bad}");
        }
        let msg = compile("exec('x')").unwrap_err().to_string();
        assert!(msg.contains("exec() is not an allowed function; allowed: abs, all, any"), "{msg}");
    }

    #[test]
    fn reports_references() {
        let (names, fields) = compile("steps.classify.output and inputs['order'] > state.n").unwrap().references();
        assert_eq!(names, vec!["steps", "inputs", "state"]);
        assert_eq!(fields, vec![("steps".into(), "classify".into()), ("inputs".into(), "order".into()), ("state".into(), "n".into())]);
    }
}
