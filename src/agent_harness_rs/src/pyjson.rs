//! JSON written exactly as Python's `json.dumps` writes it.
//!
//! Wherever a hash or a signature covers JSON — the audit chain, governance
//! records — the bytes must be the ones Python produces, or a trail written by
//! one language would not verify in the other. That means Python's separators,
//! `sort_keys`, `ensure_ascii` escaping, and Python's `repr` of a float
//! (`1e-05`, `1.5e+16`, `40.0`).

use serde_json::Value;

/// How to write it. The default is `json.dumps(value)`: `", "` and `": "`,
/// keys in the order they are held, everything outside ASCII escaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dumps {
    pub sort_keys: bool,
    /// `separators=(",", ":")`.
    pub compact: bool,
    pub ensure_ascii: bool,
}

impl Default for Dumps {
    fn default() -> Self {
        Self { sort_keys: false, compact: false, ensure_ascii: true }
    }
}

impl Dumps {
    /// `json.dumps(v, sort_keys=True, separators=(",", ":"))` — the form hashes cover.
    pub fn canonical() -> Self {
        Self { sort_keys: true, compact: true, ensure_ascii: true }
    }
}

/// `json.dumps(value)`.
pub fn dumps(value: &Value) -> String {
    dumps_with(value, Dumps::default())
}

pub fn dumps_with(value: &Value, how: Dumps) -> String {
    let mut out = String::new();
    write(&mut out, value, how);
    out
}

fn write(out: &mut String, value: &Value, how: Dumps) {
    let (item, key) = if how.compact { (",", ":") } else { (", ", ": ") };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) if !n.is_f64() => out.push_str(&i.to_string()),
            (_, Some(u), _) if !n.is_f64() => out.push_str(&u.to_string()),
            (_, _, Some(f)) => out.push_str(&float_repr(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => string(out, s, how.ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write(out, v, how);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            if how.sort_keys {
                // Python sorts str by code point; Rust's str ordering is by UTF-8
                // bytes, which orders code points the same way.
                keys.sort();
            }
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                string(out, k, how.ensure_ascii);
                out.push_str(key);
                write(out, &map[k], how);
            }
            out.push('}');
        }
    }
}

fn string(out: &mut String, s: &str, ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && !c.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, positional for
/// exponents from -4 to 15, scientific (`1e-05`, `1.5e+16`) outside them.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // `{:e}` gives the shortest round-trip digits: "1.5e-5", "4e1".
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let (head, tail) = digits.split_at(1);
        let frac = if tail.is_empty() { String::new() } else { format!(".{tail}") };
        let esign = if exp < 0 { "-" } else { "+" };
        return format!("{sign}{head}{frac}e{esign}{:02}", exp.abs());
    }
    let point = exp + 1; // digits before the decimal point
    let body = if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
    } else {
        let (a, b) = digits.split_at(point as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_as_python_writes_them() {
        for (f, want) in [
            (40.0, "40.0"),
            (0.1, "0.1"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1.5e16, "1.5e+16"),
            (1e15, "1000000000000000.0"),
            (1793978140.232628, "1793978140.232628"),
            (-2.5, "-2.5"),
            (123456789.0, "123456789.0"),
            (1e100, "1e+100"),
        ] {
            assert_eq!(float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn dumps_as_python_does() {
        let v = json!({"b": [1, 2.5, "é"], "a": {"z": null, "y": true}, "c": "q\"\n"});
        assert_eq!(dumps(&v), r#"{"b": [1, 2.5, "\u00e9"], "a": {"z": null, "y": true}, "c": "q\"\n"}"#);
        assert_eq!(dumps_with(&v, Dumps::canonical()), r#"{"a":{"y":true,"z":null},"b":[1,2.5,"\u00e9"],"c":"q\"\n"}"#);
        assert_eq!(dumps_with(&json!(["😀"]), Dumps { ensure_ascii: false, ..Dumps::default() }), "[\"😀\"]");
        assert_eq!(dumps(&json!(["😀"])), r#"["\ud83d\ude00"]"#);
    }
}
