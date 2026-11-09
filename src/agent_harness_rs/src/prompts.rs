//! Prompts as first-class, versioned objects — not string literals buried in code.
//!
//! ```
//! use agent_harness::prompts::Prompt;
//! use serde_json::json;
//!
//! let greet = Prompt::new("greet", "Say hello to {name} in {language}.");
//! let text = greet.render(&json!({"name": "Ada", "language": "French"})).unwrap();
//! assert_eq!(text, "Say hello to Ada in French.");
//! ```
//!
//! Templates use Python's `str.format` placeholders (`{name}`, `{user.name}`,
//! `{{` for a literal brace), so the same prompt files work from both
//! languages. A template containing a Jinja statement (`{%`) is rendered with
//! Jinja. A directory of Markdown files with YAML frontmatter becomes a
//! [`PromptLibrary`].

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};

/// Split `---\nkey: value\n---\nbody` into (metadata, body).
pub fn parse_frontmatter(text: &str) -> Result<(Map<String, Value>, String)> {
    if !text.starts_with("---") {
        return Ok((Map::new(), text.to_string()));
    }
    let parts: Vec<&str> = text.splitn(3, "---").collect();
    if parts.len() < 3 {
        return Ok((Map::new(), text.to_string()));
    }
    let meta: Value = serde_yaml::from_str(parts[1]).map_err(|e| Error::config(format!("bad frontmatter: {e}")))?;
    match meta {
        Value::Object(map) => Ok((map, parts[2].trim_start_matches('\n').to_string())),
        Value::Null => Ok((Map::new(), parts[2].trim_start_matches('\n').to_string())),
        _ => Ok((Map::new(), text.to_string())),
    }
}

/// One part of a prompt for [`sections`]: plain text, or a titled section.
pub enum Section<'a> {
    Text(&'a str),
    Titled(&'a str, &'a str),
}

/// Join non-empty prompt sections, titling the ones given as (heading, body).
pub fn sections(parts: &[Section<'_>]) -> String {
    let mut out = Vec::new();
    for part in parts {
        match part {
            Section::Text(t) if !t.trim().is_empty() => out.push(t.trim().to_string()),
            Section::Titled(h, b) if !b.trim().is_empty() => out.push(format!("## {h}\n{}", b.trim())),
            _ => {}
        }
    }
    out.join("\n\n")
}

/// How a template is rendered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// Jinja if the template contains `{%`, otherwise format.
    #[default]
    Auto,
    Format,
    Jinja,
}

/// A named, versioned template with default variables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prompt {
    pub name: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "one")]
    pub version: String,
    #[serde(default)]
    pub engine: Engine,
    #[serde(default)]
    pub variables: Map<String, Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Frontmatter keys this struct has no field for.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub path: Option<String>,
}

fn one() -> String {
    "1".into()
}

impl Prompt {
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            description: String::new(),
            version: one(),
            engine: Engine::Auto,
            variables: Map::new(),
            tags: Vec::new(),
            metadata: Map::new(),
            path: None,
        }
    }

    fn merged(&self, variables: &Value) -> Map<String, Value> {
        let mut merged = self.variables.clone();
        if let Value::Object(extra) = variables {
            merged.extend(extra.clone());
        }
        merged
    }

    fn uses_jinja(&self) -> bool {
        // Only a Jinja *statement* is unambiguous: `{{ }}` collides with the
        // `{{` escape that format templates use for a literal brace.
        self.engine == Engine::Jinja || (self.engine == Engine::Auto && self.template.contains("{%"))
    }

    /// Fill the template. Unknown placeholders survive as `{name}`.
    /// `variables` is a JSON object (`json!({...})`); anything else is ignored.
    pub fn render(&self, variables: &Value) -> Result<String> {
        self.render_with(variables, false)
    }

    /// Fill the template; a missing variable is an error.
    pub fn render_strict(&self, variables: &Value) -> Result<String> {
        self.render_with(variables, true)
    }

    fn render_with(&self, variables: &Value, strict: bool) -> Result<String> {
        let merged = self.merged(variables);
        if self.uses_jinja() {
            return render_jinja(&self.template, &merged);
        }
        format_template(&self.template, &merged, strict).map_err(|missing| Error::config(format!("prompt '{}' is missing variable '{missing}'", self.name)))
    }

    /// A copy with some variables already bound.
    pub fn partial(&self, variables: &Value) -> Prompt {
        let mut p = self.clone();
        p.variables = self.merged(variables);
        p
    }

    /// This prompt followed by another, separated by a blank line.
    pub fn join(&self, tail: &str) -> Prompt {
        let mut p = self.clone();
        p.template = format!("{}\n\n{tail}", self.template).trim().to_string();
        p
    }

    // ---- io -----------------------------------------------------------
    pub fn from_file(path: impl AsRef<Path>) -> Result<Prompt> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut p = Self::from_text(&text, &stem)?;
        p.path = Some(path.to_string_lossy().into_owned());
        Ok(p)
    }

    pub fn from_text(text: &str, default_name: &str) -> Result<Prompt> {
        let (mut meta, body) = parse_frontmatter(text)?;
        meta.entry("name").or_insert_with(|| Value::String(default_name.into()));
        // Coerce scalars the YAML parser read as numbers (`version: 2`).
        if let Some(v) = meta.get_mut("version")
            && !v.is_string()
        {
            *v = Value::String(v.to_string());
        }
        let mut known = Map::new();
        let mut extra = Map::new();
        for (k, v) in meta {
            match k.as_str() {
                "name" | "description" | "version" | "engine" | "variables" | "tags" | "metadata" => {
                    known.insert(k, v);
                }
                _ => {
                    extra.insert(k, v);
                }
            }
        }
        known.insert("template".into(), Value::String(body.trim().to_string()));
        let mut p: Prompt = serde_json::from_value(Value::Object(known)).map_err(|e| Error::config(format!("bad prompt frontmatter: {e}")))?;
        p.metadata.extend(extra);
        Ok(p)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut meta: IndexMap<&str, Value> = IndexMap::new();
        meta.insert("name", Value::String(self.name.clone()));
        meta.insert("description", Value::String(self.description.clone()));
        meta.insert("version", Value::String(self.version.clone()));
        if !self.tags.is_empty() {
            meta.insert("tags", serde_json::to_value(&self.tags).unwrap_or_default());
        }
        let front = serde_yaml::to_string(&meta).map_err(|e| Error::config(e.to_string()))?;
        std::fs::write(path, format!("---\n{}\n---\n\n{}\n", front.trim(), self.template))?;
        Ok(path.to_path_buf())
    }
}

impl std::fmt::Display for Prompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render(&Value::Null).unwrap_or_else(|_| self.template.clone()))
    }
}

/// Python's `str()` of a JSON value, which is what `str.format` would print.
pub(crate) fn display_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        other => other.to_string(),
    }
}

fn lookup<'a>(vars: &'a Map<String, Value>, field: &str) -> Option<&'a Value> {
    // `user.name` and `items[0]`, as str.format resolves them.
    let mut parts = field.split(['.', '[']).map(|p| p.trim_end_matches(']'));
    let mut current = vars.get(parts.next()?)?;
    for part in parts {
        current = match current {
            Value::Object(m) => m.get(part)?,
            Value::Array(a) => a.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(current)
}

/// Python `str.format`, minus format specs (`{x:>10}` renders `x` unpadded).
/// Returns the first missing field name as `Err` when `strict`.
fn format_template(template: &str, vars: &Map<String, Value>, strict: bool) -> std::result::Result<String, String> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut inner = String::new();
                let mut closed = false;
                for c in chars.by_ref() {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    inner.push(c);
                }
                if !closed {
                    out.push('{');
                    out.push_str(&inner);
                    continue;
                }
                let field = inner.split([':', '!']).next().unwrap_or("").trim();
                match lookup(vars, field) {
                    Some(v) => out.push_str(&display_value(v)),
                    None if strict => return Err(field.to_string()),
                    None => {
                        out.push('{');
                        out.push_str(field);
                        out.push('}');
                    }
                }
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

fn render_jinja(template: &str, vars: &Map<String, Value>) -> Result<String> {
    let mut env = minijinja::Environment::new();
    // A prompt is plain text on its way to a model; HTML-escaping corrupts it.
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::None);
    let syntax = minijinja::syntax::SyntaxConfig::builder().keep_trailing_newline(true).build().map_err(|e| Error::config(e.to_string()))?;
    env.set_syntax(syntax);
    env.render_str(template, minijinja::Value::from(minijinja::value::Serde(vars))).map_err(|e| Error::config(format!("jinja render failed: {e}")))
}

/// A folder of prompts, addressable by name.
#[derive(Debug, Clone, Default)]
pub struct PromptLibrary {
    prompts: IndexMap<String, Prompt>,
}

impl PromptLibrary {
    pub fn new<I: IntoIterator<Item = Prompt>>(prompts: I) -> Self {
        Self { prompts: prompts.into_iter().map(|p| (p.name.clone(), p)).collect() }
    }

    /// Every `*.md` file under `path` (recursively), sorted by path.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_matching(path, "*.md", true)
    }

    pub fn from_dir_matching(path: impl AsRef<Path>, pattern: &str, recursive: bool) -> Result<Self> {
        let root = path.as_ref();
        if !root.exists() {
            return Err(Error::config(format!("prompt directory not found: {}", root.display())));
        }
        let pat = if recursive { root.join("**").join(pattern) } else { root.join(pattern) };
        let mut files: Vec<PathBuf> =
            glob::glob(&pat.to_string_lossy()).map_err(|e| Error::config(e.to_string()))?.filter_map(std::result::Result::ok).filter(|p| p.is_file()).collect();
        files.sort();
        files.iter().map(Prompt::from_file).collect::<Result<Vec<_>>>().map(Self::new)
    }

    pub fn add(&mut self, prompt: Prompt) -> &Prompt {
        let name = prompt.name.clone();
        self.prompts.insert(name.clone(), prompt);
        &self.prompts[&name]
    }

    pub fn get(&self, name: &str) -> Result<&Prompt> {
        self.prompts.get(name).ok_or_else(|| {
            let known = if self.prompts.is_empty() { "none".into() } else { self.names().join(", ") };
            Error::config(format!("no prompt '{name}'; available: {known}"))
        })
    }

    pub fn render(&self, name: &str, variables: &Value) -> Result<String> {
        self.get(name)?.render(variables)
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.prompts.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn contains(&self, name: &str) -> bool {
        self.prompts.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Prompt> {
        self.prompts.values()
    }

    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn safe_render_keeps_unknown() {
        let p = Prompt::new("p", "Hi {name}, {missing} {{literal}}");
        assert_eq!(p.render(&json!({"name": "Ada"})).unwrap(), "Hi Ada, {missing} {literal}");
    }

    #[test]
    fn strict_render_errors() {
        let p = Prompt::new("greet", "Hi {name}");
        let err = p.render_strict(&json!({})).unwrap_err();
        assert_eq!(err.to_string(), "prompt 'greet' is missing variable 'name'");
    }

    #[test]
    fn attributes_and_defaults() {
        let p = Prompt::new("p", "{user.name} has {items[1]} ({n}, {ok})").partial(&json!({"n": 3}));
        let out = p.render(&json!({"user": {"name": "Bo"}, "items": ["a", "b"], "ok": true})).unwrap();
        assert_eq!(out, "Bo has b (3, True)");
    }

    #[test]
    fn jinja_is_detected() {
        let p = Prompt::new("p", "{% for x in xs %}{{ x }}&{% endfor %}");
        assert_eq!(p.render(&json!({"xs": [1, 2]})).unwrap(), "1&2&");
    }

    #[test]
    fn frontmatter_round_trip() {
        let text = "---\nname: greet\ndescription: says hi\nversion: 2\nowner: team-a\n---\n\nHello {who}\n";
        let p = Prompt::from_text(text, "fallback").unwrap();
        assert_eq!(p.name, "greet");
        assert_eq!(p.version, "2");
        assert_eq!(p.template, "Hello {who}");
        assert_eq!(p.metadata["owner"], "team-a");
        let dir = std::env::temp_dir().join(format!("ah-prompts-{}", crate::types::new_id("")));
        let file = p.save(dir.join("greet.md")).unwrap();
        let back = Prompt::from_file(&file).unwrap();
        assert_eq!(back.template, "Hello {who}");
        let lib = PromptLibrary::from_dir(&dir).unwrap();
        assert_eq!(lib.names(), vec!["greet"]);
        assert_eq!(lib.render("greet", &json!({"who": "you"})).unwrap(), "Hello you");
        assert!(lib.get("nope").is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn no_frontmatter() {
        let (meta, body) = parse_frontmatter("just text").unwrap();
        assert!(meta.is_empty());
        assert_eq!(body, "just text");
    }

    #[test]
    fn sections_join() {
        let s = sections(&[Section::Text(" intro "), Section::Titled("Rules", "be kind"), Section::Titled("Empty", "  ")]);
        assert_eq!(s, "intro\n\n## Rules\nbe kind");
    }
}
