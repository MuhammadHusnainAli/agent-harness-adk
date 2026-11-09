//! Skills: packaged know-how an agent can pull in when the task calls for it.
//!
//! A skill is a folder (or a single Markdown file) with YAML frontmatter:
//!
//! ```text
//! skills/refunds/SKILL.md
//! ---
//! name: refunds
//! description: How we process a refund, including the approval thresholds.
//! ---
//! 1. Check the order is inside the 30-day window...
//! ```
//!
//! Only the *name and description* of each skill go into the system prompt.
//! The body is loaded on demand through the `load_skill` tool.
//!
//! Python also imports a skill folder's `tools.py`. Rust cannot load code at
//! run time, so attach a skill's tools in code with [`Skill::with_tools`].

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::prompts::parse_frontmatter;
use crate::tools::{Tool, ToolContext};

const SKILL_FILES: [&str; 3] = ["SKILL.md", "skill.md", "README.md"];

/// Instructions plus optional tools and reference files, under one name.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub version: String,
    pub tags: Vec<String>,
    pub allowed_tools: Vec<String>,
    /// Bundled files, relative path → extension.
    pub resources: IndexMap<String, String>,
    pub metadata: Map<String, Value>,
    pub path: Option<String>,
    pub tools: Vec<Tool>,
}

fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Array(items)) => items.iter().filter_map(|i| i.as_str().map(str::to_string)).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => vec![],
    }
}

fn first_line(text: &str) -> String {
    text.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#')).unwrap_or("").to_string()
}

impl Skill {
    pub fn new(name: impl Into<String>, description: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            version: "1".into(),
            tags: vec![],
            allowed_tools: vec![],
            resources: IndexMap::new(),
            metadata: Map::new(),
            path: None,
            tools: vec![],
        }
    }

    /// Tools this skill brings with it.
    pub fn with_tools<I: IntoIterator<Item = Tool>>(mut self, tools: I) -> Self {
        self.tools.extend(tools);
        self
    }

    /// One Markdown file. A file named `SKILL.md` takes its folder's name.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Skill> {
        let file = path.as_ref();
        let (mut meta, body) = parse_frontmatter(&std::fs::read_to_string(file)?)?;
        let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let folder = file.parent().unwrap_or(Path::new("."));
        let default_name = if stem.eq_ignore_ascii_case("skill") { folder.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(stem) } else { stem };
        let take = |meta: &mut Map<String, Value>, k: &str| meta.remove(k);
        let name = take(&mut meta, "name").and_then(|v| v.as_str().map(str::to_string)).unwrap_or(default_name);
        let description = take(&mut meta, "description").and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| first_line(&body));
        let version = take(&mut meta, "version").map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_else(|| "1".into());
        let tags = strings(take(&mut meta, "tags").as_ref());
        let allowed_tools = strings(take(&mut meta, "allowed_tools").or_else(|| take(&mut meta, "allowed-tools")).as_ref());
        let mut metadata = match take(&mut meta, "metadata") {
            Some(Value::Object(m)) => m,
            _ => Map::new(),
        };
        metadata.extend(meta);
        Ok(Skill {
            name,
            description,
            instructions: body.trim().to_string(),
            version,
            tags,
            allowed_tools,
            resources: IndexMap::new(),
            metadata,
            path: Some(folder.to_string_lossy().into_owned()),
            tools: vec![],
        })
    }

    /// A skill folder: its `SKILL.md` plus every other file, listed as a resource.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Skill> {
        let root = path.as_ref();
        let doc = SKILL_FILES.iter().map(|n| root.join(n)).find(|p| p.exists()).ok_or_else(|| Error::config(format!("{} has no SKILL.md", root.display())))?;
        let mut skill = Self::from_file(&doc)?;
        skill.path = Some(root.to_string_lossy().into_owned());
        let pattern = root.join("**").join("*");
        let mut extras: Vec<PathBuf> = glob::glob(&pattern.to_string_lossy())
            .map_err(|e| Error::config(e.to_string()))?
            .filter_map(std::result::Result::ok)
            .filter(|p| p.is_file() && *p != doc && p.extension().is_none_or(|e| e != "pyc"))
            .collect();
        extras.sort();
        for extra in extras {
            let rel = extra.strip_prefix(root).unwrap_or(&extra).to_string_lossy().replace('\\', "/");
            let ext = extra.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            skill.resources.insert(rel, ext);
        }
        Ok(skill)
    }

    /// The one line that goes into the system prompt.
    pub fn index_entry(&self) -> String {
        format!("- {}: {}", self.name, self.description)
    }

    /// The full text, handed over when the agent loads the skill.
    pub fn body(&self) -> String {
        let mut out = vec![format!("# Skill: {}", self.name), self.instructions.trim().to_string()];
        if !self.resources.is_empty() {
            let base = Path::new(self.path.as_deref().unwrap_or("."));
            let mut names: Vec<&String> = self.resources.keys().collect();
            names.sort();
            let listing: Vec<String> = names.iter().map(|n| format!("- {}", base.join(n).display())).collect();
            out.push(format!("## Files bundled with this skill\n{}", listing.join("\n")));
        }
        if !self.tools.is_empty() {
            let names: Vec<&str> = self.tools.iter().map(|t| t.name.as_str()).collect();
            out.push(format!("## Tools this skill brings\n{}", names.join(", ")));
        }
        out.into_iter().filter(|p| !p.trim().is_empty()).collect::<Vec<_>>().join("\n\n")
    }
}

/// The skills an agent knows about, plus the tool that loads one.
#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: IndexMap<String, Skill>,
}

/// Load one skill's instructions.
#[derive(Deserialize, JsonSchema)]
struct LoadSkill {
    /// the skill's name, exactly as listed in the skills catalogue.
    name: String,
}

impl SkillRegistry {
    pub fn new<I: IntoIterator<Item = Skill>>(skills: I) -> Self {
        let mut reg = Self::default();
        for s in skills {
            reg.add(s);
        }
        reg
    }

    /// Every skill folder (or loose `.md` file) directly under `path`.
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref();
        if !root.exists() {
            return Err(Error::config(format!("skills directory not found: {}", root.display())));
        }
        let mut children: Vec<PathBuf> = std::fs::read_dir(root)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        children.sort();
        let mut found = Vec::new();
        for child in children {
            if child.is_dir() && SKILL_FILES.iter().any(|n| child.join(n).exists()) {
                found.push(Skill::from_dir(&child)?);
            } else if child.is_file() && child.extension().is_some_and(|e| e == "md") {
                found.push(Skill::from_file(&child)?);
            }
        }
        Ok(Self::new(found))
    }

    pub fn add(&mut self, skill: Skill) -> &Skill {
        let name = skill.name.clone();
        self.skills.insert(name.clone(), skill);
        &self.skills[&name]
    }

    /// Load a skill from a folder or a Markdown file and add it.
    pub fn add_path(&mut self, path: impl AsRef<Path>) -> Result<&Skill> {
        let p = path.as_ref();
        let skill = if p.is_dir() { Skill::from_dir(p)? } else { Skill::from_file(p)? };
        Ok(self.add(skill))
    }

    pub fn get(&self, name: &str) -> Result<&Skill> {
        self.skills.get(name).ok_or_else(|| {
            let known = if self.skills.is_empty() { "none".into() } else { self.names().join(", ") };
            Error::config(format!("no skill '{name}'; available: {known}"))
        })
    }

    pub fn select(&self, names: Option<&[String]>) -> SkillRegistry {
        match names {
            None => self.clone(),
            Some(names) => Self::new(names.iter().filter_map(|n| self.skills.get(n).cloned())),
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.skills.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.values()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// The catalogue block for the system prompt — names and one-liners only.
    pub fn index(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = self.skills.values().map(Skill::index_entry).collect();
        format!("Skills available to you (call `load_skill` to read one in full before you rely on it):\n{}", lines.join("\n"))
    }

    /// `load_skill`, plus the tools the skills bring.
    pub fn tools(&self) -> Vec<Tool> {
        let mut out = vec![self.load_tool()];
        out.extend(self.skills.values().flat_map(|s| s.tools.iter().cloned()));
        out
    }

    /// The tool that reads one skill's full instructions.
    pub fn load_tool(&self) -> Tool {
        let registry = self.clone();
        Tool::new("load_skill", move |args: LoadSkill, _ctx: ToolContext| {
            let body = registry.get(&args.name).map(Skill::body);
            async move { body }
        })
        .describe("Read a skill's full instructions before doing work it covers.")
        .tags(["builtin", "skills"])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::NoArgs;

    fn tmp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ah-skills-{}", crate::types::new_id("")));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn loads_folder_and_file_skills() {
        let root = tmp();
        let refunds = root.join("refunds");
        std::fs::create_dir_all(refunds.join("forms")).unwrap();
        std::fs::write(refunds.join("SKILL.md"), "---\ndescription: How refunds work.\ntags: [billing]\n---\n1. Check the window.\n").unwrap();
        std::fs::write(refunds.join("forms").join("refund.txt"), "form").unwrap();
        std::fs::write(root.join("tone.md"), "# Tone\nBe brief and kind.\n").unwrap();

        let reg = SkillRegistry::from_dir(&root).unwrap();
        assert_eq!(reg.names(), vec!["refunds", "tone"]);
        let refunds_skill = reg.get("refunds").unwrap();
        assert_eq!(refunds_skill.tags, vec!["billing"]);
        assert_eq!(refunds_skill.resources.get("forms/refund.txt").map(String::as_str), Some("txt"));
        assert_eq!(reg.get("tone").unwrap().description, "Be brief and kind.");
        assert!(reg.index().contains("- refunds: How refunds work."));

        let load = reg.load_tool();
        let args = serde_json::json!({"name": "refunds"}).as_object().unwrap().clone();
        let out = load.run("c", args, ToolContext::default()).await;
        assert!(out.content.starts_with("# Skill: refunds\n\n1. Check the window."), "{}", out.content);
        assert!(out.content.contains("## Files bundled with this skill"));
        let missing = load.run("c", serde_json::json!({"name": "x"}).as_object().unwrap().clone(), ToolContext::default()).await;
        assert!(missing.is_error);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tools_include_loader_and_bundled() {
        let t = Tool::new("calc", |_: NoArgs, _| async { Ok(1) });
        let reg = SkillRegistry::new([Skill::new("math", "Do sums.", "Use calc.").with_tools([t])]);
        let names: Vec<String> = reg.tools().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["load_skill", "calc"]);
        assert!(reg.get("math").unwrap().body().contains("## Tools this skill brings\ncalc"));
        assert!(SkillRegistry::default().index().is_empty());
    }
}
