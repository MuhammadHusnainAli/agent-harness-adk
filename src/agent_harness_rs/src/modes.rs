//! Modes: what kind of work an agent does, and how hard it works at it.
//!
//! ```no_run
//! use agent_harness::{Agent, modes};
//! # fn demo() -> agent_harness::Result<()> {
//! let helper = Agent::builder("helper").mode(modes::chat(None)?).build()?;
//! let analyst = Agent::builder("analyst").mode(modes::research(Some("deep"))?).build()?;
//! # Ok(()) }
//! ```
//!
//! An agent with no mode is the plain loop. A mode is a way of working laid
//! over it:
//!
//! - **chat** — a conversation; the agent keeps the thread between calls.
//! - **research** — a report someone can check: questions planned, every
//!   source recorded in a ledger and cited by number. A citation to something
//!   never recorded, or a source never actually read, is refused.
//! - **cowork** — a task carried through with a todo list, helpers, and
//!   questions to the person. (Its workspace, document and Python tools, and
//!   the files it writes coming back, arrive with workspaces in M5.)
//! - **voice** — short answers written for the ear.
//!
//! `depth` (`fast`, `balanced`, `deep`) sets the step ceiling, the sources a
//! report needs, the helpers it may start and — when named — the model tier.
//! A mode only fills in what you left unset.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use futures::future::BoxFuture;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::tools::{Tool, ToolContext};
use crate::types::{Artifact, Source, Todo, TodoStatus};

pub const DEPTHS: [&str; 3] = ["fast", "balanced", "deep"];

/// Tools carrying this tag belong to the agent that owns the run — its todo
/// list, its line to the person. A sub-agent never inherits them.
pub const LEAD: &str = "lead";

const EVIDENCE_CHARS: usize = 4_000_000;
const MAX_TODOS: usize = 60;

/// Answers `ask_user`: `(question, options)` → the person's answer.
pub type Asker = Arc<dyn Fn(String, Vec<String>) -> BoxFuture<'static, String> + Send + Sync>;

// ---------------------------------------------------------------------------
// citations
// ---------------------------------------------------------------------------
static FENCED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```").expect("valid regex"));
static CITE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[(\d{1,4}(?:\s*[,;–-]\s*\d{1,4})*)\]").expect("valid regex"));
static SOURCE_LIST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?im)^\s{0,3}(?:#{1,6}\s*|\*\*)?(?:sources|references|bibliography)\b").expect("valid regex"));

/// The source numbers a text cites: `[3]`, `[1, 2]` and `[4-6]` all count.
/// Code fences are skipped, and so is an index like `rows[0]`.
pub fn citations(text: &str) -> BTreeSet<u32> {
    let mut found = BTreeSet::new();
    let cleaned = FENCED.replace_all(text, " ");
    for cap in CITE.captures_iter(&cleaned) {
        // Python's `(?<!\w)`: a bracket hard against a word (`rows[0]`) is an index.
        let start = cap.get(0).expect("whole match").start();
        if cleaned[..start].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        for part in cap[1].split([',', ';']) {
            let ends: Vec<u32> = part.split(['–', '-']).filter_map(|n| n.trim().parse().ok()).collect();
            if ends.len() == 2 && ends[1] > ends[0] && ends[1] - ends[0] <= 50 {
                found.extend(ends[0]..=ends[1]);
            } else {
                found.extend(ends);
            }
        }
    }
    found
}

/// A reference as it is compared: case, a trailing slash and a fragment are
/// not what makes two sources different.
fn normalise(r: &str) -> String {
    let mut text = r.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase().trim_end_matches(['.', ',', ';']).to_string();
    if text.contains("://") {
        text = text.split('#').next().unwrap_or("").trim_end_matches('/').to_string();
    }
    text
}

// ---------------------------------------------------------------------------
// the notebook
// ---------------------------------------------------------------------------
/// What a mode keeps while it works: the todo list and the source ledger.
///
/// One per run. Sub-agents get a [`Notebook::child`] — their own todo list but
/// the same ledger, so a source a helper records has one number across the run.
#[derive(Debug, Default)]
pub struct Notebook {
    todos: Mutex<Vec<Todo>>,
    sources: Arc<Mutex<Vec<Source>>>,
    evidence: Arc<Mutex<VecDeque<String>>>,
    /// This run keeps a source ledger, so its helpers write in it too.
    pub ledger: bool,
    /// Keep what tools returned, so a recorded source can be checked against it.
    pub tracking: bool,
    questions: AtomicU32,
    /// Bumped on every change, so the loop knows when to report progress.
    revision: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Notebook {
    pub fn new(ledger: bool, tracking: bool) -> Self {
        Self { ledger, tracking, ..Default::default() }
    }

    pub fn child(&self) -> Notebook {
        Notebook { sources: self.sources.clone(), evidence: self.evidence.clone(), ledger: self.ledger, tracking: self.tracking, ..Default::default() }
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::SeqCst)
    }

    /// What a paused run keeps of it — Python's keys, so either can resume it.
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "todos": self.todos(), "sources": self.sources(),
            "evidence": lock(&self.evidence).iter().cloned().collect::<Vec<_>>(),
            "ledger": self.ledger, "tracking": self.tracking, "questions": self.questions(),
        })
    }

    /// A notebook from a snapshot.
    pub fn restore(held: &serde_json::Value) -> Notebook {
        let n = Notebook::new(held["ledger"].as_bool().unwrap_or(false), held["tracking"].as_bool().unwrap_or(false));
        *lock(&n.todos) = serde_json::from_value(held["todos"].clone()).unwrap_or_default();
        *lock(&n.sources) = serde_json::from_value(held["sources"].clone()).unwrap_or_default();
        *lock(&n.evidence) = serde_json::from_value(held["evidence"].clone()).unwrap_or_default();
        n.questions.store(held["questions"].as_u64().unwrap_or(0) as u32, Ordering::SeqCst);
        n
    }

    fn bump(&self) {
        self.revision.fetch_add(1, Ordering::SeqCst);
    }

    pub fn todos(&self) -> Vec<Todo> {
        lock(&self.todos).clone()
    }

    pub fn sources(&self) -> Vec<Source> {
        lock(&self.sources).clone()
    }

    pub fn questions(&self) -> u32 {
        self.questions.load(Ordering::SeqCst)
    }

    pub fn write_todos(&self, items: Vec<Todo>) {
        *lock(&self.todos) = items;
        self.bump();
    }

    pub fn open_todos(&self) -> Vec<Todo> {
        lock(&self.todos).iter().filter(|t| t.is_open()).cloned().collect()
    }

    pub fn todo_summary(&self) -> String {
        let todos = lock(&self.todos);
        if todos.is_empty() {
            return "no items".into();
        }
        let parts: Vec<String> =
            [(TodoStatus::Done, "done"), (TodoStatus::InProgress, "in progress"), (TodoStatus::Pending, "pending"), (TodoStatus::Skipped, "skipped")]
                .iter()
                .filter_map(|(s, label)| {
                    let n = todos.iter().filter(|t| t.status == *s).count();
                    (n > 0).then(|| format!("{n} {label}"))
                })
                .collect();
        format!("{} items · {}", todos.len(), parts.join(" · "))
    }

    /// Remember what the run has read — tool arguments and tool results.
    pub fn saw(&self, texts: &[&str]) {
        if !self.tracking {
            return;
        }
        let mut ev = lock(&self.evidence);
        for t in texts.iter().filter(|t| !t.is_empty()) {
            ev.push_back((*t).to_string());
        }
        let mut total: usize = ev.iter().map(String::len).sum();
        while total > EVIDENCE_CHARS && ev.len() > 1 {
            total -= ev.pop_front().map_or(0, |s| s.len());
        }
    }

    /// Did anything this run read actually mention `ref`?
    pub fn mentions(&self, r: &str) -> bool {
        let needle = normalise(r);
        if needle.chars().count() < 3 {
            return false;
        }
        let bare = needle.split("://").last().unwrap_or(&needle).to_string();
        let mut needles = vec![needle.clone(), bare.clone()];
        if let Some(rest) = bare.strip_prefix("www.") {
            needles.push(rest.to_string());
        }
        lock(&self.evidence).iter().any(|text| {
            let lowered = text.to_lowercase();
            needles.iter().any(|n| lowered.contains(n.as_str()))
        })
    }

    pub fn find(&self, r: &str) -> Option<Source> {
        let key = normalise(r);
        lock(&self.sources).iter().find(|s| normalise(&s.r#ref) == key).cloned()
    }

    /// Add a source to the ledger, or return the number it already has.
    pub fn record(&self, r: &str, finding: &str, title: &str, agent: &str) -> Source {
        let key = normalise(r);
        let mut sources = lock(&self.sources);
        let out = if let Some(existing) = sources.iter_mut().find(|s| normalise(&s.r#ref) == key) {
            if !finding.is_empty() && !existing.finding.contains(finding) {
                existing.finding = format!("{}\n{finding}", existing.finding).trim().to_string();
            }
            if !title.is_empty() && existing.title.is_empty() {
                existing.title = title.into();
            }
            existing.clone()
        } else {
            let s = Source {
                id: sources.len() as u32 + 1,
                r#ref: r.split_whitespace().collect::<Vec<_>>().join(" "),
                title: title.trim().into(),
                finding: finding.trim().into(),
                agent: agent.into(),
            };
            sources.push(s.clone());
            s
        };
        drop(sources);
        self.bump();
        out
    }

    /// What a long run cannot afford to lose when its context is compacted.
    pub fn pins(&self) -> Vec<String> {
        let mut out = Vec::new();
        let todos = self.todos();
        if !todos.is_empty() {
            out.push(format!("Your todo list: {}", todos.iter().map(Todo::line).collect::<Vec<_>>().join("; ")));
        }
        let sources = self.sources();
        if !sources.is_empty() {
            let lines: Vec<String> = sources
                .iter()
                .map(|s| {
                    let f: String = s.finding.chars().take(160).collect();
                    if f.is_empty() { s.line() } else { format!("{} ({f})", s.line()) }
                })
                .collect();
            out.push(format!("Sources recorded so far: {}", lines.join("; ")));
        }
        out
    }

    pub fn progress(&self) -> Value {
        json!({"todos": self.todos(), "sources": lock(&self.sources).len(), "summary": self.todo_summary()})
    }
}

// ---------------------------------------------------------------------------
// the mode
// ---------------------------------------------------------------------------
/// A way of working, resolved at one depth. Build one with [`chat`],
/// [`research`], [`cowork`] or [`voice`]; change a field with struct update.
#[derive(Clone)]
pub struct Mode {
    pub name: String,
    pub depth: String,
    pub description: String,
    /// The model tier this depth asks for — only when a depth was named.
    pub tier: Option<String>,
    pub max_steps: u32,
    /// How many specialists the agent may spin up. 0 turns it off.
    pub helpers: u32,
    pub workspace: bool,
    /// Consecutive runs continue one conversation.
    pub conversational: bool,
    /// Keeps a todo list, and may not finish with items still open.
    pub todos: bool,
    /// Keeps a source ledger, and may only cite what is in it.
    pub sources: bool,
    pub min_sources: u32,
    /// A source may only be recorded if something the run read mentions it.
    pub verify_sources: bool,
    /// Answers `ask_user`; without one the agent cannot ask.
    pub ask: Option<Asker>,
    pub max_questions: u32,
    pub documents: bool,
    pub python: bool,
    pub collect_files: bool,
    /// The finished answer is kept as this artefact.
    pub report: String,
    /// How many times an unfinished answer is sent back.
    pub retries: u32,
    /// (tool this line needs, the line). "" always; "!name" when it lacks it.
    pub guidance: Vec<(String, String)>,
    pub opening: String,
    pub closing: String,
    pub wrap_up: String,
}

impl std::fmt::Debug for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mode").field("name", &self.name).field("depth", &self.depth).field("max_steps", &self.max_steps).finish_non_exhaustive()
    }
}

impl Default for Mode {
    fn default() -> Self {
        Self {
            name: String::new(),
            depth: "balanced".into(),
            description: String::new(),
            tier: None,
            max_steps: 20,
            helpers: 0,
            workspace: false,
            conversational: false,
            todos: false,
            sources: false,
            min_sources: 0,
            verify_sources: true,
            ask: None,
            max_questions: 5,
            documents: false,
            python: false,
            collect_files: false,
            report: String::new(),
            retries: 2,
            guidance: vec![],
            opening: String::new(),
            closing: String::new(),
            wrap_up: "This is your last step. Answer now with what you have.".into(),
        }
    }
}

impl Mode {
    fn checked(self) -> Result<Self> {
        if !DEPTHS.contains(&self.depth.as_str()) {
            return Err(Error::config(format!("depth must be one of {} — got '{}'", DEPTHS.join(", "), self.depth)));
        }
        if self.max_steps < 1 {
            return Err(Error::config(format!("{}: max_steps must be at least 1", self.name)));
        }
        if self.helpers > 100 {
            return Err(Error::config(format!("{}: helpers must be between 0 and 100 — got {}", self.name, self.helpers)));
        }
        Ok(self)
    }

    pub fn keeps_notebook(&self) -> bool {
        self.todos || self.sources
    }

    /// The operating instructions, for the tools this agent actually has.
    pub fn prompt(&self, tool_names: &[String]) -> String {
        let has = |n: &str| tool_names.iter().any(|t| t == n);
        let lines: Vec<&str> = self
            .guidance
            .iter()
            .filter(|(needs, _)| match needs.strip_prefix('!') {
                Some(lacking) => !has(lacking),
                None => needs.is_empty() || has(needs),
            })
            .map(|(_, line)| line.as_str())
            .collect();
        let body: Vec<String> = lines.iter().enumerate().map(|(i, l)| format!("{}. {l}", i + 1)).collect();
        [self.opening.clone(), body.join("\n"), self.closing.clone()].into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n")
    }

    /// The tools this mode adds.
    pub fn tools(&self) -> Vec<Tool> {
        let mut built = Vec::new();
        if self.todos {
            built.push(todo_tool());
        }
        if self.sources {
            built.push(source_tool(self.verify_sources));
        }
        if let Some(ask) = &self.ask {
            built.push(ask_tool(ask.clone(), self.max_questions));
        }
        built
    }

    /// What this mode still requires of an answer. Empty means done.
    pub fn unmet(&self, output: &str, notebook: Option<&Notebook>) -> Vec<String> {
        let Some(nb) = notebook else { return vec![] };
        let mut problems = Vec::new();
        if self.todos {
            let left = nb.open_todos();
            if !left.is_empty() {
                let names: Vec<&str> = left.iter().take(8).map(|t| t.content.as_str()).collect();
                let more = if left.len() > 8 { format!(" (and {} more)", left.len() - 8) } else { String::new() };
                problems.push(format!(
                    "items on your todo list are still open: {}{more} — finish them, or mark each one `skipped` with a note saying why",
                    names.join("; ")
                ));
            }
        }
        if self.sources {
            let ids: BTreeSet<u32> = nb.sources().iter().map(|s| s.id).collect();
            let cited = citations(output);
            let unknown: Vec<String> = cited.difference(&ids).map(|n| format!("[{n}]")).collect();
            if !unknown.is_empty() {
                problems.push(format!("you cite {} but no such source is recorded — cite only the numbers `record_source` gave you", unknown.join(", ")));
            }
            if (ids.len() as u32) < self.min_sources {
                problems.push(format!(
                    "{} of the {} sources this report needs are recorded — read more and record each with `record_source`, or say in the report what could not be found",
                    ids.len(),
                    self.min_sources
                ));
            } else if !ids.is_empty() && cited.intersection(&ids).next().is_none() {
                problems.push("the report cites none of the sources you recorded — put [n] straight after each claim that source supports".into());
            }
        }
        problems
    }

    /// The message the agent is given so it can finish properly.
    pub fn feedback(problems: &[String]) -> String {
        let body: Vec<String> = problems.iter().map(|p| format!("- {p}")).collect();
        format!("That is not finished yet:\n{}\n\nPut it right, then answer again.", body.join("\n"))
    }

    /// The answer as delivered: a report gains its source list.
    pub fn close(&self, output: &str, notebook: Option<&Notebook>) -> String {
        let Some(nb) = notebook else { return output.into() };
        let sources = nb.sources();
        if !self.sources || sources.is_empty() || output.is_empty() || SOURCE_LIST.is_match(&FENCED.replace_all(output, " ")) {
            return output.into();
        }
        let cited = citations(output);
        let rows: Vec<&Source> = sources.iter().filter(|s| cited.contains(&s.id)).collect();
        let (heading, listed): (&str, Vec<&Source>) = if rows.is_empty() { ("Sources consulted", sources.iter().collect()) } else { ("Sources", rows) };
        let listing: Vec<String> = listed.iter().map(|s| format!("- {}", s.line())).collect();
        format!("{}\n\n## {heading}\n{}", output.trim_end(), listing.join("\n"))
    }

    /// What the run leaves behind: the report.
    pub fn artifacts(&self, output: &str, agent: &str) -> Vec<Artifact> {
        if self.report.is_empty() || output.is_empty() {
            return vec![];
        }
        vec![Artifact { media_type: "text/markdown".into(), produced_by: agent.into(), ..Artifact::new(self.report.clone(), output) }]
    }
}

/// The most files a run hands back, and the largest one carried as text.
const MAX_FILES: usize = 200;
const MAX_FILE_CHARS: u64 = 400_000;

/// The files a run wrote in its workspace since `before`, as artefacts: text
/// carried as content, anything else (or anything too large) by its path.
pub async fn collect_files(
    workspace: &dyn crate::runtime::workspace::Workspace,
    before: &crate::runtime::workspace::Snapshot,
    agent: &str,
) -> Result<Vec<Artifact>> {
    let mut out = Vec::new();
    for relative in workspace.changed(before).await?.into_iter().take(MAX_FILES) {
        let guessed = mime_guess::from_path(&relative).first().map(|m| m.essence_str().to_string());
        let mut media_type = guessed.clone().unwrap_or_else(|| "text/plain".into());
        let mut content = String::new();
        let path = match workspace.materialize(&relative).await {
            Ok(p) => {
                let small = std::fs::metadata(&p).is_ok_and(|m| m.len() <= MAX_FILE_CHARS);
                match std::fs::read(&p).ok().filter(|_| small).map(String::from_utf8) {
                    Some(Ok(text)) => content = text,
                    // Not text: the path is what is worth handing back.
                    _ => media_type = guessed.unwrap_or_else(|| "application/octet-stream".into()),
                }
                p.display().to_string()
            }
            // Too large to bring down, or gone again: say where it is instead.
            Err(_) => format!("{}/{relative}", workspace.location()),
        };
        out.push(Artifact { path: Some(path), media_type, produced_by: agent.into(), ..Artifact::new(relative, content) });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// tools
// ---------------------------------------------------------------------------
fn notebook(ctx: &ToolContext, tool: &str) -> Result<Arc<Notebook>> {
    ctx.run.as_ref().and_then(|s| s.notebook.clone()).ok_or_else(|| Error::tool(format!("{tool} only works inside a run of an agent with a mode"), tool))
}

fn todo_tool() -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {"todos": {
            "type": "array",
            "description": "Every item on the list, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What is to be done, in one line."},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "done", "skipped"]},
                    "note": {"type": "string", "description": "The outcome, or why it was skipped. Required when skipped."},
                },
                "required": ["content", "status"],
            },
        }},
        "required": ["todos"],
    });
    Tool::raw(
        "todo_write",
        "Write your todo list: the whole list, every time. Call it before you start to lay out the steps, then again whenever something changes — an item started, finished, added or dropped. You may not finish while an item is still pending or in progress.",
        schema,
        |args: Map<String, Value>, ctx: ToolContext| async move {
            let nb = notebook(&ctx, "todo_write")?;
            let err = |m: String| Error::tool(m, "todo_write");
            let raw = args.get("todos").and_then(Value::as_array).cloned().unwrap_or_default();
            if raw.len() > MAX_TODOS {
                return Err(err(format!("a todo list holds at most {MAX_TODOS} items — group the small ones")));
            }
            let mut items = Vec::new();
            for (i, item) in raw.iter().enumerate() {
                let n = i + 1;
                let Value::Object(o) = item else { return Err(err(format!("item {n} must be an object with `content` and `status`"))) };
                let s = |k: &str| o.get(k).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).unwrap_or_default().trim().to_string();
                let content = s("content");
                let status = s("status").to_lowercase();
                let status = match status.as_str() {
                    "" | "todo" | "pending" => "pending",
                    "completed" | "complete" | "done" => "done",
                    "doing" | "in-progress" | "in_progress" => "in_progress",
                    "cancelled" | "canceled" | "skipped" => "skipped",
                    other => return Err(err(format!("item {n}: status must be pending, in_progress, done or skipped — got '{other}'"))),
                };
                let note = s("note");
                if content.is_empty() {
                    return Err(err(format!("item {n} has no `content`")));
                }
                if status == "skipped" && note.is_empty() {
                    return Err(err(format!("item {n} is skipped without a `note` — say why it was not done")));
                }
                let status: TodoStatus = serde_json::from_value(json!(status)).expect("known status");
                items.push(Todo { content, status, note });
            }
            nb.write_todos(items);
            let listing: Vec<String> = nb.todos().iter().map(Todo::line).collect();
            Ok(json!(format!("{}\n{}", nb.todo_summary(), listing.join("\n")).trim().to_string()))
        },
    )
    .tags(["builtin", "mode", LEAD])
}

fn source_tool(verify: bool) -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The URL, document title, file path or id, exactly as it appeared in what you read."},
            "finding": {"type": "string", "description": "What this source established — the fact, figure or quotation you will rely on."},
            "title": {"type": "string", "description": "A readable title, if the ref is a URL or an id."},
        },
        "required": ["ref", "finding"],
    });
    Tool::raw(
        "record_source",
        "Record a source you have actually read and what it established. It returns the number to cite it by: put [n] after each claim it supports. Recording the same source again returns the same number.",
        schema,
        move |args: Map<String, Value>, ctx: ToolContext| async move {
            let nb = notebook(&ctx, "record_source")?;
            let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            let r = s("ref").split_whitespace().collect::<Vec<_>>().join(" ");
            let err = |m: String| Error::tool(m, "record_source");
            if r.is_empty() {
                return Err(err("a source needs a `ref` — its URL, title, file path or id".into()));
            }
            if s("finding").trim().is_empty() {
                return Err(err("say what this source established in `finding`".into()));
            }
            if verify && nb.tracking && nb.find(&r).is_none() && !nb.mentions(&r) {
                return Err(err(format!(
                    "nothing you have read in this run mentions '{r}'. Record a source only after a tool has returned it, and give the URL, title or id exactly as it appeared."
                )));
            }
            let source = nb.record(&r, &s("finding"), &s("title"), &ctx.agent);
            Ok(json!(format!("Recorded as [{}] — cite it as [{}]. {} sources recorded so far.", source.id, source.id, nb.sources().len())))
        },
    )
    .tags(["builtin", "mode", "research"])
}

fn ask_tool(ask: Asker, max_questions: u32) -> Tool {
    let schema = json!({
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "One clear question, with the context needed to answer it."},
            "options": {"type": "array", "items": {"type": "string"}, "description": "The choices, if there are only a few."},
        },
        "required": ["question"],
    });
    Tool::raw(
        "ask_user",
        "Ask the person you are working for one question and wait for the answer. Use it only for a decision that is theirs to make and that changes what you do next — never for something you can look up or reasonably assume.",
        schema,
        move |args: Map<String, Value>, ctx: ToolContext| {
            let ask = ask.clone();
            async move {
                let nb = notebook(&ctx, "ask_user")?;
                let question = args.get("question").and_then(Value::as_str).unwrap_or("").trim().to_string();
                if question.is_empty() {
                    return Err(Error::tool("there is no question to ask", "ask_user"));
                }
                if nb.questions.load(Ordering::SeqCst) >= max_questions {
                    return Ok(json!(format!("You have asked {max_questions} questions, which is the limit for one run. Decide this one yourself and say what you assumed.")));
                }
                nb.questions.fetch_add(1, Ordering::SeqCst);
                let options: Vec<String> = args.get("options").and_then(Value::as_array).map(|a| a.iter().filter_map(|o| o.as_str().map(str::to_string)).collect()).unwrap_or_default();
                let answer = ask(question, options).await.trim().to_string();
                Ok(json!(if answer.is_empty() { "(no answer) — make the reasonable assumption and say what you assumed.".to_string() } else { answer }))
            }
        },
    )
    .tags(["builtin", "mode", "human", LEAD])
}

// ---------------------------------------------------------------------------
// the modes
// ---------------------------------------------------------------------------
/// (the depth to work at, the tier it asks for). Naming a depth asks for a tier.
fn depth(value: Option<&str>) -> Result<(String, Option<String>)> {
    let Some(v) = value else { return Ok(("balanced".into(), None)) };
    let name = v.trim().to_lowercase().replace('-', "_");
    let name = match name.as_str() {
        "quick" | "light" => "fast",
        "standard" | "normal" | "medium" | "default" => "balanced",
        "slow" | "thorough" | "extended" => "deep",
        other => other,
    }
    .to_string();
    if !DEPTHS.contains(&name.as_str()) {
        return Err(Error::config(format!("depth must be one of {} — got '{v}'", DEPTHS.join(", "))));
    }
    Ok((name.clone(), Some(name)))
}

fn g(needs: &str, line: &str) -> (String, String) {
    (needs.into(), line.into())
}

fn by_depth<T: Copy>(level: &str, fast: T, balanced: T, deep: T) -> T {
    match level {
        "fast" => fast,
        "deep" => deep,
        _ => balanced,
    }
}

/// A conversation: quick turns, and the thread kept between them.
pub fn chat(level: Option<&str>) -> Result<Mode> {
    let (level, tier) = depth(level)?;
    Mode {
        name: "chat".into(),
        max_steps: by_depth(&level, 6, 12, 20),
        depth: level,
        tier,
        description: "A conversation that keeps its thread between turns.".into(),
        conversational: true,
        opening: "You are in a conversation. Answer the message in front of you, directly, at the length it deserves — a short question gets a short answer."
            .into(),
        guidance: vec![
            g("", "Earlier turns are context. Do not ask for what you were already told, and do not repeat what you already said."),
            g("", "If a request is ambiguous in a way that changes the answer, ask one question instead of guessing."),
            g("", "Reach for a tool when it gets you a fact you do not have. Do not turn a question into a project."),
        ],
        ..Default::default()
    }
    .checked()
}

/// A cited report: plan the questions, read, record sources, cite by number.
pub fn research(level: Option<&str>) -> Result<Mode> {
    let (level, tier) = depth(level)?;
    let need = by_depth(&level, 2, 4, 8);
    let plural = if need == 1 { "" } else { "s" };
    Mode {
        name: "research".into(),
        max_steps: by_depth(&level, 12, 30, 60),
        helpers: by_depth(&level, 0, 3, 6),
        depth: level,
        tier,
        description: "A cited report built from sources the agent read and recorded.".into(),
        todos: true,
        sources: true,
        min_sources: need,
        report: "report.md".into(),
        opening: "You are doing research. The deliverable is a report someone else can check, so every claim in it has to lead back to something you read.".into(),
        guidance: vec![
            g("", "Plan first: break the request into the questions the report must answer and write them down with `todo_write`. Mark each one done when you have the evidence for it."),
            g("", "Read before you write. Use your tools to find sources and read them; do not answer from memory, and do not stop at the first source that agrees with you."),
            g("spawn_agent", "Questions that do not depend on each other can be researched side by side: `spawn_agent` one specialist per question, in the same turn. Tell each to record its sources and cite them by number."),
            g("", "Each time a source gives you something the report will rely on, call `record_source` with the reference exactly as it appeared and what it established. It returns the number to cite."),
            g("", "Cite by number: [n] straight after the claim it supports. Never cite a number you were not given, and never state a figure, a date or a quotation without one."),
            g("", "Where sources disagree, say so and cite both. Where you found nothing, say that — do not fill the gap."),
        ],
        closing: format!(
            "Write the report itself: the answer first, then the evidence for it. Do not add a list of sources at the end — it is appended from what you recorded. This report needs at least {need} source{plural}."
        ),
        wrap_up: "This is your last step, and you cannot call tools in it. Write the report now from the sources you have recorded, cite them by number, and say plainly what you could not establish.".into(),
        ..Default::default()
    }
    .checked()
}

/// A task handed over and carried through. `ask` answers the agent's questions.
pub fn cowork(level: Option<&str>, ask: Option<Asker>) -> Result<Mode> {
    let (level, tier) = depth(level)?;
    Mode {
        name: "cowork".into(),
        max_steps: by_depth(&level, 25, 60, 120),
        helpers: by_depth(&level, 0, 3, 8),
        depth: level,
        tier,
        description: "A task carried through in a workspace, with files handed back.".into(),
        workspace: true,
        conversational: true,
        todos: true,
        ask,
        documents: true,
        collect_files: true,
        opening: "You are working alongside a person on a task they have handed over. Carry it through to a finished result — do not hand back a plan and stop.".into(),
        guidance: vec![
            g("", "Lay out the steps with `todo_write` before you take the first one, and keep the list true as you go: mark an item in progress when you start it and done when it is done, and add what you discover along the way."),
            g("fs_write", "Work in the workspace. What you were given is there (`fs_list`, `fs_read`), and what you produce goes there (`fs_write`). A deliverable is a file, not a paragraph describing one."),
            g("parse_document", "`parse_document` reads PDFs, Word files, spreadsheets and the like."),
            g("run_python", "`run_python` is for anything that should be computed rather than estimated."),
            g("shell", "`shell` runs commands in the workspace: install what you need, run what you wrote, and read the output rather than assuming it worked."),
            g("spawn_agent", "Parts that do not depend on each other can run side by side: `spawn_agent` one specialist per part, in the same turn, and check what each hands back."),
            g("ask_user", "`ask_user` is for a decision that is the person's to make and that changes what you do next. Do not ask for what you can look up or reasonably assume."),
            g("!ask_user", "You cannot ask questions in this run. Where something is unclear, make the reasonable assumption and say what you assumed."),
            g("", "If an action is refused or needs an approval you do not get, do not work around it. Say what was blocked."),
            g("", "Check your own work before you call it done: re-read what you wrote, and run what can be run."),
        ],
        closing: "Finish with a short hand-over: what you did, the files that hold it, what you assumed, and anything still open.".into(),
        wrap_up: "This is your last step, and you cannot call tools in it. Hand over now: what is finished, which files hold it, and exactly what is still open.".into(),
        ..Default::default()
    }
    .checked()
}

/// Spoken conversation: answers written for the ear, and kept short.
pub fn voice(level: Option<&str>) -> Result<Mode> {
    let named = level.is_some();
    let (level, tier) = depth(Some(level.unwrap_or("fast")))?;
    Mode {
        name: "voice".into(),
        max_steps: by_depth(&level, 6, 10, 16),
        depth: level,
        tier: if named { tier } else { None },
        description: "Spoken conversation: short, plain, written for the ear.".into(),
        opening: "You are speaking aloud with someone, and everything you write will be read out by a voice. Write for the ear.".into(),
        guidance: vec![
            g("", "Answer in one or two short sentences, then stop. If there is more, offer it rather than say it."),
            g("", "No lists, headings, markdown, emoji or code — none of it can be spoken. Say numbers, dates and amounts the way a person would say them."),
            g("", "Ask one question at a time, and only when the answer changes what you do."),
            g("", "When you need a tool, say a few words first so the silence is explained, then call it."),
            g("", "What you hear was transcribed and may be slightly wrong. If something does not make sense, say what you heard and ask."),
            g("", "If you are interrupted, do not start over. Answer what was just said."),
        ],
        wrap_up: "This is your last step. Say what you have in a sentence, and what you could not do.".into(),
        ..Default::default()
    }
    .checked()
}

/// Builds a mode at a depth.
pub type ModeFactory = Arc<dyn Fn(Option<&str>) -> Result<Mode> + Send + Sync>;

static MODES: LazyLock<RwLock<BTreeMap<String, ModeFactory>>> = LazyLock::new(|| {
    let mut m: BTreeMap<String, ModeFactory> = BTreeMap::new();
    m.insert("chat".into(), Arc::new(chat));
    m.insert("research".into(), Arc::new(research));
    m.insert("cowork".into(), Arc::new(|d| cowork(d, None)));
    m.insert("voice".into(), Arc::new(voice));
    RwLock::new(m)
});

/// Add a mode of your own.
pub fn register_mode(name: &str, factory: ModeFactory) -> Result<()> {
    let key = name.trim().to_lowercase();
    if key.is_empty() {
        return Err(Error::config("a mode needs a name"));
    }
    MODES.write().unwrap_or_else(|p| p.into_inner()).insert(key, factory);
    Ok(())
}

/// A mode by name (`chat`, `research`, `deep_research`, `cowork`, `voice`, or
/// one you registered) at a depth.
pub fn resolve_mode(name: &str, level: Option<&str>) -> Result<Mode> {
    let key = name.trim().to_lowercase().replace(' ', "_");
    let key = match key.as_str() {
        "deep_research" | "deep-research" | "deepresearch" => "research".to_string(),
        "co-work" | "co_work" => "cowork".to_string(),
        _ => key,
    };
    let modes = MODES.read().unwrap_or_else(|p| p.into_inner());
    match modes.get(&key) {
        Some(f) => f(level),
        None => Err(Error::config(format!("unknown mode '{name}'; known: {}", modes.keys().cloned().collect::<Vec<_>>().join(", ")))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_citations() {
        let c = citations("A [1], B [2, 3], C [5-7][8] and rows[0] ```[9]```");
        assert_eq!(c.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 5, 6, 7, 8]);
        assert!(citations("[1-100]").contains(&1));
        assert!(!citations("[1-100]").contains(&50));
    }

    #[test]
    fn notebook_ledger_and_child() {
        let nb = Notebook::new(true, true);
        nb.saw(&["see https://Example.com/a/ for details"]);
        assert!(nb.mentions("https://example.com/a"));
        assert!(!nb.mentions("https://other.org"));
        let s = nb.record("https://example.com/a", "fact one", "", "x");
        assert_eq!(s.id, 1);
        assert_eq!(nb.record("HTTPS://example.com/a/#frag", "fact two", "Ex", "y").id, 1);
        assert_eq!(nb.sources()[0].finding, "fact one\nfact two");
        let kid = nb.child();
        kid.record("doc.pdf", "f", "", "kid");
        assert_eq!(nb.sources().len(), 2, "children share the ledger");
        assert!(kid.todos().is_empty());
        nb.write_todos(vec![Todo { content: "a".into(), status: TodoStatus::Done, note: String::new() }]);
        assert_eq!(nb.todo_summary(), "1 items · 1 done");
        assert!(nb.pins()[1].starts_with("Sources recorded so far: [1] Ex — https://example.com/a (fact one"));
    }

    #[test]
    fn research_rules() {
        let m = research(Some("fast")).unwrap();
        assert_eq!((m.max_steps, m.min_sources, m.helpers, m.tier.as_deref()), (12, 2, 0, Some("fast")));
        let nb = Notebook::new(true, false);
        nb.record("a", "x", "", "");
        let problems = m.unmet("claim [2]", Some(&nb));
        assert_eq!(problems.len(), 2);
        assert!(problems[0].starts_with("you cite [2] but no such source is recorded"));
        assert!(problems[1].starts_with("1 of the 2 sources"));
        nb.record("b", "y", "", "");
        assert!(m.unmet("claim [1]", Some(&nb)).is_empty());
        let closed = m.close("claim [1]", Some(&nb));
        assert_eq!(closed, "claim [1]\n\n## Sources\n- [1] a");
        assert_eq!(m.close("x\n\n## References\n- a", Some(&nb)), "x\n\n## References\n- a");
        assert_eq!(m.artifacts("report", "me")[0].name, "report.md");
    }

    #[test]
    fn prompts_follow_the_tools() {
        let m = cowork(None, None).unwrap();
        let p = m.prompt(&["todo_write".into()]);
        assert!(p.contains("You cannot ask questions in this run."));
        assert!(!p.contains("`run_python` is for"));
        let with = m.prompt(&["ask_user".into()]);
        assert!(with.contains("`ask_user` is for a decision"));
        assert_eq!(m.tier, None);
        assert_eq!(resolve_mode("deep-research", Some("thorough")).unwrap().depth, "deep");
        assert!(resolve_mode("dance", None).unwrap_err().to_string().starts_with("unknown mode 'dance'"));
        assert!(chat(Some("ultra")).is_err());
        assert_eq!(voice(None).unwrap().tier, None);
    }
}
