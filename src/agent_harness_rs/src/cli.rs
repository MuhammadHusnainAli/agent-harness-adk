//! The `agent-harness` command: run an agent, chat with one, inspect what happened.
//!
//! Built with the `cli` feature:
//!
//! ```text
//! cargo install agent-harness-adk --features cli
//! agent-harness run "What is 17% of 2,400?" --tools
//! agent-harness chat --state .harness
//! ```

use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::agent::{Agent, RunOptions};
use crate::errors::{Error, Result};
use crate::harness::Harness;
use crate::runtime::workspace::Workspace;
use crate::types::{RunResult, StopReason, StreamEventType, TodoStatus};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "agent-harness", version = VERSION, about = "Run and inspect agents built with agent-harness.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// What every command that builds an agent takes.
#[derive(Args, Debug, Clone, Default)]
pub struct Common {
    /// what to call the agent
    #[arg(long, default_value = "assistant")]
    pub name: String,
    /// model id (routed by default)
    #[arg(long)]
    pub model: Option<String>,
    /// any name from `agent-harness providers` (inferred from the model)
    #[arg(long)]
    pub provider: Option<String>,
    /// the agent's instructions
    #[arg(long, default_value = "")]
    pub instructions: String,
    /// a directory of skills to load
    #[arg(long)]
    pub skills: Option<String>,
    /// give it the built-in tools (time, maths, web search and fetch)
    #[arg(long)]
    pub tools: bool,
    /// a knowledge base it can search: a vector store's name or URL, e.g. qdrant://localhost:6333/docs
    #[arg(long, value_name = "STORE")]
    pub knowledge: Option<String>,
    /// let it generate images (Gemini, OpenAI or Replicate, by the key that is set)
    #[arg(long)]
    pub images: bool,
    /// give it a web browser to use (Chrome, Chromium or Edge, headless)
    #[arg(long)]
    pub browser: bool,
    /// the same, with the browser window shown
    #[arg(long)]
    pub show_browser: bool,
    /// an OpenAPI file or URL whose operations become tools (repeatable)
    #[arg(long, value_name = "SPEC")]
    pub openapi: Vec<String>,
    /// the bearer token for --openapi (or $OPENAPI_TOKEN)
    #[arg(long)]
    pub openapi_token: Option<String>,
    /// run without memory
    #[arg(long)]
    pub no_memory: bool,
    /// how the agent works: chat keeps the thread, research writes a cited report, cowork carries a task through in a workspace
    #[arg(long)]
    pub mode: Option<String>,
    /// how hard the mode works, and the model tier it asks for
    #[arg(long, value_parser = ["fast", "balanced", "deep"])]
    pub depth: Option<String>,
    /// a directory the agent may read and write files in
    #[arg(long)]
    pub workspace: Option<String>,
    /// work inside a sandbox instead: docker, podman, e2b, daytona, kubernetes, ssh — or a URL such as docker://node:22
    #[arg(long, value_name = "NAME")]
    pub sandbox: Option<String>,
    /// pick up the sandbox an earlier run reported, instead of starting a new one
    #[arg(long, value_name = "ID")]
    pub sandbox_id: Option<String>,
    /// leave the sandbox running at exit, to continue in it later
    #[arg(long)]
    pub keep_sandbox: bool,
    /// the step ceiling (the mode's own, or 20)
    #[arg(long)]
    pub max_steps: Option<u32>,
    /// persist sessions, memory and traces under this directory
    #[arg(long)]
    pub state: Option<String>,
    /// print spans as they close
    #[arg(long)]
    pub trace: bool,
    /// ask before any tool that needs approval
    #[arg(long)]
    pub approve: bool,
    /// resume a session by id
    #[arg(long)]
    pub session: Option<String>,
    /// keep chats in a database: postgresql://…, mongodb://…, redis://…, sqlite:///chats.db
    #[arg(long, value_name = "URL")]
    pub sessions: Option<String>,
    /// who the agent is acting for; their chats are theirs
    #[arg(long)]
    pub user: Option<String>,
    /// the tenant they belong to
    #[arg(long)]
    pub tenant: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// run one task and print the answer
    Run {
        task: String,
        /// stream the answer
        #[arg(long)]
        stream: bool,
        /// a file or URL to send with the task: an image, a PDF, a recording, a document (repeatable)
        #[arg(long, value_name = "FILE")]
        attach: Vec<String>,
        /// also print the full result
        #[arg(long)]
        json: bool,
        /// print the run report
        #[arg(long)]
        report: bool,
        #[command(flatten)]
        common: Common,
    },
    /// an interactive session
    Chat {
        #[command(flatten)]
        common: Common,
    },
    /// talk to an agent from a WAV file
    Voice {
        /// a recording to answer
        #[arg(long, value_name = "WAV")]
        input: Option<String>,
        /// where to write the spoken answer
        #[arg(long, value_name = "WAV")]
        output: Option<String>,
        /// use a speech-to-speech model instead of listen, think, speak
        #[arg(long, value_parser = ["openai", "gemini"])]
        realtime: Option<String>,
        /// the voice to speak in
        #[arg(long, default_value = "alloy")]
        voice: String,
        /// the language being spoken, e.g. en
        #[arg(long)]
        language: Option<String>,
        /// said when the call opens
        #[arg(long)]
        greeting: Option<String>,
        #[command(flatten)]
        common: Common,
    },
    /// run a declared workflow from a YAML or JSON file
    Workflow {
        /// the workflow file
        file: String,
        /// the input, for a workflow that takes one
        text: Option<String>,
        /// one input; the value is read as JSON if it is JSON (repeatable)
        #[arg(long, value_name = "NAME=VALUE")]
        input: Vec<String>,
        /// give it the built-in tools (time, maths, web search and fetch)
        #[arg(long)]
        tools: bool,
        /// validate the file and print its outline; run nothing
        #[arg(long)]
        check: bool,
        /// print the full result
        #[arg(long)]
        json: bool,
        #[arg(long)]
        state: Option<String>,
        #[arg(long)]
        trace: bool,
        #[arg(long)]
        approve: bool,
    },
    /// serve an agent over the agent-to-agent protocol, or call one
    A2a {
        /// serve: serve an agent · card URL: print an agent's card · send URL TEXT: send it a message
        #[arg(value_parser = ["serve", "card", "send"])]
        action: String,
        /// the agent's URL
        target: Option<String>,
        /// what to send
        text: Option<String>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// the URL others reach this agent at, for its card
        #[arg(long)]
        public_url: Option<String>,
        /// serve: a bearer token callers must send (repeatable) · card, send: the token to send
        #[arg(long)]
        token: Vec<String>,
        /// serve: how many tasks run at once
        #[arg(long, default_value_t = 64)]
        max_concurrency: usize,
        /// send: stream the answer
        #[arg(long)]
        stream: bool,
        /// send: the context id of a conversation to carry on
        #[arg(long)]
        context: Option<String>,
        #[command(flatten)]
        common: Common,
    },
    /// serve an agent, or a blueprint's agents, as an MCP server
    #[command(name = "mcp-serve")]
    McpServe {
        /// serve the agents of an agents.yaml; each may name its own key with `mcp_key_env`
        #[arg(long, value_name = "FILE")]
        blueprint: Option<String>,
        /// a key that opens every agent (repeatable; or $MCP_API_KEY)
        #[arg(long)]
        api_key: Vec<String>,
        /// serve the agents' own tools as well
        #[arg(long)]
        expose_tools: bool,
        /// serve over stdin/stdout, for a client that starts the server itself
        #[arg(long)]
        stdio: bool,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[command(flatten)]
        common: Common,
    },
    /// show the tools an OpenAPI document becomes, or call one
    Openapi {
        /// the OpenAPI file or URL
        spec: String,
        /// where the API is, if the document does not say
        #[arg(long)]
        base_url: Option<String>,
        /// a bearer token (or $OPENAPI_TOKEN)
        #[arg(long, alias = "token")]
        openapi_token: Option<String>,
        /// the key for the document's apiKey scheme
        #[arg(long)]
        api_key: Option<String>,
        /// call this operation and print what it returns
        #[arg(long, value_name = "TOOL")]
        call: Option<String>,
        /// an argument for --call (repeatable)
        #[arg(long, value_name = "NAME=VALUE")]
        arg: Vec<String>,
        /// the tools as JSON: names, schemas, requests
        #[arg(long)]
        json: bool,
    },
    /// search the web, or list the search engines and which are set up
    Search {
        /// what to look for; none lists the engines
        query: Vec<String>,
        /// the engine to ask (repeatable: the next is asked when one fails)
        #[arg(long, value_name = "NAME")]
        engine: Vec<String>,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, value_parser = ["day", "week", "month", "year"])]
        recency: Option<String>,
        /// only this site (repeatable)
        #[arg(long, value_name = "DOMAIN")]
        domain: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// a knowledge base: add documents, search, check a vector store
    Knowledge {
        #[arg(value_parser = ["stores", "check", "add", "search", "delete"])]
        action: String,
        /// files or URLs to add, the question to search for, or document ids to delete
        items: Vec<String>,
        /// the vector store: sqlite:///knowledge.db, qdrant://host:6333/docs, …
        #[arg(long, value_name = "URL")]
        store: Option<String>,
        /// keep this knowledge base apart from others in the store
        #[arg(long, default_value = "")]
        namespace: String,
        /// the embedding model
        #[arg(long)]
        embed_model: Option<String>,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// generate an image, or list the image engines and which are set up
    Image {
        /// what to draw; none lists the engines
        prompt: Vec<String>,
        /// gemini, openai or replicate (repeatable: the next is asked when one fails)
        #[arg(long, value_name = "NAME")]
        engine: Vec<String>,
        /// the engine's model
        #[arg(long)]
        model: Option<String>,
        /// an image to work from (repeatable)
        #[arg(long, value_name = "FILE|URL")]
        reference: Vec<String>,
        /// the shape, e.g. "16:9"
        #[arg(long, default_value = "")]
        aspect: String,
        #[arg(long, default_value_t = 1)]
        count: u32,
        /// the file name, without an extension
        #[arg(long, default_value = "")]
        name: String,
        /// where to save (default ./images)
        #[arg(long, default_value = "images")]
        out: String,
    },
    /// list the models the harness knows and their prices
    Models,
    /// list the LLM providers and what each needs to connect
    Providers {
        /// describe one provider in full
        name: Option<String>,
        /// with a name: connect and check the credentials work
        #[arg(long)]
        ping: bool,
        /// only the providers already configured
        #[arg(long)]
        ready: bool,
        /// machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// list the sandboxes an agent can work in, or test one
    Sandboxes {
        /// one sandbox, by name or URL (docker://alpine)
        name: Option<String>,
        /// with a name: start it and prove it works
        #[arg(long)]
        check: bool,
        #[arg(long)]
        json: bool,
    },
    /// runs waiting for a person: list, show, approve, deny, resume
    Approvals {
        #[arg(default_value = "list", value_parser = ["list", "show", "approve", "deny", "resume", "release"])]
        action: String,
        /// the approval's id
        id: Option<String>,
        /// who is approving or denying
        #[arg(long)]
        by: Option<String>,
        /// why — the agent is told
        #[arg(long, default_value = "")]
        note: String,
        /// decide one call of a request that has several
        #[arg(long)]
        call: Option<String>,
        /// list only these: pending, approved, denied, expired, resumed, failed
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        common: Common,
    },
    /// list or show stored sessions
    Sessions {
        #[arg(long)]
        state: Option<String>,
        /// a database instead of --state: postgresql://…, mongodb://…, redis://…
        #[arg(long, value_name = "URL")]
        store: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// only this user's sessions
        #[arg(long)]
        user: Option<String>,
        /// only this tenant's
        #[arg(long)]
        tenant: Option<String>,
        /// only this agent's
        #[arg(long)]
        agent: Option<String>,
        /// print one session's transcript
        #[arg(long)]
        show: Option<String>,
        /// delete one session
        #[arg(long)]
        delete: Option<String>,
        /// prove the store works: save, load, list by owner, refuse a stale save, delete
        #[arg(long)]
        check: bool,
        /// which stores have their driver built in
        #[arg(long)]
        backends: bool,
        #[arg(long)]
        json: bool,
    },
    /// list the tools an MCP server offers
    Mcp {
        /// the server command and its arguments
        command: Vec<String>,
        /// an HTTP MCP endpoint instead
        #[arg(long)]
        url: Option<String>,
        #[arg(long, default_value = "server")]
        server_name: String,
    },
    /// print the run journal
    Journal {
        #[arg(long)]
        state: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// jurisdiction packs, evidence reports, audit verification
    Governance {
        /// packs · pack <id> · check · report · inventory · verify · dsar access|erase
        #[arg(value_parser = ["packs", "pack", "check", "report", "inventory", "verify", "dsar"])]
        action: String,
        /// the pack id (pack), an audit file (verify), or access|erase (dsar)
        target: Option<String>,
        /// a blueprint agents.yaml (inventory)
        #[arg(long)]
        agents: Option<String>,
        /// the data subject's user id (dsar)
        #[arg(long)]
        user: Option<String>,
        /// the data subject's tenant id (dsar)
        #[arg(long)]
        tenant: Option<String>,
        /// who asked for the erasure, for the record (dsar)
        #[arg(long, default_value = "data subject")]
        requested_by: String,
        /// a governance.yaml
        #[arg(long)]
        config: Option<String>,
        /// report on one pack only
        #[arg(long)]
        pack: Option<String>,
        #[arg(long, default_value = "md", value_parser = ["md", "json", "html"])]
        format: String,
        /// the harness directory (verify, dsar)
        #[arg(long)]
        state: Option<String>,
        /// environment variable holding the HMAC audit key (verify)
        #[arg(long)]
        key_env: Option<String>,
        /// machine-readable (packs)
        #[arg(long)]
        json: bool,
    },
}

/// A usage mistake: printed, exit code 2.
fn usage(message: impl Into<String>) -> Error {
    Error::Configuration(format!("usage: {}", message.into()))
}

fn pretty(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default()
}

fn chars(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// `--input amount=40` is the number 40; anything JSON cannot read is text.
fn value(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.into()))
}

fn pair(text: &str, flag: &str) -> Result<(String, String)> {
    text.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())).ok_or_else(|| usage(format!("{flag} takes name=value — got {}", crate::toolkits::openapi::py_repr(text))))
}

/// `1234567` as `1,234,567`.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn read_line(prompt: &str) -> Option<String> {
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

async fn ask_line(prompt: String) -> Option<String> {
    tokio::task::spawn_blocking(move || read_line(&prompt)).await.ok().flatten()
}

fn harness(state: Option<&str>, sessions: Option<&str>, trace: bool, approve: bool) -> Result<Harness> {
    // A run that has to wait for approval is kept where the chats are: under
    // --state, or in the database --sessions names.
    let mut h = match (state, sessions) {
        (Some(dir), None) => Harness::local(dir)?,
        (Some(dir), Some(url)) => Harness::local(dir)?.with_sessions(crate::sessions::session_provider(url)?),
        (None, Some(url)) => Harness::new().with_sessions(crate::sessions::session_provider(url)?).with_session_approvals(),
        (None, None) => Harness::new(),
    };
    if trace {
        let _ = h.tracer.clone().exporter(crate::runtime::tracing::console_exporter());
    }
    if approve {
        h.policy.approver = Some(crate::runtime::permissions::console_approver());
    }
    Ok(h)
}

fn common_harness(c: &Common) -> Result<Harness> {
    harness(c.state.as_deref(), c.sessions.as_deref(), c.trace, c.approve)
}

/// Put an agent's question to whoever is at the terminal.
fn console_asker() -> crate::modes::Asker {
    Arc::new(|question: String, options: Vec<String>| {
        Box::pin(async move {
            let choices: String = options.iter().enumerate().map(|(n, o)| format!("\n  {}. {o}", n + 1)).collect();
            let answer = ask_line(format!("\n? {question}{choices}\n> ")).await.unwrap_or_default().trim().to_string();
            match answer.parse::<usize>() {
                Ok(n) if n > 0 && n <= options.len() => options[n - 1].clone(),
                _ => answer,
            }
        })
    })
}

fn openapi_toolkit(spec: &str, base_url: Option<&str>, token: Option<&str>, api_key: Option<&str>, approve: bool) -> Result<crate::toolkits::openapi::OpenAPIToolkit> {
    use crate::toolkits::openapi::{OpenAPIOptions, OpenAPIToolkit};
    let token = token.map(str::to_string).or_else(|| std::env::var("OPENAPI_TOKEN").ok()).filter(|t| !t.is_empty());
    OpenAPIToolkit::new(
        spec,
        OpenAPIOptions {
            base_url: base_url.map(str::to_string),
            token: token.map(Into::into),
            api_key: api_key.map(str::to_string),
            writes: if approve { crate::tools::Permission::Ask } else { crate::tools::Permission::Allow },
            ..Default::default()
        },
    )
}

/// Real embeddings when a key for them is set; the offline hasher when not.
fn embedder(model: Option<&str>) -> Arc<dyn crate::memory::Embedder> {
    use crate::memory::semantic::{HashEmbedder, ProviderEmbedder};
    let set = |k: &str| std::env::var(k).is_ok_and(|v| !v.is_empty());
    if set("OPENAI_API_KEY") {
        let provider = Arc::new(crate::llm_providers::openai::OpenAIProvider::new(None));
        return Arc::new(ProviderEmbedder::new(provider, Some(model.unwrap_or("text-embedding-3-small"))));
    }
    if set("GEMINI_API_KEY") || set("GOOGLE_API_KEY") {
        let provider = Arc::new(crate::llm_providers::gemini::GeminiProvider::new(None));
        return Arc::new(ProviderEmbedder::new(provider, model));
    }
    eprintln!("(no embedding key is set — using the offline hashing embedder, which matches words, not meaning)");
    Arc::new(HashEmbedder::default())
}

fn knowledge_base(store: &str, namespace: &str, embed_model: Option<&str>) -> Result<crate::knowledge::KnowledgeBase> {
    crate::knowledge::KnowledgeBase::builder().store(store).embedder(embedder(embed_model)).namespace(namespace).build()
}

async fn build_agent(c: &Common, h: &Harness) -> Result<Agent> {
    let mut tools: Vec<crate::tools::Tool> = Vec::new();
    if c.tools {
        tools.extend(crate::toolkits::basics::basic_tools());
        tools.push(crate::toolkits::search::web_search());
        tools.push(crate::toolkits::web::http_fetch());
    }
    for spec in &c.openapi {
        tools.extend(openapi_toolkit(spec, None, c.openapi_token.as_deref(), None, c.approve)?.into_tools());
    }
    if let Some(store) = &c.knowledge {
        tools.push(knowledge_base(store, "", None)?.as_tool("search_knowledge"));
    }
    if c.images {
        use crate::toolkits::images::{ImageGenerator, ImageOptions, ImageToolOptions};
        // Into the workspace when the agent has one; into ./images when not.
        let images = ImageGenerator::new(ImageOptions { output_dir: Some(PathBuf::from("images")), ..Default::default() })?;
        tools.extend(images.tools(ImageToolOptions::default()));
    }
    if c.browser || c.show_browser {
        // Closed when the process ends: a browser never outlives what started it.
        let browser = crate::browser::Browser::new(crate::browser::BrowserOptions { headless: !c.show_browser, ..Default::default() })?;
        tools.extend(browser.tools());
    }
    let mut b = Agent::builder(&c.name).instructions(c.instructions.clone()).harness(h.clone()).tools(tools).memory(!c.no_memory);
    if let Some(m) = &c.model {
        b = b.model(m.clone());
    }
    if let Some(p) = &c.provider {
        b = b.provider_name(p.clone());
    }
    if let Some(mode) = &c.mode {
        if mode == "cowork" && std::io::stdin().is_terminal() {
            b = b.mode(crate::modes::cowork(c.depth.as_deref(), Some(console_asker()))?);
        } else {
            b = b.mode_named(mode, c.depth.as_deref());
        }
    }
    if let Some(dir) = &c.skills {
        b = b.skills(crate::skills::SkillRegistry::from_dir(dir)?);
    }
    if let Some(n) = c.max_steps {
        b = b.max_steps(n);
    }
    if c.user.is_some() || c.tenant.is_some() {
        let mut t = crate::memory::Trace::new();
        t.user_id = c.user.clone();
        t.tenant_id = c.tenant.clone();
        b = b.trace(t);
    }
    if let Some(name) = &c.sandbox {
        let mut options = Map::new();
        options.insert("sandbox".into(), json!(name));
        if let Some(id) = &c.sandbox_id {
            options.insert("id".into(), json!(id));
        }
        if c.keep_sandbox {
            options.insert("keep".into(), json!(true));
        }
        if let Some(dir) = &c.workspace {
            // The folder is mounted into the container, so the work lands in it.
            let kind = name.split("://").next().unwrap_or("").to_lowercase();
            if kind != "docker" && kind != "podman" {
                return Err(usage("--workspace with --sandbox only applies to docker and podman, which can mount a folder; other sandboxes keep their own files"));
            }
            options.insert("mount".into(), json!(dir));
        }
        let ws: Arc<dyn Workspace> = Arc::new(crate::sandboxes::sandbox(&Value::Object(options))?);
        b = b.workspace(ws);
    } else if let Some(dir) = &c.workspace {
        let ws: Arc<dyn Workspace> = Arc::new(crate::runtime::workspace::LocalWorkspace::new(dir, Default::default())?);
        b = b.workspace(ws);
    } else if c.mode.as_deref() == Some("cowork") && c.state.is_none() {
        eprintln!("note: no --workspace, so this works in a temporary folder that is removed on exit");
    }
    b.build()
}

fn run_options(c: &Common, attach: &[String]) -> RunOptions {
    RunOptions {
        session: c.session.clone().map(Into::into),
        attachments: attach.iter().map(|a| crate::types::attach(a, None)).collect(),
        sandbox_id: c.sandbox_id.clone(),
        ..Default::default()
    }
}

/// What to do about a run that stopped to ask.
fn waiting(approval: &Value) -> String {
    let id = approval["id"].as_str().unwrap_or("");
    let calls: Vec<String> = match serde_json::from_value::<crate::runtime::approvals::Approval>(approval.clone()) {
        Ok(a) => a.calls.iter().map(|c| format!("  {}{}", c.describe(), if c.reason.is_empty() { String::new() } else { format!("  — {}", c.reason) })).collect(),
        Err(_) => approval["calls"].as_array().into_iter().flatten().map(|c| format!("  {}", c["tool"].as_str().unwrap_or("?"))).collect(),
    };
    format!(
        "waiting for approval {id}:\n{}\n  agent-harness approvals approve {id} --by NAME\n  agent-harness approvals resume {id}   (with the same options as this run)",
        calls.join("\n")
    )
}

/// What a mode left behind, in a line or two for stderr.
fn footer(result: &RunResult) -> String {
    let mut lines = Vec::new();
    if !result.todos.is_empty() {
        let done = result.todos.iter().filter(|t| t.status == TodoStatus::Done).count();
        lines.push(format!("todos: {done} of {} done", result.todos.len()));
    }
    if !result.sources.is_empty() {
        lines.push(format!("sources: {}", result.sources.len()));
    }
    let files: Vec<String> = result.artifacts.iter().map(|a| a.path.clone().filter(|p| !p.is_empty()).unwrap_or_else(|| a.name.clone())).collect();
    if !files.is_empty() {
        let more = if files.len() > 12 { format!(" (+{} more)", files.len() - 12) } else { String::new() };
        lines.push(format!("files: {}{more}", files.iter().take(12).cloned().collect::<Vec<_>>().join(", ")));
    }
    lines.extend(result.violations.iter().map(|v| format!("unmet: {v}")));
    lines.join("\n")
}

fn tail(result: &RunResult) {
    let f = footer(result);
    if !f.is_empty() {
        eprintln!("\n{f}");
    }
    if result.stop_reason == StopReason::Approval
        && let Some(a) = &result.approval
    {
        eprintln!("\n{}", waiting(a));
    }
    let sandbox = if result.sandbox_id.is_empty() { String::new() } else { format!(" · sandbox {}", result.sandbox_id) };
    eprintln!("\n[{} steps · ${:.4} · session {}{sandbox}]", result.steps, result.usage.cost_usd, result.session_id);
}

async fn cmd_run(task: &str, stream: bool, attach: &[String], json_out: bool, report: bool, c: &Common) -> Result<i32> {
    let h = common_harness(c)?;
    let agent = build_agent(c, &h).await?;
    let outcome = async {
        let result = if stream {
            let mut events = agent.stream(task, run_options(c, attach)).await?;
            let mut result = None;
            while let Some(event) = events.next().await {
                match event.kind {
                    StreamEventType::Text => {
                        print!("{}", event.text);
                        let _ = std::io::stdout().flush();
                    }
                    StreamEventType::ToolResult => eprintln!("\n  · {} → {}", event.data.get("tool").and_then(Value::as_str).unwrap_or(""), chars(&event.text, 80)),
                    StreamEventType::Progress => eprintln!("\n  · {} · {} sources", event.text, event.data.get("sources").and_then(Value::as_u64).unwrap_or(0)),
                    StreamEventType::RunEnd => result = event.data.get("result").and_then(|r| serde_json::from_value::<RunResult>(r.clone()).ok()),
                    _ => {}
                }
            }
            println!();
            result
        } else {
            let r = agent.run_with(task, run_options(c, attach)).await?;
            println!("{}", r.output);
            Some(r)
        };
        let Some(result) = result else {
            eprintln!("the run produced no result");
            return Ok(1);
        };
        if let Some(e) = &result.error {
            eprintln!("\nerror: {e}");
        }
        if json_out {
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        }
        if report {
            eprintln!("{}", pretty(&h.report().await));
        }
        tail(&result);
        Ok(if result.error.is_some() { 1 } else { 0 })
    }
    .await;
    h.close().await;
    outcome
}

async fn cmd_chat(c: &Common) -> Result<i32> {
    let h = common_harness(c)?;
    let agent = build_agent(c, &h).await?;
    let how = agent.mode().map(|m| format!(" · {} ({})", m.name, agent.depth())).unwrap_or_default();
    println!("agent-harness {VERSION} · {} on {}{how}", agent.name(), agent.model());
    println!("Type /exit to leave, /new for a fresh session, /cost for the bill.\n");
    let mut session = c.session.clone();
    let outcome = async {
        loop {
            let Some(line) = ask_line("you › ".into()).await else {
                println!();
                break;
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match line {
                "/exit" | "/quit" => break,
                "/new" => {
                    session = None;
                    agent.new_session();
                    println!("(new session)");
                    continue;
                }
                "/cost" => {
                    println!("{}", pretty(&h.report().await["budget"]));
                    continue;
                }
                _ => {}
            }
            let options = RunOptions { session: session.clone().map(Into::into), ..Default::default() };
            let result = agent.run_with(line, options).await?;
            session = Some(result.session_id.clone());
            for hop in &result.handoffs {
                eprintln!("  · {}", hop.line());
            }
            let who = if result.agent.is_empty() { agent.name().to_string() } else { result.agent.clone() };
            let said = if result.output.is_empty() { result.error.clone().unwrap_or_default() } else { result.output.clone() };
            println!("\n{who} › {said}\n");
            let f = footer(&result);
            if !f.is_empty() {
                eprintln!("{f}\n");
            }
        }
        if agent.close_session(None).await.is_ok_and(|s| !s.is_empty()) {
            eprintln!("(memory updated)");
        }
        Ok(0)
    }
    .await;
    h.close().await;
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn cmd_voice(input: Option<&str>, output: Option<&str>, realtime: Option<&str>, voice: &str, language: Option<&str>, greeting: Option<&str>, c: &Common) -> Result<i32> {
    use crate::voice::{OpenAISpeech, RealtimeAgent, VoiceAgent, VoiceEventType, duration_ms, resample, unwav, wav};
    let Some(input) = input else {
        eprintln!("live voice needs a microphone, which this build does not drive — pass --input question.wav to talk from a file");
        return Ok(2);
    };
    let mut c = c.clone();
    c.mode = c.mode.or(Some("voice".into()));
    let h = common_harness(&c)?;
    let agent = build_agent(&c, &h).await?;
    let rate = 16_000;
    let (pcm, file_rate) = unwav(&std::fs::read(input)?)?;
    let pcm = if file_rate == rate { pcm } else { resample(&pcm, file_rate, rate)? };
    // A little quiet at the end, so the last word is heard to end.
    let mut audio = pcm;
    audio.extend(std::iter::repeat_n(0u8, rate as usize * 2));
    let frames: Vec<Vec<u8>> = audio.chunks(640).map(<[u8]>::to_vec).collect();
    let source = futures::stream::iter(frames).boxed();
    let (mut events, out_rate, session): (futures::stream::BoxStream<'static, crate::voice::VoiceEvent>, u32, Box<dyn Fn() -> String>) = match realtime {
        Some(provider) => {
            let mut talker = RealtimeAgent::new(agent, provider)?.rate(rate);
            if let Some(s) = &c.session {
                talker = talker.session(s);
            }
            let t = talker.clone();
            (talker.run(source), talker.speaking_rate(), Box::new(move || t.session_id()))
        }
        None => {
            let mut speech = OpenAISpeech::new(None).voice(voice);
            if let Some(l) = language {
                speech = speech.language(l);
            }
            let mut builder = VoiceAgent::builder(agent).speech(Arc::new(speech)).rate(rate);
            if let Some(s) = &c.session {
                builder = builder.session(s);
            }
            if let Some(g) = greeting {
                builder = builder.greeting(g);
            }
            let talker = builder.build()?;
            let t = talker.clone();
            (talker.run(source), talker.speaking_rate(), Box::new(move || t.session_id()))
        }
    };
    let mut spoken = Vec::new();
    while let Some(event) = events.next().await {
        match event.kind {
            VoiceEventType::Transcript if !event.data.get("partial").and_then(Value::as_bool).unwrap_or(false) => println!("you   › {}", event.text),
            VoiceEventType::Audio => spoken.extend_from_slice(&event.audio),
            VoiceEventType::TurnEnd if !event.text.is_empty() => {
                let took = event.data.get("first_audio_ms").and_then(Value::as_f64).map(|ms| format!("  [first sound after {ms:.0} ms]")).unwrap_or_default();
                println!("agent › {}{took}", event.text);
            }
            VoiceEventType::Error => eprintln!("error: {}", event.text),
            _ => {}
        }
    }
    if let Some(path) = output.filter(|_| !spoken.is_empty()) {
        std::fs::write(path, wav(&spoken, out_rate))?;
        eprintln!("\nwrote {path} ({:.1}s)", duration_ms(&spoken, out_rate) / 1000.0);
    }
    eprintln!("[session {}]", session());
    h.close().await;
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
async fn cmd_workflow(file: &str, text: Option<&str>, inputs: &[String], tools: bool, check: bool, json_out: bool, state: Option<&str>, trace: bool, approve: bool) -> Result<i32> {
    let mut extra = Vec::new();
    if tools {
        extra.extend(crate::toolkits::basics::basic_tools());
        extra.push(crate::toolkits::search::web_search());
        extra.push(crate::toolkits::web::http_fetch());
    }
    let h = harness(state, None, trace, approve)?;
    let outcome = async {
        let workflow = crate::workflow::Workflow::from_file(file)?.tools(extra).harness(h.clone()).build()?;
        if check {
            println!("{}", workflow.describe());
            return Ok(0);
        }
        let declared = workflow.spec().input_specs();
        let mut given = Map::new();
        for p in inputs {
            let (key, raw) = pair(p, "--input")?;
            // An input declared as text stays text: an order number is not a number.
            let is_text = declared.get(&key).and_then(|s| s.get("type")).and_then(Value::as_str) == Some("string");
            given.insert(key, if is_text { Value::String(raw) } else { value(&raw) });
        }
        let show: Arc<dyn Fn(crate::workflow::WorkflowEvent) + Send + Sync> = Arc::new(|e| match e.event.as_str() {
            "step_start" => eprintln!("  ▸ {} ({})", e.step, e.kind),
            "step_skipped" => eprintln!("  - {} skipped", e.step),
            "step_failed" | "step_retry" => eprintln!("  ✗ {}: {}", e.step, e.text),
            _ => {}
        });
        let input = if given.is_empty() { text.map_or(Value::Null, |t| Value::String(t.into())) } else { Value::Object(given) };
        let result = workflow.run_with(input, Map::new(), Some(show), None).await?;
        if json_out {
            println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
        } else {
            println!("{}", result.output.as_str().map_or_else(|| pretty(&result.output), str::to_string));
        }
        if let Some(e) = &result.error {
            eprintln!("\nerror: {e}");
        }
        let status = serde_json::to_value(result.status).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        eprintln!("\n[{status} · {} steps · ${:.4} · {:.1}s]", result.steps_run, result.cost_usd, result.duration_ms / 1000.0);
        Ok(if result.ok() { 0 } else { 1 })
    }
    .await;
    h.close().await;
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn cmd_a2a(action: &str, target: Option<&str>, text: Option<&str>, host: &str, port: u16, public_url: Option<&str>, tokens: &[String], max_concurrency: usize, stream: bool, context: Option<&str>, c: &Common) -> Result<i32> {
    use crate::a2a::{A2AClient, A2AServer, SendOptions};
    if action == "serve" {
        let h = common_harness(c)?;
        let agent = build_agent(c, &h).await?;
        let name = agent.name().to_string();
        let mut builder = A2AServer::builder(agent).max_concurrency(max_concurrency);
        if let Some(u) = public_url {
            builder = builder.url(u);
        }
        for t in tokens {
            builder = builder.token(t);
        }
        let server = builder.build()?;
        let url = format!("http://{host}:{port}");
        eprintln!(
            "{name} is served over A2A at {url}\n  card   {url}/.well-known/agent-card.json\n  auth   {}\nThis is the development server; for production mount its router in your own service.",
            if tokens.is_empty() { "none — anyone may call" } else { "bearer token" }
        );
        let served = server.serve(&format!("{host}:{port}")).await;
        h.close().await;
        served?;
        return Ok(0);
    }
    let Some(target) = target else { return Err(usage(format!("a2a {action} needs the agent's URL"))) };
    let mut client = A2AClient::new(target);
    if let Some(t) = tokens.first() {
        client = client.token(t);
    }
    if action == "card" {
        println!("{}", pretty(&client.card(false).await?));
        return Ok(0);
    }
    let Some(text) = text else { return Err(usage("a2a send needs something to send")) };
    let opts = SendOptions { context_id: context.map(str::to_string), ..Default::default() };
    if stream {
        let mut state = String::new();
        let mut events = client.stream(text, &opts).await?;
        while let Some(event) = events.next().await {
            let event = event?;
            if event.kind == "artifact-update" {
                print!("{}", event.text);
                let _ = std::io::stdout().flush();
            }
            if !event.state.is_empty() {
                state = event.state;
            }
        }
        println!();
        return Ok(if state == "completed" { 0 } else { 1 });
    }
    let task = client.send(text, &opts).await?;
    println!("{}", if task.text.is_empty() { task.error().unwrap_or_default() } else { task.text.clone() });
    eprintln!("\n[{} · task {} · context {}]", task.state, task.id, task.context_id);
    Ok(if task.ok() { 0 } else { 1 })
}

#[allow(clippy::too_many_arguments)]
async fn cmd_mcp_serve(blueprint: Option<&str>, api_keys: &[String], expose: bool, stdio: bool, host: &str, port: u16, c: &Common) -> Result<i32> {
    use crate::mcp::server::{ExposeTools, McpAgentServer};
    let h = common_harness(c)?;
    let mut keys: Vec<String> = api_keys.to_vec();
    if let Ok(k) = std::env::var("MCP_API_KEY")
        && !k.is_empty()
    {
        keys.push(k);
    }
    let outcome = async {
        let server = match blueprint {
            Some(file) => {
                if !keys.is_empty() {
                    return Err(usage("with --blueprint, each agent names its own key with `mcp_key_env` in the blueprint"));
                }
                crate::blueprint::Blueprint::from_file(file)?.mcp_server(None, &[], Some(h.clone()))?
            }
            None => {
                let mut b = McpAgentServer::builder([build_agent(c, &h).await?]);
                for k in &keys {
                    b = b.api_key(k.clone());
                }
                if expose {
                    b = b.expose_tools(ExposeTools::All);
                }
                b.build()?
            }
        };
        if stdio {
            server.serve_stdio(tokio::io::stdin(), tokio::io::stdout(), None).await?;
            return Ok(0);
        }
        let url = format!("http://{host}:{port}");
        eprintln!("serving over MCP at {url}/mcp\n  keys           {}", if keys.is_empty() { "none — anyone may call".to_string() } else { keys.len().to_string() });
        server.serve(&format!("{host}:{port}")).await?;
        Ok(0)
    }
    .await;
    h.close().await;
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn cmd_openapi(spec: &str, base_url: Option<&str>, token: Option<&str>, api_key: Option<&str>, call: Option<&str>, args: &[String], json_out: bool) -> Result<i32> {
    let toolkit = openapi_toolkit(spec, base_url, token, api_key, false)?;
    let Some(call) = call else {
        if json_out {
            let rows: Vec<Value> = toolkit
                .tools()
                .iter()
                .map(|t| {
                    let mut row = json!({"name": t.name, "description": t.description, "parameters": t.parameters});
                    if let Some(Value::Object(op)) = toolkit.operation(&t.name) {
                        row.as_object_mut().expect("an object").extend(op);
                    }
                    row
                })
                .collect();
            println!("{}", pretty(&Value::Array(rows)));
        } else {
            println!("{}", toolkit.describe());
        }
        return Ok(0);
    };
    let properties = toolkit.get(call)?.parameters["properties"].clone();
    let mut arguments = Map::new();
    for p in args {
        let (key, raw) = pair(p, "--arg")?;
        let kind = properties[&key]["type"].as_str().unwrap_or("");
        arguments.insert(key, if kind == "string" { Value::String(raw) } else { value(&raw) });
    }
    let answer = toolkit.call(call, Value::Object(arguments)).await?;
    println!("{}", answer.as_str().map_or_else(|| pretty(&answer), str::to_string));
    Ok(0)
}

async fn cmd_search(query: &[String], engines: &[String], limit: usize, recency: Option<&str>, domains: &[String], json_out: bool) -> Result<i32> {
    use crate::toolkits::search::{WebSearch, search_engines};
    if query.is_empty() {
        println!("{:<12} {:<10} needs", "engine", "status");
        for e in search_engines() {
            let needs: Vec<&str> = e["needs"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
            let ready = if e["ready"].as_bool().unwrap_or(false) { "ready" } else { "not set" };
            println!("{:<12} {ready:<10} {}", e["name"].as_str().unwrap_or(""), if needs.is_empty() { "nothing".into() } else { needs.join(" + ") });
        }
        println!("\na search uses: {}", WebSearch::builder().build()?.engines()?.join(", "));
        return Ok(0);
    }
    let mut builder = WebSearch::builder().cache_ttl(0.0).max_limit(limit.max(10));
    for e in engines {
        builder = builder.engine(e);
    }
    let search = builder.build()?;
    let results = search.search(&query.join(" "), Some(limit), recency, (!domains.is_empty()).then(|| domains.to_vec())).await?;
    if json_out {
        println!("{}", serde_json::to_string_pretty(&results).unwrap_or_default());
        return Ok(0);
    }
    if results.is_empty() {
        println!("no results");
    }
    for (n, r) in results.iter().enumerate() {
        let date = if r.published.is_empty() { String::new() } else { format!("  ({})", r.published) };
        println!("{}. {}{date}\n   {}", n + 1, r.title, r.url);
        if !r.snippet.is_empty() {
            println!("   {}", r.snippet);
        }
    }
    eprintln!("\nanswered by {}", search.last_engine());
    Ok(0)
}

fn steps_table(report: &Value, width: usize, precision: usize) {
    for row in report["steps"].as_array().into_iter().flatten() {
        let mark = if row["ok"].as_bool().unwrap_or(false) { "ok  " } else { "FAIL" };
        let detail = row["detail"].as_str().filter(|d| !d.is_empty()).map(|d| format!("  {d}")).unwrap_or_default();
        println!("{mark} {:<width$} {:>w2$.precision$}s{detail}", row["step"].as_str().unwrap_or(""), row["seconds"].as_f64().unwrap_or(0.0), w2 = precision + 4);
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_knowledge(action: &str, items: &[String], store: Option<&str>, namespace: &str, embed_model: Option<&str>, limit: usize, json_out: bool) -> Result<i32> {
    use crate::knowledge::NewDocument;
    if action == "stores" {
        println!("{:<22} needs", "store");
        for row in crate::knowledge::vector_stores() {
            let needs = row["needs"].as_str().filter(|n| !n.is_empty()).unwrap_or("nothing");
            println!("{:<22} {needs}", row["name"].as_str().unwrap_or(""));
        }
        return Ok(0);
    }
    let Some(store) = store else { return Err(usage(format!("knowledge {action} needs --store: a name or a URL, e.g. sqlite:///knowledge.db or qdrant://localhost:6333/docs"))) };
    if action == "check" {
        let s = crate::knowledge::vector_store(store)?;
        let report = s.check(8).await;
        s.close().await;
        steps_table(&report, 18, 3);
        let ok = report["ok"].as_bool().unwrap_or(false);
        println!("\n{} {}", report["store"].as_str().unwrap_or(store), if ok { "works" } else { "does not work" });
        return Ok(if ok { 0 } else { 1 });
    }
    let kb = knowledge_base(store, namespace, embed_model)?;
    let outcome = async {
        match action {
            "add" => {
                if items.is_empty() {
                    return Err(usage("knowledge add takes files or URLs to add"));
                }
                for item in items {
                    let doc = if item.starts_with("http://") || item.starts_with("https://") { NewDocument::url(item) } else { NewDocument::path(item) };
                    let added = kb.add(doc).await?;
                    println!("{}: {}", added.id, if added.unchanged { "unchanged".into() } else { format!("{} passages", added.chunks) });
                }
            }
            "delete" => {
                for item in items {
                    kb.delete(item).await?;
                    println!("{item}: removed");
                }
            }
            _ => {
                let found = kb.search(&items.join(" "), limit, None).await?;
                if json_out {
                    println!("{}", serde_json::to_string_pretty(&found).unwrap_or_default());
                    return Ok(0);
                }
                for (n, p) in found.iter().enumerate() {
                    let where_ = [&p.title, &p.source, &p.document].into_iter().find(|s| !s.is_empty()).cloned().unwrap_or_default();
                    println!("{}. [{:.3}] {where_}\n   {}", n + 1, p.score, chars(&p.text.split_whitespace().collect::<Vec<_>>().join(" "), 300));
                }
                if found.is_empty() {
                    println!("nothing matches");
                }
            }
        }
        Ok(0)
    }
    .await;
    kb.close().await;
    outcome
}

#[allow(clippy::too_many_arguments)]
async fn cmd_image(prompt: &[String], engines: &[String], model: Option<&str>, references: &[String], aspect: &str, count: u32, name: &str, out: &str) -> Result<i32> {
    use crate::toolkits::images::{GenerateOptions, ImageGenerator, ImageOptions, ImageProgress, image_engines};
    if prompt.is_empty() {
        println!("{:<11} {:<9} {:<36} needs", "engine", "status", "model");
        for e in image_engines() {
            println!("{:<11} {:<9} {:<36} {}", e.name, if e.ready { "ready" } else { "not set" }, e.model, e.needs);
        }
        return Ok(0);
    }
    let progress = Arc::new(|u: &ImageProgress| {
        let how_far = u.fraction.map(|f| format!(" {:.0}%", f * 100.0)).unwrap_or_default();
        eprintln!("{}", format!("[{:5.1}s] {}{how_far} {}", u.elapsed, u.status, u.message).trim_end());
    });
    let images = ImageGenerator::new(ImageOptions {
        engines: engines.iter().map(|e| e.as_str().into()).collect(),
        model: model.map(str::to_string),
        output_dir: Some(PathBuf::from(out)),
        on_progress: Some(progress),
        ..Default::default()
    })?;
    let made = images
        .generate(
            &prompt.join(" "),
            GenerateOptions {
                references: references.iter().map(|r| r.as_str().into()).collect(),
                aspect_ratio: (!aspect.is_empty()).then(|| aspect.to_string()),
                count,
                name: name.into(),
                ..Default::default()
            },
        )
        .await?;
    for image in &made {
        println!("{}", image.describe());
    }
    Ok(0)
}

fn cmd_models() -> i32 {
    let models = crate::llm_providers::base::MODELS.read().unwrap_or_else(|p| p.into_inner());
    let mut rows: Vec<_> = models.values().cloned().collect();
    rows.sort_by(|a, b| (&a.provider, &a.id).cmp(&(&b.provider, &b.id)));
    println!("{:<26} {:<10} {:<9} {:>8} {:>8} {:>10}", "model", "provider", "tier", "in $/M", "out $/M", "context");
    for m in rows {
        let tier = serde_json::to_value(m.tier).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        println!("{:<26} {:<10} {:<9} {:>8.2} {:>8.2} {:>10}", m.id, m.provider, tier, m.input_cost, m.output_cost, thousands(m.context_window));
    }
    0
}

async fn cmd_providers(name: Option<&str>, ping: bool, ready: bool, json_out: bool) -> Result<i32> {
    use crate::llm_providers::catalog::{ListOptions, describe_llm_provider, list_llm_providers_with, ping_llm_provider};
    if let Some(name) = name {
        let spec = describe_llm_provider(name)?;
        let result = if ping { Some(ping_llm_provider(name, None, &json!({})).await) } else { None };
        if json_out {
            let mut v = serde_json::to_value(&spec).unwrap_or_default();
            if let Some(r) = &result {
                v["ping"] = r.clone();
            }
            println!("{}", pretty(&v));
        } else {
            println!("{}", spec.render());
            if let Some(r) = &result {
                let state = if r["ok"].as_bool().unwrap_or(false) { "ok".to_string() } else { format!("failed — {}", r["error"].as_str().unwrap_or("")) };
                println!("  ping     {state} ({} ms)", r["latency_ms"].as_f64().unwrap_or(0.0));
            }
        }
        return Ok(if result.is_none_or(|r| r["ok"].as_bool().unwrap_or(false)) { 0 } else { 1 });
    }
    let specs = list_llm_providers_with(&ListOptions { configured_only: ready, ..Default::default() });
    if json_out {
        println!("{}", serde_json::to_string_pretty(&specs).unwrap_or_default());
        return Ok(0);
    }
    println!("{:<18} {:<12} needs", "provider", "status");
    for spec in &specs {
        let mut needs: Vec<String> = Vec::new();
        let mut groups: indexmap::IndexMap<&str, Vec<&str>> = indexmap::IndexMap::new();
        for field in spec.required_fields() {
            match field.one_of {
                Some(group) => groups.entry(group).or_default().push(field.name),
                None => needs.push(field.name.into()),
            }
        }
        needs.extend(groups.values().map(|names| names.join(" | ")));
        let status = if spec.configured { "ready" } else { "needs setup" };
        println!("{:<18} {status:<12} {}", spec.name, if needs.is_empty() { "—".to_string() } else { needs.join(", ") });
    }
    println!("\nagent-harness providers <name> for every field; --ping to test the credentials.");
    Ok(0)
}

async fn cmd_sandboxes(name: Option<&str>, check: bool, json_out: bool) -> Result<i32> {
    if let (Some(name), true) = (name, check) {
        let ws = crate::sandboxes::sandbox(&json!(name))?;
        let report = ws.check().await;
        ws.close().await;
        if json_out {
            println!("{}", pretty(&report));
        } else {
            steps_table(&report, 16, 2);
            let ok = report["ok"].as_bool().unwrap_or(false);
            println!("\n{} {} ({}s)", report["sandbox"].as_str().unwrap_or(name), if ok { "works" } else { "does not work" }, report["seconds"]);
        }
        return Ok(if report["ok"].as_bool().unwrap_or(false) { 0 } else { 1 });
    }
    let mut rows = crate::sandboxes::describe_sandboxes();
    if let Some(name) = name {
        let wanted = name.split("://").next().unwrap_or("").to_lowercase();
        let picked: Vec<Value> = rows.iter().filter(|r| r["name"] == wanted.as_str()).cloned().collect();
        if !picked.is_empty() {
            rows = picked;
        }
    }
    if json_out {
        println!("{}", pretty(&Value::Array(rows)));
        return Ok(0);
    }
    println!("{:<12} {:<12} what it is", "sandbox", "status");
    for row in &rows {
        let status = if row["ready"].as_bool().unwrap_or(false) { "ready" } else { "needs setup" };
        println!("{:<12} {status:<12} {}", row["name"].as_str().unwrap_or(""), row["summary"].as_str().unwrap_or(""));
        for gap in row["missing"].as_array().into_iter().flatten() {
            println!("{:<25} needs: {}", "", gap.as_str().unwrap_or(""));
        }
    }
    println!("\nagent-harness sandboxes <name> --check starts one and proves it works.");
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
async fn cmd_sessions(state: Option<&str>, store: Option<&str>, limit: usize, user: Option<&str>, tenant: Option<&str>, agent: Option<&str>, show: Option<&str>, delete: Option<&str>, check: bool, backends: bool, json_out: bool) -> Result<i32> {
    if backends {
        let all: BTreeMap<String, bool> = crate::sessions::session_backends();
        for (name, ready) in all {
            println!("{name:<10} {}", if ready { "ready" } else { "needs its feature" });
        }
        return Ok(0);
    }
    let store = match store {
        Some(url) => crate::sessions::session_provider(url)?,
        None => Harness::local(state.unwrap_or(".harness"))?.sessions.clone(),
    };
    if check {
        let report = store.check().await;
        steps_table(&report, 20, 3);
        let ok = report["ok"].as_bool().unwrap_or(false);
        println!("\n{} {}", report["store"].as_str().unwrap_or(""), if ok { "works" } else { "does not work" });
        return Ok(if ok { 0 } else { 1 });
    }
    if let Some(id) = show {
        for m in store.load(id).await?.messages {
            println!("{}: {}", m.role.as_str(), chars(&m.text(), 2000));
        }
        return Ok(0);
    }
    if let Some(id) = delete {
        store.delete(id).await?;
        println!("deleted {id}");
        return Ok(0);
    }
    let filter = crate::runtime::session::ListFilter { limit, user_id: user.map(str::to_string), tenant_id: tenant.map(str::to_string), agent: agent.map(str::to_string) };
    let rows = store.list(&filter).await?;
    if json_out {
        println!("{}", pretty(&Value::Array(rows.iter().map(|s| s.summary()).collect())));
        return Ok(0);
    }
    if rows.is_empty() {
        println!("no sessions yet");
        return Ok(0);
    }
    for s in rows {
        let owner: Vec<&str> = [s.tenant_id.as_deref(), s.user_id.as_deref()].into_iter().flatten().filter(|p| !p.is_empty()).collect();
        let owner = if owner.is_empty() { "-".to_string() } else { owner.join("/") };
        println!("{}  {:<14} {:>3} messages  ${:.4}  {owner:<16} {}", s.id, s.agent, s.messages.len(), s.usage.cost_usd, chars(&s.title, 48));
    }
    Ok(0)
}

async fn cmd_mcp(command: &[String], url: Option<&str>, server_name: &str) -> Result<i32> {
    use crate::mcp::client::{McpManager, McpServer};
    let server = match url {
        Some(u) => McpServer::http(server_name, u),
        None => {
            let Some((first, rest)) = command.split_first() else { return Err(usage("mcp takes the server command, or --url")) };
            McpServer::stdio(server_name, first, rest.to_vec())
        }
    };
    let manager = McpManager::new([server]).connect().await;
    if !manager.errors.is_empty() {
        eprintln!("{}", manager.errors.join("\n"));
        manager.close().await;
        return Ok(1);
    }
    for t in manager.tools() {
        println!("{}\n    {}", t.name, t.description);
    }
    manager.close().await;
    Ok(0)
}

fn cmd_journal(state: Option<&str>, limit: usize) -> i32 {
    let path = Path::new(state.unwrap_or(".harness")).join("journal.jsonl");
    if !path.exists() {
        eprintln!("no journal at {}", path.display());
        return 1;
    }
    println!("{}", crate::runtime::journal::RunJournal::load(path).render(Some(limit)));
    0
}

#[allow(clippy::too_many_arguments)]
async fn cmd_governance(
    action: &str,
    target: Option<&str>,
    agents: Option<&str>,
    user: Option<&str>,
    tenant: Option<&str>,
    requested_by: &str,
    config: Option<&str>,
    pack: Option<&str>,
    format: &str,
    state: Option<&str>,
    key_env: Option<&str>,
    json_out: bool,
) -> Result<i32> {
    use crate::governance::{Governance, GovernanceOptions, list_packs, load_pack};
    let load = |action: &str| -> Result<Governance> {
        let Some(path) = config else { return Err(usage(format!("agent-harness governance {action} --config governance.yaml"))) };
        Governance::from_file(path, GovernanceOptions::default())
    };
    match action {
        "packs" => {
            let rows: Vec<_> = list_packs().iter().map(|p| load_pack(p)).collect::<Result<_>>()?;
            if json_out {
                let v: Vec<Value> = rows.iter().map(|p| json!({"id": p.id, "name": p.name, "jurisdiction": p.jurisdiction, "kind": p.kind, "binding": p.binding, "as_of": p.as_of})).collect();
                println!("{}", pretty(&Value::Array(v)));
                return Ok(0);
            }
            for p in rows {
                let kind = if p.binding { p.kind.clone() } else { format!("{}, voluntary", p.kind) };
                println!("{:<19} {:<8} {kind:<22} {}", p.id, p.jurisdiction, p.name);
            }
            Ok(0)
        }
        "pack" => {
            let Some(id) = target else { return Err(usage("agent-harness governance pack <id>")) };
            let p = load_pack(id)?;
            println!("{}\n  {} · {} · {} · checked {}", p.name, p.jurisdiction, p.kind, if p.binding { "binding" } else { "voluntary" }, p.as_of);
            println!("  status   {}", p.status.split_whitespace().collect::<Vec<_>>().join(" "));
            println!("\nrequirements");
            for r in &p.requirements {
                let where_ = if r.manual { "manual".to_string() } else { r.controls.join(", ") };
                println!("  {:<28} {}  [{where_}]", r.r#ref, r.title);
            }
            if !p.policy.rules.is_empty() {
                println!("\nrules");
                for rule in &p.policy.rules {
                    println!("  {}: {} on {} — {}", rule.id, rule.effect, rule.on.join(", "), rule.reason);
                }
            }
            if let Some(res) = p.policy.residency.as_ref().filter(|r| !r.transfers.is_empty()) {
                println!("\nresidency");
                for (origin, dest) in &res.transfers {
                    println!("  {origin} → {}", dest.join(", "));
                }
            }
            if !p.incidents.is_empty() {
                println!("\nincident clocks");
                for (kind, clocks) in &p.incidents {
                    for c in clocks {
                        let s = |k: &str| c.get(k).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_string)).unwrap_or_default();
                        println!("  {kind}: {} within {} h ({})", s("notify"), s("within"), s("basis"));
                    }
                }
            }
            println!("\nsources\n{}", p.sources.iter().map(|u| format!("  {u}")).collect::<Vec<_>>().join("\n"));
            Ok(0)
        }
        "verify" => {
            let path = target.map(PathBuf::from).unwrap_or_else(|| Path::new(state.unwrap_or(".harness")).join("audit.jsonl"));
            if !path.exists() {
                eprintln!("no audit trail at {}", path.display());
                return Ok(1);
            }
            let mut trail = crate::runtime::audit::AuditTrail::at(&path);
            let mut signed = "signatures not checked";
            if let Some(var) = key_env {
                let Some(key) = std::env::var(var).ok().filter(|k| !k.is_empty()) else {
                    eprintln!("${var} is not set");
                    return Ok(1);
                };
                trail = trail.with_signer(Arc::new(crate::governance::records::HMACSigner::new(key.as_bytes(), None)?));
                signed = "signed, signatures checked";
            }
            let (ok, why) = trail.verify();
            println!("{}: {} entries, {why} ({signed})", path.display(), trail.len());
            Ok(if ok { 0 } else { 1 })
        }
        "check" => {
            let gov = load("check")?;
            let policy = gov.policy();
            println!("ok   {}: policy {}@{} sha256:{} · mode {}", config.unwrap_or(""), policy.name, policy.version, chars(&gov.engine().hash, 16), gov.mode());
            println!("     packs: {}", if gov.packs().is_empty() { "none".to_string() } else { gov.packs().iter().map(|p| p.id.clone()).collect::<Vec<_>>().join(", ") });
            let mut purposes: Vec<&String> = policy.purposes.keys().collect();
            purposes.sort();
            println!("     rules: {} · purposes: {}", policy.rules.len(), if purposes.is_empty() { "none".to_string() } else { purposes.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") });
            let residency = policy.residency.as_ref();
            if let Some(r) = residency.filter(|r| !r.transfers.is_empty()) {
                let mut t: Vec<_> = r.transfers.iter().collect();
                t.sort_by(|a, b| a.0.cmp(b.0));
                println!("     residency: home {}; {}", r.home.as_deref().unwrap_or("(per principal)"), t.iter().map(|(o, d)| format!("{o} → {}", d.join(", "))).collect::<Vec<_>>().join("; "));
            }
            let mut warnings = Vec::new();
            if let Some(r) = residency
                && r.home.is_none()
                && r.transfers.values().filter(|d| !d.iter().any(|x| x == "*")).count() > 1
            {
                warnings.push("no `home:` — a person with no jurisdiction tag is unrestricted");
            }
            if gov.signer().is_none() {
                warnings.push("no signing key — the audit trail is hash-chained but unsigned");
            }
            if gov.vault().path.is_none() {
                warnings.push("no `vault:` path — the subject index is lost on restart, so erasure cannot find earlier sessions");
            }
            for w in warnings {
                println!("warn {w}");
            }
            Ok(0)
        }
        "inventory" => {
            let gov = load("inventory")?;
            let Some(agents) = agents else {
                eprintln!("inventory needs --agents agents.yaml (a blueprint)");
                return Ok(2);
            };
            let h = Harness::testing(None).with_governance(gov.clone())?;
            crate::blueprint::Blueprint::from_file(agents)?.build_all(&[], Some(h))?;
            if format == "json" {
                println!("{}", pretty(&gov.inventory().to_cyclonedx()));
                return Ok(0);
            }
            for entry in gov.inventory().agents().values() {
                let identity = &entry["identity"];
                let s = |v: &Value| v.as_str().map_or_else(|| v.to_string(), str::to_string);
                println!(
                    "{}  model={}  provider={}  risk={}  owner={}",
                    s(&entry["name"]),
                    s(&entry["model"]),
                    s(&entry["provider"]),
                    s(&identity["risk"]),
                    identity["owner"].as_str().filter(|o| !o.is_empty()).unwrap_or("-")
                );
                for t in entry["tools"].as_array().into_iter().flatten() {
                    let tags: Vec<&str> = t["tags"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                    println!("    {:<24} {}  {}", t["name"].as_str().unwrap_or(""), chars(t["fingerprint"].as_str().unwrap_or(""), 12), tags.join(","));
                }
                let servers: Vec<&str> = entry["mcp_servers"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
                if !servers.is_empty() {
                    println!("    mcp: {}", servers.join(", "));
                }
            }
            Ok(0)
        }
        "dsar" => {
            let gov = load("dsar")?;
            let (Some(kind @ ("access" | "erase")), Some(user)) = (target, user) else {
                eprintln!("usage: agent-harness governance dsar access|erase --user ID [--tenant ID] --state DIR --config governance.yaml");
                return Ok(2);
            };
            let _h = Harness::local(state.unwrap_or(".harness"))?.with_governance(gov.clone())?;
            let mut trace = crate::memory::Trace::new();
            trace.user_id = Some(user.into());
            trace.tenant_id = tenant.map(str::to_string);
            let result = if kind == "access" { gov.rights().access(&trace).await? } else { Value::Object(gov.rights().erase(&trace, requested_by).await?) };
            println!("{}", pretty(&result));
            Ok(0)
        }
        "report" => {
            let gov = load("report")?;
            let mut h = Harness::testing(None);
            gov.attach(&mut h)?;
            let report = match pack {
                Some(p) => gov.report(p),
                None => gov.report_all(),
            };
            println!("{}", match format {
                "json" => report.json(),
                "html" => report.html(),
                _ => report.markdown(),
            });
            Ok(0)
        }
        _ => Ok(2),
    }
}

async fn dispatch(cli: Cli) -> Result<i32> {
    match cli.command {
        Command::Run { task, stream, attach, json, report, common } => cmd_run(&task, stream, &attach, json, report, &common).await,
        Command::Chat { common } => cmd_chat(&common).await,
        Command::Voice { input, output, realtime, voice, language, greeting, common } => cmd_voice(input.as_deref(), output.as_deref(), realtime.as_deref(), &voice, language.as_deref(), greeting.as_deref(), &common).await,
        Command::Workflow { file, text, input, tools, check, json, state, trace, approve } => cmd_workflow(&file, text.as_deref(), &input, tools, check, json, state.as_deref(), trace, approve).await,
        Command::A2a { action, target, text, host, port, public_url, token, max_concurrency, stream, context, common } => {
            cmd_a2a(&action, target.as_deref(), text.as_deref(), &host, port, public_url.as_deref(), &token, max_concurrency, stream, context.as_deref(), &common).await
        }
        Command::McpServe { blueprint, api_key, expose_tools, stdio, host, port, common } => cmd_mcp_serve(blueprint.as_deref(), &api_key, expose_tools, stdio, &host, port, &common).await,
        Command::Openapi { spec, base_url, openapi_token, api_key, call, arg, json } => cmd_openapi(&spec, base_url.as_deref(), openapi_token.as_deref(), api_key.as_deref(), call.as_deref(), &arg, json).await,
        Command::Search { query, engine, limit, recency, domain, json } => cmd_search(&query, &engine, limit, recency.as_deref(), &domain, json).await,
        Command::Knowledge { action, items, store, namespace, embed_model, limit, json } => cmd_knowledge(&action, &items, store.as_deref(), &namespace, embed_model.as_deref(), limit, json).await,
        Command::Image { prompt, engine, model, reference, aspect, count, name, out } => cmd_image(&prompt, &engine, model.as_deref(), &reference, &aspect, count, &name, &out).await,
        Command::Models => Ok(cmd_models()),
        Command::Providers { name, ping, ready, json } => cmd_providers(name.as_deref(), ping, ready, json).await,
        Command::Sandboxes { name, check, json } => cmd_sandboxes(name.as_deref(), check, json).await,
        Command::Approvals { action, id, by, note, call, status, limit, json, common } => cmd_approvals(&action, id.as_deref(), by.as_deref(), &note, call.as_deref(), status.as_deref(), limit, json, &common).await,
        Command::Sessions { state, store, limit, user, tenant, agent, show, delete, check, backends, json } => {
            cmd_sessions(state.as_deref(), store.as_deref(), limit, user.as_deref(), tenant.as_deref(), agent.as_deref(), show.as_deref(), delete.as_deref(), check, backends, json).await
        }
        Command::Mcp { command, url, server_name } => cmd_mcp(&command, url.as_deref(), &server_name).await,
        Command::Journal { state, limit } => Ok(cmd_journal(state.as_deref(), limit)),
        Command::Governance { action, target, agents, user, tenant, requested_by, config, pack, format, state, key_env, json } => {
            cmd_governance(&action, target.as_deref(), agents.as_deref(), user.as_deref(), tenant.as_deref(), &requested_by, config.as_deref(), pack.as_deref(), &format, state.as_deref(), key_env.as_deref(), json).await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_approvals(action: &str, id: Option<&str>, by: Option<&str>, note: &str, call: Option<&str>, status: Option<&str>, limit: usize, json_out: bool, c: &Common) -> Result<i32> {
    let mut c = c.clone();
    if c.state.is_none() && c.sessions.is_none() {
        c.state = Some(".harness".into());
    }
    let h = common_harness(&c)?;
    let outcome = async {
        let desk = h.approvals.clone().ok_or_else(|| Error::config("this harness keeps no approvals"))?;
        if action == "list" {
            let rows = desk.list(status, None, None, None, limit).await?;
            if json_out {
                println!("{}", pretty(&Value::Array(rows.iter().map(|a| a.summary()).collect())));
                return Ok(0);
            }
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            for a in &rows {
                let state = if a.expired() { "expired" } else { a.status.as_str() };
                println!("{}  {state:<9} {:5.1}h  {}", a.id, (now - a.created) / 3600.0, a.describe());
            }
            if rows.is_empty() {
                println!("no approvals");
            }
            return Ok(0);
        }
        let Some(id) = id else { return Err(usage(format!("approvals {action} takes the approval's id"))) };
        match action {
            "show" => {
                let r = desk.get(id).await?;
                if json_out {
                    println!("{}", pretty(&r.summary()));
                    return Ok(0);
                }
                println!(
                    "{}  {}\nagent    {}\ntask     {}\nsession  {}  step {}",
                    r.id,
                    if r.expired() { "expired" } else { r.status.as_str() },
                    r.agent,
                    chars(&r.task, 300),
                    if r.session_id.is_empty() { "(not kept)" } else { &r.session_id },
                    r.step
                );
                for call in &r.calls {
                    let who = if call.by.is_empty() { String::new() } else { format!("  by {}{}", call.by, if call.note.is_empty() { String::new() } else { format!(": {}", call.note) }) };
                    let why = if call.reason.is_empty() { String::new() } else { format!("\n      why: {}", call.reason) };
                    println!("  [{}] {}{who}{why}", call.status, call.describe());
                }
                if !r.outcome.is_empty() {
                    println!("outcome  {}", Value::Object(r.outcome.clone()));
                }
                Ok(0)
            }
            "approve" | "deny" => {
                let Some(by) = by else { return Err(usage(format!("approvals {action} needs --by: who is deciding"))) };
                let r = if action == "approve" { desk.approve(id, by, note, call).await? } else { desk.deny(id, by, note, call).await? };
                let waiting = r.waiting().len();
                println!("{} is {}{}", r.id, r.status, if waiting > 0 { format!("; still waiting on {waiting}") } else { format!(" — resume it with: agent-harness approvals resume {}", r.id) });
                Ok(0)
            }
            "release" => {
                let r = desk.release(id).await?;
                println!("{} is {} again and can be resumed", r.id, r.status);
                Ok(0)
            }
            _ => {
                // resume: the agent is built from the same options a run takes.
                let agent = build_agent(&c, &h).await?;
                let result = agent.resume_approval(id).await?;
                println!("{}", result.output);
                if let Some(e) = &result.error {
                    eprintln!("\nerror: {e}");
                }
                tail(&result);
                Ok(if result.error.is_some() { 1 } else { 0 })
            }
        }
    }
    .await;
    h.close().await;
    outcome
}

/// Run the command line with these arguments (the first is the program name).
/// Returns the process exit code.
pub fn main_with<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return e.exit_code();
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().thread_stack_size(16 * 1024 * 1024).build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let outcome = runtime.block_on(async {
        tokio::select! {
            result = dispatch(cli) => Some(result),
            _ = tokio::signal::ctrl_c() => None,
        }
    });
    match outcome {
        None => {
            eprintln!("\ninterrupted");
            130
        }
        Some(Ok(code)) => code,
        Some(Err(Error::Configuration(m))) if m.starts_with("usage: ") => {
            eprintln!("{}", &m["usage: ".len()..]);
            2
        }
        Some(Err(e)) => {
            eprintln!("error: {e}");
            1
        }
    }
}

/// Run the command line with the process's own arguments.
///
/// It runs on a thread with a roomy stack: an agent's run is a deep future,
/// and the main thread's stack is small on Windows (1 MB).
pub fn main() -> i32 {
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    match std::thread::Builder::new().name("agent-harness".into()).stack_size(32 * 1024 * 1024).spawn(move || main_with(args)) {
        Ok(handle) => handle.join().unwrap_or(101),
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_parses_like_the_python_one() {
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["agent-harness", "run", "hello", "--tools", "--mode", "research", "--depth", "deep", "--attach", "a.png", "--attach", "b.pdf", "--state", ".h"]).unwrap();
        let Command::Run { task, attach, common, .. } = cli.command else { panic!() };
        assert_eq!((task.as_str(), attach.len(), common.tools, common.mode.as_deref(), common.depth.as_deref(), common.state.as_deref()), ("hello", 2, true, Some("research"), Some("deep"), Some(".h")));
        assert_eq!(common.name, "assistant");
        assert!(Cli::try_parse_from(["agent-harness", "run", "x", "--depth", "extreme"]).is_err());
        let cli = Cli::try_parse_from(["agent-harness", "approvals"]).unwrap();
        assert!(matches!(cli.command, Command::Approvals { ref action, .. } if action == "list"));
        let cli = Cli::try_parse_from(["agent-harness", "openapi", "spec.yaml", "--token", "t", "--arg", "id=7"]).unwrap();
        assert!(matches!(cli.command, Command::Openapi { ref openapi_token, ref arg, .. } if openapi_token.as_deref() == Some("t") && arg == &["id=7"]));
        assert!(Cli::try_parse_from(["agent-harness", "mcp-serve", "--stdio"]).is_ok());
        assert_eq!(thousands(1_048_576), "1,048,576");
        assert_eq!(thousands(200), "200");
        assert_eq!(value("40"), json!(40));
        assert_eq!(value("forty"), json!("forty"));
        assert!(pair("nope", "--input").unwrap_err().to_string().contains("--input takes name=value — got 'nope'"));
    }

    #[test]
    fn commands_that_need_no_network() {
        assert_eq!(main_with(["agent-harness", "models"]), 0);
        assert_eq!(main_with(["agent-harness", "governance", "packs"]), 0);
        assert_eq!(main_with(["agent-harness", "journal", "--state", "definitely-not-here"]), 1);
        assert_eq!(main_with(["agent-harness", "knowledge", "check"]), 2);
        assert_eq!(main_with(["agent-harness", "frobnicate"]), 2);
    }
}
