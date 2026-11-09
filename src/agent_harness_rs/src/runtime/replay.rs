//! Replay and time travel: re-run from any prior step. Reproduce before you fix.
//!
//! Two different things, both useful:
//!
//! **Time travel** — [`Replayer::resume`] loads a checkpoint and continues the
//! run from that step with a live model. Use it when a run went wrong at step 7
//! and you want to try step 7 again with a changed prompt or tool.
//!
//! **Deterministic reproduction** — wrap your provider in [`RecordingProvider`]
//! once, then replay the recording with [`ReplayProvider`]. The same model
//! responses come back in the same order, with no network and no spend, so a
//! bug reproduces exactly. This is what turns "it failed in production" into a
//! test. The recording file is the Python one.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::agent::{Agent, RunOptions};
use crate::errors::{Error, ProviderError, ProviderErrorKind, Result};
use crate::llm_providers::base::{CompletionRequest, Provider, ProviderSpec};
use crate::pyjson::{Dumps, dumps_with};
use crate::runtime::checkpoints::{Checkpoint, Checkpointer};
use crate::types::{MediaKind, Message, ModelResponse, Role, RunResult};

/// A stable fingerprint of a model call, used to match a recording.
pub fn request_key(request: &CompletionRequest) -> String {
    let mut tools: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
    tools.sort();
    let tool_choice = match &request.tool_choice {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    let blob = json!({
        "model": request.model,
        "system": request.system.clone().unwrap_or_default(),
        "messages": request.messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect::<Vec<_>>(),
        "tools": tools,
        "tool_choice": tool_choice,
    });
    let digest: String = Sha256::digest(dumps_with(&blob, Dumps::canonical()).as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    digest[..32].to_string()
}

/// Wraps a real provider and writes every exchange to a JSONL file.
pub struct RecordingProvider {
    pub inner: Arc<dyn Provider>,
    pub path: PathBuf,
    name: String,
    count: AtomicU64,
    file: Mutex<()>,
}

impl RecordingProvider {
    pub fn new(inner: Arc<dyn Provider>, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let name = format!("recording:{}", inner.name());
        Self { inner, path, name, count: AtomicU64::new(0), file: Mutex::new(()) }
    }

    /// How many exchanges have been recorded.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    fn spec(&self) -> &ProviderSpec {
        self.inner.spec()
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let response = self.inner.complete(req).await?;
        let seq = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        let record = json!({"seq": seq, "key": request_key(req), "model": req.model, "response": response});
        let _held = self.file.lock().unwrap_or_else(|p| p.into_inner());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = writeln!(f, "{}", crate::pyjson::dumps(&record));
        }
        Ok(response)
    }

    async fn embed(&self, texts: &[String], model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        self.inner.embed(texts, model).await
    }
}

const REPLAY: ProviderSpec = ProviderSpec {
    name: "replay",
    display_name: "Replay (recorded)",
    description: "Replays a recording: no network, no keys, no spend.",
    docs_url: "",
    auth_type: "none",
    env_key: "",
    default_model: "",
    base_url: "",
    capabilities: &["streaming", "tools"],
    parameters: &crate::llm_providers::parameters::SAMPLING_PARAMETERS,
    modalities: &[MediaKind::Image, MediaKind::Audio, MediaKind::Video, MediaKind::Document],
    fields: &[],
};

struct ReplayState {
    used: HashSet<usize>,
    divergences: Vec<String>,
    requests: Vec<CompletionRequest>,
}

/// Replays a recording. No network, no keys, no spend.
///
/// Matches each call by fingerprint. If the run diverges from the recording —
/// because you changed a prompt — it falls back to the next unused response in
/// order, and `divergences()` tells you where that happened. `strict` fails
/// instead.
pub struct ReplayProvider {
    pub path: PathBuf,
    pub strict: bool,
    records: Vec<Value>,
    by_key: HashMap<String, Vec<usize>>,
    state: Mutex<ReplayState>,
}

impl ReplayProvider {
    pub fn new(path: impl Into<PathBuf>, strict: bool) -> Result<Self> {
        let path = path.into();
        let text = std::fs::read_to_string(&path).map_err(|_| Error::config(format!("no recording at {}", path.display())))?;
        let records: Vec<Value> = text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect();
        let mut by_key: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in records.iter().enumerate() {
            by_key.entry(r["key"].as_str().unwrap_or("").to_string()).or_default().push(i);
        }
        Ok(Self { path, strict, records, by_key, state: Mutex::new(ReplayState { used: HashSet::new(), divergences: vec![], requests: vec![] }) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ReplayState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn divergences(&self) -> Vec<String> {
        self.lock().divergences.clone()
    }

    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.lock().requests.clone()
    }

    /// Every recorded response has been used.
    pub fn exhausted(&self) -> bool {
        self.lock().used.len() >= self.records.len()
    }

    fn take(&self, req: &CompletionRequest) -> Result<Value> {
        let key = request_key(req);
        let mut s = self.lock();
        let call = s.requests.len().saturating_sub(1);
        if let Some(i) = self.by_key.get(&key).into_iter().flatten().copied().find(|i| !s.used.contains(i)) {
            s.used.insert(i);
            return Ok(self.records[i].clone());
        }
        s.divergences.push(format!("call {call}: no recording matches this request"));
        let fail = |m: String| Error::Provider(Box::new(ProviderError::new(ProviderErrorKind::Other, m).provider("replay")));
        if self.strict {
            return Err(fail(format!("replay diverged at call {call} — the request does not match the recording (re-record, or use strict=false)")));
        }
        match (0..self.records.len()).find(|i| !s.used.contains(i)) {
            Some(i) => {
                s.used.insert(i);
                Ok(self.records[i].clone())
            }
            None => Err(fail("the recording ran out of responses".into())),
        }
    }
}

#[async_trait]
impl Provider for ReplayProvider {
    fn spec(&self) -> &ProviderSpec {
        &REPLAY
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        self.lock().requests.push(req.clone());
        let record = self.take(req)?;
        serde_json::from_value(record["response"].clone()).map_err(|e| Error::config(format!("a recorded response could not be read: {e}")))
    }
}

/// Re-run a recorded run, whole or from a given step.
#[derive(Debug, Clone)]
pub struct Replayer {
    pub checkpoints: Checkpointer,
}

impl Replayer {
    pub fn new(checkpoints: Checkpointer) -> Self {
        Self { checkpoints }
    }

    pub fn history(&self, run_id: &str) -> Vec<Checkpoint> {
        self.checkpoints.history(run_id)
    }

    /// The state the agent was holding at that step.
    pub fn inspect(&self, run_id: &str, step: Option<u32>) -> Result<Checkpoint> {
        self.checkpoints.load(run_id, step)
    }

    /// A step-by-step summary of the run — what to read before you re-run it.
    pub fn timeline(&self, run_id: &str) -> Vec<Value> {
        self.history(run_id)
            .iter()
            .map(|c| {
                let last = c.messages.last();
                json!({
                    "step": c.step, "agent": c.agent, "messages": c.messages.len(),
                    "cost_usd": (c.usage.cost_usd * 1e6).round() / 1e6,
                    "tools_called": last.map(|m| m.tool_uses().iter().map(|t| t.name.clone()).collect::<Vec<_>>()).unwrap_or_default(),
                    "last": last.map(|m| m.text().chars().take(160).collect::<String>()).unwrap_or_default(),
                })
            })
            .collect()
    }

    /// Continue the run from `step` (default: the last checkpoint).
    ///
    /// The agent keeps the conversation it had at that point. Pass `task` to
    /// change what it is asked to do next — that is the point of time travel.
    pub async fn resume(&self, agent: &Agent, run_id: &str, step: Option<u32>, task: Option<&str>) -> Result<RunResult> {
        let checkpoint = self.checkpoints.load(run_id, step)?;
        let mut history = checkpoint.messages.clone();
        // Continue from a clean boundary: a trailing assistant turn with
        // unanswered tool calls would leave the next request with orphans.
        while history.last().is_some_and(|m| m.role == Role::Assistant && !m.tool_uses().is_empty()) {
            history.pop();
        }
        let last_user = history.iter().rev().find(|m| m.role == Role::User && !m.text().is_empty()).map(Message::text).unwrap_or_default();
        let prompt = task.map(str::to_string).unwrap_or_else(|| if last_user.is_empty() { "Continue.".into() } else { last_user });
        if task.is_none() && history.last().is_some_and(|m| m.role == Role::User) {
            history.pop();
        }
        agent.run_with(prompt.as_str(), RunOptions { messages: Some(history), ..Default::default() }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::FakeProvider;

    #[tokio::test]
    async fn record_then_replay_offline() {
        let path = std::env::temp_dir().join(format!("ah_rec_{}.jsonl", uuid::Uuid::new_v4().simple()));
        let live: Arc<dyn Provider> = Arc::new(FakeProvider::new(vec!["first".into(), "second".into()]));
        let rec = Arc::new(RecordingProvider::new(live, &path));
        let a = Agent::builder("w").provider(rec.clone()).memory(false).build().unwrap();
        a.run("one").await.unwrap();
        a.run("two").await.unwrap();
        assert_eq!(rec.count(), 2);

        let replay = Arc::new(ReplayProvider::new(&path, false).unwrap());
        let b = Agent::builder("w").provider(replay.clone()).memory(false).build().unwrap();
        // Asked in the other order: matched by fingerprint, not position.
        assert_eq!(b.run("two").await.unwrap().output, "second");
        assert_eq!(b.run("one").await.unwrap().output, "first");
        assert!(replay.divergences().is_empty() && replay.exhausted());

        let strict = Arc::new(ReplayProvider::new(&path, true).unwrap());
        let c = Agent::builder("w").provider(strict).memory(false).build().unwrap();
        assert!(c.run("something else").await.unwrap().error.unwrap().contains("replay diverged at call 0"));
        let _ = std::fs::remove_file(path);
    }
}
