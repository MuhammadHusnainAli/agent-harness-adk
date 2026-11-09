//! The value types that travel through the harness.
//!
//! Every type here serialises to the same JSON as its Python counterpart, so a
//! session saved by one language can be read by the other.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};

/// Short, sortable-enough ids. `new_id("call")` → `call_1a2b3c4d5e6f`.
pub fn new_id(prefix: &str) -> String {
    let raw = uuid::Uuid::new_v4().simple().to_string();
    let raw = &raw[..12];
    if prefix.is_empty() { raw.to_string() } else { format!("{prefix}_{raw}") }
}

/// Seconds since the epoch, as Python's `time.time()`.
pub fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn is_false(b: &bool) -> bool {
    !*b
}

// ---------------------------------------------------------------------------
// content blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
    /// Gemini thinking models need this back on the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

impl TextBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), thought_signature: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBlock {
    pub thinking: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Encrypted (redacted) thinking; it must go back verbatim next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redacted: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUseBlock {
    #[serde(default = "call_id")]
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub input: Map<String, Value>,
    /// Gemini thinking models need this back on the next turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

fn call_id() -> String {
    new_id("call")
}

impl ToolUseBlock {
    pub fn new(name: impl Into<String>, input: Map<String, Value>) -> Self {
        Self { id: new_id("call"), name: name.into(), input, thought_signature: None }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

/// What kind of media a [`MediaBlock`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Audio,
    Video,
    Document,
}

impl MediaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Document => "document",
        }
    }

    /// The default media type for a block of this kind.
    pub fn default_media_type(self) -> &'static str {
        match self {
            Self::Image => "image/png",
            Self::Audio => "audio/wav",
            Self::Video => "video/mp4",
            Self::Document => "application/pdf",
        }
    }

    /// `image/*`, `audio/*`, `video/*` — and everything else is a document.
    pub fn from_media_type(media_type: &str) -> Self {
        match media_type.split('/').next().unwrap_or("") {
            "image" => Self::Image,
            "audio" => Self::Audio,
            "video" => Self::Video,
            _ => Self::Document,
        }
    }
}

/// Something that is not text: an image, a recording, a clip, a file.
///
/// Held one of three ways: `path` (a local file, read when sent), `data`
/// (base64 bytes), or `url` (somewhere the provider can fetch it from).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaBlock {
    #[serde(default = "octet")]
    pub media_type: String,
    /// base64
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
}

fn octet() -> String {
    "application/octet-stream".into()
}

fn guess(name: &str) -> Option<String> {
    mime_guess::from_path(name).first().map(|m| m.essence_str().to_string())
}

impl MediaBlock {
    pub fn from_path(path: impl AsRef<Path>, media_type: Option<&str>, kind: MediaKind) -> Self {
        let path = path.as_ref();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let media_type = media_type.map(str::to_string).or_else(|| guess(&name)).unwrap_or_else(|| kind.default_media_type().into());
        Self { media_type, data: String::new(), url: None, path: Some(path.to_string_lossy().into_owned()), name }
    }

    pub fn from_bytes(bytes: &[u8], media_type: &str) -> Self {
        Self { media_type: media_type.into(), data: B64.encode(bytes), url: None, path: None, name: String::new() }
    }

    pub fn from_url(url: &str, media_type: Option<&str>, kind: MediaKind) -> Self {
        let name = url.rsplit('/').next().unwrap_or("").split('?').next().unwrap_or("").to_string();
        let media_type = media_type.map(str::to_string).or_else(|| guess(&name)).unwrap_or_else(|| kind.default_media_type().into());
        Self { media_type, data: String::new(), url: Some(url.into()), path: None, name }
    }

    /// Can its bytes be sent, rather than only pointed at?
    pub fn inline(&self) -> bool {
        !self.data.is_empty() || self.path.is_some()
    }

    /// The content as base64. Errors if `path` is gone, or it is only a URL.
    pub fn load(&self) -> Result<String> {
        if !self.data.is_empty() {
            return Ok(self.data.clone());
        }
        if let Some(path) = &self.path {
            return Ok(B64.encode(std::fs::read(path)?));
        }
        Err(Error::Io(format!("not found: {}", self.url.as_deref().unwrap_or(&self.name))))
    }

    /// The raw bytes.
    pub fn read(&self) -> Result<Vec<u8>> {
        B64.decode(self.load()?).map_err(|e| Error::Io(e.to_string()))
    }

    /// Size in bytes; 0 for one that is only a URL.
    pub fn size(&self) -> u64 {
        if !self.data.is_empty() {
            let pad = self.data.chars().rev().take(2).filter(|c| *c == '=').count();
            return (self.data.len() * 3 / 4 - pad) as u64;
        }
        if let Some(path) = &self.path {
            return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        }
        0
    }
}

/// One piece of a message. Serialised with a `type` tag, as in Python.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text(TextBlock),
    Thinking(ThinkingBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Image(MediaBlock),
    Audio(MediaBlock),
    Video(MediaBlock),
    Document(MediaBlock),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextBlock::new(text))
    }

    pub fn media(kind: MediaKind, block: MediaBlock) -> Self {
        match kind {
            MediaKind::Image => Self::Image(block),
            MediaKind::Audio => Self::Audio(block),
            MediaKind::Video => Self::Video(block),
            MediaKind::Document => Self::Document(block),
        }
    }

    /// The `type` tag.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Thinking(_) => "thinking",
            Self::ToolUse(_) => "tool_use",
            Self::ToolResult(_) => "tool_result",
            Self::Image(_) => "image",
            Self::Audio(_) => "audio",
            Self::Video(_) => "video",
            Self::Document(_) => "document",
        }
    }

    /// The media inside, and what kind it is — `None` for text and tool blocks.
    pub fn as_media(&self) -> Option<(MediaKind, &MediaBlock)> {
        match self {
            Self::Image(m) => Some((MediaKind::Image, m)),
            Self::Audio(m) => Some((MediaKind::Audio, m)),
            Self::Video(m) => Some((MediaKind::Video, m)),
            Self::Document(m) => Some((MediaKind::Document, m)),
            _ => None,
        }
    }
}

/// Whatever you have, as the block that carries it: a path or an
/// `http(s)://` / `gs://` URL. The kind is read from the media type.
///
/// ```
/// let block = agent_harness::types::attach("chart.png", None);
/// assert_eq!(block.type_name(), "image");
/// ```
pub fn attach(source: &str, media_type: Option<&str>) -> ContentBlock {
    let base = source.split('?').next().unwrap_or(source);
    let mt = media_type.map(str::to_string).or_else(|| guess(base)).unwrap_or_else(octet);
    let kind = MediaKind::from_media_type(&mt);
    let block = if source.starts_with("http://") || source.starts_with("https://") || source.starts_with("gs://") {
        MediaBlock::from_url(source, media_type, kind)
    } else {
        MediaBlock::from_path(source, media_type, kind)
    };
    ContentBlock::media(kind, block)
}

/// Bytes in memory, with what they are: `attach_bytes(&pcm, "audio/wav")`.
pub fn attach_bytes(bytes: &[u8], media_type: &str) -> ContentBlock {
    ContentBlock::media(MediaKind::from_media_type(media_type), MediaBlock::from_bytes(bytes, media_type))
}

// ---------------------------------------------------------------------------
// messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// `"user"`, `"assistant"`, ... — as it is serialised.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One turn of the conversation, always stored as a list of blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default = "now")]
    pub ts: f64,
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content, name: None, ts: now() }
    }

    /// A user turn.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    /// A user turn with attachments; they go before the text.
    pub fn user_with(text: impl Into<String>, attachments: Vec<ContentBlock>) -> Self {
        let text = text.into();
        let mut blocks = attachments;
        if !text.is_empty() || blocks.is_empty() {
            blocks.push(ContentBlock::text(text));
        }
        Self::new(Role::User, blocks)
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    pub fn assistant_blocks(content: Vec<ContentBlock>) -> Self {
        Self::new(Role::Assistant, content)
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    /// Tool output goes back as a *user* turn — what every provider expects.
    pub fn tool_results(results: Vec<ToolResultBlock>) -> Self {
        Self::new(Role::User, results.into_iter().map(ContentBlock::ToolResult).collect())
    }

    /// All text blocks, joined by newlines and trimmed.
    pub fn text(&self) -> String {
        let parts: Vec<&str> = self
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        parts.join("\n").trim().to_string()
    }

    pub fn tool_uses(&self) -> Vec<&ToolUseBlock> {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    pub fn media(&self) -> Vec<(MediaKind, &MediaBlock)> {
        self.content.iter().filter_map(ContentBlock::as_media).collect()
    }
}

// ---------------------------------------------------------------------------
// accounting and responses
// ---------------------------------------------------------------------------

/// Token and money accounting for a single call, or a whole run once summed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub cost_usd: f64,
    pub calls: u64,
}

pub(crate) fn round8(x: f64) -> f64 {
    (x * 1e8).round() / 1e8
}

impl Usage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;
    fn add(self, o: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + o.input_tokens,
            output_tokens: self.output_tokens + o.output_tokens,
            cache_read_tokens: self.cache_read_tokens + o.cache_read_tokens,
            cache_write_tokens: self.cache_write_tokens + o.cache_write_tokens,
            cost_usd: round8(self.cost_usd + o.cost_usd),
            calls: self.calls + o.calls,
        }
    }
}

impl std::ops::AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, o: &Usage) {
        *self = self.clone() + o.clone();
    }
}

/// Why a model call or a run stopped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[default]
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    MaxSteps,
    Error,
    Stopped,
    Budget,
    Handoff,
    Approval,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EndTurn => "end_turn",
            Self::ToolUse => "tool_use",
            Self::MaxTokens => "max_tokens",
            Self::StopSequence => "stop_sequence",
            Self::MaxSteps => "max_steps",
            Self::Error => "error",
            Self::Stopped => "stopped",
            Self::Budget => "budget",
            Self::Handoff => "handoff",
            Self::Approval => "approval",
        }
    }
}

/// What a provider hands back, normalised across vendors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub message: Message,
    #[serde(default)]
    pub stop_reason: StopReason,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl ModelResponse {
    pub fn new(message: Message) -> Self {
        Self { message, stop_reason: StopReason::EndTurn, usage: Usage::default(), model: String::new(), provider: String::new(), latency_ms: 0.0, raw: None }
    }

    pub fn text(&self) -> String {
        self.message.text()
    }

    pub fn tool_uses(&self) -> Vec<&ToolUseBlock> {
        self.message.tool_uses()
    }
}

/// The kind of beat in a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEventType {
    RunStart,
    StepStart,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    StepEnd,
    RunEnd,
    Error,
    Delegation,
    Progress,
    Handoff,
    ApprovalRequired,
}

/// One beat of a streaming run: a token, a tool call, a step boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub kind: StreamEventType,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub agent: String,
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub data: Map<String, Value>,
}

impl StreamEvent {
    pub fn new(kind: StreamEventType) -> Self {
        Self { kind, text: String::new(), agent: String::new(), step: 0, data: Map::new() }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Self::new(StreamEventType::Text) }
    }

    pub fn with_data(mut self, key: &str, value: Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }

    pub fn with_step(mut self, agent: &str, step: u32) -> Self {
        self.agent = agent.into();
        self.step = step;
        self
    }
}

// ---------------------------------------------------------------------------
// what runs produce
// ---------------------------------------------------------------------------

/// A tool invocation as it happened, for the journal and the audit trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default = "call_id")]
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub args: Map<String, Value>,
    #[serde(default)]
    pub agent: String,
    #[serde(default)]
    pub step: u32,
}

/// The result of running a tool, plus everything the rails want to record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub call_id: String,
    pub name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default)]
    pub cached: bool,
    /// The tool's own return value, before it was rendered to text.
    #[serde(skip)]
    pub value: Option<Value>,
    /// Images (or other media) the tool returned for the model to look at.
    #[serde(skip)]
    pub media: Vec<ContentBlock>,
    /// Set instead of a result when the call is waiting for a person.
    #[serde(skip)]
    pub pending: Option<Value>,
}

impl ToolOutcome {
    pub fn as_block(&self) -> ToolResultBlock {
        ToolResultBlock { tool_use_id: self.call_id.clone(), content: self.content.clone(), is_error: self.is_error }
    }
}

/// A file or document a run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "text_plain")]
    pub media_type: String,
    #[serde(default)]
    pub produced_by: String,
    #[serde(default = "now")]
    pub ts: f64,
    /// Anything else carried with it, as Python's `extra="allow"` keeps it:
    /// `run_id`, `version` and `digest` once it is in a deliverable store.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn text_plain() -> String {
    "text/plain".into()
}

impl Artifact {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self { name: name.into(), content: content.into(), path: None, media_type: text_plain(), produced_by: String::new(), ts: now(), extra: Map::new() }
    }

    /// A string the artefact carries beyond its fields (`run_id`, `digest`).
    pub fn extra_str(&self, key: &str) -> &str {
        self.extra.get(key).and_then(Value::as_str).unwrap_or("")
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    #[default]
    Pending,
    InProgress,
    Done,
    Skipped,
}

/// One item on the list an agent keeps while it works.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Todo {
    pub content: String,
    #[serde(default)]
    pub status: TodoStatus,
    #[serde(default)]
    pub note: String,
}

impl Todo {
    pub fn is_open(&self) -> bool {
        matches!(self.status, TodoStatus::Pending | TodoStatus::InProgress)
    }

    pub fn line(&self) -> String {
        let mark = match self.status {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Done => "[x]",
            TodoStatus::Skipped => "[-]",
        };
        if self.note.is_empty() { format!("{mark} {}", self.content) } else { format!("{mark} {} — {}", self.content, self.note) }
    }
}

/// Something an agent read and relied on. `id` is the number it is cited by.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub id: u32,
    pub r#ref: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub finding: String,
    #[serde(default)]
    pub agent: String,
}

impl Source {
    pub fn line(&self) -> String {
        if self.title.is_empty() { format!("[{}] {}", self.id, self.r#ref) } else { format!("[{}] {} — {}", self.id, self.title, self.r#ref) }
    }
}

/// One time a conversation changed hands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffRecord {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub run_id: String,
    #[serde(default = "now")]
    pub ts: f64,
}

impl HandoffRecord {
    pub fn line(&self) -> String {
        if self.reason.is_empty() { format!("{} → {}", self.source, self.target) } else { format!("{} → {}: {}", self.source, self.target, self.reason) }
    }
}

/// The one object a caller gets back from `agent.run()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunResult {
    pub output: String,
    pub data: Value,
    pub agent: String,
    pub messages: Vec<Message>,
    pub usage: Usage,
    pub steps: u32,
    pub stop_reason: StopReason,
    pub tool_calls: Vec<ToolCall>,
    pub artifacts: Vec<Artifact>,
    pub session_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub error: Option<String>,
    pub budget_exceeded: Option<String>,
    pub approval: Option<Value>,
    pub spend: Map<String, Value>,
    pub violations: Vec<String>,
    pub children: Vec<RunResult>,
    pub mode: String,
    pub depth: String,
    pub todos: Vec<Todo>,
    pub sources: Vec<Source>,
    pub sandbox_id: String,
    pub handoffs: Vec<HandoffRecord>,
    pub active_agent: String,
    pub warnings: Vec<String>,
    /// Anything else carried with it, as Python's `extra="allow"` keeps it:
    /// governance adds `disclosure`, `provenance` and `governance`.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Default for RunResult {
    fn default() -> Self {
        Self {
            output: String::new(),
            data: Value::Null,
            agent: String::new(),
            messages: Vec::new(),
            usage: Usage::default(),
            steps: 0,
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            artifacts: Vec::new(),
            session_id: String::new(),
            run_id: new_id("run"),
            trace_id: String::new(),
            error: None,
            budget_exceeded: None,
            approval: None,
            spend: Map::new(),
            violations: Vec::new(),
            children: Vec::new(),
            mode: String::new(),
            depth: String::new(),
            todos: Vec::new(),
            sources: Vec::new(),
            sandbox_id: String::new(),
            handoffs: Vec::new(),
            active_agent: String::new(),
            warnings: Vec::new(),
            extra: Map::new(),
        }
    }
}

impl RunResult {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }

    /// This run plus everything it delegated.
    pub fn cost_usd(&self) -> f64 {
        round8(self.usage.cost_usd + self.children.iter().map(RunResult::cost_usd).sum::<f64>())
    }
}

impl std::fmt::Display for RunResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.output)
    }
}

/// Free-form key/value data, as Python's `dict[str, Any]`.
pub type Data = HashMap<String, Value>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ids_have_prefix_and_length() {
        let id = new_id("call");
        assert!(id.starts_with("call_"));
        assert_eq!(id.len(), 17);
        assert_eq!(new_id("").len(), 12);
    }

    #[test]
    fn blocks_serialise_with_python_shape() {
        let msg = Message::assistant_blocks(vec![
            ContentBlock::text("hi"),
            ContentBlock::ToolUse(ToolUseBlock {
                id: "call_1".into(),
                name: "add".into(),
                input: json!({"a": 1}).as_object().unwrap().clone(),
                thought_signature: None,
            }),
        ]);
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][0], json!({"type": "text", "text": "hi"}));
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["input"]["a"], 1);
        let back: Message = serde_json::from_value(v).unwrap();
        assert_eq!(back.content, msg.content);
    }

    #[test]
    fn parses_python_message_json() {
        let raw = json!({"role": "user", "content": [
            {"type": "image", "media_type": "image/jpeg", "url": "https://x/y.jpg"},
            {"type": "text", "text": " where? "}
        ], "ts": 1.0});
        let msg: Message = serde_json::from_value(raw).unwrap();
        assert_eq!(msg.text(), "where?");
        assert_eq!(msg.media().len(), 1);
        assert_eq!(msg.media()[0].0, MediaKind::Image);
    }

    #[test]
    fn attach_picks_kind() {
        assert_eq!(attach("chart.png", None).type_name(), "image");
        assert_eq!(attach("call.mp3", None).type_name(), "audio");
        assert_eq!(attach("report.pdf", None).type_name(), "document");
        let clip = attach("https://example.com/clip.mp4?x=1", None);
        assert_eq!(clip.type_name(), "video");
        let (_, m) = clip.as_media().unwrap();
        assert_eq!(m.name, "clip.mp4");
        assert!(!m.inline());
    }

    #[test]
    fn media_bytes_round_trip() {
        let ContentBlock::Audio(m) = attach_bytes(b"hello", "audio/wav") else { panic!() };
        assert_eq!(m.size(), 5);
        assert_eq!(m.read().unwrap(), b"hello");
    }

    #[test]
    fn user_with_puts_attachments_first() {
        let msg = Message::user_with("look", vec![attach("a.png", None)]);
        assert_eq!(msg.content[0].type_name(), "image");
        assert_eq!(msg.content[1].type_name(), "text");
        assert_eq!(Message::user_with("", vec![attach("a.png", None)]).content.len(), 1);
    }

    #[test]
    fn usage_adds() {
        let mut a = Usage { input_tokens: 1, output_tokens: 2, cost_usd: 0.1, calls: 1, ..Default::default() };
        a += &Usage { input_tokens: 3, cost_usd: 0.2, calls: 1, ..Default::default() };
        assert_eq!(a.input_tokens, 4);
        assert_eq!(a.total_tokens(), 6);
        assert_eq!(a.cost_usd, 0.3);
    }

    #[test]
    fn run_result_cost_includes_children() {
        let child = RunResult { usage: Usage { cost_usd: 0.5, ..Default::default() }, ..Default::default() };
        let r = RunResult { usage: Usage { cost_usd: 0.25, ..Default::default() }, children: vec![child], ..Default::default() };
        assert_eq!(r.cost_usd(), 0.75);
        assert!(r.ok());
        assert!(r.run_id.starts_with("run_"));
    }

    #[test]
    fn todo_and_source_lines() {
        let t = Todo { content: "write".into(), status: TodoStatus::Done, note: "ok".into() };
        assert_eq!(t.line(), "[x] write — ok");
        assert!(!t.is_open());
        let s = Source { id: 2, r#ref: "https://a".into(), title: "A".into(), finding: String::new(), agent: String::new() };
        assert_eq!(s.line(), "[2] A — https://a");
        assert_eq!(serde_json::to_value(&s).unwrap()["ref"], "https://a");
    }

    #[test]
    fn stream_event_type_field() {
        let e = StreamEvent::text("hi");
        assert_eq!(serde_json::to_value(&e).unwrap()["type"], "text");
    }
}
