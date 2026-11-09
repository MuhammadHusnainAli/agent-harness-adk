//! The A2A wire format: what goes over the line, and how it maps to a run.
//!
//! This is the JSON-RPC binding of A2A 0.3. Everything on the wire is a plain
//! JSON value, built and read here, so the server and the client agree by
//! sharing one set of functions rather than two readings of the specification.

use std::fmt;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Map, Value, json};

use crate::errors::Error;
use crate::types::{Artifact, ContentBlock, MediaBlock, RunResult, attach, attach_bytes};

/// The version of A2A both sides speak.
pub const PROTOCOL_VERSION: &str = "0.3.0";
/// Where the agent card is.
pub const CARD_PATH: &str = "/.well-known/agent-card.json";
/// Where the card lived before 0.3. Still served, still looked for.
pub const LEGACY_CARD_PATH: &str = "/.well-known/agent.json";

/// A task in one of these states is over: it takes no more messages.
pub const TERMINAL: [&str; 4] = ["completed", "canceled", "failed", "rejected"];
/// The answer of a run, among the artefacts of its task.
pub const ANSWER: &str = "response";

// JSON-RPC's own errors, and the ones A2A adds.
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
pub const SERVER_BUSY: i64 = -32000;
pub const TASK_NOT_FOUND: i64 = -32001;
pub const TASK_NOT_CANCELABLE: i64 = -32002;
pub const PUSH_NOT_SUPPORTED: i64 = -32003;
pub const UNSUPPORTED_OPERATION: i64 = -32004;
pub const CONTENT_TYPE_NOT_SUPPORTED: i64 = -32005;
pub const INVALID_AGENT_RESPONSE: i64 = -32006;
pub const NO_EXTENDED_CARD: i64 = -32007;

/// The largest file an artefact carries inline. Larger ones are named, not sent.
pub const MAX_INLINE_BYTES: usize = 1_000_000;

/// Is a task in this state over?
pub fn is_terminal(state: &str) -> bool {
    TERMINAL.contains(&state)
}

/// An A2A request that was refused, or a reply that was an error.
///
/// `code` is the JSON-RPC error code; `status` the HTTP status that goes with
/// it when the refusal is about the request rather than the task.
#[derive(Debug, Clone, PartialEq)]
pub struct A2AError {
    pub message: String,
    pub code: i64,
    /// Boxed: a JSON object is large, and errors travel often.
    pub data: Option<Box<Value>>,
    pub status: u16,
    /// Seconds the server asked the caller to wait before trying again.
    pub retry_after: Option<f64>,
}

impl A2AError {
    /// An error with this code, about the task (HTTP 200).
    pub fn new(message: impl Into<String>, code: i64) -> Self {
        Self { message: message.into(), code, data: None, status: 200, retry_after: None }
    }

    /// An error about the request itself, answered with an HTTP status.
    pub fn http(message: impl Into<String>, code: i64, status: u16) -> Self {
        Self { status, ..Self::new(message, code) }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(Box::new(data));
        self
    }

    pub fn with_retry_after(mut self, seconds: f64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// The JSON-RPC `error` object.
    pub fn body(&self) -> Value {
        let mut error = json!({"code": self.code, "message": self.message});
        if let Some(d) = &self.data {
            error["data"] = (**d).clone();
        }
        error
    }
}

impl fmt::Display for A2AError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for A2AError {}

/// A harness error met while answering a request: a stale save is the
/// server's problem (`-32603`), anything else the request's (`-32602`).
impl From<Error> for A2AError {
    fn from(e: Error) -> Self {
        let code = if matches!(e, Error::SessionConflict { .. }) { INTERNAL_ERROR } else { INVALID_PARAMS };
        Self::new(e.to_string(), code)
    }
}

/// For `?` in code that returns the crate's `Result`: an A2A failure is an
/// I/O failure as far as the rest of the harness is concerned.
impl From<A2AError> for Error {
    fn from(e: A2AError) -> Self {
        Error::Io(format!("A2AError: {}", e.message))
    }
}

/// A timestamp as A2A writes one: ISO 8601, UTC.
pub fn now() -> String {
    crate::governance::utc(crate::types::now(), true)
}

pub fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ----------------------------------------------------------------------
// parts and messages
// ----------------------------------------------------------------------

pub fn text_part(text: &str) -> Value {
    json!({"kind": "text", "text": text})
}

/// A message: `role` is `user` or `agent`. Empty ids are left out; an empty
/// `message_id` gets a fresh one.
pub fn message(role: &str, parts: Vec<Value>, task_id: &str, context_id: &str, message_id: &str, metadata: Option<&Map<String, Value>>) -> Value {
    let mut out = Map::new();
    out.insert("kind".into(), json!("message"));
    out.insert("role".into(), json!(role));
    out.insert("messageId".into(), json!(if message_id.is_empty() { new_uuid() } else { message_id.to_string() }));
    out.insert("parts".into(), Value::Array(parts));
    if !task_id.is_empty() {
        out.insert("taskId".into(), json!(task_id));
    }
    if !context_id.is_empty() {
        out.insert("contextId".into(), json!(context_id));
    }
    if let Some(m) = metadata.filter(|m| !m.is_empty()) {
        out.insert("metadata".into(), Value::Object(m.clone()));
    }
    Value::Object(out)
}

/// A message of one text part.
pub fn text_message(role: &str, text: &str, task_id: &str, context_id: &str) -> Value {
    message(role, vec![text_part(text)], task_id, context_id, "", None)
}

/// An attachment as a file part: its bytes, or where to fetch it.
pub fn file_part(block: &MediaBlock) -> Value {
    let mut file = Map::new();
    file.insert("mimeType".into(), json!(block.media_type));
    if !block.name.is_empty() {
        file.insert("name".into(), json!(block.name));
    }
    if block.inline() {
        file.insert("bytes".into(), json!(block.load().unwrap_or_default()));
    } else {
        file.insert("uri".into(), json!(block.url.clone().unwrap_or_default()));
    }
    json!({"kind": "file", "file": file})
}

/// The parts of a message: the attachments as files, then the text.
pub fn parts_of(text: &str, attachments: &[ContentBlock]) -> Vec<Value> {
    let mut parts: Vec<Value> = attachments.iter().filter_map(|b| b.as_media().map(|(_, m)| file_part(m))).collect();
    if !text.is_empty() || parts.is_empty() {
        parts.push(text_part(text));
    }
    parts
}

fn kind_of(part: &Map<String, Value>) -> String {
    part.get("kind").or_else(|| part.get("type")).and_then(Value::as_str).filter(|k| !k.is_empty()).unwrap_or("").to_string()
}

fn py_json(value: &Value) -> String {
    crate::pyjson::dumps_with(value, crate::pyjson::Dumps { ensure_ascii: false, ..Default::default() })
}

/// What a message says, and what came with it.
///
/// Text parts are the text. A data part is structured input: it is written out
/// as JSON, where the model can read it. A file part becomes an attachment.
pub fn read_parts(parts: Option<&Value>) -> Result<(String, Vec<ContentBlock>), A2AError> {
    let Some(Value::Array(parts)) = parts.filter(|p| p.as_array().is_some_and(|a| !a.is_empty())) else {
        return Err(A2AError::new("a message needs at least one part", INVALID_PARAMS));
    };
    let mut texts: Vec<String> = Vec::new();
    let mut files: Vec<ContentBlock> = Vec::new();
    for part in parts {
        let Value::Object(part) = part else { return Err(A2AError::new("a part must be an object", INVALID_PARAMS)) };
        match kind_of(part).as_str() {
            "text" => texts.push(part.get("text").map(py_text).unwrap_or_default()),
            "data" => texts.push(py_json(part.get("data").unwrap_or(&Value::Null))),
            "file" => files.push(read_file(part.get("file")).map_err(|e| A2AError::new(format!("a file part could not be read: {e}"), INVALID_PARAMS))?),
            other => {
                let shown = if other.is_empty() { "None".to_string() } else { format!("'{other}'") };
                return Err(A2AError::new(format!("a part of kind {shown} is not supported"), CONTENT_TYPE_NOT_SUPPORTED));
            }
        }
    }
    let text = texts.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n\n");
    Ok((text, files))
}

/// `str(part.get("text") or "")`.
fn py_text(v: &Value) -> String {
    match v {
        Value::Null | Value::Bool(false) => String::new(),
        Value::String(s) => s.clone(),
        other => crate::governance::py_str(other),
    }
}

fn read_file(file: Option<&Value>) -> Result<ContentBlock, String> {
    let empty = Map::new();
    let file = file.and_then(Value::as_object).unwrap_or(&empty);
    let s = |k: &str| file.get(k).and_then(Value::as_str).filter(|v| !v.is_empty());
    let media_type = s("mimeType").unwrap_or("application/octet-stream");
    let name = s("name").unwrap_or("");
    if let Some(bytes) = s("bytes") {
        let raw = B64.decode(bytes).map_err(|e| e.to_string())?;
        let mut block = attach_bytes(&raw, media_type);
        set_name(&mut block, name);
        return Ok(block);
    }
    if let Some(uri) = s("uri") {
        if !(uri.starts_with("http://") || uri.starts_with("https://") || uri.starts_with("gs://")) {
            return Err("only http(s) and gs URIs are fetched".into());
        }
        let mut block = attach(uri, s("mimeType"));
        set_name(&mut block, name);
        return Ok(block);
    }
    Err("a file part needs `bytes` or `uri`".into())
}

fn set_name(block: &mut ContentBlock, name: &str) {
    if name.is_empty() {
        return;
    }
    if let ContentBlock::Image(m) | ContentBlock::Audio(m) | ContentBlock::Video(m) | ContentBlock::Document(m) = block {
        m.name = name.to_string();
    }
}

/// The text of some parts, for whoever only wants to read it.
pub fn text_of(parts: Option<&Value>) -> String {
    let mut out: Vec<String> = Vec::new();
    for part in parts.and_then(Value::as_array).into_iter().flatten() {
        let Value::Object(part) = part else { continue };
        match kind_of(part).as_str() {
            "text" => {
                if let Some(t) = part.get("text").filter(|t| !matches!(t, Value::Null | Value::Bool(false)) && t.as_str() != Some("")) {
                    out.push(py_text(t));
                }
            }
            "data" => out.push(py_json(part.get("data").unwrap_or(&Value::Null))),
            _ => {}
        }
    }
    out.join("\n")
}

// ----------------------------------------------------------------------
// tasks
// ----------------------------------------------------------------------

/// A task status: its state, when, and what the agent said about it.
pub fn status(state: &str, text: &str, task_id: &str, context_id: &str) -> Value {
    let mut out = json!({"state": state, "timestamp": now()});
    if !text.is_empty() {
        out["message"] = text_message("agent", text, task_id, context_id);
    }
    out
}

pub fn status_event(task: &Value, last: bool) -> Value {
    json!({"kind": "status-update", "taskId": task["id"], "contextId": task["contextId"], "status": task["status"], "final": last})
}

pub fn artifact_event(task: &Value, artifact: &Value, append: bool, last: bool) -> Value {
    json!({"kind": "artifact-update", "taskId": task["id"], "contextId": task["contextId"], "artifact": artifact,
           "append": append, "lastChunk": last})
}

/// The task as it is sent: with as much history as was asked for.
pub fn trimmed(task: &Value, history_length: Option<&Value>) -> Value {
    let Some(n) = history_length.filter(|v| v.is_i64() || v.is_u64()).and_then(Value::as_i64) else { return task.clone() };
    let history = task.get("history").and_then(Value::as_array).cloned().unwrap_or_default();
    let kept = if n > 0 { history[history.len().saturating_sub(n as usize)..].to_vec() } else { vec![] };
    let mut out = task.clone();
    out["history"] = Value::Array(kept);
    out
}

/// What a run produced, as A2A artefacts: the answer first, then its files.
pub fn artifacts_of(result: &RunResult, answer_id: &str) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    if !result.output.is_empty() {
        parts.push(text_part(&result.output));
    }
    if !result.data.is_null() {
        let data = if result.data.is_object() { result.data.clone() } else { json!({"value": result.data}) };
        parts.push(json!({"kind": "data", "data": data}));
    }
    let mut out: Vec<Value> = Vec::new();
    if !parts.is_empty() {
        let id = if answer_id.is_empty() { new_uuid() } else { answer_id.to_string() };
        out.push(json!({"artifactId": id, "name": ANSWER, "parts": parts}));
    }
    out.extend(result.artifacts.iter().map(artifact));
    out
}

fn artifact(produced: &Artifact) -> Value {
    let content = &produced.content;
    let kind = if produced.media_type.is_empty() { "text/plain" } else { produced.media_type.as_str() };
    let mut entry = json!({"artifactId": new_uuid(), "name": produced.name});
    if content.len() > MAX_INLINE_BYTES {
        entry["parts"] = json!([text_part(&format!("[{}: too large to send inline]", produced.name))]);
        entry["metadata"] = json!({"truncated": true});
    } else if kind.starts_with("text/") && produced.path.is_none() {
        entry["parts"] = json!([text_part(content)]);
    } else {
        entry["parts"] = json!([{"kind": "file", "file": {"name": produced.name, "mimeType": kind, "bytes": B64.encode(content.as_bytes())}}]);
    }
    entry
}

/// What a task came to, as text — wherever the server put it.
pub fn answer_of(task: &Value) -> String {
    let artifacts: Vec<&Value> = task.get("artifacts").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
    let named: Vec<&Value> = artifacts.iter().copied().filter(|a| a.get("name").and_then(Value::as_str) == Some(ANSWER)).collect();
    let named = if named.is_empty() { artifacts } else { named };
    for a in named {
        let text = text_of(a.get("parts"));
        if !text.is_empty() {
            return text;
        }
    }
    let said = text_of(task.get("status").and_then(|s| s.get("message")).and_then(|m| m.get("parts")));
    if !said.is_empty() {
        return said;
    }
    for entry in task.get("history").and_then(Value::as_array).into_iter().flatten().rev() {
        if entry.get("role").and_then(Value::as_str) == Some("agent") {
            return text_of(entry.get("parts"));
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_and_parts() {
        let m = text_message("agent", "hi", "t1", "");
        assert_eq!(m["kind"], "message");
        assert_eq!(m["role"], "agent");
        assert_eq!(m["parts"], json!([{"kind": "text", "text": "hi"}]));
        assert_eq!(m["taskId"], "t1");
        assert!(m.get("contextId").is_none());
        assert_eq!(m["messageId"].as_str().unwrap().len(), 36);

        let note = B64.encode(b"hello");
        let (text, files) = read_parts(Some(&json!([
            {"kind": "text", "text": "Read these."},
            {"kind": "data", "data": {"order": 4182, "name": "Zoë"}},
            {"type": "file", "file": {"name": "note.txt", "mimeType": "text/plain", "bytes": note}},
        ])))
        .unwrap();
        assert_eq!(text, "Read these.\n\n{\"order\": 4182, \"name\": \"Zoë\"}");
        let (_, block) = files[0].as_media().unwrap();
        assert_eq!((block.name.as_str(), block.media_type.as_str()), ("note.txt", "text/plain"));
        assert_eq!(block.read().unwrap(), b"hello");

        let err = |v: Value| read_parts(Some(&v)).unwrap_err();
        assert_eq!(read_parts(None).unwrap_err().code, INVALID_PARAMS);
        assert_eq!(err(json!([])).code, INVALID_PARAMS);
        assert_eq!(err(json!([1])).message, "a part must be an object");
        let e = err(json!([{"kind": "hologram"}]));
        assert_eq!((e.code, e.message.as_str()), (CONTENT_TYPE_NOT_SUPPORTED, "a part of kind 'hologram' is not supported"));
        assert!(err(json!([{"kind": "file", "file": {"uri": "file:///etc/passwd"}}])).message.contains("only http(s) and gs URIs"));
        assert!(err(json!([{"kind": "file", "file": {}}])).message.contains("needs `bytes` or `uri`"));
        let (_, files) = read_parts(Some(&json!([{"kind": "file", "file": {"uri": "https://x.test/a.png", "name": "chart"}}]))).unwrap();
        assert_eq!(files[0].type_name(), "image");

        let parts = parts_of("Look.", &[attach_bytes(b"abc", "image/png")]);
        assert_eq!(parts[0], json!({"kind": "file", "file": {"mimeType": "image/png", "bytes": "YWJj"}}));
        assert_eq!(parts[1], text_part("Look."));
        assert_eq!(parts_of("", &[]), vec![text_part("")]);
        assert_eq!(text_of(Some(&json!([{"kind": "text", "text": "a"}, {"kind": "data", "data": [1]}, {"kind": "file"}]))), "a\n[1]");
    }

    #[test]
    fn tasks_and_answers() {
        let s = status("failed", "Boom.", "t", "c");
        assert_eq!(s["state"], "failed");
        assert!(s["timestamp"].as_str().unwrap().ends_with('Z'));
        assert_eq!(s["message"]["parts"][0]["text"], "Boom.");
        assert!(status("working", "", "", "").get("message").is_none());

        let task = json!({"kind": "task", "id": "t", "contextId": "c", "status": s, "history": [1, 2, 3], "artifacts": []});
        assert_eq!(status_event(&task, true)["final"], true);
        assert_eq!(artifact_event(&task, &json!({}), true, false)["lastChunk"], false);
        assert_eq!(trimmed(&task, Some(&json!(2)))["history"], json!([2, 3]));
        assert_eq!(trimmed(&task, Some(&json!(0)))["history"], json!([]));
        assert_eq!(trimmed(&task, Some(&json!(true)))["history"], json!([1, 2, 3]));
        assert_eq!(trimmed(&task, None)["history"], json!([1, 2, 3]));
        assert_eq!(answer_of(&task), "Boom.");

        let result = RunResult { output: "Done.".into(), data: json!(7), artifacts: vec![Artifact::new("notes.md", "# hi")], ..Default::default() };
        let arts = artifacts_of(&result, "ans");
        assert_eq!(arts[0]["artifactId"], "ans");
        assert_eq!(arts[0]["parts"], json!([{"kind": "text", "text": "Done."}, {"kind": "data", "data": {"value": 7}}]));
        assert_eq!(arts[1]["parts"], json!([{"kind": "text", "text": "# hi"}]));
        let mut pdf = Artifact::new("r.pdf", "x");
        pdf.media_type = "application/pdf".into();
        assert_eq!(artifacts_of(&RunResult { artifacts: vec![pdf], ..Default::default() }, "")[0]["parts"][0]["file"]["bytes"], "eA==");
        let done = json!({"artifacts": [{"name": "other", "parts": [text_part("x")]}, {"name": ANSWER, "parts": [text_part("Done.")]}]});
        assert_eq!(answer_of(&done), "Done.");
        assert_eq!(answer_of(&json!({"history": [{"role": "agent", "parts": [text_part("old")]}, {"role": "user", "parts": []}]})), "old");
    }

    #[test]
    fn errors_carry_codes() {
        let e = A2AError::http("busy", SERVER_BUSY, 429).with_retry_after(2.0).with_data(json!({"x": 1}));
        assert_eq!(e.body(), json!({"code": -32000, "message": "busy", "data": {"x": 1}}));
        assert_eq!(e.status, 429);
        let conflict = A2AError::from(Error::SessionConflict { message: "stale".into(), session_id: "s".into() });
        assert_eq!(conflict.code, INTERNAL_ERROR);
        assert_eq!(A2AError::from(Error::config("bad")).code, INVALID_PARAMS);
        assert_eq!(Error::from(A2AError::new("gone", INTERNAL_ERROR)).to_string(), "A2AError: gone");
        assert!(is_terminal("canceled") && !is_terminal("working"));
    }
}
