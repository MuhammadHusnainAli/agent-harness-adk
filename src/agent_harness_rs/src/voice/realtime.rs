//! Realtime voice: audio straight into a speech-to-speech model.
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::Agent, mic: futures::stream::BoxStream<'static, Vec<u8>>) -> agent_harness::Result<()> {
//! use agent_harness::voice::{RealtimeAgent, VoiceEventType};
//! use futures::StreamExt;
//! let voice = RealtimeAgent::new(agent, "openai")?;            // or "gemini"
//! let mut events = voice.run(mic);
//! while let Some(event) = events.next().await {
//!     if event.kind == VoiceEventType::Audio { /* speaker.write(&event.audio) */ }
//! }
//! # Ok(()) }
//! ```
//!
//! One model listens, thinks and speaks, over one open connection. There is no
//! transcription step and no synthesis step, so this is as fast as a voice
//! agent gets, and the model hears tone as well as words.
//!
//! It is still the same [`Agent`]. Its instructions become the model's; its
//! tools are offered to it, and each call the model makes goes through
//! `Agent::call_tool` — the guardrails, the hooks, the permission gate, the
//! audit trail and the budget, as in any run. Where the connection goes is put
//! to the same `model_egress` check as any model call.
//!
//! What cannot be the same: the model speaks before anyone can read what it
//! said. Output guardrails here are applied to the transcript afterwards and
//! recorded, not enforced in the moment. Where an answer must be checked
//! *before* it is heard, use [`VoiceAgent`](crate::voice::VoiceAgent).

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde_json::{Map, Value, json};
use tokio::sync::{Notify, mpsc};

use crate::agent::Agent;
use crate::context::close_open_tool_calls;
use crate::errors::{Error, Result};
use crate::runtime::Session;
use crate::tools::ToolSchema;
use crate::types::{Message, RunResult, Usage, new_id};
use crate::voice::audio::Resampler;
use crate::voice::pipeline::{VoiceEvent, VoiceEventType};
use crate::voice::ws::{WebSocket, WsMessage};

const INTERRUPTED: &str = " — [the user interrupted here; the rest was not heard]";

/// What a backend made of one server message.
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    Ready,
    SpeechStarted,
    SpeechStopped,
    Transcript(String),
    TranscriptDelta(String),
    Audio(Vec<u8>),
    Text(String),
    Tool { call_id: String, name: String, args: Map<String, Value> },
    Interrupted,
    /// Tokens for one response.
    Usage(Usage),
    /// The running total so far (Gemini reports it this way).
    UsageTotal(Usage),
    TurnEnd,
    Error(String),
}

/// One realtime wire protocol.
pub trait RealtimeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    /// The rate the model takes audio at, and the rate it speaks at.
    fn input_rate(&self) -> u32;
    fn output_rate(&self) -> u32;
    fn url(&self) -> Result<String>;
    fn headers(&self) -> Result<Vec<(String, String)>>;
    fn setup(&self, instructions: &str, tools: &[ToolSchema]) -> Vec<Value>;
    fn audio(&self, pcm: &[u8]) -> Value;
    fn text(&self, text: &str) -> Vec<Value>;
    fn tool_result(&self, call_id: &str, name: &str, output: &str) -> Vec<Value>;
    fn cancel(&self) -> Vec<Value>;
    fn parse(&self, message: &Value) -> Vec<Signal>;
}

fn b64(pcm: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(pcm)
}

fn unb64(s: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(s).unwrap_or_default()
}

fn int(v: &Value) -> u64 {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).unwrap_or(0)
}

/// OpenAI's Realtime API, over its WebSocket.
///
/// `turn_detection`: `semantic_vad` waits for the *thought* to finish, not just
/// the sound; `server_vad` ends a turn on silence, and takes `silence_ms`,
/// `threshold` and `prefix_ms`; `None` leaves turns to you. `transcription`
/// is the model that writes down what the person said (`None`: off).
#[derive(Debug, Clone)]
pub struct OpenAIRealtime {
    pub model: String,
    pub voice: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub turn_detection: Option<String>,
    pub silence_ms: Option<u32>,
    pub threshold: Option<f64>,
    pub prefix_ms: Option<u32>,
    pub transcription: Option<String>,
    pub language: Option<String>,
}

impl Default for OpenAIRealtime {
    fn default() -> Self {
        Self {
            model: "gpt-realtime".into(),
            voice: "marin".into(),
            api_key: None,
            base_url: "wss://api.openai.com/v1/realtime".into(),
            turn_detection: Some("semantic_vad".into()),
            silence_ms: None,
            threshold: None,
            prefix_ms: None,
            transcription: Some("gpt-4o-mini-transcribe".into()),
            language: None,
        }
    }
}

impl RealtimeBackend for OpenAIRealtime {
    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// The API takes and returns 24 kHz PCM.
    fn input_rate(&self) -> u32 {
        24_000
    }

    fn output_rate(&self) -> u32 {
        24_000
    }

    fn url(&self) -> Result<String> {
        Ok(format!("{}?model={}", self.base_url, self.model))
    }

    fn headers(&self) -> Result<Vec<(String, String)>> {
        let key = self
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| Error::config("the OpenAI realtime API needs a key — set OPENAI_API_KEY, or pass api_key"))?;
        Ok(vec![("Authorization".into(), format!("Bearer {key}"))])
    }

    fn setup(&self, instructions: &str, tools: &[ToolSchema]) -> Vec<Value> {
        let mut listening = json!({"format": {"type": "audio/pcm", "rate": self.input_rate()}});
        listening["turn_detection"] = match &self.turn_detection {
            Some(kind) => {
                let mut d = json!({"type": kind, "create_response": true, "interrupt_response": true});
                if kind == "server_vad" {
                    if let Some(v) = self.silence_ms {
                        d["silence_duration_ms"] = json!(v);
                    }
                    if let Some(v) = self.threshold {
                        d["threshold"] = json!(v);
                    }
                    if let Some(v) = self.prefix_ms {
                        d["prefix_padding_ms"] = json!(v);
                    }
                }
                d
            }
            None => Value::Null,
        };
        if let Some(model) = &self.transcription {
            let mut t = json!({"model": model});
            if let Some(l) = &self.language {
                t["language"] = json!(l);
            }
            listening["transcription"] = t;
        }
        let mut session = json!({
            "type": "realtime", "model": self.model, "output_modalities": ["audio"],
            "audio": {"input": listening, "output": {"format": {"type": "audio/pcm", "rate": self.output_rate()}, "voice": self.voice}},
            "instructions": instructions,
        });
        if !tools.is_empty() {
            session["tools"] = tools.iter().map(|t| json!({"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters})).collect();
            session["tool_choice"] = json!("auto");
        }
        vec![json!({"type": "session.update", "session": session})]
    }

    fn audio(&self, pcm: &[u8]) -> Value {
        json!({"type": "input_audio_buffer.append", "audio": b64(pcm)})
    }

    fn text(&self, text: &str) -> Vec<Value> {
        vec![
            json!({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}),
            json!({"type": "response.create"}),
        ]
    }

    fn tool_result(&self, call_id: &str, _name: &str, output: &str) -> Vec<Value> {
        vec![json!({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call_id, "output": output}}), json!({"type": "response.create"})]
    }

    fn cancel(&self) -> Vec<Value> {
        vec![json!({"type": "response.cancel"})]
    }

    fn parse(&self, m: &Value) -> Vec<Signal> {
        let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        match m.get("type").and_then(Value::as_str).unwrap_or("") {
            "session.created" | "session.updated" => vec![Signal::Ready],
            "input_audio_buffer.speech_started" => vec![Signal::SpeechStarted],
            "input_audio_buffer.speech_stopped" => vec![Signal::SpeechStopped],
            "conversation.item.input_audio_transcription.completed" => vec![Signal::Transcript(s("transcript"))],
            "response.output_audio.delta" | "response.audio.delta" => vec![Signal::Audio(unb64(&s("delta")))],
            "response.output_audio_transcript.delta" | "response.audio_transcript.delta" | "response.output_text.delta" | "response.text.delta" => vec![Signal::Text(s("delta"))],
            "response.function_call_arguments.done" => {
                let args = serde_json::from_str::<Value>(m.get("arguments").and_then(Value::as_str).filter(|a| !a.is_empty()).unwrap_or("{}"))
                    .ok()
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default();
                vec![Signal::Tool { call_id: s("call_id"), name: s("name"), args }]
            }
            "response.done" => {
                let response = m.get("response").cloned().unwrap_or(Value::Null);
                let used = &response["usage"];
                let usage = Usage { input_tokens: int(&used["input_tokens"]), output_tokens: int(&used["output_tokens"]), calls: 1, ..Default::default() };
                let mut out = vec![];
                if response["status"] == "cancelled" {
                    out.push(Signal::Interrupted);
                }
                // A response that only called a function is not the end of the turn.
                let calls = response["output"].as_array().is_some_and(|items| items.iter().any(|i| i["type"] == "function_call"));
                out.push(Signal::Usage(usage));
                if !calls {
                    out.push(Signal::TurnEnd);
                }
                out
            }
            "error" => {
                let error = m.get("error").cloned().unwrap_or(Value::Null);
                vec![Signal::Error(error["message"].as_str().map_or_else(|| crate::pyjson::dumps(&error), str::to_string))]
            }
            _ => vec![],
        }
    }
}

/// Google's Gemini Live API, over its WebSocket.
#[derive(Debug, Clone)]
pub struct GeminiLive {
    pub model: String,
    /// A prebuilt voice name; `None` is the model's default.
    pub voice: Option<String>,
    /// Defaults to `GEMINI_API_KEY`, then `GOOGLE_API_KEY`.
    pub api_key: Option<String>,
    pub base_url: String,
    /// How long a pause ends a turn; `None` is the model's own.
    pub silence_ms: Option<u32>,
    /// `false` turns the model's own turn detection off.
    pub detect_speech: bool,
    pub language: Option<String>,
}

impl Default for GeminiLive {
    fn default() -> Self {
        Self {
            model: "gemini-3.8-live".into(),
            voice: None,
            api_key: None,
            base_url: "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent".into(),
            silence_ms: None,
            detect_speech: true,
            language: None,
        }
    }
}

impl RealtimeBackend for GeminiLive {
    fn name(&self) -> &str {
        "gemini"
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Live listens at 16 kHz and speaks at 24 kHz.
    fn input_rate(&self) -> u32 {
        16_000
    }

    fn output_rate(&self) -> u32 {
        24_000
    }

    fn url(&self) -> Result<String> {
        let key = self
            .api_key
            .clone()
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| Error::config("the Gemini Live API needs a key — set GEMINI_API_KEY, or pass api_key"))?;
        Ok(format!("{}?key={key}", self.base_url))
    }

    fn headers(&self) -> Result<Vec<(String, String)>> {
        Ok(vec![])
    }

    fn setup(&self, instructions: &str, tools: &[ToolSchema]) -> Vec<Value> {
        let mut generation = json!({"responseModalities": ["AUDIO"]});
        let mut speech = Map::new();
        if let Some(v) = &self.voice {
            speech.insert("voiceConfig".into(), json!({"prebuiltVoiceConfig": {"voiceName": v}}));
        }
        if let Some(l) = &self.language {
            speech.insert("languageCode".into(), json!(l));
        }
        if !speech.is_empty() {
            generation["speechConfig"] = Value::Object(speech);
        }
        let model = if self.model.starts_with("models/") { self.model.clone() } else { format!("models/{}", self.model) };
        let mut setup = json!({
            "model": model, "generationConfig": generation,
            "systemInstruction": {"parts": [{"text": instructions}]},
            "inputAudioTranscription": {}, "outputAudioTranscription": {},
        });
        let mut detection = Map::new();
        if !self.detect_speech {
            detection.insert("disabled".into(), json!(true));
        }
        if let Some(ms) = self.silence_ms {
            detection.insert("silenceDurationMs".into(), json!(ms));
        }
        if !detection.is_empty() {
            setup["realtimeInputConfig"] = json!({"automaticActivityDetection": detection});
        }
        if !tools.is_empty() {
            let declarations: Vec<Value> =
                tools.iter().map(|t| json!({"name": t.name, "description": t.description, "parameters": crate::llm_providers::gemini::clean_schema(&t.parameters)})).collect();
            setup["tools"] = json!([{"functionDeclarations": declarations}]);
        }
        vec![json!({"setup": setup})]
    }

    fn audio(&self, pcm: &[u8]) -> Value {
        json!({"realtimeInput": {"audio": {"data": b64(pcm), "mimeType": format!("audio/pcm;rate={}", self.input_rate())}}})
    }

    fn text(&self, text: &str) -> Vec<Value> {
        vec![json!({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": true}})]
    }

    fn tool_result(&self, call_id: &str, name: &str, output: &str) -> Vec<Value> {
        vec![json!({"toolResponse": {"functionResponses": [{"id": call_id, "name": name, "response": {"result": output}}]}})]
    }

    fn cancel(&self) -> Vec<Value> {
        vec![] // the server stops by itself when it hears the person
    }

    fn parse(&self, m: &Value) -> Vec<Signal> {
        let mut out = vec![];
        if m.get("setupComplete").is_some() {
            out.push(Signal::Ready);
        }
        let content = m.get("serverContent").cloned().unwrap_or(Value::Null);
        if let Some(heard) = content["inputTranscription"]["text"].as_str().filter(|t| !t.is_empty()) {
            out.push(Signal::TranscriptDelta(heard.into()));
        }
        for part in content["modelTurn"]["parts"].as_array().into_iter().flatten() {
            if let Some(data) = part["inlineData"]["data"].as_str().filter(|d| !d.is_empty()) {
                out.push(Signal::Audio(unb64(data)));
            }
        }
        if let Some(said) = content["outputTranscription"]["text"].as_str().filter(|t| !t.is_empty()) {
            out.push(Signal::Text(said.into()));
        }
        if content["interrupted"].as_bool().unwrap_or(false) {
            out.push(Signal::Interrupted);
        }
        for call in m["toolCall"]["functionCalls"].as_array().into_iter().flatten() {
            out.push(Signal::Tool {
                call_id: call["id"].as_str().unwrap_or("").into(),
                name: call["name"].as_str().unwrap_or("").into(),
                args: call["args"].as_object().cloned().unwrap_or_default(),
            });
        }
        if let Some(used) = m.get("usageMetadata").filter(|u| u.is_object()) {
            out.push(Signal::UsageTotal(Usage { input_tokens: int(&used["promptTokenCount"]), output_tokens: int(&used["responseTokenCount"]), calls: 1, ..Default::default() }));
        }
        if content["turnComplete"].as_bool().unwrap_or(false) {
            out.push(Signal::TurnEnd);
        }
        if m.get("goAway").is_some() {
            out.push(Signal::Error("the server is about to close this connection".into()));
        }
        out
    }
}

/// An open realtime connection: text frames each way.
#[async_trait]
pub trait RealtimeSocket: Send + Sync {
    async fn send(&self, text: &str) -> Result<()>;
    /// The next text message, `None` once the connection has closed.
    async fn recv(&self) -> Option<String>;
    async fn close(&self);
}

#[async_trait]
impl RealtimeSocket for WebSocket {
    async fn send(&self, text: &str) -> Result<()> {
        self.send_text(text).await
    }

    async fn recv(&self) -> Option<String> {
        match WebSocket::recv(self).await {
            Ok(WsMessage::Text(t)) => Some(t),
            Ok(WsMessage::Binary(b)) => Some(String::from_utf8_lossy(&b).into_owned()),
            Err(_) => None,
        }
    }

    async fn close(&self) {
        WebSocket::close(self, 1000).await;
    }
}

/// How to open the socket — yours, for a proxy or a test.
pub type Connect = Arc<dyn Fn(String, Vec<(String, String)>) -> BoxFuture<'static, Result<Arc<dyn RealtimeSocket>>> + Send + Sync>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[derive(Default)]
struct RtState {
    history: Vec<Message>,
    usage: Usage,
    turns: Vec<Map<String, Value>>,
    session: Option<Session>,
    saved: usize,
    socket: Option<Arc<dyn RealtimeSocket>>,
}

#[derive(Default)]
struct Turn {
    heard: String,
    said: String,
    speaking: bool,
    busy: usize,
    began: Option<Instant>,
    first_audio: Option<Instant>,
    interrupted: bool,
    total: Usage,
}

struct Inner {
    agent: Agent,
    backend: Arc<dyn RealtimeBackend>,
    rate: u32,
    output_rate: u32,
    linger: f64,
    connect: Connect,
    session_id: Option<String>,
    state: Mutex<RtState>,
}

/// An agent you talk to, through a speech-to-speech model. Cheap to clone.
#[derive(Clone)]
pub struct RealtimeAgent {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for RealtimeAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<RealtimeAgent {} via {}>", self.inner.agent.name(), self.inner.backend.name())
    }
}

/// The realtime backends, by name.
pub fn realtime_backend(name: &str) -> Result<Arc<dyn RealtimeBackend>> {
    match name {
        "openai" => Ok(Arc::new(OpenAIRealtime::default())),
        "gemini" => Ok(Arc::new(GeminiLive::default())),
        other => Err(Error::config(format!("unknown realtime provider '{other}'; known: gemini, openai"))),
    }
}

impl RealtimeAgent {
    /// Over a backend by name: `"openai"` or `"gemini"`, with its defaults.
    pub fn new(agent: Agent, provider: &str) -> Result<Self> {
        Ok(Self::with_backend(agent, realtime_backend(provider)?))
    }

    /// Over a backend you built — `OpenAIRealtime { voice: "cedar".into(), ..Default::default() }`.
    pub fn with_backend(agent: Agent, backend: Arc<dyn RealtimeBackend>) -> Self {
        let connect: Connect = Arc::new(|url, headers| {
            Box::pin(async move {
                let ws = WebSocket::connect(&url, &headers, 20.0).await?;
                Ok(Arc::new(ws) as Arc<dyn RealtimeSocket>)
            })
        });
        let (rate, output_rate) = (backend.input_rate(), backend.output_rate());
        Self { inner: Arc::new(Inner { agent, backend, rate, output_rate, linger: 20.0, connect, session_id: None, state: Mutex::default() }) }
    }

    fn rebuild(self, f: impl FnOnce(&mut Inner)) -> Self {
        let mut inner = Arc::try_unwrap(self.inner).unwrap_or_else(|arc| Inner {
            agent: arc.agent.clone(),
            backend: arc.backend.clone(),
            rate: arc.rate,
            output_rate: arc.output_rate,
            linger: arc.linger,
            connect: arc.connect.clone(),
            session_id: arc.session_id.clone(),
            state: Mutex::default(),
        });
        f(&mut inner);
        Self { inner: Arc::new(inner) }
    }

    /// The sample rate of the audio you feed in; it is converted to the model's.
    pub fn rate(self, rate: u32) -> Self {
        self.rebuild(|i| i.rate = rate)
    }

    /// The rate you want back, if not the model's.
    pub fn output_rate(self, rate: u32) -> Self {
        self.rebuild(|i| i.output_rate = rate)
    }

    /// A session id whose transcript this carries on from.
    pub fn session(self, id: &str) -> Self {
        let id = id.to_string();
        self.rebuild(|i| i.session_id = Some(id))
    }

    /// Seconds to wait, once your audio has ended, for an answer still being spoken.
    pub fn linger(self, seconds: f64) -> Self {
        self.rebuild(|i| i.linger = seconds)
    }

    pub fn connect(self, connect: Connect) -> Self {
        self.rebuild(|i| i.connect = connect)
    }

    pub fn history(&self) -> Vec<Message> {
        lock(&self.inner.state).history.clone()
    }

    pub fn usage(&self) -> Usage {
        lock(&self.inner.state).usage.clone()
    }

    pub fn turns(&self) -> Vec<Map<String, Value>> {
        lock(&self.inner.state).turns.clone()
    }

    /// The rate of the audio it speaks, in Hz.
    pub fn speaking_rate(&self) -> u32 {
        self.inner.output_rate
    }

    pub fn session_id(&self) -> String {
        let s = lock(&self.inner.state);
        s.session.as_ref().map(|s| s.id.clone()).or_else(|| self.inner.session_id.clone()).unwrap_or_default()
    }

    async fn open(&self) -> Result<()> {
        if lock(&self.inner.state).session.is_some() {
            return Ok(());
        }
        let me = &self.inner;
        let session = match &me.session_id {
            Some(id) => {
                let loaded = me.agent.harness().sessions.load(id).await?;
                let mut s = lock(&me.state);
                s.history = close_open_tool_calls(loaded.messages.clone());
                s.saved = s.history.len();
                loaded
            }
            None => {
                let mut s = Session::new(me.agent.name());
                if let Some(t) = me.agent.trace() {
                    s.user_id = t.user_id.clone();
                    s.tenant_id = t.tenant_id.clone();
                }
                s
            }
        };
        lock(&me.state).session = Some(session);
        Ok(())
    }

    async fn save(&self) -> Result<()> {
        self.open().await?;
        let me = &self.inner;
        let (mut session, added) = {
            let s = lock(&me.state);
            let added = s.history[s.saved.min(s.history.len())..].to_vec();
            if added.is_empty() {
                return Ok(());
            }
            let mut session = s.session.clone().expect("opened above");
            session.messages = s.history.clone();
            (session, added)
        };
        if session.title.is_empty()
            && let Some(first) = added.first()
        {
            session.title = first.text().split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect();
        }
        let mut carrier = RunResult { agent: me.agent.name().into(), session_id: session.id.clone(), ..Default::default() };
        me.agent.save_session_record(&mut session, &mut carrier, &added).await;
        let mut s = lock(&me.state);
        s.history = session.messages.clone();
        s.saved = s.history.len();
        s.session = Some(session);
        Ok(())
    }

    /// The agent's prompt — and, when picking a conversation back up, what was
    /// said in it, since a realtime model starts with no memory of an earlier call.
    fn instructions(&self) -> String {
        let names: Vec<String> = self.inner.agent.tools().iter().filter(|t| !t.tags.contains("handoff")).map(|t| t.name.clone()).collect();
        let system = self.inner.agent.system_prompt(&names);
        let history = self.history();
        if history.is_empty() {
            return system;
        }
        let skip = history.len().saturating_sub(40);
        let lines: Vec<String> = history.iter().skip(skip).filter_map(|m| {
            let t = m.text();
            (!t.is_empty()).then(|| format!("{}: {t}", m.role.as_str()))
        }).collect();
        format!("{system}\n\n## The conversation so far\n{}", lines.join("\n"))
    }

    /// Say something to the model in writing, mid-conversation.
    pub async fn send_text(&self, text: &str) -> Result<()> {
        let socket = lock(&self.inner.state).socket.clone().ok_or_else(|| Error::config("there is no open conversation to send text to"))?;
        for m in self.inner.backend.text(text) {
            socket.send(&m.to_string()).await?;
        }
        Ok(())
    }

    /// Stop the answer that is being spoken.
    pub async fn interrupt(&self) -> Result<()> {
        let socket = lock(&self.inner.state).socket.clone();
        if let Some(s) = socket {
            for m in self.inner.backend.cancel() {
                s.send(&m.to_string()).await?;
            }
        }
        Ok(())
    }

    /// A whole conversation: feed it the microphone, play what comes back.
    pub fn run(&self, audio: BoxStream<'static, Vec<u8>>) -> BoxStream<'static, VoiceEvent> {
        let me = self.clone();
        Box::pin(async_stream::stream! {
            match me.start().await {
                Err(e) => yield VoiceEvent { text: format!("{}: {e}", e.kind_name()), ..VoiceEvent::new(VoiceEventType::Error) },
                Ok((socket, run_id)) => {
                    let mut events = me.converse(socket, run_id, audio);
                    while let Some(e) = events.next().await {
                        yield e;
                    }
                }
            }
        })
    }

    async fn start(&self) -> Result<(Arc<dyn RealtimeSocket>, String)> {
        let me = &self.inner;
        let (agent, backend, harness) = (&me.agent, &me.backend, me.agent.harness());
        self.open().await?;
        let run_id = new_id("run");
        let instructions = self.instructions();
        // Where this connection goes is somebody's business: the same check a
        // model call gets, so a residency policy covers a voice call too.
        let data: Map<String, Value> = [
            ("provider", json!(backend.name())),
            ("model", json!(backend.model())),
            ("purpose", json!("realtime")),
            ("request", json!({"model": backend.model(), "system": instructions})),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let egress = agent.hook_engine().emit("model_egress", agent.name(), &run_id, 0, data).await;
        if egress.blocked {
            let detail: Map<String, Value> = [("reason".to_string(), json!(egress.reason))].into_iter().collect();
            harness.audit.record(agent.name(), "model_egress", backend.model(), "deny", &run_id, detail);
            return Err(Error::PermissionDenied {
                message: format!("{} may not open a realtime session with {}: {}", agent.name(), backend.model(), egress.reason),
                tool: backend.model().into(),
                reason: egress.reason,
            });
        }
        harness.control.check(&format!("{} realtime", agent.name()))?;
        let socket = (me.connect)(backend.url()?, backend.headers()?).await?;
        lock(&me.state).socket = Some(socket.clone());
        harness.control.enter(agent.name(), &run_id);
        let detail: Map<String, Value> = [("model".to_string(), json!(backend.model()))].into_iter().collect();
        harness.audit.record(agent.name(), "run_start", "realtime voice", "ok", &run_id, detail);
        let tools: Vec<ToolSchema> = agent.tools().iter().filter(|t| !t.tags.contains("handoff")).map(|t| t.to_schema()).collect();
        for m in backend.setup(&instructions, &tools) {
            socket.send(&m.to_string()).await?;
        }
        Ok((socket, run_id))
    }

    fn converse(&self, socket: Arc<dyn RealtimeSocket>, run_id: String, audio: BoxStream<'static, Vec<u8>>) -> BoxStream<'static, VoiceEvent> {
        let me = self.clone();
        Box::pin(async_stream::stream! {
            let (out_tx, mut out) = mpsc::unbounded_channel::<VoiceEvent>();
            let turn = Arc::new(Mutex::new(Turn::default()));
            let idle = Arc::new(Notify::new());

            let receiver = {
                let (me, socket, out_tx, turn, idle, run_id) = (me.clone(), socket.clone(), out_tx.clone(), turn.clone(), idle.clone(), run_id.clone());
                tokio::spawn(async move { me.receive(socket, out_tx, turn, idle, run_id).await })
            };
            let pump = {
                let (me, socket, turn, idle) = (me.clone(), socket.clone(), turn.clone(), idle.clone());
                let mut audio = audio;
                tokio::spawn(async move {
                    let inner = &me.inner;
                    let Ok(mut resampler) = Resampler::new(inner.rate, inner.backend.input_rate()) else { return };
                    while let Some(chunk) = audio.next().await {
                        if inner.agent.harness().control.check(&format!("{} realtime", inner.agent.name())).is_err() {
                            break;
                        }
                        if socket.send(&inner.backend.audio(&resampler.feed(&chunk)).to_string()).await.is_err() {
                            return;
                        }
                    }
                    // The microphone has closed. Let an answer in flight finish.
                    let _ = tokio::time::timeout(Duration::from_secs_f64(inner.linger), settled(&turn, &idle)).await;
                    socket.close().await;
                })
            };
            drop(out_tx);
            while let Some(e) = out.recv().await {
                yield e;
            }
            pump.abort();
            receiver.abort();
            socket.close().await;
            let inner = &me.inner;
            lock(&inner.state).socket = None;
            let harness = inner.agent.harness();
            harness.control.leave(&run_id);
            let (turns, tokens) = {
                let s = lock(&inner.state);
                (s.turns.len(), s.usage.total_tokens())
            };
            let detail: Map<String, Value> = [("turns", json!(turns)), ("tokens", json!(tokens))].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            harness.audit.record(inner.agent.name(), "run_end", "realtime voice", "ok", &run_id, detail);
            let _ = me.save().await;
        })
    }

    async fn receive(&self, socket: Arc<dyn RealtimeSocket>, out: mpsc::UnboundedSender<VoiceEvent>, turn: Arc<Mutex<Turn>>, idle: Arc<Notify>, run_id: String) {
        let inner = &self.inner;
        let (agent, backend, harness) = (&inner.agent, &inner.backend, inner.agent.harness());
        let guard = harness.guard();
        let Ok(mut resampler) = Resampler::new(backend.output_rate(), inner.output_rate) else { return };
        let send_event = |e: VoiceEvent| {
            let _ = out.send(e);
        };
        while let Some(raw) = socket.recv().await {
            let Ok(message) = serde_json::from_str::<Value>(&raw) else { continue };
            for signal in backend.parse(&message) {
                match signal {
                    Signal::Ready => {}
                    Signal::Audio(pcm) => {
                        {
                            let mut t = lock(&turn);
                            t.first_audio.get_or_insert_with(Instant::now);
                            t.speaking = true;
                        }
                        let mut e = VoiceEvent { audio: resampler.feed(&pcm), ..VoiceEvent::new(VoiceEventType::Audio) };
                        e.data.insert("rate".into(), json!(inner.output_rate));
                        send_event(e);
                    }
                    Signal::Text(t) => {
                        lock(&turn).said.push_str(&t);
                        send_event(VoiceEvent { text: t, ..VoiceEvent::new(VoiceEventType::Text) });
                    }
                    Signal::Transcript(t) => {
                        lock(&turn).heard = t.clone();
                        send_event(VoiceEvent { text: t, ..VoiceEvent::new(VoiceEventType::Transcript) });
                    }
                    Signal::TranscriptDelta(t) => {
                        lock(&turn).heard.push_str(&t);
                        let mut e = VoiceEvent { text: t, ..VoiceEvent::new(VoiceEventType::Transcript) };
                        e.data.insert("partial".into(), json!(true));
                        send_event(e);
                    }
                    Signal::SpeechStarted => {
                        send_event(VoiceEvent::new(VoiceEventType::SpeechStarted));
                        let said = {
                            let mut t = lock(&turn);
                            if t.speaking {
                                t.interrupted = true;
                                Some(t.said.clone())
                            } else {
                                None
                            }
                        };
                        if let Some(said) = said {
                            send_event(VoiceEvent { text: said, ..VoiceEvent::new(VoiceEventType::Interrupted) });
                        }
                    }
                    Signal::SpeechStopped => {
                        {
                            let mut t = lock(&turn);
                            t.began = Some(Instant::now());
                            t.first_audio = None;
                        }
                        send_event(VoiceEvent::new(VoiceEventType::SpeechStopped));
                    }
                    Signal::Interrupted => {
                        let said = {
                            let mut t = lock(&turn);
                            (!t.interrupted).then(|| {
                                t.interrupted = true;
                                t.said.clone()
                            })
                        };
                        if let Some(said) = said {
                            send_event(VoiceEvent { text: said, ..VoiceEvent::new(VoiceEventType::Interrupted) });
                        }
                    }
                    Signal::Tool { call_id, name, args } => {
                        lock(&turn).busy += 1;
                        let (agent, backend, socket, out, turn, idle) = (agent.clone(), backend.clone(), socket.clone(), out.clone(), turn.clone(), idle.clone());
                        tokio::spawn(async move {
                            let mut e = VoiceEvent { text: name.clone(), ..VoiceEvent::new(VoiceEventType::Tool) };
                            e.data.insert("args".into(), Value::Object(args.clone()));
                            let _ = out.send(e);
                            let outcome = agent.call_tool(&name, args).await;
                            for m in backend.tool_result(&call_id, &name, &outcome.content) {
                                let _ = socket.send(&m.to_string()).await;
                            }
                            lock(&turn).busy -= 1;
                            idle.notify_waiters();
                        });
                    }
                    Signal::Usage(usage) => self.count(&usage, &guard, &out),
                    Signal::UsageTotal(total) => {
                        // Gemini reports the running total, not the step.
                        let step = {
                            let mut t = lock(&turn);
                            let seen = std::mem::replace(&mut t.total, total.clone());
                            Usage { input_tokens: total.input_tokens.saturating_sub(seen.input_tokens), output_tokens: total.output_tokens.saturating_sub(seen.output_tokens), calls: 1, ..Default::default() }
                        };
                        self.count(&step, &guard, &out);
                    }
                    Signal::TurnEnd => {
                        self.end_turn(&turn, &out, &run_id).await;
                        idle.notify_waiters();
                    }
                    Signal::Error(text) => send_event(VoiceEvent { text, ..VoiceEvent::new(VoiceEventType::Error) }),
                }
            }
        }
        idle.notify_waiters();
    }

    fn count(&self, usage: &Usage, guard: &crate::runtime::BudgetGuard, out: &mpsc::UnboundedSender<VoiceEvent>) {
        let inner = &self.inner;
        {
            let mut s = lock(&inner.state);
            s.usage += usage;
            if let Some(session) = s.session.as_mut() {
                session.usage += usage;
            }
        }
        if let Err(Error::BudgetExceeded { kind, .. }) = guard.record(usage, inner.agent.name(), "realtime") {
            inner.agent.harness().audit.record(inner.agent.name(), "budget", &kind, "stop", "", Map::new());
            let mut e = VoiceEvent { text: format!("budget: {kind}"), ..VoiceEvent::new(VoiceEventType::Error) };
            e.data.insert("budget_exceeded".into(), json!(kind));
            let _ = out.send(e);
        }
    }

    async fn end_turn(&self, turn: &Mutex<Turn>, out: &mpsc::UnboundedSender<VoiceEvent>, run_id: &str) {
        let inner = &self.inner;
        let harness = inner.agent.harness();
        let (mut heard, mut said, interrupted, began, first_audio) = {
            let t = lock(turn);
            (t.heard.trim().to_string(), t.said.trim().to_string(), t.interrupted, t.began, t.first_audio)
        };
        if !heard.is_empty() || !said.is_empty() {
            // What was actually said, checked now that there is text to check.
            let checked = (|| -> Result<(String, String)> {
                let h = if heard.is_empty() { heard.clone() } else { harness.guardrails.check(&heard, "input", "voice")? };
                let s = if said.is_empty() { said.clone() } else { inner.agent.content_guardrails().check(&said, "output", inner.agent.name())? };
                Ok((h, s))
            })();
            match checked {
                Ok((h, s)) => {
                    heard = h;
                    said = s;
                }
                Err(e) => {
                    let detail: Map<String, Value> = [("reason".to_string(), json!(e.to_string().chars().take(300).collect::<String>()))].into_iter().collect();
                    harness.audit.record(inner.agent.name(), "guardrail", "realtime", "warn", run_id, detail);
                    let _ = out.send(VoiceEvent { text: format!("guardrail: {e}"), ..VoiceEvent::new(VoiceEventType::Error) });
                }
            }
            let mut s = lock(&inner.state);
            if !heard.is_empty() {
                s.history.push(Message::user(heard.clone()));
            }
            if !said.is_empty() {
                s.history.push(Message::assistant(format!("{said}{}", if interrupted { INTERRUPTED } else { "" })));
            }
        }
        let first_audio_ms = match (first_audio, began) {
            (Some(f), Some(b)) if f >= b => json!(((f - b).as_secs_f64() * 10_000.0).round() / 10.0),
            _ => Value::Null,
        };
        let metrics: Map<String, Value> =
            [("first_audio_ms", first_audio_ms.clone()), ("interrupted", json!(interrupted)), ("chars", json!(said.chars().count()))].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        lock(&inner.state).turns.push(metrics.clone());
        if let Some(ms) = first_audio_ms.as_f64() {
            harness.health.record("voice.first_audio", "voice", ms, true, "");
        }
        let _ = out.send(VoiceEvent { text: said, data: metrics, ..VoiceEvent::new(VoiceEventType::TurnEnd) });
        {
            let mut t = lock(turn);
            t.heard.clear();
            t.said.clear();
            t.speaking = false;
            t.began = None;
            t.first_audio = None;
            t.interrupted = false;
        }
        let _ = self.save().await;
    }
}

/// Wait until nothing is being said and no tool is running — and it stays that
/// way for a moment, since a tool's answer starts a new response.
async fn settled(turn: &Mutex<Turn>, idle: &Notify) {
    loop {
        let quiet = {
            let t = lock(turn);
            t.busy == 0 && !t.speaking && t.said.is_empty() && t.began.is_none()
        };
        if quiet {
            tokio::time::sleep(Duration::from_millis(300)).await;
            let t = lock(turn);
            if t.busy == 0 && !t.speaking && t.said.is_empty() {
                return;
            }
            continue;
        }
        let _ = tokio::time::timeout(Duration::from_millis(200), idle.notified()).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::FakeProvider;
    use crate::tools::Tool;
    use schemars::JsonSchema;
    use serde::Deserialize;

    /// A scripted realtime model: answers each `response.create`-worthy event
    /// with the next canned batch of server messages.
    struct FakeSocket {
        sent: Mutex<Vec<Value>>,
        inbox: tokio::sync::Mutex<mpsc::UnboundedReceiver<String>>,
        outbox: Mutex<Option<mpsc::UnboundedSender<String>>>,
        script: Mutex<Vec<Vec<Value>>>,
    }

    #[async_trait]
    impl RealtimeSocket for FakeSocket {
        async fn send(&self, text: &str) -> Result<()> {
            let v: Value = serde_json::from_str(text).unwrap();
            let kind = v["type"].as_str().unwrap_or("").to_string();
            self.sent.lock().unwrap().push(v);
            // Audio in, or a tool result: the model answers with its next batch.
            let answer = kind == "input_audio_buffer.append" || kind == "response.create";
            if answer {
                let next = {
                    let mut s = self.script.lock().unwrap();
                    if s.is_empty() { None } else { Some(s.remove(0)) }
                };
                let outbox = self.outbox.lock().unwrap().clone();
                for (m, out) in next.into_iter().flatten().zip(std::iter::repeat(outbox)) {
                    if let Some(out) = out {
                        let _ = out.send(m.to_string());
                    }
                }
            }
            Ok(())
        }

        async fn recv(&self) -> Option<String> {
            self.inbox.lock().await.recv().await
        }

        async fn close(&self) {
            // The model hangs up: what it already sent is still read, then nothing.
            self.outbox.lock().unwrap().take();
        }
    }

    /// Look up an order.
    #[derive(Deserialize, JsonSchema)]
    struct Lookup {
        order: String,
    }

    #[tokio::test]
    async fn a_realtime_call_runs_tools_under_the_rails_and_keeps_the_transcript() {
        let (outbox, inbox) = mpsc::unbounded_channel();
        let pcm = b64(&crate::voice::audio::tone(40.0, 24_000, 220.0, 9_000));
        let script = vec![
            vec![
                json!({"type": "session.updated"}),
                json!({"type": "input_audio_buffer.speech_started"}),
                json!({"type": "input_audio_buffer.speech_stopped"}),
                json!({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Where is order 7?"}),
                json!({"type": "response.function_call_arguments.done", "call_id": "c1", "name": "lookup", "arguments": "{\"order\": \"7\"}"}),
                json!({"type": "response.done", "response": {"status": "completed", "output": [{"type": "function_call"}], "usage": {"input_tokens": 10, "output_tokens": 2}}}),
            ],
            vec![
                json!({"type": "response.output_audio.delta", "delta": pcm}),
                json!({"type": "response.output_audio_transcript.delta", "delta": "Order 7 shipped."}),
                json!({"type": "response.done", "response": {"status": "completed", "output": [], "usage": {"input_tokens": 20, "output_tokens": 5}}}),
            ],
        ];
        let socket = Arc::new(FakeSocket { sent: Mutex::default(), inbox: tokio::sync::Mutex::new(inbox), outbox: Mutex::new(Some(outbox)), script: Mutex::new(script) });
        let held = socket.clone();
        let connect: Connect = Arc::new(move |_url, _headers| {
            let s = held.clone();
            Box::pin(async move { Ok(s as Arc<dyn RealtimeSocket>) })
        });
        let lookup = Tool::new("lookup", |a: Lookup, _| async move { Ok(format!("order {} shipped Tuesday", a.order)) });
        let agent = Agent::builder("phone").provider(Arc::new(FakeProvider::new(vec![]))).memory(false).instructions("Answer callers.").tool(lookup).build().unwrap();
        let backend = Arc::new(OpenAIRealtime { api_key: Some("k".into()), ..Default::default() });
        let voice = RealtimeAgent::with_backend(agent.clone(), backend).connect(connect).rate(16_000).linger(2.0);
        let mic = futures::stream::iter(vec![crate::voice::audio::voice_like(40.0)]).boxed();
        let events: Vec<VoiceEvent> = tokio::time::timeout(Duration::from_secs(20), voice.run(mic).collect()).await.expect("the call ends");
        let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
        assert!(kinds.contains(&"tool") && kinds.contains(&"audio") && kinds.contains(&"turn_end"), "{kinds:?}");
        let sent = socket.sent.lock().unwrap().clone();
        assert_eq!(sent[0]["type"], "session.update");
        assert!(sent[0]["session"]["instructions"].as_str().unwrap().contains("Answer callers."));
        assert_eq!(sent[0]["session"]["tools"][0]["name"], "lookup");
        let output = sent.iter().find(|m| m["item"]["type"] == "function_call_output").unwrap();
        assert_eq!(output["item"]["output"], "order 7 shipped Tuesday", "the tool ran through the agent");
        assert!(agent.harness().audit.entries().iter().any(|e| e.action == "tool_call" && e.target == "lookup"));
        let history = voice.history();
        assert_eq!(history.iter().map(|m| m.text()).collect::<Vec<_>>(), vec!["Where is order 7?", "Order 7 shipped."]);
        assert_eq!(voice.usage().output_tokens, 7);
        let saved = agent.harness().sessions.load(&voice.session_id()).await.unwrap();
        assert_eq!(saved.messages.len(), 2);
    }

    #[test]
    fn the_two_protocols() {
        let g = GeminiLive { api_key: Some("g".into()), voice: Some("Puck".into()), silence_ms: Some(400), ..Default::default() };
        let setup = &g.setup("Be brief.", &[])[0]["setup"];
        assert_eq!(setup["model"], "models/gemini-3.8-live");
        assert_eq!(setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"], "Puck");
        assert_eq!(setup["realtimeInputConfig"]["automaticActivityDetection"]["silenceDurationMs"], 400);
        assert!(g.url().unwrap().ends_with("?key=g"));
        let signals = g.parse(&json!({"serverContent": {"inputTranscription": {"text": "hi"}, "outputTranscription": {"text": "hello"}, "interrupted": true, "turnComplete": true}, "usageMetadata": {"promptTokenCount": 3, "responseTokenCount": 4}}));
        assert_eq!(signals[0], Signal::TranscriptDelta("hi".into()));
        assert!(signals.contains(&Signal::Interrupted) && signals.contains(&Signal::TurnEnd));
        let o = OpenAIRealtime { turn_detection: Some("server_vad".into()), silence_ms: Some(300), ..Default::default() };
        assert_eq!(o.setup("x", &[])[0]["session"]["audio"]["input"]["turn_detection"]["silence_duration_ms"], 300);
        assert_eq!(o.parse(&json!({"type": "response.done", "response": {"status": "cancelled", "output": []}}))[0], Signal::Interrupted);
        assert!(realtime_backend("nope").is_err());
    }
}
