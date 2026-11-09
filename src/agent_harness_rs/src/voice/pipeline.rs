//! A voice agent built from parts: listen, think, speak.
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::Agent, mic: futures::stream::BoxStream<'static, Vec<u8>>) -> agent_harness::Result<()> {
//! use agent_harness::voice::{OpenAISpeech, VoiceAgent, VoiceEventType};
//! use futures::StreamExt;
//! let voice = VoiceAgent::builder(agent).speech(std::sync::Arc::new(OpenAISpeech::new(None))).build()?;
//! let mut events = voice.run(mic);
//! while let Some(event) = events.next().await {
//!     match event.kind {
//!         VoiceEventType::Audio => { /* speaker.write(&event.audio) */ }
//!         VoiceEventType::Interrupted => { /* speaker.flush() */ }
//!         _ => {}
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! Speech is turned into text, the agent answers as it always does, and the
//! answer is spoken — so the agent behind the voice is an ordinary [`Agent`]:
//! any provider, its tools, its memory, its budget, its guardrails, its audit
//! trail. Nothing about the rails changes because the conversation is aloud.
//!
//! What makes it feel like talking to someone rather than to a queue:
//!
//! - **It starts speaking before it has finished thinking.** The answer is cut
//!   into sentences as the model writes them — the first at a clause — and
//!   each is synthesised while the next is being written.
//! - **It stops when you speak.** Speech over an answer cancels the rest of it,
//!   and the conversation remembers only what was actually said aloud.
//! - **It waits for you to finish.** A pause that turns out to be mid-sentence
//!   — you carry on before it has replied — is joined to what follows and
//!   answered once, not twice.
//!
//! Every turn reports how long each part took (`TurnEnd`), because latency
//! you cannot see is latency you cannot fix.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Map, Value, json};
use tokio::sync::{Semaphore, mpsc, watch};

use crate::agent::{Agent, RunOptions};
use crate::context::close_open_tool_calls;
use crate::errors::{Error, Result};
use crate::runtime::Session;
use crate::types::{Message, RunResult, StreamEventType, Usage};
use crate::voice::audio::{Resampler, duration_ms, wav};
use crate::voice::speech::{SpeechToText, TextToSpeech};
use crate::voice::text::SpeechChunker;
use crate::voice::vad::{EnergyVAD, VAD, VadEvent, VadKind};

const INTERRUPTED: &str = " — [the user interrupted here; the rest was not heard]";
/// A turn abandoned before any reply is kept to join the next — but not for ever.
const CARRY_MS: f64 = 30_000.0;

/// What happened in a spoken conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceEventType {
    /// The person began talking.
    SpeechStarted,
    /// The person finished talking (or it was a noise: `data["ignored"]`).
    SpeechStopped,
    /// What they said.
    Transcript,
    /// What the agent is writing.
    Text,
    /// PCM to play.
    Audio,
    /// The agent is using a tool.
    Tool,
    /// Stop playing: they spoke over the answer.
    Interrupted,
    /// The turn is over; `data` carries the timings.
    TurnEnd,
    Error,
}

impl VoiceEventType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SpeechStarted => "speech_started",
            Self::SpeechStopped => "speech_stopped",
            Self::Transcript => "transcript",
            Self::Text => "text",
            Self::Audio => "audio",
            Self::Tool => "tool",
            Self::Interrupted => "interrupted",
            Self::TurnEnd => "turn_end",
            Self::Error => "error",
        }
    }
}

/// One thing that happened in a spoken conversation.
#[derive(Clone, PartialEq)]
pub struct VoiceEvent {
    pub kind: VoiceEventType,
    pub text: String,
    pub audio: Vec<u8>,
    pub data: Map<String, Value>,
}

impl std::fmt::Debug for VoiceEvent {
    // Never the audio itself.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extra = if self.audio.is_empty() { String::new() } else { format!(" {} bytes", self.audio.len()) };
        write!(f, "<VoiceEvent {}{extra} {:?}>", self.kind.as_str(), self.text.chars().take(60).collect::<String>())
    }
}

impl VoiceEvent {
    pub fn new(kind: VoiceEventType) -> Self {
        Self { kind, text: String::new(), audio: vec![], data: Map::new() }
    }

    fn text(kind: VoiceEventType, text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Self::new(kind) }
    }

    fn with(mut self, data: Value) -> Self {
        if let Value::Object(m) = data {
            self.data.extend(m);
        }
        self
    }
}

#[derive(Default)]
struct VoiceState {
    /// The conversation as the agent will next see it: what was said, and of
    /// each answer only what was actually spoken.
    history: Vec<Message>,
    usage: Usage,
    turns: Vec<Map<String, Value>>,
    session: Option<Session>,
    holder: String,
    saved: usize,
    carry: Vec<u8>,
}

/// What one answer has reached so far — kept where an interruption can read it.
#[derive(Default, Clone)]
struct TurnState {
    text: String,
    spoken: Vec<String>,
    current: String,
    audio: bool,
    said: usize,
    heard: String,
}

struct Inner {
    agent: Agent,
    stt: Arc<dyn SpeechToText>,
    tts: Arc<dyn TextToSpeech>,
    vad: Mutex<Box<dyn VAD>>,
    rate: u32,
    output_rate: u32,
    barge_in: bool,
    barge_in_ms: f64,
    language: Option<String>,
    greeting: Option<String>,
    tool_filler: Option<String>,
    error_line: String,
    budget_line: String,
    prefetch: usize,
    session_id: Option<String>,
    state: Mutex<VoiceState>,
    last: Mutex<Option<Arc<Mutex<TurnState>>>>,
}

/// Builds a [`VoiceAgent`].
pub struct VoiceAgentBuilder {
    agent: Agent,
    stt: Option<Arc<dyn SpeechToText>>,
    tts: Option<Arc<dyn TextToSpeech>>,
    vad: Option<Box<dyn VAD>>,
    rate: u32,
    output_rate: Option<u32>,
    barge_in: bool,
    barge_in_ms: f64,
    language: Option<String>,
    greeting: Option<String>,
    tool_filler: Option<String>,
    error_line: String,
    budget_line: String,
    prefetch: usize,
    session: Option<String>,
}

impl VoiceAgentBuilder {
    /// Something that both transcribes and speaks (`OpenAISpeech`, `FakeSpeech`).
    pub fn speech<S: SpeechToText + TextToSpeech + 'static>(mut self, speech: Arc<S>) -> Self {
        self.stt = Some(speech.clone());
        self.tts = Some(speech);
        self
    }

    pub fn stt(mut self, stt: Arc<dyn SpeechToText>) -> Self {
        self.stt = Some(stt);
        self
    }

    pub fn tts(mut self, tts: Arc<dyn TextToSpeech>) -> Self {
        self.tts = Some(tts);
        self
    }

    /// What decides when the person has started and stopped. The default
    /// listens to loudness against the room; see [`EnergyVAD`].
    pub fn vad(mut self, vad: Box<dyn VAD>) -> Self {
        self.vad = Some(vad);
        self
    }

    /// The sample rate of the audio you feed in (16 kHz by default).
    pub fn rate(mut self, rate: u32) -> Self {
        self.rate = rate;
        self
    }

    /// The rate you want back, if not the voice's own — 8000 for a phone line.
    pub fn output_rate(mut self, rate: u32) -> Self {
        self.output_rate = Some(rate);
        self
    }

    /// Let the person interrupt (on by default).
    pub fn barge_in(mut self, on: bool) -> Self {
        self.barge_in = on;
        self
    }

    /// How much speech it takes to interrupt, on top of what the detector
    /// already needs — so a cough does not cut an answer short.
    pub fn barge_in_ms(mut self, ms: f64) -> Self {
        self.barge_in_ms = ms;
        self
    }

    pub fn language(mut self, code: &str) -> Self {
        self.language = Some(code.into());
        self
    }

    /// Said once, when the conversation opens.
    pub fn greeting(mut self, text: &str) -> Self {
        self.greeting = Some(text.into());
        self
    }

    /// Said if the agent reaches for a tool before saying anything — "One
    /// moment." — so a lookup is not a silence.
    pub fn tool_filler(mut self, text: &str) -> Self {
        self.tool_filler = Some(text.into());
        self
    }

    pub fn error_line(mut self, text: &str) -> Self {
        self.error_line = text.into();
        self
    }

    pub fn budget_line(mut self, text: &str) -> Self {
        self.budget_line = text.into();
        self
    }

    /// How many sentences may be synthesised ahead of the one being played.
    pub fn prefetch(mut self, n: usize) -> Self {
        self.prefetch = n.max(1);
        self
    }

    /// A session id to carry on from.
    pub fn session(mut self, id: &str) -> Self {
        self.session = Some(id.into());
        self
    }

    pub fn build(self) -> Result<VoiceAgent> {
        let (Some(stt), Some(tts)) = (self.stt, self.tts) else {
            return Err(Error::config(format!(
                "{}: a voice agent needs something to listen and speak with — VoiceAgent::builder(agent).speech(Arc::new(OpenAISpeech::new(None)))",
                self.agent.name()
            )));
        };
        let vad = self.vad.unwrap_or_else(|| Box::new(EnergyVAD::new(self.rate)));
        if vad.rate() != self.rate {
            return Err(Error::config(format!("the detector listens at {} Hz but the audio is {} Hz — build it with the same rate", vad.rate(), self.rate)));
        }
        let output_rate = self.output_rate.unwrap_or_else(|| tts.rate());
        Ok(VoiceAgent {
            inner: Arc::new(Inner {
                agent: self.agent,
                stt,
                tts,
                vad: Mutex::new(vad),
                rate: self.rate,
                output_rate,
                barge_in: self.barge_in,
                barge_in_ms: self.barge_in_ms,
                language: self.language,
                greeting: self.greeting,
                tool_filler: self.tool_filler,
                error_line: self.error_line,
                budget_line: self.budget_line,
                prefetch: self.prefetch,
                session_id: self.session,
                state: Mutex::default(),
                last: Mutex::new(None),
            }),
        })
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn ms_since(began: Instant, at: Option<Instant>) -> Value {
    at.map_or(Value::Null, |t| json!(((t - began).as_secs_f64() * 10_000.0).round() / 10.0))
}

/// An agent you talk to: speech in, the agent's answer spoken back. Cheap to
/// clone; clones are the same conversation.
#[derive(Clone)]
pub struct VoiceAgent {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for VoiceAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<VoiceAgent {} {} messages>", self.inner.agent.name(), lock(&self.inner.state).history.len())
    }
}

impl VoiceAgent {
    pub fn builder(agent: Agent) -> VoiceAgentBuilder {
        VoiceAgentBuilder {
            agent,
            stt: None,
            tts: None,
            vad: None,
            rate: 16_000,
            output_rate: None,
            barge_in: true,
            barge_in_ms: 120.0,
            language: None,
            greeting: None,
            tool_filler: None,
            error_line: "Sorry, something went wrong on my side.".into(),
            budget_line: "I have reached my limit for this conversation.".into(),
            prefetch: 2,
            session: None,
        }
    }

    pub fn agent(&self) -> &Agent {
        &self.inner.agent
    }

    /// The conversation as the agent will next see it.
    pub fn history(&self) -> Vec<Message> {
        lock(&self.inner.state).history.clone()
    }

    pub fn usage(&self) -> Usage {
        lock(&self.inner.state).usage.clone()
    }

    /// The timings of every finished turn, newest last.
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

    /// Typical time to the first sound of an answer, over the turns so far.
    pub fn latency(&self) -> Value {
        let mut marks: Vec<f64> = lock(&self.inner.state).turns.iter().filter_map(|t| t.get("first_audio_ms").and_then(Value::as_f64)).collect();
        if marks.is_empty() {
            return json!({"turns": 0, "median_first_audio_ms": null, "worst_first_audio_ms": null});
        }
        marks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        json!({"turns": marks.len(), "median_first_audio_ms": marks[marks.len() / 2], "worst_first_audio_ms": marks[marks.len() - 1]})
    }

    // ---- the conversation's record ---------------------------------------
    async fn open(&self) -> Result<()> {
        if lock(&self.inner.state).session.is_some() {
            return Ok(());
        }
        let me = &self.inner;
        // A voice leaves this machine twice — to be transcribed, and as the
        // words to speak. Both are put to the same check as a model call, so a
        // residency policy that covers the model covers the audio.
        for role in ["transcription", "speech"] {
            let egress = me
                .agent
                .hook_engine()
                .emit("model_egress", me.agent.name(), "", 0, [("provider", json!(role)), ("model", json!(role)), ("purpose", json!(role))].into_iter().map(|(k, v)| (k.to_string(), v)).collect())
                .await;
            if egress.blocked {
                let detail: Map<String, Value> = [("reason", json!(egress.reason)), ("purpose", json!(role))].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
                me.agent.harness().audit.record(me.agent.name(), "model_egress", role, "deny", "", detail);
                return Err(Error::PermissionDenied { message: format!("{} may not send audio for {role}: {}", me.agent.name(), egress.reason), tool: role.into(), reason: egress.reason });
            }
        }
        let session = match &me.session_id {
            Some(id) => {
                let loaded = me.agent.harness().sessions.load(id).await?;
                let mut s = lock(&me.state);
                s.history = close_open_tool_calls(loaded.messages.clone());
                s.holder = loaded.metadata.get("handoff").and_then(|h| h.get("active")).and_then(Value::as_str).unwrap_or("").to_string();
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

    async fn save(&self, usage: Option<Usage>) -> Result<()> {
        self.open().await?;
        let me = &self.inner;
        let (mut session, added, history) = {
            let s = lock(&me.state);
            let added = s.history[s.saved.min(s.history.len())..].to_vec();
            if added.is_empty() && usage.is_none() {
                return Ok(());
            }
            let mut session = s.session.clone().expect("opened above");
            session.messages = s.history.clone();
            if !s.holder.is_empty() {
                let mut handoff = session.metadata.get("handoff").and_then(Value::as_object).cloned().unwrap_or_else(|| [("entry".to_string(), json!(me.agent.name()))].into_iter().collect());
                handoff.insert("active".into(), json!(s.holder));
                session.metadata.insert("handoff".into(), Value::Object(handoff));
            }
            (session, added, s.history.clone())
        };
        let _ = history;
        if session.title.is_empty()
            && let Some(first) = added.first()
        {
            session.title = first.text().split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect();
        }
        let mut carrier = RunResult { agent: me.agent.name().into(), session_id: session.id.clone(), usage: usage.clone().unwrap_or_default(), ..Default::default() };
        if let Some(u) = &usage {
            session.usage += u;
        }
        me.agent.save_session_record(&mut session, &mut carrier, &added).await;
        // If someone else had written to the session meanwhile, their turns are
        // in it now; the conversation carries on from the whole of it.
        let mut s = lock(&me.state);
        s.history = session.messages.clone();
        s.saved = s.history.len();
        s.session = Some(session);
        Ok(())
    }

    // ---- text in, speech out ---------------------------------------------
    /// Answer one thing that was said, aloud: `Text` and `Audio` events as they
    /// are ready, then `TurnEnd`.
    ///
    /// Drop the stream early and the answer stops there; the conversation
    /// keeps only what had been spoken.
    pub fn respond(&self, text: &str) -> BoxStream<'static, VoiceEvent> {
        self.respond_from(text.to_string(), Instant::now())
    }

    fn respond_from(&self, text: String, began: Instant) -> BoxStream<'static, VoiceEvent> {
        let me = self.clone();
        Box::pin(async_stream::stream! {
            if let Err(e) = me.open().await {
                yield VoiceEvent::text(VoiceEventType::Error, format!("{}: {e}", e.kind_name()));
                return;
            }
            let inner = me.inner.clone();
            let turn = Arc::new(Mutex::new(TurnState { text: text.clone(), ..Default::default() }));
            *lock(&inner.last) = Some(turn.clone());
            let timing: Arc<Mutex<(Option<Instant>, Option<Instant>)>> = Arc::default();
            let (events_tx, mut events) = mpsc::unbounded_channel::<VoiceEvent>();
            let (order_tx, mut order) = mpsc::unbounded_channel::<(String, mpsc::UnboundedReceiver<Result<Vec<u8>>>)>();
            let gate = Arc::new(Semaphore::new(inner.prefetch));
            let synths: Arc<Mutex<Vec<tokio::task::AbortHandle>>> = Arc::default();

            let say = {
                let (inner, turn, synths, gate, order_tx) = (inner.clone(), turn.clone(), synths.clone(), gate.clone(), order_tx.clone());
                move |piece: String| {
                    let (sink, out) = mpsc::unbounded_channel();
                    lock(&turn).said += 1;
                    let (tts, gate, said) = (inner.tts.clone(), gate.clone(), piece.clone());
                    let task = tokio::spawn(async move {
                        let Ok(_permit) = gate.acquire_owned().await else { return };
                        let mut chunks = tts.synthesize(&said);
                        while let Some(c) = chunks.next().await {
                            let failed = c.is_err();
                            // One sentence lost, not the turn.
                            if sink.send(c).is_err() || failed {
                                return;
                            }
                        }
                    });
                    lock(&synths).push(task.abort_handle());
                    let _ = order_tx.send((piece, out));
                }
            };

            let thinker = {
                let (inner, turn, timing, events_tx, text) = (inner.clone(), turn.clone(), timing.clone(), events_tx.clone(), text.clone());
                let say = say.clone();
                tokio::spawn(async move {
                    let mut result: Option<RunResult> = None;
                    let mut chunker = SpeechChunker::default();
                    let (holder, history) = {
                        let s = lock(&inner.state);
                        (s.holder.clone(), s.history.clone())
                    };
                    let speaker = if holder.is_empty() { None } else { inner.agent.handoff_agent(&holder) }.unwrap_or_else(|| inner.agent.clone());
                    match speaker.stream(text.as_str(), RunOptions { messages: Some(history), ..Default::default() }).await {
                        Ok(mut stream) => {
                            while let Some(event) = stream.next().await {
                                match event.kind {
                                    StreamEventType::Text if !event.text.is_empty() => {
                                        lock(&timing).0.get_or_insert_with(Instant::now);
                                        let _ = events_tx.send(VoiceEvent::text(VoiceEventType::Text, event.text.clone()));
                                        for piece in chunker.feed(&event.text) {
                                            say(piece);
                                        }
                                    }
                                    StreamEventType::ToolCall => {
                                        let _ = events_tx.send(VoiceEvent { data: event.data.clone(), ..VoiceEvent::text(VoiceEventType::Tool, event.text.clone()) });
                                        if let Some(filler) = &inner.tool_filler
                                            && lock(&turn).said == 0
                                        {
                                            say(filler.clone());
                                        }
                                    }
                                    StreamEventType::RunEnd => result = event.data.get("result").and_then(|r| serde_json::from_value(r.clone()).ok()),
                                    _ => {}
                                }
                            }
                        }
                        Err(e) => {
                            result = Some(RunResult { agent: inner.agent.name().into(), error: Some(format!("{}: {e}", e.kind_name())), ..Default::default() });
                        }
                    }
                    for piece in chunker.flush() {
                        say(piece);
                    }
                    let said = lock(&turn).said;
                    if let Some(r) = &result {
                        if let (Some(err), 0) = (&r.error, said) {
                            let _ = events_tx.send(VoiceEvent::text(VoiceEventType::Error, err.clone()));
                            say(inner.error_line.clone());
                        } else if r.budget_exceeded.is_some() && said == 0 {
                            // Stopped by its budget before it could answer: say so,
                            // rather than leave the person listening to nothing.
                            say(inner.budget_line.clone());
                        }
                    }
                    result
                })
            };
            drop(order_tx);
            drop(say);

            let speaker_task = {
                let (inner, turn, timing, events_tx) = (inner.clone(), turn.clone(), timing.clone(), events_tx.clone());
                tokio::spawn(async move {
                    let Ok(mut resampler) = Resampler::new(inner.tts.rate(), inner.output_rate) else { return };
                    while let Some((piece, mut sink)) = order.recv().await {
                        lock(&turn).current = piece.clone();
                        let mut first = true;
                        while let Some(chunk) = sink.recv().await {
                            match chunk {
                                Err(e) => {
                                    let _ = events_tx.send(VoiceEvent::text(VoiceEventType::Error, format!("could not speak: {e}")));
                                    break;
                                }
                                Ok(pcm) => {
                                    lock(&timing).1.get_or_insert_with(Instant::now);
                                    lock(&turn).audio = true;
                                    let event = VoiceEvent { audio: resampler.feed(&pcm), ..VoiceEvent::text(VoiceEventType::Audio, if first { piece.clone() } else { String::new() }) }
                                        .with(json!({"rate": inner.output_rate}));
                                    let _ = events_tx.send(event);
                                    first = false;
                                }
                            }
                        }
                        let mut t = lock(&turn);
                        t.spoken.push(piece);
                        t.current.clear();
                    }
                })
            };
            drop(events_tx);

            // Dropped early (the person spoke over it): stop everything, and
            // remember what was heard rather than what was written.
            let guard = CutShort { voice: me.clone(), turn: turn.clone(), began, tasks: vec![thinker.abort_handle(), speaker_task.abort_handle()], synths: synths.clone(), done: false };

            while let Some(event) = events.recv().await {
                yield event;
            }
            let result = thinker.await.ok().flatten();
            let _ = speaker_task.await;
            let mut guard = guard;
            guard.done = true;
            drop(guard);

            let answer = lock(&turn).spoken.join(" ");
            {
                let mut s = lock(&inner.state);
                match &result {
                    Some(r) if r.error.is_none() => {
                        s.history = r.messages.clone();
                        if !r.active_agent.is_empty() {
                            s.holder = r.active_agent.clone();
                        }
                    }
                    _ => {
                        let said = if answer.is_empty() { inner.error_line.clone() } else { answer.clone() };
                        s.history.extend([Message::user(text.clone()), Message::assistant(said)]);
                    }
                }
                if let Some(r) = &result {
                    s.usage += &r.usage;
                }
            }
            let times = *lock(&timing);
            let metrics = me.metrics(began, times, false, result.as_ref(), &answer);
            let _ = me.save(result.as_ref().map(|r| r.usage.clone())).await;
            yield VoiceEvent { data: metrics, ..VoiceEvent::text(VoiceEventType::TurnEnd, answer) };
        })
    }

    fn metrics(&self, began: Instant, timing: (Option<Instant>, Option<Instant>), interrupted: bool, result: Option<&RunResult>, answer: &str) -> Map<String, Value> {
        let total = (began.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
        let first_audio = ms_since(began, timing.1);
        let metrics: Map<String, Value> = json!({
            "stt_ms": null,
            "first_token_ms": ms_since(began, timing.0),
            "first_audio_ms": first_audio,
            "total_ms": total,
            "interrupted": interrupted,
            "steps": result.map_or(0, |r| r.steps),
            "cost_usd": result.map_or(0.0, RunResult::cost_usd),
            "error": result.and_then(|r| r.error.clone()),
            "budget_exceeded": result.and_then(|r| r.budget_exceeded.clone()),
            "chars": answer.chars().count(),
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        lock(&self.inner.state).turns.push(metrics.clone());
        if let Some(ms) = first_audio.as_f64() {
            // The number that decides whether it feels like a conversation.
            self.inner.agent.harness().health.record("voice.first_audio", "voice", ms, true, "");
        }
        metrics
    }

    /// Speak a line of your own — a greeting, a notice — and put it in the
    /// conversation as something the agent said.
    pub fn say(&self, text: &str) -> BoxStream<'static, VoiceEvent> {
        let (me, text) = (self.clone(), text.to_string());
        Box::pin(async_stream::stream! {
            if let Err(e) = me.open().await {
                yield VoiceEvent::text(VoiceEventType::Error, format!("{}: {e}", e.kind_name()));
                return;
            }
            let rate = me.inner.output_rate;
            let Ok(mut resampler) = Resampler::new(me.inner.tts.rate(), rate) else { return };
            let mut chunks = me.inner.tts.synthesize(&text);
            let mut first = true;
            while let Some(c) = chunks.next().await {
                match c {
                    Ok(pcm) => {
                        yield VoiceEvent { audio: resampler.feed(&pcm), ..VoiceEvent::text(VoiceEventType::Audio, if first { text.clone() } else { String::new() }) }.with(json!({"rate": rate}));
                        first = false;
                    }
                    Err(e) => {
                        yield VoiceEvent::text(VoiceEventType::Error, format!("could not speak: {e}"));
                        break;
                    }
                }
            }
            lock(&me.inner.state).history.push(Message::assistant(text.clone()));
            let _ = me.save(None).await;
        })
    }

    // ---- speech in, speech out ---------------------------------------------
    /// Answer one recorded utterance — push-to-talk. `pcm` is everything that
    /// was said, at the input rate.
    pub fn listen(&self, pcm: Vec<u8>) -> BoxStream<'static, VoiceEvent> {
        let me = self.clone();
        Box::pin(async_stream::stream! {
            let began = Instant::now();
            let heard = me.inner.stt.transcribe(&wav(&pcm, me.inner.rate), "audio/wav", me.inner.language.as_deref()).await;
            let stt_ms = (began.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
            let text = match heard {
                Ok(t) => t.trim().to_string(),
                Err(e) => {
                    yield VoiceEvent::text(VoiceEventType::Error, format!("{}: {e}", e.kind_name()));
                    return;
                }
            };
            if text.is_empty() {
                yield VoiceEvent::new(VoiceEventType::TurnEnd).with(json!({"skipped": "nothing was said"}));
                return;
            }
            yield VoiceEvent::text(VoiceEventType::Transcript, text.clone()).with(json!({"stt_ms": stt_ms}));
            let mut answer = me.respond_from(text, began);
            while let Some(mut event) = answer.next().await {
                if event.kind == VoiceEventType::TurnEnd {
                    event.data.insert("stt_ms".into(), json!(stt_ms));
                    if let Some(last) = lock(&me.inner.state).turns.last_mut() {
                        last.insert("stt_ms".into(), json!(stt_ms));
                    }
                }
                yield event;
            }
        })
    }

    /// One turn of a live conversation, into the outgoing channel — until it
    /// finishes, or `cancel` says the person spoke over it.
    async fn answer(&self, pcm: Vec<u8>, out: mpsc::UnboundedSender<VoiceEvent>, mut cancel: watch::Receiver<bool>) {
        let audio = {
            let mut s = lock(&self.inner.state);
            let mut joined = std::mem::take(&mut s.carry);
            joined.extend_from_slice(&pcm);
            joined
        };
        *lock(&self.inner.last) = None;
        let mut turn = self.listen(audio.clone());
        let mut errored = false;
        loop {
            tokio::select! {
                biased;
                _ = cancel.wait_for(|c| *c) => {
                    drop(turn); // the guard keeps only what was heard
                    let last = lock(&self.inner.last).clone();
                    let heard = last.as_ref().map(|t| lock(t).clone());
                    match heard {
                        Some(t) if t.audio => {
                            // They heard part of the answer and spoke over it.
                            let _ = out.send(VoiceEvent::text(VoiceEventType::Interrupted, t.heard.clone()).with(json!({"heard_chars": t.heard.chars().count()})));
                        }
                        _ if duration_ms(&audio, self.inner.rate) <= CARRY_MS => {
                            // Nothing had been said back yet: this was the first
                            // half of a longer turn, answered with the rest.
                            lock(&self.inner.state).carry = audio;
                        }
                        _ => {}
                    }
                    return;
                }
                event = turn.next() => match event {
                    Some(e) => {
                        if e.kind == VoiceEventType::Error && e.data.is_empty() && lock(&self.inner.last).is_none() {
                            errored = true;
                        }
                        let _ = out.send(e);
                    }
                    None => break,
                },
            }
        }
        if errored {
            let detail: Map<String, Value> = [("reason".to_string(), json!("the turn failed before an answer"))].into_iter().collect();
            self.inner.agent.harness().audit.record(self.inner.agent.name(), "voice_turn", &self.session_id(), "error", "", detail);
        }
    }

    /// A whole conversation. Feed it the microphone — 16-bit mono PCM at the
    /// input rate, in chunks of any size — and play what comes back.
    ///
    /// It ends when the audio does, after the last answer has been spoken.
    pub fn run(&self, audio: BoxStream<'static, Vec<u8>>) -> BoxStream<'static, VoiceEvent> {
        let me = self.clone();
        Box::pin(async_stream::stream! {
            if let Err(e) = me.open().await {
                yield VoiceEvent::text(VoiceEventType::Error, format!("{}: {e}", e.kind_name()));
                return;
            }
            let (out_tx, mut out) = mpsc::unbounded_channel::<VoiceEvent>();
            let (utter_tx, mut utterances) = mpsc::unbounded_channel::<Vec<u8>>();
            // The current turn's cancel switch: set to interrupt it.
            let current: Arc<Mutex<Option<watch::Sender<bool>>>> = Arc::default();

            let listener = {
                let (me, out_tx, current) = (me.clone(), out_tx.clone(), current.clone());
                let mut audio = audio;
                tokio::spawn(async move {
                    let inner = &me.inner;
                    let (mut heard_ms, mut speech_at): (f64, Option<f64>) = (0.0, None);
                    let handle = |e: VadEvent, speech_at: &mut Option<f64>, heard_ms: f64| {
                        match e.kind {
                            VadKind::Start => {
                                *speech_at = Some(heard_ms);
                                let _ = out_tx.send(VoiceEvent::new(VoiceEventType::SpeechStarted).with(json!({"at_ms": e.at_ms})));
                            }
                            VadKind::Cancel => {
                                *speech_at = None;
                                let _ = out_tx.send(VoiceEvent::new(VoiceEventType::SpeechStopped).with(json!({"ignored": true})));
                            }
                            VadKind::End => {
                                *speech_at = None;
                                let ms = duration_ms(&e.audio, inner.rate).round();
                                let _ = out_tx.send(VoiceEvent::new(VoiceEventType::SpeechStopped).with(json!({"at_ms": e.at_ms, "ms": ms})));
                                let _ = utter_tx.send(e.audio);
                            }
                        }
                    };
                    while let Some(chunk) = audio.next().await {
                        heard_ms += duration_ms(&chunk, inner.rate);
                        let events = lock(&inner.vad).feed(&chunk);
                        for e in events {
                            handle(e, &mut speech_at, heard_ms);
                        }
                        if inner.barge_in && speech_at.is_some_and(|at| heard_ms - at >= inner.barge_in_ms) {
                            speech_at = None;
                            if let Some(stop) = lock(&current).as_ref() {
                                let _ = stop.send(true);
                            }
                        }
                    }
                    let events = lock(&inner.vad).flush();
                    for e in events {
                        handle(e, &mut speech_at, heard_ms);
                    }
                })
            };

            let worker = {
                let (me, out_tx, current) = (me.clone(), out_tx.clone(), current.clone());
                tokio::spawn(async move {
                    if let Some(greeting) = me.inner.greeting.clone() {
                        let mut said = me.say(&greeting);
                        while let Some(e) = said.next().await {
                            let _ = out_tx.send(e);
                        }
                    }
                    while let Some(pcm) = utterances.recv().await {
                        let (stop, cancel) = watch::channel(false);
                        *lock(&current) = Some(stop);
                        me.answer(pcm, out_tx.clone(), cancel).await;
                        *lock(&current) = None;
                    }
                })
            };
            drop(out_tx);

            let tasks = [listener.abort_handle(), worker.abort_handle()];
            struct Stop([tokio::task::AbortHandle; 2]);
            impl Drop for Stop {
                fn drop(&mut self) {
                    for t in &self.0 {
                        t.abort();
                    }
                }
            }
            let _stop = Stop(tasks);
            while let Some(event) = out.recv().await {
                yield event;
            }
            let _ = me.save(None).await;
        })
    }
}

/// What happens when an answer is dropped before it finished.
struct CutShort {
    voice: VoiceAgent,
    turn: Arc<Mutex<TurnState>>,
    began: Instant,
    tasks: Vec<tokio::task::AbortHandle>,
    synths: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
    done: bool,
}

impl Drop for CutShort {
    fn drop(&mut self) {
        for t in self.tasks.iter().chain(lock(&self.synths).iter()) {
            t.abort();
        }
        if self.done {
            return;
        }
        // The answer was stopped part-way. Remember what was heard of it — not
        // what was written — so the agent does not think it said things the
        // person never got.
        let (heard, reached, text) = {
            let mut t = lock(&self.turn);
            let mut parts = t.spoken.clone();
            parts.push(t.current.clone());
            t.heard = parts.join(" ").trim().to_string();
            (t.heard.clone(), t.audio, t.text.clone())
        };
        if !reached {
            return; // nothing reached them: the turn never happened
        }
        let inner = &self.voice.inner;
        lock(&inner.state).history.extend([Message::user(text), Message::assistant(format!("{heard}{INTERRUPTED}"))]);
        self.voice.metrics(self.began, (None, None), true, None, &heard);
        let detail: Map<String, Value> = [("heard".to_string(), json!(heard.chars().count()))].into_iter().collect();
        inner.agent.harness().audit.record(inner.agent.name(), "voice_interrupted", &self.voice.session_id(), "ok", "", detail);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::llm_providers::{FakeProvider, Script, tool_call};
    use crate::voice::audio::{silence, voice_like};
    use crate::voice::speech::FakeSpeech;

    fn agent(script: Vec<Script>) -> Agent {
        Agent::builder("concierge").provider(Arc::new(FakeProvider::new(script))).memory(false).build().unwrap()
    }

    #[tokio::test]
    async fn push_to_talk_is_heard_answered_and_spoken() {
        let speech = Arc::new(FakeSpeech::new(["Where is my order?"]));
        let voice = VoiceAgent::builder(agent(vec!["Your order shipped on Thursday. It arrives tomorrow.".into()])).speech(speech.clone()).build().unwrap();
        let events: Vec<VoiceEvent> = voice.listen(voice_like(600.0)).collect().await;
        let kinds: Vec<VoiceEventType> = events.iter().map(|e| e.kind).collect();
        assert_eq!(kinds[0], VoiceEventType::Transcript);
        assert_eq!(events[0].text, "Where is my order?");
        assert!(kinds.contains(&VoiceEventType::Audio));
        let end = events.last().unwrap();
        assert_eq!(end.kind, VoiceEventType::TurnEnd);
        assert_eq!(end.text, "Your order shipped on Thursday. It arrives tomorrow.");
        assert!(end.data["first_audio_ms"].is_number() && end.data["stt_ms"].is_number());
        assert_eq!(speech.spoken(), vec!["Your order shipped on Thursday.", "It arrives tomorrow."]);
        assert_eq!(voice.history().len(), 2);
        let saved = voice.agent().harness().sessions.load(&voice.session_id()).await.unwrap();
        assert_eq!(saved.messages.len(), 2);
        assert_eq!(voice.latency()["turns"], 1);
    }

    #[tokio::test]
    async fn a_tool_gets_a_filler_and_a_failure_is_said() {
        let speech = Arc::new(FakeSpeech::new(Vec::<String>::new()));
        let a = agent(vec![tool_call("lookup", json!({})), "Found it.".into()]);
        let voice = VoiceAgent::builder(a).speech(speech.clone()).tool_filler("One moment.").build().unwrap();
        let events: Vec<VoiceEvent> = voice.respond("find it").collect().await;
        assert!(events.iter().any(|e| e.kind == VoiceEventType::Tool));
        assert_eq!(speech.spoken()[0], "One moment.");

        let down = crate::errors::ProviderError::new(crate::errors::ProviderErrorKind::InvalidRequest, "the model refused the request").provider("fake");
        let failing = Agent::builder("x").provider(Arc::new(FakeProvider::new(vec![Script::Error(down.into())]))).memory(false).build().unwrap();
        let speech = Arc::new(FakeSpeech::new(Vec::<String>::new()));
        let voice = VoiceAgent::builder(failing).speech(speech.clone()).build().unwrap();
        let events: Vec<VoiceEvent> = voice.respond("hello").collect().await;
        assert!(events.iter().any(|e| e.kind == VoiceEventType::Error));
        assert_eq!(speech.spoken(), vec!["Sorry, something went wrong on my side."]);
    }

    #[tokio::test]
    async fn a_conversation_is_turn_taking_with_interruptions() {
        // A long answer, slowly spoken; the person speaks over it.
        let long = "This is the first part of a long answer. ".repeat(6);
        let mut speech = FakeSpeech::new(["Tell me everything.", "Stop, thanks."]);
        speech.delay = 0.02;
        let speech = Arc::new(speech);
        let a = agent(vec![Script::text(long), "Okay.".into()]);
        let voice = VoiceAgent::builder(a).speech(speech).greeting("Hello, how can I help?").build().unwrap();
        let mic = async_stream::stream! {
            yield silence(300.0, 16_000, 40);
            yield voice_like(800.0);
            for _ in 0..10 { yield silence(100.0, 16_000, 40); }   // the turn ends; the answer begins
            // Long enough for the greeting to finish and the answer to start playing.
            tokio::time::sleep(Duration::from_millis(900)).await;
            for _ in 0..30 { yield voice_like(20.0); }               // spoken over it, a frame at a time
            for _ in 0..10 { yield silence(100.0, 16_000, 40); }
            tokio::time::sleep(Duration::from_millis(600)).await;
        };
        let events: Vec<VoiceEvent> = voice.run(Box::pin(mic)).collect().await;
        let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
        assert!(events.iter().any(|e| e.kind == VoiceEventType::Audio && e.text == "Hello, how can I help?"), "the greeting is spoken: {kinds:?}");
        assert!(kinds.contains(&"interrupted"), "{kinds:?}");
        assert!(kinds.iter().filter(|k| **k == "transcript").count() == 2, "{kinds:?}");
        let history = voice.history();
        assert!(history.iter().any(|m| m.text().ends_with(INTERRUPTED)), "only what was heard is kept");
        assert_eq!(history.last().unwrap().text(), "Okay.");
    }

    #[test]
    fn a_detector_must_hear_at_the_same_rate() {
        let speech = Arc::new(FakeSpeech::new(Vec::<String>::new()));
        let err = VoiceAgent::builder(agent(vec![])).speech(speech).rate(8_000).vad(Box::new(EnergyVAD::new(16_000))).build().unwrap_err();
        assert!(err.to_string().contains("the detector listens at 16000 Hz but the audio is 8000 Hz"));
        assert!(VoiceAgent::builder(agent(vec![])).build().unwrap_err().to_string().contains("needs something to listen and speak with"));
    }
}
