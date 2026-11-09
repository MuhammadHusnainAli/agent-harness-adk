//! Voice: agents you talk to.
//!
//! Two ways to build one:
//!
//! - [`VoiceAgent`] — a pipeline: speech-to-text, the agent's own model, then
//!   text-to-speech. Every answer passes the output guardrails *before* it is
//!   spoken, and any model the harness supports can do the thinking.
//! - [`RealtimeAgent`] — one speech-to-speech model over a WebSocket (OpenAI
//!   Realtime, Gemini Live). The fastest replies, and the model hears tone; its
//!   guardrails apply to the transcript after the fact.
//!
//! Both are the same [`Agent`](crate::Agent) underneath: its tools, its hooks,
//! its permission gate, its audit trail and its budget. The pieces they are
//! made of — [`audio`], [`vad`], [`text`], [`speech`], [`ws`] — are public too.

pub mod audio;
pub mod pipeline;
pub mod realtime;
pub mod speech;
pub mod text;
pub mod vad;
pub mod ws;

pub use audio::{Resampler, duration_ms, resample, rms, silence, tone, ulaw_decode, ulaw_encode, unwav, wav};
pub use pipeline::{VoiceAgent, VoiceAgentBuilder, VoiceEvent, VoiceEventType};
pub use realtime::{Connect, GeminiLive, OpenAIRealtime, RealtimeAgent, RealtimeBackend, RealtimeSocket, Signal, realtime_backend};
pub use speech::{FakeSpeech, OpenAISpeech, SpeechToText, TextToSpeech};
pub use text::{SpeechChunker, speakable};
pub use vad::{EnergyVAD, VAD, VadEvent, VadKind, VadOptions};
pub use ws::{WebSocket, WsMessage};
