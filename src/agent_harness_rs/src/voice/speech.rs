//! Speech in, speech out: what a voice agent listens and talks through.
//!
//! Two small contracts. Something that transcribes ([`SpeechToText`]) and
//! something that speaks ([`TextToSpeech`]), as a stream so the first sound
//! leaves before the last word is synthesised.
//!
//! [`OpenAISpeech`] does both against OpenAI's audio endpoints — or anything
//! that speaks the same protocol, which covers Groq's Whisper, Azure OpenAI,
//! and the local servers people run for Whisper and Kokoro. Anything that
//! implements the two traits is as good: Deepgram, ElevenLabs, a model of
//! your own.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};

use crate::errors::{Error, ProviderError, ProviderErrorKind, Result};
use crate::voice::audio::tone;

const EXTENSIONS: [(&str, &str); 10] = [
    ("audio/wav", "wav"),
    ("audio/x-wav", "wav"),
    ("audio/wave", "wav"),
    ("audio/mpeg", "mp3"),
    ("audio/mp3", "mp3"),
    ("audio/mp4", "m4a"),
    ("audio/x-m4a", "m4a"),
    ("audio/ogg", "ogg"),
    ("audio/webm", "webm"),
    ("audio/flac", "flac"),
];

/// Something that hears: a recording in, what was said out.
#[async_trait]
pub trait SpeechToText: Send + Sync {
    async fn transcribe(&self, audio: &[u8], media_type: &str, language: Option<&str>) -> Result<String>;
}

/// Something that speaks: text in, 16-bit mono PCM at `rate()` out, in chunks.
pub trait TextToSpeech: Send + Sync {
    /// The sample rate of the PCM it produces.
    fn rate(&self) -> u32;
    fn synthesize(&self, text: &str) -> BoxStream<'static, Result<Vec<u8>>>;
}

fn speech_error(message: String, status: Option<u16>, retryable: bool) -> Error {
    let mut e = ProviderError::new(ProviderErrorKind::Other, message).provider("openai-speech").retryable(retryable);
    if let Some(s) = status {
        e = e.status(s);
    }
    Error::Provider(Box::new(e))
}

/// Transcription and speech over the OpenAI audio API.
///
/// ```no_run
/// # use agent_harness::voice::OpenAISpeech;
/// let speech = OpenAISpeech::new(None);                       // OPENAI_API_KEY
/// let speech = OpenAISpeech::new(None).voice("marin").instructions("Warm, unhurried.");
/// let local = OpenAISpeech::new(Some("-")).base_url("http://localhost:8000/v1").models("whisper-1", "kokoro");
/// ```
///
/// One HTTP client is kept open for both, so after the first call a request
/// costs no handshake — on a voice turn that is a tenth of a second back.
#[derive(Clone)]
pub struct OpenAISpeech {
    pub api_key: Option<String>,
    pub base_url: String,
    pub stt_model: String,
    pub tts_model: String,
    pub voice: String,
    /// How to say it — tone, pace — for models that take direction.
    pub instructions: Option<String>,
    /// 0.25 to 4.0; 1.0 is natural.
    pub speed: Option<f64>,
    /// An ISO-639-1 code. Saying it makes transcription faster and better.
    pub language: Option<String>,
    pub retries: u32,
    client: reqwest::Client,
}

impl std::fmt::Debug for OpenAISpeech {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<OpenAISpeech {} {}/{}>", self.base_url, self.stt_model, self.tts_model)
    }
}

impl OpenAISpeech {
    /// OpenAI's `pcm` format: 24 kHz, 16-bit, mono, little-endian.
    pub const RATE: u32 = 24_000;

    pub fn new(api_key: Option<&str>) -> Self {
        crate::llm_providers::base::install_tls();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .pool_idle_timeout(Duration::from_secs(120))
            .pool_max_idle_per_host(8)
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.map(str::to_string),
            base_url: "https://api.openai.com/v1".into(),
            stt_model: "gpt-4o-mini-transcribe".into(),
            tts_model: "gpt-4o-mini-tts".into(),
            voice: "alloy".into(),
            instructions: None,
            speed: None,
            language: None,
            retries: 1,
            client,
        }
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    pub fn models(mut self, stt: &str, tts: &str) -> Self {
        self.stt_model = stt.into();
        self.tts_model = tts.into();
        self
    }

    pub fn voice(mut self, voice: &str) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn instructions(mut self, text: &str) -> Self {
        self.instructions = Some(text.into());
        self
    }

    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn language(mut self, code: &str) -> Self {
        self.language = Some(code.into());
        self
    }

    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    fn key(&self) -> Result<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .filter(|k| !k.is_empty())
            .ok_or_else(|| Error::config("OpenAISpeech needs an API key — set OPENAI_API_KEY, or pass api_key"))
    }

    fn failed(what: &str, status: u16, body: &str) -> Error {
        speech_error(format!("{what} failed with {status}: {}", body.chars().take(300).collect::<String>()), Some(status), status == 408 || status == 429 || status >= 500)
    }
}

#[async_trait]
impl SpeechToText for OpenAISpeech {
    /// What was said in a recording.
    async fn transcribe(&self, audio: &[u8], media_type: &str, language: Option<&str>) -> Result<String> {
        let ext = EXTENSIONS.iter().find(|(m, _)| *m == media_type).map_or("wav", |(_, e)| e);
        let mut fields: Vec<crate::toolkits::openapi::FormField> =
            vec![("model".into(), self.stt_model.clone(), None), ("response_format".into(), "json".into(), None)];
        if let Some(lang) = language.or(self.language.as_deref()) {
            fields.push(("language".into(), lang.into(), None));
        }
        fields.push(("file".into(), String::new(), Some((format!("audio.{ext}"), media_type.to_string(), audio.to_vec()))));
        let (content_type, body) = crate::toolkits::openapi::multipart(&fields);
        let key = self.key()?;
        let mut last = None;
        for attempt in 0..=self.retries {
            let sent = self
                .client
                .post(format!("{}/audio/transcriptions", self.base_url))
                .bearer_auth(&key)
                .header("content-type", &content_type)
                .body(body.clone())
                .send()
                .await;
            match sent {
                Err(e) => last = Some(speech_error(format!("transcription could not be reached: {e}"), None, true)),
                Ok(r) => {
                    let status = r.status().as_u16();
                    let text = r.text().await.unwrap_or_default();
                    if status < 400 {
                        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                        return Ok(v["text"].as_str().map_or_else(|| v["text"].to_string(), str::to_string).trim().trim_matches('"').to_string());
                    }
                    let e = Self::failed("transcription", status, &text);
                    let retryable = e.as_provider().is_some_and(|p| p.retryable);
                    last = Some(e);
                    if !retryable {
                        break;
                    }
                }
            }
            if attempt < self.retries {
                tokio::time::sleep(Duration::from_secs_f64(0.2 * (attempt + 1) as f64)).await;
            }
        }
        Err(last.unwrap_or_else(|| Error::config("transcription produced nothing")))
    }
}

impl TextToSpeech for OpenAISpeech {
    fn rate(&self) -> u32 {
        Self::RATE
    }

    /// `text` as speech: 16-bit PCM at 24 kHz, in chunks as it is made.
    fn synthesize(&self, text: &str) -> BoxStream<'static, Result<Vec<u8>>> {
        let mut payload = json!({"model": self.tts_model, "voice": self.voice, "input": text, "response_format": "pcm"});
        if let Some(i) = &self.instructions {
            payload["instructions"] = json!(i);
        }
        if let Some(s) = self.speed {
            payload["speed"] = json!(s);
        }
        let me = self.clone();
        Box::pin(async_stream::try_stream! {
            let key = me.key()?;
            let response = me.client.post(format!("{}/audio/speech", me.base_url)).bearer_auth(&key).json(&payload).send().await
                .map_err(|e| speech_error(format!("speech synthesis could not be reached: {e}"), None, true))?;
            let status = response.status().as_u16();
            if status >= 400 {
                let body = response.text().await.unwrap_or_default();
                Err(Self::failed("speech synthesis", status, &body))?;
            } else {
                let mut odd: Vec<u8> = Vec::new();
                let mut chunks = response.bytes_stream();
                while let Some(chunk) = chunks.next().await {
                    let chunk = chunk.map_err(|e| speech_error(format!("speech synthesis was cut off: {e}"), None, true))?;
                    // A sample is two bytes; a chunk boundary may fall between them.
                    let mut data = std::mem::take(&mut odd);
                    data.extend_from_slice(&chunk);
                    if data.len() % 2 == 1 {
                        odd = vec![data.pop().expect("odd length")];
                    }
                    if !data.is_empty() {
                        yield data;
                    }
                }
            }
        })
    }
}

#[async_trait]
impl crate::attachments::Transcriber for OpenAISpeech {
    async fn transcribe(&self, audio: &[u8], media_type: &str) -> Result<String> {
        SpeechToText::transcribe(self, audio, media_type, None).await
    }
}

/// How a [`FakeSpeech`] answers a recording.
pub type Transcripts = Arc<dyn Fn(&[u8]) -> String + Send + Sync>;

/// What a `FakeSpeech` has heard, what it said, and how far through its lines it is.
type Scripted = (Vec<Vec<u8>>, Vec<String>, usize);

/// Scripted speech, for testing a voice agent with no network.
///
/// Each call to `transcribe` returns the next line (or what a function makes
/// of the audio). `synthesize` returns a tone whose length follows the text —
/// 60 ms a character — in 100 ms chunks, so timing, ordering and interruption
/// can all be tested.
#[derive(Clone)]
pub struct FakeSpeech {
    lines: Vec<String>,
    function: Option<Transcripts>,
    pub ms_per_char: f64,
    pub chunk_ms: u32,
    /// Seconds to wait before each synthesised chunk.
    pub delay: f64,
    /// Seconds to wait before a transcription.
    pub stt_delay: f64,
    state: Arc<Mutex<Scripted>>,
}

impl std::fmt::Debug for FakeSpeech {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<FakeSpeech {} lines>", self.lines.len())
    }
}

impl FakeSpeech {
    pub const RATE: u32 = 16_000;

    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(lines: I) -> Self {
        Self { lines: lines.into_iter().map(Into::into).collect(), function: None, ms_per_char: 60.0, chunk_ms: 100, delay: 0.0, stt_delay: 0.0, state: Arc::default() }
    }

    pub fn function(f: impl Fn(&[u8]) -> String + Send + Sync + 'static) -> Self {
        Self { function: Some(Arc::new(f)), ..Self::new(Vec::<String>::new()) }
    }

    /// Every recording it was given.
    pub fn heard(&self) -> Vec<Vec<u8>> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).0.clone()
    }

    /// Every piece of text it was asked to say.
    pub fn spoken(&self) -> Vec<String> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).1.clone()
    }
}

#[async_trait]
impl SpeechToText for FakeSpeech {
    async fn transcribe(&self, audio: &[u8], _media_type: &str, _language: Option<&str>) -> Result<String> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).0.push(audio.to_vec());
        if self.stt_delay > 0.0 {
            tokio::time::sleep(Duration::from_secs_f64(self.stt_delay)).await;
        }
        if let Some(f) = &self.function {
            return Ok(f(audio));
        }
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let line = self.lines.get(s.2).cloned().unwrap_or_default();
        if s.2 < self.lines.len() {
            s.2 += 1;
        }
        Ok(line)
    }
}

impl TextToSpeech for FakeSpeech {
    fn rate(&self) -> u32 {
        Self::RATE
    }

    fn synthesize(&self, text: &str) -> BoxStream<'static, Result<Vec<u8>>> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).1.push(text.to_string());
        let pcm = tone(text.chars().count() as f64 * self.ms_per_char, Self::RATE, 220.0, 9_000);
        let size = ((Self::RATE as f64 * self.chunk_ms as f64 / 1000.0) as usize * 2).max(2);
        let delay = self.delay;
        let chunks: Vec<Vec<u8>> = pcm.chunks(size).map(<[u8]>::to_vec).collect();
        Box::pin(async_stream::stream! {
            for c in chunks {
                if delay > 0.0 {
                    tokio::time::sleep(Duration::from_secs_f64(delay)).await;
                }
                yield Ok(c);
            }
        })
    }
}

#[async_trait]
impl crate::attachments::Transcriber for FakeSpeech {
    async fn transcribe(&self, audio: &[u8], media_type: &str) -> Result<String> {
        SpeechToText::transcribe(self, audio, media_type, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_speech_is_scripted_and_timed() {
        let s = FakeSpeech::new(["what is my balance?", "thanks"]);
        assert_eq!(s.transcribe(b"x", "audio/wav", None).await.unwrap(), "what is my balance?");
        assert_eq!(s.transcribe(b"y", "audio/wav", None).await.unwrap(), "thanks");
        assert_eq!(s.transcribe(b"z", "audio/wav", None).await.unwrap(), "");
        let chunks: Vec<Vec<u8>> = s.synthesize("hello").map(|c| c.unwrap()).collect().await;
        // 5 characters at 60 ms is 300 ms: three 100 ms chunks of 16 kHz PCM.
        assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), vec![3200, 3200, 3200]);
        assert_eq!(s.spoken(), vec!["hello"]);
        assert_eq!(s.heard().len(), 3);
    }

    #[tokio::test]
    async fn openai_speech_over_the_wire() {
        use axum::Router;
        use axum::routing::post;
        let app = Router::new()
            .route("/v1/audio/transcriptions", post(|body: axum::body::Bytes| async move {
                let text = String::from_utf8_lossy(&body);
                assert!(text.contains("name=\"model\"\r\n\r\ngpt-4o-mini-transcribe") && text.contains("filename=\"audio.wav\""), "{text}");
                axum::Json(json!({"text": " Where is my order? "}))
            }))
            .route("/v1/audio/speech", post(|axum::Json(v): axum::Json<Value>| async move {
                assert_eq!((v["voice"].as_str(), v["response_format"].as_str()), (Some("marin"), Some("pcm")));
                vec![1u8, 0, 2, 0, 3]
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let s = OpenAISpeech::new(Some("k")).base_url(&format!("http://{addr}/v1")).voice("marin");
        assert_eq!(SpeechToText::transcribe(&s, b"RIFF", "audio/wav", None).await.unwrap(), "Where is my order?");
        let pcm: Vec<u8> = s.synthesize("hi").map(|c| c.unwrap()).collect::<Vec<_>>().await.concat();
        assert_eq!(pcm, vec![1, 0, 2, 0], "a dangling odd byte is held back");
        let missing = OpenAISpeech::new(None).base_url("http://127.0.0.1:9");
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert!(SpeechToText::transcribe(&missing, b"", "audio/wav", None).await.unwrap_err().to_string().contains("needs an API key"));
        }
    }
}
