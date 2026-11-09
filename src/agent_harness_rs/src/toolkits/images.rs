//! Image generation, as a tool: a prompt in, pictures out — with reference images.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::toolkits::images::{GenerateOptions, ImageGenerator, ImageOptions, ImageToolOptions};
//!
//! let images = ImageGenerator::new(ImageOptions::default())?;        // from the environment
//! let tools = images.tools(ImageToolOptions::default());             // for an agent
//! let made = images.generate("a red fox in snow", GenerateOptions { references: vec!["fox.jpg".into()], ..Default::default() }).await?;
//! made[0].save("fox.png")?;
//! # Ok(()) }
//! ```
//!
//! The engine is whichever has a key set — Gemini (`GEMINI_API_KEY`), OpenAI
//! (`OPENAI_API_KEY`), Replicate (`REPLICATE_API_TOKEN`) — or the ones you
//! name, tried in order. Every one is spoken to over plain HTTP.
//!
//! **Reference images** are what the picture is made from or made to look
//! like: a photo to edit, a style to follow, a character to keep. The model
//! names them by what it can see — a file in its workspace, an image it
//! generated earlier, an attachment that came with the task, a web address —
//! and never by a path on this machine.
//!
//! **Progress.** Every generation is a job. `generate` waits for it; `submit`
//! returns it at once, to be watched with `job.describe()` ("running on
//! replicate — 40%, 12s"), `job.done()` and `job.wait(..)`. An agent gets the
//! same: `generate_image` waits a while, and a job that outlives the wait is
//! collected with `image_status`.
//!
//! A generation is something that fails. A call that may succeed later is
//! retried with back-off; an engine that cannot answer is passed over for the
//! next; one that keeps failing is left alone for a minute; a job that is
//! cancelled or runs out of time is cancelled at the provider too. A prompt a
//! provider *refuses* is not retried and is not taken to another provider: the
//! model is told it was refused, and why. What comes back is checked to be an
//! image before it is kept.

use std::collections::HashMap;
use std::fmt;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as B64, URL_SAFE as B64_URL};
use futures::FutureExt;
use futures::future::{BoxFuture, join_all};
use indexmap::IndexMap;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, watch};

use super::net::is_private_host;
use super::openapi::{expand_user, head, multipart, py_repr, squash};
use crate::errors::{Error, Result};
use crate::llm_providers::resilience::{CircuitBreaker, CircuitState, RetryPolicy, retry_after};
use crate::tools::{Permission, Tool, ToolContext};
use crate::types::{ContentBlock, MediaBlock, Usage, new_id};

/// Set to an engine name, or several separated by commas, to choose the
/// engines of a generator that was not given any.
pub const ENGINE_ENV: &str = "AGENT_HARNESS_IMAGES";

/// The shapes an image may be asked for in.
pub const RATIOS: [&str; 10] = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
const ALIASES: [(&str, &str); 8] =
    [("square", "1:1"), ("landscape", "16:9"), ("wide", "16:9"), ("portrait", "9:16"), ("tall", "9:16"), ("auto", ""), ("any", ""), ("none", "")];
const MAX_PROMPT: usize = 8_000;
const STATUSES: [&str; 5] = ["queued", "running", "done", "failed", "cancelled"];

static SLUG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-z0-9]+").expect("valid regex"));
static PERCENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d{1,3})\s?%").expect("valid regex"));
static QUOTA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)quota|credit|billing|insufficient").expect("valid regex"));
static POSITION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^attachment[:\s#]*(\d+)$").expect("valid regex"));
/// What a provider says when the *content* was the problem, not the request.
static REFUSED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)safety|moderat|content.?policy|policy.?violation|prohibited|nsfw|flagged|sensitive|blocked|not allowed|violat").expect("valid regex")
});

fn extension_of(media_type: &str) -> &'static str {
    match media_type {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    }
}

fn tool_error(message: impl Into<String>) -> Error {
    Error::tool(message, "generate_image")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A number of seconds as Python's `{:g}` writes it: `180`, `0.05`.
fn seconds(f: f64) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 { format!("{}", f as i64) } else { format!("{f}") }
}

/// Python's `str()` of a value sent as a form field.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

// --- environment --------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// In tests: the environment the engines see, instead of the process's.
    static ENV: std::cell::RefCell<Option<HashMap<String, String>>> = const { std::cell::RefCell::new(None) };
}

fn env(name: &str) -> Option<String> {
    #[cfg(test)]
    {
        if let Some(found) = ENV.with(|e| e.borrow().as_ref().map(|m| m.get(name).cloned())) {
            return found.filter(|v| !v.is_empty());
        }
    }
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

// ----------------------------------------------------------------------
// images, read without a library
// ----------------------------------------------------------------------

fn be16(d: &[u8], i: usize) -> u32 {
    u32::from(u16::from_be_bytes([d[i], d[i + 1]]))
}

fn le16(d: &[u8], i: usize) -> u32 {
    u32::from(u16::from_le_bytes([d[i], d[i + 1]]))
}

fn le24(d: &[u8], i: usize) -> u32 {
    u32::from(d[i]) | (u32::from(d[i + 1]) << 8) | (u32::from(d[i + 2]) << 16)
}

/// `(media type, width, height)` of an image, from its first bytes.
///
/// `None` when the bytes are not a PNG, JPEG, WebP or GIF — which is how an
/// error page saved as `photo.png` is told from a photo.
pub fn sniff(data: &[u8]) -> Option<(&'static str, u32, u32)> {
    if data.len() < 16 {
        return None;
    }
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        if data.len() >= 24 {
            let w = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
            let h = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
            return Some(("image/png", w, h));
        }
        return Some(("image/png", 0, 0));
    }
    if data.starts_with(b"\xff\xd8\xff") {
        let mut index = 2;
        while index + 9 < data.len() {
            if data[index] != 0xFF {
                index += 1;
                continue;
            }
            let marker = data[index + 1];
            if matches!(marker, 0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF) {
                return Some(("image/jpeg", be16(data, index + 7), be16(data, index + 5)));
            }
            if matches!(marker, 0xD8 | 0x01 | 0xD0..=0xD7) {
                index += 2;
                continue;
            }
            index += 2 + be16(data, index + 2) as usize;
        }
        return Some(("image/jpeg", 0, 0));
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Some(("image/gif", le16(data, 6), le16(data, 8)));
    }
    if data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
        let kind = &data[12..16];
        if kind == b"VP8X" && data.len() >= 30 {
            return Some(("image/webp", le24(data, 24) + 1, le24(data, 27) + 1));
        }
        if kind == b"VP8 " && data.len() >= 30 {
            return Some(("image/webp", le16(data, 26) & 0x3FFF, le16(data, 28) & 0x3FFF));
        }
        if kind == b"VP8L" && data.len() >= 25 {
            let bits = u32::from_le_bytes([data[21], data[22], data[23], data[24]]);
            return Some(("image/webp", (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1));
        }
        return Some(("image/webp", 0, 0));
    }
    None
}

/// A PNG of one colour. What [`FakeImages`] draws.
pub fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    fn chunk(out: &mut Vec<u8>, kind: &[u8], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        let mut crc = flate2::Crc::new();
        crc.update(kind);
        crc.update(body);
        out.extend_from_slice(&crc.sum().to_be_bytes());
    }
    let mut row = vec![0u8];
    for _ in 0..width {
        row.extend_from_slice(&rgb);
    }
    let raw: Vec<u8> = row.repeat(height as usize);
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    let _ = encoder.write_all(&raw);
    let compressed = encoder.finish().unwrap_or_default();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut out, b"IHDR", &header);
    chunk(&mut out, b"IDAT", &compressed);
    chunk(&mut out, b"IEND", b"");
    out
}

fn size_text(count: usize) -> String {
    if count >= 1_048_576 { format!("{:.1} MB", count as f64 / 1_048_576.0) } else { format!("{} KB", (count / 1024).max(1)) }
}

fn slug(text: &str, fallback: &str) -> String {
    let lowered = text.to_lowercase();
    let slug = SLUG.replace_all(&lowered, "-");
    let slug = slug.trim_matches('-');
    let words: Vec<&str> = slug.split('-').take(6).collect();
    let joined = head(&words.join("-"), 48);
    let joined = joined.trim_matches('-');
    if joined.is_empty() { fallback.into() } else { joined.into() }
}

fn decode(text: &str) -> std::result::Result<Vec<u8>, ImageFailure> {
    let cleaned: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    B64.decode(&cleaned).or_else(|_| B64_URL.decode(&cleaned)).map_err(|_| ImageFailure::new("the engine sent an image that could not be decoded"))
}

// ----------------------------------------------------------------------
// what goes in, what comes out
// ----------------------------------------------------------------------

/// An image a generation is given: to edit, to follow, to keep a likeness of.
#[derive(Clone, PartialEq)]
pub struct Reference {
    pub data: Vec<u8>,
    pub media_type: String,
    pub name: String,
}

impl Reference {
    pub fn b64(&self) -> String {
        B64.encode(&self.data)
    }

    pub fn data_uri(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.b64())
    }
}

impl fmt::Debug for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Reference {} {}, {}>", self.name, self.media_type, size_text(self.data.len()))
    }
}

/// One generation, as every engine is asked for it.
#[derive(Debug, Clone, Default)]
pub struct ImageRequest {
    pub prompt: String,
    pub references: Vec<Reference>,
    /// e.g. "16:9"; empty to leave it to the model.
    pub aspect_ratio: String,
    pub count: u32,
    /// A resolution hint for models that take one: "1K", "2K", "4K".
    pub size: String,
    pub seed: Option<i64>,
    /// Passed to the engine as they are: `quality`, `background`, a model's own.
    pub options: Map<String, Value>,
}

impl ImageRequest {
    fn engine_options(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.options.iter().filter(|(k, _)| k.as_str() != "generation_config")
    }
}

/// One picture that was made.
#[derive(Clone, Default, PartialEq)]
pub struct GeneratedImage {
    pub data: Vec<u8>,
    pub media_type: String,
    pub width: u32,
    pub height: u32,
    pub prompt: String,
    pub engine: String,
    pub model: String,
    /// What it is called: the file name it was, or would be, saved under.
    pub name: String,
    /// Where it was saved; empty when it was not.
    pub path: String,
    /// Anything the model said alongside it.
    pub text: String,
}

impl GeneratedImage {
    /// Bytes an engine returned, taken to be a PNG until they are sniffed.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, media_type: "image/png".into(), ..Default::default() }
    }

    pub fn extension(&self) -> &'static str {
        extension_of(&self.media_type)
    }

    /// The image as something a model can be shown.
    pub fn block(&self) -> ContentBlock {
        let mut block = MediaBlock::from_bytes(&self.data, &self.media_type);
        block.name = if self.name.is_empty() { "image".into() } else { self.name.clone() };
        ContentBlock::Image(block)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let target = expand_user(&path.as_ref().to_string_lossy());
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &self.data)?;
        Ok(target)
    }

    pub fn describe(&self) -> String {
        let shape = if self.width > 0 { format!("{}x{} ", self.width, self.height) } else { String::new() };
        let label = if self.path.is_empty() { &self.name } else { &self.path };
        format!("{label} — {shape}{}, {}", self.extension(), size_text(self.data.len()))
    }
}

impl fmt::Debug for GeneratedImage {
    // Never the bytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<GeneratedImage {}>", self.describe())
    }
}

/// One engine could not make the image. Returned by engines, read by [`ImageGenerator`].
#[derive(Debug, Clone)]
pub struct ImageFailure {
    pub message: String,
    /// The same request may succeed in a moment (429, 5xx, timeout).
    pub retryable: bool,
    /// Seconds the engine asked us to wait, when it said.
    pub retry_after: Option<f64>,
    /// The provider declined the *content*. Not retried, and not taken to another provider.
    pub refused: bool,
    /// Counts towards leaving the engine alone for a while.
    pub unhealthy: bool,
    pub status: Option<u16>,
}

impl ImageFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), retryable: false, retry_after: None, refused: false, unhealthy: true, status: None }
    }

    /// May succeed if asked again.
    pub fn retryable(mut self) -> Self {
        self.retryable = !self.refused;
        self
    }

    /// The content was declined: neither retried nor counted against the engine.
    pub fn refused(mut self) -> Self {
        self.refused = true;
        self.retryable = false;
        self.unhealthy = false;
        self
    }

    /// Not the engine's fault: does not count towards leaving it alone.
    pub fn healthy(mut self) -> Self {
        self.unhealthy = false;
        self
    }

    pub fn wait(mut self, seconds: Option<f64>) -> Self {
        self.retry_after = seconds;
        self
    }

    pub fn status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }
}

impl fmt::Display for ImageFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ImageFailure {}

/// Where a job has got to. Handed to `on_progress`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ImageProgress {
    pub job: String,
    pub status: String,
    pub message: String,
    /// 0.0 to 1.0 when the engine reports how far along it is; `None` when it does not.
    pub fraction: Option<f64>,
    pub engine: String,
    pub elapsed: f64,
}

/// Called with an [`ImageProgress`] as a job moves.
pub type ProgressFn = Arc<dyn Fn(&ImageProgress) + Send + Sync>;

/// How an engine says how far along it is: `(status, message, fraction)`.
pub type Reporter = Arc<dyn Fn(&str, &str, Option<f64>) + Send + Sync>;

// ----------------------------------------------------------------------
// jobs
// ----------------------------------------------------------------------

struct JobState {
    status: String,
    message: String,
    fraction: Option<f64>,
    engine: String,
    images: Vec<GeneratedImage>,
    error: Option<Error>,
    finished: Option<Instant>,
    collected: bool,
}

struct JobInner {
    id: String,
    prompt: String,
    count: u32,
    started: Instant,
    state: Mutex<JobState>,
    done: watch::Sender<bool>,
    abort: Mutex<Option<tokio::task::AbortHandle>>,
    listeners: Vec<ProgressFn>,
}

/// One generation under way, or finished. Cheap to clone: clones share it.
///
/// `status` is `queued`, `running`, `done`, `failed` or `cancelled`.
#[derive(Clone)]
pub struct ImageJob {
    inner: Arc<JobInner>,
}

impl fmt::Debug for ImageJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<ImageJob {} {}>", self.inner.id, self.describe())
    }
}

impl ImageJob {
    fn create(prompt: &str, count: u32, listeners: Vec<ProgressFn>) -> Self {
        let (done, _) = watch::channel(false);
        Self {
            inner: Arc::new(JobInner {
                id: new_id("img"),
                prompt: prompt.into(),
                count,
                started: Instant::now(),
                state: Mutex::new(JobState {
                    status: "queued".into(),
                    message: String::new(),
                    fraction: None,
                    engine: String::new(),
                    images: vec![],
                    error: None,
                    finished: None,
                    collected: false,
                }),
                done,
                abort: Mutex::new(None),
                listeners,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, JobState> {
        lock(&self.inner.state)
    }

    pub fn id(&self) -> &str {
        &self.inner.id
    }

    pub fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    pub fn count(&self) -> u32 {
        self.inner.count
    }

    pub fn status(&self) -> String {
        self.state().status.clone()
    }

    pub fn message(&self) -> String {
        self.state().message.clone()
    }

    pub fn fraction(&self) -> Option<f64> {
        self.state().fraction
    }

    pub fn engine(&self) -> String {
        self.state().engine.clone()
    }

    pub fn images(&self) -> Vec<GeneratedImage> {
        self.state().images.clone()
    }

    /// Has what it made been handed to whoever asked? (For `image_status`.)
    pub fn collected(&self) -> bool {
        self.state().collected
    }

    pub fn done(&self) -> bool {
        is_over(&self.state().status)
    }

    pub fn elapsed(&self) -> f64 {
        elapsed(self.inner.started, self.state().finished)
    }

    /// What was made. Fails with what went wrong, or `Error::Tool` if it is not over.
    pub fn result(&self) -> Result<Vec<GeneratedImage>> {
        let state = self.state();
        if let Some(e) = &state.error {
            return Err(e.clone());
        }
        if !is_over(&state.status) {
            drop(state);
            return Err(tool_error(format!("job {} is still {}", self.inner.id, self.describe())));
        }
        Ok(state.images.clone())
    }

    /// Wait for it to finish. On a timeout the job keeps running.
    pub async fn wait(&self, timeout: Option<f64>) -> Result<Vec<GeneratedImage>> {
        let mut rx = self.inner.done.subscribe();
        let finished = async move {
            let _ = rx.wait_for(|d| *d).await;
        };
        match timeout {
            Some(t) => {
                if tokio::time::timeout(Duration::from_secs_f64(t.max(0.0)), finished).await.is_err() {
                    return Err(tool_error(format!("job {} is still {}", self.inner.id, self.describe())));
                }
            }
            None => finished.await,
        }
        self.result()
    }

    /// Stop it. The engine is told to stop too, where it can be.
    pub fn cancel(&self) -> bool {
        let update = {
            let mut state = self.state();
            if is_over(&state.status) {
                return false;
            }
            let Some(handle) = lock(&self.inner.abort).take() else { return false };
            handle.abort();
            state.finished = Some(Instant::now());
            state.error = Some(tool_error(format!("job {} was cancelled", self.inner.id)));
            self.apply(&mut state, "cancelled", "", None)
        };
        self.notify(&update);
        self.inner.done.send_replace(true);
        true
    }

    /// One line on where it stands.
    pub fn describe(&self) -> String {
        let state = self.state();
        let took = format!("{:.0}s", elapsed(self.inner.started, state.finished));
        match state.status.as_str() {
            "done" => {
                let n = state.images.len();
                format!("done — {n} image{} in {took}", if n == 1 { "" } else { "s" })
            }
            "failed" => format!("failed after {took}: {}", state.error.as_ref().map(ToString::to_string).unwrap_or_default()),
            "cancelled" => format!("cancelled after {took}"),
            status => {
                let place = if state.engine.is_empty() { String::new() } else { format!(" on {}", state.engine) };
                let how_far = state.fraction.map(|f| format!(" — {:.0}%", f * 100.0)).unwrap_or_default();
                let detail = if state.message.is_empty() { String::new() } else { format!(" ({})", state.message) };
                format!("{status}{place}{how_far}, {took}{detail}")
            }
        }
    }

    fn apply(&self, state: &mut JobState, status: &str, message: &str, fraction: Option<f64>) -> ImageProgress {
        state.message = message.into();
        if fraction.is_some() || status != state.status {
            state.fraction = fraction;
        }
        if STATUSES.contains(&status) {
            state.status = status.into();
        }
        ImageProgress {
            job: self.inner.id.clone(),
            status: status.into(),
            message: message.into(),
            fraction,
            engine: state.engine.clone(),
            elapsed: elapsed(self.inner.started, state.finished),
        }
    }

    fn notify(&self, update: &ImageProgress) {
        for listener in &self.inner.listeners {
            // A listener must not fail the job.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(update)));
        }
    }

    fn report(&self, status: &str, message: &str, fraction: Option<f64>) {
        let update = {
            let mut state = self.state();
            if is_over(&state.status) {
                return;
            }
            self.apply(&mut state, status, message, fraction)
        };
        self.notify(&update);
    }

    fn set_engine(&self, engine: &str) {
        self.state().engine = engine.into();
    }

    /// The job's end, unless it was cancelled first.
    fn finish(&self, outcome: Result<Vec<GeneratedImage>>) {
        let update = {
            let mut state = self.state();
            if is_over(&state.status) {
                return;
            }
            state.finished = Some(Instant::now());
            match outcome {
                Ok(images) => {
                    state.images = images;
                    self.apply(&mut state, "done", "", Some(1.0))
                }
                Err(e) => {
                    let e = match e {
                        e @ Error::Tool { .. } => e,
                        other => tool_error(format!("image generation failed: {other}")),
                    };
                    let message = e.to_string();
                    state.error = Some(e);
                    self.apply(&mut state, "failed", &message, None)
                }
            }
        };
        self.notify(&update);
        self.inner.done.send_replace(true);
    }
}

fn is_over(status: &str) -> bool {
    matches!(status, "done" | "failed" | "cancelled")
}

fn elapsed(started: Instant, finished: Option<Instant>) -> f64 {
    finished.unwrap_or_else(Instant::now).duration_since(started).as_secs_f64()
}

// ----------------------------------------------------------------------
// engines
// ----------------------------------------------------------------------

fn redact(text: &str, secrets: &[String]) -> String {
    let mut text = text.to_string();
    for secret in secrets {
        if secret.chars().count() >= 4 {
            text = text.replace(secret.as_str(), "***");
        }
    }
    text
}

/// One image service.
///
/// Implement it for a service that is not here: `generate` takes the request
/// and an HTTP client and returns the images (as [`GeneratedImage::new`] of
/// their bytes is enough). Report progress through `report`. The generator
/// brings the retries, the fallback, the references and the jobs.
#[async_trait]
pub trait ImageEngine: Send + Sync {
    fn name(&self) -> String;
    fn model(&self) -> String;
    /// How many reference images one request may carry; 0 for none.
    fn max_references(&self) -> usize;
    /// Empty when the engine can be called; otherwise what is missing.
    fn ready(&self) -> String {
        String::new()
    }
    /// What must never appear in a message: the key.
    fn secrets(&self) -> Vec<String> {
        vec![]
    }
    async fn generate(&self, request: &ImageRequest, client: &reqwest::Client, report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure>;
}

/// What a request carries.
enum Body {
    None,
    Json(Value),
    Multipart(String, Vec<u8>),
}

/// What every HTTP engine has: a key (read from the environment when it is
/// used, not when the engine is built), a model, where it is, extra headers,
/// and options sent with every request.
#[derive(Clone)]
struct Http {
    env: &'static [&'static str],
    url: &'static str,
    api_key: Option<String>,
    model: String,
    base_url: Option<String>,
    headers: IndexMap<String, String>,
    options: Map<String, Value>,
}

impl Http {
    fn new(env: &'static [&'static str], url: &'static str, model: &str) -> Self {
        Self { env, url, api_key: None, model: model.into(), base_url: None, headers: IndexMap::new(), options: Map::new() }
    }

    fn key(&self) -> String {
        if let Some(k) = self.api_key.as_ref().filter(|k| !k.is_empty()) {
            return k.clone();
        }
        self.env.iter().find_map(|v| env(v)).unwrap_or_default()
    }

    fn base(&self) -> String {
        self.base_url.as_deref().unwrap_or(self.url).trim_end_matches('/').to_string()
    }

    fn ready(&self) -> String {
        if self.key().is_empty() {
            let needs = if self.env.is_empty() { "api_key=".to_string() } else { self.env.join(" or ") };
            return format!("no API key — set {needs}");
        }
        String::new()
    }

    /// The failure an HTTP answer amounts to; `None` for a good one.
    fn failure(status: u16, headers: &reqwest::header::HeaderMap, text: &str) -> Option<ImageFailure> {
        if (200..300).contains(&status) {
            return None;
        }
        let detail = head(&squash(&head(text, 600)), 300);
        if matches!(status, 401 | 403) && !REFUSED.is_match(&detail) {
            return Some(ImageFailure::new("the API key was refused").status(status));
        }
        if status == 402 || (status == 429 && QUOTA.is_match(&detail)) {
            return Some(ImageFailure::new("out of quota or credit").status(status));
        }
        if status == 429 {
            return Some(ImageFailure::new("rate limited").retryable().status(status).wait(retry_after(Some(headers), Some(&detail), true)));
        }
        if status == 408 || status >= 500 {
            return Some(ImageFailure::new(format!("the engine answered {status}")).retryable().status(status).wait(retry_after(Some(headers), None, false)));
        }
        if REFUSED.is_match(&detail) {
            return Some(ImageFailure::new(format!("the provider refused this: {detail}")).refused().status(status));
        }
        Some(ImageFailure::new(format!("the request was rejected ({status}): {detail}")).healthy().status(status))
    }

    fn unreachable(e: &reqwest::Error) -> ImageFailure {
        if e.is_timeout() {
            return ImageFailure::new("timed out").retryable();
        }
        let kind = if e.is_connect() { "ConnectError" } else { "RequestError" };
        ImageFailure::new(format!("could not be reached ({kind})")).retryable()
    }

    /// One HTTP request; its JSON back. A bad answer is an `ImageFailure`.
    async fn send(
        &self,
        client: &reqwest::Client,
        method: reqwest::Method,
        url: &str,
        headers: &[(&str, String)],
        body: Body,
    ) -> std::result::Result<Value, ImageFailure> {
        let mut request = client.request(method, url);
        for (k, v) in &self.headers {
            request = request.header(k.as_str(), v.as_str());
        }
        for (k, v) in headers {
            request = request.header(*k, v.as_str());
        }
        request = match body {
            Body::None => request,
            Body::Json(v) => request.header("content-type", "application/json").body(serde_json::to_vec(&v).unwrap_or_default()),
            Body::Multipart(boundary, bytes) => request.header("content-type", format!("multipart/form-data; boundary={boundary}")).body(bytes),
        };
        let response = request.send().await.map_err(|e| Self::unreachable(&e))?;
        let status = response.status().as_u16();
        let response_headers = response.headers().clone();
        let text = response.text().await.map_err(|e| Self::unreachable(&e))?;
        if let Some(problem) = Self::failure(status, &response_headers, &text) {
            return Err(problem);
        }
        serde_json::from_str(&text).map_err(|_| ImageFailure::new("answered with something that is not JSON").retryable())
    }

    /// Fetch a finished image from where the engine left it.
    async fn download(&self, client: &reqwest::Client, url: &str) -> std::result::Result<Vec<u8>, ImageFailure> {
        let mut last = ImageFailure::new("the finished image could not be fetched");
        for attempt in 0..3u32 {
            let mut target = url.to_string();
            let mut answer = None;
            // Redirects are followed here: a finished image may live anywhere.
            for _ in 0..5 {
                match client.get(&target).send().await {
                    Ok(r) if r.status().is_redirection() => {
                        let location = r.headers().get("location").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                        target = reqwest::Url::parse(&target).and_then(|b| b.join(&location)).map(|u| u.to_string()).unwrap_or(location);
                    }
                    Ok(r) => {
                        answer = Some(r);
                        break;
                    }
                    Err(e) => {
                        last = Self::unreachable(&e);
                        break;
                    }
                }
            }
            if let Some(r) = answer {
                let status = r.status().as_u16();
                if let Ok(bytes) = r.bytes().await
                    && status == 200
                    && !bytes.is_empty()
                {
                    return Ok(bytes.to_vec());
                }
                last = ImageFailure::new(format!("the finished image could not be fetched ({status})")).retryable();
            }
            tokio::time::sleep(Duration::from_secs_f64(0.5 * f64::from(attempt + 1))).await;
        }
        Err(last)
    }
}

macro_rules! http_builders {
    ($t:ty) => {
        impl $t {
            /// The API key. Left out, the environment's, read when the engine is used.
            pub fn with_key(mut self, key: impl Into<String>) -> Self {
                self.http.api_key = Some(key.into());
                self
            }

            pub fn with_model(mut self, model: impl Into<String>) -> Self {
                self.http.model = model.into();
                self
            }

            /// Somewhere else that speaks the same API.
            pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
                self.http.base_url = Some(url.into());
                self
            }

            /// A header sent with every request of this engine.
            pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
                self.http.headers.insert(name.into(), value.into());
                self
            }

            /// An option sent with every request of this engine (`quality`, a model's own).
            pub fn with_option(mut self, name: impl Into<String>, value: Value) -> Self {
                self.http.options.insert(name.into(), value);
                self
            }
        }
    };
}

/// Gemini's image models ("Nano Banana"), and Imagen, through the Gemini API.
///
/// Reference images are sent with the prompt: the model edits them, combines
/// them, or keeps to their style. An `imagen-…` model is text-to-image only.
#[derive(Clone)]
pub struct GeminiImages {
    http: Http,
}

http_builders!(GeminiImages);

/// The finish reasons that mean the content was declined.
const DECLINED: [&str; 5] = ["SAFETY", "PROHIBITED", "BLOCKLIST", "RECITATION", "SPII"];

impl Default for GeminiImages {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiImages {
    pub const ENV: &'static [&'static str] = &["GEMINI_API_KEY", "GOOGLE_API_KEY"];
    pub const DEFAULT_MODEL: &'static str = "gemini-2.5-flash-image";

    pub fn new() -> Self {
        Self { http: Http::new(Self::ENV, "https://generativelanguage.googleapis.com/v1beta", Self::DEFAULT_MODEL) }
    }

    fn auth(&self) -> [(&'static str, String); 1] {
        [("x-goog-api-key", self.http.key())]
    }

    async fn one(&self, request: &ImageRequest, client: &reqwest::Client, n: u32) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        let mut parts = vec![json!({"text": request.prompt})];
        parts.extend(request.references.iter().map(|r| json!({"inlineData": {"mimeType": r.media_type, "data": r.b64()}})));
        let mut config = Map::new();
        config.insert("responseModalities".into(), json!(["TEXT", "IMAGE"]));
        let mut shape = Map::new();
        if !request.aspect_ratio.is_empty() {
            shape.insert("aspectRatio".into(), json!(request.aspect_ratio));
        }
        if !request.size.is_empty() {
            shape.insert("imageSize".into(), json!(request.size));
        }
        if !shape.is_empty() {
            config.insert("imageConfig".into(), Value::Object(shape));
        }
        if let Some(seed) = request.seed {
            config.insert("seed".into(), json!(seed + i64::from(n)));
        }
        if let Some(Value::Object(extra)) = request.options.get("generation_config") {
            config.extend(extra.clone());
        }
        let url = format!("{}/models/{}:generateContent", self.http.base(), self.http.model);
        let body = json!({"contents": [{"role": "user", "parts": parts}], "generationConfig": config});
        let data = self.http.send(client, reqwest::Method::POST, &url, &self.auth(), Body::Json(body)).await?;
        if let Some(blocked) = data.get("promptFeedback").and_then(|f| f.get("blockReason")).filter(|b| !b.is_null()) {
            return Err(ImageFailure::new(format!("the provider refused this prompt ({})", py_str(blocked))).refused());
        }
        let candidate = data.get("candidates").and_then(|c| c.get(0)).filter(|c| c.is_object()).cloned().unwrap_or(json!({}));
        let found = candidate.get("content").and_then(|c| c.get("parts")).and_then(Value::as_array).cloned().unwrap_or_default();
        let said =
            found.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ").trim().to_string();
        let mut images = vec![];
        for part in &found {
            let Some(blob) = part.get("inlineData").or_else(|| part.get("inline_data")) else { continue };
            let Some(data) = blob.get("data").and_then(Value::as_str).filter(|d| !d.is_empty()) else { continue };
            let media_type = blob.get("mimeType").or_else(|| blob.get("mime_type")).and_then(Value::as_str).unwrap_or("image/png");
            images.push(GeneratedImage { media_type: media_type.into(), text: said.clone(), ..GeneratedImage::new(decode(data)?) });
        }
        if !images.is_empty() {
            return Ok(images);
        }
        let reason = py_str(candidate.get("finishReason").unwrap_or(&Value::Null));
        let reason = if candidate.get("finishReason").is_none_or(Value::is_null) { String::new() } else { reason };
        let said_part = if said.is_empty() { String::new() } else { format!(": {}", head(&said, 200)) };
        if DECLINED.iter().any(|mark| reason.contains(mark)) {
            return Err(ImageFailure::new(format!("the provider refused this ({reason}){said_part}")).refused());
        }
        let reason_part = if reason.is_empty() { String::new() } else { format!(" ({reason})") };
        Err(ImageFailure::new(format!("the model answered without an image{reason_part}{said_part}")).retryable().healthy())
    }

    async fn imagen(&self, request: &ImageRequest, client: &reqwest::Client) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        if !request.references.is_empty() {
            return Err(ImageFailure::new(format!("{} takes no reference images", self.http.model)).healthy());
        }
        let mut parameters = Map::new();
        parameters.insert("sampleCount".into(), json!(request.count.min(4)));
        if !request.aspect_ratio.is_empty() {
            parameters.insert("aspectRatio".into(), json!(request.aspect_ratio));
        }
        let url = format!("{}/models/{}:predict", self.http.base(), self.http.model);
        let body = json!({"instances": [{"prompt": request.prompt}], "parameters": parameters});
        let data = self.http.send(client, reqwest::Method::POST, &url, &self.auth(), Body::Json(body)).await?;
        let mut made = vec![];
        for p in data.get("predictions").and_then(Value::as_array).into_iter().flatten() {
            if let Some(b) = p.get("bytesBase64Encoded").and_then(Value::as_str).filter(|b| !b.is_empty()) {
                let media_type = p.get("mimeType").and_then(Value::as_str).unwrap_or("image/png");
                made.push(GeneratedImage { media_type: media_type.into(), ..GeneratedImage::new(decode(b)?) });
            }
        }
        if made.is_empty() {
            return Err(ImageFailure::new("the provider refused this prompt, or filtered every image it made").refused());
        }
        Ok(made)
    }
}

#[async_trait]
impl ImageEngine for GeminiImages {
    fn name(&self) -> String {
        "gemini".into()
    }
    fn model(&self) -> String {
        self.http.model.clone()
    }
    fn max_references(&self) -> usize {
        14
    }
    fn ready(&self) -> String {
        self.http.ready()
    }
    fn secrets(&self) -> Vec<String> {
        vec![self.http.key()]
    }

    async fn generate(&self, request: &ImageRequest, client: &reqwest::Client, _report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        if self.http.model.starts_with("imagen") {
            return self.imagen(request, client).await;
        }
        // One picture a call: ask as many times as pictures are wanted.
        let made = join_all((0..request.count).map(|n| self.one(request, client, n))).await;
        let mut out = vec![];
        for batch in made {
            out.extend(batch?);
        }
        out.truncate(request.count as usize);
        Ok(out)
    }
}

/// OpenAI's image API — or anything that speaks it (`with_base_url`).
///
/// With reference images the request goes to the edits endpoint, which takes
/// several for `gpt-image-1`.
#[derive(Clone)]
pub struct OpenAIImages {
    http: Http,
}

http_builders!(OpenAIImages);

impl Default for OpenAIImages {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIImages {
    pub const ENV: &'static [&'static str] = &["OPENAI_API_KEY"];
    pub const DEFAULT_MODEL: &'static str = "gpt-image-1";

    pub fn new() -> Self {
        Self { http: Http::new(Self::ENV, "https://api.openai.com/v1", Self::DEFAULT_MODEL) }
    }

    fn dimensions(&self, ratio: &str) -> String {
        let model = &self.http.model;
        if ratio.is_empty() {
            return (if model.starts_with("dall-e") { "1024x1024" } else { "auto" }).into();
        }
        let (w, h) = ratio.split_once(':').unwrap_or(("1", "1"));
        let (w, h): (u32, u32) = (w.parse().unwrap_or(1), h.parse().unwrap_or(1));
        let shape = if w == h {
            "1:1"
        } else if w > h {
            "landscape"
        } else {
            "portrait"
        };
        let size = if model == "dall-e-3" {
            match shape {
                "1:1" => "1024x1024",
                "landscape" => "1792x1024",
                _ => "1024x1792",
            }
        } else {
            match shape {
                "1:1" => "1024x1024",
                "landscape" => "1536x1024",
                _ => "1024x1536",
            }
        };
        size.into()
    }
}

#[async_trait]
impl ImageEngine for OpenAIImages {
    fn name(&self) -> String {
        "openai".into()
    }
    fn model(&self) -> String {
        self.http.model.clone()
    }
    fn max_references(&self) -> usize {
        16
    }
    fn ready(&self) -> String {
        self.http.ready()
    }
    fn secrets(&self) -> Vec<String> {
        vec![self.http.key()]
    }

    async fn generate(&self, request: &ImageRequest, client: &reqwest::Client, _report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        let model = self.http.model.clone();
        let mut fields = Map::new();
        fields.insert("model".into(), json!(model));
        fields.insert("prompt".into(), json!(request.prompt));
        fields.insert("n".into(), json!(request.count));
        fields.insert("size".into(), json!(self.dimensions(&request.aspect_ratio)));
        fields.extend(self.http.options.clone());
        fields.extend(request.engine_options().map(|(k, v)| (k.clone(), v.clone())));
        if model.starts_with("dall-e") {
            fields.entry("response_format").or_insert_with(|| json!("b64_json"));
        }
        let auth = [("Authorization", format!("Bearer {}", self.http.key()))];
        let data = if !request.references.is_empty() {
            if model == "dall-e-3" {
                return Err(ImageFailure::new("dall-e-3 takes no reference images").healthy());
            }
            let mut parts: Vec<(String, String, Option<(String, String, Vec<u8>)>)> = fields.iter().map(|(k, v)| (k.clone(), py_str(v), None)).collect();
            for r in &request.references {
                let filename = format!("{}.{}", slug(&r.name, "reference"), extension_of(&r.media_type));
                parts.push(("image[]".into(), String::new(), Some((filename, r.media_type.clone(), r.data.clone()))));
            }
            let (boundary, bytes) = multipart(&parts);
            self.http.send(client, reqwest::Method::POST, &format!("{}/images/edits", self.http.base()), &auth, Body::Multipart(boundary, bytes)).await?
        } else {
            self.http
                .send(client, reqwest::Method::POST, &format!("{}/images/generations", self.http.base()), &auth, Body::Json(Value::Object(fields.clone())))
                .await?
        };
        let format = fields.get("output_format").map(py_str).unwrap_or_else(|| "png".into());
        let kind = format!("image/{format}").replace("jpg", "jpeg");
        let mut made = vec![];
        for item in data.get("data").and_then(Value::as_array).into_iter().flatten() {
            let text = item.get("revised_prompt").and_then(Value::as_str).unwrap_or("").to_string();
            if let Some(b) = item.get("b64_json").and_then(Value::as_str).filter(|b| !b.is_empty()) {
                made.push(GeneratedImage { media_type: kind.clone(), text, ..GeneratedImage::new(decode(b)?) });
            } else if let Some(url) = item.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()) {
                made.push(GeneratedImage { text, ..GeneratedImage::new(self.http.download(client, url).await?) });
            }
        }
        if made.is_empty() {
            return Err(ImageFailure::new("the engine answered without an image").retryable());
        }
        Ok(made)
    }
}

/// Any image model hosted on Replicate — FLUX, Ideogram, Seedream, the rest.
///
/// A prediction is started and then watched: this is the engine whose progress
/// is the provider's own. `reference_key` is the name the model gives its
/// image input (`input_image`, `image`, `image_input`); `reference_list` for a
/// model that takes several. `inputs` are sent with every request.
#[derive(Clone)]
pub struct ReplicateImages {
    http: Http,
    reference_key: String,
    reference_list: bool,
    inputs: Map<String, Value>,
    poll_interval: f64,
}

http_builders!(ReplicateImages);

impl Default for ReplicateImages {
    fn default() -> Self {
        Self::new()
    }
}

/// Stops a prediction at the provider when the wait for it ends any way but
/// success — a failure, a timeout, a cancelled job.
struct StopOnDrop {
    client: reqwest::Client,
    url: Option<String>,
    auth: String,
    armed: bool,
}

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let (Some(url), Ok(runtime)) = (self.url.take(), tokio::runtime::Handle::try_current()) {
            let request = self.client.post(url).header("Authorization", self.auth.clone());
            runtime.spawn(async move {
                let _ = request.send().await;
            });
        }
    }
}

impl ReplicateImages {
    pub const ENV: &'static [&'static str] = &["REPLICATE_API_TOKEN", "REPLICATE_API_KEY"];
    pub const DEFAULT_MODEL: &'static str = "black-forest-labs/flux-kontext-pro";

    pub fn new() -> Self {
        Self {
            http: Http::new(Self::ENV, "https://api.replicate.com/v1", Self::DEFAULT_MODEL),
            reference_key: "input_image".into(),
            reference_list: false,
            inputs: Map::new(),
            poll_interval: 1.5,
        }
    }

    /// The name the model gives its image input.
    pub fn with_reference_key(mut self, key: impl Into<String>) -> Self {
        self.reference_key = key.into();
        self
    }

    /// The model takes a list of reference images (up to 10).
    pub fn with_reference_list(mut self, yes: bool) -> Self {
        self.reference_list = yes;
        self
    }

    /// An input sent with every request.
    pub fn with_input(mut self, name: impl Into<String>, value: Value) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    /// Seconds between status checks (it grows to at most 5).
    pub fn with_poll_interval(mut self, seconds: f64) -> Self {
        self.poll_interval = seconds;
        self
    }

    async fn one(&self, request: &ImageRequest, client: &reqwest::Client, report: &Reporter, n: u32) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        let token = format!("Bearer {}", self.http.key());
        let auth = [("Authorization", token.clone())];
        let mut inputs = Map::new();
        inputs.insert("prompt".into(), json!(request.prompt));
        inputs.extend(self.inputs.clone());
        if !request.aspect_ratio.is_empty() {
            inputs.insert("aspect_ratio".into(), json!(request.aspect_ratio));
        }
        if let Some(seed) = request.seed {
            inputs.insert("seed".into(), json!(seed + i64::from(n)));
        }
        if !request.references.is_empty() {
            let uris: Vec<String> = request.references.iter().map(Reference::data_uri).collect();
            inputs.insert(self.reference_key.clone(), if self.reference_list { json!(uris) } else { json!(uris[0]) });
        }
        inputs.extend(request.engine_options().map(|(k, v)| (k.clone(), v.clone())));
        let base = self.http.base();
        let started = match self.http.model.split_once(':') {
            Some((_, version)) if !version.is_empty() => {
                self.http
                    .send(client, reqwest::Method::POST, &format!("{base}/predictions"), &auth, Body::Json(json!({"version": version, "input": inputs})))
                    .await?
            }
            _ => {
                let name = self.http.model.split(':').next().unwrap_or("");
                self.http.send(client, reqwest::Method::POST, &format!("{base}/models/{name}/predictions"), &auth, Body::Json(json!({"input": inputs}))).await?
            }
        };
        let urls = started.get("urls").cloned().unwrap_or(Value::Null);
        let watch_url = urls
            .get("get")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{base}/predictions/{}", py_str(started.get("id").unwrap_or(&Value::Null))));
        let mut stop = StopOnDrop { client: client.clone(), url: urls.get("cancel").and_then(Value::as_str).map(str::to_string), auth: token, armed: true };
        let mut state = started;
        let mut misses = 0;
        let mut wait = self.poll_interval;
        loop {
            let status = state.get("status").map(py_str).unwrap_or_default();
            if status == "succeeded" {
                break;
            }
            if status == "failed" || status == "canceled" {
                let error = state.get("error").filter(|e| !e.is_null() && e.as_str() != Some("")).map(py_str).unwrap_or_else(|| status.clone());
                let why = head(&squash(&error), 300);
                let refused = REFUSED.is_match(&why);
                let failure = if refused {
                    ImageFailure::new(format!("the provider refused this: {why}")).refused()
                } else {
                    ImageFailure::new(format!("the prediction {status}: {why}")).healthy()
                };
                return Err(failure);
            }
            let logs = state.get("logs").map(py_str).unwrap_or_default();
            let tail: String = {
                let chars: Vec<char> = logs.chars().collect();
                chars[chars.len().saturating_sub(400)..].iter().collect()
            };
            let fraction = PERCENT.captures_iter(&tail).last().and_then(|c| c[1].parse::<u32>().ok()).map(|p| f64::from(p.min(100)) / 100.0);
            let shown = if status.is_empty() { "starting" } else { status.as_str() };
            report(shown, "", fraction);
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            wait = (wait * 1.3).min(5.0);
            match self.http.send(client, reqwest::Method::GET, &watch_url, &auth, Body::None).await {
                Ok(s) => {
                    state = s;
                    misses = 0;
                }
                Err(e) => {
                    // A status check that fails says nothing about the picture.
                    misses += 1;
                    if !e.retryable || misses >= 5 {
                        return Err(e);
                    }
                }
            }
        }
        stop.armed = false;
        let links: Vec<String> = match state.get("output") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(items)) => items.iter().filter_map(|o| o.as_str().map(str::to_string)).collect(),
            _ => vec![],
        };
        if links.is_empty() {
            return Err(ImageFailure::new("the prediction finished without an image"));
        }
        report("downloading", "", Some(1.0));
        let mut made = vec![];
        for link in links {
            made.push(GeneratedImage::new(self.http.download(client, &link).await?));
        }
        Ok(made)
    }
}

#[async_trait]
impl ImageEngine for ReplicateImages {
    fn name(&self) -> String {
        "replicate".into()
    }
    fn model(&self) -> String {
        self.http.model.clone()
    }
    fn max_references(&self) -> usize {
        if self.reference_list { 10 } else { 1 }
    }
    fn ready(&self) -> String {
        self.http.ready()
    }
    fn secrets(&self) -> Vec<String> {
        vec![self.http.key()]
    }

    async fn generate(&self, request: &ImageRequest, client: &reqwest::Client, report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        let made = join_all((0..request.count).map(|n| self.one(request, client, report, n))).await;
        let mut out = vec![];
        for batch in made {
            out.extend(batch?);
        }
        out.truncate(request.count as usize);
        Ok(out)
    }
}

/// An engine that draws without a network or a key — for tests and examples.
///
/// Each image is a small PNG of one colour, chosen by the prompt, so the same
/// prompt gives the same picture. `steps` and `delay` make a job take time and
/// report progress; `fail` is returned instead, once per entry.
pub struct FakeImages {
    name: String,
    steps: u32,
    delay: f64,
    side: u32,
    fail: Mutex<Vec<ImageFailure>>,
    requests: Mutex<Vec<ImageRequest>>,
}

impl Default for FakeImages {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeImages {
    pub fn new() -> Self {
        Self { name: "fake".into(), steps: 0, delay: 0.0, side: 64, fail: Mutex::new(vec![]), requests: Mutex::new(vec![]) }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Report progress this many times, `delay` seconds apart.
    pub fn with_steps(mut self, steps: u32, delay: f64) -> Self {
        self.steps = steps;
        self.delay = delay;
        self
    }

    /// The longer side of what it draws, in pixels.
    pub fn with_side(mut self, side: u32) -> Self {
        self.side = side;
        self
    }

    /// Failures returned instead of images, one per request, in order.
    pub fn with_failures(self, fail: Vec<ImageFailure>) -> Self {
        *lock(&self.fail) = fail;
        self
    }

    /// Every request this engine was given, in order.
    pub fn requests(&self) -> Vec<ImageRequest> {
        lock(&self.requests).clone()
    }
}

#[async_trait]
impl ImageEngine for FakeImages {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn model(&self) -> String {
        "fake-image-1".into()
    }
    fn max_references(&self) -> usize {
        8
    }

    async fn generate(&self, request: &ImageRequest, _client: &reqwest::Client, report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        lock(&self.requests).push(request.clone());
        {
            let mut fail = lock(&self.fail);
            if !fail.is_empty() {
                return Err(fail.remove(0));
            }
        }
        for step in 0..self.steps {
            report("processing", "", Some(f64::from(step) / f64::from(self.steps)));
            tokio::time::sleep(Duration::from_secs_f64(self.delay)).await;
        }
        let ratio = if request.aspect_ratio.is_empty() { "1:1" } else { request.aspect_ratio.as_str() };
        let (w, h) = ratio.split_once(':').unwrap_or(("1", "1"));
        let (w, h): (f64, f64) = (w.parse().unwrap_or(1.0), h.parse().unwrap_or(1.0));
        let scale = f64::from(self.side) / w.max(h);
        let width = ((w * scale).round_ties_even() as u32).max(1);
        let height = ((h * scale).round_ties_even() as u32).max(1);
        let seed = request.seed.map(|s| s.to_string()).unwrap_or_else(|| "None".into());
        let mut made = vec![];
        for n in 0..request.count {
            let colour = Sha256::digest(format!("{}|{}|{seed}|{n}", request.prompt, request.references.len()).as_bytes());
            made.push(GeneratedImage::new(solid_png(width, height, [colour[0], colour[1], colour[2]])));
        }
        Ok(made)
    }
}

type MakeFn = Arc<dyn Fn(ImageRequest) -> BoxFuture<'static, Result<Vec<GeneratedImage>>> + Send + Sync>;

/// A function of your own as an engine: `request → images`.
struct FunctionEngine {
    name: String,
    f: MakeFn,
}

#[async_trait]
impl ImageEngine for FunctionEngine {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn model(&self) -> String {
        self.name.clone()
    }
    fn max_references(&self) -> usize {
        16
    }

    async fn generate(&self, request: &ImageRequest, _client: &reqwest::Client, _report: &Reporter) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        (self.f)(request.clone()).await.map_err(|e| ImageFailure::new(head(&format!("{}: {e}", e.kind_name()), 300)))
    }
}

/// An engine for an [`ImageGenerator`]: a name (`"gemini"`, `"flux"`, …), an
/// engine of your own, or a function.
#[derive(Clone)]
pub enum EngineSpec {
    Name(String),
    Engine(Arc<dyn ImageEngine>),
}

impl EngineSpec {
    /// A function of your own as an engine.
    pub fn function<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(ImageRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Vec<GeneratedImage>>> + Send + 'static,
    {
        let name = name.into();
        Self::Engine(Arc::new(FunctionEngine { name: if name.is_empty() { "custom".into() } else { name }, f: Arc::new(move |r| f(r).boxed()) }))
    }
}

impl From<&str> for EngineSpec {
    fn from(s: &str) -> Self {
        Self::Name(s.into())
    }
}

impl From<String> for EngineSpec {
    fn from(s: String) -> Self {
        Self::Name(s)
    }
}

impl From<Arc<dyn ImageEngine>> for EngineSpec {
    fn from(e: Arc<dyn ImageEngine>) -> Self {
        Self::Engine(e)
    }
}

impl<T: ImageEngine + 'static> From<Arc<T>> for EngineSpec {
    fn from(e: Arc<T>) -> Self {
        Self::Engine(e)
    }
}

impl From<GeminiImages> for EngineSpec {
    fn from(e: GeminiImages) -> Self {
        Self::Engine(Arc::new(e))
    }
}

impl From<OpenAIImages> for EngineSpec {
    fn from(e: OpenAIImages) -> Self {
        Self::Engine(Arc::new(e))
    }
}

impl From<ReplicateImages> for EngineSpec {
    fn from(e: ReplicateImages) -> Self {
        Self::Engine(Arc::new(e))
    }
}

impl From<FakeImages> for EngineSpec {
    fn from(e: FakeImages) -> Self {
        Self::Engine(Arc::new(e))
    }
}

/// In the order they are preferred when the choice is left to the environment.
const ENGINES: [&str; 3] = ["gemini", "openai", "replicate"];
const NAMES: [(&str, &str); 8] = [
    ("google", "gemini"),
    ("nano-banana", "gemini"),
    ("nanobanana", "gemini"),
    ("imagen", "gemini"),
    ("gpt-image", "openai"),
    ("dall-e", "openai"),
    ("dalle", "openai"),
    ("flux", "replicate"),
];

fn named_engine(spec: &str, api_key: Option<&str>, model: Option<&str>) -> Result<Arc<dyn ImageEngine>> {
    let name = spec.trim().to_lowercase();
    let name = NAMES.iter().find(|(alias, _)| *alias == name).map(|(_, n)| n.to_string()).unwrap_or(name);
    macro_rules! build {
        ($e:expr) => {{
            let mut e = $e;
            if let Some(k) = api_key {
                e = e.with_key(k);
            }
            if let Some(m) = model {
                e = e.with_model(m);
            }
            Ok(Arc::new(e) as Arc<dyn ImageEngine>)
        }};
    }
    match name.as_str() {
        "gemini" => build!(GeminiImages::new()),
        "openai" => build!(OpenAIImages::new()),
        "replicate" => build!(ReplicateImages::new()),
        _ => Err(Error::config(format!("no image engine named {}; there are: {}", py_repr(spec), ENGINES.join(", ")))),
    }
}

/// One engine that ships, and whether this environment can call it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EngineInfo {
    pub name: String,
    pub ready: bool,
    /// What is missing; empty when it is ready.
    pub missing: String,
    /// The environment variables it reads its key from.
    pub needs: String,
    pub model: String,
}

/// Every engine that ships, and whether this environment can call it.
pub fn image_engines() -> Vec<EngineInfo> {
    let engines: [(Arc<dyn ImageEngine>, &[&str]); 3] = [
        (Arc::new(GeminiImages::new()), GeminiImages::ENV),
        (Arc::new(OpenAIImages::new()), OpenAIImages::ENV),
        (Arc::new(ReplicateImages::new()), ReplicateImages::ENV),
    ];
    engines
        .into_iter()
        .map(|(engine, env)| {
            let missing = engine.ready();
            EngineInfo { name: engine.name(), ready: missing.is_empty(), missing, needs: env.join(" or "), model: engine.model() }
        })
        .collect()
}

// ----------------------------------------------------------------------
// the generator
// ----------------------------------------------------------------------

/// How an [`ImageGenerator`] is made. Everything has a default.
#[derive(Clone)]
pub struct ImageOptions {
    /// The engines to fall back through, in order. Empty: `$AGENT_HARNESS_IMAGES`,
    /// else every engine with a key set.
    pub engines: Vec<EngineSpec>,
    /// The key of a single named engine.
    pub api_key: Option<String>,
    /// The model of a single named engine.
    pub model: Option<String>,
    /// The shape when none is asked for: "1:1", "16:9", "wide", …
    pub aspect_ratio: String,
    /// The most images one request may ask for.
    pub max_count: u32,
    /// How many reference images, and how large each may be.
    pub max_references: usize,
    pub max_reference_bytes: usize,
    /// Seconds for one attempt at one engine.
    pub timeout: f64,
    /// Seconds for the whole job, retries and fallbacks included.
    pub deadline: f64,
    /// Tries after the first, per engine (or set `retry`).
    pub retries: u32,
    pub retry: Option<RetryPolicy>,
    /// The most images this generator will ever make; `None` for no limit. A
    /// guard against a loop that will not stop.
    pub max_images: Option<u64>,
    /// How many jobs run at once.
    pub max_concurrency: usize,
    /// USD, charged to the run's budget for each image made.
    pub cost_per_image: Option<f64>,
    /// Where images are saved when there is no workspace.
    pub output_dir: Option<PathBuf>,
    /// The folder of the workspace images go in.
    pub directory: String,
    /// Let a reference be fetched from a private address.
    pub allow_private: bool,
    /// Called with an [`ImageProgress`] as a job moves.
    pub on_progress: Option<ProgressFn>,
    /// Failures in a row before an engine is left alone, and for how long (seconds).
    pub failure_threshold: u32,
    pub cooldown: f64,
    /// An HTTP client of your own.
    pub client: Option<reqwest::Client>,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            engines: vec![],
            api_key: None,
            model: None,
            aspect_ratio: String::new(),
            max_count: 4,
            max_references: 8,
            max_reference_bytes: 20 * 1024 * 1024,
            timeout: 180.0,
            deadline: 600.0,
            retries: 2,
            retry: None,
            max_images: None,
            max_concurrency: 3,
            cost_per_image: None,
            output_dir: None,
            directory: "images".into(),
            allow_private: false,
            on_progress: None,
            failure_threshold: 3,
            cooldown: 60.0,
            client: None,
        }
    }
}

/// Something that names a reference image.
#[derive(Clone)]
pub enum ReferenceSource {
    Reference(Reference),
    Image(GeneratedImage),
    Bytes(Vec<u8>),
    Block(MediaBlock),
    /// A file, a URL, a `data:` URI, the name of an image made earlier, an
    /// attachment (`attachment:1`), or a file in the workspace.
    Text(String),
}

impl From<&str> for ReferenceSource {
    fn from(s: &str) -> Self {
        Self::Text(s.into())
    }
}

impl From<String> for ReferenceSource {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&Path> for ReferenceSource {
    fn from(p: &Path) -> Self {
        Self::Text(p.to_string_lossy().into_owned())
    }
}

impl From<PathBuf> for ReferenceSource {
    fn from(p: PathBuf) -> Self {
        Self::Text(p.to_string_lossy().into_owned())
    }
}

impl From<Vec<u8>> for ReferenceSource {
    fn from(b: Vec<u8>) -> Self {
        Self::Bytes(b)
    }
}

impl From<&[u8]> for ReferenceSource {
    fn from(b: &[u8]) -> Self {
        Self::Bytes(b.to_vec())
    }
}

impl From<GeneratedImage> for ReferenceSource {
    fn from(i: GeneratedImage) -> Self {
        Self::Image(i)
    }
}

impl From<&GeneratedImage> for ReferenceSource {
    fn from(i: &GeneratedImage) -> Self {
        Self::Image(i.clone())
    }
}

impl From<MediaBlock> for ReferenceSource {
    fn from(b: MediaBlock) -> Self {
        Self::Block(b)
    }
}

impl From<Reference> for ReferenceSource {
    fn from(r: Reference) -> Self {
        Self::Reference(r)
    }
}

/// One generation's settings.
#[derive(Clone)]
pub struct GenerateOptions {
    /// Paths, URLs, bytes, image blocks, or images made earlier.
    pub references: Vec<ReferenceSource>,
    /// `None` for the generator's default shape.
    pub aspect_ratio: Option<String>,
    pub count: u32,
    /// A resolution hint: "1K", "2K", "4K".
    pub size: String,
    pub seed: Option<i64>,
    /// A short file name for the result, without an extension.
    pub name: String,
    pub on_progress: Option<ProgressFn>,
    /// Passed to the engine as they are: `quality`, `background`, a model's own.
    pub options: Map<String, Value>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self { references: vec![], aspect_ratio: None, count: 1, size: String::new(), seed: None, name: String::new(), on_progress: None, options: Map::new() }
    }
}

/// How many images an engine made, how many times it failed, and the last reason.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct EngineStats {
    pub made: u64,
    pub failed: u64,
    pub last_error: String,
}

/// Who is asking: your code (which may name any file) or a model (which may
/// name only what it was shown), and what it was shown.
#[derive(Clone, Default)]
struct Context {
    trusted: bool,
    workspace: Option<PathBuf>,
    attachments: Vec<ContentBlock>,
    tool: Option<ToolContext>,
}

struct Inner {
    engines: Mutex<Option<Vec<Arc<dyn ImageEngine>>>>,
    aspect_ratio: String,
    max_count: u32,
    max_references: usize,
    max_reference_bytes: usize,
    timeout: f64,
    deadline: f64,
    retry: RetryPolicy,
    max_images: Option<u64>,
    gate: Arc<Semaphore>,
    cost_per_image: Option<f64>,
    output_dir: Option<PathBuf>,
    directory: String,
    allow_private: bool,
    on_progress: Option<ProgressFn>,
    failure_threshold: u32,
    cooldown: f64,
    client: reqwest::Client,
    breakers: Mutex<HashMap<String, CircuitBreaker>>,
    resolved: Mutex<HashMap<String, bool>>,
    jobs: Mutex<IndexMap<String, ImageJob>>,
    recent: Mutex<IndexMap<String, GeneratedImage>>,
    made: Mutex<u64>,
    stats: Mutex<IndexMap<String, EngineStats>>,
}

/// Image generation with reference images, jobs, retries and fallback.
/// Cheap to clone: clones share the jobs, the engines and their health.
#[derive(Clone)]
pub struct ImageGenerator {
    inner: Arc<Inner>,
}

impl fmt::Debug for ImageGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let engines = lock(&self.inner.engines).as_ref().map(|e| e.iter().map(|e| e.name()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        write!(f, "<ImageGenerator {}>", if engines.is_empty() { "auto" } else { &engines })
    }
}

/// Undoes a half-open circuit's probe when a job is cancelled mid-call.
struct ProbeGuard(Option<CircuitBreaker>);

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        if let Some(breaker) = self.0.take()
            && breaker.state() == CircuitState::HalfOpen
        {
            breaker.record_failure();
        }
    }
}

/// Cancels a job when the call waiting for it is dropped.
struct CancelOnDrop(Option<ImageJob>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(job) = self.0.take() {
            job.cancel();
        }
    }
}

fn ratio(value: &str) -> Result<String> {
    let text = value.trim().to_lowercase().replace(['x', '/'], ":");
    let text = ALIASES.iter().find(|(a, _)| *a == text).map(|(_, r)| r.to_string()).unwrap_or(text);
    if !text.is_empty() && !RATIOS.contains(&text.as_str()) {
        return Err(tool_error(format!("aspect_ratio is one of {} — got {}", RATIOS.join(", "), py_repr(value))));
    }
    Ok(text)
}

/// A path inside a workspace: relative, and never above it.
fn inside(root: &Path, relative: &str) -> Option<PathBuf> {
    let path = Path::new(relative);
    if path.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
        return None;
    }
    Some(root.join(path))
}

fn block_label(block: &MediaBlock) -> String {
    if !block.name.is_empty() {
        return block.name.clone();
    }
    block.path.clone().or_else(|| block.url.clone()).unwrap_or_else(|| block.media_type.clone())
}

impl ImageGenerator {
    pub fn new(options: ImageOptions) -> Result<Self> {
        if (options.api_key.is_some() || options.model.is_some()) && options.engines.len() != 1 {
            return Err(Error::config(
                "api_key= and model= go with one named engine; for several, pass each as an engine of its own — GeminiImages::new().with_model(...)",
            ));
        }
        let engines = if options.engines.is_empty() {
            None
        } else {
            let mut made = vec![];
            for spec in &options.engines {
                made.push(match spec {
                    EngineSpec::Engine(e) => e.clone(),
                    EngineSpec::Name(n) => named_engine(n, options.api_key.as_deref(), options.model.as_deref())?,
                });
            }
            Some(made)
        };
        let aspect_ratio = ratio(&options.aspect_ratio)?;
        let retry = options.retry.clone().unwrap_or(RetryPolicy {
            max_retries: options.retries,
            initial_delay: 1.0,
            max_delay: 15.0,
            max_retry_after: 30.0,
            ..RetryPolicy::default()
        });
        let client = match options.client.clone() {
            Some(c) => c,
            None => {
                crate::llm_providers::base::install_tls();
                reqwest::Client::builder()
                    .timeout(Duration::from_secs_f64(options.timeout.max(0.01)))
                    .redirect(reqwest::redirect::Policy::none())
                    .user_agent(format!("agent-harness/{}", crate::VERSION))
                    .build()
                    .map_err(|e| Error::config(format!("no HTTP client for images: {e}")))?
            }
        };
        let directory = options.directory.trim_matches('/').to_string();
        Ok(Self {
            inner: Arc::new(Inner {
                engines: Mutex::new(engines),
                aspect_ratio,
                max_count: options.max_count.max(1),
                max_references: options.max_references,
                max_reference_bytes: options.max_reference_bytes,
                timeout: options.timeout,
                deadline: options.deadline,
                retry,
                max_images: options.max_images,
                gate: Arc::new(Semaphore::new(options.max_concurrency.max(1))),
                cost_per_image: options.cost_per_image,
                output_dir: options.output_dir.map(|p| expand_user(&p.to_string_lossy())),
                directory: if directory.is_empty() { "images".into() } else { directory },
                allow_private: options.allow_private,
                on_progress: options.on_progress,
                failure_threshold: options.failure_threshold,
                cooldown: options.cooldown,
                client,
                breakers: Mutex::new(HashMap::new()),
                resolved: Mutex::new(HashMap::new()),
                jobs: Mutex::new(IndexMap::new()),
                recent: Mutex::new(IndexMap::new()),
                made: Mutex::new(0),
                stats: Mutex::new(IndexMap::new()),
            }),
        })
    }

    // ---- engines --------------------------------------------------------------
    fn resolve(&self) -> Result<Vec<Arc<dyn ImageEngine>>> {
        let mut engines = lock(&self.inner.engines);
        if engines.is_none() {
            let named: Vec<String> = env(ENGINE_ENV).unwrap_or_default().split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect();
            let found = if !named.is_empty() {
                named.iter().map(|n| named_engine(n, None, None)).collect::<Result<Vec<_>>>()?
            } else {
                ENGINES.iter().filter_map(|n| named_engine(n, None, None).ok()).filter(|e| e.ready().is_empty()).collect()
            };
            *engines = Some(found);
        }
        Ok(engines.clone().unwrap_or_default())
    }

    /// The engines a generation goes through, in order.
    pub fn engines(&self) -> Result<Vec<String>> {
        Ok(self.resolve()?.iter().map(|e| e.name()).collect())
    }

    /// The model of each engine, in order.
    pub fn models(&self) -> Result<Vec<String>> {
        Ok(self.resolve()?.iter().map(|e| e.model()).collect())
    }

    fn breaker(&self, name: &str) -> CircuitBreaker {
        lock(&self.inner.breakers)
            .entry(name.into())
            .or_insert_with(|| CircuitBreaker::new(self.inner.failure_threshold, Duration::from_secs_f64(self.inner.cooldown.max(0.0))))
            .clone()
    }

    fn count(&self, name: &str, made: bool, error: &str) {
        let mut stats = lock(&self.inner.stats);
        let entry = stats.entry(name.into()).or_default();
        if made {
            entry.made += 1;
        } else {
            entry.failed += 1;
        }
        if !error.is_empty() {
            entry.last_error = error.into();
        }
    }

    /// Per engine: images made, failures, the last reason.
    pub fn stats(&self) -> IndexMap<String, EngineStats> {
        lock(&self.inner.stats).clone()
    }

    /// How many images have been made.
    pub fn made(&self) -> u64 {
        *lock(&self.inner.made)
    }

    /// The jobs of this generator, newest last (the last 64).
    pub fn jobs(&self) -> Vec<ImageJob> {
        lock(&self.inner.jobs).values().cloned().collect()
    }

    /// The newest images (the last 24) — any can be the next one's reference.
    pub fn recent(&self) -> Vec<GeneratedImage> {
        lock(&self.inner.recent).values().cloned().collect()
    }

    /// A job by its id.
    pub fn job(&self, job_id: &str) -> Result<ImageJob> {
        let jobs = lock(&self.inner.jobs);
        if let Some(job) = jobs.get(job_id.trim()) {
            return Ok(job.clone());
        }
        let ids: Vec<&str> = jobs.keys().map(String::as_str).collect();
        let known = if ids.is_empty() { "none".into() } else { ids[ids.len().saturating_sub(5)..].join(", ") };
        Err(Error::tool(format!("there is no job {}; the latest are: {known}", py_repr(job_id)), "image_status"))
    }

    // ---- the request ------------------------------------------------------------
    fn check(&self, prompt: &str, count: i64) -> Result<(String, u32)> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(tool_error("the prompt is empty — describe the image to make"));
        }
        let length = prompt.chars().count();
        if length > MAX_PROMPT {
            return Err(tool_error(format!("the prompt is {length} characters; the most is {MAX_PROMPT}")));
        }
        let asked = if count == 0 { 1 } else { count };
        let mut wanted = asked.clamp(1, i64::from(self.inner.max_count)) as u64;
        if let Some(limit) = self.inner.max_images {
            let pending: u64 = lock(&self.inner.jobs).values().filter(|j| !j.done()).map(|j| u64::from(j.count())).sum();
            let used = self.made() + pending;
            if used >= limit {
                return Err(tool_error(format!("this generator's limit of {limit} images is used up")));
            }
            wanted = wanted.min(limit - used);
        }
        Ok((prompt.to_string(), wanted as u32))
    }

    // ---- references -------------------------------------------------------------
    fn accept(&self, data: Vec<u8>, name: &str) -> Result<Reference> {
        if data.len() > self.inner.max_reference_bytes {
            return Err(tool_error(format!(
                "reference {} is {}; the most is {}",
                py_repr(name),
                size_text(data.len()),
                size_text(self.inner.max_reference_bytes)
            )));
        }
        let Some((kind, _, _)) = sniff(&data) else {
            return Err(tool_error(format!("reference {} is not a PNG, JPEG, WebP or GIF image", py_repr(name))));
        };
        Ok(Reference { data, media_type: kind.into(), name: name.into() })
    }

    /// A reference from the web: public addresses only, images only, capped.
    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        let mut url = url.to_string();
        for _ in 0..4 {
            let short = head(&url, 120);
            let parsed = reqwest::Url::parse(&url).ok();
            let host = parsed.as_ref().and_then(|p| p.host_str()).unwrap_or("").trim_matches(['[', ']']).to_lowercase();
            if !parsed.as_ref().is_some_and(|p| matches!(p.scheme(), "http" | "https")) || host.is_empty() {
                return Err(tool_error(format!("reference {} is not an http(s) address", py_repr(&short))));
            }
            if !self.inner.allow_private && is_private_host(&host, Some(&self.inner.resolved)).await {
                return Err(tool_error(format!("reference {host} is a private or loopback address")));
            }
            let mut response = match self.inner.client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    let kind = if e.is_timeout() {
                        "TimeoutException"
                    } else if e.is_connect() {
                        "ConnectError"
                    } else {
                        "RequestError"
                    };
                    return Err(tool_error(format!("reference {short} could not be fetched ({kind})")));
                }
            };
            let status = response.status().as_u16();
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                let location = response.headers().get("location").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                url = reqwest::Url::parse(&url).and_then(|b| b.join(&location)).map(|u| u.to_string()).unwrap_or(location);
                continue;
            }
            if status != 200 {
                return Err(tool_error(format!("reference {short} answered {status}")));
            }
            let mut out = vec![];
            loop {
                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        if out.len() + chunk.len() > self.inner.max_reference_bytes {
                            return Err(tool_error(format!("reference {short} is larger than {}", size_text(self.inner.max_reference_bytes))));
                        }
                        out.extend_from_slice(&chunk);
                    }
                    Ok(None) => return Ok(out),
                    Err(_) => return Err(tool_error(format!("reference {short} could not be fetched (ReadError)"))),
                }
            }
        }
        Err(tool_error("the reference redirects too many times"))
    }

    async fn reference_from_block(&self, block: &MediaBlock) -> Result<Reference> {
        let label = block_label(block);
        if block.inline() {
            return match block.read() {
                Ok(bytes) => self.accept(bytes, &label),
                Err(_) => Err(tool_error(format!("reference {} is no longer there", py_repr(&label)))),
            };
        }
        let bytes = self.fetch(block.url.as_deref().unwrap_or("")).await?;
        self.accept(bytes, &label)
    }

    /// One reference, from whatever names it.
    ///
    /// `trusted` is the difference between your code and a model: your code
    /// may name any file on this machine; a model may name only what it was shown.
    async fn reference(&self, item: &ReferenceSource, context: &Context) -> Result<Reference> {
        let text = match item {
            ReferenceSource::Reference(r) => return self.accept(r.data.clone(), &r.name),
            ReferenceSource::Image(i) => return self.accept(i.data.clone(), if i.name.is_empty() { "image" } else { &i.name }),
            ReferenceSource::Bytes(b) => return self.accept(b.clone(), "reference"),
            ReferenceSource::Block(b) => return self.reference_from_block(b).await,
            ReferenceSource::Text(t) => t.trim().to_string(),
        };
        if text.is_empty() {
            return Err(tool_error("a reference is the name of an image"));
        }
        if text.starts_with("data:image/") {
            let decoded = text.split_once(',').and_then(|(_, b)| B64.decode(b.trim()).ok());
            return match decoded {
                Some(bytes) => self.accept(bytes, "reference"),
                None => Err(tool_error("that data: reference could not be decoded")),
            };
        }
        if text.starts_with("http://") || text.starts_with("https://") {
            let last = text.rsplit('/').next().unwrap_or("").split('?').next().unwrap_or("");
            let name = if last.is_empty() { "reference" } else { last };
            let bytes = self.fetch(&text).await?;
            return self.accept(bytes, name);
        }

        // An image made earlier, by its name or its path.
        let earlier = lock(&self.inner.recent)
            .values()
            .rev()
            .find(|k| text == k.name || text == k.path || Path::new(&k.name).file_stem().is_some_and(|s| s == text.as_str()))
            .cloned();
        if let Some(known) = earlier {
            return self.accept(known.data, &known.name);
        }
        // Something that came with the task.
        let given: Vec<&MediaBlock> = context.attachments.iter().filter_map(|b| if let ContentBlock::Image(m) = b { Some(m) } else { None }).collect();
        let position = POSITION.captures(&text.to_lowercase()).and_then(|c| c[1].parse::<usize>().ok());
        for (number, block) in given.iter().enumerate() {
            if text == block.name || text == block_label(block) || position == Some(number + 1) {
                return self.reference_from_block(block).await;
            }
        }
        if let Some(root) = &context.workspace
            && let Some(file) = inside(root, &text)
            && file.is_file()
        {
            // A file that is there and is no image, or is too large, is worth saying so.
            let bytes = tokio::fs::read(&file).await.map_err(|e| tool_error(format!("reference {} could not be read: {e}", py_repr(&text))))?;
            return self.accept(bytes, &text);
        }
        if context.trusted {
            let file = expand_user(&text);
            if file.is_file() {
                let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| text.clone());
                let bytes = tokio::fs::read(&file).await.map_err(|e| tool_error(format!("reference {} could not be read: {e}", py_repr(&text))))?;
                return self.accept(bytes, &name);
            }
            return Err(tool_error(format!("there is no image at {}", py_repr(&text))));
        }
        let recent: Vec<String> = lock(&self.inner.recent).values().map(|k| if k.path.is_empty() { k.name.clone() } else { k.path.clone() }).collect();
        let mut have: Vec<String> = recent[recent.len().saturating_sub(6)..].to_vec();
        have.extend(given.iter().enumerate().map(|(n, b)| format!("attachment:{} ({})", n + 1, block_label(b))));
        Err(tool_error(format!(
            "there is no image called {}. A reference is {}an image generated earlier, an attachment, or an http(s) address{}",
            py_repr(&text),
            if context.workspace.is_some() { "a file in the workspace, " } else { "" },
            if have.is_empty() { String::new() } else { format!(". Known: {}", have.join(", ")) }
        )))
    }

    // ---- making -------------------------------------------------------------------
    /// One engine, tried until it answers, cannot, or the time is up.
    async fn ask(&self, engine: &Arc<dyn ImageEngine>, request: &ImageRequest, job: &ImageJob) -> std::result::Result<Vec<GeneratedImage>, ImageFailure> {
        let missing = engine.ready();
        if !missing.is_empty() {
            return Err(ImageFailure::new(missing));
        }
        let most = engine.max_references();
        if request.references.len() > most {
            let model = engine.model();
            let who = if model.is_empty() { engine.name() } else { model };
            let can = if most > 0 { format!("at most {most} reference image{}", if most == 1 { "" } else { "s" }) } else { "no reference images".into() };
            return Err(ImageFailure::new(format!("{who} takes {can}; {} were given", request.references.len())).healthy());
        }
        let watched = job.clone();
        let report: Reporter = Arc::new(move |status: &str, message: &str, fraction: Option<f64>| {
            watched.report("running", if message.is_empty() { status } else { message }, fraction);
        });
        let policy = &self.inner.retry;
        let mut attempt = 0u32;
        loop {
            let left = self.inner.deadline - job.elapsed();
            if left <= 0.0 {
                return Err(ImageFailure::new("the job ran out of time").healthy());
            }
            let limit = self.inner.timeout.min(left);
            let failure = match tokio::time::timeout(Duration::from_secs_f64(limit), engine.generate(request, &self.inner.client, &report)).await {
                Ok(Ok(images)) => return Ok(images),
                Ok(Err(failure)) => failure,
                Err(_) => ImageFailure::new(format!("no image within {}s", seconds(limit))).retryable(),
            };
            if !failure.retryable || attempt >= policy.max_retries {
                return Err(failure);
            }
            let wait = match failure.retry_after {
                None => policy.backoff(attempt),
                Some(w) if w > policy.max_retry_after => return Err(failure),
                Some(w) => w,
            };
            if job.elapsed() + wait >= self.inner.deadline {
                return Err(failure);
            }
            attempt += 1;
            report("retrying", &format!("{}: {failure} — trying again ({attempt} of {})", engine.name(), policy.max_retries), None);
            tokio::time::sleep(Duration::from_secs_f64(wait.max(0.0))).await;
        }
    }

    /// What an engine returned, checked to be images and measured.
    fn finish_images(raw: Vec<GeneratedImage>, request: &ImageRequest, engine: &Arc<dyn ImageEngine>) -> Vec<GeneratedImage> {
        let mut made = vec![];
        for mut image in raw {
            let Some((kind, width, height)) = sniff(&image.data) else { continue };
            image.media_type = kind.into();
            image.width = width;
            image.height = height;
            image.prompt = request.prompt.clone();
            image.engine = engine.name();
            if image.model.is_empty() {
                image.model = engine.model();
            }
            made.push(image);
        }
        made.truncate(request.count as usize);
        made
    }

    async fn run(&self, job: &ImageJob, mut request: ImageRequest, references: &[ReferenceSource], context: &Context) -> Result<Vec<GeneratedImage>> {
        let _permit = self.inner.gate.clone().acquire_owned().await.map_err(|_| tool_error("the image generator was shut down"))?;
        job.report("running", if references.is_empty() { "starting" } else { "reading the reference images" }, None);
        for item in references {
            let reference = self.reference(item, context).await?;
            request.references.push(reference);
        }
        let engines = self.resolve()?;
        if engines.is_empty() {
            let needs: Vec<String> = image_engines().iter().map(|e| e.needs.split(" or ").next().unwrap_or("").to_string()).collect();
            return Err(tool_error(format!("no image engine is set up — set {}, or pass engine=", needs.join(", "))));
        }
        let mut problems = vec![];
        for engine in &engines {
            let name = engine.name();
            let breaker = self.breaker(&name);
            if !breaker.allow() {
                problems.push(format!("{name}: left alone after repeated failures, for another {:.0}s", breaker.retry_in()));
                continue;
            }
            job.set_engine(&name);
            job.report("running", &format!("asking {name}"), None);
            let mut probe = ProbeGuard(Some(breaker.clone()));
            let outcome = match self.ask(engine, &request, job).await {
                Ok(raw) => {
                    let made = Self::finish_images(raw, &request, engine);
                    if made.is_empty() { Err(ImageFailure::new("what came back was not an image")) } else { Ok(made) }
                }
                Err(e) => Err(e),
            };
            probe.0 = None;
            match outcome {
                Ok(made) => {
                    breaker.record_success();
                    self.count(&name, true, "");
                    return Ok(made);
                }
                Err(failure) => {
                    let message = redact(&failure.message, &engine.secrets());
                    self.count(&name, false, &message);
                    if failure.unhealthy {
                        breaker.record_failure();
                    } else {
                        breaker.record_success();
                    }
                    if failure.refused {
                        // Declined for what was asked. Another provider is not a way round that.
                        return Err(tool_error(format!("{name} would not make this image — {message}. Change the prompt or the references.")));
                    }
                    problems.push(format!("{name}: {message}"));
                }
            }
        }
        Err(tool_error(format!("image generation failed — {}", problems.join("; "))))
    }

    /// Name each image and put it where it is kept.
    async fn store(&self, images: &mut [GeneratedImage], name: &str, context: &Context) -> Result<()> {
        let base = if name.is_empty() { slug(&images[0].prompt, "image") } else { slug(name, "image") };
        let several = images.len() > 1;
        for (number, image) in images.iter_mut().enumerate() {
            let stem = if several { format!("{base}-{}", number + 1) } else { base.clone() };
            let mut attempt = 1;
            loop {
                image.name = if attempt == 1 { format!("{stem}.{}", image.extension()) } else { format!("{stem}-{attempt}.{}", image.extension()) };
                let taken = if let Some(root) = &context.workspace {
                    root.join(&self.inner.directory).join(&image.name).exists()
                } else if let Some(dir) = &self.inner.output_dir {
                    dir.join(&image.name).exists()
                } else {
                    lock(&self.inner.recent).contains_key(&image.name)
                };
                if !taken {
                    break;
                }
                attempt += 1;
            }
            if let Some(root) = &context.workspace {
                let relative = format!("{}/{}", self.inner.directory, image.name);
                let target = root.join(&self.inner.directory).join(&image.name);
                if let Some(parent) = target.parent() {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(&target, &image.data).await?;
                image.path = relative;
            } else if let Some(dir) = &self.inner.output_dir {
                image.path = image.save(dir.join(&image.name))?.to_string_lossy().into_owned();
            }
            let mut recent = lock(&self.inner.recent);
            recent.insert(image.name.clone(), image.clone());
            while recent.len() > 24 {
                recent.shift_remove_index(0);
            }
        }
        Ok(())
    }

    fn charge(&self, count: usize, context: &Context) {
        let Some(price) = self.inner.cost_per_image.filter(|p| *p > 0.0) else { return };
        if count == 0 {
            return;
        }
        let spent = Usage { cost_usd: price * count as f64, ..Usage::default() };
        if let Some(ctx) = &context.tool
            && let Some(guard) = ctx.run.as_ref().and_then(|r| r.guard.clone())
        {
            // Over the limit: the loop stops the run at its next check.
            let _ = guard.record(&spent, &ctx.agent, "");
        }
    }

    /// Start a generation and return its job at once.
    ///
    /// Watch it with `job.describe()` and `job.done()`, wait with
    /// `job.wait(..)`, stop it with `job.cancel()`. What is wrong with the
    /// request itself — an empty prompt, too many references — fails here.
    pub async fn submit(&self, prompt: &str, options: GenerateOptions) -> Result<ImageJob> {
        self.submit_with(prompt, options, Context { trusted: true, ..Context::default() })
    }

    fn submit_with(&self, prompt: &str, options: GenerateOptions, context: Context) -> Result<ImageJob> {
        let (prompt, wanted) = self.check(prompt, i64::from(options.count))?;
        if options.references.len() > self.inner.max_references {
            return Err(tool_error(format!("{} reference images were given; the most is {}", options.references.len(), self.inner.max_references)));
        }
        let aspect = match &options.aspect_ratio {
            None => self.inner.aspect_ratio.clone(),
            Some(r) => {
                let r = ratio(r)?;
                if r.is_empty() { self.inner.aspect_ratio.clone() } else { r }
            }
        };
        let request = ImageRequest {
            prompt: prompt.clone(),
            references: vec![],
            aspect_ratio: aspect,
            count: wanted,
            size: options.size.clone(),
            seed: options.seed,
            options: options.options.clone(),
        };
        let listeners: Vec<ProgressFn> = options.on_progress.iter().chain(self.inner.on_progress.iter()).cloned().collect();
        let job = ImageJob::create(&prompt, wanted, listeners);
        {
            let mut jobs = lock(&self.inner.jobs);
            jobs.insert(job.id().to_string(), job.clone());
            while jobs.len() > 64 {
                jobs.shift_remove_index(0);
            }
        }
        let me = self.clone();
        let watched = job.clone();
        let name = options.name.clone();
        let references = options.references;
        let task = tokio::spawn(async move {
            let outcome = match me.run(&watched, request, &references, &context).await {
                Ok(mut images) => match me.store(&mut images, &name, &context).await {
                    Ok(()) => {
                        *lock(&me.inner.made) += images.len() as u64;
                        me.charge(images.len(), &context);
                        Ok(images)
                    }
                    Err(e) => Err(e),
                },
                Err(e) => Err(e),
            };
            watched.finish(outcome);
        });
        *lock(&job.inner.abort) = Some(task.abort_handle());
        Ok(job)
    }

    /// Make images and return them. Fails with `Error::Tool` when none could be made.
    ///
    /// Dropping this call (a cancelled task, a timeout around it) cancels the job.
    pub async fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<Vec<GeneratedImage>> {
        let job = self.submit(prompt, options).await?;
        let mut guard = CancelOnDrop(Some(job.clone()));
        let out = job.wait(None).await;
        guard.0 = None;
        out
    }

    // ---- as tools ----------------------------------------------------------------
    /// `generate_image` and `image_status`, for an agent. See [`image_tools`].
    pub fn tools(&self, options: ImageToolOptions) -> Vec<Tool> {
        image_tools(self, options)
    }
}

/// How the image tools are made.
#[derive(Clone)]
pub struct ImageToolOptions {
    pub name: String,
    pub status_name: String,
    /// Seconds `generate_image` waits before handing back a job to check on
    /// with `image_status`.
    pub wait: f64,
    /// Show the model what was made, so it can judge it and try again.
    pub show: bool,
    /// Images every generation is given — a brand's style, a character sheet —
    /// besides the ones the model names.
    pub references: Vec<ReferenceSource>,
    /// `Ask` to approve every generation: each one costs money.
    pub permission: Permission,
}

impl Default for ImageToolOptions {
    fn default() -> Self {
        Self { name: "generate_image".into(), status_name: "image_status".into(), wait: 120.0, show: true, references: vec![], permission: Permission::Allow }
    }
}

/// Generate an image from a description — or from other images, to edit or follow them.
///
/// Say everything that matters in the prompt: subject, setting, style, lighting, any text to appear. To change an existing image, pass it as a reference and describe the change.
#[derive(Deserialize, JsonSchema)]
struct GenerateImageArgs {
    /// what the image should show.
    prompt: String,
    /// images to work from — a file in the workspace, the name of an image generated earlier, an attachment ("attachment:1"), or an http(s) address.
    #[serde(default)]
    references: Option<Vec<String>>,
    /// the shape, e.g. "1:1", "16:9", "9:16", "4:3". Leave empty for the default.
    #[serde(default)]
    aspect_ratio: String,
    /// how many variations to make.
    #[serde(default = "one")]
    count: i64,
    /// a short file name for the result, without an extension.
    #[serde(default)]
    name: String,
}

fn one() -> i64 {
    1
}

/// Check on an image that is still being generated, and collect it when it is done.
#[derive(Deserialize, JsonSchema)]
struct ImageStatusArgs {
    /// the job to check. Leave empty to list the latest jobs.
    #[serde(default)]
    job_id: String,
    /// seconds to wait for it to finish before answering, at most 120.
    #[serde(default)]
    wait: f64,
    /// stop the job instead.
    #[serde(default)]
    cancel: bool,
}

/// What the tool context says the model was shown. The agent loop puts the
/// workspace root (a path) under `state["workspace"]` and the task's
/// attachments (content blocks) under `state["attachments"]`.
fn tool_context(ctx: &ToolContext) -> Context {
    let state = lock(&ctx.state);
    let workspace = state.get("workspace").and_then(Value::as_str).filter(|w| !w.is_empty()).map(PathBuf::from);
    let attachments = state
        .get("attachments")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|b| serde_json::from_value(b.clone()).ok()).collect())
        .unwrap_or_default();
    Context { trusted: false, workspace, attachments, tool: Some(ctx.clone()) }
}

/// What the model reads when a job is over.
fn handed(job: &ImageJob, show: bool) -> Result<Value> {
    let made = job.result()?;
    job.state().collected = true;
    let Some(first) = made.first() else { return Ok(json!("No image was made.")) };
    let plural = if made.len() == 1 { "" } else { "s" };
    let mut lines = vec![format!("Generated {} image{plural} with {} ({}) in {:.0}s:", made.len(), first.engine, first.model, job.elapsed())];
    lines.extend(made.iter().map(|i| format!("- {}", i.describe())));
    if let Some(said) = made.iter().map(|i| i.text.as_str()).find(|t| !t.is_empty()) {
        lines.push(format!("The model said: {}", head(said, 400)));
    }
    let label = if first.path.is_empty() { &first.name } else { &first.path };
    lines.push(format!("To change one, generate again with it as a reference: references=[\"{label}\"]."));
    let text = lines.join("\n");
    if !show {
        return Ok(Value::String(text));
    }
    let mut out = vec![Value::String(text)];
    out.extend(made.iter().map(|i| serde_json::to_value(i.block()).unwrap_or(Value::Null)));
    Ok(Value::Array(out))
}

fn pending(job: &ImageJob, status_name: &str) -> Value {
    Value::String(format!(
        "Still working: job {} is {}. Call {status_name}(job_id=\"{}\") to collect it — do not start the same image again.",
        job.id(),
        job.describe(),
        job.id()
    ))
}

/// The tools an agent makes images with: `generate_image` and `image_status`.
///
/// Images go in the workspace when the tool context names one
/// (`state["workspace"]`), else the generator's `output_dir`, else they are
/// kept in memory — and can still be built on by name.
pub fn image_tools(generator: &ImageGenerator, options: ImageToolOptions) -> Vec<Tool> {
    let tags = ["builtin", "images"];
    let images = generator.clone();
    let always = options.references.clone();
    let status_name = options.status_name.clone();
    let wait = options.wait;
    let show = options.show;
    let generate = Tool::new(options.name.clone(), move |args: GenerateImageArgs, ctx: ToolContext| {
        let images = images.clone();
        let always = always.clone();
        let status_name = status_name.clone();
        async move {
            let mut references = always;
            references.extend(args.references.unwrap_or_default().into_iter().map(ReferenceSource::Text));
            let request = GenerateOptions {
                references,
                aspect_ratio: Some(args.aspect_ratio),
                count: args.count.clamp(0, i64::from(u32::MAX)) as u32,
                name: args.name,
                ..Default::default()
            };
            let job = images.submit_with(&args.prompt, request, tool_context(&ctx))?;
            let mut guard = CancelOnDrop(Some(job.clone()));
            let waited = job.wait(Some(wait)).await;
            guard.0 = None;
            match waited {
                Err(Error::Tool { .. }) if !job.done() => Ok(pending(&job, &status_name)),
                Err(e) => Err(e),
                Ok(_) => handed(&job, show),
            }
        }
    })
    .tags(tags)
    .permission(options.permission);

    let images = generator.clone();
    let status_name = options.status_name.clone();
    let status = Tool::new(options.status_name.clone(), move |args: ImageStatusArgs, _ctx: ToolContext| {
        let images = images.clone();
        let status_name = status_name.clone();
        async move {
            if args.job_id.trim().is_empty() {
                let jobs = images.jobs();
                let latest = &jobs[jobs.len().saturating_sub(8)..];
                if latest.is_empty() {
                    return Ok(json!("No images have been asked for."));
                }
                let lines: Vec<String> = latest.iter().map(|j| format!("{}: {} — {}", j.id(), j.describe(), py_repr(&head(j.prompt(), 60)))).collect();
                return Ok(Value::String(lines.join("\n")));
            }
            let job = images.job(&args.job_id)?;
            if args.cancel {
                return Ok(Value::String(if job.cancel() {
                    format!("Job {} is being cancelled.", job.id())
                } else {
                    format!("Job {} is already over: {}", job.id(), job.describe())
                }));
            }
            if !job.done() && args.wait > 0.0 {
                let _ = job.wait(Some(args.wait.clamp(0.0, 120.0))).await;
            }
            if !job.done() {
                return Ok(pending(&job, &status_name));
            }
            handed(&job, show)
        }
    })
    .tags(tags);
    vec![generate, status]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::budget::{Budget, BudgetGuard};
    use crate::toolkits::openapi::tests::{Answer, Seen, serve};
    use crate::tools::RunScope;

    static RED: LazyLock<Vec<u8>> = LazyLock::new(|| solid_png(8, 4, [200, 30, 30]));
    static BLUE: LazyLock<Vec<u8>> = LazyLock::new(|| solid_png(4, 4, [30, 30, 200]));
    const JPEG: &[u8] =
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xc0\x00\x11\x08\x00\x20\x00\x30\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01";

    fn b64() -> String {
        B64.encode(&*RED)
    }

    fn fast() -> RetryPolicy {
        RetryPolicy { max_retries: 2, initial_delay: 0.0, max_delay: 0.0, jitter: 0.0, max_retry_after: 1.0, ..RetryPolicy::default() }
    }

    /// No keys in the environment the engines see.
    fn no_keys() {
        ENV.with(|e| *e.borrow_mut() = Some(HashMap::new()));
    }

    fn set_env(name: &str, value: &str) {
        ENV.with(|e| {
            e.borrow_mut().get_or_insert_with(HashMap::new).insert(name.into(), value.into());
        });
    }

    fn reply(status: u16, body: Value) -> (u16, Vec<(&'static str, String)>, Vec<u8>) {
        (status, vec![("content-type", "application/json".into())], serde_json::to_vec(&body).unwrap())
    }

    fn gemini_ok() -> (u16, Vec<(&'static str, String)>, Vec<u8>) {
        reply(
            200,
            json!({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Here is your fox."}, {"inlineData": {"mimeType": "image/png", "data": b64()}}]}}]}),
        )
    }

    fn generator(engines: Vec<EngineSpec>, options: ImageOptions) -> ImageGenerator {
        ImageGenerator::new(ImageOptions { engines, retry: Some(options.retry.clone().unwrap_or_else(fast)), ..options }).unwrap()
    }

    fn opts() -> GenerateOptions {
        GenerateOptions::default()
    }

    fn body(seen: &Seen) -> Value {
        serde_json::from_slice(&seen.body).unwrap()
    }

    #[test]
    fn an_image_is_recognised_and_measured_from_its_bytes() {
        assert_eq!(sniff(&RED), Some(("image/png", 8, 4)));
        assert_eq!(sniff(JPEG), Some(("image/jpeg", 48, 32)));
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&5u16.to_le_bytes());
        gif.extend_from_slice(&7u16.to_le_bytes());
        gif.extend_from_slice(&[0; 8]);
        assert_eq!(sniff(&gif), Some(("image/gif", 5, 7)));
        let mut webp = b"RIFF\x00\x00\x00\x00WEBPVP8X\x0a\x00\x00\x00\x00\x00\x00\x00".to_vec();
        webp.extend_from_slice(&99u32.to_le_bytes()[..3]);
        webp.extend_from_slice(&49u32.to_le_bytes()[..3]);
        assert_eq!(sniff(&webp), Some(("image/webp", 100, 50)));
        assert_eq!(sniff(b"<html>an error page saved as photo.png</html>"), None);
        assert_eq!(sniff(b""), None);
    }

    #[tokio::test]
    async fn gemini_sends_the_prompt_and_the_references_and_reads_the_image() {
        no_keys();
        let server = serve(Arc::new(|_| gemini_ok())).await;
        let engine = GeminiImages::new().with_key("g-secret-key").with_base_url(format!("{}/v1beta", server.url));
        let images = generator(vec![engine.clone().into()], ImageOptions::default());
        let made = images
            .generate(
                "a red fox in snow",
                GenerateOptions {
                    references: vec![BLUE.clone().into(), JPEG.into()],
                    aspect_ratio: Some("16:9".into()),
                    seed: Some(7),
                    size: "2K".into(),
                    ..opts()
                },
            )
            .await
            .unwrap();
        let request = server.last();
        assert_eq!(request.path, "/v1beta/models/gemini-2.5-flash-image:generateContent");
        assert_eq!(request.header("x-goog-api-key"), "g-secret-key");
        let sent = body(&request);
        let parts = sent["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0], json!({"text": "a red fox in snow"}));
        assert_eq!(parts[1..].iter().map(|p| p["inlineData"]["mimeType"].clone()).collect::<Vec<_>>(), [json!("image/png"), json!("image/jpeg")]);
        assert_eq!(B64.decode(parts[1]["inlineData"]["data"].as_str().unwrap()).unwrap(), *BLUE);
        assert_eq!(
            sent["generationConfig"],
            json!({"responseModalities": ["TEXT", "IMAGE"], "seed": 7, "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}})
        );

        let image = &made[0];
        assert_eq!(image.data, *RED);
        assert_eq!((image.width, image.height), (8, 4));
        assert_eq!((image.engine.as_str(), image.model.as_str()), ("gemini", "gemini-2.5-flash-image"));
        assert_eq!(image.text, "Here is your fox.");
        assert_eq!(image.name, "a-red-fox-in-snow.png");
        assert!(matches!(image.block(), ContentBlock::Image(_)));
        assert!(!format!("{image:?}").to_lowercase().contains("bytes"));

        // One picture a call: three wanted is three calls.
        let images = generator(vec![engine.into()], ImageOptions::default());
        let before = server.count();
        assert_eq!(images.generate("a fox", GenerateOptions { count: 3, ..opts() }).await.unwrap().len(), 3);
        assert_eq!(server.count() - before, 3);
    }

    #[tokio::test]
    async fn imagen_is_asked_the_other_way_and_takes_no_references() {
        no_keys();
        let server = serve(Arc::new(|_| {
            reply(200, json!({"predictions": [{"bytesBase64Encoded": b64(), "mimeType": "image/png"}, {"bytesBase64Encoded": b64(), "mimeType": "image/png"}]}))
        }))
        .await;
        let engine = GeminiImages::new().with_key("k").with_model("imagen-4.0-generate-001").with_base_url(&server.url);
        let images = generator(vec![engine.into()], ImageOptions::default());
        let made = images.generate("a fox", GenerateOptions { count: 2, aspect_ratio: Some("portrait".into()), ..opts() }).await.unwrap();
        assert!(server.last().path.ends_with("/models/imagen-4.0-generate-001:predict"));
        assert_eq!(body(&server.last()), json!({"instances": [{"prompt": "a fox"}], "parameters": {"sampleCount": 2, "aspectRatio": "9:16"}}));
        assert_eq!(made.len(), 2);
        let err = images.generate("a fox", GenerateOptions { references: vec![BLUE.clone().into()], ..opts() }).await.unwrap_err().to_string();
        assert!(err.contains("imagen-4.0-generate-001 takes no reference"), "{err}");
    }

    #[tokio::test]
    async fn openai_generates_and_edits_with_references() {
        no_keys();
        let server = serve(Arc::new(|_| reply(200, json!({"data": [{"b64_json": b64()}]})))).await;
        let engine = OpenAIImages::new().with_key("sk-secret-key").with_option("quality", json!("high")).with_base_url(format!("{}/v1", server.url));
        let images = generator(vec![engine.into()], ImageOptions::default());
        images.generate("a fox", GenerateOptions { aspect_ratio: Some("3:2".into()), count: 2, ..opts() }).await.unwrap();
        let first = server.last();
        assert_eq!(first.path, "/v1/images/generations");
        assert_eq!(first.header("authorization"), "Bearer sk-secret-key");
        assert_eq!(body(&first), json!({"model": "gpt-image-1", "prompt": "a fox", "n": 2, "size": "1536x1024", "quality": "high"}));

        images.generate("make the fox blue", GenerateOptions { references: vec![RED.clone().into(), BLUE.clone().into()], ..opts() }).await.unwrap();
        let edit = server.last();
        assert_eq!(edit.path, "/v1/images/edits");
        assert!(edit.header("content-type").starts_with("multipart/form-data"));
        let raw = String::from_utf8_lossy(&edit.body).into_owned();
        assert_eq!(raw.matches("name=\"image[]\"").count(), 2);
        assert!(edit.body.windows(RED.len()).any(|w| w == RED.as_slice()));
        assert!(raw.contains("name=\"prompt\"\r\n\r\nmake the fox blue"));

        // An older model, and a server that answers with a link.
        let url = Arc::new(Mutex::new(String::new()));
        let link = url.clone();
        let server = serve(Arc::new(move |r: &Seen| {
            if r.path == "/cdn/a.png" {
                (200, vec![("content-type", "image/png".into())], RED.clone())
            } else {
                reply(200, json!({"data": [{"url": format!("{}/cdn/a.png", lock(&link)), "revised_prompt": "A red fox."}]}))
            }
        }))
        .await;
        *lock(&url) = server.url.clone();
        let images = generator(vec![OpenAIImages::new().with_key("k").with_model("dall-e-3").with_base_url(&server.url).into()], ImageOptions::default());
        let made = images.generate("a fox", GenerateOptions { aspect_ratio: Some("9:16".into()), ..opts() }).await.unwrap();
        let asked = body(&server.seen.lock().unwrap()[0]);
        assert_eq!(asked["size"], "1024x1792");
        assert_eq!(asked["response_format"], "b64_json");
        assert_eq!(made[0].data, *RED);
        assert_eq!(made[0].text, "A red fox.");
    }

    fn replicate_server(states: Vec<Value>, cancelled: Arc<Mutex<Vec<String>>>) -> Answer {
        let states = Arc::new(Mutex::new(states));
        Arc::new(move |r: &Seen| {
            let base = format!("http://{}", r.header("host"));
            if r.path == "/delivery/out.png" {
                return (200, vec![("content-type", "image/png".into())], RED.clone());
            }
            if r.path.ends_with("/cancel") {
                lock(&cancelled).push(r.path.clone());
                return reply(200, json!({}));
            }
            if r.method == "POST" {
                return reply(
                    201,
                    json!({"id": "p1", "status": "starting", "urls": {"get": format!("{base}/v1/predictions/p1"), "cancel": format!("{base}/v1/predictions/p1/cancel")}}),
                );
            }
            let mut states = lock(&states);
            let state = if states.len() > 1 { states.remove(0) } else { states[0].clone() };
            let mut state = state;
            if let Some(Value::Array(out)) = state.get_mut("output") {
                for o in out.iter_mut() {
                    *o = json!(format!("{base}{}", o.as_str().unwrap()));
                }
            }
            reply(200, state)
        })
    }

    #[tokio::test]
    async fn replicate_starts_a_prediction_watches_it_and_reports_progress() {
        no_keys();
        let states = vec![
            json!({"status": "starting"}),
            json!({"status": "processing", "logs": "12%|█\n 40%|███"}),
            json!({"status": "processing", "logs": "40%|███\n 85%|██████"}),
            json!({"status": "succeeded", "output": ["/delivery/out.png"]}),
        ];
        let server = serve(replicate_server(states, Arc::default())).await;
        let updates: Arc<Mutex<Vec<ImageProgress>>> = Arc::default();
        let record = updates.clone();
        let engine = ReplicateImages::new().with_key("r8-secret").with_poll_interval(0.001).with_base_url(format!("{}/v1", server.url));
        let images = generator(
            vec![engine.into()],
            ImageOptions { on_progress: Some(Arc::new(move |u: &ImageProgress| lock(&record).push(u.clone()))), ..ImageOptions::default() },
        );
        let made = images
            .generate("a fox", GenerateOptions { references: vec![BLUE.clone().into()], aspect_ratio: Some("1:1".into()), seed: Some(3), ..opts() })
            .await
            .unwrap();
        let first = server.seen.lock().unwrap()[0].clone();
        let start = body(&first);
        assert_eq!(first.path, "/v1/models/black-forest-labs/flux-kontext-pro/predictions");
        assert_eq!(first.header("authorization"), "Bearer r8-secret");
        assert_eq!(start["input"]["prompt"], "a fox");
        assert_eq!(start["input"]["seed"], 3);
        assert_eq!(start["input"]["aspect_ratio"], "1:1");
        assert!(start["input"]["input_image"].as_str().unwrap().starts_with("data:image/png;base64,"));
        assert_eq!(made[0].data, *RED);

        let updates = lock(&updates).clone();
        let fractions: Vec<f64> = updates.iter().filter_map(|u| u.fraction).collect();
        assert_eq!(fractions, [0.4, 0.85, 1.0, 1.0]);
        assert_eq!(updates.last().unwrap().status, "done");
        assert!(updates.iter().all(|u| u.job == updates[0].job && (u.engine.is_empty() || u.engine == "replicate")));
    }

    #[tokio::test]
    async fn a_replicate_job_that_fails_or_is_abandoned_is_cancelled_there() {
        no_keys();
        let cancelled: Arc<Mutex<Vec<String>>> = Arc::default();
        let server = serve(replicate_server(vec![json!({"status": "processing"})], cancelled.clone())).await;
        let engine = ReplicateImages::new().with_key("k").with_poll_interval(0.01).with_base_url(format!("{}/v1", server.url));
        let images = generator(vec![engine.into()], ImageOptions::default());
        let job = images.submit("a fox", opts()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(job.status(), "running");
        assert!(job.describe().contains("running on replicate"), "{}", job.describe());
        assert!(job.cancel());
        assert!(job.wait(None).await.unwrap_err().to_string().contains("was cancelled"));
        assert_eq!(job.status(), "cancelled");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(*lock(&cancelled), ["/v1/predictions/p1/cancel"]);

        let failing = serve(Arc::new(|r: &Seen| {
            if r.method == "POST" {
                reply(201, json!({"id": "p2", "status": "starting"}))
            } else {
                reply(200, json!({"status": "failed", "error": "NSFW content detected"}))
            }
        }))
        .await;
        let engine = ReplicateImages::new().with_key("k").with_poll_interval(0.001).with_base_url(format!("{}/v1", failing.url));
        let images = generator(vec![engine.into(), FakeImages::new().with_name("second").into()], ImageOptions::default());
        let err = images.generate("a fox", opts()).await.unwrap_err().to_string();
        assert!(err.contains("replicate would not make this image"), "{err}");
        assert!(!images.stats().contains_key("second")); // a refusal is not taken elsewhere
    }

    #[tokio::test]
    async fn a_throttled_call_is_retried_and_a_spurious_empty_answer_too() {
        no_keys();
        let answers = Arc::new(Mutex::new(vec![
            (429, vec![("retry-after", "0".to_string())], vec![]),
            (503, vec![], vec![]),
            reply(200, json!({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Let me think about that."}]}}]})),
        ]));
        let server = serve(Arc::new(move |_| {
            let mut answers = lock(&answers);
            if answers.is_empty() { gemini_ok() } else { answers.remove(0) }
        }))
        .await;
        let updates: Arc<Mutex<Vec<ImageProgress>>> = Arc::default();
        let record = updates.clone();
        let retry = RetryPolicy { max_retries: 3, initial_delay: 0.0, max_delay: 0.0, jitter: 0.0, ..RetryPolicy::default() };
        let images = generator(
            vec![GeminiImages::new().with_key("k").with_base_url(&server.url).into()],
            ImageOptions { retry: Some(retry), ..ImageOptions::default() },
        );
        let made = images
            .generate("a fox", GenerateOptions { on_progress: Some(Arc::new(move |u: &ImageProgress| lock(&record).push(u.clone()))), ..opts() })
            .await
            .unwrap();
        assert_eq!(made.len(), 1);
        assert_eq!(server.count(), 4);
        let retried: Vec<String> = lock(&updates).iter().map(|u| u.message.clone()).filter(|m| m.contains("trying again")).collect();
        assert_eq!(retried.len(), 3);
        assert!(retried[0].contains("rate limited"));
        assert!(retried[2].contains("answered without an image"));
    }

    #[tokio::test]
    async fn the_next_engine_makes_it_when_one_cannot() {
        no_keys();
        let server = serve(Arc::new(|r: &Seen| {
            if r.path.starts_with("/gemini") {
                reply(401, json!({"error": {"message": "API key not valid"}}))
            } else {
                reply(200, json!({"data": [{"b64_json": b64()}]}))
            }
        }))
        .await;
        let images = generator(
            vec![
                GeminiImages::new().with_key("g-secret-key").with_base_url(format!("{}/gemini", server.url)).into(),
                OpenAIImages::new().with_key("sk-key").with_base_url(format!("{}/openai", server.url)).into(),
            ],
            ImageOptions::default(),
        );
        let made = images.generate("a fox", opts()).await.unwrap();
        assert_eq!(made[0].engine, "openai");
        assert_eq!(images.engines().unwrap(), ["gemini", "openai"]);
        assert_eq!(images.stats()["gemini"].last_error, "the API key was refused");
        let paths: Vec<String> = server.seen.lock().unwrap().iter().map(|r| r.path.split('/').nth(1).unwrap().to_string()).collect();
        assert_eq!(paths, ["gemini", "openai"]);
    }

    #[tokio::test]
    async fn a_refused_prompt_is_not_retried_and_not_taken_to_another_provider() {
        no_keys();
        let answers = [
            reply(200, json!({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})),
            reply(200, json!({"candidates": [{"finishReason": "IMAGE_SAFETY"}]})),
            reply(400, json!({"error": {"code": "moderation_blocked", "message": "rejected by the safety system"}})),
        ];
        for answer in answers {
            let server = serve(Arc::new(move |_| answer.clone())).await;
            let images = generator(
                vec![GeminiImages::new().with_key("k").with_base_url(&server.url).into(), FakeImages::new().with_name("second").into()],
                ImageOptions::default(),
            );
            let err = images.generate("something it will not draw", opts()).await.unwrap_err().to_string();
            assert!(err.contains("gemini would not make this image") && err.contains("Change the prompt"), "{err}");
            assert_eq!(server.count(), 1);
            assert!(!images.stats().contains_key("second"));
        }
    }

    #[tokio::test]
    async fn when_every_engine_fails_the_reasons_are_given_without_the_keys() {
        no_keys();
        let server = serve(Arc::new(|r: &Seen| {
            if r.path.starts_with("/openai") {
                reply(429, json!({"error": {"message": "You exceeded your current quota sk-secret-key"}}))
            } else {
                (200, vec![("content-type", "text/html".into())], b"<html>gateway</html>".to_vec())
            }
        }))
        .await;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let images = generator(
            vec![
                GeminiImages::new().with_key("g-secret-key").with_base_url(dead).into(),
                OpenAIImages::new().with_key("sk-secret-key").with_base_url(format!("{}/openai", server.url)).into(),
                ReplicateImages::new().with_key("r8-secret-key").with_base_url(format!("{}/replicate", server.url)).into(),
            ],
            ImageOptions::default(),
        );
        let message = images.generate("a fox", opts()).await.unwrap_err().to_string();
        assert!(message.contains("gemini: could not be reached (ConnectError)"), "{message}");
        assert!(message.contains("openai: out of quota or credit"), "{message}");
        assert!(message.contains("replicate: answered with something that is not JSON"), "{message}");
        assert!(!message.contains("secret-key"));
    }

    #[tokio::test]
    async fn what_comes_back_must_be_an_image_and_a_slow_engine_times_out() {
        no_keys();
        let page = EngineSpec::function("page", |_r| async { Ok(vec![GeneratedImage::new(b"<html>Service unavailable</html>".to_vec())]) });
        let err = generator(vec![page], ImageOptions::default()).generate("a fox", opts()).await.unwrap_err().to_string();
        assert!(err.contains("page: what came back was not an image"), "{err}");

        let slow = || {
            EngineSpec::function("slow", |_r| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                Ok(vec![])
            })
        };
        let images =
            generator(vec![slow(), FakeImages::new().into()], ImageOptions { timeout: 0.05, retry: Some(RetryPolicy::none()), ..ImageOptions::default() });
        let made = images.generate("a fox", opts()).await.unwrap();
        assert_eq!(made[0].engine, "fake");
        assert!(images.stats()["slow"].last_error.contains("no image within 0.05s"));

        let retry = RetryPolicy { max_retries: 50, initial_delay: 0.0, jitter: 0.0, ..RetryPolicy::default() };
        let stuck = generator(vec![slow()], ImageOptions { timeout: 0.05, deadline: 0.12, retry: Some(retry), ..ImageOptions::default() });
        let started = Instant::now();
        let err = stuck.generate("a fox", opts()).await.unwrap_err().to_string();
        assert!(err.contains("slow: "), "{err}");
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn an_engine_that_keeps_failing_is_left_alone() {
        no_keys();
        let engine = Arc::new(FakeImages::new().with_failures(vec![ImageFailure::new("down"), ImageFailure::new("down")]));
        let images = generator(vec![engine.clone().into()], ImageOptions { failure_threshold: 2, retry: Some(RetryPolicy::none()), ..ImageOptions::default() });
        for _ in 0..2 {
            assert!(images.generate("a fox", opts()).await.unwrap_err().to_string().contains("fake: down"));
        }
        assert!(images.generate("a fox", opts()).await.unwrap_err().to_string().contains("left alone after repeated failures"));
        assert_eq!(engine.requests().len(), 2);
    }

    #[tokio::test]
    async fn nothing_set_up_and_things_set_up_wrong_say_so() {
        no_keys();
        let err = ImageGenerator::new(ImageOptions::default()).unwrap().generate("a fox", opts()).await.unwrap_err().to_string();
        assert!(err.contains("no image engine is set up — set GEMINI_API_KEY"), "{err}");
        let err = ImageGenerator::new(ImageOptions { engines: vec!["gemini".into()], ..ImageOptions::default() })
            .unwrap()
            .generate("a fox", opts())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("gemini: no API key — set GEMINI_API_KEY"), "{err}");
        let err = ImageGenerator::new(ImageOptions { engines: vec!["midjourney".into()], ..ImageOptions::default() }).unwrap_err().to_string();
        assert!(err.contains("no image engine named 'midjourney'"), "{err}");
        let err = ImageGenerator::new(ImageOptions { engines: vec!["gemini".into(), "openai".into()], model: Some("x".into()), ..ImageOptions::default() })
            .unwrap_err()
            .to_string();
        assert!(err.contains("go with one named engine"));
        let err = ImageGenerator::new(ImageOptions { aspect_ratio: "7:3".into(), ..ImageOptions::default() }).unwrap_err().to_string();
        assert!(err.contains("aspect_ratio is one of"));

        set_env("OPENAI_API_KEY", "k");
        set_env("GOOGLE_API_KEY", "k");
        assert_eq!(ImageGenerator::new(ImageOptions::default()).unwrap().engines().unwrap(), ["gemini", "openai"]);
        set_env(ENGINE_ENV, "flux, nano-banana");
        assert_eq!(ImageGenerator::new(ImageOptions::default()).unwrap().engines().unwrap(), ["replicate", "gemini"]);
        let ready: Vec<(String, bool)> = image_engines().into_iter().map(|e| (e.name, e.ready)).collect();
        assert_eq!(ready, [("gemini".to_string(), true), ("openai".to_string(), true), ("replicate".to_string(), false)]);
        let chosen =
            ImageGenerator::new(ImageOptions { engines: vec!["gemini".into()], model: Some("gemini-3-pro-image-preview".into()), ..ImageOptions::default() })
                .unwrap();
        assert_eq!(chosen.models().unwrap(), ["gemini-3-pro-image-preview"]);
    }

    #[tokio::test]
    async fn the_request_is_checked_and_counts_are_capped() {
        no_keys();
        let engine = Arc::new(FakeImages::new());
        let images = generator(
            vec![engine.clone().into()],
            ImageOptions { max_count: 3, max_images: Some(4), max_references: 2, aspect_ratio: "wide".into(), ..ImageOptions::default() },
        );
        assert!(images.generate("   ", opts()).await.unwrap_err().to_string().contains("prompt is empty"));
        assert!(images.generate(&"x".repeat(9000), opts()).await.unwrap_err().to_string().contains("the most is 8000"));
        let three = GenerateOptions { references: vec![RED.clone().into(), RED.clone().into(), RED.clone().into()], ..opts() };
        assert!(images.generate("a fox", three).await.unwrap_err().to_string().contains("3 reference images were given; the most is 2"));
        assert!(engine.requests().is_empty());

        assert_eq!(images.generate("a fox", GenerateOptions { count: 99, ..opts() }).await.unwrap().len(), 3); // clamped
        assert_eq!(engine.requests()[0].aspect_ratio, "16:9"); // the default shape
        assert_eq!(images.generate("a fox", GenerateOptions { count: 3, aspect_ratio: Some("4x3".into()), ..opts() }).await.unwrap().len(), 1);
        assert_eq!(engine.requests()[1].aspect_ratio, "4:3");
        assert_eq!(images.made(), 4);
        assert!(images.generate("a fox", opts()).await.unwrap_err().to_string().contains("limit of 4 images is used up"));
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ah-images-{tag}-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn references_come_from_files_urls_blocks_and_earlier_images() {
        no_keys();
        let dir = temp_dir("refs");
        let photo = dir.join("photo.jpg");
        std::fs::write(&photo, JPEG).unwrap();
        std::fs::write(dir.join("notes.png"), "not an image at all").unwrap();
        let server = serve(Arc::new(|r: &Seen| match r.path.as_str() {
            "/moved" => (302, vec![("location", "/ref.png".to_string())], vec![]),
            "/big.png" => {
                let mut big = RED.clone();
                big.extend(std::iter::repeat_n(0u8, 5000));
                (200, vec![], big)
            }
            "/ref.png" => (200, vec![], BLUE.clone()),
            _ => (404, vec![], BLUE.clone()),
        }))
        .await;
        let engine = Arc::new(FakeImages::new());
        let images = generator(vec![engine.clone().into()], ImageOptions { max_reference_bytes: 2000, allow_private: true, ..ImageOptions::default() });
        let mut blue = MediaBlock::from_bytes(&BLUE, "image/png");
        blue.name = "blue.png".into();
        let references: Vec<ReferenceSource> = vec![
            photo.clone().into(),
            photo.to_string_lossy().as_ref().into(),
            RED.clone().into(),
            blue.into(),
            format!("{}/ref.png", server.url).into(),
            format!("{}/moved", server.url).into(),
            format!("data:image/png;base64,{}", b64()).into(),
        ];
        let first = images.generate("a fox", GenerateOptions { name: "fox".into(), references, ..opts() }).await.unwrap();
        let kinds: Vec<(String, usize)> = engine.requests()[0].references.iter().map(|r| (r.media_type.clone(), r.data.len())).collect();
        let (png, jpg) = ("image/png".to_string(), "image/jpeg".to_string());
        assert_eq!(
            kinds,
            [
                (jpg.clone(), JPEG.len()),
                (jpg, JPEG.len()),
                (png.clone(), RED.len()),
                (png.clone(), BLUE.len()),
                (png.clone(), BLUE.len()),
                (png.clone(), BLUE.len()),
                (png, RED.len())
            ]
        );

        images.generate("the same fox, at night", GenerateOptions { references: vec!["fox.png".into()], ..opts() }).await.unwrap(); // by name
        images.generate("again", GenerateOptions { references: vec![(&first[0]).into()], ..opts() }).await.unwrap(); // by value
        assert_eq!(engine.requests()[1].references[0].data, first[0].data);
        assert_eq!(engine.requests()[2].references[0].data, first[0].data);

        let cases = [
            (dir.join("notes.png").to_string_lossy().into_owned(), "is not a PNG, JPEG, WebP or GIF"),
            (dir.join("missing.png").to_string_lossy().into_owned(), "there is no image at"),
            (format!("{}/nope.png", server.url), "answered 404"),
            (format!("{}/big.png", server.url), "is larger than"),
            ("ftp://example.com/x.png".to_string(), "there is no image at"),
        ];
        for (bad, why) in cases {
            let err = images.generate("a fox", GenerateOptions { references: vec![bad.into()], ..opts() }).await.unwrap_err().to_string();
            assert!(err.contains(why), "{err} / {why}");
        }
        let guarded = generator(vec![FakeImages::new().into()], ImageOptions::default());
        for bad in ["http://127.0.0.1/x.png", "http://169.254.169.254/latest/meta-data"] {
            let err = guarded.generate("a fox", GenerateOptions { references: vec![bad.into()], ..opts() }).await.unwrap_err().to_string();
            assert!(err.contains("private or loopback"), "{err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_job_is_watched_waited_for_and_collected() {
        no_keys();
        let images = generator(vec![FakeImages::new().with_steps(4, 0.02).into()], ImageOptions::default());
        let job = images.submit("a city at dusk", GenerateOptions { count: 2, ..opts() }).await.unwrap();
        assert!(matches!(job.status().as_str(), "queued" | "running") && !job.done());
        assert_eq!(images.jobs().len(), 1);
        assert!(job.result().unwrap_err().to_string().contains("is still"));
        let err = job.wait(Some(0.03)).await.unwrap_err().to_string();
        assert!(err.contains("is still running on fake"), "{err}");
        assert!(!job.done()); // a timed-out wait leaves it running
        let mut seen = vec![];
        while !job.done() {
            seen.push(job.describe().split(',').next().unwrap().to_string());
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(seen.iter().any(|line| line.contains('%')), "{seen:?}");
        assert_eq!(job.result().unwrap().len(), 2);
        assert_eq!(job.fraction(), Some(1.0));
        assert!(job.describe().starts_with("done — 2 images in"));
        assert!(!job.cancel()); // nothing left to cancel
        assert!(images.job("img_nope").unwrap_err().to_string().contains("there is no job 'img_nope'"));
    }

    #[tokio::test]
    async fn cancelling_the_caller_cancels_the_job_and_only_so_many_run_at_once() {
        no_keys();
        let images = generator(vec![FakeImages::new().with_steps(50, 0.02).into()], ImageOptions { max_concurrency: 1, ..ImageOptions::default() });
        let caller = images.clone();
        let call = tokio::spawn(async move { caller.generate("a fox", GenerateOptions::default()).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let waiting = images.submit("a second fox", opts()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(waiting.status(), "queued"); // one at a time was asked for
        call.abort();
        assert!(call.await.unwrap_err().is_cancelled());
        tokio::time::sleep(Duration::from_millis(20)).await;
        let first = images.jobs()[0].clone();
        assert_eq!(first.status(), "cancelled");
        assert!(first.describe().contains("cancelled after"));
        waiting.cancel();
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[tokio::test]
    async fn a_model_may_only_name_what_it_was_shown() {
        no_keys();
        let dir = temp_dir("secret");
        let secret = dir.join("secret.png");
        std::fs::write(&secret, &*RED).unwrap();
        let tools = generator(vec![FakeImages::new().into()], ImageOptions::default()).tools(ImageToolOptions::default());
        let generate = &tools[0];
        for reference in [secret.to_string_lossy().into_owned(), "/etc/passwd".into(), "../secret.png".into()] {
            let outcome = generate.run("c", args(json!({"prompt": "a fox", "references": [reference]})), ToolContext::default()).await;
            assert!(outcome.is_error && outcome.content.contains("there is no image called"), "{}", outcome.content);
        }
        let outcome = generate.run("c", args(json!({"prompt": "a fox", "aspect_ratio": "7:3"})), ToolContext::default()).await;
        assert!(outcome.is_error && outcome.content.contains("aspect_ratio is one of"));

        // With no workspace the image is kept in memory, and can still be built on.
        let made = generate.run("c", args(json!({"prompt": "a fox", "name": "fox"})), ToolContext::default()).await;
        assert!(!made.is_error && made.content.contains("- fox.png — 64x64 png"), "{}", made.content);
        assert_eq!(made.media.len(), 1);
        assert_eq!(&made.media[0].as_media().unwrap().1.read().unwrap()[..4], b"\x89PNG");
        let again = generate.run("c", args(json!({"prompt": "now at night", "references": ["fox"]})), ToolContext::default()).await;
        assert!(!again.is_error, "{}", again.content);

        let quiet = generator(vec![FakeImages::new().into()], ImageOptions::default()).tools(ImageToolOptions {
            show: false,
            references: vec![BLUE.clone().into()],
            ..ImageToolOptions::default()
        });
        let outcome = quiet[0].run("c", args(json!({"prompt": "a fox"})), ToolContext::default()).await;
        assert!(outcome.media.is_empty() && outcome.content.contains("Generated 1 image"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_job_that_outlives_the_wait_is_collected_with_image_status() {
        no_keys();
        let images = generator(vec![FakeImages::new().with_steps(6, 0.03).into()], ImageOptions::default());
        let tools = images.tools(ImageToolOptions { wait: 0.05, ..ImageToolOptions::default() });
        let (generate, status) = (&tools[0], &tools[1]);
        let ctx = ToolContext::default;
        assert_eq!(status.invoke(Map::new(), ctx()).await.unwrap(), json!("No images have been asked for."));

        let said = generate.invoke(args(json!({"prompt": "a detailed map of a city"})), ctx()).await.unwrap();
        let said = said.as_str().unwrap().to_string();
        assert!(said.starts_with("Still working: job img_") && said.contains("do not start the same"), "{said}");
        let job_id = said.split("job ").nth(1).unwrap().split(' ').next().unwrap().to_string();
        assert!(said.contains(&format!("image_status(job_id=\"{job_id}")));

        assert!(status.invoke(args(json!({"job_id": job_id})), ctx()).await.unwrap().as_str().unwrap().contains("Still working"));
        assert!(status.invoke(Map::new(), ctx()).await.unwrap().as_str().unwrap().contains(&format!("{job_id}: running on fake")));
        let collected = status.invoke(args(json!({"job_id": job_id, "wait": 5})), ctx()).await.unwrap();
        assert!(collected[0].as_str().unwrap().contains("Generated 1 image with fake"));
        assert_eq!(collected[1]["type"], "image");

        let second = generate.invoke(args(json!({"prompt": "another map"})), ctx()).await.unwrap();
        let second_id = second.as_str().unwrap().split("job ").nth(1).unwrap().split(' ').next().unwrap().to_string();
        let stopped = status.invoke(args(json!({"job_id": second_id, "cancel": true})), ctx()).await.unwrap();
        assert!(stopped.as_str().unwrap().contains("is being cancelled"));
        let missing = status.run("c", args(json!({"job_id": "img_nope"})), ctx()).await;
        assert!(missing.is_error && missing.content.contains("there is no job"));
    }

    #[tokio::test]
    async fn an_agent_generates_into_its_workspace_and_from_references_and_pays_for_it() {
        no_keys();
        let dir = temp_dir("workspace");
        std::fs::write(dir.join("storefront.jpg"), JPEG).unwrap();
        let engine = Arc::new(FakeImages::new());
        let images = generator(vec![engine.clone().into()], ImageOptions { cost_per_image: Some(0.04), ..ImageOptions::default() });
        let tools = images.tools(ImageToolOptions::default());
        let guard = Arc::new(BudgetGuard::new(Budget::default()));
        let ctx = ToolContext {
            agent: "designer".into(),
            run: Some(Arc::new(RunScope { guard: Some(guard.clone()), ..RunScope::default() })),
            ..ToolContext::default()
        };
        {
            let mut state = lock(&ctx.state);
            state.insert("workspace".into(), json!(dir.to_string_lossy()));
            let mut logo = MediaBlock::from_bytes(&BLUE, "image/png");
            logo.name = "logo.png".into();
            state.insert("attachments".into(), json!([ContentBlock::Image(logo)]));
        }
        let first = tools[0]
            .run("c1", args(json!({"prompt": "a poster for the autumn sale", "references": ["storefront.jpg", "attachment:1"], "aspect_ratio": "3:4", "name": "poster"})), ctx.clone())
            .await;
        assert!(!first.is_error, "{}", first.content);
        assert!(first.content.contains("Generated 1 image with fake (fake-image-1)"));
        assert!(first.content.contains("- images/poster.png — 48x64 png"), "{}", first.content);
        assert!(first.content.contains("references=[\"images/poster.png\"]"));
        assert_eq!(first.media.len(), 1);
        let second =
            tools[0].run("c2", args(json!({"prompt": "the same, with warmer light", "references": ["images/poster.png"], "count": 2})), ctx.clone()).await;
        assert!(!second.is_error, "{}", second.content);
        assert_eq!(second.media.len(), 2);

        let mut files: Vec<String> = std::fs::read_dir(dir.join("images")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        files.sort();
        assert_eq!(files, ["poster.png", "the-same-with-warmer-light-1.png", "the-same-with-warmer-light-2.png"]);
        let poster = std::fs::read(dir.join("images/poster.png")).unwrap();
        assert_eq!(sniff(&poster), Some(("image/png", 48, 64)));
        let requests = engine.requests();
        assert_eq!(requests[0].references.iter().map(|r| r.media_type.as_str()).collect::<Vec<_>>(), ["image/jpeg", "image/png"]);
        assert_eq!(requests[0].references[1].data, *BLUE);
        assert_eq!(requests[1].references[0].data, poster);
        let spent = guard.usage().cost_usd;
        assert!((0.119..0.121).contains(&spent), "{spent}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[ignore = "live: needs AH_LIVE_IMAGES=1 and a provider key"]
    async fn live_generation() {
        if std::env::var("AH_LIVE_IMAGES").is_err() {
            return;
        }
        let images = ImageGenerator::new(ImageOptions::default()).unwrap();
        let made = images.generate("a small red square on white", opts()).await.unwrap();
        assert!(sniff(&made[0].data).is_some());
    }
}
