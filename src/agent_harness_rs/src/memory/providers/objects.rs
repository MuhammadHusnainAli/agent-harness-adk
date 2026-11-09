//! Shared object-store implementation: S3, Azure Blob, Google Cloud Storage.
//!
//! Keys are laid out so the trace and the scope are *prefixes*, which is the
//! only filter an object store can push down:
//!
//! ```text
//! {prefix}/records/{trace}/{scope}/{ts}-{id}.json
//! {prefix}/docs/{trace}/{name}
//! ```
//!
//! That makes reading one user's memory a single prefix listing. Anything
//! narrower — a `kind`, a keyword — is filtered after the fetch, so treat an
//! object store as durable archival for memory rather than a hot query path.
//! For heavy recall put SQL or Mongo in front of it.
//!
//! The keys and the JSON are the Python harness's, so a bucket written by one
//! is read by the other. One difference: Python writes `/records/...` (with a
//! leading slash) when the prefix is empty and then cannot list it back; here
//! an empty prefix simply drops out of the key.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::knowledge::base::HttpClient;
use crate::llm_providers::resilience::retry_after;
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, last};
use crate::memory::trace::Trace;
use crate::runtime::session::SessionStore;
use crate::sessions::objects::ObjectSessionStore;

/// An object as it was read: its bytes, and the tag (an ETag, a GCS
/// generation) a conditional write can name.
#[derive(Debug, Clone, PartialEq)]
pub struct Fetched {
    pub body: Vec<u8>,
    pub tag: Option<String>,
}

/// When a write may go ahead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PutIf {
    /// Whatever is there.
    Always,
    /// Only if nothing is there yet.
    Absent,
    /// Only if what is there still carries this tag.
    Matches(String),
}

/// What a bucket must provide. Four calls; everything else is built on them.
#[async_trait]
pub trait ObjectBackend: Send + Sync {
    /// Write one object. `false` when `condition` did not hold.
    async fn put(&self, key: &str, body: Vec<u8>, content_type: &str, condition: &PutIf) -> Result<bool>;
    /// Read one object; `None` when it is not there.
    async fn get(&self, key: &str) -> Result<Option<Fetched>>;
    /// Every key under `prefix`.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
    /// Remove these keys. A key that is already gone is what was wanted.
    async fn delete(&self, keys: &[String]) -> Result<()>;
    /// What the bucket is, for messages.
    fn describe(&self) -> String;
}

/// Records as one small object each, documents as one object per name.
///
/// `S3Memory`, `GcsMemory` and `AzureBlobMemory` are this over their bucket.
pub struct ObjectStoreMemory<B> {
    objects: Arc<B>,
    /// Every key starts with this (no slashes at either end).
    pub prefix: String,
    /// How many objects are fetched at once.
    pub concurrency: usize,
    /// Do sessions on this bucket use conditional writes (ETags, generations)?
    /// On by default; turn it off for an S3 look-alike that refuses them.
    pub conditional: bool,
}

impl<B> std::fmt::Debug for ObjectStoreMemory<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<ObjectStoreMemory {}>", self.prefix)
    }
}

impl<B: ObjectBackend + 'static> ObjectStoreMemory<B> {
    pub fn new(objects: B, prefix: &str) -> Self {
        Self { objects: Arc::new(objects), prefix: prefix.trim_matches('/').to_string(), concurrency: 16, conditional: true }
    }

    /// The bucket underneath.
    pub fn objects(&self) -> &Arc<B> {
        &self.objects
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    pub fn with_conditional(mut self, conditional: bool) -> Self {
        self.conditional = conditional;
        self
    }

    /// Take the shared URL options: `concurrency`, `conditional`.
    pub(crate) fn configured(mut self, options: &[(String, String)]) -> Self {
        if let Some(n) = option(options, "concurrency").and_then(|v| v.parse().ok()) {
            self.concurrency = usize::max(n, 1);
        }
        if let Some(v) = option(options, "conditional") {
            self.conditional = truthy(v);
        }
        self
    }

    fn join(parts: &[&str]) -> String {
        parts.iter().filter(|p| !p.is_empty()).copied().collect::<Vec<_>>().join("/")
    }

    /// `{prefix}/records/{trace}/[{scope}/]`.
    pub fn records_prefix(&self, trace: Option<&Trace>, scope: Option<&str>) -> String {
        let slug = trace.map(Trace::slug).unwrap_or_default();
        format!("{}/", Self::join(&[&self.prefix, "records", &slug, scope.unwrap_or("")]))
    }

    /// Where a record is kept. The timestamp leads so a prefix listing comes
    /// back in time order.
    pub fn record_key(&self, record: &MemoryRecord) -> String {
        let file = format!("{:.6}-{}.json", record.ts, record.id);
        Self::join(&[&self.prefix, "records", &slug_of(record), &record.scope, &file])
    }

    /// Where a document is kept.
    pub fn doc_object(&self, name: &str, trace: Option<&Trace>) -> String {
        let namespace = trace.map_or_else(|| "_shared".to_string(), Trace::slug);
        Self::join(&[&self.prefix, "docs", &namespace, name])
    }
}

/// The namespace a record belongs to, from the identity stamped on it.
pub fn slug_of(record: &MemoryRecord) -> String {
    Trace { user_id: record.user_id.clone(), session_id: record.session_id.clone(), tenant_id: record.tenant_id.clone(), ..Trace::default() }.slug()
}

#[async_trait]
impl<B: ObjectBackend + 'static> MemoryStore for ObjectStoreMemory<B> {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        let body = serde_json::to_vec(&record).map_err(|e| Error::Io(e.to_string()))?;
        self.objects.put(&self.record_key(&record), body, "application/json", &PutIf::Always).await?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let mut keys = self.objects.list(&self.records_prefix(query.trace.as_ref(), query.scope.as_deref())).await?;
        keys.sort(); // the key starts with the time
        let limit = query.limit.filter(|n| *n > 0);
        if let Some(n) = limit
            && query.kind.is_none()
            && keys.len() > n
        {
            keys.drain(..keys.len() - n); // newest, before fetching
        }
        let objects = &self.objects;
        let fetched: Vec<Result<Option<MemoryRecord>>> = futures::stream::iter(keys)
            .map(|key| async move { Ok(objects.get(&key).await?.and_then(|f| serde_json::from_slice::<MemoryRecord>(&f.body).ok())) })
            .buffered(self.concurrency.max(1))
            .collect()
            .await;
        let mut rows = Vec::new();
        for r in fetched {
            if let Some(r) = r? {
                rows.push(r);
            }
        }
        rows.retain(|r| query.kind.as_ref().is_none_or(|k| &r.kind == k) && query.trace.as_ref().is_none_or(|t| t.matches(r)));
        rows.sort_by(|a, b| a.ts.total_cmp(&b.ts));
        Ok(last(rows, limit))
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        let mut keys = self.objects.list(&self.records_prefix(trace, scope)).await?;
        if let Some(t) = trace {
            keys.extend(self.objects.list(&format!("{}/", Self::join(&[&self.prefix, "docs", &t.slug()]))).await?);
        } else if scope.is_none() {
            keys.extend(self.objects.list(&format!("{}/", Self::join(&[&self.prefix, "docs"]))).await?);
        }
        if !keys.is_empty() {
            self.objects.delete(&keys).await?;
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        Ok(self.objects.get(&self.doc_object(name, trace)).await?.map(|f| String::from_utf8_lossy(&f.body).into_owned()).unwrap_or_default())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        let content_type = if name.ends_with(".md") { "text/markdown" } else { "text/plain" };
        self.objects.put(&self.doc_object(name, trace), text.as_bytes().to_vec(), content_type, &PutIf::Always).await?;
        Ok(())
    }

    fn describe(&self) -> String {
        format!("{}/{}", self.objects.describe(), self.prefix)
    }

    /// Sessions in the same bucket, under `{prefix}/sessions/`. Options:
    /// `concurrency` (default 16), `conditional`.
    fn sessions(&self, options: &Map<String, Value>) -> Option<Result<Arc<dyn SessionStore>>> {
        let objects: Arc<dyn ObjectBackend> = self.objects.clone();
        let mut store = ObjectSessionStore::new(objects, &self.prefix);
        store.conditional = self.conditional;
        if let Some(n) = options.get("concurrency").and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))) {
            store.concurrency = (n as usize).max(1);
        }
        if let Some(v) = options.get("conditional") {
            store.conditional = v.as_bool().unwrap_or_else(|| v.as_str().is_some_and(truthy));
        }
        Some(Ok(Arc::new(store)))
    }
}

// ---- URLs -------------------------------------------------------------------------

/// A connection URL taken apart: `scheme://[user[:pass]@]host/path?k=v`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UrlParts {
    pub scheme: String,
    pub user: Option<String>,
    pub password: Option<String>,
    pub host: String,
    /// Without the leading slash.
    pub path: String,
    pub query: Vec<(String, String)>,
}

/// Split a URL the way Python's `urlparse` does for these backends.
pub fn split_url(url: &str) -> UrlParts {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (netloc, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, host) = match netloc.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, netloc),
    };
    let (user, password) = match userinfo {
        Some(u) => match u.split_once(':') {
            Some((a, b)) => (Some(decode(a)), Some(decode(b))),
            None => (Some(decode(u)), None),
        },
        None => (None, None),
    };
    let query = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k), decode(v))
        })
        .collect();
    UrlParts { scheme: scheme.to_ascii_lowercase(), user, password, host: host.to_string(), path: path.to_string(), query }
}

/// Percent-decoding of one URL component.
pub fn decode(text: &str) -> String {
    crate::knowledge::stores::decode(text)
}

/// Percent-encoding of one URL component (`/` kept when `keep_slash`).
pub fn encode(text: &str, keep_slash: bool) -> String {
    crate::llm_providers::sigv4::uri_encode(text, keep_slash)
}

/// The first value of a URL option, if it was given and is not empty.
pub fn option<'a>(options: &'a [(String, String)], name: &str) -> Option<&'a str> {
    options.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str()).filter(|v| !v.is_empty())
}

/// `true`, `1`, `yes`, `on`.
pub fn truthy(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

/// An environment variable, if it is set and not empty.
pub fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

// ---- raw HTTP ---------------------------------------------------------------------

/// What an object request came back with, untouched: the bytes and headers
/// (an ETag, a generation) that a JSON reply would lose.
#[derive(Debug, Clone)]
pub struct RawReply {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl RawReply {
    pub fn header(&self, name: &str) -> Option<String> {
        self.headers.get(name).and_then(|v| v.to_str().ok()).map(str::to_string)
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// [`HttpClient::send`] for objects: the same client, retries, back-off,
/// signer and redaction, but the body comes back as bytes, with the headers.
/// `ok` lists statuses besides 2xx that are an answer, not a failure.
pub async fn send_raw(http: &HttpClient, method: Method, url: &str, body: Vec<u8>, extra: &[(&str, String)], ok: &[u16]) -> Result<RawReply> {
    let started = Instant::now();
    let mut attempt = 0u32;
    loop {
        let mut headers = HeaderMap::new();
        for (k, v) in http.headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).chain(extra.iter().map(|(k, v)| (*k, v.as_str()))) {
            if let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(v)) {
                headers.insert(k, v);
            }
        }
        if let Some(sign) = &http.signer {
            sign(method.as_str(), url, &body, &mut headers)?;
        }
        let mut request = http.client.request(method.clone(), url).headers(headers);
        if !body.is_empty() || matches!(method, Method::PUT | Method::POST) {
            request = request.body(body.clone());
        }
        let (problem, wait) = match request.send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                let headers = response.headers().clone();
                let bytes = response.bytes().await.map(|b| b.to_vec()).unwrap_or_default();
                if (200..300).contains(&status) || ok.contains(&status) {
                    return Ok(RawReply { status, headers, body: bytes });
                }
                let text = String::from_utf8_lossy(&bytes);
                let squeezed: String = text.chars().take(800).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
                let detail: String = http.redact(&squeezed).chars().take(400).collect();
                let again = matches!(status, 408 | 425 | 429) || status >= 500;
                let path = url.split('?').next().unwrap_or(url);
                let problem = Error::VectorStore {
                    message: format!("{}: {} {} → {status}: {detail}", http.name, method, http.redact(path)),
                    store: http.name.clone(),
                    status: Some(status),
                    retryable: again,
                };
                (problem, retry_after(Some(&headers), None, status == 429 || status == 503))
            }
            Err(e) => {
                // Only the kind: the message of a transport error can carry the URL.
                let kind = if e.is_timeout() {
                    "TimeoutError"
                } else if e.is_connect() {
                    "ConnectError"
                } else {
                    "TransportError"
                };
                (
                    Error::VectorStore {
                        message: format!("{} could not be reached ({kind})", http.name),
                        store: http.name.clone(),
                        status: None,
                        retryable: true,
                    },
                    None,
                )
            }
        };
        let retryable = matches!(problem, Error::VectorStore { retryable: true, .. });
        if !retryable || attempt >= http.retry.max_retries {
            return Err(problem);
        }
        let delay = wait.unwrap_or_else(|| http.retry.backoff(attempt));
        if delay > http.retry.max_retry_after || http.retry.max_elapsed.is_some_and(|m| started.elapsed().as_secs_f64() + delay > m) {
            return Err(problem);
        }
        tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        attempt += 1;
    }
}

// ---- XML (S3 and Azure listings) --------------------------------------------------

/// The text of every `<tag>…</tag>` in `xml`, unescaped. Enough for the
/// listings S3 and Azure send back, which are flat and attribute-free where
/// it matters.
pub fn xml_values(xml: &str, tag: &str) -> Vec<String> {
    let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let after = &rest[start + open.len()..];
        let Some(end) = after.find(&close) else { break };
        out.push(xml_unescape(&after[..end]));
        rest = &after[end + close.len()..];
    }
    out
}

/// The first `<tag>` value, if there is one.
pub fn xml_value(xml: &str, tag: &str) -> Option<String> {
    xml_values(xml, tag).into_iter().next()
}

pub fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&apos;")
}

pub fn xml_unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(end) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let entity = &tail[1..end];
        let ch = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            e if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
            e if e.starts_with('#') => e[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => out.push(c),
            None => out.push_str(&tail[..=end]),
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
pub(crate) mod testing {
    //! What every memory backend and session store is held to, and an
    //! in-process bucket to hold the object stores to it without a network.

    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::*;
    use crate::runtime::session::{ListFilter, Session};
    use crate::types::Message;

    /// A bucket in a map, with tags that change on every write.
    #[derive(Default)]
    pub struct MapBucket {
        pub objects: Mutex<BTreeMap<String, (Vec<u8>, u64)>>,
        pub calls: Mutex<Vec<&'static str>>,
        counter: std::sync::atomic::AtomicU64,
    }

    #[async_trait]
    impl ObjectBackend for MapBucket {
        async fn put(&self, key: &str, body: Vec<u8>, _content_type: &str, condition: &PutIf) -> Result<bool> {
            self.calls.lock().unwrap().push("put");
            let mut all = self.objects.lock().unwrap();
            let held = all.get(key).map(|(_, tag)| tag.to_string());
            let fine = match condition {
                PutIf::Always => true,
                PutIf::Absent => held.is_none(),
                PutIf::Matches(t) => held.as_deref() == Some(t.as_str()),
            };
            if fine {
                let tag = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                all.insert(key.to_string(), (body, tag));
            }
            Ok(fine)
        }

        async fn get(&self, key: &str) -> Result<Option<Fetched>> {
            self.calls.lock().unwrap().push("get");
            Ok(self.objects.lock().unwrap().get(key).map(|(b, t)| Fetched { body: b.clone(), tag: Some(t.to_string()) }))
        }

        async fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.calls.lock().unwrap().push("list");
            Ok(self.objects.lock().unwrap().keys().filter(|k| k.starts_with(prefix)).cloned().collect())
        }

        async fn delete(&self, keys: &[String]) -> Result<()> {
            self.calls.lock().unwrap().push("delete");
            let mut all = self.objects.lock().unwrap();
            for k in keys {
                all.remove(k);
            }
            Ok(())
        }

        fn describe(&self) -> String {
            "map".into()
        }
    }

    fn stamped(trace: &Trace, text: &str, ts: Option<f64>) -> MemoryRecord {
        let mut r = MemoryRecord::new("user", "fact", text);
        if let Some(ts) = ts {
            r.ts = ts;
        }
        trace.stamp(&mut r);
        r
    }

    /// Scoping by trace, documents, clear: what an object store (or any
    /// backend that keys records by trace) must do. Leaves the store empty.
    pub async fn exercise_memory(store: &dyn MemoryStore) {
        let (alice, bob) = (Trace::user("alice"), Trace::user("bob"));
        store.clear(None, None).await.unwrap();
        store.append(stamped(&alice, "alice's fact about refunds", Some(1000.0))).await.unwrap();
        store.append(stamped(&alice, "alice's second", Some(1001.0))).await.unwrap();
        store.append(stamped(&bob, "bob's fact", Some(1002.0))).await.unwrap();

        let mine: Vec<String> = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().into_iter().map(|r| r.text).collect();
        assert_eq!(mine, ["alice's fact about refunds", "alice's second"]);
        let bobs: Vec<String> = store.all(RecordQuery::default().traced(Some(&bob))).await.unwrap().into_iter().map(|r| r.text).collect();
        assert_eq!(bobs, ["bob's fact"]);
        let newest = store.all(RecordQuery::default().traced(Some(&alice)).limit(1)).await.unwrap();
        assert_eq!(newest[0].text, "alice's second");
        assert_eq!(store.all(RecordQuery::scope("user").traced(Some(&alice)).kind("note")).await.unwrap().len(), 0);
        let found = store.search("refunds", None, 5, Some(&alice)).await.unwrap();
        assert_eq!(found[0].text, "alice's fact about refunds");

        store.write_doc("user.md", "- alice", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "- bob", Some(&bob)).await.unwrap();
        store.write_doc("shared.md", "{\"a\": 1}", None).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- alice");
        assert_eq!(store.read_doc("user.md", Some(&bob)).await.unwrap(), "- bob");
        assert_eq!(store.read_doc("shared.md", None).await.unwrap(), "{\"a\": 1}");
        assert_eq!(store.read_doc("nothing.md", Some(&alice)).await.unwrap(), "");

        store.clear(None, Some(&alice)).await.unwrap();
        assert!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().is_empty());
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        assert_eq!(store.read_doc("user.md", Some(&bob)).await.unwrap(), "- bob");
        assert_eq!(store.all(RecordQuery::default().traced(Some(&bob))).await.unwrap().len(), 1);

        store.clear(None, None).await.unwrap();
        assert!(store.all(RecordQuery::default()).await.unwrap().is_empty());
        assert_eq!(store.read_doc("user.md", Some(&bob)).await.unwrap(), "");
    }

    fn conversation(title: &str, user: &str, tenant: &str, agent: &str) -> Session {
        let mut s = Session { title: title.into(), user_id: Some(user.into()), tenant_id: Some(tenant.into()), ..Session::new(agent) };
        s.messages = vec![Message::user("Where is my order?"), Message::assistant("It shipped ✓")];
        s
    }

    /// Everything a chat needs of a store (the Python `exercise`). Leaves
    /// nothing behind. `atomic` also races eight saves from one version.
    pub async fn exercise_sessions(store: &dyn SessionStore, atomic: bool) {
        let mut first = conversation("Order 4182", "alice", "acme", "support");
        store.save(&mut first).await.unwrap();
        assert_eq!(first.version, 1);
        let mut back = store.load(&first.id).await.unwrap();
        assert_eq!((back.title.as_str(), back.messages.len(), back.user_id.as_deref()), ("Order 4182", 2, Some("alice")));
        back.messages.push(Message::user("Thanks."));
        store.save(&mut back).await.unwrap();
        assert_eq!(store.load(&first.id).await.unwrap().version, 2);
        assert_eq!(store.load(&first.id).await.unwrap().messages.len(), 3);

        // A stale save is refused and leaves both sides as they were.
        let mut stale = store.load(&first.id).await.unwrap();
        let mut fresh = store.load(&first.id).await.unwrap();
        fresh.title = "saved first".into();
        store.save(&mut fresh).await.unwrap();
        stale.title = "saved second".into();
        assert!(matches!(store.save(&mut stale).await, Err(Error::SessionConflict { .. })));
        assert_eq!(stale.version, 2);
        let held = store.load(&first.id).await.unwrap();
        assert_eq!((held.title.as_str(), held.version), ("saved first", 3));

        // Two that both think they are creating it: one does.
        let mut twin = Session { id: first.id.clone(), user_id: Some("alice".into()), tenant_id: Some("acme".into()), ..Session::new("support") };
        assert!(matches!(store.save(&mut twin).await, Err(Error::SessionConflict { .. })));

        // Listing: newest first, narrowed to whose it is.
        let mut made = vec![first.id.clone()];
        let mut second = conversation("Bob's", "bob", "acme", "support");
        let mut third = conversation("Another tenant", "alice", "globex", "support");
        let mut fourth = conversation("Alice again", "alice", "acme", "billing");
        for s in [&mut second, &mut third, &mut fourth] {
            tokio::time::sleep(Duration::from_millis(15)).await;
            store.save(s).await.unwrap();
            made.push(s.id.clone());
        }
        let owner = |user: &str, tenant: &str| ListFilter { limit: 50, user_id: Some(user.into()), tenant_id: Some(tenant.into()), agent: None };
        let titles = |rows: Vec<Session>| rows.into_iter().map(|s| s.title).collect::<Vec<_>>();
        assert_eq!(titles(store.list(&owner("alice", "acme")).await.unwrap()), ["Alice again", "saved first"]);
        assert_eq!(titles(store.list(&owner("bob", "acme")).await.unwrap()), ["Bob's"]);
        let acme: Vec<String> =
            store.list(&ListFilter { limit: 50, tenant_id: Some("acme".into()), ..Default::default() }).await.unwrap().into_iter().map(|s| s.id).collect();
        assert!(acme.contains(&first.id) && acme.contains(&second.id) && acme.contains(&fourth.id) && !acme.contains(&third.id));
        assert_eq!(titles(store.list(&ListFilter { agent: Some("billing".into()), ..owner("alice", "acme") }).await.unwrap()), ["Alice again"]);
        assert_eq!(titles(store.list(&ListFilter { limit: 1, ..owner("alice", "acme") }).await.unwrap()), ["Alice again"]);
        assert!(store.list(&owner("nobody", "acme")).await.unwrap().is_empty());

        // A long conversation, larger than any one row or item is allowed.
        let mut long = conversation("long", "alice", "acme", "support");
        for n in 0..700 {
            long.messages.push(Message::user(format!("turn {n} {}", crate::types::new_id("x").repeat(50))));
        }
        store.save(&mut long).await.unwrap();
        made.push(long.id.clone());
        let mut long = store.load(&long.id).await.unwrap();
        assert_eq!(long.messages.len(), 702);
        long.messages.truncate(10);
        store.save(&mut long).await.unwrap();
        assert_eq!(store.load(&long.id).await.unwrap().messages.len(), 10);

        // A fork is its own session, owned by the same person.
        let fork = store.fork(&first.id, Some(2)).await.unwrap();
        made.push(fork.id.clone());
        assert!(fork.id != first.id && fork.parent_id.as_deref() == Some(first.id.as_str()));
        assert_eq!((fork.user_id.as_deref(), fork.tenant_id.as_deref(), fork.messages.len()), (Some("alice"), Some("acme"), 2));

        // Of many racing from one version, exactly one wins.
        if atomic {
            let base = store.load(&first.id).await.unwrap();
            let outcomes = futures::future::join_all((0..8).map(|_| {
                let mut copy = base.clone();
                async move { store.save(&mut copy).await }
            }))
            .await;
            assert_eq!(outcomes.iter().filter(|o| o.is_ok()).count(), 1, "{outcomes:?}");
            assert!(outcomes.iter().all(|o| o.is_ok() || matches!(o, Err(Error::SessionConflict { .. }))));
        }

        // Delete, and deleting what is not there is not an error.
        store.delete(&first.id).await.unwrap();
        assert!(matches!(store.load(&first.id).await, Err(Error::Configuration(m)) if m.contains("no session")));
        assert!(!store.list(&owner("alice", "acme")).await.unwrap().iter().any(|s| s.id == first.id));
        store.delete(&first.id).await.unwrap();
        assert!(store.load("ses_never_was").await.is_err());
        for id in made {
            store.delete(&id).await.unwrap();
        }
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");
    }

    pub fn map_store(prefix: &str) -> ObjectStoreMemory<MapBucket> {
        ObjectStoreMemory::new(MapBucket::default(), prefix)
    }

    /// Serve a fake service on 127.0.0.1 and hand back its base URL.
    pub async fn serve(router: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.ok();
        });
        format!("http://{address}")
    }

    /// The query string of a request, decoded into pairs.
    pub fn query_of(uri: &axum::http::Uri) -> Vec<(String, String)> {
        split_url(&format!("x://h/?{}", uri.query().unwrap_or(""))).query
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn record(text: &str, ts: f64, trace: &Trace) -> MemoryRecord {
        let mut r = MemoryRecord::new("user", "fact", text);
        r.ts = ts;
        trace.stamp(&mut r);
        r
    }

    #[tokio::test]
    async fn keys_are_laid_out_so_a_trace_is_a_prefix() {
        let store = map_store("mem");
        let (alice, bob) = (Trace::user("alice"), Trace::user("bob"));
        store.append(record("alice's fact", 1000.5, &alice)).await.unwrap();
        store.append(record("bob's fact", 1001.0, &bob)).await.unwrap();
        let keys: Vec<String> = store.objects().objects.lock().unwrap().keys().cloned().collect();
        assert!(keys[0].starts_with("mem/records/u-alice/user/1000.500000-mem_"), "{keys:?}");
        assert!(keys[1].starts_with("mem/records/u-bob/user/") && keys[1].ends_with(".json"));
        assert_eq!(store.records_prefix(Some(&alice), Some("user")), "mem/records/u-alice/user/");
        assert_eq!(store.records_prefix(None, None), "mem/records/");
        assert_eq!(store.doc_object("user.md", Some(&alice)), "mem/docs/u-alice/user.md");
        assert_eq!(store.doc_object("user.md", None), "mem/docs/_shared/user.md");
        assert_eq!(store.all(RecordQuery::default()).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn records_are_fetched_after_one_listing_and_a_limit_narrows_first() {
        let store = map_store("mem").with_concurrency(8);
        let alice = Trace::user("alice");
        for i in 0..12 {
            store.append(record(&format!("fact {i}"), 1000.0 + f64::from(i), &alice)).await.unwrap();
        }
        store.objects().calls.lock().unwrap().clear();
        let rows = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.text.clone()).collect::<Vec<_>>(), (0..12).map(|i| format!("fact {i}")).collect::<Vec<_>>());
        assert_eq!(store.objects().calls.lock().unwrap().iter().filter(|c| **c == "list").count(), 1);
        store.objects().calls.lock().unwrap().clear();
        let rows = store.all(RecordQuery::default().traced(Some(&alice)).limit(3)).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["fact 9", "fact 10", "fact 11"]);
        assert_eq!(store.objects().calls.lock().unwrap().iter().filter(|c| **c == "get").count(), 3);
    }

    #[tokio::test]
    async fn the_object_store_holds_the_memory_contract() {
        exercise_memory(&map_store("mem")).await;
    }

    #[tokio::test]
    async fn sessions_on_a_bucket_hold_the_contract_and_leave_nothing() {
        let store = map_store("chats");
        let sessions = store.sessions(&Map::new()).unwrap().unwrap();
        assert_eq!(sessions.kind(), "ObjectSessionStore");
        exercise_sessions(sessions.as_ref(), true).await;
        assert!(store.objects().objects.lock().unwrap().is_empty());
    }

    #[test]
    fn urls_and_xml_come_apart() {
        let u = split_url("s3://user:p%40ss@bucket/some/prefix?region=eu-west-1&endpoint=http%3A%2F%2Flocalhost%3A9000");
        assert_eq!((u.scheme.as_str(), u.host.as_str(), u.path.as_str()), ("s3", "bucket", "some/prefix"));
        assert_eq!((u.user.as_deref(), u.password.as_deref()), (Some("user"), Some("p@ss")));
        assert_eq!(option(&u.query, "endpoint"), Some("http://localhost:9000"));
        let xml = "<R><Contents><Key>a&amp;b</Key></Contents><Contents><Key>c&#x41;</Key></Contents><IsTruncated>false</IsTruncated></R>";
        assert_eq!(xml_values(xml, "Key"), ["a&b", "cA"]);
        assert_eq!(xml_value(xml, "IsTruncated").as_deref(), Some("false"));
        assert_eq!(xml_escape("<a&'>"), "&lt;a&amp;&apos;&gt;");
    }
}
