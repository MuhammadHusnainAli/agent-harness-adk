//! The Chrome DevTools Protocol, over the harness's own WebSocket client.
//!
//! A browser is driven by sending it JSON commands and reading JSON events
//! back. This is the connection: commands answered by id, events handed to
//! listeners, and every waiting caller told at once when the browser goes away.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value, json};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::errors::{Error, Result};
use crate::voice::ws::{WebSocket, WsMessage};

/// `listener(method, params, session_id)`, for every event.
pub type Listener = Arc<dyn Fn(&str, &Value, &str) + Send + Sync>;

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>;

/// A DevTools failure: refused (`code` set), unanswered, or `gone`.
pub fn cdp_error(message: impl Into<String>, code: Option<i64>, gone: bool) -> Error {
    Error::Cdp { message: message.into(), code, gone }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

struct Shared {
    pending: Pending,
    listeners: Mutex<Vec<Listener>>,
    closed: AtomicBool,
}

impl Shared {
    fn lost(&self) {
        self.closed.store(true, Ordering::SeqCst);
        for (_, waiter) in lock(&self.pending).drain() {
            let _ = waiter.send(Err(cdp_error("the browser went away", None, true)));
        }
    }
}

/// One connection to a browser. Build it with [`Cdp::connect`].
pub struct Cdp {
    ws: Arc<WebSocket>,
    ids: AtomicU64,
    shared: Arc<Shared>,
    reader: Mutex<Option<JoinHandle<()>>>,
}

impl std::fmt::Debug for Cdp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<CDP {}>", if self.closed() { "closed" } else { "open" })
    }
}

impl Cdp {
    pub async fn connect(url: &str, timeout: f64) -> Result<Self> {
        let ws = WebSocket::connect_with(url, &[], timeout, 256 * 1024 * 1024).await?;
        Ok(Self::over(ws))
    }

    /// Over a WebSocket already open.
    pub fn over(ws: WebSocket) -> Self {
        let ws = Arc::new(ws);
        let shared = Arc::new(Shared { pending: Arc::default(), listeners: Mutex::default(), closed: AtomicBool::new(false) });
        let reader = tokio::spawn(read(ws.clone(), shared.clone()));
        Self { ws, ids: AtomicU64::new(1), shared, reader: Mutex::new(Some(reader)) }
    }

    pub fn closed(&self) -> bool {
        self.shared.closed.load(Ordering::SeqCst)
    }

    /// Call `listener(method, params, session_id)` for every event.
    pub fn on(&self, listener: Listener) {
        lock(&self.shared.listeners).push(listener);
    }

    /// Send one command and return its result.
    pub async fn call(&self, method: &str, params: Option<Value>, session: &str, timeout: f64) -> Result<Value> {
        if self.closed() {
            return Err(cdp_error("the browser is not connected", None, true));
        }
        let number = self.ids.fetch_add(1, Ordering::SeqCst);
        let mut message = json!({"id": number, "method": method, "params": params.unwrap_or_else(|| json!({}))});
        if !session.is_empty() {
            message["sessionId"] = json!(session);
        }
        let (tx, rx) = oneshot::channel();
        lock(&self.shared.pending).insert(number, tx);
        let forget = || {
            lock(&self.shared.pending).remove(&number);
        };
        if let Err(e) = self.ws.send_text(&message.to_string()).await {
            forget();
            if matches!(e, Error::ConnectionClosed { .. }) {
                self.shared.lost();
                return Err(cdp_error("the browser went away", None, true));
            }
            return Err(e);
        }
        match tokio::time::timeout(Duration::from_secs_f64(timeout.max(0.01)), rx).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => Err(cdp_error("the browser went away", None, true)),
            Err(_) => {
                forget();
                Err(cdp_error(format!("the browser did not answer {method} within {}s", crate::toolkits::search::fmt_g(timeout)), None, false))
            }
        }
    }

    /// Stop reading and close the socket. Every waiting caller is told.
    pub async fn close(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        if let Some(reader) = lock(&self.reader).take() {
            reader.abort();
        }
        self.ws.close(1000).await;
        self.shared.lost();
    }

    /// Give up on the connection without waiting for anything.
    pub(crate) fn abandon(&self) {
        self.shared.closed.store(true, Ordering::SeqCst);
        if let Some(reader) = lock(&self.reader).take() {
            reader.abort();
        }
        self.shared.lost();
    }
}

impl Drop for Cdp {
    fn drop(&mut self) {
        self.abandon();
    }
}

async fn read(ws: Arc<WebSocket>, shared: Arc<Shared>) {
    loop {
        let raw = match ws.recv().await {
            Ok(WsMessage::Text(t)) => t,
            Ok(WsMessage::Binary(b)) => String::from_utf8_lossy(&b).into_owned(),
            Err(_) => break,
        };
        let Ok(Value::Object(message)) = serde_json::from_str::<Value>(&raw) else { continue };
        if let Some(id) = message.get("id") {
            let Some(waiter) = id.as_u64().and_then(|n| lock(&shared.pending).remove(&n)) else { continue };
            let answer = match message.get("error").filter(|e| !e.is_null() && e.as_object().is_none_or(|o| !o.is_empty())) {
                Some(error) => {
                    let text = error.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).map_or_else(|| error.to_string(), str::to_string);
                    Err(cdp_error(text, error.get("code").and_then(Value::as_i64), false))
                }
                None => Ok(message.get("result").filter(|r| !r.is_null()).cloned().unwrap_or_else(|| Value::Object(Map::new()))),
            };
            let _ = waiter.send(answer);
            continue;
        }
        let Some(method) = message.get("method").and_then(Value::as_str).filter(|m| !m.is_empty()) else { continue };
        let params = message.get("params").filter(|p| !p.is_null()).cloned().unwrap_or_else(|| Value::Object(Map::new()));
        let session = message.get("sessionId").and_then(Value::as_str).unwrap_or("");
        let listeners = lock(&shared.listeners).clone();
        for listener in listeners {
            // A listener must not end the reader.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener(method, &params, session)));
        }
    }
    shared.lost();
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use base64::Engine;
    use sha1::{Digest, Sha1};
    use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::sync::mpsc;

    /// A WebSocket server for tests: every text frame from the client goes to
    /// `handler`, which answers with any number of text frames.
    pub(crate) async fn scripted_server(handler: impl Fn(Value) -> Vec<Value> + Send + Sync + 'static) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            let (sock, _) = listener.accept().await.unwrap();
            let (r, mut w) = tokio::io::split(sock);
            let mut r = BufReader::new(r);
            let mut head = Vec::new();
            let mut b = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                r.read_exact(&mut b).await.unwrap();
                head.push(b[0]);
            }
            let text = String::from_utf8_lossy(&head).to_string();
            let key = text.lines().find_map(|l| l.strip_prefix("Sec-WebSocket-Key: ")).unwrap().trim().to_string();
            let accept = base64::engine::general_purpose::STANDARD.encode(Sha1::digest(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes()));
            w.write_all(format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").as_bytes()).await.unwrap();
            let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
            tokio::spawn(async move {
                while let Some(frame) = rx.recv().await {
                    if w.write_all(&frame).await.is_err() {
                        return;
                    }
                }
            });
            let frame = |payload: &[u8]| {
                let mut out = vec![0x81u8];
                if payload.len() < 126 {
                    out.push(payload.len() as u8);
                } else {
                    out.push(126);
                    out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
                }
                out.extend_from_slice(payload);
                out
            };
            loop {
                let mut two = [0u8; 2];
                if r.read_exact(&mut two).await.is_err() {
                    return;
                }
                let mut size = (two[1] & 0x7F) as usize;
                if size == 126 {
                    let mut s = [0u8; 2];
                    r.read_exact(&mut s).await.unwrap();
                    size = u16::from_be_bytes(s) as usize;
                }
                let mut key = [0u8; 4];
                r.read_exact(&mut key).await.unwrap();
                let mut payload = vec![0u8; size];
                r.read_exact(&mut payload).await.unwrap();
                for (i, byte) in payload.iter_mut().enumerate() {
                    *byte ^= key[i % 4];
                }
                if two[0] & 0x0F == 0x8 {
                    let _ = tx.send(vec![0x88, 2, 0x03, 0xE8]);
                    return;
                }
                let Ok(message) = serde_json::from_slice::<Value>(&payload) else { continue };
                for answer in handler(message) {
                    let _ = tx.send(frame(answer.to_string().as_bytes()));
                }
            }
        });
        format!("ws://{addr}/devtools/browser/x")
    }

    #[tokio::test]
    async fn commands_are_answered_by_id_and_events_reach_listeners() {
        let url = scripted_server(|m| match m["method"].as_str().unwrap() {
            "Ok.method" => vec![json!({"method": "Page.loadEventFired", "params": {}, "sessionId": "s1"}), json!({"id": m["id"], "result": {"n": 1}})],
            "Bad.method" => vec![json!({"id": m["id"], "error": {"code": -32601, "message": "'Bad.method' wasn't found"}})],
            _ => vec![],
        })
        .await;
        let cdp = Cdp::connect(&url, 5.0).await.unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let into = seen.clone();
        cdp.on(Arc::new(move |method, _params, session| into.lock().unwrap().push(format!("{method}@{session}"))));
        assert_eq!(cdp.call("Ok.method", None, "s1", 5.0).await.unwrap(), json!({"n": 1}));
        assert_eq!(seen.lock().unwrap().clone(), vec!["Page.loadEventFired@s1"]);
        let Error::Cdp { message, code, gone } = cdp.call("Bad.method", None, "", 5.0).await.unwrap_err() else { panic!() };
        assert_eq!((message.as_str(), code, gone), ("'Bad.method' wasn't found", Some(-32601), false));
        let slow = cdp.call("Silent.method", None, "", 0.2).await.unwrap_err();
        assert_eq!(slow.to_string(), "the browser did not answer Silent.method within 0.2s");
        cdp.close().await;
        assert!(matches!(cdp.call("Ok.method", None, "", 1.0).await, Err(Error::Cdp { gone: true, .. })));
    }
}
