//! A WebSocket client, small enough to read in one sitting.
//!
//! Realtime speech models are reached over a WebSocket. Rather than add a
//! dependency for a handshake and a framing rule, this is RFC 6455's client
//! side over tokio: text and binary messages, fragmentation, ping and pong, an
//! orderly close. It does not negotiate compression — audio is already
//! compressed or does not compress — and it does not go through an HTTP proxy.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::voice::{WebSocket, WsMessage};
//! let ws = WebSocket::connect("wss://host/path", &[("authorization".into(), "…".into())], 20.0).await?;
//! ws.send_text(r#"{"type": "hello"}"#).await?;
//! while let Ok(message) = ws.recv().await {
//!     if let WsMessage::Text(t) = message { println!("{t}"); }
//! }
//! # Ok(()) }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::Engine;
use sha1::{Digest, Sha1};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, ReadHalf, WriteHalf};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};

const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const TEXT: u8 = 0x1;
const BINARY: u8 = 0x2;
const CLOSE: u8 = 0x8;
const PING: u8 = 0x9;
const PONG: u8 = 0xA;
const CONTINUATION: u8 = 0x0;

/// A whole message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    Text(String),
    Binary(Vec<u8>),
}

/// The other end closed the connection, or it was lost.
pub fn connection_closed(code: u16, reason: &str) -> Error {
    Error::ConnectionClosed { code, reason: reason.into() }
}

trait Stream: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Stream for T {}

type Reader = BufReader<ReadHalf<Box<dyn Stream>>>;
type Writer = WriteHalf<Box<dyn Stream>>;

fn random<const N: usize>() -> [u8; N] {
    let mut out = [0u8; N];
    let mut filled = 0;
    while filled < N {
        let bytes = *uuid::Uuid::new_v4().as_bytes();
        let take = (N - filled).min(16);
        out[filled..filled + take].copy_from_slice(&bytes[..take]);
        filled += take;
    }
    out
}

fn mask(data: &mut [u8], key: [u8; 4]) {
    for (i, b) in data.iter_mut().enumerate() {
        *b ^= key[i % 4];
    }
}

/// The client's frame: always masked, as RFC 6455 requires of a client.
fn frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let size = payload.len();
    let mut out = vec![0x80 | opcode];
    if size < 126 {
        out.push(0x80 | size as u8);
    } else if size < 65536 {
        out.push(0x80 | 126);
        out.extend_from_slice(&(size as u16).to_be_bytes());
    } else {
        out.push(0x80 | 127);
        out.extend_from_slice(&(size as u64).to_be_bytes());
    }
    let key = random::<4>();
    out.extend_from_slice(&key);
    let mut body = payload.to_vec();
    mask(&mut body, key);
    out.extend(body);
    out
}

/// One open connection. Build it with [`WebSocket::connect`]. Cheap to share
/// between a sending task and a receiving one: each half has its own lock.
pub struct WebSocket {
    reader: Mutex<Reader>,
    writer: Arc<Mutex<Writer>>,
    pub max_size: usize,
    closed: AtomicBool,
}

impl std::fmt::Debug for WebSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<WebSocket closed={}>", self.closed.load(Ordering::SeqCst))
    }
}

fn split_url(url: &str) -> Result<(bool, String, u16, String)> {
    let (scheme, rest) = url.split_once("://").ok_or_else(|| Error::config(format!("not a websocket URL: '{url}'")))?;
    let secure = match scheme {
        "ws" => false,
        "wss" => true,
        _ => return Err(Error::config(format!("not a websocket URL: '{url}'"))),
    };
    let (authority, path) = match rest.find(['/', '?']) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let path = if path.starts_with('?') { format!("/{path}") } else { path.to_string() };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') || authority.starts_with('[') => match p.parse::<u16>() {
            Ok(p) => (h.trim_matches(['[', ']']).to_string(), p),
            Err(_) => (authority.to_string(), if secure { 443 } else { 80 }),
        },
        _ => (authority.trim_matches(['[', ']']).to_string(), if secure { 443 } else { 80 }),
    };
    Ok((secure, host, port, path))
}

impl WebSocket {
    /// Open a connection: `ws://` or `wss://`, with extra handshake headers
    /// (an `authorization`, say).
    pub async fn connect(url: &str, headers: &[(String, String)], timeout: f64) -> Result<Self> {
        Self::connect_with(url, headers, timeout, 32 * 1024 * 1024).await
    }

    pub async fn connect_with(url: &str, headers: &[(String, String)], timeout: f64, max_size: usize) -> Result<Self> {
        let (secure, host, port, path) = split_url(url)?;
        let limit = Duration::from_secs_f64(timeout.max(0.1));
        let tcp = match tokio::time::timeout(limit, tokio::net::TcpStream::connect((host.as_str(), port))).await {
            Err(_) => return Err(Error::Io(format!("timed out connecting to {host}:{port}"))),
            Ok(Err(e)) => return Err(Error::Io(format!("could not connect to {host}:{port}: {e}"))),
            Ok(Ok(s)) => s,
        };
        let _ = tcp.set_nodelay(true);
        let stream: Box<dyn Stream> = if secure {
            crate::llm_providers::base::install_tls();
            let mut roots = rustls::RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let config = rustls::ClientConfig::builder().with_root_certificates(roots).with_no_client_auth();
            let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
            let name = rustls::pki_types::ServerName::try_from(host.clone()).map_err(|e| Error::Io(format!("{host} is not a valid server name: {e}")))?;
            match tokio::time::timeout(limit, connector.connect(name, tcp)).await {
                Ok(Ok(tls)) => Box::new(tls),
                Ok(Err(e)) => return Err(Error::Io(format!("could not connect to {host}:{port}: {e}"))),
                Err(_) => return Err(Error::Io(format!("timed out connecting to {host}:{port}"))),
            }
        } else {
            Box::new(tcp)
        };
        let (read, mut write) = tokio::io::split(stream);
        let mut reader = BufReader::new(read);

        let key = base64::engine::general_purpose::STANDARD.encode(random::<16>());
        let host_line = if port == 80 || port == 443 { host.clone() } else { format!("{host}:{port}") };
        let mut lines = vec![
            format!("GET {path} HTTP/1.1"),
            format!("Host: {host_line}"),
            "Upgrade: websocket".into(),
            "Connection: Upgrade".into(),
            format!("Sec-WebSocket-Key: {key}"),
            "Sec-WebSocket-Version: 13".into(),
        ];
        let taken = ["host", "upgrade", "connection", "sec-websocket-key", "sec-websocket-version"];
        for (name, value) in headers {
            if !taken.contains(&name.to_lowercase().as_str()) {
                lines.push(format!("{name}: {value}"));
            }
        }
        let request = format!("{}\r\n\r\n", lines.join("\r\n"));
        let failed = |what: &str| Error::Io(format!("the websocket handshake with {host} failed: {what}"));
        write.write_all(request.as_bytes()).await.map_err(|e| failed(&e.to_string()))?;
        let head = tokio::time::timeout(limit, async {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if reader.read(&mut byte).await? == 0 {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "IncompleteReadError"));
                }
                head.push(byte[0]);
                if head.len() > 64 * 1024 {
                    return Err(std::io::Error::other("the response head is too large"));
                }
            }
            Ok(head)
        })
        .await
        .map_err(|_| failed("TimeoutError"))?
        .map_err(|e| failed(&e.to_string()))?;
        let text = String::from_utf8_lossy(&head);
        let mut rows = text.split("\r\n");
        let status = rows.next().unwrap_or("").to_string();
        let answer: std::collections::HashMap<String, String> =
            rows.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string())).collect();
        if !status.contains(" 101") {
            // The body usually says why: a bad key, an unknown model.
            let mut body = vec![0u8; 2048];
            let n = tokio::time::timeout(Duration::from_secs(2), reader.read(&mut body)).await.ok().and_then(|r| r.ok()).unwrap_or(0);
            let said = String::from_utf8_lossy(&body[..n]).trim().chars().take(400).collect::<String>();
            let more = if said.is_empty() { String::new() } else { format!(" — {said}") };
            return Err(Error::Io(format!("{host} refused the websocket: {}{more}", status.trim())));
        }
        let expected = base64::engine::general_purpose::STANDARD.encode(Sha1::digest(format!("{key}{GUID}").as_bytes()));
        if answer.get("sec-websocket-accept") != Some(&expected) {
            return Err(Error::Io(format!("{host} did not complete the websocket handshake")));
        }
        Ok(Self { reader: Mutex::new(reader), writer: Arc::new(Mutex::new(write)), max_size, closed: AtomicBool::new(false) })
    }

    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    async fn write(&self, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
        let mut w = self.writer.lock().await;
        w.write_all(&frame(opcode, payload)).await?;
        w.flush().await
    }

    async fn send(&self, opcode: u8, payload: &[u8]) -> Result<()> {
        if self.closed() {
            return Err(connection_closed(1006, "the connection is closed"));
        }
        if let Err(e) = self.write(opcode, payload).await {
            self.closed.store(true, Ordering::SeqCst);
            return Err(connection_closed(1006, &e.to_string()));
        }
        Ok(())
    }

    pub async fn send_text(&self, message: &str) -> Result<()> {
        self.send(TEXT, message.as_bytes()).await
    }

    pub async fn send_binary(&self, message: &[u8]) -> Result<()> {
        self.send(BINARY, message).await
    }

    async fn read_frame(&self, reader: &mut Reader) -> std::result::Result<(bool, u8, Vec<u8>), Error> {
        let io = |e: std::io::Error| connection_closed(1006, if e.kind() == std::io::ErrorKind::UnexpectedEof { "IncompleteReadError" } else { "OSError" });
        let mut two = [0u8; 2];
        reader.read_exact(&mut two).await.map_err(io)?;
        let (first, second) = (two[0], two[1]);
        let mut size = (second & 0x7F) as u64;
        if size == 126 {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b).await.map_err(io)?;
            size = u16::from_be_bytes(b) as u64;
        } else if size == 127 {
            let mut b = [0u8; 8];
            reader.read_exact(&mut b).await.map_err(io)?;
            size = u64::from_be_bytes(b);
        }
        if size > self.max_size as u64 {
            return Err(connection_closed(1009, &format!("a {size}-byte frame is more than this connection accepts")));
        }
        let key = if second & 0x80 != 0 {
            let mut k = [0u8; 4];
            reader.read_exact(&mut k).await.map_err(io)?;
            Some(k)
        } else {
            None
        };
        let mut payload = vec![0u8; size as usize];
        reader.read_exact(&mut payload).await.map_err(io)?;
        if let Some(k) = key {
            mask(&mut payload, k);
        }
        Ok((first & 0x80 != 0, first & 0x0F, payload))
    }

    /// The next whole message. `Error::ConnectionClosed` when there is none.
    pub async fn recv(&self) -> Result<WsMessage> {
        if self.closed() {
            return Err(connection_closed(1006, "the connection is closed"));
        }
        let mut reader = self.reader.lock().await;
        let (mut kind, mut parts) = (0u8, Vec::new());
        loop {
            let (fin, opcode, payload) = match self.read_frame(&mut reader).await {
                Ok(f) => f,
                Err(e) => {
                    self.closed.store(true, Ordering::SeqCst);
                    return Err(e);
                }
            };
            match opcode {
                PING => {
                    let _ = self.write(PONG, &payload).await;
                    continue;
                }
                PONG => continue,
                CLOSE => {
                    let code = if payload.len() >= 2 { u16::from_be_bytes([payload[0], payload[1]]) } else { 1005 };
                    let reason = String::from_utf8_lossy(payload.get(2..).unwrap_or(&[])).into_owned();
                    self.goodbye(if code == 1005 { 1000 } else { code }).await;
                    return Err(connection_closed(code, &reason));
                }
                CONTINUATION => {}
                other => kind = other,
            }
            parts.extend(payload);
            if parts.len() > self.max_size {
                self.goodbye(1009).await;
                return Err(connection_closed(1009, "the message is more than this connection accepts"));
            }
            if fin {
                return Ok(if kind == TEXT { WsMessage::Text(String::from_utf8_lossy(&parts).into_owned()) } else { WsMessage::Binary(parts) });
            }
        }
    }

    async fn goodbye(&self, code: u16) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = self.write(CLOSE, &code.to_be_bytes()).await;
        let _ = self.writer.lock().await.shutdown().await;
    }

    /// Close politely. Safe to call twice.
    pub async fn close(&self, code: u16) {
        self.goodbye(code).await;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A one-connection echo server, speaking the server side of RFC 6455:
    /// it answers the handshake, pings once, then echoes until told to close.
    pub(crate) async fn echo_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
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
            let accept = base64::engine::general_purpose::STANDARD.encode(Sha1::digest(format!("{key}{GUID}").as_bytes()));
            w.write_all(format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").as_bytes()).await.unwrap();
            w.write_all(&[0x80 | PING, 2, b'h', b'i']).await.unwrap();
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
                mask(&mut payload, key);
                let opcode = two[0] & 0x0F;
                if opcode == CLOSE {
                    w.write_all(&[0x80 | CLOSE, 2, 0x03, 0xE8]).await.unwrap();
                    return;
                }
                if opcode == PONG {
                    continue;
                }
                // Echo it back, split in two fragments, unmasked as a server sends.
                let half = payload.len() / 2;
                let mut out = vec![opcode, half as u8];
                out.extend_from_slice(&payload[..half]);
                out.extend_from_slice(&[0x80 | CONTINUATION, (payload.len() - half) as u8]);
                out.extend_from_slice(&payload[half..]);
                w.write_all(&out).await.unwrap();
            }
        });
        format!("ws://127.0.0.1:{}/realtime?model=x", addr.port())
    }

    #[tokio::test]
    async fn handshake_fragments_pings_and_close() {
        let url = echo_server().await;
        let ws = WebSocket::connect(&url, &[("authorization".into(), "Bearer k".into())], 5.0).await.unwrap();
        ws.send_text("hello there").await.unwrap();
        assert_eq!(ws.recv().await.unwrap(), WsMessage::Text("hello there".into()));
        ws.send_binary(&[1, 2, 3, 4]).await.unwrap();
        assert_eq!(ws.recv().await.unwrap(), WsMessage::Binary(vec![1, 2, 3, 4]));
        ws.close(1000).await;
        assert!(ws.send_text("x").await.is_err());
        assert!(WebSocket::connect("http://x", &[], 1.0).await.unwrap_err().to_string().contains("not a websocket URL"));
        assert_eq!(split_url("wss://api.openai.com/v1/realtime?model=m").unwrap(), (true, "api.openai.com".into(), 443, "/v1/realtime?model=m".into()));
    }
}
