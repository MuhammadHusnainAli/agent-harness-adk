//! AWS Signature Version 4.
//!
//! S3, DynamoDB, S3 Vectors and Bedrock will not take an API key, so a request
//! has to be signed. This is the whole of it — cheaper than the AWS SDK, and
//! checked against the signing vectors AWS publishes.

use std::collections::BTreeMap;

use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

pub const ALGORITHM: &str = "AWS4-HMAC-SHA256";

/// An access key, a secret, and optionally a session token.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub token: Option<String>,
}

impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<AwsCredentials {}…>", self.access_key.chars().take(4).collect::<String>())
    }
}

impl AwsCredentials {
    pub fn new(access_key: impl Into<String>, secret_key: impl Into<String>, token: Option<String>) -> Self {
        Self { access_key: access_key.into(), secret_key: secret_key.into(), token }
    }

    pub fn is_set(&self) -> bool {
        !self.access_key.is_empty() && !self.secret_key.is_empty()
    }

    /// Explicit values, then the environment (`AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`).
    pub fn resolve(access_key: Option<&str>, secret_key: Option<&str>, token: Option<&str>) -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Self {
            access_key: access_key.map(str::to_string).or_else(|| env("AWS_ACCESS_KEY_ID")).unwrap_or_default(),
            secret_key: secret_key.map(str::to_string).or_else(|| env("AWS_SECRET_ACCESS_KEY")).unwrap_or_default(),
            token: token.map(str::to_string).or_else(|| env("AWS_SESSION_TOKEN")),
        }
    }
}

/// The region from `AWS_REGION` / `AWS_DEFAULT_REGION`, or `us-east-1`.
pub fn default_region() -> String {
    std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION")).ok().filter(|r| !r.is_empty()).unwrap_or_else(|| "us-east-1".into())
}

fn hmac(key: &[u8], message: &str) -> Vec<u8> {
    let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("HMAC takes a key of any length");
    mac.update(message.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of `body`, as lowercase hex.
pub fn sha256_hex(body: &[u8]) -> String {
    hex(&Sha256::digest(body))
}

/// The four-step derivation: date, region, service, then the terminator.
pub fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    hmac(&hmac(&hmac(&hmac(format!("AWS4{secret}").as_bytes(), date), region), service), "aws4_request")
}

/// Percent-encode everything but unreserved characters (and `/` when `keep_slash`).
pub fn uri_encode(text: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') || (keep_slash && b == b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Undo `%xx`, so a path that arrives encoded is not encoded twice.
fn unquote(text: &str) -> String {
    crate::knowledge::stores::decode(text)
}

/// Parameters sorted by name, each encoded, joined with `&`.
fn canonical_query(query: &str) -> String {
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (uri_encode(&unquote(k), false), uri_encode(&unquote(v), false))
        })
        .collect();
    pairs.sort();
    pairs.into_iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// `(YYYYMMDDTHHMMSSZ, YYYYMMDD)` for a moment in seconds since the epoch, UTC.
pub fn amz_dates(epoch_seconds: f64) -> (String, String) {
    let secs = epoch_seconds.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let date = format!("{y:04}{m:02}{d:02}");
    (format!("{date}T{:02}{:02}{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60), date)
}

/// What to sign.
#[derive(Debug, Clone)]
pub struct SignRequest<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub region: &'a str,
    pub service: &'a str,
    pub body: &'a [u8],
    /// Headers to sign along with `host` and `x-amz-date`.
    pub headers: &'a [(&'a str, &'a str)],
    /// Seconds since the epoch; now when `None`.
    pub now: Option<f64>,
    /// Add `x-amz-content-sha256` (required by S3, accepted everywhere else).
    pub content_sha_header: bool,
}

/// The headers a signed request needs, including `authorization`, lowercased.
pub fn sign(req: &SignRequest<'_>, credentials: &AwsCredentials) -> Vec<(String, String)> {
    let rest = req.url.split_once("://").map_or(req.url, |(_, r)| r);
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let path = uri_encode(&unquote(path), true);
    let query = canonical_query(query);
    let (amz_date, date) = amz_dates(req.now.unwrap_or_else(crate::types::now));

    let mut signed: BTreeMap<String, String> = BTreeMap::new();
    signed.insert("host".into(), host.into());
    signed.insert("x-amz-date".into(), amz_date.clone());
    for (k, v) in req.headers {
        signed.insert(k.to_ascii_lowercase(), v.trim().to_string());
    }
    if let Some(t) = credentials.token.as_ref().filter(|t| !t.is_empty()) {
        signed.insert("x-amz-security-token".into(), t.clone());
    }
    let payload_hash = sha256_hex(req.body);
    if req.content_sha_header {
        signed.insert("x-amz-content-sha256".into(), payload_hash.clone());
    }
    let canonical_headers: String = signed.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let header_list = signed.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_request = [req.method.to_ascii_uppercase(), path, query, canonical_headers, header_list.clone(), payload_hash].join("\n");
    let scope = format!("{date}/{}/{}/aws4_request", req.region, req.service);
    let to_sign = [ALGORITHM.to_string(), amz_date, scope.clone(), sha256_hex(canonical_request.as_bytes())].join("\n");
    let signature = hex(&hmac(&signing_key(&credentials.secret_key, &date, req.region, req.service), &to_sign));
    let mut out: Vec<(String, String)> = signed.into_iter().collect();
    out.push((
        "authorization".into(),
        format!("{ALGORITHM} Credential={}/{scope}, SignedHeaders={header_list}, Signature={signature}", credentials.access_key),
    ));
    out
}

/// A [`Signer`](crate::knowledge::base::Signer) for an [`HttpClient`](crate::knowledge::base::HttpClient):
/// every request it sends is signed for `service` in `region`.
pub fn signer(credentials: AwsCredentials, region: String, service: &'static str, content_sha_header: bool) -> crate::knowledge::base::Signer {
    std::sync::Arc::new(move |method: &str, url: &str, body: &[u8], headers: &mut reqwest::header::HeaderMap| {
        if !credentials.is_set() {
            return Err(crate::errors::Error::config(format!(
                "{service}: no AWS credentials — set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or pass them in"
            )));
        }
        let extra: Vec<(String, String)> = headers
            .iter()
            .filter(|(k, _)| k.as_str() == "content-type" || k.as_str().starts_with("x-amz-"))
            .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_string(), v.to_string())))
            .collect();
        let pairs: Vec<(&str, &str)> = extra.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let req = SignRequest { method, url, region: &region, service, body, headers: &pairs, now: None, content_sha_header };
        for (k, v) in sign(&req, &credentials) {
            if k == "host" {
                continue;
            }
            if let (Ok(k), Ok(v)) = (reqwest::header::HeaderName::from_bytes(k.as_bytes()), reqwest::header::HeaderValue::from_str(&v)) {
                headers.insert(k, v);
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AWS's published `get-vanilla` vector, and the documented key derivation.
    #[test]
    fn matches_the_aws_test_suite() {
        let creds = AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", None);
        // 2015-08-30T12:36:00Z
        let req = SignRequest {
            method: "GET",
            url: "https://example.amazonaws.com/",
            region: "us-east-1",
            service: "service",
            body: b"",
            headers: &[],
            now: Some(1_440_938_160.0),
            content_sha_header: false,
        };
        let out = sign(&req, &creds);
        let auth = &out.iter().find(|(k, _)| k == "authorization").unwrap().1;
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
        assert_eq!(amz_dates(1_440_938_160.0).0, "20150830T123600Z");
        assert_eq!(
            hex(&signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "20150830", "us-east-1", "iam")),
            "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"
        );
        assert_eq!(canonical_query("b=2&a=1&a%20b=x y"), "a=1&a%20b=x%20y&b=2");
        assert_eq!(uri_encode("a b/c~", true), "a%20b/c~");
    }
}
