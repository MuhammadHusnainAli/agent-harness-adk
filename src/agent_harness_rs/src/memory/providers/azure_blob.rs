//! Azure Blob Storage memory.
//!
//! Spoken to over the Blob REST API — no SDK. The keys are the Python
//! `AzureBlobMemory`'s (see [`objects`](super::objects)), so one container
//! serves both.
//!
//! ```no_run
//! # use agent_harness::memory::providers::azure_blob::AzureBlobMemory;
//! let store = AzureBlobMemory::from_url("azure://agents/memory");   // AZURE_STORAGE_CONNECTION_STRING
//! let store = AzureBlobMemory::from_url("azure://agents?account_url=https://acct.blob.core.windows.net");
//! ```
//!
//! **Where the account is**, as Python takes it: a `connection_string`
//! (`AccountName`/`AccountKey`, `BlobEndpoint`, `SharedAccessSignature`,
//! `UseDevelopmentStorage=true` for Azurite), or an `account_url` (or just an
//! `account` name) with a credential. On the URL: `connection_string`,
//! `account_url`, `account`, `account_key`, `sas_token`, `token`, `prefix`,
//! `concurrency`, `conditional`; otherwise `AZURE_STORAGE_CONNECTION_STRING`,
//! `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`,
//! `AZURE_STORAGE_SAS_TOKEN`. `abfs://container@account.dfs.core.windows.net/path`
//! names the account too.
//!
//! **Credentials**: an account key signs each request (Shared Key); a SAS
//! token rides on the URL; otherwise a bearer token for
//! `https://storage.azure.com/` — given as `token`, or from what
//! `DefaultAzureCredential` would try first: a client secret
//! (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`), workload
//! identity (`AZURE_FEDERATED_TOKEN_FILE`), then managed identity
//! (`IDENTITY_ENDPOINT` on App Service, else the instance metadata service).
//! The Azure CLI, PowerShell and developer-tool credentials are not tried.
//!
//! Python deletes up to 256 blobs per batch request; the batch API is a
//! multipart protocol, so here each blob is deleted on its own,
//! `concurrency` at a time. A blob already gone is not an error, as in Python.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use hmac::{Hmac, KeyInit, Mac};
use reqwest::Method;
use reqwest::header::HeaderMap;
use sha2::Sha256;

use crate::errors::{Error, Result};
use crate::knowledge::base::{HttpClient, Signer};
use crate::memory::providers::objects::{
    Fetched, ObjectBackend, ObjectStoreMemory, PutIf, decode, encode, env, option, send_raw, split_url, xml_value, xml_values,
};
use crate::types::now;

/// Memory in an Azure blob container: [`ObjectStoreMemory`] over an [`AzureContainer`].
pub type AzureBlobMemory = ObjectStoreMemory<AzureContainer>;

const VERSION: &str = "2021-08-06";
/// The well-known Azurite account.
const DEV_ACCOUNT: &str = "devstoreaccount1";
const DEV_KEY: &str = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==";

/// How requests to the account are authorised.
#[derive(Clone)]
pub enum AzureAuth {
    /// Shared Key: each request signed with the account key.
    SharedKey { account: String, key: Vec<u8> },
    /// A SAS token, appended to every URL.
    Sas(String),
    /// A bearer token you already have.
    Token(String),
    /// Entra ID client credentials (a secret, or a federated token file).
    ClientCredentials { authority: String, tenant: String, client_id: String, secret: Option<String>, assertion_file: Option<String> },
    /// The managed identity of the machine this runs on.
    ManagedIdentity { client_id: Option<String> },
}

impl std::fmt::Debug for AzureAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::SharedKey { .. } => "SharedKey",
            Self::Sas(_) => "Sas",
            Self::Token(_) => "Token",
            Self::ClientCredentials { .. } => "ClientCredentials",
            Self::ManagedIdentity { .. } => "ManagedIdentity",
        })
    }
}

impl AzureAuth {
    /// What `DefaultAzureCredential` tries first, from the environment.
    pub fn from_environment() -> Self {
        let authority = env("AZURE_AUTHORITY_HOST").unwrap_or_else(|| "https://login.microsoftonline.com".into());
        if let (Some(tenant), Some(client_id)) = (env("AZURE_TENANT_ID"), env("AZURE_CLIENT_ID")) {
            if let Some(secret) = env("AZURE_CLIENT_SECRET") {
                return Self::ClientCredentials { authority, tenant, client_id, secret: Some(secret), assertion_file: None };
            }
            if let Some(file) = env("AZURE_FEDERATED_TOKEN_FILE") {
                return Self::ClientCredentials { authority, tenant, client_id, secret: None, assertion_file: Some(file) };
            }
        }
        Self::ManagedIdentity { client_id: env("AZURE_CLIENT_ID") }
    }
}

/// `key=value;…` taken apart.
fn connection_parts(text: &str) -> Vec<(String, String)> {
    text.split(';').filter_map(|p| p.split_once('=')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect()
}

/// The Shared Key string to sign for a request (Blob service, version 2015+).
pub fn string_to_sign(account: &str, method: &str, url: &str, content_length: usize, headers: &HeaderMap) -> String {
    let h = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let mut ms: Vec<(String, String)> = headers
        .iter()
        .filter(|(k, _)| k.as_str().starts_with("x-ms-"))
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").trim().to_string()))
        .collect();
    ms.sort();
    let canonical_headers: String = ms.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
    let path = rest.find('/').map_or("/", |i| &rest[i..]);
    let mut params: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (decode(k).to_lowercase(), decode(v))
        })
        .collect();
    params.sort();
    let mut resource = format!("/{account}{path}");
    let mut i = 0;
    while i < params.len() {
        let name = params[i].0.clone();
        let mut values = Vec::new();
        while i < params.len() && params[i].0 == name {
            values.push(params[i].1.clone());
            i += 1;
        }
        resource.push_str(&format!("\n{name}:{}", values.join(",")));
    }
    let length = if content_length == 0 { String::new() } else { content_length.to_string() };
    [
        method.to_ascii_uppercase(),
        h("content-encoding"),
        h("content-language"),
        length,
        h("content-md5"),
        h("content-type"),
        String::new(), // Date: x-ms-date is sent instead
        h("if-modified-since"),
        h("if-match"),
        h("if-none-match"),
        h("if-unmodified-since"),
        h("range"),
    ]
    .join("\n")
        + "\n"
        + &canonical_headers
        + &resource
}

/// A signer that stamps `x-ms-date` and signs with the account key.
pub fn shared_key_signer(account: String, key: Vec<u8>) -> Signer {
    Arc::new(move |method: &str, url: &str, body: &[u8], headers: &mut HeaderMap| {
        let date = httpdate::fmt_http_date(std::time::SystemTime::now());
        headers.insert("x-ms-date", date.parse().map_err(|_| Error::config("azure: bad date"))?);
        let to_sign = string_to_sign(&account, method, url, body.len(), headers);
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&key).map_err(|_| Error::config("azure: bad account key"))?;
        mac.update(to_sign.as_bytes());
        let signature = base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
        headers.insert("authorization", format!("SharedKey {account}:{signature}").parse().map_err(|_| Error::config("azure: bad account name"))?);
        Ok(())
    })
}

/// One blob container, and how to reach and authorise against its account.
#[derive(Debug, Clone)]
pub struct AzureContainer {
    pub container: String,
    /// `https://{account}.blob.core.windows.net`, or Azurite's
    /// `http://127.0.0.1:10000/devstoreaccount1`.
    pub account_url: String,
    auth: AzureAuth,
    cached: Arc<tokio::sync::Mutex<Option<(String, f64)>>>,
    http: HttpClient,
    /// Deletes in flight at once.
    pub concurrency: usize,
}

impl AzureContainer {
    pub fn new(container: &str, account_url: &str, auth: AzureAuth) -> Self {
        let mut http = HttpClient::new("azure-blob", 60.0, 3).header("x-ms-version", VERSION);
        match &auth {
            AzureAuth::SharedKey { account, key } => http.signer = Some(shared_key_signer(account.clone(), key.clone())),
            AzureAuth::Sas(s) => http = http.secret(s.clone()),
            AzureAuth::Token(t) => http = http.secret(t.clone()),
            _ => {}
        }
        Self { container: container.into(), account_url: account_url.trim_end_matches('/').to_string(), auth, cached: Arc::default(), http, concurrency: 16 }
    }

    /// From a connection string, as `BlobServiceClient.from_connection_string`.
    pub fn from_connection_string(container: &str, text: &str) -> Result<Self> {
        let parts = connection_parts(text);
        let get = |k: &str| option(&parts, k).map(str::to_string);
        if get("UseDevelopmentStorage").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            let url = get("DevelopmentStorageProxyUri").map_or_else(|| format!("http://127.0.0.1:10000/{DEV_ACCOUNT}"), |p| format!("{p}/{DEV_ACCOUNT}"));
            return Self::keyed(container, &url, DEV_ACCOUNT, DEV_KEY);
        }
        let account = get("AccountName").unwrap_or_default();
        let url = get("BlobEndpoint").unwrap_or_else(|| {
            let protocol = get("DefaultEndpointsProtocol").unwrap_or_else(|| "https".into());
            let suffix = get("EndpointSuffix").unwrap_or_else(|| "core.windows.net".into());
            format!("{protocol}://{account}.blob.{suffix}")
        });
        if let Some(key) = get("AccountKey") {
            return Self::keyed(container, &url, &account, &key);
        }
        if let Some(sas) = get("SharedAccessSignature") {
            return Ok(Self::new(container, &url, AzureAuth::Sas(sas.trim_start_matches('?').to_string())));
        }
        Err(Error::config("AzureBlobMemory: the connection string has neither an AccountKey nor a SharedAccessSignature"))
    }

    fn keyed(container: &str, url: &str, account: &str, key: &str) -> Result<Self> {
        let key = base64::engine::general_purpose::STANDARD
            .decode(key.trim())
            .map_err(|e| Error::config(format!("AzureBlobMemory: the account key is not base64: {e}")))?;
        Ok(Self::new(container, url, AzureAuth::SharedKey { account: account.into(), key }))
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    async fn token(&self) -> Result<Option<String>> {
        let (url, body, extra): (String, Option<String>, Vec<(&str, String)>) = match &self.auth {
            AzureAuth::SharedKey { .. } | AzureAuth::Sas(_) => return Ok(None),
            AzureAuth::Token(t) => return Ok(Some(t.clone())),
            AzureAuth::ClientCredentials { authority, tenant, client_id, secret, assertion_file } => {
                let mut form = vec![
                    ("grant_type", "client_credentials".to_string()),
                    ("client_id", client_id.clone()),
                    ("scope", "https://storage.azure.com/.default".to_string()),
                ];
                if let Some(s) = secret {
                    form.push(("client_secret", s.clone()));
                }
                if let Some(file) = assertion_file {
                    let assertion = std::fs::read_to_string(file).map_err(|e| Error::config(format!("azure: {file}: {e}")))?;
                    form.push(("client_assertion_type", "urn:ietf:params:oauth:client-assertion-type:jwt-bearer".into()));
                    form.push(("client_assertion", assertion.trim().to_string()));
                }
                let body = form.iter().map(|(k, v)| format!("{k}={}", encode(v, false))).collect::<Vec<_>>().join("&");
                (format!("{}/{tenant}/oauth2/v2.0/token", authority.trim_end_matches('/')), Some(body), vec![])
            }
            AzureAuth::ManagedIdentity { client_id } => {
                let resource = encode("https://storage.azure.com/", false);
                let id = client_id.as_ref().map(|c| format!("&client_id={}", encode(c, false))).unwrap_or_default();
                match (env("IDENTITY_ENDPOINT"), env("IDENTITY_HEADER")) {
                    (Some(e), Some(h)) => (format!("{e}?api-version=2019-08-01&resource={resource}{id}"), None, vec![("x-identity-header", h)]),
                    _ => (
                        format!("http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource={resource}{id}"),
                        None,
                        vec![("metadata", "true".to_string())],
                    ),
                }
            }
        };
        let mut cached = self.cached.lock().await;
        if let Some((token, until)) = cached.as_ref()
            && *until > now()
        {
            return Ok(Some(token.clone()));
        }
        let auth_http = HttpClient::new("azure-auth", 30.0, 2);
        let reply = match body {
            Some(b) => auth_http.send(Method::POST, &url, Some(b.into_bytes()), Some("application/x-www-form-urlencoded"), &extra, &[]).await,
            None => auth_http.send(Method::GET, &url, None, None, &extra, &[]).await,
        }
        .map_err(|e| Error::config(format!("AzureBlobMemory: no access token ({e}); give a connection string, an account key or a SAS token")))?;
        let token = reply.body["access_token"].as_str().unwrap_or("").to_string();
        if token.is_empty() {
            return Err(Error::config("AzureBlobMemory: the token endpoint returned no access_token"));
        }
        let number = |v: &serde_json::Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
        let lifetime = number(&reply.body["expires_in"]).unwrap_or(3600.0);
        *cached = Some((token.clone(), now() + lifetime - 60.0));
        Ok(Some(token))
    }

    async fn headers(&self, extra: &[(&'static str, String)]) -> Result<Vec<(&'static str, String)>> {
        let mut h = extra.to_vec();
        if let Some(t) = self.token().await? {
            h.push(("authorization", format!("Bearer {t}")));
        }
        Ok(h)
    }

    /// `url` with the SAS token on it, if there is one.
    fn signed(&self, url: String) -> String {
        match &self.auth {
            AzureAuth::Sas(sas) => format!("{url}{}{sas}", if url.contains('?') { '&' } else { '?' }),
            _ => url,
        }
    }

    fn blob_url(&self, key: &str) -> String {
        self.signed(format!("{}/{}/{}", self.account_url, encode(&self.container, false), encode(key, true)))
    }
}

#[async_trait]
impl ObjectBackend for AzureContainer {
    async fn put(&self, key: &str, body: Vec<u8>, content_type: &str, condition: &PutIf) -> Result<bool> {
        let mut extra = vec![("x-ms-blob-type", "BlockBlob".to_string()), ("content-type", content_type.to_string())];
        match condition {
            PutIf::Always => {}
            PutIf::Absent => extra.push(("if-none-match", "*".into())),
            PutIf::Matches(tag) => extra.push(("if-match", tag.clone())),
        }
        let headers = self.headers(&extra).await?;
        let ok: &[u16] = if *condition == PutIf::Always { &[] } else { &[409, 412] };
        // 412: the ETag moved on. 409: the blob exists (If-None-Match: *).
        Ok(!matches!(send_raw(&self.http, Method::PUT, &self.blob_url(key), body, &headers, ok).await?.status, 409 | 412))
    }

    async fn get(&self, key: &str) -> Result<Option<Fetched>> {
        let headers = self.headers(&[]).await?;
        let reply = send_raw(&self.http, Method::GET, &self.blob_url(key), vec![], &headers, &[404]).await?;
        if reply.status == 404 {
            return Ok(None);
        }
        let tag = reply.header("etag");
        Ok(Some(Fetched { body: reply.body, tag }))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let mut marker: Option<String> = None;
        loop {
            let mut url = format!("{}/{}?restype=container&comp=list&prefix={}", self.account_url, encode(&self.container, false), encode(prefix, false));
            if let Some(m) = &marker {
                url.push_str(&format!("&marker={}", encode(m, false)));
            }
            let headers = self.headers(&[]).await?;
            let xml = send_raw(&self.http, Method::GET, &self.signed(url), vec![], &headers, &[]).await?.text();
            names.extend(xml_values(&xml, "Blob").iter().filter_map(|b| xml_value(b, "Name")));
            marker = xml_value(&xml, "NextMarker").filter(|m| !m.is_empty());
            if marker.is_none() {
                return Ok(names);
            }
        }
    }

    async fn delete(&self, keys: &[String]) -> Result<()> {
        let headers = self.headers(&[]).await?;
        let results: Vec<Result<()>> = futures::stream::iter(keys.iter().map(|k| self.blob_url(k)).collect::<Vec<String>>())
            .map(|url: String| {
                let headers = &headers;
                async move { send_raw(&self.http, Method::DELETE, &url, vec![], headers, &[404]).await.map(|_| ()) }
            })
            .buffer_unordered(self.concurrency.max(1))
            .collect()
            .await;
        results.into_iter().collect()
    }

    fn describe(&self) -> String {
        format!("azure://{}", self.container)
    }
}

impl AzureBlobMemory {
    /// `azure://container/prefix` (or `blob://`, `abfs://`). The prefix
    /// defaults to `agent-memory`. See the module docs for where the account
    /// and its credentials come from.
    pub fn from_url(url: &str) -> Result<Self> {
        let parts = split_url(url);
        let o = &parts.query;
        // abfs://container@account.dfs.core.windows.net/path
        let (container, host_account) = match parts.user.as_deref() {
            Some(c) if parts.host.contains('.') => (c.to_string(), parts.host.split('.').next().map(str::to_string)),
            _ => (parts.host.clone(), None),
        };
        if container.is_empty() {
            return Err(Error::config(format!("AzureBlobMemory needs a container: azure://container/prefix (got '{url}')")));
        }
        let given = |name: &str, var: &str| option(o, name).map(str::to_string).or_else(|| env(var));
        let mut store = if let Some(cs) = given("connection_string", "AZURE_STORAGE_CONNECTION_STRING") {
            AzureContainer::from_connection_string(&container, &cs)?
        } else {
            let account = option(o, "account").map(str::to_string).or(host_account).or_else(|| env("AZURE_STORAGE_ACCOUNT"));
            let Some(account_url) = option(o, "account_url")
                .map(str::to_string)
                .or_else(|| env("AZURE_STORAGE_ACCOUNT_URL"))
                .or_else(|| account.as_ref().map(|a| format!("https://{a}.blob.core.windows.net")))
            else {
                return Err(Error::config("AzureBlobMemory needs a connection_string or an account_url (or AZURE_STORAGE_CONNECTION_STRING)"));
            };
            // The account name a key signs for: given, or the first label of the host.
            let account = account.unwrap_or_else(|| {
                let host = account_url.split_once("://").map_or(account_url.as_str(), |(_, r)| r);
                host.split(['.', '/', ':']).next().unwrap_or("").to_string()
            });
            if let Some(key) = given("account_key", "AZURE_STORAGE_KEY").or_else(|| env("AZURE_STORAGE_ACCESS_KEY")) {
                AzureContainer::keyed(&container, &account_url, &account, &key)?
            } else if let Some(sas) = given("sas_token", "AZURE_STORAGE_SAS_TOKEN") {
                AzureContainer::new(&container, &account_url, AzureAuth::Sas(sas.trim_start_matches('?').to_string()))
            } else if let Some(token) = option(o, "token") {
                AzureContainer::new(&container, &account_url, AzureAuth::Token(token.into()))
            } else {
                AzureContainer::new(&container, &account_url, AzureAuth::from_environment())
            }
        };
        if let Some(n) = option(o, "concurrency").and_then(|n| n.parse().ok()) {
            store.concurrency = n;
        }
        let prefix = parts.path.trim_matches('/');
        let prefix = if prefix.is_empty() { option(o, "prefix").unwrap_or("agent-memory") } else { prefix };
        Ok(AzureBlobMemory::new(store, prefix).configured(o))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use axum::body::Bytes;
    use axum::http::{HeaderMap as AxHeaders, Method as AxMethod, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use serde_json::Map;

    use super::*;
    use crate::memory::base::MemoryStore;
    use crate::memory::providers::objects::testing::{exercise_memory, exercise_sessions, query_of, serve};
    use crate::memory::providers::objects::xml_escape;

    type Objects = Arc<Mutex<(BTreeMap<String, (Vec<u8>, String)>, u64)>>;

    /// Enough of Azurite: Shared Key checked for real, ETags with If-Match /
    /// If-None-Match, paged listings with markers, deletes.
    async fn fake_blob(state: Objects, method: AxMethod, uri: Uri, headers: AxHeaders, body: Bytes) -> Response {
        let mut check = HeaderMap::new();
        for (k, v) in &headers {
            if let (Ok(k), Ok(v)) = (reqwest::header::HeaderName::from_bytes(k.as_str().as_bytes()), reqwest::header::HeaderValue::from_bytes(v.as_bytes())) {
                check.insert(k, v);
            }
        }
        let given = check.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let url = format!("http://h{}", uri);
        let to_sign = string_to_sign(DEV_ACCOUNT, method.as_str(), &url, body.len(), &check);
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&base64::engine::general_purpose::STANDARD.decode(DEV_KEY).unwrap()).unwrap();
        mac.update(to_sign.as_bytes());
        let want = format!("SharedKey {DEV_ACCOUNT}:{}", base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes()));
        if given != want || !check.contains_key("x-ms-date") || check.get("x-ms-version").is_none() {
            return (StatusCode::FORBIDDEN, "AuthenticationFailed").into_response();
        }
        let path = decode(uri.path());
        let Some(rest) = path.strip_prefix(&format!("/{DEV_ACCOUNT}/chats-container")) else { return StatusCode::NOT_FOUND.into_response() };
        let key = rest.trim_start_matches('/').to_string();
        let q = query_of(&uri);
        let mut s = state.lock().unwrap();
        match (method, key.is_empty()) {
            (AxMethod::GET, true) => {
                let prefix = option(&q, "prefix").unwrap_or("").to_string();
                let start: usize = option(&q, "marker").and_then(|m| m.strip_prefix("m/")).and_then(|m| m.parse().ok()).unwrap_or(0);
                let keys: Vec<&String> = s.0.keys().filter(|k| k.starts_with(&prefix)).collect();
                let blobs: String =
                    keys.iter().skip(start).take(2).map(|k| format!("<Blob><Name>{}</Name><Properties></Properties></Blob>", xml_escape(k))).collect();
                let next = if start + 2 < keys.len() { format!("m/{}", start + 2) } else { String::new() };
                format!("<EnumerationResults ContainerName=\"c\"><Blobs>{blobs}</Blobs><NextMarker>{next}</NextMarker></EnumerationResults>").into_response()
            }
            (AxMethod::PUT, false) => {
                if check.get("x-ms-blob-type").is_none() {
                    return StatusCode::BAD_REQUEST.into_response();
                }
                let held = s.0.get(&key).map(|(_, t)| t.clone());
                let header = |n: &str| check.get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
                if header("if-none-match").as_deref() == Some("*") && held.is_some() {
                    return StatusCode::CONFLICT.into_response();
                }
                if let Some(want) = header("if-match")
                    && held.as_deref() != Some(want.as_str())
                {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
                s.1 += 1;
                let tag = format!("\"0x{}\"", s.1);
                s.0.insert(key, (body.to_vec(), tag.clone()));
                (StatusCode::CREATED, [("etag", tag)], "").into_response()
            }
            (AxMethod::GET, false) => match s.0.get(&key) {
                Some((b, t)) => ([("etag", t.clone())], b.clone()).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            (AxMethod::DELETE, false) => match s.0.remove(&key) {
                Some(_) => StatusCode::ACCEPTED.into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
        }
    }

    async fn store(prefix: &str) -> (AzureBlobMemory, Objects) {
        let state: Objects = Arc::default();
        let shared = state.clone();
        let router = axum::Router::new().fallback(move |m: AxMethod, u: Uri, h: AxHeaders, b: Bytes| fake_blob(shared.clone(), m, u, h, b));
        let base = serve(router).await;
        let connection = format!("DefaultEndpointsProtocol=http;AccountName={DEV_ACCOUNT};AccountKey={DEV_KEY};BlobEndpoint={base}/{DEV_ACCOUNT};");
        let url = format!("azure://chats-container/{prefix}?connection_string={}", encode(&connection, false));
        (AzureBlobMemory::from_url(&url).unwrap(), state)
    }

    #[tokio::test]
    async fn memory_over_the_blob_api_is_scoped_by_trace() {
        let (store, state) = store("mem").await;
        exercise_memory(&store).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn sessions_over_the_blob_api_hold_the_contract() {
        let (store, state) = store("chats").await;
        exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[test]
    fn the_string_to_sign_is_laid_out_as_azure_documents() {
        let mut h = HeaderMap::new();
        h.insert("x-ms-version", VERSION.parse().unwrap());
        h.insert("x-ms-date", "Fri, 26 Jun 2015 23:39:12 GMT".parse().unwrap());
        h.insert("content-type", "text/plain".parse().unwrap());
        let s = string_to_sign("acct", "put", "https://acct.blob.core.windows.net/c/a%20b?comp=list&restype=container", 5, &h);
        assert_eq!(
            s,
            "PUT\n\n\n5\n\ntext/plain\n\n\n\n\n\n\nx-ms-date:Fri, 26 Jun 2015 23:39:12 GMT\nx-ms-version:2021-08-06\n/acct/c/a%20b\ncomp:list\nrestype:container"
        );
    }

    #[test]
    fn accounts_come_from_where_python_takes_them() {
        let dev = AzureContainer::from_connection_string("c", "UseDevelopmentStorage=true").unwrap();
        assert_eq!(dev.account_url, "http://127.0.0.1:10000/devstoreaccount1");
        let sas = AzureContainer::from_connection_string("c", "BlobEndpoint=https://a.blob.core.windows.net;SharedAccessSignature=sv=1&sig=x").unwrap();
        assert_eq!(sas.blob_url("k"), "https://a.blob.core.windows.net/c/k?sv=1&sig=x");
        let abfs = AzureBlobMemory::from_url("abfs://box@acct.dfs.core.windows.net/mem?sas_token=sv%3D1").unwrap();
        assert_eq!(
            (abfs.objects().container.as_str(), abfs.objects().account_url.as_str(), abfs.prefix.as_str()),
            ("box", "https://acct.blob.core.windows.net", "mem")
        );
    }

    /// `AH_LIVE_AZURE_CONNECTION` (and `AH_LIVE_AZURE_CONTAINER`, default
    /// `ah-sessions-test`, which must exist) to run against Azure or Azurite.
    #[tokio::test]
    async fn a_real_account_holds_the_contract() {
        let Some(connection) = env("AH_LIVE_AZURE_CONNECTION") else { return };
        let container = env("AH_LIVE_AZURE_CONTAINER").unwrap_or_else(|| "ah-sessions-test".into());
        let url = format!("azure://{container}/ah-rust-test-{}?connection_string={}", crate::types::new_id(""), encode(&connection, false));
        let store = AzureBlobMemory::from_url(&url).unwrap();
        exercise_memory(&store).await;
        exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
    }
}
