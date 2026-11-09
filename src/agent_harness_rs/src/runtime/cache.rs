//! Result cache: an identical task with identical input is served, not re-run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use blake2::Digest;
use serde_json::{Value, json};

use crate::pyjson::{Dumps, dumps_with};
use crate::types::now;

/// The key for these parts: BLAKE2b-128 of their sorted JSON — Python's `cache_key`.
pub fn cache_key(parts: &[Value]) -> String {
    let blob = dumps_with(&Value::Array(parts.to_vec()), Dumps { sort_keys: true, ..Dumps::default() });
    blake2::Blake2b::<blake2::digest::consts::U16>::digest(blob.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

struct Inner {
    ttl: f64,
    max_entries: usize,
    enabled: bool,
    path: Option<PathBuf>,
    data: HashMap<String, (f64, Value)>,
    hits: u64,
    misses: u64,
}

/// TTL cache with an optional on-disk tier. Never pay twice for the same call.
/// Cheap to clone; clones are the same cache.
#[derive(Clone)]
pub struct ResultCache {
    inner: Arc<Mutex<Inner>>,
}

impl Default for ResultCache {
    fn default() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { ttl: 3600.0, max_entries: 2048, enabled: true, path: None, data: HashMap::new(), hits: 0, misses: 0 })) }
    }
}

impl std::fmt::Debug for ResultCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<ResultCache {}>", self.stats())
    }
}

impl ResultCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Also kept as one JSON file per key in this folder (Python's layout).
    pub fn at(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let _ = std::fs::create_dir_all(&path);
        let c = Self::default();
        c.lock().path = Some(path);
        c
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn ttl(self, seconds: f64) -> Self {
        self.lock().ttl = seconds;
        self
    }

    pub fn max_entries(self, n: usize) -> Self {
        self.lock().max_entries = n.max(1);
        self
    }

    pub fn enabled(self, on: bool) -> Self {
        self.lock().enabled = on;
        self
    }

    pub fn key(&self, parts: &[Value]) -> String {
        cache_key(parts)
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        let mut i = self.lock();
        if !i.enabled {
            return None;
        }
        let entry = i.data.get(key).cloned().or_else(|| {
            let file = i.path.as_ref()?.join(format!("{key}.json"));
            let blob: Value = serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()?;
            Some((blob["expires"].as_f64().unwrap_or(0.0), blob.get("value").cloned().unwrap_or(Value::Null)))
        });
        match entry {
            Some((expires, _)) if expires != 0.0 && expires < now() => {
                i.data.remove(key);
                i.misses += 1;
                None
            }
            Some((_, value)) => {
                i.hits += 1;
                Some(value)
            }
            None => {
                i.misses += 1;
                None
            }
        }
    }

    pub fn set(&self, key: &str, value: Value, ttl: Option<f64>) {
        let mut i = self.lock();
        if !i.enabled {
            return;
        }
        let expires = now() + ttl.unwrap_or(i.ttl);
        if i.data.len() >= i.max_entries
            && let Some(oldest) = i.data.iter().min_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap_or(std::cmp::Ordering::Equal)).map(|(k, _)| k.clone())
        {
            i.data.remove(&oldest);
        }
        if let Some(dir) = &i.path {
            // A cache that cannot write is still a working cache.
            let _ = std::fs::write(dir.join(format!("{key}.json")), crate::pyjson::dumps(&json!({"expires": expires, "value": value})));
        }
        i.data.insert(key.into(), (expires, value));
    }

    pub fn clear(&self) {
        let mut i = self.lock();
        i.data.clear();
        if let Some(dir) = &i.path
            && let Ok(entries) = std::fs::read_dir(dir)
        {
            for e in entries.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "json")) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    pub fn stats(&self) -> Value {
        let i = self.lock();
        let total = i.hits + i.misses;
        let rate = if total == 0 { 0.0 } else { (i.hits as f64 / total as f64 * 1000.0).round() / 1000.0 };
        json!({"hits": i.hits, "misses": i.misses, "entries": i.data.len(), "hit_rate": rate})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_misses_expiry_and_python_keys() {
        // cache_key("search", {"q": "x", "a": 1})
        assert_eq!(cache_key(&[json!("search"), json!({"q": "x", "a": 1})]), PYTHON_KEY);
        let c = ResultCache::new();
        assert!(c.get("k").is_none());
        c.set("k", json!([1, 2]), None);
        assert_eq!(c.get("k"), Some(json!([1, 2])));
        c.set("old", json!(1), Some(-1.0));
        assert!(c.get("old").is_none());
        assert_eq!(c.stats()["hits"], 1);
        let dir = std::env::temp_dir().join(format!("ah_cache_{}", uuid::Uuid::new_v4().simple()));
        ResultCache::at(&dir).set("disk", json!("v"), None);
        assert_eq!(ResultCache::at(&dir).get("disk"), Some(json!("v")));
        let _ = std::fs::remove_dir_all(dir);
    }

    const PYTHON_KEY: &str = "59609c7691c66033b24a46c0fee119dc";
}
