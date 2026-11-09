//! Stop control: abort a run and drain the sub-agents. A human is always in charge.
//!
//! ```
//! # let harness = agent_harness::Harness::new();
//! harness.control.stop("the customer withdrew the request", "drain", "human");
//! ```
//!
//! The loop checks the controller before every step and every tool call, so a
//! stop takes effect at the next safe boundary rather than tearing a run in
//! half. Work already in flight is allowed to finish and hand back what it has
//! (`drain`), or cancelled outright (`abort`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio::task::AbortHandle;

use crate::errors::{Error, Result};
use crate::types::now;

/// Why a run was stopped, by whom, and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StopState {
    pub stopped: bool,
    /// `drain` or `abort`.
    pub mode: String,
    pub reason: String,
    pub requested_by: String,
    pub ts: f64,
}

impl Default for StopState {
    fn default() -> Self {
        Self { stopped: false, mode: "drain".into(), reason: String::new(), requested_by: "human".into(), ts: 0.0 }
    }
}

type Listener = Arc<dyn Fn(&StopState) + Send + Sync>;

struct Inner {
    state: StopState,
    inflight: HashMap<String, AbortHandle>,
    agents: HashMap<String, String>,
    listeners: Vec<Listener>,
}

/// One switch, shared by every agent in a run. Cheap to clone; clones are the
/// same switch.
///
/// `drain` lets in-flight work finish and hand back partial results; `abort`
/// cancels the tasks it is tracking. Either way nothing *new* starts.
#[derive(Clone)]
pub struct StopController {
    inner: Arc<Mutex<Inner>>,
    signal: Arc<watch::Sender<bool>>,
}

impl Default for StopController {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { state: StopState::default(), inflight: HashMap::new(), agents: HashMap::new(), listeners: vec![] })),
            signal: Arc::new(watch::channel(false).0),
        }
    }
}

impl std::fmt::Debug for StopController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<StopController {}>", self.report())
    }
}

impl StopController {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn state(&self) -> StopState {
        self.lock().state.clone()
    }

    pub fn stopped(&self) -> bool {
        self.lock().state.stopped
    }

    pub fn draining(&self) -> bool {
        let s = &self.lock().state;
        s.stopped && s.mode == "drain"
    }

    /// Fail with `StopRequested` if a stop is in force. The loop calls this at
    /// safe boundaries.
    pub fn check(&self, place: &str) -> Result<()> {
        let s = self.state();
        if !s.stopped {
            return Ok(());
        }
        let detail = if place.is_empty() { String::new() } else { format!(" at {place}") };
        let reason = if s.reason.is_empty() { "no reason given" } else { &s.reason };
        Err(Error::StopRequested(format!("run stopped by {}{detail}: {reason}", s.requested_by)))
    }

    /// False once a stop is in force — nothing new should be started.
    pub fn may_start(&self) -> bool {
        !self.stopped()
    }

    /// Stop the run. Idempotent — the first reason is the one that sticks.
    /// `mode` is `drain` or `abort`.
    pub fn stop(&self, reason: &str, mode: &str, requested_by: &str) -> StopState {
        let (state, listeners) = {
            let mut i = self.lock();
            if i.state.stopped {
                return i.state.clone();
            }
            i.state = StopState {
                stopped: true,
                mode: if mode == "abort" { "abort".into() } else { "drain".into() },
                reason: reason.into(),
                requested_by: if requested_by.is_empty() { "human".into() } else { requested_by.into() },
                ts: now(),
            };
            (i.state.clone(), i.listeners.clone())
        };
        self.signal.send_replace(true);
        for l in listeners {
            // A listener must never break the stop.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| l(&state)));
        }
        if state.mode == "abort" {
            self.cancel_inflight();
        }
        state
    }

    /// Stop and cancel everything in flight.
    pub fn abort(&self, reason: &str, requested_by: &str) -> StopState {
        self.stop(reason, "abort", requested_by)
    }

    /// Clear the stop so the controller can be reused for the next run.
    pub fn resume(&self) {
        self.lock().state = StopState::default();
        self.signal.send_replace(false);
    }

    pub fn on_stop(&self, listener: impl Fn(&StopState) + Send + Sync + 'static) {
        self.lock().listeners.push(Arc::new(listener));
    }

    /// Wait until someone stops the run. `false` on timeout.
    pub async fn wait(&self, timeout: Option<f64>) -> bool {
        let mut rx = self.signal.subscribe();
        let waiting = rx.wait_for(|stopped| *stopped);
        match timeout {
            Some(t) => tokio::time::timeout(std::time::Duration::from_secs_f64(t.max(0.0)), waiting).await.is_ok_and(|r| r.is_ok()),
            None => waiting.await.is_ok(),
        }
    }

    // ---- tracking what is in flight --------------------------------------
    pub fn track(&self, key: &str, task: AbortHandle) {
        let mut i = self.lock();
        i.inflight.retain(|_, h| !h.is_finished());
        i.inflight.insert(key.into(), task);
    }

    pub fn enter(&self, agent: &str, run_id: &str) {
        self.lock().agents.insert(run_id.into(), agent.into());
    }

    pub fn leave(&self, run_id: &str) {
        self.lock().agents.remove(run_id);
    }

    pub fn running_agents(&self) -> Vec<String> {
        let mut v: Vec<String> = self.lock().agents.values().cloned().collect();
        v.sort();
        v
    }

    pub fn cancel_inflight(&self) -> usize {
        let i = self.lock();
        let live: Vec<&AbortHandle> = i.inflight.values().filter(|h| !h.is_finished()).collect();
        for h in &live {
            h.abort();
        }
        live.len()
    }

    /// Wait for in-flight work to finish. `true` if everything landed in time.
    pub async fn drain(&self, timeout: f64) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout.max(0.0));
        loop {
            if self.lock().inflight.values().all(AbortHandle::is_finished) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    pub fn report(&self) -> Value {
        let i = self.lock();
        let mut agents: Vec<&String> = i.agents.values().collect();
        agents.sort();
        json!({
            "stopped": i.state.stopped, "mode": i.state.mode, "reason": i.state.reason,
            "requested_by": i.state.requested_by,
            "inflight": i.inflight.values().filter(|h| !h.is_finished()).count(),
            "agents_running": agents,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stops_once_and_aborts_what_is_in_flight() {
        let c = StopController::new();
        assert!(c.check("step 1").is_ok());
        let slow = tokio::spawn(async { tokio::time::sleep(std::time::Duration::from_secs(30)).await });
        c.track("slow", slow.abort_handle());
        let waiter = {
            let c = c.clone();
            tokio::spawn(async move { c.wait(Some(5.0)).await })
        };
        c.abort("customer withdrew", "ops");
        c.stop("second reason", "drain", "x");
        assert!(waiter.await.unwrap());
        assert!(slow.await.unwrap_err().is_cancelled());
        assert_eq!(c.check("tool search").unwrap_err().to_string(), "run stopped by ops at tool search: customer withdrew");
        assert!(c.drain(1.0).await);
        c.resume();
        assert!(c.may_start());
    }
}
