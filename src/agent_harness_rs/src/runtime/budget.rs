//! Budget and rate guard: a spend ceiling and throughput limits the run respects.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::time::Instant;

use crate::errors::{Error, Result};
use crate::types::Usage;

/// What hitting a ceiling means.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnExceed {
    /// End the run cleanly: keep what was produced, note the budget ran out.
    #[default]
    Stop,
    /// Make it an error.
    Raise,
}

/// Ceilings for one run. `None` means no limit on that axis.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Budget {
    pub max_usd: Option<f64>,
    pub max_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub max_steps: Option<u32>,
    pub max_seconds: Option<f64>,
    pub max_tool_calls: Option<u32>,
    pub max_subagents: Option<u32>,
    pub on_exceed: OnExceed,
}

impl Budget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn is_set(&self) -> bool {
        self.max_usd.is_some()
            || self.max_tokens.is_some()
            || self.max_input_tokens.is_some()
            || self.max_output_tokens.is_some()
            || self.max_steps.is_some()
            || self.max_seconds.is_some()
            || self.max_tool_calls.is_some()
            || self.max_subagents.is_some()
    }
}

fn exceeded(message: String, kind: &str, limit: f64, spent: f64) -> Error {
    Error::BudgetExceeded { message, kind: kind.into(), limit, spent }
}

#[derive(Debug, Default)]
struct Spend {
    usage: Usage,
    steps: u32,
    tool_calls: u32,
    subagents: u32,
    by_agent: BTreeMap<String, Usage>,
    by_task: BTreeMap<String, f64>,
}

/// Tracks spend as the run goes and stops it at the ceiling.
/// Attribution is per sub-agent and per task, so you can charge it back.
#[derive(Debug)]
pub struct BudgetGuard {
    pub budget: Budget,
    pub parent: Option<Arc<BudgetGuard>>,
    started: Instant,
    spend: Mutex<Spend>,
}

impl BudgetGuard {
    pub fn new(budget: Budget) -> Self {
        Self { budget, parent: None, started: Instant::now(), spend: Mutex::new(Spend::default()) }
    }

    /// A sub-agent's guard: its own caps, but spend rolls up to the parent.
    pub fn child(self: &Arc<Self>, budget: Budget) -> Arc<BudgetGuard> {
        Arc::new(Self { parent: Some(self.clone()), ..Self::new(budget) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Spend> {
        self.spend.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn usage(&self) -> Usage {
        self.lock().usage.clone()
    }

    pub fn steps(&self) -> u32 {
        self.lock().steps
    }

    pub fn tool_calls(&self) -> u32 {
        self.lock().tool_calls
    }

    pub fn subagents(&self) -> u32 {
        self.lock().subagents
    }

    /// What each agent has spent under this guard.
    pub fn by_agent(&self) -> BTreeMap<String, Usage> {
        self.lock().by_agent.clone()
    }

    // ---- recording ----------------------------------------------------
    pub fn record(&self, usage: &Usage, agent: &str, task: &str) -> Result<()> {
        {
            let mut s = self.lock();
            s.usage += usage;
            if !agent.is_empty() {
                *s.by_agent.entry(agent.into()).or_default() += usage;
            }
            if !task.is_empty() {
                *s.by_task.entry(task.into()).or_default() += usage.cost_usd;
            }
        }
        if let Some(p) = &self.parent {
            p.record(usage, agent, task)?;
        }
        self.check()
    }

    pub fn step(&self) -> Result<()> {
        let steps = {
            let mut s = self.lock();
            s.steps += 1;
            s.steps
        };
        if let Some(max) = self.budget.max_steps
            && steps > max
        {
            return Err(exceeded(format!("step cap reached ({max})"), "steps", max as f64, steps as f64));
        }
        self.check()
    }

    pub fn tool_call(&self) -> Result<()> {
        let calls = {
            let mut s = self.lock();
            s.tool_calls += 1;
            s.tool_calls
        };
        if let Some(max) = self.budget.max_tool_calls
            && calls > max
        {
            return Err(exceeded(format!("tool-call cap reached ({max})"), "tool_calls", max as f64, calls as f64));
        }
        Ok(())
    }

    pub fn subagent(&self) -> Result<()> {
        let n = {
            let mut s = self.lock();
            s.subagents += 1;
            s.subagents
        };
        if let Some(max) = self.budget.max_subagents
            && n > max
        {
            return Err(exceeded(format!("sub-agent cap reached ({max})"), "subagents", max as f64, n as f64));
        }
        if let Some(p) = &self.parent {
            p.subagent()?;
        }
        Ok(())
    }

    // ---- checking -----------------------------------------------------
    pub fn check(&self) -> Result<()> {
        let b = &self.budget;
        let u = self.usage();
        if let Some(max) = b.max_usd
            && u.cost_usd > max
        {
            return Err(exceeded(format!("spend ceiling reached: ${:.4} of ${max:.4}", u.cost_usd), "usd", max, u.cost_usd));
        }
        if let Some(max) = b.max_tokens
            && u.total_tokens() > max
        {
            return Err(exceeded(format!("token cap reached: {} of {max}", u.total_tokens()), "tokens", max as f64, u.total_tokens() as f64));
        }
        if let Some(max) = b.max_input_tokens
            && u.input_tokens > max
        {
            return Err(exceeded(format!("input-token cap reached: {} of {max}", u.input_tokens), "input_tokens", max as f64, u.input_tokens as f64));
        }
        if let Some(max) = b.max_output_tokens
            && u.output_tokens > max
        {
            return Err(exceeded(format!("output-token cap reached: {} of {max}", u.output_tokens), "output_tokens", max as f64, u.output_tokens as f64));
        }
        let elapsed = self.elapsed();
        if let Some(max) = b.max_seconds
            && elapsed > max
        {
            return Err(exceeded(format!("deadline passed after {elapsed:.1}s"), "seconds", max, elapsed));
        }
        Ok(())
    }

    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn remaining_usd(&self) -> Option<f64> {
        self.budget.max_usd.map(|max| (max - self.usage().cost_usd).max(0.0))
    }

    /// What is left on each axis that has a ceiling.
    pub fn remaining(&self) -> BTreeMap<&'static str, f64> {
        let (b, u, steps) = (&self.budget, self.usage(), self.steps());
        let mut left = BTreeMap::new();
        if let Some(m) = b.max_usd {
            left.insert("usd", (m - u.cost_usd).max(0.0));
        }
        if let Some(m) = b.max_tokens {
            left.insert("tokens", m.saturating_sub(u.total_tokens()) as f64);
        }
        if let Some(m) = b.max_input_tokens {
            left.insert("input_tokens", m.saturating_sub(u.input_tokens) as f64);
        }
        if let Some(m) = b.max_output_tokens {
            left.insert("output_tokens", m.saturating_sub(u.output_tokens) as f64);
        }
        if let Some(m) = b.max_steps {
            left.insert("steps", m.saturating_sub(steps) as f64);
        }
        if let Some(m) = b.max_seconds {
            left.insert("seconds", (m - self.elapsed()).max(0.0));
        }
        left
    }

    /// Which ceiling the next call would cross, before it is made ("" for none).
    pub fn would_exceed(&self, estimated_input: u64) -> &'static str {
        let (b, u) = (&self.budget, self.usage());
        if b.max_input_tokens.is_some_and(|m| u.input_tokens + estimated_input > m) {
            return "input_tokens";
        }
        if b.max_tokens.is_some_and(|m| u.total_tokens() + estimated_input > m) {
            return "tokens";
        }
        ""
    }

    pub fn report(&self) -> Value {
        let r6 = |x: f64| (x * 1e6).round() / 1e6;
        let s = self.lock();
        json!({
            "cost_usd": r6(s.usage.cost_usd),
            "tokens": s.usage.total_tokens(),
            "calls": s.usage.calls,
            "steps": s.steps,
            "tool_calls": s.tool_calls,
            "subagents": s.subagents,
            "elapsed_s": (self.elapsed() * 1000.0).round() / 1000.0,
            "by_agent": s.by_agent.iter().map(|(k, v)| (k.clone(), json!(r6(v.cost_usd)))).collect::<serde_json::Map<_, _>>(),
            "by_task": s.by_task.iter().map(|(k, v)| (k.clone(), json!(r6(*v)))).collect::<serde_json::Map<_, _>>(),
        })
    }
}

impl Default for BudgetGuard {
    fn default() -> Self {
        Self::new(Budget::default())
    }
}

/// Throughput ceilings. `None` means no limit on that axis.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RateLimit {
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u64>,
    pub max_concurrent: Option<usize>,
}

#[derive(Debug, Default)]
struct Window {
    requests: VecDeque<Instant>,
    tokens: VecDeque<(Instant, u64)>,
    waits: u64,
    total_wait_s: f64,
}

/// A sliding-window throughput limiter that waits rather than failing.
/// Pacing ourselves costs nothing; being rate-limited costs a round trip.
#[derive(Debug)]
pub struct RateGuard {
    pub limit: RateLimit,
    pub window: Duration,
    state: tokio::sync::Mutex<Window>,
    slots: Option<Semaphore>,
}

impl RateGuard {
    pub fn new(limit: RateLimit) -> Self {
        Self::with_window(limit, Duration::from_secs(60))
    }

    pub fn with_window(limit: RateLimit, window: Duration) -> Self {
        let slots = limit.max_concurrent.filter(|n| *n > 0).map(Semaphore::new);
        Self { limit, window, state: tokio::sync::Mutex::new(Window::default()), slots }
    }

    pub fn active(&self) -> bool {
        self.limit.requests_per_minute.is_some() || self.limit.tokens_per_minute.is_some()
    }

    fn prune(&self, w: &mut Window, now: Instant) {
        let Some(cutoff) = now.checked_sub(self.window) else { return };
        while w.requests.front().is_some_and(|t| *t < cutoff) {
            w.requests.pop_front();
        }
        while w.tokens.front().is_some_and(|(t, _)| *t < cutoff) {
            w.tokens.pop_front();
        }
    }

    /// How long until this call fits inside both windows.
    fn delay(&self, w: &mut Window, now: Instant, estimated: u64) -> Duration {
        self.prune(w, now);
        let mut waits = Vec::new();
        if let Some(rpm) = self.limit.requests_per_minute
            && w.requests.len() >= rpm as usize
            && let Some(first) = w.requests.front()
        {
            waits.push((*first + self.window).saturating_duration_since(now));
        }
        if let Some(tpm) = self.limit.tokens_per_minute
            && let Some((first, _)) = w.tokens.front()
        {
            let spent: u64 = w.tokens.iter().map(|(_, t)| t).sum();
            if spent + estimated > tpm {
                waits.push((*first + self.window).saturating_duration_since(now));
            }
        }
        waits.into_iter().max().unwrap_or_default()
    }

    /// Wait until the call fits the limits. Returns seconds waited.
    pub async fn acquire(&self, estimated_tokens: u64) -> f64 {
        if !self.active() {
            return 0.0;
        }
        let mut waited = 0.0;
        loop {
            let delay = {
                let mut w = self.state.lock().await;
                let now = Instant::now();
                let d = self.delay(&mut w, now, estimated_tokens);
                if d.is_zero() {
                    w.requests.push_back(now);
                    if estimated_tokens > 0 {
                        w.tokens.push_back((now, estimated_tokens));
                    }
                    return waited;
                }
                w.waits += 1;
                w.total_wait_s += d.as_secs_f64();
                d
            };
            tokio::time::sleep(delay.min(self.window)).await;
            waited += delay.as_secs_f64();
        }
    }

    /// Correct the token window with what the call actually used.
    pub async fn record(&self, usage: &Usage) {
        if self.limit.tokens_per_minute.is_some() && usage.total_tokens() > 0 {
            self.state.lock().await.tokens.push_back((Instant::now(), usage.total_tokens()));
        }
    }

    /// Hold a concurrency slot for the duration of a call.
    pub async fn slot(&self) -> Option<SemaphorePermit<'_>> {
        match &self.slots {
            Some(s) => s.acquire().await.ok(),
            None => None,
        }
    }

    pub async fn stats(&self) -> Value {
        let mut w = self.state.lock().await;
        self.prune(&mut w, Instant::now());
        json!({
            "requests_in_window": w.requests.len(),
            "tokens_in_window": w.tokens.iter().map(|(_, t)| t).sum::<u64>(),
            "waits": w.waits,
            "total_wait_s": (w.total_wait_s * 1000.0).round() / 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(e: Error) -> String {
        match e {
            Error::BudgetExceeded { kind, .. } => kind,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn caps_trip() {
        let g = BudgetGuard::new(Budget { max_steps: Some(1), max_tool_calls: Some(1), max_usd: Some(0.5), ..Default::default() });
        g.step().unwrap();
        assert_eq!(kind(g.step().unwrap_err()), "steps");
        g.tool_call().unwrap();
        assert_eq!(kind(g.tool_call().unwrap_err()), "tool_calls");
        let err = g.record(&Usage { cost_usd: 0.6, ..Default::default() }, "a", "t").unwrap_err();
        assert_eq!(err.to_string(), "spend ceiling reached: $0.6000 of $0.5000");
        assert_eq!(g.report()["by_agent"]["a"], 0.6);
    }

    #[test]
    fn child_rolls_up() {
        let parent = Arc::new(BudgetGuard::new(Budget { max_tokens: Some(100), max_subagents: Some(1), ..Default::default() }));
        let child = parent.child(Budget::default());
        child.record(&Usage { input_tokens: 60, ..Default::default() }, "kid", "").unwrap();
        assert_eq!(parent.usage().input_tokens, 60);
        assert_eq!(parent.would_exceed(50), "tokens");
        assert_eq!(kind(child.record(&Usage { output_tokens: 50, ..Default::default() }, "", "").unwrap_err()), "tokens");
        child.subagent().unwrap();
        assert_eq!(kind(child.subagent().unwrap_err()), "subagents");
        assert_eq!(parent.remaining()["tokens"], 0.0);
    }

    #[tokio::test(start_paused = true)]
    async fn rate_guard_waits_for_room() {
        let g = RateGuard::with_window(RateLimit { requests_per_minute: Some(2), ..Default::default() }, Duration::from_secs(10));
        assert_eq!(g.acquire(0).await, 0.0);
        assert_eq!(g.acquire(0).await, 0.0);
        let waited = g.acquire(0).await;
        assert!((9.9..=10.1).contains(&waited), "{waited}");
        assert_eq!(g.stats().await["waits"], 1);
    }
}
