//! Human oversight: approvals that fail closed.
//!
//! A desk with no approver is a queue people answer through your API:
//!
//! ```no_run
//! # use agent_harness::governance::OversightDesk;
//! # let desk = OversightDesk::default();
//! for request in desk.pending() {
//!     desk.approve(&request.id, "maria@acme.com", "checked the invoice");
//! }
//! ```
//!
//! or give it an [`Approver`] and let it ask.
//!
//! The rules every framework lands on, built in:
//!
//! * **Fail closed.** No approver, no answer before the timeout, an approver
//!   that fails — all of those are a *no*.
//! * **A quorum of distinct people.** `approvers: 2` needs two different names.
//!   An anonymous yes counts as one vote, however many times it is given, so
//!   four-eyes cannot be satisfied by one rubber stamp.
//! * **Separation of duties.** The person the agent is acting for cannot
//!   approve their own request.
//! * **Any denial ends it.** One *no* outweighs any number of *yes*.
//!
//! Every request and every vote goes into the audit trail.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::Notify;

use crate::types::{new_id, now};

/// One person's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ApprovalVote {
    pub approve: bool,
    pub by: String,
    pub note: String,
    pub ts: f64,
}

impl ApprovalVote {
    pub fn new(approve: bool, by: &str, note: &str) -> Self {
        Self { approve, by: by.into(), note: note.into(), ts: now() }
    }
}

impl From<bool> for ApprovalVote {
    /// An anonymous yes or no.
    fn from(approve: bool) -> Self {
        Self::new(approve, "", "")
    }
}

/// Something waiting for people: what, for whom, how many must agree.
#[derive(Debug, Clone, PartialEq)]
pub struct ApprovalRequest {
    pub agent: String,
    pub action: String,
    pub target: String,
    pub reason: String,
    pub args: Map<String, Value>,
    pub rules: Vec<String>,
    pub approvers: i64,
    pub timeout: Option<f64>,
    pub requester: String,
    pub run_id: String,
    pub id: String,
    pub created: f64,
    /// `pending`, `approved`, `denied` or `expired`.
    pub status: String,
    pub votes: Vec<ApprovalVote>,
}

impl ApprovalRequest {
    pub fn new(agent: &str, action: &str, target: &str) -> Self {
        Self {
            agent: agent.into(),
            action: action.into(),
            target: target.into(),
            reason: String::new(),
            args: Map::new(),
            rules: vec![],
            approvers: 1,
            timeout: None,
            requester: String::new(),
            run_id: String::new(),
            id: new_id("apr"),
            created: now(),
            status: "pending".into(),
            votes: vec![],
        }
    }

    pub fn approved_by(&self) -> BTreeSet<String> {
        self.votes.iter().filter(|v| v.approve).map(|v| if v.by.is_empty() { "(anonymous)".to_string() } else { v.by.clone() }).collect()
    }

    pub fn summary(&self) -> Value {
        json!({
            "id": self.id, "agent": self.agent, "action": self.action, "target": self.target,
            "reason": self.reason, "args": self.args, "approvers": self.approvers, "status": self.status,
            "votes": self.votes.iter().map(|v| json!({"approve": v.approve, "by": v.by, "note": v.note})).collect::<Vec<_>>(),
        })
    }
}

/// Anything that answers a request: one call per vote still needed. An error is a no.
pub type Approver = Arc<dyn Fn(ApprovalRequest) -> BoxFuture<'static, std::result::Result<ApprovalVote, String>> + Send + Sync>;

/// An approver from a plain function answering a bool or a vote.
pub fn approver<V: Into<ApprovalVote>>(f: impl Fn(&ApprovalRequest) -> std::result::Result<V, String> + Send + Sync + 'static) -> Approver {
    let f = Arc::new(f);
    Arc::new(move |r: ApprovalRequest| {
        let f = f.clone();
        Box::pin(async move { f(&r).map(Into::into) })
    })
}

/// Told about each request as it starts waiting — post it to Slack, a ticket
/// queue, a web page. A notifier that fails denies the request.
pub type ApprovalNotify = Arc<dyn Fn(ApprovalRequest) -> BoxFuture<'static, std::result::Result<(), String>> + Send + Sync>;

/// Called with every request as it is asked and decided (`approval_requested`,
/// `approval_decided`); governance records these in the audit trail.
pub type ApprovalRecord = Arc<dyn Fn(&ApprovalRequest, &str) + Send + Sync>;

struct Waiting {
    request: Mutex<ApprovalRequest>,
    done: Notify,
}

/// Where approval requests wait for people.
pub struct OversightDesk {
    /// Called once per vote still needed. Leave it out to answer via `approve` /
    /// `deny` from elsewhere (a web handler, a Slack action).
    pub approver: Option<Approver>,
    pub default_timeout: f64,
    pub record: Option<ApprovalRecord>,
    pub notify: Option<ApprovalNotify>,
    requests: Mutex<IndexMap<String, Arc<Waiting>>>,
}

impl std::fmt::Debug for OversightDesk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<OversightDesk {} pending>", self.pending().len())
    }
}

impl Default for OversightDesk {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OversightDesk {
    pub fn new(approver: Option<Approver>) -> Self {
        Self { approver, default_timeout: 900.0, record: None, notify: None, requests: Mutex::new(IndexMap::new()) }
    }

    fn all(&self) -> std::sync::MutexGuard<'_, IndexMap<String, Arc<Waiting>>> {
        self.requests.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn note(&self, request: &ApprovalRequest, action: &str) {
        if let Some(r) = &self.record {
            r(request, action);
        }
    }

    fn snapshot(w: &Waiting) -> ApprovalRequest {
        w.request.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Wait for a decision. Returns the request with its final status.
    pub async fn request(&self, request: ApprovalRequest) -> ApprovalRequest {
        let id = request.id.clone();
        let waiting = Arc::new(Waiting { request: Mutex::new(request), done: Notify::new() });
        self.all().insert(id, waiting.clone());
        let current = Self::snapshot(&waiting);
        self.note(&current, "approval_requested");
        let set_denied = |vote: ApprovalVote| {
            let mut r = waiting.request.lock().unwrap_or_else(|p| p.into_inner());
            r.votes.push(vote);
            r.status = "denied".into();
        };
        if let Some(n) = &self.notify
            && let Err(e) = n(current.clone()).await
        {
            set_denied(ApprovalVote::new(false, "system", &format!("notify failed: {e}")));
            let done = Self::snapshot(&waiting);
            self.note(&done, "approval_decided");
            return done;
        }
        let timeout = current.timeout.unwrap_or(self.default_timeout).max(0.0);
        let wait = async {
            match &self.approver {
                Some(a) => self.ask(&waiting, a).await,
                None => {
                    loop {
                        if Self::snapshot(&waiting).status != "pending" {
                            break;
                        }
                        waiting.done.notified().await;
                    }
                    Ok(())
                }
            }
        };
        match tokio::time::timeout(Duration::from_secs_f64(timeout), wait).await {
            Err(_) => {
                let mut r = waiting.request.lock().unwrap_or_else(|p| p.into_inner());
                if r.status == "pending" {
                    r.status = "expired".into();
                }
            }
            // An approver that breaks is a no.
            Ok(Err(e)) => set_denied(ApprovalVote::new(false, "system", &format!("approver failed: {e}"))),
            Ok(Ok(())) => {}
        }
        {
            let mut r = waiting.request.lock().unwrap_or_else(|p| p.into_inner());
            if r.status == "pending" {
                r.status = "denied".into();
            }
        }
        let done = Self::snapshot(&waiting);
        self.note(&done, "approval_decided");
        done
    }

    async fn ask(&self, waiting: &Waiting, approver: &Approver) -> std::result::Result<(), String> {
        // Ask at most `approvers` times plus a little slack for anonymous repeats.
        let rounds = Self::snapshot(waiting).approvers.max(1) * 2;
        for _ in 0..rounds {
            let vote = approver(Self::snapshot(waiting)).await?;
            Self::vote(waiting, vote);
            if Self::snapshot(waiting).status != "pending" {
                return Ok(());
            }
        }
        waiting.request.lock().unwrap_or_else(|p| p.into_inner()).status = "denied".into();
        Ok(())
    }

    fn vote(waiting: &Waiting, vote: ApprovalVote) -> ApprovalRequest {
        let mut r = waiting.request.lock().unwrap_or_else(|p| p.into_inner());
        if r.status != "pending" {
            return r.clone();
        }
        let vote = if vote.approve && !r.requester.is_empty() && vote.by == r.requester {
            ApprovalVote::new(false, &vote.by, "separation of duties: cannot approve your own request")
        } else {
            vote
        };
        let approve = vote.approve;
        r.votes.push(vote);
        if !approve {
            r.status = "denied".into();
        } else if r.approved_by().len() as i64 >= r.approvers {
            r.status = "approved".into();
        }
        if r.status != "pending" {
            waiting.done.notify_one();
        }
        r.clone()
    }

    // ---- the human side ------------------------------------------------------

    pub fn pending(&self) -> Vec<ApprovalRequest> {
        self.all().values().map(|w| Self::snapshot(w)).filter(|r| r.status == "pending").collect()
    }

    pub fn get(&self, request_id: &str) -> Option<ApprovalRequest> {
        self.all().get(request_id).map(|w| Self::snapshot(w))
    }

    pub fn approve(&self, request_id: &str, by: &str, note: &str) -> Option<ApprovalRequest> {
        let w = self.all().get(request_id).cloned()?;
        Some(Self::vote(&w, ApprovalVote::new(true, by, note)))
    }

    pub fn deny(&self, request_id: &str, by: &str, note: &str) -> Option<ApprovalRequest> {
        let w = self.all().get(request_id).cloned()?;
        Some(Self::vote(&w, ApprovalVote::new(false, by, note)))
    }

    pub fn history(&self) -> Vec<Value> {
        self.all().values().map(|w| Self::snapshot(w).summary()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(approvers: i64, requester: &str, timeout: f64) -> ApprovalRequest {
        ApprovalRequest { approvers, requester: requester.into(), timeout: Some(timeout), ..ApprovalRequest::new("a", "tool", "pay") }
    }

    #[tokio::test]
    async fn four_eyes_needs_two_distinct_people_and_not_the_requester() {
        let desk = Arc::new(OversightDesk::default());
        let r = req(2, "alice", 2.0);
        let id = r.id.clone();
        let d = desk.clone();
        let waiting = tokio::spawn(async move { d.request(r).await });
        while desk.get(&id).is_none() {
            tokio::task::yield_now().await;
        }
        assert_eq!(desk.approve(&id, "alice", "").unwrap().status, "denied");
        assert_eq!(waiting.await.unwrap().status, "denied");

        let r = req(2, "", 2.0);
        let id = r.id.clone();
        let d = desk.clone();
        let waiting = tokio::spawn(async move { d.request(r).await });
        while desk.get(&id).is_none() {
            tokio::task::yield_now().await;
        }
        desk.approve(&id, "maria", "");
        assert_eq!(desk.approve(&id, "maria", "").unwrap().status, "pending");
        desk.approve(&id, "omar", "");
        assert_eq!(waiting.await.unwrap().status, "approved");
    }

    #[tokio::test]
    async fn approvals_fail_closed() {
        assert_eq!(OversightDesk::default().request(req(1, "", 0.01)).await.status, "expired");
        let broken = OversightDesk::new(Some(approver(|_| Err::<bool, _>("slack is down".into()))));
        assert_eq!(broken.request(req(1, "", 5.0)).await.status, "denied");
        let rubber = OversightDesk::new(Some(approver(|_| Ok(true))));
        assert_eq!(rubber.request(req(2, "", 5.0)).await.status, "denied");
        let names = Mutex::new(vec!["ben", "ana"]);
        let named = OversightDesk::new(Some(approver(move |_| Ok(ApprovalVote::new(true, names.lock().unwrap().pop().unwrap_or(""), "")))));
        assert_eq!(named.request(req(2, "", 5.0)).await.status, "approved");
    }

    #[tokio::test]
    async fn a_broken_announcer_fails_closed() {
        let desk = OversightDesk { notify: Some(Arc::new(|_| Box::pin(async { Err("slack down".to_string()) }))), ..Default::default() };
        assert_eq!(desk.request(req(1, "", 1.0)).await.status, "denied");
    }
}
