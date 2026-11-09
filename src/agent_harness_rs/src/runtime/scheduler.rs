//! Concurrency scheduler: queue, semaphore, backpressure.
//!
//! Parallel sub-agents are the point of the fan-out, but an unbounded fan-out
//! is how you get rate-limited. Everything that runs many-at-once goes
//! through here.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::errors::{Error, Result};

tokio::task_local! {
    /// How many slots of a given scheduler this task already holds, by scheduler id.
    static DEPTH: Vec<usize>;
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Debug)]
struct Counters {
    running: AtomicUsize,
    peak: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
    queued: AtomicUsize,
}

/// A semaphore with bookkeeping, plus a gather that never runs away. Cheap to
/// clone; clones share the cap.
#[derive(Clone)]
pub struct ConcurrencyScheduler {
    id: usize,
    limit: Arc<RwLock<(usize, Arc<Semaphore>)>>,
    max_queue: usize,
    counters: Arc<Counters>,
}

impl std::fmt::Debug for ConcurrencyScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<ConcurrencyScheduler {}>", self.stats())
    }
}

impl Default for ConcurrencyScheduler {
    fn default() -> Self {
        Self::new(8, 0)
    }
}

impl ConcurrencyScheduler {
    /// At most `max_concurrency` at once; at most `max_queue` waiting (0: no cap).
    pub fn new(max_concurrency: usize, max_queue: usize) -> Self {
        let limit = max_concurrency.max(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            limit: Arc::new(RwLock::new((limit, Arc::new(Semaphore::new(limit))))),
            max_queue,
            counters: Arc::new(Counters {
                running: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                completed: AtomicUsize::new(0),
                failed: AtomicUsize::new(0),
                queued: AtomicUsize::new(0),
            }),
        }
    }

    pub fn max_concurrency(&self) -> usize {
        self.limit.read().unwrap_or_else(|p| p.into_inner()).0
    }

    /// Change the cap. Set it while the scheduler is idle.
    pub fn set_max_concurrency(&self, value: usize) {
        let v = value.max(1);
        *self.limit.write().unwrap_or_else(|p| p.into_inner()) = (v, Arc::new(Semaphore::new(v)));
    }

    pub fn running(&self) -> usize {
        self.counters.running.load(Ordering::SeqCst)
    }

    pub fn queued(&self) -> usize {
        self.counters.queued.load(Ordering::SeqCst)
    }

    fn held(&self) -> bool {
        DEPTH.try_with(|d| d.contains(&self.id)).unwrap_or(false)
    }

    async fn counted<T>(&self, fut: impl Future<Output = Result<T>>) -> Result<T> {
        let c = &self.counters;
        let now = c.running.fetch_add(1, Ordering::SeqCst) + 1;
        c.peak.fetch_max(now, Ordering::SeqCst);
        let out = fut.await;
        c.running.fetch_sub(1, Ordering::SeqCst);
        match &out {
            Ok(_) => c.completed.fetch_add(1, Ordering::SeqCst),
            Err(_) => c.failed.fetch_add(1, Ordering::SeqCst),
        };
        out
    }

    /// Await `fut` under the concurrency cap.
    pub async fn run<T>(&self, fut: impl Future<Output = Result<T>>) -> Result<T> {
        // Work fanned out from inside a slot does not take a second one. Without
        // this, a manager that delegates N tasks in parallel holds every slot
        // while its sub-agents wait for one — a deadlock, not slow progress.
        if self.held() {
            return self.counted(fut).await;
        }
        if self.max_queue > 0 && self.queued() >= self.max_queue {
            return Err(Error::config(format!("scheduler queue is full ({})", self.max_queue)));
        }
        let sem = self.limit.read().unwrap_or_else(|p| p.into_inner()).1.clone();
        self.counters.queued.fetch_add(1, Ordering::SeqCst);
        let permit = sem.acquire_owned().await;
        self.counters.queued.fetch_sub(1, Ordering::SeqCst);
        let _permit = permit.map_err(|_| Error::config("the scheduler was closed"))?;
        let mut depth = DEPTH.try_with(Clone::clone).unwrap_or_default();
        depth.push(self.id);
        DEPTH.scope(depth, self.counted(fut)).await
    }

    /// Run `f` over `items` in parallel, capped. Failures come back as values.
    pub async fn map<I, T, F, Fut>(&self, items: impl IntoIterator<Item = I>, f: F) -> Vec<Result<T>>
    where
        F: Fn(I) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        futures::future::join_all(items.into_iter().map(|item| self.run(f(item)))).await
    }

    pub fn stats(&self) -> Value {
        let c = &self.counters;
        json!({
            "running": c.running.load(Ordering::SeqCst), "queued": c.queued.load(Ordering::SeqCst),
            "peak": c.peak.load(Ordering::SeqCst), "completed": c.completed.load(Ordering::SeqCst),
            "failed": c.failed.load(Ordering::SeqCst), "limit": self.max_concurrency(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn caps_and_never_deadlocks_on_nested_fan_out() {
        let s = ConcurrencyScheduler::new(2, 0);
        let out = s
            .map(0..6u64, |n| async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok(n * 2)
            })
            .await;
        assert_eq!(out.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>(), vec![0, 2, 4, 6, 8, 10]);
        assert_eq!(s.stats()["peak"], 2);
        // Two outer jobs each fan out inside their slot: with a cap of 2 this
        // would deadlock if the inner work needed slots of its own.
        let inner = s.clone();
        let nested = s
            .map(0..2u64, |n| {
                let inner = inner.clone();
                async move {
                    let sub = inner.map(0..3u64, |m| async move { Ok(m + n) }).await;
                    Ok(sub.into_iter().map(|r| r.unwrap()).sum::<u64>())
                }
            })
            .await;
        assert_eq!(nested.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>(), vec![3, 6]);
        let failing = s.run(async { Err::<(), _>(Error::config("nope")) }).await;
        assert!(failing.is_err());
        assert_eq!(s.stats()["failed"], 1);
    }
}
