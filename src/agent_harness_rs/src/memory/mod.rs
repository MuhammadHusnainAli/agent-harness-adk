//! Memory: four scopes, one manager, and pluggable storage underneath.
//!
//! ```
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::memory::{MemoryManager, MemoryOptions, Trace};
//!
//! let memory = MemoryManager::new(MemoryOptions { trace: Some(Trace::user("alice")), ..Default::default() });
//! memory.remember("Bills in EUR", "preference", "user", vec![]).await?;
//! assert_eq!(memory.user.load().await?, "- [preference] Bills in EUR");
//! # Ok(()) }
//! ```
//!
//! Every agent has one by default (`.memory(false)` turns it off): `user.md`
//! is in its system prompt, relevant precedent is recalled for each task, and
//! it can `remember` and `recall` for itself. [`memory_provider`] builds a
//! store from a URL — files, SQLite, Postgres, Redis, S3, ...

pub mod base;
pub mod manager;
pub mod providers;
pub mod semantic;
pub mod trace;

pub use base::{FileStore, InMemoryStore, MemoryRecord, MemoryStore, RecordQuery};
pub use manager::{MemoryManager, MemoryOptions, OrchestratorMemory, SessionMemory, SubAgentMemory, UserMemory, WriteHook};
pub use providers::{available as available_backends, memory_provider};
pub use semantic::{DatabaseIndex, Embedder, HashEmbedder, LocalIndex, MemoryIndex, ProviderEmbedder, SemanticMemory, cosine};
pub use trace::{Scope, Trace};
