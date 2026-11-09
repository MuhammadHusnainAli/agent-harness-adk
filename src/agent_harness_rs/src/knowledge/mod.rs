//! Knowledge bases, over any vector database.
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::AgentBuilder) -> agent_harness::Result<()> {
//! use agent_harness::knowledge::{KnowledgeBase, NewDocument};
//!
//! let kb = KnowledgeBase::new("qdrant://localhost:6333/handbook")?;
//! kb.add(NewDocument::path("handbook.md")).await?;
//! let agent = agent.tool(kb.as_tool("search_knowledge")).build()?;
//! # Ok(()) }
//! ```
//!
//! [`KnowledgeBase`] is the documents-in, passages-out layer; [`vector_store`]
//! is the database under it, and [`vector_stores`] lists the ones that ship.

pub mod base;
pub mod kb;
pub mod stores;

pub use base::{Condition, HttpClient, Reply, VectorHit, VectorRecord, VectorStore, conditions, cosine, holds, normalise};
pub use kb::{Document, KnowledgeBase, KnowledgeBaseBuilder, NewDocument, Passage, Reranker, Retriever, chunk_text};
pub use stores::{STORES, UrlParts, vector_store, vector_stores};
