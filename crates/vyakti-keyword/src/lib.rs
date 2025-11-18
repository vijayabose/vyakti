//! Vyakti Keyword Search
//!
//! This crate provides BM25-based keyword search capabilities for the Vyakti vector database.
//! It complements semantic vector search with exact term matching and phrase queries.
//!
//! # Features
//!
//! - **BM25 Scoring**: Industry-standard keyword search algorithm
//! - **Phrase Queries**: Exact phrase matching with quotes
//! - **Configurable Parameters**: Tune k1 and b for your workload
//! - **Tantivy Backend**: Fast, production-ready full-text search
//!
//! # Example
//!
//! ```no_run
//! use vyakti_keyword::{KeywordIndexBuilder, KeywordSearcher, KeywordConfig};
//! use std::collections::HashMap;
//! use std::path::Path;
//!
//! # fn example() -> anyhow::Result<()> {
//! let config = KeywordConfig::default();
//!
//! // Build an index
//! let index_path = Path::new("/tmp/my-index");
//! let mut builder = KeywordIndexBuilder::new(index_path, config.clone())?;
//!
//! let metadata = HashMap::new();
//! builder.add_document(0, "Hello world", &metadata)?;
//! builder.add_document(1, "Rust programming", &metadata)?;
//! builder.commit(index_path)?;
//!
//! // Search the index
//! let searcher = KeywordSearcher::load(index_path, config)?;
//! let results = searcher.search("rust", 10)?;
//!
//! for result in results {
//!     println!("Node {}: score {}", result.node_id, result.score);
//! }
//! # Ok(())
//! # }
//! ```

pub mod builder;
pub mod searcher;
pub mod types;

// Re-export main types for convenience
pub use builder::KeywordIndexBuilder;
pub use searcher::{IndexStats, KeywordSearcher};
pub use types::{Highlight, KeywordConfig, KeywordResult};
