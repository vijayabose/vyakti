//! HNSW (Hierarchical Navigable Small World) backend implementation.
//!
//! Provides graph-based vector search with support for selective recomputation.

pub mod builder;
pub mod graph;
pub mod pruning;
pub mod searcher;

pub use builder::HnswBackend;
pub use graph::{DocumentData, Edge, HnswConfig, HnswGraph, NodeId};
pub use pruning::{GraphPruner, PruningConfig, PruningStats};
pub use searcher::HnswSearcher;
