//! HNSW (Hierarchical Navigable Small World) backend implementation.
//!
//! Provides graph-based vector search with support for selective recomputation.

pub mod graph;
pub mod builder;
pub mod searcher;

pub use graph::HnswGraph;
