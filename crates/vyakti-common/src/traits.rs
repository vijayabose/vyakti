//! Core traits for Vyakti components.

use crate::{Result, SearchResult, Vector};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Graph degree (max connections per node)
    pub graph_degree: usize,
    /// Build complexity parameter
    pub build_complexity: usize,
    /// Search complexity parameter
    pub search_complexity: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            graph_degree: 32,
            build_complexity: 64,
            search_complexity: 32,
        }
    }
}

/// Backend trait for vector search implementations.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;

    /// Build index from vectors
    async fn build(&mut self, vectors: &[Vector], config: &BackendConfig) -> Result<()>;

    /// Search for nearest neighbors
    async fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>;

    /// Get number of vectors in the index
    fn len(&self) -> usize;

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Embedding provider trait for computing embeddings.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts into vectors
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get provider name
    fn name(&self) -> &str;
}
