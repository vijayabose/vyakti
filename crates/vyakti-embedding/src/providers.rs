//! Embedding providers.

use async_trait::async_trait;
use vyakti_common::{EmbeddingProvider, Result, Vector};

/// Mock embedding provider for testing
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        // Return zero vectors for now
        Ok(vec![vec![0.0; self.dimension]; texts.len()])
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "mock"
    }
}
