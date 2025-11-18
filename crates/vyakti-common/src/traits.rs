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
            build_complexity: 128,
            search_complexity: 64,
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

    /// Set document data (text and metadata) for a node
    /// This is called after build() to associate text/metadata with vectors
    fn set_document_data(
        &mut self,
        node_id: usize,
        text: String,
        metadata: HashMap<String, serde_json::Value>,
    ) {
        // Default implementation does nothing
        // Backends can override to store text/metadata
        let _ = (node_id, text, metadata);
    }

    /// Get number of vectors in the index
    fn len(&self) -> usize;

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get mutable reference as Any for downcasting
    /// This allows downcasting to specific backend implementations
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get reference as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
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

/// Text generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic, higher = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling parameter
    pub top_p: f32,
    /// Number of threads for generation
    pub n_threads: u32,
    /// Stop sequences that end generation
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            n_threads: num_cpus::get() as u32,
            stop_sequences: vec![],
        }
    }
}

/// Text generation provider trait for LLM inference.
#[async_trait]
pub trait TextGenerationProvider: Send + Sync {
    /// Generate text based on a prompt
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Generate text with conversation context (chat format)
    async fn chat(&self, messages: &[(String, String)], config: &GenerationConfig) -> Result<String> {
        // Default implementation: concatenate messages into a prompt
        let prompt = messages
            .iter()
            .map(|(role, content)| format!("{}: {}", role, content))
            .collect::<Vec<_>>()
            .join("\n");

        self.generate(&prompt, config).await
    }

    /// Get provider name
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Mock Backend implementation for testing
    struct MockBackend {
        name: String,
        vectors: Vec<Vector>,
    }

    #[async_trait]
    impl Backend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        async fn build(&mut self, vectors: &[Vector], _config: &BackendConfig) -> Result<()> {
            self.vectors = vectors.to_vec();
            Ok(())
        }

        async fn search(&self, _query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
            let mut results = vec![];
            for (i, _vec) in self.vectors.iter().enumerate().take(k) {
                results.push(SearchResult {
                    id: i,
                    text: format!("document {}", i),
                    score: 0.5,
                    metadata: HashMap::new(),
                });
            }
            Ok(results)
        }

        fn len(&self) -> usize {
            self.vectors.len()
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // Mock EmbeddingProvider implementation for testing
    struct MockEmbeddingProvider {
        dimension: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
            Ok(texts.iter().map(|_| vec![0.0; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.graph_degree, 32);
        assert_eq!(config.build_complexity, 64);
        assert_eq!(config.search_complexity, 32);
    }

    #[test]
    fn test_backend_config_serialization() {
        let config = BackendConfig {
            graph_degree: 64,
            build_complexity: 128,
            search_complexity: 64,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: BackendConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.graph_degree, deserialized.graph_degree);
        assert_eq!(config.build_complexity, deserialized.build_complexity);
        assert_eq!(config.search_complexity, deserialized.search_complexity);
    }

    #[tokio::test]
    async fn test_mock_backend_build() {
        let mut backend = MockBackend {
            name: "test-backend".to_string(),
            vectors: vec![],
        };

        assert_eq!(backend.name(), "test-backend");
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());

        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();

        assert_eq!(backend.len(), 2);
        assert!(!backend.is_empty());
    }

    #[tokio::test]
    async fn test_mock_backend_search() {
        let mut backend = MockBackend {
            name: "test-backend".to_string(),
            vectors: vec![],
        };

        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let config = BackendConfig::default();
        backend.build(&vectors, &config).await.unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let results = backend.search(&query, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert_eq!(results[1].id, 1);
    }

    #[tokio::test]
    async fn test_mock_embedding_provider() {
        let provider = MockEmbeddingProvider { dimension: 384 };

        assert_eq!(provider.name(), "mock");
        assert_eq!(provider.dimension(), 384);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = provider.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);
    }

    #[test]
    fn test_backend_config_clone() {
        let config = BackendConfig::default();
        let cloned = config.clone();

        assert_eq!(config.graph_degree, cloned.graph_degree);
        assert_eq!(config.build_complexity, cloned.build_complexity);
        assert_eq!(config.search_complexity, cloned.search_complexity);
    }
}
