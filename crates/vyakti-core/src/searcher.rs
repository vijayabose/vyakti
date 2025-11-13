//! Index searcher.

use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use vyakti_common::{
    Backend, EmbeddingProvider, MetadataFilterEngine, MetadataFilters, Result, SearchResult,
    VyaktiError,
};

/// Searcher for querying Vyakti indexes
pub struct VyaktiSearcher {
    /// Backend for vector search
    backend: Arc<RwLock<Box<dyn Backend>>>,
    /// Embedding provider
    embedding_provider: Arc<dyn EmbeddingProvider>,
    /// Metadata filter engine
    filter_engine: MetadataFilterEngine,
}

impl VyaktiSearcher {
    /// Create a new searcher with backend and embedding provider
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend implementation for vector search
    /// * `embedding_provider` - Provider for computing embeddings
    ///
    /// # Returns
    ///
    /// A new VyaktiSearcher instance
    pub fn new(backend: Box<dyn Backend>, embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            backend: Arc::new(RwLock::new(backend)),
            embedding_provider,
            filter_engine: MetadataFilterEngine::new(),
        }
    }

    /// Load an index from disk
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the saved index file
    /// * `backend` - Backend implementation for vector search
    /// * `embedding_provider` - Provider for computing embeddings
    ///
    /// # Returns
    ///
    /// A new VyaktiSearcher instance with the loaded index
    pub async fn load<P: AsRef<Path>>(
        path: P,
        mut backend: Box<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        // Load index data from disk
        let index_data = crate::persistence::load_index(path)?;

        // Validate embedding dimensions
        if index_data.metadata.dimension != embedding_provider.dimension() {
            return Err(VyaktiError::Embedding(format!(
                "Dimension mismatch: index has {}, embedding provider has {}",
                index_data.metadata.dimension,
                embedding_provider.dimension()
            )));
        }

        // Reconstruct full vectors array if loading compact index
        let full_vectors = if let Some(ref pruned_nodes) = index_data.pruned_nodes {
            // Compact mode: reconstruct full vectors array with zero vectors for pruned nodes
            let num_nodes = index_data.metadata.num_documents;
            let dimension = index_data.metadata.dimension;
            let pruned_set: std::collections::HashSet<usize> =
                pruned_nodes.iter().copied().collect();

            let mut full_vecs = Vec::with_capacity(num_nodes);
            let mut hub_idx = 0;

            for node_id in 0..num_nodes {
                if pruned_set.contains(&node_id) {
                    // Pruned node: insert zero vector (will be recomputed on demand)
                    full_vecs.push(vec![0.0; dimension]);
                } else {
                    // Hub node: use stored vector
                    if hub_idx < index_data.vectors.len() {
                        full_vecs.push(index_data.vectors[hub_idx].clone());
                        hub_idx += 1;
                    } else {
                        return Err(VyaktiError::Storage(format!(
                            "Compact index corrupted: not enough hub vectors (expected {} hub nodes, got {} vectors)",
                            num_nodes - pruned_nodes.len(),
                            index_data.vectors.len()
                        )));
                    }
                }
            }
            full_vecs
        } else {
            // Normal mode: use vectors as-is
            index_data.vectors
        };

        // If graph topology is available, restore it before building
        // This is critical for compact mode to work correctly
        if let Some(ref topology) = index_data.graph_topology {
            // For HNSW backend, import topology and vectors directly
            if let Some(hnsw_backend) = backend.as_any_mut().downcast_mut::<vyakti_backend_hnsw::HnswBackend>() {
                // First, restore the vectors array directly (preserving node IDs)
                for vector in full_vectors.iter() {
                    hnsw_backend.graph_mut().add_vector(vector.clone());
                }

                // Then import the graph topology (edges and entry point)
                let internal_layers: Vec<std::collections::HashMap<usize, Vec<(usize, f32)>>> = topology
                    .layers
                    .iter()
                    .map(|layer| {
                        layer
                            .iter()
                            .map(|(node_id, edges)| {
                                let edge_tuples: Vec<(usize, f32)> = edges
                                    .iter()
                                    .map(|edge| (edge.target, edge.distance))
                                    .collect();
                                (*node_id, edge_tuples)
                            })
                            .collect()
                    })
                    .collect();

                hnsw_backend.graph_mut().import_topology(internal_layers, topology.entry_point)?;

                // Skip normal build since we already restored the graph
            } else {
                // Non-HNSW backend: fall back to normal build
                backend.build(&full_vectors, &index_data.metadata.backend_config).await?;
            }
        } else {
            // No topology: build backend normally (will create new graph structure)
            backend.build(&full_vectors, &index_data.metadata.backend_config).await?;
        }

        // Load document data into backend
        for doc in index_data.documents {
            backend.set_document_data(doc.id, doc.text, doc.metadata);
        }

        // If loading compact index with HNSW backend, restore compact mode
        if let Some(ref pruned_nodes) = index_data.pruned_nodes {
            if let Some(hnsw_backend) = backend.as_any_mut().downcast_mut::<vyakti_backend_hnsw::HnswBackend>() {
                // Restore compact mode by marking nodes as pruned
                for &node_id in pruned_nodes {
                    hnsw_backend.graph_mut().prune_node_embedding(node_id)?;
                }
                // Restore compact mode (sets up recomputation service WITHOUT re-pruning)
                hnsw_backend.restore_compact_mode(embedding_provider.clone())?;
            }
        }

        Ok(Self {
            backend: Arc::new(RwLock::new(backend)),
            embedding_provider,
            filter_engine: MetadataFilterEngine::new(),
        })
    }

    /// Search with text query
    ///
    /// # Arguments
    ///
    /// * `query` - Text query
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by score
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_filters(query, k, None).await
    }

    /// Search with text query and metadata filters
    ///
    /// # Arguments
    ///
    /// * `query` - Text query
    /// * `k` - Number of results to return
    /// * `filters` - Optional metadata filters to apply (AND logic)
    ///
    /// # Returns
    ///
    /// Vector of filtered search results sorted by score
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vyakti_core::VyaktiSearcher;
    /// use vyakti_common::{FilterOperator, FilterValue, MetadataFilters};
    /// use std::collections::HashMap;
    ///
    /// # async fn example(searcher: VyaktiSearcher) -> Result<(), Box<dyn std::error::Error>> {
    /// // Create metadata filters
    /// let mut filters = MetadataFilters::new();
    ///
    /// // Filter by category == "technology"
    /// let mut category_filter = HashMap::new();
    /// category_filter.insert(
    ///     FilterOperator::Eq,
    ///     FilterValue::String("technology".to_string()),
    /// );
    /// filters.insert("category".to_string(), category_filter);
    ///
    /// // Filter by year >= 2024
    /// let mut year_filter = HashMap::new();
    /// year_filter.insert(FilterOperator::Ge, FilterValue::Integer(2024));
    /// filters.insert("year".to_string(), year_filter);
    ///
    /// // Search with filters
    /// let results = searcher.search_with_filters(
    ///     "machine learning",
    ///     10,
    ///     Some(&filters)
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search_with_filters(
        &self,
        query: &str,
        k: usize,
        filters: Option<&MetadataFilters>,
    ) -> Result<Vec<SearchResult>> {
        // Embed the query
        let query_embedding = self.embedding_provider.embed(&[query.to_string()]).await?;

        if query_embedding.is_empty() {
            return Err(VyaktiError::Embedding("Failed to embed query".to_string()));
        }

        // Search with the backend
        let backend = self.backend.read().await;
        let results = backend.search(&query_embedding[0], k).await?;

        // Apply metadata filters if provided
        if let Some(filters) = filters {
            Ok(self.filter_engine.apply_filters(results, filters))
        } else {
            Ok(results)
        }
    }

    /// Search with a vector query
    ///
    /// # Arguments
    ///
    /// * `query_vector` - Query vector
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by score
    pub async fn search_by_vector(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_by_vector_with_filters(query_vector, k, None)
            .await
    }

    /// Search with a vector query and metadata filters
    ///
    /// # Arguments
    ///
    /// * `query_vector` - Query vector
    /// * `k` - Number of results to return
    /// * `filters` - Optional metadata filters to apply (AND logic)
    ///
    /// # Returns
    ///
    /// Vector of filtered search results sorted by score
    pub async fn search_by_vector_with_filters(
        &self,
        query_vector: &[f32],
        k: usize,
        filters: Option<&MetadataFilters>,
    ) -> Result<Vec<SearchResult>> {
        let backend = self.backend.read().await;
        let vec = query_vector.to_vec();
        let results = backend.search(&vec, k).await?;

        // Apply metadata filters if provided
        if let Some(filters) = filters {
            Ok(self.filter_engine.apply_filters(results, filters))
        } else {
            Ok(results)
        }
    }

    /// Get number of indexed documents
    pub async fn len(&self) -> usize {
        let backend = self.backend.read().await;
        backend.len()
    }

    /// Check if index is empty
    pub async fn is_empty(&self) -> bool {
        let backend = self.backend.read().await;
        backend.is_empty()
    }

    /// Get backend name
    pub async fn backend_name(&self) -> String {
        let backend = self.backend.read().await;
        backend.name().to_string()
    }

    /// Get embedding provider name
    pub fn embedding_provider_name(&self) -> &str {
        self.embedding_provider.name()
    }

    /// Get reference to the backend
    pub fn backend(&self) -> Arc<RwLock<Box<dyn Backend>>> {
        Arc::clone(&self.backend)
    }

    /// Get reference to the embedding provider
    pub fn embedding_provider(&self) -> Arc<dyn EmbeddingProvider> {
        Arc::clone(&self.embedding_provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use vyakti_common::{BackendConfig, Vector};

    // Mock Backend for testing
    struct MockBackend {
        vectors: Vec<Vector>,
    }

    #[async_trait]
    impl Backend for MockBackend {
        fn name(&self) -> &str {
            "mock-backend"
        }

        async fn build(&mut self, vectors: &[Vector], _config: &BackendConfig) -> Result<()> {
            self.vectors = vectors.to_vec();
            Ok(())
        }

        async fn search(&self, _query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
            let mut results = vec![];
            for i in 0..k.min(self.vectors.len()) {
                results.push(SearchResult {
                    id: i,
                    text: format!("Document {}", i),
                    score: 0.5 + (i as f32 * 0.1),
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

    // Mock EmbeddingProvider for testing
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
            "mock-embedding"
        }
    }

    #[tokio::test]
    async fn test_searcher_new() {
        let backend = Box::new(MockBackend {
            vectors: vec![vec![1.0, 2.0, 3.0]],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        assert_eq!(searcher.embedding_provider_name(), "mock-embedding");
        assert_eq!(searcher.backend_name().await, "mock-backend");
    }

    #[tokio::test]
    async fn test_searcher_search() {
        let backend = Box::new(MockBackend {
            vectors: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 3 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let results = searcher.search("test query", 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert_eq!(results[1].id, 1);
    }

    #[tokio::test]
    async fn test_searcher_search_by_vector() {
        let backend = Box::new(MockBackend {
            vectors: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 3 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let query_vector = vec![1.0, 0.0, 0.0];
        let results = searcher.search_by_vector(&query_vector, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
        assert_eq!(results[1].id, 1);
    }

    #[tokio::test]
    async fn test_searcher_search_k_larger_than_index() {
        let backend = Box::new(MockBackend {
            vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 2 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let results = searcher.search("test query", 10).await.unwrap();

        // Should return only 2 results (all available)
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_searcher_len() {
        let backend = Box::new(MockBackend {
            vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 2 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        assert_eq!(searcher.len().await, 3);
        assert!(!searcher.is_empty().await);
    }

    #[tokio::test]
    async fn test_searcher_empty() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 2 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        assert_eq!(searcher.len().await, 0);
        assert!(searcher.is_empty().await);
    }

    #[tokio::test]
    async fn test_searcher_backend_access() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 2 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let backend = searcher.backend();
        let backend_guard = backend.read().await;
        assert_eq!(backend_guard.name(), "mock-backend");
    }

    #[tokio::test]
    async fn test_searcher_embedding_provider_access() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let provider = searcher.embedding_provider();
        assert_eq!(provider.name(), "mock-embedding");
        assert_eq!(provider.dimension(), 128);
    }

    #[tokio::test]
    async fn test_searcher_results_have_data() {
        let backend = Box::new(MockBackend {
            vectors: vec![vec![1.0], vec![2.0], vec![3.0]],
        });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 1 });

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let results = searcher.search("query", 3).await.unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "Document 0");
        assert_eq!(results[1].text, "Document 1");
        assert_eq!(results[2].text, "Document 2");

        // Check scores are increasing
        assert!(results[0].score < results[1].score);
        assert!(results[1].score < results[2].score);
    }

    #[tokio::test]
    async fn test_searcher_embedding_failure() {
        // Mock embedding provider that returns empty vector
        struct FailingEmbeddingProvider;

        #[async_trait]
        impl EmbeddingProvider for FailingEmbeddingProvider {
            async fn embed(&self, _texts: &[String]) -> Result<Vec<Vector>> {
                Ok(vec![]) // Return empty vector
            }

            fn dimension(&self) -> usize {
                128
            }

            fn name(&self) -> &str {
                "failing"
            }
        }

        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(FailingEmbeddingProvider);

        let searcher = VyaktiSearcher::new(backend, embedding_provider);

        let result = searcher.search("query", 5).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("embed"));
    }
}
