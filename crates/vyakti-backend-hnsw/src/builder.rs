//! HNSW backend implementation of the Backend trait.

use async_trait::async_trait;
use std::sync::Arc;
use vyakti_common::{Backend, BackendConfig, DistanceMetric, EmbeddingProvider, Result, SearchResult, Vector};
use vyakti_embedding::EmbeddingRecomputationService;

use crate::graph::{HnswConfig, HnswGraph};
use crate::pruning::{GraphPruner, PruningConfig, PruningStats};
use crate::searcher::HnswSearcher;

/// HNSW backend implementation
pub struct HnswBackend {
    graph: HnswGraph,
    config: BackendConfig,
    /// Recomputation service for compact mode searches
    recomputation_service: Option<Arc<EmbeddingRecomputationService>>,
}

impl HnswBackend {
    /// Create a new HNSW backend with default configuration
    pub fn new() -> Self {
        Self::with_config(BackendConfig::default())
    }

    /// Create a new HNSW backend with custom configuration
    pub fn with_config(config: BackendConfig) -> Self {
        let hnsw_config = HnswConfig {
            max_connections: config.graph_degree,
            max_connections_0: config.graph_degree * 2,
            ml: 1.0 / (config.graph_degree as f64).ln(),
            ef_construction: config.build_complexity,
            metric: DistanceMetric::Cosine,
        };

        Self {
            graph: HnswGraph::with_config(hnsw_config),
            config,
            recomputation_service: None,
        }
    }

    /// Create with specific distance metric
    pub fn with_metric(config: BackendConfig, metric: DistanceMetric) -> Self {
        let hnsw_config = HnswConfig {
            max_connections: config.graph_degree,
            max_connections_0: config.graph_degree * 2,
            ml: 1.0 / (config.graph_degree as f64).ln(),
            ef_construction: config.build_complexity,
            metric,
        };

        Self {
            graph: HnswGraph::with_config(hnsw_config),
            config,
            recomputation_service: None,
        }
    }

    /// Enable compact mode by pruning embeddings for non-hub nodes
    ///
    /// This implements the LEANN algorithm: identifies hub nodes (top 5% by degree)
    /// and prunes embeddings for the remaining 95% of nodes, achieving 95%+ storage savings.
    ///
    /// # Arguments
    ///
    /// * `pruning_config` - Optional pruning configuration (uses default if None)
    /// * `embedding_provider` - Provider to use for recomputing embeddings during search
    ///
    /// # Returns
    ///
    /// Pruning statistics showing storage savings
    pub fn enable_compact_mode(
        &mut self,
        pruning_config: Option<PruningConfig>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Result<PruningStats> {
        let pruner = GraphPruner::new(pruning_config.unwrap_or_default());
        let stats = pruner.prune_embeddings(&self.graph)?;

        // Create recomputation service for compact mode searches
        use vyakti_embedding::RecomputationConfig;
        let recomputation_service =
            EmbeddingRecomputationService::new(embedding_provider, RecomputationConfig::default());
        self.recomputation_service = Some(Arc::new(recomputation_service));

        Ok(stats)
    }

    /// Restore compact mode after loading from disk
    ///
    /// This method sets up the recomputation service without re-pruning the graph.
    /// Used when loading a compact index from disk where nodes have already been pruned.
    ///
    /// # Arguments
    ///
    /// * `embedding_provider` - Provider to use for recomputing embeddings during search
    pub fn restore_compact_mode(
        &mut self,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Result<()> {
        // Mark graph as being in compact mode
        self.graph.enable_compact_mode();

        // Create recomputation service for compact mode searches
        use vyakti_embedding::RecomputationConfig;
        let recomputation_service =
            EmbeddingRecomputationService::new(embedding_provider, RecomputationConfig::default());
        self.recomputation_service = Some(Arc::new(recomputation_service));

        Ok(())
    }

    /// Check if the backend is in compact mode
    pub fn is_compact_mode(&self) -> bool {
        self.graph.is_compact_mode()
    }

    /// Get reference to the underlying graph
    pub fn graph(&self) -> &HnswGraph {
        &self.graph
    }

    /// Get mutable reference to the underlying graph
    pub fn graph_mut(&mut self) -> &mut HnswGraph {
        &mut self.graph
    }

    /// Get number of pruned nodes
    pub fn num_pruned_nodes(&self) -> usize {
        self.graph.num_pruned_nodes()
    }
}

impl Default for HnswBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for HnswBackend {
    fn name(&self) -> &str {
        "hnsw"
    }

    async fn build(&mut self, vectors: &[Vector], _config: &BackendConfig) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Add all vectors to the graph
        for vector in vectors {
            let node_id = self.graph.add_vector(vector.clone());
            let level = self.graph.random_level();

            // Set entry point if this is the first node or if it has higher level
            if let Some((_, entry_level)) = self.graph.entry_point() {
                if level > entry_level {
                    self.graph.set_entry_point(node_id, level);
                }
            } else {
                self.graph.set_entry_point(node_id, level);
            }

            // For each layer from 0 to assigned level, connect to neighbors
            for layer in 0..=level {
                // Find nearest neighbors at this layer using existing graph
                if node_id > 0 {
                    // Simple strategy: connect to a few previous nodes
                    let max_connections = if layer == 0 {
                        self.graph.config().max_connections_0
                    } else {
                        self.graph.config().max_connections
                    };

                    let start = node_id.saturating_sub(max_connections);

                    for prev_id in start..node_id {
                        if let Some(prev_vec) = self.graph.get_vector(prev_id) {
                            let distance = self.graph.distance(vector, &prev_vec);
                            self.graph.add_edge(layer, node_id, prev_id, distance)?;
                            self.graph.add_edge(layer, prev_id, node_id, distance)?;
                        }
                    }

                    // Prune neighbors to maintain max connections
                    self.graph.prune_neighbors(layer, node_id)?;
                }
            }
        }

        Ok(())
    }

    fn set_document_data(
        &mut self,
        node_id: usize,
        text: String,
        metadata: std::collections::HashMap<String, serde_json::Value>,
    ) {
        self.graph.set_document_data(node_id, text, metadata);
    }

    async fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        let ef = self.config.search_complexity.max(k);

        // Check if we're in compact mode and use appropriate searcher method
        if let Some(recomp_service) = &self.recomputation_service {
            // Compact mode: use async search with recomputation service
            let searcher = HnswSearcher::with_recomputation(&self.graph, recomp_service.clone());
            searcher.search_async(query, k, ef).await
        } else {
            // Normal mode: use standard synchronous search
            let searcher = HnswSearcher::new(&self.graph);
            searcher.search(query, k, ef)
        }
    }

    fn len(&self) -> usize {
        self.graph.len()
    }

    fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use vyakti_common::{EmbeddingProvider, Vector};

    // Mock embedding provider for testing
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

    #[tokio::test]
    async fn test_hnsw_backend_new() {
        let backend = HnswBackend::new();
        assert_eq!(backend.name(), "hnsw");
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_backend_with_config() {
        let config = BackendConfig {
            graph_degree: 16,
            build_complexity: 32,
            search_complexity: 16,
        };
        let backend = HnswBackend::with_config(config);
        assert_eq!(backend.name(), "hnsw");
        assert_eq!(backend.graph.config().max_connections, 16);
        assert_eq!(backend.graph.config().max_connections_0, 32);
    }

    #[tokio::test]
    async fn test_hnsw_backend_with_metric() {
        let config = BackendConfig::default();
        let backend = HnswBackend::with_metric(config, DistanceMetric::Euclidean);
        assert_eq!(backend.graph.config().metric, DistanceMetric::Euclidean);
    }

    #[tokio::test]
    async fn test_hnsw_backend_build_empty() {
        let mut backend = HnswBackend::new();
        let vectors: Vec<Vector> = vec![];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();
        assert_eq!(backend.len(), 0);
        assert!(backend.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_backend_build_single_vector() {
        let mut backend = HnswBackend::new();
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();
        assert_eq!(backend.len(), 1);
        assert!(!backend.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_backend_build_multiple_vectors() {
        let mut backend = HnswBackend::new();
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
        ];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();
        assert_eq!(backend.len(), 5);
    }

    #[tokio::test]
    async fn test_hnsw_backend_search_empty() {
        let backend = HnswBackend::new();
        let query = vec![1.0, 0.0, 0.0];

        let results = backend.search(&query, 5).await.unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_hnsw_backend_search_single_vector() {
        let mut backend = HnswBackend::new();
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = backend.search(&query, 1).await.unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[tokio::test]
    async fn test_hnsw_backend_search_multiple_vectors() {
        let mut backend = HnswBackend::new();
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
            vec![0.0, 0.0, 1.0],
        ];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = backend.search(&query, 2).await.unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // Results should be sorted by score
        if results.len() == 2 {
            assert!(results[0].score <= results[1].score);
        }
    }

    #[tokio::test]
    async fn test_hnsw_backend_search_k_larger_than_index() {
        let mut backend = HnswBackend::new();
        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let config = BackendConfig::default();

        backend.build(&vectors, &config).await.unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = backend.search(&query, 10).await.unwrap();

        // Should return all vectors (2)
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_hnsw_backend_default() {
        let backend = HnswBackend::default();
        assert_eq!(backend.name(), "hnsw");
        assert!(backend.is_empty());
    }

    #[tokio::test]
    async fn test_hnsw_backend_integration() {
        // Create a larger index and verify search quality
        let mut backend = HnswBackend::new();

        // Create vectors in 3 clusters
        let mut vectors = Vec::new();

        // Cluster 1: around [1, 0, 0]
        for i in 0..10 {
            let noise = i as f32 * 0.01;
            vectors.push(vec![1.0 - noise, noise, 0.0]);
        }

        // Cluster 2: around [0, 1, 0]
        for i in 0..10 {
            let noise = i as f32 * 0.01;
            vectors.push(vec![noise, 1.0 - noise, 0.0]);
        }

        // Cluster 3: around [0, 0, 1]
        for i in 0..10 {
            let noise = i as f32 * 0.01;
            vectors.push(vec![0.0, noise, 1.0 - noise]);
        }

        let config = BackendConfig::default();
        backend.build(&vectors, &config).await.unwrap();

        // Search for vector in cluster 1
        let query = vec![1.0, 0.0, 0.0];
        let results = backend.search(&query, 5).await.unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted
        for i in 1..results.len() {
            assert!(results[i - 1].score <= results[i].score);
        }
    }

    #[tokio::test]
    async fn test_hnsw_backend_compact_mode() {
        let mut backend = HnswBackend::new();

        // Build a graph with many nodes
        let mut vectors = Vec::new();
        for i in 0..100 {
            let val = i as f32 / 100.0;
            vectors.push(vec![val, 1.0 - val, 0.0]);
        }

        let config = BackendConfig::default();
        backend.build(&vectors, &config).await.unwrap();

        // Set document data for each node (required for recomputation)
        for i in 0..100 {
            backend.set_document_data(i, format!("Document {}", i), std::collections::HashMap::new());
        }

        // Initially not in compact mode
        assert!(!backend.is_compact_mode());
        assert_eq!(backend.num_pruned_nodes(), 0);

        // Create mock embedding provider
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 3 });

        // Enable compact mode
        let stats = backend.enable_compact_mode(None, embedding_provider).unwrap();

        // Should now be in compact mode
        assert!(backend.is_compact_mode());
        assert!(backend.num_pruned_nodes() > 0);

        // Should have significant storage savings
        // Note: With 100 nodes, we get ~65-76% savings. Larger graphs (1000+ nodes) achieve 95%+ savings
        // due to stronger hub-and-spoke patterns.
        assert!(stats.savings_percent >= 65.0, "Expected >=65% savings, got {}%", stats.savings_percent);

        // Should keep only ~5% of embeddings (for small graphs, may retain more due to variability)
        assert!(stats.retention_rate() <= 0.35, "Expected <=35% retention, got {:.2}%", stats.retention_rate() * 100.0);

        // Verify recomputation service is set
        assert!(backend.recomputation_service.is_some());
    }

    #[tokio::test]
    async fn test_hnsw_backend_compact_mode_custom_config() {
        let mut backend = HnswBackend::new();

        let mut vectors = Vec::new();
        for i in 0..100 {
            vectors.push(vec![i as f32; 3]);
        }

        let config = BackendConfig::default();
        backend.build(&vectors, &config).await.unwrap();

        // Create mock embedding provider
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 3 });

        // Use custom pruning config (keep top 10% instead of 5%)
        let pruning_config = crate::pruning::PruningConfig {
            degree_threshold_percentile: 0.90,
            ..Default::default()
        };

        let stats = backend
            .enable_compact_mode(Some(pruning_config), embedding_provider)
            .unwrap();

        assert!(backend.is_compact_mode());
        // With 90th percentile, should retain ~10% of nodes (may be higher for small graphs)
        assert!(stats.retention_rate() < 0.35, "Expected <35% retention, got {:.2}%", stats.retention_rate() * 100.0);
        assert!(stats.retention_rate() > 0.05, "Expected >5% retention, got {:.2}%", stats.retention_rate() * 100.0);
    }
}
