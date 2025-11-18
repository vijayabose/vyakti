//! Embedding recomputation service for LEANN algorithm.
//!
//! This service provides on-demand recomputation of embeddings from document text
//! during search operations, enabling 97% storage savings by not storing all embeddings.

use crate::cache::EmbeddingCache;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use vyakti_common::{EmbeddingProvider, NodeId, Result, Vector, VyaktiError};

/// Statistics for recomputation operations.
#[derive(Debug, Clone, Default)]
pub struct RecomputationStats {
    /// Total number of recomputation requests
    pub total_requests: usize,
    /// Number of cache hits (embeddings served from cache)
    pub cache_hits: usize,
    /// Number of cache misses (embeddings recomputed)
    pub cache_misses: usize,
    /// Number of embeddings recomputed
    pub embeddings_recomputed: usize,
    /// Total time spent recomputing (microseconds)
    pub total_recompute_time_us: u64,
    /// Number of failed recomputation attempts
    pub failed_recomputations: usize,
}

impl RecomputationStats {
    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }

    /// Get average recomputation time in milliseconds.
    pub fn avg_recompute_time_ms(&self) -> f64 {
        if self.embeddings_recomputed == 0 {
            0.0
        } else {
            (self.total_recompute_time_us as f64 / 1000.0) / self.embeddings_recomputed as f64
        }
    }
}

/// Configuration for the recomputation service.
#[derive(Debug, Clone)]
pub struct RecomputationConfig {
    /// Size of the LRU cache for recomputed embeddings
    pub cache_size: usize,
    /// Maximum batch size for parallel recomputation
    pub max_batch_size: usize,
    /// Timeout for recomputation requests (milliseconds)
    pub timeout_ms: u64,
    /// Maximum number of retry attempts
    pub max_retries: usize,
}

impl Default for RecomputationConfig {
    fn default() -> Self {
        Self {
            cache_size: 10000,
            max_batch_size: 100,
            timeout_ms: 5000,
            max_retries: 2,
        }
    }
}

/// Embedding recomputation service.
///
/// Provides on-demand embedding computation with caching and batch processing.
pub struct EmbeddingRecomputationService {
    /// Embedding provider for computing embeddings
    provider: Arc<dyn EmbeddingProvider>,
    /// LRU cache for recomputed embeddings (keyed by node ID)
    cache: Arc<RwLock<EmbeddingCache<NodeId>>>,
    /// Configuration
    config: RecomputationConfig,
    /// Runtime statistics
    stats: Arc<RwLock<RecomputationStats>>,
}

impl EmbeddingRecomputationService {
    /// Create a new recomputation service.
    ///
    /// # Arguments
    ///
    /// * `provider` - Embedding provider for computing embeddings
    /// * `config` - Service configuration
    pub fn new(provider: Arc<dyn EmbeddingProvider>, config: RecomputationConfig) -> Self {
        let cache = Arc::new(RwLock::new(EmbeddingCache::new(config.cache_size)));

        info!(
            "EmbeddingRecomputationService initialized: cache_size={}, max_batch_size={}, timeout_ms={}",
            config.cache_size, config.max_batch_size, config.timeout_ms
        );

        Self {
            provider,
            cache,
            config,
            stats: Arc::new(RwLock::new(RecomputationStats::default())),
        }
    }

    /// Recompute a single embedding from text.
    ///
    /// This checks the cache first and only recomputes if not found.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Node ID for cache key
    /// * `text` - Document text to embed
    ///
    /// # Returns
    ///
    /// The embedding vector
    pub async fn recompute_single(
        &self,
        node_id: NodeId,
        text: String,
    ) -> Result<Vector> {
        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(embedding) = cache.get(&node_id) {
                debug!(node_id = node_id, "Cache hit for recomputation");
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                return Ok(embedding);
            }
        }

        // Cache miss - recompute
        debug!(node_id = node_id, "Cache miss, recomputing embedding");
        {
            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        }

        let start = Instant::now();

        // Recompute with timeout and retry
        let embedding = self.recompute_with_retry(&[text]).await?
            .into_iter()
            .next()
            .ok_or_else(|| VyaktiError::Embedding("No embedding returned".to_string()))?;

        let elapsed = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.embeddings_recomputed += 1;
            stats.total_recompute_time_us += elapsed.as_micros() as u64;
        }

        debug!(
            node_id = node_id,
            time_ms = elapsed.as_millis(),
            "Embedding recomputed successfully"
        );

        // Store in cache
        {
            let cache = self.cache.write().await;
            cache.insert(node_id, embedding.clone());
        }

        Ok(embedding)
    }

    /// Recompute embeddings for multiple nodes in batch.
    ///
    /// This is more efficient than calling `recompute_single` multiple times
    /// as it batches the embedding computation.
    ///
    /// # Arguments
    ///
    /// * `nodes` - List of (node_id, text) pairs
    ///
    /// # Returns
    ///
    /// HashMap mapping node IDs to embedding vectors
    pub async fn recompute_batch(
        &self,
        nodes: Vec<(NodeId, String)>,
    ) -> Result<HashMap<NodeId, Vector>> {
        if nodes.is_empty() {
            return Ok(HashMap::new());
        }

        let mut result = HashMap::new();
        let mut to_compute = Vec::new();

        // Check cache for each node
        {
            let cache = self.cache.read().await;
            let mut stats = self.stats.write().await;

            for (node_id, text) in nodes {
                stats.total_requests += 1;

                if let Some(embedding) = cache.get(&node_id) {
                    debug!(node_id = node_id, "Cache hit in batch recomputation");
                    stats.cache_hits += 1;
                    result.insert(node_id, embedding);
                } else {
                    debug!(node_id = node_id, "Cache miss in batch recomputation");
                    stats.cache_misses += 1;
                    to_compute.push((node_id, text));
                }
            }
        }

        if to_compute.is_empty() {
            return Ok(result);
        }

        info!(
            "Batch recomputing {} embeddings (cache hits: {}, misses: {})",
            to_compute.len(),
            result.len(),
            to_compute.len()
        );

        // Process in batches
        for chunk in to_compute.chunks(self.config.max_batch_size) {
            let start = Instant::now();

            let texts: Vec<String> = chunk.iter().map(|(_, text)| text.clone()).collect();
            let node_ids: Vec<NodeId> = chunk.iter().map(|(id, _)| *id).collect();

            // Recompute embeddings
            let embeddings = self.recompute_with_retry(&texts).await?;

            let elapsed = start.elapsed();

            // Update stats
            {
                let mut stats = self.stats.write().await;
                stats.embeddings_recomputed += embeddings.len();
                stats.total_recompute_time_us += elapsed.as_micros() as u64;
            }

            debug!(
                count = embeddings.len(),
                time_ms = elapsed.as_millis(),
                "Batch embeddings recomputed"
            );

            // Store in cache and result
            {
                let cache = self.cache.write().await;
                for (node_id, embedding) in node_ids.into_iter().zip(embeddings.into_iter()) {
                    cache.insert(node_id, embedding.clone());
                    result.insert(node_id, embedding);
                }
            }
        }

        Ok(result)
    }

    /// Recompute embeddings with retry logic.
    async fn recompute_with_retry(&self, texts: &[String]) -> Result<Vec<Vector>> {
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= self.config.max_retries {
            attempts += 1;

            match tokio::time::timeout(timeout, self.provider.embed(texts)).await {
                Ok(Ok(embeddings)) => return Ok(embeddings),
                Ok(Err(e)) => {
                    warn!(
                        attempt = attempts,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Recomputation attempt failed"
                    );
                    last_error = Some(e);
                }
                Err(_) => {
                    warn!(
                        attempt = attempts,
                        max_retries = self.config.max_retries,
                        timeout_ms = self.config.timeout_ms,
                        "Recomputation timed out"
                    );
                    last_error = Some(VyaktiError::Embedding(format!(
                        "Timeout after {}ms",
                        self.config.timeout_ms
                    )));
                }
            }

            // Exponential backoff before retry
            if attempts <= self.config.max_retries {
                let delay = Duration::from_millis(100 * 2u64.pow(attempts as u32 - 1));
                tokio::time::sleep(delay).await;
            }
        }

        // All retries failed
        {
            let mut stats = self.stats.write().await;
            stats.failed_recomputations += texts.len();
        }

        Err(last_error.unwrap_or_else(|| {
            VyaktiError::Embedding("Recomputation failed after retries".to_string())
        }))
    }

    /// Get current statistics.
    pub async fn stats(&self) -> RecomputationStats {
        self.stats.read().await.clone()
    }

    /// Clear the cache.
    pub async fn clear_cache(&self) {
        let cache = self.cache.write().await;
        cache.clear();
        info!("Recomputation cache cleared");
    }

    /// Get cache statistics.
    pub async fn cache_stats(&self) -> (usize, usize, f64) {
        let cache = self.cache.read().await;
        let size = cache.len();
        let hits = cache.hits();
        let misses = cache.misses();
        let hit_rate = cache.hit_rate();
        (size, hits + misses, hit_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock embedding provider for testing.
    struct MockProvider {
        dimension: usize,
        call_count: Arc<AtomicUsize>,
        fail_count: Arc<AtomicUsize>,
    }

    impl MockProvider {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                call_count: Arc::new(AtomicUsize::new(0)),
                fail_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_failures(dimension: usize, fail_count: usize) -> Self {
            Self {
                dimension,
                call_count: Arc::new(AtomicUsize::new(0)),
                fail_count: Arc::new(AtomicUsize::new(fail_count)),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            // Simulate failures
            let fails = self.fail_count.load(Ordering::SeqCst);
            if fails > 0 {
                self.fail_count.fetch_sub(1, Ordering::SeqCst);
                return Err(VyaktiError::Embedding("Simulated failure".to_string()));
            }

            // Generate mock embeddings
            let embeddings = texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    let mut vec = vec![0.0; self.dimension];
                    vec[0] = (i + text.len()) as f32;
                    vec
                })
                .collect();

            Ok(embeddings)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_recomputation_service_creation() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider, config);

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[tokio::test]
    async fn test_recompute_single() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        let embedding = service
            .recompute_single(1, "test document".to_string())
            .await
            .unwrap();

        assert_eq!(embedding.len(), 768);
        assert_eq!(provider.call_count(), 1);

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.embeddings_recomputed, 1);
    }

    #[tokio::test]
    async fn test_recompute_single_cache_hit() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        // First call - cache miss
        let embedding1 = service
            .recompute_single(1, "test document".to_string())
            .await
            .unwrap();

        // Second call - cache hit
        let embedding2 = service
            .recompute_single(1, "test document".to_string())
            .await
            .unwrap();

        assert_eq!(embedding1, embedding2);
        assert_eq!(provider.call_count(), 1); // Should only be called once

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.embeddings_recomputed, 1);
        assert_eq!(stats.cache_hit_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_recompute_batch() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        let nodes = vec![
            (1, "doc1".to_string()),
            (2, "doc2".to_string()),
            (3, "doc3".to_string()),
        ];

        let result = service.recompute_batch(nodes).await.unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.contains_key(&1));
        assert!(result.contains_key(&2));
        assert!(result.contains_key(&3));

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 3);
        assert_eq!(stats.embeddings_recomputed, 3);
    }

    #[tokio::test]
    async fn test_recompute_batch_with_cache() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        // First batch
        let nodes1 = vec![(1, "doc1".to_string()), (2, "doc2".to_string())];
        service.recompute_batch(nodes1).await.unwrap();

        // Second batch with overlap
        let nodes2 = vec![
            (1, "doc1".to_string()), // Cache hit
            (2, "doc2".to_string()), // Cache hit
            (3, "doc3".to_string()), // Cache miss
        ];
        let result = service.recompute_batch(nodes2).await.unwrap();

        assert_eq!(result.len(), 3);

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 5);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 3);
        assert_eq!(stats.embeddings_recomputed, 3);
    }

    #[tokio::test]
    async fn test_recompute_batch_empty() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider, config);

        let nodes: Vec<(NodeId, String)> = vec![];
        let result = service.recompute_batch(nodes).await.unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_recompute_batch_large() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig {
            max_batch_size: 50,
            ..Default::default()
        };
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        // Create 150 nodes (3 batches)
        let nodes: Vec<(NodeId, String)> = (0..150)
            .map(|i| (i as NodeId, format!("doc{}", i)))
            .collect();

        let result = service.recompute_batch(nodes).await.unwrap();

        assert_eq!(result.len(), 150);
        assert_eq!(provider.call_count(), 3); // 3 batches

        let stats = service.stats().await;
        assert_eq!(stats.embeddings_recomputed, 150);
    }

    #[tokio::test]
    async fn test_recompute_with_retry() {
        let provider = Arc::new(MockProvider::with_failures(768, 2)); // Fail first 2 attempts
        let config = RecomputationConfig {
            max_retries: 3,
            timeout_ms: 1000,
            ..Default::default()
        };
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        // Should succeed on 3rd attempt
        let embedding = service
            .recompute_single(1, "test".to_string())
            .await
            .unwrap();

        assert_eq!(embedding.len(), 768);
        assert_eq!(provider.call_count(), 3);
    }

    #[tokio::test]
    async fn test_recompute_retry_exhausted() {
        let provider = Arc::new(MockProvider::with_failures(768, 10)); // Always fail
        let config = RecomputationConfig {
            max_retries: 2,
            timeout_ms: 100,
            ..Default::default()
        };
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        let result = service.recompute_single(1, "test".to_string()).await;

        assert!(result.is_err());
        assert_eq!(provider.call_count(), 3); // 1 initial + 2 retries

        let stats = service.stats().await;
        assert_eq!(stats.failed_recomputations, 1);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider.clone(), config);

        // Add some items to cache
        service.recompute_single(1, "doc1".to_string()).await.unwrap();
        service.recompute_single(2, "doc2".to_string()).await.unwrap();

        let (size, _, _) = service.cache_stats().await;
        assert_eq!(size, 2);

        // Clear cache
        service.clear_cache().await;

        let (size, _, _) = service.cache_stats().await;
        assert_eq!(size, 0);

        // Next call should recompute
        service.recompute_single(1, "doc1".to_string()).await.unwrap();
        assert_eq!(provider.call_count(), 3); // 2 initial + 1 after clear
    }

    #[tokio::test]
    async fn test_stats_calculation() {
        let provider = Arc::new(MockProvider::new(768));
        let config = RecomputationConfig::default();
        let service = EmbeddingRecomputationService::new(provider, config);

        // Generate some activity
        service.recompute_single(1, "doc1".to_string()).await.unwrap();
        service.recompute_single(1, "doc1".to_string()).await.unwrap(); // Cache hit
        service.recompute_single(2, "doc2".to_string()).await.unwrap();

        let stats = service.stats().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.embeddings_recomputed, 2);
        assert_eq!(stats.cache_hit_rate(), 1.0 / 3.0);
        assert!(stats.avg_recompute_time_ms() > 0.0);
    }
}
