//! Hybrid search combining semantic vector search with keyword (BM25) search.
//!
//! This module provides fusion strategies to combine results from both
//! vector-based semantic search and BM25-based keyword search for improved
//! code search accuracy.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use vyakti_common::{Backend, EmbeddingProvider, Result, SearchResult, VyaktiError};
use vyakti_keyword::{KeywordConfig, KeywordResult, KeywordSearcher};

/// Fusion strategy for combining vector and keyword search results
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    /// score(doc) = Σ 1/(k + rank_in_result_set)
    /// Simple, effective, no parameter tuning needed
    RRF { k: usize },

    /// Weighted combination of normalized scores
    /// score(doc) = α * norm(bm25_score) + (1-α) * norm(vector_score)
    /// Requires score normalization, configurable weight α
    Weighted { alpha: f32 },

    /// Cascade: try keyword first, fallback to vector
    /// if keyword_match_count > threshold: use keyword results
    /// else: use vector results
    Cascade { threshold: usize },

    /// Vector-only mode (disable hybrid)
    VectorOnly,

    /// Keyword-only mode
    KeywordOnly,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        // RRF with k=60 is industry-proven default
        Self::RRF { k: 60 }
    }
}

/// Hybrid searcher that combines vector and keyword search
pub struct HybridSearcher {
    /// Vector search backend
    vector_backend: Arc<RwLock<Box<dyn Backend>>>,
    /// Embedding provider for vector search
    embedding_provider: Arc<dyn EmbeddingProvider>,
    /// Optional keyword searcher (None if hybrid disabled)
    keyword_searcher: Option<KeywordSearcher>,
    /// Fusion strategy
    strategy: FusionStrategy,
    /// Documents for text retrieval
    documents: Vec<(String, HashMap<String, serde_json::Value>)>,
}

impl HybridSearcher {
    /// Create a new hybrid searcher
    ///
    /// # Arguments
    ///
    /// * `vector_backend` - Backend for vector search
    /// * `embedding_provider` - Provider for computing embeddings
    /// * `keyword_searcher` - Optional keyword searcher (None to disable hybrid)
    /// * `strategy` - Fusion strategy to use
    /// * `documents` - Document texts and metadata for retrieval
    pub fn new(
        vector_backend: Box<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        keyword_searcher: Option<KeywordSearcher>,
        strategy: FusionStrategy,
        documents: Vec<(String, HashMap<String, serde_json::Value>)>,
    ) -> Self {
        info!(
            "Creating hybrid searcher (strategy: {:?}, keyword_enabled: {})",
            strategy,
            keyword_searcher.is_some()
        );

        Self {
            vector_backend: Arc::new(RwLock::new(vector_backend)),
            embedding_provider,
            keyword_searcher,
            strategy,
            documents,
        }
    }

    /// Load a hybrid index from disk
    ///
    /// # Arguments
    ///
    /// * `index_path` - Path to the index directory
    /// * `vector_backend` - Backend for vector search
    /// * `embedding_provider` - Provider for computing embeddings
    /// * `strategy` - Fusion strategy to use
    /// * `documents` - Document texts and metadata
    pub fn load<P: AsRef<Path>>(
        index_path: P,
        vector_backend: Box<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        strategy: FusionStrategy,
        documents: Vec<(String, HashMap<String, serde_json::Value>)>,
    ) -> Result<Self> {
        let index_path = index_path.as_ref();

        // Try to load keyword index if it exists
        // The keyword index directory contains both doc_map.bin and tantivy/
        let keyword_path = index_path.join("keyword");
        let keyword_searcher = if keyword_path.exists() {
            info!("Loading keyword index from: {}", keyword_path.display());
            Some(
                KeywordSearcher::load(&keyword_path, KeywordConfig::default())
                    .map_err(|e| VyaktiError::Storage(format!("Failed to load keyword index: {}", e)))?,
            )
        } else {
            debug!("No keyword index found, using vector-only mode");
            None
        };

        Ok(Self::new(
            vector_backend,
            embedding_provider,
            keyword_searcher,
            strategy,
            documents,
        ))
    }

    /// Search using the configured fusion strategy
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `top_k` - Maximum number of results to return
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        match &self.strategy {
            FusionStrategy::VectorOnly => self.vector_only_search(query, top_k).await,
            FusionStrategy::KeywordOnly => self.keyword_only_search(query, top_k).await,
            FusionStrategy::RRF { k } => self.rrf_search(query, top_k, *k).await,
            FusionStrategy::Weighted { alpha } => self.weighted_search(query, top_k, *alpha).await,
            FusionStrategy::Cascade { threshold } => self.cascade_search(query, top_k, *threshold).await,
        }
    }

    /// Vector-only search (standard semantic search)
    async fn vector_only_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing vector-only search for: '{}'", query);

        // Compute query embedding (embed expects &[String])
        let query_vec = vec![query.to_string()];
        let query_embeddings = self.embedding_provider.embed(&query_vec).await?;
        let query_embedding = query_embeddings
            .first()
            .ok_or_else(|| VyaktiError::Embedding("Failed to compute query embedding".to_string()))?;

        // Search using vector backend
        let backend = self.vector_backend.read().await;
        let results = backend.search(query_embedding, top_k).await?;

        Ok(results)
    }

    /// Keyword-only search (BM25)
    async fn keyword_only_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing keyword-only search for: '{}'", query);

        let keyword_searcher = self
            .keyword_searcher
            .as_ref()
            .ok_or_else(|| VyaktiError::Config("Keyword search not enabled".to_string()))?;

        // Always enable highlighting in keyword search (lightweight, can be ignored in display)
        let keyword_results = keyword_searcher
            .search_with_highlighting(query, top_k, true)
            .map_err(|e| VyaktiError::Backend(format!("Keyword search failed: {}", e)))?;

        Ok(self.keyword_results_to_search_results(&keyword_results))
    }

    /// Reciprocal Rank Fusion (RRF) search
    async fn rrf_search(&self, query: &str, top_k: usize, k: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing RRF fusion search (k={}) for: '{}'", k, query);

        // Get results from both indexes (fetch 2x to ensure good coverage)
        let query_vec = vec![query.to_string()];
        let query_embeddings = self.embedding_provider.embed(&query_vec).await?;
        let query_embedding = query_embeddings
            .first()
            .ok_or_else(|| VyaktiError::Embedding("Failed to compute query embedding".to_string()))?;

        let backend = self.vector_backend.read().await;
        let vector_results = backend.search(query_embedding, top_k * 2).await?;

        let keyword_results = if let Some(ref keyword_searcher) = self.keyword_searcher {
            keyword_searcher
                .search_with_highlighting(query, top_k * 2, true)
                .map_err(|e| VyaktiError::Backend(format!("Keyword search failed: {}", e)))?
        } else {
            vec![]
        };

        // Build RRF score map
        let mut rrf_scores: HashMap<usize, f32> = HashMap::new();

        // Add vector scores: RRF = 1/(k + rank)
        for (rank, result) in vector_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank + 1) as f32;
            *rrf_scores.entry(result.id).or_insert(0.0) += rrf_score;
        }

        // Add keyword scores
        for (rank, result) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank + 1) as f32;
            *rrf_scores.entry(result.node_id).or_insert(0.0) += rrf_score;
        }

        // Sort by combined RRF score
        let mut combined: Vec<_> = rrf_scores.into_iter().collect();
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to SearchResults using existing data from vector_results or documents
        let mut final_results = Vec::new();
        for (node_id, rrf_score) in combined.into_iter().take(top_k) {
            // Try to find the result in vector_results first (has full data)
            if let Some(result) = vector_results.iter().find(|r| r.id == node_id) {
                let mut new_result = result.clone();
                new_result.score = rrf_score;
                final_results.push(new_result);
            } else if node_id < self.documents.len() {
                // Fall back to documents array
                let (text, metadata) = &self.documents[node_id];
                final_results.push(SearchResult {
                    id: node_id,
                    text: text.clone(),
                    score: rrf_score,
                    metadata: metadata.clone(),
                });
            }
        }

        debug!("RRF fusion returned {} results", final_results.len());

        Ok(final_results)
    }

    /// Weighted fusion search
    async fn weighted_search(&self, query: &str, top_k: usize, alpha: f32) -> Result<Vec<SearchResult>> {
        debug!(
            "Performing weighted fusion search (alpha={}) for: '{}'",
            alpha, query
        );

        // Get results from both indexes
        let query_vec = vec![query.to_string()];
        let query_embeddings = self.embedding_provider.embed(&query_vec).await?;
        let query_embedding = query_embeddings
            .first()
            .ok_or_else(|| VyaktiError::Embedding("Failed to compute query embedding".to_string()))?;

        let backend = self.vector_backend.read().await;
        let vector_results = backend.search(query_embedding, top_k * 2).await?;

        let keyword_results = if let Some(ref keyword_searcher) = self.keyword_searcher {
            keyword_searcher
                .search_with_highlighting(query, top_k * 2, true)
                .map_err(|e| VyaktiError::Backend(format!("Keyword search failed: {}", e)))?
        } else {
            vec![]
        };

        // Normalize scores to [0, 1]
        let vector_scores = Self::normalize_scores(
            &vector_results.iter().map(|r| (r.id, r.score)).collect::<Vec<_>>(),
        );
        let keyword_scores = Self::normalize_scores(
            &keyword_results
                .iter()
                .map(|r| (r.node_id, r.score))
                .collect::<Vec<_>>(),
        );

        // Combine with weighted sum: alpha * keyword + (1-alpha) * vector
        let mut combined_scores: HashMap<usize, f32> = HashMap::new();

        for (id, score) in vector_scores {
            *combined_scores.entry(id).or_insert(0.0) += (1.0 - alpha) * score;
        }

        for (id, score) in keyword_scores {
            *combined_scores.entry(id).or_insert(0.0) += alpha * score;
        }

        // Sort and convert to results
        let mut combined: Vec<_> = combined_scores.into_iter().collect();
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut final_results = Vec::new();
        for (node_id, score) in combined.into_iter().take(top_k) {
            // Try to find the result in vector_results first
            if let Some(result) = vector_results.iter().find(|r| r.id == node_id) {
                let mut new_result = result.clone();
                new_result.score = score;
                final_results.push(new_result);
            } else if node_id < self.documents.len() {
                let (text, metadata) = &self.documents[node_id];
                final_results.push(SearchResult {
                    id: node_id,
                    text: text.clone(),
                    score,
                    metadata: metadata.clone(),
                });
            }
        }

        debug!("Weighted fusion returned {} results", final_results.len());

        Ok(final_results)
    }

    /// Cascade search: keyword first, fallback to vector
    async fn cascade_search(&self, query: &str, top_k: usize, threshold: usize) -> Result<Vec<SearchResult>> {
        debug!(
            "Performing cascade search (threshold={}) for: '{}'",
            threshold, query
        );

        // Try keyword search first
        if let Some(ref keyword_searcher) = self.keyword_searcher {
            let keyword_results = keyword_searcher
                .search_with_highlighting(query, top_k, true)
                .map_err(|e| VyaktiError::Backend(format!("Keyword search failed: {}", e)))?;

            if keyword_results.len() >= threshold {
                debug!("Cascade: using keyword results ({} >= {})", keyword_results.len(), threshold);
                return Ok(self.keyword_results_to_search_results(&keyword_results));
            }
        }

        // Fallback to vector search
        debug!("Cascade: falling back to vector search");
        self.vector_only_search(query, top_k).await
    }

    /// Convert keyword results to SearchResults
    /// If enable_highlights is true, includes highlight snippets in metadata
    fn keyword_results_to_search_results(&self, keyword_results: &[KeywordResult]) -> Vec<SearchResult> {
        keyword_results
            .iter()
            .filter_map(|kr| {
                if kr.node_id < self.documents.len() {
                    let (text, mut metadata) = self.documents[kr.node_id].clone();

                    // Store highlights in metadata if available
                    if !kr.highlights.is_empty() {
                        let highlights_json: Vec<serde_json::Value> = kr
                            .highlights
                            .iter()
                            .map(|h| {
                                serde_json::json!({
                                    "field": h.field,
                                    "fragment": h.fragment,
                                    "positions": h.positions,
                                })
                            })
                            .collect();
                        metadata.insert("_highlights".to_string(), serde_json::Value::Array(highlights_json));
                    }

                    Some(SearchResult {
                        id: kr.node_id,
                        text,
                        score: kr.score,
                        metadata,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Normalize scores to [0, 1] range using min-max normalization
    fn normalize_scores(scores: &[(usize, f32)]) -> Vec<(usize, f32)> {
        if scores.is_empty() {
            return vec![];
        }

        let min_score = scores.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
        let max_score = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);

        let range = max_score - min_score;
        if range == 0.0 {
            // All scores are the same
            scores.iter().map(|(id, _)| (*id, 1.0)).collect()
        } else {
            scores
                .iter()
                .map(|(id, score)| (*id, (score - min_score) / range))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        matches!(strategy, FusionStrategy::RRF { k: 60 });
    }

    #[test]
    fn test_normalize_scores() {
        let scores = vec![(0, 10.0), (1, 20.0), (2, 30.0)];
        let normalized = HybridSearcher::normalize_scores(&scores);

        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0].1, 0.0); // min
        assert_eq!(normalized[1].1, 0.5); // mid
        assert_eq!(normalized[2].1, 1.0); // max
    }

    #[test]
    fn test_normalize_scores_same_values() {
        let scores = vec![(0, 5.0), (1, 5.0), (2, 5.0)];
        let normalized = HybridSearcher::normalize_scores(&scores);

        assert_eq!(normalized.len(), 3);
        // All should be normalized to 1.0
        assert!(normalized.iter().all(|(_, s)| *s == 1.0));
    }

    #[test]
    fn test_normalize_scores_empty() {
        let scores = vec![];
        let normalized = HybridSearcher::normalize_scores(&scores);
        assert_eq!(normalized.len(), 0);
    }
}
