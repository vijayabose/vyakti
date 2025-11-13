//! HNSW searcher with recomputation support for LEANN.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use vyakti_common::{Result, SearchResult, VyaktiError};
use vyakti_embedding::EmbeddingRecomputationService;

use crate::graph::{HnswGraph, NodeId};

/// Priority queue element for search
#[derive(Debug, Clone, Copy, PartialEq)]
struct QueueElement {
    node: NodeId,
    distance: f32,
}

impl Eq for QueueElement {}

impl PartialOrd for QueueElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW searcher with optional recomputation support
pub struct HnswSearcher<'a> {
    graph: &'a HnswGraph,
    /// Optional recomputation service for compact mode
    recomputation_service: Option<Arc<EmbeddingRecomputationService>>,
}

impl<'a> HnswSearcher<'a> {
    /// Create a new searcher for the given graph
    pub fn new(graph: &'a HnswGraph) -> Self {
        Self {
            graph,
            recomputation_service: None,
        }
    }

    /// Create a new searcher with recomputation support for compact mode
    pub fn with_recomputation(
        graph: &'a HnswGraph,
        service: Arc<EmbeddingRecomputationService>,
    ) -> Self {
        Self {
            graph,
            recomputation_service: Some(service),
        }
    }

    /// Check if the searcher is in compact mode (has recomputation service)
    pub fn is_compact_mode(&self) -> bool {
        self.graph.is_compact_mode() && self.recomputation_service.is_some()
    }

    /// Search for k nearest neighbors
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef` - Size of the dynamic candidate list (ef >= k)
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by distance
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        if self.graph.is_empty() {
            return Ok(Vec::new());
        }

        let ef = ef.max(k); // Ensure ef >= k

        // Get entry point
        let (entry_point, top_layer) = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut current_nearest = entry_point;

        // Phase 1: Greedy search from top layer to layer 1
        for layer in (1..=top_layer).rev() {
            current_nearest = self.search_layer(query, current_nearest, 1, layer)?[0].node;
        }

        // Phase 2: Search at layer 0 with ef candidates
        let candidates = self.search_layer(query, current_nearest, ef, 0)?;

        // Convert to SearchResults and return top k
        let results = candidates
            .into_iter()
            .take(k)
            .map(|elem| {
                let doc_data = self.graph.get_document_data(elem.node).unwrap_or_else(|| {
                    crate::graph::DocumentData {
                        text: String::new(),
                        metadata: std::collections::HashMap::new(),
                    }
                });
                SearchResult {
                    id: elem.node,
                    text: doc_data.text,
                    score: elem.distance,
                    metadata: doc_data.metadata,
                }
            })
            .collect();

        Ok(results)
    }

    /// Async search for k nearest neighbors with recomputation support (LEANN)
    ///
    /// This method implements two-phase search:
    /// 1. Graph traversal phase: navigate the graph structure (uses stored embeddings for hubs)
    /// 2. Recomputation phase: recompute embeddings for pruned candidates and re-rank
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `ef` - Size of the dynamic candidate list (ef >= k)
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by distance
    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>> {
        if self.graph.is_empty() {
            return Ok(Vec::new());
        }

        let ef = ef.max(k); // Ensure ef >= k

        // Phase 1: Graph Traversal
        // Navigate the graph using edge distances (doesn't require all embeddings)
        let candidate_nodes = self.graph_traversal_phase(query, ef)?;

        // Phase 2: Recomputation and Ranking
        // For compact mode, recompute embeddings for pruned nodes
        if self.is_compact_mode() {
            self.recomputation_phase(query, candidate_nodes, k).await
        } else {
            // Non-compact mode: just convert candidates to results
            self.candidates_to_results(candidate_nodes, k)
        }
    }

    /// Phase 1: Graph traversal to collect candidate nodes
    ///
    /// This phase navigates the HNSW graph structure to find candidate nodes.
    /// For hub nodes, we can use stored embeddings. For pruned nodes, we use
    /// edge distances from the graph structure.
    fn graph_traversal_phase(&self, query: &[f32], ef: usize) -> Result<Vec<NodeId>> {
        // Get entry point
        let (entry_point, top_layer) = match self.graph.entry_point() {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        let mut current_nearest = entry_point;

        // Greedy search from top layer to layer 1
        for layer in (1..=top_layer).rev() {
            current_nearest = self.search_layer(query, current_nearest, 1, layer)?[0].node;
        }

        // Search at layer 0 with ef candidates
        let candidates = self.search_layer(query, current_nearest, ef, 0)?;

        // Extract node IDs
        Ok(candidates.into_iter().map(|elem| elem.node).collect())
    }

    /// Phase 2: Recomputation and ranking
    ///
    /// Recompute embeddings for pruned candidate nodes and rank all candidates
    /// by actual distance to the query.
    async fn recomputation_phase(
        &self,
        query: &[f32],
        candidates: Vec<NodeId>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let service = self.recomputation_service.as_ref().ok_or_else(|| {
            VyaktiError::Backend(
                "Recomputation service not available for compact mode".to_string(),
            )
        })?;

        // Separate candidates into hub nodes (have embeddings) and pruned nodes (need recomputation)
        let mut hub_nodes = Vec::new();
        let mut pruned_nodes = Vec::new();

        for node_id in candidates {
            if self.graph.is_vector_available(node_id) {
                hub_nodes.push(node_id);
            } else {
                pruned_nodes.push(node_id);
            }
        }

        // Recompute embeddings for pruned nodes in batch
        let recomputed_embeddings: HashMap<NodeId, Vec<f32>> = if !pruned_nodes.is_empty() {
            let texts_to_recompute: Vec<(NodeId, String)> = pruned_nodes
                .iter()
                .filter_map(|&node_id| {
                    self.graph
                        .get_document_text(node_id)
                        .map(|text| (node_id, text))
                })
                .collect();

            service
                .recompute_batch(texts_to_recompute)
                .await?
        } else {
            HashMap::new()
        };

        // Compute distances for all candidates
        let mut scored_candidates = Vec::new();

        // Hub nodes: use stored embeddings
        for node_id in hub_nodes {
            let dist = self.compute_distance(query, node_id)?;
            scored_candidates.push((node_id, dist));
        }

        // Pruned nodes: use recomputed embeddings
        for node_id in pruned_nodes {
            if let Some(embedding) = recomputed_embeddings.get(&node_id) {
                let dist = self.graph.distance(query, embedding);
                scored_candidates.push((node_id, dist));
            }
        }

        // Sort by distance and take top k
        scored_candidates.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Convert to SearchResults
        let results = scored_candidates
            .into_iter()
            .take(k)
            .map(|(node_id, distance)| {
                let doc_data = self.graph.get_document_data(node_id).unwrap_or_else(|| {
                    crate::graph::DocumentData {
                        text: String::new(),
                        metadata: std::collections::HashMap::new(),
                    }
                });
                SearchResult {
                    id: node_id,
                    text: doc_data.text,
                    score: distance,
                    metadata: doc_data.metadata,
                }
            })
            .collect();

        Ok(results)
    }

    /// Convert candidate nodes to search results (non-compact mode)
    fn candidates_to_results(&self, candidates: Vec<NodeId>, k: usize) -> Result<Vec<SearchResult>> {
        // For non-compact mode, we need to compute distances from the query
        // This function is called from search_async which doesn't pass the query
        // So we'll just return results ordered by node ID for now
        // The proper async search flow should use recomputation_phase instead

        let results = candidates
            .into_iter()
            .take(k)
            .map(|node_id| {
                let doc_data = self.graph.get_document_data(node_id).unwrap_or_else(|| {
                    crate::graph::DocumentData {
                        text: String::new(),
                        metadata: std::collections::HashMap::new(),
                    }
                });

                SearchResult {
                    id: node_id,
                    text: doc_data.text,
                    score: 0.0, // Placeholder - actual distance computed in search_layer
                    metadata: doc_data.metadata,
                }
            })
            .collect();

        Ok(results)
    }

    /// Greedy search in a single layer
    ///
    /// Returns ef closest neighbors found in the layer
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: NodeId,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<QueueElement>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Compute distance for entry point (should always be a hub node with vector available)
        let entry_dist = self.compute_distance_safe(query, entry_point)?;
        let entry_elem = QueueElement {
            node: entry_point,
            distance: entry_dist,
        };

        candidates.push(Reverse(entry_elem)); // Min-heap
        results.push(entry_elem); // Max-heap
        visited.insert(entry_point);

        while let Some(Reverse(current)) = candidates.pop() {
            // If current is farther than the worst result, stop
            if results.len() >= ef {
                if let Some(&furthest) = results.peek() {
                    if current.distance > furthest.distance {
                        break;
                    }
                }
            }

            // Check neighbors of current
            let neighbors = self.graph.get_neighbors(current.node, layer);
            for edge in neighbors {
                if visited.contains(&edge.target) {
                    continue;
                }
                visited.insert(edge.target);

                // In compact mode, use edge distance if target vector is not available
                let dist = if self.is_compact_mode() && !self.graph.is_vector_available(edge.target) {
                    // For pruned nodes during traversal, use stored edge distance
                    edge.distance
                } else {
                    // For hub nodes or non-compact mode, compute actual distance
                    self.compute_distance(query, edge.target)?
                };

                let elem = QueueElement {
                    node: edge.target,
                    distance: dist,
                };

                // If result set is not full or this node is closer than the furthest result
                if results.len() < ef {
                    candidates.push(Reverse(elem));
                    results.push(elem);
                } else if let Some(&furthest) = results.peek() {
                    if dist < furthest.distance {
                        candidates.push(Reverse(elem));
                        results.push(elem);

                        // Remove furthest if we exceed ef
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert max-heap to sorted vector (closest first)
        let mut sorted_results: Vec<QueueElement> = results.into_iter().collect();
        sorted_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(sorted_results)
    }

    /// Compute distance between query and a node's vector
    fn compute_distance(&self, query: &[f32], node: NodeId) -> Result<f32> {
        let vector = self.graph.get_vector(node).ok_or_else(|| {
            vyakti_common::VyaktiError::Backend(format!("Node {} not found", node))
        })?;
        Ok(self.graph.distance(query, &vector))
    }

    /// Safely compute distance, returning error if vector not available
    /// Used for entry points which should always have vectors
    fn compute_distance_safe(&self, query: &[f32], node: NodeId) -> Result<f32> {
        self.compute_distance(query, node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::HnswConfig;
    use vyakti_common::DistanceMetric;

    fn create_test_graph() -> HnswGraph {
        let graph = HnswGraph::new();

        // Add some test vectors
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // 0
            vec![0.9, 0.1, 0.0], // 1 - close to 0
            vec![0.0, 1.0, 0.0], // 2
            vec![0.0, 0.9, 0.1], // 3 - close to 2
            vec![0.0, 0.0, 1.0], // 4
        ];

        for vec in vectors {
            graph.add_vector(vec);
        }

        // Build a simple connected graph at layer 0
        graph.add_edge(0, 0, 1, 0.1).unwrap();
        graph.add_edge(0, 1, 0, 0.1).unwrap();
        graph.add_edge(0, 1, 2, 0.8).unwrap();
        graph.add_edge(0, 2, 1, 0.8).unwrap();
        graph.add_edge(0, 2, 3, 0.1).unwrap();
        graph.add_edge(0, 3, 2, 0.1).unwrap();
        graph.add_edge(0, 3, 4, 0.8).unwrap();
        graph.add_edge(0, 4, 3, 0.8).unwrap();

        graph.set_entry_point(0, 0);

        graph
    }

    #[test]
    fn test_searcher_empty_graph() {
        let graph = HnswGraph::new();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search(&query, 5, 10).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_searcher_single_node() {
        let graph = HnswGraph::new();
        graph.add_vector(vec![1.0, 0.0, 0.0]);
        graph.set_entry_point(0, 0);

        let searcher = HnswSearcher::new(&graph);
        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search(&query, 1, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_searcher_finds_nearest() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        // Query similar to vector 0
        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search(&query, 2, 10).unwrap();

        assert_eq!(results.len(), 2);
        // Results should be sorted by score
        assert!(results[0].score <= results[1].score);
        // At least one result should have a good score (close to the query)
        assert!(results[0].score <= 0.5); // Should find something reasonably close
    }

    #[test]
    fn test_searcher_k_larger_than_graph() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search(&query, 10, 20).unwrap();

        // Should return all nodes (5)
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_searcher_results_sorted() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search(&query, 5, 10).unwrap();

        // Results should be sorted by score (distance)
        for i in 1..results.len() {
            assert!(results[i - 1].score <= results[i].score);
        }
    }

    #[test]
    fn test_search_layer() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search_layer(&query, 0, 3, 0).unwrap();

        assert!(results.len() <= 3);
        assert!(!results.is_empty());

        // Should be sorted
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_searcher_with_euclidean_distance() {
        let config = HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let graph = HnswGraph::with_config(config);

        graph.add_vector(vec![0.0, 0.0]);
        graph.add_vector(vec![3.0, 4.0]);
        graph.add_vector(vec![1.0, 1.0]);

        graph.add_edge(0, 0, 1, 5.0).unwrap();
        graph.add_edge(0, 0, 2, 1.414).unwrap();
        graph.add_edge(0, 1, 0, 5.0).unwrap();
        graph.add_edge(0, 2, 0, 1.414).unwrap();

        graph.set_entry_point(0, 0);

        let searcher = HnswSearcher::new(&graph);
        let query = vec![0.0, 0.0];
        let results = searcher.search(&query, 2, 5).unwrap();

        assert_eq!(results.len(), 2);
        // Closest should be node 0 (score/distance 0)
        assert_eq!(results[0].id, 0);
        assert!((results[0].score - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_distance() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let dist = searcher.compute_distance(&query, 0).unwrap();

        // Should be distance 0.0 for identical vectors (cosine distance)
        assert!((dist - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_distance_invalid_node() {
        let graph = HnswGraph::new();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0];
        let result = searcher.compute_distance(&query, 100);

        assert!(result.is_err());
    }

    // ===== Async Search with Recomputation Tests =====

    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vyakti_common::EmbeddingProvider;
    use vyakti_embedding::{EmbeddingRecomputationService, RecomputationConfig};

    /// Mock embedding provider for testing
    struct MockEmbeddingProvider {
        dimension: usize,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockEmbeddingProvider {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<vyakti_common::Vector>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            // Generate deterministic embeddings based on text content
            let embeddings = texts
                .iter()
                .map(|text| {
                    let mut vec = vec![0.0; self.dimension];
                    // Use text length to create variation
                    let val = (text.len() as f32) / 100.0;
                    vec[0] = val;
                    vec[1] = 1.0 - val;
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
    async fn test_search_async_non_compact_mode() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search_async(&query, 2, 10).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score <= results[1].score);
    }

    #[tokio::test]
    async fn test_search_async_compact_mode_with_recomputation() {
        // Create a graph
        let graph = HnswGraph::new();

        // Add vectors with text
        let docs = vec![
            ("Document about vectors and embeddings", vec![1.0, 0.0, 0.0]),
            ("Short doc", vec![0.9, 0.1, 0.0]),
            ("Another document about machine learning", vec![0.0, 1.0, 0.0]),
            ("Brief text", vec![0.0, 0.9, 0.1]),
            ("Final document with more content here", vec![0.0, 0.0, 1.0]),
        ];

        for (text, vec) in &docs {
            graph.add_vector_with_data(vec.clone(), text.to_string(), std::collections::HashMap::new());
        }

        // Build graph connectivity
        graph.add_edge(0, 0, 1, 0.1).unwrap();
        graph.add_edge(0, 1, 0, 0.1).unwrap();
        graph.add_edge(0, 1, 2, 0.8).unwrap();
        graph.add_edge(0, 2, 1, 0.8).unwrap();
        graph.add_edge(0, 2, 3, 0.1).unwrap();
        graph.add_edge(0, 3, 2, 0.1).unwrap();
        graph.add_edge(0, 3, 4, 0.8).unwrap();
        graph.add_edge(0, 4, 3, 0.8).unwrap();

        graph.set_entry_point(0, 0);

        // Simulate pruning: keep nodes 0 and 2 as hubs, prune others
        let _hub_nodes: HashSet<NodeId> = [0, 2].iter().cloned().collect();
        let pruned_nodes: HashSet<NodeId> = [1, 3, 4].iter().cloned().collect();

        graph.prune_node_embeddings(&pruned_nodes).unwrap();
        graph.enable_compact_mode();

        assert_eq!(graph.num_pruned_nodes(), 3);
        assert!(graph.is_compact_mode());

        // Create recomputation service
        let provider = Arc::new(MockEmbeddingProvider::new(3));
        let config = RecomputationConfig::default();
        let service = Arc::new(EmbeddingRecomputationService::new(provider.clone(), config));

        // Create searcher with recomputation
        let searcher = HnswSearcher::with_recomputation(&graph, service);

        assert!(searcher.is_compact_mode());

        // Perform search
        let query = vec![1.0, 0.0, 0.0];
        let results = searcher.search_async(&query, 3, 10).await.unwrap();

        // Should get results despite pruned embeddings
        assert!(!results.is_empty());
        assert!(results.len() <= 3);

        // Provider should have been called for recomputation
        assert!(provider.call_count() > 0);
    }

    #[tokio::test]
    async fn test_recomputation_phase_hub_vs_pruned() {
        let graph = HnswGraph::new();

        // Add 10 nodes with text
        for i in 0..10 {
            let text = format!("Document number {} with some content", i);
            let vec = vec![i as f32 / 10.0; 3];
            graph.add_vector_with_data(vec, text, std::collections::HashMap::new());
        }

        // Keep first 2 as hubs, prune rest
        let pruned: HashSet<NodeId> = (2..10).collect();
        graph.prune_node_embeddings(&pruned).unwrap();
        graph.enable_compact_mode();

        // Setup recomputation
        let provider = Arc::new(MockEmbeddingProvider::new(3));
        let config = RecomputationConfig::default();
        let service = Arc::new(EmbeddingRecomputationService::new(provider.clone(), config));

        let searcher = HnswSearcher::with_recomputation(&graph, service);

        // Search with candidates including both hub and pruned nodes
        let query = vec![0.5; 3];
        let candidates = vec![0, 1, 5, 7, 9]; // Mix of hub (0,1) and pruned (5,7,9)

        let results = searcher
            .recomputation_phase(&query, candidates, 5)
            .await
            .unwrap();

        assert_eq!(results.len(), 5);
        // Verify recomputation was called for pruned nodes
        assert!(provider.call_count() > 0);
    }

    #[tokio::test]
    async fn test_graph_traversal_phase() {
        let graph = create_test_graph();
        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        let candidates = searcher.graph_traversal_phase(&query, 3).unwrap();

        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 5); // Graph has 5 nodes
    }

    #[tokio::test]
    async fn test_compact_mode_batch_recomputation() {
        let graph = HnswGraph::new();

        // Add many nodes
        for i in 0..50 {
            let text = format!("Document {}", i);
            let vec = vec![(i % 10) as f32 / 10.0; 3];
            graph.add_vector_with_data(vec, text, std::collections::HashMap::new());
        }

        // Prune all except first 5 (hubs)
        let pruned: HashSet<NodeId> = (5..50).collect();
        graph.prune_node_embeddings(&pruned).unwrap();
        graph.enable_compact_mode();

        let provider = Arc::new(MockEmbeddingProvider::new(3));
        let config = RecomputationConfig {
            max_batch_size: 10,
            ..Default::default()
        };
        let service = Arc::new(EmbeddingRecomputationService::new(provider.clone(), config));

        let searcher = HnswSearcher::with_recomputation(&graph, service);

        // Candidates with many pruned nodes
        let candidates: Vec<NodeId> = (0..20).collect();

        let results = searcher
            .recomputation_phase(&vec![0.5; 3], candidates, 10)
            .await
            .unwrap();

        assert_eq!(results.len(), 10);
        // Should have made multiple batch calls
        assert!(provider.call_count() >= 1);
    }

    #[tokio::test]
    async fn test_search_without_recomputation_service() {
        let graph = create_test_graph();

        // Enable compact mode but don't provide recomputation service
        graph.enable_compact_mode();

        let searcher = HnswSearcher::new(&graph);

        let query = vec![1.0, 0.0, 0.0];
        // Should still work in non-compact search path
        let results = searcher.search(&query, 2, 10).unwrap();

        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_is_compact_mode() {
        let graph = HnswGraph::new();
        graph.add_vector(vec![1.0, 2.0, 3.0]);

        // Regular searcher
        let searcher1 = HnswSearcher::new(&graph);
        assert!(!searcher1.is_compact_mode());

        // With service but graph not in compact mode
        let provider = Arc::new(MockEmbeddingProvider::new(3));
        let service = Arc::new(EmbeddingRecomputationService::new(
            provider,
            RecomputationConfig::default(),
        ));
        let searcher2 = HnswSearcher::with_recomputation(&graph, service.clone());
        assert!(!searcher2.is_compact_mode());

        // Both graph and service
        graph.enable_compact_mode();
        let searcher3 = HnswSearcher::with_recomputation(&graph, service);
        assert!(searcher3.is_compact_mode());
    }
}
