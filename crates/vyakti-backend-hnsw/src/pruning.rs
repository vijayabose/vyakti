//! Graph pruning logic for LEANN algorithm.
//!
//! Implements selective embedding deletion to achieve 95-97% storage savings
//! while preserving graph connectivity through hub nodes.

use crate::graph::{HnswGraph, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, info};
use vyakti_common::Result;

/// Configuration for graph pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Preserve high-degree nodes (hub nodes) during pruning
    pub preserve_high_degree_nodes: bool,
    /// Degree threshold percentile for identifying hub nodes (0.0-1.0)
    /// Default: 0.95 (top 5% of nodes by degree)
    pub degree_threshold_percentile: f64,
    /// Minimum degree to be considered a hub node
    pub min_hub_degree: usize,
    /// Whether to prune embeddings (delete non-hub embeddings)
    pub prune_embeddings: bool,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            preserve_high_degree_nodes: true,
            degree_threshold_percentile: 0.95, // Top 5%
            min_hub_degree: 10,
            prune_embeddings: true,
        }
    }
}

/// Statistics from pruning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStats {
    /// Total number of nodes in the graph
    pub total_nodes: usize,
    /// Number of embeddings kept (hub nodes)
    pub embeddings_kept: usize,
    /// Number of embeddings pruned (non-hub nodes)
    pub embeddings_pruned: usize,
    /// Storage size before pruning (bytes)
    pub storage_before_bytes: usize,
    /// Storage size after pruning (bytes)
    pub storage_after_bytes: usize,
    /// Storage savings percentage (0-100)
    pub savings_percent: f64,
    /// Number of hub nodes identified
    pub hub_nodes_count: usize,
    /// Degree threshold used for hub identification
    pub degree_threshold: usize,
}

impl PruningStats {
    /// Get storage before pruning in human-readable format.
    pub fn storage_before_human(&self) -> String {
        humanize_bytes(self.storage_before_bytes)
    }

    /// Get storage after pruning in human-readable format.
    pub fn storage_after_human(&self) -> String {
        humanize_bytes(self.storage_after_bytes)
    }

    /// Get embedding retention rate (0-1).
    pub fn retention_rate(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            self.embeddings_kept as f64 / self.total_nodes as f64
        }
    }
}

/// Graph pruner for implementing LEANN storage optimization.
pub struct GraphPruner {
    config: PruningConfig,
}

impl GraphPruner {
    /// Create a new graph pruner with the given configuration.
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Create a graph pruner with default configuration.
    pub fn default() -> Self {
        Self::new(PruningConfig::default())
    }

    /// Identify hub nodes in the graph based on degree centrality.
    ///
    /// Hub nodes are identified as nodes with degree above a threshold,
    /// typically the top 5% of nodes by total degree across all layers.
    /// The entry point is always included as a hub node.
    ///
    /// # Arguments
    ///
    /// * `graph` - The HNSW graph
    ///
    /// # Returns
    ///
    /// Set of hub node IDs and the degree threshold used
    pub fn identify_hub_nodes(&self, graph: &HnswGraph) -> (HashSet<NodeId>, usize) {
        let num_nodes = graph.len();
        if num_nodes == 0 {
            return (HashSet::new(), 0);
        }

        info!(
            "Identifying hub nodes (threshold percentile: {}, min degree: {})",
            self.config.degree_threshold_percentile, self.config.min_hub_degree
        );

        // Calculate degree for each node (sum across all layers)
        let mut node_degrees: Vec<(NodeId, usize)> = Vec::with_capacity(num_nodes);

        for node_id in 0..num_nodes {
            let mut total_degree = 0;

            // Sum degree across all layers
            for layer in 0..graph.num_layers() {
                total_degree += graph.get_neighbors(node_id, layer).len();
            }

            node_degrees.push((node_id, total_degree));
        }

        // Sort by degree (descending)
        node_degrees.sort_by(|a, b| b.1.cmp(&a.1));

        // Calculate degree threshold based on percentile
        let threshold_idx =
            ((1.0 - self.config.degree_threshold_percentile) * num_nodes as f64) as usize;
        let threshold_idx = threshold_idx.min(num_nodes.saturating_sub(1));

        let degree_threshold = node_degrees[threshold_idx]
            .1
            .max(self.config.min_hub_degree);

        debug!(
            "Degree threshold: {} (percentile: {}, min: {})",
            degree_threshold, self.config.degree_threshold_percentile, self.config.min_hub_degree
        );

        // Select hub nodes: degree >= threshold
        let mut hub_nodes: HashSet<NodeId> = node_degrees
            .iter()
            .filter(|(_, degree)| *degree >= degree_threshold)
            .map(|(node_id, _)| *node_id)
            .collect();

        // IMPORTANT: Always include the entry point as a hub node
        // This is required for search to work in compact mode
        if let Some((entry_point, _)) = graph.entry_point() {
            if !hub_nodes.contains(&entry_point) {
                debug!("Adding entry point {} to hub nodes (was not selected by degree)", entry_point);
                hub_nodes.insert(entry_point);
            }
        }

        info!(
            "Identified {} hub nodes out of {} total ({:.1}%)",
            hub_nodes.len(),
            num_nodes,
            (hub_nodes.len() as f64 / num_nodes as f64) * 100.0
        );

        (hub_nodes, degree_threshold)
    }

    /// Calculate storage size for embeddings.
    ///
    /// Assumes each embedding is a Vec<f32> with typical dimension.
    fn calculate_embedding_storage(num_embeddings: usize, dimension: usize) -> usize {
        // Each f32 is 4 bytes
        // Add overhead for Vec structure (capacity, length, pointer)
        num_embeddings * (dimension * 4 + 24)
    }

    /// Prune embeddings from the graph, keeping only hub nodes.
    ///
    /// This is the core LEANN operation: delete most embeddings (95%+)
    /// to save storage, keeping only high-degree hub nodes for graph traversal.
    ///
    /// # Arguments
    ///
    /// * `graph` - The HNSW graph to prune (modified in place)
    ///
    /// # Returns
    ///
    /// Pruning statistics showing storage savings
    pub fn prune_embeddings(&self, graph: &HnswGraph) -> Result<PruningStats> {
        info!("Starting embedding pruning process");

        let total_nodes = graph.len();
        if total_nodes == 0 {
            return Ok(PruningStats {
                total_nodes: 0,
                embeddings_kept: 0,
                embeddings_pruned: 0,
                storage_before_bytes: 0,
                storage_after_bytes: 0,
                savings_percent: 0.0,
                hub_nodes_count: 0,
                degree_threshold: 0,
            });
        }

        // Identify hub nodes
        let (hub_nodes, degree_threshold) = self.identify_hub_nodes(graph);

        // Get dimension from first vector (assume all vectors have same dimension)
        let dimension = graph
            .get_vector(0)
            .map(|v| v.len())
            .unwrap_or(768); // Default to 768 if no vectors

        // Calculate storage before pruning
        let storage_before = Self::calculate_embedding_storage(total_nodes, dimension);

        // Calculate storage after pruning (only hub nodes)
        let embeddings_kept = hub_nodes.len();
        let embeddings_pruned = total_nodes - embeddings_kept;
        let storage_after = Self::calculate_embedding_storage(embeddings_kept, dimension);

        // Calculate savings
        let savings_percent = if storage_before > 0 {
            ((storage_before - storage_after) as f64 / storage_before as f64) * 100.0
        } else {
            0.0
        };

        // Actually prune embeddings from the graph
        if self.config.prune_embeddings {
            info!("Pruning {} embeddings from graph", embeddings_pruned);

            // Create set of nodes to prune (all non-hub nodes)
            let nodes_to_prune: std::collections::HashSet<NodeId> = (0..total_nodes)
                .filter(|id| !hub_nodes.contains(id))
                .collect();

            // Prune embeddings in graph
            graph.prune_node_embeddings(&nodes_to_prune)?;

            // Enable compact mode
            graph.enable_compact_mode();

            debug!(
                "Pruned {} nodes, kept {} hub nodes",
                graph.num_pruned_nodes(),
                embeddings_kept
            );
        }

        let stats = PruningStats {
            total_nodes,
            embeddings_kept,
            embeddings_pruned,
            storage_before_bytes: storage_before,
            storage_after_bytes: storage_after,
            savings_percent,
            hub_nodes_count: hub_nodes.len(),
            degree_threshold,
        };

        info!(
            "Pruning complete: kept {}/{} embeddings ({:.1}% savings)",
            stats.embeddings_kept, stats.total_nodes, stats.savings_percent
        );

        info!(
            "Storage: {} → {} ({:.1}% reduction)",
            stats.storage_before_human(),
            stats.storage_after_human(),
            stats.savings_percent
        );

        Ok(stats)
    }

    /// Get the pruning configuration.
    pub fn config(&self) -> &PruningConfig {
        &self.config
    }
}

/// Convert bytes to human-readable format.
fn humanize_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::HnswGraph;

    fn create_test_graph(num_nodes: usize, connections_per_node: usize) -> HnswGraph {
        let graph = HnswGraph::new();

        // Add vectors
        for i in 0..num_nodes {
            let vector = vec![i as f32; 768];
            graph.add_vector(vector);
        }

        // Add edges to create varying degree distribution
        for i in 0..num_nodes {
            let num_connections = if i < num_nodes / 10 {
                // First 10% are hub nodes with many connections
                connections_per_node * 3
            } else {
                connections_per_node
            };

            for j in 0..num_connections.min(num_nodes - 1) {
                let target = (i + j + 1) % num_nodes;
                graph.add_edge(0, i, target, 0.5).unwrap();
            }
        }

        graph
    }

    #[test]
    fn test_pruning_config_default() {
        let config = PruningConfig::default();
        assert!(config.preserve_high_degree_nodes);
        assert!((config.degree_threshold_percentile - 0.95).abs() < 1e-6);
        assert_eq!(config.min_hub_degree, 10);
        assert!(config.prune_embeddings);
    }

    #[test]
    fn test_pruning_config_serialization() {
        let config = PruningConfig {
            preserve_high_degree_nodes: false,
            degree_threshold_percentile: 0.90,
            min_hub_degree: 5,
            prune_embeddings: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PruningConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(
            config.preserve_high_degree_nodes,
            deserialized.preserve_high_degree_nodes
        );
        assert_eq!(
            config.degree_threshold_percentile,
            deserialized.degree_threshold_percentile
        );
    }

    #[test]
    fn test_identify_hub_nodes_empty_graph() {
        let graph = HnswGraph::new();
        let pruner = GraphPruner::default();

        let (hub_nodes, threshold) = pruner.identify_hub_nodes(&graph);

        assert!(hub_nodes.is_empty());
        assert_eq!(threshold, 0);
    }

    #[test]
    fn test_identify_hub_nodes() {
        let graph = create_test_graph(100, 5);
        let pruner = GraphPruner::default();

        let (hub_nodes, threshold) = pruner.identify_hub_nodes(&graph);

        // Should identify roughly 5% as hub nodes (top 5%)
        assert!(hub_nodes.len() <= 10);
        assert!(hub_nodes.len() > 0);
        assert!(threshold >= pruner.config.min_hub_degree);
    }

    #[test]
    fn test_identify_hub_nodes_custom_percentile() {
        let graph = create_test_graph(100, 5);

        // Top 10% instead of top 5%
        let config = PruningConfig {
            degree_threshold_percentile: 0.90,
            ..Default::default()
        };
        let pruner = GraphPruner::new(config);

        let (hub_nodes, _) = pruner.identify_hub_nodes(&graph);

        // Should identify roughly 10% as hub nodes
        assert!(hub_nodes.len() <= 15);
        assert!(hub_nodes.len() >= 5);
    }

    #[test]
    fn test_hub_nodes_include_highest_degree() {
        let graph = create_test_graph(50, 5);
        let pruner = GraphPruner::default();

        let (hub_nodes, _) = pruner.identify_hub_nodes(&graph);

        // First few nodes (0-4) should be hub nodes as they have most connections
        let high_degree_included = (0..5).any(|i| hub_nodes.contains(&i));
        assert!(high_degree_included);
    }

    #[test]
    fn test_prune_embeddings_stats() {
        let graph = create_test_graph(1000, 5);
        let pruner = GraphPruner::default();

        let stats = pruner.prune_embeddings(&graph).unwrap();

        assert_eq!(stats.total_nodes, 1000);
        assert!(stats.embeddings_kept > 0);
        assert_eq!(
            stats.embeddings_kept + stats.embeddings_pruned,
            stats.total_nodes
        );
        assert!(stats.storage_before_bytes > stats.storage_after_bytes);
        assert!(stats.savings_percent > 0.0);
        assert!(stats.savings_percent < 100.0);
    }

    #[test]
    fn test_prune_embeddings_empty_graph() {
        let graph = HnswGraph::new();
        let pruner = GraphPruner::default();

        let stats = pruner.prune_embeddings(&graph).unwrap();

        assert_eq!(stats.total_nodes, 0);
        assert_eq!(stats.embeddings_kept, 0);
        assert_eq!(stats.embeddings_pruned, 0);
        assert_eq!(stats.storage_before_bytes, 0);
        assert_eq!(stats.storage_after_bytes, 0);
    }

    #[test]
    fn test_prune_embeddings_high_savings() {
        let graph = create_test_graph(10000, 5);
        let pruner = GraphPruner::default();

        let stats = pruner.prune_embeddings(&graph).unwrap();

        // With 95th percentile (top 5%), should keep roughly 5% and prune 95%
        // This means storage savings should be around 95%
        assert!(stats.savings_percent > 85.0); // Allow some variance
        assert!(stats.retention_rate() < 0.15); // Should retain less than 15%
        assert!(stats.embeddings_kept < stats.embeddings_pruned); // More pruned than kept
    }

    #[test]
    fn test_pruning_stats_retention_rate() {
        let stats = PruningStats {
            total_nodes: 1000,
            embeddings_kept: 50,
            embeddings_pruned: 950,
            storage_before_bytes: 1000000,
            storage_after_bytes: 50000,
            savings_percent: 95.0,
            hub_nodes_count: 50,
            degree_threshold: 15,
        };

        assert!((stats.retention_rate() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_humanize_bytes() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(500), "500 B");
        assert_eq!(humanize_bytes(1024), "1.00 KB");
        assert_eq!(humanize_bytes(1536), "1.50 KB");
        assert_eq!(humanize_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(humanize_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(humanize_bytes(5 * 1024 * 1024), "5.00 MB");
    }

    #[test]
    fn test_pruning_stats_human_readable() {
        let stats = PruningStats {
            total_nodes: 10000,
            embeddings_kept: 500,
            embeddings_pruned: 9500,
            storage_before_bytes: 100 * 1024 * 1024, // 100 MB
            storage_after_bytes: 5 * 1024 * 1024,    // 5 MB
            savings_percent: 95.0,
            hub_nodes_count: 500,
            degree_threshold: 20,
        };

        assert!(stats.storage_before_human().contains("MB"));
        assert!(stats.storage_after_human().contains("MB"));
    }

    #[test]
    fn test_graph_pruner_config() {
        let config = PruningConfig {
            degree_threshold_percentile: 0.98,
            ..Default::default()
        };
        let pruner = GraphPruner::new(config.clone());

        assert_eq!(
            pruner.config().degree_threshold_percentile,
            config.degree_threshold_percentile
        );
    }

    #[test]
    fn test_calculate_embedding_storage() {
        // 1000 embeddings of 768 dimensions
        let storage = GraphPruner::calculate_embedding_storage(1000, 768);

        // Each embedding: 768 * 4 bytes (f32) + 24 bytes overhead = 3096 bytes
        // Total: 1000 * 3096 = 3,096,000 bytes
        assert_eq!(storage, 3_096_000);
    }
}
