//! HNSW graph structure.

use parking_lot::RwLock;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vyakti_common::{DistanceMetric, Result, Vector};

/// Document data associated with a vector
#[derive(Debug, Clone)]
pub struct DocumentData {
    /// Document text
    pub text: String,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Node ID type
pub type NodeId = usize;

/// Type alias for the layer structure
type LayerMap = Arc<RwLock<Vec<HashMap<NodeId, Vec<Edge>>>>>;

/// Edge with distance
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Target node ID
    pub target: NodeId,
    /// Distance to target
    pub distance: f32,
}

/// HNSW graph configuration
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M parameter)
    pub max_connections: usize,
    /// Maximum number of connections for layer 0 (typically M * 2)
    pub max_connections_0: usize,
    /// Level generation multiplier (typically 1/ln(M))
    pub ml: f64,
    /// Construction-time search depth
    pub ef_construction: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            max_connections: m,
            max_connections_0: m * 2,
            ml: 1.0 / (m as f64).ln(),
            ef_construction: 200,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// HNSW graph implementation with multi-layer structure
pub struct HnswGraph {
    /// Configuration
    config: HnswConfig,
    /// Vectors stored in the graph (None when pruned in compact mode)
    vectors: Arc<RwLock<Vec<Option<Vector>>>>,
    /// Document data (text and metadata) for each vector
    document_data: Arc<RwLock<Vec<DocumentData>>>,
    /// Adjacency list for each layer: layer -> node -> neighbors
    /// Using Vec of HashMaps for efficiency
    layers: LayerMap,
    /// Entry point node ID and its level
    entry_point: Arc<RwLock<Option<(NodeId, usize)>>>,
    /// Random number generator
    rng: Arc<RwLock<rand::rngs::StdRng>>,
    /// Whether the graph is in compact mode (embeddings pruned)
    compact_mode: Arc<RwLock<bool>>,
    /// Set of nodes whose embeddings have been pruned
    pruned_nodes: Arc<RwLock<HashSet<NodeId>>>,
}

impl HnswGraph {
    /// Create a new HNSW graph with default configuration
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create a new HNSW graph with custom configuration
    pub fn with_config(config: HnswConfig) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            vectors: Arc::new(RwLock::new(Vec::new())),
            document_data: Arc::new(RwLock::new(Vec::new())),
            layers: Arc::new(RwLock::new(vec![HashMap::new()])), // Start with layer 0
            entry_point: Arc::new(RwLock::new(None)),
            rng: Arc::new(RwLock::new(rand::rngs::StdRng::from_entropy())),
            compact_mode: Arc::new(RwLock::new(false)),
            pruned_nodes: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Get the number of nodes in the graph
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.read().len()
    }

    /// Get the entry point (node ID, level)
    pub fn entry_point(&self) -> Option<(NodeId, usize)> {
        *self.entry_point.read()
    }

    /// Assign a random level to a new node using exponential decay
    pub fn random_level(&self) -> usize {
        let mut rng = self.rng.write();
        let uniform: f64 = rng.gen();
        (-uniform.ln() * self.config.ml).floor() as usize
    }

    /// Get neighbors of a node at a specific layer
    pub fn get_neighbors(&self, node: NodeId, layer: usize) -> Vec<Edge> {
        let layers = self.layers.read();
        if layer >= layers.len() {
            return Vec::new();
        }
        layers[layer].get(&node).cloned().unwrap_or_default()
    }

    /// Add an edge between two nodes at a specific layer
    pub fn add_edge(&self, layer: usize, from: NodeId, to: NodeId, distance: f32) -> Result<()> {
        let mut layers = self.layers.write();

        // Ensure layer exists
        while layers.len() <= layer {
            layers.push(HashMap::new());
        }

        // Add edge
        layers[layer].entry(from).or_default().push(Edge {
            target: to,
            distance,
        });

        Ok(())
    }

    /// Add a vector and return its node ID
    pub fn add_vector(&self, vector: Vector) -> NodeId {
        self.add_vector_with_data(vector, String::new(), HashMap::new())
    }

    /// Add a vector with document data (text and metadata) and return its node ID
    pub fn add_vector_with_data(
        &self,
        vector: Vector,
        text: String,
        metadata: HashMap<String, serde_json::Value>,
    ) -> NodeId {
        let mut vectors = self.vectors.write();
        let mut document_data = self.document_data.write();
        let node_id = vectors.len();
        vectors.push(Some(vector)); // Wrap in Some for compact mode support
        document_data.push(DocumentData { text, metadata });
        node_id
    }

    /// Get a vector by node ID
    /// Returns None if the node doesn't exist or if the embedding has been pruned
    pub fn get_vector(&self, node: NodeId) -> Option<Vector> {
        self.vectors.read().get(node).and_then(|v| v.clone())
    }

    /// Get document data by node ID
    pub fn get_document_data(&self, node: NodeId) -> Option<DocumentData> {
        self.document_data.read().get(node).cloned()
    }

    /// Set document data (text and metadata) for a node
    pub fn set_document_data(
        &self,
        node: NodeId,
        text: String,
        metadata: std::collections::HashMap<String, serde_json::Value>,
    ) {
        let mut data = self.document_data.write();

        // Ensure the vec is large enough
        while data.len() <= node {
            data.push(DocumentData {
                text: String::new(),
                metadata: std::collections::HashMap::new(),
            });
        }

        // Replace the document data at the specified index
        data[node] = DocumentData { text, metadata };
    }

    /// Compute distance between two vectors
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.config.metric.compute(a, b)
    }

    /// Set the entry point
    pub fn set_entry_point(&self, node: NodeId, level: usize) {
        *self.entry_point.write() = Some((node, level));
    }

    /// Get configuration
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Prune neighbors to keep only the best `max_connections` neighbors
    pub fn prune_neighbors(&self, layer: usize, node: NodeId) -> Result<()> {
        let max_connections = if layer == 0 {
            self.config.max_connections_0
        } else {
            self.config.max_connections
        };

        let mut layers = self.layers.write();
        if layer >= layers.len() {
            return Ok(());
        }

        if let Some(neighbors) = layers[layer].get_mut(&node) {
            if neighbors.len() > max_connections {
                // Sort by distance and keep only the closest neighbors
                neighbors.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                neighbors.truncate(max_connections);
            }
        }

        Ok(())
    }

    // ===== Compact Mode Methods (LEANN) =====

    /// Check if the graph is in compact mode (embeddings have been pruned)
    pub fn is_compact_mode(&self) -> bool {
        *self.compact_mode.read()
    }

    /// Enable compact mode for the graph
    pub fn enable_compact_mode(&self) {
        *self.compact_mode.write() = true;
    }

    /// Check if a vector embedding is available for a node
    ///
    /// Returns false if the node doesn't exist or if the embedding has been pruned
    pub fn is_vector_available(&self, node: NodeId) -> bool {
        self.vectors
            .read()
            .get(node)
            .map(|v| v.is_some())
            .unwrap_or(false)
    }

    /// Prune (delete) the embedding for a specific node
    ///
    /// The graph structure is preserved, but the embedding vector is removed
    /// to save memory. Document data (text) is kept for recomputation.
    pub fn prune_node_embedding(&self, node: NodeId) -> Result<()> {
        let mut vectors = self.vectors.write();
        if node >= vectors.len() {
            return Ok(());
        }

        vectors[node] = None;

        // Track pruned node
        let mut pruned = self.pruned_nodes.write();
        pruned.insert(node);

        Ok(())
    }

    /// Prune embeddings for multiple nodes at once
    pub fn prune_node_embeddings(&self, nodes: &HashSet<NodeId>) -> Result<()> {
        let mut vectors = self.vectors.write();
        let mut pruned = self.pruned_nodes.write();

        for &node in nodes {
            if node < vectors.len() {
                vectors[node] = None;
                pruned.insert(node);
            }
        }

        Ok(())
    }

    /// Get the set of pruned node IDs
    pub fn pruned_nodes(&self) -> HashSet<NodeId> {
        self.pruned_nodes.read().clone()
    }

    /// Get the number of pruned nodes
    pub fn num_pruned_nodes(&self) -> usize {
        self.pruned_nodes.read().len()
    }

    /// Get document text for a node (needed for recomputation)
    pub fn get_document_text(&self, node: NodeId) -> Option<String> {
        self.document_data
            .read()
            .get(node)
            .map(|data| data.text.clone())
    }

    /// Restore an embedding for a previously pruned node
    ///
    /// This is used during search when an embedding needs to be recomputed
    pub fn restore_node_embedding(&self, node: NodeId, embedding: Vector) -> Result<()> {
        let mut vectors = self.vectors.write();
        if node >= vectors.len() {
            return Ok(());
        }

        vectors[node] = Some(embedding);

        // Remove from pruned set (though we may want to re-prune later)
        // For now, keep it in pruned_nodes to track original pruning
        // let mut pruned = self.pruned_nodes.write();
        // pruned.remove(&node);

        Ok(())
    }

    // ===== Graph Topology Export/Import (for Persistence) =====

    /// Export graph topology for serialization
    ///
    /// Returns a tuple of (layers, entry_point) that can be serialized
    pub fn export_topology(&self) -> (Vec<HashMap<NodeId, Vec<(NodeId, f32)>>>, Option<(NodeId, usize)>) {
        let layers = self.layers.read();
        let entry_point = *self.entry_point.read();

        // Convert internal Edge structure to serializable (target, distance) tuples
        let serializable_layers: Vec<HashMap<NodeId, Vec<(NodeId, f32)>>> = layers
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|(node_id, edges)| {
                        let edge_list: Vec<(NodeId, f32)> = edges
                            .iter()
                            .map(|edge| (edge.target, edge.distance))
                            .collect();
                        (*node_id, edge_list)
                    })
                    .collect()
            })
            .collect();

        (serializable_layers, entry_point)
    }

    /// Import graph topology from deserialized data
    ///
    /// Restores the graph structure (layers and entry point) from serialized format
    pub fn import_topology(&self, serialized_layers: Vec<HashMap<NodeId, Vec<(NodeId, f32)>>>, entry_point: Option<(NodeId, usize)>) -> Result<()> {
        let mut layers = self.layers.write();

        // Clear existing layers
        layers.clear();

        // Convert serializable (target, distance) tuples back to Edge structures
        for layer in serialized_layers {
            let reconstructed_layer: HashMap<NodeId, Vec<Edge>> = layer
                .into_iter()
                .map(|(node_id, edge_list)| {
                    let edges: Vec<Edge> = edge_list
                        .into_iter()
                        .map(|(target, distance)| Edge { target, distance })
                        .collect();
                    (node_id, edges)
                })
                .collect();
            layers.push(reconstructed_layer);
        }

        // Restore entry point
        *self.entry_point.write() = entry_point;

        Ok(())
    }
}

impl Default for HnswGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_graph_new() {
        let graph = HnswGraph::new();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
        assert_eq!(graph.num_layers(), 1); // Starts with layer 0
        assert!(graph.entry_point().is_none());
    }

    #[test]
    fn test_hnsw_graph_with_config() {
        let config = HnswConfig {
            max_connections: 32,
            max_connections_0: 64,
            ml: 0.5,
            ef_construction: 100,
            metric: DistanceMetric::Euclidean,
        };
        let graph = HnswGraph::with_config(config.clone());
        assert_eq!(graph.config().max_connections, 32);
        assert_eq!(graph.config().max_connections_0, 64);
    }

    #[test]
    fn test_add_vector() {
        let graph = HnswGraph::new();
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        let id1 = graph.add_vector(vec1.clone());
        let id2 = graph.add_vector(vec2.clone());

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());

        assert_eq!(graph.get_vector(id1).unwrap(), vec1);
        assert_eq!(graph.get_vector(id2).unwrap(), vec2);
    }

    #[test]
    fn test_random_level() {
        let graph = HnswGraph::new();
        let mut levels = Vec::new();

        // Generate 1000 random levels
        for _ in 0..1000 {
            levels.push(graph.random_level());
        }

        // Check that we get a variety of levels
        let max_level = *levels.iter().max().unwrap();
        assert!(max_level >= 1); // Should have some nodes at higher levels

        // Most nodes should be at level 0
        let level_0_count = levels.iter().filter(|&&l| l == 0).count();
        assert!(level_0_count > 700); // At least 70% at level 0
    }

    #[test]
    fn test_add_edge() {
        let graph = HnswGraph::new();
        let _id1 = graph.add_vector(vec![1.0, 2.0]);
        let _id2 = graph.add_vector(vec![3.0, 4.0]);

        graph.add_edge(0, 0, 1, 0.5).unwrap();
        graph.add_edge(0, 1, 0, 0.5).unwrap();

        let neighbors_0 = graph.get_neighbors(0, 0);
        assert_eq!(neighbors_0.len(), 1);
        assert_eq!(neighbors_0[0].target, 1);
        assert_eq!(neighbors_0[0].distance, 0.5);

        let neighbors_1 = graph.get_neighbors(1, 0);
        assert_eq!(neighbors_1.len(), 1);
        assert_eq!(neighbors_1[0].target, 0);
    }

    #[test]
    fn test_multi_layer() {
        let graph = HnswGraph::new();
        let _id1 = graph.add_vector(vec![1.0]);
        let _id2 = graph.add_vector(vec![2.0]);

        // Add edges at different layers
        graph.add_edge(0, 0, 1, 1.0).unwrap();
        graph.add_edge(1, 0, 1, 1.0).unwrap();
        graph.add_edge(2, 0, 1, 1.0).unwrap();

        assert_eq!(graph.num_layers(), 3);

        // Check neighbors at each layer
        assert_eq!(graph.get_neighbors(0, 0).len(), 1);
        assert_eq!(graph.get_neighbors(0, 1).len(), 1);
        assert_eq!(graph.get_neighbors(0, 2).len(), 1);
    }

    #[test]
    fn test_entry_point() {
        let graph = HnswGraph::new();
        assert!(graph.entry_point().is_none());

        graph.set_entry_point(5, 3);
        assert_eq!(graph.entry_point(), Some((5, 3)));

        graph.set_entry_point(10, 2);
        assert_eq!(graph.entry_point(), Some((10, 2)));
    }

    #[test]
    fn test_distance_cosine() {
        let config = HnswConfig {
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let graph = HnswGraph::with_config(config);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let distance = graph.distance(&a, &b);
        assert!((distance - 0.0).abs() < 1e-5); // Same vector, distance = 0.0

        // Test orthogonal vectors
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        let distance2 = graph.distance(&c, &d);
        assert!((distance2 - 1.0).abs() < 1e-5); // Orthogonal vectors, distance = 1.0
    }

    #[test]
    fn test_distance_euclidean() {
        let config = HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let graph = HnswGraph::with_config(config);

        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = graph.distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5); // 3-4-5 triangle
    }

    #[test]
    fn test_prune_neighbors() {
        let config = HnswConfig {
            max_connections: 2,
            max_connections_0: 4,
            ..Default::default()
        };
        let graph = HnswGraph::with_config(config);

        let _id = graph.add_vector(vec![1.0]);

        // Add 5 edges to node 0 at layer 0
        graph.add_edge(0, 0, 1, 0.5).unwrap();
        graph.add_edge(0, 0, 2, 0.3).unwrap();
        graph.add_edge(0, 0, 3, 0.7).unwrap();
        graph.add_edge(0, 0, 4, 0.2).unwrap();
        graph.add_edge(0, 0, 5, 0.9).unwrap();

        assert_eq!(graph.get_neighbors(0, 0).len(), 5);

        // Prune to max_connections_0 = 4
        graph.prune_neighbors(0, 0).unwrap();
        let neighbors = graph.get_neighbors(0, 0);
        assert_eq!(neighbors.len(), 4);

        // Check that closest neighbors are kept (sorted by distance)
        assert!(neighbors[0].distance <= neighbors[1].distance);
        assert!(neighbors[1].distance <= neighbors[2].distance);
        assert!(neighbors[2].distance <= neighbors[3].distance);
    }

    #[test]
    fn test_prune_neighbors_layer_1() {
        let config = HnswConfig {
            max_connections: 2,
            ..Default::default()
        };
        let graph = HnswGraph::with_config(config);

        // Add edges at layer 1
        graph.add_edge(1, 0, 1, 0.5).unwrap();
        graph.add_edge(1, 0, 2, 0.3).unwrap();
        graph.add_edge(1, 0, 3, 0.7).unwrap();

        assert_eq!(graph.get_neighbors(0, 1).len(), 3);

        // Prune to max_connections = 2
        graph.prune_neighbors(1, 0).unwrap();
        let neighbors = graph.get_neighbors(0, 1);
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_get_nonexistent_vector() {
        let graph = HnswGraph::new();
        assert!(graph.get_vector(0).is_none());
        assert!(graph.get_vector(100).is_none());
    }

    #[test]
    fn test_get_neighbors_nonexistent_layer() {
        let graph = HnswGraph::new();
        let neighbors = graph.get_neighbors(0, 10);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_get_neighbors_nonexistent_node() {
        let graph = HnswGraph::new();
        let neighbors = graph.get_neighbors(100, 0);
        assert!(neighbors.is_empty());
    }

    // ===== Compact Mode Tests =====

    #[test]
    fn test_compact_mode_initial_state() {
        let graph = HnswGraph::new();
        assert!(!graph.is_compact_mode());
        assert_eq!(graph.num_pruned_nodes(), 0);
    }

    #[test]
    fn test_enable_compact_mode() {
        let graph = HnswGraph::new();
        assert!(!graph.is_compact_mode());

        graph.enable_compact_mode();
        assert!(graph.is_compact_mode());
    }

    #[test]
    fn test_is_vector_available() {
        let graph = HnswGraph::new();
        let id = graph.add_vector(vec![1.0, 2.0, 3.0]);

        assert!(graph.is_vector_available(id));
        assert!(!graph.is_vector_available(999)); // Non-existent node
    }

    #[test]
    fn test_prune_node_embedding() {
        let graph = HnswGraph::new();
        let id = graph.add_vector(vec![1.0, 2.0, 3.0]);

        assert!(graph.is_vector_available(id));
        assert!(graph.get_vector(id).is_some());

        // Prune the embedding
        graph.prune_node_embedding(id).unwrap();

        assert!(!graph.is_vector_available(id));
        assert!(graph.get_vector(id).is_none());
        assert_eq!(graph.num_pruned_nodes(), 1);
        assert!(graph.pruned_nodes().contains(&id));
    }

    #[test]
    fn test_prune_node_embeddings_bulk() {
        let graph = HnswGraph::new();

        // Add multiple vectors
        let ids: Vec<NodeId> = (0..10).map(|i| graph.add_vector(vec![i as f32; 3])).collect();

        // Prune half of them
        let to_prune: HashSet<NodeId> = ids.iter().step_by(2).cloned().collect();
        graph.prune_node_embeddings(&to_prune).unwrap();

        assert_eq!(graph.num_pruned_nodes(), 5);

        // Check that pruned nodes have no vectors
        for id in to_prune.iter() {
            assert!(!graph.is_vector_available(*id));
            assert!(graph.get_vector(*id).is_none());
        }

        // Check that un-pruned nodes still have vectors
        for id in ids.iter().skip(1).step_by(2) {
            assert!(graph.is_vector_available(*id));
            assert!(graph.get_vector(*id).is_some());
        }
    }

    #[test]
    fn test_get_document_text() {
        let graph = HnswGraph::new();

        let text = "This is a test document".to_string();
        let id = graph.add_vector_with_data(vec![1.0, 2.0], text.clone(), HashMap::new());

        assert_eq!(graph.get_document_text(id), Some(text));
        assert_eq!(graph.get_document_text(999), None);
    }

    #[test]
    fn test_prune_preserves_document_data() {
        let graph = HnswGraph::new();

        let text = "Important document text".to_string();
        let id = graph.add_vector_with_data(vec![1.0, 2.0], text.clone(), HashMap::new());

        // Prune embedding
        graph.prune_node_embedding(id).unwrap();

        // Document text should still be available
        assert_eq!(graph.get_document_text(id), Some(text));
        assert!(graph.get_document_data(id).is_some());
    }

    #[test]
    fn test_restore_node_embedding() {
        let graph = HnswGraph::new();

        let original = vec![1.0, 2.0, 3.0];
        let id = graph.add_vector(original.clone());

        // Prune the embedding
        graph.prune_node_embedding(id).unwrap();
        assert!(!graph.is_vector_available(id));

        // Restore with new embedding
        let restored = vec![4.0, 5.0, 6.0];
        graph.restore_node_embedding(id, restored.clone()).unwrap();

        assert!(graph.is_vector_available(id));
        assert_eq!(graph.get_vector(id), Some(restored));
    }

    #[test]
    fn test_prune_nonexistent_node() {
        let graph = HnswGraph::new();

        // Pruning non-existent node should not panic
        let result = graph.prune_node_embedding(999);
        assert!(result.is_ok());
        assert_eq!(graph.num_pruned_nodes(), 0);
    }

    #[test]
    fn test_compact_mode_workflow() {
        let graph = HnswGraph::new();

        // Add vectors
        let ids: Vec<NodeId> = (0..100)
            .map(|i| {
                graph.add_vector_with_data(
                    vec![i as f32; 768],
                    format!("Document {}", i),
                    HashMap::new(),
                )
            })
            .collect();

        // Simulate pruning 95% (keep only 5 hub nodes)
        let hub_nodes: HashSet<NodeId> = ids.iter().take(5).cloned().collect();
        let to_prune: HashSet<NodeId> = ids.iter().skip(5).cloned().collect();

        graph.prune_node_embeddings(&to_prune).unwrap();
        graph.enable_compact_mode();

        assert!(graph.is_compact_mode());
        assert_eq!(graph.num_pruned_nodes(), 95);

        // Hub nodes should have vectors
        for id in hub_nodes {
            assert!(graph.is_vector_available(id));
        }

        // Pruned nodes should not have vectors but should have text
        for id in to_prune {
            assert!(!graph.is_vector_available(id));
            assert!(graph.get_document_text(id).is_some());
        }
    }
}
