//! Compressed Sparse Row (CSR) graph format.

use serde::{Deserialize, Serialize};
use vyakti_common::{Result, VyaktiError};

/// CSR graph representation for efficient storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrGraph {
    /// Row pointers (offset into col_idx for each node)
    pub row_ptr: Vec<usize>,
    /// Column indices (neighbor node IDs)
    pub col_idx: Vec<u32>,
    /// Optional edge weights
    pub edge_data: Vec<f32>,
    /// Number of nodes
    pub num_nodes: usize,
}

impl CsrGraph {
    /// Create a new empty CSR graph.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes in the graph
    ///
    /// # Returns
    ///
    /// A new empty CSR graph with space allocated for `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            edge_data: Vec::new(),
            num_nodes,
        }
    }

    /// Get neighbors of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - Node ID to query
    ///
    /// # Returns
    ///
    /// Slice of neighbor node IDs
    ///
    /// # Panics
    ///
    /// Panics if `node` is out of bounds.
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        &self.col_idx[start..end]
    }

    /// Get edge weights for a node's neighbors.
    ///
    /// # Arguments
    ///
    /// * `node` - Node ID to query
    ///
    /// # Returns
    ///
    /// Slice of edge weights corresponding to neighbors
    pub fn edge_weights(&self, node: u32) -> &[f32] {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        &self.edge_data[start..end]
    }

    /// Get number of nodes.
    pub fn len(&self) -> usize {
        self.num_nodes
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Get total number of edges.
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Get degree (number of neighbors) of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - Node ID to query
    ///
    /// # Returns
    ///
    /// Number of neighbors for the node
    pub fn degree(&self, node: u32) -> usize {
        if node as usize >= self.num_nodes {
            return 0;
        }
        self.row_ptr[node as usize + 1] - self.row_ptr[node as usize]
    }
}

/// Builder for constructing CSR graphs.
pub struct CsrGraphBuilder {
    /// Adjacency lists for each node
    adj_lists: Vec<Vec<(u32, f32)>>,
}

impl CsrGraphBuilder {
    /// Create a new CSR graph builder.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes in the graph
    pub fn new(num_nodes: usize) -> Self {
        Self {
            adj_lists: vec![Vec::new(); num_nodes],
        }
    }

    /// Add an edge to the graph.
    ///
    /// # Arguments
    ///
    /// * `src` - Source node ID
    /// * `dst` - Destination node ID
    /// * `weight` - Edge weight
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, error if nodes are out of bounds
    pub fn add_edge(&mut self, src: u32, dst: u32, weight: f32) -> Result<()> {
        let src_idx = src as usize;
        if src_idx >= self.adj_lists.len() {
            return Err(VyaktiError::InvalidInput(format!(
                "Source node {} out of bounds (max {})",
                src,
                self.adj_lists.len() - 1
            )));
        }

        self.adj_lists[src_idx].push((dst, weight));
        Ok(())
    }

    /// Add multiple edges from a node.
    ///
    /// # Arguments
    ///
    /// * `src` - Source node ID
    /// * `neighbors` - Slice of (destination, weight) pairs
    pub fn add_edges(&mut self, src: u32, neighbors: &[(u32, f32)]) -> Result<()> {
        let src_idx = src as usize;
        if src_idx >= self.adj_lists.len() {
            return Err(VyaktiError::InvalidInput(format!(
                "Source node {} out of bounds (max {})",
                src,
                self.adj_lists.len() - 1
            )));
        }

        self.adj_lists[src_idx].extend_from_slice(neighbors);
        Ok(())
    }

    /// Build the CSR graph from the adjacency lists.
    ///
    /// # Returns
    ///
    /// The constructed CSR graph
    pub fn build(self) -> CsrGraph {
        let num_nodes = self.adj_lists.len();
        let mut row_ptr = Vec::with_capacity(num_nodes + 1);
        let mut col_idx = Vec::new();
        let mut edge_data = Vec::new();

        row_ptr.push(0);

        for adj_list in self.adj_lists {
            for (dst, weight) in adj_list {
                col_idx.push(dst);
                edge_data.push(weight);
            }
            row_ptr.push(col_idx.len());
        }

        CsrGraph {
            row_ptr,
            col_idx,
            edge_data,
            num_nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph_new() {
        let graph = CsrGraph::new(10);
        assert_eq!(graph.len(), 10);
        assert_eq!(graph.num_edges(), 0);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_csr_graph_empty() {
        let graph = CsrGraph::new(0);
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_csr_graph_builder_simple() {
        let mut builder = CsrGraphBuilder::new(3);
        builder.add_edge(0, 1, 1.0).unwrap();
        builder.add_edge(0, 2, 2.0).unwrap();
        builder.add_edge(1, 2, 3.0).unwrap();

        let graph = builder.build();

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.num_edges(), 3);

        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.edge_weights(0), &[1.0, 2.0]);

        assert_eq!(graph.neighbors(1), &[2]);
        assert_eq!(graph.edge_weights(1), &[3.0]);

        let empty: &[u32] = &[];
        assert_eq!(graph.neighbors(2), empty);
    }

    #[test]
    fn test_csr_graph_builder_add_edges() {
        let mut builder = CsrGraphBuilder::new(2);
        builder.add_edges(0, &[(1, 1.5), (1, 2.5)]).unwrap();

        let graph = builder.build();

        assert_eq!(graph.neighbors(0), &[1, 1]);
        assert_eq!(graph.edge_weights(0), &[1.5, 2.5]);
    }

    #[test]
    fn test_csr_graph_builder_out_of_bounds() {
        let mut builder = CsrGraphBuilder::new(2);
        let result = builder.add_edge(5, 1, 1.0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }

    #[test]
    fn test_csr_graph_degree() {
        let mut builder = CsrGraphBuilder::new(3);
        builder.add_edge(0, 1, 1.0).unwrap();
        builder.add_edge(0, 2, 1.0).unwrap();
        builder.add_edge(1, 2, 1.0).unwrap();

        let graph = builder.build();

        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 0);
        assert_eq!(graph.degree(10), 0); // Out of bounds
    }

    #[test]
    fn test_csr_graph_large() {
        let mut builder = CsrGraphBuilder::new(1000);

        // Add edges from each node to the next 5 nodes
        for i in 0..995 {
            for j in 1..=5 {
                builder.add_edge(i, i + j, j as f32).unwrap();
            }
        }

        let graph = builder.build();

        assert_eq!(graph.len(), 1000);
        assert_eq!(graph.num_edges(), 995 * 5);

        // Verify first node
        assert_eq!(graph.neighbors(0), &[1, 2, 3, 4, 5]);
        assert_eq!(graph.edge_weights(0), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_csr_graph_serialization() {
        let mut builder = CsrGraphBuilder::new(3);
        builder.add_edge(0, 1, 1.0).unwrap();
        builder.add_edge(1, 2, 2.0).unwrap();

        let graph = builder.build();

        // Test JSON serialization
        let json = serde_json::to_string(&graph).unwrap();
        let deserialized: CsrGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.len(), graph.len());
        assert_eq!(deserialized.num_edges(), graph.num_edges());
        assert_eq!(deserialized.neighbors(0), graph.neighbors(0));
        assert_eq!(deserialized.edge_weights(1), graph.edge_weights(1));
    }

    #[test]
    fn test_csr_graph_bincode_serialization() {
        let mut builder = CsrGraphBuilder::new(5);
        for i in 0..4 {
            builder.add_edge(i, i + 1, (i as f32) * 1.5).unwrap();
        }

        let graph = builder.build();

        // Test bincode serialization
        let bytes = bincode::serialize(&graph).unwrap();
        let deserialized: CsrGraph = bincode::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.len(), graph.len());
        assert_eq!(deserialized.num_edges(), graph.num_edges());
        assert_eq!(deserialized.row_ptr, graph.row_ptr);
        assert_eq!(deserialized.col_idx, graph.col_idx);
        assert_eq!(deserialized.edge_data, graph.edge_data);
    }

    #[test]
    fn test_csr_graph_clone() {
        let mut builder = CsrGraphBuilder::new(2);
        builder.add_edge(0, 1, 1.0).unwrap();

        let graph = builder.build();
        let cloned = graph.clone();

        assert_eq!(cloned.len(), graph.len());
        assert_eq!(cloned.neighbors(0), graph.neighbors(0));
    }

    #[test]
    fn test_csr_graph_builder_add_edges_out_of_bounds() {
        let mut builder = CsrGraphBuilder::new(2);
        let result = builder.add_edges(5, &[(1, 1.0)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out of bounds"));
    }
}
