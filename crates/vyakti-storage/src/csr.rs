//! Compressed Sparse Row (CSR) graph format.

use vyakti_common::Result;

/// CSR graph representation for efficient storage.
#[derive(Debug, Clone)]
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
    /// Create a new empty CSR graph
    pub fn new(num_nodes: usize) -> Self {
        Self {
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            edge_data: Vec::new(),
            num_nodes,
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let start = self.row_ptr[node as usize];
        let end = self.row_ptr[node as usize + 1];
        &self.col_idx[start..end]
    }

    /// Get number of nodes
    pub fn len(&self) -> usize {
        self.num_nodes
    }

    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }
}
