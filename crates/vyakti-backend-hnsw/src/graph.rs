//! HNSW graph structure.

/// HNSW graph implementation
pub struct HnswGraph {
    _private: (),
}

impl HnswGraph {
    /// Create a new HNSW graph
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for HnswGraph {
    fn default() -> Self {
        Self::new()
    }
}
