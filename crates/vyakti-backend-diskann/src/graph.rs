//! DiskANN graph structure.

/// DiskANN graph implementation
pub struct DiskAnnGraph {
    _private: (),
}

impl DiskAnnGraph {
    /// Create a new DiskANN graph
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for DiskAnnGraph {
    fn default() -> Self {
        Self::new()
    }
}
