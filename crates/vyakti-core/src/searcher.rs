//! Index searcher.

use vyakti_common::Result;

/// Searcher for querying Vyakti indexes
pub struct VyaktiSearcher {
    _private: (),
}

impl VyaktiSearcher {
    /// Create a new searcher
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for VyaktiSearcher {
    fn default() -> Self {
        Self::new()
    }
}
