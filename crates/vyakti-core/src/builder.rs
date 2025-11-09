//! Index builder.

use vyakti_common::Result;

/// Builder for creating Vyakti indexes
pub struct VyaktiBuilder {
    _private: (),
}

impl VyaktiBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for VyaktiBuilder {
    fn default() -> Self {
        Self::new()
    }
}
