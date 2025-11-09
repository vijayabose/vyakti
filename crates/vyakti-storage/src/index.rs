//! Index file format and serialization.

use vyakti_common::Result;

/// Index header structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IndexHeader {
    /// Magic bytes: "VYAK"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of vectors
    pub num_vectors: u64,
    /// Vector dimension
    pub dimension: u32,
    /// Backend type identifier
    pub backend_type: u32,
    /// Flags (compact, recompute, etc.)
    pub flags: u64,
}

impl IndexHeader {
    /// Create a new index header
    pub fn new(num_vectors: usize, dimension: usize) -> Self {
        Self {
            magic: *b"VYAK",
            version: 1,
            num_vectors: num_vectors as u64,
            dimension: dimension as u32,
            backend_type: 0,
            flags: 0,
        }
    }

    /// Validate magic bytes
    pub fn is_valid(&self) -> bool {
        &self.magic == b"VYAK"
    }
}
