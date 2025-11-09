//! Storage layer for Vyakti vector database.
//!
//! Provides efficient storage formats including:
//! - CSR (Compressed Sparse Row) graph format
//! - Memory-mapped index files
//! - Serialization/deserialization

pub mod csr;
pub mod index;

pub use csr::CsrGraph;
pub use index::*;
