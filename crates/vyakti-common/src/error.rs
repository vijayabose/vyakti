//! Error types for Vyakti.

use std::fmt;

/// Main error type for Vyakti operations.
#[derive(Debug, thiserror::Error)]
pub enum VyaktiError {
    /// Index not found
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Embedding computation error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Storage/IO error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type alias using VyaktiError.
pub type Result<T> = std::result::Result<T, VyaktiError>;
