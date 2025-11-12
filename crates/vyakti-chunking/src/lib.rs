//! Text chunking utilities for Vyakti
//!
//! This crate provides text chunking capabilities similar to LEANN:
//! - Traditional sentence-based chunking with token awareness
//! - AST-aware code chunking (optional feature)
//! - Configurable chunk size and overlap
//!
//! # Examples
//!
//! ```no_run
//! use vyakti_chunking::{TextChunker, ChunkConfig};
//!
//! let config = ChunkConfig {
//!     chunk_size: 256,
//!     chunk_overlap: 128,
//!     ..Default::default()
//! };
//!
//! let chunker = TextChunker::new(config).unwrap();
//! let text = "Your long document text here...";
//! let chunks = chunker.chunk(text);
//! ```

pub mod chunker;
pub mod token;

#[cfg(feature = "ast")]
pub mod ast;

#[cfg(any(feature = "text-formats", feature = "document-formats"))]
pub mod file_reader;

pub use chunker::{ChunkConfig, ChunkResult, TextChunker};
pub use token::{estimate_tokens, TokenEstimator};

#[cfg(feature = "ast")]
pub use ast::{AstChunker, CodeLanguage};

#[cfg(any(feature = "text-formats", feature = "document-formats"))]
pub use file_reader::FileReader;

/// Error type for chunking operations
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("Invalid chunk configuration: {0}")]
    InvalidConfig(String),

    #[error("Text processing error: {0}")]
    ProcessingError(String),

    #[cfg(feature = "ast")]
    #[error("AST parsing error: {0}")]
    AstError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ChunkError>;
