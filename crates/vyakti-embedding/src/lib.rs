//! Embedding computation layer for Vyakti.
//!
//! Supports multiple embedding providers:
//! - Local models via ONNX Runtime
//! - OpenAI API
//! - Ollama
//! - Custom providers

pub mod cache;
pub mod providers;
pub mod recomputation;

pub use cache::EmbeddingCache;
pub use providers::*;
pub use recomputation::{
    EmbeddingRecomputationService, RecomputationConfig, RecomputationStats,
};
