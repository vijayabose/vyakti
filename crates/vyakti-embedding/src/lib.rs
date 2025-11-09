//! Embedding computation layer for Vyakti.
//!
//! Supports multiple embedding providers:
//! - Local models via ONNX Runtime
//! - OpenAI API
//! - Ollama
//! - Custom providers

pub mod providers;
pub mod cache;

pub use providers::*;
