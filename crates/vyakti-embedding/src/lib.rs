//! Embedding computation layer for Vyakti.
//!
//! Supports multiple embedding providers:
//! - Local models via llama.cpp (default)
//! - Local models via ONNX Runtime
//! - OpenAI API
//! - Custom providers

pub mod providers;
pub mod cache;

#[cfg(feature = "llama-cpp")]
pub mod download;

pub use providers::*;

#[cfg(feature = "llama-cpp")]
pub use download::{
    download_default_model, download_model, ensure_model, get_model_path, get_models_dir,
    model_exists, DEFAULT_MODEL_FILE, DEFAULT_MODEL_NAME, DEFAULT_MODEL_REPO,
};
