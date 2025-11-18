//! Embedding provider implementations

#[cfg(feature = "llama-cpp")]
pub mod llama_cpp;

pub mod mock;

// Re-exports
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{LlamaCppConfig, LlamaCppProvider};

pub use mock::{OllamaConfig, OllamaProvider, OpenAIConfig, OpenAIProvider};
