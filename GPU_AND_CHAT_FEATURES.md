# GPU Testing and Chat Features

## Overview

This document describes the GPU testing capabilities and RAG (Retrieval-Augmented Generation) chat functionality added to Vyakti.

## 1. GPU Testing

### Purpose
Validate that llama.cpp embedding provider works correctly with different GPU configurations, ensuring:
- CPU-only mode works (default)
- GPU acceleration can be configured
- System gracefully handles GPU availability

### Test Files

**Location**: `crates/vyakti-embedding/tests/test_llama_cpp_gpu.rs`

### Test Coverage

#### 1. GPU Layers Configuration Test
```bash
cargo test --package vyakti-embedding --test test_llama_cpp_gpu test_gpu_layers_configuration -- --ignored
```

Tests that the provider accepts different `n_gpu_layers` settings:
- 0 layers (CPU-only)
- 16 layers (partial offload)
- 32 layers (moderate offload)
- 64 layers (high offload)

Verifies:
- ✅ Provider initialization succeeds for all configurations
- ✅ Embeddings generate correctly with 1024 dimensions
- ✅ Normalization works properly (L2 norm ~1.0)

#### 2. GPU Acceleration Test
```bash
cargo test --package vyakti-embedding --test test_llama_cpp_gpu test_gpu_acceleration -- --ignored
```

Tests full GPU offload (999 layers) with fallback to CPU if GPU unavailable.

Verifies:
- ✅ Provider handles large GPU layer counts gracefully
- ✅ Batch embedding works with GPU configuration
- ✅ Falls back to CPU if CUDA not available

#### 3. CPU-Only Mode Test
```bash
cargo test --package vyakti-embedding --test test_llama_cpp_gpu test_cpu_only_mode -- --ignored
```

Explicitly tests CPU-only mode (0 GPU layers).

Verifies:
- ✅ CPU-only mode works correctly
- ✅ Uses all available CPU cores efficiently
- ✅ Generates correct embeddings

#### 4. CPU vs GPU Benchmark
```bash
cargo test --package vyakti-embedding --test test_llama_cpp_gpu benchmark_cpu_vs_gpu -- --ignored
```

Performance comparison between CPU and GPU modes.

Measures:
- CPU-only performance (avg time per embedding)
- GPU-accelerated performance (32 layers)
- Speedup factor (if GPU available)

### CLI Integration

GPU testing is integrated into the CLI via `--gpu-layers` flag:

```bash
# CPU-only (default)
vyakti build my-docs --input ./documents

# GPU acceleration (offload 32 layers)
vyakti build my-docs --input ./documents --gpu-layers 32

# Maximum GPU offload
vyakti build my-docs --input ./documents --gpu-layers 999
```

### Running Tests

```bash
# Run all GPU tests (requires model download)
cargo test --package vyakti-embedding --test test_llama_cpp_gpu -- --ignored

# Run specific test
cargo test --package vyakti-embedding --test test_llama_cpp_gpu test_gpu_layers_configuration -- --ignored

# Run with output
cargo test --package vyakti-embedding --test test_llama_cpp_gpu -- --ignored --nocapture
```

### GPU Requirements

**For GPU Acceleration:**
- CUDA-capable NVIDIA GPU
- CUDA Toolkit installed
- llama.cpp compiled with CUDA support

**Without GPU:**
- Tests will pass but use CPU fallback
- Performance benchmarks will show similar CPU/GPU times

## 2. RAG Chat Functionality

### Purpose
Enable question-answering over indexed documents using Retrieval-Augmented Generation (RAG).

### Architecture

```
User Question
     ↓
Vector Search (retrieve relevant documents)
     ↓
Build Context (format retrieved documents)
     ↓
LLM Generation (answer based on context)
     ↓
Response to User
```

### Components

#### Chat Session (`vyakti-core/src/chat.rs`)

**`ChatSession`**: Maintains conversation history for multi-turn chat
- Retrieves relevant documents for each question
- Builds context from search results
- Sends context + conversation history to LLM
- Maintains conversation state

**`ask_question()`**: One-shot question answering without history
- Simpler API for single Q&A
- No conversation state
- Good for batch processing

### API Usage

```rust
use vyakti_core::{ChatSession, VyaktiSearcher, ask_question};
use vyakti_common::{GenerationConfig, TextGenerationProvider};
use std::sync::Arc;

// One-shot Q&A
let searcher = VyaktiSearcher::load(&index_path, backend, embedding_provider).await?;
let generator = Arc::new(MyLLMProvider::new());
let config = GenerationConfig::default();

let answer = ask_question(
    &searcher,
    generator,
    "What is vector search?",
    5, // top-k documents
    &config
).await?;

// Multi-turn chat
let mut session = ChatSession::new(searcher, generator, 5);
session.add_system_message("You are a helpful AI assistant.".to_string());

let response1 = session.ask("What is LEANN?", &config).await?;
let response2 = session.ask("How does it save storage?", &config).await?;

// Get conversation history
let history = session.history();
```

### LLM Provider Integration

The chat module uses the `TextGenerationProvider` trait defined in `vyakti-common`:

```rust
pub trait TextGenerationProvider: Send + Sync {
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;
    async fn chat(&self, messages: &[(String, String)], config: &GenerationConfig) -> Result<String>;
    fn name(&self) -> &str;
}
```

**Supported Providers** (can be implemented):
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic API (Claude)
- Ollama (local LLMs)
- Azure OpenAI
- Custom LLMs

### Configuration

```rust
use vyakti_common::GenerationConfig;

let config = GenerationConfig {
    max_tokens: 512,            // Maximum response length
    temperature: 0.7,            // Sampling temperature (0.0-2.0)
    top_p: 0.9,                  // Nucleus sampling parameter
    n_threads: 4,                // Thread count for local LLMs
    stop_sequences: vec![],      // Stop generation at these sequences
};
```

### Example: OpenAI Integration

```rust
use vyakti_common::{TextGenerationProvider, GenerationConfig, Result};
use async_trait::async_trait;

struct OpenAIProvider {
    api_key: String,
    model: String,
}

#[async_trait]
impl TextGenerationProvider for OpenAIProvider {
    async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // Call OpenAI API
        let response = reqwest::Client::new()
            .post("https://api.openai.com/v1/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["choices"][0]["text"].as_str().unwrap().to_string())
    }

    fn name(&self) -> &str {
        "openai"
    }
}
```

## Future Enhancements

### GPU Testing
1. **Automatic GPU Detection**: Auto-configure optimal GPU layers based on available VRAM
2. **Multi-GPU Support**: Test distribution across multiple GPUs
3. **Memory Benchmarks**: Track VRAM usage during inference
4. **CI Integration**: Set up GPU runners for automated testing

### Chat Functionality
1. **CLI Command**: Add `vyakti ask` command for interactive Q&A
2. **Streaming Responses**: Support streaming for long responses
3. **Citation**: Add document citations to generated responses
4. **Re-ranking**: Improve retrieval with cross-encoder re-ranking
5. **Conversation Export**: Save/load conversation history
6. **Multi-query**: Generate multiple search queries for better retrieval

## Testing

### GPU Tests
All GPU tests are marked with `#[ignore]` to avoid mandatory model downloads in CI:

```bash
# Run all ignored GPU tests
cargo test --package vyakti-embedding -- --ignored

# Run specific GPU test
cargo test --package vyakti-embedding test_gpu_layers_configuration -- --ignored
```

### Chat Tests
Integration tests for chat functionality would require:
1. Mock LLM provider (for unit tests)
2. Real index with sample documents
3. Sample questions and expected answers

Example test structure:
```rust
#[tokio::test]
async fn test_chat_session() {
    let searcher = create_test_searcher().await;
    let generator = Arc::new(MockLLMProvider::new());
    let mut session = ChatSession::new(searcher, generator, 5);

    let response = session.ask("test question", &GenerationConfig::default()).await.unwrap();
    assert!(!response.is_empty());
}
```

## Documentation References

- **llama.cpp GPU support**: https://github.com/ggerganov/llama.cpp#cuda
- **RAG patterns**: Retrieval-Augmented Generation architecture
- **OpenAI API**: https://platform.openai.com/docs/api-reference

## Summary

### Completed
✅ GPU testing framework with comprehensive test coverage
✅ RAG chat module with conversation history support
✅ `TextGenerationProvider` trait for LLM integration
✅ Flexible configuration system for generation parameters
✅ Documentation and usage examples

### Remaining
⏭️ CLI `ask` command implementation
⏭️ OpenAI/Ollama provider implementations
⏭️ Integration tests for chat functionality
⏭️ Streaming response support
⏭️ Citation and re-ranking features
