# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the **Vyakti** project - a Rust migration of the LEANN (Low-Storage Vector Index) vector database. The goal is to achieve 10-100x performance improvements while maintaining the core innovation of 97% storage savings through graph-based selective recomputation.

### Project Structure

```
vyakti/
├── README.md                   # Main project README (Rust implementation)
├── CLAUDE.md                   # This file (development guidance)
├── document/
│   ├── BRD.md                 # Business Requirements Document
│   └── MODULAR_DESIGN.md      # Architectural design document
└── LEANN/                      # Original Python implementation (reference)
    └── (Python codebase)
```

## Development Philosophy

**Mission**: Migrate LEANN from Python to Rust to create a production-ready, enterprise-grade vector database that can be used as:
1. **Library** - Import into Rust projects
2. **CLI Tool** - Standalone command-line application
3. **Server** - Network-accessible REST/gRPC service

**Key Principles**:
- **Performance First**: Leverage Rust's zero-cost abstractions
- **Memory Safety**: Use Rust's ownership model to prevent bugs
- **Modularity**: Clear separation of concerns with plugin architecture
- **Flexibility**: Support multiple deployment modes from single codebase
- **Production Ready**: Built for enterprise deployment with observability

## Reference: Original Python LEANN

This repository contains the original Python implementation in the `LEANN/` directory for reference during migration.

### Core Packages (Monorepo)

- **`packages/vyakti-core/`** - Core API, CLI, embedding management, and plugin system
- **`packages/vyakti-backend-hnsw/`** - HNSW (Hierarchical Navigable Small World) backend implementation
- **`packages/vyakti-backend-diskann/`** - DiskANN backend with PQ-based graph traversal
- **`packages/leann-mcp/`** - MCP (Model Context Protocol) server for Claude Code integration
- **`packages/astchunk-leann/`** - AST-aware code chunking for Python, Java, C#, TypeScript
- **`packages/wechat-exporter/`** - WeChat data extraction utility

### Applications (`apps/`)

RAG (Retrieval-Augmented Generation) applications demonstrating LEANN's capabilities:
- **Document RAG** (`document_rag.py`) - Process PDFs, TXT, MD files
- **Email RAG** (`email_rag.py`) - Index Apple Mail (macOS only)
- **Browser RAG** (`browser_rag.py`) - Search Chrome history
- **Chat History RAG** - ChatGPT (`chatgpt_rag.py`), Claude (`claude_rag.py`), iMessage (`imessage_rag.py`), WeChat (`wechat_rag.py`)
- **MCP Integration** - Slack (`slack_rag.py`), Twitter (`twitter_rag.py`) via Model Context Protocol
- **Code RAG** (`code_rag.py`) - AST-aware code search

Each app follows the pattern in `base_rag_example.py`.

### Benchmarks (`benchmarks/`)

Performance evaluation scripts:
- `run_evaluation.py` - Main benchmark runner (auto-downloads datasets)
- `compare_faiss_vs_leann.py` - Storage comparison
- `diskann_vs_hnsw_speed_comparison.py` - Backend comparison
- Dataset-specific: `enron_emails/`, `financebench/`, `laion/`

## Development Commands

### Environment Setup

```bash
# Install uv package manager (required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone with submodules
git clone https://github.com/yichuan-w/LEANN.git
git submodule update --init --recursive

# Install dependencies (macOS)
brew install libomp boost protobuf zeromq pkgconf
uv sync --extra diskann

# Install dependencies (Ubuntu/Debian)
sudo apt-get install libomp-dev libboost-all-dev protobuf-compiler \
  libzmq3-dev pkg-config libabsl-dev libaio-dev libprotobuf-dev libmkl-full-dev
uv sync --extra diskann

# Activate virtual environment
source .venv/bin/activate
```

### Building & Testing

```bash
# Run tests
uv run pytest                           # All tests
uv run pytest tests/test_basic.py       # Single test file
uv run pytest --cov=leann              # With coverage

# Format code (uses ruff)
ruff format                            # Format all files
ruff format --check                    # Check without modifying

# Lint code
ruff check --fix                       # Auto-fix issues
ruff check                             # Check only

# Pre-commit hooks (runs on commit)
pre-commit install                     # One-time setup
pre-commit run --all-files             # Manual run
```

### Running Examples

```bash
# Document RAG (easiest to start)
python -m apps.document_rag --query "What techniques does LEANN use?"

# Browser history
python -m apps.browser_rag --query "machine learning articles"

# Email search (requires Full Disk Access on macOS)
python -m apps.email_rag --query "DoorDash orders"

# Code search with AST-aware chunking
python -m apps.code_rag --repo-dir ./src --query "authentication logic"
```

### CLI Usage

```bash
# Install CLI globally
uv tool install vyakti-core --with leann

# Build index
leann build my-docs --docs ./documents

# Search
leann search my-docs "vector database concepts"

# Interactive chat
leann ask my-docs --interactive

# List all indexes
leann list

# Remove index
leann remove my-docs
```

## Architecture Patterns

### Backend Plugin System

LEANN uses a plugin architecture for backends via `interface.py`:

- **`LeannBackendBuilderInterface`** - Build indexes from vectors
- **`LeannBackendSearcherInterface`** - Search with recomputation support
- **`LeannBackendFactoryInterface`** - Create builder/searcher instances

Register backends in `registry.py` (`BACKEND_REGISTRY` dict).

### Embedding Computation

**Key files:**
- `api.py:compute_embeddings()` - Main embedding function supporting multiple modes
- `embedding_server_manager.py` - ZMQ-based embedding server lifecycle
- Backend-specific servers: `hnsw_embedding_server.py`, `diskann_embedding_server.py`

**Python (Reference) Modes:**
- `sentence-transformers` - Local HuggingFace models (default)
- `openai` - OpenAI API or compatible endpoints
- `ollama` - Local Ollama server
- `mlx` - Apple Silicon optimized (ARM64 macOS only)

**Rust (Vyakti) Providers:**
- `llama-cpp` - Local models via llama.cpp (default, auto-downloads mxbai-embed-large)
- `onnx` - ONNX Runtime for local models
- Future: OpenAI API, custom providers

### Graph-Based Recomputation

**Core concept:** Instead of storing all embeddings, LEANN:
1. Stores a pruned graph structure (CSR format)
2. Recomputes embeddings on-demand during search for nodes in the search path
3. Uses high-degree preserving pruning to maintain hub nodes

**Implementation:**
- `hnsw_backend.py` - HNSW graph traversal with recomputation
- `diskann_backend.py` - DiskANN with PQ compression + reranking
- `convert_to_csr.py` - Graph pruning and CSR conversion

### RAG Application Pattern

All RAG apps follow this structure (see `base_rag_example.py`):

```python
1. Parse arguments (common params + app-specific)
2. Load/export data (app-specific readers in *_data/ dirs)
3. Build index: LeannBuilder → add_text() → build_index()
4. Search/Chat: LeannSearcher or LeannChat
```

Common parameters: `--embedding-model`, `--llm`, `--top-k`, `--chunk-size`, `--backend-name`

## Important Configuration

### Embedding Models

**Small (fast):** `all-MiniLM-L6-v2` (22M params)
**Medium (balanced):** `facebook/contriever` (110M, default), `BAAI/bge-base-en-v1.5`
**Large (quality):** `Qwen/Qwen3-Embedding-0.6B` (600M), `intfloat/multilingual-e5-large`

Set via: `--embedding-mode sentence-transformers --embedding-model <model>`

### Backend Selection

**HNSW (default):** Maximum storage savings, good for most datasets
**DiskANN:** Better search performance, uses PQ compression with reranking

Set via: `--backend-name hnsw` or `--backend-name diskann`

### Index Parameters

- `--graph-degree` (default: 32) - Higher = better recall, more storage
- `--build-complexity` (default: 64) - Higher = better graph quality
- `--search-complexity` (default: 32) - Higher = more accurate search
- `--compact/--no-compact` (default: compact) - Compressed storage vs full storage
- `--recompute/--no-recompute` (default: recompute) - Enable/disable recomputation

## Testing & CI

### Test Organization

- `test_basic.py` - Core API functionality
- `test_ci_minimal.py` - Fast CI smoke tests
- `test_mcp_*.py` - MCP server integration
- `test_*_rag.py` - RAG application tests
- `test_metadata_filtering.py` - Metadata filter engine

### CI Pipeline (`.github/workflows/`)

**`build-and-publish.yml`** - Main CI:
- Runs on Ubuntu & macOS
- Python 3.9-3.13 matrix
- Linting with ruff
- Builds wheels for distribution

**`build-reusable.yml`** - Shared build workflow
**`link-check.yml`** - Documentation link validation
**`release-manual.yml`** - Manual release trigger

### Pre-commit Checks

Configured in `.pre-commit-config.yaml`:
- Trailing whitespace removal
- EOF fixing
- YAML validation
- Large file prevention
- ruff formatting + linting

## Common Development Tasks

### Adding a New Backend

1. Create `packages/leann-backend-<name>/`
2. Implement `LeannBackendBuilderInterface` and `LeannBackendSearcherInterface`
3. Create factory class implementing `LeannBackendFactoryInterface`
4. Register in `packages/vyakti-core/src/leann/registry.py`
5. Add to `pyproject.toml` optional dependencies

### Adding a New RAG Application

1. Create `apps/<name>_rag.py` following `base_rag_example.py`
2. Create data reader in `apps/<name>_data/<name>_reader.py`
3. Add app-specific arguments and data loading logic
4. Use `LeannBuilder` → `LeannSearcher`/`LeannChat` pattern
5. Add example to README.md

### Adding a New Embedding Mode

1. Add mode to `api.py:compute_embeddings()` function
2. Implement embedding logic or API client
3. Handle batching and normalization
4. Update docs in `docs/configuration-guide.md`

### Running Benchmarks

```bash
# Auto-download datasets and run
uv run benchmarks/run_evaluation.py

# Specific dataset (after download)
uv run benchmarks/run_evaluation.py benchmarks/data/indices/rpj_wiki/rpj_wiki --num-queries 2000

# Compare FAISS vs LEANN
uv run benchmarks/compare_faiss_vs_leann.py

# Backend comparison
uv run benchmarks/diskann_vs_hnsw_speed_comparison.py
```

## MCP Integration

LEANN provides an MCP server for Claude Code integration:

```bash
# Install globally
uv tool install vyakti-core --with leann

# Register with Claude Code
claude mcp add --scope user vyakti-server -- leann_mcp

# Build index for code
leann build my-project --docs $(git ls-files)

# Claude Code can now use leann_search and leann_list tools
```

See `packages/leann-mcp/README.md` for detailed setup.

## Key Technical Details

### CSR (Compressed Sparse Row) Graph Storage

- Converts dense adjacency to CSR format for memory efficiency
- See `convert_to_csr.py:prune_hnsw_embeddings_inplace()`
- Enables compact storage mode

### Metadata Filtering

- SQL-like operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `contains`, etc.
- Implemented in `metadata_filter.py:MetadataFilterEngine`
- Used in `LeannSearcher.search(metadata_filters={...})`

### AST-Aware Chunking

- Language-aware code chunking preserving semantic boundaries
- Supports Python, Java, C#, TypeScript
- Package: `packages/astchunk-leann/`
- Used via `--enable-code-chunking` flag

### Grep Search

- Exact text matching fallback when semantic search isn't needed
- Use via `searcher.search(query, use_grep=True)`
- See `docs/grep_search.md`

## Debugging Tips

### Slow Embedding Computation

- Check GPU availability: `torch.cuda.is_available()`
- Use smaller model for testing: `--embedding-model all-MiniLM-L6-v2`
- Enable OpenAI mode for cloud compute: `--embedding-mode openai`

### Index Build Failures

- Ensure `uv sync --extra diskann` completed successfully
- Check system dependencies are installed
- Verify sufficient disk space (indexes can be large)
- Use `--max-items N` to limit dataset size for testing

### Search Quality Issues

- Increase `--search-complexity` (default: 32 → try 64+)
- Use better embedding model: `--embedding-model Qwen/Qwen3-Embedding-0.6B`
- Adjust `--chunk-size` and `--chunk-overlap` for your data
- Try `--no-compact` mode (stores full embeddings, no recomputation)

### MCP Connection Issues

- Verify global install: `leann --version`
- Check MCP registration: `claude mcp list | cat`
- Rebuild index if format changed: `leann remove <index> && leann build <index> ...`

## Documentation

- `README.md` - Main documentation and examples
- `docs/configuration-guide.md` - Parameter tuning and optimization
- `docs/CONTRIBUTING.md` - Development workflow and style guide
- `docs/metadata_filtering.md` - Metadata filter syntax and examples
- `docs/ast_chunking_guide.md` - AST-aware code chunking details
- `docs/grep_search.md` - Exact text search guide
- `docs/features.md` - Detailed feature list
- `docs/roadmap.md` - Future development plans

## External Links

- Paper: https://arxiv.org/abs/2506.08276
- PyPI: https://pypi.org/project/leann/
- Issues: https://github.com/yichuan-w/LEANN/issues

---

## Rust Migration: Vyakti

### Architecture Overview

Vyakti follows a modular crate-based architecture:

```
crates/
├── vyakti-common/          # Shared types, errors, traits
├── vyakti-storage/         # CSR format, memory mapping, serialization
├── vyakti-embedding/       # Embedding models and computation
├── vyakti-core/            # Main API (Builder, Searcher, Chat)
├── vyakti-backend-hnsw/    # HNSW algorithm implementation
├── vyakti-backend-diskann/ # DiskANN algorithm with PQ compression
├── vyakti-keyword/         # BM25 keyword search for hybrid mode
├── vyakti-proto/           # Protocol buffer definitions
├── vyakti-server/          # REST + gRPC server
└── vyakti-cli/             # Command-line interface
```

**Key Features:**
- **Hybrid Search** - Combines semantic vector search with BM25 keyword search
- **LEANN Compact Mode** - 93% storage savings through intelligent pruning
- **Multi-Format Support** - 35+ file formats including code, docs, PDFs
- **MCP Integration** - Native Model Context Protocol server for Claude Code
- **Production Ready** - REST API, authentication, observability

### Development Commands (Rust)

#### Environment Setup

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install system dependencies (macOS)
brew install llvm libomp cmake

# Install system dependencies (Ubuntu)
sudo apt-get install build-essential cmake clang libomp-dev pkg-config

# Clone and build
git clone <repo-url>
cd vyakti
cargo build --workspace
```

#### Build and Test

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run all tests
cargo test --workspace

# Run specific test
cargo test --package vyakti-core --test test_builder

# Test with coverage
cargo tarpaulin --workspace --out Html

# Format code
cargo fmt --all

# Lint code
cargo clippy --all-targets --all-features -- -D warnings

# Check for security issues
cargo audit
```

#### Running Examples

```bash
# Run example from crate
cargo run --example basic_search --package vyakti-core

# Run CLI tool
cargo run --bin vyakti-cli -- build my-docs --input ./documents

# Start server
cargo run --bin vyakti-server -- --port 8080
```

#### Benchmarking

```bash
# Run all benchmarks
cargo bench --workspace

# Specific benchmark
cargo bench --bench search_performance

# With flamegraph profiling
cargo flamegraph --bench search_performance
```

### Embedding Provider: llama.cpp (Default)

Vyakti uses **llama.cpp** as the default embedding provider with automatic model management:

#### Features
- **Automatic Model Download**: Downloads mxbai-embed-large-v1 from HuggingFace on first use
- **GPU Acceleration**: Supports offloading layers to GPU via `--gpu-layers` flag
- **CPU Optimized**: Efficient multi-threaded CPU inference
- **Zero Configuration**: Works out of the box with sensible defaults

#### Model Details
- **Default Model**: mixedbread-ai/mxbai-embed-large-v1 (Q4_K_M quantized)
- **Dimensions**: 1024
- **Size**: ~500MB
- **Storage**: `~/.vyakti/models/mxbai-embed-large-v1.q4_k_m.gguf`

#### Usage

```bash
# CPU-only (default)
vyakti build my-docs --input ./documents

# With GPU acceleration (requires CUDA)
vyakti build my-docs --input ./documents --gpu-layers 32

# With custom model
vyakti build my-docs --input ./documents --model-path /path/to/model.gguf

# Customize thread count
vyakti build my-docs --input ./documents --model-threads 8
```

#### Implementation Details

**Key Files**:
- `crates/vyakti-embedding/src/providers/llama_cpp.rs` - Provider implementation
- `crates/vyakti-embedding/src/download.rs` - Model download and caching

**Architecture**:
```rust
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub n_gpu_layers: u32,      // GPU layers
    pub n_ctx: u32,              // Context size
    pub n_threads: u32,          // CPU threads
    pub dimension: usize,        // Embedding dimension
    pub normalize: bool,         // Normalize vectors
}

pub struct LlamaCppProvider {
    sender: mpsc::UnboundedSender<EmbeddingRequest>,
    config: LlamaCppConfig,
}
```

**Model Download Flow**:
1. Check if model exists at `~/.vyakti/models/`
2. If not found, download from HuggingFace Hub using `hf-hub` crate
3. Copy to local cache directory
4. Initialize llama.cpp with the model

**Thread Safety**: Uses dedicated worker thread with message passing for thread-safe concurrent access. The model and context are owned by the worker thread, and embedding requests are sent via an unbounded channel.

#### GPU Support and Testing

Vyakti supports GPU acceleration for embedding computation via llama.cpp's CUDA backend.

**GPU Configuration**:
```bash
# CPU-only (default)
vyakti build my-docs --input ./documents

# GPU acceleration (offload 32 layers to GPU)
vyakti build my-docs --input ./documents --gpu-layers 32

# Maximum GPU offload
vyakti build my-docs --input ./documents --gpu-layers 999
```

**GPU Testing** (`crates/vyakti-embedding/tests/test_llama_cpp_gpu.rs`):
- GPU configuration validation (0, 16, 32, 64 layers)
- CPU vs GPU performance benchmarks
- Graceful fallback to CPU when GPU unavailable
- All tests marked with `#[ignore]` to avoid mandatory model downloads

Run GPU tests:
```bash
cargo test --package vyakti-embedding --test test_llama_cpp_gpu -- --ignored
```

### RAG (Retrieval-Augmented Generation) Chat

Vyakti supports question-answering over indexed documents using RAG architecture.

**Location**: `crates/vyakti-core/src/chat.rs`

**Architecture**:
1. User asks question
2. Vector search retrieves relevant documents (via `VyaktiSearcher`)
3. System builds context from retrieved documents
4. LLM generates answer based on context (via `TextGenerationProvider`)

**Key Components**:
- `ChatSession`: Multi-turn conversation with history
- `ask_question()`: One-shot Q&A without conversation state
- `TextGenerationProvider` trait: Interface for LLM integration
- `GenerationConfig`: Configure temperature, max_tokens, top_p, etc.

**Example Usage**:
```rust
use vyakti_core::{ChatSession, VyaktiSearcher, ask_question};
use vyakti_common::GenerationConfig;
use std::sync::Arc;

// One-shot Q&A
let answer = ask_question(
    &searcher,
    llm_provider,
    "What is vector search?",
    5,  // top-k documents
    &GenerationConfig::default()
).await?;

// Multi-turn chat
let mut session = ChatSession::new(searcher, llm_provider, 5);
session.add_system_message("You are a helpful assistant.".to_string());

let response1 = session.ask("What is LEANN?", &config).await?;
let response2 = session.ask("How does it save storage?", &config).await?;
```

**LLM Provider Integration**:
Implement `TextGenerationProvider` trait for your LLM:
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic API (Claude)
- Ollama (local LLMs)
- Azure OpenAI
- Custom LLMs

See `GPU_AND_CHAT_FEATURES.md` for detailed documentation.

### Code Organization Principles

#### 1. Module Boundaries

Each crate has clear responsibilities:
- **vyakti-common**: No business logic, only shared types
- **vyakti-core**: Orchestration, no backend-specific code
- **leann-backend-***: Self-contained algorithm implementations
- **vyakti-cli/server**: Interface layers, thin wrappers around core

#### 2. Trait-Based Design

Use traits for extensibility:

```rust
// Backend trait for pluggable algorithms
pub trait Backend: Send + Sync {
    async fn build(&mut self, vectors: &[Vec<f32>], config: &BackendConfig) -> Result<()>;
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
}

// Embedding provider trait
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```

#### 3. Error Handling

Use `thiserror` for error types, `anyhow` for application errors:

```rust
// In library crates
#[derive(Debug, thiserror::Error)]
pub enum LeannError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
}

// In binary crates (CLI, server)
use anyhow::{Context, Result};

fn load_index(path: &Path) -> Result<Index> {
    Index::from_path(path)
        .with_context(|| format!("Failed to load index from {}", path.display()))
}
```

#### 4. Async/Await

Use `tokio` for async runtime:

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let builder = LeannBuilder::new();
    builder.add_text("document", None).await?;
    let index = builder.build().await?;
    Ok(())
}
```

#### 5. Zero-Copy Operations

Use memory mapping for efficient I/O:

```rust
use memmap2::Mmap;

pub struct MmapIndex {
    mmap: Mmap,
    header: IndexHeader,
}

impl MmapIndex {
    pub fn vectors(&self) -> &[f32] {
        // Zero-copy access to vectors in memory-mapped file
        unsafe { std::slice::from_raw_parts(/* ... */) }
    }
}
```

#### 6. Hybrid Search

Vyakti supports hybrid search that combines semantic vector search with BM25 keyword search:

```rust
use vyakti_core::{HybridSearcher, FusionStrategy};
use vyakti_keyword::KeywordConfig;

// Build hybrid index
let keyword_config = KeywordConfig {
    enabled: true,
    k1: 1.2,  // Term frequency saturation
    b: 0.75,  // Length normalization
};

let index_path = builder
    .build_index_hybrid("my-index", Some(keyword_config))
    .await?;

// Search with fusion strategy
let searcher = HybridSearcher::load(
    &index_path,
    backend,
    embedding_provider,
    FusionStrategy::RRF { k: 60 },  // Reciprocal Rank Fusion
    documents,
)?;

let results = searcher.search("async programming", 10).await?;
```

**Available Fusion Strategies:**
- `RRF { k }` - Reciprocal Rank Fusion (default, balanced)
- `Weighted { alpha }` - Weighted combination (tunable)
- `Cascade { threshold }` - Keyword first, fallback to vector
- `VectorOnly` - Semantic search only
- `KeywordOnly` - BM25 search only

**CLI Usage:**
```bash
# Build hybrid index
vyakti build my-code --input ./src --hybrid --compact

# Search with different strategies
vyakti search my-code "auth handler" --fusion rrf
vyakti search my-code "database" --fusion weighted --fusion-param 0.7
vyakti search my-code "handleRequest" --fusion keyword-only
```

**When to use:**
- ✅ Code search (function names, identifiers)
- ✅ Technical documentation
- ✅ Mixed natural language + technical terms
- ❌ Pure natural language (vector-only is sufficient)

See [HYBRID_SEARCH.md](./HYBRID_SEARCH.md) for comprehensive documentation.

### Key Implementation Patterns

#### Builder Pattern

```rust
let searcher = LeannSearcher::builder()
    .backend(HnswBackend::default())
    .embedding_model(SentenceTransformersModel::new("all-MiniLM-L6-v2")?)
    .cache_size(1000)
    .build()?;
```

#### Strategy Pattern (Runtime Backend Selection)

```rust
let backend: Box<dyn Backend> = match backend_name {
    "hnsw" => Box::new(HnswBackend::new(config)?),
    "diskann" => Box::new(DiskAnnBackend::new(config)?),
    _ => return Err(LeannError::UnknownBackend(backend_name.to_string())),
};
```

#### Dependency Injection

```rust
pub struct LeannService {
    backend: Arc<dyn Backend>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
}

impl LeannService {
    pub fn new(
        backend: Arc<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self { backend, embedding_provider }
    }
}
```

### Testing Strategy

#### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph_creation() {
        let graph = CsrGraph::new(100);
        assert_eq!(graph.num_nodes(), 100);
    }

    #[tokio::test]
    async fn test_embedding_computation() {
        let provider = MockEmbeddingProvider::new();
        let embeddings = provider.embed(&["test"]).await.unwrap();
        assert_eq!(embeddings.len(), 1);
    }
}
```

#### Integration Tests

```rust
// tests/integration_test.rs
use leann_core::{LeannBuilder, LeannSearcher};

#[tokio::test]
async fn test_end_to_end_workflow() {
    let mut builder = LeannBuilder::new();
    builder.add_text("Document 1", None).await.unwrap();
    builder.add_text("Document 2", None).await.unwrap();

    let path = builder.build_index("test-index").await.unwrap();

    let searcher = LeannSearcher::from_path(&path).unwrap();
    let results = searcher.search("query", 10).await.unwrap();

    assert!(!results.is_empty());
}
```

#### Benchmark Tests

```rust
// benches/search_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn search_benchmark(c: &mut Criterion) {
    let searcher = setup_searcher();

    c.bench_function("search top-10", |b| {
        b.iter(|| {
            searcher.search(black_box("query"), black_box(10))
        })
    });
}

criterion_group!(benches, search_benchmark);
criterion_main!(benches);
```

### Documentation Requirements

1. **Public API**: All public functions must have doc comments
2. **Examples**: Include code examples in doc comments
3. **Module-level docs**: Each module should have overview documentation
4. **README**: Each crate should have a README.md

```rust
/// Search for documents matching the query.
///
/// # Arguments
///
/// * `query` - The search query string
/// * `k` - Number of results to return
///
/// # Returns
///
/// A vector of search results ordered by relevance
///
/// # Example
///
/// ```no_run
/// use leann_core::LeannSearcher;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let searcher = LeannSearcher::from_path("my-index")?;
/// let results = searcher.search("vector database", 10).await?;
/// # Ok(())
/// # }
/// ```
pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
    // Implementation
}
```

### Performance Optimization Guidelines

1. **Profile First**: Use `cargo flamegraph` to identify bottlenecks
2. **Avoid Allocations**: Use references and slices where possible
3. **Batch Operations**: Process data in batches for cache efficiency
4. **Parallel Processing**: Use `rayon` for data parallelism
5. **SIMD**: Consider SIMD operations for vector computations

```rust
// Use iterators instead of loops for optimization
let sum: f32 = vector.iter().sum();

// Parallel processing with rayon
use rayon::prelude::*;
let embeddings: Vec<_> = texts
    .par_iter()
    .map(|text| compute_embedding(text))
    .collect();

// Avoid cloning, use references
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

### Observability

#### Logging

```rust
use tracing::{info, debug, error, instrument};

#[instrument(skip(self))]
pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
    info!(query = %query, k = k, "Starting search");

    let embedding = self.embed_query(query).await?;
    debug!(dimension = embedding.len(), "Query embedded");

    let results = self.backend.search(&embedding, k).await?;
    info!(result_count = results.len(), "Search complete");

    Ok(results)
}
```

#### Metrics

```rust
use prometheus::{Counter, Histogram, register_counter, register_histogram};

lazy_static! {
    static ref SEARCH_COUNT: Counter =
        register_counter!("leann_search_total", "Total searches").unwrap();

    static ref SEARCH_DURATION: Histogram =
        register_histogram!("leann_search_duration_seconds", "Search duration").unwrap();
}

pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
    let timer = SEARCH_DURATION.start_timer();

    let results = self.inner_search(query, k).await?;

    timer.observe_duration();
    SEARCH_COUNT.inc();

    Ok(results)
}
```

### Migration Workflow

When migrating from Python to Rust:

1. **Understand Python Implementation**: Read Python code in `LEANN/` directory
2. **Design Rust API**: Consider ownership, lifetimes, and async
3. **Implement Core Types**: Start with data structures (CSR graph, etc.)
4. **Add Business Logic**: Port algorithms (HNSW, DiskANN)
5. **Write Tests**: Unit tests, integration tests, benchmarks
6. **Optimize**: Profile and optimize hot paths
7. **Document**: Add doc comments and examples

### Common Pitfalls to Avoid

1. **Over-abstracting**: Don't create too many trait layers
2. **Fighting the Borrow Checker**: Redesign API if fighting borrowck too much
3. **Premature Optimization**: Get it working first, then optimize
4. **Ignoring Error Handling**: Always handle errors properly
5. **Blocking in Async**: Never block in async functions, use `spawn_blocking`

```rust
// ❌ Bad: Blocking in async function
pub async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
    // This blocks the async runtime!
    self.model.run(text)
}

// ✅ Good: Use spawn_blocking for CPU-intensive work
pub async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
    let text = text.to_string();
    let model = self.model.clone();

    tokio::task::spawn_blocking(move || {
        model.run(&text)
    }).await?
}
```

### Documentation References

- **README.md**: Project overview, quick start, usage examples
- **document/BRD.md**: Business requirements, success metrics, roadmap
- **document/MODULAR_DESIGN.md**: Detailed architecture, module design, API specifications
- **Original Python**: `LEANN/` directory contains reference implementation

### Development Priorities

#### Phase 1: Foundation (Current)
- Core data structures (CSR graph, index format)
- HNSW backend implementation
- Basic embedding support
- CLI tool skeleton

#### Phase 2: Features
- DiskANN backend
- Metadata filtering
- Incremental updates
- Advanced embedding providers

#### Phase 3: Production
- REST API server
- gRPC server
- Observability (metrics, tracing)
- Docker/Kubernetes deployment

#### Phase 4: Optimization
- SIMD optimizations
- GPU acceleration
- Distributed indexes
- Advanced compression

---

## Python Reference (Original LEANN)

The sections below document the original Python implementation for reference during migration.
