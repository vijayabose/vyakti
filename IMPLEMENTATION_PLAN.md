# Vyakti Implementation Plan
## Completing Feature Parity with Python LEANN

**Date**: 2025-01-18
**Status**: llama.cpp Integration ✅ COMPLETE | Critical Gaps Roadmap 📋 READY

---

## ✅ Completed: llama.cpp Integration

### Summary
Successfully migrated CLI from OllamaProvider to LlamaCppProvider, making Vyakti **100% self-contained** with no external dependencies.

### Changes Made

#### 1. **Updated CLI to Use LlamaCppProvider** (`crates/vyakti-cli/src/main.rs`)
- Replaced `OllamaProvider` with `LlamaCppProvider` for both build and search operations
- Added conditional compilation for backward compatibility (`#[cfg(feature = "llama-cpp")]`)
- Integrated automatic model download via `ensure_model()`

#### 2. **Added CLI Arguments**
```bash
# Build command
vyakti build my-docs --input ./docs \
  --embedding-model-path ~/custom-model.gguf \  # Optional: custom model
  --gpu-layers 32                                # Optional: GPU acceleration

# Search command
vyakti search my-docs "query" \
  --embedding-model-path ~/custom-model.gguf \
  --gpu-layers 32
```

#### 3. **Feature Flag Configuration** (`crates/vyakti-cli/Cargo.toml`)
```toml
[features]
default = ["all-formats", "ast-extended", "llama-cpp"]
llama-cpp = ["vyakti-embedding/llama-cpp"]
```

#### 4. **Auto-Download on First Use**
- Default model: `mxbai-embed-large-v1.q4_k_m.gguf` (~500MB)
- Storage location: `~/.vyakti/models/`
- Downloaded from HuggingFace on first `vyakti build` or `vyakti search`

### Key Benefits
✅ **No external services required** - No Ollama, no API keys
✅ **100% local execution** - Complete privacy
✅ **GPU acceleration support** - Offload layers to GPU
✅ **Auto-downloads models** - Zero configuration for users
✅ **Backward compatible** - Ollama still available if llama-cpp disabled

### Verification
```bash
# Test build
cargo build --package vyakti-cli

# Test CLI help
cargo run --bin vyakti -- --help

# Expected output shows new flags:
#   --embedding-model-path <PATH>  Path to custom embedding model file
#   --gpu-layers <NUM>            Number of GPU layers to offload
```

---

## 🎯 Critical Gaps Roadmap

Three high-priority features needed for full feature parity with Python LEANN:

1. **DiskANN Backend** - Complete algorithm implementation
2. **Interactive Chat (vyakti ask)** - RAG with LLM integration
3. **Incremental Index Updates** - Append documents without full rebuild

---

## 1. DiskANN Backend Implementation

### Current Status
**30% Complete** - Foundation laid, algorithm needs implementation

#### What Exists
- ✅ `DiskAnnGraph` structure
- ✅ `ProductQuantization` module skeleton (`pq.rs`)
- ✅ Benchmark harness (`diskann_search.rs`)
- ✅ Basic graph structure

#### What's Missing
- ❌ Full DiskANN algorithm implementation
- ❌ Graph partitioning logic
- ❌ Pruning strategies (global, local, proportional)
- ❌ PQ compression with reranking
- ❌ Smart memory configuration

### Implementation Plan

#### Phase 1: Core Algorithm (Est: 5-7 days)
**File**: `crates/vyakti-backend-diskann/src/lib.rs`

```rust
pub struct DiskAnnBackend {
    graph: DiskAnnGraph,
    pq_table: ProductQuantizer,
    config: DiskAnnConfig,
}

pub struct DiskAnnConfig {
    pub graph_degree: usize,
    pub build_complexity: usize,
    pub search_complexity: usize,
    pub pq_bytes: usize,              // PQ compression level
    pub search_memory_gb: f32,        // Memory for PQ table
    pub build_memory_gb: f32,         // Memory for construction
}

impl Backend for DiskAnnBackend {
    async fn build(&mut self, vectors: &[Vec<f32>], config: &BackendConfig) -> Result<()> {
        // 1. Build initial HNSW-like graph
        self.build_initial_graph(vectors, config).await?;

        // 2. Apply pruning strategy
        self.prune_graph(config.graph_degree)?;

        // 3. Create PQ compression table
        self.pq_table = ProductQuantizer::train(vectors, self.config.pq_bytes)?;

        // 4. Compress vectors
        self.compress_vectors(vectors)?;

        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // 1. Compress query with PQ
        let compressed_query = self.pq_table.compress(query)?;

        // 2. Graph traversal with compressed vectors
        let candidates = self.graph_search(&compressed_query, k * 10)?;

        // 3. Rerank with full precision
        let reranked = self.rerank_candidates(&candidates, query, k)?;

        Ok(reranked)
    }
}
```

**Key Functions to Implement**:
1. `build_initial_graph()` - Greedy graph construction
2. `prune_graph()` - Robust pruning algorithm
3. `ProductQuantizer::train()` - K-means clustering for PQ
4. `compress_vectors()` - Apply PQ compression
5. `graph_search()` - Beam search with PQ distance
6. `rerank_candidates()` - Full precision reranking

#### Phase 2: Graph Partitioning (Est: 3-4 days)
**File**: `crates/vyakti-backend-diskann/src/partition.rs`

```rust
pub struct GraphPartitioner {
    num_shards: usize,
    shard_size: usize,
}

impl GraphPartitioner {
    pub fn partition(&self, graph: &DiskAnnGraph) -> Result<Vec<GraphShard>> {
        // 1. Identify high-degree hub nodes
        let hubs = self.identify_hubs(graph)?;

        // 2. Partition non-hub nodes into shards
        let shards = self.partition_nodes(graph, &hubs)?;

        // 3. Replicate hub nodes across shards
        self.replicate_hubs(shards, &hubs)?;

        Ok(shards)
    }
}
```

**Reference**: Python implementation at `LEANN/packages/leann-backend-diskann/leann_backend_diskann/graph_partition.py`

#### Phase 3: Smart Memory Configuration (Est: 1-2 days)
**File**: `crates/vyakti-backend-diskann/src/config.rs`

```rust
pub fn calculate_smart_memory_config(
    num_vectors: usize,
    dimension: usize,
) -> (f32, f32) {
    let embedding_size_gb = (num_vectors * dimension * 4) as f32 / 1_073_741_824.0;

    // PQ compression: 1/10 of embedding size
    let search_memory_gb = (embedding_size_gb / 10.0).max(0.1);

    // Build memory: 50% of available RAM
    let available_memory = get_available_memory_gb();
    let total_memory = get_total_memory_gb();
    let build_memory_gb = (available_memory * 0.5).min(total_memory * 0.75).max(2.0);

    (search_memory_gb, build_memory_gb)
}
```

#### Phase 4: Testing & Benchmarks (Est: 2-3 days)
- Integration tests with real datasets
- Performance benchmarks vs HNSW
- Memory usage profiling
- Accuracy metrics (recall@k)

### Total Estimate: **11-16 days**

---

## 2. Interactive Chat Interface (vyakti ask)

### Overview
Add RAG (Retrieval-Augmented Generation) capability using llama.cpp for both embeddings and chat.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                 vyakti ask                      │
│                                                 │
│  User Question                                  │
│       ↓                                         │
│  1. VyaktiSearcher (semantic search)           │
│       ↓                                         │
│  2. Retrieve top-k relevant documents           │
│       ↓                                         │
│  3. Build prompt (context + question)           │
│       ↓                                         │
│  4. LlamaCppChatProvider (text generation)     │
│       ↓                                         │
│  Answer                                         │
└─────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Create `vyakti-chat` Crate (Est: 3-4 days)
**Location**: `crates/vyakti-chat/`

```rust
// crates/vyakti-chat/src/lib.rs

pub trait ChatProvider: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String>;
    fn model_name(&self) -> &str;
}

pub struct LlamaCppChatProvider {
    model: Arc<LlamaModel>,
    context: Arc<Mutex<LlamaContext>>,
    config: ChatConfig,
}

pub struct ChatConfig {
    pub model_path: PathBuf,
    pub max_tokens: u32,           // Default: 512
    pub temperature: f32,          // Default: 0.7
    pub top_p: f32,                // Default: 0.9
    pub n_gpu_layers: u32,         // GPU offloading
    pub n_ctx: u32,                // Context size (default: 4096)
    pub n_threads: u32,            // CPU threads
}

impl ChatProvider for LlamaCppChatProvider {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let (tx, rx) = oneshot::channel();

        self.sender.send(GenerationRequest {
            prompt: prompt.to_string(),
            response: tx,
        })?;

        rx.await?
    }

    fn model_name(&self) -> &str {
        "llama.cpp"
    }
}

pub struct VyaktiChat {
    searcher: VyaktiSearcher,
    chat_provider: Box<dyn ChatProvider>,
    top_k: usize,                  // Documents to retrieve (default: 5)
    prompt_template: String,       // RAG prompt template
}

impl VyaktiChat {
    pub async fn ask(&self, question: &str) -> Result<String> {
        // 1. Search for relevant documents
        let results = self.searcher.search(question, self.top_k).await?;

        // 2. Build prompt with context
        let prompt = self.build_prompt(question, &results);

        // 3. Generate answer with LLM
        let answer = self.chat_provider.generate(&prompt).await?;

        Ok(answer)
    }

    pub async fn interactive(&self) -> Result<()> {
        use rustyline::Editor;

        let mut rl = Editor::<()>::new()?;

        println!("Interactive chat mode. Type 'exit' or 'quit' to end.\n");

        loop {
            let line = rl.readline("You: ")?;
            let input = line.trim();

            if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
                break;
            }

            if input.is_empty() {
                continue;
            }

            println!("\nThinking...");
            let answer = self.ask(input).await?;
            println!("\nAssistant: {}\n", answer);

            rl.add_history_entry(input);
        }

        Ok(())
    }

    fn build_prompt(&self, question: &str, results: &[SearchResult]) -> String {
        let context = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("[{}] {}", i + 1, r.text))
            .collect::<Vec<_>>()
            .join("\n\n");

        self.prompt_template
            .replace("{context}", &context)
            .replace("{question}", question)
    }
}
```

**Default Prompt Template**:
```
You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:
```

#### Phase 2: Model Download for Chat (Est: 1 day)
**File**: `crates/vyakti-embedding/src/download.rs` (extend existing)

```rust
pub const DEFAULT_CHAT_MODEL_REPO: &str = "meta-llama/Llama-3.2-3B-Instruct-GGUF";
pub const DEFAULT_CHAT_MODEL_FILE: &str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";
pub const DEFAULT_CHAT_MODEL_NAME: &str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf";

pub async fn download_default_chat_model() -> Result<PathBuf> {
    let models_dir = get_models_dir()?.join("chat");
    fs::create_dir_all(&models_dir).await?;

    let model_path = models_dir.join(DEFAULT_CHAT_MODEL_NAME);

    if model_path.exists() {
        return Ok(model_path);
    }

    println!("📥 Downloading chat model (first time only, ~2GB)...");
    download_model(
        DEFAULT_CHAT_MODEL_REPO,
        DEFAULT_CHAT_MODEL_FILE,
        DEFAULT_CHAT_MODEL_NAME,
    ).await?;

    Ok(model_path)
}

pub async fn ensure_chat_model(model_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = model_path {
        if path.exists() {
            return Ok(path);
        } else {
            anyhow::bail!("Specified chat model path does not exist: {}", path.display());
        }
    }

    // Check for default model in chat/ subdirectory
    let default_path = get_models_dir()?.join("chat").join(DEFAULT_CHAT_MODEL_NAME);
    if default_path.exists() {
        return Ok(default_path);
    }

    // Download default model
    download_default_chat_model().await
}
```

**Model Storage**:
```
~/.vyakti/models/
├── embeddings/
│   └── mxbai-embed-large-v1.q4_k_m.gguf    (~500MB)
└── chat/
    └── Llama-3.2-3B-Instruct-Q4_K_M.gguf   (~2GB)
```

#### Phase 3: CLI Integration (Est: 1-2 days)
**File**: `crates/vyakti-cli/src/main.rs`

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Ask questions about indexed documents (RAG)
    Ask {
        /// Index name
        name: String,

        /// Question to ask
        question: Option<String>,

        /// Interactive chat mode
        #[arg(short, long)]
        interactive: bool,

        /// Index directory
        #[arg(short, long, default_value = ".vyakti")]
        index_dir: PathBuf,

        /// Path to chat model (GGUF format)
        #[arg(long)]
        chat_model_path: Option<PathBuf>,

        /// Number of documents to retrieve for context
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,

        /// Temperature for generation (0.0 = deterministic, 1.0 = creative)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// GPU layers to offload
        #[arg(long, default_value = "0")]
        gpu_layers: u32,
    },
}
```

**Usage Examples**:
```bash
# Single question
vyakti ask my-docs "What is LEANN?"

# Interactive mode
vyakti ask my-docs --interactive

# Custom model
vyakti ask my-docs "question" --chat-model-path ~/llama-8b.gguf

# GPU acceleration
vyakti ask my-docs "question" --gpu-layers 32

# More context documents
vyakti ask my-docs "question" --top-k 10
```

#### Phase 4: Testing & Examples (Est: 1-2 days)
- Unit tests for prompt building
- Integration tests with mock LLM
- Example with real chat model
- Documentation and examples

### Total Estimate: **6-9 days**

---

## 3. Incremental Index Updates

### Overview
Allow appending new documents to existing HNSW indexes without full rebuild.

### Python Reference
**File**: `LEANN/packages/leann-core/src/leann/api.py:644-793`

Key logic:
1. Load existing index metadata
2. Verify compatibility (backend, dimensions)
3. Check for compact mode (not supported)
4. Compute embeddings for new documents
5. Load Faiss index
6. Append new vectors with `index.add()`
7. Update passage files (`.passages.jsonl`, `.passages.idx`)
8. Save updated index

### Implementation Plan

#### Phase 1: Add Update Method to Builder (Est: 2-3 days)
**File**: `crates/vyakti-core/src/builder.rs`

```rust
impl VyaktiBuilder {
    /// Update an existing index by appending new documents
    pub async fn update_index(&mut self, index_path: &Path) -> Result<()> {
        // 1. Validate index exists
        if !index_path.exists() {
            anyhow::bail!("Index not found: {}", index_path.display());
        }

        // 2. Load existing metadata
        let metadata = self.load_index_metadata(index_path)?;

        // 3. Validate compatibility
        self.validate_update_compatibility(&metadata)?;

        // 4. Compute embeddings for new documents
        let new_embeddings = self.compute_all_embeddings().await?;

        // 5. Load existing backend
        let mut backend = self.load_backend(index_path, &metadata).await?;

        // 6. Get current document count
        let base_id = metadata.num_documents;

        // 7. Assign IDs to new documents
        for (i, doc) in self.documents.iter_mut().enumerate() {
            doc.id = base_id + i;
        }

        // 8. Add new vectors to backend
        backend.add_vectors(&new_embeddings).await?;

        // 9. Update document storage
        self.append_documents(index_path, base_id)?;

        // 10. Save updated index
        self.save_updated_index(index_path, &backend, &metadata).await?;

        Ok(())
    }

    fn validate_update_compatibility(&self, metadata: &IndexMetadata) -> Result<()> {
        // Check backend matches
        if metadata.backend_name != self.backend_name() {
            anyhow::bail!("Backend mismatch: index uses {}, builder uses {}",
                metadata.backend_name, self.backend_name());
        }

        // Compact indexes don't support updates
        if metadata.is_compact {
            anyhow::bail!("Compact indexes do not support incremental updates. Rebuild required.");
        }

        // Check dimensions match
        if let Some(dim) = metadata.embedding_dimension {
            let new_dim = self.embedding_provider.dimension();
            if dim != new_dim {
                anyhow::bail!("Dimension mismatch: index uses {}, new embeddings use {}", dim, new_dim);
            }
        }

        Ok(())
    }

    async fn load_backend(&self, index_path: &Path, metadata: &IndexMetadata) -> Result<Box<dyn Backend>> {
        // Load backend from disk
        let backend_data = fs::read(index_path.join("backend.bin")).await?;
        let backend = self.backend.deserialize(&backend_data)?;
        Ok(backend)
    }

    fn append_documents(&self, index_path: &Path, base_id: usize) -> Result<()> {
        // Append to .passages.jsonl
        let passages_file = index_path.join("index.passages.jsonl");
        let mut file = OpenOptions::new()
            .append(true)
            .open(&passages_file)?;

        // Append to .passages.idx (offset map)
        let mut offset_map = self.load_offset_map(index_path)?;

        for doc in &self.documents {
            let offset = file.seek(SeekFrom::End(0))?;
            writeln!(file, "{}", serde_json::to_string(&doc)?)?;
            offset_map.insert(doc.id.to_string(), offset);
        }

        // Save updated offset map
        self.save_offset_map(index_path, &offset_map)?;

        Ok(())
    }
}
```

#### Phase 2: Backend Support for Vector Addition (Est: 1-2 days)
**File**: `crates/vyakti-backend-hnsw/src/lib.rs`

```rust
impl Backend for HnswBackend {
    // ... existing methods ...

    async fn add_vectors(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        for vector in vectors {
            self.graph.add_node(vector.clone())?;
        }
        Ok(())
    }
}
```

**Key Implementation**:
- HNSW supports incremental addition natively
- DiskANN would require rebuild (document limitation)

#### Phase 3: CLI Command (Est: 1 day)
**File**: `crates/vyakti-cli/src/main.rs`

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Update an existing index with new documents
    Update {
        /// Index name
        name: String,

        /// Input directory or file with new documents
        #[arg(short, long)]
        input: PathBuf,

        /// Index directory
        #[arg(short = 'd', long, default_value = ".vyakti")]
        index_dir: PathBuf,

        // Same chunking options as Build
        #[arg(long, default_value = "1024")]
        chunk_size: usize,

        #[arg(long, default_value = "512")]
        chunk_overlap: usize,

        #[arg(long)]
        enable_code_chunking: bool,

        #[arg(long)]
        no_chunking: bool,
    },
}
```

**Usage**:
```bash
# Add new documents to existing index
vyakti update my-docs --input ./new-files

# With code chunking
vyakti update my-code --input ./new-code --enable-code-chunking
```

#### Phase 4: Testing & Documentation (Est: 1-2 days)
- Integration tests for update workflow
- Test with various document types
- Document limitations (no compact mode support)
- Usage examples

### Total Estimate: **5-8 days**

---

## Summary Timeline

| Feature | Complexity | Estimate | Priority |
|---------|-----------|----------|----------|
| **DiskANN Backend** | High | 11-16 days | P1 |
| **Interactive Chat** | Medium | 6-9 days | P1 |
| **Incremental Updates** | Medium | 5-8 days | P2 |

**Total Project Estimate**: **22-33 days** (4-7 weeks)

### Recommended Implementation Order

1. **Interactive Chat** (6-9 days) - Highest user impact, reuses llama.cpp infrastructure
2. **Incremental Updates** (5-8 days) - Medium complexity, important for workflows
3. **DiskANN Backend** (11-16 days) - Most complex, can be done last

---

## Success Criteria

### ✅ llama.cpp Integration (COMPLETE)
- [x] CLI uses LlamaCppProvider by default
- [x] Auto-downloads embedding model on first use
- [x] GPU acceleration support via `--gpu-layers`
- [x] Compiles without warnings
- [x] Backward compatible with Ollama provider

### 🎯 Interactive Chat
- [ ] `vyakti ask` command working
- [ ] Auto-downloads chat model (Llama 3.2 3B)
- [ ] Interactive REPL mode
- [ ] Prompt template customization
- [ ] GPU acceleration support
- [ ] Documentation and examples

### 🎯 DiskANN Backend
- [ ] Full algorithm implementation
- [ ] Product Quantization working
- [ ] Graph partitioning functional
- [ ] Performance benchmarks show improvement over HNSW
- [ ] Memory configuration auto-tuning
- [ ] Integration tests passing

### 🎯 Incremental Updates
- [ ] `vyakti update` command working
- [ ] Validates compatibility before update
- [ ] Appends vectors to existing backend
- [ ] Updates document storage correctly
- [ ] Error handling for compact indexes
- [ ] Documentation with examples

---

## Testing Strategy

### Unit Tests
- Each module has comprehensive unit tests
- Mock providers for embedding/chat
- Edge case coverage

### Integration Tests
- End-to-end workflows
- Real model downloads (in CI, cached)
- Performance benchmarks

### Performance Tests
- DiskANN vs HNSW comparison
- Memory usage profiling
- Latency measurements
- Throughput benchmarks

---

## Documentation Plan

### User Documentation
- **QUICKSTART.md** - Getting started guide
- **CLI_REFERENCE.md** - Complete CLI documentation
- **CHAT_GUIDE.md** - Using `vyakti ask` effectively
- **PERFORMANCE_TUNING.md** - GPU, memory, model selection

### Developer Documentation
- **ARCHITECTURE.md** - System design
- **CONTRIBUTING.md** - Development workflow
- **API_REFERENCE.md** - Rust API docs (via rustdoc)

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| DiskANN complexity too high | Break into smaller milestones, reference Python closely |
| Chat model too large for users | Provide smaller alternatives (1B models), document requirements |
| Incremental updates buggy | Extensive testing, clear error messages, document limitations |

### Timeline Risks
| Risk | Mitigation |
|------|------------|
| Features take longer than estimated | Prioritize Chat > Updates > DiskANN, can defer DiskANN if needed |
| Breaking changes in dependencies | Pin dependency versions, thorough testing |

---

## Next Steps

1. ✅ **Complete llama.cpp integration** - DONE
2. 📝 **Review and approve this plan** - Current step
3. 🚀 **Begin implementation** - Start with Interactive Chat
4. 🧪 **Continuous testing** - Test each phase before moving on
5. 📚 **Document as we go** - Write docs alongside code

---

**Last Updated**: 2025-01-18
**Status**: Ready for implementation 🚀
