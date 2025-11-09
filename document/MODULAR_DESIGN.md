# Modular Design Document
## Vyakti: Rust Vector Database Architecture

**Version:** 1.0
**Date:** November 9, 2024
**Status:** Design Phase

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [System Architecture](#2-system-architecture)
3. [Module Design](#3-module-design)
4. [Core Modules](#4-core-modules)
5. [Backend Modules](#5-backend-modules)
6. [Interface Modules](#6-interface-modules)
7. [Data Flow](#7-data-flow)
8. [API Design](#8-api-design)
9. [Deployment Modes](#9-deployment-modes)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)
11. [Technology Stack](#11-technology-stack)
12. [Design Patterns](#12-design-patterns)

---

## 1. Architecture Overview

### 1.1 Design Principles

1. **Modularity**: Clear separation of concerns with well-defined interfaces
2. **Extensibility**: Plugin architecture for backends and embeddings
3. **Performance**: Zero-cost abstractions, minimal allocations
4. **Safety**: Leverage Rust's type system and ownership model
5. **Flexibility**: Support CLI, library, and server modes from same codebase

### 1.2 Architecture Patterns

- **Layered Architecture**: Clear separation between presentation, business logic, and data layers
- **Plugin/Provider Pattern**: Pluggable backends and embedding providers
- **Builder Pattern**: Ergonomic API construction
- **Strategy Pattern**: Runtime selection of algorithms
- **Repository Pattern**: Abstract storage layer

### 1.3 System Context

```
┌─────────────────────────────────────────────────────────────┐
│                      External Systems                        │
├─────────────────────────────────────────────────────────────┤
│  Embedding APIs  │  Storage (S3)  │  Observability Tools    │
│  (OpenAI, etc)   │  (GCS, Azure)  │  (Prometheus, Jaeger)   │
└──────────┬────────────────┬────────────────┬────────────────┘
           │                │                │
┌──────────▼────────────────▼────────────────▼────────────────┐
│                        Vyakti                              │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │    CLI     │  │   Server   │  │  Library   │           │
│  │   Mode     │  │    Mode    │  │    API     │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │           Core Engine (vyakti-core)           │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  HNSW      │  │  DiskANN   │  │   Custom   │           │
│  │  Backend   │  │  Backend   │  │  Backends  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture

### 2.1 Module Hierarchy

```
vyakti/
├── crates/
│   ├── vyakti-common/          # Shared utilities (errors, config, traits)
│   │   ├── src/
│   │   │   ├── error.rs       # Error types
│   │   │   ├── config.rs      # Configuration
│   │   │   ├── types.rs       # Common types
│   │   │   └── traits.rs      # Core traits
│   │   └── Cargo.toml
│   │
│   ├── vyakti-storage/         # Storage layer (CSR, memory mapping)
│   │   ├── src/
│   │   │   ├── csr.rs         # Compressed Sparse Row format
│   │   │   ├── mmap.rs        # Memory-mapped files
│   │   │   ├── index.rs       # Index format
│   │   │   └── serialization.rs
│   │   └── Cargo.toml
│   │
│   ├── vyakti-embedding/       # Embedding computation
│   │   ├── src/
│   │   │   ├── models/
│   │   │   │   ├── sentence_transformers.rs
│   │   │   │   ├── openai.rs
│   │   │   │   ├── ollama.rs
│   │   │   │   └── onnx.rs
│   │   │   ├── server.rs      # Embedding server
│   │   │   ├── batching.rs    # Batch processing
│   │   │   └── cache.rs       # Embedding cache
│   │   └── Cargo.toml
│   │
│   ├── vyakti-core/            # Core API (Builder, Searcher)
│   │   ├── src/
│   │   │   ├── builder.rs     # LeannBuilder
│   │   │   ├── searcher.rs    # LeannSearcher
│   │   │   ├── chat.rs        # LeannChat (RAG)
│   │   │   ├── metadata.rs    # Metadata filtering
│   │   │   ├── registry.rs    # Backend registry
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── vyakti-backend-hnsw/    # HNSW backend
│   │   ├── src/
│   │   │   ├── graph.rs       # HNSW graph
│   │   │   ├── builder.rs     # Index building
│   │   │   ├── searcher.rs    # Search with recomputation
│   │   │   ├── pruning.rs     # Graph pruning
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── vyakti-backend-diskann/ # DiskANN backend
│   │   ├── src/
│   │   │   ├── graph.rs       # DiskANN graph
│   │   │   ├── pq.rs          # Product Quantization
│   │   │   ├── builder.rs
│   │   │   ├── searcher.rs
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── vyakti-proto/           # Protocol Buffers
│   │   ├── proto/
│   │   │   └── vyakti.proto
│   │   ├── build.rs
│   │   └── Cargo.toml
│   │
│   ├── vyakti-server/          # Network server
│   │   ├── src/
│   │   │   ├── rest/          # REST API
│   │   │   │   ├── routes.rs
│   │   │   │   ├── handlers.rs
│   │   │   │   └── middleware.rs
│   │   │   ├── grpc/          # gRPC API
│   │   │   │   ├── service.rs
│   │   │   │   └── interceptors.rs
│   │   │   ├── auth.rs        # Authentication
│   │   │   ├── metrics.rs     # Prometheus metrics
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   └── vyakti-cli/             # Command-line interface
│       ├── src/
│       │   ├── commands/
│       │   │   ├── build.rs
│       │   │   ├── search.rs
│       │   │   ├── chat.rs
│       │   │   ├── serve.rs
│       │   │   └── list.rs
│       │   ├── main.rs
│       │   └── utils.rs
│       └── Cargo.toml
```

### 2.2 Dependency Graph

```
vyakti-cli ──────┐
                ├──> vyakti-server ──┐
                │                   │
                └──> vyakti-core ────┤
                         │          │
                         ├──────────┴──> vyakti-embedding
                         │               vyakti-storage
                         │               vyakti-common
                         │
                         ├──> vyakti-backend-hnsw ──┐
                         │                          ├──> vyakti-common
                         └──> vyakti-backend-diskann─┘
```

---

## 3. Module Design

### 3.1 Module Contracts

Each module exposes:
- **Public API**: Well-documented, stable interfaces
- **Traits**: For extensibility and testing
- **Types**: Strong typing for domain concepts
- **Errors**: Module-specific error types

### 3.2 Module Independence

- Modules communicate only through defined interfaces
- No circular dependencies
- Each module can be tested independently
- Modules can be versioned independently

---

## 4. Core Modules

### 4.1 vyakti-common

**Purpose**: Shared utilities, types, and traits used across all modules

**Key Components**:

```rust
// error.rs - Unified error handling
#[derive(Debug, thiserror::Error)]
pub enum LeannError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}

pub type Result<T> = std::result::Result<T, LeannError>;

// config.rs - Configuration structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeannConfig {
    pub index: IndexConfig,
    pub embedding: EmbeddingConfig,
    pub backend: BackendConfig,
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

// traits.rs - Core trait definitions
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    async fn build(&mut self, vectors: &[Vec<f32>], config: &BackendConfig) -> Result<()>;
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
}

pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```

**Dependencies**: Standard library, serde, thiserror

---

### 4.2 vyakti-storage

**Purpose**: Efficient storage formats and I/O operations

**Key Components**:

```rust
// csr.rs - Compressed Sparse Row graph format
pub struct CsrGraph {
    // CSR format: row_ptr[i] to row_ptr[i+1] contains neighbor indices
    row_ptr: Vec<usize>,     // Offset into col_idx for each node
    col_idx: Vec<u32>,       // Neighbor node IDs
    edge_data: Vec<f32>,     // Optional edge weights/distances
    num_nodes: usize,
}

impl CsrGraph {
    pub fn new(num_nodes: usize) -> Self { /* ... */ }

    pub fn add_edge(&mut self, from: u32, to: u32, weight: f32) { /* ... */ }

    pub fn neighbors(&self, node: u32) -> &[u32] { /* ... */ }

    pub fn save(&self, path: &Path) -> Result<()> { /* ... */ }

    pub fn load(path: &Path) -> Result<Self> { /* ... */ }
}

// mmap.rs - Memory-mapped file I/O
pub struct MmapIndex {
    mmap: Mmap,
    header: IndexHeader,
}

impl MmapIndex {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // Parse header and validate
        Ok(Self { mmap, header })
    }

    pub fn vectors(&self) -> &[f32] {
        // Zero-copy access to vectors
    }

    pub fn graph(&self) -> CsrGraph {
        // Deserialize graph structure
    }
}

// index.rs - Index file format
#[repr(C)]
pub struct IndexHeader {
    magic: [u8; 4],          // "LEAN"
    version: u32,
    num_vectors: u64,
    dimension: u32,
    backend_type: u32,
    flags: u64,              // compact, recompute, etc.
}

pub struct IndexBuilder {
    vectors: Vec<Vec<f32>>,
    metadata: Vec<HashMap<String, Value>>,
    graph: CsrGraph,
}

impl IndexBuilder {
    pub fn save(&self, path: &Path) -> Result<()> {
        // Serialize to efficient binary format
    }
}
```

**Features**:
- CSR format for memory-efficient graphs
- Memory-mapped files for fast loading
- Zero-copy deserialization where possible
- Versioned file format for compatibility

**Dependencies**: memmap2, bincode, serde

---

### 4.3 vyakti-embedding

**Purpose**: Embedding model abstraction and computation

**Key Components**:

```rust
// models/sentence_transformers.rs
pub struct SentenceTransformersModel {
    model: ort::Session,     // ONNX Runtime
    tokenizer: Tokenizer,
    dimension: usize,
}

impl EmbeddingProvider for SentenceTransformersModel {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let batches = self.batch_texts(texts, 32);
        let mut embeddings = Vec::new();

        for batch in batches {
            let tokens = self.tokenizer.encode_batch(batch)?;
            let inputs = self.prepare_inputs(tokens)?;
            let outputs = self.model.run(inputs)?;
            embeddings.extend(self.extract_embeddings(outputs)?);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// models/openai.rs
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "input": texts,
                "model": self.model,
            }))
            .send()
            .await?;

        // Parse response and extract embeddings
    }
}

// server.rs - Embedding server for recomputation
pub struct EmbeddingServer {
    provider: Arc<dyn EmbeddingProvider>,
    cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,
}

impl EmbeddingServer {
    pub async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(embedding) = self.cache.read().await.get(text) {
            return Ok(embedding.clone());
        }

        // Compute and cache
        let embeddings = self.provider.embed(&[text.to_string()]).await?;
        let embedding = embeddings[0].clone();

        self.cache.write().await.put(text.to_string(), embedding.clone());
        Ok(embedding)
    }
}
```

**Features**:
- Unified interface for all embedding providers
- Batch processing for efficiency
- LRU cache for recomputation
- Async API for non-blocking operations

**Dependencies**: ort (ONNX Runtime), tokenizers, reqwest, tokio

---

### 4.4 vyakti-core

**Purpose**: High-level API for building and searching indexes

**Key Components**:

```rust
// builder.rs
pub struct LeannBuilder {
    backend: Box<dyn Backend>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    texts: Vec<String>,
    metadata: Vec<HashMap<String, Value>>,
    config: LeannConfig,
}

impl LeannBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn backend<B: Backend + 'static>(mut self, backend: B) -> Self {
        self.backend = Box::new(backend);
        self
    }

    pub fn embedding_model(mut self, provider: impl EmbeddingProvider + 'static) -> Self {
        self.embedding_provider = Arc::new(provider);
        self
    }

    pub async fn add_text(&mut self, text: impl Into<String>, metadata: Option<HashMap<String, Value>>) -> Result<()> {
        self.texts.push(text.into());
        self.metadata.push(metadata.unwrap_or_default());
        Ok(())
    }

    pub async fn add_texts(&mut self, texts: Vec<String>) -> Result<()> {
        for text in texts {
            self.add_text(text, None).await?;
        }
        Ok(())
    }

    pub async fn build_index(self, name: impl Into<String>) -> Result<PathBuf> {
        let name = name.into();

        // Compute embeddings
        let embeddings = self.embedding_provider
            .embed(&self.texts)
            .await?;

        // Build backend index
        self.backend.build(&embeddings, &self.config.backend).await?;

        // Save to disk
        let path = self.save_index(&name)?;

        Ok(path)
    }
}

// searcher.rs
pub struct LeannSearcher {
    backend: Box<dyn Backend>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    texts: Vec<String>,
    metadata: Vec<HashMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, Value>,
}

impl LeannSearcher {
    pub fn from_path(path: &Path) -> Result<Self> {
        // Load index from disk
    }

    pub async fn search(&self, query: impl Into<String>, k: usize) -> Result<Vec<SearchResult>> {
        let query_text = query.into();

        // Embed query
        let query_embedding = self.embedding_provider
            .embed(&[query_text])
            .await?[0]
            .clone();

        // Search backend
        let results = self.backend
            .search(&query_embedding, k)
            .await?;

        // Enrich with text and metadata
        Ok(self.enrich_results(results))
    }

    pub async fn search_with_filter(
        &self,
        query: impl Into<String>,
        k: usize,
        filter: MetadataFilter,
    ) -> Result<Vec<SearchResult>> {
        let results = self.search(query, k * 2).await?;

        // Apply metadata filter
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(k)
            .collect();

        Ok(filtered)
    }
}

// metadata.rs
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    Eq(String, Value),
    Ne(String, Value),
    Lt(String, Value),
    Le(String, Value),
    Gt(String, Value),
    Ge(String, Value),
    In(String, Vec<Value>),
    Contains(String, String),
    And(Vec<MetadataFilter>),
    Or(Vec<MetadataFilter>),
    Not(Box<MetadataFilter>),
}

impl MetadataFilter {
    pub fn matches(&self, metadata: &HashMap<String, Value>) -> bool {
        match self {
            Self::Eq(key, value) => metadata.get(key) == Some(value),
            Self::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Self::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            // ... other operators
        }
    }
}
```

**Dependencies**: vyakti-common, vyakti-storage, vyakti-embedding

---

## 5. Backend Modules

### 5.1 vyakti-backend-hnsw

**Purpose**: HNSW algorithm implementation with recomputation

**Key Components**:

```rust
// graph.rs
pub struct HnswGraph {
    layers: Vec<Layer>,      // Multi-layer structure
    entry_point: u32,
    max_layers: usize,
    m: usize,                // Max connections per layer
    m_max0: usize,           // Max connections in layer 0
}

pub struct Layer {
    graph: CsrGraph,
}

// builder.rs
pub struct HnswBuilder {
    config: HnswConfig,
    graph: HnswGraph,
    embeddings: Vec<Vec<f32>>,
}

impl HnswBuilder {
    pub fn build(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        for (id, vector) in vectors.iter().enumerate() {
            self.insert(id as u32, vector)?;
        }

        // Prune graph for storage efficiency
        self.prune_graph()?;

        Ok(())
    }

    fn insert(&mut self, id: u32, vector: &[f32]) -> Result<()> {
        let level = self.random_level();
        let mut ep = self.graph.entry_point;

        // Search for nearest neighbors in each layer
        for layer in (level + 1..self.graph.layers.len()).rev() {
            ep = self.search_layer(vector, ep, 1, layer)?[0];
        }

        // Insert into each layer up to assigned level
        for layer in 0..=level {
            let candidates = self.search_layer(vector, ep, self.config.ef_construction, layer)?;
            let neighbors = self.select_neighbors(candidates, self.config.m)?;

            self.graph.layers[layer].graph.add_edges(id, &neighbors);

            // Prune connections of neighbors if needed
            for &neighbor in &neighbors {
                self.prune_connections(neighbor, layer)?;
            }
        }

        Ok(())
    }
}

// searcher.rs
pub struct HnswSearcher {
    graph: HnswGraph,
    embedding_server: Arc<EmbeddingServer>,
    texts: Vec<String>,
}

impl HnswSearcher {
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut ep = self.graph.entry_point;

        // Navigate to layer 0
        for layer in (1..self.graph.layers.len()).rev() {
            ep = self.search_layer(query, ep, 1, layer).await?[0];
        }

        // Search layer 0 with recomputation
        let results = self.search_layer_with_recompute(query, ep, k, 0).await?;

        Ok(results)
    }

    async fn search_layer_with_recompute(
        &self,
        query: &[f32],
        entry: u32,
        k: usize,
        layer: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Initial candidate
        let entry_embedding = self.recompute_embedding(entry).await?;
        let entry_dist = distance(query, &entry_embedding);
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));

        while let Some(Reverse((dist, current))) = candidates.pop() {
            if results.len() >= k && dist.0 > results.peek().unwrap().0 .0 {
                break;
            }

            // Explore neighbors
            for &neighbor in self.graph.layers[layer].graph.neighbors(current) {
                if visited.insert(neighbor) {
                    // Recompute neighbor embedding
                    let neighbor_embedding = self.recompute_embedding(neighbor).await?;
                    let neighbor_dist = distance(query, &neighbor_embedding);

                    candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)));
                    results.push((OrderedFloat(neighbor_dist), neighbor));
                }
            }
        }

        Ok(results.into_sorted_vec().into_iter().take(k).collect())
    }

    async fn recompute_embedding(&self, id: u32) -> Result<Vec<f32>> {
        let text = &self.texts[id as usize];
        self.embedding_server.compute_embedding(text).await
    }
}
```

---

### 5.2 vyakti-backend-diskann

**Purpose**: DiskANN algorithm with product quantization

**Key Components**:

```rust
// pq.rs - Product Quantization
pub struct ProductQuantizer {
    num_subspaces: usize,
    bits_per_subspace: usize,
    codebooks: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        // K-means clustering in each subspace
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        // Encode vector to PQ codes
    }

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        // Decode PQ codes to approximate vector
    }

    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // Compute distance using lookup tables
    }
}

// builder.rs
pub struct DiskAnnBuilder {
    config: DiskAnnConfig,
    graph: CsrGraph,
    pq: ProductQuantizer,
    vectors: Vec<Vec<f32>>,
}

// searcher.rs
pub struct DiskAnnSearcher {
    graph: CsrGraph,
    pq: ProductQuantizer,
    vectors: Vec<Vec<f32>>,  // Optional: for reranking
}

impl DiskAnnSearcher {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // 1. Compute PQ distance lookup tables
        let lut = self.pq.compute_lookup_tables(query);

        // 2. Graph traversal using PQ distances
        let candidates = self.graph_search_pq(query, &lut, k * 10)?;

        // 3. Rerank with full vectors
        let results = self.rerank(query, &candidates, k)?;

        Ok(results)
    }
}
```

---

## 6. Interface Modules

### 6.1 vyakti-cli

**Purpose**: Command-line interface

**Structure**:

```rust
// commands/build.rs
pub async fn build_command(args: BuildArgs) -> Result<()> {
    let mut builder = LeannBuilder::new()
        .backend(create_backend(&args.backend)?)
        .embedding_model(create_embedding_provider(&args)?);

    // Load documents
    let documents = load_documents(&args.input)?;

    // Add to builder
    for doc in documents {
        builder.add_text(doc.text, Some(doc.metadata)).await?;
    }

    // Build index
    let path = builder.build_index(&args.name).await?;

    println!("Index built successfully: {}", path.display());
    Ok(())
}

// commands/search.rs
pub async fn search_command(args: SearchArgs) -> Result<()> {
    let searcher = LeannSearcher::from_path(&args.index_path)?;

    let results = if let Some(filter) = args.metadata_filter {
        searcher.search_with_filter(&args.query, args.top_k, filter).await?
    } else {
        searcher.search(&args.query, args.top_k).await?
    };

    // Print results
    for (i, result) in results.iter().enumerate() {
        println!("{}. [Score: {:.4}] {}", i + 1, result.score, result.text);
    }

    Ok(())
}

// commands/serve.rs
pub async fn serve_command(args: ServeArgs) -> Result<()> {
    let server = LeannServer::new(args.config)?;

    println!("Starting LEANN server on {}:{}", args.host, args.port);

    server.run().await?;

    Ok(())
}

// main.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "leann")]
#[command(about = "LEANN Vector Database CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Build(BuildArgs),
    Search(SearchArgs),
    Chat(ChatArgs),
    Serve(ServeArgs),
    List(ListArgs),
    Remove(RemoveArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build(args) => build_command(args).await,
        Commands::Search(args) => search_command(args).await,
        Commands::Chat(args) => chat_command(args).await,
        Commands::Serve(args) => serve_command(args).await,
        Commands::List(args) => list_command(args).await,
        Commands::Remove(args) => remove_command(args).await,
    }
}
```

---

### 6.2 vyakti-server

**Purpose**: Network server (REST + gRPC)

**Structure**:

```rust
// rest/routes.rs
pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))
        .route("/api/v1/indexes", post(create_index).get(list_indexes))
        .route("/api/v1/indexes/:name", get(get_index).delete(delete_index))
        .route("/api/v1/indexes/:name/documents", post(add_documents))
        .route("/api/v1/indexes/:name/search", post(search))
        .route("/api/v1/indexes/:name/chat", post(chat))
        .route("/metrics", get(metrics))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(RequestIdLayer)
        .with_state(state)
}

// rest/handlers.rs
pub async fn search(
    State(state): State<ServerState>,
    Path(index_name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let searcher = state.get_index(&index_name)?;

    let results = searcher
        .search(&req.query, req.top_k)
        .await?;

    Ok(Json(SearchResponse { results }))
}

pub async fn add_documents(
    State(state): State<ServerState>,
    Path(index_name): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> Result<Json<AddDocumentsResponse>, ApiError> {
    let builder = state.get_or_create_builder(&index_name)?;

    for doc in req.documents {
        builder.add_text(doc.text, doc.metadata).await?;
    }

    Ok(Json(AddDocumentsResponse {
        count: req.documents.len(),
    }))
}

// grpc/service.rs
pub struct LeannService {
    state: ServerState,
}

#[tonic::async_trait]
impl leann_proto::leann_server::Leann for LeannService {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();

        let searcher = self.state
            .get_index(&req.index_name)
            .map_err(|e| Status::not_found(e.to_string()))?;

        let results = searcher
            .search(&req.query, req.top_k as usize)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(SearchResponse {
            results: results.into_iter().map(Into::into).collect(),
        }))
    }
}

// lib.rs
pub struct LeannServer {
    config: ServerConfig,
    state: ServerState,
}

impl LeannServer {
    pub async fn run(self) -> Result<()> {
        let addr = format!("{}:{}", self.config.host, self.config.port).parse()?;

        // REST server
        let rest_handle = tokio::spawn(async move {
            let router = create_router(self.state.clone());

            axum::Server::bind(&addr)
                .serve(router.into_make_service())
                .await
        });

        // gRPC server
        let grpc_addr = format!("{}:{}", self.config.host, self.config.grpc_port).parse()?;
        let grpc_handle = tokio::spawn(async move {
            let service = LeannService::new(self.state);

            tonic::transport::Server::builder()
                .add_service(LeannServer::new(service))
                .serve(grpc_addr)
                .await
        });

        tokio::try_join!(rest_handle, grpc_handle)?;

        Ok(())
    }
}
```

---

## 7. Data Flow

### 7.1 Index Building Flow

```
User Input (Documents/Texts)
    │
    ▼
LeannBuilder.add_text()
    │
    ├─► Store text in memory
    └─► Store metadata
    │
    ▼
LeannBuilder.build_index()
    │
    ├─► EmbeddingProvider.embed()
    │       │
    │       ├─► Batch texts
    │       ├─► Call model/API
    │       └─► Return embeddings
    │
    ├─► Backend.build()
    │       │
    │       ├─► Construct graph (HNSW/DiskANN)
    │       ├─► Prune for compactness
    │       └─► Return graph structure
    │
    └─► IndexBuilder.save()
            │
            ├─► Serialize header
            ├─► Write graph (CSR format)
            ├─► Write metadata
            └─► Flush to disk
```

### 7.2 Search Flow

```
User Query (Text)
    │
    ▼
LeannSearcher.search()
    │
    ├─► EmbeddingProvider.embed()
    │       └─► Query embedding
    │
    ├─► Backend.search()
    │       │
    │       ├─► Graph traversal
    │       │
    │       ├─► For each node in search path:
    │       │   ├─► Check if embedding stored
    │       │   └─► If not: EmbeddingServer.recompute()
    │       │
    │       └─► Return nearest neighbors
    │
    ├─► MetadataFilter.apply() (optional)
    │
    └─► Return SearchResults
            │
            ├─► Text
            ├─► Score
            └─► Metadata
```

### 7.3 Server Request Flow

```
HTTP/gRPC Request
    │
    ▼
Middleware Stack
    │
    ├─► Request ID
    ├─► Authentication
    ├─► Rate Limiting
    └─► Logging/Tracing
    │
    ▼
Route Handler
    │
    ├─► Validate request
    ├─► Get/Create index
    └─► Execute operation
    │
    ▼
LeannCore (Builder/Searcher)
    │
    └─► (same as above flows)
    │
    ▼
Response
    │
    ├─► Serialize to JSON/Protobuf
    ├─► Compress
    └─► Return to client
```

---

## 8. API Design

### 8.1 Library API (Rust)

```rust
// Simple usage
use leann_core::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Build index
    let mut builder = LeannBuilder::new();
    builder.add_text("Document 1", None).await?;
    builder.add_text("Document 2", None).await?;
    let index_path = builder.build_index("my-index").await?;

    // Search index
    let searcher = LeannSearcher::from_path(&index_path)?;
    let results = searcher.search("query", 10).await?;

    for result in results {
        println!("{}: {}", result.score, result.text);
    }

    Ok(())
}

// Advanced usage with configuration
let builder = LeannBuilder::new()
    .backend(HnswBackend::new(HnswConfig {
        graph_degree: 64,
        ef_construction: 128,
        ef_search: 64,
    }))
    .embedding_model(SentenceTransformersModel::new("all-MiniLM-L6-v2")?)
    .config(LeannConfig {
        storage: StorageConfig {
            compact: true,
            recompute: true,
            memory_map: true,
        },
        ..Default::default()
    });
```

### 8.2 REST API

```http
# Create index
POST /api/v1/indexes
Content-Type: application/json

{
  "name": "my-docs",
  "backend": "hnsw",
  "embedding_model": "all-MiniLM-L6-v2",
  "config": {
    "graph_degree": 32,
    "compact": true
  }
}

# Add documents
POST /api/v1/indexes/my-docs/documents
Content-Type: application/json

{
  "documents": [
    {
      "text": "Document 1 content",
      "metadata": {"category": "tech"}
    },
    {
      "text": "Document 2 content",
      "metadata": {"category": "science"}
    }
  ]
}

# Search
POST /api/v1/indexes/my-docs/search
Content-Type: application/json

{
  "query": "vector database",
  "top_k": 10,
  "metadata_filter": {
    "$and": [
      {"category": {"$eq": "tech"}},
      {"year": {"$gte": 2020}}
    ]
  }
}

# Response
{
  "results": [
    {
      "id": 0,
      "text": "Document 1 content",
      "score": 0.95,
      "metadata": {"category": "tech"}
    }
  ],
  "total": 1,
  "query_time_ms": 1.5
}
```

### 8.3 gRPC API

```protobuf
// vyakti.proto
syntax = "proto3";

package vyakti.v1;

service Leann {
  rpc CreateIndex(CreateIndexRequest) returns (CreateIndexResponse);
  rpc AddDocuments(AddDocumentsRequest) returns (AddDocumentsResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc StreamSearch(SearchRequest) returns (stream SearchResult);
}

message SearchRequest {
  string index_name = 1;
  string query = 2;
  int32 top_k = 3;
  MetadataFilter metadata_filter = 4;
}

message SearchResponse {
  repeated SearchResult results = 1;
  double query_time_ms = 2;
}

message SearchResult {
  uint64 id = 1;
  string text = 2;
  float score = 3;
  map<string, Value> metadata = 4;
}
```

---

## 9. Deployment Modes

### 9.1 Library Mode

**Use Case**: Embed in Rust applications

**Configuration**:
```toml
[dependencies]
vyakti-core = "0.1.0"
vyakti-backend-hnsw = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

**Example**:
```rust
let mut builder = LeannBuilder::new();
// ... use directly in application
```

---

### 9.2 CLI Mode

**Use Case**: Standalone tool for data scientists/engineers

**Installation**:
```bash
cargo install vyakti-cli
```

**Usage**:
```bash
leann build my-index --input ./docs
leann search my-index "query"
```

---

### 9.3 Server Mode

**Use Case**: Network-accessible service

**Docker Deployment**:
```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin vyakti-server

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/vyakti-server /usr/local/bin/
EXPOSE 8080 50051
CMD ["vyakti-server"]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vyakti-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vyakti
  template:
    metadata:
      labels:
        app: vyakti
    spec:
      containers:
      - name: vyakti
        image: vyakti/server:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 50051
          name: grpc
        env:
        - name: RUST_LOG
          value: "info"
        - name: LEANN_WORKERS
          value: "4"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

---

## 10. Cross-Cutting Concerns

### 10.1 Error Handling

```rust
// Unified error type
#[derive(Debug, thiserror::Error)]
pub enum LeannError {
    #[error("Index error: {0}")]
    Index(String),

    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Error context with anyhow
use anyhow::Context;

let index = load_index(path)
    .context("Failed to load index")
    .with_context(|| format!("Index path: {}", path.display()))?;
```

### 10.2 Logging and Tracing

```rust
use tracing::{info, debug, error, instrument};

#[instrument(skip(self))]
pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
    info!("Starting search for query: {}, k: {}", query, k);

    let embedding = self.embed_query(query).await?;
    debug!("Query embedded, dimension: {}", embedding.len());

    let results = self.backend.search(&embedding, k).await?;
    info!("Search complete, found {} results", results.len());

    Ok(results)
}

// Structured logging
tracing::info!(
    index_name = %name,
    num_documents = docs.len(),
    backend = %backend_type,
    "Building index"
);
```

### 10.3 Metrics

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct Metrics {
    search_count: Counter,
    search_duration: Histogram,
    index_size: Gauge,
}

impl Metrics {
    pub fn record_search(&self, duration: Duration) {
        self.search_count.inc();
        self.search_duration.observe(duration.as_secs_f64());
    }
}

// Expose metrics endpoint
async fn metrics_handler(State(metrics): State<Arc<Metrics>>) -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode_to_string(&metric_families).unwrap()
}
```

### 10.4 Configuration

```rust
use config::{Config, Environment, File};

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub index: IndexConfig,
    pub logging: LoggingConfig,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let config = Config::builder()
            // Start with defaults
            .add_source(File::with_name("config/default").required(false))
            // Environment-specific config
            .add_source(File::with_name(&format!("config/{}", env)).required(false))
            // Environment variables (LEANN_*)
            .add_source(Environment::with_prefix("LEANN").separator("__"))
            .build()?;

        config.try_deserialize()
    }
}
```

---

## 11. Technology Stack

### 11.1 Core Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| **Async Runtime** | `tokio` | Async I/O, concurrency |
| **Serialization** | `serde`, `bincode` | Data serialization |
| **CLI** | `clap` | Command-line parsing |
| **HTTP Server** | `axum` | REST API |
| **gRPC** | `tonic`, `prost` | gRPC implementation |
| **Embedding** | `ort`, `tokenizers` | ONNX models, tokenization |
| **HTTP Client** | `reqwest` | API calls |
| **Error Handling** | `thiserror`, `anyhow` | Error types, context |
| **Logging** | `tracing`, `tracing-subscriber` | Structured logging |
| **Metrics** | `prometheus` | Metrics collection |
| **Configuration** | `config` | Config management |
| **Memory Mapping** | `memmap2` | Efficient file I/O |

### 11.2 Development Tools

| Tool | Purpose |
|------|---------|
| `cargo-watch` | Auto-rebuild on changes |
| `cargo-edit` | Manage dependencies |
| `cargo-outdated` | Check outdated deps |
| `cargo-audit` | Security vulnerabilities |
| `cargo-tarpaulin` | Code coverage |
| `cargo-flamegraph` | Performance profiling |
| `criterion` | Benchmarking |

---

## 12. Design Patterns

### 12.1 Builder Pattern

```rust
let searcher = LeannSearcher::builder()
    .backend(hnsw_backend)
    .embedding_provider(model)
    .cache_size(1000)
    .build()?;
```

### 12.2 Strategy Pattern

```rust
pub trait Backend {
    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
}

// Runtime selection
let backend: Box<dyn Backend> = match backend_type {
    "hnsw" => Box::new(HnswBackend::new(config)),
    "diskann" => Box::new(DiskAnnBackend::new(config)),
    _ => unreachable!(),
};
```

### 12.3 Repository Pattern

```rust
pub trait IndexRepository {
    async fn save(&self, index: &Index) -> Result<()>;
    async fn load(&self, name: &str) -> Result<Index>;
    async fn delete(&self, name: &str) -> Result<()>;
}

pub struct FileSystemRepository {
    base_path: PathBuf,
}

pub struct S3Repository {
    bucket: String,
    client: S3Client,
}
```

### 12.4 Dependency Injection

```rust
pub struct LeannService {
    backend: Arc<dyn Backend>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    repository: Arc<dyn IndexRepository>,
}

impl LeannService {
    pub fn new(
        backend: Arc<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        repository: Arc<dyn IndexRepository>,
    ) -> Self {
        Self { backend, embedding_provider, repository }
    }
}
```

---

## Conclusion

This modular design provides:

1. **Clear Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Extensibility**: New backends, embedding providers, and storage formats can be added without modifying core code
3. **Testability**: Modules can be tested in isolation with mock dependencies
4. **Flexibility**: Same codebase supports library, CLI, and server modes
5. **Performance**: Zero-cost abstractions and efficient data structures
6. **Maintainability**: Well-documented interfaces and consistent patterns

The architecture is designed to evolve with the project while maintaining backward compatibility and code quality.

---

**Document End**
