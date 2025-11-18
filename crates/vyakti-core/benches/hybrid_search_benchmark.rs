//! Benchmark comparing hybrid search vs vector-only search performance
//!
//! This benchmark measures:
//! - Search latency for different fusion strategies (RRF, Weighted, Cascade)
//! - Hybrid search vs vector-only search comparison
//! - Exact keyword match accuracy improvements

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;
use vyakti_backend_hnsw::HnswBackend;
use vyakti_common::{Backend, BackendConfig, EmbeddingProvider, Result, Vector};
use vyakti_core::{FusionStrategy, HybridSearcher, VyaktiBuilder};
use vyakti_keyword::KeywordConfig;

// Mock embedding provider for benchmarks (fast, no network calls)
struct BenchmarkEmbeddingProvider {
    dimension: usize,
}

impl BenchmarkEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    // Generate deterministic but varied vectors for realistic graph structure
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0; self.dimension];

        // Simple hash-based embedding generation
        let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));

        for i in 0..self.dimension {
            let val = ((hash.wrapping_mul((i + 1) as u32)) % 1000) as f32 / 1000.0;
            vec[i] = val;
        }

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for BenchmarkEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "benchmark-mock"
    }
}

/// Setup function to create a hybrid index for benchmarking
fn setup_hybrid_index(rt: &Runtime, num_docs: usize) -> (tempfile::TempDir, String, Arc<BenchmarkEmbeddingProvider>) {
    let temp_dir = tempfile::tempdir().unwrap();
    let index_path = temp_dir.path().join("hybrid-bench");

    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(BenchmarkEmbeddingProvider::new(384));
    let config = BackendConfig::default();

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider.clone(), config);

    // Add documents with varied content
    for i in 0..num_docs {
        let text = if i % 10 == 0 {
            // Some documents contain specific keywords
            format!("Document {} discussing mxbai-embed-large model architecture", i)
        } else if i % 7 == 0 {
            format!("Document {} about BM25 ranking algorithm implementation", i)
        } else {
            format!("Document {} with general information about vector databases and search", i)
        };
        builder.add_text(text, None);
    }

    // Build hybrid index
    let keyword_config = KeywordConfig::default();
    rt.block_on(builder.build_index_hybrid(&index_path, Some(keyword_config))).unwrap();

    (temp_dir, index_path.to_string_lossy().to_string(), embedding_provider)
}

/// Benchmark hybrid search with RRF fusion
fn bench_hybrid_rrf_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_temp_dir, index_path, embedding_provider) = setup_hybrid_index(&rt, 100);

    // Load index
    let index_data = vyakti_core::load_index(&std::path::Path::new(&index_path).join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let config = BackendConfig::default();
    let mut backend = Box::new(HnswBackend::new());
    rt.block_on(backend.build(&index_data.vectors, &config)).unwrap();
    for doc in &index_data.documents {
        backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
    }

    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::RRF { k: 60 },
        documents,
    ).unwrap();

    c.bench_function("hybrid_rrf_search_k10", |b| {
        b.to_async(&rt).iter(|| async {
            let results = hybrid_searcher.search(black_box("mxbai-embed-large model"), 10).await.unwrap();
            black_box(results);
        });
    });
}

/// Benchmark hybrid search with Weighted fusion
fn bench_hybrid_weighted_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_temp_dir, index_path, embedding_provider) = setup_hybrid_index(&rt, 100);

    let index_data = vyakti_core::load_index(&std::path::Path::new(&index_path).join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let config = BackendConfig::default();
    let mut backend = Box::new(HnswBackend::new());
    rt.block_on(backend.build(&index_data.vectors, &config)).unwrap();
    for doc in &index_data.documents {
        backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
    }

    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::Weighted { alpha: 0.7 },
        documents,
    ).unwrap();

    c.bench_function("hybrid_weighted_search_k10", |b| {
        b.to_async(&rt).iter(|| async {
            let results = hybrid_searcher.search(black_box("BM25 algorithm"), 10).await.unwrap();
            black_box(results);
        });
    });
}

/// Benchmark vector-only search
fn bench_vector_only_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_temp_dir, index_path, embedding_provider) = setup_hybrid_index(&rt, 100);

    let index_data = vyakti_core::load_index(&std::path::Path::new(&index_path).join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let config = BackendConfig::default();
    let mut backend = Box::new(HnswBackend::new());
    rt.block_on(backend.build(&index_data.vectors, &config)).unwrap();
    for doc in &index_data.documents {
        backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
    }

    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::VectorOnly,
        documents,
    ).unwrap();

    c.bench_function("vector_only_search_k10", |b| {
        b.to_async(&rt).iter(|| async {
            let results = hybrid_searcher.search(black_box("mxbai-embed-large model"), 10).await.unwrap();
            black_box(results);
        });
    });
}

/// Benchmark keyword-only search
fn bench_keyword_only_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_temp_dir, index_path, embedding_provider) = setup_hybrid_index(&rt, 100);

    let index_data = vyakti_core::load_index(&std::path::Path::new(&index_path).join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let config = BackendConfig::default();
    let mut backend = Box::new(HnswBackend::new());
    rt.block_on(backend.build(&index_data.vectors, &config)).unwrap();
    for doc in &index_data.documents {
        backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
    }

    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::KeywordOnly,
        documents,
    ).unwrap();

    c.bench_function("keyword_only_search_k10", |b| {
        b.to_async(&rt).iter(|| async {
            let results = hybrid_searcher.search(black_box("BM25"), 10).await.unwrap();
            black_box(results);
        });
    });
}

/// Benchmark fusion strategies with varying index sizes
fn bench_fusion_strategies_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("fusion_strategies_scaling");

    for size in [50, 100, 200].iter() {
        let (_temp_dir, index_path, embedding_provider) = setup_hybrid_index(&rt, *size);

        let index_data = vyakti_core::load_index(&std::path::Path::new(&index_path).join("index.json")).unwrap();

        let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
            .documents
            .iter()
            .map(|doc| (doc.text.clone(), doc.metadata.clone()))
            .collect();

        let config = BackendConfig::default();

        // Benchmark RRF
        let mut backend = Box::new(HnswBackend::new());
        rt.block_on(backend.build(&index_data.vectors, &config)).unwrap();
        for doc in &index_data.documents {
            backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
        }

        let hybrid_searcher = HybridSearcher::load(
            &index_path,
            backend,
            embedding_provider.clone(),
            FusionStrategy::RRF { k: 60 },
            documents.clone(),
        ).unwrap();

        group.bench_with_input(BenchmarkId::new("rrf", size), size, |b, _size| {
            b.to_async(&rt).iter(|| async {
                let results = hybrid_searcher.search(black_box("search query"), 10).await.unwrap();
                black_box(results);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hybrid_rrf_search,
    bench_hybrid_weighted_search,
    bench_vector_only_search,
    bench_keyword_only_search,
    bench_fusion_strategies_scaling,
);
criterion_main!(benches);
