//! Benchmark comparing compact mode vs normal mode performance
//!
//! This benchmark measures:
//! - Index build time (normal vs compact)
//! - Search latency (normal vs compact)
//! - Search throughput (normal vs compact)
//! - Memory efficiency

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::sync::Arc;
use vyakti_backend_hnsw::HnswBackend;
use vyakti_common::BackendConfig;
use vyakti_core::VyaktiBuilder;

// Mock embedding provider for benchmarks (fast, no network calls)
struct BenchmarkEmbeddingProvider {
    dimension: usize,
}

impl BenchmarkEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    // Generate deterministic but varied vectors for realistic graph structure
    fn generate_embedding(&self, doc_id: usize) -> Vec<f32> {
        let mut vec = vec![0.0; self.dimension];

        // Create varied vectors with some clustering patterns
        let cluster = (doc_id % 10) as f32 / 10.0; // 10 clusters
        let noise = ((doc_id * 7919) % 1000) as f32 / 1000.0; // Pseudo-random noise

        for i in 0..self.dimension {
            let base = if i % 3 == 0 {
                cluster
            } else if i % 3 == 1 {
                1.0 - cluster
            } else {
                0.5
            };
            vec[i] = base + noise * 0.1;
        }

        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec {
            *v /= norm;
        }

        vec
    }
}

#[async_trait::async_trait]
impl vyakti_common::EmbeddingProvider for BenchmarkEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> vyakti_common::Result<Vec<Vec<f32>>> {
        // Parse doc IDs from text (format: "Document N")
        Ok(texts
            .iter()
            .map(|text| {
                let doc_id = text
                    .split_whitespace()
                    .last()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                self.generate_embedding(doc_id)
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "benchmark-mock"
    }
}

/// Build index in normal mode
async fn build_normal_index(num_docs: usize, dimension: usize) -> VyaktiBuilder {
    let config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(config.clone()));
    let embedding_provider = Arc::new(BenchmarkEmbeddingProvider::new(dimension));

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider, config);

    for i in 0..num_docs {
        builder.add_text(format!("Document {}", i), None);
    }

    builder.build().await.unwrap();
    builder
}

/// Build index in compact mode
async fn build_compact_index(num_docs: usize, dimension: usize) -> VyaktiBuilder {
    let config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(config.clone()));
    let embedding_provider = Arc::new(BenchmarkEmbeddingProvider::new(dimension));

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider, config);

    for i in 0..num_docs {
        builder.add_text(format!("Document {}", i), None);
    }

    builder.build_compact(None).await.unwrap();
    builder
}

/// Benchmark index building time
fn bench_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_build_time");

    for num_docs in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("normal", num_docs),
            num_docs,
            |b, &num_docs| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| build_normal_index(num_docs, 384));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compact", num_docs),
            num_docs,
            |b, &num_docs| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| build_compact_index(num_docs, 384));
            },
        );
    }

    group.finish();
}

/// Benchmark search latency
fn bench_search_latency(c: &mut Criterion) {
    // Build indexes once for search benchmarks
    let rt = tokio::runtime::Runtime::new().unwrap();

    let normal_builder = rt.block_on(build_normal_index(1000, 384));
    let compact_builder = rt.block_on(build_compact_index(1000, 384));

    let mut group = c.benchmark_group("search_latency");

    // Benchmark normal mode search
    group.bench_function("normal_mode", |b| {
        b.to_async(&rt).iter(|| async {
            let backend = normal_builder.backend();
            let backend_guard = backend.read().await;

            // Search with a query vector
            let query_vec = BenchmarkEmbeddingProvider::new(384).generate_embedding(42);
            backend_guard.search(&query_vec, 10).await.unwrap()
        });
    });

    // Benchmark compact mode search (with recomputation)
    group.bench_function("compact_mode", |b| {
        b.to_async(&rt).iter(|| async {
            let backend = compact_builder.backend();
            let backend_guard = backend.read().await;

            // Search with a query vector
            let query_vec = BenchmarkEmbeddingProvider::new(384).generate_embedding(42);
            backend_guard.search(&query_vec, 10).await.unwrap()
        });
    });

    group.finish();
}

/// Benchmark search throughput (queries per second)
fn bench_search_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let normal_builder = rt.block_on(build_normal_index(1000, 384));
    let compact_builder = rt.block_on(build_compact_index(1000, 384));

    let mut group = c.benchmark_group("search_throughput");

    // Generate query vectors
    let provider = BenchmarkEmbeddingProvider::new(384);
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| provider.generate_embedding(i + 5000))
        .collect();

    // Benchmark normal mode throughput
    group.bench_function("normal_mode_batch", |b| {
        b.to_async(&rt).iter(|| async {
            let backend = normal_builder.backend();
            let backend_guard = backend.read().await;

            for query in &queries {
                let _ = black_box(backend_guard.search(query, 10).await.unwrap());
            }
        });
    });

    // Benchmark compact mode throughput
    group.bench_function("compact_mode_batch", |b| {
        b.to_async(&rt).iter(|| async {
            let backend = compact_builder.backend();
            let backend_guard = backend.read().await;

            for query in &queries {
                let _ = black_box(backend_guard.search(query, 10).await.unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_build_time,
    bench_search_latency,
    bench_search_throughput
);
criterion_main!(benches);
