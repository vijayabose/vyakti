use criterion::{criterion_group, criterion_main, Criterion};

fn hnsw_search_benchmark(c: &mut Criterion) {
    c.bench_function("hnsw_search", |b| {
        b.iter(|| {
            // Placeholder benchmark
        })
    });
}

criterion_group!(benches, hnsw_search_benchmark);
criterion_main!(benches);
