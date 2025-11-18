//! Integration test for hybrid search functionality
//!
//! This test validates that:
//! 1. Hybrid indexes can be built successfully
//! 2. All fusion strategies work correctly
//! 3. Exact keyword matches are ranked higher with hybrid search
//! 4. Hybrid + compact mode works together

use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use vyakti_backend_hnsw::HnswBackend;
use vyakti_common::{Backend, BackendConfig, EmbeddingProvider, Result, Vector};
use vyakti_core::{FusionStrategy, HybridSearcher, VyaktiBuilder};
use vyakti_keyword::KeywordConfig;

// Mock embedding provider for testing
struct TestEmbeddingProvider {
    dimension: usize,
}

impl TestEmbeddingProvider {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    // Generate deterministic embeddings based on text content
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
impl EmbeddingProvider for TestEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "test-provider"
    }
}

/// Test building a basic hybrid index
#[tokio::test]
async fn test_build_hybrid_index() {
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("hybrid-test");

    // Create test documents
    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(TestEmbeddingProvider::new(384));
    let config = BackendConfig::default();

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider, config);

    // Add test documents with distinct content
    builder.add_text("The mxbai-embed-large model is powerful", None);
    builder.add_text("Vector databases enable semantic search", None);
    builder.add_text("BM25 is a keyword ranking algorithm", None);
    builder.add_text("Hybrid search combines both approaches", None);

    // Build hybrid index with keyword search
    let keyword_config = KeywordConfig {
        enabled: true,
        k1: 1.2,
        b: 0.75,
    };

    let result = builder.build_index_hybrid(&index_path, Some(keyword_config)).await;
    assert!(result.is_ok(), "Failed to build hybrid index: {:?}", result.err());

    // Verify index structure
    assert!(index_path.exists(), "Index directory not created");
    assert!(index_path.join("index.json").exists(), "index.json not created");
    assert!(index_path.join("keyword").exists(), "keyword directory not created");
}

/// Test searching with different fusion strategies
#[tokio::test]
async fn test_fusion_strategies() {
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("fusion-test");

    // Build hybrid index
    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(TestEmbeddingProvider::new(384));
    let config = BackendConfig::default();

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider.clone(), config.clone());

    builder.add_text("Document about mxbai-embed-large model", None);
    builder.add_text("Document about vector databases", None);
    builder.add_text("Document about BM25 algorithm", None);
    builder.add_text("Document mentioning mxbai-embed-large specifically", None);

    let keyword_config = KeywordConfig::default();
    builder.build_index_hybrid(&index_path, Some(keyword_config)).await.unwrap();

    // Load index
    let index_data = vyakti_core::load_index(&index_path.join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    // Helper function to create and initialize backend
    let create_backend = || async {
        let mut backend = Box::new(HnswBackend::new());
        backend.build(&index_data.vectors, &config).await.unwrap();
        for doc in &index_data.documents {
            backend.set_document_data(doc.id, doc.text.clone(), doc.metadata.clone());
        }
        backend
    };

    // Test RRF fusion
    let backend = create_backend().await;
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider.clone(),
        FusionStrategy::RRF { k: 60 },
        documents.clone(),
    ).unwrap();

    let rrf_results = hybrid_searcher.search("mxbai-embed-large", 4).await.unwrap();
    assert!(!rrf_results.is_empty(), "RRF search returned no results");

    // Test Weighted fusion
    let backend = create_backend().await;
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider.clone(),
        FusionStrategy::Weighted { alpha: 0.7 },
        documents.clone(),
    ).unwrap();

    let weighted_results = hybrid_searcher.search("BM25", 4).await.unwrap();
    assert!(!weighted_results.is_empty(), "Weighted search returned no results");

    // Test Cascade fusion (use threshold of 1 to ensure keyword results are used)
    let backend = create_backend().await;
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider.clone(),
        FusionStrategy::Cascade { threshold: 1 },
        documents.clone(),
    ).unwrap();

    let cascade_results = hybrid_searcher.search("vector", 4).await.unwrap();
    assert!(!cascade_results.is_empty(), "Cascade search returned no results");

    // Test VectorOnly mode
    let backend = create_backend().await;
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider.clone(),
        FusionStrategy::VectorOnly,
        documents.clone(),
    ).unwrap();

    let vector_results = hybrid_searcher.search("semantic", 4).await.unwrap();
    assert!(!vector_results.is_empty(), "Vector-only search returned no results");
}

/// Test that hybrid search improves exact keyword matching
#[tokio::test]
async fn test_exact_keyword_matching() {
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("keyword-test");

    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(TestEmbeddingProvider::new(384));
    let config = BackendConfig::default();

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider.clone(), config);

    // Add documents where exact keyword match is important
    builder.add_text("This document discusses general embedding models", None);
    builder.add_text("Various neural networks for text processing", None);
    builder.add_text("The mxbai-embed-large model is highly efficient", None);
    builder.add_text("Machine learning approaches to NLP", None);

    let keyword_config = KeywordConfig::default();
    builder.build_index_hybrid(&index_path, Some(keyword_config)).await.unwrap();

    // Load index and search
    let index_data = vyakti_core::load_index(&index_path.join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let backend = Box::new(HnswBackend::new());
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::RRF { k: 60 },
        documents,
    ).unwrap();

    // Search for exact keyword
    let results = hybrid_searcher.search("mxbai-embed-large", 4).await.unwrap();

    assert!(!results.is_empty(), "No results found");

    // The document containing the exact keyword should be in top results
    let top_result = &results[0];
    assert!(
        top_result.text.contains("mxbai-embed-large"),
        "Top result doesn't contain exact keyword. Got: {}",
        top_result.text
    );
}

/// Test hybrid search with compact mode
#[tokio::test]
async fn test_hybrid_compact_mode() {
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("hybrid-compact-test");

    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(TestEmbeddingProvider::new(384));
    let config = BackendConfig {
        graph_degree: 16,
        build_complexity: 32,
        search_complexity: 16,
    };

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider.clone(), config);

    // Add enough documents for meaningful pruning
    for i in 0..50 {
        builder.add_text(
            format!("Document {} about various topics including search and retrieval", i),
            None,
        );
    }

    // Add a document with specific keyword
    builder.add_text("Special document mentioning mxbai-embed-large model", None);

    let keyword_config = KeywordConfig::default();
    let (path, stats) = builder
        .build_index_hybrid_compact(&index_path, Some(keyword_config), None)
        .await
        .unwrap();

    // Verify compact mode achieved storage savings
    assert!(
        stats.savings_percent > 20.0,
        "Expected >20% savings, got {}%",
        stats.savings_percent
    );

    // Verify hybrid index structure
    assert!(path.join("index.json").exists());
    assert!(path.join("keyword").exists());

    // Load and search the compact hybrid index
    let index_data = vyakti_core::load_index(&path.join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let backend = Box::new(HnswBackend::new());
    let hybrid_searcher = HybridSearcher::load(
        &path,
        backend,
        embedding_provider,
        FusionStrategy::RRF { k: 60 },
        documents,
    ).unwrap();

    let results = hybrid_searcher.search("mxbai-embed-large", 5).await.unwrap();

    assert!(!results.is_empty(), "Compact hybrid search returned no results");

    // Verify exact keyword match is found despite compact mode
    let found_exact = results.iter().any(|r| r.text.contains("mxbai-embed-large"));
    assert!(found_exact, "Exact keyword not found in compact hybrid search results");
}

/// Test keyword-only search mode
#[tokio::test]
async fn test_keyword_only_mode() {
    let temp_dir = tempdir().unwrap();
    let index_path = temp_dir.path().join("keyword-only-test");

    let backend = Box::new(HnswBackend::new());
    let embedding_provider = Arc::new(TestEmbeddingProvider::new(384));
    let config = BackendConfig::default();

    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider.clone(), config);

    builder.add_text("BM25 is the best matching algorithm", None);
    builder.add_text("Vector search uses embeddings", None);
    builder.add_text("BM25 parameters include k1 and b", None);
    builder.add_text("Semantic search with neural networks", None);

    let keyword_config = KeywordConfig::default();
    builder.build_index_hybrid(&index_path, Some(keyword_config)).await.unwrap();

    let index_data = vyakti_core::load_index(&index_path.join("index.json")).unwrap();

    let documents: Vec<(String, HashMap<String, serde_json::Value>)> = index_data
        .documents
        .iter()
        .map(|doc| (doc.text.clone(), doc.metadata.clone()))
        .collect();

    let backend = Box::new(HnswBackend::new());
    let hybrid_searcher = HybridSearcher::load(
        &index_path,
        backend,
        embedding_provider,
        FusionStrategy::KeywordOnly,
        documents,
    ).unwrap();

    let results = hybrid_searcher.search("BM25", 4).await.unwrap();

    assert!(!results.is_empty(), "Keyword-only search returned no results");

    // In keyword-only mode, documents containing "BM25" should be top results
    let top_two_contain_keyword = results.iter().take(2).all(|r| r.text.contains("BM25"));
    assert!(
        top_two_contain_keyword,
        "Top results in keyword-only mode should contain the search keyword"
    );
}
