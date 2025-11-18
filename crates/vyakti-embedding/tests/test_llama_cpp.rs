//! Integration tests for llama.cpp embedding provider

use vyakti_common::EmbeddingProvider;
use vyakti_embedding::{LlamaCppConfig, LlamaCppProvider, ensure_model};

/// Test basic embedding generation with llama.cpp
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_llama_cpp_basic_embedding() {
    // Ensure model is available (will download on first run)
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Create llama.cpp config
    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 0, // CPU only for testing
        n_ctx: 512,
        n_threads: 2,
        dimension: 1024,
        normalize: true,
    };

    // Create provider
    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create LlamaCppProvider");

    // Test single embedding
    let texts = vec!["Hello, world!".to_string()];
    let embeddings = provider.embed(&texts).await.expect("Failed to generate embedding");

    // Verify results
    assert_eq!(embeddings.len(), 1, "Should generate 1 embedding");
    assert_eq!(embeddings[0].len(), 1024, "Embedding should have 1024 dimensions");

    // Verify normalization (L2 norm should be ~1.0)
    let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized (L2 norm ~1.0), got {}", norm);
}

/// Test batch embedding generation
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_llama_cpp_batch_embedding() {
    // Ensure model is available
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Create llama.cpp config
    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 0,
        n_ctx: 512,
        n_threads: 2,
        dimension: 1024,
        normalize: true,
    };

    // Create provider
    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create LlamaCppProvider");

    // Test batch embedding
    let texts = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Rust is a systems programming language".to_string(),
    ];

    let embeddings = provider.embed(&texts).await.expect("Failed to generate embeddings");

    // Verify results
    assert_eq!(embeddings.len(), 3, "Should generate 3 embeddings");
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 1024, "Embedding {} should have 1024 dimensions", i);

        // Verify normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding {} should be normalized (L2 norm ~1.0), got {}", i, norm);
    }

    // Verify embeddings are different (cosine similarity should not be 1.0)
    let similarity_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let similarity_02 = cosine_similarity(&embeddings[0], &embeddings[2]);

    assert!(similarity_01 < 0.99, "Embeddings 0 and 1 should be different, similarity: {}", similarity_01);
    assert!(similarity_02 < 0.99, "Embeddings 0 and 2 should be different, similarity: {}", similarity_02);
}

/// Test embedding provider interface methods
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_llama_cpp_provider_interface() {
    // Ensure model is available
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Create llama.cpp config
    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 0,
        n_ctx: 512,
        n_threads: 2,
        dimension: 1024,
        normalize: true,
    };

    // Create provider
    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create LlamaCppProvider");

    // Test dimension() method
    assert_eq!(provider.dimension(), 1024, "Dimension should be 1024");

    // Test name() method
    assert_eq!(provider.name(), "llama-cpp", "Provider name should be 'llama-cpp'");
}

/// Helper function to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot_product / (norm_a * norm_b)
}

/// Test model download functionality
#[tokio::test]
#[ignore] // Ignore by default as it requires network access
async fn test_model_download() {
    // Test ensure_model with None (should download default model)
    let result = ensure_model(None).await;

    assert!(result.is_ok(), "Model download should succeed");

    let model_path = result.unwrap();
    assert!(model_path.exists(), "Model file should exist after download");
    assert!(model_path.is_file(), "Model path should be a file");
}

/// Test that provider works with custom thread count
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_llama_cpp_custom_threads() {
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Test with 1 thread
    let config = LlamaCppConfig {
        model_path: model_path.clone(),
        n_gpu_layers: 0,
        n_ctx: 512,
        n_threads: 1,
        dimension: 1024,
        normalize: true,
    };

    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create LlamaCppProvider with 1 thread");

    let texts = vec!["Test text".to_string()];
    let embeddings = provider.embed(&texts).await.expect("Failed to generate embedding");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 1024);

    // Test with 4 threads
    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 0,
        n_ctx: 512,
        n_threads: 4,
        dimension: 1024,
        normalize: true,
    };

    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create LlamaCppProvider with 4 threads");

    let embeddings = provider.embed(&texts).await.expect("Failed to generate embedding");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 1024);
}
