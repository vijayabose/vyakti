//! GPU-specific tests for llama.cpp embedding provider
//!
//! These tests verify GPU configuration and acceleration functionality.
//! Run with: cargo test --package vyakti-embedding --test test_llama_cpp_gpu -- --ignored

use vyakti_common::EmbeddingProvider;
use vyakti_embedding::{LlamaCppConfig, LlamaCppProvider, ensure_model};

/// Test GPU layer configuration is properly set
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_gpu_layers_configuration() {
    // Ensure model is available
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Test with different GPU layer counts
    for gpu_layers in [0, 16, 32, 64] {
        let config = LlamaCppConfig {
            model_path: model_path.clone(),
            n_gpu_layers: gpu_layers,
            n_ctx: 512,
            n_threads: 2,
            dimension: 1024,
            normalize: true,
        };

        // Create provider - should succeed even without GPU
        let provider = LlamaCppProvider::new(config.clone())
            .expect(&format!("Failed to create provider with {} GPU layers", gpu_layers));

        // Verify provider works
        assert_eq!(provider.dimension(), 1024, "Dimension should be 1024");

        // Test embedding generation
        let texts = vec!["GPU test".to_string()];
        let embeddings = provider.embed(&texts).await
            .expect(&format!("Failed to generate embedding with {} GPU layers", gpu_layers));

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1024);
    }
}

/// Test that provider works with GPU acceleration if available
#[tokio::test]
#[ignore] // Ignore by default - requires CUDA/GPU
async fn test_gpu_acceleration() {
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    // Try with full GPU offload
    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 999, // Large number to offload all layers
        n_ctx: 512,
        n_threads: 2,
        dimension: 1024,
        normalize: true,
    };

    // This should work even if GPU is not available (will fall back to CPU)
    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create provider with GPU acceleration");

    // Generate embeddings
    let texts = vec![
        "GPU acceleration test 1".to_string(),
        "GPU acceleration test 2".to_string(),
        "GPU acceleration test 3".to_string(),
    ];

    let embeddings = provider.embed(&texts).await
        .expect("Failed to generate embeddings with GPU");

    assert_eq!(embeddings.len(), 3);
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 1024, "Embedding {} should have 1024 dimensions", i);
    }

    println!("✓ GPU acceleration test passed (may have used CPU fallback if no GPU available)");
}

/// Test CPU-only mode explicitly
#[tokio::test]
#[ignore] // Ignore by default as it requires model download
async fn test_cpu_only_mode() {
    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    let config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 0, // Explicitly CPU-only
        n_ctx: 512,
        n_threads: num_cpus::get() as u32,
        dimension: 1024,
        normalize: true,
    };

    let provider = LlamaCppProvider::new(config)
        .expect("Failed to create CPU-only provider");

    // Generate embeddings
    let texts = vec!["CPU-only test".to_string()];
    let embeddings = provider.embed(&texts).await
        .expect("Failed to generate embeddings in CPU-only mode");

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 1024);

    println!("✓ CPU-only mode test passed");
}

/// Benchmark CPU vs GPU performance (if GPU available)
#[tokio::test]
#[ignore] // Manual test for performance comparison
async fn benchmark_cpu_vs_gpu() {
    use std::time::Instant;

    let model_path = ensure_model(None)
        .await
        .expect("Failed to ensure model is available");

    let test_texts: Vec<String> = (0..10)
        .map(|i| format!("Benchmark test sentence number {}", i))
        .collect();

    // Test CPU-only
    let cpu_config = LlamaCppConfig {
        model_path: model_path.clone(),
        n_gpu_layers: 0,
        n_ctx: 512,
        n_threads: num_cpus::get() as u32,
        dimension: 1024,
        normalize: true,
    };

    let cpu_provider = LlamaCppProvider::new(cpu_config)
        .expect("Failed to create CPU provider");

    let cpu_start = Instant::now();
    let _cpu_embeddings = cpu_provider.embed(&test_texts).await
        .expect("CPU embedding failed");
    let cpu_duration = cpu_start.elapsed();

    println!("CPU-only: {:?} for {} embeddings", cpu_duration, test_texts.len());
    println!("CPU avg: {:?} per embedding", cpu_duration / test_texts.len() as u32);

    // Test with GPU (if available)
    let gpu_config = LlamaCppConfig {
        model_path,
        n_gpu_layers: 32, // Offload 32 layers to GPU
        n_ctx: 512,
        n_threads: 2,
        dimension: 1024,
        normalize: true,
    };

    let gpu_provider = LlamaCppProvider::new(gpu_config)
        .expect("Failed to create GPU provider");

    let gpu_start = Instant::now();
    let _gpu_embeddings = gpu_provider.embed(&test_texts).await
        .expect("GPU embedding failed");
    let gpu_duration = gpu_start.elapsed();

    println!("GPU (32 layers): {:?} for {} embeddings", gpu_duration, test_texts.len());
    println!("GPU avg: {:?} per embedding", gpu_duration / test_texts.len() as u32);

    if gpu_duration < cpu_duration {
        println!("✓ GPU is faster (speedup: {:.2}x)", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());
    } else if gpu_duration > cpu_duration {
        println!("⚠ GPU is slower - may not be available or overhead dominates for small batches");
    } else {
        println!("⚠ Similar performance - GPU may not be utilized");
    }
}
