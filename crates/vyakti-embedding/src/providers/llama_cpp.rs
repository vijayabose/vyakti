//! llama.cpp-based embedding provider

#[cfg(feature = "llama-cpp")]
use async_trait::async_trait;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::context::params::LlamaContextParams;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::llama_backend::LlamaBackend;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::llama_batch::LlamaBatch;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::model::params::LlamaModelParams;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::model::LlamaModel;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::context::LlamaContext;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::model::AddBos;
#[cfg(feature = "llama-cpp")]
use parking_lot::Mutex;
#[cfg(feature = "llama-cpp")]
use std::path::PathBuf;
#[cfg(feature = "llama-cpp")]
use std::sync::Arc;
#[cfg(feature = "llama-cpp")]
use tracing::{debug, info};
#[cfg(feature = "llama-cpp")]
use vyakti_common::{EmbeddingProvider, VyaktiError, Result, Vector};

#[cfg(feature = "llama-cpp")]
/// Configuration for llama.cpp embedding provider
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,
    /// Number of layers to offload to GPU (0 = CPU only)
    pub n_gpu_layers: u32,
    /// Context size
    pub n_ctx: u32,
    /// Number of threads for inference
    pub n_threads: u32,
    /// Embedding dimension (1024 for mxbai-embed-large)
    pub dimension: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
}

#[cfg(feature = "llama-cpp")]
impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_gpu_layers: 0,
            n_ctx: 512,
            n_threads: num_cpus::get() as u32,
            dimension: 1024,
            normalize: true,
        }
    }
}

#[cfg(feature = "llama-cpp")]
/// llama.cpp-based embedding provider
pub struct LlamaCppProvider {
    model: Arc<LlamaModel>,
    context: Arc<Mutex<LlamaContext<'static>>>,
    config: LlamaCppConfig,
}

#[cfg(feature = "llama-cpp")]
impl LlamaCppProvider {
    /// Create a new llama.cpp embedding provider
    pub fn new(config: LlamaCppConfig) -> Result<Self> {
        info!("Initializing llama.cpp embedding provider");
        debug!("Model path: {}", config.model_path.display());
        debug!("GPU layers: {}", config.n_gpu_layers);
        debug!("Context size: {}", config.n_ctx);
        debug!("Threads: {}", config.n_threads);

        // Initialize backend
        let backend = LlamaBackend::init()
            .map_err(|e| VyaktiError::Embedding(format!("Failed to initialize llama.cpp backend: {}", e)))?;

        // Load model
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers);

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .map_err(|e| VyaktiError::Embedding(format!("Failed to load model: {}", e)))?;

        // Wrap model in Arc first so context can borrow from it
        let model = Arc::new(model);

        // Create context - it borrows from the Arc'ed model
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(config.n_ctx))
            .with_n_threads(config.n_threads)
            .with_embeddings(true);

        let context = model.new_context(&backend, ctx_params)
            .map_err(|e| VyaktiError::Embedding(format!("Failed to create context: {}", e)))?;

        info!("llama.cpp provider initialized successfully");

        Ok(Self {
            model,
            context: Arc::new(Mutex::new(context)),
            config,
        })
    }

    /// Compute embedding for a single text
    fn compute_single_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let mut context = self.context.lock();

        // Tokenize text - add BOS (beginning of sentence) token
        let tokens = self.model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| VyaktiError::Embedding(format!("Tokenization failed: {}", e)))?;

        // Create batch
        let mut batch = LlamaBatch::new(self.config.n_ctx as usize, 1);

        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], false)
                .map_err(|e| VyaktiError::Embedding(format!("Failed to add token to batch: {}", e)))?;
        }

        // Decode
        context.decode(&mut batch)
            .map_err(|e| VyaktiError::Embedding(format!("Decode failed: {}", e)))?;

        // Get embeddings
        let embeddings = context.embeddings_seq_ith(0)
            .ok_or_else(|| VyaktiError::Embedding("Failed to get embeddings".to_string()))?;

        let mut embedding = embeddings.to_vec();

        // Normalize if requested
        if self.config.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                embedding.iter_mut().for_each(|x| *x /= norm);
            }
        }

        Ok(embedding)
    }
}

#[cfg(feature = "llama-cpp")]
#[async_trait]
impl EmbeddingProvider for LlamaCppProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        debug!("Computing embeddings for {} texts", texts.len());

        // Process each text sequentially
        // Note: llama.cpp contexts are not thread-safe, so we use a mutex
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = tokio::task::spawn_blocking({
                let provider = self.clone();
                let text = text.clone();
                move || provider.compute_single_embedding(&text)
            })
            .await
            .map_err(|e| VyaktiError::Embedding(format!("Task join error: {}", e)))??;

            embeddings.push(embedding);
        }

        debug!("Generated {} embeddings", embeddings.len());
        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn name(&self) -> &str {
        "llama-cpp"
    }
}

#[cfg(feature = "llama-cpp")]
impl Clone for LlamaCppProvider {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            context: Arc::clone(&self.context),
            config: self.config.clone(),
        }
    }
}

#[cfg(feature = "llama-cpp")]
// Safe because we use Arc<Mutex<>> for internal state
unsafe impl Sync for LlamaCppProvider {}

// Re-export for convenience
#[cfg(feature = "llama-cpp")]
pub use num_cpus;
