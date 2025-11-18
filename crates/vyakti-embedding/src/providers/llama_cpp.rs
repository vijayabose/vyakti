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
use std::path::PathBuf;
#[cfg(feature = "llama-cpp")]
use tracing::{debug, info, error};
#[cfg(feature = "llama-cpp")]
use vyakti_common::{EmbeddingProvider, VyaktiError, Result, Vector};
#[cfg(feature = "llama-cpp")]
use tokio::sync::{mpsc, oneshot};

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
/// Request to compute embedding
struct EmbeddingRequest {
    text: String,
    response: oneshot::Sender<Result<Vec<f32>>>,
}

#[cfg(feature = "llama-cpp")]
/// llama.cpp-based embedding provider using a dedicated worker thread
pub struct LlamaCppProvider {
    sender: mpsc::UnboundedSender<EmbeddingRequest>,
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

        let (sender, receiver) = mpsc::unbounded_channel();

        // Clone config for the worker thread
        let worker_config = config.clone();

        // Spawn worker thread that owns the model and context
        std::thread::spawn(move || {
            if let Err(e) = Self::worker_thread(receiver, worker_config) {
                error!("llama.cpp worker thread error: {}", e);
            }
        });

        info!("llama.cpp provider initialized successfully");

        Ok(Self {
            sender,
            config,
        })
    }

    /// Worker thread that owns the model and context
    fn worker_thread(
        mut receiver: mpsc::UnboundedReceiver<EmbeddingRequest>,
        config: LlamaCppConfig,
    ) -> Result<()> {
        // Initialize backend
        let _backend = LlamaBackend::init()
            .map_err(|e| VyaktiError::Embedding(format!("Failed to initialize llama.cpp backend: {}", e)))?;

        // Load model
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers);

        let model = LlamaModel::load_from_file(&_backend, &config.model_path, &model_params)
            .map_err(|e| VyaktiError::Embedding(format!("Failed to load model: {}", e)))?;

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(config.n_ctx))
            .with_n_threads(config.n_threads)
            .with_embeddings(true);

        let mut context = model.new_context(&_backend, ctx_params)
            .map_err(|e| VyaktiError::Embedding(format!("Failed to create context: {}", e)))?;

        // Process requests
        while let Some(request) = receiver.blocking_recv() {
            let result = Self::compute_embedding_sync(
                &model,
                &mut context,
                &request.text,
                &config,
            );

            // Send response (ignore if receiver dropped)
            let _ = request.response.send(result);
        }

        Ok(())
    }

    /// Compute embedding synchronously (called from worker thread)
    fn compute_embedding_sync(
        model: &LlamaModel,
        context: &mut LlamaContext,
        text: &str,
        config: &LlamaCppConfig,
    ) -> Result<Vec<f32>> {
        // Tokenize text - add BOS (beginning of sentence) token
        let tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| VyaktiError::Embedding(format!("Tokenization failed: {}", e)))?;

        // Create batch
        let mut batch = LlamaBatch::new(config.n_ctx as usize, 1);

        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], false)
                .map_err(|e| VyaktiError::Embedding(format!("Failed to add token to batch: {}", e)))?;
        }

        // Decode
        context.decode(&mut batch)
            .map_err(|e| VyaktiError::Embedding(format!("Decode failed: {}", e)))?;

        // Get embeddings - check if the result is Ok or Err
        let embeddings = match context.embeddings_seq_ith(0) {
            Ok(emb) => emb,
            Err(e) => return Err(VyaktiError::Embedding(format!("Failed to get embeddings: {}", e))),
        };

        let mut embedding = embeddings.to_vec();

        // Normalize if requested
        if config.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                embedding.iter_mut().for_each(|x| *x /= norm);
            }
        }

        Ok(embedding)
    }

    /// Compute embedding for a single text
    async fn compute_single_embedding(&self, text: String) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel();

        self.sender.send(EmbeddingRequest {
            text,
            response: tx,
        }).map_err(|_| VyaktiError::Embedding("Worker thread died".to_string()))?;

        rx.await
            .map_err(|_| VyaktiError::Embedding("Worker thread did not respond".to_string()))?
    }
}

#[cfg(feature = "llama-cpp")]
#[async_trait]
impl EmbeddingProvider for LlamaCppProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        debug!("Computing embeddings for {} texts", texts.len());

        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.compute_single_embedding(text.clone()).await?;
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
            sender: self.sender.clone(),
            config: self.config.clone(),
        }
    }
}

// Re-export for convenience
#[cfg(feature = "llama-cpp")]
pub use num_cpus;
