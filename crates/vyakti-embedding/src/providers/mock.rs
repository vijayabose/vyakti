//! Mock embedding provider for testing

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use vyakti_common::{EmbeddingProvider, Result, Vector, VyaktiError};

/// OpenAI embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key for OpenAI
    pub api_key: String,
    /// Model name (e.g., "text-embedding-ada-002")
    pub model: String,
    /// API base URL
    pub api_base: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "text-embedding-ada-002".to_string(),
            api_base: "https://api.openai.com/v1".to_string(),
            timeout_secs: 30,
        }
    }
}

/// OpenAI embedding provider.
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl std::fmt::Debug for OpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIProvider")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    ///
    /// # Arguments
    ///
    /// * `config` - OpenAI configuration
    /// * `dimension` - Expected embedding dimension
    pub fn new(config: OpenAIConfig, dimension: usize) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(VyaktiError::Config(
                "OpenAI API key cannot be empty".to_string(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| VyaktiError::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            dimension,
        })
    }

    /// Batch embed texts with the OpenAI API.
    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        #[derive(Serialize)]
        struct Request {
            input: Vec<String>,
            model: String,
        }

        #[derive(Deserialize)]
        struct Response {
            data: Vec<EmbeddingData>,
        }

        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let request = Request {
            input: texts.to_vec(),
            model: self.config.model.clone(),
        };

        let url = format!("{}/embeddings", self.config.api_base);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| VyaktiError::Embedding(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(VyaktiError::Embedding(format!(
                "API returned error {}: {}",
                status, error_text
            )));
        }

        let response_data: Response = response
            .json()
            .await
            .map_err(|e| VyaktiError::Embedding(format!("Failed to parse response: {}", e)))?;

        Ok(response_data
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect())
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // OpenAI API supports batching, but we'll process in chunks if needed
        const MAX_BATCH_SIZE: usize = 100;

        if texts.len() <= MAX_BATCH_SIZE {
            self.batch_embed(texts).await
        } else {
            // Process in chunks
            let mut all_embeddings = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(MAX_BATCH_SIZE) {
                let embeddings = self.batch_embed(chunk).await?;
                all_embeddings.extend(embeddings);
            }
            Ok(all_embeddings)
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "openai"
    }
}

/// Ollama embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server URL
    pub base_url: String,
    /// Model name
    pub model: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "mxbai-embed-large".to_string(),
            timeout_secs: 30,
        }
    }
}

/// Ollama embedding provider.
pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
    dimension: usize,
}

impl std::fmt::Debug for OllamaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaProvider")
            .field("config", &self.config)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl OllamaProvider {
    /// Check if Ollama server is running.
    async fn check_ollama_availability(base_url: &str, client: &reqwest::Client) -> Result<()> {
        tracing::info!("Checking Ollama availability at {}", base_url);

        let url = format!("{}/api/tags", base_url);
        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                tracing::info!("✓ Ollama is running at {}", base_url);
                Ok(())
            }
            Ok(response) => {
                let status = response.status();
                Err(VyaktiError::Embedding(format!(
                    "Ollama server returned error status: {}. Is Ollama running? Try: ollama serve",
                    status
                )))
            }
            Err(e) => {
                tracing::error!("✗ Failed to connect to Ollama at {}: {}", base_url, e);
                Err(VyaktiError::Embedding(format!(
                    "Cannot connect to Ollama at {}. Please ensure Ollama is running:\n  \
                    1. Install: brew install ollama (macOS) or visit https://ollama.com\n  \
                    2. Start server: ollama serve\n  \
                    3. Verify: curl {}/api/tags\n\nError: {}",
                    base_url, base_url, e
                )))
            }
        }
    }

    /// Check if a model exists in Ollama.
    async fn check_model_exists(
        base_url: &str,
        model_name: &str,
        client: &reqwest::Client,
    ) -> Result<bool> {
        tracing::info!("Checking if model '{}' exists", model_name);

        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        let url = format!("{}/api/tags", base_url);
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| VyaktiError::Embedding(format!("Failed to query models: {}", e)))?;

        if !response.status().is_success() {
            return Err(VyaktiError::Embedding(format!(
                "Failed to list models: HTTP {}",
                response.status()
            )));
        }

        let tags: TagsResponse = response
            .json()
            .await
            .map_err(|e| VyaktiError::Embedding(format!("Failed to parse models list: {}", e)))?;

        let exists = tags.models.iter().any(|m| m.name.starts_with(model_name));

        if exists {
            tracing::info!("✓ Model '{}' found", model_name);
        } else {
            tracing::warn!("✗ Model '{}' not found", model_name);
        }

        Ok(exists)
    }

    /// Pull a model from Ollama.
    async fn pull_model(base_url: &str, model_name: &str, client: &reqwest::Client) -> Result<()> {
        tracing::info!("Pulling model '{}' from Ollama...", model_name);
        println!("📥 Downloading embedding model '{}' (this may take a few minutes)...", model_name);

        #[derive(Serialize)]
        struct PullRequest {
            name: String,
            stream: bool,
        }

        let url = format!("{}/api/pull", base_url);
        let request = PullRequest {
            name: model_name.to_string(),
            stream: false,
        };

        let response = client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(600)) // 10 minutes for model download
            .send()
            .await
            .map_err(|e| {
                VyaktiError::Embedding(format!("Failed to pull model '{}': {}", model_name, e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(VyaktiError::Embedding(format!(
                "Failed to pull model '{}': HTTP {} - {}",
                model_name, status, error_text
            )));
        }

        tracing::info!("✓ Successfully pulled model '{}'", model_name);
        println!("✓ Model '{}' downloaded successfully", model_name);
        Ok(())
    }

    /// Create a new Ollama provider with automatic setup.
    ///
    /// This will:
    /// 1. Check if Ollama is running
    /// 2. Check if the model exists
    /// 3. Automatically pull the model if missing
    ///
    /// # Arguments
    ///
    /// * `config` - Ollama configuration
    /// * `dimension` - Expected embedding dimension
    pub async fn new(config: OllamaConfig, dimension: usize) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| VyaktiError::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        // Step 1: Check if Ollama is running
        Self::check_ollama_availability(&config.base_url, &client).await?;

        // Step 2: Check if model exists
        let model_exists = Self::check_model_exists(&config.base_url, &config.model, &client).await?;

        // Step 3: Pull model if missing
        if !model_exists {
            tracing::info!(
                "Model '{}' not found, pulling automatically...",
                config.model
            );
            Self::pull_model(&config.base_url, &config.model, &client).await?;
        }

        tracing::info!(
            "✓ OllamaProvider initialized with model '{}' ({}d)",
            config.model,
            dimension
        );

        Ok(Self {
            config,
            client,
            dimension,
        })
    }

    /// Create a new Ollama provider synchronously (for backward compatibility).
    ///
    /// Note: This does NOT check if Ollama is running or pull models automatically.
    /// Use `new()` instead for full functionality.
    pub fn new_unchecked(config: OllamaConfig, dimension: usize) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| VyaktiError::Embedding(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            config,
            client,
            dimension,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Serialize)]
        struct Request {
            model: String,
            prompt: String,
        }

        #[derive(Deserialize)]
        struct Response {
            embedding: Vec<f32>,
        }

        let url = format!("{}/api/embeddings", self.config.base_url);

        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let request = Request {
                model: self.config.model.clone(),
                prompt: text.clone(),
            };

            let response = self
                .client
                .post(&url)
                .json(&request)
                .send()
                .await
                .map_err(|e| VyaktiError::Embedding(format!("API request failed: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(VyaktiError::Embedding(format!(
                    "API returned error {}: {}",
                    status, error_text
                )));
            }

            let response_data: Response = response
                .json()
                .await
                .map_err(|e| VyaktiError::Embedding(format!("Failed to parse response: {}", e)))?;

            embeddings.push(response_data.embedding);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_openai_config_default() {
        let config = OpenAIConfig::default();
        assert_eq!(config.model, "text-embedding-ada-002");
        assert_eq!(config.api_base, "https://api.openai.com/v1");
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_openai_provider_no_api_key() {
        let config = OpenAIConfig::default();
        let result = OpenAIProvider::new(config, 1536);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("API key cannot be empty"));
    }

    #[test]
    fn test_openai_provider_creation() {
        let config = OpenAIConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        let provider = OpenAIProvider::new(config, 1536).unwrap();
        assert_eq!(provider.dimension(), 1536);
        assert_eq!(provider.name(), "openai");
    }

    #[tokio::test]
    async fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "mxbai-embed-large");
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_ollama_provider_creation() {
        let config = OllamaConfig::default();
        let provider = OllamaProvider::new_unchecked(config, 768).unwrap();
        assert_eq!(provider.dimension(), 768);
        assert_eq!(provider.name(), "ollama");
    }

    #[tokio::test]
    async fn test_openai_provider_empty_texts() {
        let config = OpenAIConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        let provider = OpenAIProvider::new(config, 1536).unwrap();
        let texts: Vec<String> = vec![];
        let embeddings = provider.embed(&texts).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn test_ollama_provider_empty_texts() {
        let config = OllamaConfig::default();
        let provider = OllamaProvider::new_unchecked(config, 768).unwrap();
        let texts: Vec<String> = vec![];
        let embeddings = provider.embed(&texts).await.unwrap();
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_openai_provider_debug() {
        let config = OpenAIConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };
        let provider = OpenAIProvider::new(config, 1536).unwrap();
        let debug_str = format!("{:?}", provider);
        assert!(debug_str.contains("OpenAIProvider"));
        assert!(debug_str.contains("dimension"));
    }

    #[test]
    fn test_ollama_provider_debug() {
        let config = OllamaConfig::default();
        let provider = OllamaProvider::new_unchecked(config, 768).unwrap();
        let debug_str = format!("{:?}", provider);
        assert!(debug_str.contains("OllamaProvider"));
        assert!(debug_str.contains("dimension"));
    }

    #[test]
    fn test_openai_config_serialization() {
        let config = OpenAIConfig {
            api_key: "test-key".to_string(),
            model: "custom-model".to_string(),
            api_base: "https://custom.api".to_string(),
            timeout_secs: 60,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OpenAIConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.api_key, "test-key");
        assert_eq!(deserialized.model, "custom-model");
        assert_eq!(deserialized.api_base, "https://custom.api");
        assert_eq!(deserialized.timeout_secs, 60);
    }

    #[test]
    fn test_ollama_config_serialization() {
        let config = OllamaConfig {
            base_url: "http://custom:8080".to_string(),
            model: "custom-model".to_string(),
            timeout_secs: 45,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OllamaConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.base_url, "http://custom:8080");
        assert_eq!(deserialized.model, "custom-model");
        assert_eq!(deserialized.timeout_secs, 45);
    }
}
