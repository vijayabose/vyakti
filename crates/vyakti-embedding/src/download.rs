//! Model download utilities for HuggingFace Hub

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{info, debug};

/// Default model configuration
pub const DEFAULT_MODEL_REPO: &str = "mixedbread-ai/mxbai-embed-large-v1";
pub const DEFAULT_MODEL_FILE: &str = "gguf/mxbai-embed-large-v1.q4_k_m.gguf";
pub const DEFAULT_MODEL_NAME: &str = "mxbai-embed-large-v1.q4_k_m.gguf";

/// Get the default models directory
pub fn get_models_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .context("Failed to determine home directory")?;
    let models_dir = home.join(".vyakti").join("models");
    Ok(models_dir)
}

/// Get the full path for a model file
pub fn get_model_path(model_name: &str) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;
    Ok(models_dir.join(model_name))
}

/// Check if a model exists locally
pub async fn model_exists(model_name: &str) -> Result<bool> {
    let model_path = get_model_path(model_name)?;
    Ok(model_path.exists())
}

/// Download a model from HuggingFace Hub
pub async fn download_model(
    repo: &str,
    filename: &str,
    output_name: &str,
) -> Result<PathBuf> {
    info!("Downloading model {} from {}", filename, repo);

    // Ensure models directory exists
    let models_dir = get_models_dir()?;
    fs::create_dir_all(&models_dir).await
        .context("Failed to create models directory")?;

    let output_path = models_dir.join(output_name);

    // Check if already exists
    if output_path.exists() {
        info!("Model already exists at {}", output_path.display());
        return Ok(output_path);
    }

    // Use hf-hub to download
    let api = hf_hub::api::tokio::Api::new()?;
    let repo = api.model(repo.to_string());

    info!("Fetching model file from HuggingFace Hub...");
    let downloaded_path = repo.get(filename).await
        .context("Failed to download model from HuggingFace Hub")?;

    // Copy to our models directory
    debug!("Copying model to {}", output_path.display());
    fs::copy(&downloaded_path, &output_path).await
        .context("Failed to copy model file")?;

    info!("Model downloaded successfully to {}", output_path.display());
    Ok(output_path)
}

/// Download the default mxbai-embed-large model
pub async fn download_default_model() -> Result<PathBuf> {
    download_model(DEFAULT_MODEL_REPO, DEFAULT_MODEL_FILE, DEFAULT_MODEL_NAME).await
}

/// Ensure a model is available, downloading if necessary
pub async fn ensure_model(model_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = model_path {
        if path.exists() {
            info!("Using model at {}", path.display());
            return Ok(path);
        } else {
            anyhow::bail!("Specified model path does not exist: {}", path.display());
        }
    }

    // Check for default model
    let default_path = get_model_path(DEFAULT_MODEL_NAME)?;
    if default_path.exists() {
        info!("Using default model at {}", default_path.display());
        return Ok(default_path);
    }

    // Download default model
    info!("Default model not found, downloading...");
    download_default_model().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_dir() {
        let dir = get_models_dir().unwrap();
        assert!(dir.to_string_lossy().contains(".vyakti"));
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_model_path() {
        let path = get_model_path("test.gguf").unwrap();
        assert!(path.to_string_lossy().ends_with("test.gguf"));
    }
}
