//! Text chunking implementation
//!
//! Provides sentence-aware text chunking with configurable size and overlap.

use crate::token::TokenEstimator;
use crate::{ChunkError, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Chunk size in tokens (for text) or characters (for code)
    pub chunk_size: usize,
    /// Overlap between chunks in tokens (for text) or characters (for code)
    pub chunk_overlap: usize,
    /// Separator between sentences
    pub separator: String,
    /// Paragraph separator
    pub paragraph_separator: String,
    /// Safety factor for token limits (0.0-1.0)
    pub safety_factor: f32,
    /// Whether this is code (affects token estimation)
    pub is_code: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            chunk_overlap: 128,
            separator: " ".to_string(),
            paragraph_separator: "\n\n".to_string(),
            safety_factor: 0.9,
            is_code: false,
        }
    }
}

impl ChunkConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig(
                "chunk_size must be greater than 0".to_string(),
            ));
        }

        if self.chunk_overlap >= self.chunk_size {
            return Err(ChunkError::InvalidConfig(
                "chunk_overlap must be less than chunk_size".to_string(),
            ));
        }

        if self.safety_factor <= 0.0 || self.safety_factor > 1.0 {
            return Err(ChunkError::InvalidConfig(
                "safety_factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Result of chunking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    /// The text chunk
    pub text: String,
    /// Metadata for this chunk
    pub metadata: HashMap<String, serde_json::Value>,
    /// Estimated token count
    pub token_count: usize,
}

/// Text chunker
pub struct TextChunker {
    config: ChunkConfig,
    sentence_regex: Regex,
    token_estimator: TokenEstimator,
}

impl TextChunker {
    /// Create a new text chunker with the given configuration
    pub fn new(config: ChunkConfig) -> Result<Self> {
        config.validate()?;

        // Regex for sentence boundaries
        // Matches: . ! ? followed by whitespace or end of string
        let sentence_regex = Regex::new(r"([.!?]+)\s+|\n\n+").map_err(|e| {
            ChunkError::ProcessingError(format!("Failed to compile sentence regex: {}", e))
        })?;

        Ok(Self {
            config,
            sentence_regex,
            token_estimator: TokenEstimator::new(),
        })
    }

    /// Create a default text chunker
    pub fn default() -> Result<Self> {
        Self::new(ChunkConfig::default())
    }

    /// Chunk text into smaller pieces
    pub fn chunk(&self, text: &str) -> Vec<ChunkResult> {
        self.chunk_with_metadata(text, HashMap::new())
    }

    /// Chunk text with metadata
    pub fn chunk_with_metadata(
        &self,
        text: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Vec<ChunkResult> {
        if text.trim().is_empty() {
            return vec![];
        }

        // Split into sentences
        let sentences = self.split_sentences(text);

        // Combine sentences into chunks
        self.create_chunks(&sentences, metadata)
    }

    /// Split text into sentences
    fn split_sentences(&self, text: &str) -> Vec<String> {
        // First split by paragraphs
        let paragraphs: Vec<&str> = text.split(&self.config.paragraph_separator).collect();

        let mut sentences = Vec::new();

        for paragraph in paragraphs {
            if paragraph.trim().is_empty() {
                continue;
            }

            // Split paragraph into sentences using regex
            let mut last_end = 0;
            for mat in self.sentence_regex.find_iter(paragraph) {
                let sentence = &paragraph[last_end..mat.end()];
                if !sentence.trim().is_empty() {
                    sentences.push(sentence.trim().to_string());
                }
                last_end = mat.end();
            }

            // Add remaining text
            if last_end < paragraph.len() {
                let remaining = &paragraph[last_end..];
                if !remaining.trim().is_empty() {
                    sentences.push(remaining.trim().to_string());
                }
            }
        }

        // Fallback: if no sentences found, treat whole text as one sentence
        if sentences.is_empty() {
            sentences.push(text.trim().to_string());
        }

        sentences
    }

    /// Create chunks from sentences
    fn create_chunks(
        &self,
        sentences: &[String],
        base_metadata: HashMap<String, serde_json::Value>,
    ) -> Vec<ChunkResult> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0;

        for sentence in sentences {
            let sentence_tokens = self
                .token_estimator
                .estimate(sentence, self.config.is_code);

            // If adding this sentence exceeds chunk_size, save current chunk
            if !current_chunk.is_empty() && current_tokens + sentence_tokens > self.config.chunk_size
            {
                let mut metadata = base_metadata.clone();
                metadata.insert(
                    "chunk_index".to_string(),
                    serde_json::json!(chunks.len()),
                );

                chunks.push(ChunkResult {
                    text: current_chunk.trim().to_string(),
                    metadata,
                    token_count: current_tokens,
                });

                // Start new chunk with overlap
                current_chunk = self.get_overlap_text(&current_chunk);
                current_tokens = self
                    .token_estimator
                    .estimate(&current_chunk, self.config.is_code);
            }

            // Add sentence to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(sentence);
            current_tokens += sentence_tokens;
        }

        // Add final chunk
        if !current_chunk.is_empty() {
            let mut metadata = base_metadata.clone();
            metadata.insert(
                "chunk_index".to_string(),
                serde_json::json!(chunks.len()),
            );

            chunks.push(ChunkResult {
                text: current_chunk.trim().to_string(),
                metadata,
                token_count: current_tokens,
            });
        }

        chunks
    }

    /// Get overlap text from the end of a chunk
    fn get_overlap_text(&self, text: &str) -> String {
        if self.config.chunk_overlap == 0 {
            return String::new();
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let overlap_words = words.len().saturating_sub(self.config.chunk_overlap / 5); // Rough estimate

        if overlap_words >= words.len() {
            return text.to_string();
        }

        words[overlap_words..].join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 256);
        assert_eq!(config.chunk_overlap, 128);
    }

    #[test]
    fn test_chunk_config_validation() {
        let mut config = ChunkConfig::default();
        assert!(config.validate().is_ok());

        config.chunk_size = 0;
        assert!(config.validate().is_err());

        config.chunk_size = 100;
        config.chunk_overlap = 100;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_text_chunker_simple() {
        let chunker = TextChunker::default().unwrap();
        let text = "Hello world. This is a test. Another sentence here.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("Hello"));
    }

    #[test]
    fn test_text_chunker_long_text() {
        let config = ChunkConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            ..Default::default()
        };

        let chunker = TextChunker::new(config).unwrap();

        // Create a long text
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
        let chunks = chunker.chunk(&text);

        // Should create multiple chunks
        assert!(chunks.len() > 1);

        // Each chunk should be reasonable size
        for chunk in &chunks {
            assert!(chunk.token_count > 0);
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = TextChunker::default().unwrap();
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_metadata_preserved() {
        let chunker = TextChunker::default().unwrap();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), serde_json::json!("test.txt"));

        let text = "Hello world. This is a test.";
        let chunks = chunker.chunk_with_metadata(text, metadata);

        assert!(!chunks.is_empty());
        assert_eq!(
            chunks[0].metadata.get("source"),
            Some(&serde_json::json!("test.txt"))
        );
    }
}
