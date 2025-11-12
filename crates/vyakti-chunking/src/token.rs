//! Token estimation utilities
//!
//! Provides conservative token count estimation for text chunks
//! to ensure they fit within embedding model limits.

use unicode_segmentation::UnicodeSegmentation;

/// Conservative token estimator
///
/// Uses character-based estimation since we don't have access
/// to the actual tokenizer. This matches LEANN's fallback behavior.
pub struct TokenEstimator {
    /// Characters per token ratio for natural text
    chars_per_token_text: f32,
    /// Characters per token ratio for code (worse tokenization)
    chars_per_token_code: f32,
}

impl Default for TokenEstimator {
    fn default() -> Self {
        Self {
            chars_per_token_text: 4.0,  // ~4 chars per token for natural text
            chars_per_token_code: 1.2,  // ~1.2 tokens per char for code
        }
    }
}

impl TokenEstimator {
    /// Create a new token estimator
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate token count for text
    pub fn estimate(&self, text: &str, is_code: bool) -> usize {
        let char_count = text.graphemes(true).count();

        if is_code {
            // Code has worse tokenization - more tokens per character
            (char_count as f32 * self.chars_per_token_code).ceil() as usize
        } else {
            // Natural text - fewer tokens per character
            (char_count as f32 / self.chars_per_token_text).ceil() as usize
        }
    }

    /// Calculate safe chunk size accounting for overlap and safety margin
    pub fn calculate_safe_chunk_size(
        &self,
        model_token_limit: usize,
        overlap_tokens: usize,
        is_code: bool,
        safety_factor: f32,
    ) -> usize {
        let safe_limit = (model_token_limit as f32 * safety_factor) as usize;

        if is_code {
            // For code, convert tokens to characters
            let overlap_chars = (overlap_tokens as f32 * 3.0) as usize; // ~3 chars per token for code
            let safe_chars = (safe_limit as f32 / self.chars_per_token_code) as usize;
            safe_chars.saturating_sub(overlap_chars).max(1)
        } else {
            // For text, work directly with tokens
            safe_limit.saturating_sub(overlap_tokens).max(1)
        }
    }
}

/// Estimate token count for a text string
///
/// Simple wrapper around TokenEstimator for convenience
pub fn estimate_tokens(text: &str, is_code: bool) -> usize {
    TokenEstimator::default().estimate(text, is_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_text() {
        let estimator = TokenEstimator::new();
        let text = "Hello world! This is a test.";
        let tokens = estimator.estimate(text, false);
        // Should be around 6-8 tokens
        assert!(tokens >= 5 && tokens <= 10);
    }

    #[test]
    fn test_estimate_tokens_code() {
        let estimator = TokenEstimator::new();
        let code = "fn main() { println!(\"Hello\"); }";
        let tokens = estimator.estimate(code, true);
        // Code should have more tokens
        assert!(tokens > 30);
    }

    #[test]
    fn test_calculate_safe_chunk_size() {
        let estimator = TokenEstimator::new();

        // For text
        let chunk_size = estimator.calculate_safe_chunk_size(512, 128, false, 0.9);
        assert!(chunk_size > 0);
        assert!(chunk_size < 512);

        // For code
        let chunk_size = estimator.calculate_safe_chunk_size(512, 64, true, 0.9);
        assert!(chunk_size > 0);
    }

    #[test]
    fn test_unicode_handling() {
        let estimator = TokenEstimator::new();
        let text = "Hello 世界 🌍";
        let tokens = estimator.estimate(text, false);
        // Should handle unicode graphemes correctly
        assert!(tokens > 0);
    }
}
