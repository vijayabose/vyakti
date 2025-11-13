use serde::{Deserialize, Serialize};

/// Configuration for keyword search using BM25 algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordConfig {
    /// Enable keyword search
    pub enabled: bool,
    /// BM25 k1 parameter (term frequency saturation)
    /// Controls how quickly term frequency impact saturates
    /// Typical range: 1.2 to 2.0
    pub k1: f32,
    /// BM25 b parameter (length normalization)
    /// Controls document length normalization (0 = no normalization, 1 = full normalization)
    /// Typical range: 0.75 to 0.9
    pub b: f32,
}

impl Default for KeywordConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            k1: 1.2,
            b: 0.75,
        }
    }
}

/// A search result from keyword index
#[derive(Debug, Clone)]
pub struct KeywordResult {
    /// Node ID in the HNSW graph
    pub node_id: usize,
    /// BM25 score for this result
    pub score: f32,
    /// Optional highlighted text snippets
    pub highlights: Vec<Highlight>,
}

/// A highlighted text snippet showing where the query matched
#[derive(Debug, Clone)]
pub struct Highlight {
    /// Field name where the match occurred
    pub field: String,
    /// Text fragment with the match
    pub fragment: String,
    /// Positions of matching terms in the fragment
    pub positions: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_config_default() {
        let config = KeywordConfig::default();
        assert!(config.enabled);
        assert_eq!(config.k1, 1.2);
        assert_eq!(config.b, 0.75);
    }

    #[test]
    fn test_keyword_config_custom() {
        let config = KeywordConfig {
            enabled: false,
            k1: 1.5,
            b: 0.8,
        };
        assert!(!config.enabled);
        assert_eq!(config.k1, 1.5);
        assert_eq!(config.b, 0.8);
    }
}
