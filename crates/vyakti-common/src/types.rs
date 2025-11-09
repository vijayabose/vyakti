//! Common types used throughout Vyakti.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity
    Cosine,
    /// Euclidean distance (L2)
    Euclidean,
    /// Dot product
    DotProduct,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

/// Search result from a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID
    pub id: usize,
    /// Document text
    pub text: String,
    /// Similarity score
    pub score: f32,
    /// Metadata associated with the document
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Vector type alias
pub type Vector = Vec<f32>;

/// Document ID type
pub type DocumentId = usize;
