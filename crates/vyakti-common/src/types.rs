//! Common types used throughout Vyakti.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

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

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::DotProduct => write!(f, "dot_product"),
        }
    }
}

impl DistanceMetric {
    /// Compute distance between two vectors using this metric.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// The distance score. For all metrics, **lower values indicate more similar vectors**.
    /// - Cosine: Converts similarity [-1, 1] to distance [0, 2] via `1.0 - similarity`
    /// - Euclidean: Standard L2 distance [0, ∞)
    /// - DotProduct: Negated dot product to convert similarity to distance
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same dimension");

        match self {
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a == 0.0 || norm_b == 0.0 {
                    // Return max distance for zero vectors
                    2.0
                } else {
                    let similarity = dot / (norm_a * norm_b);
                    // Convert similarity [-1, 1] to distance [0, 2]
                    // where 0 = identical, 2 = opposite
                    1.0 - similarity
                }
            }
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::DotProduct => {
                // Negate dot product to convert similarity to distance
                // Higher dot product = more similar, so negative gives lower distance
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                -dot_product
            }
        }
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

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchResult(id={}, score={:.4}, text={}...)",
            self.id,
            self.score,
            self.text.chars().take(50).collect::<String>()
        )
    }
}

/// Vector type alias
pub type Vector = Vec<f32>;

/// Document ID type
pub type DocumentId = usize;

/// Node ID type (for graph nodes in HNSW)
pub type NodeId = usize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(DistanceMetric::Cosine.to_string(), "cosine");
        assert_eq!(DistanceMetric::Euclidean.to_string(), "euclidean");
        assert_eq!(DistanceMetric::DotProduct.to_string(), "dot_product");
    }

    #[test]
    fn test_distance_metric_equality() {
        assert_eq!(DistanceMetric::Cosine, DistanceMetric::Cosine);
        assert_ne!(DistanceMetric::Cosine, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors: similarity = 1.0, distance = 0.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let distance = DistanceMetric::Cosine.compute(&a, &b);
        assert!((distance - 0.0).abs() < 1e-6);

        // Orthogonal vectors: similarity = 0.0, distance = 1.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let distance = DistanceMetric::Cosine.compute(&a, &b);
        assert!((distance - 1.0).abs() < 1e-6);

        // Opposite vectors: similarity = -1.0, distance = 2.0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let distance = DistanceMetric::Cosine.compute(&a, &b);
        assert!((distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        // Zero vectors should return maximum distance
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let distance = DistanceMetric::Cosine.compute(&a, &b);
        assert_eq!(distance, 2.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let distance = DistanceMetric::Euclidean.compute(&a, &b);
        assert!((distance - 5.0).abs() < 1e-6);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let distance = DistanceMetric::Euclidean.compute(&a, &b);
        assert!((distance - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        // DotProduct is negated to convert similarity to distance
        // Higher dot product = more similar = lower (more negative) distance
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = DistanceMetric::DotProduct.compute(&a, &b);
        assert!((result - (-32.0)).abs() < 1e-6); // -(1*4 + 2*5 + 3*6) = -32

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = DistanceMetric::DotProduct.compute(&a, &b);
        assert!((result - 0.0).abs() < 1e-6); // Orthogonal vectors have dot product 0
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same dimension")]
    fn test_distance_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        DistanceMetric::Cosine.compute(&a, &b);
    }

    #[test]
    fn test_search_result_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), serde_json::json!("value"));

        let result = SearchResult {
            id: 1,
            text: "test document".to_string(),
            score: 0.95,
            metadata,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SearchResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, 1);
        assert_eq!(deserialized.text, "test document");
        assert!((deserialized.score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_search_result_display() {
        let result = SearchResult {
            id: 42,
            text: "This is a long test document that should be truncated in the display"
                .to_string(),
            score: 0.85,
            metadata: HashMap::new(),
        };

        let display = result.to_string();
        assert!(display.contains("id=42"));
        assert!(display.contains("0.8500"));
        assert!(display.len() < 150); // Truncated
    }

    #[test]
    fn test_distance_metric_serialization() {
        let metric = DistanceMetric::Cosine;
        let json = serde_json::to_string(&metric).unwrap();
        let deserialized: DistanceMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(metric, deserialized);

        let metric = DistanceMetric::Euclidean;
        let json = serde_json::to_string(&metric).unwrap();
        let deserialized: DistanceMetric = serde_json::from_str(&json).unwrap();
        assert_eq!(metric, deserialized);
    }

    #[test]
    fn test_vector_operations() {
        let v1: Vector = vec![1.0, 2.0, 3.0];
        let v2: Vector = vec![4.0, 5.0, 6.0];

        // Test that Vector type works as expected
        assert_eq!(v1.len(), 3);
        assert_eq!(v2[0], 4.0);

        // Test clone
        let v3 = v1.clone();
        assert_eq!(v1, v3);
    }
}
