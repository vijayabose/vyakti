//! Search quality evaluation metrics and tools.
//!
//! This module provides comprehensive evaluation metrics for assessing vector search quality,
//! including precision, recall, NDCG, MRR, and custom relevance-based metrics.

use vyakti_common::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Relevance judgment for a query-document pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceJudgment {
    /// Query text
    pub query: String,
    /// Document ID
    pub doc_id: String,
    /// Relevance score (0 = not relevant, 1 = somewhat relevant, 2 = relevant, 3 = highly relevant)
    pub relevance: u8,
}

/// Test query with ground truth relevance judgments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestQuery {
    /// Query text
    pub query: String,
    /// Expected relevant document IDs (for binary relevance)
    pub relevant_docs: Vec<String>,
    /// Graded relevance judgments (doc_id -> relevance score)
    #[serde(default)]
    pub graded_relevance: HashMap<String, u8>,
}

/// Evaluation dataset containing multiple test queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationDataset {
    /// Dataset name
    pub name: String,
    /// Description
    pub description: String,
    /// Test queries with ground truth
    pub queries: Vec<TestQuery>,
}

impl EvaluationDataset {
    /// Create a new evaluation dataset
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            queries: Vec::new(),
        }
    }

    /// Add a test query
    pub fn add_query(&mut self, query: TestQuery) {
        self.queries.push(query);
    }

    /// Load dataset from JSON file
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let dataset: EvaluationDataset = serde_json::from_str(&content)?;
        Ok(dataset)
    }

    /// Save dataset to JSON file
    pub fn to_json_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Evaluation metrics for a single query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Query text
    pub query: String,
    /// Precision at K (for various K values)
    pub precision_at_k: HashMap<usize, f64>,
    /// Recall at K (for various K values)
    pub recall_at_k: HashMap<usize, f64>,
    /// F1 score at K
    pub f1_at_k: HashMap<usize, f64>,
    /// Mean Average Precision
    pub average_precision: f64,
    /// Reciprocal Rank
    pub reciprocal_rank: f64,
    /// Normalized Discounted Cumulative Gain at K
    pub ndcg_at_k: HashMap<usize, f64>,
    /// Number of relevant documents retrieved at K
    pub relevant_retrieved_at_k: HashMap<usize, usize>,
    /// Total relevant documents
    pub total_relevant: usize,
    /// Search time in milliseconds
    pub search_time_ms: f64,
}

/// Aggregated evaluation metrics across all queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Number of queries evaluated
    pub num_queries: usize,
    /// Mean Precision at K
    pub mean_precision_at_k: HashMap<usize, f64>,
    /// Mean Recall at K
    pub mean_recall_at_k: HashMap<usize, f64>,
    /// Mean F1 at K
    pub mean_f1_at_k: HashMap<usize, f64>,
    /// Mean Average Precision (MAP)
    pub mean_average_precision: f64,
    /// Mean Reciprocal Rank (MRR)
    pub mean_reciprocal_rank: f64,
    /// Mean NDCG at K
    pub mean_ndcg_at_k: HashMap<usize, f64>,
    /// Average search time in milliseconds
    pub avg_search_time_ms: f64,
    /// Queries with zero results
    pub queries_with_zero_results: usize,
}

/// Search quality evaluator
pub struct SearchEvaluator {
    /// K values to compute metrics for
    k_values: Vec<usize>,
}

impl SearchEvaluator {
    /// Create a new evaluator with default K values [1, 3, 5, 10, 20]
    pub fn new() -> Self {
        Self {
            k_values: vec![1, 3, 5, 10, 20],
        }
    }

    /// Create an evaluator with custom K values
    pub fn with_k_values(k_values: Vec<usize>) -> Self {
        Self { k_values }
    }

    /// Evaluate a single query's results
    pub fn evaluate_query(
        &self,
        query: &TestQuery,
        results: &[SearchResult],
        search_time_ms: f64,
    ) -> QueryMetrics {
        let mut metrics = QueryMetrics {
            query: query.query.clone(),
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            f1_at_k: HashMap::new(),
            average_precision: 0.0,
            reciprocal_rank: 0.0,
            ndcg_at_k: HashMap::new(),
            relevant_retrieved_at_k: HashMap::new(),
            total_relevant: query.relevant_docs.len(),
            search_time_ms,
        };

        // Build relevant set
        let relevant_set: HashSet<String> = query.relevant_docs.iter().cloned().collect();

        // Extract result IDs
        let result_ids: Vec<String> = results.iter().map(|r| r.id.to_string()).collect();

        // Compute metrics for each K
        for &k in &self.k_values {
            let k_results = &result_ids[..result_ids.len().min(k)];

            // Precision@K = (relevant retrieved at K) / K
            let relevant_at_k: usize = k_results
                .iter()
                .filter(|id| relevant_set.contains(*id))
                .count();

            let precision = if k > 0 {
                relevant_at_k as f64 / k as f64
            } else {
                0.0
            };

            // Recall@K = (relevant retrieved at K) / total relevant
            let recall = if metrics.total_relevant > 0 {
                relevant_at_k as f64 / metrics.total_relevant as f64
            } else {
                0.0
            };

            // F1@K = harmonic mean of precision and recall
            let f1 = if precision + recall > 0.0 {
                2.0 * (precision * recall) / (precision + recall)
            } else {
                0.0
            };

            metrics.precision_at_k.insert(k, precision);
            metrics.recall_at_k.insert(k, recall);
            metrics.f1_at_k.insert(k, f1);
            metrics.relevant_retrieved_at_k.insert(k, relevant_at_k);
        }

        // Average Precision (AP)
        metrics.average_precision = self.compute_average_precision(&result_ids, &relevant_set);

        // Reciprocal Rank (RR)
        metrics.reciprocal_rank = self.compute_reciprocal_rank(&result_ids, &relevant_set);

        // NDCG@K (using graded relevance if available, otherwise binary)
        for &k in &self.k_values {
            let ndcg = if query.graded_relevance.is_empty() {
                self.compute_ndcg_binary(&result_ids, &relevant_set, k)
            } else {
                self.compute_ndcg_graded(&result_ids, &query.graded_relevance, k)
            };
            metrics.ndcg_at_k.insert(k, ndcg);
        }

        metrics
    }

    /// Compute Average Precision
    fn compute_average_precision(
        &self,
        result_ids: &[String],
        relevant_set: &HashSet<String>,
    ) -> f64 {
        if relevant_set.is_empty() {
            return 0.0;
        }

        let mut num_relevant_seen = 0;
        let mut sum_precisions = 0.0;

        for (i, result_id) in result_ids.iter().enumerate() {
            if relevant_set.contains(result_id) {
                num_relevant_seen += 1;
                let precision_at_i = num_relevant_seen as f64 / (i + 1) as f64;
                sum_precisions += precision_at_i;
            }
        }

        sum_precisions / relevant_set.len() as f64
    }

    /// Compute Reciprocal Rank
    fn compute_reciprocal_rank(
        &self,
        result_ids: &[String],
        relevant_set: &HashSet<String>,
    ) -> f64 {
        for (i, result_id) in result_ids.iter().enumerate() {
            if relevant_set.contains(result_id) {
                return 1.0 / (i + 1) as f64;
            }
        }
        0.0
    }

    /// Compute NDCG with binary relevance
    fn compute_ndcg_binary(
        &self,
        result_ids: &[String],
        relevant_set: &HashSet<String>,
        k: usize,
    ) -> f64 {
        let k_results = &result_ids[..result_ids.len().min(k)];

        // DCG: sum of (relevance / log2(rank + 1))
        let dcg: f64 = k_results
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let relevance = if relevant_set.contains(id) { 1.0 } else { 0.0 };
                relevance / (i as f64 + 2.0).log2()
            })
            .sum();

        // IDCG: DCG of perfect ranking
        let num_relevant = relevant_set.len().min(k);
        let idcg: f64 = (0..num_relevant)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }

    /// Compute NDCG with graded relevance
    fn compute_ndcg_graded(
        &self,
        result_ids: &[String],
        graded_relevance: &HashMap<String, u8>,
        k: usize,
    ) -> f64 {
        let k_results = &result_ids[..result_ids.len().min(k)];

        // DCG with graded relevance
        let dcg: f64 = k_results
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let relevance = *graded_relevance.get(id).unwrap_or(&0) as f64;
                relevance / (i as f64 + 2.0).log2()
            })
            .sum();

        // IDCG: DCG of perfect ranking (sorted by relevance)
        let mut relevances: Vec<u8> = graded_relevance.values().copied().collect();
        relevances.sort_by(|a, b| b.cmp(a)); // Sort descending

        let idcg: f64 = relevances
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &rel)| rel as f64 / (i as f64 + 2.0).log2())
            .sum();

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }

    /// Aggregate metrics across multiple queries
    pub fn aggregate_metrics(&self, query_metrics: &[QueryMetrics]) -> AggregatedMetrics {
        if query_metrics.is_empty() {
            return AggregatedMetrics {
                num_queries: 0,
                mean_precision_at_k: HashMap::new(),
                mean_recall_at_k: HashMap::new(),
                mean_f1_at_k: HashMap::new(),
                mean_average_precision: 0.0,
                mean_reciprocal_rank: 0.0,
                mean_ndcg_at_k: HashMap::new(),
                avg_search_time_ms: 0.0,
                queries_with_zero_results: 0,
            };
        }

        let num_queries = query_metrics.len();
        let mut aggregated = AggregatedMetrics {
            num_queries,
            mean_precision_at_k: HashMap::new(),
            mean_recall_at_k: HashMap::new(),
            mean_f1_at_k: HashMap::new(),
            mean_average_precision: 0.0,
            mean_reciprocal_rank: 0.0,
            mean_ndcg_at_k: HashMap::new(),
            avg_search_time_ms: 0.0,
            queries_with_zero_results: 0,
        };

        // Aggregate for each K
        for &k in &self.k_values {
            let mean_precision: f64 = query_metrics
                .iter()
                .filter_map(|m| m.precision_at_k.get(&k))
                .sum::<f64>()
                / num_queries as f64;

            let mean_recall: f64 = query_metrics
                .iter()
                .filter_map(|m| m.recall_at_k.get(&k))
                .sum::<f64>()
                / num_queries as f64;

            let mean_f1: f64 = query_metrics
                .iter()
                .filter_map(|m| m.f1_at_k.get(&k))
                .sum::<f64>()
                / num_queries as f64;

            let mean_ndcg: f64 = query_metrics
                .iter()
                .filter_map(|m| m.ndcg_at_k.get(&k))
                .sum::<f64>()
                / num_queries as f64;

            aggregated.mean_precision_at_k.insert(k, mean_precision);
            aggregated.mean_recall_at_k.insert(k, mean_recall);
            aggregated.mean_f1_at_k.insert(k, mean_f1);
            aggregated.mean_ndcg_at_k.insert(k, mean_ndcg);
        }

        // MAP and MRR
        aggregated.mean_average_precision = query_metrics
            .iter()
            .map(|m| m.average_precision)
            .sum::<f64>()
            / num_queries as f64;

        aggregated.mean_reciprocal_rank = query_metrics
            .iter()
            .map(|m| m.reciprocal_rank)
            .sum::<f64>()
            / num_queries as f64;

        // Average search time
        aggregated.avg_search_time_ms = query_metrics
            .iter()
            .map(|m| m.search_time_ms)
            .sum::<f64>()
            / num_queries as f64;

        // Count queries with zero results
        aggregated.queries_with_zero_results = query_metrics
            .iter()
            .filter(|m| m.relevant_retrieved_at_k.get(&10).unwrap_or(&0) == &0)
            .count();

        aggregated
    }

    /// Print metrics in a human-readable format
    pub fn print_metrics(&self, metrics: &AggregatedMetrics) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         VYAKTI SEARCH EVALUATION RESULTS                 ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        println!("📊 Dataset Statistics:");
        println!("  • Number of queries: {}", metrics.num_queries);
        println!("  • Queries with zero results: {}", metrics.queries_with_zero_results);
        println!("  • Average search time: {:.2}ms\n", metrics.avg_search_time_ms);

        println!("📈 Overall Metrics:");
        println!("  • Mean Average Precision (MAP): {:.4}", metrics.mean_average_precision);
        println!("  • Mean Reciprocal Rank (MRR): {:.4}\n", metrics.mean_reciprocal_rank);

        println!("📋 Metrics by K:");
        println!("  ┌──────┬───────────┬────────┬────────┬────────┐");
        println!("  │  K   │ Precision │ Recall │   F1   │  NDCG  │");
        println!("  ├──────┼───────────┼────────┼────────┼────────┤");

        for &k in &self.k_values {
            let p = metrics.mean_precision_at_k.get(&k).unwrap_or(&0.0);
            let r = metrics.mean_recall_at_k.get(&k).unwrap_or(&0.0);
            let f1 = metrics.mean_f1_at_k.get(&k).unwrap_or(&0.0);
            let ndcg = metrics.mean_ndcg_at_k.get(&k).unwrap_or(&0.0);

            println!("  │ {:>4} │  {:.4}   │ {:.4} │ {:.4} │ {:.4} │",
                     k, p, r, f1, ndcg);
        }

        println!("  └──────┴───────────┴────────┴────────┴────────┘\n");
    }
}

impl Default for SearchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_query() -> TestQuery {
        TestQuery {
            query: "test query".to_string(),
            relevant_docs: vec!["1".to_string(), "3".to_string(), "5".to_string()],
            graded_relevance: HashMap::new(),
        }
    }

    fn create_search_results() -> Vec<SearchResult> {
        vec![
            SearchResult {
                id: 1,
                text: "doc 1".to_string(),
                score: 0.1,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: 2,
                text: "doc 2".to_string(),
                score: 0.2,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: 3,
                text: "doc 3".to_string(),
                score: 0.3,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: 4,
                text: "doc 4".to_string(),
                score: 0.4,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: 5,
                text: "doc 5".to_string(),
                score: 0.5,
                metadata: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_precision_at_k() {
        let evaluator = SearchEvaluator::with_k_values(vec![5]);
        let query = create_test_query();
        let results = create_search_results();

        let metrics = evaluator.evaluate_query(&query, &results, 10.0);

        // 3 relevant out of 5 results = 0.6 precision
        assert!((metrics.precision_at_k[&5] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_recall_at_k() {
        let evaluator = SearchEvaluator::with_k_values(vec![5]);
        let query = create_test_query();
        let results = create_search_results();

        let metrics = evaluator.evaluate_query(&query, &results, 10.0);

        // 3 relevant retrieved out of 3 total relevant = 1.0 recall
        assert!((metrics.recall_at_k[&5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reciprocal_rank() {
        let evaluator = SearchEvaluator::new();
        let query = create_test_query();
        let results = create_search_results();

        let metrics = evaluator.evaluate_query(&query, &results, 10.0);

        // First relevant at position 1 (0-indexed), so RR = 1/1 = 1.0
        assert!((metrics.reciprocal_rank - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluation_dataset_serialization() {
        let mut dataset = EvaluationDataset::new(
            "test_dataset".to_string(),
            "Test dataset".to_string(),
        );
        dataset.add_query(create_test_query());

        let json = serde_json::to_string(&dataset).unwrap();
        let deserialized: EvaluationDataset = serde_json::from_str(&json).unwrap();

        assert_eq!(dataset.name, deserialized.name);
        assert_eq!(dataset.queries.len(), deserialized.queries.len());
    }
}
