//! Metadata filtering engine for search results.
//!
//! This module provides generic metadata filtering capabilities for search results.
//! Supports various operators for different data types including numbers, strings, and booleans.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::SearchResult;

/// Filter value types supported by the filtering engine
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// List of values (for 'in' and 'not_in' operators)
    List(Vec<FilterValue>),
    /// Null value
    Null,
}

impl FilterValue {
    /// Try to get as f64 for numeric comparisons
    fn as_f64(&self) -> Option<f64> {
        match self {
            FilterValue::Integer(i) => Some(*i as f64),
            FilterValue::Float(f) => Some(*f),
            FilterValue::String(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Get as string for string operations
    fn as_str(&self) -> String {
        match self {
            FilterValue::String(s) => s.clone(),
            FilterValue::Integer(i) => i.to_string(),
            FilterValue::Float(f) => f.to_string(),
            FilterValue::Bool(b) => b.to_string(),
            FilterValue::Null => "null".to_string(),
            FilterValue::List(_) => "<list>".to_string(),
        }
    }

    /// Check if value is truthy
    fn is_truthy(&self) -> bool {
        match self {
            FilterValue::Bool(b) => *b,
            FilterValue::String(s) => !s.is_empty(),
            FilterValue::Integer(i) => *i != 0,
            FilterValue::Float(f) => *f != 0.0,
            FilterValue::List(l) => !l.is_empty(),
            FilterValue::Null => false,
        }
    }
}

impl From<&JsonValue> for FilterValue {
    fn from(json: &JsonValue) -> Self {
        match json {
            JsonValue::String(s) => FilterValue::String(s.clone()),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    FilterValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    FilterValue::Float(f)
                } else {
                    FilterValue::Null
                }
            }
            JsonValue::Bool(b) => FilterValue::Bool(*b),
            JsonValue::Array(arr) => {
                FilterValue::List(arr.iter().map(FilterValue::from).collect())
            }
            JsonValue::Null => FilterValue::Null,
            JsonValue::Object(_) => FilterValue::Null, // Objects not supported as filter values
        }
    }
}

/// Filter operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOperator {
    /// Equal to
    #[serde(rename = "==")]
    Eq,
    /// Not equal to
    #[serde(rename = "!=")]
    Ne,
    /// Less than
    #[serde(rename = "<")]
    Lt,
    /// Less than or equal
    #[serde(rename = "<=")]
    Le,
    /// Greater than
    #[serde(rename = ">")]
    Gt,
    /// Greater than or equal
    #[serde(rename = ">=")]
    Ge,
    /// Value is in list
    In,
    /// Value is not in list
    NotIn,
    /// String contains substring
    Contains,
    /// String starts with prefix
    StartsWith,
    /// String ends with suffix
    EndsWith,
    /// Value is truthy
    IsTrue,
    /// Value is falsy
    IsFalse,
}

/// A single filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Operator to apply
    pub operator: FilterOperator,
    /// Value to compare against
    pub value: FilterValue,
}

/// Filter specification for a single field
pub type FieldFilter = HashMap<FilterOperator, FilterValue>;

/// Complete metadata filter specification
/// Maps field names to their filter conditions
pub type MetadataFilters = HashMap<String, FieldFilter>;

/// Metadata filtering engine
#[derive(Debug, Default)]
pub struct MetadataFilterEngine;

impl MetadataFilterEngine {
    /// Create a new metadata filter engine
    pub fn new() -> Self {
        Self
    }

    /// Apply metadata filters to search results
    ///
    /// # Arguments
    ///
    /// * `results` - Search results to filter
    /// * `filters` - Metadata filters to apply (AND logic)
    ///
    /// # Returns
    ///
    /// Filtered search results
    pub fn apply_filters(
        &self,
        results: Vec<SearchResult>,
        filters: &MetadataFilters,
    ) -> Vec<SearchResult> {
        if filters.is_empty() {
            return results;
        }

        results
            .into_iter()
            .filter(|result| self.evaluate_filters(result, filters))
            .collect()
    }

    /// Evaluate all filters against a single search result
    ///
    /// All filters must pass (AND logic) for the result to be included
    fn evaluate_filters(&self, result: &SearchResult, filters: &MetadataFilters) -> bool {
        for (field_name, field_filter) in filters {
            if !self.evaluate_field_filter(result, field_name, field_filter) {
                return false;
            }
        }
        true
    }

    /// Evaluate a single field filter against a search result
    fn evaluate_field_filter(
        &self,
        result: &SearchResult,
        field_name: &str,
        field_filter: &FieldFilter,
    ) -> bool {
        // Get field value from metadata
        let field_value = match result.metadata.get(field_name) {
            Some(value) => FilterValue::from(value),
            None => return false, // Missing field fails all filters
        };

        // Evaluate each operator in the field filter (AND logic)
        for (operator, expected_value) in field_filter {
            if !self.evaluate_operator(&field_value, *operator, expected_value) {
                return false;
            }
        }

        true
    }

    /// Evaluate a single operator against field and expected values
    fn evaluate_operator(
        &self,
        field_value: &FilterValue,
        operator: FilterOperator,
        expected_value: &FilterValue,
    ) -> bool {
        match operator {
            FilterOperator::Eq => self.op_equals(field_value, expected_value),
            FilterOperator::Ne => !self.op_equals(field_value, expected_value),
            FilterOperator::Lt => self.op_less_than(field_value, expected_value),
            FilterOperator::Le => {
                self.op_less_than(field_value, expected_value)
                    || self.op_equals(field_value, expected_value)
            }
            FilterOperator::Gt => self.op_greater_than(field_value, expected_value),
            FilterOperator::Ge => {
                self.op_greater_than(field_value, expected_value)
                    || self.op_equals(field_value, expected_value)
            }
            FilterOperator::In => self.op_in(field_value, expected_value),
            FilterOperator::NotIn => !self.op_in(field_value, expected_value),
            FilterOperator::Contains => self.op_contains(field_value, expected_value),
            FilterOperator::StartsWith => self.op_starts_with(field_value, expected_value),
            FilterOperator::EndsWith => self.op_ends_with(field_value, expected_value),
            FilterOperator::IsTrue => field_value.is_truthy(),
            FilterOperator::IsFalse => !field_value.is_truthy(),
        }
    }

    // Operator implementations

    fn op_equals(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        match (field_value, expected_value) {
            (FilterValue::String(a), FilterValue::String(b)) => a == b,
            (FilterValue::Integer(a), FilterValue::Integer(b)) => a == b,
            (FilterValue::Bool(a), FilterValue::Bool(b)) => a == b,
            (FilterValue::Null, FilterValue::Null) => true,
            // Numeric comparison with type coercion
            (a, b) => {
                if let (Some(a_num), Some(b_num)) = (a.as_f64(), b.as_f64()) {
                    (a_num - b_num).abs() < f64::EPSILON
                } else {
                    false
                }
            }
        }
    }

    fn op_less_than(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        // Try numeric comparison first
        if let (Some(a), Some(b)) = (field_value.as_f64(), expected_value.as_f64()) {
            return a < b;
        }

        // Fall back to string comparison
        field_value.as_str() < expected_value.as_str()
    }

    fn op_greater_than(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        // Try numeric comparison first
        if let (Some(a), Some(b)) = (field_value.as_f64(), expected_value.as_f64()) {
            return a > b;
        }

        // Fall back to string comparison
        field_value.as_str() > expected_value.as_str()
    }

    fn op_in(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        if let FilterValue::List(list) = expected_value {
            list.iter().any(|v| self.op_equals(field_value, v))
        } else {
            false
        }
    }

    fn op_contains(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        let field_str = field_value.as_str();
        let expected_str = expected_value.as_str();
        field_str.contains(&expected_str)
    }

    fn op_starts_with(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        let field_str = field_value.as_str();
        let expected_str = expected_value.as_str();
        field_str.starts_with(&expected_str)
    }

    fn op_ends_with(&self, field_value: &FilterValue, expected_value: &FilterValue) -> bool {
        let field_str = field_value.as_str();
        let expected_str = expected_value.as_str();
        field_str.ends_with(&expected_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_results() -> Vec<SearchResult> {
        vec![
            SearchResult {
                id: 0,
                text: "Chapter 1 content".to_string(),
                score: 0.95,
                metadata: serde_json::json!({
                    "chapter": 1,
                    "character": "Alice",
                    "word_count": 150,
                    "is_published": true,
                    "genre": "fiction"
                })
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            },
            SearchResult {
                id: 1,
                text: "Chapter 3 content".to_string(),
                score: 0.87,
                metadata: serde_json::json!({
                    "chapter": 3,
                    "character": "Bob",
                    "word_count": 250,
                    "is_published": true,
                    "genre": "fiction"
                })
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            },
            SearchResult {
                id: 2,
                text: "Chapter 5 content".to_string(),
                score: 0.82,
                metadata: serde_json::json!({
                    "chapter": 5,
                    "character": "Alice",
                    "word_count": 300,
                    "is_published": false,
                    "genre": "non-fiction"
                })
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            },
            SearchResult {
                id: 3,
                text: "Chapter 10 content".to_string(),
                score: 0.78,
                metadata: serde_json::json!({
                    "chapter": 10,
                    "character": "Charlie",
                    "word_count": 400,
                    "is_published": true,
                    "genre": "fiction"
                })
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            },
        ]
    }

    #[test]
    fn test_no_filters_returns_all() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();
        let filters = HashMap::new();

        let filtered = engine.apply_filters(results.clone(), &filters);
        assert_eq!(filtered.len(), results.len());
    }

    #[test]
    fn test_equals_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut chapter_filter = HashMap::new();
        chapter_filter.insert(FilterOperator::Eq, FilterValue::Integer(1));
        filters.insert("chapter".to_string(), chapter_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 0);
    }

    #[test]
    fn test_not_equals_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut genre_filter = HashMap::new();
        genre_filter.insert(FilterOperator::Ne, FilterValue::String("fiction".to_string()));
        filters.insert("genre".to_string(), genre_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 2); // non-fiction
    }

    #[test]
    fn test_less_than_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut chapter_filter = HashMap::new();
        chapter_filter.insert(FilterOperator::Lt, FilterValue::Integer(5));
        filters.insert("chapter".to_string(), chapter_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 2); // chapters 1 and 3
        assert!(filtered.iter().all(|r| {
            let chapter = r.metadata.get("chapter").unwrap().as_i64().unwrap();
            chapter < 5
        }));
    }

    #[test]
    fn test_greater_than_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut word_count_filter = HashMap::new();
        word_count_filter.insert(FilterOperator::Gt, FilterValue::Integer(200));
        filters.insert("word_count".to_string(), word_count_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 3); // 250, 300, 400
        assert!(filtered.iter().all(|r| {
            let wc = r.metadata.get("word_count").unwrap().as_i64().unwrap();
            wc > 200
        }));
    }

    #[test]
    fn test_in_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut character_filter = HashMap::new();
        character_filter.insert(
            FilterOperator::In,
            FilterValue::List(vec![
                FilterValue::String("Alice".to_string()),
                FilterValue::String("Bob".to_string()),
            ]),
        );
        filters.insert("character".to_string(), character_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 3); // Alice appears twice, Bob once
    }

    #[test]
    fn test_contains_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut genre_filter = HashMap::new();
        genre_filter.insert(
            FilterOperator::Contains,
            FilterValue::String("fiction".to_string()),
        );
        filters.insert("genre".to_string(), genre_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 4); // "fiction" and "non-fiction"
    }

    #[test]
    fn test_starts_with_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut genre_filter = HashMap::new();
        genre_filter.insert(
            FilterOperator::StartsWith,
            FilterValue::String("non".to_string()),
        );
        filters.insert("genre".to_string(), genre_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 2); // "non-fiction"
    }

    #[test]
    fn test_is_true_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut published_filter = HashMap::new();
        published_filter.insert(FilterOperator::IsTrue, FilterValue::Bool(true));
        filters.insert("is_published".to_string(), published_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 3); // 3 published documents
    }

    #[test]
    fn test_is_false_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut published_filter = HashMap::new();
        published_filter.insert(FilterOperator::IsFalse, FilterValue::Bool(false));
        filters.insert("is_published".to_string(), published_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 1); // 1 unpublished
        assert_eq!(filtered[0].id, 2);
    }

    #[test]
    fn test_compound_filters() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();

        // genre == "fiction"
        let mut genre_filter = HashMap::new();
        genre_filter.insert(FilterOperator::Eq, FilterValue::String("fiction".to_string()));
        filters.insert("genre".to_string(), genre_filter);

        // chapter <= 5
        let mut chapter_filter = HashMap::new();
        chapter_filter.insert(FilterOperator::Le, FilterValue::Integer(5));
        filters.insert("chapter".to_string(), chapter_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 2); // chapters 1 and 3 are fiction and <= 5
    }

    #[test]
    fn test_multiple_operators_same_field() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut word_count_filter = HashMap::new();
        word_count_filter.insert(FilterOperator::Ge, FilterValue::Integer(200));
        word_count_filter.insert(FilterOperator::Le, FilterValue::Integer(350));
        filters.insert("word_count".to_string(), word_count_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 2); // 250 and 300
    }

    #[test]
    fn test_missing_field_fails_filter() {
        let engine = MetadataFilterEngine::new();
        let results = create_test_results();

        let mut filters = HashMap::new();
        let mut nonexistent_filter = HashMap::new();
        nonexistent_filter.insert(FilterOperator::Eq, FilterValue::String("value".to_string()));
        filters.insert("nonexistent_field".to_string(), nonexistent_filter);

        let filtered = engine.apply_filters(results, &filters);
        assert_eq!(filtered.len(), 0); // No results because field doesn't exist
    }

    #[test]
    fn test_filter_value_from_json() {
        let json = serde_json::json!("test");
        let filter_val = FilterValue::from(&json);
        assert!(matches!(filter_val, FilterValue::String(_)));

        let json = serde_json::json!(42);
        let filter_val = FilterValue::from(&json);
        assert!(matches!(filter_val, FilterValue::Integer(42)));

        let json = serde_json::json!(3.14);
        let filter_val = FilterValue::from(&json);
        assert!(matches!(filter_val, FilterValue::Float(_)));

        let json = serde_json::json!(true);
        let filter_val = FilterValue::from(&json);
        assert!(matches!(filter_val, FilterValue::Bool(true)));
    }

    #[test]
    fn test_filter_value_as_f64() {
        assert_eq!(FilterValue::Integer(42).as_f64(), Some(42.0));
        assert_eq!(FilterValue::Float(3.14).as_f64(), Some(3.14));
        assert_eq!(
            FilterValue::String("123".to_string()).as_f64(),
            Some(123.0)
        );
        assert_eq!(FilterValue::Bool(true).as_f64(), None);
    }

    #[test]
    fn test_filter_value_is_truthy() {
        assert!(FilterValue::Bool(true).is_truthy());
        assert!(!FilterValue::Bool(false).is_truthy());
        assert!(FilterValue::String("hello".to_string()).is_truthy());
        assert!(!FilterValue::String("".to_string()).is_truthy());
        assert!(FilterValue::Integer(1).is_truthy());
        assert!(!FilterValue::Integer(0).is_truthy());
        assert!(!FilterValue::Null.is_truthy());
    }
}
