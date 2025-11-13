//! Integration test for metadata filtering functionality

use std::collections::HashMap;
use std::sync::Arc;
use vyakti_backend_hnsw::HnswBackend;
use vyakti_common::{BackendConfig, FilterOperator, FilterValue, MetadataFilters, Result};
use vyakti_core::{VyaktiBuilder, VyaktiSearcher};

// Mock embedding provider for testing
struct MockEmbeddingProvider {
    dimension: usize,
}

#[async_trait::async_trait]
impl vyakti_common::EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Generate deterministic embeddings based on text content
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut vec = vec![0.0; self.dimension];
                // Use text hash to generate vector
                let hash = text.len() as f32 + i as f32;
                for j in 0..self.dimension {
                    vec[j] = ((hash + j as f32) % 10.0) / 10.0;
                }
                vec
            })
            .collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "mock-embedding"
    }
}

#[tokio::test]
async fn test_metadata_filter_equals() {
    let backend_config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(backend_config.clone()));
    let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

    let mut builder = VyaktiBuilder::new(backend, embedding_provider.clone());

    // Add documents with metadata
    let mut metadata1 = HashMap::new();
    metadata1.insert(
        "category".to_string(),
        serde_json::json!("technology"),
    );
    metadata1.insert("year".to_string(), serde_json::json!(2024));

    let mut metadata2 = HashMap::new();
    metadata2.insert(
        "category".to_string(),
        serde_json::json!("science"),
    );
    metadata2.insert("year".to_string(), serde_json::json!(2023));

    let mut metadata3 = HashMap::new();
    metadata3.insert(
        "category".to_string(),
        serde_json::json!("technology"),
    );
    metadata3.insert("year".to_string(), serde_json::json!(2023));

    builder.add_text("AI and machine learning advances", Some(metadata1));
    builder.add_text("Physics research breakthroughs", Some(metadata2));
    builder.add_text("Software engineering best practices", Some(metadata3));

    let index_path = ".vyakti-test/metadata-filter-test";
    builder.build_index(index_path).await.unwrap();

    // Load and search with filters
    let backend = Box::new(HnswBackend::with_config(backend_config));
    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .unwrap();

    // Filter by category == "technology"
    let mut filters = MetadataFilters::new();
    let mut category_filter = HashMap::new();
    category_filter.insert(
        FilterOperator::Eq,
        FilterValue::String("technology".to_string()),
    );
    filters.insert("category".to_string(), category_filter);

    let results = searcher
        .search_with_filters("machine learning", 10, Some(&filters))
        .await
        .unwrap();

    assert_eq!(results.len(), 2); // Only documents with category="technology"

    // Cleanup
    std::fs::remove_dir_all(index_path).ok();
}

#[tokio::test]
async fn test_metadata_filter_greater_than() {
    let backend_config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(backend_config.clone()));
    let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

    let mut builder = VyaktiBuilder::new(backend, embedding_provider.clone());

    // Add documents with different years
    for year in 2020..=2024 {
        let mut metadata = HashMap::new();
        metadata.insert("year".to_string(), serde_json::json!(year));
        metadata.insert(
            "content".to_string(),
            serde_json::json!(format!("Document from {}", year)),
        );
        builder
            .add_text(&format!("Article from year {}", year), Some(metadata));
    }

    let index_path = ".vyakti-test/metadata-filter-gt-test";
    builder.build_index(index_path).await.unwrap();

    // Load and search with filters
    let backend = Box::new(HnswBackend::with_config(backend_config));
    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .unwrap();

    // Filter by year > 2022
    let mut filters = MetadataFilters::new();
    let mut year_filter = HashMap::new();
    year_filter.insert(FilterOperator::Gt, FilterValue::Integer(2022));
    filters.insert("year".to_string(), year_filter);

    let results = searcher
        .search_with_filters("article", 10, Some(&filters))
        .await
        .unwrap();

    assert_eq!(results.len(), 2); // Only 2023 and 2024

    // Cleanup
    std::fs::remove_dir_all(index_path).ok();
}

#[tokio::test]
async fn test_metadata_filter_compound() {
    let backend_config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(backend_config.clone()));
    let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

    let mut builder = VyaktiBuilder::new(backend, embedding_provider.clone());

    // Add documents with multiple metadata fields
    let mut metadata1 = HashMap::new();
    metadata1.insert(
        "category".to_string(),
        serde_json::json!("technology"),
    );
    metadata1.insert("year".to_string(), serde_json::json!(2024));
    metadata1.insert("published".to_string(), serde_json::json!(true));

    let mut metadata2 = HashMap::new();
    metadata2.insert(
        "category".to_string(),
        serde_json::json!("technology"),
    );
    metadata2.insert("year".to_string(), serde_json::json!(2023));
    metadata2.insert("published".to_string(), serde_json::json!(true));

    let mut metadata3 = HashMap::new();
    metadata3.insert(
        "category".to_string(),
        serde_json::json!("technology"),
    );
    metadata3.insert("year".to_string(), serde_json::json!(2024));
    metadata3.insert("published".to_string(), serde_json::json!(false));

    builder.add_text("Latest AI research", Some(metadata1));
    builder.add_text("Previous ML study", Some(metadata2));
    builder.add_text("Draft AI paper", Some(metadata3));

    let index_path = ".vyakti-test/metadata-filter-compound-test";
    builder.build_index(index_path).await.unwrap();

    // Load and search with compound filters
    let backend = Box::new(HnswBackend::with_config(backend_config));
    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .unwrap();

    // Filter by category="technology" AND year=2024 AND published=true
    let mut filters = MetadataFilters::new();

    let mut category_filter = HashMap::new();
    category_filter.insert(
        FilterOperator::Eq,
        FilterValue::String("technology".to_string()),
    );
    filters.insert("category".to_string(), category_filter);

    let mut year_filter = HashMap::new();
    year_filter.insert(FilterOperator::Eq, FilterValue::Integer(2024));
    filters.insert("year".to_string(), year_filter);

    let mut published_filter = HashMap::new();
    published_filter.insert(FilterOperator::IsTrue, FilterValue::Bool(true));
    filters.insert("published".to_string(), published_filter);

    let results = searcher
        .search_with_filters("AI", 10, Some(&filters))
        .await
        .unwrap();

    assert_eq!(results.len(), 1); // Only first document matches all conditions

    // Cleanup
    std::fs::remove_dir_all(index_path).ok();
}

#[tokio::test]
async fn test_metadata_filter_in_operator() {
    let backend_config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(backend_config.clone()));
    let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

    let mut builder = VyaktiBuilder::new(backend, embedding_provider.clone());

    // Add documents with different authors
    for author in &["Alice", "Bob", "Charlie", "David"] {
        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), serde_json::json!(author));
        builder
            .add_text(&format!("Paper by {}", author), Some(metadata));
    }

    let index_path = ".vyakti-test/metadata-filter-in-test";
    builder.build_index(index_path).await.unwrap();

    // Load and search with IN filter
    let backend = Box::new(HnswBackend::with_config(backend_config));
    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .unwrap();

    // Filter by author in ["Alice", "Bob"]
    let mut filters = MetadataFilters::new();
    let mut author_filter = HashMap::new();
    author_filter.insert(
        FilterOperator::In,
        FilterValue::List(vec![
            FilterValue::String("Alice".to_string()),
            FilterValue::String("Bob".to_string()),
        ]),
    );
    filters.insert("author".to_string(), author_filter);

    let results = searcher
        .search_with_filters("paper", 10, Some(&filters))
        .await
        .unwrap();

    assert_eq!(results.len(), 2); // Only Alice and Bob

    // Cleanup
    std::fs::remove_dir_all(index_path).ok();
}

#[tokio::test]
async fn test_metadata_filter_contains() {
    let backend_config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(backend_config.clone()));
    let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 128 });

    let mut builder = VyaktiBuilder::new(backend, embedding_provider.clone());

    // Add documents with different titles
    let titles = vec![
        "Introduction to Machine Learning",
        "Deep Learning Fundamentals",
        "Natural Language Processing",
        "Computer Vision Basics",
    ];

    for title in titles {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), serde_json::json!(title));
        builder.add_text(title, Some(metadata));
    }

    let index_path = ".vyakti-test/metadata-filter-contains-test";
    builder.build_index(index_path).await.unwrap();

    // Load and search with CONTAINS filter
    let backend = Box::new(HnswBackend::with_config(backend_config));
    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .unwrap();

    // Filter by title contains "Learning"
    let mut filters = MetadataFilters::new();
    let mut title_filter = HashMap::new();
    title_filter.insert(
        FilterOperator::Contains,
        FilterValue::String("Learning".to_string()),
    );
    filters.insert("title".to_string(), title_filter);

    let results = searcher
        .search_with_filters("learning", 10, Some(&filters))
        .await
        .unwrap();

    assert_eq!(results.len(), 2); // "Machine Learning" and "Deep Learning"

    // Cleanup
    std::fs::remove_dir_all(index_path).ok();
}
