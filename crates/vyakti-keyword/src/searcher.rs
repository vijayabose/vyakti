use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value};
use tantivy::{Index, IndexReader, ReloadPolicy, SnippetGenerator};
use tracing::{debug, info};

use crate::types::{Highlight, KeywordConfig, KeywordResult};

/// Searcher for querying keyword indexes using BM25 scoring
pub struct KeywordSearcher {
    index: Index,
    reader: IndexReader,
    schema: Schema,
    doc_map: Arc<RwLock<HashMap<u64, usize>>>,
    config: KeywordConfig,
}

impl KeywordSearcher {
    /// Load a keyword index from disk
    ///
    /// # Arguments
    ///
    /// * `path` - Directory containing the Tantivy index
    /// * `config` - Configuration for BM25 parameters
    pub fn load(path: &Path, config: KeywordConfig) -> Result<Self> {
        info!("Loading keyword index from: {}", path.display());

        // Open the Tantivy index
        let index = Index::open_in_dir(path)
            .with_context(|| format!("Failed to open Tantivy index in: {}", path.display()))?;

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create index reader")?;

        let schema = index.schema();

        // Load doc_map from disk
        let doc_map_path = path.join("doc_map.bin");
        let serialized = std::fs::read(&doc_map_path)
            .with_context(|| format!("Failed to read doc_map from: {}", doc_map_path.display()))?;
        let doc_map: HashMap<u64, usize> =
            bincode::deserialize(&serialized).context("Failed to deserialize doc_map")?;

        let doc_count = doc_map.len();
        debug!(
            "Loaded keyword index with {} documents (BM25: k1={}, b={})",
            doc_count, config.k1, config.b
        );

        Ok(Self {
            index,
            reader,
            schema,
            doc_map: Arc::new(RwLock::new(doc_map)),
            config,
        })
    }

    /// Search the keyword index without highlighting
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of KeywordResults sorted by BM25 score (descending)
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<KeywordResult>> {
        self.search_with_highlighting(query, top_k, false)
    }

    /// Search the keyword index with optional highlighting
    ///
    /// # Arguments
    ///
    /// * `query` - The search query string
    /// * `top_k` - Maximum number of results to return
    /// * `enable_highlighting` - Whether to generate highlighted snippets
    ///
    /// # Returns
    ///
    /// A vector of KeywordResults sorted by BM25 score (descending)
    pub fn search_with_highlighting(
        &self,
        query: &str,
        top_k: usize,
        enable_highlighting: bool,
    ) -> Result<Vec<KeywordResult>> {
        debug!(
            "Searching keyword index for: '{}' (top_k={}, highlighting={})",
            query, top_k, enable_highlighting
        );

        let searcher = self.reader.searcher();
        let text_field = self
            .schema
            .get_field("text")
            .context("Missing 'text' field in schema")?;

        // Parse query
        let query_parser = QueryParser::for_index(&self.index, vec![text_field]);
        let parsed_query = query_parser
            .parse_query(query)
            .with_context(|| format!("Failed to parse query: '{}'", query))?;

        // Search with BM25 scoring
        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(top_k))
            .context("Search failed")?;

        // Create snippet generator if highlighting is enabled
        let snippet_generator = if enable_highlighting {
            Some(SnippetGenerator::create(
                &searcher,
                &*parsed_query,
                text_field,
            )?)
        } else {
            None
        };

        // Convert to KeywordResults
        let doc_map = self.doc_map.read();
        let mut results = Vec::new();

        for (score, doc_address) in top_docs {
            let retrieved_doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .context("Failed to retrieve document")?;

            // Get the node ID
            let id_field = self
                .schema
                .get_field("id")
                .context("Missing 'id' field in schema")?;

            let node_id_u64 = retrieved_doc
                .get_first(id_field)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("Missing node_id in document"))?;

            // Map to HNSW node ID
            let node_id = *doc_map
                .get(&node_id_u64)
                .unwrap_or(&(node_id_u64 as usize));

            // Generate highlights if enabled
            let highlights = if let Some(ref generator) = snippet_generator {
                let snippet = generator.snippet_from_doc(&retrieved_doc);
                vec![Highlight {
                    field: "text".to_string(),
                    fragment: snippet.to_html(),
                    positions: snippet
                        .highlighted()
                        .iter()
                        .map(|h| h.start)
                        .collect(),
                }]
            } else {
                vec![]
            };

            results.push(KeywordResult {
                node_id,
                score,
                highlights,
            });
        }

        debug!(
            "Keyword search returned {} results",
            results.len()
        );

        Ok(results)
    }

    /// Search with phrase query support (exact phrase matching)
    ///
    /// # Arguments
    ///
    /// * `phrase` - The exact phrase to search for (without quotes)
    /// * `top_k` - Maximum number of results to return
    pub fn search_phrase(&self, phrase: &str, top_k: usize) -> Result<Vec<KeywordResult>> {
        // Wrap the phrase in quotes for exact matching
        let query = format!("\"{}\"", phrase);
        self.search(&query, top_k)
    }

    /// Get statistics about the index
    pub fn stats(&self) -> IndexStats {
        let searcher = self.reader.searcher();
        let num_docs = searcher.num_docs() as usize;
        let num_segments = searcher.segment_readers().len();

        IndexStats {
            num_docs,
            num_segments,
        }
    }
}

/// Statistics about the keyword index
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total number of documents in the index
    pub num_docs: usize,
    /// Number of index segments
    pub num_segments: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::KeywordIndexBuilder;
    use serde_json::Value;
    use tempfile::tempdir;

    fn create_test_index(dir: &Path) -> Result<()> {
        let config = KeywordConfig::default();
        let mut builder = KeywordIndexBuilder::new(dir, config)?;

        let docs = vec![
            (0, "Hello world", "doc1.txt"),
            (1, "Rust programming language", "doc2.txt"),
            (2, "Vector database search", "doc3.txt"),
            (3, "mxbai-embed-large model", "doc4.txt"),
        ];

        for (id, text, path) in docs {
            let mut metadata = HashMap::new();
            metadata.insert("path".to_string(), Value::String(path.to_string()));
            builder.add_document(id, text, &metadata)?;
        }

        builder.commit(dir)?;
        Ok(())
    }

    #[test]
    fn test_keyword_searcher_load() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config);
        assert!(searcher.is_ok());
    }

    #[test]
    fn test_keyword_search() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config).unwrap();

        let results = searcher.search("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn test_exact_string_match() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config).unwrap();

        let results = searcher.search("mxbai-embed-large", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 3);
    }

    #[test]
    fn test_phrase_search() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config).unwrap();

        let results = searcher.search_phrase("vector database", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 2);
    }

    #[test]
    fn test_no_results() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config).unwrap();

        let results = searcher.search("nonexistent", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_index_stats() {
        let dir = tempdir().unwrap();
        create_test_index(dir.path()).unwrap();

        let config = KeywordConfig::default();
        let searcher = KeywordSearcher::load(dir.path(), config).unwrap();

        let stats = searcher.stats();
        assert_eq!(stats.num_docs, 4);
        assert!(stats.num_segments > 0);
    }
}
