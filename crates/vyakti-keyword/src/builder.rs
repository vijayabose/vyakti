use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter};
use tracing::{debug, info};

use crate::types::KeywordConfig;

/// Builder for creating keyword search indexes using Tantivy
pub struct KeywordIndexBuilder {
    index: Index,
    writer: IndexWriter,
    schema: Schema,
    config: KeywordConfig,
    doc_map: HashMap<u64, usize>, // tantivy DocId → HNSW NodeId
}

impl KeywordIndexBuilder {
    /// Create a new keyword index builder
    ///
    /// # Arguments
    ///
    /// * `path` - Directory where the Tantivy index will be stored
    /// * `config` - Configuration for BM25 parameters
    pub fn new(path: &Path, config: KeywordConfig) -> Result<Self> {
        info!("Creating keyword index at: {}", path.display());

        // Build schema
        let mut schema_builder = Schema::builder();

        // Node ID field (stored for retrieval)
        schema_builder.add_u64_field("id", STORED);

        // Text field (indexed with positions for highlighting + stored)
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        schema_builder.add_text_field("text", text_options);

        // Path field (stored + indexed for filtering)
        let path_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_index_option(IndexRecordOption::Basic),
            )
            .set_stored();
        schema_builder.add_text_field("path", path_options);

        let schema = schema_builder.build();

        // Create index directory and index
        std::fs::create_dir_all(path)
            .with_context(|| format!("Failed to create index directory: {}", path.display()))?;

        let index = Index::create_in_dir(path, schema.clone())
            .with_context(|| format!("Failed to create Tantivy index in: {}", path.display()))?;

        // Create index writer with 50MB buffer
        let writer = index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        debug!("Keyword index builder initialized with BM25 parameters: k1={}, b={}", config.k1, config.b);

        Ok(Self {
            index,
            writer,
            schema,
            config,
            doc_map: HashMap::new(),
        })
    }

    /// Add a document to the keyword index
    ///
    /// # Arguments
    ///
    /// * `node_id` - The HNSW node ID for this document
    /// * `text` - The document text to index
    /// * `metadata` - Optional metadata for the document
    pub fn add_document(
        &mut self,
        node_id: usize,
        text: &str,
        metadata: &HashMap<String, Value>,
    ) -> Result<()> {
        let id_field = self
            .schema
            .get_field("id")
            .context("Missing 'id' field in schema")?;
        let text_field = self
            .schema
            .get_field("text")
            .context("Missing 'text' field in schema")?;
        let path_field = self
            .schema
            .get_field("path")
            .context("Missing 'path' field in schema")?;

        // Extract path from metadata if available
        let path = metadata
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Create Tantivy document
        let doc = doc!(
            id_field => node_id as u64,
            text_field => text,
            path_field => path,
        );

        // Add document to the index
        self.writer
            .add_document(doc)
            .context("Failed to add document to index")?;

        // Track the mapping from node_id to node_id (identity for now)
        // In Tantivy, we store the node_id in the document itself
        self.doc_map.insert(node_id as u64, node_id);

        debug!("Added document {} to keyword index (path: {})", node_id, path);

        Ok(())
    }

    /// Commit the index and save the document mapping
    pub fn commit(mut self, index_path: &Path) -> Result<()> {
        info!("Committing keyword index...");

        // Commit the Tantivy index
        self.writer
            .commit()
            .context("Failed to commit Tantivy index")?;

        // Save doc_map to disk
        let doc_map_path = index_path.join("doc_map.bin");
        let serialized =
            bincode::serialize(&self.doc_map).context("Failed to serialize doc_map")?;
        std::fs::write(&doc_map_path, serialized)
            .with_context(|| format!("Failed to write doc_map to: {}", doc_map_path.display()))?;

        info!(
            "Keyword index committed successfully ({} documents)",
            self.doc_map.len()
        );

        Ok(())
    }

    /// Get the number of documents added so far
    pub fn num_docs(&self) -> usize {
        self.doc_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_keyword_index_builder_creation() {
        let dir = tempdir().unwrap();
        let config = KeywordConfig::default();

        let builder = KeywordIndexBuilder::new(dir.path(), config);
        assert!(builder.is_ok());

        let builder = builder.unwrap();
        assert_eq!(builder.num_docs(), 0);
    }

    #[test]
    fn test_add_document() {
        let dir = tempdir().unwrap();
        let config = KeywordConfig::default();

        let mut builder = KeywordIndexBuilder::new(dir.path(), config).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("path".to_string(), Value::String("test.txt".to_string()));

        let result = builder.add_document(0, "Hello world", &metadata);
        assert!(result.is_ok());
        assert_eq!(builder.num_docs(), 1);
    }

    #[test]
    fn test_multiple_documents() {
        let dir = tempdir().unwrap();
        let config = KeywordConfig::default();

        let mut builder = KeywordIndexBuilder::new(dir.path(), config).unwrap();

        for i in 0..10 {
            let mut metadata = HashMap::new();
            metadata.insert(
                "path".to_string(),
                Value::String(format!("test{}.txt", i)),
            );
            builder
                .add_document(i, &format!("Document {}", i), &metadata)
                .unwrap();
        }

        assert_eq!(builder.num_docs(), 10);
    }

    #[test]
    fn test_commit() {
        let dir = tempdir().unwrap();
        let config = KeywordConfig::default();

        let mut builder = KeywordIndexBuilder::new(dir.path(), config).unwrap();

        let mut metadata = HashMap::new();
        metadata.insert("path".to_string(), Value::String("test.txt".to_string()));

        builder.add_document(0, "Hello world", &metadata).unwrap();

        let result = builder.commit(dir.path());
        assert!(result.is_ok());

        // Verify doc_map was saved
        let doc_map_path = dir.path().join("doc_map.bin");
        assert!(doc_map_path.exists());
    }
}
