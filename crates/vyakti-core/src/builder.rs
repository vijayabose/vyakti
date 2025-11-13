//! Index builder.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{info, warn};
use vyakti_common::{Backend, BackendConfig, EmbeddingProvider, Result, VyaktiError};
use vyakti_keyword::{KeywordConfig, KeywordIndexBuilder};

/// Document to be indexed
#[derive(Debug, Clone)]
pub struct Document {
    /// Document ID
    pub id: usize,
    /// Document text
    pub text: String,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Builder for creating Vyakti indexes
pub struct VyaktiBuilder {
    /// Backend for vector search
    backend: Arc<RwLock<Box<dyn Backend>>>,
    /// Embedding provider
    embedding_provider: Arc<dyn EmbeddingProvider>,
    /// Backend configuration
    backend_config: BackendConfig,
    /// Documents to be indexed
    documents: Vec<Document>,
}

impl VyaktiBuilder {
    /// Create a new builder with backend and embedding provider
    ///
    /// # Arguments
    ///
    /// * `backend` - Backend implementation for vector search
    /// * `embedding_provider` - Provider for computing embeddings
    ///
    /// # Returns
    ///
    /// A new VyaktiBuilder instance
    pub fn new(backend: Box<dyn Backend>, embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            backend: Arc::new(RwLock::new(backend)),
            embedding_provider,
            backend_config: BackendConfig::default(),
            documents: Vec::new(),
        }
    }

    /// Create a new builder with custom backend configuration
    pub fn with_config(
        backend: Box<dyn Backend>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        backend_config: BackendConfig,
    ) -> Self {
        Self {
            backend: Arc::new(RwLock::new(backend)),
            embedding_provider,
            backend_config,
            documents: Vec::new(),
        }
    }

    /// Add a text document to be indexed
    ///
    /// # Arguments
    ///
    /// * `text` - Document text
    /// * `metadata` - Optional document metadata
    ///
    /// # Returns
    ///
    /// Document ID
    pub fn add_text(
        &mut self,
        text: impl Into<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> usize {
        let id = self.documents.len();
        self.documents.push(Document {
            id,
            text: text.into(),
            metadata: metadata.unwrap_or_default(),
        });
        id
    }

    /// Add multiple text documents
    pub fn add_texts(
        &mut self,
        texts: &[String],
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) {
        for (i, text) in texts.iter().enumerate() {
            let doc_metadata = metadata
                .as_ref()
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();
            self.add_text(text.clone(), Some(doc_metadata));
        }
    }

    /// Get number of documents added
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Build the index
    ///
    /// # Returns
    ///
    /// Ok(()) if successful
    pub async fn build(&mut self) -> Result<()> {
        if self.documents.is_empty() {
            return Err(VyaktiError::Backend(
                "No documents added to builder".to_string(),
            ));
        }

        // Extract texts for embedding
        let texts: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();

        // Compute embeddings
        let embeddings = self.embedding_provider.embed(&texts).await?;

        // Validate dimensions
        let expected_dim = self.embedding_provider.dimension();
        for (i, embedding) in embeddings.iter().enumerate() {
            if embedding.len() != expected_dim {
                return Err(VyaktiError::Embedding(format!(
                    "Embedding {} has dimension {}, expected {}",
                    i,
                    embedding.len(),
                    expected_dim
                )));
            }
        }

        // Build index with backend
        let mut backend = self.backend.write().await;
        backend.build(&embeddings, &self.backend_config).await?;

        // Set document data for each node
        for (i, doc) in self.documents.iter().enumerate() {
            backend.set_document_data(i, doc.text.clone(), doc.metadata.clone());
        }

        Ok(())
    }

    /// Build the index in compact mode (LEANN)
    ///
    /// This builds the index normally, then prunes 95% of embeddings to achieve
    /// massive storage savings while maintaining search quality through
    /// on-demand recomputation.
    ///
    /// # Arguments
    ///
    /// * `pruning_config` - Optional pruning configuration
    ///
    /// # Returns
    ///
    /// Pruning statistics showing storage savings
    pub async fn build_compact(
        &mut self,
        pruning_config: Option<vyakti_backend_hnsw::PruningConfig>,
    ) -> Result<vyakti_backend_hnsw::PruningStats> {
        // First build the normal index
        self.build().await?;

        // Then enable compact mode
        // This requires downcasting to HnswBackend
        let mut backend = self.backend.write().await;

        // Try to downcast to HnswBackend
        let hnsw_backend = backend
            .as_any_mut()
            .downcast_mut::<vyakti_backend_hnsw::HnswBackend>()
            .ok_or_else(|| {
                VyaktiError::Backend(
                    "Compact mode is only supported for HNSW backend".to_string(),
                )
            })?;

        // Enable compact mode and return stats
        // Pass embedding provider for recomputation service
        hnsw_backend.enable_compact_mode(pruning_config, self.embedding_provider.clone())
    }

    /// Build the index and save to disk
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index
    ///
    /// # Returns
    ///
    /// Path to the saved index
    pub async fn build_index<P: AsRef<Path>>(&mut self, path: P) -> Result<PathBuf> {
        // Build the index
        self.build().await?;

        // Get index name from path
        let path = path.as_ref();
        let index_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();

        // Get backend name
        let backend = self.backend.read().await;
        let backend_name = backend.name().to_string();
        drop(backend);

        // Collect embeddings that were used to build the index
        let texts: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
        let vectors = self.embedding_provider.embed(&texts).await?;

        // Create index metadata
        let metadata = crate::persistence::IndexMetadata::new(
            index_name,
            self.embedding_provider.dimension(),
            self.documents.len(),
            backend_name,
            self.backend_config.clone(),
        );

        // Create stored documents
        let stored_docs: Vec<crate::persistence::StoredDocument> = self
            .documents
            .iter()
            .map(|doc| crate::persistence::StoredDocument {
                id: doc.id,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();

        // Export graph topology (if HNSW backend)
        let backend = self.backend.read().await;
        let graph_topology = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let (layers, entry_point) = hnsw.graph().export_topology();

                // Convert to persistence format
                let serializable_layers: Vec<crate::persistence::GraphLayer> = layers
                    .into_iter()
                    .map(|layer| {
                        layer
                            .into_iter()
                            .map(|(node_id, edges)| {
                                let edge_list: Vec<crate::persistence::GraphEdge> = edges
                                    .into_iter()
                                    .map(|(target, distance)| crate::persistence::GraphEdge {
                                        target,
                                        distance,
                                    })
                                    .collect();
                                (node_id, edge_list)
                            })
                            .collect()
                    })
                    .collect();

                crate::persistence::GraphTopology {
                    layers: serializable_layers,
                    entry_point,
                }
            });
        drop(backend);

        // Create index data
        let index_data = crate::persistence::IndexData {
            metadata,
            documents: stored_docs,
            vectors,
            pruned_nodes: None, // Normal mode: all vectors stored
            graph_topology,
        };

        // Save to disk
        crate::persistence::save_index(path, &index_data)?;

        Ok(path.to_path_buf())
    }

    /// Build the index in compact mode (LEANN) and save to disk
    ///
    /// This method:
    /// 1. Builds the full HNSW index
    /// 2. Identifies hub nodes (top 5% by degree)
    /// 3. Prunes embeddings for non-hub nodes (95% storage savings)
    /// 4. Saves the compact index to disk
    ///
    /// The saved index contains:
    /// - Full graph structure (connections)
    /// - Document texts for ALL nodes (needed for recomputation)
    /// - Embeddings only for hub nodes (5% of nodes)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index
    /// * `pruning_config` - Optional pruning configuration
    ///
    /// # Returns
    ///
    /// Tuple of (path to saved index, pruning statistics)
    pub async fn build_index_compact<P: AsRef<Path>>(
        &mut self,
        path: P,
        pruning_config: Option<vyakti_backend_hnsw::PruningConfig>,
    ) -> Result<(PathBuf, vyakti_backend_hnsw::PruningStats)> {
        // Build in compact mode
        let stats = self.build_compact(pruning_config).await?;

        // Get index name from path
        let path = path.as_ref();
        let index_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();

        // Get backend name and pruned nodes
        let backend = self.backend.read().await;
        let backend_name = backend.name().to_string();

        // Get pruned nodes from backend (if HNSW)
        let pruned_node_ids = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let pruned_set = hnsw.graph().pruned_nodes();
                let mut pruned_vec: Vec<usize> = pruned_set.into_iter().collect();
                pruned_vec.sort_unstable();
                pruned_vec
            });
        drop(backend);

        // Compute embeddings only for non-pruned nodes (hub nodes)
        let (vectors, pruned_nodes) = if let Some(ref pruned_ids) = pruned_node_ids {
            // Compact mode: only compute embeddings for non-pruned nodes
            let pruned_set: std::collections::HashSet<usize> = pruned_ids.iter().copied().collect();
            let hub_texts: Vec<String> = self
                .documents
                .iter()
                .filter(|doc| !pruned_set.contains(&doc.id))
                .map(|doc| doc.text.clone())
                .collect();

            let hub_vectors = self.embedding_provider.embed(&hub_texts).await?;
            (hub_vectors, Some(pruned_ids.clone()))
        } else {
            // Fallback: save all vectors if not HNSW backend
            let texts: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
            let all_vectors = self.embedding_provider.embed(&texts).await?;
            (all_vectors, None)
        };

        // Create index metadata
        let metadata = crate::persistence::IndexMetadata::new(
            index_name,
            self.embedding_provider.dimension(),
            self.documents.len(),
            backend_name,
            self.backend_config.clone(),
        );

        // Create stored documents - must include ALL documents for recomputation
        let stored_docs: Vec<crate::persistence::StoredDocument> = self
            .documents
            .iter()
            .map(|doc| crate::persistence::StoredDocument {
                id: doc.id,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();

        // Export graph topology (if HNSW backend)
        let backend = self.backend.read().await;
        let graph_topology = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let (layers, entry_point) = hnsw.graph().export_topology();

                // Convert to persistence format
                let serializable_layers: Vec<crate::persistence::GraphLayer> = layers
                    .into_iter()
                    .map(|layer| {
                        layer
                            .into_iter()
                            .map(|(node_id, edges)| {
                                let edge_list: Vec<crate::persistence::GraphEdge> = edges
                                    .into_iter()
                                    .map(|(target, distance)| crate::persistence::GraphEdge {
                                        target,
                                        distance,
                                    })
                                    .collect();
                                (node_id, edge_list)
                            })
                            .collect()
                    })
                    .collect();

                crate::persistence::GraphTopology {
                    layers: serializable_layers,
                    entry_point,
                }
            });
        drop(backend);

        // Create index data
        let index_data = crate::persistence::IndexData {
            metadata,
            documents: stored_docs,
            vectors,
            pruned_nodes, // Compact mode: list of pruned node IDs
            graph_topology,
        };

        // Save to disk
        crate::persistence::save_index(path, &index_data)?;

        Ok((path.to_path_buf(), stats))
    }

    /// Build a hybrid index (vector + keyword) and save to disk
    ///
    /// This method builds both a vector index (HNSW) and a keyword index (BM25/Tantivy)
    /// for improved search accuracy, especially for code search.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index
    /// * `keyword_config` - Optional keyword search configuration (uses defaults if None)
    ///
    /// # Returns
    ///
    /// Path to the saved hybrid index
    pub async fn build_index_hybrid<P: AsRef<Path>>(
        &mut self,
        path: P,
        keyword_config: Option<KeywordConfig>,
    ) -> Result<PathBuf> {
        let path = path.as_ref();

        info!("Building hybrid index at: {}", path.display());

        // Build vector index first
        info!("Building vector index...");
        self.build().await?;

        // Build keyword index if enabled
        let keyword_config = keyword_config.unwrap_or_default();
        if keyword_config.enabled {
            info!("Building keyword index with BM25 (k1={}, b={})...",
                  keyword_config.k1, keyword_config.b);

            let keyword_path = path.join("keyword");
            std::fs::create_dir_all(&keyword_path).map_err(|e| {
                VyaktiError::Storage(format!("Failed to create keyword index directory: {}", e))
            })?;

            let mut keyword_builder = KeywordIndexBuilder::new(&keyword_path, keyword_config.clone())
                .map_err(|e| VyaktiError::Storage(format!("Failed to create keyword index builder: {}", e)))?;

            // Add all documents to keyword index
            for doc in &self.documents {
                keyword_builder
                    .add_document(doc.id, &doc.text, &doc.metadata)
                    .map_err(|e| VyaktiError::Storage(format!("Failed to add document to keyword index: {}", e)))?;
            }

            // Commit keyword index
            keyword_builder.commit(&keyword_path)
                .map_err(|e| VyaktiError::Storage(format!("Failed to commit keyword index: {}", e)))?;

            info!("Keyword index built successfully");
        } else {
            warn!("Keyword search is disabled in config");
        }

        // Get index name from path
        let index_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();

        // Get backend name
        let backend = self.backend.read().await;
        let backend_name = backend.name().to_string();
        drop(backend);

        // Collect embeddings for vector storage
        let texts: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
        let vectors = self.embedding_provider.embed(&texts).await?;

        // Create index metadata with hybrid flag
        let mut metadata = crate::persistence::IndexMetadata::new(
            index_name,
            self.embedding_provider.dimension(),
            self.documents.len(),
            backend_name,
            self.backend_config.clone(),
        );

        // Add hybrid search metadata
        metadata.hybrid_search = Some(crate::persistence::HybridSearchMetadata {
            enabled: keyword_config.enabled,
            fusion_strategy: "rrf".to_string(), // Default strategy
            keyword_config: keyword_config.clone(),
        });

        // Create stored documents
        let stored_docs: Vec<crate::persistence::StoredDocument> = self
            .documents
            .iter()
            .map(|doc| crate::persistence::StoredDocument {
                id: doc.id,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();

        // Export graph topology (if HNSW backend)
        let backend = self.backend.read().await;
        let graph_topology = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let (layers, entry_point) = hnsw.graph().export_topology();

                // Convert to persistence format
                let serializable_layers: Vec<crate::persistence::GraphLayer> = layers
                    .into_iter()
                    .map(|layer| {
                        layer
                            .into_iter()
                            .map(|(node_id, edges)| {
                                let edge_list: Vec<crate::persistence::GraphEdge> = edges
                                    .into_iter()
                                    .map(|(target, distance)| crate::persistence::GraphEdge {
                                        target,
                                        distance,
                                    })
                                    .collect();
                                (node_id, edge_list)
                            })
                            .collect()
                    })
                    .collect();

                crate::persistence::GraphTopology {
                    layers: serializable_layers,
                    entry_point,
                }
            });
        drop(backend);

        // Create index data
        let index_data = crate::persistence::IndexData {
            metadata,
            documents: stored_docs,
            vectors,
            pruned_nodes: None, // Normal mode: all vectors stored
            graph_topology,
        };

        // Save to disk - for hybrid indexes, save as path/index.json
        // because path is already a directory containing keyword/
        let index_file_path = path.join("index.json");
        crate::persistence::save_index(&index_file_path, &index_data)?;

        info!("Hybrid index saved successfully");

        Ok(path.to_path_buf())
    }

    /// Build a hybrid index in compact mode and save to disk
    ///
    /// Combines the benefits of:
    /// - Compact mode: 95% storage savings via selective embedding storage
    /// - Hybrid search: Keyword + vector fusion for better code search
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index
    /// * `keyword_config` - Optional keyword search configuration
    /// * `pruning_config` - Optional pruning configuration for compact mode
    ///
    /// # Returns
    ///
    /// Tuple of (path, pruning stats)
    pub async fn build_index_hybrid_compact<P: AsRef<Path>>(
        &mut self,
        path: P,
        keyword_config: Option<KeywordConfig>,
        pruning_config: Option<vyakti_backend_hnsw::PruningConfig>,
    ) -> Result<(PathBuf, vyakti_backend_hnsw::PruningStats)> {
        let path = path.as_ref();

        info!("Building hybrid index in compact mode at: {}", path.display());

        // Build in compact mode
        let stats = self.build_compact(pruning_config).await?;

        // Build keyword index if enabled
        let keyword_config = keyword_config.unwrap_or_default();
        if keyword_config.enabled {
            info!("Building keyword index with BM25 (k1={}, b={})...",
                  keyword_config.k1, keyword_config.b);

            let keyword_path = path.join("keyword");
            std::fs::create_dir_all(&keyword_path).map_err(|e| {
                VyaktiError::Storage(format!("Failed to create keyword index directory: {}", e))
            })?;

            let mut keyword_builder = KeywordIndexBuilder::new(&keyword_path, keyword_config.clone())
                .map_err(|e| VyaktiError::Storage(format!("Failed to create keyword index builder: {}", e)))?;

            // Add all documents to keyword index
            for doc in &self.documents {
                keyword_builder
                    .add_document(doc.id, &doc.text, &doc.metadata)
                    .map_err(|e| VyaktiError::Storage(format!("Failed to add document to keyword index: {}", e)))?;
            }

            // Commit keyword index
            keyword_builder.commit(&keyword_path)
                .map_err(|e| VyaktiError::Storage(format!("Failed to commit keyword index: {}", e)))?;

            info!("Keyword index built successfully");
        } else {
            warn!("Keyword search is disabled in config");
        }

        // Get index name from path
        let index_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string();

        // Get backend name and pruned nodes
        let backend = self.backend.read().await;
        let backend_name = backend.name().to_string();

        // Get pruned nodes from backend (if HNSW)
        let pruned_node_ids = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let pruned_set = hnsw.graph().pruned_nodes();
                let mut pruned_vec: Vec<usize> = pruned_set.into_iter().collect();
                pruned_vec.sort_unstable();
                pruned_vec
            });
        drop(backend);

        // Compute embeddings only for non-pruned nodes (hub nodes)
        let (vectors, pruned_nodes) = if let Some(ref pruned_ids) = pruned_node_ids {
            let pruned_set: std::collections::HashSet<usize> = pruned_ids.iter().copied().collect();
            let hub_texts: Vec<String> = self
                .documents
                .iter()
                .filter(|doc| !pruned_set.contains(&doc.id))
                .map(|doc| doc.text.clone())
                .collect();

            let hub_vectors = self.embedding_provider.embed(&hub_texts).await?;
            (hub_vectors, Some(pruned_ids.clone()))
        } else {
            // Fallback: save all vectors if not HNSW backend
            let texts: Vec<String> = self.documents.iter().map(|d| d.text.clone()).collect();
            let all_vectors = self.embedding_provider.embed(&texts).await?;
            (all_vectors, None)
        };

        // Create index metadata with hybrid flag
        let mut metadata = crate::persistence::IndexMetadata::new(
            index_name,
            self.embedding_provider.dimension(),
            self.documents.len(),
            backend_name,
            self.backend_config.clone(),
        );

        // Add hybrid search metadata
        metadata.hybrid_search = Some(crate::persistence::HybridSearchMetadata {
            enabled: keyword_config.enabled,
            fusion_strategy: "rrf".to_string(), // Default strategy
            keyword_config: keyword_config.clone(),
        });

        // Create stored documents
        let stored_docs: Vec<crate::persistence::StoredDocument> = self
            .documents
            .iter()
            .map(|doc| crate::persistence::StoredDocument {
                id: doc.id,
                text: doc.text.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();

        // Export graph topology (if HNSW backend)
        let backend = self.backend.read().await;
        let graph_topology = backend
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .map(|hnsw| {
                let (layers, entry_point) = hnsw.graph().export_topology();

                // Convert to persistence format
                let serializable_layers: Vec<crate::persistence::GraphLayer> = layers
                    .into_iter()
                    .map(|layer| {
                        layer
                            .into_iter()
                            .map(|(node_id, edges)| {
                                let edge_list: Vec<crate::persistence::GraphEdge> = edges
                                    .into_iter()
                                    .map(|(target, distance)| crate::persistence::GraphEdge {
                                        target,
                                        distance,
                                    })
                                    .collect();
                                (node_id, edge_list)
                            })
                            .collect()
                    })
                    .collect();

                crate::persistence::GraphTopology {
                    layers: serializable_layers,
                    entry_point,
                }
            });
        drop(backend);

        // Create index data
        let index_data = crate::persistence::IndexData {
            metadata,
            documents: stored_docs,
            vectors,
            pruned_nodes,
            graph_topology,
        };

        // Save to disk - for hybrid indexes, save as path/index.json
        // because path is already a directory containing keyword/
        let index_file_path = path.join("index.json");
        crate::persistence::save_index(&index_file_path, &index_data)?;

        info!("Hybrid compact index saved successfully");

        Ok((path.to_path_buf(), stats))
    }

    /// Get reference to the backend
    pub fn backend(&self) -> Arc<RwLock<Box<dyn Backend>>> {
        Arc::clone(&self.backend)
    }

    /// Get reference to the embedding provider
    pub fn embedding_provider(&self) -> Arc<dyn EmbeddingProvider> {
        Arc::clone(&self.embedding_provider)
    }

    /// Get documents
    pub fn documents(&self) -> &[Document] {
        &self.documents
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use vyakti_common::{SearchResult, Vector};

    // Mock Backend for testing
    struct MockBackend {
        vectors: Vec<Vector>,
    }

    #[async_trait]
    impl Backend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }

        async fn build(&mut self, vectors: &[Vector], _config: &BackendConfig) -> Result<()> {
            self.vectors = vectors.to_vec();
            Ok(())
        }

        async fn search(&self, _query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
            let mut results = vec![];
            for i in 0..k.min(self.vectors.len()) {
                results.push(SearchResult {
                    id: i,
                    text: String::new(),
                    score: 0.5,
                    metadata: HashMap::new(),
                });
            }
            Ok(results)
        }

        fn len(&self) -> usize {
            self.vectors.len()
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    // Mock EmbeddingProvider for testing
    struct MockEmbeddingProvider {
        dimension: usize,
    }

    #[async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
            Ok(texts.iter().map(|_| vec![0.0; self.dimension]).collect())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_builder_new() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let builder = VyaktiBuilder::new(backend, embedding_provider);
        assert_eq!(builder.num_documents(), 0);
    }

    #[test]
    fn test_builder_add_text() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        let id = builder.add_text("Hello world", None);
        assert_eq!(id, 0);
        assert_eq!(builder.num_documents(), 1);
        assert_eq!(builder.documents()[0].text, "Hello world");
    }

    #[test]
    fn test_builder_add_text_with_metadata() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), serde_json::json!("test"));

        let id = builder.add_text("Hello world", Some(metadata));
        assert_eq!(id, 0);
        assert_eq!(
            builder.documents()[0].metadata.get("source").unwrap(),
            "test"
        );
    }

    #[test]
    fn test_builder_add_multiple_texts() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        let id1 = builder.add_text("First", None);
        let id2 = builder.add_text("Second", None);
        let id3 = builder.add_text("Third", None);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(builder.num_documents(), 3);
    }

    #[test]
    fn test_builder_add_texts() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        let texts = vec![
            "First".to_string(),
            "Second".to_string(),
            "Third".to_string(),
        ];
        builder.add_texts(&texts, None);

        assert_eq!(builder.num_documents(), 3);
        assert_eq!(builder.documents()[0].text, "First");
        assert_eq!(builder.documents()[1].text, "Second");
        assert_eq!(builder.documents()[2].text, "Third");
    }

    #[tokio::test]
    async fn test_builder_build() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        builder.add_text("Document 1", None);
        builder.add_text("Document 2", None);
        builder.add_text("Document 3", None);

        let result = builder.build().await;
        assert!(result.is_ok());

        // Check that backend has vectors
        let backend_arc = builder.backend();
        let backend = backend_arc.read().await;
        assert_eq!(backend.len(), 3);
    }

    #[tokio::test]
    async fn test_builder_build_empty() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        let result = builder.build().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No documents"));
    }

    #[tokio::test]
    async fn test_builder_with_config() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let config = BackendConfig {
            graph_degree: 16,
            build_complexity: 32,
            search_complexity: 16,
        };

        let mut builder = VyaktiBuilder::with_config(backend, embedding_provider, config.clone());

        builder.add_text("Test", None);
        let result = builder.build().await;
        assert!(result.is_ok());

        assert_eq!(builder.backend_config.graph_degree, 16);
        assert_eq!(builder.backend_config.build_complexity, 32);
    }

    #[tokio::test]
    async fn test_builder_build_index() {
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        builder.add_text("Document 1", None);
        builder.add_text("Document 2", None);

        let path = builder.build_index("/tmp/test-index").await;
        assert!(path.is_ok());
        assert_eq!(path.unwrap(), PathBuf::from("/tmp/test-index"));
    }

    #[tokio::test]
    async fn test_builder_dimension_mismatch() {
        // Mock embedding provider that returns wrong dimensions
        struct BadEmbeddingProvider {
            dimension: usize,
        }

        #[async_trait]
        impl EmbeddingProvider for BadEmbeddingProvider {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
                // Return vectors with wrong dimension
                Ok(texts.iter().map(|_| vec![0.0; 128]).collect())
            }

            fn dimension(&self) -> usize {
                self.dimension // Report 384 but return 128
            }

            fn name(&self) -> &str {
                "bad"
            }
        }

        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(BadEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);
        builder.add_text("Test", None);

        let result = builder.build().await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("dimension") || err.contains("Embedding"));
    }

    #[tokio::test]
    async fn test_builder_build_compact() {
        // Mock embedding provider that returns varied vectors for proper HNSW graph
        struct VariedEmbeddingProvider {
            dimension: usize,
        }

        #[async_trait]
        impl EmbeddingProvider for VariedEmbeddingProvider {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
                // Generate varied vectors based on text content to create a realistic graph
                Ok(texts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let val = (i as f32) / (texts.len() as f32);
                        let mut vec = vec![0.0; self.dimension];
                        vec[0] = val;
                        vec[1] = 1.0 - val;
                        vec[2] = val * 0.5;
                        vec
                    })
                    .collect())
            }

            fn dimension(&self) -> usize {
                self.dimension
            }

            fn name(&self) -> &str {
                "varied-mock"
            }
        }

        // Create an HNSW backend for compact mode testing
        let backend = Box::new(vyakti_backend_hnsw::HnswBackend::new());
        let embedding_provider = Arc::new(VariedEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        // Add many documents to ensure pruning is significant
        for i in 0..100 {
            builder.add_text(format!("Document number {}", i), None);
        }

        // Build in compact mode
        let stats = builder.build_compact(None).await.unwrap();

        // Verify storage savings
        // Note: With 100 nodes, we get ~70-76% savings. Larger graphs (1000+ nodes) achieve 95%+ savings
        // due to stronger hub-and-spoke patterns. We keep this test small for speed.
        assert!(stats.savings_percent >= 70.0, "Expected >=70% savings, got {}%", stats.savings_percent);
        assert!(stats.retention_rate() <= 0.30, "Expected <=30% retention, got {:.2}%", stats.retention_rate() * 100.0);
        assert_eq!(stats.total_nodes, 100);

        // Verify backend is in compact mode
        let backend_arc = builder.backend();
        let backend = backend_arc.read().await;
        let hnsw_backend = backend
            .as_ref()
            .as_any()
            .downcast_ref::<vyakti_backend_hnsw::HnswBackend>()
            .unwrap();

        assert!(hnsw_backend.is_compact_mode());
        assert!(hnsw_backend.num_pruned_nodes() > 0);
    }

    #[tokio::test]
    async fn test_builder_build_compact_with_config() {
        // Mock embedding provider that returns varied vectors
        struct VariedEmbeddingProvider {
            dimension: usize,
        }

        #[async_trait]
        impl EmbeddingProvider for VariedEmbeddingProvider {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
                Ok(texts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let val = (i as f32) / (texts.len() as f32);
                        let mut vec = vec![0.0; self.dimension];
                        vec[0] = val;
                        vec[1] = 1.0 - val;
                        vec[2] = val * 0.5;
                        vec
                    })
                    .collect())
            }

            fn dimension(&self) -> usize {
                self.dimension
            }

            fn name(&self) -> &str {
                "varied-mock"
            }
        }

        let backend = Box::new(vyakti_backend_hnsw::HnswBackend::new());
        let embedding_provider = Arc::new(VariedEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        for i in 0..100 {
            builder.add_text(format!("Doc {}", i), None);
        }

        // Custom pruning config: keep top 10% instead of 5%
        let pruning_config = vyakti_backend_hnsw::PruningConfig {
            degree_threshold_percentile: 0.90,
            ..Default::default()
        };

        let stats = builder.build_compact(Some(pruning_config)).await.unwrap();

        // Custom pruning with 90th percentile: should retain ~10% of embeddings
        // With 100 nodes, actual retention may be higher due to limited graph structure
        assert!(stats.retention_rate() <= 0.35, "Expected <=35% retention, got {:.2}%", stats.retention_rate() * 100.0);
        assert!(stats.retention_rate() > 0.05, "Expected >5% retention, got {:.2}%", stats.retention_rate() * 100.0);
    }

    #[tokio::test]
    async fn test_builder_build_compact_non_hnsw_backend() {
        // Using MockBackend which is not HNSW
        let backend = Box::new(MockBackend { vectors: vec![] });
        let embedding_provider = Arc::new(MockEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);
        builder.add_text("Test", None);

        // Should fail because compact mode only works with HNSW
        let result = builder.build_compact(None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("HNSW"));
    }

    #[tokio::test]
    async fn test_builder_build_index_compact() {
        // Mock embedding provider with varied vectors
        struct VariedEmbeddingProvider {
            dimension: usize,
        }

        #[async_trait]
        impl EmbeddingProvider for VariedEmbeddingProvider {
            async fn embed(&self, texts: &[String]) -> Result<Vec<Vector>> {
                Ok(texts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let val = (i as f32) / (texts.len() as f32);
                        let mut vec = vec![0.0; self.dimension];
                        vec[0] = val;
                        vec[1] = 1.0 - val;
                        vec[2] = val * 0.5;
                        vec
                    })
                    .collect())
            }

            fn dimension(&self) -> usize {
                self.dimension
            }

            fn name(&self) -> &str {
                "varied-mock"
            }
        }

        let backend = Box::new(vyakti_backend_hnsw::HnswBackend::new());
        let embedding_provider = Arc::new(VariedEmbeddingProvider { dimension: 384 });

        let mut builder = VyaktiBuilder::new(backend, embedding_provider);

        for i in 0..50 {
            builder.add_text(format!("Document {}", i), None);
        }

        // Build and save in compact mode
        let temp_dir = std::env::temp_dir();
        let index_path = temp_dir.join("test-compact-index");
        let (path, stats) = builder
            .build_index_compact(&index_path, None)
            .await
            .unwrap();

        // Verify path was returned
        assert_eq!(path, index_path);

        // Verify index was created
        assert!(path.exists());

        // Verify storage savings
        // Note: With 50 nodes, we get ~30-80% savings (high variability due to random graph construction).
        // Larger graphs (1000+ nodes) achieve consistent 95%+ savings with lower variability.
        assert!(stats.savings_percent >= 25.0, "Expected >=25% savings, got {}%", stats.savings_percent);
        assert!(stats.total_nodes == 50);

        // Cleanup
        std::fs::remove_dir_all(&index_path).ok();
    }
}
