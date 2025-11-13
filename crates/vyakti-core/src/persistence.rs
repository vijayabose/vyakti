//! Index persistence and serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use vyakti_common::{BackendConfig, Result, VyaktiError};
use vyakti_keyword::KeywordConfig;

/// Hybrid search configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchMetadata {
    /// Whether hybrid search is enabled
    pub enabled: bool,
    /// Default fusion strategy (rrf, weighted, cascade, etc.)
    pub fusion_strategy: String,
    /// Keyword search configuration
    pub keyword_config: KeywordConfig,
}

/// Index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Index name
    pub name: String,
    /// Index version
    pub version: u32,
    /// Embedding dimension
    pub dimension: usize,
    /// Number of documents
    pub num_documents: usize,
    /// Backend name
    pub backend_name: String,
    /// Backend configuration
    pub backend_config: BackendConfig,
    /// Creation timestamp
    pub created_at: u64,
    /// Hybrid search metadata (if hybrid search is enabled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hybrid_search: Option<HybridSearchMetadata>,
}

impl IndexMetadata {
    /// Current version
    pub const VERSION: u32 = 1;

    /// Create new index metadata
    pub fn new(
        name: String,
        dimension: usize,
        num_documents: usize,
        backend_name: String,
        backend_config: BackendConfig,
    ) -> Self {
        Self {
            name,
            version: Self::VERSION,
            dimension,
            num_documents,
            backend_name,
            backend_config,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            hybrid_search: None, // Default: no hybrid search
        }
    }
}

/// Document stored in index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredDocument {
    /// Document ID
    pub id: usize,
    /// Document text
    pub text: String,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Serializable graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Target node ID
    pub target: usize,
    /// Distance to target
    pub distance: f32,
}

/// Serializable graph layer
/// Maps node ID -> list of edges
pub type GraphLayer = HashMap<usize, Vec<GraphEdge>>;

/// Graph topology for HNSW indexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTopology {
    /// Layers (layer index -> node adjacency list)
    pub layers: Vec<GraphLayer>,
    /// Entry point (node ID, level)
    pub entry_point: Option<(usize, usize)>,
}

/// Index data to be persisted
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexData {
    /// Index metadata
    pub metadata: IndexMetadata,
    /// Stored documents
    pub documents: Vec<StoredDocument>,
    /// Serialized vectors
    pub vectors: Vec<Vec<f32>>,
    /// Pruned nodes (nodes without embeddings in compact mode)
    /// If None, all vectors are stored (normal mode)
    /// If Some, only non-pruned node vectors are in `vectors` array
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pruned_nodes: Option<Vec<usize>>,
    /// Graph topology (layers and entry point)
    /// Required for compact mode to restore graph structure
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_topology: Option<GraphTopology>,
}

/// Save index data to disk
pub fn save_index<P: AsRef<Path>>(path: P, data: &IndexData) -> Result<()> {
    let path = path.as_ref();

    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            VyaktiError::Storage(format!("Failed to create index directory: {}", e))
        })?;
    }

    // Serialize to JSON
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| VyaktiError::Serialization(format!("Failed to serialize index: {}", e)))?;

    // Write to file
    let mut file = File::create(path)
        .map_err(|e| VyaktiError::Storage(format!("Failed to create index file: {}", e)))?;

    file.write_all(json.as_bytes())
        .map_err(|e| VyaktiError::Storage(format!("Failed to write index: {}", e)))?;

    Ok(())
}

/// Load index data from disk
pub fn load_index<P: AsRef<Path>>(path: P) -> Result<IndexData> {
    let path = path.as_ref();

    // Read file
    let mut file = File::open(path)
        .map_err(|e| VyaktiError::Storage(format!("Failed to open index file: {}", e)))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| VyaktiError::Storage(format!("Failed to read index: {}", e)))?;

    // Deserialize from JSON
    let data: IndexData = serde_json::from_str(&contents)
        .map_err(|e| VyaktiError::Serialization(format!("Failed to deserialize index: {}", e)))?;

    // Validate version
    if data.metadata.version != IndexMetadata::VERSION {
        return Err(VyaktiError::Storage(format!(
            "Unsupported index version: {} (expected {})",
            data.metadata.version,
            IndexMetadata::VERSION
        )));
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_index_metadata_new() {
        let config = BackendConfig::default();
        let metadata = IndexMetadata::new(
            "test-index".to_string(),
            384,
            100,
            "hnsw".to_string(),
            config.clone(),
        );

        assert_eq!(metadata.name, "test-index");
        assert_eq!(metadata.version, IndexMetadata::VERSION);
        assert_eq!(metadata.dimension, 384);
        assert_eq!(metadata.num_documents, 100);
        assert_eq!(metadata.backend_name, "hnsw");
        assert_eq!(metadata.backend_config.graph_degree, config.graph_degree);
    }

    #[test]
    fn test_save_and_load_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        let config = BackendConfig::default();
        let metadata =
            IndexMetadata::new("test-index".to_string(), 384, 2, "hnsw".to_string(), config);

        let documents = vec![
            StoredDocument {
                id: 0,
                text: "First document".to_string(),
                metadata: HashMap::new(),
            },
            StoredDocument {
                id: 1,
                text: "Second document".to_string(),
                metadata: HashMap::new(),
            },
        ];

        let vectors = vec![vec![1.0; 384], vec![2.0; 384]];

        let data = IndexData {
            metadata,
            documents,
            vectors,
            pruned_nodes: None,
            graph_topology: None,
        };

        // Save
        save_index(&path, &data).unwrap();
        assert!(path.exists());

        // Load
        let loaded = load_index(&path).unwrap();
        assert_eq!(loaded.metadata.name, "test-index");
        assert_eq!(loaded.documents.len(), 2);
        assert_eq!(loaded.vectors.len(), 2);
        assert_eq!(loaded.documents[0].text, "First document");
        assert_eq!(loaded.documents[1].text, "Second document");
    }

    #[test]
    fn test_save_index_creates_directory() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested").join("dir").join("test.idx");

        let config = BackendConfig::default();
        let metadata = IndexMetadata::new("test".to_string(), 384, 0, "hnsw".to_string(), config);

        let data = IndexData {
            metadata,
            documents: vec![],
            vectors: vec![],
            pruned_nodes: None,
            graph_topology: None,
        };

        save_index(&path, &data).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_load_nonexistent_index() {
        let result = load_index("/nonexistent/path/test.idx");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.idx");

        // Write invalid JSON
        fs::write(&path, "not valid json").unwrap();

        let result = load_index(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_with_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.idx");

        let config = BackendConfig::default();
        let metadata = IndexMetadata::new("test".to_string(), 384, 1, "hnsw".to_string(), config);

        let mut doc_metadata = HashMap::new();
        doc_metadata.insert("source".to_string(), serde_json::json!("file.txt"));
        doc_metadata.insert("author".to_string(), serde_json::json!("test"));

        let documents = vec![StoredDocument {
            id: 0,
            text: "Test".to_string(),
            metadata: doc_metadata,
        }];

        let data = IndexData {
            metadata,
            documents,
            vectors: vec![vec![1.0; 384]],
            pruned_nodes: None,
            graph_topology: None,
        };

        save_index(&path, &data).unwrap();
        let loaded = load_index(&path).unwrap();

        assert_eq!(loaded.documents[0].metadata.len(), 2);
        assert_eq!(
            loaded.documents[0].metadata.get("source").unwrap(),
            &serde_json::json!("file.txt")
        );
    }
}
