//! REST API server implementation.

use axum::{
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use vyakti_common::SearchResult;
use vyakti_core::VyaktiSearcher;

/// Authentication middleware
async fn auth_middleware(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // If no auth token is configured, skip authentication
    if state.auth_token.is_none() {
        return Ok(next.run(request).await);
    }

    let expected_token = state.auth_token.as_ref().unwrap();

    // Get Authorization header
    let auth_header = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError::Unauthorized("Missing Authorization header".to_string()))?;

    // Check for Bearer token
    if !auth_header.starts_with("Bearer ") {
        return Err(ApiError::Unauthorized(
            "Invalid Authorization format, expected 'Bearer <token>'".to_string(),
        ));
    }

    let token = &auth_header[7..]; // Skip "Bearer "

    // Validate token
    if token != expected_token {
        return Err(ApiError::Unauthorized("Invalid token".to_string()));
    }

    Ok(next.run(request).await)
}

/// API error response
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// API error type
#[derive(Debug)]
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    Unauthorized(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(ErrorResponse {
            error,
            details: None,
        });

        (status, body).into_response()
    }
}

/// Server state
#[derive(Clone)]
pub struct AppState {
    /// Storage directory for indexes
    pub storage_dir: PathBuf,
    /// In-memory cache of loaded searchers
    pub searchers: Arc<RwLock<HashMap<String, Arc<VyaktiSearcher>>>>,
    /// Authentication token (simple auth for demo)
    pub auth_token: Option<String>,
}

impl AppState {
    /// Create new app state
    pub fn new(storage_dir: PathBuf, auth_token: Option<String>) -> Self {
        Self {
            storage_dir,
            searchers: Arc::new(RwLock::new(HashMap::new())),
            auth_token,
        }
    }
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

/// Health check endpoint
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Create index request
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateIndexRequest {
    pub name: String,
    #[serde(default)]
    pub graph_degree: Option<usize>,
    #[serde(default)]
    pub build_complexity: Option<usize>,
}

/// Create index response
#[derive(Debug, Serialize)]
pub struct CreateIndexResponse {
    pub name: String,
    pub status: String,
    pub message: String,
}

/// Create a new index
async fn create_index(
    State(state): State<AppState>,
    Json(req): Json<CreateIndexRequest>,
) -> Result<Json<CreateIndexResponse>, ApiError> {
    // Validate index name
    if req.name.is_empty() {
        return Err(ApiError::BadRequest(
            "Index name cannot be empty".to_string(),
        ));
    }

    // Check if index already exists
    let index_path = state.storage_dir.join(&req.name);
    if index_path.exists() {
        return Err(ApiError::BadRequest(format!(
            "Index '{}' already exists",
            req.name
        )));
    }

    Ok(Json(CreateIndexResponse {
        name: req.name,
        status: "created".to_string(),
        message: "Index created successfully".to_string(),
    }))
}

/// Add documents request
#[derive(Debug, Serialize, Deserialize)]
pub struct AddDocumentsRequest {
    pub documents: Vec<DocumentInput>,
    /// Enable compact mode (LEANN): prune 95% of embeddings for storage savings
    #[serde(default)]
    pub compact: bool,
}

/// Document input
#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentInput {
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Add documents response
#[derive(Debug, Serialize)]
pub struct AddDocumentsResponse {
    pub added: usize,
    pub message: String,
    /// Compact mode statistics (only present if compact mode was used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_stats: Option<CompactModeStats>,
}

/// Compact mode statistics
#[derive(Debug, Serialize)]
pub struct CompactModeStats {
    pub total_nodes: usize,
    pub embeddings_kept: usize,
    pub embeddings_pruned: usize,
    pub savings_percent: f64,
    pub storage_before_bytes: usize,
    pub storage_after_bytes: usize,
}

/// Add documents to an index
async fn add_documents(
    State(state): State<AppState>,
    Path(index_id): Path<String>,
    Json(req): Json<AddDocumentsRequest>,
) -> Result<Json<AddDocumentsResponse>, ApiError> {
    use std::sync::Arc;
    use vyakti_backend_hnsw::HnswBackend;
    use vyakti_common::BackendConfig;
    use vyakti_embedding::{OllamaConfig, OllamaProvider};

    if req.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    // Create backend and embedding provider
    let config = BackendConfig::default();
    let backend = Box::new(HnswBackend::with_config(config.clone()));
    let ollama_config = OllamaConfig::default();
    let embedding_provider = Arc::new(
        OllamaProvider::new(ollama_config, 768)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to initialize embedding provider: {}", e)))?,
    );

    // Create builder
    let mut builder = vyakti_core::VyaktiBuilder::with_config(backend, embedding_provider, config);

    // Add documents
    for doc in &req.documents {
        builder.add_text(doc.text.clone(), Some(doc.metadata.clone()));
    }

    // Build index (compact or normal mode)
    let index_path = state.storage_dir.join(&index_id);
    let compact_stats = if req.compact {
        // Build in compact mode
        let (_path, stats) = builder
            .build_index_compact(&index_path, None)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to build compact index: {}", e)))?;

        // Convert PruningStats to CompactModeStats
        Some(CompactModeStats {
            total_nodes: stats.total_nodes,
            embeddings_kept: stats.embeddings_kept,
            embeddings_pruned: stats.embeddings_pruned,
            savings_percent: stats.savings_percent,
            storage_before_bytes: stats.storage_before_bytes,
            storage_after_bytes: stats.storage_after_bytes,
        })
    } else {
        // Build in normal mode
        builder
            .build_index(&index_path)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to build index: {}", e)))?;
        None
    };

    // Remove from cache (will be reloaded on next search)
    {
        let mut cache = state.searchers.write().await;
        cache.remove(&index_id);
    }

    let message = if req.compact {
        format!(
            "Successfully added {} documents in compact mode ({:.1}% storage savings)",
            req.documents.len(),
            compact_stats.as_ref().map(|s| s.savings_percent).unwrap_or(0.0)
        )
    } else {
        format!("Successfully added {} documents", req.documents.len())
    };

    Ok(Json(AddDocumentsResponse {
        added: req.documents.len(),
        message,
        compact_stats,
    }))
}

/// Search request
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub k: usize,
}

fn default_top_k() -> usize {
    10
}

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query: String,
    pub k: usize,
}

/// Search an index
async fn search(
    State(state): State<AppState>,
    Path(index_id): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    use std::sync::Arc;

    // Check if index exists
    let index_path = state.storage_dir.join(&index_id);
    if !index_path.exists() {
        return Err(ApiError::NotFound(format!(
            "Index '{}' not found",
            index_id
        )));
    }

    // Try to get searcher from cache
    let searcher = {
        let cache = state.searchers.read().await;
        cache.get(&index_id).cloned()
    };

    // If not in cache, load from disk
    let searcher = if let Some(s) = searcher {
        s
    } else {
        // Create backend and embedding provider
        use vyakti_backend_hnsw::HnswBackend;
        use vyakti_embedding::{OllamaConfig, OllamaProvider};

        let backend = Box::new(HnswBackend::new());
        let ollama_config = OllamaConfig::default();
        let embedding_provider = Arc::new(
            OllamaProvider::new(ollama_config, 768)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to initialize embedding provider: {}", e)))?,
        );

        // Load index
        let searcher = vyakti_core::VyaktiSearcher::load(&index_path, backend, embedding_provider)
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to load index: {}", e)))?;

        let searcher = Arc::new(searcher);

        // Add to cache
        {
            let mut cache = state.searchers.write().await;
            cache.insert(index_id.clone(), searcher.clone());
        }

        searcher
    };

    // Execute search
    let results = searcher
        .search(&req.query, req.k)
        .await
        .map_err(|e| ApiError::Internal(format!("Search failed: {}", e)))?;

    Ok(Json(SearchResponse {
        results,
        query: req.query,
        k: req.k,
    }))
}

/// List indexes response
#[derive(Debug, Serialize)]
pub struct ListIndexesResponse {
    pub indexes: Vec<IndexInfo>,
}

/// Index information
#[derive(Debug, Serialize)]
pub struct IndexInfo {
    pub name: String,
    pub path: String,
}

/// List all indexes
async fn list_indexes(
    State(state): State<AppState>,
) -> Result<Json<ListIndexesResponse>, ApiError> {
    let mut indexes = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&state.storage_dir) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                indexes.push(IndexInfo {
                    name: name.clone(),
                    path: entry.path().display().to_string(),
                });
            }
        }
    }

    Ok(Json(ListIndexesResponse { indexes }))
}

/// Delete index response
#[derive(Debug, Serialize)]
pub struct DeleteIndexResponse {
    pub message: String,
}

/// Delete an index
async fn delete_index(
    State(state): State<AppState>,
    Path(index_id): Path<String>,
) -> Result<Json<DeleteIndexResponse>, ApiError> {
    let index_path = state.storage_dir.join(&index_id);

    if !index_path.exists() {
        return Err(ApiError::NotFound(format!(
            "Index '{}' not found",
            index_id
        )));
    }

    // Remove from cache
    {
        let mut searchers = state.searchers.write().await;
        searchers.remove(&index_id);
    }

    // Delete from disk
    if index_path.is_dir() {
        std::fs::remove_dir_all(&index_path)
            .map_err(|e| ApiError::Internal(format!("Failed to delete index: {}", e)))?;
    } else {
        std::fs::remove_file(&index_path)
            .map_err(|e| ApiError::Internal(format!("Failed to delete index: {}", e)))?;
    }

    Ok(Json(DeleteIndexResponse {
        message: format!("Index '{}' deleted successfully", index_id),
    }))
}

/// Create the REST API router
pub fn create_router(state: AppState) -> Router {
    // Protected routes (require authentication if token is configured)
    let protected_routes = Router::new()
        .route("/api/v1/indexes", post(create_index))
        .route("/api/v1/indexes", get(list_indexes))
        .route("/api/v1/indexes/:id/documents", post(add_documents))
        .route("/api/v1/indexes/:id/search", post(search))
        .route("/api/v1/indexes/:id", delete(delete_index))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(health_check))
        .merge(protected_routes)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let temp_dir = std::env::temp_dir().join("vyakti-test");
        std::fs::create_dir_all(&temp_dir).ok();
        AppState::new(temp_dir, None)
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_index_empty_name() {
        let state = test_state();
        let app = create_router(state);

        let request = CreateIndexRequest {
            name: "".to_string(),
            graph_degree: None,
            build_complexity: None,
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/indexes")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_string(&request).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_list_indexes() {
        let state = test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/indexes")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_delete_nonexistent_index() {
        let state = test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/api/v1/indexes/nonexistent")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_authentication_missing_header() {
        let temp_dir = std::env::temp_dir().join("vyakti-auth-test");
        std::fs::create_dir_all(&temp_dir).ok();
        let state = AppState::new(temp_dir, Some("test-token".to_string()));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/indexes")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_authentication_invalid_token() {
        let temp_dir = std::env::temp_dir().join("vyakti-auth-test-2");
        std::fs::create_dir_all(&temp_dir).ok();
        let state = AppState::new(temp_dir, Some("test-token".to_string()));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/indexes")
                    .header("Authorization", "Bearer wrong-token")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_authentication_valid_token() {
        let temp_dir = std::env::temp_dir().join("vyakti-auth-test-3");
        std::fs::create_dir_all(&temp_dir).ok();
        let state = AppState::new(temp_dir, Some("test-token".to_string()));
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/indexes")
                    .header("Authorization", "Bearer test-token")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_documents_empty() {
        let state = test_state();
        let app = create_router(state);

        let request = AddDocumentsRequest {
            documents: vec![],
            compact: false,
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/indexes/test/documents")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_string(&request).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_nonexistent_index() {
        let state = test_state();
        let app = create_router(state);

        let request = SearchRequest {
            query: "test".to_string(),
            k: 10,
        };

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/indexes/nonexistent/search")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::to_string(&request).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_add_documents_compact_mode_flag() {
        let state = test_state();
        let _app = create_router(state);

        // Test that compact mode flag is properly accepted in the request
        let request_compact = AddDocumentsRequest {
            documents: vec![DocumentInput {
                text: "Test document".to_string(),
                metadata: HashMap::new(),
            }],
            compact: true,
        };

        // Serialize to ensure the compact field is included
        let json = serde_json::to_string(&request_compact).unwrap();
        assert!(json.contains("\"compact\":true"));

        // Test that compact mode defaults to false when not specified
        let json_no_compact = r#"{"documents":[{"text":"Test","metadata":{}}]}"#;
        let parsed: AddDocumentsRequest = serde_json::from_str(json_no_compact).unwrap();
        assert!(!parsed.compact);
    }

    #[tokio::test]
    async fn test_compact_stats_response_structure() {
        // Test that CompactModeStats can be serialized correctly
        let stats = CompactModeStats {
            total_nodes: 100,
            embeddings_kept: 5,
            embeddings_pruned: 95,
            savings_percent: 95.0,
            storage_before_bytes: 400000,
            storage_after_bytes: 20000,
        };

        let response = AddDocumentsResponse {
            added: 100,
            message: "Test message".to_string(),
            compact_stats: Some(stats),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"compact_stats\""));
        assert!(json.contains("\"total_nodes\":100"));
        assert!(json.contains("\"savings_percent\":95"));

        // Test that compact_stats is omitted when None
        let response_no_stats = AddDocumentsResponse {
            added: 50,
            message: "No compact mode".to_string(),
            compact_stats: None,
        };

        let json_no_stats = serde_json::to_string(&response_no_stats).unwrap();
        assert!(!json_no_stats.contains("\"compact_stats\""));
    }
}
