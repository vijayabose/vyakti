//! Error types for Vyakti.

/// Main error type for Vyakti operations.
#[derive(Debug, thiserror::Error)]
pub enum VyaktiError {
    /// Index not found
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    /// Backend error
    #[error("Backend error: {0}")]
    Backend(String),

    /// Embedding computation error
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Text generation error
    #[error("Generation error: {0}")]
    Generation(String),

    /// Storage/IO error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

// Add From implementations for common error types
impl From<String> for VyaktiError {
    fn from(s: String) -> Self {
        VyaktiError::InvalidInput(s)
    }
}

impl From<&str> for VyaktiError {
    fn from(s: &str) -> Self {
        VyaktiError::InvalidInput(s.to_string())
    }
}

/// Result type alias using VyaktiError.
pub type Result<T> = std::result::Result<T, VyaktiError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_variants() {
        let err = VyaktiError::IndexNotFound("test-index".to_string());
        assert!(err.to_string().contains("test-index"));

        let err = VyaktiError::Backend("backend failed".to_string());
        assert!(err.to_string().contains("backend failed"));

        let err = VyaktiError::Embedding("embedding failed".to_string());
        assert!(err.to_string().contains("embedding failed"));

        let err = VyaktiError::Storage("storage failed".to_string());
        assert!(err.to_string().contains("storage failed"));

        let err = VyaktiError::Config("invalid config".to_string());
        assert!(err.to_string().contains("invalid config"));

        let err = VyaktiError::InvalidInput("bad input".to_string());
        assert!(err.to_string().contains("bad input"));

        let err = VyaktiError::Serialization("serde failed".to_string());
        assert!(err.to_string().contains("serde failed"));
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: VyaktiError = io_err.into();
        assert!(matches!(err, VyaktiError::Io(_)));
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn test_error_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid json").unwrap_err();
        let err: VyaktiError = json_err.into();
        assert!(matches!(err, VyaktiError::Json(_)));
        assert!(err.to_string().contains("JSON error"));
    }

    #[test]
    fn test_error_from_string() {
        let err: VyaktiError = "test error".into();
        assert!(matches!(err, VyaktiError::InvalidInput(_)));
        assert!(err.to_string().contains("test error"));

        let err: VyaktiError = String::from("another error").into();
        assert!(matches!(err, VyaktiError::InvalidInput(_)));
        assert!(err.to_string().contains("another error"));
    }

    #[test]
    fn test_result_type() {
        let ok_result: Result<i32> = Ok(42);
        assert!(ok_result.is_ok());
        if let Ok(value) = ok_result {
            assert_eq!(value, 42);
        }

        let err_result: Result<i32> = Err(VyaktiError::InvalidInput("test".to_string()));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_debug() {
        let err = VyaktiError::Backend("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Backend"));
    }

    #[test]
    fn test_error_chaining() {
        fn inner_fn() -> Result<()> {
            Err(VyaktiError::Backend("inner error".to_string()))
        }

        fn outer_fn() -> Result<()> {
            inner_fn()?;
            Ok(())
        }

        let result = outer_fn();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inner error"));
    }
}
