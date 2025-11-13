//! Configuration types for Vyakti.

use crate::{DistanceMetric, Result, VyaktiError};
use serde::{Deserialize, Serialize};

/// Main configuration for Vyakti index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VyaktiConfig {
    /// Index configuration
    pub index: IndexConfig,
    /// Storage configuration
    pub storage: StorageConfig,
}

impl VyaktiConfig {
    /// Validate the configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid, otherwise returns an error.
    pub fn validate(&self) -> Result<()> {
        self.index.validate()?;
        self.storage.validate()?;
        Ok(())
    }
}

/// Index-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Index name
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub metric: DistanceMetric,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            dimension: 384,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl IndexConfig {
    /// Validate the index configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid, otherwise returns an error.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(VyaktiError::Config(
                "Index name cannot be empty".to_string(),
            ));
        }

        if self.dimension == 0 {
            return Err(VyaktiError::Config(
                "Index dimension must be greater than 0".to_string(),
            ));
        }

        if self.dimension > 10000 {
            return Err(VyaktiError::Config(
                "Index dimension cannot exceed 10000".to_string(),
            ));
        }

        Ok(())
    }
}

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Use compact storage (with recomputation)
    pub compact: bool,
    /// Enable recomputation during search
    pub recompute: bool,
    /// Use memory-mapped files
    pub memory_map: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            compact: true,
            recompute: true,
            memory_map: true,
        }
    }
}

impl StorageConfig {
    /// Validate the storage configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid, otherwise returns an error.
    pub fn validate(&self) -> Result<()> {
        if self.compact && !self.recompute {
            return Err(VyaktiError::Config(
                "Compact storage requires recomputation to be enabled".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vyakti_config_default() {
        let config = VyaktiConfig::default();
        assert_eq!(config.index.name, "default");
        assert_eq!(config.index.dimension, 384);
        assert_eq!(config.index.metric, DistanceMetric::Cosine);
        assert!(config.storage.compact);
        assert!(config.storage.recompute);
        assert!(config.storage.memory_map);
    }

    #[test]
    fn test_vyakti_config_validate() {
        let config = VyaktiConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.name, "default");
        assert_eq!(config.dimension, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_index_config_validate_success() {
        let config = IndexConfig {
            name: "test-index".to_string(),
            dimension: 512,
            metric: DistanceMetric::Euclidean,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_index_config_validate_empty_name() {
        let config = IndexConfig {
            name: "".to_string(),
            dimension: 384,
            metric: DistanceMetric::Cosine,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("name cannot be empty"));
    }

    #[test]
    fn test_index_config_validate_zero_dimension() {
        let config = IndexConfig {
            name: "test".to_string(),
            dimension: 0,
            metric: DistanceMetric::Cosine,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimension must be greater than 0"));
    }

    #[test]
    fn test_index_config_validate_dimension_too_large() {
        let config = IndexConfig {
            name: "test".to_string(),
            dimension: 10001,
            metric: DistanceMetric::Cosine,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("dimension cannot exceed 10000"));
    }

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert!(config.compact);
        assert!(config.recompute);
        assert!(config.memory_map);
    }

    #[test]
    fn test_storage_config_validate_success() {
        let config = StorageConfig {
            compact: true,
            recompute: true,
            memory_map: false,
        };
        assert!(config.validate().is_ok());

        let config = StorageConfig {
            compact: false,
            recompute: false,
            memory_map: true,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_storage_config_validate_compact_without_recompute() {
        let config = StorageConfig {
            compact: true,
            recompute: false,
            memory_map: true,
        };
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Compact storage requires recomputation"));
    }

    #[test]
    fn test_config_serialization() {
        let config = VyaktiConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: VyaktiConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.index.name, deserialized.index.name);
        assert_eq!(config.index.dimension, deserialized.index.dimension);
        assert_eq!(config.index.metric, deserialized.index.metric);
    }

    #[test]
    fn test_index_config_serialization() {
        let config = IndexConfig {
            name: "my-index".to_string(),
            dimension: 768,
            metric: DistanceMetric::DotProduct,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: IndexConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.name, deserialized.name);
        assert_eq!(config.dimension, deserialized.dimension);
        assert_eq!(config.metric, deserialized.metric);
    }

    #[test]
    fn test_storage_config_serialization() {
        let config = StorageConfig {
            compact: false,
            recompute: true,
            memory_map: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: StorageConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.compact, deserialized.compact);
        assert_eq!(config.recompute, deserialized.recompute);
        assert_eq!(config.memory_map, deserialized.memory_map);
    }

    #[test]
    fn test_config_clone() {
        let config = VyaktiConfig::default();
        let cloned = config.clone();

        assert_eq!(config.index.name, cloned.index.name);
        assert_eq!(config.index.dimension, cloned.index.dimension);
    }
}
