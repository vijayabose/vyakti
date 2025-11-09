//! Configuration types for Vyakti.

use crate::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Main configuration for Vyakti index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VyaktiConfig {
    /// Index configuration
    pub index: IndexConfig,
    /// Storage configuration
    pub storage: StorageConfig,
}

impl Default for VyaktiConfig {
    fn default() -> Self {
        Self {
            index: IndexConfig::default(),
            storage: StorageConfig::default(),
        }
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
