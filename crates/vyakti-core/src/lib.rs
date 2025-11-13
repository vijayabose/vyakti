//! Core API for Vyakti vector database.
//!
//! Provides the main `VyaktiBuilder` and `VyaktiSearcher` interfaces.

pub mod builder;
pub mod evaluation;
pub mod hybrid;
pub mod persistence;
pub mod searcher;

pub use builder::{Document, VyaktiBuilder};
pub use evaluation::{
    AggregatedMetrics, EvaluationDataset, QueryMetrics, SearchEvaluator, TestQuery,
};
pub use hybrid::{FusionStrategy, HybridSearcher};
pub use persistence::{
    load_index, save_index, HybridSearchMetadata, IndexData, IndexMetadata, StoredDocument,
};
pub use searcher::VyaktiSearcher;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::builder::{Document, VyaktiBuilder};
    pub use crate::evaluation::{AggregatedMetrics, EvaluationDataset, SearchEvaluator};
    pub use crate::hybrid::{FusionStrategy, HybridSearcher};
    pub use crate::searcher::VyaktiSearcher;
    pub use vyakti_common::{Result, SearchResult, VyaktiError};
}
