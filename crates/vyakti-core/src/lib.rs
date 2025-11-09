//! Core API for Vyakti vector database.
//!
//! Provides the main `VyaktiBuilder` and `VyaktiSearcher` interfaces.

pub mod builder;
pub mod searcher;

pub use builder::VyaktiBuilder;
pub use searcher::VyaktiSearcher;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::builder::VyaktiBuilder;
    pub use crate::searcher::VyaktiSearcher;
    pub use vyakti_common::{Result, VyaktiError, SearchResult};
}
