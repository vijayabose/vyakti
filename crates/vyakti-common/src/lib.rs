//! Common types, traits, and utilities for Vyakti vector database.
//!
//! This crate provides the foundational types and traits used across all
//! Vyakti components, including error handling, configuration, and core traits.

pub mod config;
pub mod error;
pub mod metadata_filter;
pub mod traits;
pub mod types;

pub use config::*;
pub use error::{Result, VyaktiError};
pub use metadata_filter::*;
pub use traits::*;
pub use types::*;
