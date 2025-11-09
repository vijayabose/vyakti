//! Common types, traits, and utilities for Vyakti vector database.
//!
//! This crate provides the foundational types and traits used across all
//! Vyakti components, including error handling, configuration, and core traits.

pub mod error;
pub mod types;
pub mod traits;
pub mod config;

pub use error::{Result, VyaktiError};
pub use types::*;
pub use traits::*;
pub use config::*;
