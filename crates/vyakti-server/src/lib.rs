//! Vyakti server (REST + gRPC).

pub mod grpc_server;
pub mod rest;

// Re-export key types for convenience
pub use rest::{create_router, AppState};
