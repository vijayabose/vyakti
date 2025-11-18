//! Vyakti REST API server.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vyakti_server::{create_router, AppState};

#[derive(Parser)]
#[command(name = "vyakti-server")]
#[command(about = "Vyakti Vector Database REST API Server", long_about = None)]
#[command(version)]
struct Cli {
    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Storage directory for indexes
    #[arg(short, long, default_value = ".vyakti")]
    storage_dir: PathBuf,

    /// Authentication token (optional)
    #[arg(long, env = "VYAKTI_AUTH_TOKEN")]
    auth_token: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vyakti_server=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    // Create storage directory if it doesn't exist
    std::fs::create_dir_all(&cli.storage_dir)?;

    // Create app state
    let state = AppState::new(cli.storage_dir.clone(), cli.auth_token);

    // Create router
    let app = create_router(state);

    // Create listener
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", cli.port)).await?;

    tracing::info!("🚀 Vyakti server listening on port {}", cli.port);
    tracing::info!("📁 Storage directory: {:?}", cli.storage_dir);
    tracing::info!("📖 API documentation:");
    tracing::info!("   GET  /health                         - Health check");
    tracing::info!("   GET  /api/v1/indexes                 - List all indexes");
    tracing::info!("   POST /api/v1/indexes                 - Create new index");
    tracing::info!("   POST /api/v1/indexes/:id/documents   - Add documents");
    tracing::info!("   POST /api/v1/indexes/:id/search      - Search index");
    tracing::info!("   DELETE /api/v1/indexes/:id           - Delete index");

    // Start server
    axum::serve(listener, app).await?;

    Ok(())
}
