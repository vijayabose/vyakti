//! Vyakti command-line interface.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "vyakti")]
#[command(about = "Vyakti Vector Database CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a new index
    Build {
        /// Index name
        name: String,
        /// Input directory or files
        #[arg(short, long)]
        input: String,
    },
    /// Search an index
    Search {
        /// Index name
        name: String,
        /// Search query
        query: String,
        /// Number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
    },
    /// List all indexes
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Build { name, input } => {
            println!("Building index '{}' from '{}'...", name, input);
            println!("(Not yet implemented)");
        }
        Commands::Search { name, query, top_k } => {
            println!("Searching index '{}' for '{}' (top-{})...", name, query, top_k);
            println!("(Not yet implemented)");
        }
        Commands::List => {
            println!("Listing indexes...");
            println!("(Not yet implemented)");
        }
    }

    Ok(())
}
