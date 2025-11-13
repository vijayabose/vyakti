//! Vyakti search evaluation binary
//!
//! Evaluates search quality using test datasets with ground truth relevance judgments.

use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use vyakti_backend_hnsw::HnswBackend;
use vyakti_core::evaluation::{AggregatedMetrics, EvaluationDataset, QueryMetrics, SearchEvaluator};
use vyakti_core::VyaktiSearcher;
use vyakti_embedding::{OllamaConfig, OllamaProvider};

#[derive(Parser)]
#[command(name = "vyakti-evaluate")]
#[command(about = "Evaluate Vyakti search quality", long_about = None)]
struct Cli {
    /// Index name
    #[arg(short, long)]
    index: String,

    /// Evaluation dataset JSON file
    #[arg(short, long)]
    dataset: PathBuf,

    /// Index directory
    #[arg(long, default_value = ".vyakti")]
    index_dir: PathBuf,

    /// K values for evaluation (comma-separated)
    #[arg(long, default_value = "1,3,5,10,20")]
    k_values: String,

    /// Embedding model
    #[arg(long, default_value = "mxbai-embed-large")]
    embedding_model: String,

    /// Embedding dimension
    #[arg(long, default_value = "1024")]
    embedding_dimension: usize,

    /// Show per-query metrics
    #[arg(short, long)]
    verbose: bool,

    /// Output JSON results to file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "╔══════════════════════════════════════════════════════════╗".blue());
    println!("{}", "║         VYAKTI SEARCH EVALUATION                         ║".blue());
    println!("{}", "╚══════════════════════════════════════════════════════════╝".blue());
    println!();

    // Load evaluation dataset
    println!("{}", "📂 Loading evaluation dataset...".cyan());
    let dataset = EvaluationDataset::from_json_file(
        cli.dataset
            .to_str()
            .context("Invalid dataset path")?,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load evaluation dataset: {}", e))?;

    println!("  {} Dataset: {}", "•".cyan(), dataset.name.yellow());
    println!("  {} Description: {}", "•".cyan(), dataset.description);
    println!("  {} Queries: {}", "•".cyan(), dataset.queries.len());
    println!();

    // Parse K values
    let k_values: Vec<usize> = cli
        .k_values
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if k_values.is_empty() {
        anyhow::bail!("Invalid K values: {}", cli.k_values);
    }

    println!("{}", "🔧 Initializing search system...".cyan());
    println!("  {} Index: {}", "•".cyan(), cli.index);
    println!("  {} Embedding model: {}", "•".cyan(), cli.embedding_model);
    println!("  {} K values: {:?}", "•".cyan(), k_values);
    println!();

    // Load index
    let index_path = cli.index_dir.join(&cli.index);
    let backend = Box::new(HnswBackend::new());
    let ollama_config = OllamaConfig {
        base_url: "http://localhost:11434".to_string(),
        model: cli.embedding_model,
        timeout_secs: 30,
    };

    let embedding_provider = Arc::new(
        OllamaProvider::new(ollama_config, cli.embedding_dimension)
            .await
            .context("Failed to initialize Ollama provider")?,
    );

    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .context("Failed to load index")?;

    println!("{}", "✓ Index loaded successfully".green());
    println!();

    // Create evaluator
    let evaluator = SearchEvaluator::with_k_values(k_values.clone());

    // Evaluate all queries
    println!("{}", "🔍 Evaluating queries...".cyan());

    let mut all_query_metrics: Vec<QueryMetrics> = Vec::new();
    let max_k = *k_values.iter().max().unwrap_or(&20);

    for (i, test_query) in dataset.queries.iter().enumerate() {
        print!("\r  Progress: {}/{} queries", i + 1, dataset.queries.len());
        std::io::Write::flush(&mut std::io::stdout())?;

        let start = Instant::now();
        let results = searcher
            .search(&test_query.query, max_k)
            .await
            .context(format!("Failed to search query: {}", test_query.query))?;
        let search_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let metrics = evaluator.evaluate_query(test_query, &results, search_time_ms);

        if cli.verbose {
            println!("\n  Query: {}", test_query.query.yellow());
            println!("    P@10: {:.4}", metrics.precision_at_k.get(&10).unwrap_or(&0.0));
            println!("    R@10: {:.4}", metrics.recall_at_k.get(&10).unwrap_or(&0.0));
            println!("    NDCG@10: {:.4}", metrics.ndcg_at_k.get(&10).unwrap_or(&0.0));
            println!("    MRR: {:.4}", metrics.reciprocal_rank);
            println!("    Time: {:.2}ms", search_time_ms);
        }

        all_query_metrics.push(metrics);
    }

    println!("\n{}", "✓ Evaluation complete".green());
    println!();

    // Aggregate metrics
    let aggregated = evaluator.aggregate_metrics(&all_query_metrics);

    // Print results
    evaluator.print_metrics(&aggregated);

    // Save to JSON if requested
    if let Some(output_path) = cli.output {
        let output = serde_json::to_string_pretty(&aggregated)?;
        std::fs::write(&output_path, output)?;
        println!("{}", format!("💾 Results saved to: {:?}", output_path).green());
        println!();
    }

    // Provide recommendations
    print_recommendations(&aggregated);

    Ok(())
}

fn print_recommendations(metrics: &AggregatedMetrics) {
    println!("{}", "💡 Recommendations:".yellow().bold());
    println!();

    let p10 = metrics.mean_precision_at_k.get(&10).unwrap_or(&0.0);
    let r10 = metrics.mean_recall_at_k.get(&10).unwrap_or(&0.0);
    let ndcg10 = metrics.mean_ndcg_at_k.get(&10).unwrap_or(&0.0);
    let map = metrics.mean_average_precision;
    let mrr = metrics.mean_reciprocal_rank;

    if *p10 < 0.3 {
        println!("  {} {}", "⚠️".red(), "Low Precision@10 (<0.3):");
        println!("     → Increase search_complexity (try 64 or 128)");
        println!("     → Try better chunking (smaller chunk_size)");
        println!("     → Consider better embedding model");
        println!();
    }

    if *r10 < 0.4 {
        println!("  {} {}", "⚠️".red(), "Low Recall@10 (<0.4):");
        println!("     → Increase graph_degree (try 32 or 64)");
        println!("     → Increase K value when searching");
        println!("     → Check if all relevant docs are indexed");
        println!();
    }

    if *ndcg10 < 0.5 {
        println!("  {} {}", "⚠️".red(), "Low NDCG@10 (<0.5):");
        println!("     → Ranking quality needs improvement");
        println!("     → Try different embedding model");
        println!("     → Optimize chunk_size and chunk_overlap");
        println!();
    }

    if map < 0.3 {
        println!("  {} {}", "⚠️".red(), "Low MAP (<0.3):");
        println!("     → Overall ranking quality is poor");
        println!("     → Consider upgrading embedding model");
        println!("     → Review document preprocessing");
        println!();
    }

    if mrr < 0.5 {
        println!("  {} {}", "⚠️".red(), "Low MRR (<0.5):");
        println!("     → First result often not relevant");
        println!("     → Increase search_complexity");
        println!("     → Use metadata filtering for better relevance");
        println!();
    }

    if metrics.queries_with_zero_results > 0 {
        let pct = metrics.queries_with_zero_results as f64 / metrics.num_queries as f64 * 100.0;
        println!("  {} {}", "⚠️".red(), format!("{:.1}% queries with zero results", pct));
        println!("     → Check if all documents are indexed");
        println!("     → Verify embedding model compatibility");
        println!();
    }

    if *p10 >= 0.7 && *r10 >= 0.6 && *ndcg10 >= 0.7 {
        println!("  {} {}", "✓".green(), "Metrics look good!");
        println!("     → Current configuration is performing well");
        println!("     → Consider testing with more queries for confidence");
        println!();
    }
}
