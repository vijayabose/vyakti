//! Vyakti command-line interface.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

use vyakti_backend_hnsw::HnswBackend;
use vyakti_chunking::{ChunkConfig, CodeLanguage, TextChunker};
use vyakti_common::BackendConfig;
use vyakti_core::{VyaktiBuilder, VyaktiSearcher};
use vyakti_embedding::{ensure_model, LlamaCppConfig, LlamaCppProvider};

#[cfg(feature = "ast")]
use vyakti_chunking::AstChunker;

#[derive(Parser)]
#[command(name = "vyakti")]
#[command(about = "Vyakti Vector Database CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a new index from documents
    Build {
        /// Index name
        name: String,

        /// Input directory or file
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for index
        #[arg(short, long, default_value = ".vyakti")]
        output: PathBuf,

        /// Graph degree (max connections per node)
        #[arg(long, default_value = "32")]
        graph_degree: usize,

        /// Build complexity
        #[arg(long, default_value = "128")]
        build_complexity: usize,

        /// Chunk size in tokens (default: 1024)
        #[arg(long, default_value = "1024")]
        chunk_size: usize,

        /// Chunk overlap in tokens (default: 512)
        #[arg(long, default_value = "512")]
        chunk_overlap: usize,

        /// Enable AST-aware code chunking for source files
        #[arg(long)]
        enable_code_chunking: bool,

        /// Disable chunking (use whole documents)
        #[arg(long)]
        no_chunking: bool,

        /// Embedding model to use (default: mxbai-embed-large)
        #[arg(long, default_value = "mxbai-embed-large")]
        embedding_model: String,

        /// Embedding dimension (default: 1024 for mxbai-embed-large)
        #[arg(long, default_value = "1024")]
        embedding_dimension: usize,

        /// Enable compact mode (LEANN): prune 95% of embeddings for massive storage savings
        #[arg(long)]
        compact: bool,

        /// Number of GPU layers to offload (0 = CPU only, -1 = all layers)
        #[arg(long, default_value = "0")]
        gpu_layers: i32,

        /// Path to custom model file (optional, will download default if not specified)
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Number of threads for inference (default: auto-detect CPU count)
        #[arg(long)]
        model_threads: Option<u32>,
    },

    /// Search an existing index
    Search {
        /// Index name
        name: String,

        /// Search query
        query: String,

        /// Number of results to return (before filtering)
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,

        /// Index directory
        #[arg(short, long, default_value = ".vyakti")]
        index_dir: PathBuf,

        /// Embedding model to use (default: mxbai-embed-large)
        #[arg(long, default_value = "mxbai-embed-large")]
        embedding_model: String,

        /// Embedding dimension (default: 1024 for mxbai-embed-large)
        #[arg(long, default_value = "1024")]
        embedding_dimension: usize,

        /// Maximum score threshold (lower = more relevant). Results with score > threshold are filtered out.
        /// Example: --max-score 0.5 keeps only results with score ≤ 0.5
        #[arg(long)]
        max_score: Option<f32>,

        /// Minimum relevance level (highly/moderately/weakly)
        /// - highly: score < 0.3
        /// - moderately: score < 0.7
        /// - weakly: score < 1.0
        #[arg(long, value_parser = parse_relevance)]
        min_relevance: Option<f32>,

        /// Show relevance labels (Highly/Moderately/Weakly relevant) with each result
        #[arg(long)]
        show_relevance: bool,

        /// Show metadata fields in search results (comma-separated list, or 'all' for all fields)
        /// Example: --show-metadata date,path or --show-metadata all
        #[arg(long)]
        show_metadata: Option<String>,

        /// Number of GPU layers to offload (0 = CPU only, -1 = all layers)
        #[arg(long, default_value = "0")]
        gpu_layers: i32,

        /// Path to custom model file (optional, will download default if not specified)
        #[arg(long)]
        model_path: Option<PathBuf>,

        /// Number of threads for inference (default: auto-detect CPU count)
        #[arg(long)]
        model_threads: Option<u32>,
    },

    /// List all indexes
    List {
        /// Index directory
        #[arg(short, long, default_value = ".vyakti")]
        index_dir: PathBuf,
    },

    /// Remove an index
    Remove {
        /// Index name
        name: String,

        /// Index directory
        #[arg(short, long, default_value = ".vyakti")]
        index_dir: PathBuf,

        /// Skip confirmation
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

/// Parse relevance level string to score threshold
fn parse_relevance(s: &str) -> Result<f32, String> {
    match s.to_lowercase().as_str() {
        "highly" => Ok(0.3),
        "moderately" => Ok(0.7),
        "weakly" => Ok(1.0),
        _ => Err(format!(
            "Invalid relevance level '{}'. Must be 'highly', 'moderately', or 'weakly'",
            s
        )),
    }
}

/// Get relevance label for a score
fn get_relevance_label(score: f32) -> (&'static str, colored::Color) {
    if score < 0.3 {
        ("Highly relevant", colored::Color::Green)
    } else if score < 0.7 {
        ("Moderately relevant", colored::Color::Yellow)
    } else {
        ("Weakly relevant", colored::Color::Red)
    }
}

/// Format metadata value for display
fn format_metadata_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Array(arr) => {
            format!("[{}]", arr.iter().map(|v| format_metadata_value(v)).collect::<Vec<_>>().join(", "))
        }
        serde_json::Value::Object(_) => value.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

/// Build an index from input files
async fn build_index(
    name: String,
    input: PathBuf,
    output: PathBuf,
    config: BackendConfig,
    chunk_size: usize,
    chunk_overlap: usize,
    enable_code_chunking: bool,
    no_chunking: bool,
    embedding_model: String,
    embedding_dimension: usize,
    compact: bool,
    gpu_layers: i32,
    model_path: Option<PathBuf>,
    model_threads: Option<u32>,
    verbose: bool,
) -> Result<()> {
    let start = Instant::now();

    println!("{}", format!("Building index '{}'", name).green().bold());

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output)
        .with_context(|| format!("Failed to create output directory: {:?}", output))?;

    // Read input files
    let files = if input.is_file() {
        vec![input.clone()]
    } else if input.is_dir() {
        // Define ALL supported file extensions
        let text_extensions = ["txt"];
        let config_extensions = ["md", "markdown", "json", "yaml", "yml", "toml", "csv", "html", "htm"];
        let document_extensions = ["pdf", "ipynb", "docx", "xlsx", "pptx"];
        let code_extensions = [
            // Core languages
            "py", "rs", "java", "ts", "tsx", "cs",
            // Extended languages
            "js", "jsx", "mjs", "cjs", "go",
            "c", "h", "cpp", "cc", "cxx", "hpp", "hxx", "hh",
            "swift", "kt", "kts", "rb", "php"
        ];

        // Common directories to exclude from indexing
        let excluded_dirs = [
            ".git", ".svn", ".hg",           // Version control
            ".vscode", ".idea", ".vs",       // IDEs
            "node_modules", ".npm",          // Node.js
            "target", "build", "dist",       // Build outputs
            "__pycache__", ".pytest_cache",  // Python
            ".cargo", ".rustup",             // Rust
            ".venv", "venv", "env",          // Python virtual envs
            "data", "datasets",              // Data directories (often contain large files)
            "test-docs", "sample-docs",      // Test document directories
        ];

        WalkDir::new(&input)
            .into_iter()
            .filter_entry(|entry| {
                // Skip excluded directories
                if entry.file_type().is_dir() {
                    let dir_name = entry.file_name().to_str().unwrap_or("");
                    !excluded_dirs.contains(&dir_name)
                } else {
                    true
                }
            })
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                if !path.is_file() {
                    return None;
                }

                let ext = path.extension()?.to_str()?;

                // Always accept text files
                if text_extensions.contains(&ext) {
                    return Some(path.to_path_buf());
                }

                // Accept config/document files (text-formats feature)
                if config_extensions.contains(&ext) {
                    return Some(path.to_path_buf());
                }

                // Accept document files (document-formats feature)
                if document_extensions.contains(&ext) {
                    return Some(path.to_path_buf());
                }

                // Accept code files only if code chunking is enabled
                if enable_code_chunking && code_extensions.contains(&ext) {
                    return Some(path.to_path_buf());
                }

                None
            })
            .collect()
    } else {
        anyhow::bail!("Input path does not exist: {:?}", input);
    };

    if files.is_empty() {
        if enable_code_chunking {
            anyhow::bail!("No supported files found in input: {:?}\nSupported: .txt, .md, .json, .yaml, .toml, .csv, .html, .pdf, .ipynb, .docx, .xlsx, .pptx, and code files (.py, .rs, .java, .ts, .js, .go, .c, .cpp, .swift, .kt, .rb, .php)", input);
        } else {
            anyhow::bail!("No supported files (.txt, .md, .json, .yaml, .toml, .csv, .html, .pdf, .ipynb, .docx, .xlsx, .pptx) found in input: {:?}\nTip: Use --enable-code-chunking to also index code files", input);
        }
    }

    if verbose {
        println!("  {} Found {} file(s)", "→".cyan(), files.len());
        if enable_code_chunking {
            println!("  {} Indexing all supported formats including code files", "→".cyan());
        } else {
            println!("  {} Indexing text and document files (use --enable-code-chunking for code files)", "→".cyan());
        }
    }

    // Create backend and embedding provider
    let backend = Box::new(HnswBackend::with_config(config.clone()));

    if verbose {
        println!("  {} Initializing llama.cpp embedding provider...", "→".cyan());
        println!("  {} Using model: {}", "→".cyan(), embedding_model);
        println!("  {} Embedding dimension: {}", "→".cyan(), embedding_dimension);
    }

    // Ensure model is available (download if necessary)
    if verbose {
        println!("  {} Ensuring model is available...", "→".cyan());
    }

    let model_file_path = ensure_model(model_path)
        .await
        .context("Failed to ensure model is available")?;

    if verbose {
        println!("  {} Using model at: {}", "✓".green(), model_file_path.display());
    }

    // Create llama.cpp config
    let llama_config = LlamaCppConfig {
        model_path: model_file_path,
        n_gpu_layers: gpu_layers.max(0) as u32,
        n_ctx: 512,
        n_threads: model_threads.unwrap_or_else(|| num_cpus::get() as u32),
        dimension: embedding_dimension,
        normalize: true,
    };

    if verbose {
        println!("  {} GPU layers: {}", "→".cyan(), llama_config.n_gpu_layers);
        println!("  {} Threads: {}", "→".cyan(), llama_config.n_threads);
    }

    let embedding_provider = Arc::new(
        LlamaCppProvider::new(llama_config)
            .context("Failed to initialize llama.cpp embedding provider")?,
    );

    if verbose {
        println!("  {} Embedding provider initialized", "✓".green());
    }

    // Create builder
    let mut builder = VyaktiBuilder::with_config(backend, embedding_provider, config);

    // Add documents with progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Create text chunker if needed
    let text_chunker = if !no_chunking {
        let chunk_config = ChunkConfig {
            chunk_size,
            chunk_overlap,
            is_code: false,
            ..Default::default()
        };
        Some(TextChunker::new(chunk_config).context("Failed to create text chunker")?)
    } else {
        None
    };

    let mut total_chunks = 0;
    let mut skipped_files = 0;

    for file_path in &files {
        // Determine file extension
        let file_ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        // Read file content based on extension - skip file on error instead of failing
        let text = if file_ext == "txt" {
            match fs::read_to_string(file_path) {
                Ok(content) => content,
                Err(e) => {
                    if verbose {
                        eprintln!("  {} Skipping file (read error): {:?} - {}", "⚠".yellow(), file_path, e);
                    }
                    skipped_files += 1;
                    pb.inc(1);
                    continue;
                }
            }
        } else {
            match vyakti_chunking::FileReader::read_file(file_path) {
                Ok(content) => content,
                Err(e) => {
                    if verbose {
                        eprintln!("  {} Skipping file (parse error): {:?} - {}", "⚠".yellow(), file_path, e);
                    }
                    skipped_files += 1;
                    pb.inc(1);
                    continue;
                }
            }
        };

        let mut base_metadata = std::collections::HashMap::new();
        base_metadata.insert(
            "path".to_string(),
            serde_json::json!(file_path.display().to_string()),
        );

        // Determine if we should use AST chunking for code files

        if no_chunking {
            // No chunking - use whole document
            builder.add_text(text, Some(base_metadata));
            total_chunks += 1;
        } else if enable_code_chunking {
            if let Some(_language) = CodeLanguage::from_extension(file_ext) {
                // Use AST chunking for code files
                #[cfg(feature = "ast")]
                {
                    let mut ast_chunker =
                        AstChunker::new(_language, chunk_size * 4, chunk_overlap * 4) // Convert to chars
                            .context("Failed to create AST chunker")?;

                    match ast_chunker.chunk_with_metadata(&text, base_metadata.clone()) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                builder.add_text(chunk.text, Some(chunk.metadata));
                                total_chunks += 1;
                            }
                        }
                        Err(_) => {
                            // Fallback to text chunking if AST fails
                            if let Some(ref chunker) = text_chunker {
                                let chunks = chunker.chunk_with_metadata(&text, base_metadata);
                                for chunk in chunks {
                                    builder.add_text(chunk.text, Some(chunk.metadata));
                                    total_chunks += 1;
                                }
                            }
                        }
                    }
                }

                #[cfg(not(feature = "ast"))]
                {
                    // AST feature not enabled, fall back to text chunking
                    if let Some(ref chunker) = text_chunker {
                        let chunks = chunker.chunk_with_metadata(&text, base_metadata);
                        for chunk in chunks {
                            builder.add_text(chunk.text, Some(chunk.metadata));
                            total_chunks += 1;
                        }
                    }
                }
            } else {
                // Use text chunking for non-code files
                if let Some(ref chunker) = text_chunker {
                    let chunks = chunker.chunk_with_metadata(&text, base_metadata);
                    for chunk in chunks {
                        builder.add_text(chunk.text, Some(chunk.metadata));
                        total_chunks += 1;
                    }
                } else {
                    builder.add_text(text, Some(base_metadata));
                    total_chunks += 1;
                }
            }
        } else {
            // Standard text chunking
            if let Some(ref chunker) = text_chunker {
                let chunks = chunker.chunk_with_metadata(&text, base_metadata);
                for chunk in chunks {
                    builder.add_text(chunk.text, Some(chunk.metadata));
                    total_chunks += 1;
                }
            } else {
                builder.add_text(text, Some(base_metadata));
                total_chunks += 1;
            }
        }

        pb.inc(1);
    }
    let processed_count = files.len() - skipped_files;
    if skipped_files > 0 {
        pb.finish_with_message(format!(
            "Processed {} files into {} chunks ({} skipped)",
            processed_count, total_chunks, skipped_files
        ));
    } else {
        pb.finish_with_message(format!("Processed {} files into {} chunks", processed_count, total_chunks));
    }

    // Build index
    println!("  {} Building index...", "→".cyan());
    let index_path = output.join(&name);

    if compact {
        println!("  {} Using compact mode (LEANN) - will prune embeddings for storage savings", "→".cyan());
        let (path, stats) = builder
            .build_index_compact(&index_path, None)
            .await
            .with_context(|| {
                format!(
                    "Failed to build compact index '{}'. Path: {:?}",
                    name, index_path
                )
            })?;

        let elapsed = start.elapsed();
        println!(
            "{}",
            format!(
                "✓ Compact index '{}' built successfully in {:.2}s",
                name,
                elapsed.as_secs_f64()
            )
            .green()
            .bold()
        );

        // Display pruning statistics
        println!("\n{}", "Compact Mode Statistics:".cyan().bold());
        println!("  {} Storage Savings: {:.1}%", "→".cyan(), stats.savings_percent);
        println!("  {} Total Nodes: {}", "→".cyan(), stats.total_nodes);
        println!("  {} Embeddings Kept: {}", "→".cyan(), stats.embeddings_kept);
        println!("  {} Embeddings Pruned: {}", "→".cyan(), stats.embeddings_pruned);
        println!("  {} Original Size: {}", "→".cyan(), stats.storage_before_human());
        println!("  {} Compact Size: {}", "→".cyan(), stats.storage_after_human());
        println!("  {} Retention Rate: {:.1}%", "→".cyan(), stats.retention_rate() * 100.0);

        if verbose {
            println!("  {} Indexed {} documents into {} chunks", "→".cyan(), processed_count, total_chunks);
            if skipped_files > 0 {
                println!("  {} Skipped {} files due to errors", "→".yellow(), skipped_files);
            }
            println!("  {} Index saved to: {:?}", "→".cyan(), path);
        }
    } else {
        builder
            .build_index(&index_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to build index '{}'. Path: {:?}",
                    name, index_path
                )
            })?;

        let elapsed = start.elapsed();
        println!(
            "{}",
            format!(
                "✓ Index '{}' built successfully in {:.2}s",
                name,
                elapsed.as_secs_f64()
            )
            .green()
            .bold()
        );

        if verbose {
            println!("  {} Indexed {} documents into {} chunks", "→".cyan(), processed_count, total_chunks);
            if skipped_files > 0 {
                println!("  {} Skipped {} files due to errors", "→".yellow(), skipped_files);
            }
            println!("  {} Index saved to: {:?}", "→".cyan(), index_path);
        }
    }

    Ok(())
}

/// Search an index
async fn search_index(
    name: String,
    query: String,
    top_k: usize,
    index_dir: PathBuf,
    embedding_model: String,
    embedding_dimension: usize,
    max_score: Option<f32>,
    min_relevance: Option<f32>,
    show_relevance: bool,
    show_metadata: Option<String>,
    gpu_layers: i32,
    model_path: Option<PathBuf>,
    model_threads: Option<u32>,
    verbose: bool,
) -> Result<()> {
    let start = Instant::now();

    println!(
        "{}",
        format!("Searching index '{}' for: \"{}\"", name, query)
            .green()
            .bold()
    );

    let index_path = index_dir.join(&name);
    if !index_path.exists() {
        anyhow::bail!("Index '{}' not found at {:?}", name, index_path);
    }

    // Create backend and embedding provider for searching
    let backend = Box::new(HnswBackend::new());

    if verbose {
        println!("  {} Initializing llama.cpp embedding provider...", "→".cyan());
        println!("  {} Using model: {}", "→".cyan(), embedding_model);
        println!("  {} Embedding dimension: {}", "→".cyan(), embedding_dimension);
    }

    // Ensure model is available (download if necessary)
    if verbose {
        println!("  {} Ensuring model is available...", "→".cyan());
    }

    let model_file_path = ensure_model(model_path)
        .await
        .context("Failed to ensure model is available")?;

    if verbose {
        println!("  {} Using model at: {}", "✓".green(), model_file_path.display());
    }

    // Create llama.cpp config
    let llama_config = LlamaCppConfig {
        model_path: model_file_path,
        n_gpu_layers: gpu_layers.max(0) as u32,
        n_ctx: 512,
        n_threads: model_threads.unwrap_or_else(|| num_cpus::get() as u32),
        dimension: embedding_dimension,
        normalize: true,
    };

    if verbose {
        println!("  {} GPU layers: {}", "→".cyan(), llama_config.n_gpu_layers);
        println!("  {} Threads: {}", "→".cyan(), llama_config.n_threads);
    }

    let embedding_provider = Arc::new(
        LlamaCppProvider::new(llama_config)
            .context("Failed to initialize llama.cpp embedding provider")?,
    );

    if verbose {
        println!("  {} Embedding provider initialized", "✓".green());
    }

    // Load the index from disk
    if verbose {
        println!("  {} Loading index from disk...", "→".cyan());
    }

    let searcher = VyaktiSearcher::load(index_path, backend, embedding_provider)
        .await
        .context("Failed to load index")?;

    if verbose {
        println!("  {} Index loaded successfully", "✓".green());
    }

    // Search
    let all_results = searcher.search(&query, top_k).await?;

    let elapsed = start.elapsed();

    // Apply score filtering if specified
    let score_threshold = max_score.or(min_relevance);
    let results: Vec<_> = if let Some(threshold) = score_threshold {
        all_results
            .into_iter()
            .filter(|r| r.score <= threshold)
            .collect()
    } else {
        all_results
    };

    let total_results = results.len();

    if results.is_empty() {
        println!("{}", "No results found".yellow());
        if score_threshold.is_some() {
            println!(
                "  {} Try a higher score threshold or broader relevance level",
                "→".cyan()
            );
        }
    } else {
        // Show filtering info if applied
        if let Some(threshold) = score_threshold {
            println!(
                "\n{} (found in {:.3}s, filtered by score ≤ {:.2}):",
                format!("Top {} results", total_results).cyan().bold(),
                elapsed.as_secs_f64(),
                threshold
            );
        } else {
            println!(
                "\n{} (found in {:.3}s):",
                format!("Top {} results", total_results).cyan().bold(),
                elapsed.as_secs_f64()
            );
        }

        for (i, result) in results.iter().enumerate() {
            // Format result text
            let result_text = if result.text.is_empty() {
                format!("Document {}", result.id).dimmed().to_string()
            } else {
                result.text.clone()
            };

            // Print result with or without relevance label
            if show_relevance {
                let (label, color) = get_relevance_label(result.score);
                println!(
                    "\n{}. {} (score: {:.4}) [{}]",
                    (i + 1).to_string().cyan().bold(),
                    result_text,
                    result.score,
                    label.color(color)
                );
            } else {
                println!(
                    "\n{}. {} (score: {:.4})",
                    (i + 1).to_string().cyan().bold(),
                    result_text,
                    result.score
                );
            }

            // Show metadata based on --show-metadata option or verbose mode
            if let Some(ref metadata_spec) = show_metadata {
                if metadata_spec.trim().to_lowercase() == "all" {
                    // Show all metadata fields
                    if !result.metadata.is_empty() {
                        println!("   {}", "Metadata:".dimmed());
                        for (key, value) in &result.metadata {
                            println!("     {}: {}", key.cyan(), format_metadata_value(value));
                        }
                    }
                } else {
                    // Show specific metadata fields
                    let fields: Vec<&str> = metadata_spec.split(',').map(|s| s.trim()).collect();
                    let mut shown_any = false;
                    for field in fields {
                        if let Some(value) = result.metadata.get(field) {
                            if !shown_any {
                                println!("   {}", "Metadata:".dimmed());
                                shown_any = true;
                            }
                            println!("     {}: {}", field.cyan(), format_metadata_value(value));
                        }
                    }
                }
            } else if verbose && !result.metadata.is_empty() {
                // Verbose mode - show all metadata in raw format
                println!("   Metadata: {:?}", result.metadata);
            }
        }
    }

    Ok(())
}

/// List all indexes
fn list_indexes(index_dir: PathBuf, verbose: bool) -> Result<()> {
    println!("{}", "Available indexes:".green().bold());

    if !index_dir.exists() {
        println!("  {}", "No indexes found".yellow());
        println!(
            "  {} Index directory does not exist: {:?}",
            "→".cyan(),
            index_dir
        );
        return Ok(());
    }

    let entries: Vec<_> = fs::read_dir(&index_dir)
        .with_context(|| format!("Failed to read index directory: {:?}", index_dir))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir() || entry.path().is_file())
        .collect();

    if entries.is_empty() {
        println!("  {}", "No indexes found".yellow());
    } else {
        for entry in entries {
            let name = entry.file_name();
            let path = entry.path();

            print!("  {} {}", "•".cyan(), name.to_string_lossy().bold());

            if verbose {
                if let Ok(metadata) = fs::metadata(&path) {
                    let size = metadata.len();
                    print!(" ({})", format_size(size).dimmed());
                }
                println!();
                println!("    Path: {}", path.display().to_string().dimmed());
            } else {
                println!();
            }
        }
    }

    Ok(())
}

/// Remove an index
fn remove_index(name: String, index_dir: PathBuf, yes: bool) -> Result<()> {
    let index_path = index_dir.join(&name);

    if !index_path.exists() {
        anyhow::bail!("Index '{}' not found at {:?}", name, index_path);
    }

    if !yes {
        print!("{}", format!("Remove index '{}'? [y/N]: ", name).yellow());
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("{}", "Cancelled".dimmed());
            return Ok(());
        }
    }

    if index_path.is_dir() {
        fs::remove_dir_all(&index_path)
            .with_context(|| format!("Failed to remove index directory: {:?}", index_path))?;
    } else {
        fs::remove_file(&index_path)
            .with_context(|| format!("Failed to remove index file: {:?}", index_path))?;
    }

    println!(
        "{}",
        format!("✓ Index '{}' removed successfully", name).green()
    );

    Ok(())
}

/// Format file size in human-readable format
fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let subscriber = tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .compact()
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Build {
            name,
            input,
            output,
            graph_degree,
            build_complexity,
            chunk_size,
            chunk_overlap,
            enable_code_chunking,
            no_chunking,
            embedding_model,
            embedding_dimension,
            compact,
        } => {
            let config = BackendConfig {
                graph_degree,
                build_complexity,
                search_complexity: 32,
            };
            build_index(
                name,
                input,
                output,
                config,
                chunk_size,
                chunk_overlap,
                enable_code_chunking,
                no_chunking,
                embedding_model,
                embedding_dimension,
                compact,
                gpu_layers,
                model_path,
                model_threads,
                cli.verbose,
            )
            .await
        }
        Commands::Search {
            name,
            query,
            top_k,
            index_dir,
            embedding_model,
            embedding_dimension,
            max_score,
            min_relevance,
            show_relevance,
            show_metadata,
            gpu_layers,
            model_path,
            model_threads,
        } => {
            search_index(
                name,
                query,
                top_k,
                index_dir,
                embedding_model,
                embedding_dimension,
                max_score,
                min_relevance,
                show_relevance,
                show_metadata,
                gpu_layers,
                model_path,
                model_threads,
                cli.verbose,
            )
            .await
        }
        Commands::List { index_dir } => list_indexes(index_dir, cli.verbose),
        Commands::Remove {
            name,
            index_dir,
            yes,
        } => remove_index(name, index_dir, yes),
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0.00 B");
        assert_eq!(format_size(500), "500.00 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1048576), "1.00 MB");
        assert_eq!(format_size(1073741824), "1.00 GB");
    }

    #[test]
    fn test_cli_parse_build() {
        let cli = Cli::parse_from(["vyakti", "build", "test-index", "-i", "/path/to/docs"]);
        match cli.command {
            Commands::Build { name, input, .. } => {
                assert_eq!(name, "test-index");
                assert_eq!(input, PathBuf::from("/path/to/docs"));
            }
            _ => panic!("Expected Build command"),
        }
    }

    #[test]
    fn test_cli_parse_search() {
        let cli = Cli::parse_from(["vyakti", "search", "test-index", "query text"]);
        match cli.command {
            Commands::Search { name, query, .. } => {
                assert_eq!(name, "test-index");
                assert_eq!(query, "query text");
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_parse_list() {
        let cli = Cli::parse_from(["vyakti", "list"]);
        match cli.command {
            Commands::List { .. } => {}
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_parse_remove() {
        let cli = Cli::parse_from(["vyakti", "remove", "test-index", "-y"]);
        match cli.command {
            Commands::Remove { name, yes, .. } => {
                assert_eq!(name, "test-index");
                assert!(yes);
            }
            _ => panic!("Expected Remove command"),
        }
    }

    #[test]
    fn test_cli_with_verbose() {
        let cli = Cli::parse_from(["vyakti", "-v", "list"]);
        assert!(cli.verbose);
    }
}
