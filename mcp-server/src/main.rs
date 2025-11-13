//! Vyakti MCP Server - Native Rust implementation for Claude Code integration
//!
//! This server implements the Model Context Protocol (MCP) to enable Claude Code
//! to interact with Vyakti vector search indexes through semantic search.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::io::{self, BufRead, Write};
use tokio::process::Command;

/// MCP JSON-RPC request structure
#[derive(Debug, Deserialize)]
struct McpRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// MCP JSON-RPC response structure
#[derive(Debug, Serialize)]
struct McpResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

/// MCP error structure
#[derive(Debug, Serialize)]
struct McpError {
    code: i32,
    message: String,
}

/// Configuration from environment variables
struct ServerConfig {
    vyakti_bin: String,
    index_dir: String,
}

impl ServerConfig {
    fn from_env() -> Self {
        Self {
            vyakti_bin: env::var("VYAKTI_BIN").unwrap_or_else(|_| "vyakti".to_string()),
            index_dir: env::var("INDEX_DIR").unwrap_or_else(|_| ".vyakti".to_string()),
        }
    }
}

/// Handle MCP initialize request
fn handle_initialize(id: Option<Value>) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!({
            "capabilities": { "tools": {} },
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "vyakti-mcp",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        error: None,
    }
}

/// Handle MCP tools/list request
fn handle_tools_list(id: Option<Value>) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!({
            "tools": [
                {
                    "name": "vyakti_build",
                    "description": "🏗️ Build a vector search index from documents. Supports 35+ file formats including text, markdown, JSON, YAML, PDF, Office docs, and code files with AST-aware chunking.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Index name (e.g., 'my-docs', 'codebase')"
                            },
                            "input_path": {
                                "type": "string",
                                "description": "Path to file or directory to index"
                            },
                            "enable_code_chunking": {
                                "type": "boolean",
                                "description": "Enable AST-aware code chunking for programming languages (Python, Rust, Java, TypeScript, etc.)",
                                "default": false
                            },
                            "compact": {
                                "type": "boolean",
                                "description": "Enable compact mode (93% storage savings + 60% faster search)",
                                "default": false
                            },
                            "hybrid": {
                                "type": "boolean",
                                "description": "Enable hybrid search (semantic + keyword/BM25) for better code search accuracy. Default: true (enabled)",
                                "default": true
                            },
                            "bm25_k1": {
                                "type": "number",
                                "description": "BM25 k1 parameter (term frequency saturation, default: 1.2)",
                                "default": 1.2
                            },
                            "bm25_b": {
                                "type": "number",
                                "description": "BM25 b parameter (length normalization, default: 0.75)",
                                "default": 0.75
                            }
                        },
                        "required": ["name", "input_path"]
                    }
                },
                {
                    "name": "vyakti_search",
                    "description": "🔍 Search a vector index using semantic similarity. Perfect for finding relevant code, documentation, or data using natural language queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Index name to search (use vyakti_list to see available indexes)"
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query - can be natural language (e.g., 'how does authentication work') or keywords"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10, max: 50)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50
                            },
                            "max_score": {
                                "type": "number",
                                "description": "Maximum score threshold - only return results with score ≤ this value (lower score = more relevant). Example: 0.3 for highly relevant results only.",
                                "minimum": 0.0,
                                "maximum": 2.0
                            },
                            "fusion": {
                                "type": "string",
                                "description": "Fusion strategy for hybrid search: rrf (default), weighted, cascade, vector-only, keyword-only",
                                "default": "rrf",
                                "enum": ["rrf", "weighted", "cascade", "vector-only", "keyword-only"]
                            },
                            "fusion_param": {
                                "type": "number",
                                "description": "Fusion parameter: k for RRF (default: 60), alpha for weighted (default: 0.5), threshold for cascade (default: 5)"
                            },
                            "highlight": {
                                "type": "boolean",
                                "description": "Enable result highlighting to show matched text snippets with highlighted keywords (keyword/hybrid search only)",
                                "default": false
                            }
                        },
                        "required": ["name", "query"]
                    }
                },
                {
                    "name": "vyakti_list",
                    "description": "📋 List all available Vyakti indexes in the index directory.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "vyakti_remove",
                    "description": "🗑️ Remove a vector index permanently.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Index name to remove"
                            }
                        },
                        "required": ["name"]
                    }
                }
            ]
        })),
        error: None,
    }
}

/// Execute vyakti CLI command and capture output
async fn execute_vyakti_command(
    config: &ServerConfig,
    args: &[String],
) -> Result<String> {
    tracing::debug!("Executing: {} {:?}", config.vyakti_bin, args);

    let output = Command::new(&config.vyakti_bin)
        .args(args)
        .output()
        .await
        .with_context(|| format!("Failed to execute vyakti command: {:?}", args))?;

    // Combine stdout and stderr
    let mut result = String::new();

    if !output.stdout.is_empty() {
        // Strip ANSI escape codes from output
        let stripped = strip_ansi_escapes::strip(&output.stdout);
        result.push_str(&String::from_utf8_lossy(&stripped));
    }

    if !output.stderr.is_empty() {
        if !result.is_empty() {
            result.push_str("\n\n");
        }
        let stripped = strip_ansi_escapes::strip(&output.stderr);
        result.push_str(&String::from_utf8_lossy(&stripped));
    }

    if !output.status.success() {
        anyhow::bail!("Command failed with exit code: {}\n{}", output.status, result);
    }

    Ok(result.trim().to_string())
}

/// Handle vyakti_build tool call
async fn handle_build(
    config: &ServerConfig,
    args: &Value,
) -> Result<Value> {
    let name = args["name"].as_str()
        .context("Missing 'name' parameter")?;
    let input_path = args["input_path"].as_str()
        .context("Missing 'input_path' parameter")?;

    let mut cmd_args = vec![
        "build".to_string(),
        name.to_string(),
        "-i".to_string(),
        input_path.to_string(),
        "-o".to_string(),
        config.index_dir.clone(),
    ];

    if args.get("enable_code_chunking").and_then(|v| v.as_bool()).unwrap_or(false) {
        cmd_args.push("--enable-code-chunking".to_string());
    }

    if args.get("compact").and_then(|v| v.as_bool()).unwrap_or(false) {
        cmd_args.push("--compact".to_string());
    }

    // Hybrid search is enabled by default (matches CLI behavior)
    // Only add --no-hybrid if explicitly disabled
    if !args.get("hybrid").and_then(|v| v.as_bool()).unwrap_or(true) {
        cmd_args.push("--no-hybrid".to_string());
    }

    // Add BM25 parameters if provided (apply to hybrid search)
    if let Some(k1) = args.get("bm25_k1").and_then(|v| v.as_f64()) {
        cmd_args.push("--bm25-k1".to_string());
        cmd_args.push(k1.to_string());
    }

    if let Some(b) = args.get("bm25_b").and_then(|v| v.as_f64()) {
        cmd_args.push("--bm25-b".to_string());
        cmd_args.push(b.to_string());
    }

    let output = execute_vyakti_command(config, &cmd_args).await?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("✓ Index '{}' built successfully\n\n{}", name, output)
        }]
    }))
}

/// Handle vyakti_search tool call
async fn handle_search(
    config: &ServerConfig,
    args: &Value,
) -> Result<Value> {
    let name = args["name"].as_str()
        .context("Missing 'name' parameter")?;
    let query = args["query"].as_str()
        .context("Missing 'query' parameter")?;

    let top_k = args.get("top_k")
        .and_then(|v| v.as_i64())
        .unwrap_or(10);

    let mut cmd_args = vec![
        "search".to_string(),
        name.to_string(),
        query.to_string(),
        "-k".to_string(),
        top_k.to_string(),
        "-i".to_string(),
        config.index_dir.clone(),
    ];

    if let Some(max_score) = args.get("max_score").and_then(|v| v.as_f64()) {
        cmd_args.push("--max-score".to_string());
        cmd_args.push(max_score.to_string());
    }

    if let Some(fusion) = args.get("fusion").and_then(|v| v.as_str()) {
        cmd_args.push("--fusion".to_string());
        cmd_args.push(fusion.to_string());
    }

    if let Some(fusion_param) = args.get("fusion_param").and_then(|v| v.as_f64()) {
        cmd_args.push("--fusion-param".to_string());
        cmd_args.push(fusion_param.to_string());
    }

    if args.get("highlight").and_then(|v| v.as_bool()).unwrap_or(false) {
        cmd_args.push("--highlight".to_string());
    }

    let output = execute_vyakti_command(config, &cmd_args).await?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": output
        }]
    }))
}

/// Handle vyakti_list tool call
async fn handle_list(config: &ServerConfig) -> Result<Value> {
    let cmd_args = vec![
        "list".to_string(),
        "-i".to_string(),
        config.index_dir.clone(),
    ];

    let output = execute_vyakti_command(config, &cmd_args).await?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": output
        }]
    }))
}

/// Handle vyakti_remove tool call
async fn handle_remove(
    config: &ServerConfig,
    args: &Value,
) -> Result<Value> {
    let name = args["name"].as_str()
        .context("Missing 'name' parameter")?;

    let cmd_args = vec![
        "remove".to_string(),
        name.to_string(),
        "-i".to_string(),
        config.index_dir.clone(),
        "--yes".to_string(),
    ];

    let output = execute_vyakti_command(config, &cmd_args).await?;

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("✓ Index '{}' removed successfully\n\n{}", name, output)
        }]
    }))
}

/// Handle MCP tools/call request
async fn handle_tools_call(
    config: &ServerConfig,
    id: Option<Value>,
    params: &Value,
) -> McpResponse {
    let tool_name = params["name"].as_str();
    let empty_args = json!({});
    let arguments = params.get("arguments").unwrap_or(&empty_args);

    let result = match tool_name {
        Some("vyakti_build") => handle_build(config, arguments).await,
        Some("vyakti_search") => handle_search(config, arguments).await,
        Some("vyakti_list") => handle_list(config).await,
        Some("vyakti_remove") => handle_remove(config, arguments).await,
        Some(name) => {
            return McpResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(McpError {
                    code: -32601,
                    message: format!("Unknown tool: {}", name),
                }),
            }
        }
        None => {
            return McpResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(McpError {
                    code: -32602,
                    message: "Missing tool name".to_string(),
                }),
            }
        }
    };

    match result {
        Ok(content) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(content),
            error: None,
        },
        Err(e) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {:#}", e)
                }],
                "isError": true
            })),
            error: None,
        },
    }
}

/// Process a single MCP request
async fn handle_request(config: &ServerConfig, request: McpRequest) -> McpResponse {
    tracing::debug!("Received request: method={}, id={:?}", request.method, request.id);

    match request.method.as_str() {
        "initialize" => handle_initialize(request.id),
        "tools/list" => handle_tools_list(request.id),
        "tools/call" => {
            let params = request.params.unwrap_or(json!({}));
            handle_tools_call(config, request.id, &params).await
        }
        method => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(McpError {
                code: -32601,
                message: format!("Unknown method: {}", method),
            }),
        },
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "vyakti_mcp=info".to_string())
        )
        .with_target(false)
        .with_writer(io::stderr)
        .init();

    tracing::info!("Vyakti MCP Server v{} starting...", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = ServerConfig::from_env();
    tracing::info!("Configuration:");
    tracing::info!("  VYAKTI_BIN: {}", config.vyakti_bin);
    tracing::info!("  INDEX_DIR: {}", config.index_dir);

    // Read JSONRPC requests from stdin
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.context("Failed to read line from stdin")?;

        if line.trim().is_empty() {
            continue;
        }

        tracing::debug!("Raw request: {}", line);

        // Parse request
        let request: McpRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                let error_response = McpResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(McpError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                    }),
                };

                let response_json = serde_json::to_string(&error_response)?;
                writeln!(stdout, "{}", response_json)?;
                stdout.flush()?;
                continue;
            }
        };

        // Handle request
        let response = handle_request(&config, request).await;

        // Send response
        let response_json = serde_json::to_string(&response)
            .context("Failed to serialize response")?;

        tracing::debug!("Response: {}", response_json);
        writeln!(stdout, "{}", response_json)?;
        stdout.flush()?;
    }

    tracing::info!("Vyakti MCP Server shutting down");
    Ok(())
}
