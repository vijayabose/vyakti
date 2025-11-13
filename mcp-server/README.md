# Vyakti MCP Server for Claude Code

A native Rust implementation of the Model Context Protocol (MCP) server for Vyakti, enabling seamless integration with Claude Code for semantic code search and document retrieval.

## Features

- **🦀 Native Rust** - Fast, memory-safe MCP server implementation
- **🔍 Semantic Search** - Natural language queries across code and documents
- **📦 35+ File Formats** - Supports text, markdown, JSON, YAML, PDF, Office docs, code files
- **🚀 Compact Mode** - 93% storage savings with LEANN compression
- **🎯 AST-Aware** - Intelligent code chunking for 13+ programming languages
- **🔧 Zero Config** - Works out of the box with sensible defaults

## Prerequisites

1. **Vyakti CLI installed**:
   ```bash
   cargo install --path crates/vyakti-cli
   ```

2. **Ollama running** (for embedding generation):
   ```bash
   # Install Ollama
   brew install ollama  # macOS
   # or download from https://ollama.com

   # Start Ollama server
   ollama serve
   ```

3. **Claude Code** installed from [https://claude.com/code](https://claude.com/code)

## Installation

### Step 1: Build the MCP Server

```bash
cd /Users/vijay/01-all-my-code-repos/vyakti
cargo build --release -p vyakti-mcp
```

The binary will be at: `target/release/vyakti-mcp`

### Step 2: Configure Claude Code

Add the Vyakti MCP server to your Claude Code configuration:

**File:** `~/.claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vyakti": {
      "command": "/Users/vijay/01-all-my-code-repos/vyakti/target/release/vyakti-mcp",
      "env": {
        "INDEX_DIR": "/Users/vijay/.vyakti",
        "VYAKTI_BIN": "/Users/vijay/.cargo/bin/vyakti",
        "RUST_LOG": "vyakti_mcp=info"
      }
    }
  }
}
```

**Configuration Options:**
- `INDEX_DIR` - Where indexes are stored (default: `.vyakti`)
- `VYAKTI_BIN` - Path to vyakti CLI binary (default: `vyakti` from PATH)
- `RUST_LOG` - Logging level (`error`, `warn`, `info`, `debug`, `trace`)

### Step 3: Restart Claude Code

After adding the configuration, restart the Claude Code/Claude Desktop application to load the MCP server.

### Step 4: Verify Installation

In Claude Code, ask:
- "List my Vyakti indexes"
- "Build a Vyakti index called 'test' from ./documents"

If you see tool prompts from `vyakti_list` or `vyakti_build`, the integration is working!

## Available Tools

### 1. `vyakti_build`

Build a vector search index from documents.

**Parameters:**
- `name` (required) - Index name (e.g., 'my-docs', 'codebase')
- `input_path` (required) - Path to file or directory
- `enable_code_chunking` (optional) - Enable AST-aware code chunking (default: false)
- `compact` (optional) - Enable compact mode for 93% storage savings (default: false)

**Example in Claude:**
> "Build a Vyakti index called 'my-project' from the current directory with code chunking enabled"

### 2. `vyakti_search`

Search a vector index using semantic similarity.

**Parameters:**
- `name` (required) - Index name to search
- `query` (required) - Search query (natural language or keywords)
- `top_k` (optional) - Number of results (default: 10, max: 50)
- `max_score` (optional) - Max score threshold (lower = more relevant)

**Example in Claude:**
> "Search the 'my-project' index for 'authentication logic' and show me the top 5 results"

**Score Interpretation:**
- Score < 0.3: Highly relevant
- Score 0.3-0.7: Moderately relevant
- Score > 0.7: Weakly relevant

### 3. `vyakti_list`

List all available Vyakti indexes.

**Example in Claude:**
> "What Vyakti indexes are available?"

### 4. `vyakti_remove`

Remove a vector index permanently.

**Parameters:**
- `name` (required) - Index name to remove

**Example in Claude:**
> "Remove the 'old-docs' Vyakti index"

## Usage Examples

### Index Your Codebase

**Claude Chat:**
```
You: "Index my entire codebase with Vyakti using code chunking and compact mode"

Claude: [Calls vyakti_build with enable_code_chunking: true, compact: true]
"✓ Index 'codebase' built successfully
  → Indexed 487 documents into 2,341 chunks
  → Storage savings: 93.2%"
```

### Semantic Code Search

**Claude Chat:**
```
You: "Find all code related to user authentication in the codebase index"

Claude: [Calls vyakti_search with query: "user authentication"]
"Top 10 results (found in 0.153s):
1. (score: 0.12) auth/login.rs:45 - Authentication middleware implementation
2. (score: 0.28) api/users.rs:123 - User login endpoint
..."
```

### Document Q&A

**Claude Chat:**
```
You: "Build an index of my documentation folder, then search for
      information about API rate limiting"

Claude:
[1. Calls vyakti_build for docs]
[2. Calls vyakti_search with query: "API rate limiting"]
[3. Synthesizes answer from search results]

"Based on the documentation, your API uses the following rate
limiting strategy: [synthesized answer from search results]"
```

## Supported File Formats

Vyakti supports **35+ file extensions**:

**Text & Configuration:**
- .txt, .md, .markdown
- .json, .yaml, .yml, .toml
- .csv, .html, .htm

**Documents:**
- .pdf - PDF documents
- .ipynb - Jupyter notebooks
- .docx - Microsoft Word
- .xlsx - Microsoft Excel
- .pptx - Microsoft PowerPoint

**Code (with --enable-code-chunking):**
- Python, Rust, Java, TypeScript, C#
- JavaScript, Go, C, C++
- Swift, Kotlin, Ruby, PHP

**Total:** 35+ file extensions with intelligent text extraction and AST-aware code chunking.

## Workflow Integration

### 1. One-Time Setup

```bash
# Build indexes for your projects
vyakti build my-project --input ./src --enable-code-chunking --compact
vyakti build my-docs --input ./docs --compact
vyakti build my-configs --input . --no-chunking

# List indexes
vyakti list
```

### 2. Use with Claude Code

Now when working in Claude Code, you can ask:

- **Code Understanding:**
  - "How does authentication work in this codebase?"
  - "Find all database connection setup code"
  - "Show me error handling patterns"

- **Documentation Search:**
  - "What does the API documentation say about rate limiting?"
  - "Find configuration examples for deployment"

- **Refactoring Help:**
  - "Search for all uses of the old User model"
  - "Find deprecated API endpoints"

## Architecture

```
Claude Code
    ↓
MCP Protocol (JSONRPC over stdio)
    ↓
vyakti-mcp (Rust binary)
    ↓
vyakti CLI (subprocess)
    ↓
Ollama (Embeddings) + HNSW/DiskANN (Search)
```

**Key Components:**
- **vyakti-mcp** - MCP server handling JSONRPC requests
- **strip-ansi-escapes** - Removes color codes from CLI output
- **tokio** - Async command execution
- **serde_json** - JSON serialization/deserialization

## Troubleshooting

### "Command not found: vyakti"

Ensure Vyakti CLI is installed and in PATH:
```bash
which vyakti
# Should output: /Users/vijay/.cargo/bin/vyakti

# If not found, install:
cargo install --path crates/vyakti-cli
```

Update `VYAKTI_BIN` in `claude_desktop_config.json` if needed.

### "Ollama connection failed"

Start Ollama server:
```bash
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "Index not found"

List available indexes:
```bash
vyakti list

# Or check index directory
ls ~/.vyakti/
```

### "Permission denied" on MCP binary

Make sure the binary is executable:
```bash
chmod +x /Users/vijay/01-all-my-code-repos/vyakti/target/release/vyakti-mcp
```

### Enable Debug Logging

Set `RUST_LOG=debug` in the MCP configuration to see detailed logs:

```json
{
  "mcpServers": {
    "vyakti": {
      "command": "...",
      "env": {
        "RUST_LOG": "vyakti_mcp=debug"
      }
    }
  }
}
```

Logs are written to stderr and visible in Claude Code's MCP server logs.

## Performance Tips

1. **Use Compact Mode** - 93% storage savings with minimal quality loss
2. **Enable Code Chunking** - Better semantic boundaries for code files
3. **Filter by Score** - Use `max_score` to get only highly relevant results (< 0.3)
4. **Build Once** - Indexes are persistent, no need to rebuild frequently
5. **Batch Queries** - Search returns top-k results efficiently

## Comparison with Python LEANN MCP

| Feature | Vyakti (Rust) | LEANN (Python) |
|---------|---------------|----------------|
| **Language** | Rust | Python |
| **Tools** | 4 (build, search, list, remove) | 2 (search, list) |
| **Color Handling** | Native strip-ansi-escapes | Manual --non-interactive flag |
| **Performance** | Native binary, no interpreter | Python startup overhead |
| **Memory Safety** | Rust ownership model | Python GC |
| **Error Messages** | Rich context with anyhow | Standard Python exceptions |

## Development

### Build from Source

```bash
cd /Users/vijay/01-all-my-code-repos/vyakti
cargo build --release -p vyakti-mcp
```

### Run Tests

```bash
cargo test -p vyakti-mcp
```

### Manual Testing

```bash
# Test tools/list
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | \
  ./target/release/vyakti-mcp

# Test vyakti_list
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "vyakti_list", "arguments": {}}}' | \
  ./target/release/vyakti-mcp
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Related Documentation

- [Vyakti README](../README.md) - Main project documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/) - MCP protocol details
- [Claude Code Docs](https://docs.claude.com/claude-code) - Claude Code documentation
- [Ollama](https://ollama.com) - Local embedding model provider

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/vyakti/issues)
- 📖 Documentation: [README.md](../README.md)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/vyakti/discussions)
