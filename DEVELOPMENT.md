# Vyakti Development Setup

## Prerequisites

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install system dependencies (macOS)
brew install llvm libomp cmake

# Install system dependencies (Ubuntu)
sudo apt-get install build-essential cmake clang libomp-dev pkg-config
```

## Building

```bash
# Build all crates
cargo build --workspace

# Build in release mode
cargo build --release --workspace

# Check for errors without building
cargo check --workspace
```

## Running

```bash
# Run CLI
cargo run --bin vyakti -- --help

# Build an index (not yet implemented)
cargo run --bin vyakti -- build my-index --input ./documents

# Search (not yet implemented)
cargo run --bin vyakti -- search my-index "query"
```

## Testing

```bash
# Run all tests
cargo test --workspace

# Run tests for specific crate
cargo test --package vyakti-core

# Run with output
cargo test --workspace -- --nocapture
```

## Project Structure

```
vyakti/
├── Cargo.toml              # Workspace configuration
├── README.md               # Main project README
├── CLAUDE.md               # Development guide for Claude Code
├── document/
│   ├── BRD.md             # Business Requirements Document
│   └── MODULAR_DESIGN.md  # Architecture documentation
├── LEANN/                  # Original Python implementation (reference)
└── crates/
    ├── vyakti-common/      # Common types, traits, errors
    ├── vyakti-storage/     # Storage layer (CSR, index format)
    ├── vyakti-embedding/   # Embedding computation
    ├── vyakti-core/        # Main API (Builder, Searcher)
    ├── vyakti-backend-hnsw/    # HNSW backend
    ├── vyakti-backend-diskann/ # DiskANN backend
    ├── vyakti-proto/       # Protocol buffers
    ├── vyakti-server/      # REST + gRPC server
    └── vyakti-cli/         # CLI tool
```

## Development Status

🚧 **Phase 1: Foundation (In Progress)**

- [x] Project structure and workspace setup
- [x] Core types and traits defined
- [x] Basic module skeletons created
- [ ] CSR graph implementation
- [ ] HNSW backend implementation
- [ ] Embedding provider implementation
- [ ] CLI commands implementation

See `document/BRD.md` for detailed roadmap and milestones.

## Quick Start Guide

For full documentation, see the main [README.md](./README.md).

For development guidelines, see [CLAUDE.md](./CLAUDE.md).
