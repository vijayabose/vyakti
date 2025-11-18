# llama_cpp-rs Integration Plan

## Objective
Replace Ollama with llama_cpp-rs as the default embedding provider for Vyakti, with automatic model downloading for mxbai-embed-large.

## Current State
- **Build Status**: Broken (missing async-trait dependency)
- **Current Default**: Ollama (not implemented yet)
- **Target Model**: mxbai-embed-large (1024 dimensions)
- **Provider System**: Trait-based plugin architecture

## Implementation Phases

### Phase 1: Fix Build Issues & Setup Dependencies
**Status**: Pending

Tasks:
- [ ] Add `async-trait` dependency to `vyakti-embedding/Cargo.toml`
- [ ] Add `llama-cpp-rs` dependency (~0.3)
- [ ] Add `hf-hub` for HuggingFace downloads (~0.3)
- [ ] Add `sha2` for checksum verification (~0.10)
- [ ] Update workspace dependencies in root `Cargo.toml`

### Phase 2: Implement llama_cpp-rs Provider
**Status**: Pending

Files to Create:
- `crates/vyakti-embedding/src/providers/llama_cpp.rs`
- `crates/vyakti-embedding/src/providers/mod.rs`

Implementation Details:
```rust
pub struct LlamaCppProvider {
    model: Arc<Mutex<LlamaModel>>,
    dimension: usize,
    config: LlamaCppConfig,
}

pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub n_threads: u32,
}
```

Features:
- Thread-safe model access with Arc<Mutex<>>
- Batch embedding support
- GPU acceleration support
- Proper error handling

### Phase 3: Model Download System
**Status**: Pending

Files to Create:
- `crates/vyakti-embedding/src/download.rs`

Model Details:
- **Model**: mixedbread-ai/mxbai-embed-large-v1
- **Format**: GGUF (Q4_K_M quantization)
- **File**: mxbai-embed-large-v1-q4_k_m.gguf
- **Size**: ~500MB
- **HuggingFace URL**: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1-GGUF
- **Storage**: ~/.vyakti/models/

Features:
- Automatic download on first use
- Progress indicators
- Checksum verification
- Resume capability for partial downloads
- Error handling and retries

### Phase 4: CLI Integration
**Status**: Pending

Files to Modify:
- `crates/vyakti-cli/src/main.rs`

Changes:
- Replace `OllamaProvider` with `LlamaCppProvider`
- Replace `OllamaConfig` with `LlamaCppConfig`
- Add model auto-download before provider initialization
- Add CLI flags:
  - `--gpu-layers` (default: 0 for CPU, auto-detect for GPU)
  - `--model-threads` (default: num_cpus)
  - `--model-path` (override default model location)

### Phase 5: Documentation Updates
**Status**: Pending

Files to Update:
1. **README.md**
   - Replace Ollama references with llama_cpp-rs
   - Update embedding models section (line 383-386)
   - Update quick start examples
   - Add model download information
   - Update configuration examples (line 448-452)

2. **CLAUDE.md**
   - Update embedding computation section
   - Document llama_cpp-rs as default
   - Add troubleshooting for llama.cpp issues
   - Update development commands

3. **document/MODULAR_DESIGN.md**
   - Replace `ollama.rs` with `llama_cpp.rs` (line 104)
   - Update embedding provider architecture
   - Document LlamaCppProvider design

### Phase 6: Testing & Validation
**Status**: Pending

Files to Create:
- `crates/vyakti-embedding/tests/test_llama_cpp.rs`

Tests:
- Unit tests for LlamaCppProvider
- Model download functionality
- Embedding generation (verify 1024 dimensions)
- Batch processing
- Error handling

Validation:
- [ ] `cargo build --workspace` succeeds
- [ ] `cargo test --workspace` passes
- [ ] Model downloads correctly on first run
- [ ] Embeddings generate with correct dimensions
- [ ] CLI build/search commands work
- [ ] Performance benchmarks

## Technical Specifications

### Default Model Configuration
```toml
[embedding]
provider = "llama-cpp"
model = "mxbai-embed-large-v1"
model_file = "mxbai-embed-large-v1-q4_k_m.gguf"
dimension = 1024
gpu_layers = 0  # CPU-only by default
threads = "auto"  # Detect CPU count
```

### Model Download Flow
1. Check if model exists at `~/.vyakti/models/mxbai-embed-large-v1-q4_k_m.gguf`
2. If not found, download from HuggingFace Hub:
   - Repository: mixedbread-ai/mxbai-embed-large-v1-GGUF
   - File: mxbai-embed-large-v1-q4_k_m.gguf
3. Show progress bar during download
4. Verify checksum after download
5. Initialize LlamaCppProvider with model path

### Dependencies to Add

**Workspace Level** (`Cargo.toml`):
```toml
llama-cpp-rs = "0.3"
hf-hub = "0.3"
sha2 = "0.10"
dirs = "5.0"
```

**vyakti-embedding** (`crates/vyakti-embedding/Cargo.toml`):
```toml
[dependencies]
vyakti-common = { path = "../vyakti-common" }
tokio = { workspace = true }
async-trait = "0.1"
llama-cpp-rs = { workspace = true }
hf-hub = { workspace = true }
sha2 = { workspace = true }
dirs = { workspace = true }
tracing = { workspace = true }
anyhow = { workspace = true }
parking_lot = { workspace = true }
```

## Files to Create/Modify

### New Files
- `crates/vyakti-embedding/src/providers/llama_cpp.rs` (LlamaCppProvider implementation)
- `crates/vyakti-embedding/src/providers/mod.rs` (Provider module exports)
- `crates/vyakti-embedding/src/download.rs` (Model download utilities)
- `crates/vyakti-embedding/tests/test_llama_cpp.rs` (Tests)

### Modified Files
- `Cargo.toml` (workspace dependencies)
- `crates/vyakti-embedding/Cargo.toml` (crate dependencies)
- `crates/vyakti-embedding/src/lib.rs` (exports)
- `crates/vyakti-embedding/src/providers.rs` (remove/update mock provider)
- `crates/vyakti-cli/src/main.rs` (replace Ollama with LlamaCpp)
- `README.md` (documentation)
- `CLAUDE.md` (development guide)
- `document/MODULAR_DESIGN.md` (architecture)

## Success Criteria

✅ **Build Success**: `cargo build --workspace` completes without errors
✅ **Tests Pass**: `cargo test --workspace` all tests pass
✅ **Model Download**: Automatic download works on fresh install
✅ **Embeddings Work**: Generate 1024-dimensional vectors correctly
✅ **CLI Functional**: `vyakti build` and `vyakti search` commands work
✅ **Documentation**: All docs accurately reflect llama_cpp-rs as default
✅ **Performance**: Embeddings generate in reasonable time (<100ms per batch)

## Rollback Plan

If issues arise:
1. Keep Ollama references in comments for reference
2. Add feature flags: `default = ["llama-cpp"]`, optional `ollama` feature
3. Document known issues and workarounds

## Timeline

Estimated: 2-3 hours for complete implementation and testing

## Notes

- llama.cpp requires C++ compiler (clang/gcc) - document in README
- GPU support optional but recommended for production
- Model quantization (Q4_K_M) balances quality and size
- Consider adding model selection in future (different sizes/languages)

## References

- llama-cpp-rs: https://github.com/mdrokz/rust-llama.cpp
- mxbai-embed-large: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
- GGUF models: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1-GGUF
