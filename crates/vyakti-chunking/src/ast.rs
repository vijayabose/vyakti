//! AST-aware code chunking
//!
//! Provides code chunking that respects function, class, and module boundaries
//! using tree-sitter parsing.

use crate::{ChunkError, ChunkResult, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tree_sitter::{Language, Parser, Tree};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeLanguage {
    Python,
    Java,
    TypeScript,
    Rust,
    CSharp,
    #[cfg(feature = "ast-extended")]
    JavaScript,
    #[cfg(feature = "ast-extended")]
    Go,
    #[cfg(feature = "ast-extended")]
    C,
    #[cfg(feature = "ast-extended")]
    Cpp,
    #[cfg(feature = "ast-extended")]
    Swift,
    #[cfg(feature = "ast-extended")]
    Kotlin,
    #[cfg(feature = "ast-extended")]
    Ruby,
    #[cfg(feature = "ast-extended")]
    Php,
}

impl CodeLanguage {
    /// Get tree-sitter language for this code language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            CodeLanguage::Python => tree_sitter_python::language(),
            CodeLanguage::Java => tree_sitter_java::language(),
            CodeLanguage::TypeScript => tree_sitter_typescript::language_typescript(),
            CodeLanguage::Rust => tree_sitter_rust::language(),
            CodeLanguage::CSharp => tree_sitter_c_sharp::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::JavaScript => tree_sitter_javascript::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Go => tree_sitter_go::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::C => tree_sitter_c::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Cpp => tree_sitter_cpp::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Swift => tree_sitter_swift::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Kotlin => tree_sitter_kotlin::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Ruby => tree_sitter_ruby::language(),
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Php => tree_sitter_php::language(),
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "py" => Some(CodeLanguage::Python),
            "java" => Some(CodeLanguage::Java),
            "ts" | "tsx" => Some(CodeLanguage::TypeScript),
            "rs" => Some(CodeLanguage::Rust),
            "cs" => Some(CodeLanguage::CSharp),
            #[cfg(feature = "ast-extended")]
            "js" | "jsx" | "mjs" | "cjs" => Some(CodeLanguage::JavaScript),
            #[cfg(feature = "ast-extended")]
            "go" => Some(CodeLanguage::Go),
            #[cfg(feature = "ast-extended")]
            "c" | "h" => Some(CodeLanguage::C),
            #[cfg(feature = "ast-extended")]
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(CodeLanguage::Cpp),
            #[cfg(feature = "ast-extended")]
            "swift" => Some(CodeLanguage::Swift),
            #[cfg(feature = "ast-extended")]
            "kt" | "kts" => Some(CodeLanguage::Kotlin),
            #[cfg(feature = "ast-extended")]
            "rb" => Some(CodeLanguage::Ruby),
            #[cfg(feature = "ast-extended")]
            "php" => Some(CodeLanguage::Php),
            _ => None,
        }
    }

    /// Get top-level node types for this language
    fn top_level_nodes(&self) -> Vec<&'static str> {
        match self {
            CodeLanguage::Python => vec!["function_definition", "class_definition", "decorated_definition"],
            CodeLanguage::Java => vec!["method_declaration", "class_declaration", "interface_declaration"],
            CodeLanguage::TypeScript => vec!["function_declaration", "class_declaration", "interface_declaration", "method_definition"],
            CodeLanguage::Rust => vec!["function_item", "impl_item", "trait_item", "struct_item"],
            CodeLanguage::CSharp => vec!["method_declaration", "class_declaration", "interface_declaration"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::JavaScript => vec!["function_declaration", "class_declaration", "method_definition", "arrow_function"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Go => vec!["function_declaration", "method_declaration", "type_declaration"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::C => vec!["function_definition", "struct_specifier", "enum_specifier"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Cpp => vec!["function_definition", "class_specifier", "struct_specifier", "namespace_definition"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Swift => vec!["function_declaration", "class_declaration", "struct_declaration", "protocol_declaration"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Kotlin => vec!["function_declaration", "class_declaration", "object_declaration"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Ruby => vec!["method", "class", "module"],
            #[cfg(feature = "ast-extended")]
            CodeLanguage::Php => vec!["function_definition", "class_declaration", "method_declaration"],
        }
    }
}

/// AST-aware code chunker
pub struct AstChunker {
    language: CodeLanguage,
    parser: Parser,
    max_chunk_size: usize,
    chunk_overlap: usize,
}

impl AstChunker {
    /// Create a new AST chunker for the given language
    pub fn new(language: CodeLanguage, max_chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        let mut parser = Parser::new();
        parser
            .set_language(language.tree_sitter_language())
            .map_err(|e| ChunkError::AstError(format!("Failed to set language: {}", e)))?;

        Ok(Self {
            language,
            parser,
            max_chunk_size,
            chunk_overlap,
        })
    }

    /// Chunk code text using AST
    pub fn chunk(&mut self, code: &str) -> Result<Vec<ChunkResult>> {
        self.chunk_with_metadata(code, HashMap::new())
    }

    /// Chunk code with metadata
    pub fn chunk_with_metadata(
        &mut self,
        code: &str,
        base_metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<ChunkResult>> {
        // Parse code
        let tree = self
            .parser
            .parse(code, None)
            .ok_or_else(|| ChunkError::AstError("Failed to parse code".to_string()))?;

        // Extract top-level nodes
        let chunks = self.extract_chunks(&tree, code, base_metadata)?;

        Ok(chunks)
    }

    /// Extract chunks from parsed tree
    fn extract_chunks(
        &self,
        tree: &Tree,
        source: &str,
        base_metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<ChunkResult>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();
        let top_level_types = self.language.top_level_nodes();

        // Walk through children
        let mut cursor = root_node.walk();

        for child in root_node.children(&mut cursor) {
            let node_type = child.kind();

            // Check if this is a top-level node we care about
            if top_level_types.contains(&node_type) {
                let start_byte = child.start_byte();
                let end_byte = child.end_byte();
                let text = &source[start_byte..end_byte];

                // Check size
                if text.len() <= self.max_chunk_size {
                    // Single chunk
                    let mut metadata = base_metadata.clone();
                    metadata.insert("node_type".to_string(), serde_json::json!(node_type));
                    metadata.insert("chunk_index".to_string(), serde_json::json!(chunks.len()));

                    chunks.push(ChunkResult {
                        text: text.to_string(),
                        metadata,
                        token_count: text.len(), // For code, use character count
                    });
                } else {
                    // Split large nodes
                    let sub_chunks = self.split_large_node(text, node_type, &base_metadata, chunks.len());
                    chunks.extend(sub_chunks);
                }
            }
        }

        // If no top-level nodes found, fall back to character-based chunking
        if chunks.is_empty() {
            chunks = self.fallback_chunk(source, base_metadata);
        }

        Ok(chunks)
    }

    /// Split a large node into smaller chunks
    fn split_large_node(
        &self,
        text: &str,
        node_type: &str,
        base_metadata: &HashMap<String, serde_json::Value>,
        start_index: usize,
    ) -> Vec<ChunkResult> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut current_chunk = String::new();
        let mut chunk_line_count: usize = 0;

        for line in lines {
            if current_chunk.len() + line.len() > self.max_chunk_size && !current_chunk.is_empty() {
                // Save current chunk
                let mut metadata = base_metadata.clone();
                metadata.insert("node_type".to_string(), serde_json::json!(node_type));
                metadata.insert("chunk_index".to_string(), serde_json::json!(start_index + chunks.len()));
                metadata.insert("is_partial".to_string(), serde_json::json!(true));

                chunks.push(ChunkResult {
                    text: current_chunk.clone(),
                    metadata,
                    token_count: current_chunk.len(),
                });

                // Start new chunk with overlap
                if self.chunk_overlap > 0 {
                    let overlap_lines = chunk_line_count.saturating_sub(self.chunk_overlap / 80); // Rough estimate
                    let start_line = chunk_line_count.saturating_sub(overlap_lines);
                    current_chunk = current_chunk.lines().skip(start_line).collect::<Vec<_>>().join("\n");
                } else {
                    current_chunk.clear();
                }
                chunk_line_count = 0;
            }

            if !current_chunk.is_empty() {
                current_chunk.push('\n');
            }
            current_chunk.push_str(line);
            chunk_line_count += 1;
        }

        // Add final chunk
        if !current_chunk.is_empty() {
            let mut metadata = base_metadata.clone();
            metadata.insert("node_type".to_string(), serde_json::json!(node_type));
            metadata.insert("chunk_index".to_string(), serde_json::json!(start_index + chunks.len()));
            let token_count = current_chunk.len();

            chunks.push(ChunkResult {
                text: current_chunk,
                metadata,
                token_count,
            });
        }

        chunks
    }

    /// Fallback chunking when AST parsing fails
    fn fallback_chunk(
        &self,
        text: &str,
        base_metadata: HashMap<String, serde_json::Value>,
    ) -> Vec<ChunkResult> {
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + self.max_chunk_size).min(text.len());
            let chunk_text = &text[start..end];

            let mut metadata = base_metadata.clone();
            metadata.insert("chunk_index".to_string(), serde_json::json!(chunks.len()));
            metadata.insert("fallback".to_string(), serde_json::json!(true));

            chunks.push(ChunkResult {
                text: chunk_text.to_string(),
                metadata,
                token_count: chunk_text.len(),
            });

            start = end.saturating_sub(self.chunk_overlap);
            if start >= end {
                break;
            }
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(CodeLanguage::from_extension("py"), Some(CodeLanguage::Python));
        assert_eq!(CodeLanguage::from_extension("java"), Some(CodeLanguage::Java));
        assert_eq!(CodeLanguage::from_extension("ts"), Some(CodeLanguage::TypeScript));
        assert_eq!(CodeLanguage::from_extension("rs"), Some(CodeLanguage::Rust));
        assert_eq!(CodeLanguage::from_extension("cs"), Some(CodeLanguage::CSharp));
        assert_eq!(CodeLanguage::from_extension("txt"), None);
    }

    #[test]
    fn test_ast_chunker_python() {
        let code = r#"
def hello():
    print("Hello, world!")

def goodbye():
    print("Goodbye!")

class MyClass:
    def method(self):
        pass
"#;

        let mut chunker = AstChunker::new(CodeLanguage::Python, 512, 64).unwrap();
        let chunks = chunker.chunk(code).unwrap();

        // Should create separate chunks for functions and class
        assert!(chunks.len() >= 2);
        assert!(chunks[0].text.contains("def hello"));
    }

    #[test]
    fn test_ast_chunker_fallback() {
        let code = "x = 1\ny = 2\nz = 3";

        let mut chunker = AstChunker::new(CodeLanguage::Python, 10, 2).unwrap();
        let chunks = chunker.chunk(code).unwrap();

        // Should fall back to character-based chunking
        assert!(!chunks.is_empty());
    }
}
