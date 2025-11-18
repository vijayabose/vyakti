//! RAG (Retrieval-Augmented Generation) chat module
//!
//! This module provides chat functionality that combines vector search with LLM generation.
//! Typical workflow:
//! 1. User asks a question
//! 2. System retrieves relevant documents via vector search
//! 3. System sends context + question to LLM for generation
//! 4. LLM generates response based on retrieved context

use crate::{VyaktiSearcher};
use vyakti_common::{GenerationConfig, TextGenerationProvider, Result, VyaktiError};
use std::sync::Arc;

/// Chat session that maintains conversation history
pub struct ChatSession {
    /// Document searcher for retrieval
    searcher: VyaktiSearcher,
    /// Text generation provider (LLM)
    generator: Arc<dyn TextGenerationProvider>,
    /// Number of documents to retrieve for context
    top_k: usize,
    /// Conversation history: (role, message)
    history: Vec<(String, String)>,
}

impl ChatSession {
    /// Create a new chat session
    pub fn new(
        searcher: VyaktiSearcher,
        generator: Arc<dyn TextGenerationProvider>,
        top_k: usize,
    ) -> Self {
        Self {
            searcher,
            generator,
            top_k,
            history: Vec::new(),
        }
    }

    /// Add a system message to the conversation history
    pub fn add_system_message(&mut self, message: String) {
        self.history.push(("system".to_string(), message));
    }

    /// Ask a question and get a generated response based on retrieved context
    pub async fn ask(&mut self, question: &str, config: &GenerationConfig) -> Result<String> {
        // 1. Retrieve relevant documents
        let search_results = self.searcher.search(question, self.top_k).await?;

        if search_results.is_empty() {
            return Err(VyaktiError::Generation(
                "No relevant documents found for the question".to_string(),
            ));
        }

        // 2. Build context from retrieved documents
        let context = search_results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                format!("[Document {}] (score: {:.4}):\n{}\n", i + 1, result.score, result.text)
            })
            .collect::<Vec<_>>()
            .join("\n");

        // 3. Create prompt with context and question
        let system_prompt = format!(
            "You are a helpful assistant. Answer the user's question based on the following context documents.\n\n\
            Context:\n{}\n\n\
            Instructions:\n\
            - Use only information from the provided context documents\n\
            - If the context doesn't contain relevant information, say so\n\
            - Be concise and accurate\n\
            - Cite document numbers when referencing specific information",
            context
        );

        // 4. Build conversation messages
        let mut messages = vec![("system".to_string(), system_prompt)];

        // Add conversation history
        messages.extend(self.history.iter().cloned());

        // Add current question
        messages.push(("user".to_string(), question.to_string()));

        // 5. Generate response using LLM
        let response = self.generator.chat(&messages, config).await?;

        // 6. Update history
        self.history.push(("user".to_string(), question.to_string()));
        self.history.push(("assistant".to_string(), response.clone()));

        Ok(response)
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get conversation history
    pub fn history(&self) -> &[(String, String)] {
        &self.history
    }
}

/// Simple one-shot question answering (no conversation history)
pub async fn ask_question(
    searcher: &VyaktiSearcher,
    generator: Arc<dyn TextGenerationProvider>,
    question: &str,
    top_k: usize,
    config: &GenerationConfig,
) -> Result<String> {
    // Retrieve relevant documents
    let search_results = searcher.search(question, top_k).await?;

    if search_results.is_empty() {
        return Err(VyaktiError::Generation(
            "No relevant documents found for the question".to_string(),
        ));
    }

    // Build context from retrieved documents
    let context = search_results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            format!("[Document {}] (score: {:.4}):\n{}\n", i + 1, result.score, result.text)
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Create prompt with context and question
    let prompt = format!(
        "Context:\n{}\n\n\
        Question: {}\n\n\
        Answer the question based only on the provided context. \
        Be concise and accurate. If the context doesn't contain relevant information, say so.",
        context, question
    );

    // Generate response
    generator.generate(&prompt, config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyakti_common::{GenerationConfig, TextGenerationProvider};
    use async_trait::async_trait;

    // Mock generator for testing
    struct MockGenerator;

    #[async_trait]
    impl TextGenerationProvider for MockGenerator {
        async fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<String> {
            Ok(format!("Generated response for: {}", prompt.chars().take(50).collect::<String>()))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_chat_session_creation() {
        // This test just verifies the struct can be created
        // Full integration tests would require a real searcher
    }

    #[test]
    fn test_chat_history() {
        // Test would verify conversation history management
    }
}
