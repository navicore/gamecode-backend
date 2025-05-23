use anyhow::Result;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// Status events that backends can emit during processing
#[derive(Debug, Clone)]
pub enum BackendStatus {
    /// Retry attempt being made
    RetryAttempt {
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
        reason: String,
    },
    /// Request is being rate limited
    RateLimited {
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
    },
    /// Non-retryable error occurred
    NonRetryableError {
        message: String,
    },
}

/// Callback function type for status updates
pub type StatusCallback = Arc<dyn Fn(BackendStatus) + Send + Sync>;

/// Core trait for LLM backend implementations
#[async_trait::async_trait]
pub trait LLMBackend: Send + Sync {
    /// Send a chat request and get a complete response
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Send a chat request and get a streaming response
    async fn chat_stream(&self, request: ChatRequest) -> Result<ChatStream>;

    /// Send a chat request with custom retry configuration
    async fn chat_with_retry(
        &self,
        request: ChatRequest,
        retry_config: RetryConfig,
    ) -> Result<ChatResponse>;

    /// Check if this backend supports tool calling
    fn supports_tools(&self) -> bool;

    /// Get list of supported model names
    fn supported_models(&self) -> Vec<String>;

    /// Get the default model for this backend
    fn default_model(&self) -> String;
}

/// Configuration for retry behavior
#[derive(Clone, Debug)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_delay: Duration,
    pub backoff_strategy: BackoffStrategy,
    pub verbose: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 10,
            initial_delay: Duration::from_millis(2000),
            backoff_strategy: BackoffStrategy::Exponential { multiplier: 3 },
            verbose: false,
        }
    }
}

/// Backoff strategy for retries
#[derive(Clone, Debug)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Exponential backoff with multiplier
    Exponential { multiplier: u32 },
    /// Linear increase in delay
    Linear { increment: Duration },
}

/// A chat request to send to an LLM backend
#[derive(Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub model: Option<String>, // If None, use backend default
    pub tools: Option<Vec<Tool>>,
    pub inference_config: Option<InferenceConfig>,
    pub session_id: Option<Uuid>,
    #[serde(skip)]
    pub status_callback: Option<StatusCallback>,
}

impl std::fmt::Debug for ChatRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatRequest")
            .field("messages", &self.messages)
            .field("model", &self.model)
            .field("tools", &self.tools)
            .field("inference_config", &self.inference_config)
            .field("session_id", &self.session_id)
            .field("status_callback", &self.status_callback.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

/// A complete response from an LLM backend
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
    pub model: String,
    pub session_id: Option<Uuid>,
}

/// A streaming response from an LLM backend
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatStreamEvent>> + Send>>;

/// Events in a streaming chat response
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChatStreamEvent {
    /// Start of the response
    Start { role: MessageRole },
    /// Text content delta
    TextDelta { text: String },
    /// Tool call start
    ToolCallStart { id: String, name: String },
    /// Tool call input delta
    ToolCallDelta { id: String, input: String },
    /// Tool call complete
    ToolCallEnd { id: String },
    /// End of the response
    End { usage: Option<Usage> },
}

/// A message in a conversation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text(text.into())],
        }
    }

    pub fn with_tool_calls(
        role: MessageRole,
        text: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        let mut content = vec![ContentBlock::Text(text.into())];
        content.extend(tool_calls.into_iter().map(ContentBlock::ToolCall));
        Self { role, content }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: tool_call_id.into(),
                result: result.into(),
            }],
        }
    }
}

/// Role of a message in conversation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Content blocks within a message
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContentBlock {
    Text(String),
    ToolCall(ToolCall),
    ToolResult {
        tool_call_id: String,
        result: String,
    },
}

/// A tool call made by the LLM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool specification for the LLM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Inference configuration parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(4096),
        }
    }
}

/// Token usage information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Errors that can occur in backend operations
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Model not supported: {model}")]
    UnsupportedModel { model: String },

    #[error("Rate limited by provider")]
    RateLimited,

    #[error("Request validation failed: {message}")]
    ValidationError { message: String },

    #[error("Authentication failed")]
    AuthenticationError,

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Provider error: {message}")]
    ProviderError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl BackendError {
    /// Check if this error should trigger a retry
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            BackendError::RateLimited | BackendError::NetworkError { .. }
        )
    }
}

/// Result type alias for backend operations
pub type BackendResult<T> = std::result::Result<T, BackendError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::text(MessageRole::User, "Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content.len(), 1);
    }

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 10);
        assert_eq!(config.initial_delay, Duration::from_millis(2000));
    }
}
