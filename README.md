# gamecode-backend

Backend trait and common types for LLM integrations in the gamecode ecosystem.

## Overview

This crate defines the core `LLMBackend` trait and common types used across all gamecode LLM backend implementations. It provides a unified interface for different LLM providers while allowing each backend to implement provider-specific optimizations like retry logic and error handling.

## Architecture

```
gamecode-cli (or other apps)
    ↓ uses
gamecode-backend (this crate - traits & types)
    ↓ implemented by
gamecode-bedrock, gamecode-openai, etc.
```

## Key Components

- **`LLMBackend` trait**: Core interface for all LLM backends
- **`ChatRequest`/`ChatResponse`**: Unified request/response types
- **`RetryConfig`**: Configurable retry behavior
- **`Tool`**: Tool calling support
- **`BackendError`**: Standardized error types

## Usage

Backends implement the `LLMBackend` trait:

```rust
use gamecode_backend::{LLMBackend, ChatRequest, ChatResponse};

#[async_trait::async_trait]
impl LLMBackend for MyBackend {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        // Implementation here
    }
    
    // ... other methods
}
```

Applications use backends through the trait:

```rust
use gamecode_backend::{LLMBackend, Message, MessageRole};

let backend: Box<dyn LLMBackend> = create_backend();
let request = ChatRequest {
    messages: vec![Message::text(MessageRole::User, "Hello!")],
    model: None, // Use backend default
    tools: None,
    inference_config: None,
    session_id: None,
};

let response = backend.chat(request).await?;
```

## Features

- **Unified Interface**: Same API across all LLM providers
- **Configurable Retries**: Built-in retry logic with customizable strategies
- **Tool Calling**: Standardized tool/function calling support
- **Streaming**: Support for streaming responses
- **Error Handling**: Comprehensive error types with retry guidance

## License

MIT