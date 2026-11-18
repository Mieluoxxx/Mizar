//! 抽象 LLM 服务（对应 `NamiDSH/packages/llm`）。
//!
//! 定义提供方无关的统一消息、内容块、流式增量、Token 用量、错误分类、回放状态、
//! `BlockAssembler`、`LlmAdapter` 接口与 `ctx.llm` 服务；
//! 具体协议适配器（`completions` / `responses` / `messages` / `mock`）作为插件注册进来。

pub mod assembler;
pub mod error;
#[cfg(feature = "http")]
pub mod http;
pub mod runtime;
pub mod sse;
pub mod types;

pub use assembler::BlockAssembler;
pub use error::{LlmError, LlmErrorCode};
pub use runtime::{
    BoxChunkStream, EVENT_ADAPTERS_UPDATED, EVENT_LLM_STREAM, LlmAdapter, LlmContextExt, LlmRuntime,
};
pub use sse::{SseEvent, SseParser, prefix_digest};
pub use types::{
    ContentBlock, ContentBlockKind, FinishReason, GenerateOptions, Message, ModelInfo,
    ProviderInfo, ReasoningEffort, ReplayEnvelope, StreamChunk, TokenUsage, ToolCall, ToolSchema,
};
