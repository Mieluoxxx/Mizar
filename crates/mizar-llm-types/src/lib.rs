//! 提供方无关的 LLM 核心数据结构、错误分类、增量组装器与 SSE 解析器。
//!
//! 本 crate 不依赖 `cordis`、`tokio` 或任何 HTTP/TLS 传输库，供纯 UI 层与核心领域层共用。

pub mod assembler;
pub mod error;
pub mod sse;
pub mod types;

pub use assembler::BlockAssembler;
pub use error::{
    HttpErrorHints, LlmError, LlmErrorCode, classify_inband_error_value,
    classify_non_json_sse_frame, is_context_window_exceeded, is_empty_response_error,
    is_quota_exceeded,
};
pub use sse::{SseEvent, SseParser, prefix_digest};
pub use types::{
    CacheRetention, ContentBlock, ContentBlockKind, FinishReason, ImageAttachment,
    MAX_IMAGES_PER_MESSAGE, Message, ModelInfo, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, SessionAffinityMode, StreamChunk, TokenUsage, ToolCall,
    ToolSchema, clamp_openai_prompt_cache_key, detect_session_affinity_mode,
    parse_session_affinity_mode, push_header_if_absent, push_session_affinity_headers,
};
