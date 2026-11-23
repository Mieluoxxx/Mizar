//! 抽象 LLM 服务、HTTP/SSE 传输、协议适配器、声明式路由与流式重试。
//!
//! 合并自 `mizar-llm`、`mizar-llm-http`、`mizar-llm-completions`、`mizar-llm-messages`、
//! `mizar-llm-responses`、`mizar-llm-router`、`mizar-llm-retry` 与 `mizar-llm-mock`。

pub mod completions;
pub mod http;
pub mod messages;
pub mod mock;
pub mod responses;
pub mod retry;
pub mod router;
pub mod runtime;
pub mod types;

pub use mizar_types::llm::{
    BlockAssembler, CacheMissAudit, CacheRetention, ContentBlock, ContentBlockKind, FinishReason,
    ImageAttachment, LlmError, LlmErrorCode, MAX_IMAGES_PER_MESSAGE, MIN_CACHE_DROP_TOKENS,
    Message, ModelCostRates, ModelCostTier, ModelInfo, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH,
    PromptCacheLifetime, ProviderInfo, ReasoningEffort, ReplayEnvelope, RequestOrigin,
    SessionAffinityMode, SseEvent, SseParser, StreamChunk, TokenUsage, ToolCall, ToolSchema,
    UsageAuditSample, assembler, cache_audit, calculate_usage_cost, clamp_openai_prompt_cache_key,
    compute_cache_miss_audit, detect_session_affinity_mode, error, is_context_window_exceeded,
    is_empty_response_error, is_quota_exceeded, parse_session_affinity_mode, prefix_digest,
    push_header_if_absent, push_session_affinity_headers, sse,
};

pub use completions::{
    CompletionsAdapter, CompletionsCompat, CompletionsConfig, CompletionsPlugin, MaxTokensField,
    PROTOCOL_COMPLETIONS, ThinkingFormat,
};
pub use messages::{
    DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS, MessagesAdapter, MessagesConfig, MessagesPlugin,
    PROTOCOL_MESSAGES,
};
pub use mock::{MockAdapter, MockPlugin};
pub use responses::{PROTOCOL_RESPONSES, ResponsesAdapter, ResponsesConfig, ResponsesPlugin};
pub use retry::{LlmRetryPlugin, MAX_RETRY_AFTER_DELAY, RetryConfig};
pub use router::{
    CompatConfig, EVENT_ROUTER_HEADERS, LlmRouter, LlmRouterContextExt, ModelCostConfig,
    ModelCostTierConfig, ModelOverrideConfig, PromptCacheConfig, ProtocolKind, ProviderRouteConfig,
    RouterConfig, RouterHeaderInput, RouterModelConfig, parse_protocol_api, resolve_credential,
};
pub use runtime::{
    BoxChunkStream, EVENT_ADAPTERS_UPDATED, EVENT_LLM_AFTER_RESPONSE, EVENT_LLM_STREAM, LlmAdapter,
    LlmAfterResponseEvent, LlmContextExt, LlmRuntime,
};
pub use types::{GenerateOptions, resolve_cache_retention};
