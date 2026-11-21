pub use mizar_llm_types::{
    CacheRetention, ContentBlock, ContentBlockKind, FinishReason, ImageAttachment,
    MAX_IMAGES_PER_MESSAGE, Message, ModelInfo, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, SessionAffinityMode, StreamChunk, TokenUsage, ToolCall,
    ToolSchema, clamp_openai_prompt_cache_key, detect_session_affinity_mode,
    parse_session_affinity_mode, push_header_if_absent, push_session_affinity_headers,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Default)]
pub struct GenerateOptions {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSchema>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub headers: Vec<(String, String)>,
    pub session_id: Option<String>,
    pub cache_retention: Option<CacheRetention>,
    pub cancel: CancellationToken,
}

fn parse_cache_retention_str(raw: &str) -> Option<CacheRetention> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Some(CacheRetention::None),
        "short" => Some(CacheRetention::Short),
        "long" | "24h" => Some(CacheRetention::Long),
        _ => None,
    }
}

/// 解析有效缓存保留策略：调用显式值 > `MIZAR_CACHE_RETENTION` > `PI_CACHE_RETENTION` > 默认 `Short`。
pub fn resolve_cache_retention(explicit: Option<CacheRetention>) -> CacheRetention {
    if let Some(retention) = explicit {
        return retention;
    }
    for key in ["MIZAR_CACHE_RETENTION", "PI_CACHE_RETENTION"] {
        if let Ok(raw) = std::env::var(key)
            && let Some(parsed) = parse_cache_retention_str(&raw)
        {
            return parsed;
        }
    }
    CacheRetention::Short
}
