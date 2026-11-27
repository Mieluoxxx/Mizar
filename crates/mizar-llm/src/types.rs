use std::borrow::Cow;

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
    pub supports_images: Option<bool>,
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

impl GenerateOptions {
    pub fn message_for_model<'a>(&self, message: &'a Message) -> Cow<'a, Message> {
        if self.supports_images != Some(false) || message.images().is_empty() {
            return Cow::Borrowed(message);
        }
        let mut text = message.text_content();
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("[image omitted: current model does not support image input]");
        match message {
            Message::User { .. } => Cow::Owned(Message::user(text)),
            Message::Tool {
                tool_call_id,
                is_error,
                ..
            } => Cow::Owned(Message::tool(tool_call_id, text, *is_error)),
            _ => Cow::Borrowed(message),
        }
    }
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
