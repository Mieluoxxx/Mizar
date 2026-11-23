use serde::{Deserialize, Serialize};

use super::error::LlmErrorCode;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON 字符串参数。
    pub arguments: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum ContentBlockKind {
    Text,
    Reasoning,
    ToolCall,
}

/// 助手回复中的结构化内容块（保留流式顺序）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        /// JSON 字符串参数。
        arguments: String,
    },
}

impl ContentBlock {
    pub fn kind(&self) -> ContentBlockKind {
        match self {
            Self::Text { .. } => ContentBlockKind::Text,
            Self::Reasoning { .. } => ContentBlockKind::Reasoning,
            Self::ToolCall { .. } => ContentBlockKind::ToolCall,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    pub fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Reasoning { text } => Some(text),
            _ => None,
        }
    }

    pub fn to_tool_call(&self) -> Option<ToolCall> {
        match self {
            Self::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            _ => None,
        }
    }
}

/// 跨协议切换与厂商原生回放状态（对上层不透明）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayEnvelope {
    /// 生成此回复的协议名，例如 `openai-completions` / `openai-responses` / `anthropic-messages`。
    pub protocol: String,
    /// 生成此回复的 provider 路由名。
    pub provider: String,
    /// 生成此回复的模型 ID。
    pub model: String,
    /// 响应级私有元数据（如 `response_id`、`prefix_digest`）。
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub response: serde_json::Value,
    /// 与 `ContentBlock` 一一按索引对齐的块级元数据（如 Anthropic thinking `signature`）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<serde_json::Value>,
}

impl ReplayEnvelope {
    /// 判断是否与目标协议/路由/模型一致，一致时方可安全恢复厂商原生签名与 ID。
    pub fn matches(&self, protocol: &str, provider: &str, model: &str) -> bool {
        self.protocol == protocol && self.provider == provider && self.model == model
    }

    /// 仅判断协议与模型是否一致（用于直接构造适配器、未经过命名字路由的场景）。
    pub fn matches_protocol_and_model(&self, protocol: &str, model: &str) -> bool {
        self.protocol == protocol && self.model == model
    }
}

/// 单条消息允许附带的图片上限（与 `dim` `MAX_IMAGES` 对齐，兼顾 SQLite 体积硬约束）。
pub const MAX_IMAGES_PER_MESSAGE: usize = 20;

/// 用户消息附带的图片负载（base64 编码 + MIME 类型及可选尺寸）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ImageAttachment {
    /// 标准 MIME 类型，如 `"image/png"` / `"image/jpeg"` / `"image/webp"` / `"image/gif"`。
    pub media_type: String,
    /// 不含 `data:` 前缀的纯 base64 编码字节。
    pub data: String,
    /// 最终像素宽度（用于上下文预算估算与回放展示）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// 最终像素高度。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

impl ImageAttachment {
    pub fn new(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            data: data.into(),
            width: None,
            height: None,
        }
    }

    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.data)
    }
}

/// OpenAI `prompt_cache_key` 最大字符数上限。
pub const OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH: usize = 64;

/// 上游 Prompt Cache 保留策略（对齐 Pi `cacheRetention`：`none` / `short` / `long`）。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

/// 模型请求来源归属，用于策略分流（如子代理跳过自动标题等）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RequestOrigin {
    SessionTurn {
        #[serde(rename = "sessionId")]
        session_id: String,
        turn: u64,
        step: usize,
    },
    Compaction,
    SessionTitle,
    Subagent,
    Custom(String),
}

impl Default for RequestOrigin {
    fn default() -> Self {
        Self::Custom(String::new())
    }
}

impl RequestOrigin {
    pub fn is_session_turn(&self) -> bool {
        matches!(self, Self::SessionTurn { .. })
    }
}

/// 模型 Prompt Cache 保留时长配置（单位：秒）。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "camelCase")]
pub struct PromptCacheLifetime {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub long: Option<u64>,
}

impl PromptCacheLifetime {
    pub fn default_retention(&self) -> Option<CacheRetention> {
        (self.long.is_some() && self.short.is_none()).then_some(CacheRetention::Long)
    }
}

/// 分段计费阶梯（当 prompt token 数严格大于 `input_tokens_above` 时生效，单位：美元 / 1M tokens）。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelCostTier {
    pub input_tokens_above: u64,
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// 模型单价元数据（单位：美元 / 1M tokens）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelCostRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

/// 会话亲和路由请求头格式。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SessionAffinityMode {
    #[default]
    Disabled,
    OpenAi,
    OpenRouter,
}

/// 按字符边界把 `prompt_cache_key` 截断到最多 64 字符，空白字符串返回 `None`。
pub fn clamp_openai_prompt_cache_key(key: &str) -> Option<String> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return None;
    }
    let out: String = trimmed
        .chars()
        .take(OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH)
        .collect();
    (!out.is_empty()).then_some(out)
}

/// 解析 `sessionAffinityFormat` 配置字符串。
pub fn parse_session_affinity_mode(raw: &str) -> Option<SessionAffinityMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" | "off" | "disabled" => Some(SessionAffinityMode::Disabled),
        "openai" | "responses" => Some(SessionAffinityMode::OpenAi),
        "openrouter" => Some(SessionAffinityMode::OpenRouter),
        _ => None,
    }
}

/// 根据 provider ID 与 base URL 推导默认会话亲和头模式（OpenRouter 走 `x-session-id`，其余走 OpenAI 亲和头）。
pub fn detect_session_affinity_mode(provider: &str, base_url: Option<&str>) -> SessionAffinityMode {
    if provider.trim().eq_ignore_ascii_case("openrouter")
        || base_url.is_some_and(|url| url.to_ascii_lowercase().contains("openrouter.ai"))
    {
        SessionAffinityMode::OpenRouter
    } else {
        SessionAffinityMode::OpenAi
    }
}

/// 当请求头切片中尚未出现同名头（大小写不敏感）时追加一条，避免覆盖显式配置。
pub fn push_header_if_absent(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
    if !headers
        .iter()
        .any(|(existing, _)| existing.eq_ignore_ascii_case(name))
    {
        headers.push((name.to_string(), value.to_string()));
    }
}

/// 按给定亲和模式向请求头注入会话粘性标识；`completions` 在 `OpenAi` 模式下额外携带 `x-session-affinity`。
pub fn push_session_affinity_headers(
    headers: &mut Vec<(String, String)>,
    mode: SessionAffinityMode,
    session_id: Option<&str>,
    include_completions_affinity_header: bool,
) {
    let Some(sid) = session_id.map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    match mode {
        SessionAffinityMode::Disabled => {}
        SessionAffinityMode::OpenRouter => {
            push_header_if_absent(headers, "x-session-id", sid);
        }
        SessionAffinityMode::OpenAi => {
            push_header_if_absent(headers, "session_id", sid);
            push_header_if_absent(headers, "x-client-request-id", sid);
            if include_completions_affinity_header {
                push_header_if_absent(headers, "x-session-affinity", sid);
            }
        }
    }
}

fn is_false(v: &bool) -> bool {
    !*v
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageAttachment>,
    },
    Assistant {
        #[serde(default)]
        blocks: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        replay: Option<ReplayEnvelope>,
    },
    Tool {
        content: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self::User {
            content: content.into(),
            images,
        }
    }

    pub fn assistant(blocks: Vec<ContentBlock>, replay: Option<ReplayEnvelope>) -> Self {
        Self::Assistant { blocks, replay }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let blocks = if text.is_empty() {
            Vec::new()
        } else {
            vec![ContentBlock::Text { text }]
        };
        Self::Assistant {
            blocks,
            replay: None,
        }
    }

    pub fn assistant_parts(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        let mut blocks = Vec::new();
        let text = text.into();
        if !text.is_empty() {
            blocks.push(ContentBlock::Text { text });
        }
        for call in tool_calls {
            blocks.push(ContentBlock::ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            });
        }
        Self::Assistant {
            blocks,
            replay: None,
        }
    }

    pub fn tool(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error,
        }
    }

    /// 提取可见文本内容。
    pub fn text_content(&self) -> String {
        match self {
            Self::System { content } | Self::User { content, .. } | Self::Tool { content, .. } => {
                content.clone()
            }
            Self::Assistant { blocks, .. } => blocks
                .iter()
                .filter_map(ContentBlock::as_text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// 提取用户消息附带的图片列表。
    pub fn images(&self) -> &[ImageAttachment] {
        match self {
            Self::User { images, .. } => images,
            _ => &[],
        }
    }

    /// 提取思考过程文本。
    pub fn reasoning_content(&self) -> String {
        match self {
            Self::Assistant { blocks, .. } => blocks
                .iter()
                .filter_map(ContentBlock::as_reasoning)
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    }

    /// 提取工具调用列表。
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        match self {
            Self::Assistant { blocks, .. } => blocks
                .iter()
                .filter_map(ContentBlock::to_tool_call)
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// 告诉模型「有哪些工具可以用」。`parameters` 是 JSON Schema。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl TokenUsage {
    /// 一次请求的完整 prompt 体积：未命中缓存的净输入 + 缓存读取 + 缓存写入。
    ///
    /// 三种协议解析时都把命中量从 `input_tokens` 里扣了出去（分别落在
    /// `cache_read_tokens` / `cache_write_tokens`），所以直接用 `input_tokens`
    /// 会严重低估真实 prompt 体积。
    pub fn prompt_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cache_read_tokens.unwrap_or(0))
            .saturating_add(self.cache_write_tokens.unwrap_or(0))
    }

    /// 一次请求四个分量的总和。
    ///
    /// 刻意不用 provider 上报的 `total_tokens`：各家口径不一致，且可以缺省；
    /// 四项相加在口径正确的 provider 上与其 `total_tokens` 恒等，口径缺失时仍然可算。
    pub fn grand_total(&self) -> u64 {
        self.prompt_tokens().saturating_add(self.output_tokens)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FinishReason {
    Stop,
    ToolCalls,
    MaxTokens,
    Aborted,
    Error {
        code: LlmErrorCode,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamChunk {
    BlockStart {
        index: usize,
        kind: ContentBlockKind,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ReasoningDelta {
        index: usize,
        text: String,
    },
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    BlockEnd {
        index: usize,
        block: ContentBlock,
    },
    Usage {
        usage: TokenUsage,
    },
    Finish {
        reason: FinishReason,
        replay: Option<ReplayEnvelope>,
    },
}

impl StreamChunk {
    pub fn text(text: impl Into<String>) -> Self {
        Self::TextDelta {
            index: 0,
            text: text.into(),
        }
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::ReasoningDelta {
            index: 0,
            text: text.into(),
        }
    }

    pub fn tool_call(index: usize, call: ToolCall) -> Self {
        Self::BlockEnd {
            index,
            block: ContentBlock::ToolCall {
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCostRates>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache: Option<PromptCacheLifetime>,
    #[serde(default)]
    pub force_adaptive_thinking: bool,
}

impl ModelInfo {
    /// 模型是否支持接收图片输入（未声明 `input` 能力列表时默认放行，显式声明且不含 `"image"` 时拒绝）。
    pub fn supports_images(&self) -> bool {
        self.input.is_empty() || self.input.iter().any(|k| k.eq_ignore_ascii_case("image"))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    pub protocol: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_tokens_folds_cache_read_and_write_back_into_input() {
        // 归一化后的典型 Anthropic usage：input 是扣掉缓存后的净额
        let usage = TokenUsage {
            input_tokens: 1_000,
            cache_read_tokens: Some(9_000),
            cache_write_tokens: Some(2_000),
            output_tokens: 500,
            ..Default::default()
        };
        assert_eq!(usage.prompt_tokens(), 12_000);
        assert_eq!(usage.grand_total(), 12_500);
    }

    #[test]
    fn prompt_tokens_without_cache_equals_input() {
        let usage = TokenUsage {
            input_tokens: 800,
            output_tokens: 200,
            ..Default::default()
        };
        assert_eq!(usage.prompt_tokens(), 800);
        assert_eq!(usage.grand_total(), 1_000);
    }

    #[test]
    fn grand_total_ignores_provider_reported_total_tokens() {
        // provider 报了个自相矛盾的 total，四项相加仍应是权威值
        let usage = TokenUsage {
            input_tokens: 100,
            cache_read_tokens: Some(300),
            cache_write_tokens: None,
            output_tokens: 50,
            total_tokens: Some(9_999),
            ..Default::default()
        };
        assert_eq!(usage.grand_total(), 450);
    }

    #[test]
    fn prompt_tokens_saturates_instead_of_overflowing() {
        let usage = TokenUsage {
            input_tokens: u64::MAX,
            cache_read_tokens: Some(1),
            cache_write_tokens: Some(1),
            output_tokens: 1,
            ..Default::default()
        };
        assert_eq!(usage.prompt_tokens(), u64::MAX);
        assert_eq!(usage.grand_total(), u64::MAX);
    }

    #[test]
    fn message_user_omits_empty_images_and_deserializes_legacy_json() {
        let plain = Message::user("hello");
        let json = serde_json::to_string(&plain).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hello"}"#);
        assert_eq!(serde_json::from_str::<Message>(&json).unwrap(), plain);

        let with_img = Message::user_with_images(
            "look",
            vec![ImageAttachment::new("image/png", "AQID").with_dimensions(10, 20)],
        );
        let img_json = serde_json::to_string(&with_img).unwrap();
        assert!(img_json.contains("\"images\""), "{img_json}");
        assert_eq!(
            serde_json::from_str::<Message>(&img_json).unwrap(),
            with_img
        );
    }

    #[test]
    fn model_info_supports_images_respects_declared_input() {
        let undeclared = ModelInfo {
            id: "m1".into(),
            name: "M1".into(),
            ..Default::default()
        };
        assert!(undeclared.supports_images());

        let text_only = ModelInfo {
            input: vec!["text".into()],
            ..undeclared.clone()
        };
        assert!(!text_only.supports_images());

        let vision = ModelInfo {
            input: vec!["text".into(), "image".into()],
            ..undeclared
        };
        assert!(vision.supports_images());
    }

    #[test]
    fn request_origin_default_helper_and_serde_roundtrip() {
        assert_eq!(
            RequestOrigin::default(),
            RequestOrigin::Custom(String::new())
        );
        assert!(!RequestOrigin::default().is_session_turn());

        let turn = RequestOrigin::SessionTurn {
            session_id: "sess-1".into(),
            turn: 3,
            step: 2,
        };
        assert!(turn.is_session_turn());

        for origin in [
            turn,
            RequestOrigin::Compaction,
            RequestOrigin::SessionTitle,
            RequestOrigin::Subagent,
            RequestOrigin::Custom("bridge".into()),
        ] {
            let json = serde_json::to_string(&origin).unwrap();
            let back: RequestOrigin = serde_json::from_str(&json).unwrap();
            assert_eq!(back, origin);
        }
    }

    #[test]
    fn model_info_metadata_dtos_roundtrip_and_default_from_legacy_json() {
        let legacy: ModelInfo = serde_json::from_str(r#"{"id":"gpt-4o","name":"GPT-4o"}"#).unwrap();
        assert_eq!(legacy.id, "gpt-4o");
        assert!(legacy.cost.is_none());
        assert!(legacy.prompt_cache.is_none());
        assert!(!legacy.force_adaptive_thinking);

        let full = ModelInfo {
            id: "claude-sonnet-5-5".into(),
            name: "Claude Sonnet 5.5".into(),
            context_window: Some(1_000_000),
            max_tokens: Some(128_000),
            reasoning_efforts: vec![ReasoningEffort::Medium, ReasoningEffort::High],
            input: vec!["text".into(), "image".into()],
            cost: Some(ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
                tiers: vec![ModelCostTier {
                    input_tokens_above: 200_000,
                    input: 6.0,
                    output: 22.5,
                    cache_read: 0.6,
                    cache_write: 7.5,
                }],
            }),
            prompt_cache: Some(PromptCacheLifetime {
                short: Some(300),
                long: Some(3600),
            }),
            force_adaptive_thinking: true,
        };
        let json = serde_json::to_string(&full).unwrap();
        let back: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back, full);

        assert_eq!(
            PromptCacheLifetime {
                short: None,
                long: Some(3600)
            }
            .default_retention(),
            Some(CacheRetention::Long)
        );
        assert_eq!(
            PromptCacheLifetime {
                short: Some(300),
                long: Some(3600)
            }
            .default_retention(),
            None
        );
    }
}
