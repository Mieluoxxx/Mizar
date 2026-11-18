use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::LlmErrorCode;

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

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error,
        }
    }

    /// 提取可见文本内容。
    pub fn text_content(&self) -> String {
        match self {
            Self::System { content } | Self::User { content } | Self::Tool { content, .. } => {
                content.clone()
            }
            Self::Assistant { blocks, .. } => blocks
                .iter()
                .filter_map(ContentBlock::as_text)
                .collect::<Vec<_>>()
                .join(""),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<ReasoningEffort>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderInfo {
    pub id: String,
    pub name: String,
    pub protocol: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ModelInfo>,
}

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
    pub cancel: CancellationToken,
}
