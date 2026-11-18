pub use dsh_llm_types::{
    ContentBlock, ContentBlockKind, FinishReason, Message, ModelInfo, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, StreamChunk, TokenUsage, ToolCall, ToolSchema,
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
    pub cancel: CancellationToken,
}
