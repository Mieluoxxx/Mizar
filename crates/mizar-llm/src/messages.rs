//! Anthropic Messages 协议适配器。
//!
//! 支持 Anthropic 官方 API（`https://api.anthropic.com`）以及兼容 Messages 协议的网关
//! （如 DeepSeek Anthropic 兼容端点 `https://api.deepseek.com/anthropic`）。
//! 自动处理：
//! - `System` 抽取至顶层 `system` 字段
//! - 连续 `Tool` 结果折叠进 `role: "user"` 的 `tool_result` 内容块
//! - 流式 `message_start` / `content_block_*` / `message_delta` / `message_stop` 状态机
//! - 思考签名（`signature`）与前缀摘要（`prefixDigest`）绑定，跨模型切换安全降级

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use crate::http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, ensure_v1_root,
    resolve_credential, shared_http_client, spawn_sse_stream,
};
use crate::{
    BoxChunkStream, CacheRetention, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions,
    LlmAdapter, LlmContextExt, LlmError, LlmErrorCode, Message, ModelInfo, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, SseEvent, StreamChunk, TokenUsage, resolve_cache_retention,
};
use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use serde_json::{Value, json};

pub const PROTOCOL_MESSAGES: &str = "anthropic-messages";
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Debug, Clone)]
pub struct MessagesConfig {
    pub provider: String,
    pub display_name: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: String,
    pub model: String,
    pub default_max_tokens: u32,
    pub anthropic_version: String,
    /// 为 `true` 时使用 `Authorization: Bearer <key>`，否则使用原生 `x-api-key`。
    pub use_bearer_auth: bool,
    /// 为 `true` 时发送自适应思考（`thinking.type = "adaptive"` + `output_config.effort`）。
    pub force_adaptive_thinking: bool,
    pub thinking_budget_tokens: Option<u32>,
    pub headers: Vec<(String, String)>,
    pub models: Vec<ModelInfo>,
    pub idle_timeout: Duration,
}

impl Default for MessagesConfig {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            display_name: Some("Anthropic".to_string()),
            api_key: None,
            api_key_env: None,
            base_url: "https://api.anthropic.com".to_string(),
            model: "claude-sonnet-4-5".to_string(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
            anthropic_version: DEFAULT_ANTHROPIC_VERSION.to_string(),
            use_bearer_auth: false,
            force_adaptive_thinking: false,
            thinking_budget_tokens: None,
            headers: Vec::new(),
            models: Vec::new(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

const FNV1A_OFFSET: u128 = 0x6c62272e07bb0142_62b821756295c58d;
const FNV1A_PRIME: u128 = 0x0000000001000000_000000000000013b;

/// 增量流式 128-bit FNV-1a 前缀摘要器：
/// 先一次性喂入 `system` 与 `tools`，再按消息定型顺序增量追加，避免每次 O(M²×S) 克隆与整树序列化。
#[derive(Clone)]
struct StreamingPrefixHasher {
    state: u128,
    message_count: usize,
}

impl StreamingPrefixHasher {
    fn new(system_text: &str, wire_tools: &[Value]) -> Self {
        let mut hasher = Self {
            state: FNV1A_OFFSET,
            message_count: 0,
        };
        let _ = hasher.write_all(br#"{"system":"#);
        let _ = serde_json::to_writer(&mut hasher, system_text);
        let _ = hasher.write_all(br#","tools":"#);
        let _ = serde_json::to_writer(&mut hasher, wire_tools);
        let _ = hasher.write_all(br#","messages":["#);
        hasher
    }

    fn push_message(&mut self, msg: &Value) {
        if self.message_count > 0 {
            let _ = self.write_all(b",");
        }
        let _ = serde_json::to_writer(&mut *self, msg);
        self.message_count += 1;
    }

    fn snapshot_digest(&self) -> String {
        let mut snap = self.clone();
        let _ = snap.write_all(b"]}");
        format!("{:032x}", snap.state)
    }
}

impl Write for StreamingPrefixHasher {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for &byte in buf {
            self.state ^= u128::from(byte);
            self.state = self.state.wrapping_mul(FNV1A_PRIME);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn push_user_block(wire_messages: &mut Vec<Value>, block: Value) {
    if let Some(last) = wire_messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
        && let Some(arr) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        arr.push(block);
        return;
    }
    wire_messages.push(json!({
        "role": "user",
        "content": [block],
    }));
}

fn parse_tool_input_json(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    }
}

fn thinking_budget(effort: ReasoningEffort, override_budget: Option<u32>) -> Option<u32> {
    if effort == ReasoningEffort::Off {
        return None;
    }
    if let Some(b) = override_budget {
        return Some(b.max(1024));
    }
    Some(match effort {
        ReasoningEffort::Off => return None,
        ReasoningEffort::Minimal => 1024,
        ReasoningEffort::Low => 2048,
        ReasoningEffort::Medium => 4096,
        ReasoningEffort::High => 8192,
        ReasoningEffort::XHigh | ReasoningEffort::Max => 16384,
    })
}

fn cache_control_value(retention: CacheRetention) -> Option<Value> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(json!({ "type": "ephemeral" })),
        CacheRetention::Long => Some(json!({ "type": "ephemeral", "ttl": "1h" })),
    }
}

fn attach_cache_control(
    wire_tools: &mut [Value],
    wire_messages: &mut [Value],
    cache_control: &Value,
) {
    if let Some(last_tool) = wire_tools.last_mut()
        && let Some(obj) = last_tool.as_object_mut()
    {
        obj.insert("cache_control".to_string(), cache_control.clone());
    }
    if let Some(last_user) = wire_messages
        .iter_mut()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        && let Some(blocks) = last_user.get_mut("content").and_then(Value::as_array_mut)
        && let Some(last_block) = blocks.last_mut()
        && let Some(obj) = last_block.as_object_mut()
    {
        obj.insert("cache_control".to_string(), cache_control.clone());
    }
}

/// 构建 Anthropic Messages 请求体，并返回当前请求的前缀摘要（用于后续签名回放校验）。
pub fn build_request_body(
    config: &MessagesConfig,
    options: &GenerateOptions,
    model: &str,
) -> (Value, String) {
    let mut sys_parts = Vec::new();
    if let Some(sys) = &options.system
        && !sys.is_empty()
    {
        sys_parts.push(sys.as_str());
    }
    for m in &options.messages {
        if let Message::System { content } = m
            && !content.is_empty()
        {
            sys_parts.push(content.as_str());
        }
    }
    let system_text = sys_parts.join("\n\n");

    let mut wire_tools: Vec<Value> = options
        .tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        })
        .collect();

    let mut wire_messages: Vec<Value> = Vec::new();
    let mut prefix_hasher = StreamingPrefixHasher::new(&system_text, &wire_tools);
    let mut hashed_messages = 0usize;

    for msg in &options.messages {
        match msg {
            Message::System { .. } => {}
            Message::User { content, images } => {
                if images.is_empty() {
                    push_user_block(
                        &mut wire_messages,
                        json!({ "type": "text", "text": content }),
                    );
                } else {
                    for img in images {
                        push_user_block(
                            &mut wire_messages,
                            json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": img.media_type,
                                    "data": img.data,
                                },
                            }),
                        );
                    }
                    if !content.is_empty() {
                        push_user_block(
                            &mut wire_messages,
                            json!({ "type": "text", "text": content }),
                        );
                    }
                }
            }
            Message::Tool {
                content,
                tool_call_id,
                is_error,
            } => {
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": if content.is_empty() { "(no output)" } else { content.as_str() },
                });
                if *is_error {
                    block["is_error"] = json!(true);
                }
                push_user_block(&mut wire_messages, block);
            }
            Message::Assistant { blocks, replay } => {
                for finalized in &wire_messages[hashed_messages..] {
                    prefix_hasher.push_message(finalized);
                }
                hashed_messages = wire_messages.len();
                let prior_digest = prefix_hasher.snapshot_digest();

                let can_replay_thinking = replay.as_ref().is_some_and(|r| {
                    r.matches(PROTOCOL_MESSAGES, &config.provider, model)
                        && r.response.get("prefixDigest").and_then(Value::as_str)
                            == Some(prior_digest.as_str())
                        && r.blocks.len() == blocks.len()
                });

                let mut assistant_blocks = Vec::new();
                for (i, block) in blocks.iter().enumerate() {
                    match block {
                        ContentBlock::Reasoning { text } => {
                            if can_replay_thinking
                                && let Some(meta) = replay.as_ref().and_then(|r| r.blocks.get(i))
                            {
                                if meta.get("redacted").and_then(Value::as_bool) == Some(true)
                                    && let Some(data) = meta.get("data").and_then(Value::as_str)
                                {
                                    assistant_blocks.push(json!({
                                        "type": "redacted_thinking",
                                        "data": data,
                                    }));
                                } else if let Some(sig) =
                                    meta.get("signature").and_then(Value::as_str)
                                    && !sig.is_empty()
                                {
                                    assistant_blocks.push(json!({
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": sig,
                                    }));
                                }
                            }
                        }
                        ContentBlock::Text { text } => {
                            if !text.is_empty() {
                                assistant_blocks.push(json!({
                                    "type": "text",
                                    "text": text,
                                }));
                            }
                        }
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            assistant_blocks.push(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": parse_tool_input_json(arguments),
                            }));
                        }
                    }
                }

                if assistant_blocks.is_empty() {
                    assistant_blocks.push(json!({ "type": "text", "text": "" }));
                }

                wire_messages.push(json!({
                    "role": "assistant",
                    "content": assistant_blocks,
                }));
                if let Some(pushed) = wire_messages.last() {
                    prefix_hasher.push_message(pushed);
                    hashed_messages = wire_messages.len();
                }
            }
        }
    }

    for finalized in &wire_messages[hashed_messages..] {
        prefix_hasher.push_message(finalized);
    }
    // 摘要在挂载 cache_control 前计算，避免缓存断点随轮次滑动破坏思考签名回放
    let request_prefix_digest = prefix_hasher.snapshot_digest();

    let cache_control = cache_control_value(resolve_cache_retention(options.cache_retention));
    if let Some(cc) = &cache_control {
        attach_cache_control(&mut wire_tools, &mut wire_messages, cc);
    }

    let mut max_tokens = options.max_tokens.unwrap_or(config.default_max_tokens);
    let adaptive_effort = if config.force_adaptive_thinking {
        options.reasoning_effort.and_then(|effort| match effort {
            ReasoningEffort::Off => None,
            ReasoningEffort::Minimal | ReasoningEffort::Low => Some("low"),
            ReasoningEffort::Medium => Some("medium"),
            ReasoningEffort::High => Some("high"),
            ReasoningEffort::XHigh | ReasoningEffort::Max => Some("max"),
        })
    } else {
        None
    };

    let thinking_budget_val = if adaptive_effort.is_none() {
        options
            .reasoning_effort
            .and_then(|e| thinking_budget(e, config.thinking_budget_tokens))
    } else {
        None
    };

    if let Some(budget) = thinking_budget_val
        && max_tokens <= budget
    {
        max_tokens = budget.saturating_add(1024);
    }

    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "stream": true,
        "messages": wire_messages,
    });
    if !system_text.is_empty() {
        body["system"] = match cache_control {
            Some(cc) => json!([{
                "type": "text",
                "text": system_text,
                "cache_control": cc,
            }]),
            None => Value::String(system_text),
        };
    }
    if !wire_tools.is_empty() {
        body["tools"] = Value::Array(wire_tools);
    }
    if let Some(effort_str) = adaptive_effort {
        body["thinking"] = json!({ "type": "adaptive" });
        body["output_config"] = json!({ "effort": effort_str });
    } else if let Some(budget) = thinking_budget_val {
        body["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget,
        });
    } else if let Some(temp) = options.temperature {
        body["temperature"] = json!(temp);
    }

    (body, request_prefix_digest)
}

enum BlockKindState {
    Text {
        text: String,
    },
    Thinking {
        text: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        fallback_input: Value,
        partial_json: String,
    },
}

impl BlockKindState {
    fn to_content_block(&self) -> ContentBlock {
        match self {
            Self::Text { text } => ContentBlock::Text { text: text.clone() },
            Self::Thinking { text, .. } => ContentBlock::Reasoning { text: text.clone() },
            Self::RedactedThinking { .. } => ContentBlock::Reasoning {
                text: String::new(),
            },
            Self::ToolUse {
                id,
                name,
                fallback_input,
                partial_json,
            } => {
                let arguments = if partial_json.trim().is_empty() {
                    serde_json::to_string(fallback_input).unwrap_or_else(|_| "{}".to_string())
                } else {
                    partial_json.clone()
                };
                ContentBlock::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments,
                }
            }
        }
    }

    fn replay_meta(&self) -> Value {
        match self {
            Self::RedactedThinking { data } => json!({ "redacted": true, "data": data }),
            Self::Thinking { signature, .. } if !signature.is_empty() => {
                json!({ "signature": signature })
            }
            _ => json!({}),
        }
    }
}

enum BlockEntry {
    Open(BlockKindState),
    Closed(BlockKindState),
}

impl BlockEntry {
    fn state(&self) -> &BlockKindState {
        match self {
            Self::Open(s) | Self::Closed(s) => s,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EventPhase {
    Streaming,
    Stopped,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FinishPhase {
    Pending,
    Emitted,
}

/// Anthropic Messages SSE 状态机解码器。
pub struct MessagesSseDecoder {
    provider: String,
    model: String,
    prefix_digest: String,
    response_id: Option<String>,
    response_model: Option<String>,
    blocks: BTreeMap<usize, BlockEntry>,
    usage: TokenUsage,
    has_usage: bool,
    stop_reason: Option<String>,
    event_phase: EventPhase,
    finish_phase: FinishPhase,
}

impl MessagesSseDecoder {
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        prefix_digest: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            prefix_digest: prefix_digest.into(),
            response_id: None,
            response_model: None,
            blocks: BTreeMap::new(),
            usage: TokenUsage::default(),
            has_usage: false,
            stop_reason: None,
            event_phase: EventPhase::Streaming,
            finish_phase: FinishPhase::Pending,
        }
    }

    fn update_usage(&mut self, u: &Value) {
        self.has_usage = true;
        if let Some(v) = u.get("input_tokens").and_then(Value::as_u64) {
            self.usage.input_tokens = v;
        }
        if let Some(v) = u.get("output_tokens").and_then(Value::as_u64) {
            self.usage.output_tokens = v;
        }
        if let Some(v) = u.get("cache_read_input_tokens").and_then(Value::as_u64)
            && v > 0
        {
            self.usage.cache_read_tokens = Some(v);
        }
        if let Some(v) = u.get("cache_creation_input_tokens").and_then(Value::as_u64)
            && v > 0
        {
            self.usage.cache_write_tokens = Some(v);
        }
        let total = self.usage.input_tokens
            + self.usage.output_tokens
            + self.usage.cache_read_tokens.unwrap_or(0)
            + self.usage.cache_write_tokens.unwrap_or(0);
        self.usage.total_tokens = Some(total);
    }

    fn map_finish_reason(&self) -> FinishReason {
        match self.stop_reason.as_deref() {
            Some("tool_use") => FinishReason::ToolCalls,
            Some("max_tokens") => FinishReason::MaxTokens,
            Some("refusal") => FinishReason::Error {
                code: LlmErrorCode::ContentRefused,
                message: "model refused to answer under content safety review".to_string(),
                status: None,
                retry_after_ms: None,
            },
            _ => {
                if self
                    .blocks
                    .values()
                    .any(|b| matches!(b.state(), BlockKindState::ToolUse { .. }))
                {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                }
            }
        }
    }
}

impl SseStreamDecoder for MessagesSseDecoder {
    fn on_event(
        &mut self,
        event: SseEvent,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Result<bool, LlmError> {
        if self.event_phase == EventPhase::Stopped {
            return Ok(true);
        }
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(false);
        }

        let payload: Value = serde_json::from_str(data)
            .map_err(|e| LlmError::malformed(format!("invalid messages SSE json: {e}")))?;

        let ev_type = payload
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or("");

        match ev_type {
            "ping" => {}
            "error" => {
                let err_obj = payload.get("error").unwrap_or(&payload);
                let msg = err_obj
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("messages stream error");
                let etype = err_obj.get("type").and_then(Value::as_str);
                let ecode = err_obj.get("code").and_then(Value::as_str);
                return Err(LlmError::classify_provider_error(None, etype, ecode, msg));
            }
            "message_start" => {
                if let Some(msg) = payload.get("message") {
                    if let Some(id) = msg.get("id").and_then(Value::as_str) {
                        self.response_id = Some(id.to_string());
                    }
                    if let Some(m) = msg.get("model").and_then(Value::as_str) {
                        self.response_model = Some(m.to_string());
                    }
                    if let Some(u) = msg.get("usage") {
                        self.update_usage(u);
                    }
                }
            }
            "content_block_start" => {
                let index = payload
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| LlmError::malformed("messages SSE missing block index"))?;
                if self.blocks.contains_key(&index) {
                    return Ok(false);
                }
                let cb = payload
                    .get("content_block")
                    .ok_or_else(|| LlmError::malformed("messages SSE missing content_block"))?;
                let cb_type = cb.get("type").and_then(Value::as_str).unwrap_or("");

                match cb_type {
                    "text" => {
                        let init_text = cb.get("text").and_then(Value::as_str).unwrap_or("");
                        if !emit(StreamChunk::BlockStart {
                            index,
                            kind: ContentBlockKind::Text,
                        }) {
                            return Ok(true);
                        }
                        if !init_text.is_empty()
                            && !emit(StreamChunk::TextDelta {
                                index,
                                text: init_text.to_string(),
                            })
                        {
                            return Ok(true);
                        }
                        self.blocks.insert(
                            index,
                            BlockEntry::Open(BlockKindState::Text {
                                text: init_text.to_string(),
                            }),
                        );
                    }
                    "thinking" => {
                        let init_think = cb.get("thinking").and_then(Value::as_str).unwrap_or("");
                        let init_sig = cb.get("signature").and_then(Value::as_str).unwrap_or("");
                        if !emit(StreamChunk::BlockStart {
                            index,
                            kind: ContentBlockKind::Reasoning,
                        }) {
                            return Ok(true);
                        }
                        if !init_think.is_empty()
                            && !emit(StreamChunk::ReasoningDelta {
                                index,
                                text: init_think.to_string(),
                            })
                        {
                            return Ok(true);
                        }
                        self.blocks.insert(
                            index,
                            BlockEntry::Open(BlockKindState::Thinking {
                                text: init_think.to_string(),
                                signature: init_sig.to_string(),
                            }),
                        );
                    }
                    "redacted_thinking" => {
                        let rdata = cb.get("data").and_then(Value::as_str).unwrap_or("");
                        if !emit(StreamChunk::BlockStart {
                            index,
                            kind: ContentBlockKind::Reasoning,
                        }) {
                            return Ok(true);
                        }
                        self.blocks.insert(
                            index,
                            BlockEntry::Open(BlockKindState::RedactedThinking {
                                data: rdata.to_string(),
                            }),
                        );
                    }
                    "tool_use" => {
                        let id = cb
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let name = cb
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let fallback_input = cb.get("input").cloned().unwrap_or_else(|| json!({}));
                        if !emit(StreamChunk::BlockStart {
                            index,
                            kind: ContentBlockKind::ToolCall,
                        }) {
                            return Ok(true);
                        }
                        if !emit(StreamChunk::ToolCallDelta {
                            index,
                            id: (!id.is_empty()).then_some(id.clone()),
                            name: (!name.is_empty()).then_some(name.clone()),
                            arguments_delta: String::new(),
                        }) {
                            return Ok(true);
                        }
                        self.blocks.insert(
                            index,
                            BlockEntry::Open(BlockKindState::ToolUse {
                                id,
                                name,
                                fallback_input,
                                partial_json: String::new(),
                            }),
                        );
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = payload
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| LlmError::malformed("messages SSE missing delta index"))?;
                let delta = payload
                    .get("delta")
                    .ok_or_else(|| LlmError::malformed("messages SSE missing delta"))?;
                let delta_type = delta.get("type").and_then(Value::as_str).unwrap_or("");

                if let Some(BlockEntry::Open(block)) = self.blocks.get_mut(&index) {
                    match (delta_type, block) {
                        ("text_delta", BlockKindState::Text { text }) => {
                            if let Some(t) = delta.get("text").and_then(Value::as_str)
                                && !t.is_empty()
                            {
                                text.push_str(t);
                                if !emit(StreamChunk::TextDelta {
                                    index,
                                    text: t.to_string(),
                                }) {
                                    return Ok(true);
                                }
                            }
                        }
                        ("thinking_delta", BlockKindState::Thinking { text, .. }) => {
                            if let Some(t) = delta.get("thinking").and_then(Value::as_str)
                                && !t.is_empty()
                            {
                                text.push_str(t);
                                if !emit(StreamChunk::ReasoningDelta {
                                    index,
                                    text: t.to_string(),
                                }) {
                                    return Ok(true);
                                }
                            }
                        }
                        ("signature_delta", BlockKindState::Thinking { signature, .. }) => {
                            if let Some(s) = delta.get("signature").and_then(Value::as_str) {
                                signature.push_str(s);
                            }
                        }
                        ("input_json_delta", BlockKindState::ToolUse { partial_json, .. }) => {
                            if let Some(pj) = delta.get("partial_json").and_then(Value::as_str)
                                && !pj.is_empty()
                            {
                                partial_json.push_str(pj);
                                if !emit(StreamChunk::ToolCallDelta {
                                    index,
                                    id: None,
                                    name: None,
                                    arguments_delta: pj.to_string(),
                                }) {
                                    return Ok(true);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                let index = payload
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| LlmError::malformed("messages SSE missing stop index"))?;

                if let Some(entry) = self.blocks.get_mut(&index)
                    && matches!(entry, BlockEntry::Open(_))
                {
                    let inner = match std::mem::replace(
                        entry,
                        BlockEntry::Closed(BlockKindState::Text {
                            text: String::new(),
                        }),
                    ) {
                        BlockEntry::Open(s) | BlockEntry::Closed(s) => s,
                    };
                    let block = inner.to_content_block();
                    *entry = BlockEntry::Closed(inner);
                    if !emit(StreamChunk::BlockEnd { index, block }) {
                        return Ok(true);
                    }
                }
            }
            "message_delta" => {
                if let Some(sr) = payload
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                {
                    self.stop_reason = Some(sr.to_string());
                }
                if let Some(u) = payload.get("usage") {
                    self.update_usage(u);
                }
            }
            "message_stop" => {
                self.event_phase = EventPhase::Stopped;
                return Ok(true);
            }
            _ => {}
        }

        Ok(false)
    }

    fn on_finish(&mut self, emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        if self.finish_phase == FinishPhase::Emitted {
            return Ok(());
        }
        self.finish_phase = FinishPhase::Emitted;

        if self.has_usage && !emit(StreamChunk::Usage { usage: self.usage }) {
            return Ok(());
        }

        let replay_blocks: Vec<Value> = self
            .blocks
            .values()
            .map(|entry| entry.state().replay_meta())
            .collect();

        let replay = ReplayEnvelope {
            protocol: PROTOCOL_MESSAGES.to_string(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response: json!({
                "id": self.response_id,
                "responseModel": self.response_model,
                "prefixDigest": self.prefix_digest,
            }),
            blocks: replay_blocks,
        };

        let _ = emit(StreamChunk::Finish {
            reason: self.map_finish_reason(),
            replay: Some(replay),
        });
        Ok(())
    }
}

pub struct MessagesAdapter {
    config: MessagesConfig,
    client: reqwest::Client,
}

impl MessagesAdapter {
    pub fn new(config: MessagesConfig) -> Self {
        Self::with_client(config, shared_http_client())
    }

    pub fn with_client(config: MessagesConfig, client: reqwest::Client) -> Self {
        Self { config, client }
    }

    fn resolve_api_key(&self) -> Result<Option<String>, LlmError> {
        resolve_credential(
            &self.config.provider,
            self.config.api_key.as_deref(),
            self.config.api_key_env.as_deref(),
        )
    }

    fn build_headers(
        &self,
        api_key: Option<&str>,
        extra: &[(String, String)],
    ) -> Vec<(String, String)> {
        let mut headers = vec![(
            "anthropic-version".to_string(),
            self.config.anthropic_version.clone(),
        )];
        if let Some(key) = api_key {
            if self.config.use_bearer_auth {
                headers.push(("Authorization".to_string(), format!("Bearer {key}")));
            } else {
                headers.push(("x-api-key".to_string(), key.to_string()));
            }
        }
        headers.extend(self.config.headers.iter().cloned());
        headers.extend(extra.iter().cloned());
        headers
    }
}

impl LlmAdapter for MessagesAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: self.config.provider.clone(),
            name: self
                .config
                .display_name
                .clone()
                .unwrap_or_else(|| self.config.provider.clone()),
            protocol: PROTOCOL_MESSAGES.to_string(),
            models: self.config.models.clone(),
        })
    }

    fn discover_models(&self) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        self.discover_models_with_headers(&[])
    }

    fn discover_models_with_headers(
        &self,
        extra_headers: &[(String, String)],
    ) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        if !self.config.models.is_empty() {
            let models = self.config.models.clone();
            return Box::pin(async move { Ok(models) });
        }
        let api_key = match self.resolve_api_key() {
            Ok(k) => k,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let url = format!(
            "{}/models?limit=1000",
            ensure_v1_root(&self.config.base_url)
        );
        let headers = self.build_headers(api_key.as_deref(), extra_headers);
        let client = self.client.clone();
        Box::pin(async move { discover_models_http(&client, &url, &headers).await })
    }

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        let api_key = match self.resolve_api_key() {
            Ok(k) => k,
            Err(err) => {
                let (tx, rx) = mpsc::unbounded();
                let _ = tx.unbounded_send(Err(err));
                return Box::pin(rx);
            }
        };

        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let (body, req_digest) = build_request_body(&self.config, &options, &model);
        let url = format!("{}/messages", ensure_v1_root(&self.config.base_url));
        let headers = self.build_headers(api_key.as_deref(), &options.headers);
        let decoder = MessagesSseDecoder::new(self.config.provider.clone(), model, req_digest);

        spawn_sse_stream(
            SseRequest {
                client: self.client.clone(),
                url,
                headers,
                body,
                cancel: options.cancel,
                idle_timeout: self.config.idle_timeout,
            },
            decoder,
        )
    }
}

#[derive(Default, Clone, Copy)]
pub struct MessagesPlugin;

impl Plugin for MessagesPlugin {
    type Config = MessagesConfig;

    fn name(&self) -> Option<&str> {
        Some("llm-messages")
    }

    fn inject(&self) -> Inject {
        Inject::from(["llm"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let llm = ctx.llm()?;
            let provider = config.provider.clone();
            ctx.effect("llm-messages.registerAdapter", || {
                Ok::<_, LlmError>(Some(
                    llm.register_adapter(provider, Arc::new(MessagesAdapter::new(config)))?,
                ))
            })?;
            Ok(None)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockAssembler, ImageAttachment, SseParser, ToolCall, ToolSchema};
    use axum::Router;
    use axum::body::Body;
    use axum::response::Response;
    use axum::routing::post;
    use futures::StreamExt;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};
    use tokio::net::TcpListener;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct CacheEnvGuard {
        _lock: MutexGuard<'static, ()>,
        prev_mizar: Option<OsString>,
        prev_pi: Option<OsString>,
    }

    impl CacheEnvGuard {
        fn cleared() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev_mizar = std::env::var_os("MIZAR_CACHE_RETENTION");
            let prev_pi = std::env::var_os("PI_CACHE_RETENTION");
            // SAFETY: 受 `ENV_LOCK` 串行保护，仅在本 crate 单测中隔离环境变量并在 Drop 时恢复。
            unsafe {
                std::env::remove_var("MIZAR_CACHE_RETENTION");
                std::env::remove_var("PI_CACHE_RETENTION");
            }
            Self {
                _lock: lock,
                prev_mizar,
                prev_pi,
            }
        }

        fn set_mizar(&self, value: &str) {
            // SAFETY: 持有 `ENV_LOCK`，与本 crate 其他测试串行隔离。
            unsafe {
                std::env::set_var("MIZAR_CACHE_RETENTION", value);
            }
        }
    }

    impl Drop for CacheEnvGuard {
        fn drop(&mut self) {
            // SAFETY: 持有 `ENV_LOCK`，恢复进入测试前的环境变量快照。
            unsafe {
                match &self.prev_mizar {
                    Some(v) => std::env::set_var("MIZAR_CACHE_RETENTION", v),
                    None => std::env::remove_var("MIZAR_CACHE_RETENTION"),
                }
                match &self.prev_pi {
                    Some(v) => std::env::set_var("PI_CACHE_RETENTION", v),
                    None => std::env::remove_var("PI_CACHE_RETENTION"),
                }
            }
        }
    }

    #[test]
    fn merges_tool_results_and_omits_stale_thinking_when_prefix_changes() {
        let _env = CacheEnvGuard::cleared();
        let cfg = MessagesConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5".into(),
            ..Default::default()
        };

        let opts1 = GenerateOptions {
            system: Some("sys v1".into()),
            tools: vec![ToolSchema {
                name: "get_time".into(),
                description: "time".into(),
                parameters: json!({ "type": "object" }),
            }],
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let (_, digest1) = build_request_body(&cfg, &opts1, "claude-sonnet-4-5");

        let assistant = Message::assistant(
            vec![
                ContentBlock::Reasoning {
                    text: "secret thought".into(),
                },
                ContentBlock::ToolCall {
                    id: "toolu_1".into(),
                    name: "get_time".into(),
                    arguments: "{}".into(),
                },
                ContentBlock::ToolCall {
                    id: "toolu_2".into(),
                    name: "get_time".into(),
                    arguments: "{}".into(),
                },
            ],
            Some(ReplayEnvelope {
                protocol: PROTOCOL_MESSAGES.into(),
                provider: "anthropic".into(),
                model: "claude-sonnet-4-5".into(),
                response: json!({ "prefixDigest": digest1 }),
                blocks: json!([
                    { "signature": "sig_valid" },
                    {},
                    {}
                ])
                .as_array()
                .unwrap()
                .clone(),
            }),
        );

        let history = vec![
            Message::user("hi"),
            assistant,
            Message::tool("toolu_1", "10:00", false),
            Message::tool("toolu_2", "10:01", true),
        ];

        // 相同前缀：应保留 signature thinking，且合并两个 tool_result 到同一条 user 消息
        let (body_same, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                system: Some("sys v1".into()),
                tools: opts1.tools.clone(),
                messages: history.clone(),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        let msgs = body_same["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["content"][0]["type"], "thinking");
        assert_eq!(msgs[1]["content"][0]["signature"], "sig_valid");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[2]["content"][1]["is_error"], true);
        assert!(msgs[0]["content"][0].get("cache_control").is_none());
        assert!(msgs[2]["content"][0].get("cache_control").is_none());
        assert_eq!(
            msgs[2]["content"][1]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        // 前缀变更（system 改了）：应省略过期签名 thinking 块，防止 API 拒绝
        let (body_changed, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                system: Some("sys v2".into()),
                tools: opts1.tools,
                messages: history,
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        let msgs_changed = body_changed["messages"].as_array().unwrap();
        assert_eq!(msgs_changed[1]["content"][0]["type"], "tool_use");
    }

    #[test]
    fn decodes_anthropic_messages_sse_state_machine() {
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":120,\"cache_read_input_tokens\":30,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"推理中\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"查一下\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"get_time\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"tz\\\":\\\"UTC\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":45}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );

        let mut parser = SseParser::new();
        let events = parser.push(raw.as_bytes(), || {});
        let mut decoder = MessagesSseDecoder::new("anthropic", "claude-sonnet-4-5", "digest_0");
        let mut asm = BlockAssembler::new();

        for ev in events {
            let done = decoder
                .on_event(ev, &mut |chunk| {
                    asm.push(chunk);
                    true
                })
                .unwrap();
            if done {
                break;
            }
        }
        decoder
            .on_finish(&mut |chunk| {
                asm.push(chunk);
                true
            })
            .unwrap();

        assert_eq!(asm.reasoning(), "推理中");
        assert_eq!(asm.text(), "查一下");
        assert_eq!(
            asm.tool_calls(),
            vec![ToolCall {
                id: "toolu_9".into(),
                name: "get_time".into(),
                arguments: r#"{"tz":"UTC"}"#.into(),
            }]
        );
        let usage = asm.usage().unwrap();
        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.cache_read_tokens, Some(30));
        assert_eq!(usage.output_tokens, 45);
        assert_eq!(asm.finish(), FinishReason::ToolCalls);

        let replay = asm.replay().unwrap();
        assert_eq!(replay.response["prefixDigest"], "digest_0");
        assert_eq!(replay.blocks[0]["signature"], "sig_abc");
    }

    #[tokio::test]
    async fn streams_end_to_end_from_mock_messages_server() {
        let app = Router::new().route(
            "/v1/messages",
            post(|| async {
                let body = concat!(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_mock\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
                    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello Claude\"}}\n\n",
                    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                );
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let adapter = MessagesAdapter::new(MessagesConfig {
            api_key: Some("sk-ant-test".into()),
            base_url: format!("http://{addr}"),
            ..Default::default()
        });

        let mut stream = adapter.stream(GenerateOptions {
            messages: vec![Message::system("be concise"), Message::user("hi")],
            ..Default::default()
        });
        let mut asm = BlockAssembler::new();
        while let Some(item) = stream.next().await {
            asm.push(item.unwrap());
        }
        assert_eq!(asm.text(), "Hello Claude");
        assert_eq!(asm.finish(), FinishReason::Stop);
    }

    #[test]
    fn user_images_wire_shape_and_prefix_digest_identity() {
        let _env = CacheEnvGuard::cleared();
        let cfg = MessagesConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5".into(),
            ..Default::default()
        };

        let opts_plain = GenerateOptions {
            system: Some("sys v1".into()),
            messages: vec![Message::user("hi")],
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        let (body_plain, digest_plain) = build_request_body(&cfg, &opts_plain, "claude-sonnet-4-5");
        let legacy_messages = vec![json!({
            "role": "user",
            "content": [{ "type": "text", "text": "hi" }],
        })];
        let mut expected_hasher = StreamingPrefixHasher::new("sys v1", &[]);
        expected_hasher.push_message(&legacy_messages[0]);
        let expected_digest = expected_hasher.snapshot_digest();
        assert_eq!(digest_plain, expected_digest);
        assert_eq!(body_plain["system"], json!("sys v1"));
        assert_eq!(
            serde_json::to_string(&body_plain["messages"]).unwrap(),
            serde_json::to_string(&legacy_messages).unwrap()
        );

        // Short / Long 缓存断点只影响请求体结构，不改变语义 prefix_digest
        let (body_short, digest_short) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Short),
                ..opts_plain.clone()
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(digest_short, expected_digest);
        assert_eq!(
            body_short["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        let (body_long, digest_long) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Long),
                ..opts_plain
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(digest_long, expected_digest);
        assert_eq!(
            body_long["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral", "ttl": "1h" })
        );

        let opts_img_none = GenerateOptions {
            system: Some("sys v1".into()),
            messages: vec![Message::user_with_images(
                "describe",
                vec![ImageAttachment::new("image/png", "AQID")],
            )],
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        let (body_img_none, _) = build_request_body(&cfg, &opts_img_none, "claude-sonnet-4-5");
        assert_eq!(
            body_img_none["messages"],
            json!([{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AQID"
                        }
                    },
                    {
                        "type": "text",
                        "text": "describe"
                    }
                ]
            }])
        );

        // 默认 Short 时仅在末尾 user 块（此处为 text）附带 cache_control
        let (body_img_short, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: None,
                ..opts_img_none
            },
            "claude-sonnet-4-5",
        );
        assert!(
            body_img_short["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body_img_short["messages"][0]["content"][1]["cache_control"],
            json!({ "type": "ephemeral" })
        );

        // 纯图片无文字时不发送空 text 块（Anthropic API 会拒绝空 text）
        let opts_img_only_none = GenerateOptions {
            messages: vec![Message::user_with_images(
                "",
                vec![ImageAttachment::new("image/webp", "UklGRg==")],
            )],
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        };
        let (body_img_only_none, _) =
            build_request_body(&cfg, &opts_img_only_none, "claude-sonnet-4-5");
        assert_eq!(
            body_img_only_none["messages"],
            json!([{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/webp",
                        "data": "UklGRg=="
                    }
                }]
            }])
        );

        let (body_img_only_short, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: None,
                ..opts_img_only_none
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(
            body_img_only_short["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
    }

    #[test]
    fn request_body_cache_control_breakpoints_for_none_short_and_long() {
        let cfg = MessagesConfig::default();
        let base_opts = GenerateOptions {
            system: Some("static system".into()),
            tools: vec![
                ToolSchema {
                    name: "read".into(),
                    description: "read file".into(),
                    parameters: json!({ "type": "object" }),
                },
                ToolSchema {
                    name: "grep".into(),
                    description: "search".into(),
                    parameters: json!({ "type": "object" }),
                },
            ],
            messages: vec![
                Message::user("first question"),
                Message::assistant(
                    vec![ContentBlock::Text {
                        text: "first answer".into(),
                    }],
                    None,
                ),
                Message::user_with_images(
                    "follow up",
                    vec![ImageAttachment::new("image/png", "AQID")],
                ),
            ],
            ..Default::default()
        };

        // None: 不附带任何 cache_control，system 保持纯字符串
        let (body_none, digest_none) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::None),
                ..base_opts.clone()
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(body_none["system"], json!("static system"));
        assert!(body_none["tools"][0].get("cache_control").is_none());
        assert!(body_none["tools"][1].get("cache_control").is_none());
        assert!(
            body_none["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_none["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_none["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_none["messages"][2]["content"][1]
                .get("cache_control")
                .is_none()
        );

        // Short: {"type": "ephemeral"} 落于 system、最后一个工具、最后一条 user 的最后内容块
        let (body_short, digest_short) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Short),
                ..base_opts.clone()
            },
            "claude-sonnet-4-5",
        );
        let short_cc = json!({ "type": "ephemeral" });
        assert_eq!(
            body_short["system"],
            json!([{ "type": "text", "text": "static system", "cache_control": short_cc }])
        );
        assert!(body_short["tools"][0].get("cache_control").is_none());
        assert_eq!(body_short["tools"][1]["cache_control"], short_cc);
        assert!(
            body_short["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_short["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_short["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body_short["messages"][2]["content"][1]["cache_control"],
            short_cc
        );

        // Long: {"type": "ephemeral", "ttl": "1h"} 落于同样三个断点
        let (body_long, digest_long) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Long),
                ..base_opts
            },
            "claude-sonnet-4-5",
        );
        let long_cc = json!({ "type": "ephemeral", "ttl": "1h" });
        assert_eq!(
            body_long["system"],
            json!([{ "type": "text", "text": "static system", "cache_control": long_cc }])
        );
        assert!(body_long["tools"][0].get("cache_control").is_none());
        assert_eq!(body_long["tools"][1]["cache_control"], long_cc);
        assert!(
            body_long["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_long["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert!(
            body_long["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            body_long["messages"][2]["content"][1]["cache_control"],
            long_cc
        );

        assert_eq!(digest_none, digest_short);
        assert_eq!(digest_short, digest_long);
    }

    #[test]
    fn request_body_cache_retention_defaults_to_short_and_respects_env() {
        let env = CacheEnvGuard::cleared();
        let cfg = MessagesConfig::default();
        let opts = GenerateOptions {
            system: Some("sys".into()),
            tools: vec![ToolSchema {
                name: "get_time".into(),
                description: "time".into(),
                parameters: json!({ "type": "object" }),
            }],
            messages: vec![Message::user("hello")],
            cache_retention: None,
            ..Default::default()
        };

        // 未设置环境变量时默认 Short
        let (body_default, _) = build_request_body(&cfg, &opts, "claude-sonnet-4-5");
        let short_cc = json!({ "type": "ephemeral" });
        assert_eq!(body_default["system"][0]["cache_control"], short_cc);
        assert_eq!(body_default["tools"][0]["cache_control"], short_cc);
        assert_eq!(
            body_default["messages"][0]["content"][0]["cache_control"],
            short_cc
        );

        // MIZAR_CACHE_RETENTION=long 生效
        env.set_mizar("long");
        let (body_env_long, _) = build_request_body(&cfg, &opts, "claude-sonnet-4-5");
        let long_cc = json!({ "type": "ephemeral", "ttl": "1h" });
        assert_eq!(body_env_long["system"][0]["cache_control"], long_cc);
        assert_eq!(body_env_long["tools"][0]["cache_control"], long_cc);
        assert_eq!(
            body_env_long["messages"][0]["content"][0]["cache_control"],
            long_cc
        );

        // MIZAR_CACHE_RETENTION=none 生效，但显式 options.cache_retention 优先级更高
        env.set_mizar("none");
        let (body_env_none, _) = build_request_body(&cfg, &opts, "claude-sonnet-4-5");
        assert_eq!(body_env_none["system"], json!("sys"));
        assert!(body_env_none["tools"][0].get("cache_control").is_none());
        assert!(
            body_env_none["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );

        let (body_explicit_override, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Short),
                ..opts
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(
            body_explicit_override["system"][0]["cache_control"],
            short_cc
        );
        assert_eq!(
            body_explicit_override["tools"][0]["cache_control"],
            short_cc
        );
        assert_eq!(
            body_explicit_override["messages"][0]["content"][0]["cache_control"],
            short_cc
        );
    }

    #[test]
    fn request_body_cache_control_handles_trailing_assistant_and_empty_fields() {
        let cfg = MessagesConfig::default();

        // system/tools 为空，末尾为 assistant 时，仍将断点落在最后一条 user 消息上
        let (body_trailing_assistant, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                messages: vec![
                    Message::user("hi"),
                    Message::assistant(
                        vec![ContentBlock::Text {
                            text: "partial".into(),
                        }],
                        None,
                    ),
                ],
                cache_retention: Some(CacheRetention::Short),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        assert!(body_trailing_assistant.get("system").is_none());
        assert!(body_trailing_assistant.get("tools").is_none());
        assert_eq!(
            body_trailing_assistant["messages"][0]["content"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert!(
            body_trailing_assistant["messages"][1]["content"][0]
                .get("cache_control")
                .is_none()
        );

        // 完全没有 user 消息时不崩溃
        let (body_empty_msgs, _) = build_request_body(
            &cfg,
            &GenerateOptions {
                cache_retention: Some(CacheRetention::Long),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(body_empty_msgs["messages"], json!([]));
    }

    #[test]
    fn streaming_prefix_digest_is_deterministic_and_replays_multi_turn_thinking() {
        let _env = CacheEnvGuard::cleared();
        let cfg = MessagesConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5".into(),
            ..Default::default()
        };
        let tools = vec![crate::ToolSchema {
            name: "read".into(),
            description: "Read file".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }];

        // Turn 1: system + tools + user
        let turn1_msgs = vec![Message::user("check main.rs")];
        let (_, digest1) = build_request_body(
            &cfg,
            &GenerateOptions {
                system: Some("sys prompt".into()),
                tools: tools.clone(),
                messages: turn1_msgs.clone(),
                cache_retention: Some(CacheRetention::Short),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );

        let assistant1 = Message::assistant(
            vec![
                ContentBlock::Reasoning {
                    text: "plan 1".into(),
                },
                ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/main.rs"}"#.into(),
                },
            ],
            Some(ReplayEnvelope {
                protocol: PROTOCOL_MESSAGES.into(),
                provider: "anthropic".into(),
                model: "claude-sonnet-4-5".into(),
                response: json!({ "prefixDigest": digest1 }),
                blocks: vec![json!({ "signature": "sig_1" }), json!({})],
            }),
        );

        // Turn 2: 连续多条 tool result 与 user 消息会折叠进同一个 wire user 消息，且必须命中 Turn 1 摘要
        let turn2_msgs = vec![
            turn1_msgs[0].clone(),
            assistant1.clone(),
            Message::tool("call_1", "fn main() {}", false),
            Message::user("now summarize"),
        ];
        let (body2, digest2) = build_request_body(
            &cfg,
            &GenerateOptions {
                system: Some("sys prompt".into()),
                tools: tools.clone(),
                messages: turn2_msgs.clone(),
                cache_retention: Some(CacheRetention::Short),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        assert_eq!(body2["messages"][1]["content"][0]["type"], "thinking");
        assert_eq!(body2["messages"][1]["content"][0]["signature"], "sig_1");

        let assistant2 = Message::assistant(
            vec![
                ContentBlock::Reasoning {
                    text: "plan 2".into(),
                },
                ContentBlock::Text {
                    text: "Done".into(),
                },
            ],
            Some(ReplayEnvelope {
                protocol: PROTOCOL_MESSAGES.into(),
                provider: "anthropic".into(),
                model: "claude-sonnet-4-5".into(),
                response: json!({ "prefixDigest": digest2 }),
                blocks: vec![json!({ "signature": "sig_2" }), json!({})],
            }),
        );

        // Turn 3: 两轮思考签名均应完整回放，且重复调用结果完全一致
        let mut turn3_msgs = turn2_msgs;
        turn3_msgs.push(assistant2);
        turn3_msgs.push(Message::user("thanks"));
        let opts3 = GenerateOptions {
            system: Some("sys prompt".into()),
            tools: tools.clone(),
            messages: turn3_msgs.clone(),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        };
        let (body3_a, digest3_a) = build_request_body(&cfg, &opts3, "claude-sonnet-4-5");
        let (body3_b, digest3_b) = build_request_body(&cfg, &opts3, "claude-sonnet-4-5");
        assert_eq!(digest3_a, digest3_b);
        assert_eq!(body3_a, body3_b);
        assert_eq!(body3_a["messages"][1]["content"][0]["signature"], "sig_1");
        assert_eq!(body3_a["messages"][3]["content"][0]["signature"], "sig_2");

        // 篡改早期消息会导致后续两轮的 thinking 签名均失效并被安全剥离
        let mut tampered = turn3_msgs;
        tampered[0] = Message::user("check other.rs");
        let (body_tampered, digest_tampered) = build_request_body(
            &cfg,
            &GenerateOptions {
                system: Some("sys prompt".into()),
                tools,
                messages: tampered,
                cache_retention: Some(CacheRetention::Short),
                ..Default::default()
            },
            "claude-sonnet-4-5",
        );
        assert_ne!(digest_tampered, digest3_a);
        assert_ne!(
            body_tampered["messages"][1]["content"][0]["type"],
            "thinking"
        );
        assert_ne!(
            body_tampered["messages"][3]["content"][0]["type"],
            "thinking"
        );
    }
}
