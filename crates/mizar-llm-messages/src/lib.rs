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
use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions, LlmAdapter,
    LlmContextExt, LlmError, LlmErrorCode, Message, ModelInfo, ProviderInfo, ReasoningEffort,
    ReplayEnvelope, SseEvent, StreamChunk, TokenUsage, prefix_digest,
};
use mizar_llm_http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, ensure_v1_root,
    resolve_credential, shared_http_client, spawn_sse_stream,
};
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

    let wire_tools: Vec<Value> = options
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
                let prior_digest = prefix_digest(&json!({
                    "system": &system_text,
                    "tools": &wire_tools,
                    "messages": &wire_messages,
                }));

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
            }
        }
    }

    let request_prefix_digest = prefix_digest(&json!({
        "system": &system_text,
        "tools": &wire_tools,
        "messages": &wire_messages,
    }));

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
        body["system"] = Value::String(system_text);
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

enum BlockState {
    Text {
        text: String,
    },
    Thinking {
        text: String,
        signature: String,
        redacted: bool,
        redacted_data: String,
    },
    ToolUse {
        id: String,
        name: String,
        fallback_input: Value,
        partial_json: String,
    },
}

/// Anthropic Messages SSE 状态机解码器。
pub struct MessagesSseDecoder {
    provider: String,
    model: String,
    prefix_digest: String,
    response_id: Option<String>,
    response_model: Option<String>,
    blocks: BTreeMap<usize, BlockState>,
    usage: TokenUsage,
    has_usage: bool,
    stop_reason: Option<String>,
    finished_emitted: bool,
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
            finished_emitted: false,
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
                    .any(|b| matches!(b, BlockState::ToolUse { .. }))
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
                            BlockState::Text {
                                text: init_text.to_string(),
                            },
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
                            BlockState::Thinking {
                                text: init_think.to_string(),
                                signature: init_sig.to_string(),
                                redacted: false,
                                redacted_data: String::new(),
                            },
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
                            BlockState::Thinking {
                                text: String::new(),
                                signature: String::new(),
                                redacted: true,
                                redacted_data: rdata.to_string(),
                            },
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
                            BlockState::ToolUse {
                                id,
                                name,
                                fallback_input,
                                partial_json: String::new(),
                            },
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

                if let Some(block) = self.blocks.get_mut(&index) {
                    match (delta_type, block) {
                        ("text_delta", BlockState::Text { text }) => {
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
                        ("thinking_delta", BlockState::Thinking { text, .. }) => {
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
                        ("signature_delta", BlockState::Thinking { signature, .. }) => {
                            if let Some(s) = delta.get("signature").and_then(Value::as_str) {
                                signature.push_str(s);
                            }
                        }
                        ("input_json_delta", BlockState::ToolUse { partial_json, .. }) => {
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

                if let Some(state) = self.blocks.get(&index) {
                    let block = match state {
                        BlockState::Text { text } => ContentBlock::Text { text: text.clone() },
                        BlockState::Thinking { text, .. } => {
                            ContentBlock::Reasoning { text: text.clone() }
                        }
                        BlockState::ToolUse {
                            id,
                            name,
                            fallback_input,
                            partial_json,
                        } => {
                            let arguments = if partial_json.trim().is_empty() {
                                serde_json::to_string(fallback_input)
                                    .unwrap_or_else(|_| "{}".to_string())
                            } else {
                                partial_json.clone()
                            };
                            ContentBlock::ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                arguments,
                            }
                        }
                    };
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
            "message_stop" => return Ok(true),
            _ => {}
        }

        Ok(false)
    }

    fn on_finish(&mut self, emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        if self.finished_emitted {
            return Ok(());
        }
        self.finished_emitted = true;

        if self.has_usage && !emit(StreamChunk::Usage { usage: self.usage }) {
            return Ok(());
        }

        let replay_blocks: Vec<Value> = self
            .blocks
            .values()
            .map(|b| match b {
                BlockState::Thinking {
                    signature,
                    redacted,
                    redacted_data,
                    ..
                } => {
                    if *redacted {
                        json!({ "redacted": true, "data": redacted_data })
                    } else if !signature.is_empty() {
                        json!({ "signature": signature })
                    } else {
                        json!({})
                    }
                }
                _ => json!({}),
            })
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
        let headers = self.build_headers(api_key.as_deref(), &[]);
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
    use axum::Router;
    use axum::body::Body;
    use axum::response::Response;
    use axum::routing::post;
    use futures::StreamExt;
    use mizar_llm::{BlockAssembler, SseParser, ToolCall, ToolSchema};
    use tokio::net::TcpListener;

    #[test]
    fn merges_tool_results_and_omits_stale_thinking_when_prefix_changes() {
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
        use mizar_llm::ImageAttachment;

        let cfg = MessagesConfig {
            provider: "anthropic".into(),
            model: "claude-sonnet-4-5".into(),
            ..Default::default()
        };

        let opts_plain = GenerateOptions {
            system: Some("sys v1".into()),
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let (body_plain, digest_plain) = build_request_body(&cfg, &opts_plain, "claude-sonnet-4-5");
        let legacy_messages = vec![json!({
            "role": "user",
            "content": [{ "type": "text", "text": "hi" }],
        })];
        let legacy_digest = prefix_digest(&json!({
            "system": "sys v1",
            "tools": [],
            "messages": &legacy_messages,
        }));
        assert_eq!(digest_plain, legacy_digest);
        assert_eq!(
            serde_json::to_string(&body_plain["messages"]).unwrap(),
            serde_json::to_string(&legacy_messages).unwrap()
        );

        let opts_img = GenerateOptions {
            system: Some("sys v1".into()),
            messages: vec![Message::user_with_images(
                "describe",
                vec![ImageAttachment::new("image/png", "AQID")],
            )],
            ..Default::default()
        };
        let (body_img, _) = build_request_body(&cfg, &opts_img, "claude-sonnet-4-5");
        assert_eq!(
            body_img["messages"],
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

        // 纯图片无文字时不发送空 text 块（Anthropic API 会拒绝空 text）
        let opts_img_only = GenerateOptions {
            messages: vec![Message::user_with_images(
                "",
                vec![ImageAttachment::new("image/webp", "UklGRg==")],
            )],
            ..Default::default()
        };
        let (body_img_only, _) = build_request_body(&cfg, &opts_img_only, "claude-sonnet-4-5");
        assert_eq!(
            body_img_only["messages"],
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
    }
}
