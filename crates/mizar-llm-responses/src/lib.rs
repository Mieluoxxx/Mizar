//! OpenAI Responses 协议适配器（对应 `NamiDSH/packages/llm-responses`）。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, CacheRetention, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions,
    LlmAdapter, LlmContextExt, LlmError, Message, ModelInfo, ProviderInfo, ReasoningEffort,
    ReplayEnvelope, SessionAffinityMode, SseEvent, StreamChunk, TokenUsage,
    clamp_openai_prompt_cache_key, detect_session_affinity_mode, push_session_affinity_headers,
    resolve_cache_retention,
};
use mizar_llm_http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, resolve_credential,
    shared_http_client, spawn_sse_stream, trim_base_url,
};
use serde_json::{Value, json};

pub const PROTOCOL_RESPONSES: &str = "openai-responses";

#[derive(Debug, Clone)]
pub struct ResponsesConfig {
    pub provider: String,
    pub display_name: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub headers: HashMap<String, String>,
    pub supports_max_output_tokens: bool,
    pub send_session_affinity_headers: Option<bool>,
    pub session_affinity_mode: Option<SessionAffinityMode>,
    pub supports_long_cache_retention: bool,
    pub default_cache_retention: Option<CacheRetention>,
    pub models: Vec<ModelInfo>,
    pub idle_timeout: Duration,
}

impl Default for ResponsesConfig {
    fn default() -> Self {
        Self {
            provider: "responses".to_string(),
            display_name: None,
            api_key: None,
            api_key_env: None,
            base_url: None,
            model: None,
            headers: HashMap::new(),
            supports_max_output_tokens: true,
            send_session_affinity_headers: None,
            session_affinity_mode: None,
            supports_long_cache_retention: true,
            default_cache_retention: None,
            models: Vec::new(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

/// 把项目内部统一的 `Message` 翻译成 OpenAI Responses API 的 `input` 条目（不带厂商原生推理签名）。
pub fn to_wire(message: &Message) -> Vec<Value> {
    to_wire_inner(message, None)
}

/// 把 `Message` 翻译为 Responses API 的 `input` 条目；若同协议/路由/模型命中且含加密推理签名则一并回放。
pub fn to_wire_with_replay(message: &Message, provider: &str, model: &str) -> Vec<Value> {
    to_wire_inner(message, Some((provider, model)))
}

fn to_wire_inner(message: &Message, replay_target: Option<(&str, &str)>) -> Vec<Value> {
    match message {
        Message::Tool {
            content,
            tool_call_id,
            ..
        } => vec![json!({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": content,
        })],
        Message::Assistant { blocks, replay } => {
            let matched_replay = replay_target.and_then(|(provider, model)| {
                replay
                    .as_ref()
                    .filter(|r| r.matches(PROTOCOL_RESPONSES, provider, model))
            });
            let replay_reasoning_items: Vec<Value> = matched_replay
                .and_then(|r| r.response.get("reasoningItems"))
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter(|item| has_encrypted_reasoning(item))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            let replayed_reasoning = !replay_reasoning_items.is_empty();
            let call_item_ids = matched_replay
                .and_then(|r| r.response.get("callItemIds"))
                .and_then(Value::as_object);

            let mut items = Vec::with_capacity(replay_reasoning_items.len() + blocks.len());
            items.extend(replay_reasoning_items);
            let mut text_buf = String::new();
            let mut calls = Vec::new();

            for block in blocks {
                match block {
                    ContentBlock::Text { text } => text_buf.push_str(text),
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => calls.push((id, name, arguments)),
                    ContentBlock::Reasoning { .. } => {}
                }
            }

            if !text_buf.is_empty() || (calls.is_empty() && !replayed_reasoning) {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": text_buf,
                }));
            }
            for (id, name, arguments) in calls {
                let mut call_obj = json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                });
                if replayed_reasoning
                    && let Some(fc_id) = call_item_ids
                        .and_then(|map| map.get(id.as_str()))
                        .and_then(Value::as_str)
                        .filter(|s| s.starts_with("fc_"))
                    && let Some(map) = call_obj.as_object_mut()
                {
                    map.insert("id".to_string(), Value::String(fc_id.to_string()));
                }
                items.push(call_obj);
            }
            items
        }
        Message::User { content, images } => {
            if images.is_empty() {
                vec![json!({
                    "type": "message",
                    "role": "user",
                    "content": content,
                })]
            } else {
                let mut parts = Vec::with_capacity(images.len() + 1);
                if !content.is_empty() {
                    parts.push(json!({
                        "type": "input_text",
                        "text": content,
                    }));
                }
                for img in images {
                    parts.push(json!({
                        "type": "input_image",
                        "image_url": img.data_url(),
                    }));
                }
                vec![json!({
                    "type": "message",
                    "role": "user",
                    "content": parts,
                })]
            }
        }
        Message::System { content } => vec![json!({
            "type": "message",
            "role": "system",
            "content": content,
        })],
    }
}

fn has_encrypted_reasoning(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("reasoning")
        && item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty())
}

fn normalize_reasoning_item(item: &Value) -> Option<Value> {
    if item.get("type").and_then(Value::as_str) != Some("reasoning") {
        return None;
    }
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), json!("reasoning"));
    if let Some(id) = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        obj.insert("id".to_string(), json!(id));
    }
    let summary = item
        .get("summary")
        .filter(|v| v.is_array())
        .cloned()
        .unwrap_or_else(|| json!([]));
    obj.insert("summary".to_string(), summary);
    if let Some(content) = item.get("content").filter(|v| v.is_array()) {
        obj.insert("content".to_string(), content.clone());
    }
    if let Some(enc) = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        obj.insert("encrypted_content".to_string(), json!(enc));
    }
    Some(Value::Object(obj))
}

fn upsert_reasoning_item(list: &mut Vec<Value>, raw: &Value) {
    let Some(normalized) = normalize_reasoning_item(raw) else {
        return;
    };
    if let Some(item_id) = normalized.get("id").and_then(Value::as_str)
        && let Some(existing) = list
            .iter_mut()
            .find(|entry| entry.get("id").and_then(Value::as_str) == Some(item_id))
    {
        *existing = normalized;
        return;
    }
    list.push(normalized);
}

fn effort_str(effort: ReasoningEffort) -> Option<&'static str> {
    match effort {
        ReasoningEffort::Off => None,
        ReasoningEffort::Minimal => Some("minimal"),
        ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium => Some("medium"),
        ReasoningEffort::High => Some("high"),
        ReasoningEffort::XHigh | ReasoningEffort::Max => Some("high"),
    }
}

pub fn build_request_body(
    config: &ResponsesConfig,
    options: &GenerateOptions,
    model: &str,
) -> Value {
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
    let instructions = sys_parts.join("\n\n");

    let input: Vec<Value> = options
        .messages
        .iter()
        .filter(|m| !matches!(m, Message::System { .. }))
        .flat_map(|m| to_wire_with_replay(m, &config.provider, model))
        .collect();

    let mut body = json!({
        "model": model,
        "stream": true,
        "store": false,
        "input": input,
    });
    if !instructions.is_empty() {
        body["instructions"] = Value::String(instructions);
    }
    let retention =
        resolve_cache_retention(options.cache_retention.or(config.default_cache_retention));
    if retention != CacheRetention::None
        && let Some(key) = options
            .session_id
            .as_deref()
            .and_then(clamp_openai_prompt_cache_key)
    {
        body["prompt_cache_key"] = Value::String(key);
    }
    if retention == CacheRetention::Long && config.supports_long_cache_retention {
        body["prompt_cache_retention"] = json!("24h");
    }
    if config.supports_max_output_tokens
        && let Some(max_tokens) = options.max_tokens
    {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(temp) = options.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(effort) = options.reasoning_effort.and_then(effort_str) {
        body["reasoning"] = json!({ "effort": effort });
        body["include"] = json!(["reasoning.encrypted_content"]);
    }
    if !options.tools.is_empty() {
        let tools: Vec<Value> = options
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": false,
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResponsesEventPhase {
    Streaming,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResponsesFinishPhase {
    Pending,
    Emitted,
}

struct PendingCall {
    stream_idx: usize,
    call_id: String,
    name: String,
    arguments: String,
}

/// OpenAI Responses SSE 解码器。
pub struct ResponsesSseDecoder {
    provider: String,
    model: String,
    response_id: Option<String>,
    next_index: usize,
    text_idx: Option<usize>,
    reasoning_idx: Option<usize>,
    pending_calls: HashMap<String, PendingCall>,
    completed_items: std::collections::HashSet<String>,
    reasoning_items: Vec<Value>,
    call_item_ids: HashMap<String, String>,
    has_tool_calls: bool,
    usage: Option<TokenUsage>,
    finish_reason: Option<FinishReason>,
    event_phase: ResponsesEventPhase,
    finish_phase: ResponsesFinishPhase,
}

impl ResponsesSseDecoder {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            response_id: None,
            next_index: 0,
            text_idx: None,
            reasoning_idx: None,
            pending_calls: HashMap::new(),
            completed_items: std::collections::HashSet::new(),
            reasoning_items: Vec::new(),
            call_item_ids: HashMap::new(),
            has_tool_calls: false,
            usage: None,
            finish_reason: None,
            event_phase: ResponsesEventPhase::Streaming,
            finish_phase: ResponsesFinishPhase::Pending,
        }
    }

    fn alloc_index(&mut self) -> usize {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }

    fn parse_usage(resp_obj: &Value) -> Option<TokenUsage> {
        let u = resp_obj.get("usage")?;
        let raw_input = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
        let cached = u
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
        let reasoning = u
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64);
        let total = u.get("total_tokens").and_then(Value::as_u64);
        Some(TokenUsage {
            input_tokens: raw_input.saturating_sub(cached),
            output_tokens: output,
            total_tokens: total,
            cache_read_tokens: (cached > 0).then_some(cached),
            cache_write_tokens: None,
            reasoning_tokens: reasoning,
        })
    }
}

impl SseStreamDecoder for ResponsesSseDecoder {
    fn on_event(
        &mut self,
        event: SseEvent,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Result<bool, LlmError> {
        if self.event_phase == ResponsesEventPhase::Done {
            return Ok(true);
        }
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(false);
        }
        if data == "[DONE]" {
            self.event_phase = ResponsesEventPhase::Done;
            return Ok(true);
        }

        let payload: Value = serde_json::from_str(data)
            .map_err(|e| LlmError::malformed(format!("invalid responses SSE json: {e}")))?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or("");

        match event_type {
            "response.created" | "response.in_progress" => {
                if let Some(id) = payload.pointer("/response/id").and_then(Value::as_str) {
                    self.response_id = Some(id.to_string());
                }
            }
            "response.output_text.delta" => {
                if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    let idx = match self.text_idx {
                        Some(i) => i,
                        None => {
                            let i = self.alloc_index();
                            self.text_idx = Some(i);
                            if !emit(StreamChunk::BlockStart {
                                index: i,
                                kind: ContentBlockKind::Text,
                            }) {
                                return Ok(true);
                            }
                            i
                        }
                    };
                    if !emit(StreamChunk::TextDelta {
                        index: idx,
                        text: delta.to_string(),
                    }) {
                        return Ok(true);
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    let idx = match self.reasoning_idx {
                        Some(i) => i,
                        None => {
                            let i = self.alloc_index();
                            self.reasoning_idx = Some(i);
                            if !emit(StreamChunk::BlockStart {
                                index: i,
                                kind: ContentBlockKind::Reasoning,
                            }) {
                                return Ok(true);
                            }
                            i
                        }
                    };
                    if !emit(StreamChunk::ReasoningDelta {
                        index: idx,
                        text: delta.to_string(),
                    }) {
                        return Ok(true);
                    }
                }
            }
            "response.output_item.added" => {
                if let Some(item) = payload.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                    && let Some(item_id) = item.get("id").and_then(Value::as_str)
                    && !self.completed_items.contains(item_id)
                    && !self.pending_calls.contains_key(item_id)
                {
                    let stream_idx = self.alloc_index();
                    let call_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let args = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();

                    if !emit(StreamChunk::BlockStart {
                        index: stream_idx,
                        kind: ContentBlockKind::ToolCall,
                    }) {
                        return Ok(true);
                    }
                    if !args.is_empty()
                        && !emit(StreamChunk::ToolCallDelta {
                            index: stream_idx,
                            id: (!call_id.is_empty()).then_some(call_id.clone()),
                            name: (!name.is_empty()).then_some(name.clone()),
                            arguments_delta: args.clone(),
                        })
                    {
                        return Ok(true);
                    }

                    self.pending_calls.insert(
                        item_id.to_string(),
                        PendingCall {
                            stream_idx,
                            call_id,
                            name,
                            arguments: args,
                        },
                    );
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(item_id) = payload.get("item_id").and_then(Value::as_str)
                    && !self.completed_items.contains(item_id)
                    && let Some(delta) = payload.get("delta").and_then(Value::as_str)
                {
                    let entry = if let Some(existing) = self.pending_calls.get_mut(item_id) {
                        existing
                    } else {
                        let stream_idx = self.alloc_index();
                        if !emit(StreamChunk::BlockStart {
                            index: stream_idx,
                            kind: ContentBlockKind::ToolCall,
                        }) {
                            return Ok(true);
                        }
                        self.pending_calls.insert(
                            item_id.to_string(),
                            PendingCall {
                                stream_idx,
                                call_id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            },
                        );
                        self.pending_calls.get_mut(item_id).expect("inserted")
                    };

                    entry.arguments.push_str(delta);
                    if !emit(StreamChunk::ToolCallDelta {
                        index: entry.stream_idx,
                        id: None,
                        name: None,
                        arguments_delta: delta.to_string(),
                    }) {
                        return Ok(true);
                    }
                }
            }
            "response.output_item.done" => {
                if let Some(item) = payload.get("item") {
                    match item.get("type").and_then(Value::as_str) {
                        Some("reasoning") => {
                            upsert_reasoning_item(&mut self.reasoning_items, item);
                        }
                        Some("function_call") => {
                            let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
                            if !item_id.is_empty()
                                && !self.completed_items.insert(item_id.to_string())
                            {
                                return Ok(false);
                            }
                            let mut pending =
                                self.pending_calls
                                    .remove(item_id)
                                    .unwrap_or_else(|| PendingCall {
                                        stream_idx: self.alloc_index(),
                                        call_id: String::new(),
                                        name: String::new(),
                                        arguments: String::new(),
                                    });
                            if let Some(call_id) = item.get("call_id").and_then(Value::as_str)
                                && !call_id.is_empty()
                            {
                                pending.call_id = call_id.to_string();
                            }
                            if let Some(name) = item.get("name").and_then(Value::as_str)
                                && !name.is_empty()
                            {
                                pending.name = name.to_string();
                            }
                            if let Some(args) = item.get("arguments").and_then(Value::as_str)
                                && !args.is_empty()
                            {
                                pending.arguments = args.to_string();
                            }
                            if item_id.starts_with("fc_") && !pending.call_id.is_empty() {
                                self.call_item_ids
                                    .insert(pending.call_id.clone(), item_id.to_string());
                            }
                            self.has_tool_calls = true;
                            if !emit(StreamChunk::BlockEnd {
                                index: pending.stream_idx,
                                block: ContentBlock::ToolCall {
                                    id: pending.call_id,
                                    name: pending.name,
                                    arguments: pending.arguments,
                                },
                            }) {
                                return Ok(true);
                            }
                        }
                        _ => {}
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                if let Some(resp_obj) = payload.get("response") {
                    if let Some(id) = resp_obj.get("id").and_then(Value::as_str) {
                        self.response_id = Some(id.to_string());
                    }
                    if let Some(output) = resp_obj.get("output").and_then(Value::as_array) {
                        for item in output {
                            if has_encrypted_reasoning(item) {
                                upsert_reasoning_item(&mut self.reasoning_items, item);
                            }
                        }
                    }
                    if let Some(u) = Self::parse_usage(resp_obj) {
                        self.usage = Some(u);
                    }
                    if event_type == "response.incomplete"
                        || resp_obj
                            .pointer("/incomplete_details/reason")
                            .and_then(Value::as_str)
                            == Some("max_output_tokens")
                    {
                        self.finish_reason = Some(FinishReason::MaxTokens);
                    }
                }
                self.event_phase = ResponsesEventPhase::Done;
                return Ok(true);
            }
            "response.failed" | "error" => {
                let err_obj = payload
                    .pointer("/response/error")
                    .or_else(|| payload.get("error"))
                    .unwrap_or(&payload);
                let msg = err_obj
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("responses stream failed");
                let ecode = err_obj.get("code").and_then(Value::as_str);
                let etype = err_obj.get("type").and_then(Value::as_str);
                return Err(LlmError::classify_provider_error(None, etype, ecode, msg));
            }
            _ => {}
        }

        Ok(false)
    }

    fn on_finish(&mut self, emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        if self.finish_phase == ResponsesFinishPhase::Emitted {
            return Ok(());
        }
        self.finish_phase = ResponsesFinishPhase::Emitted;

        let mut remaining: Vec<PendingCall> = self.pending_calls.drain().map(|(_, p)| p).collect();
        remaining.sort_by_key(|p| p.stream_idx);
        for pending in remaining {
            self.has_tool_calls = true;
            if !emit(StreamChunk::BlockEnd {
                index: pending.stream_idx,
                block: ContentBlock::ToolCall {
                    id: pending.call_id,
                    name: pending.name,
                    arguments: pending.arguments,
                },
            }) {
                return Ok(());
            }
        }

        if let Some(usage) = self.usage
            && !emit(StreamChunk::Usage { usage })
        {
            return Ok(());
        }

        let reason = self
            .finish_reason
            .clone()
            .unwrap_or(if self.has_tool_calls {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            });
        let mut replay_response = json!({ "id": self.response_id });
        if let Some(obj) = replay_response.as_object_mut() {
            let encrypted_items: Vec<Value> = self
                .reasoning_items
                .iter()
                .filter(|item| has_encrypted_reasoning(item))
                .cloned()
                .collect();
            if !encrypted_items.is_empty() {
                obj.insert("reasoningItems".to_string(), Value::Array(encrypted_items));
            }
            if !self.call_item_ids.is_empty() {
                obj.insert("callItemIds".to_string(), json!(self.call_item_ids));
            }
        }
        let replay = ReplayEnvelope {
            protocol: PROTOCOL_RESPONSES.to_string(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response: replay_response,
            blocks: Vec::new(),
        };
        let _ = emit(StreamChunk::Finish {
            reason,
            replay: Some(replay),
        });
        Ok(())
    }
}

pub struct ResponsesAdapter {
    config: ResponsesConfig,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl ResponsesAdapter {
    pub fn new(config: ResponsesConfig) -> Self {
        Self::with_client(config, shared_http_client())
    }

    pub fn with_client(config: ResponsesConfig, client: reqwest::Client) -> Self {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = config.model.clone().unwrap_or_else(|| "gpt-5".to_string());
        Self {
            config,
            base_url,
            model,
            client,
        }
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
        session_id: Option<&str>,
    ) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        for (k, v) in &self.config.headers {
            headers.push((k.clone(), v.clone()));
        }
        headers.extend(extra.iter().cloned());
        if self.config.send_session_affinity_headers.unwrap_or(true) {
            let mode = self.config.session_affinity_mode.unwrap_or_else(|| {
                detect_session_affinity_mode(&self.config.provider, self.config.base_url.as_deref())
            });
            push_session_affinity_headers(&mut headers, mode, session_id, false);
        }
        headers
    }
}

impl LlmAdapter for ResponsesAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: self.config.provider.clone(),
            name: self
                .config
                .display_name
                .clone()
                .unwrap_or_else(|| self.config.provider.clone()),
            protocol: PROTOCOL_RESPONSES.to_string(),
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
        let url = format!("{}/models", trim_base_url(&self.base_url));
        let headers = self.build_headers(api_key.as_deref(), extra_headers, None);
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

        let model = options.model.clone().unwrap_or_else(|| self.model.clone());
        let body = build_request_body(&self.config, &options, &model);
        let url = format!("{}/responses", trim_base_url(&self.base_url));
        let headers = self.build_headers(
            api_key.as_deref(),
            &options.headers,
            options.session_id.as_deref(),
        );
        let decoder = ResponsesSseDecoder::new(self.config.provider.clone(), model);

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
pub struct ResponsesPlugin;

impl Plugin for ResponsesPlugin {
    type Config = ResponsesConfig;

    fn name(&self) -> Option<&str> {
        Some("llm-responses")
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
            ctx.effect("llm-responses.registerAdapter", || {
                Ok::<_, LlmError>(Some(
                    llm.register_adapter(provider, Arc::new(ResponsesAdapter::new(config)))?,
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
    use mizar_llm::{BlockAssembler, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH, SseParser, ToolCall};
    use tokio::net::TcpListener;

    #[test]
    fn converts_messages_to_responses_wire_format() {
        let assistant = Message::assistant_parts(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "get_time".into(),
                arguments: "{}".into(),
            }],
        );
        assert_eq!(
            to_wire(&assistant),
            vec![json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_time",
                "arguments": "{}",
            })]
        );

        let tool_out = Message::tool("call_1", "12:00", false);
        assert_eq!(
            to_wire(&tool_out),
            vec![json!({
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "12:00",
            })]
        );
    }

    #[test]
    fn parses_responses_sse_reasoning_text_usage_and_function_calls() {
        let raw = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"想一想\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"现在\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"几点\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_time\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_time\",\"arguments\":\"{}\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":50,\"output_tokens\":20,\"total_tokens\":70,\"input_tokens_details\":{\"cached_tokens\":10}}}}\n\n",
        );

        let mut parser = SseParser::new();
        let events = parser.push(raw.as_bytes(), || {});
        let mut decoder = ResponsesSseDecoder::new("responses", "gpt-5");
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

        assert_eq!(asm.reasoning(), "想一想");
        assert_eq!(asm.text(), "现在几点");
        assert_eq!(
            asm.tool_calls(),
            vec![ToolCall {
                id: "call_1".into(),
                name: "get_time".into(),
                arguments: "{}".into(),
            }]
        );
        let usage = asm.usage().unwrap();
        assert_eq!(usage.input_tokens, 40);
        assert_eq!(usage.cache_read_tokens, Some(10));
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(asm.finish(), FinishReason::ToolCalls);
        assert_eq!(asm.replay().unwrap().response["id"], "resp_1");
    }

    #[tokio::test]
    async fn streams_end_to_end_from_mock_responses_server() {
        let app = Router::new().route(
            "/responses",
            post(|| async {
                let body = concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi \"}\n\n",
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Responses\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_mock\"}}\n\n",
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

        let adapter = ResponsesAdapter::new(ResponsesConfig {
            api_key: Some("k".into()),
            base_url: Some(format!("http://{addr}")),
            ..Default::default()
        });

        let mut stream = adapter.stream(GenerateOptions {
            messages: vec![Message::system("be nice"), Message::user("hi")],
            ..Default::default()
        });
        let mut asm = BlockAssembler::new();
        while let Some(item) = stream.next().await {
            asm.push(item.unwrap());
        }
        assert_eq!(asm.text(), "Hi Responses");
        assert_eq!(asm.finish(), FinishReason::Stop);
    }

    #[test]
    fn user_images_wire_shape_and_empty_string_identity() {
        use mizar_llm::ImageAttachment;

        let plain = to_wire(&Message::user("hi"));
        let expected_plain = vec![json!({
            "type": "message",
            "role": "user",
            "content": "hi",
        })];
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            serde_json::to_string(&expected_plain).unwrap()
        );

        let with_img =
            Message::user_with_images("describe", vec![ImageAttachment::new("image/png", "AQID")]);
        assert_eq!(
            to_wire(&with_img),
            vec![json!({
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "describe" },
                    { "type": "input_image", "image_url": "data:image/png;base64,AQID" },
                ]
            })]
        );

        let img_only =
            Message::user_with_images("", vec![ImageAttachment::new("image/jpeg", "/9j/4A==")]);
        assert_eq!(
            to_wire(&img_only),
            vec![json!({
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4A==" },
                ]
            })]
        );
    }

    #[tokio::test]
    async fn injects_prompt_cache_key_retention_and_affinity_headers() {
        let recorded = Arc::new(std::sync::Mutex::new(Vec::<(
            Option<String>,
            Option<String>,
            Option<String>,
            Value,
        )>::new()));
        let seen = Arc::clone(&recorded);
        let app = Router::new().route(
            "/responses",
            post(move |headers: axum::http::HeaderMap, body: axum::Json<Value>| {
                let seen = Arc::clone(&seen);
                async move {
                    let h = |name: &str| {
                        headers
                            .get(name)
                            .and_then(|v| v.to_str().ok())
                            .map(ToString::to_string)
                    };
                    seen.lock().unwrap().push((
                        h("session_id"),
                        h("x-client-request-id"),
                        h("x-session-id"),
                        body.0,
                    ));
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cache\"}}\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let adapter = ResponsesAdapter::new(ResponsesConfig {
            provider: "openai".into(),
            api_key: Some("k".into()),
            base_url: Some(format!("http://{addr}")),
            ..Default::default()
        });

        let long_sid = "sess_".to_string() + &"x".repeat(80);
        let mut s1 = adapter.stream(GenerateOptions {
            messages: vec![Message::system("static sys"), Message::user("hi")],
            session_id: Some(long_sid.clone()),
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        });
        while let Some(chunk) = s1.next().await {
            let _ = chunk.unwrap();
        }

        // OpenRouter 路由 + CacheRetention::None 时：只发 x-session-id，不写 prompt_cache_key
        let or_adapter = ResponsesAdapter::new(ResponsesConfig {
            provider: "openrouter".into(),
            api_key: Some("k".into()),
            base_url: Some(format!("http://{addr}")),
            ..Default::default()
        });
        let mut s2 = or_adapter.stream(GenerateOptions {
            messages: vec![Message::user("hi")],
            session_id: Some("or-sess-1".into()),
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        });
        while let Some(chunk) = s2.next().await {
            let _ = chunk.unwrap();
        }

        let log = recorded.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].0.as_deref(), Some(long_sid.as_str()));
        assert_eq!(log[0].1.as_deref(), Some(long_sid.as_str()));
        assert_eq!(log[0].2, None);
        let key = log[0].3["prompt_cache_key"].as_str().unwrap();
        assert_eq!(key.chars().count(), OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH);
        assert_eq!(log[0].3["prompt_cache_retention"], "24h");

        assert_eq!(log[1].0, None);
        assert_eq!(log[1].1, None);
        assert_eq!(log[1].2.as_deref(), Some("or-sess-1"));
        assert!(log[1].3.get("prompt_cache_key").is_none());
        assert!(log[1].3.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn preserves_encrypted_reasoning_across_same_model_and_strips_on_mismatch() {
        let raw = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"想一步\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"想一步\"}],\"encrypted_content\":\"enc_v1\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{}\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_r\",\"output\":[{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"想一步\"}],\"encrypted_content\":\"enc_v1_final\"}]}}\n\n",
        );

        let mut parser = SseParser::new();
        let events = parser.push(raw.as_bytes(), || {});
        let mut decoder = ResponsesSseDecoder::new("responses", "gpt-5");
        let mut asm = BlockAssembler::new();
        for ev in events {
            if decoder
                .on_event(ev, &mut |chunk| {
                    asm.push(chunk);
                    true
                })
                .unwrap()
            {
                break;
            }
        }
        decoder
            .on_finish(&mut |chunk| {
                asm.push(chunk);
                true
            })
            .unwrap();

        let replay = asm.replay().unwrap();
        assert_eq!(
            replay.response["reasoningItems"][0]["encrypted_content"],
            "enc_v1_final"
        );
        assert_eq!(replay.response["callItemIds"]["call_1"], "fc_1");

        let assistant = Message::assistant(asm.blocks(), Some(replay));
        let same = to_wire_with_replay(&assistant, "responses", "gpt-5");
        assert_eq!(same[0]["type"], "reasoning");
        assert_eq!(same[0]["encrypted_content"], "enc_v1_final");
        assert_eq!(same[1]["type"], "function_call");
        assert_eq!(same[1]["id"], "fc_1");

        let switched = to_wire_with_replay(&assistant, "responses", "gpt-4.1");
        assert_eq!(switched.len(), 1);
        assert_eq!(switched[0]["type"], "function_call");
        assert!(switched[0].get("id").is_none());

        let req_with_effort = build_request_body(
            &ResponsesConfig::default(),
            &GenerateOptions {
                reasoning_effort: Some(ReasoningEffort::Medium),
                messages: vec![Message::user("hi")],
                ..Default::default()
            },
            "gpt-5",
        );
        assert_eq!(
            req_with_effort["include"],
            json!(["reasoning.encrypted_content"])
        );
    }
}
