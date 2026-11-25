//! OpenAI Responses 协议适配器（对应 `NamiDSH/packages/llm-responses`）。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, CacheRetention, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions,
    LlmAdapter, LlmContextExt, LlmError, LlmErrorCode, Message, ModelInfo, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, SessionAffinityMode, SseEvent, StreamChunk, TokenUsage,
    ToolSchema, clamp_openai_prompt_cache_key, detect_session_affinity_mode,
    push_session_affinity_headers, resolve_cache_retention,
};
use mizar_llm_http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, resolve_credential,
    shared_http_client, spawn_sse_stream, trim_base_url,
};
use serde_json::{Value, json};

pub const PROTOCOL_RESPONSES: &str = "openai-responses";
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u32 = 16;

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
    pub supports_developer_role: bool,
    pub use_top_level_instructions: bool,
    pub supports_strict_mode: bool,
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
            supports_developer_role: true,
            use_top_level_instructions: false,
            supports_strict_mode: false,
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

/// 把 `Message` 翻译为 Responses API 的 `input` 条目；若同协议/路由/模型命中且含原生保序输出序列则一并回放。
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
            let block_call_ids: HashSet<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolCall { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect();

            let matched_replay = replay_target.and_then(|(provider, model)| {
                replay
                    .as_ref()
                    .filter(|r| r.matches(PROTOCOL_RESPONSES, provider, model))
            });

            if let Some(output_items) = matched_replay
                .and_then(|r| r.response.get("outputItems"))
                .and_then(Value::as_array)
                && !output_items.is_empty()
            {
                let mut items = Vec::with_capacity(output_items.len());
                for item in output_items {
                    match item.get("type").and_then(Value::as_str) {
                        Some("reasoning") if has_encrypted_reasoning(item) => {
                            items.push(item.clone());
                        }
                        Some("message") => {
                            items.push(item.clone());
                        }
                        Some("function_call")
                            if item
                                .get("call_id")
                                .and_then(Value::as_str)
                                .is_some_and(|cid| block_call_ids.contains(cid)) =>
                        {
                            items.push(item.clone());
                        }
                        _ => {}
                    }
                }
                if !items.is_empty() {
                    return items;
                }
            }

            // 跨模型/跨路由或无原生保序回放数据时，仍严格按 blocks 原序输出，且剥离 reasoning 与 fc_* id 避免配对校验错误。
            let mut items = Vec::with_capacity(blocks.len().max(1));
            for block in blocks {
                match block {
                    ContentBlock::Reasoning { .. } => {}
                    ContentBlock::Text { text } => {
                        if !text.is_empty() {
                            items.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }],
                            }));
                        }
                    }
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        items.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": arguments,
                        }));
                    }
                }
            }

            if items.is_empty() {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    }],
                }));
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
            "role": "developer",
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

fn clamp_message_id(id: &str) -> String {
    if id.len() <= 64 {
        return id.to_string();
    }
    // OpenAI Responses 要求 message.id 不超过 64 字符，超长时转为确定性短哈希。
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("msg_{hash:016x}")
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

fn normalize_message_item(item: &Value, fallback_text: &str) -> Option<Value> {
    let role = item
        .get("role")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("assistant");
    let status = item
        .get("status")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("completed");

    let mut parts = Vec::new();
    if let Some(content_arr) = item.get("content").and_then(Value::as_array) {
        for part in content_arr {
            match part.get("type").and_then(Value::as_str) {
                Some("output_text") => {
                    let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                    let annotations = part
                        .get("annotations")
                        .filter(|v| v.is_array())
                        .cloned()
                        .unwrap_or_else(|| json!([]));
                    parts.push(json!({
                        "type": "output_text",
                        "text": text,
                        "annotations": annotations,
                    }));
                }
                Some("refusal") => {
                    let refusal = part.get("refusal").and_then(Value::as_str).unwrap_or("");
                    parts.push(json!({
                        "type": "refusal",
                        "refusal": refusal,
                    }));
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() && !fallback_text.is_empty() {
        parts.push(json!({
            "type": "output_text",
            "text": fallback_text,
            "annotations": [],
        }));
    }
    if parts.is_empty() {
        return None;
    }

    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), json!("message"));
    obj.insert("role".to_string(), json!(role));
    obj.insert("status".to_string(), json!(status));
    if let Some(id) = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        obj.insert("id".to_string(), json!(clamp_message_id(id)));
    }
    if let Some(phase) = item
        .get("phase")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        obj.insert("phase".to_string(), json!(phase));
    }
    obj.insert("content".to_string(), Value::Array(parts));
    Some(Value::Object(obj))
}

fn normalize_function_call_item(
    item_id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), json!("function_call"));
    if item_id.starts_with("fc_") {
        obj.insert("id".to_string(), json!(item_id));
    }
    obj.insert("call_id".to_string(), json!(call_id));
    obj.insert("name".to_string(), json!(name));
    obj.insert("arguments".to_string(), json!(arguments));
    Value::Object(obj)
}

const UNSUPPORTED_STRICT_SCHEMA_KEYS: &[&str] = &[
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

fn is_structured_schema(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    let has_structured_type = match obj.get("type") {
        Some(Value::String(t)) => t == "object" || t == "array",
        Some(Value::Array(arr)) => arr
            .iter()
            .any(|v| v.as_str().is_some_and(|t| t == "object" || t == "array")),
        _ => false,
    };
    has_structured_type || obj.contains_key("properties") || obj.contains_key("items")
}

fn schema_allows_null(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    match obj.get("type") {
        Some(Value::String(t)) if t == "null" => return true,
        Some(Value::Array(arr)) if arr.iter().any(|v| v.as_str() == Some("null")) => return true,
        _ => {}
    }
    if obj.get("const").is_some_and(Value::is_null) {
        return true;
    }
    if obj
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|arr| arr.iter().any(Value::is_null))
    {
        return true;
    }
    obj.get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|arr| arr.iter().any(schema_allows_null))
}

fn make_json_schema_node_strict(schema: &mut Value) -> bool {
    let Some(obj) = schema.as_object_mut() else {
        return false;
    };
    for key in UNSUPPORTED_STRICT_SCHEMA_KEYS {
        if obj.contains_key(*key) {
            return false;
        }
    }
    if !obj.contains_key("type") && !obj.contains_key("anyOf") && !obj.contains_key("enum") {
        return false;
    }
    if let Some(any_of) = obj.get_mut("anyOf") {
        let Some(variants) = any_of.as_array_mut() else {
            return false;
        };
        if variants.is_empty() {
            return false;
        }
        for variant in variants.iter_mut() {
            if is_structured_schema(variant) || !make_json_schema_node_strict(variant) {
                return false;
            }
        }
    }
    if let Some(items) = obj.get_mut("items")
        && (items.is_array() || !make_json_schema_node_strict(items))
    {
        return false;
    }

    let is_object_schema = obj.get("type").and_then(Value::as_str) == Some("object");
    if obj.contains_key("properties") && !is_object_schema {
        return false;
    }
    if !is_object_schema {
        return true;
    }
    if let Some(add_props) = obj.get("additionalProperties")
        && *add_props != Value::Bool(false)
    {
        return false;
    }

    let required_set: HashSet<String> = match obj.get("required") {
        None => HashSet::new(),
        Some(Value::Array(arr)) => {
            let mut set = HashSet::with_capacity(arr.len());
            for v in arr {
                let Some(name) = v.as_str() else {
                    return false;
                };
                set.insert(name.to_string());
            }
            set
        }
        Some(_) => return false,
    };

    if !obj.contains_key("properties") {
        obj.insert("properties".to_string(), json!({}));
    }
    let Some(props_obj) = obj.get_mut("properties").and_then(Value::as_object_mut) else {
        return false;
    };
    for req_key in &required_set {
        if !props_obj.contains_key(req_key) {
            return false;
        }
    }

    let mut all_prop_names = Vec::with_capacity(props_obj.len());
    for (key, prop) in props_obj.iter_mut() {
        all_prop_names.push(Value::String(key.clone()));
        if !make_json_schema_node_strict(prop) {
            return false;
        }
        if !required_set.contains(key) && !schema_allows_null(prop) {
            let inner = prop.clone();
            *prop = json!({ "anyOf": [inner, { "type": "null" }] });
        }
    }

    obj.insert("required".to_string(), Value::Array(all_prop_names));
    obj.insert("additionalProperties".to_string(), Value::Bool(false));
    true
}

/// 将工具 JSON Schema 转换为 OpenAI Responses strict 子集；若含不支持关键字则返回 `None` 以便回退非 strict。
fn make_strict_json_schema(schema: &Value) -> Option<Value> {
    let mut cloned = schema.clone();
    if !make_json_schema_node_strict(&mut cloned) {
        return None;
    }
    if cloned.get("type").and_then(Value::as_str) != Some("object") {
        return None;
    }
    Some(cloned)
}

fn convert_responses_tool(tool: &ToolSchema, supports_strict_mode: bool) -> Value {
    let (parameters, strict) = if supports_strict_mode {
        match make_strict_json_schema(&tool.parameters) {
            Some(strict_schema) => (strict_schema, true),
            None => (tool.parameters.clone(), false),
        }
    } else {
        (tool.parameters.clone(), false)
    };
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
        "strict": strict,
    })
}

fn effort_str(effort: ReasoningEffort) -> Option<&'static str> {
    match effort {
        ReasoningEffort::Off => Some("none"),
        ReasoningEffort::Minimal => Some("minimal"),
        ReasoningEffort::Low => Some("low"),
        ReasoningEffort::Medium => Some("medium"),
        ReasoningEffort::High => Some("high"),
        ReasoningEffort::XHigh | ReasoningEffort::Max => Some("xhigh"),
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
    let instruction_role = if config.supports_developer_role {
        "developer"
    } else {
        "user"
    };

    let mut input: Vec<Value> = Vec::new();
    if !instructions.is_empty() && !config.use_top_level_instructions {
        // 对顶层 instructions 不进前缀缓存的路由，内联放在 input[0] 才能稳定命中 Prompt Cache。
        input.push(json!({
            "type": "message",
            "role": instruction_role,
            "content": instructions,
        }));
    }
    input.extend(
        options
            .messages
            .iter()
            .filter(|m| !matches!(m, Message::System { .. }))
            .flat_map(|m| to_wire_with_replay(m, &config.provider, model)),
    );

    let mut body = json!({
        "model": model,
        "stream": true,
        "store": false,
        "input": input,
    });
    if !instructions.is_empty() && config.use_top_level_instructions {
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
        body["max_output_tokens"] = json!(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS));
    }
    if let Some(temp) = options.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(requested_effort) = options.reasoning_effort
        && let Some(effort) = effort_str(requested_effort)
    {
        if requested_effort == ReasoningEffort::Off {
            body["reasoning"] = json!({ "effort": effort });
        } else {
            body["reasoning"] = json!({ "effort": effort, "summary": "auto" });
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
    }
    if !options.tools.is_empty() {
        let tools: Vec<Value> = options
            .tools
            .iter()
            .map(|t| convert_responses_tool(t, config.supports_strict_mode))
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SlotKey {
    OutputIndex(usize),
    ItemId(String),
    FallbackReasoning,
    FallbackText,
}

enum ActiveSlot {
    Reasoning {
        stream_idx: usize,
        output_index: Option<usize>,
        item_id: Option<String>,
        started: bool,
        text: String,
    },
    Message {
        stream_idx: usize,
        output_index: Option<usize>,
        item_id: Option<String>,
        phase: Option<String>,
        started: bool,
        text: String,
    },
    FunctionCall {
        stream_idx: usize,
        output_index: Option<usize>,
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl ActiveSlot {
    fn item_id(&self) -> Option<&str> {
        match self {
            Self::Reasoning { item_id, .. } | Self::Message { item_id, .. } => item_id.as_deref(),
            Self::FunctionCall { item_id, .. } => (!item_id.is_empty()).then_some(item_id.as_str()),
        }
    }
}

struct RecordedOutputItem {
    item_id: Option<String>,
    output_index: Option<usize>,
    stream_idx: usize,
    item: Value,
}

fn upsert_recorded_output_item(
    list: &mut Vec<RecordedOutputItem>,
    item_id: Option<&str>,
    output_index: Option<usize>,
    stream_idx: usize,
    item: Value,
) {
    let clean_id = item_id.filter(|s| !s.is_empty());
    if let Some(id) = clean_id
        && let Some(existing) = list
            .iter_mut()
            .find(|entry| entry.item_id.as_deref() == Some(id))
    {
        existing.item = item;
        if existing.output_index.is_none() {
            existing.output_index = output_index;
        }
        return;
    }
    list.push(RecordedOutputItem {
        item_id: clean_id.map(ToString::to_string),
        output_index,
        stream_idx,
        item,
    });
}

fn classify_incomplete_finish(reason: Option<&str>) -> FinishReason {
    let normalized = reason.map(str::trim).filter(|s| !s.is_empty());
    match normalized {
        Some("max_output_tokens") => FinishReason::MaxTokens,
        Some("content_filter") => FinishReason::Error {
            code: LlmErrorCode::ContentRefused,
            message: "response incomplete: content_filter".to_string(),
            status: None,
            retry_after_ms: None,
        },
        Some(other) => {
            let err = LlmError::classify_provider_error(
                None,
                Some("incomplete"),
                Some(other),
                &format!("response incomplete: {other}"),
            );
            FinishReason::Error {
                code: err.code,
                message: err.message,
                status: err.status,
                retry_after_ms: err.retry_after_ms,
            }
        }
        None => FinishReason::Error {
            code: LlmErrorCode::MalformedResponse,
            message: "response incomplete without a provider reason".to_string(),
            status: None,
            retry_after_ms: None,
        },
    }
}

/// OpenAI Responses SSE 解码器。
pub struct ResponsesSseDecoder {
    provider: String,
    model: String,
    response_id: Option<String>,
    next_index: usize,
    active_slots: HashMap<SlotKey, ActiveSlot>,
    item_id_to_slot: HashMap<String, SlotKey>,
    completed_items: HashSet<String>,
    completed_output_items: Vec<RecordedOutputItem>,
    has_tool_calls: bool,
    saw_terminal_response: bool,
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
            active_slots: HashMap::new(),
            item_id_to_slot: HashMap::new(),
            completed_items: HashSet::new(),
            completed_output_items: Vec::new(),
            has_tool_calls: false,
            saw_terminal_response: false,
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

    fn resolve_slot_key(
        &self,
        output_index: Option<usize>,
        item_id: Option<&str>,
        fallback: Option<&SlotKey>,
    ) -> Option<SlotKey> {
        if let Some(oi) = output_index {
            let key = SlotKey::OutputIndex(oi);
            if self.active_slots.contains_key(&key) {
                return Some(key);
            }
        }
        if let Some(id) = item_id.filter(|s| !s.is_empty()) {
            if let Some(mapped) = self.item_id_to_slot.get(id)
                && self.active_slots.contains_key(mapped)
            {
                return Some(mapped.clone());
            }
            let id_key = SlotKey::ItemId(id.to_string());
            if self.active_slots.contains_key(&id_key) {
                return Some(id_key);
            }
        }
        if let Some(fk) = fallback
            && self.active_slots.contains_key(fk)
        {
            return Some(fk.clone());
        }
        None
    }

    fn remove_slot(
        &mut self,
        output_index: Option<usize>,
        item_id: Option<&str>,
        fallback: Option<&SlotKey>,
    ) -> Option<ActiveSlot> {
        let key = self.resolve_slot_key(output_index, item_id, fallback)?;
        let slot = self.active_slots.remove(&key)?;
        if let Some(id) = item_id.filter(|s| !s.is_empty()) {
            self.item_id_to_slot.remove(id);
        }
        if let Some(slot_id) = slot.item_id() {
            self.item_id_to_slot.remove(slot_id);
        }
        Some(slot)
    }

    fn parse_usage(resp_obj: &Value) -> Option<TokenUsage> {
        let u = resp_obj.get("usage")?;
        let raw_input = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
        let cached = u
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cache_write = u
            .pointer("/input_tokens_details/cache_write_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
        let reasoning = u
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_u64);
        let total = u.get("total_tokens").and_then(Value::as_u64);
        Some(TokenUsage {
            input_tokens: raw_input.saturating_sub(cached).saturating_sub(cache_write),
            output_tokens: output,
            total_tokens: total,
            cache_read_tokens: (cached > 0).then_some(cached),
            cache_write_tokens: (cache_write > 0).then_some(cache_write),
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
        let output_index = payload
            .get("output_index")
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        match event_type {
            "response.created" | "response.in_progress" => {
                if let Some(id) = payload.pointer("/response/id").and_then(Value::as_str) {
                    self.response_id = Some(id.to_string());
                }
            }
            "response.output_item.added" => {
                if let Some(item) = payload.get("item") {
                    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty());
                    let slot_key = output_index
                        .map(SlotKey::OutputIndex)
                        .or_else(|| item_id.map(|id| SlotKey::ItemId(id.to_string())));

                    match item_type {
                        "reasoning" => {
                            let key = slot_key.unwrap_or(SlotKey::FallbackReasoning);
                            if !self.active_slots.contains_key(&key) {
                                let stream_idx = self.alloc_index();
                                if let Some(id) = item_id {
                                    self.item_id_to_slot.insert(id.to_string(), key.clone());
                                }
                                self.active_slots.insert(
                                    key,
                                    ActiveSlot::Reasoning {
                                        stream_idx,
                                        output_index,
                                        item_id: item_id.map(ToString::to_string),
                                        started: false,
                                        text: String::new(),
                                    },
                                );
                            }
                        }
                        "message" => {
                            let key = slot_key.unwrap_or(SlotKey::FallbackText);
                            if !self.active_slots.contains_key(&key) {
                                let stream_idx = self.alloc_index();
                                let phase = item
                                    .get("phase")
                                    .and_then(Value::as_str)
                                    .filter(|s| !s.is_empty())
                                    .map(ToString::to_string);
                                if let Some(id) = item_id {
                                    self.item_id_to_slot.insert(id.to_string(), key.clone());
                                }
                                self.active_slots.insert(
                                    key,
                                    ActiveSlot::Message {
                                        stream_idx,
                                        output_index,
                                        item_id: item_id.map(ToString::to_string),
                                        phase,
                                        started: false,
                                        text: String::new(),
                                    },
                                );
                            }
                        }
                        "function_call" => {
                            if item_id.is_some_and(|id| self.completed_items.contains(id)) {
                                return Ok(false);
                            }
                            let Some(key) = slot_key else {
                                return Ok(false);
                            };
                            if self.active_slots.contains_key(&key) {
                                return Ok(false);
                            }
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
                            if (!call_id.is_empty() || !name.is_empty() || !args.is_empty())
                                && !emit(StreamChunk::ToolCallDelta {
                                    index: stream_idx,
                                    id: (!call_id.is_empty()).then_some(call_id.clone()),
                                    name: (!name.is_empty()).then_some(name.clone()),
                                    arguments_delta: args.clone(),
                                })
                            {
                                return Ok(true);
                            }

                            if let Some(id) = item_id {
                                self.item_id_to_slot.insert(id.to_string(), key.clone());
                            }
                            self.active_slots.insert(
                                key,
                                ActiveSlot::FunctionCall {
                                    stream_idx,
                                    output_index,
                                    item_id: item_id.unwrap_or("").to_string(),
                                    call_id,
                                    name,
                                    arguments: args,
                                },
                            );
                        }
                        _ => {}
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    let item_id = payload.get("item_id").and_then(Value::as_str);
                    let key = self
                        .resolve_slot_key(output_index, item_id, Some(&SlotKey::FallbackReasoning))
                        .unwrap_or_else(|| {
                            output_index
                                .map(SlotKey::OutputIndex)
                                .or_else(|| {
                                    item_id
                                        .filter(|s| !s.is_empty())
                                        .map(|s| SlotKey::ItemId(s.to_string()))
                                })
                                .unwrap_or(SlotKey::FallbackReasoning)
                        });
                    if !self.active_slots.contains_key(&key) {
                        let stream_idx = self.alloc_index();
                        if let Some(id) = item_id.filter(|s| !s.is_empty()) {
                            self.item_id_to_slot.insert(id.to_string(), key.clone());
                        }
                        self.active_slots.insert(
                            key.clone(),
                            ActiveSlot::Reasoning {
                                stream_idx,
                                output_index,
                                item_id: item_id.filter(|s| !s.is_empty()).map(ToString::to_string),
                                started: false,
                                text: String::new(),
                            },
                        );
                    }
                    if let Some(ActiveSlot::Reasoning {
                        stream_idx,
                        started,
                        text,
                        ..
                    }) = self.active_slots.get_mut(&key)
                    {
                        let idx = *stream_idx;
                        if !*started {
                            *started = true;
                            if !emit(StreamChunk::BlockStart {
                                index: idx,
                                kind: ContentBlockKind::Reasoning,
                            }) {
                                return Ok(true);
                            }
                        }
                        text.push_str(delta);
                        if !emit(StreamChunk::ReasoningDelta {
                            index: idx,
                            text: delta.to_string(),
                        }) {
                            return Ok(true);
                        }
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                let item_id = payload.get("item_id").and_then(Value::as_str);
                if let Some(key) =
                    self.resolve_slot_key(output_index, item_id, Some(&SlotKey::FallbackReasoning))
                    && let Some(ActiveSlot::Reasoning {
                        stream_idx,
                        started,
                        text,
                        ..
                    }) = self.active_slots.get_mut(&key)
                    && *started
                    && !text.is_empty()
                {
                    let idx = *stream_idx;
                    text.push_str("\n\n");
                    if !emit(StreamChunk::ReasoningDelta {
                        index: idx,
                        text: "\n\n".to_string(),
                    }) {
                        return Ok(true);
                    }
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if let Some(delta) = payload.get("delta").and_then(Value::as_str)
                    && !delta.is_empty()
                {
                    let item_id = payload.get("item_id").and_then(Value::as_str);
                    let key = self
                        .resolve_slot_key(output_index, item_id, Some(&SlotKey::FallbackText))
                        .unwrap_or_else(|| {
                            output_index
                                .map(SlotKey::OutputIndex)
                                .or_else(|| {
                                    item_id
                                        .filter(|s| !s.is_empty())
                                        .map(|s| SlotKey::ItemId(s.to_string()))
                                })
                                .unwrap_or(SlotKey::FallbackText)
                        });
                    if !self.active_slots.contains_key(&key) {
                        let stream_idx = self.alloc_index();
                        if let Some(id) = item_id.filter(|s| !s.is_empty()) {
                            self.item_id_to_slot.insert(id.to_string(), key.clone());
                        }
                        self.active_slots.insert(
                            key.clone(),
                            ActiveSlot::Message {
                                stream_idx,
                                output_index,
                                item_id: item_id.filter(|s| !s.is_empty()).map(ToString::to_string),
                                phase: None,
                                started: false,
                                text: String::new(),
                            },
                        );
                    }
                    if let Some(ActiveSlot::Message {
                        stream_idx,
                        started,
                        text,
                        ..
                    }) = self.active_slots.get_mut(&key)
                    {
                        let idx = *stream_idx;
                        if !*started {
                            *started = true;
                            if !emit(StreamChunk::BlockStart {
                                index: idx,
                                kind: ContentBlockKind::Text,
                            }) {
                                return Ok(true);
                            }
                        }
                        text.push_str(delta);
                        if !emit(StreamChunk::TextDelta {
                            index: idx,
                            text: delta.to_string(),
                        }) {
                            return Ok(true);
                        }
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = payload.get("item_id").and_then(Value::as_str).unwrap_or("");
                if !item_id.is_empty() && self.completed_items.contains(item_id) {
                    return Ok(false);
                }
                if let Some(delta) = payload.get("delta").and_then(Value::as_str) {
                    let fallback =
                        (!item_id.is_empty()).then(|| SlotKey::ItemId(item_id.to_string()));
                    let key = self
                        .resolve_slot_key(
                            output_index,
                            (!item_id.is_empty()).then_some(item_id),
                            fallback.as_ref(),
                        )
                        .or_else(|| output_index.map(SlotKey::OutputIndex))
                        .or(fallback);
                    let Some(key) = key else {
                        return Ok(false);
                    };
                    if !self.active_slots.contains_key(&key) {
                        let stream_idx = self.alloc_index();
                        if !emit(StreamChunk::BlockStart {
                            index: stream_idx,
                            kind: ContentBlockKind::ToolCall,
                        }) {
                            return Ok(true);
                        }
                        if !item_id.is_empty() {
                            self.item_id_to_slot
                                .insert(item_id.to_string(), key.clone());
                        }
                        self.active_slots.insert(
                            key.clone(),
                            ActiveSlot::FunctionCall {
                                stream_idx,
                                output_index,
                                item_id: item_id.to_string(),
                                call_id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                            },
                        );
                    }
                    if let Some(ActiveSlot::FunctionCall {
                        stream_idx,
                        arguments,
                        ..
                    }) = self.active_slots.get_mut(&key)
                    {
                        let idx = *stream_idx;
                        arguments.push_str(delta);
                        if !emit(StreamChunk::ToolCallDelta {
                            index: idx,
                            id: None,
                            name: None,
                            arguments_delta: delta.to_string(),
                        }) {
                            return Ok(true);
                        }
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let item_id = payload.get("item_id").and_then(Value::as_str);
                if item_id.is_some_and(|id| self.completed_items.contains(id)) {
                    return Ok(false);
                }
                if let Some(final_args) = payload.get("arguments").and_then(Value::as_str)
                    && let Some(key) = self.resolve_slot_key(output_index, item_id, None)
                    && let Some(ActiveSlot::FunctionCall { arguments, .. }) =
                        self.active_slots.get_mut(&key)
                {
                    *arguments = final_args.to_string();
                }
            }
            "response.output_item.done" => {
                if let Some(item) = payload.get("item") {
                    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                    let item_id = item.get("id").and_then(Value::as_str);

                    match item_type {
                        "reasoning" => {
                            let slot = self.remove_slot(
                                output_index,
                                item_id,
                                Some(&SlotKey::FallbackReasoning),
                            );
                            let (stream_idx, slot_output_index, started, slot_text) = match slot {
                                Some(ActiveSlot::Reasoning {
                                    stream_idx,
                                    output_index: slot_oi,
                                    started,
                                    text,
                                    ..
                                }) => (stream_idx, output_index.or(slot_oi), started, text),
                                _ => (self.alloc_index(), output_index, false, String::new()),
                            };
                            let summary_text = item
                                .get("summary")
                                .and_then(Value::as_array)
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|s| s.get("text").and_then(Value::as_str))
                                        .collect::<Vec<_>>()
                                        .join("\n\n")
                                })
                                .unwrap_or_default();
                            let content_text = item
                                .get("content")
                                .and_then(Value::as_array)
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|c| c.get("text").and_then(Value::as_str))
                                        .collect::<Vec<_>>()
                                        .join("\n\n")
                                })
                                .unwrap_or_default();
                            let final_text = if !summary_text.is_empty() {
                                summary_text
                            } else if !content_text.is_empty() {
                                content_text
                            } else {
                                slot_text
                            };
                            if started || !final_text.is_empty() {
                                if !started
                                    && !emit(StreamChunk::BlockStart {
                                        index: stream_idx,
                                        kind: ContentBlockKind::Reasoning,
                                    })
                                {
                                    return Ok(true);
                                }
                                if !emit(StreamChunk::BlockEnd {
                                    index: stream_idx,
                                    block: ContentBlock::Reasoning { text: final_text },
                                }) {
                                    return Ok(true);
                                }
                            }
                            if let Some(norm) = normalize_reasoning_item(item) {
                                upsert_recorded_output_item(
                                    &mut self.completed_output_items,
                                    item_id,
                                    slot_output_index,
                                    stream_idx,
                                    norm,
                                );
                            }
                        }
                        "message" => {
                            let slot = self.remove_slot(
                                output_index,
                                item_id,
                                Some(&SlotKey::FallbackText),
                            );
                            let (stream_idx, slot_output_index, started, slot_text, slot_phase) =
                                match slot {
                                    Some(ActiveSlot::Message {
                                        stream_idx,
                                        output_index: slot_oi,
                                        started,
                                        text,
                                        phase,
                                        ..
                                    }) => {
                                        (stream_idx, output_index.or(slot_oi), started, text, phase)
                                    }
                                    _ => (
                                        self.alloc_index(),
                                        output_index,
                                        false,
                                        String::new(),
                                        None,
                                    ),
                                };
                            let item_text = item
                                .get("content")
                                .and_then(Value::as_array)
                                .map(|arr| {
                                    let mut buf = String::new();
                                    for c in arr {
                                        match c.get("type").and_then(Value::as_str) {
                                            Some("output_text") => {
                                                if let Some(t) =
                                                    c.get("text").and_then(Value::as_str)
                                                {
                                                    buf.push_str(t);
                                                }
                                            }
                                            Some("refusal") => {
                                                if let Some(r) =
                                                    c.get("refusal").and_then(Value::as_str)
                                                {
                                                    buf.push_str(r);
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    buf
                                })
                                .unwrap_or_default();
                            let final_text = if !item_text.is_empty() {
                                item_text
                            } else {
                                slot_text
                            };
                            if started || !final_text.is_empty() {
                                if !started
                                    && !emit(StreamChunk::BlockStart {
                                        index: stream_idx,
                                        kind: ContentBlockKind::Text,
                                    })
                                {
                                    return Ok(true);
                                }
                                if !emit(StreamChunk::BlockEnd {
                                    index: stream_idx,
                                    block: ContentBlock::Text {
                                        text: final_text.clone(),
                                    },
                                }) {
                                    return Ok(true);
                                }
                            }
                            if let Some(mut norm) = normalize_message_item(item, &final_text) {
                                if norm.get("phase").is_none()
                                    && let Some(p) = slot_phase
                                    && let Some(obj) = norm.as_object_mut()
                                {
                                    obj.insert("phase".to_string(), json!(p));
                                }
                                upsert_recorded_output_item(
                                    &mut self.completed_output_items,
                                    item_id,
                                    slot_output_index,
                                    stream_idx,
                                    norm,
                                );
                            }
                        }
                        "function_call" => {
                            let id_str = item_id.unwrap_or("");
                            if !id_str.is_empty()
                                && !self.completed_items.insert(id_str.to_string())
                            {
                                return Ok(false);
                            }
                            let slot = self.remove_slot(
                                output_index,
                                (!id_str.is_empty()).then_some(id_str),
                                None,
                            );
                            let (
                                stream_idx,
                                slot_output_index,
                                had_slot,
                                mut call_id,
                                mut name,
                                mut arguments,
                            ) = match slot {
                                Some(ActiveSlot::FunctionCall {
                                    stream_idx,
                                    output_index: slot_oi,
                                    call_id,
                                    name,
                                    arguments,
                                    ..
                                }) => (
                                    stream_idx,
                                    output_index.or(slot_oi),
                                    true,
                                    call_id,
                                    name,
                                    arguments,
                                ),
                                _ => (
                                    self.alloc_index(),
                                    output_index,
                                    false,
                                    String::new(),
                                    String::new(),
                                    String::new(),
                                ),
                            };
                            if let Some(cid) = item.get("call_id").and_then(Value::as_str)
                                && !cid.is_empty()
                            {
                                call_id = cid.to_string();
                            }
                            if let Some(n) = item.get("name").and_then(Value::as_str)
                                && !n.is_empty()
                            {
                                name = n.to_string();
                            }
                            if let Some(args) = item.get("arguments").and_then(Value::as_str)
                                && !args.is_empty()
                            {
                                arguments = args.to_string();
                            }
                            if !had_slot
                                && !emit(StreamChunk::BlockStart {
                                    index: stream_idx,
                                    kind: ContentBlockKind::ToolCall,
                                })
                            {
                                return Ok(true);
                            }
                            self.has_tool_calls = true;
                            if !emit(StreamChunk::BlockEnd {
                                index: stream_idx,
                                block: ContentBlock::ToolCall {
                                    id: call_id.clone(),
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                },
                            }) {
                                return Ok(true);
                            }
                            let norm =
                                normalize_function_call_item(id_str, &call_id, &name, &arguments);
                            upsert_recorded_output_item(
                                &mut self.completed_output_items,
                                (!id_str.is_empty()).then_some(id_str),
                                slot_output_index,
                                stream_idx,
                                norm,
                            );
                        }
                        _ => {}
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                self.saw_terminal_response = true;
                if let Some(resp_obj) = payload.get("response") {
                    if let Some(id) = resp_obj.get("id").and_then(Value::as_str) {
                        self.response_id = Some(id.to_string());
                    }
                    if let Some(output) = resp_obj.get("output").and_then(Value::as_array) {
                        for (idx, item) in output.iter().enumerate() {
                            if has_encrypted_reasoning(item)
                                && let Some(norm) = normalize_reasoning_item(item)
                            {
                                let stream_idx = self.alloc_index();
                                upsert_recorded_output_item(
                                    &mut self.completed_output_items,
                                    item.get("id").and_then(Value::as_str),
                                    Some(idx),
                                    stream_idx,
                                    norm,
                                );
                            }
                        }
                    }
                    if let Some(u) = Self::parse_usage(resp_obj) {
                        self.usage = Some(u);
                    }
                    let status = resp_obj.get("status").and_then(Value::as_str);
                    let incomplete_reason = resp_obj
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str);
                    if event_type == "response.incomplete" || status == Some("incomplete") {
                        self.finish_reason = Some(classify_incomplete_finish(incomplete_reason));
                    } else if status == Some("cancelled") {
                        self.finish_reason = Some(FinishReason::Aborted);
                    } else if status == Some("failed") {
                        let err_obj = resp_obj.get("error").unwrap_or(resp_obj);
                        let msg = err_obj
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("responses stream failed");
                        let ecode = err_obj.get("code").and_then(Value::as_str);
                        let etype = err_obj.get("type").and_then(Value::as_str);
                        let err = LlmError::classify_provider_error(None, etype, ecode, msg);
                        self.finish_reason = Some(FinishReason::Error {
                            code: err.code,
                            message: err.message,
                            status: err.status,
                            retry_after_ms: err.retry_after_ms,
                        });
                    } else if incomplete_reason == Some("max_output_tokens") {
                        self.finish_reason = Some(FinishReason::MaxTokens);
                    }
                } else if event_type == "response.incomplete" {
                    self.finish_reason = Some(classify_incomplete_finish(None));
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

        let mut has_unfinished_calls = false;
        let active: Vec<ActiveSlot> = self.active_slots.drain().map(|(_, s)| s).collect();
        self.item_id_to_slot.clear();
        for slot in active {
            match slot {
                ActiveSlot::FunctionCall { .. } => {
                    // 未收到 output_item.done 的工具调用绝不发射 BlockEnd，防止参数截断或串号。
                    has_unfinished_calls = true;
                }
                ActiveSlot::Message {
                    stream_idx,
                    output_index,
                    item_id,
                    phase,
                    text,
                    ..
                } => {
                    if self.saw_terminal_response && !text.is_empty() {
                        let raw_msg = json!({
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "id": item_id,
                            "phase": phase,
                            "content": [{
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }],
                        });
                        if let Some(norm) = normalize_message_item(&raw_msg, &text) {
                            upsert_recorded_output_item(
                                &mut self.completed_output_items,
                                item_id.as_deref(),
                                output_index,
                                stream_idx,
                                norm,
                            );
                        }
                    }
                }
                ActiveSlot::Reasoning { .. } => {}
            }
        }

        if has_unfinished_calls
            && !matches!(
                self.finish_reason,
                Some(FinishReason::MaxTokens | FinishReason::Error { .. } | FinishReason::Aborted)
            )
        {
            return Err(LlmError::malformed(
                "responses stream ended with unfinished function_call",
            ));
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

        self.completed_output_items.sort_by_key(|entry| {
            (
                entry.output_index.unwrap_or(entry.stream_idx),
                entry.stream_idx,
            )
        });
        let drop_tool_calls = matches!(reason, FinishReason::MaxTokens);
        let output_items: Vec<Value> = self
            .completed_output_items
            .iter()
            .filter(
                |entry| match entry.item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => has_encrypted_reasoning(&entry.item),
                    Some("function_call") => !drop_tool_calls,
                    Some("message") => true,
                    _ => false,
                },
            )
            .map(|entry| entry.item.clone())
            .collect();

        let mut replay_response = json!({ "id": self.response_id });
        if let Some(obj) = replay_response.as_object_mut()
            && !output_items.is_empty()
        {
            obj.insert("outputItems".to_string(), Value::Array(output_items));
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
        assert_eq!(replay.response["outputItems"][0]["type"], "reasoning");
        assert_eq!(
            replay.response["outputItems"][0]["encrypted_content"],
            "enc_v1_final"
        );
        assert_eq!(replay.response["outputItems"][1]["type"], "function_call");
        assert_eq!(replay.response["outputItems"][1]["id"], "fc_1");

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
            req_with_effort["reasoning"],
            json!({ "effort": "medium", "summary": "auto" })
        );
        assert_eq!(
            req_with_effort["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    fn run_sse_decoder(raw: &str) -> (BlockAssembler, Vec<StreamChunk>, Result<(), LlmError>) {
        let mut parser = SseParser::new();
        let events = parser.push(raw.as_bytes(), || {});
        let mut decoder = ResponsesSseDecoder::new("responses", "gpt-5");
        let mut asm = BlockAssembler::new();
        let mut chunks = Vec::new();
        for ev in events {
            match decoder.on_event(ev, &mut |chunk| {
                chunks.push(chunk.clone());
                asm.push(chunk);
                true
            }) {
                Ok(true) => break,
                Ok(false) => {}
                Err(err) => return (asm, chunks, Err(err)),
            }
        }
        let finish_res = decoder.on_finish(&mut |chunk| {
            chunks.push(chunk.clone());
            asm.push(chunk);
            true
        });
        (asm, chunks, finish_res)
    }

    #[test]
    fn wire_request_body_handles_developer_role_reasoning_clamping_and_strict_mode() {
        let tools = vec![
            ToolSchema {
                name: "read_file".into(),
                description: "Read a file".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["path"]
                }),
            },
            ToolSchema {
                name: "complex_mcp".into(),
                description: "MCP tool with unsupported strict keyword".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "oneOf": [{ "type": "string" }, { "type": "integer" }] }
                    },
                    "required": ["query"]
                }),
            },
        ];

        // 1. 默认：developer 内联进 input[0]、无顶层 instructions、summary: "auto"、max_output_tokens >= 16、strict: false
        let default_body = build_request_body(
            &ResponsesConfig::default(),
            &GenerateOptions {
                system: Some("sys prompt".into()),
                messages: vec![Message::system("extra dev"), Message::user("hello")],
                max_tokens: Some(4),
                reasoning_effort: Some(ReasoningEffort::Medium),
                tools: tools.clone(),
                ..Default::default()
            },
            "gpt-5",
        );
        assert!(default_body.get("instructions").is_none());
        assert_eq!(default_body["input"][0]["role"], "developer");
        assert_eq!(
            default_body["input"][0]["content"],
            "sys prompt\n\nextra dev"
        );
        assert_eq!(default_body["input"][1]["role"], "user");
        assert_eq!(default_body["max_output_tokens"], 16);
        assert_eq!(
            default_body["reasoning"],
            json!({ "effort": "medium", "summary": "auto" })
        );
        assert_eq!(
            default_body["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert_eq!(default_body["tools"][0]["strict"], false);
        assert_eq!(default_body["tools"][1]["strict"], false);

        // 2. supports_developer_role=false 降级为 user；Off 不发 summary 与 include；supports_strict_mode=true 自动转换兼容工具
        let strict_no_dev_body = build_request_body(
            &ResponsesConfig {
                supports_developer_role: false,
                supports_strict_mode: true,
                ..Default::default()
            },
            &GenerateOptions {
                messages: vec![Message::system("sys"), Message::user("hi")],
                max_tokens: Some(256),
                reasoning_effort: Some(ReasoningEffort::Off),
                tools: tools.clone(),
                ..Default::default()
            },
            "gpt-5",
        );
        assert!(strict_no_dev_body.get("instructions").is_none());
        assert_eq!(strict_no_dev_body["input"][0]["role"], "user");
        assert_eq!(strict_no_dev_body["input"][0]["content"], "sys");
        assert_eq!(strict_no_dev_body["max_output_tokens"], 256);
        assert_eq!(strict_no_dev_body["reasoning"], json!({ "effort": "none" }));
        assert!(strict_no_dev_body.get("include").is_none());
        assert_eq!(strict_no_dev_body["tools"][0]["strict"], true);
        assert_eq!(
            strict_no_dev_body["tools"][0]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(
            strict_no_dev_body["tools"][0]["parameters"]["properties"]["limit"],
            json!({ "anyOf": [{ "type": "integer" }, { "type": "null" }] })
        );
        assert_eq!(
            strict_no_dev_body["tools"][0]["parameters"]["required"],
            json!(["limit", "path"])
        );
        // 含 oneOf 的工具安全回退为 strict: false
        assert_eq!(strict_no_dev_body["tools"][1]["strict"], false);

        // 3. use_top_level_instructions=true + XHigh/Max => xhigh
        let top_instr_body = build_request_body(
            &ResponsesConfig {
                use_top_level_instructions: true,
                ..Default::default()
            },
            &GenerateOptions {
                messages: vec![Message::system("top sys"), Message::user("go")],
                reasoning_effort: Some(ReasoningEffort::XHigh),
                ..Default::default()
            },
            "gpt-5",
        );
        assert_eq!(top_instr_body["instructions"], "top sys");
        assert_eq!(top_instr_body["input"].as_array().unwrap().len(), 1);
        assert_eq!(top_instr_body["input"][0]["role"], "user");
        assert_eq!(
            top_instr_body["reasoning"],
            json!({ "effort": "xhigh", "summary": "auto" })
        );
    }

    #[test]
    fn preserves_interleaved_output_order_and_message_metadata_in_replay() {
        let long_msg_id = format!("msg_{}", "a".repeat(80));
        let raw = format!(
            concat!(
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}}}\n\n",
                "data: {{\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"item_id\":\"rs_1\",\"delta\":\"think 1\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"think 1\"}}]}}}}\n\n",
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{{\"id\":\"{long_id}\",\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"commentary\",\"content\":[]}}}}\n\n",
                "data: {{\"type\":\"response.output_text.delta\",\"output_index\":1,\"item_id\":\"{long_id}\",\"delta\":\"Checking file...\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{{\"id\":\"{long_id}\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"phase\":\"commentary\",\"content\":[{{\"type\":\"output_text\",\"text\":\"Checking file...\",\"annotations\":[]}}]}}}}\n\n",
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"\"}}}}\n\n",
                "data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"item_id\":\"fc_1\",\"delta\":\"{{\\\"p\\\":1}}\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\"{{\\\"p\\\":1}}\"}}}}\n\n",
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":3,\"item\":{{\"id\":\"rs_2\",\"type\":\"reasoning\",\"summary\":[]}}}}\n\n",
                "data: {{\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":3,\"item_id\":\"rs_2\",\"delta\":\"think 2\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{{\"id\":\"rs_2\",\"type\":\"reasoning\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"think 2\"}}],\"encrypted_content\":\"enc_2\"}}}}\n\n",
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":4,\"item\":{{\"id\":\"msg_2\",\"type\":\"message\",\"role\":\"assistant\",\"phase\":\"final_answer\",\"content\":[]}}}}\n\n",
                "data: {{\"type\":\"response.output_text.delta\",\"output_index\":4,\"item_id\":\"msg_2\",\"delta\":\"Done.\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":4,\"item\":{{\"id\":\"msg_2\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"phase\":\"final_answer\",\"content\":[{{\"type\":\"output_text\",\"text\":\"Done.\",\"annotations\":[]}}]}}}}\n\n",
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":5,\"item\":{{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2\",\"name\":\"bash\",\"arguments\":\"\"}}}}\n\n",
                "data: {{\"type\":\"response.function_call_arguments.delta\",\"output_index\":5,\"item_id\":\"fc_2\",\"delta\":\"{{\\\"cmd\\\":\\\"ls\\\"}}\"}}\n\n",
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":5,\"item\":{{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2\",\"name\":\"bash\",\"arguments\":\"{{\\\"cmd\\\":\\\"ls\\\"}}\"}}}}\n\n",
                "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_seq\",\"status\":\"completed\",\"output\":[{{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{{\"type\":\"summary_text\",\"text\":\"think 1\"}}],\"encrypted_content\":\"enc_1\"}}],\"usage\":{{\"input_tokens\":100,\"output_tokens\":40,\"total_tokens\":140,\"input_tokens_details\":{{\"cached_tokens\":60,\"cache_write_tokens\":15}}}}}}}}\n\n"
            ),
            long_id = long_msg_id
        );

        let (asm, _chunks, res) = run_sse_decoder(&raw);
        res.unwrap();

        let blocks = asm.blocks();
        assert_eq!(blocks.len(), 6);
        assert!(matches!(&blocks[0], ContentBlock::Reasoning { text } if text == "think 1"));
        assert!(matches!(&blocks[1], ContentBlock::Text { text } if text == "Checking file..."));
        assert!(matches!(&blocks[2], ContentBlock::ToolCall { id, .. } if id == "call_1"));
        assert!(matches!(&blocks[3], ContentBlock::Reasoning { text } if text == "think 2"));
        assert!(matches!(&blocks[4], ContentBlock::Text { text } if text == "Done."));
        assert!(matches!(&blocks[5], ContentBlock::ToolCall { id, .. } if id == "call_2"));

        let usage = asm.usage().unwrap();
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.cache_read_tokens, Some(60));
        assert_eq!(usage.cache_write_tokens, Some(15));
        assert_eq!(usage.prompt_tokens(), 100);

        let replay = asm.replay().unwrap();
        let out_items = replay.response["outputItems"].as_array().unwrap().clone();
        assert_eq!(out_items.len(), 6);
        assert_eq!(out_items[0]["id"], "rs_1");
        assert_eq!(out_items[0]["encrypted_content"], "enc_1");
        assert_eq!(out_items[1]["type"], "message");
        assert_eq!(out_items[1]["phase"], "commentary");
        assert_eq!(out_items[1]["status"], "completed");
        assert!(out_items[1]["id"].as_str().unwrap().len() <= 64);
        assert_eq!(out_items[1]["content"][0]["type"], "output_text");
        assert_eq!(out_items[2]["id"], "fc_1");
        assert_eq!(out_items[3]["id"], "rs_2");
        assert_eq!(out_items[4]["id"], "msg_2");
        assert_eq!(out_items[4]["phase"], "final_answer");
        assert_eq!(out_items[5]["id"], "fc_2");

        let assistant = Message::assistant(blocks, Some(replay));
        let same_wire = to_wire_with_replay(&assistant, "responses", "gpt-5");
        assert_eq!(same_wire, *out_items);

        // 跨模型仍保序输出 message/function_call，但剥离 reasoning 与 fc_* id
        let diff_wire = to_wire_with_replay(&assistant, "responses", "gpt-4.1");
        assert_eq!(diff_wire.len(), 4);
        assert_eq!(diff_wire[0]["type"], "message");
        assert_eq!(diff_wire[0]["content"][0]["text"], "Checking file...");
        assert_eq!(diff_wire[1]["type"], "function_call");
        assert_eq!(diff_wire[1]["call_id"], "call_1");
        assert!(diff_wire[1].get("id").is_none());
        assert_eq!(diff_wire[2]["type"], "message");
        assert_eq!(diff_wire[2]["content"][0]["text"], "Done.");
        assert_eq!(diff_wire[3]["type"], "function_call");
        assert_eq!(diff_wire[3]["call_id"], "call_2");
        assert!(diff_wire[3].get("id").is_none());
    }

    #[test]
    fn classifies_incomplete_reasons_and_rejects_unfinished_tool_calls() {
        let max_tok_raw = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_cut\",\"type\":\"function_call\",\"call_id\":\"call_cut\",\"name\":\"write\",\"arguments\":\"{\\\"path\\\":\"}}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_inc1\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
        );
        let (asm_max, chunks_max, res_max) = run_sse_decoder(max_tok_raw);
        res_max.unwrap();
        assert_eq!(asm_max.finish(), FinishReason::MaxTokens);
        assert!(asm_max.tool_calls().is_empty());
        assert!(!chunks_max.iter().any(|c| matches!(
            c,
            StreamChunk::BlockEnd {
                block: ContentBlock::ToolCall { .. },
                ..
            }
        )));

        let filter_raw = "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_inc2\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n";
        let (asm_filter, _chunks_filter, res_filter) = run_sse_decoder(filter_raw);
        res_filter.unwrap();
        assert!(matches!(
            asm_filter.finish(),
            FinishReason::Error {
                code: LlmErrorCode::ContentRefused,
                ..
            }
        ));

        let missing_reason_raw = "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_inc3\",\"status\":\"incomplete\"}}\n\n";
        let (asm_missing, _chunks_missing, res_missing) = run_sse_decoder(missing_reason_raw);
        res_missing.unwrap();
        assert!(matches!(
            asm_missing.finish(),
            FinishReason::Error {
                code: LlmErrorCode::MalformedResponse,
                ..
            }
        ));

        // 异常结束未收到 output_item.done 的 function_call 绝不发射 BlockEnd/Finish，直接报 MalformedResponse
        let unclosed_call_raw = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_unclosed\",\"type\":\"function_call\",\"call_id\":\"call_u\",\"name\":\"bash\",\"arguments\":\"\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"item_id\":\"fc_unclosed\",\"delta\":\"{\\\"cmd\\\":\"}\n\n",
            "data: [DONE]\n\n",
        );
        let (_asm_unclosed, chunks_unclosed, res_unclosed) = run_sse_decoder(unclosed_call_raw);
        assert!(matches!(
            res_unclosed,
            Err(err) if err.code == LlmErrorCode::MalformedResponse
        ));
        assert!(!chunks_unclosed.iter().any(|c| matches!(
            c,
            StreamChunk::BlockEnd {
                block: ContentBlock::ToolCall { .. },
                ..
            } | StreamChunk::Finish { .. }
        )));
    }
}
