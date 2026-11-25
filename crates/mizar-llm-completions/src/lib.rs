//! OpenAI Chat Completions 兼容协议适配器。
//!
//! 通过 `CompletionsPlugin` 与 `CompletionsCompat` 方言开关接入任何 Chat Completions
//! 协议端点（OpenAI / DeepSeek / OpenRouter / vLLM / Qwen 等）。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, CacheRetention, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions,
    LlmAdapter, LlmContextExt, LlmError, LlmErrorCode, Message, ModelInfo, ProviderInfo,
    ReasoningEffort, ReplayEnvelope, SessionAffinityMode, SseEvent, StreamChunk, TokenUsage,
    clamp_openai_prompt_cache_key, detect_session_affinity_mode, push_session_affinity_headers,
    resolve_cache_retention,
};
use mizar_llm_http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, resolve_credential,
    shared_http_client, spawn_sse_stream, trim_base_url,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const PROTOCOL_COMPLETIONS: &str = "openai-completions";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ThinkingFormat {
    #[default]
    None,
    DeepSeek,
    OpenAI,
    OpenRouter,
    Qwen,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MaxTokensField {
    #[default]
    #[serde(alias = "max_tokens")]
    MaxTokens,
    #[serde(alias = "max_completion_tokens")]
    MaxCompletionTokens,
}

/// 兼容网关方言开关（对应 `llm-pi-ai` 的 `OpenAICompletionsCompat`）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "camelCase")]
pub struct CompletionsCompat {
    pub thinking_format: ThinkingFormat,
    pub max_tokens_field: MaxTokensField,
    #[serde(alias = "supportsUsageInStreaming")]
    pub include_usage: bool,
    pub send_session_affinity_headers: Option<bool>,
    pub session_affinity_mode: Option<SessionAffinityMode>,
    pub supports_long_cache_retention: Option<bool>,
    pub supports_prompt_cache_key: Option<bool>,
}

impl CompletionsCompat {
    pub fn deepseek() -> Self {
        Self {
            thinking_format: ThinkingFormat::DeepSeek,
            max_tokens_field: MaxTokensField::MaxTokens,
            include_usage: true,
            ..Default::default()
        }
    }

    pub fn openai() -> Self {
        Self {
            thinking_format: ThinkingFormat::OpenAI,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            include_usage: true,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionsConfig {
    pub provider: String,
    pub display_name: Option<String>,
    pub api_key: Option<String>,
    /// 若配置了环境变量名且 `api_key` 为空，则在每次请求时动态读取。
    pub api_key_env: Option<String>,
    pub base_url: String,
    pub model: String,
    pub compat: CompletionsCompat,
    pub headers: Vec<(String, String)>,
    pub default_cache_retention: Option<CacheRetention>,
    pub models: Vec<ModelInfo>,
    pub idle_timeout: Duration,
}

impl Default for CompletionsConfig {
    fn default() -> Self {
        Self {
            provider: "completions".to_string(),
            display_name: Some("OpenAI Completions".to_string()),
            api_key: None,
            api_key_env: None,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            compat: CompletionsCompat::openai(),
            headers: Vec::new(),
            default_cache_retention: None,
            models: Vec::new(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

impl CompletionsConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Self::default()
        }
    }
}

/// 统一消息格式 → OpenAI Chat Completions 格式（带跨协议回放降级）。
pub fn to_wire(message: &Message, provider: &str, model: &str) -> Value {
    match message {
        Message::System { content } => json!({ "role": "system", "content": content }),
        Message::User { content, images } => {
            if images.is_empty() {
                json!({ "role": "user", "content": content })
            } else {
                let mut parts = Vec::with_capacity(images.len() + 1);
                if !content.is_empty() {
                    parts.push(json!({ "type": "text", "text": content }));
                }
                for img in images {
                    parts.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": img.data_url(),
                        },
                    }));
                }
                json!({ "role": "user", "content": parts })
            }
        }
        Message::Tool {
            content,
            tool_call_id,
            ..
        } => json!({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }),
        Message::Assistant { blocks, replay } => {
            let text = message.text_content();
            let reasoning = message.reasoning_content();
            let calls: Vec<Value> = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    })),
                    _ => None,
                })
                .collect();

            let mut obj = serde_json::Map::new();
            obj.insert("role".to_string(), json!("assistant"));
            if !calls.is_empty() {
                obj.insert(
                    "content".to_string(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        json!(text)
                    },
                );
                obj.insert("tool_calls".to_string(), Value::Array(calls));
            } else {
                obj.insert("content".to_string(), json!(text));
            }

            // 仅当由同协议/同路由/同模型生成时保留 reasoning_content 供同轮工具续传。
            if !reasoning.is_empty()
                && replay
                    .as_ref()
                    .is_some_and(|r| r.matches(PROTOCOL_COMPLETIONS, provider, model))
            {
                obj.insert("reasoning_content".to_string(), json!(reasoning));
            }

            Value::Object(obj)
        }
    }
}

fn effort_str(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Off => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

pub fn build_request_body(
    config: &CompletionsConfig,
    options: &GenerateOptions,
    model: &str,
) -> Value {
    let mut messages = Vec::new();
    if let Some(sys) = &options.system
        && !sys.is_empty()
    {
        messages.push(json!({ "role": "system", "content": sys }));
    }
    for m in &options.messages {
        messages.push(to_wire(m, &config.provider, model));
    }

    let mut body = json!({
        "model": model,
        "stream": true,
        "messages": messages,
    });

    let retention =
        resolve_cache_retention(options.cache_retention.or(config.default_cache_retention));
    let supports_long = config.compat.supports_long_cache_retention.unwrap_or(true);
    let allow_cache_key = match config.compat.supports_prompt_cache_key {
        Some(flag) => flag,
        None => {
            config
                .base_url
                .to_ascii_lowercase()
                .contains("api.openai.com")
                || (retention == CacheRetention::Long && supports_long)
        }
    };
    if retention != CacheRetention::None
        && allow_cache_key
        && let Some(key) = options
            .session_id
            .as_deref()
            .and_then(clamp_openai_prompt_cache_key)
    {
        body["prompt_cache_key"] = Value::String(key);
    }
    if retention == CacheRetention::Long && supports_long {
        body["prompt_cache_retention"] = json!("24h");
    }

    if config.compat.include_usage {
        body["stream_options"] = json!({ "include_usage": true });
    }
    if let Some(temp) = options.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(max_tokens) = options.max_tokens {
        let key = match config.compat.max_tokens_field {
            MaxTokensField::MaxTokens => "max_tokens",
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
        };
        body[key] = json!(max_tokens);
    }
    if let Some(effort) = options.reasoning_effort {
        match config.compat.thinking_format {
            ThinkingFormat::None => {}
            ThinkingFormat::DeepSeek => {
                if effort == ReasoningEffort::Off {
                    body["thinking"] = json!({ "type": "disabled" });
                } else {
                    body["thinking"] = json!({ "type": "enabled" });
                }
            }
            ThinkingFormat::OpenAI => {
                if effort != ReasoningEffort::Off {
                    body["reasoning_effort"] = json!(effort_str(effort));
                }
            }
            ThinkingFormat::OpenRouter => {
                if effort != ReasoningEffort::Off {
                    body["reasoning"] = json!({ "effort": effort_str(effort) });
                }
            }
            ThinkingFormat::Qwen => {
                body["enable_thinking"] = json!(effort != ReasoningEffort::Off);
            }
        }
    }
    if !options.tools.is_empty() {
        let tools: Vec<Value> = options
            .tools
            .iter()
            .map(|t| json!({ "type": "function", "function": t }))
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<WireUsage>,
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<WirePromptDetails>,
    completion_tokens_details: Option<WireCompletionDetails>,
}

#[derive(Debug, Deserialize)]
struct WirePromptDetails {
    cached_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct WireCompletionDetails {
    reasoning_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: Option<ChunkDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ChunkToolCallPart>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCallPart {
    index: usize,
    id: Option<String>,
    function: Option<ChunkFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct ChunkFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

enum CompletionsBlockKind {
    Text(String),
    Reasoning(String),
    ToolCall {
        id: String,
        name: String,
        args: String,
    },
}

struct OpenBlock {
    index: usize,
    kind: CompletionsBlockKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompletionsEventPhase {
    Streaming,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompletionsFinishPhase {
    Pending,
    Emitted,
}

/// OpenAI Chat Completions SSE 解码器。
pub struct CompletionsSseDecoder {
    provider: String,
    model: String,
    response_id: Option<String>,
    response_model: Option<String>,
    next_index: usize,
    reasoning_idx: Option<usize>,
    text_idx: Option<usize>,
    tool_map: HashMap<usize, usize>,
    blocks: Vec<OpenBlock>,
    usage: Option<TokenUsage>,
    finish_reason: Option<String>,
    event_phase: CompletionsEventPhase,
    finish_phase: CompletionsFinishPhase,
}

impl CompletionsSseDecoder {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            response_id: None,
            response_model: None,
            next_index: 0,
            reasoning_idx: None,
            text_idx: None,
            tool_map: HashMap::new(),
            blocks: Vec::new(),
            usage: None,
            finish_reason: None,
            event_phase: CompletionsEventPhase::Streaming,
            finish_phase: CompletionsFinishPhase::Pending,
        }
    }

    fn alloc_block(
        &mut self,
        kind: ContentBlockKind,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Option<usize> {
        let idx = self.next_index;
        self.next_index += 1;
        let state = match kind {
            ContentBlockKind::Text => CompletionsBlockKind::Text(String::new()),
            ContentBlockKind::Reasoning => CompletionsBlockKind::Reasoning(String::new()),
            ContentBlockKind::ToolCall => CompletionsBlockKind::ToolCall {
                id: String::new(),
                name: String::new(),
                args: String::new(),
            },
        };
        self.blocks.push(OpenBlock {
            index: idx,
            kind: state,
        });
        if !emit(StreamChunk::BlockStart { index: idx, kind }) {
            return None;
        }
        Some(idx)
    }

    fn map_finish_reason(&self) -> FinishReason {
        match self.finish_reason.as_deref() {
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::MaxTokens,
            Some("content_filter") => FinishReason::Error {
                code: LlmErrorCode::ContentRefused,
                message: "model response blocked by content_filter".to_string(),
                status: None,
                retry_after_ms: None,
            },
            _ => {
                if self
                    .blocks
                    .iter()
                    .any(|b| matches!(b.kind, CompletionsBlockKind::ToolCall { .. }))
                {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                }
            }
        }
    }
}

impl SseStreamDecoder for CompletionsSseDecoder {
    fn on_event(
        &mut self,
        event: SseEvent,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Result<bool, LlmError> {
        if self.event_phase == CompletionsEventPhase::Done {
            return Ok(true);
        }
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(false);
        }
        if data == "[DONE]" {
            self.event_phase = CompletionsEventPhase::Done;
            return Ok(true);
        }

        let chunk: ChatCompletionChunk = serde_json::from_str(data)
            .map_err(|e| LlmError::malformed(format!("invalid completions SSE json: {e}")))?;

        if let Some(err_val) = chunk.error {
            let msg = err_val
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("completions stream error");
            let etype = err_val.get("type").and_then(Value::as_str);
            let ecode = err_val.get("code").and_then(Value::as_str);
            return Err(LlmError::classify_provider_error(None, etype, ecode, msg));
        }

        if let Some(id) = chunk.id {
            self.response_id = Some(id);
        }
        if let Some(model) = chunk.model {
            self.response_model = Some(model);
        }

        if let Some(u) = chunk.usage {
            let prompt = u.prompt_tokens.unwrap_or(0);
            let cached = u
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .unwrap_or(0);
            let output = u.completion_tokens.unwrap_or(0);
            let reasoning = u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens);
            self.usage = Some(TokenUsage {
                input_tokens: prompt.saturating_sub(cached),
                output_tokens: output,
                total_tokens: u.total_tokens,
                cache_read_tokens: (cached > 0).then_some(cached),
                cache_write_tokens: None,
                reasoning_tokens: reasoning,
            });
        }

        let Some(choice) = chunk.choices.into_iter().next() else {
            return Ok(false);
        };
        if let Some(reason) = choice.finish_reason {
            self.finish_reason = Some(reason);
        }
        let Some(delta) = choice.delta else {
            return Ok(false);
        };

        if let Some(reasoning) = delta.reasoning_content.or(delta.reasoning)
            && !reasoning.is_empty()
        {
            let idx = match self.reasoning_idx {
                Some(i) => i,
                None => {
                    let Some(i) = self.alloc_block(ContentBlockKind::Reasoning, emit) else {
                        return Ok(true);
                    };
                    self.reasoning_idx = Some(i);
                    i
                }
            };
            if let CompletionsBlockKind::Reasoning(ref mut buf) = self.blocks[idx].kind {
                buf.push_str(&reasoning);
            }
            if !emit(StreamChunk::ReasoningDelta {
                index: idx,
                text: reasoning,
            }) {
                return Ok(true);
            }
        }

        if let Some(text) = delta.content
            && !text.is_empty()
        {
            let idx = match self.text_idx {
                Some(i) => i,
                None => {
                    let Some(i) = self.alloc_block(ContentBlockKind::Text, emit) else {
                        return Ok(true);
                    };
                    self.text_idx = Some(i);
                    i
                }
            };
            if let CompletionsBlockKind::Text(ref mut buf) = self.blocks[idx].kind {
                buf.push_str(&text);
            }
            if !emit(StreamChunk::TextDelta { index: idx, text }) {
                return Ok(true);
            }
        }

        for part in delta.tool_calls {
            let idx = match self.tool_map.get(&part.index).copied() {
                Some(i) => i,
                None => {
                    let Some(i) = self.alloc_block(ContentBlockKind::ToolCall, emit) else {
                        return Ok(true);
                    };
                    self.tool_map.insert(part.index, i);
                    i
                }
            };
            let id_delta = part.id.clone();
            let mut name_delta = None;
            let mut args_delta = String::new();
            if let CompletionsBlockKind::ToolCall {
                id: ref mut tool_id,
                name: ref mut tool_name,
                args: ref mut tool_args,
            } = self.blocks[idx].kind
            {
                if let Some(id) = part.id
                    && !id.is_empty()
                {
                    *tool_id = id;
                }
                if let Some(func) = part.function {
                    if let Some(name) = func.name
                        && !name.is_empty()
                    {
                        tool_name.push_str(&name);
                        name_delta = Some(name);
                    }
                    if let Some(args) = func.arguments {
                        tool_args.push_str(&args);
                        args_delta = args;
                    }
                }
            }
            if !emit(StreamChunk::ToolCallDelta {
                index: idx,
                id: id_delta,
                name: name_delta,
                arguments_delta: args_delta,
            }) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn on_eof_without_done(&self) -> Result<(), LlmError> {
        if self.finish_reason.is_some() {
            Ok(())
        } else {
            Err(LlmError::malformed(
                "completions SSE stream truncated before [DONE] or finish_reason",
            ))
        }
    }

    fn on_finish(&mut self, emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        if self.finish_phase == CompletionsFinishPhase::Emitted {
            return Ok(());
        }
        self.finish_phase = CompletionsFinishPhase::Emitted;

        for b in &self.blocks {
            let block = match &b.kind {
                CompletionsBlockKind::Text(text) => ContentBlock::Text { text: text.clone() },
                CompletionsBlockKind::Reasoning(text) => {
                    ContentBlock::Reasoning { text: text.clone() }
                }
                CompletionsBlockKind::ToolCall { id, name, args } => ContentBlock::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: args.clone(),
                },
            };
            if !emit(StreamChunk::BlockEnd {
                index: b.index,
                block,
            }) {
                return Ok(());
            }
        }

        if let Some(usage) = self.usage
            && !emit(StreamChunk::Usage { usage })
        {
            return Ok(());
        }

        let replay = ReplayEnvelope {
            protocol: PROTOCOL_COMPLETIONS.to_string(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response: json!({
                "id": self.response_id,
                "responseModel": self.response_model,
            }),
            blocks: Vec::new(),
        };
        let _ = emit(StreamChunk::Finish {
            reason: self.map_finish_reason(),
            replay: Some(replay),
        });
        Ok(())
    }
}

pub struct CompletionsAdapter {
    config: CompletionsConfig,
    client: reqwest::Client,
}

impl CompletionsAdapter {
    pub fn new(config: CompletionsConfig) -> Self {
        Self::with_client(config, shared_http_client())
    }

    pub fn with_client(config: CompletionsConfig, client: reqwest::Client) -> Self {
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
        session_id: Option<&str>,
    ) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        headers.extend(self.config.headers.iter().cloned());
        headers.extend(extra.iter().cloned());
        let mode = self.config.compat.session_affinity_mode.unwrap_or_else(|| {
            detect_session_affinity_mode(&self.config.provider, Some(&self.config.base_url))
        });
        let send_affinity = self
            .config
            .compat
            .send_session_affinity_headers
            .unwrap_or(mode == SessionAffinityMode::OpenRouter);
        if send_affinity {
            push_session_affinity_headers(&mut headers, mode, session_id, true);
        }
        headers
    }
}

impl LlmAdapter for CompletionsAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: self.config.provider.clone(),
            name: self
                .config
                .display_name
                .clone()
                .unwrap_or_else(|| self.config.provider.clone()),
            protocol: PROTOCOL_COMPLETIONS.to_string(),
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
        let url = format!("{}/models", trim_base_url(&self.config.base_url));
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

        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let body = build_request_body(&self.config, &options, &model);
        let url = format!("{}/chat/completions", trim_base_url(&self.config.base_url));
        let headers = self.build_headers(
            api_key.as_deref(),
            &options.headers,
            options.session_id.as_deref(),
        );
        let decoder = CompletionsSseDecoder::new(self.config.provider.clone(), model);

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
pub struct CompletionsPlugin;

impl Plugin for CompletionsPlugin {
    type Config = CompletionsConfig;

    fn name(&self) -> Option<&str> {
        Some("llm-completions")
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
            ctx.effect("llm-completions.registerAdapter", || {
                Ok::<_, LlmError>(Some(
                    llm.register_adapter(provider, Arc::new(CompletionsAdapter::new(config)))?,
                ))
            })?;
            Ok(None)
        })
    }
}
