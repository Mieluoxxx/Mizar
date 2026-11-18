//! OpenAI Chat Completions 兼容协议适配器。
//!
//! 通过 `CompletionsPlugin` 与 `CompletionsCompat` 方言开关接入任何 Chat Completions
//! 协议端点（OpenAI / DeepSeek / OpenRouter / vLLM / Qwen 等）。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_llm_http::{
    DEFAULT_IDLE_TIMEOUT, SseRequest, SseStreamDecoder, discover_models_http, resolve_credential,
    spawn_sse_stream, trim_base_url,
};
use mizar_llm::{
    BoxChunkStream, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions, LlmAdapter,
    LlmContextExt, LlmError, LlmErrorCode, Message, ModelInfo, ProviderInfo, ReasoningEffort,
    ReplayEnvelope, SseEvent, StreamChunk, TokenUsage,
};
use futures::channel::mpsc;
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
}

impl CompletionsCompat {
    pub fn deepseek() -> Self {
        Self {
            thinking_format: ThinkingFormat::DeepSeek,
            max_tokens_field: MaxTokensField::MaxTokens,
            include_usage: true,
        }
    }

    pub fn openai() -> Self {
        Self {
            thinking_format: ThinkingFormat::OpenAI,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            include_usage: true,
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
        Message::User { content } => json!({ "role": "user", "content": content }),
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

struct OpenBlock {
    index: usize,
    kind: ContentBlockKind,
    text: String,
    tool_id: String,
    tool_name: String,
    tool_args: String,
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
    finished_emitted: bool,
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
            finished_emitted: false,
        }
    }

    fn alloc_block(
        &mut self,
        kind: ContentBlockKind,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Option<usize> {
        let idx = self.next_index;
        self.next_index += 1;
        self.blocks.push(OpenBlock {
            index: idx,
            kind,
            text: String::new(),
            tool_id: String::new(),
            tool_name: String::new(),
            tool_args: String::new(),
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
                    .any(|b| b.kind == ContentBlockKind::ToolCall)
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
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(false);
        }
        if data == "[DONE]" {
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
            self.blocks[idx].text.push_str(&reasoning);
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
            self.blocks[idx].text.push_str(&text);
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
            let block = &mut self.blocks[idx];
            if let Some(id) = part.id
                && !id.is_empty()
            {
                block.tool_id = id;
            }
            if let Some(func) = part.function {
                if let Some(name) = func.name
                    && !name.is_empty()
                {
                    block.tool_name.push_str(&name);
                    name_delta = Some(name);
                }
                if let Some(args) = func.arguments {
                    block.tool_args.push_str(&args);
                    args_delta = args;
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

    fn on_finish(&mut self, emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        if self.finished_emitted {
            return Ok(());
        }
        self.finished_emitted = true;

        for b in &self.blocks {
            let block = match b.kind {
                ContentBlockKind::Text => ContentBlock::Text {
                    text: b.text.clone(),
                },
                ContentBlockKind::Reasoning => ContentBlock::Reasoning {
                    text: b.text.clone(),
                },
                ContentBlockKind::ToolCall => ContentBlock::ToolCall {
                    id: b.tool_id.clone(),
                    name: b.tool_name.clone(),
                    arguments: b.tool_args.clone(),
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
        Self {
            config,
            client: reqwest::Client::new(),
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
    ) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Some(key) = api_key {
            headers.push(("Authorization".to_string(), format!("Bearer {key}")));
        }
        headers.extend(self.config.headers.iter().cloned());
        headers.extend(extra.iter().cloned());
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
        if !self.config.models.is_empty() {
            let models = self.config.models.clone();
            return Box::pin(async move { Ok(models) });
        }
        let api_key = match self.resolve_api_key() {
            Ok(k) => k,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let url = format!("{}/models", trim_base_url(&self.config.base_url));
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
        let body = build_request_body(&self.config, &options, &model);
        let url = format!("{}/chat/completions", trim_base_url(&self.config.base_url));
        let headers = self.build_headers(api_key.as_deref(), &options.headers);
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::response::Response;
    use axum::routing::post;
    use mizar_llm::{BlockAssembler, SseParser, ToolCall};
    use futures::StreamExt;
    use tokio::net::TcpListener;

    #[test]
    fn assembles_reasoning_text_tool_calls_and_usage_from_sse() {
        let raw_sse = concat!(
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"先想\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"先查\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"一下\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_\",\"arguments\":\"{\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"time\",\"arguments\":\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":25,\"prompt_tokens_details\":{\"cached_tokens\":40},\"completion_tokens_details\":{\"reasoning_tokens\":10}}}\n\n",
            "data: [DONE]\n\n",
        );

        let mut parser = SseParser::new();
        let events = parser.push(raw_sse.as_bytes(), || {});
        let mut decoder = CompletionsSseDecoder::new("completions", "gpt-4o");
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

        assert_eq!(asm.reasoning(), "先想");
        assert_eq!(asm.text(), "先查一下");
        assert_eq!(
            asm.tool_calls(),
            vec![ToolCall {
                id: "call_1".to_string(),
                name: "get_time".to_string(),
                arguments: "{}".to_string(),
            }]
        );
        let usage = asm.usage().unwrap();
        assert_eq!(usage.input_tokens, 60);
        assert_eq!(usage.cache_read_tokens, Some(40));
        assert_eq!(usage.output_tokens, 25);
        assert_eq!(usage.reasoning_tokens, Some(10));
        assert_eq!(asm.finish(), FinishReason::ToolCalls);
    }

    #[tokio::test]
    async fn streams_end_to_end_from_mock_completions_server() {
        let app = Router::new().route(
            "/chat/completions",
            post(|| async {
                let body = concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Rust\"},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n",
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

        let mut cfg = CompletionsConfig::new("test-key");
        cfg.base_url = format!("http://{addr}");
        let adapter = CompletionsAdapter::new(cfg);

        let mut stream = adapter.stream(GenerateOptions {
            messages: vec![Message::user("hi")],
            ..Default::default()
        });
        let mut asm = BlockAssembler::new();
        while let Some(item) = stream.next().await {
            asm.push(item.unwrap());
        }
        assert_eq!(asm.text(), "Hello Rust");
        assert_eq!(asm.finish(), FinishReason::Stop);
    }
}
