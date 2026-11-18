//! OpenAI Responses 协议适配器（对应 `NamiDSH/packages/llm-responses`）。

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
    LlmContextExt, LlmError, Message, ModelInfo, ProviderInfo, ReasoningEffort, ReplayEnvelope,
    SseEvent, StreamChunk, TokenUsage,
};
use futures::channel::mpsc;
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
            models: Vec::new(),
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        }
    }
}

/// 把项目内部统一的 `Message` 翻译成 OpenAI Responses API 的 `input` 条目。
pub fn to_wire(message: &Message) -> Vec<Value> {
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
        Message::Assistant { blocks, .. } => {
            let mut items = Vec::new();
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

            if !text_buf.is_empty() || calls.is_empty() {
                items.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": text_buf,
                }));
            }
            for (id, name, arguments) in calls {
                items.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                }));
            }
            items
        }
        Message::User { content } => vec![json!({
            "type": "message",
            "role": "user",
            "content": content,
        })],
        Message::System { content } => vec![json!({
            "type": "message",
            "role": "system",
            "content": content,
        })],
    }
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
        .flat_map(to_wire)
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
    has_tool_calls: bool,
    usage: Option<TokenUsage>,
    finish_reason: Option<FinishReason>,
    finished_emitted: bool,
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
            has_tool_calls: false,
            usage: None,
            finish_reason: None,
            finished_emitted: false,
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
        let data = event.data.trim();
        if data.is_empty() {
            return Ok(false);
        }
        if data == "[DONE]" {
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
                if let Some(item) = payload.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
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
            }
            "response.completed" | "response.incomplete" => {
                if let Some(resp_obj) = payload.get("response") {
                    if let Some(id) = resp_obj.get("id").and_then(Value::as_str) {
                        self.response_id = Some(id.to_string());
                    }
                    if let Some(u) = Self::parse_usage(resp_obj) {
                        self.usage = Some(u);
                    }
                    if event_type == "response.incomplete"
                        || resp_obj.pointer("/incomplete_details/reason").and_then(Value::as_str)
                            == Some("max_output_tokens")
                    {
                        self.finish_reason = Some(FinishReason::MaxTokens);
                    }
                }
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
        if self.finished_emitted {
            return Ok(());
        }
        self.finished_emitted = true;

        for (_, pending) in self.pending_calls.drain() {
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

        let reason = self.finish_reason.clone().unwrap_or(if self.has_tool_calls {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        });
        let replay = ReplayEnvelope {
            protocol: PROTOCOL_RESPONSES.to_string(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            response: json!({ "id": self.response_id }),
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
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = config
            .model
            .clone()
            .unwrap_or_else(|| "gpt-5".to_string());
        Self {
            config,
            base_url,
            model,
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
        for (k, v) in &self.config.headers {
            headers.push((k.clone(), v.clone()));
        }
        headers.extend(extra.iter().cloned());
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
        if !self.config.models.is_empty() {
            let models = self.config.models.clone();
            return Box::pin(async move { Ok(models) });
        }
        let api_key = match self.resolve_api_key() {
            Ok(k) => k,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let url = format!("{}/models", trim_base_url(&self.base_url));
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
            .unwrap_or_else(|| self.model.clone());
        let body = build_request_body(&self.config, &options, &model);
        let url = format!("{}/responses", trim_base_url(&self.base_url));
        let headers = self.build_headers(api_key.as_deref(), &options.headers);
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
    use mizar_llm::{BlockAssembler, SseParser, ToolCall};
    use futures::StreamExt;
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
}
