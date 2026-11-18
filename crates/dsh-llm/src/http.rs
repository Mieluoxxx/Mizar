use std::time::Duration;

use futures::StreamExt;
use futures::channel::mpsc;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::{LlmError, LlmErrorCode};
use crate::runtime::BoxChunkStream;
use crate::sse::{SseEvent, SseParser};
use crate::types::{ModelInfo, StreamChunk};

pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// 协议流式解码器接口，由各个具体协议适配器实现（纯解析逻辑，便于单元测试）。
pub trait SseStreamDecoder: Send + 'static {
    /// 处理单帧 SSE 事件。返回 `Ok(true)` 表示流已自然终止。
    fn on_event(
        &mut self,
        event: SseEvent,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Result<bool, LlmError>;

    /// 流结束时刷出收尾分片（例如未能提前发出的 Finish/Usage）。
    fn on_finish(&mut self, _emit: &mut dyn FnMut(StreamChunk) -> bool) -> Result<(), LlmError> {
        Ok(())
    }
}

pub struct SseRequest {
    pub client: reqwest::Client,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub cancel: CancellationToken,
    pub idle_timeout: Duration,
}

fn build_header_map(headers: &[(String, String)]) -> Result<HeaderMap, LlmError> {
    let mut map = HeaderMap::new();
    for (k, v) in headers {
        let name = HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
            LlmError::invalid_request(format!("invalid header name \"{k}\": {e}"))
        })?;
        let val = HeaderValue::from_str(v).map_err(|e| {
            LlmError::invalid_request(format!("invalid header value for \"{k}\": {e}"))
        })?;
        map.insert(name, val);
    }
    Ok(map)
}

fn parse_retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    let val = headers.get("retry-after")?.to_str().ok()?.trim();
    if let Ok(secs) = val.parse::<f64>()
        && secs.is_finite()
        && secs > 0.0
    {
        return Some((secs * 1000.0).round() as u64);
    }
    None
}

/// 共用 SSE 流式请求驱动：处理取消、HTTP 错误归类、空闲超时与帧解码。
pub fn spawn_sse_stream<D: SseStreamDecoder>(req_spec: SseRequest, mut decoder: D) -> BoxChunkStream {
    let (tx, rx) = mpsc::unbounded();

    tokio::spawn(async move {
        let header_map = match build_header_map(&req_spec.headers) {
            Ok(h) => h,
            Err(err) => {
                let _ = tx.unbounded_send(Err(err));
                return;
            }
        };

        let req = req_spec
            .client
            .post(&req_spec.url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .headers(header_map)
            .json(&req_spec.body);

        let resp = tokio::select! {
            res = req.send() => match res {
                Ok(r) => r,
                Err(e) => {
                    let code = if e.is_timeout() {
                        LlmErrorCode::Timeout
                    } else {
                        LlmErrorCode::Transport
                    };
                    let _ = tx.unbounded_send(Err(LlmError::new(code, format!("http error: {e}"))));
                    return;
                }
            },
            _ = req_spec.cancel.cancelled() => {
                let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let retry_after = parse_retry_after_ms(resp.headers());
            let body = resp.text().await.unwrap_or_default();
            let _ = tx.unbounded_send(Err(LlmError::classify_http(status, retry_after, &body)));
            return;
        }

        let mut byte_stream = resp.bytes_stream();
        let mut parser = SseParser::new();
        let mut done = false;
        let idle = req_spec.idle_timeout;

        while !done {
            let next_chunk = tokio::select! {
                item = tokio::time::timeout(idle, byte_stream.next()) => match item {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        let _ = tx.unbounded_send(Err(LlmError::timeout(format!(
                            "stream idle timeout after {}s",
                            idle.as_secs()
                        ))));
                        return;
                    }
                },
                _ = req_spec.cancel.cancelled() => {
                    let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    return;
                }
            };

            let Some(chunk_res) = next_chunk else {
                break;
            };
            let bytes = match chunk_res {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.unbounded_send(Err(LlmError::transport(format!("stream error: {e}"))));
                    return;
                }
            };

            let events = parser.push(&bytes, || {});
            for ev in events {
                let mut receiver_closed = false;
                let mut emit = |chunk: StreamChunk| -> bool {
                    if tx.unbounded_send(Ok(chunk)).is_err() {
                        receiver_closed = true;
                        false
                    } else {
                        true
                    }
                };
                match decoder.on_event(ev, &mut emit) {
                    Ok(is_done) => {
                        if receiver_closed || is_done {
                            done = true;
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = tx.unbounded_send(Err(err));
                        return;
                    }
                }
                if receiver_closed {
                    return;
                }
            }
        }

        if !done {
            let tail = parser.finish(|| {});
            for ev in tail {
                let mut emit = |chunk: StreamChunk| tx.unbounded_send(Ok(chunk)).is_ok();
                match decoder.on_event(ev, &mut emit) {
                    Ok(is_done) => {
                        if is_done {
                            break;
                        }
                    }
                    Err(err) => {
                        let _ = tx.unbounded_send(Err(err));
                        return;
                    }
                }
            }
        }

        let mut emit = |chunk: StreamChunk| tx.unbounded_send(Ok(chunk)).is_ok();
        if let Err(err) = decoder.on_finish(&mut emit) {
            let _ = tx.unbounded_send(Err(err));
        }
    });

    Box::pin(rx)
}

/// 解析 Pi 风格凭据/请求头字符串：
/// - 支持 `$$` -> `$`、`\$!` / `$!` -> `!`
/// - 以 `!` 开头时执行 shell 命令并裁剪首尾空白（按 Pi `models.json` `!command` 语义）
/// - 支持 `$VAR` 与 `${VAR}` 环境变量展开
///
/// 返回 `(resolved_value, missing_var_names)`。
pub fn expand_pi_value(provider: &str, raw: &str) -> Result<(String, Vec<String>), LlmError> {
    if let Some(cmd) = raw.strip_prefix('!') {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .map_err(|e| {
                LlmError::missing_credential(format!(
                    "failed to execute credential command for provider \"{provider}\": {e}"
                ))
            })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(LlmError::missing_credential(format!(
                "credential command for provider \"{provider}\" exited with {}: {}",
                out.status,
                stderr.trim()
            )));
        }
        let val = String::from_utf8_lossy(&out.stdout)
            .trim_end_matches(&['\r', '\n'][..])
            .to_string();
        return Ok((val, Vec::new()));
    }

    let mut out = String::with_capacity(raw.len());
    let mut missing = Vec::new();
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            let next = chars[i + 1];
            if next == '$' {
                out.push('$');
                i += 2;
                continue;
            }
            if next == '!' {
                out.push('!');
                i += 2;
                continue;
            }
            if next == '{'
                && let Some(rel_end) = chars[i + 2..].iter().position(|&c| c == '}')
            {
                let var: String = chars[i + 2..i + 2 + rel_end].iter().collect();
                if let Ok(val) = std::env::var(&var)
                    && !val.trim().is_empty()
                {
                    out.push_str(&val);
                } else if !var.is_empty() {
                    missing.push(var);
                }
                i = i + 2 + rel_end + 1;
                continue;
            }
            if next.is_ascii_alphabetic() || next == '_' {
                let mut end = i + 2;
                while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_')
                {
                    end += 1;
                }
                let var: String = chars[i + 1..end].iter().collect();
                if let Ok(val) = std::env::var(&var)
                    && !val.trim().is_empty()
                {
                    out.push_str(&val);
                } else {
                    missing.push(var);
                }
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }

    Ok((out, missing))
}

/// 按请求解析静态 `api_key`（支持 `$VAR` / `${VAR}` / `!command`）或 `api_key_env`；
/// 当引用了环境变量但缺失或为空时，返回 `LlmErrorCode::MissingCredential`。
pub fn resolve_credential(
    provider: &str,
    api_key: Option<&str>,
    api_key_env: Option<&str>,
) -> Result<Option<String>, LlmError> {
    if let Some(key) = api_key
        && !key.trim().is_empty()
    {
        let (resolved, missing) = expand_pi_value(provider, key)?;
        if resolved.trim().is_empty() {
            let detail = if missing.is_empty() {
                format!("empty credential for provider \"{provider}\"")
            } else {
                format!(
                    "missing credential env \"{}\" for provider \"{provider}\"",
                    missing.join(", ")
                )
            };
            return Err(LlmError::missing_credential(detail));
        }
        return Ok(Some(resolved));
    }
    if let Some(env_name) = api_key_env {
        let val = std::env::var(env_name)
            .ok()
            .filter(|s| !s.trim().is_empty());
        return match val {
            Some(k) => Ok(Some(k)),
            None => Err(LlmError::missing_credential(format!(
                "missing credential env \"{env_name}\" for provider \"{provider}\""
            ))),
        };
    }
    Ok(None)
}

/// 标准化根 URL（去除末尾斜杠）。
pub fn trim_base_url(base_url: &str) -> &str {
    base_url.trim_end_matches('/')
}

/// 保证路径以 `/v1` 结尾（用于 Anthropic Messages 避免重复拼接 `/v1`）。
pub fn ensure_v1_root(base_url: &str) -> String {
    let trimmed = trim_base_url(base_url);
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// 通用模型列表发现请求（兼容 OpenAI `{ data: [...] }` 及 Anthropic `{ data: [...] }` / Gemini `{ models: [...] }`）。
pub async fn discover_models_http(
    client: &reqwest::Client,
    url: &str,
    headers: &[(String, String)],
) -> Result<Vec<ModelInfo>, LlmError> {
    let header_map = build_header_map(headers)?;
    let resp = client
        .get(url)
        .headers(header_map)
        .send()
        .await
        .map_err(|e| LlmError::transport(format!("model discovery failed: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let retry_after = parse_retry_after_ms(resp.headers());
        let body = resp.text().await.unwrap_or_default();
        return Err(LlmError::classify_http(status, retry_after, &body));
    }

    let payload: Value = resp
        .json()
        .await
        .map_err(|e| LlmError::malformed(format!("invalid /models json: {e}")))?;

    Ok(parse_models_payload(&payload))
}

pub fn parse_models_payload(payload: &Value) -> Vec<ModelInfo> {
    if let Some(arr) = payload.get("data").and_then(Value::as_array) {
        return arr
            .iter()
            .filter_map(|item| {
                let id = item.get("id").and_then(Value::as_str)?.to_string();
                let name = item
                    .get("display_name")
                    .or_else(|| item.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or(&id)
                    .to_string();
                let context_window = item
                    .get("context_window")
                    .or_else(|| item.get("context_length"))
                    .or_else(|| item.get("max_input_tokens"))
                    .and_then(Value::as_u64);
                let max_tokens = item
                    .get("max_tokens")
                    .or_else(|| item.get("max_output_tokens"))
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok());
                Some(ModelInfo {
                    id,
                    name,
                    context_window,
                    max_tokens,
                    reasoning_efforts: Vec::new(),
                })
            })
            .collect();
    }

    if let Some(arr) = payload.get("models").and_then(Value::as_array) {
        return arr
            .iter()
            .filter_map(|item| {
                let raw_name = item
                    .get("id")
                    .or_else(|| item.get("name"))
                    .and_then(Value::as_str)?;
                let id = raw_name.strip_prefix("models/").unwrap_or(raw_name).to_string();
                let name = item
                    .get("displayName")
                    .or_else(|| item.get("display_name"))
                    .and_then(Value::as_str)
                    .unwrap_or(&id)
                    .to_string();
                let context_window = item
                    .get("inputTokenLimit")
                    .or_else(|| item.get("context_window"))
                    .and_then(Value::as_u64);
                let max_tokens = item
                    .get("outputTokenLimit")
                    .or_else(|| item.get("max_tokens"))
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok());
                Some(ModelInfo {
                    id,
                    name,
                    context_window,
                    max_tokens,
                    reasoning_efforts: Vec::new(),
                })
            })
            .collect();
    }

    Vec::new()
}
