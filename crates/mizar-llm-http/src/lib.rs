use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use futures::StreamExt;
use futures::channel::mpsc;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use mizar_llm::{
    BoxChunkStream, LlmError, LlmErrorCode, ModelInfo, SseEvent, SseParser, StreamChunk,
};

pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

fn format_reqwest_error(err: &reqwest::Error) -> String {
    let mut msg = err.to_string();
    let mut current: &dyn std::error::Error = err;
    while let Some(source) = current.source() {
        let s = source.to_string();
        if !s.is_empty() && !msg.contains(&s) {
            msg.push_str(": ");
            msg.push_str(&s);
        }
        current = source;
    }
    msg
}

fn load_ca_certs_from_path(path: &Path, certs: &mut Vec<reqwest::Certificate>) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let Ok(parsed) = reqwest::Certificate::from_pem_bundle(&bytes) else {
        return false;
    };
    if parsed.is_empty() {
        return false;
    }
    certs.extend(parsed);
    true
}

/// 从常见系统/代理环境变量（`SSL_CERT_FILE` / `CURL_CA_BUNDLE` / `REQUESTS_CA_BUNDLE` /
/// `NODE_EXTRA_CA_CERTS`）及标准系统证书路径加载额外根证书，确保企业或沙箱 HTTPS 代理正常校验。
pub fn configure_custom_ca_roots(mut builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut loaded_any = false;
    let mut certs = Vec::new();

    for var in [
        "SSL_CERT_FILE",
        "CURL_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
    ] {
        if let Some(raw) = std::env::var_os(var)
            && !raw.is_empty()
        {
            let path = PathBuf::from(raw);
            if seen.insert(path.clone()) && load_ca_certs_from_path(&path, &mut certs) {
                loaded_any = true;
            }
        }
    }

    if !loaded_any {
        for fallback in [
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/cert.pem",
            "/etc/ssl/certs/ca-bundle.crt",
        ] {
            let path = PathBuf::from(fallback);
            if seen.insert(path.clone()) && load_ca_certs_from_path(&path, &mut certs) {
                break;
            }
        }
    }

    for cert in certs {
        builder = builder.add_root_certificate(cert);
    }
    builder
}

/// 构造带自定义根证书与连接调优的 HTTP 客户端。
pub fn build_http_client() -> reqwest::Client {
    let builder = configure_custom_ca_roots(
        reqwest::Client::builder()
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .tcp_nodelay(true),
    );
    builder.build().unwrap_or_else(|_| reqwest::Client::new())
}

/// 获取进程级复用的共享 `reqwest::Client`（跨适配器、跨热重载复用连接池与 HTTP/2 连接）。
pub fn default_http_client() -> reqwest::Client {
    static SHARED: OnceLock<reqwest::Client> = OnceLock::new();
    SHARED.get_or_init(build_http_client).clone()
}

/// 返回进程级共享的 `reqwest::Client`，复用底层连接池、TLS 配置与系统代理状态。
pub fn shared_http_client() -> reqwest::Client {
    default_http_client()
}

/// 协议流式解码器接口，由各个具体协议适配器实现（纯解析逻辑，便于单元测试）。
pub trait SseStreamDecoder: Send + 'static {
    /// 处理单帧 SSE 事件。返回 `Ok(true)` 表示流已自然终止。
    fn on_event(
        &mut self,
        event: SseEvent,
        emit: &mut dyn FnMut(StreamChunk) -> bool,
    ) -> Result<bool, LlmError>;

    /// 底层响应流在 `on_event` 尚未返回 `Ok(true)` 时到达 EOF 的检查（默认拒绝静默截断）。
    fn on_eof_without_done(&self) -> Result<(), LlmError> {
        Err(LlmError::malformed(
            "SSE stream ended unexpectedly before completion marker",
        ))
    }

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
        let name = HeaderName::from_bytes(k.as_bytes())
            .map_err(|e| LlmError::invalid_request(format!("invalid header name \"{k}\": {e}")))?;
        let val = HeaderValue::from_str(v).map_err(|e| {
            LlmError::invalid_request(format!("invalid header value for \"{k}\": {e}"))
        })?;
        map.insert(name, val);
    }
    Ok(map)
}

fn parse_retry_after_ms(headers: &HeaderMap) -> Option<u64> {
    if let Some(val) = headers.get("retry-after").and_then(|v| v.to_str().ok())
        && let Ok(secs) = val.trim().parse::<f64>()
        && secs.is_finite()
        && secs > 0.0
    {
        return Some((secs * 1000.0).round() as u64);
    }
    if let Some(val) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok())
        && let Ok(ms) = val.trim().parse::<f64>()
        && ms.is_finite()
        && ms > 0.0
    {
        return Some(ms.round() as u64);
    }
    None
}

/// 共用 SSE 流式请求驱动：处理取消、HTTP 错误归类、空闲超时与帧解码。
pub fn spawn_sse_stream<D: SseStreamDecoder>(
    req_spec: SseRequest,
    mut decoder: D,
) -> BoxChunkStream {
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
                    let _ = tx.unbounded_send(Err(LlmError::new(
                        code,
                        format!("http error: {}", format_reqwest_error(&e)),
                    )));
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
                    let _ =
                        tx.unbounded_send(Err(LlmError::transport(format!("stream error: {e}"))));
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
                        if receiver_closed {
                            return;
                        }
                        if is_done {
                            done = true;
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

        if !done && let Err(err) = decoder.on_eof_without_done() {
            let _ = tx.unbounded_send(Err(err));
            return;
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
        .map_err(|e| {
            LlmError::transport(format!(
                "model discovery failed: {}",
                format_reqwest_error(&e)
            ))
        })?;

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
                let input = item
                    .get("input")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                Some(ModelInfo {
                    id,
                    name,
                    context_window,
                    max_tokens,
                    reasoning_efforts: Vec::new(),
                    input,
                    ..Default::default()
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
                let id = raw_name
                    .strip_prefix("models/")
                    .unwrap_or(raw_name)
                    .to_string();
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
                let input = item
                    .get("input")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                Some(ModelInfo {
                    id,
                    name,
                    context_window,
                    max_tokens,
                    reasoning_efforts: Vec::new(),
                    input,
                    ..Default::default()
                })
            })
            .collect();
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn parses_retry_after_and_retry_after_ms_headers() {
        let mut h1 = HeaderMap::new();
        h1.insert("retry-after", HeaderValue::from_static("1.5"));
        h1.insert("retry-after-ms", HeaderValue::from_static("900"));
        assert_eq!(parse_retry_after_ms(&h1), Some(1500));

        let mut h2 = HeaderMap::new();
        h2.insert("retry-after-ms", HeaderValue::from_static("2500"));
        assert_eq!(parse_retry_after_ms(&h2), Some(2500));

        let h3 = HeaderMap::new();
        assert_eq!(parse_retry_after_ms(&h3), None);
    }

    #[test]
    fn custom_ca_roots_handles_missing_or_invalid_bundle_and_builds_client() {
        let missing =
            std::env::temp_dir().join(format!("mizar_missing_ca_{}.pem", std::process::id()));
        let mut certs = Vec::new();
        assert!(!load_ca_certs_from_path(&missing, &mut certs));
        assert!(certs.is_empty());

        let invalid =
            std::env::temp_dir().join(format!("mizar_invalid_ca_{}.pem", std::process::id()));
        std::fs::write(&invalid, b"not a valid pem cert").unwrap();
        assert!(!load_ca_certs_from_path(&invalid, &mut certs));
        let _ = std::fs::remove_file(&invalid);
        assert!(certs.is_empty());

        let _client1 = build_http_client();
        let _client2 = default_http_client();
    }
}
