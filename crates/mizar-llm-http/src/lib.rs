use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, Utc};
use futures::channel::mpsc;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::time::{Instant, Sleep, sleep, sleep_until};
use tokio_util::sync::CancellationToken;

use mizar_llm::{
    BoxChunkStream, FinishReason, HttpErrorHints, LlmError, LlmErrorCode, ModelInfo, SseEvent,
    SseParser, StreamChunk,
};

pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(180);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const BODY_TIMEOUT: Duration = Duration::from_secs(15);
const ERROR_BODY_LIMIT: usize = 64 * 1024;
const MODELS_BODY_LIMIT: usize = 2 * 1024 * 1024;
const TERMINAL_GRACE: Duration = Duration::from_secs(3);

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
    let builder = || {
        reqwest::Client::builder()
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt
                    .previous()
                    .first()
                    .is_some_and(|url| url.origin() == attempt.url().origin())
                {
                    reqwest::redirect::Policy::limited(3).redirect(attempt)
                } else {
                    attempt.error("cross-origin HTTP redirect refused")
                }
            }))
    };
    // 自定义证书不可用时，回退客户端仍须限制重定向，以免凭据跨源。
    configure_custom_ca_roots(builder())
        .build()
        .unwrap_or_else(|_| builder().build().expect("failed to initialize HTTP client"))
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

    /// 代理可能在协议已给出结束原因后吞掉最后的结束标记。
    fn has_terminal_finish(&self) -> bool {
        false
    }

    /// 底层响应流在 `on_event` 尚未返回 `Ok(true)` 时到达 EOF 的检查（默认拒绝静默截断）。
    fn on_eof_without_done(&self) -> Result<(), LlmError> {
        Err(LlmError::transport(
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

fn nonnegative_number(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
}

fn remaining_ms(timestamp_ms: i64) -> u64 {
    timestamp_ms
        .saturating_sub(Utc::now().timestamp_millis())
        .max(0) as u64
}

fn http_date_delay_ms(value: &str) -> Option<u64> {
    if let Ok(date) = DateTime::parse_from_rfc2822(value) {
        return Some(remaining_ms(date.timestamp_millis()));
    }
    ["%A, %d-%b-%y %T GMT", "%a %b %e %T %Y"]
        .iter()
        .find_map(|format| {
            let date = NaiveDateTime::parse_from_str(value, format).ok()?;
            Some(remaining_ms(date.and_utc().timestamp_millis()))
        })
}

fn reset_delay_ms(value: &str, unit_ms: f64) -> Option<u64> {
    if let Some(n) = nonnegative_number(value) {
        // 网关混用相对时长和 Unix 时间戳，后者通常是十位秒或十三位毫秒。
        if n >= 1_000_000_000_000.0 {
            return Some(remaining_ms(n.ceil() as i64));
        }
        if n >= 1_000_000_000.0 {
            return Some(remaining_ms((n * 1000.0).ceil() as i64));
        }
        return Some((n * unit_ms).ceil() as u64);
    }
    if let Ok(date) = DateTime::parse_from_rfc3339(value) {
        return Some(remaining_ms(date.timestamp_millis()));
    }
    let mut rest = value.strip_prefix('+').unwrap_or(value);
    let mut millis = 0.0;
    loop {
        let unit_start = rest.find(|c: char| !c.is_ascii_digit() && c != '.')?;
        let (number, suffix) = rest.split_at(unit_start);
        let n = nonnegative_number(number)?;
        let (unit, factor) = [
            ("ns", 0.000_001),
            ("us", 0.001),
            ("µs", 0.001),
            ("μs", 0.001),
            ("ms", 1.0),
            ("s", 1000.0),
            ("m", 60_000.0),
            ("h", 3_600_000.0),
        ]
        .into_iter()
        .find(|(unit, _)| suffix.starts_with(unit))?;
        millis += n * factor;
        rest = &suffix[unit.len()..];
        if rest.is_empty() {
            return Some(millis.ceil() as u64);
        }
    }
}

fn http_error_hints(headers: &HeaderMap) -> HttpErrorHints {
    let header = |name: &str| {
        std::str::from_utf8(headers.get(name)?.as_bytes())
            .ok()
            .map(str::trim)
    };
    let retry_after_ms = header("retry-after-ms")
        .and_then(nonnegative_number)
        .map(|ms| ms.ceil() as u64)
        .or_else(|| {
            let value = header("retry-after")?;
            nonnegative_number(value)
                .map(|s| (s * 1000.0).ceil() as u64)
                .or_else(|| http_date_delay_ms(value))
        })
        .or_else(|| reset_delay_ms(header("x-ratelimit-reset-ms")?, 1.0))
        .or_else(|| reset_delay_ms(header("x-ratelimit-reset")?, 1000.0));
    let should_retry = headers
        .get_all("x-should-retry")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .filter_map(|value| match value.trim() {
            v if v == "0" || v.eq_ignore_ascii_case("false") => Some(false),
            v if v == "1" || v.eq_ignore_ascii_case("true") => Some(true),
            _ => None,
        })
        .min();
    let mut limits = headers
        .iter()
        .filter(|(name, _)| name.as_str().starts_with("x-ratelimit-limit-"))
        .peekable();
    let ratelimit_limit_zero = limits.peek().is_some()
        && limits.all(|(_, value)| {
            value
                .to_str()
                .is_ok_and(|value| value.split(',').all(|n| n.trim().parse::<u64>() == Ok(0)))
        });
    HttpErrorHints {
        retry_after_ms,
        should_retry,
        ratelimit_limit_zero,
    }
}

fn request_error(context: &str, error: reqwest::Error) -> LlmError {
    let code = if error.is_timeout() {
        LlmErrorCode::Timeout
    } else if error.is_redirect() {
        LlmErrorCode::InvalidRequest
    } else {
        LlmErrorCode::Transport
    };
    LlmError::new(code, format!("{context}: {}", format_reqwest_error(&error)))
}

async fn send_request(
    request: reqwest::RequestBuilder,
    timeout: Sleep,
    cancel: &CancellationToken,
) -> Result<reqwest::Response, LlmError> {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(LlmError::cancelled()),
        _ = timeout => Err(LlmError::timeout("timed out waiting for HTTP response headers")),
        response = request.send() => response.map_err(|e| request_error("HTTP request failed", e)),
    }
}

async fn read_bounded_body(
    response: &mut reqwest::Response,
    limit: usize,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<Vec<u8>, LlmError> {
    let error_body = !response.status().is_success();
    let too_large = || LlmError::malformed(format!("HTTP response body exceeds {limit} bytes"));
    if !error_body
        && response
            .content_length()
            .is_some_and(|len| len > limit as u64)
    {
        return Err(too_large());
    }
    let mut body = Vec::new();
    let timeout = sleep_until(deadline);
    tokio::pin!(timeout);
    loop {
        let next = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(LlmError::cancelled()),
            _ = &mut timeout => Err(LlmError::timeout("HTTP response body timed out")),
            next = response.chunk() => next.map_err(|e| request_error("HTTP response body failed", e)),
        };
        let bytes = match next {
            Ok(Some(bytes)) => bytes,
            Ok(None) => break,
            // 错误体残缺时仍以已收到的状态码和响应头为准，避免把鉴权失败当作断流重试。
            Err(_) if error_body => break,
            Err(err) => return Err(err),
        };
        let remaining = limit - body.len();
        if !error_body && bytes.len() > remaining {
            return Err(too_large());
        }
        body.extend_from_slice(&bytes[..remaining.min(bytes.len())]);
        if error_body && body.len() == limit {
            break;
        }
    }
    Ok(body)
}

async fn check_response(
    mut response: reqwest::Response,
    deadline: Instant,
    cancel: &CancellationToken,
) -> Result<reqwest::Response, LlmError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = read_bounded_body(&mut response, ERROR_BODY_LIMIT, deadline, cancel).await?;
    Err(LlmError::classify_http_with_hints(
        status.as_u16(),
        http_error_hints(response.headers()),
        &String::from_utf8_lossy(&body),
    ))
}

fn markup_response() -> LlmError {
    LlmError::malformed(
        "received a markup document from the SSE endpoint (check the provider base URL)",
    )
}

#[derive(Default)]
struct SseBody {
    parser: SseParser,
    field: Vec<u8>,
    saw_text: bool,
    started: bool,
}

impl SseBody {
    fn known_field(&self) -> bool {
        matches!(self.field.as_slice(), b"data" | b"event" | b"id" | b"retry")
    }

    fn push(&mut self, mut bytes: &[u8]) -> Result<Vec<SseEvent>, LlmError> {
        const BOM: &[u8] = b"\xef\xbb\xbf";
        // 合法 SSE 可能以未知字段开头，只有在 EOF 仍无协议字段时才能判为普通文本。
        while !self.started
            && let Some((&byte, rest)) = bytes.split_first()
        {
            if !self.saw_text
                && BOM.starts_with(&self.field)
                && BOM.get(self.field.len()) == Some(&byte)
            {
                self.field.push(byte);
                if self.field == BOM {
                    self.field.clear();
                }
                bytes = rest;
                continue;
            }
            if !self.saw_text && !byte.is_ascii_whitespace() {
                if byte == b'<' {
                    return Err(markup_response());
                }
                self.saw_text = true;
            }
            if (byte == b':' && self.field.is_empty())
                || (matches!(byte, b':' | b'\r' | b'\n') && self.known_field())
            {
                self.started = true;
                self.parser.push(&self.field, || {});
                break;
            }
            if byte == b'\n' {
                self.field.clear();
            } else if self.field.len() <= b"event".len() {
                // 多留一字节可将 event 与更长的未知字段区分开。
                self.field.push(byte);
            }
            bytes = rest;
        }
        Ok(self.parser.push(bytes, || {}))
    }

    fn finish(&mut self, declared_sse: bool) -> Result<Vec<SseEvent>, LlmError> {
        if !self.started {
            if self.known_field() {
                self.parser.push(&self.field, || {});
            } else if self.saw_text && !declared_sse {
                return Err(LlmError::malformed(
                    "SSE endpoint returned text without SSE fields (check the provider base URL)",
                ));
            }
        }
        Ok(self.parser.finish(|| {}))
    }
}

/// 共用 SSE 流式请求驱动：处理取消、HTTP 错误归类、空闲超时与帧解码。
pub fn spawn_sse_stream<D: SseStreamDecoder>(req_spec: SseRequest, decoder: D) -> BoxChunkStream {
    let (tx, rx) = mpsc::unbounded();
    tokio::spawn(async move {
        if let Err(err) = drive_sse_stream(req_spec, decoder, &tx).await {
            let _ = tx.unbounded_send(Err(err));
        }
    });
    Box::pin(rx)
}

async fn drive_sse_stream(
    req_spec: SseRequest,
    decoder: impl SseStreamDecoder,
    tx: &mpsc::UnboundedSender<Result<StreamChunk, LlmError>>,
) -> Result<(), LlmError> {
    let req = req_spec
        .client
        .post(&req_spec.url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .headers(build_header_map(&req_spec.headers)?)
        .json(&req_spec.body);
    let idle = std::env::var("MIZAR_LLM_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .map(|s| Duration::from_secs(s.clamp(10, 1800) as u64))
        .unwrap_or(req_spec.idle_timeout);
    let cancel = &req_spec.cancel;
    let response = send_request(req, sleep(idle), cancel).await?;
    let response = check_response(response, Instant::now() + BODY_TIMEOUT, cancel).await?;
    let hints = http_error_hints(response.headers());
    let mut emit = |mut chunk| {
        if let StreamChunk::Finish {
            reason: reason @ FinishReason::Error { .. },
            ..
        } = &mut chunk
            && let Some(err) = reason.clone().into_error()
        {
            *reason = err.with_http_hints(hints).into();
        }
        tx.unbounded_send(Ok(chunk)).is_ok()
    };
    // 空响应可能在上层才被判为错误，也需要这次响应的重试限制。
    if hints != HttpErrorHints::default() && !emit(StreamChunk::ResponseHints(hints)) {
        return Ok(());
    }
    decode_sse(response, decoder, idle, cancel, &mut emit)
        .await
        .map_err(|err| err.with_http_hints(hints))
}

async fn decode_sse(
    mut response: reqwest::Response,
    mut decoder: impl SseStreamDecoder,
    idle: Duration,
    cancel: &CancellationToken,
    emit: &mut (impl FnMut(StreamChunk) -> bool + Send),
) -> Result<(), LlmError> {
    if cancel.is_cancelled() {
        return Err(LlmError::cancelled());
    }
    let mime = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(str::trim)
        .unwrap_or_default();
    if mime.eq_ignore_ascii_case("text/html") || mime.eq_ignore_ascii_case("application/xhtml+xml")
    {
        return Err(markup_response());
    }
    let declared_sse = mime.eq_ignore_ascii_case("text/event-stream");
    let mut body = SseBody::default();
    let mut terminal = decoder.has_terminal_finish();
    let watchdog = sleep(if terminal {
        idle.min(TERMINAL_GRACE)
    } else {
        idle
    });
    tokio::pin!(watchdog);
    let done = 'stream: loop {
        let next = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(LlmError::cancelled()),
            _ = &mut watchdog => {
                if terminal {
                    break false;
                }
                return Err(LlmError::timeout(format!("stream idle timeout after {}s", idle.as_secs())));
            }
            next = response.chunk() => next.map_err(|e| request_error("stream error", e))?,
        };
        let events = match &next {
            Some(bytes) => body.push(bytes)?,
            None => body.finish(declared_sse)?,
        };
        for event in events {
            let mut receiver_closed = false;
            let done = decoder.on_event(event, &mut |chunk| {
                if !emit(chunk) {
                    receiver_closed = true;
                    return false;
                }
                if !terminal && let Some(deadline) = Instant::now().checked_add(idle) {
                    watchdog.as_mut().reset(deadline);
                }
                true
            })?;
            if receiver_closed {
                return Ok(());
            }
            if done {
                break 'stream true;
            }
            if !terminal && decoder.has_terminal_finish() {
                terminal = true;
                let deadline = watchdog.deadline().min(Instant::now() + TERMINAL_GRACE);
                watchdog.as_mut().reset(deadline);
            }
        }
        if next.is_none() {
            break false;
        }
    };
    if !done {
        decoder.on_eof_without_done()?;
    }
    decoder.on_finish(emit)
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
    let request = client.get(url).headers(build_header_map(headers)?);
    let deadline = Instant::now() + BODY_TIMEOUT;
    let cancel = CancellationToken::new();
    let response = send_request(request, sleep_until(deadline), &cancel).await?;
    let mut response = check_response(response, deadline, &cancel).await?;
    let body = read_bounded_body(&mut response, MODELS_BODY_LIMIT, deadline, &cancel).await?;
    let payload: Value = serde_json::from_slice(&body)
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
                })
            })
            .collect();
    }

    Vec::new()
}
