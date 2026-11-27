use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// 提供方无关的统一错误码分类。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LlmErrorCode {
    NoAdapter,
    DuplicateProvider,
    MissingCredential,
    InvalidCredential,
    Auth,
    RateLimit,
    Quota,
    ContextWindowExceeded,
    InvalidRequest,
    #[serde(rename = "payload_too_large")]
    PayloadTooLarge,
    ContentRefused,
    EmptyResponse,
    MalformedResponse,
    Server,
    Timeout,
    Transport,
    Cancelled,
    Unknown,
}

impl LlmErrorCode {
    /// 标准化错误码字符串（供结构化诊断输出）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoAdapter => "no_adapter",
            Self::DuplicateProvider => "duplicate_provider",
            Self::MissingCredential => "missing_credential",
            Self::InvalidCredential => "invalid_credential",
            Self::Auth => "auth",
            Self::RateLimit => "rate_limit",
            Self::Quota => "quota",
            Self::ContextWindowExceeded => "context_window_exceeded",
            Self::InvalidRequest => "invalid_request",
            Self::PayloadTooLarge => "payload_too_large",
            Self::ContentRefused => "content_refused",
            Self::EmptyResponse => "empty_response",
            Self::MalformedResponse => "malformed_response",
            Self::Server => "server",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }

    /// 瞬时性、适合自动重试的错误码。
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimit | Self::Server | Self::Timeout | Self::Transport
        )
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HttpErrorHints {
    pub retry_after_ms: Option<u64>,
    pub should_retry: Option<bool>,
    pub ratelimit_limit_zero: bool,
}

/// 统一的 LLM 错误结构，携带稳定分类码、HTTP 状态码与建议重试等待时间。
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct LlmError {
    pub code: LlmErrorCode,
    pub message: String,
    pub status: Option<u16>,
    pub retry_after_ms: Option<u64>,
    pub should_retry: Option<bool>,
}

impl LlmError {
    pub fn new(code: LlmErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            status: None,
            retry_after_ms: None,
            should_retry: None,
        }
    }

    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_retry_after_ms(mut self, ms: u64) -> Self {
        self.retry_after_ms = Some(ms);
        self
    }

    pub fn no_adapter() -> Self {
        Self::new(LlmErrorCode::NoAdapter, "no llm adapter registered")
    }

    pub fn duplicate_provider(provider: &str) -> Self {
        Self::new(
            LlmErrorCode::DuplicateProvider,
            format!("llm provider \"{provider}\" already registered"),
        )
    }

    pub fn missing_credential(detail: impl Into<String>) -> Self {
        Self::new(LlmErrorCode::MissingCredential, detail)
    }

    pub fn cancelled() -> Self {
        Self::new(LlmErrorCode::Cancelled, "request aborted")
    }

    pub fn transport(detail: impl Into<String>) -> Self {
        Self::new(LlmErrorCode::Transport, detail)
    }

    pub fn timeout(detail: impl Into<String>) -> Self {
        Self::new(LlmErrorCode::Timeout, detail)
    }

    pub fn malformed(detail: impl Into<String>) -> Self {
        Self::new(LlmErrorCode::MalformedResponse, detail)
    }

    pub fn invalid_request(detail: impl Into<String>) -> Self {
        Self::new(LlmErrorCode::InvalidRequest, detail)
    }

    pub fn is_retryable(&self) -> bool {
        match self.should_retry {
            Some(false) => false,
            Some(true) if self.code == LlmErrorCode::Unknown => true,
            _ => self.code.is_retryable(),
        }
    }

    pub fn classify_provider_error(
        status: Option<u16>,
        error_type: Option<&str>,
        error_code: Option<&str>,
        message: &str,
    ) -> Self {
        let lower = format!(
            "{} {} {message}",
            error_type.unwrap_or_default(),
            error_code.unwrap_or_default()
        )
        .to_ascii_lowercase();
        let contains = |patterns: &[&str]| patterns.iter().any(|pattern| lower.contains(pattern));
        let code = if matches!(status, Some(401 | 403))
            || contains(&[
                "authentication_error",
                "permission_error",
                "permission_denied",
                "unauthenticated",
                "invalid_api_key",
                "unauthorized",
                "invalid api key",
            ]) {
            LlmErrorCode::Auth
        } else if status == Some(402) || is_quota_exceeded(&lower) {
            LlmErrorCode::Quota
        } else if status == Some(429)
            || contains(&[
                "rate_limit",
                "rate limit",
                "too many requests",
                "tokens per minute",
            ])
        {
            LlmErrorCode::RateLimit
        } else if matches!(status, Some(408 | 504 | 524)) || contains(&["timeout", "timed out"]) {
            LlmErrorCode::Timeout
        } else if matches!(status, Some(500..=599))
            || contains(&[
                "server_busy",
                "overloaded",
                "internal_error",
                "service_unavailable",
                "at capacity",
                "temporarily unavailable",
                "bad record mac",
                "stream error",
                "server_error",
                "api_error",
            ])
        {
            LlmErrorCode::Server
        } else if is_context_window_exceeded(&lower) {
            LlmErrorCode::ContextWindowExceeded
        } else if contains(&["refusal", "content_filter", "content_policy_violation"]) {
            LlmErrorCode::ContentRefused
        } else if matches!(status, Some(400 | 404 | 405 | 422))
            || (status != Some(413) && lower.contains("invalid_request"))
        {
            LlmErrorCode::InvalidRequest
        } else if status == Some(413) {
            LlmErrorCode::PayloadTooLarge
        } else if is_empty_response_error(&lower) {
            LlmErrorCode::EmptyResponse
        } else {
            LlmErrorCode::Unknown
        };
        let mut err = Self::new(code, bounded_detail(message));
        err.status = status;
        err
    }

    pub fn with_http_hints(mut self, hints: HttpErrorHints) -> Self {
        if self.status == Some(429) && hints.ratelimit_limit_zero {
            self.code = LlmErrorCode::Quota;
        }
        self.retry_after_ms = hints.retry_after_ms.or(self.retry_after_ms);
        if self.should_retry != Some(false) {
            self.should_retry = hints.should_retry.or(self.should_retry);
        }
        self
    }

    pub fn classify_http(status: u16, retry_after_ms: Option<u64>, body: &str) -> Self {
        Self::classify_http_with_hints(
            status,
            HttpErrorHints {
                retry_after_ms,
                ..Default::default()
            },
            body,
        )
    }

    pub fn classify_http_with_hints(status: u16, hints: HttpErrorHints, body: &str) -> Self {
        let parsed = serde_json::from_str::<Value>(body).ok();
        let mut err = if let Some(value) = parsed {
            classify_error_value(&value, Some(status))
        } else {
            Self::classify_provider_error(Some(status), None, None, bounded_detail(body.trim()))
        };
        err.message = if err.message.is_empty() {
            format!("HTTP {status}")
        } else {
            format!("HTTP {status}: {}", err.message)
        };
        err.message.truncate(bounded_detail(&err.message).len());
        err.with_http_hints(hints)
    }
}

fn bounded_detail(text: &str) -> &str {
    text.char_indices()
        .nth(4096)
        .map_or(text, |(end, _)| &text[..end])
}

fn inband_status(value: &Value) -> Option<u16> {
    let status = if let Some(number) = value.as_u64() {
        u16::try_from(number).ok()?
    } else {
        value.as_str()?.trim().parse().ok()?
    };
    matches!(status, 400 | 404 | 408 | 413 | 422 | 429 | 500..=599).then_some(status)
}

fn retry_delay(value: &Value, unit_ms: u64) -> Option<u64> {
    if let Some(n) = value
        .as_u64()
        .or_else(|| value.as_str()?.trim().parse().ok())
    {
        return Some(n.saturating_mul(unit_ms));
    }
    let n = value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())?;
    (n.is_finite() && n >= 0.0).then(|| (n * unit_ms as f64).ceil() as u64)
}

fn classify_error_value(value: &Value, http_status: Option<u16>) -> LlmError {
    let error = value.get("error").unwrap_or(value);
    let field = |key: &str| error.get(key).or_else(|| value.get(key));
    let status = http_status.or_else(|| {
        ["status_code", "status", "code"]
            .iter()
            .find_map(|key| inband_status(field(key)?))
    });
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .or_else(|| value.get("message").and_then(Value::as_str));
    let fallback;
    let message = match message {
        Some(message) => message,
        None => {
            fallback = error.to_string();
            &fallback
        }
    };
    let provider_codes = ["code", "status", "status_code"]
        .iter()
        .filter_map(|key| field(key)?.as_str())
        .map(bounded_detail)
        .collect::<Vec<_>>()
        .join(" ");
    let mut err = LlmError::classify_provider_error(
        status,
        field("type").and_then(Value::as_str).map(bounded_detail),
        Some(&provider_codes),
        bounded_detail(message),
    );
    err.retry_after_ms = field("retry_after_ms")
        .and_then(|v| retry_delay(v, 1))
        .or_else(|| field("retry_after").and_then(|v| retry_delay(v, 1000)));
    err
}

pub fn classify_inband_error_value(value: &Value) -> LlmError {
    classify_error_value(value, None)
}

pub fn classify_non_json_sse_frame(raw: &str) -> Option<LlmError> {
    let raw = raw.trim();
    let line = raw.strip_prefix("data:").unwrap_or(raw).trim();
    if line.starts_with(['<', '{', '[']) {
        return None;
    }
    let words = line
        .strip_prefix("HTTP/1.1 ")
        .or_else(|| line.strip_prefix("HTTP/2 "))
        .or_else(|| line.strip_prefix("HTTP "))
        .unwrap_or(line);
    let status = words
        .split([' ', ':', '\t'])
        .next()?
        .parse::<u16>()
        .ok()
        .filter(|s| (400..=599).contains(s));
    let err = LlmError::classify_provider_error(status, None, None, bounded_detail(line));
    (status.is_some() || err.code.is_retryable()).then_some(err)
}

pub fn is_empty_response_error(lower: &str) -> bool {
    lower.contains("cannot both be empty")
        || lower.contains("must contain either output text or tool calls")
        || lower.contains("empty response from model")
}

pub fn is_context_window_exceeded(lower: &str) -> bool {
    const NON_OVERFLOW_PATTERNS: &[&str] = &[
        "rate limit",
        "rate_limit",
        "quota",
        "billing",
        "unauthorized",
        "tokens per minute",
    ];
    if NON_OVERFLOW_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
    {
        return false;
    }
    lower.contains("context_length_exceeded")
        || lower.contains("context_window_exceeded")
        || lower.contains("context length exceeded")
        || lower.contains("context window exceeded")
        || lower.contains("exceeds the context window")
        || lower.contains("exceeds the model's context window")
        || lower.contains("maximum context length")
        || lower.contains("exceeds the maximum number of tokens")
        || lower.contains("prompt is too long")
        || lower.contains("input is too long")
        || lower.contains("too many input tokens")
}

pub fn is_quota_exceeded(lower: &str) -> bool {
    lower.contains("insufficient_quota")
        || lower.contains("insufficient quota")
        || lower.contains("insufficient balance")
        || lower.contains("quota exceeded")
        || lower.contains("exceeded your current quota")
        || lower.contains("out of credits")
        || lower.contains("credit balance is too low")
        || lower.contains("payment required")
        || lower.contains("billing_hard_limit_reached")
        || lower.contains("billing limit")
}
