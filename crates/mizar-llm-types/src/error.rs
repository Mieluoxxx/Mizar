use serde::{Deserialize, Serialize};
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
            Self::RateLimit
                | Self::Server
                | Self::Timeout
                | Self::Transport
                | Self::EmptyResponse
                | Self::MalformedResponse
        )
    }
}

/// 统一的 LLM 错误结构，携带稳定分类码、HTTP 状态码与建议重试等待时间。
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("{message}")]
pub struct LlmError {
    pub code: LlmErrorCode,
    pub message: String,
    pub status: Option<u16>,
    pub retry_after_ms: Option<u64>,
}

impl LlmError {
    pub fn new(code: LlmErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            status: None,
            retry_after_ms: None,
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
        self.code.is_retryable()
    }

    /// 结合 HTTP 状态码、服务端错误字段与文本内容进行统一归类。
    pub fn classify_provider_error(
        status: Option<u16>,
        error_type: Option<&str>,
        error_code: Option<&str>,
        message: &str,
    ) -> Self {
        let combined = format!(
            "{} {} {}",
            error_type.unwrap_or(""),
            error_code.unwrap_or(""),
            message
        );
        let lower = combined.to_ascii_lowercase();

        let code = if is_context_window_exceeded(&lower) {
            LlmErrorCode::ContextWindowExceeded
        } else if is_quota_exceeded(&lower) {
            LlmErrorCode::Quota
        } else if is_empty_response_error(&lower) {
            LlmErrorCode::EmptyResponse
        } else if matches!(status, Some(401 | 403))
            || lower.contains("authentication_error")
            || lower.contains("permission_error")
            || lower.contains("invalid_api_key")
            || lower.contains("unauthorized")
        {
            LlmErrorCode::Auth
        } else if status == Some(429)
            || lower.contains("rate_limit")
            || lower.contains("rate limit")
            || lower.contains("too many requests")
        {
            LlmErrorCode::RateLimit
        } else if lower.contains("refusal") || lower.contains("content_filter") {
            LlmErrorCode::ContentRefused
        } else if matches!(status, Some(400 | 404 | 413 | 422)) || lower.contains("invalid_request")
        {
            LlmErrorCode::InvalidRequest
        } else if matches!(status, Some(500..=599))
            || lower.contains("overloaded")
            || lower.contains("server_error")
            || lower.contains("api_error")
        {
            LlmErrorCode::Server
        } else if lower.contains("timeout") || lower.contains("timed out") {
            LlmErrorCode::Timeout
        } else {
            LlmErrorCode::Unknown
        };

        let mut err = Self::new(code, message.to_string());
        err.status = status;
        err
    }

    pub fn classify_http(status: u16, retry_after_ms: Option<u64>, body: &str) -> Self {
        let extracted = extract_provider_error_fields(body);
        let msg = if let Some((_, _, ref detail)) = extracted {
            format!("HTTP {status}: {detail}")
        } else if body.trim().is_empty() {
            format!("HTTP {status}")
        } else {
            format!("HTTP {status}: {}", body.trim())
        };
        let (etype, ecode) = extracted
            .as_ref()
            .map(|(t, c, _)| (t.as_deref(), c.as_deref()))
            .unwrap_or((None, None));
        let mut err = Self::classify_provider_error(Some(status), etype, ecode, &msg);
        err.retry_after_ms = retry_after_ms;
        err
    }
}

fn extract_provider_error_fields(body: &str) -> Option<(Option<String>, Option<String>, String)> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let err_obj = v.get("error").unwrap_or(&v);
    let msg = err_obj
        .get("message")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)?;
    let etype = err_obj
        .get("type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let ecode = err_obj
        .get("code")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    Some((etype, ecode, msg))
}

pub fn is_empty_response_error(lower: &str) -> bool {
    lower.contains("cannot both be empty")
        || lower.contains("must contain either output text or tool calls")
        || lower.contains("empty response from model")
}

pub fn is_context_window_exceeded(lower: &str) -> bool {
    lower.contains("context_length_exceeded")
        || lower.contains("context_window_exceeded")
        || lower.contains("context length exceeded")
        || lower.contains("context window")
        || lower.contains("maximum context length")
        || lower.contains("exceeds the maximum number of tokens")
        || lower.contains("prompt is too long")
        || lower.contains("input is too long")
        || lower.contains("request too large for")
}

pub fn is_quota_exceeded(lower: &str) -> bool {
    lower.contains("insufficient_quota")
        || lower.contains("insufficient quota")
        || lower.contains("insufficient balance")
        || lower.contains("quota exceeded")
        || lower.contains("exceeded your current quota")
        || lower.contains("out of credits")
        || lower.contains("credit balance is too low")
}
