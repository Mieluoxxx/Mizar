//! LLM 流式自动重试中间件（对应 `NamiDSH/packages/llm-retry`）。
//!
//! 通过 `ctx.waterfall("llm/stream", ...)` 无侵入拦截所有适配器流：
//! - 仅在尚未向下游发出任何实质或流式内容前触发重试（`ReasoningDelta` 为保住 TTFT 立即透传，
//!   若最终无正文或工具调用而以 `Stop` 结束，则在末尾转为可重试的 `EmptyResponse`）；
//! - 若错误属于瞬时故障（`RateLimit` / `Server` / `Timeout` / `Transport` / `EmptyResponse` / `MalformedResponse`），
//!   按 `retry_after_ms` 或指数退避等待后自动重连；
//! - 全程响应 `CancellationToken`。

use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Next, Plugin};
use futures::StreamExt;
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, ContentBlock, EVENT_LLM_STREAM, FinishReason, GenerateOptions, LlmContextExt,
    LlmError, LlmErrorCode, LlmRuntime, StreamChunk,
};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    /// 若流在未返回任何可见内容块的情况下正常结束，是否视为 `EmptyResponse` 并重试。
    pub retry_on_empty_response: bool,
    /// 默认按 `LlmErrorCode::is_retryable()` 判断，也可在此追加需要重试的错误码（例如 `Quota`）。
    pub extra_retryable_codes: Vec<LlmErrorCode>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(400),
            max_delay: Duration::from_secs(10),
            retry_on_empty_response: true,
            extra_retryable_codes: Vec::new(),
        }
    }
}

impl RetryConfig {
    fn is_retryable(&self, code: LlmErrorCode) -> bool {
        code.is_retryable() || self.extra_retryable_codes.contains(&code)
    }

    fn delay_for(&self, attempt: usize, retry_after_ms: Option<u64>) -> Duration {
        if let Some(ms) = retry_after_ms {
            return Duration::from_millis(ms).min(self.max_delay);
        }
        let factor = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
        self.initial_delay
            .saturating_mul(factor)
            .min(self.max_delay)
    }
}

fn is_substantive_content_chunk(chunk: &StreamChunk) -> bool {
    match chunk {
        StreamChunk::TextDelta { text, .. } => !text.trim().is_empty(),
        StreamChunk::ToolCallDelta {
            id,
            name,
            arguments_delta,
            ..
        } => {
            !arguments_delta.is_empty()
                || id.as_deref().is_some_and(|s| !s.is_empty())
                || name.as_deref().is_some_and(|s| !s.is_empty())
        }
        StreamChunk::BlockEnd { block, .. } => match block {
            ContentBlock::ToolCall { .. } => true,
            ContentBlock::Text { text } => !text.trim().is_empty(),
            ContentBlock::Reasoning { .. } => false,
        },
        StreamChunk::BlockStart { .. }
        | StreamChunk::ReasoningDelta { .. }
        | StreamChunk::Usage { .. }
        | StreamChunk::Finish { .. } => false,
    }
}

fn is_reasoning_delta_chunk(chunk: &StreamChunk) -> bool {
    matches!(chunk, StreamChunk::ReasoningDelta { text, .. } if !text.is_empty())
}

enum AttemptState {
    Buffering {
        buffered_non_content: Vec<StreamChunk>,
    },
    /// 已透传 `ReasoningDelta` 以保住 TTFT，但尚未收到正文或工具调用。
    StreamingReasoningOnly,
    TerminalFinishBuffered {
        buffered_non_content: Vec<StreamChunk>,
    },
    StreamingSubstantive,
}

enum AttemptOutcome {
    Done,
    Retry(LlmError),
    TerminalErr,
}

fn wrap_with_retry(
    llm: Arc<LlmRuntime>,
    cfg: RetryConfig,
    options: GenerateOptions,
    first_stream: BoxChunkStream,
) -> BoxChunkStream {
    let (tx, rx) = mpsc::unbounded();

    tokio::spawn(async move {
        let mut current_stream = first_stream;

        for attempt in 0..=cfg.max_retries {
            let mut state = AttemptState::Buffering {
                buffered_non_content: Vec::new(),
            };
            let mut outcome: Option<AttemptOutcome> = None;

            while let Some(item) = current_stream.next().await {
                match item {
                    Ok(chunk) => {
                        if is_substantive_content_chunk(&chunk) {
                            match std::mem::replace(&mut state, AttemptState::StreamingSubstantive)
                            {
                                AttemptState::Buffering {
                                    buffered_non_content,
                                }
                                | AttemptState::TerminalFinishBuffered {
                                    buffered_non_content,
                                } => {
                                    for pending in buffered_non_content {
                                        if tx.unbounded_send(Ok(pending)).is_err() {
                                            return;
                                        }
                                    }
                                }
                                AttemptState::StreamingReasoningOnly
                                | AttemptState::StreamingSubstantive => {}
                            }
                            if tx.unbounded_send(Ok(chunk)).is_err() {
                                return;
                            }
                            continue;
                        }

                        if matches!(state, AttemptState::Buffering { .. })
                            && is_reasoning_delta_chunk(&chunk)
                        {
                            // 为保住 TTFT，收到推理流增量立即刷出并透传；若最终无实质输出再转 EmptyResponse 交给上层整步重试。
                            if let AttemptState::Buffering {
                                buffered_non_content,
                            } =
                                std::mem::replace(&mut state, AttemptState::StreamingReasoningOnly)
                            {
                                for pending in buffered_non_content {
                                    if tx.unbounded_send(Ok(pending)).is_err() {
                                        return;
                                    }
                                }
                            }
                            if tx.unbounded_send(Ok(chunk)).is_err() {
                                return;
                            }
                            continue;
                        }

                        match &mut state {
                            AttemptState::StreamingSubstantive => {
                                if tx.unbounded_send(Ok(chunk)).is_err() {
                                    return;
                                }
                            }
                            AttemptState::StreamingReasoningOnly => {
                                if let StreamChunk::Finish {
                                    reason: FinishReason::Stop,
                                    ..
                                } = chunk
                                    && cfg.retry_on_empty_response
                                {
                                    let _ = tx.unbounded_send(Err(LlmError::new(
                                        LlmErrorCode::EmptyResponse,
                                        "model output contained only reasoning without visible text or tool calls",
                                    )));
                                    outcome = Some(AttemptOutcome::TerminalErr);
                                    break;
                                }
                                let is_finish = matches!(chunk, StreamChunk::Finish { .. });
                                if tx.unbounded_send(Ok(chunk)).is_err() {
                                    return;
                                }
                                if is_finish {
                                    state = AttemptState::StreamingSubstantive;
                                }
                            }
                            AttemptState::TerminalFinishBuffered {
                                buffered_non_content,
                            } => {
                                buffered_non_content.push(chunk);
                            }
                            AttemptState::Buffering {
                                buffered_non_content,
                            } => {
                                if let StreamChunk::Finish {
                                    reason:
                                        FinishReason::Error {
                                            code,
                                            ref message,
                                            status,
                                            retry_after_ms,
                                        },
                                    ..
                                } = chunk
                                {
                                    if attempt < cfg.max_retries && cfg.is_retryable(code) {
                                        outcome = Some(AttemptOutcome::Retry(LlmError {
                                            code,
                                            message: message.clone(),
                                            status,
                                            retry_after_ms,
                                        }));
                                        break;
                                    }
                                    let mut buf = std::mem::take(buffered_non_content);
                                    buf.push(chunk);
                                    state = AttemptState::TerminalFinishBuffered {
                                        buffered_non_content: buf,
                                    };
                                    continue;
                                }

                                if let StreamChunk::Finish {
                                    reason: FinishReason::Stop,
                                    ..
                                } = chunk
                                    && cfg.retry_on_empty_response
                                    && attempt < cfg.max_retries
                                {
                                    outcome = Some(AttemptOutcome::Retry(LlmError::new(
                                        LlmErrorCode::EmptyResponse,
                                        "model returned empty response",
                                    )));
                                    break;
                                }

                                buffered_non_content.push(chunk);
                            }
                        }
                    }
                    Err(err) => {
                        let buffered = match &mut state {
                            AttemptState::Buffering {
                                buffered_non_content,
                            } => {
                                if attempt < cfg.max_retries && cfg.is_retryable(err.code) {
                                    outcome = Some(AttemptOutcome::Retry(err));
                                    break;
                                }
                                std::mem::take(buffered_non_content)
                            }
                            AttemptState::TerminalFinishBuffered {
                                buffered_non_content,
                            } => std::mem::take(buffered_non_content),
                            AttemptState::StreamingReasoningOnly
                            | AttemptState::StreamingSubstantive => Vec::new(),
                        };
                        for pending in buffered {
                            if tx.unbounded_send(Ok(pending)).is_err() {
                                return;
                            }
                        }
                        let _ = tx.unbounded_send(Err(err));
                        outcome = Some(AttemptOutcome::TerminalErr);
                        break;
                    }
                }
            }

            let resolved = outcome.unwrap_or_else(|| match state {
                AttemptState::Buffering {
                    buffered_non_content,
                } => {
                    if cfg.retry_on_empty_response && attempt < cfg.max_retries {
                        AttemptOutcome::Retry(LlmError::new(
                            LlmErrorCode::EmptyResponse,
                            "model returned empty stream",
                        ))
                    } else {
                        for pending in buffered_non_content {
                            if tx.unbounded_send(Ok(pending)).is_err() {
                                return AttemptOutcome::TerminalErr;
                            }
                        }
                        AttemptOutcome::Done
                    }
                }
                AttemptState::StreamingReasoningOnly => {
                    if cfg.retry_on_empty_response {
                        let _ = tx.unbounded_send(Err(LlmError::new(
                            LlmErrorCode::EmptyResponse,
                            "model stream ended with only reasoning without visible text or tool calls",
                        )));
                        AttemptOutcome::TerminalErr
                    } else {
                        AttemptOutcome::Done
                    }
                }
                AttemptState::TerminalFinishBuffered {
                    buffered_non_content,
                } => {
                    for pending in buffered_non_content {
                        if tx.unbounded_send(Ok(pending)).is_err() {
                            return AttemptOutcome::TerminalErr;
                        }
                    }
                    AttemptOutcome::Done
                }
                AttemptState::StreamingSubstantive => AttemptOutcome::Done,
            });

            match resolved {
                AttemptOutcome::Retry(err) => {
                    let delay = cfg.delay_for(attempt, err.retry_after_ms);
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = options.cancel.cancelled() => {
                            let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                            return;
                        }
                    }
                    match llm.stream_direct(options.clone()) {
                        Ok(next_stream) => {
                            current_stream = next_stream;
                            continue;
                        }
                        Err(err) => {
                            let _ = tx.unbounded_send(Err(err));
                            return;
                        }
                    }
                }
                AttemptOutcome::Done | AttemptOutcome::TerminalErr => return,
            }
        }
    });

    Box::pin(rx)
}

#[derive(Default, Clone, Copy)]
pub struct LlmRetryPlugin;

impl Plugin for LlmRetryPlugin {
    type Config = RetryConfig;

    fn name(&self) -> Option<&str> {
        Some("llm-retry")
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
            ctx.on_waterfall(
                EVENT_LLM_STREAM,
                move |options: GenerateOptions, next: Next<GenerateOptions, BoxChunkStream>| {
                    let first = next.call(options.clone());
                    wrap_with_retry(Arc::clone(&llm), config.clone(), options, first)
                },
            )?;
            Ok(None)
        })
    }
}
