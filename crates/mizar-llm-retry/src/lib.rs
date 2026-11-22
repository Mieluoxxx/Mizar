//! LLM 流式自动重试中间件（对应 `NamiDSH/packages/llm-retry`）。
//!
//! 通过 `ctx.waterfall("llm/stream", ...)` 无侵入拦截所有适配器流：
//! - 仅在尚未向下游发出任何可见内容（`TextDelta` / `ReasoningDelta` / `ToolCallDelta` / `BlockEnd`）前触发重试；
//! - 若错误属于瞬时故障（`RateLimit` / `Server` / `Timeout` / `Transport` / `EmptyResponse` / `MalformedResponse`），
//!   按 `retry_after_ms` 或指数退避等待后自动重连；
//! - 全程响应 `CancellationToken`。

use std::sync::Arc;
use std::time::Duration;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Next, Plugin};
use futures::StreamExt;
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, EVENT_LLM_STREAM, FinishReason, GenerateOptions, LlmContextExt, LlmError,
    LlmErrorCode, LlmRuntime, StreamChunk,
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

fn is_visible_content_chunk(chunk: &StreamChunk) -> bool {
    matches!(
        chunk,
        StreamChunk::TextDelta { .. }
            | StreamChunk::ReasoningDelta { .. }
            | StreamChunk::ToolCallDelta { .. }
            | StreamChunk::BlockEnd { .. }
    )
}

enum AttemptState {
    Buffering {
        buffered_non_content: Vec<StreamChunk>,
    },
    TerminalFinishBuffered {
        buffered_non_content: Vec<StreamChunk>,
    },
    StreamingThrough,
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
                        if is_visible_content_chunk(&chunk) {
                            match std::mem::replace(&mut state, AttemptState::StreamingThrough) {
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
                                AttemptState::StreamingThrough => {}
                            }
                            if tx.unbounded_send(Ok(chunk)).is_err() {
                                return;
                            }
                            continue;
                        }

                        match &mut state {
                            AttemptState::StreamingThrough => {
                                if tx.unbounded_send(Ok(chunk)).is_err() {
                                    return;
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
                            AttemptState::StreamingThrough => Vec::new(),
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
                AttemptState::StreamingThrough => AttemptOutcome::Done,
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

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::Service;
    use mizar_llm::{BlockAssembler, LlmAdapter, LlmRuntime, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio_util::sync::CancellationToken;

    struct FlakyAdapter {
        attempts: Arc<AtomicUsize>,
        fail_until: usize,
        fail_after_content: bool,
    }

    impl LlmAdapter for FlakyAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::unbounded();
            if n < self.fail_until {
                if self.fail_after_content {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("partial ")));
                }
                let _ = tx.unbounded_send(Err(LlmError::new(
                    LlmErrorCode::RateLimit,
                    "429 rate limited",
                )));
            } else {
                let _ = tx.unbounded_send(Ok(StreamChunk::text("recovered")));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            }
            Box::pin(rx)
        }
    }

    #[tokio::test]
    async fn retries_transient_failure_before_visible_content() {
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(
            LlmRetryPlugin,
            RetryConfig {
                max_retries: 2,
                initial_delay: Duration::from_millis(5),
                max_delay: Duration::from_millis(20),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let llm = app.llm().unwrap();
        let _reg = llm
            .register_adapter(
                "flaky",
                Arc::new(FlakyAdapter {
                    attempts: Arc::clone(&attempts),
                    fail_until: 2,
                    fail_after_content: false,
                }),
            )
            .unwrap();

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("flaky".into()),
                messages: vec![Message::user("hi")],
                cancel: CancellationToken::new(),
                ..Default::default()
            })
            .unwrap();

        let after_events = Arc::new(std::sync::Mutex::new(
            Vec::<mizar_llm::LlmAfterResponseEvent>::new(),
        ));
        let after_cap = Arc::clone(&after_events);
        app.on(
            mizar_llm::EVENT_LLM_AFTER_RESPONSE,
            move |ev: &mizar_llm::LlmAfterResponseEvent| {
                after_cap.lock().unwrap().push(ev.clone());
            },
        )
        .unwrap();

        let mut asm = BlockAssembler::new();
        while let Some(item) = stream.next().await {
            asm.push(item.unwrap());
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(asm.text(), "recovered");
        assert_eq!(after_events.lock().unwrap().len(), 1);
        assert_eq!(after_events.lock().unwrap()[0].provider, "flaky");
    }

    #[tokio::test]
    async fn does_not_retry_after_partial_content_was_emitted() {
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(
            LlmRetryPlugin,
            RetryConfig {
                max_retries: 3,
                initial_delay: Duration::from_millis(5),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let llm = app.llm().unwrap();
        let _reg = llm
            .register_adapter(
                "flaky2",
                Arc::new(FlakyAdapter {
                    attempts: Arc::clone(&attempts),
                    fail_until: 2,
                    fail_after_content: true,
                }),
            )
            .unwrap();

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("flaky2".into()),
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item);
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].is_ok());
        assert_eq!(
            chunks[1].as_ref().unwrap_err().code,
            LlmErrorCode::RateLimit
        );
    }
}
