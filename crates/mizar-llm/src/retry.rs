//! LLM 流式自动重试中间件（对应 `NamiDSH/packages/llm-retry`）。
//!
//! 通过 `ctx.waterfall("llm/stream", ...)` 无侵入拦截所有适配器流：
//! - 仅在尚未向下游发出任何可见内容（`TextDelta` / `ReasoningDelta` / `ToolCallDelta` / `BlockEnd`）前触发重试；
//! - 若错误属于瞬时故障（`RateLimit` / `Server` / `Timeout` / `Transport` / `EmptyResponse` / `MalformedResponse`），
//!   按 `retry_after_ms` 或指数退避等待后自动重连；
//! - 全程响应 `CancellationToken`。

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::{
    BoxChunkStream, ContentBlock, EVENT_LLM_STREAM, FinishReason, GenerateOptions, LlmContextExt,
    LlmError, LlmErrorCode, LlmRuntime, StreamChunk,
};
use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Next, Plugin};
use futures::StreamExt;
use futures::channel::mpsc;

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

pub const MAX_RETRY_AFTER_DELAY: Duration = Duration::from_secs(60);

fn jitter_unit_sample() -> f64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut z = RandomState::new()
        .hash_one(seq)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn apply_backoff_jitter(base: Duration, unit_sample: f64) -> Duration {
    if base.is_zero() {
        return Duration::ZERO;
    }
    let scale = 0.75 + unit_sample.clamp(0.0, 1.0) * 0.5;
    let nanos = (base.as_nanos() as f64 * scale).round();
    if !nanos.is_finite() || nanos <= 0.0 {
        return Duration::ZERO;
    }
    if nanos >= u64::MAX as f64 {
        return Duration::try_from_secs_f64(base.as_secs_f64() * scale).unwrap_or(base);
    }
    Duration::from_nanos(nanos as u64)
}

impl RetryConfig {
    fn is_retryable(&self, code: LlmErrorCode) -> bool {
        code.is_retryable() || self.extra_retryable_codes.contains(&code)
    }

    fn delay_for(&self, attempt: usize, retry_after_ms: Option<u64>) -> Duration {
        if let Some(ms) = retry_after_ms {
            return Duration::from_millis(ms).min(MAX_RETRY_AFTER_DELAY);
        }
        let factor = 1u32.checked_shl(attempt as u32).unwrap_or(u32::MAX);
        let base = self
            .initial_delay
            .saturating_mul(factor)
            .min(self.max_delay);
        apply_backoff_jitter(base, jitter_unit_sample())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockAssembler, LlmAdapter, LlmRuntime, Message};
    use cordis::Service;
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
            Vec::<crate::LlmAfterResponseEvent>::new(),
        ));
        let after_cap = Arc::clone(&after_events);
        app.on(
            crate::EVENT_LLM_AFTER_RESPONSE,
            move |ev: &crate::LlmAfterResponseEvent| {
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

    #[tokio::test]
    async fn retries_buffered_reasoning_only_stop_and_flags_streamed_reasoning_only() {
        struct BufferedReasoningOnlyAdapter(Arc<AtomicUsize>);
        impl LlmAdapter for BufferedReasoningOnlyAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                if n == 0 {
                    // 无 ReasoningDelta，仅 BlockEnd(Reasoning) + Stop -> 仍在 Buffering，直接内部无感重试
                    let _ = tx.unbounded_send(Ok(StreamChunk::BlockEnd {
                        index: 0,
                        block: ContentBlock::Reasoning {
                            text: "silent thought".into(),
                        },
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                } else {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("ok after retry")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
                Box::pin(rx)
            }
        }

        struct StreamedReasoningOnlyAdapter {
            finish_reason: FinishReason,
        }
        impl LlmAdapter for StreamedReasoningOnlyAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let (tx, rx) = mpsc::unbounded();
                let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                    index: 0,
                    text: "streaming thought".into(),
                }));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: self.finish_reason.clone(),
                    replay: None,
                }));
                Box::pin(rx)
            }
        }

        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(
            LlmRetryPlugin,
            RetryConfig {
                max_retries: 2,
                initial_delay: Duration::from_millis(5),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let llm = app.llm().unwrap();
        let attempts = Arc::new(AtomicUsize::new(0));
        let _r1 = llm
            .register_adapter(
                "buf-reasoning",
                Arc::new(BufferedReasoningOnlyAdapter(Arc::clone(&attempts))),
            )
            .unwrap();
        let mut s1 = llm
            .stream(GenerateOptions {
                provider: Some("buf-reasoning".into()),
                messages: vec![Message::user("q")],
                ..Default::default()
            })
            .unwrap();
        let mut asm = BlockAssembler::new();
        while let Some(item) = s1.next().await {
            asm.push(item.unwrap());
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(asm.text(), "ok after retry");

        // 已透传 ReasoningDelta 后遇到 Stop：保持 ReasoningDelta 立即透传并在末尾转为 Err(EmptyResponse)
        let _r2 = llm
            .register_adapter(
                "stream-reasoning-stop",
                Arc::new(StreamedReasoningOnlyAdapter {
                    finish_reason: FinishReason::Stop,
                }),
            )
            .unwrap();
        let mut s2 = llm
            .stream(GenerateOptions {
                provider: Some("stream-reasoning-stop".into()),
                messages: vec![Message::user("q")],
                ..Default::default()
            })
            .unwrap();
        let mut s2_chunks = Vec::new();
        while let Some(item) = s2.next().await {
            s2_chunks.push(item);
        }
        assert_eq!(s2_chunks.len(), 2);
        assert!(matches!(
            s2_chunks[0],
            Ok(StreamChunk::ReasoningDelta { .. })
        ));
        assert_eq!(
            s2_chunks[1].as_ref().unwrap_err().code,
            LlmErrorCode::EmptyResponse
        );

        // 已透传 ReasoningDelta 后遇到 MaxTokens：原样保留 MaxTokens，不转 EmptyResponse
        let _r3 = llm
            .register_adapter(
                "stream-reasoning-max",
                Arc::new(StreamedReasoningOnlyAdapter {
                    finish_reason: FinishReason::MaxTokens,
                }),
            )
            .unwrap();
        let mut s3 = llm
            .stream(GenerateOptions {
                provider: Some("stream-reasoning-max".into()),
                messages: vec![Message::user("q")],
                ..Default::default()
            })
            .unwrap();
        let mut asm3 = BlockAssembler::new();
        while let Some(item) = s3.next().await {
            asm3.push(item.unwrap());
        }
        assert_eq!(asm3.reasoning(), "streaming thought");
        assert_eq!(asm3.finish(), FinishReason::MaxTokens);
    }

    #[test]
    fn backoff_applies_plus_minus_25_percent_jitter_and_caps_retry_after_at_60s() {
        let base = Duration::from_millis(400);
        assert_eq!(apply_backoff_jitter(base, 0.0), Duration::from_millis(300));
        assert_eq!(apply_backoff_jitter(base, 0.5), Duration::from_millis(400));
        assert_eq!(apply_backoff_jitter(base, 1.0), Duration::from_millis(500));
        assert_eq!(apply_backoff_jitter(Duration::ZERO, 0.8), Duration::ZERO);

        let cfg = RetryConfig {
            initial_delay: Duration::from_millis(400),
            max_delay: Duration::from_secs(10),
            ..Default::default()
        };

        let samples: Vec<Duration> = (0..24).map(|_| cfg.delay_for(1, None)).collect();
        let expected_base = Duration::from_millis(800);
        let low = Duration::from_millis(600);
        let high = Duration::from_millis(1000);
        assert!(
            samples.iter().all(|&d| d >= low && d <= high),
            "samples out of [600ms, 1000ms]: {samples:?}"
        );
        assert!(
            samples.iter().any(|&d| d != expected_base),
            "expected non-constant jittered samples: {samples:?}"
        );

        // Retry-After 不受 max_delay(10s) 压扁，上限提升至 60s
        assert_eq!(cfg.delay_for(0, Some(30_000)), Duration::from_secs(30));
        assert_eq!(cfg.delay_for(0, Some(60_000)), Duration::from_secs(60));
        assert_eq!(cfg.delay_for(0, Some(120_000)), Duration::from_secs(60));
    }
}
