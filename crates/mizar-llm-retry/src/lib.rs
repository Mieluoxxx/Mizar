use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures::StreamExt;
use mizar_llm::{BlockAssembler, GenerateOptions, LlmError, LlmRuntime, StreamChunk};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub max_mid_thinking_retries: usize,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub max_retry_after: Duration,
    pub jitter_ratio: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            max_mid_thinking_retries: 2,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(25),
            max_retry_after: Duration::from_secs(60),
            jitter_ratio: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    Wait(Duration),
    GiveUp(GiveUpReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GiveUpReason {
    NonRetryable,
    Exhausted,
    RetryAfterExceedsMax { retry_after_ms: u64, max_ms: u64 },
}

impl RetryConfig {
    pub fn from_env() -> Self {
        fn read<T: std::str::FromStr>(key: &str) -> Option<T> {
            std::env::var(key).ok()?.trim().parse().ok()
        }
        let defaults = Self::default();
        Self {
            max_retries: read("MIZAR_RETRY_MAX_RETRIES").unwrap_or(defaults.max_retries),
            max_mid_thinking_retries: read("MIZAR_RETRY_MAX_MID_THINKING")
                .unwrap_or(defaults.max_mid_thinking_retries),
            initial_delay: read("MIZAR_RETRY_INITIAL_DELAY_MS")
                .map(Duration::from_millis)
                .unwrap_or(defaults.initial_delay),
            max_delay: read("MIZAR_RETRY_MAX_DELAY_MS")
                .map(Duration::from_millis)
                .unwrap_or(defaults.max_delay),
            max_retry_after: read("MIZAR_RETRY_MAX_RETRY_AFTER_MS")
                .map(Duration::from_millis)
                .unwrap_or(defaults.max_retry_after),
            ..defaults
        }
    }

    pub fn decide(&self, err: &LlmError, retries: usize) -> RetryDecision {
        if !err.is_retryable() {
            return RetryDecision::GiveUp(GiveUpReason::NonRetryable);
        }
        if retries >= self.max_retries {
            return RetryDecision::GiveUp(GiveUpReason::Exhausted);
        }
        if let Some(ms) = err.retry_after_ms {
            let delay = Duration::from_millis(ms);
            return if delay <= self.max_retry_after {
                RetryDecision::Wait(delay)
            } else {
                RetryDecision::GiveUp(GiveUpReason::RetryAfterExceedsMax {
                    retry_after_ms: ms,
                    max_ms: self.max_retry_after.as_millis() as u64,
                })
            };
        }
        static SEED: LazyLock<RandomState> = LazyLock::new(RandomState::new);
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let factor = 1u64.checked_shl(retries.min(64) as u32).unwrap_or(u64::MAX);
        let ms = self
            .initial_delay
            .as_millis()
            .saturating_mul(u128::from(factor))
            .min(self.max_delay.as_millis())
            .min(u128::from(u64::MAX)) as u64;
        let random =
            SEED.hash_one(SEQUENCE.fetch_add(1, Ordering::Relaxed)) as f64 / u64::MAX as f64;
        let reduction = (ms as f64 * self.jitter_ratio.clamp(0.0, 1.0) * random) as u64;
        RetryDecision::Wait(Duration::from_millis(ms.saturating_sub(reduction)))
    }
}

pub async fn sleep_cancellable(
    delay: Duration,
    cancel: &CancellationToken,
) -> Result<(), LlmError> {
    tokio::select! {
        biased;
        () = cancel.cancelled() => Err(LlmError::cancelled()),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}

pub async fn collect_attempt(
    llm: &LlmRuntime,
    options: GenerateOptions,
    assembler: &mut BlockAssembler,
    mut on_chunk: impl FnMut(&StreamChunk),
) -> Result<(), LlmError> {
    let cancel = options.cancel.clone();
    if cancel.is_cancelled() {
        return Err(LlmError::cancelled());
    }
    let mut stream = llm.stream(options)?;
    loop {
        let next = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(LlmError::cancelled()),
            next = stream.next() => next,
        };
        let Some(chunk) = next else { break };
        let chunk = chunk?;
        on_chunk(&chunk);
        assembler.push(chunk);
    }
    assembler.finish().into_error().map_or(Ok(()), Err)
}

pub async fn collect_with_retry(
    llm: &LlmRuntime,
    options: GenerateOptions,
    config: &RetryConfig,
    retries_used: &mut usize,
    mut on_text_progress: impl FnMut(usize),
) -> Result<BlockAssembler, LlmError> {
    loop {
        let mut assembler = BlockAssembler::new();
        let mut chars_seen = 0;
        let result = collect_attempt(llm, options.clone(), &mut assembler, |chunk| {
            if let StreamChunk::TextDelta { text, .. } = chunk
                && !text.is_empty()
            {
                chars_seen += text.chars().count();
                on_text_progress(chars_seen);
            }
        })
        .await;
        match result {
            Ok(()) => return Ok(assembler),
            Err(err) => match config.decide(&err, *retries_used) {
                RetryDecision::GiveUp(_) => return Err(err),
                RetryDecision::Wait(delay) => {
                    *retries_used += 1;
                    on_text_progress(0);
                    sleep_cancellable(delay, &options.cancel).await?;
                }
            },
        }
    }
}
