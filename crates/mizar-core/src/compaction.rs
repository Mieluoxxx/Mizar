//! Mizar 单阶段上下文压缩引擎。
//!
//! 整合核心工程实践：
//! - **单阶段检查点摘要**：步前按真实可用窗口 80% 触发（优先尾置指令复用主对话 KV Cache 前缀，
//!   失败自动回退纯净转写），配合确定性 `<mizar-digest>` 硬事实与膨胀保护；
//! - **溢出兜底恢复**：`agent_loop.request_error_hook()` 捕获 `ContextOverflow` 时以 `retain=0` 强制压缩并重试；
//! - **统一 `write` 写前读守卫**：对已存在且超过 50 行的文件执行整文件 `write` 前，
//!   要求本会话已 `read` 且该读取未被压出上下文，始终放行新建文件、`<=50` 行小文件与 `edit`。

pub mod budget;
pub mod digest;
pub mod guards;
pub mod pruning;
pub mod service;
pub mod summarizer;

use mizar_llm::LlmError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use budget::{
    ContextBudget, compute_budget, estimate_session_context_tokens, estimate_text_tokens,
};
pub use digest::{CutPlan, build_compaction_digest, plan_compaction_cut};
pub use pruning::{
    DEFAULT_IDLE_FLUSH_MS, DEFAULT_WARM_SUFFIX_TOKEN_LIMIT, PruneMarker, SUPERSEDED_NOTICE,
    TailPruneOptions, TailPrunePlan, TailPruneReport, USELESS_NOTICE, plan_tail_pruning,
    prune_session_tail,
};
pub use service::CompactionRuntime;
pub use summarizer::strip_summary_wrapper;

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("会话 `{0}` 正在执行压缩，请稍候")]
    AlreadyRunning(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionTrigger {
    AutoThreshold,
    Manual,
    Overflow,
}

impl CompactionTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AutoThreshold => "auto-threshold",
            Self::Manual => "manual",
            Self::Overflow => "overflow",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionOutcome {
    Compacted,
    Skipped,
    Failed,
}

impl CompactionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Compacted => "compacted",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub auto_compact: bool,
    pub default_context_window: u64,
    /// 可选上下文硬上限（默认 `None`，释放模型真实可用窗口；设为 `Some(cap)` 时按 `usable_tokens.min(cap)` 封顶）。
    pub max_context_tokens: Option<u64>,
    pub trigger_ratio: f64,
    pub keep_recent_tokens: u64,
    /// 是否裁剪被同文件后续完整 `read` 覆盖的旧 `read` 结果（默认 `true`）。
    pub supersede_reads: bool,
    /// 是否裁剪零命中/无信息量工具结果（默认 `true`）。
    pub drop_useless: bool,
    /// 热缓存期间尾部可裁剪的后续消息估算 token 上限（默认 `8_000`）。
    pub prune_warm_suffix_tokens: u64,
    /// 会话闲置达到该毫秒数后允许清仓最新压缩边界之后的全部可见候选（默认 90 分钟）。
    pub prune_idle_flush_ms: u128,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_compact: true,
            default_context_window: 128_000,
            max_context_tokens: None,
            trigger_ratio: 0.80,
            keep_recent_tokens: 20_000,
            supersede_reads: true,
            drop_useless: true,
            prune_warm_suffix_tokens: DEFAULT_WARM_SUFFIX_TOKEN_LIMIT,
            prune_idle_flush_ms: DEFAULT_IDLE_FLUSH_MS,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ManualCompactOptions {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub focus: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompactionReport {
    pub outcome: CompactionOutcome,
    pub trigger: CompactionTrigger,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub first_kept_seq: Option<u64>,
    pub summary_preview: Option<String>,
    pub skipped_reason: Option<String>,
}

impl CompactionReport {
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            outcome: CompactionOutcome::Skipped,
            trigger: CompactionTrigger::Manual,
            tokens_before: 0,
            tokens_after: 0,
            first_kept_seq: None,
            summary_preview: None,
            skipped_reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextPressureReport {
    pub estimated_tokens: u64,
    pub context_window: u64,
    pub trigger_tokens: u64,
    pub effective_window: u64,
    pub ratio: f64,
    pub compacted_up_to_seq: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CompactionStartEvent {
    pub session_id: String,
    pub trigger: CompactionTrigger,
    pub tokens_before: u64,
    pub context_window: u64,
}

#[derive(Debug, Clone)]
pub struct CompactionProgressEvent {
    pub session_id: String,
    pub stage: String,
    pub delta_chars: usize,
}

#[derive(Debug, Clone)]
pub struct CompactionEndEvent {
    pub session_id: String,
    pub trigger: CompactionTrigger,
    pub outcome: CompactionOutcome,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub first_kept_seq: Option<u64>,
    pub summary_preview: Option<String>,
    pub skipped_reason: Option<String>,
}
