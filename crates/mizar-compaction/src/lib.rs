//! Mizar 四级渐进上下文压缩引擎（对应 `references/压缩与记忆.md` 综合设计方案）。
//!
//! 整合各参考项目的工程精华并贴合 Mizar 微内核边界：
//! - **L1 零模型批量剪枝**（`deepseek-harness` `ToolResultPruner` + `billion-context` 锁存视图）
//! - **L2 软硬分离检查点摘要**（`dim` 确定性 `<mizar-digest>` + `<= 0.5*B` Re-plan + 实证关闭思考、
//!   `obelisk` 已证实事实/未证实推断分级、`dsh` 同前缀尾置指令与 `answeredRoute`）
//! - **L3 溢出与隐性连续截断自动恢复**（`pi` 0 输出 `length` + `DS-Bot` `cutOffs >= 2`）
//! - **压后工具安全护栏**（`dim` `FileReads` 拦 `write` 放 `edit` + `dsh-ops` 跨压缩感知 `read-dedup`）

pub mod budget;
pub mod digest;
pub mod guards;
pub mod pruner;
pub mod service;
pub mod summarizer;

use cordis::CordisError;
use mizar_llm::LlmError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use budget::{
    ContextBudget, compute_budget, estimate_session_context_tokens, estimate_text_tokens,
};
pub use digest::{CutPlan, build_compaction_digest, plan_compaction_cut};
pub use pruner::{PrunePlan, plan_tool_result_prune};
pub use service::{CompactionContextExt, CompactionRuntime};
pub use summarizer::strip_summary_wrapper;

pub const EVENT_COMPACTION_START: &str = "compaction/start";
pub const EVENT_COMPACTION_PROGRESS: &str = "compaction/progress";
pub const EVENT_COMPACTION_END: &str = "compaction/end";

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error(transparent)]
    Cordis(#[from] CordisError),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("会话 `{0}` 正在执行压缩，请稍候")]
    AlreadyRunning(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionTrigger {
    AutoThreshold,
    ColdCache,
    Manual,
    Overflow,
}

impl CompactionTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AutoThreshold => "auto-threshold",
            Self::ColdCache => "cold-cache",
            Self::Manual => "manual",
            Self::Overflow => "overflow",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompactionOutcome {
    PrunedOnly,
    Compacted,
    Skipped,
    Failed,
}

impl CompactionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrunedOnly => "pruned-only",
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
    pub auto_prune: bool,
    pub default_context_window: u64,
    /// 超大窗口软顶（默认 380k，避免 1M 窗口延迟压缩导致注意力与缓存效益下降；设为 0 关闭软顶）。
    pub soft_window_cap: u64,
    pub trigger_ratio: f64,
    pub prune_target_ratio: f64,
    pub post_compact_max_ratio: f64,
    pub keep_recent_ratio: f64,
    pub regrowth_ratio: f64,
    pub prune_threshold_chars: usize,
    pub prune_head_chars: usize,
    pub prune_tail_chars: usize,
    pub prune_protect_recent_surface: usize,
    /// 空闲超过该秒数后若上下文过半则允许冷缓存前置整理（默认 300s，设为 0 关闭）。
    pub cold_cache_idle_secs: u64,
    /// 优先用尾置指令复用主对话 KV Cache 前缀（默认 true）。
    pub reuse_prefix_cache: bool,
    /// 生成摘要时默认关闭推理思考以防挤占输出预算（默认 true，遵循 `dim #170` 实证结论）。
    pub disable_summary_reasoning: bool,
    /// 跨压缩感知的 `read-dedup`（默认 true）。
    pub enable_read_dedup: bool,
    /// 压后禁止对已被压出上下文的已存在文件做整文件 `write`（默认 true，放行 `edit`）。
    pub enable_write_after_compact_guard: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_compact: true,
            auto_prune: true,
            default_context_window: 128_000,
            soft_window_cap: 380_000,
            trigger_ratio: 0.80,
            prune_target_ratio: 0.55,
            post_compact_max_ratio: 0.50,
            keep_recent_ratio: 0.20,
            regrowth_ratio: 0.25,
            prune_threshold_chars: 6_000,
            prune_head_chars: 1_800,
            prune_tail_chars: 800,
            prune_protect_recent_surface: 4,
            cold_cache_idle_secs: 300,
            reuse_prefix_cache: true,
            disable_summary_reasoning: true,
            enable_read_dedup: true,
            enable_write_after_compact_guard: true,
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
    pub pruned_count: usize,
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
            pruned_count: 0,
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
    pub pruned_count: usize,
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
    pub pruned_count: usize,
    pub first_kept_seq: Option<u64>,
    pub summary_preview: Option<String>,
    pub skipped_reason: Option<String>,
}
