//! 被动 `Cache Re-billed` 浪费审计契约重导出（实现位于 `mizar-llm-types::cache_audit`）。

pub use mizar_llm_types::cache_audit::{
    CacheMissAudit, MIN_CACHE_DROP_TOKENS, UsageAuditSample, calculate_usage_cost,
    compute_cache_miss_audit,
};
