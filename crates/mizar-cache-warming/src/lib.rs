//! Prompt Cache 后台定时续命保活插件。
//!
//! 监听主循环请求回执（`llm/after-response`）、回合边界（`agent/turn-start` / `agent/turn-end`）
//! 与会话事件（`session/event`），按四道硬门禁与 `$0.05` 期望净收益测算决定是否通过
//! `LlmRuntime::stream_direct` 发起最小 `max_tokens=1` 无重试保活重放；
//! 成功后写入 `SessionEvent::Usage(kind = "cache_warm")` 并同步刷新 `compaction` 活跃时间戳。

pub mod service;

pub use mizar_client_core::CacheWarmingSessionStatus;
pub use mizar_llm::CacheWarmingMode;
pub use service::{
    CacheWarmingConfig, CacheWarmingContextExt, CacheWarmingError, CacheWarmingNoticeEvent,
    CacheWarmingRuntime, EVENT_CACHE_WARMING_NOTICE, SERVICE_CACHE_WARMING,
};
