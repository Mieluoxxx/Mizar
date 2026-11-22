//! 抽象 LLM 服务（对应 `NamiDSH/packages/llm`）。
//!
//! 定义 `GenerateOptions`、`LlmAdapter` 接口与 `ctx.llm` 运行时服务；
//! 核心纯数据结构、错误分类、组装器与 SSE 解析器由 `mizar-llm-types` 提供并重导出；
//! HTTP/SSE 传输辅助由独立 crate `mizar-llm-http` 提供。

pub mod cache_warming;
pub mod runtime;
pub mod types;

pub use cache_warming::{
    CACHE_WARMING_IDLE_CONTINUATION_PROBABILITY, CACHE_WARMING_MAX_AGE_MS,
    CACHE_WARMING_MAX_IDLE_AGE_MS, CACHE_WARMING_MIN_TTL_MS,
    CACHE_WARMING_MINIMUM_EXPECTED_SAVINGS, CacheWarmingDecision, CacheWarmingEvaluation,
    CacheWarmingPhase, calculate_cost, evaluate_cache_warming, get_cache_warming_delay_ms,
    get_cache_warming_refresh_deadline_offset_ms, get_prompt_cache_ttl_ms,
    is_cache_warming_replayable,
};
pub use mizar_llm_types::*;
pub use runtime::{
    BoxChunkStream, EVENT_ADAPTERS_UPDATED, EVENT_LLM_AFTER_RESPONSE, EVENT_LLM_STREAM, LlmAdapter,
    LlmAfterResponseEvent, LlmContextExt, LlmRuntime,
};
pub use types::{GenerateOptions, resolve_cache_retention};
