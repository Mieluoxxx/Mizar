//! 抽象 LLM 服务（对应 `NamiDSH/packages/llm`）。
//!
//! 定义 `GenerateOptions`、`LlmAdapter` 接口与 `ctx.llm` 运行时服务；
//! 核心纯数据结构、错误分类、组装器与 SSE 解析器由 `dsh-llm-types` 提供并重导出；
//! HTTP/SSE 传输辅助由独立 crate `dsh-llm-http` 提供。

pub mod runtime;
pub mod types;

pub use dsh_llm_types::*;
pub use runtime::{
    BoxChunkStream, EVENT_ADAPTERS_UPDATED, EVENT_LLM_STREAM, LlmAdapter, LlmContextExt, LlmRuntime,
};
pub use types::GenerateOptions;
