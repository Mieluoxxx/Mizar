//! `AGENTS.md` 与 `CLAUDE.md` 多层级发现、去重、预算裁切与 `ContextNotice` 动态注入插件。
//!
//! - 全局目录：`~/.dsh-rs/`（支持 `DSH_RS_HOME` 覆盖）与 `~/.agents/`（支持 `DSH_AGENTS_HOME` 覆盖）
//! - 项目祖先链：从 `project_root`（向上查找 `.git`）到 `workspace`，按 `AGENTS.md` / `CLAUDE.md` 与 `.local.md` 顺序加载
//! - 子目录按需加载：监听 `tools/result`，在成功 `read` / `write` / `edit` 命中子目录文件后，于下一步自动补充增量指令
//! - 动态变更与 Compaction：热感知文件修改/删除，在历史压缩后自动重建基线

pub mod config;
pub mod digest;
pub mod files;
pub mod render;
pub mod state;

pub use config::{
    AGENT_INSTRUCTIONS_SOURCE, DEFAULT_MAX_BYTES, DEFAULT_MAX_SOURCE_BYTES, InstructionScopeKind,
    InstructionScopeStatus, InstructionsConfig, SessionInstructionsReport,
    default_instruction_candidates, default_local_instruction_candidates,
    default_project_root_markers,
};
pub use digest::{sha256_hex, trimmed_instruction_digest};
pub use files::{
    ChainKind, DiscoveredFile, DiscoveryResult, FsFingerprint, ScopeKey, ScopeRoot,
    descendant_dirs_between, discover_instructions, find_project_root, resolve_agents_home,
    resolve_dsh_home,
};
pub use state::AgentInstructions;

use cordis::{Context, CordisError};
use std::sync::Arc;

pub trait InstructionsContextExt {
    fn agent_instructions(&self) -> Result<Arc<AgentInstructions>, CordisError>;
}

impl InstructionsContextExt for Context {
    fn agent_instructions(&self) -> Result<Arc<AgentInstructions>, CordisError> {
        self.service::<AgentInstructions>("agentInstructions")
    }
}
