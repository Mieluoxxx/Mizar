//! 内置模块控制器（`SpinePlugins`）：
//!
//! 在启动阶段静态装配全部可选内置模块后，以轻量 `AtomicBool` 门控与 `ToolRuntime` 分组屏蔽实现零销毁运行期启停
//! （仅重量级外部进程 `mcp` 在启动禁用时延迟到首次启用再初始化），并对核心内核服务实施 Gate 2 自保护。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;
#[cfg(feature = "mcp")]
use tokio::sync::Mutex as AsyncMutex;

use crate::compaction::CompactionRuntime;
use crate::instructions::AgentInstructions;
use crate::session_title::SessionTitleRuntime;
use crate::skills::SkillRegistry;
use crate::spine::config::normalize_disabled_plugins;
use crate::subagent::SubagentRuntime;
use crate::trajectory::TrajectoryRuntime;
#[cfg(feature = "mcp")]
use mizar_mcp::{McpManager, McpPluginConfig};
use mizar_tools::ToolRuntime;

/// 受 Gate 2 自保护的核心服务 ID：任何情况下均不允许禁用。
const PROTECTED_MODULES: &[(&str, &str, &str)] = &[
    (
        "llm",
        "模型运行时",
        "提供流式模型调用与协议适配器注册（核心内核服务）",
    ),
    (
        "sessions",
        "会话存储",
        "管理只追加会话日志、分支与持久化桥接（核心内核服务）",
    ),
    (
        "tools",
        "工具运行时",
        "管理全局工具注册表与三段执行管线（核心内核服务）",
    ),
    (
        "systemPrompt",
        "系统提示词",
        "按权重聚合与构建系统提示词段落（核心内核服务）",
    ),
    (
        "agentLoop",
        "Agent 主循环",
        "驱动模型流式推理与工具并发/串行执行循环（核心内核服务）",
    ),
    (
        "spinePlugins",
        "模块控制器",
        "管理内置模块启停开关与自保护策略（核心内核服务）",
    ),
];

/// 所有已知的内置可选模块稳定 ID（含 Cargo feature 下可能未激活的模块）。
const KNOWN_OPTIONAL_IDS: &[&str] = &[
    "tools-coding",
    "tool-search",
    "codemode",
    "mcp",
    "skills",
    "instructions",
    "compaction",
    "recall",
    "session-title",
    "subagent",
    "trajectory",
    "repeat-guard",
];

/// 官方客户端宿主与 TUI 扩展模块 ID（由上层宿主/TUI 读取 `disabledPlugins` 自行控制挂载，
/// `SpinePlugins` 识别为合法已知 ID，不触发未知模块告警）。
pub const KNOWN_HOST_EXTENSION_IDS: &[&str] = &[
    "file-mentions",
    "local-bash",
    "approval-gate",
    "tui",
    "tui-emphasis",
    "tui-side-panel",
    "tui-timeline-rail",
    "tui-trajectory-view",
    "tui-selection",
    "tui-clipboard",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpineModuleStatus {
    pub id: String,
    pub label: String,
    pub description: String,
    pub protected: bool,
    pub enabled: bool,
}

#[derive(Debug, Error)]
pub enum SpinePluginError {
    #[error("未知内置模块 `{0}`")]
    UnknownModule(String),
    #[error("核心受保护模块 `{0}` 禁止禁用")]
    ProtectedModule(String),
    #[error("模块 `{id}` 生命周期切换失败：{reason}")]
    Lifecycle { id: String, reason: String },
}

#[derive(Debug, Clone)]
pub(crate) struct SpineModuleMeta {
    pub id: String,
    pub label: String,
    pub description: String,
}

#[cfg(feature = "mcp")]
#[derive(Default)]
pub(crate) struct McpSlot {
    pub active: Option<Arc<McpManager>>,
    pub pending: Option<McpPluginConfig>,
}

pub(crate) struct ModuleToggle {
    meta: SpineModuleMeta,
    enabled: Arc<AtomicBool>,
}

impl ModuleToggle {
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    fn status(&self) -> SpineModuleStatus {
        SpineModuleStatus {
            id: self.meta.id.clone(),
            label: self.meta.label.clone(),
            description: self.meta.description.clone(),
            protected: false,
            enabled: self.is_enabled(),
        }
    }
}

pub fn is_protected_module_id(id: &str) -> bool {
    PROTECTED_MODULES.iter().any(|(pid, _, _)| *pid == id)
}

fn tool_group_for_module(id: &str) -> Option<(&'static [&'static str], &'static [&'static str])> {
    match id {
        "tools-coding" => Some((&["read", "bash", "edit", "write", "grep", "find"], &[])),
        "tool-search" => Some((&["tool_search"], &[])),
        "codemode" => Some((&["codemode"], &[])),
        "mcp" => Some((
            &[
                "list_mcp_resources",
                "list_mcp_resource_templates",
                "read_mcp_resource",
            ],
            &["mcp__"],
        )),
        "skills" => Some((&["skill"], &[])),
        "recall" => Some((&["recall"], &[])),
        "subagent" => Some((&["subagent"], &[])),
        _ => None,
    }
}

pub(crate) struct SpineServiceRefs {
    pub tools: Arc<ToolRuntime>,
    pub compaction: Option<Arc<CompactionRuntime>>,
    pub instructions: Option<Arc<AgentInstructions>>,
    pub skills: Option<Arc<SkillRegistry>>,
    pub session_title: Option<Arc<SessionTitleRuntime>>,
    pub subagent: Option<Arc<SubagentRuntime>>,
    pub trajectory: Option<Arc<TrajectoryRuntime>>,
    #[cfg(feature = "mcp")]
    pub mcp_slot: Arc<AsyncMutex<McpSlot>>,
}

pub(crate) struct PreparedToggles {
    pub modules: Vec<ModuleToggle>,
    pub startup_errors: Vec<String>,
}

impl PreparedToggles {
    pub fn prepare(metas: Vec<SpineModuleMeta>, raw_disabled: Vec<String>) -> Self {
        let normalized_disabled = normalize_disabled_plugins(raw_disabled);
        let available_ids: HashSet<&str> = metas.iter().map(|m| m.id.as_str()).collect();
        let mut disabled_set: HashSet<String> = HashSet::new();
        let mut startup_errors = Vec::new();

        for id in normalized_disabled {
            if is_protected_module_id(&id) {
                let msg =
                    format!("核心受保护模块 `{id}` 禁止禁用，已忽略 disabledPlugins 中的屏蔽项");
                tracing::warn!(target: "mizar::spine-plugins", "{msg}");
                startup_errors.push(msg);
            } else if available_ids.contains(id.as_str()) {
                disabled_set.insert(id);
            } else if !KNOWN_OPTIONAL_IDS.contains(&id.as_str())
                && !KNOWN_HOST_EXTENSION_IDS.contains(&id.as_str())
            {
                let msg = format!("未知内置模块 `{id}`，已忽略 disabledPlugins 中的配置");
                tracing::warn!(target: "mizar::spine-plugins", "{msg}");
                startup_errors.push(msg);
            }
        }

        let modules = metas
            .into_iter()
            .map(|meta| {
                let enabled = !disabled_set.contains(&meta.id);
                ModuleToggle {
                    meta,
                    enabled: Arc::new(AtomicBool::new(enabled)),
                }
            })
            .collect();

        Self {
            modules,
            startup_errors,
        }
    }

    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub fn is_enabled(&self, id: &str) -> bool {
        self.modules
            .iter()
            .find(|m| m.meta.id == id)
            .is_some_and(ModuleToggle::is_enabled)
    }

    pub fn flag_for(&self, id: &str) -> Option<Arc<AtomicBool>> {
        self.modules
            .iter()
            .find(|m| m.meta.id == id)
            .map(|m| Arc::clone(&m.enabled))
    }
}

/// 内置模块控制器（持有直接的运行时引用与原子开关，零动态查表）。
pub struct SpinePlugins {
    tools: Arc<ToolRuntime>,
    compaction: Option<Arc<CompactionRuntime>>,
    instructions: Option<Arc<AgentInstructions>>,
    skills: Option<Arc<SkillRegistry>>,
    session_title: Option<Arc<SessionTitleRuntime>>,
    subagent: Option<Arc<SubagentRuntime>>,
    trajectory: Option<Arc<TrajectoryRuntime>>,
    modules: Vec<ModuleToggle>,
    startup_errors: Vec<String>,
    #[cfg(feature = "mcp")]
    mcp_slot: Arc<AsyncMutex<McpSlot>>,
}

impl SpinePlugins {
    pub(crate) fn from_prepared(prepared: PreparedToggles, refs: SpineServiceRefs) -> Arc<Self> {
        Arc::new(Self {
            tools: refs.tools,
            compaction: refs.compaction,
            instructions: refs.instructions,
            skills: refs.skills,
            session_title: refs.session_title,
            subagent: refs.subagent,
            trajectory: refs.trajectory,
            modules: prepared.modules,
            startup_errors: prepared.startup_errors,
            #[cfg(feature = "mcp")]
            mcp_slot: refs.mcp_slot,
        })
    }

    fn sync_service_gate(&self, id: &str, enabled: bool) {
        match id {
            "compaction" => {
                if let Some(rt) = &self.compaction {
                    rt.set_module_enabled(enabled);
                }
            }
            "instructions" => {
                if let Some(rt) = &self.instructions {
                    rt.set_enabled(enabled);
                }
            }
            "skills" => {
                if let Some(rt) = &self.skills {
                    rt.set_enabled(enabled);
                }
            }
            "session-title" => {
                if let Some(rt) = &self.session_title {
                    rt.set_module_enabled(enabled);
                }
            }
            "subagent" => {
                if let Some(rt) = &self.subagent {
                    rt.set_module_enabled(enabled);
                }
            }
            "trajectory" => {
                if let Some(rt) = &self.trajectory {
                    rt.set_enabled(enabled);
                }
            }
            _ => {}
        }
    }

    /// 同步将启动期处于禁用状态的工具分组与模块开关应用到运行时。
    pub(crate) fn apply_initial_gates(&self) {
        for m in &self.modules {
            if m.is_enabled() {
                continue;
            }
            let id = &m.meta.id;
            if let Some((names, prefixes)) = tool_group_for_module(id) {
                self.tools
                    .set_disabled_tool_group(id, names, prefixes, true);
            }
            self.sync_service_gate(id, false);
        }
    }

    /// 启动阶段校验 `disabledPlugins` 时记录的拒绝或告警信息。
    pub fn startup_errors(&self) -> Vec<String> {
        self.startup_errors.clone()
    }

    /// 列出全部受管模块状态（包含受保护核心模块与当前运行时可选模块）。
    pub fn list(&self) -> Vec<SpineModuleStatus> {
        let mut out: Vec<SpineModuleStatus> = PROTECTED_MODULES
            .iter()
            .map(|(id, label, desc)| SpineModuleStatus {
                id: (*id).to_string(),
                label: (*label).to_string(),
                description: (*desc).to_string(),
                protected: true,
                enabled: true,
            })
            .collect();
        out.extend(self.modules.iter().map(ModuleToggle::status));
        out
    }

    /// 列出当前运行时可用的可选模块状态（供 `/settings` `[模块]` 分组展示）。
    pub fn optional_modules(&self) -> Vec<SpineModuleStatus> {
        self.modules.iter().map(ModuleToggle::status).collect()
    }

    /// 查询单个模块状态。
    pub fn get(&self, id: &str) -> Option<SpineModuleStatus> {
        if let Some((pid, label, desc)) = PROTECTED_MODULES.iter().find(|(pid, _, _)| *pid == id) {
            return Some(SpineModuleStatus {
                id: (*pid).to_string(),
                label: (*label).to_string(),
                description: (*desc).to_string(),
                protected: true,
                enabled: true,
            });
        }
        self.modules
            .iter()
            .find(|m| m.meta.id == id)
            .map(ModuleToggle::status)
    }

    pub fn is_protected(&self, id: &str) -> bool {
        is_protected_module_id(id)
    }

    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub fn is_enabled(&self, id: &str) -> bool {
        if is_protected_module_id(id) {
            return true;
        }
        self.modules
            .iter()
            .find(|m| m.meta.id == id)
            .is_some_and(ModuleToggle::is_enabled)
    }

    /// 返回当前处于禁用状态的可选模块 ID（已排序去重，可直接回写 `"disabledPlugins"`）。
    pub fn disabled_ids(&self) -> Vec<String> {
        normalize_disabled_plugins(
            self.modules
                .iter()
                .filter(|m| !m.is_enabled())
                .map(|m| m.meta.id.clone()),
        )
    }

    fn apply_runtime_state(&self, id: &str, enabled: bool) {
        if let Some((names, prefixes)) = tool_group_for_module(id) {
            self.tools
                .set_disabled_tool_group(id, names, prefixes, !enabled);
        }
        self.sync_service_gate(id, enabled);
    }

    /// 热切换指定可选内置模块的启用状态。
    ///
    /// 返回 `Ok(true)` 表示状态发生切换；`Ok(false)` 表示状态未变。
    pub async fn set_enabled(&self, id: &str, enabled: bool) -> Result<bool, SpinePluginError> {
        if is_protected_module_id(id) {
            if !enabled {
                return Err(SpinePluginError::ProtectedModule(id.to_string()));
            }
            return Ok(false);
        }

        let Some(toggle) = self.modules.iter().find(|m| m.meta.id == id) else {
            return Err(SpinePluginError::UnknownModule(id.to_string()));
        };
        if toggle.is_enabled() == enabled {
            return Ok(false);
        }

        #[cfg(feature = "mcp")]
        if id == "mcp" && enabled {
            let mut guard = self.mcp_slot.lock().await;
            if let Some(cfg) = guard.pending.take() {
                let mgr = McpManager::start(Arc::clone(&self.tools), cfg).await;
                guard.active = Some(mgr);
            }
        }

        toggle.enabled.store(enabled, Ordering::Release);
        self.apply_runtime_state(id, enabled);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spine::{AgentSpineConfig, MizarRuntime};
    use crate::system_prompt::{PromptSection, SystemPrompt, SystemPromptConfig};
    use mizar_tools::ToolRuntime;
    use mizar_tools::coding::{CodingToolsConfig, build_tools_guide, register_coding_tools};

    #[tokio::test]
    async fn core_modules_are_protected_under_minimal_features() {
        let cfg = AgentSpineConfig::new(std::env::temp_dir()).with_disabled_plugins([
            "llm",
            "sessions",
            "tools",
            "systemPrompt",
            "agentLoop",
            "spinePlugins",
            "unknown-mod",
        ]);
        let runtime = MizarRuntime::build(cfg).await.unwrap();
        let ctrl = &runtime.modules;

        let errs = ctrl.startup_errors();
        for core in [
            "llm",
            "sessions",
            "tools",
            "systemPrompt",
            "agentLoop",
            "spinePlugins",
        ] {
            assert!(ctrl.is_protected(core));
            assert!(ctrl.is_enabled(core));
            assert!(
                errs.iter().any(|e| e.contains(core)),
                "expected startup rejection error for {core}, got: {errs:?}"
            );
            let res = ctrl.set_enabled(core, false).await;
            assert!(
                matches!(res, Err(SpinePluginError::ProtectedModule(ref m)) if m == core),
                "expected ProtectedModule({core}), got {res:?}"
            );
        }
        assert!(errs.iter().any(|e| e.contains("unknown-mod")));
        assert!(ctrl.disabled_ids().is_empty());

        let unknown_res = ctrl.set_enabled("does-not-exist", false).await;
        assert!(matches!(
            unknown_res,
            Err(SpinePluginError::UnknownModule(_))
        ));

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn known_host_extension_ids_are_skipped_without_startup_errors() {
        let mut disabled: Vec<&str> = KNOWN_HOST_EXTENSION_IDS.to_vec();
        disabled.push("still-unknown-id");
        let cfg = AgentSpineConfig::new(std::env::temp_dir()).with_disabled_plugins(disabled);
        let runtime = MizarRuntime::build(cfg).await.unwrap();

        let ctrl = &runtime.modules;
        let errs = ctrl.startup_errors();
        assert_eq!(
            errs.len(),
            1,
            "expected only unknown id error, got: {errs:?}"
        );
        assert!(errs[0].contains("still-unknown-id"), "got: {errs:?}");
        assert!(ctrl.disabled_ids().is_empty());

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn coding_mode_toggles_tools_coding_with_tools_guide_and_tool_search() {
        let ws =
            std::env::temp_dir().join(format!("mizar_spine_coding_toggle_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();

        let mut cfg =
            AgentSpineConfig::coding(&ws).with_disabled_plugins(["tool-search", "mcp", "codemode"]);
        cfg.subagent = None;
        let runtime = MizarRuntime::build(cfg).await.unwrap();

        let ctrl = &runtime.modules;
        let tools = &runtime.tools;
        let prompt = &runtime.system_prompt;
        assert!(!ctrl.is_enabled("tool-search"));
        assert!(ctrl.is_enabled("tools-coding"));
        assert!(!tools.list_all().iter().any(|t| t.name == "tool_search"));
        assert!(tools.list().iter().any(|t| t.name == "read"));
        assert!(prompt.build().contains("Available tools:"));

        assert!(ctrl.set_enabled("tool-search", true).await.unwrap());
        assert!(ctrl.is_enabled("tool-search"));
        assert!(tools.list_all().iter().any(|t| t.name == "tool_search"));
        // 0 个外部 MCP 工具时，启用 tool-search 仍保持按需激活语义（不向模型默认声明）
        assert!(!tools.list().iter().any(|t| t.name == "tool_search"));

        assert!(ctrl.set_enabled("tools-coding", false).await.unwrap());
        assert!(!ctrl.is_enabled("tools-coding"));
        assert!(!tools.list().iter().any(|t| t.name == "read"));
        assert!(!tools.list().iter().any(|t| t.name == "bash"));
        assert!(!prompt.build().contains("Available tools:"));

        assert!(ctrl.set_enabled("tools-coding", true).await.unwrap());
        assert!(tools.list().iter().any(|t| t.name == "read"));
        assert!(prompt.build().contains("Available tools:"));

        runtime.shutdown().await;
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn tools_guide_renders_english_guidance_and_respects_session_tool_scope() {
        let root = std::env::temp_dir().join(format!(
            "mizar_coding_late_prompt_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let tools = ToolRuntime::new();
        let _tool_guards = register_coding_tools(&tools, &CodingToolsConfig::new(&root)).unwrap();
        let prompt = SystemPrompt::new(SystemPromptConfig::default());

        let tools_for_section = Arc::clone(&tools);
        let guide_guard = prompt.section(PromptSection::new_for_session(
            "tools_guide",
            20,
            move |session_id| build_tools_guide(&tools_for_section.list_declared_infos(session_id)),
        ));

        let built = prompt.build();
        assert!(built.contains("Available tools:"), "prompt:\n{built}");
        assert!(built.contains("Tool usage:"), "prompt:\n{built}");
        assert!(built.contains("edits[]"), "prompt:\n{built}");
        assert!(
            built.contains("never run `cat`, `head`, `tail`, `sed`, `grep`, or `find` via `bash`"),
            "prompt:\n{built}"
        );
        assert!(
            built.contains("Batch independent tool calls in one turn"),
            "prompt:\n{built}"
        );
        assert!(
            !built
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
            "tools_guide must be pure English:\n{built}"
        );

        tools.set_active_tools("s-ro", &["read".into()]);
        let built_ro = prompt.build_for_session(Some("s-ro"));
        assert!(built_ro.contains("- `read`:"), "s-ro:\n{built_ro}");
        assert!(!built_ro.contains("- `bash`:"), "s-ro:\n{built_ro}");
        assert!(
            prompt
                .build_for_session(Some("s-other"))
                .contains("- `bash`:")
        );

        drop(guide_guard);
        assert!(!prompt.build().contains("Available tools:"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
