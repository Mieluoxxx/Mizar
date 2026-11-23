//! 内置模块控制器（`"spinePlugins"`）：
//!
//! 在启动阶段静态挂载全部可选内置模块后，以轻量 `AtomicBool` 门控与 `ToolRuntime` 分组屏蔽实现零销毁运行期启停
//! （仅重量级外部进程 `mcp` 在启动禁用时延迟到首次启用再初始化），并对核心内核服务实施 Gate 2 自保护。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(any(feature = "mcp", test))]
use cordis::Service;
use cordis::{BoxError, BoxFuture, Cleanup, Context, CordisError, Inject, Plugin};
use thiserror::Error;
#[cfg(feature = "mcp")]
use tokio::sync::Mutex as AsyncMutex;

use crate::compaction::CompactionRuntime;
use crate::instructions::AgentInstructions;
use crate::session_title::SessionTitleRuntime;
use crate::skills::SkillRegistry;
use crate::spine::config::normalize_disabled_plugins;
use crate::subagent::SubagentRuntime;
use crate::system_prompt::{PromptSection, SystemPrompt};
use crate::trajectory::TrajectoryRuntime;
#[cfg(feature = "mcp")]
use mizar_mcp::{McpManager, McpPluginConfig};
use mizar_tools::coding::build_tools_guide;
use mizar_tools::{ToolRuntime, ToolsContextExt};

pub const SERVICE_SPINE_PLUGINS: &str = "spinePlugins";

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
    #[error(transparent)]
    Cordis(#[from] CordisError),
}

#[derive(Debug, Clone)]
pub(crate) struct SpineModuleMeta {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// 向 `SystemPrompt` 动态注册 `order = 20` 的 `"tools_guide"` 段落（按当前会话已声明工具过滤）。
#[derive(Default, Clone, Copy)]
pub struct ToolsGuidePromptPlugin;

impl Plugin for ToolsGuidePromptPlugin {
    type Config = ();

    fn name(&self) -> Option<&str> {
        Some("tools-guide-prompt")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        (): Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let tools = ctx.tools()?;
            let prompt_handle = ctx.try_inject(["systemPrompt"], move |c| {
                let tools = Arc::clone(&tools);
                async move {
                    let sp = c.service::<SystemPrompt>("systemPrompt")?;
                    let gate_ctx = c.clone();
                    c.effect("tools-coding.system-prompt", move || {
                        Ok::<_, CordisError>(Some(sp.section(PromptSection::new_for_session(
                            "tools_guide",
                            20,
                            move |session_id| {
                                if let Some(plugins) =
                                    gate_ctx.get::<SpinePlugins>(SERVICE_SPINE_PLUGINS)
                                    && !plugins.is_enabled("tools-coding")
                                {
                                    return None;
                                }
                                build_tools_guide(&tools.list_declared_infos(session_id))
                            },
                        ))))
                    })?;
                    Ok::<_, CordisError>(())
                }
            })?;
            if ctx.get::<SystemPrompt>("systemPrompt").is_some() {
                prompt_handle.wait_settled().await?;
            }
            Ok(Some(Cleanup::async_fn(move || async move {
                prompt_handle.dispose().await;
            })))
        })
    }
}

struct ModuleToggle {
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

fn service_name_for_module(id: &str) -> Option<&'static str> {
    match id {
        "compaction" => Some("compaction"),
        "instructions" => Some("agentInstructions"),
        "skills" => Some("skills"),
        "session-title" => Some("sessionTitle"),
        "subagent" => Some("subagent"),
        "trajectory" => Some("trajectory"),
        _ => None,
    }
}

fn sync_service_gate(ctx: &Context, id: &str, enabled: bool) {
    match id {
        "compaction" => {
            if let Some(rt) = ctx.get::<CompactionRuntime>("compaction") {
                rt.set_module_enabled(enabled);
            }
        }
        "instructions" => {
            if let Some(rt) = ctx.get::<AgentInstructions>("agentInstructions") {
                rt.set_enabled(enabled);
            }
        }
        "skills" => {
            if let Some(rt) = ctx.get::<SkillRegistry>("skills") {
                rt.set_enabled(enabled);
            }
        }
        "session-title" => {
            if let Some(rt) = ctx.get::<SessionTitleRuntime>("sessionTitle") {
                rt.set_module_enabled(enabled);
            }
        }
        "subagent" => {
            if let Some(rt) = ctx.get::<SubagentRuntime>("subagent") {
                rt.set_module_enabled(enabled);
            }
        }
        "trajectory" => {
            if let Some(rt) = ctx.get::<TrajectoryRuntime>("trajectory") {
                rt.set_enabled(enabled);
            }
        }
        _ => {}
    }
}

/// 内置模块控制器服务（`"spinePlugins"`）。
pub struct SpinePlugins {
    ctx: Context,
    modules: Vec<ModuleToggle>,
    startup_errors: Vec<String>,
    #[cfg(feature = "mcp")]
    mcp_pending: AsyncMutex<Option<McpPluginConfig>>,
}

impl SpinePlugins {
    pub(crate) fn initialize(
        ctx: Context,
        metas: Vec<SpineModuleMeta>,
        raw_disabled: Vec<String>,
    ) -> Arc<Self> {
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

        Arc::new(Self {
            ctx,
            modules,
            startup_errors,
            #[cfg(feature = "mcp")]
            mcp_pending: AsyncMutex::new(None),
        })
    }

    #[cfg(feature = "mcp")]
    pub(crate) fn defer_mcp_mount(&self, cfg: McpPluginConfig) {
        if let Ok(mut guard) = self.mcp_pending.try_lock() {
            *guard = Some(cfg);
        }
    }

    pub(crate) fn flag_for(&self, id: &str) -> Option<Arc<AtomicBool>> {
        self.modules
            .iter()
            .find(|m| m.meta.id == id)
            .map(|m| Arc::clone(&m.enabled))
    }

    /// 对启动阶段被屏蔽的可选模块挂接一次性状态同步，在对应服务就绪后读取实时开关。
    pub(crate) fn attach_initial_gates(&self) -> Result<(), CordisError> {
        let disabled_tool_modules: Vec<(String, Arc<AtomicBool>)> = self
            .modules
            .iter()
            .filter(|m| !m.is_enabled() && tool_group_for_module(&m.meta.id).is_some())
            .map(|m| (m.meta.id.clone(), Arc::clone(&m.enabled)))
            .collect();

        if !disabled_tool_modules.is_empty() {
            self.ctx.try_inject(["tools"], move |c| {
                let groups = disabled_tool_modules.clone();
                async move {
                    if let Ok(tools) = c.tools() {
                        for (id, flag) in groups {
                            if !flag.load(Ordering::Acquire)
                                && let Some((names, prefixes)) = tool_group_for_module(&id)
                            {
                                tools.set_disabled_tool_group(&id, names, prefixes, true);
                            }
                        }
                    }
                    Ok::<_, CordisError>(())
                }
            })?;
        }

        for m in &self.modules {
            if m.is_enabled() {
                continue;
            }
            let Some(svc) = service_name_for_module(&m.meta.id) else {
                continue;
            };
            let id = m.meta.id.clone();
            let flag = Arc::clone(&m.enabled);
            self.ctx.try_inject([svc], move |c| {
                let id = id.clone();
                let flag = Arc::clone(&flag);
                async move {
                    sync_service_gate(&c, &id, flag.load(Ordering::Acquire));
                    Ok::<_, CordisError>(())
                }
            })?;
        }
        Ok(())
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
        if let Some((names, prefixes)) = tool_group_for_module(id)
            && let Some(tools) = self.ctx.get::<ToolRuntime>("tools")
        {
            tools.set_disabled_tool_group(id, names, prefixes, !enabled);
        }
        sync_service_gate(&self.ctx, id, enabled);
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
            let mut guard = self.mcp_pending.lock().await;
            if let Some(cfg) = guard.clone() {
                let handle = self.ctx.try_plugin(McpManager::plugin(), cfg)?;
                if let Err(err) = handle.wait_settled().await {
                    handle.dispose().await;
                    return Err(SpinePluginError::Lifecycle {
                        id: id.to_string(),
                        reason: err.to_string(),
                    });
                }
                *guard = None;
            }
        }

        toggle.enabled.store(enabled, Ordering::Release);
        self.apply_runtime_state(id, enabled);
        Ok(true)
    }
}

pub trait SpinePluginsContextExt {
    fn spine_plugins(&self) -> Result<Arc<SpinePlugins>, CordisError>;
}

impl SpinePluginsContextExt for Context {
    fn spine_plugins(&self) -> Result<Arc<SpinePlugins>, CordisError> {
        self.service::<SpinePlugins>(SERVICE_SPINE_PLUGINS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loop::AgentLoopContextExt;
    use crate::spine::{AgentSpineConfig, AgentSpinePlugin};
    use crate::system_prompt::SystemPromptContextExt;
    use cordis::Context;
    use mizar_llm::LlmContextExt;
    use mizar_session::SessionContextExt;
    use mizar_tools::ToolsContextExt;
    use mizar_tools::coding::{CodingToolsConfig, CodingToolsPlugin};

    #[tokio::test]
    async fn core_modules_are_protected_under_minimal_features() {
        let app = Context::new();

        let cfg = AgentSpineConfig::new(std::env::temp_dir()).with_disabled_plugins([
            "llm",
            "sessions",
            "tools",
            "systemPrompt",
            "agentLoop",
            "spinePlugins",
            "unknown-mod",
        ]);
        app.plugin(AgentSpinePlugin, cfg).await.unwrap();
        for rt in app.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }

        let ctrl = app.spine_plugins().unwrap();
        assert!(app.llm().is_ok());
        assert!(app.sessions().is_ok());
        assert!(app.tools().is_ok());
        assert!(app.system_prompt().is_ok());
        assert!(app.agent_loop().is_ok());

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

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn known_host_extension_ids_are_skipped_without_startup_errors() {
        let app = Context::new();

        let mut disabled: Vec<&str> = KNOWN_HOST_EXTENSION_IDS.to_vec();
        disabled.push("still-unknown-id");
        let cfg = AgentSpineConfig::new(std::env::temp_dir()).with_disabled_plugins(disabled);
        app.plugin(AgentSpinePlugin, cfg).await.unwrap();
        for rt in app.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }

        let ctrl = app.spine_plugins().unwrap();
        let errs = ctrl.startup_errors();
        assert_eq!(
            errs.len(),
            1,
            "expected only unknown id error, got: {errs:?}"
        );
        assert!(errs[0].contains("still-unknown-id"), "got: {errs:?}");
        assert!(ctrl.disabled_ids().is_empty());

        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn coding_mode_toggles_tools_coding_with_tools_guide_and_tool_search() {
        let app = Context::new();

        let ws =
            std::env::temp_dir().join(format!("mizar_spine_coding_toggle_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();

        let mut cfg =
            AgentSpineConfig::coding(&ws).with_disabled_plugins(["tool-search", "mcp", "codemode"]);
        cfg.subagent = None;
        app.plugin(AgentSpinePlugin, cfg).await.unwrap();
        for rt in app.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }

        let ctrl = app.spine_plugins().unwrap();
        let tools = app.tools().unwrap();
        let prompt = app.system_prompt().unwrap();
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

        let _ = std::fs::remove_dir_all(&ws);
        app.fiber().dispose().await;
    }

    #[tokio::test]
    async fn registers_tools_guide_when_system_prompt_mounts_after_tools_guide_plugin() {
        use crate::system_prompt::SystemPromptConfig;
        use mizar_tools::ToolRuntime;

        let root = std::env::temp_dir().join(format!(
            "mizar_coding_late_prompt_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();

        // 先挂载 CodingToolsPlugin 与 ToolsGuidePromptPlugin，再挂载 SystemPrompt
        let coding = app.plugin(CodingToolsPlugin, CodingToolsConfig::new(&root));
        let guide = app.plugin(ToolsGuidePromptPlugin, ());
        coding.wait_settled().await.unwrap();
        guide.wait_settled().await.unwrap();
        assert!(app.get::<SystemPrompt>("systemPrompt").is_none());

        app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
            .await
            .unwrap();
        for rt in app.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }

        let prompt = app.service::<SystemPrompt>("systemPrompt").unwrap();
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

        let tools = app.tools().unwrap();
        tools.set_active_tools("s-ro", &["read".into()]);
        let built_ro = prompt.build_for_session(Some("s-ro"));
        assert!(built_ro.contains("- `read`:"), "s-ro:\n{built_ro}");
        assert!(!built_ro.contains("- `bash`:"), "s-ro:\n{built_ro}");
        assert!(
            prompt
                .build_for_session(Some("s-other"))
                .contains("- `bash`:")
        );

        guide.dispose().await;
        assert!(!prompt.build().contains("Available tools:"));

        let _ = std::fs::remove_dir_all(&root);
        app.fiber().dispose().await;
    }
}
