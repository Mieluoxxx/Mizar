//! 内置 Cordis 模块控制器（`"spinePlugins"`）：
//!
//! 统一管理 `mizar-agent-spine` 装配的可选内置 Cordis 模块生命周期（启动屏蔽与运行期热挂载/热卸载），
//! 并对核心内核服务（`llm`、`sessions`、`tools`、`systemPrompt`、`agentLoop`、`spinePlugins`）实施 Gate 2 自保护。

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, CordisError, Fiber, Inject, Plugin, Service};
use thiserror::Error;
use tokio::sync::Mutex as AsyncMutex;

use crate::compaction::{CompactionConfig, CompactionRuntime};
use crate::instructions::{AgentInstructions, InstructionsConfig};
use crate::session_title::{SessionTitleConfig, SessionTitleRuntime};
use crate::skills::{SkillRegistry, SkillRegistryConfig, SkillsConfig, SkillsPlugin};
use crate::subagent::{SubagentConfig, SubagentRuntime};
use crate::system_prompt::{PromptSection, SystemPrompt};
use crate::trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, TrajectoryConfig, TrajectoryRuntime,
};
#[cfg(feature = "codemode")]
use mizar_codemode::{CodemodeConfig, CodemodePlugin};
#[cfg(feature = "mcp")]
use mizar_mcp::{McpManager, McpPluginConfig};
use mizar_session::recall::{RecallConfig, RecallPlugin};
use mizar_tools::ToolsContextExt;
use mizar_tools::coding::{CodingToolsConfig, CodingToolsPlugin, build_tools_guide};
use mizar_tools::search::{ToolSearchConfig, ToolSearchPlugin};

use crate::spine::config::normalize_disabled_plugins;

pub const SERVICE_SPINE_PLUGINS: &str = "spinePlugins";
pub const EVENT_SPINE_PLUGINS_CHANGE: &str = "spinePlugins/change";

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
        "管理内置 Cordis 模块热启停与自保护策略（核心内核服务）",
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
pub struct SpinePluginsChangeEvent {
    pub id: String,
    pub enabled: bool,
}

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
            let prompt_fiber = ctx.try_inject(["systemPrompt"], move |c| {
                let tools = Arc::clone(&tools);
                async move {
                    let sp = c.service::<SystemPrompt>("systemPrompt")?;
                    c.effect("tools-coding.system-prompt", move || {
                        let tools = tools;
                        Ok::<_, CordisError>(Some(sp.section(PromptSection::new_for_session(
                            "tools_guide",
                            20,
                            move |session_id| {
                                build_tools_guide(&tools.list_declared_infos(session_id))
                            },
                        ))))
                    })?;
                    Ok::<_, CordisError>(())
                }
            })?;
            // 若 systemPrompt 已就绪则等待子 fiber 落稳，保证热启用返回后提示段立即可见；
            // 若尚未挂载（启动并发或后挂 SystemPrompt 单测）则不阻塞，待其到达后自动补注。
            if ctx.get::<SystemPrompt>("systemPrompt").is_some() {
                prompt_fiber.wait_settled().await?;
            }
            Ok(Some(Cleanup::async_fn(move || async move {
                prompt_fiber.dispose().await;
            })))
        })
    }
}

#[derive(Clone)]
pub(crate) enum ModuleSpec {
    ToolsCoding(CodingToolsConfig),
    ToolSearch(ToolSearchConfig),
    #[cfg(feature = "codemode")]
    Codemode(CodemodeConfig),
    #[cfg(feature = "mcp")]
    Mcp(McpPluginConfig),
    Skills(SkillsConfig),
    Instructions(InstructionsConfig),
    Compaction(CompactionConfig),
    Recall(RecallConfig),
    SessionTitle(SessionTitleConfig),
    Subagent(SubagentConfig),
    Trajectory(TrajectoryConfig),
    RepeatGuard(RepeatGuardConfig),
}

struct ModuleSlot {
    meta: SpineModuleMeta,
    spec: ModuleSpec,
    fibers: Vec<Fiber>,
}

impl ModuleSlot {
    fn is_enabled(&self) -> bool {
        !self.fibers.is_empty()
            && self.fibers.iter().all(|f| {
                matches!(
                    f.state(),
                    cordis::FiberState::Pending
                        | cordis::FiberState::Loading
                        | cordis::FiberState::Active
                )
            })
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

fn is_known_optional_id(id: &str) -> bool {
    KNOWN_OPTIONAL_IDS.contains(&id)
}

fn mount_module_spawn(ctx: &Context, spec: &ModuleSpec) -> Result<Vec<Fiber>, CordisError> {
    match *spec {
        ModuleSpec::ToolsCoding(ref cfg) => {
            let tools_fiber = ctx.try_plugin(CodingToolsPlugin, cfg.clone())?;
            if !cfg.inject_system_prompt {
                return Ok(vec![tools_fiber]);
            }
            let prompt_fiber = match ctx.try_plugin(ToolsGuidePromptPlugin, ()) {
                Ok(f) => f,
                Err(err) => {
                    tokio::spawn(tools_fiber.dispose());
                    return Err(err);
                }
            };
            Ok(vec![tools_fiber, prompt_fiber])
        }
        ModuleSpec::ToolSearch(ref cfg) => Ok(vec![ctx.try_plugin(ToolSearchPlugin, cfg.clone())?]),
        #[cfg(feature = "codemode")]
        ModuleSpec::Codemode(ref cfg) => Ok(vec![ctx.try_plugin(CodemodePlugin, cfg.clone())?]),
        #[cfg(feature = "mcp")]
        ModuleSpec::Mcp(ref cfg) => Ok(vec![ctx.try_plugin(McpManager::plugin(), cfg.clone())?]),
        ModuleSpec::Skills(ref cfg) => {
            let reg_fiber =
                ctx.try_plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())?;
            let plugin_fiber = match ctx.try_plugin(SkillsPlugin, cfg.clone()) {
                Ok(f) => f,
                Err(err) => {
                    tokio::spawn(reg_fiber.dispose());
                    return Err(err);
                }
            };
            Ok(vec![reg_fiber, plugin_fiber])
        }
        ModuleSpec::Instructions(ref cfg) => Ok(vec![
            ctx.try_plugin(AgentInstructions::plugin(), cfg.clone())?,
        ]),
        ModuleSpec::Compaction(ref cfg) => Ok(vec![
            ctx.try_plugin(CompactionRuntime::plugin(), cfg.clone())?,
        ]),
        ModuleSpec::Recall(ref cfg) => Ok(vec![ctx.try_plugin(RecallPlugin, cfg.clone())?]),
        ModuleSpec::SessionTitle(ref cfg) => Ok(vec![
            ctx.try_plugin(SessionTitleRuntime::plugin(), cfg.clone())?,
        ]),
        ModuleSpec::Subagent(ref cfg) => Ok(vec![
            ctx.try_plugin(SubagentRuntime::plugin(), cfg.clone())?,
        ]),
        ModuleSpec::Trajectory(ref cfg) => Ok(vec![
            ctx.try_plugin(TrajectoryRuntime::plugin(), cfg.clone())?,
        ]),
        ModuleSpec::RepeatGuard(ref cfg) => {
            Ok(vec![ctx.try_plugin(RepeatToolGuardPlugin, cfg.clone())?])
        }
    }
}

async fn mount_single_and_wait<P: cordis::Plugin>(
    ctx: &Context,
    id: &str,
    plugin: P,
    cfg: P::Config,
) -> Result<Vec<Fiber>, SpinePluginError> {
    let fiber = ctx.try_plugin(plugin, cfg)?;
    if let Err(err) = fiber.wait_settled().await {
        fiber.dispose().await;
        return Err(SpinePluginError::Lifecycle {
            id: id.to_string(),
            reason: err.to_string(),
        });
    }
    Ok(vec![fiber])
}

async fn mount_module_and_wait(
    ctx: &Context,
    id: &str,
    spec: &ModuleSpec,
) -> Result<Vec<Fiber>, SpinePluginError> {
    match *spec {
        ModuleSpec::ToolsCoding(ref cfg) => {
            let tools_fiber = ctx.try_plugin(CodingToolsPlugin, cfg.clone())?;
            if let Err(err) = tools_fiber.wait_settled().await {
                tools_fiber.dispose().await;
                return Err(SpinePluginError::Lifecycle {
                    id: id.to_string(),
                    reason: err.to_string(),
                });
            }
            if !cfg.inject_system_prompt {
                return Ok(vec![tools_fiber]);
            }
            let prompt_fiber = match ctx.try_plugin(ToolsGuidePromptPlugin, ()) {
                Ok(f) => f,
                Err(err) => {
                    tools_fiber.dispose().await;
                    return Err(err.into());
                }
            };
            if let Err(err) = prompt_fiber.wait_settled().await {
                prompt_fiber.dispose().await;
                tools_fiber.dispose().await;
                return Err(SpinePluginError::Lifecycle {
                    id: id.to_string(),
                    reason: err.to_string(),
                });
            }
            Ok(vec![tools_fiber, prompt_fiber])
        }
        ModuleSpec::ToolSearch(ref cfg) => {
            mount_single_and_wait(ctx, id, ToolSearchPlugin, cfg.clone()).await
        }
        #[cfg(feature = "codemode")]
        ModuleSpec::Codemode(ref cfg) => {
            mount_single_and_wait(ctx, id, CodemodePlugin, cfg.clone()).await
        }
        #[cfg(feature = "mcp")]
        ModuleSpec::Mcp(ref cfg) => {
            mount_single_and_wait(ctx, id, McpManager::plugin(), cfg.clone()).await
        }
        ModuleSpec::Skills(ref cfg) => {
            let reg_fiber =
                ctx.try_plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())?;
            if let Err(err) = reg_fiber.wait_settled().await {
                reg_fiber.dispose().await;
                return Err(SpinePluginError::Lifecycle {
                    id: id.to_string(),
                    reason: err.to_string(),
                });
            }
            let plugin_fiber = match ctx.try_plugin(SkillsPlugin, cfg.clone()) {
                Ok(f) => f,
                Err(err) => {
                    reg_fiber.dispose().await;
                    return Err(err.into());
                }
            };
            if let Err(err) = plugin_fiber.wait_settled().await {
                plugin_fiber.dispose().await;
                reg_fiber.dispose().await;
                return Err(SpinePluginError::Lifecycle {
                    id: id.to_string(),
                    reason: err.to_string(),
                });
            }
            Ok(vec![reg_fiber, plugin_fiber])
        }
        ModuleSpec::Instructions(ref cfg) => {
            mount_single_and_wait(ctx, id, AgentInstructions::plugin(), cfg.clone()).await
        }
        ModuleSpec::Compaction(ref cfg) => {
            mount_single_and_wait(ctx, id, CompactionRuntime::plugin(), cfg.clone()).await
        }
        ModuleSpec::Recall(ref cfg) => {
            mount_single_and_wait(ctx, id, RecallPlugin, cfg.clone()).await
        }
        ModuleSpec::SessionTitle(ref cfg) => {
            mount_single_and_wait(ctx, id, SessionTitleRuntime::plugin(), cfg.clone()).await
        }
        ModuleSpec::Subagent(ref cfg) => {
            mount_single_and_wait(ctx, id, SubagentRuntime::plugin(), cfg.clone()).await
        }
        ModuleSpec::Trajectory(ref cfg) => {
            mount_single_and_wait(ctx, id, TrajectoryRuntime::plugin(), cfg.clone()).await
        }
        ModuleSpec::RepeatGuard(ref cfg) => {
            mount_single_and_wait(ctx, id, RepeatToolGuardPlugin, cfg.clone()).await
        }
    }
}

/// 内置 Cordis 模块控制器服务（`"spinePlugins"`）。
pub struct SpinePlugins {
    ctx: Context,
    slots: Mutex<Vec<ModuleSlot>>,
    startup_errors: Vec<String>,
    op_lock: AsyncMutex<()>,
}

impl SpinePlugins {
    pub(crate) fn initialize(
        ctx: Context,
        specs: Vec<(SpineModuleMeta, ModuleSpec)>,
        raw_disabled: Vec<String>,
    ) -> Result<Arc<Self>, CordisError> {
        let normalized_disabled = normalize_disabled_plugins(raw_disabled);
        let available_ids: HashSet<&str> = specs.iter().map(|(m, _)| m.id.as_str()).collect();
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
            } else if !is_known_optional_id(&id) && !KNOWN_HOST_EXTENSION_IDS.contains(&id.as_str())
            {
                let msg = format!("未知内置模块 `{id}`，已忽略 disabledPlugins 中的配置");
                tracing::warn!(target: "mizar::spine-plugins", "{msg}");
                startup_errors.push(msg);
            }
        }

        let mut slots = Vec::with_capacity(specs.len());
        for (meta, spec) in specs {
            let fibers = if disabled_set.contains(&meta.id) {
                Vec::new()
            } else {
                mount_module_spawn(&ctx, &spec)?
            };
            slots.push(ModuleSlot { meta, spec, fibers });
        }

        Ok(Arc::new(Self {
            ctx,
            slots: Mutex::new(slots),
            startup_errors,
            op_lock: AsyncMutex::new(()),
        }))
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
        let slots = self.slots.lock().unwrap();
        out.extend(slots.iter().map(ModuleSlot::status));
        out
    }

    /// 列出当前运行时可用的可选模块状态（供 `/settings` `[模块]` 分组展示）。
    pub fn optional_modules(&self) -> Vec<SpineModuleStatus> {
        let slots = self.slots.lock().unwrap();
        slots.iter().map(ModuleSlot::status).collect()
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
        let slots = self.slots.lock().unwrap();
        slots
            .iter()
            .find(|s| s.meta.id == id)
            .map(ModuleSlot::status)
    }

    pub fn is_protected(&self, id: &str) -> bool {
        is_protected_module_id(id)
    }

    pub fn is_enabled(&self, id: &str) -> bool {
        if is_protected_module_id(id) {
            return true;
        }
        self.slots
            .lock()
            .unwrap()
            .iter()
            .find(|s| s.meta.id == id)
            .is_some_and(ModuleSlot::is_enabled)
    }

    /// 返回当前处于禁用状态的可选模块 ID（已排序去重，可直接回写 `"disabledPlugins"`）。
    pub fn disabled_ids(&self) -> Vec<String> {
        let slots = self.slots.lock().unwrap();
        normalize_disabled_plugins(
            slots
                .iter()
                .filter(|s| !s.is_enabled())
                .map(|s| s.meta.id.clone()),
        )
    }

    /// 热切换指定可选内置模块的启用状态。
    ///
    /// 返回 `Ok(true)` 表示实际发生了挂载/卸载切换；`Ok(false)` 表示状态未变。
    pub async fn set_enabled(&self, id: &str, enabled: bool) -> Result<bool, SpinePluginError> {
        if is_protected_module_id(id) {
            if !enabled {
                return Err(SpinePluginError::ProtectedModule(id.to_string()));
            }
            return Ok(false);
        }

        let _op = self.op_lock.lock().await;

        enum Action {
            Noop(Vec<Fiber>),
            Disable(Vec<Fiber>),
            Enable {
                old_fibers: Vec<Fiber>,
                spec: ModuleSpec,
            },
        }

        let action = {
            let mut slots = self.slots.lock().unwrap();
            let Some(slot) = slots.iter_mut().find(|s| s.meta.id == id) else {
                return Err(SpinePluginError::UnknownModule(id.to_string()));
            };
            let currently_enabled = slot.is_enabled();
            if currently_enabled == enabled {
                let stale = if !enabled {
                    std::mem::take(&mut slot.fibers)
                } else {
                    Vec::new()
                };
                Action::Noop(stale)
            } else if !enabled {
                // 卸载前同步热更新后的易变配置（如 compaction.*），避免重新启用时状态脑裂
                if id == "compaction"
                    && let Some(comp) = self.ctx.get::<CompactionRuntime>("compaction")
                {
                    slot.spec = ModuleSpec::Compaction(comp.config());
                }
                if id == "session-title"
                    && let Some(st) = self.ctx.get::<SessionTitleRuntime>("sessionTitle")
                {
                    slot.spec = ModuleSpec::SessionTitle(st.config());
                }
                if id == "subagent"
                    && let Some(sub) = self.ctx.get::<SubagentRuntime>("subagent")
                {
                    slot.spec = ModuleSpec::Subagent(sub.config());
                }
                Action::Disable(std::mem::take(&mut slot.fibers))
            } else {
                Action::Enable {
                    old_fibers: std::mem::take(&mut slot.fibers),
                    spec: slot.spec.clone(),
                }
            }
        };

        match action {
            Action::Noop(stale) => {
                for fiber in stale.into_iter().rev() {
                    fiber.dispose().await;
                }
                Ok(false)
            }
            Action::Disable(fibers) => {
                for fiber in fibers.into_iter().rev() {
                    fiber.dispose().await;
                }
                self.ctx.emit(
                    EVENT_SPINE_PLUGINS_CHANGE,
                    SpinePluginsChangeEvent {
                        id: id.to_string(),
                        enabled: false,
                    },
                );
                Ok(true)
            }
            Action::Enable { old_fibers, spec } => {
                for fiber in old_fibers.into_iter().rev() {
                    fiber.dispose().await;
                }
                let fibers = mount_module_and_wait(&self.ctx, id, &spec).await?;
                {
                    let mut slots = self.slots.lock().unwrap();
                    if let Some(slot) = slots.iter_mut().find(|s| s.meta.id == id) {
                        slot.fibers = fibers;
                    }
                }
                self.ctx.emit(
                    EVENT_SPINE_PLUGINS_CHANGE,
                    SpinePluginsChangeEvent {
                        id: id.to_string(),
                        enabled: true,
                    },
                );
                Ok(true)
            }
        }
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

        assert!(ctrl.set_enabled("tools-coding", false).await.unwrap());
        assert!(!ctrl.is_enabled("tools-coding"));
        assert!(!tools.list().iter().any(|t| t.name == "read"));
        assert!(!tools.list().iter().any(|t| t.name == "bash"));
        assert!(!prompt.build().contains("Available tools:"));

        assert!(ctrl.set_enabled("tools-coding", true).await.unwrap());
        assert!(tools.list().iter().any(|t| t.name == "read"));
        assert!(prompt.build().contains("Available tools:"));

        // 若外部直接释放某可选模块的底层 Fiber，is_enabled/disabled_ids 必须实时反映为禁用且允许干净重挂
        let slot_fiber = {
            let slots = ctrl.slots.lock().unwrap();
            slots
                .iter()
                .find(|s| s.meta.id == "tool-search")
                .unwrap()
                .fibers[0]
                .clone()
        };
        slot_fiber.dispose().await;
        assert!(!ctrl.is_enabled("tool-search"));
        assert!(ctrl.disabled_ids().contains(&"tool-search".to_string()));
        assert!(ctrl.set_enabled("tool-search", true).await.unwrap());
        assert!(ctrl.is_enabled("tool-search"));

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
