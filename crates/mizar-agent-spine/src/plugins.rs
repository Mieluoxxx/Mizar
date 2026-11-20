//! 内置 Cordis 模块控制器（`"spinePlugins"`）：
//!
//! 统一管理 `mizar-agent-spine` 装配的可选内置 Cordis 模块生命周期（启动屏蔽与运行期热挂载/热卸载），
//! 并对核心内核服务（`llm`、`sessions`、`tools`、`systemPrompt`、`agentLoop`、`spinePlugins`）实施 Gate 2 自保护。

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[cfg(any(
    feature = "coding",
    feature = "compaction",
    feature = "session-title",
    feature = "trajectory"
))]
use cordis::Service;
use cordis::{Context, CordisError, Fiber};
use thiserror::Error;
use tokio::sync::Mutex as AsyncMutex;

#[cfg(feature = "coding")]
use mizar_codemode::{CodemodeConfig, CodemodePlugin};
#[cfg(feature = "compaction")]
use mizar_compaction::{CompactionConfig, CompactionRuntime};
#[cfg(feature = "coding")]
use mizar_instructions::{AgentInstructions, InstructionsConfig};
#[cfg(feature = "coding")]
use mizar_mcp::{McpManager, McpPluginConfig};
#[cfg(feature = "compaction")]
use mizar_recall::{RecallConfig, RecallPlugin};
#[cfg(feature = "session-title")]
use mizar_session_title::{SessionTitleConfig, SessionTitleRuntime};
#[cfg(feature = "coding")]
use mizar_skills::{SkillRegistry, SkillRegistryConfig, SkillsConfig, SkillsPlugin};
#[cfg(feature = "legacy")]
use mizar_tool_fs::{ToolFsConfig, ToolFsPlugin};
#[cfg(feature = "coding")]
use mizar_tool_search::{ToolSearchConfig, ToolSearchPlugin};
#[cfg(feature = "legacy")]
use mizar_tool_time::ToolTimePlugin;
#[cfg(feature = "coding")]
use mizar_tools_coding::{CodingToolsConfig, CodingToolsPlugin};
#[cfg(feature = "trajectory")]
use mizar_trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, TrajectoryConfig, TrajectoryRuntime,
};

use crate::config::normalize_disabled_plugins;

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

/// 所有已知的内置可选模块稳定 ID（含不同模式或 Cargo feature 下可能未激活的模块）。
const KNOWN_OPTIONAL_IDS: &[&str] = &[
    "tool-time",
    "tool-fs",
    "tools-coding",
    "tool-search",
    "codemode",
    "mcp",
    "skills",
    "instructions",
    "compaction",
    "recall",
    "session-title",
    "trajectory",
    "repeat-guard",
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

#[derive(Clone)]
pub(crate) enum ModuleSpec {
    #[cfg(feature = "legacy")]
    ToolTime,
    #[cfg(feature = "legacy")]
    ToolFs(ToolFsConfig),
    #[cfg(feature = "coding")]
    ToolsCoding(CodingToolsConfig),
    #[cfg(feature = "coding")]
    ToolSearch(ToolSearchConfig),
    #[cfg(feature = "coding")]
    Codemode(CodemodeConfig),
    #[cfg(feature = "coding")]
    Mcp(McpPluginConfig),
    #[cfg(feature = "coding")]
    Skills(SkillsConfig),
    #[cfg(feature = "coding")]
    Instructions(InstructionsConfig),
    #[cfg(feature = "compaction")]
    Compaction(CompactionConfig),
    #[cfg(feature = "compaction")]
    Recall(RecallConfig),
    #[cfg(feature = "session-title")]
    SessionTitle(SessionTitleConfig),
    #[cfg(feature = "trajectory")]
    Trajectory(TrajectoryConfig),
    #[cfg(feature = "trajectory")]
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
    let _ = ctx;
    match *spec {
        #[cfg(feature = "legacy")]
        ModuleSpec::ToolTime => Ok(vec![ctx.try_plugin(ToolTimePlugin, ())?]),
        #[cfg(feature = "legacy")]
        ModuleSpec::ToolFs(ref cfg) => Ok(vec![ctx.try_plugin(ToolFsPlugin, cfg.clone())?]),
        #[cfg(feature = "coding")]
        ModuleSpec::ToolsCoding(ref cfg) => {
            Ok(vec![ctx.try_plugin(CodingToolsPlugin, cfg.clone())?])
        }
        #[cfg(feature = "coding")]
        ModuleSpec::ToolSearch(ref cfg) => Ok(vec![ctx.try_plugin(ToolSearchPlugin, cfg.clone())?]),
        #[cfg(feature = "coding")]
        ModuleSpec::Codemode(ref cfg) => Ok(vec![ctx.try_plugin(CodemodePlugin, cfg.clone())?]),
        #[cfg(feature = "coding")]
        ModuleSpec::Mcp(ref cfg) => Ok(vec![ctx.try_plugin(McpManager::plugin(), cfg.clone())?]),
        #[cfg(feature = "coding")]
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
        #[cfg(feature = "coding")]
        ModuleSpec::Instructions(ref cfg) => Ok(vec![
            ctx.try_plugin(AgentInstructions::plugin(), cfg.clone())?,
        ]),
        #[cfg(feature = "compaction")]
        ModuleSpec::Compaction(ref cfg) => Ok(vec![
            ctx.try_plugin(CompactionRuntime::plugin(), cfg.clone())?,
        ]),
        #[cfg(feature = "compaction")]
        ModuleSpec::Recall(ref cfg) => Ok(vec![ctx.try_plugin(RecallPlugin, cfg.clone())?]),
        #[cfg(feature = "session-title")]
        ModuleSpec::SessionTitle(ref cfg) => Ok(vec![
            ctx.try_plugin(SessionTitleRuntime::plugin(), cfg.clone())?,
        ]),
        #[cfg(feature = "trajectory")]
        ModuleSpec::Trajectory(ref cfg) => Ok(vec![
            ctx.try_plugin(TrajectoryRuntime::plugin(), cfg.clone())?,
        ]),
        #[cfg(feature = "trajectory")]
        ModuleSpec::RepeatGuard(ref cfg) => {
            Ok(vec![ctx.try_plugin(RepeatToolGuardPlugin, cfg.clone())?])
        }
    }
}

#[cfg_attr(
    not(any(
        feature = "legacy",
        feature = "coding",
        feature = "compaction",
        feature = "session-title",
        feature = "trajectory"
    )),
    allow(dead_code)
)]
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
    let _ = (ctx, id);
    match *spec {
        #[cfg(feature = "legacy")]
        ModuleSpec::ToolTime => mount_single_and_wait(ctx, id, ToolTimePlugin, ()).await,
        #[cfg(feature = "legacy")]
        ModuleSpec::ToolFs(ref cfg) => {
            mount_single_and_wait(ctx, id, ToolFsPlugin, cfg.clone()).await
        }
        #[cfg(feature = "coding")]
        ModuleSpec::ToolsCoding(ref cfg) => {
            mount_single_and_wait(ctx, id, CodingToolsPlugin, cfg.clone()).await
        }
        #[cfg(feature = "coding")]
        ModuleSpec::ToolSearch(ref cfg) => {
            mount_single_and_wait(ctx, id, ToolSearchPlugin, cfg.clone()).await
        }
        #[cfg(feature = "coding")]
        ModuleSpec::Codemode(ref cfg) => {
            mount_single_and_wait(ctx, id, CodemodePlugin, cfg.clone()).await
        }
        #[cfg(feature = "coding")]
        ModuleSpec::Mcp(ref cfg) => {
            mount_single_and_wait(ctx, id, McpManager::plugin(), cfg.clone()).await
        }
        #[cfg(feature = "coding")]
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
        #[cfg(feature = "coding")]
        ModuleSpec::Instructions(ref cfg) => {
            mount_single_and_wait(ctx, id, AgentInstructions::plugin(), cfg.clone()).await
        }
        #[cfg(feature = "compaction")]
        ModuleSpec::Compaction(ref cfg) => {
            mount_single_and_wait(ctx, id, CompactionRuntime::plugin(), cfg.clone()).await
        }
        #[cfg(feature = "compaction")]
        ModuleSpec::Recall(ref cfg) => {
            mount_single_and_wait(ctx, id, RecallPlugin, cfg.clone()).await
        }
        #[cfg(feature = "session-title")]
        ModuleSpec::SessionTitle(ref cfg) => {
            mount_single_and_wait(ctx, id, SessionTitleRuntime::plugin(), cfg.clone()).await
        }
        #[cfg(feature = "trajectory")]
        ModuleSpec::Trajectory(ref cfg) => {
            mount_single_and_wait(ctx, id, TrajectoryRuntime::plugin(), cfg.clone()).await
        }
        #[cfg(feature = "trajectory")]
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
        let logger = ctx.logger_named("spine-plugins");

        for id in normalized_disabled {
            if is_protected_module_id(&id) {
                let msg =
                    format!("核心受保护模块 `{id}` 禁止禁用，已忽略 disabledPlugins 中的屏蔽项");
                logger.warn(msg.clone());
                startup_errors.push(msg);
            } else if available_ids.contains(id.as_str()) {
                disabled_set.insert(id);
            } else if !is_known_optional_id(&id) {
                let msg = format!("未知内置模块 `{id}`，已忽略 disabledPlugins 中的配置");
                logger.warn(msg.clone());
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
                #[cfg(feature = "compaction")]
                if id == "compaction"
                    && let Some(comp) = self.ctx.get::<CompactionRuntime>("compaction")
                {
                    slot.spec = ModuleSpec::Compaction(comp.config());
                }
                #[cfg(feature = "session-title")]
                if id == "session-title"
                    && let Some(st) = self.ctx.get::<SessionTitleRuntime>("sessionTitle")
                {
                    slot.spec = ModuleSpec::SessionTitle(st.config());
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
    use crate::{AgentSpineConfig, AgentSpinePlugin};
    use cordis::Context;
    use mizar_agent_loop::AgentLoopContextExt;
    use mizar_llm::LlmContextExt;
    use mizar_session::SessionContextExt;
    use mizar_system_prompt::SystemPromptContextExt;
    use mizar_tools::ToolsContextExt;

    #[tokio::test]
    async fn core_modules_are_protected_under_minimal_features() {
        let app = Context::new();
        app.logger().set_silent();

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

    #[cfg(feature = "legacy")]
    #[tokio::test]
    async fn legacy_mode_toggles_tool_time_and_tool_fs() {
        let app = Context::new();
        app.logger().set_silent();

        let cfg = AgentSpineConfig::new(std::env::temp_dir()).with_disabled_plugins(["tool-time"]);
        app.plugin(AgentSpinePlugin, cfg).await.unwrap();
        for rt in app.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }

        let ctrl = app.spine_plugins().unwrap();
        let tools = app.tools().unwrap();
        assert!(!ctrl.is_enabled("tool-time"));
        assert!(ctrl.is_enabled("tool-fs"));
        assert!(!tools.list().iter().any(|t| t.name == "get_time"));
        assert!(tools.list().iter().any(|t| t.name == "read_file"));

        assert!(ctrl.set_enabled("tool-time", true).await.unwrap());
        assert!(ctrl.is_enabled("tool-time"));
        assert!(tools.list().iter().any(|t| t.name == "get_time"));

        assert!(ctrl.set_enabled("tool-fs", false).await.unwrap());
        assert!(!ctrl.is_enabled("tool-fs"));
        assert!(!tools.list().iter().any(|t| t.name == "read_file"));
        assert!(!tools.list().iter().any(|t| t.name == "list_dir"));
        assert_eq!(ctrl.disabled_ids(), vec!["tool-fs".to_string()]);

        assert!(ctrl.set_enabled("tool-fs", true).await.unwrap());
        assert!(tools.list().iter().any(|t| t.name == "read_file"));
        assert!(ctrl.disabled_ids().is_empty());

        // 若外部直接释放某可选模块的底层 Fiber，is_enabled/disabled_ids 必须实时反映为禁用且允许干净重挂
        let slot_fiber = {
            let slots = ctrl.slots.lock().unwrap();
            slots
                .iter()
                .find(|s| s.meta.id == "tool-time")
                .unwrap()
                .fibers[0]
                .clone()
        };
        slot_fiber.dispose().await;
        assert!(!ctrl.is_enabled("tool-time"));
        assert_eq!(ctrl.disabled_ids(), vec!["tool-time".to_string()]);
        assert!(ctrl.set_enabled("tool-time", true).await.unwrap());
        assert!(ctrl.is_enabled("tool-time"));

        app.fiber().dispose().await;
    }
}
