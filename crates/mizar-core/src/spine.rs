//! 套件插件（对应 `NamiDSH/packages/agent-spine` + `pi` Coding Agent 套件）。
//!
//! 把每个 Agent 都需要的核心服务一次挂上；
//! 「连哪个模型」和「入口（命令行 / TUI）」故意留在外面，由使用方决定。

pub mod config;
pub mod plugins;

use std::path::PathBuf;
use std::sync::Arc;

pub use crate::compaction::{
    CompactionConfig, CompactionContextExt, CompactionEndEvent, CompactionError, CompactionOutcome,
    CompactionProgressEvent, CompactionReport, CompactionRuntime, CompactionStartEvent,
    CompactionTrigger, ContextPressureReport, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS,
    EVENT_COMPACTION_START, ManualCompactOptions, strip_summary_wrapper,
};
pub use crate::instructions::{
    AGENT_INSTRUCTIONS_SOURCE, AgentInstructions, InstructionScopeKind, InstructionScopeStatus,
    InstructionsConfig, InstructionsContextExt, SessionInstructionsReport,
};
pub use crate::session_title::{
    DEFAULT_REFRESH_INTERVAL_TURNS, ParsedModelTitle, SERVICE_SESSION_TITLE, SessionTitleConfig,
    SessionTitleContextExt, SessionTitleError, SessionTitleRuntime, SessionTitleType,
    format_session_title, parse_model_ref, parse_model_title_output,
};
pub use crate::subagent::{
    DEFAULT_SUBAGENT_MAX_CONCURRENT, DEFAULT_SUBAGENT_MAX_STEPS, DEFAULT_SUBAGENT_MAX_STEPS_CAP,
    DEFAULT_SUBAGENT_MAX_WATCHDOG_ALERTS, DEFAULT_SUBAGENT_STALL_TIMEOUT_SECS,
    DEFAULT_SUBAGENT_TIMEOUT_SECS, EVENT_SUBAGENT_STATE, SERVICE_SUBAGENT, SUBAGENT_TOOL_NAME,
    SubagentConfig, SubagentContextExt, SubagentError, SubagentMode, SubagentPlugin,
    SubagentRunStatus, SubagentRuntime, SubagentStateEvent, SubagentTaskSnapshot,
};
pub use config::{
    CONFIG_DIR_NAME, CliLlmOverride, CodemodeSettings, CompactionSettings, EnvLlmOverride,
    EnvProviderKind, InstructionsSettings, LlmRouteError, LoadedMizarConfig, RecallSettings,
    ResolvedLlmRouter, SessionTitleSettings, SettingsConfig, SkillsSettings, ToolSearchSettings,
    UiSettings, load_mizar_config, load_mizar_config_isolated, load_mizar_config_with_global,
    normalize_disabled_plugins, normalize_session_title_thinking_level, patch_settings_file,
    persist_disabled_plugins, resolve_default_session_db_path, resolve_global_config_dir,
    resolve_global_settings_path, resolve_project_config_dir,
};
pub use mizar_session::recall::{RECALL_TOOL_NAME, RecallConfig, RecallPlugin};
pub use plugins::{
    KNOWN_HOST_EXTENSION_IDS, SERVICE_SPINE_PLUGINS, SpineModuleStatus, SpinePluginError,
    SpinePlugins, SpinePluginsContextExt, ToolsGuidePromptPlugin, is_protected_module_id,
};

use crate::agent_loop::{AgentLoop, AgentLoopConfig};
use crate::skills::{SkillRegistry, SkillRegistryConfig, SkillsConfig, SkillsPlugin};
use crate::system_prompt::{SystemPrompt, SystemPromptConfig};
use crate::trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, TrajectoryConfig, TrajectoryRuntime,
};
use cordis::{BoxError, BoxFuture, Cleanup, Context, Plugin, Service};
#[cfg(feature = "codemode")]
use mizar_codemode::{CodemodeConfig, CodemodePlugin};
use mizar_llm::LlmRuntime;
#[cfg(feature = "mcp")]
use mizar_mcp::{McpManager, McpPluginConfig};
use mizar_session::sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
use mizar_session::{SessionContextExt, SessionStore};
use mizar_tools::coding::{CodingToolsConfig, CodingToolsPlugin, CodingToolsPreset};
use mizar_tools::search::{ToolSearchConfig, ToolSearchPlugin};
use mizar_tools::{ToolRuntime, ToolsContextExt};

use plugins::SpineModuleMeta;

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AgentToolsMode {
    /// 不预装默认工具，仅提供底层核心服务。
    None,
    /// 完整编程套件：`read` + `bash` + `edit` + `write` + `grep` + `find` + `skill` + `tool_search` + (`codemode` / `mcp`)。
    Coding {
        coding: CodingToolsConfig,
        tool_search: ToolSearchConfig,
        #[cfg(feature = "codemode")]
        codemode: CodemodeConfig,
        #[cfg(feature = "mcp")]
        mcp: McpPluginConfig,
        skills: SkillsConfig,
        instructions: InstructionsConfig,
    },
}

#[derive(Debug, Clone)]
pub struct AgentSpineConfig {
    /// 文件工具允许访问的根目录。
    pub workspace: PathBuf,
    pub persona: Option<String>,
    pub agent: AgentLoopConfig,
    pub tools_mode: AgentToolsMode,
    /// 可选：全局工具白名单（支持 `*` 通配）。
    pub include_tools: Option<Vec<String>>,
    /// 可选：全局工具黑名单（支持 `*` 通配，例如 `mcp__fff__multi_grep`）。
    pub exclude_tools: Vec<String>,
    /// 是否启用轨迹采集与异常检测服务（默认开启）。
    pub trajectory: Option<TrajectoryConfig>,
    /// 是否启用重复工具调用守卫（默认开启）。
    pub repeat_guard: Option<RepeatGuardConfig>,
    /// 是否启用上下文压缩与压后工具守卫服务（默认开启）。
    pub compaction: Option<CompactionConfig>,
    /// 是否启用无损情景回溯工具（静态常驻 `recall`，默认开启）。
    pub recall: Option<RecallConfig>,
    /// 是否启用基于对话主题的会话标题自动生成服务（默认关闭，由交互式 TUI 按需开启）。
    pub session_title: Option<SessionTitleConfig>,
    /// 是否启用 UNIX 式单层 Subagent 调度工具（默认开启）。
    pub subagent: Option<SubagentConfig>,
    /// 可选：会话 SQLite 持久化数据库路径；为 `None` 时保持纯内存模式（适合测试）。
    pub session_db_path: Option<PathBuf>,
    /// 启动阶段屏蔽的内置可选模块 ID 列表（如 `mcp`、`codemode` 等）。
    pub disabled_plugins: Vec<String>,
}

impl AgentSpineConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            persona: None,
            agent: AgentLoopConfig::default(),
            tools_mode: AgentToolsMode::None,
            include_tools: None,
            exclude_tools: Vec::new(),
            trajectory: Some(TrajectoryConfig::default()),
            repeat_guard: Some(RepeatGuardConfig::default()),
            compaction: Some(CompactionConfig::default()),
            recall: Some(RecallConfig::default()),
            session_title: None,
            subagent: Some(SubagentConfig::default()),
            session_db_path: None,
            disabled_plugins: Vec::new(),
        }
    }

    pub fn coding(workspace: impl Into<PathBuf>) -> Self {
        let ws = workspace.into();
        Self {
            workspace: ws.clone(),
            persona: Some(crate::system_prompt::DEFAULT_PERSONA.to_string()),
            agent: AgentLoopConfig::default(),
            tools_mode: AgentToolsMode::Coding {
                coding: CodingToolsConfig::new(&ws),
                tool_search: ToolSearchConfig::default(),
                #[cfg(feature = "codemode")]
                codemode: CodemodeConfig::default(),
                #[cfg(feature = "mcp")]
                mcp: McpPluginConfig::new(&ws),
                skills: SkillsConfig::new(&ws),
                instructions: InstructionsConfig::new(&ws),
            },
            include_tools: None,
            exclude_tools: Vec::new(),
            trajectory: Some(TrajectoryConfig::default()),
            repeat_guard: Some(RepeatGuardConfig::default()),
            compaction: Some(CompactionConfig::default()),
            recall: Some(RecallConfig::default()),
            session_title: None,
            subagent: Some(SubagentConfig::default()),
            session_db_path: None,
            disabled_plugins: Vec::new(),
        }
    }

    pub fn with_session_db(mut self, path: impl Into<PathBuf>) -> Self {
        self.session_db_path = Some(path.into());
        self
    }

    pub fn with_disabled_plugins<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.disabled_plugins = ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_coding_preset(mut self, preset: CodingToolsPreset) -> Self {
        if let AgentToolsMode::Coding { coding, .. } = &mut self.tools_mode {
            coding.preset = preset;
        }
        self
    }
}

#[derive(Default, Clone, Copy)]
pub struct AgentSpinePlugin;

impl Plugin for AgentSpinePlugin {
    type Config = AgentSpineConfig;

    fn name(&self) -> Option<&str> {
        Some("agent-spine")
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let default_cwd = config.workspace.display().to_string();
            ctx.try_plugin(LlmRuntime::plugin(), ())?;
            if let Some(db_path) = config.session_db_path {
                ctx.try_plugin(
                    SqlitePersistencePlugin,
                    SqlitePersistenceConfig::new(db_path),
                )?;
            }
            ctx.try_plugin(SessionStore::plugin(), ())?;
            ctx.try_inject(["sessions"], move |c| {
                let cwd = default_cwd.clone();
                async move {
                    if let Ok(sessions) = c.sessions() {
                        sessions.set_default_cwd(Some(cwd));
                    }
                    Ok::<_, cordis::CordisError>(())
                }
            })?;
            ctx.try_plugin(ToolRuntime::plugin(), ())?;
            if config.include_tools.is_some() || !config.exclude_tools.is_empty() {
                let inc = config.include_tools;
                let exc = config.exclude_tools;
                ctx.try_inject(["tools"], move |c| {
                    let inc = inc.clone();
                    let exc = exc.clone();
                    async move {
                        if let Ok(rt) = c.tools() {
                            rt.set_tool_selection(inc, exc);
                        }
                        Ok::<_, cordis::CordisError>(())
                    }
                })?;
            }
            ctx.try_plugin(
                SystemPrompt::plugin(),
                SystemPromptConfig {
                    persona: config.persona,
                    runtime_protocol: Some(
                        crate::system_prompt::DEFAULT_RUNTIME_PROTOCOL.to_string(),
                    ),
                },
            )?;
            ctx.try_plugin(AgentLoop::plugin(), config.agent)?;

            let mut metas: Vec<SpineModuleMeta> = Vec::new();
            let push_meta =
                |metas: &mut Vec<SpineModuleMeta>, id: &str, label: &str, desc: &str| {
                    metas.push(SpineModuleMeta {
                        id: id.to_string(),
                        label: label.to_string(),
                        description: desc.to_string(),
                    });
                };

            if config.compaction.is_some() {
                push_meta(
                    &mut metas,
                    "compaction",
                    "上下文压缩",
                    "上下文压缩与压后写文件守卫",
                );
            }
            if matches!(config.tools_mode, AgentToolsMode::Coding { .. }) {
                push_meta(
                    &mut metas,
                    "tools-coding",
                    "基础编程工具",
                    "提供 read / bash / edit / write / grep / find 编程工具",
                );
                push_meta(
                    &mut metas,
                    "tool-search",
                    "延迟工具检索",
                    "按需检索与激活延迟工具及 MCP 工具 (tool_search)",
                );
                #[cfg(feature = "codemode")]
                push_meta(
                    &mut metas,
                    "codemode",
                    "脚本执行沙箱",
                    "通过 QuickJS 沙箱批量编排与调用工具 (codemode)",
                );
                #[cfg(feature = "mcp")]
                push_meta(
                    &mut metas,
                    "mcp",
                    "MCP 客户端",
                    "连接外部 MCP 服务器并暴露其工具与提示词",
                );
                push_meta(
                    &mut metas,
                    "skills",
                    "技能加载器",
                    "发现并按需激活全局与项目级 Skill 扩展",
                );
                push_meta(
                    &mut metas,
                    "instructions",
                    "项目指令",
                    "自动发现并注入 AGENTS.md / CLAUDE.md 指令规范",
                );
            }
            if config.recall.is_some() {
                push_meta(
                    &mut metas,
                    "recall",
                    "情景记忆回溯",
                    "静态常驻的历史会话检索与回溯工具 (recall)",
                );
            }
            if config.session_title.is_some() {
                push_meta(
                    &mut metas,
                    "session-title",
                    "会话标题",
                    "后台基于用户输入生成与维护会话标题",
                );
            }
            if config.subagent.is_some() {
                push_meta(
                    &mut metas,
                    "subagent",
                    "单层子代理",
                    "在独立子会话中执行只读并行调研或受控编辑子任务 (subagent)",
                );
            }
            if config.trajectory.is_some() {
                push_meta(
                    &mut metas,
                    "trajectory",
                    "轨迹采集",
                    "记录模型/工具执行轨迹并生成异常与状态投影",
                );
            }
            if config.repeat_guard.is_some() {
                push_meta(
                    &mut metas,
                    "repeat-guard",
                    "重复调用守卫",
                    "拦截连续重复的失败工具调用，防止死循环",
                );
            }

            let controller = SpinePlugins::initialize(ctx.clone(), metas, config.disabled_plugins);
            ctx.provide(SERVICE_SPINE_PLUGINS, Arc::clone(&controller))?;

            // 在 `instructions` / `skills` / `recall` 之前装配 `compaction`：
            // 使同一次 `agent/pre-step` 中先完成压缩推进 `compacted_up_to_seq`，
            // 随后指令、技能与回溯工具立即可见新压缩边界并当步完成重注/激活。
            if let Some(compaction_cfg) = config.compaction {
                ctx.try_plugin(CompactionRuntime::plugin(), compaction_cfg)?;
            }

            match config.tools_mode {
                AgentToolsMode::None => {
                    let _ = config.workspace;
                }
                AgentToolsMode::Coding {
                    coding,
                    tool_search,
                    #[cfg(feature = "codemode")]
                    codemode,
                    #[cfg(feature = "mcp")]
                    mcp,
                    skills,
                    instructions,
                } => {
                    let inject_prompt = coding.inject_system_prompt;
                    ctx.try_plugin(CodingToolsPlugin, coding)?;
                    if inject_prompt {
                        ctx.try_plugin(ToolsGuidePromptPlugin, ())?;
                    }
                    ctx.try_plugin(ToolSearchPlugin, tool_search)?;
                    #[cfg(feature = "codemode")]
                    ctx.try_plugin(CodemodePlugin, codemode)?;
                    #[cfg(feature = "mcp")]
                    if controller.is_enabled("mcp") {
                        ctx.try_plugin(McpManager::plugin(), mcp)?;
                    } else {
                        controller.defer_mcp_mount(mcp);
                    }
                    ctx.try_plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())?;
                    ctx.try_plugin(SkillsPlugin, skills)?;
                    ctx.try_plugin(AgentInstructions::plugin(), instructions)?;
                }
            }

            if let Some(recall_cfg) = config.recall {
                ctx.try_plugin(RecallPlugin, recall_cfg)?;
            }
            if let Some(st_cfg) = config.session_title {
                ctx.try_plugin(SessionTitleRuntime::plugin(), st_cfg)?;
            }
            if let Some(sub_cfg) = config.subagent {
                ctx.try_plugin(SubagentRuntime::plugin(), sub_cfg)?;
            }
            if let Some(traj_cfg) = config.trajectory {
                ctx.try_plugin(TrajectoryRuntime::plugin(), traj_cfg)?;
            }
            if let Some(mut guard_cfg) = config.repeat_guard {
                guard_cfg.enabled_flag = controller.flag_for("repeat-guard");
                ctx.try_plugin(RepeatToolGuardPlugin, guard_cfg)?;
            }

            controller.attach_initial_gates()?;
            Ok(None)
        })
    }
}
