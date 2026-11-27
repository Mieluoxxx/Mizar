//! 套件插件（对应 `NamiDSH/packages/agent-spine` + `pi` Coding Agent 套件）。
//!
//! 把每个 Agent 都需要的核心服务一次挂上；
//! 「连哪个模型」和「入口（命令行 / TUI）」故意留在外面，由使用方决定。

pub mod config;
pub mod plugins;

use std::path::PathBuf;

pub use config::{
    CONFIG_DIR_NAME, CliLlmOverride, CodemodeSettings, CompactionSettings, EnvLlmOverride,
    EnvProviderKind, InstructionsSettings, LoadedMizarConfig, RecallSettings, SessionTitleSettings,
    SettingsConfig, SkillsSettings, ToolSearchSettings, UiSettings, load_mizar_config,
    load_mizar_config_isolated, load_mizar_config_with_global, normalize_disabled_plugins,
    normalize_session_title_thinking_level, patch_settings_file, persist_disabled_plugins,
    resolve_default_session_db_path, resolve_global_config_dir, resolve_global_settings_path,
    resolve_project_config_dir,
};
#[cfg(feature = "router")]
pub use config::{LlmRouteError, ResolvedLlmRouter};
#[cfg(feature = "compaction")]
pub use mizar_compaction::{
    CompactionConfig, CompactionContextExt, CompactionEndEvent, CompactionError, CompactionOutcome,
    CompactionProgressEvent, CompactionReport, CompactionRuntime, CompactionStartEvent,
    CompactionTrigger, ContextPressureReport, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS,
    EVENT_COMPACTION_START, ManualCompactOptions, strip_summary_wrapper,
};
#[cfg(feature = "coding")]
pub use mizar_instructions::{
    AGENT_INSTRUCTIONS_SECTION_ID, AgentInstructions, InstructionScopeKind, InstructionScopeStatus,
    InstructionsConfig, InstructionsContextExt, SessionInstructionsReport,
};
#[cfg(feature = "compaction")]
pub use mizar_recall::{RECALL_TOOL_NAME, RecallConfig, RecallPlugin};
#[cfg(feature = "session-title")]
pub use mizar_session_title::{
    DEFAULT_REFRESH_INTERVAL_TURNS, ParsedModelTitle, SERVICE_SESSION_TITLE, SessionTitleConfig,
    SessionTitleContextExt, SessionTitleError, SessionTitleRuntime, SessionTitleType,
    format_session_title, parse_model_ref, parse_model_title_output,
};
pub use plugins::{
    EVENT_SPINE_PLUGINS_CHANGE, KNOWN_HOST_EXTENSION_IDS, SERVICE_SPINE_PLUGINS, SpineModuleStatus,
    SpinePluginError, SpinePlugins, SpinePluginsChangeEvent, SpinePluginsContextExt,
    is_protected_module_id,
};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Plugin, Service};
use mizar_agent_loop::{AgentLoop, AgentLoopConfig};
#[cfg(feature = "coding")]
use mizar_codemode::CodemodeConfig;
use mizar_llm::LlmRuntime;
#[cfg(feature = "coding")]
use mizar_mcp::McpPluginConfig;
use mizar_session::{SessionContextExt, SessionStore};
#[cfg(feature = "sqlite")]
use mizar_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
#[cfg(feature = "coding")]
use mizar_skills::SkillsConfig;
use mizar_system_prompt::{DEFAULT_RUNTIME_PROTOCOL, SystemPrompt, SystemPromptConfig};
#[cfg(feature = "legacy")]
use mizar_tool_fs::ToolFsConfig;
#[cfg(feature = "coding")]
use mizar_tool_search::ToolSearchConfig;
use mizar_tools::{ToolRuntime, ToolsContextExt};
#[cfg(feature = "coding")]
use mizar_tools_coding::{CodingToolsConfig, CodingToolsPreset};
#[cfg(feature = "trajectory")]
use mizar_trajectory::{RepeatGuardConfig, TrajectoryConfig};

use plugins::{ModuleSpec, SpineModuleMeta};

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AgentToolsMode {
    /// 不预装默认工具，仅提供底层核心服务。
    None,
    /// 保留原有最小只读套件（`get_time` + `list_dir` + `read_file`）。
    #[cfg(feature = "legacy")]
    Legacy,
    #[cfg(feature = "coding")]
    Coding {
        coding: CodingToolsConfig,
        tool_search: ToolSearchConfig,
        codemode: CodemodeConfig,
        mcp: McpPluginConfig,
        skills: SkillsConfig,
        instructions: InstructionsConfig,
    },
}

#[derive(Debug, Clone)]
pub struct AgentSpineConfig {
    /// 文件工具允许访问的根目录。
    pub workspace: PathBuf,
    pub personality: Option<String>,
    pub agent: AgentLoopConfig,
    pub tools_mode: AgentToolsMode,
    /// 可选：全局工具白名单（支持 `*` 通配）。
    pub include_tools: Option<Vec<String>>,
    /// 可选：全局工具黑名单（支持 `*` 通配，例如 `mcp__fff__multi_grep`）。
    pub exclude_tools: Vec<String>,
    /// 是否启用轨迹采集与异常检测服务（默认开启）。
    #[cfg(feature = "trajectory")]
    pub trajectory: Option<TrajectoryConfig>,
    /// 是否启用重复工具调用守卫（默认开启）。
    #[cfg(feature = "trajectory")]
    pub repeat_guard: Option<RepeatGuardConfig>,
    /// 是否启用上下文压缩、缓存感知尾部剪枝与压后写文件守卫服务（默认开启）。
    #[cfg(feature = "compaction")]
    pub compaction: Option<CompactionConfig>,
    /// 是否启用单一静态情景回溯工具（`recall`，默认开启）。
    #[cfg(feature = "compaction")]
    pub recall: Option<RecallConfig>,
    /// 是否启用基于对话主题的会话标题自动生成服务（默认开启）。
    #[cfg(feature = "session-title")]
    pub session_title: Option<SessionTitleConfig>,
    /// 可选：会话 SQLite 持久化数据库路径；为 `None` 时保持纯内存模式（适合测试）。
    pub session_db_path: Option<PathBuf>,
    /// 启动阶段屏蔽的内置可选模块 ID 列表（如 `mcp`、`skills` 等）。
    pub disabled_plugins: Vec<String>,
}

impl AgentSpineConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        #[cfg(feature = "legacy")]
        let default_mode = AgentToolsMode::Legacy;
        #[cfg(not(feature = "legacy"))]
        let default_mode = AgentToolsMode::None;

        Self {
            workspace: workspace.into(),
            personality: None,
            agent: AgentLoopConfig::default(),
            tools_mode: default_mode,
            include_tools: None,
            exclude_tools: Vec::new(),
            #[cfg(feature = "trajectory")]
            trajectory: Some(TrajectoryConfig::default()),
            #[cfg(feature = "trajectory")]
            repeat_guard: Some(RepeatGuardConfig::default()),
            #[cfg(feature = "compaction")]
            compaction: Some(CompactionConfig::default()),
            #[cfg(feature = "compaction")]
            recall: Some(RecallConfig::default()),
            #[cfg(feature = "session-title")]
            session_title: Some(SessionTitleConfig::default()),
            session_db_path: None,
            disabled_plugins: Vec::new(),
        }
    }

    #[cfg(feature = "coding")]
    pub fn coding(workspace: impl Into<PathBuf>) -> Self {
        let ws = workspace.into();
        Self {
            workspace: ws.clone(),
            personality: None,
            agent: AgentLoopConfig::default(),
            tools_mode: AgentToolsMode::Coding {
                coding: CodingToolsConfig::new(&ws),
                tool_search: ToolSearchConfig::default(),
                codemode: CodemodeConfig::default(),
                mcp: McpPluginConfig::new(&ws),
                skills: SkillsConfig::new(&ws),
                instructions: InstructionsConfig::new(&ws),
            },
            include_tools: None,
            exclude_tools: Vec::new(),
            #[cfg(feature = "trajectory")]
            trajectory: Some(TrajectoryConfig::default()),
            #[cfg(feature = "trajectory")]
            repeat_guard: Some(RepeatGuardConfig::default()),
            #[cfg(feature = "compaction")]
            compaction: Some(CompactionConfig::default()),
            #[cfg(feature = "compaction")]
            recall: Some(RecallConfig::default()),
            #[cfg(feature = "session-title")]
            session_title: Some(SessionTitleConfig::default()),
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

    #[cfg(feature = "coding")]
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
            #[cfg(feature = "sqlite")]
            if let Some(db_path) = config.session_db_path {
                ctx.try_plugin(
                    SqlitePersistencePlugin,
                    SqlitePersistenceConfig::new(db_path),
                )?;
            }
            #[cfg(not(feature = "sqlite"))]
            let _ = config.session_db_path;
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
                    identity: None,
                    engineering_principles: None,
                    personality: config.personality,
                    runtime_protocol: Some(DEFAULT_RUNTIME_PROTOCOL.to_string()),
                },
            )?;
            ctx.try_plugin(AgentLoop::plugin(), config.agent)?;

            #[cfg_attr(
                not(any(
                    feature = "legacy",
                    feature = "coding",
                    feature = "compaction",
                    feature = "session-title",
                    feature = "trajectory"
                )),
                allow(unused_mut)
            )]
            let mut specs: Vec<(SpineModuleMeta, ModuleSpec)> = Vec::new();

            // 在 `instructions` / `skills` / `recall` 之前装配 `compaction`：
            // 使同一次 `agent/pre-step` 中先完成压缩推进 `compacted_up_to_seq`，
            // 随后指令、技能与回溯工具立即可见新压缩边界并当步完成重注/激活。
            #[cfg(feature = "compaction")]
            if let Some(compaction_cfg) = config.compaction {
                specs.push((
                    SpineModuleMeta {
                        id: "compaction".to_string(),
                        label: "上下文压缩".to_string(),
                        description: "缓存感知尾部剪枝、结构化检查点摘要与压后写文件守卫"
                            .to_string(),
                    },
                    ModuleSpec::Compaction(compaction_cfg),
                ));
            }

            match config.tools_mode {
                AgentToolsMode::None => {
                    let _ = config.workspace;
                }
                #[cfg(feature = "legacy")]
                AgentToolsMode::Legacy => {
                    specs.push((
                        SpineModuleMeta {
                            id: "tool-time".to_string(),
                            label: "时间工具".to_string(),
                            description: "提供只读当前时间查询工具 (get_time)".to_string(),
                        },
                        ModuleSpec::ToolTime,
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "tool-fs".to_string(),
                            label: "只读文件工具".to_string(),
                            description: "提供工作区只读列表与读文件工具 (list_dir / read_file)"
                                .to_string(),
                        },
                        ModuleSpec::ToolFs(ToolFsConfig::new(config.workspace)),
                    ));
                }
                #[cfg(feature = "coding")]
                AgentToolsMode::Coding {
                    coding,
                    tool_search,
                    codemode,
                    mcp,
                    skills,
                    instructions,
                } => {
                    specs.push((
                        SpineModuleMeta {
                            id: "tools-coding".to_string(),
                            label: "基础编程工具".to_string(),
                            description: "提供 read / bash / edit / write / grep / find 编程工具"
                                .to_string(),
                        },
                        ModuleSpec::ToolsCoding(coding),
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "tool-search".to_string(),
                            label: "延迟工具检索".to_string(),
                            description: "按需检索与激活延迟工具及 MCP 工具 (tool_search)"
                                .to_string(),
                        },
                        ModuleSpec::ToolSearch(tool_search),
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "codemode".to_string(),
                            label: "脚本执行沙箱".to_string(),
                            description: "按需通过 QuickJS 沙箱编排工具调用".to_string(),
                        },
                        ModuleSpec::Codemode(codemode),
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "mcp".to_string(),
                            label: "MCP 客户端".to_string(),
                            description: "连接外部 MCP 服务器并暴露其工具与提示词".to_string(),
                        },
                        ModuleSpec::Mcp(mcp),
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "skills".to_string(),
                            label: "技能加载器".to_string(),
                            description: "发现并按需激活全局与项目级 Skill 扩展".to_string(),
                        },
                        ModuleSpec::Skills(skills),
                    ));
                    specs.push((
                        SpineModuleMeta {
                            id: "instructions".to_string(),
                            label: "项目指令".to_string(),
                            description: "自动发现并注入 AGENTS.md / CLAUDE.md 指令规范"
                                .to_string(),
                        },
                        ModuleSpec::Instructions(instructions),
                    ));
                }
            }

            #[cfg(feature = "compaction")]
            if let Some(recall_cfg) = config.recall {
                specs.push((
                    SpineModuleMeta {
                        id: "recall".to_string(),
                        label: "情景记忆回溯".to_string(),
                        description: "静态常驻的历史会话搜索与原文读取工具 (recall)".to_string(),
                    },
                    ModuleSpec::Recall(recall_cfg),
                ));
            }

            #[cfg(feature = "session-title")]
            if let Some(st_cfg) = config.session_title {
                specs.push((
                    SpineModuleMeta {
                        id: "session-title".to_string(),
                        label: "会话标题".to_string(),
                        description: "后台基于用户输入生成与维护会话标题".to_string(),
                    },
                    ModuleSpec::SessionTitle(st_cfg),
                ));
            }

            #[cfg(feature = "trajectory")]
            {
                if let Some(traj_cfg) = config.trajectory {
                    specs.push((
                        SpineModuleMeta {
                            id: "trajectory".to_string(),
                            label: "轨迹采集".to_string(),
                            description: "记录模型/工具执行轨迹并生成异常与状态投影".to_string(),
                        },
                        ModuleSpec::Trajectory(traj_cfg),
                    ));
                }
                if let Some(guard_cfg) = config.repeat_guard {
                    specs.push((
                        SpineModuleMeta {
                            id: "repeat-guard".to_string(),
                            label: "重复调用守卫".to_string(),
                            description: "拦截连续重复的失败工具调用，防止死循环".to_string(),
                        },
                        ModuleSpec::RepeatGuard(guard_cfg),
                    ));
                }
            }

            let controller = SpinePlugins::initialize(ctx.clone(), specs, config.disabled_plugins)?;
            ctx.provide(SERVICE_SPINE_PLUGINS, controller)?;
            Ok(None)
        })
    }
}
