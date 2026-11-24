//! 套件组合根（对应 `NamiDSH/packages/agent-spine` + `pi` Coding Agent 套件）。
//!
//! 通过 `MizarRuntime::build` 在编译期按确定 DAG 顺序静态装配全部核心服务与可选模块；
//! 「连哪个模型」与「宿主入口（CLI / TUI / 桌面端）」留在外部按需挂载。

pub mod config;
pub mod plugins;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub use crate::compaction::{
    CompactionConfig, CompactionEndEvent, CompactionError, CompactionOutcome,
    CompactionProgressEvent, CompactionReport, CompactionRuntime, CompactionStartEvent,
    CompactionTrigger, ContextPressureReport, ManualCompactOptions, strip_summary_wrapper,
};
pub use crate::instructions::{
    AGENT_INSTRUCTIONS_SOURCE, AgentInstructions, InstructionScopeKind, InstructionScopeStatus,
    InstructionsConfig, SessionInstructionsReport,
};
pub use crate::session_title::{
    DEFAULT_REFRESH_INTERVAL_TURNS, ParsedModelTitle, SessionTitleConfig, SessionTitleError,
    SessionTitleRuntime, SessionTitleType, format_session_title, parse_model_ref,
    parse_model_title_output,
};
pub use crate::subagent::{
    DEFAULT_SUBAGENT_MAX_CONCURRENT, DEFAULT_SUBAGENT_MAX_STEPS, DEFAULT_SUBAGENT_MAX_STEPS_CAP,
    DEFAULT_SUBAGENT_MAX_WATCHDOG_ALERTS, DEFAULT_SUBAGENT_STALL_TIMEOUT_SECS,
    DEFAULT_SUBAGENT_TIMEOUT_SECS, SUBAGENT_TOOL_NAME, SubagentConfig, SubagentError, SubagentMode,
    SubagentRunStatus, SubagentRuntime, SubagentStateEvent, SubagentTaskSnapshot,
};
pub use config::{
    CONFIG_DIR_NAME, CliLlmOverride, CodemodeSettings, CompactionSettings, EnvLlmOverride,
    EnvProviderKind, InstructionsSettings, LlmRouteError, LoadedMizarConfig, ResolvedLlmRouter,
    SessionTitleSettings, SettingsConfig, SkillsSettings, ToolSearchSettings, UiSettings,
    load_mizar_config, load_mizar_config_isolated, load_mizar_config_with_global,
    normalize_disabled_plugins, normalize_session_title_thinking_level, patch_settings_file,
    persist_disabled_plugins, resolve_default_session_db_path, resolve_global_config_dir,
    resolve_global_settings_path, resolve_project_config_dir,
};
pub use plugins::{
    KNOWN_HOST_EXTENSION_IDS, SpineModuleStatus, SpinePluginError, SpinePlugins,
    is_protected_module_id,
};

use crate::agent_loop::{AgentLoop, AgentLoopConfig};
use crate::skills::{SkillRegistry, SkillRegistryConfig, SkillsConfig, install_skills};
use crate::system_prompt::{PromptSection, SystemPrompt, SystemPromptConfig};
use crate::trajectory::{
    GuardRepeatEvent, RepeatGuardConfig, RepeatGuardValidationError, TrajectoryConfig,
    TrajectoryRuntime, install_repeat_guard,
};
#[cfg(feature = "codemode")]
use mizar_codemode::{CodemodeConfig, register_codemode};
use mizar_llm::{
    LlmAdapter, LlmError, LlmRouter, LlmRuntime, RetryConfig, RouterConfig, install_retry,
    register_mock_adapter,
};
#[cfg(feature = "mcp")]
use mizar_mcp::{McpManager, McpPluginConfig};
use mizar_session::persistence::PersistenceError;
use mizar_session::sqlite::SqliteSessionPersistence;
use mizar_session::{SessionPersistence, SessionStore};
use mizar_tools::coding::{
    CodingToolsConfig, CodingToolsPreset, build_tools_guide, register_coding_tools,
};
use mizar_tools::search::{ToolSearchConfig, register_tool_search};
use mizar_tools::{ToolError, ToolRuntime};
use mizar_types::{HookGuard, SyncEventBus};
#[cfg(feature = "mcp")]
use plugins::McpSlot;
use plugins::{PreparedToggles, SpineModuleMeta, SpineServiceRefs};
use thiserror::Error;
#[cfg(feature = "mcp")]
use tokio::sync::Mutex as AsyncMutex;

#[derive(Debug, Error)]
pub enum SpineError {
    #[error(transparent)]
    Persistence(#[from] PersistenceError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    RepeatGuard(#[from] RepeatGuardValidationError),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("技能模块装配失败: {0}")]
    Skills(String),
}

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

/// 静态组合根：按确定 DAG 顺序构建与持有全部核心运行时及可选模块。
pub struct MizarRuntime {
    pub llm: Arc<LlmRuntime>,
    pub tools: Arc<ToolRuntime>,
    pub sessions: Arc<SessionStore>,
    pub persistence: Option<Arc<SqliteSessionPersistence>>,
    pub system_prompt: Arc<SystemPrompt>,
    pub agent_loop: Arc<AgentLoop>,
    pub compaction: Option<Arc<CompactionRuntime>>,
    pub instructions: Option<Arc<AgentInstructions>>,
    pub skills: Option<Arc<SkillRegistry>>,
    pub trajectory: Option<Arc<TrajectoryRuntime>>,
    pub session_title: Option<Arc<SessionTitleRuntime>>,
    pub subagent: Option<Arc<SubagentRuntime>>,
    pub modules: Arc<SpinePlugins>,
    #[cfg(feature = "mcp")]
    mcp_slot: Arc<AsyncMutex<McpSlot>>,
    guards: Mutex<Vec<HookGuard>>,
    router: Mutex<Option<Arc<LlmRouter>>>,
}

impl MizarRuntime {
    pub async fn build(config: AgentSpineConfig) -> Result<Arc<Self>, SpineError> {
        let default_cwd = config.workspace.display().to_string();

        // 1. 模型运行时
        let llm = LlmRuntime::new();

        // 2. 会话持久化（可选）与会话存储
        let persistence = match config.session_db_path {
            Some(ref db_path) => Some(SqliteSessionPersistence::open_async(db_path).await?),
            None => None,
        };
        let sessions = match &persistence {
            Some(p) => SessionStore::with_persistence(Arc::clone(p) as Arc<dyn SessionPersistence>),
            None => SessionStore::new(),
        };
        sessions.set_default_cwd(Some(default_cwd));

        // 3. 工具运行时
        let tools = ToolRuntime::new();
        if config.include_tools.is_some() || !config.exclude_tools.is_empty() {
            tools.set_tool_selection(config.include_tools, config.exclude_tools);
        }

        // 4. 系统提示词与 Agent 主循环
        let system_prompt = SystemPrompt::new(SystemPromptConfig {
            persona: config.persona,
            runtime_protocol: Some(crate::system_prompt::DEFAULT_RUNTIME_PROTOCOL.to_string()),
        });
        let agent_loop = AgentLoop::new(
            Arc::clone(&llm),
            Arc::clone(&tools),
            Arc::clone(&sessions),
            Arc::clone(&system_prompt),
            config.agent,
        );

        // 5. 预构建模块启停开关元信息（确保 Gate 2 校验在模块安装前可用）
        let mut metas: Vec<SpineModuleMeta> = Vec::new();
        let push_meta = |metas: &mut Vec<SpineModuleMeta>, id: &str, label: &str, desc: &str| {
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

        let prepared = PreparedToggles::prepare(metas, config.disabled_plugins);
        let mut guards: Vec<HookGuard> = Vec::new();

        // 6. 在 `instructions` / `skills` 之前装配 `compaction`：
        // 使同一次 `pre_step` 中先完成压缩推进 `compacted_up_to_seq`，
        // 随后指令与技能立即可见新压缩边界并当步完成重注/激活。
        let compaction = if let Some(compaction_cfg) = config.compaction {
            let (rt, guard) = CompactionRuntime::install(
                Arc::clone(&llm),
                &sessions,
                &tools,
                Arc::clone(&system_prompt),
                &agent_loop,
                compaction_cfg,
            );
            guards.push(guard);
            Some(rt)
        } else {
            None
        };

        // 7. 编程工具栈（Coding / Search / Codemode / MCP / Skills / Instructions）
        #[cfg(feature = "mcp")]
        let mcp_slot = Arc::new(AsyncMutex::new(McpSlot::default()));
        let mut skills: Option<Arc<SkillRegistry>> = None;
        let mut instructions: Option<Arc<AgentInstructions>> = None;

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
                skills: skills_cfg,
                instructions: instr_cfg,
            } => {
                let inject_prompt = coding.inject_system_prompt;
                guards.extend(register_coding_tools(&tools, &coding)?);
                if inject_prompt {
                    let tools_ref = Arc::clone(&tools);
                    let coding_flag = prepared
                        .flag_for("tools-coding")
                        .unwrap_or_else(|| Arc::new(AtomicBool::new(true)));
                    guards.push(system_prompt.section(PromptSection::new_for_session(
                        "tools_guide",
                        20,
                        move |session_id| {
                            if !coding_flag.load(Ordering::Acquire) {
                                return None;
                            }
                            build_tools_guide(&tools_ref.list_declared_infos(session_id))
                        },
                    )));
                }
                guards.extend(register_tool_search(Arc::clone(&tools), tool_search)?);
                #[cfg(feature = "codemode")]
                guards.extend(register_codemode(Arc::clone(&tools), codemode)?);
                #[cfg(feature = "mcp")]
                {
                    let mut slot = mcp_slot.lock().await;
                    if prepared.is_enabled("mcp") {
                        slot.active = Some(McpManager::start(Arc::clone(&tools), mcp).await);
                    } else {
                        slot.pending = Some(mcp);
                    }
                }

                let skill_reg = SkillRegistry::new(SkillRegistryConfig::default());
                guards.push(
                    install_skills(
                        Arc::clone(&skill_reg),
                        Arc::clone(&tools),
                        &agent_loop,
                        skills_cfg,
                    )
                    .await
                    .map_err(|e| SpineError::Skills(e.to_string()))?,
                );
                skills = Some(skill_reg);

                let (instr_rt, instr_guard) =
                    AgentInstructions::install(&tools, &agent_loop, instr_cfg);
                guards.push(instr_guard);
                instructions = Some(instr_rt);
            }
        }

        // 8. 会话标题自动生成
        let session_title = if let Some(st_cfg) = config.session_title {
            let (rt, guard) = SessionTitleRuntime::install(Arc::clone(&llm), &agent_loop, st_cfg);
            guards.push(guard);
            Some(rt)
        } else {
            None
        };

        // 10. 单层子代理调度
        let subagent = if let Some(sub_cfg) = config.subagent {
            let (rt, guard) = SubagentRuntime::install(
                Arc::clone(&llm),
                Arc::clone(&tools),
                Arc::clone(&sessions),
                Arc::clone(&agent_loop),
                &system_prompt,
                sub_cfg,
            )?;
            guards.push(guard);
            Some(rt)
        } else {
            None
        };

        // 11. 重复工具调用守卫（须先于或并列提供总线给 TrajectoryRuntime）
        let repeat_bus: Option<Arc<SyncEventBus<GuardRepeatEvent>>> =
            if let Some(mut guard_cfg) = config.repeat_guard {
                guard_cfg.enabled_flag = prepared.flag_for("repeat-guard");
                let (bus, guard) = install_repeat_guard(&tools, Some(&agent_loop), guard_cfg)?;
                guards.push(guard);
                Some(bus)
            } else {
                None
            };

        // 12. 轨迹采集与异常检测
        let trajectory = if let Some(traj_cfg) = config.trajectory {
            let (rt, guard) = TrajectoryRuntime::install(
                Some(&sessions),
                Some(&tools),
                Some(&agent_loop),
                repeat_bus.as_ref(),
                traj_cfg,
            );
            guards.push(guard);
            Some(rt)
        } else {
            None
        };

        // 13. 控制器初始化与启动期禁用门控同步
        let modules = SpinePlugins::from_prepared(
            prepared,
            SpineServiceRefs {
                tools: Arc::clone(&tools),
                compaction: compaction.clone(),
                instructions: instructions.clone(),
                skills: skills.clone(),
                session_title: session_title.clone(),
                subagent: subagent.clone(),
                trajectory: trajectory.clone(),
                #[cfg(feature = "mcp")]
                mcp_slot: Arc::clone(&mcp_slot),
            },
        );
        modules.apply_initial_gates();

        Ok(Arc::new(Self {
            llm,
            tools,
            sessions,
            persistence,
            system_prompt,
            agent_loop,
            compaction,
            instructions,
            skills,
            trajectory,
            session_title,
            subagent,
            modules,
            #[cfg(feature = "mcp")]
            mcp_slot,
            guards: Mutex::new(guards),
            router: Mutex::new(None),
        }))
    }

    pub fn push_guard(&self, guard: HookGuard) {
        self.guards.lock().unwrap().push(guard);
    }

    pub fn install_retry(&self, config: RetryConfig) {
        self.push_guard(install_retry(&self.llm, config));
    }

    pub fn install_router(&self, config: RouterConfig) -> Arc<LlmRouter> {
        let router = LlmRouter::new(Arc::clone(&self.llm), config);
        *self.router.lock().unwrap() = Some(Arc::clone(&router));
        router
    }

    pub fn router(&self) -> Option<Arc<LlmRouter>> {
        self.router.lock().unwrap().clone()
    }

    pub fn register_adapter(
        &self,
        name: impl Into<String>,
        adapter: Arc<dyn LlmAdapter>,
    ) -> Result<(), LlmError> {
        let guard = self.llm.register_adapter(name, adapter)?;
        self.push_guard(guard);
        Ok(())
    }

    pub fn register_mock_adapter(&self) -> Result<(), LlmError> {
        let guard = register_mock_adapter(&self.llm)?;
        self.push_guard(guard);
        Ok(())
    }

    #[cfg(feature = "mcp")]
    pub async fn mcp(&self) -> Option<Arc<McpManager>> {
        self.mcp_slot.lock().await.active.clone()
    }

    #[cfg(feature = "mcp")]
    pub fn try_mcp(&self) -> Option<Arc<McpManager>> {
        self.mcp_slot
            .try_lock()
            .ok()
            .and_then(|guard| guard.active.clone())
    }

    /// 显式异步优雅关机：按 LIFO 逆序卸载钩子与工具、断开 MCP 服务器、刷盘会话并关闭 SQLite 连接。
    pub async fn shutdown(&self) {
        let guards = {
            let mut g = self.guards.lock().unwrap();
            std::mem::take(&mut *g)
        };
        for guard in guards.into_iter().rev() {
            drop(guard);
        }

        let _router = self.router.lock().unwrap().take();

        #[cfg(feature = "mcp")]
        {
            let active = self.mcp_slot.lock().await.active.take();
            if let Some(mgr) = active {
                mgr.close_all().await;
            }
        }

        for session in self.sessions.list() {
            let _ = session.flush_async().await;
        }
        if let Some(persistence) = &self.persistence {
            let p = Arc::clone(persistence);
            let _ = tokio::task::spawn_blocking(move || p.shutdown()).await;
        }
    }
}

impl Drop for MizarRuntime {
    fn drop(&mut self) {
        let guards = {
            let mut g = self.guards.lock().unwrap();
            std::mem::take(&mut *g)
        };
        for guard in guards.into_iter().rev() {
            drop(guard);
        }
        let _router = self.router.lock().unwrap().take();
    }
}
