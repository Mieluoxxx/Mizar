//! 套件插件（对应 `NamiDSH/packages/agent-spine` + `pi` Coding Agent 套件）。
//!
//! 把每个 Agent 都需要的核心服务一次挂上；
//! 「连哪个模型」和「入口（网页 / 命令行 / TUI）」故意留在外面，由使用方决定。

pub mod config;

use std::path::PathBuf;

pub use config::{
    CONFIG_DIR_NAME, CodemodeSettings, EnvLlmOverride, LoadedDshConfig, ResolvedLlmRouter,
    SettingsConfig, SkillsSettings, ToolSearchSettings, load_dsh_config,
    load_dsh_config_with_global, resolve_default_session_db_path, resolve_global_config_dir,
    resolve_project_config_dir,
};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Plugin, Service};
use dsh_agent_loop::{AgentLoop, AgentLoopConfig};
use dsh_codemode::{CodemodeConfig, CodemodePlugin};
use dsh_llm::LlmRuntime;
use dsh_mcp::{McpManager, McpPluginConfig};
use dsh_session::{SessionContextExt, SessionStore};
use dsh_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
use dsh_skills::{SkillRegistry, SkillRegistryConfig, SkillsConfig, SkillsPlugin};
use dsh_system_prompt::{SystemPrompt, SystemPromptConfig};
use dsh_tool_fs::{ToolFsConfig, ToolFsPlugin};
use dsh_tool_search::{ToolSearchConfig, ToolSearchPlugin};
use dsh_tool_time::ToolTimePlugin;
use dsh_tools::{ToolRuntime, ToolsContextExt};
use dsh_tools_coding::{CodingToolsConfig, CodingToolsPlugin, CodingToolsPreset};
use dsh_trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, TrajectoryConfig, TrajectoryRuntime,
};

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum AgentToolsMode {
    /// 保留原有最小只读套件（`get_time` + `list_dir` + `read_file`）。
    Legacy,
    /// 完整编程套件：`read` + `bash` + `edit` + `write` + `grep` + `find` + `skill` + `tool_search` + `codemode` + `mcp`。
    Coding {
        coding: CodingToolsConfig,
        tool_search: ToolSearchConfig,
        codemode: CodemodeConfig,
        mcp: McpPluginConfig,
        skills: SkillsConfig,
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
    /// 可选：会话 SQLite 持久化数据库路径；为 `None` 时保持纯内存模式（适合测试）。
    pub session_db_path: Option<PathBuf>,
}

impl AgentSpineConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            persona: None,
            agent: AgentLoopConfig::default(),
            tools_mode: AgentToolsMode::Legacy,
            include_tools: None,
            exclude_tools: Vec::new(),
            trajectory: Some(TrajectoryConfig::default()),
            repeat_guard: Some(RepeatGuardConfig::default()),
            session_db_path: None,
        }
    }

    pub fn coding(workspace: impl Into<PathBuf>) -> Self {
        let ws = workspace.into();
        Self {
            workspace: ws.clone(),
            persona: Some(
                "你是一个高效、务实的编程与终端 AI 助手，擅长检查代码、执行命令和精确修改文件。回答简洁准确。".to_string(),
            ),
            agent: AgentLoopConfig::default(),
            tools_mode: AgentToolsMode::Coding {
                coding: CodingToolsConfig::new(&ws),
                tool_search: ToolSearchConfig::default(),
                codemode: CodemodeConfig::default(),
                mcp: McpPluginConfig::new(&ws),
                skills: SkillsConfig::new(&ws),
            },
            include_tools: None,
            exclude_tools: Vec::new(),
            trajectory: Some(TrajectoryConfig::default()),
            repeat_guard: Some(RepeatGuardConfig::default()),
            session_db_path: None,
        }
    }

    pub fn with_session_db(mut self, path: impl Into<PathBuf>) -> Self {
        self.session_db_path = Some(path.into());
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
            ctx.try_plugin(
                SystemPrompt::plugin(),
                SystemPromptConfig {
                    persona: config.persona,
                },
            )?;

            match config.tools_mode {
                AgentToolsMode::Legacy => {
                    ctx.try_plugin(ToolTimePlugin, ())?;
                    ctx.try_plugin(ToolFsPlugin, ToolFsConfig::new(config.workspace))?;
                }
                AgentToolsMode::Coding {
                    coding,
                    tool_search,
                    codemode,
                    mcp,
                    skills,
                } => {
                    ctx.try_plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())?;
                    ctx.try_plugin(CodingToolsPlugin, coding)?;
                    ctx.try_plugin(ToolSearchPlugin, tool_search)?;
                    ctx.try_plugin(CodemodePlugin, codemode)?;
                    ctx.try_plugin(McpManager::plugin(), mcp)?;
                    ctx.try_plugin(SkillsPlugin, skills)?;
                }
            }

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

            ctx.try_plugin(AgentLoop::plugin(), config.agent)?;
            if let Some(traj_cfg) = config.trajectory {
                ctx.try_plugin(TrajectoryRuntime::plugin(), traj_cfg)?;
            }
            if let Some(guard_cfg) = config.repeat_guard {
                ctx.try_plugin(RepeatToolGuardPlugin, guard_cfg)?;
            }
            Ok(None)
        })
    }
}
