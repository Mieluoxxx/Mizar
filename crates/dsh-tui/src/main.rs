//! 终端交互入口 `dsh-tui`。
//!
//! CLI 参数：
//! - `--workspace <dir>` / `-C <dir>`：工作目录
//! - `--minimal`：仅启用极简 4 工具（`read` / `bash` / `edit` / `write`）
//! - `--read-only`：仅启用只读 3 工具（`read` / `grep` / `find`）
//! - `--tools <a,b,...>` / `-t`：工具白名单
//! - `--exclude-tools <a,b,...>` / `-xt`：工具黑名单（支持 `*` 通配）
//! - `--inline`：以 Inline 视口模式启动（保留终端原生滚动历史）
//! - `--approve`：开启高危工具交互式审批闸门
//! - `--continue` / `-c`：恢复当前工作区最近一次持久化的历史会话
//! - `--resume <id>`：按会话 ID 恢复历史会话
//! - `--no-persist`：禁用 SQLite 会话落盘（仅当前进程内存模式）
//! - `--no-instructions`：禁用自动发现与注入 `AGENTS.md` / `CLAUDE.md`
//! - `--theme <preset>`：指定预设主题（`auto` / `mist-dark` / `mist-light` / `sakura-dark` / `system` / `ansi16`）
//!
//! 配置文件目录（全局 `~/.dsh-rs/` + 项目 `<workspace>/.dsh-rs/`，支持 `DSH_RS_HOME` 覆盖）：
//! - `settings.json`：默认提供方/模型、推理强度、工具预设与黑白名单
//! - `models.json`：多模型路由配置
//! - `mcp.json`：MCP Server 配置
//!
//! 环境变量（优先覆盖配置文件）：
//! - `API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`（未配置时使用离线假模型）
//! - `PROTOCOL`：`completions`（默认）/ `responses` / `messages`
//! - `BASE_URL`、`MODEL`、`WORKSPACE`、`DSH_RS_HOME`

use std::io::IsTerminal;
use std::path::PathBuf;

use cordis::{Context, Service};
use dsh_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, load_dsh_config,
    resolve_default_session_db_path,
};
use dsh_llm_mock::MockPlugin;
use dsh_llm_retry::{LlmRetryPlugin, RetryConfig};
use dsh_llm_router::LlmRouter;
use dsh_tools_coding::CodingToolsPreset;
use dsh_tui::{
    ApprovalMode, NoticeLevel, ResumeTarget, ScreenMode, ThemePreset, TuiConfig,
    format_resume_command, run_tui_with_summary,
};

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

#[derive(Default)]
struct CliArgs {
    workspace: Option<PathBuf>,
    preset: Option<CodingToolsPreset>,
    include_tools: Option<Vec<String>>,
    exclude_tools: Vec<String>,
    inline: bool,
    approve: bool,
    no_persist: bool,
    no_instructions: bool,
    resume_session: Option<ResumeTarget>,
    theme: Option<ThemePreset>,
}

fn parse_cli_args() -> CliArgs {
    let mut out = CliArgs::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workspace" | "-C" => {
                if let Some(val) = args.next() {
                    out.workspace = Some(PathBuf::from(val));
                }
            }
            "--minimal" => out.preset = Some(CodingToolsPreset::MinimalCoding),
            "--read-only" => out.preset = Some(CodingToolsPreset::ReadOnly),
            "--tools" | "-t" => {
                if let Some(val) = args.next() {
                    let list: Vec<String> = val
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToString::to_string)
                        .collect();
                    out.include_tools = Some(list);
                }
            }
            "--exclude-tools" | "-xt" => {
                if let Some(val) = args.next() {
                    out.exclude_tools.extend(
                        val.split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(ToString::to_string),
                    );
                }
            }
            "--inline" => out.inline = true,
            "--approve" => out.approve = true,
            "--no-persist" => out.no_persist = true,
            "--no-instructions" => out.no_instructions = true,
            "--continue" | "-c" => out.resume_session = Some(ResumeTarget::LatestInWorkspace),
            "--resume" | "-r" => {
                if let Some(val) = args.next() {
                    out.resume_session = Some(ResumeTarget::Id(val));
                }
            }
            "--theme" => {
                if let Some(val) = args.next() {
                    out.theme = ThemePreset::parse(&val);
                }
            }
            _ => {}
        }
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = parse_cli_args();
    let app = Context::new();
    // 进入终端界面前先静默，防止启动日志先刷到 stdout 再闪屏
    app.logger().set_silent();

    let explicit_workspace = cli.workspace.clone();
    let workspace = cli
        .workspace
        .or_else(|| env_non_empty("WORKSPACE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));

    let loaded_cfg = load_dsh_config(&workspace);
    let mut spine_cfg = AgentSpineConfig::coding(workspace.clone());
    if !cli.no_persist {
        spine_cfg.session_db_path = resolve_default_session_db_path();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    if cli.no_persist {
        spine_cfg.session_db_path = None;
    }
    if cli.no_instructions
        && let AgentToolsMode::Coding {
            ref mut instructions,
            ..
        } = spine_cfg.tools_mode
    {
        instructions.enabled = false;
    }

    if let Some(preset) = cli.preset {
        spine_cfg = spine_cfg.with_coding_preset(preset);
    }
    if cli.include_tools.is_some() {
        spine_cfg.include_tools = cli.include_tools;
    }
    for pat in cli.exclude_tools {
        if !spine_cfg.exclude_tools.contains(&pat) {
            spine_cfg.exclude_tools.push(pat);
        }
    }

    let resolved_llm = loaded_cfg.resolve_llm_router();
    if let Some(ref resolved) = resolved_llm {
        spine_cfg.agent.provider = Some(resolved.active_provider.clone());
        spine_cfg.agent.model = resolved.active_model.clone();
    }

    let mut tui_cfg = TuiConfig::new(workspace);
    tui_cfg.max_steps = spine_cfg.agent.max_steps;
    tui_cfg.reasoning_effort = spine_cfg.agent.reasoning_effort;
    tui_cfg.theme_preset = cli.theme;
    tui_cfg.resume_session = cli.resume_session;
    tui_cfg.restrict_to_workspace = loaded_cfg.settings.restrict_to_workspace.unwrap_or(false);
    if cli.inline {
        tui_cfg.screen_mode = ScreenMode::Inline;
    }
    if cli.approve {
        tui_cfg.approval_mode = ApprovalMode::Ask;
    }
    for err in loaded_cfg.errors {
        tui_cfg
            .startup_notices
            .push((NoticeLevel::Warn, format!("配置解析失败：{err}")));
    }

    app.plugin(AgentSpinePlugin, spine_cfg);
    app.plugin(LlmRetryPlugin, RetryConfig::default());

    if let Some(resolved) = resolved_llm {
        app.plugin(LlmRouter::plugin(), resolved.router);
        tui_cfg.provider = Some(resolved.active_provider);
        tui_cfg.model = resolved.active_model;
    } else {
        app.plugin(MockPlugin, ());
        tui_cfg.provider = Some("mock".to_string());
        tui_cfg.model = Some("mock-v1".to_string());
        tui_cfg.startup_notices.push((
            NoticeLevel::Warn,
            "未检测到 API_KEY 或 ~/.dsh-rs/models.json，当前使用离线假模型（试试「项目里有哪些文件」或「搜索 Cargo」，输入 /tools 查看工具）"
                .to_string(),
        ));
    }

    let run_res = run_tui_with_summary(app.clone(), tui_cfg).await;
    app.fiber().dispose().await;
    match run_res {
        Ok(summary) => {
            if std::io::stdout().is_terminal()
                && let Some(cmd) = format_resume_command(
                    &summary,
                    explicit_workspace.as_deref(),
                    std::env::current_exe().ok().as_deref(),
                    std::env::current_dir().ok().as_deref(),
                )
            {
                println!("\n可使用以下命令恢复此会话：\n  {cmd}");
            }
            Ok(())
        }
        Err(err) => Err(err),
    }
}
