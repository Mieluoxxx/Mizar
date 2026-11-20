//! 终端交互入口 `mizar`（`mizar-tui`）。
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
//! - `--help` / `-h`：打印命令行帮助信息并退出
//! - `--version` / `-V`：打印版本号并退出
//!
//! 子命令：
//! - `mizar update [--version <ver>] [--force] [--check]`：从 GitHub Releases 检查并更新 `mizar` 二进制
//!
//! 配置文件目录（全局 `~/.mizar/` + 项目 `<workspace>/.mizar/`，支持 `MIZAR_HOME` 覆盖）：
//! - `settings.json`：默认提供方/模型、推理强度、工具预设与黑白名单
//! - `models.json`：多模型路由配置
//! - `mcp.json`：MCP Server 配置
//!
//! 环境变量（优先覆盖配置文件）：
//! - `API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`（未配置时使用离线假模型）
//! - `PROTOCOL`：`completions`（默认）/ `responses` / `messages`
//! - `BASE_URL`、`MODEL`、`WORKSPACE`、`MIZAR_HOME`
//! - `MIZAR_SKIP_VERSION_CHECK` / `MIZAR_OFFLINE`：跳过启动时版本检查（`MIZAR_OFFLINE` 同时禁用更新下载）
//! - `MIZAR_RELEASE_BASE_URL`：覆盖 GitHub Release 基址（默认为 `https://github.com/Mieluoxxx/Mizar`）

use std::io::IsTerminal;
use std::path::PathBuf;

use cordis::{Context, Service};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, load_mizar_config,
    resolve_default_session_db_path,
};
use mizar_llm_mock::MockPlugin;
use mizar_llm_retry::{LlmRetryPlugin, RetryConfig};
use mizar_llm_router::LlmRouter;
use mizar_tools_coding::CodingToolsPreset;
use mizar_tui::update::{current_version, is_version_check_disabled, run_update_cli};
use mizar_tui::{
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
    help: bool,
    version: bool,
}

fn format_help_text() -> String {
    format!(
        "\
mizar {}
Terminal-native AI coding workbench

Usage:
  mizar [OPTIONS]
  mizar update [OPTIONS]

Commands:
  update                        Check and install the latest release binary

Options:
  -C, --workspace <dir>         Workspace directory (defaults to current directory)
      --minimal                 Enable minimal coding preset (read, bash, edit, write)
      --read-only               Enable read-only preset (read, grep, find)
  -t, --tools <a,b,...>         Tool allowlist (comma-separated, supports *)
 -xt, --exclude-tools <a,b,...> Tool denylist (comma-separated, supports *)
      --inline                  Start in inline viewport mode (preserves scrollback)
      --approve                 Prompt for approval on high-risk tools
  -c, --continue                Resume the latest persisted session in workspace
  -r, --resume <id>             Resume a specific session by ID
      --no-persist              Disable SQLite session persistence
      --no-instructions         Disable AGENTS.md / CLAUDE.md discovery
      --theme <preset>          Theme preset (auto, mist-dark, mist-light, sakura-dark, system, ansi16)
  -h, --help                    Print help
  -V, --version                 Print version",
        current_version()
    )
}

fn parse_cli_args_from<I>(args: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut out = CliArgs::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--workspace" | "-C" => {
                let val = args.next().ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                out.workspace = Some(PathBuf::from(val));
            }
            "--minimal" => out.preset = Some(CodingToolsPreset::MinimalCoding),
            "--read-only" => out.preset = Some(CodingToolsPreset::ReadOnly),
            "--tools" | "-t" => {
                let val = args.next().ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                let list: Vec<String> = val
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .collect();
                out.include_tools = Some(list);
            }
            "--exclude-tools" | "-xt" => {
                let val = args.next().ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                out.exclude_tools.extend(
                    val.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToString::to_string),
                );
            }
            "--inline" => out.inline = true,
            "--approve" => out.approve = true,
            "--no-persist" => out.no_persist = true,
            "--no-instructions" => out.no_instructions = true,
            "--continue" | "-c" => out.resume_session = Some(ResumeTarget::LatestInWorkspace),
            "--resume" | "-r" => {
                let val = args.next().ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                out.resume_session = Some(ResumeTarget::Id(val));
            }
            "--theme" => {
                let val = args.next().ok_or_else(|| format!("缺少 {arg} 的参数值"))?;
                out.theme = ThemePreset::parse(&val);
            }
            "--help" | "-h" => out.help = true,
            "--version" | "-V" => out.version = true,
            other => {
                return Err(format!(
                    "未知参数：{other}（运行 `mizar --help` 查看可用参数）"
                ));
            }
        }
    }
    Ok(out)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if let Some((cmd, rest)) = raw_args.split_first()
        && cmd == "update"
    {
        if let Err(err) = run_update_cli(rest).await {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let cli = match parse_cli_args_from(raw_args) {
        Ok(cli) => cli,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    };
    if cli.help {
        println!("{}", format_help_text());
        return Ok(());
    }
    if cli.version {
        println!("mizar {}", current_version());
        return Ok(());
    }
    let app = Context::new();
    // 进入终端界面前先静默，防止启动日志先刷到 stdout 再闪屏
    app.logger().set_silent();

    let explicit_workspace = cli.workspace.clone();
    let workspace = cli
        .workspace
        .or_else(|| env_non_empty("WORKSPACE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));

    let loaded_cfg = load_mizar_config(&workspace);
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
    // 记录强度来源：settings.json 显式指定时才算「配置的默认」，否则留给会话记录覆盖
    tui_cfg.reasoning_effort_from_settings = loaded_cfg.settings.reasoning_effort.is_some();
    // 模型来源：环境变量 MODEL 压过一切；settings.json 的 defaultModel 次之；
    // 两者都没有时 `model` 仍可能是 `models.json` 推导出来的，那不算显式指定。
    tui_cfg.model_from_env = env_non_empty("MODEL").is_some();
    tui_cfg.model_from_settings = loaded_cfg.settings.default_model.is_some();
    if let Some(ui) = &loaded_cfg.settings.ui {
        if let Some(ref theme_name) = ui.theme {
            tui_cfg.theme_preset = mizar_tui_ui::theme::ThemePreset::parse(theme_name);
        }
        if let Some(ref mode) = ui.screen_mode {
            tui_cfg.screen_mode = match mode.trim().to_ascii_lowercase().as_str() {
                "inline" => ScreenMode::Inline,
                _ => ScreenMode::Fullscreen,
            };
        }
        tui_cfg.show_thinking = ui.show_thinking;
        tui_cfg.verbose_expanded = ui.verbose_expanded;
        tui_cfg.side_panel_open = ui.side_panel_open;
        tui_cfg.mouse_capture = ui.mouse_capture;
        if let Some(ref approval) = ui.approval_mode {
            tui_cfg.approval_mode = match approval.trim().to_ascii_lowercase().as_str() {
                "ask" => ApprovalMode::Ask,
                _ => ApprovalMode::Auto,
            };
        }
    }
    if cli.theme.is_some() {
        tui_cfg.theme_preset = cli.theme;
    }
    tui_cfg.resume_session = cli.resume_session;
    tui_cfg.restrict_to_workspace = loaded_cfg.settings.restrict_to_workspace.unwrap_or(false);
    tui_cfg.check_for_updates = !is_version_check_disabled();
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
            "未检测到 API_KEY 或 ~/.mizar/models.json，当前使用离线假模型（试试「项目里有哪些文件」或「搜索 Cargo」，输入 /tools 查看工具）"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_version_and_help_flags() {
        assert!(
            parse_cli_args_from(["--version".to_string()])
                .unwrap()
                .version
        );
        assert!(parse_cli_args_from(["-V".to_string()]).unwrap().version);
        assert!(parse_cli_args_from(["--help".to_string()]).unwrap().help);
        assert!(parse_cli_args_from(["-h".to_string()]).unwrap().help);
        assert!(parse_cli_args_from(["--bogus".to_string()]).is_err());
        assert!(parse_cli_args_from(["-C".to_string()]).is_err());
    }

    #[test]
    fn resolve_version_and_help_output_non_empty() {
        let ver = current_version();
        assert!(!ver.is_empty());
        let help = format_help_text();
        assert!(help.contains(&format!("mizar {ver}")));
        assert!(help.contains("update"));
        assert!(help.contains("--version"));
        assert!(help.contains("--workspace"));
    }

    #[test]
    fn parse_existing_cli_flags_preserved() {
        let parsed = parse_cli_args_from([
            "-C".to_string(),
            "/tmp/repo".to_string(),
            "--minimal".to_string(),
            "-t".to_string(),
            "read, edit".to_string(),
            "-xt".to_string(),
            "bash".to_string(),
            "--inline".to_string(),
            "--approve".to_string(),
            "--no-persist".to_string(),
            "--no-instructions".to_string(),
            "-c".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.workspace, Some(PathBuf::from("/tmp/repo")));
        assert!(matches!(
            parsed.preset,
            Some(CodingToolsPreset::MinimalCoding)
        ));
        assert_eq!(
            parsed.include_tools,
            Some(vec!["read".to_string(), "edit".to_string()])
        );
        assert_eq!(parsed.exclude_tools, vec!["bash".to_string()]);
        assert!(parsed.inline);
        assert!(parsed.approve);
        assert!(parsed.no_persist);
        assert!(parsed.no_instructions);
        assert!(matches!(
            parsed.resume_session,
            Some(ResumeTarget::LatestInWorkspace)
        ));
    }
}
