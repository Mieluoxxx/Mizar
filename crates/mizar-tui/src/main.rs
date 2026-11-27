//! 终端交互与无头单次执行入口 `mizar`（`mizar-tui`）。
//!
//! 子命令：
//! - `mizar exec [<prompt>]`：非交互单次无头执行（无 explicit prompt 且 stdin 非 TTY 时读取 stdin 至 EOF）
//! - `mizar update [--version <ver>] [--force] [--check]`：从 GitHub Releases 检查并更新 `mizar` 二进制
//!
//! CLI 参数：
//! - `--workspace <dir>` / `-C <dir>`：工作目录
//! - `-p <text>` / `--prompt <text>`：无头模式直接传入任务文本（可用于 root 或 `exec`）
//! - `--prompt-file <path>`：从文件读取无头任务文本（可用于 root 或 `exec`）
//! - `--provider <id>`：指定提供方（覆盖配置与环境变量）
//! - `--model <id>`：指定模型 ID（覆盖配置与环境变量）
//! - `--protocol <api>` / `--api-format <api>`：指定协议（`completions` / `responses` / `messages` 等）
//! - `--base-url <url>`：覆盖 API 基址
//! - `--reasoning-effort <level>`：推理强度（`off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`）
//! - `--max-steps <n>`：限制单轮最大工具循环步数
//! - `--timeout <secs>`：限制无头单次任务总超时秒数
//! - `--trace-out <path>`：无头模式退出前导出结构化轨迹 JSON（可用环境变量 `MIZAR_TRACE_OUT` 指定，CLI 优先）
//! - `--isolated`：隔离模式，跳过个人 `~/.mizar` / `~/.agents` 配置、MCP、Skills、全局指令与全局 SQLite
//! - `--read-only`：仅启用只读 3 工具（`read` / `grep` / `find`）
//! - `--tools <a,b,...>` / `-t`：工具白名单
//! - `--exclude-tools <a,b,...>` / `-xt`：工具黑名单（支持 `*` 通配）
//! - `--inline`：以 Inline 视口模式启动 TUI（保留终端原生滚动历史）
//! - `--approve`：开启高危工具交互式审批闸门（TUI 专用；无头模式不兼容并直接报错）
//! - `--continue` / `-c`：恢复当前工作区最近一次持久化的历史会话（TUI）
//! - `--resume [id]` / `-r [id]`：按会话 ID 恢复历史会话；省略 ID 时弹出交互式会话选择器（TUI）
//! - `--no-persist`：禁用 SQLite 会话落盘（仅当前进程内存模式）
//! - `--no-instructions`：禁用自动发现与注入 `AGENTS.md` / `CLAUDE.md`
//! - `--theme <preset>`：指定预设主题（`auto` / `mist-dark` / `mist-light` / `sakura-dark` / `system` / `ansi16`）
//! - `--help` / `-h`：打印命令行帮助信息并退出
//! - `--version` / `-V`：打印版本号并退出
//!
//! 配置文件目录（全局 `~/.mizar/` + 项目 `<workspace>/.mizar/`，支持 `MIZAR_HOME` 覆盖）：
//! - `settings.json`：默认提供方/模型、推理强度、工具预设与黑白名单
//! - `models.json`：多模型路由配置
//! - `mcp.json`：MCP Server 配置
//!
//! 环境变量（优先覆盖配置文件）：
//! - `API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`（TUI 未配置时回落离线假模型；无头模式严禁回落假模型）
//! - `PROTOCOL`：`completions`（默认）/ `responses` / `messages`
//! - `BASE_URL`、`MODEL`、`WORKSPACE`、`MIZAR_HOME`
//! - `MIZAR_SKIP_VERSION_CHECK` / `MIZAR_OFFLINE`：跳过启动时版本检查（`MIZAR_OFFLINE` 同时禁用更新下载）
//! - `MIZAR_RELEASE_BASE_URL`：覆盖 GitHub Release 基址（默认为 `https://github.com/Mieluoxxx/Mizar`）
//! - `MIZAR_TRACE_OUT`：无头模式结构化轨迹 JSON 导出路径（低于 `--trace-out` 优先级）

use std::io::IsTerminal;
use std::path::PathBuf;

use cordis::{Context, Service};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, AgentToolsMode, CliLlmOverride, EnvLlmOverride,
    load_mizar_config, parse_context_window, resolve_default_session_db_path,
};
use mizar_llm::ReasoningEffort;
use mizar_llm_mock::MockPlugin;
use mizar_llm_router::LlmRouter;
use mizar_tools_coding::CodingToolsPreset;
use mizar_tui::exec::{
    ExecConfig, ExecOutputFormat, report_exec_input_error, resolve_exec_prompt,
    resolve_trace_out_path, run_exec_cli,
};
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
    explicit_exec: bool,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    positional_prompt: Option<String>,
    isolated: bool,
    provider: Option<String>,
    model: Option<String>,
    protocol: Option<String>,
    base_url: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
    max_steps: Option<usize>,
    max_tokens: Option<u32>,
    context_window: Option<u64>,
    timeout_secs: Option<u64>,
    output_format: Option<ExecOutputFormat>,
    trace_out: Option<PathBuf>,
}

impl CliArgs {
    fn is_exec_mode(&self) -> bool {
        self.explicit_exec || self.prompt.is_some() || self.prompt_file.is_some()
    }

    fn has_exec_only_flags(&self) -> bool {
        self.isolated
            || self.provider.is_some()
            || self.model.is_some()
            || self.protocol.is_some()
            || self.base_url.is_some()
            || self.reasoning_effort.is_some()
            || self.max_steps.is_some()
            || self.max_tokens.is_some()
            || self.timeout_secs.is_some()
            || self.output_format.is_some()
            || self.trace_out.is_some()
    }
}

fn format_help_text() -> String {
    format!(
        "\
mizar {}
Terminal-native AI coding workbench

Usage:
  mizar [OPTIONS]
  mizar exec [OPTIONS] [<prompt>]
  mizar update [OPTIONS]

Commands:
  exec                          Run a single non-interactive task and exit
  update                        Check and install the latest release binary

Options:
  -C, --workspace <dir>         Workspace directory (defaults to current directory)
  -p, --prompt <text>           Task prompt (runs headless exec when used on root)
      --prompt-file <path>      Read task prompt from file (runs headless exec)
      --provider <id>           Model provider ID override
      --model <id>              Model ID override
      --protocol <api>          Protocol override (completions, responses, messages)
      --api-format <api>        Alias for --protocol
      --base-url <url>          Base URL override
      --reasoning-effort <lvl>  Reasoning effort (off, minimal, low, medium, high, xhigh, max)
      --max-steps <n>           Maximum agent loop steps (>= 1)
      --max-tokens <n>          Maximum output tokens per model call (>= 1)
      --context-window <tokens> Override the model context window in TUI and exec (>= 1, env: MIZAR_CONTEXT_WINDOW)
      --timeout <secs>          Total execution timeout in seconds (>= 1)
      --output-format <fmt>     text (default), json or ndjson (one flushed event per line, result last)
      --json                    Shorthand for --output-format ndjson
      --trace-out <path>        Export structured trajectory JSON on exit (env: MIZAR_TRACE_OUT)
      --isolated                Skip personal global config, MCP, skills, instructions, and DB
      --minimal                 Use the default coding preset (read, bash, edit, write)
      --read-only               Enable read-only preset (read, grep, find)
  -t, --tools <a,b,...>         Tool allowlist (comma-separated, supports *)
 -xt, --exclude-tools <a,b,...> Tool denylist (comma-separated, supports *)
      --inline                  Start in inline viewport mode (TUI only)
      --approve                 Prompt for approval on high-risk tools (TUI only)
  -c, --continue                Resume the latest persisted session in workspace
  -r, --resume [id]             Open picker in TUI, or resume persisted session by ID in exec
      --no-persist              Disable SQLite session persistence
      --no-instructions         Disable AGENTS.md / CLAUDE.md discovery
      --theme <preset>          Theme preset (auto, mist-dark, mist-light, sakura-dark, system, ansi16)
  -h, --help                    Print help
  -V, --version                 Print version",
        current_version()
    )
}

fn split_flag_value(raw: &str) -> (&str, Option<String>) {
    if raw.starts_with("--")
        && let Some((name, val)) = raw.split_once('=')
    {
        return (name, Some(val.to_string()));
    }
    for prefix in ["-C=", "-p=", "-t=", "-r=", "-xt="] {
        if let Some(val) = raw.strip_prefix(prefix) {
            let flag = &prefix[..prefix.len() - 1];
            return (flag, Some(val.to_string()));
        }
    }
    (raw, None)
}

fn take_value<I>(flag: &str, inline: Option<String>, args: &mut I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    match inline {
        Some(v) => Ok(v),
        None => args.next().ok_or_else(|| format!("缺少 {flag} 的参数值")),
    }
}

fn parse_cli_reasoning_effort(raw: &str) -> Result<ReasoningEffort, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" => Ok(ReasoningEffort::Off),
        "minimal" => Ok(ReasoningEffort::Minimal),
        "low" => Ok(ReasoningEffort::Low),
        "medium" => Ok(ReasoningEffort::Medium),
        "high" => Ok(ReasoningEffort::High),
        "xhigh" => Ok(ReasoningEffort::XHigh),
        "max" => Ok(ReasoningEffort::Max),
        _ => Err(format!(
            "无效的 --reasoning-effort \"{raw}\"（可选：off, minimal, low, medium, high, xhigh, max）"
        )),
    }
}

fn parse_cli_args_into<I>(args: I, out: &mut CliArgs) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter().peekable();
    let mut after_double_dash = false;
    let mut first_error = None;

    // 输出格式可能排在无效参数之后，仍须按调用方选择的格式报告首个错误。
    while let Err(err) = parse_cli_args(&mut args, out, &mut after_double_dash) {
        first_error.get_or_insert(err);
    }
    first_error.map_or(Ok(()), Err)
}

fn parse_cli_args<I: Iterator<Item = String>>(
    args: &mut std::iter::Peekable<I>,
    out: &mut CliArgs,
    after_double_dash: &mut bool,
) -> Result<(), String> {
    while let Some(raw_arg) = args.next() {
        if !*after_double_dash && raw_arg == "--" {
            *after_double_dash = true;
            continue;
        }
        if *after_double_dash || !raw_arg.starts_with('-') || raw_arg == "-" {
            if !*after_double_dash
                && !out.explicit_exec
                && out.positional_prompt.is_none()
                && raw_arg == "exec"
            {
                out.explicit_exec = true;
                continue;
            }
            if out.explicit_exec || *after_double_dash {
                if out.positional_prompt.is_some() {
                    return Err(format!("存在多余的位置参数：{raw_arg}"));
                }
                out.positional_prompt = Some(raw_arg);
                continue;
            }
            if out.prompt.is_some() || out.prompt_file.is_some() {
                return Err(format!(
                    "多余的位置参数（已指定 -p 或 --prompt-file）：{raw_arg}"
                ));
            }
            return Err(format!(
                "未知参数：{raw_arg}（运行 `mizar --help` 查看可用参数）"
            ));
        }

        let (flag, inline) = split_flag_value(&raw_arg);
        match flag {
            "--workspace" | "-C" => {
                let val = take_value(flag, inline, args)?;
                out.workspace = Some(PathBuf::from(val));
            }
            "--minimal" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.preset = Some(CodingToolsPreset::MinimalCoding);
            }
            "--read-only" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.preset = Some(CodingToolsPreset::ReadOnly);
            }
            "--tools" | "-t" => {
                let val = take_value(flag, inline, args)?;
                let list: Vec<String> = val
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
                    .collect();
                out.include_tools = Some(list);
            }
            "--exclude-tools" | "-xt" => {
                let val = take_value(flag, inline, args)?;
                out.exclude_tools.extend(
                    val.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToString::to_string),
                );
            }
            "--inline" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.inline = true;
            }
            "--approve" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.approve = true;
            }
            "--no-persist" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.no_persist = true;
            }
            "--no-instructions" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.no_instructions = true;
            }
            "--continue" | "-c" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.resume_session = Some(ResumeTarget::LatestInWorkspace);
            }
            "--resume" | "-r" => {
                if let Some(val) = inline {
                    let trimmed = val.trim();
                    out.resume_session = if trimmed.is_empty() {
                        Some(ResumeTarget::Picker)
                    } else {
                        Some(ResumeTarget::Id(trimmed.to_string()))
                    };
                } else if args.peek().is_some_and(|next| !next.starts_with('-')) {
                    let val = args.next().unwrap();
                    let trimmed = val.trim();
                    out.resume_session = if trimmed.is_empty() {
                        Some(ResumeTarget::Picker)
                    } else {
                        Some(ResumeTarget::Id(trimmed.to_string()))
                    };
                } else {
                    out.resume_session = Some(ResumeTarget::Picker);
                }
            }
            "--theme" => {
                let val = take_value(flag, inline, args)?;
                out.theme = ThemePreset::parse(&val);
            }
            "--isolated" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.isolated = true;
            }
            "--prompt" | "-p" => {
                if out.prompt.is_some() {
                    return Err("重复指定了 -p / --prompt 参数".to_string());
                }
                let val = take_value(flag, inline, args)?;
                out.prompt = Some(val);
            }
            "--prompt-file" => {
                if out.prompt_file.is_some() {
                    return Err("重复指定了 --prompt-file 参数".to_string());
                }
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err("--prompt-file 路径不能为空".to_string());
                }
                out.prompt_file = Some(PathBuf::from(val));
            }
            "--provider" => {
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err("--provider 不能为空".to_string());
                }
                out.provider = Some(val);
            }
            "--model" => {
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err("--model 不能为空".to_string());
                }
                out.model = Some(val);
            }
            "--protocol" | "--api-format" => {
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err(format!("{flag} 不能为空"));
                }
                out.protocol = Some(val);
            }
            "--base-url" => {
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err("--base-url 不能为空".to_string());
                }
                out.base_url = Some(val);
            }
            "--reasoning-effort" => {
                let val = take_value(flag, inline, args)?;
                out.reasoning_effort = Some(parse_cli_reasoning_effort(&val)?);
            }
            "--max-steps" => {
                let val = take_value(flag, inline, args)?;
                let parsed = val
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| format!("无效的 --max-steps 数值：{val}"))?;
                if parsed == 0 {
                    return Err("--max-steps 必须大于等于 1".to_string());
                }
                out.max_steps = Some(parsed);
            }
            "--max-tokens" => {
                let val = take_value(flag, inline, args)?;
                let parsed = val
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| format!("无效的 --max-tokens 数值：{val}"))?;
                if parsed == 0 {
                    return Err("--max-tokens 必须大于等于 1".to_string());
                }
                out.max_tokens = Some(parsed);
            }
            "--context-window" => {
                if inline.is_none() && args.peek().is_some_and(|next| next.starts_with('-')) {
                    return Err(format!("{flag} 需要正整数参数"));
                }
                let val = take_value(flag, inline, args)?;
                out.context_window = Some(parse_context_window(flag, &val)?);
            }
            "--output-format" => {
                let val = take_value(flag, inline, args)?;
                out.output_format = Some(match val.trim().to_ascii_lowercase().as_str() {
                    "text" => ExecOutputFormat::Text,
                    "json" | "ndjson" => ExecOutputFormat::Json,
                    other => {
                        return Err(format!(
                            "无效的 --output-format \"{other}\"（可选：text, json, ndjson）"
                        ));
                    }
                });
            }
            "--json" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.output_format = Some(ExecOutputFormat::Json);
            }
            "--trace-out" => {
                let val = take_value(flag, inline, args)?;
                if val.trim().is_empty() {
                    return Err("--trace-out 路径不能为空".to_string());
                }
                out.trace_out = Some(PathBuf::from(val));
            }
            "--timeout" => {
                let val = take_value(flag, inline, args)?;
                let parsed = val
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| format!("无效的 --timeout 秒数：{val}"))?;
                if parsed == 0 {
                    return Err("--timeout 必须大于等于 1".to_string());
                }
                out.timeout_secs = Some(parsed);
            }
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.help = true;
            }
            "--version" | "-V" => {
                if inline.is_some() {
                    return Err(format!("参数 {flag} 不接受值"));
                }
                out.version = true;
            }
            other => {
                return Err(format!(
                    "未知参数：{other}（运行 `mizar --help` 查看可用参数）"
                ));
            }
        }
    }
    Ok(())
}

fn default_tokio_worker_threads(parallelism: usize) -> usize {
    parallelism.clamp(2, 4)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(default_tokio_worker_threads(workers))
        .enable_all()
        .build()?;
    runtime.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
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

    let mut cli = CliArgs::default();
    if let Err(err) = parse_cli_args_into(raw_args, &mut cli) {
        if cli.is_exec_mode() {
            report_exec_input_error(cli.output_format.unwrap_or_default(), &err);
        } else {
            eprintln!("error: {err}");
        }
        std::process::exit(1);
    }
    if cli.help {
        println!("{}", format_help_text());
        return Ok(());
    }
    if cli.version {
        println!("mizar {}", current_version());
        return Ok(());
    }

    let explicit_workspace = cli.workspace.clone();
    let workspace = cli
        .workspace
        .clone()
        .or_else(|| env_non_empty("WORKSPACE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));

    if cli.is_exec_mode() {
        let format = cli.output_format.unwrap_or_default();
        if cli.approve {
            report_exec_input_error(
                format,
                "--approve is incompatible with non-interactive exec mode (code=invalid_request)",
            );
            std::process::exit(1);
        }
        if cli.inline {
            report_exec_input_error(
                format,
                "--inline is incompatible with non-interactive exec mode (code=invalid_request)",
            );
            std::process::exit(1);
        }
        let prompt = match resolve_exec_prompt(
            cli.explicit_exec,
            cli.prompt,
            cli.prompt_file,
            cli.positional_prompt,
        ) {
            Ok(p) => p,
            Err(err) => {
                report_exec_input_error(format, &err);
                std::process::exit(1);
            }
        };
        let code = run_exec_cli(ExecConfig {
            prompt,
            workspace,
            provider: cli.provider,
            model: cli.model,
            protocol: cli.protocol,
            base_url: cli.base_url,
            reasoning_effort: cli.reasoning_effort,
            max_steps: cli.max_steps,
            max_tokens: cli.max_tokens,
            context_window: cli.context_window,
            timeout_secs: cli.timeout_secs,
            isolated: cli.isolated,
            no_persist: cli.no_persist,
            no_instructions: cli.no_instructions,
            approve: cli.approve,
            preset: cli.preset,
            include_tools: cli.include_tools,
            exclude_tools: cli.exclude_tools,
            resume_session: cli.resume_session,
            output_format: format,
            trace_out: resolve_trace_out_path(cli.trace_out, env_non_empty("MIZAR_TRACE_OUT")),
        })
        .await;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }

    if cli.has_exec_only_flags() {
        eprintln!(
            "error: headless options (--isolated, --provider, --model, --protocol, --base-url, --reasoning-effort, --max-steps, --max-tokens, --timeout, --output-format, --json, --trace-out) require `exec`, `-p/--prompt`, or `--prompt-file`"
        );
        std::process::exit(1);
    }

    let app = Context::new();
    // 进入终端界面前先静默，防止启动日志先刷到 stdout 再闪屏
    app.logger().set_silent();

    let loaded_cfg = load_mizar_config(&workspace);
    let mut spine_cfg = AgentSpineConfig::coding(workspace.clone());
    if !cli.no_persist {
        spine_cfg.session_db_path = resolve_default_session_db_path();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    let cli_override = CliLlmOverride {
        context_window: cli.context_window,
        ..Default::default()
    };
    let env = match EnvLlmOverride::from_env(&cli_override) {
        Ok(env) => env,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    };
    cli_override.apply_context_window(&env, &mut spine_cfg);
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

    let resolved_llm = loaded_cfg.resolve_llm_router_with_env(env);
    if let Some(ref resolved) = resolved_llm {
        spine_cfg.agent.provider = Some(resolved.active_provider.clone());
        spine_cfg.agent.model = resolved.active_model.clone();
    }

    let mut tui_cfg = TuiConfig::new(workspace);
    tui_cfg.max_steps = spine_cfg.agent.max_steps;
    tui_cfg.session_db_path = spine_cfg.session_db_path.clone();
    tui_cfg.reasoning_effort = spine_cfg.agent.reasoning_effort;
    // 记录强度来源：settings.json 显式指定时才算「配置的默认」，否则留给会话记录覆盖
    tui_cfg.reasoning_effort_from_settings = loaded_cfg.settings.reasoning_effort.is_some();
    // 模型来源：环境变量 MODEL 压过一切；settings.json 的 defaultModel 次之；
    // 两者都没有时 `model` 仍可能是 `models.json` 推导出来的，那不算显式指定。
    tui_cfg.model_from_env = env_non_empty("MODEL").is_some();
    tui_cfg.model_from_settings = loaded_cfg.settings.default_model.is_some();
    tui_cfg.disabled_plugins = loaded_cfg.settings.disabled_plugins.clone();
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
