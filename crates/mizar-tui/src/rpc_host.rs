//! 面向轻量桌面前端（如 Herdr Mayonaka）的无头单会话 RPC 宿主（`mizar rpc`）。
//!
//! 职责：
//! - 镜像交互式 `mizar` TUI 的运行时装配（`load_mizar_config`、`SessionTitleConfig`、
//!   模型路由/回落、`install_tui_emphasis` 与 `Selection` 继承），跳过终端 raw mode 与 `ratatui`；
//! - 在 `~/.mizar/run/` 注册单写者进程元信息（`0600` `p-<pid>.json`）与双通道 [`RpcServer`]；
//! - 监听就绪后若指定 `--ready-file`，以 `0600` 权限原子写入 [`ProcessMeta`] JSON；
//! - 支持 `--idle-exit-secs` 断连空闲自动退出与 `--parent-pid` 宿主存活看护。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use mizar_client::actor::{SessionActor, SessionActorConfig};
use mizar_client::install_silent_tracing;
use mizar_client::run_dir::{
    ProcessMeta, ProcessRegistration, check_session_writable, ensure_run_dir, is_pid_alive,
    resolve_run_dir_path, write_meta_atomic,
};
use mizar_client::session_view::{
    SelectionSources, latest_workspace_selection, resolve_selection, to_ui_selection,
};
use mizar_client::{RpcServer, RpcServerConfig};
use mizar_core::spine::{
    AgentSpineConfig, AgentToolsMode, MizarRuntime, SessionTitleConfig, load_mizar_config,
    resolve_default_session_db_path,
};
use mizar_llm::retry::RetryConfig;
use mizar_session::{SessionCreateOptions, SessionListQuery, SessionOrigin};
use mizar_tui_ui::app::SessionSelection;
use mizar_tui_ui::{ApprovalMode, EmphasisConfig, HostExtensionFlags};
use mizar_types::BoxError;
use thiserror::Error;

use crate::runner::install_tui_emphasis;
use crate::serve::wait_for_shutdown_signal;

/// `mizar rpc` 宿主参数与生命周期错误。
#[derive(Debug, Error)]
pub enum RpcHostError {
    #[error("{0}")]
    InvalidArgs(String),
    #[error("工作区路径无效: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Default, PartialEq, Eq)]
struct RpcCliParsed {
    help: bool,
    workspace: Option<PathBuf>,
    resume_id: Option<String>,
    continue_latest: bool,
    ready_file: Option<PathBuf>,
    idle_exit_secs: u64,
    parent_pid: Option<u32>,
    approve: bool,
    no_persist: bool,
    no_instructions: bool,
}

fn split_flag_value(raw: &str) -> (&str, Option<String>) {
    if raw.starts_with("--")
        && let Some((name, val)) = raw.split_once('=')
    {
        return (name, Some(val.to_string()));
    }
    for prefix in ["-C=", "-r="] {
        if let Some(val) = raw.strip_prefix(prefix) {
            let flag = &prefix[..prefix.len() - 1];
            return (flag, Some(val.to_string()));
        }
    }
    (raw, None)
}

fn take_value<I>(flag: &str, inline: Option<String>, iter: &mut I) -> Result<String, RpcHostError>
where
    I: Iterator<Item = String>,
{
    match inline {
        Some(v) => Ok(v),
        None => iter
            .next()
            .ok_or_else(|| RpcHostError::InvalidArgs(format!("缺少 {flag} 的参数值"))),
    }
}

fn parse_rpc_cli_args(args: &[String]) -> Result<RpcCliParsed, RpcHostError> {
    let mut out = RpcCliParsed::default();
    let mut iter = args.iter().cloned();
    while let Some(raw) = iter.next() {
        let (flag, inline) = split_flag_value(&raw);
        match flag {
            "--help" | "-h" => {
                if inline.is_some() {
                    return Err(RpcHostError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.help = true;
            }
            "--workspace" | "-C" => {
                let val = take_value(flag, inline, &mut iter)?;
                if val.trim().is_empty() {
                    return Err(RpcHostError::InvalidArgs(format!("{flag} 不能为空")));
                }
                out.workspace = Some(PathBuf::from(val));
            }
            "--resume" | "-r" => {
                let val = take_value(flag, inline, &mut iter)?;
                let trimmed = val.trim();
                if trimmed.is_empty() {
                    return Err(RpcHostError::InvalidArgs(format!("{flag} 需要指定会话 ID")));
                }
                out.resume_id = Some(trimmed.to_string());
            }
            "--continue" | "-c" => {
                if inline.is_some() {
                    return Err(RpcHostError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.continue_latest = true;
            }
            "--ready-file" => {
                let val = take_value(flag, inline, &mut iter)?;
                if val.trim().is_empty() {
                    return Err(RpcHostError::InvalidArgs("--ready-file 不能为空".into()));
                }
                out.ready_file = Some(PathBuf::from(val));
            }
            "--idle-exit-secs" => {
                let val = take_value(flag, inline, &mut iter)?;
                out.idle_exit_secs = val.trim().parse::<u64>().map_err(|_| {
                    RpcHostError::InvalidArgs(format!("无效 --idle-exit-secs 秒数: {val}"))
                })?;
            }
            "--parent-pid" => {
                let val = take_value(flag, inline, &mut iter)?;
                let pid = val
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| RpcHostError::InvalidArgs(format!("无效 --parent-pid: {val}")))?;
                if pid == 0 {
                    return Err(RpcHostError::InvalidArgs("--parent-pid 必须大于 0".into()));
                }
                out.parent_pid = Some(pid);
            }
            "--approve" => {
                if inline.is_some() {
                    return Err(RpcHostError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.approve = true;
            }
            "--no-persist" => {
                if inline.is_some() {
                    return Err(RpcHostError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.no_persist = true;
            }
            "--no-instructions" => {
                if inline.is_some() {
                    return Err(RpcHostError::InvalidArgs(format!("参数 {flag} 不接受值")));
                }
                out.no_instructions = true;
            }
            other => {
                return Err(RpcHostError::InvalidArgs(format!(
                    "未知参数：{other}（运行 `mizar rpc --help` 查看可用参数）"
                )));
            }
        }
    }
    Ok(out)
}

fn format_rpc_help_text() -> &'static str {
    "\
mizar rpc — Start a headless single-session JSON-RPC host

Usage:
  mizar rpc [OPTIONS]

Options:
  -C, --workspace <dir>         Workspace directory (defaults to current directory)
  -r, --resume <id>             Resume persisted session ID (falls back to new session if unavailable)
  -c, --continue                Resume the latest persisted session in workspace
      --ready-file <path>       Atomically write 0600 ProcessMeta JSON once RPC is listening
      --idle-exit-secs <secs>   Exit after clients stay disconnected for N seconds (0 disables)
      --parent-pid <pid>        Exit cleanly if the given parent PID exits
      --approve                 Require interactive approval on high-risk tools
      --no-persist              Disable SQLite session persistence
      --no-instructions         Disable AGENTS.md / CLAUDE.md discovery
  -h, --help                    Print help"
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

fn write_ready_file_from_meta_path(meta_path: &Path, ready_path: &Path) -> Result<(), BoxError> {
    if let Some(parent) = ready_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let meta: ProcessMeta = serde_json::from_slice(&std::fs::read(meta_path)?)?;
    write_meta_atomic(ready_path, &meta)?;
    Ok(())
}

/// `mizar rpc` 子命令入口：构建完整单会话内核并托管至信号、父进程退出或空闲超时。
pub async fn run_rpc_host_cli(args: &[String]) -> Result<(), BoxError> {
    let parsed = parse_rpc_cli_args(args)?;
    if parsed.help {
        println!("{}", format_rpc_help_text());
        return Ok(());
    }

    let _tracing_guard = install_silent_tracing();

    let workspace = parsed
        .workspace
        .or_else(|| env_non_empty("WORKSPACE").map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().expect("failed to resolve cwd"));

    let loaded_cfg = load_mizar_config(&workspace);
    let mut spine_cfg = AgentSpineConfig::coding(workspace.clone());
    spine_cfg.session_title = Some(SessionTitleConfig::default());
    if !parsed.no_persist {
        spine_cfg.session_db_path = resolve_default_session_db_path();
    }
    loaded_cfg.apply_to_spine(&mut spine_cfg);
    if parsed.no_persist {
        spine_cfg.session_db_path = None;
    }
    if parsed.no_instructions
        && let AgentToolsMode::Coding {
            ref mut instructions,
            ..
        } = spine_cfg.tools_mode
    {
        instructions.enabled = false;
    }

    let resolved_llm = loaded_cfg.resolve_llm_router();
    if let Some(ref resolved) = resolved_llm {
        spine_cfg.agent.provider = Some(resolved.active_provider.clone());
        spine_cfg.agent.model = resolved.active_model.clone();
    }

    let mut provider = spine_cfg.agent.provider.clone();
    let mut model = spine_cfg.agent.model.clone();
    let mut reasoning_effort = spine_cfg.agent.reasoning_effort;
    let restrict_to_workspace = loaded_cfg.settings.restrict_to_workspace.unwrap_or(false);
    let reasoning_effort_from_settings = loaded_cfg.settings.reasoning_effort.is_some();
    let model_from_env = env_non_empty("MODEL").is_some();
    let model_from_settings = loaded_cfg.settings.default_model.is_some();
    let mut approval_mode = if let Some(ref ui) = loaded_cfg.settings.ui
        && let Some(ref raw_mode) = ui.approval_mode
        && raw_mode.trim().eq_ignore_ascii_case("ask")
    {
        ApprovalMode::Ask
    } else {
        ApprovalMode::Auto
    };
    if parsed.approve {
        approval_mode = ApprovalMode::Ask;
    }

    let disabled_plugins = spine_cfg.disabled_plugins.clone();
    let max_steps = spine_cfg.agent.max_steps;
    let session_db_path = spine_cfg.session_db_path.clone();

    let runtime = MizarRuntime::build(spine_cfg).await?;
    runtime.install_retry(RetryConfig::default());
    if let Some(resolved) = resolved_llm {
        runtime.install_router(resolved.router);
    } else {
        runtime.register_mock_adapter()?;
        provider = Some("mock".to_string());
        model = Some("mock-v1".to_string());
    }

    let mut guards = Vec::new();
    let ext_flags = HostExtensionFlags::from_disabled_plugins(&disabled_plugins);
    if ext_flags.tui_emphasis
        && let Some(guard) = install_tui_emphasis(&runtime, &EmphasisConfig::from_env())
    {
        guards.push(guard);
    }

    let run_dir = resolve_run_dir_path()?;
    ensure_run_dir(&run_dir)?;

    let create_fresh_session = || {
        runtime.sessions.create_with(SessionCreateOptions {
            cwd: Some(workspace.display().to_string()),
            origin: Some(SessionOrigin::User),
            ..Default::default()
        })
    };

    let (active_session, resumed) = if let Some(ref target_id) = parsed.resume_id {
        if let Some(found) = runtime.sessions.get(target_id)
            && check_session_writable(&run_dir, &found.id, std::process::id()).is_ok()
        {
            (found, true)
        } else {
            (create_fresh_session(), false)
        }
    } else if parsed.continue_latest {
        let ws_str = workspace.display().to_string();
        let latest_id = runtime
            .sessions
            .list_headers(&SessionListQuery {
                cwd: Some(ws_str),
                limit: Some(1),
                exclude_subagents: true,
            })
            .ok()
            .and_then(|mut list| list.drain(..).next())
            .map(|h| h.id);
        if let Some(id) = latest_id
            && let Some(found) = runtime.sessions.get(&id)
            && check_session_writable(&run_dir, &found.id, std::process::id()).is_ok()
        {
            (found, true)
        } else {
            (create_fresh_session(), false)
        }
    } else {
        (create_fresh_session(), false)
    };

    let sources = SelectionSources {
        model_from_env,
        model_from_settings,
        reasoning_effort_from_settings,
        startup: SessionSelection {
            provider: provider.clone(),
            model: model.clone(),
            reasoning_effort,
        },
        session: resumed
            .then(|| active_session.latest_selection())
            .flatten()
            .map(to_ui_selection),
        latest_workspace: if resumed {
            None
        } else {
            latest_workspace_selection(&runtime.sessions, &workspace).map(to_ui_selection)
        },
    };
    if let Some(resolved_sel) = resolve_selection(&sources, resumed) {
        provider = resolved_sel.provider;
        model = resolved_sel.model;
        reasoning_effort = resolved_sel.reasoning_effort;
    }

    let mut actor_cfg = SessionActorConfig::new(&workspace);
    actor_cfg.session_db_path = session_db_path;
    actor_cfg.run_dir = Some(run_dir.clone());
    actor_cfg.provider = provider.clone();
    actor_cfg.model = model.clone();
    actor_cfg.reasoning_effort = reasoning_effort;
    actor_cfg.approval_mode = approval_mode;
    actor_cfg.max_steps = max_steps;
    actor_cfg.restrict_to_workspace = restrict_to_workspace;
    actor_cfg.host_extensions = ext_flags;

    let actor = SessionActor::spawn(Arc::clone(&runtime), Arc::clone(&active_session), actor_cfg);
    let meta = ProcessMeta::for_current_process(
        &run_dir,
        &active_session.id,
        workspace.display().to_string(),
        provider,
        model,
    );
    let registration = ProcessRegistration::create(&run_dir, meta)?;
    let rpc_server = RpcServer::start(actor.clone(), registration, RpcServerConfig::default())?;

    if let Some(ref ready_path) = parsed.ready_file {
        write_ready_file_from_meta_path(rpc_server.meta_path(), ready_path)?;
    }

    if let Some(addr) = rpc_server.tcp_addr() {
        println!(
            "mizar rpc listening on tcp://{} (session {})",
            addr, active_session.id
        );
    }

    let shutdown = wait_for_shutdown_signal();
    tokio::pin!(shutdown);
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut idle_since: Option<tokio::time::Instant> = None;

    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => break,
            _ = tick.tick() => {
                if let Some(ppid) = parsed.parent_pid
                    && !is_pid_alive(ppid)
                {
                    break;
                }
                if parsed.idle_exit_secs > 0 {
                    let active_sid = actor.active_session_id();
                    let is_idle = rpc_server.ever_connected()
                        && rpc_server.active_clients() == 0
                        && !runtime.agent_loop.is_session_busy(&active_sid);
                    if is_idle {
                        let start = *idle_since.get_or_insert_with(tokio::time::Instant::now);
                        if start.elapsed() >= Duration::from_secs(parsed.idle_exit_secs) {
                            break;
                        }
                    } else {
                        idle_since = None;
                    }
                }
            }
        }
    }

    rpc_server.shutdown().await;
    actor.shutdown().await;
    for guard in guards.into_iter().rev() {
        drop(guard);
    }
    runtime.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_cli_parses_flags_and_rejects_invalid_values() {
        let parsed = parse_rpc_cli_args(&[
            "-C=/tmp/ws".into(),
            "--resume".into(),
            "sess-1".into(),
            "--ready-file=/tmp/ready.json".into(),
            "--idle-exit-secs".into(),
            "45".into(),
            "--parent-pid".into(),
            "999".into(),
            "--approve".into(),
            "--no-persist".into(),
            "--no-instructions".into(),
        ])
        .unwrap();
        assert_eq!(parsed.workspace, Some(PathBuf::from("/tmp/ws")));
        assert_eq!(parsed.resume_id.as_deref(), Some("sess-1"));
        assert_eq!(parsed.ready_file, Some(PathBuf::from("/tmp/ready.json")));
        assert_eq!(parsed.idle_exit_secs, 45);
        assert_eq!(parsed.parent_pid, Some(999));
        assert!(parsed.approve);
        assert!(parsed.no_persist);
        assert!(parsed.no_instructions);

        assert!(parse_rpc_cli_args(&["--parent-pid".into(), "0".into()]).is_err());
        assert!(parse_rpc_cli_args(&["--resume".into(), "   ".into()]).is_err());
        assert!(parse_rpc_cli_args(&["--unknown".into()]).is_err());
    }
}
