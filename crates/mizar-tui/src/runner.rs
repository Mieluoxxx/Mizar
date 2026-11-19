//! 异步运行循环：将 Cordis / AgentLoop / Trajectory 事件接入终端界面。

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use cordis::{
    BoxFuture, Cleanup, Context, CordisError, DisposableHandle, LogLevel, LogRecord, LogSink, Next,
};
use crossterm::event::{Event as TermEvent, EventStream};
use mizar_agent_loop::{
    AgentLoopContextExt, AgentReasoningDeltaEvent, AgentStatus, AgentStatusEvent,
    AgentStepFirstTokenEvent, AgentStepStartEvent, AgentTextDeltaEvent, AgentToolCallEvent,
    AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent, AgentTurnStartEvent,
    AgentUsageEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_STATUS,
    EVENT_AGENT_STEP_FIRST_TOKEN, EVENT_AGENT_STEP_START, EVENT_AGENT_TEXT_DELTA,
    EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT, EVENT_AGENT_TOOL_UPDATE,
    EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_START, EVENT_AGENT_USAGE, RunOptions, TurnEndReason,
};
use mizar_instructions::AgentInstructions;
use mizar_llm::{ContentBlock, LlmContextExt, ReasoningEffort};
use mizar_mcp::{McpConnectionState, McpManager};
use mizar_session::{
    Session, SessionContextExt, SessionCreateOptions, SessionEvent, SessionListQuery,
    SessionPersistenceContextExt,
};
use mizar_skills::{EVENT_SKILLS_CHANGE, SkillLookupOptions, SkillRegistry};
use mizar_system_prompt::{PromptSection, SystemPromptContextExt};
use mizar_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolExecution, ToolExposure, ToolRuntime,
};
use mizar_tools_coding::bash::execute_bash_process;
use mizar_trajectory::{
    EVENT_TRAJECTORY_ANOMALY, TimelineMode, TrajectoryAnomalyEvent, TrajectoryRuntime,
};
use mizar_tui_ui::app::{
    AppAction, AppState, ApprovalMode, LocalBashOutput, NoticeLevel, SessionReplayItem,
    SessionViewData, UiAgentStatus, UiEvent, UiLogLevel, UiLogRecord, UiTurnEndReason, short_id,
    truncate_display,
};
use mizar_tui_ui::emphasis::{EmphasisConfig, reply_guidance_prompt};
use mizar_tui_ui::overlay::{ApprovalDecision, SessionTreeEntry, SkillPickerItem};
use mizar_tui_ui::side_panel::{FileTouchRow, InstructionSummaryRow, McpSummaryRow, ToolSummaryRow};
use mizar_tui_ui::terminal::{
    ProgramStatusState, ScreenMode, TerminalGuard, apply_terminal_modes, emit_program_status,
    resume_after_external_editor, set_mouse_capture_enabled, suspend_for_external_editor,
    write_begin_sync, write_end_sync,
};
use mizar_tui_ui::theme::ThemePreset;
use mizar_tui_ui::ui::{
    build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback, render,
};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::{Terminal, TerminalOptions, Viewport};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeTarget {
    LatestInWorkspace,
    Id(String),
}

/// TUI 退出时返回给宿主进程的可恢复会话摘要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiExitSummary {
    pub session_id: String,
    pub persisted: bool,
}

fn shell_quote_arg(raw: &str) -> String {
    if !raw.is_empty()
        && raw.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'_' | b'-' | b'.' | b'/' | b'~' | b':' | b'@')
        })
    {
        return raw.to_string();
    }
    format!("'{}'", raw.replace('\'', "'\\''"))
}

fn resolve_resume_executable(current_exe: Option<&Path>, cwd: Option<&Path>) -> String {
    if let Some(exe) = current_exe {
        let normalized = exe.to_string_lossy().replace('\\', "/");
        let is_cargo_target =
            normalized.contains("/target/debug/") || normalized.contains("/target/release/");
        if is_cargo_target {
            if let Some(cwd_path) = cwd
                && let Ok(rel) = exe.strip_prefix(cwd_path)
            {
                let rel_str = rel.to_string_lossy();
                let dot_rel = if rel_str.starts_with('.') {
                    rel_str.into_owned()
                } else {
                    format!("./{rel_str}")
                };
                return shell_quote_arg(&dot_rel);
            }
            return shell_quote_arg(&exe.display().to_string());
        }
        if let Some(name) = exe.file_name().and_then(|s| s.to_str())
            && !name.trim().is_empty()
        {
            return shell_quote_arg(name);
        }
    }
    "mizar".to_string()
}

/// 根据退出会话信息构造可直接复制执行的 `--resume` 恢复命令；未开启持久化或空会话 ID 时返回 `None`。
pub fn format_resume_command(
    summary: &TuiExitSummary,
    explicit_workspace: Option<&Path>,
    current_exe: Option<&Path>,
    cwd: Option<&Path>,
) -> Option<String> {
    if !summary.persisted || summary.session_id.trim().is_empty() {
        return None;
    }
    let mut parts = vec![resolve_resume_executable(current_exe, cwd)];
    if let Some(ws) = explicit_workspace {
        parts.push("-C".to_string());
        parts.push(shell_quote_arg(&ws.display().to_string()));
    }
    parts.push("--resume".to_string());
    parts.push(shell_quote_arg(&summary.session_id));
    Some(parts.join(" "))
}

#[derive(Debug, Clone)]
pub struct TuiConfig {
    pub workspace: PathBuf,
    pub max_steps: Option<usize>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub approval_mode: ApprovalMode,
    pub theme_preset: Option<ThemePreset>,
    pub screen_mode: ScreenMode,
    pub resume_session: Option<ResumeTarget>,
    pub restrict_to_workspace: bool,
    pub startup_notices: Vec<(NoticeLevel, String)>,
}

impl TuiConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            max_steps: None,
            provider: None,
            model: None,
            reasoning_effort: None,
            approval_mode: ApprovalMode::Auto,
            theme_preset: None,
            screen_mode: ScreenMode::Fullscreen,
            resume_session: None,
            restrict_to_workspace: false,
            startup_notices: Vec::new(),
        }
    }
}

struct LoggerRestoreGuard {
    ctx: Context,
}

impl Drop for LoggerRestoreGuard {
    fn drop(&mut self) {
        self.ctx.logger().set_sink(LogSink::Stdio);
    }
}

/// 将交互式 TUI 回复视觉语法提示段挂载到系统提示词注册表。
pub fn mount_reply_guidance_section(
    ctx: &Context,
    emphasis: &EmphasisConfig,
) -> Result<Option<Cleanup>, CordisError> {
    let Ok(sp) = ctx.system_prompt() else {
        return Ok(None);
    };
    if !(emphasis.enabled && emphasis.guidance) {
        return Ok(None);
    }
    let prompt = reply_guidance_prompt();
    Ok(Some(sp.section(PromptSection::new(
        "tui_reply_guidance",
        95,
        move || Some(prompt.clone()),
    ))))
}

fn to_ui_status(status: AgentStatus) -> UiAgentStatus {
    match status {
        AgentStatus::Busy => UiAgentStatus::Busy,
        AgentStatus::Idle => UiAgentStatus::Idle,
    }
}

fn to_ui_turn_end_reason(reason: TurnEndReason) -> UiTurnEndReason {
    match reason {
        TurnEndReason::Completed => UiTurnEndReason::Completed,
        TurnEndReason::MaxSteps => UiTurnEndReason::MaxSteps,
        TurnEndReason::Cancelled => UiTurnEndReason::Cancelled,
        TurnEndReason::Error { message } => UiTurnEndReason::Error { message },
    }
}

pub fn to_ui_log_record(record: &LogRecord) -> UiLogRecord {
    let level = match record.level {
        LogLevel::Silent => UiLogLevel::Silent,
        LogLevel::Error => UiLogLevel::Error,
        LogLevel::Warn => UiLogLevel::Warn,
        LogLevel::Info => UiLogLevel::Info,
        LogLevel::Debug => UiLogLevel::Debug,
    };
    UiLogRecord {
        level,
        name: record.name.clone(),
        message: record.message.clone(),
    }
}

/// 将持久化会话日志投影为纯 UI 可重放的 `SessionViewData`（含 `!` 与 `!!` 本地命令记录）。
pub fn project_session_view(session: &Session, finish_notice: impl Into<String>) -> SessionViewData {
    let header = session.header();
    let items = session
        .all_entries()
        .into_iter()
        .filter_map(|entry| match entry.event {
            SessionEvent::User { text } => Some(SessionReplayItem::User { text }),
            SessionEvent::ContextNotice { source, text } => {
                Some(SessionReplayItem::ContextNotice { source, text })
            }
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                ..
            } => Some(SessionReplayItem::Compaction {
                summary,
                first_kept_seq,
            }),
            SessionEvent::Assistant { blocks, .. } => {
                Some(SessionReplayItem::Assistant { blocks })
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => Some(SessionReplayItem::ToolResult {
                tool_call_id,
                content,
                is_error,
            }),
            SessionEvent::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            } => Some(SessionReplayItem::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            }),
            SessionEvent::Trace { .. } => None,
        })
        .collect();
    SessionViewData {
        session_id: session.id.clone(),
        session_title: header.title,
        items,
        finish_notice: finish_notice.into(),
    }
}

/// 生成恢复会话的 UI 重放视图。
pub fn project_resume_view(session: &Session) -> SessionViewData {
    let header = session.header();
    let title_suffix = header
        .title
        .as_deref()
        .map(|t| format!(" · {t}"))
        .unwrap_or_default();
    let finish_notice = format!(
        "已恢复历史会话 `{}`{}（共 {} 条表面事件）",
        short_id(&session.id),
        title_suffix,
        header.surface_count,
    );
    project_session_view(session, finish_notice)
}

/// 生成分叉会话的 UI 重放视图。
pub fn project_fork_view(session: &Session, from_seq: u64) -> SessionViewData {
    let finish_notice = format!(
        "🌿 已从事件 #{from_seq} 分叉至新会话 `{}`",
        short_id(&session.id),
    );
    project_session_view(session, finish_notice)
}

/// 将持久化会话日志投影为 `/tree` 覆盖层条目列表（保留 `!` 与 `!!` 归属标识）。
pub fn project_session_tree(session: &Session) -> Vec<SessionTreeEntry> {
    session
        .all_entries()
        .into_iter()
        .filter_map(|entry| match entry.event {
            SessionEvent::User { text } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: "USER".to_string(),
                summary: truncate_display(&text.replace('\n', " "), 72),
                depth: 0,
                can_fork: true,
            }),
            SessionEvent::ContextNotice { source, text } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: "CTX ".to_string(),
                summary: format!("[{source}] {}", truncate_display(&text.replace('\n', " "), 60)),
                depth: 1,
                can_fork: false,
            }),
            SessionEvent::Assistant { blocks, .. } => {
                let mut parts = Vec::new();
                for block in &blocks {
                    match block {
                        ContentBlock::Text { text } if !text.trim().is_empty() => {
                            parts.push(text.trim().replace('\n', " "));
                        }
                        ContentBlock::ToolCall { name, .. } => {
                            parts.push(format!("[call:{name}]"));
                        }
                        ContentBlock::Reasoning { .. } | ContentBlock::Text { .. } => {}
                    }
                }
                let summary = if parts.is_empty() {
                    "(empty assistant turn)".to_string()
                } else {
                    truncate_display(&parts.join(" "), 72)
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "ASST".to_string(),
                    summary,
                    depth: 1,
                    can_fork: true,
                })
            }
            SessionEvent::ToolResult {
                name,
                content,
                is_error,
                ..
            } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: if is_error { "ERR " } else { "TOOL" }.to_string(),
                summary: format!("{name}: {}", truncate_display(&content.replace('\n', " "), 60)),
                depth: 1,
                can_fork: true,
            }),
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                ..
            } => Some(SessionTreeEntry {
                seq: entry.seq,
                kind_tag: "COMP".to_string(),
                summary: format!(
                    "[keep>={first_kept_seq}] {}",
                    truncate_display(&summary.replace('\n', " "), 56)
                ),
                depth: 0,
                can_fork: false,
            }),
            SessionEvent::BashExecution {
                command,
                exit_code,
                cancelled,
                exclude_from_context,
                ..
            } => {
                let kind_tag = if exclude_from_context {
                    "BASH!!"
                } else {
                    "BASH! "
                };
                let prefix = if exclude_from_context { "!!" } else { "!" };
                let status_hint = if cancelled {
                    "[取消] "
                } else if let Some(code) = exit_code
                    && code != 0
                {
                    "[失败] "
                } else {
                    ""
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: kind_tag.to_string(),
                    summary: format!(
                        "{prefix} {status_hint}{}",
                        truncate_display(&command.replace('\n', " "), 60)
                    ),
                    depth: 0,
                    can_fork: true,
                })
            }
            SessionEvent::Trace { .. } => None,
        })
        .collect()
}

/// 订阅当前会话相关的 Agent 与轨迹事件并投递到 `tx`。
pub fn subscribe_session_events(
    ctx: &Context,
    session_id_rx: Arc<std::sync::RwLock<String>>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<Vec<DisposableHandle>, CordisError> {
    let mut handles = Vec::new();
    let is_current = move |sid: &str| *session_id_rx.read().unwrap() == sid;

    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STATUS, move |ev: &AgentStatusEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::Status(to_ui_status(ev.status)));
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::TurnStart { turn: ev.turn });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::TurnEnd {
                    reason: to_ui_turn_end_reason(ev.reason.clone()),
                });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STEP_START, move |ev: &AgentStepStartEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::StepStart { ts_ms: ev.ts_ms });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_STEP_FIRST_TOKEN, move |ev: &AgentStepFirstTokenEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::FirstToken { ts_ms: ev.ts_ms });
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TEXT_DELTA, move |ev: &AgentTextDeltaEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::TextDelta(ev.text.clone()));
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_REASONING_DELTA, move |ev: &AgentReasoningDeltaEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::ReasoningDelta(ev.text.clone()));
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::ToolCall(ev.call.clone()));
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TOOL_UPDATE, move |ev: &AgentToolUpdateEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::ToolUpdate {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    preview: ev.preview.clone(),
                });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TOOL_RESULT, move |ev: &AgentToolResultEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::ToolResult {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    content: ev.content.clone(),
                    is_error: ev.is_error,
                    blocked: ev.blocked,
                    details: ev.details.clone(),
                });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_USAGE, move |ev: &AgentUsageEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::Usage(ev.usage));
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current;
        handles.push(
            ctx.on(EVENT_TRAJECTORY_ANOMALY, move |ev: &TrajectoryAnomalyEvent| {
                if check(&ev.session_id) {
                    let _ = tx.send(UiEvent::Anomaly(ev.finding.summary.clone()));
                }
            })?,
        );
    }

    Ok(handles)
}

/// 等待 `agentLoop`、`sessions`、`llm` 核心服务以及所有已挂载插件就绪（消除异步挂载竞态）。
pub async fn wait_for_services(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.get::<mizar_agent_loop::AgentLoop>("agentLoop").is_none()
        || ctx.get::<mizar_session::SessionStore>("sessions").is_none()
        || ctx.get::<mizar_llm::LlmRuntime>("llm").is_none()
    {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let tx = Arc::new(std::sync::Mutex::new(Some(tx)));
        let probe = ctx.try_inject(["agentLoop", "sessions", "llm"], move |_| {
            let tx = Arc::clone(&tx);
            async move {
                if let Some(sender) = tx.lock().unwrap().take() {
                    let _ = sender.send(());
                }
                Ok::<_, CordisError>(())
            }
        })?;
        let wait_res = tokio::time::timeout(Duration::from_secs(5), rx).await;
        probe.dispose().await;
        wait_res.map_err(|_| "等待核心服务（agentLoop / sessions / llm）超时")??;
    }

    for rt in ctx.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }
    Ok(())
}

/// 在已装配 `AgentSpinePlugin` 与模型适配器的 `Context` 上启动全屏 TUI。
pub async fn run_tui(ctx: Context, config: TuiConfig) -> Result<(), Box<dyn std::error::Error>> {
    run_tui_with_summary(ctx, config).await.map(|_| ())
}

/// 在已装配 `AgentSpinePlugin` 与模型适配器的 `Context` 上启动 TUI，并在恢复终端后返回当前会话摘要。
pub async fn run_tui_with_summary(
    ctx: Context,
    config: TuiConfig,
) -> Result<TuiExitSummary, Box<dyn std::error::Error>> {
    wait_for_services(&ctx).await?;
    let sessions = ctx.sessions()?;
    let mut resume_notice: Option<(NoticeLevel, String)> = None;
    let (initial_session, resumed) = match &config.resume_session {
        Some(ResumeTarget::Id(target_id)) => {
            if let Some(found) = sessions.get(target_id) {
                (found, true)
            } else {
                resume_notice = Some((
                    NoticeLevel::Warn,
                    format!("未找到历史会话 `{target_id}`，已开启新会话"),
                ));
                (sessions.create(None), false)
            }
        }
        Some(ResumeTarget::LatestInWorkspace) => {
            let ws_str = config.workspace.display().to_string();
            let latest_id = sessions
                .list_headers(&SessionListQuery {
                    cwd: Some(ws_str),
                    limit: Some(1),
                })
                .ok()
                .and_then(|mut list| list.drain(..).next())
                .map(|h| h.id);
            if let Some(id) = latest_id
                && let Some(found) = sessions.get(&id)
            {
                (found, true)
            } else {
                resume_notice = Some((
                    NoticeLevel::Info,
                    "当前工作区暂无可恢复的历史会话，已开启新会话".to_string(),
                ));
                (sessions.create(None), false)
            }
        }
        None => (sessions.create(None), false),
    };
    let providers = ctx
        .llm()
        .map(|l| l.provider_infos())
        .unwrap_or_default();

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();

    // 重定向 Cordis 日志，确保任何插件输出都不会污染终端 stdout
    let log_tx = ui_tx.clone();
    ctx.logger().set_handler(move |record| {
        if record.level >= LogLevel::Warn {
            let _ = log_tx.send(UiEvent::Log(to_ui_log_record(record)));
        }
    });
    let _logger_restore = LoggerRestoreGuard { ctx: ctx.clone() };

    let mut active_session = Arc::clone(&initial_session);
    let active_sid = Arc::new(std::sync::RwLock::new(active_session.id.clone()));
    let handles = subscribe_session_events(&ctx, Arc::clone(&active_sid), ui_tx.clone())?;

    let mut app = AppState::new(
        config.workspace.clone(),
        active_session.id.clone(),
        providers,
        config.provider.clone(),
        config.model.clone(),
        config.reasoning_effort,
    );
    app.session_title = active_session.header().title;
    if resumed {
        app.restore_session_view(project_resume_view(&active_session));
    }
    if let Some((lvl, msg)) = resume_notice {
        app.push_notice(lvl, msg);
    }
    app.user_skill_names = collect_user_skill_names(&ctx, &config.workspace).await;
    if let Some(reg) = ctx.get::<SkillRegistry>("skills") {
        let tx_skills = ui_tx.clone();
        let ws_skills = config.workspace.clone();
        let reg_skills = Arc::clone(&reg);
        let _ = ctx.on_signal(EVENT_SKILLS_CHANGE, move || {
            let tx = tx_skills.clone();
            let ws = ws_skills.clone();
            let reg = Arc::clone(&reg_skills);
            tokio::spawn(async move {
                let list = reg
                    .list(SkillLookupOptions {
                        cwd: Some(ws),
                        cancel: CancellationToken::new(),
                    })
                    .await;
                let names: Vec<String> = list
                    .into_iter()
                    .filter(|s| s.is_user_invocable())
                    .map(|s| s.name)
                    .collect();
                let _ = tx.send(UiEvent::UserSkillsUpdated(names));
            });
        });
    }
    for (level, msg) in &config.startup_notices {
        app.push_notice(*level, msg.clone());
    }

    // 挂载交互式 TUI 回复视觉语法提示段（退出时卸载）
    let reply_guidance = mount_reply_guidance_section(&ctx, &app.emphasis)?;

    // 共享审批快照（供 tools/pre-execute 异步瀑布流无锁读取当前审批策略）
    let approval_snap = Arc::new(RwLock::new(ApprovalSnapshot {
        mode: app.approval_mode,
        always_allowed: app.always_allowed_tools.clone(),
    }));
    let pre_exec_handle = register_approval_hook(&ctx, ui_tx.clone(), Arc::clone(&approval_snap))?;
    let mention_hook_handle = crate::mentions::register_mention_hook(
        &ctx,
        config.workspace.clone(),
        config.restrict_to_workspace,
        ui_tx.clone(),
    )?;

    app.approval_mode = config.approval_mode;
    app.screen_mode = config.screen_mode;
    app.mouse_capture_enabled = config.screen_mode == ScreenMode::Fullscreen;
    if let Some(preset) = config.theme_preset {
        app.set_theme_preset(preset);
    }
    populate_side_panel_data(&ctx, &mut app).await;

    // 进入终端 raw mode（Fullscreen / Inline 双模支持），并注入 OSC 10/11/4 + DA1 探测到的终端自适应色板
    let _terminal_guard =
        TerminalGuard::enter_with_mode(app.screen_mode, app.mouse_capture_enabled)?;
    app.apply_terminal_colors(_terminal_guard.probed_colors.clone());

    let mut terminal = create_terminal(app.screen_mode)?;

    let mut term_stream = EventStream::new();
    let mut spinner_tick = tokio::time::interval(Duration::from_millis(90));
    spinner_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut reveal_tick = tokio::time::interval(Duration::from_millis(33));
    reveal_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut active_cancel: Option<CancellationToken> = None;
    let mut pending_steer: Option<String> = None;
    let mut user_cancelled = false;

    while !app.should_quit {
        if app.screen_mode == ScreenMode::Inline && !app.is_busy && !app.is_bash_running {
            flush_inline_scrollback(&mut terminal, &mut app);
        }
        write_begin_sync();
        let draw_res = terminal.draw(|frame| render(frame, &mut app));
        write_end_sync();
        draw_res?;

        let action = tokio::select! {
            maybe_term = term_stream.next() => {
                match maybe_term {
                    Some(Ok(TermEvent::Key(key))) => app.handle_key(key),
                    Some(Ok(TermEvent::Mouse(mouse))) => app.handle_mouse(mouse),
                    Some(Ok(TermEvent::Paste(data))) => {
                        app.handle_paste(&data);
                        AppAction::None
                    }
                    Some(Ok(TermEvent::Resize(_, _))) => AppAction::None,
                    Some(Ok(TermEvent::FocusGained)) => {
                        app.terminal_focused = true;
                        AppAction::None
                    }
                    Some(Ok(TermEvent::FocusLost)) => {
                        app.terminal_focused = false;
                        AppAction::None
                    }
                    Some(Err(err)) => {
                        app.push_notice(NoticeLevel::Error, format!("终端事件读取失败：{err}"));
                        AppAction::None
                    }
                    None => AppAction::Quit,
                }
            }
            maybe_ui = ui_rx.recv() => {
                match maybe_ui {
                    Some(ev) => {
                        let mut next_action = process_ui_event(
                            &mut app,
                            &mut active_cancel,
                            &mut pending_steer,
                            &mut user_cancelled,
                            ev,
                        );
                        // 把缓冲区积压的 delta 一次性排空，降低高并发流式输出时的重绘次数
                        if next_action == AppAction::None {
                            while let Ok(more) = ui_rx.try_recv() {
                                let act = process_ui_event(
                                    &mut app,
                                    &mut active_cancel,
                                    &mut pending_steer,
                                    &mut user_cancelled,
                                    more,
                                );
                                if act != AppAction::None {
                                    next_action = act;
                                    break;
                                }
                            }
                        }
                        next_action
                    }
                    None => AppAction::Quit,
                }
            }
            _ = reveal_tick.tick(), if app.terminal_focused && (app.reveal.has_pending() || app.effort_ignition_at.is_some()) => {
                app.reveal.tick();
                AppAction::None
            }
            _ = spinner_tick.tick(), if app.is_busy || app.is_bash_running || (!app.terminal_focused && (app.reveal.has_pending() || app.effort_ignition_at.is_some())) => {
                if app.is_busy || app.is_bash_running {
                    app.spinner_tick = app.spinner_tick.wrapping_add(1);
                }
                if !app.terminal_focused {
                    app.reveal.tick();
                }
                AppAction::None
            }
        };

        handle_action(
            &ctx,
            &config,
            &mut app,
            &mut active_session,
            &mut terminal,
            &active_sid,
            &ui_tx,
            &mut active_cancel,
            &mut pending_steer,
            &mut user_cancelled,
            action,
        )
        .await;

        if let Ok(mut guard) = approval_snap.write() {
            if guard.mode != app.approval_mode {
                guard.mode = app.approval_mode;
            }
            if guard.always_allowed != app.always_allowed_tools {
                guard.always_allowed = app.always_allowed_tools.clone();
            }
        }
    }

    emit_program_status(ProgramStatusState::Idle, None, None);
    if let Some(cancel) = active_cancel.take() {
        cancel.cancel();
    }
    drop(mention_hook_handle.dispose());
    drop(pre_exec_handle.dispose());
    if let Some(cleanup) = reply_guidance {
        match cleanup {
            cordis::Cleanup::Sync(f) => f(),
            cordis::Cleanup::Async(f) => f().await,
        }
    }
    for handle in handles {
        drop(handle.dispose());
    }
    Ok(TuiExitSummary {
        session_id: app.session_id.clone(),
        persisted: ctx.session_persistence().is_some(),
    })
}

#[derive(Default)]
struct ApprovalSnapshot {
    mode: ApprovalMode,
    always_allowed: HashSet<String>,
}

fn register_approval_hook(
    ctx: &Context,
    ui_tx: mpsc::UnboundedSender<UiEvent>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
) -> Result<DisposableHandle, CordisError> {
    let ctx_lookup = ctx.clone();
    ctx.on_waterfall(
        EVENT_TOOLS_PRE_EXECUTE,
        move |exec: ToolExecution, next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let ui_tx = ui_tx.clone();
            let snap_lock = Arc::clone(&approval_snap);
            let ctx_inner = ctx_lookup.clone();
            Box::pin(async move {
                let needs_prompt = {
                    let guard = snap_lock.read().ok();
                    match guard {
                        Some(g)
                            if g.mode == ApprovalMode::Ask
                                && !g.always_allowed.contains(&exec.name) =>
                        {
                            tool_needs_approval(&ctx_inner, &exec.name)
                        }
                        _ => false,
                    }
                };

                if !needs_prompt {
                    return next.call(exec).await;
                }

                let args_preview = serde_json::to_string_pretty(&exec.args)
                    .unwrap_or_else(|_| exec.args.to_string());
                let args_preview = truncate_display(&args_preview.replace('\n', " "), 400);
                let (tx, rx) = futures::channel::oneshot::channel();
                if ui_tx
                    .send(UiEvent::ApprovalRequested {
                        tool_name: exec.name.clone(),
                        args_preview,
                        responder: Arc::new(Mutex::new(Some(tx))),
                    })
                    .is_err()
                {
                    return next.call(exec).await;
                }

                match rx.await {
                    Ok(ApprovalDecision::AllowOnce) | Ok(ApprovalDecision::AllowAlwaysForTool) => {
                        next.call(exec).await
                    }
                    Ok(ApprovalDecision::DenyWithReason(reason)) => {
                        PreExecuteDecision::deny(reason)
                    }
                    Ok(ApprovalDecision::Deny) | Err(_) => {
                        PreExecuteDecision::deny("用户拒绝执行该工具")
                    }
                }
            })
        },
    )
}

fn tool_needs_approval(ctx: &Context, tool_name: &str) -> bool {
    if let Some(rt) = ctx.get::<ToolRuntime>("tools")
        && let Some(info) = rt.list_all().into_iter().find(|i| i.name == tool_name)
    {
        return !info.annotations.read_only;
    }
    !matches!(
        tool_name,
        "read" | "grep" | "find" | "ls" | "tool_search" | "mcp_list_resources" | "mcp_read_resource"
    )
}

fn process_ui_event(
    app: &mut AppState,
    active_cancel: &mut Option<CancellationToken>,
    pending_steer: &mut Option<String>,
    user_cancelled: &mut bool,
    event: UiEvent,
) -> AppAction {
    if matches!(event, UiEvent::LocalBashFinished { .. }) {
        *active_cancel = None;
        *user_cancelled = false;
        app.apply_ui_event(event);
        return AppAction::None;
    }

    if let UiEvent::RunFinished(ref res) = event {
        let was_aborted =
            *user_cancelled || res.as_ref().is_err_and(|e| e.contains("request aborted"));
        let is_ok = res.is_ok();
        *active_cancel = None;
        *user_cancelled = false;
        app.apply_ui_event(event);

        if let Some(steer) = pending_steer.take() {
            app.push_user_message(&steer);
            return AppAction::SubmitPrompt(steer);
        }
        if !was_aborted && is_ok && !app.queued_prompts.is_empty() {
            let next_prompt = app.queued_prompts.remove(0);
            app.push_user_message(&next_prompt);
            return AppAction::SubmitPrompt(next_prompt);
        }
        if !was_aborted && !is_ok && !app.queued_prompts.is_empty() {
            app.push_notice(
                NoticeLevel::Warn,
                format!("上一轮未成功，已保留 {} 条排队消息", app.queued_prompts.len()),
            );
        }
        return AppAction::RefreshSidePanelData;
    }

    app.apply_ui_event(event);
    AppAction::None
}

fn spawn_agent_run(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    active_cancel: &mut Option<CancellationToken>,
    prompt: String,
) {
    let Ok(agent_loop) = ctx.agent_loop() else {
        app.push_notice(NoticeLevel::Error, "agentLoop 服务未就绪");
        return;
    };
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    app.is_busy = true;

    let session: Arc<Session> = Arc::clone(active_session);
    let run_opts = RunOptions {
        provider: app.provider.clone(),
        model: app.model.clone(),
        max_steps: config.max_steps,
        max_tokens: None,
        temperature: None,
        reasoning_effort: app.reasoning_effort,
    };
    let tx = ui_tx.clone();
    tokio::spawn(async move {
        let res = agent_loop
            .run_with_options(session, prompt, cancel, run_opts)
            .await
            .map_err(|e| e.to_string());
        let _ = tx.send(UiEvent::RunFinished(res));
    });
}

static LOCAL_BASH_SEQ: AtomicU64 = AtomicU64::new(1);

fn spawn_local_bash_run(
    app: &mut AppState,
    active_session: &Arc<Session>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    active_cancel: &mut Option<CancellationToken>,
    command: String,
    exclude_from_context: bool,
) {
    if app.is_bash_running {
        app.set_flash("已有 Bash 命令正在运行，请先按 Esc 取消");
        return;
    }
    if app.is_busy {
        app.set_flash("模型正在生成中，请等待回合结束或按 Esc 取消后再运行 Bash 命令");
        return;
    }

    let seq = LOCAL_BASH_SEQ.fetch_add(1, Ordering::Relaxed);
    let call_id = format!("local-bash-{seq}");
    let cancel = CancellationToken::new();
    *active_cancel = Some(cancel.clone());
    app.start_local_bash(&call_id, &command, exclude_from_context);

    let session = Arc::clone(active_session);
    let cwd = app.workspace.clone();
    let tx = ui_tx.clone();
    let update_cid = call_id.clone();
    let update_tx = tx.clone();

    tokio::spawn(async move {
        let res = execute_bash_process(&command, &cwd, None, cancel, move |preview| {
            let _ = update_tx.send(UiEvent::ToolUpdate {
                call_id: update_cid.clone(),
                name: "bash".to_string(),
                preview: preview.to_string(),
            });
        })
        .await;

        let (session_out, exit_code, cancelled) = match &res {
            Ok(out) => (out.session_output(), out.exit_code, out.cancelled),
            Err(err) => (err.clone(), Some(1), false),
        };
        session.append(SessionEvent::BashExecution {
            command: command.clone(),
            output: session_out.clone(),
            exit_code,
            cancelled,
            exclude_from_context,
        });

        let payload = res.map(|out| LocalBashOutput {
            display: out.display,
            session_output: session_out,
            exit_code: out.exit_code,
            cancelled: out.cancelled,
            timed_out: out.timed_out,
        });

        let _ = tx.send(UiEvent::LocalBashFinished {
            call_id,
            command,
            exclude_from_context,
            output: payload,
        });
    });
}

fn create_terminal(
    mode: ScreenMode,
) -> Result<Terminal<CrosstermBackend<std::io::Stdout>>, std::io::Error> {
    let backend = CrosstermBackend::new(std::io::stdout());
    match mode {
        ScreenMode::Fullscreen => {
            let mut t = Terminal::new(backend)?;
            t.clear()?;
            Ok(t)
        }
        ScreenMode::Inline => {
            let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
            let viewport_h = rows.clamp(7, 14);
            Terminal::with_options(
                backend,
                TerminalOptions {
                    viewport: Viewport::Inline(viewport_h),
                },
            )
        }
    }
}

fn insert_scrollback_lines<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    lines: &[ratatui::text::Line<'static>],
    chunk_max: usize,
) {
    for chunk in lines.chunks(chunk_max) {
        let h = chunk.len() as u16;
        let _ = terminal.insert_before(h, |buf| {
            for (row_off, line) in chunk.iter().enumerate() {
                buf.set_line(
                    buf.area.x,
                    buf.area.y + row_off as u16,
                    line,
                    buf.area.width,
                );
            }
        });
    }
}

fn flush_inline_scrollback<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
) {
    if app.inline_committed_items >= app.items.len() {
        return;
    }
    let Ok(size) = terminal.size() else {
        return;
    };
    let width = size.width.max(20) as usize;
    let chunk_max = size.height.saturating_sub(2).max(4) as usize;

    // 首次写入原生 scrollback 时先写入欢迎页序言（含 1 行尾空行），切回全屏再返 Inline 不重复追加
    if !app.inline_welcome_committed {
        let welcome_lines = build_welcome_lines_for_scrollback(app, width);
        insert_scrollback_lines(terminal, &welcome_lines, chunk_max);
        app.inline_welcome_committed = true;
    }

    while app.inline_committed_items < app.items.len() {
        let idx = app.inline_committed_items;
        let lines = build_item_lines_for_scrollback_at(
            &app.items,
            idx,
            &app.tool_extras,
            width,
            &app.theme,
            &app.emphasis,
            app.show_thinking,
            app.verbose_expanded,
        );
        insert_scrollback_lines(terminal, &lines, chunk_max);
        app.inline_committed_items += 1;
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_action(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    active_sid: &Arc<std::sync::RwLock<String>>,
    ui_tx: &mpsc::UnboundedSender<UiEvent>,
    active_cancel: &mut Option<CancellationToken>,
    pending_steer: &mut Option<String>,
    user_cancelled: &mut bool,
    action: AppAction,
) {
    match action {
        AppAction::None => {}
        AppAction::Quit => {
            app.should_quit = true;
        }
        AppAction::CancelRun => {
            *pending_steer = None;
            *user_cancelled = true;
            if let Some(cancel) = active_cancel.as_ref() {
                cancel.cancel();
            }
        }
        AppAction::NewSession => {
            *pending_steer = None;
            *user_cancelled = false;
            if let Some(cancel) = active_cancel.take() {
                cancel.cancel();
            }
            if let Ok(sessions) = ctx.sessions() {
                let next = sessions.create(None);
                let title = next.header().title;
                *active_sid.write().unwrap() = next.id.clone();
                app.reset_for_new_session(next.id.clone(), title);
                *active_session = next;
                populate_side_panel_data(ctx, app).await;
            }
        }
        AppAction::SubmitPrompt(prompt) => {
            spawn_agent_run(
                ctx,
                config,
                app,
                active_session,
                ui_tx,
                active_cancel,
                prompt,
            );
        }
        AppAction::SteerPrompt(prompt) => {
            if let Some(cancel) = active_cancel.as_ref() {
                *pending_steer = Some(prompt);
                *user_cancelled = false;
                cancel.cancel();
            } else {
                app.push_user_message(&prompt);
                spawn_agent_run(
                    ctx,
                    config,
                    app,
                    active_session,
                    ui_tx,
                    active_cancel,
                    prompt,
                );
            }
        }
        AppAction::RunLocalBash {
            command,
            exclude_from_context,
        } => {
            spawn_local_bash_run(
                app,
                active_session,
                ui_tx,
                active_cancel,
                command,
                exclude_from_context,
            );
        }
        AppAction::ShowStats => {
            if let Some(traj) = ctx.get::<TrajectoryRuntime>("trajectory") {
                if let Some(report) = traj.inspect(&app.session_id, TimelineMode::Sequence) {
                    let s = &report.snapshot.stats;
                    app.push_notice(
                        NoticeLevel::Info,
                        format!(
                            "会话统计：turns={} | steps={} | tools={} (err={}) | tokens ↑{} ↓{} | avg TTFT={}ms | findings={}",
                            s.turns,
                            s.steps,
                            s.tool_calls,
                            s.tool_errors,
                            s.input_tokens,
                            s.output_tokens,
                            s.ttft_ms.checked_div(s.ttft_steps).unwrap_or(0),
                            report.findings.len()
                        ),
                    );
                } else {
                    app.push_notice(NoticeLevel::Info, "当前会话暂无轨迹统计");
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "trajectory 服务未启用");
            }
        }
        AppAction::ShowTools => {
            if let Some(rt) = ctx.get::<ToolRuntime>("tools") {
                let declared: std::collections::HashSet<String> = rt
                    .list_declared(Some(&app.session_id))
                    .into_iter()
                    .map(|t| t.name)
                    .collect();
                let all = rt.list_all();
                let mut lines = vec![format!(
                    "工具列表（当前会话已声明 {} / 共 {} 个可用）：",
                    declared.len(),
                    all.len()
                )];
                for info in all {
                    let active_mark = if declared.contains(&info.name) {
                        "●"
                    } else {
                        "○"
                    };
                    let exp = match info.exposure {
                        ToolExposure::Direct => "direct",
                        ToolExposure::ModelOnly => "model-only",
                        ToolExposure::Codemode => "codemode",
                        ToolExposure::Deferred => "deferred",
                        ToolExposure::Hidden => "hidden",
                    };
                    lines.push(format!("  {active_mark} {} [{exp}]", info.name));
                }
                app.push_notice(NoticeLevel::Info, lines.join("\n"));
            } else {
                app.push_notice(NoticeLevel::Warn, "tools 服务未就绪");
            }
        }
        AppAction::ShowMcp => {
            if let Some(mgr) = ctx.get::<McpManager>("mcp") {
                let statuses = mgr.statuses();
                if statuses.is_empty() {
                    app.push_notice(
                        NoticeLevel::Info,
                        "未配置任何 MCP Server（可在 ~/.mizar/mcp.json 或 <workspace>/.mizar/mcp.json 配置）",
                    );
                } else {
                    let mut lines = vec![format!("MCP Server 状态（共 {} 个）：", statuses.len())];
                    for s in statuses {
                        let st_str = match s.state {
                            McpConnectionState::Connecting => "连接中…".to_string(),
                            McpConnectionState::Connected {
                                tools_count,
                                has_resources,
                            } => format!(
                                "已连接（{tools_count} 工具{}）",
                                if has_resources { " + 资源" } else { "" }
                            ),
                            McpConnectionState::Disabled => "已禁用".to_string(),
                            McpConnectionState::Failed(err) => format!("失败：{err}"),
                        };
                        lines.push(format!("  - {} [{:?}]: {st_str}", s.name, s.exposure));
                    }
                    app.push_notice(NoticeLevel::Info, lines.join("\n"));
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "MCP 服务未启用");
            }
        }
        AppAction::ShowSkills => {
            if let Some(reg) = ctx.get::<SkillRegistry>("skills") {
                let list = reg
                    .list(SkillLookupOptions {
                        cwd: Some(app.workspace.clone()),
                        cancel: CancellationToken::new(),
                    })
                    .await;
                app.user_skill_names = list
                    .iter()
                    .filter(|s| s.is_user_invocable())
                    .map(|s| s.name.clone())
                    .collect();
                if list.is_empty() {
                    app.push_notice(
                        NoticeLevel::Info,
                        "未发现任何技能（可在 ~/.agents/skills 或 <project>/.agents/skills 添加 <name>/SKILL.md）",
                    );
                } else {
                    let mut lines = vec![format!("已发现技能（共 {} 个）：", list.len())];
                    for s in list {
                        let scope_tag = match (s.is_model_invocable(), s.is_user_invocable()) {
                            (true, true) => "model+user",
                            (true, false) => "model-only",
                            (false, true) => "user-only",
                            (false, false) => "hidden",
                        };
                        let loc = s
                            .path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| s.provider.clone());
                        lines.push(format!(
                            "  - /{} [{} | {}]: {} ({})",
                            s.name,
                            s.source.as_str(),
                            scope_tag,
                            s.description,
                            loc
                        ));
                    }
                    lines.push("提示：直接输入 /<技能名> 可显式加载技能全文".to_string());
                    app.push_notice(NoticeLevel::Info, lines.join("\n"));
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "skills 服务未启用");
            }
        }
        AppAction::ShowInstructions => {
            if let Some(inst) = ctx.get::<AgentInstructions>("agentInstructions") {
                let report = inst.probe_report(Some(&app.session_id));
                if !report.enabled {
                    app.push_notice(NoticeLevel::Info, "工作区指令加载已禁用（--no-instructions 或 settings.json）");
                } else if report.scopes.is_empty() {
                    app.push_notice(
                        NoticeLevel::Info,
                        "未发现 AGENTS.md / CLAUDE.md 指令文件（可在 ~/.mizar/、~/.agents/ 或工作区/Git 祖先目录添加）",
                    );
                } else {
                    let mut lines = vec![format!("已探测指令文件（共 {} 个）：", report.scopes.len())];
                    for s in report.scopes {
                        let status = if s.omitted {
                            "已省略(超出预算)"
                        } else if s.truncated {
                            "已截断"
                        } else {
                            "已注入"
                        };
                        lines.push(format!(
                            "  - {} [{} | {} | {} 字节]",
                            s.display_path,
                            s.scope_kind.as_str(),
                            status,
                            s.bytes
                        ));
                    }
                    for w in report.warnings {
                        lines.push(format!("  ! {w}"));
                    }
                    app.push_notice(NoticeLevel::Info, lines.join("\n"));
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "agentInstructions 服务未启用");
            }
        }
        AppAction::ExportTrajectory(custom_path) => {
            if let Some(traj) = ctx.get::<TrajectoryRuntime>("trajectory") {
                if let Some(jsonl) = traj.export_jsonl(&app.session_id) {
                    let target = custom_path.unwrap_or_else(|| {
                        app.workspace
                            .join(format!("{}.trajectory.jsonl", app.session_id))
                    });
                    match tokio::fs::write(&target, jsonl).await {
                        Ok(()) => app.push_notice(
                            NoticeLevel::Info,
                            format!("轨迹已导出至：{}", target.display()),
                        ),
                        Err(err) => {
                            app.push_notice(NoticeLevel::Error, format!("导出失败：{err}"))
                        }
                    }
                } else {
                    app.push_notice(NoticeLevel::Info, "当前会话暂无轨迹事件可导出");
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "trajectory 服务未启用");
            }
        }
        AppAction::ShowOrSwitchModel(target) => {
            if let Ok(llm) = ctx.llm() {
                app.providers = llm.provider_infos();
            }
            match target {
                None if app.providers.iter().any(|p| !p.models.is_empty()) => {
                    let _ = app.open_model_selector();
                }
                None => {
                    let mut lines = vec![format!("当前模型：{}", app.active_model_label())];
                    if !app.providers.is_empty() {
                        lines.push("已注册提供方：".to_string());
                        for p in &app.providers {
                            let models = if p.models.is_empty() {
                                "（动态/默认模型）".to_string()
                            } else {
                                p.models
                                    .iter()
                                    .map(|m| m.id.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            lines.push(format!("  - {} [{}]: {}", p.id, p.protocol, models));
                        }
                    }
                    lines.push("用法：/model <模型ID> 或 /model <provider>/<模型ID>".to_string());
                    app.push_notice(NoticeLevel::Info, lines.join("\n"));
                }
                Some(spec) => {
                    let (next_provider, next_model) =
                        resolve_model_switch(&app.providers, app.provider.as_deref(), &spec);
                    if let Some(p) = next_provider {
                        app.provider = Some(p);
                    }
                    app.model = Some(next_model);
                    app.push_notice(
                        NoticeLevel::Info,
                        format!("已切换后续请求模型为：{}", app.active_model_label()),
                    );
                }
            }
        }
        AppAction::RefreshSidePanelData => {
            populate_side_panel_data(ctx, app).await;
        }
        AppAction::OpenTrajectoryScene(mode) => {
            let report = ctx
                .get::<TrajectoryRuntime>("trajectory")
                .and_then(|traj| traj.inspect(&app.session_id, mode));
            app.open_trajectory_with_report(report);
        }
        AppAction::OpenSessionTree => {
            app.open_session_tree_with(project_session_tree(active_session));
        }
        AppAction::OpenSkillPicker => {
            if let Some(reg) = ctx.get::<SkillRegistry>("skills") {
                let list = reg
                    .list(SkillLookupOptions {
                        cwd: Some(app.workspace.clone()),
                        cancel: CancellationToken::new(),
                    })
                    .await;
                let mut items: Vec<SkillPickerItem> = list
                    .into_iter()
                    .filter(|s| s.is_user_invocable())
                    .map(|s| SkillPickerItem {
                        name: s.name,
                        description: s.description,
                        source: s.source.as_str().to_string(),
                    })
                    .collect();
                items.sort_by(|a, b| a.name.cmp(&b.name));
                if items.is_empty() {
                    app.push_notice(
                        NoticeLevel::Info,
                        "未发现可交互调用的技能（可在 ~/.agents/skills 或 <project>/.agents/skills 添加 <name>/SKILL.md）",
                    );
                } else {
                    app.overlay.open_skill_picker(items);
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "skills 服务未启用");
            }
        }
        AppAction::ForkSessionAtSeq(target_seq) => {
            if let Ok(sessions) = ctx.sessions() {
                match sessions.fork(&app.session_id, target_seq, SessionCreateOptions::default()) {
                    Ok(next) => {
                        *active_sid.write().unwrap() = next.id.clone();
                        app.restore_session_view(project_fork_view(&next, target_seq));
                        *active_session = next;
                        populate_side_panel_data(ctx, app).await;
                    }
                    Err(err) => {
                        app.push_notice(NoticeLevel::Error, format!("分叉会话失败：{err}"));
                    }
                }
            }
        }
        AppAction::OpenExternalEditor => {
            let draft = app.input.expand_text();
            let tmp_path =
                std::env::temp_dir().join(format!("mizar-draft-{}.md", std::process::id()));
            if let Err(err) = std::fs::write(&tmp_path, &draft) {
                app.push_notice(NoticeLevel::Error, format!("写入临时草稿文件失败：{err}"));
                return;
            }
            let editor_cmd = std::env::var("VISUAL")
                .or_else(|_| std::env::var("EDITOR"))
                .unwrap_or_else(|_| "vi".to_string());
            let mut parts = editor_cmd.split_whitespace();
            let bin = parts.next().unwrap_or("vi").to_string();
            let extra_args: Vec<String> = parts.map(ToString::to_string).collect();

            let _ = suspend_for_external_editor();
            let status = std::process::Command::new(&bin)
                .args(&extra_args)
                .arg(&tmp_path)
                .status();
            let _ = resume_after_external_editor(app.screen_mode, app.mouse_capture_enabled);
            if let Ok(rebuilt) = create_terminal(app.screen_mode) {
                *terminal = rebuilt;
            }

            match status {
                Ok(s) if s.success() => match std::fs::read_to_string(&tmp_path) {
                    Ok(edited) => {
                        let clean = edited.strip_suffix('\n').unwrap_or(&edited);
                        app.input.set_text_end(clean);
                        app.set_flash("✓ 已从外部编辑器同步草稿");
                    }
                    Err(err) => {
                        app.push_notice(NoticeLevel::Error, format!("读取编辑结果失败：{err}"));
                    }
                },
                Ok(s) => {
                    app.push_notice(
                        NoticeLevel::Warn,
                        format!("外部编辑器 `{bin}` 退出码：{:?}", s.code()),
                    );
                }
                Err(err) => {
                    app.push_notice(
                        NoticeLevel::Error,
                        format!("启动外部编辑器 `{bin}` 失败：{err}"),
                    );
                }
            }
            let _ = std::fs::remove_file(&tmp_path);
        }
        AppAction::SwitchScreenMode(next_mode) => {
            if next_mode == app.screen_mode {
                return;
            }
            app.screen_mode = next_mode;
            app.mouse_capture_enabled = next_mode == ScreenMode::Fullscreen;
            app.selection.clear();
            app.item_cache.clear();
            let _ = apply_terminal_modes(next_mode, app.mouse_capture_enabled);
            match create_terminal(next_mode) {
                Ok(rebuilt) => {
                    *terminal = rebuilt;
                    if next_mode == ScreenMode::Fullscreen {
                        app.scroll_to_bottom();
                    }
                    app.set_flash(format!("✓ 视图已切换为 {} (F3 切换)", next_mode.as_str()));
                }
                Err(err) => {
                    app.push_notice(NoticeLevel::Error, format!("切换终端视图失败：{err}"));
                }
            }
        }
        AppAction::SetMouseCapture(enabled) => {
            app.mouse_capture_enabled = enabled;
            if let Err(err) = set_mouse_capture_enabled(enabled) {
                app.push_notice(NoticeLevel::Error, format!("切换鼠标捕捉失败：{err}"));
            } else if enabled {
                app.set_flash("🖱 已启用应用鼠标选区与滚动 (F2 切换)");
            } else {
                app.set_flash("🖱 已暂停鼠标追踪，终端原生划词复制可用 (F2 恢复)");
            }
        }
    }
}

async fn populate_side_panel_data(ctx: &Context, app: &mut AppState) {
    app.refresh_context_breakdown();
    app.side_panel_data.files = app.touched_files.clone();
    if let Some(inst) = ctx.get::<AgentInstructions>("agentInstructions") {
        let report = inst.probe_report(Some(&app.session_id));
        app.side_panel_data.instructions = report
            .scopes
            .into_iter()
            .map(|s| InstructionSummaryRow {
                path: s.display_path,
                scope: s.scope_kind.as_str().to_string(),
                bytes: s.bytes,
                truncated: s.truncated,
                omitted: s.omitted,
            })
            .collect();
    }
    let (git_changes, git_diff_stat) = collect_git_summary(&app.workspace).await;
    app.side_panel_data.git_changes = git_changes;
    app.side_panel_data.git_diff_stat = git_diff_stat;

    if let Some(rt) = ctx.get::<ToolRuntime>("tools") {
        let declared: HashSet<String> = rt
            .list_declared(Some(&app.session_id))
            .into_iter()
            .map(|t| t.name)
            .collect();
        app.side_panel_data.tools = rt
            .list_all()
            .into_iter()
            .map(|info| {
                let exposure = match info.exposure {
                    ToolExposure::Direct => "direct",
                    ToolExposure::ModelOnly => "model-only",
                    ToolExposure::Codemode => "codemode",
                    ToolExposure::Deferred => "deferred",
                    ToolExposure::Hidden => "hidden",
                }
                .to_string();
                let is_decl = declared.contains(&info.name);
                ToolSummaryRow {
                    name: info.name,
                    exposure,
                    declared: is_decl,
                }
            })
            .collect();
    }

    if let Some(mgr) = ctx.get::<McpManager>("mcp") {
        app.side_panel_data.mcps = mgr
            .statuses()
            .into_iter()
            .map(|s| {
                let (status, ok) = match s.state {
                    McpConnectionState::Connecting => ("连接中…".to_string(), false),
                    McpConnectionState::Connected {
                        tools_count,
                        has_resources,
                    } => (
                        format!(
                            "已连接 ({tools_count} 工具{})",
                            if has_resources { " + 资源" } else { "" }
                        ),
                        true,
                    ),
                    McpConnectionState::Disabled => ("已禁用".to_string(), false),
                    McpConnectionState::Failed(err) => (format!("失败: {err}"), false),
                };
                McpSummaryRow {
                    name: s.name,
                    status,
                    ok,
                }
            })
            .collect();
    }

    let mut findings = app.anomalies.clone();
    if let Some(traj) = ctx.get::<TrajectoryRuntime>("trajectory")
        && let Some(report) = traj.inspect(&app.session_id, TimelineMode::Sequence)
    {
        for f in report.findings {
            let line = format!("[{:?}] {}: {}", f.severity, f.title, f.summary);
            if !findings.contains(&line) {
                findings.push(line);
            }
        }
    }
    app.side_panel_data.findings = findings;
}

async fn collect_git_summary(workspace: &std::path::Path) -> (Vec<FileTouchRow>, String) {
    let status_fut = tokio::process::Command::new("git")
        .arg("status")
        .arg("--porcelain")
        .current_dir(workspace)
        .output();
    let mut changes = Vec::new();
    if let Ok(Ok(out)) = tokio::time::timeout(Duration::from_millis(300), status_fut).await
        && out.status.success()
    {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines().take(24) {
            if line.len() < 4 {
                continue;
            }
            let code = line[..2].trim();
            let path = line[3..].trim().to_string();
            if !path.is_empty() {
                changes.push(FileTouchRow {
                    path,
                    op: if code.is_empty() {
                        "?".to_string()
                    } else {
                        code.to_string()
                    },
                });
            }
        }
    }

    let stat_fut = tokio::process::Command::new("git")
        .arg("diff")
        .arg("--shortstat")
        .current_dir(workspace)
        .output();
    let mut stat = String::new();
    if let Ok(Ok(out)) = tokio::time::timeout(Duration::from_millis(300), stat_fut).await
        && out.status.success()
    {
        stat = String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    (changes, stat)
}

async fn collect_user_skill_names(ctx: &Context, workspace: &std::path::Path) -> Vec<String> {
    let Some(reg) = ctx.get::<SkillRegistry>("skills") else {
        return Vec::new();
    };
    reg.list(SkillLookupOptions {
        cwd: Some(workspace.to_path_buf()),
        cancel: CancellationToken::new(),
    })
    .await
    .into_iter()
    .filter(|s| s.is_user_invocable())
    .map(|s| s.name)
    .collect()
}

fn find_model_in_provider<'a>(
    provider: &'a mizar_llm::ProviderInfo,
    spec: &str,
) -> Option<&'a mizar_llm::ModelInfo> {
    provider
        .models
        .iter()
        .find(|m| m.id == spec)
        .or_else(|| {
            let lower = spec.to_ascii_lowercase();
            provider
                .models
                .iter()
                .find(|m| m.name.eq_ignore_ascii_case(&lower))
        })
}

/// 解析 `/model <spec>`：优先识别带 `/` 的完整模型 ID（如 `mayoru-completions/gemini-4-hydrogen`），
/// 仅当 `/` 前缀匹配已注册 provider ID 时才拆分 `<provider>/<model>`。
fn resolve_model_switch(
    providers: &[mizar_llm::ProviderInfo],
    current_provider: Option<&str>,
    spec: &str,
) -> (Option<String>, String) {
    if let Some(cur) = current_provider
        && let Some(prov) = providers.iter().find(|p| p.id == cur)
        && let Some(model) = find_model_in_provider(prov, spec)
    {
        return (None, model.id.clone());
    }

    if let Some((prov_prefix, rest)) = spec.split_once('/') {
        let prov_prefix = prov_prefix.trim();
        let rest = rest.trim();
        if !rest.is_empty()
            && let Some(prov) = providers.iter().find(|p| p.id == prov_prefix)
        {
            let matched = find_model_in_provider(prov, rest)
                .map(|m| m.id.clone())
                .unwrap_or_else(|| rest.to_string());
            return (Some(prov.id.clone()), matched);
        }
    }

    for prov in providers {
        if let Some(model) = find_model_in_provider(prov, spec) {
            return (Some(prov.id.clone()), model.id.clone());
        }
    }

    (None, spec.to_string())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use mizar_llm::{ModelInfo, ProviderInfo};
    use mizar_tui_ui::app::{AppAction, AppState, ChatItem, UiEvent};

    use super::{TuiExitSummary, format_resume_command, process_ui_event, resolve_model_switch};

    #[test]
    fn resolve_model_switch_handles_slashes_in_model_ids() {
        let providers = vec![ProviderInfo {
            id: "magpie".to_string(),
            name: "magpie".to_string(),
            protocol: "openai-completions".to_string(),
            models: vec![
                ModelInfo {
                    id: "mayoru-completions/gemini-4-hydrogen".to_string(),
                    name: "Gemini 4 Hydrogen".to_string(),
                    context_window: None,
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                },
                ModelInfo {
                    id: "mayoru-messages/claude-sonnet-5-5".to_string(),
                    name: "Claude Sonnet 5.5".to_string(),
                    context_window: None,
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                },
            ],
        }];

        // 带 "/" 的完整模型 ID 不应被误拆成 provider = "mayoru-completions"
        assert_eq!(
            resolve_model_switch(
                &providers,
                Some("magpie"),
                "mayoru-messages/claude-sonnet-5-5",
            ),
            (None, "mayoru-messages/claude-sonnet-5-5".to_string())
        );

        // 前缀精确命中已注册 provider ID 时，正常切换 provider 与内部带 "/" 的模型 ID
        assert_eq!(
            resolve_model_switch(
                &providers,
                None,
                "magpie/mayoru-completions/gemini-4-hydrogen",
            ),
            (
                Some("magpie".to_string()),
                "mayoru-completions/gemini-4-hydrogen".to_string(),
            )
        );
    }

    #[test]
    fn format_resume_command_handles_cargo_target_installed_bin_and_unpersisted() {
        let persisted = TuiExitSummary {
            session_id: "sess-abc-123".to_string(),
            persisted: true,
        };
        assert_eq!(
            format_resume_command(
                &persisted,
                None,
                Some(Path::new("/repo/target/debug/mizar")),
                Some(Path::new("/repo")),
            ),
            Some("./target/debug/mizar --resume sess-abc-123".to_string())
        );
        assert_eq!(
            format_resume_command(
                &persisted,
                Some(Path::new("/tmp/my workspace")),
                Some(Path::new("/repo/target/release/mizar")),
                Some(Path::new("/other")),
            ),
            Some(
                "/repo/target/release/mizar -C '/tmp/my workspace' --resume sess-abc-123"
                    .to_string()
            )
        );
        assert_eq!(
            format_resume_command(
                &persisted,
                None,
                Some(Path::new("/usr/local/bin/mizar")),
                Some(Path::new("/repo")),
            ),
            Some("mizar --resume sess-abc-123".to_string())
        );

        let unpersisted = TuiExitSummary {
            session_id: "sess-mem".to_string(),
            persisted: false,
        };
        assert_eq!(format_resume_command(&unpersisted, None, None, None), None);
    }

    #[test]
    fn process_ui_event_records_user_items_for_steer_and_queued_prompts() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queue",
            Vec::new(),
            None,
            None,
            None,
        );
        app.is_busy = true;
        app.queued_prompts.push("queued follow-up".into());

        let mut active_cancel = None;
        let mut pending_steer = Some("urgent steer".into());
        let mut user_cancelled = false;

        let act1 = process_ui_event(
            &mut app,
            &mut active_cancel,
            &mut pending_steer,
            &mut user_cancelled,
            UiEvent::RunFinished(Err("request aborted".into())),
        );
        assert_eq!(act1, AppAction::SubmitPrompt("urgent steer".into()));
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::User { text }) if text == "urgent steer"
        ));
        assert_eq!(app.queued_prompts, vec!["queued follow-up".to_string()]);

        let act2 = process_ui_event(
            &mut app,
            &mut active_cancel,
            &mut pending_steer,
            &mut user_cancelled,
            UiEvent::RunFinished(Ok("done".into())),
        );
        assert_eq!(act2, AppAction::SubmitPrompt("queued follow-up".into()));
        assert!(app.queued_prompts.is_empty());
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::User { text }) if text == "queued follow-up"
        ));
    }

    #[tokio::test]
    async fn spawn_local_bash_run_executes_persists_and_supports_cancel() {
        use cordis::Service;
        use mizar_session::{SessionContextExt, SessionStore};

        let ctx = cordis::Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let session = ctx.sessions().unwrap().create(Some("runner-bash".into()));

        let mut app = AppState::new(
            std::env::temp_dir(),
            &session.id,
            Vec::new(),
            None,
            None,
            None,
        );
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut active_cancel = None;
        let mut pending_steer = None;
        let mut user_cancelled = false;

        super::spawn_local_bash_run(
            &mut app,
            &session,
            &tx,
            &mut active_cancel,
            "echo RUNNER_LOCAL_OK".into(),
            false,
        );
        assert!(app.is_bash_running);
        assert!(active_cancel.is_some());

        while let Some(ev) = rx.recv().await {
            let is_done = matches!(ev, UiEvent::LocalBashFinished { .. });
            let act = process_ui_event(
                &mut app,
                &mut active_cancel,
                &mut pending_steer,
                &mut user_cancelled,
                ev,
            );
            assert_eq!(act, AppAction::None);
            if is_done {
                break;
            }
        }
        assert!(!app.is_bash_running);
        assert!(active_cancel.is_none());
        assert_eq!(session.log().len(), 1);
        assert!(matches!(
            &session.log()[0].event,
            mizar_session::SessionEvent::BashExecution {
                command,
                output,
                exit_code: Some(0),
                cancelled: false,
                exclude_from_context: false,
            } if command == "echo RUNNER_LOCAL_OK" && output.contains("RUNNER_LOCAL_OK")
        ));

        // 测试 Esc 取消（active_cancel.cancel()）
        super::spawn_local_bash_run(
            &mut app,
            &session,
            &tx,
            &mut active_cancel,
            "sleep 10".into(),
            true,
        );
        assert!(app.is_bash_running);
        active_cancel.as_ref().unwrap().cancel();
        while let Some(ev) = rx.recv().await {
            let is_done = matches!(ev, UiEvent::LocalBashFinished { .. });
            process_ui_event(
                &mut app,
                &mut active_cancel,
                &mut pending_steer,
                &mut user_cancelled,
                ev,
            );
            if is_done {
                break;
            }
        }
        assert!(!app.is_bash_running);
        assert!(active_cancel.is_none());
        assert_eq!(session.all_entries().len(), 2);
        assert!(matches!(
            &session.all_entries()[1].event,
            mizar_session::SessionEvent::BashExecution {
                cancelled: true,
                exclude_from_context: true,
                ..
            }
        ));

        ctx.fiber().dispose().await;
    }

    #[test]
    fn flush_inline_scrollback_commits_welcome_once_and_resets_on_new_session() {
        use mizar_tui_ui::app::NoticeLevel;
        use mizar_tui_ui::terminal::ScreenMode;
        use ratatui::backend::TestBackend;
        use ratatui::{Terminal, TerminalOptions, Viewport};

        let mut terminal = Terminal::with_options(
            TestBackend::new(80, 24),
            TerminalOptions {
                viewport: Viewport::Inline(10),
            },
        )
        .unwrap();

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-inline-flush",
            Vec::new(),
            Some("mock".into()),
            Some("mock-v1".into()),
            None,
        );
        app.screen_mode = ScreenMode::Inline;

        // 空会话调用 flush_inline_scrollback 不提前提交欢迎页
        super::flush_inline_scrollback(&mut terminal, &mut app);
        assert!(!app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 0);

        // 首条通知到来：先写欢迎页（inline_welcome_committed = true），再写通知
        app.push_notice(NoticeLevel::Info, "已切换后续请求模型为：mock/mock-v1");
        super::flush_inline_scrollback(&mut terminal, &mut app);
        assert!(app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 1);

        // 切到 Fullscreen 后新增条目再切回 Inline，不重置或重复写入欢迎页
        app.screen_mode = ScreenMode::Fullscreen;
        app.push_user_message("追加问题");
        app.screen_mode = ScreenMode::Inline;
        super::flush_inline_scrollback(&mut terminal, &mut app);
        assert!(app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 2);

        // /new 重置后再次允许写入新会话欢迎页
        app.reset_for_new_session("sess-inline-flush-2", None);
        assert!(!app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 0);
        super::flush_inline_scrollback(&mut terminal, &mut app);
        assert!(app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 1);
    }
}
