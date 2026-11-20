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
use futures::StreamExt;
use mizar_agent_loop::{
    AgentLoopContextExt, AgentReasoningDeltaEvent, AgentStatus, AgentStatusEvent,
    AgentStepEndEvent, AgentStepFirstTokenEvent, AgentStepStartEvent, AgentTextDeltaEvent,
    AgentToolCallEvent, AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent,
    AgentTurnStartEvent, AgentUsageEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_STATUS,
    EVENT_AGENT_STEP_END, EVENT_AGENT_STEP_FIRST_TOKEN, EVENT_AGENT_STEP_START,
    EVENT_AGENT_TEXT_DELTA, EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT,
    EVENT_AGENT_TOOL_UPDATE, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_START, EVENT_AGENT_USAGE,
    RunOptions, TurnEndReason,
};
use mizar_agent_spine::{
    CompactionContextExt, CompactionEndEvent, CompactionError, CompactionProgressEvent,
    CompactionStartEvent, CompactionTrigger, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS,
    EVENT_COMPACTION_START, ManualCompactOptions, SpinePluginsContextExt, patch_settings_file,
    persist_disabled_plugins, resolve_global_settings_path, strip_summary_wrapper,
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
    EVENT_TOOLS_CHANGE, EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolExecution, ToolExposure,
    ToolRuntime,
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
use mizar_tui_ui::overlay::{
    ApprovalDecision, SessionTreeEntry, SettingEntry, SettingItemKind, SkillPickerItem,
};
use mizar_tui_ui::side_panel::{
    FileTouchRow, InstructionSummaryRow, McpSummaryRow, ToolSummaryRow,
};
use mizar_tui_ui::terminal::{
    ProgramStatusState, ScreenMode, TerminalGuard, apply_terminal_modes, emit_program_status,
    resume_after_external_editor, set_mouse_capture_enabled, suspend_for_external_editor,
    write_begin_sync, write_end_sync,
};
use mizar_tui_ui::theme::ThemePreset;
use mizar_tui_ui::ui::{
    build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback, render,
};
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
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b'~' | b':' | b'@')
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
    pub show_thinking: Option<bool>,
    pub verbose_expanded: Option<bool>,
    pub side_panel_open: Option<bool>,
    pub mouse_capture: Option<bool>,
    pub resume_session: Option<ResumeTarget>,
    pub restrict_to_workspace: bool,
    pub check_for_updates: bool,
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
            show_thinking: None,
            verbose_expanded: None,
            side_panel_open: None,
            mouse_capture: None,
            resume_session: None,
            restrict_to_workspace: false,
            check_for_updates: false,
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
pub fn project_session_view(
    session: &Session,
    finish_notice: impl Into<String>,
) -> SessionViewData {
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
                summary: strip_summary_wrapper(&summary),
                first_kept_seq,
            }),
            SessionEvent::Assistant { blocks, .. } => Some(SessionReplayItem::Assistant { blocks }),
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
            SessionEvent::ToolResultPrune { .. } | SessionEvent::Trace { .. } => None,
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
                summary: format!(
                    "[{source}] {}",
                    truncate_display(&text.replace('\n', " "), 60)
                ),
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
                summary: format!(
                    "{name}: {}",
                    truncate_display(&content.replace('\n', " "), 60)
                ),
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
            SessionEvent::ToolResultPrune { .. } | SessionEvent::Trace { .. } => None,
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
        handles.push(
            ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::TurnStart { turn: ev.turn });
                }
            })?,
        );
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
        handles.push(
            ctx.on(EVENT_AGENT_STEP_START, move |ev: &AgentStepStartEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::StepStart { ts_ms: ev.ts_ms });
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_AGENT_STEP_FIRST_TOKEN,
            move |ev: &AgentStepFirstTokenEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::FirstToken { ts_ms: ev.ts_ms });
                }
            },
        )?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
            if check(&ev.session.id) {
                let _ = tx.send(UiEvent::StepEnd {
                    ts_ms: ev.ts_ms,
                    usage: ev.usage,
                });
            }
        })?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TEXT_DELTA, move |ev: &AgentTextDeltaEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::TextDelta {
                        text: ev.text.clone(),
                        ts_ms: ev.ts_ms,
                    });
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_AGENT_REASONING_DELTA,
            move |ev: &AgentReasoningDeltaEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::ReasoningDelta {
                        text: ev.text.clone(),
                        ts_ms: ev.ts_ms,
                    });
                }
            },
        )?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::ToolCall(ev.call.clone()));
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_UPDATE, move |ev: &AgentToolUpdateEvent| {
                if check(&ev.session.id) {
                    let _ = tx.send(UiEvent::ToolUpdate {
                        call_id: ev.tool_call_id.clone(),
                        name: ev.name.clone(),
                        preview: ev.preview.clone(),
                    });
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_RESULT, move |ev: &AgentToolResultEvent| {
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
            })?,
        );
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
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_TRAJECTORY_ANOMALY,
            move |ev: &TrajectoryAnomalyEvent| {
                if check(&ev.session_id) {
                    let _ = tx.send(UiEvent::Anomaly(ev.finding.summary.clone()));
                }
            },
        )?);
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_COMPACTION_START, move |ev: &CompactionStartEvent| {
                if check(&ev.session_id) {
                    let _ = tx.send(UiEvent::CompactionStarted {
                        trigger: ev.trigger.as_str().to_string(),
                    });
                }
            })?,
        );
    }
    {
        let tx = tx.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_COMPACTION_PROGRESS,
            move |ev: &CompactionProgressEvent| {
                if check(&ev.session_id) {
                    let _ = tx.send(UiEvent::CompactionProgress {
                        stage: ev.stage.clone(),
                        delta_chars: ev.delta_chars,
                    });
                }
            },
        )?);
    }
    {
        let tx = tx.clone();
        let check = is_current;
        handles.push(
            ctx.on(EVENT_COMPACTION_END, move |ev: &CompactionEndEvent| {
                if check(&ev.session_id) {
                    let _ = tx.send(UiEvent::CompactionFinished {
                        outcome: ev.outcome.as_str().to_string(),
                        tokens_before: Some(ev.tokens_before),
                        tokens_after: Some(ev.tokens_after),
                        pruned_count: ev.pruned_count,
                        first_kept_seq: ev.first_kept_seq,
                        summary_preview: ev.summary_preview.clone(),
                        message: ev.skipped_reason.clone(),
                        is_manual: ev.trigger == CompactionTrigger::Manual,
                    });
                }
            })?,
        );
    }

    Ok(handles)
}

/// 等待 `agentLoop`、`sessions`、`llm` 核心服务以及所有已挂载插件就绪（消除异步挂载竞态）。
pub async fn wait_for_services(ctx: &Context) -> Result<(), Box<dyn std::error::Error>> {
    if ctx
        .get::<mizar_agent_loop::AgentLoop>("agentLoop")
        .is_none()
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
    let providers = ctx.llm().map(|l| l.provider_infos()).unwrap_or_default();

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
    let mut handles = subscribe_session_events(&ctx, Arc::clone(&active_sid), ui_tx.clone())?;

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
    {
        // 不在启动阶段缓存旧 `Arc<SkillRegistry>`，确保即使启动屏蔽或运行期重挂 `skills` 也能动态感知最新列表
        let ctx_skills = ctx.clone();
        let ws_skills = config.workspace.clone();
        let tx_skills = ui_tx.clone();
        handles.push(ctx.on_signal(EVENT_SKILLS_CHANGE, move || {
            let ctx_inner = ctx_skills.clone();
            let ws = ws_skills.clone();
            let tx = tx_skills.clone();
            tokio::spawn(async move {
                let names = collect_user_skill_names(&ctx_inner, &ws).await;
                let _ = tx.send(UiEvent::UserSkillsUpdated(names));
            });
        })?);
    }
    {
        let tx_tools = ui_tx.clone();
        handles.push(ctx.on_signal(EVENT_TOOLS_CHANGE, move || {
            let _ = tx_tools.send(UiEvent::ToolsUpdated);
        })?);
    }
    if let Ok(plugins) = ctx.spine_plugins() {
        for err_msg in plugins.startup_errors() {
            app.push_notice(NoticeLevel::Warn, err_msg);
        }
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
    if let Some(v) = config.show_thinking {
        app.show_thinking = v;
    }
    if let Some(v) = config.verbose_expanded {
        app.verbose_expanded = v;
    }
    if let Some(v) = config.side_panel_open {
        app.side_panel.open = v;
    }
    app.mouse_capture_enabled = config
        .mouse_capture
        .unwrap_or(config.screen_mode == ScreenMode::Fullscreen);
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

    let update_check_task = if config.check_for_updates {
        let tx_update = ui_tx.clone();
        Some(tokio::spawn(async move {
            if let Some(latest_tag) = crate::update::check_for_new_version().await {
                let _ = tx_update.send(UiEvent::Notice {
                    level: NoticeLevel::Info,
                    text: crate::update::format_update_notice(&latest_tag),
                });
            }
        }))
    } else {
        None
    };

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

    if let Some(task) = update_check_task {
        task.abort();
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
        move |exec: ToolExecution,
              next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
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
        "read"
            | "grep"
            | "find"
            | "ls"
            | "tool_search"
            | "recall_search"
            | "recall_read"
            | "mcp_list_resources"
            | "mcp_read_resource"
    )
}

fn process_ui_event(
    app: &mut AppState,
    active_cancel: &mut Option<CancellationToken>,
    pending_steer: &mut Option<String>,
    user_cancelled: &mut bool,
    event: UiEvent,
) -> AppAction {
    if matches!(event, UiEvent::ToolsUpdated) {
        return AppAction::RefreshSidePanelTools;
    }

    if matches!(event, UiEvent::LocalBashFinished { .. }) {
        *active_cancel = None;
        *user_cancelled = false;
        app.apply_ui_event(event);
        return AppAction::None;
    }

    if let UiEvent::CompactionFinished { is_manual, .. } = event {
        if is_manual {
            *active_cancel = None;
            *user_cancelled = false;
        }
        app.apply_ui_event(event);
        return AppAction::RefreshSidePanelData;
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
                format!(
                    "上一轮未成功，已保留 {} 条排队消息",
                    app.queued_prompts.len()
                ),
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
        AppAction::CompactSession(focus) => {
            let Ok(compaction) = ctx.compaction() else {
                app.push_notice(NoticeLevel::Warn, "compaction 服务未启用");
                return;
            };
            if !compaction.config().enabled {
                app.push_notice(NoticeLevel::Info, "上下文压缩已在配置中禁用");
                return;
            }
            let cancel = CancellationToken::new();
            *active_cancel = Some(cancel.clone());
            *user_cancelled = false;
            app.is_compacting = true;
            app.compaction_status = Some("手动压缩中…".to_string());

            let session = Arc::clone(active_session);
            let opts = ManualCompactOptions {
                provider: app.provider.clone(),
                model: app.model.clone(),
                focus,
            };
            let tx = ui_tx.clone();
            tokio::spawn(async move {
                match compaction.compact_now(session, opts, cancel).await {
                    Ok(_) | Err(CompactionError::Llm(_)) => {
                        // Pipeline 内部已广播 `EVENT_COMPACTION_END`，避免重复投递
                    }
                    Err(
                        err @ (CompactionError::AlreadyRunning(_) | CompactionError::Cordis(_)),
                    ) => {
                        let _ = tx.send(UiEvent::CompactionFinished {
                            outcome: "failed".to_string(),
                            tokens_before: None,
                            tokens_after: None,
                            pruned_count: 0,
                            first_kept_seq: None,
                            summary_preview: None,
                            message: Some(err.to_string()),
                            is_manual: true,
                        });
                    }
                }
            });
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
                    app.push_notice(
                        NoticeLevel::Info,
                        "工作区指令加载已禁用（--no-instructions 或 settings.json）",
                    );
                } else if report.scopes.is_empty() {
                    app.push_notice(
                        NoticeLevel::Info,
                        "未发现 AGENTS.md / CLAUDE.md 指令文件（可在 ~/.mizar/、~/.agents/ 或工作区/Git 祖先目录添加）",
                    );
                } else {
                    let mut lines =
                        vec![format!("已探测指令文件（共 {} 个）：", report.scopes.len())];
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
                        Err(err) => app.push_notice(NoticeLevel::Error, format!("导出失败：{err}")),
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
        AppAction::RefreshSidePanelTools => {
            populate_tools_and_mcps(ctx, app);
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
        AppAction::OpenSettings(initial_query) => {
            let items = build_settings_entries(ctx, app);
            let save_hint = resolve_global_settings_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "~/.mizar/settings.json".to_string());
            app.overlay.open_settings(items, initial_query, save_hint);
        }
        AppAction::ApplySetting { key, value } => {
            if let Some(plugin_id) = key.strip_prefix("plugin.") {
                let settings_path = resolve_global_settings_path();
                apply_module_toggle_setting(
                    ctx,
                    app,
                    settings_path.as_deref(),
                    plugin_id,
                    value == "true",
                )
                .await;
                return;
            }
            apply_host_setting_change(ctx, app, terminal, &key, &value).await;
            if let Err(err) = persist_setting_change(&key, &value) {
                app.push_notice(
                    NoticeLevel::Warn,
                    format!("配置 `{key}` 已实时生效，但写入 settings.json 失败：{err}"),
                );
            }
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

fn choice_index(options: &[&str], current: &str) -> usize {
    options
        .iter()
        .position(|opt| opt.eq_ignore_ascii_case(current))
        .unwrap_or(0)
}

fn format_soft_window_cap_label(cap: u64) -> &'static str {
    match cap {
        0 => "off",
        1..=160_000 => "128k",
        160_001..=260_000 => "200k",
        260_001..=440_000 => "380k",
        _ => "512k",
    }
}

fn parse_soft_window_cap_label(label: &str) -> u64 {
    match label {
        "off" | "0" => 0,
        "128k" => 128_000,
        "200k" => 200_000,
        "380k" => 380_000,
        "512k" => 512_000,
        _ => 380_000,
    }
}

fn build_settings_entries(ctx: &Context, app: &AppState) -> Vec<SettingEntry> {
    let theme_opts = [
        "auto",
        "mist-dark",
        "mist-light",
        "sakura-dark",
        "system",
        "ansi16",
    ];
    let screen_opts = ["fullscreen", "inline"];
    let approval_opts = ["auto", "ask"];
    let effort_opts = [
        "default", "off", "minimal", "low", "medium", "high", "xhigh", "max",
    ];
    let current_effort = app
        .reasoning_effort
        .map(|e| format!("{e:?}").to_lowercase())
        .unwrap_or_else(|| "default".to_string());

    let mut items = vec![
        SettingEntry {
            key: "ui.theme".into(),
            category: "界面".into(),
            label: "主题配色".into(),
            description: "切换 Oklab / 自适应终端主题，切换后即时刷新".into(),
            kind: SettingItemKind::Choice {
                options: theme_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&theme_opts, app.theme_preset.name()),
            },
        },
        SettingEntry {
            key: "ui.screenMode".into(),
            category: "界面".into(),
            label: "终端视图模式".into(),
            description: "fullscreen（全屏独立缓冲区）或 inline（内联保留终端滚屏历史）".into(),
            kind: SettingItemKind::Choice {
                options: screen_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&screen_opts, app.screen_mode.as_str()),
            },
        },
        SettingEntry {
            key: "ui.showThinking".into(),
            category: "界面".into(),
            label: "显示思考过程".into(),
            description: "在对话流中显示模型的推理思考块（快捷键 Ctrl+O）".into(),
            kind: SettingItemKind::Toggle {
                value: app.show_thinking,
            },
        },
        SettingEntry {
            key: "ui.verboseExpanded".into(),
            category: "界面".into(),
            label: "展开工具详情".into(),
            description: "默认展开工具参数与折叠 diff 预览（快捷键 Ctrl+O）".into(),
            kind: SettingItemKind::Toggle {
                value: app.verbose_expanded,
            },
        },
        SettingEntry {
            key: "ui.sidePanelOpen".into(),
            category: "界面".into(),
            label: "右侧信息分栏".into(),
            description: "展开或隐藏右侧上下文占用、文件变更与工具侧栏（快捷键 Ctrl+B）".into(),
            kind: SettingItemKind::Toggle {
                value: app.side_panel.open,
            },
        },
        SettingEntry {
            key: "ui.mouseCapture".into(),
            category: "界面".into(),
            label: "鼠标追踪与点击".into(),
            description: "开启应用内鼠标滚动/点击；关闭后可直接使用终端原生划词复制（F2）".into(),
            kind: SettingItemKind::Toggle {
                value: app.mouse_capture_enabled,
            },
        },
        SettingEntry {
            key: "ui.approvalMode".into(),
            category: "运行".into(),
            label: "危险工具审批模式".into(),
            description: "auto 自动放行；ask 在写文件与执行 Shell 前弹窗确认".into(),
            kind: SettingItemKind::Choice {
                options: approval_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&approval_opts, app.approval_mode.as_str()),
            },
        },
        SettingEntry {
            key: "reasoningEffort".into(),
            category: "运行".into(),
            label: "默认思考强度".into(),
            description: "控制推理模型的思考预算（default 使用模型默认；也可用 Tab 快速循环）"
                .into(),
            kind: SettingItemKind::Choice {
                options: effort_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&effort_opts, &current_effort),
            },
        },
    ];

    if let Ok(comp) = ctx.compaction() {
        let cfg = comp.config();
        let cap_opts = ["off", "128k", "200k", "380k", "512k"];
        items.extend([
            SettingEntry {
                key: "compaction.enabled".into(),
                category: "压缩".into(),
                label: "启用上下文压缩".into(),
                description: "总开关：关闭后不会执行自动剪枝、自动压缩或 /compact".into(),
                kind: SettingItemKind::Toggle { value: cfg.enabled },
            },
            SettingEntry {
                key: "compaction.autoPrune".into(),
                category: "压缩".into(),
                label: "L1 零模型大输出剪枝".into(),
                description:
                    "达到水位时优先对较早轮次大工具输出留头尾+recall句柄，省去 LLM 摘要费用".into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.auto_prune,
                },
            },
            SettingEntry {
                key: "compaction.autoCompact".into(),
                category: "压缩".into(),
                label: "L2 自动检查点摘要".into(),
                description: "L1 剪枝仍超阈值时自动生成软硬分离检查点摘要".into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.auto_compact,
                },
            },
            SettingEntry {
                key: "compaction.reusePrefixCache".into(),
                category: "压缩".into(),
                label: "尾置指令复用前缀缓存".into(),
                description: "摘要请求复用主对话前缀以吃满 Prompt Cache，失败自动回退纯净转写"
                    .into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.reuse_prefix_cache,
                },
            },
            SettingEntry {
                key: "compaction.disableSummaryReasoning".into(),
                category: "压缩".into(),
                label: "摘要请求关闭思考".into(),
                description: "生成压缩摘要时设 reasoning_effort=off，防止隐藏思考耗尽输出预算"
                    .into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.disable_summary_reasoning,
                },
            },
            SettingEntry {
                key: "compaction.enableReadDedup".into(),
                category: "压缩".into(),
                label: "跨压缩感知重复读去重".into(),
                description: "同一压缩代次内重复 read 未改动文件区间时返回简短提示，压后自动失效"
                    .into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.enable_read_dedup,
                },
            },
            SettingEntry {
                key: "compaction.enableWriteAfterCompactGuard".into(),
                category: "压缩".into(),
                label: "压后陈旧整文件写保护".into(),
                description:
                    "拦截对压缩前 read 过的已存在文件直接整文件 write，始终放行 edit 与新建".into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.enable_write_after_compact_guard,
                },
            },
            SettingEntry {
                key: "compaction.softWindowCap".into(),
                category: "压缩".into(),
                label: "超大窗口软顶".into(),
                description:
                    "为 1M 等超大窗口模型设置触发软上限（默认 380k），保持注意力与缓存效率".into(),
                kind: SettingItemKind::Choice {
                    options: cap_opts.iter().map(|s| s.to_string()).collect(),
                    selected: choice_index(
                        &cap_opts,
                        format_soft_window_cap_label(cfg.soft_window_cap),
                    ),
                },
            },
        ]);
    }

    if let Ok(plugins) = ctx.spine_plugins() {
        for module in plugins.optional_modules() {
            items.push(SettingEntry {
                key: format!("plugin.{}", module.id),
                category: "模块".into(),
                label: format!("{} ({})", module.label, module.id),
                description: module.description,
                kind: SettingItemKind::Toggle {
                    value: module.enabled,
                },
            });
        }
    }

    items
}

async fn apply_module_toggle_setting(
    ctx: &Context,
    app: &mut AppState,
    settings_path: Option<&Path>,
    plugin_id: &str,
    enabled: bool,
) {
    if app.is_busy || app.is_bash_running || app.is_compacting {
        app.overlay
            .set_setting_toggle(&format!("plugin.{plugin_id}"), !enabled);
        let msg = "当前有正在执行的任务，请等待完成后再切换模块";
        app.push_notice(NoticeLevel::Warn, msg);
        app.set_flash(msg);
        app.overlay
            .refresh_settings(build_settings_entries(ctx, app));
        return;
    }

    let Ok(plugins) = ctx.spine_plugins() else {
        app.push_notice(NoticeLevel::Error, "spinePlugins 控制器服务不可用");
        app.overlay
            .refresh_settings(build_settings_entries(ctx, app));
        return;
    };

    let label = plugins
        .get(plugin_id)
        .map(|m| format!("{} ({})", m.label, m.id))
        .unwrap_or_else(|| plugin_id.to_string());

    match plugins.set_enabled(plugin_id, enabled).await {
        Ok(_) => {
            let disabled = plugins.disabled_ids();
            if let Some(path) = settings_path {
                if let Err(err) = persist_disabled_plugins(path, &disabled) {
                    app.push_notice(
                        NoticeLevel::Warn,
                        format!("模块 `{plugin_id}` 已实时切换，但写入 settings.json 失败：{err}"),
                    );
                }
            } else {
                app.push_notice(
                    NoticeLevel::Warn,
                    format!(
                        "模块 `{plugin_id}` 已实时切换，但无法解析 ~/.mizar/settings.json 路径"
                    ),
                );
            }
            app.user_skill_names = collect_user_skill_names(ctx, &app.workspace).await;
            populate_side_panel_data(ctx, app).await;
            app.overlay
                .refresh_settings(build_settings_entries(ctx, app));
            let state_word = if enabled { "启用" } else { "禁用" };
            let msg = format!("已{state_word}模块 {label}");
            app.set_flash(format!("✓ {msg}"));
            app.push_notice(NoticeLevel::Info, msg);
        }
        Err(err) => {
            app.overlay
                .refresh_settings(build_settings_entries(ctx, app));
            app.push_notice(
                NoticeLevel::Error,
                format!("切换模块 `{plugin_id}` 失败：{err}"),
            );
            app.set_flash(format!("切换模块 `{plugin_id}` 失败"));
        }
    }
}

async fn apply_host_setting_change(
    ctx: &Context,
    app: &mut AppState,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    key: &str,
    value: &str,
) {
    match key {
        "ui.screenMode" => {
            let next_mode = if value == "inline" {
                ScreenMode::Inline
            } else {
                ScreenMode::Fullscreen
            };
            if next_mode != app.screen_mode {
                app.screen_mode = next_mode;
                app.selection.clear();
                app.item_cache.clear();
                let _ = apply_terminal_modes(next_mode, app.mouse_capture_enabled);
                if let Ok(rebuilt) = create_terminal(next_mode) {
                    *terminal = rebuilt;
                    if next_mode == ScreenMode::Fullscreen {
                        app.scroll_to_bottom();
                    }
                }
            }
        }
        "ui.mouseCapture" => {
            let enabled = value == "true";
            app.mouse_capture_enabled = enabled;
            let _ = set_mouse_capture_enabled(enabled);
        }
        "ui.sidePanelOpen" => {
            if app.side_panel.open {
                populate_side_panel_data(ctx, app).await;
            }
        }
        k if k.starts_with("compaction.") => {
            if let Ok(comp) = ctx.compaction() {
                let flag = value == "true";
                comp.update_config(|cfg| match k {
                    "compaction.enabled" => cfg.enabled = flag,
                    "compaction.autoPrune" => cfg.auto_prune = flag,
                    "compaction.autoCompact" => cfg.auto_compact = flag,
                    "compaction.reusePrefixCache" => cfg.reuse_prefix_cache = flag,
                    "compaction.disableSummaryReasoning" => cfg.disable_summary_reasoning = flag,
                    "compaction.enableReadDedup" => cfg.enable_read_dedup = flag,
                    "compaction.enableWriteAfterCompactGuard" => {
                        cfg.enable_write_after_compact_guard = flag
                    }
                    "compaction.softWindowCap" => {
                        cfg.soft_window_cap = parse_soft_window_cap_label(value)
                    }
                    _ => {}
                });
            }
        }
        _ => {}
    }
}

fn upsert_nested_json(
    root: &mut serde_json::Map<String, serde_json::Value>,
    section: &str,
    field: &str,
    val: serde_json::Value,
) {
    let entry = root
        .entry(section.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        *entry = serde_json::Value::Object(serde_json::Map::new());
    }
    if let Some(obj) = entry.as_object_mut() {
        obj.insert(field.to_string(), val);
    }
}

fn persist_setting_change(key: &str, value: &str) -> Result<(), String> {
    let path = resolve_global_settings_path()
        .ok_or_else(|| "无法解析 ~/.mizar/settings.json 路径".to_string())?;
    patch_settings_file(&path, |root| match key {
        "reasoningEffort" => {
            if value == "default" {
                root.remove("reasoningEffort");
            } else {
                root.insert(
                    "reasoningEffort".to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
        "ui.theme" => upsert_nested_json(
            root,
            "ui",
            "theme",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.screenMode" => upsert_nested_json(
            root,
            "ui",
            "screenMode",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.approvalMode" => upsert_nested_json(
            root,
            "ui",
            "approvalMode",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.showThinking" => upsert_nested_json(
            root,
            "ui",
            "showThinking",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.verboseExpanded" => upsert_nested_json(
            root,
            "ui",
            "verboseExpanded",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.sidePanelOpen" => upsert_nested_json(
            root,
            "ui",
            "sidePanelOpen",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.mouseCapture" => upsert_nested_json(
            root,
            "ui",
            "mouseCapture",
            serde_json::Value::Bool(value == "true"),
        ),
        "compaction.softWindowCap" => upsert_nested_json(
            root,
            "compaction",
            "softWindowCap",
            serde_json::Value::Number(serde_json::Number::from(parse_soft_window_cap_label(value))),
        ),
        k if let Some(field) = k.strip_prefix("compaction.") => upsert_nested_json(
            root,
            "compaction",
            field,
            serde_json::Value::Bool(value == "true"),
        ),
        _ => {}
    })
}

fn populate_tools_and_mcps(ctx: &Context, app: &mut AppState) {
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
    } else {
        app.side_panel_data.tools.clear();
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
    } else {
        app.side_panel_data.mcps.clear();
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
    } else {
        app.side_panel_data.instructions.clear();
    }
    let (git_changes, git_diff_stat) = collect_git_summary(&app.workspace).await;
    app.side_panel_data.git_changes = git_changes;
    app.side_panel_data.git_diff_stat = git_diff_stat;

    populate_tools_and_mcps(ctx, app);

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
    provider.models.iter().find(|m| m.id == spec).or_else(|| {
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

    #[tokio::test]
    async fn settings_module_entries_toggle_persists_disabled_plugins_and_refreshes_runtime_state()
    {
        use mizar_agent_spine::{AgentSpineConfig, AgentSpinePlugin, CompactionContextExt};
        use mizar_session::SessionContextExt;
        use mizar_tui_ui::overlay::{OverlayKind, SettingItemKind};

        let tmp = std::env::temp_dir().join(format!("mizar_tui_modules_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let ws = tmp.join("ws");
        std::fs::create_dir_all(ws.join(".git")).unwrap();
        std::fs::create_dir_all(ws.join(".agents/skills/tui-demo")).unwrap();
        std::fs::write(ws.join("AGENTS.md"), "# TUI Rule\nKeep clean\n").unwrap();
        std::fs::write(
            ws.join(".agents/skills/tui-demo/SKILL.md"),
            "---\nname: tui-demo\ndescription: TUI demo skill\n---\nDEMO\n",
        )
        .unwrap();
        let settings_path = tmp.join(".mizar/settings.json");
        std::fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
        std::fs::write(&settings_path, r#"{"defaultModel":"gpt-4o"}"#).unwrap();

        let ctx = cordis::Context::new();
        let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["compaction"]);
        if let mizar_agent_spine::AgentToolsMode::Coding { ref mut mcp, .. } = cfg.tools_mode {
            mcp.global_config_dir = Some(tmp.join(".mizar"));
        }
        ctx.plugin(AgentSpinePlugin, cfg).await.unwrap();
        super::wait_for_services(&ctx).await.unwrap();

        let session = ctx.sessions().unwrap().create(None);
        let mut app = AppState::new(ws.clone(), session.id.clone(), Vec::new(), None, None, None);
        app.user_skill_names = super::collect_user_skill_names(&ctx, &ws).await;
        assert!(app.user_skill_names.contains(&"tui-demo".to_string()));
        super::populate_side_panel_data(&ctx, &mut app).await;
        assert!(!app.side_panel_data.instructions.is_empty());
        assert!(!app.side_panel_data.tools.is_empty());

        // 启动时 compaction 被屏蔽：/settings 中不含 [压缩] 配置项，且 plugin.compaction 为 false
        let initial_entries = super::build_settings_entries(&ctx, &app);
        assert!(
            !initial_entries.iter().any(|e| e.category == "压缩"),
            "compaction entries must be omitted while module is disabled"
        );
        let module_keys: Vec<String> = initial_entries
            .iter()
            .filter(|e| e.category == "模块")
            .map(|e| e.key.clone())
            .collect();
        assert!(module_keys.contains(&"plugin.compaction".to_string()));
        assert!(module_keys.contains(&"plugin.skills".to_string()));
        assert!(module_keys.contains(&"plugin.instructions".to_string()));
        app.overlay
            .open_settings(initial_entries, None, settings_path.display().to_string());

        // 1. 运行期启用 compaction -> 服务上线 + 弹窗自动包含 [压缩] + 写入 disabledPlugins: []
        super::apply_module_toggle_setting(
            &ctx,
            &mut app,
            Some(&settings_path),
            "compaction",
            true,
        )
        .await;
        assert!(ctx.compaction().is_ok());
        match &app.overlay.active {
            Some(OverlayKind::SettingsModal { items, .. }) => {
                assert!(items.iter().any(|e| e.key == "compaction.enabled"));
                let entry = items.iter().find(|e| e.key == "plugin.compaction").unwrap();
                assert_eq!(entry.kind, SettingItemKind::Toggle { value: true });
            }
            _ => panic!("expected SettingsModal open after module toggle"),
        }
        let persisted_1: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(persisted_1["defaultModel"], "gpt-4o");
        assert_eq!(persisted_1["disabledPlugins"], serde_json::json!([]));

        // 2. 禁用 skills / instructions / compaction -> skills 补全清空、侧栏指令清空、[压缩] 移除、排序去重落盘
        for id in ["skills", "instructions", "compaction"] {
            super::apply_module_toggle_setting(&ctx, &mut app, Some(&settings_path), id, false)
                .await;
        }
        assert!(app.user_skill_names.is_empty());
        assert!(app.side_panel_data.instructions.is_empty());
        assert!(ctx.compaction().is_err());
        let after_disable_entries = super::build_settings_entries(&ctx, &app);
        assert!(!after_disable_entries.iter().any(|e| e.category == "压缩"));
        let persisted_2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(
            persisted_2["disabledPlugins"],
            serde_json::json!(["compaction", "instructions", "skills"])
        );

        // 3. 运行期繁忙保护：is_compacting 时拒绝启用 skills，状态与 settings.json 保持不变
        app.is_compacting = true;
        super::apply_module_toggle_setting(&ctx, &mut app, Some(&settings_path), "skills", true)
            .await;
        assert!(ctx.get::<mizar_skills::SkillRegistry>("skills").is_none());
        let persisted_3: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(
            persisted_3["disabledPlugins"],
            serde_json::json!(["compaction", "instructions", "skills"])
        );

        // 4. process_ui_event 收到 ToolsUpdated 应触发 RefreshSidePanelTools
        let mut active_cancel = None;
        let mut pending_steer = None;
        let mut user_cancelled = false;
        let act = process_ui_event(
            &mut app,
            &mut active_cancel,
            &mut pending_steer,
            &mut user_cancelled,
            UiEvent::ToolsUpdated,
        );
        assert_eq!(act, AppAction::RefreshSidePanelTools);

        ctx.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
