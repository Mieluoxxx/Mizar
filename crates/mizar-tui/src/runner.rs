//! 异步运行循环：将 Cordis / AgentLoop / Trajectory 事件接入终端界面。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local};
use cordis::{
    BoxFuture, Cleanup, Context, CordisError, DisposableHandle, LogLevel, LogRecord, LogSink, Next,
};
use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind, MouseEventKind};
use futures::StreamExt;
use mizar_agent_loop::{
    AgentLoopContextExt, AgentReasoningDeltaEvent, AgentStatus, AgentStatusEvent,
    AgentStepEndEvent, AgentStepFirstTokenEvent, AgentStepStartEvent, AgentTextDeltaEvent,
    AgentToolCallEvent, AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent,
    AgentTurnStartEvent, AgentUsageEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_STATUS,
    EVENT_AGENT_STEP_END, EVENT_AGENT_STEP_FIRST_TOKEN, EVENT_AGENT_STEP_START,
    EVENT_AGENT_TEXT_DELTA, EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT,
    EVENT_AGENT_TOOL_UPDATE, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_START, EVENT_AGENT_USAGE,
    RunOptions, StepOutcome, TurnEndReason,
};
use mizar_agent_spine::{
    CompactionContextExt, CompactionEndEvent, CompactionError, CompactionProgressEvent,
    CompactionStartEvent, CompactionTrigger, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS,
    EVENT_COMPACTION_START, ManualCompactOptions, SpinePluginsContextExt,
    normalize_session_title_thinking_level, patch_settings_file, persist_disabled_plugins,
    resolve_global_settings_path, strip_summary_wrapper,
};
use mizar_instructions::AgentInstructions;
use mizar_llm::{ContentBlock, ImageAttachment, LlmContextExt, ReasoningEffort, TokenUsage};
use mizar_mcp::{McpConnectionState, McpManager};
use mizar_session::{
    EVENT_SESSION_TITLE_UPDATED, Session, SessionContextExt, SessionCreateOptions, SessionEvent,
    SessionListQuery, SessionPersistenceContextExt, SessionStore, SessionTitleUpdatedPayload,
    SessionTraceEvent, TraceSeverity,
};
use mizar_session_title::{SessionTitleContextExt, parse_model_ref};
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
    AppAction, AppState, ApprovalMode, ChatItem, LocalBashOutput, NoticeLevel, SessionReplayItem,
    SessionSelection, SessionViewData, UiAgentStatus, UiEvent, UiLogLevel, UiLogRecord,
    UiStepOutcome, UiTurnEndReason, abbreviate_home_path, format_tokens, short_id,
    truncate_display,
};
use mizar_tui_ui::emphasis::{EmphasisConfig, reply_guidance_prompt};
use mizar_tui_ui::overlay::{
    ApprovalDecision, SessionPickerItem, SessionTreeEntry, SettingEntry, SettingItemKind,
    SkillPickerItem,
};
use mizar_tui_ui::side_panel::{
    FileTouchRow, InstructionSummaryRow, McpSummaryRow, ToolSummaryRow,
};
use mizar_tui_ui::terminal::{
    BEGIN_SYNCHRONIZED_OUTPUT, END_SYNCHRONIZED_OUTPUT, ProgramStatusState, ScreenMode,
    TerminalGuard, apply_terminal_modes, emit_program_status, resume_after_external_editor,
    set_mouse_capture_enabled, suspend_for_external_editor,
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

use crate::{clipboard, images, mentions};

/// 把单帧 `BSU + ratatui 差分序列 + ESU` 压进同一个缓冲 Writer 并阻止中间提前 flush，
/// 让终端整帧通过单次底层 `write()` 系统调用原子落盘，减少切帧闪烁与 syscall 开销。
struct SyncWriterInner<W: std::io::Write> {
    inner: W,
    defer_flush: bool,
}

impl<W: std::io::Write> SyncWriterInner<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            defer_flush: false,
        }
    }

    fn begin_sync_frame(&mut self) -> std::io::Result<()> {
        self.defer_flush = true;
        self.inner.write_all(BEGIN_SYNCHRONIZED_OUTPUT.as_bytes())
    }

    fn end_sync_frame(&mut self) -> std::io::Result<()> {
        self.defer_flush = false;
        self.inner.write_all(END_SYNCHRONIZED_OUTPUT.as_bytes())?;
        self.inner.flush()
    }
}

impl<W: std::io::Write> std::io::Write for SyncWriterInner<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.defer_flush {
            return Ok(());
        }
        self.inner.flush()
    }
}

#[derive(Clone)]
struct SharedSyncWriter {
    inner: Arc<Mutex<SyncWriterInner<std::io::BufWriter<std::io::Stdout>>>>,
}

impl std::io::Write for SharedSyncWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.lock().unwrap().flush()
    }
}

struct ManagedTerminal {
    terminal: Terminal<CrosstermBackend<SharedSyncWriter>>,
    sync: SharedSyncWriter,
}

impl ManagedTerminal {
    fn draw_synced<F>(&mut self, render_cb: F) -> std::io::Result<()>
    where
        F: FnOnce(&mut ratatui::Frame<'_>),
    {
        let _ = self.sync.inner.lock().unwrap().begin_sync_frame();
        let draw_res = self.terminal.draw(render_cb);
        let end_res = self.sync.inner.lock().unwrap().end_sync_frame();
        draw_res?;
        end_res
    }
}

type BufferedTerminal = ManagedTerminal;

async fn poll_term_event(stream: Option<&mut EventStream>) -> Option<std::io::Result<TermEvent>> {
    match stream {
        Some(s) => s.next().await,
        None => std::future::pending().await,
    }
}

fn handle_term_event_dirty(app: &mut AppState, ev: TermEvent) -> (AppAction, bool) {
    match ev {
        TermEvent::Key(key) => {
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                return (AppAction::None, false);
            }
            (app.handle_key(key), true)
        }
        TermEvent::Mouse(mouse) => {
            if matches!(mouse.kind, MouseEventKind::Moved) {
                let prev = (app.side_panel.hovered_divider, app.hovered_rail_row);
                let act = app.handle_mouse(mouse);
                let dirty = act != AppAction::None
                    || app.overlay.is_open()
                    || app.trajectory_scene.open
                    || (app.side_panel.hovered_divider, app.hovered_rail_row) != prev;
                return (act, dirty);
            }
            (app.handle_mouse(mouse), true)
        }
        TermEvent::Paste(data) => {
            app.handle_paste(&data);
            (AppAction::None, true)
        }
        TermEvent::Resize(_, _) => (AppAction::None, true),
        TermEvent::FocusGained => {
            app.terminal_focused = true;
            (AppAction::None, false)
        }
        TermEvent::FocusLost => {
            app.terminal_focused = false;
            (AppAction::None, false)
        }
    }
}

fn process_loop_ui_msg_dirty(
    app: &mut AppState,
    task_state: &mut RunnerTaskState,
    msg: LoopUiMsg,
) -> (AppAction, bool) {
    let is_delta = matches!(
        msg,
        LoopUiMsg::Bus(UiEvent::TextDelta { .. } | UiEvent::ReasoningDelta { .. })
    );
    let had_assistant = matches!(app.items.last(), Some(ChatItem::Assistant { .. }));
    let prev_reveal_gen = app.reveal.generation();

    let action = process_loop_ui_msg(app, task_state, msg);
    if action != AppAction::None {
        return (action, true);
    }

    let can_defer_to_reveal = is_delta
        && app.reveal.enabled
        && app.is_busy()
        && had_assistant
        && app.reveal.generation() == prev_reveal_gen;
    (AppAction::None, !can_defer_to_reveal)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeTarget {
    Picker,
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
    /// 模型是否由环境变量 `MODEL` 显式指定（优先级最高，会压过会话记录）。
    pub model_from_env: bool,
    /// 模型是否由 `settings.json` 的 `defaultModel` 指定（优先级低于会话记录）。
    pub model_from_settings: bool,
    /// 强度是否由 `settings.json` 的 `reasoningEffort` 指定（优先级低于会话记录）。
    pub reasoning_effort_from_settings: bool,
    pub approval_mode: ApprovalMode,
    pub theme_preset: Option<ThemePreset>,
    pub screen_mode: ScreenMode,
    pub show_thinking: Option<bool>,
    pub verbose_expanded: Option<bool>,
    pub side_panel_open: Option<bool>,
    pub mouse_capture: Option<bool>,
    pub resume_session: Option<ResumeTarget>,
    pub session_db_path: Option<PathBuf>,
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
            model_from_env: false,
            model_from_settings: false,
            reasoning_effort_from_settings: false,
            approval_mode: ApprovalMode::Auto,
            theme_preset: None,
            screen_mode: ScreenMode::Fullscreen,
            show_thinking: None,
            verbose_expanded: None,
            side_panel_open: None,
            mouse_capture: None,
            resume_session: None,
            session_db_path: None,
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
        TurnEndReason::Truncated => UiTurnEndReason::Truncated,
        TurnEndReason::MaxSteps => UiTurnEndReason::MaxSteps,
        TurnEndReason::Cancelled => UiTurnEndReason::Cancelled,
        TurnEndReason::Error { message } => UiTurnEndReason::Error { message },
    }
}

fn to_ui_step_outcome(outcome: &StepOutcome) -> UiStepOutcome {
    match outcome {
        StepOutcome::Completed => UiStepOutcome::Completed,
        StepOutcome::ToolCalls => UiStepOutcome::ToolCalls,
        StepOutcome::MaxTokens => UiStepOutcome::MaxTokens,
        StepOutcome::Cancelled => UiStepOutcome::Cancelled,
        StepOutcome::Error { message } => UiStepOutcome::Error {
            message: message.clone(),
        },
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

/// 选择解析的输入：各来源的值与「是否显式指定」标记。
///
/// 优先级链（高的先，逐字段独立判定）：
/// ```text
/// 1. 环境变量 MODEL            → 压过一切，包括会话记录
/// 2. 会话记录（仅 resume）     → 恢复语义：会话当时的模型优先于全局配置
/// 3. settings.json 的默认值
/// 4. 最近一个会话的记录（仅冷启动）  → 「沿用上次」
/// 5. 现有推导（models.json 字典序 + models[0]）
/// ```
#[derive(Debug, Clone, Default)]
pub struct SelectionSources {
    /// 环境变量是否指定了模型。
    pub model_from_env: bool,
    /// `settings.json` 是否写了 `defaultModel`。
    pub model_from_settings: bool,
    /// `settings.json` 是否写了 `reasoningEffort`。
    pub reasoning_effort_from_settings: bool,
    /// 启动时已解析出的默认选择（来自 `models.json` 推导或 settings）。
    pub startup: SessionSelection,
    /// resume 时该会话自己的记录。
    pub session: Option<SessionSelection>,
    /// 当前工作区最近一个会话的记录（冷启动用）。
    pub latest_workspace: Option<SessionSelection>,
}

/// 按文档化的优先级链算出应采用的模型与思考强度。
///
/// 返回 `None` 表示启动值已经是最优解，调用方无需覆盖。
/// 返回 `Some` 时三个字段各自可能为 `None` —— 表示该字段本次没有更高优先级的来源。
pub fn resolve_selection(sources: &SelectionSources, resumed: bool) -> Option<SessionSelection> {
    // 环境变量压过一切，直接不覆盖
    if sources.model_from_env {
        return None;
    }

    // resume：会话记录优先于 settings；冷启动：settings 优先于最近会话记录
    let session = resumed.then_some(sources.session.as_ref()).flatten();
    let latest = (!resumed)
        .then_some(sources.latest_workspace.as_ref())
        .flatten();

    let model = session
        .and_then(|s| s.model.clone())
        .or_else(|| {
            (!sources.model_from_settings)
                .then(|| latest.and_then(|s| s.model.clone()))
                .flatten()
        })
        .or_else(|| sources.startup.model.clone());
    // provider 与 model 同源优先级，避免 provider/model 交错
    let provider = session
        .and_then(|s| s.provider.clone())
        .or_else(|| {
            (!sources.model_from_settings)
                .then(|| latest.and_then(|s| s.provider.clone()))
                .flatten()
        })
        .or_else(|| sources.startup.provider.clone());
    let effort = session
        .and_then(|s| s.reasoning_effort)
        .or_else(|| {
            (!sources.reasoning_effort_from_settings)
                .then(|| latest.and_then(|s| s.reasoning_effort))
                .flatten()
        })
        .or(sources.startup.reasoning_effort);

    let resolved = SessionSelection {
        provider,
        model,
        reasoning_effort: effort,
    };
    (resolved != sources.startup).then_some(resolved)
}

/// 把会话层的选择投影为 UI 层形态。两个 crate 互不依赖（`mizar-tui-ui` 禁止依赖 `mizar-session`），
/// 所以转换只能放在宿主层。
fn to_ui_selection(sel: mizar_session::SessionSelection) -> SessionSelection {
    SessionSelection {
        provider: sel.provider,
        model: sel.model,
        reasoning_effort: sel.reasoning_effort,
    }
}

/// 取当前工作区最近一个会话记录的选择。
///
/// 只在 `settings.json` 未指定模型时使用，且任务本就是「跨会话沿用上次」而不是「跨工作区」，
/// 所以按 `cwd` 过滤（`idx_sessions_cwd_updated` 索引直出）。
pub fn latest_workspace_selection(
    sessions: &mizar_session::SessionStore,
    workspace: &Path,
) -> Option<mizar_session::SessionSelection> {
    let latest = sessions
        .list_headers(&SessionListQuery {
            cwd: Some(workspace.display().to_string()),
            limit: Some(1),
        })
        .ok()?
        .into_iter()
        .next()?;
    sessions.peek_selection(&latest.id)
}

fn accumulate_usage(acc: &mut TokenUsage, usage: &TokenUsage) {
    acc.input_tokens = acc.input_tokens.saturating_add(usage.input_tokens);
    acc.output_tokens = acc.output_tokens.saturating_add(usage.output_tokens);
    if let Some(cr) = usage.cache_read_tokens {
        acc.cache_read_tokens = Some(acc.cache_read_tokens.unwrap_or(0).saturating_add(cr));
    }
    if let Some(cw) = usage.cache_write_tokens {
        acc.cache_write_tokens = Some(acc.cache_write_tokens.unwrap_or(0).saturating_add(cw));
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn format_local_datetime_ms(ts_ms: u128) -> String {
    if ts_ms == 0 {
        return "未知".to_string();
    }
    let Ok(ms) = i64::try_from(ts_ms) else {
        return "未知".to_string();
    };
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(ms)
        .map(|utc| {
            utc.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|| "未知".to_string())
}

fn format_relative_activity(updated_at_ms: u128, now_ms: u128) -> String {
    if updated_at_ms == 0 {
        return "未知".to_string();
    }
    let elapsed_secs = now_ms.saturating_sub(updated_at_ms) / 1_000;
    if elapsed_secs < 60 {
        "刚刚".to_string()
    } else if elapsed_secs < 3_600 {
        format!("{} 分钟前", elapsed_secs / 60)
    } else if elapsed_secs < 86_400 {
        format!("{} 小时前", elapsed_secs / 3_600)
    } else {
        format!("{} 天前", elapsed_secs / 86_400)
    }
}

fn format_context_window(window: u64) -> String {
    if window >= 1_000_000 && window.is_multiple_of(1_000_000) {
        format!("{}M", window / 1_000_000)
    } else if (1_000..1_000_000).contains(&window) && window.is_multiple_of(1_000) {
        format!("{}k", window / 1_000)
    } else {
        format_tokens(window)
    }
}

fn join_mid_dot(parts: &[String]) -> String {
    let mut out = String::new();
    for part in parts {
        if !out.is_empty() {
            if out.ends_with('）') {
                out.push_str("· ");
            } else {
                out.push_str(" · ");
            }
        }
        out.push_str(part);
    }
    out
}

/// 构造 `/session` 命令的只读会话身份与账目报告。
pub fn build_session_info_report(
    ctx: &Context,
    config: &TuiConfig,
    session: &Session,
    app: &AppState,
) -> String {
    build_session_info_report_at(
        session,
        app,
        ctx.session_persistence().is_some(),
        config.session_db_path.as_deref(),
        now_millis(),
    )
}

/// 以显式持久化标记、路径与当前时间戳构造 `/session` 报告（便于确定性测试）。
pub fn build_session_info_report_at(
    session: &Session,
    app: &AppState,
    persisted: bool,
    session_db_path: Option<&Path>,
    now_ms: u128,
) -> String {
    let header = session.header();

    let mut total_events = 0u64;
    let mut surface_events = 0u64;
    let mut user_count = 0u64;
    let mut assistant_count = 0u64;
    let mut tool_result_count = 0u64;
    let mut tool_error_count = 0u64;
    let mut bash_count = 0u64;
    let mut bash_local_count = 0u64;
    let mut pruned_seqs = HashSet::<u64>::new();
    let mut injection_counts = HashMap::<String, u64>::new();
    let mut compaction_count = 0u64;
    let mut latest_compaction: Option<(u64, Option<u64>, Option<u64>)> = None;
    let mut total_usage = TokenUsage::default();
    let mut latest_context_tokens: Option<u64> = None;
    let mut saw_prune_after_last_context = false;

    session.with_all_entries(|entries| {
        total_events = entries.len() as u64;
        for entry in entries {
            if entry.is_surface() {
                surface_events += 1;
            }
            match &entry.event {
                SessionEvent::User { .. } => user_count += 1,
                SessionEvent::ContextNotice { source, .. } => {
                    *injection_counts.entry(source.clone()).or_insert(0) += 1;
                }
                SessionEvent::Assistant { usage, .. } => {
                    assistant_count += 1;
                    if let Some(u) = usage {
                        accumulate_usage(&mut total_usage, u);
                        let prompt = u.prompt_tokens();
                        if prompt > 0 {
                            latest_context_tokens = Some(prompt);
                            saw_prune_after_last_context = false;
                        }
                    }
                }
                SessionEvent::ToolResult { is_error, .. } => {
                    tool_result_count += 1;
                    if *is_error {
                        tool_error_count += 1;
                    }
                }
                SessionEvent::Compaction {
                    first_kept_seq,
                    tokens_before,
                    tokens_after,
                    ..
                } => {
                    compaction_count += 1;
                    latest_compaction = Some((*first_kept_seq, *tokens_before, *tokens_after));
                    if let Some(after) = tokens_after.filter(|&v| v > 0) {
                        latest_context_tokens = Some(after);
                        saw_prune_after_last_context = false;
                    }
                }
                SessionEvent::ToolResultPrune { pruned } => {
                    for item in pruned {
                        pruned_seqs.insert(item.target_seq);
                    }
                    saw_prune_after_last_context = true;
                }
                SessionEvent::BashExecution {
                    exclude_from_context,
                    ..
                } => {
                    bash_count += 1;
                    if *exclude_from_context {
                        bash_local_count += 1;
                    }
                }
                SessionEvent::Trace { .. } | SessionEvent::Selection { .. } => {}
            }
        }
    });

    let log_only_events = total_events.saturating_sub(surface_events);
    let title = header
        .title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("（未命名）");
    let cwd = header
        .cwd
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("（未记录）");
    let created_str = format_local_datetime_ms(header.created_at_ms);
    let activity_str = format_relative_activity(header.updated_at_ms, now_ms);

    let mut lines = vec![
        "会话信息".to_string(),
        String::new(),
        format!("ID       {}", header.id),
        format!("标题     {title}"),
        format!("工作区   {cwd}"),
        format!("创建     {created_str}  ·  最后活动 {activity_str}"),
    ];

    if let Some(parent_id) = header.parent_id.as_deref().filter(|s| !s.is_empty()) {
        let lineage = match header.fork_seq {
            Some(seq) => format!("分叉自 {} @ #{seq}", short_id(parent_id)),
            None => format!("分叉自 {}", short_id(parent_id)),
        };
        lines.push(format!("谱系     {lineage}"));
    }

    lines.push(String::new());

    let storage = if persisted {
        match session_db_path {
            Some(path) => {
                format!(
                    "SQLite · {}",
                    abbreviate_home_path(&path.display().to_string())
                )
            }
            None => "SQLite（已持久化）".to_string(),
        }
    } else {
        "仅内存（未持久化）".to_string()
    };
    lines.push(format!("存储     {storage}"));
    lines.push(format!(
        "事件     {total_events} 条（surface {surface_events} / log-only {log_only_events}）"
    ));
    lines.push(format!(
        "对话     user {user_count}  ·  assistant {assistant_count}"
    ));

    let mut tool_parts = Vec::new();
    if tool_error_count > 0 {
        tool_parts.push(format!(
            "tool-result {tool_result_count}（err {tool_error_count}）"
        ));
    } else {
        tool_parts.push(format!("tool-result {tool_result_count}"));
    }
    if bash_count > 0 {
        if bash_local_count > 0 {
            tool_parts.push(format!("bash {bash_count}（本地 {bash_local_count}）"));
        } else {
            tool_parts.push(format!("bash {bash_count}"));
        }
    }
    if !pruned_seqs.is_empty() {
        tool_parts.push(format!("prune {}", pruned_seqs.len()));
    }
    lines.push(format!("工具     {}", join_mid_dot(&tool_parts)));

    const KNOWN_INJECTION_SOURCES: [(&str, &str); 5] = [
        ("agent-instructions", "instructions"),
        ("skill-catalog", "skill-catalog"),
        ("skill-invocation", "skill-invocation"),
        ("file-mentions", "file-mentions"),
        ("repeat-tool-reminder", "repeat-tool-reminder"),
    ];
    let mut inj_parts = Vec::new();
    for (raw_src, label) in KNOWN_INJECTION_SOURCES {
        if let Some(count) = injection_counts.remove(raw_src)
            && count > 0
        {
            inj_parts.push(format!("{label} {count}"));
        }
    }
    let mut extra_sources: Vec<(String, u64)> = injection_counts
        .into_iter()
        .filter(|(_, c)| *c > 0)
        .collect();
    extra_sources.sort_by(|a, b| a.0.cmp(&b.0));
    for (src, count) in extra_sources {
        inj_parts.push(format!("{src} {count}"));
    }
    let inj_summary = if inj_parts.is_empty() {
        "无".to_string()
    } else {
        inj_parts.join(" · ")
    };
    lines.push(format!("注入     {inj_summary}"));

    if let Some((first_kept_seq, tokens_before, tokens_after)) = latest_compaction {
        let mut comp_parts = vec![
            format!("{compaction_count} 次"),
            format!("最新保留自 #{first_kept_seq}"),
        ];
        match (tokens_before, tokens_after) {
            (Some(before), Some(after)) => comp_parts.push(format!(
                "最近 {} → {}",
                format_tokens(before),
                format_tokens(after)
            )),
            (Some(before), None) => comp_parts.push(format!("压前 {}", format_tokens(before))),
            _ => {}
        }
        lines.push(format!("压缩     {}", comp_parts.join(" · ")));
    }

    let prompt_total = total_usage.prompt_tokens();
    let cache_read = total_usage.cache_read_tokens.unwrap_or(0);
    let cache_write = total_usage.cache_write_tokens.unwrap_or(0);
    let cache_str = match (cache_read, cache_write) {
        (0, 0) => String::new(),
        (cr, 0) => format!("（cache {}）", format_tokens(cr)),
        (0, cw) => format!("（cache {} 写）", format_tokens(cw)),
        (cr, cw) => format!(
            "（cache {} 读 / {} 写）",
            format_tokens(cr),
            format_tokens(cw)
        ),
    };
    lines.push(format!(
        "Token    累计 ↑{}{}↓{}",
        format_tokens(prompt_total),
        if cache_str.is_empty() {
            " "
        } else {
            &cache_str
        },
        format_tokens(total_usage.output_tokens),
    ));

    let window = app.active_context_window().max(1);
    let effective_measured = if saw_prune_after_last_context {
        app.last_prompt_tokens.or(latest_context_tokens)
    } else {
        latest_context_tokens.or(app.last_prompt_tokens)
    };
    let ctx_summary = if let Some(measured) = effective_measured {
        let pct = (measured as f64 / window as f64) * 100.0;
        format!(
            "{} / {}（{pct:.1}%）",
            format_tokens(measured),
            format_context_window(window)
        )
    } else if surface_events == 0 {
        format!("0 / {}（0.0%）", format_context_window(window))
    } else {
        let est = app
            .side_panel_data
            .context
            .system_tokens
            .saturating_add(app.side_panel_data.context.user_tokens)
            .saturating_add(app.side_panel_data.context.assistant_tokens)
            .saturating_add(app.side_panel_data.context.reasoning_tokens)
            .saturating_add(app.side_panel_data.context.tool_tokens);
        let pct = (est as f64 / window as f64) * 100.0;
        format!(
            "~{} / {}（{pct:.1}%）",
            format_tokens(est),
            format_context_window(window)
        )
    };
    lines.push(format!("上下文   {ctx_summary}"));

    lines.join("\n")
}

/// 将持久化会话日志投影为纯 UI 可重放的 `SessionViewData`（含 `!` 与 `!!` 本地命令记录）。
pub fn project_session_view(
    session: &Session,
    finish_notice: impl Into<String>,
) -> SessionViewData {
    let header = session.header();
    let mut total_usage = TokenUsage::default();
    let mut saw_usage = false;
    let mut last_prompt_tokens: Option<u64> = None;
    let mut items = Vec::new();
    for entry in session.all_entries() {
        match entry.event {
            SessionEvent::User { text, images } => {
                let display = if !images.is_empty() && !text.contains("[Image #") {
                    if text.is_empty() {
                        format!("[图片 {} 张]", images.len())
                    } else {
                        format!("[图片 {} 张]\n{text}", images.len())
                    }
                } else {
                    text
                };
                items.push(SessionReplayItem::User { text: display });
            }
            SessionEvent::ContextNotice {
                source,
                text,
                images,
            } => {
                let display = if !images.is_empty() && text.is_empty() {
                    format!("<attached-image count=\"{}\" />", images.len())
                } else {
                    text
                };
                items.push(SessionReplayItem::ContextNotice {
                    source,
                    text: display,
                });
            }
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                tokens_after,
                ..
            } => {
                if let Some(after) = tokens_after.filter(|&v| v > 0) {
                    last_prompt_tokens = Some(after);
                }
                items.push(SessionReplayItem::Compaction {
                    summary: strip_summary_wrapper(&summary),
                    first_kept_seq,
                });
            }
            SessionEvent::Assistant { blocks, usage, .. } => {
                if let Some(u) = usage {
                    saw_usage = true;
                    accumulate_usage(&mut total_usage, &u);
                    let prompt = u.prompt_tokens();
                    if prompt > 0 {
                        last_prompt_tokens = Some(prompt);
                    }
                }
                items.push(SessionReplayItem::Assistant { blocks });
            }
            SessionEvent::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => items.push(SessionReplayItem::ToolResult {
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
            } => items.push(SessionReplayItem::BashExecution {
                command,
                output,
                exit_code,
                cancelled,
                exclude_from_context,
            }),
            SessionEvent::Trace { detail } => match detail {
                SessionTraceEvent::TurnNotice { severity, text, .. } => {
                    let level = match severity {
                        TraceSeverity::Info => NoticeLevel::Info,
                        TraceSeverity::Warn => NoticeLevel::Warn,
                        TraceSeverity::Error => NoticeLevel::Error,
                    };
                    items.push(SessionReplayItem::Notice { level, text });
                }
                SessionTraceEvent::TurnEnd { reason, .. } if reason == "truncated" => {
                    items.push(SessionReplayItem::Notice {
                        level: NoticeLevel::Warn,
                        text: "输出因达到 max_tokens 上限被截断，回复可能不完整".to_string(),
                    });
                }
                _ => {}
            },
            SessionEvent::ToolResultPrune { .. } | SessionEvent::Selection { .. } => {}
        }
    }
    if saw_usage {
        total_usage.total_tokens = Some(total_usage.grand_total());
    }
    SessionViewData {
        session_id: session.id.clone(),
        session_title: header.title,
        items,
        finish_notice: finish_notice.into(),
        selection: session.latest_selection().map(|sel| SessionSelection {
            provider: sel.provider,
            model: sel.model,
            reasoning_effort: sel.reasoning_effort,
        }),
        total_usage,
        last_prompt_tokens,
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
            SessionEvent::User { text, images } => {
                let prefix = if images.is_empty() {
                    String::new()
                } else {
                    format!("[图×{}] ", images.len())
                };
                Some(SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "USER".to_string(),
                    summary: format!("{prefix}{}", truncate_display(&text.replace('\n', " "), 64)),
                    depth: 0,
                    can_fork: true,
                })
            }
            SessionEvent::ContextNotice { source, text, .. } => Some(SessionTreeEntry {
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
            SessionEvent::ToolResultPrune { .. }
            | SessionEvent::Selection { .. }
            | SessionEvent::Trace { .. } => None,
        })
        .collect()
}

/// 事件循环内部统一 FIFO 消息信封：保证总线事件（如 `TurnEnd`）与后台任务结束事件（如 `RunFinished`）
/// 严格按投递顺序到达，同时为本地派生任务附带 `(run_id, session_id)` 代次标识，杜绝过期任务或旧会话污染新状态。
#[derive(Debug, Clone)]
enum LoopUiMsg {
    Bus(UiEvent),
    Task {
        run_id: u64,
        session_id: String,
        event: UiEvent,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveTaskKind {
    AgentRun,
    LocalBash,
    ManualCompaction,
}

#[derive(Debug, Clone)]
struct ActiveTask {
    run_id: u64,
    session_id: String,
    kind: ActiveTaskKind,
    cancel: CancellationToken,
    pending_steer: Option<PendingPrompt>,
    user_cancelled: bool,
}

#[derive(Debug, Default)]
struct RunnerTaskState {
    next_run_id: u64,
    active: Option<ActiveTask>,
}

impl RunnerTaskState {
    fn start(&mut self, session_id: String, kind: ActiveTaskKind) -> (u64, CancellationToken) {
        if let Some(prev) = self.active.take() {
            prev.cancel.cancel();
        }
        self.next_run_id = self.next_run_id.wrapping_add(1).max(1);
        let run_id = self.next_run_id;
        let cancel = CancellationToken::new();
        self.active = Some(ActiveTask {
            run_id,
            session_id,
            kind,
            cancel: cancel.clone(),
            pending_steer: None,
            user_cancelled: false,
        });
        (run_id, cancel)
    }

    fn cancel_active(&mut self, steer: Option<PendingPrompt>) {
        if let Some(task) = self.active.as_mut() {
            match steer {
                Some(prompt) => {
                    task.pending_steer = Some(prompt);
                    task.user_cancelled = false;
                }
                None => {
                    task.pending_steer = None;
                    task.user_cancelled = true;
                }
            }
            task.cancel.cancel();
        }
    }

    fn abort_for_session_switch(&mut self) {
        if let Some(prev) = self.active.take() {
            prev.cancel.cancel();
        }
    }

    fn is_current(&self, run_id: u64, msg_session_id: &str, current_session_id: &str) -> bool {
        self.active.as_ref().is_some_and(|cur| {
            cur.run_id == run_id
                && cur.session_id == msg_session_id
                && msg_session_id == current_session_id
        })
    }

    fn cancel_token(&self) -> Option<&CancellationToken> {
        self.active.as_ref().map(|t| &t.cancel)
    }

    fn finish_if_current(
        &mut self,
        run_id: Option<u64>,
        msg_session_id: Option<&str>,
        current_session_id: &str,
        expected_kind: ActiveTaskKind,
    ) -> Option<ActiveTask> {
        let Some(cur) = self.active.as_ref() else {
            return (run_id.is_none()).then(|| ActiveTask {
                run_id: 0,
                session_id: current_session_id.to_string(),
                kind: expected_kind,
                cancel: CancellationToken::new(),
                pending_steer: None,
                user_cancelled: false,
            });
        };
        if let Some(rid) = run_id
            && cur.run_id != rid
        {
            return None;
        }
        if let Some(sid) = msg_session_id
            && (sid != cur.session_id || sid != current_session_id)
        {
            return None;
        }
        if cur.kind != expected_kind {
            return None;
        }
        self.active.take()
    }
}

/// 订阅当前会话相关的 Agent 与轨迹事件并投递到 `tx`。
pub fn subscribe_session_events(
    ctx: &Context,
    session_id_rx: Arc<std::sync::RwLock<String>>,
    tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<Vec<DisposableHandle>, CordisError> {
    subscribe_session_events_into(ctx, session_id_rx, move |ev| {
        let _ = tx.send(ev);
    })
}

fn subscribe_session_events_into<F>(
    ctx: &Context,
    session_id_rx: Arc<std::sync::RwLock<String>>,
    send: F,
) -> Result<Vec<DisposableHandle>, CordisError>
where
    F: Fn(UiEvent) + Clone + Send + Sync + 'static,
{
    let mut handles = Vec::new();
    let is_current = move |sid: &str| *session_id_rx.read().unwrap() == sid;

    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STATUS, move |ev: &AgentStatusEvent| {
            if check(&ev.session.id) {
                send(UiEvent::Status(to_ui_status(ev.status)));
            }
        })?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::TurnStart { turn: ev.turn });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
            if check(&ev.session.id) {
                send(UiEvent::TurnEnd {
                    reason: to_ui_turn_end_reason(ev.reason.clone()),
                });
            }
        })?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_STEP_START, move |ev: &AgentStepStartEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::StepStart { ts_ms: ev.ts_ms });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_AGENT_STEP_FIRST_TOKEN,
            move |ev: &AgentStepFirstTokenEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::FirstToken { ts_ms: ev.ts_ms });
                }
            },
        )?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
            if check(&ev.session.id) {
                send(UiEvent::StepEnd {
                    step: ev.step,
                    ts_ms: ev.ts_ms,
                    outcome: to_ui_step_outcome(&ev.outcome),
                    usage: ev.usage,
                });
            }
        })?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TEXT_DELTA, move |ev: &AgentTextDeltaEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::TextDelta {
                        text: ev.text.clone(),
                        ts_ms: ev.ts_ms,
                    });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_AGENT_REASONING_DELTA,
            move |ev: &AgentReasoningDeltaEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::ReasoningDelta {
                        text: ev.text.clone(),
                        ts_ms: ev.ts_ms,
                    });
                }
            },
        )?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::ToolCall(ev.call.clone()));
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_UPDATE, move |ev: &AgentToolUpdateEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::ToolUpdate {
                        call_id: ev.tool_call_id.clone(),
                        name: ev.name.clone(),
                        preview: ev.preview.clone(),
                    });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TOOL_RESULT, move |ev: &AgentToolResultEvent| {
                if check(&ev.session.id) {
                    send(UiEvent::ToolResult {
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
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_USAGE, move |ev: &AgentUsageEvent| {
            if check(&ev.session.id) {
                send(UiEvent::Usage(ev.usage));
            }
        })?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_TRAJECTORY_ANOMALY,
            move |ev: &TrajectoryAnomalyEvent| {
                if check(&ev.session_id) {
                    send(UiEvent::Anomaly(ev.finding.summary.clone()));
                }
            },
        )?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_COMPACTION_START, move |ev: &CompactionStartEvent| {
                if check(&ev.session_id) {
                    send(UiEvent::CompactionStarted {
                        trigger: ev.trigger.as_str().to_string(),
                    });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(
            EVENT_COMPACTION_PROGRESS,
            move |ev: &CompactionProgressEvent| {
                if check(&ev.session_id) {
                    send(UiEvent::CompactionProgress {
                        stage: ev.stage.clone(),
                        delta_chars: ev.delta_chars,
                    });
                }
            },
        )?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_COMPACTION_END, move |ev: &CompactionEndEvent| {
                if check(&ev.session_id) {
                    send(UiEvent::CompactionFinished {
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
    {
        let send = send;
        let check = is_current;
        handles.push(ctx.on(
            EVENT_SESSION_TITLE_UPDATED,
            move |ev: &SessionTitleUpdatedPayload| {
                if check(&ev.session.id) {
                    send(UiEvent::SessionTitleUpdated(ev.title.clone()));
                }
            },
        )?);
    }

    Ok(handles)
}

fn format_cwd_tail(raw: Option<&str>) -> String {
    let Some(cwd) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return "-".to_string();
    };
    let segs: Vec<&str> = cwd.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    match segs.len() {
        0 => cwd.to_string(),
        1 => segs[0].to_string(),
        n => {
            let two = format!("{}/{}", segs[n - 2], segs[n - 1]);
            if two.chars().count() <= 20 {
                two
            } else {
                segs[n - 1].to_string()
            }
        }
    }
}

fn format_session_updated_at(ms: u128) -> String {
    i64::try_from(ms)
        .ok()
        .filter(|&v| v > 0)
        .and_then(DateTime::from_timestamp_millis)
        .map(|utc| utc.with_timezone(&Local).format("%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "-".to_string())
}

/// 仅通过 `list_headers` 轻量查询非空历史会话列表用于 `/resume` 选择器（不加载事件日志）。
pub fn collect_session_picker_items(sessions: &SessionStore) -> Vec<SessionPickerItem> {
    sessions
        .list_headers(&SessionListQuery {
            cwd: None,
            limit: Some(100),
        })
        .unwrap_or_default()
        .into_iter()
        .filter(|h| h.event_count > 0)
        .map(|h| {
            let title = h
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("(无标题 #{})", short_id(&h.id)));
            SessionPickerItem {
                title,
                cwd_tail: format_cwd_tail(h.cwd.as_deref()),
                event_count: h.event_count,
                updated_at: format_session_updated_at(h.updated_at_ms),
                session_id: h.id,
            }
        })
        .collect()
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
    let mut startup_picker_items: Option<Vec<SessionPickerItem>> = None;
    let (initial_session, resumed) = match &config.resume_session {
        Some(ResumeTarget::Picker) => {
            startup_picker_items = Some(collect_session_picker_items(&sessions));
            (sessions.create(None), false)
        }
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

    let (loop_tx, mut loop_rx) = mpsc::unbounded_channel::<LoopUiMsg>();
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
    let mut handles = {
        let bus_tx = loop_tx.clone();
        subscribe_session_events_into(&ctx, Arc::clone(&active_sid), move |ev| {
            let _ = bus_tx.send(LoopUiMsg::Bus(ev));
        })?
    };

    let mut app = AppState::new(
        config.workspace.clone(),
        active_session.id.clone(),
        providers,
        config.provider.clone(),
        config.model.clone(),
        config.reasoning_effort,
    );
    app.session_title = active_session.header().title;

    let sources = SelectionSources {
        model_from_env: config.model_from_env,
        model_from_settings: config.model_from_settings,
        reasoning_effort_from_settings: config.reasoning_effort_from_settings,
        startup: SessionSelection {
            provider: app.provider.clone(),
            model: app.model.clone(),
            reasoning_effort: app.reasoning_effort,
        },
        session: resumed
            .then(|| active_session.latest_selection())
            .flatten()
            .map(to_ui_selection),
        latest_workspace: if resumed {
            None
        } else {
            latest_workspace_selection(&sessions, &config.workspace).map(to_ui_selection)
        },
    };
    if let Some(resolved) = resolve_selection(&sources, resumed) {
        app.provider = resolved.provider;
        app.model = resolved.model;
        app.reasoning_effort = resolved.reasoning_effort;
        app.push_notice(
            NoticeLevel::Info,
            format!(
                "已沿用{}的模型与思考强度：{}",
                if resumed {
                    "本会话记录"
                } else {
                    "上次会话"
                },
                app.active_model_label()
            ),
        );
    }

    if resumed {
        app.restore_session_view(project_resume_view(&active_session));
    }
    if let Some((lvl, msg)) = resume_notice {
        app.push_notice(lvl, msg);
    }
    if let Some(items) = startup_picker_items {
        if items.is_empty() {
            app.push_notice(NoticeLevel::Info, "暂无可恢复的历史会话，已开启新会话");
        } else {
            app.overlay.open_session_picker(items);
        }
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
    let pre_exec_handle = register_approval_hook(
        &ctx,
        loop_tx.clone(),
        Arc::clone(&active_sid),
        Arc::clone(&approval_snap),
    )?;
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
    if app.side_panel.open {
        populate_side_panel_data(&ctx, &mut app).await;
    }

    // 进入终端 raw mode（Fullscreen / Inline 双模支持），并注入 OSC 10/11/4 + DA1 探测到的终端自适应色板
    let _terminal_guard =
        TerminalGuard::enter_with_mode(app.screen_mode, app.mouse_capture_enabled)?;
    app.apply_terminal_colors(_terminal_guard.probed_colors.clone());

    let mut terminal = create_terminal(app.screen_mode)?;

    let mut term_stream = Some(EventStream::new());
    let mut spinner_tick = tokio::time::interval(Duration::from_millis(90));
    spinner_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut reveal_tick = tokio::time::interval(Duration::from_millis(33));
    reveal_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut task_state = RunnerTaskState::default();
    let mut needs_redraw = true;

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
        if app.screen_mode == ScreenMode::Inline && !app.is_running_activity() {
            flush_inline_scrollback(&mut terminal.terminal, &mut app);
        }
        if needs_redraw {
            terminal.draw_synced(|frame| render(frame, &mut app))?;
            needs_redraw = false;
        }

        let (action, step_dirty) = tokio::select! {
            maybe_term = poll_term_event(term_stream.as_mut()) => {
                match maybe_term {
                    Some(Ok(ev)) => {
                        let (mut next_action, mut dirty) = handle_term_event_dirty(&mut app, ev);
                        // 使用非阻塞 poll(0)+read() 排空底层就绪的连发键盘/滚动/鼠标事件，
                        // 绝不向 EventStream 注入 noop_waker（否则会占住后台 wake_task 导致后续输入失活）。
                        while next_action == AppAction::None
                            && term_stream.is_some()
                            && matches!(crossterm::event::poll(Duration::ZERO), Ok(true))
                        {
                            match crossterm::event::read() {
                                Ok(more) => {
                                    let (act, ev_dirty) = handle_term_event_dirty(&mut app, more);
                                    dirty |= ev_dirty;
                                    if act != AppAction::None {
                                        next_action = act;
                                        break;
                                    }
                                }
                                Err(err) => {
                                    app.push_notice(
                                        NoticeLevel::Error,
                                        format!("终端事件读取失败：{err}"),
                                    );
                                    dirty = true;
                                    break;
                                }
                            }
                        }
                        (next_action, dirty)
                    }
                    Some(Err(err)) => {
                        app.push_notice(NoticeLevel::Error, format!("终端事件读取失败：{err}"));
                        (AppAction::None, true)
                    }
                    None => (AppAction::Quit, true),
                }
            }
            maybe_loop = loop_rx.recv() => {
                match maybe_loop {
                    Some(msg) => {
                        let (mut next_action, mut dirty) = process_loop_ui_msg_dirty(
                            &mut app,
                            &mut task_state,
                            msg,
                        );
                        // 把缓冲区积压的 delta 一次性排空，降低高并发流式输出时的重绘次数
                        if next_action == AppAction::None {
                            while let Ok(more) = loop_rx.try_recv() {
                                let (act, ev_dirty) = process_loop_ui_msg_dirty(
                                    &mut app,
                                    &mut task_state,
                                    more,
                                );
                                dirty |= ev_dirty;
                                if act != AppAction::None {
                                    next_action = act;
                                    break;
                                }
                            }
                        }
                        (next_action, dirty)
                    }
                    None => (AppAction::Quit, true),
                }
            }
            maybe_ui = ui_rx.recv() => {
                match maybe_ui {
                    Some(ev) => {
                        let (mut next_action, mut dirty) = process_loop_ui_msg_dirty(
                            &mut app,
                            &mut task_state,
                            LoopUiMsg::Bus(ev),
                        );
                        if next_action == AppAction::None {
                            while let Ok(more) = ui_rx.try_recv() {
                                let (act, ev_dirty) = process_loop_ui_msg_dirty(
                                    &mut app,
                                    &mut task_state,
                                    LoopUiMsg::Bus(more),
                                );
                                dirty |= ev_dirty;
                                if act != AppAction::None {
                                    next_action = act;
                                    break;
                                }
                            }
                        }
                        (next_action, dirty)
                    }
                    None => (AppAction::Quit, true),
                }
            }
            _ = reveal_tick.tick(), if app.terminal_focused && (app.reveal.has_pending() || app.effort_ignition_at.is_some()) => {
                let ticked = app.reveal.tick();
                (AppAction::None, ticked || app.effort_ignition_at.is_some())
            }
            _ = spinner_tick.tick(), if app.is_running_activity() || (!app.terminal_focused && (app.reveal.has_pending() || app.effort_ignition_at.is_some())) => {
                let mut dirty = false;
                if app.is_running_activity() {
                    app.spinner_tick = app.spinner_tick.wrapping_add(1);
                    dirty = true;
                }
                if !app.terminal_focused {
                    dirty |= app.reveal.tick() || app.effort_ignition_at.is_some();
                }
                (AppAction::None, dirty)
            }
        };

        needs_redraw |= step_dirty || action != AppAction::None;

        handle_action(
            &ctx,
            &config,
            &mut app,
            &mut active_session,
            &mut terminal,
            &mut term_stream,
            &active_sid,
            &loop_tx,
            &mut task_state,
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
    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    emit_program_status(ProgramStatusState::Idle, None, None);
    task_state.abort_for_session_switch();
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
    loop_tx: mpsc::UnboundedSender<LoopUiMsg>,
    active_sid: Arc<std::sync::RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
) -> Result<DisposableHandle, CordisError> {
    let ctx_lookup = ctx.clone();
    ctx.on_waterfall(
        EVENT_TOOLS_PRE_EXECUTE,
        move |exec: ToolExecution,
              next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let loop_tx = loop_tx.clone();
            let active_sid = Arc::clone(&active_sid);
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

                // 若当前工具来自已经切换走的旧会话，直接拒绝，避免弹出孤儿审批框。
                if exec
                    .session_id
                    .as_deref()
                    .is_some_and(|sid| sid != *active_sid.read().unwrap())
                {
                    return PreExecuteDecision::deny("会话已切换，取消过期审批请求");
                }

                let args_preview = serde_json::to_string_pretty(&exec.args)
                    .unwrap_or_else(|_| exec.args.to_string());
                let args_preview = truncate_display(&args_preview.replace('\n', " "), 400);
                let (tx, rx) = futures::channel::oneshot::channel();
                if loop_tx
                    .send(LoopUiMsg::Bus(UiEvent::ApprovalRequested {
                        tool_name: exec.name.clone(),
                        args_preview,
                        responder: Arc::new(Mutex::new(Some(tx))),
                    }))
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingPrompt {
    text: String,
    images: Vec<ImageAttachment>,
}

impl PendingPrompt {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    fn into_action(self) -> AppAction {
        if self.images.is_empty() {
            AppAction::SubmitPrompt(self.text)
        } else {
            AppAction::SubmitPromptWithImages {
                text: self.text,
                images: self.images,
            }
        }
    }
}

fn process_loop_ui_msg(
    app: &mut AppState,
    task_state: &mut RunnerTaskState,
    msg: LoopUiMsg,
) -> AppAction {
    let (run_id, msg_session_id, event) = match msg {
        LoopUiMsg::Bus(event) => (None, None, event),
        LoopUiMsg::Task {
            run_id,
            session_id,
            event,
        } => {
            if !matches!(
                event,
                UiEvent::LocalBashFinished { .. }
                    | UiEvent::CompactionFinished { .. }
                    | UiEvent::RunFinished(_)
            ) && !task_state.is_current(run_id, &session_id, &app.session_id)
            {
                return AppAction::None;
            }
            (Some(run_id), Some(session_id), event)
        }
    };

    if matches!(event, UiEvent::ToolsUpdated) {
        return AppAction::RefreshSidePanelTools;
    }

    if matches!(event, UiEvent::LocalBashFinished { .. }) {
        if task_state
            .finish_if_current(
                run_id,
                msg_session_id.as_deref(),
                &app.session_id,
                ActiveTaskKind::LocalBash,
            )
            .is_none()
        {
            return AppAction::None;
        }
        app.apply_ui_event(event);
        return AppAction::RefreshSidePanelData;
    }

    if let UiEvent::CompactionFinished { is_manual, .. } = event {
        if is_manual
            && task_state
                .finish_if_current(
                    run_id,
                    msg_session_id.as_deref(),
                    &app.session_id,
                    ActiveTaskKind::ManualCompaction,
                )
                .is_none()
        {
            return AppAction::None;
        }
        app.apply_ui_event(event);
        return AppAction::RefreshSidePanelData;
    }

    if let UiEvent::RunFinished(ref res) = event {
        let Some(mut finished_task) = task_state.finish_if_current(
            run_id,
            msg_session_id.as_deref(),
            &app.session_id,
            ActiveTaskKind::AgentRun,
        ) else {
            return AppAction::None;
        };
        let was_aborted = finished_task.user_cancelled
            || res.as_ref().is_err_and(|e| e.contains("request aborted"))
            || matches!(app.last_turn_end_reason, Some(UiTurnEndReason::Cancelled));
        let is_ok = res.is_ok();
        let turn_completed = is_ok
            && matches!(
                app.last_turn_end_reason,
                None | Some(UiTurnEndReason::Completed)
            );
        app.apply_ui_event(event);

        if let Some(steer) = finished_task.pending_steer.take() {
            app.push_user_message(&steer.text);
            return steer.into_action();
        }
        if !was_aborted
            && turn_completed
            && let Some(next) = app.dequeue_next()
        {
            if !next.images.is_empty() && !app.active_model_supports_images() {
                app.input
                    .replace_draft_from_queue_with_images(&next.text, &next.images);
                app.reject_unsupported_images_notice();
                return AppAction::RefreshSidePanelData;
            }
            app.push_user_message(&next.text);
            if next.images.is_empty() {
                return AppAction::SubmitPrompt(next.text);
            }
            return AppAction::SubmitPromptWithImages {
                text: next.text,
                images: next.images.into_iter().map(|(_, img)| img).collect(),
            };
        }
        if !was_aborted && !turn_completed && !app.queued_prompts.is_empty() {
            app.push_notice(
                NoticeLevel::Warn,
                format!(
                    "上一轮未正常结束，已保留 {} 条排队消息",
                    app.queued_prompts.len()
                ),
            );
        }
        return AppAction::RefreshSidePanelData;
    }

    app.apply_ui_event(event);
    AppAction::None
}

#[cfg(test)]
fn process_ui_event(
    app: &mut AppState,
    task_state: &mut RunnerTaskState,
    event: UiEvent,
) -> AppAction {
    process_loop_ui_msg(app, task_state, LoopUiMsg::Bus(event))
}

fn format_clipboard_file_ref(workspace: &Path, path: &Path) -> String {
    let display = if let Ok(rel) = path.strip_prefix(workspace) {
        let rel_str = rel.display().to_string();
        if rel_str.is_empty() {
            path.display().to_string()
        } else {
            rel_str
        }
    } else {
        path.display().to_string()
    };
    if display.chars().any(char::is_whitespace) {
        format!("@\"{display}\"")
    } else {
        format!("@{display}")
    }
}

pub async fn apply_clipboard_content(
    app: &mut AppState,
    workspace: &Path,
    result: Result<clipboard::ClipboardContent, clipboard::ClipboardError>,
) {
    match result {
        Err(clipboard::ClipboardError::Empty) => {
            app.set_flash("剪贴板为空，或没有可粘贴的图片/文本");
        }
        Err(err) => {
            let msg = format!("剪贴板读取失败：{err}");
            app.set_flash(msg.clone());
            app.push_notice(NoticeLevel::Warn, msg);
        }
        Ok(clipboard::ClipboardContent::Text(text)) => {
            app.handle_paste(&text);
        }
        Ok(clipboard::ClipboardContent::Image {
            bytes,
            source_label,
        }) => match images::process_image_bytes(&bytes) {
            Ok(processed) => {
                let suffix = processed.marker_suffix();
                let _ = app.insert_pasted_image(suffix.as_deref(), processed.attachment);
            }
            Err(err) => {
                let msg = format!("剪贴板图片处理失败（{source_label}）：{err}");
                app.set_flash(msg.clone());
                app.push_notice(NoticeLevel::Error, msg);
            }
        },
        Ok(clipboard::ClipboardContent::Files(paths)) => {
            let mut text_refs = Vec::new();
            for path in paths {
                let Ok(meta) = tokio::fs::metadata(&path).await else {
                    text_refs.push(format_clipboard_file_ref(workspace, &path));
                    continue;
                };
                if meta.is_file()
                    && meta.len() <= 32 * 1024 * 1024
                    && let Ok(bytes) = tokio::fs::read(&path).await
                    && images::sniff_image_format(&bytes).is_some()
                {
                    match images::process_image_bytes(&bytes) {
                        Ok(processed) => {
                            let suffix = processed.marker_suffix();
                            if app
                                .insert_pasted_image(suffix.as_deref(), processed.attachment)
                                .is_none()
                            {
                                break;
                            }
                        }
                        Err(err) => {
                            let msg = format!("图片 `{}` 处理失败：{err}", path.display());
                            app.set_flash(msg.clone());
                            app.push_notice(NoticeLevel::Error, msg);
                        }
                    }
                    continue;
                }
                text_refs.push(format_clipboard_file_ref(workspace, &path));
            }
            if !text_refs.is_empty() {
                let joined = text_refs.join(" ");
                let needs_sep =
                    !app.input.is_empty() && !app.input.text().ends_with(char::is_whitespace);
                if needs_sep {
                    app.input.insert_char(' ');
                }
                app.input.insert_paste(&joined);
                app.input.insert_char(' ');
                app.sync_input_completion_overlay();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_agent_run(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopUiMsg>,
    task_state: &mut RunnerTaskState,
    prompt: String,
    images: Vec<ImageAttachment>,
) {
    let Ok(agent_loop) = ctx.agent_loop() else {
        app.push_notice(NoticeLevel::Error, "agentLoop 服务未就绪");
        return;
    };
    let (run_id, cancel) = task_state.start(active_session.id.clone(), ActiveTaskKind::AgentRun);
    app.set_busy_state(true);

    // 每次真正发请求前快照当前选择：`/model`、Ctrl+L 选择器、`/effort`、Tab、`/settings`
    // 五条路最终都汇到这里，无需在 UI 层（纯前端、无 Session 句柄）重复埋点。
    // `record_selection` 内部去重，与最新一条相同时不写。
    active_session.record_selection(
        app.provider.clone(),
        app.model.clone(),
        app.reasoning_effort,
    );

    let session: Arc<Session> = Arc::clone(active_session);
    let task_session_id = session.id.clone();
    let run_opts = RunOptions {
        provider: app.provider.clone(),
        model: app.model.clone(),
        max_steps: config.max_steps,
        max_tokens: None,
        temperature: None,
        reasoning_effort: app.reasoning_effort,
        images,
    };
    let tx = loop_tx.clone();
    tokio::spawn(async move {
        let res = agent_loop
            .run_with_options(session, prompt, cancel, run_opts)
            .await
            .map_err(|e| e.to_string());
        let _ = tx.send(LoopUiMsg::Task {
            run_id,
            session_id: task_session_id,
            event: UiEvent::RunFinished(res),
        });
    });
}

static LOCAL_BASH_SEQ: AtomicU64 = AtomicU64::new(1);

fn spawn_local_bash_run(
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopUiMsg>,
    task_state: &mut RunnerTaskState,
    command: String,
    exclude_from_context: bool,
) {
    if app.is_bash_running() {
        app.set_flash("已有 Bash 命令正在运行，请先按 Esc 取消");
        return;
    }
    if app.is_busy() {
        app.set_flash("模型正在生成中，请等待回合结束或按 Esc 取消后再运行 Bash 命令");
        return;
    }
    if app.is_compacting() {
        app.set_flash("正在执行上下文压缩，请等待完成后再运行 Bash 命令");
        return;
    }

    let seq = LOCAL_BASH_SEQ.fetch_add(1, Ordering::Relaxed);
    let call_id = format!("local-bash-{seq}");
    let (run_id, cancel) = task_state.start(active_session.id.clone(), ActiveTaskKind::LocalBash);
    app.start_local_bash(&call_id, &command, exclude_from_context);

    let session = Arc::clone(active_session);
    let task_session_id = session.id.clone();
    let update_session_id = task_session_id.clone();
    let cwd = app.workspace.clone();
    let tx = loop_tx.clone();
    let update_cid = call_id.clone();
    let update_tx = tx.clone();

    tokio::spawn(async move {
        let res = execute_bash_process(&command, &cwd, None, cancel, move |preview| {
            let _ = update_tx.send(LoopUiMsg::Task {
                run_id,
                session_id: update_session_id.clone(),
                event: UiEvent::ToolUpdate {
                    call_id: update_cid.clone(),
                    name: "bash".to_string(),
                    preview: preview.to_string(),
                },
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

        let _ = tx.send(LoopUiMsg::Task {
            run_id,
            session_id: task_session_id,
            event: UiEvent::LocalBashFinished {
                call_id,
                command,
                exclude_from_context,
                output: payload,
            },
        });
    });
}

fn create_terminal(mode: ScreenMode) -> Result<BufferedTerminal, std::io::Error> {
    let sync = SharedSyncWriter {
        inner: Arc::new(Mutex::new(SyncWriterInner::new(
            std::io::BufWriter::with_capacity(64 * 1024, std::io::stdout()),
        ))),
    };
    let backend = CrosstermBackend::new(sync.clone());
    let terminal = match mode {
        ScreenMode::Fullscreen => {
            let mut t = Terminal::new(backend)?;
            t.clear()?;
            t
        }
        ScreenMode::Inline => {
            let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
            let viewport_h = rows.clamp(7, 14);
            Terminal::with_options(
                backend,
                TerminalOptions {
                    viewport: Viewport::Inline(viewport_h),
                },
            )?
        }
    };
    Ok(ManagedTerminal { terminal, sync })
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
    terminal: &mut BufferedTerminal,
    term_stream: &mut Option<EventStream>,
    active_sid: &Arc<std::sync::RwLock<String>>,
    loop_tx: &mpsc::UnboundedSender<LoopUiMsg>,
    task_state: &mut RunnerTaskState,
    action: AppAction,
) {
    match action {
        AppAction::None => {}
        AppAction::Quit => {
            app.should_quit = true;
        }
        AppAction::PasteClipboard => {
            let res = clipboard::read_clipboard_with_osc5522_hook(|| async {
                *term_stream = None;
                let osc_res = tokio::task::spawn_blocking(clipboard::read_osc5522_from_tty)
                    .await
                    .map_err(|e| clipboard::ClipboardError::CommandFailed(e.to_string()))?;
                *term_stream = Some(EventStream::new());
                osc_res
            })
            .await;
            apply_clipboard_content(app, &config.workspace, res).await;
        }
        AppAction::CancelRun => {
            app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
            task_state.cancel_active(None);
        }
        AppAction::CompactSession(focus) => {
            if app.is_running_activity() {
                app.push_notice(
                    NoticeLevel::Warn,
                    "当前已有运行中的任务，请等待完成后再压缩",
                );
                return;
            }
            let Ok(compaction) = ctx.compaction() else {
                app.push_notice(NoticeLevel::Warn, "compaction 服务未启用");
                return;
            };
            if !compaction.config().enabled {
                app.push_notice(NoticeLevel::Info, "上下文压缩已在配置中禁用");
                return;
            }
            let (run_id, cancel) =
                task_state.start(active_session.id.clone(), ActiveTaskKind::ManualCompaction);
            app.set_compacting_state("手动压缩中…");

            let session = Arc::clone(active_session);
            let task_session_id = session.id.clone();
            let opts = ManualCompactOptions {
                provider: app.provider.clone(),
                model: app.model.clone(),
                focus,
            };
            let tx = loop_tx.clone();
            tokio::spawn(async move {
                match compaction.compact_now(session, opts, cancel).await {
                    Ok(_) | Err(CompactionError::Llm(_)) => {
                        // Pipeline 内部已广播 `EVENT_COMPACTION_END`，避免重复投递
                    }
                    Err(
                        err @ (CompactionError::AlreadyRunning(_) | CompactionError::Cordis(_)),
                    ) => {
                        let _ = tx.send(LoopUiMsg::Task {
                            run_id,
                            session_id: task_session_id,
                            event: UiEvent::CompactionFinished {
                                outcome: "failed".to_string(),
                                tokens_before: None,
                                tokens_after: None,
                                pruned_count: 0,
                                first_kept_seq: None,
                                summary_preview: None,
                                message: Some(err.to_string()),
                                is_manual: true,
                            },
                        });
                    }
                }
            });
        }
        AppAction::NewSession => {
            app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
            task_state.abort_for_session_switch();
            if app.is_running_activity() {
                app.set_idle_state();
            }
            if let Ok(sessions) = ctx.sessions() {
                let next = sessions.create(None);
                let title = next.header().title;
                *active_sid.write().unwrap() = next.id.clone();
                app.reset_for_new_session(next.id.clone(), title);
                *active_session = next;
                if app.side_panel.open {
                    populate_side_panel_data(ctx, app).await;
                }
            }
        }
        AppAction::OpenSessionPicker => {
            if let Ok(sessions) = ctx.sessions() {
                let items = collect_session_picker_items(&sessions);
                if items.is_empty() {
                    app.push_notice(NoticeLevel::Info, "暂无可恢复的历史会话");
                } else {
                    app.overlay.open_session_picker(items);
                }
            } else {
                app.push_notice(NoticeLevel::Warn, "sessions 服务未就绪");
            }
        }
        AppAction::ResumeSession(target_id) => {
            app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
            task_state.abort_for_session_switch();
            if app.is_running_activity() {
                app.set_idle_state();
            }
            let _ =
                resume_session_into_app(ctx, config, app, active_session, active_sid, &target_id)
                    .await;
        }
        AppAction::RenameSession(title_arg) => {
            if let Some(new_title) = title_arg {
                if active_session.set_title(new_title.clone()) {
                    app.session_title = active_session.header().title;
                    app.set_flash(format!("✓ 会话标题已重命名为：{new_title}"));
                    app.push_notice(NoticeLevel::Info, format!("会话标题已更新为：{new_title}"));
                } else {
                    app.push_notice(NoticeLevel::Warn, "标题未变化或为空，未做修改");
                }
            } else {
                let Ok(st) = ctx.session_title() else {
                    app.push_notice(
                        NoticeLevel::Warn,
                        "session-title 服务未启用（可在 /settings 中开启）",
                    );
                    return;
                };
                let session = Arc::clone(active_session);
                let provider = app.provider.clone();
                let model = app.model.clone();
                let tx = loop_tx.clone();
                app.set_flash("正在重新生成会话标题…");
                tokio::spawn(async move {
                    match st.regenerate_now(session, provider, model).await {
                        Ok(new_title) => {
                            let _ = tx.send(LoopUiMsg::Bus(UiEvent::Flash(format!(
                                "✓ 标题已生成：{new_title}"
                            ))));
                            let _ = tx.send(LoopUiMsg::Bus(UiEvent::Notice {
                                level: NoticeLevel::Info,
                                text: format!("会话标题已重新生成为：{new_title}"),
                            }));
                        }
                        Err(err) => {
                            let _ = tx.send(LoopUiMsg::Bus(UiEvent::Notice {
                                level: NoticeLevel::Warn,
                                text: format!("无法重新生成会话标题：{err}"),
                            }));
                        }
                    }
                });
            }
        }
        AppAction::SubmitPrompt(prompt) => {
            if prompt.contains('@')
                && !app.active_model_supports_images()
                && mentions::mentions_resolve_any_image(
                    &config.workspace,
                    config.restrict_to_workspace,
                    &prompt,
                )
                .await
            {
                app.input.set_text_end(&prompt);
                app.reject_unsupported_images_notice();
                return;
            }
            spawn_agent_run(
                ctx,
                config,
                app,
                active_session,
                loop_tx,
                task_state,
                prompt,
                Vec::new(),
            );
        }
        AppAction::SubmitPromptWithImages { text, images } => {
            if !app.active_model_supports_images() {
                app.reject_unsupported_images_notice();
                return;
            }
            spawn_agent_run(
                ctx,
                config,
                app,
                active_session,
                loop_tx,
                task_state,
                text,
                images,
            );
        }
        AppAction::SteerPrompt(prompt) => {
            if prompt.contains('@')
                && !app.active_model_supports_images()
                && mentions::mentions_resolve_any_image(
                    &config.workspace,
                    config.restrict_to_workspace,
                    &prompt,
                )
                .await
            {
                app.input.set_text_end(&prompt);
                app.reject_unsupported_images_notice();
                return;
            }
            if task_state.cancel_token().is_some() {
                app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
                task_state.cancel_active(Some(PendingPrompt::text(prompt)));
            } else {
                app.push_user_message(&prompt);
                spawn_agent_run(
                    ctx,
                    config,
                    app,
                    active_session,
                    loop_tx,
                    task_state,
                    prompt,
                    Vec::new(),
                );
            }
        }
        AppAction::SteerPromptWithImages { text, images } => {
            if !app.active_model_supports_images() {
                app.reject_unsupported_images_notice();
                return;
            }
            if task_state.cancel_token().is_some() {
                app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
                task_state.cancel_active(Some(PendingPrompt { text, images }));
            } else {
                app.push_user_message(&text);
                spawn_agent_run(
                    ctx,
                    config,
                    app,
                    active_session,
                    loop_tx,
                    task_state,
                    text,
                    images,
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
                loop_tx,
                task_state,
                command,
                exclude_from_context,
            );
        }
        AppAction::ShowStats => {
            if let Some(traj) = ctx.get::<TrajectoryRuntime>("trajectory") {
                if let Some(report) = traj.inspect(&app.session_id, TimelineMode::Sequence) {
                    let s = &report.snapshot.stats;
                    let interrupted = if s.interrupted_steps > 0 {
                        format!(" | interrupted={}", s.interrupted_steps)
                    } else {
                        String::new()
                    };
                    app.push_notice(
                        NoticeLevel::Info,
                        format!(
                            "会话统计：turns={} | steps={} | tools={} (err={}){} | tokens ↑{} ↓{} | avg TTFT={}ms | findings={}",
                            s.turns,
                            s.steps,
                            s.tool_calls,
                            s.tool_errors,
                            interrupted,
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
        AppAction::ShowSessionInfo => {
            let report = build_session_info_report(ctx, config, active_session, app);
            app.push_notice(NoticeLevel::Info, report);
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
            if app.side_panel.open {
                populate_side_panel_data(ctx, app).await;
            }
        }
        AppAction::RefreshSidePanelTools => {
            if app.side_panel.open {
                populate_tools_and_mcps(ctx, app);
            }
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
            app.pending_settings_model_pick = None;
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
            app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
            task_state.abort_for_session_switch();
            if app.is_running_activity() {
                app.set_idle_state();
            }
            if let Ok(sessions) = ctx.sessions() {
                match sessions.fork(&app.session_id, target_seq, SessionCreateOptions::default()) {
                    Ok(next) => {
                        *active_sid.write().unwrap() = next.id.clone();
                        app.restore_session_view(project_fork_view(&next, target_seq));
                        *active_session = next;
                        if app.side_panel.open {
                            populate_side_panel_data(ctx, app).await;
                        }
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

/// 在已运行的 TUI 状态上切换到指定历史会话（`/resume` 选择器与 `--resume` 共用投影与路由恢复）。
pub async fn resume_session_into_app(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    active_sid: &Arc<RwLock<String>>,
    target_id: &str,
) -> bool {
    let Ok(sessions) = ctx.sessions() else {
        app.push_notice(NoticeLevel::Warn, "sessions 服务未就绪");
        return false;
    };
    let Some(next) = sessions.get(target_id) else {
        app.push_notice(NoticeLevel::Warn, format!("未找到历史会话 `{target_id}`"));
        return false;
    };
    app.pending_settings_model_pick = None;
    *active_sid.write().unwrap() = next.id.clone();

    let sources = SelectionSources {
        model_from_env: config.model_from_env,
        model_from_settings: config.model_from_settings,
        reasoning_effort_from_settings: config.reasoning_effort_from_settings,
        startup: SessionSelection {
            provider: config.provider.clone(),
            model: config.model.clone(),
            reasoning_effort: config.reasoning_effort,
        },
        session: next.latest_selection().map(to_ui_selection),
        latest_workspace: None,
    };
    if let Some(resolved) = resolve_selection(&sources, true) {
        app.provider = resolved.provider;
        app.model = resolved.model;
        app.reasoning_effort = resolved.reasoning_effort;
    }

    app.restore_session_view(project_resume_view(&next));
    *active_session = next;
    if app.side_panel.open {
        populate_side_panel_data(ctx, app).await;
    }
    app.set_flash(format!("✓ 已恢复会话 ({})", short_id(target_id)));
    true
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

pub fn build_settings_entries(ctx: &Context, app: &AppState) -> Vec<SettingEntry> {
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

    if let Ok(st) = ctx.session_title() {
        let cfg = st.config();
        let thinking_opts = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
        let current_thinking = cfg.reasoning_effort.unwrap_or(ReasoningEffort::Low);
        let current_thinking_str = format!("{current_thinking:?}").to_lowercase();
        items.extend([
            SettingEntry {
                key: "sessionTitle.enabled".into(),
                category: "会话标题".into(),
                label: "自动生成会话标题".into(),
                description: "首轮输入及周期性自动概括会话主题，人工改名后不自动覆盖".into(),
                kind: SettingItemKind::Toggle { value: cfg.enabled },
            },
            SettingEntry {
                key: "sessionTitle.model".into(),
                category: "会话标题".into(),
                label: "标题生成模型".into(),
                description: "指定生成会话标题所用模型（Enter 选择，← 恢复跟随会话模型）".into(),
                kind: SettingItemKind::Model {
                    provider: cfg.provider,
                    model: cfg.model,
                },
            },
            SettingEntry {
                key: "sessionTitle.thinkingLevel".into(),
                category: "会话标题".into(),
                label: "标题生成思考强度".into(),
                description: "控制后台生成标题时的推理强度，默认 low 兼顾速度与成本".into(),
                kind: SettingItemKind::Choice {
                    options: thinking_opts.iter().map(|s| s.to_string()).collect(),
                    selected: choice_index(&thinking_opts, &current_thinking_str),
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
    if app.is_running_activity() {
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
    terminal: &mut BufferedTerminal,
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
                app.mouse_capture_enabled = next_mode == ScreenMode::Fullscreen;
                app.overlay
                    .set_setting_toggle("ui.mouseCapture", app.mouse_capture_enabled);
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
        k if k.starts_with("compaction.") || k.starts_with("sessionTitle.") => {
            apply_runtime_setting_change(ctx, key, value);
        }
        _ => {}
    }
}

/// 运行期热更新与终端实例无关的模块配置（`compaction.*`、`sessionTitle.*`）。
pub fn apply_runtime_setting_change(ctx: &Context, key: &str, value: &str) {
    if key.starts_with("compaction.")
        && let Ok(comp) = ctx.compaction()
    {
        let flag = value == "true";
        comp.update_config(|cfg| match key {
            "compaction.enabled" => cfg.enabled = flag,
            "compaction.autoPrune" => cfg.auto_prune = flag,
            "compaction.autoCompact" => cfg.auto_compact = flag,
            "compaction.reusePrefixCache" => cfg.reuse_prefix_cache = flag,
            "compaction.disableSummaryReasoning" => cfg.disable_summary_reasoning = flag,
            "compaction.enableReadDedup" => cfg.enable_read_dedup = flag,
            "compaction.enableWriteAfterCompactGuard" => {
                cfg.enable_write_after_compact_guard = flag
            }
            "compaction.softWindowCap" => cfg.soft_window_cap = parse_soft_window_cap_label(value),
            _ => {}
        });
    } else if key.starts_with("sessionTitle.")
        && let Ok(st) = ctx.session_title()
    {
        st.update_config(|cfg| match key {
            "sessionTitle.enabled" => cfg.enabled = value == "true",
            "sessionTitle.model" => {
                if let Some((provider, model)) = parse_model_ref(value) {
                    cfg.provider = provider;
                    cfg.model = Some(model);
                } else {
                    cfg.provider = None;
                    cfg.model = None;
                }
            }
            "sessionTitle.thinkingLevel" => {
                cfg.reasoning_effort = Some(normalize_session_title_thinking_level(Some(value)));
            }
            _ => {}
        });
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
    persist_setting_change_to(&path, key, value)
}

/// 将单条设置原子写入指定的 `settings.json` 文件。
pub fn persist_setting_change_to(path: &Path, key: &str, value: &str) -> Result<(), String> {
    patch_settings_file(path, |root| match key {
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
        "sessionTitle.enabled" => upsert_nested_json(
            root,
            "sessionTitle",
            "enabled",
            serde_json::Value::Bool(value == "true"),
        ),
        "sessionTitle.model" => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                if let Some(obj) = root.get_mut("sessionTitle").and_then(|v| v.as_object_mut()) {
                    obj.remove("model");
                }
            } else {
                upsert_nested_json(
                    root,
                    "sessionTitle",
                    "model",
                    serde_json::Value::String(trimmed.to_string()),
                );
            }
        }
        "sessionTitle.thinkingLevel" => upsert_nested_json(
            root,
            "sessionTitle",
            "thinkingLevel",
            serde_json::Value::String(value.trim().to_ascii_lowercase()),
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

    use mizar_tui_ui::app::SessionSelection;

    use super::{
        ActiveTaskKind, LoopUiMsg, RunnerTaskState, SelectionSources, TuiExitSummary,
        format_resume_command, process_loop_ui_msg, process_ui_event, resolve_model_switch,
        resolve_selection,
    };

    fn sel(provider: &str, model: &str, effort: mizar_llm::ReasoningEffort) -> SessionSelection {
        SessionSelection {
            provider: Some(provider.to_string()),
            model: Some(model.to_string()),
            reasoning_effort: Some(effort),
        }
    }

    /// 环境变量 MODEL 压过一切来源，包括 resume 的会话记录。
    #[test]
    fn resolve_selection_env_wins_over_everything() {
        let sources = SelectionSources {
            model_from_env: true,
            model_from_settings: true,
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "from-env", mizar_llm::ReasoningEffort::High),
            session: Some(sel(
                "magpie",
                "from-session",
                mizar_llm::ReasoningEffort::Max,
            )),
            latest_workspace: Some(sel(
                "magpie",
                "from-latest",
                mizar_llm::ReasoningEffort::Low,
            )),
        };
        assert_eq!(resolve_selection(&sources, true), None);
        assert_eq!(resolve_selection(&sources, false), None);
    }

    /// resume 时会话记录压过 settings（方案甲）。
    #[test]
    fn resolve_selection_resume_prefers_session_over_settings() {
        let sources = SelectionSources {
            model_from_settings: true,
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "settings-model", mizar_llm::ReasoningEffort::Low),
            session: Some(sel(
                "isok",
                "session-model",
                mizar_llm::ReasoningEffort::Max,
            )),
            latest_workspace: Some(sel(
                "magpie",
                "latest-model",
                mizar_llm::ReasoningEffort::Medium,
            )),
            ..Default::default()
        };
        assert_eq!(
            resolve_selection(&sources, true),
            Some(sel(
                "isok",
                "session-model",
                mizar_llm::ReasoningEffort::Max
            ))
        );
    }

    /// 冷启动时 settings 优先于最近会话记录（settings 是企业级配置，会话只是遗留状态）。
    #[test]
    fn resolve_selection_cold_start_prefers_settings_over_latest() {
        let sources = SelectionSources {
            model_from_settings: true,
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "settings-model", mizar_llm::ReasoningEffort::Low),
            session: None,
            latest_workspace: Some(sel("isok", "latest-model", mizar_llm::ReasoningEffort::Max)),
            ..Default::default()
        };
        assert_eq!(resolve_selection(&sources, false), None);
    }

    /// 冷启动且 settings 没写模型时，沿用最近会话的记录。
    #[test]
    fn resolve_selection_cold_start_carries_over_latest_when_settings_silent() {
        let sources = SelectionSources {
            startup: sel(
                "magpie",
                "derived-models[0]",
                mizar_llm::ReasoningEffort::Medium,
            ),
            latest_workspace: Some(sel("isok", "last-used", mizar_llm::ReasoningEffort::Max)),
            ..Default::default()
        };
        assert_eq!(
            resolve_selection(&sources, false),
            Some(sel("isok", "last-used", mizar_llm::ReasoningEffort::Max))
        );
    }

    /// 三层都没有更高优先级来源时返回 None，说明启动值已是最优。
    #[test]
    fn resolve_selection_returns_none_when_startup_already_wins() {
        let startup = sel("magpie", "derived", mizar_llm::ReasoningEffort::Medium);
        let sources = SelectionSources {
            startup: startup.clone(),
            ..Default::default()
        };
        assert_eq!(resolve_selection(&sources, false), None);
        assert_eq!(resolve_selection(&sources, true), None);

        // 即便有会话记录，只要与启动值一致也不应产生覆盖
        let same = SelectionSources {
            startup: startup.clone(),
            session: Some(startup.clone()),
            latest_workspace: Some(startup),
            ..Default::default()
        };
        assert_eq!(resolve_selection(&same, true), None);
        assert_eq!(resolve_selection(&same, false), None);
    }

    /// 强度被 settings 显式钉住时，冷启动不被最近会话覆盖；resume 仍可覆盖。
    #[test]
    fn resolve_selection_effort_pinned_by_settings_except_on_resume() {
        let sources = SelectionSources {
            reasoning_effort_from_settings: true,
            startup: sel("magpie", "settings-model", mizar_llm::ReasoningEffort::Low),
            latest_workspace: Some(SessionSelection {
                provider: None,
                model: None,
                reasoning_effort: Some(mizar_llm::ReasoningEffort::Max),
            }),
            ..Default::default()
        };
        // 冷启动：模型来自 settings 且与启动一致，强度被钉住 → 无变化
        assert_eq!(resolve_selection(&sources, false), None);

        // resume：会话里有强度记录时会覆盖 settings
        let resumed = SelectionSources {
            session: Some(SessionSelection {
                provider: None,
                model: None,
                reasoning_effort: Some(mizar_llm::ReasoningEffort::Max),
            }),
            ..sources
        };
        assert_eq!(
            resolve_selection(&resumed, true).map(|s| s.reasoning_effort),
            Some(Some(mizar_llm::ReasoningEffort::Max))
        );
    }

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
                    input: Vec::new(),
                },
                ModelInfo {
                    id: "mayoru-messages/claude-sonnet-5-5".to_string(),
                    name: "Claude Sonnet 5.5".to_string(),
                    context_window: None,
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
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
        app.set_busy_state(true);
        app.queued_prompts.push(mizar_tui_ui::QueuedPrompt {
            id: 1,
            text: "queued follow-up".into(),
            images: Vec::new(),
        });

        let mut task_state = RunnerTaskState::default();
        let (_rid1, cancel1) = task_state.start("sess-queue".into(), ActiveTaskKind::AgentRun);
        task_state.cancel_active(Some(super::PendingPrompt::text("urgent steer")));
        assert!(cancel1.is_cancelled());

        let act1 = process_ui_event(
            &mut app,
            &mut task_state,
            UiEvent::RunFinished(Err("request aborted".into())),
        );
        assert_eq!(act1, AppAction::SubmitPrompt("urgent steer".into()));
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::User { text }) if text == "urgent steer"
        ));
        assert_eq!(
            app.queued_prompts
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            vec!["queued follow-up"]
        );

        app.set_busy_state(true);
        let (_rid2, _) = task_state.start("sess-queue".into(), ActiveTaskKind::AgentRun);
        let act2 = process_ui_event(
            &mut app,
            &mut task_state,
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
        let mut task_state = RunnerTaskState::default();

        super::spawn_local_bash_run(
            &mut app,
            &session,
            &tx,
            &mut task_state,
            "echo RUNNER_LOCAL_OK".into(),
            false,
        );
        assert!(app.is_bash_running());
        assert!(task_state.cancel_token().is_some());

        while let Some(msg) = rx.recv().await {
            let is_done = matches!(
                &msg,
                LoopUiMsg::Task {
                    event: UiEvent::LocalBashFinished { .. },
                    ..
                }
            );
            let act = process_loop_ui_msg(&mut app, &mut task_state, msg);
            if is_done {
                assert_eq!(act, AppAction::RefreshSidePanelData);
                break;
            }
            assert_eq!(act, AppAction::None);
        }
        assert!(!app.is_bash_running());
        assert!(task_state.cancel_token().is_none());
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

        // 测试 Esc 取消（task_state.cancel_token().cancel()）
        super::spawn_local_bash_run(
            &mut app,
            &session,
            &tx,
            &mut task_state,
            "sleep 10".into(),
            true,
        );
        assert!(app.is_bash_running());
        task_state.cancel_token().unwrap().cancel();
        while let Some(msg) = rx.recv().await {
            let is_done = matches!(
                &msg,
                LoopUiMsg::Task {
                    event: UiEvent::LocalBashFinished { .. },
                    ..
                }
            );
            process_loop_ui_msg(&mut app, &mut task_state, msg);
            if is_done {
                break;
            }
        }
        assert!(!app.is_bash_running());
        assert!(task_state.cancel_token().is_none());
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
        app.set_compacting_state("手动压缩中…");
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
        let mut task_state = RunnerTaskState::default();
        let act = process_ui_event(&mut app, &mut task_state, UiEvent::ToolsUpdated);
        assert_eq!(act, AppAction::RefreshSidePanelTools);

        ctx.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn to_ui_turn_end_and_step_outcome_preserve_truncation_variants() {
        use super::*;
        assert_eq!(
            to_ui_turn_end_reason(TurnEndReason::Truncated),
            UiTurnEndReason::Truncated
        );
        assert_eq!(
            to_ui_step_outcome(&StepOutcome::MaxTokens),
            UiStepOutcome::MaxTokens
        );
    }

    #[test]
    fn process_ui_event_retains_queued_prompts_after_max_steps_or_truncated() {
        use super::*;
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use mizar_tui_ui::app::ChatItem;

        for (end_reason, step_outcome) in [
            (UiTurnEndReason::MaxSteps, UiStepOutcome::ToolCalls),
            (UiTurnEndReason::Truncated, UiStepOutcome::MaxTokens),
        ] {
            let mut app = AppState::new(
                PathBuf::from("."),
                "sess-queued-retain",
                Vec::new(),
                None,
                None,
                None,
            );
            let mut task_state = RunnerTaskState::default();

            app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
            app.input.insert_str("next queued prompt");
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert_eq!(app.queued_prompts.len(), 1);

            process_ui_event(&mut app, &mut task_state, UiEvent::StepStart { ts_ms: 10 });
            process_ui_event(
                &mut app,
                &mut task_state,
                UiEvent::StepEnd {
                    step: 1,
                    ts_ms: 20,
                    outcome: step_outcome,
                    usage: None,
                },
            );
            process_ui_event(
                &mut app,
                &mut task_state,
                UiEvent::TurnEnd { reason: end_reason },
            );

            let action = process_ui_event(
                &mut app,
                &mut task_state,
                UiEvent::RunFinished(Ok("non-error return string".into())),
            );
            assert_eq!(action, AppAction::RefreshSidePanelData);
            assert_eq!(app.queued_prompts.len(), 1);
            assert_eq!(app.queued_prompts[0].text, "next queued prompt");
            assert!(app.items.iter().any(|item| matches!(
                item,
                ChatItem::Notice {
                    level: NoticeLevel::Warn,
                    text,
                } if text.contains("已保留 1 条排队消息")
            )));
        }

        // 正常 Completed 时应继续自动出队下一条。
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queued-complete",
            Vec::new(),
            None,
            None,
            None,
        );
        let mut task_state = RunnerTaskState::default();
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.input.insert_str("auto dequeue me");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        process_ui_event(
            &mut app,
            &mut task_state,
            UiEvent::TurnEnd {
                reason: UiTurnEndReason::Completed,
            },
        );
        let action = process_ui_event(
            &mut app,
            &mut task_state,
            UiEvent::RunFinished(Ok("done".into())),
        );
        assert_eq!(action, AppAction::SubmitPrompt("auto dequeue me".into()));
        assert!(app.queued_prompts.is_empty());
    }

    #[test]
    fn process_loop_ui_msg_ignores_stale_task_events_after_run_or_session_switch() {
        use super::*;

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-current",
            Vec::new(),
            None,
            None,
            None,
        );
        let mut task_state = RunnerTaskState::default();
        let (stale_run_id, stale_cancel) =
            task_state.start("sess-current".into(), ActiveTaskKind::AgentRun);
        let (current_run_id, current_cancel) =
            task_state.start("sess-current".into(), ActiveTaskKind::LocalBash);
        assert!(stale_cancel.is_cancelled());
        assert!(!current_cancel.is_cancelled());
        app.set_local_bash_state();

        // 1. 旧 run_id 的 RunFinished / ToolUpdate 不得清空新任务的 RunnerTaskState 或更改 UI 状态
        let stale_act = process_loop_ui_msg(
            &mut app,
            &mut task_state,
            LoopUiMsg::Task {
                run_id: stale_run_id,
                session_id: "sess-current".into(),
                event: UiEvent::RunFinished(Ok("stale".into())),
            },
        );
        assert_eq!(stale_act, AppAction::None);
        assert!(app.is_bash_running());
        assert!(task_state.cancel_token().is_some());

        let stale_tool = process_loop_ui_msg(
            &mut app,
            &mut task_state,
            LoopUiMsg::Task {
                run_id: stale_run_id,
                session_id: "sess-current".into(),
                event: UiEvent::ToolUpdate {
                    call_id: "local-bash-999".into(),
                    name: "bash".into(),
                    preview: "stale output".into(),
                },
            },
        );
        assert_eq!(stale_tool, AppAction::None);

        // 2. 切换会话后的过期 LocalBashFinished 不得作用于新会话
        task_state.abort_for_session_switch();
        app.reset_for_new_session("sess-next", None);
        let cross_sess_act = process_loop_ui_msg(
            &mut app,
            &mut task_state,
            LoopUiMsg::Task {
                run_id: current_run_id,
                session_id: "sess-current".into(),
                event: UiEvent::LocalBashFinished {
                    call_id: "local-bash-1".into(),
                    command: "echo hi".into(),
                    exclude_from_context: false,
                    output: Ok(super::LocalBashOutput {
                        display: "hi".into(),
                        session_output: "hi".into(),
                        exit_code: Some(0),
                        cancelled: false,
                        timed_out: false,
                    }),
                },
            },
        );
        assert_eq!(cross_sess_act, AppAction::None);
        assert!(!app.is_running_activity());

        // 3. 过期 ManualCompaction 任务返回错误通知也不得污染当前空闲/新会话
        let (compact_rid, _) =
            task_state.start("sess-next".into(), ActiveTaskKind::ManualCompaction);
        app.set_compacting_state("手动压缩中…");
        task_state.abort_for_session_switch();
        app.set_idle_state();
        let stale_compact = process_loop_ui_msg(
            &mut app,
            &mut task_state,
            LoopUiMsg::Task {
                run_id: compact_rid,
                session_id: "sess-next".into(),
                event: UiEvent::CompactionFinished {
                    outcome: "failed".into(),
                    tokens_before: None,
                    tokens_after: None,
                    pruned_count: 0,
                    first_kept_seq: None,
                    summary_preview: None,
                    message: Some("stale error".into()),
                    is_manual: true,
                },
            },
        );
        assert_eq!(stale_compact, AppAction::None);
    }

    #[tokio::test]
    async fn project_resume_view_maps_turn_notice_and_truncated_turn_end_to_notice_never_assistant()
    {
        use super::*;
        use mizar_agent_spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};

        let ctx = Context::new();
        ctx.logger().set_silent();
        let mut cfg = AgentSpineConfig::coding(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        if let AgentToolsMode::Coding {
            ref mut mcp,
            ref mut skills,
            ref mut instructions,
            ..
        } = cfg.tools_mode
        {
            mcp.load_config_files = false;
            skills.include_default_roots = false;
            instructions.include_default_global = false;
        }
        ctx.plugin(AgentSpinePlugin, cfg);
        wait_for_services(&ctx).await.unwrap();

        let session = ctx.sessions().unwrap().create(None);
        session.append(SessionEvent::user("hi"));
        session.append_trace(SessionTraceEvent::TurnNotice {
            turn: 1,
            severity: TraceSeverity::Warn,
            text: "Reached max steps (max_steps=2)".into(),
        });
        session.append_trace(SessionTraceEvent::TurnEnd {
            turn: 1,
            reason: "max_steps".into(),
        });
        session.append_trace(SessionTraceEvent::TurnEnd {
            turn: 2,
            reason: "truncated".into(),
        });

        assert!(
            !session
                .log()
                .iter()
                .any(|e| matches!(&e.event, SessionEvent::Assistant { .. })),
            "TurnNotice 不得进入 session.log() 作为 Assistant 消息"
        );
        assert!(
            !session
                .derive_messages()
                .iter()
                .any(|m| format!("{m:?}").contains("Reached max steps")),
            "TurnNotice 不得污染 derive_messages() 模型上下文"
        );

        let view = project_resume_view(&session);
        assert!(
            !view
                .items
                .iter()
                .any(|i| matches!(i, SessionReplayItem::Assistant { .. })),
            "重放条目中不应出现伪造的 Assistant"
        );
        assert!(view.items.iter().any(|i| matches!(
            i,
            SessionReplayItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text == "Reached max steps (max_steps=2)"
        )));
        assert!(view.items.iter().any(|i| matches!(
            i,
            SessionReplayItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text.contains("max_tokens")
        )));

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn build_session_info_report_covers_full_identity_ledger_and_empty_omissions() {
        use cordis::Service;
        use mizar_session::{
            CompactionDigest, PrunedToolResult, SessionContextExt, SessionCreateOptions,
            SessionEvent, SessionStore, SessionTraceEvent,
        };

        let ctx = cordis::Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let store = ctx.sessions().unwrap();

        // 1. 全要素会话：分叉 + 多种表面/审计事件 + 剪枝 + 压缩 + cache_read/cache_write 用量
        let parent = store.create_with(SessionCreateOptions {
            id: Some("3a91c2de-1111-4222-8333-444455556666".into()),
            cwd: Some("/Users/moguw/workspace/dsh-space/mizar".into()),
            ..Default::default()
        });
        parent.append(SessionEvent::user("parent turn"));

        let child_id = "6f3d2c1a-8b4e-4f2a-9c71-2e5d8a0b4f13";
        let child = store
            .fork(
                &parent.id,
                1,
                SessionCreateOptions {
                    id: Some(child_id.into()),
                    cwd: Some("/Users/moguw/workspace/dsh-space/mizar".into()),
                    title: Some("feat(session): 新增 /session 命令".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        child.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
        child.record_selection(Some("mock".into()), Some("mock-v1".into()), None);
        child.append(SessionEvent::context_notice(
            "agent-instructions",
            "AGENTS.md",
        ));
        child.append(SessionEvent::context_notice("skill-catalog", "skills"));
        child.append(SessionEvent::context_notice("file-mentions", "src/lib.rs"));
        child.append(SessionEvent::Assistant {
            blocks: vec![mizar_llm::ContentBlock::Text {
                text: "step 1".into(),
            }],
            replay: None,
            usage: Some(mizar_llm::TokenUsage {
                input_tokens: 10_000,
                output_tokens: 8_700,
                cache_read_tokens: Some(40_000),
                cache_write_tokens: Some(2_000),
                ..Default::default()
            }),
        });
        child.append(SessionEvent::ToolResult {
            tool_call_id: "t1".into(),
            name: "read".into(),
            content: "ok".into(),
            is_error: false,
        });
        child.append(SessionEvent::ToolResult {
            tool_call_id: "t2".into(),
            name: "bash".into(),
            content: "boom".into(),
            is_error: true,
        });
        child.append(SessionEvent::BashExecution {
            command: "cargo check".into(),
            output: "ok".into(),
            exit_code: Some(0),
            cancelled: false,
            exclude_from_context: false,
        });
        child.append(SessionEvent::BashExecution {
            command: "git status".into(),
            output: "clean".into(),
            exit_code: Some(0),
            cancelled: false,
            exclude_from_context: true,
        });
        child.append_prune(vec![
            PrunedToolResult {
                target_seq: 6,
                replacement: "pruned".into(),
                original_chars: 9000,
            },
            PrunedToolResult {
                target_seq: 7,
                replacement: "pruned".into(),
                original_chars: 9000,
            },
        ]);
        child.compact_at(
            "checkpoint summary",
            6,
            Some(42_100),
            Some(6_800),
            CompactionDigest::default(),
        );
        child.append(SessionEvent::Assistant {
            blocks: vec![mizar_llm::ContentBlock::Text {
                text: "step 2".into(),
            }],
            replay: None,
            usage: Some(mizar_llm::TokenUsage {
                input_tokens: 14_300,
                output_tokens: 10_000,
                cache_read_tokens: Some(10_000),
                ..Default::default()
            }),
        });

        // 刻意把 app.workspace 设成不同路径，验证工作区展示的是会话自身 cwd 而非 app.workspace
        let app = AppState::new(
            PathBuf::from("/completely/different/workspace"),
            &child.id,
            Vec::new(),
            None,
            None,
            None,
        );
        // 测试前快照：只读查询不能改写会话头或条目数
        let entries_before = child.all_entries().len();
        let header_before = child.header();

        let home_dir = std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let fake_db = home_dir
            .as_ref()
            .map(|h| h.join(".mizar/sessions.sqlite"))
            .unwrap_or_else(|| PathBuf::from("/tmp/.mizar/sessions.sqlite"));
        let now_ms = header_before.updated_at_ms + 3 * 60 * 1_000;
        let report =
            super::build_session_info_report_at(&child, &app, true, Some(&fake_db), now_ms);

        assert_eq!(child.all_entries().len(), entries_before);
        assert_eq!(child.header(), header_before);

        let report_lines: Vec<&str> = report.lines().collect();
        assert_eq!(report_lines.len(), 16, "report:\n{report}");
        assert!(report.contains(&format!("ID       {child_id}")), "{report}");
        assert!(
            report.contains("标题     feat(session): 新增 /session 命令"),
            "{report}"
        );
        assert!(
            report.contains("工作区   /Users/moguw/workspace/dsh-space/mizar"),
            "{report}"
        );
        assert!(
            !report.contains("/completely/different/workspace"),
            "should display session cwd, not app.workspace:\n{report}"
        );
        let expected_created = super::format_local_datetime_ms(header_before.created_at_ms);
        assert!(
            report.contains(&format!(
                "创建     {expected_created}  ·  最后活动 3 分钟前"
            )),
            "{report}"
        );
        assert!(report.contains("谱系     分叉自 3a91c2de @ #1"), "{report}");
        if home_dir.is_some() {
            assert!(
                report.contains("存储     SQLite · ~/.mizar/sessions.sqlite"),
                "{report}"
            );
        }
        // 14 个事件（1 fork user + 13 追加）：log-only = TurnStart + Selection + !!bash + prune = 4，surface = 10
        assert!(
            report.contains("事件     14 条（surface 10 / log-only 4）"),
            "{report}"
        );
        assert!(
            report.contains("对话     user 1  ·  assistant 2"),
            "{report}"
        );
        assert!(
            report.contains("工具     tool-result 2（err 1）· bash 2（本地 1）· prune 2"),
            "{report}"
        );
        assert!(
            report.contains("注入     instructions 1 · skill-catalog 1 · file-mentions 1"),
            "{report}"
        );
        assert!(
            report.contains("压缩     1 次 · 最新保留自 #6 · 最近 42.1k → 6.8k"),
            "{report}"
        );
        // prompt 累计 = (10000+40000+2000) + (14300+10000) = 76300 -> 76.3k
        // cache_read = 50.0k, cache_write = 2.0k, output = 18.7k
        assert!(
            report.contains("Token    累计 ↑76.3k（cache 50.0k 读 / 2.0k 写）↓18.7k"),
            "{report}"
        );
        // 最近一次请求 prompt = 14300 + 10000 = 24.3k / 128k（19.0%）
        assert!(
            report.contains("上下文   24.3k / 128k（19.0%）"),
            "{report}"
        );

        // 2. 无谱系、无压缩、--no-persist 空白会话：省略「谱系」「压缩」，共 14 行
        let fresh = store.create(None);
        let fresh_header = fresh.header();
        let fresh_app = AppState::new(PathBuf::from("."), &fresh.id, Vec::new(), None, None, None);
        let fresh_report = super::build_session_info_report_at(
            &fresh,
            &fresh_app,
            false,
            None,
            fresh_header.updated_at_ms + 5_000,
        );
        let fresh_lines: Vec<&str> = fresh_report.lines().collect();
        assert_eq!(fresh_lines.len(), 14, "fresh_report:\n{fresh_report}");
        assert!(!fresh_report.contains("谱系"), "{fresh_report}");
        assert!(!fresh_report.contains("压缩"), "{fresh_report}");
        assert!(
            fresh_report.contains("标题     （未命名）"),
            "{fresh_report}"
        );
        assert!(
            fresh_report.contains("存储     仅内存（未持久化）"),
            "{fresh_report}"
        );
        assert!(fresh_report.contains("注入     无"), "{fresh_report}");
        assert!(
            fresh_report.contains("Token    累计 ↑0 ↓0"),
            "{fresh_report}"
        );
        assert!(
            fresh_report.contains("上下文   0 / 128k（0.0%）"),
            "{fresh_report}"
        );
        assert!(fresh_report.contains("最后活动 刚刚"), "{fresh_report}");

        // 3. project_session_view 同样汇总 Assistant.usage，供 --resume 恢复状态栏与侧栏读数
        let view = super::project_session_view(&child, "resumed");
        assert_eq!(view.total_usage.input_tokens, 24_300);
        assert_eq!(view.total_usage.output_tokens, 18_700);
        assert_eq!(view.total_usage.cache_read_tokens, Some(50_000));
        assert_eq!(view.total_usage.cache_write_tokens, Some(2_000));
        assert_eq!(view.total_usage.total_tokens, Some(95_000));
        assert_eq!(view.last_prompt_tokens, Some(24_300));

        // 相对时间区间检查
        assert_eq!(
            super::format_relative_activity(1_000, 1_000 + 2 * 3_600_000),
            "2 小时前"
        );
        assert_eq!(
            super::format_relative_activity(1_000, 1_000 + 3 * 86_400_000),
            "3 天前"
        );

        ctx.fiber().dispose().await;
    }

    #[test]
    fn dirty_gate_skips_key_release_focus_and_noop_mouse_move() {
        use crossterm::event::{
            Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers,
            MouseButton, MouseEvent, MouseEventKind,
        };
        use mizar_tui_ui::reveal::RevealSlot;

        let mut app = AppState::new(PathBuf::from("."), "s-dirty", Vec::new(), None, None, None);

        // 1. KeyEventKind::Release 与 Focus 事件不触发重绘
        let release = KeyEvent {
            code: KeyCode::Char('a'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Release,
            state: KeyEventState::NONE,
        };
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::Key(release)),
            (AppAction::None, false)
        );
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::FocusLost),
            (AppAction::None, false)
        );
        assert!(!app.terminal_focused);
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::FocusGained),
            (AppAction::None, false)
        );
        assert!(app.terminal_focused);

        // 2. KeyEventKind::Press 触发重绘
        let press = KeyEvent {
            code: KeyCode::Char('x'),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        };
        let (_, dirty_press) = super::handle_term_event_dirty(&mut app, TermEvent::Key(press));
        assert!(dirty_press);

        // 3. MouseEventKind::Moved 在未改变悬停分割线或时间轴行时不触发重绘
        app.side_panel.divider_col = Some(40);
        let move_noop = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 10,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::Mouse(move_noop)),
            (AppAction::None, false)
        );
        let move_hit_divider = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 40,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::Mouse(move_hit_divider)),
            (AppAction::None, true)
        );
        // 在同一分割线列内垂直移动时 hovered_divider 保持 true，不重复重绘
        let move_same_divider = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 40,
            row: 6,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(
            super::handle_term_event_dirty(&mut app, TermEvent::Mouse(move_same_divider)),
            (AppAction::None, false)
        );

        let down = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert!(super::handle_term_event_dirty(&mut app, TermEvent::Mouse(down)).1);

        // 4. reveal 节流开启且已有活跃 Assistant 槽位时，高频 TextDelta 延后给 reveal_tick (30fps)
        app.reveal.enabled = true;
        let mut task_state = super::RunnerTaskState::default();
        let (_, d_start) = super::process_loop_ui_msg_dirty(
            &mut app,
            &mut task_state,
            super::LoopUiMsg::Bus(UiEvent::TurnStart { turn: 1 }),
        );
        assert!(d_start);
        // 首个 TextDelta 创建 Assistant 条目，先触发首帧展示
        let (_, d_first) = super::process_loop_ui_msg_dirty(
            &mut app,
            &mut task_state,
            super::LoopUiMsg::Bus(UiEvent::TextDelta {
                text: "Hello".into(),
                ts_ms: 1,
            }),
        );
        assert!(d_first);
        let item_idx = app.items.len() - 1;
        app.reveal
            .observe(RevealSlot::AssistantText(item_idx), 5, true);
        // 后续紧邻 TextDelta 仅推进 target_bytes，不立即同步重绘
        let (act2, d_next) = super::process_loop_ui_msg_dirty(
            &mut app,
            &mut task_state,
            super::LoopUiMsg::Bus(UiEvent::TextDelta {
                text: " world".into(),
                ts_ms: 2,
            }),
        );
        assert_eq!(act2, AppAction::None);
        assert!(!d_next);
        assert!(app.reveal.has_pending());
    }

    #[test]
    fn sync_buf_writer_holds_flush_until_end_sync_frame() {
        use std::io::Write;
        let mut buf = Vec::<u8>::new();
        {
            let mut w = super::SyncWriterInner::new(std::io::BufWriter::new(&mut buf));
            w.begin_sync_frame().unwrap();
            w.write_all(b"FRAME_PAYLOAD").unwrap();
            // defer_flush 期间 flush 保持缓冲不落底层 sink
            w.flush().unwrap();
            w.end_sync_frame().unwrap();
        }
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with(mizar_tui_ui::terminal::BEGIN_SYNCHRONIZED_OUTPUT));
        assert!(s.contains("FRAME_PAYLOAD"));
        assert!(s.ends_with(mizar_tui_ui::terminal::END_SYNCHRONIZED_OUTPUT));
    }
}
