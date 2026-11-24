//! 异步运行循环：通过 `SessionActor` / `SessionActorHandle` 驱动单会话核心循环，并将零序列化 `ClientEvent` 接入终端界面。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::EventStream;
use mizar_client::actor::{SessionActor, SessionActorConfig};
use mizar_client::logging::{env_log_level_override, install_tracing_handler};
use mizar_client::run_dir::{check_session_writable, ensure_run_dir, resolve_run_dir};
use mizar_client::workspace::collect_user_skill_names;
use mizar_core::spine::MizarRuntime;
use mizar_core::system_prompt::{PromptSection, SystemPrompt};
use mizar_llm::ReasoningEffort;
use mizar_session::{Session, SessionListQuery, SessionStore};
use mizar_tui_ui::TuiFeatures;
use mizar_tui_ui::app::{
    AppAction, AppState, ApprovalMode, ChatItem, NoticeLevel, SessionSelection, ToolState, UiEvent,
    UiLogLevel, UiTurnEndReason,
};
use mizar_tui_ui::emphasis::{EmphasisConfig, reply_guidance_prompt};
use mizar_tui_ui::overlay::{ApprovalDecision, SessionPickerItem, SettingEntry};
use mizar_tui_ui::terminal::{ProgramStatusState, ScreenMode, TerminalGuard, emit_program_status};
use mizar_tui_ui::theme::ThemePreset;
use mizar_tui_ui::ui::{
    build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback, render,
};
use mizar_types::HookGuard;
use mizar_types::client::HostExtensionFlags;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

mod actions;
mod terminal_io;

pub use actions::{apply_clipboard_content, resume_session_into_app};
pub use mizar_client::events::{
    ActiveTask, ActiveTaskKind, LoopClientMsg, RunController, steer_session,
    subscribe_session_events, subscribe_session_events_into, to_ui_status, to_ui_step_outcome,
    to_ui_turn_end_reason,
};
pub use mizar_client::session_view::{
    SelectionSources, collect_session_picker_items, find_model_in_provider,
    latest_workspace_selection, project_fork_view, project_resume_view, project_session_tree,
    project_session_view, resolve_model_switch, resolve_selection, to_ui_selection,
};
pub use mizar_client::settings::{
    ModuleToggleOutcome, SettingsViewSnapshot, apply_runtime_setting_change,
    persist_setting_change, persist_setting_change_to, toggle_spine_module,
};

use actions::{
    SidePanelRefreshController, TuiActorSync, handle_action, populate_side_panel_data,
    process_actor_client_event_dirty, sync_queue_after_term_event,
};
use terminal_io::{
    create_terminal, flush_inline_scrollback, handle_term_event_dirty, insert_scrollback_lines,
    poll_term_event, process_loop_ui_msg_dirty,
};

fn inline_finalized_commit_limit(app: &AppState) -> usize {
    if !app.is_running_activity() {
        return app.items.len();
    }
    let mut limit = app.inline_committed_items.min(app.items.len());
    while limit < app.items.len() {
        let finalized = match &app.items[limit] {
            ChatItem::User { .. } | ChatItem::Notice { .. } => true,
            ChatItem::Assistant { .. } => {
                // 仅当后续出现非 Notice 项时，流式助手消息才已定型且关闭 reveal
                app.items[limit + 1..]
                    .iter()
                    .any(|it| !matches!(it, ChatItem::Notice { .. }))
            }
            ChatItem::ToolCall { status, .. } => {
                !matches!(status, ToolState::Running) && limit + 1 < app.items.len()
            }
        };
        if !finalized {
            break;
        }
        limit += 1;
    }
    limit
}

fn flush_inline_scrollback_incremental<B: ratatui::backend::Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut AppState,
) -> bool {
    if !app.is_running_activity() {
        let before = app.inline_committed_items;
        flush_inline_scrollback(terminal, app);
        return app.inline_committed_items > before;
    }

    let target = inline_finalized_commit_limit(app);
    if app.inline_committed_items >= target {
        return false;
    }
    let Ok(size) = terminal.size() else {
        return false;
    };
    let width = size.width.max(20) as usize;
    let chunk_max = size.height.saturating_sub(2).max(4) as usize;

    if !app.inline_welcome_committed {
        let welcome_lines = build_welcome_lines_for_scrollback(app, width);
        insert_scrollback_lines(terminal, &welcome_lines, chunk_max);
        app.inline_welcome_committed = true;
    }

    let before = app.inline_committed_items;
    while app.inline_committed_items < target {
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
    app.inline_committed_items > before
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
    /// 单写者锁与本地 Socket 运行目录（测试默认 `None` 以避免触碰全局 `~/.mizar/run/`）。
    pub run_dir: Option<PathBuf>,
    /// 是否启用本地 RPC Unix Socket（`TuiConfig::new` 默认 `false`，交互式 CLI 启动设为 `true`）。
    pub enable_rpc_socket: bool,
    pub restrict_to_workspace: bool,
    pub check_for_updates: bool,
    pub disabled_plugins: Vec<String>,
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
            run_dir: None,
            enable_rpc_socket: false,
            restrict_to_workspace: false,
            check_for_updates: false,
            disabled_plugins: Vec::new(),
            startup_notices: Vec::new(),
        }
    }
}

/// 将交互式 TUI 回复视觉语法提示段挂载到系统提示词注册表（`order = 60`，子代理会话自动跳过）。
pub fn mount_reply_guidance_section(
    sp: &Arc<SystemPrompt>,
    sessions: Option<Arc<SessionStore>>,
    emphasis: &EmphasisConfig,
) -> Option<HookGuard> {
    if !(emphasis.enabled && emphasis.guidance) {
        return None;
    }
    let prompt = reply_guidance_prompt();
    Some(sp.section(PromptSection::new_for_session(
        "tui_reply_guidance",
        60,
        move |session_id| {
            if let (Some(sid), Some(store)) = (session_id, sessions.as_ref())
                && store
                    .get(sid)
                    .is_some_and(|s| s.header().is_subagent_child())
            {
                return None;
            }
            Some(prompt.clone())
        },
    )))
}

/// 将 TUI 回复视觉语法提示段挂载到 `MizarRuntime` 上，返回可选 RAII `HookGuard`。
pub fn install_tui_emphasis(
    runtime: &MizarRuntime,
    emphasis: &EmphasisConfig,
) -> Option<HookGuard> {
    mount_reply_guidance_section(
        &runtime.system_prompt,
        Some(Arc::clone(&runtime.sessions)),
        emphasis,
    )
}

/// 构造 `/session` 命令的只读会话身份与账目报告（`TuiConfig` + `AppState` 薄包装）。
pub fn build_session_info_report(
    runtime: &MizarRuntime,
    config: &TuiConfig,
    session: &Session,
    app: &AppState,
) -> String {
    mizar_client::session_view::build_session_info_report(
        runtime,
        session,
        config.session_db_path.as_deref(),
        app.active_context_window(),
        app.last_prompt_tokens,
        &app.side_panel_data.context,
    )
}

/// 以显式持久化标记、路径与当前时间戳构造 `/session` 报告（`AppState` 薄包装，便于确定性测试）。
pub fn build_session_info_report_at(
    session: &Session,
    app: &AppState,
    persisted: bool,
    session_db_path: Option<&Path>,
    now_ms: u128,
) -> String {
    mizar_client::session_view::build_session_info_report_at(
        session,
        persisted,
        session_db_path,
        app.active_context_window(),
        app.last_prompt_tokens,
        &app.side_panel_data.context,
        now_ms,
    )
}

fn settings_snapshot_from_app(app: &AppState) -> SettingsViewSnapshot {
    SettingsViewSnapshot {
        show_thinking: app.show_thinking,
        verbose_expanded: app.verbose_expanded,
        side_panel_open: app.side_panel.open,
        mouse_capture_enabled: app.mouse_capture_enabled,
        approval_mode: app.approval_mode,
        reasoning_effort: app.reasoning_effort,
        theme_preset_id: app.theme_preset.name().to_string(),
        screen_mode_id: app.screen_mode.as_str().to_string(),
    }
}

/// 基于 `AppState` 与 `MizarRuntime` 运行时服务状态构建 `/settings` 条目列表（薄包装）。
pub fn build_settings_entries(runtime: &MizarRuntime, app: &AppState) -> Vec<SettingEntry> {
    mizar_client::settings::build_settings_entries(runtime, &settings_snapshot_from_app(app))
}

pub(super) fn process_loop_ui_msg(
    app: &mut AppState,
    task_state: &mut RunController,
    msg: LoopClientMsg,
) -> AppAction {
    let (run_id, msg_session_id, event) = match msg {
        LoopClientMsg::Bus(event) => (None, None, event),
        LoopClientMsg::Task {
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
        return if app.features.side_panel {
            AppAction::RefreshSidePanelTools
        } else {
            AppAction::None
        };
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
        let finished_task = if run_id.is_some() {
            let Some(task) = task_state.finish_if_current(
                run_id,
                msg_session_id.as_deref(),
                &app.session_id,
                ActiveTaskKind::AgentRun,
            ) else {
                return AppAction::None;
            };
            Some(task)
        } else {
            None
        };
        let was_aborted = finished_task.as_ref().is_some_and(|t| t.user_cancelled)
            || res.as_ref().is_err_and(|e| e.contains("request aborted"))
            || matches!(app.last_turn_end_reason, Some(UiTurnEndReason::Cancelled));
        let is_ok = res.is_ok();
        let turn_completed = is_ok
            && matches!(
                app.last_turn_end_reason,
                None | Some(UiTurnEndReason::Completed)
            );
        app.apply_ui_event(event);

        // 队列自动出队由 SessionActor 内部 run_wake_loop 独占驱动，此处仅在非正常结束时提示保留条数
        if !was_aborted && !turn_completed && !app.queue.is_empty() {
            app.push_notice(
                NoticeLevel::Warn,
                format!("上一轮未正常结束，已保留 {} 条排队消息", app.queue.len()),
            );
        }
        return AppAction::RefreshSidePanelData;
    }

    app.apply_ui_event(event);
    AppAction::None
}

async fn poll_client_event(rx: &mut tokio::sync::broadcast::Receiver<UiEvent>) -> Option<UiEvent> {
    loop {
        match rx.recv().await {
            Ok(ev) => return Some(ev),
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
        }
    }
}

/// 在已装配的 `MizarRuntime` 上启动全屏 TUI。
pub async fn run_tui(
    runtime: Arc<MizarRuntime>,
    config: TuiConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    run_tui_with_summary(runtime, config).await.map(|_| ())
}

/// 在已装配的 `MizarRuntime` 上启动 TUI，并在恢复终端后返回当前会话摘要。
pub async fn run_tui_with_summary(
    runtime: Arc<MizarRuntime>,
    config: TuiConfig,
) -> Result<TuiExitSummary, Box<dyn std::error::Error>> {
    let ext_flags = HostExtensionFlags::from_disabled_plugins(&config.disabled_plugins);
    if !ext_flags.tui {
        return Err("TUI 已被配置 disabledPlugins 中的 \"tui\" 禁用".into());
    }

    let sessions = &runtime.sessions;
    let mut resume_notice: Option<(NoticeLevel, String)> = None;
    let effective_run_dir = match &config.run_dir {
        Some(dir) => match ensure_run_dir(dir) {
            Ok(()) => Some(dir.clone()),
            Err(err) => {
                resume_notice = Some((NoticeLevel::Warn, format!("运行目录初始化失败：{err}")));
                None
            }
        },
        None if config.enable_rpc_socket => match resolve_run_dir() {
            Ok(dir) => Some(dir),
            Err(err) => {
                resume_notice = Some((NoticeLevel::Warn, format!("运行目录初始化失败：{err}")));
                None
            }
        },
        None => None,
    };

    let mut startup_picker_items: Option<Vec<SessionPickerItem>> = None;
    let (initial_session, resumed) = match &config.resume_session {
        Some(ResumeTarget::Picker) => {
            startup_picker_items = Some(collect_session_picker_items(sessions));
            (sessions.create(None), false)
        }
        Some(ResumeTarget::Id(target_id)) => {
            if let Some(found) = sessions.get(target_id) {
                if let Some(ref rd) = effective_run_dir
                    && let Err(lock_err) = check_session_writable(rd, &found.id, std::process::id())
                {
                    resume_notice = Some((
                        NoticeLevel::Error,
                        format!("无法恢复会话 `{target_id}`：{lock_err}，已开启新会话"),
                    ));
                    (sessions.create(None), false)
                } else {
                    (found, true)
                }
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
                    exclude_subagents: true,
                })
                .ok()
                .and_then(|mut list| list.drain(..).next())
                .map(|h| h.id);
            if let Some(id) = latest_id
                && let Some(found) = sessions.get(&id)
            {
                if let Some(ref rd) = effective_run_dir
                    && let Err(lock_err) = check_session_writable(rd, &found.id, std::process::id())
                {
                    resume_notice = Some((
                        NoticeLevel::Error,
                        format!("无法恢复最近会话 `{}`：{lock_err}，已开启新会话", found.id),
                    ));
                    (sessions.create(None), false)
                } else {
                    (found, true)
                }
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
    let providers = runtime.llm.provider_infos();

    let (loop_tx, mut loop_rx) = mpsc::unbounded_channel::<LoopClientMsg>();
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();

    // 桥接 tracing 日志到 UI 事件，确保任何输出都不会污染终端 stdout
    let log_threshold = match env_log_level_override() {
        Some(UiLogLevel::Silent) => UiLogLevel::Silent,
        Some(UiLogLevel::Error) => UiLogLevel::Error,
        _ => UiLogLevel::Warn,
    };
    let log_tx = ui_tx.clone();
    let _tracing_guard = install_tracing_handler(log_threshold, move |record| {
        let _ = log_tx.send(UiEvent::Log(record.clone()));
    });

    let mut active_session = Arc::clone(&initial_session);
    let mut guards = Vec::new();

    let mut app = AppState::new(
        config.workspace.clone(),
        active_session.id.clone(),
        providers,
        config.provider.clone(),
        config.model.clone(),
        config.reasoning_effort,
    );
    app.session_title = active_session.header().title;
    app.set_features(TuiFeatures::from(&ext_flags));

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
            latest_workspace_selection(sessions, &config.workspace).map(to_ui_selection)
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
    app.user_skill_names = collect_user_skill_names(&runtime, &config.workspace).await;
    if let Some(skills) = runtime.skills.as_ref() {
        let rt_skills = Arc::clone(&runtime);
        let ws_skills = config.workspace.clone();
        let tx_skills = ui_tx.clone();
        guards.push(skills.on_change(move |_| {
            let rt_inner = Arc::clone(&rt_skills);
            let ws = ws_skills.clone();
            let tx = tx_skills.clone();
            tokio::spawn(async move {
                let names = collect_user_skill_names(&rt_inner, &ws).await;
                let _ = tx.send(UiEvent::UserSkillsUpdated(names));
            });
        }));
    }
    {
        let tx_tools = ui_tx.clone();
        guards.push(runtime.tools.on_change(move |_| {
            let _ = tx_tools.send(UiEvent::ToolsUpdated);
        }));
    }
    for err_msg in runtime.modules.startup_errors() {
        app.push_notice(NoticeLevel::Warn, err_msg);
    }
    for (level, msg) in &config.startup_notices {
        app.push_notice(*level, msg.clone());
    }

    app.approval_mode = config.approval_mode;
    app.screen_mode = config.screen_mode;
    if let Some(v) = config.show_thinking {
        app.show_thinking = v;
    }
    if let Some(v) = config.verbose_expanded {
        app.verbose_expanded = v;
    }
    if let Some(v) = config.side_panel_open {
        app.side_panel.open = v && app.features.side_panel;
    }
    if !app.features.side_panel {
        app.side_panel.open = false;
    }
    app.mouse_capture_enabled = config
        .mouse_capture
        .unwrap_or(config.screen_mode == ScreenMode::Fullscreen);
    if let Some(preset) = config.theme_preset {
        app.set_theme_preset(preset);
    }

    if ext_flags.tui_emphasis
        && let Some(guard) = install_tui_emphasis(&runtime, &app.emphasis)
    {
        guards.push(guard);
    }

    let mut actor_cfg = SessionActorConfig::new(&config.workspace);
    actor_cfg.session_db_path = config.session_db_path.clone();
    actor_cfg.run_dir = effective_run_dir.clone();
    actor_cfg.provider = app.provider.clone();
    actor_cfg.model = app.model.clone();
    actor_cfg.reasoning_effort = app.reasoning_effort;
    actor_cfg.approval_mode = app.approval_mode;
    actor_cfg.always_allowed_tools = app.always_allowed_tools.clone();
    actor_cfg.max_steps = config.max_steps;
    actor_cfg.restrict_to_workspace = config.restrict_to_workspace;
    actor_cfg.host_extensions = ext_flags;

    let actor = SessionActor::spawn(Arc::clone(&runtime), Arc::clone(&active_session), actor_cfg);
    let mut client_rx = actor.subscribe_client_events();
    let mut sync = TuiActorSync::new(&active_session, &app);

    if app.features.side_panel && app.side_panel.open {
        populate_side_panel_data(&runtime, &mut app).await;
    }

    let mut task_state = RunController::default();
    let loop_res: Result<(), Box<dyn std::error::Error>> = async {
        // 进入终端 raw mode（Fullscreen / Inline 双模支持），并注入 OSC 10/11/4 + DA1 探测到的终端自适应色板
        let _terminal_guard =
            TerminalGuard::enter_with_mode(app.screen_mode, app.mouse_capture_enabled)?;
        app.apply_terminal_colors(_terminal_guard.probed_colors.clone());

        let mut terminal = create_terminal(app.screen_mode)?;

        let mut term_stream = Some(EventStream::new());
        let (mut panel_ctrl, mut panel_rx) = SidePanelRefreshController::new();
        let mut spinner_tick = tokio::time::interval(Duration::from_millis(90));
        spinner_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut reveal_tick = tokio::time::interval(Duration::from_millis(33));
        reveal_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

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

        let run_loop_res: Result<(), Box<dyn std::error::Error>> = async {
            while !app.should_quit {
                if app.screen_mode == ScreenMode::Inline
                    && flush_inline_scrollback_incremental(&mut terminal.terminal, &mut app)
                {
                    needs_redraw = true;
                }
                if needs_redraw {
                    terminal.draw_synced(|frame| render(frame, &mut app))?;
                    needs_redraw = false;
                }

                let (action, step_dirty) = tokio::select! {
                    maybe_term = poll_term_event(term_stream.as_mut()) => {
                        match maybe_term {
                            Some(Ok(ev)) => {
                                let prev_queue = app.queue.items().to_vec();
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
                                if sync_queue_after_term_event(
                                    &actor,
                                    &config,
                                    &mut app,
                                    &mut sync,
                                    &prev_queue,
                                )
                                .await
                                {
                                    dirty = true;
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
                    maybe_client = poll_client_event(&mut client_rx) => {
                        match maybe_client {
                            Some(ev) => {
                                let (mut next_action, mut dirty) = process_actor_client_event_dirty(
                                    &actor,
                                    &mut active_session,
                                    &mut app,
                                    &mut task_state,
                                    &mut sync,
                                    ev,
                                );
                                // 把缓冲区积压的 delta 一次性排空，降低高并发流式输出时的重绘次数
                                if next_action == AppAction::None {
                                    loop {
                                        match client_rx.try_recv() {
                                            Ok(more) => {
                                                let (act, ev_dirty) = process_actor_client_event_dirty(
                                                    &actor,
                                                    &mut active_session,
                                                    &mut app,
                                                    &mut task_state,
                                                    &mut sync,
                                                    more,
                                                );
                                                dirty |= ev_dirty;
                                                if act != AppAction::None {
                                                    next_action = act;
                                                    break;
                                                }
                                            }
                                            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
                                            Err(
                                                tokio::sync::broadcast::error::TryRecvError::Empty
                                                | tokio::sync::broadcast::error::TryRecvError::Closed,
                                            ) => break,
                                        }
                                    }
                                }
                                (next_action, dirty)
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
                                    LoopClientMsg::Bus(ev),
                                );
                                if next_action == AppAction::None {
                                    while let Ok(more) = ui_rx.try_recv() {
                                        let (act, ev_dirty) = process_loop_ui_msg_dirty(
                                            &mut app,
                                            &mut task_state,
                                            LoopClientMsg::Bus(more),
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
                    Some((panel_gen, panel_sid, snapshot)) = panel_rx.recv() => {
                        let dirty = if panel_gen == panel_ctrl.current_gen()
                            && panel_sid == app.session_id
                            && app.features.side_panel
                            && app.side_panel.open
                        {
                            snapshot.apply_to_side_panel(&mut app.side_panel_data, &app.anomalies);
                            true
                        } else {
                            false
                        };
                        (AppAction::None, dirty)
                    }
                };

                needs_redraw |= step_dirty || action != AppAction::None;

                if action != AppAction::None {
                    sync.sync_config_if_changed(&actor, &app).await;
                }
                handle_action(
                    &actor,
                    &runtime,
                    &config,
                    &mut app,
                    &mut active_session,
                    &mut terminal,
                    &mut term_stream,
                    &loop_tx,
                    &mut task_state,
                    &mut panel_ctrl,
                    &mut sync,
                    action,
                )
                .await;
                sync.sync_config_if_changed(&actor, &app).await;
            }
            Ok(())
        }
        .await;

        if let Some(task) = update_check_task {
            task.abort();
        }
        panel_ctrl.abort();
        run_loop_res
    }
    .await;

    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    emit_program_status(ProgramStatusState::Idle, None, None);
    task_state.abort_for_session_switch();
    actor.shutdown().await;
    for guard in guards.into_iter().rev() {
        drop(guard);
    }
    loop_res?;

    Ok(TuiExitSummary {
        session_id: app.session_id.clone(),
        persisted: runtime.persistence.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mizar_tui_ui::app::{AppAction, AppState, ChatItem, UiEvent};

    use super::{
        ActiveTaskKind, LoopClientMsg, RunController, install_tui_emphasis, process_loop_ui_msg,
    };

    fn process_ui_event(
        app: &mut AppState,
        task_state: &mut RunController,
        event: UiEvent,
    ) -> AppAction {
        process_loop_ui_msg(app, task_state, LoopClientMsg::Bus(event))
    }

    #[test]
    fn process_ui_event_records_user_items_for_cancel_and_queued_prompts() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queue",
            Vec::new(),
            None,
            None,
            None,
        );
        app.set_busy_state(true);
        app.queue.enqueue("queued follow-up", Vec::new());

        let mut task_state = RunController::default();
        let (_rid1, cancel1) = task_state.start("sess-queue".into(), ActiveTaskKind::AgentRun);
        task_state.cancel_active();
        assert!(cancel1.is_cancelled());

        let act1 = process_ui_event(
            &mut app,
            &mut task_state,
            UiEvent::RunFinished(Err("request aborted".into())),
        );
        assert_eq!(act1, AppAction::RefreshSidePanelData);
        assert_eq!(
            app.queue
                .items()
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
        // 正常结束时由 SessionActor run_wake_loop 独占出队并经 QueueUpdated 同步，runner 不得重复派发 SubmitPrompt
        assert_eq!(act2, AppAction::RefreshSidePanelData);
        assert_eq!(app.queue.len(), 1);
    }

    #[tokio::test]
    async fn spawn_local_bash_run_executes_persists_and_supports_cancel() {
        use mizar_session::SessionStore;

        let store = SessionStore::new();
        let session = store.create(Some("runner-bash".into()));

        let mut app = AppState::new(
            std::env::temp_dir(),
            &session.id,
            Vec::new(),
            None,
            None,
            None,
        );
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut task_state = RunController::default();

        super::actions::spawn_local_bash_run(
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
                LoopClientMsg::Task {
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
        super::actions::spawn_local_bash_run(
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
                LoopClientMsg::Task {
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
    }

    #[tokio::test]
    async fn settings_module_entries_toggle_persists_disabled_plugins_and_refreshes_runtime_state()
    {
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
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

        let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["compaction"]);
        if let mizar_core::spine::AgentToolsMode::Coding { ref mut mcp, .. } = cfg.tools_mode {
            mcp.global_config_dir = Some(tmp.join(".mizar"));
        }
        let runtime = MizarRuntime::build(cfg).await.unwrap();

        let session = runtime.sessions.create(None);
        let mut app = AppState::new(ws.clone(), session.id.clone(), Vec::new(), None, None, None);
        app.user_skill_names = super::collect_user_skill_names(&runtime, &ws).await;
        assert!(app.user_skill_names.contains(&"tui-demo".to_string()));
        super::actions::populate_side_panel_data(&runtime, &mut app).await;
        assert!(!app.side_panel_data.instructions.is_empty());
        assert!(!app.side_panel_data.tools.is_empty());

        // 启动时 compaction 被屏蔽：/settings 中不含 [压缩] 配置项，且 plugin.compaction 为 false
        let initial_entries = super::build_settings_entries(&runtime, &app);
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
        super::actions::apply_module_toggle_setting(
            &runtime,
            &mut app,
            Some(&settings_path),
            "compaction",
            true,
        )
        .await;
        assert!(runtime.compaction.as_ref().unwrap().is_module_enabled());
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
            super::actions::apply_module_toggle_setting(
                &runtime,
                &mut app,
                Some(&settings_path),
                id,
                false,
            )
            .await;
        }
        assert!(app.user_skill_names.is_empty());
        assert!(app.side_panel_data.instructions.is_empty());
        assert!(!runtime.compaction.as_ref().unwrap().is_module_enabled());
        let after_disable_entries = super::build_settings_entries(&runtime, &app);
        assert!(!after_disable_entries.iter().any(|e| e.category == "压缩"));
        let persisted_2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(
            persisted_2["disabledPlugins"],
            serde_json::json!(["compaction", "instructions", "skills"])
        );

        // 3. 运行期繁忙保护：is_compacting 时拒绝启用 skills，状态与 settings.json 保持不变
        app.set_compacting_state("手动压缩中…");
        super::actions::apply_module_toggle_setting(
            &runtime,
            &mut app,
            Some(&settings_path),
            "skills",
            true,
        )
        .await;
        assert!(!runtime.skills.as_ref().unwrap().is_enabled());
        let persisted_3: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(
            persisted_3["disabledPlugins"],
            serde_json::json!(["compaction", "instructions", "skills"])
        );

        // 4. process_ui_event 收到 ToolsUpdated 应触发 RefreshSidePanelTools
        let mut task_state = RunController::default();
        let act = process_ui_event(&mut app, &mut task_state, UiEvent::ToolsUpdated);
        assert_eq!(act, AppAction::RefreshSidePanelTools);

        runtime.shutdown().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn process_ui_event_retains_queued_prompts_after_max_steps_or_truncated() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use mizar_tui_ui::app::{ChatItem, NoticeLevel, UiStepOutcome, UiTurnEndReason};

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
            let mut task_state = RunController::default();

            app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
            app.input.insert_str("next queued prompt");
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert_eq!(app.queue.len(), 1);

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
            assert_eq!(app.queue.len(), 1);
            assert_eq!(app.queue.items()[0].text, "next queued prompt");
            assert!(app.items.iter().any(|item| matches!(
                item,
                ChatItem::Notice {
                    level: NoticeLevel::Warn,
                    text,
                } if text.contains("已保留 1 条排队消息")
            )));
        }

        // 正常 Completed 时保留给 SessionActor 自动出队，process_loop_ui_msg 不提前清空或重复 SubmitPrompt。
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queued-complete",
            Vec::new(),
            None,
            None,
            None,
        );
        let mut task_state = RunController::default();
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
        assert_eq!(action, AppAction::RefreshSidePanelData);
        assert_eq!(app.queue.len(), 1);
    }

    #[test]
    fn process_loop_ui_msg_ignores_stale_task_events_after_run_or_session_switch() {
        use mizar_tui_ui::app::LocalBashOutput;

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-current",
            Vec::new(),
            None,
            None,
            None,
        );
        let mut task_state = RunController::default();
        let (stale_run_id, stale_cancel) =
            task_state.start("sess-current".into(), ActiveTaskKind::AgentRun);
        let (current_run_id, current_cancel) =
            task_state.start("sess-current".into(), ActiveTaskKind::LocalBash);
        assert!(stale_cancel.is_cancelled());
        assert!(!current_cancel.is_cancelled());
        app.set_local_bash_state();

        // 1. 旧 run_id 的 RunFinished / ToolUpdate 不得清空新任务的 RunController 或更改 UI 状态
        let stale_act = process_loop_ui_msg(
            &mut app,
            &mut task_state,
            LoopClientMsg::Task {
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
            LoopClientMsg::Task {
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
            LoopClientMsg::Task {
                run_id: current_run_id,
                session_id: "sess-current".into(),
                event: UiEvent::LocalBashFinished {
                    call_id: "local-bash-1".into(),
                    command: "echo hi".into(),
                    exclude_from_context: false,
                    output: Ok(LocalBashOutput {
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
            LoopClientMsg::Task {
                run_id: compact_rid,
                session_id: "sess-next".into(),
                event: UiEvent::CompactionFinished {
                    outcome: "failed".into(),
                    tokens_before: None,
                    tokens_after: None,
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
    async fn build_session_info_report_thin_wrappers_match_app_and_config() {
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_session::{
            CompactionDigest, SessionCreateOptions, SessionEvent, SessionTraceEvent,
        };

        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();
        let store = &runtime.sessions;

        let parent = store.create(None);
        parent.append(SessionEvent::user("parent"));
        let child_id = "6f3d2c1a-8b4e-4f2a-9c71-2e5d8a0b4f13";
        let child = store
            .fork(
                &parent.id,
                1,
                SessionCreateOptions {
                    id: Some(child_id.into()),
                    cwd: Some("/repo/session-ws".into()),
                    title: Some("feat(session): /session".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        child.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
        child.append(SessionEvent::Assistant {
            blocks: vec![mizar_llm::ContentBlock::Text { text: "ok".into() }],
            replay: None,
            usage: Some(mizar_llm::TokenUsage {
                input_tokens: 10_000,
                output_tokens: 8_700,
                cache_read_tokens: Some(40_000),
                cache_write_tokens: Some(2_000),
                ..Default::default()
            }),
        });
        child.compact_at(
            "sum",
            2,
            Some(42_100),
            Some(6_800),
            CompactionDigest::default(),
        );

        let app = AppState::new(
            PathBuf::from("/completely/different/workspace"),
            &child.id,
            Vec::new(),
            None,
            None,
            None,
        );
        let fake_db = PathBuf::from("/tmp/mizar_sessions.sqlite");
        let mut tui_cfg = super::TuiConfig::new("/repo/session-ws");
        tui_cfg.session_db_path = Some(fake_db.clone());

        let report_at = super::build_session_info_report_at(
            &child,
            &app,
            true,
            Some(&fake_db),
            child.header().updated_at_ms + 3 * 60 * 1_000,
        );
        assert!(
            report_at.contains(&format!("ID       {child_id}")),
            "{report_at}"
        );
        assert!(
            report_at.contains("工作区   /repo/session-ws"),
            "{report_at}"
        );
        assert!(
            report_at.contains("上下文   6.8k / 128k（5.3%）"),
            "{report_at}"
        );

        let report_live = super::build_session_info_report(&runtime, &tui_cfg, &child, &app);
        assert!(
            report_live.contains(&format!("ID       {child_id}")),
            "{report_live}"
        );
        assert!(report_live.contains("仅内存（未持久化）"), "{report_live}");

        runtime.shutdown().await;
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
        let mut task_state = RunController::default();
        let (_, d_start) = super::process_loop_ui_msg_dirty(
            &mut app,
            &mut task_state,
            LoopClientMsg::Bus(UiEvent::TurnStart { turn: 1 }),
        );
        assert!(d_start);
        // 首个 TextDelta 创建 Assistant 条目，先触发首帧展示
        let (_, d_first) = super::process_loop_ui_msg_dirty(
            &mut app,
            &mut task_state,
            LoopClientMsg::Bus(UiEvent::TextDelta {
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
            LoopClientMsg::Bus(UiEvent::TextDelta {
                text: " world".into(),
                ts_ms: 2,
            }),
        );
        assert_eq!(act2, AppAction::None);
        assert!(!d_next);
        assert!(app.reveal.has_pending());
    }

    #[tokio::test]
    async fn tui_emphasis_plugin_respects_enable_and_dispose() {
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_tui_ui::emphasis::EmphasisConfig;
        use std::sync::Arc;

        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();
        let sp = &runtime.system_prompt;

        assert!(!sp.build().contains("These rules shape replies"));

        // 1. enabled=false 时不注入
        let disabled_cfg = EmphasisConfig {
            enabled: false,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        assert!(install_tui_emphasis(&runtime, &disabled_cfg).is_none());
        assert!(!sp.build().contains("These rules shape replies"));

        // 2. enabled=true && guidance=true 时挂载注入，释放守卫后干净移除
        let enabled_cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let guard1 = install_tui_emphasis(&runtime, &enabled_cfg).unwrap();
        assert!(sp.build().contains("These rules shape replies"));

        drop(guard1);
        assert!(!sp.build().contains("These rules shape replies"));

        // 3. 子代理会话调用 build_for_session 时，不得收到 TUI 回复样式引导
        let sessions = &runtime.sessions;
        let root_sess = sessions.create(Some("tui-root".into()));
        let child_sess = sessions.create_with(mizar_session::SessionCreateOptions {
            id: Some("tui-child".into()),
            parent_id: Some(root_sess.id.clone()),
            origin: Some(mizar_session::SessionOrigin::Subagent),
            ..Default::default()
        });
        let guard2 = install_tui_emphasis(&runtime, &enabled_cfg).unwrap();
        assert!(
            sp.build_for_session(Some(&root_sess.id))
                .contains("These rules shape replies")
        );
        assert!(
            !sp.build_for_session(Some(&child_sess.id))
                .contains("These rules shape replies")
        );
        drop(guard2);
        let _ = Arc::strong_count(&runtime);

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn run_tui_with_summary_refuses_to_start_when_tui_plugin_disabled() {
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use std::sync::Arc;

        let raw_before = crossterm::terminal::is_raw_mode_enabled().ok();
        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();

        let mut cfg = super::TuiConfig::new(std::env::temp_dir());
        cfg.disabled_plugins = vec![mizar_types::client::PLUGIN_TUI.to_string()];

        let err = super::run_tui_with_summary(Arc::clone(&runtime), cfg)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("tui"), "unexpected error: {err}");

        if let Some(before) = raw_before {
            assert_eq!(
                crossterm::terminal::is_raw_mode_enabled().ok(),
                Some(before)
            );
        }
        assert!(
            !runtime
                .system_prompt
                .build()
                .contains("These rules shape replies")
        );

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn runtime_feature_gates_block_disabled_clipboard_local_bash_and_side_panel() {
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_tui_ui::TuiFeatures;
        use mizar_types::client::{
            HostExtensionFlags, PLUGIN_LOCAL_BASH, PLUGIN_TUI_CLIPBOARD, PLUGIN_TUI_SIDE_PANEL,
        };

        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();
        let session = runtime.sessions.create(None);

        let flags = HostExtensionFlags::from_disabled_plugins(&[
            PLUGIN_TUI_CLIPBOARD.to_string(),
            PLUGIN_LOCAL_BASH.to_string(),
            PLUGIN_TUI_SIDE_PANEL.to_string(),
        ]);
        let mut app = AppState::new(
            std::env::temp_dir(),
            &session.id,
            Vec::new(),
            None,
            None,
            None,
        )
        .with_features(TuiFeatures::from(&flags));

        // 1. clipboard 禁用时 apply_clipboard_content 拒绝写入输入草稿
        let ws = app.workspace.clone();
        super::apply_clipboard_content(
            &mut app,
            &ws,
            Ok(crate::clipboard::ClipboardContent::Text(
                "should not paste".into(),
            )),
        )
        .await;
        assert!(app.input.is_empty());
        assert!(
            app.active_flash()
                .is_some_and(|s| s.contains("剪贴板扩展已禁用"))
        );

        // 2. local-bash 禁用时 spawn_local_bash_run 拒绝启动后台命令
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut task_state = RunController::default();
        super::actions::spawn_local_bash_run(
            &mut app,
            &session,
            &tx,
            &mut task_state,
            "echo blocked".into(),
            false,
        );
        assert!(!app.is_bash_running());
        assert!(rx.try_recv().is_err());
        assert!(session.log().is_empty());

        // 3. side-panel 禁用时 populate_side_panel_data 跳过侧栏与工具/Git 采集
        app.side_panel.open = true;
        super::actions::populate_side_panel_data(&runtime, &mut app).await;
        assert!(app.side_panel_data.tools.is_empty());
        assert!(app.side_panel_data.git_changes.is_empty());

        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn handle_steer_prompt_steers_without_cancel_while_cancel_run_hard_aborts_and_wake_drains_followup()
     {
        use std::sync::Arc;

        use futures::channel::mpsc as futures_mpsc;
        use mizar_client::actor::{SessionActor, SessionActorConfig};
        use mizar_core::agent_loop::{DeliveryMode, TurnInput};
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_llm::{
            BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, Message, ModelInfo,
            ProviderInfo, ReasoningEffort, StreamChunk, ToolCall,
        };
        use mizar_tools::{ToolDefinition, ToolOutput};
        use mizar_types::client::ClientEvent;
        use tokio::sync::Notify;

        struct SteerTuiAdapter {
            recorded: Arc<std::sync::Mutex<Vec<Vec<Message>>>>,
        }

        impl LlmAdapter for SteerTuiAdapter {
            fn info(&self) -> Option<ProviderInfo> {
                Some(ProviderInfo {
                    id: "mock-tui".into(),
                    name: "Mock TUI".into(),
                    protocol: "mock".into(),
                    models: vec![ModelInfo {
                        id: "m-tui".into(),
                        name: "Mock Model".into(),
                        context_window: Some(32_000),
                        max_tokens: Some(4_096),
                        reasoning_efforts: vec![ReasoningEffort::Off],
                        input: Vec::new(),
                        ..Default::default()
                    }],
                })
            }

            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.recorded.lock().unwrap().push(options.messages.clone());
                let joined: String = options
                    .messages
                    .iter()
                    .map(|m| m.text_content())
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_wake = joined.rsplit_once("wake check").is_some();
                let needs_abort_gate = !has_wake
                    && joined
                        .rsplit_once("abort turn")
                        .is_some_and(|(_, tail)| !tail.contains("gate_done"));
                let needs_first_gate =
                    !has_wake && !joined.contains("abort turn") && !joined.contains("gate_done");
                let has_steer = joined.contains("gentle steer");
                let (tx, rx) = futures_mpsc::unbounded();
                tokio::spawn(async move {
                    if has_wake {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("woke followup ok")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else if needs_first_gate || needs_abort_gate {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: if needs_abort_gate {
                                    "call_gate_abort".into()
                                } else {
                                    "call_gate".into()
                                },
                                name: "tui_gate".into(),
                                arguments: "{}".into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else if has_steer {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("steered reply")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("plain reply")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();

        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let adapter = Arc::new(SteerTuiAdapter {
            recorded: Arc::clone(&recorded),
        });
        runtime.register_adapter("mock-tui", adapter).unwrap();

        let started_round = Arc::new(Notify::new());
        let release_round = Arc::new(Notify::new());
        let started_sig = Arc::clone(&started_round);
        let release_sig = Arc::clone(&release_round);
        let _tool_reg = runtime
            .tools
            .register(ToolDefinition::async_rich_tool(
                "tui_gate",
                "blocks until released or cancelled",
                serde_json::json!({}),
                move |_, call_ctx| {
                    let s = Arc::clone(&started_sig);
                    let r = Arc::clone(&release_sig);
                    async move {
                        s.notify_one();
                        tokio::select! {
                            _ = r.notified() => Ok::<_, mizar_types::BoxError>(ToolOutput::text("gate_done")),
                            _ = call_ctx.cancel.cancelled() => Err("cancelled".into()),
                        }
                    }
                },
            ))
            .unwrap();

        let mut sess = runtime.sessions.create(None);
        let cfg = super::TuiConfig::new(std::env::temp_dir());
        let mut app = AppState::new(
            std::env::temp_dir(),
            &sess.id,
            Vec::new(),
            Some("mock-tui".into()),
            Some("m-tui".into()),
            None,
        );
        let actor = SessionActor::spawn(
            Arc::clone(&runtime),
            Arc::clone(&sess),
            SessionActorConfig::new(std::env::temp_dir()).with_provider_model(
                Some("mock-tui".into()),
                Some("m-tui".into()),
                None,
            ),
        );
        let mut client_rx = actor.subscribe_client_events();
        let mut sync = super::actions::TuiActorSync::new(&sess, &app);
        let mut task_state = super::RunController::default();

        // 1. 启动第一轮（进入阻塞工具），然后发温和 SteerPrompt：不得取消活跃轮次
        app.push_user_message("first turn");
        super::actions::submit_turn_via_actor(
            &actor,
            &mut app,
            &mut sync,
            "first turn".into(),
            Vec::new(),
        )
        .await;
        started_round.notified().await;

        super::actions::handle_steer_prompt_via_actor(
            &actor,
            &cfg,
            &mut app,
            &mut sync,
            "gentle steer".into(),
            Vec::new(),
        )
        .await;
        assert!(app.active_flash().is_some_and(|s| s.contains("步间插话")));

        release_round.notify_one();
        let finish_msg = loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let is_done = match &ev {
                ClientEvent::RunFinished(res) => Some(res.clone()),
                _ => None,
            };
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if let Some(res) = is_done {
                break res;
            }
        };
        assert_eq!(finish_msg.unwrap(), "steered reply");
        assert_eq!(
            app.items
                .iter()
                .filter(|it| matches!(it, ChatItem::User { text } if text == "gentle steer"))
                .count(),
            1
        );

        // 2. 启动第二轮并预塞 FollowUp，然后执行硬取消：立即取消、清空用户项并保留后台通知为 Inject
        let agent_loop = &runtime.agent_loop;
        app.push_user_message("abort turn");
        super::actions::submit_turn_via_actor(
            &actor,
            &mut app,
            &mut sync,
            "abort turn".into(),
            Vec::new(),
        )
        .await;
        started_round.notified().await;
        agent_loop.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::user("queued-drop"),
        );
        agent_loop.inbox().push_appending(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("subagent:w1", "bg result kept across esc"),
            "subagent:settled",
            "subagent:batch",
        );

        super::actions::cancel_active_run_via_actor(&actor, &mut app, &mut task_state, &mut sync)
            .await;
        assert!(agent_loop.inbox().pop_followup(&sess.id).is_none());
        let kept = agent_loop.inbox().snapshot(&sess.id);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].mode, DeliveryMode::Inject);
        assert!(kept[0].input.text().contains("bg result kept across esc"));
        agent_loop.inbox().clear(&sess.id);

        let abort_res = loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let is_done = match &ev {
                ClientEvent::RunFinished(res) => Some(res.clone()),
                _ => None,
            };
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if let Some(res) = is_done {
                break res;
            }
        };
        assert!(abort_res.is_err());

        // 3. 空闲时推送 FollowUp 由 SessionActor 内部唤醒循环自动启动，无需 runner 自己订阅 subscribe_wake
        let wake_push = agent_loop.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "wake check"),
        );
        assert_eq!(wake_push.effective_mode, DeliveryMode::FollowUp);
        assert!(!wake_push.degraded);

        let followup_res = loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let is_done = match &ev {
                ClientEvent::RunFinished(res) => Some(res.clone()),
                _ => None,
            };
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if let Some(res) = is_done {
                break res;
            }
        };
        assert_eq!(followup_res.unwrap(), "woke followup ok");
        assert!(app.items.iter().any(|it| matches!(
            it,
            ChatItem::Notice { text, .. } if text.contains("[cron] wake check")
        )));

        actor.shutdown().await;
        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn session_actor_tui_dismisses_active_approval_overlay_on_approval_resolved() {
        use std::sync::Arc;

        use futures::channel::mpsc as futures_mpsc;
        use mizar_client::actor::{SessionActor, SessionActorConfig};
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_llm::{
            BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, ModelInfo, ProviderInfo,
            StreamChunk, ToolCall,
        };
        use mizar_tools::{ToolDefinition, ToolOutput};
        use mizar_tui_ui::app::ApprovalMode;
        use mizar_tui_ui::overlay::OverlayKind;
        use mizar_types::client::ClientEvent;
        use mizar_types::rpc::{InteractionResolveParams, ResolvedBy, RpcApprovalDecision};

        struct ApprovalTuiAdapter;

        impl LlmAdapter for ApprovalTuiAdapter {
            fn info(&self) -> Option<ProviderInfo> {
                Some(ProviderInfo {
                    id: "mock-apr".into(),
                    name: "Mock Approval".into(),
                    protocol: "mock".into(),
                    models: vec![ModelInfo {
                        id: "m-apr".into(),
                        name: "Mock Approval".into(),
                        ..Default::default()
                    }],
                })
            }

            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                let joined: String = options
                    .messages
                    .iter()
                    .map(|m| m.text_content())
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_tool_done = joined.contains("secret written");
                let (tx, rx) = futures_mpsc::unbounded();
                tokio::spawn(async move {
                    if !has_tool_done {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "call-apr-tui".into(),
                                name: "write_secret".into(),
                                arguments: r#"{"file":"a.txt"}"#.into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("approval completed")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let ws = std::env::temp_dir().join(format!("mizar_tui_apr_dismiss_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(&ws).unwrap();

        let runtime = MizarRuntime::build(AgentSpineConfig::new(&ws))
            .await
            .unwrap();
        runtime
            .register_adapter("mock-apr", Arc::new(ApprovalTuiAdapter))
            .unwrap();
        let _tool_reg = runtime
            .tools
            .register(ToolDefinition::async_rich_tool(
                "write_secret",
                "requires interactive approval",
                serde_json::json!({}),
                |_, _| async { Ok::<_, mizar_types::BoxError>(ToolOutput::text("secret written")) },
            ))
            .unwrap();

        let mut sess = runtime.sessions.create(None);
        let mut app = AppState::new(
            ws.clone(),
            &sess.id,
            Vec::new(),
            Some("mock-apr".into()),
            Some("m-apr".into()),
            None,
        );
        app.approval_mode = ApprovalMode::Ask;

        let actor = SessionActor::spawn(
            Arc::clone(&runtime),
            Arc::clone(&sess),
            SessionActorConfig::new(&ws)
                .with_approval_mode(ApprovalMode::Ask)
                .with_provider_model(Some("mock-apr".into()), Some("m-apr".into()), None),
        );
        let mut client_rx = actor.subscribe_client_events();
        let mut sync = super::actions::TuiActorSync::new(&sess, &app);
        let mut task_state = super::RunController::default();

        app.push_user_message("run write_secret");
        super::actions::submit_turn_via_actor(
            &actor,
            &mut app,
            &mut sync,
            "run write_secret".into(),
            Vec::new(),
        )
        .await;

        let req_id = loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if let Some(OverlayKind::Approval(ref ap)) = app.overlay.active {
                break ap.request_id.clone();
            }
        };

        // 异源（RPC）抢占裁决后，SessionActor 广播 ClientEvent::ApprovalResolved，TUI 应立即关闭弹窗并完成工具执行
        let res = actor
            .resolve_interaction(
                InteractionResolveParams {
                    session_id: Some(sess.id.clone()),
                    request_id: req_id.clone(),
                    decision: RpcApprovalDecision::AllowOnce,
                    note: None,
                },
                ResolvedBy::Rpc,
            )
            .await
            .unwrap();
        assert!(res.ok);

        let mut saw_resolved = false;
        let finish_res = loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let resolved_hit = matches!(
                &ev,
                ClientEvent::ApprovalResolved { request_id: rid } if rid == &req_id
            );
            let done = match &ev {
                ClientEvent::RunFinished(r) => Some(r.clone()),
                _ => None,
            };
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if resolved_hit {
                saw_resolved = true;
                assert!(
                    app.overlay.active.is_none(),
                    "active approval overlay must dismiss on ApprovalResolved"
                );
            }
            if let Some(r) = done {
                break r;
            }
        };
        assert!(saw_resolved);
        assert_eq!(finish_res.unwrap(), "approval completed");
        assert!(app.overlay.active.is_none());

        actor.shutdown().await;
        runtime.shutdown().await;
        let _ = std::fs::remove_dir_all(&ws);
    }

    #[tokio::test]
    async fn resume_session_on_locked_session_surfaces_error_without_switching_active_session() {
        use std::process::{Child, Command, Stdio};
        use std::sync::Arc;

        use mizar_client::actor::{SessionActor, SessionActorConfig};
        use mizar_client::run_dir::{ProcessMeta, meta_path_for_pid};
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_tui_ui::app::NoticeLevel;

        struct ChildGuard(Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }

        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "mizar_tui_locked_resume_{}_{uniq}",
            std::process::id()
        ));
        let ws = root.join("ws");
        let run_dir = root.join("run");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&run_dir).unwrap();

        let mut cfg = super::TuiConfig::new(&ws);
        assert!(!cfg.enable_rpc_socket);
        assert!(cfg.run_dir.is_none());
        cfg.run_dir = Some(run_dir.clone());

        let runtime = MizarRuntime::build(AgentSpineConfig::new(&ws))
            .await
            .unwrap();
        let mut active_session = runtime.sessions.create(Some("sess-init".into()));
        let sess_locked = runtime.sessions.create(Some("sess-locked".into()));
        let sess_free = runtime.sessions.create(Some("sess-free".into()));

        let sleeper = ChildGuard(
            Command::new("sleep")
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let other_pid = sleeper.0.id();
        let other_meta = ProcessMeta {
            pid: other_pid,
            ppid: None,
            tty: None,
            herdr_pane_id: None,
            session_id: sess_locked.id.clone(),
            cwd: ws.display().to_string(),
            socket: run_dir
                .join(format!("p-{other_pid}.sock"))
                .display()
                .to_string(),
            provider: None,
            model: None,
            started_at_ms: 1,
        };
        std::fs::write(
            meta_path_for_pid(&run_dir, other_pid),
            serde_json::to_vec(&other_meta).unwrap(),
        )
        .unwrap();

        let actor = SessionActor::spawn(
            Arc::clone(&runtime),
            Arc::clone(&active_session),
            SessionActorConfig::new(&ws).with_run_dir(&run_dir),
        );
        let mut app = AppState::new(ws.clone(), &active_session.id, Vec::new(), None, None, None);
        let mut sync = super::actions::TuiActorSync::new(&active_session, &app);
        let mut task_state = super::RunController::default();

        let switched_locked = super::actions::resume_session_via_actor(
            &actor,
            &runtime,
            &cfg,
            &mut app,
            &mut active_session,
            &mut sync,
            &mut task_state,
            &sess_locked.id,
        )
        .await;
        assert!(!switched_locked);
        assert_eq!(active_session.id, "sess-init");
        assert_eq!(actor.active_session_id(), "sess-init");
        assert_eq!(app.session_id, "sess-init");
        assert!(app.items.iter().any(|it| matches!(
            it,
            ChatItem::Notice {
                level: NoticeLevel::Error,
                text,
            } if text.contains(&other_pid.to_string()) && text.contains("单写者锁")
        )));

        // 未被占用的会话应正常切换
        let switched_free = super::actions::resume_session_via_actor(
            &actor,
            &runtime,
            &cfg,
            &mut app,
            &mut active_session,
            &mut sync,
            &mut task_state,
            &sess_free.id,
        )
        .await;
        assert!(switched_free);
        assert_eq!(active_session.id, "sess-free");
        assert_eq!(actor.active_session_id(), "sess-free");
        assert_eq!(app.session_id, "sess-free");

        actor.shutdown().await;
        runtime.shutdown().await;
        drop(sleeper);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn session_actor_tui_queue_drain_and_config_sync_work_end_to_end() {
        use std::sync::Arc;

        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        use futures::channel::mpsc as futures_mpsc;
        use mizar_client::actor::{SessionActor, SessionActorConfig};
        use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
        use mizar_llm::{
            BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, ModelInfo, ProviderInfo,
            ReasoningEffort, StreamChunk, ToolCall,
        };
        use mizar_tools::{ToolDefinition, ToolOutput};
        use mizar_tui_ui::app::ApprovalMode;
        use mizar_types::client::ClientEvent;
        use mizar_types::rpc::RpcApprovalMode;
        use tokio::sync::Notify;

        type SeenModelCalls = Arc<std::sync::Mutex<Vec<(Option<String>, Option<ReasoningEffort>)>>>;

        struct QueueSyncAdapter {
            seen_models: SeenModelCalls,
        }

        impl LlmAdapter for QueueSyncAdapter {
            fn info(&self) -> Option<ProviderInfo> {
                Some(ProviderInfo {
                    id: "mock-q".into(),
                    name: "Mock Queue".into(),
                    protocol: "mock".into(),
                    models: vec![
                        ModelInfo {
                            id: "m-1".into(),
                            name: "M1".into(),
                            reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
                            ..Default::default()
                        },
                        ModelInfo {
                            id: "m-2".into(),
                            name: "M2".into(),
                            ..Default::default()
                        },
                    ],
                })
            }

            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.seen_models
                    .lock()
                    .unwrap()
                    .push((options.model.clone(), options.reasoning_effort));
                let joined: String = options
                    .messages
                    .iter()
                    .map(|m| m.text_content())
                    .collect::<Vec<_>>()
                    .join("\n");
                let is_second = joined.contains("queued-msg");
                let needs_gate = !is_second && !joined.contains("qgate_done");
                let (tx, rx) = futures_mpsc::unbounded();
                tokio::spawn(async move {
                    if needs_gate {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "call-qgate".into(),
                                name: "qgate".into(),
                                arguments: "{}".into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else if is_second {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("queued done")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("first done")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::temp_dir()))
            .await
            .unwrap();
        let seen_models = Arc::new(std::sync::Mutex::new(Vec::new()));
        let adapter = Arc::new(QueueSyncAdapter {
            seen_models: Arc::clone(&seen_models),
        });
        runtime.register_adapter("mock-q", adapter).unwrap();

        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let s_sig = Arc::clone(&started);
        let r_sig = Arc::clone(&release);
        let _tool = runtime
            .tools
            .register(ToolDefinition::async_rich_tool(
                "qgate",
                "blocks for queue test",
                serde_json::json!({}),
                move |_, _| {
                    let s = Arc::clone(&s_sig);
                    let r = Arc::clone(&r_sig);
                    async move {
                        s.notify_one();
                        r.notified().await;
                        Ok::<_, mizar_types::BoxError>(ToolOutput::text("qgate_done"))
                    }
                },
            ))
            .unwrap();

        let mut sess = runtime.sessions.create(None);
        let cfg = super::TuiConfig::new(std::env::temp_dir());
        let mut app = AppState::new(
            std::env::temp_dir(),
            &sess.id,
            runtime.llm.provider_infos(),
            Some("mock-q".into()),
            Some("m-1".into()),
            Some(ReasoningEffort::Low),
        );
        let actor = SessionActor::spawn(
            Arc::clone(&runtime),
            Arc::clone(&sess),
            SessionActorConfig::new(std::env::temp_dir()).with_provider_model(
                Some("mock-q".into()),
                Some("m-1".into()),
                Some(ReasoningEffort::Low),
            ),
        );
        let mut client_rx = actor.subscribe_client_events();
        let mut sync = super::actions::TuiActorSync::new(&sess, &app);
        let mut task_state = super::RunController::default();

        // 1. 发起首轮并在工具阻塞期间通过键盘排队一条消息 -> 同步进 SessionActor 队列
        app.push_user_message("first prompt");
        super::actions::submit_turn_via_actor(
            &actor,
            &mut app,
            &mut sync,
            "first prompt".into(),
            Vec::new(),
        )
        .await;
        started.notified().await;
        app.set_busy_state(true);

        let prev_q = app.queue.items().to_vec();
        app.input.insert_str("queued-msg");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(
            super::actions::sync_queue_after_term_event(
                &actor, &cfg, &mut app, &mut sync, &prev_q,
            )
            .await
        );
        let snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(snap.queued_inputs.len(), 1);
        assert_eq!(snap.queued_inputs[0].text, "queued-msg");
        assert_eq!(app.queue.items()[0].id, snap.queued_inputs[0].id);

        // 2. 放行第一轮 -> SessionActor 自动出队执行 queued-msg，TUI 队列清空且渲染 queued-msg 气泡
        release.notify_one();
        let mut finished_count = 0;
        loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let is_fin = matches!(ev, ClientEvent::RunFinished(_));
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if is_fin {
                finished_count += 1;
                if finished_count == 2 {
                    break;
                }
            }
        }
        assert!(app.queue.is_empty());
        assert!(app.items.iter().any(|it| matches!(
            it,
            ChatItem::User { text } if text == "queued-msg"
        )));

        // 3. 切换 /effort、/approval 与 /model 并通过 sync_config_if_changed 同步到 SessionActor，验证后续回合生效
        app.input.set_text_end("/approval ask");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.input.set_text_end("/effort high");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.model = Some("m-2".into());
        sync.sync_config_if_changed(&actor, &app).await;
        let snap_after = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(snap_after.config.approval_mode, RpcApprovalMode::Ask);
        assert_eq!(
            snap_after.config.reasoning_effort,
            Some(ReasoningEffort::High)
        );
        assert_eq!(snap_after.config.model.as_deref(), Some("m-2"));
        assert_eq!(app.approval_mode, ApprovalMode::Ask);

        app.push_user_message("verify updated config");
        super::actions::submit_turn_via_actor(
            &actor,
            &mut app,
            &mut sync,
            "verify updated config".into(),
            Vec::new(),
        )
        .await;
        loop {
            let ev = super::poll_client_event(&mut client_rx).await.unwrap();
            let done = matches!(ev, ClientEvent::RunFinished(_));
            let _ = super::actions::process_actor_client_event_dirty(
                &actor,
                &mut sess,
                &mut app,
                &mut task_state,
                &mut sync,
                ev,
            );
            if done {
                break;
            }
        }
        assert_eq!(
            seen_models.lock().unwrap().last(),
            Some(&(Some("m-2".into()), Some(ReasoningEffort::High)))
        );

        actor.shutdown().await;
        runtime.shutdown().await;
    }

    #[test]
    fn inline_finalized_commit_limit_flushes_settled_prefix_while_busy() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-inline-inc",
            Vec::new(),
            None,
            None,
            None,
        );
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.push_user_message("hello");
        app.apply_ui_event(UiEvent::TextDelta {
            text: "thinking done, calling tool".into(),
            ts_ms: 10,
        });
        // 正在流式输出的 Assistant 尚未定型，仅其前面的 User 已可推进 scrollback
        assert_eq!(super::inline_finalized_commit_limit(&app), 1);

        // ToolCall 到来会闭合上一条 Assistant，且工具仍在运行，因此 User + Assistant 均可增量提交
        app.apply_ui_event(UiEvent::ToolCall(mizar_llm::ToolCall {
            id: "tc-1".into(),
            name: "read".into(),
            arguments: r#"{"path":"a.rs"}"#.into(),
        }));
        assert_eq!(super::inline_finalized_commit_limit(&app), 2);

        // 工具完成后仍在最后一项（后续相邻工具可能影响 group_end），等下一个条目到达即提交
        app.apply_ui_event(UiEvent::ToolResult {
            call_id: "tc-1".into(),
            name: "read".into(),
            content: "ok".into(),
            is_error: false,
            blocked: false,
            details: None,
        });
        assert_eq!(super::inline_finalized_commit_limit(&app), 2);

        app.apply_ui_event(UiEvent::TextDelta {
            text: "final answer".into(),
            ts_ms: 20,
        });
        assert_eq!(super::inline_finalized_commit_limit(&app), 3);

        // TurnEnd 空闲后全量收尾提交
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: super::UiTurnEndReason::Completed,
        });
        assert_eq!(super::inline_finalized_commit_limit(&app), 4);
    }
}
