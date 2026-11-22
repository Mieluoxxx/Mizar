//! 异步运行循环：将 Cordis / AgentLoop / Trajectory 事件接入终端界面，并按 `disabledPlugins` 条件挂载客户端与 TUI 扩展。

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, Inject, LogLevel, LogSink, Plugin,
};
use crossterm::event::EventStream;
use mizar_agent_loop::AgentLoopContextExt;
use mizar_agent_spine::{SpinePluginsContextExt, SubagentContextExt};
use mizar_client::approval::{ApprovalSnapshot, register_approval_hook_with};
use mizar_client::mentions::{FileMentionsConfig, FileMentionsPlugin};
use mizar_client::workspace::collect_user_skill_names;
use mizar_client_core::{HostExtensionFlags, PLUGIN_TUI_EMPHASIS};
use mizar_llm::{LlmContextExt, ReasoningEffort};
use mizar_session::{Session, SessionContextExt, SessionListQuery, SessionPersistenceContextExt};
use mizar_skills::EVENT_SKILLS_CHANGE;
use mizar_system_prompt::{PromptSection, SystemPromptContextExt};
use mizar_tools::EVENT_TOOLS_CHANGE;
use mizar_tui_ui::TuiFeatures;
use mizar_tui_ui::app::{
    AppAction, AppState, ApprovalMode, NoticeLevel, SessionSelection, UiEvent, UiTurnEndReason,
};
use mizar_tui_ui::emphasis::{EmphasisConfig, reply_guidance_prompt};
use mizar_tui_ui::overlay::{ApprovalDecision, SessionPickerItem, SettingEntry};
use mizar_tui_ui::terminal::{ProgramStatusState, ScreenMode, TerminalGuard, emit_program_status};
use mizar_tui_ui::theme::ThemePreset;
use mizar_tui_ui::ui::render;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

mod actions;
mod terminal_io;

pub use actions::{apply_clipboard_content, resume_session_into_app};
pub use mizar_client::events::{
    ActiveTask, ActiveTaskKind, LoopClientMsg, RunController, WaitServicesError, steer_session,
    subscribe_session_events, subscribe_session_events_into, to_ui_log_record, to_ui_status,
    to_ui_step_outcome, to_ui_turn_end_reason, wait_for_services,
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

use actions::{handle_action, populate_side_panel_data};
use terminal_io::{
    create_terminal, flush_inline_scrollback, handle_term_event_dirty, poll_term_event,
    process_loop_ui_msg_dirty,
};

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
            restrict_to_workspace: false,
            check_for_updates: false,
            disabled_plugins: Vec::new(),
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

/// 官方 TUI 回复视觉语法提示段 Cordis 插件（依赖 `systemPrompt`，卸载时自动清理段落）。
#[derive(Default, Clone, Copy)]
pub struct TuiEmphasisPlugin;

impl Plugin for TuiEmphasisPlugin {
    type Config = EmphasisConfig;

    fn name(&self) -> Option<&str> {
        Some(PLUGIN_TUI_EMPHASIS)
    }

    fn inject(&self) -> Inject {
        Inject::from(["systemPrompt"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let _ = ctx.system_prompt()?;
            let c = ctx.clone();
            ctx.effect("mizar-tui-emphasis", move || {
                mount_reply_guidance_section(&c, &config)
            })?;
            Ok(None)
        })
    }
}

/// 构造 `/session` 命令的只读会话身份与账目报告（`TuiConfig` + `AppState` 薄包装）。
pub fn build_session_info_report(
    ctx: &Context,
    config: &TuiConfig,
    session: &Session,
    app: &AppState,
) -> String {
    mizar_client::session_view::build_session_info_report(
        ctx,
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

/// 基于 `AppState` 与 Cordis 运行时服务状态构建 `/settings` 条目列表（薄包装）。
pub fn build_settings_entries(ctx: &Context, app: &AppState) -> Vec<SettingEntry> {
    mizar_client::settings::build_settings_entries(ctx, &settings_snapshot_from_app(app))
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
        let Some(finished_task) = task_state.finish_if_current(
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

async fn poll_inbox_wake(
    rx: Option<&mut tokio::sync::broadcast::Receiver<String>>,
) -> Result<String, tokio::sync::broadcast::error::RecvError> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
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
    let ext_flags = HostExtensionFlags::from_disabled_plugins(&config.disabled_plugins);
    if !ext_flags.tui {
        return Err("TUI 已被配置 disabledPlugins 中的 \"tui\" 禁用".into());
    }

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
                    exclude_subagents: true,
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

    let (loop_tx, mut loop_rx) = mpsc::unbounded_channel::<LoopClientMsg>();
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
    let active_sid = Arc::new(RwLock::new(active_session.id.clone()));
    let mut handles = {
        let bus_tx = loop_tx.clone();
        subscribe_session_events_into(&ctx, Arc::clone(&active_sid), move |ev| {
            let _ = bus_tx.send(LoopClientMsg::Bus(ev));
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

    let mut plugin_fibers = Vec::new();
    if ext_flags.tui_emphasis && ctx.system_prompt().is_ok() {
        plugin_fibers.push(ctx.plugin(TuiEmphasisPlugin, app.emphasis).await?);
    }

    // 共享审批快照（供 tools/pre-execute 异步瀑布流无锁读取当前审批策略）
    let approval_snap = Arc::new(RwLock::new(ApprovalSnapshot {
        mode: app.approval_mode,
        always_allowed: app.always_allowed_tools.clone(),
    }));
    if ext_flags.approval_gate {
        let approval_tx = loop_tx.clone();
        handles.push(register_approval_hook_with(
            &ctx,
            Arc::clone(&active_sid),
            Arc::clone(&approval_snap),
            move |ev| {
                let _ = approval_tx.send(LoopClientMsg::Bus(ev));
            },
        )?);
    }
    if ext_flags.file_mentions {
        let mention_tx = loop_tx.clone();
        plugin_fibers.push(
            ctx.plugin(
                FileMentionsPlugin,
                FileMentionsConfig::with_callback(
                    config.workspace.clone(),
                    config.restrict_to_workspace,
                    move |ev| {
                        let _ = mention_tx.send(LoopClientMsg::Bus(ev));
                    },
                ),
            )
            .await?,
        );
    }

    if app.features.side_panel && app.side_panel.open {
        populate_side_panel_data(&ctx, &mut app).await;
    }

    let mut task_state = RunController::default();
    let loop_res: Result<(), Box<dyn std::error::Error>> = async {
        // 进入终端 raw mode（Fullscreen / Inline 双模支持），并注入 OSC 10/11/4 + DA1 探测到的终端自适应色板
        let _terminal_guard =
            TerminalGuard::enter_with_mode(app.screen_mode, app.mouse_capture_enabled)?;
        app.apply_terminal_colors(_terminal_guard.probed_colors.clone());

        let mut terminal = create_terminal(app.screen_mode)?;

        let mut term_stream = Some(EventStream::new());
        let mut wake_rx = ctx.agent_loop().ok().map(|a| a.inbox().subscribe_wake());
        let mut inbox_wake_ready = false;
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
                    maybe_wake = poll_inbox_wake(wake_rx.as_mut()), if !app.is_running_activity() && task_state.active_task().is_none() => {
                        match maybe_wake {
                            Ok(sid) => {
                                if sid == active_session.id {
                                    inbox_wake_ready = true;
                                }
                                (AppAction::None, false)
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                inbox_wake_ready = true;
                                (AppAction::None, false)
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                wake_rx = None;
                                (AppAction::None, false)
                            }
                        }
                    }
                };

                needs_redraw |= step_dirty || action != AppAction::None;

                let was_cancel_run = matches!(action, AppAction::CancelRun);
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
                if was_cancel_run {
                    if let Some(ref mut rx) = wake_rx {
                        while let Ok(_)
                        | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) =
                            rx.try_recv()
                        {}
                    }
                    inbox_wake_ready = false;
                }

                let can_drain_inbox_followup = !app.is_running_activity()
                    && task_state.active_task().is_none()
                    && (inbox_wake_ready
                        || matches!(
                            app.last_turn_end_reason,
                            None | Some(UiTurnEndReason::Completed)
                        ));
                inbox_wake_ready = false;
                if can_drain_inbox_followup
                    && actions::drain_next_inbox_followup(
                        &ctx,
                        &config,
                        &mut app,
                        &active_session,
                        &loop_tx,
                        &mut task_state,
                    )
                {
                    needs_redraw = true;
                }

                if let Ok(mut guard) = approval_snap.write() {
                    if guard.mode != app.approval_mode {
                        guard.mode = app.approval_mode;
                    }
                    if guard.always_allowed != app.always_allowed_tools {
                        guard.always_allowed = app.always_allowed_tools.clone();
                    }
                }
            }
            Ok(())
        }
        .await;

        if let Some(task) = update_check_task {
            task.abort();
        }
        run_loop_res
    }
    .await;

    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    emit_program_status(ProgramStatusState::Idle, None, None);
    task_state.abort_for_session_switch();
    if let Ok(sub) = ctx.subagent() {
        sub.cancel_parent_subagents(&app.session_id);
    }
    for fiber in plugin_fibers.into_iter().rev() {
        fiber.dispose().await;
    }
    for handle in handles.into_iter().rev() {
        handle.dispose().await;
    }
    loop_res?;

    Ok(TuiExitSummary {
        session_id: app.session_id.clone(),
        persisted: ctx.session_persistence().is_some(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use mizar_tui_ui::app::{AppAction, AppState, ChatItem, UiEvent};

    use super::{
        ActiveTaskKind, LoopClientMsg, RunController, TuiEmphasisPlugin, process_loop_ui_msg,
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
        assert_eq!(act2, AppAction::SubmitPrompt("queued follow-up".into()));
        assert!(app.queue.is_empty());
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

        ctx.fiber().dispose().await;
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
        super::actions::populate_side_panel_data(&ctx, &mut app).await;
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
        super::actions::apply_module_toggle_setting(
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
            super::actions::apply_module_toggle_setting(
                &ctx,
                &mut app,
                Some(&settings_path),
                id,
                false,
            )
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
        super::actions::apply_module_toggle_setting(
            &ctx,
            &mut app,
            Some(&settings_path),
            "skills",
            true,
        )
        .await;
        assert!(ctx.get::<mizar_skills::SkillRegistry>("skills").is_none());
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

        ctx.fiber().dispose().await;
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

        // 正常 Completed 时应继续自动出队下一条。
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
        assert_eq!(action, AppAction::SubmitPrompt("auto dequeue me".into()));
        assert!(app.queue.is_empty());
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
    async fn build_session_info_report_thin_wrappers_match_app_and_config() {
        use cordis::Service;
        use mizar_session::{
            CompactionDigest, SessionContextExt, SessionCreateOptions, SessionEvent, SessionStore,
            SessionTraceEvent,
        };

        let ctx = cordis::Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        let store = ctx.sessions().unwrap();

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

        let report_live = super::build_session_info_report(&ctx, &tui_cfg, &child, &app);
        assert!(
            report_live.contains(&format!("ID       {child_id}")),
            "{report_live}"
        );
        assert!(report_live.contains("仅内存（未持久化）"), "{report_live}");

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
        use cordis::Service;
        use mizar_system_prompt::{SystemPrompt, SystemPromptConfig, SystemPromptContextExt};
        use mizar_tui_ui::emphasis::EmphasisConfig;

        let ctx = cordis::Context::new();
        ctx.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
            .await
            .unwrap();
        let sp = ctx.system_prompt().unwrap();

        assert!(!sp.build().contains("These rules shape replies"));

        // 1. enabled=false 时不注入
        let disabled_cfg = EmphasisConfig {
            enabled: false,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let disabled_fiber = ctx.plugin(TuiEmphasisPlugin, disabled_cfg).await.unwrap();
        assert!(!sp.build().contains("These rules shape replies"));
        disabled_fiber.dispose().await;

        // 2. enabled=true && guidance=true 时挂载注入，卸载后干净移除
        let enabled_cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let plugin_fiber = ctx.plugin(TuiEmphasisPlugin, enabled_cfg).await.unwrap();
        assert!(sp.build().contains("These rules shape replies"));

        plugin_fiber.dispose().await;
        assert!(!sp.build().contains("These rules shape replies"));

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn run_tui_with_summary_refuses_to_start_when_tui_plugin_disabled() {
        use cordis::Service;
        use mizar_system_prompt::{SystemPrompt, SystemPromptConfig, SystemPromptContextExt};

        let raw_before = crossterm::terminal::is_raw_mode_enabled().ok();
        let ctx = cordis::Context::new();
        ctx.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
            .await
            .unwrap();

        let mut cfg = super::TuiConfig::new(std::env::temp_dir());
        cfg.disabled_plugins = vec![mizar_client_core::PLUGIN_TUI.to_string()];

        let err = super::run_tui_with_summary(ctx.clone(), cfg)
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
            !ctx.system_prompt()
                .unwrap()
                .build()
                .contains("These rules shape replies")
        );

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn runtime_feature_gates_block_disabled_clipboard_local_bash_and_side_panel() {
        use cordis::Service;
        use mizar_client_core::{
            HostExtensionFlags, PLUGIN_LOCAL_BASH, PLUGIN_TUI_CLIPBOARD, PLUGIN_TUI_SIDE_PANEL,
        };
        use mizar_session::{SessionContextExt, SessionStore};
        use mizar_tools::ToolRuntime;
        use mizar_tui_ui::TuiFeatures;

        let ctx = cordis::Context::new();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        ctx.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        let session = ctx.sessions().unwrap().create(None);

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
        super::actions::populate_side_panel_data(&ctx, &mut app).await;
        assert!(app.side_panel_data.tools.is_empty());
        assert!(app.side_panel_data.git_changes.is_empty());

        ctx.fiber().dispose().await;
    }

    #[tokio::test]
    async fn handle_steer_prompt_steers_without_cancel_while_cancel_run_hard_aborts_and_wake_drains_followup()
     {
        use std::sync::Arc;

        use cordis::{Context, PluginDef, Service};
        use futures::channel::mpsc as futures_mpsc;
        use mizar_agent_loop::{
            AgentLoop, AgentLoopConfig, AgentLoopContextExt, DeliveryMode, TurnInput,
        };
        use mizar_llm::{
            BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError,
            LlmRuntime, Message, ModelInfo, ProviderInfo, ReasoningEffort, StreamChunk, ToolCall,
        };
        use mizar_session::{SessionContextExt, SessionStore};
        use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
        use mizar_tools::{ToolDefinition, ToolOutput, ToolRuntime, ToolsContextExt};
        use tokio::sync::{Notify, mpsc};

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

        let ctx = Context::new();
        ctx.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        ctx.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        ctx.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
            .await
            .unwrap();
        ctx.plugin(AgentLoop::plugin(), AgentLoopConfig::default())
            .await
            .unwrap();

        let recorded = Arc::new(std::sync::Mutex::new(Vec::new()));
        let adapter = Arc::new(SteerTuiAdapter {
            recorded: Arc::clone(&recorded),
        });
        ctx.plugin(
            PluginDef::sync("mock-tui-prov", move |c, ()| {
                let llm = c.llm()?;
                let a = Arc::clone(&adapter);
                c.effect("mock-tui-prov.reg", move || {
                    Ok::<_, LlmError>(Some(llm.register_adapter("mock-tui", a)?))
                })?;
                Ok::<_, cordis::CordisError>(())
            })
            .inject(["llm"]),
            (),
        )
        .await
        .unwrap();

        let started_round = Arc::new(Notify::new());
        let release_round = Arc::new(Notify::new());
        let started_sig = Arc::clone(&started_round);
        let release_sig = Arc::clone(&release_round);
        let _tool_reg = ctx
            .tools()
            .unwrap()
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
                            _ = r.notified() => Ok::<_, cordis::BoxError>(ToolOutput::text("gate_done")),
                            _ = call_ctx.cancel.cancelled() => Err("cancelled".into()),
                        }
                    }
                },
            ))
            .unwrap();

        let sess = ctx.sessions().unwrap().create(None);
        let cfg = super::TuiConfig::new(std::env::temp_dir());
        let mut app = AppState::new(
            std::env::temp_dir(),
            &sess.id,
            Vec::new(),
            Some("mock-tui".into()),
            Some("m-tui".into()),
            None,
        );
        let (loop_tx, mut loop_rx) = mpsc::unbounded_channel::<super::LoopClientMsg>();
        let mut task_state = super::RunController::default();

        // 1. 启动第一轮（进入阻塞工具），然后发温和 SteerPrompt：不得取消活跃 CancellationToken
        super::actions::spawn_agent_run(
            &ctx,
            &cfg,
            &mut app,
            &sess,
            &loop_tx,
            &mut task_state,
            "first turn".into(),
            Vec::new(),
        );
        started_round.notified().await;

        super::actions::handle_steer_prompt(
            &ctx,
            &cfg,
            &mut app,
            &sess,
            &loop_tx,
            &mut task_state,
            "gentle steer".into(),
            Vec::new(),
        )
        .await;

        let active_task = task_state.active_task().unwrap();
        assert!(!active_task.cancel.is_cancelled());
        assert!(!active_task.user_cancelled);
        assert!(app.active_flash().is_some_and(|s| s.contains("步间插话")));

        release_round.notify_one();
        let finish_msg = loop {
            match loop_rx.recv().await.unwrap() {
                super::LoopClientMsg::Task {
                    event: super::UiEvent::RunFinished(res),
                    ..
                } => break res,
                _ => continue,
            }
        };
        assert_eq!(finish_msg.unwrap(), "steered reply");
        task_state.active = None;
        app.set_busy_state(false);

        // 2. 启动第二轮并预塞 FollowUp，然后执行硬取消 cancel_active_run：立即取消且清空收件箱
        let agent_loop = ctx.agent_loop().unwrap();
        super::actions::spawn_agent_run(
            &ctx,
            &cfg,
            &mut app,
            &sess,
            &loop_tx,
            &mut task_state,
            "abort turn".into(),
            Vec::new(),
        );
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

        super::actions::cancel_active_run(&ctx, &mut app, &sess, &mut task_state);
        assert!(task_state.active_task().unwrap().cancel.is_cancelled());
        assert!(task_state.active_task().unwrap().user_cancelled);
        assert!(agent_loop.inbox().pop_followup(&sess.id).is_none());
        let kept = agent_loop.inbox().snapshot(&sess.id);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].mode, DeliveryMode::Inject);
        assert!(kept[0].input.text().contains("bg result kept across esc"));
        agent_loop.inbox().clear(&sess.id);

        let abort_res = loop {
            match loop_rx.recv().await.unwrap() {
                super::LoopClientMsg::Task {
                    event: super::UiEvent::RunFinished(res),
                    ..
                } => break res,
                _ => continue,
            }
        };
        assert!(abort_res.is_err());
        task_state.active = None;
        app.set_busy_state(false);

        // 3. 空闲时推送 FollowUp 应唤醒 poll_inbox_wake 并由 drain_next_inbox_followup 启动下一轮
        let mut wake_rx = agent_loop.inbox().subscribe_wake();
        let wake_push = agent_loop.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "wake check"),
        );
        assert_eq!(wake_push.effective_mode, DeliveryMode::FollowUp);
        assert!(!wake_push.degraded);
        let woken_sid = super::poll_inbox_wake(Some(&mut wake_rx)).await.unwrap();
        assert_eq!(woken_sid, sess.id);

        assert!(super::actions::drain_next_inbox_followup(
            &ctx,
            &cfg,
            &mut app,
            &sess,
            &loop_tx,
            &mut task_state,
        ));
        let followup_res = loop {
            match loop_rx.recv().await.unwrap() {
                super::LoopClientMsg::Task {
                    event: super::UiEvent::RunFinished(res),
                    ..
                } => break res,
                _ => continue,
            }
        };
        assert_eq!(followup_res.unwrap(), "woke followup ok");

        ctx.fiber().dispose().await;
    }
}
