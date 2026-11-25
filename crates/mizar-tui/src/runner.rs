//! 异步运行循环：将 Cordis / AgentLoop / Trajectory 事件接入终端界面，并按 `disabledPlugins` 条件挂载客户端与 TUI 扩展。

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, Inject, LogLevel, LogSink, Plugin,
};
use crossterm::event::EventStream;
use mizar_agent_spine::SpinePluginsContextExt;
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
    ActiveTask, ActiveTaskKind, LoopClientMsg, PendingPrompt, RunController, WaitServicesError,
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

fn pending_prompt_into_action(prompt: PendingPrompt) -> AppAction {
    if prompt.images.is_empty() {
        AppAction::SubmitPrompt(prompt.text)
    } else {
        AppAction::SubmitPromptWithImages {
            text: prompt.text,
            images: prompt.images,
        }
    }
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
            return pending_prompt_into_action(steer);
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
