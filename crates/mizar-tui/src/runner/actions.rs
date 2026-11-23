//! 工作台动作执行器：派发代理回合、本地 Bash、会话切换/恢复、剪贴板粘贴、侧栏数据采集与设置热更新。

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};

use cordis::Context;
use crossterm::event::EventStream;
use mizar_agent_loop::{AgentLoopContextExt, DeliveryMode, RunOptions, TurnInput};
use mizar_agent_spine::{
    CompactionContextExt, CompactionError, ManualCompactOptions, SubagentContextExt,
    resolve_global_settings_path,
};
use mizar_client::events::{ActiveTaskKind, LoopClientMsg, RunController, steer_session};
use mizar_client::images;
use mizar_client::local_bash::{execute_local_bash_for_session, next_local_bash_call_id};
use mizar_client::mentions;
use mizar_client::session_view::{
    SelectionSources, collect_session_picker_items, project_fork_view, project_resume_view,
    project_session_tree, resolve_model_switch, resolve_selection, to_ui_selection,
};
use mizar_client::settings::{
    apply_runtime_setting_change, persist_setting_change, toggle_spine_module,
};
use mizar_client::workspace::{
    collect_mcps_summary, collect_tools_summary, collect_user_skill_names,
    collect_workspace_panel_snapshot,
};
use mizar_client_core::{HostExtensionFlags, short_id};
use mizar_instructions::AgentInstructions;
use mizar_llm::{ImageAttachment, LlmContextExt};
use mizar_mcp::{McpConnectionState, McpManager};
use mizar_session::{Session, SessionContextExt, SessionCreateOptions};
use mizar_session_title::SessionTitleContextExt;
use mizar_skills::{SkillLookupOptions, SkillRegistry};
use mizar_tools::{ToolExposure, ToolRuntime};
use mizar_trajectory::{TimelineMode, TrajectoryRuntime};
use mizar_tui_ui::app::{AppAction, AppState, NoticeLevel, SessionSelection, UiEvent};
use mizar_tui_ui::overlay::{ApprovalDecision, SkillPickerItem};
use mizar_tui_ui::terminal::{
    ScreenMode, apply_terminal_modes, resume_after_external_editor, set_mouse_capture_enabled,
    suspend_for_external_editor,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::TuiConfig;
use super::terminal_io::{BufferedTerminal, create_terminal};
use crate::clipboard;

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

/// 将读取到的系统/终端剪贴板内容粘贴入输入框（受 `app.features.clipboard` 开关门控）。
pub async fn apply_clipboard_content(
    app: &mut AppState,
    workspace: &Path,
    result: Result<clipboard::ClipboardContent, clipboard::ClipboardError>,
) {
    if !app.features.clipboard {
        app.set_flash("剪贴板扩展已禁用（disabledPlugins 包含 tui-clipboard）");
        return;
    }
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
pub(super) fn spawn_agent_run(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    prompt: String,
    images: Vec<ImageAttachment>,
) {
    spawn_agent_run_input(
        ctx,
        config,
        app,
        active_session,
        loop_tx,
        task_state,
        TurnInput::user_with_images(prompt, images),
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_agent_run_input(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    input: TurnInput,
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
        images: Vec::new(),
        ..Default::default()
    };
    let tx = loop_tx.clone();
    tokio::spawn(async move {
        let res = agent_loop
            .run_with_options(session, input, cancel, run_opts)
            .await
            .map_err(|e| e.to_string());
        let _ = tx.send(LoopClientMsg::Task {
            run_id,
            session_id: task_session_id,
            event: UiEvent::RunFinished(res),
        });
    });
}

/// 从 `SessionInbox` 出队当前活跃会话的下一条待唤醒 `FollowUp` 并启动一轮对话。
pub(super) fn drain_next_inbox_followup(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
) -> bool {
    let Ok(agent_loop) = ctx.agent_loop() else {
        return false;
    };
    if agent_loop.is_session_busy(&active_session.id) {
        return false;
    }
    let Some(next_input) = agent_loop.inbox().pop_followup(&active_session.id) else {
        return false;
    };
    match &next_input {
        TurnInput::User { text, images } => {
            if !images.is_empty() && !app.active_model_supports_images() {
                app.reject_unsupported_images_notice();
                return true;
            }
            app.push_user_message(text);
        }
        TurnInput::Notice { source, text, .. } => {
            if !(source.starts_with("subagent:") || source.starts_with("subagent-watchdog:")) {
                app.push_notice(NoticeLevel::Info, format!("[{source}] {text}"));
            }
        }
    }
    spawn_agent_run_input(
        ctx,
        config,
        app,
        active_session,
        loop_tx,
        task_state,
        next_input,
    );
    true
}

/// 执行硬取消：关闭审批浮层、触发 `CancellationToken` 并清理用户输入项（已落箱后台回执转 `Inject` 保留）。
pub(super) fn cancel_active_run(
    ctx: &Context,
    app: &mut AppState,
    active_session: &Session,
    task_state: &mut RunController,
) {
    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    task_state.cancel_active();
    if let Ok(agent_loop) = ctx.agent_loop() {
        agent_loop
            .inbox()
            .clear_user_items_and_defer_notices(&active_session.id);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_steer_prompt(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    prompt: String,
    images: Vec<ImageAttachment>,
) {
    if images.is_empty() {
        let file_mentions_enabled =
            HostExtensionFlags::from_disabled_plugins(&config.disabled_plugins).file_mentions;
        if file_mentions_enabled
            && prompt.contains('@')
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
    } else if !app.active_model_supports_images() {
        app.reject_unsupported_images_notice();
        return;
    }

    let is_active_agent_run = task_state.active_task().is_some_and(|task| {
        task.kind == ActiveTaskKind::AgentRun && task.session_id == active_session.id
    });

    if is_active_agent_run && let Ok(agent_loop) = ctx.agent_loop() {
        app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
        let push_res = steer_session(
            &agent_loop,
            &active_session.id,
            TurnInput::user_with_images(prompt.clone(), images.clone()),
        );
        if push_res.effective_mode == DeliveryMode::Steer {
            app.push_user_message(&prompt);
            app.set_flash("⚡ 已投递步间插话，将在当前步完成后继续…");
            return;
        }
        if push_res.effective_mode == DeliveryMode::FollowUp {
            app.set_flash("⚡ 已排入下一轮续接…");
            return;
        }
    }

    app.push_user_message(&prompt);
    spawn_agent_run(
        ctx,
        config,
        app,
        active_session,
        loop_tx,
        task_state,
        prompt,
        images,
    );
}

pub(super) fn spawn_local_bash_run(
    app: &mut AppState,
    active_session: &Arc<Session>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    command: String,
    exclude_from_context: bool,
) {
    if !app.features.local_bash {
        app.set_flash("本地 Bash 扩展已禁用（disabledPlugins 包含 local-bash）");
        return;
    }
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

    let call_id = next_local_bash_call_id();
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
        let ev = execute_local_bash_for_session(
            &session,
            call_id,
            command,
            &cwd,
            exclude_from_context,
            cancel,
            move |preview| {
                let _ = update_tx.send(LoopClientMsg::Task {
                    run_id,
                    session_id: update_session_id.clone(),
                    event: UiEvent::ToolUpdate {
                        call_id: update_cid.clone(),
                        name: "bash".to_string(),
                        preview: preview.to_string(),
                    },
                });
            },
        )
        .await;

        let _ = tx.send(LoopClientMsg::Task {
            run_id,
            session_id: task_session_id,
            event: ev,
        });
    });
}

pub(super) fn populate_tools_and_mcps(ctx: &Context, app: &mut AppState) {
    if !app.features.side_panel {
        return;
    }
    app.side_panel_data.tools = collect_tools_summary(ctx, Some(&app.session_id));
    app.side_panel_data.mcps = collect_mcps_summary(ctx);
}

pub(super) async fn populate_side_panel_data(ctx: &Context, app: &mut AppState) {
    if !app.features.side_panel {
        return;
    }
    app.refresh_context_breakdown();
    app.side_panel_data.files = app.touched_files.clone();
    let snapshot =
        collect_workspace_panel_snapshot(ctx, &app.workspace, Some(&app.session_id)).await;
    snapshot.apply_to_side_panel(&mut app.side_panel_data, &app.anomalies);
}

pub(super) async fn apply_module_toggle_setting(
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
            .refresh_settings(super::build_settings_entries(ctx, app));
        return;
    }

    let outcome = toggle_spine_module(ctx, settings_path, plugin_id, enabled).await;
    if outcome.ok {
        app.user_skill_names = collect_user_skill_names(ctx, &app.workspace).await;
        if app.features.side_panel {
            populate_side_panel_data(ctx, app).await;
        }
    }
    app.overlay
        .refresh_settings(super::build_settings_entries(ctx, app));
    if let Some(flash) = outcome.flash {
        app.set_flash(flash);
    }
    for (level, msg) in outcome.notices {
        app.push_notice(level, msg);
    }
}

pub(super) async fn apply_host_setting_change(
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
            if app.features.side_panel && app.side_panel.open {
                populate_side_panel_data(ctx, app).await;
            }
        }
        k if k.starts_with("compaction.")
            || k.starts_with("sessionTitle.")
            || k == "cacheWarming"
            || k == "showCacheMissNotices" =>
        {
            apply_runtime_setting_change(ctx, key, value);
        }
        _ => {}
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
    if let Ok(agent_loop) = ctx.agent_loop() {
        agent_loop.inbox().clear(&next.id);
    }
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
    if app.features.side_panel && app.side_panel.open {
        populate_side_panel_data(ctx, app).await;
    }
    app.set_flash(format!("✓ 已恢复会话 ({})", short_id(target_id)));
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_action(
    ctx: &Context,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    terminal: &mut BufferedTerminal,
    term_stream: &mut Option<EventStream>,
    active_sid: &Arc<RwLock<String>>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    action: AppAction,
) {
    match action {
        AppAction::None => {}
        AppAction::Quit => {
            app.should_quit = true;
        }
        AppAction::PasteClipboard => {
            if !app.features.clipboard {
                app.set_flash("剪贴板扩展已禁用（disabledPlugins 包含 tui-clipboard）");
                return;
            }
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
            cancel_active_run(ctx, app, active_session, task_state);
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
                        let _ = tx.send(LoopClientMsg::Task {
                            run_id,
                            session_id: task_session_id,
                            event: UiEvent::CompactionFinished {
                                outcome: "failed".to_string(),
                                tokens_before: None,
                                tokens_after: None,
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
            if let Ok(sub) = ctx.subagent() {
                sub.cancel_parent_subagents(&active_session.id);
            }
            if let Ok(agent_loop) = ctx.agent_loop() {
                agent_loop.inbox().clear(&active_session.id);
            }
            if app.is_running_activity() {
                app.set_idle_state();
            }
            if let Ok(sessions) = ctx.sessions() {
                let next = sessions.create(None);
                let title = next.header().title;
                *active_sid.write().unwrap() = next.id.clone();
                app.reset_for_new_session(next.id.clone(), title);
                *active_session = next;
                if app.features.side_panel && app.side_panel.open {
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
            if let Ok(sub) = ctx.subagent() {
                sub.cancel_parent_subagents(&active_session.id);
            }
            if let Ok(agent_loop) = ctx.agent_loop() {
                agent_loop.inbox().clear(&active_session.id);
            }
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
                            let _ = tx.send(LoopClientMsg::Bus(UiEvent::Flash(format!(
                                "✓ 标题已生成：{new_title}"
                            ))));
                            let _ = tx.send(LoopClientMsg::Bus(UiEvent::Notice {
                                level: NoticeLevel::Info,
                                text: format!("会话标题已重新生成为：{new_title}"),
                            }));
                        }
                        Err(err) => {
                            let _ = tx.send(LoopClientMsg::Bus(UiEvent::Notice {
                                level: NoticeLevel::Warn,
                                text: format!("无法重新生成会话标题：{err}"),
                            }));
                        }
                    }
                });
            }
        }
        AppAction::SubmitPrompt(prompt) => {
            let file_mentions_enabled =
                HostExtensionFlags::from_disabled_plugins(&config.disabled_plugins).file_mentions;
            if file_mentions_enabled
                && prompt.contains('@')
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
            handle_steer_prompt(
                ctx,
                config,
                app,
                active_session,
                loop_tx,
                task_state,
                prompt,
                Vec::new(),
            )
            .await;
        }
        AppAction::SteerPromptWithImages { text, images } => {
            handle_steer_prompt(
                ctx,
                config,
                app,
                active_session,
                loop_tx,
                task_state,
                text,
                images,
            )
            .await;
        }
        AppAction::RunLocalBash {
            command,
            exclude_from_context,
        } => {
            if !app.features.local_bash {
                app.set_flash("本地 Bash 扩展已禁用（disabledPlugins 包含 local-bash）");
                return;
            }
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
            let report = super::build_session_info_report(ctx, config, active_session, app);
            app.push_notice(NoticeLevel::Info, report);
        }
        AppAction::ShowTools => {
            if let Some(rt) = ctx.get::<ToolRuntime>("tools") {
                let declared: HashSet<String> = rt
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
            if app.features.side_panel && app.side_panel.open {
                populate_side_panel_data(ctx, app).await;
            }
        }
        AppAction::RefreshSidePanelTools => {
            if app.features.side_panel && app.side_panel.open {
                populate_tools_and_mcps(ctx, app);
            }
        }
        AppAction::OpenTrajectoryScene(mode) => {
            if !app.features.trajectory_view {
                return;
            }
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
            let items = super::build_settings_entries(ctx, app);
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
            if let Ok(sub) = ctx.subagent() {
                sub.cancel_parent_subagents(&active_session.id);
            }
            if let Ok(agent_loop) = ctx.agent_loop() {
                agent_loop.inbox().clear(&active_session.id);
            }
            if app.is_running_activity() {
                app.set_idle_state();
            }
            if let Ok(sessions) = ctx.sessions() {
                match sessions.fork(&app.session_id, target_seq, SessionCreateOptions::default()) {
                    Ok(next) => {
                        *active_sid.write().unwrap() = next.id.clone();
                        app.restore_session_view(project_fork_view(&next, target_seq));
                        *active_session = next;
                        if app.features.side_panel && app.side_panel.open {
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
