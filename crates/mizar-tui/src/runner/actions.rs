//! 工作台动作执行器：通过 `SessionActorHandle` 派发代理回合/审批/队列/会话切换，并承载本地 Bash、剪贴板粘贴、侧栏数据采集与设置热更新。

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use chrono::{DateTime, Local};
use crossterm::event::EventStream;
use mizar_client::actor::{ActorError, SessionActorHandle};
use mizar_client::events::{ActiveTaskKind, LoopClientMsg, RunController};
use mizar_client::images;
use mizar_client::local_bash::{execute_local_bash_for_session, next_local_bash_call_id};
use mizar_client::mentions;
use mizar_client::run_dir::check_session_writable;
use mizar_client::session_view::{
    SelectionSources, project_fork_view, project_resume_view, resolve_model_switch,
    resolve_selection, to_ui_selection,
};
use mizar_client::settings::{
    apply_runtime_setting_change, persist_setting_change, toggle_spine_module,
};
use mizar_client::workspace::{
    WorkspacePanelSnapshot, collect_mcps_summary, collect_tools_summary, collect_user_skill_names,
    collect_workspace_panel_snapshot,
};
use mizar_core::skills::SkillLookupOptions;
use mizar_core::spine::{
    CompactionError, ManualCompactOptions, MizarRuntime, resolve_global_settings_path,
};
use mizar_core::trajectory::TimelineMode;
use mizar_llm::{ImageAttachment, ReasoningEffort};
use mizar_mcp::McpConnectionState;
use mizar_session::{Session, SessionEvent};
use mizar_tools::ToolExposure;
use mizar_tui_ui::app::{
    AppAction, AppState, ApprovalMode, ChatItem, NoticeLevel, QueuedPrompt, SessionSelection,
    UiEvent,
};
use mizar_tui_ui::overlay::{ApprovalDecision, SessionPickerItem, SkillPickerItem};
use mizar_tui_ui::terminal::{
    ScreenMode, apply_terminal_modes, resume_after_external_editor, set_mouse_capture_enabled,
    suspend_for_external_editor,
};
use mizar_types::client::{ClientEvent, HostExtensionFlags, short_id};
use mizar_types::rpc::{
    QueuedInputDto, ResolvedBy, RpcApprovalMode, RpcAttachmentDto, SessionConfigParams,
    SessionForkParams, SessionListItemDto, SessionListParams, SubmitMode, TurnSubmitParams,
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
        }) => {
            match tokio::task::spawn_blocking(move || images::process_image_bytes(&bytes)).await {
                Ok(Ok(processed)) => {
                    let suffix = processed.marker_suffix();
                    let _ = app.insert_pasted_image(suffix.as_deref(), processed.attachment);
                }
                Ok(Err(err)) => {
                    let msg = format!("剪贴板图片处理失败（{source_label}）：{err}");
                    app.set_flash(msg.clone());
                    app.push_notice(NoticeLevel::Error, msg);
                }
                Err(err) => {
                    let msg = format!("剪贴板图片处理任务异常（{source_label}）：{err}");
                    app.set_flash(msg.clone());
                    app.push_notice(NoticeLevel::Error, msg);
                }
            }
        }
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
                    match tokio::task::spawn_blocking(move || images::process_image_bytes(&bytes))
                        .await
                    {
                        Ok(Ok(processed)) => {
                            let suffix = processed.marker_suffix();
                            if app
                                .insert_pasted_image(suffix.as_deref(), processed.attachment)
                                .is_none()
                            {
                                break;
                            }
                        }
                        Ok(Err(err)) => {
                            let msg = format!("图片 `{}` 处理失败：{err}", path.display());
                            app.set_flash(msg.clone());
                            app.push_notice(NoticeLevel::Error, msg);
                        }
                        Err(err) => {
                            let msg = format!("图片 `{}` 处理任务异常：{err}", path.display());
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SyncedActorConfig {
    session_id: String,
    provider: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
    approval_mode: ApprovalMode,
}

impl SyncedActorConfig {
    fn from_app(app: &AppState) -> Self {
        Self {
            session_id: app.session_id.clone(),
            provider: app.provider.clone(),
            model: app.model.clone(),
            reasoning_effort: app.reasoning_effort,
            approval_mode: app.approval_mode,
        }
    }
}

/// TUI 终端状态与 `SessionActor` 内核之间的增量同步器（已提交用户输入回声去重 + 运行配置同步）。
pub(super) struct TuiActorSync {
    last_synced_entry_seq: u64,
    pending_tui_user_echoes: VecDeque<String>,
    synced_config: SyncedActorConfig,
}

impl TuiActorSync {
    pub(super) fn new(session: &Session, app: &AppState) -> Self {
        Self {
            last_synced_entry_seq: max_entry_seq(session),
            pending_tui_user_echoes: VecDeque::new(),
            synced_config: SyncedActorConfig::from_app(app),
        }
    }

    pub(super) fn reset_for_session(&mut self, session: &Session, app: &AppState) {
        self.last_synced_entry_seq = max_entry_seq(session);
        self.pending_tui_user_echoes.clear();
        self.synced_config = SyncedActorConfig::from_app(app);
    }

    pub(super) fn mark_session_id(&mut self, session_id: String) {
        self.synced_config.session_id = session_id;
    }

    pub(super) fn record_user_echo(&mut self, text: String) {
        self.pending_tui_user_echoes.push_back(text);
    }

    pub(super) fn rollback_user_echo(&mut self, text: &str) {
        if let Some(pos) = self
            .pending_tui_user_echoes
            .iter()
            .rposition(|item| item == text)
        {
            self.pending_tui_user_echoes.remove(pos);
        }
    }

    pub(super) fn clear_user_echoes(&mut self) {
        self.pending_tui_user_echoes.clear();
    }

    /// 扫描当前活跃会话新增的已提交 `User` 与外部 `ContextNotice` 条目，补齐由 `SessionActor`
    /// 自动排空排队/唤醒 `FollowUp`/外部 RPC 触发的回合气泡（跳过本地已先推入的回声）。
    pub(super) fn sync_committed_session_inputs(
        &mut self,
        session: &Session,
        app: &mut AppState,
    ) -> bool {
        let since = self.last_synced_entry_seq;
        let mut max_seq = since;
        let mut committed = Vec::new();
        session.with_all_entries(|entries| {
            for entry in entries.iter().rev() {
                if entry.seq <= since {
                    break;
                }
                max_seq = max_seq.max(entry.seq);
                match &entry.event {
                    SessionEvent::User { text, .. } => {
                        committed.push((true, text.clone(), None));
                    }
                    SessionEvent::ContextNotice { source, text, .. }
                        if !is_internal_context_notice_source(source) =>
                    {
                        committed.push((false, text.clone(), Some(source.clone())));
                    }
                    _ => {}
                }
            }
        });
        if max_seq > self.last_synced_entry_seq {
            self.last_synced_entry_seq = max_seq;
        }
        if committed.is_empty() {
            return false;
        }

        let mut changed = false;
        committed.reverse();
        for (is_user, text, source) in committed {
            if is_user {
                if let Some(pos) = self.pending_tui_user_echoes.iter().position(|p| p == &text) {
                    self.pending_tui_user_echoes.remove(pos);
                } else {
                    app.push_user_message(&text);
                    changed = true;
                }
            } else if let Some(src) = source {
                app.push_notice(NoticeLevel::Info, format!("[{src}] {text}"));
                changed = true;
            }
        }
        changed
    }

    /// 当 `/model`、`Ctrl+L`、`/effort`、`Tab`、`/approval` 或 `/settings` 变更运行配置时同步到 `SessionActor`。
    pub(super) async fn sync_config_if_changed(
        &mut self,
        actor: &SessionActorHandle,
        app: &AppState,
    ) {
        let current = SyncedActorConfig::from_app(app);
        if current == self.synced_config {
            return;
        }
        let provider = (current.provider != self.synced_config.provider)
            .then(|| current.provider.clone().unwrap_or_default());
        let model = (current.model != self.synced_config.model)
            .then(|| current.model.clone().unwrap_or_default());
        let reasoning_effort = (current.reasoning_effort != self.synced_config.reasoning_effort)
            .then_some(current.reasoning_effort)
            .flatten();
        let approval_mode = (current.approval_mode != self.synced_config.approval_mode)
            .then(|| RpcApprovalMode::from(current.approval_mode));

        if provider.is_some()
            || model.is_some()
            || reasoning_effort.is_some()
            || approval_mode.is_some()
        {
            let _ = actor
                .update_config(SessionConfigParams {
                    session_id: app.session_id.clone(),
                    provider,
                    model,
                    reasoning_effort,
                    approval_mode,
                })
                .await;
        }
        self.synced_config = current;
    }
}

fn max_entry_seq(session: &Session) -> u64 {
    session.with_all_entries(|entries| entries.last().map_or(0, |e| e.seq))
}

fn is_internal_context_notice_source(source: &str) -> bool {
    source == "file-mentions"
        || source == "environment-context"
        || source == "skill-catalog"
        || source == "agent-instructions"
        || source == "subagent-budget"
        || source.starts_with("subagent:")
        || source.starts_with("subagent-watchdog:")
}

fn ui_images_to_attachments(images: &[ImageAttachment]) -> Vec<RpcAttachmentDto> {
    images
        .iter()
        .map(|img| RpcAttachmentDto {
            path: None,
            mime: Some(img.media_type.clone()),
            data: Some(img.data.clone()),
        })
        .collect()
}

fn attachments_to_ui_images(attachments: &[RpcAttachmentDto]) -> Vec<(String, ImageAttachment)> {
    attachments
        .iter()
        .enumerate()
        .filter_map(|(idx, att)| {
            let data = att.data.as_ref()?.trim();
            if data.is_empty() {
                return None;
            }
            let mime = att.mime.clone().unwrap_or_else(|| "image/png".to_string());
            Some((
                format!("[image #{}]", idx + 1),
                ImageAttachment::new(mime, data.to_string()),
            ))
        })
        .collect()
}

fn replace_local_queue_item_id(app: &mut AppState, local_id: u64, actor_id: u64) {
    if local_id == actor_id {
        return;
    }
    let prev_selected = app.queue.selected_id();
    let items: Vec<QueuedPrompt> = app
        .queue
        .items()
        .iter()
        .cloned()
        .map(|mut item| {
            if item.id == local_id {
                item.id = actor_id;
            }
            item
        })
        .collect();
    app.queue.clear();
    for item in items {
        app.queue.push_raw(item);
    }
    if let Some(sel) = prev_selected {
        app.queue
            .set_selected(Some(if sel == local_id { actor_id } else { sel }));
    }
}

pub(super) fn apply_queue_snapshot_to_app(app: &mut AppState, items: Vec<QueuedInputDto>) {
    let prev_items = app.queue.items().to_vec();
    let prev_selected = app.queue.selected_id();
    let selected_pos = prev_selected.and_then(|sel| prev_items.iter().position(|it| it.id == sel));

    app.queue.clear();
    for (idx, dto) in items.into_iter().enumerate() {
        let images = prev_items
            .iter()
            .find(|old| old.id == dto.id)
            .or_else(|| {
                prev_items
                    .get(idx)
                    .filter(|old| old.text == dto.text && old.images.len() == dto.attachments.len())
            })
            .map(|old| old.images.clone())
            .unwrap_or_else(|| attachments_to_ui_images(&dto.attachments));
        app.queue.push_raw(QueuedPrompt {
            id: dto.id,
            text: dto.text,
            images,
        });
    }
    if let Some(sel_id) = prev_selected {
        if app.queue.items().iter().any(|it| it.id == sel_id) {
            app.queue.set_selected(Some(sel_id));
        } else if let Some(pos) = selected_pos
            && let Some(it) = app.queue.items().get(pos)
        {
            app.queue.set_selected(Some(it.id));
        }
    }
}

/// 按键批处理后同步 TUI 本地队列操作（入队 / 撤回 / 丢弃 / 队首发送）到 `SessionActor` 单真源队列。
pub(super) async fn sync_queue_after_term_event(
    actor: &SessionActorHandle,
    config: &TuiConfig,
    app: &mut AppState,
    sync: &mut TuiActorSync,
    prev_queue: &[QueuedPrompt],
) -> bool {
    let curr_queue = app.queue.items().to_vec();
    if curr_queue == prev_queue {
        return false;
    }

    let mut dirty = false;
    let file_mentions_enabled =
        HostExtensionFlags::from_disabled_plugins(&config.disabled_plugins).file_mentions;

    for old in prev_queue {
        if !curr_queue.iter().any(|cur| cur.id == old.id) {
            let _ = actor.queue_remove(&app.session_id, old.id).await;
            dirty = true;
        }
    }

    for new in curr_queue {
        if prev_queue.iter().any(|old| old.id == new.id) {
            continue;
        }
        if new.images.is_empty()
            && file_mentions_enabled
            && new.text.contains('@')
            && !app.active_model_supports_images()
            && mentions::mentions_resolve_any_image(
                &config.workspace,
                config.restrict_to_workspace,
                &new.text,
            )
            .await
        {
            app.dequeue_prompt(new.id);
            if app.input.is_empty() {
                app.input.set_text_end(&new.text);
            }
            app.reject_unsupported_images_notice();
            dirty = true;
            continue;
        }

        let raw_images: Vec<ImageAttachment> =
            new.images.iter().map(|(_, img)| img.clone()).collect();
        let attachments = ui_images_to_attachments(&raw_images);
        match actor
            .submit_turn(TurnSubmitParams {
                session_id: app.session_id.clone(),
                text: new.text.clone(),
                attachments: (!attachments.is_empty()).then_some(attachments),
                mode: Some(SubmitMode::Send),
            })
            .await
        {
            Ok(res) if res.mode == SubmitMode::Send => {
                app.dequeue_prompt(new.id);
                app.push_user_message(&new.text);
                sync.record_user_echo(new.text);
                app.set_busy_state(true);
                dirty = true;
            }
            Ok(res) => {
                if let Some(actor_id) = res.queue_id {
                    replace_local_queue_item_id(app, new.id, actor_id);
                }
                dirty = true;
            }
            Err(err) => {
                app.dequeue_prompt(new.id);
                app.push_notice(NoticeLevel::Error, format!("消息入队失败：{err}"));
                dirty = true;
            }
        }
    }

    dirty
}

/// 消费 `SessionActor` 发出的同进程零序列化 `ClientEvent` 并同步会话切换、队列快照与用户输入气泡。
pub(super) fn process_actor_client_event_dirty(
    actor: &SessionActorHandle,
    active_session: &mut Arc<Session>,
    app: &mut AppState,
    task_state: &mut RunController,
    sync: &mut TuiActorSync,
    event: ClientEvent,
) -> (AppAction, bool) {
    match event {
        ClientEvent::QueueUpdated { items } => {
            apply_queue_snapshot_to_app(app, items);
            (AppAction::None, true)
        }
        ClientEvent::SessionSwitched { new_session_id, .. } => {
            if app.session_id == new_session_id {
                return (AppAction::None, false);
            }
            let next = actor.active_session();
            if next.id != new_session_id {
                return (AppAction::None, false);
            }
            app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
            task_state.abort_for_session_switch();
            if app.is_running_activity() {
                app.set_idle_state();
            }
            if next.all_entries().is_empty() {
                app.reset_for_new_session(next.id.clone(), next.header().title);
            } else {
                app.restore_session_view(project_resume_view(&next));
            }
            *active_session = next;
            sync.reset_for_session(active_session, app);
            (AppAction::RefreshSidePanelData, true)
        }
        other => {
            let mut extra_dirty = false;
            if matches!(
                other,
                ClientEvent::StepStart { .. }
                    | ClientEvent::TurnEnd { .. }
                    | ClientEvent::RunFinished(_)
            ) {
                extra_dirty |= sync.sync_committed_session_inputs(active_session, app);
            }
            let (act, ev_dirty) =
                super::terminal_io::process_client_event_dirty(app, task_state, other);
            (act, extra_dirty || ev_dirty)
        }
    }
}

pub(super) async fn submit_turn_via_actor(
    actor: &SessionActorHandle,
    app: &mut AppState,
    sync: &mut TuiActorSync,
    prompt: String,
    images: Vec<ImageAttachment>,
) {
    let attachments = ui_images_to_attachments(&images);
    sync.record_user_echo(prompt.clone());
    match actor
        .submit_turn(TurnSubmitParams {
            session_id: app.session_id.clone(),
            text: prompt.clone(),
            attachments: (!attachments.is_empty()).then_some(attachments),
            mode: Some(SubmitMode::Send),
        })
        .await
    {
        Ok(res) if res.mode == SubmitMode::Queue => {
            // 若在空闲判定与投递瞬间已被另一端抢占并转入排队，撤回早先推入的气泡，交由队列展示
            sync.rollback_user_echo(&prompt);
            rollback_trailing_user_bubble(app, &prompt);
        }
        Ok(_) => {
            app.set_busy_state(true);
        }
        Err(err) => {
            sync.rollback_user_echo(&prompt);
            rollback_trailing_user_bubble(app, &prompt);
            app.push_notice(NoticeLevel::Error, format!("发送消息失败：{err}"));
        }
    }
}

/// 执行硬取消：关闭审批浮层、中止本地任务并委托 `SessionActor` 取消当前回合。
pub(super) async fn cancel_active_run_via_actor(
    actor: &SessionActorHandle,
    app: &mut AppState,
    task_state: &mut RunController,
    sync: &mut TuiActorSync,
) {
    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    task_state.cancel_active();
    sync.clear_user_echoes();
    let _ = actor.cancel_turn_by(&app.session_id, ResolvedBy::Tui).await;
}

pub(super) async fn handle_steer_prompt_via_actor(
    actor: &SessionActorHandle,
    config: &TuiConfig,
    app: &mut AppState,
    sync: &mut TuiActorSync,
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

    if app.is_busy() {
        app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    }
    let attachments = ui_images_to_attachments(&images);
    match actor
        .submit_turn(TurnSubmitParams {
            session_id: app.session_id.clone(),
            text: prompt.clone(),
            attachments: (!attachments.is_empty()).then_some(attachments),
            mode: Some(SubmitMode::Steer),
        })
        .await
    {
        Ok(res) => {
            app.push_user_message(&prompt);
            sync.record_user_echo(prompt);
            if res.mode == SubmitMode::Steer {
                app.set_flash("⚡ 已投递步间插话，将在当前步完成后继续…");
            } else {
                app.set_busy_state(true);
            }
        }
        Err(err) => {
            app.push_notice(NoticeLevel::Error, format!("发送步间插话失败：{err}"));
        }
    }
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
        const LOCAL_BASH_DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
        let timeout_cancel = cancel.clone();
        let timeout_guard = tokio::spawn(async move {
            tokio::time::sleep(LOCAL_BASH_DEFAULT_TIMEOUT).await;
            timeout_cancel.cancel();
        });
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
        timeout_guard.abort();

        let _ = tx.send(LoopClientMsg::Task {
            run_id,
            session_id: task_session_id,
            event: ev,
        });
    });
}

pub(super) struct SidePanelRefreshController {
    tx: mpsc::UnboundedSender<(u64, String, WorkspacePanelSnapshot)>,
    generation: u64,
    in_flight: Option<tokio::task::JoinHandle<()>>,
}

impl SidePanelRefreshController {
    pub(super) fn new() -> (
        Self,
        mpsc::UnboundedReceiver<(u64, String, WorkspacePanelSnapshot)>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                tx,
                generation: 0,
                in_flight: None,
            },
            rx,
        )
    }

    pub(super) fn current_gen(&self) -> u64 {
        self.generation
    }

    pub(super) fn abort(&mut self) {
        if let Some(handle) = self.in_flight.take() {
            handle.abort();
        }
    }
}

pub(super) fn schedule_side_panel_refresh(
    runtime: &Arc<MizarRuntime>,
    app: &mut AppState,
    panel_ctrl: &mut SidePanelRefreshController,
) {
    if !app.features.side_panel || !app.side_panel.open {
        return;
    }
    app.refresh_context_breakdown();
    app.side_panel_data.files = app.touched_files.clone();
    panel_ctrl.abort();
    panel_ctrl.generation = panel_ctrl.generation.wrapping_add(1);
    let generation = panel_ctrl.generation;
    let runtime_bg = Arc::clone(runtime);
    let ws_bg = app.workspace.clone();
    let sid_bg = app.session_id.clone();
    let tx = panel_ctrl.tx.clone();
    panel_ctrl.in_flight = Some(tokio::spawn(async move {
        let snapshot = collect_workspace_panel_snapshot(&runtime_bg, &ws_bg, Some(&sid_bg)).await;
        let _ = tx.send((generation, sid_bg, snapshot));
    }));
}

fn rollback_trailing_user_bubble(app: &mut AppState, expected_text: &str) {
    if app.items.len() > app.inline_committed_items
        && matches!(app.items.last(), Some(ChatItem::User { text }) if text == expected_text)
    {
        let idx = app.items.len() - 1;
        app.items.pop();
        app.item_cache.invalidate_item(idx);
    }
}

pub(super) fn populate_tools_and_mcps(runtime: &MizarRuntime, app: &mut AppState) {
    if !app.features.side_panel {
        return;
    }
    app.side_panel_data.tools = collect_tools_summary(runtime, Some(&app.session_id));
    app.side_panel_data.mcps = collect_mcps_summary(runtime);
}

pub(super) async fn populate_side_panel_data(runtime: &MizarRuntime, app: &mut AppState) {
    if !app.features.side_panel {
        return;
    }
    app.refresh_context_breakdown();
    app.side_panel_data.files = app.touched_files.clone();
    let snapshot =
        collect_workspace_panel_snapshot(runtime, &app.workspace, Some(&app.session_id)).await;
    snapshot.apply_to_side_panel(&mut app.side_panel_data, &app.anomalies);
}

#[cfg(test)]
pub(super) async fn apply_module_toggle_setting(
    runtime: &Arc<MizarRuntime>,
    app: &mut AppState,
    settings_path: Option<&Path>,
    plugin_id: &str,
    enabled: bool,
) {
    apply_module_toggle_setting_with_panel(runtime, app, settings_path, plugin_id, enabled, None)
        .await;
}

pub(super) async fn apply_module_toggle_setting_with_panel(
    runtime: &Arc<MizarRuntime>,
    app: &mut AppState,
    settings_path: Option<&Path>,
    plugin_id: &str,
    enabled: bool,
    panel_ctrl: Option<&mut SidePanelRefreshController>,
) {
    if app.is_running_activity() {
        app.overlay
            .set_setting_toggle(&format!("plugin.{plugin_id}"), !enabled);
        let msg = "当前有正在执行的任务，请等待完成后再切换模块";
        app.push_notice(NoticeLevel::Warn, msg);
        app.set_flash(msg);
        app.overlay
            .refresh_settings(super::build_settings_entries(runtime, app));
        return;
    }

    let outcome = toggle_spine_module(runtime, settings_path, plugin_id, enabled).await;
    if outcome.ok {
        app.user_skill_names = collect_user_skill_names(runtime, &app.workspace).await;
        if let Some(ctrl) = panel_ctrl {
            schedule_side_panel_refresh(runtime, app, ctrl);
        } else {
            populate_side_panel_data(runtime, app).await;
        }
    }
    app.overlay
        .refresh_settings(super::build_settings_entries(runtime, app));
    if let Some(flash) = outcome.flash {
        app.set_flash(flash);
    }
    for (level, msg) in outcome.notices {
        app.push_notice(level, msg);
    }
}

pub(super) async fn apply_host_setting_change(
    runtime: &Arc<MizarRuntime>,
    app: &mut AppState,
    terminal: &mut BufferedTerminal,
    panel_ctrl: &mut SidePanelRefreshController,
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
            schedule_side_panel_refresh(runtime, app, panel_ctrl);
        }
        k if k.starts_with("compaction.") || k.starts_with("sessionTitle.") => {
            apply_runtime_setting_change(runtime, key, value);
        }
        _ => {}
    }
}

fn format_picker_cwd_tail(raw: Option<&str>) -> String {
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

fn format_picker_updated_at(ms: u64) -> String {
    i64::try_from(ms)
        .ok()
        .filter(|&v| v > 0)
        .and_then(DateTime::from_timestamp_millis)
        .map(|utc| utc.with_timezone(&Local).format("%m-%d %H:%M").to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn build_picker_items_from_list(items: Vec<SessionListItemDto>) -> Vec<SessionPickerItem> {
    let self_pid = std::process::id();
    items
        .into_iter()
        .filter(|h| h.event_count > 0)
        .map(|h| {
            let base_title = h
                .title
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("(无标题 #{})", short_id(&h.id)));
            let title = match h.active_writer_pid {
                Some(pid) if pid != self_pid => format!("{base_title} [PID {pid} 占用]"),
                _ => base_title,
            };
            SessionPickerItem {
                title,
                cwd_tail: format_picker_cwd_tail(h.cwd.as_deref()),
                event_count: h.event_count,
                updated_at: format_picker_updated_at(h.updated_at_ms),
                session_id: h.id,
            }
        })
        .collect()
}

/// 在已运行的 TUI 状态上切换到指定历史会话（`/resume` 选择器与 `--resume` 共用投影与路由恢复）。
pub async fn resume_session_into_app(
    runtime: &MizarRuntime,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    active_sid: &Arc<RwLock<String>>,
    target_id: &str,
) -> bool {
    if let Some(ref run_dir) = config.run_dir
        && let Err(err) = check_session_writable(run_dir, target_id, std::process::id())
    {
        app.push_notice(
            NoticeLevel::Error,
            format!("无法恢复会话 `{target_id}`：{err}（单写者锁已拒绝双写）"),
        );
        app.set_flash(format!("⚠ 会话被 PID {} 锁定，未切换", err.owner_pid));
        return false;
    }
    let Some(next) = runtime.sessions.get(target_id) else {
        app.push_notice(NoticeLevel::Warn, format!("未找到历史会话 `{target_id}`"));
        return false;
    };
    app.pending_settings_model_pick = None;
    runtime.agent_loop.inbox().clear(&next.id);
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
        populate_side_panel_data(runtime, app).await;
    }
    app.set_flash(format!("✓ 已恢复会话 ({})", short_id(target_id)));
    true
}

/// 通过 `SessionActorHandle::switch_session` 执行带单写者锁校验的会话切换，并同步恢复 TUI 投影与模型选择。
#[allow(clippy::too_many_arguments)]
pub(super) async fn resume_session_via_actor(
    actor: &SessionActorHandle,
    runtime: &MizarRuntime,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    sync: &mut TuiActorSync,
    task_state: &mut RunController,
    target_id: &str,
) -> bool {
    let next = match actor.switch_session(target_id).await {
        Ok(sess) => sess,
        Err(ActorError::SessionLocked(err)) => {
            app.push_notice(
                NoticeLevel::Error,
                format!("无法恢复会话 `{target_id}`：{err}（单写者锁已拒绝双写）"),
            );
            app.set_flash(format!("⚠ 会话被 PID {} 锁定，未切换", err.owner_pid));
            return false;
        }
        Err(ActorError::SessionNotFound(id)) => {
            app.push_notice(NoticeLevel::Warn, format!("未找到历史会话 `{id}`"));
            return false;
        }
        Err(err) => {
            app.push_notice(NoticeLevel::Error, format!("切换会话失败：{err}"));
            return false;
        }
    };

    app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
    task_state.abort_for_session_switch();
    if app.is_running_activity() {
        app.set_idle_state();
    }
    app.pending_settings_model_pick = None;

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
    if let Ok(snap) = actor.subscribe_snapshot(&active_session.id, None).await
        && !snap.queued_inputs.is_empty()
    {
        apply_queue_snapshot_to_app(app, snap.queued_inputs);
    }
    sync.mark_session_id(active_session.id.clone());
    sync.sync_config_if_changed(actor, app).await;
    sync.reset_for_session(active_session, app);

    if app.features.side_panel && app.side_panel.open {
        populate_side_panel_data(runtime, app).await;
    }
    app.set_flash(format!("✓ 已恢复会话 ({})", short_id(target_id)));
    true
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_action(
    actor: &SessionActorHandle,
    runtime: &Arc<MizarRuntime>,
    config: &TuiConfig,
    app: &mut AppState,
    active_session: &mut Arc<Session>,
    terminal: &mut BufferedTerminal,
    term_stream: &mut Option<EventStream>,
    loop_tx: &mpsc::UnboundedSender<LoopClientMsg>,
    task_state: &mut RunController,
    panel_ctrl: &mut SidePanelRefreshController,
    sync: &mut TuiActorSync,
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
            cancel_active_run_via_actor(actor, app, task_state, sync).await;
        }
        AppAction::CompactSession(focus) => {
            if app.is_running_activity() {
                app.push_notice(
                    NoticeLevel::Warn,
                    "当前已有运行中的任务，请等待完成后再压缩",
                );
                return;
            }
            let Some(compaction) = runtime.compaction.as_ref().map(Arc::clone) else {
                app.push_notice(NoticeLevel::Warn, "compaction 服务未启用");
                return;
            };
            if !compaction.is_module_enabled() {
                app.push_notice(NoticeLevel::Warn, "compaction 服务未启用");
                return;
            }
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
                        // Pipeline 内部已广播 `on_end` 事件，避免重复投递
                    }
                    Err(err @ CompactionError::AlreadyRunning(_)) => {
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
        AppAction::NewSession => match actor.new_session(None).await {
            Ok(next) => {
                app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
                task_state.abort_for_session_switch();
                if app.is_running_activity() {
                    app.set_idle_state();
                }
                let title = next.header().title;
                app.reset_for_new_session(next.id.clone(), title);
                *active_session = next;
                sync.reset_for_session(active_session, app);
                schedule_side_panel_refresh(runtime, app, panel_ctrl);
            }
            Err(err) => {
                app.push_notice(NoticeLevel::Error, format!("开启新会话失败：{err}"));
            }
        },
        AppAction::OpenSessionPicker => {
            match actor
                .list_sessions(SessionListParams {
                    cwd: None,
                    query: None,
                    limit: Some(100),
                    cursor: None,
                    exclude_subagents: Some(true),
                })
                .await
            {
                Ok(res) => {
                    let items = build_picker_items_from_list(res.sessions);
                    if items.is_empty() {
                        app.push_notice(NoticeLevel::Info, "暂无可恢复的历史会话");
                    } else {
                        app.overlay.open_session_picker(items);
                    }
                }
                Err(err) => {
                    app.push_notice(NoticeLevel::Error, format!("读取历史会话列表失败：{err}"));
                }
            }
        }
        AppAction::ResumeSession(target_id) => {
            if resume_session_via_actor(
                actor,
                runtime,
                config,
                app,
                active_session,
                sync,
                task_state,
                &target_id,
            )
            .await
            {
                schedule_side_panel_refresh(runtime, app, panel_ctrl);
            }
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
                let Some(st) = runtime.session_title.as_ref().map(Arc::clone) else {
                    app.push_notice(
                        NoticeLevel::Warn,
                        "session-title 服务未启用（可在 /settings 中开启）",
                    );
                    return;
                };
                if !st.is_module_enabled() {
                    app.push_notice(
                        NoticeLevel::Warn,
                        "session-title 服务未启用（可在 /settings 中开启）",
                    );
                    return;
                }
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
                rollback_trailing_user_bubble(app, &prompt);
                app.input.set_text_end(&prompt);
                app.reject_unsupported_images_notice();
                return;
            }
            submit_turn_via_actor(actor, app, sync, prompt, Vec::new()).await;
        }
        AppAction::SubmitPromptWithImages { text, images } => {
            if !app.active_model_supports_images() {
                rollback_trailing_user_bubble(app, &text);
                app.reject_unsupported_images_notice();
                return;
            }
            submit_turn_via_actor(actor, app, sync, text, images).await;
        }
        AppAction::SteerPrompt(prompt) => {
            handle_steer_prompt_via_actor(actor, config, app, sync, prompt, Vec::new()).await;
        }
        AppAction::SteerPromptWithImages { text, images } => {
            handle_steer_prompt_via_actor(actor, config, app, sync, text, images).await;
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
            if let Some(traj) = runtime.trajectory.as_ref() {
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
            let report = super::build_session_info_report(runtime, config, active_session, app);
            app.push_notice(NoticeLevel::Info, report);
        }
        AppAction::ShowTools => {
            let rt = &runtime.tools;
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
        }
        AppAction::ShowMcp => {
            if !runtime.modules.is_enabled("mcp") {
                app.push_notice(NoticeLevel::Warn, "MCP 服务未启用");
            } else if let Some(mgr) = runtime.mcp().await {
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
            if let Some(reg) = runtime.skills.as_ref()
                && reg.is_enabled()
            {
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
            if let Some(inst) = runtime.instructions.as_ref() {
                let report = inst.report();
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
            if let Some(traj) = runtime.trajectory.as_ref() {
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
            app.providers = runtime.llm.provider_infos();
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
            schedule_side_panel_refresh(runtime, app, panel_ctrl);
        }
        AppAction::RefreshSidePanelTools => {
            if app.features.side_panel && app.side_panel.open {
                populate_tools_and_mcps(runtime, app);
            }
        }
        AppAction::OpenTrajectoryScene(mode) => {
            if !app.features.trajectory_view {
                return;
            }
            let report = runtime
                .trajectory
                .as_ref()
                .and_then(|traj| traj.inspect(&app.session_id, mode));
            app.open_trajectory_with_report(report);
        }
        AppAction::OpenSessionTree => match actor.session_tree(&app.session_id).await {
            Ok(res) => app.open_session_tree_with(res.entries),
            Err(err) => {
                app.push_notice(NoticeLevel::Error, format!("读取会话树失败：{err}"));
            }
        },
        AppAction::OpenSettings(initial_query) => {
            app.pending_settings_model_pick = None;
            let items = super::build_settings_entries(runtime, app);
            let save_hint = resolve_global_settings_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "~/.mizar/settings.json".to_string());
            app.overlay.open_settings(items, initial_query, save_hint);
        }
        AppAction::ApplySetting { key, value } => {
            if let Some(plugin_id) = key.strip_prefix("plugin.") {
                let settings_path = resolve_global_settings_path();
                apply_module_toggle_setting_with_panel(
                    runtime,
                    app,
                    settings_path.as_deref(),
                    plugin_id,
                    value == "true",
                    Some(panel_ctrl),
                )
                .await;
                return;
            }
            apply_host_setting_change(runtime, app, terminal, panel_ctrl, &key, &value).await;
            if let Err(err) = persist_setting_change(&key, &value) {
                app.push_notice(
                    NoticeLevel::Warn,
                    format!("配置 `{key}` 已实时生效，但写入 settings.json 失败：{err}"),
                );
            }
        }
        AppAction::OpenSkillPicker => {
            if let Some(reg) = runtime.skills.as_ref()
                && reg.is_enabled()
            {
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
            match actor
                .fork_session(SessionForkParams {
                    session_id: app.session_id.clone(),
                    at_seq: target_seq,
                })
                .await
            {
                Ok(forked) => match actor.switch_session(&forked.session.id).await {
                    Ok(next) => {
                        app.overlay.resolve_all_approvals(ApprovalDecision::Deny);
                        task_state.abort_for_session_switch();
                        if app.is_running_activity() {
                            app.set_idle_state();
                        }
                        app.restore_session_view(project_fork_view(&next, target_seq));
                        *active_session = next;
                        sync.reset_for_session(active_session, app);
                        schedule_side_panel_refresh(runtime, app, panel_ctrl);
                    }
                    Err(err) => {
                        app.push_notice(NoticeLevel::Error, format!("切换分叉会话失败：{err}"));
                    }
                },
                Err(err) => {
                    app.push_notice(NoticeLevel::Error, format!("分叉会话失败：{err}"));
                }
            }
        }
        AppAction::OpenExternalEditor => {
            let draft = app.input.expand_text();
            let tmp_path: PathBuf =
                std::env::temp_dir().join(format!("mizar-draft-{}.md", std::process::id()));
            if let Err(err) = tokio::fs::write(&tmp_path, &draft).await {
                app.push_notice(NoticeLevel::Error, format!("写入临时草稿文件失败：{err}"));
                return;
            }
            let editor_cmd = std::env::var("VISUAL")
                .or_else(|_| std::env::var("EDITOR"))
                .unwrap_or_else(|_| "vi".to_string());
            let mut parts = editor_cmd.split_whitespace();
            let bin = parts.next().unwrap_or("vi").to_string();
            let extra_args: Vec<String> = parts.map(ToString::to_string).collect();

            // 挂起前必须释放 EventStream，防止后台 tty 读取线程与外部编辑器争抢按键导致丢键
            *term_stream = None;
            let _ = suspend_for_external_editor();
            let bin_owned = bin.clone();
            let tmp_owned = tmp_path.clone();
            let status = tokio::task::spawn_blocking(move || {
                std::process::Command::new(&bin_owned)
                    .args(&extra_args)
                    .arg(&tmp_owned)
                    .status()
            })
            .await;
            let _ = resume_after_external_editor(app.screen_mode, app.mouse_capture_enabled);
            if let Ok(rebuilt) = create_terminal(app.screen_mode) {
                *terminal = rebuilt;
            }
            *term_stream = Some(EventStream::new());

            match status {
                Ok(Ok(s)) if s.success() => match tokio::fs::read_to_string(&tmp_path).await {
                    Ok(edited) => {
                        let clean = edited.strip_suffix('\n').unwrap_or(&edited);
                        app.input.set_text_end(clean);
                        app.set_flash("✓ 已从外部编辑器同步草稿");
                    }
                    Err(err) => {
                        app.push_notice(NoticeLevel::Error, format!("读取编辑结果失败：{err}"));
                    }
                },
                Ok(Ok(s)) => {
                    app.push_notice(
                        NoticeLevel::Warn,
                        format!("外部编辑器 `{bin}` 退出码：{:?}", s.code()),
                    );
                }
                Ok(Err(err)) => {
                    app.push_notice(
                        NoticeLevel::Error,
                        format!("启动外部编辑器 `{bin}` 失败：{err}"),
                    );
                }
                Err(err) => {
                    app.push_notice(
                        NoticeLevel::Error,
                        format!("等待外部编辑器 `{bin}` 任务失败：{err}"),
                    );
                }
            }
            let _ = tokio::fs::remove_file(&tmp_path).await;
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

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use mizar_llm::{ModelInfo, ProviderInfo};
    use mizar_tui_ui::app::{AppAction, AppState, ChatItem};

    use super::{mentions, rollback_trailing_user_bubble};

    #[tokio::test]
    async fn at_image_rejection_on_non_vision_model_rolls_back_ghost_user_bubble() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ws = std::env::temp_dir().join(format!(
            "mizar_tui_ghost_bubble_{}_{uniq}",
            std::process::id()
        ));
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(
            ws.join("shot.png"),
            [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
        )
        .unwrap();

        let providers = vec![ProviderInfo {
            id: "mock".into(),
            name: "Mock".into(),
            protocol: "mock".into(),
            models: vec![ModelInfo {
                id: "text-only".into(),
                name: "Text Only".into(),
                input: vec!["text".into()],
                ..Default::default()
            }],
        }];
        let mut app = AppState::new(
            ws.clone(),
            "sess-ghost",
            providers,
            Some("mock".into()),
            Some("text-only".into()),
            None,
        );
        app.input.insert_str("请看 @shot.png ");
        let action = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        let AppAction::SubmitPrompt(prompt) = action else {
            panic!("expected SubmitPrompt, got {action:?}");
        };
        assert_eq!(app.items.len(), 1);

        assert!(!app.active_model_supports_images());
        assert!(mentions::mentions_resolve_any_image(&ws, false, &prompt).await);

        rollback_trailing_user_bubble(&mut app, &prompt);
        app.input.set_text_end(&prompt);
        app.reject_unsupported_images_notice();

        assert!(
            !app.items
                .iter()
                .any(|it| matches!(it, ChatItem::User { .. })),
            "rejected @image prompt must not leave a ghost user bubble: {:?}",
            app.items
        );
        assert_eq!(app.input.text(), "请看 @shot.png");

        let _ = std::fs::remove_dir_all(&ws);
    }
}
