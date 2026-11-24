//! 前端无关的会话运行内核（`SessionActor` 与 `SessionActorHandle`）。
//!
//! 统一承接单会话主循环调度、`ActiveTurnBuffer` 瞬态流缓冲（工具输出尾部 UTF-8 安全上限 64 KiB）、
//! `PendingInteractions` 双源原子抢占、单真源 `queued_inputs`、`FollowUp` 自动唤醒、
//! `SessionStore::on_event` 提交桥接、`SubagentTaskSnapshot` 状态同步与单写者会话锁校验。
//!
//! 同时向外投递两路事件流：
//! - 零序列化的 `ClientEvent`（供同进程 TUI / Merak Desktop 直接消费）；
//! - 强类型 `RpcNotification`（供本地 Unix Socket JSON-RPC 服务广播给多端观察者）。

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use mizar_core::agent_loop::{
    AgentError, AgentEvent, DeliveryMode, RunOptions, TurnEndReason, TurnInput,
};
use mizar_core::skills::SkillLookupOptions;
use mizar_core::spine::{CompactionTrigger, MizarRuntime};
use mizar_core::subagent::{SubagentMode, SubagentRunStatus, SubagentTaskSnapshot};
use mizar_llm::{ImageAttachment, ReasoningEffort};
use mizar_mcp::{McpConnectionState, McpManager};
use mizar_session::{
    EntryVisibility, PersistenceError, Session, SessionCreateOptions, SessionEntry, SessionEvent,
    SessionListQuery, SessionOrigin, SessionTraceEvent,
};
use mizar_types::HookGuard;
use mizar_types::client::{
    ApprovalDecision, ApprovalMode, ClientEvent, HostExtensionFlags, McpSummaryRow, SkillPickerItem,
};
use mizar_types::rpc::{
    ActiveTurnDto, CapabilitiesListResult, CommittedEventDto, InteractionResolveParams,
    InteractionResolveResult, JsonRpcError, QueuePopAllResult, QueueRemoveResult,
    QueueUpdatedNotification, QueuedInputDto, ResolvedBy, RpcApprovalMode, RpcAttachmentDto,
    RpcEventVisibility, RpcNotification, RpcSubagentMode, RpcSubagentStatus, RunningToolDto,
    SessionConfigDto, SessionConfigParams, SessionConfigResult, SessionContextUsageDto,
    SessionEventsParams, SessionEventsResult, SessionForkParams, SessionForkResult,
    SessionListItemDto, SessionListParams, SessionListResult, SessionMetaDto, SessionRuntimeState,
    SessionSubscribeParams, SessionSubscribeSnapshot, SessionSwitchedNotification,
    SessionTreeResult, SlashCommandDto, StreamDeltaKind, StreamDeltaNotification,
    SubagentSnapshotDto, SubagentUpdatedNotification, SubmitMode, TurnCancelResult,
    TurnStateNotification, TurnSubmitParams, TurnSubmitResult,
};
use serde_json::json;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::approval::{
    ApprovalGateConfig, ApprovalSideEffects, ApprovalSnapshot, PendingInteractions,
    ResolveInteractionError, install_approval_gate,
};
use crate::events::{
    ActiveTaskKind, RunController, to_ui_status, to_ui_step_outcome, to_ui_turn_end_reason,
};
use crate::images::{ImageProcessError, process_image_bytes};
use crate::mentions::{FileMentionsConfig, install_file_mentions};
use crate::run_dir::{
    ProcessRegistration, SessionLockError, check_session_writable, scan_live_processes,
};
use crate::session_view::project_session_tree;
use crate::workspace::collect_tools_summary;

/// `ActiveTurnBuffer` 中每个运行中工具的 `output_tail` 最大字节上限（64 KiB）。
pub const ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES: usize = 64 * 1024;
const DEFAULT_BROADCAST_CAPACITY: usize = 4096;
const DEFAULT_SESSION_LIST_LIMIT: usize = 50;
const DEFAULT_EVENTS_PAGE_LIMIT: usize = 200;
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

/// 按 UTF-8 字符边界安全截取字符串尾部，确保结果字节长度严格 `<= max_bytes`。
pub fn tail_utf8_capped(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut start = s.len().saturating_sub(max_bytes);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    s[start..].to_string()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// 内存中尚未完全落盘的回合生成瞬态缓冲（支撑断线重连或新观察者水合）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActiveTurnBuffer {
    pub turn: u64,
    pub step: usize,
    pub reasoning: String,
    pub text: String,
    pub running_tools: Vec<RunningToolDto>,
}

impl ActiveTurnBuffer {
    pub fn new(turn: u64) -> Self {
        Self {
            turn,
            step: 0,
            reasoning: String::new(),
            text: String::new(),
            running_tools: Vec::new(),
        }
    }

    pub fn begin_step(&mut self, step: usize) {
        self.step = step;
        self.reasoning.clear();
        self.text.clear();
    }

    pub fn clear_step_text(&mut self) {
        self.reasoning.clear();
        self.text.clear();
    }

    pub fn push_reasoning(&mut self, delta: &str) {
        self.reasoning.push_str(delta);
    }

    pub fn push_text(&mut self, delta: &str) {
        self.text.push_str(delta);
    }

    pub fn start_tool(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) {
        let call_id = call_id.into();
        let name = name.into();
        let arguments = arguments.into();
        if let Some(existing) = self.running_tools.iter_mut().find(|t| t.call_id == call_id) {
            existing.name = name;
            existing.arguments = arguments;
            return;
        }
        self.running_tools.push(RunningToolDto {
            call_id,
            name,
            arguments,
            output_tail: String::new(),
        });
    }

    pub fn update_tool_output(&mut self, call_id: &str, preview: &str) {
        if let Some(tool) = self.running_tools.iter_mut().find(|t| t.call_id == call_id) {
            tool.output_tail = tail_utf8_capped(preview, ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES);
        }
    }

    pub fn append_tool_output(&mut self, call_id: &str, chunk: &str) {
        if let Some(tool) = self.running_tools.iter_mut().find(|t| t.call_id == call_id) {
            tool.output_tail.push_str(chunk);
            if tool.output_tail.len() > ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES {
                tool.output_tail =
                    tail_utf8_capped(&tool.output_tail, ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES);
            }
        }
    }

    pub fn finish_tool(&mut self, call_id: &str) {
        self.running_tools.retain(|t| t.call_id != call_id);
    }

    pub fn to_dto(&self) -> ActiveTurnDto {
        ActiveTurnDto {
            turn: self.turn,
            step: self.step,
            reasoning: self.reasoning.clone(),
            text: self.text.clone(),
            running_tools: self.running_tools.clone(),
        }
    }
}

/// `SessionActor` 启动配置。
#[derive(Debug)]
pub struct SessionActorConfig {
    pub workspace: PathBuf,
    pub session_db_path: Option<PathBuf>,
    pub run_dir: Option<PathBuf>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub approval_mode: ApprovalMode,
    pub always_allowed_tools: HashSet<String>,
    pub max_steps: Option<usize>,
    pub restrict_to_workspace: bool,
    pub host_extensions: HostExtensionFlags,
    pub registration: Option<ProcessRegistration>,
}

impl Default for SessionActorConfig {
    fn default() -> Self {
        Self::new(PathBuf::from("."))
    }
}

impl SessionActorConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            session_db_path: None,
            run_dir: None,
            provider: None,
            model: None,
            reasoning_effort: None,
            approval_mode: ApprovalMode::Auto,
            always_allowed_tools: HashSet::new(),
            max_steps: None,
            restrict_to_workspace: false,
            host_extensions: HostExtensionFlags::default(),
            registration: None,
        }
    }

    pub fn with_run_dir(mut self, run_dir: impl Into<PathBuf>) -> Self {
        self.run_dir = Some(run_dir.into());
        self
    }

    pub fn with_approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    pub fn with_registration(mut self, registration: ProcessRegistration) -> Self {
        self.registration = Some(registration);
        self
    }

    pub fn with_provider_model(
        mut self,
        provider: Option<String>,
        model: Option<String>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        self.provider = provider;
        self.model = model;
        self.reasoning_effort = reasoning_effort;
        self
    }
}

/// `SessionActorHandle` 操作失败错误类型，可直接映射为 JSON-RPC 错误码。
#[derive(Debug, Error)]
pub enum ActorError {
    #[error("会话未找到: {0}")]
    SessionNotFound(String),
    #[error(transparent)]
    SessionLocked(#[from] SessionLockError),
    #[error("无效参数: {0}")]
    InvalidParams(String),
    #[error("持久化失败: {0}")]
    Persistence(#[from] PersistenceError),
    #[error("附件处理失败: {0}")]
    Attachment(String),
    #[error("会话内核已关闭")]
    Shutdown,
}

impl ActorError {
    pub fn to_jsonrpc_error(&self) -> JsonRpcError {
        match self {
            Self::SessionNotFound(id) => JsonRpcError::session_not_found(id),
            Self::SessionLocked(err) => JsonRpcError::session_locked(
                err.to_string(),
                Some(json!({
                    "session_id": err.session_id,
                    "owner_pid": err.owner_pid,
                    "socket": err.socket_path,
                })),
            ),
            Self::InvalidParams(msg) | Self::Attachment(msg) => {
                JsonRpcError::invalid_params(msg.clone())
            }
            Self::Persistence(err) => JsonRpcError::internal_error(err.to_string()),
            Self::Shutdown => JsonRpcError::internal_error("session actor is shut down"),
        }
    }
}

#[derive(Debug, Clone)]
struct RuntimeConfigState {
    provider: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
    approval_mode: ApprovalMode,
}

impl RuntimeConfigState {
    fn to_dto(&self) -> SessionConfigDto {
        SessionConfigDto {
            provider: self.provider.clone(),
            model: self.model.clone(),
            reasoning_effort: self.reasoning_effort,
            approval_mode: RpcApprovalMode::from(self.approval_mode),
        }
    }
}

#[derive(Debug, Clone)]
struct QueuedInputEntry {
    dto: QueuedInputDto,
    images: Vec<ImageAttachment>,
}

#[derive(Default)]
struct QueueState {
    next_id: u64,
    by_session: HashMap<String, VecDeque<QueuedInputEntry>>,
}

impl QueueState {
    fn push(
        &mut self,
        session_id: &str,
        text: String,
        attachments: Vec<RpcAttachmentDto>,
        images: Vec<ImageAttachment>,
    ) -> (QueuedInputDto, Vec<QueuedInputDto>) {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let dto = QueuedInputDto {
            id: self.next_id,
            text,
            attachments,
            created_at_ms: now_unix_ms(),
        };
        let queue = self.by_session.entry(session_id.to_string()).or_default();
        queue.push_back(QueuedInputEntry {
            dto: dto.clone(),
            images,
        });
        let snapshot = queue.iter().map(|e| e.dto.clone()).collect();
        (dto, snapshot)
    }

    fn remove(
        &mut self,
        session_id: &str,
        id: u64,
    ) -> (Option<QueuedInputDto>, Vec<QueuedInputDto>) {
        let Some(queue) = self.by_session.get_mut(session_id) else {
            return (None, Vec::new());
        };
        let pos = queue.iter().position(|e| e.dto.id == id);
        let removed = pos.and_then(|idx| queue.remove(idx)).map(|e| e.dto);
        let snapshot = queue.iter().map(|e| e.dto.clone()).collect();
        (removed, snapshot)
    }

    fn pop_all(&mut self, session_id: &str) -> (Vec<QueuedInputDto>, Vec<QueuedInputDto>) {
        let Some(queue) = self.by_session.get_mut(session_id) else {
            return (Vec::new(), Vec::new());
        };
        let popped = queue.drain(..).map(|e| e.dto).collect();
        (popped, Vec::new())
    }

    fn pop_front(&mut self, session_id: &str) -> Option<(QueuedInputEntry, Vec<QueuedInputDto>)> {
        let queue = self.by_session.get_mut(session_id)?;
        let item = queue.pop_front()?;
        let snapshot = queue.iter().map(|e| e.dto.clone()).collect();
        Some((item, snapshot))
    }

    fn list(&self, session_id: &str) -> Vec<QueuedInputDto> {
        self.by_session
            .get(session_id)
            .map(|q| q.iter().map(|e| e.dto.clone()).collect())
            .unwrap_or_default()
    }
}

#[derive(Debug)]
struct RunDoneSignal {
    session_id: String,
    user_cancelled: bool,
    allow_queue_drain: bool,
}

struct ActorInner {
    runtime: Arc<MizarRuntime>,
    workspace: PathBuf,
    run_dir: Option<PathBuf>,
    max_steps: Option<usize>,
    active_session: Mutex<Arc<Session>>,
    active_session_id: Arc<RwLock<String>>,
    runtime_config: Mutex<RuntimeConfigState>,
    approval_snapshot: Arc<RwLock<ApprovalSnapshot>>,
    pending_interactions: Arc<PendingInteractions>,
    run_controller: Mutex<RunController>,
    active_turns: Mutex<HashMap<String, ActiveTurnBuffer>>,
    queued_inputs: Mutex<QueueState>,
    registration: Mutex<Option<ProcessRegistration>>,
    client_tx: broadcast::Sender<ClientEvent>,
    notif_tx: broadcast::Sender<RpcNotification>,
    run_done_tx: mpsc::UnboundedSender<RunDoneSignal>,
    shutdown_cancel: CancellationToken,
    is_shutdown: AtomicBool,
    guards: Mutex<Vec<HookGuard>>,
    wake_task: Mutex<Option<JoinHandle<()>>>,
}

impl ActorInner {
    fn current_session(&self) -> Arc<Session> {
        Arc::clone(&self.active_session.lock().unwrap())
    }

    fn current_session_id(&self) -> String {
        self.active_session_id.read().unwrap().clone()
    }

    fn emit_client(&self, ev: ClientEvent) {
        let _ = self.client_tx.send(ev);
    }

    fn emit_rpc(&self, notif: RpcNotification) {
        let _ = self.notif_tx.send(notif);
    }

    fn emit_queue_updated(&self, session_id: &str, items: Vec<QueuedInputDto>) {
        if session_id == self.current_session_id() {
            self.emit_client(ClientEvent::QueueUpdated {
                items: items.clone(),
            });
        }
        self.emit_rpc(RpcNotification::QueueUpdated(QueueUpdatedNotification {
            session_id: session_id.to_string(),
            items,
        }));
    }

    fn resolve_session_state(&self, session_id: &str) -> SessionRuntimeState {
        if self
            .pending_interactions
            .has_pending_for_session(session_id)
        {
            return SessionRuntimeState::WaitingApproval;
        }
        let has_active_turn = self.active_turns.lock().unwrap().contains_key(session_id);
        if has_active_turn || self.runtime.agent_loop.is_session_busy(session_id) {
            return SessionRuntimeState::Streaming;
        }
        if let Some(task) = self.run_controller.lock().unwrap().active_task()
            && task.session_id == session_id
        {
            return SessionRuntimeState::Streaming;
        }
        SessionRuntimeState::Idle
    }

    fn try_start_turn_if_idle(
        self: &Arc<Self>,
        session: Arc<Session>,
        produce_input: impl FnOnce() -> Option<TurnInput>,
        provider_override: Option<String>,
        model_override: Option<String>,
        reasoning_override: Option<ReasoningEffort>,
    ) -> Option<u64> {
        let sid = session.id.clone();
        let (run_id, cancel) = {
            let mut ctrl = self.run_controller.lock().unwrap();
            if ctrl.active_task().is_some() || self.runtime.agent_loop.is_session_busy(&sid) {
                return None;
            }
            ctrl.start(sid.clone(), ActiveTaskKind::AgentRun)
        };

        let Some(input) = produce_input() else {
            let cur_sid = self.current_session_id();
            self.run_controller.lock().unwrap().finish_if_current(
                Some(run_id),
                Some(&sid),
                &cur_sid,
                ActiveTaskKind::AgentRun,
            );
            return None;
        };

        let next_turn = peek_next_turn(&session);
        let cfg = self.runtime_config.lock().unwrap().clone();
        let provider = provider_override.or(cfg.provider);
        let model = model_override.or(cfg.model);
        let reasoning_effort = reasoning_override.or(cfg.reasoning_effort);

        session.record_selection(provider.clone(), model.clone(), reasoning_effort);

        let run_opts = RunOptions {
            provider,
            model,
            max_steps: self.max_steps,
            max_tokens: None,
            temperature: None,
            reasoning_effort,
            images: Vec::new(),
            ..Default::default()
        };

        let inner = Arc::clone(self);
        tokio::spawn(async move {
            let res = inner
                .runtime
                .agent_loop
                .run_with_report(Arc::clone(&session), input, cancel, run_opts)
                .await;

            let cur_sid = inner.current_session_id();
            let finished = inner.run_controller.lock().unwrap().finish_if_current(
                Some(run_id),
                Some(&sid),
                &cur_sid,
                ActiveTaskKind::AgentRun,
            );
            let Some(task) = finished else {
                return;
            };

            let allow_queue_drain = !task.user_cancelled
                && matches!(&res, Ok(report) if report.reason == TurnEndReason::Completed);
            let client_res = match res {
                Ok(report) => Ok(report.text),
                Err(AgentError::Llm(err)) => Err(err.to_string()),
                Err(err) => Err(err.to_string()),
            };
            inner.emit_client(ClientEvent::RunFinished(client_res));
            let _ = inner.run_done_tx.send(RunDoneSignal {
                session_id: sid,
                user_cancelled: task.user_cancelled,
                allow_queue_drain,
            });
        });

        Some(next_turn)
    }
}

impl Drop for ActorInner {
    fn drop(&mut self) {
        self.shutdown_cancel.cancel();
        let guards = std::mem::take(&mut *self.guards.lock().unwrap());
        for g in guards.into_iter().rev() {
            drop(g);
        }
    }
}

/// 单会话运行内核宿主，负责装配事件桥接、审批门禁、`FollowUp` 唤醒与异步生命周期管理。
pub struct SessionActor;

impl SessionActor {
    /// 构建并启动 `SessionActor`，返回可跨端共享克隆的 `SessionActorHandle`。
    pub fn spawn(
        runtime: Arc<MizarRuntime>,
        initial_session: Arc<Session>,
        mut config: SessionActorConfig,
    ) -> SessionActorHandle {
        let initial_sid = initial_session.id.clone();
        let active_session_id = Arc::new(RwLock::new(initial_sid.clone()));
        let approval_snapshot = Arc::new(RwLock::new(ApprovalSnapshot {
            mode: config.approval_mode,
            always_allowed: config.always_allowed_tools.clone(),
        }));
        let pending_interactions = Arc::new(PendingInteractions::new());
        let (client_tx, _) = broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
        let (notif_tx, _) = broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
        let (run_done_tx, run_done_rx) = mpsc::unbounded_channel();

        if let Some(ref mut reg) = config.registration {
            let _ = reg.update_session(&initial_sid);
            let _ = reg.update_model(config.provider.clone(), config.model.clone());
        }

        let inner = Arc::new(ActorInner {
            runtime: Arc::clone(&runtime),
            workspace: config.workspace.clone(),
            run_dir: config.run_dir.clone(),
            max_steps: config.max_steps,
            active_session: Mutex::new(initial_session),
            active_session_id: Arc::clone(&active_session_id),
            runtime_config: Mutex::new(RuntimeConfigState {
                provider: config.provider.clone(),
                model: config.model.clone(),
                reasoning_effort: config.reasoning_effort,
                approval_mode: config.approval_mode,
            }),
            approval_snapshot: Arc::clone(&approval_snapshot),
            pending_interactions: Arc::clone(&pending_interactions),
            run_controller: Mutex::new(RunController::new()),
            active_turns: Mutex::new(HashMap::new()),
            queued_inputs: Mutex::new(QueueState::default()),
            registration: Mutex::new(config.registration.take()),
            client_tx,
            notif_tx,
            run_done_tx,
            shutdown_cancel: CancellationToken::new(),
            is_shutdown: AtomicBool::new(false),
            guards: Mutex::new(Vec::new()),
            wake_task: Mutex::new(None),
        });

        let mut guards = Vec::new();

        // 1. 审批门禁（若启用则注册；无论是否启用，均绑定 PendingInteractions 副作用汇）
        {
            let inner_client = Arc::clone(&inner);
            let inner_rpc = Arc::clone(&inner);
            pending_interactions.attach_side_effects(ApprovalSideEffects {
                sessions: Some(Arc::clone(&runtime.sessions)),
                approval_snapshot: Some(Arc::clone(&approval_snapshot)),
                emit_client: Some(Arc::new(move |ev| inner_client.emit_client(ev))),
                emit_rpc: Some(Arc::new(move |notif| {
                    emit_interaction_rpc_with_turn_state(&inner_rpc, notif);
                })),
            });
        }
        if config.host_extensions.approval_gate {
            let inner_client = Arc::clone(&inner);
            let inner_rpc = Arc::clone(&inner);
            let gate_cfg = ApprovalGateConfig::with_callback(
                Arc::clone(&active_session_id),
                Arc::clone(&approval_snapshot),
                move |ev| inner_client.emit_client(ev),
            )
            .with_pending(Arc::clone(&pending_interactions))
            .with_rpc_callback(move |notif| {
                emit_interaction_rpc_with_turn_state(&inner_rpc, notif);
            });
            guards.push(install_approval_gate(&runtime, gate_cfg));
        }

        // 2. 文件 @mention 扩展
        if config.host_extensions.file_mentions {
            let inner_mentions = Arc::clone(&inner);
            guards.push(install_file_mentions(
                &runtime,
                FileMentionsConfig::with_callback(
                    config.workspace.clone(),
                    config.restrict_to_workspace,
                    move |ev| inner_mentions.emit_client(ev),
                ),
            ));
        }

        // 3. AgentEvent -> ActiveTurnBuffer + ClientEvent + RpcNotification 桥接
        {
            let inner_agent = Arc::clone(&inner);
            guards.push(runtime.agent_loop.on_event(move |ev| {
                handle_agent_event(&inner_agent, ev);
            }));
        }

        // 4. SessionStore::on_event -> RpcNotification::EventCommitted 桥接
        {
            let inner_events = Arc::clone(&inner);
            guards.push(runtime.sessions.on_event(move |payload| {
                if matches!(payload.entry.event, SessionEvent::Assistant { .. })
                    && let Some(buf) = inner_events
                        .active_turns
                        .lock()
                        .unwrap()
                        .get_mut(&payload.session.id)
                {
                    buf.clear_step_text();
                }
                let dto = session_entry_to_dto(&payload.session.id, &payload.entry);
                inner_events.emit_rpc(RpcNotification::EventCommitted(dto));
            }));
        }

        // 5. 会话标题更新 -> ClientEvent::SessionTitleUpdated
        {
            let inner_title = Arc::clone(&inner);
            guards.push(runtime.sessions.on_title_updated(move |ev| {
                if ev.session.id == inner_title.current_session_id() {
                    inner_title.emit_client(ClientEvent::SessionTitleUpdated(ev.title.clone()));
                }
            }));
        }

        // 6. 轨迹异常与上下文压缩事件 -> ClientEvent
        if let Some(traj) = &runtime.trajectory {
            let inner_traj = Arc::clone(&inner);
            guards.push(traj.on_anomaly(move |ev| {
                if ev.session_id == inner_traj.current_session_id() {
                    inner_traj.emit_client(ClientEvent::Anomaly(ev.finding.summary.clone()));
                }
            }));
        }
        if let Some(comp) = &runtime.compaction {
            {
                let inner_comp = Arc::clone(&inner);
                guards.push(comp.on_start(move |ev| {
                    if ev.session_id == inner_comp.current_session_id() {
                        inner_comp.emit_client(ClientEvent::CompactionStarted {
                            trigger: ev.trigger.as_str().to_string(),
                        });
                    }
                }));
            }
            {
                let inner_comp = Arc::clone(&inner);
                guards.push(comp.on_progress(move |ev| {
                    if ev.session_id == inner_comp.current_session_id() {
                        inner_comp.emit_client(ClientEvent::CompactionProgress {
                            stage: ev.stage.clone(),
                            delta_chars: ev.delta_chars,
                        });
                    }
                }));
            }
            {
                let inner_comp = Arc::clone(&inner);
                guards.push(comp.on_end(move |ev| {
                    if ev.session_id == inner_comp.current_session_id() {
                        inner_comp.emit_client(ClientEvent::CompactionFinished {
                            outcome: ev.outcome.as_str().to_string(),
                            tokens_before: Some(ev.tokens_before),
                            tokens_after: Some(ev.tokens_after),
                            first_kept_seq: ev.first_kept_seq,
                            summary_preview: ev.summary_preview.clone(),
                            message: ev.skipped_reason.clone(),
                            is_manual: ev.trigger == CompactionTrigger::Manual,
                        });
                    }
                }));
            }
        }

        // 7. Subagent 状态变更 -> ClientEvent::SubagentState + RpcNotification::SubagentUpdated
        if let Some(sub) = &runtime.subagent {
            let inner_sub = Arc::clone(&inner);
            let sub_ref = Arc::clone(sub);
            guards.push(sub.on_state(move |ev| {
                if ev.parent_session_id == inner_sub.current_session_id() {
                    inner_sub.emit_client(ClientEvent::SubagentState {
                        running_count: ev.running_count,
                        notice: ev.notice.clone(),
                    });
                }
                let task_dto = sub_ref
                    .list_subagents_for_parent(&ev.parent_session_id)
                    .into_iter()
                    .find(|snap| snap.child_session_id == ev.child_session_id)
                    .map(subagent_snapshot_to_dto)
                    .unwrap_or_else(|| SubagentSnapshotDto {
                        name: ev.name.clone(),
                        parent_session_id: Some(ev.parent_session_id.clone()),
                        child_session_id: ev.child_session_id.clone(),
                        mode: subagent_mode_to_dto(ev.mode),
                        prompt_preview: String::new(),
                        status: subagent_status_to_dto(ev.status),
                        steps: 0,
                        current_step: 0,
                        last_tool: None,
                        idle_secs: 0,
                        watchdog_alerts: 0,
                        elapsed_secs: 0,
                        summary: None,
                        output_path: None,
                    });
                inner_sub.emit_rpc(RpcNotification::SubagentUpdated(
                    SubagentUpdatedNotification {
                        session_id: ev.parent_session_id.clone(),
                        task: task_dto,
                        notice: ev.notice.clone(),
                    },
                ));
            }));
        }

        *inner.guards.lock().unwrap() = guards;

        // 8. FollowUp 唤醒与单真源队列自动出队循环
        let wake_handle = {
            let inner_wake = Arc::clone(&inner);
            let wake_rx = runtime.agent_loop.inbox().subscribe_wake();
            tokio::spawn(run_wake_loop(inner_wake, wake_rx, run_done_rx))
        };
        *inner.wake_task.lock().unwrap() = Some(wake_handle);

        SessionActorHandle { inner }
    }
}

/// 可克隆共享的 `SessionActor` 异步控制句柄（供 TUI、RPC Socket 服务与自动化测试共同调用）。
#[derive(Clone)]
pub struct SessionActorHandle {
    inner: Arc<ActorInner>,
}

impl SessionActorHandle {
    /// 获取当前活跃会话句柄。
    pub fn active_session(&self) -> Arc<Session> {
        self.inner.current_session()
    }

    /// 获取当前活跃会话 ID。
    pub fn active_session_id(&self) -> String {
        self.inner.current_session_id()
    }

    /// 返回共享挂起交互表句柄（供直连组件或测试检查挂起审批）。
    pub fn pending_interactions(&self) -> Arc<PendingInteractions> {
        Arc::clone(&self.inner.pending_interactions)
    }

    /// 挂接运行期进程元数据注册句柄（同步刷新当前 `session_id` 与模型配置）。
    pub fn attach_registration(&self, mut registration: ProcessRegistration) {
        let sid = self.inner.current_session_id();
        let cfg = self.inner.runtime_config.lock().unwrap().clone();
        let _ = registration.update_session(sid);
        let _ = registration.update_model(cfg.provider, cfg.model);
        *self.inner.registration.lock().unwrap() = Some(registration);
    }

    /// 订阅全量强类型 RPC 广播通知（涵盖 8 类 `RpcNotification`）。
    pub fn subscribe_notifications(&self) -> broadcast::Receiver<RpcNotification> {
        self.inner.notif_tx.subscribe()
    }

    /// 订阅同进程零序列化 `ClientEvent` 事件流（供 `mizar-tui` / Merak 消费）。
    pub fn subscribe_client_events(&self) -> broadcast::Receiver<ClientEvent> {
        self.inner.client_tx.subscribe()
    }

    /// 构造指定会话（支持活跃会话或子代理 `child_session_id`）的完整水合快照。
    pub async fn subscribe_snapshot(
        &self,
        session_id: &str,
        after_seq: Option<u64>,
    ) -> Result<SessionSubscribeSnapshot, ActorError> {
        self.subscribe_snapshot_with(SessionSubscribeParams {
            session_id: session_id.to_string(),
            after_seq,
            limit: None,
        })
        .await
    }

    /// 带分页条数上限的 `session.subscribe` 完整水合快照构造。
    pub async fn subscribe_snapshot_with(
        &self,
        params: SessionSubscribeParams,
    ) -> Result<SessionSubscribeSnapshot, ActorError> {
        self.ensure_alive()?;
        let session = self.resolve_readable_session(&params.session_id)?;
        let active_sid = self.inner.current_session_id();

        let state = self.inner.resolve_session_state(&params.session_id);
        let config = if params.session_id == active_sid {
            self.inner.runtime_config.lock().unwrap().to_dto()
        } else {
            let sel = session.latest_selection();
            let approval_mode = self.inner.runtime_config.lock().unwrap().approval_mode;
            SessionConfigDto {
                provider: sel.as_ref().and_then(|s| s.provider.clone()),
                model: sel.as_ref().and_then(|s| s.model.clone()),
                reasoning_effort: sel.and_then(|s| s.reasoning_effort),
                approval_mode: RpcApprovalMode::from(approval_mode),
            }
        };

        let context = compute_session_context_usage(
            &self.inner.runtime,
            &session,
            config.provider.as_deref(),
            config.model.as_deref(),
        );

        let limit = params.limit.unwrap_or(usize::MAX);
        let (raw_entries, _) = session.list_events_paged(None, params.after_seq, limit);
        let events = raw_entries
            .iter()
            .map(|entry| session_entry_to_dto(&session.id, entry))
            .collect();

        let active_turn = self
            .inner
            .active_turns
            .lock()
            .unwrap()
            .get(&params.session_id)
            .map(ActiveTurnBuffer::to_dto);

        let pending_interactions = self
            .inner
            .pending_interactions
            .list_pending_for_session(&params.session_id);

        let queued_inputs = self
            .inner
            .queued_inputs
            .lock()
            .unwrap()
            .list(&params.session_id);

        let subagents = self
            .inner
            .runtime
            .subagent
            .as_ref()
            .map(|sub| {
                sub.list_subagents_for_parent(&params.session_id)
                    .into_iter()
                    .map(subagent_snapshot_to_dto)
                    .collect()
            })
            .unwrap_or_default();

        Ok(SessionSubscribeSnapshot {
            session: session_to_meta_dto(&session),
            state,
            config,
            context,
            events,
            active_turn,
            pending_interactions,
            queued_inputs,
            subagents,
        })
    }

    /// 提交一轮用户输入或排队/步间插话（`send` / `queue` / `steer`）。
    pub async fn submit_turn(
        &self,
        params: TurnSubmitParams,
    ) -> Result<TurnSubmitResult, ActorError> {
        self.ensure_alive()?;
        let active_session = self.inner.current_session();
        if params.session_id != active_session.id {
            if self
                .inner
                .runtime
                .sessions
                .get(&params.session_id)
                .is_none()
            {
                return Err(ActorError::SessionNotFound(params.session_id));
            }
            return Err(ActorError::InvalidParams(format!(
                "会话 `{}` 非当前活跃会话，仅支持向 `{}` 提交输入",
                params.session_id, active_session.id
            )));
        }

        let raw_attachments = params.attachments.clone().unwrap_or_default();
        if params.text.trim().is_empty() && raw_attachments.is_empty() {
            return Err(ActorError::InvalidParams(
                "提交消息的 text 与 attachments 不能同时为空".to_string(),
            ));
        }

        let images = resolve_attachments(&raw_attachments).await?;
        let mode = params.effective_mode();

        match mode {
            SubmitMode::Send => {
                let mut input_slot = Some(TurnInput::user_with_images(
                    params.text.clone(),
                    images.clone(),
                ));
                if let Some(turn) = self.inner.try_start_turn_if_idle(
                    Arc::clone(&active_session),
                    || input_slot.take(),
                    None,
                    None,
                    None,
                ) {
                    Ok(TurnSubmitResult {
                        accepted: true,
                        mode: SubmitMode::Send,
                        queue_id: None,
                        turn: Some(turn),
                    })
                } else {
                    let (item, snapshot) = self.inner.queued_inputs.lock().unwrap().push(
                        &active_session.id,
                        params.text,
                        raw_attachments,
                        images,
                    );
                    self.inner.emit_queue_updated(&active_session.id, snapshot);
                    Ok(TurnSubmitResult {
                        accepted: true,
                        mode: SubmitMode::Queue,
                        queue_id: Some(item.id),
                        turn: None,
                    })
                }
            }
            SubmitMode::Queue => {
                let (item, snapshot) = self.inner.queued_inputs.lock().unwrap().push(
                    &active_session.id,
                    params.text,
                    raw_attachments,
                    images,
                );
                self.inner.emit_queue_updated(&active_session.id, snapshot);
                Ok(TurnSubmitResult {
                    accepted: true,
                    mode: SubmitMode::Queue,
                    queue_id: Some(item.id),
                    turn: None,
                })
            }
            SubmitMode::Steer => {
                let mut input_slot = Some(TurnInput::user_with_images(params.text, images));
                if let Some(turn) = self.inner.try_start_turn_if_idle(
                    Arc::clone(&active_session),
                    || input_slot.take(),
                    None,
                    None,
                    None,
                ) {
                    Ok(TurnSubmitResult {
                        accepted: true,
                        mode: SubmitMode::Send,
                        queue_id: None,
                        turn: Some(turn),
                    })
                } else {
                    let input = input_slot.expect("input not consumed while busy");
                    let push_res = self.inner.runtime.agent_loop.inbox().push(
                        &active_session.id,
                        DeliveryMode::Steer,
                        input,
                    );
                    let active_turn_num = self
                        .inner
                        .active_turns
                        .lock()
                        .unwrap()
                        .get(&active_session.id)
                        .map(|t| t.turn);
                    Ok(TurnSubmitResult {
                        accepted: true,
                        mode: SubmitMode::Steer,
                        queue_id: Some(push_res.id),
                        turn: active_turn_num,
                    })
                }
            }
        }
    }

    /// 取消当前正在运行的对话回合（默认来源记为 `ResolvedBy::Rpc`）。
    pub async fn cancel_turn(&self, session_id: &str) -> Result<TurnCancelResult, ActorError> {
        self.cancel_turn_by(session_id, ResolvedBy::Rpc).await
    }

    /// 指定取消来源端（TUI / RPC）执行回合硬取消，保留 `queued_inputs` 并将后台通知转为 `Inject`。
    pub async fn cancel_turn_by(
        &self,
        session_id: &str,
        resolved_by: ResolvedBy,
    ) -> Result<TurnCancelResult, ActorError> {
        self.ensure_alive()?;
        let _ = self.resolve_readable_session(session_id)?;

        self.inner.pending_interactions.resolve_all_for_session(
            session_id,
            ApprovalDecision::Deny,
            resolved_by,
        );

        let cancelled = {
            let mut ctrl = self.inner.run_controller.lock().unwrap();
            if ctrl
                .active_task()
                .is_some_and(|t| t.session_id == session_id)
            {
                ctrl.cancel_active();
                true
            } else {
                self.inner.runtime.agent_loop.is_session_busy(session_id)
            }
        };

        self.inner
            .runtime
            .agent_loop
            .inbox()
            .clear_user_items_and_defer_notices(session_id);

        Ok(TurnCancelResult { cancelled })
    }

    /// 从单真源输入队列中撤回指定 `id` 的待发送消息，并广播 `queue.updated`。
    pub async fn queue_remove(
        &self,
        session_id: &str,
        id: u64,
    ) -> Result<QueueRemoveResult, ActorError> {
        self.ensure_alive()?;
        let _ = self.resolve_readable_session(session_id)?;
        let (removed, snapshot) = self
            .inner
            .queued_inputs
            .lock()
            .unwrap()
            .remove(session_id, id);
        if removed.is_some() {
            self.inner.emit_queue_updated(session_id, snapshot);
        }
        Ok(QueueRemoveResult {
            removed: removed.is_some(),
            item: removed,
        })
    }

    /// 弹出当前队列全部待发送条目（常用于退回编辑框），并广播清空后的 `queue.updated`。
    pub async fn queue_pop_all(&self, session_id: &str) -> Result<QueuePopAllResult, ActorError> {
        self.ensure_alive()?;
        let _ = self.resolve_readable_session(session_id)?;
        let (items, snapshot) = self.inner.queued_inputs.lock().unwrap().pop_all(session_id);
        if !items.is_empty() {
            self.inner.emit_queue_updated(session_id, snapshot);
        }
        Ok(QueuePopAllResult { items })
    }

    /// 原子裁决一条挂起的审批交互请求（首个胜出方返回 `{ ok: true }`，后到竞争方返回 `{ ok: false, reason: "already_resolved" }`）。
    pub async fn resolve_interaction(
        &self,
        params: InteractionResolveParams,
        resolved_by: ResolvedBy,
    ) -> Result<InteractionResolveResult, ActorError> {
        self.ensure_alive()?;
        let decision = params.decision.to_approval_decision(params.note);
        match self.inner.pending_interactions.resolve_for_session(
            params.session_id.as_deref(),
            &params.request_id,
            decision,
            resolved_by,
        ) {
            Ok(_) => Ok(InteractionResolveResult::ok()),
            Err(ResolveInteractionError::AlreadyResolved(_)) => {
                Ok(InteractionResolveResult::already_resolved())
            }
            Err(ResolveInteractionError::SessionMismatch {
                expected_session_id,
                actual_session_id,
                ..
            }) => Ok(InteractionResolveResult::rejected(format!(
                "session_mismatch: expected {expected_session_id}, got {actual_session_id}"
            ))),
        }
    }

    /// 热更新当前会话模型 / 思考强度 / 审批模式，并同步写回 `Selection` 快照与进程注册元信息。
    pub async fn update_config(
        &self,
        params: SessionConfigParams,
    ) -> Result<SessionConfigResult, ActorError> {
        self.ensure_alive()?;
        let active_session = self.inner.current_session();
        if params.session_id != active_session.id {
            if self
                .inner
                .runtime
                .sessions
                .get(&params.session_id)
                .is_none()
            {
                return Err(ActorError::SessionNotFound(params.session_id));
            }
            return Err(ActorError::InvalidParams(format!(
                "会话 `{}` 非当前活跃会话，仅支持调整 `{}` 的运行配置",
                params.session_id, active_session.id
            )));
        }

        let cfg = {
            let mut guard = self.inner.runtime_config.lock().unwrap();
            if let Some(provider) = params.provider {
                guard.provider = normalize_non_empty(provider);
            }
            if let Some(model) = params.model {
                guard.model = normalize_non_empty(model);
            }
            if let Some(effort) = params.reasoning_effort {
                guard.reasoning_effort = Some(effort);
            }
            if let Some(mode) = params.approval_mode {
                let next_mode = ApprovalMode::from(mode);
                guard.approval_mode = next_mode;
                self.inner.approval_snapshot.write().unwrap().mode = next_mode;
            }
            guard.clone()
        };

        active_session.record_selection(
            cfg.provider.clone(),
            cfg.model.clone(),
            cfg.reasoning_effort,
        );

        if let Some(ref mut reg) = *self.inner.registration.lock().unwrap() {
            let _ = reg.update_model(cfg.provider.clone(), cfg.model.clone());
        }

        let context = compute_session_context_usage(
            &self.inner.runtime,
            &active_session,
            cfg.provider.as_deref(),
            cfg.model.as_deref(),
        );

        Ok(SessionConfigResult {
            config: cfg.to_dto(),
            context,
        })
    }

    /// 切换当前活跃会话（强制校验 `run_dir` 单写者会话锁并广播 `session.switched`）。
    pub async fn switch_session(&self, session_id: &str) -> Result<Arc<Session>, ActorError> {
        self.ensure_alive()?;
        let old_session = self.inner.current_session();
        if old_session.id == session_id {
            return Ok(old_session);
        }

        if let Some(ref run_dir) = self.inner.run_dir {
            check_session_writable(run_dir, session_id, std::process::id())?;
        }

        let next_session = self
            .inner
            .runtime
            .sessions
            .get(session_id)
            .ok_or_else(|| ActorError::SessionNotFound(session_id.to_string()))?;

        self.apply_session_transition(old_session, Arc::clone(&next_session))
            .await;
        Ok(next_session)
    }

    /// 创建全新会话并原子切换过去（广播 `session.switched`）。
    pub async fn new_session(&self, title: Option<String>) -> Result<Arc<Session>, ActorError> {
        self.ensure_alive()?;
        let old_session = self.inner.current_session();
        let next_session = self
            .inner
            .runtime
            .sessions
            .create_with(SessionCreateOptions {
                cwd: Some(self.inner.workspace.display().to_string()),
                title,
                origin: Some(SessionOrigin::User),
                ..Default::default()
            });

        self.apply_session_transition(old_session, Arc::clone(&next_session))
            .await;
        Ok(next_session)
    }

    /// 按工作区、关键字与键集游标轻量分页列出历史会话摘要。
    pub async fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> Result<SessionListResult, ActorError> {
        self.ensure_alive()?;
        let query_norm = params
            .query
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase);

        let parsed_cursor = if let Some(ref raw) = params.cursor {
            Some(parse_session_cursor(raw)?)
        } else {
            None
        };

        let headers = self
            .inner
            .runtime
            .sessions
            .list_headers_async(SessionListQuery {
                cwd: params.cwd,
                limit: None,
                exclude_subagents: params.exclude_subagents.unwrap_or(true),
            })
            .await?;

        let writer_pids = self.collect_live_writer_pids();

        let mut filtered: Vec<_> = headers
            .into_iter()
            .filter(|h| {
                if let Some(ref q) = query_norm {
                    let id_hit = h.id.to_ascii_lowercase().contains(q);
                    let title_hit = h
                        .title
                        .as_deref()
                        .is_some_and(|t| t.to_ascii_lowercase().contains(q));
                    let cwd_hit = h
                        .cwd
                        .as_deref()
                        .is_some_and(|c| c.to_ascii_lowercase().contains(q));
                    if !(id_hit || title_hit || cwd_hit) {
                        return false;
                    }
                }
                if let Some((cursor_ts, ref cursor_id)) = parsed_cursor {
                    return h.updated_at_ms < cursor_ts
                        || (h.updated_at_ms == cursor_ts && h.id.as_str() > cursor_id.as_str());
                }
                true
            })
            .collect();

        let limit = params.limit.unwrap_or(DEFAULT_SESSION_LIST_LIMIT);
        let next_cursor = if limit == 0 {
            filtered.clear();
            None
        } else if filtered.len() > limit {
            filtered.truncate(limit);
            filtered
                .last()
                .map(|last| format!("{}:{}", last.updated_at_ms, last.id))
        } else {
            None
        };

        let sessions = filtered
            .into_iter()
            .map(|h| {
                let active_writer_pid = writer_pids.get(&h.id).copied();
                SessionListItemDto {
                    id: h.id,
                    cwd: h.cwd,
                    title: h.title,
                    parent_id: h.parent_id,
                    fork_seq: h.fork_seq,
                    event_count: h.event_count,
                    created_at_ms: u64::try_from(h.created_at_ms).unwrap_or(u64::MAX),
                    updated_at_ms: u64::try_from(h.updated_at_ms).unwrap_or(u64::MAX),
                    active_writer_pid,
                }
            })
            .collect();

        Ok(SessionListResult {
            sessions,
            next_cursor,
        })
    }

    /// 双向游标分页查询指定会话的已落盘事件流（底层调用 `Session::list_events_paged`）。
    pub async fn list_events(
        &self,
        params: SessionEventsParams,
    ) -> Result<SessionEventsResult, ActorError> {
        self.ensure_alive()?;
        let session = self.resolve_readable_session(&params.session_id)?;
        let limit = params.limit.unwrap_or(DEFAULT_EVENTS_PAGE_LIMIT);
        let (entries, has_more) =
            session.list_events_paged(params.before_seq, params.after_seq, limit);
        let events = entries
            .iter()
            .map(|e| session_entry_to_dto(&session.id, e))
            .collect();
        Ok(SessionEventsResult {
            session_id: session.id.clone(),
            events,
            has_more,
        })
    }

    /// 基于指定会话与 `at_seq` 分叉出新会话并返回新会话元信息。
    pub async fn fork_session(
        &self,
        params: SessionForkParams,
    ) -> Result<SessionForkResult, ActorError> {
        self.ensure_alive()?;
        let _ = self.resolve_readable_session(&params.session_id)?;
        let forked = self.inner.runtime.sessions.fork(
            &params.session_id,
            params.at_seq,
            SessionCreateOptions {
                origin: Some(SessionOrigin::User),
                ..Default::default()
            },
        )?;
        Ok(SessionForkResult {
            session: session_to_meta_dto(&forked),
        })
    }

    /// 导出指定会话的 `/tree` 可分叉事件摘要树。
    pub async fn session_tree(&self, session_id: &str) -> Result<SessionTreeResult, ActorError> {
        self.ensure_alive()?;
        let session = self.resolve_readable_session(session_id)?;
        let entries = project_session_tree(&session);
        Ok(SessionTreeResult {
            session_id: session.id.clone(),
            entries,
        })
    }

    /// 列举当前运行时的可用斜杠命令、技能、工具、MCP 与模型能力目录。
    pub async fn list_capabilities(
        &self,
        session_id: Option<&str>,
    ) -> Result<CapabilitiesListResult, ActorError> {
        self.ensure_alive()?;
        let active_sid = self.inner.current_session_id();
        let target_sid = session_id.unwrap_or(&active_sid);
        if session_id.is_some() {
            let _ = self.resolve_readable_session(target_sid)?;
        }

        let commands = builtin_slash_commands();
        let skills = if let Some(reg) = self.inner.runtime.skills.as_ref() {
            reg.list(SkillLookupOptions {
                cwd: Some(self.inner.workspace.clone()),
                cancel: CancellationToken::new(),
            })
            .await
            .into_iter()
            .filter(|s| s.is_user_invocable())
            .map(|s| SkillPickerItem {
                name: s.name,
                description: s.description,
                source: s.source.as_str().to_string(),
            })
            .collect()
        } else {
            Vec::new()
        };

        let tools = collect_tools_summary(&self.inner.runtime, Some(target_sid));
        let mcps = collect_mcps_summary_async(&self.inner.runtime).await;
        let providers = self.inner.runtime.llm.provider_infos();
        let cfg = self.inner.runtime_config.lock().unwrap().clone();
        let max_context_tokens = resolve_max_context_tokens(
            &self.inner.runtime,
            &providers,
            cfg.provider.as_deref(),
            cfg.model.as_deref(),
        );

        Ok(CapabilitiesListResult {
            commands,
            skills,
            tools,
            mcps,
            providers,
            max_context_tokens: Some(max_context_tokens),
        })
    }

    /// 优雅关闭 `SessionActor`：终止唤醒循环、排空挂起审批、取消活跃运行并卸载钩子。
    pub async fn shutdown(&self) {
        if self.inner.is_shutdown.swap(true, Ordering::AcqRel) {
            return;
        }
        self.inner.shutdown_cancel.cancel();

        let old_sid = self.inner.current_session_id();
        self.inner
            .pending_interactions
            .resolve_all(ApprovalDecision::Deny, ResolvedBy::Tui);
        self.inner
            .run_controller
            .lock()
            .unwrap()
            .abort_for_session_switch();
        if let Some(sub) = self.inner.runtime.subagent.as_ref() {
            sub.cancel_parent_subagents(&old_sid);
        }
        self.inner.runtime.agent_loop.inbox().clear(&old_sid);

        let wake_handle = self.inner.wake_task.lock().unwrap().take();
        if let Some(handle) = wake_handle {
            let _ = handle.await;
        }

        let guards = std::mem::take(&mut *self.inner.guards.lock().unwrap());
        for guard in guards.into_iter().rev() {
            drop(guard);
        }

        let _reg = self.inner.registration.lock().unwrap().take();
    }

    fn ensure_alive(&self) -> Result<(), ActorError> {
        if self.inner.is_shutdown.load(Ordering::Acquire) {
            Err(ActorError::Shutdown)
        } else {
            Ok(())
        }
    }

    fn resolve_readable_session(&self, session_id: &str) -> Result<Arc<Session>, ActorError> {
        let active = self.inner.current_session();
        if active.id == session_id {
            return Ok(active);
        }
        self.inner
            .runtime
            .sessions
            .get(session_id)
            .ok_or_else(|| ActorError::SessionNotFound(session_id.to_string()))
    }

    fn collect_live_writer_pids(&self) -> HashMap<String, u32> {
        let Some(ref run_dir) = self.inner.run_dir else {
            return HashMap::new();
        };
        let Ok(live) = scan_live_processes(run_dir) else {
            return HashMap::new();
        };
        live.into_iter()
            .map(|meta| (meta.session_id, meta.pid))
            .collect()
    }

    async fn apply_session_transition(
        &self,
        old_session: Arc<Session>,
        next_session: Arc<Session>,
    ) {
        let old_id = old_session.id.clone();
        let new_id = next_session.id.clone();

        self.inner.pending_interactions.resolve_all_for_session(
            &old_id,
            ApprovalDecision::Deny,
            ResolvedBy::Tui,
        );
        self.inner
            .run_controller
            .lock()
            .unwrap()
            .abort_for_session_switch();
        if let Some(sub) = self.inner.runtime.subagent.as_ref() {
            sub.cancel_parent_subagents(&old_id);
        }
        self.inner.runtime.agent_loop.inbox().clear(&old_id);
        self.inner.active_turns.lock().unwrap().remove(&old_id);

        *self.inner.active_session.lock().unwrap() = Arc::clone(&next_session);
        *self.inner.active_session_id.write().unwrap() = new_id.clone();

        if let Some(ref mut reg) = *self.inner.registration.lock().unwrap() {
            let _ = reg.update_session(&new_id);
        }

        let _ = old_session.flush_async().await;

        self.inner.emit_client(ClientEvent::SessionSwitched {
            old_session_id: old_id.clone(),
            new_session_id: new_id.clone(),
        });
        self.inner.emit_rpc(RpcNotification::SessionSwitched(
            SessionSwitchedNotification {
                old_session_id: old_id,
                new_session_id: new_id,
            },
        ));
    }
}

async fn run_wake_loop(
    inner: Arc<ActorInner>,
    mut wake_rx: broadcast::Receiver<String>,
    mut run_done_rx: mpsc::UnboundedReceiver<RunDoneSignal>,
) {
    loop {
        tokio::select! {
            _ = inner.shutdown_cancel.cancelled() => {
                break;
            }
            wake_res = wake_rx.recv() => {
                match wake_res {
                    Ok(woken_sid) => {
                        if woken_sid == inner.current_session_id() {
                            try_dispatch_followup_or_queue(&inner, &woken_sid, false);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let sid = inner.current_session_id();
                        try_dispatch_followup_or_queue(&inner, &sid, false);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // 即使广播关闭也继续服务 run_done_rx，直到显式 shutdown。
                        std::future::pending::<()>().await;
                    }
                }
            }
            maybe_done = run_done_rx.recv() => {
                let Some(done) = maybe_done else {
                    break;
                };
                if !done.user_cancelled && done.session_id == inner.current_session_id() {
                    try_dispatch_followup_or_queue(&inner, &done.session_id, done.allow_queue_drain);
                }
            }
        }
    }
}

fn try_dispatch_followup_or_queue(inner: &Arc<ActorInner>, sid: &str, allow_queue_drain: bool) {
    if inner.is_shutdown.load(Ordering::Acquire) {
        return;
    }
    let session = inner.current_session();
    if session.id != sid {
        return;
    }
    let _ = inner.try_start_turn_if_idle(
        session,
        || {
            if let Some(next_input) = inner.runtime.agent_loop.inbox().pop_followup(sid) {
                return Some(next_input);
            }
            if allow_queue_drain
                && let Some((entry, snapshot)) = inner.queued_inputs.lock().unwrap().pop_front(sid)
            {
                inner.emit_queue_updated(sid, snapshot);
                return Some(TurnInput::user_with_images(entry.dto.text, entry.images));
            }
            None
        },
        None,
        None,
        None,
    );
}

fn emit_interaction_rpc_with_turn_state(inner: &Arc<ActorInner>, notif: RpcNotification) {
    match &notif {
        RpcNotification::InteractionRequested(dto) => {
            if let Some(ref sid) = dto.session_id {
                let turn_step = inner
                    .active_turns
                    .lock()
                    .unwrap()
                    .get(sid)
                    .map(|b| (b.turn, b.step));
                if let Some((turn, step)) = turn_step {
                    inner.emit_rpc(RpcNotification::TurnState(TurnStateNotification {
                        session_id: sid.clone(),
                        turn,
                        step,
                        state: SessionRuntimeState::WaitingApproval,
                    }));
                }
            }
            inner.emit_rpc(notif);
        }
        RpcNotification::InteractionResolved(resolved) => {
            let sid_opt = resolved.session_id.clone();
            inner.emit_rpc(notif);
            if let Some(sid) = sid_opt
                && !inner.pending_interactions.has_pending_for_session(&sid)
            {
                let turn_step = inner
                    .active_turns
                    .lock()
                    .unwrap()
                    .get(&sid)
                    .map(|b| (b.turn, b.step));
                if let Some((turn, step)) = turn_step {
                    inner.emit_rpc(RpcNotification::TurnState(TurnStateNotification {
                        session_id: sid,
                        turn,
                        step,
                        state: SessionRuntimeState::Streaming,
                    }));
                }
            }
        }
        _ => inner.emit_rpc(notif),
    }
}

fn handle_agent_event(inner: &Arc<ActorInner>, ev: &AgentEvent) {
    let active_sid = inner.current_session_id();
    match ev {
        AgentEvent::Status(ev) => {
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::Status(to_ui_status(ev.status)));
            }
        }
        AgentEvent::TurnStart(ev) => {
            {
                let mut turns = inner.active_turns.lock().unwrap();
                turns.insert(ev.session.id.clone(), ActiveTurnBuffer::new(ev.turn));
            }
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::TurnStart { turn: ev.turn });
            }
            inner.emit_rpc(RpcNotification::TurnState(TurnStateNotification {
                session_id: ev.session.id.clone(),
                turn: ev.turn,
                step: 0,
                state: SessionRuntimeState::Streaming,
            }));
        }
        AgentEvent::StepStart(ev) => {
            {
                let mut turns = inner.active_turns.lock().unwrap();
                let buf = turns
                    .entry(ev.session.id.clone())
                    .or_insert_with(|| ActiveTurnBuffer::new(ev.turn));
                buf.begin_step(ev.step);
            }
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::StepStart { ts_ms: ev.ts_ms });
            }
            let state = if inner
                .pending_interactions
                .has_pending_for_session(&ev.session.id)
            {
                SessionRuntimeState::WaitingApproval
            } else {
                SessionRuntimeState::Streaming
            };
            inner.emit_rpc(RpcNotification::TurnState(TurnStateNotification {
                session_id: ev.session.id.clone(),
                turn: ev.turn,
                step: ev.step,
                state,
            }));
        }
        AgentEvent::StepFirstToken(ev) => {
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::FirstToken { ts_ms: ev.ts_ms });
            }
        }
        AgentEvent::StepEnd(ev) => {
            if let Some(buf) = inner.active_turns.lock().unwrap().get_mut(&ev.session.id) {
                buf.clear_step_text();
            }
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::StepEnd {
                    step: ev.step,
                    ts_ms: ev.ts_ms,
                    outcome: to_ui_step_outcome(&ev.outcome),
                    usage: ev.usage,
                });
            }
        }
        AgentEvent::ReasoningDelta(ev) => {
            let (turn, step) = {
                let mut turns = inner.active_turns.lock().unwrap();
                let buf = turns
                    .entry(ev.session.id.clone())
                    .or_insert_with(|| ActiveTurnBuffer::new(peek_current_turn(&ev.session)));
                buf.push_reasoning(&ev.text);
                (buf.turn, buf.step)
            };
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::ReasoningDelta {
                    text: ev.text.clone(),
                    ts_ms: ev.ts_ms,
                });
            }
            inner.emit_rpc(RpcNotification::StreamDelta(StreamDeltaNotification {
                session_id: ev.session.id.clone(),
                turn,
                step,
                kind: StreamDeltaKind::Reasoning,
                call_id: None,
                delta: ev.text.clone(),
            }));
        }
        AgentEvent::TextDelta(ev) => {
            let (turn, step) = {
                let mut turns = inner.active_turns.lock().unwrap();
                let buf = turns
                    .entry(ev.session.id.clone())
                    .or_insert_with(|| ActiveTurnBuffer::new(peek_current_turn(&ev.session)));
                buf.push_text(&ev.text);
                (buf.turn, buf.step)
            };
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::TextDelta {
                    text: ev.text.clone(),
                    ts_ms: ev.ts_ms,
                });
            }
            inner.emit_rpc(RpcNotification::StreamDelta(StreamDeltaNotification {
                session_id: ev.session.id.clone(),
                turn,
                step,
                kind: StreamDeltaKind::Text,
                call_id: None,
                delta: ev.text.clone(),
            }));
        }
        AgentEvent::ToolCall(ev) => {
            let (turn, step) = {
                let mut turns = inner.active_turns.lock().unwrap();
                let buf = turns
                    .entry(ev.session.id.clone())
                    .or_insert_with(|| ActiveTurnBuffer::new(peek_current_turn(&ev.session)));
                buf.start_tool(
                    ev.call.id.clone(),
                    ev.call.name.clone(),
                    ev.call.arguments.clone(),
                );
                (buf.turn, buf.step)
            };
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::ToolCall(ev.call.clone()));
            }
            inner.emit_rpc(RpcNotification::StreamDelta(StreamDeltaNotification {
                session_id: ev.session.id.clone(),
                turn,
                step,
                kind: StreamDeltaKind::ToolArgs,
                call_id: Some(ev.call.id.clone()),
                delta: ev.call.arguments.clone(),
            }));
        }
        AgentEvent::ToolUpdate(ev) => {
            let (turn, step) = {
                let mut turns = inner.active_turns.lock().unwrap();
                if let Some(buf) = turns.get_mut(&ev.session.id) {
                    buf.update_tool_output(&ev.tool_call_id, &ev.preview);
                    (buf.turn, buf.step)
                } else {
                    (peek_current_turn(&ev.session), 0)
                }
            };
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::ToolUpdate {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    preview: ev.preview.clone(),
                });
            }
            inner.emit_rpc(RpcNotification::StreamDelta(StreamDeltaNotification {
                session_id: ev.session.id.clone(),
                turn,
                step,
                kind: StreamDeltaKind::ToolOutput,
                call_id: Some(ev.tool_call_id.clone()),
                delta: ev.preview.clone(),
            }));
        }
        AgentEvent::ToolResult(ev) => {
            if let Some(buf) = inner.active_turns.lock().unwrap().get_mut(&ev.session.id) {
                buf.finish_tool(&ev.tool_call_id);
            }
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::ToolResult {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    content: ev.content.clone(),
                    is_error: ev.is_error,
                    blocked: ev.blocked,
                    details: ev.details.clone(),
                });
            }
        }
        AgentEvent::Usage(ev) => {
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::Usage(ev.usage));
            }
        }
        AgentEvent::TurnEnd(ev) => {
            let last_step = inner
                .active_turns
                .lock()
                .unwrap()
                .remove(&ev.session.id)
                .map_or(0, |b| b.step);
            if ev.session.id == active_sid {
                inner.emit_client(ClientEvent::TurnEnd {
                    reason: to_ui_turn_end_reason(ev.reason.clone()),
                });
            }
            let state = if inner
                .pending_interactions
                .has_pending_for_session(&ev.session.id)
            {
                SessionRuntimeState::WaitingApproval
            } else {
                SessionRuntimeState::Idle
            };
            inner.emit_rpc(RpcNotification::TurnState(TurnStateNotification {
                session_id: ev.session.id.clone(),
                turn: ev.turn,
                step: last_step,
                state,
            }));
        }
        AgentEvent::Request(_) | AgentEvent::TurnInput(_) => {}
    }
}

fn peek_next_turn(session: &Session) -> u64 {
    let max_turn = session.with_all_entries(|entries| {
        entries
            .iter()
            .filter_map(|e| match &e.event {
                SessionEvent::Trace {
                    detail:
                        SessionTraceEvent::TurnStart { turn }
                        | SessionTraceEvent::TurnEnd { turn, .. },
                } => Some(*turn),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    });
    max_turn.saturating_add(1)
}

fn peek_current_turn(session: &Session) -> u64 {
    session.with_all_entries(|entries| {
        entries
            .iter()
            .rev()
            .find_map(|e| match &e.event {
                SessionEvent::Trace {
                    detail: SessionTraceEvent::TurnStart { turn },
                } => Some(*turn),
                _ => None,
            })
            .unwrap_or(1)
    })
}

fn session_entry_to_dto(session_id: &str, entry: &SessionEntry) -> CommittedEventDto {
    let visibility = match entry.visibility() {
        EntryVisibility::Surface => RpcEventVisibility::Surface,
        EntryVisibility::LogOnly => RpcEventVisibility::LogOnly,
    };
    CommittedEventDto {
        session_id: session_id.to_string(),
        seq: entry.seq,
        ts_ms: u64::try_from(entry.ts).unwrap_or(u64::MAX),
        visibility,
        kind: entry.event.kind_name().to_string(),
        payload: serde_json::to_value(&entry.event).unwrap_or(serde_json::Value::Null),
    }
}

fn session_to_meta_dto(session: &Session) -> SessionMetaDto {
    let header = session.header();
    SessionMetaDto {
        id: header.id,
        cwd: header.cwd,
        title: header.title,
        parent_id: header.parent_id,
        fork_seq: header.fork_seq,
    }
}

fn subagent_mode_to_dto(mode: SubagentMode) -> RpcSubagentMode {
    match mode {
        SubagentMode::ReadOnly => RpcSubagentMode::ReadOnly,
        SubagentMode::Edit => RpcSubagentMode::Edit,
    }
}

fn subagent_status_to_dto(status: SubagentRunStatus) -> RpcSubagentStatus {
    match status {
        SubagentRunStatus::Running => RpcSubagentStatus::Running,
        SubagentRunStatus::Completed => RpcSubagentStatus::Completed,
        SubagentRunStatus::MaxSteps => RpcSubagentStatus::MaxSteps,
        SubagentRunStatus::Truncated => RpcSubagentStatus::Truncated,
        SubagentRunStatus::Failed => RpcSubagentStatus::Failed,
        SubagentRunStatus::TimedOut => RpcSubagentStatus::TimedOut,
        SubagentRunStatus::Stopped => RpcSubagentStatus::Stopped,
    }
}

/// 将领域层的 `SubagentTaskSnapshot` 转换为纯协议 DTO `SubagentSnapshotDto`。
pub fn subagent_snapshot_to_dto(snap: SubagentTaskSnapshot) -> SubagentSnapshotDto {
    SubagentSnapshotDto {
        name: snap.name,
        parent_session_id: Some(snap.parent_session_id),
        child_session_id: snap.child_session_id,
        mode: subagent_mode_to_dto(snap.mode),
        prompt_preview: snap.prompt_preview,
        status: subagent_status_to_dto(snap.status),
        steps: snap.steps,
        current_step: snap.current_step,
        last_tool: snap.last_tool,
        idle_secs: snap.idle_secs,
        watchdog_alerts: snap.watchdog_alerts,
        elapsed_secs: snap.elapsed_secs,
        summary: snap.summary,
        output_path: snap.output_path,
    }
}

fn resolve_max_context_tokens(
    runtime: &MizarRuntime,
    providers: &[mizar_llm::ProviderInfo],
    provider: Option<&str>,
    model: Option<&str>,
) -> u64 {
    if let Some(model_id) = model {
        let matched = if let Some(prov_id) = provider {
            providers
                .iter()
                .find(|p| p.id == prov_id)
                .and_then(|p| p.models.iter().find(|m| m.id == model_id))
        } else {
            providers
                .iter()
                .flat_map(|p| p.models.iter())
                .find(|m| m.id == model_id)
        };
        if let Some(cw) = matched.and_then(|m| m.context_window) {
            return cw;
        }
    }
    runtime
        .compaction
        .as_ref()
        .map_or(DEFAULT_CONTEXT_WINDOW, |c| {
            c.config().default_context_window
        })
}

fn compute_session_context_usage(
    runtime: &MizarRuntime,
    session: &Session,
    provider: Option<&str>,
    model: Option<&str>,
) -> SessionContextUsageDto {
    let used_tokens = session.with_visible_entries(|entries| {
        for entry in entries.iter().rev() {
            match &entry.event {
                SessionEvent::Assistant { usage: Some(u), .. } => {
                    let prompt = u.prompt_tokens();
                    if prompt > 0 {
                        return prompt;
                    }
                }
                SessionEvent::Compaction {
                    tokens_after: Some(t),
                    ..
                } if *t > 0 => {
                    return *t;
                }
                _ => {}
            }
        }
        0
    });

    let providers = runtime.llm.provider_infos();
    let max_tokens = resolve_max_context_tokens(runtime, &providers, provider, model);
    SessionContextUsageDto {
        used_tokens,
        max_tokens: Some(max_tokens),
    }
}

async fn collect_mcps_summary_async(runtime: &MizarRuntime) -> Vec<McpSummaryRow> {
    if !runtime.modules.is_enabled("mcp") {
        return Vec::new();
    }
    let mgr: Option<Arc<McpManager>> = runtime.mcp().await;

    let Some(mgr) = mgr else {
        return Vec::new();
    };
    mgr.statuses()
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
        .collect()
}

fn normalize_non_empty(raw: String) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn parse_session_cursor(raw: &str) -> Result<(u128, String), ActorError> {
    let Some((ts_str, id)) = raw.split_once(':') else {
        return Err(ActorError::InvalidParams(format!(
            "无效的会话分页游标 `{raw}`"
        )));
    };
    let ts = ts_str
        .parse::<u128>()
        .map_err(|_| ActorError::InvalidParams(format!("无效的游标时间戳 `{ts_str}`")))?;
    if id.is_empty() {
        return Err(ActorError::InvalidParams(format!(
            "无效的会话分页游标 `{raw}`"
        )));
    }
    Ok((ts, id.to_string()))
}

async fn resolve_attachments(
    attachments: &[RpcAttachmentDto],
) -> Result<Vec<ImageAttachment>, ActorError> {
    let mut images = Vec::with_capacity(attachments.len());
    for att in attachments {
        if let Some(ref data) = att.data {
            let cleaned = data.trim();
            if cleaned.is_empty() {
                return Err(ActorError::Attachment("图片附件 base64 数据为空".into()));
            }
            let bytes = BASE64_STANDARD
                .decode(cleaned)
                .map_err(|e| ActorError::Attachment(format!("图片 base64 解码失败: {e}")))?;
            match process_image_bytes(&bytes) {
                Ok(processed) => images.push(processed.attachment),
                Err(
                    ImageProcessError::UnsupportedFormat
                    | ImageProcessError::CorruptHeader(_)
                    | ImageProcessError::DecodeFailed(_),
                ) if att.mime.as_deref().is_some_and(|m| m.starts_with("image/")) => {
                    images.push(ImageAttachment::new(
                        att.mime.clone().unwrap(),
                        cleaned.to_string(),
                    ));
                }
                Err(err) => return Err(ActorError::Attachment(err.to_string())),
            }
        } else if let Some(ref path) = att.path {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| ActorError::Attachment(format!("读取附件 `{path}` 失败: {e}")))?;
            let processed = tokio::task::spawn_blocking(move || process_image_bytes(&bytes))
                .await
                .map_err(|e| ActorError::Attachment(format!("图片处理任务异常: {e}")))?
                .map_err(|e| ActorError::Attachment(e.to_string()))?;
            images.push(processed.attachment);
        } else {
            return Err(ActorError::Attachment(
                "附件必须提供 data 或 path 字段".to_string(),
            ));
        }
    }
    Ok(images)
}

fn builtin_slash_commands() -> Vec<SlashCommandDto> {
    vec![
        ("help", "打开快捷键与命令帮助面板", false),
        ("resume", "恢复历史会话（打开会话选择器或指定 ID）", true),
        ("rename", "重命名当前会话标题（留空由模型重新生成）", true),
        (
            "settings",
            "打开交互式配置面板（实时生效并自动保存）",
            false,
        ),
        ("model", "交互切换模型 / Provider (Ctrl+L)", true),
        ("effort", "调整思考强度 (Tab 切换 / off..max)", true),
        ("compact", "手动压缩当前会话上下文（可附聚焦说明）", true),
        (
            "session",
            "查看当前会话 ID、存储、事件构成与压缩边界",
            false,
        ),
        ("theme", "切换 Oklab / 自适应主题", true),
        ("trace", "打开全屏三泳道轨迹分析 (Ctrl+T)", false),
        ("panel", "展开/折叠右侧分栏面板 (Ctrl+B)", false),
        ("approval", "切换危险工具交互审批模式 (auto / ask)", true),
        ("thinking", "显示/隐藏推理过程 (Ctrl+O)", false),
        ("tools", "查看已声明工具", false),
        ("mcp", "查看 MCP 服务连接状态", false),
        ("skills", "打开交互式技能选择器 / 列表", false),
        (
            "instructions",
            "查看已加载的 AGENTS.md / CLAUDE.md 指令",
            false,
        ),
        ("tree", "查看会话时间线与分叉回溯树", false),
        ("view", "切换双模视图 (fullscreen / inline)", true),
        ("mouse", "切换鼠标追踪 / 放行原生划词 (F2)", false),
        ("stats", "查看会话统计摘要", false),
        ("export", "导出会话轨迹 JSONL", true),
        ("clear", "开启全新会话 (/new)", false),
        ("exit", "退出终端界面", false),
    ]
    .into_iter()
    .map(|(cmd, desc, takes_arg)| SlashCommandDto {
        command: cmd.to_string(),
        description: desc.to_string(),
        takes_arg,
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::AtomicU64;
    use std::time::Duration;

    use futures::channel::mpsc as futures_mpsc;
    use mizar_core::spine::AgentSpineConfig;
    use mizar_core::subagent::SubagentRunStatus;
    use mizar_llm::{
        BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmError, ModelInfo,
        ProviderInfo, StreamChunk, TokenUsage, ToolCall,
    };
    use mizar_session::{
        SessionEvent, SessionPersistence, SessionTraceEvent, SqliteSessionPersistence,
    };
    use mizar_tools::{ToolAnnotations, ToolDefinition, ToolOutput};
    use mizar_types::BoxError;
    use mizar_types::rpc::{ERR_SESSION_LOCKED, RpcApprovalDecision};
    use tokio::sync::Notify;

    use super::*;
    use crate::run_dir::{ProcessMeta, meta_path_for_pid};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

    struct TempDirGuard(PathBuf);
    impl TempDirGuard {
        fn new(tag: &str) -> Self {
            let seq = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "mizar_actor_{tag}_{}_{}",
                std::process::id(),
                seq
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &PathBuf {
            &self.0
        }
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct ChildGuard(Option<Child>);
    impl ChildGuard {
        fn spawn_sleeper() -> Self {
            let child = Command::new("sleep")
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn sleep helper");
            Self(Some(child))
        }

        fn pid(&self) -> u32 {
            self.0.as_ref().unwrap().id()
        }
    }
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    enum ScriptStep {
        /// 先发送前置 chunk，再等待 gate，再发送后续 chunk
        Gated {
            before: Vec<StreamChunk>,
            gate: Arc<Notify>,
            entered: Arc<Notify>,
            after: Vec<StreamChunk>,
        },
        Immediate(Vec<StreamChunk>),
        EchoUser {
            prefix: String,
        },
    }

    struct ScriptedAdapter {
        steps: Mutex<VecDeque<ScriptStep>>,
    }

    impl ScriptedAdapter {
        fn new(steps: Vec<ScriptStep>) -> Arc<Self> {
            Arc::new(Self {
                steps: Mutex::new(VecDeque::from(steps)),
            })
        }
    }

    impl LlmAdapter for ScriptedAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "scripted".into(),
                name: "Scripted".into(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: "script-v1".into(),
                    name: "Script V1".into(),
                    context_window: Some(64_000),
                    max_tokens: Some(4096),
                    reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
                    input: Vec::new(),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let step =
                self.steps
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or_else(|| ScriptStep::EchoUser {
                        prefix: "echo:".into(),
                    });
            let (tx, rx) = futures_mpsc::unbounded();
            tokio::spawn(async move {
                match step {
                    ScriptStep::Gated {
                        before,
                        gate,
                        entered,
                        after,
                    } => {
                        for chunk in before {
                            let _ = tx.unbounded_send(Ok(chunk));
                        }
                        entered.notify_waiters();
                        tokio::select! {
                            _ = gate.notified() => {}
                            _ = options.cancel.cancelled() => {
                                let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                                return;
                            }
                        }
                        for chunk in after {
                            let _ = tx.unbounded_send(Ok(chunk));
                        }
                    }
                    ScriptStep::Immediate(chunks) => {
                        for chunk in chunks {
                            if options.cancel.is_cancelled() {
                                let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                                return;
                            }
                            let _ = tx.unbounded_send(Ok(chunk));
                        }
                    }
                    ScriptStep::EchoUser { prefix } => {
                        let last_text = options
                            .messages
                            .last()
                            .map(|m| m.text_content())
                            .unwrap_or_default();
                        let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                            index: 0,
                            text: format!("{prefix}{last_text}"),
                        }));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                }
            });
            Box::pin(rx)
        }
    }

    async fn next_rpc_matching<F>(
        rx: &mut broadcast::Receiver<RpcNotification>,
        mut pred: F,
    ) -> RpcNotification
    where
        F: FnMut(&RpcNotification) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match rx.recv().await {
                    Ok(notif) if pred(&notif) => return notif,
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(e) => panic!("notification channel closed: {e}"),
                }
            }
        })
        .await
        .expect("timed out waiting for matching RpcNotification")
    }

    async fn next_client_matching<F>(
        rx: &mut broadcast::Receiver<ClientEvent>,
        mut pred: F,
    ) -> ClientEvent
    where
        F: FnMut(&ClientEvent) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match rx.recv().await {
                    Ok(ev) if pred(&ev) => return ev,
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(e) => panic!("client event channel closed: {e}"),
                }
            }
        })
        .await
        .expect("timed out waiting for matching ClientEvent")
    }

    #[test]
    fn active_turn_buffer_caps_output_tail_at_64kib_utf8_safe() {
        let mut buf = ActiveTurnBuffer::new(1);
        buf.begin_step(1);
        buf.start_tool("c1", "bash", "{}");

        // 构造含多字节 UTF-8（中文 3B + emoji 4B）超过 64 KiB 的输出
        let pattern = "猫娘日志🚀";
        let mut huge = pattern.repeat((ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES / pattern.len()) + 50);
        huge.push_str("TAIL_OK");
        assert!(huge.len() > ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES);

        buf.update_tool_output("c1", &huge);
        let tail1 = &buf.running_tools[0].output_tail;
        assert!(tail1.len() <= ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES);
        assert!(std::str::from_utf8(tail1.as_bytes()).is_ok());
        assert!(tail1.ends_with("TAIL_OK"));

        // 通过 append_tool_output 继续追加超过 64 KiB 也同样安全截断在尾部
        buf.append_tool_output("c1", &huge);
        let tail2 = &buf.running_tools[0].output_tail;
        assert!(tail2.len() <= ACTIVE_TURN_OUTPUT_TAIL_MAX_BYTES);
        assert!(std::str::from_utf8(tail2.as_bytes()).is_ok());
        assert!(tail2.ends_with("TAIL_OK"));
    }

    #[tokio::test]
    async fn session_actor_turn_execution_midstream_snapshot_and_subagent_snapshot() {
        let ws = TempDirGuard::new("midstream");
        let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
            .await
            .unwrap();

        let step1_entered = Arc::new(Notify::new());
        let step1_gate = Arc::new(Notify::new());
        let tool_entered = Arc::new(Notify::new());
        let tool_gate = Arc::new(Notify::new());

        let tool_entered_clone = Arc::clone(&tool_entered);
        let tool_gate_clone = Arc::clone(&tool_gate);
        let _slow_guard = rt
            .tools
            .register(
                ToolDefinition::async_rich_tool(
                    "slow_inspect",
                    "slow inspection tool",
                    json!({ "type": "object", "properties": {} }),
                    move |_args, ctx| {
                        let entered = Arc::clone(&tool_entered_clone);
                        let gate = Arc::clone(&tool_gate_clone);
                        async move {
                            ctx.emit_update("streaming tool tail line 1\nline 2", None);
                            entered.notify_waiters();
                            gate.notified().await;
                            Ok::<_, BoxError>(ToolOutput::text("slow_done"))
                        }
                    },
                )
                .with_annotations(ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                }),
            )
            .unwrap();

        let child_entered = Arc::new(Notify::new());
        let child_gate = Arc::new(Notify::new());

        let adapter = ScriptedAdapter::new(vec![
            // 1. 父会话 Step 1：先阻塞在流式 reasoning + text 中，放行后调 slow_inspect
            ScriptStep::Gated {
                before: vec![
                    StreamChunk::ReasoningDelta {
                        index: 0,
                        text: "思考中...".into(),
                    },
                    StreamChunk::TextDelta {
                        index: 1,
                        text: "前半段回复".into(),
                    },
                ],
                gate: Arc::clone(&step1_gate),
                entered: Arc::clone(&step1_entered),
                after: vec![
                    StreamChunk::tool_call(
                        2,
                        ToolCall {
                            id: "call_slow_1".into(),
                            name: "slow_inspect".into(),
                            arguments: r#"{"target":"all"}"#.into(),
                        },
                    ),
                    StreamChunk::Usage {
                        usage: TokenUsage {
                            input_tokens: 120,
                            output_tokens: 30,
                            ..Default::default()
                        },
                    },
                    StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    },
                ],
            },
            // 2. 父会话 Step 2：发起后台 subagent(spawn)
            ScriptStep::Immediate(vec![
                StreamChunk::tool_call(
                    0,
                    ToolCall {
                        id: "call_sub_1".into(),
                        name: "subagent".into(),
                        arguments: json!({
                            "action": "spawn",
                            "name": "worker-a",
                            "mode": "read_only",
                            "prompt": "check repo",
                        })
                        .to_string(),
                    },
                ),
                StreamChunk::Finish {
                    reason: FinishReason::ToolCalls,
                    replay: None,
                },
            ]),
            // 3. 父会话 Step 3：先直接收尾（此时后台子代理同时运行）
            ScriptStep::Immediate(vec![
                StreamChunk::TextDelta {
                    index: 0,
                    text: "已派发子代理".into(),
                },
                StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                },
            ]),
            // 4. 子代理自己的 Turn：停顿中供 subscribe_snapshot(child_session_id) 检查！
            ScriptStep::Gated {
                before: vec![StreamChunk::TextDelta {
                    index: 0,
                    text: "child midstream".into(),
                }],
                gate: Arc::clone(&child_gate),
                entered: Arc::clone(&child_entered),
                after: vec![StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }],
            },
            // 5. 子代理结算唤醒父会话后，父会话的自动汇总回应
            ScriptStep::Immediate(vec![
                StreamChunk::TextDelta {
                    index: 0,
                    text: "汇总完成".into(),
                },
                StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                },
            ]),
        ]);
        rt.register_adapter("scripted", adapter).unwrap();

        let sess = rt.sessions.create(Some("sess-main".into()));
        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&sess),
            SessionActorConfig::new(ws.path()).with_provider_model(
                Some("scripted".into()),
                Some("script-v1".into()),
                None,
            ),
        );
        let mut rpc_rx = actor.subscribe_notifications();
        let mut client_rx = actor.subscribe_client_events();

        let submit = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "run inspection".into(),
                attachments: None,
                mode: Some(SubmitMode::Send),
            })
            .await
            .unwrap();
        assert!(submit.accepted);
        assert_eq!(submit.turn, Some(1));

        // 等 Step 1 流式产出停顿，验证 mid-stream subscribe_snapshot
        step1_entered.notified().await;
        // 给同步监听器广播留 1 tick 调度
        tokio::task::yield_now().await;
        let mid_snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(mid_snap.state, SessionRuntimeState::Streaming);
        let active = mid_snap.active_turn.expect("active_turn present midstream");
        assert_eq!(active.turn, 1);
        assert_eq!(active.step, 1);
        assert_eq!(active.reasoning, "思考中...");
        assert_eq!(active.text, "前半段回复");
        assert!(active.running_tools.is_empty());

        // 放行 Step 1 进入 slow_inspect 工具执行
        step1_gate.notify_waiters();
        tool_entered.notified().await;
        tokio::task::yield_now().await;

        let tool_snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        let active_tool = tool_snap.active_turn.expect("active_turn present in tool");
        assert_eq!(active_tool.running_tools.len(), 1);
        assert_eq!(active_tool.running_tools[0].call_id, "call_slow_1");
        assert_eq!(active_tool.running_tools[0].name, "slow_inspect");
        assert_eq!(
            active_tool.running_tools[0].output_tail,
            "streaming tool tail line 1\nline 2"
        );
        // Assistant 事件已提交，transient text/reasoning 清空无重影
        assert!(active_tool.text.is_empty());
        assert!(active_tool.reasoning.is_empty());
        assert!(tool_snap.events.iter().any(|e| e.kind == "assistant"));

        // 放行 slow_inspect，子代理 worker-a 启动并阻塞在流式生成中
        tool_gate.notify_waiters();
        child_entered.notified().await;

        let sub_notif = next_rpc_matching(&mut rpc_rx, |n| {
            matches!(
                n,
                RpcNotification::SubagentUpdated(u)
                    if u.task.name == "worker-a" && u.task.status == RpcSubagentStatus::Running
            )
        })
        .await;
        let child_sid = match sub_notif {
            RpcNotification::SubagentUpdated(u) => u.task.child_session_id,
            _ => unreachable!(),
        };

        // 验证主会话快照包含 subagents 且子代理 child_session_id 可独立 subscribe_snapshot
        let parent_with_sub = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(parent_with_sub.subagents.len(), 1);
        assert_eq!(parent_with_sub.subagents[0].child_session_id, child_sid);

        let child_snap = actor.subscribe_snapshot(&child_sid, None).await.unwrap();
        assert_eq!(child_snap.session.id, child_sid);
        assert_eq!(child_snap.state, SessionRuntimeState::Streaming);
        let child_active = child_snap.active_turn.expect("child midstream active_turn");
        assert_eq!(child_active.text, "child midstream");

        // 放行子代理完成 -> 触发 SubagentUpdated(Completed) 并自动唤醒父会话完成汇总
        child_gate.notify_waiters();
        next_rpc_matching(&mut rpc_rx, |n| {
            matches!(
                n,
                RpcNotification::SubagentUpdated(u)
                    if u.task.name == "worker-a" && u.task.status == RpcSubagentStatus::Completed
            )
        })
        .await;

        next_client_matching(
            &mut client_rx,
            |ev| matches!(ev, ClientEvent::TextDelta { text, .. } if text == "汇总完成"),
        )
        .await;
        next_rpc_matching(&mut rpc_rx, |n| {
            matches!(
                n,
                RpcNotification::TurnState(t)
                    if t.session_id == sess.id && t.state == SessionRuntimeState::Idle
            )
        })
        .await;

        let done_snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert!(done_snap.active_turn.is_none());

        // 验证 SubagentTaskSnapshot 转换函数纯映射无误
        let dto = subagent_snapshot_to_dto(SubagentTaskSnapshot {
            name: "s1".into(),
            parent_session_id: "p1".into(),
            child_session_id: "c1".into(),
            mode: SubagentMode::Edit,
            prompt_preview: "fix".into(),
            status: SubagentRunStatus::TimedOut,
            steps: 5,
            current_step: 2,
            last_tool: Some("edit".into()),
            idle_secs: 3,
            watchdog_alerts: 1,
            elapsed_secs: 10,
            summary: Some("sum".into()),
            output_path: Some("/tmp/out".into()),
        });
        assert_eq!(dto.mode, RpcSubagentMode::Edit);
        assert_eq!(dto.status, RpcSubagentStatus::TimedOut);
        assert_eq!(dto.parent_session_id.as_deref(), Some("p1"));

        actor.shutdown().await;
        rt.shutdown().await;
    }

    #[tokio::test]
    async fn dual_source_resolve_interaction_preemption_and_sqlite_persistence() {
        let ws = TempDirGuard::new("approval_sqlite");
        let db_path = ws.path().join("sessions.sqlite");
        let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()).with_session_db(&db_path))
            .await
            .unwrap();

        let _danger_guard = rt
            .tools
            .register(
                ToolDefinition::sync_tool(
                    "write_secret",
                    "dangerous write",
                    json!({ "type": "object", "properties": {} }),
                    |_, _| Ok::<_, BoxError>("secret_written".to_string()),
                )
                .with_annotations(ToolAnnotations {
                    read_only: false,
                    ..Default::default()
                }),
            )
            .unwrap();

        let adapter = ScriptedAdapter::new(vec![
            ScriptStep::Immediate(vec![
                StreamChunk::tool_call(
                    0,
                    ToolCall {
                        id: "call-sec-1".into(),
                        name: "write_secret".into(),
                        arguments: r#"{"file":"secret.txt"}"#.into(),
                    },
                ),
                StreamChunk::Finish {
                    reason: FinishReason::ToolCalls,
                    replay: None,
                },
            ]),
            ScriptStep::Immediate(vec![
                StreamChunk::TextDelta {
                    index: 0,
                    text: "已安全写入".into(),
                },
                StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                },
            ]),
        ]);
        rt.register_adapter("scripted", adapter).unwrap();

        let sess = rt.sessions.create(Some("sess-sqlite-apr".into()));
        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&sess),
            SessionActorConfig::new(ws.path())
                .with_approval_mode(ApprovalMode::Ask)
                .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
        );
        let mut rpc_rx = actor.subscribe_notifications();
        let mut client_rx = actor.subscribe_client_events();

        actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "write secret".into(),
                attachments: None,
                mode: Some(SubmitMode::Send),
            })
            .await
            .unwrap();

        let req_dto = match next_rpc_matching(&mut rpc_rx, |n| {
            matches!(n, RpcNotification::InteractionRequested(_))
        })
        .await
        {
            RpcNotification::InteractionRequested(dto) => dto,
            _ => unreachable!(),
        };
        assert_eq!(req_dto.tool, "write_secret");
        assert_eq!(req_dto.call_id.as_deref(), Some("call-sec-1"));

        let tui_responder = match next_client_matching(&mut client_rx, |ev| {
            matches!(ev, ClientEvent::ApprovalRequested { .. })
        })
        .await
        {
            ClientEvent::ApprovalRequested {
                request_id,
                responder,
                ..
            } => {
                assert_eq!(request_id, req_dto.request_id);
                responder.lock().unwrap().take().unwrap()
            }
            _ => unreachable!(),
        };

        let waiting_snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(waiting_snap.state, SessionRuntimeState::WaitingApproval);
        assert_eq!(waiting_snap.pending_interactions.len(), 1);

        // RPC 率先抢占 AllowOnce；TUI 并发 Deny 竞态失败并返回 already_resolved
        let win = actor
            .resolve_interaction(
                InteractionResolveParams {
                    session_id: Some(sess.id.clone()),
                    request_id: req_dto.request_id.clone(),
                    decision: RpcApprovalDecision::AllowOnce,
                    note: None,
                },
                ResolvedBy::Rpc,
            )
            .await
            .unwrap();
        assert_eq!(win, InteractionResolveResult::ok());

        let lose = actor
            .resolve_interaction(
                InteractionResolveParams {
                    session_id: Some(sess.id.clone()),
                    request_id: req_dto.request_id.clone(),
                    decision: RpcApprovalDecision::Deny,
                    note: None,
                },
                ResolvedBy::Tui,
            )
            .await
            .unwrap();
        assert_eq!(lose, InteractionResolveResult::already_resolved());
        let _ = tui_responder.send(ApprovalDecision::Deny);

        next_client_matching(&mut client_rx, |ev| {
            matches!(ev, ClientEvent::RunFinished(Ok(_)))
        })
        .await;

        let sid = sess.id.clone();
        sess.flush_async().await.unwrap();
        actor.shutdown().await;
        drop(sess);
        rt.shutdown().await;

        // 冷重开 SQLite 验证 SessionTraceEvent::Approval 完整持久化落盘
        let reopened = SqliteSessionPersistence::open_async(&db_path)
            .await
            .unwrap();
        let loaded = reopened
            .load_session(&sid)
            .unwrap()
            .expect("session exists");
        assert!(loaded.entries.iter().any(|entry| matches!(
            &entry.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::Approval {
                    tool_call_id,
                    tool,
                    decision,
                    resolved_by,
                }
            } if tool_call_id == "call-sec-1"
                && tool == "write_secret"
                && decision == "allow_once"
                && resolved_by == "rpc"
        )));
        let _ = reopened.shutdown();
    }

    #[tokio::test]
    async fn followup_auto_wake_and_single_source_queued_inputs() {
        let ws = TempDirGuard::new("queue_wakeup");
        let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
            .await
            .unwrap();

        let turn1_entered = Arc::new(Notify::new());
        let turn1_gate = Arc::new(Notify::new());
        let adapter = ScriptedAdapter::new(vec![
            ScriptStep::Gated {
                before: vec![StreamChunk::TextDelta {
                    index: 0,
                    text: "turn-1-start".into(),
                }],
                gate: Arc::clone(&turn1_gate),
                entered: Arc::clone(&turn1_entered),
                after: vec![StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }],
            },
            ScriptStep::EchoUser {
                prefix: "q:".into(),
            },
            ScriptStep::EchoUser {
                prefix: "fw:".into(),
            },
        ]);
        rt.register_adapter("scripted", adapter).unwrap();

        let sess = rt.sessions.create(Some("sess-queue".into()));
        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&sess),
            SessionActorConfig::new(ws.path()).with_provider_model(
                Some("scripted".into()),
                Some("script-v1".into()),
                None,
            ),
        );
        let mut rpc_rx = actor.subscribe_notifications();
        let mut client_rx = actor.subscribe_client_events();

        // 启动 Turn 1 并卡住，期间往单真源队列入队 Q1, Q2, Q3
        actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "first".into(),
                attachments: None,
                mode: Some(SubmitMode::Send),
            })
            .await
            .unwrap();
        turn1_entered.notified().await;

        let q1 = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "queued-1".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();
        let q2 = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "queued-2".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();
        let q3 = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "queued-3".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();
        assert_eq!(q1.mode, SubmitMode::Queue);
        let q2_id = q2.queue_id.unwrap();
        let q3_id = q3.queue_id.unwrap();

        let snap_q = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert_eq!(snap_q.queued_inputs.len(), 3);

        // 从队列撤回 Q2，验证幂等与 QueueUpdated 通知
        let rm = actor.queue_remove(&sess.id, q2_id).await.unwrap();
        assert!(rm.removed);
        assert_eq!(rm.item.unwrap().text, "queued-2");
        let rm_again = actor.queue_remove(&sess.id, q2_id).await.unwrap();
        assert!(!rm_again.removed);

        // 先放行 Q3 留在后面测试 pop_all？不行，Turn 1 完成后会先自动出队 Q1！
        // 如果测试 pop_all 留在后面，在 Q1 刚弹出时还有 Q3；但为了避免 Q1 跑完后自动弹 Q3，
        // 我们可以先 pop_all 剩余或等 Q1 执行期间检查，或者直接现在先测 pop_all 的话就把 Q1 放行！
        // 这里我们先删 Q3 -> 留 Q1 自动出队，然后再在 Turn 空闲后重新测试 queue_pop_all，确定性 100%！
        let rm_q3 = actor.queue_remove(&sess.id, q3_id).await.unwrap();
        assert!(rm_q3.removed);

        // 放行 Turn 1 -> 自动出队执行 Q1 ("queued-1")
        turn1_gate.notify_waiters();
        next_client_matching(
            &mut client_rx,
            |ev| matches!(ev, ClientEvent::RunFinished(Ok(txt)) if txt == "q:queued-1"),
        )
        .await;
        next_rpc_matching(&mut rpc_rx, |n| {
            matches!(
                n,
                RpcNotification::QueueUpdated(q) if q.session_id == sess.id && q.items.is_empty()
            )
        })
        .await;

        // 测试 queue_pop_all 一次性弹出多项
        let qa = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "draft-a".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();
        let qb = actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "draft-b".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();
        assert!(qa.queue_id.is_some() && qb.queue_id.is_some());
        let popped = actor.queue_pop_all(&sess.id).await.unwrap();
        assert_eq!(
            popped
                .items
                .iter()
                .map(|i| i.text.as_str())
                .collect::<Vec<_>>(),
            vec!["draft-a", "draft-b"]
        );
        assert!(
            actor
                .subscribe_snapshot(&sess.id, None)
                .await
                .unwrap()
                .queued_inputs
                .is_empty()
        );

        // 测试 FollowUp inbox 自动唤醒执行
        rt.agent_loop.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::user("auto-wakeup-prompt"),
        );
        next_client_matching(
            &mut client_rx,
            |ev| matches!(ev, ClientEvent::RunFinished(Ok(txt)) if txt == "fw:auto-wakeup-prompt"),
        )
        .await;

        actor.shutdown().await;
        rt.shutdown().await;
    }

    #[tokio::test]
    async fn switch_session_enforces_single_writer_lock_and_updates_registration() {
        let ws = TempDirGuard::new("single_writer");
        let run_dir = TempDirGuard::new("run_dir");
        let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
            .await
            .unwrap();

        let sess_init = rt.sessions.create(Some("sess-init".into()));
        let sess_locked = rt.sessions.create(Some("sess-locked".into()));
        let sess_free = rt.sessions.create(Some("sess-free".into()));

        // 启动另一活进程持有 sess-locked 写锁
        let _other_proc = ChildGuard::spawn_sleeper();
        let other_pid = _other_proc.pid();
        let other_meta = ProcessMeta {
            pid: other_pid,
            ppid: None,
            tty: None,
            herdr_pane_id: None,
            session_id: sess_locked.id.clone(),
            cwd: ws.path().display().to_string(),
            socket: run_dir
                .path()
                .join(format!("p-{other_pid}.sock"))
                .display()
                .to_string(),
            provider: None,
            model: None,
            started_at_ms: 1,
        };
        std::fs::write(
            meta_path_for_pid(run_dir.path(), other_pid),
            serde_json::to_vec(&other_meta).unwrap(),
        )
        .unwrap();

        let self_meta = ProcessMeta::for_current_process(
            run_dir.path(),
            &sess_init.id,
            ws.path().display().to_string(),
            None,
            None,
        );
        let self_meta_path = meta_path_for_pid(run_dir.path(), self_meta.pid);
        let reg = ProcessRegistration::create(run_dir.path(), self_meta).unwrap();

        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&sess_init),
            SessionActorConfig::new(ws.path()).with_run_dir(run_dir.path()),
        );
        actor.attach_registration(reg);
        let mut rpc_rx = actor.subscribe_notifications();
        let mut client_rx = actor.subscribe_client_events();

        // 1. 尝试切换到已被活进程占用的 sess-locked -> 拒绝并返回 ERR_SESSION_LOCKED
        let lock_err = actor.switch_session(&sess_locked.id).await.unwrap_err();
        let rpc_err = lock_err.to_jsonrpc_error();
        assert_eq!(rpc_err.code, ERR_SESSION_LOCKED);
        assert_eq!(rpc_err.data.as_ref().unwrap()["owner_pid"], other_pid);
        assert_eq!(actor.active_session_id(), sess_init.id);

        // 2. 切换到空闲的 sess-free -> 成功并同步更新 ProcessRegistration 与广播 SessionSwitched
        let switched = actor.switch_session(&sess_free.id).await.unwrap();
        assert_eq!(switched.id, sess_free.id);
        assert_eq!(actor.active_session_id(), sess_free.id);

        let disk_after_switch: ProcessMeta =
            serde_json::from_slice(&std::fs::read(&self_meta_path).unwrap()).unwrap();
        assert_eq!(disk_after_switch.session_id, sess_free.id);

        next_rpc_matching(&mut rpc_rx, |n| {
            matches!(
                n,
                RpcNotification::SessionSwitched(s)
                    if s.old_session_id == sess_init.id && s.new_session_id == sess_free.id
            )
        })
        .await;
        next_client_matching(&mut client_rx, |ev| {
            matches!(
                ev,
                ClientEvent::SessionSwitched { old_session_id, new_session_id }
                    if old_session_id == &sess_init.id && new_session_id == &sess_free.id
            )
        })
        .await;

        // 3. update_config 同步原子刷新 ProcessRegistration 的 provider/model 与 approval_mode
        let cfg_res = actor
            .update_config(SessionConfigParams {
                session_id: sess_free.id.clone(),
                provider: Some("deepseek".into()),
                model: Some("deepseek-reasoner".into()),
                reasoning_effort: Some(ReasoningEffort::High),
                approval_mode: Some(RpcApprovalMode::Ask),
            })
            .await
            .unwrap();
        assert_eq!(cfg_res.config.provider.as_deref(), Some("deepseek"));
        assert_eq!(cfg_res.config.model.as_deref(), Some("deepseek-reasoner"));
        assert_eq!(cfg_res.config.approval_mode, RpcApprovalMode::Ask);

        let disk_after_cfg: ProcessMeta =
            serde_json::from_slice(&std::fs::read(&self_meta_path).unwrap()).unwrap();
        assert_eq!(disk_after_cfg.provider.as_deref(), Some("deepseek"));
        assert_eq!(disk_after_cfg.model.as_deref(), Some("deepseek-reasoner"));

        actor.shutdown().await;
        rt.shutdown().await;
    }

    #[tokio::test]
    async fn session_fork_tree_list_events_and_capabilities() {
        let ws = TempDirGuard::new("queries");
        let rt = MizarRuntime::build(AgentSpineConfig::coding(ws.path()))
            .await
            .unwrap();
        rt.register_adapter("scripted", ScriptedAdapter::new(Vec::new()))
            .unwrap();

        let s1 = rt.sessions.create_with(SessionCreateOptions {
            id: Some("s-alpha".into()),
            cwd: Some("/proj/alpha".into()),
            title: Some("Alpha Task".into()),
            origin: Some(SessionOrigin::User),
            ..Default::default()
        });
        s1.append(SessionEvent::user("u1"));
        s1.append(SessionEvent::assistant_text("a1"));
        s1.append(SessionEvent::user("u2"));
        s1.append(SessionEvent::assistant_text("a2"));
        s1.append(SessionEvent::user("u3"));

        let s2 = rt.sessions.create_with(SessionCreateOptions {
            id: Some("s-beta".into()),
            cwd: Some("/proj/alpha".into()),
            title: Some("Beta Task".into()),
            origin: Some(SessionOrigin::User),
            ..Default::default()
        });
        s2.append(SessionEvent::user("hi"));

        let s3 = rt.sessions.create_with(SessionCreateOptions {
            id: Some("s-gamma".into()),
            cwd: Some("/proj/other".into()),
            title: Some("Gamma".into()),
            origin: Some(SessionOrigin::User),
            ..Default::default()
        });
        s3.append(SessionEvent::user("yo"));

        let _sub_child = rt.sessions.create_with(SessionCreateOptions {
            id: Some("s-sub".into()),
            parent_id: Some("s-alpha".into()),
            origin: Some(SessionOrigin::Subagent),
            ..Default::default()
        });

        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&s1),
            SessionActorConfig::new(ws.path()).with_provider_model(
                Some("scripted".into()),
                Some("script-v1".into()),
                None,
            ),
        );

        // 1. session.tree
        let tree = actor.session_tree(&s1.id).await.unwrap();
        assert_eq!(tree.session_id, "s-alpha");
        assert_eq!(tree.entries.len(), 5);
        assert_eq!(tree.entries[0].kind_tag, "USER");
        assert_eq!(tree.entries[1].kind_tag, "ASST");

        // 2. session.fork (at_seq = 2)
        let forked = actor
            .fork_session(SessionForkParams {
                session_id: "s-alpha".into(),
                at_seq: 2,
            })
            .await
            .unwrap();
        assert_eq!(forked.session.parent_id.as_deref(), Some("s-alpha"));
        assert_eq!(forked.session.fork_seq, Some(2));
        let forked_events = actor
            .list_events(SessionEventsParams {
                session_id: forked.session.id.clone(),
                before_seq: None,
                after_seq: None,
                limit: Some(10),
            })
            .await
            .unwrap();
        assert_eq!(forked_events.events.len(), 2);

        // 3. session.events 双向分页
        let page_fwd = actor
            .list_events(SessionEventsParams {
                session_id: "s-alpha".into(),
                before_seq: None,
                after_seq: Some(1),
                limit: Some(2),
            })
            .await
            .unwrap();
        assert!(page_fwd.has_more);
        assert_eq!(
            page_fwd.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![2, 3]
        );

        let page_back = actor
            .list_events(SessionEventsParams {
                session_id: "s-alpha".into(),
                before_seq: Some(5),
                after_seq: None,
                limit: Some(2),
            })
            .await
            .unwrap();
        assert!(page_back.has_more);
        assert_eq!(
            page_back.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![3, 4]
        );

        // 4. session.list 过滤 + 游标分页（默认排除子代理）
        let by_cwd = actor
            .list_sessions(SessionListParams {
                cwd: Some("/proj/alpha".into()),
                query: None,
                limit: Some(2),
                cursor: None,
                exclude_subagents: Some(true),
            })
            .await
            .unwrap();
        assert!(!by_cwd.sessions.iter().any(|s| s.id == "s-sub"));
        assert!(!by_cwd.sessions.iter().any(|s| s.id == "s-gamma"));
        assert_eq!(by_cwd.sessions.len(), 2);
        assert!(by_cwd.next_cursor.is_some());

        let page2 = actor
            .list_sessions(SessionListParams {
                cwd: Some("/proj/alpha".into()),
                query: None,
                limit: Some(2),
                cursor: by_cwd.next_cursor.clone(),
                exclude_subagents: Some(true),
            })
            .await
            .unwrap();
        // s-alpha, s-beta, plus forked s-alpha share cwd=/proj/alpha => total 3 sessions across 2 pages
        assert_eq!(page2.sessions.len(), 1);
        assert!(page2.next_cursor.is_none());

        let by_query = actor
            .list_sessions(SessionListParams {
                cwd: None,
                query: Some("BETA".into()),
                limit: None,
                cursor: None,
                exclude_subagents: Some(true),
            })
            .await
            .unwrap();
        assert_eq!(by_query.sessions.len(), 1);
        assert_eq!(by_query.sessions[0].id, "s-beta");

        // 5. capabilities.list
        let caps = actor.list_capabilities(Some(&s1.id)).await.unwrap();
        assert!(caps.commands.iter().any(|c| c.command == "resume"));
        assert!(caps.tools.iter().any(|t| t.name == "read"));
        assert!(caps.providers.iter().any(|p| p.id == "scripted"));
        assert_eq!(caps.max_context_tokens, Some(64_000));

        // 6. new_session + subscribe_snapshot after_seq 增量过滤 + cancel_turn 保留排队
        let fresh = actor.new_session(Some("Fresh".into())).await.unwrap();
        assert_ne!(fresh.id, s1.id);
        assert_eq!(actor.active_session_id(), fresh.id);

        let snap_after = actor.subscribe_snapshot(&s1.id, Some(3)).await.unwrap();
        assert_eq!(
            snap_after.events.iter().map(|e| e.seq).collect::<Vec<_>>(),
            vec![4, 5]
        );

        let missing = actor.subscribe_snapshot("non-existent-sid", None).await;
        assert!(matches!(missing, Err(ActorError::SessionNotFound(_))));

        actor.shutdown().await;
        rt.shutdown().await;
    }

    #[tokio::test]
    async fn cancel_turn_denies_pending_approvals_and_preserves_queue() {
        let ws = TempDirGuard::new("cancel_turn");
        let rt = MizarRuntime::build(AgentSpineConfig::new(ws.path()))
            .await
            .unwrap();
        let _danger_guard = rt
            .tools
            .register(
                ToolDefinition::sync_tool(
                    "danger_cmd",
                    "danger",
                    json!({ "type": "object", "properties": {} }),
                    |_, _| Ok::<_, BoxError>("ok".to_string()),
                )
                .with_annotations(ToolAnnotations {
                    read_only: false,
                    ..Default::default()
                }),
            )
            .unwrap();
        let adapter = ScriptedAdapter::new(vec![ScriptStep::Immediate(vec![
            StreamChunk::tool_call(
                0,
                ToolCall {
                    id: "call-cancel-1".into(),
                    name: "danger_cmd".into(),
                    arguments: "{}".into(),
                },
            ),
            StreamChunk::Finish {
                reason: FinishReason::ToolCalls,
                replay: None,
            },
        ])]);
        rt.register_adapter("scripted", adapter).unwrap();

        let sess = rt.sessions.create(Some("sess-cancel".into()));
        let actor = SessionActor::spawn(
            Arc::clone(&rt),
            Arc::clone(&sess),
            SessionActorConfig::new(ws.path())
                .with_approval_mode(ApprovalMode::Ask)
                .with_provider_model(Some("scripted".into()), Some("script-v1".into()), None),
        );
        let mut rpc_rx = actor.subscribe_notifications();
        let mut client_rx = actor.subscribe_client_events();

        actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "run danger".into(),
                attachments: None,
                mode: Some(SubmitMode::Send),
            })
            .await
            .unwrap();
        next_rpc_matching(&mut rpc_rx, |n| {
            matches!(n, RpcNotification::InteractionRequested(_))
        })
        .await;

        actor
            .submit_turn(TurnSubmitParams {
                session_id: sess.id.clone(),
                text: "keep me queued".into(),
                attachments: None,
                mode: Some(SubmitMode::Queue),
            })
            .await
            .unwrap();

        let cancelled = actor.cancel_turn(&sess.id).await.unwrap();
        assert!(cancelled.cancelled);

        next_client_matching(&mut client_rx, |ev| {
            matches!(ev, ClientEvent::RunFinished(Err(_)))
        })
        .await;

        let snap = actor.subscribe_snapshot(&sess.id, None).await.unwrap();
        assert!(snap.pending_interactions.is_empty());
        assert_eq!(snap.queued_inputs.len(), 1);
        assert_eq!(snap.queued_inputs[0].text, "keep me queued");

        actor.shutdown().await;
        rt.shutdown().await;
    }
}
