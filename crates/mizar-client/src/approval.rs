//! 工具执行前危险操作审批门禁（`tools.pre_execute_hook()` 瀑布流钩子与 `PendingInteractions` 并发抢占注册表）。

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::oneshot;
use mizar_core::spine::MizarRuntime;
use mizar_session::SessionStore;
use mizar_tools::{PreExecuteDecision, ToolExecution, ToolRuntime};
use mizar_types::client::{ApprovalDecision, ApprovalMode, ClientEvent, truncate_display};
use mizar_types::rpc::{
    InteractionResolvedNotification, PendingInteractionDto, ResolvedBy, RpcApprovalDecision,
    RpcNotification,
};
use mizar_types::{BoxFuture, HookGuard, Next};
use thiserror::Error;
use tokio::sync::mpsc;

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_ENTRY_ORDER: AtomicU64 = AtomicU64::new(1);

fn next_approval_request_id() -> String {
    let seq = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    format!("apr-{}-{seq}", std::process::id())
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// 运行期共享审批策略快照（供异步工具执行前瀑布流无锁检查）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalSnapshot {
    pub mode: ApprovalMode,
    pub always_allowed: HashSet<String>,
}

/// 单条挂起中的工具审批请求（持有一致性元信息与工具执行解锁通道）。
pub struct PendingInteractionEntry {
    pub dto: PendingInteractionDto,
    pub session_id: String,
    pub responder: oneshot::Sender<ApprovalDecision>,
    order: u64,
}

impl fmt::Debug for PendingInteractionEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingInteractionEntry")
            .field("dto", &self.dto)
            .field("session_id", &self.session_id)
            .field("order", &self.order)
            .finish_non_exhaustive()
    }
}

/// 成功抢占并完成一次审批裁决后的结果元数据。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalResolveMeta {
    pub session_id: String,
    pub request_id: String,
    pub call_id: Option<String>,
    pub tool: String,
    pub decision: ApprovalDecision,
    pub resolved_by: ResolvedBy,
}

/// 审批裁决抢占失败原因。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResolveInteractionError {
    #[error("交互请求 `{0}` 已被裁决或不存在")]
    AlreadyResolved(String),
    #[error(
        "交互请求 `{request_id}` 归属于会话 `{actual_session_id}`，与请求的 `{expected_session_id}` 不匹配"
    )]
    SessionMismatch {
        request_id: String,
        expected_session_id: String,
        actual_session_id: String,
    },
}

/// 审批裁决完成后的同步副作用汇（白名单更新、SQLite 审计写盘与多端事件广播）。
#[derive(Clone, Default)]
pub struct ApprovalSideEffects {
    pub sessions: Option<Arc<SessionStore>>,
    pub approval_snapshot: Option<Arc<RwLock<ApprovalSnapshot>>>,
    pub emit_client: Option<Arc<dyn Fn(ClientEvent) + Send + Sync>>,
    pub emit_rpc: Option<Arc<dyn Fn(RpcNotification) + Send + Sync>>,
}

impl fmt::Debug for ApprovalSideEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalSideEffects")
            .field("has_sessions", &self.sessions.is_some())
            .field("has_snapshot", &self.approval_snapshot.is_some())
            .field("has_emit_client", &self.emit_client.is_some())
            .field("has_emit_rpc", &self.emit_rpc.is_some())
            .finish()
    }
}

/// 多端共享的挂起交互请求注册表：通过 `HashMap::remove(&request_id)` 实现原子抢占，
/// 保证同一审批在 TUI 本地快捷键与外部 RPC 并发裁决时仅首个胜出方生效。
#[derive(Default)]
pub struct PendingInteractions {
    entries: Mutex<HashMap<String, PendingInteractionEntry>>,
    side_effects: RwLock<Option<Arc<ApprovalSideEffects>>>,
}

impl fmt::Debug for PendingInteractions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.entries.lock().unwrap().len();
        f.debug_struct("PendingInteractions")
            .field("pending_count", &count)
            .finish()
    }
}

impl PendingInteractions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_side_effects(side_effects: ApprovalSideEffects) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            side_effects: RwLock::new(Some(Arc::new(side_effects))),
        }
    }

    /// 绑定或更新审批完成时的持久化与多端广播副作用汇。
    pub fn attach_side_effects(&self, side_effects: ApprovalSideEffects) {
        *self.side_effects.write().unwrap() = Some(Arc::new(side_effects));
    }

    /// 注册一条待人工裁决的交互条目（必须在向外广播请求事件之前调用，避免早到响应找不到条目）。
    pub fn register(
        &self,
        dto: PendingInteractionDto,
        session_id: impl Into<String>,
        responder: oneshot::Sender<ApprovalDecision>,
    ) {
        let request_id = dto.request_id.clone();
        let entry = PendingInteractionEntry {
            dto,
            session_id: session_id.into(),
            responder,
            order: NEXT_ENTRY_ORDER.fetch_add(1, Ordering::Relaxed),
        };
        self.entries.lock().unwrap().insert(request_id, entry);
    }

    /// 原子抢占并裁决指定 `request_id`；若已被另一端先一步处理则返回 `AlreadyResolved`。
    pub fn resolve(
        &self,
        request_id: &str,
        decision: ApprovalDecision,
        resolved_by: ResolvedBy,
    ) -> Result<ApprovalResolveMeta, ResolveInteractionError> {
        self.resolve_for_session(None, request_id, decision, resolved_by)
    }

    /// 带可选 `session_id` 校验的原子抢占裁决（校验不匹配时不移除原条目）。
    pub fn resolve_for_session(
        &self,
        expected_session_id: Option<&str>,
        request_id: &str,
        decision: ApprovalDecision,
        resolved_by: ResolvedBy,
    ) -> Result<ApprovalResolveMeta, ResolveInteractionError> {
        let entry = {
            let mut map = self.entries.lock().unwrap();
            if let Some(expected) = expected_session_id
                && let Some(existing) = map.get(request_id)
                && existing.session_id != expected
            {
                return Err(ResolveInteractionError::SessionMismatch {
                    request_id: request_id.to_string(),
                    expected_session_id: expected.to_string(),
                    actual_session_id: existing.session_id.clone(),
                });
            }
            map.remove(request_id)
                .ok_or_else(|| ResolveInteractionError::AlreadyResolved(request_id.to_string()))?
        };
        Ok(self.finalize_entry(entry, decision, resolved_by))
    }

    /// 以同一裁决排空全部挂起交互请求（如进程退出清理）。
    pub fn resolve_all(
        &self,
        decision: ApprovalDecision,
        resolved_by: ResolvedBy,
    ) -> Vec<ApprovalResolveMeta> {
        let mut drained: Vec<PendingInteractionEntry> = {
            let mut map = self.entries.lock().unwrap();
            map.drain().map(|(_, v)| v).collect()
        };
        drained.sort_by_key(|e| e.order);
        drained
            .into_iter()
            .map(|entry| self.finalize_entry(entry, decision.clone(), resolved_by))
            .collect()
    }

    /// 仅排空指定 `session_id` 名下全部挂起交互请求（如取消当前回合或切换会话）。
    pub fn resolve_all_for_session(
        &self,
        session_id: &str,
        decision: ApprovalDecision,
        resolved_by: ResolvedBy,
    ) -> Vec<ApprovalResolveMeta> {
        let mut drained = Vec::new();
        {
            let mut map = self.entries.lock().unwrap();
            let keys: Vec<String> = map
                .iter()
                .filter(|(_, v)| v.session_id == session_id)
                .map(|(k, _)| k.clone())
                .collect();
            for key in keys {
                if let Some(entry) = map.remove(&key) {
                    drained.push(entry);
                }
            }
        }
        drained.sort_by_key(|e| e.order);
        drained
            .into_iter()
            .map(|entry| self.finalize_entry(entry, decision.clone(), resolved_by))
            .collect()
    }

    /// 按注册先后顺序返回当前全部挂起交互的协议 DTO 快照。
    pub fn list_pending(&self) -> Vec<PendingInteractionDto> {
        let map = self.entries.lock().unwrap();
        let mut entries: Vec<&PendingInteractionEntry> = map.values().collect();
        entries.sort_by_key(|e| e.order);
        entries.into_iter().map(|e| e.dto.clone()).collect()
    }

    /// 返回指定会话当前挂起中的交互条目列表。
    pub fn list_pending_for_session(&self, session_id: &str) -> Vec<PendingInteractionDto> {
        let map = self.entries.lock().unwrap();
        let mut entries: Vec<&PendingInteractionEntry> = map
            .values()
            .filter(|e| e.session_id == session_id)
            .collect();
        entries.sort_by_key(|e| e.order);
        entries.into_iter().map(|e| e.dto.clone()).collect()
    }

    /// 判断指定会话是否存在正在等待人工确认的交互请求。
    pub fn has_pending_for_session(&self, session_id: &str) -> bool {
        self.entries
            .lock()
            .unwrap()
            .values()
            .any(|e| e.session_id == session_id)
    }

    fn finalize_entry(
        &self,
        entry: PendingInteractionEntry,
        decision: ApprovalDecision,
        resolved_by: ResolvedBy,
    ) -> ApprovalResolveMeta {
        let request_id = entry.dto.request_id.clone();
        let call_id = entry.dto.call_id.clone();
        let tool = entry.dto.tool.clone();
        let session_id = entry.session_id;
        let (rpc_decision, note) = RpcApprovalDecision::from_approval_decision(&decision);

        let side_effects = self.side_effects.read().unwrap().clone();
        if let Some(effects) = side_effects {
            if decision == ApprovalDecision::AllowAlwaysForTool
                && let Some(snap) = &effects.approval_snapshot
                && let Ok(mut guard) = snap.write()
            {
                guard.always_allowed.insert(tool.clone());
            }

            // 在放行工具执行前先落盘审批审计记录，保证回放与轨迹时间线中 Approval 早于 ToolResult。
            if let Some(store) = &effects.sessions
                && let Some(session) = store.get(&session_id)
            {
                let tool_call_id = call_id.clone().unwrap_or_else(|| request_id.clone());
                session.append_approval(
                    tool_call_id,
                    &tool,
                    rpc_decision.as_str(),
                    resolved_by.as_str(),
                );
            }

            if let Some(emit_rpc) = &effects.emit_rpc {
                emit_rpc(RpcNotification::InteractionResolved(
                    InteractionResolvedNotification {
                        session_id: Some(session_id.clone()),
                        request_id: request_id.clone(),
                        decision: rpc_decision,
                        note,
                        resolved_by,
                        ts_ms: now_unix_ms(),
                    },
                ));
            }
            if let Some(emit_client) = &effects.emit_client {
                emit_client(ClientEvent::ApprovalResolved {
                    request_id: request_id.clone(),
                });
            }
        }

        let _ = entry.responder.send(decision.clone());

        ApprovalResolveMeta {
            session_id,
            request_id,
            call_id,
            tool,
            decision,
            resolved_by,
        }
    }
}

/// 危险操作审批门禁配置：关联活跃会话 ID、审批策略快照、共享交互表与事件回调。
#[derive(Clone)]
pub struct ApprovalGateConfig {
    pub active_session_id: Arc<RwLock<String>>,
    pub snapshot: Arc<RwLock<ApprovalSnapshot>>,
    pub emit: Arc<dyn Fn(ClientEvent) + Send + Sync>,
    pub pending: Arc<PendingInteractions>,
    pub emit_rpc: Option<Arc<dyn Fn(RpcNotification) + Send + Sync>>,
}

impl fmt::Debug for ApprovalGateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalGateConfig")
            .field("active_session_id", &self.active_session_id)
            .field("snapshot", &self.snapshot)
            .field("pending", &self.pending)
            .field("has_emit_rpc", &self.emit_rpc.is_some())
            .finish_non_exhaustive()
    }
}

impl ApprovalGateConfig {
    pub fn new(
        active_session_id: Arc<RwLock<String>>,
        snapshot: Arc<RwLock<ApprovalSnapshot>>,
        tx: mpsc::UnboundedSender<ClientEvent>,
    ) -> Self {
        Self::with_callback(active_session_id, snapshot, move |ev| {
            let _ = tx.send(ev);
        })
    }

    pub fn with_callback<F>(
        active_session_id: Arc<RwLock<String>>,
        snapshot: Arc<RwLock<ApprovalSnapshot>>,
        emit: F,
    ) -> Self
    where
        F: Fn(ClientEvent) + Send + Sync + 'static,
    {
        Self {
            active_session_id,
            snapshot,
            emit: Arc::new(emit),
            pending: Arc::new(PendingInteractions::new()),
            emit_rpc: None,
        }
    }

    pub fn with_pending(mut self, pending: Arc<PendingInteractions>) -> Self {
        self.pending = pending;
        self
    }

    pub fn with_rpc_callback<R>(mut self, emit_rpc: R) -> Self
    where
        R: Fn(RpcNotification) + Send + Sync + 'static,
    {
        self.emit_rpc = Some(Arc::new(emit_rpc));
        self
    }
}

/// 将交互式工具执行前审批门禁挂载到 `MizarRuntime` 上，返回 RAII `HookGuard`。
pub fn install_approval_gate(runtime: &MizarRuntime, config: ApprovalGateConfig) -> HookGuard {
    let emit = Arc::clone(&config.emit);
    let emit_rpc = config.emit_rpc.clone();
    register_approval_hook_with_interactions(
        &runtime.tools,
        Some(Arc::clone(&runtime.sessions)),
        config.active_session_id,
        config.snapshot,
        config.pending,
        move |ev| emit(ev),
        emit_rpc,
    )
}

/// 判断给定工具在 `Ask` 模式下是否需要向用户请求交互审批（只读工具自动豁免）。
pub fn tool_needs_approval(tools: &ToolRuntime, tool_name: &str) -> bool {
    if let Some(info) = tools.list_all().into_iter().find(|i| i.name == tool_name) {
        return !info.annotations.read_only;
    }
    !matches!(
        tool_name,
        "read"
            | "grep"
            | "find"
            | "ls"
            | "tool_search"
            | "mcp_list_resources"
            | "mcp_read_resource"
    )
}

/// 注册 `tools.pre_execute_hook()` 审批瀑布流钩子，并通过 `mpsc::UnboundedSender<ClientEvent>` 投递弹窗请求。
pub fn register_approval_hook(
    tools: &Arc<ToolRuntime>,
    sessions: Option<Arc<SessionStore>>,
    tx: mpsc::UnboundedSender<ClientEvent>,
    active_sid: Arc<RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
) -> HookGuard {
    register_approval_hook_with(tools, sessions, active_sid, approval_snap, move |ev| {
        let _ = tx.send(ev);
    })
}

/// 注册 `tools.pre_execute_hook()` 审批瀑布流钩子，向自定义回调投递 `ClientEvent::ApprovalRequested`。
pub fn register_approval_hook_with<F>(
    tools: &Arc<ToolRuntime>,
    sessions: Option<Arc<SessionStore>>,
    active_sid: Arc<RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
    emit: F,
) -> HookGuard
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    register_approval_hook_with_interactions(
        tools,
        sessions,
        active_sid,
        approval_snap,
        Arc::new(PendingInteractions::new()),
        emit,
        None,
    )
}

struct PendingDropGuard {
    pending: Arc<PendingInteractions>,
    request_id: String,
}

impl Drop for PendingDropGuard {
    fn drop(&mut self) {
        // 若工具 Future 在外部强制取消时仍处于挂起状态，兜底清理挂起表并通知前端关闭弹窗。
        let _ = self
            .pending
            .resolve(&self.request_id, ApprovalDecision::Deny, ResolvedBy::Tui);
    }
}

/// 注册支持 `PendingInteractions` 双源原子抢占与 `RpcNotification` 广播的审批钩子。
pub fn register_approval_hook_with_interactions<F>(
    tools: &Arc<ToolRuntime>,
    sessions: Option<Arc<SessionStore>>,
    active_sid: Arc<RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
    pending: Arc<PendingInteractions>,
    emit_client: F,
    emit_rpc: Option<Arc<dyn Fn(RpcNotification) + Send + Sync>>,
) -> HookGuard
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    let client_arc: Arc<dyn Fn(ClientEvent) + Send + Sync> = Arc::new(emit_client.clone());
    pending.attach_side_effects(ApprovalSideEffects {
        sessions: sessions.clone(),
        approval_snapshot: Some(Arc::clone(&approval_snap)),
        emit_client: Some(Arc::clone(&client_arc)),
        emit_rpc: emit_rpc.clone(),
    });

    let tools_ref = Arc::clone(tools);
    tools.pre_execute_hook().register(
        move |exec: ToolExecution,
              next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let emit_client = emit_client.clone();
            let emit_rpc = emit_rpc.clone();
            let pending = Arc::clone(&pending);
            let active_sid = Arc::clone(&active_sid);
            let snap_lock = Arc::clone(&approval_snap);
            let tools_inner = Arc::clone(&tools_ref);
            let sessions_inner = sessions.clone();
            Box::pin(async move {
                let needs_prompt = {
                    let guard = snap_lock.read().ok();
                    match guard {
                        Some(g)
                            if g.mode == ApprovalMode::Ask
                                && !g.always_allowed.contains(&exec.name) =>
                        {
                            tool_needs_approval(&tools_inner, &exec.name)
                        }
                        _ => false,
                    }
                };

                if !needs_prompt {
                    return next.call(exec).await;
                }

                // 区分直系后台/同步子代理会话与真正的过期旧会话，避免误报“会话已切换”，同时不让后台任务抢占前台交互审批框。
                let active_id = active_sid.read().unwrap().clone();
                if let Some(sid) = exec.session_id.as_deref()
                    && sid != active_id
                {
                    if let Some(store) = sessions_inner.as_ref()
                        && let Some(loaded) = store.inspect_readonly(sid)
                        && loaded.header.is_subagent_child()
                        && loaded.header.parent_id.as_deref() == Some(active_id.as_str())
                    {
                        return PreExecuteDecision::deny(
                            "当前处于 --approve 审批模式，子代理不支持弹出交互审批框执行非只读工具；请改用 read_only 子代理调研，或由主代理直接执行修改/命令",
                        );
                    }
                    return PreExecuteDecision::deny("会话已切换，取消过期审批请求");
                }

                let session_id = exec.session_id.clone().unwrap_or(active_id);
                let request_id = next_approval_request_id();
                let call_id = (!exec.call_id.is_empty()).then(|| exec.call_id.clone());

                let args_preview = serde_json::to_string_pretty(&exec.args)
                    .unwrap_or_else(|_| exec.args.to_string());
                let args_preview = truncate_display(&args_preview.replace('\n', " "), 400);

                let dto = PendingInteractionDto {
                    session_id: Some(session_id.clone()),
                    request_id: request_id.clone(),
                    kind: "approval".to_string(),
                    call_id: call_id.clone(),
                    tool: exec.name.clone(),
                    summary: format!("{} {}", exec.name, args_preview),
                    args: exec.args.clone(),
                    options: vec![
                        "allow_once".to_string(),
                        "allow_always".to_string(),
                        "deny".to_string(),
                    ],
                };

                let (tool_tx, mut tool_rx) = oneshot::channel();
                let (tui_tx, tui_rx) = oneshot::channel();
                pending.register(dto.clone(), session_id, tool_tx);
                let _drop_guard = PendingDropGuard {
                    pending: Arc::clone(&pending),
                    request_id: request_id.clone(),
                };

                if let Some(rpc_cb) = &emit_rpc {
                    rpc_cb(RpcNotification::InteractionRequested(dto));
                }
                emit_client(ClientEvent::ApprovalRequested {
                    request_id: request_id.clone(),
                    call_id,
                    tool_name: exec.name.clone(),
                    args_preview,
                    responder: Arc::new(Mutex::new(Some(tui_tx))),
                });

                let mut tui_rx = Some(tui_rx);
                let decision = loop {
                    tokio::select! {
                        res = &mut tool_rx => {
                            break res.unwrap_or(ApprovalDecision::Deny);
                        }
                        tui_res = async {
                            match tui_rx.as_mut() {
                                Some(rx) => rx.await,
                                None => std::future::pending().await,
                            }
                        }, if tui_rx.is_some() => {
                            tui_rx = None;
                            // 仅当 TUI 显式发出裁决时尝试原子抢占；若无人消费（Canceled）则继续等待 RPC 裁决。
                            if let Ok(dec) = tui_res {
                                let _ = pending.resolve(&request_id, dec, ResolvedBy::Tui);
                            }
                        }
                    }
                };

                match decision {
                    ApprovalDecision::AllowOnce | ApprovalDecision::AllowAlwaysForTool => {
                        next.call(exec).await
                    }
                    ApprovalDecision::DenyWithReason(reason) => PreExecuteDecision::deny(reason),
                    ApprovalDecision::Deny => PreExecuteDecision::deny("用户拒绝执行该工具"),
                }
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use mizar_session::{SessionEvent, SessionTraceEvent};
    use mizar_tools::{ToolAnnotations, ToolDefinition};
    use mizar_types::BoxError;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn approval_gate_enforces_ask_mode_exemptions_and_cleanup() {
        let tools = ToolRuntime::new();
        let sessions = SessionStore::new();
        let sess = sessions.create(Some("sess-1".into()));
        let _safe_g = tools
            .register(
                ToolDefinition::sync_tool("safe_read", "read only", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".to_string())
                })
                .with_annotations(ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                }),
            )
            .unwrap();
        let _danger_g = tools
            .register(
                ToolDefinition::sync_tool("danger_write", "write op", json!({}), |_, _| {
                    Ok::<_, BoxError>("written".to_string())
                })
                .with_annotations(ToolAnnotations {
                    read_only: false,
                    ..Default::default()
                }),
            )
            .unwrap();

        let active_sid = Arc::new(RwLock::new(sess.id.clone()));
        let snap = Arc::new(RwLock::new(ApprovalSnapshot {
            mode: ApprovalMode::Ask,
            always_allowed: HashSet::new(),
        }));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let config = ApprovalGateConfig::new(Arc::clone(&active_sid), Arc::clone(&snap), tx);
        let emit = Arc::clone(&config.emit);
        let pending = Arc::clone(&config.pending);
        let hook_guard = register_approval_hook_with_interactions(
            &tools,
            Some(Arc::clone(&sessions)),
            config.active_session_id,
            config.snapshot,
            Arc::clone(&pending),
            move |ev| emit(ev),
            None,
        );

        // 1. 只读工具无需弹窗审批
        let read_out = tools
            .execute(ToolExecution::new(
                "safe_read",
                json!({}),
                Some("sess-1".into()),
            ))
            .await;
        assert_eq!(read_out.content, "ok");
        assert!(rx.try_recv().is_err());

        // 2. 非当前活跃会话的工具调用被直接拦截；直系非 Fork 子代理给予诚实说明，避免误报“会话已切换”
        let stale_out = tools
            .execute(ToolExecution::new(
                "danger_write",
                json!({}),
                Some("sess-old".into()),
            ))
            .await;
        assert!(stale_out.is_error);
        assert!(stale_out.content.contains("会话已切换"));

        let child_sess = sessions.create_with(mizar_session::SessionCreateOptions {
            parent_id: Some("sess-1".into()),
            ..Default::default()
        });
        let child_out = tools
            .execute(ToolExecution::new(
                "danger_write",
                json!({}),
                Some(child_sess.id.clone()),
            ))
            .await;
        assert!(child_out.is_error);
        assert!(
            child_out.content.contains("--approve 审批模式")
                && !child_out.content.contains("会话已切换"),
            "{}",
            child_out.content
        );

        // 3. 活跃会话的写工具触发审批请求，允许后正常放行、写入审计并清理挂起表
        let tools_exec = Arc::clone(&tools);
        let fut = tokio::spawn(async move {
            tools_exec
                .execute(
                    ToolExecution::new(
                        "danger_write",
                        json!({"path": "a.txt"}),
                        Some("sess-1".into()),
                    )
                    .with_call_id("call-w-1"),
                )
                .await
        });
        let captured_req_id = match rx.recv().await.unwrap() {
            ClientEvent::ApprovalRequested {
                request_id,
                call_id,
                tool_name,
                responder,
                ..
            } => {
                assert_eq!(tool_name, "danger_write");
                assert_eq!(call_id.as_deref(), Some("call-w-1"));
                assert_eq!(pending.list_pending().len(), 1);
                responder
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap()
                    .send(ApprovalDecision::AllowOnce)
                    .unwrap();
                request_id
            }
            other => panic!("unexpected event: {other:?}"),
        };
        let write_out = fut.await.unwrap();
        assert!(!write_out.is_error);
        assert_eq!(write_out.content, "written");
        assert!(pending.list_pending().is_empty());

        match rx.recv().await.unwrap() {
            ClientEvent::ApprovalResolved { request_id } => {
                assert_eq!(request_id, captured_req_id);
            }
            other => panic!("expected ApprovalResolved, got {other:?}"),
        }
        assert!(sess.all_entries().iter().any(|entry| matches!(
            &entry.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::Approval {
                    tool_call_id,
                    tool,
                    decision,
                    resolved_by,
                }
            } if tool_call_id == "call-w-1"
                && tool == "danger_write"
                && decision == "allow_once"
                && resolved_by == "tui"
        )));

        // 4. 释放守卫后钩子失效，直接放行
        drop(hook_guard);
        let post_dispose = tools
            .execute(ToolExecution::new(
                "danger_write",
                json!({}),
                Some("sess-1".into()),
            ))
            .await;
        assert!(!post_dispose.is_error);
        assert_eq!(post_dispose.content, "written");
    }

    #[tokio::test]
    async fn pending_interactions_enforces_atomic_dual_source_preemption() {
        let tools = ToolRuntime::new();
        let sessions = SessionStore::new();
        let sess = sessions.create(Some("sess-race".into()));
        let _danger_g = tools
            .register(
                ToolDefinition::sync_tool("bash", "shell", json!({}), |_, _| {
                    Ok::<_, BoxError>("ran".to_string())
                })
                .with_annotations(ToolAnnotations {
                    read_only: false,
                    ..Default::default()
                }),
            )
            .unwrap();

        let active_sid = Arc::new(RwLock::new(sess.id.clone()));
        let snap = Arc::new(RwLock::new(ApprovalSnapshot {
            mode: ApprovalMode::Ask,
            always_allowed: HashSet::new(),
        }));
        let pending = Arc::new(PendingInteractions::new());
        let (client_tx, mut client_rx) = mpsc::unbounded_channel();
        let (rpc_tx, mut rpc_rx) = mpsc::unbounded_channel();

        let _guard = register_approval_hook_with_interactions(
            &tools,
            Some(Arc::clone(&sessions)),
            active_sid,
            snap,
            Arc::clone(&pending),
            move |ev| {
                let _ = client_tx.send(ev);
            },
            Some(Arc::new(move |notif| {
                let _ = rpc_tx.send(notif);
            })),
        );

        let tools_exec = Arc::clone(&tools);
        let exec_handle = tokio::spawn(async move {
            tools_exec
                .execute(
                    ToolExecution::new(
                        "bash",
                        json!({"command": "echo hi"}),
                        Some("sess-race".into()),
                    )
                    .with_call_id("call-race-1"),
                )
                .await
        });

        let dto = match rpc_rx.recv().await.unwrap() {
            RpcNotification::InteractionRequested(dto) => dto,
            other => panic!("unexpected rpc notif: {other:?}"),
        };
        assert_eq!(dto.call_id.as_deref(), Some("call-race-1"));
        assert_eq!(pending.list_pending().len(), 1);

        let tui_responder = match client_rx.recv().await.unwrap() {
            ClientEvent::ApprovalRequested {
                request_id,
                responder,
                ..
            } => {
                assert_eq!(request_id, dto.request_id);
                responder.lock().unwrap().take().unwrap()
            }
            other => panic!("unexpected client event: {other:?}"),
        };

        // RPC 率先抢占 AllowOnce；随后 TUI 发送 Deny 被原子判为过期竞争方
        let first = pending
            .resolve(
                &dto.request_id,
                ApprovalDecision::AllowOnce,
                ResolvedBy::Rpc,
            )
            .expect("first RPC resolution should win");
        assert_eq!(first.resolved_by, ResolvedBy::Rpc);
        assert!(pending.list_pending().is_empty());

        let second = pending.resolve(&dto.request_id, ApprovalDecision::Deny, ResolvedBy::Tui);
        assert_eq!(
            second,
            Err(ResolveInteractionError::AlreadyResolved(
                dto.request_id.clone()
            ))
        );

        // 即使 TUI 通道也发了 Deny，工具仍按首个胜出方（RPC 的 AllowOnce）执行成功
        let _ = tui_responder.send(ApprovalDecision::Deny);
        let out = exec_handle.await.unwrap();
        assert!(!out.is_error);
        assert_eq!(out.content, "ran");

        match rpc_rx.recv().await.unwrap() {
            RpcNotification::InteractionResolved(resolved) => {
                assert_eq!(resolved.request_id, dto.request_id);
                assert_eq!(resolved.decision, RpcApprovalDecision::AllowOnce);
                assert_eq!(resolved.resolved_by, ResolvedBy::Rpc);
            }
            other => panic!("expected InteractionResolved, got {other:?}"),
        }
        match client_rx.recv().await.unwrap() {
            ClientEvent::ApprovalResolved { request_id } => {
                assert_eq!(request_id, dto.request_id);
            }
            other => panic!("expected ApprovalResolved, got {other:?}"),
        }

        let approvals: Vec<_> = sess
            .all_entries()
            .into_iter()
            .filter(|e| {
                matches!(
                    e.event,
                    SessionEvent::Trace {
                        detail: SessionTraceEvent::Approval { .. }
                    }
                )
            })
            .collect();
        assert_eq!(approvals.len(), 1);
    }
}
