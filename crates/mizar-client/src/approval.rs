//! 工具执行前危险操作审批门禁（`tools.pre_execute_hook()` 瀑布流钩子与 `install_approval_gate`）。

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use mizar_core::spine::MizarRuntime;
use mizar_session::SessionStore;
use mizar_tools::{PreExecuteDecision, ToolExecution, ToolRuntime};
use mizar_types::client::{ApprovalDecision, ApprovalMode, ClientEvent, truncate_display};
use mizar_types::{BoxFuture, HookGuard, Next};
use tokio::sync::mpsc;

/// 运行期共享审批策略快照（供异步工具执行前瀑布流无锁检查）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalSnapshot {
    pub mode: ApprovalMode,
    pub always_allowed: HashSet<String>,
}

/// 危险操作审批门禁配置：关联活跃会话 ID、审批策略快照与事件回调。
#[derive(Clone)]
pub struct ApprovalGateConfig {
    pub active_session_id: Arc<RwLock<String>>,
    pub snapshot: Arc<RwLock<ApprovalSnapshot>>,
    pub emit: Arc<dyn Fn(ClientEvent) + Send + Sync>,
}

impl fmt::Debug for ApprovalGateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalGateConfig")
            .field("active_session_id", &self.active_session_id)
            .field("snapshot", &self.snapshot)
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
        }
    }
}

/// 将交互式工具执行前审批门禁挂载到 `MizarRuntime` 上，返回 RAII `HookGuard`。
pub fn install_approval_gate(runtime: &MizarRuntime, config: ApprovalGateConfig) -> HookGuard {
    let emit = Arc::clone(&config.emit);
    register_approval_hook_with(
        &runtime.tools,
        Some(Arc::clone(&runtime.sessions)),
        config.active_session_id,
        config.snapshot,
        move |ev| emit(ev),
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
    let tools_ref = Arc::clone(tools);
    tools.pre_execute_hook().register(
        move |exec: ToolExecution,
              next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let emit = emit.clone();
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
                if let Some(sid) = exec.session_id.as_deref() {
                    let active_id = active_sid.read().unwrap().clone();
                    if sid != active_id {
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
                }

                let args_preview = serde_json::to_string_pretty(&exec.args)
                    .unwrap_or_else(|_| exec.args.to_string());
                let args_preview = truncate_display(&args_preview.replace('\n', " "), 400);
                let (tx, rx) = futures::channel::oneshot::channel();
                emit(ClientEvent::ApprovalRequested {
                    tool_name: exec.name.clone(),
                    args_preview,
                    responder: Arc::new(Mutex::new(Some(tx))),
                });

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

#[cfg(test)]
mod tests {
    use mizar_tools::{ToolAnnotations, ToolDefinition};
    use mizar_types::BoxError;
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn approval_gate_enforces_ask_mode_exemptions_and_cleanup() {
        let tools = ToolRuntime::new();
        let sessions = SessionStore::new();
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

        let active_sid = Arc::new(RwLock::new("sess-1".to_string()));
        let snap = Arc::new(RwLock::new(ApprovalSnapshot {
            mode: ApprovalMode::Ask,
            always_allowed: HashSet::new(),
        }));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let config = ApprovalGateConfig::new(Arc::clone(&active_sid), Arc::clone(&snap), tx);
        let emit = Arc::clone(&config.emit);
        let hook_guard = register_approval_hook_with(
            &tools,
            Some(Arc::clone(&sessions)),
            config.active_session_id,
            config.snapshot,
            move |ev| emit(ev),
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

        // 3. 活跃会话的写工具触发审批请求，允许后正常放行
        let tools_exec = Arc::clone(&tools);
        let fut = tokio::spawn(async move {
            tools_exec
                .execute(ToolExecution::new(
                    "danger_write",
                    json!({"path": "a.txt"}),
                    Some("sess-1".into()),
                ))
                .await
        });
        match rx.recv().await.unwrap() {
            ClientEvent::ApprovalRequested {
                tool_name,
                responder,
                ..
            } => {
                assert_eq!(tool_name, "danger_write");
                responder
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap()
                    .send(ApprovalDecision::AllowOnce)
                    .unwrap();
            }
            other => panic!("unexpected event: {other:?}"),
        }
        let write_out = fut.await.unwrap();
        assert!(!write_out.is_error);
        assert_eq!(write_out.content, "written");

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
}
