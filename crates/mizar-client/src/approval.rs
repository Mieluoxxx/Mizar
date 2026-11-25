//! 工具执行前危险操作审批门禁（`tools/pre-execute` 瀑布流钩子与 `ApprovalGatePlugin`）。

use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, DisposableHandle, Inject, Next, Plugin,
};
use mizar_client_core::{
    ApprovalDecision, ApprovalMode, ClientEvent, PLUGIN_APPROVAL_GATE, truncate_display,
};
use mizar_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolExecution, ToolRuntime, ToolsContextExt,
};
use tokio::sync::mpsc;

/// 运行期共享审批策略快照（供异步工具执行前瀑布流无锁检查）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ApprovalSnapshot {
    pub mode: ApprovalMode,
    pub always_allowed: HashSet<String>,
}

/// `ApprovalGatePlugin` 插件配置：关联活跃会话 ID、审批策略快照与事件回调。
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

/// 官方危险工具审批门禁 Cordis 插件（依赖 `tools` 服务，在卸载时自动撤销拦截钩子）。
#[derive(Default, Clone, Copy)]
pub struct ApprovalGatePlugin;

impl Plugin for ApprovalGatePlugin {
    type Config = ApprovalGateConfig;

    fn name(&self) -> Option<&str> {
        Some(PLUGIN_APPROVAL_GATE)
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let _ = ctx.tools()?;
            let c = ctx.clone();
            ctx.effect("mizar-client-approval-gate", move || {
                let emit = Arc::clone(&config.emit);
                let handle = register_approval_hook_with(
                    &c,
                    config.active_session_id,
                    config.snapshot,
                    move |ev| emit(ev),
                )?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    handle.dispose().await;
                })))
            })?;
            Ok(None)
        })
    }
}

/// 判断给定工具在 `Ask` 模式下是否需要向用户请求交互审批（只读工具自动豁免）。
pub fn tool_needs_approval(ctx: &Context, tool_name: &str) -> bool {
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
            | "recall"
            | "mcp_list_resources"
            | "mcp_read_resource"
    )
}

/// 注册 `tools/pre-execute` 审批瀑布流钩子，并通过 `mpsc::UnboundedSender<ClientEvent>` 投递弹窗请求。
pub fn register_approval_hook(
    ctx: &Context,
    tx: mpsc::UnboundedSender<ClientEvent>,
    active_sid: Arc<RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
) -> Result<DisposableHandle, CordisError> {
    register_approval_hook_with(ctx, active_sid, approval_snap, move |ev| {
        let _ = tx.send(ev);
    })
}

/// 注册 `tools/pre-execute` 审批瀑布流钩子，向自定义回调投递 `ClientEvent::ApprovalRequested`。
pub fn register_approval_hook_with<F>(
    ctx: &Context,
    active_sid: Arc<RwLock<String>>,
    approval_snap: Arc<RwLock<ApprovalSnapshot>>,
    emit: F,
) -> Result<DisposableHandle, CordisError>
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    let ctx_lookup = ctx.clone();
    ctx.on_waterfall(
        EVENT_TOOLS_PRE_EXECUTE,
        move |exec: ToolExecution,
              next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let emit = emit.clone();
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
    use cordis::Service;
    use mizar_tools::{ToolAnnotations, ToolDefinition};
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn approval_gate_plugin_enforces_ask_mode_exemptions_and_cleanup() {
        let ctx = Context::new();
        ctx.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        let tools = ctx.tools().unwrap();
        tools
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
        tools
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
        let plugin_fiber = ctx
            .plugin(
                ApprovalGatePlugin,
                ApprovalGateConfig::new(Arc::clone(&active_sid), Arc::clone(&snap), tx),
            )
            .await
            .unwrap();

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

        // 2. 非当前活跃会话的工具调用被直接拦截
        let stale_out = tools
            .execute(ToolExecution::new(
                "danger_write",
                json!({}),
                Some("sess-old".into()),
            ))
            .await;
        assert!(stale_out.is_error);
        assert!(stale_out.content.contains("会话已切换"));

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

        // 4. 卸载插件后钩子失效，直接放行
        plugin_fiber.dispose().await;
        let post_dispose = tools
            .execute(ToolExecution::new(
                "danger_write",
                json!({}),
                Some("sess-1".into()),
            ))
            .await;
        assert!(!post_dispose.is_error);
        assert_eq!(post_dispose.content, "written");

        ctx.fiber().dispose().await;
    }
}
