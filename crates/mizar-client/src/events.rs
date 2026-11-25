//! 客户端事件订阅、后端枚举投影、核心服务就绪等待与活跃任务代次控制器。

use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use cordis::{Context, CordisError, DisposableHandle, LogLevel, LogRecord};
use mizar_agent_loop::{
    AgentReasoningDeltaEvent, AgentStatus, AgentStatusEvent, AgentStepEndEvent,
    AgentStepFirstTokenEvent, AgentStepStartEvent, AgentTextDeltaEvent, AgentToolCallEvent,
    AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent, AgentTurnStartEvent,
    AgentUsageEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_STATUS, EVENT_AGENT_STEP_END,
    EVENT_AGENT_STEP_FIRST_TOKEN, EVENT_AGENT_STEP_START, EVENT_AGENT_TEXT_DELTA,
    EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT, EVENT_AGENT_TOOL_UPDATE, EVENT_AGENT_TURN_END,
    EVENT_AGENT_TURN_START, EVENT_AGENT_USAGE, StepOutcome, TurnEndReason,
};
use mizar_agent_spine::{
    CompactionEndEvent, CompactionProgressEvent, CompactionStartEvent, CompactionTrigger,
    EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS, EVENT_COMPACTION_START,
};
use mizar_client_core::{
    ClientEvent, UiAgentStatus, UiLogLevel, UiLogRecord, UiStepOutcome, UiTurnEndReason,
};
use mizar_llm::{ImageAttachment, LlmRuntime};
use mizar_session::{EVENT_SESSION_TITLE_UPDATED, SessionStore, SessionTitleUpdatedPayload};
use mizar_trajectory::{EVENT_TRAJECTORY_ANOMALY, TrajectoryAnomalyEvent};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// 等待核心后端服务就绪时的失败原因。
#[derive(Debug, Error)]
pub enum WaitServicesError {
    #[error("等待核心服务（agentLoop / sessions / llm）超时")]
    Timeout,
    #[error("等待核心服务探针通道提前关闭")]
    ProbeDropped,
    #[error(transparent)]
    Cordis(#[from] CordisError),
}

pub fn to_ui_status(status: AgentStatus) -> UiAgentStatus {
    match status {
        AgentStatus::Busy => UiAgentStatus::Busy,
        AgentStatus::Idle => UiAgentStatus::Idle,
    }
}

pub fn to_ui_turn_end_reason(reason: TurnEndReason) -> UiTurnEndReason {
    match reason {
        TurnEndReason::Completed => UiTurnEndReason::Completed,
        TurnEndReason::Truncated => UiTurnEndReason::Truncated,
        TurnEndReason::MaxSteps => UiTurnEndReason::MaxSteps,
        TurnEndReason::Cancelled => UiTurnEndReason::Cancelled,
        TurnEndReason::Error { message } => UiTurnEndReason::Error { message },
    }
}

pub fn to_ui_step_outcome(outcome: &StepOutcome) -> UiStepOutcome {
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

/// 事件循环内部统一 FIFO 消息信封：保证总线事件（如 `TurnEnd`）与后台任务结束事件（如 `RunFinished`）
/// 严格按投递顺序到达，同时为本地派生任务附带 `(run_id, session_id)` 代次标识，杜绝过期任务或旧会话污染新状态。
#[derive(Debug, Clone)]
pub enum LoopClientMsg {
    Bus(ClientEvent),
    Task {
        run_id: u64,
        session_id: String,
        event: ClientEvent,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTaskKind {
    AgentRun,
    LocalBash,
    ManualCompaction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingPrompt {
    pub text: String,
    pub images: Vec<ImageAttachment>,
}

impl PendingPrompt {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            images: Vec::new(),
        }
    }

    pub fn with_images(text: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self {
            text: text.into(),
            images,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActiveTask {
    pub run_id: u64,
    pub session_id: String,
    pub kind: ActiveTaskKind,
    pub cancel: CancellationToken,
    pub pending_steer: Option<PendingPrompt>,
    pub user_cancelled: bool,
}

/// 后台异步任务代次控制器：保证任何时刻最多只有一个活跃任务，并精确隔离旧 `run_id` / 旧会话遗留事件。
#[derive(Debug, Default)]
pub struct RunController {
    pub next_run_id: u64,
    pub active: Option<ActiveTask>,
}

impl RunController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self, session_id: String, kind: ActiveTaskKind) -> (u64, CancellationToken) {
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

    pub fn cancel_active(&mut self, steer: Option<PendingPrompt>) {
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

    pub fn abort_for_session_switch(&mut self) {
        if let Some(prev) = self.active.take() {
            prev.cancel.cancel();
        }
    }

    pub fn is_current(&self, run_id: u64, msg_session_id: &str, current_session_id: &str) -> bool {
        self.active.as_ref().is_some_and(|cur| {
            cur.run_id == run_id
                && cur.session_id == msg_session_id
                && msg_session_id == current_session_id
        })
    }

    pub fn cancel_token(&self) -> Option<&CancellationToken> {
        self.active.as_ref().map(|t| &t.cancel)
    }

    pub fn active_task(&self) -> Option<&ActiveTask> {
        self.active.as_ref()
    }

    pub fn finish_if_current(
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
    session_id_rx: Arc<RwLock<String>>,
    tx: mpsc::UnboundedSender<ClientEvent>,
) -> Result<Vec<DisposableHandle>, CordisError> {
    subscribe_session_events_into(ctx, session_id_rx, move |ev| {
        let _ = tx.send(ev);
    })
}

/// 订阅当前会话相关的 Agent、压缩、轨迹与标题事件，并通过自定义闭包投递。
pub fn subscribe_session_events_into<F>(
    ctx: &Context,
    session_id_rx: Arc<RwLock<String>>,
    send: F,
) -> Result<Vec<DisposableHandle>, CordisError>
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    let mut handles = Vec::new();
    let is_current = move |sid: &str| *session_id_rx.read().unwrap() == sid;

    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STATUS, move |ev: &AgentStatusEvent| {
            if check(&ev.session.id) {
                send(ClientEvent::Status(to_ui_status(ev.status)));
            }
        })?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(
            ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                if check(&ev.session.id) {
                    send(ClientEvent::TurnStart { turn: ev.turn });
                }
            })?,
        );
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
            if check(&ev.session.id) {
                send(ClientEvent::TurnEnd {
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
                    send(ClientEvent::StepStart { ts_ms: ev.ts_ms });
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
                    send(ClientEvent::FirstToken { ts_ms: ev.ts_ms });
                }
            },
        )?);
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        handles.push(ctx.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
            if check(&ev.session.id) {
                send(ClientEvent::StepEnd {
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
                    send(ClientEvent::TextDelta {
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
                    send(ClientEvent::ReasoningDelta {
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
                    send(ClientEvent::ToolCall(ev.call.clone()));
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
                    send(ClientEvent::ToolUpdate {
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
                    send(ClientEvent::ToolResult {
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
                send(ClientEvent::Usage(ev.usage));
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
                    send(ClientEvent::Anomaly(ev.finding.summary.clone()));
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
                    send(ClientEvent::CompactionStarted {
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
                    send(ClientEvent::CompactionProgress {
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
                    send(ClientEvent::CompactionFinished {
                        outcome: ev.outcome.as_str().to_string(),
                        tokens_before: Some(ev.tokens_before),
                        tokens_after: Some(ev.tokens_after),
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
                    send(ClientEvent::SessionTitleUpdated(ev.title.clone()));
                }
            },
        )?);
    }

    Ok(handles)
}

/// 等待 `agentLoop`、`sessions`、`llm` 核心服务以及所有已挂载插件就绪（消除异步挂载竞态）。
pub async fn wait_for_services(ctx: &Context) -> Result<(), WaitServicesError> {
    if ctx
        .get::<mizar_agent_loop::AgentLoop>("agentLoop")
        .is_none()
        || ctx.get::<SessionStore>("sessions").is_none()
        || ctx.get::<LlmRuntime>("llm").is_none()
    {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let tx = Arc::new(Mutex::new(Some(tx)));
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
        match wait_res {
            Ok(Ok(())) => {}
            Ok(Err(_)) => return Err(WaitServicesError::ProbeDropped),
            Err(_) => return Err(WaitServicesError::Timeout),
        }
    }

    // 插件在 apply() 内通过 ctx.try_inject 派生的子 fiber 可能在首轮快照之后出现，
    // 需迭代直到不再产生新 fiber 且全部落稳（PENDING 态子 fiber 的 inertia 为 false，不会阻塞）。
    let mut seen = std::collections::HashSet::new();
    loop {
        let fibers: Vec<cordis::Fiber> = ctx
            .registry()
            .values()
            .into_iter()
            .flat_map(|rt| rt.fibers())
            .collect();
        let mut grew = false;
        for f in fibers {
            if let Some(uid) = f.uid()
                && seen.insert(uid)
            {
                grew = true;
            }
            let _ = f.wait_settled().await;
        }
        if !grew {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_controller_tracks_generations_and_session_isolation() {
        let mut ctrl = RunController::new();
        let (rid1, cancel1) = ctrl.start("sess-a".into(), ActiveTaskKind::AgentRun);
        let (rid2, cancel2) = ctrl.start("sess-a".into(), ActiveTaskKind::LocalBash);
        assert!(cancel1.is_cancelled());
        assert!(!cancel2.is_cancelled());
        assert!(!ctrl.is_current(rid1, "sess-a", "sess-a"));
        assert!(ctrl.is_current(rid2, "sess-a", "sess-a"));
        assert!(!ctrl.is_current(rid2, "sess-a", "sess-b"));

        // 旧 run_id 或不匹配的 kind 均不得结束当前任务
        assert!(
            ctrl.finish_if_current(
                Some(rid1),
                Some("sess-a"),
                "sess-a",
                ActiveTaskKind::AgentRun
            )
            .is_none()
        );
        assert!(
            ctrl.finish_if_current(
                Some(rid2),
                Some("sess-a"),
                "sess-a",
                ActiveTaskKind::AgentRun
            )
            .is_none()
        );

        let finished = ctrl
            .finish_if_current(
                Some(rid2),
                Some("sess-a"),
                "sess-a",
                ActiveTaskKind::LocalBash,
            )
            .unwrap();
        assert_eq!(finished.run_id, rid2);
        assert!(ctrl.cancel_token().is_none());

        // 取消时记录 steer 并置位 cancel token
        let (_rid3, cancel3) = ctrl.start("sess-b".into(), ActiveTaskKind::AgentRun);
        ctrl.cancel_active(Some(PendingPrompt::text("steer now")));
        assert!(cancel3.is_cancelled());
        let active = ctrl.active_task().unwrap();
        assert!(!active.user_cancelled);
        assert_eq!(
            active.pending_steer.as_ref().map(|p| p.text.as_str()),
            Some("steer now")
        );

        ctrl.abort_for_session_switch();
        assert!(ctrl.active_task().is_none());
    }

    #[tokio::test]
    async fn wait_for_services_settles_late_try_inject_child_without_blocking_on_pending_optional()
    {
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicBool, Ordering};

        use cordis::PluginDef;
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

        // 永不就绪的可选依赖（保持 PENDING）不应阻塞 wait_for_services
        let _pending_optional = ctx
            .try_inject(["nonExistentOptionalService"], |_| async {
                Ok::<_, CordisError>(())
            })
            .unwrap();

        // 在异步插件体内延迟派生 try_inject 子 fiber，模拟 CodingToolsPlugin / McpManager 挂接 systemPrompt
        let child_settled = Arc::new(AtomicBool::new(false));
        let child_settled_cb = Arc::clone(&child_settled);
        ctx.plugin(
            PluginDef::sync("late-try-inject-spawner", move |c, ()| {
                let flag = Arc::clone(&child_settled_cb);
                c.try_inject(["systemPrompt"], move |_| {
                    let inner = Arc::clone(&flag);
                    async move {
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        inner.store(true, Ordering::SeqCst);
                        Ok::<_, CordisError>(())
                    }
                })?;
                Ok::<_, CordisError>(())
            }),
            (),
        );

        wait_for_services(&ctx).await.unwrap();
        assert!(
            child_settled.load(Ordering::SeqCst),
            "wait_for_services returned before dynamically spawned try_inject child settled"
        );

        ctx.fiber().dispose().await;
    }
}
