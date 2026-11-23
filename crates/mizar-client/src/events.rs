//! 客户端事件订阅、后端枚举投影与活跃任务代次控制器。

use std::sync::{Arc, RwLock};

use mizar_core::agent_loop::{
    AgentEvent, AgentLoop, AgentStatus, DeliveryMode, InboxPushResult, StepOutcome, TurnEndReason,
    TurnInput,
};
use mizar_core::spine::{CompactionTrigger, MizarRuntime};
use mizar_types::HookGuard;
use mizar_types::client::{ClientEvent, UiAgentStatus, UiStepOutcome, UiTurnEndReason};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

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

#[derive(Debug, Clone)]
pub struct ActiveTask {
    pub run_id: u64,
    pub session_id: String,
    pub kind: ActiveTaskKind,
    pub cancel: CancellationToken,
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
            user_cancelled: false,
        });
        (run_id, cancel)
    }

    /// 显式强制打断当前任务（触发 `CancellationToken` 中止流式生成或工具执行）。
    pub fn cancel_active(&mut self) {
        if let Some(task) = self.active.as_mut() {
            task.user_cancelled = true;
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

/// 向活跃会话投递温和的步间插话（`DeliveryMode::Steer`），不中止正在执行的工具与当前轮。
pub fn steer_session(
    agent_loop: &AgentLoop,
    session_id: &str,
    input: impl Into<TurnInput>,
) -> InboxPushResult {
    agent_loop
        .inbox()
        .push(session_id, DeliveryMode::Steer, input.into())
}

/// 订阅当前会话相关的 Agent 与轨迹事件并投递到 `tx`。
pub fn subscribe_session_events(
    runtime: &MizarRuntime,
    session_id_rx: Arc<RwLock<String>>,
    tx: mpsc::UnboundedSender<ClientEvent>,
) -> HookGuard {
    subscribe_session_events_into(runtime, session_id_rx, move |ev| {
        let _ = tx.send(ev);
    })
}

/// 订阅当前会话相关的 Agent、压缩、轨迹与标题事件，并通过自定义闭包投递。
pub fn subscribe_session_events_into<F>(
    runtime: &MizarRuntime,
    session_id_rx: Arc<RwLock<String>>,
    send: F,
) -> HookGuard
where
    F: Fn(ClientEvent) + Clone + Send + Sync + 'static,
{
    let mut guards = Vec::new();
    let is_current = move |sid: &str| *session_id_rx.read().unwrap() == sid;

    {
        let send = send.clone();
        let check = is_current.clone();
        guards.push(runtime.agent_loop.on_event(move |ev| match ev {
            AgentEvent::Status(ev) if check(&ev.session.id) => {
                send(ClientEvent::Status(to_ui_status(ev.status)));
            }
            AgentEvent::TurnStart(ev) if check(&ev.session.id) => {
                send(ClientEvent::TurnStart { turn: ev.turn });
            }
            AgentEvent::TurnEnd(ev) if check(&ev.session.id) => {
                send(ClientEvent::TurnEnd {
                    reason: to_ui_turn_end_reason(ev.reason.clone()),
                });
            }
            AgentEvent::StepStart(ev) if check(&ev.session.id) => {
                send(ClientEvent::StepStart { ts_ms: ev.ts_ms });
            }
            AgentEvent::StepFirstToken(ev) if check(&ev.session.id) => {
                send(ClientEvent::FirstToken { ts_ms: ev.ts_ms });
            }
            AgentEvent::StepEnd(ev) if check(&ev.session.id) => {
                send(ClientEvent::StepEnd {
                    step: ev.step,
                    ts_ms: ev.ts_ms,
                    outcome: to_ui_step_outcome(&ev.outcome),
                    usage: ev.usage,
                });
            }
            AgentEvent::TextDelta(ev) if check(&ev.session.id) => {
                send(ClientEvent::TextDelta {
                    text: ev.text.clone(),
                    ts_ms: ev.ts_ms,
                });
            }
            AgentEvent::ReasoningDelta(ev) if check(&ev.session.id) => {
                send(ClientEvent::ReasoningDelta {
                    text: ev.text.clone(),
                    ts_ms: ev.ts_ms,
                });
            }
            AgentEvent::ToolCall(ev) if check(&ev.session.id) => {
                send(ClientEvent::ToolCall(ev.call.clone()));
            }
            AgentEvent::ToolUpdate(ev) if check(&ev.session.id) => {
                send(ClientEvent::ToolUpdate {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    preview: ev.preview.clone(),
                });
            }
            AgentEvent::ToolResult(ev) if check(&ev.session.id) => {
                send(ClientEvent::ToolResult {
                    call_id: ev.tool_call_id.clone(),
                    name: ev.name.clone(),
                    content: ev.content.clone(),
                    is_error: ev.is_error,
                    blocked: ev.blocked,
                    details: ev.details.clone(),
                });
            }
            AgentEvent::Usage(ev) if check(&ev.session.id) => {
                send(ClientEvent::Usage(ev.usage));
            }
            AgentEvent::Status(_)
            | AgentEvent::Request(_)
            | AgentEvent::TurnStart(_)
            | AgentEvent::TurnInput(_)
            | AgentEvent::TurnEnd(_)
            | AgentEvent::StepStart(_)
            | AgentEvent::StepFirstToken(_)
            | AgentEvent::StepEnd(_)
            | AgentEvent::TextDelta(_)
            | AgentEvent::ReasoningDelta(_)
            | AgentEvent::ToolCall(_)
            | AgentEvent::ToolUpdate(_)
            | AgentEvent::ToolResult(_)
            | AgentEvent::Usage(_) => {}
        }));
    }
    if let Some(traj) = &runtime.trajectory {
        let send = send.clone();
        let check = is_current.clone();
        guards.push(traj.on_anomaly(move |ev| {
            if check(&ev.session_id) {
                send(ClientEvent::Anomaly(ev.finding.summary.clone()));
            }
        }));
    }
    if let Some(comp) = &runtime.compaction {
        {
            let send = send.clone();
            let check = is_current.clone();
            guards.push(comp.on_start(move |ev| {
                if check(&ev.session_id) {
                    send(ClientEvent::CompactionStarted {
                        trigger: ev.trigger.as_str().to_string(),
                    });
                }
            }));
        }
        {
            let send = send.clone();
            let check = is_current.clone();
            guards.push(comp.on_progress(move |ev| {
                if check(&ev.session_id) {
                    send(ClientEvent::CompactionProgress {
                        stage: ev.stage.clone(),
                        delta_chars: ev.delta_chars,
                    });
                }
            }));
        }
        {
            let send = send.clone();
            let check = is_current.clone();
            guards.push(comp.on_end(move |ev| {
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
            }));
        }
    }
    {
        let send = send.clone();
        let check = is_current.clone();
        guards.push(runtime.sessions.on_title_updated(move |ev| {
            if check(&ev.session.id) {
                send(ClientEvent::SessionTitleUpdated(ev.title.clone()));
            }
        }));
    }
    if let Some(sub) = &runtime.subagent {
        let send = send;
        let check = is_current;
        guards.push(sub.on_state(move |ev| {
            if check(&ev.parent_session_id) {
                send(ClientEvent::SubagentState {
                    running_count: ev.running_count,
                    notice: ev.notice.clone(),
                });
            }
        }));
    }

    HookGuard::group(guards)
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

        // 显式取消置位 user_cancelled 与 cancel token
        let (_rid3, cancel3) = ctrl.start("sess-b".into(), ActiveTaskKind::AgentRun);
        ctrl.cancel_active();
        assert!(cancel3.is_cancelled());
        let active = ctrl.active_task().unwrap();
        assert!(active.user_cancelled);

        ctrl.abort_for_session_switch();
        assert!(ctrl.active_task().is_none());
    }
}
