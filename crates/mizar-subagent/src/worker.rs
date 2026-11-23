//! 后台 Subagent 执行器、内嵌零 Token 活性看门狗、超时修复与无竞态续跑循环。

use std::sync::Arc;
use std::time::{Duration, Instant};

use mizar_agent_loop::{
    AgentError, AgentLoop, DeliveryMode, RunOptions, TurnEndReason, TurnInput, TurnRunReport,
};
use mizar_llm::LlmErrorCode;
use mizar_session::Session;
use tokio_util::sync::CancellationToken;

use crate::guardrails::{ChildSessionGuard, shorten_prompt};
use crate::tasks::{ChildActivityTracker, SubagentRunStatus, SubagentStateEvent, WorkerTurnAction};
use crate::{SubagentMode, SubagentRuntime};

pub(crate) struct BackgroundWorkerArgs {
    pub run_id: u64,
    pub parent_id: String,
    pub assigned_name: String,
    pub child_session: Arc<Session>,
    pub child_guard: ChildSessionGuard,
    pub child_cancel: CancellationToken,
    pub activity: Arc<ChildActivityTracker>,
    pub agent_loop: Arc<AgentLoop>,
    pub initial_prompt: String,
    pub run_opts: RunOptions,
    pub mode: SubagentMode,
    pub max_steps: usize,
    pub timeout_secs: u64,
    pub stall_timeout_secs: u64,
    pub max_watchdog_alerts: usize,
}

enum InnerTurnOutcome {
    Disposed,
    Finished(Result<TurnRunReport, AgentError>),
    TimedOut,
}

impl SubagentRuntime {
    pub(crate) fn spawn_background_worker(self: &Arc<Self>, args: BackgroundWorkerArgs) {
        let rt = Arc::clone(self);
        tokio::spawn(async move {
            let BackgroundWorkerArgs {
                run_id,
                parent_id,
                assigned_name,
                child_session,
                child_guard,
                child_cancel,
                activity,
                agent_loop,
                initial_prompt,
                run_opts,
                mode,
                max_steps,
                timeout_secs,
                stall_timeout_secs,
                max_watchdog_alerts,
            } = args;

            let child_id = child_session.id.clone();
            let mut next_input: TurnInput = TurnInput::user(initial_prompt);

            let final_settlement: Option<(SubagentRunStatus, usize, u64, String, bool)>;

            loop {
                let hard_deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
                let inner_outcome = {
                    let mut alerts_sent = 0usize;
                    let mut last_alert_at: Option<Instant> = None;

                    let run_fut = agent_loop.run_with_report(
                        Arc::clone(&child_session),
                        next_input,
                        child_cancel.clone(),
                        run_opts.clone(),
                    );
                    tokio::pin!(run_fut);

                    loop {
                        let watchdog_sleep =
                            if stall_timeout_secs > 0 && alerts_sent < max_watchdog_alerts {
                                let stall = Duration::from_secs(stall_timeout_secs);
                                let base = match last_alert_at {
                                    Some(t) => activity.last_activity_instant().max(t),
                                    None => activity.last_activity_instant(),
                                };
                                let wake_at = base.checked_add(stall).unwrap_or(base);
                                let remaining = wake_at.saturating_duration_since(Instant::now());
                                Some(tokio::time::sleep(remaining))
                            } else {
                                None
                            };

                        let watchdog_fut = async {
                            match watchdog_sleep {
                                Some(sleep) => sleep.await,
                                None => std::future::pending::<()>().await,
                            }
                        };

                        tokio::select! {
                            biased;
                            _ = rt.dispose_token.cancelled() => {
                                break InnerTurnOutcome::Disposed;
                            }
                            res = &mut run_fut => {
                                break InnerTurnOutcome::Finished(res);
                            }
                            _ = tokio::time::sleep_until(hard_deadline) => {
                                child_cancel.cancel();
                                let _ = tokio::time::timeout(Duration::from_millis(200), &mut run_fut).await;
                                break InnerTurnOutcome::TimedOut;
                            }
                            _ = watchdog_fut => {
                                let stall = Duration::from_secs(stall_timeout_secs);
                                let idle = activity.idle_duration();
                                let since_last_alert = last_alert_at.map_or(stall, |t| t.elapsed());
                                if idle >= stall
                                    && since_last_alert >= stall
                                    && !child_cancel.is_cancelled()
                                    && !rt.dispose_token.is_cancelled()
                                {
                                    alerts_sent += 1;
                                    last_alert_at = Some(Instant::now());
                                    rt.fire_watchdog_alert(
                                        run_id,
                                        &parent_id,
                                        &assigned_name,
                                        &child_id,
                                        mode,
                                        &activity,
                                        &agent_loop,
                                        alerts_sent,
                                        max_watchdog_alerts,
                                    );
                                } else if !child_cancel.is_cancelled() && !rt.dispose_token.is_cancelled() {
                                    // 亚毫秒精度边界微小余量，小睡 1ms 防 0-duration 忙循环
                                    let wait_left = stall.saturating_sub(idle.min(since_last_alert));
                                    if wait_left.is_zero() {
                                        tokio::time::sleep(Duration::from_millis(1)).await;
                                    }
                                }
                            }
                        }
                    }
                };

                match inner_outcome {
                    InnerTurnOutcome::Disposed => {
                        child_cancel.cancel();
                        agent_loop.inbox().clear(&child_id);
                        drop(child_guard);
                        let _ = child_session.repair_dangling_tool_calls();
                        let _ = rt.tasks.lock().unwrap().mark_settled(
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            SubagentRunStatus::Stopped,
                            0,
                            Some("子代理插件已卸载".into()),
                        );
                        return;
                    }
                    InnerTurnOutcome::TimedOut => {
                        agent_loop.inbox().clear(&child_id);
                        drop(child_guard);
                        let _ = child_session.repair_dangling_tool_calls();
                        let body = format!("[Subagent timed out after {timeout_secs}s limit]");
                        let settled = rt.tasks.lock().unwrap().mark_settled(
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            SubagentRunStatus::TimedOut,
                            0,
                            Some(shorten_prompt(&body, 120)),
                        );
                        final_settlement =
                            settled.map(|(was_stop, total_steps, _, elapsed_secs)| {
                                (
                                    SubagentRunStatus::TimedOut,
                                    total_steps,
                                    elapsed_secs,
                                    body,
                                    was_stop,
                                )
                            });
                        break;
                    }
                    InnerTurnOutcome::Finished(Ok(report))
                        if !child_cancel.is_cancelled()
                            && !rt.dispose_token.is_cancelled()
                            && matches!(
                                report.reason,
                                TurnEndReason::Completed
                                    | TurnEndReason::MaxSteps
                                    | TurnEndReason::Truncated
                            ) =>
                    {
                        let (status_if_settled, body) = match &report.reason {
                            TurnEndReason::Completed => {
                                let text = if report.text.trim().is_empty() {
                                    format!(
                                        "(Completed in {} steps with no text output)",
                                        report.steps
                                    )
                                } else {
                                    report.text.clone()
                                };
                                (SubagentRunStatus::Completed, text)
                            }
                            TurnEndReason::MaxSteps => (
                                SubagentRunStatus::MaxSteps,
                                format!(
                                    "[Reached max_steps limit ({}/{})]\n{}",
                                    report.steps, max_steps, report.text
                                ),
                            ),
                            TurnEndReason::Truncated => (
                                SubagentRunStatus::Truncated,
                                format!(
                                    "[Output truncated at max_tokens ({} steps)]\n{}",
                                    report.steps, report.text
                                ),
                            ),
                            _ => unreachable!(),
                        };

                        let turn_steps = report.steps;
                        let summary_preview = shorten_prompt(&body, 120);
                        let decision = {
                            let mut reg = rt.tasks.lock().unwrap();
                            reg.begin_normal_finish_or_continue(
                                &parent_id,
                                &assigned_name,
                                &child_id,
                                run_id,
                                turn_steps,
                                status_if_settled,
                                Some(summary_preview),
                                agent_loop.inbox(),
                                |txt| shorten_prompt(txt, 48),
                            )
                        };
                        match decision {
                            Some(WorkerTurnAction::Continue {
                                next_input: queued_input,
                            }) => {
                                next_input = queued_input;
                                continue;
                            }
                            Some(WorkerTurnAction::SettledWaitGuard { was_stop }) => {
                                // 先于释放 active_run_id 彻底卸除旧守卫与 tools 策略，零纳秒并发窗
                                drop(child_guard);
                                let finished = rt.tasks.lock().unwrap().finish_active_run(
                                    &parent_id,
                                    &assigned_name,
                                    &child_id,
                                    run_id,
                                );
                                final_settlement =
                                    finished.map(|(w, total_steps, _, elapsed_secs)| {
                                        (
                                            if was_stop || w {
                                                SubagentRunStatus::Stopped
                                            } else {
                                                status_if_settled
                                            },
                                            total_steps,
                                            elapsed_secs,
                                            body,
                                            was_stop || w,
                                        )
                                    });
                                break;
                            }
                            None => {
                                drop(child_guard);
                                return;
                            }
                        }
                    }
                    InnerTurnOutcome::Finished(other) => {
                        agent_loop.inbox().clear(&child_id);
                        drop(child_guard);
                        let _ = child_session.repair_dangling_tool_calls();

                        let (status, turn_steps, body) = match other {
                            Ok(report) => match report.reason {
                                TurnEndReason::Cancelled => (
                                    SubagentRunStatus::Stopped,
                                    report.steps,
                                    "[Subagent cancelled]".to_string(),
                                ),
                                TurnEndReason::Error { message } => (
                                    SubagentRunStatus::Failed,
                                    report.steps,
                                    format!(
                                        "[Subagent failed after {} steps: {message}]",
                                        report.steps
                                    ),
                                ),
                                TurnEndReason::Completed
                                | TurnEndReason::MaxSteps
                                | TurnEndReason::Truncated => (
                                    SubagentRunStatus::Stopped,
                                    report.steps,
                                    "[Subagent cancelled]".to_string(),
                                ),
                            },
                            Err(AgentError::Llm(err))
                                if err.code == LlmErrorCode::Cancelled
                                    || child_cancel.is_cancelled() =>
                            {
                                (
                                    SubagentRunStatus::Stopped,
                                    0,
                                    "[Subagent cancelled]".to_string(),
                                )
                            }
                            Err(err) => (
                                SubagentRunStatus::Failed,
                                0,
                                format!("[Subagent failed: {err}]"),
                            ),
                        };
                        let settled = rt.tasks.lock().unwrap().mark_settled(
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            status,
                            turn_steps,
                            Some(shorten_prompt(&body, 120)),
                        );
                        final_settlement =
                            settled.map(|(was_stop, total_steps, _, elapsed_secs)| {
                                (status, total_steps, elapsed_secs, body, was_stop)
                            });
                        break;
                    }
                }
            }

            let Some((status, steps, elapsed_secs, notice_body, was_stop)) = final_settlement
            else {
                return;
            };
            if was_stop || rt.dispose_token.is_cancelled() {
                return;
            }

            let inbox_text = format!(
                "Subagent \"{assigned_name}\" {} (mode={}, steps={steps}, elapsed={elapsed_secs}s, child_session={child_id}):\n\n{notice_body}",
                status.as_str(),
                mode.as_str()
            );
            let push_res = agent_loop.inbox().push_appending(
                &parent_id,
                DeliveryMode::FollowUp,
                TurnInput::notice(format!("subagent:{assigned_name}"), inbox_text),
                "subagent:settled",
                "subagent:batch",
            );
            let ui_notice = format_background_ui_notice(
                &assigned_name,
                status,
                steps,
                max_steps,
                elapsed_secs,
                timeout_secs,
                push_res.degraded,
            );
            rt.emit_state_event(SubagentStateEvent {
                parent_session_id: parent_id,
                name: assigned_name,
                child_session_id: child_id,
                mode,
                status,
                running_count: 0,
                notice: Some(ui_notice),
            });
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn fire_watchdog_alert(
        &self,
        run_id: u64,
        parent_id: &str,
        assigned_name: &str,
        child_id: &str,
        mode: SubagentMode,
        activity: &ChildActivityTracker,
        agent_loop: &AgentLoop,
        alerts_sent: usize,
        max_watchdog_alerts: usize,
    ) {
        let recorded = self.tasks.lock().unwrap().record_watchdog_alert(
            parent_id,
            assigned_name,
            child_id,
            run_id,
        );
        let Some((_, running_count)) = recorded else {
            return;
        };

        let (step, last_tool, idle_secs) = activity.snapshot();
        let elapsed_secs = self
            .tasks
            .lock()
            .unwrap()
            .snapshots(parent_id, Some(assigned_name))
            .into_iter()
            .find(|s| s.child_session_id == child_id)
            .map_or(idle_secs, |s| s.elapsed_secs);

        let tool_desc = last_tool
            .as_deref()
            .map(|t| format!(", last_tool=\"{t}\""))
            .unwrap_or_default();
        let ui_tool = last_tool
            .as_deref()
            .map(|t| format!("：{t}"))
            .unwrap_or_default();

        let watchdog_text = format!(
            "Subagent watchdog notice ({alerts_sent}/{max_watchdog_alerts}): subagent \"{assigned_name}\" (mode={}, child_session={child_id}) has produced no new activity for {idle_secs}s (elapsed={elapsed_secs}s, step={step}{tool_desc}).\n\
Check whether this task is legitimately slow or stalled: inform the user briefly if still expected to finish, steer/clarify with subagent(action=\"send\", name=\"{assigned_name}\", ...), or cancel via subagent(action=\"stop\", name=\"{assigned_name}\").",
            mode.as_str()
        );

        let inbox = agent_loop.inbox();
        let has_pending_wd = inbox.has_pending_followup_key(parent_id, "subagent:watchdog");
        let remaining = inbox.remaining_auto_wakes(parent_id);
        let wd_mode = if has_pending_wd || remaining >= 2 {
            DeliveryMode::FollowUp
        } else {
            DeliveryMode::Inject
        };

        let push_res = inbox.push_appending(
            parent_id,
            wd_mode,
            TurnInput::notice(format!("subagent-watchdog:{assigned_name}"), watchdog_text),
            "subagent:watchdog",
            "subagent-watchdog:batch",
        );

        let ui_msg = if push_res.degraded || wd_mode == DeliveryMode::Inject {
            format!(
                "后台子代理 {assigned_name} 已静默 {idle_secs}s（第 {step} 步{ui_tool}，保留最终汇总唤醒额度，将于下次对话提醒主代理）"
            )
        } else {
            format!(
                "后台子代理 {assigned_name} 已静默 {idle_secs}s（第 {step} 步{ui_tool}），已提醒主代理关照"
            )
        };

        self.emit_state_event(SubagentStateEvent {
            parent_session_id: parent_id.to_string(),
            name: assigned_name.to_string(),
            child_session_id: child_id.to_string(),
            mode,
            status: SubagentRunStatus::Running,
            running_count,
            notice: Some(ui_msg),
        });
    }
}

pub(crate) fn format_background_ui_notice(
    name: &str,
    status: SubagentRunStatus,
    steps: usize,
    max_steps: usize,
    elapsed_secs: u64,
    timeout_secs: u64,
    degraded: bool,
) -> String {
    let follow_suffix = if degraded {
        "（已达连续自动唤醒上限，结果将在你下次发言时附带）"
    } else {
        "，交由主代理汇总"
    };
    let error_suffix = if degraded {
        "（已达连续自动唤醒上限，结果将在你下次发言时附带）"
    } else {
        "，交由主代理处理"
    };
    match status {
        SubagentRunStatus::Completed => {
            format!("后台子代理 {name} 已完成（{steps} 步，{elapsed_secs}s）{follow_suffix}")
        }
        SubagentRunStatus::MaxSteps => {
            format!(
                "后台子代理 {name} 达单轮步数上限（单轮上限 {max_steps}，累计 {steps} 步，{elapsed_secs}s）{follow_suffix}"
            )
        }
        SubagentRunStatus::Truncated => {
            format!("后台子代理 {name} 输出截断（{steps} 步，{elapsed_secs}s）{follow_suffix}")
        }
        SubagentRunStatus::TimedOut => {
            format!("后台子代理 {name} 执行超时（超过 {timeout_secs}s 上限）{error_suffix}")
        }
        SubagentRunStatus::Failed => {
            format!("后台子代理 {name} 执行失败（{steps} 步，{elapsed_secs}s）{error_suffix}")
        }
        SubagentRunStatus::Stopped => format!("后台子代理 {name} 已停止"),
        SubagentRunStatus::Running => format!("后台子代理 {name} 运行中"),
    }
}
