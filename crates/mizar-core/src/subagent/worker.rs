//! 后台 Subagent 执行器、内嵌零 Token 活性看门狗、超时修复与无竞态续跑循环。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::agent_loop::inbox::SubagentSlotKind;
use crate::agent_loop::{
    AgentError, AgentLoop, DeliveryMode, RunOptions, TurnEndReason, TurnInput, TurnRunReport,
};
use mizar_llm::LlmErrorCode;
use mizar_session::Session;
use tokio_util::sync::CancellationToken;

use crate::subagent::guardrails::{ChildSessionGuard, shorten_prompt};
use crate::subagent::tasks::{
    ChildActivityTracker, SubagentRunStatus, SubagentStateEvent, WorkerTurnAction,
    format_terminal_output_for_delivery,
};
use crate::subagent::{SubagentConfig, SubagentMode, SubagentRuntime};

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
    pub soft_budget: usize,
    pub allowed_tools: Vec<String>,
    pub timeout_secs: u64,
    pub stall_timeout_secs: u64,
    pub max_watchdog_alerts: usize,
}

enum InnerTurnOutcome {
    Disposed,
    Finished(Result<TurnRunReport, AgentError>),
    TimedOut,
}

struct WatchdogMonitorCtx<'a> {
    run_id: u64,
    parent_id: &'a str,
    assigned_name: &'a str,
    child_id: &'a str,
    mode: SubagentMode,
    activity: &'a ChildActivityTracker,
    stall_timeout_secs: u64,
    max_watchdog_alerts: usize,
}

#[allow(clippy::too_many_arguments)]
async fn run_monitored_turn(
    rt: &Arc<SubagentRuntime>,
    agent_loop: &Arc<AgentLoop>,
    child_session: &Arc<Session>,
    input: TurnInput,
    child_cancel: &CancellationToken,
    run_opts: RunOptions,
    hard_deadline: Option<tokio::time::Instant>,
    watchdog: &WatchdogMonitorCtx<'_>,
    alerts_sent: &mut usize,
    last_alert_at: &mut Option<Instant>,
) -> InnerTurnOutcome {
    let run_fut = agent_loop.run_with_report(
        Arc::clone(child_session),
        input,
        child_cancel.clone(),
        run_opts,
    );
    tokio::pin!(run_fut);

    let timeout_fut = async {
        match hard_deadline {
            Some(dl) => tokio::time::sleep_until(dl).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(timeout_fut);

    loop {
        let watchdog_sleep =
            if watchdog.stall_timeout_secs > 0 && *alerts_sent < watchdog.max_watchdog_alerts {
                let stall = Duration::from_secs(watchdog.stall_timeout_secs);
                let base = match *last_alert_at {
                    Some(t) => watchdog.activity.last_activity_instant().max(t),
                    None => watchdog.activity.last_activity_instant(),
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
            _ = &mut timeout_fut => {
                child_cancel.cancel();
                let _ = tokio::time::timeout(Duration::from_millis(200), &mut run_fut).await;
                break InnerTurnOutcome::TimedOut;
            }
            _ = watchdog_fut => {
                let stall = Duration::from_secs(watchdog.stall_timeout_secs);
                let idle = watchdog.activity.idle_duration();
                let since_last_alert = last_alert_at.map_or(stall, |t| t.elapsed());
                if idle >= stall
                    && since_last_alert >= stall
                    && !child_cancel.is_cancelled()
                    && !rt.dispose_token.is_cancelled()
                {
                    *alerts_sent += 1;
                    *last_alert_at = Some(Instant::now());
                    rt.fire_watchdog_alert(
                        watchdog.run_id,
                        watchdog.parent_id,
                        watchdog.assigned_name,
                        watchdog.child_id,
                        watchdog.mode,
                        watchdog.activity,
                        agent_loop,
                        *alerts_sent,
                        watchdog.max_watchdog_alerts,
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
}

enum NormalOrHardAbort {
    Settled {
        status: SubagentRunStatus,
        turn_steps: usize,
        raw_body: String,
        allow_continue: bool,
    },
    Disposed,
    TimedOut {
        turn_steps: usize,
        raw_body: String,
    },
    Cancelled {
        turn_steps: usize,
    },
    Failed {
        turn_steps: usize,
        raw_body: String,
    },
}

fn classify_aborted_or_error(
    outcome: Result<TurnRunReport, AgentError>,
    child_cancel: &CancellationToken,
) -> NormalOrHardAbort {
    match outcome {
        Ok(report) => match report.reason {
            TurnEndReason::Cancelled => NormalOrHardAbort::Cancelled {
                turn_steps: report.steps,
            },
            TurnEndReason::Error { message } => NormalOrHardAbort::Failed {
                turn_steps: report.steps,
                raw_body: format!("[Subagent failed after {} steps: {message}]", report.steps),
            },
            TurnEndReason::Completed | TurnEndReason::MaxSteps | TurnEndReason::Truncated => {
                NormalOrHardAbort::Cancelled {
                    turn_steps: report.steps,
                }
            }
        },
        Err(AgentError::Llm(err))
            if err.code == LlmErrorCode::Cancelled || child_cancel.is_cancelled() =>
        {
            NormalOrHardAbort::Cancelled { turn_steps: 0 }
        }
        Err(err) => NormalOrHardAbort::Failed {
            turn_steps: 0,
            raw_body: format!("[Subagent failed: {err}]"),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_mark_settled_with_notice(
    rt: &SubagentRuntime,
    agent_loop: &AgentLoop,
    child_session: &Arc<Session>,
    child_guard: ChildSessionGuard,
    parent_id: &str,
    assigned_name: &str,
    child_id: &str,
    run_id: u64,
    mode: SubagentMode,
    status: SubagentRunStatus,
    turn_steps: usize,
    summary_preview: String,
    delivered_body: String,
    output_path: Option<PathBuf>,
    soft_budget: usize,
    timeout_secs: u64,
) -> Option<(SubagentRunStatus, usize, String)> {
    agent_loop.inbox().clear(child_id);
    drop(child_guard);
    let _ = child_session.repair_dangling_tool_calls();
    let build_notice = |final_status: SubagentRunStatus, steps: usize, elapsed_secs: u64| {
        Some(format_settled_inbox_message(
            assigned_name,
            child_id,
            mode,
            final_status,
            steps,
            elapsed_secs,
            &delivered_body,
        ))
    };
    let settled = rt.tasks.lock().unwrap().mark_settled(
        parent_id,
        assigned_name,
        child_id,
        run_id,
        status,
        turn_steps,
        Some(summary_preview),
        Some(delivered_body.clone()),
        output_path,
        agent_loop.inbox(),
        (!rt.dispose_token.is_cancelled()).then_some(&build_notice as &_),
    );
    if let Some((was_stop, total_steps, _, elapsed_secs, degraded)) = settled
        && !was_stop
        && !rt.dispose_token.is_cancelled()
    {
        let ui_notice = format_background_ui_notice(
            assigned_name,
            status,
            total_steps,
            soft_budget,
            elapsed_secs,
            timeout_secs,
            degraded,
        );
        return Some((status, total_steps, ui_notice));
    }
    None
}

fn format_settled_inbox_message(
    assigned_name: &str,
    child_id: &str,
    mode: SubagentMode,
    final_status: SubagentRunStatus,
    steps: usize,
    elapsed_secs: u64,
    delivered_body: &str,
) -> (String, bool) {
    let urgent = matches!(
        final_status,
        SubagentRunStatus::Failed | SubagentRunStatus::TimedOut
    );
    let text = format!(
        "Subagent \"{assigned_name}\" {} (mode={}, steps={steps}, elapsed={elapsed_secs}s, child_session={child_id}):\n\n{delivered_body}",
        final_status.as_str(),
        mode.as_str()
    );
    (text, urgent)
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
                mut run_opts,
                mode,
                soft_budget,
                allowed_tools,
                timeout_secs,
                stall_timeout_secs,
                max_watchdog_alerts,
            } = args;

            let child_id = child_session.id.clone();
            let stop_threshold = SubagentConfig::force_stop_steps(soft_budget);
            run_opts.max_steps = stop_threshold;
            let mut next_input: TurnInput = TurnInput::user(initial_prompt);

            let watchdog_ctx = WatchdogMonitorCtx {
                run_id,
                parent_id: &parent_id,
                assigned_name: &assigned_name,
                child_id: &child_id,
                mode,
                activity: &activity,
                stall_timeout_secs,
                max_watchdog_alerts,
            };

            let mut final_ui_event: Option<(SubagentRunStatus, usize, String)> = None;

            loop {
                activity.reset_turn_budget();
                let mut alerts_sent = 0usize;
                let mut last_alert_at: Option<Instant> = None;
                let hard_deadline = (timeout_secs > 0)
                    .then(|| tokio::time::Instant::now() + Duration::from_secs(timeout_secs));

                let inner_outcome = run_monitored_turn(
                    &rt,
                    &agent_loop,
                    &child_session,
                    next_input,
                    &child_cancel,
                    run_opts.clone(),
                    hard_deadline,
                    &watchdog_ctx,
                    &mut alerts_sent,
                    &mut last_alert_at,
                )
                .await;

                let phase_result = match inner_outcome {
                    InnerTurnOutcome::Disposed => NormalOrHardAbort::Disposed,
                    InnerTurnOutcome::TimedOut => NormalOrHardAbort::TimedOut {
                        turn_steps: 0,
                        raw_body: format!("[Subagent timed out after {timeout_secs}s limit]"),
                    },
                    InnerTurnOutcome::Finished(Ok(report))
                        if !child_cancel.is_cancelled()
                            && !rt.dispose_token.is_cancelled()
                            && report.reason == TurnEndReason::MaxSteps
                            && soft_budget > 0 =>
                    {
                        // 到达 1.5x 软预算：立即停用全部工具并执行 1 步强制总结交付已有发现
                        activity.enter_forced_wrapup();
                        activity.advance_step_offset(report.steps);
                        rt.tools.set_active_tools(&child_id, &[]);

                        let stop_at = stop_threshold.unwrap_or(report.steps);
                        let wrap_input = TurnInput::notice(
                            "subagent-budget-stop",
                            format!(
                                "[budget stop] Free tool loop stopped after {} steps (1.5x soft budget: {soft_budget}). Tools are now disabled for this final step. Summarize your partial findings, verified evidence, and remaining gaps immediately.",
                                report.steps
                            ),
                        );
                        let wrap_opts = RunOptions {
                            max_steps: Some(1),
                            ..run_opts.clone()
                        };
                        let wrap_outcome = run_monitored_turn(
                            &rt,
                            &agent_loop,
                            &child_session,
                            wrap_input,
                            &child_cancel,
                            wrap_opts,
                            hard_deadline,
                            &watchdog_ctx,
                            &mut alerts_sent,
                            &mut last_alert_at,
                        )
                        .await;

                        match wrap_outcome {
                            InnerTurnOutcome::Disposed => NormalOrHardAbort::Disposed,
                            InnerTurnOutcome::TimedOut => NormalOrHardAbort::TimedOut {
                                turn_steps: report.steps,
                                raw_body: format!(
                                    "[Subagent timed out during budget wrap-up after {timeout_secs}s limit]\n{}",
                                    report.text
                                ),
                            },
                            InnerTurnOutcome::Finished(Ok(wrap_report))
                                if !child_cancel.is_cancelled()
                                    && !rt.dispose_token.is_cancelled() =>
                            {
                                let total_turn_steps =
                                    report.steps.saturating_add(wrap_report.steps);
                                match wrap_report.reason {
                                    TurnEndReason::Completed
                                        if !wrap_report.text.trim().is_empty() =>
                                    {
                                        NormalOrHardAbort::Settled {
                                            status: SubagentRunStatus::Completed,
                                            turn_steps: total_turn_steps,
                                            raw_body: wrap_report.text,
                                            allow_continue: true,
                                        }
                                    }
                                    TurnEndReason::Truncated
                                        if !wrap_report.text.trim().is_empty() =>
                                    {
                                        NormalOrHardAbort::Settled {
                                            status: SubagentRunStatus::Truncated,
                                            turn_steps: total_turn_steps,
                                            raw_body: format!(
                                                "[Output truncated at max_tokens during budget wrap-up ({total_turn_steps} steps)]\n{}",
                                                wrap_report.text
                                            ),
                                            allow_continue: true,
                                        }
                                    }
                                    _ => NormalOrHardAbort::Settled {
                                        status: SubagentRunStatus::MaxSteps,
                                        turn_steps: total_turn_steps,
                                        raw_body: format!(
                                            "[Reached 1.5x step limit ({}/{stop_at}); forced wrap-up did not deliver a final report]\n{}",
                                            report.steps, report.text
                                        ),
                                        allow_continue: false,
                                    },
                                }
                            }
                            InnerTurnOutcome::Finished(other) => {
                                match classify_aborted_or_error(other, &child_cancel) {
                                    NormalOrHardAbort::Cancelled { turn_steps } => {
                                        NormalOrHardAbort::Cancelled {
                                            turn_steps: report.steps.saturating_add(turn_steps),
                                        }
                                    }
                                    NormalOrHardAbort::Failed {
                                        turn_steps,
                                        raw_body: err_body,
                                    } => NormalOrHardAbort::Settled {
                                        status: SubagentRunStatus::MaxSteps,
                                        turn_steps: report.steps.saturating_add(turn_steps),
                                        raw_body: format!(
                                            "[Reached 1.5x step limit ({}/{stop_at}); forced wrap-up failed: {err_body}]\n{}",
                                            report.steps, report.text
                                        ),
                                        allow_continue: false,
                                    },
                                    _ => unreachable!(),
                                }
                            }
                        }
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
                        let (status, raw_body) = match &report.reason {
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
                                    "[Reached step limit ({} steps)]\n{}",
                                    report.steps, report.text
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
                        NormalOrHardAbort::Settled {
                            status,
                            turn_steps: report.steps,
                            raw_body,
                            allow_continue: true,
                        }
                    }
                    InnerTurnOutcome::Finished(other) => {
                        classify_aborted_or_error(other, &child_cancel)
                    }
                };

                match phase_result {
                    NormalOrHardAbort::Disposed => {
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
                            Some("子代理插件已卸载".into()),
                            None,
                            agent_loop.inbox(),
                            None,
                        );
                        return;
                    }
                    NormalOrHardAbort::Settled {
                        status: status_if_settled,
                        turn_steps,
                        raw_body,
                        allow_continue,
                    } => {
                        let summary_preview = shorten_prompt(&raw_body, 120);
                        let (delivered_body, output_path): (String, Option<PathBuf>) =
                            format_terminal_output_for_delivery(&raw_body, &child_id, run_id);

                        let decision = if allow_continue {
                            let mut reg = rt.tasks.lock().unwrap();
                            reg.begin_normal_finish_or_continue(
                                &parent_id,
                                &assigned_name,
                                &child_id,
                                run_id,
                                turn_steps,
                                status_if_settled,
                                Some(summary_preview.clone()),
                                Some(delivered_body.clone()),
                                output_path.clone(),
                                agent_loop.inbox(),
                                |txt| shorten_prompt(txt, 48),
                            )
                        } else {
                            None
                        };

                        match decision {
                            Some(WorkerTurnAction::Continue {
                                next_input: queued_input,
                            }) => {
                                // 若刚经历 1.5x 空工具收尾后又收到父代理 send 续跑，恢复原白名单工具集
                                rt.tools.set_active_tools(&child_id, &allowed_tools);
                                next_input = queued_input;
                                continue;
                            }
                            Some(WorkerTurnAction::SettledWaitGuard { was_stop }) => {
                                drop(child_guard);
                                let build_notice =
                                    |final_status: SubagentRunStatus,
                                     steps: usize,
                                     elapsed_secs: u64| {
                                        Some(format_settled_inbox_message(
                                            &assigned_name,
                                            &child_id,
                                            mode,
                                            final_status,
                                            steps,
                                            elapsed_secs,
                                            &delivered_body,
                                        ))
                                    };
                                let finished = rt.tasks.lock().unwrap().finish_active_run(
                                    &parent_id,
                                    &assigned_name,
                                    &child_id,
                                    run_id,
                                    agent_loop.inbox(),
                                    (!was_stop && !rt.dispose_token.is_cancelled())
                                        .then_some(&build_notice as &_),
                                );
                                if let Some((w, total_steps, _, elapsed_secs, degraded)) = finished
                                    && !was_stop
                                    && !w
                                    && !rt.dispose_token.is_cancelled()
                                {
                                    let ui_notice = format_background_ui_notice(
                                        &assigned_name,
                                        status_if_settled,
                                        total_steps,
                                        soft_budget,
                                        elapsed_secs,
                                        timeout_secs,
                                        degraded,
                                    );
                                    final_ui_event =
                                        Some((status_if_settled, total_steps, ui_notice));
                                }
                                break;
                            }
                            None if !allow_continue => {
                                final_ui_event = finalize_mark_settled_with_notice(
                                    &rt,
                                    &agent_loop,
                                    &child_session,
                                    child_guard,
                                    &parent_id,
                                    &assigned_name,
                                    &child_id,
                                    run_id,
                                    mode,
                                    status_if_settled,
                                    turn_steps,
                                    summary_preview,
                                    delivered_body,
                                    output_path,
                                    soft_budget,
                                    timeout_secs,
                                );
                                break;
                            }
                            None => {
                                drop(child_guard);
                                return;
                            }
                        }
                    }
                    NormalOrHardAbort::TimedOut {
                        turn_steps,
                        raw_body,
                    } => {
                        let summary_preview = shorten_prompt(&raw_body, 120);
                        let (delivered_body, output_path) =
                            format_terminal_output_for_delivery(&raw_body, &child_id, run_id);
                        final_ui_event = finalize_mark_settled_with_notice(
                            &rt,
                            &agent_loop,
                            &child_session,
                            child_guard,
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            mode,
                            SubagentRunStatus::TimedOut,
                            turn_steps,
                            summary_preview,
                            delivered_body,
                            output_path,
                            soft_budget,
                            timeout_secs,
                        );
                        break;
                    }
                    NormalOrHardAbort::Failed {
                        turn_steps,
                        raw_body,
                    } => {
                        let summary_preview = shorten_prompt(&raw_body, 120);
                        let (delivered_body, output_path) =
                            format_terminal_output_for_delivery(&raw_body, &child_id, run_id);
                        final_ui_event = finalize_mark_settled_with_notice(
                            &rt,
                            &agent_loop,
                            &child_session,
                            child_guard,
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            mode,
                            SubagentRunStatus::Failed,
                            turn_steps,
                            summary_preview,
                            delivered_body,
                            output_path,
                            soft_budget,
                            timeout_secs,
                        );
                        break;
                    }
                    NormalOrHardAbort::Cancelled { turn_steps } => {
                        agent_loop.inbox().clear(&child_id);
                        drop(child_guard);
                        let _ = child_session.repair_dangling_tool_calls();
                        let body = "[Subagent cancelled]".to_string();
                        let _ = rt.tasks.lock().unwrap().mark_settled(
                            &parent_id,
                            &assigned_name,
                            &child_id,
                            run_id,
                            SubagentRunStatus::Stopped,
                            turn_steps,
                            Some(shorten_prompt(&body, 120)),
                            Some(body),
                            None,
                            agent_loop.inbox(),
                            None,
                        );
                        return;
                    }
                }
            }

            let Some((status, _steps, ui_notice)) = final_ui_event else {
                return;
            };
            if rt.dispose_token.is_cancelled() {
                return;
            }
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

        let push_res = inbox.push_subagent_slot(
            parent_id,
            wd_mode,
            SubagentSlotKind::Watchdog,
            child_id,
            run_id,
            format!("subagent-watchdog:{assigned_name}"),
            watchdog_text,
            true,
            false,
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
    soft_budget: usize,
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
            let stop_at = SubagentConfig::force_stop_steps(soft_budget).unwrap_or(steps);
            format!(
                "后台子代理 {name} 达 1.5x 步数硬上限（阈值 {stop_at}，软预算 {soft_budget}，累计 {steps} 步，{elapsed_secs}s）{follow_suffix}"
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
