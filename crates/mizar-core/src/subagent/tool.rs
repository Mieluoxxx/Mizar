//! `subagent` 工具定义与动作分发（`spawn` / `send` / `status` / `stop` / `yield`）。

use std::sync::Arc;

use crate::agent_loop::RunOptions;
use mizar_llm::RequestOrigin;
use mizar_session::{SessionCreateOptions, SessionOrigin};
use mizar_tools::{
    ToolAnnotations, ToolCallContext, ToolDefinition, ToolExecutionMode, ToolExposure, ToolOutput,
};
use mizar_types::BoxError;
use serde_json::{Value, json};

use crate::subagent::guardrails::{
    ChildSessionGuard, compute_allowed_tools, ensure_not_nested_parent, resolve_model_route,
    shorten_prompt,
};
use crate::subagent::tasks::{
    SendOutcome, SubagentRunStatus, SubagentStateEvent, validate_custom_name,
};
use crate::subagent::worker::BackgroundWorkerArgs;
use crate::subagent::{SUBAGENT_TOOL_NAME, SubagentError, SubagentMode, SubagentRuntime};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum SubagentAction {
    #[default]
    Spawn,
    Send,
    Status,
    Stop,
    Yield,
}

impl SubagentAction {
    fn parse(raw: &str) -> Result<Self, SubagentError> {
        match raw.trim() {
            "spawn" => Ok(Self::Spawn),
            "send" => Ok(Self::Send),
            "status" => Ok(Self::Status),
            "stop" => Ok(Self::Stop),
            "yield" => Ok(Self::Yield),
            other => Err(SubagentError::InvalidAction(other.to_string())),
        }
    }
}

pub(crate) fn build_subagent_tool_definition(runtime: Arc<SubagentRuntime>) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        SUBAGENT_TOOL_NAME,
        "Delegate a focused, single-layer child agent task or follow up on running/settled subagents. \
Use action=\"spawn\" (default) with `prompt` to launch a background subagent (`read_only` default or `edit`). \
Use action=\"send\" with `name` and `prompt` to steer a running subagent or wake a settled subagent to continue with its existing session memory. \
Use action=\"status\" (optional `name`) for requested progress checks or watchdog alerts, or action=\"stop\" (`name` or \"all\") to cancel. \
Use action=\"yield\" with `message` when the next useful step needs a subagent result. It ends your turn without waiting or stopping the children; results arrive as follow-ups.",
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "send", "status", "stop", "yield"],
                    "default": "spawn",
                    "description": "Operation to perform: spawn (default), send, status, stop, or yield."
                },
                "message": {
                    "type": "string",
                    "description": "Brief user-facing progress message explaining what is pending (required for yield)."
                },
                "prompt": {
                    "type": "string",
                    "description": "Self-contained task instruction (required for \"spawn\" and \"send\")."
                },
                "mode": {
                    "type": "string",
                    "enum": ["read_only", "edit"],
                    "default": "read_only",
                    "description": "Execution mode for \"spawn\". \"read_only\" restricts tools to read-only inspection; \"edit\" allows workspace edits. Cannot be escalated via \"send\"."
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override for \"spawn\" or resuming a settled subagent via \"send\" (\"model_id\" or \"provider/model_id\")."
                },
                "name": {
                    "type": "string",
                    "description": "Optional child name (1-32 chars [a-z0-9_-]) for \"spawn\" / \"status\"; required target name for \"send\"; target name or \"all\" for \"stop\"."
                }
            },
            "required": []
        }),
        move |args, call_ctx| {
            let rt = Arc::clone(&runtime);
            async move { rt.execute_tool_call(args, call_ctx).await }
        },
    )
    .with_exposure(ToolExposure::ModelOnly)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
    })
    .with_prompt_snippet(
        "Delegate or follow up with isolated single-layer child agent sessions (`spawn` default, `send`, `status`, `stop`, `yield`)",
    )
}

impl SubagentRuntime {
    pub(crate) async fn execute_tool_call(
        self: &Arc<Self>,
        args: Value,
        call_ctx: ToolCallContext,
    ) -> Result<ToolOutput, BoxError> {
        let action = match parse_optional_trimmed_string(&args, "action", || {
            SubagentError::InvalidActionType
        })? {
            None => SubagentAction::Spawn,
            Some(raw) => SubagentAction::parse(&raw)?,
        };
        let requested_name =
            parse_optional_trimmed_string(&args, "name", || SubagentError::InvalidNameType)?;

        let sessions = &self.sessions;
        let parent_session_id = call_ctx.session_id().map(str::to_string);
        ensure_not_nested_parent(
            parent_session_id.as_deref(),
            &self.active_children,
            sessions,
        )?;

        match action {
            SubagentAction::Yield => {
                let parent_id = parent_session_id.ok_or(SubagentError::MissingParentSession)?;
                let message = parse_optional_trimmed_string(&args, "message", || {
                    SubagentError::InvalidYieldMessageType
                })?
                .ok_or(SubagentError::MissingYieldMessage)?;
                let inbox = self.agent_loop.inbox();
                let (running, has_followup) = {
                    let reg = self.tasks.lock().unwrap();
                    let running = reg.running_count(&parent_id);
                    let has_followup = if running > 0 {
                        inbox.arm_yield_barrier(&parent_id);
                        true
                    } else {
                        inbox.has_deliverable_followup(&parent_id)
                    };
                    (running, has_followup)
                };
                if running == 0 && !has_followup {
                    return Ok(ToolOutput::text(
                        "No background subagents are currently running and no pending follow-up results are queued. Answer the user directly instead of calling yield.",
                    )
                    .with_details(json!({
                        "action": "yield",
                        "yielded": false,
                        "running": 0,
                    })));
                }
                Ok(ToolOutput::end_turn(message).with_details(json!({
                    "action": "yield",
                    "yielded": true,
                    "running": running,
                })))
            }
            SubagentAction::Status => {
                Ok(self
                    .execute_status_action(parent_session_id.as_deref(), requested_name.as_deref()))
            }
            SubagentAction::Stop => {
                self.execute_stop_action(parent_session_id.as_deref(), requested_name.as_deref())
            }
            SubagentAction::Send => {
                self.execute_send_action(&args, &call_ctx, parent_session_id, requested_name)
                    .await
            }
            SubagentAction::Spawn => {
                self.execute_spawn_action(&args, &call_ctx, parent_session_id, requested_name)
                    .await
            }
        }
    }

    fn execute_status_action(
        &self,
        parent_session_id: Option<&str>,
        filter_name: Option<&str>,
    ) -> ToolOutput {
        let Some(pid) = parent_session_id else {
            return ToolOutput::text("No parent session context; no background subagents.")
                .with_details(json!({ "action": "status", "running": 0, "tasks": [] }));
        };
        let (snapshots, running, single_detail) = {
            let reg = self.tasks.lock().unwrap();
            reg.inspect_status(pid, filter_name, self.agent_loop.inbox())
        };
        if snapshots.is_empty() {
            let msg = match filter_name {
                Some(name) => format!("No subagent matching \"{name}\" in this session."),
                None => "No background subagents in this session.".to_string(),
            };
            return ToolOutput::text(msg)
                .with_details(json!({ "action": "status", "running": running, "tasks": [] }));
        }
        if let Some(detail) = single_detail {
            let t = &detail.snapshot;
            let body = detail
                .last_output
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(t.summary.as_deref().filter(|s| !s.is_empty()))
                .unwrap_or("(no output)");
            let text = format!(
                "Subagent \"{}\" [{}] mode={} steps={} elapsed={}s session={}:\n\n{}",
                t.name,
                t.status.as_str(),
                t.mode.as_str(),
                t.steps,
                t.elapsed_secs,
                t.child_session_id,
                body
            );
            let mut details = json!({
                "action": "status",
                "running": running,
                "tasks": snapshots,
                "output": body,
            });
            if let Some(ref path) = detail.last_output_path {
                details["outputPath"] = json!(path.display().to_string());
            }
            return ToolOutput::text(text).with_details(details);
        }
        let lines: Vec<String> = snapshots
            .iter()
            .map(|t| {
                let mut line = format!(
                    "- {} [{}] mode={} elapsed={}s session={}",
                    t.name,
                    t.status.as_str(),
                    t.mode.as_str(),
                    t.elapsed_secs,
                    t.child_session_id
                );
                if t.status.is_terminal() {
                    line.push_str(&format!(" steps={}", t.steps));
                } else {
                    line.push_str(&format!(" step={} idle={}s", t.current_step, t.idle_secs));
                    if let Some(ref tool) = t.last_tool {
                        line.push_str(&format!(" tool=\"{tool}\""));
                    }
                    if t.watchdog_alerts > 0 {
                        line.push_str(&format!(" watchdog={}", t.watchdog_alerts));
                    }
                }
                if !t.prompt_preview.is_empty() {
                    line.push_str(&format!(" | {}", t.prompt_preview));
                }
                if t.status.is_terminal()
                    && let Some(ref summary) = t.summary
                    && !summary.is_empty()
                {
                    line.push_str(&format!(" => {summary}"));
                }
                line
            })
            .collect();
        ToolOutput::text(format!(
            "Subagents (running={running}, shown={}):\n{}",
            snapshots.len(),
            lines.join("\n")
        ))
        .with_details(json!({
            "action": "status",
            "running": running,
            "tasks": snapshots,
        }))
    }

    fn execute_stop_action(
        &self,
        parent_session_id: Option<&str>,
        target: Option<&str>,
    ) -> Result<ToolOutput, BoxError> {
        let target = target.ok_or(SubagentError::MissingStopTarget)?;
        let parent_id = parent_session_id.ok_or(SubagentError::MissingParentSession)?;
        let (stopped, running_count) = {
            let mut reg = self.tasks.lock().unwrap();
            let stopped = reg.request_stop(parent_id, target, self.agent_loop.inbox());
            let running_count = reg.running_count(parent_id);
            (stopped, running_count)
        };
        if stopped.is_empty() {
            return Ok(ToolOutput::text(format!(
                "No running subagent matching \"{target}\" to stop."
            ))
            .with_details(json!({
                "action": "stop",
                "stopped": [],
                "running": running_count,
            })));
        }
        let stopped_names: Vec<&str> = stopped.iter().map(|(n, _, _)| n.as_str()).collect();
        let notice = if stopped_names.len() == 1 {
            format!("后台子代理 {} 已停止", stopped_names[0])
        } else {
            format!(
                "已停止 {} 个后台子代理：{}",
                stopped_names.len(),
                stopped_names.join(", ")
            )
        };
        if let Some((name, child_session_id, mode)) = stopped.last() {
            self.emit_state_event(SubagentStateEvent {
                parent_session_id: parent_id.to_string(),
                name: name.clone(),
                child_session_id: child_session_id.clone(),
                mode: *mode,
                status: SubagentRunStatus::Stopped,
                running_count,
                notice: Some(notice),
            });
        }
        Ok(ToolOutput::text(format!(
            "Stopped subagent(s): {} (remaining running: {running_count}).",
            stopped_names.join(", ")
        ))
        .with_details(json!({
            "action": "stop",
            "stopped": stopped_names,
            "running": running_count,
        })))
    }

    async fn execute_spawn_action(
        self: &Arc<Self>,
        args: &Value,
        call_ctx: &ToolCallContext,
        parent_session_id: Option<String>,
        requested_name: Option<String>,
    ) -> Result<ToolOutput, BoxError> {
        let prompt = parse_required_prompt(args)?;
        let parent_id = parent_session_id.ok_or(SubagentError::MissingParentSession)?;

        let mode =
            match parse_optional_trimmed_string(args, "mode", || SubagentError::InvalidModeType)? {
                None => SubagentMode::ReadOnly,
                Some(raw) => SubagentMode::parse(&raw)?,
            };
        let requested_model =
            parse_optional_trimmed_string(args, "model", || SubagentError::InvalidModelType)?;

        let sessions = &self.sessions;
        let tools = &self.tools;
        let agent_loop = Arc::clone(&self.agent_loop);

        if call_ctx.cancel.is_cancelled() || self.dispose_token.is_cancelled() {
            return Ok(ToolOutput::error("子代理已取消执行"));
        }

        let cfg = self.config();
        self.tasks.lock().unwrap().preflight_spawn(
            &parent_id,
            requested_name.as_deref(),
            cfg.effective_max_concurrent(),
        )?;

        let parent_cwd = sessions
            .get(&parent_id)
            .and_then(|s| s.header().cwd)
            .or_else(|| {
                sessions
                    .inspect_readonly(&parent_id)
                    .and_then(|l| l.header.cwd)
            })
            .or_else(|| sessions.default_cwd());

        let (parent_provider, parent_model) = self.resolve_parent_route(Some(&parent_id));
        let providers = self.llm.provider_infos();
        let (provider, model) = resolve_model_route(
            requested_model.as_deref(),
            parent_provider,
            parent_model,
            &providers,
        );

        let short_prompt = shorten_prompt(&prompt, 48);
        let child_session = sessions.create_with(SessionCreateOptions {
            cwd: parent_cwd,
            parent_id: Some(parent_id.clone()),
            origin: Some(SessionOrigin::Subagent),
            title: Some(format!("Subagent: {short_prompt}")),
            ..Default::default()
        });

        let candidate_tools = tools.get_active_tools(&parent_id);
        let all_infos = tools.list_all();
        let allowed_tools = compute_allowed_tools(&candidate_tools, &all_infos, mode);

        let soft_budget = cfg.soft_budget_for(mode);
        let child_cancel = self.dispose_token.child_token();
        let (run_id, assigned_name, running_count, activity) = {
            let mut reg = self.tasks.lock().unwrap();
            if call_ctx.cancel.is_cancelled() || self.dispose_token.is_cancelled() {
                return Ok(ToolOutput::error("子代理已取消执行"));
            }
            reg.register_running(
                &parent_id,
                requested_name.as_deref(),
                child_session.id.clone(),
                mode,
                provider.clone(),
                model.clone(),
                short_prompt.clone(),
                child_cancel.clone(),
                cfg.effective_max_concurrent(),
                soft_budget,
                agent_loop.inbox(),
            )?
        };

        let child_guard = ChildSessionGuard::register(
            Arc::clone(&self.active_children),
            Arc::clone(&self.activity_trackers),
            Arc::clone(tools),
            child_session.id.clone(),
            run_id,
            mode,
            Arc::clone(&activity),
        );
        tools.set_active_tools(&child_session.id, &allowed_tools);

        if child_cancel.is_cancelled() || self.dispose_token.is_cancelled() {
            drop(child_guard);
            let _ = self.tasks.lock().unwrap().mark_settled(
                &parent_id,
                &assigned_name,
                &child_session.id,
                run_id,
                SubagentRunStatus::Stopped,
                0,
                Some("子代理已取消执行".into()),
                Some("子代理已取消执行".into()),
                None,
                agent_loop.inbox(),
                None,
            );
            return Ok(ToolOutput::error("子代理已取消执行"));
        }

        let _ = child_session.set_title(format!("Subagent [{assigned_name}]: {short_prompt}"));
        let child_id = child_session.id.clone();

        self.emit_state_event(SubagentStateEvent {
            parent_session_id: parent_id.clone(),
            name: assigned_name.clone(),
            child_session_id: child_id.clone(),
            mode,
            status: SubagentRunStatus::Running,
            running_count,
            notice: Some(format!(
                "后台子代理 {assigned_name} 已启动 ({})",
                mode.as_str()
            )),
        });

        let timeout_secs = cfg.effective_timeout_secs();
        let run_opts = RunOptions {
            provider,
            model,
            origin: Some(RequestOrigin::Subagent),
            ..Default::default()
        };

        self.spawn_background_worker(BackgroundWorkerArgs {
            run_id,
            parent_id,
            assigned_name: assigned_name.clone(),
            child_session,
            child_guard,
            child_cancel,
            activity,
            agent_loop,
            initial_prompt: prompt,
            run_opts,
            mode,
            soft_budget,
            allowed_tools,
            timeout_secs,
            stall_timeout_secs: cfg.stall_timeout_secs,
            max_watchdog_alerts: cfg.max_watchdog_alerts,
        });

        Ok(ToolOutput::text(format!(
            "Started background subagent \"{assigned_name}\" (mode={}, child_session={child_id}). Continue other useful work or end this turn if waiting for results.",
            mode.as_str()
        ))
        .with_details(json!({
            "action": "spawn",
            "name": assigned_name,
            "childSessionId": child_id,
            "mode": mode.as_str(),
            "status": "running",
            "background": true,
        })))
    }

    async fn execute_send_action(
        self: &Arc<Self>,
        args: &Value,
        call_ctx: &ToolCallContext,
        parent_session_id: Option<String>,
        requested_name: Option<String>,
    ) -> Result<ToolOutput, BoxError> {
        let target_raw = requested_name.ok_or(SubagentError::MissingSendTarget)?;
        if target_raw.trim().eq_ignore_ascii_case("all") {
            return Err(SubagentError::MissingSendTarget.into());
        }
        let target_name = validate_custom_name(&target_raw)?;
        let prompt = parse_required_prompt(args)?;
        let parent_id = parent_session_id.ok_or(SubagentError::MissingParentSession)?;

        let requested_mode =
            match parse_optional_trimmed_string(args, "mode", || SubagentError::InvalidModeType)? {
                None => None,
                Some(raw) => Some(SubagentMode::parse(&raw)?),
            };
        let requested_model =
            parse_optional_trimmed_string(args, "model", || SubagentError::InvalidModelType)?;

        let sessions = &self.sessions;
        let tools = &self.tools;
        let agent_loop = Arc::clone(&self.agent_loop);

        if call_ctx.cancel.is_cancelled() || self.dispose_token.is_cancelled() {
            return Ok(ToolOutput::error("子代理已取消执行"));
        }

        let cfg = self.config();
        let short_prompt = shorten_prompt(&prompt, 48);
        let outcome = self.tasks.lock().unwrap().prepare_send(
            &parent_id,
            &target_name,
            short_prompt,
            requested_mode,
            requested_model.as_deref(),
            agent_loop.inbox(),
            &prompt,
            &cfg,
            &self.dispose_token,
        )?;

        match outcome {
            SendOutcome::SteeredRunning {
                name,
                child_session_id,
                mode,
            } => Ok(ToolOutput::text(format!(
                "Sent steering update to running subagent \"{name}\" (mode={}, child_session={child_session_id}). It will apply on the next step.",
                mode.as_str()
            ))
            .with_details(json!({
                "action": "send",
                "delivery": "steer",
                "name": name,
                "childSessionId": child_session_id,
                "mode": mode.as_str(),
                "status": "running",
            }))),
            SendOutcome::Reactivated {
                run_id,
                name,
                child_session_id,
                mode,
                prev_provider,
                prev_model,
                activity,
                cancel: child_cancel,
                rollback,
            } => {
                if child_cancel.is_cancelled()
                    || call_ctx.cancel.is_cancelled()
                    || self.dispose_token.is_cancelled()
                {
                    self.tasks.lock().unwrap().rollback_reactivation(
                        &parent_id,
                        &name,
                        &child_session_id,
                        run_id,
                        rollback,
                    );
                    return Ok(ToolOutput::error("子代理已取消执行"));
                }

                let Some(child_session) = sessions.get(&child_session_id) else {
                    self.tasks.lock().unwrap().rollback_reactivation(
                        &parent_id,
                        &name,
                        &child_session_id,
                        run_id,
                        rollback,
                    );
                    return Err(
                        SubagentError::SessionUnavailable(name, child_session_id).into()
                    );
                };

                // 唤醒前先幂等修复可能因上次超时或异常遗留的悬挂 ToolCall，避免触发 LLM 400
                let _ = child_session.repair_dangling_tool_calls();

                let candidate_tools = tools.get_active_tools(&parent_id);
                let all_infos = tools.list_all();
                let allowed_tools = compute_allowed_tools(&candidate_tools, &all_infos, mode);
                let child_guard = ChildSessionGuard::register(
                    Arc::clone(&self.active_children),
                    Arc::clone(&self.activity_trackers),
                    Arc::clone(tools),
                    child_session.id.clone(),
                    run_id,
                    mode,
                    Arc::clone(&activity),
                );
                tools.set_active_tools(&child_session.id, &allowed_tools);

                if child_cancel.is_cancelled()
                    || call_ctx.cancel.is_cancelled()
                    || self.dispose_token.is_cancelled()
                {
                    drop(child_guard);
                    self.tasks.lock().unwrap().rollback_reactivation(
                        &parent_id,
                        &name,
                        &child_session_id,
                        run_id,
                        rollback,
                    );
                    return Ok(ToolOutput::error("子代理已取消执行"));
                }

                let (provider, model) = if requested_model.is_some() {
                    let providers = self.llm.provider_infos();
                    let (fb_provider, fb_model) = if prev_provider.is_some() || prev_model.is_some() {
                        (prev_provider, prev_model)
                    } else {
                        self.resolve_parent_route(Some(&parent_id))
                    };
                    resolve_model_route(requested_model.as_deref(), fb_provider, fb_model, &providers)
                } else if prev_provider.is_some() || prev_model.is_some() {
                    (prev_provider, prev_model)
                } else {
                    self.resolve_parent_route(Some(&parent_id))
                };

                self.tasks.lock().unwrap().update_route(
                    &parent_id,
                    &name,
                    &child_session_id,
                    run_id,
                    provider.clone(),
                    model.clone(),
                );

                self.emit_state_event(SubagentStateEvent {
                    parent_session_id: parent_id.clone(),
                    name: name.clone(),
                    child_session_id: child_session_id.clone(),
                    mode,
                    status: SubagentRunStatus::Running,
                    running_count: 0,
                    notice: Some(format!("后台子代理 {name} 已唤醒继续执行")),
                });

                let soft_budget = cfg.soft_budget_for(mode);
                let timeout_secs = cfg.effective_timeout_secs();
                let run_opts = RunOptions {
                    provider,
                    model,
                    origin: Some(RequestOrigin::Subagent),
                    ..Default::default()
                };

                self.spawn_background_worker(BackgroundWorkerArgs {
                    run_id,
                    parent_id,
                    assigned_name: name.clone(),
                    child_session,
                    child_guard,
                    child_cancel,
                    activity,
                    agent_loop,
                    initial_prompt: prompt,
                    run_opts,
                    mode,
                    soft_budget,
                    allowed_tools,
                    timeout_secs,
                    stall_timeout_secs: cfg.stall_timeout_secs,
                    max_watchdog_alerts: cfg.max_watchdog_alerts,
                });

                Ok(ToolOutput::text(format!(
                    "Resumed background subagent \"{name}\" (mode={}, child_session={child_session_id}). Continue other useful work or end this turn if waiting for results.",
                    mode.as_str()
                ))
                .with_details(json!({
                    "action": "send",
                    "delivery": "resume",
                    "name": name,
                    "childSessionId": child_session_id,
                    "mode": mode.as_str(),
                    "status": "running",
                })))
            }
        }
    }
}

fn parse_required_prompt(args: &Value) -> Result<String, SubagentError> {
    match args.get("prompt") {
        None | Some(Value::Null) => Err(SubagentError::MissingPrompt),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Err(SubagentError::MissingPrompt)
            } else {
                Ok(trimmed.to_string())
            }
        }
        Some(_) => Err(SubagentError::InvalidPromptType),
    }
}

fn parse_optional_trimmed_string<F>(
    args: &Value,
    key: &str,
    on_wrong_type: F,
) -> Result<Option<String>, SubagentError>
where
    F: FnOnce() -> SubagentError,
{
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            Ok((!trimmed.is_empty()).then(|| trimmed.to_string()))
        }
        Some(_) => Err(on_wrong_type()),
    }
}
