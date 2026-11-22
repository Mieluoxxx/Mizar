//! UNIX 进程式单层 `subagent` 插件（`"subagent"` 服务）：
//!
//! 向模型暴露窄接口工具 `subagent(prompt, mode?, model?)`，在独立 `parent_id` 子会话中
//! 执行受控子任务并将精炼总结作为 `tool_result` 返回父会话：
//! - 严格深度 1：子会话工具白名单显式剔除 `subagent`，并在 `tools/pre-execute` 二次兜底拦截；
//! - 默认 `read_only` 并发安全：支持同一步内多个只读子代理并发扇出；
//! - 取消级联与步数/超时封顶：绑定 `parent_cancel.child_token()` 与 `timeout_secs`；
//! - 身份隔离：子会话全部 LLM 调用标注 `RequestOrigin::Subagent`，跳过预热与标题生成。

pub mod guardrails;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, EventOptions, Inject, Next, Plugin,
    Service, ServicePlugin,
};
use mizar_agent_loop::{
    AgentError, AgentLoopContextExt, AgentTurnEndEvent, AgentTurnInputEvent, EVENT_AGENT_TURN_END,
    EVENT_AGENT_TURN_INPUT, RunOptions, TurnEndReason,
};
use mizar_llm::{LlmContextExt, LlmErrorCode, RequestOrigin};
use mizar_session::{SessionContextExt, SessionCreateOptions};
use mizar_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolAnnotations, ToolCallContext, ToolDefinition,
    ToolExecution, ToolExecutionMode, ToolExposure, ToolOutput, ToolsContextExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use guardrails::{
    ChildSessionGuard, check_pre_execute, compute_allowed_tools, ensure_not_nested_parent,
    resolve_model_route, shorten_prompt,
};

pub const SERVICE_SUBAGENT: &str = "subagent";
pub const SUBAGENT_TOOL_NAME: &str = "subagent";
pub const DEFAULT_SUBAGENT_MAX_STEPS: usize = 20;
pub const DEFAULT_SUBAGENT_MAX_STEPS_CAP: usize = 40;
pub const DEFAULT_SUBAGENT_TIMEOUT_SECS: u64 = 180;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentConfig {
    pub default_max_steps: usize,
    pub max_steps_cap: usize,
    pub timeout_secs: u64,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            default_max_steps: DEFAULT_SUBAGENT_MAX_STEPS,
            max_steps_cap: DEFAULT_SUBAGENT_MAX_STEPS_CAP,
            timeout_secs: DEFAULT_SUBAGENT_TIMEOUT_SECS,
        }
    }
}

impl SubagentConfig {
    /// 返回经 `max_steps_cap` 封顶后的有效单次最大步数（至少为 1）。
    pub fn effective_max_steps(&self) -> usize {
        let cap = self.max_steps_cap.max(1);
        self.default_max_steps.clamp(1, cap)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentMode {
    #[default]
    ReadOnly,
    Edit,
}

impl SubagentMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Edit => "edit",
        }
    }

    pub fn parse(raw: &str) -> Result<Self, SubagentError> {
        match raw.trim() {
            "read_only" => Ok(Self::ReadOnly),
            "edit" => Ok(Self::Edit),
            other => Err(SubagentError::InvalidMode(other.to_string())),
        }
    }
}

#[derive(Debug, Error)]
pub enum SubagentError {
    #[error("缺少或为空的 prompt 参数")]
    MissingPrompt,
    #[error("prompt 参数必须为字符串")]
    InvalidPromptType,
    #[error("无效的 mode 参数 `{0}`，仅支持 \"read_only\" 或 \"edit\"")]
    InvalidMode(String),
    #[error("mode 参数必须为字符串")]
    InvalidModeType,
    #[error("model 参数必须为字符串")]
    InvalidModelType,
    #[error("子代理严禁嵌套调用 subagent（深度严格限制为 1）")]
    NestedSubagentDenied,
    #[error("子代理插件依赖服务不可用: {0}")]
    ServiceUnavailable(String),
}

type TurnRouteMap = HashMap<String, (Option<String>, Option<String>)>;

pub struct SubagentRuntime {
    ctx: Context,
    config: Arc<Mutex<SubagentConfig>>,
    active_children: Arc<Mutex<HashMap<String, SubagentMode>>>,
    parent_turn_routes: Arc<Mutex<TurnRouteMap>>,
    dispose_token: CancellationToken,
}

impl SubagentRuntime {
    pub fn plugin() -> ServicePlugin<Self> {
        ServicePlugin::new()
    }

    pub fn config(&self) -> SubagentConfig {
        self.config.lock().unwrap().clone()
    }

    /// 热更新子代理配置（供运行期调参或 `SpinePlugins` 重挂同步）。
    pub fn update_config(&self, f: impl FnOnce(&mut SubagentConfig)) {
        let mut guard = self.config.lock().unwrap();
        f(&mut guard);
    }

    pub fn active_child_count(&self) -> usize {
        self.active_children.lock().unwrap().len()
    }

    pub fn active_child_mode(&self, session_id: &str) -> Option<SubagentMode> {
        self.active_children
            .lock()
            .unwrap()
            .get(session_id)
            .copied()
    }

    fn resolve_parent_route(
        &self,
        parent_session_id: Option<&str>,
    ) -> (Option<String>, Option<String>) {
        let Some(pid) = parent_session_id else {
            return (None, None);
        };
        if let Some(route) = self.parent_turn_routes.lock().unwrap().get(pid).cloned()
            && (route.0.is_some() || route.1.is_some())
        {
            return route;
        }
        if let Ok(sessions) = self.ctx.sessions()
            && let Some(parent) = sessions.get(pid)
            && let Some(sel) = parent.latest_selection()
        {
            return (sel.provider, sel.model);
        }
        (None, None)
    }

    async fn execute_tool_call(
        self: &Arc<Self>,
        args: Value,
        call_ctx: ToolCallContext,
    ) -> Result<ToolOutput, BoxError> {
        let prompt = match args.get("prompt") {
            None | Some(Value::Null) => return Err(SubagentError::MissingPrompt.into()),
            Some(Value::String(s)) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return Err(SubagentError::MissingPrompt.into());
                }
                trimmed.to_string()
            }
            Some(_) => return Err(SubagentError::InvalidPromptType.into()),
        };

        let mode = match args.get("mode") {
            None | Some(Value::Null) => SubagentMode::ReadOnly,
            Some(Value::String(s)) => SubagentMode::parse(s)?,
            Some(_) => return Err(SubagentError::InvalidModeType.into()),
        };

        let requested_model = match args.get("model") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => {
                let trimmed = s.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            Some(_) => return Err(SubagentError::InvalidModelType.into()),
        };

        let sessions = self
            .ctx
            .sessions()
            .map_err(|e| SubagentError::ServiceUnavailable(e.to_string()))?;
        let tools = self
            .ctx
            .tools()
            .map_err(|e| SubagentError::ServiceUnavailable(e.to_string()))?;
        let agent_loop = self
            .ctx
            .agent_loop()
            .map_err(|e| SubagentError::ServiceUnavailable(e.to_string()))?;

        let parent_session_id = call_ctx.session_id().map(str::to_string);
        ensure_not_nested_parent(
            parent_session_id.as_deref(),
            &self.active_children,
            &sessions,
        )?;

        if call_ctx.cancel.is_cancelled() || self.dispose_token.is_cancelled() {
            return Ok(ToolOutput::error("子代理已取消执行"));
        }

        let parent_cwd = parent_session_id
            .as_deref()
            .and_then(|pid| {
                sessions
                    .get(pid)
                    .and_then(|s| s.header().cwd)
                    .or_else(|| sessions.inspect_readonly(pid).and_then(|l| l.header.cwd))
            })
            .or_else(|| sessions.default_cwd());

        let (parent_provider, parent_model) =
            self.resolve_parent_route(parent_session_id.as_deref());
        let providers = self
            .ctx
            .llm()
            .map(|llm| llm.provider_infos())
            .unwrap_or_default();
        let (provider, model) = resolve_model_route(
            requested_model.as_deref(),
            parent_provider,
            parent_model,
            &providers,
        );

        let short_prompt = shorten_prompt(&prompt, 48);
        let child_session = sessions.create_with(SessionCreateOptions {
            cwd: parent_cwd,
            parent_id: parent_session_id,
            title: Some(format!("Subagent: {short_prompt}")),
            ..Default::default()
        });

        let candidate_tools = match call_ctx.session_id() {
            Some(pid) => tools.get_active_tools(pid),
            None => tools.get_active_tools(&child_session.id),
        };
        let all_infos = tools.list_all();
        let allowed_tools = compute_allowed_tools(&candidate_tools, &all_infos, mode);

        let _child_guard = ChildSessionGuard::register(
            Arc::clone(&self.active_children),
            Arc::clone(&tools),
            child_session.id.clone(),
            mode,
        );
        tools.set_active_tools(&child_session.id, &allowed_tools);

        let cfg = self.config();
        let max_steps = cfg.effective_max_steps();
        let timeout_secs = cfg.timeout_secs;
        let child_cancel = call_ctx.cancel.child_token();

        call_ctx.emit_update(
            format!("子代理启动 ({})...", mode.as_str()),
            Some(json!({
                "childSessionId": child_session.id,
                "mode": mode.as_str(),
            })),
        );

        let run_opts = RunOptions {
            provider,
            model,
            max_steps: Some(max_steps),
            origin: Some(RequestOrigin::Subagent),
            ..Default::default()
        };

        let child_id = child_session.id.clone();
        let run_fut = agent_loop.run_with_report(
            Arc::clone(&child_session),
            prompt,
            child_cancel.clone(),
            run_opts,
        );

        let outcome = tokio::select! {
            biased;
            _ = self.dispose_token.cancelled() => {
                child_cancel.cancel();
                return Ok(ToolOutput::error("子代理插件已卸载，子任务已终止").with_details(json!({
                    "childSessionId": child_id,
                    "mode": mode.as_str(),
                    "reason": "disposed",
                })));
            }
            timed = tokio::time::timeout(Duration::from_secs(timeout_secs), run_fut) => timed,
        };

        match outcome {
            Err(_) => {
                child_cancel.cancel();
                Ok(
                    ToolOutput::error(format!("子代理执行超时（已超过 {timeout_secs}s 上限）"))
                        .with_details(json!({
                            "childSessionId": child_id,
                            "mode": mode.as_str(),
                            "reason": "timeout",
                            "timeoutSecs": timeout_secs,
                        })),
                )
            }
            Ok(Ok(report)) => {
                let details = json!({
                    "childSessionId": child_id,
                    "mode": mode.as_str(),
                    "steps": report.steps,
                    "reason": report.reason.as_str(),
                });
                let content = match report.reason {
                    TurnEndReason::Completed => {
                        if report.text.trim().is_empty() {
                            format!("（子代理已完成，共 {} 步，无文本输出）", report.steps)
                        } else {
                            report.text
                        }
                    }
                    TurnEndReason::MaxSteps => format!(
                        "[子代理已达步数上限 (steps={}/{})]\n{}",
                        report.steps, max_steps, report.text
                    ),
                    TurnEndReason::Truncated => format!(
                        "[子代理输出达到 max_tokens 被截断 (steps={})]\n{}",
                        report.steps, report.text
                    ),
                    TurnEndReason::Cancelled => {
                        return Ok(ToolOutput::error("子代理已取消执行").with_details(details));
                    }
                    TurnEndReason::Error { ref message } => {
                        return Ok(ToolOutput::error(format!(
                            "子代理执行失败（steps={}）：{message}",
                            report.steps
                        ))
                        .with_details(details));
                    }
                };
                Ok(ToolOutput::text(content).with_details(details))
            }
            Ok(Err(AgentError::Llm(err)))
                if err.code == LlmErrorCode::Cancelled || child_cancel.is_cancelled() =>
            {
                Ok(ToolOutput::error("子代理已取消执行").with_details(json!({
                    "childSessionId": child_id,
                    "mode": mode.as_str(),
                    "reason": "cancelled",
                })))
            }
            Ok(Err(err)) => Ok(
                ToolOutput::error(format!("子代理执行失败：{err}")).with_details(json!({
                    "childSessionId": child_id,
                    "mode": mode.as_str(),
                    "reason": "error",
                })),
            ),
        }
    }
}

fn build_subagent_tool_definition(runtime: Arc<SubagentRuntime>) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        SUBAGENT_TOOL_NAME,
        "Delegate a focused, self-contained task to an isolated single-layer child agent session and return its final summary. Multiple read_only calls in the same step execute concurrently. Use mode=\"read_only\" (default) for codebase investigation or review, or mode=\"edit\" when file modifications are required.",
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Clear, self-contained task description for the child agent."
                },
                "mode": {
                    "type": "string",
                    "enum": ["read_only", "edit"],
                    "default": "read_only",
                    "description": "Execution mode. \"read_only\" restricts tools to read-only inspection; \"edit\" allows workspace editing tools."
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override (\"model_id\" or \"provider/model_id\")."
                }
            },
            "required": ["prompt"]
        }),
        move |args, call_ctx| {
            let rt = Arc::clone(&runtime);
            async move { rt.execute_tool_call(args, call_ctx).await }
        },
    )
    .with_exposure(ToolExposure::Direct)
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
    })
    .with_prompt_snippet(
        "Delegate an isolated single-layer sub-task to a child agent session and return its summary",
    )
}

impl Service for SubagentRuntime {
    type Config = SubagentConfig;

    fn service_name() -> &'static str {
        SERVICE_SUBAGENT
    }

    fn plugin_name() -> &'static str {
        "subagent"
    }

    fn inject() -> Inject {
        Inject::from(["tools", "sessions", "agentLoop"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let _ = ctx.tools()?;
        let _ = ctx.sessions()?;
        let _ = ctx.agent_loop()?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: Arc::new(Mutex::new(config)),
            active_children: Arc::new(Mutex::new(HashMap::new())),
            parent_turn_routes: Arc::new(Mutex::new(HashMap::new())),
            dispose_token: CancellationToken::new(),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let tools = ctx.tools()?;
            let sessions = ctx.sessions()?;

            // 1. 先在 `tools/pre-execute` 最外层挂载硬护栏，再注册工具（LIFO 卸载时先下线工具再撤护栏）
            let active_guard = Arc::clone(&self.active_children);
            let tools_guard = Arc::clone(&tools);
            let sessions_guard = Arc::clone(&sessions);
            let ctx_guard = ctx.clone();
            ctx.effect("subagent.pre-execute-guard", move || {
                let active_children = Arc::clone(&active_guard);
                let tools = Arc::clone(&tools_guard);
                let sessions = Arc::clone(&sessions_guard);
                let h = ctx_guard.on_waterfall_with(
                    EVENT_TOOLS_PRE_EXECUTE,
                    move |exec: ToolExecution,
                          next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
                        let active_children = Arc::clone(&active_children);
                        let tools = Arc::clone(&tools);
                        let sessions = Arc::clone(&sessions);
                        Box::pin(async move {
                            let decision =
                                check_pre_execute(&exec, &active_children, &tools, &sessions);
                            if !decision.allow {
                                return decision;
                            }
                            next.call(exec).await
                        })
                            as BoxFuture<'static, PreExecuteDecision>
                    },
                    EventOptions {
                        priority: -100,
                        prepend: true,
                    },
                )?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            // 2. 注册 `subagent` 工具（随插件卸载自动注销）
            let rt_tool = Arc::clone(&self);
            let tools_reg = Arc::clone(&tools);
            ctx.effect("subagent.register-tool", move || {
                let def = build_subagent_tool_definition(rt_tool);
                Ok::<_, mizar_tools::ToolError>(Some(tools_reg.register(def)?))
            })?;

            // 3. 追踪活跃父轮次路由，确保子会话在未传 `model` 时准确继承父会话的 provider/model
            let routes_in = Arc::clone(&self.parent_turn_routes);
            let active_in = Arc::clone(&self.active_children);
            let ctx_in = ctx.clone();
            ctx.effect("subagent.track-turn-input", move || {
                let h = ctx_in.on(EVENT_AGENT_TURN_INPUT, move |ev: &AgentTurnInputEvent| {
                    if !active_in.lock().unwrap().contains_key(&ev.session.id) {
                        routes_in.lock().unwrap().insert(
                            ev.session.id.clone(),
                            (ev.provider.clone(), ev.model.clone()),
                        );
                    }
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let routes_end = Arc::clone(&self.parent_turn_routes);
            let ctx_end = ctx.clone();
            ctx.effect("subagent.track-turn-end", move || {
                let h = ctx_end.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
                    routes_end.lock().unwrap().remove(&ev.session.id);
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            // 4. 最后登记 dispose_token：LIFO 卸载时优先中止在飞子代理
            let dispose_token = self.dispose_token.clone();
            ctx.effect("subagent.dispose-token", move || {
                Ok::<_, CordisError>(Some(Cleanup::sync(move || {
                    dispose_token.cancel();
                })))
            })?;

            Ok(())
        })
    }
}

/// 便捷插件别名（与无状态插件调用约定兼容：`ctx.plugin(SubagentPlugin, cfg)`）。
#[derive(Default, Clone, Copy)]
pub struct SubagentPlugin;

impl Plugin for SubagentPlugin {
    type Config = SubagentConfig;

    fn name(&self) -> Option<&str> {
        Some(SubagentRuntime::plugin_name())
    }

    fn inject(&self) -> Inject {
        SubagentRuntime::inject()
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        SubagentRuntime::plugin().apply(ctx, config)
    }
}

pub trait SubagentContextExt {
    fn subagent(&self) -> Result<Arc<SubagentRuntime>, CordisError>;
}

impl SubagentContextExt for Context {
    fn subagent(&self) -> Result<Arc<SubagentRuntime>, CordisError> {
        self.service::<SubagentRuntime>(SERVICE_SUBAGENT)
    }
}
