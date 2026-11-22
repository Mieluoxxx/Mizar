//! UNIX 进程式单层 `subagent` 插件（`"subagent"` 服务）：
//!
//! 向模型暴露窄接口只读/受控后台分身工具 `subagent(action?, prompt?, mode?, model?, name?)`，在独立子会话中
//! 执行长周期异步子任务：
//! - **后台常驻与 `send` 唤醒/纠偏**：通过 `tokio::spawn` 异步执行子会话并立即释放主对话框；支持通过 `send`
//!   向运行中分身插话（`Steer`）或唤醒已休眠的 SQLite 子会话继续执行；
//! - **零 Token Rust 活性看门狗**：订阅 `agent/*` 流式与工具事件刷新活性时间戳，连续静默超时时向父 `SessionInbox`
//!   投递告警通知（保留最后 1 格自动唤醒预算给最终结算回执），由主代理主动关照与处置；
//! - **严格深度 1 与单调代次防越权**：`ToolExposure::ModelOnly` 禁止脚本嵌套，`tools/pre-execute(-100)` 与绑定 `run_id`
//!   的 `ChildSessionGuard` 彻底消除嵌套调用、停止过渡重入与并发降权竞态；
//! - **身份隔离**：子会话标记显式 `SessionOrigin::Subagent` 与 `RequestOrigin::Subagent`，跳过预热与自动标题生成。

pub mod guardrails;
pub mod tasks;
pub mod tool;
pub mod worker;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{
    BoxError, BoxFuture, Cleanup, Context, CordisError, EventOptions, Inject, Next, Plugin,
    Service, ServicePlugin,
};
use mizar_agent_loop::{
    AgentLoopContextExt, AgentReasoningDeltaEvent, AgentStepStartEvent, AgentTextDeltaEvent,
    AgentToolCallEvent, AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent,
    AgentTurnInputEvent, EVENT_AGENT_REASONING_DELTA, EVENT_AGENT_STEP_START,
    EVENT_AGENT_TEXT_DELTA, EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT,
    EVENT_AGENT_TOOL_UPDATE, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_INPUT,
};
use mizar_session::SessionContextExt;
use mizar_system_prompt::{PromptSection, SystemPromptContextExt};
use mizar_tools::{EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolExecution, ToolsContextExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use guardrails::{
    ActiveChildrenMap, ActivityTrackersMap, check_pre_execute, format_tool_activity_label,
};
use tasks::{BackgroundTaskRegistry, ChildActivityTracker};
pub use tasks::{
    EVENT_SUBAGENT_STATE, SubagentRunStatus, SubagentStateEvent, SubagentTaskSnapshot,
};
use tool::build_subagent_tool_definition;

pub const SERVICE_SUBAGENT: &str = "subagent";
pub const SUBAGENT_TOOL_NAME: &str = "subagent";
pub const DEFAULT_SUBAGENT_MAX_STEPS: usize = 20;
pub const DEFAULT_SUBAGENT_MAX_STEPS_CAP: usize = 40;
pub const DEFAULT_SUBAGENT_TIMEOUT_SECS: u64 = 180;
pub const DEFAULT_SUBAGENT_MAX_CONCURRENT: usize = 4;
pub const DEFAULT_SUBAGENT_STALL_TIMEOUT_SECS: u64 = 45;
pub const DEFAULT_SUBAGENT_MAX_WATCHDOG_ALERTS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentConfig {
    pub default_max_steps: usize,
    pub max_steps_cap: usize,
    pub timeout_secs: u64,
    pub max_concurrent_per_parent: usize,
    pub stall_timeout_secs: u64,
    pub max_watchdog_alerts: usize,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            default_max_steps: DEFAULT_SUBAGENT_MAX_STEPS,
            max_steps_cap: DEFAULT_SUBAGENT_MAX_STEPS_CAP,
            timeout_secs: DEFAULT_SUBAGENT_TIMEOUT_SECS,
            max_concurrent_per_parent: DEFAULT_SUBAGENT_MAX_CONCURRENT,
            stall_timeout_secs: DEFAULT_SUBAGENT_STALL_TIMEOUT_SECS,
            max_watchdog_alerts: DEFAULT_SUBAGENT_MAX_WATCHDOG_ALERTS,
        }
    }
}

impl SubagentConfig {
    /// 返回经 `max_steps_cap` 封顶后的有效单轮最大步数（至少为 1）。
    pub fn effective_max_steps(&self) -> usize {
        let cap = self.max_steps_cap.max(1);
        self.default_max_steps.clamp(1, cap)
    }

    /// 返回单父会话允许并发运行的后台子代理上限（至少为 1）。
    pub fn effective_max_concurrent(&self) -> usize {
        self.max_concurrent_per_parent.max(1)
    }

    /// 返回有效单轮超时秒数（至少为 1）。
    pub fn effective_timeout_secs(&self) -> u64 {
        self.timeout_secs.max(1)
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
    #[error("无效的 action 参数 `{0}`，仅支持 \"spawn\"、\"send\"、\"status\" 或 \"stop\"")]
    InvalidAction(String),
    #[error("action 参数必须为字符串")]
    InvalidActionType,
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
    #[error(
        "无效的 name 参数 `{0}`：须为 1..=32 位小写字母/数字/`-`/`_`，且以字母或数字开头、不可为 \"all\""
    )]
    InvalidName(String),
    #[error("name 参数必须为字符串")]
    InvalidNameType,
    #[error("stop 操作缺少目标 name 参数（可传具体名称或 \"all\"）")]
    MissingStopTarget,
    #[error("send 操作缺少目标 name 参数（必须为具体子代理名称，不可为 \"all\"）")]
    MissingSendTarget,
    #[error("未找到名为 `{0}` 的子代理；请先使用 action=\"spawn\" 启动")]
    SubagentNotFound(String),
    #[error("子代理 `{0}` 正在停止收尾中，请稍后再试")]
    StoppingInProgress(String),
    #[error("不允许修改已有子代理 `{0}` 的权限模式（原 mode={1}，请求 mode={2}）")]
    ModeChangeNotAllowed(String, String, String),
    #[error(
        "子代理 `{0}` 正在运行中，无法在运行期间热切 model；请等待其休眠后 send 或 stop 后重派"
    )]
    ModelChangeWhileRunning(String),
    #[error("无法加载子代理 `{0}` 对应的会话 `{1}`")]
    SessionUnavailable(String, String),
    #[error("缺少父会话上下文，无法启动后台子代理")]
    MissingParentSession,
    #[error("同名后台子代理 `{0}` 正在运行中")]
    NameAlreadyRunning(String),
    #[error("当前会话后台子代理并发数已达上限（最多 {0} 个）")]
    ConcurrencyLimitReached(usize),
    #[error("子代理严禁嵌套调用 subagent（深度严格限制为 1）")]
    NestedSubagentDenied,
    #[error("子代理插件依赖服务不可用: {0}")]
    ServiceUnavailable(String),
}

type TurnRouteMap = HashMap<String, (Option<String>, Option<String>)>;

pub struct SubagentRuntime {
    pub(crate) ctx: Context,
    pub(crate) config: Arc<Mutex<SubagentConfig>>,
    pub(crate) active_children: Arc<Mutex<ActiveChildrenMap>>,
    pub(crate) activity_trackers: Arc<Mutex<ActivityTrackersMap>>,
    pub(crate) tasks: Arc<Mutex<BackgroundTaskRegistry>>,
    pub(crate) state_emit_lock: Arc<Mutex<()>>,
    pub(crate) parent_turn_routes: Arc<Mutex<TurnRouteMap>>,
    pub(crate) dispose_token: CancellationToken,
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
            .map(|(_, m)| *m)
    }

    /// 查询指定父会话当前仍在后台运行的子代理数量。
    pub fn running_subagents_for_parent(&self, parent_session_id: &str) -> usize {
        self.tasks.lock().unwrap().running_count(parent_session_id)
    }

    /// 列出指定父会话的后台子代理快照（运行中及最近已结束条目）。
    pub fn list_subagents_for_parent(&self, parent_session_id: &str) -> Vec<SubagentTaskSnapshot> {
        self.tasks
            .lock()
            .unwrap()
            .snapshots(parent_session_id, None)
    }

    /// 取消指定父会话名下全部仍在运行的后台子代理（供切换会话或退出清理调用）。
    pub fn cancel_parent_subagents(&self, parent_session_id: &str) -> usize {
        let stopped = self.tasks.lock().unwrap().cancel_parent(parent_session_id);
        let count = stopped.len();
        if let Some((name, child_session_id, mode)) = stopped.into_iter().last() {
            self.emit_state_event(SubagentStateEvent {
                parent_session_id: parent_session_id.to_string(),
                name,
                child_session_id,
                mode,
                status: SubagentRunStatus::Stopped,
                running_count: 0,
                notice: None,
            });
        }
        count
    }

    pub(crate) fn emit_state_event(&self, mut ev: SubagentStateEvent) {
        // 统一在 `state_emit_lock -> tasks` 锁序下现场重读真值并串行广播，防止并发结算乱序卡住状态栏。
        let _guard = self.state_emit_lock.lock().unwrap();
        ev.running_count = self
            .tasks
            .lock()
            .unwrap()
            .running_count(&ev.parent_session_id);
        self.ctx.emit(EVENT_SUBAGENT_STATE, ev);
    }

    pub(crate) fn resolve_parent_route(
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

    fn tracker_for_child(&self, session_id: &str) -> Option<Arc<ChildActivityTracker>> {
        self.activity_trackers
            .lock()
            .unwrap()
            .get(session_id)
            .map(|(_, tracker)| Arc::clone(tracker))
    }
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
        Inject::from(["tools", "sessions", "agentLoop", "systemPrompt"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let _ = ctx.tools()?;
        let _ = ctx.sessions()?;
        let _ = ctx.agent_loop()?;
        let _ = ctx.system_prompt()?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: Arc::new(Mutex::new(config)),
            active_children: Arc::new(Mutex::new(HashMap::new())),
            activity_trackers: Arc::new(Mutex::new(HashMap::new())),
            tasks: Arc::new(Mutex::new(BackgroundTaskRegistry::default())),
            state_emit_lock: Arc::new(Mutex::new(())),
            parent_turn_routes: Arc::new(Mutex::new(HashMap::new())),
            dispose_token: CancellationToken::new(),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let tools = ctx.tools()?;
            let sessions = ctx.sessions()?;
            let system_prompt = ctx.system_prompt()?;

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

            // 2. 订阅运行事件维护子会话实时活性快照（无阻塞、不影响主会话）
            let rt_step = Arc::clone(&self);
            let ctx_step = ctx.clone();
            ctx.effect("subagent.activity-step-start", move || {
                let h = ctx_step.on(EVENT_AGENT_STEP_START, move |ev: &AgentStepStartEvent| {
                    if let Some(tracker) = rt_step.tracker_for_child(&ev.session.id) {
                        tracker.set_turn_step(ev.step);
                    }
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let rt_tcall = Arc::clone(&self);
            let ctx_tcall = ctx.clone();
            ctx.effect("subagent.activity-tool-call", move || {
                let h = ctx_tcall.on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
                    if let Some(tracker) = rt_tcall.tracker_for_child(&ev.session.id) {
                        let label = format_tool_activity_label(&ev.call.name, &ev.call.arguments);
                        tracker.set_tool(&label);
                    }
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let rt_tupd = Arc::clone(&self);
            let ctx_tupd = ctx.clone();
            ctx.effect("subagent.activity-tool-update", move || {
                let h =
                    ctx_tupd.on(EVENT_AGENT_TOOL_UPDATE, move |ev: &AgentToolUpdateEvent| {
                        if let Some(tracker) = rt_tupd.tracker_for_child(&ev.session.id) {
                            if tracker.last_tool().is_none() {
                                tracker.set_tool(&ev.name);
                            } else {
                                tracker.touch();
                            }
                        }
                    })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let rt_tres = Arc::clone(&self);
            let ctx_tres = ctx.clone();
            ctx.effect("subagent.activity-tool-result", move || {
                let h =
                    ctx_tres.on(EVENT_AGENT_TOOL_RESULT, move |ev: &AgentToolResultEvent| {
                        if let Some(tracker) = rt_tres.tracker_for_child(&ev.session.id) {
                            if tracker.last_tool().is_none() {
                                tracker.set_tool(&ev.name);
                            } else {
                                tracker.touch();
                            }
                        }
                    })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let rt_text = Arc::clone(&self);
            let ctx_text = ctx.clone();
            ctx.effect("subagent.activity-text-delta", move || {
                let h = ctx_text.on(EVENT_AGENT_TEXT_DELTA, move |ev: &AgentTextDeltaEvent| {
                    if let Some(tracker) = rt_text.tracker_for_child(&ev.session.id) {
                        tracker.touch();
                    }
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            let rt_reason = Arc::clone(&self);
            let ctx_reason = ctx.clone();
            ctx.effect("subagent.activity-reasoning-delta", move || {
                let h = ctx_reason.on(
                    EVENT_AGENT_REASONING_DELTA,
                    move |ev: &AgentReasoningDeltaEvent| {
                        if let Some(tracker) = rt_reason.tracker_for_child(&ev.session.id) {
                            tracker.touch();
                        }
                    },
                )?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    h.dispose().await;
                })))
            })?;

            // 3. 注册 `subagent` 工具（随插件卸载自动注销）
            let rt_tool = Arc::clone(&self);
            let tools_reg = Arc::clone(&tools);
            ctx.effect("subagent.register-tool", move || {
                let def = build_subagent_tool_definition(rt_tool);
                Ok::<_, mizar_tools::ToolError>(Some(tools_reg.register(def)?))
            })?;

            // 4. 向系统提示词注入精简静态分流契约（KV Cache 前缀 100% 稳定，禁止无意义轮询等待）
            let sp = Arc::clone(&system_prompt);
            ctx.effect("subagent.prompt-guidance", move || {
                Ok::<_, CordisError>(Some(sp.section(PromptSection::new(
                    "subagent",
                    45,
                    || {
                        Some(
                            "Subagent delegation (`subagent` tool, only when available):\n\
- Balance responsiveness and efficiency: handle quick 1-3 step lookups directly, and delegate heavy, multi-file, log-heavy, or parallel tasks to background subagents (`action: \"spawn\"`, default) so you stay responsive to the user immediately.\n\
- Launch independent subagents together in one step, give the user a brief acknowledgement, and end your turn instead of sleeping or polling — settled results arrive as follow-up notices.\n\
- Use `action: \"send\"` (`name`, `prompt`) to steer a running subagent or wake a settled one with its retained session memory; use `action: \"status\"` for live step/tool/idle inspection or `action: \"stop\"` to cancel.\n\
- When a `subagent-watchdog:*` alert arrives, inspect the snapshot and decide whether to reassure the user and wait, redirect via `send`, or cancel via `stop`."
                                .to_string(),
                        )
                    },
                ))))
            })?;

            // 5. 追踪活跃父轮次路由，确保子会话在未传 `model` 时准确继承父会话的 provider/model
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

            // 6. 最后登记清理句柄：LIFO 卸载时先触发 dispose_token，再停止在飞任务并清零每个父会话状态栏计数
            let tasks_cleanup = Arc::clone(&self.tasks);
            let emit_lock_cleanup = Arc::clone(&self.state_emit_lock);
            let ctx_cleanup = ctx.clone();
            let dispose_token = self.dispose_token.clone();
            ctx.effect("subagent.dispose-token", move || {
                Ok::<_, CordisError>(Some(Cleanup::sync(move || {
                    dispose_token.cancel();
                    let stopped_parents = tasks_cleanup.lock().unwrap().cancel_all();
                    for (parent_session_id, name, child_session_id, mode) in stopped_parents {
                        let _guard = emit_lock_cleanup.lock().unwrap();
                        let running_count = tasks_cleanup
                            .lock()
                            .unwrap()
                            .running_count(&parent_session_id);
                        ctx_cleanup.emit(
                            EVENT_SUBAGENT_STATE,
                            SubagentStateEvent {
                                parent_session_id,
                                name,
                                child_session_id,
                                mode,
                                status: SubagentRunStatus::Stopped,
                                running_count,
                                notice: None,
                            },
                        );
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::format_background_ui_notice;

    #[test]
    fn background_ui_notice_honestly_reports_degraded_auto_wake() {
        let normal = format_background_ui_notice(
            "scout-a",
            SubagentRunStatus::Completed,
            3,
            20,
            2,
            180,
            false,
        );
        assert!(normal.contains("交由主代理汇总"), "{normal}");

        let degraded = format_background_ui_notice(
            "scout-b",
            SubagentRunStatus::Completed,
            3,
            20,
            2,
            180,
            true,
        );
        assert!(degraded.contains("连续自动唤醒上限"), "{degraded}");
        assert!(!degraded.contains("交由主代理汇总"), "{degraded}");
    }

    #[test]
    fn config_clamps_effective_timeout_and_steps() {
        let cfg = SubagentConfig {
            default_max_steps: 99,
            max_steps_cap: 15,
            timeout_secs: 0,
            max_concurrent_per_parent: 0,
            ..Default::default()
        };
        assert_eq!(cfg.effective_max_steps(), 15);
        assert_eq!(cfg.effective_timeout_secs(), 1);
        assert_eq!(cfg.effective_max_concurrent(), 1);
    }
}
