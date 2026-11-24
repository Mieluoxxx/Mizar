//! Agent 循环（对应 `NamiDSH/packages/agent-loop`）。
//!
//! 聚焦「流式调模型 → 执行工具 → 回写会话」主循环，通过构造函数静态注入
//! `LlmRuntime`、`ToolRuntime`、`SessionStore`、`SystemPrompt` 四个核心依赖，
//! 并对外暴露 `pre_step` / `request_error` / `turn_stopping` 强类型钩子与 `AgentEvent` 事件总线。

pub mod inbox;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::system_prompt::SystemPrompt;
use futures::StreamExt;
use futures::future::join_all;
pub use inbox::{
    DEFAULT_AUTO_WAKE_LIMIT, DeliveryMode, InboxItem, InboxPushResult, SessionInbox, TurnInput,
};
use mizar_llm::{
    BlockAssembler, FinishReason, GenerateOptions, ImageAttachment, LlmError, LlmErrorCode,
    LlmRuntime, Message, ReasoningEffort, RequestOrigin, StreamChunk, TokenUsage, ToolCall,
};
use mizar_session::{Session, SessionEvent, SessionStore, SessionTraceEvent, TraceSeverity};
use mizar_tools::{
    ToolExecution, ToolExecutionMode, ToolResult, ToolRuntime, ToolUpdate, ToolUpdateCallback,
};
use mizar_types::{BoxFuture, HookGuard, HookOptions, SerialHook, WaterfallHook};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub const ENVIRONMENT_CONTEXT_SOURCE: &str = "environment-context";

fn build_environment_context_notice_if_needed(
    session: &Arc<Session>,
    inputs: &[TurnInput],
) -> Option<TurnInput> {
    let already_in_inputs = inputs.iter().any(|item| {
        matches!(
            item,
            TurnInput::Notice { source, .. } if source == ENVIRONMENT_CONTEXT_SOURCE
        )
    });
    if already_in_inputs {
        return None;
    }
    let already_visible = session.with_visible_entries(|entries| {
        entries.iter().any(|e| {
            matches!(
                &e.event,
                SessionEvent::ContextNotice { source, .. } if source == ENVIRONMENT_CONTEXT_SOURCE
            )
        })
    });
    if already_visible {
        return None;
    }

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let cwd = session
        .header()
        .cwd
        .map(|raw| raw.replace('\\', "/"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let mut lines = Vec::with_capacity(4);
    lines.push("<environment_context>".to_string());
    if let Some(cwd) = cwd {
        lines.push(format!("<cwd>{cwd}</cwd>"));
    }
    lines.push(format!("<current_date>{date}</current_date>"));
    lines.push("</environment_context>".to_string());
    Some(TurnInput::notice(
        ENVIRONMENT_CONTEXT_SOURCE,
        lines.join("\n"),
    ))
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Busy,
    Idle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TurnEndReason {
    Completed,
    Truncated,
    MaxSteps,
    Cancelled,
    Error { message: String },
}

impl TurnEndReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Truncated => "truncated",
            Self::MaxSteps => "max_steps",
            Self::Cancelled => "cancelled",
            Self::Error { .. } => "error",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepOutcome {
    Completed,
    ToolCalls,
    MaxTokens,
    Cancelled,
    Error { message: String },
}

impl StepOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::ToolCalls => "tool_calls",
            Self::MaxTokens => "max_tokens",
            Self::Cancelled => "cancelled",
            Self::Error { .. } => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentStatusEvent {
    pub session: Arc<Session>,
    pub status: AgentStatus,
}

/// `agent/pre-step` waterfall 载荷：全系统唯一允许改写、展开、注入或拒收新输入的关口。
#[derive(Clone)]
pub struct AgentPreStepInput {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub cancel: CancellationToken,
    pub inputs: Vec<TurnInput>,
}

/// `agent/pre-step` waterfall 决策：放行（提交 `inputs` 入库）或拒收（不落盘、干净停步）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentPreStepDecision {
    Enter { inputs: Vec<TurnInput> },
    Reject { reason: String },
}

/// 模型请求错误分类（供 `agent/request-error` waterfall 识别自愈策略）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRequestErrorKind {
    ContextOverflow,
    StreamError,
    ProviderError,
}

/// `agent/request-error` waterfall 载荷：统一承接上下文溢出、流中断与提供方报错的恢复链路。
#[derive(Clone)]
pub struct AgentRequestErrorInput {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub kind: AgentRequestErrorKind,
    pub error_message: String,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRequestErrorDecision {
    Retry,
    Propagate,
}

/// `agent/turn-stopping` 无返回值 `serial` 事件载荷：停轮守卫可向 `SessionInbox` 投递 `Steer` 延续下一步。
#[derive(Clone)]
pub struct AgentTurnStoppingEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub cancel: CancellationToken,
    pub last_text: String,
}

#[derive(Debug, Clone)]
pub struct AgentRequestEvent {
    pub session: Arc<Session>,
    pub step: usize,
}

#[derive(Debug, Clone)]
pub struct AgentTextDeltaEvent {
    pub session: Arc<Session>,
    pub text: String,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentReasoningDeltaEvent {
    pub session: Arc<Session>,
    pub text: String,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentToolCallEvent {
    pub session: Arc<Session>,
    pub call: ToolCall,
}

#[derive(Debug, Clone)]
pub struct AgentToolUpdateEvent {
    pub session: Arc<Session>,
    pub tool_call_id: String,
    pub name: String,
    pub preview: String,
    pub details: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct AgentToolResultEvent {
    pub session: Arc<Session>,
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
    pub blocked: bool,
    pub details: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct AgentUsageEvent {
    pub session: Arc<Session>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone)]
pub struct AgentTurnStartEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentTurnInputEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentTurnEndEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub reason: TurnEndReason,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentStepStartEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub system_prompt_chars: usize,
    pub tools_count: usize,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentStepFirstTokenEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub ts_ms: u128,
}

#[derive(Debug, Clone)]
pub struct AgentStepEndEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub outcome: StepOutcome,
    pub usage: Option<TokenUsage>,
    pub ts_ms: u128,
}

/// Agent 主循环对外广播的强类型事件。
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Status(AgentStatusEvent),
    Request(AgentRequestEvent),
    TurnStart(AgentTurnStartEvent),
    TurnInput(AgentTurnInputEvent),
    TurnEnd(AgentTurnEndEvent),
    StepStart(AgentStepStartEvent),
    StepFirstToken(AgentStepFirstTokenEvent),
    StepEnd(AgentStepEndEvent),
    TextDelta(AgentTextDeltaEvent),
    ReasoningDelta(AgentReasoningDeltaEvent),
    ToolCall(AgentToolCallEvent),
    ToolUpdate(AgentToolUpdateEvent),
    ToolResult(AgentToolResultEvent),
    Usage(AgentUsageEvent),
}

const AGENT_EVENT_BROADCAST_CAPACITY: usize = 2048;

struct AgentEventBus {
    listeners: mizar_types::SyncEventBus<AgentEvent>,
    tx: tokio::sync::broadcast::Sender<AgentEvent>,
}

impl AgentEventBus {
    fn new() -> Arc<Self> {
        let (tx, _) = tokio::sync::broadcast::channel(AGENT_EVENT_BROADCAST_CAPACITY);
        Arc::new(Self {
            listeners: mizar_types::SyncEventBus::new(),
            tx,
        })
    }

    fn subscribe(&self) -> tokio::sync::broadcast::Receiver<AgentEvent> {
        self.tx.subscribe()
    }

    fn on_event(&self, cb: impl Fn(&AgentEvent) + Send + Sync + 'static) -> AgentEventGuard {
        self.listeners.on(cb)
    }

    fn emit(&self, ev: AgentEvent) {
        self.listeners.emit(&ev);
        let _ = self.tx.send(ev);
    }
}

/// 强类型 `AgentEvent` 同步订阅句柄：Drop 时自动注销监听器。
pub type AgentEventGuard = HookGuard;

#[derive(Debug, Clone, Default)]
pub struct AgentLoopConfig {
    /// 用哪个适配器，默认第一个注册的。
    pub provider: Option<String>,
    pub model: Option<String>,
    /// 单轮最大工具步数上限；默认 `None` 表示不限步数（与 pi / deepseek-harness 一致），
    /// 靠模型自然收敛、重复工具守卫与 `CancellationToken` 控制终止。
    pub max_steps: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_steps: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub images: Vec<ImageAttachment>,
    pub origin: Option<RequestOrigin>,
}

/// 单轮执行收尾报告（供 CLI / 评测精确判定收敛原因与最后一步结果）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRunReport {
    pub text: String,
    pub reason: TurnEndReason,
    pub last_step_outcome: Option<StepOutcome>,
    pub steps: usize,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("session \"{0}\" is already running")]
    SessionBusy(String),
    #[error("pre-step rejected: {0}")]
    PreStepRejected(String),
}

struct SessionLockGuard {
    inbox: Arc<SessionInbox>,
    session_id: String,
}

impl Drop for SessionLockGuard {
    fn drop(&mut self) {
        self.inbox.release_session(&self.session_id);
    }
}

fn flush_interrupted_assistant(session: &Arc<Session>, assembler: &BlockAssembler) {
    let partial = assembler.interrupted_blocks();
    if !partial.is_empty() {
        session.append(SessionEvent::Assistant {
            blocks: partial,
            replay: None,
            usage: assembler.usage(),
        });
    }
}

fn partition_tool_batches(tools: &ToolRuntime, calls: Vec<ToolCall>) -> Vec<Vec<ToolCall>> {
    let mut batches: Vec<Vec<ToolCall>> = Vec::new();
    let mut current_parallel: Vec<ToolCall> = Vec::new();
    for call in calls {
        if tools.execution_mode(&call.name) == ToolExecutionMode::Parallel {
            current_parallel.push(call);
        } else {
            if !current_parallel.is_empty() {
                batches.push(std::mem::take(&mut current_parallel));
            }
            batches.push(vec![call]);
        }
    }
    if !current_parallel.is_empty() {
        batches.push(current_parallel);
    }
    batches
}

const TRACE_ERROR_MAX_BYTES: usize = 2_000;

fn truncate_trace_error(message: &str) -> String {
    if message.len() <= TRACE_ERROR_MAX_BYTES {
        return message.to_string();
    }
    message[..message.floor_char_boundary(TRACE_ERROR_MAX_BYTES)].to_string()
}

struct TurnGuard {
    events: Arc<AgentEventBus>,
    session: Arc<Session>,
    turn: u64,
    reason: TurnEndReason,
}

impl TurnGuard {
    fn set_reason(&mut self, reason: TurnEndReason) {
        self.reason = reason;
    }
}

impl Drop for TurnGuard {
    fn drop(&mut self) {
        let error = match &self.reason {
            TurnEndReason::Error { message } if !message.is_empty() => {
                Some(truncate_trace_error(message))
            }
            _ => None,
        };
        self.session.append_trace(SessionTraceEvent::TurnEnd {
            turn: self.turn,
            reason: self.reason.as_str().to_string(),
            error,
        });
        self.events.emit(AgentEvent::TurnEnd(AgentTurnEndEvent {
            session: Arc::clone(&self.session),
            turn: self.turn,
            reason: self.reason.clone(),
            ts_ms: now_millis(),
        }));
        self.events.emit(AgentEvent::Status(AgentStatusEvent {
            session: Arc::clone(&self.session),
            status: AgentStatus::Idle,
        }));
    }
}

struct StepGuard {
    events: Arc<AgentEventBus>,
    session: Arc<Session>,
    turn: u64,
    step: usize,
    provider: Option<String>,
    model: Option<String>,
    outcome: StepOutcome,
    usage: Option<TokenUsage>,
}

impl StepGuard {
    fn finish(&mut self, outcome: StepOutcome, usage: Option<TokenUsage>) {
        self.outcome = outcome;
        self.usage = usage;
    }
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let error = match &self.outcome {
            StepOutcome::Error { message } if !message.is_empty() => {
                Some(truncate_trace_error(message))
            }
            _ => None,
        };
        self.session.append_trace(SessionTraceEvent::StepEnd {
            turn: self.turn,
            step: self.step,
            outcome: self.outcome.as_str().to_string(),
            usage: self.usage,
            error,
        });
        self.events.emit(AgentEvent::StepEnd(AgentStepEndEvent {
            session: Arc::clone(&self.session),
            turn: self.turn,
            step: self.step,
            provider: self.provider.clone(),
            model: self.model.clone(),
            outcome: self.outcome.clone(),
            usage: self.usage,
            ts_ms: now_millis(),
        }));
    }
}

const MAX_EMPTY_RESPONSE_RETRIES: usize = 2;

struct TurnRecoveryState {
    last_overflow_step: Option<usize>,
    overflow_recoveries: usize,
    error_retries: usize,
    empty_response_retries: usize,
    consecutive_short_cutoffs: usize,
}

fn parse_tool_arguments(tool_name: &str, raw_arguments: &str) -> Result<Value, String> {
    if raw_arguments.trim().is_empty() {
        return Ok(json!({}));
    }
    match serde_json::from_str::<Value>(raw_arguments) {
        Ok(val) if val.is_object() => Ok(val),
        Ok(val) => {
            let kind = match val {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            Err(format!(
                "Invalid arguments for tool '{tool_name}': expected a JSON object, got {kind}. Please retry with a valid JSON object matching the tool schema."
            ))
        }
        Err(err) => Err(format!(
            "Invalid JSON arguments for tool '{tool_name}': {err}. Please fix the JSON syntax and retry with a valid JSON object."
        )),
    }
}

pub struct AgentLoop {
    llm: Arc<LlmRuntime>,
    tools: Arc<ToolRuntime>,
    #[allow(dead_code)]
    sessions: Arc<SessionStore>,
    system_prompt: Arc<SystemPrompt>,
    config: AgentLoopConfig,
    inbox: Arc<SessionInbox>,
    short_cutoffs_by_session: Arc<Mutex<HashMap<String, usize>>>,
    events: Arc<AgentEventBus>,
    pre_step: WaterfallHook<AgentPreStepInput, BoxFuture<'static, AgentPreStepDecision>>,
    request_error:
        WaterfallHook<AgentRequestErrorInput, BoxFuture<'static, AgentRequestErrorDecision>>,
    turn_stopping: SerialHook<AgentTurnStoppingEvent, ()>,
    _env_guard: HookGuard,
}

impl AgentLoop {
    /// 静态构造 `AgentLoop` 并预装内置 `<environment_context>` 步前注入钩子（`priority = -10`）。
    pub fn new(
        llm: Arc<LlmRuntime>,
        tools: Arc<ToolRuntime>,
        sessions: Arc<SessionStore>,
        system_prompt: Arc<SystemPrompt>,
        config: AgentLoopConfig,
    ) -> Arc<Self> {
        let pre_step = WaterfallHook::new();
        let env_guard = pre_step.register_with(
            HookOptions {
                priority: -10,
                ..Default::default()
            },
            |input: AgentPreStepInput,
             next: mizar_types::Next<
                AgentPreStepInput,
                BoxFuture<'static, AgentPreStepDecision>,
            >| {
                Box::pin(async move {
                    if input.cancel.is_cancelled() {
                        return AgentPreStepDecision::Reject {
                            reason: "cancelled".to_string(),
                        };
                    }
                    let session = Arc::clone(&input.session);
                    match next.call(input).await {
                        AgentPreStepDecision::Enter { mut inputs } => {
                            if let Some(notice) =
                                build_environment_context_notice_if_needed(&session, &inputs)
                            {
                                inputs.insert(0, notice);
                            }
                            AgentPreStepDecision::Enter { inputs }
                        }
                        reject @ AgentPreStepDecision::Reject { .. } => reject,
                    }
                }) as BoxFuture<'static, AgentPreStepDecision>
            },
        );
        Arc::new(Self {
            llm,
            tools,
            sessions,
            system_prompt,
            config,
            inbox: Arc::new(SessionInbox::new()),
            short_cutoffs_by_session: Arc::new(Mutex::new(HashMap::new())),
            events: AgentEventBus::new(),
            pre_step,
            request_error: WaterfallHook::new(),
            turn_stopping: SerialHook::new(),
            _env_guard: env_guard,
        })
    }

    pub fn config(&self) -> &AgentLoopConfig {
        &self.config
    }

    pub fn pre_step_hook(
        &self,
    ) -> &WaterfallHook<AgentPreStepInput, BoxFuture<'static, AgentPreStepDecision>> {
        &self.pre_step
    }

    pub fn request_error_hook(
        &self,
    ) -> &WaterfallHook<AgentRequestErrorInput, BoxFuture<'static, AgentRequestErrorDecision>> {
        &self.request_error
    }

    pub fn turn_stopping_hook(&self) -> &SerialHook<AgentTurnStoppingEvent, ()> {
        &self.turn_stopping
    }

    /// 显式执行一次 `pre_step` 拦截链（供无需启动整轮模型推理的场景使用）。
    pub async fn run_pre_step(&self, input: AgentPreStepInput) -> AgentPreStepDecision {
        self.pre_step
            .run(input, |inp| {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            })
            .await
    }

    /// 获取纯内存会话信箱（供 TUI / 桌面端 / 宿主扩展投递 `Steer`、`FollowUp` 与 `Inject` 消息）。
    pub fn inbox(&self) -> &SessionInbox {
        &self.inbox
    }

    /// 订阅强类型 `AgentEvent` 异步广播通道（容量 2048，适合异步观察者/TUI）。
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// 注册强类型 `AgentEvent` 同步有序监听器（零丢失、按注册顺序执行，Drop 返回句柄即注销）。
    pub fn on_event(&self, cb: impl Fn(&AgentEvent) + Send + Sync + 'static) -> AgentEventGuard {
        self.events.on_event(cb)
    }

    /// 通过 `AgentEventBus` 派发强类型事件（同时通知同步监听器与广播通道）。
    pub fn emit_event(&self, ev: AgentEvent) {
        self.events.emit(ev);
    }

    pub fn is_session_busy(&self, session_id: &str) -> bool {
        self.inbox.is_session_active(session_id)
    }

    fn lock_session(&self, session_id: &str) -> Result<SessionLockGuard, AgentError> {
        if !self.inbox.try_acquire_session(session_id) {
            return Err(AgentError::SessionBusy(session_id.to_string()));
        }
        Ok(SessionLockGuard {
            inbox: Arc::clone(&self.inbox),
            session_id: session_id.to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_request_error(
        &self,
        session: &Arc<Session>,
        turn: u64,
        step: usize,
        provider: &Option<String>,
        model: &Option<String>,
        cancel: &CancellationToken,
        kind: AgentRequestErrorKind,
        error_message: String,
    ) -> AgentRequestErrorDecision {
        self.request_error
            .run(
                AgentRequestErrorInput {
                    session: Arc::clone(session),
                    turn,
                    step,
                    provider: provider.clone(),
                    model: model.clone(),
                    kind,
                    error_message,
                    cancel: cancel.clone(),
                },
                |_| Box::pin(async { AgentRequestErrorDecision::Propagate }),
            )
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_recover_request_error(
        &self,
        session: &Arc<Session>,
        turn: u64,
        step: usize,
        provider: &Option<String>,
        model: &Option<String>,
        cancel: &CancellationToken,
        kind: AgentRequestErrorKind,
        error_message: String,
        recovery: &mut TurnRecoveryState,
    ) -> bool {
        if cancel.is_cancelled() {
            return false;
        }
        let allowed = match kind {
            AgentRequestErrorKind::ContextOverflow => {
                recovery.last_overflow_step != Some(step) && recovery.overflow_recoveries < 3
            }
            AgentRequestErrorKind::StreamError | AgentRequestErrorKind::ProviderError => {
                recovery.error_retries < 3
            }
        };
        if !allowed {
            return false;
        }
        let decision = self
            .dispatch_request_error(
                session,
                turn,
                step,
                provider,
                model,
                cancel,
                kind,
                error_message,
            )
            .await;
        if decision != AgentRequestErrorDecision::Retry {
            return false;
        }
        match kind {
            AgentRequestErrorKind::ContextOverflow => {
                recovery.last_overflow_step = Some(step);
                recovery.overflow_recoveries += 1;
                recovery.consecutive_short_cutoffs = 0;
                self.short_cutoffs_by_session
                    .lock()
                    .unwrap()
                    .remove(&session.id);
            }
            AgentRequestErrorKind::StreamError | AgentRequestErrorKind::ProviderError => {
                recovery.error_retries += 1;
            }
        }
        true
    }

    /// 跑一次：用户输入 → [调模型 → 有工具调用就执行 → 结果写回会话 → 再调模型] → 最终回答。
    /// 过程中发出 `agent/*` 事件，网页端、TUI 和轨迹检测插件靠它们拿到实时流和生命周期边界。
    pub async fn run(
        &self,
        session: Arc<Session>,
        input: impl Into<TurnInput>,
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        self.run_with_options(session, input, cancel, RunOptions::default())
            .await
    }

    /// 与 [`Self::run`] 相同，但允许在单次调用时覆盖 `provider`、`model`、`reasoning_effort` 等参数。
    pub async fn run_with_options(
        &self,
        session: Arc<Session>,
        input: impl Into<TurnInput>,
        cancel: CancellationToken,
        options: RunOptions,
    ) -> Result<String, AgentError> {
        self.run_with_report(session, input, cancel, options)
            .await
            .map(|report| report.text)
    }

    /// 执行一轮对话并返回完整收尾报告（含终止原因与最后一步结果），供无头模式判定退出码。
    pub async fn run_with_report(
        &self,
        session: Arc<Session>,
        input: impl Into<TurnInput>,
        cancel: CancellationToken,
        options: RunOptions,
    ) -> Result<TurnRunReport, AgentError> {
        let _session_lock = self.lock_session(&session.id)?;
        let res = self
            .run_turn_locked(Arc::clone(&session), input.into(), cancel, options)
            .await;
        let _ = session.flush_async().await;
        res
    }

    async fn run_turn_locked(
        &self,
        session: Arc<Session>,
        mut entry_input: TurnInput,
        cancel: CancellationToken,
        options: RunOptions,
    ) -> Result<TurnRunReport, AgentError> {
        let llm = Arc::clone(&self.llm);
        let tools = Arc::clone(&self.tools);
        let system_prompt = Arc::clone(&self.system_prompt);

        let provider = options.provider.or_else(|| self.config.provider.clone());
        let model = options.model.or_else(|| self.config.model.clone());
        let max_steps = options
            .max_steps
            .or(self.config.max_steps)
            .map(|n| n.max(1));
        let max_tokens = options.max_tokens.or(self.config.max_tokens);
        let temperature = options.temperature.or(self.config.temperature);
        let reasoning_effort = options.reasoning_effort.or(self.config.reasoning_effort);
        let origin_override = options.origin;
        if !options.images.is_empty() {
            match &mut entry_input {
                TurnInput::User { images, .. } | TurnInput::Notice { images, .. } => {
                    images.extend(options.images);
                }
            }
        }
        if entry_input.is_user() {
            self.inbox.record_human_input(&session.id);
        }

        let turn = session.next_turn();
        session.append_trace(SessionTraceEvent::TurnStart { turn });
        self.emit_event(AgentEvent::TurnStart(AgentTurnStartEvent {
            session: Arc::clone(&session),
            turn,
            ts_ms: now_millis(),
        }));
        self.emit_event(AgentEvent::Status(AgentStatusEvent {
            session: Arc::clone(&session),
            status: AgentStatus::Busy,
        }));
        let mut turn_guard = TurnGuard {
            events: Arc::clone(&self.events),
            session: Arc::clone(&session),
            turn,
            reason: TurnEndReason::Cancelled,
        };

        let mut pending_entry = Some(entry_input);
        let mut turn_input_emitted = false;
        let mut step = 0usize;
        let mut steps_executed = 0usize;
        let mut last_step_outcome: Option<StepOutcome> = None;
        let mut notice_steers_this_turn = 0usize;
        let initial_cutoffs = self
            .short_cutoffs_by_session
            .lock()
            .unwrap()
            .get(&session.id)
            .copied()
            .unwrap_or(0);
        let mut recovery = TurnRecoveryState {
            last_overflow_step: None,
            overflow_recoveries: 0,
            error_retries: 0,
            empty_response_retries: 0,
            consecutive_short_cutoffs: initial_cutoffs,
        };
        loop {
            step += 1;
            if cancel.is_cancelled() {
                turn_guard.set_reason(TurnEndReason::Cancelled);
                return Err(AgentError::Llm(LlmError::cancelled()));
            }
            if let Some(limit) = max_steps
                && step > limit
            {
                break;
            }

            let mut step_inputs = Vec::new();
            if let Some(entry) = pending_entry.take() {
                step_inputs.push(entry);
            }
            step_inputs.extend(self.inbox.drain_step_inputs(&session.id));

            let pre_step_decision = self
                .run_pre_step(AgentPreStepInput {
                    session: Arc::clone(&session),
                    turn,
                    step,
                    provider: provider.clone(),
                    model: model.clone(),
                    cancel: cancel.clone(),
                    inputs: step_inputs,
                })
                .await;

            if cancel.is_cancelled() {
                turn_guard.set_reason(TurnEndReason::Cancelled);
                return Err(AgentError::Llm(LlmError::cancelled()));
            }

            match pre_step_decision {
                AgentPreStepDecision::Enter { inputs } => {
                    for item in inputs {
                        session.append(item.into_session_event());
                    }
                    if !turn_input_emitted {
                        turn_input_emitted = true;
                        self.emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
                            session: Arc::clone(&session),
                            turn,
                            provider: provider.clone(),
                            model: model.clone(),
                            ts_ms: now_millis(),
                        }));
                    }
                }
                AgentPreStepDecision::Reject { reason } => {
                    turn_guard.set_reason(TurnEndReason::Error {
                        message: reason.clone(),
                    });
                    return Err(AgentError::PreStepRejected(reason));
                }
            }

            steps_executed += 1;
            self.emit_event(AgentEvent::Request(AgentRequestEvent {
                session: Arc::clone(&session),
                step,
            }));

            let prompt_text = system_prompt.build_for_session(Some(&session.id));
            let tool_list = tools.list_declared(Some(&session.id));
            let system_prompt_chars = prompt_text.chars().count();
            let tools_count = tool_list.len();
            session.append_trace(SessionTraceEvent::StepStart {
                turn,
                step,
                system_prompt_chars,
                tools_count,
            });
            self.emit_event(AgentEvent::StepStart(AgentStepStartEvent {
                session: Arc::clone(&session),
                turn,
                step,
                system_prompt_chars,
                tools_count,
                ts_ms: now_millis(),
            }));
            let mut step_guard = StepGuard {
                events: Arc::clone(&self.events),
                session: Arc::clone(&session),
                turn,
                step,
                provider: provider.clone(),
                model: model.clone(),
                outcome: StepOutcome::Cancelled,
                usage: None,
            };

            let mut messages = vec![Message::System {
                content: prompt_text,
            }];
            messages.extend(session.derive_messages());

            let mut stream = match llm.stream(GenerateOptions {
                messages,
                tools: tool_list,
                model: model.clone(),
                provider: provider.clone(),
                system: None,
                max_tokens,
                temperature,
                reasoning_effort,
                headers: Vec::new(),
                session_id: Some(session.id.clone()),
                cache_retention: None,
                cancel: cancel.clone(),
                origin: origin_override
                    .clone()
                    .unwrap_or_else(|| RequestOrigin::SessionTurn {
                        session_id: session.id.clone(),
                        turn,
                        step,
                    }),
                disable_continuation: false,
            }) {
                Ok(s) => s,
                Err(err) => {
                    if err.code == LlmErrorCode::Cancelled {
                        step_guard.finish(StepOutcome::Cancelled, None);
                        turn_guard.set_reason(TurnEndReason::Cancelled);
                        return Err(AgentError::Llm(err));
                    }
                    let kind = if err.code == LlmErrorCode::ContextWindowExceeded {
                        AgentRequestErrorKind::ContextOverflow
                    } else {
                        AgentRequestErrorKind::ProviderError
                    };
                    if self
                        .try_recover_request_error(
                            &session,
                            turn,
                            step,
                            &provider,
                            &model,
                            &cancel,
                            kind,
                            err.to_string(),
                            &mut recovery,
                        )
                        .await
                    {
                        step_guard.finish(
                            StepOutcome::Error {
                                message: "request error recovered; retrying".to_string(),
                            },
                            None,
                        );
                        drop(step_guard);
                        step = step.saturating_sub(1);
                        continue;
                    }
                    let msg = err.to_string();
                    step_guard.finish(
                        StepOutcome::Error {
                            message: msg.clone(),
                        },
                        None,
                    );
                    turn_guard.set_reason(TurnEndReason::Error { message: msg });
                    return Err(AgentError::Llm(err));
                }
            };

            let mut assembler = BlockAssembler::new();
            let mut first_token_emitted = false;
            let mut stream_err: Option<LlmError> = None;
            while let Some(chunk_res) = stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(err) => {
                        if err.code == LlmErrorCode::Cancelled {
                            flush_interrupted_assistant(&session, &assembler);
                            step_guard.finish(StepOutcome::Cancelled, assembler.usage());
                            turn_guard.set_reason(TurnEndReason::Cancelled);
                            return Err(AgentError::Llm(err));
                        }
                        stream_err = Some(err);
                        break;
                    }
                };
                match &chunk {
                    StreamChunk::TextDelta { text, .. } if !text.is_empty() => {
                        if !first_token_emitted {
                            first_token_emitted = true;
                            session.append_trace(SessionTraceEvent::FirstToken { turn, step });
                            self.emit_event(AgentEvent::StepFirstToken(AgentStepFirstTokenEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                ts_ms: now_millis(),
                            }));
                        }
                        self.emit_event(AgentEvent::TextDelta(AgentTextDeltaEvent {
                            session: Arc::clone(&session),
                            text: text.clone(),
                            ts_ms: now_millis(),
                        }));
                    }
                    StreamChunk::ReasoningDelta { text, .. } if !text.is_empty() => {
                        if !first_token_emitted {
                            first_token_emitted = true;
                            session.append_trace(SessionTraceEvent::FirstToken { turn, step });
                            self.emit_event(AgentEvent::StepFirstToken(AgentStepFirstTokenEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                ts_ms: now_millis(),
                            }));
                        }
                        self.emit_event(AgentEvent::ReasoningDelta(AgentReasoningDeltaEvent {
                            session: Arc::clone(&session),
                            text: text.clone(),
                            ts_ms: now_millis(),
                        }));
                    }
                    StreamChunk::ToolCallDelta {
                        id,
                        name,
                        arguments_delta,
                        ..
                    } if !arguments_delta.is_empty()
                        || id.as_deref().is_some_and(|s| !s.is_empty())
                        || name.as_deref().is_some_and(|s| !s.is_empty()) =>
                    {
                        if !first_token_emitted {
                            first_token_emitted = true;
                            session.append_trace(SessionTraceEvent::FirstToken { turn, step });
                            self.emit_event(AgentEvent::StepFirstToken(AgentStepFirstTokenEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                ts_ms: now_millis(),
                            }));
                        }
                    }
                    StreamChunk::BlockEnd {
                        block: mizar_llm::ContentBlock::ToolCall { .. },
                        ..
                    } => {
                        if !first_token_emitted {
                            first_token_emitted = true;
                            session.append_trace(SessionTraceEvent::FirstToken { turn, step });
                            self.emit_event(AgentEvent::StepFirstToken(AgentStepFirstTokenEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                ts_ms: now_millis(),
                            }));
                        }
                    }
                    StreamChunk::Usage { usage } => {
                        self.emit_event(AgentEvent::Usage(AgentUsageEvent {
                            session: Arc::clone(&session),
                            usage: *usage,
                        }));
                    }
                    _ => {}
                }
                assembler.push(chunk);
            }

            if let Some(err) = stream_err {
                let kind = if err.code == LlmErrorCode::ContextWindowExceeded {
                    AgentRequestErrorKind::ContextOverflow
                } else {
                    AgentRequestErrorKind::StreamError
                };
                if self
                    .try_recover_request_error(
                        &session,
                        turn,
                        step,
                        &provider,
                        &model,
                        &cancel,
                        kind,
                        err.to_string(),
                        &mut recovery,
                    )
                    .await
                    || (!cancel.is_cancelled()
                        && err.code == LlmErrorCode::EmptyResponse
                        && recovery.empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES)
                {
                    if err.code == LlmErrorCode::EmptyResponse {
                        recovery.empty_response_retries += 1;
                    }
                    step_guard.finish(
                        StepOutcome::Error {
                            message: "stream error recovered; retrying".to_string(),
                        },
                        None,
                    );
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
                let msg = err.to_string();
                step_guard.finish(
                    StepOutcome::Error {
                        message: msg.clone(),
                    },
                    None,
                );
                turn_guard.set_reason(TurnEndReason::Error { message: msg });
                return Err(AgentError::Llm(err));
            }

            let finish_reason = assembler.finish();
            match &finish_reason {
                FinishReason::Aborted => {
                    flush_interrupted_assistant(&session, &assembler);
                    step_guard.finish(StepOutcome::Cancelled, assembler.usage());
                    turn_guard.set_reason(TurnEndReason::Cancelled);
                    return Err(AgentError::Llm(LlmError::cancelled()));
                }
                FinishReason::Error {
                    code,
                    message,
                    status,
                    retry_after_ms,
                } => {
                    let err = LlmError {
                        code: *code,
                        message: message.clone(),
                        status: *status,
                        retry_after_ms: *retry_after_ms,
                    };
                    if err.code == LlmErrorCode::Cancelled {
                        flush_interrupted_assistant(&session, &assembler);
                        step_guard.finish(StepOutcome::Cancelled, assembler.usage());
                        turn_guard.set_reason(TurnEndReason::Cancelled);
                        return Err(AgentError::Llm(err));
                    }
                    let kind = if err.code == LlmErrorCode::ContextWindowExceeded {
                        AgentRequestErrorKind::ContextOverflow
                    } else {
                        AgentRequestErrorKind::StreamError
                    };
                    if self
                        .try_recover_request_error(
                            &session,
                            turn,
                            step,
                            &provider,
                            &model,
                            &cancel,
                            kind,
                            err.to_string(),
                            &mut recovery,
                        )
                        .await
                        || (!cancel.is_cancelled()
                            && err.code == LlmErrorCode::EmptyResponse
                            && recovery.empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES)
                    {
                        if err.code == LlmErrorCode::EmptyResponse {
                            recovery.empty_response_retries += 1;
                        }
                        let outcome = StepOutcome::Error {
                            message: "request error recovered; retrying".to_string(),
                        };
                        last_step_outcome = Some(outcome.clone());
                        step_guard.finish(outcome, None);
                        drop(step_guard);
                        step = step.saturating_sub(1);
                        continue;
                    }
                    let msg = err.to_string();
                    step_guard.finish(
                        StepOutcome::Error {
                            message: msg.clone(),
                        },
                        None,
                    );
                    turn_guard.set_reason(TurnEndReason::Error { message: msg });
                    return Err(AgentError::Llm(err));
                }
                FinishReason::Stop | FinishReason::ToolCalls | FinishReason::MaxTokens => {}
            }

            let text = assembler.text();
            let calls = assembler.tool_calls();
            let usage = assembler.usage();
            let blocks = assembler.blocks();
            let replay = assembler.replay();

            // 识别隐性小窗口截断：0 输出 `MaxTokens`（pi）或连续两次极短 `MaxTokens`（DS-Bot `cutOffs >= 2`）
            let is_zero_len_cutoff = matches!(finish_reason, FinishReason::MaxTokens)
                && calls.is_empty()
                && blocks.is_empty();
            let is_short_cutoff = matches!(finish_reason, FinishReason::MaxTokens)
                && calls.is_empty()
                && text.chars().count() < 200;
            if is_short_cutoff {
                recovery.consecutive_short_cutoffs += 1;
                self.short_cutoffs_by_session
                    .lock()
                    .unwrap()
                    .insert(session.id.clone(), recovery.consecutive_short_cutoffs);
            } else {
                recovery.consecutive_short_cutoffs = 0;
                self.short_cutoffs_by_session
                    .lock()
                    .unwrap()
                    .remove(&session.id);
            }

            if is_zero_len_cutoff || recovery.consecutive_short_cutoffs >= 2 {
                let reason = if is_zero_len_cutoff {
                    "zero-output max_tokens cutoff (suspected context window overflow)".to_string()
                } else {
                    format!(
                        "consecutive short max_tokens cutoffs ({}, suspected implicit small context window)",
                        recovery.consecutive_short_cutoffs
                    )
                };
                if self
                    .try_recover_request_error(
                        &session,
                        turn,
                        step,
                        &provider,
                        &model,
                        &cancel,
                        AgentRequestErrorKind::ContextOverflow,
                        reason,
                        &mut recovery,
                    )
                    .await
                {
                    let outcome = StepOutcome::Error {
                        message: "implicit cutoff overflow recovered; retrying".to_string(),
                    };
                    last_step_outcome = Some(outcome.clone());
                    step_guard.finish(outcome, usage);
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
            }

            let has_visible_text = blocks.iter().any(|b| match b {
                mizar_llm::ContentBlock::Text { text } => !text.trim().is_empty(),
                _ => false,
            });
            if calls.is_empty()
                && !has_visible_text
                && !matches!(finish_reason, FinishReason::MaxTokens)
            {
                let detail = if blocks.is_empty() {
                    "model output must contain either output text or tool calls"
                } else {
                    "model output contained only reasoning without visible text or tool calls"
                };
                if self
                    .try_recover_request_error(
                        &session,
                        turn,
                        step,
                        &provider,
                        &model,
                        &cancel,
                        AgentRequestErrorKind::StreamError,
                        detail.to_string(),
                        &mut recovery,
                    )
                    .await
                    || (!cancel.is_cancelled()
                        && recovery.empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES)
                {
                    recovery.empty_response_retries += 1;
                    let outcome = StepOutcome::Error {
                        message: "empty response; retrying".to_string(),
                    };
                    last_step_outcome = Some(outcome.clone());
                    step_guard.finish(outcome, usage);
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
                let err = LlmError::new(LlmErrorCode::EmptyResponse, detail);
                let msg = err.to_string();
                step_guard.finish(
                    StepOutcome::Error {
                        message: msg.clone(),
                    },
                    usage,
                );
                turn_guard.set_reason(TurnEndReason::Error { message: msg });
                return Err(AgentError::Llm(err));
            }

            recovery.empty_response_retries = 0;
            session.append(SessionEvent::Assistant {
                blocks,
                replay,
                usage,
            });

            let is_truncated = matches!(finish_reason, FinishReason::MaxTokens);
            let step_outcome = if !calls.is_empty() {
                StepOutcome::ToolCalls
            } else if is_truncated {
                StepOutcome::MaxTokens
            } else {
                StepOutcome::Completed
            };
            last_step_outcome = Some(step_outcome.clone());
            step_guard.finish(step_outcome, usage);
            drop(step_guard);

            if calls.is_empty() {
                if !is_truncated && has_visible_text {
                    if !self.inbox.has_pending_steer(&session.id) && !cancel.is_cancelled() {
                        let _: Option<()> = self
                            .turn_stopping
                            .run(AgentTurnStoppingEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                provider: provider.clone(),
                                model: model.clone(),
                                cancel: cancel.clone(),
                                last_text: text.clone(),
                            })
                            .await;
                    }
                    if cancel.is_cancelled() {
                        turn_guard.set_reason(TurnEndReason::Cancelled);
                        return Err(AgentError::Llm(LlmError::cancelled()));
                    }
                    if self.inbox.has_pending_steer(&session.id) {
                        if self.inbox.has_pending_user_steer(&session.id) {
                            continue;
                        }
                        notice_steers_this_turn += 1;
                        let budget = self.inbox.auto_wake_limit(&session.id);
                        if notice_steers_this_turn <= budget {
                            continue;
                        }
                        // 超出每轮自动插话预算时将 Notice Steer 降级为 Inject，防止释放锁后再次唤醒新轮。
                        self.inbox.degrade_notice_steers_to_inject(&session.id);
                    }
                }
                let reason = if is_truncated || !has_visible_text {
                    TurnEndReason::Truncated
                } else {
                    TurnEndReason::Completed
                };
                turn_guard.set_reason(reason.clone());
                return Ok(TurnRunReport {
                    text,
                    reason,
                    last_step_outcome,
                    steps: steps_executed,
                });
            }

            let batches = partition_tool_batches(&tools, calls);
            let mut end_turn_message = None;
            for (batch_idx, batch) in batches.iter().enumerate() {
                if cancel.is_cancelled() {
                    for pending in batches[batch_idx..].iter().flatten() {
                        let content = "已取消执行".to_string();
                        self.emit_event(AgentEvent::ToolResult(AgentToolResultEvent {
                            session: Arc::clone(&session),
                            tool_call_id: pending.id.clone(),
                            name: pending.name.clone(),
                            content: content.clone(),
                            is_error: true,
                            blocked: false,
                            details: None,
                        }));
                        session.append(SessionEvent::ToolResult {
                            tool_call_id: pending.id.clone(),
                            name: pending.name.clone(),
                            content,
                            is_error: true,
                        });
                    }
                    turn_guard.set_reason(TurnEndReason::Cancelled);
                    return Err(AgentError::Llm(LlmError::cancelled()));
                }

                let mut results: Vec<Option<ToolResult>> = vec![None; batch.len()];
                let mut futs = Vec::new();
                let mut fut_indices = Vec::new();
                for (idx, call) in batch.iter().enumerate() {
                    session.append_trace(SessionTraceEvent::ToolCall {
                        turn,
                        step,
                        call_id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    });
                    self.emit_event(AgentEvent::ToolCall(AgentToolCallEvent {
                        session: Arc::clone(&session),
                        call: call.clone(),
                    }));
                    let args = match parse_tool_arguments(&call.name, &call.arguments) {
                        Ok(args) => args,
                        Err(msg) => {
                            // 畸形工具参数不再吞成 `{}`，直接返回校验错误供模型自我修正重试。
                            results[idx] = Some(ToolResult::err(msg));
                            continue;
                        }
                    };
                    let update_events = Arc::clone(&self.events);
                    let update_session = Arc::clone(&session);
                    let call_id_for_upd = call.id.clone();
                    let call_name_for_upd = call.name.clone();
                    let on_update: ToolUpdateCallback = Arc::new(move |upd: ToolUpdate| {
                        update_events.emit(AgentEvent::ToolUpdate(AgentToolUpdateEvent {
                            session: Arc::clone(&update_session),
                            tool_call_id: call_id_for_upd.clone(),
                            name: call_name_for_upd.clone(),
                            preview: upd.preview,
                            details: upd.details,
                        }));
                    });
                    fut_indices.push(idx);
                    futs.push(
                        tools.execute_with(
                            ToolExecution::new(call.name.clone(), args, Some(session.id.clone()))
                                .with_call_id(call.id.clone()),
                            cancel.clone(),
                            Some(on_update),
                        ),
                    );
                }

                if futs.len() == 1 {
                    let idx = fut_indices[0];
                    results[idx] = Some(futs.into_iter().next().unwrap().await);
                } else if !futs.is_empty() {
                    for (idx, res) in fut_indices.into_iter().zip(join_all(futs).await) {
                        results[idx] = Some(res);
                    }
                }
                let results: Vec<ToolResult> = results
                    .into_iter()
                    .map(|r| r.expect("tool result populated"))
                    .collect();

                for (call, result) in batch.iter().zip(results) {
                    if result.end_turn && !result.is_error && !result.blocked {
                        end_turn_message.get_or_insert_with(|| result.content.clone());
                    }
                    let extra_contexts = result.additional_contexts.clone();
                    self.emit_event(AgentEvent::ToolResult(AgentToolResultEvent {
                        session: Arc::clone(&session),
                        tool_call_id: call.id.clone(),
                        name: call.name.clone(),
                        content: result.content.clone(),
                        is_error: result.is_error,
                        blocked: result.blocked,
                        details: result.details,
                    }));
                    if result.blocked {
                        session.append_trace(SessionTraceEvent::ToolOutcome {
                            tool_call_id: call.id.clone(),
                            blocked: true,
                        });
                    }
                    session.append(SessionEvent::ToolResult {
                        tool_call_id: call.id.clone(),
                        name: call.name.clone(),
                        content: result.content,
                        is_error: result.is_error,
                    });
                    for note in extra_contexts {
                        session.append(SessionEvent::context_notice(note.source, note.text));
                    }
                }
            }
            // 完整写回同一模型请求的工具结果，并将 end_turn 文本作为正式助手输出发射与落盘；不清空等待交付的 FollowUp。
            if let Some(message) = end_turn_message {
                if cancel.is_cancelled() {
                    turn_guard.set_reason(TurnEndReason::Cancelled);
                    return Err(AgentError::Llm(LlmError::cancelled()));
                }
                if !message.trim().is_empty() {
                    self.emit_event(AgentEvent::TextDelta(AgentTextDeltaEvent {
                        session: Arc::clone(&session),
                        text: message.clone(),
                        ts_ms: now_millis(),
                    }));
                    session.append(SessionEvent::Assistant {
                        blocks: vec![mizar_llm::ContentBlock::Text {
                            text: message.clone(),
                        }],
                        replay: None,
                        usage: None,
                    });
                }
                turn_guard.set_reason(TurnEndReason::Completed);
                return Ok(TurnRunReport {
                    text: message,
                    reason: TurnEndReason::Completed,
                    last_step_outcome,
                    steps: steps_executed,
                });
            }
        }

        turn_guard.set_reason(TurnEndReason::MaxSteps);
        let msg = format!(
            "（已达到 max_steps={} 上限并停止，当前进度已保留，发送「继续」即可接着执行）",
            max_steps.unwrap_or(steps_executed)
        );
        session.append_trace(SessionTraceEvent::TurnNotice {
            turn,
            severity: TraceSeverity::Warn,
            text: msg.clone(),
        });
        Ok(TurnRunReport {
            text: msg,
            reason: TurnEndReason::MaxSteps,
            last_step_outcome,
            steps: steps_executed,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::system_prompt::{SystemPrompt, SystemPromptConfig};
    use futures::channel::mpsc;
    use mizar_llm::{BoxChunkStream, LlmAdapter, LlmRuntime};
    use mizar_session::SessionStore;
    use mizar_tools::{ToolDefinition, ToolRuntime};
    use mizar_types::{BoxError, Next};

    use super::*;

    struct TestHarness {
        #[allow(dead_code)]
        llm: Arc<LlmRuntime>,
        sessions: Arc<SessionStore>,
        tools: Arc<ToolRuntime>,
        #[allow(dead_code)]
        system_prompt: Arc<SystemPrompt>,
        agent_loop: Arc<AgentLoop>,
        _adapter_guard: HookGuard,
    }

    fn setup_app_with_adapter<A: LlmAdapter>(adapter: A) -> TestHarness {
        let llm = LlmRuntime::new();
        let sessions = SessionStore::new();
        let tools = ToolRuntime::new();
        let system_prompt = SystemPrompt::new(SystemPromptConfig::default());
        let agent_loop = AgentLoop::new(
            Arc::clone(&llm),
            Arc::clone(&tools),
            Arc::clone(&sessions),
            Arc::clone(&system_prompt),
            AgentLoopConfig::default(),
        );
        let adapter_guard = llm.register_adapter("test", Arc::new(adapter)).unwrap();
        TestHarness {
            llm,
            sessions,
            tools,
            system_prompt,
            agent_loop,
            _adapter_guard: adapter_guard,
        }
    }

    #[tokio::test]
    async fn preserves_interrupted_partial_blocks_and_overrides_model() {
        struct CancelStreamAdapter {
            seen_models: Arc<Mutex<Vec<Option<String>>>>,
        }

        impl LlmAdapter for CancelStreamAdapter {
            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.seen_models.lock().unwrap().push(options.model.clone());
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                        index: 0,
                        text: "思考了一半".into(),
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                        index: 1,
                        text: "回答了一半".into(),
                    }));
                    options.cancel.cancelled().await;
                    let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                });
                Box::pin(rx)
            }
        }

        let seen_models = Arc::new(Mutex::new(Vec::new()));
        let app = setup_app_with_adapter(CancelStreamAdapter {
            seen_models: Arc::clone(&seen_models),
        });

        let session = app.sessions.create(None);
        let cancel = CancellationToken::new();
        let cancel_trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel_trigger.cancel();
        });

        let err = app
            .agent_loop
            .run_with_options(
                Arc::clone(&session),
                "测试中断",
                cancel,
                RunOptions {
                    model: Some("custom-model".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(err, AgentError::Llm(e) if e.code == LlmErrorCode::Cancelled));
        assert_eq!(
            *seen_models.lock().unwrap(),
            vec![Some("custom-model".into())]
        );

        let log = session.log();
        assert_eq!(log.len(), 3);
        match &log[2].event {
            SessionEvent::Assistant { blocks, .. } => {
                assert_eq!(
                    blocks,
                    &[
                        mizar_llm::ContentBlock::Reasoning {
                            text: "思考了一半".into()
                        },
                        mizar_llm::ContentBlock::Text {
                            text: "回答了一半".into()
                        }
                    ]
                );
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_concurrent_runs_on_same_session_and_empty_output() {
        struct SlowOrEmptyAdapter(AtomicUsize);

        impl LlmAdapter for SlowOrEmptyAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let idx = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if idx == 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("ok")));
                    }
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(SlowOrEmptyAdapter(AtomicUsize::new(0)));
        let session = app.sessions.create(None);
        let agent = Arc::clone(&app.agent_loop);

        let s1 = Arc::clone(&session);
        let a1 = Arc::clone(&agent);
        let h1 = tokio::spawn(async move { a1.run(s1, "first", CancellationToken::new()).await });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let busy_err = agent
            .run(Arc::clone(&session), "second", CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(busy_err, AgentError::SessionBusy(_)));

        assert_eq!(h1.await.unwrap().unwrap(), "ok");

        // 第二次流返回空块应触发 EmptyResponse 错误，不写入空助手消息
        let empty_err = agent
            .run(Arc::clone(&session), "third", CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(empty_err, AgentError::Llm(e) if e.code == LlmErrorCode::EmptyResponse));
        assert_eq!(session.log().len(), 4); // environment-context + first user + first assistant + third user
    }

    #[tokio::test]
    async fn runs_parallel_tools_concurrently_and_forwards_updates() {
        use mizar_tools::ToolDefinition;
        use std::sync::atomic::AtomicBool;

        struct MultiCallAdapter(AtomicUsize);
        impl LlmAdapter for MultiCallAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let step = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if step == 0 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "c1".into(),
                                name: "par_tool".into(),
                                arguments: r#"{"id":1}"#.into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            1,
                            ToolCall {
                                id: "c2".into(),
                                name: "par_tool".into(),
                                arguments: r#"{"id":2}"#.into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("all done")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(MultiCallAdapter(AtomicUsize::new(0)));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicBool::new(false));
        let updates = Arc::new(Mutex::new(Vec::<String>::new()));

        let updates_sub = Arc::clone(&updates);
        let _updates_guard = app.agent_loop.on_event(move |ev| {
            if let AgentEvent::ToolUpdate(ev) = ev {
                updates_sub.lock().unwrap().push(ev.preview.clone());
            }
        });

        let tools = &app.tools;
        let in_flight_reg = Arc::clone(&in_flight);
        let overlapped_reg = Arc::clone(&overlapped);
        let _reg = tools
            .register(
                ToolDefinition::async_rich_tool(
                    "par_tool",
                    "parallel test",
                    json!({}),
                    move |args, call_ctx| {
                        let cur = Arc::clone(&in_flight_reg);
                        let ov = Arc::clone(&overlapped_reg);
                        async move {
                            call_ctx.emit_update("running...", None);
                            let now = cur.fetch_add(1, Ordering::SeqCst) + 1;
                            if now >= 2 {
                                ov.store(true, Ordering::SeqCst);
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                            cur.fetch_sub(1, Ordering::SeqCst);
                            Ok::<_, BoxError>(mizar_tools::ToolOutput::text(format!(
                                "id={}",
                                args["id"]
                            )))
                        }
                    },
                )
                .with_execution_mode(ToolExecutionMode::Parallel),
            )
            .unwrap();

        let session = app.sessions.create(None);
        let answer = app
            .agent_loop
            .run(Arc::clone(&session), "go", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(answer, "all done");
        assert!(
            overlapped.load(Ordering::SeqCst),
            "parallel tools should overlap"
        );
        assert_eq!(updates.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn unbounded_default_allows_long_tool_chains_and_respects_explicit_cap() {
        struct ChainAdapter(AtomicUsize);
        impl LlmAdapter for ChainAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n != 12 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{n}"),
                                name: "ping".into(),
                                arguments: json!({"step": n}).to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("marathon_complete")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(ChainAdapter(AtomicUsize::new(0)));
        let _reg = app
            .tools
            .register(ToolDefinition::sync_tool(
                "ping",
                "ping",
                json!({}),
                |_, _| Ok::<_, BoxError>("pong".to_string()),
            ))
            .unwrap();

        // 默认不限步数：12 步工具链应顺利跑完而非被旧的 8 步限制打断
        let s1 = app.sessions.create(None);
        let reply = app
            .agent_loop
            .run(Arc::clone(&s1), "marathon", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(reply, "marathon_complete");

        // 显式指定 RunOptions.max_steps 时应按上限中止并记录 MaxSteps
        let end_reasons = Arc::new(Mutex::new(Vec::<TurnEndReason>::new()));
        let reasons_sub = Arc::clone(&end_reasons);
        let _reasons_guard = app.agent_loop.on_event(move |ev| {
            if let AgentEvent::TurnEnd(ev) = ev {
                reasons_sub.lock().unwrap().push(ev.reason.clone());
            }
        });

        let s2 = app.sessions.create(None);
        let capped = app
            .agent_loop
            .run_with_options(
                Arc::clone(&s2),
                "capped",
                CancellationToken::new(),
                RunOptions {
                    max_steps: Some(2),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(capped.contains("max_steps=2"));
        assert_eq!(
            end_reasons.lock().unwrap().last(),
            Some(&TurnEndReason::MaxSteps)
        );
        // 步数上限提示不得写入表面日志（session.log()）或污染模型上下文
        assert!(s2.log().iter().all(|e| match &e.event {
            SessionEvent::Assistant { blocks, .. } => {
                !blocks
                    .iter()
                    .any(|b| b.as_text().is_some_and(|t| t.contains("max_steps=2")))
            }
            SessionEvent::ContextNotice { text, .. } => !text.contains("max_steps=2"),
            _ => true,
        }));
        assert!(
            s2.derive_messages()
                .iter()
                .all(|m| !format!("{m:?}").contains("max_steps=2"))
        );
        assert!(s2.all_entries().iter().any(|e| matches!(
            &e.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::TurnNotice { text, .. }
            } if text.contains("max_steps=2")
        )));
        assert!(s2.all_entries().iter().any(|e| matches!(
            &e.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::TurnEnd { reason, .. }
            } if reason == "max_steps"
        )));
    }

    #[tokio::test]
    async fn max_tokens_truncation_sets_truncated_turn_end_reason() {
        struct TruncatedAdapter(AtomicUsize);
        impl LlmAdapter for TruncatedAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let idx = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if idx == 0 {
                        let _ =
                            tx.unbounded_send(Ok(StreamChunk::text("这是一段被截断的半截回答")));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                            index: 0,
                            text: "思考耗尽了输出预算".into(),
                        }));
                    }
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::MaxTokens,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(TruncatedAdapter(AtomicUsize::new(0)));
        let step_outcomes = Arc::new(Mutex::new(Vec::<StepOutcome>::new()));
        let end_reasons = Arc::new(Mutex::new(Vec::<TurnEndReason>::new()));

        let so_sub = Arc::clone(&step_outcomes);
        let tr_sub = Arc::clone(&end_reasons);
        let _outcomes_guard = app.agent_loop.on_event(move |ev| match ev {
            AgentEvent::StepEnd(ev) => so_sub.lock().unwrap().push(ev.outcome.clone()),
            AgentEvent::TurnEnd(ev) => tr_sub.lock().unwrap().push(ev.reason.clone()),
            _ => {}
        });

        // 1. 文本在 FinishReason::MaxTokens 下截断：步级为 MaxTokens，轮级必须为 Truncated 而非 Completed
        let s1 = app.sessions.create(None);
        let reply1 = app
            .agent_loop
            .run(Arc::clone(&s1), "请写长文", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(reply1, "这是一段被截断的半截回答");
        assert_eq!(
            step_outcomes.lock().unwrap().as_slice(),
            &[StepOutcome::MaxTokens]
        );
        assert_eq!(
            end_reasons.lock().unwrap().as_slice(),
            &[TurnEndReason::Truncated]
        );
        assert!(s1.all_entries().iter().any(|e| matches!(
            &e.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::TurnEnd { reason, .. }
            } if reason == "truncated"
        )));

        // 2. 仅思考耗尽预算、无正文的 MaxTokens 截断：同样不得记成 Completed
        let s2 = app.sessions.create(None);
        let reply2 = app
            .agent_loop
            .run(Arc::clone(&s2), "深度推理", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(reply2, "");
        assert_eq!(
            end_reasons.lock().unwrap().last(),
            Some(&TurnEndReason::Truncated)
        );
    }

    #[tokio::test]
    async fn run_with_report_distinguishes_max_tokens_and_max_steps() {
        struct ModeAdapter {
            calls: AtomicUsize,
        }
        impl LlmAdapter for ModeAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n == 0 {
                        // 正常完成
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("ok_stop")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else if n == 1 {
                        // 有文本输出但因 max_tokens 截断
                        let _ = tx.unbounded_send(Ok(StreamChunk::text(
                            "partial_answer_longer_than_usual",
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::MaxTokens,
                            replay: None,
                        }));
                    } else {
                        // 持续请求工具以耗尽 max_steps
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: format!("call_{n}"),
                                name: "ping".into(),
                                arguments: json!({}).to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(ModeAdapter {
            calls: AtomicUsize::new(0),
        });
        let _reg = app
            .tools
            .register(ToolDefinition::sync_tool(
                "ping",
                "ping",
                json!({}),
                |_, _| Ok::<_, BoxError>("pong".to_string()),
            ))
            .unwrap();
        let agent = &app.agent_loop;

        let s_ok = app.sessions.create(None);
        let r_ok = agent
            .run_with_report(s_ok, "hi", CancellationToken::new(), RunOptions::default())
            .await
            .unwrap();
        assert_eq!(r_ok.text, "ok_stop");
        assert_eq!(r_ok.reason, TurnEndReason::Completed);
        assert_eq!(r_ok.last_step_outcome, Some(StepOutcome::Completed));
        assert_eq!(r_ok.steps, 1);

        let s_trunc = app.sessions.create(None);
        let r_trunc = agent
            .run_with_report(
                s_trunc,
                "trunc",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r_trunc.reason, TurnEndReason::Truncated);
        assert_eq!(r_trunc.last_step_outcome, Some(StepOutcome::MaxTokens));
        assert_eq!(r_trunc.steps, 1);

        let s_cap = app.sessions.create(None);
        let r_cap = agent
            .run_with_report(
                s_cap,
                "cap",
                CancellationToken::new(),
                RunOptions {
                    max_steps: Some(2),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(r_cap.reason, TurnEndReason::MaxSteps);
        assert_eq!(r_cap.last_step_outcome, Some(StepOutcome::ToolCalls));
        assert_eq!(r_cap.steps, 2);
    }

    #[tokio::test]
    async fn consecutive_short_cutoffs_across_turns_and_later_step_overflow_trigger_recovery() {
        struct MultiRecoveryAdapter {
            calls: AtomicUsize,
        }
        impl LlmAdapter for MultiRecoveryAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    match n {
                        // Turn 1: 短截断 #1 (<200 chars + MaxTokens)
                        0 => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("short 1")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::MaxTokens,
                                replay: None,
                            }));
                        }
                        // Turn 2: 短截断 #2 -> 达到 consecutive_short_cutoffs >= 2 触发 overflow 恢复后重试
                        1 => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("short 2")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::MaxTokens,
                                replay: None,
                            }));
                        }
                        // Turn 2 重试成功 -> 调用工具进入 Step 2
                        2 => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: "c_step1".into(),
                                    name: "ping".into(),
                                    arguments: json!({}).to_string(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                        // Turn 2 Step 2 遭遇 ContextWindowExceeded -> 必须仍能触发第二次 overflow 恢复
                        3 => {
                            let _ = tx.unbounded_send(Err(LlmError::new(
                                LlmErrorCode::ContextWindowExceeded,
                                "context length exceeded at step 2",
                            )));
                        }
                        // Turn 2 Step 2 重试成功
                        _ => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("all recovered")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        }
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(MultiRecoveryAdapter {
            calls: AtomicUsize::new(0),
        });
        let _reg = app
            .tools
            .register(ToolDefinition::sync_tool(
                "ping",
                "ping",
                json!({}),
                |_, _| Ok::<_, BoxError>("pong".to_string()),
            ))
            .unwrap();

        let overflow_reasons = Arc::new(Mutex::new(Vec::<String>::new()));
        let rec = Arc::clone(&overflow_reasons);
        let _err_hook = app.agent_loop.request_error_hook().register(
            move |input: AgentRequestErrorInput,
                  next: Next<
                AgentRequestErrorInput,
                BoxFuture<'static, AgentRequestErrorDecision>,
            >| {
                let rec = Arc::clone(&rec);
                Box::pin(async move {
                    if input.kind == AgentRequestErrorKind::ContextOverflow {
                        rec.lock().unwrap().push(input.error_message);
                        return AgentRequestErrorDecision::Retry;
                    }
                    next.call(input).await
                }) as BoxFuture<'static, AgentRequestErrorDecision>
            },
        );

        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);

        let r1 = agent
            .run_with_report(
                Arc::clone(&sess),
                "turn 1",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r1.reason, TurnEndReason::Truncated);
        assert!(overflow_reasons.lock().unwrap().is_empty());

        let r2 = agent
            .run_with_report(
                Arc::clone(&sess),
                "turn 2",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r2.reason, TurnEndReason::Completed);
        assert_eq!(r2.text, "all recovered");
        let reasons = overflow_reasons.lock().unwrap().clone();
        assert_eq!(reasons.len(), 2, "reasons={reasons:?}");
        assert!(reasons[0].contains("consecutive short max_tokens cutoffs"));
        assert!(reasons[1].contains("context length exceeded at step 2"));
    }

    #[tokio::test]
    async fn reasoning_only_response_without_visible_text_never_completes_silently() {
        struct ReasoningOnlyAdapter(AtomicUsize);
        impl LlmAdapter for ReasoningOnlyAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                        index: 0,
                        text: "internal chain of thought only".into(),
                    }));
                    let reason = if n <= MAX_EMPTY_RESPONSE_RETRIES {
                        FinishReason::Stop
                    } else {
                        FinishReason::MaxTokens
                    };
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(ReasoningOnlyAdapter(AtomicUsize::new(0)));
        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);

        // 1. Stop 但仅有 Reasoning 无任何 Text/ToolCall -> 必须报 EmptyResponse，不得空转 Completed
        let err = agent
            .run_with_report(
                Arc::clone(&sess),
                "q1",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::Llm(e) if e.code == LlmErrorCode::EmptyResponse));

        // 2. MaxTokens 且仅有 Reasoning 无正文 -> 必须返回 TurnEndReason::Truncated
        let rep = agent
            .run_with_report(
                Arc::clone(&sess),
                "q2",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(rep.reason, TurnEndReason::Truncated);
    }

    #[tokio::test]
    async fn inter_step_steer_continues_running_turn_without_cancelling_or_losing_tool_result() {
        struct SteerAdapter {
            step: AtomicUsize,
            recorded: Arc<Mutex<Vec<Vec<Message>>>>,
        }

        impl LlmAdapter for SteerAdapter {
            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.recorded.lock().unwrap().push(options.messages.clone());
                let n = self.step.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n == 0 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "c_gate".into(),
                                name: "gate_tool".into(),
                                arguments: "{}".into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("steered ok")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let recorded = Arc::new(Mutex::new(Vec::new()));
        let app = setup_app_with_adapter(SteerAdapter {
            step: AtomicUsize::new(0),
            recorded: Arc::clone(&recorded),
        });

        let tool_started = Arc::new(tokio::sync::Notify::new());
        let tool_release = Arc::new(tokio::sync::Notify::new());
        let started_sig = Arc::clone(&tool_started);
        let release_sig = Arc::clone(&tool_release);
        let _reg = app
            .tools
            .register(ToolDefinition::async_tool(
                "gate_tool",
                "waits before returning",
                json!({}),
                move |_, _| {
                    let s = Arc::clone(&started_sig);
                    let r = Arc::clone(&release_sig);
                    async move {
                        s.notify_one();
                        r.notified().await;
                        Ok::<_, BoxError>("tool_done".to_string())
                    }
                },
            ))
            .unwrap();

        let agent = Arc::clone(&app.agent_loop);
        let sess = app.sessions.create(None);
        let cancel = CancellationToken::new();

        let agent_bg = Arc::clone(&agent);
        let sess_bg = Arc::clone(&sess);
        let cancel_bg = cancel.clone();
        let run_handle = tokio::spawn(async move {
            agent_bg
                .run_with_report(sess_bg, "start alpha", cancel_bg, RunOptions::default())
                .await
        });

        tool_started.notified().await;
        assert!(agent.is_session_busy(&sess.id));
        assert!(!cancel.is_cancelled());

        let push = agent
            .inbox()
            .push(&sess.id, DeliveryMode::Steer, TurnInput::user("steer beta"));
        assert_eq!(push.effective_mode, DeliveryMode::Steer);
        assert!(!push.degraded);
        assert!(
            !cancel.is_cancelled(),
            "gentle inter-step steer must not cancel running turn token"
        );

        tool_release.notify_one();
        let report = run_handle.await.unwrap().unwrap();
        assert_eq!(report.reason, TurnEndReason::Completed);
        assert_eq!(report.text, "steered ok");
        assert_eq!(report.steps, 2);
        assert!(!cancel.is_cancelled());

        let calls = recorded.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        let step2_texts: Vec<String> = calls[1].iter().map(|m| m.text_content()).collect();
        assert!(
            step2_texts.iter().any(|t| t.contains("tool_done")),
            "step 2 must include completed tool output: {step2_texts:?}"
        );
        assert!(
            step2_texts.iter().any(|t| t.contains("steer beta")),
            "step 2 must include steered user prompt: {step2_texts:?}"
        );
    }

    #[tokio::test]
    async fn turn_stopping_guard_steers_next_step_and_caps_notice_steers_at_auto_wake_limit() {
        struct DraftAdapter(AtomicUsize);
        impl LlmAdapter for DraftAdapter {
            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let has_fix_notice = options
                    .messages
                    .iter()
                    .any(|m| m.text_content().contains("retry with fix"));
                let reply = if has_fix_notice { "fixed-2" } else { "draft-1" };
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text(format!("{reply}#{n}"))));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(DraftAdapter(AtomicUsize::new(0)));
        let agent = Arc::clone(&app.agent_loop);
        let s1 = app.sessions.create(None);
        let s2 = app.sessions.create(None);
        agent.inbox().set_auto_wake_limit(&s1.id, 2);
        agent.inbox().set_auto_wake_limit(&s2.id, 2);

        let s1_id = s1.id.clone();
        let s2_id = s2.id.clone();
        let s1_stops = Arc::new(AtomicUsize::new(0));
        let s2_stops = Arc::new(AtomicUsize::new(0));
        let s1_stops_hook = Arc::clone(&s1_stops);
        let s2_stops_hook = Arc::clone(&s2_stops);
        let agent_hook = Arc::clone(&agent);
        let _stop_guard =
            app.agent_loop
                .turn_stopping_hook()
                .register(move |ev: &AgentTurnStoppingEvent| {
                    let agent = Arc::clone(&agent_hook);
                    let s1_id = s1_id.clone();
                    let s2_id = s2_id.clone();
                    let s1_stops = Arc::clone(&s1_stops_hook);
                    let s2_stops = Arc::clone(&s2_stops_hook);
                    let sid = ev.session.id.clone();
                    let last_text = ev.last_text.clone();
                    Box::pin(async move {
                        if sid == s1_id {
                            s1_stops.fetch_add(1, Ordering::SeqCst);
                            if last_text.starts_with("draft-1") {
                                agent.inbox().push(
                                    &sid,
                                    DeliveryMode::Steer,
                                    TurnInput::notice("verifier", "retry with fix"),
                                );
                            }
                        } else if sid == s2_id {
                            s2_stops.fetch_add(1, Ordering::SeqCst);
                            agent.inbox().push(
                                &sid,
                                DeliveryMode::Steer,
                                TurnInput::notice("loop-guard", "never stop"),
                            );
                        }
                        Some(())
                    })
                });

        // 1. 一次拦停后满足条件放行
        let r1 = agent
            .run_with_report(
                Arc::clone(&s1),
                "write code",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r1.reason, TurnEndReason::Completed);
        assert!(r1.text.starts_with("fixed-2"), "{}", r1.text);
        assert_eq!(r1.steps, 2);
        assert_eq!(s1_stops.load(Ordering::SeqCst), 2);

        // 2. 无限拦停在达到 auto_wake_limit=2 后必须停机，且多余 Notice Steer 降级为 Inject 不再唤醒
        let mut wake_rx = agent.inbox().subscribe_wake();
        let r2 = agent
            .run_with_report(
                Arc::clone(&s2),
                "run infinite guard",
                CancellationToken::new(),
                RunOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r2.reason, TurnEndReason::Completed);
        assert_eq!(r2.steps, 3);
        assert_eq!(s2_stops.load(Ordering::SeqCst), 3);
        assert!(agent.inbox().pop_followup(&s2.id).is_none());
        assert!(wake_rx.try_recv().is_err());
        let snap = agent.inbox().snapshot(&s2.id);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].mode, DeliveryMode::Inject);
    }

    #[tokio::test]
    async fn followup_auto_wake_degrades_past_limit_and_resets_on_human_input() {
        struct EchoAllAdapter(Arc<Mutex<Vec<Vec<Message>>>>);
        impl LlmAdapter for EchoAllAdapter {
            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.0.lock().unwrap().push(options.messages);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("ack")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = setup_app_with_adapter(EchoAllAdapter(Arc::clone(&seen)));
        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);
        agent.inbox().set_auto_wake_limit(&sess.id, 2);
        let mut wake_rx = agent.inbox().subscribe_wake();

        let p1 = agent.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "wake-1"),
        );
        let p2 = agent.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "wake-2"),
        );
        let p3 = agent.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "degraded-3"),
        );
        assert_eq!(p1.effective_mode, DeliveryMode::FollowUp);
        assert!(!p1.degraded);
        assert_eq!(p2.effective_mode, DeliveryMode::FollowUp);
        assert!(!p2.degraded);
        assert_eq!(p3.effective_mode, DeliveryMode::Inject);
        assert!(p3.degraded);
        assert_eq!(agent.inbox().auto_wakes_since_human(&sess.id), 2);
        assert_eq!(wake_rx.try_recv().unwrap(), sess.id);
        assert_eq!(wake_rx.try_recv().unwrap(), sess.id);
        assert!(wake_rx.try_recv().is_err());

        assert!(matches!(
            agent.inbox().pop_followup(&sess.id),
            Some(TurnInput::Notice { text, .. }) if text == "wake-1"
        ));
        assert!(matches!(
            agent.inbox().pop_followup(&sess.id),
            Some(TurnInput::Notice { text, .. }) if text == "wake-2"
        ));
        assert!(agent.inbox().pop_followup(&sess.id).is_none());

        // 人类输入运行后必须重置 auto_wakes_since_human，同时把之前降级的 Inject 一并带入当步
        let ans = agent
            .run(Arc::clone(&sess), "human resume", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(ans, "ack");
        assert_eq!(agent.inbox().auto_wakes_since_human(&sess.id), 0);

        let msgs = seen.lock().unwrap()[0].clone();
        let joined = msgs
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("degraded-3"), "{joined}");
        assert!(joined.contains("human resume"), "{joined}");

        let p4 = agent.inbox().push(
            &sess.id,
            DeliveryMode::FollowUp,
            TurnInput::notice("cron", "wake-after-reset"),
        );
        assert_eq!(p4.effective_mode, DeliveryMode::FollowUp);
        assert!(!p4.degraded);
        assert_eq!(agent.inbox().auto_wakes_since_human(&sess.id), 1);
    }

    #[tokio::test]
    async fn pre_step_reject_skips_commit_and_env_context_reinjects_after_compaction() {
        struct RecordingEcho(Arc<Mutex<Vec<Vec<Message>>>>);
        impl LlmAdapter for RecordingEcho {
            fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
                self.0.lock().unwrap().push(options.messages);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("ok")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                });
                Box::pin(rx)
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = setup_app_with_adapter(RecordingEcho(Arc::clone(&seen)));
        let _pre_step_guard = app.agent_loop.pre_step_hook().register(
            |input: AgentPreStepInput,
             next: Next<AgentPreStepInput, BoxFuture<'static, AgentPreStepDecision>>| {
                Box::pin(async move {
                    if input.inputs.iter().any(|item| match item {
                        TurnInput::User { text, .. } => text.contains("forbidden"),
                        TurnInput::Notice { .. } => false,
                    }) {
                        return AgentPreStepDecision::Reject {
                            reason: "blocked by policy".into(),
                        };
                    }
                    next.call(input).await
                }) as BoxFuture<'static, AgentPreStepDecision>
            },
        );

        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);

        let err = agent
            .run(
                Arc::clone(&sess),
                "forbidden action",
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::PreStepRejected(ref msg) if msg == "blocked by policy"));
        assert!(
            sess.log().is_empty(),
            "rejected pre-step must not commit prompt or environment_context"
        );

        agent
            .run(Arc::clone(&sess), "allowed step", CancellationToken::new())
            .await
            .unwrap();
        assert!(sess.derive_messages().iter().any(|m| {
            matches!(m, Message::User { .. }) && m.text_content().contains("<environment_context>")
        }));

        // 模拟压缩折叠早期条目后，下一轮 pre-step 应再次注入 <environment_context>
        sess.compact_at(
            "## Summary\nprior work",
            sess.last_seq(),
            None,
            None,
            Default::default(),
        );
        agent
            .run(
                Arc::clone(&sess),
                "post compact step",
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let calls = seen.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        assert!(
            calls[1]
                .iter()
                .any(|m| m.text_content().contains("<environment_context>")),
            "post-compaction call must re-inject environment_context"
        );
    }

    #[tokio::test]
    async fn request_error_hook_recovers_stream_error_with_retry() {
        struct FlakyStreamAdapter(AtomicUsize);
        impl LlmAdapter for FlakyStreamAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n == 0 {
                        let _ = tx.unbounded_send(Err(LlmError::new(
                            LlmErrorCode::Server,
                            "transient upstream 503",
                        )));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("recovered stream")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(FlakyStreamAdapter(AtomicUsize::new(0)));
        let seen_kinds = Arc::new(Mutex::new(Vec::new()));
        let rec = Arc::clone(&seen_kinds);
        let retried = Arc::new(AtomicUsize::new(0));
        let retried_hook = Arc::clone(&retried);
        let _req_err_guard = app.agent_loop.request_error_hook().register(
            move |input: AgentRequestErrorInput,
                  next: Next<
                AgentRequestErrorInput,
                BoxFuture<'static, AgentRequestErrorDecision>,
            >| {
                let rec = Arc::clone(&rec);
                let retried_hook = Arc::clone(&retried_hook);
                Box::pin(async move {
                    rec.lock()
                        .unwrap()
                        .push((input.kind, input.error_message.clone()));
                    if input.kind == AgentRequestErrorKind::StreamError
                        && retried_hook.fetch_add(1, Ordering::SeqCst) == 0
                    {
                        return AgentRequestErrorDecision::Retry;
                    }
                    next.call(input).await
                }) as BoxFuture<'static, AgentRequestErrorDecision>
            },
        );

        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);
        let text = agent
            .run(Arc::clone(&sess), "retry me", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text, "recovered stream");
        let recorded = seen_kinds.lock().unwrap().clone();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, AgentRequestErrorKind::StreamError);
        assert!(recorded[0].1.contains("transient upstream 503"));
    }

    #[tokio::test]
    async fn emits_first_token_on_tool_call_delta_without_text() {
        struct ToolFirstTokenAdapter(AtomicUsize);
        impl LlmAdapter for ToolFirstTokenAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let step = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if step == 0 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::BlockStart {
                            index: 0,
                            kind: mizar_llm::ContentBlockKind::ToolCall,
                        }));
                        let _ = tx.unbounded_send(Ok(StreamChunk::ToolCallDelta {
                            index: 0,
                            id: Some("call_ttft".into()),
                            name: Some("echo".into()),
                            arguments_delta: "{}".into(),
                        }));
                        let _ = tx.unbounded_send(Ok(StreamChunk::BlockEnd {
                            index: 0,
                            block: mizar_llm::ContentBlock::ToolCall {
                                id: "call_ttft".into(),
                                name: "echo".into(),
                                arguments: "{}".into(),
                            },
                        }));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("done")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(ToolFirstTokenAdapter(AtomicUsize::new(0)));
        let _reg = app
            .tools
            .register(ToolDefinition::sync_tool(
                "echo",
                "echo",
                json!({ "type": "object", "properties": {} }),
                |_, _| Ok::<_, BoxError>("ok".to_string()),
            ))
            .unwrap();

        let first_tokens = Arc::new(Mutex::new(Vec::<usize>::new()));
        let ft_cap = Arc::clone(&first_tokens);
        let _ft_guard = app.agent_loop.on_event(move |ev| {
            if let AgentEvent::StepFirstToken(ev) = ev {
                ft_cap.lock().unwrap().push(ev.step);
            }
        });

        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);
        let text = agent
            .run(Arc::clone(&sess), "call tool", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text, "done");
        assert_eq!(*first_tokens.lock().unwrap(), vec![1, 2]);
        let trace_ft: Vec<usize> = sess
            .all_entries()
            .into_iter()
            .filter_map(|e| match e.event {
                SessionEvent::Trace {
                    detail: SessionTraceEvent::FirstToken { step, .. },
                } => Some(step),
                _ => None,
            })
            .collect();
        assert_eq!(trace_ft, vec![1, 2]);
    }

    #[tokio::test]
    async fn malformed_tool_arguments_return_validation_error_without_executing_tool() {
        struct MalformedArgsAdapter(AtomicUsize);
        impl LlmAdapter for MalformedArgsAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let step = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    match step {
                        0 => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: "c_bad_syntax".into(),
                                    name: "write_item".into(),
                                    arguments: "{\"path\":".into(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                1,
                                ToolCall {
                                    id: "c_non_obj".into(),
                                    name: "write_item".into(),
                                    arguments: "[1, 2]".into(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                        1 => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: "c_fixed".into(),
                                    name: "write_item".into(),
                                    arguments: "{\"path\":\"ok.txt\"}".into(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                        _ => {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("fixed and completed")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        }
                    }
                });
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(MalformedArgsAdapter(AtomicUsize::new(0)));
        let executed = Arc::new(Mutex::new(Vec::<Value>::new()));
        let exec_cap = Arc::clone(&executed);
        let _reg = app
            .tools
            .register(ToolDefinition::sync_tool(
                "write_item",
                "write item",
                json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
                move |args, _| {
                    exec_cap.lock().unwrap().push(args);
                    Ok::<_, BoxError>("wrote ok.txt".to_string())
                },
            ))
            .unwrap();

        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);
        let text = agent
            .run(Arc::clone(&sess), "do write", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text, "fixed and completed");

        // 畸形参数调用绝不能执行底层工具，只有修正后的第三次调用命中 handler
        let calls = executed.lock().unwrap().clone();
        assert_eq!(calls, vec![json!({ "path": "ok.txt" })]);

        let tool_results: Vec<(String, String, bool)> = sess
            .log()
            .into_iter()
            .filter_map(|e| match e.event {
                SessionEvent::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                    ..
                } => Some((tool_call_id, content, is_error)),
                _ => None,
            })
            .collect();
        assert_eq!(tool_results.len(), 3);
        assert_eq!(tool_results[0].0, "c_bad_syntax");
        assert!(tool_results[0].2);
        assert!(tool_results[0].1.contains("Invalid JSON arguments"));
        assert_eq!(tool_results[1].0, "c_non_obj");
        assert!(tool_results[1].2);
        assert!(tool_results[1].1.contains("expected a JSON object"));
        assert_eq!(tool_results[2].0, "c_fixed");
        assert!(!tool_results[2].2);
    }

    #[tokio::test]
    async fn retries_reasoning_only_empty_response_with_and_without_retry_plugin() {
        struct ReasoningThenAnswerAdapter(AtomicUsize);
        impl LlmAdapter for ReasoningThenAnswerAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n == 0 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                            index: 0,
                            text: "thinking without output".into(),
                        }));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("final answer")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        // 1. 未装配 Retry 拦截时，AgentLoop 内部自动重试并成功
        let app_no_retry = setup_app_with_adapter(ReasoningThenAnswerAdapter(AtomicUsize::new(0)));
        let sess1 = app_no_retry.sessions.create(None);
        let text1 = app_no_retry
            .agent_loop
            .run(Arc::clone(&sess1), "hi", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text1, "final answer");

        // 2. 模拟 LlmRetryPlugin 对仅推理流返回 Err(EmptyResponse) 时，AgentLoop 同样自动重试并成功
        struct StreamEmptyErrThenAnswerAdapter(AtomicUsize);
        impl LlmAdapter for StreamEmptyErrThenAnswerAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let (tx, rx) = mpsc::unbounded();
                tokio::spawn(async move {
                    if n == 0 {
                        let _ = tx.unbounded_send(Ok(StreamChunk::ReasoningDelta {
                            index: 0,
                            text: "streamed reasoning".into(),
                        }));
                        let _ = tx.unbounded_send(Err(LlmError::new(
                            LlmErrorCode::EmptyResponse,
                            "model output contained only reasoning without visible text or tool calls",
                        )));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("recovered answer")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
                Box::pin(rx)
            }
        }

        let app_stream_err =
            setup_app_with_adapter(StreamEmptyErrThenAnswerAdapter(AtomicUsize::new(0)));
        let sess2 = app_stream_err.sessions.create(None);
        let text2 = app_stream_err
            .agent_loop
            .run(Arc::clone(&sess2), "hi", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(text2, "recovered answer");
    }

    #[tokio::test]
    async fn agent_event_bus_on_event_preserves_order_and_unregisters_on_drop() {
        struct NoopAdapter;
        impl LlmAdapter for NoopAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let (_tx, rx) = mpsc::unbounded();
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(NoopAdapter);
        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);

        let log = Arc::new(Mutex::new(Vec::<(&'static str, usize)>::new()));
        let log1 = Arc::clone(&log);
        let g1 = agent.on_event(move |ev| {
            if let AgentEvent::Request(req) = ev {
                log1.lock().unwrap().push(("first", req.step));
            }
        });
        let log2 = Arc::clone(&log);
        let g2 = agent.on_event(move |ev| {
            if let AgentEvent::Request(req) = ev {
                log2.lock().unwrap().push(("second", req.step));
            }
        });

        agent.emit_event(AgentEvent::Request(AgentRequestEvent {
            session: Arc::clone(&sess),
            step: 1,
        }));
        assert_eq!(
            *log.lock().unwrap(),
            vec![("first", 1), ("second", 1)],
            "sync listeners must fire in registration order"
        );

        drop(g1);
        agent.emit_event(AgentEvent::Request(AgentRequestEvent {
            session: Arc::clone(&sess),
            step: 2,
        }));
        assert_eq!(
            *log.lock().unwrap(),
            vec![("first", 1), ("second", 1), ("second", 2)],
            "dropped guard must unregister listener"
        );

        drop(g2);
    }

    #[tokio::test]
    async fn agent_event_bus_broadcast_subscribe_receives_typed_events() {
        struct NoopAdapter;
        impl LlmAdapter for NoopAdapter {
            fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
                let (_tx, rx) = mpsc::unbounded();
                Box::pin(rx)
            }
        }

        let app = setup_app_with_adapter(NoopAdapter);
        let agent = &app.agent_loop;
        let sess = app.sessions.create(None);

        let mut rx = agent.subscribe();
        agent.emit_event(AgentEvent::Status(AgentStatusEvent {
            session: Arc::clone(&sess),
            status: AgentStatus::Busy,
        }));
        agent.emit_event(AgentEvent::TextDelta(AgentTextDeltaEvent {
            session: Arc::clone(&sess),
            text: "hello".into(),
            ts_ms: 42,
        }));

        match rx.recv().await.unwrap() {
            AgentEvent::Status(ev) => {
                assert_eq!(ev.session.id, sess.id);
                assert_eq!(ev.status, AgentStatus::Busy);
            }
            other => panic!("unexpected first event: {other:?}"),
        }
        match rx.recv().await.unwrap() {
            AgentEvent::TextDelta(ev) => {
                assert_eq!(ev.session.id, sess.id);
                assert_eq!(ev.text, "hello");
                assert_eq!(ev.ts_ms, 42);
            }
            other => panic!("unexpected second event: {other:?}"),
        }
    }
}
