//! Agent 循环（对应 `NamiDSH/packages/agent-loop`）。
//!
//! Agent 循环本身也只是一个插件。它不知道模型是哪家、有哪些工具，只认四个服务名：
//! `llm`、`tools`、`sessions`、`systemPrompt`。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, CordisError, Inject, Service};
use futures::StreamExt;
use futures::future::join_all;
use mizar_llm::{
    BlockAssembler, FinishReason, GenerateOptions, ImageAttachment, LlmContextExt, LlmError,
    LlmErrorCode, Message, ReasoningEffort, StreamChunk, TokenUsage, ToolCall,
};
use mizar_session::{Session, SessionEvent, SessionTraceEvent, TraceSeverity};
use mizar_system_prompt::SystemPromptContextExt;
use mizar_tools::{
    ToolExecution, ToolExecutionMode, ToolResult, ToolRuntime, ToolUpdate, ToolUpdateCallback,
    ToolsContextExt,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

pub const EVENT_AGENT_STATUS: &str = "agent/status";
pub const EVENT_AGENT_PREPARE_TURN: &str = "agent/prepare-turn";
pub const EVENT_AGENT_PRE_STEP: &str = "agent/pre-step";
pub const EVENT_AGENT_CONTEXT_OVERFLOW: &str = "agent/context-overflow";
pub const EVENT_AGENT_REQUEST: &str = "agent/request";
pub const EVENT_AGENT_TEXT_DELTA: &str = "agent/text-delta";
pub const EVENT_AGENT_REASONING_DELTA: &str = "agent/reasoning-delta";
pub const EVENT_AGENT_TOOL_CALL: &str = "agent/tool-call";
pub const EVENT_AGENT_TOOL_UPDATE: &str = "agent/tool-update";
pub const EVENT_AGENT_TOOL_RESULT: &str = "agent/tool-result";
pub const EVENT_AGENT_USAGE: &str = "agent/usage";
pub const EVENT_AGENT_TURN_START: &str = "agent/turn-start";
pub const EVENT_AGENT_TURN_INPUT: &str = "agent/turn-input";
pub const EVENT_AGENT_TURN_END: &str = "agent/turn-end";
pub const EVENT_AGENT_STEP_START: &str = "agent/step-start";
pub const EVENT_AGENT_STEP_FIRST_TOKEN: &str = "agent/step-first-token";
pub const EVENT_AGENT_STEP_END: &str = "agent/step-end";
pub const ENVIRONMENT_CONTEXT_SOURCE: &str = "environment-context";

fn ensure_environment_context(session: &Arc<Session>) {
    let already_visible = session.with_visible_entries(|entries| {
        entries.iter().any(|e| {
            matches!(
                &e.event,
                SessionEvent::ContextNotice { source, .. } if source == ENVIRONMENT_CONTEXT_SOURCE
            )
        })
    });
    if already_visible {
        return;
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
    session.append(SessionEvent::context_notice(
        ENVIRONMENT_CONTEXT_SOURCE,
        lines.join("\n"),
    ));
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

#[derive(Clone)]
pub struct AgentStatusEvent {
    pub session: Arc<Session>,
    pub status: AgentStatus,
}

/// 在首轮用户消息落盘前派发（紧随 `<environment_context>` 之后），允许技能等模块按 Turn-0 严格顺序前置注入上下文。
#[derive(Clone)]
pub struct AgentPrepareTurnEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub input: String,
    pub cancel: CancellationToken,
}

#[derive(Clone)]
pub struct AgentPreStepEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub cancel: CancellationToken,
}

/// 当模型调用遭遇上下文窗口溢出或隐性连续截断时发出（`serial` 派发，返回 `Some(true)` 表示已释放空间并允许当前步重试）。
#[derive(Clone)]
pub struct AgentContextOverflowEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reason: String,
    pub cancel: CancellationToken,
}

#[derive(Clone)]
pub struct AgentRequestEvent {
    pub session: Arc<Session>,
    pub step: usize,
}

#[derive(Clone)]
pub struct AgentTextDeltaEvent {
    pub session: Arc<Session>,
    pub text: String,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentReasoningDeltaEvent {
    pub session: Arc<Session>,
    pub text: String,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentToolCallEvent {
    pub session: Arc<Session>,
    pub call: ToolCall,
}

#[derive(Clone)]
pub struct AgentToolUpdateEvent {
    pub session: Arc<Session>,
    pub tool_call_id: String,
    pub name: String,
    pub preview: String,
    pub details: Option<Value>,
}

#[derive(Clone)]
pub struct AgentToolResultEvent {
    pub session: Arc<Session>,
    pub tool_call_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
    pub blocked: bool,
    pub details: Option<Value>,
}

#[derive(Clone)]
pub struct AgentUsageEvent {
    pub session: Arc<Session>,
    pub usage: TokenUsage,
}

#[derive(Clone)]
pub struct AgentTurnStartEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentTurnInputEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentTurnEndEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub reason: TurnEndReason,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentStepStartEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub system_prompt_chars: usize,
    pub tools_count: usize,
    pub ts_ms: u128,
}

#[derive(Clone)]
pub struct AgentStepFirstTokenEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub ts_ms: u128,
}

#[derive(Clone)]
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
    Cordis(#[from] CordisError),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("session \"{0}\" is already running")]
    SessionBusy(String),
}

struct SessionLockGuard {
    active_sessions: Arc<Mutex<HashSet<String>>>,
    session_id: String,
}

impl Drop for SessionLockGuard {
    fn drop(&mut self) {
        self.active_sessions
            .lock()
            .unwrap()
            .remove(&self.session_id);
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

struct TurnGuard {
    ctx: Context,
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
        self.session.append_trace(SessionTraceEvent::TurnEnd {
            turn: self.turn,
            reason: self.reason.as_str().to_string(),
        });
        self.ctx.emit(
            EVENT_AGENT_TURN_END,
            AgentTurnEndEvent {
                session: Arc::clone(&self.session),
                turn: self.turn,
                reason: self.reason.clone(),
                ts_ms: now_millis(),
            },
        );
        self.ctx.emit(
            EVENT_AGENT_STATUS,
            AgentStatusEvent {
                session: Arc::clone(&self.session),
                status: AgentStatus::Idle,
            },
        );
    }
}

struct StepGuard {
    ctx: Context,
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
        self.session.append_trace(SessionTraceEvent::StepEnd {
            turn: self.turn,
            step: self.step,
            outcome: self.outcome.as_str().to_string(),
            usage: self.usage,
        });
        self.ctx.emit(
            EVENT_AGENT_STEP_END,
            AgentStepEndEvent {
                session: Arc::clone(&self.session),
                turn: self.turn,
                step: self.step,
                provider: self.provider.clone(),
                model: self.model.clone(),
                outcome: self.outcome.clone(),
                usage: self.usage,
                ts_ms: now_millis(),
            },
        );
    }
}

const MAX_EMPTY_RESPONSE_RETRIES: usize = 2;

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
    ctx: Context,
    config: AgentLoopConfig,
    active_sessions: Arc<Mutex<HashSet<String>>>,
    short_cutoffs_by_session: Arc<Mutex<HashMap<String, usize>>>,
}

impl AgentLoop {
    fn lock_session(&self, session_id: &str) -> Result<SessionLockGuard, AgentError> {
        let mut active = self.active_sessions.lock().unwrap();
        if !active.insert(session_id.to_string()) {
            return Err(AgentError::SessionBusy(session_id.to_string()));
        }
        Ok(SessionLockGuard {
            active_sessions: Arc::clone(&self.active_sessions),
            session_id: session_id.to_string(),
        })
    }

    /// 跑一次：用户输入 → [调模型 → 有工具调用就执行 → 结果写回会话 → 再调模型] → 最终回答。
    /// 过程中发出 `agent/*` 事件，网页端、TUI 和轨迹检测插件靠它们拿到实时流和生命周期边界。
    pub async fn run(
        &self,
        session: Arc<Session>,
        input: impl Into<String>,
        cancel: CancellationToken,
    ) -> Result<String, AgentError> {
        self.run_with_options(session, input, cancel, RunOptions::default())
            .await
    }

    /// 与 [`Self::run`] 相同，但允许在单次调用时覆盖 `provider`、`model`、`reasoning_effort` 等参数。
    pub async fn run_with_options(
        &self,
        session: Arc<Session>,
        input: impl Into<String>,
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
        input: impl Into<String>,
        cancel: CancellationToken,
        options: RunOptions,
    ) -> Result<TurnRunReport, AgentError> {
        let _session_lock = self.lock_session(&session.id)?;
        let res = self
            .run_turn_locked(Arc::clone(&session), input, cancel, options)
            .await;
        let _ = session.flush_async().await;
        res
    }

    async fn run_turn_locked(
        &self,
        session: Arc<Session>,
        input: impl Into<String>,
        cancel: CancellationToken,
        options: RunOptions,
    ) -> Result<TurnRunReport, AgentError> {
        let llm = self.ctx.llm()?;
        let tools = self.ctx.tools()?;
        let system_prompt = self.ctx.system_prompt()?;

        let provider = options.provider.or_else(|| self.config.provider.clone());
        let model = options.model.or_else(|| self.config.model.clone());
        let max_steps = options
            .max_steps
            .or(self.config.max_steps)
            .map(|n| n.max(1));
        let max_tokens = options.max_tokens.or(self.config.max_tokens);
        let temperature = options.temperature.or(self.config.temperature);
        let reasoning_effort = options.reasoning_effort.or(self.config.reasoning_effort);

        let turn = session.next_turn();
        session.append_trace(SessionTraceEvent::TurnStart { turn });
        self.ctx.emit(
            EVENT_AGENT_TURN_START,
            AgentTurnStartEvent {
                session: Arc::clone(&session),
                turn,
                ts_ms: now_millis(),
            },
        );
        let mut turn_guard = TurnGuard {
            ctx: self.ctx.clone(),
            session: Arc::clone(&session),
            turn,
            reason: TurnEndReason::Cancelled,
        };
        let input_text = input.into();
        ensure_environment_context(&session);
        let _ = self
            .ctx
            .serial::<AgentPrepareTurnEvent, ()>(
                EVENT_AGENT_PREPARE_TURN,
                AgentPrepareTurnEvent {
                    session: Arc::clone(&session),
                    turn,
                    input: input_text.clone(),
                    cancel: cancel.clone(),
                },
            )
            .await;
        if cancel.is_cancelled() {
            return Err(AgentError::Llm(LlmError::cancelled()));
        }
        session.append(SessionEvent::user_with_images(input_text, options.images));
        self.ctx.emit(
            EVENT_AGENT_TURN_INPUT,
            AgentTurnInputEvent {
                session: Arc::clone(&session),
                turn,
                provider: provider.clone(),
                model: model.clone(),
                ts_ms: now_millis(),
            },
        );
        self.ctx.emit(
            EVENT_AGENT_STATUS,
            AgentStatusEvent {
                session: Arc::clone(&session),
                status: AgentStatus::Busy,
            },
        );

        let mut step = 0usize;
        let mut steps_executed = 0usize;
        let mut last_step_outcome: Option<StepOutcome> = None;
        let mut last_overflow_step: Option<usize> = None;
        let mut overflow_recoveries_this_turn = 0usize;
        let mut empty_response_retries = 0usize;
        let mut consecutive_short_cutoffs = self
            .short_cutoffs_by_session
            .lock()
            .unwrap()
            .get(&session.id)
            .copied()
            .unwrap_or(0);
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
            let _ = self
                .ctx
                .serial::<AgentPreStepEvent, ()>(
                    EVENT_AGENT_PRE_STEP,
                    AgentPreStepEvent {
                        session: Arc::clone(&session),
                        turn,
                        step,
                        provider: provider.clone(),
                        model: model.clone(),
                        cancel: cancel.clone(),
                    },
                )
                .await;
            if cancel.is_cancelled() {
                turn_guard.set_reason(TurnEndReason::Cancelled);
                return Err(AgentError::Llm(LlmError::cancelled()));
            }
            ensure_environment_context(&session);
            steps_executed += 1;
            self.ctx.emit(
                EVENT_AGENT_REQUEST,
                AgentRequestEvent {
                    session: Arc::clone(&session),
                    step,
                },
            );

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
            self.ctx.emit(
                EVENT_AGENT_STEP_START,
                AgentStepStartEvent {
                    session: Arc::clone(&session),
                    turn,
                    step,
                    system_prompt_chars,
                    tools_count,
                    ts_ms: now_millis(),
                },
            );
            let mut step_guard = StepGuard {
                ctx: self.ctx.clone(),
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
            let can_recover_overflow =
                last_overflow_step != Some(step) && overflow_recoveries_this_turn < 3;

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
            }) {
                Ok(s) => s,
                Err(err) => {
                    if err.code == LlmErrorCode::ContextWindowExceeded
                        && can_recover_overflow
                        && !cancel.is_cancelled()
                    {
                        let recovered = self
                            .ctx
                            .serial::<AgentContextOverflowEvent, bool>(
                                EVENT_AGENT_CONTEXT_OVERFLOW,
                                AgentContextOverflowEvent {
                                    session: Arc::clone(&session),
                                    turn,
                                    step,
                                    provider: provider.clone(),
                                    model: model.clone(),
                                    reason: err.to_string(),
                                    cancel: cancel.clone(),
                                },
                            )
                            .await
                            .unwrap_or(false);
                        if recovered {
                            last_overflow_step = Some(step);
                            overflow_recoveries_this_turn += 1;
                            consecutive_short_cutoffs = 0;
                            self.short_cutoffs_by_session
                                .lock()
                                .unwrap()
                                .remove(&session.id);
                            step_guard.finish(
                                StepOutcome::Error {
                                    message: "context overflow recovered; retrying".to_string(),
                                },
                                None,
                            );
                            drop(step_guard);
                            step = step.saturating_sub(1);
                            continue;
                        }
                    }
                    if err.code == LlmErrorCode::Cancelled {
                        step_guard.finish(StepOutcome::Cancelled, None);
                        turn_guard.set_reason(TurnEndReason::Cancelled);
                    } else {
                        let msg = err.to_string();
                        step_guard.finish(
                            StepOutcome::Error {
                                message: msg.clone(),
                            },
                            None,
                        );
                        turn_guard.set_reason(TurnEndReason::Error { message: msg });
                    }
                    return Err(AgentError::Llm(err));
                }
            };

            let mut assembler = BlockAssembler::new();
            let mut first_token_emitted = false;
            let mut stream_err: Option<LlmError> = None;
            let mut emit_first_token = || {
                if !first_token_emitted {
                    first_token_emitted = true;
                    session.append_trace(SessionTraceEvent::FirstToken { turn, step });
                    self.ctx.emit(
                        EVENT_AGENT_STEP_FIRST_TOKEN,
                        AgentStepFirstTokenEvent {
                            session: Arc::clone(&session),
                            turn,
                            step,
                            ts_ms: now_millis(),
                        },
                    );
                }
            };
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
                        let can_retry_overflow = err.code == LlmErrorCode::ContextWindowExceeded
                            && can_recover_overflow
                            && !cancel.is_cancelled();
                        let can_retry_empty = err.code == LlmErrorCode::EmptyResponse
                            && !cancel.is_cancelled()
                            && empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES;
                        if can_retry_overflow || can_retry_empty {
                            stream_err = Some(err);
                            break;
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
                match &chunk {
                    StreamChunk::TextDelta { text, .. } if !text.is_empty() => {
                        emit_first_token();
                        self.ctx.emit(
                            EVENT_AGENT_TEXT_DELTA,
                            AgentTextDeltaEvent {
                                session: Arc::clone(&session),
                                text: text.clone(),
                                ts_ms: now_millis(),
                            },
                        );
                    }
                    StreamChunk::ReasoningDelta { text, .. } if !text.is_empty() => {
                        emit_first_token();
                        self.ctx.emit(
                            EVENT_AGENT_REASONING_DELTA,
                            AgentReasoningDeltaEvent {
                                session: Arc::clone(&session),
                                text: text.clone(),
                                ts_ms: now_millis(),
                            },
                        );
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
                        emit_first_token();
                    }
                    StreamChunk::BlockEnd {
                        block: mizar_llm::ContentBlock::ToolCall { .. },
                        ..
                    } => {
                        emit_first_token();
                    }
                    StreamChunk::Usage { usage } => {
                        self.ctx.emit(
                            EVENT_AGENT_USAGE,
                            AgentUsageEvent {
                                session: Arc::clone(&session),
                                usage: *usage,
                            },
                        );
                    }
                    _ => {}
                }
                assembler.push(chunk);
            }

            if let Some(err) = stream_err {
                if err.code == LlmErrorCode::EmptyResponse {
                    empty_response_retries += 1;
                    let outcome = StepOutcome::Error {
                        message: "stream error recovered; retrying".to_string(),
                    };
                    last_step_outcome = Some(outcome.clone());
                    step_guard.finish(outcome, None);
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
                let recovered = self
                    .ctx
                    .serial::<AgentContextOverflowEvent, bool>(
                        EVENT_AGENT_CONTEXT_OVERFLOW,
                        AgentContextOverflowEvent {
                            session: Arc::clone(&session),
                            turn,
                            step,
                            provider: provider.clone(),
                            model: model.clone(),
                            reason: err.to_string(),
                            cancel: cancel.clone(),
                        },
                    )
                    .await
                    .unwrap_or(false);
                if recovered {
                    last_overflow_step = Some(step);
                    overflow_recoveries_this_turn += 1;
                    consecutive_short_cutoffs = 0;
                    self.short_cutoffs_by_session
                        .lock()
                        .unwrap()
                        .remove(&session.id);
                    step_guard.finish(
                        StepOutcome::Error {
                            message: "context overflow recovered; retrying".to_string(),
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
                    if err.code == LlmErrorCode::ContextWindowExceeded
                        && can_recover_overflow
                        && !cancel.is_cancelled()
                    {
                        let recovered = self
                            .ctx
                            .serial::<AgentContextOverflowEvent, bool>(
                                EVENT_AGENT_CONTEXT_OVERFLOW,
                                AgentContextOverflowEvent {
                                    session: Arc::clone(&session),
                                    turn,
                                    step,
                                    provider: provider.clone(),
                                    model: model.clone(),
                                    reason: err.to_string(),
                                    cancel: cancel.clone(),
                                },
                            )
                            .await
                            .unwrap_or(false);
                        if recovered {
                            last_overflow_step = Some(step);
                            overflow_recoveries_this_turn += 1;
                            consecutive_short_cutoffs = 0;
                            self.short_cutoffs_by_session
                                .lock()
                                .unwrap()
                                .remove(&session.id);
                            let outcome = StepOutcome::Error {
                                message: "context overflow recovered; retrying".to_string(),
                            };
                            last_step_outcome = Some(outcome.clone());
                            step_guard.finish(outcome, None);
                            drop(step_guard);
                            step = step.saturating_sub(1);
                            continue;
                        }
                    }
                    if !cancel.is_cancelled()
                        && err.code == LlmErrorCode::EmptyResponse
                        && empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES
                    {
                        empty_response_retries += 1;
                        let outcome = StepOutcome::Error {
                            message: "empty response; retrying".to_string(),
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
                consecutive_short_cutoffs += 1;
                self.short_cutoffs_by_session
                    .lock()
                    .unwrap()
                    .insert(session.id.clone(), consecutive_short_cutoffs);
            } else {
                consecutive_short_cutoffs = 0;
                self.short_cutoffs_by_session
                    .lock()
                    .unwrap()
                    .remove(&session.id);
            }

            if (is_zero_len_cutoff || consecutive_short_cutoffs >= 2)
                && can_recover_overflow
                && !cancel.is_cancelled()
            {
                let reason = if is_zero_len_cutoff {
                    "zero-output max_tokens cutoff (suspected context window overflow)".to_string()
                } else {
                    format!(
                        "consecutive short max_tokens cutoffs ({consecutive_short_cutoffs}, suspected implicit small context window)"
                    )
                };
                let recovered = self
                    .ctx
                    .serial::<AgentContextOverflowEvent, bool>(
                        EVENT_AGENT_CONTEXT_OVERFLOW,
                        AgentContextOverflowEvent {
                            session: Arc::clone(&session),
                            turn,
                            step,
                            provider: provider.clone(),
                            model: model.clone(),
                            reason,
                            cancel: cancel.clone(),
                        },
                    )
                    .await
                    .unwrap_or(false);
                if recovered {
                    last_overflow_step = Some(step);
                    overflow_recoveries_this_turn += 1;
                    consecutive_short_cutoffs = 0;
                    self.short_cutoffs_by_session
                        .lock()
                        .unwrap()
                        .remove(&session.id);
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
                if !cancel.is_cancelled() && empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES {
                    empty_response_retries += 1;
                    let outcome = StepOutcome::Error {
                        message: "empty response; retrying".to_string(),
                    };
                    last_step_outcome = Some(outcome.clone());
                    step_guard.finish(outcome, usage);
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
                let detail = if blocks.is_empty() {
                    "model output must contain either output text or tool calls"
                } else {
                    "model output contained only reasoning without visible text or tool calls"
                };
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

            empty_response_retries = 0;
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
            for (batch_idx, batch) in batches.iter().enumerate() {
                if cancel.is_cancelled() {
                    for pending in batches[batch_idx..].iter().flatten() {
                        let content = "已取消执行".to_string();
                        self.ctx.emit(
                            EVENT_AGENT_TOOL_RESULT,
                            AgentToolResultEvent {
                                session: Arc::clone(&session),
                                tool_call_id: pending.id.clone(),
                                name: pending.name.clone(),
                                content: content.clone(),
                                is_error: true,
                                blocked: false,
                                details: None,
                            },
                        );
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
                    self.ctx.emit(
                        EVENT_AGENT_TOOL_CALL,
                        AgentToolCallEvent {
                            session: Arc::clone(&session),
                            call: call.clone(),
                        },
                    );
                    let args = match parse_tool_arguments(&call.name, &call.arguments) {
                        Ok(args) => args,
                        Err(msg) => {
                            // 畸形工具参数不再吞成 `{}`，直接返回校验错误供模型自我修正重试。
                            results[idx] = Some(ToolResult::err(msg));
                            continue;
                        }
                    };
                    let update_ctx = self.ctx.clone();
                    let update_session = Arc::clone(&session);
                    let call_id_for_upd = call.id.clone();
                    let call_name_for_upd = call.name.clone();
                    let on_update: ToolUpdateCallback = Arc::new(move |upd: ToolUpdate| {
                        update_ctx.emit(
                            EVENT_AGENT_TOOL_UPDATE,
                            AgentToolUpdateEvent {
                                session: Arc::clone(&update_session),
                                tool_call_id: call_id_for_upd.clone(),
                                name: call_name_for_upd.clone(),
                                preview: upd.preview,
                                details: upd.details,
                            },
                        );
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
                    let extra_contexts = result.additional_contexts.clone();
                    self.ctx.emit(
                        EVENT_AGENT_TOOL_RESULT,
                        AgentToolResultEvent {
                            session: Arc::clone(&session),
                            tool_call_id: call.id.clone(),
                            name: call.name.clone(),
                            content: result.content.clone(),
                            is_error: result.is_error,
                            blocked: result.blocked,
                            details: result.details,
                        },
                    );
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

impl Service for AgentLoop {
    type Config = AgentLoopConfig;

    fn service_name() -> &'static str {
        "agentLoop"
    }

    fn plugin_name() -> &'static str {
        "AgentLoop"
    }

    fn inject() -> Inject {
        Inject::from(["llm", "tools", "sessions", "systemPrompt"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config,
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            short_cutoffs_by_session: Arc::new(Mutex::new(HashMap::new())),
        }))
    }
}

pub trait AgentLoopContextExt {
    fn agent_loop(&self) -> Result<Arc<AgentLoop>, CordisError>;
}

impl AgentLoopContextExt for Context {
    fn agent_loop(&self) -> Result<Arc<AgentLoop>, CordisError> {
        self.service::<AgentLoop>("agentLoop")
    }
}
