//! Agent 循环（对应 `NamiDSH/packages/agent-loop`）。
//!
//! Agent 循环本身也只是一个插件。它不知道模型是哪家、有哪些工具，只认四个服务名：
//! `llm`、`tools`、`sessions`、`systemPrompt`。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cordis::{Context, CordisError, Inject, Service};
use futures::future::join_all;
use mizar_llm::{
    BlockAssembler, ContentBlock, ContentBlockKind, FinishReason, GenerateOptions, ImageAttachment,
    LlmContextExt, LlmError, LlmErrorCode, Message, ReasoningEffort, StreamChunk, TokenUsage,
    ToolCall,
};
use mizar_llm_retry::{RetryConfig, RetryDecision, collect_attempt, sleep_cancellable};
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
pub const EVENT_AGENT_RETRY: &str = "agent/retry";
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
pub struct AgentRetryEvent {
    pub session: Arc<Session>,
    pub turn: u64,
    pub step: usize,
    pub attempt: usize,
    pub max_attempts: usize,
    pub delay_ms: u64,
    pub reason: String,
    pub reset_draft: bool,
    pub ts_ms: u128,
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

#[derive(Debug, Clone)]
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
    pub retry: RetryConfig,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            max_steps: None,
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            retry: RetryConfig::from_env(),
        }
    }
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

#[derive(Default)]
struct AttemptProgress {
    saw_reasoning_delta: bool,
    has_visible_output: bool,
    has_tool_calls: bool,
    first_token_emitted: bool,
}

impl StepGuard {
    fn observe_chunk(&self, chunk: &StreamChunk, progress: &mut AttemptProgress) {
        match chunk {
            StreamChunk::TextDelta { text, .. } if !text.is_empty() => {
                progress.has_visible_output = true
            }
            StreamChunk::ReasoningDelta { text, .. } if !text.is_empty() => {
                progress.saw_reasoning_delta = true
            }
            StreamChunk::BlockEnd {
                block: ContentBlock::Text { text },
                ..
            } if !text.is_empty() => progress.has_visible_output = true,
            StreamChunk::BlockEnd {
                block: ContentBlock::Reasoning { text },
                ..
            } if !text.is_empty() => progress.saw_reasoning_delta = true,
            StreamChunk::BlockStart {
                kind: ContentBlockKind::ToolCall,
                ..
            }
            | StreamChunk::ToolCallDelta { .. }
            | StreamChunk::BlockEnd {
                block: ContentBlock::ToolCall { .. },
                ..
            } => progress.has_tool_calls = true,
            _ => return,
        }
        let ts_ms = now_millis();
        if !progress.first_token_emitted {
            progress.first_token_emitted = true;
            self.session.append_trace(SessionTraceEvent::FirstToken {
                turn: self.turn,
                step: self.step,
            });
            self.ctx.emit(
                EVENT_AGENT_STEP_FIRST_TOKEN,
                AgentStepFirstTokenEvent {
                    session: Arc::clone(&self.session),
                    turn: self.turn,
                    step: self.step,
                    ts_ms,
                },
            );
        }
        match chunk {
            StreamChunk::TextDelta { text, .. } => self.ctx.emit(
                EVENT_AGENT_TEXT_DELTA,
                AgentTextDeltaEvent {
                    session: Arc::clone(&self.session),
                    text: text.clone(),
                    ts_ms,
                },
            ),
            StreamChunk::ReasoningDelta { text, .. } => self.ctx.emit(
                EVENT_AGENT_REASONING_DELTA,
                AgentReasoningDeltaEvent {
                    session: Arc::clone(&self.session),
                    text: text.clone(),
                    ts_ms,
                },
            ),
            _ => {}
        }
    }

    fn retry(
        &self,
        attempt: usize,
        max_attempts: usize,
        delay: Duration,
        reason: &str,
        reset_draft: bool,
    ) {
        let delay_ms = delay.as_millis() as u64;
        self.ctx.logger_named("agent-loop").info(format_args!(
            "step={} retry={attempt}/{max_attempts} wait={delay_ms}ms reason={reason}",
            self.step
        ));
        self.session.append_trace(SessionTraceEvent::StepRetry {
            turn: self.turn,
            step: self.step,
            attempt,
            max_attempts,
            delay_ms,
            reason: reason.to_owned(),
            reset_draft,
        });
        self.ctx.emit(
            EVENT_AGENT_RETRY,
            AgentRetryEvent {
                session: Arc::clone(&self.session),
                turn: self.turn,
                step: self.step,
                attempt,
                max_attempts,
                delay_ms,
                reason: reason.to_owned(),
                reset_draft,
                ts_ms: now_millis(),
            },
        );
    }

    fn finish(&mut self, outcome: StepOutcome, usage: Option<TokenUsage>) {
        self.outcome = outcome;
        self.usage = usage;
    }
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        // 用量是本次请求的累计快照，只有最终尝试才能计入本步。
        if let Some(usage) = self.usage {
            self.ctx.emit(
                EVENT_AGENT_USAGE,
                AgentUsageEvent {
                    session: Arc::clone(&self.session),
                    usage,
                },
            );
        }
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
        let mut overflow_recoveries_this_turn = 0usize;
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

            let build_messages = || {
                let mut messages = vec![Message::System {
                    content: prompt_text.clone(),
                }];
                messages.extend(session.derive_messages());
                messages
            };
            let mut request = GenerateOptions {
                messages: build_messages(),
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
            };
            let retry = &self.config.retry;
            let mut step_retries = 0;
            let mut mid_thinking_retries = 0;
            let mut recovered_overflow = false;
            let (assembler, text, calls) = 'attempt: loop {
                let mut assembler = BlockAssembler::new();
                let mut progress = AttemptProgress::default();
                let mut error = collect_attempt(&llm, request.clone(), &mut assembler, |chunk| {
                    step_guard.observe_chunk(chunk, &mut progress);
                })
                .await
                .err();
                let text = assembler.text();
                let calls = assembler.tool_calls();
                let is_cutoff = error.is_none() && assembler.finish() == FinishReason::MaxTokens;
                let short_cutoff =
                    is_cutoff && calls.is_empty() && text.chars().take(200).count() < 200;
                let implicit_overflow = short_cutoff
                    && (assembler.blocks().is_empty() || consecutive_short_cutoffs > 0);
                if error.is_none() && !is_cutoff && calls.is_empty() && text.trim().is_empty() {
                    error = Some(
                        LlmError::transport("model returned empty response")
                            .with_http_hints(assembler.response_hints()),
                    );
                }
                if let Some(err) = &mut error
                    && err.code == LlmErrorCode::EmptyResponse
                {
                    err.code = LlmErrorCode::Transport;
                }
                let can_replay = !cancel.is_cancelled()
                    && !progress.has_visible_output
                    && !progress.has_tool_calls
                    && step_retries < retry.max_retries
                    && (!progress.saw_reasoning_delta
                        || mid_thinking_retries < retry.max_mid_thinking_retries);
                let overflow_hints = match &error {
                    Some(err) if err.code == LlmErrorCode::ContextWindowExceeded => {
                        Some((err.should_retry, err.retry_after_ms))
                    }
                    None if implicit_overflow => {
                        let hints = assembler.response_hints();
                        Some((hints.should_retry, hints.retry_after_ms))
                    }
                    _ => None,
                };
                let overflow_delay = overflow_hints
                    .filter(|(allowed, _)| *allowed != Some(false))
                    .map(|(_, ms)| Duration::from_millis(ms.unwrap_or(0)))
                    .filter(|delay| *delay <= retry.max_retry_after);
                let next = if can_replay
                    && !recovered_overflow
                    && overflow_recoveries_this_turn < 3
                    && let Some(delay) = overflow_delay
                    // 压缩可能立即调用同一网关，等待应从收到溢出响应开始。
                    && (delay.is_zero() || sleep_cancellable(delay, &cancel).await.is_ok())
                    && self
                        .ctx
                        .serial::<AgentContextOverflowEvent, bool>(
                            EVENT_AGENT_CONTEXT_OVERFLOW,
                            AgentContextOverflowEvent {
                                session: Arc::clone(&session),
                                turn,
                                step,
                                provider: provider.clone(),
                                model: model.clone(),
                                reason: error.as_ref().map_or_else(
                                    || {
                                        "suspected context overflow after max_tokens cutoff"
                                            .to_string()
                                    },
                                    ToString::to_string,
                                ),
                                cancel: cancel.clone(),
                            },
                        )
                        .await
                        .unwrap_or(false)
                {
                    recovered_overflow = true;
                    overflow_recoveries_this_turn += 1;
                    consecutive_short_cutoffs = 0;
                    self.short_cutoffs_by_session
                        .lock()
                        .unwrap()
                        .remove(&session.id);
                    // 压缩会改变可见历史并重置工具激活态，两者必须取自同一轮压缩之后。
                    request.messages = build_messages();
                    request.tools = tools.list_declared(Some(&session.id));
                    Some((Duration::ZERO, "context_window_exceeded"))
                } else if can_replay
                    && let Some(err) = &error
                    && let RetryDecision::Wait(delay) = retry.decide(err, step_retries)
                {
                    Some((delay, err.code.as_str()))
                } else {
                    None
                };
                if let Some((delay, reason)) = next {
                    step_retries += 1;
                    if progress.saw_reasoning_delta {
                        mid_thinking_retries += 1;
                    }
                    step_guard.retry(
                        step_retries,
                        retry.max_retries,
                        delay,
                        reason,
                        progress.saw_reasoning_delta,
                    );
                    sleep_cancellable(delay, &cancel).await?;
                    continue 'attempt;
                }
                if cancel.is_cancelled() {
                    error = Some(LlmError::cancelled());
                }
                if let Some(err) = error {
                    if err.code == LlmErrorCode::Cancelled {
                        if cancel.is_cancelled() {
                            flush_interrupted_assistant(&session, &assembler);
                        }
                        step_guard.finish(StepOutcome::Cancelled, assembler.usage());
                        turn_guard.set_reason(TurnEndReason::Cancelled);
                    } else {
                        let message = err.to_string();
                        step_guard.finish(
                            StepOutcome::Error {
                                message: message.clone(),
                            },
                            assembler.usage(),
                        );
                        turn_guard.set_reason(TurnEndReason::Error { message });
                    }
                    return Err(err.into());
                }
                if short_cutoff {
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
                break 'attempt (assembler, text, calls);
            };
            let usage = assembler.usage();
            let finish_reason = assembler.finish();
            session.append(SessionEvent::Assistant {
                blocks: assembler.blocks(),
                replay: assembler.replay(),
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
                let reason = if is_truncated {
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
