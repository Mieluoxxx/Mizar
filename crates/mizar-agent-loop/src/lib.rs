//! Agent 循环（对应 `NamiDSH/packages/agent-loop`）。
//!
//! Agent 循环本身也只是一个插件。它不知道模型是哪家、有哪些工具，只认四个服务名：
//! `llm`、`tools`、`sessions`、`systemPrompt`。

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, CordisError, Inject, Service};
use futures::StreamExt;
use futures::future::join_all;
use mizar_llm::{
    BlockAssembler, FinishReason, GenerateOptions, LlmContextExt, LlmError, LlmErrorCode, Message,
    ReasoningEffort, StreamChunk, TokenUsage, ToolCall,
};
use mizar_session::{Session, SessionEvent, SessionTraceEvent};
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
pub const EVENT_AGENT_TURN_END: &str = "agent/turn-end";
pub const EVENT_AGENT_STEP_START: &str = "agent/step-start";
pub const EVENT_AGENT_STEP_FIRST_TOKEN: &str = "agent/step-first-token";
pub const EVENT_AGENT_STEP_END: &str = "agent/step-end";

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
    MaxSteps,
    Cancelled,
    Error { message: String },
}

impl TurnEndReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
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

pub struct AgentLoop {
    ctx: Context,
    config: AgentLoopConfig,
    active_sessions: Arc<Mutex<HashSet<String>>>,
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
    ) -> Result<String, AgentError> {
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
        session.append(SessionEvent::User { text: input.into() });
        self.ctx.emit(
            EVENT_AGENT_STATUS,
            AgentStatusEvent {
                session: Arc::clone(&session),
                status: AgentStatus::Busy,
            },
        );
        let mut turn_guard = TurnGuard {
            ctx: self.ctx.clone(),
            session: Arc::clone(&session),
            turn,
            reason: TurnEndReason::Cancelled,
        };

        let mut step = 0usize;
        let mut overflow_recovered = false;
        let mut consecutive_short_cutoffs = 0usize;
        loop {
            step += 1;
            if let Some(limit) = max_steps
                && step > limit
            {
                break;
            }
            if cancel.is_cancelled() {
                turn_guard.set_reason(TurnEndReason::Cancelled);
                return Err(AgentError::Llm(LlmError::cancelled()));
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
            self.ctx.emit(
                EVENT_AGENT_REQUEST,
                AgentRequestEvent {
                    session: Arc::clone(&session),
                    step,
                },
            );

            let prompt_text = system_prompt.build();
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
                cancel: cancel.clone(),
            }) {
                Ok(s) => s,
                Err(err) => {
                    if err.code == LlmErrorCode::ContextWindowExceeded
                        && !overflow_recovered
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
                            overflow_recovered = true;
                            consecutive_short_cutoffs = 0;
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
            let mut stream_overflow_err: Option<LlmError> = None;
            while let Some(chunk_res) = stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(err) => {
                        if err.code == LlmErrorCode::ContextWindowExceeded
                            && !overflow_recovered
                            && !cancel.is_cancelled()
                        {
                            stream_overflow_err = Some(err);
                            break;
                        }
                        if err.code == LlmErrorCode::Cancelled {
                            flush_interrupted_assistant(&session, &assembler);
                            step_guard.finish(StepOutcome::Cancelled, assembler.usage());
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
                match &chunk {
                    StreamChunk::TextDelta { text, .. } if !text.is_empty() => {
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
                        self.ctx.emit(
                            EVENT_AGENT_REASONING_DELTA,
                            AgentReasoningDeltaEvent {
                                session: Arc::clone(&session),
                                text: text.clone(),
                                ts_ms: now_millis(),
                            },
                        );
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

            if let Some(err) = stream_overflow_err {
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
                    overflow_recovered = true;
                    consecutive_short_cutoffs = 0;
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
                        && !overflow_recovered
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
                            overflow_recovered = true;
                            consecutive_short_cutoffs = 0;
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
            } else {
                consecutive_short_cutoffs = 0;
            }

            if (is_zero_len_cutoff || consecutive_short_cutoffs >= 2)
                && !overflow_recovered
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
                    overflow_recovered = true;
                    consecutive_short_cutoffs = 0;
                    step_guard.finish(
                        StepOutcome::Error {
                            message: "implicit cutoff overflow recovered; retrying".to_string(),
                        },
                        usage,
                    );
                    drop(step_guard);
                    step = step.saturating_sub(1);
                    continue;
                }
            }

            if calls.is_empty() && blocks.is_empty() {
                let err = LlmError::new(
                    LlmErrorCode::EmptyResponse,
                    "model output must contain either output text or tool calls",
                );
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

            session.append(SessionEvent::Assistant {
                blocks,
                replay,
                usage,
            });

            let step_outcome = if !calls.is_empty() {
                StepOutcome::ToolCalls
            } else if matches!(finish_reason, FinishReason::MaxTokens) {
                StepOutcome::MaxTokens
            } else {
                StepOutcome::Completed
            };
            step_guard.finish(step_outcome, usage);
            drop(step_guard);

            if calls.is_empty() {
                turn_guard.set_reason(TurnEndReason::Completed);
                return Ok(text);
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

                let mut futs = Vec::with_capacity(batch.len());
                for call in batch {
                    let args: Value = if call.arguments.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}))
                    };
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
                    futs.push(tools.execute_with(
                        ToolExecution {
                            name: call.name.clone(),
                            args,
                            session_id: Some(session.id.clone()),
                        },
                        cancel.clone(),
                        Some(on_update),
                    ));
                }

                let results: Vec<ToolResult> = if futs.len() == 1 {
                    vec![futs.into_iter().next().unwrap().await]
                } else {
                    join_all(futs).await
                };

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
                        session.append(SessionEvent::ContextNotice {
                            source: note.source,
                            text: note.text,
                        });
                    }
                }
            }
        }

        turn_guard.set_reason(TurnEndReason::MaxSteps);
        let msg = format!(
            "（已达到 max_steps={} 上限并停止，当前进度已保留，发送「继续」即可接着执行）",
            max_steps.unwrap_or(step.saturating_sub(1))
        );
        session.append(SessionEvent::assistant_text(msg.clone()));
        Ok(msg)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cordis::PluginDef;
    use futures::channel::mpsc;
    use mizar_llm::{BoxChunkStream, LlmAdapter, LlmRuntime};
    use mizar_session::{SessionContextExt, SessionStore};
    use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
    use mizar_tools::{ToolDefinition, ToolRuntime};

    use super::*;

    async fn setup_app_with_adapter<A: LlmAdapter>(adapter: A) -> Context {
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(SessionStore::plugin(), ()).await.unwrap();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
            .await
            .unwrap();
        app.plugin(AgentLoop::plugin(), AgentLoopConfig::default())
            .await
            .unwrap();

        let adapter = Arc::new(adapter);
        app.plugin(
            PluginDef::sync("test-adapter", move |ctx, ()| {
                let llm = ctx.llm()?;
                let a = Arc::clone(&adapter);
                ctx.effect("test-adapter.reg", move || {
                    Ok::<_, LlmError>(Some(llm.register_adapter("test", a)?))
                })?;
                Ok::<_, CordisError>(())
            })
            .inject(["llm"]),
            (),
        )
        .await
        .unwrap();
        app
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
        })
        .await;

        let session = app.sessions().unwrap().create(None);
        let cancel = CancellationToken::new();
        let cancel_trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            cancel_trigger.cancel();
        });

        let err = app
            .agent_loop()
            .unwrap()
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
        assert_eq!(log.len(), 2);
        match &log[1].event {
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

        app.fiber().dispose().await;
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

        let app = setup_app_with_adapter(SlowOrEmptyAdapter(AtomicUsize::new(0))).await;
        let session = app.sessions().unwrap().create(None);
        let agent = app.agent_loop().unwrap();

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
        assert_eq!(session.log().len(), 3); // first user + first assistant + third user

        app.fiber().dispose().await;
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

        let app = setup_app_with_adapter(MultiCallAdapter(AtomicUsize::new(0))).await;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicBool::new(false));
        let updates = Arc::new(Mutex::new(Vec::<String>::new()));

        let updates_sub = Arc::clone(&updates);
        app.on(EVENT_AGENT_TOOL_UPDATE, move |ev: &AgentToolUpdateEvent| {
            updates_sub.lock().unwrap().push(ev.preview.clone());
        })
        .unwrap();

        let tools = app.tools().unwrap();
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
                            Ok::<_, cordis::BoxError>(mizar_tools::ToolOutput::text(format!(
                                "id={}",
                                args["id"]
                            )))
                        }
                    },
                )
                .with_execution_mode(ToolExecutionMode::Parallel),
            )
            .unwrap();

        let session = app.sessions().unwrap().create(None);
        let answer = app
            .agent_loop()
            .unwrap()
            .run(Arc::clone(&session), "go", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(answer, "all done");
        assert!(
            overlapped.load(Ordering::SeqCst),
            "parallel tools should overlap"
        );
        assert_eq!(updates.lock().unwrap().len(), 2);

        app.fiber().dispose().await;
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

        let app = setup_app_with_adapter(ChainAdapter(AtomicUsize::new(0))).await;
        let _reg = app
            .tools()
            .unwrap()
            .register(ToolDefinition::sync_tool(
                "ping",
                "ping",
                json!({}),
                |_, _| Ok::<_, cordis::BoxError>("pong".to_string()),
            ))
            .unwrap();

        // 默认不限步数：12 步工具链应顺利跑完而非被旧的 8 步限制打断
        let s1 = app.sessions().unwrap().create(None);
        let reply = app
            .agent_loop()
            .unwrap()
            .run(Arc::clone(&s1), "marathon", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(reply, "marathon_complete");

        // 显式指定 RunOptions.max_steps 时应按上限中止并记录 MaxSteps
        let end_reasons = Arc::new(Mutex::new(Vec::<TurnEndReason>::new()));
        let reasons_sub = Arc::clone(&end_reasons);
        app.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
            reasons_sub.lock().unwrap().push(ev.reason.clone());
        })
        .unwrap();

        let s2 = app.sessions().unwrap().create(None);
        let capped = app
            .agent_loop()
            .unwrap()
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

        app.fiber().dispose().await;
    }
}
