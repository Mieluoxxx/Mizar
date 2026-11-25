//! Mizar Agent 核心层：主循环、系统提示词、上下文压缩、指令、技能、轨迹、标题、子代理与装配根。
//!
//! 合并自 `mizar-system-prompt`、`mizar-agent-loop`、`mizar-compaction`、`mizar-instructions`、
//! `mizar-skills`、`mizar-trajectory`、`mizar-session-title`、`mizar-subagent` 与 `mizar-agent-spine`。

pub mod agent_loop;
pub mod compaction;
pub mod instructions;
pub mod session_title;
pub mod skills;
pub mod spine;
pub mod subagent;
pub mod system_prompt;
pub mod trajectory;

pub use agent_loop::{
    AgentError, AgentEvent, AgentEventGuard, AgentLoop, AgentLoopConfig, AgentPreStepDecision,
    AgentPreStepInput, AgentReasoningDeltaEvent, AgentRequestEvent, AgentStatus, AgentStatusEvent,
    AgentStepEndEvent, AgentStepFirstTokenEvent, AgentStepStartEvent, AgentTextDeltaEvent,
    AgentToolCallEvent, AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent,
    AgentTurnInputEvent, AgentTurnStartEvent, AgentUsageEvent, DEFAULT_AUTO_WAKE_LIMIT,
    DeliveryMode, RunOptions, SessionInbox, StepOutcome, TurnEndReason, TurnInput, TurnRunReport,
};
pub use skills::{
    SkillLookupOptions, SkillProvider, SkillRegistry, SkillRegistryConfig, SkillsConfig,
    install_skills,
};
pub use spine::*;
pub use system_prompt::{
    DEFAULT_PERSONA, DEFAULT_RUNTIME_PROTOCOL, PromptSection, SystemPrompt, SystemPromptConfig,
};
pub use trajectory::{
    GuardRepeatEvent, RepeatGuardConfig, RepeatGuardValidationError, TrajectoryAnomalyEvent,
    TrajectoryConfig, TrajectoryEventEmitted, TrajectoryExport, TrajectoryRuntime,
    install_repeat_guard, prepare_trace_jsonl_file, validate_repeat_guard_config,
};
