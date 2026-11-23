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
    AgentError, AgentLoop, AgentLoopConfig, AgentLoopContextExt, AgentPreStepDecision,
    AgentPreStepInput, AgentStepStartEvent, AgentTextDeltaEvent, AgentToolCallEvent,
    AgentToolResultEvent, AgentToolUpdateEvent, AgentTurnEndEvent, AgentTurnInputEvent,
    DEFAULT_AUTO_WAKE_LIMIT, DeliveryMode, EVENT_AGENT_PRE_STEP, EVENT_AGENT_REASONING_DELTA,
    EVENT_AGENT_REQUEST, EVENT_AGENT_REQUEST_ERROR, EVENT_AGENT_STATUS, EVENT_AGENT_STEP_START,
    EVENT_AGENT_TEXT_DELTA, EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TOOL_RESULT,
    EVENT_AGENT_TOOL_UPDATE, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_INPUT, EVENT_AGENT_TURN_START,
    EVENT_AGENT_TURN_STOPPING, RunOptions, SessionInbox, StepOutcome, TurnEndReason, TurnInput,
    TurnRunReport,
};
pub use skills::{
    EVENT_SKILLS_CHANGE, SkillLookupOptions, SkillProvider, SkillRegistry, SkillRegistryConfig,
    SkillsConfig, SkillsContextExt, SkillsPlugin,
};
pub use spine::*;
pub use system_prompt::{
    DEFAULT_PERSONA, DEFAULT_RUNTIME_PROTOCOL, PromptSection, SystemPrompt, SystemPromptConfig,
    SystemPromptContextExt,
};
pub use trajectory::{
    EVENT_GUARD_REPEAT, EVENT_TRAJECTORY_ANOMALY, EVENT_TRAJECTORY_EVENT, GuardRepeatEvent,
    RepeatGuardConfig, RepeatToolGuardPlugin, TrajectoryAnomalyEvent, TrajectoryConfig,
    TrajectoryContextExt, TrajectoryExport, TrajectoryRuntime,
};
