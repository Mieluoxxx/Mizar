//! 轨迹领域模型：原始边界事件、纯折叠账本记录、全会话统计与检测告警结构。

use mizar_llm_types::{TokenUsage, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum TraceEventKind {
    TurnStart {
        turn: u64,
    },
    TurnEnd {
        turn: u64,
        reason: String,
    },
    StepStart {
        turn: u64,
        step: usize,
        #[serde(default)]
        system_prompt_chars: usize,
        #[serde(default)]
        tools_count: usize,
    },
    StepRetry {
        turn: u64,
        step: usize,
        attempt: usize,
        max_attempts: usize,
        delay_ms: u64,
        reason: String,
        reset_draft: bool,
    },
    FirstToken {
        turn: u64,
        step: usize,
    },
    StepEnd {
        turn: u64,
        step: usize,
        outcome: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    UserMessage {
        text: String,
    },
    ContextNotice {
        source: String,
        text: String,
    },
    AssistantMessage {
        text: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        reasoning: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    ToolCall {
        turn: u64,
        step: usize,
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolResult {
        tool_call_id: String,
        name: String,
        content: String,
        #[serde(default)]
        is_error: bool,
        #[serde(default)]
        blocked: bool,
    },
    Anomaly {
        rule_id: String,
        severity: Severity,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<usize>,
        summary: String,
        #[serde(default)]
        evidence: Value,
    },
}

/// 一条带单调序号与毫秒时间戳的权威轨迹事实事件。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TraceEvent {
    pub seq: u64,
    pub ts_ms: u128,
    pub session_id: String,
    pub kind: TraceEventKind,
}

/// 全日志统计口径（参考 `deepseek-harness/packages/session/session-stats`）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    /// 至少包含一个已闭合步的轮次数。
    pub turns: u64,
    /// 已闭合步数（`step/end` 总数）。
    pub steps: u64,
    /// 失败步骤不会生成助手消息，其模型耗时仍计入总量。
    pub llm_ms: u64,
    /// 成功配对 `tool/call -> tool/result` 的工具墙钟总耗时（毫秒）。
    pub tool_ms: u64,
    /// 首 token 延迟总和（毫秒）。
    pub ttft_ms: u64,
    /// 携带首 token 记录的步数。
    pub ttft_steps: u64,
    /// 上报用量的步的解码总耗时（毫秒）。
    pub decode_ms: u64,
    /// 上报用量的步的输出 token 总数。
    pub decode_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub tool_calls: u64,
    pub tool_errors: u64,
    pub blocked_calls: u64,
    pub interrupted_steps: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RecordKind {
    User,
    Context,
    Assistant,
    Tool,
    Anomaly,
}

/// 轨迹账本中的单条折叠记录（对应 `TrajectoryCellProps`）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryRecord {
    pub index: usize,
    pub seq: u64,
    pub turn: Option<u64>,
    pub step: Option<usize>,
    pub kind: RecordKind,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub started_at_ms: Option<u128>,
    /// 未落定记录保持 `None`，不伪造耗时。
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub blocked: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryStep {
    pub step: usize,
    pub started_at_ms: Option<u128>,
    pub ended_at_ms: Option<u128>,
    pub duration_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub outcome: Option<String>,
    pub system_prompt_chars: usize,
    pub tools_count: usize,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryTurn {
    pub turn: u64,
    pub started_at_ms: Option<u128>,
    pub ended_at_ms: Option<u128>,
    pub duration_ms: Option<u64>,
    pub reason: Option<String>,
    pub steps: Vec<TrajectoryStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectorySnapshot {
    pub session_id: String,
    pub stats: SessionStats,
    pub turns: Vec<TrajectoryTurn>,
    pub records: Vec<TrajectoryRecord>,
}

/// 一条轨迹异常检测告警。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryFinding {
    pub rule_id: String,
    pub severity: Severity,
    pub title: String,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    #[serde(default)]
    pub evidence: Value,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TimelineMode {
    #[default]
    Sequence,
    Actual,
    Duration,
}

impl TimelineMode {
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.unwrap_or("sequence").to_ascii_lowercase().as_str() {
            "actual" | "time" => Self::Actual,
            "duration" => Self::Duration,
            _ => Self::Sequence,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TimelineSpan {
    pub index: usize,
    pub lane: u8,
    pub kind: RecordKind,
    pub label: String,
    pub start: u64,
    pub end: u64,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_offset: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TimelineTurnBoundary {
    pub turn: u64,
    pub pos: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TimelineModel {
    pub mode: TimelineMode,
    pub start: u64,
    pub end: u64,
    pub spans: Vec<TimelineSpan>,
    pub turn_boundaries: Vec<TimelineTurnBoundary>,
}

/// 返回给前端或 CLI 的完整轨迹检测报告。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectoryReport {
    pub snapshot: TrajectorySnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline: Option<TimelineModel>,
    pub findings: Vec<TrajectoryFinding>,
}

/// 会话列表摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TrajectorySummary {
    pub session_id: String,
    pub event_count: usize,
    pub turns: u64,
    pub steps: u64,
    pub tool_calls: u64,
    pub findings_count: usize,
    pub has_errors: bool,
}
