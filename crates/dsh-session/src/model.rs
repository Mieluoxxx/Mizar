//! 会话事件日志的数据模型、头信息与可见性定义。

use dsh_llm::{ContentBlock, ReplayEnvelope, TokenUsage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SESSION_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EntryVisibility {
    /// 对模型可见并参与对话展示（User / ContextNotice / Assistant / ToolResult / Compaction）。
    Surface,
    /// 仅保存在审计与轨迹日志中，不参与模型消息构造，也不出现在 `session.log()` 中。
    LogOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum TraceSeverity {
    Info,
    Warn,
    Error,
}

/// 仅用于轨迹回放与诊断审计的边界事件（不直接发给 LLM）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SessionTraceEvent {
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
    ToolCall {
        turn: u64,
        step: usize,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// 记录工具执行是否被审批或守卫拦截（对应轨迹里 `ToolResult.blocked`）。
    ToolOutcome {
        tool_call_id: String,
        blocked: bool,
    },
    Anomaly {
        rule_id: String,
        severity: TraceSeverity,
        title: String,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        turn: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<usize>,
        #[serde(default)]
        evidence: Value,
    },
}

impl SessionTraceEvent {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::TurnStart { .. } => "turn-start",
            Self::TurnEnd { .. } => "turn-end",
            Self::StepStart { .. } => "step-start",
            Self::FirstToken { .. } => "first-token",
            Self::StepEnd { .. } => "step-end",
            Self::ToolCall { .. } => "tool-call",
            Self::ToolOutcome { .. } => "tool-outcome",
            Self::Anomaly { .. } => "anomaly",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SessionEvent {
    User {
        text: String,
    },
    /// 插件注入的合成上下文消息（例如重复工具提醒），对模型可见但在轨迹视图中区分归属。
    ContextNotice {
        source: String,
        text: String,
    },
    Assistant {
        blocks: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        replay: Option<ReplayEnvelope>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
    },
    ToolResult {
        tool_call_id: String,
        name: String,
        content: String,
        is_error: bool,
    },
    /// 上下文压缩节点：将 `seq < first_kept_seq` 的表面历史折叠为摘要，原始事件仍保留在持久日志中。
    Compaction {
        summary: String,
        first_kept_seq: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_before: Option<u64>,
    },
    /// 内部审计 / 轨迹边界事件（`EntryVisibility::LogOnly`）。
    Trace {
        detail: SessionTraceEvent,
    },
}

impl SessionEvent {
    pub fn assistant_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let blocks = if text.is_empty() {
            Vec::new()
        } else {
            vec![ContentBlock::Text { text }]
        };
        Self::Assistant {
            blocks,
            replay: None,
            usage: None,
        }
    }

    pub fn visibility(&self) -> EntryVisibility {
        match self {
            Self::Trace { .. } => EntryVisibility::LogOnly,
            Self::User { .. }
            | Self::ContextNotice { .. }
            | Self::Assistant { .. }
            | Self::ToolResult { .. }
            | Self::Compaction { .. } => EntryVisibility::Surface,
        }
    }

    pub fn is_surface(&self) -> bool {
        self.visibility() == EntryVisibility::Surface
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::User { .. } => "user",
            Self::ContextNotice { .. } => "context-notice",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "tool-result",
            Self::Compaction { .. } => "compaction",
            Self::Trace { detail } => detail.kind_name(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionEntry {
    pub seq: u64,
    pub event: SessionEvent,
    pub ts: u128,
}

impl SessionEntry {
    pub fn visibility(&self) -> EntryVisibility {
        self.event.visibility()
    }

    pub fn is_surface(&self) -> bool {
        self.event.is_surface()
    }
}

fn default_format_version() -> u32 {
    SESSION_FORMAT_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionHeader {
    pub id: String,
    #[serde(default = "default_format_version")]
    pub format_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork_seq: Option<u64>,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
    #[serde(default)]
    pub event_count: u64,
    #[serde(default)]
    pub surface_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compacted_up_to_seq: Option<u64>,
}

impl SessionHeader {
    pub fn new(id: impl Into<String>, now_ms: u128) -> Self {
        Self {
            id: id.into(),
            format_version: SESSION_FORMAT_VERSION,
            cwd: None,
            title: None,
            parent_id: None,
            fork_seq: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            event_count: 0,
            surface_count: 0,
            compacted_up_to_seq: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SessionCreateOptions {
    pub id: Option<String>,
    pub cwd: Option<String>,
    pub title: Option<String>,
    pub parent_id: Option<String>,
    pub fork_seq: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct SessionHeaderPatch {
    pub title: Option<String>,
    pub updated_at_ms: Option<u128>,
    pub compacted_up_to_seq: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct SessionListQuery {
    pub cwd: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub header: SessionHeader,
    pub entries: Vec<SessionEntry>,
}

/// 从首条用户输入提取简短单行会话标题（超过 72 字符自动截断）。
pub fn derive_session_title(input: &str) -> Option<String> {
    let first_line = input
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    let collapsed = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    const MAX_CHARS: usize = 72;
    let char_count = collapsed.chars().count();
    if char_count <= MAX_CHARS {
        Some(collapsed)
    } else {
        let mut prefix: String = collapsed.chars().take(MAX_CHARS - 1).collect();
        prefix.push('…');
        Some(prefix)
    }
}
