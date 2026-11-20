//! 会话事件日志的数据模型、头信息与可见性定义。

use mizar_llm::{ContentBlock, ImageAttachment, ReasoningEffort, ReplayEnvelope, TokenUsage};
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
    /// 压缩事务开始标记（LogOnly，崩溃/回放时未闭合的 start 自动忽略，避免卡死）。
    CompactionStart {
        trigger: String,
    },
    /// 压缩事务结束标记（LogOnly）。
    CompactionEnd {
        status: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_kept_seq: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_before: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_after: Option<u64>,
    },
    /// 仅用于日志留存与 UI 回放的轮次级提示（`LogOnly`，不进入 `session.log()` 与模型上下文）。
    TurnNotice {
        turn: u64,
        severity: TraceSeverity,
        text: String,
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
            Self::CompactionStart { .. } => "compaction-start",
            Self::CompactionEnd { .. } => "compaction-end",
            Self::TurnNotice { .. } => "turn-notice",
        }
    }
}

/// 若压缩切点落在当前未完成 turn 中途，钉住该 turn 的原始用户输入并附带 `seq` 句柄。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PinnedInput {
    pub seq: u64,
    pub text: String,
    #[serde(default)]
    pub truncated: bool,
}

/// 压缩检查点的确定性硬事实摘要：由代码从历史事件直接提取并跨轮滚动，不依赖 LLM 复述。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CompactionDigest {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modified_files: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands_run: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pinned_inputs: Vec<PinnedInput>,
}

impl CompactionDigest {
    pub fn is_empty(&self) -> bool {
        self.modified_files.is_empty()
            && self.commands_run.is_empty()
            && self.pinned_inputs.is_empty()
    }
}

/// 投影层工具结果剪枝项：不修改原始 `ToolResult`，仅在构造模型消息时替换为紧凑预览与 recall 句柄。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PrunedToolResult {
    pub target_seq: u64,
    pub replacement: String,
    pub original_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum SessionEvent {
    User {
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageAttachment>,
    },
    /// 插件注入的合成上下文消息（例如重复工具提醒），对模型可见但在轨迹视图中区分归属。
    ContextNotice {
        source: String,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ImageAttachment>,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_after: Option<u64>,
        #[serde(default, skip_serializing_if = "CompactionDigest::is_empty")]
        digest: CompactionDigest,
    },
    /// 零模型投影层批量剪枝记录（`EntryVisibility::LogOnly`）：
    /// 一次性锁存较早 `ToolResult` 的紧凑预览，原文仍完整保留在日志中供 `recall` 查阅。
    ToolResultPrune { pruned: Vec<PrunedToolResult> },
    /// 本地执行的 Shell 命令（`!` 载入上下文 / `!!` 仅本地审计不载入上下文）。
    BashExecution {
        command: String,
        output: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
        #[serde(default)]
        cancelled: bool,
        #[serde(default)]
        exclude_from_context: bool,
    },
    /// 内部审计 / 轨迹边界事件（`EntryVisibility::LogOnly`）。
    Trace { detail: SessionTraceEvent },
    /// 当前模型与推理强度的选择快照（`EntryVisibility::LogOnly`）：
    /// 不进模型上下文、不占压缩预算，仅供恢复会话与跨会话沿用上次选择时读取。
    Selection {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<ReasoningEffort>,
    },
}

/// 一次 `Selection` 事件的只读投影。三个字段各自可缺省，`None` 表示「当时未指定」。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionSelection {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl SessionEvent {
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            text: text.into(),
            images: Vec::new(),
        }
    }

    pub fn user_with_images(text: impl Into<String>, images: Vec<ImageAttachment>) -> Self {
        Self::User {
            text: text.into(),
            images,
        }
    }

    pub fn context_notice(source: impl Into<String>, text: impl Into<String>) -> Self {
        Self::ContextNotice {
            source: source.into(),
            text: text.into(),
            images: Vec::new(),
        }
    }

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
            Self::Trace { .. } | Self::ToolResultPrune { .. } | Self::Selection { .. } => {
                EntryVisibility::LogOnly
            }
            Self::BashExecution {
                exclude_from_context,
                ..
            } => {
                if *exclude_from_context {
                    EntryVisibility::LogOnly
                } else {
                    EntryVisibility::Surface
                }
            }
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

    /// 若本事件是 `Selection` 快照，投影为 `SessionSelection`。
    pub fn as_selection(&self) -> Option<SessionSelection> {
        match self {
            Self::Selection {
                provider,
                model,
                reasoning_effort,
            } => Some(SessionSelection {
                provider: provider.clone(),
                model: model.clone(),
                reasoning_effort: *reasoning_effort,
            }),
            _ => None,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::User { .. } => "user",
            Self::ContextNotice { .. } => "context-notice",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "tool-result",
            Self::Compaction { .. } => "compaction",
            Self::ToolResultPrune { .. } => "tool-result-prune",
            Self::BashExecution { .. } => "bash-execution",
            Self::Trace { detail } => detail.kind_name(),
            Self::Selection { .. } => "selection",
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

/// 面向情景回溯（`recall`）的持久化层检索参数。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RecallSearchQuery {
    pub query: String,
    /// 限定搜索的会话 ID 列表；为空时结合 `cwd` 搜索工作区会话。
    #[serde(default)]
    pub session_ids: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    pub limit: usize,
}

/// 单条情景回溯命中结果（带会话来源与 `seq` 锚点）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RecallSearchHit {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_title: Option<String>,
    pub seq: u64,
    pub ts: u128,
    pub kind: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub struct LoadedSession {
    pub header: SessionHeader,
    pub entries: Vec<SessionEntry>,
    /// 读取时因无法反序列化而跳过的条目数（例如用旧版二进制打开新版写入的会话）。
    /// `> 0` 时调用方应告警：历史并未丢，仅本次水合并未载入这些条目。
    pub skipped_events: usize,
}

/// 从首条用户输入提取简短单行会话标题（超过 72 字符自动截断）。
pub fn derive_session_title(input: &str) -> Option<String> {
    let first_line = input.lines().map(str::trim).find(|line| !line.is_empty())?;
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
