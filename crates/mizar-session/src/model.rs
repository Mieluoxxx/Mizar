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
    /// 非上下文用量账本记录（`EntryVisibility::LogOnly`，例如 `cache_warm` / `session_title` / `subagent`）：
    /// 依托既有 `(kind, payload_json)` 持久化表落盘，绝不进入 `derive_messages()`。
    Usage {
        kind: String,
        provider: String,
        model: String,
        usage: TokenUsage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
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

    pub fn usage(
        kind: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        usage: TokenUsage,
        note: Option<String>,
    ) -> Self {
        Self::Usage {
            kind: kind.into(),
            provider: provider.into(),
            model: model.into(),
            usage,
            note,
        }
    }

    pub fn visibility(&self) -> EntryVisibility {
        match self {
            Self::Trace { .. }
            | Self::ToolResultPrune { .. }
            | Self::Selection { .. }
            | Self::Usage { .. } => EntryVisibility::LogOnly,
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
            Self::Usage { .. } => "usage",
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum TitleSource {
    Auto,
    Manual,
}

impl TitleSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "manual" => Some(Self::Manual),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum TitlePatch {
    #[default]
    Keep,
    Set(String),
    Clear,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TitleSourcePatch {
    #[default]
    Keep,
    Set(TitleSource),
    Clear,
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
    pub title_source: Option<TitleSource>,
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
            title_source: None,
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
    pub title: TitlePatch,
    pub title_source: TitleSourcePatch,
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
    /// 持久化存储中实际见到的最大事件 seq（含被跳过的未知事件）。
    pub max_persisted_seq: Option<u64>,
    /// 读取时因无法反序列化而跳过的条目数（例如用旧版二进制打开新版写入的会话）。
    /// `> 0` 时调用方应告警：历史并未丢，仅本次水合并未载入这些条目。
    pub skipped_events: usize,
}

fn is_mizar_bin_token(tok: &str) -> bool {
    let trimmed = tok.trim_matches(|c| c == '"' || c == '\'');
    let base = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    base == "mizar" || base == "mizar.exe"
}

fn strip_mizar_resume_prefix(line: &str) -> &str {
    let bytes = line.as_bytes();
    let next_token = |start: usize| -> Option<(usize, usize, &str)> {
        let mut i = start;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        let quote = if bytes[i] == b'"' || bytes[i] == b'\'' {
            let q = bytes[i];
            i += 1;
            Some(q)
        } else {
            None
        };
        let tok_start = if quote.is_some() { i - 1 } else { i };
        while i < bytes.len() {
            if let Some(q) = quote {
                if bytes[i] == q {
                    i += 1;
                    break;
                }
            } else if bytes[i].is_ascii_whitespace() {
                break;
            }
            i += 1;
        }
        Some((tok_start, i, &line[tok_start..i]))
    };

    let Some((_, mut cursor, first_tok)) = next_token(0) else {
        return line;
    };
    if !is_mizar_bin_token(first_tok) {
        return line;
    }

    let mut saw_resume = false;
    while let Some((_, tok_end, tok)) = next_token(cursor) {
        match tok {
            "--resume" | "-r" => {
                saw_resume = true;
                cursor = tok_end;
                if let Some((_, id_end, id_tok)) = next_token(cursor)
                    && !id_tok.starts_with('-')
                {
                    cursor = id_end;
                }
                break;
            }
            t if t.starts_with("--resume=") => {
                saw_resume = true;
                cursor = tok_end;
                break;
            }
            "-C" | "--workspace" | "-t" | "--tools" | "-xt" | "--exclude-tools" | "--theme" => {
                cursor = tok_end;
                if let Some((_, val_end, _)) = next_token(cursor) {
                    cursor = val_end;
                } else {
                    break;
                }
            }
            t if t.starts_with('-') => {
                cursor = tok_end;
            }
            _ => break,
        }
    }

    if saw_resume {
        line[cursor..].trim_start()
    } else {
        line
    }
}

fn is_abs_path_prefix_boundary(prev: char) -> bool {
    prev.is_whitespace()
        || matches!(
            prev,
            '(' | '[' | '{' | '<' | '"' | '\'' | '`' | ',' | '=' | ';'
        )
        || !prev.is_ascii()
}

fn is_path_stop_char(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '"' | '\''
                | '`'
                | ')'
                | ']'
                | '}'
                | '>'
                | ','
                | ';'
                | '，'
                | '。'
                | '；'
                | '：'
                | '、'
                | '（'
                | '）'
                | '【'
                | '】'
                | '《'
                | '》'
                | '“'
                | '”'
                | '‘'
                | '’'
        )
}

fn shorten_absolute_paths(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    let bytes = input.as_bytes();

    while i < input.len() {
        let ch = input[i..].chars().next().unwrap();
        let prev_char = if i == 0 {
            None
        } else {
            input[..i].chars().next_back()
        };

        // Unix 绝对路径：`/a/b/c`（排除 URL `://` 与根下只有 1~2 段的 `/settings`、`/tmp/x`）
        if ch == '/'
            && prev_char.is_none_or(is_abs_path_prefix_boundary)
            && !input[i + 1..].starts_with('/')
        {
            let mut end = i + 1;
            while end < input.len() {
                let next_ch = input[end..].chars().next().unwrap();
                if is_path_stop_char(next_ch) {
                    break;
                }
                end += next_ch.len_utf8();
            }
            let raw_path = &input[i..end];
            let trimmed_path = raw_path.trim_end_matches('/');
            let segs: Vec<&str> = trimmed_path.split('/').filter(|s| !s.is_empty()).collect();
            if segs.len() >= 3 {
                out.push_str(segs[segs.len() - 2]);
                out.push('/');
                out.push_str(segs[segs.len() - 1]);
                i = end;
                continue;
            }
        }

        // Windows 驱动器绝对路径：`C:\a\b\c`
        if ch.is_ascii_alphabetic()
            && prev_char.is_none_or(is_abs_path_prefix_boundary)
            && bytes.get(i + 1) == Some(&b':')
            && bytes.get(i + 2) == Some(&b'\\')
        {
            let mut end = i + 3;
            while end < input.len() {
                let next_ch = input[end..].chars().next().unwrap();
                if is_path_stop_char(next_ch) {
                    break;
                }
                end += next_ch.len_utf8();
            }
            let raw_path = &input[i + 3..end];
            let segs: Vec<&str> = raw_path
                .trim_end_matches('\\')
                .split('\\')
                .filter(|s| !s.is_empty())
                .collect();
            if segs.len() >= 3 {
                out.push_str(segs[segs.len() - 2]);
                out.push('\\');
                out.push_str(segs[segs.len() - 1]);
                i = end;
                continue;
            }
        }

        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_uuid_boundary_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_'
}

fn looks_like_uuid_at(bytes: &[u8], idx: usize) -> bool {
    if idx + 36 > bytes.len() {
        return false;
    }
    if idx > 0 && is_uuid_boundary_char(bytes[idx - 1]) {
        return false;
    }
    if idx + 36 < bytes.len() && is_uuid_boundary_char(bytes[idx + 36]) {
        return false;
    }
    for (offset, &b) in bytes[idx..idx + 36].iter().enumerate() {
        if matches!(offset, 8 | 13 | 18 | 23) {
            if b != b'-' {
                return false;
            }
        } else if !b.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

fn shorten_uuids(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    while i < input.len() {
        if looks_like_uuid_at(bytes, i) {
            out.push_str(&input[i..i + 8]);
            i += 36;
        } else {
            let ch = input[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// 从首条用户输入提取简短单行会话标题（剥离 `mizar --resume` 前缀、缩窄绝对路径与 UUID，超 72 字符截断）。
pub fn derive_session_title(input: &str) -> Option<String> {
    let first_line = input
        .lines()
        .map(str::trim)
        .map(strip_mizar_resume_prefix)
        .find(|line| !line.is_empty())?;
    let paths_cleaned = shorten_absolute_paths(first_line);
    let uuid_cleaned = shorten_uuids(&paths_cleaned);
    let collapsed = uuid_cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
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
