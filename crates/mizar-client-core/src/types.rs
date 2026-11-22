//! 客户端核心共享契约类型（供 TUI 与桌面端 Merak 复用，零终端/异步运行时依赖）。
//!
//! 迁移自 `mizar-tui-ui` 的 `app.rs` / `overlay.rs` / `side_panel.rs` 纯数据契约与辅助方法。

use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use mizar_llm_types::{ContentBlock, ImageAttachment, ReasoningEffort, TokenUsage, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache_audit::{CacheMissAudit, CacheWarmingSessionStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolState {
    Running,
    Done,
    Error,
    Blocked,
}

/// 主界面前后台运行状态机：替代分散状态字段，消除状态组合失效与互相踩踏。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ActivityState {
    #[default]
    Idle,
    /// Agent 对话回合运行中；自动压缩作为子阶段内嵌于此状态中，不会遮蔽 `is_busy()` 语义。
    Agent {
        streaming_assistant_open: bool,
        compaction_status: Option<String>,
    },
    /// 用户触发 `!` / `!!` 本地 Bash 命令执行中。
    LocalBash,
    /// 独立触发的 `/compact` 手动上下文压缩进行中。
    Compacting { status: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoticeLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ApprovalMode {
    #[default]
    Auto,
    Ask,
}

impl ApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    AllowOnce,
    AllowAlwaysForTool,
    Deny,
    DenyWithReason(String),
}

#[derive(Debug)]
pub struct PendingApproval {
    pub tool_name: String,
    pub args_preview: String,
    pub editing_reason: bool,
    pub deny_reason: String,
    pub responder: Option<oneshot::Sender<ApprovalDecision>>,
}

/// 工具卡富展示附属数据（存放在 side-table 以保持聊天条目轻量）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExtraMeta {
    pub full_content: String,
    pub diff_raw: Option<String>,
    pub full_args: String,
    pub inline_meta: Option<String>,
    /// 为 `true` 时（如 `!!` 本地命令），不计入模型上下文估算。
    pub exclude_from_context: bool,
}

/// 输入前缀识别得到的 Bash 交互模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BashInputMode {
    /// `!` 开头：执行结果载入后续对话上下文。
    Context,
    /// `!!` 开头：执行结果仅本地展示与审计，不载入上下文。
    NoContext,
}

impl BashInputMode {
    pub fn from_input(input: &str) -> Option<Self> {
        let trimmed = input.trim_start();
        if trimmed.starts_with("!!") {
            Some(Self::NoContext)
        } else if trimmed.starts_with('!') {
            Some(Self::Context)
        } else {
            None
        }
    }

    pub fn is_context(self) -> bool {
        matches!(self, Self::Context)
    }

    pub fn exclude_from_context(self) -> bool {
        matches!(self, Self::NoContext)
    }

    /// 剥离 `!` / `!!` 前缀并裁剪两端空白，返回待执行命令。
    pub fn extract_command(self, input: &str) -> &str {
        let trimmed = input.trim();
        match self {
            Self::NoContext => trimmed.strip_prefix("!!").unwrap_or("").trim(),
            Self::Context => trimmed.strip_prefix('!').unwrap_or("").trim(),
        }
    }
}

/// 本地 Bash 命令异步执行完成后的 UI 投影负载。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalBashOutput {
    pub display: String,
    pub session_output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiAgentStatus {
    Busy,
    Idle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiTurnEndReason {
    Completed,
    Truncated,
    MaxSteps,
    Cancelled,
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiStepOutcome {
    Completed,
    ToolCalls,
    MaxTokens,
    Cancelled,
    Error { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiLogLevel {
    Silent,
    Error,
    Warn,
    Info,
    Debug,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiLogRecord {
    pub level: UiLogLevel,
    pub name: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionReplayItem {
    User {
        text: String,
    },
    ContextNotice {
        source: String,
        text: String,
    },
    Notice {
        level: NoticeLevel,
        text: String,
    },
    Compaction {
        summary: String,
        first_kept_seq: u64,
    },
    Assistant {
        blocks: Vec<ContentBlock>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
    BashExecution {
        command: String,
        output: String,
        exit_code: Option<i32>,
        cancelled: bool,
        exclude_from_context: bool,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionViewData {
    pub session_id: String,
    pub session_title: Option<String>,
    pub items: Vec<SessionReplayItem>,
    pub finish_notice: String,
    /// 该会话最后记录的模型 / 推理强度快照；`None` 表示日志里没有（旧会话）。
    /// 仅作数据携带——是否采用、如何叠加进优先级链由宿主层决定，本层不参与决策。
    pub selection: Option<SessionSelection>,
    /// 该会话累计的 Token 用量（来自 `SessionEvent::Assistant.usage` 累加）。
    pub total_usage: TokenUsage,
    /// 该会话最近一次实测（或压缩后）上下文 token 数，供跨 resume 恢复占用估算。
    pub last_prompt_tokens: Option<u64>,
    /// 当前会话的 Prompt Cache 保活实时状态（仅运行期可读时填充）。
    #[serde(default)]
    pub cache_warming: Option<CacheWarmingSessionStatus>,
    /// 从会话账本统计出的意外缓存重计费（`Cache Re-billed`）汇总。
    #[serde(default)]
    pub cache_miss_audit: CacheMissAudit,
}

/// 会话内记录的模型与推理强度选择（客户端侧的纯数据形态）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSelection {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// 排队待发送的一条消息。
///
/// 用单调递增的 `id` 而不是下标寻址：`RunFinished` 的自动出队会在一瞬间改变下标，
/// 选中态若按位置记录就会「高亮第二条却撤回第三条」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedPrompt {
    pub id: u64,
    pub text: String,
    pub images: Vec<(String, ImageAttachment)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillPickerItem {
    pub name: String,
    pub description: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTreeEntry {
    pub seq: u64,
    pub kind_tag: String,
    pub summary: String,
    pub depth: usize,
    pub can_fork: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPickerItem {
    pub provider_id: String,
    pub model_id: String,
    pub label: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPickerItem {
    pub session_id: String,
    pub title: String,
    pub cwd_tail: String,
    pub event_count: u64,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettingItemKind {
    Toggle {
        value: bool,
    },
    Choice {
        options: Vec<String>,
        selected: usize,
    },
    Model {
        provider: Option<String>,
        model: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettingEntry {
    pub key: String,
    pub category: String,
    pub label: String,
    pub description: String,
    pub kind: SettingItemKind,
}

impl SettingEntry {
    pub fn display_value(&self) -> String {
        match &self.kind {
            SettingItemKind::Toggle { value } => {
                if *value {
                    "● ON".to_string()
                } else {
                    "○ OFF".to_string()
                }
            }
            SettingItemKind::Choice { options, selected } => options
                .get(*selected)
                .map(|v| format!("◂ {v} ▸"))
                .unwrap_or_default(),
            SettingItemKind::Model { provider, model } => {
                let p = provider.as_deref().map(str::trim).unwrap_or("");
                let m = model.as_deref().map(str::trim).unwrap_or("");
                match (p.is_empty(), m.is_empty()) {
                    (true, true) => "跟随会话模型".to_string(),
                    (false, false) => format!("{p}/{m}"),
                    (true, false) => m.to_string(),
                    (false, true) => p.to_string(),
                }
            }
        }
    }

    pub fn step_value(&mut self, delta: isize) -> String {
        match &mut self.kind {
            SettingItemKind::Toggle { value } => {
                *value = !*value;
                if *value { "true" } else { "false" }.to_string()
            }
            SettingItemKind::Choice { options, selected } => {
                if options.is_empty() {
                    return String::new();
                }
                let len = options.len() as isize;
                let next = ((*selected as isize + delta).rem_euclid(len)) as usize;
                *selected = next;
                options[next].clone()
            }
            SettingItemKind::Model { provider, model } => {
                let p = provider.as_deref().map(str::trim).unwrap_or("");
                let m = model.as_deref().map(str::trim).unwrap_or("");
                match (p.is_empty(), m.is_empty()) {
                    (false, false) => format!("{p}/{m}"),
                    (true, false) => m.to_string(),
                    _ => String::new(),
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSummaryRow {
    pub name: String,
    pub exposure: String,
    pub declared: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSummaryRow {
    pub name: String,
    pub status: String,
    pub ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTouchRow {
    pub path: String,
    pub op: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionSummaryRow {
    pub path: String,
    pub scope: String,
    pub bytes: usize,
    pub truncated: bool,
    pub omitted: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextBreakdownSnapshot {
    pub system_tokens: u64,
    pub user_tokens: u64,
    pub assistant_tokens: u64,
    pub reasoning_tokens: u64,
    pub tool_tokens: u64,
    pub context_window: u64,
    pub tps: Option<f64>,
    pub ttft_ms: Option<u64>,
    pub tps_history: Vec<f64>,
    pub ttft_history: Vec<u64>,
    /// 累计缓存读取量（provider 命中缓存、未计入 `input_tokens` 的部分）。
    pub cache_read_tokens: u64,
    /// 累计缓存写入量（Anthropic `cache_creation_input_tokens`）。
    pub cache_write_tokens: u64,
    /// 累计净输入（未命中缓存的部分）。**不是** prompt 全量。
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// 累计四项总和（`input + cache_read + cache_write + output`）。
    pub total_tokens: u64,
    /// 最近一次请求的完整 prompt 体积，用于估算当前上下文占用；无用量时回退到字符估算。
    pub last_prompt_tokens: Option<u64>,
}

impl ContextBreakdownSnapshot {
    /// 累计请求过的 prompt 总量（含命中与写入缓存的部分）。
    pub fn cumulative_prompt_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }

    /// 累计缓存命中率（`0.0..=100.0`）。分母是 prompt 全量，不是净输入。
    pub fn cache_hit_rate(&self) -> f64 {
        let prompt = self.cumulative_prompt_tokens();
        if prompt == 0 {
            return 0.0;
        }
        (self.cache_read_tokens as f64 / prompt as f64) * 100.0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SidePanelData {
    pub context: ContextBreakdownSnapshot,
    pub instructions: Vec<InstructionSummaryRow>,
    pub files: Vec<FileTouchRow>,
    pub git_changes: Vec<FileTouchRow>,
    pub git_diff_stat: String,
    pub tools: Vec<ToolSummaryRow>,
    pub mcps: Vec<McpSummaryRow>,
    pub findings: Vec<String>,
}

/// 由后端异步事件与宿主扩展投递给前端（TUI / Merak）的统一客户端事件。
#[derive(Debug, Clone)]
pub enum ClientEvent {
    Status(UiAgentStatus),
    TurnStart {
        turn: u64,
    },
    TurnEnd {
        reason: UiTurnEndReason,
    },
    StepStart {
        ts_ms: u128,
    },
    StepEnd {
        step: usize,
        ts_ms: u128,
        outcome: UiStepOutcome,
        usage: Option<TokenUsage>,
    },
    FirstToken {
        ts_ms: u128,
    },
    ReasoningDelta {
        text: String,
        ts_ms: u128,
    },
    TextDelta {
        text: String,
        ts_ms: u128,
    },
    ToolCall(ToolCall),
    ToolUpdate {
        call_id: String,
        name: String,
        preview: String,
    },
    ToolResult {
        call_id: String,
        name: String,
        content: String,
        is_error: bool,
        blocked: bool,
        details: Option<Value>,
    },
    Usage(TokenUsage),
    Anomaly(String),
    Flash(String),
    Notice {
        level: NoticeLevel,
        text: String,
    },
    CacheWarmingNotice {
        text: String,
    },
    SubagentState {
        running_count: usize,
        notice: Option<String>,
    },
    UserSkillsUpdated(Vec<String>),
    SessionTitleUpdated(Option<String>),
    ToolsUpdated,
    Log(UiLogRecord),
    RunFinished(Result<String, String>),
    LocalBashFinished {
        call_id: String,
        command: String,
        exclude_from_context: bool,
        output: Result<LocalBashOutput, String>,
    },
    ApprovalRequested {
        tool_name: String,
        args_preview: String,
        responder: Arc<Mutex<Option<oneshot::Sender<ApprovalDecision>>>>,
    },
    CompactionStarted {
        trigger: String,
    },
    CompactionProgress {
        stage: String,
        delta_chars: usize,
    },
    CompactionFinished {
        outcome: String,
        tokens_before: Option<u64>,
        tokens_after: Option<u64>,
        pruned_count: usize,
        first_kept_seq: Option<u64>,
        summary_preview: Option<String>,
        message: Option<String>,
        is_manual: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_client_dtos_roundtrip_serde_and_keep_helpers() {
        assert_eq!(ApprovalMode::Auto.as_str(), "auto");
        assert_eq!(ApprovalMode::Ask.as_str(), "ask");
        assert_eq!(
            BashInputMode::from_input("  !! ls -la").map(|m| (m, m.extract_command("!! ls -la"))),
            Some((BashInputMode::NoContext, "ls -la"))
        );

        let mut setting = SettingEntry {
            key: "theme".into(),
            category: "外观".into(),
            label: "主题".into(),
            description: "切换主题".into(),
            kind: SettingItemKind::Choice {
                options: vec!["dark".into(), "light".into()],
                selected: 0,
            },
        };
        assert_eq!(setting.display_value(), "◂ dark ▸");
        assert_eq!(setting.step_value(-1), "light");

        let panel = SidePanelData {
            context: ContextBreakdownSnapshot {
                input_tokens: 200,
                cache_read_tokens: 600,
                cache_write_tokens: 200,
                ..ContextBreakdownSnapshot::default()
            },
            tools: vec![ToolSummaryRow {
                name: "read".into(),
                exposure: "direct".into(),
                declared: true,
            }],
            ..SidePanelData::default()
        };
        assert_eq!(panel.context.cumulative_prompt_tokens(), 1_000);
        assert!((panel.context.cache_hit_rate() - 60.0).abs() < 1e-9);

        let json = serde_json::to_string(&panel).unwrap();
        let decoded: SidePanelData = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, panel);

        let view = SessionViewData {
            session_id: "s-1".into(),
            session_title: Some("标题".into()),
            items: vec![
                SessionReplayItem::User {
                    text: "hello".into(),
                },
                SessionReplayItem::BashExecution {
                    command: "pwd".into(),
                    output: "/tmp".into(),
                    exit_code: Some(0),
                    cancelled: false,
                    exclude_from_context: true,
                },
            ],
            selection: Some(SessionSelection {
                provider: Some("deepseek".into()),
                model: Some("deepseek-chat".into()),
                reasoning_effort: Some(ReasoningEffort::Medium),
            }),
            ..SessionViewData::default()
        };
        let view_json = serde_json::to_string(&view).unwrap();
        assert_eq!(
            serde_json::from_str::<SessionViewData>(&view_json).unwrap(),
            view
        );
    }
}
