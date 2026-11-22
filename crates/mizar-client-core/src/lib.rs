//! 跨终端与桌面（Merak）共享的纯客户端契约、排队队列、文件缓存与算法层。
//!
//! 本 crate 严禁引入 `tokio`、`cordis`、`reqwest`、`rquickjs`、`rusqlite`、`ratatui` 或 `crossterm`。

use serde::{Deserialize, Serialize};

pub mod cache_audit;
pub mod mentions;
pub mod queue;
pub mod reveal;
pub mod tool_format;
pub mod tps;
pub mod types;
pub mod workspace_cache;

pub use cache_audit::{
    CacheMissAudit, CacheWarmingSessionStatus, MIN_CACHE_DROP_TOKENS, UsageAuditSample,
    calculate_usage_cost, compute_cache_miss_audit,
};
pub use mizar_llm_types::CacheWarmingMode;

pub use mentions::{
    FILE_MENTIONS_SOURCE, MentionCandidate, MentionCandidateKind, MentionCaret, MentionToken,
    extract_mentions, format_mention_replacement, fuzzy_subsequence_score, is_cjk_punctuation,
    is_mention_boundary_char, is_opening_wrapper, is_path_like_query, matching_closer,
    mention_at_caret, split_line_suffix,
};
pub use queue::PromptQueue;
pub use reveal::{
    REVEAL_CATCHUP_FRAMES, REVEAL_MIN_STEP_BYTES, RevealController, RevealSlot, reveal_step,
};
pub use tool_format::{
    MAX_ARGS_SUMMARY_COLS, MAX_PREVIEW_LINE_COLS, abbreviate_home_path, abbreviate_home_with,
    bash_inline_meta, bash_scope_label, clean_bash_output_lines, count_diff_stats,
    extract_inline_meta, format_bash_display, format_diff_stat, format_expanded_tool_args,
    format_tokens, is_path_arg_tool, is_read_only_compact_tool, short_id, single_line_text,
    summarize_live_preview, summarize_tool_args, summarize_tool_result, truncate_display,
};
pub use tps::{MIN_TPS_DECODE_MS, TpsLedger, estimate_tokens_fraction};
pub use types::{
    ActivityState, ApprovalDecision, ApprovalMode, BashInputMode, ClientEvent,
    ContextBreakdownSnapshot, FileTouchRow, InstructionSummaryRow, LocalBashOutput, McpSummaryRow,
    ModelPickerItem, NoticeLevel, PendingApproval, QueuedPrompt, SessionPickerItem,
    SessionReplayItem, SessionSelection, SessionTreeEntry, SessionViewData, SettingEntry,
    SettingItemKind, SidePanelData, SkillPickerItem, ToolExtraMeta, ToolState, ToolSummaryRow,
    UiAgentStatus, UiLogLevel, UiLogRecord, UiStepOutcome, UiTurnEndReason,
};
pub use workspace_cache::{
    WorkspaceFileCache, expand_home_dir_prefix, list_path_candidates, rank_mention_candidates,
};

pub const PLUGIN_FILE_MENTIONS: &str = "file-mentions";
pub const PLUGIN_LOCAL_BASH: &str = "local-bash";
pub const PLUGIN_APPROVAL_GATE: &str = "approval-gate";
pub const PLUGIN_TUI: &str = "tui";
pub const PLUGIN_TUI_EMPHASIS: &str = "tui-emphasis";
pub const PLUGIN_TUI_SIDE_PANEL: &str = "tui-side-panel";
pub const PLUGIN_TUI_TIMELINE_RAIL: &str = "tui-timeline-rail";
pub const PLUGIN_TUI_TRAJECTORY_VIEW: &str = "tui-trajectory-view";
pub const PLUGIN_TUI_SELECTION: &str = "tui-selection";
pub const PLUGIN_TUI_CLIPBOARD: &str = "tui-clipboard";

/// 客户端宿主的官方扩展插件启用开关。
///
/// 由 `settings.json` 中的 `disabledPlugins` 归一化派生：禁用总开关 `"tui"` 会级联关闭
/// 全部 `tui_*` 子特性，但不影响宿主的 `"file-mentions"` / `"local-bash"` / `"approval-gate"`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostExtensionFlags {
    pub file_mentions: bool,
    pub local_bash: bool,
    pub approval_gate: bool,
    pub tui: bool,
    pub tui_emphasis: bool,
    pub tui_side_panel: bool,
    pub tui_timeline_rail: bool,
    pub tui_trajectory_view: bool,
    pub tui_selection: bool,
    pub tui_clipboard: bool,
}

impl Default for HostExtensionFlags {
    fn default() -> Self {
        Self {
            file_mentions: true,
            local_bash: true,
            approval_gate: true,
            tui: true,
            tui_emphasis: true,
            tui_side_panel: true,
            tui_timeline_rail: true,
            tui_trajectory_view: true,
            tui_selection: true,
            tui_clipboard: true,
        }
    }
}

impl HostExtensionFlags {
    /// 根据 `disabledPlugins` ID 列表派生客户端扩展开关。
    pub fn from_disabled_plugins(ids: &[String]) -> Self {
        let mut flags = Self::default();
        for raw in ids {
            let id = raw.trim();
            if id.is_empty() {
                continue;
            }
            match id {
                PLUGIN_FILE_MENTIONS => flags.file_mentions = false,
                PLUGIN_LOCAL_BASH => flags.local_bash = false,
                PLUGIN_APPROVAL_GATE => flags.approval_gate = false,
                PLUGIN_TUI => {
                    flags.tui = false;
                    flags.tui_emphasis = false;
                    flags.tui_side_panel = false;
                    flags.tui_timeline_rail = false;
                    flags.tui_trajectory_view = false;
                    flags.tui_selection = false;
                    flags.tui_clipboard = false;
                }
                PLUGIN_TUI_EMPHASIS => flags.tui_emphasis = false,
                PLUGIN_TUI_SIDE_PANEL => flags.tui_side_panel = false,
                PLUGIN_TUI_TIMELINE_RAIL => flags.tui_timeline_rail = false,
                PLUGIN_TUI_TRAJECTORY_VIEW => flags.tui_trajectory_view = false,
                PLUGIN_TUI_SELECTION => flags.tui_selection = false,
                PLUGIN_TUI_CLIPBOARD => flags.tui_clipboard = false,
                _ => {}
            }
        }
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_enables_all_official_extensions() {
        let flags = HostExtensionFlags::default();
        assert_eq!(
            flags,
            HostExtensionFlags {
                file_mentions: true,
                local_bash: true,
                approval_gate: true,
                tui: true,
                tui_emphasis: true,
                tui_side_panel: true,
                tui_timeline_rail: true,
                tui_trajectory_view: true,
                tui_selection: true,
                tui_clipboard: true,
            }
        );
    }

    #[test]
    fn disabling_tui_cascades_to_all_tui_subflags_only() {
        let disabled = vec![
            "".to_string(),
            "   ".to_string(),
            " tui ".to_string(),
            "unknown-plugin".to_string(),
        ];
        let flags = HostExtensionFlags::from_disabled_plugins(&disabled);
        assert!(!flags.tui);
        assert!(!flags.tui_emphasis);
        assert!(!flags.tui_side_panel);
        assert!(!flags.tui_timeline_rail);
        assert!(!flags.tui_trajectory_view);
        assert!(!flags.tui_selection);
        assert!(!flags.tui_clipboard);

        // 非 TUI 宿主扩展不受 `"tui"` 级联禁用影响
        assert!(flags.file_mentions);
        assert!(flags.local_bash);
        assert!(flags.approval_gate);

        // 显式禁用个别插件时精确关闭对应位
        let selective = HostExtensionFlags::from_disabled_plugins(&[
            PLUGIN_LOCAL_BASH.to_string(),
            PLUGIN_TUI_CLIPBOARD.to_string(),
        ]);
        assert!(selective.tui);
        assert!(selective.file_mentions);
        assert!(!selective.local_bash);
        assert!(selective.approval_gate);
        assert!(!selective.tui_clipboard);
        assert!(selective.tui_emphasis);
    }
}
