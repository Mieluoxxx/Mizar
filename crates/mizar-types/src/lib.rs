//! Mizar 纯核心数据结构、轻量钩子、轨迹模型与跨端纯客户端契约层。
//!
//! 本 crate 严禁引入 `tokio`、`reqwest`、`rquickjs`、`rusqlite`、`ratatui`、`crossterm` 或 `tracing`，
//! 供核心运行时、纯 UI 层与 Merak Desktop 共用。

pub mod client;
pub mod hooks;
pub mod llm;
pub mod rpc;
pub mod trajectory;

pub use hooks::{
    BoxError, BoxFuture, HookGuard, HookOptions, Next, SerialHook, SyncEventBus, WaterfallHook,
};

pub use client::{
    ActivityState, ApprovalDecision, ApprovalMode, BashInputMode, ClientEvent,
    ContextBreakdownSnapshot, FILE_MENTIONS_SOURCE, FileTouchRow, HostExtensionFlags,
    InstructionSummaryRow, LocalBashOutput, MAX_ARGS_SUMMARY_COLS, MAX_PREVIEW_LINE_COLS,
    MIN_TPS_DECODE_MS, McpSummaryRow, MentionCandidate, MentionCandidateKind, MentionCaret,
    MentionToken, ModelPickerItem, NoticeLevel, PLUGIN_APPROVAL_GATE, PLUGIN_FILE_MENTIONS,
    PLUGIN_LOCAL_BASH, PLUGIN_TUI, PLUGIN_TUI_CLIPBOARD, PLUGIN_TUI_EMPHASIS, PLUGIN_TUI_SELECTION,
    PLUGIN_TUI_SIDE_PANEL, PLUGIN_TUI_TIMELINE_RAIL, PLUGIN_TUI_TRAJECTORY_VIEW, PendingApproval,
    PromptQueue, QueuedPrompt, REVEAL_CATCHUP_FRAMES, REVEAL_MIN_STEP_BYTES, RevealController,
    RevealSlot, SessionPickerItem, SessionReplayItem, SessionSelection, SessionTreeEntry,
    SessionViewData, SettingEntry, SettingItemKind, SidePanelData, SkillPickerItem, ToolExtraMeta,
    ToolState, ToolSummaryRow, TpsLedger, UiAgentStatus, UiLogLevel, UiLogRecord, UiStepOutcome,
    UiTurnEndReason, WorkspaceFileCache, abbreviate_home_path, abbreviate_home_with,
    bash_inline_meta, bash_scope_label, clean_bash_output_lines, count_diff_stats,
    estimate_tokens_fraction, expand_home_dir_prefix, extract_inline_meta, extract_mentions,
    format_bash_display, format_diff_stat, format_expanded_tool_args, format_mention_replacement,
    format_tokens, fuzzy_subsequence_score, is_cjk_punctuation, is_mention_boundary_char,
    is_opening_wrapper, is_path_arg_tool, is_path_like_query, is_read_only_compact_tool,
    list_path_candidates, matching_closer, mention_at_caret, mentions, queue,
    rank_mention_candidates, reveal, reveal_step, short_id, single_line_text, split_line_suffix,
    summarize_live_preview, summarize_tool_args, summarize_tool_result, tool_format, tps,
    truncate_display, workspace_cache,
};
pub use llm::{
    BlockAssembler, CacheMissAudit, CacheRetention, ContentBlock, ContentBlockKind, FinishReason,
    ImageAttachment, LlmError, LlmErrorCode, MAX_IMAGES_PER_MESSAGE, MIN_CACHE_DROP_TOKENS,
    Message, ModelCostRates, ModelCostTier, ModelInfo, OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH,
    PromptCacheLifetime, ProviderInfo, ReasoningEffort, ReplayEnvelope, RequestOrigin,
    SessionAffinityMode, SseEvent, SseParser, StreamChunk, TokenUsage, ToolCall, ToolSchema,
    UsageAuditSample, assembler, cache_audit, calculate_usage_cost, clamp_openai_prompt_cache_key,
    compute_cache_miss_audit, detect_session_affinity_mode, error, is_context_window_exceeded,
    is_empty_response_error, is_quota_exceeded, parse_session_affinity_mode, prefix_digest,
    push_header_if_absent, push_session_affinity_headers, sse,
};
pub use trajectory::{
    DetectorConfig, IncrementalDetector, IncrementalFolder, RecordKind, SessionStats, Severity,
    TimelineMode, TimelineModel, TimelineSpan, TimelineTurnBoundary, TraceEvent, TraceEventKind,
    TrajectoryFinding, TrajectoryRecord, TrajectoryReport, TrajectorySnapshot, TrajectoryStep,
    TrajectorySummary, TrajectoryTurn, canonicalize_args_str, canonicalize_json, derive_timeline,
    detect_anomalies, detector, fold_trajectory, model, preview_chars, projection, sort_json_value,
    timeline, wildcard_matches,
};
