//! 轨迹采集、纯折叠投影、三泳道时间轴、异常检测与重复调用守卫
//!（参考 `deepseek-harness` `ui-trajectory` / `session-stats` / `repeat-tool-reminder`）。

pub mod collector;
pub mod export;
pub mod guard;

pub use collector::{
    TrajectoryAnomalyEvent, TrajectoryConfig, TrajectoryEventEmitted, TrajectoryRuntime,
    prepare_trace_jsonl_file, trace_events_from_entries, trace_events_from_session,
};
pub use export::{
    DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS, TRAJECTORY_EXPORT_SCHEMA_VERSION, TraceCompactionEvent,
    TraceExportMeta, TraceStepSummary, TraceToolCallEntry, TraceToolResultSummary,
    TraceTurnSummary, TraceUsageSummary, TrajectoryExport, build_trajectory_export,
    extract_compaction_events_from_entries, truncate_tool_result_summary,
    write_trace_export_atomic,
};
pub use guard::{
    GuardRepeatEvent, RepeatGuardConfig, RepeatGuardValidationError, install_repeat_guard,
    validate_repeat_guard_config,
};
pub use mizar_types::trajectory::{
    DetectorConfig, RecordKind, SessionStats, Severity, TimelineMode, TimelineModel, TimelineSpan,
    TimelineTurnBoundary, TraceEvent, TraceEventKind, TrajectoryFinding, TrajectoryRecord,
    TrajectoryReport, TrajectorySnapshot, TrajectoryStep, TrajectorySummary, TrajectoryTurn,
    canonicalize_args_str, canonicalize_json, derive_timeline, detect_anomalies, fold_trajectory,
    preview_chars, sort_json_value, wildcard_matches,
};
