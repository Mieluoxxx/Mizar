//! 轨迹采集、纯折叠投影、三泳道时间轴、异常检测与重复调用守卫
//!（参考 `deepseek-harness` `ui-trajectory` / `session-stats` / `repeat-tool-reminder`）。

pub mod collector;
pub mod detector;
pub mod guard;
pub mod model;
pub mod projection;
pub mod timeline;

pub use collector::{
    EVENT_TRAJECTORY_ANOMALY, EVENT_TRAJECTORY_EVENT, TrajectoryAnomalyEvent, TrajectoryConfig,
    TrajectoryContextExt, TrajectoryEventEmitted, TrajectoryRuntime, trace_events_from_entries,
    trace_events_from_session,
};
pub use detector::{
    DetectorConfig, canonicalize_args_str, canonicalize_json, detect_anomalies, preview_chars,
    sort_json_value, wildcard_matches,
};
pub use guard::{EVENT_GUARD_REPEAT, GuardRepeatEvent, RepeatGuardConfig, RepeatToolGuardPlugin};
pub use model::{
    RecordKind, SessionStats, Severity, TimelineMode, TimelineModel, TimelineSpan,
    TimelineTurnBoundary, TraceEvent, TraceEventKind, TrajectoryFinding, TrajectoryRecord,
    TrajectoryReport, TrajectorySnapshot, TrajectoryStep, TrajectorySummary, TrajectoryTurn,
};
pub use projection::fold_trajectory;
pub use timeline::derive_timeline;
