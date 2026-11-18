//! 纯轨迹领域模型、统计折叠投影、三泳道时间轴与异常检测规则引擎。
//!
//! 本 crate 不依赖 `cordis`、`tokio` 或任何在线运行时，供 `mizar-trajectory` 与纯 UI 层共用。

pub mod detector;
pub mod model;
pub mod projection;
pub mod timeline;

pub use detector::{
    DetectorConfig, canonicalize_args_str, canonicalize_json, detect_anomalies, preview_chars,
    sort_json_value, wildcard_matches,
};
pub use model::{
    RecordKind, SessionStats, Severity, TimelineMode, TimelineModel, TimelineSpan,
    TimelineTurnBoundary, TraceEvent, TraceEventKind, TrajectoryFinding, TrajectoryRecord,
    TrajectoryReport, TrajectorySnapshot, TrajectoryStep, TrajectorySummary, TrajectoryTurn,
};
pub use projection::fold_trajectory;
pub use timeline::derive_timeline;
