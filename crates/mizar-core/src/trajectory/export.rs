//! 无头模式与离线分析的结构化轨迹 JSON 导出与原子落盘。
//!
//! 从权威 `TrajectoryReport` 与原始 `SessionEntry` 组装自包含的 `TrajectoryExport`，
//! 保留首字时间、每轮 token/cache 用量、完整工具调用参数、截断后工具结果摘要、
//! 压缩事件、异常检测报告与终止原因，并支持跨目录自动创建与原子写盘。

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_llm::TokenUsage;
use mizar_session::{SessionEntry, SessionEvent, SessionTraceEvent};
use mizar_types::trajectory::{
    RecordKind, SessionStats, TrajectoryReport, TrajectorySnapshot, preview_chars,
};
use serde::{Deserialize, Serialize};

/// 导出结构版本号。
pub const TRAJECTORY_EXPORT_SCHEMA_VERSION: u32 = 1;
/// 工具结果摘要默认最大字符数。
pub const DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS: usize = 512;

static ATOMIC_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// 导出元信息（由执行宿主在退出前传入）。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceExportMeta {
    pub exit_code: i32,
    pub termination_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub isolated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
}

/// 汇总后的 token 与缓存用量。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceUsageSummary {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub reasoning_tokens: u64,
    pub total_tokens: u64,
}

impl TraceUsageSummary {
    fn add_usage(&mut self, usage: &TokenUsage) {
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(usage.cache_read_tokens.unwrap_or(0));
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(usage.cache_write_tokens.unwrap_or(0));
        self.reasoning_tokens = self
            .reasoning_tokens
            .saturating_add(usage.reasoning_tokens.unwrap_or(0));
        self.total_tokens = self.total_tokens.saturating_add(usage.grand_total());
    }
}

/// 单步执行摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceStepSummary {
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

/// 单轮对话摘要（包含首字时间与每轮汇总用量）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceTurnSummary {
    pub turn: u64,
    pub started_at_ms: Option<u128>,
    pub ended_at_ms: Option<u128>,
    pub duration_ms: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub reason: Option<String>,
    pub usage: TraceUsageSummary,
    pub steps: Vec<TraceStepSummary>,
}

/// 完整工具调用记录（保留未截断参数）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceToolCallEntry {
    pub seq: u64,
    pub turn: Option<u64>,
    pub step: Option<usize>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub started_at_ms: Option<u128>,
}

/// 截断后的工具执行结果摘要。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceToolResultSummary {
    pub seq: u64,
    pub turn: Option<u64>,
    pub step: Option<usize>,
    pub tool_call_id: String,
    pub name: String,
    pub content_summary: String,
    pub original_chars: usize,
    pub truncated: bool,
    pub is_error: bool,
    pub blocked: bool,
    pub duration_ms: Option<u64>,
}

/// 结构化上下文压缩事件。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceCompactionEvent {
    pub seq: u64,
    pub ts_ms: u128,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_kept_seq: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_before: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_after: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_preview: Option<String>,
}

/// 面向无头评测与排障导出的完整 `trajectory.json` 顶层结构。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrajectoryExport {
    pub schema_version: u32,
    pub exported_at_ms: u128,
    pub session_id: String,
    pub exit_code: i32,
    pub termination_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub duration_ms: u64,
    pub ttft_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub isolated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    pub usage: TraceUsageSummary,
    pub turns: Vec<TraceTurnSummary>,
    pub tool_calls: Vec<TraceToolCallEntry>,
    pub tool_results: Vec<TraceToolResultSummary>,
    pub compactions: Vec<TraceCompactionEvent>,
    pub report: TrajectoryReport,
}

impl TrajectoryExport {
    /// 构造未生成会话或早退场景下的空结构化导出，确保任何失败路径都能完整落盘。
    pub fn empty(session_id: impl Into<String>, meta: TraceExportMeta) -> Self {
        let session_id = session_id.into();
        Self {
            schema_version: TRAJECTORY_EXPORT_SCHEMA_VERSION,
            exported_at_ms: now_millis(),
            session_id: session_id.clone(),
            exit_code: meta.exit_code,
            termination_reason: meta.termination_reason,
            error: meta.error,
            duration_ms: meta.duration_ms,
            ttft_ms: None,
            provider: meta.provider,
            model: meta.model,
            isolated: meta.isolated,
            output_format: meta.output_format,
            usage: TraceUsageSummary::default(),
            turns: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            compactions: Vec::new(),
            report: TrajectoryReport {
                snapshot: TrajectorySnapshot {
                    session_id,
                    stats: SessionStats::default(),
                    turns: Vec::new(),
                    records: Vec::new(),
                },
                timeline: None,
                findings: Vec::new(),
            },
        }
    }
}

/// 按 Unicode 字符数截断工具输出为摘要，返回 `(摘要文本, 原始字符数, 是否截断)`。
pub fn truncate_tool_result_summary(content: &str, max_chars: usize) -> (String, usize, bool) {
    let original_chars = content.chars().count();
    if original_chars <= max_chars {
        return (content.to_string(), original_chars, false);
    }
    let head: String = content.chars().take(max_chars).collect();
    (
        format!("{head}… [truncated {original_chars} chars]"),
        original_chars,
        true,
    )
}

/// 从原始 `SessionEntry` 序列提取结构化压缩事件（start / checkpoint / end）。
pub fn extract_compaction_events_from_entries(
    entries: &[SessionEntry],
    max_summary_chars: usize,
) -> Vec<TraceCompactionEvent> {
    let mut events = Vec::new();
    for entry in entries {
        match &entry.event {
            SessionEvent::Trace {
                detail: SessionTraceEvent::CompactionStart { trigger },
            } => {
                events.push(TraceCompactionEvent {
                    seq: entry.seq,
                    ts_ms: entry.ts,
                    kind: "start".to_string(),
                    trigger: Some(trigger.clone()),
                    status: None,
                    first_kept_seq: None,
                    tokens_before: None,
                    tokens_after: None,
                    summary_preview: None,
                });
            }
            SessionEvent::Compaction {
                summary,
                first_kept_seq,
                tokens_before,
                tokens_after,
                ..
            } => {
                events.push(TraceCompactionEvent {
                    seq: entry.seq,
                    ts_ms: entry.ts,
                    kind: "checkpoint".to_string(),
                    trigger: None,
                    status: Some("compacted".to_string()),
                    first_kept_seq: Some(*first_kept_seq),
                    tokens_before: *tokens_before,
                    tokens_after: *tokens_after,
                    summary_preview: Some(preview_chars(summary, max_summary_chars)),
                });
            }
            SessionEvent::Trace {
                detail:
                    SessionTraceEvent::CompactionEnd {
                        status,
                        first_kept_seq,
                        tokens_before,
                        tokens_after,
                    },
            } => {
                events.push(TraceCompactionEvent {
                    seq: entry.seq,
                    ts_ms: entry.ts,
                    kind: "end".to_string(),
                    trigger: None,
                    status: Some(status.clone()),
                    first_kept_seq: *first_kept_seq,
                    tokens_before: *tokens_before,
                    tokens_after: *tokens_after,
                    summary_preview: None,
                });
            }
            _ => {}
        }
    }
    events
}

/// 从 `TrajectoryReport` 与可选 `SessionEntry` 切片构造 `TrajectoryExport`。
pub fn build_trajectory_export(
    mut report: TrajectoryReport,
    session_entries: Option<&[SessionEntry]>,
    meta: TraceExportMeta,
    max_result_chars: usize,
) -> TrajectoryExport {
    let mut total_usage = TraceUsageSummary::default();
    let mut turns = Vec::with_capacity(report.snapshot.turns.len());

    for t in &report.snapshot.turns {
        let mut turn_usage = TraceUsageSummary::default();
        let mut steps = Vec::with_capacity(t.steps.len());
        let mut turn_ttft = None;

        for s in &t.steps {
            if turn_ttft.is_none() {
                turn_ttft = s.ttft_ms;
            }
            if let Some(ref u) = s.usage {
                turn_usage.add_usage(u);
                total_usage.add_usage(u);
            }
            steps.push(TraceStepSummary {
                step: s.step,
                started_at_ms: s.started_at_ms,
                ended_at_ms: s.ended_at_ms,
                duration_ms: s.duration_ms,
                ttft_ms: s.ttft_ms,
                outcome: s.outcome.clone(),
                system_prompt_chars: s.system_prompt_chars,
                tools_count: s.tools_count,
                usage: s.usage,
            });
        }

        turns.push(TraceTurnSummary {
            turn: t.turn,
            started_at_ms: t.started_at_ms,
            ended_at_ms: t.ended_at_ms,
            duration_ms: t.duration_ms,
            ttft_ms: turn_ttft,
            reason: t.reason.clone(),
            usage: turn_usage,
            steps,
        });
    }

    let ttft_ms = turns.iter().find_map(|t| t.ttft_ms);

    if turns.is_empty()
        && (report.snapshot.stats.input_tokens > 0
            || report.snapshot.stats.output_tokens > 0
            || report.snapshot.stats.cache_read_tokens > 0)
    {
        let stats = &report.snapshot.stats;
        total_usage.input_tokens = stats.input_tokens;
        total_usage.output_tokens = stats.output_tokens;
        total_usage.cache_read_tokens = stats.cache_read_tokens;
        total_usage.total_tokens = stats
            .input_tokens
            .saturating_add(stats.output_tokens)
            .saturating_add(stats.cache_read_tokens);
    }

    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for rec in &mut report.snapshot.records {
        if rec.kind != RecordKind::Tool {
            continue;
        }
        let tool_name = rec.tool_name.clone().unwrap_or_else(|| rec.label.clone());
        let call_id = rec.call_id.clone().unwrap_or_default();

        if let Some(ref arguments) = rec.input_detail {
            tool_calls.push(TraceToolCallEntry {
                seq: rec.seq,
                turn: rec.turn,
                step: rec.step,
                call_id: call_id.clone(),
                name: tool_name.clone(),
                arguments: arguments.clone(),
                started_at_ms: rec.started_at_ms,
            });
        }

        if let Some(ref content) = rec.output_detail {
            let (summary, original_chars, truncated) =
                truncate_tool_result_summary(content, max_result_chars);
            // 同步截断内嵌 report 中工具结果明细，避免大输出导致 JSON 体积膨胀
            if truncated {
                rec.output_detail = Some(summary.clone());
            }
            tool_results.push(TraceToolResultSummary {
                seq: rec.seq,
                turn: rec.turn,
                step: rec.step,
                tool_call_id: call_id,
                name: tool_name,
                content_summary: summary,
                original_chars,
                truncated,
                is_error: rec.is_error,
                blocked: rec.blocked,
                duration_ms: rec.duration_ms,
            });
        }
    }

    let compactions = if let Some(entries) = session_entries {
        extract_compaction_events_from_entries(entries, max_result_chars)
    } else {
        report
            .snapshot
            .records
            .iter()
            .filter(|rec| {
                rec.kind == RecordKind::Context && rec.source.as_deref() == Some("compaction")
            })
            .map(|rec| TraceCompactionEvent {
                seq: rec.seq,
                ts_ms: rec.started_at_ms.unwrap_or(0),
                kind: "checkpoint".to_string(),
                trigger: None,
                status: Some("compacted".to_string()),
                first_kept_seq: None,
                tokens_before: None,
                tokens_after: None,
                summary_preview: rec
                    .input_detail
                    .as_deref()
                    .map(|s| preview_chars(s, max_result_chars)),
            })
            .collect()
    };

    TrajectoryExport {
        schema_version: TRAJECTORY_EXPORT_SCHEMA_VERSION,
        exported_at_ms: now_millis(),
        session_id: report.snapshot.session_id.clone(),
        exit_code: meta.exit_code,
        termination_reason: meta.termination_reason,
        error: meta.error,
        duration_ms: meta.duration_ms,
        ttft_ms,
        provider: meta.provider,
        model: meta.model,
        isolated: meta.isolated,
        output_format: meta.output_format,
        usage: total_usage,
        turns,
        tool_calls,
        tool_results,
        compactions,
        report,
    }
}

/// 自动创建父目录并以同目录临时文件 + rename 原子写入 `trajectory.json`。
pub fn write_trace_export_atomic(path: &Path, export: &TrajectoryExport) -> std::io::Result<()> {
    if path.as_os_str().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace output path is empty",
        ));
    }
    if path.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("trace output path is a directory: {}", path.display()),
        ));
    }

    let parent_dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => {
            std::fs::create_dir_all(p)?;
            p
        }
        _ => Path::new("."),
    };

    let mut payload = serde_json::to_vec_pretty(export).map_err(|e| {
        std::io::Error::other(format!("failed to serialize trajectory export: {e}"))
    })?;
    payload.push(b'\n');

    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("trace");
    let seq = ATOMIC_TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_path = parent_dir.join(format!(".{file_name}.tmp.{}.{seq}", std::process::id()));

    let write_res = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        file.write_all(&payload)?;
        file.flush()?;
        let _ = file.sync_all();
        drop(file);
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    })();

    if write_res.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    write_res
}
