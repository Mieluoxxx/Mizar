//! 三泳道时间轴投影（参考 `deepseek-harness/packages/client/ui-trajectory/src/client/timeline.ts`）。
//!
//! - Lane 0：输入 / 上下文提醒 / 告警（`user` / `context` / `anomaly`）
//! - Lane 1：模型响应（`assistant`，可携带 `ttft_offset` 区分首 token 等待与解码阶段）
//! - Lane 2：工具调用（`tool`）

use super::model::{
    RecordKind, TimelineMode, TimelineModel, TimelineSpan, TimelineTurnBoundary, TrajectoryRecord,
};

fn lane_for(kind: RecordKind) -> u8 {
    match kind {
        RecordKind::Tool => 2,
        RecordKind::Assistant => 1,
        RecordKind::User | RecordKind::Context | RecordKind::Anomaly => 0,
    }
}

/// 将账本记录投影为三泳道时间轴模型。
pub fn derive_timeline(records: &[TrajectoryRecord], mode: TimelineMode) -> Option<TimelineModel> {
    if records.is_empty() {
        return None;
    }

    match mode {
        TimelineMode::Sequence => derive_sequence(records),
        TimelineMode::Actual => derive_timed(records, false),
        TimelineMode::Duration => derive_timed(records, true),
    }
}

fn derive_sequence(records: &[TrajectoryRecord]) -> Option<TimelineModel> {
    let mut spans = Vec::with_capacity(records.len());
    let mut turn_boundaries = Vec::new();
    let mut last_turn: Option<u64> = None;

    for (pos, rec) in records.iter().enumerate() {
        if let Some(turn) = rec.turn
            && last_turn != Some(turn)
        {
            last_turn = Some(turn);
            turn_boundaries.push(TimelineTurnBoundary {
                turn,
                pos: pos as u64,
            });
        }
        spans.push(TimelineSpan {
            index: rec.index,
            lane: lane_for(rec.kind),
            kind: rec.kind,
            label: rec.label.clone(),
            start: pos as u64,
            end: (pos + 1) as u64,
            is_error: rec.is_error,
            ttft_offset: None,
        });
    }

    Some(TimelineModel {
        mode: TimelineMode::Sequence,
        start: 0,
        end: spans.len() as u64,
        spans,
        turn_boundaries,
    })
}

fn derive_timed(records: &[TrajectoryRecord], compress_idle: bool) -> Option<TimelineModel> {
    struct RawSpan<'a> {
        rec: &'a TrajectoryRecord,
        start_ms: u128,
        end_ms: u128,
    }

    let raw: Vec<RawSpan<'_>> = records
        .iter()
        .filter_map(|rec| {
            let start_ms = rec.started_at_ms?;
            let end_ms = start_ms.saturating_add(rec.duration_ms.unwrap_or(0) as u128);
            Some(RawSpan {
                rec,
                start_ms,
                end_ms,
            })
        })
        .collect();

    if raw.is_empty() {
        return None;
    }

    let origin = raw.iter().map(|r| r.start_ms).min()?;

    // 压缩空闲模式：如果两个相邻活动区间之间有空隙，把空隙累计扣除（对齐 ui-trajectory `compressIdle`）
    let mut sorted_idx: Vec<usize> = (0..raw.len()).collect();
    sorted_idx.sort_by_key(|&i| (raw[i].start_ms, raw[i].end_ms));

    let mut idle_before = vec![0_u128; raw.len()];
    if compress_idle {
        let mut removed_idle: u128 = 0;
        let mut covered_until: Option<u128> = None;
        for idx in sorted_idx {
            let s = raw[idx].start_ms;
            let e = raw[idx].end_ms;
            if let Some(cu) = covered_until
                && s > cu
            {
                removed_idle += s - cu;
            }
            idle_before[idx] = removed_idle;
            covered_until = Some(covered_until.map_or(e, |cu| cu.max(e)));
        }
    }

    let mut spans = Vec::with_capacity(raw.len());
    let mut turn_boundaries = Vec::new();
    let mut last_turn: Option<u64> = None;
    let mut max_end: u64 = 0;

    for (i, item) in raw.iter().enumerate() {
        let rel_start = (item.start_ms - origin - idle_before[i]) as u64;
        let dur = item.rec.duration_ms.unwrap_or(0);
        let rel_end = rel_start.saturating_add(dur);
        max_end = max_end.max(rel_end);

        if let Some(turn) = item.rec.turn
            && last_turn != Some(turn)
        {
            last_turn = Some(turn);
            turn_boundaries.push(TimelineTurnBoundary {
                turn,
                pos: rel_start,
            });
        }

        let ttft_offset = item.rec.ttft_ms.map(|ttft| ttft.min(dur));

        spans.push(TimelineSpan {
            index: item.rec.index,
            lane: lane_for(item.rec.kind),
            kind: item.rec.kind,
            label: item.rec.label.clone(),
            start: rel_start,
            end: rel_end,
            is_error: item.rec.is_error,
            ttft_offset,
        });
    }

    Some(TimelineModel {
        mode: if compress_idle {
            TimelineMode::Duration
        } else {
            TimelineMode::Actual
        },
        start: 0,
        end: max_end.max(1),
        spans,
        turn_boundaries,
    })
}
