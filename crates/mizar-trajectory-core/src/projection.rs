//! 纯函数轨迹投影与统计折叠（参考 `deepseek-harness/packages/session/session-stats` 与 `ui-trajectory`）。
//!
//! 折叠本身不产生任何副作用，也不依赖在线运行时对象，可用于实时会话和离线回放。

use std::collections::{BTreeMap, HashMap};

use crate::model::{
    RecordKind, SessionStats, Severity, TraceEvent, TraceEventKind, TrajectoryRecord,
    TrajectorySnapshot, TrajectoryStep, TrajectoryTurn,
};

struct OpenStep {
    turn: u64,
    step: usize,
    started_at_ms: u128,
    first_token_ms: Option<u128>,
    step_vec_idx: usize,
}

struct PendingToolCall {
    rec_idx: usize,
    started_at_ms: u128,
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let single_line: String = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    let count = single_line.chars().count();
    if count <= max_chars {
        single_line
    } else {
        let head: String = single_line.chars().take(max_chars).collect();
        format!("{head}…")
    }
}

/// 将有序 `TraceEvent` 流纯折叠为结构化的 `TrajectorySnapshot`。
pub fn fold_trajectory(session_id: &str, events: &[TraceEvent]) -> TrajectorySnapshot {
    let mut stats = SessionStats::default();
    let mut turns_map: BTreeMap<u64, TrajectoryTurn> = BTreeMap::new();
    let mut records: Vec<TrajectoryRecord> = Vec::new();
    let mut open_step: Option<OpenStep> = None;
    let mut pending_calls: HashMap<String, PendingToolCall> = HashMap::new();
    let mut current_turn: Option<u64> = None;
    let mut current_step: Option<usize> = None;
    let mut last_counted_turn: Option<u64> = None;

    for ev in events {
        match &ev.kind {
            TraceEventKind::TurnStart { turn } => {
                current_turn = Some(*turn);
                current_step = None;
                turns_map.entry(*turn).or_insert_with(|| TrajectoryTurn {
                    turn: *turn,
                    started_at_ms: Some(ev.ts_ms),
                    ended_at_ms: None,
                    duration_ms: None,
                    reason: None,
                    steps: Vec::new(),
                });
            }
            TraceEventKind::TurnEnd { turn, reason } => {
                let entry = turns_map.entry(*turn).or_insert_with(|| TrajectoryTurn {
                    turn: *turn,
                    started_at_ms: None,
                    ended_at_ms: None,
                    duration_ms: None,
                    reason: None,
                    steps: Vec::new(),
                });
                entry.ended_at_ms = Some(ev.ts_ms);
                entry.reason = Some(reason.clone());
                if let Some(start) = entry.started_at_ms {
                    entry.duration_ms = Some(ev.ts_ms.saturating_sub(start) as u64);
                }
                if current_turn == Some(*turn) {
                    current_step = None;
                }
            }
            TraceEventKind::StepStart {
                turn,
                step,
                system_prompt_chars,
                tools_count,
            } => {
                current_turn = Some(*turn);
                current_step = Some(*step);
                let turn_entry = turns_map.entry(*turn).or_insert_with(|| TrajectoryTurn {
                    turn: *turn,
                    started_at_ms: Some(ev.ts_ms),
                    ended_at_ms: None,
                    duration_ms: None,
                    reason: None,
                    steps: Vec::new(),
                });
                let step_vec_idx = turn_entry.steps.len();
                turn_entry.steps.push(TrajectoryStep {
                    step: *step,
                    started_at_ms: Some(ev.ts_ms),
                    ended_at_ms: None,
                    duration_ms: None,
                    ttft_ms: None,
                    outcome: None,
                    system_prompt_chars: *system_prompt_chars,
                    tools_count: *tools_count,
                    usage: None,
                });
                open_step = Some(OpenStep {
                    turn: *turn,
                    step: *step,
                    started_at_ms: ev.ts_ms,
                    first_token_ms: None,
                    step_vec_idx,
                });
            }
            TraceEventKind::FirstToken { turn, step } => {
                if let Some(open) = &mut open_step
                    && open.turn == *turn
                    && open.step == *step
                    && open.first_token_ms.is_none()
                {
                    open.first_token_ms = Some(ev.ts_ms);
                }
            }
            TraceEventKind::StepEnd {
                turn,
                step,
                outcome,
                usage,
            } => {
                stats.steps += 1;
                if last_counted_turn != Some(*turn) {
                    last_counted_turn = Some(*turn);
                    stats.turns += 1;
                }
                if outcome == "cancelled" || outcome.starts_with("error") {
                    stats.interrupted_steps += 1;
                }
                if let Some(turn_entry) = turns_map.get_mut(turn)
                    && let Some(step_entry) = turn_entry.steps.iter_mut().rev().find(|s| s.step == *step)
                {
                    step_entry.ended_at_ms = Some(ev.ts_ms);
                    step_entry.outcome = Some(outcome.clone());
                    if step_entry.usage.is_none() {
                        step_entry.usage = *usage;
                    }
                    if step_entry.duration_ms.is_none()
                        && let Some(start) = step_entry.started_at_ms
                    {
                        step_entry.duration_ms = Some(ev.ts_ms.saturating_sub(start) as u64);
                    }
                }
                if open_step
                    .as_ref()
                    .is_some_and(|o| o.turn == *turn && o.step == *step)
                {
                    open_step = None;
                }
            }
            TraceEventKind::UserMessage { text } => {
                let preview = preview_text(text, 80);
                let index = records.len() + 1;
                records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: current_turn,
                    step: None,
                    kind: RecordKind::User,
                    label: if preview.is_empty() {
                        "用户消息".to_string()
                    } else {
                        preview
                    },
                    source: Some("user".to_string()),
                    input_detail: Some(text.clone()),
                    output_detail: None,
                    reasoning_detail: None,
                    call_id: None,
                    tool_name: None,
                    started_at_ms: Some(ev.ts_ms),
                    duration_ms: Some(0),
                    ttft_ms: None,
                    is_error: false,
                    blocked: false,
                    usage: None,
                });
            }
            TraceEventKind::ContextNotice { source, text } => {
                let preview = preview_text(text, 80);
                let index = records.len() + 1;
                records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: current_turn,
                    step: current_step,
                    kind: RecordKind::Context,
                    label: format!("[{source}] {preview}"),
                    source: Some(source.clone()),
                    input_detail: Some(text.clone()),
                    output_detail: None,
                    reasoning_detail: None,
                    call_id: None,
                    tool_name: None,
                    started_at_ms: Some(ev.ts_ms),
                    duration_ms: Some(0),
                    ttft_ms: None,
                    is_error: false,
                    blocked: false,
                    usage: None,
                });
            }
            TraceEventKind::AssistantMessage {
                text,
                reasoning,
                tool_calls,
                usage,
            } => {
                let mut started_at_ms = Some(ev.ts_ms);
                let mut duration_ms = None;
                let mut ttft_ms = None;

                if let Some(open) = &open_step {
                    started_at_ms = Some(open.started_at_ms);
                    let llm_dur = ev.ts_ms.saturating_sub(open.started_at_ms) as u64;
                    duration_ms = Some(llm_dur);
                    stats.llm_ms += llm_dur;

                    if let Some(ft) = open.first_token_ms {
                        let ttft = ft.saturating_sub(open.started_at_ms) as u64;
                        ttft_ms = Some(ttft);
                        stats.ttft_ms += ttft;
                        stats.ttft_steps += 1;
                        if let Some(u) = usage {
                            let decode = ev.ts_ms.saturating_sub(ft) as u64;
                            stats.decode_ms += decode;
                            stats.decode_tokens += u.output_tokens;
                        }
                    }

                    if let Some(turn_entry) = turns_map.get_mut(&open.turn)
                        && let Some(step_entry) = turn_entry.steps.get_mut(open.step_vec_idx)
                    {
                        step_entry.duration_ms = duration_ms;
                        step_entry.ttft_ms = ttft_ms;
                        step_entry.usage = *usage;
                    }
                }

                if let Some(u) = usage {
                    stats.input_tokens += u.input_tokens;
                    stats.output_tokens += u.output_tokens;
                    stats.cache_read_tokens += u.cache_read_tokens.unwrap_or(0);
                }

                let preview = preview_text(text, 80);
                let label = if !preview.is_empty() {
                    preview
                } else if !tool_calls.is_empty() {
                    let names: Vec<&str> = tool_calls.iter().map(|c| c.name.as_str()).collect();
                    format!("调用工具: {}", names.join(", "))
                } else if !reasoning.is_empty() {
                    format!("[思考] {}", preview_text(reasoning, 60))
                } else {
                    "助手空回复".to_string()
                };

                let index = records.len() + 1;
                records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: open_step.as_ref().map(|o| o.turn).or(current_turn),
                    step: open_step.as_ref().map(|o| o.step).or(current_step),
                    kind: RecordKind::Assistant,
                    label,
                    source: Some("assistant".to_string()),
                    input_detail: None,
                    output_detail: (!text.is_empty()).then(|| text.clone()),
                    reasoning_detail: (!reasoning.is_empty()).then(|| reasoning.clone()),
                    call_id: None,
                    tool_name: None,
                    started_at_ms,
                    duration_ms,
                    ttft_ms,
                    is_error: false,
                    blocked: false,
                    usage: *usage,
                });
            }
            TraceEventKind::ToolCall {
                turn,
                step,
                call_id,
                name,
                arguments,
            } => {
                stats.tool_calls += 1;
                let index = records.len() + 1;
                let rec_idx = records.len();
                let preview = preview_text(arguments, 60);
                let label = if preview.is_empty() || preview == "{}" {
                    name.clone()
                } else {
                    format!("{name} {preview}")
                };
                records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: Some(*turn),
                    step: Some(*step),
                    kind: RecordKind::Tool,
                    label,
                    source: Some("tool".to_string()),
                    input_detail: Some(arguments.clone()),
                    output_detail: None,
                    reasoning_detail: None,
                    call_id: Some(call_id.clone()),
                    tool_name: Some(name.clone()),
                    started_at_ms: Some(ev.ts_ms),
                    // 未落定的调用不编造耗时
                    duration_ms: None,
                    ttft_ms: None,
                    is_error: false,
                    blocked: false,
                    usage: None,
                });
                pending_calls.insert(
                    call_id.clone(),
                    PendingToolCall {
                        rec_idx,
                        started_at_ms: ev.ts_ms,
                    },
                );
            }
            TraceEventKind::ToolResult {
                tool_call_id,
                name,
                content,
                is_error,
                blocked,
            } => {
                if *is_error {
                    stats.tool_errors += 1;
                }
                if *blocked {
                    stats.blocked_calls += 1;
                }
                if let Some(pending) = pending_calls.remove(tool_call_id) {
                    let dur = ev.ts_ms.saturating_sub(pending.started_at_ms) as u64;
                    stats.tool_ms += dur;
                    if let Some(rec) = records.get_mut(pending.rec_idx) {
                        rec.output_detail = Some(content.clone());
                        rec.duration_ms = Some(dur);
                        rec.is_error = *is_error;
                        rec.blocked = *blocked;
                    }
                } else {
                    let index = records.len() + 1;
                    records.push(TrajectoryRecord {
                        index,
                        seq: ev.seq,
                        turn: current_turn,
                        step: current_step,
                        kind: RecordKind::Tool,
                        label: name.clone(),
                        source: Some("tool".to_string()),
                        input_detail: None,
                        output_detail: Some(content.clone()),
                        reasoning_detail: None,
                        call_id: Some(tool_call_id.clone()),
                        tool_name: Some(name.clone()),
                        started_at_ms: Some(ev.ts_ms),
                        duration_ms: None,
                        ttft_ms: None,
                        is_error: *is_error,
                        blocked: *blocked,
                        usage: None,
                    });
                }
            }
            TraceEventKind::Anomaly {
                rule_id,
                severity,
                turn,
                step,
                summary,
                evidence,
            } => {
                let index = records.len() + 1;
                records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: (*turn).or(current_turn),
                    step: (*step).or(current_step),
                    kind: RecordKind::Anomaly,
                    label: format!("[{rule_id}] {summary}"),
                    source: Some(rule_id.clone()),
                    input_detail: None,
                    output_detail: Some(serde_json::to_string_pretty(evidence).unwrap_or_default()),
                    reasoning_detail: None,
                    call_id: None,
                    tool_name: None,
                    started_at_ms: Some(ev.ts_ms),
                    duration_ms: Some(0),
                    ttft_ms: None,
                    is_error: *severity == Severity::Error,
                    blocked: false,
                    usage: None,
                });
            }
        }
    }

    TrajectorySnapshot {
        session_id: session_id.to_string(),
        stats,
        turns: turns_map.into_values().collect(),
        records,
    }
}
