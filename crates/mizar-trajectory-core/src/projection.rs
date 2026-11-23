//! 纯函数轨迹投影与统计折叠（参考 `deepseek-harness/packages/session/session-stats` 与 `ui-trajectory`）。
//!
//! 折叠本身不产生任何副作用，也不依赖在线运行时对象，可用于实时会话和离线回放。

use std::collections::{BTreeMap, HashMap};

use crate::model::{
    RecordKind, SessionStats, Severity, TraceEvent, TraceEventKind, TrajectoryRecord,
    TrajectorySnapshot, TrajectoryStep, TrajectoryTurn,
};

#[derive(Clone)]
struct OpenStep {
    turn: u64,
    step: usize,
    started_at_ms: u128,
    first_token_ms: Option<u128>,
    step_vec_idx: usize,
}

#[derive(Clone)]
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

/// 常驻 `IncrementalFolder` 保留各条记录 `*detail` 预览的默认字节上限（UTF-8 边界安全）。
pub const DEFAULT_DETAIL_MAX_BYTES: usize = 512;

/// 按 UTF-8 字符边界安全截断文本，确保返回值总字节数不超过 `max_bytes`。
pub fn cap_detail_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    const ELLIPSIS: &str = "…";
    if max_bytes < ELLIPSIS.len() {
        let cut = text.floor_char_boundary(max_bytes);
        return text[..cut].to_string();
    }
    let body_budget = max_bytes - ELLIPSIS.len();
    let cut = text.floor_char_boundary(body_budget);
    let mut out = String::with_capacity(cut + ELLIPSIS.len());
    out.push_str(&text[..cut]);
    out.push_str(ELLIPSIS);
    out
}

/// 增量轨迹折叠器：每次仅消费新增 `TraceEvent`，避免在长会话评估时重复重扫全量事件历史。
#[derive(Clone)]
pub struct IncrementalFolder {
    session_id: String,
    stats: SessionStats,
    turns_map: BTreeMap<u64, TrajectoryTurn>,
    records: Vec<TrajectoryRecord>,
    open_step: Option<OpenStep>,
    pending_calls: HashMap<String, PendingToolCall>,
    current_turn: Option<u64>,
    current_step: Option<usize>,
    last_counted_turn: Option<u64>,
    detail_max_bytes: Option<usize>,
}

impl IncrementalFolder {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_detail_cap(session_id, Some(DEFAULT_DETAIL_MAX_BYTES))
    }

    /// 构造指定 `*detail` 字节上限的增量折叠器（`None` 表示保留完整原文，供离线无损折叠使用）。
    pub fn with_detail_cap(session_id: impl Into<String>, detail_max_bytes: Option<usize>) -> Self {
        Self {
            session_id: session_id.into(),
            stats: SessionStats::default(),
            turns_map: BTreeMap::new(),
            records: Vec::new(),
            open_step: None,
            pending_calls: HashMap::new(),
            current_turn: None,
            current_step: None,
            last_counted_turn: None,
            detail_max_bytes,
        }
    }

    fn cap_detail(&self, text: &str) -> String {
        match self.detail_max_bytes {
            Some(max_bytes) => cap_detail_bytes(text, max_bytes),
            None => text.to_string(),
        }
    }

    pub fn records(&self) -> &[TrajectoryRecord] {
        &self.records
    }

    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }

    pub fn push_event(&mut self, ev: &TraceEvent) {
        match &ev.kind {
            TraceEventKind::TurnStart { turn } => {
                self.current_turn = Some(*turn);
                self.current_step = None;
                self.turns_map
                    .entry(*turn)
                    .or_insert_with(|| TrajectoryTurn {
                        turn: *turn,
                        started_at_ms: Some(ev.ts_ms),
                        ended_at_ms: None,
                        duration_ms: None,
                        reason: None,
                        steps: Vec::new(),
                    });
            }
            TraceEventKind::TurnEnd { turn, reason } => {
                let entry = self
                    .turns_map
                    .entry(*turn)
                    .or_insert_with(|| TrajectoryTurn {
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
                if self.current_turn == Some(*turn) {
                    self.current_step = None;
                }
            }
            TraceEventKind::StepStart {
                turn,
                step,
                system_prompt_chars,
                tools_count,
            } => {
                self.current_turn = Some(*turn);
                self.current_step = Some(*step);
                let turn_entry = self
                    .turns_map
                    .entry(*turn)
                    .or_insert_with(|| TrajectoryTurn {
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
                self.open_step = Some(OpenStep {
                    turn: *turn,
                    step: *step,
                    started_at_ms: ev.ts_ms,
                    first_token_ms: None,
                    step_vec_idx,
                });
            }
            TraceEventKind::FirstToken { turn, step } => {
                if let Some(open) = &mut self.open_step
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
                self.stats.steps += 1;
                if self.last_counted_turn != Some(*turn) {
                    self.last_counted_turn = Some(*turn);
                    self.stats.turns += 1;
                }
                if outcome == "cancelled" || outcome == "max_tokens" || outcome.starts_with("error")
                {
                    self.stats.interrupted_steps += 1;
                }
                if let Some(turn_entry) = self.turns_map.get_mut(turn)
                    && let Some(step_entry) =
                        turn_entry.steps.iter_mut().rev().find(|s| s.step == *step)
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
                if self
                    .open_step
                    .as_ref()
                    .is_some_and(|o| o.turn == *turn && o.step == *step)
                {
                    self.open_step = None;
                }
            }
            TraceEventKind::UserMessage { text } => {
                let preview = preview_text(text, 80);
                let index = self.records.len() + 1;
                self.records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: self.current_turn,
                    step: None,
                    kind: RecordKind::User,
                    label: if preview.is_empty() {
                        "用户消息".to_string()
                    } else {
                        preview
                    },
                    source: Some("user".to_string()),
                    input_detail: Some(self.cap_detail(text)),
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
                let index = self.records.len() + 1;
                self.records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: self.current_turn,
                    step: self.current_step,
                    kind: RecordKind::Context,
                    label: format!("[{source}] {preview}"),
                    source: Some(source.clone()),
                    input_detail: Some(self.cap_detail(text)),
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

                if let Some(open) = &self.open_step {
                    started_at_ms = Some(open.started_at_ms);
                    let llm_dur = ev.ts_ms.saturating_sub(open.started_at_ms) as u64;
                    duration_ms = Some(llm_dur);
                    self.stats.llm_ms += llm_dur;

                    if let Some(ft) = open.first_token_ms {
                        let ttft = ft.saturating_sub(open.started_at_ms) as u64;
                        ttft_ms = Some(ttft);
                        self.stats.ttft_ms += ttft;
                        self.stats.ttft_steps += 1;
                        if let Some(u) = usage {
                            let decode = ev.ts_ms.saturating_sub(ft) as u64;
                            self.stats.decode_ms += decode;
                            self.stats.decode_tokens += u.output_tokens;
                        }
                    }

                    if let Some(turn_entry) = self.turns_map.get_mut(&open.turn)
                        && let Some(step_entry) = turn_entry.steps.get_mut(open.step_vec_idx)
                    {
                        step_entry.duration_ms = duration_ms;
                        step_entry.ttft_ms = ttft_ms;
                        step_entry.usage = *usage;
                    }
                }

                if let Some(u) = usage {
                    self.stats.input_tokens += u.input_tokens;
                    self.stats.output_tokens += u.output_tokens;
                    self.stats.cache_read_tokens += u.cache_read_tokens.unwrap_or(0);
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

                let index = self.records.len() + 1;
                self.records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: self
                        .open_step
                        .as_ref()
                        .map(|o| o.turn)
                        .or(self.current_turn),
                    step: self
                        .open_step
                        .as_ref()
                        .map(|o| o.step)
                        .or(self.current_step),
                    kind: RecordKind::Assistant,
                    label,
                    source: Some("assistant".to_string()),
                    input_detail: None,
                    output_detail: (!text.is_empty()).then(|| self.cap_detail(text)),
                    reasoning_detail: (!reasoning.is_empty()).then(|| self.cap_detail(reasoning)),
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
                self.stats.tool_calls += 1;
                let index = self.records.len() + 1;
                let rec_idx = self.records.len();
                let preview = preview_text(arguments, 60);
                let label = if preview.is_empty() || preview == "{}" {
                    name.clone()
                } else {
                    format!("{name} {preview}")
                };
                self.records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: Some(*turn),
                    step: Some(*step),
                    kind: RecordKind::Tool,
                    label,
                    source: Some("tool".to_string()),
                    input_detail: Some(self.cap_detail(arguments)),
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
                self.pending_calls.insert(
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
                    self.stats.tool_errors += 1;
                }
                if *blocked {
                    self.stats.blocked_calls += 1;
                }
                if let Some(pending) = self.pending_calls.remove(tool_call_id) {
                    let dur = ev.ts_ms.saturating_sub(pending.started_at_ms) as u64;
                    self.stats.tool_ms += dur;
                    let capped_output = self.cap_detail(content);
                    if let Some(rec) = self.records.get_mut(pending.rec_idx) {
                        rec.output_detail = Some(capped_output);
                        rec.duration_ms = Some(dur);
                        rec.is_error = *is_error;
                        rec.blocked = *blocked;
                    }
                } else {
                    let index = self.records.len() + 1;
                    self.records.push(TrajectoryRecord {
                        index,
                        seq: ev.seq,
                        turn: self.current_turn,
                        step: self.current_step,
                        kind: RecordKind::Tool,
                        label: name.clone(),
                        source: Some("tool".to_string()),
                        input_detail: None,
                        output_detail: Some(self.cap_detail(content)),
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
                let index = self.records.len() + 1;
                self.records.push(TrajectoryRecord {
                    index,
                    seq: ev.seq,
                    turn: (*turn).or(self.current_turn),
                    step: (*step).or(self.current_step),
                    kind: RecordKind::Anomaly,
                    label: format!("[{rule_id}] {summary}"),
                    source: Some(rule_id.clone()),
                    input_detail: None,
                    output_detail: Some(
                        self.cap_detail(
                            &serde_json::to_string_pretty(evidence).unwrap_or_default(),
                        ),
                    ),
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

    pub fn snapshot(&self) -> TrajectorySnapshot {
        TrajectorySnapshot {
            session_id: self.session_id.clone(),
            stats: self.stats.clone(),
            turns: self.turns_map.values().cloned().collect(),
            records: self.records.clone(),
        }
    }

    pub fn into_snapshot(self) -> TrajectorySnapshot {
        TrajectorySnapshot {
            session_id: self.session_id,
            stats: self.stats,
            turns: self.turns_map.into_values().collect(),
            records: self.records,
        }
    }
}

/// 将有序 `TraceEvent` 流纯折叠为结构化的 `TrajectorySnapshot`。
pub fn fold_trajectory(session_id: &str, events: &[TraceEvent]) -> TrajectorySnapshot {
    let mut folder = IncrementalFolder::with_detail_cap(session_id, None);
    for ev in events {
        folder.push_event(ev);
    }
    folder.into_snapshot()
}
