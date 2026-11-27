use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::Context;
use mizar_agent_spine::{CompactionEndEvent, CompactionOutcome, EVENT_COMPACTION_END};
use mizar_session::{
    EVENT_SESSION_EVENT, Session, SessionEntry, SessionEvent, SessionEventPayload,
    SessionTraceEvent,
};
use mizar_trajectory::TraceUsageSummary;
use serde_json::{Value, json};

use super::{is_secret_key, sanitize_diagnostic_text};

pub(super) fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(super) fn write_event(mut payload: Value, ts: u128) -> io::Result<()> {
    payload["ts_ms"] = json!(ts);
    let mut out = io::stdout().lock();
    writeln!(out, "{payload}")?;
    out.flush()
}

fn arguments_preview(raw: &str, secrets: &[String]) -> String {
    fn redact(value: &mut Value, secrets: &[String]) {
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    if is_secret_key(key) {
                        *value = "[REDACTED]".into();
                    } else {
                        redact(value, secrets);
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    redact(item, secrets);
                }
            }
            Value::String(text) => *text = sanitize_diagnostic_text(text, secrets),
            _ => {}
        }
    }
    // 先解码转义的字段和值，避免引号或换行掩盖凭据。
    let clean = match serde_json::from_str(raw) {
        Ok(mut value) => {
            redact(&mut value, secrets);
            sanitize_diagnostic_text(&value.to_string(), secrets)
        }
        Err(_) => sanitize_diagnostic_text(raw, secrets),
    };
    clean.chars().take(512).collect()
}

#[derive(Default)]
pub(super) struct ExecStream {
    step: usize,
    step_start: u128,
    step_ttft: Option<u128>,
    llm_ms: u128,
    ttft_ms: Option<u128>,
    tool_starts: HashMap<String, u128>,
    tool_durations: HashMap<String, u64>,
    tool_ms: u128,
    tool_calls: usize,
    tool_errors: usize,
    retries: usize,
    compactions: usize,
    usage: TraceUsageSummary,
    pub(super) write_error: Option<io::Error>,
}

impl ExecStream {
    pub(super) fn emit(&mut self, payload: Value, ts: u128) {
        if self.write_error.is_none() {
            self.write_error = write_event(payload, ts).err();
        }
    }

    fn on_entry(&mut self, entry: &SessionEntry, secrets: &[String]) {
        let ts = entry.ts;
        let detail = match &entry.event {
            SessionEvent::Trace { detail } => detail,
            SessionEvent::ToolResult {
                tool_call_id,
                name,
                is_error,
                images,
                ..
            } => {
                let start = self.tool_starts.remove(tool_call_id);
                let duration = self
                    .tool_durations
                    .remove(tool_call_id)
                    .map(u128::from)
                    .or_else(|| start.map(|s| ts.saturating_sub(s)));
                self.tool_ms += duration.unwrap_or(0);
                self.tool_errors += usize::from(*is_error);
                self.emit(json!({"type": "tool_result", "step": self.step, "call_id": tool_call_id,
                    "name": name, "is_error": is_error, "duration_ms": duration, "images": images.len()}), ts);
                return;
            }
            _ => return,
        };
        let event = match detail {
            SessionTraceEvent::StepStart { step, .. } => {
                self.step = *step;
                self.step_start = ts;
                self.step_ttft = None;
                json!({"type": "step_start", "step": step})
            }
            SessionTraceEvent::FirstToken { .. } => {
                self.step_ttft
                    .get_or_insert(ts.saturating_sub(self.step_start));
                return;
            }
            SessionTraceEvent::StepRetry {
                step,
                attempt,
                max_attempts,
                delay_ms,
                reason,
                reset_draft,
                ..
            } => {
                self.retries += 1;
                if *reset_draft {
                    self.step_ttft = None;
                }
                json!({"type": "retry", "step": step, "attempt": attempt, "max_attempts": max_attempts,
                    "delay_ms": delay_ms, "reason": sanitize_diagnostic_text(reason, secrets), "reset_draft": reset_draft})
            }
            SessionTraceEvent::StepEnd { step, usage, .. } => {
                let duration = ts.saturating_sub(self.step_start);
                self.llm_ms += duration;
                self.ttft_ms = self.ttft_ms.or(self.step_ttft);
                let mut step_usage = TraceUsageSummary::default();
                if let Some(usage) = usage {
                    step_usage.add_usage(usage);
                    self.usage.add_usage(usage);
                }
                json!({"type": "step_end", "step": step, "duration_ms": duration, "ttft_ms": self.step_ttft, "usage": step_usage})
            }
            SessionTraceEvent::ToolCall {
                step,
                call_id,
                name,
                arguments,
                ..
            } => {
                self.tool_calls += 1;
                self.tool_starts.insert(call_id.clone(), ts);
                let preview = arguments_preview(arguments, secrets);
                json!({"type": "tool_call", "step": step, "call_id": call_id, "name": name, "arguments_preview": preview})
            }
            SessionTraceEvent::ToolOutcome {
                tool_call_id,
                duration_ms,
                ..
            } => {
                if let Some(ms) = duration_ms {
                    self.tool_durations.insert(tool_call_id.clone(), *ms);
                }
                return;
            }
            _ => return,
        };
        self.emit(event, ts);
    }

    pub(super) fn result_fields(
        &self,
        session: Option<&Session>,
        resumed: bool,
        duration_ms: u64,
    ) -> Value {
        let (turns, steps, session_usage) = session
            .map(|s| s.with_all_entries(session_totals))
            .unwrap_or_default();
        json!({
            "session_id": session.map(|s| s.id.as_str()), "resumed": resumed, "steps": self.step, "duration_ms": duration_ms,
            "llm_ms": self.llm_ms, "ttft_ms": self.ttft_ms, "tool_ms": self.tool_ms, "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors, "retries": self.retries, "compactions": self.compactions,
            "session_turns": turns, "session_steps": steps, "usage": self.usage, "session_usage": session_usage,
        })
    }
}

fn session_totals(entries: &[SessionEntry]) -> (usize, usize, TraceUsageSummary) {
    let mut usage = TraceUsageSummary::default();
    let (mut turns, mut steps) = (0, 0);
    let mut last_turn = None;
    let mut open_step = false;
    let mut counted_assistant_usage = false;
    let mut new_user = false;
    for entry in entries {
        match &entry.event {
            SessionEvent::User { .. } => new_user = true,
            SessionEvent::Trace {
                detail: SessionTraceEvent::StepStart { .. },
            } => {
                open_step = true;
                counted_assistant_usage = false;
            }
            SessionEvent::Assistant {
                usage: measured, ..
            } => {
                // 旧会话或被强杀的步骤可能只在助手消息上记录用量。
                if let Some(u) = measured {
                    usage.add_usage(u);
                    counted_assistant_usage = true;
                }
                if !open_step {
                    turns += usize::from(new_user);
                    new_user = false;
                    steps += 1;
                }
            }
            SessionEvent::Trace {
                detail:
                    SessionTraceEvent::StepEnd {
                        turn,
                        usage: measured,
                        ..
                    },
            } => {
                turns += usize::from(last_turn != Some(*turn));
                last_turn = Some(*turn);
                steps += 1;
                open_step = false;
                new_user = false;
                if !counted_assistant_usage && let Some(u) = measured {
                    usage.add_usage(u);
                }
                counted_assistant_usage = false;
            }
            _ => {}
        }
    }
    (turns, steps, usage)
}

pub(super) fn subscribe(
    app: &Context,
    session_id: &str,
    stream: &Arc<Mutex<ExecStream>>,
    secrets: &Arc<Mutex<Vec<String>>>,
) {
    let events = Arc::clone(stream);
    let id = session_id.to_owned();
    let secrets = Arc::clone(secrets);
    app.on(EVENT_SESSION_EVENT, move |ev: &SessionEventPayload| {
        if ev.session.id == id {
            events
                .lock()
                .unwrap()
                .on_entry(&ev.entry, &secrets.lock().unwrap());
        }
    })
    .expect("active exec context");

    let events = Arc::clone(stream);
    let id = session_id.to_owned();
    app.on(EVENT_COMPACTION_END, move |ev: &CompactionEndEvent| {
        if ev.session_id == id {
            let mut stream = events.lock().unwrap();
            stream.compactions += usize::from(ev.outcome == CompactionOutcome::Compacted);
            stream.emit(
                json!({"type": "compaction", "trigger": ev.trigger, "outcome": ev.outcome,
                "tokens_before": ev.tokens_before, "tokens_after": ev.tokens_after}),
                now_ms(),
            );
        }
    })
    .expect("active exec context");
}
