//! 轨迹采集服务 `ctx.trajectory`（`TrajectoryRuntime`）。
//!
//! 对已接入 `SessionStore` 的会话直接以 `Session` 作为唯一权威事实源（支持从 SQLite
//! 持久化历史重建完整轨迹报告与导出，不再在内存中双份缓存大文本载荷）；
//! 当未挂载 `SessionStore`（如独立的工具级单元测试）时，仍保留轻量内存缓冲作为兜底。

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{BoxFuture, Context, CordisError, Service};
use mizar_agent_loop::{
    AgentStepEndEvent, AgentStepFirstTokenEvent, AgentStepStartEvent, AgentToolCallEvent,
    AgentTurnEndEvent, AgentTurnStartEvent, EVENT_AGENT_STEP_END, EVENT_AGENT_STEP_FIRST_TOKEN,
    EVENT_AGENT_STEP_START, EVENT_AGENT_TOOL_CALL, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_START,
};
use mizar_session::{
    EVENT_SESSION_CREATED, EVENT_SESSION_EVENT, Session, SessionCreatedPayload, SessionEntry,
    SessionEvent, SessionEventPayload, SessionListQuery, SessionPersistenceContextExt,
    SessionStore, SessionTraceEvent, TraceSeverity,
};
use mizar_tools::{EVENT_TOOLS_RESULT, ToolResultEvent};
use serde_json::json;

use mizar_trajectory_core::{
    DetectorConfig, Severity, TimelineMode, TraceEvent, TraceEventKind, TrajectoryFinding,
    TrajectoryReport, TrajectorySummary, derive_timeline, detect_anomalies, fold_trajectory,
    preview_chars,
};

use crate::guard::{EVENT_GUARD_REPEAT, GuardRepeatEvent};

pub const EVENT_TRAJECTORY_EVENT: &str = "trajectory/event";
pub const EVENT_TRAJECTORY_ANOMALY: &str = "trajectory/anomaly";

#[derive(Clone)]
pub struct TrajectoryEventEmitted {
    pub session_id: String,
    pub event: TraceEvent,
}

#[derive(Clone)]
pub struct TrajectoryAnomalyEvent {
    pub session_id: String,
    pub finding: TrajectoryFinding,
}

#[derive(Debug, Clone, Default)]
pub struct TrajectoryConfig {
    pub detector: DetectorConfig,
    /// 可选：将每条轨迹事件实时追加写入 `{export_dir}/{session_id}.jsonl`。
    pub export_dir: Option<PathBuf>,
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn map_from_session_severity(s: TraceSeverity) -> Severity {
    match s {
        TraceSeverity::Info => Severity::Info,
        TraceSeverity::Warn => Severity::Warn,
        TraceSeverity::Error => Severity::Error,
    }
}

fn map_to_session_severity(s: Severity) -> TraceSeverity {
    match s {
        Severity::Info => TraceSeverity::Info,
        Severity::Warn => TraceSeverity::Warn,
        Severity::Error => TraceSeverity::Error,
    }
}

fn entry_to_trace_event(
    session_id: &str,
    entry: &SessionEntry,
    blocked_ids: &mut HashSet<String>,
) -> Option<TraceEvent> {
    let kind = match &entry.event {
        SessionEvent::User { text, .. } => TraceEventKind::UserMessage { text: text.clone() },
        SessionEvent::ContextNotice { source, text, .. } => TraceEventKind::ContextNotice {
            source: source.clone(),
            text: text.clone(),
        },
        SessionEvent::Compaction {
            summary,
            first_kept_seq,
            ..
        } => TraceEventKind::ContextNotice {
            source: "compaction".to_string(),
            text: format!("[上下文压缩（保留自 #{first_kept_seq}）] {summary}"),
        },
        SessionEvent::Assistant { blocks, usage, .. } => {
            let text: String = blocks
                .iter()
                .filter_map(|b| b.as_text())
                .collect::<Vec<_>>()
                .join("");
            let reasoning: String = blocks
                .iter()
                .filter_map(|b| b.as_reasoning())
                .collect::<Vec<_>>()
                .join("");
            let tool_calls = blocks.iter().filter_map(|b| b.to_tool_call()).collect();
            TraceEventKind::AssistantMessage {
                text,
                reasoning,
                tool_calls,
                usage: *usage,
            }
        }
        SessionEvent::ToolResult {
            tool_call_id,
            name,
            content,
            is_error,
        } => TraceEventKind::ToolResult {
            tool_call_id: tool_call_id.clone(),
            name: name.clone(),
            content: content.clone(),
            is_error: *is_error,
            blocked: blocked_ids.remove(tool_call_id),
        },
        SessionEvent::BashExecution {
            command,
            output,
            exit_code,
            cancelled,
            exclude_from_context,
        } => {
            let source = if *exclude_from_context {
                "bash-local"
            } else {
                "bash-context"
            };
            let mut text = format!("! {command}\n");
            if *cancelled {
                text.push_str("[cancelled]\n");
            } else if let Some(code) = exit_code
                && *code != 0
            {
                text.push_str(&format!("[exit code: {code}]\n"));
            }
            if !output.is_empty() {
                text.push_str(output);
            }
            TraceEventKind::ContextNotice {
                source: source.to_string(),
                text,
            }
        }
        SessionEvent::ToolResultPrune { .. } | SessionEvent::Selection { .. } => return None,
        SessionEvent::Trace { detail } => match detail {
            SessionTraceEvent::TurnStart { turn } => TraceEventKind::TurnStart { turn: *turn },
            SessionTraceEvent::TurnEnd { turn, reason } => TraceEventKind::TurnEnd {
                turn: *turn,
                reason: reason.clone(),
            },
            SessionTraceEvent::StepStart {
                turn,
                step,
                system_prompt_chars,
                tools_count,
            } => TraceEventKind::StepStart {
                turn: *turn,
                step: *step,
                system_prompt_chars: *system_prompt_chars,
                tools_count: *tools_count,
            },
            SessionTraceEvent::FirstToken { turn, step } => TraceEventKind::FirstToken {
                turn: *turn,
                step: *step,
            },
            SessionTraceEvent::StepEnd {
                turn,
                step,
                outcome,
                usage,
            } => TraceEventKind::StepEnd {
                turn: *turn,
                step: *step,
                outcome: outcome.clone(),
                usage: *usage,
            },
            SessionTraceEvent::ToolCall {
                turn,
                step,
                call_id,
                name,
                arguments,
            } => TraceEventKind::ToolCall {
                turn: *turn,
                step: *step,
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            },
            SessionTraceEvent::ToolOutcome {
                tool_call_id,
                blocked,
            } => {
                if *blocked {
                    blocked_ids.insert(tool_call_id.clone());
                }
                return None;
            }
            SessionTraceEvent::Anomaly {
                rule_id,
                severity,
                summary,
                turn,
                step,
                evidence,
                ..
            } => TraceEventKind::Anomaly {
                rule_id: rule_id.clone(),
                severity: map_from_session_severity(*severity),
                turn: *turn,
                step: *step,
                summary: summary.clone(),
                evidence: evidence.clone(),
            },
            SessionTraceEvent::CompactionStart { .. }
            | SessionTraceEvent::CompactionEnd { .. }
            | SessionTraceEvent::TurnNotice { .. } => {
                return None;
            }
        },
    };

    Some(TraceEvent {
        seq: entry.seq,
        ts_ms: entry.ts,
        session_id: session_id.to_string(),
        kind,
    })
}

/// 将任意 `SessionEntry` 切片映射为 `Vec<TraceEvent>`（单源派生，不额外复制底层存储）。
pub fn trace_events_from_entries(session_id: &str, entries: &[SessionEntry]) -> Vec<TraceEvent> {
    let mut blocked_ids = HashSet::new();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(ev) = entry_to_trace_event(session_id, entry, &mut blocked_ids) {
            out.push(ev);
        }
    }
    out
}

/// 从 `Session` 的完整只追加日志派生轨迹事件流。
pub fn trace_events_from_session(session: &Session) -> Vec<TraceEvent> {
    session.with_all_entries(|entries| trace_events_from_entries(&session.id, entries))
}

#[derive(Default)]
struct SessionTraceBuffer {
    /// 仅在没有对应 `SessionStore` 管理会话时（独立工具/事件测试）使用的内存事件缓冲。
    standalone_events: Vec<TraceEvent>,
    current_turn: u64,
    current_step: usize,
    pending_tool_calls: VecDeque<(String, String)>,
    blocked_tool_calls: HashSet<String>,
    emitted_finding_keys: HashSet<(String, Option<u64>)>,
}

pub struct TrajectoryRuntime {
    ctx: Context,
    config: TrajectoryConfig,
    buffers: Mutex<HashMap<String, SessionTraceBuffer>>,
}

impl TrajectoryRuntime {
    fn session_in_memory(&self, session_id: &str) -> bool {
        self.ctx
            .get::<SessionStore>("sessions")
            .is_some_and(|s| s.contains_in_memory(session_id))
    }

    fn resolve_session(&self, session_id: &str) -> Option<Arc<Session>> {
        self.ctx
            .get::<SessionStore>("sessions")
            .and_then(|s| s.get(session_id))
    }

    fn write_export_line(&self, session_id: &str, event: &TraceEvent) {
        if let Some(dir) = &self.config.export_dir {
            let _ = std::fs::create_dir_all(dir);
            let file_path = dir.join(format!("{session_id}.jsonl"));
            if let Ok(line) = serde_json::to_string(event) {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file_path)
                {
                    let _ = writeln!(file, "{line}");
                }
            }
        }
    }

    fn emit_trace_event(&self, session_id: &str, event: TraceEvent) {
        self.write_export_line(session_id, &event);
        self.ctx.emit(
            EVENT_TRAJECTORY_EVENT,
            TrajectoryEventEmitted {
                session_id: session_id.to_string(),
                event,
            },
        );
    }

    fn record_standalone_event(
        &self,
        session_id: &str,
        ts_ms: u128,
        kind: TraceEventKind,
    ) -> TraceEvent {
        let event = {
            let mut buffers = self.buffers.lock().unwrap();
            let buf = buffers.entry(session_id.to_string()).or_default();
            let seq = (buf.standalone_events.len() as u64) + 1;
            let event = TraceEvent {
                seq,
                ts_ms,
                session_id: session_id.to_string(),
                kind,
            };
            buf.standalone_events.push(event.clone());
            event
        };

        self.emit_trace_event(session_id, event.clone());
        event
    }

    fn evaluate_and_emit_new_findings(&self, session_id: &str) {
        let Some(report) = self.inspect(session_id, TimelineMode::Sequence) else {
            return;
        };
        let new_findings: Vec<TrajectoryFinding> = {
            let mut buffers = self.buffers.lock().unwrap();
            let buf = buffers.entry(session_id.to_string()).or_default();
            let mut fresh = Vec::new();
            for finding in report.findings {
                let key = (finding.rule_id.clone(), finding.seq);
                if buf.emitted_finding_keys.insert(key) {
                    fresh.push(finding);
                }
            }
            fresh
        };
        for finding in new_findings {
            self.ctx.emit(
                EVENT_TRAJECTORY_ANOMALY,
                TrajectoryAnomalyEvent {
                    session_id: session_id.to_string(),
                    finding,
                },
            );
        }
    }

    pub fn events(&self, session_id: &str) -> Vec<TraceEvent> {
        if let Some(session) = self.resolve_session(session_id) {
            return trace_events_from_session(&session);
        }
        self.buffers
            .lock()
            .unwrap()
            .get(session_id)
            .map(|b| b.standalone_events.clone())
            .unwrap_or_default()
    }

    /// 对指定会话（内存激活或 SQLite 持久化历史）做一次完整轨迹折叠、时间轴投影与异常检测。
    pub fn inspect(&self, session_id: &str, mode: TimelineMode) -> Option<TrajectoryReport> {
        if let Some(session) = self.resolve_session(session_id) {
            let events = trace_events_from_session(&session);
            return Some(self.inspect_events(session_id, &events, mode));
        }
        let events = {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(session_id).map(|b| b.standalone_events.clone())
        }?;
        Some(self.inspect_events(session_id, &events, mode))
    }

    /// 纯离线分析：对任意传入的 `TraceEvent` 切片运行折叠 + 时间轴 + 异常检测。
    pub fn inspect_events(
        &self,
        session_id: &str,
        events: &[TraceEvent],
        mode: TimelineMode,
    ) -> TrajectoryReport {
        let snapshot = fold_trajectory(session_id, events);
        let timeline = derive_timeline(&snapshot.records, mode);
        let mut findings = detect_anomalies(events, &snapshot, &self.config.detector);

        // 将在线守卫直接记入轨迹的 Anomaly 事件并入报告（避免与 R01 重复输出同一 seq）
        let existing_r01_seqs: HashSet<Option<u64>> = findings
            .iter()
            .filter(|f| f.rule_id == "R01_REPEAT_TOOL_CALL")
            .map(|f| f.seq)
            .collect();
        for ev in events {
            if let TraceEventKind::Anomaly {
                rule_id,
                severity,
                turn,
                step,
                summary,
                evidence,
            } = &ev.kind
            {
                if rule_id == "GUARD_REPEAT_TOOL" && existing_r01_seqs.contains(&Some(ev.seq)) {
                    continue;
                }
                findings.push(TrajectoryFinding {
                    rule_id: rule_id.clone(),
                    severity: *severity,
                    title: format!("守卫告警 ({rule_id})"),
                    summary: summary.clone(),
                    turn: *turn,
                    step: *step,
                    seq: Some(ev.seq),
                    evidence: evidence.clone(),
                });
            }
        }

        TrajectoryReport {
            snapshot,
            timeline,
            findings,
        }
    }

    pub fn list_sessions(&self) -> Vec<TrajectorySummary> {
        let mut summaries = Vec::new();
        let mut seen = HashSet::new();

        // 1. 优先汇总内存活跃 Session
        if let Some(store) = self.ctx.get::<SessionStore>("sessions") {
            for s in store.list() {
                if !seen.insert(s.id.clone()) {
                    continue;
                }
                let events = trace_events_from_session(&s);
                let report = self.inspect_events(&s.id, &events, TimelineMode::Sequence);
                let has_errors = report
                    .findings
                    .iter()
                    .any(|f| f.severity == Severity::Error);
                summaries.push(TrajectorySummary {
                    session_id: s.id.clone(),
                    event_count: events.len(),
                    turns: report.snapshot.stats.turns,
                    steps: report.snapshot.stats.steps,
                    tool_calls: report.snapshot.stats.tool_calls,
                    findings_count: report.findings.len(),
                    has_errors,
                });
            }
        }

        // 2. 补充仅在持久化数据库中的历史会话（不把冷会话全部灌入内存缓存）
        if let Some(backend) = self.ctx.session_persistence()
            && let Ok(headers) = backend.list_sessions(&SessionListQuery {
                cwd: None,
                limit: Some(50),
            })
        {
            for h in headers {
                if !seen.insert(h.id.clone()) {
                    continue;
                }
                if let Ok(Some(loaded)) = backend.load_session(&h.id) {
                    let events = trace_events_from_entries(&h.id, &loaded.entries);
                    let report = self.inspect_events(&h.id, &events, TimelineMode::Sequence);
                    let has_errors = report
                        .findings
                        .iter()
                        .any(|f| f.severity == Severity::Error);
                    summaries.push(TrajectorySummary {
                        session_id: h.id,
                        event_count: events.len(),
                        turns: report.snapshot.stats.turns,
                        steps: report.snapshot.stats.steps,
                        tool_calls: report.snapshot.stats.tool_calls,
                        findings_count: report.findings.len(),
                        has_errors,
                    });
                }
            }
        }

        // 3. 补充独立事件模式缓冲
        let standalone_ids: Vec<(String, Vec<TraceEvent>)> = {
            let buffers = self.buffers.lock().unwrap();
            buffers
                .iter()
                .filter(|(id, b)| !seen.contains(*id) && !b.standalone_events.is_empty())
                .map(|(id, b)| (id.clone(), b.standalone_events.clone()))
                .collect()
        };
        for (id, events) in standalone_ids {
            let report = self.inspect_events(&id, &events, TimelineMode::Sequence);
            let has_errors = report
                .findings
                .iter()
                .any(|f| f.severity == Severity::Error);
            summaries.push(TrajectorySummary {
                session_id: id,
                event_count: events.len(),
                turns: report.snapshot.stats.turns,
                steps: report.snapshot.stats.steps,
                tool_calls: report.snapshot.stats.tool_calls,
                findings_count: report.findings.len(),
                has_errors,
            });
        }

        summaries.sort_by(|a, b| a.session_id.cmp(&b.session_id));
        summaries
    }

    /// 导出某会话的 JSONL 格式轨迹文本。
    pub fn export_jsonl(&self, session_id: &str) -> Option<String> {
        let events = if let Some(session) = self.resolve_session(session_id) {
            trace_events_from_session(&session)
        } else {
            let buffers = self.buffers.lock().unwrap();
            buffers.get(session_id)?.standalone_events.clone()
        };
        let mut out = String::new();
        for ev in events {
            if let Ok(line) = serde_json::to_string(&ev) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        Some(out)
    }

    /// 解析 JSONL 文本为 `Vec<TraceEvent>`，供离线检验使用。
    pub fn parse_jsonl(input: &str) -> Result<Vec<TraceEvent>, String> {
        let mut events = Vec::new();
        for (idx, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let ev: TraceEvent = serde_json::from_str(line)
                .map_err(|e| format!("invalid JSONL at line {}: {e}", idx + 1))?;
            events.push(ev);
        }
        Ok(events)
    }
}

impl Service for TrajectoryRuntime {
    type Config = TrajectoryConfig;

    fn service_name() -> &'static str {
        "trajectory"
    }

    fn plugin_name() -> &'static str {
        "TrajectoryRuntime"
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config,
            buffers: Mutex::new(HashMap::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let rt = Arc::clone(&self);
        let ctx = ctx.clone();
        Box::pin(async move {
            let rt_created = Arc::clone(&rt);
            ctx.on(EVENT_SESSION_CREATED, move |ev: &SessionCreatedPayload| {
                let mut buffers = rt_created.buffers.lock().unwrap();
                buffers.entry(ev.session.id.clone()).or_default();
            })?;

            let rt_turn_start = Arc::clone(&rt);
            ctx.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                {
                    let mut buffers = rt_turn_start.buffers.lock().unwrap();
                    let buf = buffers.entry(ev.session.id.clone()).or_default();
                    buf.current_turn = ev.turn;
                    buf.current_step = 0;
                }
                if !rt_turn_start.session_in_memory(&ev.session.id) {
                    rt_turn_start.record_standalone_event(
                        &ev.session.id,
                        ev.ts_ms,
                        TraceEventKind::TurnStart { turn: ev.turn },
                    );
                }
            })?;

            let rt_turn_end = Arc::clone(&rt);
            ctx.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
                if !rt_turn_end.session_in_memory(&ev.session.id) {
                    rt_turn_end.record_standalone_event(
                        &ev.session.id,
                        ev.ts_ms,
                        TraceEventKind::TurnEnd {
                            turn: ev.turn,
                            reason: ev.reason.as_str().to_string(),
                        },
                    );
                }
                rt_turn_end.evaluate_and_emit_new_findings(&ev.session.id);
            })?;

            let rt_step_start = Arc::clone(&rt);
            ctx.on(EVENT_AGENT_STEP_START, move |ev: &AgentStepStartEvent| {
                {
                    let mut buffers = rt_step_start.buffers.lock().unwrap();
                    let buf = buffers.entry(ev.session.id.clone()).or_default();
                    buf.current_turn = ev.turn;
                    buf.current_step = ev.step;
                }
                if !rt_step_start.session_in_memory(&ev.session.id) {
                    rt_step_start.record_standalone_event(
                        &ev.session.id,
                        ev.ts_ms,
                        TraceEventKind::StepStart {
                            turn: ev.turn,
                            step: ev.step,
                            system_prompt_chars: ev.system_prompt_chars,
                            tools_count: ev.tools_count,
                        },
                    );
                }
            })?;

            let rt_ft = Arc::clone(&rt);
            ctx.on(
                EVENT_AGENT_STEP_FIRST_TOKEN,
                move |ev: &AgentStepFirstTokenEvent| {
                    if !rt_ft.session_in_memory(&ev.session.id) {
                        rt_ft.record_standalone_event(
                            &ev.session.id,
                            ev.ts_ms,
                            TraceEventKind::FirstToken {
                                turn: ev.turn,
                                step: ev.step,
                            },
                        );
                    }
                },
            )?;

            let rt_step_end = Arc::clone(&rt);
            ctx.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
                if !rt_step_end.session_in_memory(&ev.session.id) {
                    rt_step_end.record_standalone_event(
                        &ev.session.id,
                        ev.ts_ms,
                        TraceEventKind::StepEnd {
                            turn: ev.turn,
                            step: ev.step,
                            outcome: ev.outcome.as_str().to_string(),
                            usage: ev.usage,
                        },
                    );
                }
                rt_step_end.evaluate_and_emit_new_findings(&ev.session.id);
            })?;

            let rt_tool_call = Arc::clone(&rt);
            ctx.on(EVENT_AGENT_TOOL_CALL, move |ev: &AgentToolCallEvent| {
                let managed = rt_tool_call.session_in_memory(&ev.session.id);
                let (turn, step) = {
                    let mut buffers = rt_tool_call.buffers.lock().unwrap();
                    let buf = buffers.entry(ev.session.id.clone()).or_default();
                    if !managed {
                        buf.pending_tool_calls
                            .push_back((ev.call.id.clone(), ev.call.name.clone()));
                    }
                    (buf.current_turn, buf.current_step)
                };
                if !managed {
                    rt_tool_call.record_standalone_event(
                        &ev.session.id,
                        now_millis(),
                        TraceEventKind::ToolCall {
                            turn,
                            step,
                            call_id: ev.call.id.clone(),
                            name: ev.call.name.clone(),
                            arguments: ev.call.arguments.clone(),
                        },
                    );
                }
            })?;

            let rt_tool_result = Arc::clone(&rt);
            ctx.on(EVENT_TOOLS_RESULT, move |ev: &ToolResultEvent| {
                let Some(session_id) = ev.exec.session_id.as_deref() else {
                    return;
                };
                if rt_tool_result.session_in_memory(session_id) {
                    return;
                }
                let (call_id, missing_call_meta) = {
                    let mut buffers = rt_tool_result.buffers.lock().unwrap();
                    let buf = buffers.entry(session_id.to_string()).or_default();
                    let matched_pos = buf
                        .pending_tool_calls
                        .iter()
                        .position(|(_, name)| name == &ev.exec.name);
                    let (id, missing) = match matched_pos {
                        Some(pos) => (
                            buf.pending_tool_calls
                                .remove(pos)
                                .map(|(id, _)| id)
                                .unwrap(),
                            None,
                        ),
                        None => (
                            format!("direct_{}", buf.standalone_events.len() + 1),
                            Some((buf.current_turn, buf.current_step)),
                        ),
                    };
                    (id, missing)
                };
                if let Some((turn, step)) = missing_call_meta {
                    rt_tool_result.record_standalone_event(
                        session_id,
                        now_millis(),
                        TraceEventKind::ToolCall {
                            turn,
                            step,
                            call_id: call_id.clone(),
                            name: ev.exec.name.clone(),
                            arguments: ev.exec.args.to_string(),
                        },
                    );
                }
                rt_tool_result.record_standalone_event(
                    session_id,
                    now_millis(),
                    TraceEventKind::ToolResult {
                        tool_call_id: call_id,
                        name: ev.exec.name.clone(),
                        content: ev.result.content.clone(),
                        is_error: ev.result.is_error,
                        blocked: ev.result.blocked,
                    },
                );
                rt_tool_result.evaluate_and_emit_new_findings(session_id);
            })?;

            let rt_session = Arc::clone(&rt);
            ctx.on(EVENT_SESSION_EVENT, move |ev: &SessionEventPayload| {
                let sid = &ev.session.id;
                let trace_event = {
                    let mut buffers = rt_session.buffers.lock().unwrap();
                    let buf = buffers.entry(sid.clone()).or_default();
                    entry_to_trace_event(sid, &ev.entry, &mut buf.blocked_tool_calls)
                };
                if let Some(trace_event) = trace_event {
                    let should_eval = matches!(
                        trace_event.kind,
                        TraceEventKind::ToolResult { .. }
                            | TraceEventKind::StepEnd { .. }
                            | TraceEventKind::TurnEnd { .. }
                    );
                    rt_session.emit_trace_event(sid, trace_event);
                    if should_eval {
                        rt_session.evaluate_and_emit_new_findings(sid);
                    }
                }
            })?;

            let rt_guard = Arc::clone(&rt);
            ctx.on(EVENT_GUARD_REPEAT, move |ev: &GuardRepeatEvent| {
                let (turn, step) = {
                    let buffers = rt_guard.buffers.lock().unwrap();
                    buffers
                        .get(&ev.session_id)
                        .map(|b| (b.current_turn, b.current_step))
                        .unwrap_or((0, 0))
                };
                let severity = if ev.blocked || ev.count >= 5 {
                    Severity::Error
                } else {
                    Severity::Warn
                };
                let title = format!("重复调用守卫触发 ({})", ev.tool_name);
                let summary = format!(
                    "守卫检测到工具 {} 连续重复调用 {} 次{}",
                    ev.tool_name,
                    ev.count,
                    if ev.blocked {
                        "（已强制拦截）"
                    } else {
                        "（已向模型注入提醒）"
                    }
                );
                let evidence = json!({
                    "tool": ev.tool_name,
                    "count": ev.count,
                    "blocked": ev.blocked,
                    "arguments": preview_chars(&ev.canonical_args, 240),
                });

                let seq = if let Some(session) = rt_guard.resolve_session(&ev.session_id) {
                    session.append_trace(SessionTraceEvent::Anomaly {
                        rule_id: "GUARD_REPEAT_TOOL".to_string(),
                        severity: map_to_session_severity(severity),
                        title: title.clone(),
                        summary: summary.clone(),
                        turn: (turn > 0).then_some(turn),
                        step: (step > 0).then_some(step),
                        evidence: evidence.clone(),
                    });
                    session.with_all_entries(|entries| entries.last().map(|e| e.seq))
                } else {
                    let recorded = rt_guard.record_standalone_event(
                        &ev.session_id,
                        now_millis(),
                        TraceEventKind::Anomaly {
                            rule_id: "GUARD_REPEAT_TOOL".to_string(),
                            severity,
                            turn: (turn > 0).then_some(turn),
                            step: (step > 0).then_some(step),
                            summary: summary.clone(),
                            evidence: evidence.clone(),
                        },
                    );
                    Some(recorded.seq)
                };

                if let Some(seq_val) = seq {
                    rt_guard
                        .buffers
                        .lock()
                        .unwrap()
                        .entry(ev.session_id.clone())
                        .or_default()
                        .emitted_finding_keys
                        .insert(("GUARD_REPEAT_TOOL".to_string(), Some(seq_val)));
                }

                rt_guard.ctx.emit(
                    EVENT_TRAJECTORY_ANOMALY,
                    TrajectoryAnomalyEvent {
                        session_id: ev.session_id.clone(),
                        finding: TrajectoryFinding {
                            rule_id: "GUARD_REPEAT_TOOL".to_string(),
                            severity,
                            title,
                            summary,
                            turn: (turn > 0).then_some(turn),
                            step: (step > 0).then_some(step),
                            seq,
                            evidence,
                        },
                    },
                );
            })?;

            Ok(())
        })
    }
}

pub trait TrajectoryContextExt {
    fn trajectory(&self) -> Result<Arc<TrajectoryRuntime>, CordisError>;
}

impl TrajectoryContextExt for Context {
    fn trajectory(&self) -> Result<Arc<TrajectoryRuntime>, CordisError> {
        self.service::<TrajectoryRuntime>("trajectory")
    }
}
