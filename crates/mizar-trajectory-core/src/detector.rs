//! 轨迹异常检测规则引擎 + JSON 规范化（参考 `deepseek-harness/packages/guard/repeat-tool-reminder`）。

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Value, json};

use crate::model::{
    RecordKind, Severity, TraceEvent, TraceEventKind, TrajectoryFinding, TrajectorySnapshot,
};

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// 连续重复相同 `(tool, canonical_args)` 的告警阈值（默认 3）。
    pub repeat_threshold: usize,
    /// 连续工具报错告警阈值（默认 3）。
    pub consecutive_error_threshold: usize,
    /// 单步 LLM 耗时告警阈值（毫秒，默认 60_000）。
    pub slow_step_ms: u64,
    /// 单次工具耗时告警阈值（毫秒，默认 30_000）。
    pub slow_tool_ms: u64,
    /// 首 token 延迟告警阈值（毫秒，默认 15_000）。
    pub slow_ttft_ms: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: 3,
            consecutive_error_threshold: 3,
            slow_step_ms: 60_000,
            slow_tool_ms: 30_000,
            slow_ttft_ms: 15_000,
        }
    }
}

/// 递归对 JSON 对象的 key 字典序重排（对齐 DSH `sortJsonValue`），确保属性顺序不影响重复判定。
pub fn sort_json_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sort_json_value).collect()),
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), sort_json_value(v));
            }
            let mut out = serde_json::Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// 规范化 JSON Value 为确定性紧凑字符串。
pub fn canonicalize_json(value: &Value) -> String {
    serde_json::to_string(&sort_json_value(value)).unwrap_or_else(|_| value.to_string())
}

/// 规范化工具参数字符串（合法 JSON 先深度排序，畸形 JSON 回退到去空白原文）。
pub fn canonicalize_args_str(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "{}".to_string();
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(val) => canonicalize_json(&val),
        Err(_) => serde_json::to_string(trimmed).unwrap_or_else(|_| trimmed.to_string()),
    }
}

/// 带省略计数的前缀截断，用于展示重复参数。
pub fn preview_chars(text: &str, cap: usize) -> String {
    let total = text.chars().count();
    if total <= cap {
        return text.to_string();
    }
    let head: String = text.chars().take(cap).collect();
    format!("{head}… (+{} more chars)", total - cap)
}

/// `*` 通配符匹配（对齐 DSH `wildcardToRegExp`）。
pub fn wildcard_matches(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut rest = text;
    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if idx == 0 {
            let Some(after) = rest.strip_prefix(part) else {
                return false;
            };
            rest = after;
        } else if idx == parts.len() - 1 {
            return rest.ends_with(part);
        } else if let Some(pos) = rest.find(part) {
            rest = &rest[pos + part.len()..];
        } else {
            return false;
        }
    }
    true
}

/// 对一段有序轨迹事件和折叠快照执行全量异常规则检测。
pub fn detect_anomalies(
    events: &[TraceEvent],
    snapshot: &TrajectorySnapshot,
    cfg: &DetectorConfig,
) -> Vec<TrajectoryFinding> {
    let mut findings = Vec::new();

    // R01 重复工具调用链 & R02 连续工具报错 & R03 被拦截后继续重试
    struct RepeatStreak {
        name: String,
        canonical_args: String,
        count: usize,
        turn: u64,
        step: usize,
        last_seq: u64,
    }

    let mut repeat_streak: Option<RepeatStreak> = None;
    let flush_repeat = |streak: &mut Option<RepeatStreak>, out: &mut Vec<TrajectoryFinding>| {
        if let Some(s) = streak.take()
            && s.count >= cfg.repeat_threshold
        {
            out.push(TrajectoryFinding {
                rule_id: "R01_REPEAT_TOOL_CALL".to_string(),
                severity: if s.count >= cfg.repeat_threshold + 2 {
                    Severity::Error
                } else {
                    Severity::Warn
                },
                title: format!("检测到连续重复调用 {}", s.name),
                summary: format!(
                    "工具 {} 使用相同参数连续调用了 {} 次（阈值 {}）",
                    s.name, s.count, cfg.repeat_threshold
                ),
                turn: Some(s.turn),
                step: Some(s.step),
                seq: Some(s.last_seq),
                evidence: json!({
                    "tool": s.name,
                    "count": s.count,
                    "arguments": preview_chars(&s.canonical_args, 240),
                }),
            });
        }
    };

    let mut error_streak: usize = 0;
    let mut error_last_info: Option<(Option<u64>, Option<usize>, u64, String)> = None;
    let flush_errors = |cnt: &mut usize,
                        info: &mut Option<(Option<u64>, Option<usize>, u64, String)>,
                        out: &mut Vec<TrajectoryFinding>| {
        if *cnt >= cfg.consecutive_error_threshold
            && let Some((turn, step, seq, last_tool)) = info.take()
        {
            out.push(TrajectoryFinding {
                rule_id: "R02_CONSECUTIVE_TOOL_ERRORS".to_string(),
                severity: Severity::Error,
                title: "工具连续报错空转".to_string(),
                summary: format!("连续 {} 次工具执行失败（最近一次：{}）", *cnt, last_tool),
                turn,
                step,
                seq: Some(seq),
                evidence: json!({ "consecutiveErrors": *cnt, "lastTool": last_tool }),
            });
        }
        *cnt = 0;
        *info = None;
    };

    let mut call_meta: HashMap<String, (u64, usize, String, String)> = HashMap::new();
    let mut denied_keys: HashSet<(String, String)> = HashSet::new();
    let mut open_steps: HashMap<(u64, usize), u64> = HashMap::new();
    let mut open_tools: HashMap<String, (u64, usize, u64, String)> = HashMap::new();
    let mut current_turn: Option<u64> = None;

    for ev in events {
        match &ev.kind {
            TraceEventKind::TurnStart { turn } => {
                // 如果上一个 turn 还留有未闭合的 step/tool，记为 R05
                if let Some(prev_turn) = current_turn
                    && prev_turn != *turn
                {
                    report_unclosed_for_turn(
                        prev_turn,
                        &mut open_steps,
                        &mut open_tools,
                        &mut findings,
                    );
                }
                current_turn = Some(*turn);
            }
            TraceEventKind::UserMessage { .. } => {
                flush_repeat(&mut repeat_streak, &mut findings);
                flush_errors(&mut error_streak, &mut error_last_info, &mut findings);
                denied_keys.clear();
            }
            TraceEventKind::StepStart { turn, step, .. } => {
                open_steps.insert((*turn, *step), ev.seq);
            }
            TraceEventKind::StepEnd { turn, step, .. } => {
                open_steps.remove(&(*turn, *step));
            }
            TraceEventKind::TurnEnd { turn, reason } => {
                report_unclosed_for_turn(*turn, &mut open_steps, &mut open_tools, &mut findings);
                if reason == "max_steps" {
                    findings.push(TrajectoryFinding {
                        rule_id: "R04_MAX_STEPS_EXHAUSTED".to_string(),
                        severity: Severity::Error,
                        title: "达到最大步数上限".to_string(),
                        summary: format!(
                            "Turn #{turn} 在未得出最终答案前达到 max_steps 上限并停止"
                        ),
                        turn: Some(*turn),
                        step: None,
                        seq: Some(ev.seq),
                        evidence: json!({ "turn": turn, "reason": reason }),
                    });
                }
            }
            TraceEventKind::ToolCall {
                turn,
                step,
                call_id,
                name,
                arguments,
            } => {
                let canon = canonicalize_args_str(arguments);
                call_meta.insert(call_id.clone(), (*turn, *step, name.clone(), canon.clone()));
                open_tools.insert(call_id.clone(), (*turn, *step, ev.seq, name.clone()));

                // R03：如果同工具同参数之前被拦截过仍重试
                if denied_keys.contains(&(name.clone(), canon.clone())) {
                    findings.push(TrajectoryFinding {
                        rule_id: "R03_RETRY_AFTER_DENIED".to_string(),
                        severity: Severity::Warn,
                        title: format!("重复尝试被拦截的工具 {}", name),
                        summary: format!(
                            "工具 {} 在本轮已被拦截后，仍使用相同参数再次发起调用",
                            name
                        ),
                        turn: Some(*turn),
                        step: Some(*step),
                        seq: Some(ev.seq),
                        evidence: json!({
                            "tool": name,
                            "arguments": preview_chars(&canon, 240),
                        }),
                    });
                }

                // R01：连续相同调用统计
                match &mut repeat_streak {
                    Some(s) if s.name == *name && s.canonical_args == canon => {
                        s.count += 1;
                        s.turn = *turn;
                        s.step = *step;
                        s.last_seq = ev.seq;
                    }
                    _ => {
                        flush_repeat(&mut repeat_streak, &mut findings);
                        repeat_streak = Some(RepeatStreak {
                            name: name.clone(),
                            canonical_args: canon,
                            count: 1,
                            turn: *turn,
                            step: *step,
                            last_seq: ev.seq,
                        });
                    }
                }
            }
            TraceEventKind::ToolResult {
                tool_call_id,
                name,
                is_error,
                blocked,
                ..
            } => {
                open_tools.remove(tool_call_id);
                let meta = call_meta.get(tool_call_id);
                let turn = meta.map(|m| m.0).or(current_turn);
                let step = meta.map(|m| m.1);

                if *blocked && let Some((_, _, orig_name, canon)) = meta {
                    denied_keys.insert((orig_name.clone(), canon.clone()));
                }

                if *is_error {
                    error_streak += 1;
                    error_last_info = Some((turn, step, ev.seq, name.clone()));
                } else {
                    flush_errors(&mut error_streak, &mut error_last_info, &mut findings);
                }
            }
            _ => {}
        }
    }

    flush_repeat(&mut repeat_streak, &mut findings);
    flush_errors(&mut error_streak, &mut error_last_info, &mut findings);

    // R06：基于折叠后的耗时记录检测高延迟
    for rec in &snapshot.records {
        match rec.kind {
            RecordKind::Assistant => {
                if let Some(dur) = rec.duration_ms
                    && dur > cfg.slow_step_ms
                {
                    findings.push(TrajectoryFinding {
                        rule_id: "R06_HIGH_LATENCY".to_string(),
                        severity: Severity::Warn,
                        title: "模型单步响应耗时过长".to_string(),
                        summary: format!("模型响应耗时 {} ms，超过阈值 {} ms", dur, cfg.slow_step_ms),
                        turn: rec.turn,
                        step: rec.step,
                        seq: Some(rec.seq),
                        evidence: json!({ "kind": "llm", "durationMs": dur, "thresholdMs": cfg.slow_step_ms }),
                    });
                }
                if let Some(ttft) = rec.ttft_ms
                    && ttft > cfg.slow_ttft_ms
                {
                    findings.push(TrajectoryFinding {
                        rule_id: "R06_HIGH_LATENCY".to_string(),
                        severity: Severity::Warn,
                        title: "首 token 延迟过高".to_string(),
                        summary: format!("TTFT 为 {} ms，超过阈值 {} ms", ttft, cfg.slow_ttft_ms),
                        turn: rec.turn,
                        step: rec.step,
                        seq: Some(rec.seq),
                        evidence: json!({ "kind": "ttft", "ttftMs": ttft, "thresholdMs": cfg.slow_ttft_ms }),
                    });
                }
            }
            RecordKind::Tool => {
                if let Some(dur) = rec.duration_ms
                    && dur > cfg.slow_tool_ms
                {
                    findings.push(TrajectoryFinding {
                        rule_id: "R06_HIGH_LATENCY".to_string(),
                        severity: Severity::Warn,
                        title: format!(
                            "工具 {} 执行耗时过长",
                            rec.tool_name.as_deref().unwrap_or("unknown")
                        ),
                        summary: format!(
                            "工具执行耗时 {} ms，超过阈值 {} ms",
                            dur, cfg.slow_tool_ms
                        ),
                        turn: rec.turn,
                        step: rec.step,
                        seq: Some(rec.seq),
                        evidence: json!({
                            "kind": "tool",
                            "tool": rec.tool_name,
                            "durationMs": dur,
                            "thresholdMs": cfg.slow_tool_ms,
                        }),
                    });
                }
            }
            _ => {}
        }
    }

    findings
}

fn report_unclosed_for_turn(
    turn: u64,
    open_steps: &mut HashMap<(u64, usize), u64>,
    open_tools: &mut HashMap<String, (u64, usize, u64, String)>,
    findings: &mut Vec<TrajectoryFinding>,
) {
    let step_keys: Vec<(u64, usize)> = open_steps
        .keys()
        .copied()
        .filter(|(t, _)| *t == turn)
        .collect();
    for key in step_keys {
        if let Some(seq) = open_steps.remove(&key) {
            findings.push(TrajectoryFinding {
                rule_id: "R05_UNCLOSED_BOUNDARY".to_string(),
                severity: Severity::Warn,
                title: "存在未闭合的 Step 边界".to_string(),
                summary: format!(
                    "Turn #{} Step #{} 在轮次结束前未记录 step/end",
                    key.0, key.1
                ),
                turn: Some(key.0),
                step: Some(key.1),
                seq: Some(seq),
                evidence: json!({ "kind": "step", "turn": key.0, "step": key.1 }),
            });
        }
    }

    let tool_keys: Vec<String> = open_tools
        .iter()
        .filter(|(_, (t, _, _, _))| *t == turn)
        .map(|(k, _)| k.clone())
        .collect();
    for call_id in tool_keys {
        if let Some((t, s, seq, name)) = open_tools.remove(&call_id) {
            findings.push(TrajectoryFinding {
                rule_id: "R05_UNCLOSED_BOUNDARY".to_string(),
                severity: Severity::Warn,
                title: format!("工具调用 {} 缺少闭合结果", name),
                summary: format!(
                    "Turn #{t} Step #{s} 的工具调用 {name} ({call_id}) 未收到 tool/result"
                ),
                turn: Some(t),
                step: Some(s),
                seq: Some(seq),
                evidence: json!({ "kind": "tool", "callId": call_id, "tool": name }),
            });
        }
    }
}
