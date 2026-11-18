use mizar_llm_types::TokenUsage;
use mizar_trajectory_core::{
    DetectorConfig, TimelineMode, TraceEvent, TraceEventKind, canonicalize_json, detect_anomalies,
    derive_timeline, fold_trajectory, wildcard_matches,
};
use serde_json::json;

fn ev(seq: u64, ts_ms: u128, kind: TraceEventKind) -> TraceEvent {
    TraceEvent {
        seq,
        ts_ms,
        session_id: "s1".to_string(),
        kind,
    }
}

#[test]
fn canonicalize_json_is_key_order_independent() {
    let a = json!({ "b": 2, "a": { "y": 1, "x": [ { "d": 4, "c": 3 } ] } });
    let b = json!({ "a": { "x": [ { "c": 3, "d": 4 } ], "y": 1 }, "b": 2 });
    assert_eq!(canonicalize_json(&a), canonicalize_json(&b));
}

#[test]
fn wildcard_matching_works() {
    assert!(wildcard_matches("mcp_*", "mcp_search"));
    assert!(!wildcard_matches("mcp_*", "search"));
    assert!(wildcard_matches("*_file", "read_file"));
    assert!(wildcard_matches("todo_write", "todo_write"));
    assert!(!wildcard_matches("todo_write", "todo_read"));
}

#[test]
fn folds_stats_timeline_and_detects_all_rules() {
    let events = vec![
        ev(1, 1000, TraceEventKind::TurnStart { turn: 1 }),
        ev(2, 1000, TraceEventKind::UserMessage { text: "查一下".into() }),
        ev(
            3,
            1010,
            TraceEventKind::StepStart {
                turn: 1,
                step: 1,
                system_prompt_chars: 120,
                tools_count: 2,
            },
        ),
        ev(4, 1060, TraceEventKind::FirstToken { turn: 1, step: 1 }),
        ev(
            5,
            1210,
            TraceEventKind::AssistantMessage {
                text: String::new(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    input_tokens: 100,
                    output_tokens: 20,
                    ..Default::default()
                }),
            },
        ),
        ev(
            6,
            1210,
            TraceEventKind::StepEnd {
                turn: 1,
                step: 1,
                outcome: "tool_calls".into(),
                usage: None,
            },
        ),
        // 连续 3 次同参数工具调用，且均失败（其中第 1 次被拦截）-> 触发 R01 + R02 + R03
        ev(
            7,
            1220,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c1".into(),
                name: "read_file".into(),
                arguments: r#"{"b":2,"a":1}"#.into(),
            },
        ),
        ev(
            8,
            1250,
            TraceEventKind::ToolResult {
                tool_call_id: "c1".into(),
                name: "read_file".into(),
                content: "denied".into(),
                is_error: true,
                blocked: true,
            },
        ),
        ev(
            9,
            1260,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c2".into(),
                name: "read_file".into(),
                arguments: r#"{"a":1,"b":2}"#.into(),
            },
        ),
        ev(
            10,
            1300,
            TraceEventKind::ToolResult {
                tool_call_id: "c2".into(),
                name: "read_file".into(),
                content: "err2".into(),
                is_error: true,
                blocked: false,
            },
        ),
        ev(
            11,
            1310,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c3".into(),
                name: "read_file".into(),
                arguments: r#"{"a":1,"b":2}"#.into(),
            },
        ),
        ev(
            12,
            1350,
            TraceEventKind::ToolResult {
                tool_call_id: "c3".into(),
                name: "read_file".into(),
                content: "err3".into(),
                is_error: true,
                blocked: false,
            },
        ),
        // 未闭合 Step 2 + 达到 max_steps -> 触发 R04 + R05
        ev(
            13,
            1360,
            TraceEventKind::StepStart {
                turn: 1,
                step: 2,
                system_prompt_chars: 120,
                tools_count: 2,
            },
        ),
        ev(
            14,
            1400,
            TraceEventKind::TurnEnd {
                turn: 1,
                reason: "max_steps".into(),
            },
        ),
    ];

    let snap = fold_trajectory("s1", &events);
    assert_eq!(snap.stats.turns, 1);
    assert_eq!(snap.stats.steps, 1);
    assert_eq!(snap.stats.llm_ms, 200);
    assert_eq!(snap.stats.ttft_ms, 50);
    assert_eq!(snap.stats.decode_ms, 150);
    assert_eq!(snap.stats.decode_tokens, 20);
    assert_eq!(snap.stats.tool_calls, 3);
    assert_eq!(snap.stats.tool_errors, 3);
    assert_eq!(snap.stats.blocked_calls, 1);
    assert_eq!(snap.stats.tool_ms, 30 + 40 + 40);

    let tl_seq = derive_timeline(&snap.records, TimelineMode::Sequence).unwrap();
    assert_eq!(tl_seq.spans.len(), snap.records.len());
    let tl_actual = derive_timeline(&snap.records, TimelineMode::Actual).unwrap();
    assert!(tl_actual.end >= 350);
    let tl_dur = derive_timeline(&snap.records, TimelineMode::Duration).unwrap();
    assert!(tl_dur.end <= tl_actual.end);

    let cfg = DetectorConfig {
        repeat_threshold: 3,
        consecutive_error_threshold: 3,
        slow_step_ms: 100, // 故意设低以覆盖 R06
        slow_tool_ms: 35,
        slow_ttft_ms: 40,
    };
    let findings = detect_anomalies(&events, &snap, &cfg);
    let ids: Vec<&str> = findings.iter().map(|f| f.rule_id.as_str()).collect();
    assert!(ids.contains(&"R01_REPEAT_TOOL_CALL"), "ids={ids:?}");
    assert!(ids.contains(&"R02_CONSECUTIVE_TOOL_ERRORS"), "ids={ids:?}");
    assert!(ids.contains(&"R03_RETRY_AFTER_DENIED"), "ids={ids:?}");
    assert!(ids.contains(&"R04_MAX_STEPS_EXHAUSTED"), "ids={ids:?}");
    assert!(ids.contains(&"R05_UNCLOSED_BOUNDARY"), "ids={ids:?}");
    assert!(ids.contains(&"R06_HIGH_LATENCY"), "ids={ids:?}");
}
