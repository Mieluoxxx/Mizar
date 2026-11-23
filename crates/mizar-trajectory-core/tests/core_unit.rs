use mizar_llm_types::TokenUsage;
use mizar_trajectory_core::{
    DetectorConfig, TimelineMode, TraceEvent, TraceEventKind, derive_timeline, detect_anomalies,
    fold_trajectory,
};

fn ev(seq: u64, ts_ms: u128, kind: TraceEventKind) -> TraceEvent {
    TraceEvent {
        seq,
        ts_ms,
        session_id: "s1".to_string(),
        kind,
    }
}

#[test]
fn folds_stats_timeline_and_detects_all_rules() {
    let events = vec![
        ev(1, 1000, TraceEventKind::TurnStart { turn: 1 }),
        ev(
            2,
            1000,
            TraceEventKind::UserMessage {
                text: "查一下".into(),
            },
        ),
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

fn latency_trace(ttft_ms: u128, llm_ms: u128, tool_ms: u128) -> Vec<TraceEvent> {
    let base = 1_000;
    vec![
        ev(1, base, TraceEventKind::TurnStart { turn: 1 }),
        ev(
            2,
            base,
            TraceEventKind::UserMessage {
                text: "run cargo test".into(),
            },
        ),
        ev(
            3,
            base,
            TraceEventKind::StepStart {
                turn: 1,
                step: 1,
                system_prompt_chars: 120,
                tools_count: 1,
            },
        ),
        ev(
            4,
            base + ttft_ms,
            TraceEventKind::FirstToken { turn: 1, step: 1 },
        ),
        ev(
            5,
            base + llm_ms,
            TraceEventKind::AssistantMessage {
                text: String::new(),
                reasoning: "thinking".into(),
                tool_calls: Vec::new(),
                usage: None,
            },
        ),
        ev(
            6,
            base + llm_ms,
            TraceEventKind::StepEnd {
                turn: 1,
                step: 1,
                outcome: "tool_calls".into(),
                usage: None,
            },
        ),
        ev(
            7,
            base + llm_ms,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"cargo test"}"#.into(),
            },
        ),
        ev(
            8,
            base + llm_ms + tool_ms,
            TraceEventKind::ToolResult {
                tool_call_id: "c1".into(),
                name: "bash".into(),
                content: "ok".into(),
                is_error: false,
                blocked: false,
            },
        ),
        ev(
            9,
            base + llm_ms + tool_ms,
            TraceEventKind::TurnEnd {
                turn: 1,
                reason: "stop".into(),
            },
        ),
    ]
}

#[test]
fn r06_high_latency_respects_relaxed_default_thresholds() {
    let cfg = DetectorConfig::default();

    // 旧阈值之上、新阈值之内的正常耗时不触发 R06_HIGH_LATENCY
    let normal_events = latency_trace(14_000, 50_000, 25_000);
    let normal_snap = fold_trajectory("s1", &normal_events);
    assert_eq!(normal_snap.stats.ttft_ms, 14_000);
    assert_eq!(normal_snap.stats.llm_ms, 50_000);
    assert_eq!(normal_snap.stats.tool_ms, 25_000);
    let normal_findings = detect_anomalies(&normal_events, &normal_snap, &cfg);
    assert!(
        !normal_findings
            .iter()
            .any(|f| f.rule_id == "R06_HIGH_LATENCY"),
        "unexpected R06 findings: {normal_findings:?}"
    );

    // 刚好等于新阈值也不触发（判定条件为严格大于）
    let boundary_events = latency_trace(15_000, 60_000, 30_000);
    let boundary_snap = fold_trajectory("s1", &boundary_events);
    let boundary_findings = detect_anomalies(&boundary_events, &boundary_snap, &cfg);
    assert!(
        !boundary_findings
            .iter()
            .any(|f| f.rule_id == "R06_HIGH_LATENCY"),
        "unexpected R06 findings at boundary: {boundary_findings:?}"
    );

    // 超过新阈值时触发对应 3 条 R06_HIGH_LATENCY 告警
    let slow_events = latency_trace(16_000, 65_000, 35_000);
    let slow_snap = fold_trajectory("s1", &slow_events);
    assert_eq!(slow_snap.stats.ttft_ms, 16_000);
    assert_eq!(slow_snap.stats.llm_ms, 65_000);
    assert_eq!(slow_snap.stats.tool_ms, 35_000);
    let slow_findings = detect_anomalies(&slow_events, &slow_snap, &cfg);
    let r06: Vec<_> = slow_findings
        .iter()
        .filter(|f| f.rule_id == "R06_HIGH_LATENCY")
        .collect();
    assert_eq!(r06.len(), 3, "r06={r06:?}");

    let llm = r06
        .iter()
        .find(|f| f.evidence["kind"] == "llm")
        .expect("missing llm R06 finding");
    assert_eq!(llm.evidence["durationMs"], 65_000);
    assert_eq!(llm.evidence["thresholdMs"], 60_000);

    let ttft = r06
        .iter()
        .find(|f| f.evidence["kind"] == "ttft")
        .expect("missing ttft R06 finding");
    assert_eq!(ttft.evidence["ttftMs"], 16_000);
    assert_eq!(ttft.evidence["thresholdMs"], 15_000);

    let tool = r06
        .iter()
        .find(|f| f.evidence["kind"] == "tool")
        .expect("missing tool R06 finding");
    assert_eq!(tool.evidence["tool"], "bash");
    assert_eq!(tool.evidence["durationMs"], 35_000);
    assert_eq!(tool.evidence["thresholdMs"], 30_000);
}

#[test]
fn fold_trajectory_counts_max_tokens_as_interrupted_step() {
    let events = vec![
        ev(1, 1000, TraceEventKind::TurnStart { turn: 1 }),
        ev(
            2,
            1000,
            TraceEventKind::UserMessage {
                text: "write a long essay".into(),
            },
        ),
        ev(
            3,
            1010,
            TraceEventKind::StepStart {
                turn: 1,
                step: 1,
                system_prompt_chars: 100,
                tools_count: 0,
            },
        ),
        ev(4, 1050, TraceEventKind::FirstToken { turn: 1, step: 1 }),
        ev(
            5,
            1500,
            TraceEventKind::AssistantMessage {
                text: "half answer".into(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    input_tokens: 50,
                    output_tokens: 100,
                    ..Default::default()
                }),
            },
        ),
        ev(
            6,
            1500,
            TraceEventKind::StepEnd {
                turn: 1,
                step: 1,
                outcome: "max_tokens".into(),
                usage: Some(TokenUsage {
                    input_tokens: 50,
                    output_tokens: 100,
                    ..Default::default()
                }),
            },
        ),
        ev(
            7,
            1501,
            TraceEventKind::TurnEnd {
                turn: 1,
                reason: "truncated".into(),
            },
        ),
    ];

    let snap = fold_trajectory("s-trunc", &events);
    assert_eq!(snap.stats.turns, 1);
    assert_eq!(snap.stats.steps, 1);
    assert_eq!(snap.stats.interrupted_steps, 1);
    assert_eq!(snap.turns.len(), 1);
    assert_eq!(snap.turns[0].reason.as_deref(), Some("truncated"));
    assert_eq!(
        snap.turns[0].steps[0].outcome.as_deref(),
        Some("max_tokens")
    );
}

#[test]
fn incremental_folder_and_detector_match_batch_parity() {
    let events = vec![
        ev(1, 1000, TraceEventKind::TurnStart { turn: 1 }),
        ev(
            2,
            1000,
            TraceEventKind::UserMessage {
                text: "check files".into(),
            },
        ),
        ev(
            3,
            1010,
            TraceEventKind::StepStart {
                turn: 1,
                step: 1,
                system_prompt_chars: 100,
                tools_count: 1,
            },
        ),
        ev(4, 1040, TraceEventKind::FirstToken { turn: 1, step: 1 }),
        ev(
            5,
            1200,
            TraceEventKind::AssistantMessage {
                text: "checking".into(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                usage: Some(TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                }),
            },
        ),
        ev(
            6,
            1205,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c1".into(),
                name: "read".into(),
                arguments: r#"{"b":2,"a":1}"#.into(),
            },
        ),
        ev(
            7,
            1250,
            TraceEventKind::ToolResult {
                tool_call_id: "c1".into(),
                name: "read".into(),
                content: "err1".into(),
                is_error: true,
                blocked: true,
            },
        ),
        ev(
            8,
            1255,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c2".into(),
                name: "read".into(),
                arguments: r#"{"a":1,"b":2}"#.into(),
            },
        ),
        ev(
            9,
            1300,
            TraceEventKind::ToolResult {
                tool_call_id: "c2".into(),
                name: "read".into(),
                content: "err2".into(),
                is_error: true,
                blocked: false,
            },
        ),
        ev(
            10,
            1305,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c3".into(),
                name: "read".into(),
                arguments: r#"{"a":1,"b":2}"#.into(),
            },
        ),
        ev(
            11,
            1350,
            TraceEventKind::ToolResult {
                tool_call_id: "c3".into(),
                name: "read".into(),
                content: "err3".into(),
                is_error: true,
                blocked: false,
            },
        ),
        ev(
            12,
            1360,
            TraceEventKind::TurnEnd {
                turn: 1,
                reason: "max_steps".into(),
            },
        ),
    ];

    let cfg = DetectorConfig {
        repeat_threshold: 3,
        consecutive_error_threshold: 3,
        slow_step_ms: 100,
        slow_tool_ms: 40,
        slow_ttft_ms: 25,
    };

    let mut folder = mizar_trajectory_core::IncrementalFolder::new("s-inc");
    let mut detector = mizar_trajectory_core::IncrementalDetector::new(cfg.clone());
    for (idx, e) in events.iter().enumerate() {
        folder.push_event(e);
        detector.push_event(e, folder.records());

        let inc_snap = folder.snapshot();
        let batch_snap = fold_trajectory("s-inc", &events[..=idx]);
        assert_eq!(inc_snap.stats, batch_snap.stats, "at idx={idx}");
        assert_eq!(inc_snap.turns, batch_snap.turns, "at idx={idx}");
        assert_eq!(inc_snap.records, batch_snap.records, "at idx={idx}");

        let inc_findings = detector.findings(folder.records());
        let batch_findings = detect_anomalies(&events[..=idx], &batch_snap, &cfg);
        assert_eq!(inc_findings, batch_findings, "at idx={idx}");
    }
}

#[test]
fn incremental_folder_caps_detail_previews_at_512_utf8_bytes() {
    let long_cjk = "猫娘工程师浮浮酱测试多字节截断".repeat(40);
    let long_ascii = "x".repeat(2048);
    let mut folder = mizar_trajectory_core::IncrementalFolder::new("s-cap");
    let events = vec![
        ev(1, 100, TraceEventKind::TurnStart { turn: 1 }),
        ev(
            2,
            101,
            TraceEventKind::UserMessage {
                text: long_cjk.clone(),
            },
        ),
        ev(
            3,
            102,
            TraceEventKind::AssistantMessage {
                text: long_cjk.clone(),
                reasoning: long_cjk.clone(),
                tool_calls: Vec::new(),
                usage: None,
            },
        ),
        ev(
            4,
            103,
            TraceEventKind::ToolCall {
                turn: 1,
                step: 1,
                call_id: "c1".into(),
                name: "read".into(),
                arguments: long_ascii.clone(),
            },
        ),
        ev(
            5,
            104,
            TraceEventKind::ToolResult {
                tool_call_id: "c1".into(),
                name: "read".into(),
                content: long_cjk.clone(),
                is_error: false,
                blocked: false,
            },
        ),
    ];
    for e in &events {
        folder.push_event(e);
    }

    let snap = folder.into_snapshot();
    assert_eq!(snap.records.len(), 3);
    for rec in &snap.records {
        for detail in [
            rec.input_detail.as_deref(),
            rec.output_detail.as_deref(),
            rec.reasoning_detail.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            assert!(
                detail.len() <= mizar_trajectory_core::projection::DEFAULT_DETAIL_MAX_BYTES,
                "detail len {} exceeded 512",
                detail.len()
            );
            assert!(detail.ends_with('…'));
        }
    }
}
