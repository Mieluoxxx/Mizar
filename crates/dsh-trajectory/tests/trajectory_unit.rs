use cordis::{BoxFuture, Context, CordisError, Next, Service};
use dsh_llm::TokenUsage;
use dsh_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolDefinition, ToolExecution, ToolRuntime,
    ToolsContextExt,
};
use dsh_trajectory::{
    DetectorConfig, RepeatGuardConfig, RepeatToolGuardPlugin, Severity, TimelineMode, TraceEvent,
    TraceEventKind, TrajectoryConfig, TrajectoryRuntime, canonicalize_json, detect_anomalies,
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

#[tokio::test]
async fn repeat_guard_validates_config_fail_loud() {
    let app = Context::new();
    let bad_empty = app.plugin(
        RepeatToolGuardPlugin,
        RepeatGuardConfig {
            thresholds: vec![],
            ..Default::default()
        },
    );
    assert!(bad_empty.await.is_err());

    let bad_lt2 = app.plugin(
        RepeatToolGuardPlugin,
        RepeatGuardConfig {
            thresholds: vec![1, 3],
            ..Default::default()
        },
    );
    assert!(bad_lt2.await.is_err());

    let bad_dup = app.plugin(
        RepeatToolGuardPlugin,
        RepeatGuardConfig {
            thresholds: vec![3, 3],
            ..Default::default()
        },
    );
    assert!(bad_dup.await.is_err());
}

#[tokio::test]
async fn repeat_guard_counts_through_excluded_tools_and_denied_calls() {
    let app = Context::new();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(
        TrajectoryRuntime::plugin(),
        TrajectoryConfig::default(),
    )
    .await
    .unwrap();
    app.plugin(
        RepeatToolGuardPlugin,
        RepeatGuardConfig {
            thresholds: vec![2, 3],
            exclude: vec!["todo_*".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();

    // 拦截第 2 次 grep，验证被拦截的调用仍计入重复链
    app.plugin(
        cordis::PluginDef::sync("deny-second", |ctx, ()| {
            let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            ctx.on_waterfall(
                EVENT_TOOLS_PRE_EXECUTE,
                move |exec: ToolExecution,
                      next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
                    let calls = std::sync::Arc::clone(&calls);
                    Box::pin(async move {
                        if exec.name == "grep" && calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                            PreExecuteDecision::deny("test deny")
                        } else {
                            next.call(exec).await
                        }
                    })
                },
            )?;
            Ok::<_, CordisError>(())
        }),
        (),
    )
    .await
    .unwrap();

    let tools = app.tools().unwrap();
    app.effect("reg.tools", || {
        let c1 = tools
            .register(ToolDefinition::sync_tool(
                "grep",
                "search",
                json!({}),
                |_, _| Ok::<_, cordis::BoxError>("ok".into()),
            ))
            .unwrap();
        let c2 = tools
            .register(ToolDefinition::sync_tool(
                "todo_write",
                "note",
                json!({}),
                |_, _| Ok::<_, cordis::BoxError>("noted".into()),
            ))
            .unwrap();
        Ok::<_, CordisError>(Some(cordis::Cleanup::sync(move || {
            let _ = (c1, c2);
        })))
    })
    .unwrap();

    // 1. grep {"b":2,"a":1}
    let r1 = tools
        .execute(ToolExecution {
            name: "grep".into(),
            args: json!({"b": 2, "a": 1}),
            session_id: Some("sess-x".into()),
        })
        .await;
    assert!(r1.additional_contexts.is_empty());

    // 2. todo_write (在 exclude 里，应当透明，不打断链条)
    let r_todo = tools
        .execute(ToolExecution {
            name: "todo_write".into(),
            args: json!({"step": 1}),
            session_id: Some("sess-x".into()),
        })
        .await;
    assert!(r_todo.additional_contexts.is_empty());

    // 3. grep {"a":1,"b":2} -> 命中阈值 2 且被 pre-execute 拦截，仍收到温和提醒
    let r2 = tools
        .execute(ToolExecution {
            name: "grep".into(),
            args: json!({"a": 1, "b": 2}),
            session_id: Some("sess-x".into()),
        })
        .await;
    assert!(r2.blocked);
    assert_eq!(r2.additional_contexts.len(), 1);
    assert_eq!(r2.additional_contexts[0].source, "repeat-tool-reminder");
    assert!(r2.additional_contexts[0].text.contains("Carefully analyze"));

    // 4. 第 3 次 grep -> 命中阈值 3，收到详细提醒
    let r3 = tools
        .execute(ToolExecution {
            name: "grep".into(),
            args: json!({"a": 1, "b": 2}),
            session_id: Some("sess-x".into()),
        })
        .await;
    assert!(!r3.blocked);
    assert_eq!(r3.additional_contexts.len(), 1);
    assert!(r3.additional_contexts[0].text.contains("consecutive_calls: 3"));

    // 5. 再来一次连续同参 grep（此时是连续第 3 次 grep，无 todo_write 间隔）-> 额外触发 R01_REPEAT_TOOL_CALL
    let _r4 = tools
        .execute(ToolExecution {
            name: "grep".into(),
            args: json!({"b": 2, "a": 1}),
            session_id: Some("sess-x".into()),
        })
        .await;

    // 轨迹服务已记录本次会话并产出告警报告
    let report = app
        .service::<TrajectoryRuntime>("trajectory")
        .unwrap()
        .inspect("sess-x", TimelineMode::Sequence)
        .unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule_id == "GUARD_REPEAT_TOOL" && f.severity == Severity::Warn),
        "findings={:?}",
        report.findings
    );
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule_id == "R01_REPEAT_TOOL_CALL"),
        "findings={:?}",
        report.findings
    );
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.rule_id == "R03_RETRY_AFTER_DENIED"),
        "findings={:?}",
        report.findings
    );
}
