use cordis::{BoxFuture, Context, CordisError, Next, Service};
use mizar_core::trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, Severity, TimelineMode, TrajectoryConfig,
    TrajectoryRuntime,
};
use mizar_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolDefinition, ToolExecution, ToolRuntime,
    ToolsContextExt,
};
use serde_json::json;

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
    app.plugin(TrajectoryRuntime::plugin(), TrajectoryConfig::default())
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
                        if exec.name == "grep"
                            && calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1
                        {
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
        .execute(ToolExecution::new(
            "grep",
            json!({"b": 2, "a": 1}),
            Some("sess-x".into()),
        ))
        .await;
    assert!(r1.additional_contexts.is_empty());

    // 2. todo_write (在 exclude 里，应当透明，不打断链条)
    let r_todo = tools
        .execute(ToolExecution::new(
            "todo_write",
            json!({"step": 1}),
            Some("sess-x".into()),
        ))
        .await;
    assert!(r_todo.additional_contexts.is_empty());

    // 3. grep {"a":1,"b":2} -> 命中阈值 2 且被 pre-execute 拦截，仍收到温和提醒
    let r2 = tools
        .execute(ToolExecution::new(
            "grep",
            json!({"a": 1, "b": 2}),
            Some("sess-x".into()),
        ))
        .await;
    assert!(r2.blocked);
    assert_eq!(r2.additional_contexts.len(), 1);
    assert_eq!(r2.additional_contexts[0].source, "repeat-tool-reminder");
    assert!(r2.additional_contexts[0].text.contains("Carefully analyze"));

    // 4. 第 3 次 grep -> 命中阈值 3，收到详细提醒
    let r3 = tools
        .execute(ToolExecution::new(
            "grep",
            json!({"a": 1, "b": 2}),
            Some("sess-x".into()),
        ))
        .await;
    assert!(!r3.blocked);
    assert_eq!(r3.additional_contexts.len(), 1);
    assert!(
        r3.additional_contexts[0]
            .text
            .contains("consecutive_calls: 3")
    );

    // 5. 再来一次连续同参 grep（此时是连续第 3 次 grep，无 todo_write 间隔）-> 额外触发 R01_REPEAT_TOOL_CALL
    let _r4 = tools
        .execute(ToolExecution::new(
            "grep",
            json!({"b": 2, "a": 1}),
            Some("sess-x".into()),
        ))
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

#[tokio::test]
async fn incremental_anomaly_eval_emits_new_findings_without_duplicating_old_ones() {
    use mizar_core::trajectory::EVENT_TRAJECTORY_ANOMALY;
    use mizar_core::trajectory::TrajectoryAnomalyEvent;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let app = Context::new();
    let anomalies = Arc::new(std::sync::Mutex::new(Vec::<(String, Option<u64>)>::new()));
    let anomalies_sink = Arc::clone(&anomalies);
    app.on(
        EVENT_TRAJECTORY_ANOMALY,
        move |ev: &TrajectoryAnomalyEvent| {
            anomalies_sink
                .lock()
                .unwrap()
                .push((ev.finding.rule_id.clone(), ev.finding.seq));
        },
    )
    .unwrap();

    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(
        TrajectoryRuntime::plugin(),
        TrajectoryConfig {
            detector: mizar_types::trajectory::DetectorConfig {
                repeat_threshold: 10,
                consecutive_error_threshold: 2,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let tools = app.tools().unwrap();
    app.effect("reg.failing", || {
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = tools
            .register(ToolDefinition::sync_tool(
                "boom",
                "failing test tool",
                json!({}),
                move |_, _| {
                    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    Err::<String, cordis::BoxError>(format!("err #{n}").into())
                },
            ))
            .unwrap();
        Ok::<_, CordisError>(Some(cordis::Cleanup::sync(move || drop(c1))))
    })
    .unwrap();

    for i in 1..=3 {
        let res = tools
            .execute(ToolExecution::new(
                "boom",
                json!({ "n": i }),
                Some("sess-err-inc".into()),
            ))
            .await;
        assert!(res.is_error);
    }

    let emitted = anomalies.lock().unwrap().clone();
    let r02_seqs: Vec<Option<u64>> = emitted
        .iter()
        .filter(|(id, _)| id == "R02_CONSECUTIVE_TOOL_ERRORS")
        .map(|(_, s)| *s)
        .collect();
    // 第 2 次失败发射一次 R02，第 3 次失败发射新的 seq，但第 2 次的 seq 绝不能重复发射
    assert_eq!(r02_seqs.len(), 2, "emitted={emitted:?}");
    assert_ne!(r02_seqs[0], r02_seqs[1]);

    let rt = app.service::<TrajectoryRuntime>("trajectory").unwrap();
    let report = rt.inspect("sess-err-inc", TimelineMode::Sequence).unwrap();
    let events = rt.events("sess-err-inc");
    let offline = rt.inspect_events("sess-err-inc", &events, TimelineMode::Sequence);
    assert_eq!(report.snapshot.stats, offline.snapshot.stats);
    assert_eq!(report.findings, offline.findings);
}

#[tokio::test]
async fn build_trajectory_export_captures_ttft_usage_tools_compaction_and_atomic_write() {
    use mizar_core::trajectory::{
        DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS, TRAJECTORY_EXPORT_SCHEMA_VERSION, TraceExportMeta,
        TrajectoryExport, build_trajectory_export, write_trace_export_atomic,
    };
    use mizar_llm::{ContentBlock, TokenUsage};
    use mizar_session::{Session, SessionEvent, SessionTraceEvent};
    use std::sync::Arc;

    let app = Context::new();
    app.plugin(TrajectoryRuntime::plugin(), TrajectoryConfig::default())
        .await
        .unwrap();

    let sess = Arc::new(Session::new(app.clone(), "sess-export-unit"));
    sess.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
    sess.append(SessionEvent::user("inspect project"));
    sess.append_trace(SessionTraceEvent::StepStart {
        turn: 1,
        step: 1,
        system_prompt_chars: 1200,
        tools_count: 4,
    });
    sess.append_trace(SessionTraceEvent::FirstToken { turn: 1, step: 1 });

    let long_args = json!({
        "path": "src/main.rs",
        "note": "x".repeat(800),
    })
    .to_string();
    let step1_usage = TokenUsage {
        input_tokens: 100,
        output_tokens: 25,
        total_tokens: Some(165),
        cache_read_tokens: Some(40),
        cache_write_tokens: Some(10),
        reasoning_tokens: Some(5),
    };
    sess.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "call_1".to_string(),
            name: "read".to_string(),
            arguments: long_args.clone(),
        }],
        usage: Some(step1_usage),
        replay: None,
    });
    sess.append_trace(SessionTraceEvent::ToolCall {
        turn: 1,
        step: 1,
        call_id: "call_1".to_string(),
        name: "read".to_string(),
        arguments: long_args.clone(),
    });
    let big_result = format!("头部内容-{}", "测试字符".repeat(220));
    let expected_chars = big_result.chars().count();
    assert!(expected_chars > DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS);
    sess.append(SessionEvent::ToolResult {
        tool_call_id: "call_1".to_string(),
        name: "read".to_string(),
        content: big_result,
        is_error: false,
    });
    sess.append_trace(SessionTraceEvent::StepEnd {
        turn: 1,
        step: 1,
        outcome: "tool_calls".to_string(),
        usage: Some(step1_usage),
    });

    sess.append_trace(SessionTraceEvent::CompactionStart {
        trigger: "auto".to_string(),
    });
    sess.append(SessionEvent::Compaction {
        summary: "已压缩早期文件读取输出".to_string(),
        first_kept_seq: 4,
        tokens_before: Some(9000),
        tokens_after: Some(2500),
        digest: Default::default(),
    });
    sess.append_trace(SessionTraceEvent::CompactionEnd {
        status: "compacted".to_string(),
        first_kept_seq: Some(4),
        tokens_before: Some(9000),
        tokens_after: Some(2500),
    });

    sess.append_trace(SessionTraceEvent::StepStart {
        turn: 1,
        step: 2,
        system_prompt_chars: 900,
        tools_count: 4,
    });
    sess.append_trace(SessionTraceEvent::FirstToken { turn: 1, step: 2 });
    let step2_usage = TokenUsage {
        input_tokens: 50,
        output_tokens: 15,
        total_tokens: Some(145),
        cache_read_tokens: Some(80),
        cache_write_tokens: None,
        reasoning_tokens: None,
    };
    sess.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "done".to_string(),
        }],
        usage: Some(step2_usage),
        replay: None,
    });
    sess.append_trace(SessionTraceEvent::StepEnd {
        turn: 1,
        step: 2,
        outcome: "completed".to_string(),
        usage: Some(step2_usage),
    });
    sess.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "completed".to_string(),
    });

    let rt = app.service::<TrajectoryRuntime>("trajectory").unwrap();
    let events = mizar_core::trajectory::trace_events_from_session(&sess);
    let report = rt.inspect_events(&sess.id, &events, TimelineMode::Sequence);
    let export = sess.with_all_entries(|entries| {
        build_trajectory_export(
            report,
            Some(entries),
            TraceExportMeta {
                exit_code: 0,
                termination_reason: "completed".to_string(),
                error: None,
                duration_ms: 42,
                provider: Some("openai".to_string()),
                model: Some("gpt-test".to_string()),
                isolated: true,
                output_format: Some("text".to_string()),
            },
            DEFAULT_TOOL_RESULT_SUMMARY_MAX_CHARS,
        )
    });

    assert_eq!(export.schema_version, TRAJECTORY_EXPORT_SCHEMA_VERSION);
    assert_eq!(export.session_id, "sess-export-unit");
    assert_eq!(export.exit_code, 0);
    assert_eq!(export.termination_reason, "completed");
    assert!(export.ttft_ms.is_some());
    assert_eq!(export.turns.len(), 1);
    assert_eq!(export.turns[0].steps.len(), 2);
    assert_eq!(export.turns[0].usage.input_tokens, 150);
    assert_eq!(export.turns[0].usage.output_tokens, 40);
    assert_eq!(export.turns[0].usage.cache_read_tokens, 120);
    assert_eq!(export.turns[0].usage.cache_write_tokens, 10);
    assert_eq!(export.turns[0].usage.reasoning_tokens, 5);
    assert_eq!(export.usage, export.turns[0].usage);
    assert_eq!(export.tool_calls.len(), 1);
    assert_eq!(export.tool_calls[0].arguments, long_args);
    assert_eq!(export.tool_results.len(), 1);
    assert!(export.tool_results[0].truncated);
    assert_eq!(export.tool_results[0].original_chars, expected_chars);
    assert!(
        export.tool_results[0]
            .content_summary
            .contains("[truncated")
    );
    assert_eq!(export.compactions.len(), 3);
    assert_eq!(export.compactions[0].kind, "start");
    assert_eq!(export.compactions[1].kind, "checkpoint");
    assert_eq!(export.compactions[1].first_kept_seq, Some(4));
    assert_eq!(export.compactions[2].kind, "end");

    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mizar_traj_export_{}_{uniq}", std::process::id()));
    let target = dir.join("nested/sub/trajectory.json");
    write_trace_export_atomic(&target, &export).unwrap();
    let raw = std::fs::read_to_string(&target).unwrap();
    let decoded: TrajectoryExport = serde_json::from_str(&raw).unwrap();
    assert_eq!(decoded, export);

    // 不得残留临时文件，且目标路径为目录时应明确返回错误
    let entries: Vec<_> = std::fs::read_dir(target.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert_eq!(entries.len(), 1);
    assert!(write_trace_export_atomic(&dir, &export).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn findings_only_matches_inspect_findings_without_snapshot_clone() {
    use mizar_core::trajectory::TrajectoryContextExt;
    use mizar_session::{SessionContextExt, SessionEvent, SessionStore, SessionTraceEvent};

    let ctx = Context::new();
    ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
    ctx.plugin(TrajectoryRuntime::plugin(), TrajectoryConfig::default())
        .await
        .unwrap();

    let sessions = ctx.sessions().unwrap();
    let s = sessions.create(None);
    let traj = ctx.trajectory().unwrap();

    for step in 1..=4 {
        s.append_trace(SessionTraceEvent::StepStart {
            turn: 1,
            step,
            system_prompt_chars: 100,
            tools_count: 1,
        });
        s.append_trace(SessionTraceEvent::ToolCall {
            turn: 1,
            step,
            call_id: format!("c-{step}"),
            name: "read".into(),
            arguments: r#"{"path":"foo.rs"}"#.into(),
        });
        s.append(SessionEvent::ToolResult {
            tool_call_id: format!("c-{step}"),
            name: "read".into(),
            content: "err".into(),
            is_error: true,
        });
        s.append_trace(SessionTraceEvent::StepEnd {
            turn: 1,
            step,
            outcome: "tool_calls".into(),
            usage: None,
        });
    }

    let expected = traj
        .inspect(&s.id, TimelineMode::Sequence)
        .unwrap()
        .findings;
    let fast = traj.findings_only(&s.id);
    assert!(!fast.is_empty());
    assert_eq!(fast, expected);
}
