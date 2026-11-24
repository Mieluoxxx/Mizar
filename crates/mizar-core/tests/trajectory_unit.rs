use mizar_core::agent_loop::{AgentLoop, AgentLoopConfig};
use mizar_core::system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_core::trajectory::{
    RepeatGuardConfig, Severity, TimelineMode, TrajectoryConfig, TrajectoryRuntime,
    install_repeat_guard, validate_repeat_guard_config,
};
use mizar_llm::LlmRuntime;
use mizar_session::SessionStore;
use mizar_tools::{PreExecuteDecision, ToolDefinition, ToolExecution, ToolRuntime};
use mizar_types::{BoxError, BoxFuture, Next};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn repeat_guard_validates_config_fail_loud() {
    let tools = ToolRuntime::new();
    let bad_empty = install_repeat_guard(
        &tools,
        None,
        RepeatGuardConfig {
            thresholds: vec![],
            ..Default::default()
        },
    );
    assert!(bad_empty.is_err());

    let bad_lt2 = validate_repeat_guard_config(&RepeatGuardConfig {
        thresholds: vec![1, 3],
        ..Default::default()
    });
    assert!(bad_lt2.is_err());

    let bad_dup = install_repeat_guard(
        &tools,
        None,
        RepeatGuardConfig {
            thresholds: vec![3, 3],
            ..Default::default()
        },
    );
    assert!(bad_dup.is_err());
}

#[tokio::test]
async fn repeat_guard_counts_through_excluded_tools_and_denied_calls() {
    let llm = LlmRuntime::new();
    let sessions = SessionStore::new();
    let tools = ToolRuntime::new();
    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let agent_loop = AgentLoop::new(
        llm,
        Arc::clone(&tools),
        Arc::clone(&sessions),
        sp,
        AgentLoopConfig::default(),
    );
    let (repeat_bus, _guard_handle) = install_repeat_guard(
        &tools,
        Some(&agent_loop),
        RepeatGuardConfig {
            thresholds: vec![2, 3],
            exclude: vec!["todo_*".into()],
            ..Default::default()
        },
    )
    .unwrap();
    let (traj, _traj_guard) = TrajectoryRuntime::install(
        Some(&sessions),
        Some(&tools),
        Some(&agent_loop),
        Some(&repeat_bus),
        TrajectoryConfig::default(),
    );

    // 拦截第 2 次 grep，验证被拦截的调用仍计入重复链
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let _deny_guard = tools.pre_execute_hook().register(
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
    );

    let _c1 = tools
        .register(ToolDefinition::sync_tool(
            "grep",
            "search",
            json!({}),
            |_, _| Ok::<_, BoxError>("ok".into()),
        ))
        .unwrap();
    let _c2 = tools
        .register(ToolDefinition::sync_tool(
            "todo_write",
            "note",
            json!({}),
            |_, _| Ok::<_, BoxError>("noted".into()),
        ))
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
    let report = traj.inspect("sess-x", TimelineMode::Sequence).unwrap();
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let tools = ToolRuntime::new();
    let (traj, _traj_guard) = TrajectoryRuntime::install(
        None,
        Some(&tools),
        None,
        None,
        TrajectoryConfig {
            detector: mizar_types::trajectory::DetectorConfig {
                repeat_threshold: 10,
                consecutive_error_threshold: 2,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let anomalies = Arc::new(std::sync::Mutex::new(Vec::<(String, Option<u64>)>::new()));
    let anomalies_sink = Arc::clone(&anomalies);
    let _anom_guard = traj.on_anomaly(move |ev| {
        anomalies_sink
            .lock()
            .unwrap()
            .push((ev.finding.rule_id.clone(), ev.finding.seq));
    });

    let counter = Arc::new(AtomicUsize::new(0));
    let _c1 = tools
        .register(ToolDefinition::sync_tool(
            "boom",
            "failing test tool",
            json!({}),
            move |_, _| {
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                Err::<String, BoxError>(format!("err #{n}").into())
            },
        ))
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

    let rt = &traj;
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

    let _traj_export = TrajectoryRuntime::new(None, TrajectoryConfig::default());

    let sess = Arc::new(Session::new("sess-export-unit"));
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
        error: None,
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
        error: None,
    });
    sess.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "completed".to_string(),
        error: None,
    });

    let rt = &_traj_export;
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
    use mizar_session::{SessionEvent, SessionStore, SessionTraceEvent};

    let sessions = SessionStore::new();
    let (traj, _traj_guard) = TrajectoryRuntime::install(
        Some(&sessions),
        None,
        None,
        None,
        TrajectoryConfig::default(),
    );
    let s = sessions.create(None);

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
            error: None,
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

#[tokio::test]
async fn export_file_streams_jsonl_lines_incrementally_appends_and_coexists_with_export_dir() {
    use mizar_core::trajectory::prepare_trace_jsonl_file;
    use mizar_session::{SessionEvent, SessionStore, SessionTraceEvent};

    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "mizar_traj_export_file_{}_{}",
        std::process::id(),
        uniq
    ));
    let _ = std::fs::remove_dir_all(&root);
    let jsonl_path = root.join("nested/dir/trace.jsonl");
    let export_dir = root.join("by_session");

    // 预检：嵌套文件可成功预创建，已存在目录应报错
    prepare_trace_jsonl_file(&jsonl_path).unwrap();
    assert!(jsonl_path.exists());
    std::fs::create_dir_all(&export_dir).unwrap();
    assert!(prepare_trace_jsonl_file(&export_dir).is_err());

    // 第一次运行：写入事件并实时校验在会话结束前即可读出合法 JSONL
    let sessions1 = SessionStore::new();
    let (traj1, guard1) = TrajectoryRuntime::install(
        Some(&sessions1),
        None,
        None,
        None,
        TrajectoryConfig {
            export_dir: Some(export_dir.clone()),
            export_file: Some(jsonl_path.clone()),
            ..Default::default()
        },
    );
    let s1 = sessions1.create(None);
    s1.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
    s1.append(SessionEvent::User {
        text: "first run".into(),
        images: Vec::new(),
    });

    // 会话尚未结束时已实时落盘
    let mid_raw = std::fs::read_to_string(&jsonl_path).unwrap();
    let mid_events = TrajectoryRuntime::parse_jsonl(&mid_raw).unwrap();
    assert_eq!(mid_events.len(), 2);

    s1.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "stop".into(),
        error: None,
    });
    let expected_s1 = traj1.export_jsonl(&s1.id).unwrap();
    let file_after_s1 = std::fs::read_to_string(&jsonl_path).unwrap();
    assert_eq!(file_after_s1, expected_s1);
    let dir_s1 = std::fs::read_to_string(export_dir.join(format!("{}.jsonl", s1.id))).unwrap();
    assert_eq!(dir_s1, expected_s1);
    drop(guard1);

    // 第二次运行指向同一路径：应以追加模式保留前序记录
    let sessions2 = SessionStore::new();
    let (_traj2, guard2) = TrajectoryRuntime::install(
        Some(&sessions2),
        None,
        None,
        None,
        TrajectoryConfig {
            export_file: Some(jsonl_path.clone()),
            ..Default::default()
        },
    );
    let s2 = sessions2.create(None);
    s2.append(SessionEvent::User {
        text: "second run".into(),
        images: Vec::new(),
    });
    drop(guard2);

    let combined_raw = std::fs::read_to_string(&jsonl_path).unwrap();
    let combined = TrajectoryRuntime::parse_jsonl(&combined_raw).unwrap();
    assert_eq!(combined.len(), 4);
    assert_eq!(combined[0].session_id, s1.id);
    assert_eq!(combined[3].session_id, s2.id);

    let _ = std::fs::remove_dir_all(&root);
}
