use cordis::{BoxFuture, Context, CordisError, Next, Service};
use mizar_tools::{
    EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolDefinition, ToolExecution, ToolRuntime,
    ToolsContextExt,
};
use mizar_trajectory::{
    RepeatGuardConfig, RepeatToolGuardPlugin, Severity, TimelineMode, TrajectoryConfig,
    TrajectoryRuntime,
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
    use mizar_trajectory::EVENT_TRAJECTORY_ANOMALY;
    use mizar_trajectory::TrajectoryAnomalyEvent;
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
            detector: mizar_trajectory_core::DetectorConfig {
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
