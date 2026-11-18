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
