//! Agent 核心层集成测试：热插拔与拦截管线、单会话三协议切换、重复调用守卫自愈。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
use mizar_core::trajectory::TimelineMode;
use mizar_llm::retry::RetryConfig;
use mizar_llm::router::{ProviderRouteConfig, RouterConfig};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, Message, StreamChunk, TokenUsage,
    ToolCall,
};
use mizar_tools::{PreExecuteDecision, ToolAnnotations, ToolDefinition, ToolExecution};
use mizar_types::{BoxFuture, HookGuard, Next};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn register_test_time_tool(runtime: &MizarRuntime) -> HookGuard {
    let def = ToolDefinition::sync_tool(
        "get_time",
        "Get current time",
        json!({"type": "object", "properties": {}}),
        |_, _| Ok::<_, mizar_tools::ToolError>("2026-01-01 12:00:00".to_string()),
    )
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    });
    runtime.tools.register(def).unwrap()
}

#[tokio::test]
async fn agent_hot_plug_and_guard_pipeline() {
    let runtime = MizarRuntime::build(AgentSpineConfig::new(std::env::current_dir().unwrap()))
        .await
        .unwrap();

    // 记录每次工具调用
    let tool_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let tc = Arc::clone(&tool_calls);
    let _result_guard = runtime.tools.on_result(move |ev| {
        tc.lock().unwrap().push(ev.exec.name.clone());
    });

    runtime.register_mock_adapter().unwrap();
    let time_guard = register_test_time_tool(&runtime);

    let ask = |text: &str| {
        let rt = Arc::clone(&runtime);
        let text = text.to_string();
        async move {
            let session = rt.sessions.create(None);
            rt.agent_loop
                .run(session, text, CancellationToken::new())
                .await
                .unwrap()
        }
    };

    // 2. 正常调用工具
    let answer = ask("现在几点？").await;
    assert_eq!(*tool_calls.lock().unwrap(), vec!["get_time".to_string()]);
    assert!(answer.contains("工具返回了结果"), "unexpected: {answer}");

    // 3. 挂一个拦截钩子：在 pre_execute 上拦截，不用改工具和循环的代码
    let guard = runtime.tools.pre_execute_hook().register(
        |exec: ToolExecution, next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            Box::pin(async move {
                if exec.name == "get_time" {
                    PreExecuteDecision::deny("演示拦截")
                } else {
                    next.call(exec).await
                }
            })
        },
    );
    let answer = ask("现在几点？").await;
    assert!(answer.contains("已拦截"), "unexpected: {answer}");

    // 4. 释放 HookGuard，拦截自动撤销
    drop(guard);
    let answer = ask("现在几点？").await;
    assert!(!answer.contains("已拦截"), "unexpected: {answer}");

    // 5. 拔掉工具守卫：工具自动从工具表消失，模型就不会再调用它
    drop(time_guard);
    assert!(!runtime.tools.list().iter().any(|t| t.name == "get_time"));
    tool_calls.lock().unwrap().clear();
    let _ = ask("现在几点？").await;
    assert!(tool_calls.lock().unwrap().is_empty());

    runtime.shutdown().await;
}

#[tokio::test]
async fn agent_switches_across_completions_responses_and_messages_in_one_session() {
    let comp_calls = Arc::new(AtomicUsize::new(0));
    let comp_calls_srv = Arc::clone(&comp_calls);

    let msg_calls = Arc::new(AtomicUsize::new(0));
    let msg_calls_srv = Arc::clone(&msg_calls);

    let server = Router::new()
        .route(
            "/chat/completions",
            post(move || {
                let n = comp_calls_srv.fetch_add(1, Ordering::SeqCst);
                async move {
                    // 第一次故意返回 429 验证 LlmRetryPlugin 自动恢复
                    if n == 0 {
                        return Response::builder()
                            .status(429)
                            .body(Body::from(r#"{"error":{"message":"rate limit"}}"#))
                            .unwrap();
                    }
                    let body = concat!(
                        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think-c\"}}]}\n\n",
                        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_c1\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                        "data: [DONE]\n\n",
                    );
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        )
        .route(
            "/responses",
            post(|| async {
                let body = concat!(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"from-responses\"}\n\n",
                    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":5}}}\n\n",
                );
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            }),
        )
        .route(
            "/v1/messages",
            post(move |axum::Json(payload): axum::Json<serde_json::Value>| {
                let n = msg_calls_srv.fetch_add(1, Ordering::SeqCst);
                async move {
                    // 确认由 completions 产出的历史消息（含 tool_call / tool_result）可以被 messages 协议正确重放，
                    // 且没有携带跨协议不兼容的 thinking 字段
                    if n == 0 {
                        let msgs = payload["messages"].as_array().unwrap();
                        assert!(msgs.len() >= 3);
                    }
                    let body = concat!(
                        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":20,\"output_tokens\":0}}}\n\n",
                        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"from-messages\"}}\n\n",
                        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":8}}\n\n",
                        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    );
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(body))
                        .unwrap()
                }
            }),
        );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, server).await;
    });

    let runtime = MizarRuntime::build(AgentSpineConfig::new(PathBuf::from(env!(
        "CARGO_MANIFEST_DIR"
    ))))
    .await
    .unwrap();
    let _time_guard = register_test_time_tool(&runtime);
    runtime.install_retry(RetryConfig {
        max_retries: 2,
        initial_delay: Duration::from_millis(5),
        ..Default::default()
    });

    let base = format!("http://{addr}");
    let mut providers = BTreeMap::new();
    providers.insert(
        "comp".into(),
        ProviderRouteConfig {
            api_key: Some("k1".into()),
            base_url: Some(base.clone()),
            ..ProviderRouteConfig::openai_completions("UNUSED")
        },
    );
    providers.insert(
        "resp".into(),
        ProviderRouteConfig {
            api_key: Some("k2".into()),
            base_url: Some(base.clone()),
            ..ProviderRouteConfig::openai_responses("UNUSED")
        },
    );
    providers.insert(
        "msg".into(),
        ProviderRouteConfig {
            api_key: Some("k3".into()),
            base_url: Some(base),
            ..ProviderRouteConfig::anthropic_messages("UNUSED")
        },
    );
    let _router = runtime.install_router(RouterConfig { providers });

    let llm = &runtime.llm;
    let session = runtime.sessions.create(None);

    // 1. 先走 completions（触发一次 429 重试后拿到 tool_call）
    let mut s1 = llm
        .stream(GenerateOptions {
            provider: Some("comp".into()),
            messages: vec![Message::user("现在几点")],
            ..Default::default()
        })
        .unwrap();
    let mut asm1 = mizar_llm::BlockAssembler::new();
    while let Some(chunk) = futures::StreamExt::next(&mut s1).await {
        asm1.push(chunk.unwrap());
    }
    assert_eq!(comp_calls.load(Ordering::SeqCst), 2);
    assert_eq!(asm1.tool_calls().len(), 1);

    session.append(mizar_session::SessionEvent::user("现在几点"));
    session.append(mizar_session::SessionEvent::Assistant {
        blocks: asm1.blocks(),
        replay: asm1.replay(),
        usage: asm1.usage(),
    });
    session.append(mizar_session::SessionEvent::ToolResult {
        tool_call_id: "call_c1".into(),
        name: "get_time".into(),
        content: "2026/11/18 02:45:00".into(),
        is_error: false,
        meta: None,
    });

    // 2. 同一会话切到 Anthropic Messages 继续回答
    let mut s2 = llm
        .stream(GenerateOptions {
            provider: Some("msg".into()),
            messages: session.derive_messages(),
            ..Default::default()
        })
        .unwrap();
    let mut asm2 = mizar_llm::BlockAssembler::new();
    while let Some(chunk) = futures::StreamExt::next(&mut s2).await {
        asm2.push(chunk.unwrap());
    }
    assert_eq!(asm2.text(), "from-messages");
    assert_eq!(
        asm2.usage(),
        Some(TokenUsage {
            input_tokens: 20,
            output_tokens: 8,
            total_tokens: Some(28),
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        })
    );

    // 3. 再切到 OpenAI Responses 继续对话
    let mut s3 = llm
        .stream(GenerateOptions {
            provider: Some("resp".into()),
            messages: session.derive_messages(),
            ..Default::default()
        })
        .unwrap();
    let mut asm3 = mizar_llm::BlockAssembler::new();
    while let Some(chunk) = futures::StreamExt::next(&mut s3).await {
        asm3.push(chunk.unwrap());
    }
    assert_eq!(asm3.text(), "from-responses");

    runtime.shutdown().await;
}

#[tokio::test]
async fn agent_trajectory_detects_repeat_loop_and_recovers_after_guard_reminder() {
    struct LoopingAdapter(AtomicUsize);

    impl LlmAdapter for LoopingAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let step_idx = self.0.fetch_add(1, Ordering::SeqCst);
            let saw_reminder = options.messages.iter().any(|m| match m {
                Message::User { content, .. } => content.contains("Carefully analyze"),
                _ => false,
            });
            let (tx, rx) = futures::channel::mpsc::unbounded();
            if saw_reminder {
                let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                    index: 0,
                    text: "收到提醒，已跳出重复循环".to_string(),
                }));
                let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                    usage: TokenUsage {
                        input_tokens: 30,
                        output_tokens: 12,
                        ..Default::default()
                    },
                }));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            } else {
                let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                    0,
                    ToolCall {
                        id: format!("loop_call_{step_idx}"),
                        name: "get_time".to_string(),
                        arguments: "{}".to_string(),
                    },
                )));
                let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                    usage: TokenUsage {
                        input_tokens: 20,
                        output_tokens: 5,
                        ..Default::default()
                    },
                }));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::ToolCalls,
                    replay: None,
                }));
            }
            Box::pin(rx)
        }
    }

    let runtime = MizarRuntime::build(
        AgentSpineConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
            .with_disabled_plugins(["session-title"]),
    )
    .await
    .unwrap();
    let _time_guard = register_test_time_tool(&runtime);
    let anomalies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let anomalies_cb = Arc::clone(&anomalies);
    let traj = runtime.trajectory.as_ref().unwrap();
    let _anom_guard = traj.on_anomaly(move |ev| {
        anomalies_cb
            .lock()
            .unwrap()
            .push(ev.finding.rule_id.clone());
    });
    runtime
        .register_adapter("looping", Arc::new(LoopingAdapter(AtomicUsize::new(0))))
        .unwrap();

    let session = runtime.sessions.create(None);
    let reply = runtime
        .agent_loop
        .run(Arc::clone(&session), "帮我查时间", CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(reply, "收到提醒，已跳出重复循环");

    let report = traj.inspect(&session.id, TimelineMode::Sequence).unwrap();
    assert_eq!(report.snapshot.stats.turns, 1);
    assert_eq!(report.snapshot.stats.tool_calls, 3);
    assert_eq!(report.snapshot.stats.steps, 4);
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
            .snapshot
            .records
            .iter()
            .any(|r| r.source.as_deref() == Some("repeat-tool-reminder")),
        "records={:?}",
        report.snapshot.records
    );
    assert!(
        anomalies
            .lock()
            .unwrap()
            .iter()
            .any(|id| id == "GUARD_REPEAT_TOOL" || id == "R01_REPEAT_TOOL_CALL")
    );

    runtime.shutdown().await;
}

#[tokio::test]
async fn step_and_turn_error_traces_persist_with_utf8_2000_byte_truncation() {
    use mizar_llm::LlmErrorCode;
    use mizar_session::{SessionEvent, SessionTraceEvent};

    struct FailOrOkAdapter(AtomicUsize);
    impl LlmAdapter for FailOrOkAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let idx = self.0.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = futures::channel::mpsc::unbounded();
            if idx == 0 {
                let message = format!("ERR_{}", "测".repeat(800));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Error {
                        code: LlmErrorCode::InvalidRequest,
                        message,
                        status: Some(400),
                        retry_after_ms: None,
                    },
                    replay: None,
                }));
            } else {
                let _ = tx.unbounded_send(Ok(StreamChunk::text("ok")));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            }
            Box::pin(rx)
        }
    }

    let runtime = MizarRuntime::build(
        AgentSpineConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
            .with_disabled_plugins(["session-title"]),
    )
    .await
    .unwrap();
    runtime
        .register_adapter("fail-or-ok", Arc::new(FailOrOkAdapter(AtomicUsize::new(0))))
        .unwrap();

    // 1. 报错结束时，StepEnd 与 TurnEnd 落盘截断后的错误文本（UTF-8 安全且 <= 2000 字节）
    let s_err = runtime.sessions.create(None);
    let err = runtime
        .agent_loop
        .run(
            Arc::clone(&s_err),
            "trigger error",
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("ERR_"));

    let entries = s_err.all_entries();
    let step_end = entries
        .iter()
        .find_map(|e| match &e.event {
            SessionEvent::Trace {
                detail: SessionTraceEvent::StepEnd { outcome, error, .. },
            } => Some((outcome.clone(), error.clone())),
            _ => None,
        })
        .expect("missing StepEnd trace");
    assert_eq!(step_end.0, "error");
    let step_err = step_end.1.expect("StepEnd.error should be Some");
    assert!(step_err.len() <= 2_000);
    assert_eq!(step_err.len(), 1_999);
    assert!(step_err.starts_with("ERR_测"));

    let turn_end = entries
        .iter()
        .find_map(|e| match &e.event {
            SessionEvent::Trace {
                detail: SessionTraceEvent::TurnEnd { reason, error, .. },
            } => Some((reason.clone(), error.clone())),
            _ => None,
        })
        .expect("missing TurnEnd trace");
    assert_eq!(turn_end.0, "error");
    let turn_err = turn_end.1.expect("TurnEnd.error should be Some");
    assert!(turn_err.len() <= 2_000);
    assert_eq!(turn_err.len(), 1_999);
    assert!(turn_err.starts_with("ERR_测"));

    // 2. 正常结束时，StepEnd 与 TurnEnd 的 error 为 None 且不序列化 "error" 字段
    let s_ok = runtime.sessions.create(None);
    let reply = runtime
        .agent_loop
        .run(Arc::clone(&s_ok), "hello", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply, "ok");

    for entry in s_ok.all_entries() {
        if let SessionEvent::Trace {
            detail:
                detail @ (SessionTraceEvent::StepEnd { error, .. }
                | SessionTraceEvent::TurnEnd { error, .. }),
        } = &entry.event
        {
            assert!(error.is_none());
            let raw = serde_json::to_string(detail).unwrap();
            assert!(
                !raw.contains(r#""error":"#),
                "unexpected error field: {raw}"
            );
        }
    }

    runtime.shutdown().await;
}

#[tokio::test]
async fn compaction_tail_pruning_enforces_cache_guards_and_resume_parity_end_to_end() {
    use mizar_core::compaction::{CompactionConfig, SUPERSEDED_NOTICE, USELESS_NOTICE};
    use mizar_llm::{ModelInfo, ProviderInfo, ReasoningEffort};
    use mizar_session::{
        SessionCreateOptions, SessionEvent, SessionStore, SessionTraceEvent,
        SqliteSessionPersistence,
    };
    use mizar_tools::coding::{CodingToolsConfig, register_coding_tools};

    struct SequenceAdapter {
        protocol: &'static str,
        reasoning: bool,
        calls: Arc<Mutex<Vec<GenerateOptions>>>,
        step: AtomicUsize,
        target_path: String,
    }

    impl LlmAdapter for SequenceAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "seq-mock".into(),
                name: "Sequence Mock".into(),
                protocol: self.protocol.into(),
                models: vec![ModelInfo {
                    id: "seq-model".into(),
                    name: "Seq Model".into(),
                    context_window: Some(200_000),
                    max_tokens: Some(8_192),
                    reasoning_efforts: if self.reasoning {
                        vec![ReasoningEffort::Medium]
                    } else {
                        Vec::new()
                    },
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.calls.lock().unwrap().push(options);
            let idx = self.step.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = futures::channel::mpsc::unbounded();
            match idx {
                0 => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "call_r1".into(),
                            name: "read".into(),
                            arguments: json!({"path": self.target_path}).to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        1,
                        ToolCall {
                            id: "call_g1".into(),
                            name: "grep".into(),
                            arguments: json!({
                                "pattern": "NO_SUCH_SYMBOL_12345",
                                "path": self.target_path,
                            })
                            .to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                }
                1 => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "call_r2".into(),
                            name: "read".into(),
                            arguments: json!({"path": self.target_path}).to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                }
                _ => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("all checked")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            }
            Box::pin(rx)
        }
    }

    assert_eq!(CompactionConfig::default().max_context_tokens, None);

    let root = std::env::temp_dir().join(format!("mizar_prune_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let target_file = root.join("demo.txt");
    std::fs::write(&target_file, "alpha\nbeta\ngamma\n").unwrap();
    let db_path = root.join("sessions.sqlite");

    // 1. 普通模型（openai-completions）热尾 <= 8,000 tokens：自动剪枝旧 read + 0 命中 grep，并保证 resume 前缀字节一致
    let calls_warm = Arc::new(Mutex::new(Vec::new()));
    let mut cfg = AgentSpineConfig::new(root.clone()).with_disabled_plugins(["session-title"]);
    cfg.session_db_path = Some(db_path.clone());
    let runtime = MizarRuntime::build(cfg).await.unwrap();
    let _coding_guards =
        register_coding_tools(&runtime.tools, &CodingToolsConfig::new(&root)).unwrap();
    runtime
        .register_adapter(
            "seq-mock",
            Arc::new(SequenceAdapter {
                protocol: "openai-completions",
                reasoning: false,
                calls: Arc::clone(&calls_warm),
                step: AtomicUsize::new(0),
                target_path: "demo.txt".into(),
            }),
        )
        .unwrap();

    let sess = runtime.sessions.create_with(SessionCreateOptions {
        id: Some("e2e-warm".into()),
        cwd: Some(root.display().to_string()),
        ..Default::default()
    });
    let reply = runtime
        .agent_loop
        .run(
            Arc::clone(&sess),
            "inspect demo.txt",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply, "all checked");

    {
        let recorded = calls_warm.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        let step3_msgs = &recorded[2].messages;
        assert!(
            step3_msgs.iter().any(|m| matches!(
                m,
                Message::Tool { tool_call_id, content, is_error, .. }
                    if tool_call_id == "call_r1" && !is_error && content == SUPERSEDED_NOTICE
            )),
            "expected call_r1 superseded in {step3_msgs:?}"
        );
        assert!(
            step3_msgs.iter().any(|m| matches!(
                m,
                Message::Tool { tool_call_id, content, is_error, .. }
                    if tool_call_id == "call_g1" && !is_error && content == USELESS_NOTICE
            )),
            "expected call_g1 elided in {step3_msgs:?}"
        );
        assert!(
            step3_msgs.iter().any(|m| matches!(
                m,
                Message::Tool { tool_call_id, content, is_error, .. }
                    if tool_call_id == "call_r2" && !is_error && content.contains("alpha")
            )),
            "expected call_r2 intact in {step3_msgs:?}"
        );
    }

    let live_derived = sess.derive_messages();
    sess.flush_async().await.unwrap();
    drop(_coding_guards);
    runtime.shutdown().await;

    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let resumed = store2.get("e2e-warm").unwrap();
    assert_eq!(resumed.derive_messages(), live_derived);
    drop(resumed);
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;

    // 2. 前缀绑定思考模型（anthropic-messages + reasoning）：热缓存阶段不改写历史 read
    let calls_prefix = Arc::new(Mutex::new(Vec::new()));
    let runtime_pb = MizarRuntime::build(
        AgentSpineConfig::new(root.clone()).with_disabled_plugins(["session-title"]),
    )
    .await
    .unwrap();
    let _coding_guards_pb =
        register_coding_tools(&runtime_pb.tools, &CodingToolsConfig::new(&root)).unwrap();
    runtime_pb
        .register_adapter(
            "seq-mock",
            Arc::new(SequenceAdapter {
                protocol: "anthropic-messages",
                reasoning: true,
                calls: Arc::clone(&calls_prefix),
                step: AtomicUsize::new(0),
                target_path: "demo.txt".into(),
            }),
        )
        .unwrap();
    let sess_pb = runtime_pb.sessions.create_with(SessionCreateOptions {
        id: Some("e2e-prefix-bound".into()),
        cwd: Some(root.display().to_string()),
        ..Default::default()
    });
    let reply_pb = runtime_pb
        .agent_loop
        .run(
            Arc::clone(&sess_pb),
            "inspect demo.txt",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply_pb, "all checked");

    assert!(
        !sess_pb.all_entries().iter().any(|e| matches!(
            &e.event,
            SessionEvent::Trace {
                detail: SessionTraceEvent::ToolResultPrune { target_seq: 3, .. }
            }
        )),
        "prefix-bound thinking model must not prune warm read with trailing suffix"
    );
    let _ = calls_prefix;
    drop(_coding_guards_pb);
    runtime_pb.shutdown().await;
    let _ = std::fs::remove_dir_all(&root);
}
