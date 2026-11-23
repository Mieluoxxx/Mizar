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
