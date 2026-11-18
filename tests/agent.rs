//! Agent 层测试（对应 `NamiDSH/tests/agent.ts`）+ 网页服务 SSE 集成测试。

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::response::Response;
use axum::routing::post;
use cordis::{BoxFuture, Context, CordisError, Next, PluginDef, Service};
use dsh_agent_loop::{AgentLoop, AgentLoopContextExt};
use dsh_agent_spine::{AgentSpineConfig, AgentSpinePlugin};
use dsh_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmRuntime, Message,
    StreamChunk, TokenUsage, ToolCall,
};
use dsh_llm_mock::MockPlugin;
use dsh_llm_retry::{LlmRetryPlugin, RetryConfig};
use dsh_llm_router::{LlmRouter, ProviderRouteConfig, RouterConfig};
use dsh_session::SessionContextExt;
use dsh_tool_time::ToolTimePlugin;
use dsh_tools::{
    EVENT_TOOLS_PRE_EXECUTE, EVENT_TOOLS_RESULT, PreExecuteDecision, ToolExecution,
    ToolResultEvent, ToolsContextExt,
};
use dsh_trajectory::{EVENT_TRAJECTORY_ANOMALY, TimelineMode, TrajectoryAnomalyEvent, TrajectoryRuntime};
use dsh_webserver::{EVENT_WEBSERVER_LISTENING, WebServerConfig, WebServerPlugin};
use serde_json::json;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn tick(ms: u64) {
    tokio::time::sleep(Duration::from_millis(ms)).await;
}

#[tokio::test]
async fn agent_hot_plug_and_guard_pipeline() {
    let app = Context::new();

    // 记录每次工具调用
    let tool_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let tc = Arc::clone(&tool_calls);
    app.on(EVENT_TOOLS_RESULT, move |ev: &ToolResultEvent| {
        tc.lock().unwrap().push(ev.exec.name.clone());
    })
    .unwrap();

    // 1. 乱序挂载：先挂模型适配器（依赖 llm，先等着），再挂核心套件
    app.plugin(MockPlugin, ());
    app.plugin(
        AgentSpinePlugin,
        AgentSpineConfig::new(std::env::current_dir().unwrap()),
    );
    tick(30).await;

    let ask = |text: &str| {
        let app = app.clone();
        let text = text.to_string();
        async move {
            let session = app.sessions().unwrap().create(None);
            app.agent_loop()
                .unwrap()
                .run(session, text, CancellationToken::new())
                .await
                .unwrap()
        }
    };

    // 2. 正常调用工具
    let answer = ask("现在几点？").await;
    assert_eq!(*tool_calls.lock().unwrap(), vec!["get_time".to_string()]);
    assert!(answer.contains("工具返回了结果"), "unexpected: {answer}");

    // 3. 挂一个审批插件：在 tools/pre-execute 上拦截，不用改工具和循环的代码
    let guard = app.plugin(
        PluginDef::sync("guard", |ctx, ()| {
            ctx.on_waterfall(
                EVENT_TOOLS_PRE_EXECUTE,
                |exec: ToolExecution, next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
                    Box::pin(async move {
                        if exec.name == "get_time" {
                            PreExecuteDecision::deny("演示拦截")
                        } else {
                            next.call(exec).await
                        }
                    })
                },
            )?;
            Ok::<_, CordisError>(())
        }),
        (),
    );
    tick(20).await;
    let answer = ask("现在几点？").await;
    assert!(answer.contains("已拦截"), "unexpected: {answer}");

    // 4. 卸载审批插件，监听自动撤销
    guard.dispose().await;
    let answer = ask("现在几点？").await;
    assert!(!answer.contains("已拦截"), "unexpected: {answer}");

    // 5. 拔掉工具插件：工具自动从工具表消失，模型就不会再调用它
    app.registry().delete(&ToolTimePlugin);
    tick(20).await;
    assert!(
        !app.tools()
            .unwrap()
            .list()
            .iter()
            .any(|t| t.name == "get_time")
    );
    tool_calls.lock().unwrap().clear();
    let _ = ask("现在几点？").await;
    assert!(tool_calls.lock().unwrap().is_empty());

    // 6. 拔掉核心服务 llm：依赖它的 agentLoop 自动停止
    let llm_rt = app
        .registry()
        .values()
        .into_iter()
        .find(|r| r.name.as_deref() == Some(LlmRuntime::plugin_name()))
        .expect("LlmRuntime registered");
    for fiber in llm_rt.fibers() {
        fiber.dispose().await;
    }
    tick(20).await;
    assert!(app.get::<AgentLoop>("agentLoop").is_none());
}

#[tokio::test]
async fn webserver_serves_index_and_streams_chat_sse() {
    let app = Context::new();
    let (addr_tx, addr_rx) = oneshot::channel::<SocketAddr>();
    let addr_tx = Arc::new(Mutex::new(Some(addr_tx)));
    app.on(EVENT_WEBSERVER_LISTENING, move |addr: &SocketAddr| {
        if let Some(tx) = addr_tx.lock().unwrap().take() {
            let _ = tx.send(*addr);
        }
    })
    .unwrap();

    app.plugin(
        AgentSpinePlugin,
        AgentSpineConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    );
    app.plugin(MockPlugin, ());
    app.plugin(
        WebServerPlugin,
        WebServerConfig {
            port: 0,
            host: "127.0.0.1".to_string(),
            static_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public"),
        },
    );

    let addr = tokio::time::timeout(Duration::from_secs(3), addr_rx)
        .await
        .expect("webserver did not start in time")
        .unwrap();

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let html = client
        .get(format!("http://{addr}/"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("NamiDSH Agent"));

    let sse = client
        .post(format!("http://{addr}/api/chat"))
        .json(&json!({ "message": "现在几点？" }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(sse.contains("event: session"));
    assert!(sse.contains("event: tool"));
    assert!(sse.contains("get_time"));
    assert!(sse.contains("event: delta"));
    assert!(sse.contains("event: done"));

    // 从首轮 SSE 提取 sessionId 并继续多轮对话（触发 list_dir）
    let session_id = sse
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .and_then(|v| v.get("id").and_then(|id| id.as_str()).map(str::to_string))
        .expect("session id in first sse event");

    let sse2 = client
        .post(format!("http://{addr}/api/chat"))
        .json(&json!({ "sessionId": session_id, "message": "项目里有哪些文件？" }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(sse2.contains("event: tool"));
    assert!(sse2.contains("list_dir"));
    let reply_text: String = sse2
        .split("\n\n")
        .filter_map(|part| {
            let mut event = None;
            let mut data = None;
            for line in part.lines() {
                if let Some(e) = line.strip_prefix("event: ") {
                    event = Some(e);
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = Some(d);
                }
            }
            if event == Some("delta") {
                data.and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
            } else {
                None
            }
        })
        .collect();
    assert!(reply_text.contains("Cargo.toml"), "reply: {reply_text}");
    assert!(sse2.contains("event: done"));

    // 验证轨迹查询 / JSONL 导出 / 离线 JSONL 检测接口
    let sessions_json: serde_json::Value = client
        .get(format!("http://{addr}/api/sessions"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(sessions_json.as_array().unwrap().iter().any(|s| s["sessionId"] == session_id));

    let traj_report: serde_json::Value = client
        .get(format!("http://{addr}/api/sessions/{session_id}/trajectory?mode=actual"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(traj_report["snapshot"]["stats"]["turns"], 2);
    assert_eq!(traj_report["snapshot"]["stats"]["toolCalls"], 2);
    assert_eq!(traj_report["snapshot"]["stats"]["toolErrors"], 0);
    assert!(traj_report["timeline"]["spans"].as_array().unwrap().len() >= 6);

    let jsonl = client
        .get(format!("http://{addr}/api/sessions/{session_id}/export"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(jsonl.contains("\"type\":\"turn-start\""));
    assert!(jsonl.contains("\"type\":\"tool-call\""));

    let offline_report: serde_json::Value = client
        .post(format!("http://{addr}/api/trajectory/inspect"))
        .json(&json!({ "jsonl": jsonl, "mode": "duration" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(offline_report["snapshot"]["stats"]["turns"], 2);
    assert_eq!(offline_report["snapshot"]["stats"]["toolCalls"], 2);

    app.fiber().dispose().await;
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

    let app = Context::new();
    app.plugin(
        AgentSpinePlugin,
        AgentSpineConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    );
    app.plugin(
        LlmRetryPlugin,
        RetryConfig {
            max_retries: 2,
            initial_delay: Duration::from_millis(5),
            ..Default::default()
        },
    );

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
    app.plugin(LlmRouter::plugin(), RouterConfig { providers });
    tick(30).await;

    let llm = app.llm().unwrap();
    let session = app.sessions().unwrap().create(None);

    // 1. 先走 completions（触发一次 429 重试后拿到 tool_call）
    let mut s1 = llm
        .stream(GenerateOptions {
            provider: Some("comp".into()),
            messages: vec![Message::user("现在几点")],
            ..Default::default()
        })
        .unwrap();
    let mut asm1 = dsh_llm::BlockAssembler::new();
    while let Some(chunk) = futures::StreamExt::next(&mut s1).await {
        asm1.push(chunk.unwrap());
    }
    assert_eq!(comp_calls.load(Ordering::SeqCst), 2);
    assert_eq!(asm1.tool_calls().len(), 1);

    session.append(dsh_session::SessionEvent::User {
        text: "现在几点".into(),
    });
    session.append(dsh_session::SessionEvent::Assistant {
        blocks: asm1.blocks(),
        replay: asm1.replay(),
        usage: asm1.usage(),
    });
    session.append(dsh_session::SessionEvent::ToolResult {
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
    let mut asm2 = dsh_llm::BlockAssembler::new();
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
    let mut asm3 = dsh_llm::BlockAssembler::new();
    while let Some(chunk) = futures::StreamExt::next(&mut s3).await {
        asm3.push(chunk.unwrap());
    }
    assert_eq!(asm3.text(), "from-responses");

    app.fiber().dispose().await;
}

#[tokio::test]
async fn agent_trajectory_detects_repeat_loop_and_recovers_after_guard_reminder() {
    struct LoopingAdapter(AtomicUsize);

    impl LlmAdapter for LoopingAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let step_idx = self.0.fetch_add(1, Ordering::SeqCst);
            let saw_reminder = options.messages.iter().any(|m| match m {
                Message::User { content } => content.contains("Carefully analyze"),
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

    let app = Context::new();
    let anomalies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let anomalies_cb = Arc::clone(&anomalies);
    app.on(EVENT_TRAJECTORY_ANOMALY, move |ev: &TrajectoryAnomalyEvent| {
        anomalies_cb.lock().unwrap().push(ev.finding.rule_id.clone());
    })
    .unwrap();

    app.plugin(
        AgentSpinePlugin,
        AgentSpineConfig::new(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    );
    app.plugin(
        PluginDef::sync("looping-adapter", |ctx, ()| {
            let llm = ctx.llm()?;
            ctx.effect("looping.reg", || {
                Ok::<_, dsh_llm::LlmError>(Some(
                    llm.register_adapter("looping", Arc::new(LoopingAdapter(AtomicUsize::new(0))))?,
                ))
            })?;
            Ok::<_, CordisError>(())
        })
        .inject(["llm"]),
        (),
    );
    tick(30).await;

    let session = app.sessions().unwrap().create(None);
    let reply = app
        .agent_loop()
        .unwrap()
        .run(Arc::clone(&session), "帮我查时间", CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(reply, "收到提醒，已跳出重复循环");

    let traj = app.service::<TrajectoryRuntime>("trajectory").unwrap();
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

    app.fiber().dispose().await;
}
