//! `mizar-subagent` 集成测试：验证后台常驻扇出、`send` 运行中 `Steer`/休眠唤醒/并发守卫、
//! `stop` 过渡重入保护、超时悬挂修复、零 Token 活性看门狗、实时进度与深度 1 硬护栏。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cordis::{Context, Fiber, PluginDef, Service};
use futures::channel::mpsc;
use mizar_agent_loop::{AgentLoop, AgentLoopConfig, AgentLoopContextExt, RunOptions, TurnInput};
use mizar_llm::{
    BoxChunkStream, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt,
    LlmError, Message, ModelInfo, ProviderInfo, RequestOrigin, StreamChunk, ToolCall,
};
use mizar_recall::{RecallConfig, RecallPlugin};
use mizar_session::{SessionContextExt, SessionOrigin, SessionStore, TitleSource};
use mizar_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
use mizar_session_title::{SessionTitleConfig, SessionTitleRuntime};
use mizar_subagent::{
    EVENT_SUBAGENT_STATE, SUBAGENT_TOOL_NAME, SubagentConfig, SubagentContextExt, SubagentMode,
    SubagentPlugin, SubagentRunStatus, SubagentStateEvent,
};
use mizar_system_prompt::{SystemPrompt, SystemPromptConfig, SystemPromptContextExt};
use mizar_tool_fs::{ToolFsConfig, ToolFsPlugin};
use mizar_tool_time::ToolTimePlugin;
use mizar_tools::{
    ToolAnnotations, ToolDefinition, ToolExecution, ToolExposure, ToolRuntime, ToolsContextExt,
};
use mizar_tools_coding::{CodingToolsConfig, CodingToolsPlugin};
use serde_json::json;
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "mizar_subagent_{tag}_{}_{nanos}",
        std::process::id()
    ))
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct RecordedCall {
    origin: RequestOrigin,
    provider: Option<String>,
    model: Option<String>,
    session_id: Option<String>,
    tools: Vec<String>,
}

struct StateWatch {
    events: Arc<Mutex<Vec<SubagentStateEvent>>>,
    seq_rx: tokio::sync::watch::Receiver<u64>,
    cursor: usize,
    _handle: cordis::DisposableHandle,
}

impl StateWatch {
    fn attach(app: &Context, parent_id: &str) -> Self {
        let events = Arc::new(Mutex::new(Vec::<SubagentStateEvent>::new()));
        let (seq_tx, seq_rx) = tokio::sync::watch::channel(0u64);
        let events_cb = Arc::clone(&events);
        let target_parent = parent_id.to_string();
        let seq_counter = Arc::new(AtomicUsize::new(0));
        let handle = app
            .on(EVENT_SUBAGENT_STATE, move |ev: &SubagentStateEvent| {
                if ev.parent_session_id == target_parent {
                    events_cb.lock().unwrap().push(ev.clone());
                    let next = seq_counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = seq_tx.send(next as u64);
                }
            })
            .unwrap();
        Self {
            events,
            seq_rx,
            cursor: 0,
            _handle: handle,
        }
    }

    fn advance_to_end(&mut self) {
        self.cursor = self.events.lock().unwrap().len();
    }

    async fn wait_for_event(
        &mut self,
        timeout_secs: u64,
        mut pred: impl FnMut(&SubagentStateEvent) -> bool,
    ) -> SubagentStateEvent {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
        loop {
            {
                let guard = self.events.lock().unwrap();
                while self.cursor < guard.len() {
                    let ev = &guard[self.cursor];
                    self.cursor += 1;
                    if pred(ev) {
                        return ev.clone();
                    }
                }
            }
            tokio::time::timeout_at(deadline, self.seq_rx.changed())
                .await
                .expect("timed out waiting for SubagentStateEvent")
                .unwrap();
        }
    }

    async fn wait_all_settled(&mut self, timeout_secs: u64) -> SubagentStateEvent {
        self.wait_for_event(timeout_secs, |ev| {
            ev.status.is_terminal() && ev.running_count == 0
        })
        .await
    }

    fn snapshot(&self) -> Vec<SubagentStateEvent> {
        self.events.lock().unwrap().clone()
    }
}

async fn setup_base_app(
    ws: &PathBuf,
    sub_cfg: SubagentConfig,
    use_sqlite: bool,
) -> (Context, Option<PathBuf>) {
    let (app, db_path, _sub_fiber) =
        setup_base_app_with_subagent_fiber(ws, sub_cfg, use_sqlite).await;
    (app, db_path)
}

async fn setup_base_app_with_subagent_fiber(
    ws: &PathBuf,
    sub_cfg: SubagentConfig,
    use_sqlite: bool,
) -> (Context, Option<PathBuf>, Fiber) {
    let app = Context::new();
    app.logger().set_silent();
    app.plugin(mizar_llm::LlmRuntime::plugin(), ())
        .await
        .unwrap();
    let db_path = if use_sqlite {
        let p = ws.join("sessions.sqlite");
        app.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&p))
            .await
            .unwrap();
        Some(p)
    } else {
        None
    };
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    app.plugin(
        AgentLoop::plugin(),
        AgentLoopConfig {
            provider: Some("main".into()),
            model: Some("main-1".into()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    app.plugin(CodingToolsPlugin, CodingToolsConfig::new(ws))
        .await
        .unwrap();
    let sub_fiber = app.plugin(SubagentPlugin, sub_cfg).await.unwrap();
    app.sessions()
        .unwrap()
        .set_default_cwd(Some(ws.display().to_string()));
    (app, db_path, sub_fiber)
}

#[tokio::test]
async fn parallel_read_only_subagents_isolate_history_and_persist_lineage() {
    let ws = temp_dir("parallel");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("alpha.txt"), "ALPHA_SECRET_42\n").unwrap();
    std::fs::write(ws.join("beta.txt"), "BETA_SECRET_99\n").unwrap();

    let (app, _db) = setup_base_app(&ws, SubagentConfig::default(), true).await;
    app.plugin(ToolTimePlugin, ()).await.unwrap();
    app.plugin(ToolFsPlugin, ToolFsConfig::new(&ws))
        .await
        .unwrap();
    app.plugin(RecallPlugin, RecallConfig::default())
        .await
        .unwrap();
    app.plugin(SessionTitleRuntime::plugin(), SessionTitleConfig::default())
        .await
        .unwrap();
    let custom_ro_calls = Arc::new(AtomicUsize::new(0));
    let custom_ro_ref = Arc::clone(&custom_ro_calls);
    app.plugin(
        PluginDef::sync("custom-ro", move |ctx, ()| {
            let tools = ctx.tools()?;
            let c = Arc::clone(&custom_ro_ref);
            let def = ToolDefinition::sync_tool(
                "custom_inspect",
                "Read-only custom inspection tool",
                json!({"type": "object", "properties": {}}),
                move |_, _| {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, mizar_tools::ToolError>("custom-ro-ok".to_string())
                },
            )
            .with_annotations(ToolAnnotations {
                read_only: true,
                ..Default::default()
            });
            ctx.effect("custom-ro.reg", move || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(def)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["tools"]),
        (),
    )
    .await
    .unwrap();

    let recorded = Arc::new(Mutex::new(Vec::<RecordedCall>::new()));
    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));
    let child_barrier = Arc::new(Barrier::new(2));
    let parent_step = Arc::new(AtomicUsize::new(0));

    struct ParallelMockAdapter {
        recorded: Arc<Mutex<Vec<RecordedCall>>>,
        in_flight: Arc<AtomicUsize>,
        max_concurrent: Arc<AtomicUsize>,
        child_barrier: Arc<Barrier>,
        parent_step: Arc<AtomicUsize>,
    }

    impl LlmAdapter for ParallelMockAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "main".into(),
                name: "Main".into(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: "main-1".into(),
                    name: "Main 1".into(),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let tool_names: Vec<String> = options.tools.iter().map(|t| t.name.clone()).collect();
            self.recorded.lock().unwrap().push(RecordedCall {
                origin: options.origin.clone(),
                provider: options.provider.clone(),
                model: options.model.clone(),
                session_id: options.session_id.clone(),
                tools: tool_names,
            });

            let (tx, rx) = mpsc::unbounded();
            let in_flight = Arc::clone(&self.in_flight);
            let max_concurrent = Arc::clone(&self.max_concurrent);
            let barrier = Arc::clone(&self.child_barrier);
            let parent_step = Arc::clone(&self.parent_step);
            tokio::spawn(async move {
                match &options.origin {
                    RequestOrigin::SessionTurn { .. } => {
                        let step = parent_step.fetch_add(1, Ordering::SeqCst);
                        if step == 0 {
                            let c1 = ToolCall {
                                id: "sub_1".into(),
                                name: "subagent".into(),
                                arguments: json!({
                                    "prompt": "读取 alpha.txt 里的暗号",
                                    "name": "scout-a"
                                })
                                .to_string(),
                            };
                            let c2 = ToolCall {
                                id: "sub_2".into(),
                                name: "subagent".into(),
                                arguments: json!({
                                    "prompt": "读取 beta.txt 里的暗号",
                                    "mode": "read_only"
                                })
                                .to_string(),
                            };
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(0, c1)));
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(1, c2)));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        } else if step == 1 {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("已后台派发")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let notice_text = options
                                .messages
                                .iter()
                                .rev()
                                .find_map(|m| match m {
                                    Message::User { content, .. } => Some(content.clone()),
                                    _ => None,
                                })
                                .unwrap_or_default();
                            let _ = tx.unbounded_send(Ok(StreamChunk::text(format!(
                                "父汇总: {notice_text}"
                            ))));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        }
                    }
                    RequestOrigin::Subagent => {
                        let last = options.messages.last();
                        if let Some(Message::Tool { content, .. }) = last {
                            let _ = tx
                                .unbounded_send(Ok(StreamChunk::text(format!("子结论:{content}"))));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                            max_concurrent.fetch_max(cur, Ordering::SeqCst);
                            let _ =
                                tokio::time::timeout(Duration::from_secs(3), barrier.wait()).await;
                            in_flight.fetch_sub(1, Ordering::SeqCst);

                            let user_text = options
                                .messages
                                .iter()
                                .rev()
                                .find_map(|m| match m {
                                    Message::User { content, .. } => Some(content.as_str()),
                                    _ => None,
                                })
                                .unwrap_or("");
                            let target = if user_text.contains("alpha") {
                                "alpha.txt"
                            } else {
                                "beta.txt"
                            };
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: format!("read_{target}"),
                                    name: "read".into(),
                                    arguments: json!({ "path": target }).to_string(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                    }
                    RequestOrigin::SessionTitle => {
                        let joined = options
                            .messages
                            .iter()
                            .map(Message::text_content)
                            .collect::<Vec<_>>()
                            .join("\n");
                        assert!(
                            !joined.contains("alpha.txt") && !joined.contains("beta.txt"),
                            "session-title must never run on subagent child prompts: {joined}"
                        );
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("EXP｜并发调研测试")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                    other => panic!("unexpected RequestOrigin: {other:?}"),
                }
            });
            Box::pin(rx)
        }
    }

    let adapter = Arc::new(ParallelMockAdapter {
        recorded: Arc::clone(&recorded),
        in_flight: Arc::clone(&in_flight),
        max_concurrent: Arc::clone(&max_concurrent),
        child_barrier: Arc::clone(&child_barrier),
        parent_step: Arc::clone(&parent_step),
    });
    let _llm_cleanup = app
        .llm()
        .unwrap()
        .register_adapter("main", adapter)
        .unwrap();

    let sessions = app.sessions().unwrap();
    let parent = sessions.create_with(mizar_session::SessionCreateOptions {
        id: Some("parent-parallel".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    let mut watch = StateWatch::attach(&app, &parent.id);
    let agent_loop = app.agent_loop().unwrap();

    let first_reply = agent_loop
        .run(
            Arc::clone(&parent),
            "并发调查 alpha 与 beta",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(first_reply, "已后台派发");

    watch.wait_all_settled(5).await;

    let merged_followup = agent_loop
        .inbox()
        .pop_followup(&parent.id)
        .expect("merged followup notice expected");
    assert!(agent_loop.inbox().pop_followup(&parent.id).is_none());

    let final_reply = agent_loop
        .run_with_options(
            Arc::clone(&parent),
            merged_followup,
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await
        .unwrap();

    assert!(
        final_reply.contains("ALPHA_SECRET_42"),
        "reply: {final_reply}"
    );
    assert!(
        final_reply.contains("BETA_SECRET_99"),
        "reply: {final_reply}"
    );
    assert_eq!(
        max_concurrent.load(Ordering::SeqCst),
        2,
        "two background subagent calls must execute concurrently"
    );

    // 父会话消息不被子会话内部工具调用污染
    let parent_msgs = parent.derive_messages();
    assert!(
        !parent_msgs.iter().any(|m| matches!(
            m,
            Message::Assistant { blocks, .. }
                if blocks.iter().any(|b| matches!(b, ContentBlock::ToolCall { name, .. } if name == "read"))
        )),
        "parent session must not contain child `read` tool calls"
    );

    tokio::time::sleep(Duration::from_millis(30)).await;

    let all_sessions = sessions.list();
    let children: Vec<_> = all_sessions
        .iter()
        .filter(|s| s.header().parent_id.as_deref() == Some("parent-parallel"))
        .collect();
    assert_eq!(children.len(), 2, "expected 2 child sessions");
    for child in &children {
        let h = child.header();
        assert!(
            h.title
                .as_deref()
                .is_some_and(|t| t.starts_with("Subagent")),
            "unexpected child title: {:?}",
            h.title
        );
        assert_eq!(h.title_source, Some(TitleSource::Manual));
        assert_eq!(h.origin, Some(SessionOrigin::Subagent));
        assert!(h.is_subagent_child());
        assert!(h.fork_seq.is_none());
        assert_eq!(h.cwd.as_deref(), parent.header().cwd.as_deref());
    }

    let child_ids: HashSet<String> = children.iter().map(|c| c.id.clone()).collect();
    {
        let calls = recorded.lock().unwrap();
        let child_calls: Vec<_> = calls
            .iter()
            .filter(|c| c.origin == RequestOrigin::Subagent)
            .collect();
        assert_eq!(child_calls.len(), 4);
        for c in &child_calls {
            assert_eq!(c.provider.as_deref(), Some("main"));
            assert_eq!(c.model.as_deref(), Some("main-1"));
            assert!(
                c.session_id
                    .as_deref()
                    .is_some_and(|sid| child_ids.contains(sid)),
                "child call must carry child session_id, got: {:?}",
                c.session_id
            );
            let tool_set: HashSet<&str> = c.tools.iter().map(String::as_str).collect();
            assert!(!tool_set.contains(SUBAGENT_TOOL_NAME));
            assert!(!tool_set.contains("write"));
            assert!(!tool_set.contains("edit"));
            assert!(!tool_set.contains("bash"));
            assert!(tool_set.contains("read"));
            assert!(tool_set.contains("grep"));
            assert!(tool_set.contains("find"));
            assert!(tool_set.contains("get_time"));
            assert!(tool_set.contains("list_dir"));
            assert!(tool_set.contains("read_file"));
            assert!(tool_set.contains("custom_inspect"));
        }
    }

    assert_eq!(app.subagent().unwrap().active_child_count(), 0);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn depth_one_guardrail_blocks_subagent_and_write_tools_in_read_only_mode() {
    let ws = temp_dir("guardrail_ro");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let child_seen_tools = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let nested_denied_seen = Arc::new(AtomicBool::new(false));
    let tools_ref = app.tools().unwrap();

    // 验证 subagent 暴露级别收紧为 ModelOnly（与 codemode 平级隔离）
    let sub_def = tools_ref
        .list_all()
        .into_iter()
        .find(|t| t.name == SUBAGENT_TOOL_NAME)
        .expect("subagent tool must be registered");
    assert_eq!(sub_def.exposure, ToolExposure::ModelOnly);

    struct RogueChildAdapter {
        step: AtomicUsize,
        child_seen_tools: Arc<Mutex<Vec<Vec<String>>>>,
        tools: Arc<ToolRuntime>,
        nested_denied_seen: Arc<AtomicBool>,
    }

    impl LlmAdapter for RogueChildAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            if options.origin == RequestOrigin::Subagent {
                self.child_seen_tools
                    .lock()
                    .unwrap()
                    .push(options.tools.iter().map(|t| t.name.clone()).collect());
                let step = self.step.fetch_add(1, Ordering::SeqCst);
                let tools = Arc::clone(&self.tools);
                let sid = options.session_id.clone();
                let nested_flag = Arc::clone(&self.nested_denied_seen);
                tokio::spawn(async move {
                    if step == 0 {
                        let nested_write = tools
                            .execute_nested(
                                ToolExecution::new(
                                    "write",
                                    json!({"path": "hack.txt", "content": "nope"}),
                                    sid.clone(),
                                ),
                                CancellationToken::new(),
                                None,
                            )
                            .await;
                        let nested_sub = tools
                            .execute_nested(
                                ToolExecution::new("subagent", json!({"prompt": "recursive"}), sid),
                                CancellationToken::new(),
                                None,
                            )
                            .await;
                        if nested_write.blocked && nested_sub.blocked {
                            nested_flag.store(true, Ordering::SeqCst);
                        }

                        let calls = vec![
                            ToolCall {
                                id: "bad_sub".into(),
                                name: "subagent".into(),
                                arguments: json!({"prompt": "nested"}).to_string(),
                            },
                            ToolCall {
                                id: "bad_write".into(),
                                name: "write".into(),
                                arguments: json!({"path": "evil.txt", "content": "x"}).to_string(),
                            },
                            ToolCall {
                                id: "bad_bash".into(),
                                name: "bash".into(),
                                arguments: json!({"command": "echo hacked"}).to_string(),
                            },
                        ];
                        for (i, c) in calls.into_iter().enumerate() {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(i, c)));
                        }
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    } else {
                        let tool_results: Vec<String> = options
                            .messages
                            .iter()
                            .filter_map(|m| match m {
                                Message::Tool { content, .. } => Some(content.clone()),
                                _ => None,
                            })
                            .collect();
                        let _ = tx.unbounded_send(Ok(StreamChunk::text(tool_results.join(" || "))));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                });
            }
            Box::pin(rx)
        }
    }

    let adapter = Arc::new(RogueChildAdapter {
        step: AtomicUsize::new(0),
        child_seen_tools: Arc::clone(&child_seen_tools),
        tools: Arc::clone(&tools_ref),
        nested_denied_seen: Arc::clone(&nested_denied_seen),
    });
    let _reg = app
        .llm()
        .unwrap()
        .register_adapter("main", adapter)
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);
    let res = tools_ref
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "try prohibited ops" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!res.is_error, "unexpected error: {}", res.content);
    let child_id = res.details.unwrap()["childSessionId"]
        .as_str()
        .unwrap()
        .to_string();

    watch.wait_all_settled(5).await;
    let followup = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .expect("settled notice expected");
    let body = followup.text();

    assert!(nested_denied_seen.load(Ordering::SeqCst));
    assert!(
        body.contains("已拦截") && body.contains("subagent"),
        "{body}"
    );
    assert!(
        body.contains("只读模式子代理禁止调用非只读工具 write"),
        "{body}"
    );
    assert!(
        body.contains("只读模式子代理禁止调用非只读工具 bash"),
        "{body}"
    );
    assert!(!ws.join("evil.txt").exists());
    assert!(!ws.join("hack.txt").exists());

    {
        let seen = child_seen_tools.lock().unwrap();
        assert!(!seen.is_empty());
        for list in seen.iter() {
            assert!(!list.contains(&"subagent".to_string()));
            assert!(!list.contains(&"write".to_string()));
            assert!(!list.contains(&"edit".to_string()));
            assert!(!list.contains(&"bash".to_string()));
        }
    }

    let replay_res = tools_ref
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "second hop" }),
            Some(child_id),
        ))
        .await;
    assert!(replay_res.is_error);
    assert!(replay_res.blocked);

    assert_eq!(app.subagent().unwrap().active_child_count(), 0);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn edit_mode_allows_write_and_model_override_while_blocking_subagent_recursion() {
    let ws = temp_dir("edit_mode");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    type DualCallLog = Vec<(String, Option<String>, Vec<String>)>;

    struct DualAdapter {
        id: &'static str,
        step: AtomicUsize,
        seen: Arc<Mutex<DualCallLog>>,
    }

    impl LlmAdapter for DualAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: self.id.into(),
                name: self.id.into(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: format!("{}-model", self.id),
                    name: format!("{} Model", self.id),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let tool_names: Vec<String> = options.tools.iter().map(|t| t.name.clone()).collect();
            self.seen.lock().unwrap().push((
                self.id.to_string(),
                options.model.clone(),
                tool_names,
            ));
            let step = self.step.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                if step == 0 {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "w1".into(),
                            name: "write".into(),
                            arguments: json!({
                                "path": "created_by_child.txt",
                                "content": "edit-mode-ok"
                            })
                            .to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        1,
                        ToolCall {
                            id: "s1".into(),
                            name: "subagent".into(),
                            arguments: json!({"prompt": "nested in edit"}).to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                } else {
                    let tool_results: Vec<String> = options
                        .messages
                        .iter()
                        .filter_map(|m| match m {
                            Message::Tool { content, .. } => Some(content.clone()),
                            _ => None,
                        })
                        .collect();
                    let _ = tx.unbounded_send(Ok(StreamChunk::text(format!(
                        "编辑完成: {}",
                        tool_results.join(" | ")
                    ))));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let _r1 = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(DualAdapter {
                id: "main",
                step: AtomicUsize::new(0),
                seen: Arc::clone(&seen),
            }),
        )
        .unwrap();
    let _r2 = app
        .llm()
        .unwrap()
        .register_adapter(
            "alt",
            Arc::new(DualAdapter {
                id: "alt",
                step: AtomicUsize::new(0),
                seen: Arc::clone(&seen),
            }),
        )
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);
    let res = app
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({
                "prompt": "创建 created_by_child.txt",
                "mode": SubagentMode::Edit.as_str(),
                "model": "alt/alt-model"
            }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!res.is_error, "unexpected error: {}", res.content);

    watch.wait_all_settled(5).await;
    let followup = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .expect("edit notice expected");
    let body = followup.text();

    assert_eq!(
        std::fs::read_to_string(ws.join("created_by_child.txt")).unwrap(),
        "edit-mode-ok"
    );
    assert!(
        body.contains("已拦截") && body.contains("subagent"),
        "{body}"
    );

    {
        let recorded = seen.lock().unwrap();
        assert_eq!(recorded[0].0, "alt");
        assert_eq!(recorded[0].1.as_deref(), Some("alt-model"));
        assert!(recorded[0].2.contains(&"write".to_string()));
        assert!(!recorded[0].2.contains(&"subagent".to_string()));
    }

    assert_eq!(app.subagent().unwrap().active_child_count(), 0);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn pre_cancelled_spawn_is_rejected_and_parent_cleanup_cancels_worker() {
    let ws = temp_dir("cancel_cascade");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let (started_tx, mut started_rx) = tokio::sync::watch::channel(false);
    let child_cancelled_observed = Arc::new(AtomicBool::new(false));

    struct SlowAdapter {
        started_tx: tokio::sync::watch::Sender<bool>,
        cancelled_observed: Arc<AtomicBool>,
    }

    impl LlmAdapter for SlowAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let started_tx = self.started_tx.clone();
            let obs = Arc::clone(&self.cancelled_observed);
            tokio::spawn(async move {
                let _ = started_tx.send(true);
                options.cancel.cancelled().await;
                obs.store(true, Ordering::SeqCst);
                let _ = tx.unbounded_send(Err(LlmError::cancelled()));
            });
            Box::pin(rx)
        }
    }

    let _reg = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(SlowAdapter {
                started_tx,
                cancelled_observed: Arc::clone(&child_cancelled_observed),
            }),
        )
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let tools = app.tools().unwrap();
    let sub = app.subagent().unwrap();

    // 1. 传入已取消的 call_ctx.cancel 直接拒绝启动
    let pre_cancel = CancellationToken::new();
    pre_cancel.cancel();
    let denied = tools
        .execute_with(
            ToolExecution::new(
                SUBAGENT_TOOL_NAME,
                json!({"prompt": "should cancel"}),
                Some(parent.id.clone()),
            ),
            pre_cancel,
            None,
        )
        .await;
    assert!(denied.is_error);
    assert!(denied.content.contains("取消"));
    assert_eq!(sub.active_child_count(), 0);

    // 2. 启动后调用 cancel_parent_subagents 立刻取消后台 Worker 与子 LLM 请求
    let launched = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({"prompt": "long task"}),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!launched.is_error, "{}", launched.content);
    tokio::time::timeout(Duration::from_secs(3), started_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(sub.active_child_count(), 1);

    assert_eq!(sub.cancel_parent_subagents(&parent.id), 1);
    tokio::time::timeout(Duration::from_secs(3), async {
        while !child_cancelled_observed.load(Ordering::SeqCst) || sub.active_child_count() > 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("parent cleanup must cancel worker and clean active children");

    app.fiber().dispose().await;
}

#[tokio::test]
async fn enforces_max_steps_cap_and_timeout_and_validates_args() {
    assert_eq!(
        SubagentConfig {
            default_max_steps: 99,
            max_steps_cap: 40,
            timeout_secs: 180,
            ..SubagentConfig::default()
        }
        .effective_max_steps(),
        40
    );

    // 1. max_steps 上限验证（单轮 steps/max_steps 格式）
    let ws1 = temp_dir("max_steps");
    let _g1 = DirGuard(ws1.clone());
    std::fs::create_dir_all(&ws1).unwrap();
    let (app1, _) = setup_base_app(
        &ws1,
        SubagentConfig {
            default_max_steps: 10,
            max_steps_cap: 2,
            timeout_secs: 180,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    struct EndlessToolAdapter {
        calls: Arc<AtomicUsize>,
    }

    impl LlmAdapter for EndlessToolAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                    0,
                    ToolCall {
                        id: format!("loop_{n}"),
                        name: "find".into(),
                        arguments: json!({"pattern": "*"}).to_string(),
                    },
                )));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::ToolCalls,
                    replay: None,
                }));
            });
            Box::pin(rx)
        }
    }

    let step_calls = Arc::new(AtomicUsize::new(0));
    let _reg1 = app1
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(EndlessToolAdapter {
                calls: Arc::clone(&step_calls),
            }),
        )
        .unwrap();

    let tools1 = app1.tools().unwrap();
    let parent1 = app1.sessions().unwrap().create(None);

    let bad_prompt = tools1
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "   " }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(bad_prompt.is_error);
    assert!(bad_prompt.content.contains("prompt"));

    let bad_mode = tools1
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "ok", "mode": "yolo" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(bad_mode.is_error);
    assert!(bad_mode.content.contains("无效的 mode 参数"));

    let mut watch1 = StateWatch::attach(&app1, &parent1.id);
    let max_spawn = tools1
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "keep searching" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(!max_spawn.is_error, "{}", max_spawn.content);

    let ev1 = watch1.wait_all_settled(5).await;
    assert_eq!(ev1.status, SubagentRunStatus::MaxSteps);
    assert!(
        ev1.notice
            .as_deref()
            .is_some_and(|n| n.contains("单轮上限 2") && n.contains("累计 2 步")),
        "{:?}",
        ev1.notice
    );
    assert_eq!(step_calls.load(Ordering::SeqCst), 2);
    let f1 = app1
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent1.id)
        .unwrap();
    assert!(
        f1.text().contains("Reached max_steps limit (2/2)"),
        "{}",
        f1.text()
    );
    assert_eq!(app1.subagent().unwrap().active_child_count(), 0);
    app1.fiber().dispose().await;

    // 2. timeout_secs 超时封顶验证
    let ws2 = temp_dir("timeout");
    let _g2 = DirGuard(ws2.clone());
    std::fs::create_dir_all(&ws2).unwrap();
    let (app2, _) = setup_base_app(
        &ws2,
        SubagentConfig {
            default_max_steps: 20,
            max_steps_cap: 40,
            timeout_secs: 1,
            stall_timeout_secs: 0,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    struct HangingAdapter;
    impl LlmAdapter for HangingAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                tokio::select! {
                    _ = options.cancel.cancelled() => {
                        let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    }
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {}
                }
            });
            Box::pin(rx)
        }
    }

    let _reg2 = app2
        .llm()
        .unwrap()
        .register_adapter("main", Arc::new(HangingAdapter))
        .unwrap();
    let parent2 = app2.sessions().unwrap().create(None);
    let mut watch2 = StateWatch::attach(&app2, &parent2.id);

    let t_spawn = app2
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "hang" }),
            Some(parent2.id.clone()),
        ))
        .await;
    assert!(!t_spawn.is_error, "{}", t_spawn.content);

    let ev2 = watch2.wait_all_settled(5).await;
    assert_eq!(ev2.status, SubagentRunStatus::TimedOut);
    let f2 = app2
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent2.id)
        .unwrap();
    assert!(
        f2.text().contains("timed out after 1s limit"),
        "{}",
        f2.text()
    );
    assert_eq!(app2.subagent().unwrap().active_child_count(), 0);
    app2.fiber().dispose().await;
}

#[tokio::test]
async fn parent_turn_route_is_inherited_without_session_selection() {
    let ws = temp_dir("inherit_route");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    type RouteCheckLog = Vec<(String, Option<String>, RequestOrigin)>;

    struct RouteCheckAdapter {
        id: &'static str,
        seen: Arc<Mutex<RouteCheckLog>>,
    }

    impl LlmAdapter for RouteCheckAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: self.id.into(),
                name: self.id.into(),
                protocol: "mock".into(),
                models: vec![],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.seen.lock().unwrap().push((
                self.id.to_string(),
                options.model.clone(),
                options.origin.clone(),
            ));
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                match &options.origin {
                    RequestOrigin::SessionTurn { .. } => {
                        if matches!(options.messages.last(), Some(Message::Tool { .. })) {
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("parent-done")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: "c1".into(),
                                    name: "subagent".into(),
                                    arguments: json!({"prompt": "inherit model"}).to_string(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                    }
                    RequestOrigin::Subagent => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("child-done")));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                    other => panic!("unexpected origin: {other:?}"),
                }
            });
            Box::pin(rx)
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let _r1 = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(RouteCheckAdapter {
                id: "main",
                seen: Arc::clone(&seen),
            }),
        )
        .unwrap();
    let _r2 = app
        .llm()
        .unwrap()
        .register_adapter(
            "custom",
            Arc::new(RouteCheckAdapter {
                id: "custom",
                seen: Arc::clone(&seen),
            }),
        )
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);
    let reply = app
        .agent_loop()
        .unwrap()
        .run_with_options(
            Arc::clone(&parent),
            "test inherit",
            CancellationToken::new(),
            RunOptions {
                provider: Some("custom".into()),
                model: Some("custom-m".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply, "parent-done");

    watch.wait_all_settled(5).await;

    {
        let calls = seen.lock().unwrap();
        let child_call = calls
            .iter()
            .find(|(_, _, o)| *o == RequestOrigin::Subagent)
            .expect("expected child call");
        assert_eq!(child_call.0, "custom");
        assert_eq!(child_call.1.as_deref(), Some("custom-m"));
    }

    app.fiber().dispose().await;
}

#[tokio::test]
async fn background_spawn_batches_parallel_followups_and_emits_state_events() {
    let ws = temp_dir("bg_fire_and_notify");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("alpha.txt"), "BG_ALPHA_VALUE\n").unwrap();
    std::fs::write(ws.join("beta.txt"), "BG_BETA_VALUE\n").unwrap();

    let (app, _db) = setup_base_app(&ws, SubagentConfig::default(), true).await;

    let sys_prompt = app.system_prompt().unwrap().build();
    assert!(
        sys_prompt.contains("end your turn instead of sleeping or polling"),
        "system prompt missing background contract:\n{sys_prompt}"
    );

    let (gate_tx, gate_rx) = tokio::sync::watch::channel(false);
    let child_started = Arc::new(AtomicUsize::new(0));
    let (started_tx, mut started_rx) = tokio::sync::watch::channel(0usize);

    struct BgAdapter {
        parent_calls: Arc<AtomicUsize>,
        seen_immediate_tools: Arc<Mutex<Vec<String>>>,
        gate_rx: tokio::sync::watch::Receiver<bool>,
        child_started: Arc<AtomicUsize>,
        started_tx: tokio::sync::watch::Sender<usize>,
        seen_parent_followup: Arc<Mutex<Option<String>>>,
    }

    impl LlmAdapter for BgAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let parent_calls = Arc::clone(&self.parent_calls);
            let seen_immediate = Arc::clone(&self.seen_immediate_tools);
            let mut gate_rx = self.gate_rx.clone();
            let started = Arc::clone(&self.child_started);
            let started_tx = self.started_tx.clone();
            let seen_followup = Arc::clone(&self.seen_parent_followup);

            tokio::spawn(async move {
                match options.origin {
                    RequestOrigin::SessionTurn { .. } => {
                        let call_idx = parent_calls.fetch_add(1, Ordering::SeqCst);
                        if call_idx == 0 {
                            let c1 = ToolCall {
                                id: "bg_1".into(),
                                name: "subagent".into(),
                                arguments: json!({
                                    "prompt": "读取 alpha.txt",
                                    "name": "scout-a"
                                })
                                .to_string(),
                            };
                            let c2 = ToolCall {
                                id: "bg_2".into(),
                                name: "subagent".into(),
                                arguments: json!({
                                    "prompt": "读取 beta.txt"
                                })
                                .to_string(),
                            };
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(0, c1)));
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(1, c2)));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        } else if call_idx == 1 {
                            let tool_results: Vec<String> = options
                                .messages
                                .iter()
                                .filter_map(|m| match m {
                                    Message::Tool { content, .. } => Some(content.clone()),
                                    _ => None,
                                })
                                .collect();
                            *seen_immediate.lock().unwrap() = tool_results;
                            let _ = tx.unbounded_send(Ok(StreamChunk::text(
                                "已派发后台子代理，主轮次先行结束",
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let notice_msg = options.messages.iter().rev().find_map(|m| match m {
                                Message::User { content, .. } => Some(content.clone()),
                                _ => None,
                            });
                            *seen_followup.lock().unwrap() = notice_msg;
                            let _ = tx.unbounded_send(Ok(StreamChunk::text("主代理已汇总结果")));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        }
                    }
                    RequestOrigin::Subagent => {
                        let last = options.messages.last();
                        if let Some(Message::Tool { content, .. }) = last {
                            let tag = if content.contains("BG_ALPHA_VALUE") {
                                "ALPHA_OK"
                            } else {
                                "BETA_OK"
                            };
                            let _ = tx.unbounded_send(Ok(StreamChunk::text(tag)));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let count = started.fetch_add(1, Ordering::SeqCst) + 1;
                            let _ = started_tx.send(count);
                            let _ = gate_rx.wait_for(|&open| open).await;
                            let user_text = options
                                .messages
                                .iter()
                                .rev()
                                .find_map(|m| match m {
                                    Message::User { content, .. } => Some(content.as_str()),
                                    _ => None,
                                })
                                .unwrap_or("");
                            let path = if user_text.contains("alpha") {
                                "alpha.txt"
                            } else {
                                "beta.txt"
                            };
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: format!("rd_{path}"),
                                    name: "read".into(),
                                    arguments: json!({ "path": path }).to_string(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        }
                    }
                    _ => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    }
                }
            });
            Box::pin(rx)
        }
    }

    let seen_immediate = Arc::new(Mutex::new(Vec::new()));
    let seen_followup = Arc::new(Mutex::new(None));
    let _llm_reg = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(BgAdapter {
                parent_calls: Arc::new(AtomicUsize::new(0)),
                seen_immediate_tools: Arc::clone(&seen_immediate),
                gate_rx,
                child_started: Arc::clone(&child_started),
                started_tx,
                seen_parent_followup: Arc::clone(&seen_followup),
            }),
        )
        .unwrap();

    let agent_loop = app.agent_loop().unwrap();
    let mut wake_rx = agent_loop.inbox().subscribe_wake();
    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    let first_reply = agent_loop
        .run_with_options(
            Arc::clone(&parent),
            "并行调研这两个文件",
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(first_reply, "已派发后台子代理，主轮次先行结束");

    tokio::time::timeout(Duration::from_secs(3), started_rx.wait_for(|&n| n == 2))
        .await
        .expect("both background children should start running")
        .unwrap();
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent.id),
        2
    );
    {
        let imm = seen_immediate.lock().unwrap();
        assert_eq!(imm.len(), 2);
        assert!(imm.iter().any(|s| s.contains("scout-a")));
        assert!(imm.iter().any(|s| s.contains("agent-1")));
    }

    let _ = gate_tx.send(true);
    watch.wait_all_settled(5).await;
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent.id),
        0
    );
    assert_eq!(app.subagent().unwrap().active_child_count(), 0);

    let woken_sid = tokio::time::timeout(Duration::from_secs(3), wake_rx.recv())
        .await
        .expect("WakeDriver should receive notification")
        .unwrap();
    assert_eq!(woken_sid, parent.id);

    // 两个并发子代理结算合并成单条 subagent:batch FollowUp，只消耗 1 格自动唤醒预算
    assert_eq!(agent_loop.inbox().auto_wakes_since_human(&parent.id), 1);
    let batch_notice = agent_loop
        .inbox()
        .pop_followup(&parent.id)
        .expect("merged FollowUp notice must arrive");
    assert!(agent_loop.inbox().pop_followup(&parent.id).is_none());

    match &batch_notice {
        TurnInput::Notice { source, text, .. } => {
            assert_eq!(source, "subagent:batch");
            assert!(text.contains("ALPHA_OK"), "{text}");
            assert!(text.contains("BETA_OK"), "{text}");
        }
        other => panic!("expected Notice, got {other:?}"),
    }

    let next_reply = agent_loop
        .run_with_options(
            Arc::clone(&parent),
            batch_notice,
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(next_reply, "主代理已汇总结果");
    assert!(
        seen_followup
            .lock()
            .unwrap()
            .as_deref()
            .is_some_and(|t| t.contains("ALPHA_OK") && t.contains("BETA_OK"))
    );

    let snapshots = app
        .subagent()
        .unwrap()
        .list_subagents_for_parent(&parent.id);
    assert_eq!(snapshots.len(), 2);
    assert!(
        snapshots
            .iter()
            .all(|s| s.status == SubagentRunStatus::Completed && s.steps == 2)
    );

    let status_out = app
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "status" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!status_out.is_error, "{}", status_out.content);
    assert!(
        status_out.content.contains("ALPHA_OK") && status_out.content.contains("BETA_OK"),
        "status text should expose terminal summary: {}",
        status_out.content
    );

    let max_running_seen = watch
        .snapshot()
        .into_iter()
        .map(|e| e.running_count)
        .max()
        .unwrap_or(0);
    assert_eq!(max_running_seen, 2);

    app.fiber().dispose().await;
}

#[tokio::test]
async fn background_subagent_survives_parent_turn_cancel_and_supports_status_and_stop() {
    let ws = temp_dir("bg_survive_and_stop");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let (step2_tx, mut step2_rx) = tokio::sync::watch::channel(false);
    let (child_started_tx, mut child_started_rx) = tokio::sync::watch::channel(false);
    let child_cancelled = Arc::new(AtomicBool::new(false));

    struct SurviveStopAdapter {
        parent_calls: Arc<AtomicUsize>,
        step2_tx: tokio::sync::watch::Sender<bool>,
        child_started_tx: tokio::sync::watch::Sender<bool>,
        child_cancelled: Arc<AtomicBool>,
    }

    impl LlmAdapter for SurviveStopAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let p_calls = Arc::clone(&self.parent_calls);
            let step2_tx = self.step2_tx.clone();
            let child_started_tx = self.child_started_tx.clone();
            let child_cancelled = Arc::clone(&self.child_cancelled);

            tokio::spawn(async move {
                match options.origin {
                    RequestOrigin::SessionTurn { .. } => {
                        let idx = p_calls.fetch_add(1, Ordering::SeqCst);
                        if idx == 0 {
                            let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                                0,
                                ToolCall {
                                    id: "bg_worker".into(),
                                    name: "subagent".into(),
                                    arguments: json!({
                                        "prompt": "long background task",
                                        "name": "worker-x"
                                    })
                                    .to_string(),
                                },
                            )));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::ToolCalls,
                                replay: None,
                            }));
                        } else {
                            let _ = step2_tx.send(true);
                            options.cancel.cancelled().await;
                            let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                        }
                    }
                    RequestOrigin::Subagent => {
                        let _ = child_started_tx.send(true);
                        options.cancel.cancelled().await;
                        child_cancelled.store(true, Ordering::SeqCst);
                        let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    }
                    _ => {}
                }
            });
            Box::pin(rx)
        }
    }

    let _llm_reg = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(SurviveStopAdapter {
                parent_calls: Arc::new(AtomicUsize::new(0)),
                step2_tx,
                child_started_tx,
                child_cancelled: Arc::clone(&child_cancelled),
            }),
        )
        .unwrap();

    let agent_loop = app.agent_loop().unwrap();
    let tools = app.tools().unwrap();
    let parent = app.sessions().unwrap().create(None);
    let parent_cancel = CancellationToken::new();

    let parent_loop = Arc::clone(&agent_loop);
    let parent_sess = Arc::clone(&parent);
    let parent_tok = parent_cancel.clone();
    let run_handle = tokio::spawn(async move {
        parent_loop
            .run_with_options(
                parent_sess,
                "start worker",
                parent_tok,
                RunOptions::default(),
            )
            .await
    });

    tokio::time::timeout(Duration::from_secs(3), step2_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), child_started_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();

    parent_cancel.cancel();
    let _ = run_handle.await.unwrap();
    tokio::task::yield_now().await;

    assert!(
        !child_cancelled.load(Ordering::SeqCst),
        "cancelling foreground parent turn must not kill background subagent"
    );
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent.id),
        1
    );

    let status_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "status" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!status_out.is_error, "{}", status_out.content);
    assert!(status_out.content.contains("worker-x [running]"));
    assert!(status_out.content.contains("step=1"));

    let stop_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "stop", "name": "worker-x" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!stop_out.is_error, "{}", stop_out.content);
    assert!(stop_out.content.contains("Stopped subagent(s): worker-x"));
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent.id),
        0
    );

    tokio::time::timeout(Duration::from_secs(3), async {
        while !child_cancelled.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("explicit stop must cancel child worker");

    assert!(agent_loop.inbox().pop_followup(&parent.id).is_none());

    app.fiber().dispose().await;
}

#[tokio::test]
async fn background_concurrency_limit_and_parent_session_switch_cleanup() {
    let ws = temp_dir("bg_limit_switch");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _, sub_fiber) = setup_base_app_with_subagent_fiber(
        &ws,
        SubagentConfig {
            max_concurrent_per_parent: 2,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    struct BlockingChildAdapter {
        cancelled_count: Arc<AtomicUsize>,
        started_count: Arc<AtomicUsize>,
    }

    impl LlmAdapter for BlockingChildAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let cancelled = Arc::clone(&self.cancelled_count);
            let started = Arc::clone(&self.started_count);
            tokio::spawn(async move {
                started.fetch_add(1, Ordering::SeqCst);
                options.cancel.cancelled().await;
                cancelled.fetch_add(1, Ordering::SeqCst);
                let _ = tx.unbounded_send(Err(LlmError::cancelled()));
            });
            Box::pin(rx)
        }
    }

    let started = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let _llm_reg = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(BlockingChildAdapter {
                cancelled_count: Arc::clone(&cancelled),
                started_count: Arc::clone(&started),
            }),
        )
        .unwrap();

    let tools = app.tools().unwrap();
    let sessions = app.sessions().unwrap();
    let parent1 = sessions.create(None);
    let parent2 = sessions.create(None);
    let dispose_events = Arc::new(Mutex::new(Vec::<SubagentStateEvent>::new()));
    let dispose_events_cb = Arc::clone(&dispose_events);
    let parent2_id = parent2.id.clone();
    let _dispose_state_h = app
        .on(EVENT_SUBAGENT_STATE, move |ev: &SubagentStateEvent| {
            if ev.parent_session_id == parent2_id {
                dispose_events_cb.lock().unwrap().push(ev.clone());
            }
        })
        .unwrap();

    let r1 = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "task 1", "name": "c1" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(!r1.is_error, "{}", r1.content);

    let dup = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "dup", "name": "c1" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(dup.is_error);
    assert!(dup.content.contains("正在运行中"));

    let bad_name = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "bad", "name": "all" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(bad_name.is_error);
    assert!(bad_name.content.contains("无效的 name 参数"));

    let r2 = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "task 2", "name": "c2" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(!r2.is_error, "{}", r2.content);

    let r3 = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "task 3", "name": "c3" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(r3.is_error);
    assert!(r3.content.contains("并发数已达上限"));

    let other_parent = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "other parent task", "name": "p2" }),
            Some(parent2.id.clone()),
        ))
        .await;
    assert!(!other_parent.is_error, "{}", other_parent.content);

    tokio::time::timeout(Duration::from_secs(3), async {
        while started.load(Ordering::SeqCst) < 3 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    assert_eq!(
        app.subagent().unwrap().cancel_parent_subagents(&parent1.id),
        2
    );
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent1.id),
        0
    );
    assert_eq!(
        app.subagent()
            .unwrap()
            .running_subagents_for_parent(&parent2.id),
        1
    );

    tokio::time::timeout(Duration::from_secs(3), async {
        while cancelled.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(cancelled.load(Ordering::SeqCst), 2);

    sub_fiber.dispose().await;
    assert!(
        dispose_events
            .lock()
            .unwrap()
            .iter()
            .any(|ev| ev.status == SubagentRunStatus::Stopped && ev.running_count == 0),
        "dispose must emit Stopped event with running_count=0: {:?}",
        dispose_events.lock().unwrap()
    );
    tokio::time::timeout(Duration::from_secs(3), async {
        while cancelled.load(Ordering::SeqCst) < 3 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    app.fiber().dispose().await;
}

#[tokio::test]
async fn send_wakes_settled_subagent_preserving_memory_enforces_mode_and_concurrency_cap() {
    let ws = temp_dir("send_settled_resume");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(
        &ws,
        SubagentConfig {
            max_concurrent_per_parent: 1,
            ..SubagentConfig::default()
        },
        true,
    )
    .await;

    struct MemoryAdapter {
        step: AtomicUsize,
        block_gate: tokio::sync::watch::Receiver<bool>,
    }

    impl LlmAdapter for MemoryAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let step = self.step.fetch_add(1, Ordering::SeqCst);
            let mut gate = self.block_gate.clone();
            tokio::spawn(async move {
                let last_user = options
                    .messages
                    .iter()
                    .rev()
                    .find_map(|m| match m {
                        Message::User { content, .. } => Some(content.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();

                if last_user.contains("hold-slot") {
                    let _ = gate.wait_for(|&open| open).await;
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("holder done")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                    return;
                }

                // 验证并发二次 send 期间，只读分身仍尝试写文件以实证护栏未被破坏
                if last_user.contains("concurrent write attempt") {
                    let try_write_res = options.messages.iter().rev().find_map(|m| match m {
                        Message::Tool {
                            tool_call_id,
                            content,
                            ..
                        } if tool_call_id == "try_write" => Some(content.clone()),
                        _ => None,
                    });
                    if let Some(last_res) = try_write_res {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text(format!(
                            "WRITE_CHECK:{last_res}"
                        ))));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
                            replay: None,
                        }));
                    } else {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "try_write".into(),
                                name: "write".into(),
                                arguments: json!({
                                    "path": "must_not_exist.txt",
                                    "content": "hacked"
                                })
                                .to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    return;
                }

                // 第二轮唤醒时检查先前的对话记忆仍在 options.messages 中
                let remembered = options
                    .messages
                    .iter()
                    .any(|m| m.text_content().contains("MAGIC-42"));
                let reply = if remembered && step > 0 {
                    format!("RECALLED_MAGIC_42 | q={last_user}")
                } else {
                    "STORED_FIRST_ROUND".to_string()
                };
                let _ = tx.unbounded_send(Ok(StreamChunk::text(reply)));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            });
            Box::pin(rx)
        }
    }

    let (hold_tx, hold_rx) = tokio::sync::watch::channel(false);
    let _llm = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(MemoryAdapter {
                step: AtomicUsize::new(0),
                block_gate: hold_rx,
            }),
        )
        .unwrap();

    let tools = app.tools().unwrap();
    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    // 1. 首轮 spawn 记住 MAGIC-42
    let spawn_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "mem-scout", "prompt": "remember MAGIC-42" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!spawn_out.is_error, "{}", spawn_out.content);
    let child_id_1 = spawn_out.details.unwrap()["childSessionId"]
        .as_str()
        .unwrap()
        .to_string();
    watch.wait_all_settled(5).await;
    let _ = app.agent_loop().unwrap().inbox().pop_followup(&parent.id);

    // 2. 尝试通过 send 提权到 edit 必须报错拒绝
    let escalate_err = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "mem-scout", "mode": "edit", "prompt": "escalate" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(escalate_err.is_error);
    assert!(escalate_err.content.contains("不允许修改"));

    // 3. 正常 send 唤醒已休眠分身：沿用同 child_session_id 且记忆还在
    let send_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "mem-scout", "prompt": "what was magic?" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!send_out.is_error, "{}", send_out.content);
    let send_details = send_out.details.unwrap();
    assert_eq!(send_details["delivery"], "resume");
    assert_eq!(send_details["childSessionId"], child_id_1);

    watch.wait_all_settled(5).await;
    let f2 = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .unwrap();
    assert!(f2.text().contains("RECALLED_MAGIC_42"), "{}", f2.text());
    let snaps = app
        .subagent()
        .unwrap()
        .list_subagents_for_parent(&parent.id);
    assert_eq!(
        snaps[0].steps, 2,
        "cumulative steps must add across resumes"
    );

    // 4. 并发两次 send 同一休眠分身：先拿锁方从 Settled 唤醒 (resume)，后拿锁方转为 Steer 插话，且 read_only 护栏绝不被破坏
    let (r_a, r_b) = tokio::join!(
        tools.execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "mem-scout", "prompt": "concurrent write attempt A" }),
            Some(parent.id.clone()),
        )),
        tools.execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "mem-scout", "prompt": "concurrent write attempt B" }),
            Some(parent.id.clone()),
        )),
    );
    assert!(!r_a.is_error, "{}", r_a.content);
    assert!(!r_b.is_error, "{}", r_b.content);
    let mut deliveries = vec![
        r_a.details.unwrap()["delivery"]
            .as_str()
            .unwrap()
            .to_string(),
        r_b.details.unwrap()["delivery"]
            .as_str()
            .unwrap()
            .to_string(),
    ];
    deliveries.sort();
    assert_eq!(deliveries, vec!["resume".to_string(), "steer".to_string()]);

    watch.wait_all_settled(5).await;
    let f_conc = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .unwrap();
    assert!(
        f_conc
            .text()
            .contains("只读模式子代理禁止调用非只读工具 write"),
        "guardrail must stay intact under concurrent send: {}",
        f_conc.text()
    );
    assert!(!ws.join("must_not_exist.txt").exists());

    // 5. 并发已占满时（cap=1）尝试 send 唤醒休眠分身必须被 ConcurrencyLimitReached 拒绝
    let hold_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "holder", "prompt": "hold-slot" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!hold_out.is_error, "{}", hold_out.content);

    let cap_err = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "mem-scout", "prompt": "wake over cap" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(cap_err.is_error);
    assert!(cap_err.content.contains("并发数已达上限"));

    let _ = hold_tx.send(true);
    watch.wait_all_settled(5).await;
    app.fiber().dispose().await;
}

#[tokio::test]
async fn stop_while_winding_down_blocks_send_with_stopping_in_progress_then_allows_resume() {
    let ws = temp_dir("stop_winding_down");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let (started_tx, mut started_rx) = tokio::sync::watch::channel(false);
    let (release_cancel_tx, release_cancel_rx) = tokio::sync::watch::channel(false);
    let exited = Arc::new(AtomicBool::new(false));

    struct WindingAdapter {
        round: AtomicUsize,
        started_tx: tokio::sync::watch::Sender<bool>,
        release_cancel_rx: tokio::sync::watch::Receiver<bool>,
        exited: Arc<AtomicBool>,
    }

    impl LlmAdapter for WindingAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let round = self.round.fetch_add(1, Ordering::SeqCst);
            let started_tx = self.started_tx.clone();
            let mut release_rx = self.release_cancel_rx.clone();
            let exited = Arc::clone(&self.exited);
            tokio::spawn(async move {
                if round == 0 {
                    let _ = started_tx.send(true);
                    options.cancel.cancelled().await;
                    // 人为阻塞在 stop 收尾窗口中，模拟工具释放或 flush 尚在进行
                    let _ = release_rx.wait_for(|&ok| ok).await;
                    exited.store(true, Ordering::SeqCst);
                    let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                } else {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("RESUMED_AFTER_WIND_DOWN")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    let _llm = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(WindingAdapter {
                round: AtomicUsize::new(0),
                started_tx,
                release_cancel_rx,
                exited: Arc::clone(&exited),
            }),
        )
        .unwrap();

    let tools = app.tools().unwrap();
    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    let sp = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "slow-w", "prompt": "step 1" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!sp.is_error, "{}", sp.content);
    tokio::time::timeout(Duration::from_secs(3), started_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();

    // 触发 stop：状态已置 Stopped (running_count=0)，但原 Worker 仍在收尾
    let stop_out = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "stop", "name": "slow-w" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!stop_out.is_error, "{}", stop_out.content);

    // 收尾未完前调 send 必须精准拒绝返回 StoppingInProgress，绝不拉起第二条并发 Worker
    let during_wind_down = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "slow-w", "prompt": "too early" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(during_wind_down.is_error);
    assert!(
        during_wind_down.content.contains("正在停止收尾中"),
        "expected StoppingInProgress, got: {}",
        during_wind_down.content
    );

    // 放行收尾并等待原 Worker 清理完守卫与 active_run_id，随后 send 可干净唤醒续跑
    let _ = release_cancel_tx.send(true);
    tokio::time::timeout(Duration::from_secs(3), async {
        while !exited.load(Ordering::SeqCst) || app.subagent().unwrap().active_child_count() > 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    watch.advance_to_end();

    let after_wind_down = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "slow-w", "prompt": "try again safely" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!after_wind_down.is_error, "{}", after_wind_down.content);

    watch.wait_all_settled(5).await;
    let followup = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .unwrap();
    assert!(
        followup.text().contains("RESUMED_AFTER_WIND_DOWN"),
        "{}",
        followup.text()
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn send_steers_running_subagent_in_completion_window_and_rejects_model_change() {
    let ws = temp_dir("steer_running");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let (in_stream_tx, mut in_stream_rx) = tokio::sync::watch::channel(false);
    let (finish_round1_tx, finish_round1_rx) = tokio::sync::watch::channel(false);

    struct SteerAdapter {
        round: AtomicUsize,
        in_stream_tx: tokio::sync::watch::Sender<bool>,
        finish_round1_rx: tokio::sync::watch::Receiver<bool>,
    }

    impl LlmAdapter for SteerAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let round = self.round.fetch_add(1, Ordering::SeqCst);
            let in_stream_tx = self.in_stream_tx.clone();
            let mut finish_rx = self.finish_round1_rx.clone();
            tokio::spawn(async move {
                if round == 0 {
                    let _ = in_stream_tx.send(true);
                    let _ = finish_rx.wait_for(|&go| go).await;
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("ROUND1_DONE")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                } else {
                    let saw_steer = options
                        .messages
                        .iter()
                        .any(|m| m.text_content().contains("SWITCH_TO_PLAN_B"));
                    let msg = if saw_steer {
                        "STEER_WINDOW_OK"
                    } else {
                        "MISSING_STEER"
                    };
                    let _ = tx.unbounded_send(Ok(StreamChunk::text(msg)));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    let _llm = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(SteerAdapter {
                round: AtomicUsize::new(0),
                in_stream_tx,
                finish_round1_rx,
            }),
        )
        .unwrap();

    let tools = app.tools().unwrap();
    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    let sp = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "steer-w", "prompt": "do plan A" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!sp.is_error, "{}", sp.content);
    tokio::time::timeout(Duration::from_secs(3), in_stream_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();

    // 运行中尝试修改 model 必须被拒绝
    let model_err = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "steer-w", "model": "alt/m2", "prompt": "nope" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(model_err.is_error);
    assert!(model_err.content.contains("无法在运行期间热切 model"));

    // 在第 0 轮即将返回 Stop 前注入 Steer：由 release_session 升为 FollowUp 并由同一 Worker 无缝接力跑完第 2 轮
    let steer_ok = tools
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "steer-w", "prompt": "SWITCH_TO_PLAN_B" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!steer_ok.is_error, "{}", steer_ok.content);
    assert_eq!(steer_ok.details.unwrap()["delivery"], "steer");

    let _ = finish_round1_tx.send(true);
    let ev = watch.wait_all_settled(5).await;
    assert_eq!(ev.status, SubagentRunStatus::Completed);

    let followup = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .unwrap();
    assert!(
        followup.text().contains("STEER_WINDOW_OK"),
        "{}",
        followup.text()
    );
    assert!(
        app.agent_loop()
            .unwrap()
            .inbox()
            .pop_followup(&parent.id)
            .is_none()
    );

    let snaps = app
        .subagent()
        .unwrap()
        .list_subagents_for_parent(&parent.id);
    assert_eq!(snaps[0].steps, 2);

    app.fiber().dispose().await;
}

#[tokio::test]
async fn timeout_repairs_dangling_tool_call_enabling_clean_send_resume() {
    let ws = temp_dir("timeout_repair_resume");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(
        &ws,
        SubagentConfig {
            timeout_secs: 1,
            stall_timeout_secs: 0,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    // 注册一个永不返回的慢工具（阻塞在未触发的 Notify 上）模拟超时强停留下悬挂 ToolCall
    let tools_reg = app.tools().unwrap();
    let slow_def = ToolDefinition::async_rich_tool(
        "slow_readonly",
        "Always stalls until cancelled",
        json!({"type": "object", "properties": {}}),
        |_, call_ctx| async move {
            call_ctx.cancel.cancelled().await;
            std::future::pending::<()>().await;
            Ok(mizar_tools::ToolOutput::text("unreachable"))
        },
    )
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    });
    let _slow_h = tools_reg.register(slow_def).unwrap();

    let resumed_saw_repaired_tool = Arc::new(AtomicBool::new(false));
    let saw_flag = Arc::clone(&resumed_saw_repaired_tool);

    struct TimeoutResumeAdapter {
        round: AtomicUsize,
        saw_repaired: Arc<AtomicBool>,
    }

    impl LlmAdapter for TimeoutResumeAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let round = self.round.fetch_add(1, Ordering::SeqCst);
            let saw_repaired = Arc::clone(&self.saw_repaired);
            tokio::spawn(async move {
                if round == 0 {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "call_hang_1".into(),
                            name: "slow_readonly".into(),
                            arguments: json!({}).to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                } else {
                    let has_unknown_result = options.messages.iter().any(|m| {
                        matches!(
                            m,
                            Message::Tool {
                                tool_call_id,
                                is_error: true,
                                content,
                                ..
                            } if tool_call_id == "call_hang_1"
                                && content.contains("TOOL_OUTCOME_UNKNOWN")
                        )
                    });
                    saw_repaired.store(has_unknown_result, Ordering::SeqCst);
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("CLEAN_AFTER_TIMEOUT")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    let _llm = app
        .llm()
        .unwrap()
        .register_adapter(
            "main",
            Arc::new(TimeoutResumeAdapter {
                round: AtomicUsize::new(0),
                saw_repaired: saw_flag,
            }),
        )
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    let sp = tools_reg
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "timeout-w", "prompt": "run slow tool" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!sp.is_error, "{}", sp.content);

    let ev_to = watch.wait_all_settled(5).await;
    assert_eq!(ev_to.status, SubagentRunStatus::TimedOut);
    let _ = app.agent_loop().unwrap().inbox().pop_followup(&parent.id);

    // 唤醒 TimedOut 会话：不得因悬挂 ToolCall 导致协议异常
    let resume = tools_reg
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "send", "name": "timeout-w", "prompt": "recover and summarize" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!resume.is_error, "{}", resume.content);

    let ev_done = watch.wait_all_settled(5).await;
    assert_eq!(ev_done.status, SubagentRunStatus::Completed);
    assert!(
        resumed_saw_repaired_tool.load(Ordering::SeqCst),
        "resumed child session must have synthetic TOOL_OUTCOME_UNKNOWN pairing call_hang_1"
    );
    let followup = app
        .agent_loop()
        .unwrap()
        .inbox()
        .pop_followup(&parent.id)
        .unwrap();
    assert!(
        followup.text().contains("CLEAN_AFTER_TIMEOUT"),
        "{}",
        followup.text()
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn watchdog_alerts_on_stall_reserves_last_wake_budget_and_skips_active_child() {
    // Part A: 静默超时触发看门狗，并在仅剩 1 格预算时让路给最终结算 FollowUp
    let ws1 = temp_dir("watchdog_stall");
    let _g1 = DirGuard(ws1.clone());
    std::fs::create_dir_all(&ws1).unwrap();

    let (app1, _) = setup_base_app(
        &ws1,
        SubagentConfig {
            stall_timeout_secs: 1,
            max_watchdog_alerts: 2,
            timeout_secs: 20,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    let (wd_release_tx, wd_release_rx) = tokio::sync::watch::channel(false);

    struct StallThenFinishAdapter {
        wd_release_rx: tokio::sync::watch::Receiver<bool>,
    }

    impl LlmAdapter for StallThenFinishAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let mut release_rx = self.wd_release_rx.clone();
            tokio::spawn(async move {
                let _ = release_rx.wait_for(|&ok| ok).await;
                let _ = tx.unbounded_send(Ok(StreamChunk::text("FINAL_SETTLED_AFTER_WD")));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            });
            Box::pin(rx)
        }
    }

    let _llm1 = app1
        .llm()
        .unwrap()
        .register_adapter("main", Arc::new(StallThenFinishAdapter { wd_release_rx }))
        .unwrap();

    let parent1 = app1.sessions().unwrap().create(None);
    let loop1 = app1.agent_loop().unwrap();
    let inbox1 = loop1.inbox();
    // 设限为 2：第 1 次看门狗消耗 1 格剩 1 格；消费后剩 1 格时第 2 次看门狗必须降级为 Inject，把最后 1 格留给结算
    inbox1.set_auto_wake_limit(&parent1.id, 2);
    let mut watch1 = StateWatch::attach(&app1, &parent1.id);

    let sp = app1
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "wd-worker", "prompt": "slow stall check" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(!sp.is_error, "{}", sp.content);

    // 等第 1 次看门狗触发事件
    let wd_ev1 = watch1
        .wait_for_event(5, |ev| {
            ev.notice.as_deref().is_some_and(|n| n.contains("已静默"))
        })
        .await;
    assert!(wd_ev1.notice.unwrap().contains("已静默"));

    // 父代理消费这次看门狗唤醒，使 remaining_auto_wakes 降到 1
    let wd_msg1 = inbox1
        .pop_followup(&parent1.id)
        .expect("watchdog FollowUp #1");
    assert!(wd_msg1.text().contains("Subagent watchdog notice (1/2)"));
    assert_eq!(inbox1.remaining_auto_wakes(&parent1.id), 1);

    // 等第 2 次看门狗触发：因 remaining == 1，必须降级为 Inject（通知 UI 保留最后 1 格额度给最终汇总）
    let wd_ev2 = watch1
        .wait_for_event(5, |ev| {
            ev.notice
                .as_deref()
                .is_some_and(|n| n.contains("保留最终汇总唤醒额度"))
        })
        .await;
    assert!(wd_ev2.notice.unwrap().contains("保留最终汇总唤醒额度"));
    assert_eq!(inbox1.remaining_auto_wakes(&parent1.id), 1);
    assert!(
        inbox1.pop_followup(&parent1.id).is_none(),
        "second watchdog must degrade to Inject and not steal the last FollowUp slot"
    );

    // 放行子代理结束：最终结算通知必须顺利拿到这最后 1 格 FollowUp 唤醒，绝不降级
    let _ = wd_release_tx.send(true);
    let ev_final = watch1.wait_all_settled(5).await;
    assert_eq!(ev_final.status, SubagentRunStatus::Completed);
    let settled_msg = inbox1
        .pop_followup(&parent1.id)
        .expect("final settlement must get the reserved FollowUp slot");
    assert!(settled_msg.text().contains("FINAL_SETTLED_AFTER_WD"));
    assert_eq!(inbox1.remaining_auto_wakes(&parent1.id), 0);

    app1.fiber().dispose().await;

    // Part B: 持续有流输出的活跃子代理超过 stall_timeout_secs 也绝不误报看门狗
    let ws2 = temp_dir("watchdog_active_no_false_alarm");
    let _g2 = DirGuard(ws2.clone());
    std::fs::create_dir_all(&ws2).unwrap();

    let (app2, _) = setup_base_app(
        &ws2,
        SubagentConfig {
            stall_timeout_secs: 1,
            max_watchdog_alerts: 2,
            timeout_secs: 20,
            ..SubagentConfig::default()
        },
        false,
    )
    .await;

    struct TrickleActiveAdapter;
    impl LlmAdapter for TrickleActiveAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                // 每 250ms 吐一块增量，共 1.5s > stall_timeout_secs(1s)
                for i in 0..6 {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    let _ = tx.unbounded_send(Ok(StreamChunk::text(format!("tick-{i} "))));
                }
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            });
            Box::pin(rx)
        }
    }

    let _llm2 = app2
        .llm()
        .unwrap()
        .register_adapter("main", Arc::new(TrickleActiveAdapter))
        .unwrap();
    let parent2 = app2.sessions().unwrap().create(None);
    let mut watch2 = StateWatch::attach(&app2, &parent2.id);

    let sp2 = app2
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "active-w", "prompt": "stream steadily" }),
            Some(parent2.id.clone()),
        ))
        .await;
    assert!(!sp2.is_error, "{}", sp2.content);

    watch2.wait_all_settled(5).await;
    let snaps2 = app2
        .subagent()
        .unwrap()
        .list_subagents_for_parent(&parent2.id);
    assert_eq!(snaps2[0].watchdog_alerts, 0);
    assert!(
        !watch2
            .snapshot()
            .iter()
            .any(|e| e.notice.as_deref().is_some_and(|n| n.contains("已静默"))),
        "active streaming child must never trigger false watchdog alerts"
    );

    app2.fiber().dispose().await;
}

#[tokio::test]
async fn status_exposes_live_step_tool_and_idle_seconds() {
    let ws = temp_dir("live_status");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let (in_tool_tx, mut in_tool_rx) = tokio::sync::watch::channel(false);
    let (release_tool_tx, release_tool_rx) = tokio::sync::watch::channel(false);
    let tools_reg = app.tools().unwrap();
    let live_probe_def = ToolDefinition::async_rich_tool(
        "probe_ro",
        "Holds until test checks status",
        json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        move |_, _| {
            let in_tool_tx = in_tool_tx.clone();
            let mut rx = release_tool_rx.clone();
            async move {
                let _ = in_tool_tx.send(true);
                let _ = rx.wait_for(|&go| go).await;
                Ok(mizar_tools::ToolOutput::text("probe ok"))
            }
        },
    )
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    });
    let _probe_h = tools_reg.register(live_probe_def).unwrap();

    struct LiveStatusAdapter;
    impl LlmAdapter for LiveStatusAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                if matches!(options.messages.last(), Some(Message::Tool { .. })) {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("DONE")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                } else {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "p1".into(),
                            name: "probe_ro".into(),
                            arguments: json!({"query": "scan-auth-module"}).to_string(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    let _llm = app
        .llm()
        .unwrap()
        .register_adapter("main", Arc::new(LiveStatusAdapter))
        .unwrap();
    let parent = app.sessions().unwrap().create(None);
    let mut watch = StateWatch::attach(&app, &parent.id);

    let sp = tools_reg
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "name": "live-w", "prompt": "probe auth" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!sp.is_error, "{}", sp.content);

    tokio::time::timeout(Duration::from_secs(3), in_tool_rx.wait_for(|&v| v))
        .await
        .unwrap()
        .unwrap();

    let status_out = tools_reg
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "action": "status", "name": "live-w" }),
            Some(parent.id.clone()),
        ))
        .await;
    assert!(!status_out.is_error, "{}", status_out.content);
    assert!(
        status_out.content.contains("step=1"),
        "{}",
        status_out.content
    );
    assert!(
        status_out
            .content
            .contains("tool=\"probe_ro: scan-auth-module\""),
        "{}",
        status_out.content
    );
    assert!(
        status_out.content.contains("idle="),
        "{}",
        status_out.content
    );

    let task_json = &status_out.details.unwrap()["tasks"][0];
    assert_eq!(task_json["current_step"], 1);
    assert_eq!(task_json["last_tool"], "probe_ro: scan-auth-module");

    let _ = release_tool_tx.send(true);
    watch.wait_all_settled(5).await;
    app.fiber().dispose().await;
}
