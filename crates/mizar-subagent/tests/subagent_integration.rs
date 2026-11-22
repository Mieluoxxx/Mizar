//! `mizar-subagent` 集成测试：验证并发只读扇出、会话谱系隔离、严格深度 1 护栏、
//! `edit` 模式授权、父取消级联以及 `max_steps` / 超时封顶与资源清理。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cordis::{Context, PluginDef, Service};
use futures::channel::mpsc;
use mizar_agent_loop::{AgentLoop, AgentLoopConfig, AgentLoopContextExt, RunOptions};
use mizar_llm::{
    BoxChunkStream, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt,
    LlmError, Message, ModelInfo, ProviderInfo, RequestOrigin, StreamChunk, ToolCall,
};
use mizar_recall::{RecallConfig, RecallPlugin};
use mizar_session::{SessionContextExt, SessionStore, TitleSource};
use mizar_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};
use mizar_session_title::{SessionTitleConfig, SessionTitleRuntime};
use mizar_subagent::{
    SUBAGENT_TOOL_NAME, SubagentConfig, SubagentContextExt, SubagentMode, SubagentPlugin,
};
use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_tool_fs::{ToolFsConfig, ToolFsPlugin};
use mizar_tool_time::ToolTimePlugin;
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecution, ToolRuntime, ToolsContextExt};
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

async fn setup_base_app(
    ws: &PathBuf,
    sub_cfg: SubagentConfig,
    use_sqlite: bool,
) -> (Context, Option<PathBuf>) {
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
    app.plugin(SubagentPlugin, sub_cfg).await.unwrap();
    app.sessions()
        .unwrap()
        .set_default_cwd(Some(ws.display().to_string()));
    (app, db_path)
}

#[tokio::test]
async fn parallel_read_only_subagents_isolate_history_and_persist_lineage() {
    let ws = temp_dir("parallel");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("alpha.txt"), "ALPHA_SECRET_42\n").unwrap();
    std::fs::write(ws.join("beta.txt"), "BETA_SECRET_99\n").unwrap();

    let (app, _db) = setup_base_app(&ws, SubagentConfig::default(), true).await;
    // 同时挂载 legacy 工具、Recall 与带 read_only=true 注解的自定义工具，校验白名单兼容
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

    struct ParallelMockAdapter {
        recorded: Arc<Mutex<Vec<RecordedCall>>>,
        in_flight: Arc<AtomicUsize>,
        max_concurrent: Arc<AtomicUsize>,
        child_barrier: Arc<Barrier>,
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
            tokio::spawn(async move {
                match &options.origin {
                    RequestOrigin::SessionTurn { .. } => {
                        let last = options.messages.last();
                        if matches!(last, Some(Message::Tool { .. })) {
                            let summaries: Vec<String> = options
                                .messages
                                .iter()
                                .filter_map(|m| match m {
                                    Message::Tool { content, .. } => Some(content.clone()),
                                    _ => None,
                                })
                                .collect();
                            let _ = tx.unbounded_send(Ok(StreamChunk::text(format!(
                                "父汇总: {}",
                                summaries.join(" | ")
                            ))));
                            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                                reason: FinishReason::Stop,
                                replay: None,
                            }));
                        } else {
                            let c1 = ToolCall {
                                id: "sub_1".into(),
                                name: "subagent".into(),
                                arguments: json!({
                                    "prompt": "读取 alpha.txt 里的暗号"
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
                            // 用 Barrier 实证同一父步内的两个只读子代理确实并发执行
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
    let reply = app
        .agent_loop()
        .unwrap()
        .run(
            Arc::clone(&parent),
            "并发调查 alpha 与 beta",
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(reply.contains("ALPHA_SECRET_42"), "reply: {reply}");
    assert!(reply.contains("BETA_SECRET_99"), "reply: {reply}");
    assert_eq!(
        max_concurrent.load(Ordering::SeqCst),
        2,
        "two read_only subagent calls in one step must run concurrently"
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

    // 等待一小段时间确认后台 session-title 不会误触子会话
    tokio::time::sleep(Duration::from_millis(30)).await;

    // 子会话持久化在 SQLite，具备 parent_id、标题前缀且与父会话隔离
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
                .is_some_and(|t| t.starts_with("Subagent: ")),
            "unexpected child title: {:?}",
            h.title
        );
        assert_eq!(h.title_source, Some(TitleSource::Manual));
        assert!(h.fork_seq.is_none());
        assert_eq!(h.cwd.as_deref(), parent.header().cwd.as_deref());
    }

    // 校验所有子会话 LLM 调用都带 RequestOrigin::Subagent，且声明工具严格不含 subagent 与写工具
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
                        // 同时模拟外部沙箱（execute_nested）在只读子会话下越权调用 write/subagent
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

                        // 模型自身也尝试直接调用 subagent、write 和 bash
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
    let res = tools_ref
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "try prohibited ops" }),
            Some(parent.id.clone()),
        ))
        .await;

    assert!(!res.is_error, "unexpected error: {}", res.content);
    assert!(
        nested_denied_seen.load(Ordering::SeqCst),
        "nested/sandbox calls from read_only child session must be denied at pre-execute"
    );
    assert!(
        res.content.contains("已拦截") && res.content.contains("subagent"),
        "expected subagent denial in child tool_results: {}",
        res.content
    );
    assert!(
        res.content
            .contains("只读模式子代理禁止调用非只读工具 write"),
        "expected write denial: {}",
        res.content
    );
    assert!(
        res.content
            .contains("只读模式子代理禁止调用非只读工具 bash"),
        "expected bash denial: {}",
        res.content
    );
    assert!(!ws.join("evil.txt").exists());
    assert!(!ws.join("hack.txt").exists());

    // 校验 child declared tools 不含 subagent、write、edit、bash
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

    // 子代理完成后，用该子会话 ID 再次手动调用 subagent 也必须被拦截
    let child_id = res.details.unwrap()["childSessionId"]
        .as_str()
        .unwrap()
        .to_string();
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
    let main_adapter = Arc::new(DualAdapter {
        id: "main",
        step: AtomicUsize::new(0),
        seen: Arc::clone(&seen),
    });
    let alt_adapter = Arc::new(DualAdapter {
        id: "alt",
        step: AtomicUsize::new(0),
        seen: Arc::clone(&seen),
    });
    let _r1 = app
        .llm()
        .unwrap()
        .register_adapter("main", main_adapter)
        .unwrap();
    let _r2 = app
        .llm()
        .unwrap()
        .register_adapter("alt", alt_adapter)
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
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
    assert_eq!(
        std::fs::read_to_string(ws.join("created_by_child.txt")).unwrap(),
        "edit-mode-ok"
    );
    assert!(
        res.content.contains("已拦截") && res.content.contains("subagent"),
        "subagent recursion must still be blocked in edit mode: {}",
        res.content
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
async fn parent_cancellation_cascades_immediately_and_cleans_tracking() {
    let ws = temp_dir("cancel_cascade");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let (app, _) = setup_base_app(&ws, SubagentConfig::default(), false).await;

    let child_started = Arc::new(tokio::sync::Notify::new());
    let child_cancelled_observed = Arc::new(AtomicBool::new(false));

    struct SlowAdapter {
        started: Arc<tokio::sync::Notify>,
        cancelled_observed: Arc<AtomicBool>,
    }

    impl LlmAdapter for SlowAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let (tx, rx) = mpsc::unbounded();
            let started = Arc::clone(&self.started);
            let obs = Arc::clone(&self.cancelled_observed);
            tokio::spawn(async move {
                started.notify_one();
                tokio::select! {
                    _ = options.cancel.cancelled() => {
                        obs.store(true, Ordering::SeqCst);
                        let _ = tx.unbounded_send(Err(LlmError::cancelled()));
                    }
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("too-late")));
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

    let adapter = Arc::new(SlowAdapter {
        started: Arc::clone(&child_started),
        cancelled_observed: Arc::clone(&child_cancelled_observed),
    });
    let _reg = app
        .llm()
        .unwrap()
        .register_adapter("main", adapter)
        .unwrap();

    let parent = app.sessions().unwrap().create(None);
    let parent_cancel = CancellationToken::new();
    let cancel_trigger = parent_cancel.clone();
    let tools = app.tools().unwrap();
    let sub = app.subagent().unwrap();

    let exec_handle = tokio::spawn({
        let tools = Arc::clone(&tools);
        let parent_id = parent.id.clone();
        async move {
            tools
                .execute_with(
                    ToolExecution::new(
                        SUBAGENT_TOOL_NAME,
                        json!({"prompt": "long task"}),
                        Some(parent_id),
                    ),
                    parent_cancel,
                    None,
                )
                .await
        }
    });

    child_started.notified().await;
    assert_eq!(sub.active_child_count(), 1);
    cancel_trigger.cancel();

    let res = tokio::time::timeout(Duration::from_secs(2), exec_handle)
        .await
        .expect("cancellation must finish immediately")
        .unwrap();

    assert!(res.is_error);
    assert!(
        res.content.contains("取消"),
        "expected Chinese cancel message: {}",
        res.content
    );
    assert!(child_cancelled_observed.load(Ordering::SeqCst));
    assert_eq!(sub.active_child_count(), 0);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn enforces_max_steps_cap_and_timeout_and_validates_args() {
    assert_eq!(
        SubagentConfig {
            default_max_steps: 99,
            max_steps_cap: 40,
            timeout_secs: 180,
        }
        .effective_max_steps(),
        40
    );

    // 1. max_steps 上限验证
    let ws1 = temp_dir("max_steps");
    let _g1 = DirGuard(ws1.clone());
    std::fs::create_dir_all(&ws1).unwrap();
    let (app1, _) = setup_base_app(
        &ws1,
        SubagentConfig {
            default_max_steps: 10,
            max_steps_cap: 2,
            timeout_secs: 180,
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

    // 校验中文参数错误
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

    let max_res = tools1
        .execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "keep searching" }),
            Some(parent1.id.clone()),
        ))
        .await;
    assert!(!max_res.is_error, "max_steps returns truncation report");
    assert_eq!(step_calls.load(Ordering::SeqCst), 2);
    assert!(
        max_res.content.contains("步数上限") && max_res.content.contains("steps=2/2"),
        "unexpected max_steps output: {}",
        max_res.content
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
    let timeout_res = tokio::time::timeout(
        Duration::from_secs(4),
        app2.tools().unwrap().execute(ToolExecution::new(
            SUBAGENT_TOOL_NAME,
            json!({ "prompt": "hang" }),
            Some(parent2.id.clone()),
        )),
    )
    .await
    .expect("subagent timeout must trigger promptly");

    assert!(timeout_res.is_error);
    assert!(
        timeout_res.content.contains("超时"),
        "unexpected timeout error: {}",
        timeout_res.content
    );
    assert_eq!(app2.subagent().unwrap().active_child_count(), 0);
    app2.fiber().dispose().await;
}

#[tokio::test]
async fn parent_turn_route_is_inherited_without_session_selection() {
    let ws = temp_dir("inherit_route");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    // AgentLoop 默认指向 main，但父轮次通过 RunOptions 指定 custom/custom-m
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
