use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{Context, PluginDef, Service};
use futures::channel::mpsc;
use mizar_agent_loop::{AgentLoop, AgentLoopConfig, AgentLoopContextExt};
use mizar_compaction::{
    CompactionConfig, CompactionContextExt, CompactionOutcome, CompactionRuntime,
    ManualCompactOptions,
};
use mizar_llm::{
    BoxChunkStream, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt,
    LlmError, LlmErrorCode, LlmRuntime, ModelInfo, ProviderInfo, ReasoningEffort, StreamChunk,
    TokenUsage,
};
use mizar_session::{SessionContextExt, SessionCreateOptions, SessionEvent, SessionStore};
use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_tools::{ToolDefinition, ToolExecution, ToolExposure, ToolRuntime, ToolsContextExt};
use mizar_tools_coding::{CodingToolsConfig, CodingToolsPlugin};
use serde_json::json;
use tokio_util::sync::CancellationToken;

type MockResponse = Result<(String, FinishReason), LlmError>;

struct RecordingAdapter {
    calls: Arc<Mutex<Vec<GenerateOptions>>>,
    responses: Arc<Mutex<Vec<MockResponse>>>,
    context_window: u64,
}

impl RecordingAdapter {
    fn new(
        context_window: u64,
        responses: Vec<MockResponse>,
    ) -> (Self, Arc<Mutex<Vec<GenerateOptions>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let adapter = Self {
            calls: Arc::clone(&calls),
            responses: Arc::new(Mutex::new(responses)),
            context_window,
        };
        (adapter, calls)
    }
}

impl LlmAdapter for RecordingAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "mock-compact".into(),
            name: "Mock Compact".into(),
            protocol: "mock".into(),
            models: vec![ModelInfo {
                id: "m-test".into(),
                name: "Test Model".into(),
                context_window: Some(self.context_window),
                max_tokens: Some(4_096),
                reasoning_efforts: vec![ReasoningEffort::Off, ReasoningEffort::Medium],
                input: Vec::new(),
            }],
        })
    }

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        self.calls.lock().unwrap().push(options);
        let next = {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                Ok((
                    "## Goal & Scope\n- Mock summary\n## Verified Facts & Decisions\n- Confirmed"
                        .into(),
                    FinishReason::Stop,
                ))
            } else {
                q.remove(0)
            }
        };
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            match next {
                Ok((text, reason)) => {
                    if !text.is_empty() {
                        let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta { index: 0, text }));
                    }
                    let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                        usage: TokenUsage {
                            input_tokens: 120,
                            output_tokens: 30,
                            total_tokens: Some(150),
                            ..TokenUsage::default()
                        },
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason,
                        replay: None,
                    }));
                }
                Err(err) => {
                    let _ = tx.unbounded_send(Err(err));
                }
            }
        });
        Box::pin(rx)
    }
}

async fn setup_stack(
    adapter: RecordingAdapter,
    cfg: CompactionConfig,
    workspace: Option<std::path::PathBuf>,
) -> Context {
    let app = Context::new();
    app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    app.plugin(CompactionRuntime::plugin(), cfg).await.unwrap();
    if let Some(ws) = workspace {
        app.sessions()
            .unwrap()
            .set_default_cwd(Some(ws.display().to_string()));
        app.plugin(CodingToolsPlugin, CodingToolsConfig::new(ws))
            .await
            .unwrap();
    }
    app.plugin(AgentLoop::plugin(), AgentLoopConfig::default())
        .await
        .unwrap();

    let adapter = Arc::new(adapter);
    app.plugin(
        PluginDef::sync("mock-compact-adapter", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter);
            ctx.effect("mock-compact-adapter.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("mock-compact", a)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();
    app
}

#[tokio::test]
async fn l1_pruner_handles_pressure_without_llm_call() {
    let (adapter, calls) = RecordingAdapter::new(
        16_000,
        vec![Ok(("正常回答，未触发摘要".into(), FinishReason::Stop))],
    );
    let cfg = CompactionConfig {
        default_context_window: 16_000,
        trigger_ratio: 0.70,
        prune_target_ratio: 0.55,
        prune_threshold_chars: 2_000,
        prune_head_chars: 400,
        prune_tail_chars: 200,
        prune_protect_recent_surface: 2,
        ..Default::default()
    };

    let app = setup_stack(adapter, cfg, None).await;
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();

    let s = store.create(Some("l1-prune".into()));
    s.append(SessionEvent::user("读一下超大日志"));
    // 约 9_000 tokens 的旧 ToolResult（超过 16k 窗口的 70% 触发线，但剪完后远低于 55%）
    s.append(SessionEvent::ToolResult {
        tool_call_id: "c1".into(),
        name: "read".into(),
        content: "x".repeat(36_000),
        is_error: false,
    });
    s.append(SessionEvent::assistant_text("已读完日志。"));

    let reply = agent
        .run(Arc::clone(&s), "继续分析", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply, "正常回答，未触发摘要");

    // 只应发起 1 次主模型调用，0 次 LLM 摘要调用！
    {
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 1);
    }
    assert!(!s.pruned_map().is_empty());
    assert!(s.header().compacted_up_to_seq.is_none());
    let derived_str = format!("{:?}", s.derive_messages());
    assert!(derived_str.contains("recall_read(seq=2)"));
    // 原始日志中的 36_000 字节原文完好无损
    assert!(matches!(
        &s.all_entries()[1].event,
        SessionEvent::ToolResult { content, .. } if content.len() == 36_000
    ));

    app.fiber().dispose().await;
}

#[tokio::test]
async fn l2_manual_compaction_builds_digest_disables_reasoning_and_resets_session_tools() {
    let (adapter, calls) = RecordingAdapter::new(
        64_000,
        vec![Ok((
            "<mizar-summary>\n## Goal & Scope\n- Refactor parser\n## Verified Facts & Decisions\n- Updated src/lib.rs\n</mizar-summary>".into(),
            FinishReason::Stop,
        ))],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = app.sessions().unwrap();
    let tools = app.tools().unwrap();
    let compaction = app.compaction().unwrap();

    // 注册一个延迟工具并在本会话激活，验证压缩后会被清理回默认态
    let _reg = tools
        .register(
            ToolDefinition::sync_tool(
                "extra_deferred",
                "Extra deferred tool",
                json!({}),
                |_, _| Ok::<_, String>("ok".into()),
            )
            .with_exposure(ToolExposure::Deferred),
        )
        .unwrap();

    let s = store.create(Some("l2-manual".into()));
    tools.activate_tools(&s.id, &["extra_deferred".to_string()]);
    assert!(
        tools
            .list_declared(Some(&s.id))
            .iter()
            .any(|t| t.name == "extra_deferred")
    );

    s.append(SessionEvent::user("请改造 parser 并跑测试"));
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c1".into(),
            name: "edit".into(),
            arguments: r#"{"path":"crates/demo/src/lib.rs","old_text":"a","new_text":"b"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "c1".into(),
        name: "edit".into(),
        content: "edited ok".into(),
        is_error: false,
    });
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c2".into(),
            name: "bash".into(),
            arguments: r#"{"command":"cargo test -p demo"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "c2".into(),
        name: "bash".into(),
        content: "all passed".into(),
        is_error: false,
    });
    s.append(SessionEvent::user("继续处理下一步 ".repeat(300)));

    let report = compaction
        .compact_now(
            Arc::clone(&s),
            ManualCompactOptions {
                provider: Some("mock-compact".into()),
                model: Some("m-test".into()),
                focus: Some("focus on parser changes".into()),
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(report.outcome, CompactionOutcome::Compacted);
    assert!(s.header().compacted_up_to_seq.is_some());

    // 确认摘要请求关闭了思考（dim #170）并携带了用户 focus
    {
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].reasoning_effort, Some(ReasoningEffort::Off));
        let last_msg = format!("{:?}", recorded[0].messages.last().unwrap());
        assert!(last_msg.contains("focus on parser changes"));
    }

    // 确认压后 derive_messages 包含确定性 <mizar-digest> 且剥离了冗余 <mizar-summary> 包装
    let derived = s.derive_messages();
    let first_msg = format!("{:?}", derived[0]);
    assert!(first_msg.contains("<mizar-digest>"), "{first_msg}");
    assert!(first_msg.contains("crates/demo/src/lib.rs"), "{first_msg}");
    assert!(first_msg.contains("cargo test -p demo"), "{first_msg}");
    assert!(first_msg.contains("Refactor parser"), "{first_msg}");

    // 压后延迟激活工具已被清空
    assert!(
        !tools
            .list_declared(Some(&s.id))
            .iter()
            .any(|t| t.name == "extra_deferred")
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn l3_overflow_recovery_automatically_compacts_and_retries_turn() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let (adapter, _calls) = RecordingAdapter::new(
        128_000,
        vec![
            // 第 1 次主请求模拟上游返回 ContextWindowExceeded
            Err(LlmError::new(
                LlmErrorCode::ContextWindowExceeded,
                "maximum context length exceeded",
            )),
            // 第 2 次调用是 CompactionRuntime 发出的纯净转写摘要请求
            Ok((
                "## Goal & Scope\n- Recovered summary".into(),
                FinishReason::Stop,
            )),
            // 第 3 次是 AgentLoop 压缩后自动重试的主请求
            Ok(("压缩后自动重试成功！".into(), FinishReason::Stop)),
        ],
    );

    // 提高自动阈值，确保步前不触发，专门验证 L3 运行期溢出兜底
    let cfg = CompactionConfig {
        default_context_window: 128_000,
        auto_prune: false,
        ..Default::default()
    };

    let app = setup_stack(adapter, cfg, None).await;
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();

    let s = store.create(Some("l3-overflow".into()));
    s.append(SessionEvent::user("历史消息 1 ".repeat(100)));
    s.append(SessionEvent::assistant_text("历史回复 1 ".repeat(100)));

    let res = agent
        .run(Arc::clone(&s), "触发溢出", CancellationToken::new())
        .await
        .unwrap();
    let _ = call_count.fetch_add(1, Ordering::Relaxed);
    assert_eq!(res, "压缩后自动重试成功！");
    assert!(s.header().compacted_up_to_seq.is_some());

    app.fiber().dispose().await;
}

#[tokio::test]
async fn post_compaction_guards_enforce_read_dedup_and_block_stale_write_while_allowing_edit() {
    let ws = std::env::temp_dir().join(format!("mizar_compaction_guards_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();
    let target = ws.join("sample.rs");
    let body = format!(
        "fn alpha() {{}}\n{}",
        "// padding line for dedup threshold\n".repeat(30)
    );
    std::fs::write(&target, &body).unwrap();

    let (adapter, _) = RecordingAdapter::new(128_000, vec![]);
    let app = setup_stack(adapter, CompactionConfig::default(), Some(ws.clone())).await;
    let store = app.sessions().unwrap();
    let tools = app.tools().unwrap();

    let s = store.create_with(SessionCreateOptions {
        id: Some("guard-1".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });

    // 1. 首次 read 正常返回全文
    let r1 = tools
        .execute(ToolExecution::new(
            "read",
            json!({ "path": "sample.rs" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!r1.is_error);
    assert!(r1.content.contains("fn alpha()"));
    s.append(SessionEvent::ToolResult {
        tool_call_id: "r1".into(),
        name: "read".into(),
        content: r1.content.clone(),
        is_error: false,
    });

    // 2. 同一压缩代次内重复 read 同区间 -> 触发 read-dedup 返回短提示
    let r2 = tools
        .execute(ToolExecution::new(
            "read",
            json!({ "path": "sample.rs" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!r2.is_error);
    assert!(
        r2.content.contains("Unchanged file range"),
        "expected read-dedup notice, got: {}",
        r2.content
    );

    // 3. 发生 Compaction 把之前的 read 压出上下文
    s.append(SessionEvent::user("后续问题"));
    s.compact("早期摘要", 1, Some(500));

    // 4. 压后对已存在文件直接整文件 write -> 应被 FileReads 守卫拦截！
    let w_blocked = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "sample.rs", "content": "fn overwritten() {}" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(
        w_blocked.blocked,
        "expected stale write to be blocked: {:?}",
        w_blocked
    );

    // 5. 压后使用精确匹配 edit -> 必须放行！
    let e_ok = tools
        .execute(ToolExecution::new(
            "edit",
            json!({
                "path": "sample.rs",
                "old_text": "fn alpha() {}",
                "new_text": "fn beta() {}"
            }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(
        !e_ok.is_error && !e_ok.blocked,
        "edit must be allowed after compaction: {:?}",
        e_ok
    );

    // 6. 压后重读同一文件：read-dedup 已因压缩失效，应返回更新后的全文
    let r3 = tools
        .execute(ToolExecution::new(
            "read",
            json!({ "path": "sample.rs" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!r3.is_error);
    assert!(
        r3.content.contains("fn beta()"),
        "expected full updated file body, got: {}",
        r3.content
    );

    // 7. 重读后再整文件 write -> 放行！
    let w_ok = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "sample.rs", "content": "fn gamma() {}\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(
        !w_ok.is_error && !w_ok.blocked,
        "write after re-read should succeed: {:?}",
        w_ok
    );

    // 8. 新建不存在的文件即使在压后也始终放行
    let w_new = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "brand_new.rs", "content": "fn fresh() {}\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!w_new.is_error && !w_new.blocked);

    let _ = std::fs::remove_dir_all(&ws);
    app.fiber().dispose().await;
}

#[tokio::test]
async fn nested_exec_and_l1_prune_and_resumed_session_guards_work() {
    use mizar_session::{
        LoadedSession, PersistenceResult, PrunedToolResult, SERVICE_SESSION_PERSISTENCE,
        SessionEntry, SessionHeader, SessionHeaderPatch, SessionListQuery, SessionPersistence,
        SessionPersistenceHandle,
    };
    use std::collections::HashMap;

    #[derive(Default)]
    struct MemPersistence(Mutex<HashMap<String, (SessionHeader, Vec<SessionEntry>)>>);

    impl SessionPersistence for MemPersistence {
        fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
            self.0
                .lock()
                .unwrap()
                .insert(header.id.clone(), (header.clone(), Vec::new()));
            Ok(())
        }
        fn update_header(&self, _id: &str, _patch: SessionHeaderPatch) -> PersistenceResult<()> {
            Ok(())
        }
        fn load_session(&self, id: &str) -> PersistenceResult<Option<LoadedSession>> {
            Ok(self.0.lock().unwrap().get(id).map(|(h, e)| LoadedSession {
                header: h.clone(),
                entries: e.clone(),
                max_persisted_seq: e.last().map(|x| x.seq),
                skipped_events: 0,
            }))
        }
        fn list_sessions(&self, _q: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .values()
                .map(|(h, _)| h.clone())
                .collect())
        }
        fn append_events(&self, id: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
            if let Some((_, log)) = self.0.lock().unwrap().get_mut(id) {
                log.extend(entries.iter().cloned());
            }
            Ok(())
        }
        fn flush(&self, _session_id: Option<&str>) -> PersistenceResult<()> {
            Ok(())
        }
        fn flush_async(
            &self,
            _session_id: Option<String>,
        ) -> cordis::BoxFuture<'static, PersistenceResult<()>> {
            Box::pin(async { Ok(()) })
        }
        fn fork_session(
            &self,
            _from: &str,
            _up_to: u64,
            new_header: &SessionHeader,
        ) -> PersistenceResult<LoadedSession> {
            Ok(LoadedSession {
                header: new_header.clone(),
                entries: Vec::new(),
                max_persisted_seq: None,
                skipped_events: 0,
            })
        }
    }

    let ws = std::env::temp_dir().join(format!(
        "mizar_compaction_guards_extra_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();
    let sample = ws.join("sample.rs");
    std::fs::write(&sample, "fn initial() {}\n").unwrap();

    let (adapter, _) = RecordingAdapter::new(128_000, vec![]);
    let app = setup_stack(adapter, CompactionConfig::default(), Some(ws.clone())).await;
    let store = app.sessions().unwrap();
    let tools = app.tools().unwrap();

    // 1. 读盘后通过 L1 Prune 剪枝：整文件 write 被拦截，edit 放行，nested write 放行
    let s1 = store.create_with(SessionCreateOptions {
        id: Some("guard-l1".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    let r1 = tools
        .execute(
            ToolExecution::new("read", json!({ "path": "sample.rs" }), Some(s1.id.clone()))
                .with_call_id("rc1"),
        )
        .await;
    assert!(!r1.is_error);
    s1.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "rc1".into(),
            name: "read".into(),
            arguments: json!({ "path": "sample.rs" }).to_string(),
        }],
        replay: None,
        usage: None,
    });
    s1.append(SessionEvent::ToolResult {
        tool_call_id: "rc1".into(),
        name: "read".into(),
        content: r1.content,
        is_error: false,
    });
    let read_seq = s1.last_seq();
    s1.append(SessionEvent::user("later turn"));
    s1.append_prune(vec![PrunedToolResult {
        target_seq: read_seq,
        replacement: "pruned preview".into(),
        original_chars: 100,
    }]);

    let w_pruned = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "sample.rs", "content": "fn wiped() {}" }),
            Some(s1.id.clone()),
        ))
        .await;
    assert!(
        w_pruned.blocked && w_pruned.content.contains("pruned"),
        "write on L1-pruned read must be blocked: {:?}",
        w_pruned
    );

    let e_pruned = tools
        .execute(ToolExecution::new(
            "edit",
            json!({ "path": "sample.rs", "old_text": "fn initial() {}", "new_text": "fn edited() {}" }),
            Some(s1.id.clone()),
        ))
        .await;
    assert!(!e_pruned.is_error && !e_pruned.blocked, "{:?}", e_pruned);

    // 压缩后 nested write 必须绕过 FileReads 守卫（edit 后新增一轮用户消息以将 edit 压出可见窗口）
    s1.append(SessionEvent::user("after edit"));
    s1.append(SessionEvent::user("latest ask"));
    s1.compact("sum", 1, None);
    let w_after_compact = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "sample.rs", "content": "fn stale() {}" }),
            Some(s1.id.clone()),
        ))
        .await;
    assert!(w_after_compact.blocked);

    let w_nested = tools
        .execute(
            ToolExecution::new(
                "write",
                json!({ "path": "sample.rs", "content": "fn nested_ok() {}\n" }),
                Some(s1.id.clone()),
            )
            .with_nested(true),
        )
        .await;
    assert!(
        !w_nested.is_error && !w_nested.blocked,
        "exec.nested should skip FileGuardTracker: {:?}",
        w_nested
    );
    app.fiber().dispose().await;

    // 2. 冷加载恢复会话时从历史事件重建 last_read_seq 并拦截陈旧 write
    let persistence = Arc::new(MemPersistence::default());
    let mut header = SessionHeader::new("resumed-guard", 1000);
    header.cwd = Some(ws.display().to_string());
    persistence.create_session(&header).unwrap();
    let entries = vec![
        SessionEntry {
            seq: 1,
            ts: 1001,
            event: SessionEvent::Assistant {
                blocks: vec![ContentBlock::ToolCall {
                    id: "tc-read".into(),
                    name: "read".into(),
                    arguments: json!({ "path": "sample.rs" }).to_string(),
                }],
                replay: None,
                usage: None,
            },
        },
        SessionEntry {
            seq: 2,
            ts: 1002,
            event: SessionEvent::ToolResult {
                tool_call_id: "tc-read".into(),
                name: "read".into(),
                content: "fn nested_ok() {}\n".into(),
                is_error: false,
            },
        },
        SessionEntry {
            seq: 3,
            ts: 1003,
            event: SessionEvent::user("after read"),
        },
        SessionEntry {
            seq: 4,
            ts: 1004,
            event: SessionEvent::Compaction {
                summary: "summary".into(),
                first_kept_seq: 3,
                tokens_before: None,
                tokens_after: None,
                digest: Default::default(),
            },
        },
    ];
    persistence
        .append_events("resumed-guard", &entries)
        .unwrap();

    let (adapter2, _) = RecordingAdapter::new(128_000, vec![]);
    let app2 = Context::new();
    app2.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(
            Arc::clone(&persistence) as Arc<dyn SessionPersistence>
        )),
    )
    .unwrap();
    app2.plugin(LlmRuntime::plugin(), ()).await.unwrap();
    app2.plugin(SessionStore::plugin(), ()).await.unwrap();
    app2.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app2.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    app2.plugin(CompactionRuntime::plugin(), CompactionConfig::default())
        .await
        .unwrap();
    app2.plugin(CodingToolsPlugin, CodingToolsConfig::new(ws.clone()))
        .await
        .unwrap();
    let adapter2 = Arc::new(adapter2);
    app2.plugin(
        PluginDef::sync("mock-a2", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter2);
            ctx.effect("mock-a2.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("mock-compact", a)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();

    let _resumed = app2.sessions().unwrap().get("resumed-guard").unwrap();
    let w_resumed_blocked = app2
        .tools()
        .unwrap()
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "sample.rs", "content": "fn blocked_after_resume() {}" }),
            Some("resumed-guard".into()),
        ))
        .await;
    assert!(
        w_resumed_blocked.blocked,
        "resumed session must rebuild last_read_seq and block pre-compaction write: {:?}",
        w_resumed_blocked
    );

    let _ = std::fs::remove_dir_all(&ws);
    app2.fiber().dispose().await;
}

#[tokio::test]
async fn missing_llm_pairs_compaction_traces_and_cold_cache_and_route_work() {
    use cordis::BoxFuture;
    use mizar_agent_loop::{
        AgentPreStepDecision, AgentPreStepInput, AgentStepEndEvent, EVENT_AGENT_PRE_STEP,
        EVENT_AGENT_STEP_END, StepOutcome,
    };
    use mizar_compaction::{
        CompactionEndEvent, CompactionStartEvent, EVENT_COMPACTION_END, EVENT_COMPACTION_START,
    };
    use mizar_session::SessionTraceEvent;

    // 1. llm 服务缺失时 CompactionStart 与 CompactionEnd/EVENT_COMPACTION_END 成对触发
    let app = Context::new();
    let llm_fiber = app.plugin(LlmRuntime::plugin(), ());
    let _ = llm_fiber.wait_settled().await;
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    app.plugin(CompactionRuntime::plugin(), CompactionConfig::default())
        .await
        .unwrap();

    let starts = Arc::new(Mutex::new(Vec::<CompactionStartEvent>::new()));
    let ends = Arc::new(Mutex::new(Vec::<CompactionEndEvent>::new()));
    let s_cap = Arc::clone(&starts);
    let e_cap = Arc::clone(&ends);
    let _h1 = app
        .on(EVENT_COMPACTION_START, move |ev: &CompactionStartEvent| {
            s_cap.lock().unwrap().push(ev.clone());
        })
        .unwrap();
    let _h2 = app
        .on(EVENT_COMPACTION_END, move |ev: &CompactionEndEvent| {
            e_cap.lock().unwrap().push(ev.clone());
        })
        .unwrap();

    let sess = app.sessions().unwrap().create(Some("no-llm".into()));
    for i in 0..4 {
        sess.append(SessionEvent::user(format!(
            "user msg {i} {}",
            "x".repeat(600)
        )));
        sess.append(SessionEvent::assistant_text(format!(
            "asst msg {i} {}",
            "y".repeat(600)
        )));
    }

    // 提前取出 CompactionRuntime 引用，随后卸载 LlmRuntime 使 ctx.llm() 返回 Err
    let compaction = app.compaction().unwrap();
    llm_fiber.dispose().await;
    let compact_res = compaction
        .compact_now(
            Arc::clone(&sess),
            ManualCompactOptions::default(),
            CancellationToken::new(),
        )
        .await;
    assert!(compact_res.is_err());

    assert_eq!(starts.lock().unwrap().len(), 1);
    {
        let end_ev = ends.lock().unwrap();
        assert_eq!(end_ev.len(), 1);
        assert_eq!(end_ev[0].outcome, CompactionOutcome::Failed);
    }
    let traces: Vec<SessionTraceEvent> = sess.with_all_entries(|entries| {
        entries
            .iter()
            .filter_map(|e| match &e.event {
                SessionEvent::Trace { detail } => Some(detail.clone()),
                _ => None,
            })
            .collect()
    });
    assert!(matches!(
        traces.first(),
        Some(SessionTraceEvent::CompactionStart { .. })
    ));
    assert!(matches!(
        traces.last(),
        Some(SessionTraceEvent::CompactionEnd { status, .. }) if status == "failed"
    ));
    app.fiber().dispose().await;

    // 2. ColdCache 遵循 keep_recent_tokens 下限与 regrowth_ratio，且 last_answered_route 不跨 provider 混用 model
    struct ProviderRecordingAdapter {
        provider_id: String,
        model_id: String,
        calls: Arc<Mutex<Vec<GenerateOptions>>>,
        window: u64,
    }

    impl LlmAdapter for ProviderRecordingAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: self.provider_id.clone(),
                name: self.provider_id.clone(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: self.model_id.clone(),
                    name: self.model_id.clone(),
                    context_window: Some(self.window),
                    max_tokens: Some(2_000),
                    reasoning_efforts: vec![ReasoningEffort::Off],
                    input: Vec::new(),
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.calls.lock().unwrap().push(options);
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                    index: 0,
                    text: "## Goal\n- Summary".into(),
                }));
                let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }));
            });
            Box::pin(rx)
        }
    }

    let app2 = Context::new();
    app2.plugin(LlmRuntime::plugin(), ()).await.unwrap();
    app2.plugin(SessionStore::plugin(), ()).await.unwrap();
    app2.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app2.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    // usable = 10_000 - 2_000 = 8_000; trigger = 3_200 (half=1_600), keep_recent = 2_000 (> 1_600), regrowth = 1_600
    let cfg = CompactionConfig {
        default_context_window: 10_000,
        soft_window_cap: 0,
        trigger_ratio: 0.40,
        keep_recent_ratio: 0.25,
        regrowth_ratio: 0.20,
        post_compact_max_ratio: 0.85,
        prune_threshold_chars: 100_000,
        cold_cache_idle_secs: 1,
        ..Default::default()
    };
    app2.plugin(CompactionRuntime::plugin(), cfg).await.unwrap();

    let calls_a = Arc::new(Mutex::new(Vec::<GenerateOptions>::new()));
    let calls_b = Arc::new(Mutex::new(Vec::<GenerateOptions>::new()));
    let a = Arc::new(ProviderRecordingAdapter {
        provider_id: "prov-a".into(),
        model_id: "model-a".into(),
        calls: Arc::clone(&calls_a),
        window: 10_000,
    });
    let b = Arc::new(ProviderRecordingAdapter {
        provider_id: "prov-b".into(),
        model_id: "model-b".into(),
        calls: Arc::clone(&calls_b),
        window: 10_000,
    });
    app2.plugin(
        PluginDef::sync("two-provs", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a_ref: Arc<dyn LlmAdapter> = Arc::clone(&a) as Arc<dyn LlmAdapter>;
            let b_ref: Arc<dyn LlmAdapter> = Arc::clone(&b) as Arc<dyn LlmAdapter>;
            ctx.effect("two-provs.reg", move || {
                let c1 = llm.register_adapter("prov-a", a_ref)?;
                let c2 = llm.register_adapter("prov-b", b_ref)?;
                Ok::<_, LlmError>(Some(cordis::Cleanup::sync(move || {
                    drop(c1);
                    drop(c2);
                })))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();

    let s2 = app2.sessions().unwrap().create(Some("cold-route".into()));
    for i in 0..6 {
        s2.append(SessionEvent::user(format!("m{i} {}", "a".repeat(1_000))));
        s2.append(SessionEvent::assistant_text(format!(
            "r{i} {}",
            "b".repeat(40)
        )));
    }
    // estimated ~ 1_800 tokens: 高于 trigger/2=1_600 但低于 keep_recent=2_000
    app2.emit(
        EVENT_AGENT_STEP_END,
        AgentStepEndEvent {
            session: Arc::clone(&s2),
            turn: 1,
            step: 1,
            outcome: StepOutcome::Completed,
            usage: None,
            provider: Some("prov-a".into()),
            model: None,
            ts_ms: 0,
        },
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let _ = app2
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&s2),
                turn: 2,
                step: 1,
                provider: Some("prov-b".into()),
                model: Some("model-b".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await;
    assert!(
        s2.header().compacted_up_to_seq.is_none(),
        "ColdCache must respect keep_recent_tokens floor"
    );

    // 追加两轮对话推高到 ~2_850（消息总计 > keep_recent=2_000 且整体 < trigger=3_200），空闲后触发 ColdCache 并使用 prov-a 且不带 model-b
    s2.append(SessionEvent::user(
        "u_extra_1 ".to_string() + &"c".repeat(2_000),
    ));
    s2.append(SessionEvent::assistant_text(
        "a_extra_1 ".to_string() + &"d".repeat(2_000),
    ));
    app2.emit(
        EVENT_AGENT_STEP_END,
        AgentStepEndEvent {
            session: Arc::clone(&s2),
            turn: 2,
            step: 1,
            outcome: StepOutcome::Completed,
            usage: None,
            provider: Some("prov-a".into()),
            model: None,
            ts_ms: 0,
        },
    );
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let _ = app2
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&s2),
                turn: 3,
                step: 1,
                provider: Some("prov-b".into()),
                model: Some("model-b".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await;
    assert!(s2.header().compacted_up_to_seq.is_some());
    assert_eq!(calls_a.lock().unwrap().len(), 1);
    assert_eq!(calls_b.lock().unwrap().len(), 0);
    assert_eq!(
        calls_a.lock().unwrap()[0].model,
        None,
        "last_answered_route on prov-a with None model must not mix model-b from prov-b"
    );

    // 紧接着未增长到 regrowth_tokens 即使再次空闲，也不得重复触发 ColdCache
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let _ = app2
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&s2),
                turn: 4,
                step: 1,
                provider: Some("prov-b".into()),
                model: Some("model-b".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await;
    assert_eq!(
        calls_a.lock().unwrap().len(),
        1,
        "regrowth_tokens hysteresis must block repeated ColdCache"
    );

    app2.fiber().dispose().await;
}

#[tokio::test]
async fn touch_session_activity_resets_cold_cache_idle_window() {
    use cordis::BoxFuture;
    use mizar_agent_loop::{
        AgentPreStepDecision, AgentPreStepInput, AgentStepEndEvent, EVENT_AGENT_PRE_STEP,
        EVENT_AGENT_STEP_END, StepOutcome,
    };

    // context_window = 12_000, max_tokens = 4_096 -> usable = 7_904
    // trigger_ratio = 0.40 -> trigger = 3_161 (half = 1_580), keep_recent = 1_976
    let (adapter, calls) = RecordingAdapter::new(12_000, Vec::new());
    let app = Context::new();
    app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    app.plugin(
        CompactionRuntime::plugin(),
        CompactionConfig {
            default_context_window: 12_000,
            soft_window_cap: 0,
            trigger_ratio: 0.40,
            keep_recent_ratio: 0.25,
            regrowth_ratio: 0.20,
            post_compact_max_ratio: 0.85,
            prune_threshold_chars: 100_000,
            cold_cache_idle_secs: 1,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let adapter_arc = Arc::new(adapter);
    app.plugin(
        PluginDef::sync("mock-prov-touch", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter_arc);
            ctx.effect("mock-prov-touch.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("mock-compact", a)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();

    let s = app.sessions().unwrap().create(Some("touch-idle".into()));
    for i in 0..8 {
        s.append(SessionEvent::user(format!("u{i} {}", "a".repeat(1_000))));
        s.append(SessionEvent::assistant_text(format!(
            "a{i} {}",
            "b".repeat(400)
        )));
    }
    // 记录 STEP_END 建立 last_activity_at 起点
    app.emit(
        EVENT_AGENT_STEP_END,
        AgentStepEndEvent {
            session: Arc::clone(&s),
            turn: 1,
            step: 1,
            outcome: StepOutcome::Completed,
            usage: None,
            provider: Some("mock-compact".into()),
            model: Some("m-test".into()),
            ts_ms: 0,
        },
    );

    // 经过 600ms 后调用 touch_session_activity，再等 600ms（距 STEP_END 已 1200ms > 1s，但距 touch 仅 600ms < 1s）
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    app.compaction().unwrap().touch_session_activity(&s.id);
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    let _ = app
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&s),
                turn: 2,
                step: 1,
                provider: Some("mock-compact".into()),
                model: Some("m-test".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await;
    assert!(
        s.header().compacted_up_to_seq.is_none(),
        "touch_session_activity must postpone ColdCache idle deadline"
    );
    assert_eq!(calls.lock().unwrap().len(), 0);

    // 再等 550ms 后已超过 1s 空闲阈值，应正常触发 ColdCache
    tokio::time::sleep(std::time::Duration::from_millis(550)).await;
    let _ = app
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&s),
                turn: 3,
                step: 1,
                provider: Some("mock-compact".into()),
                model: Some("m-test".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await;
    assert!(
        s.header().compacted_up_to_seq.is_some(),
        "ColdCache should trigger once idle elapses past last touch_session_activity"
    );
    assert_eq!(calls.lock().unwrap().len(), 1);

    app.fiber().dispose().await;
}
