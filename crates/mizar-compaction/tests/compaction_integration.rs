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
