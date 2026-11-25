use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use mizar_core::agent_loop::{AgentLoop, AgentLoopConfig};
use mizar_core::compaction::{
    CompactionConfig, CompactionOutcome, CompactionRuntime, ManualCompactOptions,
};
use mizar_core::system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_llm::{
    BoxChunkStream, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmError,
    LlmErrorCode, LlmRuntime, ModelInfo, ProviderInfo, ReasoningEffort, RequestOrigin, StreamChunk,
    TokenUsage,
};
use mizar_session::{
    LoadedSession, PersistenceResult, SessionCreateOptions, SessionEntry, SessionEvent,
    SessionHeader, SessionHeaderPatch, SessionListQuery, SessionPersistence, SessionStore,
};
use mizar_tools::coding::{CodingToolsConfig, register_coding_tools};
use mizar_tools::{ToolDefinition, ToolExecution, ToolExposure, ToolRuntime};
use mizar_types::HookGuard;
use serde_json::json;
use tokio_util::sync::CancellationToken;

type MockResponse = Result<(String, FinishReason), LlmError>;

struct RecordingAdapter {
    calls: Arc<Mutex<Vec<GenerateOptions>>>,
    responses: Arc<Mutex<Vec<MockResponse>>>,
    context_window: u64,
    max_tokens: Option<u32>,
}

impl RecordingAdapter {
    fn new(
        context_window: u64,
        max_tokens: Option<u32>,
        responses: Vec<MockResponse>,
    ) -> (Self, Arc<Mutex<Vec<GenerateOptions>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let adapter = Self {
            calls: Arc::clone(&calls),
            responses: Arc::new(Mutex::new(responses)),
            context_window,
            max_tokens,
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
                max_tokens: self.max_tokens,
                reasoning_efforts: vec![ReasoningEffort::Off, ReasoningEffort::Medium],
                input: Vec::new(),
                ..Default::default()
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

struct CompactionHarness {
    sessions: Arc<SessionStore>,
    tools: Arc<ToolRuntime>,
    agent_loop: Arc<AgentLoop>,
    compaction: Arc<CompactionRuntime>,
    _guards: Vec<HookGuard>,
}

async fn setup_stack(
    adapter: RecordingAdapter,
    cfg: CompactionConfig,
    workspace: Option<std::path::PathBuf>,
) -> CompactionHarness {
    let llm = LlmRuntime::new();
    let sessions = SessionStore::new();
    let tools = ToolRuntime::new();
    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let agent_loop = AgentLoop::new(
        Arc::clone(&llm),
        Arc::clone(&tools),
        Arc::clone(&sessions),
        Arc::clone(&sp),
        AgentLoopConfig::default(),
    );
    let (compaction, compact_guard) =
        CompactionRuntime::install(Arc::clone(&llm), &sessions, &tools, sp, &agent_loop, cfg);
    let mut guards = vec![compact_guard];
    if let Some(ws) = workspace {
        sessions.set_default_cwd(Some(ws.display().to_string()));
        guards.extend(register_coding_tools(&tools, &CodingToolsConfig::new(ws)).unwrap());
    }
    guards.push(
        llm.register_adapter("mock-compact", Arc::new(adapter))
            .unwrap(),
    );
    CompactionHarness {
        sessions,
        tools,
        agent_loop,
        compaction,
        _guards: guards,
    }
}

#[tokio::test]
async fn one_million_window_does_not_trigger_at_200k_or_380k_and_triggers_at_80_percent() {
    let (adapter, calls) = RecordingAdapter::new(
        1_048_576,
        Some(131_072),
        vec![
            Ok(("reply at 200k".into(), FinishReason::Stop)),
            Ok(("reply at 380k".into(), FinishReason::Stop)),
            Ok((
                "## Goal & Scope\n- Compacted 1M session".into(),
                FinishReason::Stop,
            )),
            Ok(("reply after 80% compact".into(), FinishReason::Stop)),
        ],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = &app.sessions;
    let agent = &app.agent_loop;
    let compaction = &app.compaction;

    let s = store.create(Some("1m-window".into()));
    let pressure = compaction.inspect_context(&s, Some("mock-compact"), Some("m-test"));
    assert_eq!(pressure.effective_window, 917_504);
    assert_eq!(pressure.trigger_tokens, 734_003);

    // 追加足够超过 keep_recent_tokens(20k) 的历史正文，确保一旦越过 80% 阈值即可切分
    for i in 0..8 {
        s.append(SessionEvent::user(format!(
            "history turn {i}: {}",
            "a".repeat(12_000)
        )));
        s.append(SessionEvent::assistant_text(format!(
            "history reply {i}: {}",
            "b".repeat(12_000)
        )));
    }

    // 1. 200k 不触发
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "checkpoint 200k".into(),
        }],
        replay: None,
        usage: Some(TokenUsage {
            input_tokens: 200_000,
            ..TokenUsage::default()
        }),
    });
    let r1 = agent
        .run(Arc::clone(&s), "step 200k", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r1, "reply at 200k");
    assert!(s.latest_compaction_boundary().is_none());

    // 2. 380k 仍不触发
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "checkpoint 380k".into(),
        }],
        replay: None,
        usage: Some(TokenUsage {
            input_tokens: 380_000,
            ..TokenUsage::default()
        }),
    });
    let r2 = agent
        .run(Arc::clone(&s), "step 380k", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r2, "reply at 380k");
    assert!(s.latest_compaction_boundary().is_none());

    // 3. 达到 >= 734_003（80% 真实可用窗口）后步前自动触发压缩
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "checkpoint 735k".into(),
        }],
        replay: None,
        usage: Some(TokenUsage {
            input_tokens: 735_000,
            ..TokenUsage::default()
        }),
    });
    let r3 = agent
        .run(Arc::clone(&s), "step 735k", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r3, "reply after 80% compact");
    assert!(s.latest_compaction_boundary().is_some());

    {
        let recorded = calls.lock().unwrap();
        assert_eq!(
            recorded
                .iter()
                .filter(|c| c.origin == RequestOrigin::Compaction)
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn high_cache_hit_prompt_tokens_triggers_auto_compaction() {
    let (adapter, calls) = RecordingAdapter::new(
        128_000,
        Some(16_384),
        vec![
            Ok((
                "## Goal & Scope\n- Triggered via cache_read".into(),
                FinishReason::Stop,
            )),
            Ok(("continued after compaction".into(), FinishReason::Stop)),
        ],
    );
    let cfg = CompactionConfig {
        keep_recent_tokens: 1_000,
        ..Default::default()
    };
    let app = setup_stack(adapter, cfg, None).await;
    let store = &app.sessions;
    let agent = &app.agent_loop;

    let s = store.create(Some("cache-hit-anchor".into()));
    s.append(SessionEvent::user("early large user context ".repeat(200)));
    s.append(SessionEvent::assistant_text(
        "early large reply ".repeat(200),
    ));
    // 净 input_tokens 仅 4k，但 cache_read_tokens 达 88k -> prompt_tokens() = 92k > 89_292 (128k 可用窗口 80%)
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "cached turn".into(),
        }],
        replay: None,
        usage: Some(TokenUsage {
            input_tokens: 4_000,
            cache_read_tokens: Some(88_000),
            output_tokens: 120,
            ..TokenUsage::default()
        }),
    });

    let res = agent
        .run(Arc::clone(&s), "next step", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(res, "continued after compaction");
    assert!(s.latest_compaction_boundary().is_some());

    {
        let recorded = calls.lock().unwrap();
        assert!(
            recorded
                .iter()
                .any(|c| c.origin == RequestOrigin::Compaction),
            "expected compaction triggered by prompt_tokens() including cache_read"
        );
    }
}

#[tokio::test]
async fn summarizer_omits_temperature_sets_reasoning_off_preserves_prefix_and_rewrites_tail_note() {
    let (adapter, calls) = RecordingAdapter::new(
        64_000,
        Some(4_096),
        vec![
            // 第 1 次验证 reasoning 错误时自动清空 reasoning_effort 回退重试
            Err(LlmError::new(
                LlmErrorCode::InvalidRequest,
                "unsupported reasoning effort parameter",
            )),
            Ok((
                "<mizar-summary>\n## Goal & Scope\n- Refactor parser\n## Verified Facts & Decisions\n- Updated src/lib.rs\n</mizar-summary>".into(),
                FinishReason::Stop,
            )),
        ],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = &app.sessions;
    let tools = &app.tools;
    let compaction = &app.compaction;

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

    let s = store.create(Some("manual-prefix".into()));
    tools.activate_tools(&s.id, &["extra_deferred".to_string()]);

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
        content: "LARGE_TOOL_OUTPUT_UNPRUNED ".repeat(400),
        is_error: false,
        meta: None,
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
        meta: None,
    });
    s.append(SessionEvent::user(
        "BOUNDARY_MARKER_NEXT_TASK: continue parser cleanup".repeat(5),
    ));

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

    {
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 2);

        // 首次摘要请求：无 temperature，强制 ReasoningEffort::Off，不含 recall 引用，tail_note 不含 seq #
        let first = &recorded[0];
        assert_eq!(first.origin, RequestOrigin::Compaction);
        assert_eq!(first.temperature, None);
        assert_eq!(first.reasoning_effort, Some(ReasoningEffort::Off));
        let prompt_str = match first.messages.last().unwrap() {
            mizar_llm::Message::User { content, .. } => content.clone(),
            other => format!("{other:?}"),
        };
        assert!(!prompt_str.contains("`recall`"), "{prompt_str}");
        assert!(!prompt_str.contains("recall_read"), "{prompt_str}");
        assert!(
            !prompt_str.contains("starting at seq #") && !prompt_str.contains("BEFORE seq #"),
            "tail_note must not cite seq # handles absent from visible messages: {prompt_str}"
        );
        assert!(
            prompt_str.contains("User message") && prompt_str.contains("BOUNDARY_MARKER_NEXT_TASK"),
            "tail_note should preview boundary message role/content: {prompt_str}"
        );
        assert!(prompt_str.contains("focus on parser changes"));

        // reasoning fallback：保留无 temperature，且清空 reasoning_effort
        let fallback = &recorded[1];
        assert_eq!(fallback.temperature, None);
        assert_eq!(fallback.reasoning_effort, None);
    }

    // 确认压后 <mizar-digest> 硬事实保留且延迟激活工具已重置
    let derived = s.derive_messages();
    let first_msg = format!("{:?}", derived[0]);
    assert!(first_msg.contains("<mizar-digest>"), "{first_msg}");
    assert!(first_msg.contains("crates/demo/src/lib.rs"), "{first_msg}");
    assert!(first_msg.contains("cargo test -p demo"), "{first_msg}");
    assert!(
        !tools
            .list_declared(Some(&s.id))
            .iter()
            .any(|t| t.name == "extra_deferred")
    );
}

#[tokio::test]
async fn unknown_model_falls_back_to_default_context_window_without_borrowing_first_model() {
    let (adapter, _) = RecordingAdapter::new(1_048_576, Some(131_072), vec![]);
    let cfg = CompactionConfig {
        default_context_window: 64_000,
        ..Default::default()
    };
    let app = setup_stack(adapter, cfg, None).await;
    let store = &app.sessions;
    let compaction = &app.compaction;
    let s = store.create(None);

    let unknown = compaction.inspect_context(&s, Some("mock-compact"), Some("unknown-model"));
    assert_eq!(unknown.context_window, 64_000);
    assert_eq!(unknown.effective_window, 64_000 - 16_384);

    let known = compaction.inspect_context(&s, Some("mock-compact"), Some("m-test"));
    assert_eq!(known.context_window, 1_048_576);
    assert_eq!(known.effective_window, 1_048_576 - 131_072);
}

#[tokio::test]
async fn overflow_recovery_retries_with_zero_tail_and_clean_transcript() {
    let (adapter, calls) = RecordingAdapter::new(
        128_000,
        Some(16_384),
        vec![
            Err(LlmError::new(
                LlmErrorCode::ContextWindowExceeded,
                "maximum context length exceeded",
            )),
            Ok((
                "## Goal & Scope\n- Recovered summary".into(),
                FinishReason::Stop,
            )),
            Ok(("压缩后自动重试成功！".into(), FinishReason::Stop)),
        ],
    );

    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = &app.sessions;
    let agent = &app.agent_loop;

    let s = store.create(Some("overflow-recovery".into()));
    s.append(SessionEvent::user("历史消息 1 ".repeat(100)));
    s.append(SessionEvent::assistant_text("历史回复 1 ".repeat(100)));

    let res = agent
        .run(Arc::clone(&s), "触发溢出", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(res, "压缩后自动重试成功！");
    assert!(s.latest_compaction_boundary().is_some());

    {
        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        let summary_call = &recorded[1];
        assert_eq!(summary_call.origin, RequestOrigin::Compaction);
        assert!(
            summary_call.tools.is_empty(),
            "overflow should use clean transcript"
        );
        assert_eq!(summary_call.temperature, None);
        assert_eq!(summary_call.reasoning_effort, Some(ReasoningEffort::Off));
    }
}

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
    ) -> mizar_types::BoxFuture<'static, PersistenceResult<()>> {
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

#[tokio::test]
async fn unified_write_guard_enforces_50_line_first_read_and_post_compaction_reread() {
    let ws = std::env::temp_dir().join(format!("mizar_unified_write_guard_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();

    // small50.txt：恰好 50 行；big60.txt：60 行
    std::fs::write(ws.join("small50.txt"), "small\n".repeat(50)).unwrap();
    let big60_initial: String = (1..=60).map(|i| format!("line {i}\n")).collect();
    std::fs::write(ws.join("big60.txt"), &big60_initial).unwrap();

    let (adapter, _) = RecordingAdapter::new(
        128_000,
        Some(16_384),
        vec![Ok((
            "## Goal & Scope\n- Compacted with read guard".into(),
            FinishReason::Stop,
        ))],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), Some(ws.clone())).await;
    let store = &app.sessions;
    let tools = &app.tools;
    let compaction = &app.compaction;
    let s = store.create_with(SessionCreateOptions {
        id: Some("write-guard".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });

    // 1. 新建不存在的大文件 -> 放行
    let w_new = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "brand_new.txt", "content": "new\n".repeat(80) }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!w_new.is_error && !w_new.blocked, "{w_new:?}");

    // 2. 已存在且 <= 50 行的小文件，未 read 直接 write -> 放行
    let w_small = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "small50.txt", "content": "updated small\n".repeat(50) }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!w_small.is_error && !w_small.blocked, "{w_small:?}");

    // 3. 已存在且 > 50 行的大文件，本会话尚未 read -> 拦截 write
    let w_unread = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "overwrite without read\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(w_unread.blocked && w_unread.is_error, "{w_unread:?}");
    assert!(w_unread.content.contains("read") && w_unread.content.contains("edit"));

    // 4. 对未读大文件执行 edit -> 放行，且 edit 后仍未整文件 read，再次 write 依然拦截
    let e_big = tools
        .execute(ToolExecution::new(
            "edit",
            json!({ "path": "big60.txt", "old_text": "line 1\nline 2", "new_text": "line ONE\nline 2" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!e_big.is_error && !e_big.blocked, "{e_big:?}");

    let w_after_edit_only = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "overwrite after edit only\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(w_after_edit_only.blocked, "{w_after_edit_only:?}");

    // 5. read 大文件之后再 write -> 放行
    let r_big = tools
        .execute(
            ToolExecution::new("read", json!({ "path": "big60.txt" }), Some(s.id.clone()))
                .with_call_id("rc-big-1"),
        )
        .await;
    assert!(!r_big.is_error && !r_big.blocked, "{r_big:?}");
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "rc-big-1".into(),
            name: "read".into(),
            arguments: json!({ "path": "big60.txt" }).to_string(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "rc-big-1".into(),
        name: "read".into(),
        content: r_big.content.clone(),
        is_error: false,
        meta: None,
    });

    let v2_content: String = (1..=60).map(|i| format!("v2_line_{i:02}\n")).collect();
    let w_after_read = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": v2_content }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(
        !w_after_read.is_error && !w_after_read.blocked,
        "{w_after_read:?}"
    );

    // 6. 压缩把上述 read/write 折叠出上下文，再次 write big60.txt 必须被拦截
    s.append(SessionEvent::user("中间对话"));
    s.append(SessionEvent::user("保留尾部消息".repeat(20)));
    let compact = compaction
        .compact_now(
            Arc::clone(&s),
            ManualCompactOptions {
                provider: Some("mock-compact".into()),
                model: Some("m-test".into()),
                focus: None,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(compact.outcome, CompactionOutcome::Compacted);

    let w_stale = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "stale overwrite\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(w_stale.blocked && w_stale.is_error, "{w_stale:?}");
    assert!(w_stale.content.contains("compacted out of context"));

    // 压后对 small50.txt 直接 write 与对 big60.txt 执行 edit 均放行
    let w_small_post = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "small50.txt", "content": "post compact small\n".repeat(50) }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!w_small_post.is_error && !w_small_post.blocked);

    let e_post = tools
        .execute(ToolExecution::new(
            "edit",
            json!({ "path": "big60.txt", "old_text": "v2_line_01", "new_text": "v2_line_01_edited" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!e_post.is_error && !e_post.blocked);

    // 7. 压后重 read 再 write -> 放行；nested 调用始终放行
    let r_again = tools
        .execute(ToolExecution::new(
            "read",
            json!({ "path": "big60.txt" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!r_again.is_error);

    let w_reread = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "line v3\n".repeat(60) }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(!w_reread.is_error && !w_reread.blocked);

    let w_nested = tools
        .execute(
            ToolExecution::new(
                "write",
                json!({ "path": "big60.txt", "content": "nested write\n".repeat(60) }),
                Some(s.id.clone()),
            )
            .with_nested(true),
        )
        .await;
    assert!(!w_nested.is_error && !w_nested.blocked);

    drop(app);

    // 8. 持久化冷重载后仍能从历史事件重建 last_read_seq 并拦截压缩前读取的大文件覆写
    let persistence = Arc::new(MemPersistence::default());
    let mut header = SessionHeader::new("resumed-guard", 1000);
    header.cwd = Some(ws.display().to_string());
    persistence.create_session(&header).unwrap();
    persistence
        .append_events(
            "resumed-guard",
            &[
                SessionEntry {
                    seq: 1,
                    ts: 1001,
                    event: SessionEvent::Assistant {
                        blocks: vec![ContentBlock::ToolCall {
                            id: "tc-read".into(),
                            name: "read".into(),
                            arguments: r#"{"path":"big60.txt"}"#.into(),
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
                        content: "line v3\n".repeat(60),
                        is_error: false,
                        meta: None,
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
            ],
        )
        .unwrap();

    let (adapter2, _) = RecordingAdapter::new(128_000, Some(16_384), vec![]);
    let llm2 = LlmRuntime::new();
    let sessions2 =
        SessionStore::with_persistence(Arc::clone(&persistence) as Arc<dyn SessionPersistence>);
    let tools2 = ToolRuntime::new();
    let sp2 = SystemPrompt::new(SystemPromptConfig::default());
    let agent2 = AgentLoop::new(
        Arc::clone(&llm2),
        Arc::clone(&tools2),
        Arc::clone(&sessions2),
        Arc::clone(&sp2),
        AgentLoopConfig::default(),
    );
    let (_c2, _cg2) = CompactionRuntime::install(
        Arc::clone(&llm2),
        &sessions2,
        &tools2,
        sp2,
        &agent2,
        CompactionConfig::default(),
    );
    let _tool_guards2 =
        register_coding_tools(&tools2, &CodingToolsConfig::new(ws.clone())).unwrap();
    let _reg2 = llm2
        .register_adapter("mock-compact", Arc::new(adapter2))
        .unwrap();

    let _resumed = sessions2.get("resumed-guard").unwrap();
    let w_resumed_blocked = tools2
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "blocked after resume\n" }),
            Some("resumed-guard".into()),
        ))
        .await;
    assert!(w_resumed_blocked.blocked, "{w_resumed_blocked:?}");

    let _ = std::fs::remove_dir_all(&ws);
}
