use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Context, PluginDef, Service};
use futures::channel::mpsc;
use mizar_agent_loop::{
    AgentLoop, AgentLoopConfig, AgentLoopContextExt, AgentPreStepEvent, AgentStepEndEvent,
    EVENT_AGENT_PRE_STEP, EVENT_AGENT_STEP_END, StepOutcome,
};
use mizar_compaction::{
    CompactionConfig, CompactionContextExt, CompactionEndEvent, CompactionOutcome,
    CompactionRuntime, CompactionStartEvent, EVENT_COMPACTION_END, EVENT_COMPACTION_START,
    ManualCompactOptions, SUPERSEDED_NOTICE, USELESS_NOTICE,
};
use mizar_llm::{
    BoxChunkStream, ContentBlock, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt,
    LlmError, LlmErrorCode, LlmRuntime, ModelInfo, ProviderInfo, ReasoningEffort, StreamChunk,
    TokenUsage,
};
use mizar_session::{
    LoadedSession, PersistenceResult, SERVICE_SESSION_PERSISTENCE, SessionContextExt,
    SessionCreateOptions, SessionEntry, SessionEvent, SessionHeader, SessionHeaderPatch,
    SessionListQuery, SessionPersistence, SessionPersistenceHandle, SessionStore,
    SessionTraceEvent,
};
use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_tools::{ToolDefinition, ToolExecution, ToolExposure, ToolRuntime, ToolsContextExt};
use mizar_tools_coding::{CodingToolsConfig, CodingToolsPlugin};
use serde_json::json;
use tokio_util::sync::CancellationToken;

type MockResponse = Result<(String, FinishReason), LlmError>;

struct RecordingAdapter {
    provider_id: String,
    protocol: String,
    calls: Arc<Mutex<Vec<GenerateOptions>>>,
    responses: Arc<Mutex<Vec<MockResponse>>>,
    context_window: u64,
    max_tokens: Option<u32>,
    reasoning_efforts: Vec<ReasoningEffort>,
}

impl RecordingAdapter {
    fn new(
        context_window: u64,
        max_tokens: Option<u32>,
        responses: Vec<MockResponse>,
    ) -> (Self, Arc<Mutex<Vec<GenerateOptions>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let adapter = Self {
            provider_id: "mock-compact".into(),
            protocol: "mock".into(),
            calls: Arc::clone(&calls),
            responses: Arc::new(Mutex::new(responses)),
            context_window,
            max_tokens,
            reasoning_efforts: vec![ReasoningEffort::Off, ReasoningEffort::Medium],
        };
        (adapter, calls)
    }
}

impl LlmAdapter for RecordingAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: self.provider_id.clone(),
            name: "Mock Compact".into(),
            protocol: self.protocol.clone(),
            models: vec![ModelInfo {
                id: "m-test".into(),
                name: "Test Model".into(),
                context_window: Some(self.context_window),
                max_tokens: self.max_tokens,
                reasoning_efforts: self.reasoning_efforts.clone(),
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
    let provider_id = adapter.provider_id.clone();
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
            let pid = provider_id.clone();
            ctx.effect("mock-compact-adapter.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter(pid, a)?))
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
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();
    let compaction = app.compaction().unwrap();

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

    // 3 次主对话 + 1 次摘要请求 = 4 次模型调用
    assert_eq!(calls.lock().unwrap().len(), 4);

    app.fiber().dispose().await;
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
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();

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

    // 1 次压缩摘要 + 1 次主请求
    assert_eq!(calls.lock().unwrap().len(), 2);

    app.fiber().dispose().await;
}

#[tokio::test]
async fn pre_step_runs_cache_aware_tail_pruning_without_compacting_and_respects_prefix_bound_thinking()
 {
    // 1. 普通模型：步前自动裁剪热尾被覆盖 read 与零命中 grep，不触发 LLM 压缩，保留原文
    let (adapter, calls) = RecordingAdapter::new(
        128_000,
        Some(16_384),
        vec![Ok(("正常继续工作".into(), FinishReason::Stop))],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();

    let s = store.create(Some("tail-prune-prestep".into()));
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "r1".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "r1".into(),
        name: "read".into(),
        content: "old file content ".repeat(100),
        is_error: false,
    });
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "g1".into(),
            name: "grep".into(),
            arguments: r#"{"pattern":"missing_sym"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "g1".into(),
        name: "grep".into(),
        content: "No matches found".into(),
        is_error: false,
    });
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "r2".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "r2".into(),
        name: "read".into(),
        content: "fresh file content".into(),
        is_error: false,
    });

    let reply = agent
        .run(Arc::clone(&s), "继续", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply, "正常继续工作");
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert!(s.latest_compaction_boundary().is_none());

    let pruned = s.pruned_map();
    assert_eq!(pruned.get(&2).map(String::as_str), Some(SUPERSEDED_NOTICE));
    assert_eq!(pruned.get(&4).map(String::as_str), Some(USELESS_NOTICE));
    assert!(!pruned.contains_key(&6));

    let derived_str = format!("{:?}", s.derive_messages());
    assert!(derived_str.contains(SUPERSEDED_NOTICE));
    assert!(derived_str.contains(USELESS_NOTICE));
    assert!(matches!(
        &s.all_entries()[1].event,
        SessionEvent::ToolResult { content, .. } if content.len() == "old file content ".len() * 100
    ));
    app.fiber().dispose().await;

    // 2. 前缀绑定思考模型（anthropic-messages + reasoning_efforts）：热缓存期间热尾上限 0，不得破坏前缀
    let (mut anthropic_adapter, _) = RecordingAdapter::new(
        128_000,
        Some(16_384),
        vec![Ok(("thinking ok".into(), FinishReason::Stop))],
    );
    anthropic_adapter.protocol = "anthropic-messages".into();
    let app_think = setup_stack(anthropic_adapter, CompactionConfig::default(), None).await;
    let s_think = app_think
        .sessions()
        .unwrap()
        .create(Some("think-prune".into()));
    s_think.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "r1".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s_think.append(SessionEvent::ToolResult {
        tool_call_id: "r1".into(),
        name: "read".into(),
        content: "old file".into(),
        is_error: false,
    });
    s_think.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "r2".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    s_think.append(SessionEvent::ToolResult {
        tool_call_id: "r2".into(),
        name: "read".into(),
        content: "new file".into(),
        is_error: false,
    });

    let think_reply = app_think
        .agent_loop()
        .unwrap()
        .run(Arc::clone(&s_think), "继续", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(think_reply, "thinking ok");
    assert!(
        s_think.pruned_map().is_empty(),
        "prefix-bound thinking model must keep warm cache prefix intact"
    );
    app_think.fiber().dispose().await;
}

#[tokio::test]
async fn summarizer_omits_temperature_sets_reasoning_off_cites_recall_and_rewrites_tail_note() {
    let (adapter, calls) = RecordingAdapter::new(
        64_000,
        Some(4_096),
        vec![
            // 第 1 次验证 thinking/effort 错误时自动清空 reasoning_effort 回退重试
            Err(LlmError::new(
                LlmErrorCode::InvalidRequest,
                "unsupported thinking parameter",
            )),
            Ok((
                "<mizar-summary>\n## Goal & Scope\n- Refactor parser\n## Verified Facts & Decisions\n- Updated src/lib.rs\n</mizar-summary>".into(),
                FinishReason::Stop,
            )),
        ],
    );
    let app = setup_stack(adapter, CompactionConfig::default(), None).await;
    let store = app.sessions().unwrap();
    let tools = app.tools().unwrap();
    let compaction = app.compaction().unwrap();

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

        // 首次摘要请求：无 temperature，强制 ReasoningEffort::Off，引用统一工具名 `recall`，tail_note 不含 seq #
        let first = &recorded[0];
        assert_eq!(first.temperature, None);
        assert_eq!(first.reasoning_effort, Some(ReasoningEffort::Off));
        let prompt_str = match first.messages.last().unwrap() {
            mizar_llm::Message::User { content, .. } => content.clone(),
            other => format!("{other:?}"),
        };
        assert!(prompt_str.contains("`recall`"), "{prompt_str}");
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

    app.fiber().dispose().await;
}

#[tokio::test]
async fn unknown_model_falls_back_to_default_context_window_without_borrowing_first_model() {
    let (adapter, _) = RecordingAdapter::new(1_048_576, Some(131_072), vec![]);
    let cfg = CompactionConfig {
        default_context_window: 64_000,
        ..Default::default()
    };
    let app = setup_stack(adapter, cfg, None).await;
    let store = app.sessions().unwrap();
    let compaction = app.compaction().unwrap();
    let s = store.create(None);

    let unknown = compaction.inspect_context(&s, Some("mock-compact"), Some("unknown-model"));
    assert_eq!(unknown.context_window, 64_000);
    assert_eq!(unknown.effective_window, 64_000 - 16_384);

    let known = compaction.inspect_context(&s, Some("mock-compact"), Some("m-test"));
    assert_eq!(known.context_window, 1_048_576);
    assert_eq!(known.effective_window, 1_048_576 - 131_072);

    app.fiber().dispose().await;
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
    let store = app.sessions().unwrap();
    let agent = app.agent_loop().unwrap();

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
        assert!(
            summary_call.tools.is_empty(),
            "overflow should use clean transcript"
        );
        assert_eq!(summary_call.temperature, None);
        assert_eq!(summary_call.reasoning_effort, Some(ReasoningEffort::Off));
    }

    app.fiber().dispose().await;
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
    ) -> BoxFuture<'static, PersistenceResult<()>> {
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
    let store = app.sessions().unwrap();
    let tools = app.tools().unwrap();
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

    // 5. 先 read 再整文件 write -> 放行
    let r_big = tools
        .execute(
            ToolExecution::new("read", json!({ "path": "big60.txt" }), Some(s.id.clone()))
                .with_call_id("rc-big-1"),
        )
        .await;
    assert!(!r_big.is_error);
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
    });

    let big60_v2: String = (1..=60).map(|i| format!("v2_line_{i:02}\n")).collect();
    let w_after_read = tools
        .execute(
            ToolExecution::new(
                "write",
                json!({ "path": "big60.txt", "content": big60_v2 }),
                Some(s.id.clone()),
            )
            .with_call_id("wc-big-1"),
        )
        .await;
    assert!(
        !w_after_read.is_error && !w_after_read.blocked,
        "{w_after_read:?}"
    );
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "wc-big-1".into(),
            name: "write".into(),
            arguments: json!({ "path": "big60.txt" }).to_string(),
        }],
        replay: None,
        usage: None,
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "wc-big-1".into(),
        name: "write".into(),
        content: w_after_read.content.clone(),
        is_error: false,
    });

    // 6. 发生压缩，将此前的 read/write 推到 compacted_up_to_seq 之前 -> 再次直接 write 被拦截
    s.append(SessionEvent::user("after read and write"));
    s.compact("summary", 1, None);

    let w_stale = tools
        .execute(ToolExecution::new(
            "write",
            json!({ "path": "big60.txt", "content": "stale overwrite\n" }),
            Some(s.id.clone()),
        ))
        .await;
    assert!(w_stale.blocked && w_stale.is_error, "{w_stale:?}");
    assert!(w_stale.content.contains("compacted out of context"));

    // 6. 压后 <= 50 行的小文件与 edit 工具始终放行
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

    app.fiber().dispose().await;

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
            json!({ "path": "big60.txt", "content": "blocked after resume\n" }),
            Some("resumed-guard".into()),
        ))
        .await;
    assert!(w_resumed_blocked.blocked, "{w_resumed_blocked:?}");

    let _ = std::fs::remove_dir_all(&ws);
    app2.fiber().dispose().await;
}

#[tokio::test]
async fn missing_llm_pairs_compaction_traces_and_route_does_not_mix_models() {
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

    // 2. last_answered_route 不跨 provider 混用 model
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
    // usable = 10_000 - 2_000 = 8_000; trigger = 3_200, keep_recent = 2_000
    let cfg = CompactionConfig {
        default_context_window: 10_000,
        trigger_ratio: 0.40,
        keep_recent_tokens: 2_000,
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

    let s2 = app2.sessions().unwrap().create(Some("route-no-mix".into()));
    for i in 0..8 {
        s2.append(SessionEvent::user(format!("m{i} {}", "a".repeat(1_400))));
        s2.append(SessionEvent::assistant_text(format!(
            "r{i} {}",
            "b".repeat(1_400)
        )));
    }
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
    let _ = app2
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&s2),
                turn: 2,
                step: 1,
                provider: Some("prov-b".into()),
                model: Some("model-b".into()),
                cancel: CancellationToken::new(),
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

    app2.fiber().dispose().await;
}
