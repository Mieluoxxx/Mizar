use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::channel::mpsc;
use mizar_core::agent_loop::{AgentEvent, AgentLoop, AgentLoopConfig, AgentTurnInputEvent};
use mizar_core::session_title::{SessionTitleConfig, SessionTitleRuntime};
use mizar_core::system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmError, LlmErrorCode, LlmRuntime,
    ModelInfo, ProviderInfo, ReasoningEffort, StreamChunk,
};
use mizar_session::{SessionCreateOptions, SessionEvent, SessionStore, TitleSource};
use mizar_tools::ToolRuntime;
use mizar_types::HookGuard;

struct ScriptedReply {
    text: Result<String, LlmError>,
    delay_ms: u64,
}

struct RecordingTitleAdapter {
    provider_name: String,
    calls: Arc<Mutex<Vec<GenerateOptions>>>,
    replies: Arc<Mutex<VecDeque<ScriptedReply>>>,
    call_count: Arc<AtomicUsize>,
}

impl RecordingTitleAdapter {
    fn new(provider_name: &str, replies: Vec<ScriptedReply>) -> Arc<Self> {
        Arc::new(Self {
            provider_name: provider_name.to_string(),
            calls: Arc::new(Mutex::new(Vec::new())),
            replies: Arc::new(Mutex::new(replies.into())),
            call_count: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl LlmAdapter for RecordingTitleAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "test".to_string(),
            name: self.provider_name.clone(),
            protocol: "mock".to_string(),
            models: vec![ModelInfo {
                id: "fast-title".to_string(),
                name: "Fast Title".to_string(),
                context_window: Some(32_768),
                max_tokens: Some(4096),
                reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::Medium],
                input: Vec::new(),
                ..Default::default()
            }],
        })
    }

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        self.calls.lock().unwrap().push(options);
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ScriptedReply {
                text: Ok("EXP|默认响应".into()),
                delay_ms: 0,
            });
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            if reply.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(reply.delay_ms)).await;
            }
            match reply.text {
                Ok(text) => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text(text)));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
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

struct TitleTestHarness {
    llm: Arc<LlmRuntime>,
    sessions: Arc<SessionStore>,
    agent_loop: Arc<AgentLoop>,
    session_title: Arc<SessionTitleRuntime>,
    _guards: HookGuard,
}

async fn setup_app(
    adapter_provider: &str,
    adapter: Arc<dyn LlmAdapter>,
    cfg: SessionTitleConfig,
) -> (TitleTestHarness, HookGuard) {
    let llm = LlmRuntime::new();
    let sessions = SessionStore::new();
    let tools = ToolRuntime::new();
    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let agent_loop = AgentLoop::new(
        Arc::clone(&llm),
        tools,
        Arc::clone(&sessions),
        sp,
        AgentLoopConfig::default(),
    );
    let llm_cleanup = llm.register_adapter(adapter_provider, adapter).unwrap();
    let (session_title, title_guard) =
        SessionTitleRuntime::install(Arc::clone(&llm), &agent_loop, cfg);
    (
        TitleTestHarness {
            llm,
            sessions,
            agent_loop,
            session_title,
            _guards: title_guard,
        },
        llm_cleanup,
    )
}

async fn wait_until(mut cond: impl FnMut() -> bool) {
    for _ in 0..80 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("condition not satisfied before timeout");
}

#[tokio::test]
async fn turn1_and_periodic_cadence_update_title_using_only_user_messages() {
    let adapter = RecordingTitleAdapter::new(
        "Mock",
        vec![
            ScriptedReply {
                text: Ok("FEA｜实现会话标题生成".into()),
                delay_ms: 0,
            },
            ScriptedReply {
                text: Ok("OPT｜补全标题测试与周期刷新".into()),
                delay_ms: 0,
            },
        ],
    );

    let (app, _cleanup) = setup_app(
        "test",
        adapter.clone(),
        SessionTitleConfig {
            refresh_interval_turns: 3,
            ..Default::default()
        },
    )
    .await;

    let updated_events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&updated_events);
    let _ev_handle = app.sessions.on_title_updated(move |p| {
        sink.lock()
            .unwrap()
            .push((p.session.id.clone(), p.title.clone(), p.title_source));
    });

    let session = app.sessions.create_with(SessionCreateOptions {
        id: Some("s-cadence".into()),
        ..Default::default()
    });

    // Turn 1: 记录一条 User 与各种非 User 消息，应仅向标题模型发送 User 文本
    session.append(SessionEvent::user("请帮我实现会话标题生成"));
    session.append(SessionEvent::assistant_text("ASSISTANT_SECRET"));
    session.append(SessionEvent::ToolResult {
        tool_call_id: "c1".into(),
        name: "read".into(),
        content: "TOOL_SECRET".into(),
        is_error: false,
    });
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));

    wait_until(|| {
        session
            .header()
            .title
            .as_deref()
            .is_some_and(|t| t.contains("｜FEA｜实现会话标题生成"))
    })
    .await;

    assert_eq!(session.header().title_source, Some(TitleSource::Auto));
    assert_eq!(adapter.call_count.load(Ordering::SeqCst), 1);
    {
        let calls = adapter.calls.lock().unwrap();
        assert_eq!(calls[0].max_tokens, Some(2048));
        assert_eq!(calls[0].temperature, Some(0.2));
        assert_eq!(calls[0].reasoning_effort, Some(ReasoningEffort::Low));
        let user_prompt = calls[0].messages[1].text_content();
        assert!(user_prompt.contains("请帮我实现会话标题生成"));
        assert!(!user_prompt.contains("ASSISTANT_SECRET"));
        assert!(!user_prompt.contains("TOOL_SECRET"));
    }

    // Turn 2: 未命中刷新间隔，不应再调用模型
    session.append(SessionEvent::user("继续调试第 2 轮"));
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 2,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(adapter.call_count.load(Ordering::SeqCst), 1);

    // Turn 3: 命中 refresh_interval_turns=3，再次刷新标题
    session.append(SessionEvent::user("顺便补全周期刷新与测试"));
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 3,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));

    wait_until(|| {
        session
            .header()
            .title
            .as_deref()
            .is_some_and(|t| t.contains("｜OPT｜补全标题测试与周期刷新"))
    })
    .await;
    assert_eq!(adapter.call_count.load(Ordering::SeqCst), 2);
    assert!(
        updated_events
            .lock()
            .unwrap()
            .iter()
            .any(|(_, title, src)| {
                title
                    .as_deref()
                    .is_some_and(|t| t.contains("｜OPT｜补全标题测试与周期刷新"))
                    && *src == Some(TitleSource::Auto)
            })
    );
}

#[tokio::test]
async fn respects_manual_ownership_and_inflight_race_and_allows_explicit_regenerate() {
    let adapter = RecordingTitleAdapter::new(
        "Mock",
        vec![
            ScriptedReply {
                text: Ok("FEA｜后台滞后标题".into()),
                delay_ms: 80,
            },
            ScriptedReply {
                text: Ok("FIX｜手动重生成结果".into()),
                delay_ms: 0,
            },
        ],
    );

    let (app, _cleanup) = setup_app(
        "test",
        adapter.clone(),
        SessionTitleConfig {
            refresh_interval_turns: 2,
            ..Default::default()
        },
    )
    .await;

    let session = app.sessions.create_with(SessionCreateOptions {
        id: Some("s-own".into()),
        ..Default::default()
    });
    session.append(SessionEvent::user("测试人工改名保护"));
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));

    // 后台模型仍在飞行中时，用户人工改名，后台完成后不得覆盖
    tokio::time::sleep(Duration::from_millis(15)).await;
    assert!(session.set_title("我的手工标题"));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(session.header().title.as_deref(), Some("我的手工标题"));
    assert_eq!(session.header().title_source, Some(TitleSource::Manual));

    // Turn 2 到达：已是 Manual 所有权，不再触发自动刷新
    session.append(SessionEvent::user("第 2 轮提问"));
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 2,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(adapter.call_count.load(Ordering::SeqCst), 1);

    // 显式 /rename 无参数重生成可以重新生成并切回 Auto 所有权
    let rt = &app.session_title;
    let updated = rt
        .regenerate_now(
            Arc::clone(&session),
            Some("test".into()),
            Some("fast-title".into()),
        )
        .await
        .unwrap();
    assert!(updated.contains("｜FIX｜手动重生成结果"));
    assert_eq!(session.header().title.as_deref(), Some(updated.as_str()));
    assert_eq!(session.header().title_source, Some(TitleSource::Auto));
}

#[tokio::test]
async fn fails_silently_without_retry_and_skips_offline_mock() {
    // 1. 错误不重试且保留兜底标题
    let failing_adapter = RecordingTitleAdapter::new(
        "Mock",
        vec![ScriptedReply {
            text: Err(LlmError {
                code: LlmErrorCode::Unknown,
                message: "boom".into(),
                status: None,
                retry_after_ms: None,
            }),
            delay_ms: 0,
        }],
    );

    let (app, _cleanup) = setup_app(
        "test",
        failing_adapter.clone(),
        SessionTitleConfig {
            refresh_interval_turns: 20,
            ..Default::default()
        },
    )
    .await;

    let session = app.sessions.create_with(SessionCreateOptions {
        id: Some("s-fail".into()),
        ..Default::default()
    });
    session.append(SessionEvent::user("保留的兜底标题"));
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("test".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(failing_adapter.call_count.load(Ordering::SeqCst), 1);
    assert_eq!(session.header().title.as_deref(), Some("保留的兜底标题"));

    // 2. Offline Mock 静默跳过（0 次调用）
    let offline_adapter = RecordingTitleAdapter::new("Offline Mock", Vec::new());
    let (app2, _cleanup2) = setup_app(
        "mock",
        offline_adapter.clone(),
        SessionTitleConfig::default(),
    )
    .await;
    let s2 = app2.sessions.create_with(SessionCreateOptions {
        id: Some("s-offline".into()),
        ..Default::default()
    });
    s2.append(SessionEvent::user("离线模式测试"));
    app2.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&s2),
            turn: 1,
            provider: Some("mock".into()),
            model: Some("mock-v1".into()),
            ts_ms: s2.header().created_at_ms,
        }));
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(offline_adapter.call_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_or_skipped_attempt_does_not_latch_turn() {
    struct IdAdapter {
        id: &'static str,
        name: &'static str,
        calls: Arc<AtomicUsize>,
        replies: Arc<Mutex<VecDeque<ScriptedReply>>>,
    }

    impl LlmAdapter for IdAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: self.id.to_string(),
                name: self.name.to_string(),
                protocol: "mock".to_string(),
                models: vec![ModelInfo {
                    id: "fast-title".to_string(),
                    name: "Fast Title".to_string(),
                    context_window: Some(32_768),
                    max_tokens: Some(4096),
                    reasoning_efforts: vec![ReasoningEffort::Low],
                    input: Vec::new(),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let reply = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(ScriptedReply {
                    text: Ok("EXP|默认响应".into()),
                    delay_ms: 0,
                });
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                match reply.text {
                    Ok(text) => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text(text)));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::Stop,
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

    let real_calls = Arc::new(AtomicUsize::new(0));
    let real_adapter = Arc::new(IdAdapter {
        id: "real",
        name: "Real",
        calls: Arc::clone(&real_calls),
        replies: Arc::new(Mutex::new(VecDeque::from([
            ScriptedReply {
                text: Err(LlmError {
                    code: LlmErrorCode::Unknown,
                    message: "transient".into(),
                    status: None,
                    retry_after_ms: None,
                }),
                delay_ms: 0,
            },
            ScriptedReply {
                text: Ok("FIX|恢复重试成功".into()),
                delay_ms: 0,
            },
        ]))),
    });
    let mock_adapter = Arc::new(IdAdapter {
        id: "mock",
        name: "Offline Mock",
        calls: Arc::new(AtomicUsize::new(0)),
        replies: Arc::new(Mutex::new(VecDeque::new())),
    });

    let (app, _cleanup) =
        setup_app("real", real_adapter.clone(), SessionTitleConfig::default()).await;
    let _mock_cleanup = app.llm.register_adapter("mock", mock_adapter).unwrap();
    let adapter_calls = &real_calls;

    let session = app.sessions.create_with(SessionCreateOptions {
        id: Some("s-retry-turn".into()),
        ..Default::default()
    });
    session.append(SessionEvent::user("测试 turn 1 不被锁死"));

    // 1. Offline mock 在 turn=1 跳过后，不应锁死 turn=1
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("mock".into()),
            model: Some("mock-v1".into()),
            ts_ms: session.header().created_at_ms,
        }));
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(adapter_calls.load(Ordering::SeqCst), 0);

    // 2. 切到真实模型但首轮调用出错，出错后也不应永久锁死 turn=1
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("real".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));
    wait_until(|| adapter_calls.load(Ordering::SeqCst) == 1).await;

    // 3. 再次触发 turn=1 仍可生成并应用标题
    tokio::time::sleep(Duration::from_millis(20)).await;
    app.agent_loop
        .emit_event(AgentEvent::TurnInput(AgentTurnInputEvent {
            session: Arc::clone(&session),
            turn: 1,
            provider: Some("real".into()),
            model: Some("fast-title".into()),
            ts_ms: session.header().created_at_ms,
        }));
    wait_until(|| {
        session
            .header()
            .title
            .as_deref()
            .is_some_and(|t| t.contains("FIX｜恢复重试成功"))
    })
    .await;
    assert_eq!(adapter_calls.load(Ordering::SeqCst), 2);
}
