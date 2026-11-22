use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Context, Service};
use mizar_llm::{ContentBlock, ImageAttachment, Message, TokenUsage};
use mizar_session::{
    EVENT_SESSION_TITLE_UPDATED, EntryVisibility, LoadedSession, PersistenceError,
    PersistenceResult, SERVICE_SESSION_PERSISTENCE, SessionContextExt, SessionCreateOptions,
    SessionEntry, SessionEvent, SessionHeader, SessionHeaderPatch, SessionListQuery,
    SessionPersistence, SessionPersistenceHandle, SessionStore, SessionTitleUpdatedPayload,
    SessionTraceEvent, TitlePatch, TitleSource, TitleSourcePatch, derive_session_title,
    repair_dangling_tool_calls,
};

#[derive(Default)]
struct InMemoryMockBackend {
    headers: Mutex<HashMap<String, SessionHeader>>,
    events: Mutex<HashMap<String, Vec<SessionEntry>>>,
}

impl SessionPersistence for InMemoryMockBackend {
    fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
        self.headers
            .lock()
            .unwrap()
            .insert(header.id.clone(), header.clone());
        self.events
            .lock()
            .unwrap()
            .entry(header.id.clone())
            .or_default();
        Ok(())
    }

    fn update_header(&self, session_id: &str, patch: SessionHeaderPatch) -> PersistenceResult<()> {
        let mut headers = self.headers.lock().unwrap();
        let h = headers
            .get_mut(session_id)
            .ok_or_else(|| PersistenceError::NotFound(session_id.to_string()))?;
        match patch.title {
            TitlePatch::Keep => {}
            TitlePatch::Set(title) => h.title = Some(title),
            TitlePatch::Clear => h.title = None,
        }
        match patch.title_source {
            TitleSourcePatch::Keep => {}
            TitleSourcePatch::Set(source) => h.title_source = Some(source),
            TitleSourcePatch::Clear => h.title_source = None,
        }
        if let Some(ts) = patch.updated_at_ms {
            h.updated_at_ms = h.updated_at_ms.max(ts);
        }
        if let Some(seq) = patch.compacted_up_to_seq {
            h.compacted_up_to_seq = Some(seq);
        }
        Ok(())
    }

    fn load_session(&self, session_id: &str) -> PersistenceResult<Option<LoadedSession>> {
        let header = self.headers.lock().unwrap().get(session_id).cloned();
        let Some(header) = header else {
            return Ok(None);
        };
        let entries = self
            .events
            .lock()
            .unwrap()
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        let max_seq = entries.last().map(|e| e.seq);
        Ok(Some(LoadedSession {
            header,
            entries,
            max_persisted_seq: max_seq,
            skipped_events: 0,
        }))
    }

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        let mut list: Vec<SessionHeader> = self
            .headers
            .lock()
            .unwrap()
            .values()
            .filter(|h| {
                query
                    .cwd
                    .as_ref()
                    .is_none_or(|c| h.cwd.as_deref() == Some(c.as_str()))
            })
            .cloned()
            .collect();
        list.sort_by_key(|a| std::cmp::Reverse(a.updated_at_ms));
        if let Some(limit) = query.limit {
            list.truncate(limit);
        }
        Ok(list)
    }

    fn append_events(&self, session_id: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
        let mut ev_map = self.events.lock().unwrap();
        let vec = ev_map
            .get_mut(session_id)
            .ok_or_else(|| PersistenceError::NotFound(session_id.to_string()))?;
        vec.extend(entries.iter().cloned());
        if let Some(h) = self.headers.lock().unwrap().get_mut(session_id) {
            h.event_count = vec.len() as u64;
            h.surface_count = vec.iter().filter(|e| e.is_surface()).count() as u64;
            if let Some(last) = entries.last() {
                h.updated_at_ms = h.updated_at_ms.max(last.ts);
            }
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
        from_session_id: &str,
        up_to_seq: u64,
        new_header: &SessionHeader,
    ) -> PersistenceResult<LoadedSession> {
        let entries: Vec<SessionEntry> = self
            .events
            .lock()
            .unwrap()
            .get(from_session_id)
            .ok_or_else(|| PersistenceError::NotFound(from_session_id.to_string()))?
            .iter()
            .filter(|e| e.seq <= up_to_seq)
            .cloned()
            .collect();
        let mut header = new_header.clone();
        header.event_count = entries.len() as u64;
        header.surface_count = entries.iter().filter(|e| e.is_surface()).count() as u64;
        self.headers
            .lock()
            .unwrap()
            .insert(header.id.clone(), header.clone());
        self.events
            .lock()
            .unwrap()
            .insert(header.id.clone(), entries.clone());
        let max_seq = entries.last().map(|e| e.seq);
        Ok(LoadedSession {
            header,
            entries,
            max_persisted_seq: max_seq,
            skipped_events: 0,
        })
    }
}

#[tokio::test]
async fn separates_surface_from_log_only_and_compacts_cleanly() {
    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store = app.sessions().unwrap();
    let s = store.create_with(SessionCreateOptions {
        cwd: Some("/repo/demo".into()),
        ..Default::default()
    });

    s.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
    s.append(SessionEvent::user("  第一行标题\n第二行补充  "));
    s.append(SessionEvent::assistant_text("收到 1"));
    s.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "completed".into(),
    });

    s.append_trace(SessionTraceEvent::TurnStart { turn: 2 });
    s.append(SessionEvent::user("继续"));
    s.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: "{}".into(),
        }],
        replay: None,
        usage: Some(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        }),
    });
    s.append(SessionEvent::ToolResult {
        tool_call_id: "c1".into(),
        name: "read".into(),
        content: "file body".into(),
        is_error: false,
    });
    s.append(SessionEvent::assistant_text("完成"));

    assert_eq!(s.header().title.as_deref(), Some("第一行标题"));
    // log() 仅返回 Surface 事件，不含 TurnStart/TurnEnd
    assert_eq!(s.log().len(), 6);
    assert_eq!(s.all_entries().len(), 9);
    assert_eq!(s.derive_messages().len(), 6);

    // 保留最近 2 条表面事件（ToolResult 会自动向前回退包含它的 Assistant(ToolCall) -> 共保留 3 条）
    let comp_seq = s.compact("前情：用户先打招呼并执行了工具准备", 2, Some(500));
    assert!(comp_seq > 0);
    // 底层完整日志仍保留全部事件
    assert_eq!(s.all_entries().len(), 10);

    let vis = s.visible_log();
    assert!(matches!(vis[0].event, SessionEvent::Compaction { .. }));
    assert!(matches!(vis[1].event, SessionEvent::Assistant { .. }));
    assert!(matches!(vis[2].event, SessionEvent::ToolResult { .. }));
    assert!(matches!(vis[3].event, SessionEvent::Assistant { .. }));

    let msgs = s.derive_messages();
    assert_eq!(msgs.len(), 4);
    assert!(matches!(&msgs[0], Message::User { content, .. } if content.contains("前情：")));

    // 序列化 round-trip 验证
    for entry in s.all_entries() {
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: SessionEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, entry);
        assert_eq!(
            decoded.visibility(),
            if matches!(decoded.event, SessionEvent::Trace { .. }) {
                EntryVisibility::LogOnly
            } else {
                EntryVisibility::Surface
            }
        );
    }

    app.fiber().dispose().await;
}

#[tokio::test]
async fn hydrates_repairs_dangling_tools_and_forks_via_seam() {
    let backend = Arc::new(InMemoryMockBackend::default());

    let app1 = Context::new();
    app1.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend.clone())),
    )
    .unwrap();
    app1.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store1 = app1.sessions().unwrap();
    let s1 = store1.create_with(SessionCreateOptions {
        id: Some("sess-1".into()),
        cwd: Some("/repo/app".into()),
        ..Default::default()
    });
    s1.append_trace(SessionTraceEvent::TurnStart { turn: 3 });
    s1.append(SessionEvent::user("帮我跑测试"));
    s1.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "call_hang".into(),
            name: "bash".into(),
            arguments: r#"{"command":"cargo test"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    app1.fiber().dispose().await;

    // 模拟全新进程重启后懒加载与断线自愈
    let app2 = Context::new();
    app2.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend)),
    )
    .unwrap();
    app2.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store2 = app2.sessions().unwrap();

    assert!(store2.list().is_empty());
    let loaded = store2.get("sess-1").expect("should load from backend");
    let surface = loaded.log();
    assert_eq!(surface.len(), 3);
    assert!(matches!(
        &surface[2].event,
        SessionEvent::ToolResult {
            tool_call_id,
            is_error: true,
            content,
            ..
        } if tool_call_id == "call_hang" && content.contains("TOOL_OUTCOME_UNKNOWN")
    ));
    // turn_counter 恢复为上一轮次 3，继续 next_turn() 应得到 4
    assert_eq!(loaded.next_turn(), 4);

    // Fork 前 2 条事件（恰好在 User 之后），验证独立分叉及 parent_id/fork_seq 记录
    let forked = store2
        .fork("sess-1", 2, SessionCreateOptions::default())
        .unwrap();
    let fh = forked.header();
    assert_eq!(fh.parent_id.as_deref(), Some("sess-1"));
    assert_eq!(fh.fork_seq, Some(2));
    assert_eq!(forked.log().len(), 1);

    let list = store2
        .list_headers(&SessionListQuery {
            cwd: Some("/repo/app".into()),
            limit: Some(10),
        })
        .unwrap();
    assert_eq!(list.len(), 2);

    app2.fiber().dispose().await;
}

#[test]
fn repair_helper_noop_when_tool_already_resolved() {
    let mut loaded = LoadedSession {
        header: SessionHeader::new("s", 1),
        entries: vec![
            SessionEntry {
                seq: 1,
                event: SessionEvent::Assistant {
                    blocks: vec![ContentBlock::ToolCall {
                        id: "c1".into(),
                        name: "read".into(),
                        arguments: "{}".into(),
                    }],
                    replay: None,
                    usage: None,
                },
                ts: 1,
            },
            SessionEntry {
                seq: 2,
                event: SessionEvent::ToolResult {
                    tool_call_id: "c1".into(),
                    name: "read".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                ts: 2,
            },
        ],
        max_persisted_seq: Some(2),
        skipped_events: 0,
    };
    let repaired = repair_dangling_tool_calls(&mut loaded);
    assert!(repaired.is_empty());
}

#[tokio::test]
async fn bash_execution_separates_context_and_log_only_across_derive_and_compaction() {
    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store = app.sessions().unwrap();
    let s = store.create(Some("sess-bash".into()));

    let ctx_ev = SessionEvent::BashExecution {
        command: "echo hello".into(),
        output: "hello\n".into(),
        exit_code: Some(0),
        cancelled: false,
        exclude_from_context: false,
    };
    let local_ev = SessionEvent::BashExecution {
        command: "cat secret.txt".into(),
        output: "SECRET_LOCAL_ONLY\n".into(),
        exit_code: Some(0),
        cancelled: false,
        exclude_from_context: true,
    };
    let err_ev = SessionEvent::BashExecution {
        command: "false".into(),
        output: "".into(),
        exit_code: Some(2),
        cancelled: false,
        exclude_from_context: false,
    };
    let cancel_ev = SessionEvent::BashExecution {
        command: "sleep 10".into(),
        output: "partial\n".into(),
        exit_code: Some(-1),
        cancelled: true,
        exclude_from_context: false,
    };

    assert_eq!(ctx_ev.visibility(), EntryVisibility::Surface);
    assert_eq!(local_ev.visibility(), EntryVisibility::LogOnly);
    assert_eq!(ctx_ev.kind_name(), "bash-execution");

    // 序列化往返保真
    let roundtrip: SessionEvent =
        serde_json::from_str(&serde_json::to_string(&local_ev).unwrap()).unwrap();
    assert_eq!(roundtrip, local_ev);

    s.append(ctx_ev);
    s.append(local_ev);
    s.append(err_ev);
    s.append(cancel_ev);
    s.append(SessionEvent::user("分析以上命令输出"));

    // 首条事件为 BashExecution 时自动派生标题
    assert_eq!(s.header().title.as_deref(), Some("! echo hello"));

    // session.log() 仅包含表面事件（不含 !!）
    assert_eq!(s.log().len(), 4);
    assert_eq!(s.all_entries().len(), 5);

    // derive_messages() 必须包含 !（含命令、输出、退出码与取消标记），彻底排除 !!
    let msgs = s.derive_messages();
    assert_eq!(msgs.len(), 4);
    match &msgs[0] {
        Message::User { content, .. } => {
            assert!(content.contains("Ran `echo hello`"), "{content}");
            assert!(content.contains("hello"), "{content}");
        }
        other => panic!("unexpected msg[0]: {other:?}"),
    }
    match &msgs[1] {
        Message::User { content, .. } => {
            assert!(content.contains("Ran `false`"), "{content}");
            assert!(content.contains("(no output)"), "{content}");
            assert!(content.contains("Command exited with code 2"), "{content}");
        }
        other => panic!("unexpected msg[1]: {other:?}"),
    }
    match &msgs[2] {
        Message::User { content, .. } => {
            assert!(content.contains("Ran `sleep 10`"), "{content}");
            assert!(content.contains("partial"), "{content}");
            assert!(content.contains("(command cancelled)"), "{content}");
        }
        other => panic!("unexpected msg[2]: {other:?}"),
    }
    for m in &msgs {
        if let Message::User { content, .. } = m {
            assert!(!content.contains("SECRET_LOCAL_ONLY"), "{content}");
            assert!(!content.contains("secret.txt"), "{content}");
        }
    }

    // Compaction 后 !! 依旧不可见
    s.append(SessionEvent::BashExecution {
        command: "ls /private".into(),
        output: "PRIVATE_AFTER_COMPACT".into(),
        exit_code: Some(0),
        cancelled: false,
        exclude_from_context: true,
    });
    s.compact("summary", 1, None);
    let after_compact = s.derive_messages();
    assert!(after_compact.iter().all(|m| match m {
        Message::User { content, .. } => {
            !content.contains("SECRET_LOCAL_ONLY") && !content.contains("PRIVATE_AFTER_COMPACT")
        }
        _ => true,
    }));

    app.fiber().dispose().await;
}

#[tokio::test]
async fn selection_is_log_only_deduped_and_survives_derive_messages() {
    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store = app.sessions().unwrap();
    let s = store.create(Some("sess-selection".into()));

    assert_eq!(s.latest_selection(), None);

    // 首次写入：provider + model + effort 全给
    assert!(s.record_selection(
        Some("magpie".into()),
        Some("mayoru-responses/deepseek-flash".into()),
        Some(mizar_llm::ReasoningEffort::High),
    ));
    assert_eq!(
        s.latest_selection(),
        Some(mizar_session::SessionSelection {
            provider: Some("magpie".into()),
            model: Some("mayoru-responses/deepseek-flash".into()),
            reasoning_effort: Some(mizar_llm::ReasoningEffort::High),
        })
    );

    // 完全相同不重复写
    assert!(!s.record_selection(
        Some("magpie".into()),
        Some("mayoru-responses/deepseek-flash".into()),
        Some(mizar_llm::ReasoningEffort::High),
    ));
    assert_eq!(s.all_entries().len(), 1);

    // 只改强度也算变化
    assert!(s.record_selection(
        Some("magpie".into()),
        Some("mayoru-responses/deepseek-flash".into()),
        Some(mizar_llm::ReasoningEffort::Max),
    ));
    assert_eq!(s.all_entries().len(), 2);
    assert_eq!(
        s.latest_selection().unwrap().reasoning_effort,
        Some(mizar_llm::ReasoningEffort::Max)
    );

    // LogOnly：不进模型上下文、不出现在 surface 视图里
    assert_eq!(
        SessionEvent::Selection {
            provider: None,
            model: None,
            reasoning_effort: None,
        }
        .visibility(),
        EntryVisibility::LogOnly
    );
    assert!(s.log().is_empty());
    assert!(s.derive_messages().is_empty());
    assert_eq!(s.header().surface_count, 0);

    // 但仍在完整事件流里
    assert_eq!(s.all_entries().len(), 2);

    app.fiber().dispose().await;
}

#[tokio::test]
async fn user_with_images_persists_through_backend_and_restores_via_derive_messages() {
    let backend = Arc::new(InMemoryMockBackend::default());
    let img = ImageAttachment {
        media_type: "image/png".into(),
        data: "iVBORw0KGgo=".into(),
        width: Some(640),
        height: Some(480),
    };

    let app1 = Context::new();
    app1.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend.clone())),
    )
    .unwrap();
    app1.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store1 = app1.sessions().unwrap();
    let s1 = store1.create_with(SessionCreateOptions {
        id: Some("sess-images".into()),
        ..Default::default()
    });
    s1.append(SessionEvent::user_with_images(
        "请分析这张截图里的报错",
        vec![img.clone()],
    ));
    s1.flush_async().await.unwrap();
    app1.fiber().dispose().await;

    let app2 = Context::new();
    app2.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend)),
    )
    .unwrap();
    app2.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store2 = app2.sessions().unwrap();
    let reloaded = store2
        .get("sess-images")
        .expect("reloaded session should exist");

    assert_eq!(
        reloaded.header().title.as_deref(),
        Some("请分析这张截图里的报错")
    );
    let msgs = reloaded.derive_messages();
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        msgs[0],
        Message::user_with_images("请分析这张截图里的报错", vec![img])
    );

    app2.fiber().dispose().await;
}

#[test]
fn derive_session_title_strips_noise_and_keeps_valid_tokens() {
    assert_eq!(
        derive_session_title("mizar --resume 12345678-1234-1234-1234-123456789abc 继续修 bug"),
        Some("继续修 bug".into())
    );
    assert_eq!(
        derive_session_title(
            "./target/debug/mizar -C /home/user/repo -r 12345678-1234-1234-1234-123456789abc"
        ),
        None
    );
    assert_eq!(
        derive_session_title(
            "mizar --resume 12345678-1234-1234-1234-123456789abc\n排查会话标题生成问题"
        ),
        Some("排查会话标题生成问题".into())
    );
    assert_eq!(
        derive_session_title(
            "请检查 /home/antigravity/Mieluoxxx/Mizar/crates/mizar-session/src/model.rs"
        ),
        Some("请检查 src/model.rs".into())
    );
    assert_eq!(
        derive_session_title(
            "打开 C:\\Users\\alice\\repo\\src\\main.rs 并访问 https://example.com/a/b/c 与 /settings"
        ),
        Some("打开 src\\main.rs 并访问 https://example.com/a/b/c 与 /settings".into())
    );
    assert_eq!(
        derive_session_title("恢复会话 12345678-90ab-cdef-1234-567890abcdef 的上下文"),
        Some("恢复会话 12345678 的上下文".into())
    );
}

#[tokio::test]
async fn set_title_clear_title_emit_event_and_fork_does_not_copy_parent_title() {
    let backend = Arc::new(InMemoryMockBackend::default());
    let app = Context::new();
    app.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend.clone())),
    )
    .unwrap();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();

    let store = app.sessions().unwrap();

    type RecordedTitleEvent = (String, Option<String>, Option<TitleSource>);
    let events: Arc<Mutex<Vec<RecordedTitleEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_sink = events.clone();
    let _title_handle = app
        .on(
            EVENT_SESSION_TITLE_UPDATED,
            move |p: &SessionTitleUpdatedPayload| {
                events_sink.lock().unwrap().push((
                    p.session.id.clone(),
                    p.title.clone(),
                    p.title_source,
                ));
            },
        )
        .unwrap();

    let parent = store.create_with(SessionCreateOptions {
        id: Some("p-1".into()),
        ..Default::default()
    });
    parent.append(SessionEvent::user("修复登录问题"));
    assert_eq!(parent.header().title.as_deref(), Some("修复登录问题"));
    assert_eq!(parent.header().title_source, Some(TitleSource::Auto));

    // 空标题、无变化调用均返回 false 且不发事件
    assert!(!parent.set_title("   \n  "));
    assert!(!parent.set_title_with_source("修复登录问题", TitleSource::Auto));

    assert!(parent.set_title("  0320｜FIX｜登录认证修复  "));
    assert_eq!(
        parent.header().title.as_deref(),
        Some("0320｜FIX｜登录认证修复")
    );
    assert_eq!(parent.header().title_source, Some(TitleSource::Manual));
    assert_eq!(
        backend.headers.lock().unwrap()["p-1"].title_source,
        Some(TitleSource::Manual)
    );

    assert!(parent.clear_title());
    assert_eq!(parent.header().title, None);
    assert_eq!(parent.header().title_source, None);
    assert!(!parent.clear_title());
    assert_eq!(backend.headers.lock().unwrap()["p-1"].title, None);

    // 重新给父会话设标题并 fork：默认不继承父标题，除非 opts.title 显式指定
    assert!(parent.set_title("父会话标题"));
    let forked_no_title = store
        .fork(
            "p-1",
            1,
            SessionCreateOptions {
                id: Some("f-1".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(forked_no_title.header().title, None);
    assert_eq!(forked_no_title.header().title_source, None);

    let forked_with_title = store
        .fork(
            "p-1",
            1,
            SessionCreateOptions {
                id: Some("f-2".into()),
                title: Some("自定义分支标题".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        forked_with_title.header().title.as_deref(),
        Some("自定义分支标题")
    );
    assert_eq!(
        forked_with_title.header().title_source,
        Some(TitleSource::Manual)
    );

    let recorded = events.lock().unwrap().clone();
    assert_eq!(
        recorded,
        vec![
            (
                "p-1".into(),
                Some("修复登录问题".into()),
                Some(TitleSource::Auto)
            ),
            (
                "p-1".into(),
                Some("0320｜FIX｜登录认证修复".into()),
                Some(TitleSource::Manual)
            ),
            ("p-1".into(), None, None),
            (
                "p-1".into(),
                Some("父会话标题".into()),
                Some(TitleSource::Manual)
            ),
        ]
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn repair_skips_past_unparsed_tail_seq_and_inspect_readonly_does_not_mutate_or_cache() {
    use mizar_session::PersistenceError;
    use std::collections::HashMap;

    struct SkippedTailBackend {
        headers: Mutex<HashMap<String, SessionHeader>>,
        raw_max_seq: u64,
        entries: Vec<SessionEntry>,
        appended: Arc<Mutex<Vec<SessionEntry>>>,
    }

    impl SessionPersistence for SkippedTailBackend {
        fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
            self.headers
                .lock()
                .unwrap()
                .insert(header.id.clone(), header.clone());
            Ok(())
        }

        fn update_header(&self, _: &str, _: SessionHeaderPatch) -> PersistenceResult<()> {
            Ok(())
        }

        fn load_session(&self, session_id: &str) -> PersistenceResult<Option<LoadedSession>> {
            let Some(header) = self.headers.lock().unwrap().get(session_id).cloned() else {
                return Ok(None);
            };
            Ok(Some(LoadedSession {
                header,
                entries: self.entries.clone(),
                max_persisted_seq: Some(self.raw_max_seq),
                skipped_events: 1,
            }))
        }

        fn list_sessions(&self, _: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
            Ok(self.headers.lock().unwrap().values().cloned().collect())
        }

        fn append_events(&self, _: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
            self.appended
                .lock()
                .unwrap()
                .extend(entries.iter().cloned());
            Ok(())
        }

        fn flush(&self, _: Option<&str>) -> PersistenceResult<()> {
            Ok(())
        }

        fn flush_async(&self, _: Option<String>) -> BoxFuture<'static, PersistenceResult<()>> {
            Box::pin(async { Ok(()) })
        }

        fn fork_session(
            &self,
            from_session_id: &str,
            _: u64,
            _: &SessionHeader,
        ) -> PersistenceResult<LoadedSession> {
            Err(PersistenceError::NotFound(from_session_id.to_string()))
        }
    }

    let appended = Arc::new(Mutex::new(Vec::new()));
    let mut header = SessionHeader::new("cold-parent", 10);
    header.parent_id = Some("cold-grand".into());
    header.compacted_up_to_seq = None;

    let mut headers = HashMap::new();
    headers.insert("cold-parent".into(), header);
    headers.insert("cold-grand".into(), SessionHeader::new("cold-grand", 5));

    let backend = Arc::new(SkippedTailBackend {
        headers: Mutex::new(headers),
        raw_max_seq: 5, // 例如 seq=5 是未来版本写出的未知事件，被跳过了
        entries: vec![
            SessionEntry {
                seq: 1,
                event: SessionEvent::Compaction {
                    summary: "earlier".into(),
                    first_kept_seq: 2,
                    tokens_before: Some(100),
                    tokens_after: Some(20),
                    digest: Default::default(),
                },
                ts: 11,
            },
            SessionEntry {
                seq: 2,
                event: SessionEvent::Assistant {
                    blocks: vec![ContentBlock::ToolCall {
                        id: "hang-1".into(),
                        name: "bash".into(),
                        arguments: "{}".into(),
                    }],
                    replay: None,
                    usage: None,
                },
                ts: 12,
            },
        ],
        appended: Arc::clone(&appended),
    });

    let app = Context::new();
    app.provide(
        mizar_session::SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(backend)),
    )
    .unwrap();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store = app.sessions().unwrap();

    // 1. 只读遍历 lineage / inspect_readonly 不应写入修复事件，也不应水合缓存
    let chain = store.lineage_session_ids("cold-parent");
    assert_eq!(chain, vec!["cold-parent", "cold-grand"]);
    assert!(appended.lock().unwrap().is_empty());
    assert!(!store.contains_in_memory("cold-parent"));

    let snapshot = store.inspect_readonly("cold-parent").unwrap();
    assert_eq!(snapshot.skipped_events, 1);
    assert_eq!(snapshot.max_persisted_seq, Some(5));
    assert!(appended.lock().unwrap().is_empty());

    // 2. 真正 get() 触发修复时，reparied entry seq 必须跳过 max_persisted_seq=5，且 header.compacted_up_to_seq 被推导
    let loaded_sess = store.get("cold-parent").unwrap();
    assert_eq!(loaded_sess.header().compacted_up_to_seq, Some(2));
    let written = appended.lock().unwrap().clone();
    assert_eq!(written.len(), 1);
    assert_eq!(written[0].seq, 6);

    // 3. 原子 compare_and_set_title_auto 应拒绝并发手动/不同旧值的覆盖
    assert_eq!(
        loaded_sess.compare_and_set_title_auto(None, "auto-one", true),
        Ok("auto-one".into())
    );
    assert!(loaded_sess.set_title("manual-override"));
    assert_eq!(
        loaded_sess.compare_and_set_title_auto(Some("auto-one"), "stale-auto", true),
        Err(Some("manual-override".into()))
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn usage_event_is_log_only_excluded_from_derived_messages_and_roundtrips_json() {
    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let store = app.sessions().unwrap();
    let s = store.create(None);

    s.append(SessionEvent::user("hi"));
    let usage = TokenUsage {
        input_tokens: 120,
        output_tokens: 1,
        cache_read_tokens: Some(4_000),
        cache_write_tokens: Some(0),
        total_tokens: Some(4_121),
        reasoning_tokens: None,
    };
    let u_ev = SessionEvent::usage(
        "cache_warm",
        "anthropic",
        "claude-sonnet-4-6",
        usage,
        Some("ttl=5m".into()),
    );
    assert_eq!(u_ev.visibility(), EntryVisibility::LogOnly);
    assert_eq!(u_ev.kind_name(), "usage");
    s.append(u_ev.clone());
    s.append(SessionEvent::usage(
        "session_title",
        "openai",
        "gpt-4o-mini",
        usage,
        None,
    ));

    // 只进入 all_entries，不进入表面 log()/visible_log()，也不进入 derive_messages()
    assert_eq!(s.log().len(), 1);
    assert_eq!(s.visible_log().len(), 1);
    assert_eq!(s.all_entries().len(), 3);
    assert_eq!(s.header().surface_count, 1);
    assert_eq!(s.header().event_count, 3);
    let msgs = s.derive_messages();
    assert_eq!(msgs, vec![Message::user("hi")]);

    let json = serde_json::to_string(&u_ev).unwrap();
    assert!(json.contains("\"type\":\"usage\""));
    assert!(json.contains("\"note\":\"ttl=5m\""));
    let decoded: SessionEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, u_ev);

    let no_note = SessionEvent::usage("subagent", "openai", "gpt-5", usage, None);
    let no_note_json = serde_json::to_string(&no_note).unwrap();
    assert!(!no_note_json.contains("\"note\""));
    assert_eq!(
        serde_json::from_str::<SessionEvent>(&no_note_json).unwrap(),
        no_note
    );

    app.fiber().dispose().await;
}
