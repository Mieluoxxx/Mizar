use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Context, Service};
use mizar_llm::{ContentBlock, Message, TokenUsage};
use mizar_session::{
    EntryVisibility, LoadedSession, PersistenceError, PersistenceResult,
    SERVICE_SESSION_PERSISTENCE, SessionContextExt, SessionCreateOptions, SessionEntry,
    SessionEvent, SessionHeader, SessionHeaderPatch, SessionListQuery, SessionPersistence,
    SessionPersistenceHandle, SessionStore, SessionTraceEvent, repair_dangling_tool_calls,
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
        self.events.lock().unwrap().entry(header.id.clone()).or_default();
        Ok(())
    }

    fn update_header(&self, session_id: &str, patch: SessionHeaderPatch) -> PersistenceResult<()> {
        let mut headers = self.headers.lock().unwrap();
        let h = headers
            .get_mut(session_id)
            .ok_or_else(|| PersistenceError::NotFound(session_id.to_string()))?;
        if let Some(title) = patch.title {
            h.title = Some(title);
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
        Ok(Some(LoadedSession { header, entries }))
    }

    fn list_sessions(&self, query: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
        let mut list: Vec<SessionHeader> = self
            .headers
            .lock()
            .unwrap()
            .values()
            .filter(|h| query.cwd.as_ref().is_none_or(|c| h.cwd.as_deref() == Some(c.as_str())))
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

    fn flush_async(&self, _session_id: Option<String>) -> BoxFuture<'static, PersistenceResult<()>> {
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
        Ok(LoadedSession { header, entries })
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
    s.append(SessionEvent::User {
        text: "  第一行标题\n第二行补充  ".into(),
    });
    s.append(SessionEvent::assistant_text("收到 1"));
    s.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "completed".into(),
    });

    s.append_trace(SessionTraceEvent::TurnStart { turn: 2 });
    s.append(SessionEvent::User { text: "继续".into() });
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
    assert!(matches!(&msgs[0], Message::User { content } if content.contains("前情：")));

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
    s1.append(SessionEvent::User {
        text: "帮我跑测试".into(),
    });
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
    };
    let repaired = repair_dangling_tool_calls(&mut loaded);
    assert!(repaired.is_empty());
}
