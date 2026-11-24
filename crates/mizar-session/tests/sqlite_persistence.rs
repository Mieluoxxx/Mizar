use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_session::{
    SessionCreateOptions, SessionEvent, SessionListQuery, SessionPersistence, SessionStore,
    SessionTraceEvent, SqliteSessionPersistence, TitleSource, TraceSeverity,
};
use mizar_types::llm::{ContentBlock, Message, ReplayEnvelope, TokenUsage};

fn temp_db_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir()
        .join(format!(
            "mizar_sqlite_test_{name}_{}_{nanos}",
            std::process::id()
        ))
        .join("sessions.sqlite")
}

struct TempDbGuard(PathBuf);

impl Drop for TempDbGuard {
    fn drop(&mut self) {
        if let Some(dir) = self.0.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

#[tokio::test]
async fn persists_and_hydrates_sessions_across_restarts() {
    let db_path = temp_db_path("restart");
    let _guard = TempDbGuard(db_path.clone());

    let backend1 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store1 = SessionStore::with_persistence(backend1.clone());
    let s1 = store1.create_with(SessionCreateOptions {
        id: Some("s-persist".into()),
        cwd: Some("/work/repo-a".into()),
        ..Default::default()
    });
    s1.append_trace(SessionTraceEvent::TurnStart { turn: 1 });
    s1.append(SessionEvent::user("帮我检查会话持久化\n第二行详细说明"));
    s1.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::Text {
            text: "没问题".into(),
        }],
        replay: Some(ReplayEnvelope {
            protocol: "openai-responses".into(),
            provider: "default".into(),
            model: "gpt-4o".into(),
            response: serde_json::json!({"responseId": "resp_123"}),
            blocks: Vec::new(),
        }),
        usage: Some(TokenUsage {
            input_tokens: 42,
            output_tokens: 15,
            total_tokens: Some(57),
            ..Default::default()
        }),
    });
    s1.append_trace(SessionTraceEvent::Anomaly {
        rule_id: "TEST_RULE".into(),
        severity: TraceSeverity::Warn,
        title: "测试告警".into(),
        summary: "记录测试告警".into(),
        turn: Some(1),
        step: Some(1),
        evidence: serde_json::json!({"k": "v"}),
    });
    s1.append_trace(SessionTraceEvent::TurnEnd {
        turn: 1,
        reason: "completed".into(),
    });
    s1.flush_async().await.unwrap();
    drop(store1);
    let _ = tokio::task::spawn_blocking(move || backend1.shutdown()).await;

    // 模拟全新进程重启，重新打开同一个 SQLite 文件
    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let headers = store2
        .list_headers(&SessionListQuery {
            cwd: Some("/work/repo-a".into()),
            limit: Some(10),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].id, "s-persist");
    assert_eq!(headers[0].title.as_deref(), Some("帮我检查会话持久化"));
    assert_eq!(headers[0].event_count, 5);
    assert_eq!(headers[0].surface_count, 2);

    // 未在内存中，get() 自动从 SQLite 懒加载完整 Session（含 ReplayEnvelope 与 Trace）
    let restored = store2.get("s-persist").expect("must hydrate from sqlite");
    assert_eq!(restored.log().len(), 2);
    assert_eq!(restored.all_entries().len(), 5);
    assert_eq!(restored.next_turn(), 2);

    let msgs = restored.derive_messages();
    assert_eq!(msgs.len(), 2);
    match &msgs[1] {
        Message::Assistant { replay, .. } => {
            assert_eq!(replay.as_ref().unwrap().protocol, "openai-responses");
        }
        other => panic!("unexpected message: {other:?}"),
    }

    let _ = store2.flush_async(None).await;
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
}

#[tokio::test]
async fn supports_fork_compaction_and_crash_recovery_in_sqlite() {
    let db_path = temp_db_path("fork_compact");
    let _guard = TempDbGuard(db_path.clone());

    let backend = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store = SessionStore::with_persistence(backend.clone());
    let parent = store.create_with(SessionCreateOptions {
        id: Some("parent".into()),
        cwd: Some("/repo/b".into()),
        ..Default::default()
    });
    parent.append(SessionEvent::user("Q1"));
    parent.append(SessionEvent::assistant_text("A1"));
    parent.append(SessionEvent::user("Q2"));
    parent.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c_hang".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        }],
        replay: None,
        usage: None,
    });
    parent.flush_async().await.unwrap();

    // 在 seq=4（未返 ToolResult 的 Assistant）处分叉，应持久化复制前缀并自愈悬挂 tool_call
    let child = store
        .fork(
            "parent",
            4,
            SessionCreateOptions {
                id: Some("child".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(child.header().parent_id.as_deref(), Some("parent"));
    assert_eq!(child.header().fork_seq, Some(4));
    assert_eq!(child.log().len(), 5);
    assert!(matches!(
        &child.log()[4].event,
        SessionEvent::ToolResult {
            tool_call_id,
            is_error: true,
            ..
        } if tool_call_id == "c_hang"
    ));

    // 子会话继续压缩并追加不会影响父会话
    child.compact("summary", 1, Some(1000));
    child.append(SessionEvent::user("Q3"));
    child.flush_async().await.unwrap();

    assert_eq!(parent.log().len(), 4);
    // Compaction 保留最近 1 条时自动回退保留完整的 (Assistant(ToolCall), ToolResult) 对，加上新 Q3 共 4 条
    assert_eq!(child.visible_log().len(), 4);

    let _ = store.flush_async(None).await;
    drop(store);
    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;

    // 重启验证 child 的 compaction 与 repaired tool_result 都落了盘
    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let child_reloaded = store2.get("child").unwrap();
    assert_eq!(child_reloaded.header().parent_id.as_deref(), Some("parent"));
    assert_eq!(child_reloaded.visible_log().len(), 4);
    assert!(child_reloaded.header().compacted_up_to_seq.is_some());
    let _ = store2.flush_async(None).await;
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
}

#[tokio::test]
async fn works_with_in_memory_sqlite_config() {
    let backend = SqliteSessionPersistence::open_async(":memory:")
        .await
        .unwrap();
    let store = SessionStore::with_persistence(backend.clone());
    let s = store.create_with(SessionCreateOptions {
        id: Some("mem-1".into()),
        cwd: Some("/mem".into()),
        ..Default::default()
    });
    s.append(SessionEvent::user("hello memory sqlite"));
    s.flush_async().await.unwrap();

    // 清空 SessionStore 内存表后再通过 get 从 :memory: SQLite 读回
    store.clear();
    let hydrated = store.get("mem-1").expect("should load from :memory: db");
    assert_eq!(hydrated.log().len(), 1);
    let _ = store.flush_async(None).await;
    drop(store);
    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
}

#[tokio::test]
async fn persists_and_reloads_bash_execution_context_and_local_only() {
    let db = temp_db_path("bash_exec");
    let _guard = TempDbGuard(db.clone());

    {
        let backend = SqliteSessionPersistence::open_async(&db).await.unwrap();
        let store = SessionStore::with_persistence(backend.clone());
        let s = store.create_with(SessionCreateOptions {
            id: Some("bash-db-1".into()),
            cwd: Some("/repo".into()),
            ..Default::default()
        });
        s.append(SessionEvent::BashExecution {
            command: "echo ctx_ok".into(),
            output: "ctx_ok\n".into(),
            exit_code: Some(0),
            cancelled: false,
            exclude_from_context: false,
        });
        s.append(SessionEvent::BashExecution {
            command: "echo secret_local".into(),
            output: "secret_local\n".into(),
            exit_code: Some(1),
            cancelled: false,
            exclude_from_context: true,
        });
        s.append(SessionEvent::user("请总结"));
        s.flush_async().await.unwrap();
        drop(store);
        let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
    }

    let backend2 = SqliteSessionPersistence::open_async(&db).await.unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let reloaded = store2
        .get("bash-db-1")
        .expect("should reload session with bash executions");

    assert_eq!(reloaded.header().event_count, 3);
    // 只有 ! 与 User 是 surface 事件，!! 不占 surface_count
    assert_eq!(reloaded.header().surface_count, 2);
    assert_eq!(reloaded.all_entries().len(), 3);
    assert_eq!(reloaded.log().len(), 2);

    assert!(matches!(
        &reloaded.all_entries()[1].event,
        SessionEvent::BashExecution {
            command,
            output,
            exit_code: Some(1),
            cancelled: false,
            exclude_from_context: true,
        } if command == "echo secret_local" && output == "secret_local\n"
    ));

    let msgs = reloaded.derive_messages();
    assert_eq!(msgs.len(), 2);
    assert!(matches!(
        &msgs[0],
        mizar_types::llm::Message::User { content, .. } if content.contains("ctx_ok")
    ));
    assert!(!format!("{msgs:?}").contains("secret_local"));

    let _ = store2.flush_async(None).await;
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
}

#[tokio::test]
async fn persists_and_derives_user_images_through_sqlite() {
    use mizar_types::llm::ImageAttachment;

    let db = temp_db_path("user_images");
    let _guard = TempDbGuard(db.clone());
    let img = ImageAttachment {
        media_type: "image/jpeg".into(),
        data: "/9j/4AAQSkZJRg==".into(),
        width: Some(800),
        height: Some(600),
    };

    {
        let backend = SqliteSessionPersistence::open_async(&db).await.unwrap();
        let store = SessionStore::with_persistence(backend.clone());
        let s = store.create_with(SessionCreateOptions {
            id: Some("img-db-1".into()),
            ..Default::default()
        });
        s.append(SessionEvent::user_with_images(
            "看图定位异常",
            vec![img.clone()],
        ));
        s.flush_async().await.unwrap();
        drop(store);
        let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
    }

    let backend2 = SqliteSessionPersistence::open_async(&db).await.unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let reloaded = store2
        .get("img-db-1")
        .expect("should reload image session from sqlite");
    let msgs = reloaded.derive_messages();
    assert_eq!(
        msgs,
        vec![Message::user_with_images("看图定位异常", vec![img])]
    );
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
}

#[tokio::test]
async fn migrates_v1_schema_to_v2_and_persists_title_source_clear_and_fork() {
    let db_path = temp_db_path("v1_to_v2");
    let _guard = TempDbGuard(db_path.clone());
    std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();

    // 1. 用旧版 v1 schema 手动建表（不含 title_source 列）并写入一条记录
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                 id TEXT PRIMARY KEY,
                 format_version INTEGER NOT NULL,
                 cwd TEXT,
                 title TEXT,
                 parent_id TEXT,
                 fork_seq INTEGER,
                 created_at_ms INTEGER NOT NULL,
                 updated_at_ms INTEGER NOT NULL,
                 event_count INTEGER NOT NULL DEFAULT 0,
                 surface_count INTEGER NOT NULL DEFAULT 0,
                 compacted_up_to_seq INTEGER
             );
             CREATE TABLE session_events (
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 ts_ms INTEGER NOT NULL,
                 visibility TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 payload_json TEXT NOT NULL,
                 PRIMARY KEY (session_id, seq)
             );
             PRAGMA user_version = 1;",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions (id, format_version, cwd, title, created_at_ms, updated_at_ms, event_count, surface_count)
             VALUES ('v1-sess', 1, '/work/legacy', 'legacy title', 1000, 1000, 1, 1);",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session_events (session_id, seq, ts_ms, visibility, kind, payload_json)
             VALUES ('v1-sess', 1, 1000, 'surface', 'user', '{\"type\":\"user\",\"text\":\"legacy\"}');",
            [],
        )
        .unwrap();
    }

    // 2. 启动迁移，验证 user_version=4 与 title_source 读写
    let backend1 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store1 = SessionStore::with_persistence(backend1.clone());

    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let ver: u32 = conn
            .query_row("PRAGMA user_version;", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, 4);
    }

    let migrated = store1.get("v1-sess").unwrap();
    assert_eq!(migrated.header().title.as_deref(), Some("legacy title"));
    assert_eq!(migrated.header().title_source, None);

    assert!(migrated.set_title("0320｜FIX｜人工修改标题"));
    assert!(migrated.clear_title());
    assert_eq!(migrated.header().title, None);
    assert_eq!(migrated.header().title_source, None);

    let forked = store1
        .fork(
            "v1-sess",
            1,
            SessionCreateOptions {
                id: Some("fork-child".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(forked.header().title, None);

    // 创建一个新会话，设置 Auto 标题并保留到重启后验证
    let s2 = store1.create_with(SessionCreateOptions {
        id: Some("s2-title".into()),
        ..Default::default()
    });
    assert!(s2.set_title_with_source("0320｜FEA｜自动生成标题", TitleSource::Auto));
    store1.flush_async(None).await.unwrap();
    drop(store1);
    let _ = tokio::task::spawn_blocking(move || backend1.shutdown()).await;

    // 3. 重启后确认 Clear 与 Auto title_source 均被正确持久化
    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let v1_reloaded = store2.get("v1-sess").unwrap();
    assert_eq!(v1_reloaded.header().title, None);
    assert_eq!(v1_reloaded.header().title_source, None);

    let s2_reloaded = store2.get("s2-title").unwrap();
    assert_eq!(
        s2_reloaded.header().title.as_deref(),
        Some("0320｜FEA｜自动生成标题")
    );
    assert_eq!(s2_reloaded.header().title_source, Some(TitleSource::Auto));

    let _ = store2.flush_async(None).await;
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
}

#[tokio::test]
async fn supports_out_of_order_create_fork_compaction_and_writer_lifecycle() {
    use mizar_session::sqlite::SqliteSessionPersistence;
    use mizar_session::{PersistenceError, SessionEntry, SessionHeader, SessionPersistence};

    let db_path = temp_db_path("edge_cases");
    let _guard = TempDbGuard(db_path.clone());
    let persistence = SqliteSessionPersistence::open(&db_path).unwrap();

    // 1. AppendEvents 先于 CreateSession 到达时，CreateSession 必须补全 cwd/title/parent 等元数据
    persistence
        .append_events(
            "out-of-order",
            &[SessionEntry {
                seq: 1,
                event: SessionEvent::user("先到的消息"),
                ts: 2000,
            }],
        )
        .unwrap();
    let mut full_header = SessionHeader::new("out-of-order", 1000);
    full_header.cwd = Some("/repo/out-of-order".into());
    full_header.title = Some("完整元数据标题".into());
    full_header.title_source = Some(TitleSource::Manual);
    full_header.parent_id = Some("root-anc".into());
    full_header.fork_seq = Some(9);
    persistence.create_session(&full_header).unwrap();
    let loaded_ooo = persistence.load_session("out-of-order").unwrap().unwrap();
    assert_eq!(loaded_ooo.header.cwd.as_deref(), Some("/repo/out-of-order"));
    assert_eq!(loaded_ooo.header.title.as_deref(), Some("完整元数据标题"));
    assert_eq!(loaded_ooo.header.title_source, Some(TitleSource::Manual));
    assert_eq!(loaded_ooo.header.parent_id.as_deref(), Some("root-anc"));
    assert_eq!(loaded_ooo.header.fork_seq, Some(9));

    // 2. 写入较大 ToolResult 与 Compaction 事件
    let mut deep_text = "x".repeat(5200);
    deep_text.push_str(" DEEP_NEEDLE_98765 ");
    persistence
        .append_events(
            "out-of-order",
            &[SessionEntry {
                seq: 2,
                event: SessionEvent::ToolResult {
                    tool_call_id: "tc-deep".into(),
                    name: "read".into(),
                    content: deep_text,
                    is_error: false,
                },
                ts: 2500,
            }],
        )
        .unwrap();
    persistence
        .append_events(
            "out-of-order",
            &[SessionEntry {
                seq: 3,
                event: SessionEvent::Compaction {
                    summary: "compact at 3".into(),
                    first_kept_seq: 2,
                    tokens_before: Some(200),
                    tokens_after: Some(50),
                    digest: Default::default(),
                },
                ts: 2600,
            }],
        )
        .unwrap();
    persistence.flush(None).unwrap();

    // 3. fork 前缀若包含 Compaction，子会话 compacted_up_to_seq 必须被推导保存
    let forked = persistence
        .fork_session("out-of-order", 3, &SessionHeader::new("ooo-fork", 3000))
        .unwrap();
    assert_eq!(forked.header.compacted_up_to_seq, Some(2));
    assert_eq!(forked.max_persisted_seq, Some(3));

    // 4. shutdown 后再写必须明确返回 PersistenceError::Closed
    persistence.shutdown().unwrap();
    let err = persistence.flush(None).unwrap_err();
    assert!(matches!(err, PersistenceError::Closed), "got {err:?}");
}

#[test]
fn concurrent_cold_open_and_writes_do_not_fail_with_database_locked() {
    let dir = std::env::temp_dir().join(format!(
        "mizar_sqlite_concurrent_open_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    for round in 0..4 {
        let db_path = dir.join(format!("race_{round}.sqlite"));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(6));
        let mut handles = Vec::new();
        for idx in 0..6 {
            let path = db_path.clone();
            let b = std::sync::Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || -> Result<(), String> {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                b.wait();
                rt.block_on(async move {
                    let backend = SqliteSessionPersistence::open_async(&path)
                        .await
                        .map_err(|e| format!("sqlite boot failed: {e}"))?;
                    let store = SessionStore::with_persistence(backend.clone());
                    let sid = format!("s_{idx}");
                    let s = store.create_with(SessionCreateOptions {
                        id: Some(sid),
                        cwd: Some("/tmp/ws".into()),
                        ..Default::default()
                    });
                    s.append(SessionEvent::user(format!("hello_{idx}")));
                    s.flush_async()
                        .await
                        .map_err(|e| format!("flush failed: {e}"))?;
                    drop(store);
                    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
                    Ok(())
                })
            }));
        }
        for h in handles {
            h.join()
                .expect("worker thread panicked")
                .expect("concurrent cold open/write failed");
        }
        let headers = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let backend = SqliteSessionPersistence::open_async(&db_path)
                    .await
                    .unwrap();
                let headers = backend
                    .list_sessions(&SessionListQuery {
                        cwd: None,
                        limit: Some(20),
                        ..Default::default()
                    })
                    .unwrap();
                let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
                headers
            })
        })
        .join()
        .unwrap();
        assert_eq!(headers.len(), 6, "round={round} headers={headers:?}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn usage_event_persists_and_reloads_without_schema_migration_and_excludes_from_messages() {
    use mizar_session::sqlite::schema::CURRENT_SCHEMA_VERSION;
    use mizar_types::llm::{Message, TokenUsage};

    let dir = std::env::temp_dir().join(format!("mizar_sqlite_usage_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("sessions.sqlite");

    let usage = TokenUsage {
        input_tokens: 42,
        output_tokens: 1,
        cache_read_tokens: Some(8_192),
        cache_write_tokens: None,
        total_tokens: Some(8_235),
        reasoning_tokens: None,
    };

    let backend1 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store1 = SessionStore::with_persistence(backend1.clone());
    let s1 = store1.create(Some("usage-persist".into()));
    s1.append(SessionEvent::user("hello"));
    s1.append(SessionEvent::usage(
        "cache_warm",
        "anthropic",
        "claude-sonnet-4-6",
        usage,
        Some("keepalive".into()),
    ));
    s1.flush_async().await.unwrap();
    drop(store1);
    let _ = tokio::task::spawn_blocking(move || backend1.shutdown()).await;

    // 校验 SQLite 表结构仍为现有 schema 版本 2 且 kind/visibility/payload_json 正确落盘
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let ver: u32 = conn
            .query_row("PRAGMA user_version;", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, CURRENT_SCHEMA_VERSION);
        let (vis, kind): (String, String) = conn
            .query_row(
                "SELECT visibility, kind FROM session_events WHERE session_id = ?1 AND seq = 2;",
                ["usage-persist"],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(vis, "log_only");
        assert_eq!(kind, "usage");
    }

    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let reloaded = store2.get("usage-persist").unwrap();

    assert_eq!(reloaded.header().event_count, 2);
    assert_eq!(reloaded.header().surface_count, 1);
    assert_eq!(reloaded.log().len(), 1);
    assert_eq!(reloaded.all_entries().len(), 2);
    assert!(matches!(
        &reloaded.all_entries()[1].event,
        SessionEvent::Usage {
            kind,
            provider,
            model,
            usage: u,
            note,
        } if kind == "cache_warm"
            && provider == "anthropic"
            && model == "claude-sonnet-4-6"
            && *u == usage
            && note.as_deref() == Some("keepalive")
    ));
    assert_eq!(reloaded.derive_messages(), vec![Message::user("hello")]);

    let _ = store2.flush_async(None).await;
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn list_sessions_exclude_subagents_skips_newer_child_for_continue() {
    let db_path = temp_db_path("exclude_subagents");
    let _guard = TempDbGuard(db_path.clone());

    let backend = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store = SessionStore::with_persistence(backend.clone());
    let root = store.create_with(SessionCreateOptions {
        id: Some("parent-sess".into()),
        cwd: Some("/work/repo-sub".into()),
        ..Default::default()
    });
    root.append(SessionEvent::user("start parent"));
    let forked = store
        .fork(
            "parent-sess",
            1,
            SessionCreateOptions {
                id: Some("fork-sess".into()),
                ..Default::default()
            },
        )
        .unwrap();
    forked.append(SessionEvent::user("branch"));

    std::thread::sleep(std::time::Duration::from_millis(5));
    let child = store.create_with(SessionCreateOptions {
        id: Some("subagent-child".into()),
        cwd: Some("/work/repo-sub".into()),
        parent_id: Some("parent-sess".into()),
        ..Default::default()
    });
    child.append(SessionEvent::user("subagent step"));

    store.flush(None).unwrap();

    let all = store
        .list_headers(&SessionListQuery {
            cwd: Some("/work/repo-sub".into()),
            limit: Some(1),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(all[0].id, "subagent-child");

    let filtered = store
        .list_headers(&SessionListQuery {
            cwd: Some("/work/repo-sub".into()),
            limit: Some(1),
            exclude_subagents: true,
        })
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_ne!(filtered[0].id, "subagent-child");
    assert_eq!(filtered[0].id, "fork-sess");

    let _ = store.flush_async(None).await;
    drop(store);
    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
}

#[tokio::test]
async fn migrates_v2_sessions_backfilling_origin_and_filters_explicit_subagent_forks() {
    let db_path = temp_db_path("migrate_v2_to_v3_origin");
    let _guard = TempDbGuard(db_path.clone());

    // 1. 构造 v2 schema 数据库（不含 origin 列），含根会话、Fork 会话、历史子代理会话
    {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                 id TEXT PRIMARY KEY,
                 format_version INTEGER NOT NULL,
                 cwd TEXT,
                 title TEXT,
                 title_source TEXT,
                 parent_id TEXT,
                 fork_seq INTEGER,
                 created_at_ms INTEGER NOT NULL,
                 updated_at_ms INTEGER NOT NULL,
                 event_count INTEGER NOT NULL DEFAULT 0,
                 surface_count INTEGER NOT NULL DEFAULT 0,
                 compacted_up_to_seq INTEGER
             );
             CREATE TABLE session_events (
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 ts_ms INTEGER NOT NULL,
                 visibility TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 payload_json TEXT NOT NULL,
                 PRIMARY KEY (session_id, seq)
             );
             CREATE TABLE session_recall_docs (
                 doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 ts_ms INTEGER NOT NULL,
                 kind TEXT NOT NULL,
                 preview TEXT NOT NULL,
                 UNIQUE(session_id, seq)
             );
             CREATE VIRTUAL TABLE session_recall_fts USING fts5(terms, content='', tokenize='ascii');
             INSERT INTO sessions (id, format_version, cwd, parent_id, fork_seq, created_at_ms, updated_at_ms)
             VALUES
                 ('v2-root', 1, '/work/v2', NULL, NULL, 1000, 1000),
                 ('v2-fork', 1, '/work/v2', 'v2-root', 1, 1001, 1001),
                 ('v2-child', 1, '/work/v2', 'v2-root', NULL, 1002, 1002);
             PRAGMA user_version = 2;",
        )
        .unwrap();
    }

    // 2. 启动迁移到 v4 并核验回填结果与旧 recall 表清理
    let backend = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store = SessionStore::with_persistence(backend.clone());

    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let ver: u32 = conn
            .query_row("PRAGMA user_version;", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, 4);
        let leftover_recall_tables: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table' AND name IN ('session_recall_docs', 'session_recall_fts');",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leftover_recall_tables, 0);
    }

    assert_eq!(
        store.get("v2-root").unwrap().header().origin,
        Some(mizar_session::SessionOrigin::User)
    );
    assert_eq!(
        store.get("v2-fork").unwrap().header().origin,
        Some(mizar_session::SessionOrigin::User)
    );
    assert_eq!(
        store.get("v2-child").unwrap().header().origin,
        Some(mizar_session::SessionOrigin::Subagent)
    );

    // 3. 新建带 fork_seq 但显式标记 origin=Subagent 的会话，exclude_subagents 必须将其排除
    let explicit_forked_sub = store.create_with(SessionCreateOptions {
        id: Some("explicit-sub-fork".into()),
        cwd: Some("/work/v2".into()),
        parent_id: Some("v2-root".into()),
        fork_seq: Some(1),
        origin: Some(mizar_session::SessionOrigin::Subagent),
        ..Default::default()
    });
    assert!(explicit_forked_sub.header().is_subagent_child());
    store.flush(None).unwrap();

    let filtered = store
        .list_headers(&SessionListQuery {
            cwd: Some("/work/v2".into()),
            limit: None,
            exclude_subagents: true,
        })
        .unwrap();
    let ids: Vec<&str> = filtered.iter().map(|h| h.id.as_str()).collect();
    assert!(ids.contains(&"v2-root") && ids.contains(&"v2-fork"));
    assert!(!ids.contains(&"v2-child"));
    assert!(!ids.contains(&"explicit-sub-fork"));

    let _ = store.flush_async(None).await;
    drop(store);
    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
}

#[tokio::test]
async fn drops_legacy_recall_tables_when_migrating_v3_database() {
    let db_path = temp_db_path("migrate_v3_to_v4_drop_recall");
    let _guard = TempDbGuard(db_path.clone());

    {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (
                 id TEXT PRIMARY KEY,
                 format_version INTEGER NOT NULL,
                 cwd TEXT,
                 title TEXT,
                 title_source TEXT,
                 parent_id TEXT,
                 fork_seq INTEGER,
                 origin TEXT,
                 created_at_ms INTEGER NOT NULL,
                 updated_at_ms INTEGER NOT NULL,
                 event_count INTEGER NOT NULL DEFAULT 0,
                 surface_count INTEGER NOT NULL DEFAULT 0,
                 compacted_up_to_seq INTEGER
             );
             CREATE TABLE session_events (
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 ts_ms INTEGER NOT NULL,
                 visibility TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 payload_json TEXT NOT NULL,
                 PRIMARY KEY (session_id, seq)
             );
             CREATE TABLE session_recall_docs (
                 doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 ts_ms INTEGER NOT NULL,
                 kind TEXT NOT NULL,
                 preview TEXT NOT NULL,
                 UNIQUE(session_id, seq)
             );
             CREATE INDEX idx_recall_docs_session_seq ON session_recall_docs(session_id, seq);
             CREATE VIRTUAL TABLE session_recall_fts USING fts5(terms, content='', tokenize='ascii');
             INSERT INTO sessions (id, format_version, cwd, origin, created_at_ms, updated_at_ms, event_count, surface_count)
             VALUES ('v3-keep', 1, '/work/v3', 'user', 1000, 1000, 1, 1);
             INSERT INTO session_events (session_id, seq, ts_ms, visibility, kind, payload_json)
             VALUES ('v3-keep', 1, 1000, 'surface', 'user', '{\"type\":\"user\",\"text\":\"preserved across v4\"}');
             INSERT INTO session_recall_docs (session_id, seq, ts_ms, kind, preview)
             VALUES ('v3-keep', 1, 1000, 'user', 'preserved across v4');
             INSERT INTO session_recall_fts (rowid, terms) VALUES (1, 'preserved');
             PRAGMA user_version = 3;",
        )
        .unwrap();
    }

    let backend = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store = SessionStore::with_persistence(backend.clone());

    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let ver: u32 = conn
            .query_row("PRAGMA user_version;", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, 4);
        let leftover: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE name IN ('session_recall_docs', 'session_recall_fts', 'idx_recall_docs_session_seq');",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(leftover, 0);
    }

    let kept = store.get("v3-keep").unwrap();
    assert_eq!(
        kept.derive_messages(),
        vec![Message::user("preserved across v4")]
    );

    drop(store);
    let _ = tokio::task::spawn_blocking(move || backend.shutdown()).await;
}

#[tokio::test]
async fn async_reads_and_transient_busy_do_not_poison_writer() {
    use mizar_session::sqlite::SqliteSessionPersistence;
    use mizar_session::{SessionEntry, SessionHeader, SessionPersistence, SessionSelection};
    use std::time::Duration;

    let db_path = temp_db_path("async_busy_fork");
    let _guard = TempDbGuard(db_path.clone());

    let persistence =
        SqliteSessionPersistence::open_with_busy_timeout(&db_path, Duration::from_millis(50))
            .unwrap();

    // 1. 写入父会话消息、Selection 与多条检索条目
    let mut parent_h = SessionHeader::new("parent-async", 1000);
    parent_h.cwd = Some("/ws/async".into());
    parent_h.title = Some("Async Parent".into());
    persistence.create_session(&parent_h).unwrap();
    persistence
        .append_events(
            "parent-async",
            &[
                SessionEntry {
                    seq: 1,
                    event: SessionEvent::user("父会话早期关键标记 PARENT_TOKEN_111"),
                    ts: 1001,
                },
                SessionEntry {
                    seq: 2,
                    event: SessionEvent::Selection {
                        provider: Some("openai".into()),
                        model: Some("gpt-5".into()),
                        reasoning_effort: Some(mizar_types::llm::ReasoningEffort::High),
                    },
                    ts: 1002,
                },
                SessionEntry {
                    seq: 3,
                    event: SessionEvent::user("分叉切点之后的父会话独占内容 PARENT_AFTER_FORK_999"),
                    ts: 1003,
                },
            ],
        )
        .unwrap();
    persistence.flush_async(None).await.unwrap();

    // 2. 轻量异步读取 load_header_async / latest_selection_async
    let hdr = persistence
        .load_header_async("parent-async".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(hdr.title.as_deref(), Some("Async Parent"));
    let sel = persistence
        .latest_selection_async("parent-async".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        sel,
        SessionSelection {
            provider: Some("openai".into()),
            model: Some("gpt-5".into()),
            reasoning_effort: Some(mizar_types::llm::ReasoningEffort::High),
        }
    );

    // 3. fork_session 仅复制 <= fork_seq 的前缀且不泄漏 > fork_seq
    let mut child_h = SessionHeader::new("child-async", 2000);
    child_h.cwd = Some("/ws/async".into());
    let _forked = persistence
        .fork_session("parent-async", 2, &child_h)
        .unwrap();
    persistence
        .append_events(
            "child-async",
            &[SessionEntry {
                seq: 3,
                event: SessionEvent::user("子会话新增词条 CHILD_POST_FORK_222"),
                ts: 2001,
            }],
        )
        .unwrap();
    persistence.flush_async(None).await.unwrap();

    let child_loaded = persistence
        .load_session_async("child-async".into())
        .await
        .unwrap()
        .unwrap();
    let child_seqs: Vec<u64> = child_loaded.entries.iter().map(|e| e.seq).collect();
    assert_eq!(child_seqs, vec![1, 2, 3]);
    assert!(matches!(
        &child_loaded.entries[0].event,
        SessionEvent::User { text, .. } if text.contains("PARENT_TOKEN_111")
    ));
    assert!(matches!(
        &child_loaded.entries[2].event,
        SessionEvent::User { text, .. } if text.contains("CHILD_POST_FORK_222")
    ));

    // 4. 瞬时跨连接写锁冲突 (SQLITE_BUSY) 不得永久毒化 sticky_error / last_error
    let blocker = rusqlite::Connection::open(&db_path).unwrap();
    blocker.execute_batch("BEGIN EXCLUSIVE;").unwrap();

    persistence
        .append_events(
            "child-async",
            &[SessionEntry {
                seq: 4,
                event: SessionEvent::user("锁释放前尝试写入"),
                ts: 2002,
            }],
        )
        .unwrap();
    let busy_err = persistence.flush_async(None).await;
    assert!(busy_err.is_err(), "expected transient SQLITE_BUSY error");

    blocker.execute_batch("ROLLBACK;").unwrap();
    drop(blocker);

    persistence
        .append_events(
            "child-async",
            &[SessionEntry {
                seq: 4,
                event: SessionEvent::user("锁释放后恢复写入 RECOVERED_AFTER_BUSY"),
                ts: 2003,
            }],
        )
        .unwrap();
    persistence.flush_async(None).await.unwrap();

    let recovered = persistence
        .load_session_async("child-async".into())
        .await
        .unwrap()
        .unwrap();
    assert!(recovered.entries.iter().any(|e| {
        e.seq == 4
            && matches!(
                &e.event,
                SessionEvent::User { text, .. } if text.contains("RECOVERED_AFTER_BUSY")
            )
    }));

    persistence.shutdown().unwrap();

    // 6. 验证 open_async 可正常从异步上下文初始化并读取
    let reopened = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let loaded = reopened
        .load_session_async("child-async".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.header.id, "child-async");
    reopened.shutdown().unwrap();
}

#[tokio::test]
async fn transient_busy_in_prior_batch_surfaces_to_subsequent_flush_without_poisoning() {
    use mizar_session::{SessionEntry, SessionHeader};
    use tokio::time::{Duration, sleep};

    let db_path = temp_db_path("prior_batch_busy");
    let _guard = TempDbGuard(db_path.clone());

    let persistence =
        SqliteSessionPersistence::open_with_busy_timeout(&db_path, Duration::from_millis(30))
            .unwrap();

    let mut header = SessionHeader::new("prior-busy", 1000);
    header.cwd = Some("/ws/prior-busy".into());
    persistence.create_session(&header).unwrap();
    persistence.flush_async(None).await.unwrap();

    let blocker = rusqlite::Connection::open(&db_path).unwrap();
    blocker.execute_batch("BEGIN EXCLUSIVE;").unwrap();

    // 先写入但不立即 flush，等待超过 busy_timeout 确保前一批次无 flush 先失败并暂存瞬时错误
    persistence
        .append_events(
            "prior-busy",
            &[SessionEntry {
                seq: 1,
                event: SessionEvent::user("锁持有期间入队写入"),
                ts: 1001,
            }],
        )
        .unwrap();
    sleep(Duration::from_millis(100)).await;

    let busy_err = persistence.flush_async(None).await;
    assert!(
        busy_err.is_err(),
        "expected prior batch transient SQLITE_BUSY error to surface on subsequent flush"
    );

    blocker.execute_batch("ROLLBACK;").unwrap();
    drop(blocker);

    persistence
        .append_events(
            "prior-busy",
            &[SessionEntry {
                seq: 1,
                event: SessionEvent::user("锁释放后恢复写入 PRIOR_BUSY_RECOVERED"),
                ts: 1002,
            }],
        )
        .unwrap();
    persistence.flush_async(None).await.unwrap();

    let recovered = persistence
        .load_session_async("prior-busy".into())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recovered.entries.len(), 1);
    assert_eq!(recovered.entries[0].seq, 1);
    assert!(matches!(
        &recovered.entries[0].event,
        SessionEvent::User { text, .. } if text.contains("PRIOR_BUSY_RECOVERED")
    ));

    persistence.shutdown().unwrap();
}
