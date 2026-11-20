use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, Service};
use mizar_llm::{ContentBlock, Message, ReplayEnvelope, TokenUsage};
use mizar_session::{
    SessionContextExt, SessionCreateOptions, SessionEvent, SessionListQuery, SessionStore,
    SessionTraceEvent, TraceSeverity,
};
use mizar_session_sqlite::{SqlitePersistenceConfig, SqlitePersistencePlugin};

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

    let app1 = Context::new();
    app1.plugin(
        SqlitePersistencePlugin,
        SqlitePersistenceConfig::new(&db_path),
    );
    app1.plugin(SessionStore::plugin(), ());
    app1.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store1 = app1.sessions().unwrap();
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
    app1.fiber().dispose().await;

    // 模拟全新进程重启，重新打开同一个 SQLite 文件
    let app2 = Context::new();
    app2.plugin(
        SqlitePersistencePlugin,
        SqlitePersistenceConfig::new(&db_path),
    );
    app2.plugin(SessionStore::plugin(), ());
    app2.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store2 = app2.sessions().unwrap();
    let headers = store2
        .list_headers(&SessionListQuery {
            cwd: Some("/work/repo-a".into()),
            limit: Some(10),
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

    app2.fiber().dispose().await;
}

#[tokio::test]
async fn supports_fork_compaction_and_crash_recovery_in_sqlite() {
    let db_path = temp_db_path("fork_compact");
    let _guard = TempDbGuard(db_path.clone());

    let app = Context::new();
    app.plugin(
        SqlitePersistencePlugin,
        SqlitePersistenceConfig::new(&db_path),
    );
    app.plugin(SessionStore::plugin(), ());
    app.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store = app.sessions().unwrap();
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

    app.fiber().dispose().await;

    // 重启验证 child 的 compaction 与 repaired tool_result 都落了盘
    let app2 = Context::new();
    app2.plugin(
        SqlitePersistencePlugin,
        SqlitePersistenceConfig::new(&db_path),
    );
    app2.plugin(SessionStore::plugin(), ());
    app2.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store2 = app2.sessions().unwrap();
    let child_reloaded = store2.get("child").unwrap();
    assert_eq!(child_reloaded.header().parent_id.as_deref(), Some("parent"));
    assert_eq!(child_reloaded.visible_log().len(), 4);
    assert!(child_reloaded.header().compacted_up_to_seq.is_some());
    app2.fiber().dispose().await;
}

#[tokio::test]
async fn works_with_in_memory_sqlite_config() {
    let app = Context::new();
    app.plugin(
        SqlitePersistencePlugin,
        SqlitePersistenceConfig::new(":memory:"),
    );
    app.plugin(SessionStore::plugin(), ());
    app.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store = app.sessions().unwrap();
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
    app.fiber().dispose().await;
}

#[tokio::test]
async fn persists_and_reloads_bash_execution_context_and_local_only() {
    let db = temp_db_path("bash_exec");
    let _guard = TempDbGuard(db.clone());

    {
        let app = Context::new();
        app.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&db));
        app.plugin(SessionStore::plugin(), ());
        app.plugin(
            cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
            (),
        )
        .await
        .unwrap();

        let store = app.sessions().unwrap();
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
        app.fiber().dispose().await;
    }

    let app2 = Context::new();
    app2.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&db));
    app2.plugin(SessionStore::plugin(), ());
    app2.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store2 = app2.sessions().unwrap();
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
        mizar_llm::Message::User { content, .. } if content.contains("ctx_ok")
    ));
    assert!(!format!("{msgs:?}").contains("secret_local"));

    app2.fiber().dispose().await;
}

#[tokio::test]
async fn fts5_ngram_recall_searches_cjk_code_across_compaction_and_fork() {
    use mizar_session::RecallSearchQuery;

    let db = temp_db_path("recall_fts5");
    let _guard = TempDbGuard(db.clone());

    let app = Context::new();
    app.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&db));
    app.plugin(SessionStore::plugin(), ());
    app.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store = app.sessions().unwrap();
    let s1 = store.create_with(SessionCreateOptions {
        id: Some("s1".into()),
        cwd: Some("/ws/alpha".into()),
        ..Default::default()
    });
    s1.append(SessionEvent::user(
        "早期决策：使用 compacted_up_to_seq 追踪历史压缩水位",
    ));
    s1.append(SessionEvent::user(
        "我们来排查 Ryugi 画板与任务看板的双向状态同步问题",
    ));
    s1.append(SessionEvent::ToolResult {
        tool_call_id: "tc-1".into(),
        name: "read".into(),
        content: "fn resolve_anchor_node() { /* unique_anchor_marker_99 */ }".into(),
        is_error: false,
    });
    // 即使发生压缩，压缩前的原始事件仍必须可穿透检索
    s1.compact("已确认看板边界", 0, Some(1000));
    s1.flush_async().await.unwrap();

    // 分叉子会话也应自动具备前缀事件的 FTS5 索引
    let forked = store
        .fork("s1", 2, SessionCreateOptions::default())
        .unwrap();
    forked.flush_async().await.unwrap();

    // Rust 标识符检索
    let hits_ident = store
        .search_recall(&RecallSearchQuery {
            query: "compacted_up_to_seq".into(),
            session_ids: vec![forked.id.clone()],
            cwd: None,
            limit: 10,
        })
        .unwrap()
        .expect("sqlite backend should return Some(hits)");
    assert!(
        hits_ident
            .iter()
            .any(|h| h.session_id == forked.id && h.seq == 1),
        "expected forked session FTS hit, got: {hits_ident:?}"
    );

    // 中文 2 字短词检索（穿透 Compaction 边界）
    let hits_cjk = store
        .search_recall(&RecallSearchQuery {
            query: "画板".into(),
            session_ids: vec!["s1".into()],
            cwd: None,
            limit: 10,
        })
        .unwrap()
        .unwrap();
    assert!(
        hits_cjk.iter().any(|h| h.session_id == "s1" && h.seq == 2),
        "expected Chinese 2-char hit across compaction, got: {hits_cjk:?}"
    );

    // ToolResult 原文子串检索（穿透 Compaction 边界）
    let hits_tool = store
        .search_recall(&RecallSearchQuery {
            query: "unique_anchor_marker_99".into(),
            session_ids: vec!["s1".into()],
            cwd: None,
            limit: 10,
        })
        .unwrap()
        .unwrap();
    assert!(
        hits_tool.iter().any(|h| h.session_id == "s1" && h.seq == 3),
        "expected tool result hit across compaction, got: {hits_tool:?}"
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn persists_and_derives_user_images_through_sqlite() {
    use mizar_llm::ImageAttachment;

    let db = temp_db_path("user_images");
    let _guard = TempDbGuard(db.clone());
    let img = ImageAttachment {
        media_type: "image/jpeg".into(),
        data: "/9j/4AAQSkZJRg==".into(),
        width: Some(800),
        height: Some(600),
    };

    {
        let app = Context::new();
        app.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&db));
        app.plugin(SessionStore::plugin(), ());
        app.plugin(
            cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
            (),
        )
        .await
        .unwrap();

        let store = app.sessions().unwrap();
        let s = store.create_with(SessionCreateOptions {
            id: Some("img-db-1".into()),
            ..Default::default()
        });
        s.append(SessionEvent::user_with_images(
            "看图定位异常",
            vec![img.clone()],
        ));
        s.flush_async().await.unwrap();
        app.fiber().dispose().await;
    }

    let app2 = Context::new();
    app2.plugin(SqlitePersistencePlugin, SqlitePersistenceConfig::new(&db));
    app2.plugin(SessionStore::plugin(), ());
    app2.plugin(
        cordis::PluginDef::sync("noop", |_, ()| Ok::<_, cordis::CordisError>(())),
        (),
    )
    .await
    .unwrap();

    let store2 = app2.sessions().unwrap();
    let reloaded = store2
        .get("img-db-1")
        .expect("should reload image session from sqlite");
    let msgs = reloaded.derive_messages();
    assert_eq!(
        msgs,
        vec![Message::user_with_images("看图定位异常", vec![img])]
    );
    app2.fiber().dispose().await;
}
