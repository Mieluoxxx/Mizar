use std::sync::Arc;

use mizar_session::{
    SessionCreateOptions, SessionEvent, SessionStore, SessionTraceEvent, SqliteSessionPersistence,
};
use mizar_types::llm::{ContentBlock, Message};

#[tokio::test]
async fn sqlite_resume_and_fork_match_live_prefix_with_tool_result_prune() {
    let dir = std::env::temp_dir().join(format!("mizar_prune_sqlite_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("sessions.sqlite");

    let backend1 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store1 = SessionStore::with_persistence(backend1.clone());

    let live = store1.create_with(SessionCreateOptions {
        id: Some("prune-sess".into()),
        cwd: Some("/repo".into()),
        ..Default::default()
    });
    live.append(SessionEvent::user("inspect foo.rs"));
    live.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: r#"{"path":"foo.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    live.append(SessionEvent::ToolResult {
        tool_call_id: "c1".into(),
        name: "read".into(),
        content: "STALE_READ_BODY".into(),
        is_error: false,
    });
    live.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c2".into(),
            name: "grep".into(),
            arguments: r#"{"pattern":"missing"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    live.append(SessionEvent::ToolResult {
        tool_call_id: "c2".into(),
        name: "grep".into(),
        content: "No matches found".into(),
        is_error: false,
    });
    live.append(SessionEvent::Assistant {
        blocks: vec![ContentBlock::ToolCall {
            id: "c3".into(),
            name: "read".into(),
            arguments: r#"{"path":"foo.rs"}"#.into(),
        }],
        replay: None,
        usage: None,
    });
    live.append(SessionEvent::ToolResult {
        tool_call_id: "c3".into(),
        name: "read".into(),
        content: "FRESH_READ_BODY".into(),
        is_error: false,
    });
    live.append_trace(SessionTraceEvent::ToolResultPrune {
        target_seq: 3,
        notice: "[Superseded by a newer read of this file]".into(),
    });
    live.append_trace(SessionTraceEvent::ToolResultPrune {
        target_seq: 5,
        notice: "[Uneventful result elided]".into(),
    });
    live.flush_async().await.unwrap();

    let live_messages = live.derive_messages();
    assert_eq!(
        live_messages[2],
        Message::Tool {
            content: "[Superseded by a newer read of this file]".into(),
            tool_call_id: "c1".into(),
            is_error: false,
        }
    );
    assert_eq!(
        live_messages[4],
        Message::Tool {
            content: "[Uneventful result elided]".into(),
            tool_call_id: "c2".into(),
            is_error: false,
        }
    );

    let forked = store1
        .fork(&live.id, live.last_seq(), SessionCreateOptions::default())
        .unwrap();
    assert_eq!(forked.derive_messages(), live_messages);

    drop(forked);
    drop(live);
    drop(store1);
    let _ = tokio::task::spawn_blocking(move || backend1.shutdown()).await;

    // 模拟 --resume 重新打开 SQLite 库并水合会话，验证与实时前缀逐字节一致
    let backend2 = SqliteSessionPersistence::open_async(&db_path)
        .await
        .unwrap();
    let store2 = SessionStore::with_persistence(backend2.clone());
    let resumed: Arc<mizar_session::Session> = store2.get("prune-sess").expect("session resumed");
    assert_eq!(resumed.derive_messages(), live_messages);

    let resumed_fork = store2
        .fork(
            "prune-sess",
            resumed.last_seq(),
            SessionCreateOptions::default(),
        )
        .unwrap();
    assert_eq!(resumed_fork.derive_messages(), live_messages);

    drop(resumed_fork);
    drop(resumed);
    drop(store2);
    let _ = tokio::task::spawn_blocking(move || backend2.shutdown()).await;
    let _ = std::fs::remove_dir_all(&dir);
}
