use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::mpsc;
use mizar_core::spine::{AgentSpineConfig, MizarRuntime};
use mizar_core::trajectory::TimelineMode;
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, StreamChunk, TokenUsage, ToolCall,
};
use mizar_session::SessionListQuery;
use mizar_tools::{ToolAnnotations, ToolDefinition};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "mizar_spine_sqlite_{name}_{}_{nanos}",
        std::process::id()
    ))
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct TwoTurnAdapter {
    calls: AtomicUsize,
    seen_history_lens: Arc<Mutex<Vec<usize>>>,
}

impl LlmAdapter for TwoTurnAdapter {
    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        self.seen_history_lens
            .lock()
            .unwrap()
            .push(options.messages.len());
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            match call {
                0 => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "tc_1".into(),
                            name: "get_time".into(),
                            arguments: "{}".into(),
                        },
                    )));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                        usage: TokenUsage {
                            input_tokens: 15,
                            output_tokens: 5,
                            ..Default::default()
                        },
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }));
                }
                1 => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("第一轮已完成")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
                _ => {
                    let _ = tx.unbounded_send(Ok(StreamChunk::text("重启后恢复续聊成功")));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            }
        });
        Box::pin(rx)
    }
}

async fn mount_spine_with_db(
    db_path: &PathBuf,
    ws: &PathBuf,
    adapter: Arc<TwoTurnAdapter>,
) -> Arc<MizarRuntime> {
    let runtime = MizarRuntime::build(
        AgentSpineConfig::new(ws)
            .with_session_db(db_path)
            .with_disabled_plugins(["session-title"]),
    )
    .await
    .unwrap();
    let def = ToolDefinition::sync_tool(
        "get_time",
        "Get current time",
        json!({"type": "object", "properties": {}}),
        |_, _| Ok::<_, mizar_tools::ToolError>("2026-01-01 12:00:00".to_string()),
    )
    .with_annotations(ToolAnnotations {
        read_only: true,
        ..Default::default()
    });
    runtime.push_guard(runtime.tools.register(def).unwrap());
    runtime.register_adapter("test", adapter).unwrap();
    runtime
}

#[tokio::test]
async fn agent_loop_persists_surface_and_trajectory_across_process_restart() {
    let tmp = temp_dir("e2e");
    let _guard = DirGuard(tmp.clone());
    let ws = tmp.join("repo");
    std::fs::create_dir_all(&ws).unwrap();
    let db_path = tmp.join("sessions.sqlite");

    // 1. 第一次运行：执行一轮带工具调用的 AgentLoop
    let seen1 = Arc::new(Mutex::new(Vec::new()));
    let adapter1 = Arc::new(TwoTurnAdapter {
        calls: AtomicUsize::new(0),
        seen_history_lens: Arc::clone(&seen1),
    });
    let app1 = mount_spine_with_db(&db_path, &ws, adapter1).await;
    let s1 = app1.sessions.create(Some("e2e-session".into()));

    let reply1 = app1
        .agent_loop
        .run(
            Arc::clone(&s1),
            "帮我看看现在几点",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply1, "第一轮已完成");
    assert_eq!(
        s1.header().cwd.as_deref(),
        Some(ws.to_string_lossy().as_ref())
    );
    assert_eq!(s1.header().title.as_deref(), Some("帮我看看现在几点"));
    app1.shutdown().await;

    // 2. 模拟进程完全重启，挂载新 Runtime + SQLite，恢复同一会话并继续第二轮对话
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let adapter2 = Arc::new(TwoTurnAdapter {
        calls: AtomicUsize::new(2),
        seen_history_lens: Arc::clone(&seen2),
    });
    let app2 = mount_spine_with_db(&db_path, &ws, adapter2).await;

    let sessions2 = &app2.sessions;
    let headers = sessions2
        .list_headers(&SessionListQuery {
            cwd: Some(ws.to_string_lossy().to_string()),
            limit: Some(10),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].id, "e2e-session");

    // 重启后仅靠 TrajectoryRuntime 也可直接恢复并检查第一轮完整轨迹（含 Turn/Step/ToolCall）
    let traj2 = app2.trajectory.as_ref().unwrap();
    let list_summaries = traj2.list_sessions();
    assert_eq!(list_summaries.len(), 1);
    assert_eq!(list_summaries[0].turns, 1);
    assert_eq!(list_summaries[0].steps, 2);
    assert_eq!(list_summaries[0].tool_calls, 1);

    // 续聊第二轮，验证带上首轮 <environment_context> + 第一轮 4 条表面消息 + System + 新 User = 7 条
    let resumed = sessions2.get("e2e-session").unwrap();
    let reply2 = app2
        .agent_loop
        .run(resumed, "继续吧", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply2, "重启后恢复续聊成功");
    assert_eq!(*seen2.lock().unwrap(), vec![7]);

    let report2 = traj2
        .inspect("e2e-session", TimelineMode::Sequence)
        .unwrap();
    assert_eq!(report2.snapshot.stats.turns, 2);
    assert_eq!(report2.snapshot.stats.steps, 3);
    assert_eq!(report2.snapshot.stats.tool_calls, 1);

    app2.shutdown().await;
}
