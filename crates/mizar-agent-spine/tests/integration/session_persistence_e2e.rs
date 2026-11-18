use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, CordisError, PluginDef};
use mizar_agent_loop::AgentLoopContextExt;
use mizar_agent_spine::{AgentSpineConfig, AgentSpinePlugin};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError,
    StreamChunk, TokenUsage, ToolCall,
};
use mizar_session::{SessionContextExt, SessionListQuery};
use mizar_trajectory::{TimelineMode, TrajectoryRuntime};
use futures::channel::mpsc;
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

async fn mount_spine_with_db(db_path: &PathBuf, ws: &PathBuf, adapter: Arc<TwoTurnAdapter>) -> Context {
    let app = Context::new();
    let spine_fiber = app.plugin(
        AgentSpinePlugin,
        AgentSpineConfig::new(ws).with_session_db(db_path),
    );
    app.plugin(
        PluginDef::sync("test-llm", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter);
            ctx.effect("test-llm.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("test", a)?))
            })?;
            Ok::<_, CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();
    let _ = spine_fiber.await;
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }
    app
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
    let s1 = app1.sessions().unwrap().create(Some("e2e-session".into()));

    let reply1 = app1
        .agent_loop()
        .unwrap()
        .run(Arc::clone(&s1), "帮我看看现在几点", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply1, "第一轮已完成");
    assert_eq!(s1.header().cwd.as_deref(), Some(ws.to_string_lossy().as_ref()));
    assert_eq!(s1.header().title.as_deref(), Some("帮我看看现在几点"));
    app1.fiber().dispose().await;

    // 2. 模拟进程完全重启，挂载新 Context + SQLite，恢复同一会话并继续第二轮对话
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let adapter2 = Arc::new(TwoTurnAdapter {
        calls: AtomicUsize::new(2),
        seen_history_lens: Arc::clone(&seen2),
    });
    let app2 = mount_spine_with_db(&db_path, &ws, adapter2).await;

    let sessions2 = app2.sessions().unwrap();
    let headers = sessions2
        .list_headers(&SessionListQuery {
            cwd: Some(ws.to_string_lossy().to_string()),
            limit: Some(10),
        })
        .unwrap();
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0].id, "e2e-session");

    // 重启后仅靠 TrajectoryRuntime 也可直接恢复并检查第一轮完整轨迹（含 Turn/Step/ToolCall）
    let traj2 = app2.service::<TrajectoryRuntime>("trajectory").unwrap();
    let list_summaries = traj2.list_sessions();
    assert_eq!(list_summaries.len(), 1);
    assert_eq!(list_summaries[0].turns, 1);
    assert_eq!(list_summaries[0].steps, 2);
    assert_eq!(list_summaries[0].tool_calls, 1);

    // 续聊第二轮，验证带上第一轮的 4 条表面消息 + System + 新 User = 6 条
    let resumed = sessions2.get("e2e-session").unwrap();
    let reply2 = app2
        .agent_loop()
        .unwrap()
        .run(resumed, "继续吧", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply2, "重启后恢复续聊成功");
    assert_eq!(*seen2.lock().unwrap(), vec![6]);

    let report2 = traj2.inspect("e2e-session", TimelineMode::Sequence).unwrap();
    assert_eq!(report2.snapshot.stats.turns, 2);
    assert_eq!(report2.snapshot.stats.steps, 3);
    assert_eq!(report2.snapshot.stats.tool_calls, 1);

    app2.fiber().dispose().await;
}
