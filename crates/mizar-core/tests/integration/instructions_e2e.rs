use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::mpsc;
use mizar_core::spine::{
    AgentSpineConfig, AgentToolsMode, MizarRuntime, load_mizar_config_with_global,
};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, Message, StreamChunk, ToolCall,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "mizar_spine_instr_{name}_{}_{nanos}",
        std::process::id()
    ))
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn coding_stack_loads_static_instructions_ignores_subdirs_and_reloads_on_restart() {
    let tmp = temp_dir("e2e");
    let _guard = DirGuard(tmp.clone());

    let global_dir = tmp.join("global-mizar");
    let agents_home = tmp.join("global-agents");
    let ws = tmp.join("repo");
    let db_path = tmp.join("sessions.sqlite");

    std::fs::create_dir_all(&global_dir).unwrap();
    std::fs::create_dir_all(agents_home.join("skills/deploy")).unwrap();
    std::fs::create_dir_all(ws.join(".git")).unwrap();
    std::fs::create_dir_all(ws.join(".mizar")).unwrap();
    std::fs::create_dir_all(ws.join("sub")).unwrap();

    // settings.json 定制 instructions，并同时写入 AGENTS.md、CLAUDE.md、技能与不应加载的子目录指令
    std::fs::write(
        ws.join(".mizar/settings.json"),
        r#"{
            "instructions": {
                "enabled": true,
                "maxBytes": 65536
            }
        }"#,
    )
    .unwrap();
    std::fs::write(global_dir.join("AGENTS.md"), "GLOBAL_MIZAR_RULE").unwrap();
    std::fs::write(ws.join("AGENTS.md"), "WS_AGENTS_RULE").unwrap();
    std::fs::write(ws.join("CLAUDE.md"), "WS_CLAUDE_RULE").unwrap();
    std::fs::write(ws.join("sub/AGENTS.md"), "SUB_AGENTS_RULE").unwrap();
    std::fs::write(ws.join("sub/CLAUDE.md"), "SUB_CLAUDE_RULE").unwrap();
    std::fs::write(ws.join("sub/target.txt"), "hello subdir").unwrap();
    std::fs::write(
        agents_home.join("skills/deploy/SKILL.md"),
        "---\nname: deploy\ndescription: Deploy helper\n---\nDEPLOY_SKILL_BODY\n",
    )
    .unwrap();

    struct InspectAdapter {
        step: AtomicUsize,
        seen_messages: Arc<Mutex<Vec<Vec<String>>>>,
    }
    impl LlmAdapter for InspectAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let texts: Vec<String> = options
                .messages
                .iter()
                .map(|m| match m {
                    Message::System { content } => format!("SYS:{content}"),
                    Message::User { content, .. } => format!("USR:{content}"),
                    Message::Assistant { .. } => "ASST".to_string(),
                    Message::Tool { content, .. } => format!("TOOL:{content}"),
                })
                .collect();
            self.seen_messages.lock().unwrap().push(texts);
            let step = self.step.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                match step {
                    0 => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "c1".into(),
                                name: "read".into(),
                                arguments: json!({"path": "sub/target.txt"}).to_string(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    _ => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("done_turn")));
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

    let build_cfg = || {
        let loaded = load_mizar_config_with_global(&ws, Some(&global_dir));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        let mut cfg = AgentSpineConfig::coding(&ws)
            .with_session_db(&db_path)
            .with_disabled_plugins(["session-title"]);
        loaded.apply_to_spine(&mut cfg);
        if let AgentToolsMode::Coding {
            #[cfg(feature = "mcp")]
            ref mut mcp,
            ref mut skills,
            ref mut instructions,
            ..
        } = cfg.tools_mode
        {
            #[cfg(feature = "mcp")]
            {
                mcp.load_config_files = false;
            }
            skills.agents_home = Some(agents_home.clone());
            instructions.agents_home = Some(agents_home.clone());
        }
        cfg
    };

    // 1. 首次启动运行：指令进 SystemPrompt 静态段，/deploy 技能展开保留在消息流，且 read 触达 sub/ 后仍不注入子目录指令
    let seen1 = Arc::new(Mutex::new(Vec::new()));
    let runtime1 = MizarRuntime::build(build_cfg()).await.unwrap();
    let adapter1 = Arc::new(InspectAdapter {
        step: AtomicUsize::new(0),
        seen_messages: Arc::clone(&seen1),
    });
    runtime1.register_adapter("test", adapter1).unwrap();

    let s1 = runtime1.sessions.create(Some("instr-sess".into()));
    let reply1 = runtime1
        .agent_loop
        .run(
            Arc::clone(&s1),
            "/deploy 请读取 sub/target.txt",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply1, "done_turn");

    {
        let steps = seen1.lock().unwrap();
        assert_eq!(steps.len(), 2);
        let step0_msgs = &steps[0];
        let step1_msgs = &steps[1];

        let sys0 = &step0_msgs[0];
        let sys1 = &step1_msgs[0];
        assert!(sys0.starts_with("SYS:"));
        assert_eq!(
            sys0, sys1,
            "system prompt must stay byte-identical across tool calls"
        );
        assert!(sys0.contains("GLOBAL_MIZAR_RULE"), "sys0:\n{sys0}");
        assert!(sys0.contains("WS_AGENTS_RULE"), "sys0:\n{sys0}");
        assert!(sys0.contains("WS_CLAUDE_RULE"), "sys0:\n{sys0}");
        assert!(!sys0.contains("SUB_AGENTS_RULE"), "sys0:\n{sys0}");
        assert!(!sys0.contains("SUB_CLAUDE_RULE"), "sys0:\n{sys0}");

        let pos_env = step0_msgs
            .iter()
            .position(|m| m.contains("<environment_context>"))
            .expect("missing <environment_context>");
        let pos_skill = step0_msgs
            .iter()
            .position(|m| m.contains("DEPLOY_SKILL_BODY"))
            .expect("missing DEPLOY_SKILL_BODY");
        let pos_user = step0_msgs
            .iter()
            .position(|m| m.contains("/deploy 请读取 sub/target.txt"))
            .expect("missing user message");
        assert!(
            pos_env < pos_skill && pos_skill < pos_user,
            "unexpected Turn-0 ordering: env={pos_env}, skill={pos_skill}, user={pos_user}, msgs={step0_msgs:?}"
        );

        let step1_all = step1_msgs.join("\n");
        assert!(
            !step1_all.contains("SUB_AGENTS_RULE"),
            "step1:\n{step1_all}"
        );
        assert!(
            !step1_all.contains("SUB_CLAUDE_RULE"),
            "step1:\n{step1_all}"
        );
    }

    let report1 = runtime1.instructions.as_ref().unwrap().report();
    assert!(report1.scopes.iter().any(|s| s.display_path == "AGENTS.md"));
    assert!(report1.scopes.iter().any(|s| s.display_path == "CLAUDE.md"));
    assert!(
        !report1
            .scopes
            .iter()
            .any(|s| s.display_path == "sub/AGENTS.md")
    );
    runtime1.shutdown().await;

    // 2. 重启进程恢复会话：离线改写 CLAUDE.md 后新实例启动静态段立即反映新内容
    std::fs::write(ws.join("CLAUDE.md"), "WS_CLAUDE_RULE_OFFLINE_EDIT").unwrap();
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let runtime2 = MizarRuntime::build(build_cfg()).await.unwrap();
    let adapter2 = Arc::new(InspectAdapter {
        step: AtomicUsize::new(10),
        seen_messages: Arc::clone(&seen2),
    });
    runtime2.register_adapter("test", adapter2).unwrap();

    let resumed = runtime2.sessions.get("instr-sess").unwrap();
    let reply2 = runtime2
        .agent_loop
        .run(resumed, "继续", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply2, "done_turn");
    let step2_sys = seen2.lock().unwrap()[0][0].clone();
    assert!(step2_sys.contains("WS_CLAUDE_RULE_OFFLINE_EDIT"));

    runtime2.shutdown().await;
}
