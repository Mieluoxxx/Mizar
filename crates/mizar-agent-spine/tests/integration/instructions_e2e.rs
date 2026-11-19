use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, CordisError, PluginDef};
use futures::channel::mpsc;
use mizar_agent_loop::AgentLoopContextExt;
use mizar_agent_spine::{
    AgentInstructions, AgentSpineConfig, AgentSpinePlugin, AgentToolsMode,
    load_mizar_config_with_global,
};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, Message,
    StreamChunk, ToolCall,
};
use mizar_session::SessionContextExt;
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
async fn coding_stack_loads_agents_claude_skills_subdirs_and_persists_across_restart() {
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

    // settings.json 定制 instructions，并同时写入 AGENTS.md、CLAUDE.md、技能与子目录指令
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
                    Message::User { content } => format!("USR:{content}"),
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
        let mut cfg = AgentSpineConfig::coding(&ws).with_session_db(&db_path);
        loaded.apply_to_spine(&mut cfg);
        if let AgentToolsMode::Coding {
            ref mut mcp,
            ref mut skills,
            ref mut instructions,
            ..
        } = cfg.tools_mode
        {
            mcp.load_config_files = false;
            skills.agents_home = Some(agents_home.clone());
            instructions.agents_home = Some(agents_home.clone());
        }
        cfg
    };

    // 1. 首次启动运行：同时验证 /deploy 技能手势展开不被 ContextNotice 截断，且 read 触达 sub/ 后注入子目录指令
    let seen1 = Arc::new(Mutex::new(Vec::new()));
    let app1 = Context::new();
    let spine1 = app1.plugin(AgentSpinePlugin, build_cfg());
    let adapter1 = Arc::new(InspectAdapter {
        step: AtomicUsize::new(0),
        seen_messages: Arc::clone(&seen1),
    });
    app1.plugin(
        PluginDef::sync("test-llm", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter1);
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
    let _ = spine1.await;
    for _ in 0..4 {
        for rt in app1.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }
    }

    let s1 = app1.sessions().unwrap().create(Some("instr-sess".into()));
    let reply1 = app1
        .agent_loop()
        .unwrap()
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
        let step0 = steps[0].join("\n");
        assert!(step0.contains("GLOBAL_MIZAR_RULE"), "step0:\n{step0}");
        assert!(step0.contains("WS_AGENTS_RULE"), "step0:\n{step0}");
        assert!(step0.contains("WS_CLAUDE_RULE"), "step0:\n{step0}");
        assert!(step0.contains("DEPLOY_SKILL_BODY"), "step0:\n{step0}");
        assert!(!step0.contains("SUB_AGENTS_RULE"), "step0:\n{step0}");

        let step1 = steps[1].join("\n");
        assert!(step1.contains("SUB_AGENTS_RULE"), "step1:\n{step1}");
        assert!(step1.contains("SUB_CLAUDE_RULE"), "step1:\n{step1}");
    }

    let report1 = app1
        .service::<AgentInstructions>("agentInstructions")
        .unwrap()
        .session_report(&s1.id)
        .unwrap();
    assert!(report1.scopes.iter().any(|s| s.display_path == "AGENTS.md"));
    assert!(report1.scopes.iter().any(|s| s.display_path == "CLAUDE.md"));
    assert!(
        report1
            .scopes
            .iter()
            .any(|s| s.display_path == "sub/AGENTS.md")
    );
    app1.fiber().dispose().await;

    // 2. 重启进程恢复会话（且文件未改，但之前有 baseline + sub delta）：续聊应不重复追加无意义基线，且若离线改写则能感知更新
    std::fs::write(ws.join("CLAUDE.md"), "WS_CLAUDE_RULE_OFFLINE_EDIT").unwrap();
    let seen2 = Arc::new(Mutex::new(Vec::new()));
    let app2 = Context::new();
    let spine2 = app2.plugin(AgentSpinePlugin, build_cfg());
    let adapter2 = Arc::new(InspectAdapter {
        step: AtomicUsize::new(10),
        seen_messages: Arc::clone(&seen2),
    });
    app2.plugin(
        PluginDef::sync("test-llm-2", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter2);
            ctx.effect("test-llm-2.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("test", a)?))
            })?;
            Ok::<_, CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();
    let _ = spine2.await;
    for _ in 0..4 {
        for rt in app2.registry().values() {
            for f in rt.fibers() {
                let _ = f.wait_settled().await;
            }
        }
    }

    let resumed = app2.sessions().unwrap().get("instr-sess").unwrap();
    let reply2 = app2
        .agent_loop()
        .unwrap()
        .run(resumed, "继续", CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply2, "done_turn");
    let step2_all = seen2.lock().unwrap()[0].join("\n");
    assert!(step2_all.contains("WS_CLAUDE_RULE_OFFLINE_EDIT"));

    app2.fiber().dispose().await;
}
