use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{Context, Service};
use dsh_agent_loop::{AgentLoop, AgentLoopConfig, AgentLoopContextExt};
use dsh_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmRuntime, Message,
    StreamChunk, ToolCall,
};
use dsh_session::{SessionContextExt, SessionEvent};
use dsh_skills::{
    SKILL_CATALOG_SOURCE, SKILL_INVOCATION_SOURCE, SkillInvocationPolicy, SkillRegistration,
    SkillRegistry, SkillRegistryConfig, SkillResourceBase, SkillSource, SkillsConfig,
    SkillsContextExt, SkillsPlugin, extract_skill_gestures, is_skill_name, parse_skill_markdown,
};
use dsh_system_prompt::{SystemPrompt, SystemPromptConfig};
use dsh_tools::{ToolRuntime, ToolsContextExt};
use futures::channel::mpsc;
use tokio_util::sync::CancellationToken;

#[test]
fn validates_skill_names_frontmatter_and_gestures() {
    assert!(is_skill_name("rust-review"));
    assert!(is_skill_name("v2"));
    assert!(!is_skill_name("-bad"));
    assert!(!is_skill_name("BadCase"));
    assert!(!is_skill_name("bad--name"));

    let parsed = parse_skill_markdown(
        r#"---
name: rust-review
description: |
  Review Rust crates carefully.
  Check error handling.
disable-model-invocation: true
user-invocable: yes
when-to-use: Before merging Rust changes
metadata:
  owner: platform
---

1. Check `unsafe` blocks.
2. Run clippy.
"#,
    )
    .expect("valid frontmatter");
    assert_eq!(parsed.name, "rust-review");
    assert_eq!(
        parsed.description,
        "Review Rust crates carefully.\nCheck error handling."
    );
    assert!(!parsed.invocation.model_invocable);
    assert!(parsed.invocation.user_invocable);
    assert_eq!(
        parsed.when_to_use.as_deref(),
        Some("Before merging Rust changes")
    );
    assert!(parsed.body.contains("Run clippy."));

    assert!(parse_skill_markdown("---\ndisableModelInvocation: true\n---").is_err());

    assert_eq!(
        extract_skill_gestures("check /usr/bin and 5/8 then /rust-review and /deploy-now please /rust-review"),
        vec!["rust-review".to_string(), "deploy-now".to_string()]
    );
}

#[tokio::test]
async fn discovers_skills_by_layer_rank_and_runs_progressive_loop() {
    let base = std::env::temp_dir().join(format!("dsh_skills_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    let sub_cwd = repo.join("crates/app");
    let user_dsh = base.join("user-dsh");
    let custom_dir = base.join("custom");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(&sub_cwd).unwrap();
    std::fs::create_dir_all(repo.join(".dsh-rs/skills/alpha-skill")).unwrap();
    std::fs::create_dir_all(repo.join(".dsh-rs/skills/manual-only")).unwrap();
    std::fs::create_dir_all(repo.join(".agents/skills/shared-skill")).unwrap();
    std::fs::create_dir_all(user_dsh.join("skills/.system/ignored-skill")).unwrap();
    std::fs::create_dir_all(user_dsh.join("skills/alpha-skill")).unwrap();
    std::fs::create_dir_all(&custom_dir).unwrap();

    // 同名 alpha-skill：project-dsh (rank 100) 覆盖 user-dsh (rank 400)
    std::fs::write(
        repo.join(".dsh-rs/skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: Project alpha guide\n---\nPROJECT_BODY_V1",
    )
    .unwrap();
    std::fs::write(
        user_dsh.join("skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: User alpha guide\n---\nUSER_BODY",
    )
    .unwrap();
    std::fs::write(
        user_dsh.join("skills/.system/ignored-skill/SKILL.md"),
        "---\nname: ignored-skill\ndescription: Should be skipped\n---\nIGNORED",
    )
    .unwrap();
    // disable-model-invocation: true 仅能通过 /manual-only 触发
    std::fs::write(
        repo.join(".dsh-rs/skills/manual-only/SKILL.md"),
        "---\nname: manual-only\ndescription: Deploy release\ndisable-model-invocation: true\n---\nMANUAL_BODY",
    )
    .unwrap();
    // 扁平 .md + project-agents 优先级验证
    std::fs::write(
        repo.join(".agents/skills/shared-skill/SKILL.md"),
        "---\nname: shared-skill\ndescription: From project agents\n---\nAGENTS_BODY",
    )
    .unwrap();
    std::fs::write(
        custom_dir.join("shared-skill.md"),
        "---\nname: shared-skill\ndescription: From custom\n---\nCUSTOM_BODY",
    )
    .unwrap();
    std::fs::write(
        custom_dir.join("model-only.md"),
        "---\nname: model-only\ndescription: Hidden from slash\nuser-invocable: false\n---\nMODEL_ONLY_BODY",
    )
    .unwrap();

    struct ScriptedAdapter {
        step_calls: AtomicUsize,
        seen_messages: Arc<Mutex<Vec<Vec<Message>>>>,
        seen_tools: Arc<Mutex<Vec<Vec<String>>>>,
    }
    impl LlmAdapter for ScriptedAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let n = self.step_calls.fetch_add(1, Ordering::SeqCst);
            self.seen_messages
                .lock()
                .unwrap()
                .push(options.messages.clone());
            self.seen_tools.lock().unwrap().push(
                options.tools.iter().map(|t| t.name.clone()).collect(),
            );
            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                match n {
                    0 => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::tool_call(
                            0,
                            ToolCall {
                                id: "c1".into(),
                                name: "skill".into(),
                                arguments: r#"{"name":"alpha-skill"}"#.into(),
                            },
                        )));
                        let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                            reason: FinishReason::ToolCalls,
                            replay: None,
                        }));
                    }
                    _ => {
                        let _ = tx.unbounded_send(Ok(StreamChunk::text("done")));
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

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    let seen_tools = Arc::new(Mutex::new(Vec::new()));

    let ctx = Context::new();
    ctx.plugin(LlmRuntime::plugin(), ()).await.unwrap();
    ctx.plugin(dsh_session::SessionStore::plugin(), ())
        .await
        .unwrap();
    ctx.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    ctx.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    ctx.plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())
        .await
        .unwrap();

    let mut skills_cfg = SkillsConfig::new(&sub_cwd);
    skills_cfg.dsh_home = Some(user_dsh.clone());
    skills_cfg.custom_skill_dirs = vec![custom_dir.clone()];
    skills_cfg.agents_home = Some(base.join("missing-agents"));
    ctx.plugin(SkillsPlugin, skills_cfg).await.unwrap();
    ctx.plugin(AgentLoop::plugin(), AgentLoopConfig::default())
        .await
        .unwrap();

    for rt in ctx.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }

    // 注册运行时 skill (rank 250)，优先级低于 project-agents (200) 但高于 custom (300)
    let _rt_guard = ctx
        .skills()
        .unwrap()
        .register(SkillRegistration {
            name: "runtime-skill".to_string(),
            description: "Runtime helper".to_string(),
            when_to_use: None,
            invocation: Some(SkillInvocationPolicy::default()),
            source: SkillSource::Runtime,
            path: None,
            resource_base: Some(SkillResourceBase::Opaque {
                description: "in-memory".to_string(),
            }),
            content: "RUNTIME_BODY".to_string(),
            metadata: None,
        })
        .unwrap();

    let adapter = Arc::new(ScriptedAdapter {
        step_calls: AtomicUsize::new(0),
        seen_messages: Arc::clone(&seen_messages),
        seen_tools: Arc::clone(&seen_tools),
    });
    let _llm_cleanup = ctx.llm().unwrap().register_adapter("scripted", adapter).unwrap();

    let session = ctx.sessions().unwrap().create(None);
    let agent = ctx.agent_loop().unwrap();

    // 第一轮：触发目录注入（不含 manual-only 与 ignored-skill），同时用户显式输入 /manual-only，
    // 而 /model-only (user-invocable: false) 应被手势层忽略
    let reply = agent
        .run(
            Arc::clone(&session),
            "please run /manual-only and ignore /model-only",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply, "done");

    {
        let calls = seen_messages.lock().unwrap();
        assert_eq!(calls.len(), 2);

        // Step 1 消息中必须出现 skill-catalog + manual-only 的 skill_content，且 catalog 不包含正文或路径
        let step1_text: String = calls[0]
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n---\n");
        assert!(step1_text.contains("<available_skills>"));
        assert!(step1_text.contains("`alpha-skill`: Project alpha guide"));
        assert!(step1_text.contains("`shared-skill`: From project agents"));
        assert!(step1_text.contains("`runtime-skill`: Runtime helper"));
        assert!(!step1_text.contains("manual-only`: Deploy release"));
        assert!(!step1_text.contains("ignored-skill"));
        assert!(!step1_text.contains("PROJECT_BODY_V1"));
        assert!(step1_text.contains("<skill_content name=\"manual-only\">"));
        assert!(step1_text.contains("MANUAL_BODY"));
        assert!(!step1_text.contains("MODEL_ONLY_BODY"));

        // Step 1 调用 skill("alpha-skill") 的工具结果应拿到 PROJECT_BODY_V1
        let step2_tool_msg = calls[1]
            .iter()
            .find_map(|m| match m {
                Message::Tool { content, .. } => Some(content.clone()),
                _ => None,
            })
            .expect("tool result");
        assert!(step2_tool_msg.contains("<skill_content name=\"alpha-skill\">"));
        assert!(step2_tool_msg.contains("PROJECT_BODY_V1"));
    }

    // 确认 catalog 只注入过 1 次（Step 2 未重复发送）
    let catalog_count_in_log = session
        .log()
        .iter()
        .filter(|e| {
            matches!(
                &e.event,
                SessionEvent::ContextNotice { source, .. } if source == SKILL_CATALOG_SOURCE
            )
        })
        .count();
    assert_eq!(catalog_count_in_log, 1);

    let inv_count_in_log = session
        .log()
        .iter()
        .filter(|e| {
            matches!(
                &e.event,
                SessionEvent::ContextNotice { source, .. } if source == SKILL_INVOCATION_SOURCE
            )
        })
        .count();
    assert_eq!(inv_count_in_log, 1);

    // 验证文件修改后无需手动刷新即可拿到新正文并触发替换目录
    std::fs::write(
        repo.join(".dsh-rs/skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: Project alpha guide v2\n---\nPROJECT_BODY_V2",
    )
    .unwrap();
    let loaded_v2 = ctx
        .skills()
        .unwrap()
        .get(
            "alpha-skill",
            dsh_skills::SkillLookupOptions {
                cwd: Some(sub_cwd.clone()),
                cancel: CancellationToken::new(),
            },
        )
        .await
        .unwrap()
        .expect("updated skill");
    assert_eq!(loaded_v2.content, "PROJECT_BODY_V2");

    // 运行第 2 轮确认发布更新后的 catalog（含 "Project alpha guide v2"）
    let _ = agent
        .run(Arc::clone(&session), "next turn", CancellationToken::new())
        .await
        .unwrap();
    let log_after_update = session.log();
    let catalog_msgs: Vec<&str> = log_after_update
        .iter()
        .filter_map(|e| match &e.event {
            SessionEvent::ContextNotice { source, text } if source == SKILL_CATALOG_SOURCE => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(catalog_msgs.len(), 2);
    assert!(catalog_msgs[1].contains("replaces every earlier available-skills list"));
    assert!(catalog_msgs[1].contains("Project alpha guide v2"));

    // 当 skill 工具被屏蔽时，新会话不应收到任何 catalog 注入
    ctx.tools()
        .unwrap()
        .set_tool_selection(None, vec!["skill".to_string()]);
    let isolated_session = ctx.sessions().unwrap().create(None);
    let _ = agent
        .run(
            Arc::clone(&isolated_session),
            "hello",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(isolated_session.log().iter().all(|e| !matches!(
        &e.event,
        SessionEvent::ContextNotice { source, .. } if source == SKILL_CATALOG_SOURCE
    )));

    let _ = std::fs::remove_dir_all(&base);
}
