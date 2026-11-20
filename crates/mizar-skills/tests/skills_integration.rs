use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cordis::{Context, Service};
use futures::channel::mpsc;
use mizar_agent_loop::{AgentLoop, AgentLoopConfig, AgentLoopContextExt};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmRuntime, Message,
    StreamChunk, ToolCall,
};
use mizar_session::{SessionContextExt, SessionEvent};
use mizar_skills::{
    SKILL_CATALOG_SOURCE, SKILL_INVOCATION_SOURCE, SkillInvocationPolicy, SkillRegistration,
    SkillRegistry, SkillRegistryConfig, SkillResourceBase, SkillSource, SkillsConfig,
    SkillsContextExt, SkillsPlugin, extract_skill_gestures, is_skill_name, parse_skill_markdown,
};
use mizar_system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_tools::{ToolRuntime, ToolsContextExt};
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
        extract_skill_gestures(
            "check /usr/bin and 5/8 then /rust-review and /deploy-now please /rust-review"
        ),
        vec!["rust-review".to_string(), "deploy-now".to_string()]
    );
}

#[tokio::test]
async fn discovers_skills_by_layer_rank_and_runs_progressive_loop() {
    let base = std::env::temp_dir().join(format!("mizar_skills_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    let sub_cwd = repo.join("crates/app");
    let user_agents = base.join("user-agents");
    let custom_dir = base.join("custom");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(&sub_cwd).unwrap();
    std::fs::create_dir_all(repo.join(".agents/skills/alpha-skill")).unwrap();
    std::fs::create_dir_all(repo.join(".agents/skills/manual-only")).unwrap();
    std::fs::create_dir_all(repo.join(".agents/skills/shared-skill")).unwrap();
    // .mizar/skills 不再作为默认读取路径
    std::fs::create_dir_all(repo.join(".mizar/skills/legacy-ignored")).unwrap();
    std::fs::create_dir_all(user_agents.join("skills/.system/ignored-skill")).unwrap();
    std::fs::create_dir_all(user_agents.join("skills/alpha-skill")).unwrap();
    std::fs::create_dir_all(user_agents.join("skills/pack/skills/nested-skill")).unwrap();
    std::fs::create_dir_all(&custom_dir).unwrap();

    // 同名 alpha-skill：project-agents (rank 200) 覆盖 user-agents (rank 500)
    std::fs::write(
        repo.join(".agents/skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: Project alpha guide\n---\nPROJECT_BODY_V1",
    )
    .unwrap();
    std::fs::write(
        user_agents.join("skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: User alpha guide\n---\nUSER_BODY",
    )
    .unwrap();
    std::fs::write(
        user_agents.join("skills/.system/ignored-skill/SKILL.md"),
        "---\nname: ignored-skill\ndescription: Should be skipped\n---\nIGNORED",
    )
    .unwrap();
    std::fs::write(
        repo.join(".mizar/skills/legacy-ignored/SKILL.md"),
        "---\nname: legacy-ignored\ndescription: Legacy path skipped\n---\nLEGACY",
    )
    .unwrap();
    // 技能包嵌套目录（如 ~/.agents/skills/pstack/skills/<name>/SKILL.md）与包内非根 README.md 处理
    std::fs::write(
        user_agents.join("skills/pack/README.md"),
        "# Pack Readme without frontmatter",
    )
    .unwrap();
    std::fs::write(
        user_agents.join("skills/pack/skills/nested-skill/SKILL.md"),
        "---\nname: nested-skill\ndescription: Nested pack helper\n---\nNESTED_BODY",
    )
    .unwrap();
    // disable-model-invocation: true 仅能通过 /manual-only 触发
    std::fs::write(
        repo.join(".agents/skills/manual-only/SKILL.md"),
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
            self.seen_tools
                .lock()
                .unwrap()
                .push(options.tools.iter().map(|t| t.name.clone()).collect());
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
    ctx.plugin(mizar_session::SessionStore::plugin(), ())
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
    skills_cfg.agents_home = Some(user_agents.clone());
    skills_cfg.custom_skill_dirs = vec![custom_dir.clone()];
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
    let _llm_cleanup = ctx
        .llm()
        .unwrap()
        .register_adapter("scripted", adapter)
        .unwrap();

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
                Message::User { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n---\n");
        assert!(step1_text.contains("<available_skills>"));
        assert!(step1_text.contains("`alpha-skill`: Project alpha guide"));
        assert!(step1_text.contains("`shared-skill`: From project agents"));
        assert!(step1_text.contains("`runtime-skill`: Runtime helper"));
        assert!(step1_text.contains("`nested-skill`: Nested pack helper"));
        assert!(!step1_text.contains("manual-only`: Deploy release"));
        assert!(!step1_text.contains("ignored-skill"));
        assert!(!step1_text.contains("legacy-ignored"));
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
        repo.join(".agents/skills/alpha-skill/SKILL.md"),
        "---\nname: alpha-skill\ndescription: Project alpha guide v2\n---\nPROJECT_BODY_V2",
    )
    .unwrap();
    let loaded_v2 = ctx
        .skills()
        .unwrap()
        .get(
            "alpha-skill",
            mizar_skills::SkillLookupOptions {
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
            SessionEvent::ContextNotice { source, text, .. } if source == SKILL_CATALOG_SOURCE => {
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

#[tokio::test]
async fn include_user_roots_false_skips_user_skills_but_keeps_project_skills() {
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("mizar_skills_iso_{}_{uniq}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);

    let home_dir = base.join("home");
    let user_agents = home_dir.join(".agents");
    let repo = base.join("repo");
    std::fs::create_dir_all(user_agents.join("skills/user-only")).unwrap();
    std::fs::create_dir_all(repo.join(".agents/skills/proj-only")).unwrap();

    std::fs::write(
        user_agents.join("skills/user-only/SKILL.md"),
        "---\nname: user-only\ndescription: Global personal skill\n---\nUSER_ONLY_BODY",
    )
    .unwrap();
    std::fs::write(
        repo.join(".agents/skills/proj-only/SKILL.md"),
        "---\nname: proj-only\ndescription: Workspace skill\n---\nPROJ_ONLY_BODY",
    )
    .unwrap();

    let ctx = Context::new();
    ctx.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    ctx.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    ctx.plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())
        .await
        .unwrap();
    let mut cfg = SkillsConfig::new(&repo);
    cfg.agents_home = Some(user_agents.clone());
    cfg.include_user_roots = false;
    ctx.plugin(SkillsPlugin, cfg).await.unwrap();
    for rt in ctx.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }

    let names: Vec<String> = ctx
        .skills()
        .unwrap()
        .list(mizar_skills::SkillLookupOptions {
            cwd: Some(repo.clone()),
            cancel: CancellationToken::new(),
        })
        .await
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert!(names.contains(&"proj-only".to_string()), "{names:?}");
    assert!(!names.contains(&"user-only".to_string()), "{names:?}");

    // 即使 cwd 恰好等于 home_dir（其下有 .agents/skills），也必须屏蔽 user-only
    let home_names: Vec<String> = ctx
        .skills()
        .unwrap()
        .list(mizar_skills::SkillLookupOptions {
            cwd: Some(home_dir.clone()),
            cancel: CancellationToken::new(),
        })
        .await
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert!(
        !home_names.contains(&"user-only".to_string()),
        "{home_names:?}"
    );

    ctx.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn concurrent_collect_invalidate_does_not_cache_stale_and_resumed_dedups_catalog() {
    use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
    use mizar_skills::{
        SkillCandidate, SkillDefinition, SkillProvider, SkillProviderObservation, SkillSummary,
    };

    struct SlowProvider {
        skills: Arc<Mutex<Vec<(String, String)>>>,
        started: Arc<tokio::sync::Notify>,
        proceed: Arc<tokio::sync::Notify>,
        block_first: std::sync::atomic::AtomicBool,
    }

    impl SkillProvider for SlowProvider {
        fn name(&self) -> &str {
            "slow-test"
        }

        fn fingerprint(
            &self,
            _options: &mizar_skills::SkillLookupOptions,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<u64>> + Send + '_>> {
            let count = self.skills.lock().unwrap().len() as u64;
            Box::pin(async move { Some(count) })
        }

        fn list(
            &self,
            _options: mizar_skills::SkillLookupOptions,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<SkillProviderObservation, String>>
                    + Send
                    + '_,
            >,
        > {
            let skills = Arc::clone(&self.skills);
            let started = Arc::clone(&self.started);
            let proceed = Arc::clone(&self.proceed);
            let should_block = self
                .block_first
                .swap(false, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move {
                let initial_snap = skills.lock().unwrap().clone();
                if should_block {
                    started.notify_one();
                    proceed.notified().await;
                }
                let candidates = initial_snap
                    .into_iter()
                    .map(|(n, d)| SkillCandidate {
                        summary: SkillSummary {
                            path: None,
                            name: n,
                            description: d,
                            when_to_use: None,
                            invocation: SkillInvocationPolicy::default(),
                            source: SkillSource::Runtime,
                            provider: "slow-test".to_string(),
                            resource_base: None,
                        },
                        rank: 10,
                        locator: Arc::new(()),
                        metadata: None,
                    })
                    .collect();
                Ok(SkillProviderObservation {
                    candidates,
                    complete: true,
                })
            })
        }

        fn get(
            &self,
            candidate: SkillCandidate,
            _options: mizar_skills::SkillLookupOptions,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<Option<SkillDefinition>, String>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async move {
                Ok(Some(SkillDefinition {
                    summary: candidate.summary,
                    content: "slow body".to_string(),
                    metadata: None,
                }))
            })
        }
    }

    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base =
        std::env::temp_dir().join(format!("mizar_skills_race_{}_{uniq}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join(".agents/skills/alpha")).unwrap();
    std::fs::create_dir_all(base.join(".agents/skills/beta")).unwrap();
    std::fs::write(
        base.join(".agents/skills/alpha/SKILL.md"),
        "---\nname: alpha\ndescription: Alpha skill\n---\nALPHA_BODY",
    )
    .unwrap();
    std::fs::write(
        base.join(".agents/skills/beta/SKILL.md"),
        "---\nname: beta\ndescription: Beta skill\n---\nBETA_BODY",
    )
    .unwrap();

    // 1. 并发 invalidate_cache 与 collect 竞态：中途 revision 失效后不得将旧快照写入 cache
    let ctx_race = Context::new();
    ctx_race
        .plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())
        .await
        .unwrap();
    let reg = ctx_race.skills().unwrap();
    let started = Arc::new(tokio::sync::Notify::new());
    let proceed = Arc::new(tokio::sync::Notify::new());
    let provider_skills = Arc::new(Mutex::new(vec![("v1-skill".into(), "V1".into())]));
    let provider = Arc::new(SlowProvider {
        skills: Arc::clone(&provider_skills),
        started: Arc::clone(&started),
        proceed: Arc::clone(&proceed),
        block_first: std::sync::atomic::AtomicBool::new(true),
    });
    let _p_cleanup = reg.register_provider(provider).unwrap();

    let reg_task = Arc::clone(&reg);
    let fut = tokio::spawn(async move {
        reg_task
            .list(mizar_skills::SkillLookupOptions {
                cwd: None,
                cancel: CancellationToken::new(),
            })
            .await
    });
    started.notified().await;
    // 在 list 返回前更新数据源并 invalidate_cache()
    provider_skills
        .lock()
        .unwrap()
        .push(("v2-skill".into(), "V2".into()));
    reg.invalidate_cache();
    proceed.notify_one();
    let _first = fut.await.unwrap();
    proceed.notify_waiters();

    // 下次 list 不得命中 stale v1 cache，必须包含 v2-skill
    let second = reg
        .list(mizar_skills::SkillLookupOptions {
            cwd: None,
            cancel: CancellationToken::new(),
        })
        .await;
    assert!(
        second.iter().any(|s| s.name == "v2-skill"),
        "expected v2-skill after concurrent invalidation, got: {:?}",
        second.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    ctx_race.fiber().dispose().await;

    // 2. 恢复会话去重 catalog + 取消的 pre-step 不锁存 gesture seq
    let ctx1 = Context::new();
    ctx1.plugin(mizar_session::SessionStore::plugin(), ())
        .await
        .unwrap();
    ctx1.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    ctx1.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    ctx1.plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())
        .await
        .unwrap();
    let mut cfg = SkillsConfig::new(&base);
    cfg.include_user_roots = false;
    ctx1.plugin(SkillsPlugin, cfg.clone()).await.unwrap();

    let s = ctx1
        .sessions()
        .unwrap()
        .create(Some("s-skill-resume".into()));
    s.append(SessionEvent::user("run /alpha please"));
    let _ = ctx1
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&s),
                turn: 1,
                step: 1,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
            },
        )
        .await;

    // 模拟已取消的 pre-step 带有 /beta：不得推进 processed_gesture_seq
    s.append(SessionEvent::user("and /beta too"));
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let _ = ctx1
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&s),
                turn: 2,
                step: 1,
                provider: None,
                model: None,
                cancel: cancelled,
            },
        )
        .await;
    // 下一次未取消的 pre-step 仍应成功展开 /beta，仅此一次
    let _ = ctx1
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&s),
                turn: 2,
                step: 1,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
            },
        )
        .await;
    let _ = ctx1
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&s),
                turn: 2,
                step: 2,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
            },
        )
        .await;
    let beta_count = s.with_visible_entries(|entries| {
        entries
            .iter()
            .filter(|e| match &e.event {
                SessionEvent::ContextNotice { source, text, .. } => {
                    source == SKILL_INVOCATION_SOURCE && text.contains("BETA_BODY")
                }
                _ => false,
            })
            .count()
    });
    assert_eq!(
        beta_count, 1,
        "cancelled gesture check must allow retry once and not duplicate"
    );

    let snapshot_entries = s.all_entries();
    let snapshot_header = s.header();
    ctx1.fiber().dispose().await;

    // 恢复到新 Context（空的 published_catalogs）时不应重复追加相同 catalog
    let ctx2 = Context::new();
    ctx2.plugin(mizar_session::SessionStore::plugin(), ())
        .await
        .unwrap();
    ctx2.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    ctx2.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    ctx2.plugin(SkillRegistry::plugin(), SkillRegistryConfig::default())
        .await
        .unwrap();
    ctx2.plugin(SkillsPlugin, cfg).await.unwrap();

    let resumed = Arc::new(mizar_session::Session::from_loaded(
        ctx2.clone(),
        mizar_session::LoadedSession {
            header: snapshot_header,
            entries: snapshot_entries,
            max_persisted_seq: None,
            skipped_events: 0,
        },
    ));
    let before_cat = resumed.with_visible_entries(|e| {
        e.iter()
            .filter(|x| matches!(&x.event, SessionEvent::ContextNotice { source, .. } if source == SKILL_CATALOG_SOURCE))
            .count()
    });
    assert_eq!(before_cat, 1);

    let _ = ctx2
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&resumed),
                turn: 3,
                step: 1,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
            },
        )
        .await;
    let after_cat = resumed.with_visible_entries(|e| {
        e.iter()
            .filter(|x| matches!(&x.event, SessionEvent::ContextNotice { source, .. } if source == SKILL_CATALOG_SOURCE))
            .count()
    });
    assert_eq!(
        after_cat, 1,
        "resumed session must hydrate catalog state instead of duplicating SKILL_CATALOG_SOURCE"
    );

    ctx2.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&base);
}
