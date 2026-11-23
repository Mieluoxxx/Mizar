//! 集成测试：验证 `"spinePlugins"` 在真实 Cordis 运行时下的 Gate 2 核心自保护、
//! 启动屏蔽与热卸载/热挂载生命周期（工具、系统提示词、waterfall 钩子、事件监听与服务注册）。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{BoxFuture, Context, PluginDef};
use futures::channel::mpsc;
use mizar_agent_loop::{
    AgentLoopContextExt, AgentPreStepDecision, AgentPreStepInput, EVENT_AGENT_PRE_STEP,
};
use mizar_agent_spine::{
    AgentInstructions, AgentSpineConfig, AgentSpinePlugin, CompactionContextExt,
    EVENT_SPINE_PLUGINS_CHANGE, RECALL_TOOL_NAME, SUBAGENT_TOOL_NAME, SessionTitleContextExt,
    SessionTitleRuntime, SpinePluginError, SpinePluginsChangeEvent, SpinePluginsContextExt,
    SubagentContextExt, SubagentRuntime,
};
use mizar_compaction::CompactionRuntime;
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, ModelInfo,
    ProviderInfo, StreamChunk,
};
use mizar_mcp::McpManager;
use mizar_session::{SessionContextExt, SessionEvent};
use mizar_skills::SkillRegistry;
use mizar_system_prompt::SystemPromptContextExt;
use mizar_tools::{ToolDefinition, ToolExecution, ToolsContextExt};
use mizar_trajectory::TrajectoryRuntime;
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "mizar_spine_plugins_{tag}_{}_{nanos}",
        std::process::id()
    ))
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct SimpleReplyAdapter;

impl LlmAdapter for SimpleReplyAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "mock".into(),
            name: "Mock".into(),
            protocol: "mock".into(),
            models: vec![ModelInfo {
                id: "mock-1".into(),
                name: "Mock 1".into(),
                context_window: Some(32_000),
                max_tokens: Some(1_024),
                reasoning_efforts: vec![],
                input: Vec::new(),
                ..Default::default()
            }],
        })
    }

    fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                index: 0,
                text: "core-loop-ok".into(),
            }));
            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            }));
        });
        Box::pin(rx)
    }
}

async fn wait_all_settled(app: &Context) {
    for rt in app.registry().values() {
        for fiber in rt.fibers() {
            let _ = fiber.wait_settled().await;
        }
    }
}

#[tokio::test]
async fn protected_core_modules_cannot_be_disabled_at_startup_or_runtime() {
    let ws = temp_dir("protected");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(&ws).unwrap();

    let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins([
        "llm",
        "sessions",
        "tools",
        "systemPrompt",
        "agentLoop",
        "spinePlugins",
        "nonexistent-module",
    ]);
    cfg.agent.provider = Some("mock".into());
    cfg.agent.model = Some("mock-1".into());

    let app = Context::new();
    app.plugin(AgentSpinePlugin, cfg).await.unwrap();
    app.plugin(
        PluginDef::sync("mock-adapter", |ctx, ()| {
            let llm = ctx.llm()?;
            let adapter = Arc::new(SimpleReplyAdapter);
            ctx.effect("mock-adapter.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("mock", adapter)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();
    wait_all_settled(&app).await;

    let ctrl = app.spine_plugins().unwrap();
    let errs = ctrl.startup_errors();
    for protected_id in [
        "llm",
        "sessions",
        "tools",
        "systemPrompt",
        "agentLoop",
        "spinePlugins",
    ] {
        assert!(
            ctrl.is_protected(protected_id) && ctrl.is_enabled(protected_id),
            "core module {protected_id} must stay protected and active"
        );
        assert!(
            errs.iter().any(|e| e.contains(protected_id)),
            "expected startup error mentioning {protected_id}, got: {errs:?}"
        );
        let err = ctrl.set_enabled(protected_id, false).await.unwrap_err();
        assert!(
            matches!(err, SpinePluginError::ProtectedModule(ref id) if id == protected_id),
            "expected ProtectedModule({protected_id}), got: {err:?}"
        );
        assert!(!ctrl.set_enabled(protected_id, true).await.unwrap());
    }
    assert!(errs.iter().any(|e| e.contains("nonexistent-module")));
    assert!(matches!(
        ctrl.set_enabled("nonexistent-module", false).await,
        Err(SpinePluginError::UnknownModule(_))
    ));

    // 主循环与核心服务依然完好可用
    let sessions = app.sessions().unwrap();
    let _tools = app.tools().unwrap();
    let _prompt = app.system_prompt().unwrap();
    let loop_svc = app.agent_loop().unwrap();
    let session = sessions.create(None);
    let answer = loop_svc
        .run(Arc::clone(&session), "ping", CancellationToken::new())
        .await
        .unwrap();
    assert!(answer.contains("core-loop-ok"), "unexpected: {answer}");
}

#[tokio::test]
async fn optional_modules_startup_disable_hot_mount_and_hot_unload_cleanly() {
    let ws = temp_dir("hot_lifecycle");
    let _guard = DirGuard(ws.clone());
    std::fs::create_dir_all(ws.join(".git")).unwrap();
    std::fs::create_dir_all(ws.join(".agents/skills/demo-skill")).unwrap();
    std::fs::write(ws.join("AGENTS.md"), "# Workspace Rule\nHOT_RELOAD_RULE\n").unwrap();
    std::fs::write(
        ws.join(".agents/skills/demo-skill/SKILL.md"),
        "---\nname: demo-skill\ndescription: Demo hot skill\n---\nDEMO_BODY\n",
    )
    .unwrap();

    let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins([
        "compaction",
        "tools-coding",
        "tool-search",
        "codemode",
        "mcp",
        "skills",
        "instructions",
        "recall",
        "session-title",
        "subagent",
        "trajectory",
        "repeat-guard",
    ]);
    assert!(cfg.session_title.is_none());
    cfg.session_title = Some(mizar_agent_spine::SessionTitleConfig::default());

    let app = Context::new();
    let events: Arc<Mutex<Vec<(String, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let events_sub = Arc::clone(&events);
    let _sub = app
        .on(
            EVENT_SPINE_PLUGINS_CHANGE,
            move |ev: &SpinePluginsChangeEvent| {
                events_sub.lock().unwrap().push((ev.id.clone(), ev.enabled));
            },
        )
        .unwrap();

    app.plugin(AgentSpinePlugin, cfg).await.unwrap();

    // 注册一个探测工具，用于验证 repeat-guard 钩子的动态挂载/卸载
    let probe_calls = Arc::new(Mutex::new(0usize));
    let probe_calls_ref = Arc::clone(&probe_calls);
    app.plugin(
        PluginDef::sync("probe-tool", move |ctx, ()| {
            let tools = ctx.tools()?;
            let counter = Arc::clone(&probe_calls_ref);
            let def = ToolDefinition::sync_tool(
                "probe_echo",
                "Echo probe tool for lifecycle verification",
                json!({"type": "object", "properties": {"v": {"type": "string"}}}),
                move |args, _exec| {
                    *counter.lock().unwrap() += 1;
                    Ok::<_, mizar_tools::ToolError>(format!("probe:{args}"))
                },
            );
            ctx.effect("probe-tool.reg", move || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(def)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["tools"]),
        (),
    )
    .await
    .unwrap();

    wait_all_settled(&app).await;

    let ctrl = app.spine_plugins().unwrap();
    let tools = app.tools().unwrap();
    let prompt = app.system_prompt().unwrap();
    let sessions = app.sessions().unwrap();

    // 1. 启动阶段全部可选模块均已跳过，工具与可选服务均未挂载
    assert_eq!(ctrl.disabled_ids().len(), 12);
    for svc in [
        "compaction",
        "sessionTitle",
        "subagent",
        "mcp",
        "skills",
        "agentInstructions",
        "trajectory",
    ] {
        assert!(
            app.get::<CompactionRuntime>(svc).is_none()
                && app.get::<SessionTitleRuntime>(svc).is_none()
                && app.get::<SubagentRuntime>(svc).is_none()
                && app.get::<McpManager>(svc).is_none()
                && app.get::<SkillRegistry>(svc).is_none()
                && app.get::<AgentInstructions>(svc).is_none()
                && app.get::<TrajectoryRuntime>(svc).is_none(),
            "optional service {svc} should not be mounted while disabled at startup"
        );
    }
    let startup_tool_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert_eq!(
        startup_tool_names,
        HashSet::from(["probe_echo".to_string()])
    );
    assert!(!prompt.build().contains("Available tools:"));
    assert!(!prompt.build().contains("当前工作目录："));

    // 2. 热挂载 + 热卸载基础编程工具 (`tools-coding`)，含系统提示词提供者清理验证
    assert!(ctrl.set_enabled("tools-coding", true).await.unwrap());
    assert!(!ctrl.set_enabled("tools-coding", true).await.unwrap());
    let coding_tools: HashSet<String> = tools.list().into_iter().map(|t| t.name).collect();
    for expected in ["read", "bash", "edit", "write", "grep", "find"] {
        assert!(coding_tools.contains(expected), "missing {expected}");
    }
    assert!(prompt.build().contains("Available tools:"));
    assert!(!prompt.build().contains("当前工作目录："));
    assert!(ctrl.set_enabled("tools-coding", false).await.unwrap());
    let after_disable_coding: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert_eq!(
        after_disable_coding,
        HashSet::from(["probe_echo".to_string()])
    );
    assert!(!prompt.build().contains("Available tools:"));

    // 3. 热挂载 tool-search / codemode / mcp / skills / instructions / recall
    for id in [
        "tool-search",
        "codemode",
        "mcp",
        "skills",
        "instructions",
        "recall",
    ] {
        assert!(ctrl.set_enabled(id, true).await.unwrap(), "enable {id}");
    }
    assert!(app.get::<McpManager>("mcp").is_some());
    assert!(app.get::<SkillRegistry>("skills").is_some());
    assert!(app.get::<AgentInstructions>("agentInstructions").is_some());

    let enabled_tool_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert!(enabled_tool_names.contains("tool_search"));
    assert!(enabled_tool_names.contains("codemode"));
    assert!(enabled_tool_names.contains("skill"));
    assert!(enabled_tool_names.contains(RECALL_TOOL_NAME));

    // 触发一次 pre-step，验证 instructions 与 skills 监听器均正常向会话注入上下文
    let session = sessions.create(None);
    session.append(SessionEvent::user("hello"));
    if let AgentPreStepDecision::Enter { inputs } = app
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&session),
                turn: 1,
                step: 1,
                provider: Some("mock".into()),
                model: Some("mock-1".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await
    {
        for item in inputs {
            session.append(item.into_session_event());
        }
    }
    assert!(session.visible_log().iter().any(|e| matches!(
        &e.event,
        SessionEvent::ContextNotice { text, .. } if text.contains("HOT_RELOAD_RULE")
    )));
    assert!(session.visible_log().iter().any(|e| matches!(
        &e.event,
        SessionEvent::ContextNotice { text, .. } if text.contains("demo-skill")
    )));

    // 4. 热卸载 tool-search / codemode / mcp / skills / instructions / recall
    for id in [
        "tool-search",
        "codemode",
        "mcp",
        "skills",
        "instructions",
        "recall",
    ] {
        assert!(ctrl.set_enabled(id, false).await.unwrap(), "disable {id}");
    }
    assert!(app.get::<McpManager>("mcp").is_none());
    assert!(app.get::<SkillRegistry>("skills").is_none());
    assert!(app.get::<AgentInstructions>("agentInstructions").is_none());
    let after_unload_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert_eq!(
        after_unload_names,
        HashSet::from(["probe_echo".to_string()])
    );

    // 确认 instructions 与 skills 监听器已被卸载：新建会话不再收到对应 ContextNotice
    let session_after_unload = sessions.create(None);
    session_after_unload.append(SessionEvent::user("after unload"));
    if let AgentPreStepDecision::Enter { inputs } = app
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&session_after_unload),
                turn: 1,
                step: 1,
                provider: Some("mock".into()),
                model: Some("mock-1".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await
    {
        for item in inputs {
            session_after_unload.append(item.into_session_event());
        }
    }
    assert!(!session_after_unload.visible_log().iter().any(|e| matches!(
        &e.event,
        SessionEvent::ContextNotice { text, .. } if text.contains("HOT_RELOAD_RULE") || text.contains("demo-skill")
    )));

    // 再次热挂载 skills 与 instructions，验证可重复恢复
    assert!(ctrl.set_enabled("skills", true).await.unwrap());
    assert!(ctrl.set_enabled("instructions", true).await.unwrap());
    assert!(app.get::<SkillRegistry>("skills").is_some());
    assert!(app.get::<AgentInstructions>("agentInstructions").is_some());
    if let AgentPreStepDecision::Enter { inputs } = app
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(&session_after_unload),
                turn: 2,
                step: 1,
                provider: Some("mock".into()),
                model: Some("mock-1".into()),
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
            },
        )
        .await
    {
        for item in inputs {
            session_after_unload.append(item.into_session_event());
        }
    }
    assert!(session_after_unload.visible_log().iter().any(|e| matches!(
        &e.event,
        SessionEvent::ContextNotice { text, .. } if text.contains("HOT_RELOAD_RULE")
    )));
    assert!(session_after_unload.visible_log().iter().any(|e| matches!(
        &e.event,
        SessionEvent::ContextNotice { text, .. } if text.contains("demo-skill")
    )));

    // 5. compaction：易变配置热更新 + 热卸载 + 再热挂载保留最新配置（防状态脑裂）
    assert!(ctrl.set_enabled("compaction", true).await.unwrap());
    let comp = app.compaction().unwrap();
    comp.update_config(|c| {
        c.auto_compact = false;
    });
    assert!(ctrl.set_enabled("compaction", false).await.unwrap());
    assert!(app.get::<CompactionRuntime>("compaction").is_none());
    assert!(ctrl.set_enabled("compaction", true).await.unwrap());
    let comp_reloaded = app.compaction().unwrap();
    assert!(
        !comp_reloaded.config().auto_compact,
        "re-enabled compaction must preserve prior runtime config mutations"
    );

    // 5b. session-title：热挂载 + 易变配置热更新 + 热卸载 + 再热挂载保留最新配置
    assert!(ctrl.set_enabled("session-title", true).await.unwrap());
    let st = app.session_title().unwrap();
    st.update_config(|c| {
        c.enabled = false;
        c.model = Some("fast-title".into());
    });
    assert!(ctrl.set_enabled("session-title", false).await.unwrap());
    assert!(app.get::<SessionTitleRuntime>("sessionTitle").is_none());
    assert!(ctrl.set_enabled("session-title", true).await.unwrap());
    let st_reloaded = app.session_title().unwrap();
    assert!(!st_reloaded.config().enabled);
    assert_eq!(st_reloaded.config().model.as_deref(), Some("fast-title"));
    assert!(ctrl.set_enabled("session-title", false).await.unwrap());

    // 5d. subagent：热挂载 + 工具注册 + 易变配置热更新 + 热卸载 + 再热挂载保留最新配置
    assert!(ctrl.set_enabled("subagent", true).await.unwrap());
    assert!(
        tools
            .list_all()
            .into_iter()
            .any(|t| t.name == SUBAGENT_TOOL_NAME)
    );
    let sub = app.subagent().unwrap();
    sub.update_config(|c| {
        c.default_max_steps = 8;
        c.timeout_secs = 45;
    });
    assert!(ctrl.set_enabled("subagent", false).await.unwrap());
    assert!(app.get::<SubagentRuntime>("subagent").is_none());
    assert!(
        !tools
            .list_all()
            .into_iter()
            .any(|t| t.name == SUBAGENT_TOOL_NAME)
    );
    assert!(ctrl.set_enabled("subagent", true).await.unwrap());
    let sub_reloaded = app.subagent().unwrap();
    assert_eq!(sub_reloaded.config().default_max_steps, 8);
    assert_eq!(sub_reloaded.config().timeout_secs, 45);
    assert!(ctrl.set_enabled("subagent", false).await.unwrap());

    // 6. trajectory & repeat-guard：waterfall 钩子与服务卸载及恢复
    assert!(ctrl.set_enabled("trajectory", true).await.unwrap());
    assert!(ctrl.set_enabled("repeat-guard", true).await.unwrap());
    assert!(app.get::<TrajectoryRuntime>("trajectory").is_some());

    let guard_session = sessions.create(None);
    for _ in 0..2 {
        let _ = tools
            .execute(ToolExecution::new(
                "probe_echo",
                json!({"v": "same"}),
                Some(guard_session.id.clone()),
            ))
            .await;
    }
    let third = tools
        .execute(ToolExecution::new(
            "probe_echo",
            json!({"v": "same"}),
            Some(guard_session.id.clone()),
        ))
        .await;
    assert!(
        third
            .additional_contexts
            .iter()
            .any(|c| c.source == "repeat-tool-reminder"),
        "expected repeat-guard reminder while enabled, got: {:?}",
        third.additional_contexts
    );

    assert!(ctrl.set_enabled("repeat-guard", false).await.unwrap());
    assert!(ctrl.set_enabled("trajectory", false).await.unwrap());
    assert!(app.get::<TrajectoryRuntime>("trajectory").is_none());

    let after_guard_off = tools
        .execute(ToolExecution::new(
            "probe_echo",
            json!({"v": "same"}),
            Some(guard_session.id.clone()),
        ))
        .await;
    assert!(
        after_guard_off.additional_contexts.is_empty(),
        "repeat-guard waterfall hook must be cleanly removed after disable"
    );

    assert!(ctrl.set_enabled("repeat-guard", true).await.unwrap());
    let recorded = events.lock().unwrap();
    assert!(recorded.contains(&("tools-coding".to_string(), true)));
    assert!(recorded.contains(&("tools-coding".to_string(), false)));
    assert!(recorded.contains(&("repeat-guard".to_string(), false)));
}
