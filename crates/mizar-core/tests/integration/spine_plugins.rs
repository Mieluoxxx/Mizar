//! 集成测试：验证 `"spinePlugins"` 的 Gate 2 核心自保护、启动屏蔽、延迟 MCP 挂载与零销毁原子开关切换
//! （工具可见性、系统提示词、waterfall 钩子、服务启用门控与配置保留）。

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::channel::mpsc;
use mizar_core::agent_loop::{AgentPreStepDecision, AgentPreStepInput};
use mizar_core::spine::{
    AgentSpineConfig, MizarRuntime, RECALL_TOOL_NAME, SUBAGENT_TOOL_NAME, SpinePluginError,
};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, ModelInfo, ProviderInfo, StreamChunk,
};
use mizar_session::SessionEvent;
use mizar_tools::{ToolDefinition, ToolExecution};
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

    let runtime = MizarRuntime::build(cfg).await.unwrap();
    runtime
        .register_adapter("mock", Arc::new(SimpleReplyAdapter))
        .unwrap();

    let ctrl = &runtime.modules;
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
    let session = runtime.sessions.create(None);
    let answer = runtime
        .agent_loop
        .run(Arc::clone(&session), "ping", CancellationToken::new())
        .await
        .unwrap();
    assert!(answer.contains("core-loop-ok"), "unexpected: {answer}");
    runtime.shutdown().await;
}

#[tokio::test]
async fn optional_modules_startup_disable_and_runtime_toggle_cleanly() {
    let ws = temp_dir("toggle_lifecycle");
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
    cfg.session_title = Some(mizar_core::spine::SessionTitleConfig::default());

    let runtime = MizarRuntime::build(cfg).await.unwrap();

    // 注册一个探测工具，用于验证 repeat-guard 钩子的动态启用/禁用
    let probe_calls = Arc::new(Mutex::new(0usize));
    let probe_calls_ref = Arc::clone(&probe_calls);
    let def = ToolDefinition::sync_tool(
        "probe_echo",
        "Echo probe tool for lifecycle verification",
        json!({"type": "object", "properties": {"v": {"type": "string"}}}),
        move |args, _exec| {
            *probe_calls_ref.lock().unwrap() += 1;
            Ok::<_, mizar_tools::ToolError>(format!("probe:{args}"))
        },
    );
    runtime.push_guard(runtime.tools.register(def).unwrap());

    let ctrl = &runtime.modules;
    let tools = &runtime.tools;
    let prompt = &runtime.system_prompt;
    let sessions = &runtime.sessions;

    // 1. 启动阶段全部可选模块处于禁用状态：MCP 延迟未挂载，其余模块门控为 false 且工具被屏蔽
    assert_eq!(ctrl.disabled_ids().len(), 12);
    assert!(
        runtime.mcp().await.is_none(),
        "mcp must stay unmounted while disabled at startup"
    );
    assert!(!runtime.compaction.as_ref().unwrap().is_module_enabled());
    assert!(!runtime.session_title.as_ref().unwrap().is_module_enabled());
    assert!(!runtime.subagent.as_ref().unwrap().is_module_enabled());
    assert!(!runtime.skills.as_ref().unwrap().is_enabled());
    assert!(!runtime.instructions.as_ref().unwrap().is_enabled());
    assert!(!runtime.trajectory.as_ref().unwrap().is_enabled());

    let startup_tool_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert_eq!(
        startup_tool_names,
        HashSet::from(["probe_echo".to_string()])
    );
    assert!(
        tools
            .execute(ToolExecution::new(
                "read",
                json!({"path": "AGENTS.md"}),
                None
            ))
            .await
            .is_error
    );
    assert!(!prompt.build().contains("Available tools:"));
    assert!(!prompt.build().contains("Subagent delegation"));
    assert!(!prompt.build().contains("当前工作目录："));

    // 2. 启用与禁用基础编程工具 (`tools-coding`)，含 tools_guide 系统提示词切换验证
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

    // 3. 启用 tool-search / codemode / mcp / skills / instructions / recall
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
    assert!(runtime.mcp().await.is_some());
    assert!(runtime.skills.as_ref().unwrap().is_enabled());
    assert!(runtime.instructions.as_ref().unwrap().is_enabled());

    let enabled_tool_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert!(enabled_tool_names.contains("tool_search"));
    assert!(enabled_tool_names.contains("codemode"));
    assert!(enabled_tool_names.contains("skill"));
    assert!(enabled_tool_names.contains(RECALL_TOOL_NAME));
    // 0 个外部 MCP 工具时，tool_search 仍保持按需自动激活语义
    assert!(!tools.list().iter().any(|t| t.name == "tool_search"));

    // 触发一次 pre-step，验证 instructions 与 skills 监听器均正常向会话注入上下文
    let session = sessions.create(None);
    session.append(SessionEvent::user("hello"));
    if let AgentPreStepDecision::Enter { inputs } = runtime
        .agent_loop
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 1,
            step: 1,
            provider: Some("mock".into()),
            model: Some("mock-1".into()),
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
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

    // 4. 禁用 tool-search / codemode / mcp / skills / instructions / recall
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
    assert!(!runtime.skills.as_ref().unwrap().is_enabled());
    assert!(!runtime.instructions.as_ref().unwrap().is_enabled());
    let after_unload_names: HashSet<String> =
        tools.list_all().into_iter().map(|t| t.name).collect();
    assert_eq!(
        after_unload_names,
        HashSet::from(["probe_echo".to_string()])
    );

    // 确认 instructions 与 skills 已停止注入：新建会话不再收到对应 ContextNotice
    let session_after_unload = sessions.create(None);
    session_after_unload.append(SessionEvent::user("after disable"));
    if let AgentPreStepDecision::Enter { inputs } = runtime
        .agent_loop
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(&session_after_unload),
            turn: 1,
            step: 1,
            provider: Some("mock".into()),
            model: Some("mock-1".into()),
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
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

    // 再次启用 skills 与 instructions，验证可重复恢复
    assert!(ctrl.set_enabled("skills", true).await.unwrap());
    assert!(ctrl.set_enabled("instructions", true).await.unwrap());
    assert!(runtime.skills.as_ref().unwrap().is_enabled());
    assert!(runtime.instructions.as_ref().unwrap().is_enabled());
    if let AgentPreStepDecision::Enter { inputs } = runtime
        .agent_loop
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(&session_after_unload),
            turn: 2,
            step: 1,
            provider: Some("mock".into()),
            model: Some("mock-1".into()),
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
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

    // 5. compaction：易变配置热更新 + 禁用 + 再启用保留最新配置
    assert!(ctrl.set_enabled("compaction", true).await.unwrap());
    let comp = runtime.compaction.as_ref().unwrap();
    assert!(comp.is_module_enabled());
    comp.update_config(|c| {
        c.auto_compact = false;
    });
    assert!(ctrl.set_enabled("compaction", false).await.unwrap());
    assert!(!comp.is_module_enabled());
    assert!(ctrl.set_enabled("compaction", true).await.unwrap());
    assert!(comp.is_module_enabled());
    assert!(
        !comp.config().auto_compact,
        "re-enabled compaction must preserve prior runtime config mutations"
    );

    // 5b. session-title：启用 + 易变配置热更新 + 禁用 + 再启用保留最新配置
    assert!(ctrl.set_enabled("session-title", true).await.unwrap());
    let st = runtime.session_title.as_ref().unwrap();
    assert!(st.is_module_enabled());
    st.update_config(|c| {
        c.enabled = false;
        c.model = Some("fast-title".into());
    });
    assert!(ctrl.set_enabled("session-title", false).await.unwrap());
    assert!(!st.is_module_enabled());
    assert!(ctrl.set_enabled("session-title", true).await.unwrap());
    assert!(st.is_module_enabled());
    assert!(!st.config().enabled);
    assert_eq!(st.config().model.as_deref(), Some("fast-title"));
    assert!(ctrl.set_enabled("session-title", false).await.unwrap());

    // 5d. subagent：启用 + 工具可见 + 易变配置热更新 + 禁用 + 再启用保留最新配置
    assert!(ctrl.set_enabled("subagent", true).await.unwrap());
    assert!(
        tools
            .list_all()
            .into_iter()
            .any(|t| t.name == SUBAGENT_TOOL_NAME)
    );
    let sub = runtime.subagent.as_ref().unwrap();
    assert!(sub.is_module_enabled());
    sub.update_config(|c| {
        c.read_only_soft_budget = 8;
        c.timeout_secs = 45;
    });
    assert!(ctrl.set_enabled("subagent", false).await.unwrap());
    assert!(!sub.is_module_enabled());
    assert!(
        !tools
            .list_all()
            .into_iter()
            .any(|t| t.name == SUBAGENT_TOOL_NAME)
    );
    assert!(ctrl.set_enabled("subagent", true).await.unwrap());
    assert!(sub.is_module_enabled());
    assert_eq!(sub.config().read_only_soft_budget, 8);
    assert_eq!(sub.config().timeout_secs, 45);
    assert!(ctrl.set_enabled("subagent", false).await.unwrap());

    // 6. trajectory & repeat-guard：waterfall 钩子与服务禁用及恢复
    assert!(ctrl.set_enabled("trajectory", true).await.unwrap());
    assert!(ctrl.set_enabled("repeat-guard", true).await.unwrap());
    assert!(runtime.trajectory.as_ref().unwrap().is_enabled());

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
    assert!(!runtime.trajectory.as_ref().unwrap().is_enabled());

    let after_guard_off = tools
        .execute(ToolExecution::new(
            "probe_echo",
            json!({"v": "same"}),
            Some(guard_session.id.clone()),
        ))
        .await;
    assert!(
        after_guard_off.additional_contexts.is_empty(),
        "repeat-guard waterfall hook must pass through cleanly after disable"
    );

    assert!(ctrl.set_enabled("repeat-guard", true).await.unwrap());
    assert!(ctrl.is_enabled("repeat-guard"));
    runtime.shutdown().await;
}
