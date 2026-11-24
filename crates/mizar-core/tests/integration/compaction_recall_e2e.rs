use std::sync::{Arc, Mutex};

use futures::channel::mpsc;
use mizar_core::agent_loop::{AgentPreStepDecision, AgentPreStepInput};
use mizar_core::spine::{
    AgentSpineConfig, AgentToolsMode, CompactionOutcome, ManualCompactOptions, MizarRuntime,
    RECALL_TOOL_NAME,
};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, ModelInfo, ProviderInfo, StreamChunk,
};
use mizar_session::SessionEvent;
use mizar_tools::{ToolDefinition, ToolExecution, ToolExposure, ToolNamespace};
use serde_json::json;
use tokio_util::sync::CancellationToken;

struct SummaryMockAdapter {
    seen_requests: Arc<Mutex<Vec<GenerateOptions>>>,
}

impl LlmAdapter for SummaryMockAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: "spine-mock".into(),
            name: "Spine Mock".into(),
            protocol: "mock".into(),
            models: vec![ModelInfo {
                id: "m-spine".into(),
                name: "Spine Test".into(),
                context_window: Some(64_000),
                max_tokens: Some(4_096),
                reasoning_efforts: vec![],
                input: Vec::new(),
                ..Default::default()
            }],
        })
    }

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        self.seen_requests.lock().unwrap().push(options);
        let (tx, rx) = mpsc::unbounded();
        tokio::spawn(async move {
            let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                index: 0,
                text: "## Goal & Scope\n- Verify full spine compaction + recall\n## Verified Facts & Decisions\n- AGENTS.md loaded and compacted".into(),
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
async fn full_spine_compaction_reinjects_instructions_and_activates_sqlite_recall() {
    let ws = std::env::temp_dir().join(format!("mizar_spine_comp_rec_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(
        ws.join("AGENTS.md"),
        "# Workspace Rules\nAlways preserve determinism.\n",
    )
    .unwrap();
    let db_path = ws.join("sessions.sqlite");

    let mut cfg = AgentSpineConfig::coding(&ws).with_session_db(&db_path);
    cfg.agent.provider = Some("spine-mock".into());
    cfg.agent.model = Some("m-spine".into());
    cfg.subagent = None;
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
            mcp.background_connect = false;
        }
        skills.include_user_roots = false;
        skills.agents_home = Some(ws.join("isolated-agents"));
        instructions.include_default_global = false;
    }

    let runtime = MizarRuntime::build(cfg).await.unwrap();

    let seen_requests = Arc::new(Mutex::new(Vec::new()));
    let adapter = Arc::new(SummaryMockAdapter {
        seen_requests: Arc::clone(&seen_requests),
    });
    runtime.register_adapter("spine-mock", adapter).unwrap();

    let sessions = &runtime.sessions;
    let tools = &runtime.tools;
    let compaction = runtime.compaction.as_ref().unwrap();
    let agent_loop = &runtime.agent_loop;

    let session = sessions.create(Some("spine-e2e-1".into()));

    // 1. 0-MCP / 0-Skill 无头模式静态冻结精确 7 个工具，压缩前后声明集完全不变
    let mut declared_before: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    declared_before.sort();
    assert_eq!(
        declared_before,
        vec!["bash", "edit", "find", "grep", "read", "recall", "write"]
    );

    // 2. 启动静态注入应使系统提示词包含 AGENTS.md
    assert!(
        runtime
            .system_prompt
            .build_for_session(Some(&session.id))
            .contains("Always preserve determinism"),
        "expected AGENTS.md in system prompt before compaction"
    );
    session.append(SessionEvent::user("早期关键约束：暗号为 星尘压缩密钥_2026"));
    if let AgentPreStepDecision::Enter { inputs } = agent_loop
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 1,
            step: 1,
            provider: Some("spine-mock".into()),
            model: Some("m-spine".into()),
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
        .await
    {
        for item in inputs {
            session.append(item.into_session_event());
        }
    }

    session.append(SessionEvent::assistant_text("已收到早期约束"));
    session.append(SessionEvent::user("继续执行下一阶段任务"));

    // 3. 手动压缩：应产出检查点并自动激活 recall 工具
    let report = compaction
        .compact_now(
            Arc::clone(&session),
            ManualCompactOptions {
                provider: Some("spine-mock".into()),
                model: Some("m-spine".into()),
                focus: None,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(report.outcome, CompactionOutcome::Compacted);
    session.flush_async().await.unwrap();

    let mut declared_after: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    declared_after.sort();
    assert_eq!(
        declared_after, declared_before,
        "tools list must stay frozen across compaction"
    );

    // 4. 压后下一步 pre-step：系统提示词静态段仍保留 AGENTS.md
    if let AgentPreStepDecision::Enter { inputs } = agent_loop
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(&session),
            turn: 2,
            step: 1,
            provider: Some("spine-mock".into()),
            model: Some("m-spine".into()),
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
        .await
    {
        for item in inputs {
            session.append(item.into_session_event());
        }
    }
    assert!(
        runtime
            .system_prompt
            .build_for_session(Some(&session.id))
            .contains("Always preserve determinism"),
        "expected AGENTS.md to remain in system prompt after compaction"
    );

    // 5. 通过单一静态 recall 穿透压缩边界与 SQLite FTS5 找回早期中文暗号
    let search_res = tools
        .execute(ToolExecution::new(
            RECALL_TOOL_NAME,
            json!({ "query": "星尘压缩密钥", "scope": "current" }),
            Some(session.id.clone()),
        ))
        .await;
    assert!(!search_res.is_error, "{}", search_res.content);
    assert!(
        search_res.content.contains("seq #1"),
        "{}",
        search_res.content
    );

    let read_res = tools
        .execute(ToolExecution::new(
            RECALL_TOOL_NAME,
            json!({ "seq": 1 }),
            Some(session.id.clone()),
        ))
        .await;
    assert!(!read_res.is_error, "{}", read_res.content);
    assert!(read_res.content.contains("星尘压缩密钥_2026"));

    runtime.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
}

#[tokio::test]
async fn headless_zero_mcp_zero_skill_freezes_seven_tools_and_codemode_activates_on_demand() {
    let ws = std::env::temp_dir().join(format!("mizar_spine_7tools_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();

    let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["session-title"]);
    cfg.subagent = None;
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
            mcp.background_connect = false;
        }
        skills.include_user_roots = false;
        skills.agents_home = Some(ws.join("isolated-agents"));
        instructions.include_default_global = false;
    }

    let runtime = MizarRuntime::build(cfg).await.unwrap();

    let sessions = &runtime.sessions;
    let tools = &runtime.tools;
    let session = sessions.create(Some("seven-tools-1".into()));

    let mut initial_names: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    initial_names.sort();
    assert_eq!(
        initial_names,
        vec!["bash", "edit", "find", "grep", "read", "recall", "write"]
    );
    assert!(!initial_names.contains(&"codemode".to_string()));
    assert!(!initial_names.contains(&"tool_search".to_string()));
    assert!(!initial_names.contains(&"skill".to_string()));

    // 接入外部 Codemode MCP 工具后，codemode 与 tool_search 自动激活
    let _mcp_guard = tools
        .register(
            ToolDefinition::sync_tool(
                "mcp__demo__ping",
                "Ping demo server",
                json!({"type": "object", "properties": {}}),
                |_, _| Ok::<_, mizar_tools::ToolError>("pong".into()),
            )
            .with_exposure(ToolExposure::Codemode)
            .with_namespace(ToolNamespace::new("mcp__demo")),
        )
        .unwrap();

    let active_names: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    #[cfg(feature = "codemode")]
    assert!(active_names.contains(&"codemode".to_string()));
    assert!(active_names.contains(&"tool_search".to_string()));

    runtime.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
}
