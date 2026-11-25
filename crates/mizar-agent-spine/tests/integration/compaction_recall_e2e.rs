use std::sync::{Arc, Mutex};

use cordis::{Context, PluginDef};
use futures::channel::mpsc;
use mizar_agent_loop::{AgentLoopContextExt, AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_agent_spine::{
    AgentSpineConfig, AgentSpinePlugin, CompactionContextExt, CompactionOutcome,
    ManualCompactOptions, RECALL_TOOL_NAME,
};
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, ModelInfo,
    ProviderInfo, StreamChunk,
};
use mizar_session::{SessionContextExt, SessionEvent};
use mizar_system_prompt::SystemPromptContextExt;
use mizar_tools::{ToolExecution, ToolsContextExt};
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

    let app = Context::new();
    app.plugin(AgentSpinePlugin, cfg).await.unwrap();

    let seen_requests = Arc::new(Mutex::new(Vec::new()));
    let adapter = Arc::new(SummaryMockAdapter {
        seen_requests: Arc::clone(&seen_requests),
    });
    app.plugin(
        PluginDef::sync("spine-mock-adapter", move |ctx, ()| {
            let llm = ctx.llm()?;
            let a = Arc::clone(&adapter);
            ctx.effect("spine-mock-adapter.reg", move || {
                Ok::<_, LlmError>(Some(llm.register_adapter("spine-mock", a)?))
            })?;
            Ok::<_, cordis::CordisError>(())
        })
        .inject(["llm"]),
        (),
    )
    .await
    .unwrap();

    // 等核心插件全部 settled
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }

    let sessions = app.sessions().unwrap();
    let tools = app.tools().unwrap();
    let compaction = app.compaction().unwrap();
    let prompt = app.system_prompt().unwrap();
    let _agent_loop = app.agent_loop().unwrap();

    let session = sessions.create(Some("spine-e2e-1".into()));

    // 1. recall 工具作为单一静态常驻工具在压缩前已声明，且 AGENTS.md 已静态注入系统提示词
    let declared_before: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(declared_before.contains(&RECALL_TOOL_NAME.to_string()));
    assert!(
        prompt
            .build_for_session(Some(&session.id))
            .contains("Always preserve determinism"),
        "expected AGENTS.md in static system prompt"
    );

    // 2. 首步 pre-step 不会改动静态工具列表，也不会重复注入 AGENTS.md
    session.append(SessionEvent::user("早期关键约束：暗号为 星尘压缩密钥_2026"));
    let _ = app
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(&session),
                turn: 1,
                step: 1,
                provider: Some("spine-mock".into()),
                model: Some("m-spine".into()),
                cancel: CancellationToken::new(),
            },
        )
        .await;

    session.append(SessionEvent::assistant_text("已收到早期约束"));
    session.append(SessionEvent::user("继续执行下一阶段任务"));

    // 3. 手动压缩：产出检查点且工具列表保持冻结不变（KV Cache 前缀稳定）
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

    let declared_after: Vec<String> = tools
        .list_declared(Some(&session.id))
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(declared_before, declared_after);

    // 4. 压后系统提示词依然稳定保留 AGENTS.md
    assert!(
        prompt
            .build_for_session(Some(&session.id))
            .contains("Always preserve determinism"),
        "expected AGENTS.md to stay in static system prompt after compaction"
    );

    // 5. 通过单一 recall 工具（search + read）穿透压缩边界与 SQLite FTS5 找回早期中文暗号
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

    app.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&ws);
}
