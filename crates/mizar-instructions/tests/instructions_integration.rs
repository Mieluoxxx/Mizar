use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, Service};
use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_instructions::{AgentInstructions, InstructionsConfig, InstructionsContextExt};
use mizar_session::{SessionContextExt, SessionEvent, SessionStore};
use mizar_tools::{ToolExecution, ToolRuntime, ToolsContextExt};
use mizar_tools_coding::create_read_tool;
use tokio_util::sync::CancellationToken;

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "mizar_instr_int_{name}_{}_{nanos}",
        std::process::id()
    ))
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn emit_pre_step(ctx: &Context, session: &Arc<mizar_session::Session>, step: usize) {
    let _ = ctx
        .serial::<AgentPreStepEvent, ()>(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepEvent {
                session: Arc::clone(session),
                turn: 1,
                step,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
            },
        )
        .await;
}

#[tokio::test]
async fn discovers_global_ancestor_workspace_subdir_hot_reload_and_compaction() {
    let root = temp_dir("lifecycle");
    let _guard = DirGuard(root.clone());

    let mizar_home = root.join("mizar-home");
    let agents_home = root.join("agents-home");
    let repo = root.join("repo");
    let ws = repo.join("packages/core");
    let sub = ws.join("src");
    std::fs::create_dir_all(&mizar_home).unwrap();
    std::fs::create_dir_all(&agents_home).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(&sub).unwrap();

    // 全局指令：~/.mizar/AGENTS.md 与 ~/.agents/AGENTS.md（内容不同，应同时加载）
    std::fs::write(mizar_home.join("AGENTS.md"), "global mizar rule").unwrap();
    std::fs::write(agents_home.join("AGENTS.md"), "global agents rule").unwrap();
    // 项目根：AGENTS.md 与 CLAUDE.md
    std::fs::write(repo.join("AGENTS.md"), "repo agents rule").unwrap();
    std::fs::write(repo.join("CLAUDE.md"), "repo claude rule").unwrap();
    // 工作区：AGENTS.md 与 CLAUDE.md + CLAUDE.local.md
    std::fs::write(ws.join("AGENTS.md"), "ws agents rule").unwrap();
    std::fs::write(ws.join("CLAUDE.md"), "ws claude rule").unwrap();
    std::fs::write(ws.join("CLAUDE.local.md"), "ws local rule").unwrap();
    // 子目录（初始不应加载，需 read 触达后再加载）
    std::fs::write(sub.join("AGENTS.md"), "sub agents rule").unwrap();
    std::fs::write(sub.join("CLAUDE.md"), "sub claude rule").unwrap();
    std::fs::write(sub.join("lib.rs"), "pub fn hi() {}\n").unwrap();

    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();

    let tools = app.tools().unwrap();
    let _read_cleanup = tools.register(create_read_tool(ws.clone(), false)).unwrap();

    let mut cfg = InstructionsConfig::new(&ws);
    cfg.mizar_home = Some(mizar_home.clone());
    cfg.agents_home = Some(agents_home.clone());
    app.plugin(AgentInstructions::plugin(), cfg).await.unwrap();

    let session = app.sessions().unwrap().create(None);
    session.append(SessionEvent::User {
        text: "hello".into(),
    });

    // 1. 第 1 步：注入全量基线（不含未触及的 src/）
    emit_pre_step(&app, &session, 1).await;

    let msgs1 = session.derive_messages();
    assert_eq!(msgs1.len(), 2);
    let baseline_text = match &msgs1[1] {
        mizar_llm::Message::User { content } => content.clone(),
        other => panic!("expected user ContextNotice, got {other:?}"),
    };
    assert!(baseline_text.contains("<system-reminder>"));
    assert!(baseline_text.contains("global mizar rule"));
    assert!(baseline_text.contains("global agents rule"));
    assert!(baseline_text.contains("repo agents rule"));
    assert!(baseline_text.contains("repo claude rule"));
    assert!(baseline_text.contains("ws agents rule"));
    assert!(baseline_text.contains("ws claude rule"));
    assert!(baseline_text.contains("ws local rule"));
    assert!(!baseline_text.contains("sub agents rule"));

    // 2. 第 2 步无变更也不触达子目录 -> 零新增消息
    emit_pre_step(&app, &session, 2).await;
    assert_eq!(session.derive_messages().len(), 2);

    // 3. 通过 ToolRuntime 执行 read("src/lib.rs") -> 触发 src/AGENTS.md 与 src/CLAUDE.md 增量加载
    let res = tools
        .execute(ToolExecution::new(
            "read",
            serde_json::json!({"path": "src/lib.rs"}),
            Some(session.id.clone()),
        ))
        .await;
    assert!(!res.is_error, "{}", res.content);

    emit_pre_step(&app, &session, 3).await;
    let msgs3 = session.derive_messages();
    assert_eq!(msgs3.len(), 3);
    let delta_sub = match &msgs3[2] {
        mizar_llm::Message::User { content } => content.clone(),
        other => panic!("expected delta ContextNotice, got {other:?}"),
    };
    assert!(delta_sub.contains("Additional instructions from: src/AGENTS.md"));
    assert!(delta_sub.contains("sub agents rule"));
    assert!(delta_sub.contains("Additional instructions from: src/CLAUDE.md"));
    assert!(delta_sub.contains("sub claude rule"));

    // 4. 修改 ws/CLAUDE.md 与删除 ws/CLAUDE.local.md -> 触发 Updated 与 Removed
    std::fs::write(ws.join("CLAUDE.md"), "ws claude rule v2").unwrap();
    std::fs::remove_file(ws.join("CLAUDE.local.md")).unwrap();
    emit_pre_step(&app, &session, 4).await;
    let msgs4 = session.derive_messages();
    assert_eq!(msgs4.len(), 4);
    let delta_mod = match &msgs4[3] {
        mizar_llm::Message::User { content } => content.clone(),
        other => panic!("expected update ContextNotice, got {other:?}"),
    };
    assert!(delta_mod.contains("Updated instructions from: CLAUDE.md"));
    assert!(delta_mod.contains("ws claude rule v2"));
    assert!(delta_mod.contains("Instructions removed: CLAUDE.local.md"));

    // 5. Compaction 把旧基线折叠出可见窗口 -> 下一步应自动重建全量基线
    session.append(SessionEvent::User {
        text: "after compact".into(),
    });
    session.compact("早期摘要", 1, None);
    emit_pre_step(&app, &session, 5).await;
    let msgs5 = session.derive_messages();
    let last_msg = match msgs5.last().unwrap() {
        mizar_llm::Message::User { content } => content.clone(),
        other => panic!("expected rebuilt baseline, got {other:?}"),
    };
    assert!(last_msg.contains("Instructions from:"));
    assert!(last_msg.contains("ws claude rule v2"));
    assert!(last_msg.contains("sub agents rule"));
    assert!(!last_msg.contains("ws local rule"));

    let report = app
        .agent_instructions()
        .unwrap()
        .session_report(&session.id)
        .unwrap();
    assert!(report.enabled);
    assert!(report.scopes.iter().any(|s| s.display_path == "CLAUDE.md"));
    assert!(
        report
            .scopes
            .iter()
            .any(|s| s.display_path == "src/AGENTS.md")
    );

    app.fiber().dispose().await;
}

#[tokio::test]
async fn disabled_instructions_skips_injection() {
    let root = temp_dir("disabled");
    let _guard = DirGuard(root.clone());
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("AGENTS.md"), "should not load").unwrap();

    let app = Context::new();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    let mut cfg = InstructionsConfig::new(&root);
    cfg.enabled = false;
    cfg.include_default_global = false;
    app.plugin(AgentInstructions::plugin(), cfg).await.unwrap();

    let session = app.sessions().unwrap().create(None);
    session.append(SessionEvent::User { text: "hi".into() });
    emit_pre_step(&app, &session, 1).await;

    assert_eq!(session.derive_messages().len(), 1);
    assert!(!app.agent_instructions().unwrap().is_enabled());
    assert!(
        !app.agent_instructions()
            .unwrap()
            .probe_report(Some(&session.id))
            .enabled
    );

    app.fiber().dispose().await;
}
