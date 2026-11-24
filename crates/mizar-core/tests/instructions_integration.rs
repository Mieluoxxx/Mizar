use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_core::agent_loop::{AgentLoop, AgentLoopConfig, AgentPreStepDecision, AgentPreStepInput};
use mizar_core::instructions::{AgentInstructions, InstructionScopeKind, InstructionsConfig};
use mizar_core::system_prompt::{SystemPrompt, SystemPromptConfig};
use mizar_llm::LlmRuntime;
use mizar_session::{SessionEvent, SessionStore};
use mizar_tools::coding::create_read_tool;
use mizar_tools::{ToolExecution, ToolRuntime};
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

async fn emit_pre_step(agent: &AgentLoop, session: &Arc<mizar_session::Session>, step: usize) {
    if let AgentPreStepDecision::Enter { inputs } = agent
        .run_pre_step(AgentPreStepInput {
            session: Arc::clone(session),
            turn: 1,
            step,
            provider: None,
            model: None,
            cancel: CancellationToken::new(),
            inputs: Vec::new(),
        })
        .await
    {
        for item in inputs {
            session.append(item.into_session_event());
        }
    }
}

#[tokio::test]
async fn discovers_global_ancestor_workspace_into_system_prompt_without_subdirs_or_hot_reload() {
    let root = temp_dir("static_lifecycle");
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
    // 子目录（方案 B 静态加载下永不注入，即便 read 触达亦不加载）
    std::fs::write(sub.join("AGENTS.md"), "sub agents rule").unwrap();
    std::fs::write(sub.join("CLAUDE.md"), "sub claude rule").unwrap();
    std::fs::write(sub.join("lib.rs"), "pub fn hi() {}\n").unwrap();

    let sessions = SessionStore::new();
    let tools = ToolRuntime::new();
    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let agent = AgentLoop::new(
        LlmRuntime::new(),
        Arc::clone(&tools),
        Arc::clone(&sessions),
        Arc::clone(&sp),
        AgentLoopConfig::default(),
    );
    let _read_cleanup = tools.register(create_read_tool(ws.clone(), false)).unwrap();

    let mut cfg = InstructionsConfig::new(&ws);
    cfg.mizar_home = Some(mizar_home.clone());
    cfg.agents_home = Some(agents_home.clone());
    let (instr, _instr_guard) = AgentInstructions::install(&sp, cfg);

    // 1. 启动期直接注入 SystemPrompt 静态段，全局在前、祖先次之、工作区在后，且不含子目录规则
    let prompt_initial = sp.build();
    for expected in [
        "<project_instructions",
        "global mizar rule",
        "global agents rule",
        "repo agents rule",
        "repo claude rule",
        "ws agents rule",
        "ws claude rule",
        "ws local rule",
    ] {
        assert!(
            prompt_initial.contains(expected),
            "missing `{expected}` in:\n{prompt_initial}"
        );
    }
    assert!(!prompt_initial.contains("sub agents rule"));
    assert!(!prompt_initial.contains("sub claude rule"));
    assert!(
        prompt_initial.find("global mizar rule").unwrap()
            < prompt_initial.find("repo agents rule").unwrap()
    );
    assert!(
        prompt_initial.find("repo claude rule").unwrap()
            < prompt_initial.find("ws agents rule").unwrap()
    );

    // 2. 运行 pre_step、读取子目录文件、盘上改写工作区指令与触发 compaction，均不产生 ContextNotice 且前缀字节不变
    let session = sessions.create(None);
    session.append(SessionEvent::user("hello"));
    emit_pre_step(&agent, &session, 1).await;

    let res = tools
        .execute(ToolExecution::new(
            "read",
            serde_json::json!({"path": "src/lib.rs"}),
            Some(session.id.clone()),
        ))
        .await;
    assert!(!res.is_error, "{}", res.content);

    std::fs::write(ws.join("CLAUDE.md"), "ws claude rule v2").unwrap();
    std::fs::remove_file(ws.join("CLAUDE.local.md")).unwrap();
    emit_pre_step(&agent, &session, 2).await;

    session.append(SessionEvent::user("after compact"));
    session.compact("早期摘要", 1, None);
    emit_pre_step(&agent, &session, 3).await;

    assert_eq!(
        sp.build(),
        prompt_initial,
        "system prompt section must remain byte-identical across steps and compaction"
    );
    for msg in session.derive_messages() {
        let text = msg.text_content();
        assert!(!text.contains("ws agents rule"));
        assert!(!text.contains("sub agents rule"));
        assert!(!text.contains("ws claude rule v2"));
    }

    let report = instr.report();
    assert!(report.enabled);
    assert!(report.scopes.iter().any(|s| s.display_path == "CLAUDE.md"));
    assert!(
        !report
            .scopes
            .iter()
            .any(|s| s.display_path == "src/AGENTS.md")
    );
}

#[tokio::test]
async fn disabled_instructions_skips_injection() {
    let root = temp_dir("disabled");
    let _guard = DirGuard(root.clone());
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("AGENTS.md"), "should not load").unwrap();

    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let mut cfg = InstructionsConfig::new(&root);
    cfg.enabled = false;
    cfg.include_default_global = false;
    let (instr, _instr_guard) = AgentInstructions::install(&sp, cfg);

    assert!(!sp.build().contains("should not load"));
    assert!(!instr.is_enabled());
    assert!(!instr.report().enabled);

    // 显式通过 --no-instructions / settings 禁用时，运行期模块开关无法反向开启
    instr.set_enabled(true);
    assert!(!instr.is_enabled());
    assert!(!sp.build().contains("should not load"));
}

#[test]
fn runtime_toggle_and_budget_truncation_preserve_expected_report() {
    let root = temp_dir("toggle_budget");
    let _guard = DirGuard(root.clone());
    let ws = root.join("repo/pkg");
    std::fs::create_dir_all(root.join("repo/.git")).unwrap();
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(root.join("repo/AGENTS.md"), "A".repeat(250)).unwrap();
    std::fs::write(ws.join("AGENTS.md"), "W".repeat(300)).unwrap();

    let sp = SystemPrompt::new(SystemPromptConfig::default());
    let mut cfg = InstructionsConfig::new(&ws);
    cfg.include_default_global = false;
    cfg.max_bytes = 350;
    let (instr, guard) = AgentInstructions::install(&sp, cfg);

    let report = instr.report();
    assert!(report.enabled);
    assert_eq!(report.scopes.len(), 2);
    assert_eq!(
        report.scopes[0].scope_kind,
        InstructionScopeKind::ProjectAncestor
    );
    assert!(report.scopes[0].omitted);
    assert_eq!(report.scopes[1].scope_kind, InstructionScopeKind::Workspace);
    assert!(report.scopes[1].truncated);

    let built = sp.build();
    assert!(built.contains("Omitted 1 ancestor instruction"));

    instr.set_enabled(false);
    assert!(!sp.build().contains("<project_instructions"));
    assert!(!instr.report().enabled);

    instr.set_enabled(true);
    assert_eq!(sp.build(), built);

    drop(guard);
    assert!(!sp.build().contains("<project_instructions"));
}
