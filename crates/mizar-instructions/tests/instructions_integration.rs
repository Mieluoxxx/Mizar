use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{Context, Service};
use mizar_instructions::{
    AGENT_INSTRUCTIONS_SECTION_ID, AgentInstructions, InstructionScopeKind, InstructionsConfig,
    InstructionsContextExt,
};
use mizar_system_prompt::{
    DEFAULT_RUNTIME_PROTOCOL, PromptSection, SystemPrompt, SystemPromptConfig,
    SystemPromptContextExt,
};

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

#[tokio::test]
async fn discovers_static_hierarchy_injects_system_prompt_and_cleans_up_on_unload() {
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
    // 子目录（静态启动方案不加载未配置的子目录文件）
    std::fs::write(sub.join("AGENTS.md"), "sub agents rule").unwrap();

    let app = Context::new();
    app.plugin(
        SystemPrompt::plugin(),
        SystemPromptConfig {
            persona: None,
            runtime_protocol: Some(DEFAULT_RUNTIME_PROTOCOL.to_string()),
        },
    )
    .await
    .unwrap();
    let sp = app.system_prompt().unwrap();
    let _tools_guard = app
        .effect("test.tools", || {
            Ok::<_, cordis::CordisError>(Some(sp.section(PromptSection::new(
                "coding_tools",
                20,
                || Some("TOOLS_GUIDE_PLACEHOLDER".to_string()),
            ))))
        })
        .unwrap();

    let mut cfg = InstructionsConfig::new(&ws);
    cfg.mizar_home = Some(mizar_home);
    cfg.agents_home = Some(agents_home);
    let inst_fiber = app.plugin(AgentInstructions::plugin(), cfg);
    inst_fiber.wait_settled().await.unwrap();

    let prompt = sp.build();
    assert!(prompt.contains("global mizar rule"), "prompt:\n{prompt}");
    assert!(prompt.contains("global agents rule"));
    assert!(prompt.contains("repo agents rule"));
    assert!(prompt.contains("repo claude rule"));
    assert!(prompt.contains("ws agents rule"));
    assert!(prompt.contains("ws claude rule"));
    assert!(prompt.contains("ws local rule"));
    assert!(!prompt.contains("sub agents rule"));

    // 验证四段式静态顺序：persona(0) < coding_tools(20) < agent_instructions(30) < runtime_protocol(40)
    let idx_persona = prompt.find("Engineering principles:").unwrap();
    let idx_tools = prompt.find("TOOLS_GUIDE_PLACEHOLDER").unwrap();
    let idx_inst = prompt
        .find("Project-specific instructions and guidelines:")
        .unwrap();
    let idx_proto = prompt.find("Runtime context blocks").unwrap();
    assert!(
        idx_persona < idx_tools && idx_tools < idx_inst && idx_inst < idx_proto,
        "unexpected section order in prompt:\n{prompt}"
    );

    let svc = app.agent_instructions().unwrap();
    let report = svc.report();
    assert!(report.enabled);
    assert_eq!(report.scopes.len(), 7);
    assert_eq!(
        report.scopes[0].scope_kind,
        InstructionScopeKind::UserGlobal
    );
    assert_eq!(
        report.scopes[2].scope_kind,
        InstructionScopeKind::ProjectAncestor
    );
    assert_eq!(report.scopes[6].scope_kind, InstructionScopeKind::Workspace);

    // 开关切换与卸载清理验证
    svc.set_enabled(false);
    assert!(!sp.build().contains("ws local rule"));
    assert!(!svc.report().enabled);

    svc.set_enabled(true);
    assert!(sp.build().contains("ws local rule"));

    // 启动后磁盘内容修改不会破坏静态系统前缀（仅启动期读一次）
    std::fs::write(ws.join("AGENTS.md"), "ws agents rule mutated").unwrap();
    assert!(!sp.build().contains("ws agents rule mutated"));

    inst_fiber.dispose().await;
    assert!(!sp.build().contains("ws local rule"));
    assert!(!sp.build().contains(AGENT_INSTRUCTIONS_SECTION_ID));

    app.fiber().dispose().await;
}

#[tokio::test]
async fn injects_section_when_system_prompt_mounts_after_instructions_and_skips_when_disabled() {
    let root = temp_dir("late_mount");
    let _guard = DirGuard(root.clone());
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("AGENTS.md"), "LATE_MOUNT_RULE").unwrap();

    // 1. 先挂载 AgentInstructions，再挂载 SystemPrompt
    let app = Context::new();
    let mut cfg = InstructionsConfig::new(&root);
    cfg.include_default_global = false;
    let inst = app.plugin(AgentInstructions::plugin(), cfg);
    inst.wait_settled().await.unwrap();
    assert!(app.get::<SystemPrompt>("systemPrompt").is_none());

    app.plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    for rt in app.registry().values() {
        for f in rt.fibers() {
            let _ = f.wait_settled().await;
        }
    }

    let sp = app.system_prompt().unwrap();
    assert!(sp.build().contains("LATE_MOUNT_RULE"));
    app.fiber().dispose().await;

    // 2. enabled = false 时不读取磁盘、不注入段落、report().enabled == false
    let app_off = Context::new();
    app_off
        .plugin(SystemPrompt::plugin(), SystemPromptConfig::default())
        .await
        .unwrap();
    let mut off_cfg = InstructionsConfig::new(&root);
    off_cfg.include_default_global = false;
    off_cfg.enabled = false;
    app_off
        .plugin(AgentInstructions::plugin(), off_cfg)
        .await
        .unwrap();

    let sp_off = app_off.system_prompt().unwrap();
    assert!(!sp_off.build().contains("LATE_MOUNT_RULE"));
    let rep_off = app_off.agent_instructions().unwrap().report();
    assert!(!rep_off.enabled);
    assert!(rep_off.scopes.is_empty());

    app_off.fiber().dispose().await;
}
