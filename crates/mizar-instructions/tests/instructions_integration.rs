use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use cordis::{BoxFuture, Context, Service};
use mizar_agent_loop::{AgentPreStepDecision, AgentPreStepInput, EVENT_AGENT_PRE_STEP};
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
        .waterfall(
            EVENT_AGENT_PRE_STEP,
            AgentPreStepInput {
                session: Arc::clone(session),
                turn: 1,
                step,
                provider: None,
                model: None,
                cancel: CancellationToken::new(),
                inputs: Vec::new(),
            },
            |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
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
    session.append(SessionEvent::user("hello"));

    // 1. 第 1 步：注入全量基线（不含未触及的 src/）
    emit_pre_step(&app, &session, 1).await;

    let msgs1 = session.derive_messages();
    assert_eq!(msgs1.len(), 2);
    let baseline_text = match &msgs1[1] {
        mizar_llm::Message::User { content, .. } => content.clone(),
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
        mizar_llm::Message::User { content, .. } => content.clone(),
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
        mizar_llm::Message::User { content, .. } => content.clone(),
        other => panic!("expected update ContextNotice, got {other:?}"),
    };
    assert!(delta_mod.contains("Updated instructions from: CLAUDE.md"));
    assert!(delta_mod.contains("ws claude rule v2"));
    assert!(delta_mod.contains("Instructions removed: CLAUDE.local.md"));

    // 5. Compaction 把旧基线折叠出可见窗口 -> 下一步应自动重建全量基线
    session.append(SessionEvent::user("after compact"));
    session.compact("早期摘要", 1, None);
    emit_pre_step(&app, &session, 5).await;
    let msgs5 = session.derive_messages();
    let last_msg = match msgs5.last().unwrap() {
        mizar_llm::Message::User { content, .. } => content.clone(),
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
    session.append(SessionEvent::user("hi"));
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

#[tokio::test]
async fn empty_repo_multi_notice_resume_and_delta_omitted_retry() {
    use mizar_session::{
        LoadedSession, PersistenceResult, SERVICE_SESSION_PERSISTENCE, SessionEntry, SessionHeader,
        SessionHeaderPatch, SessionListQuery, SessionPersistence, SessionPersistenceHandle,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemPersistence(Mutex<HashMap<String, (SessionHeader, Vec<SessionEntry>)>>);

    impl SessionPersistence for MemPersistence {
        fn create_session(&self, header: &SessionHeader) -> PersistenceResult<()> {
            self.0
                .lock()
                .unwrap()
                .insert(header.id.clone(), (header.clone(), Vec::new()));
            Ok(())
        }
        fn update_header(&self, _id: &str, _patch: SessionHeaderPatch) -> PersistenceResult<()> {
            Ok(())
        }
        fn load_session(&self, id: &str) -> PersistenceResult<Option<LoadedSession>> {
            Ok(self.0.lock().unwrap().get(id).map(|(h, e)| LoadedSession {
                header: h.clone(),
                entries: e.clone(),
                max_persisted_seq: e.last().map(|x| x.seq),
                skipped_events: 0,
            }))
        }
        fn list_sessions(&self, _q: &SessionListQuery) -> PersistenceResult<Vec<SessionHeader>> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .values()
                .map(|(h, _)| h.clone())
                .collect())
        }
        fn append_events(&self, id: &str, entries: &[SessionEntry]) -> PersistenceResult<()> {
            if let Some((_, log)) = self.0.lock().unwrap().get_mut(id) {
                log.extend(entries.iter().cloned());
            }
            Ok(())
        }
        fn flush(&self, _session_id: Option<&str>) -> PersistenceResult<()> {
            Ok(())
        }
        fn flush_async(
            &self,
            _session_id: Option<String>,
        ) -> cordis::BoxFuture<'static, PersistenceResult<()>> {
            Box::pin(async { Ok(()) })
        }
        fn fork_session(
            &self,
            _from: &str,
            _up_to: u64,
            new_header: &SessionHeader,
        ) -> PersistenceResult<LoadedSession> {
            Ok(LoadedSession {
                header: new_header.clone(),
                entries: Vec::new(),
                max_persisted_seq: None,
                skipped_events: 0,
            })
        }
    }

    let root = temp_dir("resume_omitted");
    let _guard = DirGuard(root.clone());
    let ws = root.join("repo");
    let sub_a = ws.join("a");
    let sub_b = ws.join("b");
    std::fs::create_dir_all(ws.join(".git")).unwrap();
    std::fs::create_dir_all(&sub_a).unwrap();
    std::fs::create_dir_all(&sub_b).unwrap();
    std::fs::write(sub_a.join("file.rs"), "pub fn a() {}\n").unwrap();
    std::fs::write(sub_b.join("file.rs"), "pub fn b() {}\n").unwrap();

    // 1. 空仓库首步不追加通知，随后新建 root/AGENTS.md 能增量加载且不会重复
    let persistence = Arc::new(MemPersistence::default());
    let app = Context::new();
    app.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(
            Arc::clone(&persistence) as Arc<dyn SessionPersistence>
        )),
    )
    .unwrap();
    app.plugin(SessionStore::plugin(), ()).await.unwrap();
    app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    let tools = app.tools().unwrap();
    let _read_cleanup = tools.register(create_read_tool(ws.clone(), false)).unwrap();

    let mut cfg = InstructionsConfig::new(&ws);
    cfg.include_default_global = false;
    cfg.max_bytes = 200;
    app.plugin(AgentInstructions::plugin(), cfg.clone())
        .await
        .unwrap();

    let session = app.sessions().unwrap().create(Some("s-resume".into()));
    session.append(SessionEvent::user("step1"));
    emit_pre_step(&app, &session, 1).await;
    assert_eq!(session.derive_messages().len(), 1);

    // 无变化再走一步仍保持 1 条
    emit_pre_step(&app, &session, 2).await;
    assert_eq!(session.derive_messages().len(), 1);

    // 新增根级 AGENTS.md，产生第 1 条通知
    std::fs::write(ws.join("AGENTS.md"), "root rule concise").unwrap();
    emit_pre_step(&app, &session, 3).await;
    let count_after_add = session.with_visible_entries(|e| {
        e.iter()
            .filter(|x| {
                matches!(
                    &x.event,
                    SessionEvent::ContextNotice { source, .. }
                        if source == mizar_instructions::AGENT_INSTRUCTIONS_SOURCE
                )
            })
            .count()
    });
    assert_eq!(count_after_add, 1);

    // 删除再重建成多条指令通知（模拟既有基线又有增量）后，重建 Context 恢复会话，不得重复基线
    std::fs::write(
        ws.join("AGENTS.md"),
        "root rule updated with longer content",
    )
    .unwrap();
    emit_pre_step(&app, &session, 4).await;
    app.sessions().unwrap().flush(Some("s-resume")).unwrap();
    let notice_count_before_restart = session.with_visible_entries(|e| {
        e.iter()
            .filter(|x| {
                matches!(
                    &x.event,
                    SessionEvent::ContextNotice { source, .. }
                        if source == mizar_instructions::AGENT_INSTRUCTIONS_SOURCE
                )
            })
            .count()
    });
    assert!(notice_count_before_restart >= 2);
    // 让日志包含与当前基线匹配的 Notice 供恢复识别
    let base_session = app.sessions().unwrap().create(Some("s-resume-2".into()));
    base_session.append(SessionEvent::user("init"));
    emit_pre_step(&app, &base_session, 1).await;
    std::fs::write(sub_a.join("AGENTS.md"), "sub a rule").unwrap();
    let _ = tools
        .execute(ToolExecution::new(
            "read",
            serde_json::json!({"path": "a/file.rs"}),
            Some("s-resume-2".into()),
        ))
        .await;
    emit_pre_step(&app, &base_session, 2).await;
    app.sessions().unwrap().flush(Some("s-resume-2")).unwrap();

    // 重建 Context（清空内存会话状态），模拟进程重启恢复：visible_instruction_notices.len() == 2
    app.fiber().dispose().await;
    let app2 = Context::new();
    app2.provide(
        SERVICE_SESSION_PERSISTENCE,
        Arc::new(SessionPersistenceHandle::new(
            Arc::clone(&persistence) as Arc<dyn SessionPersistence>
        )),
    )
    .unwrap();
    app2.plugin(SessionStore::plugin(), ()).await.unwrap();
    app2.plugin(ToolRuntime::plugin(), ()).await.unwrap();
    let tools2 = app2.tools().unwrap();
    let _r2 = tools2
        .register(create_read_tool(ws.clone(), false))
        .unwrap();
    app2.plugin(AgentInstructions::plugin(), cfg.clone())
        .await
        .unwrap();

    let resumed = app2.sessions().unwrap().get("s-resume-2").unwrap();
    let before_resumed_notices = resumed.with_visible_entries(|e| {
        e.iter()
            .filter(|x| {
                matches!(
                    &x.event,
                    SessionEvent::ContextNotice { source, .. }
                        if source == mizar_instructions::AGENT_INSTRUCTIONS_SOURCE
                )
            })
            .count()
    });
    assert_eq!(before_resumed_notices, 2);
    emit_pre_step(&app2, &resumed, 1).await;
    let after_resumed_notices = resumed.with_visible_entries(|e| {
        e.iter()
            .filter(|x| {
                matches!(
                    &x.event,
                    SessionEvent::ContextNotice { source, .. }
                        if source == mizar_instructions::AGENT_INSTRUCTIONS_SOURCE
                )
            })
            .count()
    });
    assert_eq!(
        after_resumed_notices, before_resumed_notices,
        "resumed session with baseline + delta notices must not re-inject baseline"
    );

    // 2. Delta 预算溢出导致尾部新增子目录文件被省略，随后删除前置文件释放预算后必须重试注入而非永久跳过
    let sub_c = ws.join("c");
    let sub_d = ws.join("d");
    std::fs::create_dir_all(&sub_c).unwrap();
    std::fs::create_dir_all(&sub_d).unwrap();
    std::fs::write(sub_c.join("file.rs"), "pub fn c() {}\n").unwrap();
    std::fs::write(sub_d.join("file.rs"), "pub fn d() {}\n").unwrap();
    std::fs::write(sub_a.join("AGENTS.md"), "A".repeat(180)).unwrap();
    std::fs::write(sub_b.join("AGENTS.md"), "B".repeat(180)).unwrap();
    std::fs::write(sub_c.join("AGENTS.md"), "C".repeat(180)).unwrap();
    std::fs::write(sub_d.join("AGENTS.md"), "D".repeat(120)).unwrap();
    for p in ["a/file.rs", "b/file.rs", "c/file.rs", "d/file.rs"] {
        let _ = tools2
            .execute(ToolExecution::new(
                "read",
                serde_json::json!({"path": p}),
                Some("s-resume-2".into()),
            ))
            .await;
    }
    emit_pre_step(&app2, &resumed, 2).await;
    let rep1 = app2
        .agent_instructions()
        .unwrap()
        .session_report("s-resume-2")
        .unwrap();
    let d_status1 = rep1
        .scopes
        .iter()
        .find(|s| s.display_path == "d/AGENTS.md")
        .unwrap();
    assert!(
        d_status1.omitted,
        "d/AGENTS.md should be marked omitted when pushed past delta budget: {:?}",
        rep1.scopes
    );

    // 删除前置文件腾出预算并触发变动
    std::fs::remove_file(sub_a.join("AGENTS.md")).unwrap();
    std::fs::remove_file(sub_b.join("AGENTS.md")).unwrap();
    std::fs::remove_file(sub_c.join("AGENTS.md")).unwrap();
    emit_pre_step(&app2, &resumed, 3).await;
    let rep2 = app2
        .agent_instructions()
        .unwrap()
        .session_report("s-resume-2")
        .unwrap();
    let d_status2 = rep2
        .scopes
        .iter()
        .find(|s| s.display_path == "d/AGENTS.md")
        .unwrap();
    assert!(
        !d_status2.omitted,
        "previously omitted d/AGENTS.md must be retried and represented once space frees up"
    );

    app2.fiber().dispose().await;
}
