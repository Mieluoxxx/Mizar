use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use mizar_core::agent_loop::RunOptions;
use mizar_core::spine::{AgentSpineConfig, AgentToolsMode, MizarRuntime};
use mizar_core::trajectory::TimelineMode;
use mizar_llm::{
    BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, ModelInfo, ProviderInfo,
    ReasoningEffort, StreamChunk,
};
use mizar_session::{SessionCreateOptions, SessionEvent, TitleSource};
use mizar_tui::app::ToolExtraMeta;
use mizar_tui::editor::display_width;
use mizar_tui::emphasis::mount_reply_guidance_section;
use mizar_tui::overlay::{ApprovalDecision, SkillPickerItem};
use mizar_tui::side_panel::FileTouchRow;
use mizar_tui::{
    AppAction, AppState, ChatItem, EmphasisConfig, SessionReplayItem, TuiConfig, UiEvent,
    UiLogLevel, collect_session_picker_items, install_tracing_handler, project_fork_view,
    project_resume_view, project_session_tree, render, resume_session_into_app,
    subscribe_session_events,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn buffer_text(backend: &TestBackend) -> String {
    let buf = backend.buffer();
    let mut lines = Vec::new();
    for y in 0..buf.area.height {
        let mut row = String::new();
        let mut x = 0;
        while x < buf.area.width {
            let sym = buf[(x, y)].symbol();
            row.push_str(sym);
            let w = display_width(sym).max(1) as u16;
            x += w;
        }
        lines.push(row.trim_end().to_string());
    }
    lines.join("\n")
}

#[tokio::test]
async fn renders_welcome_and_streams_mock_tool_roundtrip_into_test_backend() {
    let mut spine_cfg = AgentSpineConfig::coding(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let runtime = MizarRuntime::build(spine_cfg).await.unwrap();
    runtime.register_mock_adapter().unwrap();

    let session = runtime.sessions.create(None);
    let active_sid = Arc::new(RwLock::new(session.id.clone()));
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let handles = subscribe_session_events(&runtime, Arc::clone(&active_sid), tx.clone());

    // 重定向 tracing 日志到 UI 事件
    let log_tx = tx.clone();
    let _log_guard = install_tracing_handler(UiLogLevel::Warn, move |rec| {
        let _ = log_tx.send(UiEvent::Log(rec.clone()));
    });

    let mut state = AppState::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        &session.id,
        Vec::new(),
        Some("mock".into()),
        Some("mock-v1".into()),
        None,
    );

    let backend = TestBackend::new(80, 48);
    let mut terminal = Terminal::new(backend).unwrap();

    // 1. 初始欢迎画面
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let welcome = buffer_text(terminal.backend());
    assert!(welcome.contains("Mizar"), "welcome:\n{welcome}");
    assert!(welcome.contains("mock/mock-v1"), "welcome:\n{welcome}");

    // 2. 触发一条真实 tracing::warn!，确认通过 UI 显示而不走 stdout
    tracing::warn!(target: "mizar::tui-test", "测试告警消息");

    // 3. 跑一轮会调用编码工具 find 的对话
    state.items.push(mizar_tui::ChatItem::User {
        text: "项目里有哪些文件？".into(),
    });
    let reply = runtime
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            "项目里有哪些文件？",
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await
        .unwrap();
    let _ = tx.send(UiEvent::RunFinished(Ok(reply)));

    // 把通道里的事件全部应用到 AppState
    tokio::time::sleep(Duration::from_millis(20)).await;
    while let Ok(ev) = rx.try_recv() {
        state.apply_ui_event(ev);
    }

    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("测试告警消息"), "screen:\n{screen}");
    assert!(screen.contains("项目里有哪些文件"), "screen:\n{screen}");
    assert!(screen.contains("● find(*)"), "screen:\n{screen}");
    assert!(screen.contains("工具返回了结果"), "screen:\n{screen}");
    assert!(screen.contains("● 就绪"), "screen:\n{screen}");

    // 4. 窄终端尺寸下渲染不 panic
    let narrow_backend = TestBackend::new(20, 8);
    let mut narrow_term = Terminal::new(narrow_backend).unwrap();
    narrow_term.draw(|f| render(f, &mut state)).unwrap();

    drop(handles);
    runtime.shutdown().await;
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

async fn boot_mock_ctx(ws: PathBuf) -> Arc<MizarRuntime> {
    let mut spine_cfg = AgentSpineConfig::coding(ws);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let runtime = MizarRuntime::build(spine_cfg).await.unwrap();
    runtime.register_mock_adapter().unwrap();
    runtime
}

#[tokio::test]
async fn renders_emphasis_cjk_bold_edit_diff_and_mounts_prompt_section() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;

    // 1. 验证 pi-emphasis 回复引导片段只在挂载期注入系统提示词，卸载后干净消失
    let cleanup = mount_reply_guidance_section(
        &app_ctx.system_prompt,
        Some(Arc::clone(&app_ctx.sessions)),
        &EmphasisConfig::default(),
    )
    .expect("emphasis section cleanup");
    let sp = &app_ctx.system_prompt;
    let built_with = sp.build();
    assert!(
        built_with.contains("These rules shape replies"),
        "prompt:\n{built_with}"
    );

    drop(cleanup);
    let built_without = sp.build();
    assert!(
        !built_without.contains("These rules shape replies"),
        "prompt after cleanup:\n{built_without}"
    );

    // 2. 验证 [[Done]] 胶囊、CJK 相邻粗体 **证据与影响：**源码中、词级 Diff、思考与正文空行隔离及去重
    let session = app_ctx.sessions.create(None);
    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);
    state.items.push(ChatItem::Assistant {
        reasoning: "Let me write Chinese content.".into(),
        text: "**Investigate** [[Investigate]] 我先梳理各个 crate 的依赖与职责、agent 主循环与工具调度机制，并确认 **`cargo fmt --all` 单独提交**。\n\n[[Done]] **证据与影响：**源码中 `[[Done]]` 已被美化，而历史仍保持原始文本。".into(),
    });
    state.items.push(ChatItem::ToolCall {
        id: "edit-1".into(),
        name: "edit".into(),
        args_summary: "src/lib.rs".into(),
        status: mizar_tui::ToolState::Done,
        output_preview: "已更新 1 处".into(),
    });
    state.tool_extras.insert(
        "edit-1".into(),
        ToolExtraMeta {
            full_content: "已更新 1 处".into(),
            diff_raw: Some("-let old_val = 1;\n+let new_val = 2;\n".into()),
            ..Default::default()
        },
    );

    let backend = TestBackend::new(110, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();

    let screen = buffer_text(terminal.backend());
    assert!(
        !screen.contains("[▸ 展开]") && !screen.contains("[▾ 收起]"),
        "button should be removed:\n{screen}"
    );
    assert!(screen.contains("Done"), "screen:\n{screen}");
    assert!(screen.contains("证据与影响：源码中"), "screen:\n{screen}");
    assert!(
        screen.contains("cargo fmt --all 单独提交"),
        "screen:\n{screen}"
    );
    assert!(
        !screen.contains("**cargo"),
        "raw asterisks leaked:\n{screen}"
    );
    assert_eq!(
        screen.matches("Investigate").count(),
        1,
        "duplicate Investigate leaked:\n{screen}"
    );
    let lines: Vec<&str> = screen.lines().collect();
    let think_idx = lines
        .iter()
        .position(|l| l.contains("Let me write Chinese content."))
        .expect("thinking line");
    assert!(
        lines
            .get(think_idx + 1)
            .is_some_and(|l| l.trim().is_empty()),
        "expected blank line between thinking and response:\n{screen}"
    );
    assert!(
        lines
            .get(think_idx + 2)
            .is_some_and(|l| l.contains("Investigate")),
        "expected standalone mode pill after blank line:\n{screen}"
    );
    assert!(
        lines
            .get(think_idx + 3)
            .is_some_and(|l| l.starts_with("我先梳理")),
        "expected prose at col 0:\n{screen}"
    );
    assert!(screen.contains("old_val"), "screen:\n{screen}");
    assert!(screen.contains("new_val"), "screen:\n{screen}");

    // 验证全靠 Ctrl+O 统一展开与折叠（思考块与多行工具输出同步）
    state.items.push(ChatItem::ToolCall {
        id: "bash-1".into(),
        name: "bash".into(),
        args_summary: "ls".into(),
        status: mizar_tui::ToolState::Done,
        output_preview: "line1\nline2\nline3\nline4\nline5\nline6".into(),
    });
    state.tool_extras.insert(
        "bash-1".into(),
        ToolExtraMeta {
            full_content: "line1\nline2\nline3\nline4\nline5\nline6\nline7".into(),
            diff_raw: None,
            ..Default::default()
        },
    );
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let collapsed_screen = buffer_text(terminal.backend());
    assert!(
        collapsed_screen.contains("(Ctrl+O 展开)"),
        "expected inline Ctrl+O expand hint:\n{collapsed_screen}"
    );
    assert!(
        !collapsed_screen.contains("点击或按"),
        "unexpected click hint:\n{collapsed_screen}"
    );
    assert!(
        !collapsed_screen.contains("还有"),
        "unexpected multi-line footer hint:\n{collapsed_screen}"
    );
    assert!(!collapsed_screen.contains("line7"));

    let items_before = state.items.len();
    state.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
    assert!(state.verbose_expanded);
    assert!(state.show_thinking);
    assert_eq!(state.items.len(), items_before);
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let expanded_screen = buffer_text(terminal.backend());
    assert!(
        expanded_screen.contains("line7"),
        "expected expanded output:\n{expanded_screen}"
    );
    assert!(expanded_screen.contains("Let me write Chinese content."));
    assert!(!expanded_screen.contains("[▾ 收起]"));

    state.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
    assert!(!state.verbose_expanded);
    assert!(!state.show_thinking);
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let folded_screen = buffer_text(terminal.backend());
    assert!(
        folded_screen.contains("思考过程已折叠") && folded_screen.contains("按 Ctrl+O 展开"),
        "expected thinking folded with Ctrl+O hint:\n{folded_screen}"
    );
    assert!(!folded_screen.contains("line7"));
    assert!(session.log().is_empty());

    app_ctx.shutdown().await;
}

#[tokio::test]
async fn supports_session_tree_fork_approval_reason_skill_picker_and_sparklines() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let session = app_ctx.sessions.create(None);
    session.append(SessionEvent::user(
        "first step investigate_very_long_identifier_for_horizontal_panning",
    ));
    session.append(SessionEvent::assistant_text("first answer"));
    session.append(SessionEvent::user("second step should not be forked"));

    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);

    // 1. `/tree` 树状浏览 + 左右平移 + 选择节点触发 ForkSessionAtSeq
    state.input.set_text_end("/tree");
    let tree_act = state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(tree_act, AppAction::OpenSessionTree);
    state.open_session_tree_with(project_session_tree(&session));
    assert!(state.overlay.is_modal());

    let backend = TestBackend::new(90, 22);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let tree_screen = buffer_text(terminal.backend());
    assert!(tree_screen.contains("分支树"), "screen:\n{tree_screen}");

    // 向上移动到第 1 条用户消息并向右平移，随后按 `f` 触发分叉
    state.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    state.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    state.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    let fork_act = state.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE));
    assert_eq!(fork_act, AppAction::ForkSessionAtSeq(1));

    // 重放截断日志验证 restore_session_view
    let forked = app_ctx.sessions.create(None);
    for e in session.log() {
        if e.seq <= 1 {
            forked.append(e.event);
        }
    }
    state.restore_session_view(project_fork_view(&forked, 1));
    assert!(
        state
            .items
            .iter()
            .any(|it| matches!(it, ChatItem::User { text } if text.contains("first step")))
    );
    assert!(
        !state
            .items
            .iter()
            .any(|it| matches!(it, ChatItem::User { text } if text.contains("second step")))
    );

    // 2. 审批弹窗 `[r]` 附加拒绝理由并回传
    let (tx, rx) = futures::channel::oneshot::channel();
    state.apply_ui_event(UiEvent::ApprovalRequested {
        request_id: "req-headless-1".into(),
        call_id: Some("call-headless-1".into()),
        tool_name: "bash".into(),
        args_preview: "{\"command\":\"rm -rf tmp\"}".into(),
        responder: Arc::new(std::sync::Mutex::new(Some(tx))),
    });
    state.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE));
    for ch in "use rg".chars() {
        state.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
    }
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let approval_screen = buffer_text(terminal.backend());
    assert!(
        approval_screen.contains("use rg"),
        "screen:\n{approval_screen}"
    );
    state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(
        rx.await.unwrap(),
        ApprovalDecision::DenyWithReason("use rg".to_string())
    );

    // 3. `/skills` 交互选择器 & SidePanel 性能火花线 / Git 状态渲染
    state.input.set_text_end("/skills");
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        AppAction::OpenSkillPicker
    );
    state.overlay.open_skill_picker(vec![SkillPickerItem {
        name: "rust-audit".into(),
        description: "审计 crates".into(),
        source: "project".into(),
    }]);
    state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(state.input.text(), "/rust-audit ");

    state.input.clear();
    state.tps_history = vec![22.0, 48.0, 74.0];
    state.ttft_history = vec![320, 180, 110];
    state.refresh_context_breakdown();
    state.side_panel_data.git_changes.push(FileTouchRow {
        path: "crates/mizar-tui/src/ui.rs".into(),
        op: "M".into(),
    });
    state.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
    let wide_backend = TestBackend::new(115, 24);
    let mut wide_term = Terminal::new(wide_backend).unwrap();
    wide_term.draw(|f| render(f, &mut state)).unwrap();
    let sp_screen = buffer_text(wide_term.backend());
    assert!(sp_screen.contains("▁▅█"), "sp context:\n{sp_screen}");
    assert!(sp_screen.contains("█▃▁"), "sp context:\n{sp_screen}");

    // 切到 Files 页签验证 Git 工作区变更显示
    state.side_panel.tab = state.side_panel.tab.next();
    wide_term.draw(|f| render(f, &mut state)).unwrap();
    let files_screen = buffer_text(wide_term.backend());
    assert!(
        files_screen.contains("crates/mizar-tui/src/ui.rs"),
        "sp files:\n{files_screen}"
    );

    app_ctx.shutdown().await;
}

#[tokio::test]
async fn expands_at_mentions_via_pre_step_and_folds_on_resume() {
    let ws = std::env::temp_dir().join(format!(
        "mizar_tui_headless_mentions_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join("src")).unwrap();
    std::fs::create_dir_all(ws.join("sub")).unwrap();
    std::fs::write(ws.join("src/demo.rs"), "line 1\nline 2\nline 3\nline 4\n").unwrap();
    std::fs::write(ws.join("sub/inside.txt"), "SECRET_NESTED\n").unwrap();

    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let store = &app_ctx.sessions;
    let session = store.create(None);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let hook_handle = mizar_tui::mentions::register_mention_hook(
        &app_ctx.agent_loop,
        Some(Arc::clone(&app_ctx.llm)),
        ws.clone(),
        false,
        tx.clone(),
    );

    let prompt = "请查看 @src/demo.rs#L2-3、@sub 与 @does_not_exist.rs";
    let _reply = app_ctx
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            prompt,
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await
        .unwrap();

    // 1. User 事件保持干净原始文本；file-mentions ContextNotice 按预期附加片段
    let (user_texts, mention_notices) = session.with_visible_entries(|entries| {
        let mut users = Vec::new();
        let mut notices = Vec::new();
        for e in entries {
            match &e.event {
                SessionEvent::User { text, .. } => users.push(text.clone()),
                SessionEvent::ContextNotice { source, text, .. } if source == "file-mentions" => {
                    notices.push(text.clone());
                }
                _ => {}
            }
        }
        (users, notices)
    });
    assert_eq!(user_texts, vec![prompt.to_string()]);
    assert_eq!(mention_notices.len(), 1);
    let body = &mention_notices[0];
    assert!(
        body.contains(
            "<attached-file path=\"src/demo.rs\" lines=\"2-3\">\nline 2\nline 3\n</attached-file>"
        ),
        "body:\n{body}"
    );
    assert!(!body.contains("line 1"), "body:\n{body}");
    assert!(!body.contains("line 4"), "body:\n{body}");
    assert!(
        body.contains("<attached-directory path=\"sub\">\ninside.txt\n</attached-directory>"),
        "body:\n{body}"
    );

    // 2. UI 收到 Flash 与 Missing Warn Notice
    let mut state = AppState::new(ws.clone(), &session.id, Vec::new(), None, None, None);
    while let Ok(ev) = rx.try_recv() {
        state.apply_ui_event(ev);
    }
    assert_eq!(state.active_flash(), Some("✓ 已附加 2 个 @ 引用"));
    assert!(
        state.items.iter().any(|item| matches!(
            item,
            ChatItem::Notice {
                level: mizar_tui::NoticeLevel::Warn,
                text
            } if text.contains("does_not_exist.rs")
        )),
        "items: {:?}",
        state.items
    );

    // 3. --resume 回放折叠 file-mentions，不把原始文件正文展开进聊天记录
    let empty = store.create(None);
    let mut resumed = AppState::new(ws.clone(), &empty.id, Vec::new(), None, None, None);
    resumed.restore_session_view(project_resume_view(&session));
    let backend = TestBackend::new(120, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut resumed)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(
        screen.contains("已附加引用：src/demo.rs, sub"),
        "screen:\n{screen}"
    );
    assert!(!screen.contains("line 2"), "screen:\n{screen}");

    drop(hook_handle);
    app_ctx.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
}

#[tokio::test]
async fn bash_mode_e2e_executes_isolates_cancels_truncates_and_replays_sqlite() {
    use mizar_llm::{
        BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, Message, ModelInfo,
        ProviderInfo, StreamChunk,
    };
    use mizar_tools::coding::bash::execute_bash_process;
    use mizar_tools::{PreExecuteDecision, ToolExecution};
    use mizar_tui::app::LocalBashOutput;
    use mizar_types::{BoxFuture, Next};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SpyLlm {
        seen_messages: Arc<Mutex<Vec<Vec<Message>>>>,
    }

    impl LlmAdapter for SpyLlm {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "spy".into(),
                name: "Spy".into(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: "spy-1".into(),
                    name: "Spy 1".into(),
                    context_window: Some(32_768),
                    max_tokens: Some(4096),
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.seen_messages.lock().unwrap().push(options.messages);
            let chunks = vec![
                Ok(StreamChunk::text("模型已收到结果")),
                Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }),
            ];
            Box::pin(futures::stream::iter(chunks))
        }
    }

    let tmp_root = std::env::temp_dir().join(format!("mizar_tui_bash_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&tmp_root).unwrap();
    let db_path = tmp_root.join("sessions.sqlite");

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    let pre_exec_hits = Arc::new(AtomicUsize::new(0));

    let boot_ctx = |root: PathBuf, db: PathBuf| async move {
        let mut spine_cfg = AgentSpineConfig::coding(root)
            .with_session_db(db)
            .with_disabled_plugins(["session-title"]);
        if let AgentToolsMode::Coding {
            ref mut mcp,
            ref mut skills,
            ref mut instructions,
            ..
        } = spine_cfg.tools_mode
        {
            mcp.load_config_files = false;
            skills.include_default_roots = false;
            instructions.include_default_global = false;
        }
        let runtime = MizarRuntime::build(spine_cfg).await.unwrap();
        runtime.register_mock_adapter().unwrap();
        runtime
    };

    let ctx = boot_ctx(tmp_root.clone(), db_path.clone()).await;
    ctx.register_adapter(
        "spy",
        Arc::new(SpyLlm {
            seen_messages: Arc::clone(&seen_messages),
        }),
    )
    .unwrap();

    // 注册拦截器以验证 `!` / `!!` 本地命令不触发 `tools.pre_execute_hook()` 弹窗审批
    let hits_clone = Arc::clone(&pre_exec_hits);
    let _pre_hook = ctx.tools.pre_execute_hook().register(
        move |_exec: ToolExecution,
              _next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
            let hits_inner = Arc::clone(&hits_clone);
            Box::pin(async move {
                hits_inner.fetch_add(1, Ordering::Relaxed);
                PreExecuteDecision::deny("should not run")
            })
        },
    );

    let session = ctx.sessions.create(Some("e2e-bash-sess".into()));
    let mut state = AppState::new(
        tmp_root.clone(),
        &session.id,
        Vec::new(),
        Some("spy".into()),
        Some("spy-1".into()),
        None,
    );

    // 1. 输入框 `! echo CTX_VISIBLE_E2E` -> 提交得到 RunLocalBash，不触发 AgentLoop
    state.input.set_text_end("! echo CTX_VISIBLE_E2E");
    let act1 = state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(
        act1,
        AppAction::RunLocalBash {
            command: "echo CTX_VISIBLE_E2E".into(),
            exclude_from_context: false,
        }
    );
    state.start_local_bash("local-1", "echo CTX_VISIBLE_E2E", false);
    let res1 = execute_bash_process(
        "echo CTX_VISIBLE_E2E",
        &tmp_root,
        None,
        CancellationToken::new(),
        |_| {},
    )
    .await
    .unwrap();
    let out1 = res1.session_output();
    session.append(SessionEvent::BashExecution {
        command: "echo CTX_VISIBLE_E2E".into(),
        output: out1.clone(),
        exit_code: res1.exit_code,
        cancelled: res1.cancelled,
        exclude_from_context: false,
    });
    state.apply_ui_event(UiEvent::LocalBashFinished {
        call_id: "local-1".into(),
        command: "echo CTX_VISIBLE_E2E".into(),
        exclude_from_context: false,
        output: Ok(LocalBashOutput {
            display: res1.display,
            session_output: out1,
            exit_code: res1.exit_code,
            cancelled: res1.cancelled,
            timed_out: res1.timed_out,
        }),
    });
    // 执行完 ! 后不应自动触发模型回复
    assert!(!state.is_busy());
    assert!(seen_messages.lock().unwrap().is_empty());

    // 2. 输入框 `!! echo LOCAL_HIDDEN_E2E` -> 仅本地执行，不进入上下文
    state.input.set_text_end("!! echo LOCAL_HIDDEN_E2E");
    let act2 = state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(
        act2,
        AppAction::RunLocalBash {
            command: "echo LOCAL_HIDDEN_E2E".into(),
            exclude_from_context: true,
        }
    );
    state.start_local_bash("local-2", "echo LOCAL_HIDDEN_E2E", true);
    let res2 = execute_bash_process(
        "echo LOCAL_HIDDEN_E2E",
        &tmp_root,
        None,
        CancellationToken::new(),
        |_| {},
    )
    .await
    .unwrap();
    let out2 = res2.session_output();
    session.append(SessionEvent::BashExecution {
        command: "echo LOCAL_HIDDEN_E2E".into(),
        output: out2.clone(),
        exit_code: res2.exit_code,
        cancelled: res2.cancelled,
        exclude_from_context: true,
    });
    state.apply_ui_event(UiEvent::LocalBashFinished {
        call_id: "local-2".into(),
        command: "echo LOCAL_HIDDEN_E2E".into(),
        exclude_from_context: true,
        output: Ok(LocalBashOutput {
            display: res2.display,
            session_output: out2,
            exit_code: res2.exit_code,
            cancelled: res2.cancelled,
            timed_out: res2.timed_out,
        }),
    });

    // 3. 大输出截断落盘到 temp_dir().join("mizar") 与 CancellationToken 中断测试
    let big_res = execute_bash_process(
        "seq 1 2500",
        &tmp_root,
        None,
        CancellationToken::new(),
        |_| {},
    )
    .await
    .unwrap();
    assert!(big_res.truncated);
    let full_path = PathBuf::from(big_res.full_output_path.as_ref().expect("temp log path"));
    assert!(full_path.starts_with(std::env::temp_dir().join("mizar")));
    assert!(full_path.exists());
    let _ = std::fs::remove_file(&full_path);

    let cancel = CancellationToken::new();
    let cancel_trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        cancel_trigger.cancel();
    });
    let cancelled_res = execute_bash_process("sleep 10", &tmp_root, None, cancel, |_| {})
        .await
        .unwrap();
    assert!(cancelled_res.cancelled);

    assert_eq!(pre_exec_hits.load(Ordering::Relaxed), 0);

    // 4. 随后向模型提问：SpyLlm 必须能看见 ! 命令与输出，绝不可见 !! 命令与输出
    let reply = ctx
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            "分析上面命令的输出",
            CancellationToken::new(),
            RunOptions {
                provider: Some("spy".into()),
                model: Some("spy-1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply, "模型已收到结果");

    {
        let recorded = seen_messages.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        let serialized = format!("{:?}", recorded[0]);
        assert!(serialized.contains("echo CTX_VISIBLE_E2E"), "{serialized}");
        assert!(serialized.contains("CTX_VISIBLE_E2E"), "{serialized}");
        assert!(
            !serialized.contains("LOCAL_HIDDEN_E2E"),
            "!! output leaked to LLM:\n{serialized}"
        );
    }

    // 5. 关闭当前宿主并从 SQLite 重建宿主验证 --resume 与 /tree 回放
    session.flush_async().await.unwrap();
    ctx.shutdown().await;

    let mut spine_cfg2 = AgentSpineConfig::coding(tmp_root.clone()).with_session_db(db_path);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg2.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let ctx2 = MizarRuntime::build(spine_cfg2).await.unwrap();

    let reloaded = ctx2
        .sessions
        .get("e2e-bash-sess")
        .expect("reloaded session from sqlite");

    // /tree 包含 ! 与 !! 且保留归属标识
    let tree = project_session_tree(&reloaded);
    assert!(
        tree.iter().any(|e| e.kind_tag.contains("BASH!")
            && !e.kind_tag.contains("!!")
            && e.summary.contains("echo CTX_VISIBLE_E2E")),
        "tree missing ! entry: {tree:?}"
    );
    assert!(
        tree.iter()
            .any(|e| e.kind_tag == "BASH!!" && e.summary.contains("!! echo LOCAL_HIDDEN_E2E")),
        "tree missing !! entry: {tree:?}"
    );

    // --resume 回放还原两张极简 bash 卡片且上下文统计排除 !!
    let view = project_resume_view(&reloaded);
    let mut resumed_state =
        AppState::new(tmp_root.clone(), &reloaded.id, Vec::new(), None, None, None);
    resumed_state.restore_session_view(view);

    let backend = TestBackend::new(110, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut resumed_state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(
        screen.contains("● bash(echo CTX_VISIBLE_E2E) · 已入上下文"),
        "{screen}"
    );
    assert!(
        screen.contains("● bash(echo LOCAL_HIDDEN_E2E) · 仅本地"),
        "{screen}"
    );
    assert!(screen.contains("CTX_VISIBLE_E2E"), "{screen}");
    assert!(screen.contains("LOCAL_HIDDEN_E2E"), "{screen}");

    // 验证仅有 ! 计入 tool_tokens（若去掉 ! 卡片仅留 !! 则 tool_tokens 为 0）
    assert!(resumed_state.side_panel_data.context.tool_tokens > 0);
    let mut only_local_state = AppState::new(tmp_root.clone(), "s2", Vec::new(), None, None, None);
    only_local_state.restore_session_view(mizar_tui::SessionViewData {
        session_id: "s2".into(),
        session_title: None,
        items: vec![mizar_tui::SessionReplayItem::BashExecution {
            command: "echo SECRET_ONLY".into(),
            output: "SECRET_ONLY_BODY\n".repeat(20),
            exit_code: Some(0),
            cancelled: false,
            exclude_from_context: true,
        }],
        finish_notice: "ok".into(),
        selection: None,
        ..Default::default()
    });
    assert_eq!(only_local_state.side_panel_data.context.tool_tokens, 0);

    ctx2.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp_root);
}

#[tokio::test]
async fn mention_image_respects_model_input_capability_and_delivers_multimodal_payload() {
    use futures::channel::mpsc as futures_mpsc;
    use mizar_llm::{
        BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, Message, ModelInfo,
        ProviderInfo, StreamChunk,
    };
    use std::io::Cursor;
    use std::sync::Mutex;

    struct VisionSpyAdapter {
        seen: Mutex<Vec<Vec<Message>>>,
    }

    impl LlmAdapter for VisionSpyAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "spy".into(),
                name: "Spy".into(),
                protocol: "mock".into(),
                models: vec![
                    ModelInfo {
                        id: "text-only".into(),
                        name: "Text Only".into(),
                        context_window: Some(32_768),
                        max_tokens: Some(4096),
                        reasoning_efforts: Vec::new(),
                        input: vec!["text".into()],
                        ..Default::default()
                    },
                    ModelInfo {
                        id: "vision-ok".into(),
                        name: "Vision OK".into(),
                        context_window: Some(32_768),
                        max_tokens: Some(4096),
                        reasoning_efforts: Vec::new(),
                        input: vec!["text".into(), "image".into()],
                        ..Default::default()
                    },
                ],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.seen.lock().unwrap().push(options.messages);
            let (tx, rx) = futures_mpsc::unbounded();
            let _ = tx.unbounded_send(Ok(StreamChunk::text("已分析图片")));
            let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                reason: FinishReason::Stop,
                replay: None,
            }));
            Box::pin(rx)
        }
    }

    let tmp = std::env::temp_dir().join(format!("mizar_tui_mention_vis_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let img = image::DynamicImage::ImageRgba8(image::ImageBuffer::from_pixel(
        64,
        48,
        image::Rgba([255, 128, 0, 255]),
    ));
    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .unwrap();
    std::fs::write(tmp.join("shot.png"), &png_bytes).unwrap();

    let mut cfg = AgentSpineConfig::coding(tmp.clone()).with_disabled_plugins(["session-title"]);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let ctx = MizarRuntime::build(cfg).await.unwrap();

    let spy = Arc::new(VisionSpyAdapter {
        seen: Mutex::new(Vec::new()),
    });
    ctx.register_adapter("spy", spy.clone()).unwrap();

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();
    let _mention_guard = mizar_tui::mentions::register_mention_hook(
        &ctx.agent_loop,
        Some(Arc::clone(&ctx.llm)),
        tmp.clone(),
        true,
        ui_tx,
    );

    let session = ctx.sessions.create(None);
    let agent = &ctx.agent_loop;

    // 1. 使用仅声明 input=["text"] 的模型时，@shot.png 应在发出模型请求前被明确取消并提示错误
    let blocked = agent
        .run_with_options(
            Arc::clone(&session),
            "看下 @shot.png",
            CancellationToken::new(),
            RunOptions {
                provider: Some("spy".into()),
                model: Some("text-only".into()),
                ..Default::default()
            },
        )
        .await;
    assert!(blocked.is_err());
    assert!(
        spy.seen.lock().unwrap().is_empty(),
        "text-only model must not receive request"
    );
    let mut notices = Vec::new();
    while let Ok(ev) = ui_rx.try_recv() {
        if let UiEvent::Notice { level, text } = ev {
            notices.push((level, text));
        }
    }
    assert!(
        notices
            .iter()
            .any(|(lvl, txt)| *lvl == mizar_tui::NoticeLevel::Error
                && txt.contains("不支持图片输入")),
        "expected error notice for unsupported model, got {notices:?}"
    );

    // 2. 换用声明了 image 能力的模型，@shot.png 作为图片附件送达适配器，绝不落成 <attached-file>
    let reply = agent
        .run_with_options(
            Arc::clone(&session),
            "再看一次 @shot.png",
            CancellationToken::new(),
            RunOptions {
                provider: Some("spy".into()),
                model: Some("vision-ok".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply, "已分析图片");

    {
        let seen = spy.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let img_msg = seen[0]
            .iter()
            .find(|m| !m.images().is_empty())
            .expect("should contain user/context message with image attachment");
        assert_eq!(img_msg.images().len(), 1);
        assert_eq!(img_msg.images()[0].media_type, "image/png");
        assert_eq!(img_msg.images()[0].width, Some(64));
        assert_eq!(img_msg.images()[0].height, Some(48));
        assert!(
            img_msg
                .text_content()
                .contains("<attached-image path=\"shot.png\" />")
        );
        assert!(!img_msg.text_content().contains("<attached-file"));
    }

    ctx.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp);
}

/// 端到端回归测试：AgentLoop -> subscribe_session_events -> AppState -> render() / TrajectoryRuntime / --resume
/// 覆盖 MaxSteps（不污染 session.log）与 FinishReason::MaxTokens 截断（⚠ 输出截断、step 提示、interrupted_steps=1）。
#[tokio::test]
async fn end_to_end_max_steps_and_max_tokens_truncation_status_trajectory_and_resume() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::stream;
    use mizar_llm::{
        BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, StreamChunk, ToolCall,
    };
    use mizar_tui::terminal::ProgramStatusState;
    use mizar_tui::{NoticeLevel, UiStepOutcome, UiTurnEndReason};

    struct ScriptedEndStatesAdapter {
        mode: Arc<AtomicUsize>,
    }

    impl LlmAdapter for ScriptedEndStatesAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let mode = self.mode.load(Ordering::SeqCst);
            let chunks = if mode == 0 {
                vec![
                    Ok(StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "call-read-1".into(),
                            name: "read".into(),
                            arguments: r#"{"path":"Cargo.toml"}"#.into(),
                        },
                    )),
                    Ok(StreamChunk::Finish {
                        reason: FinishReason::ToolCalls,
                        replay: None,
                    }),
                ]
            } else {
                vec![
                    Ok(StreamChunk::text("被 MaxTokens 截断的半截回答")),
                    Ok(StreamChunk::Finish {
                        reason: FinishReason::MaxTokens,
                        replay: None,
                    }),
                ]
            };
            Box::pin(stream::iter(chunks))
        }
    }

    let mut spine_cfg = AgentSpineConfig::coding(PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .with_disabled_plugins(["session-title"]);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let app_ctx = MizarRuntime::build(spine_cfg).await.unwrap();

    let mode = Arc::new(AtomicUsize::new(0));
    app_ctx
        .register_adapter(
            "scripted-end-states",
            Arc::new(ScriptedEndStatesAdapter {
                mode: Arc::clone(&mode),
            }),
        )
        .unwrap();

    // 1. Phase A: MaxSteps (max_steps=1)
    let session = app_ctx.sessions.create(None);
    let active_sid = Arc::new(RwLock::new(session.id.clone()));
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let _handles = subscribe_session_events(&app_ctx, Arc::clone(&active_sid), tx.clone());

    let mut state = AppState::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        session.id.clone(),
        Vec::new(),
        Some("scripted-end-states".into()),
        None,
        None,
    );
    let mut terminal = Terminal::new(TestBackend::new(120, 24)).unwrap();

    let max_steps_ret = app_ctx
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            "执行工具直至步数上限",
            CancellationToken::new(),
            RunOptions {
                provider: Some("scripted-end-states".into()),
                max_steps: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        max_steps_ret.contains("max_steps=1"),
        "expected max_steps return text, got: {max_steps_ret}"
    );
    tokio::task::yield_now().await;
    while let Ok(ev) = rx.try_recv() {
        state.apply_ui_event(ev);
    }

    assert_eq!(state.last_turn_end_reason, Some(UiTurnEndReason::MaxSteps));
    assert_ne!(
        state.last_program_status.as_ref().map(|(s, _)| *s),
        Some(ProgramStatusState::Done)
    );
    assert!(state.items.iter().any(|item| matches!(
        item,
        ChatItem::Notice {
            level: NoticeLevel::Warn,
            text,
        } if text.contains("步数上限")
    )));
    assert!(
        !session.log().iter().any(|e| matches!(
            &e.event,
            SessionEvent::Assistant { blocks, .. } if format!("{blocks:?}").contains("max_steps=1")
        )),
        "MaxSteps 提示不应作为 Assistant 消息写入 session.log()"
    );

    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen_a = buffer_text(terminal.backend());
    assert!(
        screen_a.contains("⚠ 步数上限"),
        "expected ⚠ 步数上限 on screen:\n{screen_a}"
    );
    assert!(
        !screen_a.contains("● 就绪"),
        "MaxSteps must not render 就绪:\n{screen_a}"
    );

    // 2. Phase B: FinishReason::MaxTokens -> Truncated
    let s2 = app_ctx.sessions.create(None);
    *active_sid.write().unwrap() = s2.id.clone();
    state.reset_for_new_session(&s2.id, None);
    mode.store(1, Ordering::SeqCst);

    let trunc_ret = app_ctx
        .agent_loop
        .run_with_options(
            Arc::clone(&s2),
            "触发 MaxTokens 截断",
            CancellationToken::new(),
            RunOptions {
                provider: Some("scripted-end-states".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(trunc_ret, "被 MaxTokens 截断的半截回答");
    tokio::task::yield_now().await;
    while let Ok(ev) = rx.try_recv() {
        state.apply_ui_event(ev);
    }

    assert_eq!(state.last_turn_end_reason, Some(UiTurnEndReason::Truncated));
    assert_eq!(state.last_step_outcome, Some((1, UiStepOutcome::MaxTokens)));
    assert_ne!(
        state.last_program_status.as_ref().map(|(s, _)| *s),
        Some(ProgramStatusState::Done)
    );
    assert!(state.items.iter().any(|item| matches!(
        item,
        ChatItem::Notice {
            level: NoticeLevel::Warn,
            text,
        } if text.contains("第 1 步") && text.contains("max_tokens")
    )));

    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen_b = buffer_text(terminal.backend());
    assert!(
        screen_b.contains("⚠ 输出截断"),
        "expected ⚠ 输出截断 on screen:\n{screen_b}"
    );
    assert!(
        !screen_b.contains("● 就绪"),
        "Truncated must not render 就绪:\n{screen_b}"
    );

    let traj = app_ctx.trajectory.as_ref().unwrap();
    let report = traj.inspect(&s2.id, TimelineMode::Sequence).unwrap();
    assert_eq!(
        report.snapshot.stats.interrupted_steps, 1,
        "FinishReason::MaxTokens step must increment stats.interrupted_steps"
    );

    // 3. Phase C: --resume projection (project_resume_view)
    let view_a = project_resume_view(&session);
    assert!(view_a.items.iter().any(|item| matches!(
        item,
        SessionReplayItem::Notice {
            level: NoticeLevel::Warn,
            text,
        } if text.contains("max_steps=1")
    )));
    assert!(!view_a.items.iter().any(|item| matches!(
        item,
        SessionReplayItem::Assistant { blocks } if format!("{blocks:?}").contains("max_steps=1")
    )));

    let view_b = project_resume_view(&s2);
    assert!(view_b.items.iter().any(|item| matches!(
        item,
        SessionReplayItem::Notice {
            level: NoticeLevel::Warn,
            text,
        } if text.contains("max_tokens")
    )));

    drop(_handles);
    app_ctx.shutdown().await;
}

#[tokio::test]
async fn e2e_session_title_generation_sqlite_persistence_rename_and_resume_picker() {
    use std::sync::Mutex;

    struct TitleAwareAdapter {
        calls: Arc<Mutex<Vec<GenerateOptions>>>,
    }

    impl LlmAdapter for TitleAwareAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "fake".into(),
                name: "FakeCloud".into(),
                protocol: "mock".into(),
                models: vec![ModelInfo {
                    id: "smart-1".into(),
                    name: "Smart 1".into(),
                    context_window: Some(64_000),
                    max_tokens: Some(4096),
                    reasoning_efforts: vec![ReasoningEffort::Low],
                    input: Vec::new(),
                    ..Default::default()
                }],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.calls.lock().unwrap().push(options.clone());
            let is_title_prompt = options
                .messages
                .first()
                .map(|m| m.text_content().contains("TYPE｜Topic"))
                .unwrap_or(false);
            let reply = if is_title_prompt {
                "FEA｜终端会话标题与恢复"
            } else {
                "收到，已处理完成"
            };
            let chunks = vec![
                Ok(StreamChunk::text(reply)),
                Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }),
            ];
            Box::pin(futures::stream::iter(chunks))
        }
    }

    let tmp =
        std::env::temp_dir().join(format!("mizar_tui_title_resume_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let ws = tmp.join("my-repo");
    std::fs::create_dir_all(&ws).unwrap();
    let db_path = tmp.join("sessions.sqlite");

    let recorded_calls = Arc::new(Mutex::new(Vec::new()));
    let adapter = Arc::new(TitleAwareAdapter {
        calls: Arc::clone(&recorded_calls),
    });

    let mut cfg = AgentSpineConfig::coding(&ws).with_session_db(&db_path);
    cfg.session_title = Some(mizar_core::spine::SessionTitleConfig::default());
    cfg.agent.provider = Some("fake".into());
    cfg.agent.model = Some("smart-1".into());
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let ctx = MizarRuntime::build(cfg).await.unwrap();
    ctx.register_adapter("fake", adapter).unwrap();

    // 1. 创建一个空会话（event_count == 0）与一个真实对话会话
    let empty_sess = ctx.sessions.create_with(SessionCreateOptions {
        id: Some("sess-empty".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    assert_eq!(empty_sess.header().event_count, 0);

    let mut active_session = ctx.sessions.create_with(SessionCreateOptions {
        id: Some("sess-target".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    let active_sid = Arc::new(RwLock::new(active_session.id.clone()));
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel();
    let _subs = subscribe_session_events(&ctx, Arc::clone(&active_sid), ui_tx.clone());

    let mut app = AppState::new(
        ws.clone(),
        active_session.id.clone(),
        ctx.llm.provider_infos(),
        Some("fake".into()),
        Some("smart-1".into()),
        Some(ReasoningEffort::Medium),
    );

    // 2. 跑一轮真实 AgentLoop：首轮异步触发标题生成，回写并经由事件流更新 app.session_title
    let reply = ctx
        .agent_loop
        .run(
            Arc::clone(&active_session),
            "mizar --resume 12345678-1234-1234-1234-123456789abc 实现会话标题生成",
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(reply, "收到，已处理完成");

    let mut generated_title = None;
    for _ in 0..80 {
        while let Ok(ev) = ui_rx.try_recv() {
            app.apply_ui_event(ev);
        }
        if active_session
            .header()
            .title
            .as_deref()
            .is_some_and(|t| t.contains("｜FEA｜终端会话标题与恢复"))
        {
            generated_title = active_session.header().title.clone();
            break;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    while let Ok(ev) = ui_rx.try_recv() {
        app.apply_ui_event(ev);
    }
    let generated_title = generated_title.expect("background title must update session header");
    assert_eq!(app.session_title.as_deref(), Some(generated_title.as_str()));
    assert_eq!(
        active_session.header().title_source,
        Some(TitleSource::Auto)
    );

    // 3. collect_session_picker_items 过滤空会话（仅留 sess-target），开启新会话后通过选择器切换回来
    let picker_items = collect_session_picker_items(&ctx.sessions);
    assert_eq!(picker_items.len(), 1);
    assert_eq!(picker_items[0].session_id, "sess-target");
    assert_eq!(picker_items[0].title, generated_title);
    assert_eq!(picker_items[0].cwd_tail, "my-repo");

    // 切到一个新会话，再从 SessionPicker 选回 sess-target
    let fresh = ctx.sessions.create_with(SessionCreateOptions {
        id: Some("sess-fresh".into()),
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    *active_sid.write().unwrap() = fresh.id.clone();
    app.reset_for_new_session(fresh.id.clone(), None);
    active_session = fresh;

    app.overlay.open_session_picker(picker_items);
    let select_act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(select_act, AppAction::ResumeSession("sess-target".into()));

    let tui_cfg = TuiConfig::new(&ws);
    assert!(
        resume_session_into_app(
            &ctx,
            &tui_cfg,
            &mut app,
            &mut active_session,
            &active_sid,
            "sess-target",
        )
        .await
    );
    assert_eq!(app.session_id, "sess-target");
    assert_eq!(app.session_title.as_deref(), Some(generated_title.as_str()));

    // 4. 手工 /rename 后标题及 Manual 所有权应落盘，重启 SQLite 仍可读取
    assert!(active_session.set_title("0402｜FIX｜手工修订标题"));
    while let Ok(ev) = ui_rx.try_recv() {
        app.apply_ui_event(ev);
    }
    assert_eq!(
        app.session_title.as_deref(),
        Some("0402｜FIX｜手工修订标题")
    );
    active_session.flush_async().await.unwrap();
    ctx.shutdown().await;

    let mut cfg_r = AgentSpineConfig::coding(&ws).with_session_db(&db_path);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = cfg_r.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let ctx_restarted = MizarRuntime::build(cfg_r).await.unwrap();

    let reloaded = ctx_restarted.sessions.get("sess-target").unwrap();
    assert_eq!(
        reloaded.header().title.as_deref(),
        Some("0402｜FIX｜手工修订标题")
    );
    assert_eq!(reloaded.header().title_source, Some(TitleSource::Manual));

    ctx_restarted.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn tui_runtime_with_trace_jsonl_streams_complete_trace_events() {
    use futures::stream;
    use mizar_core::trajectory::{TraceEventKind, TrajectoryRuntime};
    use mizar_tui::exec::configure_spine_trace_jsonl;

    struct StaticReplyAdapter;

    impl LlmAdapter for StaticReplyAdapter {
        fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
            let chunks = vec![
                Ok(StreamChunk::text("TUI_TRACE_JSONL_OK")),
                Ok(StreamChunk::Finish {
                    reason: FinishReason::Stop,
                    replay: None,
                }),
            ];
            Box::pin(stream::iter(chunks))
        }
    }

    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = std::env::temp_dir().join(format!(
        "mizar_tui_trace_jsonl_{}_{}",
        std::process::id(),
        uniq
    ));
    let ws = tmp.join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let jsonl_path = tmp.join("out/tui-session.jsonl");

    let mut spine_cfg =
        AgentSpineConfig::coding(&ws).with_disabled_plugins(["session-title", "trajectory"]);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    configure_spine_trace_jsonl(&mut spine_cfg, Some(&jsonl_path));

    let runtime = MizarRuntime::build(spine_cfg).await.unwrap();
    runtime
        .register_adapter("static-tui", Arc::new(StaticReplyAdapter))
        .unwrap();

    let session = runtime.sessions.create(None);
    let reply = runtime
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            "hello tui trace",
            CancellationToken::new(),
            RunOptions {
                provider: Some("static-tui".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply, "TUI_TRACE_JSONL_OK");

    let raw = std::fs::read_to_string(&jsonl_path).expect("TUI trace JSONL must be written");
    let events = TrajectoryRuntime::parse_jsonl(&raw).expect("TUI trace JSONL must parse cleanly");
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::UserMessage { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::TurnStart { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::StepStart { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::FirstToken { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::AssistantMessage { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::StepEnd { .. }))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, TraceEventKind::TurnEnd { .. }))
    );

    let expected = runtime
        .trajectory
        .as_ref()
        .unwrap()
        .export_jsonl(&session.id)
        .unwrap();
    assert_eq!(raw, expected);

    runtime.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn subagent_yield_renders_assistant_message_persists_resume_and_delivers_batched_wake() {
    use futures::StreamExt;
    use mizar_llm::{ContentBlock, RequestOrigin, ToolCall};
    use serde_json::json;

    const YIELD_MSG: &str = "正在后台分析模块 A 与模块 B，请稍候。";

    struct YieldTuiAdapter {
        parent_calls: std::sync::atomic::AtomicUsize,
        child_gate: Arc<tokio::sync::Notify>,
    }

    impl LlmAdapter for YieldTuiAdapter {
        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            let finish = |reason| StreamChunk::Finish {
                reason,
                replay: None,
            };
            if options.origin == RequestOrigin::Subagent {
                let gate = Arc::clone(&self.child_gate);
                return Box::pin(
                    futures::stream::once(async move {
                        gate.notified().await;
                        Ok(StreamChunk::text("TUI_CHILD_REPORT"))
                    })
                    .chain(futures::stream::iter([Ok(finish(FinishReason::Stop))])),
                );
            }
            let call_n = self
                .parent_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let chunks = if call_n == 0 {
                vec![
                    StreamChunk::tool_call(
                        0,
                        ToolCall {
                            id: "sp_1".into(),
                            name: "subagent".into(),
                            arguments: json!({"name": "tui-scout", "prompt": "inspect A"})
                                .to_string(),
                        },
                    ),
                    StreamChunk::tool_call(
                        1,
                        ToolCall {
                            id: "yd_1".into(),
                            name: "subagent".into(),
                            arguments: json!({"action": "yield", "message": YIELD_MSG}).to_string(),
                        },
                    ),
                    finish(FinishReason::ToolCalls),
                ]
            } else {
                let saw_child = options
                    .messages
                    .iter()
                    .any(|m| m.text_content().contains("TUI_CHILD_REPORT"));
                let reply = if saw_child {
                    "已汇总 TUI 子代理发现。"
                } else {
                    "MISSING_CHILD_REPORT"
                };
                vec![StreamChunk::text(reply), finish(FinishReason::Stop)]
            };
            Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
        }
    }

    let tmp = std::env::temp_dir().join(format!(
        "mizar_tui_subagent_yield_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let ws = tmp.join("ws");
    std::fs::create_dir_all(&ws).unwrap();

    let mut spine_cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["session-title"]);
    if let AgentToolsMode::Coding {
        ref mut mcp,
        ref mut skills,
        ref mut instructions,
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
        instructions.include_default_global = false;
    }
    let runtime = MizarRuntime::build(spine_cfg).await.unwrap();
    let child_gate = Arc::new(tokio::sync::Notify::new());
    runtime
        .register_adapter(
            "yield-tui",
            Arc::new(YieldTuiAdapter {
                parent_calls: std::sync::atomic::AtomicUsize::new(0),
                child_gate: Arc::clone(&child_gate),
            }),
        )
        .unwrap();

    let session = runtime.sessions.create_with(SessionCreateOptions {
        cwd: Some(ws.display().to_string()),
        ..Default::default()
    });
    let active_sid = Arc::new(RwLock::new(session.id.clone()));
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();
    let _handles = subscribe_session_events(&runtime, Arc::clone(&active_sid), ui_tx.clone());
    let mut wake_rx = runtime.agent_loop.inbox().subscribe_wake();

    let mut state = AppState::new(
        ws.clone(),
        &session.id,
        Vec::new(),
        Some("yield-tui".into()),
        Some("m1".into()),
        None,
    );
    let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();

    state.items.push(ChatItem::User {
        text: "调研项目".into(),
    });
    let reply1 = runtime
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            "调研项目",
            CancellationToken::new(),
            RunOptions {
                provider: Some("yield-tui".into()),
                model: Some("m1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply1, YIELD_MSG);
    let _ = ui_tx.send(UiEvent::RunFinished(Ok(reply1)));

    tokio::time::sleep(Duration::from_millis(20)).await;
    while let Ok(ev) = ui_rx.try_recv() {
        state.apply_ui_event(ev);
    }

    // 1. 实时 TUI 必须将 yield message 作为正式 Assistant 气泡渲染，而非仅藏在 ToolResult
    assert!(
        state
            .items
            .iter()
            .any(|item| matches!(item, ChatItem::Assistant { text, .. } if text == YIELD_MSG)),
        "expected ChatItem::Assistant with yield message, items={:?}",
        state.items
    );
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains(YIELD_MSG), "screen:\n{screen}");

    // 2. --resume 投影必须保留 yield assistant 消息
    let resume_view = project_resume_view(&session);
    assert!(
        resume_view.items.iter().any(|item| matches!(
            item,
            SessionReplayItem::Assistant { blocks }
            if blocks.iter().any(|b| matches!(b, ContentBlock::Text { text } if text == YIELD_MSG))
        )),
        "resume view missing yield assistant message: {:?}",
        resume_view.items
    );

    // 3. 放行子代理后，wake_rx 收到唤醒并由 FollowUp 驱动第二轮合成回复
    child_gate.notify_one();
    let woken_sid = tokio::time::timeout(Duration::from_secs(5), wake_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(woken_sid, session.id);

    let followup = runtime
        .agent_loop
        .inbox()
        .pop_followup(&session.id)
        .expect("follow-up must arrive after child settles");
    let reply2 = runtime
        .agent_loop
        .run_with_options(
            Arc::clone(&session),
            followup,
            CancellationToken::new(),
            RunOptions {
                provider: Some("yield-tui".into()),
                model: Some("m1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(reply2, "已汇总 TUI 子代理发现。");

    runtime.shutdown().await;
    let _ = std::fs::remove_dir_all(&tmp);
}
