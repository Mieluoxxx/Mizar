use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cordis::{Cleanup, Context, LogLevel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use mizar_agent_loop::{AgentLoopContextExt, RunOptions};
use mizar_agent_spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};
use mizar_llm_mock::MockPlugin;
use mizar_session::{SessionContextExt, SessionEvent};
use mizar_system_prompt::SystemPromptContextExt;
use mizar_trajectory::{TimelineMode, TrajectoryRuntime};
use mizar_tui::app::ToolExtraMeta;
use mizar_tui::editor::display_width;
use mizar_tui::emphasis::mount_reply_guidance_section;
use mizar_tui::overlay::{ApprovalDecision, SkillPickerItem};
use mizar_tui::selection::extract_selected_text;
use mizar_tui::side_panel::{FileTouchRow, ToolSummaryRow};
use mizar_tui::{
    AppAction, AppState, ApprovalMode, ChatItem, EmphasisConfig, ScreenMode, UiEvent,
    build_item_lines_for_scrollback, project_fork_view, project_resume_view,
    project_session_tree, render, subscribe_session_events, to_ui_log_record, wait_for_services,
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
    let app_ctx = Context::new();
    app_ctx.logger().set_silent();

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
    app_ctx.plugin(AgentSpinePlugin, spine_cfg);
    app_ctx.plugin(MockPlugin, ());
    wait_for_services(&app_ctx).await.unwrap();

    let session = app_ctx.sessions().unwrap().create(None);
    let active_sid = Arc::new(RwLock::new(session.id.clone()));
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let handles = subscribe_session_events(&app_ctx, Arc::clone(&active_sid), tx.clone()).unwrap();

    // 重定向 logger 到 UI 事件
    let log_tx = tx.clone();
    app_ctx.logger().set_handler(move |rec| {
        if rec.level >= LogLevel::Warn {
            let _ = log_tx.send(UiEvent::Log(to_ui_log_record(rec)));
        }
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

    // 2. 触发一条真实 logger.warn，确认通过 UI 显示而不走 stdout
    app_ctx.logger_named("tui-test").warn("测试告警消息");

    // 3. 跑一轮会调用编码工具 find 的对话
    state.items.push(mizar_tui::ChatItem::User {
        text: "项目里有哪些文件？".into(),
    });
    let reply = app_ctx
        .agent_loop()
        .unwrap()
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

    for h in handles {
        drop(h.dispose());
    }
    app_ctx.fiber().dispose().await;
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

async fn boot_mock_ctx(ws: PathBuf) -> Context {
    let app_ctx = Context::new();
    app_ctx.logger().set_silent();
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
    app_ctx.plugin(AgentSpinePlugin, spine_cfg);
    app_ctx.plugin(MockPlugin, ());
    wait_for_services(&app_ctx).await.unwrap();
    app_ctx
}

#[tokio::test]
async fn renders_emphasis_cjk_bold_edit_diff_and_mounts_prompt_section() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;

    // 1. 验证 pi-emphasis 回复引导片段只在挂载期注入系统提示词，卸载后干净消失
    let cleanup = mount_reply_guidance_section(&app_ctx, &EmphasisConfig::default())
        .unwrap()
        .expect("emphasis section cleanup");
    let sp = app_ctx.system_prompt().unwrap();
    let built_with = sp.build();
    assert!(
        built_with.contains("These rules shape replies"),
        "prompt:\n{built_with}"
    );

    match cleanup {
        Cleanup::Sync(f) => f(),
        Cleanup::Async(f) => f().await,
    }
    let built_without = sp.build();
    assert!(
        !built_without.contains("These rules shape replies"),
        "prompt after cleanup:\n{built_without}"
    );

    // 2. 验证 [[Done]] 胶囊、CJK 相邻粗体 **证据与影响：**源码中、词级 Diff、思考与正文空行隔离及去重
    let session = app_ctx.sessions().unwrap().create(None);
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
    assert!(!screen.contains("[▸ 展开]") && !screen.contains("[▾ 收起]"), "button should be removed:\n{screen}");
    assert!(screen.contains("Done"), "screen:\n{screen}");
    assert!(screen.contains("证据与影响：源码中"), "screen:\n{screen}");
    assert!(screen.contains("cargo fmt --all 单独提交"), "screen:\n{screen}");
    assert!(!screen.contains("**cargo"), "raw asterisks leaked:\n{screen}");
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
        lines.get(think_idx + 1).is_some_and(|l| l.trim().is_empty()),
        "expected blank line between thinking and response:\n{screen}"
    );
    assert!(
        lines.get(think_idx + 2).is_some_and(|l| l.contains("Investigate")),
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
    assert!(expanded_screen.contains("line7"), "expected expanded output:\n{expanded_screen}");
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

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn renders_side_panel_timeline_rail_and_multi_pane_selection_fence() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;

    let session = app_ctx.sessions().unwrap().create(None);
    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);

    // 构造多轮对话触发 TimelineRail（>=2 turns）并测试 Ctrl+B 双栏侧栏
    for idx in 1..=4 {
        state.items.push(ChatItem::User {
            text: format!("第 {idx} 轮提问：检查模块边界与渲染性能"),
        });
        state.items.push(ChatItem::Assistant {
            reasoning: String::new(),
            text: format!("第 {idx} 轮结论：[[Done]] 完成第 {idx} 轮核查。\n第一行摘要\n第二行详细说明"),
        });
    }
    state.side_panel_data.tools.push(ToolSummaryRow {
        name: "bash".into(),
        exposure: "direct".into(),
        declared: true,
    });

    // Ctrl+B 打开侧栏
    let act = state.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
    assert_eq!(act, AppAction::RefreshSidePanelData);

    let backend = TestBackend::new(120, 22);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();

    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("Context"), "screen:\n{screen}");
    assert!(state.rail_geo.is_some(), "expected TimelineRail geometry");
    let chat_rect = state.selection_geom.chat_rect.expect("chat rect");
    let side_rect = state.selection_geom.side_panel_rect.expect("side panel rect");
    assert!(chat_rect.x + chat_rect.width < side_rect.x);

    // 拖拽分割线调整侧栏宽度
    let div_col = state.side_panel.divider_col.expect("divider col");
    let orig_panel_w = state.side_panel.width;
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: div_col,
        row: 4,
        modifiers: KeyModifiers::NONE,
    });
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: div_col.saturating_sub(6),
        row: 4,
        modifiers: KeyModifiers::NONE,
    });
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Up(MouseButton::Left),
        column: div_col.saturating_sub(6),
        row: 4,
        modifiers: KeyModifiers::NONE,
    });
    assert!(state.side_panel.width > orig_panel_w);

    // 在 Chat 栏起手跨栏拖选直到屏幕最右侧，受围栏保护绝不能把右侧 SidePanel 文案复制出来
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: chat_rect.x,
        row: chat_rect.y,
        modifiers: KeyModifiers::NONE,
    });
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 119,
        row: chat_rect.y + 4,
        modifiers: KeyModifiers::NONE,
    });
    state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Up(MouseButton::Left),
        column: 119,
        row: chat_rect.y + 4,
        modifiers: KeyModifiers::NONE,
    });
    let copied = extract_selected_text(
        state.last_buffer.as_ref().unwrap(),
        &state.selection,
        &state.selection_geom,
    )
    .expect("extracted chat selection");
    assert!(copied.contains("检查模块边界与渲染性能"), "copied:\n{copied}");
    assert!(copied.contains("完成第 2 轮核查"), "copied:\n{copied}");
    assert!(!copied.contains("Context"), "leaked side panel:\n{copied}");
    assert!(!copied.contains('\u{e0b6}') && !copied.contains('\u{e0b4}'));

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn handles_overlays_theme_switch_and_approval_modal() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;

    let session = app_ctx.sessions().unwrap().create(None);
    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);

    // 1. 键入 `/` 自动打开 Slash 补全浮层，`Esc` 关闭
    state.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
    assert!(state.overlay.is_open());
    state.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!state.overlay.is_open());

    // 2. 执行 `/theme mist-light` 切换主题，并在画面渲染
    state.input.clear();
    state.input.set_text_end("/theme mist-light");
    state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(state.theme.id, "mist-light");
    assert!(state.theme.is_light);

    // 3. `/approval ask` + `UiEvent::ApprovalRequested` 弹出审批模态框并由 `y` 放行
    state.input.set_text_end("/approval ask");
    state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(state.approval_mode, ApprovalMode::Ask);

    let (tx, rx) = futures::channel::oneshot::channel();
    state.apply_ui_event(UiEvent::ApprovalRequested {
        tool_name: "bash".into(),
        args_preview: "{\"command\":\"cargo test\"}".into(),
        responder: Arc::new(std::sync::Mutex::new(Some(tx))),
    });
    assert!(state.overlay.is_modal());

    let backend = TestBackend::new(90, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("工具执行审批"), "screen:\n{screen}");
    assert!(screen.contains("bash"), "screen:\n{screen}");

    state.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
    assert!(!state.overlay.is_open());
    assert_eq!(rx.await.unwrap(), ApprovalDecision::AllowOnce);

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn opens_and_renders_trajectory_scene_from_runtime() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;

    let session = app_ctx.sessions().unwrap().create(None);
    let agent_loop = app_ctx.agent_loop().unwrap();
    let res = agent_loop
        .run_with_options(
            Arc::clone(&session),
            "项目里有哪些文件".to_string(),
            CancellationToken::new(),
            RunOptions::default(),
        )
        .await;
    assert!(res.is_ok(), "unexpected error: {res:?}");

    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);
    let act = state.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL));
    assert!(matches!(act, AppAction::OpenTrajectoryScene(_)));

    let traj = app_ctx.get::<TrajectoryRuntime>("trajectory").unwrap();
    let report = traj.inspect(&session.id, TimelineMode::Duration);
    state.open_trajectory_with_report(report);
    assert!(state.trajectory_scene.open);

    let backend = TestBackend::new(100, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("会话轨迹分析"), "screen:\n{screen}");
    assert!(screen.contains("Timeline"), "screen:\n{screen}");

    // `Tab` 切到热点聚合视图，`Esc` 退出全屏轨迹模式
    state.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let hotspots_screen = buffer_text(terminal.backend());
    assert!(hotspots_screen.contains("find"), "hotspots:\n{hotspots_screen}");

    state.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!state.trajectory_scene.open);

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn supports_session_tree_fork_approval_reason_skill_picker_and_sparklines() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let session = app_ctx.sessions().unwrap().create(None);
    session.append(SessionEvent::User {
        text: "first step investigate_very_long_identifier_for_horizontal_panning".into(),
    });
    session.append(SessionEvent::assistant_text("first answer"));
    session.append(SessionEvent::User {
        text: "second step should not be forked".into(),
    });

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
    let forked = app_ctx.sessions().unwrap().create(None);
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
    assert!(approval_screen.contains("use rg"), "screen:\n{approval_screen}");
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

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn supports_mode_toggles_external_editor_and_launchpad_actions() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let session = app_ctx.sessions().unwrap().create(None);
    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);

    // Launchpad 快捷芯片可点击激活侧栏
    let backend = TestBackend::new(100, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let (_, chip_rect) = state
        .launchpad_hit_rects
        .iter()
        .find(|(k, _)| matches!(k, mizar_tui::LaunchpadQuickAction::ToggleSidePanel))
        .expect("side panel launchpad chip")
        .clone();
    let act = state.handle_mouse(MouseEvent {
        kind: MouseEventKind::Up(MouseButton::Left),
        column: chip_rect.x + 1,
        row: chip_rect.y,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(act, AppAction::RefreshSidePanelData);
    assert!(state.side_panel.open);

    // F2 / F3 / Ctrl+G 动作分发与 Inline Scrollback 行构造
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE)),
        AppAction::SetMouseCapture(false)
    );
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::F(3), KeyModifiers::NONE)),
        AppAction::SwitchScreenMode(ScreenMode::Inline)
    );
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL)),
        AppAction::OpenExternalEditor
    );

    let scrollback_lines = build_item_lines_for_scrollback(
        &ChatItem::User {
            text: "inline mode line".into(),
        },
        None,
        80,
        &state.theme,
        &state.emphasis,
        state.show_thinking,
        state.verbose_expanded,
    );
    assert!(!scrollback_lines.is_empty());

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn resumes_session_with_compaction_and_trace_entries_cleanly() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let store = app_ctx.sessions().unwrap();
    let session = store.create(None);
    session.append_trace(mizar_session::SessionTraceEvent::TurnStart { turn: 1 });
    session.append(SessionEvent::User {
        text: "历史提问".into(),
    });
    session.append(SessionEvent::assistant_text("历史回答"));
    session.compact("早期讨论摘要", 2, Some(256));

    let empty = store.create(None);
    let mut state = AppState::new(ws, &empty.id, Vec::new(), None, None, None);
    state.restore_session_view(project_resume_view(&session));

    let backend = TestBackend::new(100, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("历史提问"), "screen:\n{screen}");
    assert!(screen.contains("历史回答"), "screen:\n{screen}");
    assert!(screen.contains("上下文压缩"), "screen:\n{screen}");
    assert!(screen.contains("已恢复历史会话"), "screen:\n{screen}");

    app_ctx.fiber().dispose().await;
}

#[tokio::test]
async fn folds_instruction_context_notices_on_resume_and_renders_side_panel_instructions() {
    let ws = crate_root();
    let app_ctx = boot_mock_ctx(ws.clone()).await;
    let store = app_ctx.sessions().unwrap();
    let session = store.create(None);
    session.append(SessionEvent::User {
        text: "测试问题".into(),
    });
    session.append(SessionEvent::ContextNotice {
        source: "agent-instructions".into(),
        text: "<system-reminder>\nInstructions from: AGENTS.md\n\nSECRET_RAW_BODY_DO_NOT_DUMP\n\nInstructions from: CLAUDE.md\n\nMORE_RAW\n</system-reminder>".into(),
    });
    session.append(SessionEvent::ContextNotice {
        source: "skill-catalog".into(),
        text: "<system-reminder>\n<available_skills>\n- `foo`: raw\n</available_skills>\n</system-reminder>".into(),
    });
    session.append(SessionEvent::assistant_text("测试回答"));

    let empty = store.create(None);
    let mut state = AppState::new(ws, &empty.id, Vec::new(), None, None, None);
    state.restore_session_view(project_resume_view(&session));
    state.side_panel.open = true;
    state.side_panel_data.instructions = vec![mizar_tui::side_panel::InstructionSummaryRow {
        path: "AGENTS.md".into(),
        scope: "workspace".into(),
        bytes: 42,
        truncated: false,
        omitted: false,
    }];

    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("已加载指令：AGENTS.md, CLAUDE.md"), "screen:\n{screen}");
    assert!(!screen.contains("SECRET_RAW_BODY_DO_NOT_DUMP"), "screen:\n{screen}");
    assert!(screen.contains("已加载可用技能目录"), "screen:\n{screen}");
    assert!(screen.contains("指令文件 (1)"), "screen:\n{screen}");

    state.input.insert_str("/instructions");
    let act = state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(act, AppAction::ShowInstructions);

    // 验证新键位：Ctrl+L 呼出模型选择器，Tab 循环思考等级，运行中 Enter 排队、Ctrl+Enter 立即发送、Esc 暂停
    let providers = vec![mizar_llm::ProviderInfo {
        id: "mock".into(),
        name: "Mock".into(),
        protocol: "mock".into(),
        models: vec![mizar_llm::ModelInfo {
            id: "mock-v1".into(),
            name: "Mock V1".into(),
            context_window: Some(32_768),
            max_tokens: Some(4096),
            reasoning_efforts: vec![
                mizar_llm::ReasoningEffort::Off,
                mizar_llm::ReasoningEffort::Low,
                mizar_llm::ReasoningEffort::High,
            ],
        }],
    }];
    state.providers = providers;
    state.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL));
    assert!(state.overlay.is_modal());
    state.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!state.overlay.is_open());

    state.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(state.reasoning_effort, Some(mizar_llm::ReasoningEffort::Low));
    state.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(state.reasoning_effort, Some(mizar_llm::ReasoningEffort::High));

    state.is_busy = true;
    state.input.insert_str("follow up");
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        AppAction::None
    );
    assert_eq!(state.queued_prompts, vec!["follow up".to_string()]);
    state.input.insert_str("steer now");
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
        AppAction::SteerPrompt("steer now".into())
    );
    assert_eq!(
        state.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        AppAction::CancelRun
    );

    app_ctx.fiber().dispose().await;
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
    let store = app_ctx.sessions().unwrap();
    let session = store.create(None);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let hook_handle =
        mizar_tui::mentions::register_mention_hook(&app_ctx, ws.clone(), false, tx.clone())
            .unwrap();

    let prompt = "请查看 @src/demo.rs#L2-3、@sub 与 @does_not_exist.rs";
    let _reply = app_ctx
        .agent_loop()
        .unwrap()
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
                SessionEvent::User { text } => users.push(text.clone()),
                SessionEvent::ContextNotice { source, text } if source == "file-mentions" => {
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
        body.contains("<attached-file path=\"src/demo.rs\" lines=\"2-3\">\nline 2\nline 3\n</attached-file>"),
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

    drop(hook_handle.dispose());
    app_ctx.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&ws);
}

#[tokio::test]
async fn renders_compact_consecutive_tools_and_restores_details_on_ctrl_o() {
    let ws = crate_root();
    let mut state = AppState::new(ws, "sess-compact-tools", Vec::new(), None, None, None);

    // 1. 连续 4 次只读调用应紧排成 4 行，无行间空行、无 │ 预览框，且仅末行末尾带 (Ctrl+O 展开)
    let read_paths = [
        "crates/cordis/src/lib.rs",
        "crates/mizar-agent-spine/Cargo.toml",
        "crates/mizar-tui-ui/src/ui.rs",
        "crates/mizar-tools/src/lib.rs",
    ];
    for (idx, path) in read_paths.iter().enumerate() {
        let call_id = format!("r-{idx}");
        state.apply_ui_event(UiEvent::ToolCall(mizar_llm::ToolCall {
            id: call_id.clone(),
            name: "read".into(),
            arguments: format!(r#"{{"path":"{path}"}}"#),
        }));
        state.apply_ui_event(UiEvent::ToolResult {
            call_id,
            name: "read".into(),
            content: "fn alpha() {}\nfn beta() {}\nfn gamma() {}".into(),
            is_error: false,
            blocked: false,
            details: Some(serde_json::json!({
                "path": path,
                "start_line": 1,
                "end_line": 3,
                "total_lines": 120,
                "truncated": false
            })),
        });
    }

    // 2. 长参数 bash（超过旧版 48 字符上限）+ 多行输出 + 1 条错误工具
    let long_cmd = "cd crates/mizar-tui-ui && cargo check --all-targets --message-format=short";
    state.apply_ui_event(UiEvent::ToolCall(mizar_llm::ToolCall {
        id: "b-long".into(),
        name: "bash".into(),
        arguments: format!(r#"{{"command":"{long_cmd}"}}"#),
    }));
    let bash_out = (1..=30)
        .map(|i| format!("build output line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    state.apply_ui_event(UiEvent::ToolResult {
        call_id: "b-long".into(),
        name: "bash".into(),
        content: bash_out,
        is_error: false,
        blocked: false,
        details: Some(serde_json::json!({ "exit_code": 0, "duration_ms": 42 })),
    });

    state.apply_ui_event(UiEvent::ToolCall(mizar_llm::ToolCall {
        id: "r-err".into(),
        name: "read".into(),
        arguments: r#"{"path":"missing/file.rs"}"#.into(),
    }));
    state.apply_ui_event(UiEvent::ToolResult {
        call_id: "r-err".into(),
        name: "read".into(),
        content: "File not found: missing/file.rs".into(),
        is_error: true,
        blocked: false,
        details: None,
    });

    let backend = TestBackend::new(130, 56);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let collapsed = buffer_text(terminal.backend());

    let tool_lines: Vec<&str> = collapsed
        .lines()
        .filter(|l| l.starts_with("● ") || l.starts_with("✗ ") || l.starts_with("  │ "))
        .collect();
    // 4 个 read (各 1 行) + 1 个 bash (1 头 + 2 尾行) + 1 个 error read (1 头 + 1 错误行) = 9 行
    assert_eq!(tool_lines.len(), 9, "collapsed:\n{collapsed}");
    assert!(
        collapsed.contains(long_cmd),
        "long bash command should not be cut at 48 chars:\n{collapsed}"
    );
    assert!(!collapsed.contains("前省略"), "collapsed:\n{collapsed}");
    assert!(
        !collapsed.contains("fn alpha"),
        "read body should stay folded:\n{collapsed}"
    );
    assert!(
        collapsed.contains("build output line 30"),
        "collapsed:\n{collapsed}"
    );
    assert!(
        !collapsed.contains("build output line 1\n"),
        "collapsed:\n{collapsed}"
    );
    assert!(
        collapsed.contains("File not found: missing/file.rs"),
        "error should remain visible when collapsed:\n{collapsed}"
    );
    assert_eq!(
        collapsed.matches("(Ctrl+O 展开)").count(),
        1,
        "expand hint should only appear once at group end:\n{collapsed}"
    );

    // 紧邻 read 行之间不应夹杂空行
    let screen_lines: Vec<&str> = collapsed.lines().collect();
    let first_read_row = screen_lines
        .iter()
        .position(|l| l.contains("crates/cordis/src/lib.rs"))
        .expect("first read row");
    assert!(
        screen_lines[first_read_row + 1].contains("crates/mizar-agent-spine/Cargo.toml"),
        "consecutive tools should not have blank line between them:\n{collapsed}"
    );

    // Ctrl+O 展开后恢复 read 全文与 bash 早期输出
    state.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
    terminal.draw(|f| render(f, &mut state)).unwrap();
    let expanded = buffer_text(terminal.backend());
    assert!(expanded.contains("fn alpha() {}"), "expanded:\n{expanded}");
    assert!(
        expanded.contains("build output line 1"),
        "expanded:\n{expanded}"
    );
}

#[test]
fn renders_wide_markdown_table_with_intact_short_col_wrapping_and_right_borders() {
    let ws = crate_root();
    let mut state = AppState::new(ws, "sess-wide-table", Vec::new(), None, None, None);

    // 放入 2 轮用户对话以激活 TimelineRail，覆盖全屏+导轨及侧栏开启时的视口宽度边界
    state.items.push(ChatItem::User {
        text: "第一轮".into(),
    });
    state.items.push(ChatItem::Assistant {
        reasoning: String::new(),
        text: "收到。".into(),
    });
    state.items.push(ChatItem::User {
        text: "总结四类工作".into(),
    });
    state.items.push(ChatItem::Assistant {
        reasoning: String::new(),
        text: "\
能干的事分四类：

| 类别 | 具体 |
| --- | --- |
| 读代码 | 读文件、全文搜索（`grep`）、按文件名查找、解释某段逻辑、追调用链、对照 `docs/plugins.md` 拓扑 |
| 跑命令 | `cargo check -p <crate>`、`cargo check --workspace`、`cargo test --workspace`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo fmt`、`git` |
| 改代码 | 用精确编辑做局部手术改动，新增 crate / 加工具插件 / 加 LLM 协议适配器；跨 crate 枚举改完自动跑全量 check |
| 守规矩 | 按仓库约定来：不用 `mod.rs`、错误用 `thiserror`、副作用包 `ctx.effect`、工具描述写英文、`inject` 漏写要查、feature 组合要验最小集 |"
            .into(),
    });

    // 1. 100 列全屏（TimelineRail 开启，chat content_area.width = 98）
    let mut term_rail = Terminal::new(TestBackend::new(100, 30)).unwrap();
    term_rail.draw(|f| render(f, &mut state)).unwrap();
    let screen_rail = buffer_text(term_rail.backend());
    for line in screen_rail
        .lines()
        .filter(|l| l.starts_with('│') && !l.contains("输入消息") && l.matches('│').count() >= 3)
    {
        let mut cols = line.split('│');
        let c1 = cols.nth(1).unwrap_or("");
        let c2 = cols.next().unwrap_or("");
        assert!(
            !c1.contains('…') && !c2.contains('…'),
            "unexpected ellipsis in table row: {line:?}\n{screen_rail}"
        );
    }
    for cat in ["读代码", "跑命令", "改代码", "守规矩"] {
        assert!(screen_rail.contains(cat), "missing {cat}:\n{screen_rail}");
    }
    assert!(
        screen_rail.contains("┌"),
        "missing top border:\n{screen_rail}"
    );
    assert!(
        screen_rail.contains("┐"),
        "right top corner clipped:\n{screen_rail}"
    );
    assert!(
        screen_rail.contains("┘"),
        "right bottom corner clipped:\n{screen_rail}"
    );
    assert!(
        screen_rail.contains("feature 组合要验最小集"),
        "wrapped cell content missing:\n{screen_rail}"
    );
    assert!(
        screen_rail.contains("-- -D warnings"),
        "wrapped command missing:\n{screen_rail}"
    );

    // 2. 开启 SidePanel（120 列总宽，chat + divider + side_panel）仍不裁剪右边框
    state.side_panel.open = true;
    let mut term_split = Terminal::new(TestBackend::new(120, 32)).unwrap();
    term_split.draw(|f| render(f, &mut state)).unwrap();
    let screen_split = buffer_text(term_split.backend());
    for line in screen_split
        .lines()
        .filter(|l| l.starts_with('│') && !l.contains("输入消息") && l.matches('│').count() >= 3)
    {
        let mut cols = line.split('│');
        let c1 = cols.nth(1).unwrap_or("");
        let c2 = cols.next().unwrap_or("");
        assert!(
            !c1.contains('…') && !c2.contains('…'),
            "unexpected ellipsis in split table row: {line:?}\n{screen_split}"
        );
    }
    for cat in ["读代码", "跑命令", "改代码", "守规矩"] {
        assert!(
            screen_split.contains(cat),
            "missing {cat} in split:\n{screen_split}"
        );
    }
    assert!(
        screen_split.contains("┐"),
        "split right top corner clipped:\n{screen_split}"
    );
    assert!(
        screen_split.contains("┘"),
        "split right bottom corner clipped:\n{screen_split}"
    );

    // 3. 20 列极窄终端无 panic
    state.side_panel.open = false;
    let mut term_narrow = Terminal::new(TestBackend::new(20, 24)).unwrap();
    term_narrow.draw(|f| render(f, &mut state)).unwrap();
    let screen_narrow = buffer_text(term_narrow.backend());
    assert!(
        screen_narrow.contains("┌"),
        "narrow render failed:\n{screen_narrow}"
    );
}

#[tokio::test]
async fn bash_mode_e2e_executes_isolates_cancels_truncates_and_replays_sqlite() {
    use cordis::{BoxFuture, Next};
    use mizar_llm::{
        BoxChunkStream, FinishReason, GenerateOptions, LlmAdapter, LlmContextExt, Message,
        ModelInfo, ProviderInfo, StreamChunk,
    };
    use mizar_tools::{EVENT_TOOLS_PRE_EXECUTE, PreExecuteDecision, ToolExecution};
    use mizar_tools_coding::bash::execute_bash_process;
    use mizar_tui::app::LocalBashOutput;
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
        let ctx = Context::new();
        ctx.logger().set_silent();
        let mut spine_cfg = AgentSpineConfig::coding(root).with_session_db(db);
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
        ctx.plugin(AgentSpinePlugin, spine_cfg);
        ctx.plugin(MockPlugin, ());
        wait_for_services(&ctx).await.unwrap();
        ctx
    };

    let ctx = boot_ctx(tmp_root.clone(), db_path.clone()).await;
    let _spy_handle = ctx
        .llm()
        .unwrap()
        .register_adapter(
            "spy",
            Arc::new(SpyLlm {
                seen_messages: Arc::clone(&seen_messages),
            }),
        )
        .unwrap();

    // 注册拦截器以验证 `!` / `!!` 本地命令不触发 `tools/pre-execute` 弹窗审批
    let hits_clone = Arc::clone(&pre_exec_hits);
    let _pre_hook = ctx
        .on_waterfall(
            EVENT_TOOLS_PRE_EXECUTE,
            move |_exec: ToolExecution,
                  _next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
                let hits_inner = Arc::clone(&hits_clone);
                Box::pin(async move {
                    hits_inner.fetch_add(1, Ordering::Relaxed);
                    PreExecuteDecision::deny("should not run")
                })
            },
        )
        .unwrap();

    let session = ctx.sessions().unwrap().create(Some("e2e-bash-sess".into()));
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
    assert!(!state.is_busy);
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
        .agent_loop()
        .unwrap()
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
    ctx.fiber().dispose().await;

    let ctx2 = Context::new();
    ctx2.logger().set_silent();
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
    ctx2.plugin(AgentSpinePlugin, spine_cfg2);
    wait_for_services(&ctx2).await.unwrap();

    let reloaded = ctx2
        .sessions()
        .unwrap()
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
    });
    assert_eq!(only_local_state.side_panel_data.context.tool_tokens, 0);

    ctx2.fiber().dispose().await;
    let _ = std::fs::remove_dir_all(&tmp_root);
}
