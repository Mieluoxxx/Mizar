use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cordis::{Cleanup, Context, LogLevel};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use dsh_agent_loop::{AgentLoopContextExt, RunOptions};
use dsh_agent_spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};
use dsh_llm_mock::MockPlugin;
use dsh_session::{SessionContextExt, SessionEvent};
use dsh_system_prompt::SystemPromptContextExt;
use dsh_trajectory::{TimelineMode, TrajectoryRuntime};
use dsh_tui::app::ToolExtraMeta;
use dsh_tui::editor::display_width;
use dsh_tui::emphasis::mount_reply_guidance_section;
use dsh_tui::overlay::{ApprovalDecision, SkillPickerItem};
use dsh_tui::selection::extract_selected_text;
use dsh_tui::side_panel::{FileTouchRow, ToolSummaryRow};
use dsh_tui::{
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
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
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
    assert!(welcome.contains("dsh-rs TUI"), "welcome:\n{welcome}");
    assert!(welcome.contains("mock/mock-v1"), "welcome:\n{welcome}");

    // 2. 触发一条真实 logger.warn，确认通过 UI 显示而不走 stdout
    app_ctx.logger_named("tui-test").warn("测试告警消息");

    // 3. 跑一轮会调用编码工具 find 的对话
    state.items.push(dsh_tui::ChatItem::User {
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
    assert!(screen.contains("✓ find(*)"), "screen:\n{screen}");
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
        ..
    } = spine_cfg.tools_mode
    {
        mcp.load_config_files = false;
        skills.include_default_roots = false;
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

    // 2. 验证 [[Done]] 胶囊、CJK 相邻粗体 **证据与影响：**源码中、词级 Diff 与原始 Session 不被污染
    let session = app_ctx.sessions().unwrap().create(None);
    let mut state = AppState::new(ws, &session.id, Vec::new(), None, None, None);
    state.items.push(ChatItem::Assistant {
        reasoning: String::new(),
        text: "[[Done]] **证据与影响：**源码中 `[[Done]]` 已被美化，而历史仍保持原始文本。".into(),
    });
    state.items.push(ChatItem::ToolCall {
        id: "edit-1".into(),
        name: "edit".into(),
        args_summary: "src/lib.rs".into(),
        status: dsh_tui::ToolState::Done,
        output_preview: "已更新 1 处".into(),
    });
    state.tool_extras.insert(
        "edit-1".into(),
        ToolExtraMeta {
            full_content: "已更新 1 处".into(),
            diff_raw: Some("-let old_val = 1;\n+let new_val = 2;\n".into()),
            local_expanded: None,
        },
    );

    let backend = TestBackend::new(110, 26);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| render(f, &mut state)).unwrap();

    let screen = buffer_text(terminal.backend());
    assert!(screen.contains("Done"), "screen:\n{screen}");
    assert!(screen.contains("证据与影响：源码中"), "screen:\n{screen}");
    assert!(screen.contains("old_val"), "screen:\n{screen}");
    assert!(screen.contains("new_val"), "screen:\n{screen}");
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
        path: "crates/dsh-tui/src/ui.rs".into(),
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
        files_screen.contains("crates/dsh-tui/src/ui.rs"),
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
        .find(|(k, _)| matches!(k, dsh_tui::LaunchpadQuickAction::ToggleSidePanel))
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
    session.append_trace(dsh_session::SessionTraceEvent::TurnStart { turn: 1 });
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
