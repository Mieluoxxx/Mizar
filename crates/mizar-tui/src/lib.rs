//! 终端交互式 TUI 客户端宿主：桥接纯前端工作台 `mizar-tui-ui` 与 Cordis / AgentLoop / Trajectory 后端。

pub mod mentions;
pub mod runner;

pub mod emphasis {
    pub use crate::runner::mount_reply_guidance_section;
    pub use mizar_tui_ui::emphasis::*;
}

pub use mizar_tui_ui::{
    app, editor, markdown, overlay, reveal, selection, side_panel, terminal, theme, timeline_rail,
    trajectory_scene, ui,
};
pub use mizar_tui_ui::{
    AppAction, AppState, ApprovalMode, ChatItem, EmphasisConfig, LaunchpadQuickAction, NoticeLevel,
    ScreenMode, SessionReplayItem, SessionViewData, TerminalColors, Theme, ThemePreset, ToolState,
    UiAgentStatus, UiEvent, UiLogLevel, UiLogRecord, UiTurnEndReason,
    build_item_lines_for_scrollback, build_item_lines_for_scrollback_at,
    build_welcome_lines_for_scrollback, render,
};
pub use runner::{
    ResumeTarget, TuiConfig, TuiExitSummary, format_resume_command, mount_reply_guidance_section,
    project_fork_view, project_resume_view, project_session_tree, project_session_view, run_tui,
    run_tui_with_summary, subscribe_session_events, to_ui_log_record, wait_for_services,
};
