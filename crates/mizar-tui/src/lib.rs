//! 终端交互式 TUI 客户端宿主：桥接纯前端工作台 `mizar-tui-ui` 与 Cordis / AgentLoop / Trajectory 后端。

pub mod clipboard;
pub mod exec;
pub mod images;
pub mod mentions;
pub mod runner;
pub mod update;

pub mod emphasis {
    pub use crate::runner::mount_reply_guidance_section;
    pub use mizar_tui_ui::emphasis::*;
}

pub use mizar_tui_ui::{
    AppAction, AppState, ApprovalMode, ChatItem, EmphasisConfig, LaunchpadQuickAction, NoticeLevel,
    PendingSettingsModelPick, ScreenMode, SessionPickerItem, SessionReplayItem, SessionSelection,
    SessionViewData, TerminalColors, Theme, ThemePreset, ToolState, UiAgentStatus, UiEvent,
    UiLogLevel, UiLogRecord, UiStepOutcome, UiTurnEndReason, build_item_lines_for_scrollback,
    build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback, render,
    turn_end_program_status,
};
pub use mizar_tui_ui::{
    app, editor, markdown, overlay, reveal, selection, side_panel, terminal, theme, timeline_rail,
    trajectory_scene, ui,
};
pub use runner::{
    ResumeTarget, SelectionSources, TuiConfig, TuiExitSummary, apply_runtime_setting_change,
    build_session_info_report, build_session_info_report_at, build_settings_entries,
    collect_session_picker_items, format_resume_command, latest_workspace_selection,
    mount_reply_guidance_section, persist_setting_change_to, project_fork_view,
    project_resume_view, project_session_tree, project_session_view, resolve_selection,
    resume_session_into_app, run_tui, run_tui_with_summary, subscribe_session_events,
    to_ui_log_record, wait_for_services,
};
