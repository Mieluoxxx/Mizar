//! 终端交互式 TUI 客户端宿主：桥接纯前端工作台 `mizar-tui-ui`、`mizar-client` 与 `MizarRuntime` 后端。

pub mod clipboard;
pub mod exec;
pub mod rpc_host;
pub mod runner;
pub mod serve;
pub mod update;

pub use mizar_client::events::{
    ActiveTask, ActiveTaskKind, LoopClientMsg, RunController, steer_session,
    subscribe_session_events, subscribe_session_events_into, to_ui_status, to_ui_step_outcome,
    to_ui_turn_end_reason,
};
pub use mizar_client::logging::{
    TracingBridgeGuard, install_silent_tracing, install_tracing_handler,
};
pub use mizar_client::session_view::{
    SelectionSources, collect_session_picker_items, find_model_in_provider,
    latest_workspace_selection, project_fork_view, project_resume_view, project_session_tree,
    project_session_view, resolve_model_switch, resolve_selection, to_ui_selection,
};
pub use mizar_client::settings::{
    ModuleToggleOutcome, SettingsViewSnapshot, apply_runtime_setting_change,
    persist_setting_change, persist_setting_change_to, toggle_spine_module,
};
pub use mizar_client::{images, mentions};

pub mod emphasis {
    pub use crate::runner::{install_tui_emphasis, mount_reply_guidance_section};
    pub use mizar_tui_ui::emphasis::*;
}

pub use mizar_tui_ui::{
    AppAction, AppState, ApprovalMode, ChatItem, EmphasisConfig, HostExtensionFlags,
    LaunchpadQuickAction, NoticeLevel, PendingSettingsModelPick, PromptQueue, ScreenMode,
    SessionPickerItem, SessionReplayItem, SessionSelection, SessionViewData, TerminalColors, Theme,
    ThemePreset, ToolState, TuiFeatures, UiAgentStatus, UiEvent, UiLogLevel, UiLogRecord,
    UiStepOutcome, UiTurnEndReason, build_item_lines_for_scrollback,
    build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback, render,
    turn_end_program_status,
};
pub use mizar_tui_ui::{
    app, editor, markdown, overlay, reveal, selection, side_panel, terminal, theme, timeline_rail,
    trajectory_scene, ui,
};
pub use rpc_host::{RpcHostError, run_rpc_host_cli};
pub use runner::{
    ResumeTarget, TuiConfig, TuiExitSummary, build_session_info_report,
    build_session_info_report_at, build_settings_entries, format_resume_command,
    install_tui_emphasis, mount_reply_guidance_section, resume_session_into_app, run_tui,
    run_tui_with_summary,
};
pub use serve::{ServeConfig, ServeError, ServeHandle, run_serve_cli, start_serve};
