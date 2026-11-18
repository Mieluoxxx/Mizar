//! 终端交互式 TUI 客户端：双栏工作台、时间轴导轨、自适应 OKLab 色彩、`pi-emphasis` 视觉语法与词级 Diff。

pub mod app;
pub mod editor;
pub mod emphasis;
pub mod markdown;
pub mod overlay;
pub mod reveal;
pub mod runner;
pub mod selection;
pub mod side_panel;
pub mod terminal;
pub mod theme;
pub mod timeline_rail;
pub mod trajectory_scene;
pub mod ui;

pub use app::{
    AppAction, AppState, ApprovalMode, ChatItem, LaunchpadQuickAction, NoticeLevel, ToolState,
    UiEvent,
};
pub use emphasis::EmphasisConfig;
pub use runner::{
    ResumeTarget, TuiConfig, TuiExitSummary, format_resume_command, run_tui,
    run_tui_with_summary, subscribe_session_events, wait_for_services,
};
pub use terminal::ScreenMode;
pub use theme::{TerminalColors, Theme, ThemePreset};
pub use ui::{build_item_lines_for_scrollback, render};
