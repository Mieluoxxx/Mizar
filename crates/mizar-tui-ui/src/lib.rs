//! 纯终端工作台 UI 层：双栏工作台、时间轴导轨、自适应 OKLab 色彩、`pi-emphasis` 视觉语法与词级 Diff。
//!
//! 本 crate 不依赖 `tokio`、`cordis`、`mizar-session`、`mizar-agent-loop` 或任何模型/工具运行时。

pub mod app;
pub mod editor;
pub mod emphasis;
pub mod markdown;
pub mod mentions;
pub mod overlay;
pub mod reveal;
pub mod selection;
pub mod side_panel;
pub mod terminal;
pub mod theme;
pub mod timeline_rail;
pub mod trajectory_scene;
pub mod ui;

pub use app::{
    AppAction, AppState, ApprovalMode, ChatItem, LaunchpadQuickAction, NoticeLevel,
    SessionReplayItem, SessionViewData, ToolState, UiAgentStatus, UiEvent, UiLogLevel,
    UiLogRecord, UiTurnEndReason,
};
pub use emphasis::EmphasisConfig;
pub use terminal::ScreenMode;
pub use theme::{TerminalColors, Theme, ThemePreset};
pub use ui::{build_item_lines_for_scrollback, build_item_lines_for_scrollback_at, render};
