//! 基于对话主题的会话标题自动生成模块。
//!
//! 监听每轮用户输入事件（`AgentEvent::TurnInput`），仅提取 `SessionEvent::User` 消息并在后台
//! 异步调用轻量模型生成 `MMDD｜TYPE｜Topic` 标题（首轮触发、随后每隔 N 轮刷新，尊重人工改名）。

pub mod format;
pub mod prompt;
pub mod service;

pub use format::{
    ParsedModelTitle, SessionTitleType, format_session_title, parse_model_ref,
    parse_model_title_output,
};
pub use prompt::{build_title_messages, extract_user_messages};
pub use service::{
    DEFAULT_REFRESH_INTERVAL_TURNS, SessionTitleConfig, SessionTitleError, SessionTitleRuntime,
};
