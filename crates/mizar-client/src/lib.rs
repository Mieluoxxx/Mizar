//! 前端无关的 Mizar 客户端宿主运行时与官方扩展插件集。
//!
//! 供 `mizar-tui` 与桌面端（Merak）共同复用，严禁引入 `ratatui`、`crossterm` 或 `mizar-tui-ui`。

pub mod approval;
pub mod events;
pub use mizar_image as images;
pub mod local_bash;
pub mod mentions;
pub mod session_view;
pub mod settings;
pub mod workspace;

pub use approval::{
    ApprovalGateConfig, ApprovalGatePlugin, ApprovalSnapshot, register_approval_hook,
    register_approval_hook_with, tool_needs_approval,
};
pub use events::{
    ActiveTask, ActiveTaskKind, LoopClientMsg, PendingPrompt, RunController, WaitServicesError,
    subscribe_session_events, subscribe_session_events_into, to_ui_log_record, to_ui_status,
    to_ui_step_outcome, to_ui_turn_end_reason, wait_for_services,
};
pub use images::{
    ImageProcessError, JPEG_QUALITY, MAX_IMAGE_BASE64_BYTES, MAX_IMAGE_EDGE, MAX_IMAGE_PIXELS,
    ProcessedImage, base64_encoded_len, media_type_for_format, process_image_bytes,
    sniff_image_format,
};
pub use local_bash::{execute_local_bash_for_session, next_local_bash_call_id};
pub use mentions::{
    FileMentionsConfig, FileMentionsPlugin, MENTION_MAX_DIR_ENTRIES, MENTION_MAX_FILE_CHARS,
    MENTION_MAX_TOTAL_CHARS, MentionExpansionResult, expand_text_mentions,
    mentions_resolve_any_image, register_mention_hook, register_mention_hook_with,
};
pub use session_view::{
    SelectionSources, build_session_info_report, build_session_info_report_at,
    collect_session_picker_items, find_model_in_provider, latest_workspace_selection,
    project_fork_view, project_resume_view, project_session_tree, project_session_view,
    resolve_model_switch, resolve_selection, to_ui_selection,
};
pub use settings::{
    ModuleToggleOutcome, SettingsViewSnapshot, apply_runtime_setting_change,
    build_settings_entries, persist_setting_change, persist_setting_change_to, toggle_spine_module,
};
pub use workspace::{
    WorkspacePanelSnapshot, collect_git_summary, collect_instructions_summary,
    collect_mcps_summary, collect_tools_summary, collect_trajectory_findings,
    collect_user_skill_names, collect_workspace_panel_snapshot,
};
