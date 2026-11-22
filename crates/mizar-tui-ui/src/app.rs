//! TUI 核心状态机：会话投影、主题与 `pi-emphasis` 状态、插件开关、队列收口与生命周期管理。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub use mizar_client_core::tool_format::{
    abbreviate_home_path, abbreviate_home_with, bash_inline_meta, bash_scope_label,
    clean_bash_output_lines, count_diff_stats, extract_inline_meta, format_bash_display,
    format_diff_stat, format_expanded_tool_args, format_tokens, is_path_arg_tool,
    is_read_only_compact_tool, short_id, single_line_text, summarize_live_preview,
    summarize_tool_args, summarize_tool_result, truncate_display,
};
pub use mizar_client_core::{
    ActivityState, ApprovalMode, BashInputMode, LocalBashOutput, NoticeLevel, PromptQueue,
    QueuedPrompt, SessionReplayItem, SessionSelection, SessionViewData, ToolExtraMeta, ToolState,
    UiAgentStatus, UiLogLevel, UiLogRecord, UiStepOutcome, UiTurnEndReason,
};
use mizar_llm_types::{
    ContentBlock, ImageAttachment, MAX_IMAGES_PER_MESSAGE, ModelInfo, ProviderInfo,
    ReasoningEffort, TokenUsage,
};
use mizar_trajectory_core::{TimelineMode, TrajectoryReport};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use serde_json::Value;

use crate::editor::TextInput;
use crate::emphasis::EmphasisConfig;
use crate::features::TuiFeatures;
use crate::mentions::FILE_MENTIONS_SOURCE;
use crate::overlay::{
    ApprovalDecision, OverlayState, SessionTreeEntry, SettingEntry, WorkspaceFileCache,
};
use crate::reveal::RevealController;
use crate::selection::{SelectionGeometry, SelectionState};
use crate::side_panel::{ContextBreakdownSnapshot, FileTouchRow, SidePanelData, SidePanelState};
use crate::terminal::{ProgramStatusState, ScreenMode, emit_program_status};
use crate::theme::{TerminalColors, Theme, ThemePreset};
use crate::timeline_rail::{RailGeometry, TimelineSnapshot};
use crate::tps::TpsLedger;
use crate::trajectory_scene::TrajectorySceneState;

mod cache;
mod events;
mod input;

pub(crate) use cache::StreamingMarkdownPrefixCache;
pub use cache::{CachedItemLines, ItemRenderKey, MAX_CACHED_RENDER_ITEMS, TranscriptRenderCache};

/// 由后端异步事件与终端外部状态投递给 UI 主循环的消息。
pub type UiEvent = mizar_client_core::ClientEvent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatItem {
    User {
        text: String,
    },
    Assistant {
        reasoning: String,
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        args_summary: String,
        status: ToolState,
        output_preview: String,
    },
    Notice {
        level: NoticeLevel,
        text: String,
    },
}

/// 将 UI 轮次终态映射为 OSC 7501 终端程序状态与说明，保证非正常收敛不误标为 `Done`。
pub fn turn_end_program_status(reason: &UiTurnEndReason) -> (ProgramStatusState, &'static str) {
    match reason {
        UiTurnEndReason::Completed => (ProgramStatusState::Done, "Turn completed"),
        UiTurnEndReason::Truncated => (ProgramStatusState::Idle, "Output truncated"),
        UiTurnEndReason::MaxSteps => (ProgramStatusState::Idle, "Max steps reached"),
        UiTurnEndReason::Cancelled => (ProgramStatusState::Idle, "Cancelled"),
        UiTurnEndReason::Error { .. } => (ProgramStatusState::Error, "Request failed"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchpadQuickAction {
    ModelPicker,
    ThemePicker,
    SkillPicker,
    TrajectoryScene,
    ToggleSidePanel,
}

/// 从 `/settings` 暂跳到 `ModelPicker` 选择配置项模型时保存的现场（避免污染当前会话活跃模型）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingSettingsModelPick {
    pub setting_key: String,
    pub items: Vec<SettingEntry>,
    pub query: String,
    pub selected: usize,
    pub save_path_hint: String,
}

/// UI 层处理按键/提交后需要 Runner 执行的外部动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppAction {
    None,
    PasteClipboard,
    SubmitPrompt(String),
    SubmitPromptWithImages {
        text: String,
        images: Vec<ImageAttachment>,
    },
    SteerPrompt(String),
    SteerPromptWithImages {
        text: String,
        images: Vec<ImageAttachment>,
    },
    RunLocalBash {
        command: String,
        exclude_from_context: bool,
    },
    CancelRun,
    CompactSession(Option<String>),
    NewSession,
    OpenSessionPicker,
    ResumeSession(String),
    RenameSession(Option<String>),
    ShowStats,
    ShowSessionInfo,
    ShowTools,
    ShowMcp,
    ShowSkills,
    ShowInstructions,
    OpenSkillPicker,
    OpenSettings(Option<String>),
    ApplySetting {
        key: String,
        value: String,
    },
    OpenSessionTree,
    ForkSessionAtSeq(u64),
    OpenExternalEditor,
    SwitchScreenMode(ScreenMode),
    SetMouseCapture(bool),
    ExportTrajectory(Option<PathBuf>),
    ShowOrSwitchModel(Option<String>),
    RefreshSidePanelData,
    RefreshSidePanelTools,
    OpenTrajectoryScene(TimelineMode),
    Quit,
}

pub struct AppState {
    pub workspace: PathBuf,
    pub session_id: String,
    pub session_title: Option<String>,
    pub items: Vec<ChatItem>,
    pub tool_extras: HashMap<String, ToolExtraMeta>,
    pub item_cache: TranscriptRenderCache,

    pub input: TextInput,
    pub queue: PromptQueue,
    pub activity: ActivityState,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub providers: Vec<ProviderInfo>,
    pub user_skill_names: Vec<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub show_thinking: bool,
    pub verbose_expanded: bool,
    pub approval_mode: ApprovalMode,
    pub always_allowed_tools: HashSet<String>,

    // 滚动与虚拟化状态
    pub scroll_from_top: u16,
    pub auto_scroll: bool,
    pub last_viewport_height: u16,
    pub last_total_lines: u16,

    // 用量与性能仪表
    pub total_usage: TokenUsage,
    /// 最近一次完成步骤的完整 prompt 体积（含缓存命中/写入），用于上下文占用估算。
    pub last_prompt_tokens: Option<u64>,
    pub last_step_start_ms: Option<u128>,
    pub last_ttft_ms: Option<u64>,
    pub ttft_history: Vec<u64>,
    pub tps: TpsLedger,
    pub tps_history: Vec<f64>,
    pub spinner_tick: usize,
    pub should_quit: bool,

    // 双模视图与鼠标开关
    pub screen_mode: ScreenMode,
    pub inline_committed_items: usize,
    pub inline_welcome_committed: bool,
    pub mouse_capture_enabled: bool,
    pub launchpad_hit_rects: Vec<(LaunchpadQuickAction, Rect)>,

    // 主题、视觉语法与插件开关
    pub theme_preset: ThemePreset,
    pub probed_colors: TerminalColors,
    pub theme: Theme,
    pub emphasis: EmphasisConfig,
    pub features: TuiFeatures,
    pub reveal: RevealController,
    pub effort_ignition_at: Option<Instant>,

    // 选区与剪贴板
    pub selection: SelectionState,
    pub selection_geom: SelectionGeometry,
    pub last_buffer: Option<Buffer>,
    pub flash_toast: Option<(String, Instant)>,

    // 侧栏、导轨、轨迹场景与浮层栈
    pub side_panel: SidePanelState,
    pub side_panel_data: SidePanelData,
    pub touched_files: Vec<FileTouchRow>,
    pub anomalies: Vec<String>,
    pub timeline_snap: TimelineSnapshot,
    pub rail_geo: Option<RailGeometry>,
    pub hovered_rail_row: Option<u16>,
    pub trajectory_scene: TrajectorySceneState,
    pub overlay: OverlayState,
    pub pending_settings_model_pick: Option<PendingSettingsModelPick>,
    pub file_cache: WorkspaceFileCache,
    dismissed_mention: Option<(usize, String)>,

    // 确认退出 / 终端状态记忆
    pub terminal_focused: bool,
    last_ctrl_c_at: Option<Instant>,
    /// 最近一次回合结束原因，供状态栏与排队消息调度区分完成/截断/步数上限/中断/失败。
    pub last_turn_end_reason: Option<UiTurnEndReason>,
    /// 最近一次 step 结束编号与结果，用于截断提示定位具体步骤。
    pub last_step_outcome: Option<(usize, UiStepOutcome)>,
    /// 最近一次上报的终端 ProgramStatus，用于测试直接断言非 Completed 终态不产生 Done。
    pub last_program_status: Option<(ProgramStatusState, Option<String>)>,
    /// 最近一条连续 Prompt Cache 保活通知的 `(item_idx, fold_count)`，用于折叠防刷屏。
    last_cache_warming_notice: Option<(usize, usize)>,
    /// 当前活跃会话正在后台运行的 Subagent 数量。
    pub running_subagents: usize,
}

impl AppState {
    pub fn new(
        workspace: PathBuf,
        session_id: impl Into<String>,
        providers: Vec<ProviderInfo>,
        provider: Option<String>,
        model: Option<String>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        let theme_preset = ThemePreset::Auto;
        let probed_colors = TerminalColors::default();
        let theme = Theme::from_preset(theme_preset, Some(&probed_colors));
        Self {
            workspace,
            session_id: session_id.into(),
            session_title: None,
            items: Vec::new(),
            tool_extras: HashMap::new(),
            item_cache: TranscriptRenderCache::default(),
            input: TextInput::new(),
            queue: PromptQueue::new(),
            activity: ActivityState::Idle,
            provider,
            model,
            providers,
            user_skill_names: Vec::new(),
            reasoning_effort,
            show_thinking: true,
            verbose_expanded: false,
            approval_mode: ApprovalMode::Auto,
            always_allowed_tools: HashSet::new(),
            scroll_from_top: 0,
            auto_scroll: true,
            last_viewport_height: 0,
            last_total_lines: 0,
            total_usage: TokenUsage::default(),
            last_prompt_tokens: None,
            last_step_start_ms: None,
            last_ttft_ms: None,
            ttft_history: Vec::new(),
            tps: TpsLedger::default(),
            tps_history: Vec::new(),
            spinner_tick: 0,
            should_quit: false,
            screen_mode: ScreenMode::Fullscreen,
            inline_committed_items: 0,
            inline_welcome_committed: false,
            mouse_capture_enabled: true,
            launchpad_hit_rects: Vec::new(),
            theme_preset,
            probed_colors,
            theme,
            emphasis: EmphasisConfig::from_env(),
            features: TuiFeatures::default(),
            reveal: RevealController::new(true),
            effort_ignition_at: None,
            selection: SelectionState::default(),
            selection_geom: SelectionGeometry::default(),
            last_buffer: None,
            flash_toast: None,
            side_panel: SidePanelState::default(),
            side_panel_data: SidePanelData {
                context: ContextBreakdownSnapshot {
                    context_window: 128_000,
                    ..Default::default()
                },
                ..Default::default()
            },
            touched_files: Vec::new(),
            anomalies: Vec::new(),
            timeline_snap: TimelineSnapshot::default(),
            rail_geo: None,
            hovered_rail_row: None,
            trajectory_scene: TrajectorySceneState::default(),
            overlay: OverlayState::default(),
            pending_settings_model_pick: None,
            file_cache: WorkspaceFileCache::default(),
            dismissed_mention: None,
            terminal_focused: true,
            last_ctrl_c_at: None,
            last_turn_end_reason: None,
            last_step_outcome: None,
            last_program_status: None,
            last_cache_warming_notice: None,
            running_subagents: 0,
        }
    }

    pub fn with_features(mut self, features: TuiFeatures) -> Self {
        self.set_features(features);
        self
    }

    /// 运行时更新插件化特性开关，并同步控制 `emphasis`、侧栏、轨迹场景与选区状态。
    pub fn set_features(&mut self, features: TuiFeatures) {
        self.features = features;
        self.emphasis.enabled = features.emphasis && EmphasisConfig::from_env().enabled;
        if !features.side_panel && self.side_panel.open {
            self.side_panel.open = false;
            self.side_panel.dragging_divider = false;
            self.side_panel.hovered_divider = false;
            self.selection.clear();
        }
        if !features.trajectory_view && self.trajectory_scene.open {
            self.trajectory_scene.open = false;
        }
        if !features.selection {
            self.selection.clear();
        }
        if !features.timeline_rail {
            self.rail_geo = None;
            self.hovered_rail_row = None;
        }
        self.item_cache.clear();
    }

    pub fn is_busy(&self) -> bool {
        matches!(self.activity, ActivityState::Agent { .. })
    }

    pub fn is_bash_running(&self) -> bool {
        matches!(self.activity, ActivityState::LocalBash)
    }

    pub fn is_compacting(&self) -> bool {
        matches!(
            self.activity,
            ActivityState::Compacting { .. }
                | ActivityState::Agent {
                    compaction_status: Some(_),
                    ..
                }
        )
    }

    pub fn is_running_activity(&self) -> bool {
        !matches!(self.activity, ActivityState::Idle)
    }

    pub fn compaction_status(&self) -> Option<&str> {
        match &self.activity {
            ActivityState::Compacting { status } => Some(status.as_str()),
            ActivityState::Agent {
                compaction_status: Some(status),
                ..
            } => Some(status.as_str()),
            _ => None,
        }
    }

    pub fn set_busy_state(&mut self, busy: bool) {
        if busy {
            if !matches!(self.activity, ActivityState::Agent { .. }) {
                self.activity = ActivityState::Agent {
                    streaming_assistant_open: false,
                    compaction_status: None,
                };
            }
        } else if matches!(self.activity, ActivityState::Agent { .. }) {
            self.activity = ActivityState::Idle;
        }
    }

    pub fn set_local_bash_state(&mut self) {
        self.activity = ActivityState::LocalBash;
    }

    pub fn set_compacting_state(&mut self, status: impl Into<String>) {
        let status = status.into();
        match &mut self.activity {
            ActivityState::Agent {
                compaction_status, ..
            } => *compaction_status = Some(status),
            _ => self.activity = ActivityState::Compacting { status },
        }
    }

    pub fn set_idle_state(&mut self) {
        self.activity = ActivityState::Idle;
    }

    pub(super) fn close_streaming_assistant(&mut self) {
        if let ActivityState::Agent {
            streaming_assistant_open,
            ..
        } = &mut self.activity
        {
            *streaming_assistant_open = false;
        }
    }

    pub(super) fn promote_next_approval(&mut self) -> bool {
        if let Some(tool_name) = self.overlay.promote_next_approval() {
            self.record_program_status(
                ProgramStatusState::Blocked,
                Some("permission"),
                Some(&format!("Approve {tool_name}")),
            );
            true
        } else {
            false
        }
    }

    pub(super) fn record_touched_file(&mut self, name: &str, arguments: &str) {
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(arguments)
            && let Some(p) = map
                .get("path")
                .or_else(|| map.get("file"))
                .and_then(Value::as_str)
        {
            let op = name.to_string();
            if !self.touched_files.iter().any(|f| f.path == p && f.op == op) {
                self.touched_files.push(FileTouchRow {
                    path: p.to_string(),
                    op,
                });
            }
        }
    }

    /// 记录并向终端发送 ProgramStatus（供无头测试与状态栏统一观察）。
    pub(super) fn record_program_status(
        &mut self,
        state: ProgramStatusState,
        kind: Option<&str>,
        message: Option<&str>,
    ) {
        self.last_program_status = Some((state, message.map(str::to_string)));
        emit_program_status(state, kind, message);
    }

    pub fn apply_terminal_colors(&mut self, probed: TerminalColors) {
        self.probed_colors = probed;
        self.theme = Theme::from_preset(self.theme_preset, Some(&self.probed_colors));
        self.item_cache.clear();
    }

    pub fn set_theme_preset(&mut self, preset: ThemePreset) {
        self.theme_preset = preset;
        self.theme = Theme::from_preset(preset, Some(&self.probed_colors));
        self.item_cache.clear();
    }

    pub fn set_flash(&mut self, msg: impl Into<String>) {
        self.flash_toast = Some((msg.into(), Instant::now()));
    }

    pub fn active_flash(&mut self) -> Option<&str> {
        if let Some((_, at)) = self.flash_toast
            && at.elapsed() > Duration::from_secs(2)
        {
            self.flash_toast = None;
        }
        self.flash_toast.as_ref().map(|(s, _)| s.as_str())
    }

    pub fn active_model_label(&self) -> String {
        let p = self
            .provider
            .as_deref()
            .or_else(|| self.providers.first().map(|p| p.id.as_str()));
        let m = self.model.as_deref().or_else(|| {
            let target = p?;
            self.providers
                .iter()
                .find(|info| info.id == target)
                .and_then(|info| info.models.first().map(|m| m.id.as_str()))
        });
        match (p, m) {
            (Some(p), Some(m)) if p != "default" => format!("{p}/{m}"),
            (_, Some(m)) => m.to_string(),
            (Some(p), None) => p.to_string(),
            (None, None) => "default".to_string(),
        }
    }

    pub fn active_model_info(&self) -> Option<&ModelInfo> {
        let p = self.provider.as_deref();
        let m = self.model.as_deref();
        for prov in &self.providers {
            if p.is_some_and(|pid| pid != prov.id) {
                continue;
            }
            for model in &prov.models {
                if m.is_none() || m == Some(model.id.as_str()) {
                    return Some(model);
                }
            }
        }
        None
    }

    pub fn active_model_supports_images(&self) -> bool {
        self.active_model_info()
            .is_none_or(ModelInfo::supports_images)
    }

    pub fn insert_pasted_image(
        &mut self,
        summary_suffix: Option<&str>,
        image: ImageAttachment,
    ) -> Option<String> {
        if self.input.image_count() >= MAX_IMAGES_PER_MESSAGE {
            let msg = format!(
                "单条消息最多附加 {MAX_IMAGES_PER_MESSAGE} 张图片（已达上限，无法继续添加）"
            );
            self.set_flash(msg.clone());
            self.push_notice(NoticeLevel::Warn, msg);
            return None;
        }
        let token = self.input.insert_image(summary_suffix, image);
        self.sync_input_completion_overlay();
        self.set_flash(format!("已插入图片 {token}"));
        Some(token)
    }

    pub fn active_context_window(&self) -> u64 {
        self.active_model_info()
            .and_then(|model| model.context_window)
            .unwrap_or(128_000)
    }

    pub fn available_efforts(&self) -> Vec<ReasoningEffort> {
        if let Some(model) = self.active_model_info()
            && !model.reasoning_efforts.is_empty()
        {
            return model.reasoning_efforts.clone();
        }
        crate::overlay::all_efforts().to_vec()
    }

    pub fn bash_mode(&self) -> Option<BashInputMode> {
        if !self.features.local_bash {
            return None;
        }
        BashInputMode::from_input(self.input.text())
    }

    pub fn push_user_message(&mut self, text: &str) {
        self.items.push(ChatItem::User {
            text: text.to_string(),
        });
        self.close_streaming_assistant();
        self.scroll_to_bottom();
    }

    /// 在对话流插入一张处于运行态的本地 `!` / `!!` Bash 卡片。
    pub fn start_local_bash(
        &mut self,
        call_id: impl Into<String>,
        command: &str,
        exclude_from_context: bool,
    ) {
        let call_id = call_id.into();
        self.activity = ActivityState::LocalBash;
        self.reveal.snap_all();
        let args_json = serde_json::json!({ "command": command }).to_string();
        let args_summary = summarize_tool_args(&args_json);
        let inline_meta = bash_scope_label(exclude_from_context).to_string();
        self.tool_extras.insert(
            call_id.clone(),
            ToolExtraMeta {
                full_content: String::new(),
                diff_raw: None,
                full_args: args_json,
                inline_meta: Some(inline_meta),
                exclude_from_context,
            },
        );
        self.items.push(ChatItem::ToolCall {
            id: call_id,
            name: "bash".to_string(),
            args_summary,
            status: ToolState::Running,
            output_preview: String::new(),
        });
        let idx = self.items.len() - 1;
        self.item_cache
            .invalidate_tool_neighborhood(&self.items, idx);
        self.scroll_to_bottom();
    }

    pub fn open_model_selector(&mut self) -> AppAction {
        self.pending_settings_model_pick = None;
        if self.providers.iter().any(|p| !p.models.is_empty()) {
            let current_hint = match (self.provider.as_deref(), self.model.as_deref()) {
                (Some(p), Some(m)) => Some(format!("{p}/{m}")),
                (_, Some(m)) => Some(m.to_string()),
                _ => None,
            };
            self.overlay
                .open_model_picker(&self.providers, current_hint.as_deref());
            AppAction::None
        } else {
            AppAction::ShowOrSwitchModel(None)
        }
    }

    pub fn refresh_context_breakdown(&mut self) {
        let mut user_chars = 0usize;
        let mut asst_chars = 0usize;
        let mut reason_chars = 0usize;
        let mut tool_chars = 0usize;
        for item in &self.items {
            match item {
                ChatItem::User { text } => user_chars += text.len(),
                ChatItem::Assistant { reasoning, text } => {
                    reason_chars += reasoning.len();
                    asst_chars += text.len();
                }
                ChatItem::ToolCall {
                    id,
                    args_summary,
                    output_preview,
                    ..
                } => {
                    let extra = self.tool_extras.get(id);
                    if extra.is_some_and(|e| e.exclude_from_context) {
                        continue;
                    }
                    let body_len = extra
                        .map(|e| e.full_content.len())
                        .unwrap_or(output_preview.len());
                    tool_chars += args_summary.len() + body_len;
                }
                ChatItem::Notice { .. } => {}
            }
        }
        // ~3.5 chars/token 估算比例，总占用以实测 total_usage 为主
        let est = |chars: usize| -> u64 { (chars as u64).div_ceil(4) };
        let used_est =
            (est(user_chars) + est(asst_chars) + est(reason_chars) + est(tool_chars)).max(1);
        // 上下文占用优先用最近一次请求的实测 prompt 全量（含缓存命中与写入），
        // 缺失时才以字符估算为基准；`scale` 仅用于把各段字符比例拉回字数。
        let measured = self.last_prompt_tokens.filter(|v| *v > 0);
        let scale = match measured {
            Some(v) => v as f64 / used_est as f64,
            None => 1.0,
        };
        self.side_panel_data.context = ContextBreakdownSnapshot {
            system_tokens: match measured {
                Some(v) => (v / 10).max(512),
                None => 512,
            },
            user_tokens: ((est(user_chars) as f64) * scale).round() as u64,
            assistant_tokens: ((est(asst_chars) as f64) * scale).round() as u64,
            reasoning_tokens: ((est(reason_chars) as f64) * scale).round() as u64,
            tool_tokens: ((est(tool_chars) as f64) * scale).round() as u64,
            context_window: self.active_context_window(),
            tps: self.tps.rate(),
            ttft_ms: self.last_ttft_ms,
            tps_history: self.tps_history.clone(),
            ttft_history: self.ttft_history.clone(),
            cache_read_tokens: self.total_usage.cache_read_tokens.unwrap_or(0),
            cache_write_tokens: self.total_usage.cache_write_tokens.unwrap_or(0),
            input_tokens: self.total_usage.input_tokens,
            output_tokens: self.total_usage.output_tokens,
            total_tokens: self.total_usage.grand_total(),
            last_prompt_tokens: measured,
        };
        self.side_panel_data.files = self.touched_files.clone();
        self.side_panel_data.findings = self.anomalies.clone();
    }

    pub fn push_notice(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.items.push(ChatItem::Notice {
            level,
            text: text.into(),
        });
        self.last_cache_warming_notice = None;
        self.close_streaming_assistant();
    }

    /// 写入 Prompt Cache 保活通知；若末尾紧邻同源通知且尚未在 Inline 模式落屏，则原地折叠更新计数。
    pub fn push_cache_warming_notice(&mut self, text: impl Into<String>) {
        let text = text.into();
        if let Some((idx, count)) = self.last_cache_warming_notice
            && idx + 1 == self.items.len()
            && idx >= self.inline_committed_items
            && let Some(ChatItem::Notice {
                text: slot_text, ..
            }) = self.items.get_mut(idx)
        {
            let next_count = count + 1;
            *slot_text = format!("{text}（已连续折叠 ×{next_count}）");
            self.item_cache.invalidate_item(idx);
            self.last_cache_warming_notice = Some((idx, next_count));
            return;
        }
        let idx = self.items.len();
        self.items.push(ChatItem::Notice {
            level: NoticeLevel::Info,
            text,
        });
        self.last_cache_warming_notice = Some((idx, 1));
        self.close_streaming_assistant();
    }

    pub fn open_trajectory_with_report(&mut self, report: Option<TrajectoryReport>) {
        if !self.features.trajectory_view {
            return;
        }
        self.trajectory_scene.report = report;
        self.trajectory_scene.open = true;
    }

    fn apply_effort(&mut self, effort: ReasoningEffort, with_notice: bool) {
        let changed = self.reasoning_effort != Some(effort);
        self.reasoning_effort = Some(effort);
        if changed
            && matches!(
                effort,
                ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max
            )
        {
            self.effort_ignition_at = Some(Instant::now());
        }
        let label = format!("{effort:?}").to_lowercase();
        self.set_flash(format!("思考等级：{label}"));
        if with_notice {
            self.push_notice(NoticeLevel::Info, format!("推理强度已切换为：{label}"));
        }
    }

    pub fn set_effort(&mut self, effort: ReasoningEffort) {
        self.apply_effort(effort, true);
    }

    pub fn cycle_reasoning_effort(&mut self) -> ReasoningEffort {
        let efforts = self.available_efforts();
        let next = match self.reasoning_effort {
            Some(cur) => {
                if let Some(pos) = efforts.iter().position(|&e| e == cur) {
                    efforts[(pos + 1) % efforts.len()]
                } else {
                    let order = crate::overlay::all_efforts();
                    let cur_rank = order.iter().position(|&e| e == cur).unwrap_or(0);
                    efforts
                        .iter()
                        .copied()
                        .find(|e| order.iter().position(|o| o == e).unwrap_or(0) > cur_rank)
                        .unwrap_or(efforts[0])
                }
            }
            None => efforts
                .iter()
                .copied()
                .find(|&e| e != ReasoningEffort::Off)
                .unwrap_or(efforts[0]),
        };
        self.apply_effort(next, false);
        next
    }

    fn page_step(&self) -> u16 {
        self.last_viewport_height.saturating_sub(2).max(3)
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.selection.clear();
        let max_scroll = self
            .last_total_lines
            .saturating_sub(self.last_viewport_height);
        if max_scroll == 0 {
            self.auto_scroll = true;
            self.scroll_from_top = 0;
            return;
        }
        let cur = if self.auto_scroll {
            max_scroll
        } else {
            self.scroll_from_top.min(max_scroll)
        };
        self.auto_scroll = false;
        self.scroll_from_top = cur.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: u16) {
        self.selection.clear();
        let max_scroll = self
            .last_total_lines
            .saturating_sub(self.last_viewport_height);
        let next = self.scroll_from_top.saturating_add(lines);
        if next >= max_scroll {
            self.auto_scroll = true;
            self.scroll_from_top = max_scroll;
        } else {
            self.scroll_from_top = next;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.selection.clear();
        self.auto_scroll = true;
        self.scroll_from_top = self
            .last_total_lines
            .saturating_sub(self.last_viewport_height);
    }

    pub fn effective_scroll(&mut self, viewport_height: u16, total_lines: u16) -> u16 {
        self.last_viewport_height = viewport_height;
        self.last_total_lines = total_lines;
        let max_scroll = total_lines.saturating_sub(viewport_height);
        if self.auto_scroll {
            self.scroll_from_top = max_scroll;
        } else if max_scroll == 0 {
            self.scroll_from_top = 0;
            self.auto_scroll = true;
        } else if self.scroll_from_top > max_scroll {
            self.scroll_from_top = max_scroll;
        }
        self.scroll_from_top
    }

    pub fn reset_for_new_session(
        &mut self,
        session_id: impl Into<String>,
        session_title: Option<String>,
    ) {
        self.session_id = session_id.into();
        self.session_title = session_title;
        self.pending_settings_model_pick = None;
        self.overlay.resolve_all_approvals(ApprovalDecision::Deny);
        self.items.clear();
        self.tool_extras.clear();
        self.item_cache.clear();
        self.touched_files.clear();
        self.anomalies.clear();
        self.queue.clear();
        self.activity = ActivityState::Idle;
        self.always_allowed_tools.clear();
        self.reveal.clear();
        self.selection.clear();
        self.total_usage = TokenUsage::default();
        self.last_prompt_tokens = None;
        self.last_step_start_ms = None;
        self.last_ttft_ms = None;
        self.ttft_history.clear();
        self.tps.reset();
        self.tps_history.clear();
        self.inline_committed_items = 0;
        self.inline_welcome_committed = false;
        self.scroll_from_top = 0;
        self.auto_scroll = true;
        self.dismissed_mention = None;
        self.last_turn_end_reason = None;
        self.last_step_outcome = None;
        self.last_program_status = None;
        self.last_cache_warming_notice = None;
        self.running_subagents = 0;
        self.refresh_context_breakdown();
        self.push_notice(
            NoticeLevel::Info,
            format!("已开启新会话 ({})", short_id(&self.session_id)),
        );
    }

    /// 从投影后的纯会话视图重建界面对话流（供 `/tree` 分叉回溯与 `--resume` / `--continue` 使用）。
    pub fn restore_session_view(&mut self, view: SessionViewData) {
        self.reset_for_new_session(view.session_id, view.session_title);
        // 清理 reset 推送的默认欢迎通知，由真实历史替代
        self.items.clear();
        self.total_usage = view.total_usage;
        self.last_prompt_tokens = view.last_prompt_tokens;

        for item in view.items {
            match item {
                SessionReplayItem::User { text } => {
                    self.items.push(ChatItem::User { text });
                }
                SessionReplayItem::ContextNotice { source, text } => {
                    self.items.push(ChatItem::Notice {
                        level: NoticeLevel::Info,
                        text: summarize_context_notice_for_replay(&source, &text),
                    });
                }
                SessionReplayItem::Notice { level, text } => {
                    self.items.push(ChatItem::Notice { level, text });
                }
                SessionReplayItem::Compaction {
                    summary,
                    first_kept_seq,
                } => {
                    self.items.push(ChatItem::Notice {
                        level: NoticeLevel::Info,
                        text: format!("[上下文压缩（保留自 #{first_kept_seq}）]\n{summary}"),
                    });
                }
                SessionReplayItem::Assistant { blocks } => {
                    let mut reasoning = String::new();
                    let mut text = String::new();
                    let mut calls = Vec::new();
                    for b in blocks {
                        match b {
                            ContentBlock::Reasoning { text: thinking } => {
                                if !reasoning.is_empty() {
                                    reasoning.push('\n');
                                }
                                reasoning.push_str(&thinking);
                            }
                            ContentBlock::Text { text: t } => text.push_str(&t),
                            ContentBlock::ToolCall {
                                id,
                                name,
                                arguments,
                            } => calls.push((id, name, arguments)),
                        }
                    }
                    if !reasoning.is_empty() || !text.is_empty() {
                        self.items.push(ChatItem::Assistant { reasoning, text });
                    }
                    for (id, name, arguments) in calls {
                        let args_summary = summarize_tool_args(&arguments);
                        self.record_touched_file(&name, &arguments);
                        self.tool_extras.insert(
                            id.clone(),
                            ToolExtraMeta {
                                full_content: String::new(),
                                diff_raw: None,
                                full_args: arguments,
                                inline_meta: None,
                                exclude_from_context: false,
                            },
                        );
                        self.items.push(ChatItem::ToolCall {
                            id,
                            name,
                            args_summary,
                            status: ToolState::Running,
                            output_preview: String::new(),
                        });
                    }
                }
                SessionReplayItem::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                } => {
                    let mut matched_name = String::new();
                    if let Some(ChatItem::ToolCall {
                        name,
                        status,
                        output_preview,
                        ..
                    }) = self.items.iter_mut().rev().find(|it| match it {
                        ChatItem::ToolCall { id, .. } => id == &tool_call_id,
                        _ => false,
                    }) {
                        matched_name = name.clone();
                        *status = if is_error {
                            ToolState::Error
                        } else {
                            ToolState::Done
                        };
                        *output_preview =
                            summarize_tool_result(&matched_name, &content, None, is_error);
                    }
                    let inline_meta = (!matched_name.is_empty())
                        .then(|| extract_inline_meta(&matched_name, &content, None, is_error))
                        .flatten();
                    let ext = self.tool_extras.entry(tool_call_id).or_default();
                    ext.full_content = content;
                    ext.inline_meta = inline_meta;
                }
                SessionReplayItem::BashExecution {
                    command,
                    output,
                    exit_code,
                    cancelled,
                    exclude_from_context,
                } => {
                    let call_id = format!("bash-replay-{}", self.items.len());
                    let status = if cancelled {
                        ToolState::Blocked
                    } else if exit_code.unwrap_or(0) != 0 {
                        ToolState::Error
                    } else {
                        ToolState::Done
                    };
                    let display = format_bash_display(&output, exit_code, cancelled);
                    let inline_meta = bash_inline_meta(exclude_from_context, exit_code, cancelled);
                    self.apply_local_bash_completion(
                        call_id,
                        &command,
                        exclude_from_context,
                        display,
                        status,
                        inline_meta,
                    );
                }
            }
        }
        self.push_notice(NoticeLevel::Info, view.finish_notice);
        self.refresh_context_breakdown();
        self.scroll_to_bottom();
    }

    pub fn open_session_tree_with(&mut self, entries: Vec<SessionTreeEntry>) {
        if entries.is_empty() {
            self.push_notice(NoticeLevel::Info, "当前会话尚无事件记录");
            return;
        }
        self.overlay.open_session_tree(entries);
    }

    /// 入队并把选择器自动定位到新条目：消息一旦入队，就能立即用 ↑ 撤回编辑。
    pub(super) fn enqueue_prompt_with_images(
        &mut self,
        prompt: String,
        images: Vec<(String, ImageAttachment)>,
    ) {
        let _id = self.queue.enqueue(prompt, images);
        self.set_flash(format!(
            "已加入队列 (共 {} 条待发)，↑ 可撤回编辑",
            self.queue.len()
        ));
    }

    pub fn reject_unsupported_images_notice(&mut self) {
        let model = self.active_model_label();
        let msg = format!(
            "当前模型 {model} 不支持图片输入，请按 Ctrl+L 切换支持视觉的模型或删除图片占位符后重试"
        );
        self.set_flash(msg.clone());
        self.push_notice(NoticeLevel::Error, msg);
    }

    /// 出队唯一入口：`prepare_submit`、选择器与 `runner` 的自动续发都走这里，
    /// 两处各自 `remove(0)` 会重复发送。
    pub fn dequeue_prompt(&mut self, id: u64) -> Option<QueuedPrompt> {
        self.queue.dequeue_by_id(id)
    }

    /// 取出队首一条；`runner` 在上一轮成功收尾后调用它继续发送。
    pub fn dequeue_next(&mut self) -> Option<QueuedPrompt> {
        self.queue.dequeue_next()
    }

    /// 把队列项撤回输入框：草稿为空则直接填入，非空则与草稿交换。
    /// 交换而非覆盖，是为了不丢主人已经写下的字；两者互斥，保证同一段文字只在一处。
    pub(super) fn retract_queued(&mut self, id: u64) -> bool {
        if !self.queue.items().iter().any(|item| item.id == id) {
            return false;
        }
        let had_draft = !self.input.text().trim().is_empty() || self.input.has_images();
        let (draft_raw, draft_images) = if had_draft {
            self.input.take_with_images()
        } else {
            (String::new(), Vec::new())
        };
        let Some((text, images)) = self.queue.retract_by_id(id, &draft_raw, draft_images) else {
            return false;
        };
        self.input
            .replace_draft_from_queue_with_images(&text, &images);
        if had_draft {
            self.set_flash("已交换：草稿已存入队尾");
        } else {
            self.set_flash("已撤回队列项，Enter 重新发送");
        }
        self.sync_input_completion_overlay();
        true
    }

    /// 丢弃选中的队列项（不发送）。
    pub(super) fn discard_queued(&mut self, id: u64) -> bool {
        let dropped = self.queue.discard_by_id(id);
        if dropped {
            self.set_flash("已丢弃队列项");
        }
        dropped
    }

    /// 立即发送队首：打断当前回合，在 `RunFinished` 后直接发出。
    pub(super) fn steer_queue_head(&mut self) -> AppAction {
        let Some(item) = self.dequeue_next() else {
            return AppAction::None;
        };
        if !item.images.is_empty() && !self.active_model_supports_images() {
            self.input
                .replace_draft_from_queue_with_images(&item.text, &item.images);
            self.reject_unsupported_images_notice();
            return AppAction::None;
        }
        self.set_flash("⚡ 正在中断当前回合并立即发送队列消息…");
        if item.images.is_empty() {
            AppAction::SteerPrompt(item.text)
        } else {
            AppAction::SteerPromptWithImages {
                text: item.text,
                images: item.images.into_iter().map(|(_, img)| img).collect(),
            }
        }
    }

    /// 空闲时把队首当作普通消息发出（唤醒中断后滞留的队列）。
    pub(super) fn submit_queue_head(&mut self) -> AppAction {
        let Some(item) = self.dequeue_next() else {
            return AppAction::None;
        };
        if !item.images.is_empty() && !self.active_model_supports_images() {
            self.input
                .replace_draft_from_queue_with_images(&item.text, &item.images);
            self.reject_unsupported_images_notice();
            return AppAction::None;
        }
        self.push_user_message(&item.text);
        if item.images.is_empty() {
            AppAction::SubmitPrompt(item.text)
        } else {
            AppAction::SubmitPromptWithImages {
                text: item.text,
                images: item.images.into_iter().map(|(_, img)| img).collect(),
            }
        }
    }
}

fn summarize_context_notice_for_replay(source: &str, text: &str) -> String {
    if source.starts_with("subagent:") || source.starts_with("subagent-watchdog:") {
        let first_line = text
            .lines()
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("已返回结果");
        return format!("[{source}] {}", truncate_display(first_line, 100));
    }
    if source == "skill-catalog" {
        return "[skill-catalog] 已加载可用技能目录".to_string();
    }
    if source == FILE_MENTIONS_SOURCE {
        let mut refs = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if (trimmed.starts_with("<attached-file ")
                || trimmed.starts_with("<attached-directory ")
                || trimmed.starts_with("<attached-image "))
                && let Some(p_start) = trimmed.find("path=\"")
            {
                let after = &trimmed[p_start + 6..];
                if let Some(p_end) = after.find('"') {
                    let path = &after[..p_end];
                    if !path.is_empty() && !refs.iter().any(|r| r == path) {
                        refs.push(path.to_string());
                    }
                }
            }
        }
        return if refs.is_empty() {
            "[file-mentions] 已附加引用内容".to_string()
        } else {
            format!("[file-mentions] 已附加引用：{}", refs.join(", "))
        };
    }
    if source == "agent-instructions" {
        let mut paths = Vec::new();
        for line in text.lines() {
            let extracted = line
                .strip_prefix("Instructions from: ")
                .or_else(|| line.strip_prefix("Additional instructions from: "))
                .or_else(|| line.strip_prefix("Updated instructions from: "))
                .or_else(|| line.strip_prefix("Instructions removed: "));
            if let Some(rest) = extracted {
                let path_part = rest.split(" (").next().unwrap_or(rest).trim();
                if !path_part.is_empty() && !paths.iter().any(|p| p == path_part) {
                    paths.push(path_part.to_string());
                }
            }
        }
        return if paths.is_empty() {
            format!(
                "[{source}] {}",
                truncate_display(&text.replace('\n', " "), 96)
            )
        } else {
            format!("[{source}] 已加载指令：{}", paths.join(", "))
        };
    }
    if source == "environment-context" {
        let extract_tag = |tag: &str| -> Option<&str> {
            let open = format!("<{tag}>");
            let close = format!("</{tag}>");
            let rest = text.split_once(&open)?.1;
            let val = rest.split_once(&close)?.0.trim();
            (!val.is_empty()).then_some(val)
        };
        return match (extract_tag("cwd"), extract_tag("current_date")) {
            (Some(cwd), Some(date)) => format!("[{source}] {cwd} ({date})"),
            (Some(cwd), None) => format!("[{source}] {cwd}"),
            (None, Some(date)) => format!("[{source}] ({date})"),
            (None, None) => format!("[{source}] 已注入运行环境上下文"),
        };
    }
    format!("[{source}] {text}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crossterm::event::{
        KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    };
    use mizar_llm_types::{ModelInfo, ToolCall};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::overlay::{OverlayKind, SettingItemKind};

    #[test]
    fn streams_reasoning_tools_and_text_into_items() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-12345678",
            Vec::new(),
            Some("default".into()),
            Some("deepseek-chat".into()),
            None,
        );

        app.input.insert_str("现在几点");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("现在几点".into())
        );

        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 100 });
        app.apply_ui_event(UiEvent::ReasoningDelta {
            text: "先查时间".into(),
            ts_ms: 200,
        });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 280 });
        assert_eq!(app.last_ttft_ms, Some(180));

        app.apply_ui_event(UiEvent::ToolCall(ToolCall {
            id: "c1".into(),
            name: "get_time".into(),
            arguments: "{}".into(),
        }));
        app.apply_ui_event(UiEvent::ToolResult {
            call_id: "c1".into(),
            name: "get_time".into(),
            content: "2026/11/18 09:30:00".into(),
            is_error: false,
            blocked: false,
            details: None,
        });
        app.apply_ui_event(UiEvent::TextDelta {
            text: "现在是 09:30".into(),
            ts_ms: 300,
        });
        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 100,
            output_tokens: 25,
            ..Default::default()
        }));
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });

        assert!(!app.is_busy());
        assert_eq!(app.total_usage.input_tokens, 100);
        assert_eq!(app.total_usage.output_tokens, 25);
        assert_eq!(app.items.len(), 4);
        assert!(matches!(
            &app.items[2],
            ChatItem::ToolCall {
                status: ToolState::Done,
                ..
            }
        ));
        assert!(matches!(
            &app.items[3],
            ChatItem::Assistant { text, .. } if text == "现在是 09:30"
        ));
    }

    #[test]
    fn usage_totals_fold_cache_read_and_write_into_both_status_bar_and_side_panel() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-cache-usage",
            Vec::new(),
            None,
            None,
            None,
        );

        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            cache_read_tokens: Some(9_000),
            cache_write_tokens: Some(2_000),
            ..Default::default()
        }));

        assert_eq!(app.total_usage.total_tokens, Some(12_500));
        assert_eq!(app.total_usage.grand_total(), 12_500);
        assert_eq!(app.total_usage.cache_write_tokens, Some(2_000));

        let ctx = &app.side_panel_data.context;
        assert_eq!(ctx.total_tokens, 12_500);
        assert_eq!(ctx.cumulative_prompt_tokens(), 12_000);
        assert_eq!(ctx.cache_write_tokens, 2_000);
        assert!(
            (ctx.cache_hit_rate() - 75.0).abs() < 1e-9,
            "{}",
            ctx.cache_hit_rate()
        );
        assert_eq!(app.last_prompt_tokens, Some(12_000));
        assert_eq!(ctx.last_prompt_tokens, Some(12_000));

        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 500,
            output_tokens: 100,
            ..Default::default()
        }));
        assert_eq!(app.total_usage.total_tokens, Some(13_100));
        assert_eq!(app.last_prompt_tokens, Some(500));
        assert_eq!(app.side_panel_data.context.last_prompt_tokens, Some(500));
    }

    #[test]
    fn slash_commands_and_esc_cancel_work() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-12345678",
            Vec::new(),
            None,
            None,
            None,
        );

        app.set_busy_state(true);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );
        app.set_busy_state(false);

        app.input.insert_str("/effort high");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::High));
        assert!(app.effort_ignition_at.is_some());

        app.input.insert_str("/skills");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::OpenSkillPicker
        );

        app.input.insert_str("/skills list");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::ShowSkills
        );

        app.user_skill_names = vec!["rust-review".to_string()];
        app.input.insert_str("/rust-review crates/mizar-skills");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("/rust-review crates/mizar-skills".to_string())
        );

        let abs_path_prompt = "/Users/alice/workspace/mizar/docs/plugins.md 帮我更新一下文档";
        app.input.insert_str(abs_path_prompt);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt(abs_path_prompt.to_string())
        );

        app.input.insert_str("/unknown-cmd hello");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("/unknown-cmd hello".to_string())
        );

        app.input.insert_str("/exit");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::Quit
        );
        assert!(app.should_quit);
    }

    #[test]
    fn keymap_supports_ctrl_l_model_picker_tab_effort_and_enter_queue_vs_ctrl_enter_steer() {
        let providers = vec![ProviderInfo {
            id: "magpie".into(),
            name: "Magpie".into(),
            protocol: "openai-completions".into(),
            models: vec![
                ModelInfo {
                    id: "model-limited".into(),
                    name: "Limited".into(),
                    context_window: Some(64_000),
                    max_tokens: None,
                    reasoning_efforts: vec![
                        ReasoningEffort::Low,
                        ReasoningEffort::Medium,
                        ReasoningEffort::High,
                    ],
                    input: Vec::new(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "model-all".into(),
                    name: "All".into(),
                    context_window: Some(128_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
                    ..Default::default()
                },
            ],
        }];
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-12345678",
            providers,
            Some("magpie".into()),
            Some("model-limited".into()),
            None,
        );

        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Low));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Medium));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::High));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Low));

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::ModelPicker { .. })
        ));
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.model.as_deref(), Some("model-all"));
        assert!(!app.overlay.is_open());

        app.reasoning_effort = Some(ReasoningEffort::Max);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Off));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Minimal));

        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::SlashComplete { .. })
        ));
        let before = app.reasoning_effort;
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, before);
        assert_eq!(app.input.text(), "/help ");
        app.input.clear();

        app.input.insert_str("a");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        app.input.insert_str("b");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            AppAction::SubmitPrompt("a\nb".into())
        );

        app.set_busy_state(true);
        app.input.insert_str("queued one");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(
            app.queue
                .items()
                .iter()
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>(),
            vec!["queued one"]
        );
        assert!(app.input.is_empty());

        app.input.insert_str("urgent steer");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            AppAction::SteerPrompt("urgent steer".into())
        );
        assert!(app.input.is_empty());

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );

        app.input.insert_str("draft while busy");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(app.input.is_empty());
        assert!(!app.should_quit);

        app.set_busy_state(false);
        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::SlashComplete { .. })
        ));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(app.input.is_empty());
        assert!(!app.overlay.is_open());
        assert!(!app.should_quit);
    }

    #[test]
    fn mention_completion_supports_directory_drilldown_line_suffix_esc_dismiss_and_replay_fold() {
        let ws = std::env::temp_dir().join(format!("mizar_app_mention_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(ws.join("src/nested")).unwrap();
        std::fs::write(ws.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();
        std::fs::write(ws.join("src/nested/mod.rs"), "// nested\n").unwrap();

        let mut app = AppState::new(ws.clone(), "sess-mention", Vec::new(), None, None, None);

        app.handle_key(KeyEvent::new(KeyCode::Char('@'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE));
        let src_dir_idx = match &app.overlay.active {
            Some(OverlayKind::FileMention { matches, .. }) => matches
                .iter()
                .position(|c| {
                    c.path == "src/" && c.kind == crate::mentions::MentionCandidateKind::Directory
                })
                .expect("expected src/ directory candidate"),
            _ => panic!("expected FileMention overlay"),
        };
        for _ in 0..src_dir_idx {
            app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "@src/");
        match &app.overlay.active {
            Some(OverlayKind::FileMention { query, matches, .. }) => {
                assert_eq!(query, "src/");
                assert!(matches.iter().any(|c| c.path == "src/lib.rs"));
            }
            _ => panic!("expected continued FileMention after dir accept"),
        }

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(!app.overlay.is_open());
        assert_eq!(app.input.text(), "@src/");

        app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::FileMention { .. })
        ));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "@src/lib.rs ");
        assert!(!app.overlay.is_open());

        app.input.clear();
        for ch in "@nested#L10-20".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert!(!app.overlay.is_open());
        for _ in 0..7 {
            app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE));
        }
        match &app.overlay.active {
            Some(OverlayKind::FileMention { query, matches, .. }) => {
                assert_eq!(query, "nested");
                assert!(matches.iter().any(|c| c.path == "src/nested/mod.rs"));
            }
            _ => panic!("expected FileMention before #L10-20"),
        }
        let mod_idx = match &app.overlay.active {
            Some(OverlayKind::FileMention { matches, .. }) => matches
                .iter()
                .position(|c| c.path == "src/nested/mod.rs")
                .expect("mod.rs candidate"),
            _ => unreachable!(),
        };
        for _ in 0..mod_idx {
            app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        }
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "@src/nested/mod.rs#L10-20 ");

        let folded = summarize_context_notice_for_replay(
            FILE_MENTIONS_SOURCE,
            "<attached-file path=\"src/lib.rs\" lines=\"1-2\">\nSECRET_FILE_BODY\n</attached-file>\n<attached-directory path=\"src\">\na/\n</attached-directory>",
        );
        assert_eq!(folded, "[file-mentions] 已附加引用：src/lib.rs, src");
        assert!(!folded.contains("SECRET_FILE_BODY"));

        let env_folded = summarize_context_notice_for_replay(
            "environment-context",
            "<environment_context>\n<cwd>/workspace/mizar</cwd>\n<current_date>2026-11-22</current_date>\n</environment_context>",
        );
        assert_eq!(
            env_folded,
            "[environment-context] /workspace/mizar (2026-11-22)"
        );

        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn summarize_tool_result_and_extras_keep_full_args_and_compact_previews() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-tool-meta",
            Vec::new(),
            None,
            None,
            None,
        );
        let raw_args =
            r#"{"command":"echo line1 && echo line2 && echo very_long_command_preserved"}"#;
        app.apply_ui_event(UiEvent::ToolCall(ToolCall {
            id: "b1".into(),
            name: "bash".into(),
            arguments: raw_args.into(),
        }));
        let out = (1..=10)
            .map(|i| format!("out {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.apply_ui_event(UiEvent::ToolResult {
            call_id: "b1".into(),
            name: "bash".into(),
            content: out,
            is_error: false,
            blocked: false,
            details: Some(serde_json::json!({ "exit_code": 0, "duration_ms": 15 })),
        });

        let extra = app.tool_extras.get("b1").expect("tool extra");
        assert_eq!(extra.full_args, raw_args);
        assert_eq!(extra.inline_meta.as_deref(), Some("10 lines"));
        match &app.items[0] {
            ChatItem::ToolCall { output_preview, .. } => {
                assert_eq!(output_preview, "out 9\nout 10");
                assert!(!output_preview.contains("前省略"));
            }
            other => panic!("unexpected item: {other:?}"),
        }
    }

    #[test]
    fn bash_mode_prefix_detection_esc_slash_bypass_history_and_context_stats() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-bash-ui",
            Vec::new(),
            None,
            None,
            None,
        );

        assert_eq!(BashInputMode::from_input("normal prompt"), None);
        assert_eq!(
            BashInputMode::from_input("  !ls -la"),
            Some(BashInputMode::Context)
        );
        assert_eq!(
            BashInputMode::from_input("  !! ls -la"),
            Some(BashInputMode::NoContext)
        );
        assert_eq!(BashInputMode::from_input("!"), Some(BashInputMode::Context));
        assert_eq!(
            BashInputMode::from_input("!!"),
            Some(BashInputMode::NoContext)
        );

        app.input.set_text_end("/he");
        app.sync_input_completion_overlay();
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::SlashComplete { .. })
        ));
        app.input.set_text_end("!/help");
        app.sync_input_completion_overlay();
        assert_eq!(app.bash_mode(), Some(BashInputMode::Context));
        assert!(app.overlay.active.is_none());

        let act_slash_in_bash = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            act_slash_in_bash,
            AppAction::RunLocalBash {
                command: "/help".into(),
                exclude_from_context: false,
            }
        );
        assert!(app.overlay.active.is_none());
        assert_eq!(app.input.history(), &["!/help".to_string()]);

        for empty_cmd in ["!", "!!", "  !  "] {
            app.input.set_text_end(empty_cmd);
            let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert_eq!(act, AppAction::None);
            assert!(app.active_flash().is_some_and(|f| f.contains("Bash")));
            assert_eq!(app.input.text(), empty_cmd);
        }

        app.input.set_text_end("!! pwd");
        assert_eq!(app.bash_mode(), Some(BashInputMode::NoContext));
        let esc_act = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(esc_act, AppAction::None);
        assert!(app.input.is_empty());
        assert_eq!(app.bash_mode(), None);

        app.set_busy_state(true);
        app.input.set_text_end("! echo hi");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "! echo hi");
        assert!(app.active_flash().is_some());
        app.set_busy_state(false);

        app.start_local_bash("lb-1", "sleep 5", false);
        assert!(app.is_bash_running());
        app.input.set_text_end("!! echo two");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "!! echo two");
        assert!(app.active_flash().is_some_and(|f| f.contains("Esc")));

        app.input.clear();
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );

        app.apply_ui_event(UiEvent::LocalBashFinished {
            call_id: "lb-1".into(),
            command: "sleep 5".into(),
            exclude_from_context: false,
            output: Ok(LocalBashOutput {
                display: "context output payload\n".repeat(4),
                session_output: "context output payload\n".repeat(4),
                exit_code: Some(0),
                cancelled: false,
                timed_out: false,
            }),
        });
        assert!(!app.is_bash_running());
        let tokens_after_ctx = app.side_panel_data.context.tool_tokens;
        assert!(tokens_after_ctx > 0);

        app.start_local_bash("lb-2", "cat huge_local.log", true);
        app.apply_ui_event(UiEvent::LocalBashFinished {
            call_id: "lb-2".into(),
            command: "cat huge_local.log".into(),
            exclude_from_context: true,
            output: Ok(LocalBashOutput {
                display: "local only massive text\n".repeat(50),
                session_output: "local only massive text\n".repeat(50),
                exit_code: Some(0),
                cancelled: false,
                timed_out: false,
            }),
        });
        assert_eq!(
            app.side_panel_data.context.tool_tokens, tokens_after_ctx,
            "!! local bash must not pollute context breakdown"
        );
        assert!(
            app.items
                .iter()
                .all(|it| !matches!(it, ChatItem::User { .. }))
        );
    }

    #[test]
    fn compact_slash_command_esc_cancel_and_usage_preservation() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-compact-ui",
            Vec::new(),
            None,
            None,
            None,
        );
        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 1_200,
            output_tokens: 300,
            total_tokens: Some(1_500),
            ..TokenUsage::default()
        }));

        app.input.set_text_end("/compact 保留重构决策");
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(act, AppAction::CompactSession(Some("保留重构决策".into())));

        app.apply_ui_event(UiEvent::CompactionStarted {
            trigger: "manual".into(),
        });
        assert!(app.is_compacting());
        app.apply_ui_event(UiEvent::CompactionProgress {
            stage: "summary".into(),
            delta_chars: 180,
        });
        assert!(app.compaction_status().is_some_and(|s| s.contains("180")));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );

        app.apply_ui_event(UiEvent::CompactionFinished {
            outcome: "compacted".into(),
            tokens_before: Some(12_000),
            tokens_after: Some(2_400),
            pruned_count: 2,
            first_kept_seq: Some(10),
            summary_preview: Some("已完成核心重构".into()),
            message: None,
            is_manual: true,
        });
        assert!(!app.is_compacting());
        assert!(app.compaction_status().is_none());
        assert_eq!(app.total_usage.input_tokens, 1_200);
        assert_eq!(app.total_usage.output_tokens, 300);
        assert!(app.items.iter().any(|it| matches!(
            it,
            ChatItem::Notice { text, .. } if text.contains("保留自 #10") && text.contains("已完成核心重构")
        )));
    }

    #[test]
    fn settings_module_toggle_dispatches_when_idle_and_blocks_while_busy() {
        use crate::overlay::{OverlayKind, SettingEntry, SettingItemKind};

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-settings-modules",
            Vec::new(),
            None,
            None,
            None,
        );

        let entries = vec![SettingEntry {
            key: "plugin.mcp".into(),
            category: "模块".into(),
            label: "MCP 客户端 (mcp)".into(),
            description: "连接外部 MCP 服务器".into(),
            kind: SettingItemKind::Toggle { value: true },
        }];
        app.overlay
            .open_settings(entries, None, "~/.mizar/settings.json".into());

        for case in 0..3 {
            match case {
                0 => app.set_busy_state(true),
                1 => app.set_local_bash_state(),
                _ => app.set_compacting_state("压缩中…"),
            }
            let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert_eq!(act, AppAction::None);
            match &app.overlay.active {
                Some(OverlayKind::SettingsModal { items, .. }) => {
                    assert_eq!(items[0].kind, SettingItemKind::Toggle { value: true });
                }
                _ => panic!("expected SettingsModal to stay open after busy rejection"),
            }
            assert!(app.items.iter().any(|item| matches!(
                item,
                ChatItem::Notice { level: NoticeLevel::Warn, text, .. }
                    if text.contains("当前有正在执行的任务，请等待完成后再切换模块")
            )));
        }

        app.set_idle_state();
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            act,
            AppAction::ApplySetting {
                key: "plugin.mcp".into(),
                value: "false".into(),
            }
        );
    }

    fn queued_app(prompts: &[&str]) -> AppState {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queue",
            Vec::new(),
            None,
            None,
            None,
        );
        app.set_busy_state(true);
        for text in prompts {
            app.input.insert_str(text);
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        }
        app
    }

    fn queued_texts(app: &AppState) -> Vec<&str> {
        app.queue
            .items()
            .iter()
            .map(|item| item.text.as_str())
            .collect()
    }

    #[test]
    fn prompt_queue_retracts_swaps_and_discards_without_duplication() {
        let mut app = queued_app(&["first", "second"]);
        app.input.push_history("older prompt");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            AppAction::None
        );
        let last_id = app.queue.items().last().unwrap().id;
        assert_eq!(app.queue.selected_id(), Some(last_id));
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "second");
        assert_eq!(queued_texts(&app), vec!["first"]);
        assert_eq!(app.queue.selected_id(), None);

        app.input.clear();
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.queue.selected_id(), None);
        assert_eq!(app.input.text(), "older prompt");
        assert_eq!(queued_texts(&app), vec!["first"]);

        app.input.clear();
        app.input.insert_str("draft");
        app.queue.set_selected(Some(app.queue.items()[0].id));
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "first");
        assert_eq!(queued_texts(&app), vec!["draft"]);

        app.input.clear();
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE));
        assert!(app.queue.is_empty());
        assert_eq!(app.queue.selected_id(), None);

        let mut two = queued_app(&["a", "b"]);
        two.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        let sel_b = two.queue.selected_id().unwrap();
        assert!(two.dequeue_prompt(sel_b).is_some());
        assert_eq!(two.queue.selected_id(), None);
        two.queue.set_selected(Some(two.queue.items()[0].id));
        assert_eq!(two.dequeue_next().map(|i| i.text), Some("a".into()));
        assert_eq!(two.queue.selected_id(), None);
    }

    #[test]
    fn prompt_queue_shortcuts_steer_submit_and_passthrough() {
        let mut app = queued_app(&["head", "tail"]);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            AppAction::SteerPrompt("head".into())
        );
        assert_eq!(queued_texts(&app), vec!["tail"]);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            AppAction::SteerPrompt("tail".into())
        );
        assert!(app.queue.is_empty());

        let mut newline_app = queued_app(&["keep"]);
        newline_app.input.insert_str("draft");
        assert_eq!(
            newline_app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            AppAction::None
        );
        assert_eq!(newline_app.input.text(), "draft\n");
        assert_eq!(queued_texts(&newline_app), vec!["keep"]);

        let mut alt_app = queued_app(&["first", "tail"]);
        alt_app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT));
        assert_eq!(alt_app.input.text(), "tail");
        assert_eq!(queued_texts(&alt_app), vec!["first"]);
        alt_app.input.clear();
        alt_app.input.insert_str("draft2");
        alt_app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT));
        assert_eq!(alt_app.input.text(), "first");
        assert_eq!(queued_texts(&alt_app), vec!["draft2"]);

        alt_app.input.clear();
        alt_app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            alt_app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(alt_app.queue.selected_id(), None);
        alt_app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        alt_app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE));
        assert_eq!(alt_app.queue.selected_id(), None);
        assert_eq!(alt_app.input.text(), "h");

        alt_app.input.clear();
        alt_app.set_busy_state(false);
        assert_eq!(
            alt_app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("draft2".into())
        );
        assert!(alt_app.queue.is_empty());
    }

    #[test]
    fn ctrl_v_triggers_clipboard_action_and_enforces_20_image_limit() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-img-limit",
            Vec::new(),
            None,
            None,
            None,
        );
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)),
            AppAction::PasteClipboard
        );

        let sample = ImageAttachment::new("image/png", "iVBORw==").with_dimensions(64, 64);
        for _ in 0..MAX_IMAGES_PER_MESSAGE {
            app.insert_pasted_image(None, sample.clone())
                .expect("should allow up to 20 images");
        }
        assert_eq!(app.input.image_count(), MAX_IMAGES_PER_MESSAGE);

        assert!(app.insert_pasted_image(None, sample.clone()).is_none());
        assert_eq!(app.input.image_count(), MAX_IMAGES_PER_MESSAGE);
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice { level: NoticeLevel::Warn, text }) if text.contains("20")
        ));

        app.input.backspace();
        assert_eq!(app.input.image_count(), MAX_IMAGES_PER_MESSAGE - 1);
        assert!(app.insert_pasted_image(None, sample).is_some());
        assert_eq!(app.input.image_count(), MAX_IMAGES_PER_MESSAGE);
    }

    #[test]
    fn model_input_capability_rejects_images_without_clearing_draft() {
        let providers = vec![ProviderInfo {
            id: "openai".into(),
            name: "OpenAI".into(),
            protocol: "openai-responses".into(),
            models: vec![
                ModelInfo {
                    id: "text-only".into(),
                    name: "Text Only".into(),
                    context_window: Some(64_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: vec!["text".into()],
                    ..Default::default()
                },
                ModelInfo {
                    id: "vision".into(),
                    name: "Vision".into(),
                    context_window: Some(128_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: vec!["text".into(), "image".into()],
                    ..Default::default()
                },
            ],
        }];
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-vision",
            providers,
            Some("openai".into()),
            Some("text-only".into()),
            None,
        );
        let img = ImageAttachment::new("image/png", "QUFB").with_dimensions(100, 100);
        let tok = app.insert_pasted_image(None, img.clone()).unwrap();
        app.input.insert_str(" 请帮我分析");

        assert!(!app.active_model_supports_images());
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.image_count(), 1);
        assert!(app.input.text().contains(&tok));
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice { level: NoticeLevel::Error, text }) if text.contains("不支持图片输入")
        ));

        app.provider = Some("openai".into());
        app.model = Some("vision".into());
        assert!(app.active_model_supports_images());
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            act,
            AppAction::SubmitPromptWithImages {
                text: format!("{tok} 请帮我分析"),
                images: vec![img],
            }
        );
        assert!(app.input.is_empty());
    }

    #[test]
    fn queued_prompt_with_images_retracts_and_resubmits_intact() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queue-img",
            Vec::new(),
            None,
            None,
            None,
        );
        app.set_busy_state(true);
        let img = ImageAttachment::new("image/jpeg", "QkJC").with_dimensions(800, 600);
        let tok = app
            .insert_pasted_image(Some("image/png → image/jpeg"), img.clone())
            .unwrap();
        app.input.insert_str(" 看下");

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.queue.len(), 1);
        assert_eq!(
            app.queue.items()[0].images,
            vec![(tok.clone(), img.clone())]
        );

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            AppAction::None
        );
        assert!(app.queue.is_empty());
        assert_eq!(app.input.image_count(), 1);

        app.input.move_left();
        app.set_busy_state(false);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPromptWithImages {
                text: format!("{tok} 看下"),
                images: vec![img],
            }
        );
    }

    #[test]
    fn turn_end_reasons_distinguish_program_status_notices_and_replay() {
        let mut app = queued_app(&[]);

        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 10 });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 100,
            outcome: UiStepOutcome::Completed,
            usage: None,
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });
        assert_eq!(
            app.last_program_status,
            Some((ProgramStatusState::Done, Some("Turn completed".to_string())))
        );
        assert_eq!(app.last_turn_end_reason, Some(UiTurnEndReason::Completed));
        assert!(
            !app.items
                .iter()
                .any(|item| matches!(item, ChatItem::Notice { .. })),
            "正常 Completed 不应写入 Notice"
        );

        app.apply_ui_event(UiEvent::TurnStart { turn: 2 });
        assert_eq!(app.last_turn_end_reason, None);
        assert_eq!(app.last_step_outcome, None);
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 200 });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 2,
            ts_ms: 300,
            outcome: UiStepOutcome::MaxTokens,
            usage: None,
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Truncated,
        });
        assert_eq!(app.last_turn_end_reason, Some(UiTurnEndReason::Truncated));
        assert_eq!(app.last_step_outcome, Some((2, UiStepOutcome::MaxTokens)));
        assert_ne!(
            app.last_program_status.as_ref().map(|(s, _)| *s),
            Some(ProgramStatusState::Done)
        );
        assert_eq!(
            app.last_program_status,
            Some((
                ProgramStatusState::Idle,
                Some("Output truncated".to_string())
            ))
        );
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice {
                level: NoticeLevel::Warn,
                text,
            }) if text.contains("第 2 步") && text.contains("max_tokens")
        ));

        app.apply_ui_event(UiEvent::TurnStart { turn: 3 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 400 });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 3,
            ts_ms: 500,
            outcome: UiStepOutcome::ToolCalls,
            usage: None,
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::MaxSteps,
        });
        assert_eq!(app.last_turn_end_reason, Some(UiTurnEndReason::MaxSteps));
        assert_ne!(
            app.last_program_status.as_ref().map(|(s, _)| *s),
            Some(ProgramStatusState::Done)
        );
        assert_eq!(
            app.last_program_status,
            Some((
                ProgramStatusState::Idle,
                Some("Max steps reached".to_string())
            ))
        );
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice {
                level: NoticeLevel::Warn,
                text,
            }) if text.contains("步数上限")
        ));

        let before_err_notices = app
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    ChatItem::Notice {
                        level: NoticeLevel::Error,
                        ..
                    }
                )
            })
            .count();
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Error {
                message: "upstream 500".into(),
            },
        });
        assert_eq!(
            app.last_program_status,
            Some((
                ProgramStatusState::Error,
                Some("Request failed".to_string())
            ))
        );
        app.apply_ui_event(UiEvent::RunFinished(Err("upstream 500".into())));
        let after_err_notices = app
            .items
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    ChatItem::Notice {
                        level: NoticeLevel::Error,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(after_err_notices, before_err_notices + 1);

        app.reset_for_new_session("sess-next", None);
        assert_eq!(app.last_turn_end_reason, None);
        assert_eq!(app.last_step_outcome, None);
        assert_eq!(app.last_program_status, None);

        app.restore_session_view(SessionViewData {
            session_id: "sess-replay".into(),
            session_title: None,
            items: vec![SessionReplayItem::Notice {
                level: NoticeLevel::Warn,
                text: "Reached max steps (max_steps=2)".into(),
            }],
            finish_notice: "已恢复".into(),
            selection: None,
            ..Default::default()
        });
        assert!(app.items.iter().any(|item| matches!(
            item,
            ChatItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text == "Reached max steps (max_steps=2)"
        )));
    }

    #[test]
    fn restore_session_view_preserves_usage_and_compaction_updates_context_tokens() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-empty",
            Vec::new(),
            None,
            None,
            None,
        );
        let restored_usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            total_tokens: Some(12_500),
            cache_read_tokens: Some(9_000),
            cache_write_tokens: Some(2_000),
            reasoning_tokens: None,
        };
        app.restore_session_view(SessionViewData {
            session_id: "sess-resumed".into(),
            session_title: Some("Resumed".into()),
            items: vec![
                SessionReplayItem::User {
                    text: "hello".into(),
                },
                SessionReplayItem::Assistant {
                    blocks: vec![ContentBlock::Text {
                        text: "world".into(),
                    }],
                },
            ],
            finish_notice: "已恢复历史会话".into(),
            selection: None,
            total_usage: restored_usage,
            last_prompt_tokens: Some(12_000),
            ..SessionViewData::default()
        });

        assert_eq!(app.total_usage, restored_usage);
        assert_eq!(app.last_prompt_tokens, Some(12_000));
        assert_eq!(app.side_panel_data.context.total_tokens, 12_500);
        assert_eq!(app.side_panel_data.context.last_prompt_tokens, Some(12_000));

        app.apply_ui_event(UiEvent::CompactionFinished {
            outcome: "compacted".into(),
            tokens_before: Some(12_000),
            tokens_after: Some(2_400),
            pruned_count: 0,
            first_kept_seq: Some(2),
            summary_preview: None,
            message: None,
            is_manual: false,
        });
        assert_eq!(app.total_usage, restored_usage);
        assert_eq!(app.last_prompt_tokens, Some(2_400));
        assert_eq!(app.side_panel_data.context.last_prompt_tokens, Some(2_400));
    }

    fn test_providers() -> Vec<ProviderInfo> {
        vec![ProviderInfo {
            id: "openai".into(),
            name: "OpenAI".into(),
            protocol: "openai-responses".into(),
            models: vec![
                ModelInfo {
                    id: "gpt-5".into(),
                    name: "GPT-5".into(),
                    context_window: Some(128_000),
                    max_tokens: Some(16_384),
                    reasoning_efforts: vec![ReasoningEffort::Low, ReasoningEffort::High],
                    input: Vec::new(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "gpt-4o-mini".into(),
                    name: "GPT-4o Mini".into(),
                    context_window: Some(128_000),
                    max_tokens: Some(4_096),
                    reasoning_efforts: vec![ReasoningEffort::Low],
                    input: Vec::new(),
                    ..Default::default()
                },
            ],
        }]
    }

    #[test]
    fn settings_model_picker_does_not_mutate_active_session_model_and_esc_restores_row() {
        let mut app = AppState::new(
            PathBuf::from("/tmp/ws"),
            "s-model-pick",
            test_providers(),
            Some("openai".into()),
            Some("gpt-5".into()),
            Some(ReasoningEffort::Medium),
        );

        app.overlay.open_settings(
            vec![
                SettingEntry {
                    key: "sessionTitle.enabled".into(),
                    category: "会话标题".into(),
                    label: "自动生成标题".into(),
                    description: "开关".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
                SettingEntry {
                    key: "sessionTitle.model".into(),
                    category: "会话标题".into(),
                    label: "标题生成模型".into(),
                    description: "选择生成会话标题所用模型".into(),
                    kind: SettingItemKind::Model {
                        provider: None,
                        model: None,
                    },
                },
            ],
            None,
            "~/.mizar/settings.json".into(),
        );
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(app.pending_settings_model_pick.is_some());
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::ModelPicker { .. })
        ));

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(app.pending_settings_model_pick.is_none());
        assert_eq!(app.provider.as_deref(), Some("openai"));
        assert_eq!(app.model.as_deref(), Some("gpt-5"));
        match &app.overlay.active {
            Some(OverlayKind::SettingsModal { selected, .. }) => assert_eq!(*selected, 1),
            _ => panic!("expected SettingsModal restored on Esc"),
        }

        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            act,
            AppAction::ApplySetting {
                key: "sessionTitle.model".into(),
                value: "openai/gpt-4o-mini".into(),
            }
        );
        assert_eq!(app.provider.as_deref(), Some("openai"));
        assert_eq!(app.model.as_deref(), Some("gpt-5"));
        match &app.overlay.active {
            Some(OverlayKind::SettingsModal {
                items, selected, ..
            }) => {
                assert_eq!(*selected, 1);
                assert_eq!(items[1].display_value(), "openai/gpt-4o-mini");
            }
            _ => panic!("expected SettingsModal restored after pick"),
        }

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        app.open_model_selector();
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        let normal_act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(normal_act, AppAction::None);
        assert_eq!(app.provider.as_deref(), Some("openai"));
        assert_eq!(app.model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn activity_state_nests_auto_compaction_and_enforces_guards() {
        let mut app = AppState::new(
            PathBuf::from("/tmp/ws"),
            "s-act",
            Vec::new(),
            None,
            None,
            None,
        );

        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        assert!(app.is_busy());
        assert!(!app.is_compacting());

        app.apply_ui_event(UiEvent::CompactionStarted {
            trigger: "auto-threshold".into(),
        });
        assert!(app.is_busy(), "auto-compaction must preserve is_busy()");
        assert!(app.is_compacting());
        assert!(app.compaction_status().is_some());

        app.apply_ui_event(UiEvent::CompactionFinished {
            outcome: "compacted".into(),
            tokens_before: Some(10_000),
            tokens_after: Some(3_000),
            pruned_count: 1,
            first_kept_seq: Some(4),
            summary_preview: None,
            message: None,
            is_manual: false,
        });
        assert!(
            app.is_busy(),
            "ending auto-compaction must keep Agent busy until TurnEnd"
        );
        assert!(!app.is_compacting());

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });
        assert!(!app.is_running_activity());

        app.apply_ui_event(UiEvent::CompactionStarted {
            trigger: "manual".into(),
        });
        assert!(app.is_compacting());
        assert!(!app.is_busy());
        assert!(app.is_running_activity());

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(!app.should_quit);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        app.input.set_text_end("/new");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        app.input.set_text_end("! ls");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        app.input.clear();
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );
    }

    #[test]
    fn approval_queue_and_theme_ctrl_c_and_scroll_lock_preserve_state() {
        let mut app = AppState::new(
            PathBuf::from("/tmp/ws"),
            "s-queue-ov",
            Vec::new(),
            None,
            None,
            None,
        );
        app.set_theme_preset(ThemePreset::MistDark);

        app.overlay.open_theme_picker(app.theme_preset);
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        assert_ne!(app.theme_preset, ThemePreset::MistDark);
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert_eq!(app.theme_preset, ThemePreset::MistDark);
        assert!(!app.overlay.is_open());

        app.overlay.open_help();
        let (tx1, mut rx1) = futures::channel::oneshot::channel();
        let (tx2, mut rx2) = futures::channel::oneshot::channel();
        app.apply_ui_event(UiEvent::ApprovalRequested {
            tool_name: "bash".into(),
            args_preview: "ls".into(),
            responder: Arc::new(std::sync::Mutex::new(Some(tx1))),
        });
        app.apply_ui_event(UiEvent::ApprovalRequested {
            tool_name: "write".into(),
            args_preview: "a.txt".into(),
            responder: Arc::new(std::sync::Mutex::new(Some(tx2))),
        });
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::HelpModal { .. })
        ));
        assert!(rx1.try_recv().unwrap().is_none());
        assert!(rx2.try_recv().unwrap().is_none());

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::Approval(ref ap)) if ap.tool_name == "bash"
        ));
        app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
        assert_eq!(rx1.try_recv().unwrap(), Some(ApprovalDecision::AllowOnce));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::Approval(ref ap)) if ap.tool_name == "write"
        ));
        app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE));
        assert_eq!(rx2.try_recv().unwrap(), Some(ApprovalDecision::Deny));
        assert!(!app.overlay.is_open());

        app.effective_scroll(20, 100);
        app.scroll_up(15);
        assert!(!app.auto_scroll);
        app.effective_scroll(20, 90);
        assert!(
            !app.auto_scroll,
            "shrink while scrollable must preserve lock"
        );

        app.restore_session_view(SessionViewData {
            session_id: "s-restored".into(),
            session_title: None,
            items: vec![SessionReplayItem::Assistant {
                blocks: vec![ContentBlock::ToolCall {
                    id: "tc-1".into(),
                    name: "edit".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                }],
            }],
            finish_notice: "ok".into(),
            selection: None,
            total_usage: TokenUsage::default(),
            last_prompt_tokens: None,
            ..SessionViewData::default()
        });
        assert_eq!(app.touched_files.len(), 1);
        assert_eq!(app.touched_files[0].path, "src/lib.rs");
        assert_eq!(app.touched_files[0].op, "edit");
    }

    #[test]
    fn tui_features_disable_interactions_and_rendering_safely() {
        let mut app = AppState::new(
            PathBuf::from("/tmp/ws"),
            "sess-features",
            Vec::new(),
            None,
            None,
            None,
        );
        app.side_panel.open = true;
        app.trajectory_scene.open = true;
        app.items.push(ChatItem::User {
            text: "turn 1".into(),
        });
        app.items.push(ChatItem::Assistant {
            reasoning: String::new(),
            text: "reply 1".into(),
        });
        app.items.push(ChatItem::User {
            text: "turn 2".into(),
        });
        app.items.push(ChatItem::Assistant {
            reasoning: String::new(),
            text: "reply 2".into(),
        });

        let disabled = TuiFeatures {
            emphasis: false,
            side_panel: false,
            timeline_rail: false,
            trajectory_view: false,
            selection: false,
            clipboard: false,
            local_bash: false,
        };
        app.set_features(disabled);

        // 1. set_features 自动关闭 emphasis、已开启的侧栏和轨迹场景
        assert!(!app.emphasis.enabled);
        assert!(!app.side_panel.open);
        assert!(!app.trajectory_scene.open);

        // 2. 快捷键、斜杠命令与 Launchpad 动作均安全跳过
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(!app.side_panel.open);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        app.input.set_text_end("/panel");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(!app.side_panel.open);
        app.input.set_text_end("/trace");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );

        // 3. clipboard 禁用时 Ctrl+V 提示并拒绝 PasteClipboard
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(app.active_flash().is_some_and(|msg| msg.contains("剪贴板")));

        // 4. local_bash 禁用时 `!` / `!!` 不识别为 Bash 模式，按普通文本提交
        app.input.set_text_end("!ls -la");
        assert_eq!(app.bash_mode(), None);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("!ls -la".into())
        );

        // 5. 渲染跳过侧栏分栏、时间轴导轨与轨迹场景，且鼠标选区拦截跳过
        app.side_panel.open = true;
        app.trajectory_scene.open = true;
        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| crate::ui::render(f, &mut app)).unwrap();
        assert_eq!(app.side_panel.divider_col, None);
        assert_eq!(app.rail_geo, None);
        assert!(app.selection_geom.side_panel_rect.is_none());

        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 5,
            modifiers: KeyModifiers::NONE,
        });
        assert!(!app.selection.is_active());
    }

    #[test]
    fn consecutive_cache_warming_notices_fold_cleanly_until_interrupted() {
        let mut app = AppState::new(
            PathBuf::from("/tmp"),
            "sess-cw-fold",
            Vec::new(),
            None,
            None,
            None,
        );

        app.apply_ui_event(UiEvent::CacheWarmingNotice {
            text: "已保活 Prompt Cache #1".into(),
        });
        app.apply_ui_event(UiEvent::CacheWarmingNotice {
            text: "已保活 Prompt Cache #2".into(),
        });
        app.apply_ui_event(UiEvent::CacheWarmingNotice {
            text: "已保活 Prompt Cache #3".into(),
        });

        assert_eq!(app.items.len(), 1);
        match &app.items[0] {
            ChatItem::Notice { level, text } => {
                assert_eq!(*level, NoticeLevel::Info);
                assert!(text.contains("已保活 Prompt Cache #3"), "{text}");
                assert!(text.contains("×3"), "{text}");
            }
            other => panic!("expected Notice, got: {other:?}"),
        }

        // 中间插入普通用户消息后，下一条保活通知应新开条目而不是覆盖旧消息
        app.items.push(ChatItem::User {
            text: "继续".into(),
        });
        app.apply_ui_event(UiEvent::CacheWarmingNotice {
            text: "已保活 Prompt Cache #4".into(),
        });
        assert_eq!(app.items.len(), 3);
        assert!(matches!(
            &app.items[2],
            ChatItem::Notice { text, .. } if text == "已保活 Prompt Cache #4"
        ));
    }

    #[test]
    fn subagent_state_updates_count_posts_notice_resets_and_summarizes_replay() {
        let mut app = AppState::new(
            PathBuf::from("/tmp"),
            "sess-sub-ui",
            Vec::new(),
            None,
            None,
            None,
        );
        app.apply_ui_event(UiEvent::SubagentState {
            running_count: 2,
            notice: Some("后台子代理 scout 已启动 (read_only)".into()),
        });
        assert_eq!(app.running_subagents, 2);
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice { level: NoticeLevel::Info, text })
                if text == "后台子代理 scout 已启动 (read_only)"
        ));

        app.reset_for_new_session("sess-sub-ui-2", None);
        assert_eq!(app.running_subagents, 0);

        let replayed = summarize_context_notice_for_replay(
            "subagent:scout",
            "Subagent \"scout\" completed (mode=read_only, steps=3, elapsed=2s):\n\nvery long body line",
        );
        assert!(replayed.starts_with("[subagent:scout] Subagent \"scout\" completed"));
        assert!(!replayed.contains("very long body line"));

        let wd_replayed = summarize_context_notice_for_replay(
            "subagent-watchdog:scout",
            "Subagent watchdog notice (1/2): scout idle 46s\n\ndetail instructions",
        );
        assert!(wd_replayed.starts_with("[subagent-watchdog:scout] Subagent watchdog notice"));
        assert!(!wd_replayed.contains("detail instructions"));
    }
}
