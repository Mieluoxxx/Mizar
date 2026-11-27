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

#[derive(Default)]
enum DraftTtft {
    #[default]
    Pending,
    // 满 32 项时须记住被挤出的旧样本，撤销重试才不会缩短历史。
    Sampled {
        evicted_ms: Option<u64>,
    },
}

#[derive(Default)]
struct StepDraft {
    // 通知可能把同一次尝试的草稿分成多段，重试时要保留夹在中间的通知。
    first_item: Option<usize>,
    ttft: DraftTtft,
}

pub struct AppState {
    pub workspace: PathBuf,
    pub session_id: String,
    pub session_title: Option<String>,
    pub items: Vec<ChatItem>,
    draft: StepDraft,
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
            draft: StepDraft::default(),
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
                    retry_status: None,
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
        self.draft = StepDraft::default();
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
