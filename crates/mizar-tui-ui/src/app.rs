//! TUI 核心状态机：会话投影、主题与 `pi-emphasis` 状态、浮层栈、多栏选区、虚拟化行缓存与事件分发。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use mizar_llm_types::{
    ContentBlock, ImageAttachment, MAX_IMAGES_PER_MESSAGE, ModelInfo, ProviderInfo,
    ReasoningEffort, TokenUsage, ToolCall,
};
use mizar_trajectory_core::{TimelineMode, TrajectoryReport};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use serde_json::Value;

use crate::editor::{TextInput, char_width};
use crate::emphasis::EmphasisConfig;
use crate::markdown::RenderedLine;
use crate::mentions::{FILE_MENTIONS_SOURCE, format_mention_replacement};
use crate::overlay::{
    ApprovalDecision, OverlayAction, OverlayKind, OverlayState, SessionTreeEntry,
    WorkspaceFileCache,
};
use crate::reveal::{RevealController, RevealSlot};
use crate::selection::{
    SelectionGeometry, SelectionState, copy_to_clipboard, extract_selected_text,
};
use crate::side_panel::{ContextBreakdownSnapshot, FileTouchRow, SidePanelData, SidePanelState};
use crate::terminal::{
    ProgramStatusState, ScreenMode, emit_program_status, emit_terminal_progress,
};
use crate::theme::{TerminalColors, Theme, ThemePreset};
use crate::timeline_rail::{RailGeometry, RailHit, TimelineSnapshot, hit_test_rail};
use crate::tps::TpsLedger;
use crate::trajectory_scene::{TrajectoryKeyOutcome, TrajectorySceneState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolState {
    Running,
    Done,
    Error,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalMode {
    #[default]
    Auto,
    Ask,
}

impl ApprovalMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ask => "ask",
        }
    }
}

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

/// 工具卡富展示附属数据（存放在 side-table 以保持 `ChatItem` 向后兼容枚举构造）。
#[derive(Debug, Clone, Default)]
pub struct ToolExtraMeta {
    pub full_content: String,
    pub diff_raw: Option<String>,
    pub full_args: String,
    pub inline_meta: Option<String>,
    /// 为 `true` 时（如 `!!` 本地命令），不计入模型上下文估算。
    pub exclude_from_context: bool,
}

/// 输入前缀识别得到的 Bash 交互模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashInputMode {
    /// `!` 开头：执行结果载入后续对话上下文。
    Context,
    /// `!!` 开头：执行结果仅本地展示与审计，不载入上下文。
    NoContext,
}

impl BashInputMode {
    pub fn from_input(input: &str) -> Option<Self> {
        let trimmed = input.trim_start();
        if trimmed.starts_with("!!") {
            Some(Self::NoContext)
        } else if trimmed.starts_with('!') {
            Some(Self::Context)
        } else {
            None
        }
    }

    pub fn is_context(self) -> bool {
        matches!(self, Self::Context)
    }

    pub fn exclude_from_context(self) -> bool {
        matches!(self, Self::NoContext)
    }

    /// 剥离 `!` / `!!` 前缀并裁剪两端空白，返回待执行命令。
    pub fn extract_command(self, input: &str) -> &str {
        let trimmed = input.trim();
        match self {
            Self::NoContext => trimmed.strip_prefix("!!").unwrap_or("").trim(),
            Self::Context => trimmed.strip_prefix('!').unwrap_or("").trim(),
        }
    }
}

/// 本地 Bash 命令异步执行完成后的 UI 投影负载。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalBashOutput {
    pub display: String,
    pub session_output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub timed_out: bool,
}

/// 单个消息条目的行级渲染缓存键。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemRenderKey {
    pub width: usize,
    pub theme_id: &'static str,
    pub emphasis_enabled: bool,
    pub round_caps: bool,
    pub expanded: bool,
    pub status_code: u8,
    pub reasoning_len: usize,
    pub text_len: usize,
    pub preview_len: usize,
}

#[derive(Debug, Clone)]
pub struct CachedItemLines {
    pub key: ItemRenderKey,
    pub lines: Vec<RenderedLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiAgentStatus {
    Busy,
    Idle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiTurnEndReason {
    Completed,
    Truncated,
    MaxSteps,
    Cancelled,
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiStepOutcome {
    Completed,
    ToolCalls,
    MaxTokens,
    Cancelled,
    Error { message: String },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiLogLevel {
    Silent,
    Error,
    Warn,
    Info,
    Debug,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiLogRecord {
    pub level: UiLogLevel,
    pub name: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionReplayItem {
    User {
        text: String,
    },
    ContextNotice {
        source: String,
        text: String,
    },
    Notice {
        level: NoticeLevel,
        text: String,
    },
    Compaction {
        summary: String,
        first_kept_seq: u64,
    },
    Assistant {
        blocks: Vec<ContentBlock>,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        is_error: bool,
    },
    BashExecution {
        command: String,
        output: String,
        exit_code: Option<i32>,
        cancelled: bool,
        exclude_from_context: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionViewData {
    pub session_id: String,
    pub session_title: Option<String>,
    pub items: Vec<SessionReplayItem>,
    pub finish_notice: String,
    /// 该会话最后记录的模型 / 推理强度快照；`None` 表示日志里没有（旧会话）。
    /// 仅作数据携带——是否采用、如何叠加进优先级链由宿主层决定，本层不参与决策。
    pub selection: Option<SessionSelection>,
}

/// 会话内记录的模型与推理强度选择（`mizar-tui-ui` 侧的纯数据形态）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionSelection {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
}
/// 由后端异步事件与终端外部状态投递给 UI 主循环的消息。
#[derive(Debug, Clone)]
pub enum UiEvent {
    Status(UiAgentStatus),
    TurnStart {
        turn: u64,
    },
    TurnEnd {
        reason: UiTurnEndReason,
    },
    StepStart {
        ts_ms: u128,
    },
    StepEnd {
        step: usize,
        ts_ms: u128,
        outcome: UiStepOutcome,
        usage: Option<TokenUsage>,
    },
    FirstToken {
        ts_ms: u128,
    },
    ReasoningDelta {
        text: String,
        ts_ms: u128,
    },
    TextDelta {
        text: String,
        ts_ms: u128,
    },
    ToolCall(ToolCall),
    ToolUpdate {
        call_id: String,
        name: String,
        preview: String,
    },
    ToolResult {
        call_id: String,
        name: String,
        content: String,
        is_error: bool,
        blocked: bool,
        details: Option<Value>,
    },
    Usage(TokenUsage),
    Anomaly(String),
    Flash(String),
    Notice {
        level: NoticeLevel,
        text: String,
    },
    UserSkillsUpdated(Vec<String>),
    ToolsUpdated,
    Log(UiLogRecord),
    RunFinished(Result<String, String>),
    LocalBashFinished {
        call_id: String,
        command: String,
        exclude_from_context: bool,
        output: Result<LocalBashOutput, String>,
    },
    ApprovalRequested {
        tool_name: String,
        args_preview: String,
        responder:
            Arc<std::sync::Mutex<Option<futures::channel::oneshot::Sender<ApprovalDecision>>>>,
    },
    CompactionStarted {
        trigger: String,
    },
    CompactionProgress {
        stage: String,
        delta_chars: usize,
    },
    CompactionFinished {
        outcome: String,
        tokens_before: Option<u64>,
        tokens_after: Option<u64>,
        pruned_count: usize,
        first_kept_seq: Option<u64>,
        summary_preview: Option<String>,
        message: Option<String>,
        is_manual: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchpadQuickAction {
    ModelPicker,
    ThemePicker,
    SkillPicker,
    TrajectoryScene,
    ToggleSidePanel,
}

/// 排队待发送的一条消息。
///
/// 用单调递增的 `id` 而不是下标寻址：`RunFinished` 的自动出队会在一瞬间改变下标，
/// 选中态若按位置记录就会「高亮第二条却撤回第三条」。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedPrompt {
    pub id: u64,
    pub text: String,
    pub images: Vec<(String, ImageAttachment)>,
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
    ShowStats,
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
    pub item_cache: HashMap<usize, CachedItemLines>,

    pub input: TextInput,
    pub queued_prompts: Vec<QueuedPrompt>,
    /// 队列选择器当前高亮的条目 id；`None` 表示选择器关闭。
    pub queue_selected: Option<u64>,
    next_queue_id: u64,
    pub is_busy: bool,
    pub is_bash_running: bool,
    pub is_compacting: bool,
    pub compaction_status: Option<String>,
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
    streaming_assistant_open: bool,

    // 双模视图与鼠标开关
    pub screen_mode: ScreenMode,
    pub inline_committed_items: usize,
    pub inline_welcome_committed: bool,
    pub mouse_capture_enabled: bool,
    pub launchpad_hit_rects: Vec<(LaunchpadQuickAction, Rect)>,

    // 主题与视觉语法
    pub theme_preset: ThemePreset,
    pub probed_colors: TerminalColors,
    pub theme: Theme,
    pub emphasis: EmphasisConfig,
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
    pub file_cache: WorkspaceFileCache,
    dismissed_mention: Option<(usize, String)>,

    // 确认退出 / 终端状态记忆
    pub terminal_focused: bool,
    last_ctrl_c_at: Option<Instant>,
    pub pending_exit_status: Option<(ProgramStatusState, Option<String>)>,
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
            tool_extras: HashMap::new(),
            item_cache: HashMap::new(),
            input: TextInput::new(),
            queued_prompts: Vec::new(),
            queue_selected: None,
            next_queue_id: 1,
            is_busy: false,
            is_bash_running: false,
            is_compacting: false,
            compaction_status: None,
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
            streaming_assistant_open: false,
            screen_mode: ScreenMode::Fullscreen,
            inline_committed_items: 0,
            inline_welcome_committed: false,
            mouse_capture_enabled: true,
            launchpad_hit_rects: Vec::new(),
            theme_preset,
            probed_colors,
            theme,
            emphasis: EmphasisConfig::from_env(),
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
            file_cache: WorkspaceFileCache::default(),
            dismissed_mention: None,
            terminal_focused: true,
            last_ctrl_c_at: None,
            pending_exit_status: None,
            last_turn_end_reason: None,
            last_step_outcome: None,
            last_program_status: None,
        }
    }

    /// 记录并向终端发送 ProgramStatus（供无头测试与状态栏统一观察）。
    fn record_program_status(
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
        BashInputMode::from_input(self.input.text())
    }

    pub fn push_user_message(&mut self, text: &str) {
        self.items.push(ChatItem::User {
            text: text.to_string(),
        });
        self.streaming_assistant_open = false;
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
        self.is_bash_running = true;
        self.streaming_assistant_open = false;
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
        self.item_cache.clear();
        self.scroll_to_bottom();
    }

    fn apply_local_bash_completion(
        &mut self,
        call_id: String,
        command: &str,
        exclude_from_context: bool,
        display: String,
        status: ToolState,
        inline_meta: String,
    ) {
        let failed = !matches!(status, ToolState::Done);
        let preview = summarize_tool_result("bash", &display, None, failed);
        if let Some(ChatItem::ToolCall {
            status: st,
            output_preview,
            ..
        }) = self.items.iter_mut().rev().find(|item| match item {
            ChatItem::ToolCall { id, .. } => id == &call_id,
            _ => false,
        }) {
            *st = status;
            *output_preview = preview;
        } else {
            let args_json = serde_json::json!({ "command": command }).to_string();
            let args_summary = summarize_tool_args(&args_json);
            self.tool_extras
                .entry(call_id.clone())
                .or_insert_with(|| ToolExtraMeta {
                    full_args: args_json,
                    exclude_from_context,
                    ..Default::default()
                });
            self.items.push(ChatItem::ToolCall {
                id: call_id.clone(),
                name: "bash".to_string(),
                args_summary,
                status,
                output_preview: preview,
            });
        }
        let ext = self.tool_extras.entry(call_id).or_default();
        ext.full_content = display;
        ext.inline_meta = Some(inline_meta);
        ext.exclude_from_context = exclude_from_context;
    }

    pub fn open_model_selector(&mut self) -> AppAction {
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
        self.streaming_assistant_open = false;
    }

    fn ensure_open_assistant(&mut self) -> (usize, &mut ChatItem) {
        if !self.streaming_assistant_open
            || !matches!(self.items.last(), Some(ChatItem::Assistant { .. }))
        {
            self.items.push(ChatItem::Assistant {
                reasoning: String::new(),
                text: String::new(),
            });
            self.streaming_assistant_open = true;
        }
        let idx = self.items.len() - 1;
        (idx, &mut self.items[idx])
    }

    pub fn apply_ui_event(&mut self, event: UiEvent) {
        match event {
            UiEvent::Status(status) => {
                self.is_busy = status == UiAgentStatus::Busy;
                if !self.is_busy {
                    self.streaming_assistant_open = false;
                    self.reveal.snap_all();
                }
            }
            UiEvent::TurnStart { .. } => {
                self.is_busy = true;
                self.streaming_assistant_open = false;
                self.auto_scroll = true;
                self.last_turn_end_reason = None;
                self.last_step_outcome = None;
                self.tps.on_turn_start();
                emit_terminal_progress(true);
                self.record_program_status(ProgramStatusState::Working, None, Some("Generating"));
            }
            UiEvent::TurnEnd { reason } => {
                self.is_busy = false;
                self.streaming_assistant_open = false;
                self.reveal.snap_all();
                emit_terminal_progress(false);
                let (program_state, program_msg) = turn_end_program_status(&reason);
                self.record_program_status(program_state, None, Some(program_msg));
                match &reason {
                    UiTurnEndReason::Completed => {}
                    UiTurnEndReason::Truncated => {
                        if let Some((step, UiStepOutcome::MaxTokens)) = self.last_step_outcome
                            && step > 0
                        {
                            self.push_notice(
                                NoticeLevel::Warn,
                                format!(
                                    "第 {step} 步输出因达到 max_tokens 上限被截断，回复可能不完整"
                                ),
                            );
                        } else {
                            self.push_notice(
                                NoticeLevel::Warn,
                                "输出因达到 max_tokens 上限被截断，回复可能不完整",
                            );
                        }
                    }
                    UiTurnEndReason::MaxSteps => {
                        self.push_notice(NoticeLevel::Warn, "已达到最大执行步数上限，本轮提前结束");
                    }
                    UiTurnEndReason::Cancelled => {
                        self.push_notice(NoticeLevel::Info, "已中断本轮生成");
                    }
                    UiTurnEndReason::Error { .. } => {}
                }
                self.last_turn_end_reason = Some(reason);
                if let Some(tps) = self.tps.on_turn_end() {
                    self.tps_history.push(tps);
                    if self.tps_history.len() > 32 {
                        self.tps_history.remove(0);
                    }
                }
                self.refresh_context_breakdown();
            }
            UiEvent::StepStart { ts_ms } => {
                self.last_step_start_ms = Some(ts_ms);
                self.tps.on_step_start();
            }
            UiEvent::StepEnd {
                step,
                ts_ms,
                outcome,
                usage,
            } => {
                self.last_step_outcome = Some((step, outcome));
                self.tps.on_step_end(ts_ms, usage);
            }
            UiEvent::FirstToken { ts_ms } => {
                if let Some(start) = self.last_step_start_ms.take() {
                    let ttft = ts_ms.saturating_sub(start) as u64;
                    self.last_ttft_ms = Some(ttft);
                    self.ttft_history.push(ttft);
                    if self.ttft_history.len() > 32 {
                        self.ttft_history.remove(0);
                    }
                }
                self.tps.on_first_token(ts_ms);
            }
            UiEvent::ReasoningDelta { text, ts_ms } => {
                let busy = self.is_busy;
                let (idx, item) = self.ensure_open_assistant();
                if let ChatItem::Assistant { reasoning, .. } = item {
                    reasoning.push_str(&text);
                    let snapshot = reasoning.clone();
                    self.reveal
                        .observe(RevealSlot::AssistantReasoning(idx), &snapshot, busy);
                }
                self.tps.on_token_delta(&text, ts_ms);
            }
            UiEvent::TextDelta { text, ts_ms } => {
                let busy = self.is_busy;
                let (idx, item) = self.ensure_open_assistant();
                if let ChatItem::Assistant { text: body, .. } = item {
                    body.push_str(&text);
                    let snapshot = body.clone();
                    self.reveal
                        .observe(RevealSlot::AssistantText(idx), &snapshot, busy);
                }
                self.tps.on_token_delta(&text, ts_ms);
            }
            UiEvent::ToolCall(call) => {
                self.streaming_assistant_open = false;
                self.reveal.snap_all();
                let args_summary = summarize_tool_args(&call.arguments);
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&call.arguments)
                    && let Some(p) = map
                        .get("path")
                        .or_else(|| map.get("file"))
                        .and_then(Value::as_str)
                {
                    let op = call.name.clone();
                    if !self.touched_files.iter().any(|f| f.path == p && f.op == op) {
                        self.touched_files.push(FileTouchRow {
                            path: p.to_string(),
                            op,
                        });
                    }
                }
                self.tool_extras.insert(
                    call.id.clone(),
                    ToolExtraMeta {
                        full_content: String::new(),
                        diff_raw: None,
                        full_args: call.arguments,
                        inline_meta: None,
                        exclude_from_context: false,
                    },
                );
                self.items.push(ChatItem::ToolCall {
                    id: call.id,
                    name: call.name,
                    args_summary,
                    status: ToolState::Running,
                    output_preview: String::new(),
                });
            }
            UiEvent::ToolUpdate {
                call_id, preview, ..
            } => {
                if let Some(ext) = self.tool_extras.get_mut(&call_id) {
                    ext.full_content = preview.clone();
                }
                if let Some(ChatItem::ToolCall { output_preview, .. }) =
                    self.items.iter_mut().rev().find(|item| match item {
                        ChatItem::ToolCall { id, .. } => id == &call_id,
                        _ => false,
                    })
                {
                    *output_preview = summarize_live_preview(&preview);
                }
            }
            UiEvent::ToolResult {
                call_id,
                name,
                content,
                is_error,
                blocked,
                details,
            } => {
                let status = if blocked {
                    ToolState::Blocked
                } else if is_error {
                    ToolState::Error
                } else {
                    ToolState::Done
                };
                let failed = is_error || blocked;
                let diff_raw = if !failed && name == "edit" {
                    details
                        .as_ref()
                        .and_then(|d| d.get("diff"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                } else {
                    None
                };
                let inline_meta = extract_inline_meta(&name, &content, details.as_ref(), failed);
                {
                    let extra = self.tool_extras.entry(call_id.clone()).or_default();
                    extra.full_content = content.clone();
                    extra.diff_raw = diff_raw;
                    extra.inline_meta = inline_meta;
                }
                let preview = summarize_tool_result(&name, &content, details.as_ref(), failed);
                if let Some(ChatItem::ToolCall {
                    status: st,
                    output_preview,
                    ..
                }) = self.items.iter_mut().rev().find(|item| match item {
                    ChatItem::ToolCall { id, .. } => id == &call_id,
                    _ => false,
                }) {
                    *st = status;
                    *output_preview = preview;
                } else {
                    self.items.push(ChatItem::ToolCall {
                        id: call_id,
                        name,
                        args_summary: String::new(),
                        status,
                        output_preview: preview,
                    });
                }
            }
            UiEvent::Usage(usage) => {
                self.total_usage.input_tokens = self
                    .total_usage
                    .input_tokens
                    .saturating_add(usage.input_tokens);
                self.total_usage.output_tokens = self
                    .total_usage
                    .output_tokens
                    .saturating_add(usage.output_tokens);
                if let Some(cr) = usage.cache_read_tokens {
                    self.total_usage.cache_read_tokens = Some(
                        self.total_usage
                            .cache_read_tokens
                            .unwrap_or(0)
                            .saturating_add(cr),
                    );
                }
                if let Some(cw) = usage.cache_write_tokens {
                    self.total_usage.cache_write_tokens = Some(
                        self.total_usage
                            .cache_write_tokens
                            .unwrap_or(0)
                            .saturating_add(cw),
                    );
                }
                // 口径统一由 `TokenUsage::grand_total` 定义：input 只是缓存未命中的净额，
                // 漏掉 cache_read / cache_write 会把开了 prompt cache 的会话读成个位数百分比。
                self.total_usage.total_tokens = Some(self.total_usage.grand_total());
                // 上下文占用估算用「最近一次请求」的 prompt 全量，而不是累计值。
                let last = usage.prompt_tokens();
                if last > 0 {
                    self.last_prompt_tokens = Some(last);
                }
                self.refresh_context_breakdown();
            }
            UiEvent::Anomaly(summary) => {
                self.anomalies.push(summary.clone());
                self.push_notice(NoticeLevel::Warn, format!("轨迹告警：{summary}"));
            }
            UiEvent::Flash(msg) => {
                self.set_flash(msg);
            }
            UiEvent::Notice { level, text } => {
                self.push_notice(level, text);
            }
            UiEvent::UserSkillsUpdated(names) => {
                self.user_skill_names = names;
            }
            UiEvent::ToolsUpdated => {}
            UiEvent::Log(record) => {
                let level = match record.level {
                    UiLogLevel::Error => NoticeLevel::Error,
                    UiLogLevel::Warn => NoticeLevel::Warn,
                    UiLogLevel::Debug | UiLogLevel::Info | UiLogLevel::Silent => return,
                };
                self.push_notice(level, format!("[{}] {}", record.name, record.message));
            }
            UiEvent::RunFinished(res) => {
                self.is_busy = false;
                self.streaming_assistant_open = false;
                self.reveal.snap_all();
                emit_terminal_progress(false);
                if let Err(err) = res
                    && !err.contains("request aborted")
                {
                    self.push_notice(NoticeLevel::Error, format!("请求失败：{err}"));
                    self.record_program_status(
                        ProgramStatusState::Error,
                        None,
                        Some("Request failed"),
                    );
                    if self.last_turn_end_reason.is_none() {
                        self.last_turn_end_reason = Some(UiTurnEndReason::Error { message: err });
                    }
                }
                self.refresh_context_breakdown();
            }
            UiEvent::LocalBashFinished {
                call_id,
                command,
                exclude_from_context,
                output,
            } => {
                self.is_bash_running = false;
                match output {
                    Ok(out) => {
                        let status = if out.cancelled {
                            ToolState::Blocked
                        } else if out.exit_code.unwrap_or(0) != 0 {
                            ToolState::Error
                        } else {
                            ToolState::Done
                        };
                        let inline_meta =
                            bash_inline_meta(exclude_from_context, out.exit_code, out.cancelled);
                        self.apply_local_bash_completion(
                            call_id,
                            &command,
                            exclude_from_context,
                            out.display,
                            status,
                            inline_meta,
                        );
                    }
                    Err(err) => {
                        let inline_meta =
                            format!("{} · 失败", bash_scope_label(exclude_from_context));
                        self.apply_local_bash_completion(
                            call_id,
                            &command,
                            exclude_from_context,
                            err,
                            ToolState::Error,
                            inline_meta,
                        );
                    }
                }
                self.item_cache.clear();
                self.refresh_context_breakdown();
            }
            UiEvent::ApprovalRequested {
                tool_name,
                args_preview,
                responder,
            } => {
                if let Some(tx) = responder.lock().unwrap().take() {
                    self.record_program_status(
                        ProgramStatusState::Blocked,
                        Some("permission"),
                        Some(&format!("Approve {tool_name}")),
                    );
                    self.overlay.open_approval(tool_name, args_preview, tx);
                }
            }
            UiEvent::CompactionStarted { trigger } => {
                self.is_compacting = true;
                let label = match trigger.as_str() {
                    "manual" => "手动压缩中…",
                    "overflow" => "溢出恢复压缩中…",
                    "cold-cache" => "冷缓存整理中…",
                    _ => "上下文压缩中…",
                };
                self.compaction_status = Some(label.to_string());
            }
            UiEvent::CompactionProgress { stage, delta_chars } => {
                self.is_compacting = true;
                let status = match stage.as_str() {
                    "pruned" => "压缩中 · 已剪枝大输出".to_string(),
                    "prefill" => "压缩中 · prefill…".to_string(),
                    "transcript" => "压缩中 · 整理记录…".to_string(),
                    "summary" => format!("压缩中 · 摘要 {delta_chars} 字"),
                    other => format!("压缩中 · {other}"),
                };
                self.compaction_status = Some(status);
            }
            UiEvent::CompactionFinished {
                outcome,
                tokens_before,
                tokens_after,
                pruned_count,
                first_kept_seq,
                summary_preview,
                message,
                is_manual,
            } => {
                self.is_compacting = false;
                self.compaction_status = None;
                // 遵循 dsh-TUI 原则：total_usage 计费读数永不随压缩回退，仅刷新上下文占用视图
                self.refresh_context_breakdown();

                match outcome.as_str() {
                    "pruned-only" => {
                        let delta = match (tokens_before, tokens_after) {
                            (Some(b), Some(a)) if b > a => {
                                format!("，{} → {} tok（↓{}）", b, a, b - a)
                            }
                            _ => String::new(),
                        };
                        self.push_notice(
                            NoticeLevel::Info,
                            format!(
                                "✓ 零模型剪枝完成：裁减 {pruned_count} 条大输出{delta}（原文保留，可用 recall_read 随时查阅）"
                            ),
                        );
                    }
                    "compacted" => {
                        let kept_tag = first_kept_seq
                            .map(|s| format!("保留自 #{s}"))
                            .unwrap_or_else(|| "已折叠".to_string());
                        let tok_tag = match (tokens_before, tokens_after) {
                            (Some(b), Some(a)) => {
                                let saved = b.saturating_sub(a);
                                format!(" · {b} → {a} tok (↓{saved})")
                            }
                            _ => String::new(),
                        };
                        let prune_tag = if pruned_count > 0 {
                            format!(" · 剪枝 {pruned_count} 条")
                        } else {
                            String::new()
                        };
                        let body = summary_preview
                            .as_deref()
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(|s| format!("\n{s}"))
                            .unwrap_or_default();
                        self.push_notice(
                            NoticeLevel::Info,
                            format!("✓ 上下文压缩完成（{kept_tag}{tok_tag}{prune_tag}）{body}"),
                        );
                    }
                    "skipped" => {
                        if is_manual {
                            let reason =
                                message.unwrap_or_else(|| "当前历史较短，暂无可压缩段落".into());
                            self.push_notice(NoticeLevel::Info, format!("跳过压缩：{reason}"));
                        }
                    }
                    "failed" => {
                        let err_msg = message.unwrap_or_else(|| "未知错误".into());
                        if !err_msg.contains("request aborted") {
                            self.push_notice(
                                NoticeLevel::Warn,
                                format!("上下文压缩失败：{err_msg}"),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn open_trajectory_with_report(&mut self, report: Option<TrajectoryReport>) {
        self.trajectory_scene.report = report;
        self.trajectory_scene.open = true;
    }

    /// 根据光标上下文同步 `/` 或 `@` 补全浮层。
    pub fn sync_input_completion_overlay(&mut self) {
        if self.overlay.is_modal() {
            return;
        }
        // Bash 模式下禁止触发 `/` 斜杠命令补全
        if self.bash_mode().is_none()
            && let Some((start, query)) = self.input.active_slash_trigger()
        {
            self.dismissed_mention = None;
            let selected = match &self.overlay.active {
                Some(OverlayKind::SlashComplete { selected, .. }) => *selected,
                _ => 0,
            };
            self.overlay.active = Some(OverlayKind::SlashComplete {
                token_start: start,
                query,
                selected,
            });
            return;
        }
        if let Some(caret) = self.input.active_mention_caret() {
            if self
                .dismissed_mention
                .as_ref()
                .is_some_and(|(s, q)| *s == caret.start && q == &caret.query)
            {
                if matches!(self.overlay.active, Some(OverlayKind::FileMention { .. })) {
                    self.overlay.active = None;
                }
                return;
            }
            self.dismissed_mention = None;
            let matches = self.file_cache.match_query(&self.workspace, &caret.query);
            let selected = match &self.overlay.active {
                Some(OverlayKind::FileMention {
                    matches: prev_matches,
                    selected: prev_sel,
                    ..
                }) => prev_matches
                    .get(*prev_sel)
                    .and_then(|prev| matches.iter().position(|m| m.path == prev.path))
                    .unwrap_or((*prev_sel).min(matches.len().saturating_sub(1))),
                _ => 0,
            };
            self.overlay.active = Some(OverlayKind::FileMention {
                token_start: caret.start,
                token_end: caret.end,
                path_end: caret.path_end,
                is_quoted: caret.is_quoted,
                query: caret.query,
                matches,
                selected,
            });
            return;
        }
        self.dismissed_mention = None;
        if matches!(
            self.overlay.active,
            Some(OverlayKind::SlashComplete { .. } | OverlayKind::FileMention { .. })
        ) {
            self.overlay.active = None;
        }
    }

    fn apply_overlay_action(&mut self, action: OverlayAction) -> AppAction {
        match action {
            OverlayAction::None => AppAction::None,
            OverlayAction::Close => {
                if let Some(OverlayKind::FileMention {
                    token_start, query, ..
                }) = self.overlay.active.take()
                {
                    self.dismissed_mention = Some((token_start, query));
                }
                AppAction::None
            }
            OverlayAction::CompleteSlash {
                token_start,
                command,
                execute_now,
            } => {
                self.overlay.active = None;
                let end = self.input.cursor_byte();
                if execute_now {
                    self.input.clear();
                    self.handle_slash_command(&command)
                } else {
                    self.input
                        .replace_byte_range(token_start, end, &format!("/{command} "));
                    AppAction::None
                }
            }
            OverlayAction::CompleteMention {
                token_start,
                token_end,
                path_end,
                is_quoted,
                candidate,
            } => {
                self.overlay.active = None;
                self.dismissed_mention = None;
                let len = self.input.text().len();
                let safe_end = token_end.min(len);
                let safe_path_end = path_end.min(safe_end);
                let suffix = self.input.text()[safe_path_end..safe_end].to_string();
                let (replacement, cursor_offset) =
                    format_mention_replacement(&candidate, &suffix, is_quoted);
                self.input.replace_byte_range_with_cursor(
                    token_start,
                    safe_end,
                    &replacement,
                    cursor_offset,
                );
                self.sync_input_completion_overlay();
                AppAction::None
            }
            OverlayAction::SelectModel { provider, model } => {
                self.overlay.active = None;
                self.provider = Some(provider);
                self.model = Some(model);
                self.push_notice(
                    NoticeLevel::Info,
                    format!("已切换后续请求模型为：{}", self.active_model_label()),
                );
                AppAction::None
            }
            OverlayAction::SelectEffort(effort) => {
                self.overlay.active = None;
                self.set_effort(effort);
                AppAction::None
            }
            OverlayAction::PreviewTheme(preset) => {
                self.set_theme_preset(preset);
                AppAction::None
            }
            OverlayAction::CommitTheme(preset) => {
                self.overlay.active = None;
                self.set_theme_preset(preset);
                self.set_flash(format!("✓ 主题已切换为 {}", preset.name()));
                AppAction::None
            }
            OverlayAction::RevertTheme(orig) => {
                self.overlay.active = None;
                self.set_theme_preset(orig);
                AppAction::None
            }
            OverlayAction::ApplyHistory(text) => {
                self.overlay.active = None;
                self.input.set_text_end(&text);
                AppAction::None
            }
            OverlayAction::InsertSkillSlash(skill_name) => {
                self.overlay.active = None;
                self.input.set_text_end(&format!("/{skill_name} "));
                AppAction::None
            }
            OverlayAction::ForkSessionAtSeq(seq) => {
                self.overlay.active = None;
                if self.is_busy {
                    self.push_notice(NoticeLevel::Warn, "请先中断当前生成后再分支回溯");
                    AppAction::None
                } else {
                    AppAction::ForkSessionAtSeq(seq)
                }
            }
            OverlayAction::UpdateSetting { key, value } => {
                if key.starts_with("plugin.") {
                    if self.is_busy || self.is_bash_running || self.is_compacting {
                        self.overlay.set_setting_toggle(&key, value != "true");
                        let msg = "当前有正在执行的任务，请等待完成后再切换模块";
                        self.push_notice(NoticeLevel::Warn, msg);
                        self.set_flash(msg);
                        return AppAction::None;
                    }
                    return AppAction::ApplySetting { key, value };
                }
                match key.as_str() {
                    "ui.theme" => {
                        if let Some(preset) = ThemePreset::parse(&value) {
                            self.set_theme_preset(preset);
                        }
                    }
                    "ui.showThinking" => {
                        self.show_thinking = value == "true";
                        self.item_cache.clear();
                    }
                    "ui.verboseExpanded" => {
                        self.verbose_expanded = value == "true";
                        self.item_cache.clear();
                    }
                    "ui.sidePanelOpen" => {
                        self.side_panel.open = value == "true";
                        self.item_cache.clear();
                    }
                    "ui.mouseCapture" => {
                        self.mouse_capture_enabled = value == "true";
                    }
                    "ui.approvalMode" => {
                        self.approval_mode = if value == "ask" {
                            ApprovalMode::Ask
                        } else {
                            ApprovalMode::Auto
                        };
                    }
                    "reasoningEffort" => {
                        if value == "default" {
                            self.reasoning_effort = None;
                        } else if let Some(eff) = parse_effort(&value) {
                            self.reasoning_effort = Some(eff);
                        }
                    }
                    _ => {}
                }
                self.set_flash(format!("✓ 已更新 {key} = {value}"));
                AppAction::ApplySetting { key, value }
            }
            OverlayAction::ResolveApproval { decision } => {
                if let Some(OverlayKind::Approval(mut ap)) = self.overlay.active.take() {
                    if decision == ApprovalDecision::AllowAlwaysForTool {
                        self.always_allowed_tools.insert(ap.tool_name.clone());
                    }
                    if let Some(tx) = ap.responder.take() {
                        let _ = tx.send(decision);
                    }
                    self.record_program_status(
                        if self.is_busy {
                            ProgramStatusState::Working
                        } else {
                            ProgramStatusState::Idle
                        },
                        None,
                        None,
                    );
                }
                AppAction::None
            }
        }
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

    pub fn handle_key(&mut self, key: KeyEvent) -> AppAction {
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return AppAction::None;
        }

        // 1. 全屏轨迹场景优先接管全部键盘输入
        if self.trajectory_scene.open {
            return match self.trajectory_scene.handle_key(key) {
                TrajectoryKeyOutcome::ReloadReport => {
                    AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                }
                TrajectoryKeyOutcome::Consumed | TrajectoryKeyOutcome::Closed => AppAction::None,
            };
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        // 2. Ctrl+C：若有活跃鼠标选区，优先复制选区并保持界面；队列选择器打开时只收起它；
        // 模态弹窗则先关闭/拒绝；
        // 非空输入优先由单次 Ctrl+C 清空（含补全浮层与运行中草稿）；否则走中断或双击退出
        if matches!(key.code, KeyCode::Char('c')) && ctrl {
            if self.selection.is_active()
                && let Some(buf) = &self.last_buffer
                && let Some(copied) =
                    extract_selected_text(buf, &self.selection, &self.selection_geom)
            {
                let _ = copy_to_clipboard(&copied);
                let chars = copied.chars().count();
                self.set_flash(format!("✓ 已复制 {chars} 字符"));
                return AppAction::None;
            }
            // 选择器打开时 Ctrl+C 只收起选择器，不能误触成 CancelRun / 退出。
            if self.queue_selected.is_some() {
                self.queue_selected = None;
                return AppAction::None;
            }
            if self.overlay.is_modal() {
                if let Some(OverlayKind::Approval(_)) = self.overlay.active {
                    return self.apply_overlay_action(OverlayAction::ResolveApproval {
                        decision: ApprovalDecision::Deny,
                    });
                }
                self.overlay.active = None;
                return AppAction::None;
            }
            if !self.input.is_empty() {
                self.input.clear();
                self.sync_input_completion_overlay();
                self.last_ctrl_c_at = None;
                return AppAction::None;
            }
            if self.is_busy || self.is_bash_running || self.is_compacting {
                return AppAction::CancelRun;
            }
            let double_press = self
                .last_ctrl_c_at
                .is_some_and(|t| t.elapsed() < Duration::from_millis(1200));
            if double_press || self.items.is_empty() {
                self.should_quit = true;
                return AppAction::Quit;
            }
            self.last_ctrl_c_at = Some(Instant::now());
            self.set_flash("再按一次 Ctrl+C 退出");
            return AppAction::None;
        }

        // 3. 浮层栈（模态与补全）按键分派
        if let Some(action) =
            self.overlay
                .handle_key(key, &self.user_skill_names, self.input.history())
        {
            return self.apply_overlay_action(action);
        }

        // 3.5 队列选择器：浮层之后、全局快捷键之前，才不会被补全或历史回溯劫走
        if self.queue_selected.is_some()
            && let Some(action) = self.handle_queue_selector_key(key, ctrl, alt, shift)
        {
            return action;
        }

        // 4. 全局功能快捷键
        match key.code {
            KeyCode::Char('d') if ctrl => {
                if !self.is_busy && !self.is_bash_running && self.input.is_empty() {
                    self.should_quit = true;
                    return AppAction::Quit;
                }
                return AppAction::None;
            }
            KeyCode::Char('t') if ctrl => {
                // 保持原有 Ctrl+T 开关思考的兼容，但若加 Shift 或无 Alt 歧义：
                // 这里把 Ctrl+T 设为打开全屏轨迹（对齐 dsh-TUI），Ctrl+O 开关思考/详情展开！
                return AppAction::OpenTrajectoryScene(self.trajectory_scene.mode);
            }
            KeyCode::Char('o') if ctrl => {
                let next = !self.verbose_expanded;
                self.verbose_expanded = next;
                self.show_thinking = next;
                self.item_cache.clear();
                let state = if next { "展开" } else { "收起" };
                self.set_flash(format!("思考与工具详情已{state}"));
                return AppAction::None;
            }
            KeyCode::Char('b') if ctrl => {
                self.side_panel.open = !self.side_panel.open;
                self.item_cache.clear();
                if self.side_panel.open {
                    return AppAction::RefreshSidePanelData;
                }
                return AppAction::None;
            }
            KeyCode::Char('r') if ctrl => {
                self.overlay.open_history_search(self.input.history());
                return AppAction::None;
            }
            KeyCode::Char('e') if ctrl => {
                if self.input.expand_pastes_in_place() {
                    self.set_flash("已展开折叠粘贴块");
                }
                return AppAction::None;
            }
            KeyCode::Char('z') if ctrl => {
                self.input.undo();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('y') if ctrl => {
                if self.input.yank() {
                    self.sync_input_completion_overlay();
                }
                return AppAction::None;
            }
            KeyCode::Char('g') if ctrl => {
                if !self.is_busy && !self.is_bash_running {
                    return AppAction::OpenExternalEditor;
                }
                return AppAction::None;
            }
            KeyCode::F(2) => {
                self.mouse_capture_enabled = !self.mouse_capture_enabled;
                return AppAction::SetMouseCapture(self.mouse_capture_enabled);
            }
            KeyCode::F(3) => {
                let next = match self.screen_mode {
                    ScreenMode::Fullscreen => ScreenMode::Inline,
                    ScreenMode::Inline => ScreenMode::Fullscreen,
                };
                return AppAction::SwitchScreenMode(next);
            }
            KeyCode::Char('w') if ctrl => {
                self.input.delete_word_backward();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('k') if ctrl => {
                self.input.delete_to_line_end();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char('l') if ctrl => {
                return self.open_model_selector();
            }
            KeyCode::Char('v' | 'V') if ctrl && !alt => {
                return AppAction::PasteClipboard;
            }
            KeyCode::Esc => {
                if self.selection.is_active() {
                    self.selection.clear();
                    return AppAction::None;
                }
                if self.is_busy || self.is_bash_running || self.is_compacting {
                    return AppAction::CancelRun;
                }
                if !self.input.is_empty() {
                    self.input.clear();
                    self.sync_input_completion_overlay();
                }
                return AppAction::None;
            }
            KeyCode::Char('?') if !ctrl && !alt && self.input.is_empty() => {
                self.overlay.open_help();
                return AppAction::None;
            }
            KeyCode::PageUp => {
                self.scroll_up(self.page_step());
                return AppAction::None;
            }
            KeyCode::PageDown => {
                self.scroll_down(self.page_step());
                return AppAction::None;
            }
            KeyCode::Char('u') if ctrl => {
                self.scroll_up(self.page_step());
                return AppAction::None;
            }
            KeyCode::Tab if !ctrl && !alt => {
                self.cycle_reasoning_effort();
                return AppAction::None;
            }
            KeyCode::Enter if ctrl && !alt && !shift => return self.prepare_submit(true),
            KeyCode::Enter if alt && !ctrl && !shift => {
                // 草稿空时 Alt+Enter 原来的语义（往空草稿里塞换行）没有意义，
                // 正好当作不依赖 Kitty 键盘协议的立即发送备用入口。
                if self.input.is_empty() && !self.queued_prompts.is_empty() {
                    return self.prepare_submit(true);
                }
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Enter if shift || alt || ctrl => {
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Char('j') if ctrl => {
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Enter => return self.prepare_submit(false),
            KeyCode::Backspace => {
                self.input.backspace();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Delete => {
                self.input.delete();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Left if alt || ctrl => {
                self.input.move_word_left();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Right if alt || ctrl => {
                self.input.move_word_right();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Left => {
                self.input.move_left();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Right => {
                self.input.move_right();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Up if alt && !ctrl && !shift => {
                // Alt+↑：一步把队尾撤回输入框直接编辑，不用先进选择器再按 ⏎。
                // 草稿非空时与草稿交换（草稿进队尾），不会丢字。
                //
                // 终端兼容性：kitty / xterm modifyOtherKeys / Ghostty / iTerm2 发的是
                // CSI 1;3A，解析为 Up+ALT，走这条路；少数开启 altSendsEscape 的终端
                // 发的是 ESC ESC [ A，crossterm 会拆成 Esc + 字面 "[A"，
                // 那种终端请用 ↑ 进选择器（实测见 issue #15）。
                if let Some(last) = self.queued_prompts.last().map(|item| item.id)
                    && !self.is_bash_running
                {
                    self.retract_queued(last);
                    return AppAction::None;
                }
                self.input.move_up();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Up => {
                // 草稿空且队列非空时，↑ 进入队列选择器（优先于历史回溯）。
                if self.input.is_empty()
                    && !self.queued_prompts.is_empty()
                    && !self.overlay.is_open()
                    && !self.is_bash_running
                {
                    self.queue_selected = self.queued_prompts.last().map(|item| item.id);
                    return AppAction::None;
                }
                self.input.move_up();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Down => {
                self.input.move_down();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Home if ctrl => {
                self.auto_scroll = false;
                self.scroll_from_top = 0;
                return AppAction::None;
            }
            KeyCode::End if ctrl => {
                self.scroll_to_bottom();
                return AppAction::None;
            }
            KeyCode::Home => {
                self.input.move_home();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::End => {
                self.input.move_end();
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            KeyCode::Char(ch) if !ctrl && !alt => {
                self.input.insert_char(ch);
                self.sync_input_completion_overlay();
                return AppAction::None;
            }
            _ => {}
        }
        AppAction::None
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> AppAction {
        if self.trajectory_scene.open {
            return match self.trajectory_scene.handle_mouse(mouse) {
                TrajectoryKeyOutcome::ReloadReport => {
                    AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                }
                _ => AppAction::None,
            };
        }

        if let Some(action) = self.overlay.handle_mouse(mouse, &self.user_skill_names) {
            return self.apply_overlay_action(action);
        }

        let col = mouse.column;
        let row = mouse.row;
        let total_w = self
            .last_buffer
            .as_ref()
            .map(|b| b.area.width)
            .unwrap_or(120);

        match mouse.kind {
            MouseEventKind::Moved => {
                self.side_panel.hovered_divider = self.side_panel.divider_col == Some(col);
                self.hovered_rail_row = self
                    .rail_geo
                    .as_ref()
                    .and_then(|g| hit_test_rail(g, col, row).map(|_| row));
            }
            MouseEventKind::Down(MouseButton::Left) => {
                // 1. 侧栏分割线拖拽
                if self.side_panel.divider_col == Some(col) {
                    self.side_panel.dragging_divider = true;
                    return AppAction::None;
                }
                // 2. 侧栏页签点击
                for (tab, rect) in &self.side_panel.tab_hit_boxes {
                    if col >= rect.x
                        && col < rect.x + rect.width
                        && row >= rect.y
                        && row < rect.y + rect.height
                    {
                        self.side_panel.tab = *tab;
                        self.side_panel.scroll = 0;
                        return AppAction::RefreshSidePanelData;
                    }
                }
                // 3. 时间轴导轨点击跳转
                if let Some(geo) = &self.rail_geo
                    && let Some(hit) = hit_test_rail(geo, col, row)
                {
                    let target_idx = match hit {
                        RailHit::Up => self.timeline_snap.up_idx,
                        RailHit::Down => self.timeline_snap.down_idx,
                        RailHit::Tick(idx) => Some(idx),
                    };
                    if let Some(idx) = target_idx
                        && let Some(turn) = self.timeline_snap.turns.get(idx)
                    {
                        self.auto_scroll = false;
                        self.scroll_from_top = turn.top_line.min(u16::MAX as usize) as u16;
                    }
                    return AppAction::None;
                }
                // 4. 启动应用级文本选区
                self.selection.start_press(
                    col,
                    row,
                    &self.selection_geom,
                    self.last_buffer.as_ref(),
                );
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                if self.side_panel.dragging_divider {
                    self.side_panel.set_width_from_divider_col(total_w, col);
                    self.item_cache.clear();
                    return AppAction::None;
                }
                self.selection
                    .update_drag(col, row, self.last_buffer.as_ref());
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.side_panel.dragging_divider {
                    self.side_panel.dragging_divider = false;
                    return AppAction::None;
                }
                let had_drag_selection = self.selection.finish_release();
                if had_drag_selection {
                    if let Some(buf) = &self.last_buffer
                        && let Some(text) =
                            extract_selected_text(buf, &self.selection, &self.selection_geom)
                    {
                        let _ = copy_to_clipboard(&text);
                        self.set_flash(format!("✓ 已复制 {} 字符", text.chars().count()));
                    }
                    return AppAction::None;
                }
                // 单击未拖动：检查是否点中 Launchpad 快捷入口
                for (qa, rect) in &self.launchpad_hit_rects {
                    if col >= rect.x
                        && col < rect.x + rect.width
                        && row >= rect.y
                        && row < rect.y + rect.height
                    {
                        return match qa.clone() {
                            LaunchpadQuickAction::ModelPicker => self.open_model_selector(),
                            LaunchpadQuickAction::ThemePicker => {
                                self.overlay.open_theme_picker(self.theme_preset);
                                AppAction::None
                            }
                            LaunchpadQuickAction::SkillPicker => AppAction::OpenSkillPicker,
                            LaunchpadQuickAction::TrajectoryScene => {
                                AppAction::OpenTrajectoryScene(self.trajectory_scene.mode)
                            }
                            LaunchpadQuickAction::ToggleSidePanel => {
                                self.side_panel.open = !self.side_panel.open;
                                self.item_cache.clear();
                                if self.side_panel.open {
                                    AppAction::RefreshSidePanelData
                                } else {
                                    AppAction::None
                                }
                            }
                        };
                    }
                }
            }
            MouseEventKind::ScrollUp => {
                if let Some(sp_rect) = self.selection_geom.side_panel_rect
                    && col >= sp_rect.x
                {
                    self.side_panel.scroll = self.side_panel.scroll.saturating_sub(2);
                } else {
                    self.scroll_up(3);
                }
            }
            MouseEventKind::ScrollDown => {
                if let Some(sp_rect) = self.selection_geom.side_panel_rect
                    && col >= sp_rect.x
                {
                    self.side_panel.scroll = self.side_panel.scroll.saturating_add(2);
                } else {
                    self.scroll_down(3);
                }
            }
            _ => {}
        }
        AppAction::None
    }

    pub fn handle_paste(&mut self, data: &str) {
        self.input.insert_paste(data);
        self.sync_input_completion_overlay();
    }

    fn page_step(&self) -> u16 {
        self.last_viewport_height.saturating_sub(2).max(3)
    }

    pub fn scroll_up(&mut self, lines: u16) {
        let max_scroll = self
            .last_total_lines
            .saturating_sub(self.last_viewport_height);
        let cur = if self.auto_scroll {
            max_scroll
        } else {
            self.scroll_from_top.min(max_scroll)
        };
        self.auto_scroll = false;
        self.scroll_from_top = cur.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: u16) {
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
        } else if self.scroll_from_top >= max_scroll {
            self.scroll_from_top = max_scroll;
            self.auto_scroll = true;
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
        self.items.clear();
        self.tool_extras.clear();
        self.item_cache.clear();
        self.touched_files.clear();
        self.anomalies.clear();
        self.queued_prompts.clear();
        self.queue_selected = None;
        self.is_bash_running = false;
        self.is_compacting = false;
        self.compaction_status = None;
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
        self.streaming_assistant_open = false;
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
    fn enqueue_prompt_with_images(
        &mut self,
        prompt: String,
        images: Vec<(String, ImageAttachment)>,
    ) {
        let id = self.next_queue_id;
        self.next_queue_id = self.next_queue_id.wrapping_add(1);
        self.queued_prompts.push(QueuedPrompt {
            id,
            text: prompt,
            images,
        });
        self.queue_selected = None;
        self.set_flash(format!(
            "已加入队列 (共 {} 条待发)，↑ 可撤回编辑",
            self.queued_prompts.len()
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
        let index = self.queued_prompts.iter().position(|item| item.id == id)?;
        let item = self.queued_prompts.remove(index);
        self.sync_queue_selection();
        Some(item)
    }

    /// 取出队首一条；`runner` 在上一轮成功收尾后调用它继续发送。
    pub fn dequeue_next(&mut self) -> Option<QueuedPrompt> {
        let id = self.queued_prompts.first()?.id;
        self.dequeue_prompt(id)
    }

    /// 选中项已不存在（出队/丢弃/会话切换）时收起选择器，避免高亮指向空行。
    fn sync_queue_selection(&mut self) {
        if let Some(selected) = self.queue_selected
            && !self.queued_prompts.iter().any(|item| item.id == selected)
        {
            self.queue_selected = None;
        }
    }

    /// 把队列项撤回输入框：草稿为空则直接填入，非空则与草稿交换。
    /// 交换而非覆盖，是为了不丢主人已经写下的字；两者互斥，保证同一段文字只在一处。
    fn retract_queued(&mut self, id: u64) -> bool {
        let Some(item) = self.dequeue_prompt(id) else {
            return false;
        };
        if self.input.text().trim().is_empty() {
            self.input
                .replace_draft_from_queue_with_images(&item.text, &item.images);
            self.set_flash("已撤回队列项，Enter 重新发送");
        } else {
            let (draft_raw, draft_images) = self.input.take_with_images();
            let draft = draft_raw.trim().to_string();
            let swap_id = self.next_queue_id;
            self.next_queue_id = self.next_queue_id.wrapping_add(1);
            self.queued_prompts.push(QueuedPrompt {
                id: swap_id,
                text: draft,
                images: draft_images,
            });
            self.input
                .replace_draft_from_queue_with_images(&item.text, &item.images);
            self.set_flash("已交换：草稿已存入队尾");
        }
        self.sync_input_completion_overlay();
        true
    }

    /// 丢弃选中的队列项（不发送）。
    fn discard_queued(&mut self, id: u64) -> bool {
        let dropped = self.dequeue_prompt(id).is_some();
        if dropped {
            self.set_flash("已丢弃队列项");
        }
        dropped
    }

    /// 立即发送队首：打断当前回合，在 `RunFinished` 后直接发出。
    fn steer_queue_head(&mut self) -> AppAction {
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
    fn submit_queue_head(&mut self) -> AppAction {
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

    /// 队列选择器的按键处理：↑↓ 移动，⏎ 撤回，Ctrl/Alt+⏎ 撤回并立即发送，Del 丢弃，Esc 关闭。
    fn handle_queue_selector_key(
        &mut self,
        key: KeyEvent,
        ctrl: bool,
        alt: bool,
        shift: bool,
    ) -> Option<AppAction> {
        let selected = self.queue_selected?;
        let index = self
            .queued_prompts
            .iter()
            .position(|item| item.id == selected);
        let Some(index) = index else {
            self.queue_selected = None;
            return None;
        };
        let len = self.queued_prompts.len();
        match key.code {
            KeyCode::Up if !ctrl && !alt && !shift => {
                if index == 0 {
                    // 越过首项就交出选择器，把这次按键还给历史回溯。
                    self.queue_selected = None;
                    self.input.move_up();
                    self.sync_input_completion_overlay();
                } else {
                    self.queue_selected = Some(self.queued_prompts[index - 1].id);
                }
                Some(AppAction::None)
            }
            KeyCode::Down if !ctrl && !alt && !shift => {
                if index + 1 >= len {
                    self.queue_selected = None;
                    self.input.move_down();
                    self.sync_input_completion_overlay();
                } else {
                    self.queue_selected = Some(self.queued_prompts[index + 1].id);
                }
                Some(AppAction::None)
            }
            KeyCode::Enter if ctrl || alt => {
                let item = self.dequeue_prompt(selected)?;
                if !item.images.is_empty() && !self.active_model_supports_images() {
                    self.queue_selected = None;
                    self.input
                        .replace_draft_from_queue_with_images(&item.text, &item.images);
                    self.reject_unsupported_images_notice();
                    return Some(AppAction::None);
                }
                self.set_flash("⚡ 已撤回队列项并立即发送…");
                if item.images.is_empty() {
                    Some(AppAction::SteerPrompt(item.text))
                } else {
                    Some(AppAction::SteerPromptWithImages {
                        text: item.text,
                        images: item.images.into_iter().map(|(_, img)| img).collect(),
                    })
                }
            }
            KeyCode::Enter if !shift => {
                self.retract_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Delete if !ctrl && !alt => {
                self.discard_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Char('x') if ctrl => {
                self.discard_queued(selected);
                Some(AppAction::None)
            }
            KeyCode::Esc => {
                self.queue_selected = None;
                Some(AppAction::None)
            }
            // 其他键（含普通字符）交还给正常编辑：先收起选择器，再让按键继续下落。
            _ => {
                self.queue_selected = None;
                None
            }
        }
    }

    fn prepare_submit(&mut self, immediate: bool) -> AppAction {
        let preview = self.input.text().trim();
        if preview.is_empty() {
            self.input.take();
            // 草稿为空时队列才是「主人现在想发的东西」：立即发送插队发队首，
            // 普通回车在空闲时直接发队首，运行中则保留排队等本轮收尾。
            if self.queue_selected.is_some() {
                return AppAction::None;
            }
            if !self.queued_prompts.is_empty() {
                if immediate {
                    return self.steer_queue_head();
                }
                if !self.is_busy && !self.is_bash_running && !self.is_compacting {
                    return self.submit_queue_head();
                }
            }
            return AppAction::None;
        }

        if let Some(mode) = BashInputMode::from_input(preview) {
            let expanded = self.input.expand_text();
            let cmd = mode.extract_command(&expanded);
            if cmd.is_empty() {
                self.set_flash("请输入要在 Bash 中执行的命令（或按 Esc 退出）");
                return AppAction::None;
            }
            if self.is_busy {
                self.set_flash("模型正在生成中，请等待回合结束或按 Esc 取消后再运行 Bash 命令");
                return AppAction::None;
            }
            if self.is_bash_running {
                self.set_flash("已有 Bash 命令正在运行，请先按 Esc 取消");
                return AppAction::None;
            }

            let raw = self.input.take();
            let trimmed = raw.trim();
            let command = mode.extract_command(trimmed).to_string();
            self.overlay.active = None;
            self.dismissed_mention = None;
            self.input.push_history(trimmed);
            self.scroll_to_bottom();
            return AppAction::RunLocalBash {
                command,
                exclude_from_context: mode.exclude_from_context(),
            };
        }

        if self.is_bash_running {
            let is_safe_slash = preview
                .strip_prefix('/')
                .and_then(|s| s.split_whitespace().next())
                .is_some_and(|c| {
                    matches!(
                        c.to_ascii_lowercase().as_str(),
                        "exit" | "quit" | "q" | "help" | "h" | "view" | "mouse"
                    )
                });
            if !is_safe_slash {
                self.set_flash("Bash 命令正在运行，请等待完成或按 Esc 取消后再发送");
                return AppAction::None;
            }
        }

        let is_builtin_cmd = preview
            .strip_prefix('/')
            .map(str::trim)
            .and_then(|s| s.split_whitespace().next())
            .is_some_and(|first| is_builtin_slash_cmd(&first.to_ascii_lowercase()));

        if !is_builtin_cmd && self.input.has_images() {
            if self.input.image_count() > MAX_IMAGES_PER_MESSAGE {
                let msg = format!(
                    "单条消息最多附加 {MAX_IMAGES_PER_MESSAGE} 张图片，请先删减多余图片后再发送"
                );
                self.set_flash(msg.clone());
                self.push_notice(NoticeLevel::Warn, msg);
                return AppAction::None;
            }
            if !self.active_model_supports_images() {
                self.reject_unsupported_images_notice();
                return AppAction::None;
            }
        }

        let (raw, images) = self.input.take_with_images();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return AppAction::None;
        }
        self.overlay.active = None;
        self.dismissed_mention = None;
        self.input.push_history(trimmed);
        self.scroll_to_bottom();

        if let Some(cmd_line) = trimmed.strip_prefix('/') {
            let cmd_line = cmd_line.trim();
            let first = cmd_line.split_whitespace().next().unwrap_or("");
            // 对齐 dsh-TUI / pi：仅拦截已注册内置命令；
            // 技能手势（/<skill>）、绝对路径（/Users/...）以及未知 / 前缀一律作为普通消息发给模型。
            if is_builtin_slash_cmd(&first.to_ascii_lowercase()) {
                return self.handle_slash_command(cmd_line);
            }
        }

        let raw_images: Vec<ImageAttachment> = images.iter().map(|(_, img)| img.clone()).collect();
        if self.is_busy {
            let prompt = trimmed.to_string();
            if immediate {
                self.set_flash("⚡ 正在中断当前回合并立即发送…");
                if raw_images.is_empty() {
                    return AppAction::SteerPrompt(prompt);
                }
                return AppAction::SteerPromptWithImages {
                    text: prompt,
                    images: raw_images,
                };
            }
            self.enqueue_prompt_with_images(prompt, images);
            return AppAction::None;
        }

        self.push_user_message(trimmed);
        if raw_images.is_empty() {
            AppAction::SubmitPrompt(trimmed.to_string())
        } else {
            AppAction::SubmitPromptWithImages {
                text: trimmed.to_string(),
                images: raw_images,
            }
        }
    }

    fn handle_slash_command(&mut self, cmd_line: &str) -> AppAction {
        let mut parts = cmd_line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
        let arg = parts.next().map(str::trim).filter(|s| !s.is_empty());

        match cmd.as_str() {
            "exit" | "quit" | "q" => {
                self.should_quit = true;
                AppAction::Quit
            }
            "help" | "h" => {
                self.overlay.open_help();
                AppAction::None
            }
            "settings" | "config" => AppAction::OpenSettings(arg.map(ToString::to_string)),
            "tools" => AppAction::ShowTools,
            "mcp" => AppAction::ShowMcp,
            "skills" => match arg {
                Some("list") => AppAction::ShowSkills,
                _ => AppAction::OpenSkillPicker,
            },
            "instructions" => AppAction::ShowInstructions,
            "tree" => AppAction::OpenSessionTree,
            "view" => {
                let next = match arg {
                    Some("inline") => ScreenMode::Inline,
                    Some("fullscreen" | "full") => ScreenMode::Fullscreen,
                    None => match self.screen_mode {
                        ScreenMode::Fullscreen => ScreenMode::Inline,
                        ScreenMode::Inline => ScreenMode::Fullscreen,
                    },
                    Some(other) => {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!("未知视图模式 `{other}`，可选：fullscreen / inline"),
                        );
                        return AppAction::None;
                    }
                };
                AppAction::SwitchScreenMode(next)
            }
            "mouse" => {
                let next = match arg {
                    Some("on" | "true") => true,
                    Some("off" | "false") => false,
                    _ => !self.mouse_capture_enabled,
                };
                self.mouse_capture_enabled = next;
                AppAction::SetMouseCapture(next)
            }
            "trace" => AppAction::OpenTrajectoryScene(self.trajectory_scene.mode),
            "panel" => {
                self.side_panel.open = !self.side_panel.open;
                self.item_cache.clear();
                if self.side_panel.open {
                    AppAction::RefreshSidePanelData
                } else {
                    AppAction::None
                }
            }
            "theme" => match arg {
                None => {
                    self.overlay.open_theme_picker(self.theme_preset);
                    AppAction::None
                }
                Some(name) => {
                    if let Some(preset) = ThemePreset::parse(name) {
                        self.set_theme_preset(preset);
                        self.set_flash(format!("✓ 主题已切换为 {}", preset.name()));
                    } else {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!(
                                "未知主题 `{name}`，可选：auto / mist-dark / mist-light / sakura-dark / system / ansi16"
                            ),
                        );
                    }
                    AppAction::None
                }
            },
            "approval" => match arg {
                None => {
                    self.push_notice(
                        NoticeLevel::Info,
                        format!(
                            "当前工具审批模式：{}（可用 `/approval auto` 或 `/approval ask` 切换）",
                            self.approval_mode.as_str()
                        ),
                    );
                    AppAction::None
                }
                Some("auto" | "off") => {
                    self.approval_mode = ApprovalMode::Auto;
                    self.push_notice(NoticeLevel::Info, "工具审批模式已设为：auto（自动放行）");
                    AppAction::None
                }
                Some("ask" | "on") => {
                    self.approval_mode = ApprovalMode::Ask;
                    self.push_notice(
                        NoticeLevel::Info,
                        "工具审批模式已设为：ask（写文件与 Shell 将弹窗确认）",
                    );
                    AppAction::None
                }
                Some(other) => {
                    self.push_notice(
                        NoticeLevel::Warn,
                        format!("未知审批模式 `{other}`，可选：auto / ask"),
                    );
                    AppAction::None
                }
            },
            "clear" | "new" => {
                if self.is_busy {
                    self.push_notice(NoticeLevel::Warn, "请先按 Esc 中断当前任务再开启新会话");
                    AppAction::None
                } else {
                    AppAction::NewSession
                }
            }
            "thinking" => {
                self.show_thinking = !self.show_thinking;
                self.item_cache.clear();
                let state = if self.show_thinking {
                    "开启"
                } else {
                    "隐藏"
                };
                self.push_notice(NoticeLevel::Info, format!("思考过程显示已{state}"));
                AppAction::None
            }
            "model" => match arg {
                None => self.open_model_selector(),
                other => AppAction::ShowOrSwitchModel(other.map(ToString::to_string)),
            },
            "effort" => match arg {
                None => {
                    self.overlay.open_effort_picker(self.reasoning_effort);
                    AppAction::None
                }
                Some(raw_effort) => {
                    if let Some(parsed) = parse_effort(raw_effort) {
                        self.set_effort(parsed);
                    } else {
                        self.push_notice(
                            NoticeLevel::Warn,
                            format!(
                                "未知推理强度 `{raw_effort}`，可选：off / minimal / low / medium / high / xhigh / max"
                            ),
                        );
                    }
                    AppAction::None
                }
            },
            "stats" => AppAction::ShowStats,
            "compact" => {
                if self.is_busy || self.is_bash_running || self.is_compacting {
                    self.push_notice(
                        NoticeLevel::Warn,
                        "当前已有任务或压缩正在运行，请先等待完成或按 Esc 取消后再执行 /compact",
                    );
                    AppAction::None
                } else {
                    AppAction::CompactSession(arg.map(ToString::to_string))
                }
            }
            "export" => AppAction::ExportTrajectory(arg.map(PathBuf::from)),
            _ => AppAction::None,
        }
    }
}

fn is_builtin_slash_cmd(cmd: &str) -> bool {
    matches!(
        cmd,
        "exit"
            | "quit"
            | "q"
            | "help"
            | "h"
            | "settings"
            | "config"
            | "tools"
            | "mcp"
            | "skills"
            | "instructions"
            | "clear"
            | "new"
            | "thinking"
            | "model"
            | "effort"
            | "stats"
            | "compact"
            | "export"
            | "trace"
            | "tree"
            | "view"
            | "mouse"
            | "panel"
            | "theme"
            | "approval"
    )
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
    format!("[{source}] {text}")
}

fn parse_effort(raw: &str) -> Option<ReasoningEffort> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" => Some(ReasoningEffort::Off),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

pub fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

pub fn format_tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

pub fn truncate_display(s: &str, max_cols: usize) -> String {
    let mut out = String::new();
    let mut width = 0usize;
    for ch in s.chars() {
        let ch = if ch == '\n' || ch == '\r' { ' ' } else { ch };
        let w = char_width(ch);
        if width + w > max_cols.saturating_sub(1) {
            out.push('…');
            return out;
        }
        out.push(ch);
        width += w;
    }
    out
}

const MAX_ARGS_SUMMARY_COLS: usize = 512;
const MAX_PREVIEW_LINE_COLS: usize = 240;

fn single_line_text(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_home_prefix_boundary(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '"' | '\'' | '=' | '(' | ',' | ':' | ';' | '`')
}

fn is_home_suffix_boundary(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '/' | '\\' | '"' | '\'' | ')' | ',' | ':' | ';' | '&' | '|' | '`'
        )
}

/// 将文本中的用户家目录绝对路径缩写为 `~`（带边界检查，避免误替换同前缀用户名路径）。
pub(crate) fn abbreviate_home_with(text: &str, home: Option<&str>) -> String {
    let Some(raw_home) = home else {
        return text.to_string();
    };
    let home_trimmed = raw_home.trim_end_matches(['/', '\\']);
    if home_trimmed.is_empty() || home_trimmed == "/" {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut pos = 0usize;
    while pos < text.len() {
        let Some(rel) = text[pos..].find(home_trimmed) else {
            out.push_str(&text[pos..]);
            break;
        };
        let match_start = pos + rel;
        let match_end = match_start + home_trimmed.len();
        out.push_str(&text[pos..match_start]);

        let left_ok = match_start == 0
            || text[..match_start]
                .chars()
                .next_back()
                .is_some_and(is_home_prefix_boundary);
        let right_ok = match_end == text.len()
            || text[match_end..]
                .chars()
                .next()
                .is_some_and(is_home_suffix_boundary);

        if left_ok && right_ok {
            out.push('~');
            pos = match_end;
        } else if let Some(ch) = text[match_start..].chars().next() {
            out.push(ch);
            pos = match_start + ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// 使用当前系统家目录将路径或命令中的 `$HOME` 缩写为 `~`。
pub fn abbreviate_home_path(text: &str) -> String {
    let home = dirs::home_dir();
    let home_str = home.as_ref().and_then(|p| p.to_str());
    abbreviate_home_with(text, home_str)
}

/// 判断是否为只读/检索/发现类工具（`Done` 态在默认折叠模式下单行收口）。
pub(crate) fn is_read_only_compact_tool(name: &str) -> bool {
    matches!(
        name,
        "read"
            | "read_file"
            | "list_dir"
            | "ls"
            | "grep"
            | "find"
            | "tool_search"
            | "skill"
            | "list_mcp_resources"
            | "list_mcp_resource_templates"
            | "read_mcp_resource"
            | "get_time"
    )
}

pub(crate) fn bash_scope_label(exclude_from_context: bool) -> &'static str {
    if exclude_from_context {
        "仅本地"
    } else {
        "已入上下文"
    }
}

pub(crate) fn bash_inline_meta(
    exclude_from_context: bool,
    exit_code: Option<i32>,
    cancelled: bool,
) -> String {
    let scope = bash_scope_label(exclude_from_context);
    if cancelled {
        format!("{scope} · 已取消")
    } else if let Some(code) = exit_code
        && code != 0
    {
        format!("{scope} · 退出码 {code}")
    } else {
        scope.to_string()
    }
}

pub(crate) fn format_bash_display(output: &str, exit_code: Option<i32>, cancelled: bool) -> String {
    let mut display = if output.trim().is_empty() {
        "(no output)".to_string()
    } else {
        output.to_string()
    };
    if cancelled {
        display.push_str("\n\nCommand aborted");
    } else if let Some(code) = exit_code
        && code != 0
    {
        display.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    display
}

/// 判断工具的主要参数是否为路径（窄宽度时优先保留首部前缀与尾部文件名）。
pub(crate) fn is_path_arg_tool(name: &str) -> bool {
    matches!(
        name,
        "read" | "read_file" | "list_dir" | "ls" | "edit" | "write" | "read_mcp_resource"
    )
}

pub(crate) fn count_diff_stats(diff: &str) -> (usize, usize) {
    let mut adds = 0usize;
    let mut dels = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++ ") || line.starts_with("--- ") {
            continue;
        }
        if line.starts_with('+') {
            adds += 1;
        } else if line.starts_with('-') {
            dels += 1;
        }
    }
    (adds, dels)
}

pub(crate) fn format_diff_stat(adds: usize, dels: usize) -> Option<String> {
    if adds == 0 && dels == 0 {
        None
    } else {
        Some(format!("+{adds} -{dels}"))
    }
}

fn plural_unit(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("1 {singular}")
    } else {
        format!("{count} {plural}")
    }
}

pub(crate) fn clean_bash_output_lines(content: &str, keep_error_footer: bool) -> Vec<&str> {
    content
        .lines()
        .map(str::trim_end)
        .filter(|line| {
            let t = line.trim();
            if t.is_empty()
                || t == "(no output)"
                || t.starts_with("[Showing lines ")
                || t.starts_with("[Showing last ")
            {
                return false;
            }
            if !keep_error_footer
                && (t == "Command aborted"
                    || t.starts_with("Command timed out after ")
                    || t.starts_with("Command exited with code "))
            {
                return false;
            }
            true
        })
        .collect()
}

fn value_inline(val: &Value) -> String {
    match val {
        Value::String(s) => single_line_text(s),
        other => other.to_string(),
    }
}

/// 为展开态（`Ctrl+O`）解析完整参数与附带脚本/写入正文。
pub(crate) fn format_expanded_tool_args(
    name: &str,
    args_summary: &str,
    extra: Option<&ToolExtraMeta>,
) -> (String, Vec<String>) {
    let Some(raw) = extra
        .map(|e| e.full_args.trim())
        .filter(|s| !s.is_empty() && *s != "{}")
    else {
        return (args_summary.to_string(), Vec::new());
    };

    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(raw) else {
        return (single_line_text(raw), Vec::new());
    };

    if name == "bash"
        && let Some(cmd) = map.get("command").and_then(Value::as_str)
    {
        if cmd.contains('\n') {
            let mut extra_lines = Vec::new();
            for (i, line) in cmd.lines().enumerate() {
                if i == 0 {
                    extra_lines.push(format!("$ {line}"));
                } else {
                    extra_lines.push(format!("  {line}"));
                }
            }
            let first = single_line_text(cmd.lines().next().unwrap_or(""));
            return (first, extra_lines);
        }
        return (cmd.trim().to_string(), Vec::new());
    }

    if name == "codemode"
        && let Some(code) = map.get("code").and_then(Value::as_str)
    {
        let n = code.lines().count().max(1);
        let extra_lines: Vec<String> = code.lines().map(ToString::to_string).collect();
        return (format!("{n} lines"), extra_lines);
    }

    if name == "write"
        && let Some(path) = map.get("path").and_then(Value::as_str)
    {
        let extra_lines = map
            .get("content")
            .and_then(Value::as_str)
            .map(|c| c.lines().map(ToString::to_string).collect())
            .unwrap_or_default();
        return (path.to_string(), extra_lines);
    }

    if name == "edit"
        && let Some(path) = map.get("path").and_then(Value::as_str)
    {
        return (path.to_string(), Vec::new());
    }

    let mut pairs = Vec::new();
    for (k, v) in &map {
        pairs.push(format!("{k}={}", value_inline(v)));
    }
    if map.len() == 1
        && let Some(val) = map.values().next()
    {
        return (value_inline(val), Vec::new());
    }
    (pairs.join(", "), Vec::new())
}

pub fn summarize_tool_args(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return String::new();
    }
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
        if let Some(code) = map.get("code").and_then(Value::as_str) {
            let n = code.lines().count().max(1);
            let first = single_line_text(code.lines().next().unwrap_or(""));
            let summary = abbreviate_home_path(&format!("{n} lines: {first}"));
            return truncate_display(&summary, MAX_ARGS_SUMMARY_COLS);
        }
        if let (Some(pat), Some(path_val)) = (map.get("pattern"), map.get("path")) {
            let pat_s = value_inline(pat);
            let path_s = abbreviate_home_path(&value_inline(path_val));
            let combined = if path_s.is_empty() || path_s == "." {
                pat_s
            } else {
                format!("{pat_s} in {path_s}")
            };
            return truncate_display(&combined, MAX_ARGS_SUMMARY_COLS);
        }
        for priority_key in [
            "path", "file", "command", "query", "pattern", "uri", "server", "name",
        ] {
            if let Some(val) = map.get(priority_key) {
                let mut text = abbreviate_home_path(&value_inline(val));
                if priority_key == "path"
                    && map.get("edits").is_none()
                    && map.get("content").is_none()
                {
                    let off = map.get("offset").and_then(Value::as_u64);
                    let lim = map.get("limit").and_then(Value::as_u64);
                    text = match (off, lim) {
                        (Some(o), Some(l)) => {
                            format!("{text}:{o}-{}", o.saturating_add(l.saturating_sub(1)))
                        }
                        (Some(o), None) => format!("{text}:{o}+"),
                        (None, Some(l)) => format!("{text}:1-{l}"),
                        (None, None) => text,
                    };
                }
                return truncate_display(&text, MAX_ARGS_SUMMARY_COLS);
            }
        }
        let pairs: Vec<String> = map
            .iter()
            .take(3)
            .map(|(k, v)| format!("{k}={}", abbreviate_home_path(&value_inline(v))))
            .collect();
        return truncate_display(&pairs.join(", "), MAX_ARGS_SUMMARY_COLS);
    }
    let cleaned = abbreviate_home_path(&single_line_text(trimmed));
    truncate_display(&cleaned, MAX_ARGS_SUMMARY_COLS)
}

fn summarize_live_preview(preview: &str) -> String {
    let lines: Vec<&str> = preview
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let start = lines.len().saturating_sub(2);
    lines[start..]
        .iter()
        .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn extract_inline_meta(
    name: &str,
    content: &str,
    details: Option<&Value>,
    is_error: bool,
) -> Option<String> {
    if is_error {
        if name == "bash"
            && let Some(d) = details
        {
            if d.get("cancelled").and_then(Value::as_bool) == Some(true) {
                return Some("cancelled".to_string());
            }
            if d.get("timed_out").and_then(Value::as_bool) == Some(true) {
                return Some("timeout".to_string());
            }
            if let Some(code) = d.get("exit_code").and_then(Value::as_i64)
                && code != 0
            {
                return Some(format!("exit {code}"));
            }
        }
        return None;
    }

    match name {
        "read" => {
            if let Some(d) = details
                && let Some(total) = d.get("total_lines").and_then(Value::as_u64)
            {
                if total == 0 {
                    return Some("empty".to_string());
                }
                let start = d.get("start_line").and_then(Value::as_u64).unwrap_or(1);
                let end = d.get("end_line").and_then(Value::as_u64).unwrap_or(total);
                let truncated = d.get("truncated").and_then(Value::as_bool).unwrap_or(false);
                if truncated || start > 1 || end < total {
                    return Some(format!("L{start}-{end}/{total}"));
                }
                return Some(plural_unit(total as usize, "line", "lines"));
            }
            if content.trim().is_empty() || content.trim() == "(empty file)" {
                Some("empty".to_string())
            } else {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            }
        }
        "read_file"
        | "read_mcp_resource"
        | "list_mcp_resources"
        | "list_mcp_resource_templates" => {
            if content.trim().is_empty() {
                Some("empty".to_string())
            } else {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            }
        }
        "list_dir" | "ls" => {
            let trimmed = content.trim();
            if trimmed.is_empty() || trimmed == "（空目录）" {
                Some("empty".to_string())
            } else {
                let n = content.lines().filter(|l| !l.trim().is_empty()).count();
                Some(plural_unit(n, "item", "items"))
            }
        }
        "grep" => {
            if let Some(d) = details
                && let Some(m) = d.get("matches").and_then(Value::as_u64)
            {
                if m == 0 {
                    return Some("0 matches".to_string());
                }
                let limited = d
                    .get("limit_reached")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if limited {
                    return Some(format!("{m}+ matches"));
                }
                return Some(plural_unit(m as usize, "match", "matches"));
            }
            if content.trim_start().starts_with("No matches") {
                Some("0 matches".to_string())
            } else if !content.trim().is_empty() {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            } else {
                None
            }
        }
        "find" => {
            if let Some(d) = details
                && let Some(c) = d.get("count").and_then(Value::as_u64)
            {
                if c == 0 {
                    return Some("0 files".to_string());
                }
                let limited = d
                    .get("limit_reached")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if limited {
                    return Some(format!("{c}+ files"));
                }
                return Some(plural_unit(c as usize, "file", "files"));
            }
            if content.trim_start().starts_with("No files found") {
                Some("0 files".to_string())
            } else if !content.trim().is_empty() {
                let n = content.lines().filter(|l| !l.trim().is_empty()).count();
                Some(plural_unit(n, "file", "files"))
            } else {
                None
            }
        }
        "tool_search" => details
            .and_then(|d| d.get("loaded"))
            .and_then(Value::as_array)
            .map(|arr| format!("{} loaded", arr.len())),
        "skill" => Some("loaded".to_string()),
        "edit" => details
            .and_then(|d| d.get("diff"))
            .and_then(Value::as_str)
            .and_then(|diff| {
                let (adds, dels) = count_diff_stats(diff);
                format_diff_stat(adds, dels)
            }),
        "write" => details
            .and_then(|d| d.get("lines"))
            .and_then(Value::as_u64)
            .map(|n| plural_unit(n as usize, "line", "lines")),
        "bash" => {
            let lines = clean_bash_output_lines(content, false);
            if lines.len() > 2 {
                Some(plural_unit(lines.len(), "line", "lines"))
            } else {
                None
            }
        }
        "codemode" => details
            .and_then(|d| d.get("nested_calls"))
            .and_then(Value::as_array)
            .filter(|arr| !arr.is_empty())
            .map(|arr| plural_unit(arr.len(), "call", "calls")),
        "get_time" => {
            let t = content.trim();
            if !t.is_empty() && !t.contains('\n') {
                Some(t.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn summarize_tool_result(
    name: &str,
    content: &str,
    details: Option<&Value>,
    is_error: bool,
) -> String {
    if is_error {
        if name == "bash" {
            let lines = clean_bash_output_lines(content, true);
            let chosen = if lines.is_empty() {
                content
                    .lines()
                    .map(str::trim_end)
                    .filter(|l| !l.trim().is_empty())
                    .collect::<Vec<_>>()
            } else {
                lines
            };
            let start = chosen.len().saturating_sub(4);
            return chosen[start..]
                .iter()
                .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                .collect::<Vec<_>>()
                .join("\n");
        }
        return content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(4)
            .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
            .collect::<Vec<_>>()
            .join("\n");
    }

    match name {
        "skill" | "write" => String::new(),
        "edit" => {
            if let Some(diff) = details.and_then(|d| d.get("diff")).and_then(Value::as_str) {
                let lines: Vec<String> = diff
                    .lines()
                    .filter(|l| {
                        !l.starts_with("--- ")
                            && !l.starts_with("+++ ")
                            && (l.starts_with('+') || l.starts_with('-'))
                    })
                    .take(8)
                    .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                    .collect();
                if !lines.is_empty() {
                    return lines.join("\n");
                }
            }
            String::new()
        }
        "bash" => {
            let lines = clean_bash_output_lines(content, false);
            let start = lines.len().saturating_sub(2);
            lines[start..]
                .iter()
                .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                .collect::<Vec<_>>()
                .join("\n")
        }
        "codemode" => {
            let mut out = Vec::new();
            if let Some(calls) = details
                .and_then(|d| d.get("nested_calls"))
                .and_then(Value::as_array)
                && !calls.is_empty()
            {
                let mut parts = Vec::new();
                for c in calls.iter().take(3) {
                    let status = c.get("status").and_then(Value::as_str).unwrap_or("✓");
                    let cname = c.get("name").and_then(Value::as_str).unwrap_or("?");
                    parts.push(format!("{status} {cname}"));
                }
                let mut line = parts.join(", ");
                if calls.len() > 3 {
                    line.push_str(&format!(" … ({} calls)", calls.len()));
                }
                out.push(truncate_display(&line, MAX_PREVIEW_LINE_COLS));
            }
            for l in content.lines().filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with("Nested tool calls:")
                    && !t.starts_with("[Truncated to ")
            }) {
                if out.len() >= 2 {
                    break;
                }
                out.push(truncate_display(l, MAX_PREVIEW_LINE_COLS));
            }
            out.join("\n")
        }
        _ => content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(2)
            .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!(!app.is_busy);
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

    fn new_app_for_tps() -> AppState {
        AppState::new(
            PathBuf::from("."),
            "sess-tps",
            Vec::new(),
            Some("default".into()),
            Some("deepseek-chat".into()),
            None,
        )
    }

    /// 一个 turn 里两次模型调用：工具间隙不进分母，TPS 不随步数翻倍。
    #[test]
    fn multi_step_turn_does_not_inflate_tps() {
        let mut app = new_app_for_tps();
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });

        // 第一步：100 token / 1s
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 0 });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 0 });
        app.apply_ui_event(UiEvent::TextDelta {
            text: "x".repeat(400),
            ts_ms: 500,
        });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 1_000,
            outcome: UiStepOutcome::ToolCalls,
            usage: Some(TokenUsage {
                output_tokens: 100,
                ..TokenUsage::default()
            }),
        });
        assert_eq!(app.tps.rate(), Some(100.0));

        // 工具跑了 51 秒，然后第二步：300 token / 3s
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 52_000 });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 52_000 });
        app.apply_ui_event(UiEvent::TextDelta {
            text: "x".repeat(400),
            ts_ms: 52_500,
        });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 2,
            ts_ms: 55_000,
            outcome: UiStepOutcome::Completed,
            usage: Some(TokenUsage {
                output_tokens: 300,
                ..TokenUsage::default()
            }),
        });

        // (100 + 300) / (1s + 3s) = 100 tok/s。旧实现会累积分子、重置分母，得 400 tok/s。
        assert_eq!(app.tps.rate(), Some(100.0));

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });
        assert_eq!(app.tps_history, vec![100.0]);
    }

    /// 中文输出按脚本加权估算，不是 chars/4。
    #[test]
    fn cjk_output_is_not_underestimated() {
        let mut app = new_app_for_tps();
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 0 });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 0 });
        // 80 个汉字在 2 秒内输出完，约 57 token → 约 28.6 tok/s
        app.apply_ui_event(UiEvent::TextDelta {
            text: "中".repeat(80),
            ts_ms: 1_000,
        });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 2_000,
            outcome: UiStepOutcome::Completed,
            usage: None,
        });
        let rate = app.tps.rate().expect("CJK 输出必须有读数");
        // chars/4 只会给 20 token → 10 tok/s
        assert!(rate > 25.0, "CJK 被低估：{rate}");
    }

    /// 一批 delta 在同一时间戳排空时，读数仍按事件时间戳算，不出现零耗时尖峰。
    #[test]
    fn burst_of_same_timestamp_deltas_uses_event_time() {
        let mut app = new_app_for_tps();
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 0 });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 0 });
        app.apply_ui_event(UiEvent::TextDelta {
            text: "x".repeat(400),
            ts_ms: 600,
        });
        let single = app.tps.rate().expect("单次 delta 应有读数");

        // runner 的 try_recv 循环会把积压的 delta 一次性交给 UI。
        for _ in 0..50 {
            app.apply_ui_event(UiEvent::TextDelta {
                text: "x".repeat(400),
                ts_ms: 600,
            });
        }
        let burst = app.tps.rate().expect("批量 delta 应有读数");
        // 旧实现用消费时刻 elpased，50 个 delta 一批会冲出几十倍。这里只该按 token 量增大。
        assert!(
            burst / single < 60.0,
            "批量排空产生了异常尖峰：{single} → {burst}"
        );

        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 1_600,
            outcome: UiStepOutcome::Completed,
            usage: Some(TokenUsage {
                output_tokens: 800,
                ..TokenUsage::default()
            }),
        });
        // 800 token / 1.6s，真实用量定案
        assert_eq!(app.tps.rate(), Some(500.0));
    }

    /// 没有可测量样本的轮次保留上一个读数，不清空。
    #[test]
    fn turn_without_samples_keeps_the_previous_rate() {
        let mut app = new_app_for_tps();
        app.apply_ui_event(UiEvent::TurnStart { turn: 1 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 0 });
        app.apply_ui_event(UiEvent::FirstToken { ts_ms: 0 });
        app.apply_ui_event(UiEvent::TextDelta {
            text: "x".repeat(400),
            ts_ms: 600,
        });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 2_000,
            outcome: UiStepOutcome::Completed,
            usage: Some(TokenUsage {
                output_tokens: 100,
                ..TokenUsage::default()
            }),
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });
        assert_eq!(app.tps.rate(), Some(50.0));

        // 第二轮被取消：没有样本，读数保持。
        app.apply_ui_event(UiEvent::TurnStart { turn: 2 });
        app.apply_ui_event(UiEvent::StepStart { ts_ms: 10_000 });
        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 10_100,
            outcome: UiStepOutcome::Cancelled,
            usage: None,
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Cancelled,
        });
        assert_eq!(app.tps.rate(), Some(50.0));
        assert_eq!(app.tps_history, vec![50.0]);
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

        // 模拟 Anthropic：prompt 共 10000，命中缓存 9000，写入缓存 2000，输出 500。
        // 解析层已把命中量从 input 里扣掉，所以 input 只剩未命中的 1000。
        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            cache_read_tokens: Some(9_000),
            cache_write_tokens: Some(2_000),
            ..Default::default()
        }));

        // 累计四项相加，不能用 input + output（那是 1500，低估 87%）。
        assert_eq!(app.total_usage.total_tokens, Some(12_500));
        assert_eq!(app.total_usage.grand_total(), 12_500);
        // cache_write 必须被累加，否则 Anthropic 写缓存量整体丢失。
        assert_eq!(app.total_usage.cache_write_tokens, Some(2_000));

        let ctx = &app.side_panel_data.context;
        assert_eq!(ctx.total_tokens, 12_500);
        assert_eq!(ctx.cumulative_prompt_tokens(), 12_000);
        assert_eq!(ctx.cache_write_tokens, 2_000);
        // 命中率分母是 prompt 全量（12000），而不是净输入（1000）——否则会算出 900%。
        assert!(
            (ctx.cache_hit_rate() - 75.0).abs() < 1e-9,
            "{}",
            ctx.cache_hit_rate()
        );
        // 上下文占用用最近一次请求的 prompt 全量，而不是累计 12000。
        assert_eq!(app.last_prompt_tokens, Some(12_000));
        assert_eq!(ctx.last_prompt_tokens, Some(12_000));

        // 第二次请求：只有纯 input，无缓存。累计与最近一次各自独立推进。
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
    fn new_session_resets_usage_totals_and_last_prompt() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-reset-usage",
            Vec::new(),
            None,
            None,
            None,
        );
        app.apply_ui_event(UiEvent::Usage(TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: Some(85),
            ..Default::default()
        }));
        assert_eq!(app.total_usage.total_tokens, Some(100));
        assert_eq!(app.last_prompt_tokens, Some(95));

        app.reset_for_new_session("sess-next", None);
        assert_eq!(app.total_usage.total_tokens, None);
        assert_eq!(app.total_usage.cache_read_tokens, None);
        assert_eq!(app.last_prompt_tokens, None);
        assert_eq!(app.side_panel_data.context.total_tokens, 0);
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

        app.is_busy = true;
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );
        app.is_busy = false;

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

        // 绝对路径与未知斜杠前缀应直接作为普通消息发给模型，不被误判为未知命令
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
                },
                ModelInfo {
                    id: "model-all".into(),
                    name: "All".into(),
                    context_window: Some(128_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: Vec::new(),
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

        // 1. Tab 在当前模型声明了 reasoning_efforts 时按模型列表循环
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Low));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Medium));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::High));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Low));

        // 2. Ctrl+L 打开模型选择器，并用 Enter 切换到未声明支持列表的模型后，Tab 回退到全 7 档循环
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

        // 从 Low 切换：Low -> Medium -> High -> XHigh -> Max -> Off -> Minimal
        app.reasoning_effort = Some(ReasoningEffort::Max);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Off));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.reasoning_effort, Some(ReasoningEffort::Minimal));

        // 3. `/` 补全浮层打开时，Tab 仍用于补全而不切换思考等级
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

        // 4. 空闲时 Enter 与 Ctrl+Enter 均立即发送，Shift+Enter 插入换行
        app.input.insert_str("a");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT));
        app.input.insert_str("b");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            AppAction::SubmitPrompt("a\nb".into())
        );

        // 5. 运行中 Enter 加入队列，Ctrl+Enter 立即打断插队，Esc 暂停
        app.is_busy = true;
        app.input.insert_str("queued one");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(
            app.queued_prompts
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

        // 6. 单个 Ctrl+C 清空输入（运行中不误中断，补全浮层打开时一并关闭，空会话时不误退出）
        app.input.insert_str("draft while busy");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert!(app.input.is_empty());
        assert!(!app.should_quit);

        app.is_busy = false;
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
    fn ctrl_o_toggles_thinking_and_tool_details_in_sync_without_polluting_items() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-12345678",
            Vec::new(),
            None,
            None,
            None,
        );
        assert!(app.show_thinking);
        assert!(!app.verbose_expanded);

        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(app.verbose_expanded);
        assert!(app.show_thinking);
        assert!(app.items.is_empty());
        assert_eq!(app.active_flash(), Some("思考与工具详情已展开"));

        app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL));
        assert!(!app.verbose_expanded);
        assert!(!app.show_thinking);
        assert!(app.items.is_empty());
        assert_eq!(app.active_flash(), Some("思考与工具详情已收起"));
    }

    #[test]
    fn mention_completion_supports_directory_drilldown_line_suffix_esc_dismiss_and_replay_fold() {
        let ws = std::env::temp_dir().join(format!("mizar_app_mention_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        std::fs::create_dir_all(ws.join("src/nested")).unwrap();
        std::fs::write(ws.join("src/lib.rs"), "pub fn ok() {}\n").unwrap();
        std::fs::write(ws.join("src/nested/mod.rs"), "// nested\n").unwrap();

        let mut app = AppState::new(ws.clone(), "sess-mention", Vec::new(), None, None, None);

        // 1. 输入 `@sr` 并接受目录 `src/`：不追加尾随空格，并连续保持在目录下一跳补全
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

        // 2. Esc 仅局部关闭当前 `@src/` 弹窗，不清空输入也不触发停止
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::None
        );
        assert!(!app.overlay.is_open());
        assert_eq!(app.input.text(), "@src/");

        // 继续输入 `l` 改变 query 后应重新弹出，Tab 完成文件后带尾随空格
        app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE));
        assert!(matches!(
            app.overlay.active,
            Some(OverlayKind::FileMention { .. })
        ));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.input.text(), "@src/lib.rs ");
        assert!(!app.overlay.is_open());

        // 3. 在 `#L10-20` 前编辑 path 并接受候选时，应保留行号后缀
        app.input.clear();
        for ch in "@nested#L10-20".chars() {
            app.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert!(!app.overlay.is_open());
        // 左移 7 个字符回到 `nested` 与 `#` 之间
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

        // 4. --resume 回放折叠 `file-mentions` 通知，不将原文灌入聊天流
        let folded = summarize_context_notice_for_replay(
            FILE_MENTIONS_SOURCE,
            "<attached-file path=\"src/lib.rs\" lines=\"1-2\">\nSECRET_FILE_BODY\n</attached-file>\n<attached-directory path=\"src\">\na/\n</attached-directory>",
        );
        assert_eq!(folded, "[file-mentions] 已附加引用：src/lib.rs, src");
        assert!(!folded.contains("SECRET_FILE_BODY"));

        let _ = std::fs::remove_dir_all(&ws);
    }

    #[test]
    fn abbreviate_home_with_handles_paths_commands_and_prefix_boundaries() {
        let home = Some("/Users/alice");
        assert_eq!(abbreviate_home_with("/Users/alice", home), "~");
        assert_eq!(
            abbreviate_home_with("/Users/alice/workspace/mizar/src/lib.rs", home),
            "~/workspace/mizar/src/lib.rs"
        );
        assert_eq!(
            abbreviate_home_with(
                "cd /Users/alice/workspace && cat \"/Users/alice/.bashrc\"",
                home
            ),
            "cd ~/workspace && cat \"~/.bashrc\""
        );
        // 不能误替换同前缀用户名或非路径边界子串
        assert_eq!(
            abbreviate_home_with("/Users/alice_other/foo", home),
            "/Users/alice_other/foo"
        );
        assert_eq!(
            abbreviate_home_with("/tmp/Users/alice/foo", home),
            "/tmp/Users/alice/foo"
        );
    }

    #[test]
    fn summarize_tool_args_preserves_long_commands_ranges_and_grep_context() {
        let long_cmd =
            "cd /tmp/very-long-workspace-directory-name && cargo test --workspace --all-targets";
        let cmd_raw = format!(r#"{{"command":"{long_cmd}"}}"#);
        let cmd_summary = summarize_tool_args(&cmd_raw);
        assert_eq!(cmd_summary, long_cmd);
        assert!(cmd_summary.len() > 48);

        assert_eq!(
            summarize_tool_args(
                r#"{"path":"crates/mizar-tui-ui/src/ui.rs","offset":100,"limit":50}"#
            ),
            "crates/mizar-tui-ui/src/ui.rs:100-149"
        );
        assert_eq!(
            summarize_tool_args(r#"{"pattern":"build_tool_lines","path":"crates/mizar-tui-ui"}"#),
            "build_tool_lines in crates/mizar-tui-ui"
        );
        assert_eq!(summarize_tool_args(r#"{"pattern":"*","path":"."}"#), "*");
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

        // bash 无输出或报错场景
        assert_eq!(
            summarize_tool_result("bash", "(no output)", None, false),
            ""
        );
        let err_preview = summarize_tool_result(
            "bash",
            "error[E0425]: missing\nCommand exited with code 1",
            Some(&serde_json::json!({ "exit_code": 1 })),
            true,
        );
        assert!(err_preview.contains("error[E0425]: missing"));
        assert_eq!(
            extract_inline_meta(
                "bash",
                "Command exited with code 2",
                Some(&serde_json::json!({ "exit_code": 2 })),
                true
            )
            .as_deref(),
            Some("exit 2")
        );
        assert_eq!(
            extract_inline_meta(
                "edit",
                "ok",
                Some(&serde_json::json!({ "diff": "--- a\n+++ b\n-old\n+new1\n+new2\n" })),
                false
            )
            .as_deref(),
            Some("+2 -1")
        );
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

        // 1. `!` / `!!` 禁止触发 `/` 斜杠补全与内置斜杠拦截
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

        // 2. 空命令 `!` / `!!` 按 Enter 给出温和 flash 提示不执行
        for empty_cmd in ["!", "!!", "  !  "] {
            app.input.set_text_end(empty_cmd);
            let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            assert_eq!(act, AppAction::None);
            assert!(app.active_flash().is_some_and(|f| f.contains("Bash")));
            assert_eq!(app.input.text(), empty_cmd);
        }

        // 3. 非运行态按 Esc 退出 Bash 模式并清空输入
        app.input.set_text_end("!! pwd");
        assert_eq!(app.bash_mode(), Some(BashInputMode::NoContext));
        let esc_act = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert_eq!(esc_act, AppAction::None);
        assert!(app.input.is_empty());
        assert_eq!(app.bash_mode(), None);

        // 4. 并发安全：is_busy 与 is_bash_running 下拒绝运行并保留输入
        app.is_busy = true;
        app.input.set_text_end("! echo hi");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "! echo hi");
        assert!(app.active_flash().is_some());
        app.is_busy = false;

        app.start_local_bash("lb-1", "sleep 5", false);
        assert!(app.is_bash_running);
        app.input.set_text_end("!! echo two");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "!! echo two");
        assert!(app.active_flash().is_some_and(|f| f.contains("Esc")));

        // 清空输入后按 Esc 触发 CancelRun 中断运行中的 bash
        app.input.clear();
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );

        // 5. 上下文估算隔离：! 计入 tool_tokens，!! 不污染上下文估算
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
        assert!(!app.is_bash_running);
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

        // 1. `/compact 保留重构决策` -> CompactSession(Some(...))
        app.input.set_text_end("/compact 保留重构决策");
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(act, AppAction::CompactSession(Some("保留重构决策".into())));

        // 2. 压缩中状态更新 + Esc 中断
        app.apply_ui_event(UiEvent::CompactionStarted {
            trigger: "manual".into(),
        });
        assert!(app.is_compacting);
        app.apply_ui_event(UiEvent::CompactionProgress {
            stage: "summary".into(),
            delta_chars: 180,
        });
        assert!(
            app.compaction_status
                .as_deref()
                .is_some_and(|s| s.contains("180"))
        );
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::CancelRun
        );

        // 3. 压缩完成后清状态、展示摘要通知，且计费 total_usage 不回退
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
        assert!(!app.is_compacting);
        assert!(app.compaction_status.is_none());
        assert_eq!(app.total_usage.input_tokens, 1_200);
        assert_eq!(app.total_usage.output_tokens, 300);
        assert!(app.items.iter().any(|it| matches!(
            it,
            ChatItem::Notice { text, .. } if text.contains("保留自 #10") && text.contains("已完成核心重构")
        )));
    }

    #[test]
    fn settings_slash_command_and_modal_live_toggle_work() {
        use crate::overlay::{SettingEntry, SettingItemKind};

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-settings-ui",
            Vec::new(),
            None,
            None,
            None,
        );

        // 1. `/settings 压缩` 触发 OpenSettings(Some("压缩"))
        app.input.set_text_end("/settings 压缩");
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(act, AppAction::OpenSettings(Some("压缩".into())));

        // 2. 打开设置浮层后按 Enter 切换第一项并保持浮层展开
        app.overlay.open_settings(
            vec![
                SettingEntry {
                    key: "ui.showThinking".into(),
                    category: "界面".into(),
                    label: "显示思考过程".into(),
                    description: "开关推理块".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
                SettingEntry {
                    key: "compaction.autoPrune".into(),
                    category: "压缩".into(),
                    label: "L1 零模型剪枝".into(),
                    description: "自动剪枝".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
            ],
            None,
            "~/.mizar/settings.json".into(),
        );
        assert!(app.overlay.is_modal());
        assert!(app.show_thinking);

        let toggle_act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            toggle_act,
            AppAction::ApplySetting {
                key: "ui.showThinking".into(),
                value: "false".into(),
            }
        );
        assert!(!app.show_thinking);
        assert!(
            app.overlay.is_open(),
            "settings modal should stay open after toggle"
        );

        // 3. 输入关键词过滤后按一次 Esc 清空过滤，再按一次 Esc 关闭浮层
        let _ = app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE));
        let _ = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(app.overlay.is_open());
        let _ = app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!app.overlay.is_open());
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

        // 运行中（is_busy / is_bash_running / is_compacting）禁止切换，弹窗状态保持不变并提示中文警告
        for case in 0..3 {
            app.is_busy = case == 0;
            app.is_bash_running = case == 1;
            app.is_compacting = case == 2;
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

        // 空闲时允许切换并派发 ApplySetting
        app.is_busy = false;
        app.is_bash_running = false;
        app.is_compacting = false;
        let act = app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            act,
            AppAction::ApplySetting {
                key: "plugin.mcp".into(),
                value: "false".into(),
            }
        );
    }

    /// 构造一个运行中、已经排了若干条的 AppState。
    fn queued_app(prompts: &[&str]) -> AppState {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queue",
            Vec::new(),
            None,
            None,
            None,
        );
        app.is_busy = true;
        for text in prompts {
            app.input.insert_str(text);
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        }
        app
    }

    fn queued_texts(app: &AppState) -> Vec<&str> {
        app.queued_prompts
            .iter()
            .map(|item| item.text.as_str())
            .collect()
    }

    #[test]
    fn empty_draft_up_enters_queue_selector_and_enter_retracts_without_residue() {
        let mut app = queued_app(&["first", "second"]);
        assert!(app.input.is_empty());

        // 空草稿按 ↑ 进入选择器，高亮队尾，不写历史也不改输入框。
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)),
            AppAction::None
        );
        let last_id = app.queued_prompts.last().unwrap().id;
        assert_eq!(app.queue_selected, Some(last_id));
        assert!(app.input.is_empty());

        // ⏎ 撤回：该项出队并进入输入框，队列里不能留副本。
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "second");
        assert_eq!(queued_texts(&app), vec!["first"]);
        assert_eq!(app.queue_selected, None);
    }

    #[test]
    fn up_past_first_closes_selector_and_falls_back_to_history() {
        let mut app = queued_app(&["only"]);
        app.input.push_history("older prompt");
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert!(app.queue_selected.is_some());

        // 已到队首，再按 ↑ 交出选择器，并把这次按键还给历史回溯。
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(app.queue_selected, None);
        assert_eq!(app.input.text(), "older prompt");
        assert_eq!(queued_texts(&app), vec!["only"]);
    }

    #[test]
    fn retract_with_nonempty_draft_swaps_draft_into_queue_tail() {
        let mut app = queued_app(&["queued item"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        app.input.insert_str("draft");
        // 有草稿时不能直接撤回（↑ 已被选择器吃掉），先手动构造“选择器 + 草稿”状态。
        app.queue_selected = Some(app.queued_prompts[0].id);

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        // 交换：队列项进输入框，草稿进队尾，两处各一份。
        assert_eq!(app.input.text(), "queued item");
        assert_eq!(queued_texts(&app), vec!["draft"]);
    }

    #[test]
    fn ctrl_enter_on_empty_draft_steers_queue_head() {
        let mut app = queued_app(&["head item", "tail item"]);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL)),
            AppAction::SteerPrompt("head item".into())
        );
        assert_eq!(queued_texts(&app), vec!["tail item"]);
    }

    #[test]
    fn alt_enter_on_empty_draft_steers_queue_head_without_keyboard_protocol() {
        let mut app = queued_app(&["head item"]);
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            AppAction::SteerPrompt("head item".into())
        );
        assert!(app.queued_prompts.is_empty());
    }

    #[test]
    fn alt_enter_with_draft_still_inserts_newline() {
        let mut app = queued_app(&["queued item"]);
        app.input.insert_str("draft");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "draft\n");
        assert_eq!(queued_texts(&app), vec!["queued item"]);
    }

    #[test]
    fn alt_up_retracts_queue_tail_straight_into_the_composer() {
        // Alt+↑ 一步撤回队尾直接编辑，不需要再进选择器按 ⏎。
        let mut app = queued_app(&["first", "tail item"]);
        assert!(app.input.is_empty());
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "tail item");
        // 撤回即出队：队列里不能留副本，选择器也不该被打开。
        assert_eq!(queued_texts(&app), vec!["first"]);
        assert_eq!(app.queue_selected, None);
    }

    #[test]
    fn alt_up_swaps_when_a_draft_is_already_present() {
        let mut app = queued_app(&["queued item"]);
        app.input.insert_str("draft");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "queued item");
        assert_eq!(queued_texts(&app), vec!["draft"]);
    }

    #[test]
    fn alt_up_without_queue_falls_back_to_history_recall() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-noqueue",
            Vec::new(),
            None,
            None,
            None,
        );
        app.input.push_history("older prompt");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            AppAction::None
        );
        assert_eq!(app.input.text(), "older prompt");
    }

    #[test]
    fn plain_up_still_opens_the_selector_as_the_zero_protocol_path() {
        // ESC 前缀式 Alt+↑（ESC ESC [ A）在部分终端会被拆成 Esc + "[A"，
        // 那种环境下必须还能靠普通 ↑ 进选择器撤回。
        let mut app = queued_app(&["tail item"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert!(app.queue_selected.is_some());
        assert_eq!(queued_texts(&app), vec!["tail item"]);
    }

    #[test]
    fn idle_enter_on_empty_draft_submits_queue_head() {
        let mut app = queued_app(&["stuck item"]);
        app.is_busy = false;
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("stuck item".into())
        );
        assert!(app.queued_prompts.is_empty());
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::User { text }) if text == "stuck item"
        ));
    }

    #[test]
    fn esc_and_ctrl_c_inside_selector_close_it_without_cancelling_run() {
        let mut app = queued_app(&["item"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.queue_selected, None);
        assert_eq!(queued_texts(&app), vec!["item"]);

        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            AppAction::None
        );
        assert_eq!(app.queue_selected, None);
        assert_eq!(queued_texts(&app), vec!["item"]);
    }

    #[test]
    fn delete_inside_selector_drops_item_without_sending() {
        let mut app = queued_app(&["keep", "drop me"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(queued_texts(&app), vec!["keep"]);
        assert_eq!(app.input.text(), "");
        assert_eq!(app.queue_selected, None);
    }

    #[test]
    fn dequeue_clears_selection_pointing_at_the_removed_item() {
        let mut app = queued_app(&["a", "b"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        let selected = app.queue_selected.unwrap();
        assert!(app.dequeue_prompt(selected).is_some());
        assert_eq!(app.queue_selected, None);
        // 队首出队后选择器若指向别的条目，应保留。
        app.queue_selected = Some(app.queued_prompts[0].id);
        assert_eq!(app.dequeue_next().map(|item| item.text), Some("a".into()));
        assert!(app.queued_prompts.is_empty());
        assert_eq!(app.queue_selected, None);
    }

    #[test]
    fn queue_selector_does_not_hijack_typing() {
        let mut app = queued_app(&["item"]);
        app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE)),
            AppAction::None
        );
        // 普通字符先收起选择器，再正常落进输入框。
        assert_eq!(app.queue_selected, None);
        assert_eq!(app.input.text(), "h");
        assert_eq!(queued_texts(&app), vec!["item"]);
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

        // 第 21 张必须明确失败并保留已插入的 20 张及提示用户
        assert!(app.insert_pasted_image(None, sample.clone()).is_none());
        assert_eq!(app.input.image_count(), MAX_IMAGES_PER_MESSAGE);
        assert!(matches!(
            app.items.last(),
            Some(ChatItem::Notice { level: NoticeLevel::Warn, text }) if text.contains("20")
        ));

        // 删掉一张占位符后应允许再贴一张
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
                },
                ModelInfo {
                    id: "vision".into(),
                    name: "Vision".into(),
                    context_window: Some(128_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
                    input: vec!["text".into(), "image".into()],
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

        // 不支持图片的模型禁止发送，且不能清空用户草稿
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

        // 切到支持图片的模型后立刻允许提交，且图片负载完整送出
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
        app.is_busy = true;
        let img = ImageAttachment::new("image/jpeg", "QkJC").with_dimensions(800, 600);
        let tok = app
            .insert_pasted_image(Some("image/png → image/jpeg"), img.clone())
            .unwrap();
        app.input.insert_str(" 看下");

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::None
        );
        assert_eq!(app.queued_prompts.len(), 1);
        assert_eq!(
            app.queued_prompts[0].images,
            vec![(tok.clone(), img.clone())]
        );

        // Alt+↑ 撤回后应恢复原子占位符与图片附件
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            AppAction::None
        );
        assert!(app.queued_prompts.is_empty());
        assert_eq!(app.input.image_count(), 1);

        // 单次 Backspace 不应因内部空格破坏整个占位符
        app.input.move_left(); // cursor before "看下"
        app.is_busy = false;
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

        // 1. Completed -> ProgramStatus::Done，不追加警告 notice
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

        // 2. Truncated -> 绝不产生 Done，必须给出定位到 step 的 Warn notice
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

        // 3. MaxSteps -> 绝不产生 Done，给出步数上限 Warn notice
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

        // 4. Error -> TurnEnd 不重复写入 Notice，由 RunFinished(Err) 统一提示一次
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

        // 5. reset_for_new_session 清空终态字段，restore_session_view 支持 SessionReplayItem::Notice
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
        });
        assert!(app.items.iter().any(|item| matches!(
            item,
            ChatItem::Notice {
                level: NoticeLevel::Warn,
                text,
            } if text == "Reached max steps (max_steps=2)"
        )));
    }
}
