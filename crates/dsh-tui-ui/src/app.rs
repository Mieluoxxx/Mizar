//! TUI 核心状态机：会话投影、主题与 `pi-emphasis` 状态、浮层栈、多栏选区、虚拟化行缓存与事件分发。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use dsh_llm_types::{ContentBlock, ModelInfo, ProviderInfo, ReasoningEffort, TokenUsage, ToolCall};
use dsh_trajectory_core::{TimelineMode, TrajectoryReport};
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
use crate::side_panel::{
    ContextBreakdownSnapshot, FileTouchRow, SidePanelData, SidePanelState,
};
use crate::terminal::{ProgramStatusState, ScreenMode, emit_program_status, emit_terminal_progress};
use crate::theme::{TerminalColors, Theme, ThemePreset};
use crate::timeline_rail::{
    RailGeometry, RailHit, TimelineSnapshot, hit_test_rail,
};
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
    MaxSteps,
    Cancelled,
    Error { message: String },
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionViewData {
    pub session_id: String,
    pub session_title: Option<String>,
    pub items: Vec<SessionReplayItem>,
    pub finish_notice: String,
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
    FirstToken {
        ts_ms: u128,
    },
    ReasoningDelta(String),
    TextDelta(String),
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
    Log(UiLogRecord),
    RunFinished(Result<String, String>),
    ApprovalRequested {
        tool_name: String,
        args_preview: String,
        responder: Arc<std::sync::Mutex<Option<futures::channel::oneshot::Sender<ApprovalDecision>>>>,
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

/// UI 层处理按键/提交后需要 Runner 执行的外部动作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppAction {
    None,
    SubmitPrompt(String),
    SteerPrompt(String),
    CancelRun,
    NewSession,
    ShowStats,
    ShowTools,
    ShowMcp,
    ShowSkills,
    ShowInstructions,
    OpenSkillPicker,
    OpenSessionTree,
    ForkSessionAtSeq(u64),
    OpenExternalEditor,
    SwitchScreenMode(ScreenMode),
    SetMouseCapture(bool),
    ExportTrajectory(Option<PathBuf>),
    ShowOrSwitchModel(Option<String>),
    RefreshSidePanelData,
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
    pub queued_prompts: Vec<String>,
    pub is_busy: bool,
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
    pub last_step_start_ms: Option<u128>,
    pub last_ttft_ms: Option<u64>,
    pub ttft_history: Vec<u64>,
    pub decode_tokens_acc: u64,
    pub decode_started_at: Option<Instant>,
    pub last_tps: Option<f64>,
    pub tps_history: Vec<f64>,
    pub spinner_tick: usize,
    pub should_quit: bool,
    streaming_assistant_open: bool,

    // 双模视图与鼠标开关
    pub screen_mode: ScreenMode,
    pub inline_committed_items: usize,
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
            is_busy: false,
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
            last_step_start_ms: None,
            last_ttft_ms: None,
            ttft_history: Vec::new(),
            decode_tokens_acc: 0,
            decode_started_at: None,
            last_tps: None,
            tps_history: Vec::new(),
            spinner_tick: 0,
            should_quit: false,
            streaming_assistant_open: false,
            screen_mode: ScreenMode::Fullscreen,
            inline_committed_items: 0,
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
        }
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

    pub fn push_user_message(&mut self, text: &str) {
        self.items.push(ChatItem::User {
            text: text.to_string(),
        });
        self.streaming_assistant_open = false;
        self.scroll_to_bottom();
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
                    let body_len = self
                        .tool_extras
                        .get(id)
                        .map(|e| e.full_content.len())
                        .unwrap_or(output_preview.len());
                    tool_chars += args_summary.len() + body_len;
                }
                ChatItem::Notice { .. } => {}
            }
        }
        // ~3.5 chars/token 估算比例，总占用以实测 total_usage 为主
        let est = |chars: usize| -> u64 { (chars as u64).div_ceil(4) };
        let used_est = (est(user_chars) + est(asst_chars) + est(reason_chars) + est(tool_chars)).max(1);
        let measured = self.total_usage.input_tokens + self.total_usage.output_tokens;
        let scale = if measured > 0 {
            measured as f64 / used_est as f64
        } else {
            1.0
        };
        self.side_panel_data.context = ContextBreakdownSnapshot {
            system_tokens: (measured / 10).max(512),
            user_tokens: ((est(user_chars) as f64) * scale).round() as u64,
            assistant_tokens: ((est(asst_chars) as f64) * scale).round() as u64,
            reasoning_tokens: ((est(reason_chars) as f64) * scale).round() as u64,
            tool_tokens: ((est(tool_chars) as f64) * scale).round() as u64,
            context_window: self.active_context_window(),
            tps: self.last_tps,
            ttft_ms: self.last_ttft_ms,
            tps_history: self.tps_history.clone(),
            ttft_history: self.ttft_history.clone(),
            cache_read_tokens: self.total_usage.cache_read_tokens.unwrap_or(0),
            input_tokens: self.total_usage.input_tokens,
            output_tokens: self.total_usage.output_tokens,
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
                self.decode_tokens_acc = 0;
                self.decode_started_at = None;
                emit_terminal_progress(true);
                emit_program_status(ProgramStatusState::Working, None, Some("Generating"));
            }
            UiEvent::TurnEnd { reason } => {
                self.is_busy = false;
                self.streaming_assistant_open = false;
                self.reveal.snap_all();
                emit_terminal_progress(false);
                if matches!(reason, UiTurnEndReason::Cancelled) {
                    self.push_notice(NoticeLevel::Info, "已中断本轮生成");
                    emit_program_status(ProgramStatusState::Idle, None, Some("Cancelled"));
                } else {
                    emit_program_status(ProgramStatusState::Done, None, Some("Turn completed"));
                }
                if let Some(tps) = self.last_tps {
                    self.tps_history.push(tps);
                    if self.tps_history.len() > 32 {
                        self.tps_history.remove(0);
                    }
                }
                self.refresh_context_breakdown();
            }
            UiEvent::StepStart { ts_ms } => {
                self.last_step_start_ms = Some(ts_ms);
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
                self.decode_started_at = Some(Instant::now());
            }
            UiEvent::ReasoningDelta(delta) => {
                let busy = self.is_busy;
                let (idx, item) = self.ensure_open_assistant();
                if let ChatItem::Assistant { reasoning, .. } = item {
                    reasoning.push_str(&delta);
                    let len = reasoning.len();
                    let snapshot = reasoning.clone();
                    self.reveal
                        .observe(RevealSlot::AssistantReasoning(idx), &snapshot, busy);
                    self.decode_tokens_acc += (delta.chars().count() as u64).div_ceil(4).max(1);
                    if let Some(t0) = self.decode_started_at {
                        let secs = t0.elapsed().as_secs_f64();
                        if secs >= 0.15 {
                            self.last_tps = Some(self.decode_tokens_acc as f64 / secs);
                        }
                    }
                    let _ = len;
                }
            }
            UiEvent::TextDelta(delta) => {
                let busy = self.is_busy;
                let (idx, item) = self.ensure_open_assistant();
                if let ChatItem::Assistant { text, .. } = item {
                    text.push_str(&delta);
                    let snapshot = text.clone();
                    self.reveal
                        .observe(RevealSlot::AssistantText(idx), &snapshot, busy);
                    self.decode_tokens_acc += (delta.chars().count() as u64).div_ceil(4).max(1);
                    if let Some(t0) = self.decode_started_at {
                        let secs = t0.elapsed().as_secs_f64();
                        if secs >= 0.15 {
                            self.last_tps = Some(self.decode_tokens_acc as f64 / secs);
                        }
                    }
                }
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
                    if !self
                        .touched_files
                        .iter()
                        .any(|f| f.path == p && f.op == op)
                    {
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
                self.total_usage.total_tokens = Some(
                    self.total_usage
                        .input_tokens
                        .saturating_add(self.total_usage.output_tokens),
                );
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
                    emit_program_status(
                        ProgramStatusState::Error,
                        None,
                        Some("Request failed"),
                    );
                }
                self.refresh_context_breakdown();
            }
            UiEvent::ApprovalRequested {
                tool_name,
                args_preview,
                responder,
            } => {
                if let Some(tx) = responder.lock().unwrap().take() {
                    emit_program_status(
                        ProgramStatusState::Blocked,
                        Some("permission"),
                        Some(&format!("Approve {tool_name}")),
                    );
                    self.overlay.open_approval(tool_name, args_preview, tx);
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
        if let Some((start, query)) = self.input.active_slash_trigger() {
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
            OverlayAction::ResolveApproval { decision } => {
                if let Some(OverlayKind::Approval(mut ap)) = self.overlay.active.take() {
                    if decision == ApprovalDecision::AllowAlwaysForTool {
                        self.always_allowed_tools.insert(ap.tool_name.clone());
                    }
                    if let Some(tx) = ap.responder.take() {
                        let _ = tx.send(decision);
                    }
                    emit_program_status(
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
        if changed && matches!(effort, ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max) {
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

        // 2. Ctrl+C：若有活跃鼠标选区，优先复制选区并保持界面；模态弹窗则先关闭/拒绝；
        // 非空输入优先由单次 Ctrl+C 清空（含补全浮层与运行中草稿）；否则走中断或双击退出
        if matches!(key.code, KeyCode::Char('c')) && ctrl {
            if self.selection.is_active()
                && let Some(buf) = &self.last_buffer
                && let Some(copied) = extract_selected_text(buf, &self.selection, &self.selection_geom)
            {
                let _ = copy_to_clipboard(&copied);
                let chars = copied.chars().count();
                self.set_flash(format!("✓ 已复制 {chars} 字符"));
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
            if self.is_busy {
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

        // 4. 全局功能快捷键
        match key.code {
            KeyCode::Char('d') if ctrl => {
                if !self.is_busy && self.input.is_empty() {
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
                if !self.is_busy {
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
            KeyCode::Esc => {
                if self.selection.is_active() {
                    self.selection.clear();
                    return AppAction::None;
                }
                if self.is_busy {
                    return AppAction::CancelRun;
                }
                if !self.input.is_empty() {
                    self.input.clear();
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
            KeyCode::Up => {
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
        self.always_allowed_tools.clear();
        self.reveal.clear();
        self.selection.clear();
        self.total_usage = TokenUsage::default();
        self.last_step_start_ms = None;
        self.last_ttft_ms = None;
        self.ttft_history.clear();
        self.last_tps = None;
        self.tps_history.clear();
        self.inline_committed_items = 0;
        self.scroll_from_top = 0;
        self.auto_scroll = true;
        self.streaming_assistant_open = false;
        self.dismissed_mention = None;
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

    fn prepare_submit(&mut self, immediate: bool) -> AppAction {
        let raw = self.input.take();
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

        if self.is_busy {
            let prompt = trimmed.to_string();
            if immediate {
                self.set_flash("⚡ 正在中断当前回合并立即发送…");
                return AppAction::SteerPrompt(prompt);
            }
            self.queued_prompts.push(prompt);
            self.set_flash(format!(
                "已加入队列 (共 {} 条待发)",
                self.queued_prompts.len()
            ));
            return AppAction::None;
        }

        self.push_user_message(trimmed);
        AppAction::SubmitPrompt(trimmed.to_string())
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
                    self.push_notice(
                        NoticeLevel::Warn,
                        "请先按 Esc 中断当前任务再开启新会话",
                    );
                    AppAction::None
                } else {
                    AppAction::NewSession
                }
            }
            "thinking" => {
                self.show_thinking = !self.show_thinking;
                self.item_cache.clear();
                let state = if self.show_thinking { "开启" } else { "隐藏" };
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
            if (trimmed.starts_with("<attached-file ") || trimmed.starts_with("<attached-directory "))
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
            format!("[{source}] {}", truncate_display(&text.replace('\n', " "), 96))
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
        app.apply_ui_event(UiEvent::ReasoningDelta("先查时间".into()));
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
        app.apply_ui_event(UiEvent::TextDelta("现在是 09:30".into()));
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
        app.input.insert_str("/rust-review crates/dsh-skills");
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            AppAction::SubmitPrompt("/rust-review crates/dsh-skills".to_string())
        );

        // 绝对路径与未知斜杠前缀应直接作为普通消息发给模型，不被误判为未知命令
        let abs_path_prompt =
            "/Users/moguw/workspace/dsh-space/dsh-rs/docs/plugins.md 帮我更新一下文档";
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
                },
                ModelInfo {
                    id: "model-all".into(),
                    name: "All".into(),
                    context_window: Some(128_000),
                    max_tokens: None,
                    reasoning_efforts: Vec::new(),
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
        assert_eq!(app.queued_prompts, vec!["queued one".to_string()]);
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
        let ws = std::env::temp_dir().join(format!("dsh_app_mention_{}", std::process::id()));
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
                    c.path == "src/"
                        && c.kind == crate::mentions::MentionCandidateKind::Directory
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
            abbreviate_home_with("/Users/alice/workspace/dsh-rs/src/lib.rs", home),
            "~/workspace/dsh-rs/src/lib.rs"
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
                r#"{"path":"crates/dsh-tui-ui/src/ui.rs","offset":100,"limit":50}"#
            ),
            "crates/dsh-tui-ui/src/ui.rs:100-149"
        );
        assert_eq!(
            summarize_tool_args(r#"{"pattern":"build_tool_lines","path":"crates/dsh-tui-ui"}"#),
            "build_tool_lines in crates/dsh-tui-ui"
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
}
