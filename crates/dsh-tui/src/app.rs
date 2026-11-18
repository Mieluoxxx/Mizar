//! TUI 核心状态机：会话投影、主题与 `pi-emphasis` 状态、浮层栈、多栏选区、虚拟化行缓存与事件分发。

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cordis::{LogLevel, LogRecord};
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use dsh_agent_loop::{AgentStatus, TurnEndReason};
use dsh_llm::{ContentBlock, ProviderInfo, ReasoningEffort, TokenUsage, ToolCall};
use dsh_session::{Session, SessionEvent};
use dsh_trajectory::{TimelineMode, TrajectoryReport};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use serde_json::Value;

use crate::editor::{TextInput, char_width};
use crate::emphasis::EmphasisConfig;
use crate::markdown::RenderedLine;
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
    pub local_expanded: Option<bool>,
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

/// 由后端异步事件与终端外部状态投递给 UI 主循环的消息。
#[derive(Debug, Clone)]
pub enum UiEvent {
    Status(AgentStatus),
    TurnStart {
        turn: u64,
    },
    TurnEnd {
        reason: TurnEndReason,
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
    UserSkillsUpdated(Vec<String>),
    Log(LogRecord),
    RunFinished(Result<String, String>),
    ApprovalRequested {
        tool_name: String,
        args_preview: String,
        responder: Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<ApprovalDecision>>>>,
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
    OpenSkillPicker,
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
    pub session: Arc<Session>,
    pub items: Vec<ChatItem>,
    pub tool_extras: HashMap<String, ToolExtraMeta>,
    pub item_cache: HashMap<usize, CachedItemLines>,
    pub tool_hit_rects: Vec<(String, Rect)>,

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

    // 确认退出 / 终端状态记忆
    pub terminal_focused: bool,
    last_ctrl_c_at: Option<Instant>,
    pub pending_exit_status: Option<(ProgramStatusState, Option<String>)>,
}

impl AppState {
    pub fn new(
        workspace: PathBuf,
        session: Arc<Session>,
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
            session,
            items: Vec::new(),
            tool_extras: HashMap::new(),
            item_cache: HashMap::new(),
            tool_hit_rects: Vec::new(),
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

    pub fn active_context_window(&self) -> u64 {
        let p = self.provider.as_deref();
        let m = self.model.as_deref();
        for prov in &self.providers {
            if p.is_some_and(|pid| pid != prov.id) {
                continue;
            }
            for model in &prov.models {
                if (m.is_none() || m == Some(model.id.as_str()))
                    && let Some(w) = model.context_window
                {
                    return w;
                }
            }
        }
        128_000
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
                self.is_busy = status == AgentStatus::Busy;
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
                if matches!(reason, TurnEndReason::Cancelled) {
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
                        local_expanded: None,
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
                let diff_raw = if !is_error && name == "edit" {
                    details
                        .as_ref()
                        .and_then(|d| d.get("diff"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                } else {
                    None
                };
                self.tool_extras.insert(
                    call_id.clone(),
                    ToolExtraMeta {
                        full_content: content.clone(),
                        diff_raw,
                        local_expanded: None,
                    },
                );
                let preview = summarize_tool_result(&name, &content, details.as_ref(), is_error);
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
            UiEvent::UserSkillsUpdated(names) => {
                self.user_skill_names = names;
            }
            UiEvent::Log(record) => {
                let level = match record.level {
                    LogLevel::Error => NoticeLevel::Error,
                    LogLevel::Warn => NoticeLevel::Warn,
                    LogLevel::Debug | LogLevel::Info | LogLevel::Silent => return,
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
        if let Some((start, query)) = self.input.active_mention_trigger() {
            let matches = self.file_cache.match_query(&self.workspace, &query);
            let selected = match &self.overlay.active {
                Some(OverlayKind::FileMention { selected, .. }) => {
                    (*selected).min(matches.len().saturating_sub(1))
                }
                _ => 0,
            };
            self.overlay.active = Some(OverlayKind::FileMention {
                token_start: start,
                query,
                matches,
                selected,
            });
            return;
        }
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
                self.overlay.active = None;
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
            OverlayAction::CompleteMention { token_start, path } => {
                self.overlay.active = None;
                let end = self.input.cursor_byte();
                self.input
                    .replace_byte_range(token_start, end, &format!("@{path} "));
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

    pub fn set_effort(&mut self, effort: ReasoningEffort) {
        let changed = self.reasoning_effort != Some(effort);
        self.reasoning_effort = Some(effort);
        if changed && matches!(effort, ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max) {
            self.effort_ignition_at = Some(Instant::now());
        }
        self.push_notice(
            NoticeLevel::Info,
            format!("推理强度已切换为：{}", format!("{effort:?}").to_lowercase()),
        );
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

        // 2. Ctrl+C：若有活跃鼠标选区，优先复制选区并保持界面；否则走中断/清空/退出
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
            if self.overlay.is_open() {
                if let Some(OverlayKind::Approval(_)) = self.overlay.active {
                    return self.apply_overlay_action(OverlayAction::ResolveApproval {
                        decision: ApprovalDecision::Deny,
                    });
                }
                self.overlay.active = None;
                return AppAction::None;
            }
            if self.is_busy {
                return AppAction::CancelRun;
            }
            if !self.input.is_empty() {
                self.input.clear();
                return AppAction::None;
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
                self.verbose_expanded = !self.verbose_expanded;
                self.show_thinking = !self.show_thinking;
                self.item_cache.clear();
                let state = if self.verbose_expanded { "展开" } else { "收起" };
                self.push_notice(NoticeLevel::Info, format!("思考与工具详情已{state}"));
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
                return AppAction::None;
            }
            KeyCode::Char('l') if ctrl => {
                if self.is_busy {
                    self.push_notice(
                        NoticeLevel::Warn,
                        "当前正在生成，请先按 Esc 中断后再清屏",
                    );
                    return AppAction::None;
                }
                return AppAction::NewSession;
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
            KeyCode::Tab if self.is_busy && !self.input.is_empty() => {
                // 运行中按 Tab 排入 Follow-up 队列，等当前回合结束自动发送
                let queued = self.input.take();
                let trimmed = queued.trim().to_string();
                if !trimmed.is_empty() {
                    self.queued_prompts.push(trimmed);
                    self.set_flash(format!(
                        "已排入队列 (共 {} 条待发)",
                        self.queued_prompts.len()
                    ));
                }
                return AppAction::None;
            }
            KeyCode::Enter if shift || alt || ctrl => {
                if (ctrl || alt) && self.is_busy && !self.input.is_empty() {
                    let prompt = self.input.take();
                    return AppAction::SteerPrompt(prompt);
                }
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Char('j') if ctrl => {
                self.input.insert_char('\n');
                return AppAction::None;
            }
            KeyCode::Enter => return self.prepare_submit(),
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
                return AppAction::None;
            }
            KeyCode::Right if alt || ctrl => {
                self.input.move_word_right();
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
                return AppAction::None;
            }
            KeyCode::Down => {
                self.input.move_down();
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
                return AppAction::None;
            }
            KeyCode::End => {
                self.input.move_end();
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
                            LaunchpadQuickAction::ModelPicker => {
                                if !self.providers.is_empty() {
                                    self.overlay
                                        .open_model_picker(&self.providers, self.model.as_deref());
                                    AppAction::None
                                } else {
                                    AppAction::ShowOrSwitchModel(None)
                                }
                            }
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
                // 检查是否点中某个工具卡以局部展开/收起
                for (tool_id, rect) in &self.tool_hit_rects {
                    if col >= rect.x
                        && col < rect.x + rect.width
                        && row >= rect.y
                        && row < rect.y + rect.height
                    {
                        if let Some(extra) = self.tool_extras.get_mut(tool_id) {
                            let cur = extra.local_expanded.unwrap_or(self.verbose_expanded);
                            extra.local_expanded = Some(!cur);
                            self.item_cache.clear();
                        }
                        break;
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

    pub fn reset_for_new_session(&mut self, session: Arc<Session>) {
        self.session = session;
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
        self.refresh_context_breakdown();
        self.push_notice(
            NoticeLevel::Info,
            format!("已开启新会话 ({})", short_id(&self.session.id)),
        );
    }

    /// 从会话日志重放并重建界面对话流（供 `/tree` 分叉回溯使用）。
    pub fn restore_from_session(&mut self, session: Arc<Session>, source_seq: u64) {
        let notice = format!(
            "已从事件 #{source_seq} 分叉至新会话 ({})，可继续发送新指令",
            short_id(&session.id)
        );
        self.replay_session_entries(session, notice);
    }

    /// 从持久化历史会话重放并重建界面对话流（供 `--resume` / `--continue` 使用）。
    pub fn resume_session(&mut self, session: Arc<Session>) {
        let header = session.header();
        let label = header
            .title
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(|t| format!("「{t}」"))
            .unwrap_or_else(|| short_id(&session.id).to_string());
        let notice = format!("已恢复历史会话 {} ({label})", short_id(&session.id));
        self.replay_session_entries(session, notice);
    }

    fn replay_session_entries(&mut self, session: Arc<Session>, finish_notice: String) {
        self.reset_for_new_session(Arc::clone(&session));
        // 清理 reset 推送的默认欢迎通知，由真实历史替代
        self.items.clear();

        for entry in session.log() {
            match entry.event {
                SessionEvent::User { text } => {
                    self.items.push(ChatItem::User { text });
                }
                SessionEvent::ContextNotice { source, text } => {
                    self.items.push(ChatItem::Notice {
                        level: NoticeLevel::Info,
                        text: format!("[{source}] {text}"),
                    });
                }
                SessionEvent::Compaction {
                    summary,
                    first_kept_seq,
                    ..
                } => {
                    self.items.push(ChatItem::Notice {
                        level: NoticeLevel::Info,
                        text: format!("[上下文压缩（保留自 #{first_kept_seq}）]\n{summary}"),
                    });
                }
                SessionEvent::Assistant { blocks, .. } => {
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
                                local_expanded: None,
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
                SessionEvent::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                    ..
                } => {
                    if let Some(ChatItem::ToolCall {
                        status,
                        output_preview,
                        ..
                    }) = self.items.iter_mut().rev().find(|it| match it {
                        ChatItem::ToolCall { id, .. } => id == &tool_call_id,
                        _ => false,
                    }) {
                        *status = if is_error {
                            ToolState::Error
                        } else {
                            ToolState::Done
                        };
                        *output_preview = summarize_live_preview(&content);
                    }
                    if let Some(ext) = self.tool_extras.get_mut(&tool_call_id) {
                        ext.full_content = content;
                    }
                }
                SessionEvent::Trace { .. } => {}
            }
        }
        self.push_notice(NoticeLevel::Info, finish_notice);
        self.refresh_context_breakdown();
        self.scroll_to_bottom();
    }

    pub fn open_session_tree(&mut self) {
        let log = self.session.log();
        if log.is_empty() {
            self.push_notice(NoticeLevel::Info, "当前会话尚无事件记录");
            return;
        }
        let entries: Vec<SessionTreeEntry> = log
            .into_iter()
            .map(|entry| match entry.event {
                SessionEvent::User { text } => SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "User".into(),
                    summary: text.replace('\n', " "),
                    depth: 0,
                    can_fork: true,
                },
                SessionEvent::ContextNotice { source, text } => SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: format!("Notice:{source}"),
                    summary: text.replace('\n', " "),
                    depth: 1,
                    can_fork: false,
                },
                SessionEvent::Assistant { blocks, .. } => {
                    let text_preview: String = blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            ContentBlock::ToolCall { name, .. } => Some(name.as_str()),
                            ContentBlock::Reasoning { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" · ");
                    SessionTreeEntry {
                        seq: entry.seq,
                        kind_tag: "Asst".into(),
                        summary: text_preview.replace('\n', " "),
                        depth: 1,
                        can_fork: true,
                    }
                }
                SessionEvent::ToolResult {
                    name,
                    content,
                    is_error,
                    ..
                } => SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: if is_error {
                        format!("Err:{name}")
                    } else {
                        format!("Tool:{name}")
                    },
                    summary: content.replace('\n', " "),
                    depth: 1,
                    can_fork: true,
                },
                SessionEvent::Compaction {
                    summary,
                    first_kept_seq,
                    ..
                } => SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: format!("Compact>={first_kept_seq}"),
                    summary: summary.replace('\n', " "),
                    depth: 0,
                    can_fork: false,
                },
                SessionEvent::Trace { .. } => SessionTreeEntry {
                    seq: entry.seq,
                    kind_tag: "Trace".into(),
                    summary: String::new(),
                    depth: 1,
                    can_fork: false,
                },
            })
            .collect();
        self.overlay.open_session_tree(entries);
    }

    fn prepare_submit(&mut self) -> AppAction {
        let raw = self.input.take();
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return AppAction::None;
        }
        self.overlay.active = None;
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
            self.input.insert_str(trimmed);
            self.push_notice(
                NoticeLevel::Warn,
                "上一轮仍在运行：按 Tab 排入待办队列，或 Ctrl+Enter 打断并插队",
            );
            return AppAction::None;
        }

        self.items.push(ChatItem::User {
            text: trimmed.to_string(),
        });
        self.streaming_assistant_open = false;
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
            "tree" => {
                self.open_session_tree();
                AppAction::None
            }
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
                None if !self.providers.is_empty() => {
                    self.overlay
                        .open_model_picker(&self.providers, self.model.as_deref());
                    AppAction::None
                }
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

pub fn summarize_tool_args(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return String::new();
    }
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
        if let Some(code) = map.get("code").and_then(Value::as_str) {
            let n = code.lines().count().max(1);
            let first = code.lines().next().unwrap_or("").trim();
            return truncate_display(&format!("{n} lines: {first}"), 48);
        }
        for priority_key in [
            "path", "file", "command", "query", "pattern", "uri", "server", "name",
        ] {
            if let Some(val) = map.get(priority_key) {
                let text = match val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                return truncate_display(&text, 48);
            }
        }
        let pairs: Vec<String> = map
            .iter()
            .take(2)
            .map(|(k, v)| match v {
                Value::String(s) => format!("{k}={s}"),
                other => format!("{k}={other}"),
            })
            .collect();
        return truncate_display(&pairs.join(", "), 48);
    }
    truncate_display(trimmed, 48)
}

fn summarize_live_preview(preview: &str) -> String {
    let lines: Vec<&str> = preview.lines().collect();
    let start = lines.len().saturating_sub(5);
    lines[start..]
        .iter()
        .map(|l| truncate_display(l, 96))
        .collect::<Vec<_>>()
        .join("\n")
}

fn summarize_tool_result(
    name: &str,
    content: &str,
    details: Option<&Value>,
    is_error: bool,
) -> String {
    if !is_error
        && name == "skill"
        && let Some(skill_name) = details.and_then(|d| d.get("name")).and_then(Value::as_str)
    {
        return format!("已加载技能 `{skill_name}`");
    }

    if !is_error
        && name == "edit"
        && let Some(diff) = details.and_then(|d| d.get("diff")).and_then(Value::as_str)
    {
        let mut lines: Vec<String> = diff
            .lines()
            .filter(|l| !l.starts_with("--- ") && !l.starts_with("+++ "))
            .take(12)
            .map(|l| truncate_display(l, 96))
            .collect();
        let total = diff
            .lines()
            .filter(|l| !l.starts_with("--- ") && !l.starts_with("+++ "))
            .count();
        if total > 12 {
            lines.push(format!("…（共 {total} 行 diff）"));
        }
        if !lines.is_empty() {
            return lines.join("\n");
        }
    }

    if name == "bash" {
        let lines: Vec<&str> = content.lines().collect();
        let tail_start = lines.len().saturating_sub(5);
        let mut preview: Vec<String> = lines[tail_start..]
            .iter()
            .map(|l| truncate_display(l, 96))
            .collect();
        if tail_start > 0 {
            preview.insert(0, format!("…（前省略 {tail_start} 行）"));
        }
        if let Some(d) = details {
            let ms = d.get("duration_ms").and_then(Value::as_u64);
            let exit = d.get("exit_code").and_then(Value::as_i64);
            if let (Some(code), Some(dur)) = (exit, ms)
                && code != 0
            {
                preview.push(format!("exit={code} ({dur}ms)"));
            }
        }
        return preview.join("\n");
    }

    if name == "codemode"
        && let Some(calls) = details
            .and_then(|d| d.get("nested_calls"))
            .and_then(Value::as_array)
    {
        let mut out = Vec::new();
        for c in calls.iter().take(4) {
            let status = c.get("status").and_then(Value::as_str).unwrap_or("✓");
            let cname = c.get("name").and_then(Value::as_str).unwrap_or("?");
            let dur = c.get("duration_ms").and_then(Value::as_u64).unwrap_or(0);
            out.push(format!("{status} {cname} ({dur}ms)"));
        }
        if calls.len() > 4 {
            out.push(format!("…（共 {} 次嵌套调用）", calls.len()));
        }
        out.extend(content.lines().take(3).map(|l| truncate_display(l, 96)));
        return out.join("\n");
    }

    let total_lines = content.lines().count();
    let mut preview_lines: Vec<String> = content
        .lines()
        .take(4)
        .map(|line| truncate_display(line, 96))
        .collect();
    if total_lines > 4 {
        preview_lines.push(format!("…（共 {total_lines} 行）"));
    }
    preview_lines.join("\n")
}

#[cfg(test)]
mod tests {
    use cordis::Context;

    use super::*;

    #[test]
    fn streams_reasoning_tools_and_text_into_items() {
        let ctx = Context::new();
        let session = Arc::new(Session::new(ctx, "sess-12345678"));
        let mut app = AppState::new(
            PathBuf::from("."),
            session,
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
            reason: TurnEndReason::Completed,
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
        let ctx = Context::new();
        let session = Arc::new(Session::new(ctx, "sess-12345678"));
        let mut app = AppState::new(
            PathBuf::from("."),
            session,
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
}
