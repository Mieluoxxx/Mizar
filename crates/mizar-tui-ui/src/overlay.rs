//! 声明式浮层栈（`OverlayState`）：
//!
//! 统一承接 `/` 命令补全、`@` 文件模糊引用、交互式 `/model`、`/effort`、
//! `/theme` 选择器、`Ctrl+R` 历史模糊搜索、`?` 帮助面板，以及可选的 `tools/pre-execute` 审批弹窗。

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use futures::channel::oneshot;
pub use mizar_client_core::{
    ApprovalDecision, MentionCandidate, MentionCandidateKind, ModelPickerItem, PendingApproval,
    SessionPickerItem, SessionTreeEntry, SettingEntry, SettingItemKind, SkillPickerItem,
    WorkspaceFileCache,
};
use mizar_llm_types::{ProviderInfo, ReasoningEffort};
use ratatui::layout::Rect;

use crate::theme::ThemePreset;

mod render;

pub use render::{format_session_picker_row, render_overlay};

#[derive(Debug, Clone)]
pub struct SlashEntry {
    pub command: String,
    pub description: String,
    pub takes_arg: bool,
}

pub fn builtin_slash_entries(skill_names: &[String]) -> Vec<SlashEntry> {
    let mut list = vec![
        SlashEntry {
            command: "help".into(),
            description: "打开快捷键与命令帮助面板".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "resume".into(),
            description: "恢复历史会话（打开会话选择器或指定 ID）".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "rename".into(),
            description: "重命名当前会话标题（留空由模型重新生成）".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "settings".into(),
            description: "打开交互式配置面板（实时生效并自动保存）".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "model".into(),
            description: "交互切换模型 / Provider (Ctrl+L)".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "effort".into(),
            description: "调整思考强度 (Tab 切换 / off..max)".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "compact".into(),
            description: "手动压缩当前会话上下文（可附聚焦说明）".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "session".into(),
            description: "查看当前会话 ID、存储、事件构成与压缩边界".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "theme".into(),
            description: "切换 Oklab / 自适应主题".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "trace".into(),
            description: "打开全屏三泳道轨迹分析 (Ctrl+T)".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "panel".into(),
            description: "展开/折叠右侧分栏面板 (Ctrl+B)".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "approval".into(),
            description: "切换危险工具交互审批模式 (auto / ask)".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "thinking".into(),
            description: "显示/隐藏推理过程 (Ctrl+O)".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "tools".into(),
            description: "查看已声明工具".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "mcp".into(),
            description: "查看 MCP 服务连接状态".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "skills".into(),
            description: "打开交互式技能选择器 / 列表".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "instructions".into(),
            description: "查看已加载的 AGENTS.md / CLAUDE.md 指令".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "tree".into(),
            description: "查看会话时间线与分叉回溯树".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "view".into(),
            description: "切换双模视图 (fullscreen / inline)".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "mouse".into(),
            description: "切换鼠标追踪 / 放行原生划词 (F2)".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "stats".into(),
            description: "查看会话统计摘要".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "export".into(),
            description: "导出会话轨迹 JSONL".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "clear".into(),
            description: "开启全新会话 (/new)".into(),
            takes_arg: false,
        },
        SlashEntry {
            command: "exit".into(),
            description: "退出终端界面".into(),
            takes_arg: false,
        },
    ];
    for sk in skill_names {
        list.push(SlashEntry {
            command: sk.clone(),
            description: "用户技能".into(),
            takes_arg: true,
        });
    }
    list
}

pub enum OverlayKind {
    SlashComplete {
        token_start: usize,
        query: String,
        selected: usize,
    },
    FileMention {
        token_start: usize,
        token_end: usize,
        path_end: usize,
        is_quoted: bool,
        query: String,
        matches: Vec<MentionCandidate>,
        selected: usize,
    },
    ModelPicker {
        items: Vec<ModelPickerItem>,
        query: String,
        selected: usize,
    },
    EffortPicker {
        selected: usize,
    },
    ThemePicker {
        selected: usize,
        original: ThemePreset,
    },
    HistorySearch {
        query: String,
        matches: Vec<String>,
        selected: usize,
    },
    SkillPicker {
        items: Vec<SkillPickerItem>,
        query: String,
        selected: usize,
    },
    SessionPicker {
        items: Vec<SessionPickerItem>,
        query: String,
        selected: usize,
    },
    SessionTree {
        entries: Vec<SessionTreeEntry>,
        selected: usize,
        pan_x: usize,
    },
    HelpModal {
        scroll: usize,
    },
    SettingsModal {
        items: Vec<SettingEntry>,
        query: String,
        selected: usize,
        save_path_hint: String,
    },
    Approval(PendingApproval),
}

pub enum OverlayAction {
    None,
    Close,
    CompleteSlash {
        token_start: usize,
        command: String,
        execute_now: bool,
    },
    CompleteMention {
        token_start: usize,
        token_end: usize,
        path_end: usize,
        is_quoted: bool,
        candidate: MentionCandidate,
    },
    SelectModel {
        provider: String,
        model: String,
    },
    SelectEffort(ReasoningEffort),
    PreviewTheme(ThemePreset),
    CommitTheme(ThemePreset),
    RevertTheme(ThemePreset),
    ApplyHistory(String),
    InsertSkillSlash(String),
    SelectSession(String),
    ForkSessionAtSeq(u64),
    OpenSettingModelPicker {
        key: String,
    },
    UpdateSetting {
        key: String,
        value: String,
    },
    ResolveApproval {
        decision: ApprovalDecision,
    },
}

#[derive(Default)]
pub struct OverlayState {
    pub active: Option<OverlayKind>,
    /// 排队等待处理的工具审批请求，避免后到的审批静默覆盖当前浮层并丢弃 oneshot 响应器。
    pub approval_queue: std::collections::VecDeque<PendingApproval>,
    pub last_box: Option<Rect>,
    pub item_rects: Vec<(usize, Rect)>,
}

impl OverlayState {
    pub fn is_modal(&self) -> bool {
        matches!(
            self.active,
            Some(
                OverlayKind::ModelPicker { .. }
                    | OverlayKind::EffortPicker { .. }
                    | OverlayKind::ThemePicker { .. }
                    | OverlayKind::HistorySearch { .. }
                    | OverlayKind::SkillPicker { .. }
                    | OverlayKind::SessionPicker { .. }
                    | OverlayKind::SessionTree { .. }
                    | OverlayKind::HelpModal { .. }
                    | OverlayKind::SettingsModal { .. }
                    | OverlayKind::Approval(_)
            )
        )
    }

    pub fn is_open(&self) -> bool {
        self.active.is_some()
    }

    pub fn has_pending_approvals(&self) -> bool {
        matches!(self.active, Some(OverlayKind::Approval(_))) || !self.approval_queue.is_empty()
    }

    /// 若当前无激活浮层且审批队列非空，将队首审批提升为激活弹窗并返回被请求的工具名。
    pub fn promote_next_approval(&mut self) -> Option<String> {
        if self.active.is_some() {
            return None;
        }
        let next = self.approval_queue.pop_front()?;
        let tool = next.tool_name.clone();
        self.active = Some(OverlayKind::Approval(next));
        Some(tool)
    }

    /// 以指定裁决排空所有尚未响应的审批弹窗与队列（用于取消运行/切换会话/退出清理）。
    pub fn resolve_all_approvals(&mut self, decision: ApprovalDecision) {
        if let Some(OverlayKind::Approval(mut ap)) = self.active.take()
            && let Some(tx) = ap.responder.take()
        {
            let _ = tx.send(decision.clone());
        }
        while let Some(mut ap) = self.approval_queue.pop_front() {
            if let Some(tx) = ap.responder.take() {
                let _ = tx.send(decision.clone());
            }
        }
    }

    pub fn open_model_picker(&mut self, providers: &[ProviderInfo], current_model: Option<&str>) {
        self.open_model_picker_with_follow(providers, current_model, false);
    }

    pub fn open_setting_model_picker(
        &mut self,
        providers: &[ProviderInfo],
        current_model: Option<&str>,
    ) {
        self.open_model_picker_with_follow(providers, current_model, true);
    }

    fn open_model_picker_with_follow(
        &mut self,
        providers: &[ProviderInfo],
        current_model: Option<&str>,
        allow_follow_session: bool,
    ) {
        let mut items = Vec::new();
        if allow_follow_session {
            items.push(ModelPickerItem {
                provider_id: String::new(),
                model_id: String::new(),
                label: "跟随会话模型".to_string(),
                detail: "随会话当前模型自动切换 (default)".to_string(),
            });
        }
        for p in providers {
            for m in &p.models {
                let ctx = m
                    .context_window
                    .map(|c| format!("ctx {}k", c / 1000))
                    .unwrap_or_default();
                items.push(ModelPickerItem {
                    provider_id: p.id.clone(),
                    model_id: m.id.clone(),
                    label: format!("{}/{}", p.id, m.id),
                    detail: format!("{} {}", p.protocol, ctx).trim().to_string(),
                });
            }
        }
        let selected = current_model
            .and_then(|cur| {
                items.iter().position(|it| {
                    !it.model_id.is_empty() && (it.model_id == cur || it.label == cur)
                })
            })
            .unwrap_or(0);
        self.active = Some(OverlayKind::ModelPicker {
            items,
            query: String::new(),
            selected,
        });
    }

    pub fn open_effort_picker(&mut self, current: Option<ReasoningEffort>) {
        let all = all_efforts();
        let selected = current
            .and_then(|c| all.iter().position(|e| *e == c))
            .unwrap_or(3);
        self.active = Some(OverlayKind::EffortPicker { selected });
    }

    pub fn open_theme_picker(&mut self, current: ThemePreset) {
        let selected = ThemePreset::all()
            .iter()
            .position(|p| *p == current)
            .unwrap_or(0);
        self.active = Some(OverlayKind::ThemePicker {
            selected,
            original: current,
        });
    }

    pub fn open_history_search(&mut self, history: &[String]) {
        let matches: Vec<String> = history.iter().rev().cloned().collect();
        self.active = Some(OverlayKind::HistorySearch {
            query: String::new(),
            matches,
            selected: 0,
        });
    }

    pub fn open_skill_picker(&mut self, items: Vec<SkillPickerItem>) {
        self.active = Some(OverlayKind::SkillPicker {
            items,
            query: String::new(),
            selected: 0,
        });
    }

    pub fn open_session_picker(&mut self, items: Vec<SessionPickerItem>) {
        self.active = Some(OverlayKind::SessionPicker {
            items,
            query: String::new(),
            selected: 0,
        });
    }

    pub fn open_session_tree(&mut self, entries: Vec<SessionTreeEntry>) {
        let selected = entries.len().saturating_sub(1);
        self.active = Some(OverlayKind::SessionTree {
            entries,
            selected,
            pan_x: 0,
        });
    }

    pub fn open_help(&mut self) {
        self.active = Some(OverlayKind::HelpModal { scroll: 0 });
    }

    pub fn open_settings(
        &mut self,
        items: Vec<SettingEntry>,
        initial_query: Option<String>,
        save_path_hint: String,
    ) {
        self.active = Some(OverlayKind::SettingsModal {
            items,
            query: initial_query.unwrap_or_default(),
            selected: 0,
            save_path_hint,
        });
    }

    pub fn refresh_settings(&mut self, new_items: Vec<SettingEntry>) {
        if let Some(OverlayKind::SettingsModal {
            items,
            query,
            selected,
            ..
        }) = &mut self.active
        {
            let prev_key = filter_setting_indices(items, query)
                .get(*selected)
                .and_then(|&idx| items.get(idx))
                .map(|entry| entry.key.clone());
            *items = new_items;
            let filtered = filter_setting_indices(items, query);
            if filtered.is_empty() {
                *selected = 0;
            } else if let Some(prev) = prev_key
                && let Some(pos) = filtered.iter().position(|&idx| items[idx].key == prev)
            {
                *selected = pos;
            } else {
                *selected = (*selected).min(filtered.len() - 1);
            }
        }
    }

    pub fn set_setting_toggle(&mut self, key: &str, new_value: bool) {
        if let Some(OverlayKind::SettingsModal { items, .. }) = &mut self.active
            && let Some(item) = items.iter_mut().find(|entry| entry.key == key)
            && let SettingItemKind::Toggle { value } = &mut item.kind
        {
            *value = new_value;
        }
    }

    pub fn set_setting_model(
        &mut self,
        key: &str,
        new_provider: Option<String>,
        new_model: Option<String>,
    ) {
        if let Some(OverlayKind::SettingsModal { items, .. }) = &mut self.active
            && let Some(item) = items.iter_mut().find(|entry| entry.key == key)
            && let SettingItemKind::Model { provider, model } = &mut item.kind
        {
            *provider = new_provider;
            *model = new_model;
        }
    }

    /// 请求工具审批；若当前已展示审批或其它模态浮层，先压入 `approval_queue` 避免丢失并发响应通道。
    /// 返回 `true` 表示该审批立即成为当前活动弹窗；返回 `false` 表示已排队等待当前浮层关闭。
    pub fn open_approval(
        &mut self,
        tool_name: String,
        args_preview: String,
        responder: oneshot::Sender<ApprovalDecision>,
    ) -> bool {
        let req = PendingApproval {
            tool_name,
            args_preview,
            editing_reason: false,
            deny_reason: String::new(),
            responder: Some(responder),
        };
        if self.is_modal() {
            self.approval_queue.push_back(req);
            false
        } else {
            self.active = Some(OverlayKind::Approval(req));
            true
        }
    }

    pub fn handle_key(
        &mut self,
        key: KeyEvent,
        skill_names: &[String],
        history: &[String],
    ) -> Option<OverlayAction> {
        let active = self.active.as_mut()?;
        match active {
            OverlayKind::SlashComplete {
                token_start,
                query,
                selected,
            } => {
                let entries = filter_slash(skill_names, query);
                match key.code {
                    KeyCode::Esc => Some(OverlayAction::Close),
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down => {
                        if !entries.is_empty() {
                            *selected = (*selected + 1).min(entries.len() - 1);
                        }
                        Some(OverlayAction::None)
                    }
                    KeyCode::Tab => entries
                        .get(*selected)
                        .map(|e| OverlayAction::CompleteSlash {
                            token_start: *token_start,
                            command: e.command.clone(),
                            execute_now: false,
                        }),
                    KeyCode::Enter
                        if key.modifiers == KeyModifiers::NONE && !entries.is_empty() =>
                    {
                        let chosen = &entries[(*selected).min(entries.len() - 1)];
                        Some(OverlayAction::CompleteSlash {
                            token_start: *token_start,
                            command: chosen.command.clone(),
                            execute_now: !chosen.takes_arg,
                        })
                    }
                    _ => None, // 交给编辑器继续输入过滤
                }
            }
            OverlayKind::FileMention {
                token_start,
                token_end,
                path_end,
                is_quoted,
                matches,
                selected,
                ..
            } => match key.code {
                KeyCode::Esc => Some(OverlayAction::Close),
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    Some(OverlayAction::None)
                }
                KeyCode::Down => {
                    if !matches.is_empty() {
                        *selected = (*selected + 1).min(matches.len() - 1);
                    }
                    Some(OverlayAction::None)
                }
                KeyCode::Tab | KeyCode::Enter
                    if key.modifiers == KeyModifiers::NONE && !matches.is_empty() =>
                {
                    Some(OverlayAction::CompleteMention {
                        token_start: *token_start,
                        token_end: *token_end,
                        path_end: *path_end,
                        is_quoted: *is_quoted,
                        candidate: matches[*selected].clone(),
                    })
                }
                _ => None,
            },
            OverlayKind::ModelPicker {
                items,
                query,
                selected,
            } => {
                let filtered: Vec<&ModelPickerItem> = items
                    .iter()
                    .filter(|it| {
                        query.is_empty()
                            || it
                                .label
                                .to_ascii_lowercase()
                                .contains(&query.to_ascii_lowercase())
                    })
                    .collect();
                match key.code {
                    KeyCode::Esc => Some(OverlayAction::Close),
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down => {
                        if !filtered.is_empty() {
                            *selected = (*selected + 1).min(filtered.len() - 1);
                        }
                        Some(OverlayAction::None)
                    }
                    KeyCode::Enter => {
                        filtered
                            .get(*selected)
                            .map(|it| OverlayAction::SelectModel {
                                provider: it.provider_id.clone(),
                                model: it.model_id.clone(),
                            })
                    }
                    KeyCode::Backspace => {
                        query.pop();
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        query.push(c);
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::EffortPicker { selected } => {
                let all = all_efforts();
                match key.code {
                    KeyCode::Esc => Some(OverlayAction::Close),
                    KeyCode::Up | KeyCode::Left => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down | KeyCode::Right => {
                        *selected = (*selected + 1).min(all.len() - 1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Enter => Some(OverlayAction::SelectEffort(all[*selected])),
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::ThemePicker { selected, original } => {
                let all = ThemePreset::all();
                match key.code {
                    KeyCode::Esc => Some(OverlayAction::RevertTheme(*original)),
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::PreviewTheme(all[*selected]))
                    }
                    KeyCode::Down => {
                        *selected = (*selected + 1).min(all.len() - 1);
                        Some(OverlayAction::PreviewTheme(all[*selected]))
                    }
                    KeyCode::Enter => Some(OverlayAction::CommitTheme(all[*selected])),
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::HistorySearch {
                query,
                matches,
                selected,
            } => match key.code {
                KeyCode::Esc => Some(OverlayAction::Close),
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    Some(OverlayAction::None)
                }
                KeyCode::Down => {
                    if !matches.is_empty() {
                        *selected = (*selected + 1).min(matches.len() - 1);
                    }
                    Some(OverlayAction::None)
                }
                KeyCode::Enter => matches
                    .get(*selected)
                    .cloned()
                    .map(OverlayAction::ApplyHistory)
                    .or(Some(OverlayAction::Close)),
                KeyCode::Backspace => {
                    query.pop();
                    *matches = filter_history(history, query);
                    *selected = 0;
                    Some(OverlayAction::None)
                }
                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    if !matches.is_empty() {
                        *selected = (*selected + 1) % matches.len();
                    }
                    Some(OverlayAction::None)
                }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    query.push(c);
                    *matches = filter_history(history, query);
                    *selected = 0;
                    Some(OverlayAction::None)
                }
                _ => Some(OverlayAction::None),
            },
            OverlayKind::SkillPicker {
                items,
                query,
                selected,
            } => {
                let filtered: Vec<&SkillPickerItem> = items
                    .iter()
                    .filter(|s| {
                        query.is_empty()
                            || s.name
                                .to_ascii_lowercase()
                                .contains(&query.to_ascii_lowercase())
                            || s.description
                                .to_ascii_lowercase()
                                .contains(&query.to_ascii_lowercase())
                    })
                    .collect();
                match key.code {
                    KeyCode::Esc => Some(OverlayAction::Close),
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down => {
                        if !filtered.is_empty() {
                            *selected = (*selected + 1).min(filtered.len() - 1);
                        }
                        Some(OverlayAction::None)
                    }
                    KeyCode::Enter => filtered
                        .get(*selected)
                        .map(|it| OverlayAction::InsertSkillSlash(it.name.clone())),
                    KeyCode::Backspace => {
                        query.pop();
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        query.push(c);
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::SessionPicker {
                items,
                query,
                selected,
            } => {
                let filtered = filter_session_picker_indices(items, query);
                match key.code {
                    KeyCode::Esc => {
                        if !query.is_empty() {
                            query.clear();
                            *selected = 0;
                            Some(OverlayAction::None)
                        } else {
                            Some(OverlayAction::Close)
                        }
                    }
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down => {
                        if !filtered.is_empty() {
                            *selected = (*selected + 1).min(filtered.len() - 1);
                        }
                        Some(OverlayAction::None)
                    }
                    KeyCode::Enter => filtered
                        .get(*selected)
                        .and_then(|&idx| items.get(idx))
                        .map(|it| OverlayAction::SelectSession(it.session_id.clone())),
                    KeyCode::Backspace => {
                        query.pop();
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        query.push(c);
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::SessionTree {
                entries,
                selected,
                pan_x,
            } => match key.code {
                KeyCode::Esc | KeyCode::Char('q') => Some(OverlayAction::Close),
                KeyCode::Up | KeyCode::Char('k') => {
                    *selected = selected.saturating_sub(1);
                    Some(OverlayAction::None)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if !entries.is_empty() {
                        *selected = (*selected + 1).min(entries.len() - 1);
                    }
                    Some(OverlayAction::None)
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    *pan_x = pan_x.saturating_sub(8);
                    Some(OverlayAction::None)
                }
                KeyCode::Right | KeyCode::Char('l') => {
                    *pan_x = pan_x.saturating_add(8);
                    Some(OverlayAction::None)
                }
                KeyCode::Enter | KeyCode::Char('f') => {
                    if let Some(node) = entries.get(*selected)
                        && node.can_fork
                    {
                        Some(OverlayAction::ForkSessionAtSeq(node.seq))
                    } else {
                        Some(OverlayAction::None)
                    }
                }
                _ => Some(OverlayAction::None),
            },
            OverlayKind::HelpModal { scroll } => match key.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?') => {
                    Some(OverlayAction::Close)
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    *scroll = scroll.saturating_sub(1);
                    Some(OverlayAction::None)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    *scroll += 1;
                    Some(OverlayAction::None)
                }
                _ => Some(OverlayAction::None),
            },
            OverlayKind::SettingsModal {
                items,
                query,
                selected,
                ..
            } => {
                let filtered_indices = filter_setting_indices(items, query);
                match key.code {
                    KeyCode::Esc => {
                        if !query.is_empty() {
                            query.clear();
                            *selected = 0;
                            Some(OverlayAction::None)
                        } else {
                            Some(OverlayAction::Close)
                        }
                    }
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        Some(OverlayAction::None)
                    }
                    KeyCode::Down => {
                        if !filtered_indices.is_empty() {
                            *selected = (*selected + 1).min(filtered_indices.len() - 1);
                        }
                        Some(OverlayAction::None)
                    }
                    KeyCode::Left => {
                        if let Some(&item_idx) = filtered_indices.get(*selected) {
                            let item = &mut items[item_idx];
                            if let SettingItemKind::Model { provider, model } = &mut item.kind {
                                *provider = None;
                                *model = None;
                                Some(OverlayAction::UpdateSetting {
                                    key: item.key.clone(),
                                    value: String::new(),
                                })
                            } else {
                                let value = item.step_value(-1);
                                Some(OverlayAction::UpdateSetting {
                                    key: item.key.clone(),
                                    value,
                                })
                            }
                        } else {
                            Some(OverlayAction::None)
                        }
                    }
                    KeyCode::Right | KeyCode::Enter => {
                        if let Some(&item_idx) = filtered_indices.get(*selected) {
                            let item = &mut items[item_idx];
                            if matches!(item.kind, SettingItemKind::Model { .. }) {
                                Some(OverlayAction::OpenSettingModelPicker {
                                    key: item.key.clone(),
                                })
                            } else {
                                let value = item.step_value(1);
                                Some(OverlayAction::UpdateSetting {
                                    key: item.key.clone(),
                                    value,
                                })
                            }
                        } else {
                            Some(OverlayAction::None)
                        }
                    }
                    KeyCode::Char(' ') if query.is_empty() => {
                        if let Some(&item_idx) = filtered_indices.get(*selected) {
                            let item = &mut items[item_idx];
                            if matches!(item.kind, SettingItemKind::Model { .. }) {
                                Some(OverlayAction::OpenSettingModelPicker {
                                    key: item.key.clone(),
                                })
                            } else {
                                let value = item.step_value(1);
                                Some(OverlayAction::UpdateSetting {
                                    key: item.key.clone(),
                                    value,
                                })
                            }
                        } else {
                            Some(OverlayAction::None)
                        }
                    }
                    KeyCode::Backspace => {
                        query.pop();
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        query.push(c);
                        *selected = 0;
                        Some(OverlayAction::None)
                    }
                    _ => Some(OverlayAction::None),
                }
            }
            OverlayKind::Approval(ap) => {
                if ap.editing_reason {
                    match key.code {
                        KeyCode::Esc => {
                            ap.editing_reason = false;
                            Some(OverlayAction::None)
                        }
                        KeyCode::Enter => {
                            let reason = ap.deny_reason.trim().to_string();
                            let decision = if reason.is_empty() {
                                ApprovalDecision::Deny
                            } else {
                                ApprovalDecision::DenyWithReason(reason)
                            };
                            Some(OverlayAction::ResolveApproval { decision })
                        }
                        KeyCode::Backspace => {
                            ap.deny_reason.pop();
                            Some(OverlayAction::None)
                        }
                        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                            ap.deny_reason.push(c);
                            Some(OverlayAction::None)
                        }
                        _ => Some(OverlayAction::None),
                    }
                } else {
                    match key.code {
                        KeyCode::Char('y') | KeyCode::Enter => {
                            Some(OverlayAction::ResolveApproval {
                                decision: ApprovalDecision::AllowOnce,
                            })
                        }
                        KeyCode::Char('a') => Some(OverlayAction::ResolveApproval {
                            decision: ApprovalDecision::AllowAlwaysForTool,
                        }),
                        KeyCode::Char('r') => {
                            ap.editing_reason = true;
                            Some(OverlayAction::None)
                        }
                        KeyCode::Char('n') | KeyCode::Esc => Some(OverlayAction::ResolveApproval {
                            decision: ApprovalDecision::Deny,
                        }),
                        _ => Some(OverlayAction::None),
                    }
                }
            }
        }
    }

    pub fn handle_mouse(
        &mut self,
        mouse: MouseEvent,
        skill_names: &[String],
    ) -> Option<OverlayAction> {
        let active = self.active.as_mut()?;
        let col = mouse.column;
        let row = mouse.row;
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                for (idx, rect) in &self.item_rects {
                    if col >= rect.x
                        && col < rect.x + rect.width
                        && row >= rect.y
                        && row < rect.y + rect.height
                    {
                        let idx = *idx;
                        return match active {
                            OverlayKind::SlashComplete {
                                token_start, query, ..
                            } => {
                                let list = filter_slash(skill_names, query);
                                list.get(idx).map(|e| OverlayAction::CompleteSlash {
                                    token_start: *token_start,
                                    command: e.command.clone(),
                                    execute_now: !e.takes_arg,
                                })
                            }
                            OverlayKind::FileMention {
                                token_start,
                                token_end,
                                path_end,
                                is_quoted,
                                matches,
                                ..
                            } => matches.get(idx).map(|c| OverlayAction::CompleteMention {
                                token_start: *token_start,
                                token_end: *token_end,
                                path_end: *path_end,
                                is_quoted: *is_quoted,
                                candidate: c.clone(),
                            }),
                            OverlayKind::ModelPicker { items, query, .. } => {
                                let filtered: Vec<&ModelPickerItem> = items
                                    .iter()
                                    .filter(|it| {
                                        query.is_empty()
                                            || it
                                                .label
                                                .to_ascii_lowercase()
                                                .contains(&query.to_ascii_lowercase())
                                    })
                                    .collect();
                                filtered.get(idx).map(|it| OverlayAction::SelectModel {
                                    provider: it.provider_id.clone(),
                                    model: it.model_id.clone(),
                                })
                            }
                            OverlayKind::EffortPicker { .. } => all_efforts()
                                .get(idx)
                                .copied()
                                .map(OverlayAction::SelectEffort),
                            OverlayKind::ThemePicker { .. } => ThemePreset::all()
                                .get(idx)
                                .copied()
                                .map(OverlayAction::CommitTheme),
                            OverlayKind::HistorySearch { matches, .. } => {
                                matches.get(idx).cloned().map(OverlayAction::ApplyHistory)
                            }
                            OverlayKind::SkillPicker { items, query, .. } => {
                                let filtered: Vec<&SkillPickerItem> = items
                                    .iter()
                                    .filter(|s| {
                                        query.is_empty()
                                            || s.name
                                                .to_ascii_lowercase()
                                                .contains(&query.to_ascii_lowercase())
                                            || s.description
                                                .to_ascii_lowercase()
                                                .contains(&query.to_ascii_lowercase())
                                    })
                                    .collect();
                                filtered
                                    .get(idx)
                                    .map(|it| OverlayAction::InsertSkillSlash(it.name.clone()))
                            }
                            OverlayKind::SessionPicker { items, query, .. } => {
                                let filtered = filter_session_picker_indices(items, query);
                                filtered
                                    .get(idx)
                                    .and_then(|&item_idx| items.get(item_idx))
                                    .map(|it| OverlayAction::SelectSession(it.session_id.clone()))
                            }
                            OverlayKind::SessionTree { entries, .. } => {
                                if let Some(node) = entries.get(idx)
                                    && node.can_fork
                                {
                                    Some(OverlayAction::ForkSessionAtSeq(node.seq))
                                } else {
                                    None
                                }
                            }
                            OverlayKind::Approval(ap) => match idx {
                                0 => Some(OverlayAction::ResolveApproval {
                                    decision: ApprovalDecision::AllowOnce,
                                }),
                                1 => Some(OverlayAction::ResolveApproval {
                                    decision: ApprovalDecision::AllowAlwaysForTool,
                                }),
                                2 => {
                                    ap.editing_reason = true;
                                    Some(OverlayAction::None)
                                }
                                _ => Some(OverlayAction::ResolveApproval {
                                    decision: ApprovalDecision::Deny,
                                }),
                            },
                            OverlayKind::SettingsModal {
                                items,
                                query,
                                selected,
                                ..
                            } => {
                                let filtered_indices = filter_setting_indices(items, query);
                                if let Some(&item_idx) = filtered_indices.get(idx) {
                                    *selected = idx;
                                    let item = &mut items[item_idx];
                                    if matches!(item.kind, SettingItemKind::Model { .. }) {
                                        Some(OverlayAction::OpenSettingModelPicker {
                                            key: item.key.clone(),
                                        })
                                    } else {
                                        let value = item.step_value(1);
                                        Some(OverlayAction::UpdateSetting {
                                            key: item.key.clone(),
                                            value,
                                        })
                                    }
                                } else {
                                    None
                                }
                            }
                            OverlayKind::HelpModal { .. } => None,
                        };
                    }
                }
                if let Some(box_r) = self.last_box
                    && !(col >= box_r.x
                        && col < box_r.x + box_r.width
                        && row >= box_r.y
                        && row < box_r.y + box_r.height)
                {
                    if let OverlayKind::ThemePicker { original, .. } = active {
                        return Some(OverlayAction::RevertTheme(*original));
                    }
                    if !matches!(active, OverlayKind::Approval(_)) {
                        return Some(OverlayAction::Close);
                    }
                }
                Some(OverlayAction::None)
            }
            _ => None,
        }
    }
}

pub fn all_efforts() -> [ReasoningEffort; 7] {
    [
        ReasoningEffort::Off,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ]
}

pub(super) fn filter_slash(skill_names: &[String], query: &str) -> Vec<SlashEntry> {
    let q = query.trim().to_ascii_lowercase();
    builtin_slash_entries(skill_names)
        .into_iter()
        .filter(|e| q.is_empty() || e.command.to_ascii_lowercase().contains(&q))
        .take(10)
        .collect()
}

fn filter_history(history: &[String], query: &str) -> Vec<String> {
    let q = query.trim().to_ascii_lowercase();
    history
        .iter()
        .rev()
        .filter(|h| q.is_empty() || h.to_ascii_lowercase().contains(&q))
        .take(10)
        .cloned()
        .collect()
}

pub(super) fn filter_setting_indices(items: &[SettingEntry], query: &str) -> Vec<usize> {
    let q = query.trim().to_ascii_lowercase();
    items
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            q.is_empty()
                || item.key.to_ascii_lowercase().contains(&q)
                || item.category.to_ascii_lowercase().contains(&q)
                || item.label.to_ascii_lowercase().contains(&q)
                || item.description.to_ascii_lowercase().contains(&q)
        })
        .map(|(i, _)| i)
        .collect()
}

pub fn filter_session_picker_indices(items: &[SessionPickerItem], query: &str) -> Vec<usize> {
    let q = query.trim().to_ascii_lowercase();
    items
        .iter()
        .enumerate()
        .filter(|(_, item)| {
            q.is_empty()
                || item.title.to_ascii_lowercase().contains(&q)
                || item.cwd_tail.to_ascii_lowercase().contains(&q)
                || item.session_id.to_ascii_lowercase().contains(&q)
                || item.updated_at.to_ascii_lowercase().contains(&q)
        })
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::display_width;

    #[test]
    fn session_picker_aligns_columns_with_fullwidth_chars_and_selects_filtered_item() {
        let items = vec![
            SessionPickerItem {
                session_id: "s-1".into(),
                title: "0402｜FEA｜会话标题生成与 picker".into(),
                cwd_tail: "repo-a".into(),
                event_count: 14,
                updated_at: "04-02 10:00".into(),
            },
            SessionPickerItem {
                session_id: "s-2".into(),
                title: "ASCII title".into(),
                cwd_tail: "my-proj".into(),
                event_count: 3,
                updated_at: "04-01 09:30".into(),
            },
        ];

        let (left1, right1) = format_session_picker_row(&items[0], 34, 10);
        let (left2, right2) = format_session_picker_row(&items[1], 34, 10);
        assert_eq!(display_width(&left1), 34);
        assert_eq!(display_width(&left2), 34);
        assert_eq!(display_width(&right1), display_width(&right2));

        let mut state = OverlayState::default();
        state.open_session_picker(items);
        let _ = state.handle_key(KeyEvent::from(KeyCode::Char('a')), &[], &[]);
        let _ = state.handle_key(KeyEvent::from(KeyCode::Char('s')), &[], &[]);
        let _ = state.handle_key(KeyEvent::from(KeyCode::Char('c')), &[], &[]);
        let act = state.handle_key(KeyEvent::from(KeyCode::Enter), &[], &[]);
        assert!(matches!(act, Some(OverlayAction::SelectSession(ref id)) if id == "s-2"));
    }
}
