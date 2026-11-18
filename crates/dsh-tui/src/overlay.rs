//! 声明式浮层栈（`OverlayStack`）：
//!
//! 统一承接 `/` 命令补全、`@` 文件模糊引用（接 `ignore` crate）、交互式 `/model`、`/effort`、
//! `/theme` 选择器、`Ctrl+R` 历史模糊搜索、`?` 帮助面板，以及可选的 `tools/pre-execute` 审批弹窗。

use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use dsh_llm::{ProviderInfo, ReasoningEffort};
use ignore::WalkBuilder;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use tokio::sync::oneshot;

use crate::editor::display_width;
use crate::theme::{Theme, ThemePreset};

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
            command: "model".into(),
            description: "交互切换模型 / Provider".into(),
            takes_arg: true,
        },
        SlashEntry {
            command: "effort".into(),
            description: "调整思考强度 (off..max)".into(),
            takes_arg: true,
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
            description: "开启全新会话 (Ctrl+L)".into(),
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

/// 缓存 2 秒的工作区文件索引（利用 `ignore` crate 自动遵守 `.gitignore`）。
#[derive(Debug, Default)]
pub struct WorkspaceFileCache {
    files: Vec<String>,
    refreshed_at: Option<Instant>,
}

impl WorkspaceFileCache {
    pub fn files(&mut self, workspace: &Path) -> &[String] {
        let stale = self
            .refreshed_at
            .is_none_or(|t| t.elapsed() > Duration::from_secs(2));
        if stale {
            let mut found = Vec::new();
            let walker = WalkBuilder::new(workspace)
                .hidden(false)
                .max_depth(Some(8))
                .build();
            for entry in walker.flatten() {
                if found.len() >= 256 {
                    break;
                }
                if entry.file_type().is_some_and(|ft| ft.is_file())
                    && let Ok(rel) = entry.path().strip_prefix(workspace)
                {
                    found.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
            found.sort();
            self.files = found;
            self.refreshed_at = Some(Instant::now());
        }
        &self.files
    }

    pub fn match_query(&mut self, workspace: &Path, query: &str) -> Vec<String> {
        let q = query.trim().to_ascii_lowercase();
        let all = self.files(workspace);
        if q.is_empty() {
            return all.iter().take(12).cloned().collect();
        }
        let mut scored: Vec<(i32, String)> = all
            .iter()
            .filter_map(|path| fuzzy_score(path, &q).map(|s| (s, path.clone())))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.len().cmp(&b.1.len())));
        scored.into_iter().take(12).map(|(_, p)| p).collect()
    }
}

fn fuzzy_score(candidate: &str, query: &str) -> Option<i32> {
    let lower = candidate.to_ascii_lowercase();
    if lower == query {
        return Some(1000);
    }
    if lower.starts_with(query) {
        return Some(800 - candidate.len() as i32);
    }
    if let Some(pos) = lower.find(query) {
        return Some(500 - pos as i32 - candidate.len() as i32);
    }
    // 简单子序列模糊匹配（`@mep` -> `markdown_emphasis.rs`）
    let mut q_chars = query.chars().peekable();
    let mut hits = 0i32;
    for ch in lower.chars() {
        if q_chars.peek() == Some(&ch) {
            q_chars.next();
            hits += 10;
        }
    }
    if q_chars.peek().is_none() {
        Some(hits - candidate.len() as i32)
    } else {
        None
    }
}

#[derive(Debug)]
pub struct PendingApproval {
    pub tool_name: String,
    pub args_preview: String,
    pub editing_reason: bool,
    pub deny_reason: String,
    pub responder: Option<oneshot::Sender<ApprovalDecision>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    AllowOnce,
    AllowAlwaysForTool,
    Deny,
    DenyWithReason(String),
}

#[derive(Debug, Clone)]
pub struct SkillPickerItem {
    pub name: String,
    pub description: String,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct SessionTreeEntry {
    pub seq: u64,
    pub kind_tag: String,
    pub summary: String,
    pub depth: usize,
    pub can_fork: bool,
}

#[derive(Debug, Clone)]
pub struct ModelPickerItem {
    pub provider_id: String,
    pub model_id: String,
    pub label: String,
    pub detail: String,
}

pub enum OverlayKind {
    SlashComplete {
        token_start: usize,
        query: String,
        selected: usize,
    },
    FileMention {
        token_start: usize,
        query: String,
        matches: Vec<String>,
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
    SessionTree {
        entries: Vec<SessionTreeEntry>,
        selected: usize,
        pan_x: usize,
    },
    HelpModal {
        scroll: usize,
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
        path: String,
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
    ForkSessionAtSeq(u64),
    ResolveApproval {
        decision: ApprovalDecision,
    },
}

#[derive(Default)]
pub struct OverlayState {
    pub active: Option<OverlayKind>,
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
                    | OverlayKind::SessionTree { .. }
                    | OverlayKind::HelpModal { .. }
                    | OverlayKind::Approval(_)
            )
        )
    }

    pub fn is_open(&self) -> bool {
        self.active.is_some()
    }

    pub fn open_model_picker(&mut self, providers: &[ProviderInfo], current_model: Option<&str>) {
        let mut items = Vec::new();
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
            .and_then(|cur| items.iter().position(|it| it.model_id == cur || it.label == cur))
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

    pub fn open_approval(&mut self, tool_name: String, args_preview: String, responder: oneshot::Sender<ApprovalDecision>) {
        self.active = Some(OverlayKind::Approval(PendingApproval {
            tool_name,
            args_preview,
            editing_reason: false,
            deny_reason: String::new(),
            responder: Some(responder),
        }));
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
                    KeyCode::Tab => entries.get(*selected).map(|e| OverlayAction::CompleteSlash {
                        token_start: *token_start,
                        command: e.command.clone(),
                        execute_now: false,
                    }),
                    KeyCode::Enter
                        if key.modifiers == KeyModifiers::NONE && !entries.is_empty() =>
                    {
                        let chosen = &entries[*selected];
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
                        path: matches[*selected].clone(),
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
                            || it.label.to_ascii_lowercase().contains(&query.to_ascii_lowercase())
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
                    KeyCode::Enter => filtered.get(*selected).map(|it| OverlayAction::SelectModel {
                        provider: it.provider_id.clone(),
                        model: it.model_id.clone(),
                    }),
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
                        KeyCode::Char('y') | KeyCode::Enter => Some(OverlayAction::ResolveApproval {
                            decision: ApprovalDecision::AllowOnce,
                        }),
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
                                token_start,
                                query,
                                ..
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
                                matches,
                                ..
                            } => matches.get(idx).map(|p| OverlayAction::CompleteMention {
                                token_start: *token_start,
                                path: p.clone(),
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
                            OverlayKind::EffortPicker { .. } => {
                                all_efforts().get(idx).copied().map(OverlayAction::SelectEffort)
                            }
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
                            OverlayKind::SessionTree { entries, .. } => {
                                if let Some(node) = entries.get(idx)
                                    && node.can_fork
                                {
                                    Some(OverlayAction::ForkSessionAtSeq(node.seq))
                                } else {
                                    None
                                }
                            }
                            OverlayKind::Approval(_) => {
                                let decision = match idx {
                                    0 => ApprovalDecision::AllowOnce,
                                    1 => ApprovalDecision::AllowAlwaysForTool,
                                    _ => ApprovalDecision::Deny,
                                };
                                Some(OverlayAction::ResolveApproval { decision })
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

fn filter_slash(skill_names: &[String], query: &str) -> Vec<SlashEntry> {
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

pub fn render_overlay(
    frame: &mut Frame<'_>,
    full_area: Rect,
    input_area: Rect,
    state: &mut OverlayState,
    skill_names: &[String],
    theme: &Theme,
) {
    state.last_box = None;
    state.item_rects.clear();
    let Some(active) = &state.active else {
        return;
    };

    match active {
        OverlayKind::SlashComplete { query, selected, .. } => {
            let entries = filter_slash(skill_names, query);
            let rows: Vec<(String, String)> = entries
                .iter()
                .map(|e| (format!("/{}", e.command), e.description.clone()))
                .collect();
            render_anchored_list(
                frame,
                full_area,
                input_area,
                " 命令补全 (Tab 填入 · Enter 执行) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::FileMention {
            query,
            matches,
            selected,
            ..
        } => {
            let rows: Vec<(String, String)> = matches
                .iter()
                .map(|p| (format!("@{p}"), String::new()))
                .collect();
            let title = format!(" 文件引用: @{query} (Tab/Enter 插入) ");
            render_anchored_list(
                frame,
                full_area,
                input_area,
                &title,
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::ModelPicker {
            items,
            query,
            selected,
        } => {
            let filtered: Vec<(String, String)> = items
                .iter()
                .filter(|it| {
                    query.is_empty()
                        || it
                            .label
                            .to_ascii_lowercase()
                            .contains(&query.to_ascii_lowercase())
                })
                .map(|it| (it.label.clone(), it.detail.clone()))
                .collect();
            let title = format!(" 选择模型 [搜索: {query}] ");
            render_centered_list(frame, full_area, &title, &filtered, *selected, theme, state);
        }
        OverlayKind::EffortPicker { selected } => {
            let rows: Vec<(String, String)> = all_efforts()
                .iter()
                .map(|e| {
                    let desc = match e {
                        ReasoningEffort::Off => "关闭显式思考",
                        ReasoningEffort::Minimal => "极少推理 token",
                        ReasoningEffort::Low => "轻量快速推理",
                        ReasoningEffort::Medium => "均衡模式",
                        ReasoningEffort::High => "深度推理",
                        ReasoningEffort::XHigh => "超长推理预算",
                        ReasoningEffort::Max => "最大推理深度",
                    };
                    (format!("{e:?}").to_lowercase(), desc.to_string())
                })
                .collect();
            render_centered_list(
                frame,
                full_area,
                " 选择思考强度 (Reasoning Effort) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::ThemePicker { selected, .. } => {
            let rows: Vec<(String, String)> = ThemePreset::all()
                .iter()
                .map(|p| {
                    let desc = match p {
                        ThemePreset::Auto => "按 OSC 11 自动切换深浅雾蓝",
                        ThemePreset::MistDark => "雾蓝深色 (默认)",
                        ThemePreset::MistLight => "雾蓝浅色",
                        ThemePreset::SakuraDark => "樱粉深色",
                        ThemePreset::System => "OKHSL 终端色板自动拟合",
                        ThemePreset::Ansi16 => "16 色高兼容回退",
                    };
                    (p.name().to_string(), desc.to_string())
                })
                .collect();
            render_centered_list(
                frame,
                full_area,
                " 切换主题 (上下即时预览 · Enter 确认) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::HistorySearch {
            query,
            matches,
            selected,
        } => {
            let rows: Vec<(String, String)> = matches
                .iter()
                .map(|m| (m.replace('\n', " ↵ "), String::new()))
                .collect();
            let title = format!(" 历史模糊搜索 (Ctrl+R): {query} ");
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::SkillPicker {
            items,
            query,
            selected,
        } => {
            let rows: Vec<(String, String)> = items
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
                .map(|s| {
                    (
                        format!("/{} [{}]", s.name, s.source),
                        s.description.clone(),
                    )
                })
                .collect();
            let title = format!(" 交互式技能选择器 [过滤: {query}] (Enter 填入) ");
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::SessionTree {
            entries,
            selected,
            pan_x,
        } => {
            let rows: Vec<(String, String)> = entries
                .iter()
                .map(|e| {
                    let branch = if e.depth == 0 {
                        format!("#{} {}", e.seq, e.kind_tag)
                    } else {
                        format!("  └─ #{} {}", e.seq, e.kind_tag)
                    };
                    let shifted: String = e.summary.chars().skip(*pan_x).collect();
                    let fork_hint = if e.can_fork { " [f 分叉]" } else { "" };
                    (branch, format!("{shifted}{fork_hint}"))
                })
                .collect();
            let title = format!(
                " 会话时间线与分支树 (/tree · ↑↓选行 · ←→平移 x={} · Enter/f 分叉) ",
                *pan_x
            );
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::HelpModal { scroll } => {
            render_help_modal(frame, full_area, *scroll, theme, state);
        }
        OverlayKind::Approval(ap) => {
            let tool_name = ap.tool_name.clone();
            let args_preview = ap.args_preview.clone();
            let editing_reason = ap.editing_reason;
            let deny_reason = ap.deny_reason.clone();
            render_approval_modal(
                frame,
                full_area,
                &tool_name,
                &args_preview,
                editing_reason,
                &deny_reason,
                theme,
                state,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_anchored_list(
    frame: &mut Frame<'_>,
    full_area: Rect,
    anchor_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_h = (rows.len().clamp(1, 8) + 2) as u16;
    let box_w = full_area.width.clamp(24, 72);
    let y = anchor_area.y.saturating_sub(box_h);
    let x = anchor_area.x.min(full_area.width.saturating_sub(box_w));
    let rect = Rect::new(x, y, box_w, box_h.min(anchor_area.y.max(3)));
    draw_list_box(frame, rect, title, rows, selected, theme, state);
}

fn render_centered_list(
    frame: &mut Frame<'_>,
    full_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(32, 74);
    let box_h = (rows.len().clamp(1, 12) + 2) as u16;
    let box_h = box_h.min(full_area.height.saturating_sub(2).max(4));
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    draw_list_box(
        frame,
        Rect::new(x, y, box_w, box_h),
        title,
        rows,
        selected,
        theme,
        state,
    );
}

fn draw_list_box(
    frame: &mut Frame<'_>,
    rect: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border_focus))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    if rows.is_empty() || inner.height == 0 {
        frame.render_widget(
            Paragraph::new(Span::styled("  (无匹配项)", Style::default().fg(theme.text_dim))),
            inner,
        );
        return;
    }

    let sel = selected.min(rows.len() - 1);
    let cap = inner.height as usize;
    let start = sel.saturating_sub(cap / 2).min(rows.len().saturating_sub(cap));
    let end = (start + cap).min(rows.len());

    let mut lines = Vec::new();
    for (offset, (left, right)) in rows[start..end].iter().enumerate() {
        let abs = start + offset;
        let is_sel = abs == sel;
        let row_rect = Rect::new(inner.x, inner.y + offset as u16, inner.width, 1);
        state.item_rects.push((abs, row_rect));

        let bg = if is_sel {
            theme.card_bg_alt
        } else {
            theme.panel_bg
        };
        let ptr = if is_sel { "▸ " } else { "  " };
        let mut spans = vec![
            Span::styled(
                ptr,
                Style::default()
                    .fg(if is_sel { theme.accent } else { theme.text_dim })
                    .bg(bg),
            ),
            Span::styled(
                left.clone(),
                Style::default()
                    .fg(if is_sel { theme.accent } else { theme.text })
                    .bg(bg)
                    .add_modifier(if is_sel {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ];
        if !right.is_empty() {
            let used = 2 + display_width(left) + 2;
            if inner.width as usize > used + 4 {
                spans.push(Span::styled(
                    format!("  {right}"),
                    Style::default().fg(theme.text_muted).bg(bg),
                ));
            }
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

#[allow(clippy::too_many_arguments)]
fn render_approval_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    tool_name: &str,
    args_preview: &str,
    editing_reason: bool,
    deny_reason: &str,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(40, 82);
    let box_h = if editing_reason { 11u16 } else { 10u16 }.min(full_area.height.saturating_sub(2).max(6));
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            " ⚠ 工具执行审批 (Approval Required) ",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.warning))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("工具：", Style::default().fg(theme.text_muted)),
            Span::styled(
                tool_name.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!("参数：{args_preview}"),
            Style::default().fg(theme.text),
        )),
        Line::default(),
    ];

    if editing_reason {
        lines.push(Line::from(vec![
            Span::styled(
                "拒绝理由（Enter 提交 / Esc 返回）：",
                Style::default().fg(theme.warning).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{deny_reason}▏"),
                Style::default().fg(theme.text),
            ),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled(
                " [y / Enter] 允许本次 ",
                Style::default()
                    .fg(theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                " [a] 始终允许 ",
                Style::default().fg(theme.info).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                " [r] 拒绝附理由 ",
                Style::default().fg(theme.warning).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                " [n/Esc] 拒绝 ",
                Style::default().fg(theme.error).add_modifier(Modifier::BOLD),
            ),
        ]));
    }
    if !editing_reason && inner.height >= 4 {
        state.item_rects.push((0, Rect::new(inner.x, inner.y + 3, 18, 1)));
        state
            .item_rects
            .push((1, Rect::new(inner.x + 19, inner.y + 3, 15, 1)));
        state
            .item_rects
            .push((2, Rect::new(inner.x + 48, inner.y + 3, 16, 1)));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_help_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    scroll: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(40, 80);
    let box_h = full_area.height.saturating_sub(4).clamp(10, 22);
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            " dsh-TUI 快捷键与命令指南 (Esc 关闭) ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border_focus))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    let items = [
        ("Enter / Tab (运行中)", "空闲发送；运行中 Tab 排入 Follow-up 队列，Ctrl+Enter 立即插队"),
        ("Shift+Enter / Ctrl+J", "插入换行；粘贴超长文本自动折叠为原子块"),
        ("鼠标划词 / 双击 / 三击", "应用级分栏选区（自动拼接软换行），松手自动复制；F2 切换鼠标追踪"),
        ("Ctrl+B / Ctrl+O", "Ctrl+B 开关侧栏；Ctrl+O 展开/折叠思考与完整工具输出"),
        ("Ctrl+T / /trace", "打开全屏三泳道轨迹分析器（WaveBand + Timeline + Hotspots）"),
        ("Ctrl+G / Ctrl+E", "Ctrl+G 调用 $EDITOR 编辑草稿；Ctrl+E 展开折叠粘贴块"),
        ("Ctrl+R / Ctrl+Z / Ctrl+Y", "历史模糊搜索 / 词级撤销 / KillRing 粘回"),
        ("/tree /skills /view", "会话分支树回溯 / 交互技能选择 / Fullscreen↔Inline 视图热切"),
        ("/model /effort /theme", "交互切换模型、思考深度与 Oklab 自适应主题"),
        ("/approval [auto|ask]", "开启或关闭工具调用交互式审批确认弹窗（支持拒绝附理由）"),
        ("Esc / Ctrl+C", "优先取消选区或浮层 -> 中断生成 -> 清空输入 -> 双击退出"),
    ];
    let cap = inner.height as usize;
    let start = scroll.min(items.len().saturating_sub(cap));
    let end = (start + cap).min(items.len());
    let lines: Vec<Line<'static>> = items[start..end]
        .iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(
                    format!("{k:<24}"),
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled((*v).to_string(), Style::default().fg(theme.text)),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_ranks_exact_and_prefix_and_subsequence_matches() {
        assert!(fuzzy_score("crates/dsh-tui/src/app.rs", "app").is_some());
        let s1 = fuzzy_score("app.rs", "app").unwrap();
        let s2 = fuzzy_score("crates/dsh-tui/src/app.rs", "app").unwrap();
        assert!(s1 > s2);
    }
}
