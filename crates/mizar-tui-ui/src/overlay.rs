//! 声明式浮层栈（`OverlayStack`）：
//!
//! 统一承接 `/` 命令补全、`@` 文件模糊引用（接 `ignore` crate）、交互式 `/model`、`/effort`、
//! `/theme` 选择器、`Ctrl+R` 历史模糊搜索、`?` 帮助面板，以及可选的 `tools/pre-execute` 审批弹窗。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use futures::channel::oneshot;
use ignore::WalkBuilder;
use mizar_llm_types::{ProviderInfo, ReasoningEffort};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::editor::display_width;
use crate::mentions::{
    MentionCandidate, MentionCandidateKind, fuzzy_subsequence_score, is_path_like_query,
};
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

const WORKSPACE_INDEX_TTL: Duration = Duration::from_secs(5);
const WORKSPACE_INDEX_MAX_ENTRIES: usize = 8_192;
const WORKSPACE_COLD_WAIT: Duration = Duration::from_millis(60);
const MENTION_RESULT_LIMIT: usize = 25;

#[derive(Debug, Default)]
struct CachedWorkspaceState {
    workspace: Option<PathBuf>,
    entries: Vec<MentionCandidate>,
    refreshed_at: Option<Instant>,
    scanning: bool,
    generation: u64,
}

/// 工作区文件与目录索引缓存（利用 `ignore` crate 遵守 `.gitignore`，后台非阻塞刷新 + 路径直列模式）。
#[derive(Debug, Clone)]
pub struct WorkspaceFileCache {
    shared: Arc<(Mutex<CachedWorkspaceState>, Condvar)>,
    last_seen_gen: u64,
}

impl Default for WorkspaceFileCache {
    fn default() -> Self {
        Self {
            shared: Arc::new((Mutex::new(CachedWorkspaceState::default()), Condvar::new())),
            last_seen_gen: 0,
        }
    }
}

impl WorkspaceFileCache {
    fn spawn_scan_if_needed(&self, workspace: &Path) -> bool {
        let mut guard = self.shared.0.lock().unwrap();
        let ws_changed = guard.workspace.as_deref() != Some(workspace);
        if ws_changed {
            guard.workspace = Some(workspace.to_path_buf());
            guard.entries.clear();
            guard.refreshed_at = None;
            guard.scanning = false;
        }
        let stale = guard
            .refreshed_at
            .is_none_or(|t| t.elapsed() > WORKSPACE_INDEX_TTL);
        let cold = guard.entries.is_empty() && guard.refreshed_at.is_none();
        if stale && !guard.scanning {
            guard.scanning = true;
            let shared = Arc::clone(&self.shared);
            let ws = workspace.to_path_buf();
            std::thread::spawn(move || {
                let scanned = scan_workspace_entries(&ws);
                let mut g = shared.0.lock().unwrap();
                if g.workspace.as_deref() == Some(ws.as_path()) {
                    g.entries = scanned;
                    g.refreshed_at = Some(Instant::now());
                    g.generation = g.generation.wrapping_add(1);
                }
                g.scanning = false;
                shared.1.notify_all();
            });
        }
        if cold {
            let deadline = Instant::now() + WORKSPACE_COLD_WAIT;
            while guard.scanning && guard.entries.is_empty() {
                let rem = deadline.saturating_duration_since(Instant::now());
                if rem.is_zero() {
                    break;
                }
                let (next_guard, res) = self.shared.1.wait_timeout(guard, rem).unwrap();
                guard = next_guard;
                if res.timed_out() {
                    break;
                }
            }
            if guard.entries.is_empty() {
                // 超大型仓库首次扫描未在 60ms 内结束时，先同步返回首层目录条目避免卡顿
                guard.entries = list_shallow_workspace_entries(workspace);
            }
        }
        cold
    }

    /// 检查后台索引是否刚更新了新代次，供展开的 `@` 浮层无感刷新候选列表。
    pub fn consume_generation_update(&mut self) -> bool {
        let g = self.shared.0.lock().unwrap();
        if g.generation != self.last_seen_gen {
            self.last_seen_gen = g.generation;
            true
        } else {
            false
        }
    }

    pub fn files(&mut self, workspace: &Path) -> Vec<String> {
        self.spawn_scan_if_needed(workspace);
        let g = self.shared.0.lock().unwrap();
        self.last_seen_gen = g.generation;
        g.entries
            .iter()
            .filter(|e| e.kind == MentionCandidateKind::File)
            .map(|e| e.path.clone())
            .collect()
    }

    pub fn match_query(&mut self, workspace: &Path, query: &str) -> Vec<MentionCandidate> {
        if is_path_like_query(query) {
            return list_path_candidates(workspace, query, MENTION_RESULT_LIMIT);
        }
        self.spawn_scan_if_needed(workspace);
        let g = self.shared.0.lock().unwrap();
        self.last_seen_gen = g.generation;
        rank_mention_candidates(&g.entries, query, MENTION_RESULT_LIMIT)
    }
}

fn scan_workspace_entries(workspace: &Path) -> Vec<MentionCandidate> {
    let mut found = Vec::new();
    let walker = WalkBuilder::new(workspace)
        .hidden(false)
        .max_depth(Some(8))
        .build();
    for entry in walker.flatten() {
        if found.len() >= WORKSPACE_INDEX_MAX_ENTRIES {
            break;
        }
        let Ok(rel) = entry.path().strip_prefix(workspace) else {
            continue;
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.is_empty() {
            continue;
        }
        let Some(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            found.push(MentionCandidate {
                path: format!("{}/", rel_str.trim_end_matches('/')),
                name,
                kind: MentionCandidateKind::Directory,
            });
        } else if ft.is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            found.push(MentionCandidate {
                path: rel_str,
                name,
                kind: MentionCandidateKind::File,
            });
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found
}

fn list_shallow_workspace_entries(workspace: &Path) -> Vec<MentionCandidate> {
    let Ok(rd) = std::fs::read_dir(workspace) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".git" || name == ".DS_Store" {
            continue;
        }
        let is_dir = entry.path().is_dir();
        out.push(MentionCandidate {
            path: if is_dir {
                format!("{name}/")
            } else {
                name.clone()
            },
            name,
            kind: if is_dir {
                MentionCandidateKind::Directory
            } else {
                MentionCandidateKind::File
            },
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn expand_home_dir_prefix(raw: &str) -> PathBuf {
    if raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = raw.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(raw)
}

fn list_path_candidates(workspace: &Path, query: &str, limit: usize) -> Vec<MentionCandidate> {
    let normalized = query.replace('\\', "/");
    let raw_slash = query.rfind(['/', '\\']);
    let slash = normalized.rfind('/');
    let bare_dir = slash.is_none() && matches!(normalized.as_str(), "." | ".." | "~");

    let dir_part = if bare_dir {
        format!("{normalized}/")
    } else if let Some(pos) = slash {
        normalized[..=pos].to_string()
    } else {
        String::new()
    };

    let display_dir = if bare_dir {
        format!("{query}/")
    } else if let Some(pos) = raw_slash {
        query[..=pos].to_string()
    } else {
        String::new()
    };

    let display_sep = if display_dir.ends_with('\\') {
        '\\'
    } else {
        '/'
    };
    let name_query = if bare_dir {
        ""
    } else if let Some(pos) = slash {
        &normalized[pos + 1..]
    } else {
        normalized.as_str()
    };

    let expanded = expand_home_dir_prefix(&dir_part);
    let target_dir = if expanded.is_absolute() {
        expanded
    } else if dir_part.is_empty() {
        workspace.to_path_buf()
    } else {
        workspace.join(expanded)
    };

    let Ok(rd) = std::fs::read_dir(&target_dir) else {
        return Vec::new();
    };
    let show_dot_hidden = name_query.starts_with('.');
    let mut entries = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if (name == ".git" || name == ".DS_Store") && !show_dot_hidden {
            continue;
        }
        let is_dir = entry.path().is_dir();
        let path = if is_dir {
            format!("{display_dir}{name}{display_sep}")
        } else {
            format!("{display_dir}{name}")
        };
        entries.push(MentionCandidate {
            path,
            name,
            kind: if is_dir {
                MentionCandidateKind::Directory
            } else {
                MentionCandidateKind::File
            },
        });
    }

    rank_mention_candidates(&entries, name_query, limit)
}

fn rank_mention_candidates(
    entries: &[MentionCandidate],
    query: &str,
    limit: usize,
) -> Vec<MentionCandidate> {
    let mut scored: Vec<(i32, MentionCandidate)> = entries
        .iter()
        .filter_map(|c| {
            fuzzy_subsequence_score(query, &c.path, &c.name, c.kind).map(|s| (s, c.clone()))
        })
        .collect();
    scored.sort_by(|(sa, a), (sb, b)| {
        sb.cmp(sa)
            .then_with(|| {
                let da = matches!(a.kind, MentionCandidateKind::Directory);
                let db = matches!(b.kind, MentionCandidateKind::Directory);
                db.cmp(&da)
            })
            .then_with(|| a.path.len().cmp(&b.path.len()))
            .then_with(|| a.path.cmp(&b.path))
    });
    scored.into_iter().take(limit).map(|(_, c)| c).collect()
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingItemKind {
    Toggle {
        value: bool,
    },
    Choice {
        options: Vec<String>,
        selected: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingEntry {
    pub key: String,
    pub category: String,
    pub label: String,
    pub description: String,
    pub kind: SettingItemKind,
}

impl SettingEntry {
    pub fn display_value(&self) -> String {
        match &self.kind {
            SettingItemKind::Toggle { value } => {
                if *value {
                    "● ON".to_string()
                } else {
                    "○ OFF".to_string()
                }
            }
            SettingItemKind::Choice { options, selected } => options
                .get(*selected)
                .map(|v| format!("◂ {v} ▸"))
                .unwrap_or_default(),
        }
    }

    pub fn step_value(&mut self, delta: isize) -> String {
        match &mut self.kind {
            SettingItemKind::Toggle { value } => {
                *value = !*value;
                if *value { "true" } else { "false" }.to_string()
            }
            SettingItemKind::Choice { options, selected } => {
                if options.is_empty() {
                    return String::new();
                }
                let len = options.len() as isize;
                let next = ((*selected as isize + delta).rem_euclid(len)) as usize;
                *selected = next;
                options[next].clone()
            }
        }
    }
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
    ForkSessionAtSeq(u64),
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
                    | OverlayKind::SettingsModal { .. }
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
            .and_then(|cur| {
                items
                    .iter()
                    .position(|it| it.model_id == cur || it.label == cur)
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

    pub fn open_approval(
        &mut self,
        tool_name: String,
        args_preview: String,
        responder: oneshot::Sender<ApprovalDecision>,
    ) {
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
                            let value = item.step_value(-1);
                            Some(OverlayAction::UpdateSetting {
                                key: item.key.clone(),
                                value,
                            })
                        } else {
                            Some(OverlayAction::None)
                        }
                    }
                    KeyCode::Right | KeyCode::Enter => {
                        if let Some(&item_idx) = filtered_indices.get(*selected) {
                            let item = &mut items[item_idx];
                            let value = item.step_value(1);
                            Some(OverlayAction::UpdateSetting {
                                key: item.key.clone(),
                                value,
                            })
                        } else {
                            Some(OverlayAction::None)
                        }
                    }
                    KeyCode::Char(' ') if query.is_empty() => {
                        if let Some(&item_idx) = filtered_indices.get(*selected) {
                            let item = &mut items[item_idx];
                            let value = item.step_value(1);
                            Some(OverlayAction::UpdateSetting {
                                key: item.key.clone(),
                                value,
                            })
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
                                    let value = item.step_value(1);
                                    Some(OverlayAction::UpdateSetting {
                                        key: item.key.clone(),
                                        value,
                                    })
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

fn filter_setting_indices(items: &[SettingEntry], query: &str) -> Vec<usize> {
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

#[allow(clippy::too_many_arguments)]
fn render_settings_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    items: &[SettingEntry],
    query: &str,
    selected: usize,
    save_path_hint: &str,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let filtered = filter_setting_indices(items, query);
    let box_w = full_area.width.saturating_sub(4).clamp(44, 86);
    let box_h = full_area.height.saturating_sub(4).clamp(11, 22);
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);

    frame.render_widget(Clear, rect);
    let title = if query.is_empty() {
        " ⚙ 设置中心 (/settings · ↑↓ 移动 · ←/→/Enter 切换 · Esc 关闭) ".to_string()
    } else {
        format!(" ⚙ 设置中心 [过滤: {query}] (Esc 清空过滤) ")
    };
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

    if inner.height < 4 || inner.width < 16 {
        return;
    }

    // 底部预留 3 行（分割线 + 当前项说明 + 保存路径提示）
    let list_h = (inner.height as usize).saturating_sub(3).max(1);
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(inner.height as usize);

    if filtered.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (没有匹配的配置项)",
            Style::default().fg(theme.text_dim),
        )));
        while lines.len() < list_h {
            lines.push(Line::default());
        }
    } else {
        let sel = selected.min(filtered.len() - 1);
        let start = sel
            .saturating_sub(list_h / 2)
            .min(filtered.len().saturating_sub(list_h));
        let end = (start + list_h).min(filtered.len());

        for (offset, &item_idx) in filtered[start..end].iter().enumerate() {
            let abs = start + offset;
            let is_sel = abs == sel;
            let item = &items[item_idx];
            let row_rect = Rect::new(inner.x, inner.y + offset as u16, inner.width, 1);
            state.item_rects.push((abs, row_rect));

            let bg = if is_sel {
                theme.card_bg_alt
            } else {
                theme.panel_bg
            };
            let ptr = if is_sel { "▸ " } else { "  " };
            let cat_tag = format!("[{}] ", item.category);
            let val_str = item.display_value();
            let val_style = match &item.kind {
                SettingItemKind::Toggle { value: true } => Style::default()
                    .fg(theme.success)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
                SettingItemKind::Toggle { value: false } => {
                    Style::default().fg(theme.text_dim).bg(bg)
                }
                SettingItemKind::Choice { .. } => Style::default()
                    .fg(theme.accent)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
            };

            let left_w = 2 + display_width(&cat_tag) + display_width(&item.label);
            let right_w = display_width(&val_str) + 1;
            let pad = (inner.width as usize)
                .saturating_sub(left_w + right_w)
                .max(2);

            lines.push(Line::from(vec![
                Span::styled(
                    ptr,
                    Style::default()
                        .fg(if is_sel { theme.accent } else { theme.text_dim })
                        .bg(bg),
                ),
                Span::styled(cat_tag, Style::default().fg(theme.info).bg(bg)),
                Span::styled(
                    item.label.clone(),
                    Style::default()
                        .fg(if is_sel { theme.accent } else { theme.text })
                        .bg(bg)
                        .add_modifier(if is_sel {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(" ".repeat(pad), Style::default().bg(bg)),
                Span::styled(val_str, val_style),
                Span::styled(" ", Style::default().bg(bg)),
            ]));
        }
        while lines.len() < list_h {
            lines.push(Line::default());
        }
    }

    lines.push(Line::from(Span::styled(
        "─".repeat(inner.width as usize),
        Style::default().fg(theme.border),
    )));
    let desc = filtered
        .get(selected.min(filtered.len().saturating_sub(1)))
        .map(|&idx| items[idx].description.as_str())
        .unwrap_or("输入关键词快速搜索配置项");
    lines.push(Line::from(Span::styled(
        format!(" {desc}"),
        Style::default().fg(theme.text_muted),
    )));
    lines.push(Line::from(Span::styled(
        format!(" 自动即时生效并保存至: {save_path_hint}"),
        Style::default().fg(theme.text_dim),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
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
        OverlayKind::SlashComplete {
            query, selected, ..
        } => {
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
            let max_w = full_area.width.clamp(28, 104);
            let left_budget = (max_w as usize).saturating_sub(6).max(12);
            let rows: Vec<(String, String)> = matches
                .iter()
                .map(|c| {
                    let display = clip_left_path(&format!("@{}", c.path), left_budget);
                    (display, String::new())
                })
                .collect();
            let title = format!(" 文件/目录引用: @{query} (Tab/Enter 插入 · Esc 隐藏) ");
            render_anchored_list_sized(
                frame, full_area, input_area, &title, &rows, *selected, max_w, 10, theme, state,
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
                .map(|s| (format!("/{} [{}]", s.name, s.source), s.description.clone()))
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
        OverlayKind::SettingsModal {
            items,
            query,
            selected,
            save_path_hint,
        } => {
            let items_cloned = items.clone();
            let query_cloned = query.clone();
            let sel = *selected;
            let save_hint = save_path_hint.clone();
            render_settings_modal(
                frame,
                full_area,
                &items_cloned,
                &query_cloned,
                sel,
                &save_hint,
                theme,
                state,
            );
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
    let box_w = full_area.width.clamp(24, 72);
    render_anchored_list_sized(
        frame,
        full_area,
        anchor_area,
        title,
        rows,
        selected,
        box_w,
        8,
        theme,
        state,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_anchored_list_sized(
    frame: &mut Frame<'_>,
    full_area: Rect,
    anchor_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    box_w: u16,
    max_rows: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_h = (rows.len().clamp(1, max_rows) + 2) as u16;
    let y = anchor_area.y.saturating_sub(box_h);
    let x = anchor_area.x.min(full_area.width.saturating_sub(box_w));
    let rect = Rect::new(x, y, box_w, box_h.min(anchor_area.y.max(3)));
    draw_list_box(frame, rect, title, rows, selected, theme, state);
}

fn clip_left_path(s: &str, max_cols: usize) -> String {
    if display_width(s) <= max_cols {
        return s.to_string();
    }
    let mut kept = Vec::new();
    let mut w = 1usize; // 预留前缀 '…'
    for ch in s.chars().rev() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > max_cols {
            break;
        }
        kept.push(ch);
        w += cw;
    }
    kept.reverse();
    let tail: String = kept.into_iter().collect();
    format!("…{tail}")
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
            Paragraph::new(Span::styled(
                "  (无匹配项)",
                Style::default().fg(theme.text_dim),
            )),
            inner,
        );
        return;
    }

    let sel = selected.min(rows.len() - 1);
    let cap = inner.height as usize;
    let start = sel
        .saturating_sub(cap / 2)
        .min(rows.len().saturating_sub(cap));
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
    let box_h =
        if editing_reason { 11u16 } else { 10u16 }.min(full_area.height.saturating_sub(2).max(6));
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
                Style::default()
                    .fg(theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("{deny_reason}▏"), Style::default().fg(theme.text)),
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
                Style::default()
                    .fg(theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                " [n/Esc] 拒绝 ",
                Style::default()
                    .fg(theme.error)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }
    if !editing_reason && inner.height >= 4 {
        state
            .item_rects
            .push((0, Rect::new(inner.x, inner.y + 3, 18, 1)));
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
            " Mizar 快捷键与命令指南 (Esc 关闭) ",
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
        (
            "Enter (空闲 / 运行中)",
            "空闲直接发送；运行中自动加入 Follow-up 队列",
        ),
        (
            "!cmd / !!cmd",
            "直接执行本地 Bash 命令（! 载入上下文 / !! 仅本地不入上下文）",
        ),
        ("Ctrl+Enter", "立即发送（运行中打断当前回合并立即插队发送）"),
        (
            "Shift+Enter / Ctrl+J",
            "插入换行；粘贴超长文本自动折叠为原子块",
        ),
        ("Tab", "循环切换思考等级（补全浮层内为补全选中项）"),
        ("Ctrl+L", "打开交互式模型选择器（等同 /model）"),
        ("Esc", "取消选区或浮层 -> 暂停当前生成 -> 清空输入"),
        (
            "Ctrl+C",
            "复制选区 / 关闭弹窗 / 单击清空输入 / 中断或双击退出",
        ),
        (
            "鼠标划词 / 双击 / 三击",
            "应用级分栏选区（自动拼接软换行），松手自动复制；F2 切换鼠标追踪",
        ),
        (
            "Ctrl+B / Ctrl+O",
            "Ctrl+B 开关侧栏；Ctrl+O 展开/折叠思考与完整工具输出",
        ),
        (
            "Ctrl+T / /trace",
            "打开全屏三泳道轨迹分析器（WaveBand + Timeline + Hotspots）",
        ),
        (
            "Ctrl+G / Ctrl+E",
            "Ctrl+G 调用 $EDITOR 编辑草稿；Ctrl+E 展开折叠粘贴块",
        ),
        (
            "Ctrl+R / Ctrl+Z / Ctrl+Y",
            "历史模糊搜索 / 词级撤销 / KillRing 粘回",
        ),
        (
            "/settings /compact",
            "打开交互式设置面板（即时生效并持久化）/ 手动压缩上下文",
        ),
        (
            "/tree /skills /view",
            "会话分支树回溯 / 交互技能选择 / Fullscreen↔Inline 视图热切",
        ),
        (
            "/model /effort /theme",
            "交互切换模型、思考深度与 Oklab 自适应主题",
        ),
        (
            "/approval [auto|ask]",
            "开启或关闭工具调用交互式审批确认弹窗（支持拒绝附理由）",
        ),
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
        assert!(
            fuzzy_subsequence_score(
                "app",
                "crates/mizar-tui/src/app.rs",
                "app.rs",
                MentionCandidateKind::File,
            )
            .is_some()
        );
        let s1 =
            fuzzy_subsequence_score("app", "app.rs", "app.rs", MentionCandidateKind::File).unwrap();
        let s2 = fuzzy_subsequence_score(
            "app",
            "crates/mizar-tui/src/app.rs",
            "app.rs",
            MentionCandidateKind::File,
        )
        .unwrap();
        assert!(s1 > s2);
    }

    #[test]
    fn workspace_file_cache_indexes_over_256_entries_and_lists_directories() {
        let tmp =
            std::env::temp_dir().join(format!("mizar_tui_file_cache_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("crates/sub")).unwrap();
        for idx in 0..320 {
            std::fs::write(tmp.join(format!("crates/file_{idx:03}.rs")), "").unwrap();
        }
        std::fs::write(tmp.join("crates/sub/target_deep.rs"), "").unwrap();

        let mut cache = WorkspaceFileCache::default();
        let matches = cache.match_query(&tmp, "target_deep");
        assert!(
            matches
                .iter()
                .any(|c| c.path == "crates/sub/target_deep.rs"),
            "expected deep file beyond 256 limit, got: {matches:?}"
        );

        let dir_matches = cache.match_query(&tmp, "crates/s");
        assert!(
            dir_matches
                .iter()
                .any(|c| c.path == "crates/sub/" && c.kind == MentionCandidateKind::Directory),
            "expected directory candidate in path-like query, got: {dir_matches:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn file_mention_overlay_renders_directory_without_dir_badge() {
        let mut state = OverlayState {
            active: Some(OverlayKind::FileMention {
                token_start: 0,
                token_end: 8,
                path_end: 8,
                is_quoted: false,
                query: "crates/".into(),
                matches: vec![
                    MentionCandidate {
                        path: "crates/mizar-tui/".into(),
                        name: "mizar-tui/".into(),
                        kind: MentionCandidateKind::Directory,
                    },
                    MentionCandidate {
                        path: "crates/Cargo.toml".into(),
                        name: "Cargo.toml".into(),
                        kind: MentionCandidateKind::File,
                    },
                ],
                selected: 0,
            }),
            ..Default::default()
        };
        let theme = Theme::from_preset(ThemePreset::Auto, None);
        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_overlay(
                    frame,
                    Rect::new(0, 0, 80, 20),
                    Rect::new(0, 15, 80, 5),
                    &mut state,
                    &[],
                    &theme,
                );
            })
            .unwrap();

        let buf = terminal.backend().buffer();
        let mut lines = Vec::new();
        for y in 0..buf.area.height {
            let mut row = String::new();
            for x in 0..buf.area.width {
                row.push_str(buf[(x, y)].symbol());
            }
            lines.push(row);
        }
        let rendered = lines.join("\n");
        assert!(
            rendered.contains("@crates/mizar-tui/"),
            "rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("@crates/Cargo.toml"),
            "rendered:\n{rendered}"
        );
        assert!(
            !rendered.contains("  dir"),
            "unexpected dir badge in rendered overlay:\n{rendered}"
        );
    }

    #[test]
    fn settings_modal_renders_modules_category_and_refresh_preserves_selected_key() {
        let mut state = OverlayState::default();
        state.open_settings(
            vec![
                SettingEntry {
                    key: "ui.showThinking".into(),
                    category: "界面".into(),
                    label: "显示思考过程".into(),
                    description: "开关推理块".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
                SettingEntry {
                    key: "compaction.enabled".into(),
                    category: "压缩".into(),
                    label: "自动压缩".into(),
                    description: "启用上下文压缩".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
                SettingEntry {
                    key: "plugin.mcp".into(),
                    category: "模块".into(),
                    label: "MCP 客户端 (mcp)".into(),
                    description: "连接外部 MCP 服务器并暴露其工具与提示词".into(),
                    kind: SettingItemKind::Toggle { value: true },
                },
                SettingEntry {
                    key: "plugin.codemode".into(),
                    category: "模块".into(),
                    label: "脚本执行沙箱 (codemode)".into(),
                    description: "通过 QuickJS 沙箱批量编排与调用工具".into(),
                    kind: SettingItemKind::Toggle { value: false },
                },
            ],
            None,
            "~/.mizar/settings.json".into(),
        );
        if let Some(OverlayKind::SettingsModal { selected, .. }) = &mut state.active {
            *selected = 2; // 指向 plugin.mcp
        }

        let theme = Theme::from_preset(ThemePreset::Auto, None);
        let backend = ratatui::backend::TestBackend::new(90, 20);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                render_overlay(
                    frame,
                    Rect::new(0, 0, 90, 20),
                    Rect::new(0, 15, 90, 5),
                    &mut state,
                    &[],
                    &theme,
                );
            })
            .unwrap();

        let buf = terminal.backend().buffer();
        let mut rendered = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                rendered.push_str(buf[(x, y)].symbol());
            }
            rendered.push('\n');
        }
        let compact = rendered.replace(' ', "");
        assert!(
            compact.contains("[模块]MCP客户端(mcp)") && rendered.contains("● ON"),
            "rendered:\n{rendered}"
        );
        assert!(
            compact.contains("[模块]脚本执行沙箱(codemode)") && rendered.contains("○ OFF"),
            "rendered:\n{rendered}"
        );

        // set_setting_toggle 回退某一项
        state.set_setting_toggle("plugin.mcp", false);
        // refresh_settings 删除 compaction.enabled 后光标应仍锚定到 plugin.mcp（新索引 1）
        state.refresh_settings(vec![
            SettingEntry {
                key: "ui.showThinking".into(),
                category: "界面".into(),
                label: "显示思考过程".into(),
                description: "开关推理块".into(),
                kind: SettingItemKind::Toggle { value: true },
            },
            SettingEntry {
                key: "plugin.mcp".into(),
                category: "模块".into(),
                label: "MCP 客户端 (mcp)".into(),
                description: "连接外部 MCP 服务器并暴露其工具与提示词".into(),
                kind: SettingItemKind::Toggle { value: false },
            },
        ]);
        match &state.active {
            Some(OverlayKind::SettingsModal {
                items, selected, ..
            }) => {
                assert_eq!(*selected, 1);
                assert_eq!(items[*selected].key, "plugin.mcp");
                assert_eq!(
                    items[*selected].kind,
                    SettingItemKind::Toggle { value: false }
                );
            }
            _ => panic!("expected SettingsModal after refresh_settings"),
        }
    }
}
