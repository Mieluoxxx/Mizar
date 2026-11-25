//! 工作区文件与目录索引缓存（遵守 `.gitignore` 的后台非阻塞刷新与路径直列模式）。
//!
//! 迁移自 `mizar-tui-ui/src/overlay.rs`，供 TUI `@` 补全与 Merak 共用。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ignore::WalkBuilder;

use crate::mentions::is_path_like_query;
pub use crate::mentions::{MentionCandidate, MentionCandidateKind, fuzzy_subsequence_score};

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

pub fn expand_home_dir_prefix(raw: &str) -> PathBuf {
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

pub fn list_path_candidates(workspace: &Path, query: &str, limit: usize) -> Vec<MentionCandidate> {
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

pub fn rank_mention_candidates(
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
