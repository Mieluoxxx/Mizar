//! Cache-Aware 尾部工具结果剪枝（借鉴 `references/omp` `pruning.ts` 与 `session-maintenance.ts`）。
//!
//! 核心不变量：
//! 1. **仅剪两类确定性死重**：
//!    - `supersedeReads`：被同文件后续完整（或同区间）`read` 覆盖的旧 `read` 结果 -> `[Superseded by a newer read of this file]`；
//!    - `dropUseless`：零命中 / 无信息量工具结果 -> `[Uneventful result elided]`；
//!    - 始终保护 `skill://` / `SKILL.md` 读取。
//! 2. **严格遵守 Prompt Cache 边界**：
//!    - 热缓存期改写旧回执会使前缀缓存失效，敏感协议需保护完整历史；
//!    - 达到闲置时限或上下文承压时可清仓，避免为重复结果付出摘要开销；
//!    - 已折叠至最新 `Compaction` 边界之前的条目永不触碰。
//! 3. **Append-only 持久化**：只追加 `SessionEvent::ToolResultPrune`，绝不原地改写历史条目，保证 `--resume` 与 `fork` 前缀字节一致。

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_llm::ContentBlock;
use mizar_session::{PrunedToolResult, Session, SessionEntry, SessionEvent, SessionTraceEvent};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::budget::{
    estimate_entry_tokens_with_pruned, estimate_text_tokens, estimate_tool_output_tokens,
};

pub const SUPERSEDED_NOTICE: &str = "[Superseded by a newer read of this file]";
pub const USELESS_NOTICE: &str = "[Uneventful result elided]";
pub const DEFAULT_WARM_SUFFIX_TOKEN_LIMIT: u64 = 8_000;
pub const DEFAULT_IDLE_FLUSH_MS: u128 = 90 * 60 * 1_000;

#[derive(Debug, Clone)]
pub struct TailPruneOptions {
    pub supersede_reads: bool,
    pub drop_useless: bool,
    pub warm_suffix_tokens: u64,
    pub idle_flush_ms: u128,
    pub preserve_warm_prefix: bool,
    pub now_ms: Option<u128>,
}

impl Default for TailPruneOptions {
    fn default() -> Self {
        Self {
            supersede_reads: true,
            drop_useless: true,
            warm_suffix_tokens: DEFAULT_WARM_SUFFIX_TOKEN_LIMIT,
            idle_flush_ms: DEFAULT_IDLE_FLUSH_MS,
            preserve_warm_prefix: false,
            now_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneMarker {
    pub target_seq: u64,
    pub notice: &'static str,
    pub tokens_saved: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TailPrunePlan {
    pub items: Vec<PruneMarker>,
    pub tokens_saved: u64,
    pub idle_flushed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TailPruneReport {
    pub pruned_count: usize,
    pub tokens_saved: u64,
    pub idle_flushed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ReadKey {
    Whole(String),
    Range {
        path: String,
        offset: u64,
        limit: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, Default)]
struct NewerReadState {
    saw_any: bool,
    saw_success: bool,
    newest_success_complete: bool,
}

struct Candidate {
    visible_idx: usize,
    target_seq: u64,
    notice: &'static str,
    tokens_saved: u64,
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn idle_flush_due(entries: &[SessionEntry], opts: &TailPruneOptions) -> bool {
    // 新一轮用户输入和插件通知尚未发给模型，不能据此把已闲置的缓存重新算热。
    let last = entries
        .iter()
        .rev()
        .find(|entry| {
            matches!(
                entry.event,
                SessionEvent::Assistant { .. }
                    | SessionEvent::ToolResult { .. }
                    | SessionEvent::Compaction { .. }
                    | SessionEvent::Trace {
                        detail: SessionTraceEvent::StepEnd { .. }
                            | SessionTraceEvent::StepStart { .. }
                    }
            )
        })
        .or_else(|| entries.last());
    last.is_some_and(|entry| {
        opts.now_ms
            .unwrap_or_else(now_millis)
            .saturating_sub(entry.ts)
            >= opts.idle_flush_ms
    })
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else if !matches!(
                    out.components().next_back(),
                    Some(Component::RootDir | Component::Prefix(_))
                ) {
                    out.push(Component::ParentDir);
                }
            }
            Component::Normal(s) => out.push(s),
        }
    }
    out
}

fn normalize_read_key_path(workspace: Option<&str>, raw_path: &str) -> String {
    let trimmed = raw_path.trim();
    let without_at = trimmed.strip_prefix('@').unwrap_or(trimmed);
    let expanded = if without_at == "~" {
        home_dir().unwrap_or_else(|| PathBuf::from("~"))
    } else if let Some(rest) = without_at.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        home.join(rest)
    } else {
        PathBuf::from(without_at)
    };
    let abs = if expanded.is_absolute() {
        expanded
    } else if let Some(ws) = workspace {
        Path::new(ws).join(expanded)
    } else {
        expanded
    };
    let canon = std::fs::canonicalize(&abs).unwrap_or(abs);
    normalize_lexical(&canon).to_string_lossy().into_owned()
}

fn extract_read_raw_path(args: &Value) -> Option<&str> {
    args.get("path")
        .or_else(|| args.get("file_path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn is_skill_path(raw_path: &str) -> bool {
    let trimmed = raw_path.trim();
    if trimmed
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case("skill://"))
    {
        return true;
    }
    Path::new(trimmed)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("SKILL.md"))
}

fn is_protected_tool_result(tool_name: &str, args: Option<&Value>) -> bool {
    if tool_name == "read"
        && let Some(args) = args
        && let Some(raw_path) = extract_read_raw_path(args)
    {
        return is_skill_path(raw_path);
    }
    false
}

fn parse_read_key(workspace: Option<&str>, args: &Value) -> Option<ReadKey> {
    let raw_path = extract_read_raw_path(args)?;
    if raw_path.contains("://") {
        return None;
    }
    let norm = normalize_read_key_path(workspace, raw_path);
    let offset = args
        .get("offset")
        .and_then(Value::as_u64)
        .filter(|&v| v > 1);
    let limit = args.get("limit").and_then(Value::as_u64).filter(|&v| v > 0);
    if offset.is_none() && limit.is_none() {
        Some(ReadKey::Whole(norm))
    } else {
        Some(ReadKey::Range {
            path: norm,
            offset: offset.unwrap_or(1),
            limit,
        })
    }
}

/// 判断一次成功的无偏移/无上限 `read` 结果是否完整呈现了整个文件（未发生行截断、字节截断或单行截断）。
fn is_complete_read_result(key: &ReadKey, content: &str, is_error: bool) -> bool {
    if is_error || !matches!(key, ReadKey::Whole(_)) {
        return false;
    }
    if content.starts_with("[Line ") && content.contains("exceeding ") {
        return false;
    }
    if content.contains("\n\n[Showing lines ") && content.contains("Use offset=") {
        return false;
    }
    if content.contains("\n[Some lines exceeded ") && content.contains("chars and were truncated.]")
    {
        return false;
    }
    true
}

fn is_useless_result(tool_name: &str, content: &str, is_error: bool) -> bool {
    if is_error {
        return false;
    }
    let trimmed = content.trim();
    match tool_name {
        "grep" | "search" => {
            trimmed == "No matches found" || trimmed.starts_with("No matches found\n")
        }
        "find" => {
            trimmed == "No files found matching pattern"
                || trimmed.starts_with("No files found matching pattern\n")
        }
        "tool_search" => trimmed == "No matching tools found.",
        _ => false,
    }
}

fn collect_tool_calls<'a>(
    visible: &'a [SessionEntry],
    all_entries: &'a [SessionEntry],
) -> HashMap<&'a str, (&'a str, Option<Value>)> {
    let mut map = HashMap::new();
    let mut ingest = |entries: &'a [SessionEntry]| {
        for entry in entries {
            if let SessionEvent::Assistant { blocks, .. } = &entry.event {
                for b in blocks {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } = b
                    {
                        let parsed = serde_json::from_str::<Value>(arguments).ok();
                        map.insert(id.as_str(), (name.as_str(), parsed));
                    }
                }
            }
        }
    };
    // 若跨压缩边界存在引用早期 Assistant 的工具结果，优先按全量历史填充再由可见区覆盖最新调用
    if all_entries.len() > visible.len() {
        ingest(all_entries);
    }
    ingest(visible);
    map
}

/// 纯规划函数：基于当前可见表面日志、`ToolResultPrune` 映射与提示词缓存边界计算本次应追加的剪枝标记。
pub fn plan_tail_pruning(
    visible: &[SessionEntry],
    all_entries: &[SessionEntry],
    pruned_map: &HashMap<u64, String>,
    workspace: Option<&str>,
    opts: &TailPruneOptions,
) -> TailPrunePlan {
    if (!opts.supersede_reads && !opts.drop_useless) || visible.is_empty() {
        return TailPrunePlan::default();
    }

    let idle_flushed = idle_flush_due(
        if all_entries.is_empty() {
            visible
        } else {
            all_entries
        },
        opts,
    );
    let suffix_limit = if opts.preserve_warm_prefix {
        0
    } else {
        opts.warm_suffix_tokens
    };
    if !idle_flushed && suffix_limit == 0 {
        return TailPrunePlan::default();
    }

    let tool_calls = collect_tool_calls(visible, all_entries);
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut superseded_seqs: HashSet<u64> = HashSet::new();

    if opts.supersede_reads {
        let mut newer_by_key: HashMap<ReadKey, NewerReadState> = HashMap::new();
        for (idx, entry) in visible.iter().enumerate().rev() {
            let SessionEvent::ToolResult {
                tool_call_id,
                name,
                content,
                is_error,
                images,
            } = &entry.event
            else {
                continue;
            };
            if pruned_map.contains_key(&entry.seq) || name != "read" {
                continue;
            }
            let Some((call_name, call_args)) = tool_calls.get(tool_call_id.as_str()) else {
                continue;
            };
            if is_protected_tool_result(call_name, call_args.as_ref()) {
                continue;
            }
            let Some(args) = call_args.as_ref() else {
                continue;
            };
            let Some(key) = parse_read_key(workspace, args) else {
                continue;
            };

            let same = newer_by_key.get(&key).copied();
            let parent = match &key {
                ReadKey::Whole(_) => None,
                ReadKey::Range { path, .. } => {
                    newer_by_key.get(&ReadKey::Whole(path.clone())).copied()
                }
            };
            let superseded = if *is_error {
                same.is_some_and(|s| s.saw_any) || parent.is_some_and(|p| p.saw_success)
            } else {
                same.is_some_and(|s| s.saw_success)
                    || parent.is_some_and(|p| p.newest_success_complete)
            };

            let mut state = same.unwrap_or_default();
            state.saw_any = true;
            if !*is_error && !state.saw_success {
                state.saw_success = true;
                state.newest_success_complete = is_complete_read_result(&key, content, *is_error);
            }
            newer_by_key.insert(key, state);

            if superseded {
                superseded_seqs.insert(entry.seq);
                candidates.push(Candidate {
                    visible_idx: idx,
                    target_seq: entry.seq,
                    notice: SUPERSEDED_NOTICE,
                    tokens_saved: estimate_tool_output_tokens(content, images)
                        .saturating_sub(estimate_text_tokens(SUPERSEDED_NOTICE)),
                });
            }
        }
    }

    if opts.drop_useless {
        for (idx, entry) in visible.iter().enumerate() {
            let SessionEvent::ToolResult {
                tool_call_id,
                name,
                content,
                is_error,
                images,
            } = &entry.event
            else {
                continue;
            };
            if pruned_map.contains_key(&entry.seq) || superseded_seqs.contains(&entry.seq) {
                continue;
            }
            let call_args = tool_calls
                .get(tool_call_id.as_str())
                .and_then(|(_, a)| a.as_ref());
            if is_protected_tool_result(name, call_args) {
                continue;
            }
            if images.is_empty() && is_useless_result(name, content, *is_error) {
                candidates.push(Candidate {
                    visible_idx: idx,
                    target_seq: entry.seq,
                    notice: USELESS_NOTICE,
                    tokens_saved: estimate_text_tokens(content)
                        .saturating_sub(estimate_text_tokens(USELESS_NOTICE)),
                });
            }
        }
    }

    // 小回执的占位文本可能更长，清仓时写这种标记只会使实测用量锚点失效。
    if opts.idle_flush_ms == 0 && opts.warm_suffix_tokens == 0 {
        candidates.retain(|candidate| candidate.tokens_saved > 0);
    }
    if candidates.is_empty() {
        return TailPrunePlan {
            idle_flushed,
            ..Default::default()
        };
    }
    candidates.sort_by_key(|c| c.visible_idx);

    let selected: Vec<Candidate> = if idle_flushed {
        candidates
    } else {
        let mut suffix_tokens = 0u64;
        let mut next_cand = candidates.len();
        let mut kept_reversed = Vec::new();

        for idx in (0..visible.len()).rev() {
            if suffix_tokens > suffix_limit || next_cand == 0 {
                break;
            }
            while next_cand > 0 && candidates[next_cand - 1].visible_idx == idx {
                next_cand -= 1;
                let popped = &candidates[next_cand];
                kept_reversed.push(Candidate {
                    visible_idx: popped.visible_idx,
                    target_seq: popped.target_seq,
                    notice: popped.notice,
                    tokens_saved: popped.tokens_saved,
                });
            }
            suffix_tokens = suffix_tokens
                .saturating_add(estimate_entry_tokens_with_pruned(&visible[idx], pruned_map));
        }
        kept_reversed.reverse();
        kept_reversed
    };

    if selected.is_empty() {
        return TailPrunePlan::default();
    }

    let tokens_saved: u64 = selected.iter().map(|c| c.tokens_saved).sum();
    let items = selected
        .into_iter()
        .map(|c| PruneMarker {
            target_seq: c.target_seq,
            notice: c.notice,
            tokens_saved: c.tokens_saved,
        })
        .collect();

    TailPrunePlan {
        items,
        tokens_saved,
        idle_flushed,
    }
}

/// 针对实时 `Session` 执行一次 cache-aware 尾部剪枝，并将选定结果以 append-only `ToolResultPrune` 落盘。
pub fn prune_session_tail(
    session: &Arc<Session>,
    workspace: Option<&str>,
    opts: &TailPruneOptions,
) -> TailPruneReport {
    prune_session_tail_cancellable(session, workspace, opts, None)
}

pub(crate) fn prune_session_tail_cancellable(
    session: &Arc<Session>,
    workspace: Option<&str>,
    opts: &TailPruneOptions,
    cancel: Option<&CancellationToken>,
) -> TailPruneReport {
    if cancel.is_some_and(CancellationToken::is_cancelled)
        || (opts.preserve_warm_prefix
            && !session.with_all_entries(|entries| idle_flush_due(entries, opts)))
    {
        return TailPruneReport::default();
    }
    let visible = session.visible_log();
    let all_entries = session.all_entries();
    let pruned_map = session.pruned_map();
    let plan = plan_tail_pruning(&visible, &all_entries, &pruned_map, workspace, opts);
    if plan.items.is_empty() {
        return TailPruneReport {
            idle_flushed: plan.idle_flushed,
            ..Default::default()
        };
    }
    let chars_by_seq: HashMap<u64, usize> = visible
        .iter()
        .filter_map(|e| match &e.event {
            SessionEvent::ToolResult { content, .. } => Some((e.seq, content.chars().count())),
            _ => None,
        })
        .collect();
    let pruned_count = plan.items.len();
    let pruned_items: Vec<PrunedToolResult> = plan
        .items
        .into_iter()
        .map(|item| PrunedToolResult {
            target_seq: item.target_seq,
            replacement: item.notice.to_string(),
            original_chars: chars_by_seq.get(&item.target_seq).copied().unwrap_or(0),
        })
        .collect();
    if cancel.is_some_and(CancellationToken::is_cancelled) {
        return TailPruneReport::default();
    }
    session.append_prune(pruned_items);
    TailPruneReport {
        pruned_count,
        tokens_saved: plan.tokens_saved,
        idle_flushed: plan.idle_flushed,
    }
}
