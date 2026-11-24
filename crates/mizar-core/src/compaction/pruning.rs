//! Cache-Aware 尾部工具结果剪枝（借鉴 `references/omp` `pruning.ts` 与 `session-maintenance.ts`）。
//!
//! 核心不变量：
//! 1. **仅剪两类确定性死重**：
//!    - `supersedeReads`：被同文件后续完整（或同区间）`read` 覆盖的旧 `read` 结果 -> `[Superseded by a newer read of this file]`；
//!    - `dropUseless`：零命中 / 无信息量工具结果 -> `[Uneventful result elided]`；
//!    - 始终保护 `skill` 工具与 `skill://` / `SKILL.md` 读取。
//! 2. **严格遵守 Prompt Cache 边界**：
//!    - 热缓存期间仅裁剪后续可见消息估算 token 总和 `<= 8,000` 的尾部候选（前缀绑定思考模型直接置 `0`）；
//!    - 仅当会话闲置超过 `90 分钟`（覆盖 1 小时 prompt cache TTL）时，才对最新压缩边界之后的全部可见候选清仓；
//!    - 已折叠至最新 `Compaction` 边界之前的条目永不触碰。
//! 3. **Append-only 持久化**：只追加 `SessionTraceEvent::ToolResultPrune`，绝不原地改写历史条目，保证 `--resume` 与 `fork` 前缀字节一致。

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use mizar_llm::ContentBlock;
use mizar_session::{Session, SessionEntry, SessionEvent, SessionTraceEvent};
use serde_json::Value;

use crate::compaction::budget::{estimate_entry_tokens_with_pruned, estimate_text_tokens};

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
    pub is_prefix_bound_thinking: bool,
    pub now_ms: Option<u128>,
}

impl Default for TailPruneOptions {
    fn default() -> Self {
        Self {
            supersede_reads: true,
            drop_useless: true,
            warm_suffix_tokens: DEFAULT_WARM_SUFFIX_TOKEN_LIMIT,
            idle_flush_ms: DEFAULT_IDLE_FLUSH_MS,
            is_prefix_bound_thinking: false,
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

impl ReadKey {
    fn whole_path(&self) -> &str {
        match self {
            Self::Whole(p) | Self::Range { path: p, .. } => p,
        }
    }
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
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"))
    } else if let Some(rest) = without_at.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
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
    if tool_name == "skill" {
        return true;
    }
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

fn estimate_savings(content: &str, notice: &str) -> u64 {
    estimate_text_tokens(content).saturating_sub(estimate_text_tokens(notice))
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
            let _ = key.whole_path();
            newer_by_key.insert(key, state);

            if superseded {
                superseded_seqs.insert(entry.seq);
                candidates.push(Candidate {
                    visible_idx: idx,
                    target_seq: entry.seq,
                    notice: SUPERSEDED_NOTICE,
                    tokens_saved: estimate_savings(content, SUPERSEDED_NOTICE),
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
            if is_useless_result(name, content, *is_error) {
                candidates.push(Candidate {
                    visible_idx: idx,
                    target_seq: entry.seq,
                    notice: USELESS_NOTICE,
                    tokens_saved: estimate_savings(content, USELESS_NOTICE),
                });
            }
        }
    }

    if candidates.is_empty() {
        return TailPrunePlan::default();
    }
    candidates.sort_by_key(|c| c.visible_idx);

    let now_ms = opts.now_ms.unwrap_or_else(now_millis);
    let last_visible_ts = visible.iter().map(|e| e.ts).max();
    let idle_flushed =
        last_visible_ts.is_some_and(|ts| now_ms.saturating_sub(ts) >= opts.idle_flush_ms);

    let selected: Vec<Candidate> = if idle_flushed {
        candidates
    } else {
        let suffix_limit = if opts.is_prefix_bound_thinking {
            0
        } else {
            opts.warm_suffix_tokens
        };
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

/// 针对实时 `Session` 执行一次 cache-aware 尾部剪枝，并将选定结果追加为只追加 `ToolResultPrune` 标记。
pub fn prune_session_tail(
    session: &Arc<Session>,
    workspace: Option<&str>,
    opts: &TailPruneOptions,
) -> TailPruneReport {
    let visible = session.visible_log();
    let all_entries = session.all_entries();
    let pruned_map = session.pruned_map();
    let plan = plan_tail_pruning(&visible, &all_entries, &pruned_map, workspace, opts);
    if plan.items.is_empty() {
        return TailPruneReport::default();
    }
    let pruned_count = plan.items.len();
    for item in plan.items {
        session.append_trace(SessionTraceEvent::ToolResultPrune {
            target_seq: item.target_seq,
            notice: item.notice.to_string(),
        });
    }
    TailPruneReport {
        pruned_count,
        tokens_saved: plan.tokens_saved,
        idle_flushed: plan.idle_flushed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_session::SessionStore;

    fn mk_entry(seq: u64, ts: u128, event: SessionEvent) -> SessionEntry {
        SessionEntry { seq, ts, event }
    }

    fn mk_call(id: &str, name: &str, args: Value, ts: u128, seq: u64) -> SessionEntry {
        mk_entry(
            seq,
            ts,
            SessionEvent::Assistant {
                blocks: vec![ContentBlock::ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: args.to_string(),
                }],
                replay: None,
                usage: None,
            },
        )
    }

    fn mk_result(
        id: &str,
        name: &str,
        content: &str,
        is_error: bool,
        ts: u128,
        seq: u64,
    ) -> SessionEntry {
        mk_entry(
            seq,
            ts,
            SessionEvent::ToolResult {
                tool_call_id: id.into(),
                name: name.into(),
                content: content.into(),
                is_error,
            },
        )
    }

    #[test]
    fn supersedes_older_reads_respecting_completeness_range_and_skill_protection() {
        let ts = 1_000_000;
        let entries = vec![
            mk_call(
                "c1",
                "read",
                serde_json::json!({"path": "src/lib.rs", "offset": 10, "limit": 20}),
                ts,
                1,
            ),
            mk_result("c1", "read", &"range-old ".repeat(20), false, ts + 1, 2),
            mk_call(
                "c2",
                "read",
                serde_json::json!({"path": "./src/lib.rs"}),
                ts + 2,
                3,
            ),
            mk_result("c2", "read", "1\tcomplete file body", false, ts + 3, 4),
            // 新一次完整裸读覆盖上面第 2 与第 4 条；但再后面的分段读取不应覆盖完整裸读
            mk_call(
                "c3",
                "read",
                serde_json::json!({"path": "src/lib.rs"}),
                ts + 4,
                5,
            ),
            mk_result(
                "c3",
                "read",
                "1\tnewest complete file body",
                false,
                ts + 5,
                6,
            ),
            mk_call(
                "c4",
                "read",
                serde_json::json!({"path": "src/lib.rs", "offset": 1, "limit": 5}),
                ts + 6,
                7,
            ),
            mk_result("c4", "read", "1\tpartial", false, ts + 7, 8),
            // 被分页截断的裸读不得覆盖先前的分段读取
            mk_call(
                "c5",
                "read",
                serde_json::json!({"path": "big.log", "offset": 2500, "limit": 10}),
                ts + 8,
                9,
            ),
            mk_result("c5", "read", "2500\ttail line", false, ts + 9, 10),
            mk_call(
                "c6",
                "read",
                serde_json::json!({"path": "big.log"}),
                ts + 10,
                11,
            ),
            mk_result(
                "c6",
                "read",
                "1\thead\n\n[Showing lines 1-2000 of 3000. Use offset=2001 to continue.]",
                false,
                ts + 11,
                12,
            ),
            // skill 工具与 skill 路径读取始终受保护
            mk_call(
                "c7",
                "read",
                serde_json::json!({"path": ".agents/skills/demo/SKILL.md"}),
                ts + 12,
                13,
            ),
            mk_result("c7", "read", "old skill body", false, ts + 13, 14),
            mk_call(
                "c8",
                "read",
                serde_json::json!({"path": ".agents/skills/demo/SKILL.md"}),
                ts + 14,
                15,
            ),
            mk_result("c8", "read", "new skill body", false, ts + 15, 16),
            mk_call(
                "c9",
                "read",
                serde_json::json!({"path": "skill://demo"}),
                ts + 16,
                17,
            ),
            mk_result("c9", "read", "old skill url", false, ts + 17, 18),
            mk_call(
                "c10",
                "read",
                serde_json::json!({"path": "skill://demo"}),
                ts + 18,
                19,
            ),
            mk_result("c10", "read", "new skill url", false, ts + 19, 20),
            // 零命中搜索由 dropUseless 剪枝
            mk_call(
                "c11",
                "grep",
                serde_json::json!({"pattern": "nope"}),
                ts + 20,
                21,
            ),
            mk_result("c11", "grep", "No matches found", false, ts + 21, 22),
            // 报错的 grep 不得被 dropUseless 误剪
            mk_call(
                "c12",
                "grep",
                serde_json::json!({"pattern": "["}),
                ts + 22,
                23,
            ),
            mk_result("c12", "grep", "No matches found", true, ts + 23, 24),
        ];

        let plan = plan_tail_pruning(
            &entries,
            &entries,
            &HashMap::new(),
            Some("/repo"),
            &TailPruneOptions {
                now_ms: Some(ts + 100),
                ..Default::default()
            },
        );
        assert!(!plan.idle_flushed);
        let pruned_seqs: Vec<(u64, &str)> = plan
            .items
            .iter()
            .map(|m| (m.target_seq, m.notice))
            .collect();
        assert_eq!(
            pruned_seqs,
            vec![
                (2, SUPERSEDED_NOTICE),
                (4, SUPERSEDED_NOTICE),
                (22, USELESS_NOTICE),
            ]
        );
    }

    #[test]
    fn enforces_warm_suffix_limit_prefix_bound_zero_and_90m_idle_flush_after_compaction() {
        let base_ts = 10_000_000u128;
        let big_filler = "word ".repeat(8_500); // ~10.6K tokens > 8,000
        let visible = vec![
            // seq 5 为 compaction 检查点，之前的历史不可见
            mk_entry(
                5,
                base_ts,
                SessionEvent::Compaction {
                    summary: "older summary".into(),
                    first_kept_seq: 6,
                    tokens_before: None,
                    tokens_after: None,
                    digest: Default::default(),
                },
            ),
            // 深层候选（距离尾部 > 8,000 tokens）
            mk_call(
                "old_r1",
                "read",
                serde_json::json!({"path": "deep.rs"}),
                base_ts + 1,
                6,
            ),
            mk_result(
                "old_r1",
                "read",
                &"old deep ".repeat(200),
                false,
                base_ts + 2,
                7,
            ),
            mk_call(
                "old_r2",
                "read",
                serde_json::json!({"path": "deep.rs"}),
                base_ts + 3,
                8,
            ),
            mk_result("old_r2", "read", "new deep", false, base_ts + 4, 9),
            mk_entry(10, base_ts + 5, SessionEvent::assistant_text(big_filler)),
            // 热尾候选（距离尾部 <= 8,000 tokens）
            mk_call(
                "tail_r1",
                "read",
                serde_json::json!({"path": "tail.rs"}),
                base_ts + 6,
                11,
            ),
            mk_result(
                "tail_r1",
                "read",
                &"old tail ".repeat(50),
                false,
                base_ts + 7,
                12,
            ),
            mk_call(
                "tail_r2",
                "read",
                serde_json::json!({"path": "tail.rs"}),
                base_ts + 8,
                13,
            ),
            mk_result("tail_r2", "read", "new tail", false, base_ts + 9, 14),
        ];
        // compaction 之前的过期 read，绝不应被触碰
        let mut all = vec![
            mk_call(
                "pre_comp_1",
                "read",
                serde_json::json!({"path": "pre.rs"}),
                base_ts - 10,
                1,
            ),
            mk_result("pre_comp_1", "read", "old pre", false, base_ts - 9, 2),
            mk_call(
                "pre_comp_2",
                "read",
                serde_json::json!({"path": "pre.rs"}),
                base_ts - 8,
                3,
            ),
            mk_result("pre_comp_2", "read", "new pre", false, base_ts - 7, 4),
        ];
        all.extend(visible.clone());

        // 1. 热缓存（< 90m）：只剪热尾 seq 12，深层 seq 7 留在热前缀中
        let warm_plan = plan_tail_pruning(
            &visible,
            &all,
            &HashMap::new(),
            Some("/repo"),
            &TailPruneOptions {
                now_ms: Some(base_ts + 60_000),
                ..Default::default()
            },
        );
        assert!(!warm_plan.idle_flushed);
        assert_eq!(
            warm_plan
                .items
                .iter()
                .map(|i| i.target_seq)
                .collect::<Vec<_>>(),
            vec![12]
        );

        // 2. 前缀绑定思考模型（热尾上限 0）：热缓存期间一个都不剪
        let prefix_bound_plan = plan_tail_pruning(
            &visible,
            &all,
            &HashMap::new(),
            Some("/repo"),
            &TailPruneOptions {
                is_prefix_bound_thinking: true,
                now_ms: Some(base_ts + 60_000),
                ..Default::default()
            },
        );
        assert!(prefix_bound_plan.items.is_empty());

        // 3. 闲置 89 分钟（< 90m）：仍按热缓存处理，不触发全段清仓
        let almost_idle_plan = plan_tail_pruning(
            &visible,
            &all,
            &HashMap::new(),
            Some("/repo"),
            &TailPruneOptions {
                now_ms: Some(base_ts + 9 + (89 * 60 * 1_000)),
                ..Default::default()
            },
        );
        assert!(!almost_idle_plan.idle_flushed);
        assert_eq!(
            almost_idle_plan
                .items
                .iter()
                .map(|i| i.target_seq)
                .collect::<Vec<_>>(),
            vec![12]
        );

        // 4. 闲置 >= 90 分钟：清仓最新压缩边界之后的全部可见候选（seq 7 与 12），且不碰压前 seq 2
        let idle_plan = plan_tail_pruning(
            &visible,
            &all,
            &HashMap::new(),
            Some("/repo"),
            &TailPruneOptions {
                is_prefix_bound_thinking: true,
                now_ms: Some(base_ts + 9 + DEFAULT_IDLE_FLUSH_MS),
                ..Default::default()
            },
        );
        assert!(idle_plan.idle_flushed);
        assert_eq!(
            idle_plan
                .items
                .iter()
                .map(|i| i.target_seq)
                .collect::<Vec<_>>(),
            vec![7, 12]
        );
    }

    #[test]
    fn prune_session_tail_is_idempotent_and_preserves_non_error_failed_read_semantics() {
        let store = SessionStore::new();
        let s = store.create(None);
        s.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"foo.rs"}"#.into(),
            }],
            replay: None,
            usage: None,
        });
        s.append(SessionEvent::ToolResult {
            tool_call_id: "r1".into(),
            name: "read".into(),
            content: "old content".repeat(50),
            is_error: false,
        });
        s.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"foo.rs"}"#.into(),
            }],
            replay: None,
            usage: None,
        });
        s.append(SessionEvent::ToolResult {
            tool_call_id: "r2".into(),
            name: "read".into(),
            content: "File not found".into(),
            is_error: true,
        });

        // 新的失败 read 不得覆盖旧的成功 read
        let first = prune_session_tail(&s, None, &TailPruneOptions::default());
        assert_eq!(first.pruned_count, 0);

        // 再来一次成功完整 read 后覆盖旧的成功和失败结果
        s.append(SessionEvent::Assistant {
            blocks: vec![ContentBlock::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"foo.rs"}"#.into(),
            }],
            replay: None,
            usage: None,
        });
        s.append(SessionEvent::ToolResult {
            tool_call_id: "r3".into(),
            name: "read".into(),
            content: "fresh content".into(),
            is_error: false,
        });

        let second = prune_session_tail(&s, None, &TailPruneOptions::default());
        assert_eq!(second.pruned_count, 2);

        // 幂等：再次执行不得重复写入 ToolResultPrune
        let third = prune_session_tail(&s, None, &TailPruneOptions::default());
        assert_eq!(third.pruned_count, 0);
    }
}
