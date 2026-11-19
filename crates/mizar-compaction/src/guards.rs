//! 压后工具安全守卫与跨压缩感知读取去重（综合 `dim` `FileReads` + `dsh-ops` `read-dedup`）。
//!
//! 挂载在 Cordis 的 `tools/pre-execute` 与 `tools/post-execute` waterfall 上，零侵入 `mizar-tools-coding`：
//! 1. **压后整文件覆写拦截（`FileReads`）**：若 `write` 覆盖已存在文件，且该文件此前在压缩前曾被 `read` 过、
//!    但压缩后尚未重新 `read` / `write`，则拦截并提示先重读；`edit` 因自带精确 `old_text` 锚点始终放行。
//! 2. **跨压缩感知 `read-dedup`**：每次 `read` 仍真实读盘，但若同一 `compacted_up_to_seq` 代次内可见上下文中
//!    已存在一模一样的原文且未被剪枝，则输出极短去重提示，大幅减少重复文件 token。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mizar_session::SessionStore;
use mizar_tools::{PreExecuteDecision, ToolExecution, ToolResult};
use serde_json::Value;

const MIN_DEDUP_BYTES: usize = 512;
const MAX_DEDUP_ENTRIES_PER_SESSION: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReadRangeKey {
    path: PathBuf,
    offset: u64,
    limit: Option<u64>,
}

#[derive(Debug, Clone)]
struct ReadDedupRecord {
    content_hash: u64,
    compaction_epoch: Option<u64>,
    approx_seq: u64,
    raw_path: String,
    start_line: u64,
    end_line: u64,
}

#[derive(Debug, Default)]
struct SessionFileGuardState {
    /// 每个规范路径最近一次成功 `read` / `write` 的事件序号。
    last_read_seq: HashMap<PathBuf, u64>,
    /// 同一压缩代次内的重复读去重记录。
    dedup: HashMap<ReadRangeKey, ReadDedupRecord>,
}

#[derive(Debug, Default)]
pub struct FileGuardTracker {
    sessions: Mutex<HashMap<String, SessionFileGuardState>>,
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn normalize_tool_path(workspace: Option<&str>, raw_path: &str) -> PathBuf {
    let p = Path::new(raw_path);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else if let Some(ws) = workspace {
        Path::new(ws).join(p)
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(p)
    };
    std::fs::canonicalize(&abs).unwrap_or(abs)
}

fn extract_raw_path(args: &Value) -> Option<&str> {
    args.get("path")
        .or_else(|| args.get("file_path"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

impl FileGuardTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// `tools/pre-execute`：压后拦截对已被压出上下文的已存在文件执行整文件 `write`（`edit` 始终放行）。
    pub fn check_pre_execute(
        &self,
        exec: &ToolExecution,
        sessions: &SessionStore,
        enable_write_guard: bool,
    ) -> PreExecuteDecision {
        if !enable_write_guard || exec.name != "write" {
            return PreExecuteDecision::allow();
        }
        let Some(sid) = exec.session_id.as_deref() else {
            return PreExecuteDecision::allow();
        };
        let Some(raw_path) = extract_raw_path(&exec.args) else {
            return PreExecuteDecision::allow();
        };
        let Some(session) = sessions.get(sid) else {
            return PreExecuteDecision::allow();
        };
        let Some(compacted_up_to) = session.header().compacted_up_to_seq else {
            return PreExecuteDecision::allow();
        };

        let ws = session.header().cwd.or_else(|| sessions.default_cwd());
        let canon = normalize_tool_path(ws.as_deref(), raw_path);
        // 新建不存在的文件始终放行，只保护已存在文件的整文件覆写
        if !canon.exists() {
            return PreExecuteDecision::allow();
        }

        let guard = self.sessions.lock().unwrap();
        let Some(st) = guard.get(sid) else {
            return PreExecuteDecision::allow();
        };
        if let Some(&last_seq) = st.last_read_seq.get(&canon)
            && last_seq < compacted_up_to
        {
            return PreExecuteDecision::deny(format!(
                "Refused whole-file overwrite on '{raw_path}': your earlier read (seq #{last_seq}) was compacted out of context (compacted_up_to_seq #{compacted_up_to}). Please call `read` on '{raw_path}' first before using `write`, or use `edit` for targeted replacements."
            ));
        }

        PreExecuteDecision::allow()
    }

    /// `tools/post-execute`：记录 `read`/`write`/`edit` 状态并执行跨压缩感知 `read-dedup`。
    pub fn handle_post_execute(
        &self,
        exec: &ToolExecution,
        mut result: ToolResult,
        sessions: &SessionStore,
        enable_read_dedup: bool,
    ) -> ToolResult {
        if result.is_error || result.blocked {
            return result;
        }
        let Some(sid) = exec.session_id.as_deref() else {
            return result;
        };
        let Some(raw_path) = extract_raw_path(&exec.args) else {
            return result;
        };
        let Some(session) = sessions.get(sid) else {
            return result;
        };

        let ws = session.header().cwd.or_else(|| sessions.default_cwd());
        let canon = normalize_tool_path(ws.as_deref(), raw_path);
        let approx_seq = session.last_seq().saturating_add(1);
        let compaction_epoch = session.header().compacted_up_to_seq;

        match exec.name.as_str() {
            "write" | "edit" => {
                let mut guard = self.sessions.lock().unwrap();
                let st = guard.entry(sid.to_string()).or_default();
                // 成功覆写/编辑后，该路径在当前上下文中处于已知状态，同时失效所有旧 read-dedup 缓存
                st.last_read_seq.insert(canon.clone(), approx_seq);
                st.dedup.retain(|k, _| k.path != canon);
                result
            }
            "read" => {
                let offset = exec
                    .args
                    .get("offset")
                    .and_then(Value::as_u64)
                    .map(|v| v.max(1))
                    .unwrap_or(1);
                let limit = exec
                    .args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| v.max(1));
                let key = ReadRangeKey {
                    path: canon.clone(),
                    offset,
                    limit,
                };
                let content_hash = fnv1a_64(result.content.as_bytes());
                let pruned_map = session.pruned_map();

                let mut guard = self.sessions.lock().unwrap();
                let st = guard.entry(sid.to_string()).or_default();

                if enable_read_dedup
                    && result.content.len() >= MIN_DEDUP_BYTES
                    && let Some(prev) = st.dedup.get(&key)
                {
                    let same_epoch = prev.compaction_epoch == compaction_epoch;
                    let still_visible = prev.approx_seq >= compaction_epoch.unwrap_or(0);
                    let not_pruned = !pruned_map.contains_key(&prev.approx_seq);
                    if same_epoch
                        && still_visible
                        && not_pruned
                        && prev.content_hash == content_hash
                    {
                        // 注意：去重命中时并未重新向模型吐出全文，绝不刷新 `last_read_seq`，
                        // 确保真正承载正文的那条旧 `read` 一旦被压出上下文，`FileReads` 守卫仍能准确拦截陈旧覆写。
                        result.content = format!(
                            "[Unchanged file range: '{}' lines {}-{} is already present verbatim in current context (around seq #{}). Re-read omitted to save context.]",
                            prev.raw_path, prev.start_line, prev.end_line, prev.approx_seq
                        );
                        return result;
                    }
                }

                st.last_read_seq.insert(canon, approx_seq);

                let start_line = result
                    .structured_content
                    .as_ref()
                    .and_then(|v| v.get("start_line"))
                    .and_then(Value::as_u64)
                    .unwrap_or(offset);
                let end_line = result
                    .structured_content
                    .as_ref()
                    .and_then(|v| v.get("end_line"))
                    .and_then(Value::as_u64)
                    .unwrap_or(start_line);

                if st.dedup.len() >= MAX_DEDUP_ENTRIES_PER_SESSION {
                    st.dedup.clear();
                }
                st.dedup.insert(
                    key,
                    ReadDedupRecord {
                        content_hash,
                        compaction_epoch,
                        approx_seq,
                        raw_path: raw_path.to_string(),
                        start_line,
                        end_line,
                    },
                );
                result
            }
            _ => result,
        }
    }

    /// 当会话发生压缩时清理旧的 `read-dedup` 缓存，保证压后首次重读吐出完整正文。
    pub fn on_compaction(&self, session_id: &str) {
        let mut guard = self.sessions.lock().unwrap();
        if let Some(st) = guard.get_mut(session_id) {
            st.dedup.clear();
        }
    }
}
