//! 统一 `write` 写前读守卫（收口 D5 首读要求 + 压后陈旧整文件覆写拦截）。
//!
//! 挂载在 Cordis 的 `tools/pre-execute` 与 `tools/post-execute` waterfall 上，零侵入 `mizar-tools-coding`：
//! 1. **新建与小文件放行**：目标文件尚不存在，或已存在文件行数 `<= 50` 时允许直接 `write`；
//! 2. **已存在且超过 50 行的文件**：本会话尚未成功 `read` 过、或最近一次有效 `read`/`write`
//!    已被压缩出当前上下文（`seq < compacted_up_to_seq`）时，拦截整文件 `write` 并提示先 `read` 或改用 `edit`；
//! 3. **`edit` 与嵌套调用放行**：`edit` 自带精确上下文锚点匹配，始终放行（且不刷新整文件 `read` 状态）。

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mizar_llm::ContentBlock;
use mizar_session::{Session, SessionEvent, SessionStore};
use mizar_tools::{PreExecuteDecision, ToolExecution, ToolResult};
use serde_json::Value;

const WRITE_GUARD_LINE_THRESHOLD: usize = 50;

#[derive(Debug, Default)]
struct SessionFileGuardState {
    /// 是否已从会话历史重建过（覆盖 resume / fork 冷加载场景）。
    initialized: bool,
    /// 每个规范路径最近一次成功 `read` / `write` 的事件序号。
    last_read_seq: HashMap<PathBuf, u64>,
    /// 待与 `EVENT_SESSION_EVENT` 实际落盘序号对齐的近期工具调用。
    pending_calls: HashMap<String, PathBuf>,
}

#[derive(Debug, Default)]
pub struct FileGuardTracker {
    sessions: Mutex<HashMap<String, SessionFileGuardState>>,
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

/// 流式统计文件行数是否严格超过 `threshold`，达到 `threshold + 1` 行时提前返回以保持 O(1) 内存。
fn file_exceeds_line_threshold(path: &Path, threshold: usize) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; 8192];
    let mut newlines = 0usize;
    let mut ends_with_newline = true;
    let mut has_any_bytes = false;

    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                has_any_bytes = true;
                for &b in &buf[..n] {
                    if b == b'\n' {
                        newlines += 1;
                        if newlines > threshold {
                            return true;
                        }
                    }
                }
                ends_with_newline = buf[n - 1] == b'\n';
            }
            Err(_) => return false,
        }
    }

    if !has_any_bytes {
        return false;
    }
    let total_lines = if ends_with_newline {
        newlines
    } else {
        newlines + 1
    };
    total_lines > threshold
}

fn ensure_session_initialized(
    st: &mut SessionFileGuardState,
    session: &Session,
    workspace: Option<&str>,
) {
    if st.initialized {
        return;
    }
    st.initialized = true;

    let mut pending_paths: HashMap<String, PathBuf> = HashMap::new();
    for entry in session.all_entries() {
        match &entry.event {
            SessionEvent::Assistant { blocks, .. } => {
                for block in blocks {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } = block
                        && matches!(name.as_str(), "read" | "write")
                        && let Ok(args) = serde_json::from_str::<Value>(arguments)
                        && let Some(raw_path) = extract_raw_path(&args)
                    {
                        pending_paths.insert(id.clone(), normalize_tool_path(workspace, raw_path));
                    }
                }
            }
            SessionEvent::ToolResult {
                tool_call_id,
                is_error,
                ..
            } => {
                if let Some(canon) = pending_paths.remove(tool_call_id)
                    && !is_error
                {
                    st.last_read_seq
                        .entry(canon)
                        .and_modify(|prev| *prev = (*prev).max(entry.seq))
                        .or_insert(entry.seq);
                }
            }
            _ => {}
        }
    }
}

impl FileGuardTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// 由 `EVENT_SESSION_EVENT` 将工具写入会话后的真实 `Entry.seq` 回填至 `last_read_seq`。
    pub fn reconcile_tool_result_seq(&self, session_id: &str, tool_call_id: &str, actual_seq: u64) {
        if tool_call_id.is_empty() {
            return;
        }
        let mut guard = self.sessions.lock().unwrap();
        let Some(st) = guard.get_mut(session_id) else {
            return;
        };
        let Some(canon) = st.pending_calls.remove(tool_call_id) else {
            return;
        };
        st.last_read_seq
            .entry(canon)
            .and_modify(|prev| *prev = (*prev).max(actual_seq))
            .or_insert(actual_seq);
    }

    /// `tools/pre-execute`：拦截对未读或已被压出上下文的大文件（>50 行）执行整文件 `write`（`edit` 与嵌套调用放行）。
    pub fn check_pre_execute(
        &self,
        exec: &ToolExecution,
        sessions: &SessionStore,
    ) -> PreExecuteDecision {
        if exec.nested || exec.name != "write" {
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

        let header = session.header();
        let ws = header.cwd.clone().or_else(|| sessions.default_cwd());
        let canon = normalize_tool_path(ws.as_deref(), raw_path);
        // 新建不存在的文件及 `<= 50` 行的小文件始终放行，只保护已存在大文件的整文件覆写
        if !canon.is_file() || !file_exceeds_line_threshold(&canon, WRITE_GUARD_LINE_THRESHOLD) {
            return PreExecuteDecision::allow();
        }

        let compacted_up_to = header.compacted_up_to_seq;
        let mut guard = self.sessions.lock().unwrap();
        let st = guard.entry(sid.to_string()).or_default();
        ensure_session_initialized(st, &session, ws.as_deref());

        match st.last_read_seq.get(&canon).copied() {
            None => PreExecuteDecision::deny(format!(
                "Refused whole-file overwrite on '{raw_path}' (existing file exceeds {WRITE_GUARD_LINE_THRESHOLD} lines): this file has not been read in the current session. Please call `read` on '{raw_path}' first before using `write`, or use `edit` for targeted replacements."
            )),
            Some(last_seq) if compacted_up_to.is_some_and(|c| last_seq < c) => {
                PreExecuteDecision::deny(format!(
                    "Refused whole-file overwrite on '{raw_path}': your earlier read (seq #{last_seq}) was compacted out of context (compacted_up_to_seq #{}). Please call `read` on '{raw_path}' first before using `write`, or use `edit` for targeted replacements.",
                    compacted_up_to.unwrap_or(0)
                ))
            }
            Some(_) => PreExecuteDecision::allow(),
        }
    }

    /// `tools/post-execute`：记录成功 `read`/`write` 的文件可见序号，确保同代次写守卫识别最新读取状态。
    pub fn handle_post_execute(
        &self,
        exec: &ToolExecution,
        result: ToolResult,
        sessions: &SessionStore,
    ) -> ToolResult {
        if exec.nested
            || result.is_error
            || result.blocked
            || !matches!(exec.name.as_str(), "read" | "write")
        {
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

        let header = session.header();
        let ws = header.cwd.clone().or_else(|| sessions.default_cwd());
        let canon = normalize_tool_path(ws.as_deref(), raw_path);
        let approx_seq = session.last_seq().saturating_add(1);

        let mut guard = self.sessions.lock().unwrap();
        let st = guard.entry(sid.to_string()).or_default();
        ensure_session_initialized(st, &session, ws.as_deref());
        st.last_read_seq.insert(canon.clone(), approx_seq);

        if !exec.call_id.is_empty() {
            if st.pending_calls.len() >= 128 {
                st.pending_calls.clear();
            }
            st.pending_calls.insert(exec.call_id.clone(), canon);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_threshold_distinguishes_50_vs_51_lines() {
        let dir = std::env::temp_dir().join(format!("mizar_guards_lines_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let p50 = dir.join("exact50.txt");
        let p50_no_nl = dir.join("exact50_nonl.txt");
        let p51 = dir.join("over51.txt");
        let p_empty = dir.join("empty.txt");

        std::fs::write(&p50, "line\n".repeat(50)).unwrap();
        let mut s50 = "line\n".repeat(49);
        s50.push_str("line50");
        std::fs::write(&p50_no_nl, s50).unwrap();
        let mut s51 = "line\n".repeat(50);
        s51.push_str("line51");
        std::fs::write(&p51, s51).unwrap();
        std::fs::write(&p_empty, "").unwrap();

        assert!(!file_exceeds_line_threshold(&p_empty, 50));
        assert!(!file_exceeds_line_threshold(&p50, 50));
        assert!(!file_exceeds_line_threshold(&p50_no_nl, 50));
        assert!(file_exceeds_line_threshold(&p51, 50));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
