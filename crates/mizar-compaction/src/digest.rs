//! 切分规划器与确定性硬事实提取器（借鉴 `dim` 的软硬事实四层隔离 + `pi` Split Turn 保护）。
//!
//! 核心不变量：
//! 1. 绝不拆散 `Assistant(ToolCall)` 与其对应的 `ToolResult` 配对；
//! 2. 已修改文件（`write`/`edit`）与关键命令（`bash`/`BashExecution`）由代码直接从事件流提取，
//!    多轮压缩之间从结构化 `CompactionDigest` 直接滚动，严禁从自然语言摘要反向解析；
//! 3. 若切点落在当前未完成 turn 中途，钉住该 turn 的用户输入原话并附 `[seq #N]` 句柄。

use std::collections::{HashMap, HashSet};

use mizar_llm::ContentBlock;
use mizar_session::{CompactionDigest, PinnedInput, SessionEntry, SessionEvent};
use serde_json::Value;

use crate::budget::estimate_entry_tokens_with_pruned;

const MAX_MODIFIED_FILES: usize = 64;
const MAX_COMMANDS_RUN: usize = 32;
const MAX_PINNED_INPUT_CHARS: usize = 2_000;
const MAX_COMMAND_PREVIEW_CHARS: usize = 140;

#[derive(Debug, Clone)]
pub struct CutPlan {
    /// 上一次已有的压缩摘要与硬事实（若本会话曾压缩过）。
    pub previous_summary: Option<String>,
    pub previous_digest: CompactionDigest,
    /// 本次将被折叠进摘要的历史条目（不含上一次 `Compaction` 自身）。
    pub compacted_entries: Vec<SessionEntry>,
    /// 将被保留在窗口尾部的表面条目。
    pub kept_entries: Vec<SessionEntry>,
    /// 写入新 `Compaction` 节点的 `first_kept_seq`。
    pub first_kept_seq: u64,
    /// 被折叠区的估算 token 数。
    pub compacted_tokens: u64,
    /// 保留尾部的估算 token 数。
    pub kept_tokens: u64,
    /// 若发生 Split Turn，钉住的当前 turn 用户输入。
    pub pinned_inputs: Vec<PinnedInput>,
}

/// 规划压缩切分点：从 `visible` 表面事件尾部倒序累加至 `keep_recent_tokens`，
/// 并严格回退确保不把 `Assistant(ToolCall)` 与 `ToolResult` 切开。
pub fn plan_compaction_cut(
    visible: &[SessionEntry],
    last_seq: u64,
    keep_recent_tokens: u64,
    force_zero_tail: bool,
) -> Option<CutPlan> {
    plan_compaction_cut_with_pruned(
        visible,
        last_seq,
        keep_recent_tokens,
        force_zero_tail,
        &HashMap::new(),
    )
}

/// 规划压缩切分点（支持结合 `ToolResultPrune` 映射按实际投影后 token 估算切分位置）。
pub fn plan_compaction_cut_with_pruned(
    visible: &[SessionEntry],
    last_seq: u64,
    keep_recent_tokens: u64,
    force_zero_tail: bool,
    pruned_map: &HashMap<u64, String>,
) -> Option<CutPlan> {
    let mut previous_summary: Option<String> = None;
    let mut previous_digest = CompactionDigest::default();
    let mut regular: Vec<&SessionEntry> = Vec::with_capacity(visible.len());

    for entry in visible {
        match &entry.event {
            SessionEvent::Compaction {
                summary, digest, ..
            } => {
                previous_summary = Some(summary.clone());
                previous_digest = digest.clone();
            }
            _ => regular.push(entry),
        }
    }

    if regular.is_empty() {
        return None;
    }

    let cut_idx = if force_zero_tail || keep_recent_tokens == 0 {
        // 极端溢出兜底：尽可能只留最后一条合法的用户边界（若最后一条是工具结果则连带保住该工具调用对）
        if regular.len() <= 1 {
            regular.len()
        } else {
            let mut idx = regular.len();
            // 若最后一条是用户消息或刚完成的工具结果，保留至少最后 1 条以便继续当前步
            if matches!(
                regular.last().map(|e| &e.event),
                Some(SessionEvent::User { .. } | SessionEvent::ToolResult { .. })
            ) {
                idx = regular.len() - 1;
                idx = rewind_for_tool_pairing(&regular, idx);
            }
            idx
        }
    } else {
        let mut kept_acc = 0u64;
        let mut idx = regular.len();
        while idx > 0 {
            let entry = regular[idx - 1];
            let tok = estimate_entry_tokens_with_pruned(entry, pruned_map);
            if idx < regular.len() && kept_acc.saturating_add(tok) > keep_recent_tokens {
                break;
            }
            kept_acc = kept_acc.saturating_add(tok);
            idx -= 1;
        }
        rewind_for_tool_pairing(&regular, idx)
    };

    if cut_idx == 0 {
        // 没有任何条目可折叠（说明单个尾部 turn 就超过了保留预算）；若是强制零尾仍尝试折叠除最后配对外内容
        return None;
    }

    let compacted_entries: Vec<SessionEntry> =
        regular[..cut_idx].iter().map(|e| (*e).clone()).collect();
    let kept_entries: Vec<SessionEntry> = regular[cut_idx..].iter().map(|e| (*e).clone()).collect();

    let first_kept_seq = kept_entries
        .first()
        .map_or(last_seq.saturating_add(1), |e| e.seq);

    let compacted_tokens: u64 = compacted_entries
        .iter()
        .map(|e| estimate_entry_tokens_with_pruned(e, pruned_map))
        .sum();
    let kept_tokens: u64 = kept_entries
        .iter()
        .map(|e| estimate_entry_tokens_with_pruned(e, pruned_map))
        .sum();

    let pinned_inputs = detect_split_turn_pinned_inputs(
        &compacted_entries,
        &kept_entries,
        &previous_digest.pinned_inputs,
    );

    Some(CutPlan {
        previous_summary,
        previous_digest,
        compacted_entries,
        kept_entries,
        first_kept_seq,
        compacted_tokens,
        kept_tokens,
        pinned_inputs,
    })
}

fn rewind_for_tool_pairing(regular: &[&SessionEntry], mut idx: usize) -> usize {
    if idx >= regular.len() {
        return idx;
    }
    while idx > 0 && matches!(regular[idx].event, SessionEvent::ContextNotice { .. }) {
        idx -= 1;
    }
    // 并行回执之间可能夹有其他可见事件，仅向前跳过相邻回执会遗失发起调用。
    let mut results = HashMap::new();
    for (i, entry) in regular.iter().enumerate().rev() {
        match &entry.event {
            SessionEvent::ToolResult { tool_call_id, .. } => {
                results.entry(tool_call_id.as_str()).or_insert(i);
            }
            SessionEvent::Assistant { blocks, .. } => {
                for block in blocks {
                    if let ContentBlock::ToolCall { id, .. } = block
                        && results.remove(id.as_str()).is_some_and(|end| end >= idx)
                    {
                        idx = idx.min(i);
                    }
                }
            }
            _ => {}
        }
    }
    idx
}

/// 若保留尾部不是以 `User` 开局（说明当前进行中的 turn 被腰斩），
/// 找出该 turn 对应的最后一条被压掉的 `User` 原文并钉住，防止模型丢失当前正在做的指令。
fn detect_split_turn_pinned_inputs(
    compacted: &[SessionEntry],
    kept: &[SessionEntry],
    previous: &[PinnedInput],
) -> Vec<PinnedInput> {
    if kept.is_empty() {
        return Vec::new();
    }
    if matches!(kept[0].event, SessionEvent::User { .. }) {
        return Vec::new();
    }
    let Some(last_user) = compacted
        .iter()
        .rev()
        .find(|e| matches!(e.event, SessionEvent::User { .. }))
    else {
        // 同一轮再次切分时，原始用户输入可能只存在于上个检查点。
        return previous.to_vec();
    };
    let SessionEvent::User { text, .. } = &last_user.event else {
        return Vec::new();
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let total_chars = trimmed.chars().count();
    let (pinned_text, truncated) = if total_chars > MAX_PINNED_INPUT_CHARS {
        let head: String = trimmed
            .chars()
            .take(MAX_PINNED_INPUT_CHARS * 2 / 3)
            .collect();
        let tail_keep = MAX_PINNED_INPUT_CHARS / 4;
        let tail: String = trimmed
            .chars()
            .skip(total_chars.saturating_sub(tail_keep))
            .collect();
        (format!("{head}\n...\n{tail}"), true)
    } else {
        (trimmed.to_string(), false)
    };
    vec![PinnedInput {
        seq: last_user.seq,
        text: pinned_text,
        truncated,
    }]
}

/// 从上一份 `CompactionDigest` + 本次被折叠的 `compacted_entries` 中确定性提取硬事实。
pub fn build_compaction_digest(
    previous: &CompactionDigest,
    compacted: &[SessionEntry],
    pinned_inputs: Vec<PinnedInput>,
) -> CompactionDigest {
    let mut modified_files: Vec<String> = previous.modified_files.clone();
    let mut modified_set: HashSet<String> = modified_files.iter().cloned().collect();

    let mut commands_run: Vec<String> = previous.commands_run.clone();

    // 收集被折叠区内每个 tool_call_id 的 (tool_name, args_json) 映射
    let mut calls_by_id: HashMap<String, (String, Value)> = HashMap::new();
    for entry in compacted {
        match &entry.event {
            SessionEvent::Assistant { blocks, .. } => {
                for b in blocks {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } = b
                    {
                        let parsed =
                            serde_json::from_str::<Value>(arguments).unwrap_or(Value::Null);
                        calls_by_id.insert(id.clone(), (name.clone(), parsed));
                    }
                }
            }
            SessionEvent::ToolResult {
                tool_call_id,
                name,
                is_error,
                ..
            } => {
                let Some((call_name, args)) = calls_by_id.get(tool_call_id) else {
                    continue;
                };
                let effective_name = if call_name.is_empty() {
                    name.as_str()
                } else {
                    call_name.as_str()
                };
                match effective_name {
                    "write" | "edit" if !*is_error => {
                        if let Some(path) = args
                            .get("path")
                            .or_else(|| args.get("file_path"))
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                        {
                            let p = path.to_string();
                            if modified_set.insert(p.clone()) {
                                modified_files.push(p);
                            }
                        }
                    }
                    "bash" => {
                        if let Some(cmd) = args
                            .get("command")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                        {
                            let line = format_command_digest(cmd, *is_error, None);
                            push_bounded_unique_tail(&mut commands_run, line, MAX_COMMANDS_RUN);
                        }
                    }
                    _ => {}
                }
            }
            SessionEvent::BashExecution {
                command,
                exit_code,
                cancelled,
                exclude_from_context,
                ..
            } if !*exclude_from_context => {
                let is_err = *cancelled || exit_code.is_some_and(|c| c != 0);
                let line = format_command_digest(command, is_err, *exit_code);
                push_bounded_unique_tail(&mut commands_run, line, MAX_COMMANDS_RUN);
            }
            _ => {}
        }
    }

    if modified_files.len() > MAX_MODIFIED_FILES {
        let drop_n = modified_files.len() - MAX_MODIFIED_FILES;
        modified_files.drain(..drop_n);
    }

    CompactionDigest {
        modified_files,
        commands_run,
        pinned_inputs,
    }
}

fn format_command_digest(command: &str, is_error: bool, exit_code: Option<i32>) -> String {
    let first_line = command
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or(command.trim());
    let truncated: String = first_line.chars().take(MAX_COMMAND_PREVIEW_CHARS).collect();
    let suffix = if let Some(code) = exit_code {
        format!(" [exit={code}]")
    } else if is_error {
        " [error]".to_string()
    } else {
        " [ok]".to_string()
    };
    format!("`{truncated}`{suffix}")
}

fn push_bounded_unique_tail(vec: &mut Vec<String>, item: String, max_len: usize) {
    if vec.last().is_some_and(|last| last == &item) {
        return;
    }
    vec.push(item);
    if vec.len() > max_len {
        let drop_n = vec.len() - max_len;
        vec.drain(..drop_n);
    }
}
