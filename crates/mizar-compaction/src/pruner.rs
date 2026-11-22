//! L1 零模型批量剪枝器（综合 `deepseek-harness` `ToolResultPruner` + `billion-context` `viewLatch`）。
//!
//! - 在调用 LLM 摘要前，优先扫描较早轮次的大 `ToolResult`，按 Unicode 字符边界留存首尾预览
//!   并附上 `[seq #N, recall_read]` 句柄；
//! - 保护最近 `prune_protect_recent_surface` 条工作区表面事件不动；
//! - 一次性追加单条 `SessionEvent::ToolResultPrune` 锁存，不破坏原始事件且避免步步抖动 KV Cache。

use mizar_session::{PrunedToolResult, Session, SessionEvent};

use crate::CompactionConfig;
use crate::budget::estimate_text_tokens;

#[derive(Debug, Clone)]
pub struct PrunePlan {
    pub items: Vec<PrunedToolResult>,
    pub estimated_tokens_saved: u64,
}

fn take_head_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

fn take_tail_chars(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    s.chars().skip(total - max_chars).collect()
}

/// 规划一批零模型剪枝条目（只读计算，不修改会话）。
pub fn plan_tool_result_prune(session: &Session, config: &CompactionConfig) -> PrunePlan {
    plan_tool_result_prune_with_pending(session, config, 0)
}

/// 规划一批零模型剪枝条目，将当前步待提交 `TurnInput` 条数计入最近表面保护窗口，
/// 避免 `pre-step` 提前到提交前运行导致最近条目保护名额少算当前轮输入。
pub fn plan_tool_result_prune_with_pending(
    session: &Session,
    config: &CompactionConfig,
    pending_surface_count: usize,
) -> PrunePlan {
    let visible = session.visible_log();
    let already_pruned = session.pruned_map();
    let protect = config
        .prune_protect_recent_surface
        .saturating_sub(pending_surface_count);
    if visible.len() <= protect {
        return PrunePlan {
            items: Vec::new(),
            estimated_tokens_saved: 0,
        };
    }

    let cutoff_idx = visible.len() - protect;
    let threshold = config.prune_threshold_chars.max(1_024);
    let head_chars = config.prune_head_chars.min(threshold / 2);
    let tail_chars = config.prune_tail_chars.min(threshold / 3);

    let mut items = Vec::new();
    let mut saved_tokens = 0u64;

    for entry in &visible[..cutoff_idx] {
        if already_pruned.contains_key(&entry.seq) {
            continue;
        }
        let SessionEvent::ToolResult { name, content, .. } = &entry.event else {
            continue;
        };
        let char_count = content.chars().count();
        if char_count <= threshold {
            continue;
        }
        let omitted = char_count.saturating_sub(head_chars + tail_chars);
        if omitted < 256 {
            continue;
        }

        let head = take_head_chars(content, head_chars);
        let tail = take_tail_chars(content, tail_chars);
        let replacement = format!(
            "{head}\n\n[... pruned {omitted} chars from `{name}` output (original {char_count} chars); full verbatim text preserved at seq #{}, call recall_read(seq={}) if needed ...]\n\n{tail}",
            entry.seq, entry.seq
        );

        let before_tok = estimate_text_tokens(content);
        let after_tok = estimate_text_tokens(&replacement);
        if after_tok >= before_tok {
            continue;
        }

        saved_tokens = saved_tokens.saturating_add(before_tok - after_tok);
        items.push(PrunedToolResult {
            target_seq: entry.seq,
            replacement,
            original_chars: char_count,
        });
    }

    PrunePlan {
        items,
        estimated_tokens_saved: saved_tokens,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cordis::Context;
    use mizar_session::Session;

    use super::*;

    #[test]
    fn prunes_old_large_tool_results_and_protects_recent_tail() {
        let ctx = Context::new();
        let s = Arc::new(Session::new(ctx, "prune-1"));
        s.append(SessionEvent::user("inspect big file"));
        s.append(SessionEvent::ToolResult {
            tool_call_id: "c1".into(),
            name: "read".into(),
            content: "a".repeat(12_000),
            is_error: false,
        });
        // 保护尾部的 4 条表面消息
        for i in 0..4 {
            s.append(SessionEvent::user(format!("follow-up {i}")));
        }
        // 最新一条也是超大 ToolResult，但在保护尾区内不应被剪
        s.append(SessionEvent::ToolResult {
            tool_call_id: "c2".into(),
            name: "read".into(),
            content: "b".repeat(12_000),
            is_error: false,
        });

        let cfg = CompactionConfig::default();
        let plan = plan_tool_result_prune(&s, &cfg);
        assert_eq!(plan.items.len(), 1);
        assert_eq!(plan.items[0].target_seq, 2);
        assert!(plan.items[0].replacement.contains("recall_read(seq=2)"));
        assert!(plan.estimated_tokens_saved > 1_000);

        // 锁存后再次规划应幂等返回空
        s.append_prune(plan.items);
        let plan2 = plan_tool_result_prune(&s, &cfg);
        assert!(plan2.items.is_empty());
    }
}
