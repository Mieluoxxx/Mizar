//! 上下文预算公式、CJK/代码混合 token 估算与真实 `TokenUsage` 锚点校准。
//!
//! 统一基于模型真实可用窗口（`context_window - reserved_output`）与可选 `max_context_tokens`
//! 推导触发阈值，并以固定的 `keep_recent_tokens`（小窗自动钳位）切分保留尾部；
//! 实测校准锚点使用 `u.prompt_tokens()`（已含 `cache_read` 与 `cache_write`）。

use std::collections::HashMap;

use mizar_llm::{ContentBlock, ImageAttachment, Message};
use mizar_session::{Session, SessionEntry, SessionEvent};

use crate::CompactionConfig;

/// 根据模型窗口与配置推导出的单次上下文预算阈值表。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub context_window: u64,
    pub reserved_output: u64,
    pub usable_tokens: u64,
    pub effective_window: u64,
    pub trigger_tokens: u64,
    pub keep_recent_tokens: u64,
}

pub fn compute_budget(
    context_window: u64,
    max_output: Option<u32>,
    config: &CompactionConfig,
) -> ContextBudget {
    let window = context_window.max(4_096);
    let declared_out = u64::from(max_output.unwrap_or(16_384));
    let reserved_output = declared_out.min(131_072).min(window / 2);
    let usable_tokens = window.saturating_sub(reserved_output).max(2_048);
    let effective_window = config
        .max_context_tokens
        .filter(|&cap| cap > 0)
        .map_or(usable_tokens, |cap| usable_tokens.min(cap).max(2_048));

    let trigger_ratio = config.trigger_ratio.clamp(0.20, 0.95);
    let trigger_tokens = (((effective_window as f64) * trigger_ratio).floor() as u64).max(1_024);
    let keep_recent_tokens = config.keep_recent_tokens.min(effective_window / 4);

    ContextBudget {
        context_window: window,
        reserved_output,
        usable_tokens,
        effective_window,
        trigger_tokens,
        keep_recent_tokens,
    }
}

/// 兼顾中文/非 ASCII 字符与英文代码的轻量 token 估算：
/// - 每个非 ASCII 字符约计 1 token（中文单字在常见 tokenizer 中约 0.7~1.3 tok）
/// - ASCII 字节按 `bytes / 4` 估算。
pub fn estimate_text_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut non_ascii_chars = 0u64;
    let mut ascii_bytes = 0u64;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_bytes += 1;
        } else {
            non_ascii_chars += 1;
        }
    }
    non_ascii_chars + ascii_bytes.div_ceil(4)
}

/// 估算单张图片占据的上下文 token（按像素面积近似，不直接将 base64 字符长度除以 4）。
pub fn estimate_image_tokens(image: &ImageAttachment) -> u64 {
    match (image.width, image.height) {
        (Some(w), Some(h)) if w > 0 && h > 0 => {
            ((u64::from(w) * u64::from(h)).div_ceil(750)).max(85)
        }
        _ => (image.data.len() as u64 / 512).max(1_024),
    }
}

pub fn estimate_message_tokens(msg: &Message) -> u64 {
    let base = 8u64;
    match msg {
        Message::System { content } => base + estimate_text_tokens(content),
        Message::User { content, images } => {
            let img_tokens: u64 = images.iter().map(estimate_image_tokens).sum();
            base + estimate_text_tokens(content) + img_tokens
        }
        Message::Assistant { blocks, .. } => {
            let mut sum = base;
            for b in blocks {
                match b {
                    ContentBlock::Text { text } | ContentBlock::Reasoning { text } => {
                        sum += estimate_text_tokens(text);
                    }
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } => {
                        sum += 12 + estimate_text_tokens(name) + estimate_text_tokens(arguments);
                    }
                }
            }
            sum
        }
        Message::Tool { content, .. } => base + 8 + estimate_text_tokens(content),
    }
}

pub fn estimate_entry_tokens(entry: &SessionEntry) -> u64 {
    let base = 8u64;
    match &entry.event {
        SessionEvent::User { text, images } | SessionEvent::ContextNotice { text, images, .. } => {
            let img_tokens: u64 = images.iter().map(estimate_image_tokens).sum();
            base + estimate_text_tokens(text) + img_tokens
        }
        SessionEvent::Assistant { blocks, .. } => {
            let mut sum = base;
            for b in blocks {
                match b {
                    ContentBlock::Text { text } => sum += estimate_text_tokens(text),
                    // 推理块在多协议中通常不重放或仅占极小开销；保守按 1/4 计入避免低估
                    ContentBlock::Reasoning { text } => sum += estimate_text_tokens(text) / 4,
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } => {
                        sum += 12 + estimate_text_tokens(name) + estimate_text_tokens(arguments);
                    }
                }
            }
            sum
        }
        SessionEvent::ToolResult { content, .. } => base + 8 + estimate_text_tokens(content),
        SessionEvent::Compaction {
            summary, digest, ..
        } => {
            let rendered = mizar_session::render_compaction_message(summary, digest);
            base + estimate_text_tokens(&rendered)
        }
        SessionEvent::BashExecution {
            command,
            output,
            exclude_from_context,
            ..
        } => {
            if *exclude_from_context {
                0
            } else {
                base + estimate_text_tokens(command) + estimate_text_tokens(output)
            }
        }
        SessionEvent::ToolResultPrune { .. }
        | SessionEvent::Selection { .. }
        | SessionEvent::Trace { .. } => 0,
    }
}

/// 估算单条会话事件在应用 `ToolResultPrune` 投影替换后的上下文 token 数。
pub fn estimate_entry_tokens_with_pruned(
    entry: &SessionEntry,
    pruned_map: &HashMap<u64, String>,
) -> u64 {
    if let SessionEvent::ToolResult { .. } = &entry.event
        && let Some(notice) = pruned_map.get(&entry.seq)
    {
        return 8 + 8 + estimate_text_tokens(notice);
    }
    estimate_entry_tokens(entry)
}

/// 估算当前会话发往模型的总上下文 token 数：
/// 若最新一次边界变更之后存在有效的实测 `Assistant.usage.prompt_tokens()`，
/// 则以该实测值（含 `cache_read` + `cache_write`）作为锚点，加上锚点之后新增条目的增量估算；
/// 否则全量按估算值求和。
pub fn estimate_session_context_tokens(
    session: &Session,
    system_prompt_text: &str,
    tools_schema_chars: usize,
) -> u64 {
    let visible = session.visible_log();
    let all = session.all_entries();
    let pruned_map = session.pruned_map();

    // 找到最近一次改变前缀形态的边界序号（Compaction 或 ToolResultPrune）
    let last_mutation_seq = all
        .iter()
        .rev()
        .find_map(|e| match e.event {
            SessionEvent::Compaction { .. } | SessionEvent::ToolResultPrune { .. } => Some(e.seq),
            _ => None,
        })
        .unwrap_or(0);

    // 寻找晚于最新前缀变更的最近有效 Assistant 实测 prompt_tokens()
    let mut calibrated_anchor: Option<(u64, u64)> = None;
    for entry in visible.iter().rev() {
        if entry.seq <= last_mutation_seq {
            continue;
        }
        if let SessionEvent::Assistant { usage: Some(u), .. } = &entry.event {
            let prompt_tokens = u.prompt_tokens();
            if prompt_tokens > 0 {
                calibrated_anchor = Some((entry.seq, prompt_tokens));
                break;
            }
        }
    }

    if let Some((anchor_seq, anchor_input)) = calibrated_anchor {
        let mut total = anchor_input;
        for entry in &visible {
            if entry.seq >= anchor_seq {
                total = total.saturating_add(estimate_entry_tokens_with_pruned(entry, &pruned_map));
            }
        }
        return total;
    }

    let mut total =
        estimate_text_tokens(system_prompt_text) + (tools_schema_chars as u64).div_ceil(4);
    for entry in &visible {
        total = total.saturating_add(estimate_entry_tokens_with_pruned(entry, &pruned_map));
    }
    total
}
