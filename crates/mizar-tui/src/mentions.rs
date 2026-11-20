//! 提交时 `@` 文件/目录引用展开与 `agent/pre-step` 钩子。
//!
//! 保留用户输入原文在 `SessionEvent::User` 中（气泡只渲染用户所写），
//! 同时把解析得到的 `<attached-file ...>` 与 `<attached-directory ...>` 内容块
//! 以 `SessionEvent::ContextNotice { source: "file-mentions", .. }` 注入本轮模型上下文。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use cordis::{Context, CordisError, DisposableHandle};
use mizar_agent_loop::{AgentPreStepEvent, EVENT_AGENT_PRE_STEP};
use mizar_session::SessionEvent;
use mizar_tools_coding::path_utils::resolve_read_path;
use mizar_tui_ui::app::{NoticeLevel, UiEvent};
use mizar_tui_ui::mentions::{FILE_MENTIONS_SOURCE, extract_mentions};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// 单文件最大附着字符数（防止误 `@` 大文件挤爆上下文）。
pub const MENTION_MAX_FILE_CHARS: usize = 50_000;
/// 单条消息所有 `@` 附件共享的总字符预算。
pub const MENTION_MAX_TOTAL_CHARS: usize = 200_000;
/// 目录列举最大条目数。
pub const MENTION_MAX_DIR_ENTRIES: usize = 200;
/// 单文件读取上限（16 MiB），避免超大二进制文件引发内存膨胀。
const MENTION_MAX_READ_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MentionExpansionResult {
    pub blocks: Vec<String>,
    pub attached: Vec<String>,
    pub missing: Vec<String>,
}

fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn truncate_chars(text: &str, max_chars: usize) -> (&str, bool) {
    if let Some((byte_idx, _)) = text.char_indices().nth(max_chars) {
        (&text[..byte_idx], true)
    } else {
        (text, false)
    }
}

async fn try_resolve_mention(
    workspace: &Path,
    raw: &str,
    restrict_to_workspace: bool,
) -> Option<(PathBuf, std::fs::Metadata)> {
    let path = resolve_read_path(workspace, raw, restrict_to_workspace)
        .await
        .ok()?;
    let meta = tokio::fs::metadata(&path).await.ok()?;
    Some((path, meta))
}

/// 解析用户文本中的所有 `@` 引用，按出现顺序生成 `<attached-file>` / `<attached-directory>` 块。
pub async fn expand_text_mentions(
    workspace: &Path,
    restrict_to_workspace: bool,
    text: &str,
    cancel: &CancellationToken,
) -> MentionExpansionResult {
    let mentions = extract_mentions(text);
    let mut out = MentionExpansionResult::default();
    if mentions.is_empty() {
        return out;
    }

    let mut budget = MENTION_MAX_TOTAL_CHARS;

    for mention in mentions {
        if cancel.is_cancelled() || budget == 0 {
            break;
        }
        let display = mention
            .literal
            .as_deref()
            .unwrap_or(&mention.path)
            .to_string();

        // strip-first + literal-fallback（兼容文件名本身含 `#L...` 的文件）
        let mut resolved =
            try_resolve_mention(workspace, &mention.path, restrict_to_workspace).await;
        let mut literal_fallback = false;
        if resolved.is_none()
            && let Some(lit) = &mention.literal
        {
            resolved = try_resolve_mention(workspace, lit, restrict_to_workspace).await;
            literal_fallback = resolved.is_some();
        }

        let Some((target, meta)) = resolved else {
            out.missing.push(display);
            continue;
        };

        let shown_path = if literal_fallback {
            display.clone()
        } else if mention.path.starts_with('~') {
            target.display().to_string()
        } else {
            mention.path.clone()
        };
        let shown_attr = escape_xml_attr(&shown_path);

        if meta.is_file() {
            if meta.len() > MENTION_MAX_READ_BYTES {
                out.missing.push(display);
                continue;
            }
            let Ok(bytes) = tokio::fs::read(&target).await else {
                out.missing.push(display);
                continue;
            };
            let Ok(raw_text) = String::from_utf8(bytes) else {
                out.missing.push(display);
                continue;
            };
            let clean_text = raw_text.strip_prefix('\u{FEFF}').unwrap_or(&raw_text);

            let mut header = format!("<attached-file path=\"{shown_attr}\">");
            let body_owned: String;
            let body_slice: &str = if let (Some(start_line), Some(end_line)) =
                (mention.start_line, mention.end_line)
                && !literal_fallback
            {
                let lines: Vec<&str> = clean_text
                    .split('\n')
                    .map(|l| l.strip_suffix('\r').unwrap_or(l))
                    .collect();
                if start_line > lines.len() {
                    header = format!(
                        "<attached-file path=\"{shown_attr}\" lines=\"{start_line}-{end_line}\" note=\"requested lines beyond EOF (file has {} line{}); whole file attached\">",
                        lines.len(),
                        if lines.len() == 1 { "" } else { "s" }
                    );
                    clean_text
                } else {
                    let clamped_end = end_line.min(lines.len());
                    let range_attr = if clamped_end == start_line {
                        format!("{start_line}")
                    } else {
                        format!("{start_line}-{clamped_end}")
                    };
                    header =
                        format!("<attached-file path=\"{shown_attr}\" lines=\"{range_attr}\">");
                    body_owned = lines[start_line - 1..clamped_end].join("\n");
                    &body_owned
                }
            } else {
                clean_text
            };

            let cap = MENTION_MAX_FILE_CHARS.min(budget);
            let (clipped, truncated) = truncate_chars(body_slice, cap);
            let used = clipped.chars().count();
            budget = budget.saturating_sub(used);
            let suffix = if truncated { "\n[… truncated]" } else { "" };
            out.blocks
                .push(format!("{header}\n{clipped}{suffix}\n</attached-file>"));
            out.attached.push(display);
            continue;
        }

        if meta.is_dir() {
            let Ok(mut rd) = tokio::fs::read_dir(&target).await else {
                out.missing.push(display);
                continue;
            };
            let mut entries = Vec::new();
            while let Ok(Some(entry)) = rd.next_entry().await {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name == ".git" || name == ".DS_Store" {
                    continue;
                }
                let is_dir = entry
                    .file_type()
                    .await
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false);
                entries.push(if is_dir { format!("{name}/") } else { name });
            }
            entries.sort();
            let total = entries.len();
            let mut listing: Vec<String> =
                entries.into_iter().take(MENTION_MAX_DIR_ENTRIES).collect();
            if total > MENTION_MAX_DIR_ENTRIES {
                listing.push(format!("… ({} more)", total - MENTION_MAX_DIR_ENTRIES));
            }
            let raw_listing = listing.join("\n");
            let (clipped, truncated) = truncate_chars(&raw_listing, budget);
            let used = clipped.chars().count();
            budget = budget.saturating_sub(used);
            let suffix = if truncated { "\n[… truncated]" } else { "" };
            out.blocks.push(format!(
                "<attached-directory path=\"{shown_attr}\">\n{clipped}{suffix}\n</attached-directory>"
            ));
            out.attached.push(display);
            continue;
        }

        out.missing.push(display);
    }

    out
}

/// 挂载 `agent/pre-step` 监听器：在每轮首次请求模型前，自动展开最新用户消息中的 `@` 引用。
pub fn register_mention_hook(
    ctx: &Context,
    workspace: PathBuf,
    restrict_to_workspace: bool,
    ui_tx: mpsc::UnboundedSender<UiEvent>,
) -> Result<DisposableHandle, CordisError> {
    let processed_seq_by_session: Arc<Mutex<HashMap<String, u64>>> =
        Arc::new(Mutex::new(HashMap::new()));

    ctx.on_serial::<AgentPreStepEvent, (), _, _>(
        EVENT_AGENT_PRE_STEP,
        move |ev: AgentPreStepEvent| {
            let ws = workspace.clone();
            let ui_tx = ui_tx.clone();
            let processed = Arc::clone(&processed_seq_by_session);
            async move {
                if ev.cancel.is_cancelled() {
                    return None;
                }

                let sid = ev.session.id.clone();
                let last_done_seq = processed.lock().unwrap().get(&sid).copied().unwrap_or(0);

                let trailing_users: Vec<(u64, String)> =
                    ev.session.with_visible_entries(|entries| {
                        let mut collected = Vec::new();
                        for entry in entries.iter().rev() {
                            if entry.seq <= last_done_seq {
                                break;
                            }
                            match &entry.event {
                                SessionEvent::User { text } => {
                                    collected.push((entry.seq, text.clone()))
                                }
                                SessionEvent::ContextNotice { source, .. }
                                    if source != FILE_MENTIONS_SOURCE => {}
                                SessionEvent::ContextNotice { .. }
                                | SessionEvent::Assistant { .. }
                                | SessionEvent::ToolResult { .. }
                                | SessionEvent::Compaction { .. }
                                | SessionEvent::ToolResultPrune { .. }
                                | SessionEvent::BashExecution { .. }
                                | SessionEvent::Selection { .. }
                                | SessionEvent::Trace { .. } => break,
                            }
                        }
                        collected.reverse();
                        collected
                    });

                if trailing_users.is_empty() {
                    return None;
                }

                let max_seq = trailing_users.iter().map(|(s, _)| *s).max().unwrap_or(0);
                let combined_text = trailing_users
                    .iter()
                    .map(|(_, t)| t.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let result =
                    expand_text_mentions(&ws, restrict_to_workspace, &combined_text, &ev.cancel)
                        .await;

                if ev.cancel.is_cancelled() {
                    return None;
                }

                processed.lock().unwrap().insert(sid, max_seq);

                if !result.blocks.is_empty() {
                    ev.session.append(SessionEvent::ContextNotice {
                        source: FILE_MENTIONS_SOURCE.to_string(),
                        text: result.blocks.join("\n\n"),
                    });
                }

                if !result.attached.is_empty() {
                    let _ = ui_tx.send(UiEvent::Flash(format!(
                        "✓ 已附加 {} 个 @ 引用",
                        result.attached.len()
                    )));
                }
                if !result.missing.is_empty() {
                    let _ = ui_tx.send(UiEvent::Notice {
                        level: NoticeLevel::Warn,
                        text: format!("未找到或无法读取 @ 引用：{}", result.missing.join(", ")),
                    });
                }

                None
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expands_files_line_ranges_directories_and_missing_mentions() {
        let tmp =
            std::env::temp_dir().join(format!("mizar_tui_mention_expand_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("src/sub")).unwrap();
        std::fs::write(tmp.join("src/a.rs"), "line1\nline2\nline3\nline4\nline5\n").unwrap();
        std::fs::write(tmp.join("src/sub/b.rs"), "pub fn b() {}\n").unwrap();
        std::fs::write(tmp.join("spaced file.txt"), "hello spaces").unwrap();
        std::fs::write(tmp.join("literal#L2.txt"), "literal file").unwrap();

        let res = expand_text_mentions(
            &tmp,
            true,
            "看 @src/a.rs#L2-4 和 @\"spaced file.txt\" 以及 @src/sub 和 @literal#L2.txt 和 @missing.rs",
            &CancellationToken::new(),
        )
        .await;

        assert_eq!(res.attached.len(), 4);
        assert_eq!(res.missing, vec!["missing.rs".to_string()]);
        assert!(res.blocks[0].contains("lines=\"2-4\""));
        assert!(res.blocks[0].contains("line2\nline3\nline4"));
        assert!(!res.blocks[0].contains("line1"));
        assert!(res.blocks[1].contains("hello spaces"));
        assert!(res.blocks[2].contains("<attached-directory path=\"src/sub\">"));
        assert!(res.blocks[2].contains("b.rs"));
        assert!(res.blocks[3].contains("literal file"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
