//! LLM 检查点摘要生成器（同前缀尾置指令、`ReasoningEffort::Off` + 错误放宽回退、
//! `obelisk` 已证实事实/未证实推断分级）。

use futures::StreamExt;
use mizar_llm::{
    BlockAssembler, ContentBlock, FinishReason, GenerateOptions, LlmError, LlmErrorCode,
    LlmRuntime, Message, ReasoningEffort, StreamChunk, ToolSchema,
};
use mizar_session::{SessionEntry, SessionEvent};
use tokio_util::sync::CancellationToken;

use crate::digest::CutPlan;

const TRANSCRIPT_TOOL_RESULT_MAX_CHARS: usize = 2_000;
const TRANSCRIPT_TOTAL_MAX_CHARS: usize = 96_000;
const BOUNDARY_PREVIEW_MAX_CHARS: usize = 120;

const SUMMARY_SECTIONS_GUIDE: &str = "\
Write a concise, high-signal checkpoint summary with these exact sections:
## Goal & Scope
- What the user is trying to achieve and key constraints.

## Verified Facts & Decisions
- ONLY facts directly verified by tool output, code read, test results, or explicit user confirmation.

## Hypotheses & Open Questions
- Unverified inferences, suspected root causes, or unresolved ambiguities. NEVER present guesses as settled facts.

## Current Progress & Next Steps
- Completed work, in-progress items, and the immediate next step.

## Recovery Handles
- Important file paths, symbol names, or `[seq #N]` handles for `recall` if verbatim detail is later needed.";

pub struct SummarizeParams<'a> {
    pub llm: &'a LlmRuntime,
    pub session_id: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: String,
    pub declared_tools: Vec<ToolSchema>,
    pub current_visible_messages: Vec<Message>,
    pub cut: &'a CutPlan,
    pub focus: Option<&'a str>,
    pub reuse_prefix_cache: bool,
    pub cancel: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct SummaryOutput {
    pub summary: String,
    pub used_prefix_cache: bool,
}

/// 剥离模型偶发回显的 `<mizar-summary>` 包装标签，保留纯净摘要正文。
pub fn strip_summary_wrapper(raw: &str) -> String {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("<mizar-compaction>") {
        s = rest.trim();
    }
    if let Some(rest) = s.strip_suffix("</mizar-compaction>") {
        s = rest.trim();
    }
    if let Some(rest) = s.strip_prefix("<mizar-summary>") {
        s = rest.trim();
    }
    if let Some(rest) = s.strip_suffix("</mizar-summary>") {
        s = rest.trim();
    }
    s.to_string()
}

/// 生成结构化检查点摘要：优先尝试尾置指令复用主对话 KV Cache 前缀，
/// 若遇溢出、模型误调工具或空响应则自动回退到纯净转写请求。
pub async fn generate_checkpoint_summary<F>(
    params: SummarizeParams<'_>,
    mut on_progress: F,
) -> Result<SummaryOutput, LlmError>
where
    F: FnMut(&'static str, usize),
{
    if params.reuse_prefix_cache && !params.current_visible_messages.is_empty() {
        on_progress("prefill", 0);
        match run_prefix_reuse_summary(&params, &mut on_progress).await {
            Ok(text) if !text.trim().is_empty() => {
                return Ok(SummaryOutput {
                    summary: strip_summary_wrapper(&text),
                    used_prefix_cache: true,
                });
            }
            Ok(_) => {
                // 模型返回空文本或误调工具，降级至纯净转写路径
            }
            Err(err) if err.code == LlmErrorCode::Cancelled => return Err(err),
            Err(_) => {
                // 前缀复用失败（如接近窗口死线或工具参数限制），自动回退纯净转写
            }
        }
    }

    on_progress("transcript", 0);
    let text = run_clean_transcript_summary(&params, &mut on_progress).await?;
    let cleaned = strip_summary_wrapper(&text);
    if cleaned.is_empty() {
        return Err(LlmError::new(
            LlmErrorCode::EmptyResponse,
            "compaction summarizer returned empty summary",
        ));
    }
    Ok(SummaryOutput {
        summary: cleaned,
        used_prefix_cache: false,
    })
}

fn compact_preview(raw: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut prev_space = false;
    let mut count = 0usize;
    let mut truncated = false;
    for ch in raw.trim().chars() {
        let normalized = if ch.is_whitespace() { ' ' } else { ch };
        if normalized == ' ' && prev_space {
            continue;
        }
        prev_space = normalized == ' ';
        if count >= max_chars {
            truncated = true;
            break;
        }
        out.push(normalized);
        count += 1;
    }
    let trimmed = out.trim().to_string();
    if truncated {
        format!("{trimmed}...")
    } else {
        trimmed
    }
}

/// 生成保留边界首条消息的角色与内容特征预览（不依赖模型上下文中并不存在的 `seq #N`）。
fn describe_boundary_entry(entry: &SessionEntry) -> String {
    match &entry.event {
        SessionEvent::User { text, images } => {
            let preview = compact_preview(text, BOUNDARY_PREVIEW_MAX_CHARS);
            if !preview.is_empty() {
                format!("User message (\"{preview}\")")
            } else if !images.is_empty() {
                format!("User message ([{} image attachment(s)])", images.len())
            } else {
                "User message".to_string()
            }
        }
        SessionEvent::Assistant { blocks, .. } => {
            let text_joined: String = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } if !text.trim().is_empty() => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            let preview = compact_preview(&text_joined, BOUNDARY_PREVIEW_MAX_CHARS);
            if !preview.is_empty() {
                format!("Assistant message (\"{preview}\")")
            } else {
                let tool_names: Vec<&str> = blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolCall { name, .. } => Some(name.as_str()),
                        _ => None,
                    })
                    .collect();
                if tool_names.is_empty() {
                    "Assistant message".to_string()
                } else {
                    format!("Assistant tool call(s) ({})", tool_names.join(", "))
                }
            }
        }
        SessionEvent::ToolResult { name, content, .. } => {
            let preview = compact_preview(content, BOUNDARY_PREVIEW_MAX_CHARS);
            if preview.is_empty() {
                format!("ToolResult for `{name}`")
            } else {
                format!("ToolResult for `{name}` (\"{preview}\")")
            }
        }
        SessionEvent::BashExecution { command, .. } => {
            let preview = compact_preview(command, BOUNDARY_PREVIEW_MAX_CHARS);
            format!("local bash execution (`{preview}`)")
        }
        SessionEvent::ContextNotice { text, .. } => {
            let preview = compact_preview(text, BOUNDARY_PREVIEW_MAX_CHARS);
            format!("ContextNotice (\"{preview}\")")
        }
        _ => "boundary message".to_string(),
    }
}

async fn run_prefix_reuse_summary<F>(
    params: &SummarizeParams<'_>,
    on_progress: &mut F,
) -> Result<String, LlmError>
where
    F: FnMut(&'static str, usize),
{
    let mut messages = Vec::with_capacity(params.current_visible_messages.len() + 2);
    messages.push(Message::System {
        content: params.system_prompt.clone(),
    });
    messages.extend(params.current_visible_messages.iter().cloned());

    let tail_note = if let Some(first_kept) = params.cut.kept_entries.first() {
        let boundary_desc = describe_boundary_entry(first_kept);
        format!(
            "Important tail boundary: the most recent {} visible message(s) starting from the {boundary_desc} will be preserved verbatim after this checkpoint. Focus your summary on everything BEFORE that boundary message. Do NOT answer the user's latest pending request and do NOT call any tools.",
            params.cut.kept_entries.len(),
        )
    } else {
        "Do NOT answer any pending user request and do NOT call any tools.".to_string()
    };

    let focus_block = params
        .focus
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|f| format!("\n\nUser requested special focus for this compaction:\n{f}"))
        .unwrap_or_default();

    let instruction = format!(
        "[Context Compaction Request]\n\
         Create a structured handoff summary of our conversation so far.\n\
         {tail_note}\n\
         Note: Authoritative modified-file paths and shell command records are preserved separately by the runtime; focus on goals, verified facts, open hypotheses, decisions, and next steps.\n\n\
         {SUMMARY_SECTIONS_GUIDE}{focus_block}\n\n\
         Output ONLY the markdown summary text (no tool calls)."
    );
    messages.push(Message::user(instruction));

    let opts = GenerateOptions {
        messages,
        tools: params.declared_tools.clone(),
        model: params.model.clone(),
        provider: params.provider.clone(),
        system: None,
        max_tokens: Some(4_096),
        temperature: None,
        reasoning_effort: Some(ReasoningEffort::Off),
        headers: Vec::new(),
        session_id: params.session_id.clone(),
        cache_retention: None,
        cancel: params.cancel.clone(),
    };

    run_summary_stream_with_reasoning_fallback(params.llm, opts, true, on_progress).await
}

async fn run_clean_transcript_summary<F>(
    params: &SummarizeParams<'_>,
    on_progress: &mut F,
) -> Result<String, LlmError>
where
    F: FnMut(&'static str, usize),
{
    let transcript = serialize_transcript_for_summary(params.cut);
    let focus_block = params
        .focus
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|f| format!("\n\nUser requested special focus:\n{f}"))
        .unwrap_or_default();

    let prompt = format!(
        "Summarize the following coding-agent conversation transcript into a structured checkpoint.\n\
         Note: Authoritative modified files and shell commands are tracked separately by code.\n\n\
         {SUMMARY_SECTIONS_GUIDE}{focus_block}\n\n\
         <conversation>\n{transcript}\n</conversation>\n\n\
         Output ONLY the markdown summary text."
    );

    let opts = GenerateOptions {
        messages: vec![
            Message::System {
                content: "You are a precise technical context-compaction assistant for a coding agent. Never fabricate verified facts.".to_string(),
            },
            Message::user(prompt),
        ],
        tools: Vec::new(),
        model: params.model.clone(),
        provider: params.provider.clone(),
        system: None,
        max_tokens: Some(4_096),
        temperature: None,
        reasoning_effort: Some(ReasoningEffort::Off),
        headers: Vec::new(),
        session_id: None,
        cache_retention: None,
        cancel: params.cancel.clone(),
    };

    run_summary_stream_with_reasoning_fallback(params.llm, opts, false, on_progress).await
}

async fn run_summary_stream_with_reasoning_fallback<F>(
    llm: &LlmRuntime,
    opts: GenerateOptions,
    reject_tool_calls: bool,
    on_progress: &mut F,
) -> Result<String, LlmError>
where
    F: FnMut(&'static str, usize),
{
    let retry_without_reasoning = opts.reasoning_effort.is_some();
    let fallback_opts = if retry_without_reasoning {
        let mut cloned = opts.clone();
        cloned.reasoning_effort = None;
        Some(cloned)
    } else {
        None
    };

    match collect_summary_stream(llm, opts, reject_tool_calls, on_progress).await {
        Ok(text) => Ok(text),
        Err(err)
            if retry_without_reasoning
                && matches!(
                    err.code,
                    LlmErrorCode::InvalidRequest | LlmErrorCode::Unknown
                )
                && {
                    let msg = err.message.to_lowercase();
                    msg.contains("reason") || msg.contains("effort") || msg.contains("thinking")
                } =>
        {
            if let Some(fb) = fallback_opts {
                collect_summary_stream(llm, fb, reject_tool_calls, on_progress).await
            } else {
                Err(err)
            }
        }
        Err(err) => Err(err),
    }
}

async fn collect_summary_stream<F>(
    llm: &LlmRuntime,
    opts: GenerateOptions,
    reject_tool_calls: bool,
    on_progress: &mut F,
) -> Result<String, LlmError>
where
    F: FnMut(&'static str, usize),
{
    let mut stream = llm.stream(opts)?;
    let mut assembler = BlockAssembler::new();
    let mut chars_seen = 0usize;

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res?;
        if let StreamChunk::TextDelta { ref text, .. } = chunk
            && !text.is_empty()
        {
            chars_seen += text.chars().count();
            on_progress("summary", chars_seen);
        }
        assembler.push(chunk);
    }

    match assembler.finish() {
        FinishReason::Aborted => return Err(LlmError::cancelled()),
        FinishReason::Error {
            code,
            message,
            status,
            retry_after_ms,
        } => {
            return Err(LlmError {
                code,
                message,
                status,
                retry_after_ms,
            });
        }
        FinishReason::Stop | FinishReason::ToolCalls | FinishReason::MaxTokens => {}
    }

    if reject_tool_calls && !assembler.tool_calls().is_empty() && assembler.text().trim().is_empty()
    {
        return Ok(String::new());
    }

    Ok(assembler.text())
}

pub fn serialize_transcript_for_summary(cut: &CutPlan) -> String {
    let mut out = String::new();
    if let Some(prev) = &cut.previous_summary {
        out.push_str("<earlier-summary>\n");
        out.push_str(prev.trim());
        out.push_str("\n</earlier-summary>\n\n");
    }

    for entry in &cut.compacted_entries {
        if out.chars().count() >= TRANSCRIPT_TOTAL_MAX_CHARS {
            out.push_str("\n[... earlier transcript truncated for summarizer budget ...]\n");
            break;
        }
        if let Some(line) = format_entry_for_transcript(entry) {
            out.push_str(&line);
            out.push_str("\n\n");
        }
    }
    out
}

fn format_entry_for_transcript(entry: &SessionEntry) -> Option<String> {
    match &entry.event {
        SessionEvent::User { text, images } => {
            if images.is_empty() {
                Some(format!("[seq #{} | User]\n{text}", entry.seq))
            } else {
                Some(format!(
                    "[seq #{} | User]\n[图片 {} 张]\n{text}",
                    entry.seq,
                    images.len()
                ))
            }
        }
        SessionEvent::Assistant { blocks, .. } => {
            let mut parts = Vec::new();
            for b in blocks {
                match b {
                    ContentBlock::Text { text } if !text.trim().is_empty() => {
                        parts.push(text.clone());
                    }
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } => {
                        let args_snip: String = arguments.chars().take(400).collect();
                        parts.push(format!("[call {name}] {args_snip}"));
                    }
                    ContentBlock::Reasoning { .. } | ContentBlock::Text { .. } => {}
                }
            }
            if parts.is_empty() {
                None
            } else {
                Some(format!(
                    "[seq #{} | Assistant]\n{}",
                    entry.seq,
                    parts.join("\n")
                ))
            }
        }
        SessionEvent::ToolResult {
            name,
            content,
            is_error,
            ..
        } => {
            let status = if *is_error { "error" } else { "ok" };
            let body = truncate_with_ellipsis(content, TRANSCRIPT_TOOL_RESULT_MAX_CHARS);
            Some(format!(
                "[seq #{} | ToolResult {name} ({status})]\n{body}",
                entry.seq
            ))
        }
        SessionEvent::BashExecution {
            command,
            output,
            exit_code,
            exclude_from_context,
            ..
        } if !*exclude_from_context => {
            let body = truncate_with_ellipsis(output, TRANSCRIPT_TOOL_RESULT_MAX_CHARS);
            let code = exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "?".into());
            Some(format!(
                "[seq #{} | LocalBash exit={code}]\n$ {command}\n{body}",
                entry.seq
            ))
        }
        SessionEvent::ContextNotice { .. }
        | SessionEvent::Compaction { .. }
        | SessionEvent::ToolResultPrune { .. }
        | SessionEvent::BashExecution { .. }
        | SessionEvent::Selection { .. }
        | SessionEvent::Trace { .. } => None,
    }
}

fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{head}\n[... truncated {} chars ...]", count - max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_llm::ImageAttachment;

    #[test]
    fn transcript_summarizes_image_count_without_base64() {
        let entry = SessionEntry {
            seq: 1,
            ts: 0,
            event: SessionEvent::user_with_images(
                "看看这个报错",
                vec![ImageAttachment {
                    media_type: "image/png".into(),
                    data: "UNIQUE_BASE64_PAYLOAD_XYZ".into(),
                    width: Some(100),
                    height: Some(100),
                }],
            ),
        };
        let rendered = format_entry_for_transcript(&entry).expect("user entry should render");
        assert!(rendered.contains("[图片 1 张]"), "{rendered}");
        assert!(rendered.contains("看看这个报错"), "{rendered}");
        assert!(
            !rendered.contains("UNIQUE_BASE64_PAYLOAD_XYZ"),
            "{rendered}"
        );
    }
}
