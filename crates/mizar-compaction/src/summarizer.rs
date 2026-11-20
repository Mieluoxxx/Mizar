//! L2 LLM 检查点摘要生成器（综合 `dsh` 同前缀尾置指令、`DS-Bot` `checkpointTailNote`、
//! `dim #170` 实证关闭思考、`obelisk` 已证实事实/未证实推断分级）。

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
- Important file paths, symbol names, or `[seq #N]` handles for `recall_read` if verbatim detail is later needed.";

pub struct SummarizeParams<'a> {
    pub llm: &'a LlmRuntime,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: String,
    pub declared_tools: Vec<ToolSchema>,
    pub current_visible_messages: Vec<Message>,
    pub cut: &'a CutPlan,
    pub focus: Option<&'a str>,
    pub reuse_prefix_cache: bool,
    pub disable_reasoning: bool,
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

    let tail_note = if !params.cut.kept_entries.is_empty() {
        format!(
            "Important tail boundary: the most recent {} visible message(s) starting at seq #{} will be preserved verbatim after this checkpoint. Focus your summary on everything BEFORE seq #{}. Do NOT answer the user's latest pending request and do NOT call any tools.",
            params.cut.kept_entries.len(),
            params.cut.first_kept_seq,
            params.cut.first_kept_seq
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
        temperature: Some(0.2),
        reasoning_effort: params.disable_reasoning.then_some(ReasoningEffort::Off),
        headers: Vec::new(),
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
        temperature: Some(0.2),
        reasoning_effort: params.disable_reasoning.then_some(ReasoningEffort::Off),
        headers: Vec::new(),
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
                && err.message.to_lowercase().contains("reason") =>
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
