//! LLM 检查点摘要生成器（同前缀尾置指令、`ReasoningEffort::Off` + 错误放宽回退、
//! `obelisk` 已证实事实/未证实推断分级）。

use std::borrow::Cow;
use std::fmt::{self, Write};
use std::time::Duration;

use mizar_llm::{
    BlockAssembler, ContentBlock, FinishReason, GenerateOptions, LlmError, LlmErrorCode,
    LlmRuntime, Message, ReasoningEffort, ToolSchema,
};
use mizar_llm_retry::{RetryConfig, collect_with_retry, sleep_cancellable};
use mizar_session::{SessionEntry, SessionEvent};
use tokio_util::sync::CancellationToken;

use crate::digest::CutPlan;

const TRANSCRIPT_TOOL_RESULT_MAX_CHARS: usize = 2_000;
const TRANSCRIPT_TOTAL_MAX_CHARS: usize = 96_000;
const TRANSCRIPT_MIN_BUDGET: usize = 24_000;
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
- Important file paths and symbol names.";

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
    let mut retry = RetryConfig::from_env();
    retry.max_retries = retry.max_retries.min(2);
    let mut retries_used = 0;
    let mut reuse_prefix = params.reuse_prefix_cache && !params.current_visible_messages.is_empty();
    let mut budget = TRANSCRIPT_TOTAL_MAX_CHARS;
    let mut reasoning_effort = Some(ReasoningEffort::Off);

    loop {
        if params.cancel.is_cancelled() {
            return Err(LlmError::cancelled());
        }
        let opts = if reuse_prefix {
            on_progress("prefill", 0);
            prefix_reuse_options(&params, reasoning_effort)
        } else {
            on_progress("transcript", 0);
            clean_transcript_options(&params, budget, reasoning_effort)
        };
        let result = collect_with_retry(params.llm, opts, &retry, &mut retries_used, |chars| {
            on_progress("summary", chars);
        })
        .await;
        let (err, can_transcribe) = match result {
            Ok(assembler) => match checkpoint_text(&assembler) {
                Ok(summary) => {
                    return Ok(SummaryOutput {
                        summary,
                        used_prefix_cache: reuse_prefix,
                    });
                }
                Err(err) => (err.with_http_hints(assembler.response_hints()), true),
            },
            // 收集器已停止重试的瞬时错误不能借换提示词再次请求。
            Err(err) if err.is_retryable() => return Err(err),
            Err(err) => {
                let can_transcribe = matches!(
                    err.code,
                    LlmErrorCode::ContextWindowExceeded
                        | LlmErrorCode::PayloadTooLarge
                        | LlmErrorCode::InvalidRequest
                        | LlmErrorCode::EmptyResponse
                );
                (err, can_transcribe)
            }
        };

        let retry_after = err.retry_after_ms.map(Duration::from_millis);
        if err.should_retry == Some(false)
            || retry_after.is_some_and(|delay| delay > retry.max_retry_after)
        {
            return Err(err);
        }
        if reasoning_effort.is_some() && rejects_reasoning_effort(&err) {
            reasoning_effort = None;
        } else if reuse_prefix && can_transcribe {
            reuse_prefix = false;
        } else if err.code == LlmErrorCode::ContextWindowExceeded && budget > TRANSCRIPT_MIN_BUDGET
        {
            budget /= 2;
        } else {
            return Err(err);
        }
        if let Some(delay) = retry_after {
            sleep_cancellable(delay, &params.cancel).await?;
        }
    }
}

fn checkpoint_text(assembler: &BlockAssembler) -> Result<String, LlmError> {
    let finish = assembler.finish();
    if finish == FinishReason::MaxTokens {
        return Err(LlmError::malformed(
            "compaction summary reached the output token limit",
        ));
    }
    let blocks = assembler.blocks();
    if finish == FinishReason::ToolCalls
        || blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. }))
    {
        return Err(LlmError::malformed(
            "compaction summarizer requested a tool",
        ));
    }
    let text: String = blocks.iter().filter_map(ContentBlock::as_text).collect();
    let summary = strip_summary_wrapper(&text);
    if summary.is_empty() {
        return Err(LlmError::new(
            LlmErrorCode::EmptyResponse,
            "compaction summarizer returned empty summary",
        ));
    }
    Ok(summary)
}

fn rejects_reasoning_effort(err: &LlmError) -> bool {
    if !matches!(
        err.code,
        LlmErrorCode::InvalidRequest | LlmErrorCode::Unknown
    ) {
        return false;
    }
    let message = err.message.to_ascii_lowercase();
    message.contains("reason") || message.contains("effort") || message.contains("thinking")
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

fn prefix_reuse_options(
    params: &SummarizeParams<'_>,
    reasoning_effort: Option<ReasoningEffort>,
) -> GenerateOptions {
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

    GenerateOptions {
        messages,
        tools: params.declared_tools.clone(),
        model: params.model.clone(),
        supports_images: None,
        provider: params.provider.clone(),
        system: None,
        max_tokens: Some(4_096),
        temperature: None,
        reasoning_effort,
        headers: Vec::new(),
        session_id: params.session_id.clone(),
        cache_retention: None,
        cancel: params.cancel.clone(),
    }
}

fn clean_transcript_options(
    params: &SummarizeParams<'_>,
    budget: usize,
    reasoning_effort: Option<ReasoningEffort>,
) -> GenerateOptions {
    const SYSTEM: &str = "You are a precise technical context-compaction assistant for a coding agent. Never fabricate verified facts.";
    const END: &str = "\n</conversation>\n\nOutput ONLY the markdown summary text.";
    // 手动指定的重点可能很长，不能让它挤掉转录或抵消缩小后的预算。
    let focus_limit = budget / 4;
    let focus_block = params
        .focus
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|f| format!("\n\nUser requested special focus:\n{f:.focus_limit$}"))
        .unwrap_or_default();
    let mut prompt = format!(
        "Summarize the following coding-agent conversation transcript into a structured checkpoint.\n\
         Note: Authoritative modified files and shell commands are tracked separately by code.\n\n\
         {SUMMARY_SECTIONS_GUIDE}{focus_block}\n\n<conversation>\n"
    );
    let transcript_budget = budget - SYSTEM.len() - prompt.chars().count() - END.len();
    prompt.push_str(&serialize_transcript_with_budget(
        params.cut,
        transcript_budget,
    ));
    prompt.push_str(END);

    GenerateOptions {
        messages: vec![
            Message::System {
                content: SYSTEM.to_string(),
            },
            Message::user(prompt),
        ],
        tools: Vec::new(),
        model: params.model.clone(),
        supports_images: None,
        provider: params.provider.clone(),
        system: None,
        max_tokens: Some(4_096),
        temperature: None,
        reasoning_effort,
        headers: Vec::new(),
        session_id: None,
        cache_retention: None,
        cancel: params.cancel.clone(),
    }
}

pub fn serialize_transcript_for_summary(cut: &CutPlan) -> String {
    serialize_transcript_with_budget(cut, TRANSCRIPT_TOTAL_MAX_CHARS)
}

struct Transcript {
    text: String,
    remaining: usize,
}

impl Write for Transcript {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for ch in s.chars() {
            if self.remaining == 0 {
                return Err(fmt::Error);
            }
            self.text.push(ch);
            self.remaining -= 1;
        }
        Ok(())
    }
}

impl Transcript {
    fn mark_truncated(&mut self) {
        self.text.pop();
        self.text.push('…');
    }
}

fn serialize_transcript_with_budget(cut: &CutPlan, budget: usize) -> String {
    let mut out = Transcript {
        text: String::new(),
        remaining: budget,
    };
    if let Some(prev) = &cut.previous_summary {
        const END: &str = "\n</earlier-summary>\n\n";
        // 给新记录留足空间，旧摘要的标签和省略号也占预算。
        out.remaining = budget / 4 - END.len();
        if write!(out, "<earlier-summary>\n{}", prev.trim()).is_err() {
            out.mark_truncated();
        }
        out.text.push_str(END);
        out.remaining = budget - out.text.chars().count();
    }
    for entry in &cut.compacted_entries {
        if write_transcript_entry(&mut out, entry).is_err() {
            out.mark_truncated();
            break;
        }
    }
    out.text
}

fn write_transcript_entry(out: &mut Transcript, entry: &SessionEntry) -> fmt::Result {
    match &entry.event {
        SessionEvent::User { text, images } => {
            writeln!(out, "[seq #{} | User]", entry.seq)?;
            if !images.is_empty() {
                writeln!(out, "[图片 {} 张]", images.len())?;
            }
            writeln!(out, "{text}\n")
        }
        SessionEvent::Assistant { blocks, .. } => {
            let mut parts = blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } if !text.trim().is_empty() => {
                        Some(Cow::Borrowed(text.as_str()))
                    }
                    ContentBlock::ToolCall {
                        name, arguments, ..
                    } => Some(Cow::Owned(format!("[call {name}] {arguments:.400}"))),
                    _ => None,
                })
                .peekable();
            if parts.peek().is_none() {
                return Ok(());
            }
            writeln!(out, "[seq #{} | Assistant]", entry.seq)?;
            for part in parts {
                writeln!(out, "{part}")?;
            }
            writeln!(out)
        }
        SessionEvent::ToolResult {
            name,
            content,
            is_error,
            ..
        } => {
            let status = if *is_error { "error" } else { "ok" };
            let body = truncate_with_ellipsis(content, TRANSCRIPT_TOOL_RESULT_MAX_CHARS);
            writeln!(
                out,
                "[seq #{} | ToolResult {name} ({status})]\n{body}\n",
                entry.seq
            )
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
            writeln!(
                out,
                "[seq #{} | LocalBash exit={code}]\n$ {command}\n{body}\n",
                entry.seq
            )
        }
        SessionEvent::ContextNotice { .. }
        | SessionEvent::Compaction { .. }
        | SessionEvent::ToolResultPrune { .. }
        | SessionEvent::BashExecution { .. }
        | SessionEvent::Selection { .. }
        | SessionEvent::Trace { .. } => Ok(()),
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
