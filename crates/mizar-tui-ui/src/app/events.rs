//! `UiEvent` 异步后端事件消费与状态归约（含流式增量、工具卡片与上下文压缩进度）。

use serde_json::Value;

use super::{
    ActivityState, AppState, ChatItem, NoticeLevel, ToolExtraMeta, ToolState, UiAgentStatus,
    UiEvent, UiLogLevel, UiStepOutcome, UiTurnEndReason, bash_inline_meta, bash_scope_label,
    extract_inline_meta, summarize_live_preview, summarize_tool_args, summarize_tool_result,
    turn_end_program_status,
};
use crate::reveal::RevealSlot;
use crate::terminal::{ProgramStatusState, emit_terminal_progress};

impl AppState {
    pub(super) fn apply_local_bash_completion(
        &mut self,
        call_id: String,
        command: &str,
        exclude_from_context: bool,
        display: String,
        status: ToolState,
        inline_meta: String,
    ) -> usize {
        let failed = !matches!(status, ToolState::Done);
        let preview = summarize_tool_result("bash", &display, None, failed);
        let idx = if let Some(pos) = self.items.iter().rposition(|item| match item {
            ChatItem::ToolCall { id, .. } => id == &call_id,
            _ => false,
        }) {
            if let ChatItem::ToolCall {
                status: st,
                output_preview,
                ..
            } = &mut self.items[pos]
            {
                *st = status;
                *output_preview = preview;
            }
            pos
        } else {
            let args_json = serde_json::json!({ "command": command }).to_string();
            let args_summary = summarize_tool_args(&args_json);
            self.tool_extras
                .entry(call_id.clone())
                .or_insert_with(|| ToolExtraMeta {
                    full_args: args_json,
                    exclude_from_context,
                    ..Default::default()
                });
            self.items.push(ChatItem::ToolCall {
                id: call_id.clone(),
                name: "bash".to_string(),
                args_summary,
                status,
                output_preview: preview,
            });
            self.items.len() - 1
        };
        let ext = self.tool_extras.entry(call_id).or_default();
        ext.full_content = display;
        ext.inline_meta = Some(inline_meta);
        ext.exclude_from_context = exclude_from_context;
        idx
    }

    fn ensure_open_assistant(&mut self) -> (usize, &mut ChatItem) {
        let already_open = match &self.activity {
            ActivityState::Agent {
                streaming_assistant_open,
                ..
            } => *streaming_assistant_open,
            _ => true,
        } && matches!(self.items.last(), Some(ChatItem::Assistant { .. }));
        if !already_open {
            self.items.push(ChatItem::Assistant {
                reasoning: String::new(),
                text: String::new(),
            });
            if let ActivityState::Agent {
                streaming_assistant_open,
                ..
            } = &mut self.activity
            {
                *streaming_assistant_open = true;
            }
        }
        let idx = self.items.len() - 1;
        (idx, &mut self.items[idx])
    }

    pub fn apply_ui_event(&mut self, event: UiEvent) {
        match event {
            UiEvent::Status(status) => match status {
                UiAgentStatus::Busy => {
                    if !matches!(self.activity, ActivityState::Agent { .. }) {
                        self.activity = ActivityState::Agent {
                            streaming_assistant_open: false,
                            compaction_status: None,
                        };
                    }
                }
                UiAgentStatus::Idle => {
                    if matches!(self.activity, ActivityState::Agent { .. }) {
                        self.activity = ActivityState::Idle;
                        self.reveal.snap_all();
                    }
                }
            },
            UiEvent::TurnStart { .. } => {
                let preserved_compaction = match &mut self.activity {
                    ActivityState::Agent {
                        compaction_status, ..
                    } => compaction_status.take(),
                    _ => None,
                };
                self.activity = ActivityState::Agent {
                    streaming_assistant_open: false,
                    compaction_status: preserved_compaction,
                };
                self.auto_scroll = true;
                self.last_turn_end_reason = None;
                self.last_step_outcome = None;
                self.tps.on_turn_start();
                emit_terminal_progress(true);
                self.record_program_status(ProgramStatusState::Working, None, Some("Generating"));
            }
            UiEvent::TurnEnd { reason } => {
                self.activity = ActivityState::Idle;
                self.reveal.snap_all();
                emit_terminal_progress(false);
                let (program_state, program_msg) = turn_end_program_status(&reason);
                self.record_program_status(program_state, None, Some(program_msg));
                match &reason {
                    UiTurnEndReason::Completed => {}
                    UiTurnEndReason::Truncated => {
                        if let Some((step, UiStepOutcome::MaxTokens)) = self.last_step_outcome
                            && step > 0
                        {
                            self.push_notice(
                                NoticeLevel::Warn,
                                format!(
                                    "第 {step} 步输出因达到 max_tokens 上限被截断，回复可能不完整"
                                ),
                            );
                        } else {
                            self.push_notice(
                                NoticeLevel::Warn,
                                "输出因达到 max_tokens 上限被截断，回复可能不完整",
                            );
                        }
                    }
                    UiTurnEndReason::MaxSteps => {
                        self.push_notice(NoticeLevel::Warn, "已达到最大执行步数上限，本轮提前结束");
                    }
                    UiTurnEndReason::Cancelled => {
                        self.push_notice(NoticeLevel::Info, "已中断本轮生成");
                    }
                    UiTurnEndReason::Error { .. } => {}
                }
                self.last_turn_end_reason = Some(reason);
                if let Some(tps) = self.tps.on_turn_end() {
                    self.tps_history.push(tps);
                    if self.tps_history.len() > 32 {
                        self.tps_history.remove(0);
                    }
                }
                self.refresh_context_breakdown();
            }
            UiEvent::StepStart { ts_ms } => {
                self.last_step_start_ms = Some(ts_ms);
                self.tps.on_step_start();
            }
            UiEvent::StepEnd {
                step,
                ts_ms,
                outcome,
                usage,
            } => {
                self.last_step_outcome = Some((step, outcome));
                self.tps.on_step_end(ts_ms, usage);
            }
            UiEvent::FirstToken { ts_ms } => {
                if let Some(start) = self.last_step_start_ms.take() {
                    let ttft = ts_ms.saturating_sub(start) as u64;
                    self.last_ttft_ms = Some(ttft);
                    self.ttft_history.push(ttft);
                    if self.ttft_history.len() > 32 {
                        self.ttft_history.remove(0);
                    }
                }
                self.tps.on_first_token(ts_ms);
            }
            UiEvent::ReasoningDelta { text, ts_ms } => {
                let busy = self.is_busy();
                let (idx, item) = self.ensure_open_assistant();
                let len = if let ChatItem::Assistant { reasoning, .. } = item {
                    reasoning.push_str(&text);
                    reasoning.len()
                } else {
                    0
                };
                self.reveal
                    .observe(RevealSlot::AssistantReasoning(idx), len, busy);
                self.tps.on_token_delta(&text, ts_ms);
            }
            UiEvent::TextDelta { text, ts_ms } => {
                let busy = self.is_busy();
                let (idx, item) = self.ensure_open_assistant();
                let len = if let ChatItem::Assistant { text: body, .. } = item {
                    body.push_str(&text);
                    body.len()
                } else {
                    0
                };
                self.reveal
                    .observe(RevealSlot::AssistantText(idx), len, busy);
                self.tps.on_token_delta(&text, ts_ms);
            }
            UiEvent::ToolCall(call) => {
                self.close_streaming_assistant();
                self.reveal.snap_all();
                let args_summary = summarize_tool_args(&call.arguments);
                self.record_touched_file(&call.name, &call.arguments);
                self.tool_extras.insert(
                    call.id.clone(),
                    ToolExtraMeta {
                        full_content: String::new(),
                        diff_raw: None,
                        full_args: call.arguments,
                        inline_meta: None,
                        exclude_from_context: false,
                    },
                );
                self.items.push(ChatItem::ToolCall {
                    id: call.id,
                    name: call.name,
                    args_summary,
                    status: ToolState::Running,
                    output_preview: String::new(),
                });
                let idx = self.items.len() - 1;
                self.item_cache
                    .invalidate_tool_neighborhood(&self.items, idx);
            }
            UiEvent::ToolUpdate {
                call_id, preview, ..
            } => {
                if let Some(ext) = self.tool_extras.get_mut(&call_id) {
                    ext.full_content = preview.clone();
                }
                if let Some(pos) = self.items.iter().rposition(|item| match item {
                    ChatItem::ToolCall { id, .. } => id == &call_id,
                    _ => false,
                }) {
                    if let ChatItem::ToolCall { output_preview, .. } = &mut self.items[pos] {
                        *output_preview = summarize_live_preview(&preview);
                    }
                    self.item_cache
                        .invalidate_tool_neighborhood(&self.items, pos);
                }
            }
            UiEvent::ToolResult {
                call_id,
                name,
                content,
                is_error,
                blocked,
                details,
            } => {
                let status = if blocked {
                    ToolState::Blocked
                } else if is_error {
                    ToolState::Error
                } else {
                    ToolState::Done
                };
                let failed = is_error || blocked;
                let diff_raw = if !failed && name == "edit" {
                    details
                        .as_ref()
                        .and_then(|d| d.get("diff"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                } else {
                    None
                };
                let inline_meta = extract_inline_meta(&name, &content, details.as_ref(), failed);
                {
                    let extra = self.tool_extras.entry(call_id.clone()).or_default();
                    extra.full_content = content.clone();
                    extra.diff_raw = diff_raw;
                    extra.inline_meta = inline_meta;
                }
                let preview = summarize_tool_result(&name, &content, details.as_ref(), failed);
                let idx = if let Some(pos) = self.items.iter().rposition(|item| match item {
                    ChatItem::ToolCall { id, .. } => id == &call_id,
                    _ => false,
                }) {
                    if let ChatItem::ToolCall {
                        status: st,
                        output_preview,
                        ..
                    } = &mut self.items[pos]
                    {
                        *st = status;
                        *output_preview = preview;
                    }
                    pos
                } else {
                    self.items.push(ChatItem::ToolCall {
                        id: call_id,
                        name,
                        args_summary: String::new(),
                        status,
                        output_preview: preview,
                    });
                    self.items.len() - 1
                };
                self.item_cache
                    .invalidate_tool_neighborhood(&self.items, idx);
            }
            UiEvent::Usage(usage) => {
                self.total_usage.input_tokens = self
                    .total_usage
                    .input_tokens
                    .saturating_add(usage.input_tokens);
                self.total_usage.output_tokens = self
                    .total_usage
                    .output_tokens
                    .saturating_add(usage.output_tokens);
                if let Some(cr) = usage.cache_read_tokens {
                    self.total_usage.cache_read_tokens = Some(
                        self.total_usage
                            .cache_read_tokens
                            .unwrap_or(0)
                            .saturating_add(cr),
                    );
                }
                if let Some(cw) = usage.cache_write_tokens {
                    self.total_usage.cache_write_tokens = Some(
                        self.total_usage
                            .cache_write_tokens
                            .unwrap_or(0)
                            .saturating_add(cw),
                    );
                }
                // 口径统一由 `TokenUsage::grand_total` 定义：input 只是缓存未命中的净额，
                // 漏掉 cache_read / cache_write 会把开了 prompt cache 的会话读成个位数百分比。
                self.total_usage.total_tokens = Some(self.total_usage.grand_total());
                // 上下文占用估算用「最近一次请求」的 prompt 全量，而不是累计值。
                let last = usage.prompt_tokens();
                if last > 0 {
                    self.last_prompt_tokens = Some(last);
                }
                self.refresh_context_breakdown();
            }
            UiEvent::Anomaly(summary) => {
                self.anomalies.push(summary.clone());
                self.push_notice(NoticeLevel::Warn, format!("轨迹告警：{summary}"));
            }
            UiEvent::Flash(msg) => {
                self.set_flash(msg);
            }
            UiEvent::Notice { level, text } => {
                self.push_notice(level, text);
            }
            UiEvent::UserSkillsUpdated(names) => {
                self.user_skill_names = names;
            }
            UiEvent::SessionTitleUpdated(title) => {
                self.session_title = title;
            }
            UiEvent::ToolsUpdated => {}
            UiEvent::Log(record) => {
                let level = match record.level {
                    UiLogLevel::Error => NoticeLevel::Error,
                    UiLogLevel::Warn => NoticeLevel::Warn,
                    UiLogLevel::Debug | UiLogLevel::Info | UiLogLevel::Silent => return,
                };
                self.push_notice(level, format!("[{}] {}", record.name, record.message));
            }
            UiEvent::RunFinished(res) => {
                if matches!(self.activity, ActivityState::Agent { .. }) {
                    self.activity = ActivityState::Idle;
                }
                self.reveal.snap_all();
                emit_terminal_progress(false);
                if let Err(err) = res
                    && !err.contains("request aborted")
                {
                    self.push_notice(NoticeLevel::Error, format!("请求失败：{err}"));
                    self.record_program_status(
                        ProgramStatusState::Error,
                        None,
                        Some("Request failed"),
                    );
                    if self.last_turn_end_reason.is_none() {
                        self.last_turn_end_reason = Some(UiTurnEndReason::Error { message: err });
                    }
                }
                self.refresh_context_breakdown();
            }
            UiEvent::LocalBashFinished {
                call_id,
                command,
                exclude_from_context,
                output,
            } => {
                if matches!(self.activity, ActivityState::LocalBash) {
                    self.activity = ActivityState::Idle;
                }
                let idx = match output {
                    Ok(out) => {
                        let status = if out.cancelled {
                            ToolState::Blocked
                        } else if out.exit_code.unwrap_or(0) != 0 {
                            ToolState::Error
                        } else {
                            ToolState::Done
                        };
                        let inline_meta =
                            bash_inline_meta(exclude_from_context, out.exit_code, out.cancelled);
                        self.apply_local_bash_completion(
                            call_id,
                            &command,
                            exclude_from_context,
                            out.display,
                            status,
                            inline_meta,
                        )
                    }
                    Err(err) => {
                        let inline_meta =
                            format!("{} · 失败", bash_scope_label(exclude_from_context));
                        self.apply_local_bash_completion(
                            call_id,
                            &command,
                            exclude_from_context,
                            err,
                            ToolState::Error,
                            inline_meta,
                        )
                    }
                };
                self.item_cache
                    .invalidate_tool_neighborhood(&self.items, idx);
                self.refresh_context_breakdown();
            }
            UiEvent::ApprovalRequested {
                tool_name,
                args_preview,
                responder,
            } => {
                if let Some(tx) = responder.lock().unwrap().take() {
                    let opened_now =
                        self.overlay
                            .open_approval(tool_name.clone(), args_preview, tx);
                    if opened_now {
                        self.record_program_status(
                            ProgramStatusState::Blocked,
                            Some("permission"),
                            Some(&format!("Approve {tool_name}")),
                        );
                    }
                }
            }
            UiEvent::CompactionStarted { trigger } => {
                let label = match trigger.as_str() {
                    "manual" => "手动压缩中…",
                    "overflow" => "溢出恢复压缩中…",
                    "cold-cache" => "冷缓存整理中…",
                    _ => "上下文压缩中…",
                }
                .to_string();
                match &mut self.activity {
                    ActivityState::Agent {
                        compaction_status, ..
                    } => *compaction_status = Some(label),
                    _ => self.activity = ActivityState::Compacting { status: label },
                }
            }
            UiEvent::CompactionProgress { stage, delta_chars } => {
                let status = match stage.as_str() {
                    "pruned" => "压缩中 · 已剪枝大输出".to_string(),
                    "prefill" => "压缩中 · prefill…".to_string(),
                    "transcript" => "压缩中 · 整理记录…".to_string(),
                    "summary" => format!("压缩中 · 摘要 {delta_chars} 字"),
                    other => format!("压缩中 · {other}"),
                };
                match &mut self.activity {
                    ActivityState::Agent {
                        compaction_status, ..
                    } => *compaction_status = Some(status),
                    _ => self.activity = ActivityState::Compacting { status },
                }
            }
            UiEvent::CompactionFinished {
                outcome,
                tokens_before,
                tokens_after,
                first_kept_seq,
                summary_preview,
                message,
                is_manual,
            } => {
                self.apply_compaction_finished(
                    outcome,
                    tokens_before,
                    tokens_after,
                    first_kept_seq,
                    summary_preview,
                    message,
                    is_manual,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_compaction_finished(
        &mut self,
        outcome: String,
        tokens_before: Option<u64>,
        tokens_after: Option<u64>,
        first_kept_seq: Option<u64>,
        summary_preview: Option<String>,
        message: Option<String>,
        is_manual: bool,
    ) {
        match &mut self.activity {
            ActivityState::Agent {
                compaction_status, ..
            } => *compaction_status = None,
            ActivityState::Compacting { .. } => self.activity = ActivityState::Idle,
            _ => {}
        }
        // 遵循 dsh-TUI 原则：total_usage 计费读数永不随压缩回退，仅刷新上下文占用视图
        if outcome == "compacted"
            && let Some(after) = tokens_after.filter(|&v| v > 0)
        {
            self.last_prompt_tokens = Some(after);
        }
        self.refresh_context_breakdown();

        match outcome.as_str() {
            "compacted" => {
                let kept_tag = first_kept_seq
                    .map(|s| format!("保留自 #{s}"))
                    .unwrap_or_else(|| "已折叠".to_string());
                let tok_tag = match (tokens_before, tokens_after) {
                    (Some(b), Some(a)) => {
                        let saved = b.saturating_sub(a);
                        format!(" · {b} → {a} tok (↓{saved})")
                    }
                    _ => String::new(),
                };
                let body = summary_preview
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("\n{s}"))
                    .unwrap_or_default();
                self.push_notice(
                    NoticeLevel::Info,
                    format!("✓ 上下文压缩完成（{kept_tag}{tok_tag}）{body}"),
                );
            }
            "skipped" => {
                if is_manual {
                    let reason = message.unwrap_or_else(|| "当前历史较短，暂无可压缩段落".into());
                    self.push_notice(NoticeLevel::Info, format!("跳过压缩：{reason}"));
                }
            }
            "failed" => {
                let err_msg = message.unwrap_or_else(|| "未知错误".into());
                if !err_msg.contains("request aborted") {
                    self.push_notice(NoticeLevel::Warn, format!("上下文压缩失败：{err_msg}"));
                }
            }
            _ => {}
        }
    }
}
