//! `CompactionRuntime` Cordis 服务（`"compaction"`）：
//! 统一编排 L1 剪枝、L2 软硬分离摘要、L3 溢出恢复、手动 `/compact`、上下文压力探测与压后工具守卫。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Next, Service};
use mizar_agent_loop::{
    AgentPreStepDecision, AgentPreStepInput, AgentRequestErrorDecision, AgentRequestErrorInput,
    AgentRequestErrorKind, AgentStepEndEvent, EVENT_AGENT_PRE_STEP, EVENT_AGENT_REQUEST_ERROR,
    EVENT_AGENT_STEP_END, StepOutcome,
};
use mizar_llm::LlmContextExt;
use mizar_session::{Session, SessionContextExt, SessionTraceEvent};
use mizar_system_prompt::SystemPromptContextExt;
use mizar_tools::{
    EVENT_TOOLS_POST_EXECUTE, EVENT_TOOLS_PRE_EXECUTE, PostExecuteInput, PreExecuteDecision,
    ToolExecution, ToolResult, ToolsContextExt,
};
use tokio_util::sync::CancellationToken;

use crate::budget::{
    ContextBudget, compute_budget, estimate_session_context_tokens, estimate_text_tokens,
};
use crate::digest::{build_compaction_digest, plan_compaction_cut};
use crate::guards::FileGuardTracker;
use crate::pruner::plan_tool_result_prune_with_pending;
use crate::summarizer::{SummarizeParams, generate_checkpoint_summary};
use crate::{
    CompactionConfig, CompactionEndEvent, CompactionError, CompactionOutcome,
    CompactionProgressEvent, CompactionReport, CompactionStartEvent, CompactionTrigger,
    ContextPressureReport, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS, EVENT_COMPACTION_START,
    ManualCompactOptions,
};

#[derive(Debug, Default)]
struct SessionCompactionMeta {
    last_compacted_tokens: Option<u64>,
    last_answered_route: Option<(Option<String>, Option<String>)>,
    last_activity_at: Option<Instant>,
}

struct InFlightGuard {
    in_flight: Arc<Mutex<HashSet<String>>>,
    session_id: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight.lock().unwrap().remove(&self.session_id);
    }
}

pub struct CompactionRuntime {
    ctx: Context,
    config: Arc<Mutex<CompactionConfig>>,
    in_flight: Arc<Mutex<HashSet<String>>>,
    sessions_meta: Arc<Mutex<HashMap<String, SessionCompactionMeta>>>,
    file_guards: Arc<FileGuardTracker>,
}

impl CompactionRuntime {
    /// 刷新指定会话的最近活跃时间戳（供缓存保活等外部模块同步冷缓存判定窗口）。
    pub fn touch_session_activity(&self, session_id: &str) {
        let mut guard = self.sessions_meta.lock().unwrap();
        let entry = guard.entry(session_id.to_string()).or_default();
        entry.last_activity_at = Some(Instant::now());
    }

    pub fn config(&self) -> CompactionConfig {
        self.config.lock().unwrap().clone()
    }

    /// 运行时热更新压缩配置（供 `/settings` 等交互界面即时生效，无需重建服务）。
    pub fn update_config(&self, f: impl FnOnce(&mut CompactionConfig)) {
        let mut guard = self.config.lock().unwrap();
        f(&mut guard);
    }

    fn try_lock_session(&self, session_id: &str) -> Result<InFlightGuard, CompactionError> {
        let mut set = self.in_flight.lock().unwrap();
        if !set.insert(session_id.to_string()) {
            return Err(CompactionError::AlreadyRunning(session_id.to_string()));
        }
        Ok(InFlightGuard {
            in_flight: Arc::clone(&self.in_flight),
            session_id: session_id.to_string(),
        })
    }

    /// 结合 `LlmRuntime::provider_infos()` 查表推导当前路由的模型窗口与最大输出。
    fn resolve_model_limits(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        default_window: u64,
    ) -> (u64, Option<u32>) {
        if let Ok(llm) = self.ctx.llm() {
            let providers = llm.provider_infos();
            let chosen_provider = provider
                .and_then(|pid| providers.iter().find(|p| p.id == pid))
                .or_else(|| providers.first());
            if let Some(p) = chosen_provider {
                let chosen_model = model
                    .and_then(|mid| p.models.iter().find(|m| m.id == mid))
                    .or_else(|| p.models.first());
                if let Some(m) = chosen_model {
                    return (m.context_window.unwrap_or(default_window), m.max_tokens);
                }
            }
        }
        (default_window, None)
    }

    fn compute_session_budget(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        cfg: &CompactionConfig,
    ) -> ContextBudget {
        let (window, max_out) =
            self.resolve_model_limits(provider, model, cfg.default_context_window);
        compute_budget(window, max_out, cfg)
    }

    fn current_prompt_and_tools(
        &self,
        session_id: &str,
    ) -> (String, Vec<mizar_llm::ToolSchema>, usize) {
        let system_prompt = self
            .ctx
            .system_prompt()
            .map(|sp| sp.build())
            .unwrap_or_default();
        let tools = self
            .ctx
            .tools()
            .map(|rt| rt.list_declared(Some(session_id)))
            .unwrap_or_default();
        let tools_chars: usize = tools
            .iter()
            .map(|t| t.name.len() + t.description.len() + t.parameters.to_string().len())
            .sum();
        (system_prompt, tools, tools_chars)
    }

    /// 探测当前会话上下文压力快照（供 TUI 状态栏、`/stats` 与调试面板展示）。
    pub fn inspect_context(
        &self,
        session: &Session,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> ContextPressureReport {
        let cfg = self.config();
        let budget = self.compute_session_budget(provider, model, &cfg);
        let (sys_prompt, _, tools_chars) = self.current_prompt_and_tools(&session.id);
        let estimated_tokens = estimate_session_context_tokens(session, &sys_prompt, tools_chars);
        let ratio = (estimated_tokens as f64) / (budget.context_window.max(1) as f64);
        ContextPressureReport {
            estimated_tokens,
            context_window: budget.context_window,
            trigger_tokens: budget.trigger_tokens,
            effective_window: budget.effective_window,
            ratio,
            compacted_up_to_seq: session.header().compacted_up_to_seq,
            pruned_count: session.pruned_map().len(),
        }
    }

    /// 手动触发压缩（供 TUI `/compact [focus]` 或宿主 API 调用），支持 `CancellationToken` 中断。
    pub async fn compact_now(
        &self,
        session: Arc<Session>,
        opts: ManualCompactOptions,
        cancel: CancellationToken,
    ) -> Result<CompactionReport, CompactionError> {
        if !self.config().enabled {
            return Ok(CompactionReport::skipped("compaction is disabled"));
        }
        self.run_compaction_pipeline(
            session,
            CompactionTrigger::Manual,
            opts.provider,
            opts.model,
            opts.focus,
            false,
            0,
            cancel,
        )
        .await
    }

    async fn handle_pre_step(&self, ev: &AgentPreStepInput) {
        let cfg = self.config();
        if !cfg.enabled || (!cfg.auto_compact && !cfg.auto_prune) {
            return;
        }
        let budget = self.compute_session_budget(ev.provider.as_deref(), ev.model.as_deref(), &cfg);
        let (sys_prompt, _, tools_chars) = self.current_prompt_and_tools(&ev.session.id);
        let pending_tokens: u64 = ev
            .inputs
            .iter()
            .map(|inp| estimate_text_tokens(inp.text()))
            .sum();
        let estimated = estimate_session_context_tokens(&ev.session, &sys_prompt, tools_chars)
            .saturating_add(pending_tokens);

        let (last_compacted_tokens, idle_secs) = {
            let guard = self.sessions_meta.lock().unwrap();
            let meta = guard.get(&ev.session.id);
            let last_tok = meta.and_then(|m| m.last_compacted_tokens);
            let idle = meta
                .and_then(|m| m.last_activity_at)
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(0);
            (last_tok, idle)
        };

        let regrowth_ok = last_compacted_tokens
            .is_none_or(|prev_tok| estimated.saturating_sub(prev_tok) >= budget.regrowth_tokens);
        let cold_cache_threshold = (budget.trigger_tokens / 2).max(budget.keep_recent_tokens);
        let cold_cache_eligible = cfg.cold_cache_idle_secs > 0
            && idle_secs >= cfg.cold_cache_idle_secs
            && estimated >= cold_cache_threshold
            && regrowth_ok;

        let trigger = if estimated >= budget.trigger_tokens {
            let urgent = estimated >= budget.urgent_tokens;
            if !urgent && !regrowth_ok {
                return;
            }
            CompactionTrigger::AutoThreshold
        } else if cold_cache_eligible {
            CompactionTrigger::ColdCache
        } else {
            return;
        };

        let _ = self
            .run_compaction_pipeline(
                Arc::clone(&ev.session),
                trigger,
                ev.provider.clone(),
                ev.model.clone(),
                None,
                false,
                ev.inputs.len(),
                ev.cancel.clone(),
            )
            .await;
    }

    async fn handle_overflow(&self, ev: &AgentRequestErrorInput) -> bool {
        if !self.config().enabled {
            return false;
        }
        match self
            .run_compaction_pipeline(
                Arc::clone(&ev.session),
                CompactionTrigger::Overflow,
                ev.provider.clone(),
                ev.model.clone(),
                None,
                true,
                0,
                ev.cancel.clone(),
            )
            .await
        {
            Ok(report) => matches!(
                report.outcome,
                CompactionOutcome::PrunedOnly | CompactionOutcome::Compacted
            ),
            Err(_) => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_compaction_pipeline(
        &self,
        session: Arc<Session>,
        trigger: CompactionTrigger,
        provider: Option<String>,
        model: Option<String>,
        focus: Option<String>,
        force_overflow: bool,
        pending_surface_count: usize,
        cancel: CancellationToken,
    ) -> Result<CompactionReport, CompactionError> {
        let _lock = self.try_lock_session(&session.id)?;
        let cfg = self.config();

        // 路由选择：优先采用本会话最近真正成功答复过的完整 `(provider, model)`，禁止跨 provider 混拼 model
        let (effective_provider, effective_model) = {
            let answered = self
                .sessions_meta
                .lock()
                .unwrap()
                .get(&session.id)
                .and_then(|m| m.last_answered_route.clone());
            match answered {
                Some((Some(ap), am)) => {
                    let matched_model = am.or_else(|| {
                        if provider.as_deref() == Some(ap.as_str()) {
                            model
                        } else {
                            None
                        }
                    });
                    (Some(ap), matched_model)
                }
                Some((None, Some(am))) => (provider, Some(am)),
                _ => (provider, model),
            }
        };

        let budget = self.compute_session_budget(
            effective_provider.as_deref(),
            effective_model.as_deref(),
            &cfg,
        );
        let (sys_prompt, declared_tools, tools_chars) = self.current_prompt_and_tools(&session.id);
        let tokens_before = estimate_session_context_tokens(&session, &sys_prompt, tools_chars);

        struct CompactionTraceGuard {
            ctx: Context,
            session: Arc<Session>,
            trigger: CompactionTrigger,
            tokens_before: u64,
            finished: bool,
        }
        impl Drop for CompactionTraceGuard {
            fn drop(&mut self) {
                if !self.finished {
                    self.session.append_trace(SessionTraceEvent::CompactionEnd {
                        status: "failed".to_string(),
                        first_kept_seq: None,
                        tokens_before: Some(self.tokens_before),
                        tokens_after: Some(self.tokens_before),
                    });
                    self.ctx.emit(
                        EVENT_COMPACTION_END,
                        CompactionEndEvent {
                            session_id: self.session.id.clone(),
                            trigger: self.trigger,
                            outcome: CompactionOutcome::Failed,
                            tokens_before: self.tokens_before,
                            tokens_after: self.tokens_before,
                            pruned_count: 0,
                            first_kept_seq: None,
                            summary_preview: None,
                            skipped_reason: Some(
                                "compaction aborted before completion".to_string(),
                            ),
                        },
                    );
                }
            }
        }

        // 发送开始审计与 UI 事件
        session.append_trace(SessionTraceEvent::CompactionStart {
            trigger: trigger.as_str().to_string(),
        });
        self.ctx.emit(
            EVENT_COMPACTION_START,
            CompactionStartEvent {
                session_id: session.id.clone(),
                trigger,
                tokens_before,
                context_window: budget.context_window,
            },
        );
        let mut trace_guard = CompactionTraceGuard {
            ctx: self.ctx.clone(),
            session: Arc::clone(&session),
            trigger,
            tokens_before,
            finished: false,
        };

        // --- L1：零模型批量剪枝 ---
        let mut pruned_count = 0usize;
        if cfg.auto_prune {
            let prune_plan =
                plan_tool_result_prune_with_pending(&session, &cfg, pending_surface_count);
            if !prune_plan.items.is_empty() {
                pruned_count = prune_plan.items.len();
                session.append_prune(prune_plan.items);
                self.ctx.emit(
                    EVENT_COMPACTION_PROGRESS,
                    CompactionProgressEvent {
                        session_id: session.id.clone(),
                        stage: "pruned".to_string(),
                        delta_chars: 0,
                    },
                );
            }
        }

        let tokens_after_prune =
            estimate_session_context_tokens(&session, &sys_prompt, tools_chars);

        // 若自动触发（非手动 `/compact`）且 L1 剪枝后已经压回安全线以下，则零模型成本直接结束！
        if pruned_count > 0
            && !matches!(trigger, CompactionTrigger::Manual)
            && !force_overflow
            && tokens_after_prune <= budget.prune_target_tokens
        {
            self.record_post_compaction_meta(&session.id, tokens_after_prune);
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: "pruned_only".to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_after_prune),
            });
            let report = CompactionReport {
                outcome: CompactionOutcome::PrunedOnly,
                trigger,
                tokens_before,
                tokens_after: tokens_after_prune,
                pruned_count,
                first_kept_seq: None,
                summary_preview: None,
                skipped_reason: None,
            };
            self.emit_end_event(&session.id, &report);
            trace_guard.finished = true;
            return Ok(report);
        }

        // 若禁用了自动摘要且本次是自动触发，则止步于 L1
        if !cfg.auto_compact && !matches!(trigger, CompactionTrigger::Manual) && !force_overflow {
            let report = if pruned_count > 0 {
                CompactionReport {
                    outcome: CompactionOutcome::PrunedOnly,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: None,
                }
            } else {
                CompactionReport::skipped("auto_compact disabled")
            };
            self.record_post_compaction_meta(&session.id, tokens_after_prune);
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: report.outcome.as_str().to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_after_prune),
            });
            self.emit_end_event(&session.id, &report);
            trace_guard.finished = true;
            return Ok(report);
        }

        // --- L2：切分规划（支持 1 次 Re-plan 以满足 `<= 0.5 * B` 硬上界）---
        let visible = session.visible_log();
        let pruned_map = session.pruned_map();
        let last_seq = session.last_seq();

        let keep_target = match trigger {
            CompactionTrigger::Manual => (budget.keep_recent_tokens / 2).max(512),
            CompactionTrigger::Overflow => 0,
            CompactionTrigger::AutoThreshold | CompactionTrigger::ColdCache => {
                budget.keep_recent_tokens
            }
        };

        let cut_opt =
            plan_compaction_cut(&visible, &pruned_map, last_seq, keep_target, force_overflow)
                .or_else(|| {
                    // 手动 `/compact` 时，即使总长度尚小于常规保留预算，也收紧为仅保留最新 1 条合法边界（对齐 `dim` 手动压缩语义）
                    if matches!(trigger, CompactionTrigger::Manual) {
                        plan_compaction_cut(&visible, &pruned_map, last_seq, 0, true)
                    } else {
                        None
                    }
                });

        let Some(mut cut) = cut_opt else {
            let report = if pruned_count > 0 {
                CompactionReport {
                    outcome: CompactionOutcome::PrunedOnly,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: None,
                }
            } else {
                CompactionReport {
                    outcome: CompactionOutcome::Skipped,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count: 0,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: Some("当前历史较短，暂无可压缩段落".to_string()),
                }
            };
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: report.outcome.as_str().to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_after_prune),
            });
            self.emit_end_event(&session.id, &report);
            trace_guard.finished = true;
            return Ok(report);
        };

        let llm = match self.ctx.llm() {
            Ok(llm) => llm,
            Err(err) => {
                session.append_trace(SessionTraceEvent::CompactionEnd {
                    status: "failed".to_string(),
                    first_kept_seq: None,
                    tokens_before: Some(tokens_before),
                    tokens_after: Some(tokens_after_prune),
                });
                let err_report = CompactionReport {
                    outcome: CompactionOutcome::Failed,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: Some(err.to_string()),
                };
                self.emit_end_event(&session.id, &err_report);
                trace_guard.finished = true;
                return Err(err.into());
            }
        };
        let current_visible_messages = session.derive_messages();
        let ctx_for_progress = self.ctx.clone();
        let sid_for_progress = session.id.clone();

        // L3 溢出场景当前前缀已爆窗，禁止复用前缀，直接走纯净转写请求
        let reuse_prefix = cfg.reuse_prefix_cache && !force_overflow;

        let summary_res = generate_checkpoint_summary(
            SummarizeParams {
                llm: &llm,
                session_id: Some(session.id.clone()),
                provider: effective_provider.clone(),
                model: effective_model.clone(),
                system_prompt: sys_prompt.clone(),
                declared_tools,
                current_visible_messages,
                cut: &cut,
                focus: focus.as_deref(),
                reuse_prefix_cache: reuse_prefix,
                disable_reasoning: cfg.disable_summary_reasoning,
                cancel: cancel.clone(),
            },
            |stage, delta_chars| {
                ctx_for_progress.emit(
                    EVENT_COMPACTION_PROGRESS,
                    CompactionProgressEvent {
                        session_id: sid_for_progress.clone(),
                        stage: stage.to_string(),
                        delta_chars,
                    },
                );
            },
        )
        .await;

        let summary_out = match summary_res {
            Ok(out) => out,
            Err(err) => {
                session.append_trace(SessionTraceEvent::CompactionEnd {
                    status: "failed".to_string(),
                    first_kept_seq: None,
                    tokens_before: Some(tokens_before),
                    tokens_after: Some(tokens_after_prune),
                });
                let err_report = CompactionReport {
                    outcome: CompactionOutcome::Failed,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: Some(err.to_string()),
                };
                self.emit_end_event(&session.id, &err_report);
                trace_guard.finished = true;
                return Err(CompactionError::Llm(err));
            }
        };

        // 构造确定性硬事实 `<mizar-digest>`
        let mut digest = build_compaction_digest(
            &cut.previous_digest,
            &cut.compacted_entries,
            cut.pinned_inputs.clone(),
        );
        let rendered_checkpoint =
            mizar_session::render_compaction_message(&summary_out.summary, &digest);
        let checkpoint_tokens = estimate_text_tokens(&rendered_checkpoint);
        let resident_overhead =
            estimate_text_tokens(&sys_prompt) + (tools_chars as u64).div_ceil(4);
        let projected_after = resident_overhead + checkpoint_tokens + cut.kept_tokens;

        // `dim` 压后 `<= 0.5 * B` 1 次 Re-plan：若实测摘要 + 尾部超标，收紧尾部重切一次（不重调 LLM）
        if !force_overflow
            && projected_after > budget.post_compact_max_tokens
            && cut.kept_tokens > 0
        {
            let allowed_tail = budget
                .post_compact_max_tokens
                .saturating_sub(resident_overhead + checkpoint_tokens);
            if let Some(tighter) = plan_compaction_cut(
                &visible,
                &pruned_map,
                last_seq,
                allowed_tail,
                allowed_tail == 0,
            ) {
                digest = build_compaction_digest(
                    &tighter.previous_digest,
                    &tighter.compacted_entries,
                    tighter.pinned_inputs.clone(),
                );
                cut = tighter;
            }
        }

        // 膨胀保护（`shrinks`）：自动触发时若摘要反而比被折叠内容更长，则放弃写入避免负优化
        if !matches!(trigger, CompactionTrigger::Manual)
            && !force_overflow
            && checkpoint_tokens >= cut.compacted_tokens
        {
            let report = if pruned_count > 0 {
                CompactionReport {
                    outcome: CompactionOutcome::PrunedOnly,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: None,
                }
            } else {
                CompactionReport {
                    outcome: CompactionOutcome::Skipped,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_after_prune,
                    pruned_count: 0,
                    first_kept_seq: None,
                    summary_preview: None,
                    skipped_reason: Some("摘要未能净缩减上下文占用，已跳过本次压缩".to_string()),
                }
            };
            self.record_post_compaction_meta(&session.id, tokens_after_prune);
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: report.outcome.as_str().to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_after_prune),
            });
            self.emit_end_event(&session.id, &report);
            trace_guard.finished = true;
            return Ok(report);
        }

        // 落盘 Compaction 检查点
        let tokens_after_est = resident_overhead + checkpoint_tokens + cut.kept_tokens;

        // 压后联动：清空本会话临时激活的延迟工具（recall 会由其事件监听重新激活）与旧 read-dedup 缓存
        if let Ok(tools_rt) = self.ctx.tools() {
            tools_rt.reset_session_activations(&session.id);
        }
        self.file_guards.on_compaction(&session.id);

        session.compact_at(
            summary_out.summary.clone(),
            cut.first_kept_seq,
            Some(tokens_before),
            Some(tokens_after_est),
            digest,
        );

        let tokens_after_actual =
            estimate_session_context_tokens(&session, &sys_prompt, tools_chars);
        self.record_post_compaction_meta(&session.id, tokens_after_actual);

        session.append_trace(SessionTraceEvent::CompactionEnd {
            status: "compacted".to_string(),
            first_kept_seq: Some(cut.first_kept_seq),
            tokens_before: Some(tokens_before),
            tokens_after: Some(tokens_after_actual),
        });

        let preview = build_summary_preview(&summary_out.summary, 280);
        let report = CompactionReport {
            outcome: CompactionOutcome::Compacted,
            trigger,
            tokens_before,
            tokens_after: tokens_after_actual,
            pruned_count,
            first_kept_seq: Some(cut.first_kept_seq),
            summary_preview: Some(preview),
            skipped_reason: None,
        };
        self.emit_end_event(&session.id, &report);
        trace_guard.finished = true;
        Ok(report)
    }

    fn record_post_compaction_meta(&self, session_id: &str, tokens_after: u64) {
        let mut guard = self.sessions_meta.lock().unwrap();
        let entry = guard.entry(session_id.to_string()).or_default();
        entry.last_compacted_tokens = Some(tokens_after);
        entry.last_activity_at = Some(Instant::now());
    }

    fn emit_end_event(&self, session_id: &str, report: &CompactionReport) {
        self.ctx.emit(
            EVENT_COMPACTION_END,
            CompactionEndEvent {
                session_id: session_id.to_string(),
                trigger: report.trigger,
                outcome: report.outcome,
                tokens_before: report.tokens_before,
                tokens_after: report.tokens_after,
                pruned_count: report.pruned_count,
                first_kept_seq: report.first_kept_seq,
                summary_preview: report.summary_preview.clone(),
                skipped_reason: report.skipped_reason.clone(),
            },
        );
    }
}

fn build_summary_preview(summary: &str, max_chars: usize) -> String {
    let cleaned = summary.trim();
    let count = cleaned.chars().count();
    if count <= max_chars {
        return cleaned.to_string();
    }
    let head: String = cleaned.chars().take(max_chars).collect();
    format!("{head}…")
}

impl Service for CompactionRuntime {
    type Config = CompactionConfig;

    fn service_name() -> &'static str {
        "compaction"
    }

    fn plugin_name() -> &'static str {
        "CompactionRuntime"
    }

    fn inject() -> Inject {
        Inject::from(["llm", "sessions", "tools", "systemPrompt"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: Arc::new(Mutex::new(config)),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            sessions_meta: Arc::new(Mutex::new(HashMap::new())),
            file_guards: Arc::new(FileGuardTracker::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            // 1. 步前自动检查水位（L1 剪枝 -> L2 摘要）
            {
                let rt = Arc::clone(&self);
                let c = ctx.clone();
                ctx.effect("compaction.pre-step", move || {
                    let h = c.on_waterfall_with(
                        EVENT_AGENT_PRE_STEP,
                        move |input: AgentPreStepInput,
                              next: Next<
                            AgentPreStepInput,
                            BoxFuture<'static, AgentPreStepDecision>,
                        >| {
                            let rt = Arc::clone(&rt);
                            Box::pin(async move {
                                if input.cancel.is_cancelled() {
                                    return AgentPreStepDecision::Reject {
                                        reason: "cancelled".to_string(),
                                    };
                                }
                                rt.handle_pre_step(&input).await;
                                next.call(input).await
                            })
                                as BoxFuture<'static, AgentPreStepDecision>
                        },
                        cordis::EventOptions {
                            priority: -20,
                            ..Default::default()
                        },
                    )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 2. 步后记录 answeredRoute 与活跃时间戳（供冷缓存窗口判断）
            {
                let meta = Arc::clone(&self.sessions_meta);
                let c = ctx.clone();
                ctx.effect("compaction.step-end", move || {
                    let h = c.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
                        let mut guard = meta.lock().unwrap();
                        let entry = guard.entry(ev.session.id.clone()).or_default();
                        entry.last_activity_at = Some(Instant::now());
                        if matches!(ev.outcome, StepOutcome::Completed | StepOutcome::ToolCalls)
                            && (ev.provider.is_some() || ev.model.is_some())
                        {
                            entry.last_answered_route =
                                Some((ev.provider.clone(), ev.model.clone()));
                        }
                    })?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 3. L3 溢出与隐性连续截断自动恢复
            {
                let rt = Arc::clone(&self);
                let c = ctx.clone();
                ctx.effect("compaction.request-error", move || {
                    let h = c.on_waterfall(
                        EVENT_AGENT_REQUEST_ERROR,
                        move |input: AgentRequestErrorInput,
                              next: Next<
                            AgentRequestErrorInput,
                            BoxFuture<'static, AgentRequestErrorDecision>,
                        >| {
                            let rt = Arc::clone(&rt);
                            Box::pin(async move {
                                if input.kind == AgentRequestErrorKind::ContextOverflow
                                    && rt.handle_overflow(&input).await
                                {
                                    return AgentRequestErrorDecision::Retry;
                                }
                                next.call(input).await
                            })
                                as BoxFuture<'static, AgentRequestErrorDecision>
                        },
                    )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 4. 压后 FileReads 整文件覆写拦截（拦 `write`，放行 `edit`）
            {
                let guards = Arc::clone(&self.file_guards);
                let cfg_ref = Arc::clone(&self.config);
                let c = ctx.clone();
                let c_inner = ctx.clone();
                ctx.effect("compaction.pre-execute-guard", move || {
                    let h =
                        c.on_waterfall(
                            EVENT_TOOLS_PRE_EXECUTE,
                            move |exec: ToolExecution,
                                  next: Next<
                                ToolExecution,
                                BoxFuture<'static, PreExecuteDecision>,
                            >| {
                                let guards = Arc::clone(&guards);
                                let cfg_ref = Arc::clone(&cfg_ref);
                                let ctx_ref = c_inner.clone();
                                Box::pin(async move {
                                    let enable_write_guard =
                                        cfg_ref.lock().unwrap().enable_write_after_compact_guard;
                                    if let Ok(sessions) = ctx_ref.sessions() {
                                        let decision = guards.check_pre_execute(
                                            &exec,
                                            &sessions,
                                            enable_write_guard,
                                        );
                                        if !decision.allow {
                                            return decision;
                                        }
                                    }
                                    next.call(exec).await
                                })
                                    as BoxFuture<'static, PreExecuteDecision>
                            },
                        )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 5. 跨压缩感知 `read-dedup` 与文件读取/写入水位追踪
            {
                let guards = Arc::clone(&self.file_guards);
                let cfg_ref = Arc::clone(&self.config);
                let c = ctx.clone();
                let c_inner = ctx.clone();
                ctx.effect("compaction.post-execute-guard", move || {
                    let h = c.on_waterfall(
                        EVENT_TOOLS_POST_EXECUTE,
                        move |input: PostExecuteInput,
                              next: Next<PostExecuteInput, BoxFuture<'static, ToolResult>>| {
                            let guards = Arc::clone(&guards);
                            let cfg_ref = Arc::clone(&cfg_ref);
                            let ctx_ref = c_inner.clone();
                            Box::pin(async move {
                                let enable_dedup = cfg_ref.lock().unwrap().enable_read_dedup;
                                let exec = input.exec.clone();
                                let res = next.call(input).await;
                                if let Ok(sessions) = ctx_ref.sessions() {
                                    guards.handle_post_execute(&exec, res, &sessions, enable_dedup)
                                } else {
                                    res
                                }
                            }) as BoxFuture<'static, ToolResult>
                        },
                    )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 6. 追踪会话真实落盘序号，对齐 FileGuardTracker 的 `last_read_seq` 与 `dedup` seq
            {
                let guards = Arc::clone(&self.file_guards);
                let c = ctx.clone();
                ctx.effect("compaction.session-event-seq", move || {
                    let h = c.on(
                        mizar_session::EVENT_SESSION_EVENT,
                        move |ev: &mizar_session::SessionEventPayload| {
                            if let mizar_session::SessionEvent::ToolResult {
                                tool_call_id,
                                is_error: false,
                                ..
                            } = &ev.entry.event
                            {
                                guards.reconcile_tool_result_seq(
                                    &ev.session.id,
                                    tool_call_id,
                                    ev.entry.seq,
                                );
                            }
                        },
                    )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            Ok(())
        })
    }
}

pub trait CompactionContextExt {
    fn compaction(&self) -> Result<Arc<CompactionRuntime>, CordisError>;
}

impl CompactionContextExt for Context {
    fn compaction(&self) -> Result<Arc<CompactionRuntime>, CordisError> {
        self.service::<CompactionRuntime>("compaction")
    }
}
