//! `CompactionRuntime` Cordis 服务（`"compaction"`）：
//! 统一编排 Cache-aware 尾部剪枝、单阶段检查点摘要、溢出兜底恢复、手动 `/compact`、
//! 上下文压力探测与统一 `write` 写前读守卫。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Next, Service};
use mizar_agent_loop::{
    AgentContextOverflowEvent, AgentPreStepEvent, AgentStepEndEvent, EVENT_AGENT_CONTEXT_OVERFLOW,
    EVENT_AGENT_PRE_STEP, EVENT_AGENT_STEP_END, StepOutcome,
};
use mizar_llm::{LlmContextExt, LlmError, LlmErrorCode, Message};
use mizar_session::{Session, SessionContextExt, SessionEvent, SessionTraceEvent};
use mizar_system_prompt::SystemPromptContextExt;
use mizar_tools::{
    EVENT_TOOLS_POST_EXECUTE, EVENT_TOOLS_PRE_EXECUTE, PostExecuteInput, PreExecuteDecision,
    ToolExecution, ToolResult, ToolsContextExt,
};
use tokio_util::sync::CancellationToken;

use crate::budget::{
    ContextBudget, compute_budget, estimate_message_tokens, estimate_projected_tokens,
    estimate_session_context_tokens, estimate_text_tokens,
};
use crate::digest::{build_compaction_digest, plan_compaction_cut};
use crate::guards::FileGuardTracker;
use crate::pruning::{TailPruneOptions, TailPruneReport, prune_session_tail_cancellable};
use crate::summarizer::{SummarizeParams, generate_checkpoint_summary};
use crate::{
    CompactionConfig, CompactionEndEvent, CompactionError, CompactionOutcome,
    CompactionProgressEvent, CompactionReport, CompactionStartEvent, CompactionTrigger,
    ContextPressureReport, EVENT_COMPACTION_END, EVENT_COMPACTION_PROGRESS, EVENT_COMPACTION_START,
    ManualCompactOptions,
};

#[derive(Debug, Default)]
struct SessionCompactionMeta {
    last_answered_route: Option<(Option<String>, Option<String>)>,
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

struct CompactionTraceGuard<'a> {
    runtime: &'a CompactionRuntime,
    session: &'a Arc<Session>,
    report: CompactionReport,
}

impl Drop for CompactionTraceGuard<'_> {
    fn drop(&mut self) {
        let report = &self.report;
        self.session.append_trace(SessionTraceEvent::CompactionEnd {
            status: report.outcome.as_str().to_string(),
            first_kept_seq: report.first_kept_seq,
            tokens_before: Some(report.tokens_before),
            tokens_after: Some(report.tokens_after),
        });
        self.runtime.emit_end_event(&self.session.id, report);
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

    /// 结合 `LlmRuntime::provider_infos()` 查表推导当前路由的协议与模型元数据；
    /// 当明确指定 `model` 但未在模型表中命中时不借用其它模型，返回 `None` 以便回退 `default_window`。
    fn resolve_provider_and_model(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> (Option<String>, Option<mizar_llm::ModelInfo>) {
        let Ok(llm) = self.ctx.llm() else {
            return (None, None);
        };
        let providers = llm.provider_infos();
        let found = match provider {
            Some(pid) => providers.iter().find(|p| p.id == pid).map(|p| {
                let m = match model {
                    Some(mid) => p.models.iter().find(|m| m.id == mid).cloned(),
                    None => p.models.first().cloned(),
                };
                (p.protocol.clone(), m)
            }),
            None => match model {
                Some(mid) => providers.iter().find_map(|p| {
                    p.models
                        .iter()
                        .find(|m| m.id == mid)
                        .map(|m| (p.protocol.clone(), Some(m.clone())))
                }),
                None => providers
                    .first()
                    .map(|p| (p.protocol.clone(), p.models.first().cloned())),
            },
        };
        match found {
            Some((proto, m)) => (Some(proto), m),
            None => (None, None),
        }
    }

    fn resolve_model_limits(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        default_window: u64,
    ) -> (u64, Option<u32>) {
        if let (_, Some(m)) = self.resolve_provider_and_model(provider, model) {
            return (m.context_window.unwrap_or(default_window), m.max_tokens);
        }
        (default_window, None)
    }

    fn is_prefix_bound_thinking(
        &self,
        session: &Session,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> bool {
        if let (Some(proto), Some(m)) = self.resolve_provider_and_model(provider, model)
            && proto == "anthropic-messages"
            && !m.reasoning_efforts.is_empty()
        {
            return true;
        }
        session.with_visible_entries(|entries| {
            entries.iter().any(|entry| {
                matches!(
                    &entry.event,
                    mizar_session::SessionEvent::Assistant { replay: Some(r), .. }
                        if r.protocol == "anthropic-messages"
                            && r.response.get("prefixDigest").is_some()
                            && r.blocks.iter().any(|b| {
                                b.get("signature").is_some() || b.get("redacted").is_some()
                            })
                )
            })
        })
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
            cancel,
        )
        .await
    }

    fn prune_tail(
        &self,
        session: &Arc<Session>,
        opts: &TailPruneOptions,
        cancel: &CancellationToken,
    ) -> TailPruneReport {
        let workspace = session
            .header()
            .cwd
            .or_else(|| self.ctx.sessions().ok().and_then(|s| s.default_cwd()));
        prune_session_tail_cancellable(session, workspace.as_deref(), opts, Some(cancel))
    }

    async fn handle_pre_step(&self, ev: &AgentPreStepEvent) {
        let cfg = self.config();
        if !cfg.enabled {
            return;
        }
        if (cfg.supersede_reads || cfg.drop_useless)
            && let Ok(_lock) = self.try_lock_session(&ev.session.id)
        {
            let opts = TailPruneOptions {
                supersede_reads: cfg.supersede_reads,
                drop_useless: cfg.drop_useless,
                warm_suffix_tokens: cfg.prune_warm_suffix_tokens,
                idle_flush_ms: cfg.prune_idle_flush_ms,
                is_prefix_bound_thinking: self.is_prefix_bound_thinking(
                    &ev.session,
                    ev.provider.as_deref(),
                    ev.model.as_deref(),
                ),
                now_ms: None,
            };
            self.prune_tail(&ev.session, &opts, &ev.cancel);
        }
        if !cfg.auto_compact {
            return;
        }
        let budget = self.compute_session_budget(ev.provider.as_deref(), ev.model.as_deref(), &cfg);
        let (sys_prompt, _, tools_chars) = self.current_prompt_and_tools(&ev.session.id);
        let estimated = estimate_session_context_tokens(&ev.session, &sys_prompt, tools_chars);

        if estimated < budget.trigger_tokens {
            return;
        }

        let _ = self
            .run_compaction_pipeline(
                Arc::clone(&ev.session),
                CompactionTrigger::AutoThreshold,
                ev.provider.clone(),
                ev.model.clone(),
                None,
                false,
                ev.cancel.clone(),
            )
            .await;
    }

    async fn handle_overflow(&self, ev: &AgentContextOverflowEvent) -> bool {
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
                ev.cancel.clone(),
            )
            .await
        {
            Ok(report) => {
                report.outcome == CompactionOutcome::Compacted
                    && report.tokens_after < report.tokens_before
                    && !ev.cancel.is_cancelled()
            }
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
        let resident_overhead =
            estimate_text_tokens(&sys_prompt) + (tools_chars as u64).div_ceil(4);
        // 剪枝会使上一次实测用量失效，溢出前后必须用同一套投影估算。
        let history_tokens = |first_kept_seq: Option<u64>| {
            let pruned = session.pruned_map();
            session.with_visible_entries(|entries| {
                let kept = entries.iter().copied().filter(|entry| {
                    first_kept_seq.is_none_or(|seq| {
                        entry.seq >= seq && !matches!(entry.event, SessionEvent::Compaction { .. })
                    })
                });
                estimate_projected_tokens(kept, &pruned, 0)
            })
        };
        let current_overhead = || {
            let (prompt, _, chars) = self.current_prompt_and_tools(&session.id);
            estimate_text_tokens(&prompt) + (chars as u64).div_ceil(4)
        };
        let tokens_before = if force_overflow {
            resident_overhead + history_tokens(None)
        } else {
            estimate_session_context_tokens(&session, &sys_prompt, tools_chars)
        };

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
        let mut trace = CompactionTraceGuard {
            runtime: self,
            session: &session,
            report: CompactionReport {
                outcome: CompactionOutcome::Failed,
                trigger,
                tokens_before,
                tokens_after: tokens_before,
                first_kept_seq: None,
                summary_preview: None,
                skipped_reason: Some("compaction aborted before completion".into()),
            },
        };
        let pruned = if force_overflow && (cfg.supersede_reads || cfg.drop_useless) {
            let opts = TailPruneOptions {
                supersede_reads: cfg.supersede_reads,
                drop_useless: cfg.drop_useless,
                warm_suffix_tokens: 0,
                // 溢出后没有可保留的热前缀，仅将热尾设为零会漏掉较早的候选。
                idle_flush_ms: 0,
                ..Default::default()
            };
            self.prune_tail(&session, &opts, &cancel).pruned_count > 0
        } else {
            false
        };
        if pruned {
            trace.report.tokens_after = resident_overhead + history_tokens(None);
        }
        let tokens_after_prune = trace.report.tokens_after;
        let mut without_checkpoint = |result: Result<&str, CompactionError>| {
            let report = &mut trace.report;
            if pruned {
                // 摘要等待期间历史和已激活的工具都可能增长，须以当前窗口判断能否恢复。
                report.tokens_after = current_overhead() + history_tokens(None);
            }
            let err = match result {
                // 服务端也能主动中止流，此时本地取消令牌可能仍未触发。
                Err(CompactionError::Llm(err)) if err.code == LlmErrorCode::Cancelled => err.into(),
                _ if cancel.is_cancelled() => LlmError::cancelled().into(),
                _ if tokens_after_prune < tokens_before && report.tokens_after < tokens_before => {
                    report.outcome = CompactionOutcome::Compacted;
                    report.skipped_reason = None;
                    return Ok(report.clone());
                }
                Ok(reason) => {
                    report.outcome = CompactionOutcome::Skipped;
                    report.skipped_reason = Some(reason.into());
                    return Ok(report.clone());
                }
                Err(err) => err,
            };
            report.skipped_reason = Some(err.to_string());
            Err(err)
        };
        if cancel.is_cancelled() {
            return without_checkpoint(Err(LlmError::cancelled().into()));
        }

        let last_seq = session.last_seq();
        let mut visible = session.visible_log();
        let mut pruned_map = session.pruned_map();
        // 干净转录也应看到剪枝后的内容，避免把刚移除的输出又塞回摘要请求。
        for entry in &mut visible {
            if let SessionEvent::ToolResult {
                content, images, ..
            } = &mut entry.event
                && let Some(notice) = pruned_map.remove(&entry.seq)
            {
                *content = notice;
                images.clear();
            }
        }

        let keep_target = match trigger {
            CompactionTrigger::Manual => (budget.keep_recent_tokens / 2).max(512),
            CompactionTrigger::Overflow => 0,
            CompactionTrigger::AutoThreshold => budget.keep_recent_tokens,
        };

        let cut_opt =
            plan_compaction_cut(&visible, last_seq, keep_target, force_overflow).or_else(|| {
                // 手动请求允许只留最新合法边界，否则短会话总是无法压缩。
                if matches!(trigger, CompactionTrigger::Manual) {
                    plan_compaction_cut(&visible, last_seq, 0, true)
                } else {
                    None
                }
            });
        let Some(cut) = cut_opt else {
            return without_checkpoint(Ok("当前历史较短，暂无可压缩段落"));
        };
        let llm = match self.ctx.llm() {
            Ok(llm) => llm,
            Err(err) => return without_checkpoint(Err(err.into())),
        };
        let ctx_for_progress = self.ctx.clone();
        let sid_for_progress = session.id.clone();

        // 溢出场景当前前缀已爆窗，禁止复用前缀，直接走纯净转写请求
        let reuse_prefix = !force_overflow;

        let summary_res = generate_checkpoint_summary(
            SummarizeParams {
                llm: &llm,
                session_id: Some(session.id.clone()),
                provider: effective_provider,
                model: effective_model,
                system_prompt: sys_prompt,
                declared_tools,
                current_visible_messages: if reuse_prefix {
                    session.derive_messages()
                } else {
                    Vec::new()
                },
                cut: &cut,
                focus: focus.as_deref(),
                reuse_prefix_cache: reuse_prefix,
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
            Err(err) => return without_checkpoint(Err(err.into())),
        };

        let digest = build_compaction_digest(
            &cut.previous_digest,
            &cut.compacted_entries,
            cut.pinned_inputs.clone(),
        );
        let rendered_checkpoint =
            mizar_session::render_compaction_message(&summary_out.summary, &digest);
        let checkpoint_tokens = estimate_message_tokens(&Message::user(rendered_checkpoint));
        let tokens_after_est = if force_overflow {
            // 等待摘要期间追加的消息也会留在窗口内。
            current_overhead() + checkpoint_tokens + history_tokens(Some(cut.first_kept_seq))
        } else {
            resident_overhead + checkpoint_tokens + cut.kept_tokens
        };
        if (force_overflow && tokens_after_est >= tokens_before.min(tokens_after_prune))
            || (trigger == CompactionTrigger::AutoThreshold
                && checkpoint_tokens >= cut.compacted_tokens)
        {
            return without_checkpoint(Ok("摘要未能净缩减上下文占用，已跳过本次压缩"));
        }
        if cancel.is_cancelled() {
            return without_checkpoint(Err(LlmError::cancelled().into()));
        }

        session.compact_at(
            summary_out.summary.clone(),
            cut.first_kept_seq,
            Some(tokens_before),
            Some(tokens_after_est),
            digest,
        );
        if let Ok(tools) = self.ctx.tools() {
            tools.reset_session_activations(&session.id);
        }
        let tokens_after_actual = current_overhead() + history_tokens(None);

        trace.report.outcome = CompactionOutcome::Compacted;
        trace.report.tokens_after = tokens_after_actual;
        trace.report.first_kept_seq = Some(cut.first_kept_seq);
        trace.report.summary_preview = Some(build_summary_preview(&summary_out.summary, 280));
        trace.report.skipped_reason = None;
        Ok(trace.report.clone())
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
            // 1. 步前自动执行 cache-aware 尾部剪枝与水位检查（priority = -20）
            {
                let rt = Arc::clone(&self);
                let c = ctx.clone();
                ctx.effect("compaction.pre-step", move || {
                    let h = c.on_serial_with::<AgentPreStepEvent, (), _, _>(
                        EVENT_AGENT_PRE_STEP,
                        move |ev: AgentPreStepEvent| {
                            let rt = Arc::clone(&rt);
                            async move {
                                rt.handle_pre_step(&ev).await;
                                None
                            }
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

            // 2. 步后记录 answeredRoute
            {
                let meta = Arc::clone(&self.sessions_meta);
                let c = ctx.clone();
                ctx.effect("compaction.step-end", move || {
                    let h = c.on(EVENT_AGENT_STEP_END, move |ev: &AgentStepEndEvent| {
                        if matches!(ev.outcome, StepOutcome::Completed | StepOutcome::ToolCalls)
                            && (ev.provider.is_some() || ev.model.is_some())
                        {
                            let mut guard = meta.lock().unwrap();
                            let entry = guard.entry(ev.session.id.clone()).or_default();
                            entry.last_answered_route =
                                Some((ev.provider.clone(), ev.model.clone()));
                        }
                    })?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 3. 溢出与隐性连续截断自动恢复
            {
                let rt = Arc::clone(&self);
                let c = ctx.clone();
                ctx.effect("compaction.overflow", move || {
                    let h = c.on_serial::<AgentContextOverflowEvent, bool, _, _>(
                        EVENT_AGENT_CONTEXT_OVERFLOW,
                        move |ev: AgentContextOverflowEvent| {
                            let rt = Arc::clone(&rt);
                            async move { Some(rt.handle_overflow(&ev).await) }
                        },
                    )?;
                    Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                        h.dispose().await;
                    })))
                })?;
            }

            // 4. 统一 `write` 写前读守卫（>50 行首读要求 + 压后陈旧整文件覆写拦截）
            {
                let guards = Arc::clone(&self.file_guards);
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
                                let ctx_ref = c_inner.clone();
                                Box::pin(async move {
                                    if let Ok(sessions) = ctx_ref.sessions() {
                                        let decision = guards.check_pre_execute(&exec, &sessions);
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

            // 5. 追踪成功 `read`/`write` 的文件可见序号
            {
                let guards = Arc::clone(&self.file_guards);
                let c = ctx.clone();
                let c_inner = ctx.clone();
                ctx.effect("compaction.post-execute-guard", move || {
                    let h = c.on_waterfall(
                        EVENT_TOOLS_POST_EXECUTE,
                        move |input: PostExecuteInput,
                              next: Next<PostExecuteInput, BoxFuture<'static, ToolResult>>| {
                            let guards = Arc::clone(&guards);
                            let ctx_ref = c_inner.clone();
                            Box::pin(async move {
                                let exec = input.exec.clone();
                                let res = next.call(input).await;
                                if let Ok(sessions) = ctx_ref.sessions() {
                                    guards.handle_post_execute(&exec, res, &sessions)
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

            // 6. 追踪会话真实落盘序号，对齐 FileGuardTracker 的 `last_read_seq`
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
