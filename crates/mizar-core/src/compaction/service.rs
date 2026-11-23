//! `CompactionRuntime` Cordis 服务（`"compaction"`）：
//! 统一编排单阶段检查点摘要、溢出兜底恢复、手动 `/compact`、上下文压力探测与 `write` 写前读守卫。

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::agent_loop::{
    AgentEvent, AgentPreStepDecision, AgentPreStepInput, AgentRequestErrorDecision,
    AgentRequestErrorInput, AgentRequestErrorKind, EVENT_AGENT_PRE_STEP, EVENT_AGENT_REQUEST_ERROR,
    StepOutcome, subscribe_agent_events_when_ready,
};
use crate::system_prompt::SystemPromptContextExt;
use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Next, Service};
use mizar_llm::LlmContextExt;
use mizar_session::{Session, SessionContextExt, SessionTraceEvent};
use mizar_tools::{
    EVENT_TOOLS_POST_EXECUTE, EVENT_TOOLS_PRE_EXECUTE, PostExecuteInput, PreExecuteDecision,
    ToolExecution, ToolResult, ToolsContextExt,
};
use tokio_util::sync::CancellationToken;

use crate::compaction::budget::{
    ContextBudget, compute_budget, estimate_session_context_tokens, estimate_text_tokens,
};
use crate::compaction::digest::{build_compaction_digest, plan_compaction_cut};
use crate::compaction::guards::FileGuardTracker;
use crate::compaction::summarizer::{SummarizeParams, generate_checkpoint_summary};
use crate::compaction::{
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

pub struct CompactionRuntime {
    ctx: Context,
    config: Arc<Mutex<CompactionConfig>>,
    module_enabled: AtomicBool,
    in_flight: Arc<Mutex<HashSet<String>>>,
    sessions_meta: Arc<Mutex<HashMap<String, SessionCompactionMeta>>>,
    file_guards: Arc<FileGuardTracker>,
}

impl CompactionRuntime {
    pub fn is_module_enabled(&self) -> bool {
        self.module_enabled.load(Ordering::Acquire)
    }

    pub fn set_module_enabled(&self, enabled: bool) {
        self.module_enabled.store(enabled, Ordering::Release);
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

    /// 结合 `LlmRuntime::provider_infos()` 查表推导当前路由的模型窗口与最大输出；
    /// 当明确指定 `model` 但未在模型表中命中时直接回退 `default_window`，不借用其它模型窗口。
    fn resolve_model_limits(
        &self,
        provider: Option<&str>,
        model: Option<&str>,
        default_window: u64,
    ) -> (u64, Option<u32>) {
        if let Ok(llm) = self.ctx.llm() {
            let providers = llm.provider_infos();
            let chosen_model = match provider {
                Some(pid) => providers
                    .iter()
                    .find(|p| p.id == pid)
                    .and_then(|p| match model {
                        Some(mid) => p.models.iter().find(|m| m.id == mid),
                        None => p.models.first(),
                    }),
                None => match model {
                    Some(mid) => providers
                        .iter()
                        .find_map(|p| p.models.iter().find(|m| m.id == mid)),
                    None => providers.first().and_then(|p| p.models.first()),
                },
            };
            if let Some(m) = chosen_model {
                return (m.context_window.unwrap_or(default_window), m.max_tokens);
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
            .map(|sp| sp.build_for_session(Some(session_id)))
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
        if !self.is_module_enabled() || !self.config().enabled {
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

    async fn handle_pre_step(&self, ev: &AgentPreStepInput) {
        if !self.is_module_enabled() {
            return;
        }
        let cfg = self.config();
        if !cfg.enabled || !cfg.auto_compact {
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

    async fn handle_overflow(&self, ev: &AgentRequestErrorInput) -> bool {
        if !self.is_module_enabled() || !self.config().enabled {
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
            Ok(report) => matches!(report.outcome, CompactionOutcome::Compacted),
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

        // 切分规划（单阶段检查点，不再做 L1 剪枝与 Re-plan 二次硬切）
        let visible = session.visible_log();
        let last_seq = session.last_seq();

        let keep_target = match trigger {
            CompactionTrigger::Manual => (budget.keep_recent_tokens / 2).max(512),
            CompactionTrigger::Overflow => 0,
            CompactionTrigger::AutoThreshold => budget.keep_recent_tokens,
        };

        let cut_opt =
            plan_compaction_cut(&visible, last_seq, keep_target, force_overflow).or_else(|| {
                if matches!(trigger, CompactionTrigger::Manual) {
                    plan_compaction_cut(&visible, last_seq, 0, true)
                } else {
                    None
                }
            });

        let Some(cut) = cut_opt else {
            let report = CompactionReport {
                outcome: CompactionOutcome::Skipped,
                trigger,
                tokens_before,
                tokens_after: tokens_before,
                first_kept_seq: None,
                summary_preview: None,
                skipped_reason: Some("当前历史较短，暂无可压缩段落".to_string()),
            };
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: report.outcome.as_str().to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_before),
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
                    tokens_after: Some(tokens_before),
                });
                let err_report = CompactionReport {
                    outcome: CompactionOutcome::Failed,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_before,
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

        // 溢出场景当前前缀已爆窗，禁止复用前缀，直接走纯净转写请求
        let reuse_prefix = !force_overflow;

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
                    tokens_after: Some(tokens_before),
                });
                let err_report = CompactionReport {
                    outcome: CompactionOutcome::Failed,
                    trigger,
                    tokens_before,
                    tokens_after: tokens_before,
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
        let digest = build_compaction_digest(
            &cut.previous_digest,
            &cut.compacted_entries,
            cut.pinned_inputs.clone(),
        );
        let rendered_checkpoint =
            mizar_session::render_compaction_message(&summary_out.summary, &digest);
        let checkpoint_tokens = estimate_text_tokens(&rendered_checkpoint);
        let resident_overhead =
            estimate_text_tokens(&sys_prompt) + (tools_chars as u64).div_ceil(4);

        // 膨胀保护（`shrinks`）：自动触发时若摘要反而比被折叠内容更长，则放弃写入避免负优化
        if !matches!(trigger, CompactionTrigger::Manual)
            && !force_overflow
            && checkpoint_tokens >= cut.compacted_tokens
        {
            let report = CompactionReport {
                outcome: CompactionOutcome::Skipped,
                trigger,
                tokens_before,
                tokens_after: tokens_before,
                first_kept_seq: None,
                summary_preview: None,
                skipped_reason: Some("摘要未能净缩减上下文占用，已跳过本次压缩".to_string()),
            };
            session.append_trace(SessionTraceEvent::CompactionEnd {
                status: report.outcome.as_str().to_string(),
                first_kept_seq: None,
                tokens_before: Some(tokens_before),
                tokens_after: Some(tokens_before),
            });
            self.emit_end_event(&session.id, &report);
            trace_guard.finished = true;
            return Ok(report);
        }

        // 落盘 Compaction 检查点
        let tokens_after_est = resident_overhead + checkpoint_tokens + cut.kept_tokens;

        // 压后联动：清空本会话临时激活的延迟工具（recall 会由其事件监听重新激活）
        if let Ok(tools_rt) = self.ctx.tools() {
            tools_rt.reset_session_activations(&session.id);
        }

        session.compact_at(
            summary_out.summary.clone(),
            cut.first_kept_seq,
            Some(tokens_before),
            Some(tokens_after_est),
            digest,
        );

        let tokens_after_actual =
            estimate_session_context_tokens(&session, &sys_prompt, tools_chars);

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
            first_kept_seq: Some(cut.first_kept_seq),
            summary_preview: Some(preview),
            skipped_reason: None,
        };
        self.emit_end_event(&session.id, &report);
        trace_guard.finished = true;
        Ok(report)
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
            module_enabled: AtomicBool::new(true),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            sessions_meta: Arc::new(Mutex::new(HashMap::new())),
            file_guards: Arc::new(FileGuardTracker::new()),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            // 1. 步前自动检查水位并触发单阶段检查点压缩
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

            // 2. 步后记录最近真正成功答复过的 `(provider, model)` 路由
            {
                let rt_step = Arc::clone(&self);
                let meta = Arc::clone(&self.sessions_meta);
                subscribe_agent_events_when_ready(&ctx, "compaction.step-end", move |ev| {
                    if !rt_step.is_module_enabled() {
                        return;
                    }
                    if let AgentEvent::StepEnd(ev) = ev
                        && matches!(ev.outcome, StepOutcome::Completed | StepOutcome::ToolCalls)
                        && (ev.provider.is_some() || ev.model.is_some())
                    {
                        let mut guard = meta.lock().unwrap();
                        let entry = guard.entry(ev.session.id.clone()).or_default();
                        entry.last_answered_route = Some((ev.provider.clone(), ev.model.clone()));
                    }
                })
                .await?;
            }

            // 3. 溢出错误自动压缩并重试恢复
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

            // 4. 统一 `write` 写前读守卫（>50 行首读要求 + 压后陈旧整文件覆写拦截）
            {
                let rt_pre = Arc::clone(&self);
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
                                let rt_pre = Arc::clone(&rt_pre);
                                let guards = Arc::clone(&guards);
                                let ctx_ref = c_inner.clone();
                                Box::pin(async move {
                                    if !rt_pre.is_module_enabled() {
                                        return next.call(exec).await;
                                    }
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

            // 5. 追踪成功 `read`/`write` 后的文件可见状态
            {
                let rt_post = Arc::clone(&self);
                let guards = Arc::clone(&self.file_guards);
                let c = ctx.clone();
                let c_inner = ctx.clone();
                ctx.effect("compaction.post-execute-guard", move || {
                    let h = c.on_waterfall(
                        EVENT_TOOLS_POST_EXECUTE,
                        move |input: PostExecuteInput,
                              next: Next<PostExecuteInput, BoxFuture<'static, ToolResult>>| {
                            let rt_post = Arc::clone(&rt_post);
                            let guards = Arc::clone(&guards);
                            let ctx_ref = c_inner.clone();
                            Box::pin(async move {
                                if !rt_post.is_module_enabled() {
                                    return next.call(input).await;
                                }
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
                let rt_seq = Arc::clone(&self);
                let guards = Arc::clone(&self.file_guards);
                let c = ctx.clone();
                ctx.effect("compaction.session-event-seq", move || {
                    let h = c.on(
                        mizar_session::EVENT_SESSION_EVENT,
                        move |ev: &mizar_session::SessionEventPayload| {
                            if !rt_seq.is_module_enabled() {
                                return;
                            }
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
