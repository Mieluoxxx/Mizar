//! `CompactionRuntime` 上下文压缩服务：
//! 统一编排单阶段检查点摘要、溢出兜底恢复、手动 `/compact`、上下文压力探测与 `write` 写前读守卫。

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::agent_loop::{
    AgentEvent, AgentLoop, AgentPreStepDecision, AgentPreStepInput, AgentRequestErrorDecision,
    AgentRequestErrorInput, AgentRequestErrorKind, StepOutcome,
};
use crate::system_prompt::SystemPrompt;
use mizar_llm::LlmRuntime;
use mizar_session::{Session, SessionStore, SessionTraceEvent};
use mizar_tools::{PostExecuteInput, PreExecuteDecision, ToolExecution, ToolResult, ToolRuntime};
use mizar_types::{BoxFuture, HookGuard, HookOptions, Next, SyncEventBus};
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
    ContextPressureReport, ManualCompactOptions,
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
    llm: Arc<LlmRuntime>,
    tools: Arc<ToolRuntime>,
    system_prompt: Arc<SystemPrompt>,
    config: Arc<Mutex<CompactionConfig>>,
    module_enabled: AtomicBool,
    in_flight: Arc<Mutex<HashSet<String>>>,
    sessions_meta: Arc<Mutex<HashMap<String, SessionCompactionMeta>>>,
    file_guards: Arc<FileGuardTracker>,
    on_start: SyncEventBus<CompactionStartEvent>,
    on_progress: SyncEventBus<CompactionProgressEvent>,
    on_end: Arc<SyncEventBus<CompactionEndEvent>>,
}

impl CompactionRuntime {
    pub fn new(
        llm: Arc<LlmRuntime>,
        tools: Arc<ToolRuntime>,
        system_prompt: Arc<SystemPrompt>,
        config: CompactionConfig,
    ) -> Arc<Self> {
        Arc::new(Self {
            llm,
            tools,
            system_prompt,
            config: Arc::new(Mutex::new(config)),
            module_enabled: AtomicBool::new(true),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            sessions_meta: Arc::new(Mutex::new(HashMap::new())),
            file_guards: Arc::new(FileGuardTracker::new()),
            on_start: SyncEventBus::new(),
            on_progress: SyncEventBus::new(),
            on_end: Arc::new(SyncEventBus::new()),
        })
    }

    pub fn install(
        llm: Arc<LlmRuntime>,
        sessions: &Arc<SessionStore>,
        tools: &Arc<ToolRuntime>,
        system_prompt: Arc<SystemPrompt>,
        agent_loop: &Arc<AgentLoop>,
        config: CompactionConfig,
    ) -> (Arc<Self>, HookGuard) {
        let rt = Self::new(llm, Arc::clone(tools), system_prompt, config);
        let mut guards = Vec::new();

        // 1. 步前自动检查水位并触发单阶段检查点压缩（priority = -20，早于 instructions/skills）
        {
            let weak = Arc::downgrade(&rt);
            guards.push(
                agent_loop.pre_step_hook().register_with(
                    HookOptions {
                        priority: -20,
                        ..Default::default()
                    },
                    move |input: AgentPreStepInput,
                          next: Next<
                        AgentPreStepInput,
                        BoxFuture<'static, AgentPreStepDecision>,
                    >| {
                        let weak = weak.clone();
                        Box::pin(async move {
                            if input.cancel.is_cancelled() {
                                return AgentPreStepDecision::Reject {
                                    reason: "cancelled".to_string(),
                                };
                            }
                            if let Some(rt) = weak.upgrade() {
                                rt.handle_pre_step(&input).await;
                            }
                            next.call(input).await
                        }) as BoxFuture<'static, AgentPreStepDecision>
                    },
                ),
            );
        }

        // 2. 步后记录最近真正成功答复过的 `(provider, model)` 路由
        {
            let weak = Arc::downgrade(&rt);
            let meta = Arc::clone(&rt.sessions_meta);
            let ev_guard = agent_loop.on_event(move |ev| {
                let Some(rt_step) = weak.upgrade() else {
                    return;
                };
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
            });
            guards.push(HookGuard::new(move || drop(ev_guard)));
        }

        // 3. 溢出错误自动压缩并重试恢复
        {
            let weak = Arc::downgrade(&rt);
            guards.push(agent_loop.request_error_hook().register(
                move |input: AgentRequestErrorInput,
                      next: Next<
                    AgentRequestErrorInput,
                    BoxFuture<'static, AgentRequestErrorDecision>,
                >| {
                    let weak = weak.clone();
                    Box::pin(async move {
                        if input.kind == AgentRequestErrorKind::ContextOverflow
                            && let Some(rt) = weak.upgrade()
                            && rt.handle_overflow(&input).await
                        {
                            return AgentRequestErrorDecision::Retry;
                        }
                        next.call(input).await
                    }) as BoxFuture<'static, AgentRequestErrorDecision>
                },
            ));
        }

        // 4. 统一 `write` 写前读守卫（>50 行首读要求 + 压后陈旧整文件覆写拦截）
        {
            let weak = Arc::downgrade(&rt);
            let file_guards = Arc::clone(&rt.file_guards);
            let sessions_pre = Arc::clone(sessions);
            guards.push(tools.pre_execute_hook().register(
                move |exec: ToolExecution,
                      next: Next<ToolExecution, BoxFuture<'static, PreExecuteDecision>>| {
                    let weak = weak.clone();
                    let file_guards = Arc::clone(&file_guards);
                    let sessions_ref = Arc::clone(&sessions_pre);
                    Box::pin(async move {
                        if weak
                            .upgrade()
                            .is_some_and(|rt| rt.is_module_enabled())
                        {
                            let decision = file_guards.check_pre_execute(&exec, &sessions_ref);
                            if !decision.allow {
                                return decision;
                            }
                        }
                        next.call(exec).await
                    }) as BoxFuture<'static, PreExecuteDecision>
                },
            ));
        }

        // 5. 追踪成功 `read`/`write` 后的文件可见状态
        {
            let weak = Arc::downgrade(&rt);
            let file_guards = Arc::clone(&rt.file_guards);
            let sessions_post = Arc::clone(sessions);
            guards.push(tools.post_execute_hook().register(
                move |input: PostExecuteInput,
                      next: Next<PostExecuteInput, BoxFuture<'static, ToolResult>>| {
                    let weak = weak.clone();
                    let file_guards = Arc::clone(&file_guards);
                    let sessions_ref = Arc::clone(&sessions_post);
                    Box::pin(async move {
                        if !weak
                            .upgrade()
                            .is_some_and(|rt| rt.is_module_enabled())
                        {
                            return next.call(input).await;
                        }
                        let exec = input.exec.clone();
                        let res = next.call(input).await;
                        file_guards.handle_post_execute(&exec, res, &sessions_ref)
                    }) as BoxFuture<'static, ToolResult>
                },
            ));
        }

        // 6. 追踪会话真实落盘序号，对齐 FileGuardTracker 的 `last_read_seq`
        {
            let weak = Arc::downgrade(&rt);
            let file_guards = Arc::clone(&rt.file_guards);
            guards.push(sessions.on_event(move |ev| {
                if !weak.upgrade().is_some_and(|rt| rt.is_module_enabled()) {
                    return;
                }
                if let mizar_session::SessionEvent::ToolResult {
                    tool_call_id,
                    is_error: false,
                    ..
                } = &ev.entry.event
                {
                    file_guards.reconcile_tool_result_seq(
                        &ev.session.id,
                        tool_call_id,
                        ev.entry.seq,
                    );
                }
            }));
        }

        (rt, HookGuard::group(guards))
    }

    pub fn on_start(
        &self,
        cb: impl Fn(&CompactionStartEvent) + Send + Sync + 'static,
    ) -> HookGuard {
        self.on_start.on(cb)
    }

    pub fn on_progress(
        &self,
        cb: impl Fn(&CompactionProgressEvent) + Send + Sync + 'static,
    ) -> HookGuard {
        self.on_progress.on(cb)
    }

    pub fn on_end(&self, cb: impl Fn(&CompactionEndEvent) + Send + Sync + 'static) -> HookGuard {
        self.on_end.on(cb)
    }
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
        let providers = self.llm.provider_infos();
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
        let system_prompt = self.system_prompt.build_for_session(Some(session_id));
        let tools = self.tools.list_declared(Some(session_id));
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
            on_end: Arc<SyncEventBus<CompactionEndEvent>>,
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
                    self.on_end.emit(&CompactionEndEvent {
                        session_id: self.session.id.clone(),
                        trigger: self.trigger,
                        outcome: CompactionOutcome::Failed,
                        tokens_before: self.tokens_before,
                        tokens_after: self.tokens_before,
                        first_kept_seq: None,
                        summary_preview: None,
                        skipped_reason: Some("compaction aborted before completion".to_string()),
                    });
                }
            }
        }

        // 发送开始审计与 UI 事件
        session.append_trace(SessionTraceEvent::CompactionStart {
            trigger: trigger.as_str().to_string(),
        });
        self.on_start.emit(&CompactionStartEvent {
            session_id: session.id.clone(),
            trigger,
            tokens_before,
            context_window: budget.context_window,
        });
        let mut trace_guard = CompactionTraceGuard {
            on_end: Arc::clone(&self.on_end),
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

        let current_visible_messages = session.derive_messages();
        let sid_for_progress = session.id.clone();

        // 溢出场景当前前缀已爆窗，禁止复用前缀，直接走纯净转写请求
        let reuse_prefix = !force_overflow;

        let summary_res = generate_checkpoint_summary(
            SummarizeParams {
                llm: &self.llm,
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
                self.on_progress.emit(&CompactionProgressEvent {
                    session_id: sid_for_progress.clone(),
                    stage: stage.to_string(),
                    delta_chars,
                });
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
        self.tools.reset_session_activations(&session.id);

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
        self.on_end.emit(&CompactionEndEvent {
            session_id: session_id.to_string(),
            trigger: report.trigger,
            outcome: report.outcome,
            tokens_before: report.tokens_before,
            tokens_after: report.tokens_after,
            first_kept_seq: report.first_kept_seq,
            summary_preview: report.summary_preview.clone(),
            skipped_reason: report.skipped_reason.clone(),
        });
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
