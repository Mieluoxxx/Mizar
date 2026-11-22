//! `CacheWarmingRuntime` Cordis 服务（`"cacheWarming"`）：
//! 在后台独立任务中维护单会话 Prompt Cache 续命定时器与状态投影，对主循环零侵入。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Service};
use futures::StreamExt;
use mizar_agent_loop::{
    AgentTurnEndEvent, AgentTurnStartEvent, EVENT_AGENT_TURN_END, EVENT_AGENT_TURN_START,
};
use mizar_client_core::CacheWarmingSessionStatus;
#[cfg(feature = "compaction")]
use mizar_compaction::CompactionRuntime;
use mizar_llm::{
    CACHE_WARMING_MAX_AGE_MS, CACHE_WARMING_MAX_IDLE_AGE_MS, CacheWarmingDecision,
    CacheWarmingMode, CacheWarmingPhase, EVENT_LLM_AFTER_RESPONSE, FinishReason, GenerateOptions,
    LlmAfterResponseEvent, LlmContextExt, ModelInfo, RequestOrigin, TokenUsage, calculate_cost,
    evaluate_cache_warming, get_cache_warming_delay_ms,
    get_cache_warming_refresh_deadline_offset_ms, get_prompt_cache_ttl_ms,
    is_cache_warming_replayable,
};
use mizar_session::{
    EVENT_SESSION_EVENT, Session, SessionContextExt, SessionEvent, SessionEventPayload,
    SessionStore,
};
use thiserror::Error;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Prompt Cache 保活注册到 Cordis 的服务名。
pub const SERVICE_CACHE_WARMING: &str = "cacheWarming";
/// 启用 `show_cache_miss_notices` 时向前端投递的轻量化保活提示事件。
pub const EVENT_CACHE_WARMING_NOTICE: &str = "cache-warming/notice";

/// `CacheWarmingRuntime` 插件配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheWarmingConfig {
    pub mode: CacheWarmingMode,
    pub show_cache_miss_notices: bool,
}

impl Default for CacheWarmingConfig {
    fn default() -> Self {
        Self {
            mode: CacheWarmingMode::Idle,
            show_cache_miss_notices: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum CacheWarmingError {
    #[error("核心模型服务不可用: {0}")]
    LlmUnavailable(String),
    #[error("核心会话服务不可用: {0}")]
    SessionsUnavailable(String),
}

/// 后台缓存续命完成后发往客户端的只读提示负载。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheWarmingNoticeEvent {
    pub session_id: String,
    pub text: String,
}

#[derive(Clone)]
struct LastRealRequest {
    request: Arc<GenerateOptions>,
    provider: String,
    model: String,
    model_info: ModelInfo,
    prompt_tokens: u64,
    ttl_ms: u64,
    delay_ms: u64,
    refresh_deadline_offset_ms: u64,
    started_at: Instant,
}

struct ActiveTimer {
    generation: u64,
    cancel: CancellationToken,
    next_warm_at: Instant,
    refresh_deadline: Instant,
}

struct SessionWarmingState {
    phase: CacheWarmingPhase,
    in_agent_turn: bool,
    generation: u64,
    active_timer: Option<ActiveTimer>,
    last_real_request: Option<LastRealRequest>,
    baseline_visible_len: usize,
    baseline_compacted_up_to_seq: Option<u64>,
    warm_count: usize,
    total_warm_cost_usd: f64,
    last_stop_reason: Option<String>,
    last_warm_cost_usd: Option<f64>,
    last_miss_cost_usd: Option<f64>,
}

impl Default for SessionWarmingState {
    fn default() -> Self {
        Self {
            phase: CacheWarmingPhase::Idle,
            in_agent_turn: false,
            generation: 0,
            active_timer: None,
            last_real_request: None,
            baseline_visible_len: 0,
            baseline_compacted_up_to_seq: None,
            warm_count: 0,
            total_warm_cost_usd: 0.0,
            last_stop_reason: None,
            last_warm_cost_usd: None,
            last_miss_cost_usd: None,
        }
    }
}

impl SessionWarmingState {
    fn cancel_active_timer(&mut self, reason: Option<&str>) {
        self.generation = self.generation.wrapping_add(1);
        if let Some(timer) = self.active_timer.take() {
            timer.cancel.cancel();
        }
        if let Some(r) = reason {
            self.last_stop_reason = Some(r.to_string());
        }
    }
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn max_age_for_phase(phase: CacheWarmingPhase) -> u64 {
    match phase {
        CacheWarmingPhase::Streaming => CACHE_WARMING_MAX_AGE_MS,
        CacheWarmingPhase::Idle => CACHE_WARMING_MAX_IDLE_AGE_MS,
    }
}

fn selection_diverges(session: &Session, provider: &str, model: &str) -> bool {
    let Some(sel) = session.latest_selection() else {
        return false;
    };
    sel.provider.as_deref().is_some_and(|p| p != provider)
        || sel.model.as_deref().is_some_and(|m| m != model)
}

fn lookup_session_in_memory(sessions: &SessionStore, session_id: &str) -> Option<Arc<Session>> {
    if !sessions.contains_in_memory(session_id) {
        return None;
    }
    sessions.get(session_id)
}

fn touch_compaction_activity(ctx: &Context, session_id: &str) {
    #[cfg(feature = "compaction")]
    if let Some(comp) = ctx.reflect().get::<CompactionRuntime>("compaction") {
        comp.touch_session_activity(session_id);
    }
    #[cfg(not(feature = "compaction"))]
    let _ = (ctx, session_id);
}

/// Prompt Cache 续命后台运行时服务（`ctx.cache_warming()`）。
pub struct CacheWarmingRuntime {
    ctx: Context,
    config: Arc<Mutex<CacheWarmingConfig>>,
    sessions_state: Arc<Mutex<HashMap<String, SessionWarmingState>>>,
    dispose_token: CancellationToken,
}

impl CacheWarmingRuntime {
    /// 读取当前缓存保活配置快照。
    pub fn config(&self) -> CacheWarmingConfig {
        self.config.lock().unwrap().clone()
    }

    /// 热更新缓存保活配置。
    pub fn update_config(&self, f: impl FnOnce(&mut CacheWarmingConfig)) {
        let next_mode = {
            let mut guard = self.config.lock().unwrap();
            f(&mut guard);
            guard.mode
        };
        self.reconcile_mode_change(next_mode);
    }

    /// 设置保活模式（`off` / `streaming` / `idle`），并立即收敛受影响的活跃定时器。
    pub fn set_mode(&self, mode: CacheWarmingMode) {
        {
            let mut guard = self.config.lock().unwrap();
            guard.mode = mode;
        }
        self.reconcile_mode_change(mode);
    }

    /// 开关保活/失效提示通知。
    pub fn set_show_notices(&self, show: bool) {
        self.config.lock().unwrap().show_cache_miss_notices = show;
    }

    /// 查询指定会话当前的缓存保活状态投影。
    pub fn status(&self, session_id: &str) -> Option<CacheWarmingSessionStatus> {
        let mode = self.config.lock().unwrap().mode;
        let guard = self.sessions_state.lock().unwrap();
        let st = guard.get(session_id)?;
        let now_instant = Instant::now();
        let now_ms = now_epoch_ms();
        let (active, next_warm_in_secs, next_warm_at_ms) = match &st.active_timer {
            Some(timer) if !timer.cancel.is_cancelled() => {
                let rem = timer.next_warm_at.saturating_duration_since(now_instant);
                let secs = rem.as_secs();
                let at_ms = now_ms.saturating_add(rem.as_millis());
                (true, Some(secs), Some(at_ms))
            }
            _ => (false, None, None),
        };
        Some(CacheWarmingSessionStatus {
            mode,
            active,
            next_warm_in_secs,
            next_warm_at_ms,
            warm_count: st.warm_count,
            total_warm_cost_usd: st.total_warm_cost_usd,
            last_stop_reason: st.last_stop_reason.clone(),
            last_warm_cost_usd: st.last_warm_cost_usd,
            last_miss_cost_usd: st.last_miss_cost_usd,
        })
    }

    fn reconcile_mode_change(&self, mode: CacheWarmingMode) {
        let mut guard = self.sessions_state.lock().unwrap();
        for st in guard.values_mut() {
            match mode {
                CacheWarmingMode::Off => {
                    if st.active_timer.is_some() {
                        st.cancel_active_timer(Some("cache warming disabled"));
                    }
                }
                CacheWarmingMode::Streaming => {
                    if st.active_timer.is_some() && st.phase == CacheWarmingPhase::Idle {
                        st.cancel_active_timer(Some("streaming mode stopped while idle"));
                    }
                }
                CacheWarmingMode::Idle => {}
            }
        }
    }

    fn handle_after_response(self: &Arc<Self>, ev: &LlmAfterResponseEvent) {
        if self.dispose_token.is_cancelled() {
            return;
        }
        let RequestOrigin::SessionTurn { session_id, .. } = &ev.request.origin else {
            return;
        };
        if session_id.is_empty() {
            return;
        }
        let session_id = session_id.clone();

        let mode = self.config.lock().unwrap().mode;

        // 无论是否放行，同一会话先取消此前挂起的保活定时器
        let phase = {
            let mut guard = self.sessions_state.lock().unwrap();
            let st = guard.entry(session_id.clone()).or_default();
            st.cancel_active_timer(None);
            st.phase
        };

        if mode == CacheWarmingMode::Off {
            self.record_stop(&session_id, "cache warming disabled", None, None);
            return;
        }
        if mode == CacheWarmingMode::Streaming && phase == CacheWarmingPhase::Idle {
            self.record_stop(
                &session_id,
                "streaming mode requires active turn",
                None,
                None,
            );
            return;
        }

        let Ok(llm) = self.ctx.llm() else {
            self.record_stop(&session_id, "llm runtime unavailable", None, None);
            return;
        };
        let Ok(sessions) = self.ctx.sessions() else {
            self.record_stop(&session_id, "session store unavailable", None, None);
            return;
        };

        // 1. 查找匹配协议与目录元数据，未声明 prompt_cache 或 cost 时跳过
        let Some(provider_info) = llm
            .provider_infos()
            .into_iter()
            .find(|p| p.id == ev.provider)
        else {
            self.record_stop(&session_id, "provider metadata not found", None, None);
            return;
        };
        let Some(model_info) = provider_info.models.into_iter().find(|m| m.id == ev.model) else {
            self.record_stop(&session_id, "model metadata not found", None, None);
            return;
        };

        if model_info.prompt_cache.is_none() || model_info.cost.is_none() {
            self.record_stop(
                &session_id,
                "model lacks promptCache or cost metadata",
                None,
                None,
            );
            return;
        }

        let Some(ttl_ms) = get_prompt_cache_ttl_ms(&model_info, ev.request.cache_retention) else {
            self.record_stop(
                &session_id,
                "prompt cache TTL unavailable for retention",
                None,
                None,
            );
            return;
        };
        let Some(delay_ms) = get_cache_warming_delay_ms(ttl_ms) else {
            self.record_stop(&session_id, "prompt cache TTL too short", None, None);
            return;
        };

        // 2. 重放安全门禁：拒绝旧版 Anthropic 非 adaptive reasoning 请求
        if !is_cache_warming_replayable(&provider_info.protocol, &model_info, &ev.request) {
            self.record_stop(
                &session_id,
                "non-adaptive Anthropic reasoning cannot safely replay",
                None,
                None,
            );
            return;
        }

        // 3. $0.05 期望净收益门禁
        let prompt_tokens = ev.usage.as_ref().map_or(0, TokenUsage::prompt_tokens);
        let eval = evaluate_cache_warming(&model_info, prompt_tokens, phase);
        if eval.decision == CacheWarmingDecision::Stop {
            self.record_stop(
                &session_id,
                "expected savings below $0.05",
                Some(eval.warm_cost),
                Some(eval.miss_cost),
            );
            return;
        }

        // 4. 会话一致性快照与最大保活窗口门禁
        let Some(session) = lookup_session_in_memory(&sessions, &session_id) else {
            self.record_stop(&session_id, "session not active in memory", None, None);
            return;
        };
        if selection_diverges(&session, &ev.provider, &ev.model) {
            self.record_stop(&session_id, "session model selection diverged", None, None);
            return;
        }
        let max_age_ms = max_age_for_phase(phase);
        if delay_ms > max_age_ms {
            self.record_stop(
                &session_id,
                "warming delay exceeds max warming age",
                Some(eval.warm_cost),
                Some(eval.miss_cost),
            );
            return;
        }

        let visible_len = session.with_visible_entries(|e| e.len());
        let compacted_up_to_seq = session.header().compacted_up_to_seq;
        let refresh_offset_ms = get_cache_warming_refresh_deadline_offset_ms(ttl_ms, delay_ms);
        let started_at = Instant::now();

        let last_req = LastRealRequest {
            request: Arc::clone(&ev.request),
            provider: ev.provider.clone(),
            model: ev.model.clone(),
            model_info,
            prompt_tokens,
            ttl_ms,
            delay_ms,
            refresh_deadline_offset_ms: refresh_offset_ms,
            started_at,
        };

        let spawn_spec = {
            let mut guard = self.sessions_state.lock().unwrap();
            let st = guard.entry(session_id.clone()).or_default();
            st.last_warm_cost_usd = Some(eval.warm_cost);
            st.last_miss_cost_usd = Some(eval.miss_cost);
            st.baseline_visible_len = visible_len;
            st.baseline_compacted_up_to_seq = compacted_up_to_seq;
            st.last_real_request = Some(last_req.clone());
            st.last_stop_reason = None;
            Self::arm_next_timer_locked(st, &self.dispose_token, started_at)
        };

        if let Some((gen_id, timer_cancel, next_warm_at)) = spawn_spec {
            self.spawn_timer_task(session_id, gen_id, timer_cancel, next_warm_at);
        }
    }

    fn arm_next_timer_locked(
        st: &mut SessionWarmingState,
        dispose_token: &CancellationToken,
        cycle_start: Instant,
    ) -> Option<(u64, CancellationToken, Instant)> {
        let req = st.last_real_request.as_ref()?;
        let delay = Duration::from_millis(req.delay_ms);
        let next_warm_at = cycle_start + delay;
        let max_age_ms = max_age_for_phase(st.phase);
        let projected_age_ms = next_warm_at
            .saturating_duration_since(req.started_at)
            .as_millis() as u64;
        if projected_age_ms > max_age_ms {
            st.cancel_active_timer(Some("max warming age reached"));
            return None;
        }

        st.generation = st.generation.wrapping_add(1);
        if let Some(prev) = st.active_timer.take() {
            prev.cancel.cancel();
        }
        let gen_id = st.generation;
        let timer_cancel = dispose_token.child_token();
        let refresh_deadline = cycle_start + Duration::from_millis(req.refresh_deadline_offset_ms);
        st.active_timer = Some(ActiveTimer {
            generation: gen_id,
            cancel: timer_cancel.clone(),
            next_warm_at,
            refresh_deadline,
        });
        Some((gen_id, timer_cancel, next_warm_at))
    }

    fn record_stop(
        &self,
        session_id: &str,
        reason: &str,
        warm_cost: Option<f64>,
        miss_cost: Option<f64>,
    ) {
        let mut guard = self.sessions_state.lock().unwrap();
        let st = guard.entry(session_id.to_string()).or_default();
        if warm_cost.is_some() {
            st.last_warm_cost_usd = warm_cost;
        }
        if miss_cost.is_some() {
            st.last_miss_cost_usd = miss_cost;
        }
        st.cancel_active_timer(Some(reason));
    }

    fn handle_turn_start(&self, ev: &AgentTurnStartEvent) {
        let mut guard = self.sessions_state.lock().unwrap();
        let st = guard.entry(ev.session.id.clone()).or_default();
        st.phase = CacheWarmingPhase::Streaming;
        st.in_agent_turn = true;
        // 新对话回合开始时取消旧空闲定时器，避免与新一轮真实请求并发撞车
        if st.active_timer.is_some() {
            st.cancel_active_timer(Some("new turn started"));
        }
    }

    fn handle_turn_end(&self, ev: &AgentTurnEndEvent) {
        let mode = self.config.lock().unwrap().mode;
        let visible_len = ev.session.with_visible_entries(|e| e.len());
        let compacted_up_to_seq = ev.session.header().compacted_up_to_seq;

        let mut guard = self.sessions_state.lock().unwrap();
        let st = guard.entry(ev.session.id.clone()).or_default();
        st.phase = CacheWarmingPhase::Idle;
        st.in_agent_turn = false;
        st.baseline_visible_len = visible_len;
        st.baseline_compacted_up_to_seq = compacted_up_to_seq;

        if st.active_timer.is_none() {
            return;
        }

        match mode {
            CacheWarmingMode::Off => {
                st.cancel_active_timer(Some("cache warming disabled"));
            }
            CacheWarmingMode::Streaming => {
                st.cancel_active_timer(Some("turn ended in streaming mode"));
            }
            CacheWarmingMode::Idle => {
                let Some(timer) = st.active_timer.as_ref() else {
                    return;
                };
                let Some(req) = st.last_real_request.as_ref() else {
                    st.cancel_active_timer(Some("missing request snapshot"));
                    return;
                };
                let projected_idle_age_ms = timer
                    .next_warm_at
                    .saturating_duration_since(req.started_at)
                    .as_millis() as u64;
                if projected_idle_age_ms > CACHE_WARMING_MAX_IDLE_AGE_MS {
                    st.cancel_active_timer(Some("idle max warming age exceeded"));
                    return;
                }
                let eval = evaluate_cache_warming(
                    &req.model_info,
                    req.prompt_tokens,
                    CacheWarmingPhase::Idle,
                );
                st.last_warm_cost_usd = Some(eval.warm_cost);
                st.last_miss_cost_usd = Some(eval.miss_cost);
                if eval.decision == CacheWarmingDecision::Stop {
                    st.cancel_active_timer(Some("expected idle savings below $0.05"));
                }
            }
        }
    }

    fn handle_session_event(&self, ev: &SessionEventPayload) {
        let mut guard = self.sessions_state.lock().unwrap();
        let Some(st) = guard.get_mut(&ev.session.id) else {
            return;
        };
        if st.active_timer.is_none() {
            return;
        }

        match &ev.entry.event {
            SessionEvent::Usage { kind, .. } if kind == "cache_warm" => {}
            SessionEvent::Usage { .. } | SessionEvent::Trace { .. } => {}
            SessionEvent::Compaction { .. } => {
                st.cancel_active_timer(Some("session compacted"));
            }
            SessionEvent::ToolResultPrune { .. } => {
                st.cancel_active_timer(Some("session context pruned"));
            }
            SessionEvent::Selection {
                provider, model, ..
            } => {
                if let Some(req) = st.last_real_request.as_ref() {
                    let diverges = provider.as_deref().is_some_and(|p| p != req.provider)
                        || model.as_deref().is_some_and(|m| m != req.model);
                    if diverges {
                        st.cancel_active_timer(Some("model selection changed"));
                    }
                }
            }
            SessionEvent::Assistant { .. } | SessionEvent::ToolResult { .. } => {
                if st.in_agent_turn {
                    // 同一活跃回合内主循环流尾刚落盘本步 Assistant / ToolResult，更新基线而不误取消
                    st.baseline_visible_len = ev.session.with_visible_entries(|e| e.len());
                } else {
                    st.cancel_active_timer(Some("session context updated"));
                }
            }
            SessionEvent::User { .. } | SessionEvent::ContextNotice { .. } => {
                st.cancel_active_timer(Some("session context updated"));
            }
            SessionEvent::BashExecution {
                exclude_from_context,
                ..
            } => {
                if !*exclude_from_context {
                    st.cancel_active_timer(Some("session context updated"));
                }
            }
        }
    }

    fn spawn_timer_task(
        self: &Arc<Self>,
        session_id: String,
        gen_id: u64,
        timer_cancel: CancellationToken,
        next_warm_at: Instant,
    ) {
        let rt = Arc::clone(self);
        tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = timer_cancel.cancelled() => return,
                _ = tokio::time::sleep_until(next_warm_at) => {}
            }
            rt.fire_warming_cycle(session_id, gen_id, timer_cancel)
                .await;
        });
    }

    async fn fire_warming_cycle(
        self: &Arc<Self>,
        session_id: String,
        gen_id: u64,
        timer_cancel: CancellationToken,
    ) {
        if self.dispose_token.is_cancelled() || timer_cancel.is_cancelled() {
            return;
        }

        let Ok(llm) = self.ctx.llm() else {
            return;
        };
        let Ok(sessions) = self.ctx.sessions() else {
            return;
        };

        let mode = self.config.lock().unwrap().mode;

        // 校验全部前置门禁并构建不可变重放请求，不在持锁期间跨 `.await`
        let prepared = {
            let mut guard = self.sessions_state.lock().unwrap();
            let Some(st) = guard.get_mut(&session_id) else {
                return;
            };
            if st.generation != gen_id {
                return;
            }
            let Some(timer) = st.active_timer.as_ref() else {
                return;
            };
            if timer.generation != gen_id {
                return;
            }

            let now = Instant::now();
            // 1. 休眠/调度漂移硬截止线保护
            if now > timer.refresh_deadline {
                st.cancel_active_timer(Some("cache refresh deadline missed"));
                return;
            }

            if mode == CacheWarmingMode::Off {
                st.cancel_active_timer(Some("cache warming disabled"));
                return;
            }
            if mode == CacheWarmingMode::Streaming && st.phase == CacheWarmingPhase::Idle {
                st.cancel_active_timer(Some("streaming mode stopped while idle"));
                return;
            }

            let Some(req) = st.last_real_request.clone() else {
                st.cancel_active_timer(Some("missing request snapshot"));
                return;
            };

            // 2. 最大持续保活时间上限保护
            let age_ms = now.saturating_duration_since(req.started_at).as_millis() as u64;
            if age_ms > max_age_for_phase(st.phase) {
                st.cancel_active_timer(Some("max warming age reached"));
                return;
            }

            // 3. 按当前 phase 二次复核 $0.05 期望净收益
            let eval = evaluate_cache_warming(&req.model_info, req.prompt_tokens, st.phase);
            st.last_warm_cost_usd = Some(eval.warm_cost);
            st.last_miss_cost_usd = Some(eval.miss_cost);
            if eval.decision == CacheWarmingDecision::Stop {
                st.cancel_active_timer(Some("expected savings below $0.05"));
                return;
            }

            // 4. 会话上下文一致性校验
            let Some(session) = lookup_session_in_memory(&sessions, &session_id) else {
                st.cancel_active_timer(Some("session not active in memory"));
                return;
            };
            if session.header().compacted_up_to_seq != st.baseline_compacted_up_to_seq {
                st.cancel_active_timer(Some("session compacted"));
                return;
            }
            if !st.in_agent_turn
                && session.with_visible_entries(|e| e.len()) != st.baseline_visible_len
            {
                st.cancel_active_timer(Some("session context updated"));
                return;
            }
            if selection_diverges(&session, &req.provider, &req.model) {
                st.cancel_active_timer(Some("model selection changed"));
                return;
            }

            let request_cancel = timer_cancel.child_token();
            let mut replay_opts = (*req.request).clone();
            if replay_opts.provider.is_none() {
                replay_opts.provider = Some(req.provider.clone());
            }
            if replay_opts.model.is_none() {
                replay_opts.model = Some(req.model.clone());
            }
            replay_opts.max_tokens = Some(1);
            replay_opts.disable_continuation = true;
            replay_opts.origin = RequestOrigin::CacheWarm;
            replay_opts.cancel = request_cancel.clone();

            (req, session, replay_opts, request_cancel, eval.miss_cost)
        };

        let (req, session, replay_opts, request_cancel, expected_miss_cost) = prepared;

        let mut stream = match llm.stream_direct(replay_opts) {
            Ok(s) => s,
            Err(err) => {
                self.finish_failed_cycle(
                    &session_id,
                    gen_id,
                    &format!("warm request failed: {err}"),
                );
                return;
            }
        };

        let mut latest_usage: Option<TokenUsage> = None;
        let mut finish_reason: Option<FinishReason> = None;
        let mut stream_err: Option<String> = None;

        while let Some(item) = stream.next().await {
            if request_cancel.is_cancelled() || self.dispose_token.is_cancelled() {
                return;
            }
            match item {
                Ok(mizar_llm::StreamChunk::Usage { usage }) => {
                    latest_usage = Some(usage);
                }
                Ok(mizar_llm::StreamChunk::Finish { reason, .. }) => {
                    finish_reason = Some(reason);
                }
                Err(err) => {
                    stream_err = Some(err.to_string());
                    break;
                }
                Ok(_) => {}
            }
        }

        if let Some(err) = stream_err {
            self.finish_failed_cycle(&session_id, gen_id, &format!("warm stream error: {err}"));
            return;
        }
        let clean_finish = finish_reason
            .as_ref()
            .is_some_and(|r| !matches!(r, FinishReason::Aborted | FinishReason::Error { .. }));
        if !clean_finish || request_cancel.is_cancelled() || self.dispose_token.is_cancelled() {
            self.finish_failed_cycle(&session_id, gen_id, "warm response aborted or errored");
            return;
        }

        let effective_usage = latest_usage.unwrap_or(TokenUsage {
            output_tokens: 1,
            cache_read_tokens: Some(req.prompt_tokens),
            ..Default::default()
        });
        let warm_cost = req
            .model_info
            .cost
            .as_ref()
            .map_or(0.0, |rates| calculate_cost(rates, &effective_usage));

        let next_arm = {
            let mut guard = self.sessions_state.lock().unwrap();
            let Some(st) = guard.get_mut(&session_id) else {
                return;
            };
            if st.generation != gen_id {
                // 飞行期间状态已被新请求或取消变更，丢弃过时完成结果
                return;
            }
            st.active_timer = None;
            st.warm_count += 1;
            st.total_warm_cost_usd += warm_cost;
            st.last_warm_cost_usd = Some(warm_cost);

            let warm_count = st.warm_count;
            let next_spec = Self::arm_next_timer_locked(st, &self.dispose_token, Instant::now());
            (warm_count, next_spec)
        };

        let (warm_count, next_spec) = next_arm;

        // 仅在保活成功收尾后落盘 Usage("cache_warm") 并同步刷新 compaction 活跃时间
        let note = format!(
            "Prompt Cache 续命第 {warm_count} 次（TTL {}s，本次费用 ${warm_cost:.4}）",
            req.ttl_ms / 1000
        );
        session.append(SessionEvent::usage(
            "cache_warm",
            &req.provider,
            &req.model,
            effective_usage,
            Some(note),
        ));
        touch_compaction_activity(&self.ctx, &session_id);

        if self.config.lock().unwrap().show_cache_miss_notices {
            let text = format!(
                "已保活 Prompt Cache（本次费用 ${warm_cost:.4}，避免重计费约 ${expected_miss_cost:.2}）"
            );
            self.ctx.emit(
                EVENT_CACHE_WARMING_NOTICE,
                CacheWarmingNoticeEvent {
                    session_id: session_id.clone(),
                    text,
                },
            );
        }

        if let Some((next_gen, next_cancel, next_warm_at)) = next_spec {
            self.spawn_timer_task(session_id, next_gen, next_cancel, next_warm_at);
        }
    }

    fn finish_failed_cycle(&self, session_id: &str, gen_id: u64, reason: &str) {
        let mut guard = self.sessions_state.lock().unwrap();
        let Some(st) = guard.get_mut(session_id) else {
            return;
        };
        if st.generation == gen_id {
            st.cancel_active_timer(Some(reason));
        }
    }
}

impl Service for CacheWarmingRuntime {
    type Config = CacheWarmingConfig;

    fn service_name() -> &'static str {
        SERVICE_CACHE_WARMING
    }

    fn plugin_name() -> &'static str {
        "cache-warming"
    }

    fn inject() -> Inject {
        Inject::from(["llm", "sessions"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let _ = ctx.llm()?;
        let _ = ctx.sessions()?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: Arc::new(Mutex::new(config)),
            sessions_state: Arc::new(Mutex::new(HashMap::new())),
            dispose_token: CancellationToken::new(),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let rt = Arc::clone(&self);
            let c = ctx.clone();
            let dispose = self.dispose_token.clone();
            let states_for_cleanup = Arc::clone(&self.sessions_state);
            ctx.effect("cache-warming.listeners", move || {
                let rt_after = Arc::clone(&rt);
                let h_after = c.on(
                    EVENT_LLM_AFTER_RESPONSE,
                    move |ev: &LlmAfterResponseEvent| {
                        rt_after.handle_after_response(ev);
                    },
                )?;

                let rt_start = Arc::clone(&rt);
                let h_start = c.on(EVENT_AGENT_TURN_START, move |ev: &AgentTurnStartEvent| {
                    rt_start.handle_turn_start(ev);
                })?;

                let rt_end = Arc::clone(&rt);
                let h_end = c.on(EVENT_AGENT_TURN_END, move |ev: &AgentTurnEndEvent| {
                    rt_end.handle_turn_end(ev);
                })?;

                let rt_sess = Arc::clone(&rt);
                let h_sess = c.on(EVENT_SESSION_EVENT, move |ev: &SessionEventPayload| {
                    rt_sess.handle_session_event(ev);
                })?;

                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    dispose.cancel();
                    {
                        let mut guard = states_for_cleanup.lock().unwrap();
                        for st in guard.values_mut() {
                            st.cancel_active_timer(Some("plugin disposed"));
                        }
                    }
                    h_after.dispose().await;
                    h_start.dispose().await;
                    h_end.dispose().await;
                    h_sess.dispose().await;
                })))
            })?;
            Ok(())
        })
    }
}

/// 为 `cordis::Context` 补充 `ctx.cache_warming()` 扩展方法。
pub trait CacheWarmingContextExt {
    fn cache_warming(&self) -> Result<Arc<CacheWarmingRuntime>, CordisError>;
}

impl CacheWarmingContextExt for Context {
    fn cache_warming(&self) -> Result<Arc<CacheWarmingRuntime>, CordisError> {
        self.service::<CacheWarmingRuntime>(SERVICE_CACHE_WARMING)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cordis::PluginDef;
    use futures::channel::mpsc;
    use mizar_agent_loop::TurnEndReason;
    use mizar_llm::{
        BoxChunkStream, LlmAdapter, LlmError, ModelCostRates, PromptCacheLifetime, ProviderInfo,
        ReasoningEffort, StreamChunk,
    };
    use mizar_session::CompactionDigest;

    use super::*;

    #[derive(Debug, Clone)]
    struct RecordedRequest {
        origin: RequestOrigin,
        max_tokens: Option<u32>,
        disable_continuation: bool,
    }

    struct WarmingTestAdapter {
        protocol: String,
        model_info: ModelInfo,
        calls: Arc<AtomicUsize>,
        recorded_origins: Arc<Mutex<Vec<RecordedRequest>>>,
        prompt_tokens_on_turn: u64,
        fail_cache_warm: bool,
    }

    impl WarmingTestAdapter {
        fn new(protocol: &str, model_info: ModelInfo, prompt_tokens_on_turn: u64) -> Self {
            Self {
                protocol: protocol.to_string(),
                model_info,
                calls: Arc::new(AtomicUsize::new(0)),
                recorded_origins: Arc::new(Mutex::new(Vec::new())),
                prompt_tokens_on_turn,
                fail_cache_warm: false,
            }
        }
    }

    impl LlmAdapter for WarmingTestAdapter {
        fn info(&self) -> Option<ProviderInfo> {
            Some(ProviderInfo {
                id: "test-prov".into(),
                name: "Test Provider".into(),
                protocol: self.protocol.clone(),
                models: vec![self.model_info.clone()],
            })
        }

        fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.recorded_origins.lock().unwrap().push(RecordedRequest {
                origin: options.origin.clone(),
                max_tokens: options.max_tokens,
                disable_continuation: options.disable_continuation,
            });
            let is_warm = options.origin == RequestOrigin::CacheWarm;
            let fail_warm = self.fail_cache_warm;
            let prompt = self.prompt_tokens_on_turn;

            let (tx, rx) = mpsc::unbounded();
            tokio::spawn(async move {
                if is_warm && fail_warm {
                    let _ = tx.unbounded_send(Err(LlmError::transport("boom")));
                    return;
                }
                if is_warm {
                    let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                        usage: TokenUsage {
                            input_tokens: 0,
                            cache_read_tokens: Some(prompt),
                            output_tokens: 1,
                            ..Default::default()
                        },
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::MaxTokens,
                        replay: None,
                    }));
                } else {
                    let _ = tx.unbounded_send(Ok(StreamChunk::TextDelta {
                        index: 0,
                        text: "ok".into(),
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Usage {
                        usage: TokenUsage {
                            input_tokens: 500,
                            cache_write_tokens: Some(prompt.saturating_sub(500)),
                            output_tokens: 20,
                            ..Default::default()
                        },
                    }));
                    let _ = tx.unbounded_send(Ok(StreamChunk::Finish {
                        reason: FinishReason::Stop,
                        replay: None,
                    }));
                }
            });
            Box::pin(rx)
        }
    }

    fn rich_anthropic_model(short_secs: u64, adaptive: bool) -> ModelInfo {
        ModelInfo {
            id: "claude-sonnet".into(),
            name: "Claude Sonnet".into(),
            cost: Some(ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
                tiers: Vec::new(),
            }),
            prompt_cache: Some(PromptCacheLifetime {
                short: Some(short_secs),
                long: Some(3600),
            }),
            force_adaptive_thinking: adaptive,
            ..Default::default()
        }
    }

    async fn settle() {
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }

    async fn setup_test_ctx(
        adapter: Arc<WarmingTestAdapter>,
        cfg: CacheWarmingConfig,
    ) -> (Context, Arc<mizar_llm::LlmRuntime>, Arc<SessionStore>) {
        let ctx = Context::new();
        ctx.plugin(mizar_llm::LlmRuntime::plugin(), ())
            .await
            .unwrap();
        ctx.plugin(SessionStore::plugin(), ()).await.unwrap();
        ctx.plugin(
            PluginDef::sync("test-adapter", move |c, ()| {
                let llm = c.llm()?;
                let ad = Arc::clone(&adapter);
                c.effect("test-adapter.reg", move || {
                    Ok::<_, LlmError>(Some(llm.register_adapter("test-prov", ad)?))
                })?;
                Ok::<_, CordisError>(())
            })
            .inject(["llm"]),
            (),
        )
        .await
        .unwrap();
        ctx.plugin(CacheWarmingRuntime::plugin(), cfg)
            .await
            .unwrap();

        let llm = ctx.llm().unwrap();
        let sessions = ctx.sessions().unwrap();
        (ctx, llm, sessions)
    }

    #[tokio::test(start_paused = true)]
    async fn warms_idle_session_with_positive_savings_writes_usage_and_emits_notice() {
        // 150k prompt -> miss_cost = 150k*(3.75-0.3)/1M = $0.5175 -> 0.15*0.5175 - warm_cost > $0.05
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        ));
        let (ctx, llm, sessions) = setup_test_ctx(
            Arc::clone(&adapter),
            CacheWarmingConfig {
                mode: CacheWarmingMode::Idle,
                show_cache_miss_notices: true,
            },
        )
        .await;

        let notices = Arc::new(Mutex::new(Vec::<CacheWarmingNoticeEvent>::new()));
        let notices_sub = Arc::clone(&notices);
        let _h = ctx
            .on(
                EVENT_CACHE_WARMING_NOTICE,
                move |ev: &CacheWarmingNoticeEvent| {
                    notices_sub.lock().unwrap().push(ev.clone());
                },
            )
            .unwrap();

        let session = sessions.create(None);
        session.append(SessionEvent::user("hello"));

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}

        let cw = ctx.cache_warming().unwrap();
        let st = cw.status(&session.id).unwrap();
        assert!(st.active);
        assert_eq!(st.warm_count, 0);

        // 推进 270 秒（300s * 0.9）触发首次保活
        tokio::time::advance(Duration::from_secs(271)).await;
        settle().await;

        let origins = adapter.recorded_origins.lock().unwrap().clone();
        assert_eq!(
            origins.len(),
            2,
            "expected initial + 1 warm call: {origins:?}"
        );
        assert_eq!(origins[1].origin, RequestOrigin::CacheWarm);
        assert_eq!(origins[1].max_tokens, Some(1));
        assert!(
            origins[1].disable_continuation,
            "disable_continuation must be true"
        );

        // 会话应多出 Usage("cache_warm") 且不新增 Assistant 消息
        let entries = session.all_entries();
        assert!(
            entries.iter().any(|e| matches!(
                &e.event,
                SessionEvent::Usage { kind, provider, model, .. }
                if kind == "cache_warm" && provider == "test-prov" && model == "claude-sonnet"
            )),
            "expected Usage(cache_warm) entry, got: {entries:?}"
        );
        assert!(
            !entries
                .iter()
                .any(|e| matches!(e.event, SessionEvent::Assistant { .. })),
            "cache_warm must not produce Assistant entry"
        );
        assert_eq!(notices.lock().unwrap().len(), 1);
        let after = cw.status(&session.id).unwrap();
        assert_eq!(after.warm_count, 1);
        assert!(after.total_warm_cost_usd > 0.0);

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn ignores_non_session_turn_origins_and_blocks_non_adaptive_anthropic_reasoning() {
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, false),
            200_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        let cw = ctx.cache_warming().unwrap();
        let session = sessions.create(None);

        // 1. Subagent / Compaction / SessionTitle / CacheWarm 必须完全忽略
        for origin in [
            RequestOrigin::Subagent,
            RequestOrigin::Compaction,
            RequestOrigin::SessionTitle,
            RequestOrigin::CacheWarm,
        ] {
            let mut stream = llm
                .stream(GenerateOptions {
                    provider: Some("test-prov".into()),
                    model: Some("claude-sonnet".into()),
                    session_id: Some(session.id.clone()),
                    origin,
                    ..Default::default()
                })
                .unwrap();
            while stream.next().await.is_some() {}
            assert!(cw.status(&session.id).is_none());
        }

        // 2. 开启 Reasoning 且非 force_adaptive_thinking 的 Anthropic 请求应被安全门禁拦截
        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                reasoning_effort: Some(ReasoningEffort::High),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}

        let st = cw.status(&session.id).unwrap();
        assert!(!st.active);
        assert!(
            st.last_stop_reason
                .as_deref()
                .is_some_and(|r| r.contains("non-adaptive"))
        );

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn streaming_mode_cancels_timer_immediately_on_turn_end() {
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        ));
        let (ctx, llm, sessions) = setup_test_ctx(
            Arc::clone(&adapter),
            CacheWarmingConfig {
                mode: CacheWarmingMode::Streaming,
                show_cache_miss_notices: false,
            },
        )
        .await;
        let cw = ctx.cache_warming().unwrap();
        let session = sessions.create(None);

        ctx.emit(
            EVENT_AGENT_TURN_START,
            AgentTurnStartEvent {
                session: Arc::clone(&session),
                turn: 1,
                ts_ms: 1,
            },
        );
        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}
        assert!(cw.status(&session.id).unwrap().active);

        ctx.emit(
            EVENT_AGENT_TURN_END,
            AgentTurnEndEvent {
                session: Arc::clone(&session),
                turn: 1,
                reason: TurnEndReason::Completed,
                ts_ms: 2,
            },
        );
        assert!(!cw.status(&session.id).unwrap().active);

        // 推进超过 270 秒也不应触发保活请求
        tokio::time::advance(Duration::from_secs(290)).await;
        tokio::task::yield_now().await;
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 1);

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn sleep_drift_past_refresh_deadline_aborts_without_sending_request() {
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        let cw = ctx.cache_warming().unwrap();
        let session = sessions.create(None);

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}
        assert!(cw.status(&session.id).unwrap().active);

        // delay=270s, ttl=300s -> deadline=285s；一次性推进 295s 模拟系统休眠唤醒超期
        tokio::time::advance(Duration::from_secs(295)).await;
        settle().await;

        assert_eq!(
            adapter.calls.load(Ordering::SeqCst),
            1,
            "missed deadline must abort before sending stream_direct"
        );
        let st = cw.status(&session.id).unwrap();
        assert!(!st.active);
        assert_eq!(
            st.last_stop_reason.as_deref(),
            Some("cache refresh deadline missed")
        );

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn compaction_or_selection_change_cancels_pending_timer() {
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        let cw = ctx.cache_warming().unwrap();
        let session = sessions.create(None);

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}
        assert!(cw.status(&session.id).unwrap().active);

        session.compact_at(
            "summary",
            1,
            Some(150_000),
            Some(10_000),
            CompactionDigest::default(),
        );
        assert!(!cw.status(&session.id).unwrap().active);

        tokio::time::advance(Duration::from_secs(280)).await;
        tokio::task::yield_now().await;
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 1);

        // 重新触发一次后切换模型也应立刻取消
        let mut stream2 = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 2,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream2.next().await.is_some() {}
        assert!(cw.status(&session.id).unwrap().active);

        session.record_selection(Some("test-prov".into()), Some("other-model".into()), None);
        assert!(!cw.status(&session.id).unwrap().active);

        ctx.fiber().dispose().await;
    }

    #[cfg(feature = "compaction")]
    #[tokio::test]
    async fn warming_refreshes_compaction_last_activity_at_when_present() {
        use cordis::BoxFuture;
        use mizar_agent_loop::{
            AgentPreStepDecision, AgentPreStepInput, AgentStepEndEvent, EVENT_AGENT_PRE_STEP,
            EVENT_AGENT_STEP_END, StepOutcome,
        };
        use mizar_compaction::{CompactionConfig, CompactionRuntime};

        // short=11s -> delay_ms = 1_000ms，配合 cold_cache_idle_secs=1 验证真实保活刷新活跃时间
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(11, true),
            300_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        ctx.plugin(mizar_tools::ToolRuntime::plugin(), ())
            .await
            .unwrap();
        ctx.plugin(
            mizar_system_prompt::SystemPrompt::plugin(),
            Default::default(),
        )
        .await
        .unwrap();
        ctx.plugin(
            CompactionRuntime::plugin(),
            CompactionConfig {
                enabled: true,
                auto_compact: true,
                auto_prune: false,
                default_context_window: 2_000,
                soft_window_cap: 2_000,
                cold_cache_idle_secs: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(
            ctx.reflect()
                .get::<CompactionRuntime>("compaction")
                .is_some()
        );

        let session = sessions.create(None);
        for idx in 0..6 {
            session.append(SessionEvent::user(format!(
                "turn {idx} {}",
                "x".repeat(450)
            )));
            session.append(SessionEvent::assistant_text(format!(
                "reply {idx} {}",
                "y".repeat(450)
            )));
        }

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}

        ctx.emit(
            EVENT_AGENT_STEP_END,
            AgentStepEndEvent {
                session: Arc::clone(&session),
                turn: 1,
                step: 1,
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                outcome: StepOutcome::Completed,
                usage: None,
                ts_ms: 1,
            },
        );

        // 等待 1.2s：初始步骤已超过 1s cold_cache_idle_secs，但 1.0s 时保活已刷新 last_activity_at
        tokio::time::sleep(Duration::from_millis(1_200)).await;
        assert_eq!(
            ctx.cache_warming()
                .unwrap()
                .status(&session.id)
                .unwrap()
                .warm_count,
            1
        );

        let _ = ctx
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                AgentPreStepInput {
                    session: Arc::clone(&session),
                    turn: 2,
                    step: 1,
                    provider: Some("test-prov".into()),
                    model: Some("claude-sonnet".into()),
                    cancel: CancellationToken::new(),
                    inputs: Vec::new(),
                },
                |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        assert!(
            session.latest_compaction_boundary().is_none(),
            "cache_warm must reset compaction last_activity_at and prevent premature ColdCache compaction"
        );

        // 关闭保活后真正闲置 > 1s，ColdCache 应正常触发作为反证对照
        ctx.cache_warming().unwrap().set_mode(CacheWarmingMode::Off);
        tokio::time::sleep(Duration::from_millis(1_150)).await;
        let _ = ctx
            .waterfall(
                EVENT_AGENT_PRE_STEP,
                AgentPreStepInput {
                    session: Arc::clone(&session),
                    turn: 3,
                    step: 1,
                    provider: Some("test-prov".into()),
                    model: Some("claude-sonnet".into()),
                    cancel: CancellationToken::new(),
                    inputs: Vec::new(),
                },
                |inp| -> BoxFuture<'static, AgentPreStepDecision> {
                    Box::pin(async move { AgentPreStepDecision::Enter { inputs: inp.inputs } })
                },
            )
            .await;
        assert!(
            session.latest_compaction_boundary().is_some(),
            "once warming is disabled and idle exceeds 1s, ColdCache compaction should trigger"
        );

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn idle_warming_respects_thirty_minute_cap_without_resetting_started_at() {
        // 270s 周期，30m = 1800s -> 第 1..6 次触发在 270, 540, 810, 1080, 1350, 1620s，
        // 第 7 次预计在 1890s > 1800s，应当在第 6 次完成后停止续排（共 1 次初始 + 6 次保活 = 7 次调用）
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        let session = sessions.create(None);

        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}

        for _ in 0..8 {
            tokio::time::advance(Duration::from_secs(271)).await;
            tokio::task::yield_now().await;
            tokio::task::yield_now().await;
        }

        let st = ctx.cache_warming().unwrap().status(&session.id).unwrap();
        assert_eq!(st.warm_count, 6);
        assert!(!st.active);
        assert_eq!(
            st.last_stop_reason.as_deref(),
            Some("max warming age reached")
        );
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 7);

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn turn_end_reevaluates_idle_savings_and_cancels_when_below_five_cents() {
        // 100k prompt -> streaming(1.0) 净收益 ~$0.315 >= $0.05 放行；
        // turn_end 切回 idle(0.15) 后净收益 ~$0.02175 < $0.05，必须立刻取消定时器
        let adapter = Arc::new(WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            100_000,
        ));
        let (ctx, llm, sessions) =
            setup_test_ctx(Arc::clone(&adapter), CacheWarmingConfig::default()).await;
        let session = sessions.create(None);

        ctx.emit(
            EVENT_AGENT_TURN_START,
            AgentTurnStartEvent {
                session: Arc::clone(&session),
                turn: 1,
                ts_ms: 1,
            },
        );
        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}
        assert!(
            ctx.cache_warming()
                .unwrap()
                .status(&session.id)
                .unwrap()
                .active
        );

        ctx.emit(
            EVENT_AGENT_TURN_END,
            AgentTurnEndEvent {
                session: Arc::clone(&session),
                turn: 1,
                reason: TurnEndReason::Completed,
                ts_ms: 2,
            },
        );
        let st = ctx.cache_warming().unwrap().status(&session.id).unwrap();
        assert!(!st.active);
        assert_eq!(
            st.last_stop_reason.as_deref(),
            Some("expected idle savings below $0.05")
        );

        ctx.fiber().dispose().await;
    }

    #[tokio::test(start_paused = true)]
    async fn failed_warm_stream_stops_without_retrying_or_writing_usage() {
        let mut adapter_raw = WarmingTestAdapter::new(
            "anthropic-messages",
            rich_anthropic_model(300, true),
            300_000,
        );
        adapter_raw.fail_cache_warm = true;
        let adapter = Arc::new(adapter_raw);
        let (ctx, llm, sessions) = setup_test_ctx(
            Arc::clone(&adapter),
            CacheWarmingConfig {
                mode: CacheWarmingMode::Idle,
                show_cache_miss_notices: true,
            },
        )
        .await;
        let notices = Arc::new(Mutex::new(Vec::<CacheWarmingNoticeEvent>::new()));
        let notices_sub = Arc::clone(&notices);
        let _h = ctx
            .on(
                EVENT_CACHE_WARMING_NOTICE,
                move |ev: &CacheWarmingNoticeEvent| {
                    notices_sub.lock().unwrap().push(ev.clone());
                },
            )
            .unwrap();

        let session = sessions.create(None);
        let mut stream = llm
            .stream(GenerateOptions {
                provider: Some("test-prov".into()),
                model: Some("claude-sonnet".into()),
                session_id: Some(session.id.clone()),
                origin: RequestOrigin::SessionTurn {
                    session_id: session.id.clone(),
                    turn: 1,
                    step: 1,
                },
                ..Default::default()
            })
            .unwrap();
        while stream.next().await.is_some() {}

        tokio::time::advance(Duration::from_secs(271)).await;
        settle().await;

        // 仅发起 1 次原始 + 1 次失败的保活（0 重试）
        assert_eq!(adapter.calls.load(Ordering::SeqCst), 2);
        assert!(session.all_entries().is_empty());
        assert!(notices.lock().unwrap().is_empty());
        let st = ctx.cache_warming().unwrap().status(&session.id).unwrap();
        assert!(!st.active);
        assert_eq!(st.warm_count, 0);
        assert!(
            st.last_stop_reason
                .as_deref()
                .is_some_and(|r| r.contains("warm stream error"))
        );

        ctx.fiber().dispose().await;
    }
}
