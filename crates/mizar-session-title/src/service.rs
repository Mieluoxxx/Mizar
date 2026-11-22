//! `SessionTitleRuntime` Cordis 服务（`"sessionTitle"`）：
//! 监听用户每轮输入事件，在后台异步生成并维护简洁的 `MMDD｜TYPE｜Topic` 会话标题。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Service, ServicePlugin};
use futures::StreamExt;
use mizar_agent_loop::{AgentTurnInputEvent, EVENT_AGENT_TURN_INPUT};
use mizar_llm::{
    BlockAssembler, FinishReason, GenerateOptions, LlmContextExt, LlmError, ReasoningEffort,
    RequestOrigin,
};
use mizar_session::{Session, SessionContextExt, TitleSource};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::format::{format_session_title, parse_model_title_output};
use crate::prompt::{build_title_messages, extract_user_messages};

pub const SERVICE_SESSION_TITLE: &str = "sessionTitle";
pub const DEFAULT_REFRESH_INTERVAL_TURNS: u64 = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTitleConfig {
    pub enabled: bool,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub refresh_interval_turns: u64,
}

impl Default for SessionTitleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: None,
            model: None,
            reasoning_effort: Some(ReasoningEffort::Low),
            refresh_interval_turns: DEFAULT_REFRESH_INTERVAL_TURNS,
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionTitleError {
    #[error("会话标题自动生成已关闭")]
    Disabled,
    #[error("会话缺少可用的用户输入，无法生成标题")]
    NoUserMessages,
    #[error("离线 Mock 模型不用于自动生成会话标题")]
    OfflineMock,
    #[error("当前会话正在生成标题，请稍候")]
    AlreadyRunning,
    #[error("模型服务不可用: {0}")]
    LlmUnavailable(String),
    #[error("模型调用失败: {0}")]
    Llm(#[from] LlmError),
    #[error("模型输出不符合 `TYPE｜Topic` 格式: {0}")]
    InvalidOutput(String),
}

#[derive(Debug, Default, Clone)]
struct SessionTitleState {
    last_auto_title: Option<String>,
    last_triggered_turn: Option<u64>,
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

pub struct SessionTitleRuntime {
    ctx: Context,
    config: Arc<Mutex<SessionTitleConfig>>,
    in_flight: Arc<Mutex<HashSet<String>>>,
    states: Arc<Mutex<HashMap<String, SessionTitleState>>>,
    dispose_token: CancellationToken,
}

impl SessionTitleRuntime {
    pub fn plugin() -> ServicePlugin<Self> {
        ServicePlugin::new()
    }

    pub fn config(&self) -> SessionTitleConfig {
        self.config.lock().unwrap().clone()
    }

    /// 热更新配置，供 `/settings` 或宿主同步切换使用。
    pub fn update_config(&self, f: impl FnOnce(&mut SessionTitleConfig)) {
        let mut guard = self.config.lock().unwrap();
        f(&mut guard);
    }

    fn try_lock_session(&self, session_id: &str) -> Result<InFlightGuard, SessionTitleError> {
        let mut guard = self.in_flight.lock().unwrap();
        if !guard.insert(session_id.to_string()) {
            return Err(SessionTitleError::AlreadyRunning);
        }
        Ok(InFlightGuard {
            in_flight: Arc::clone(&self.in_flight),
            session_id: session_id.to_string(),
        })
    }

    fn resolve_route(
        &self,
        cfg: &SessionTitleConfig,
        fallback_provider: Option<&str>,
        fallback_model: Option<&str>,
    ) -> Result<(Option<String>, Option<String>), SessionTitleError> {
        let llm = self
            .ctx
            .llm()
            .map_err(|err| SessionTitleError::LlmUnavailable(err.to_string()))?;

        let (provider, model) = if cfg.provider.is_some() || cfg.model.is_some() {
            (cfg.provider.clone(), cfg.model.clone())
        } else {
            (
                fallback_provider.map(str::to_string),
                fallback_model.map(str::to_string),
            )
        };

        let providers = llm.provider_infos();
        let chosen = provider
            .as_deref()
            .and_then(|pid| providers.iter().find(|p| p.id == pid))
            .or_else(|| providers.first());

        if let Some(info) = chosen
            && info.name == "Offline Mock"
        {
            return Err(SessionTitleError::OfflineMock);
        }

        Ok((provider, model))
    }

    fn handle_turn_input(self: &Arc<Self>, ev: &AgentTurnInputEvent) {
        if self.dispose_token.is_cancelled() {
            return;
        }
        let cfg = self.config();
        if !cfg.enabled {
            return;
        }

        let session = Arc::clone(&ev.session);
        let header = session.header();
        if header.title_source == Some(TitleSource::Manual) {
            return;
        }

        let cadence_hit = ev.turn == 1
            || (cfg.refresh_interval_turns > 0
                && ev.turn > 1
                && ev.turn.is_multiple_of(cfg.refresh_interval_turns));
        if !cadence_hit {
            return;
        }

        let in_flight_guard = match self.try_lock_session(&session.id) {
            Ok(guard) => guard,
            Err(_) => return,
        };

        let Ok((provider, model)) =
            self.resolve_route(&cfg, ev.provider.as_deref(), ev.model.as_deref())
        else {
            drop(in_flight_guard);
            return;
        };

        let user_messages = extract_user_messages(&session.all_entries());
        if user_messages.is_empty() {
            drop(in_flight_guard);
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            drop(in_flight_guard);
            return;
        };

        let expected_title = header.title.clone();
        let prev_triggered_turn = {
            let mut states = self.states.lock().unwrap();
            let st = states.entry(session.id.clone()).or_default();
            if st.last_triggered_turn == Some(ev.turn) {
                drop(in_flight_guard);
                return;
            }
            if let Some(last_auto) = &st.last_auto_title
                && header.title.as_deref() != Some(last_auto.as_str())
            {
                drop(in_flight_guard);
                return;
            }
            let prev = st.last_triggered_turn;
            st.last_triggered_turn = Some(ev.turn);
            prev
        };

        let rt = Arc::clone(self);
        let ts_ms = if header.created_at_ms > 0 {
            header.created_at_ms
        } else {
            ev.ts_ms
        };
        let cancel = self.dispose_token.child_token();
        let turn_num = ev.turn;
        let session_id_for_rollback = session.id.clone();

        handle.spawn(async move {
            let _guard = in_flight_guard;
            if let Err(err) = rt
                .run_generation_and_apply(GenerationParams {
                    session,
                    user_messages,
                    provider,
                    model,
                    reasoning_effort: cfg.reasoning_effort.or(Some(ReasoningEffort::Low)),
                    ts_ms,
                    expected_title,
                    respect_ownership: true,
                    cancel,
                })
                .await
            {
                {
                    let mut states = rt.states.lock().unwrap();
                    if let Some(st) = states.get_mut(&session_id_for_rollback)
                        && st.last_triggered_turn == Some(turn_num)
                    {
                        st.last_triggered_turn = prev_triggered_turn;
                    }
                }
                rt.ctx
                    .logger_named("session-title")
                    .debug(format!("后台生成会话标题未应用: {err}"));
            }
        });
    }

    /// 显式立即重生成当前会话标题（供 `/rename` 无参调用）。
    pub async fn regenerate_now(
        &self,
        session: Arc<Session>,
        provider: Option<String>,
        model: Option<String>,
    ) -> Result<String, SessionTitleError> {
        let cfg = self.config();
        let _guard = self.try_lock_session(&session.id)?;

        let sel = session.latest_selection();
        let fallback_provider = provider.or_else(|| sel.as_ref().and_then(|s| s.provider.clone()));
        let fallback_model = model.or_else(|| sel.as_ref().and_then(|s| s.model.clone()));
        let (eff_provider, eff_model) = self.resolve_route(
            &cfg,
            fallback_provider.as_deref(),
            fallback_model.as_deref(),
        )?;

        let user_messages = extract_user_messages(&session.all_entries());
        if user_messages.is_empty() {
            return Err(SessionTitleError::NoUserMessages);
        }

        let header = session.header();
        let ts_ms = header.created_at_ms;
        let expected_title = header.title.clone();
        let cancel = self.dispose_token.child_token();

        self.run_generation_and_apply(GenerationParams {
            session,
            user_messages,
            provider: eff_provider,
            model: eff_model,
            reasoning_effort: cfg.reasoning_effort.or(Some(ReasoningEffort::Low)),
            ts_ms,
            expected_title,
            respect_ownership: false,
            cancel,
        })
        .await
    }

    async fn run_generation_and_apply(
        &self,
        params: GenerationParams,
    ) -> Result<String, SessionTitleError> {
        if params.cancel.is_cancelled() {
            return Err(SessionTitleError::Llm(LlmError::cancelled()));
        }
        let llm = self
            .ctx
            .llm()
            .map_err(|err| SessionTitleError::LlmUnavailable(err.to_string()))?;

        let opts = GenerateOptions {
            messages: build_title_messages(&params.user_messages),
            tools: Vec::new(),
            model: params.model,
            provider: params.provider,
            system: None,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            reasoning_effort: params.reasoning_effort,
            headers: Vec::new(),
            session_id: None,
            cache_retention: None,
            cancel: params.cancel.clone(),
            origin: RequestOrigin::SessionTitle,
            disable_continuation: false,
        };

        let mut stream = llm.stream(opts)?;
        let mut assembler = BlockAssembler::new();
        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res?;
            assembler.push(chunk);
        }

        match assembler.finish() {
            FinishReason::Aborted => return Err(SessionTitleError::Llm(LlmError::cancelled())),
            FinishReason::Error {
                code,
                message,
                status,
                retry_after_ms,
            } => {
                return Err(SessionTitleError::Llm(LlmError {
                    code,
                    message,
                    status,
                    retry_after_ms,
                }));
            }
            FinishReason::Stop | FinishReason::ToolCalls | FinishReason::MaxTokens => {}
        }

        if params.cancel.is_cancelled() {
            return Err(SessionTitleError::Llm(LlmError::cancelled()));
        }

        let raw_output = assembler.text();
        let parsed = parse_model_title_output(&raw_output)
            .ok_or_else(|| SessionTitleError::InvalidOutput(raw_output.trim().to_string()))?;
        let formatted = format_session_title(params.ts_ms, &parsed);

        match params.session.compare_and_set_title_auto(
            params.expected_title.as_deref(),
            formatted.clone(),
            params.respect_ownership,
        ) {
            Ok(applied) => {
                let mut states = self.states.lock().unwrap();
                let st = states.entry(params.session.id.clone()).or_default();
                st.last_auto_title = Some(applied.clone());
                Ok(applied)
            }
            Err(current) => Ok(current.unwrap_or(formatted)),
        }
    }
}

struct GenerationParams {
    session: Arc<Session>,
    user_messages: Vec<String>,
    provider: Option<String>,
    model: Option<String>,
    reasoning_effort: Option<ReasoningEffort>,
    ts_ms: u128,
    expected_title: Option<String>,
    respect_ownership: bool,
    cancel: CancellationToken,
}

impl Service for SessionTitleRuntime {
    type Config = SessionTitleConfig;

    fn service_name() -> &'static str {
        SERVICE_SESSION_TITLE
    }

    fn plugin_name() -> &'static str {
        "session-title"
    }

    fn inject() -> Inject {
        Inject::from(["llm", "sessions"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        let _ = ctx.sessions()?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            config: Arc::new(Mutex::new(config)),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            states: Arc::new(Mutex::new(HashMap::new())),
            dispose_token: CancellationToken::new(),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let ctx = ctx.clone();
        Box::pin(async move {
            let rt = Arc::clone(&self);
            let c = ctx.clone();
            let cancel_on_dispose = self.dispose_token.clone();
            ctx.effect("session-title.turn-input", move || {
                let h = c.on(EVENT_AGENT_TURN_INPUT, move |ev: &AgentTurnInputEvent| {
                    rt.handle_turn_input(ev);
                })?;
                Ok::<_, CordisError>(Some(Cleanup::async_fn(move || async move {
                    cancel_on_dispose.cancel();
                    h.dispose().await;
                })))
            })?;
            Ok(())
        })
    }
}

pub trait SessionTitleContextExt {
    fn session_title(&self) -> Result<Arc<SessionTitleRuntime>, CordisError>;
}

impl SessionTitleContextExt for Context {
    fn session_title(&self) -> Result<Arc<SessionTitleRuntime>, CordisError> {
        self.service::<SessionTitleRuntime>(SERVICE_SESSION_TITLE)
    }
}
