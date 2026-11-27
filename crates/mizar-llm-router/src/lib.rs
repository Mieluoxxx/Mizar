//! 多提供方声明式路由与动态重载服务（兼容 Pi `~/.pi/agent/models.json` 与 `RouterConfig`）。
//!
//! 通过一份配置把多个提供方及单模型覆盖路由分别映射到 `openai-completions`、`openai-responses`、
//! `anthropic-messages` 或 `mock` 协议适配器，支持：
//! - `apiKey` / `apiKeyEnv` 按请求动态读取（支持字面量、`$VAR`、`${VAR}`、`!command`）
//! - 同一 Provider 下不同模型独立覆盖 `api`、`baseUrl`、`headers`、`compat`、`thinkingLevelMap`
//! - 运行时热更新配置（不可变适配器快照：在飞请求不中断，新请求立即生效）
//! - `llm-router/request-headers` waterfall 钩子动态注入请求头
//! - 统一 `provider_infos()` 与远端 `/models` 模型发现

pub mod config;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub use config::{
    CompatConfig, ModelCostConfig, ModelCostTierConfig, ModelOverrideConfig, PromptCacheConfig,
    ProtocolKind, ProviderRouteConfig, RouterConfig, RouterModelConfig, parse_protocol_api,
};
use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Service};
use futures::channel::mpsc;
use mizar_llm::{
    BoxChunkStream, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, ModelInfo, ProviderInfo,
    SessionAffinityMode, detect_session_affinity_mode, push_session_affinity_headers,
};
use mizar_llm_completions::{CompletionsAdapter, CompletionsCompat, CompletionsConfig};
pub use mizar_llm_http::resolve_credential;
use mizar_llm_http::{DEFAULT_IDLE_TIMEOUT, expand_pi_value};
use mizar_llm_messages::{
    DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS, MessagesAdapter, MessagesConfig,
};
use mizar_llm_mock::MockAdapter;
use mizar_llm_responses::{ResponsesAdapter, ResponsesConfig};

pub const EVENT_ROUTER_HEADERS: &str = "llm-router/request-headers";

#[derive(Debug, Clone)]
pub struct RouterHeaderInput {
    pub provider: String,
    pub protocol: ProtocolKind,
    pub model: Option<String>,
    pub headers: Vec<(String, String)>,
}

struct UnsupportedApiAdapter {
    provider: String,
    api: String,
    info: ProviderInfo,
}

impl LlmAdapter for UnsupportedApiAdapter {
    fn info(&self) -> Option<ProviderInfo> {
        Some(self.info.clone())
    }

    fn stream(&self, _options: GenerateOptions) -> BoxChunkStream {
        let (tx, rx) = mpsc::unbounded();
        let _ = tx.unbounded_send(Err(LlmError::invalid_request(format!(
            "unsupported api \"{}\" on provider \"{}\"",
            self.api, self.provider
        ))));
        Box::pin(rx)
    }
}

struct ResolvedModelTarget {
    protocol: ProtocolKind,
    adapter: Arc<dyn LlmAdapter>,
    headers: BTreeMap<String, String>,
    base_url: Option<String>,
    compat: Option<CompatConfig>,
    model_cfg: RouterModelConfig,
}

struct RoutedAdapter {
    ctx: Context,
    provider: String,
    display_name: Option<String>,
    protocol: ProtocolKind,
    default_model: Option<String>,
    base_url: Option<String>,
    compat: Option<CompatConfig>,
    provider_headers: BTreeMap<String, String>,
    model_infos: Vec<ModelInfo>,
    inner: Arc<dyn LlmAdapter>,
    model_routes: BTreeMap<String, ResolvedModelTarget>,
}

fn error_stream(err: LlmError) -> BoxChunkStream {
    let (tx, rx) = mpsc::unbounded();
    let _ = tx.unbounded_send(Err(err));
    Box::pin(rx)
}

fn resolve_header_map(
    provider: &str,
    base: &BTreeMap<String, String>,
    overlay: Option<&BTreeMap<String, String>>,
) -> Result<Vec<(String, String)>, LlmError> {
    let mut merged = base.clone();
    if let Some(extra) = overlay {
        for (k, v) in extra {
            merged.insert(k.clone(), v.clone());
        }
    }
    let mut out = Vec::with_capacity(merged.len());
    for (k, raw_v) in merged {
        let (val, _) = expand_pi_value(provider, &raw_v)?;
        out.push((k, val));
    }
    Ok(out)
}

impl LlmAdapter for RoutedAdapter {
    fn default_model(&self) -> Option<&str> {
        self.default_model
            .as_deref()
            .or_else(|| self.inner.default_model())
    }

    fn info(&self) -> Option<ProviderInfo> {
        Some(ProviderInfo {
            id: self.provider.clone(),
            name: self
                .display_name
                .clone()
                .unwrap_or_else(|| self.provider.clone()),
            protocol: self.protocol.as_str().to_string(),
            models: self.model_infos.clone(),
        })
    }

    fn discover_models(&self) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        self.discover_models_with_headers(&[])
    }

    fn discover_models_with_headers(
        &self,
        extra_headers: &[(String, String)],
    ) -> BoxFuture<'static, Result<Vec<ModelInfo>, LlmError>> {
        if !self.model_infos.is_empty() {
            let models = self.model_infos.clone();
            return Box::pin(async move { Ok(models) });
        }
        let mut resolved_headers =
            match resolve_header_map(&self.provider, &self.provider_headers, None) {
                Ok(h) => h,
                Err(err) => return Box::pin(async move { Err(err) }),
            };
        resolved_headers.extend(extra_headers.iter().cloned());
        let out = self.ctx.waterfall(
            EVENT_ROUTER_HEADERS,
            RouterHeaderInput {
                provider: self.provider.clone(),
                protocol: self.protocol,
                model: None,
                headers: resolved_headers,
            },
            |input| input,
        );
        self.inner.discover_models_with_headers(&out.headers)
    }

    fn stream(&self, mut options: GenerateOptions) -> BoxChunkStream {
        let model = options.model.clone().or_else(|| self.default_model.clone());
        if options.model.is_none() {
            options.model = model.clone();
        }

        let target = model.as_deref().and_then(|id| self.model_routes.get(id));

        let (protocol, adapter, overlay_headers, target_base_url, target_compat) = match target {
            Some(t) => {
                if let Some(effort) = options.reasoning_effort {
                    options.reasoning_effort = t.model_cfg.resolve_thinking_level(effort);
                }
                if options.max_tokens.is_none() {
                    options.max_tokens = t.model_cfg.max_tokens;
                }
                if options.temperature.is_none() {
                    options.temperature =
                        t.model_cfg.effective_temperature(options.reasoning_effort);
                }
                if options.cache_retention.is_none()
                    && let Some(cache) = &t.model_cfg.prompt_cache
                {
                    options.cache_retention = cache.default_retention();
                }
                (
                    t.protocol,
                    Arc::clone(&t.adapter),
                    Some(&t.headers),
                    t.base_url.as_deref(),
                    t.compat.as_ref(),
                )
            }
            None => (
                self.protocol,
                Arc::clone(&self.inner),
                None,
                self.base_url.as_deref(),
                self.compat.as_ref(),
            ),
        };

        let resolved_headers =
            match resolve_header_map(&self.provider, &self.provider_headers, overlay_headers) {
                Ok(h) => h,
                Err(err) => return error_stream(err),
            };
        if !resolved_headers.is_empty() {
            let mut combined = resolved_headers;
            combined.append(&mut options.headers);
            options.headers = combined;
        }

        let eff_mode = target_compat
            .and_then(CompatConfig::parsed_session_affinity_mode)
            .unwrap_or_else(|| detect_session_affinity_mode(&self.provider, target_base_url));
        let default_send = match protocol {
            ProtocolKind::OpenaiResponses => true,
            ProtocolKind::OpenaiCompletions => eff_mode == SessionAffinityMode::OpenRouter,
            ProtocolKind::AnthropicMessages | ProtocolKind::Mock => false,
        };
        if target_compat
            .and_then(|c| c.send_session_affinity_headers)
            .unwrap_or(default_send)
        {
            push_session_affinity_headers(
                &mut options.headers,
                eff_mode,
                options.session_id.as_deref(),
                protocol == ProtocolKind::OpenaiCompletions,
            );
        }

        let out = self.ctx.waterfall(
            EVENT_ROUTER_HEADERS,
            RouterHeaderInput {
                provider: self.provider.clone(),
                protocol,
                model,
                headers: std::mem::take(&mut options.headers),
            },
            |input| input,
        );
        options.headers = out.headers;
        adapter.stream(options)
    }
}

struct AdapterBuildSpec<'a> {
    provider: &'a str,
    display_name: Option<&'a str>,
    protocol: ProtocolKind,
    unsupported_api: Option<&'a str>,
    base_url: Option<&'a str>,
    api_key: Option<&'a str>,
    api_key_env: Option<&'a str>,
    default_model: Option<&'a str>,
    max_tokens: Option<u32>,
    anthropic_version: Option<&'a str>,
    use_bearer_auth: bool,
    thinking_budget_tokens: Option<u32>,
    compat: Option<&'a CompatConfig>,
    prompt_cache: Option<&'a PromptCacheConfig>,
    supports_max_output_tokens: Option<bool>,
    models: Vec<ModelInfo>,
}

fn build_protocol_adapter(spec: AdapterBuildSpec<'_>) -> Arc<dyn LlmAdapter> {
    if let Some(api) = spec.unsupported_api {
        return Arc::new(UnsupportedApiAdapter {
            provider: spec.provider.to_string(),
            api: api.to_string(),
            info: ProviderInfo {
                id: spec.provider.to_string(),
                name: spec.display_name.unwrap_or(spec.provider).to_string(),
                protocol: api.to_string(),
                models: spec.models,
            },
        });
    }

    let compat_cfg = spec.compat.cloned().unwrap_or_default();
    let default_cache_retention = spec
        .prompt_cache
        .and_then(PromptCacheConfig::default_retention);
    match spec.protocol {
        ProtocolKind::OpenaiCompletions => {
            let compat = compat_cfg.to_completions_compat(CompletionsCompat::openai());
            Arc::new(CompletionsAdapter::new(CompletionsConfig {
                provider: spec.provider.to_string(),
                display_name: spec.display_name.map(ToString::to_string),
                api_key: spec.api_key.map(ToString::to_string),
                api_key_env: spec.api_key_env.map(ToString::to_string),
                base_url: spec
                    .base_url
                    .unwrap_or("https://api.openai.com/v1")
                    .to_string(),
                model: spec.default_model.unwrap_or("gpt-4o").to_string(),
                compat,
                headers: Vec::new(),
                default_cache_retention,
                models: spec.models,
                idle_timeout: DEFAULT_IDLE_TIMEOUT,
            }))
        }
        ProtocolKind::OpenaiResponses => {
            let supports_max_output = spec
                .compat
                .and_then(|c| c.supports_max_output_tokens)
                .or(spec.supports_max_output_tokens)
                .unwrap_or(true);
            Arc::new(ResponsesAdapter::new(ResponsesConfig {
                provider: spec.provider.to_string(),
                display_name: spec.display_name.map(ToString::to_string),
                api_key: spec.api_key.map(ToString::to_string),
                api_key_env: spec.api_key_env.map(ToString::to_string),
                base_url: spec.base_url.map(ToString::to_string),
                model: spec.default_model.map(ToString::to_string),
                headers: Default::default(),
                supports_max_output_tokens: supports_max_output,
                supports_developer_role: compat_cfg.supports_developer_role.unwrap_or(true),
                use_top_level_instructions: compat_cfg.use_top_level_instructions.unwrap_or(false),
                supports_strict_mode: compat_cfg.supports_strict_mode.unwrap_or(false),
                send_session_affinity_headers: compat_cfg.send_session_affinity_headers,
                session_affinity_mode: compat_cfg.parsed_session_affinity_mode(),
                supports_long_cache_retention: compat_cfg
                    .supports_long_cache_retention
                    .unwrap_or(true),
                default_cache_retention,
                models: spec.models,
                idle_timeout: DEFAULT_IDLE_TIMEOUT,
            }))
        }
        ProtocolKind::AnthropicMessages => Arc::new(MessagesAdapter::new(MessagesConfig {
            provider: spec.provider.to_string(),
            display_name: spec.display_name.map(ToString::to_string),
            api_key: spec.api_key.map(ToString::to_string),
            api_key_env: spec.api_key_env.map(ToString::to_string),
            base_url: spec
                .base_url
                .unwrap_or("https://api.anthropic.com")
                .to_string(),
            model: spec
                .default_model
                .unwrap_or("claude-sonnet-4-5")
                .to_string(),
            default_max_tokens: spec.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            anthropic_version: spec
                .anthropic_version
                .unwrap_or(DEFAULT_ANTHROPIC_VERSION)
                .to_string(),
            use_bearer_auth: spec.use_bearer_auth,
            force_adaptive_thinking: compat_cfg.force_adaptive_thinking.unwrap_or(false),
            thinking_budget_tokens: spec.thinking_budget_tokens,
            headers: Vec::new(),
            models: spec.models,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
        })),
        ProtocolKind::Mock => Arc::new(MockAdapter),
    }
}

fn build_routed_adapter(
    ctx: &Context,
    provider: &str,
    route: &ProviderRouteConfig,
) -> RoutedAdapter {
    let resolved_models = route.resolved_models();
    let model_infos: Vec<ModelInfo> = resolved_models
        .iter()
        .map(RouterModelConfig::to_model_info)
        .collect();
    let default_model = route.default_model_id();
    let default_model_prompt_cache = default_model
        .as_deref()
        .and_then(|id| resolved_models.iter().find(|m| m.id == id))
        .and_then(|m| m.prompt_cache.as_ref());

    let inner = build_protocol_adapter(AdapterBuildSpec {
        provider,
        display_name: route.display_name.as_deref(),
        protocol: route.protocol,
        unsupported_api: route.unsupported_api.as_deref(),
        base_url: route.base_url.as_deref(),
        api_key: route.api_key.as_deref(),
        api_key_env: route.api_key_env.as_deref(),
        default_model: default_model.as_deref(),
        max_tokens: route.max_tokens,
        anthropic_version: route.anthropic_version.as_deref(),
        use_bearer_auth: route.use_bearer_auth,
        thinking_budget_tokens: route.thinking_budget_tokens,
        compat: route.compat.as_ref(),
        prompt_cache: default_model_prompt_cache,
        supports_max_output_tokens: route.supports_max_output_tokens,
        models: model_infos.clone(),
    });

    let mut model_routes = BTreeMap::new();
    for m in resolved_models {
        let mut unsupported_api = None;
        let model_protocol = match m.api.as_deref() {
            Some(raw_api) => match parse_protocol_api(raw_api) {
                Ok(p) => p,
                Err(unknown) => {
                    unsupported_api = Some(unknown);
                    route.protocol
                }
            },
            None => {
                unsupported_api = route.unsupported_api.clone();
                route.protocol
            }
        };

        let merged_compat = match (&route.compat, &m.compat) {
            (Some(p), Some(ov)) => Some(p.merge_overlay(ov)),
            (Some(p), None) => Some(p.clone()),
            (None, Some(ov)) => Some(ov.clone()),
            (None, None) => None,
        };
        let resolved_base_url = m.base_url.clone().or_else(|| route.base_url.clone());

        let model_adapter = build_protocol_adapter(AdapterBuildSpec {
            provider,
            display_name: route.display_name.as_deref(),
            protocol: model_protocol,
            unsupported_api: unsupported_api.as_deref(),
            base_url: resolved_base_url.as_deref(),
            api_key: route.api_key.as_deref(),
            api_key_env: route.api_key_env.as_deref(),
            default_model: Some(&m.id),
            max_tokens: m.max_tokens.or(route.max_tokens),
            anthropic_version: route.anthropic_version.as_deref(),
            use_bearer_auth: route.use_bearer_auth,
            thinking_budget_tokens: route.thinking_budget_tokens,
            compat: merged_compat.as_ref(),
            prompt_cache: m.prompt_cache.as_ref(),
            supports_max_output_tokens: route.supports_max_output_tokens,
            models: model_infos.clone(),
        });

        let model_id = m.id.clone();
        let model_headers = m.headers.clone();
        model_routes.insert(
            model_id,
            ResolvedModelTarget {
                protocol: model_protocol,
                adapter: model_adapter,
                headers: model_headers,
                base_url: resolved_base_url,
                compat: merged_compat,
                model_cfg: m,
            },
        );
    }

    RoutedAdapter {
        ctx: ctx.clone(),
        provider: provider.to_string(),
        display_name: route.display_name.clone(),
        protocol: route.protocol,
        default_model,
        base_url: route.base_url.clone(),
        compat: route.compat.clone(),
        provider_headers: route.headers.clone(),
        model_infos,
        inner,
        model_routes,
    }
}

struct RouterState {
    config: RouterConfig,
    active_routes: Vec<String>,
}

pub struct LlmRouter {
    ctx: Context,
    state: Mutex<RouterState>,
}

impl LlmRouter {
    pub fn config(&self) -> RouterConfig {
        self.state.lock().unwrap().config.clone()
    }

    /// 原子更新路由表：先在单一锁下计算差量与更新 `config`，解锁后再一次性同步给 `LlmRuntime`（仅触发一次更新事件）。
    pub fn apply_config(&self, next: RouterConfig) -> Result<(), CordisError> {
        let llm = self.ctx.llm()?;

        let (removed, upserts) = {
            let mut state = self.state.lock().unwrap();
            let removed: Vec<String> = state
                .active_routes
                .iter()
                .filter(|old_key| !next.providers.contains_key(*old_key))
                .cloned()
                .collect();

            let mut upserts = Vec::with_capacity(next.providers.len());
            let mut next_active = Vec::with_capacity(next.providers.len());
            for (provider, route) in &next.providers {
                let adapter: Arc<dyn LlmAdapter> =
                    Arc::new(build_routed_adapter(&self.ctx, provider, route));
                upserts.push((provider.clone(), adapter));
                next_active.push(provider.clone());
            }

            state.active_routes = next_active;
            state.config = next;
            (removed, upserts)
        };

        llm.sync_adapters(&removed, upserts);
        Ok(())
    }

    pub fn clear(&self) {
        let keys = {
            let mut state = self.state.lock().unwrap();
            std::mem::take(&mut state.active_routes)
        };
        if let Ok(llm) = self.ctx.llm() {
            llm.sync_adapters(&keys, Vec::new());
        }
    }
}

impl Service for LlmRouter {
    type Config = RouterConfig;

    fn service_name() -> &'static str {
        "llmRouter"
    }

    fn plugin_name() -> &'static str {
        "LlmRouter"
    }

    fn inject() -> Inject {
        Inject::from(["llm"])
    }

    fn create(ctx: &Context, config: Self::Config) -> Result<Arc<Self>, CordisError> {
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            state: Mutex::new(RouterState {
                config,
                active_routes: Vec::new(),
            }),
        }))
    }

    fn init(self: Arc<Self>, ctx: &Context) -> BoxFuture<'static, Result<(), CordisError>> {
        let router = Arc::clone(&self);
        let ctx = ctx.clone();
        Box::pin(async move {
            let initial = router.config();
            router.apply_config(initial)?;

            let weak = Arc::downgrade(&router);
            ctx.effect("llmRouter.routes", move || {
                Ok::<_, CordisError>(Some(Cleanup::sync(move || {
                    if let Some(r) = weak.upgrade() {
                        r.clear();
                    }
                })))
            })?;
            Ok(())
        })
    }
}

pub trait LlmRouterContextExt {
    fn llm_router(&self) -> Result<Arc<LlmRouter>, CordisError>;
}

impl LlmRouterContextExt for Context {
    fn llm_router(&self) -> Result<Arc<LlmRouter>, CordisError> {
        self.service::<LlmRouter>("llmRouter")
    }
}
