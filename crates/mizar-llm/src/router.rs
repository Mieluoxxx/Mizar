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

use crate::completions::{CompletionsAdapter, CompletionsCompat, CompletionsConfig};
pub use crate::http::resolve_credential;
use crate::http::{DEFAULT_IDLE_TIMEOUT, expand_pi_value};
use crate::messages::{
    DEFAULT_ANTHROPIC_VERSION, DEFAULT_MAX_TOKENS, MessagesAdapter, MessagesConfig,
};
use crate::mock::MockAdapter;
use crate::responses::{ResponsesAdapter, ResponsesConfig};
use crate::{
    BoxChunkStream, GenerateOptions, LlmAdapter, LlmContextExt, LlmError, Message, ModelInfo,
    ProviderInfo, SessionAffinityMode, detect_session_affinity_mode, push_session_affinity_headers,
};
pub use config::{
    CompatConfig, ModelCostConfig, ModelCostTierConfig, ModelOverrideConfig, PromptCacheConfig,
    ProtocolKind, ProviderRouteConfig, RouterConfig, RouterModelConfig, parse_protocol_api,
};
use cordis::{BoxFuture, Cleanup, Context, CordisError, Inject, Service};
use futures::channel::mpsc;

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

fn append_resolved_headers(
    provider: &str,
    base: &BTreeMap<String, String>,
    overlay: Option<&BTreeMap<String, String>>,
    dst: &mut Vec<(String, String)>,
) -> Result<(), LlmError> {
    let mut merged = base.clone();
    if let Some(extra) = overlay {
        for (k, v) in extra {
            merged.insert(k.clone(), v.clone());
        }
    }
    for (k, raw_v) in merged {
        // 已存在同名请求头（调用方显式指定或先前已规整注入）则跳过，保证幂等且不覆盖调用方值
        if dst
            .iter()
            .any(|(existing, _)| existing.eq_ignore_ascii_case(&k))
        {
            continue;
        }
        let (val, _) = expand_pi_value(provider, &raw_v)?;
        dst.push((k, val));
    }
    Ok(())
}

fn strip_unsupported_images(messages: &mut [Message]) {
    for msg in messages {
        if let Message::User { content, images } = msg
            && !images.is_empty()
        {
            let count = images.len();
            images.clear();
            let placeholder = if count == 1 {
                "[image omitted: target model does not support image input]".to_string()
            } else {
                format!("[{count} images omitted: target model does not support image input]")
            };
            if content.trim().is_empty() {
                *content = placeholder;
            } else {
                content.push_str("\n\n");
                content.push_str(&placeholder);
            }
        }
    }
}

struct TargetRouteParams<'a> {
    protocol: ProtocolKind,
    adapter: Arc<dyn LlmAdapter>,
    overlay_headers: Option<&'a BTreeMap<String, String>>,
    base_url: Option<&'a str>,
    compat: Option<&'a CompatConfig>,
}

impl RoutedAdapter {
    fn resolve_target_params(&self, model: Option<&str>) -> TargetRouteParams<'_> {
        match model.and_then(|id| self.model_routes.get(id)) {
            Some(t) => TargetRouteParams {
                protocol: t.protocol,
                adapter: Arc::clone(&t.adapter),
                overlay_headers: Some(&t.headers),
                base_url: t.base_url.as_deref(),
                compat: t.compat.as_ref(),
            },
            None => TargetRouteParams {
                protocol: self.protocol,
                adapter: Arc::clone(&self.inner),
                overlay_headers: None,
                base_url: self.base_url.as_deref(),
                compat: self.compat.as_ref(),
            },
        }
    }

    fn model_supports_images(&self, model: Option<&str>) -> bool {
        model
            .and_then(|id| self.model_infos.iter().find(|m| m.id == id))
            .is_none_or(ModelInfo::supports_images)
    }

    fn normalize_options_internal(
        &self,
        mut options: GenerateOptions,
    ) -> (GenerateOptions, Option<LlmError>) {
        if options.model.is_none() {
            options.model = self.default_model.clone();
        }

        if let Some(target) = options
            .model
            .as_deref()
            .and_then(|id| self.model_routes.get(id))
        {
            if let Some(effort) = options.reasoning_effort {
                options.reasoning_effort = target.model_cfg.resolve_thinking_level(effort);
            }
            if options.max_tokens.is_none() {
                options.max_tokens = target.model_cfg.max_tokens;
            }
            if options.temperature.is_none() {
                options.temperature = target
                    .model_cfg
                    .effective_temperature(options.reasoning_effort);
            }
            if options.cache_retention.is_none()
                && let Some(cache) = &target.model_cfg.prompt_cache
            {
                options.cache_retention = cache.default_retention();
            }
        }

        if !self.model_supports_images(options.model.as_deref()) {
            strip_unsupported_images(&mut options.messages);
        }

        let target_params = self.resolve_target_params(options.model.as_deref());
        let header_err = append_resolved_headers(
            &self.provider,
            &self.provider_headers,
            target_params.overlay_headers,
            &mut options.headers,
        )
        .err();

        let eff_mode = target_params
            .compat
            .and_then(CompatConfig::parsed_session_affinity_mode)
            .unwrap_or_else(|| {
                detect_session_affinity_mode(&self.provider, target_params.base_url)
            });
        let default_send = match target_params.protocol {
            ProtocolKind::OpenaiResponses => true,
            ProtocolKind::OpenaiCompletions => eff_mode == SessionAffinityMode::OpenRouter,
            ProtocolKind::AnthropicMessages | ProtocolKind::Mock => false,
        };
        if target_params
            .compat
            .and_then(|c| c.send_session_affinity_headers)
            .unwrap_or(default_send)
        {
            push_session_affinity_headers(
                &mut options.headers,
                eff_mode,
                options.session_id.as_deref(),
                target_params.protocol == ProtocolKind::OpenaiCompletions,
            );
        }

        (options, header_err)
    }
}

impl LlmAdapter for RoutedAdapter {
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

    fn normalize_options(&self, options: GenerateOptions) -> GenerateOptions {
        self.normalize_options_internal(options).0
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
        let mut resolved_headers = Vec::new();
        if let Err(err) = append_resolved_headers(
            &self.provider,
            &self.provider_headers,
            None,
            &mut resolved_headers,
        ) {
            return Box::pin(async move { Err(err) });
        }
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

    fn stream(&self, options: GenerateOptions) -> BoxChunkStream {
        let (mut options, header_err) = self.normalize_options_internal(options);
        if let Some(err) = header_err {
            return error_stream(err);
        }

        let target_params = self.resolve_target_params(options.model.as_deref());
        let model = options.model.clone();
        let out = self.ctx.waterfall(
            EVENT_ROUTER_HEADERS,
            RouterHeaderInput {
                provider: self.provider.clone(),
                protocol: target_params.protocol,
                model,
                headers: std::mem::take(&mut options.headers),
            },
            |input| input,
        );
        options.headers = out.headers;
        target_params.adapter.stream(options)
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
                supports_mid_convo_effort: compat_cfg.supports_mid_convo_effort.unwrap_or(false),
                supports_additional_tools: compat_cfg.supports_additional_tools.unwrap_or(false),
                supports_tool_search_output: compat_cfg
                    .supports_tool_search_output
                    .unwrap_or(false),
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
        .map(|m| m.to_model_info_with_compat(route.compat.as_ref()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockAssembler, LlmErrorCode, LlmRuntime, Message};
    use axum::Router;
    use axum::body::Body;
    use axum::http::HeaderMap;
    use axum::response::Response;
    use axum::routing::post;
    use cordis::Next;
    use futures::StreamExt;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn routes_across_all_three_protocols_and_hot_reloads() {
        let seen_headers: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_clone = Arc::clone(&seen_headers);

        let server = Router::new()
            .route(
                "/chat/completions",
                post(move |headers: HeaderMap| {
                    let seen = Arc::clone(&seen_clone);
                    async move {
                        if let Some(v) = headers.get("x-trace") {
                            seen.lock().unwrap().push(v.to_str().unwrap().to_string());
                        }
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"from completions\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                            ))
                            .unwrap()
                    }
                }),
            )
            .route(
                "/responses",
                post(|| async {
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"from responses\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\"}}\n\n",
                        ))
                        .unwrap()
                }),
            )
            .route(
                "/v1/messages",
                post(|| async {
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"from messages\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                        ))
                        .unwrap()
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, server).await;
        });

        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();

        app.on_waterfall(
            EVENT_ROUTER_HEADERS,
            |mut input: RouterHeaderInput, next: Next<RouterHeaderInput, RouterHeaderInput>| {
                input.headers.push(("x-trace".into(), "trace-123".into()));
                next.call(input)
            },
        )
        .unwrap();

        let base = format!("http://{addr}");
        let mut providers = BTreeMap::new();
        providers.insert(
            "p-comp".to_string(),
            ProviderRouteConfig {
                api_key: Some("k1".into()),
                base_url: Some(base.clone()),
                ..ProviderRouteConfig::openai_completions("UNUSED")
            },
        );
        providers.insert(
            "p-resp".to_string(),
            ProviderRouteConfig {
                api_key: Some("k2".into()),
                base_url: Some(base.clone()),
                ..ProviderRouteConfig::openai_responses("UNUSED")
            },
        );
        providers.insert(
            "p-msg".to_string(),
            ProviderRouteConfig {
                api_key: Some("k3".into()),
                base_url: Some(base.clone()),
                ..ProviderRouteConfig::anthropic_messages("UNUSED")
            },
        );
        providers.insert(
            "p-missing".to_string(),
            ProviderRouteConfig::anthropic_messages("DEFINITELY_MISSING_KEY_XYZ"),
        );

        app.plugin(LlmRouter::plugin(), RouterConfig { providers })
            .await
            .unwrap();

        let llm = app.llm().unwrap();
        assert_eq!(llm.providers().len(), 4);

        for (prov, expected) in [
            ("p-comp", "from completions"),
            ("p-resp", "from responses"),
            ("p-msg", "from messages"),
        ] {
            let mut stream = llm
                .stream(GenerateOptions {
                    provider: Some(prov.into()),
                    messages: vec![Message::user("hi")],
                    ..Default::default()
                })
                .unwrap();
            let mut asm = BlockAssembler::new();
            while let Some(item) = stream.next().await {
                asm.push(item.unwrap());
            }
            assert_eq!(asm.text(), expected);
        }

        assert_eq!(*seen_headers.lock().unwrap(), vec!["trace-123".to_string()]);

        // 缺少环境变量的路由应在请求时明确返回 MissingCredential
        let mut missing_stream = llm
            .stream(GenerateOptions {
                provider: Some("p-missing".into()),
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .unwrap();
        let err = missing_stream.next().await.unwrap().unwrap_err();
        assert_eq!(err.code, LlmErrorCode::MissingCredential);

        // 热更新配置移除 p-missing
        let router = app.llm_router().unwrap();
        let mut updated = router.config();
        updated.providers.remove("p-missing");
        router.apply_config(updated).unwrap();
        assert_eq!(llm.providers().len(), 3);
    }

    #[tokio::test]
    async fn parses_pi_models_json_and_routes_per_model_api_compat_and_thinking() {
        type RecordedRequest = (String, serde_json::Value, Option<String>);
        let seen_bodies: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_comp = Arc::clone(&seen_bodies);
        let seen_msg = Arc::clone(&seen_bodies);

        let server = Router::new()
            .route(
                "/v1/chat/completions",
                post(move |headers: HeaderMap, body: axum::Json<serde_json::Value>| {
                    let seen = Arc::clone(&seen_comp);
                    async move {
                        let extra = headers
                            .get("x-dyn")
                            .and_then(|v| v.to_str().ok())
                            .map(ToString::to_string);
                        seen.lock()
                            .unwrap()
                            .push(("completions".to_string(), body.0, extra));
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"from gemini\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                            ))
                            .unwrap()
                    }
                }),
            )
            .route(
                "/v1/messages",
                post(move |headers: HeaderMap, body: axum::Json<serde_json::Value>| {
                    let seen = Arc::clone(&seen_msg);
                    async move {
                        let extra = headers
                            .get("x-dyn")
                            .and_then(|v| v.to_str().ok())
                            .map(ToString::to_string);
                        seen.lock()
                            .unwrap()
                            .push(("messages".to_string(), body.0, extra));
                        Response::builder()
                            .header("content-type", "text/event-stream")
                            .body(Body::from(
                                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"from claude\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                            ))
                            .unwrap()
                    }
                }),
            );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, server).await;
        });

        let pi_json = format!(
            r#"{{
              "$schema": "https://pi.dev/schemas/models.schema.json",
              "providers": {{
                "magpie": {{
                  "api": "openai-completions",
                  "apiKey": "!printf 'secret-from-cmd'",
                  "baseUrl": "http://{addr}/v1",
                  "name": "magpie",
                  "headers": {{ "x-dyn": "!printf 'hdr-cmd'" }},
                  "compat": {{ "sendSessionAffinityHeaders": true }},
                  "models": [
                    {{
                      "id": "mayoru-completions/gemini-4-hydrogen",
                      "name": "Gemini 4 Hydrogen · mayoru",
                      "contextWindow": 1048576,
                      "maxTokens": 65536,
                      "input": ["text", "image"],
                      "cost": {{ "input": 1.5, "output": 7.5, "cacheRead": 0.075, "cacheWrite": 0 }},
                      "reasoning": true,
                      "thinkingLevelMap": {{
                        "off": null,
                        "minimal": null,
                        "low": "low",
                        "medium": "medium",
                        "high": "high",
                        "xhigh": "xhigh",
                        "max": null
                      }}
                    }},
                    {{
                      "id": "mayoru-messages/claude-sonnet-5-5",
                      "name": "Claude Sonnet 5.5 · mayoru-messages",
                      "api": "anthropic-messages",
                      "baseUrl": "http://{addr}",
                      "compat": {{ "forceAdaptiveThinking": true }},
                      "promptCache": {{ "short": 300, "long": 3600 }},
                      "contextWindow": 1000000,
                      "maxTokens": 128000,
                      "reasoning": true,
                      "thinkingLevelMap": {{
                        "minimal": null,
                        "low": "low",
                        "medium": "medium",
                        "high": "high",
                        "xhigh": "xhigh",
                        "max": "max"
                      }}
                    }}
                  ]
                }}
              }}
            }}"#
        );

        let cfg: RouterConfig = serde_json::from_str(&pi_json).unwrap();
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(LlmRouter::plugin(), cfg).await.unwrap();

        let llm = app.llm().unwrap();
        let infos = llm.provider_infos();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, "magpie");
        assert_eq!(infos[0].models.len(), 2);
        assert_eq!(
            infos[0].models[0].input,
            vec!["text".to_string(), "image".to_string()]
        );
        assert!(infos[0].models[0].supports_images());
        assert_eq!(
            infos[0].models[0].cost,
            Some(crate::ModelCostRates {
                input: 1.5,
                output: 7.5,
                cache_read: 0.075,
                cache_write: 0.0,
                tiers: Vec::new(),
            })
        );
        assert!(!infos[0].models[0].force_adaptive_thinking);
        assert!(infos[0].models[1].input.is_empty());
        assert!(infos[0].models[1].supports_images());
        assert_eq!(
            infos[0].models[1].prompt_cache,
            Some(crate::PromptCacheLifetime {
                short: Some(300),
                long: Some(3600),
            })
        );
        assert!(infos[0].models[1].force_adaptive_thinking);
        assert_eq!(
            infos[0].models[0].reasoning_efforts,
            vec![
                crate::ReasoningEffort::Low,
                crate::ReasoningEffort::Medium,
                crate::ReasoningEffort::High,
                crate::ReasoningEffort::XHigh,
            ]
        );

        // 1. 默认 / Completions 模型：requested Max -> 裁剪为 XHigh
        let mut s1 = llm
            .stream(GenerateOptions {
                provider: Some("magpie".into()),
                model: Some("mayoru-completions/gemini-4-hydrogen".into()),
                reasoning_effort: Some(crate::ReasoningEffort::Max),
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .unwrap();
        let mut asm1 = BlockAssembler::new();
        while let Some(item) = s1.next().await {
            asm1.push(item.unwrap());
        }
        assert_eq!(asm1.text(), "from gemini");

        // 2. 同 Provider 下覆盖为 anthropic-messages 的模型：走 /v1/messages + forceAdaptiveThinking
        let mut s2 = llm
            .stream(GenerateOptions {
                provider: Some("magpie".into()),
                model: Some("mayoru-messages/claude-sonnet-5-5".into()),
                reasoning_effort: Some(crate::ReasoningEffort::Max),
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .unwrap();
        let mut asm2 = BlockAssembler::new();
        while let Some(item) = s2.next().await {
            asm2.push(item.unwrap());
        }
        assert_eq!(asm2.text(), "from claude");

        let recorded = seen_bodies.lock().unwrap();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].0, "completions");
        assert_eq!(recorded[0].1["reasoning_effort"], "xhigh");
        assert_eq!(recorded[0].2.as_deref(), Some("hdr-cmd"));

        assert_eq!(recorded[1].0, "messages");
        assert_eq!(recorded[1].1["max_tokens"], 128000);
        assert_eq!(recorded[1].1["thinking"]["type"], "adaptive");
        assert_eq!(recorded[1].1["output_config"]["effort"], "max");
    }

    #[tokio::test]
    async fn router_wires_session_affinity_and_prompt_cache_config_to_responses() {
        type ReqSnap = (
            Option<String>,
            Option<String>,
            Option<String>,
            serde_json::Value,
        );
        let seen: Arc<Mutex<Vec<ReqSnap>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_srv = Arc::clone(&seen);

        let server = Router::new().route(
            "/v1/responses",
            post(move |headers: HeaderMap, body: axum::Json<serde_json::Value>| {
                let seen = Arc::clone(&seen_srv);
                async move {
                    let h = |name: &str| {
                        headers
                            .get(name)
                            .and_then(|v| v.to_str().ok())
                            .map(ToString::to_string)
                    };
                    seen.lock().unwrap().push((
                        h("session_id"),
                        h("x-client-request-id"),
                        h("x-session-id"),
                        body.0,
                    ));
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"routed ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_r1\"}}\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, server).await;
        });

        let cfg_json = format!(
            r#"{{
              "providers": {{
                "oai": {{
                  "api": "openai-responses",
                  "apiKey": "k-oai",
                  "baseUrl": "http://{addr}/v1",
                  "compat": {{
                    "sendSessionAffinityHeaders": true,
                    "supportsLongCacheRetention": true
                  }},
                  "models": [
                    {{
                      "id": "gpt-long",
                      "promptCache": {{ "long": 3600 }}
                    }},
                    {{
                      "id": "gpt-or-hdr",
                      "compat": {{ "sessionAffinityFormat": "openrouter" }}
                    }}
                  ]
                }}
              }}
            }}"#
        );

        let cfg: RouterConfig = serde_json::from_str(&cfg_json).unwrap();
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(LlmRouter::plugin(), cfg).await.unwrap();

        let hook_seen_sess_hdr = Arc::new(Mutex::new(Vec::<Option<String>>::new()));
        let hook_cap = Arc::clone(&hook_seen_sess_hdr);
        app.on_waterfall(
            EVENT_ROUTER_HEADERS,
            move |input: RouterHeaderInput, next: Next<RouterHeaderInput, RouterHeaderInput>| {
                let found = input
                    .headers
                    .iter()
                    .find(|(k, _)| {
                        k.eq_ignore_ascii_case("session_id")
                            || k.eq_ignore_ascii_case("x-session-id")
                    })
                    .map(|(_, v)| v.clone());
                hook_cap.lock().unwrap().push(found);
                next.call(input)
            },
        )
        .unwrap();

        let llm = app.llm().unwrap();
        for model_id in ["gpt-long", "gpt-or-hdr"] {
            let mut s = llm
                .stream(GenerateOptions {
                    provider: Some("oai".into()),
                    model: Some(model_id.into()),
                    session_id: Some("sess-route-42".into()),
                    messages: vec![Message::user("hi")],
                    ..Default::default()
                })
                .unwrap();
            while let Some(c) = s.next().await {
                let _ = c.unwrap();
            }
        }

        let log = seen.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].0.as_deref(), Some("sess-route-42"));
        assert_eq!(log[0].1.as_deref(), Some("sess-route-42"));
        assert_eq!(log[0].2, None);
        assert_eq!(log[0].3["prompt_cache_key"], "sess-route-42");
        assert_eq!(log[0].3["prompt_cache_retention"], "24h");

        assert_eq!(log[1].0, None);
        assert_eq!(log[1].1, None);
        assert_eq!(log[1].2.as_deref(), Some("sess-route-42"));
        assert_eq!(log[1].3["prompt_cache_key"], "sess-route-42");
        assert!(log[1].3.get("prompt_cache_retention").is_none());

        assert_eq!(
            *hook_seen_sess_hdr.lock().unwrap(),
            vec![Some("sess-route-42".into()), Some("sess-route-42".into())]
        );
    }

    #[tokio::test]
    async fn normalize_options_is_idempotent_and_strips_images_for_text_only_models() {
        let seen_bodies: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_srv = Arc::clone(&seen_bodies);

        let server = Router::new().route(
            "/v1/chat/completions",
            post(move |body: axum::Json<serde_json::Value>| {
                let seen = Arc::clone(&seen_srv);
                async move {
                    seen.lock().unwrap().push(body.0);
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, server).await;
        });

        let cfg_json = format!(
            r#"{{
              "providers": {{
                "oai": {{
                  "api": "openai-completions",
                  "apiKey": "k-test",
                  "baseUrl": "http://{addr}/v1",
                  "headers": {{ "x-prov": "p-val" }},
                  "compat": {{
                    "sendSessionAffinityHeaders": true,
                    "sessionAffinityFormat": "openai",
                    "forceAdaptiveThinking": true
                  }},
                  "models": [
                    {{
                      "id": "text-only",
                      "input": ["text"],
                      "maxTokens": 4096,
                      "promptCache": {{ "long": 3600 }},
                      "samplingParams": {{ "temperature": 0.25 }},
                      "headers": {{ "x-model": "m-val" }},
                      "cost": {{ "input": 1.0, "output": 2.0, "cacheRead": 0.1, "cacheWrite": 0.0 }}
                    }},
                    {{
                      "id": "vision-model",
                      "input": ["text", "image"]
                    }}
                  ]
                }}
              }}
            }}"#
        );

        let cfg: RouterConfig = serde_json::from_str(&cfg_json).unwrap();
        let route = cfg.providers.get("oai").unwrap().clone();
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(LlmRouter::plugin(), cfg).await.unwrap();

        // 1. 直接验证 RoutedAdapter::normalize_options 幂等性与元数据传播
        let adapter = build_routed_adapter(&app, "oai", &route);
        let infos = adapter.info().unwrap().models;
        assert!(!infos[0].supports_images());
        assert!(infos[0].force_adaptive_thinking);
        assert!(infos[0].cost.is_some());
        assert!(infos[1].supports_images());

        let raw = GenerateOptions {
            provider: Some("oai".into()),
            session_id: Some("sess-idem".into()),
            headers: vec![("x-prov".into(), "caller-wins".into())],
            messages: vec![
                Message::user_with_images(
                    "inspect this",
                    vec![crate::ImageAttachment::new("image/png", "AQID")],
                ),
                Message::user_with_images(
                    "",
                    vec![
                        crate::ImageAttachment::new("image/png", "AQID"),
                        crate::ImageAttachment::new("image/jpeg", "/9j/"),
                    ],
                ),
            ],
            ..Default::default()
        };

        let once = adapter.normalize_options(raw);
        let twice = adapter.normalize_options(once.clone());
        assert_eq!(once.model, twice.model);
        assert_eq!(once.model.as_deref(), Some("text-only"));
        assert_eq!(once.max_tokens, Some(4096));
        assert_eq!(once.temperature, Some(0.25));
        assert_eq!(once.cache_retention, Some(crate::CacheRetention::Long));
        assert_eq!(once.headers, twice.headers);
        assert_eq!(once.messages, twice.messages);
        assert!(once.messages[0].images().is_empty());
        assert_eq!(
            once.messages[0].text_content(),
            "inspect this\n\n[image omitted: target model does not support image input]"
        );
        assert!(once.messages[1].images().is_empty());
        assert_eq!(
            once.messages[1].text_content(),
            "[2 images omitted: target model does not support image input]"
        );
        // 调用方显式设置的 x-prov 不被覆盖且不重复
        assert_eq!(
            once.headers
                .iter()
                .filter(|(k, _)| k.eq_ignore_ascii_case("x-prov"))
                .count(),
            1
        );
        assert_eq!(
            once.headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("x-prov"))
                .map(|(_, v)| v.as_str()),
            Some("caller-wins")
        );

        // 2. 通过 LlmRuntime::stream 验证 llm/stream 与 llm/after-response 快照一致反映剥离后的载荷
        let waterfall_images = Arc::new(Mutex::new(Vec::<usize>::new()));
        let wf_cap = Arc::clone(&waterfall_images);
        app.on_waterfall(
            crate::EVENT_LLM_STREAM,
            move |opts: GenerateOptions, next: Next<GenerateOptions, BoxChunkStream>| {
                let imgs: usize = opts.messages.iter().map(|m| m.images().len()).sum();
                wf_cap.lock().unwrap().push(imgs);
                next.call(opts)
            },
        )
        .unwrap();

        let after_events = Arc::new(Mutex::new(Vec::<crate::LlmAfterResponseEvent>::new()));
        let after_cap = Arc::clone(&after_events);
        app.on(
            crate::EVENT_LLM_AFTER_RESPONSE,
            move |ev: &crate::LlmAfterResponseEvent| {
                after_cap.lock().unwrap().push(ev.clone());
            },
        )
        .unwrap();

        let llm = app.llm().unwrap();
        let mut s_text = llm
            .stream(GenerateOptions {
                provider: Some("oai".into()),
                model: Some("text-only".into()),
                messages: vec![Message::user_with_images(
                    "look",
                    vec![crate::ImageAttachment::new("image/png", "AQID")],
                )],
                ..Default::default()
            })
            .unwrap();
        while let Some(c) = s_text.next().await {
            let _ = c.unwrap();
        }

        let mut s_vision = llm
            .stream(GenerateOptions {
                provider: Some("oai".into()),
                model: Some("vision-model".into()),
                messages: vec![Message::user_with_images(
                    "look",
                    vec![crate::ImageAttachment::new("image/png", "AQID")],
                )],
                ..Default::default()
            })
            .unwrap();
        while let Some(c) = s_vision.next().await {
            let _ = c.unwrap();
        }

        assert_eq!(*waterfall_images.lock().unwrap(), vec![0, 1]);
        let evs = after_events.lock().unwrap();
        assert_eq!(evs.len(), 2);
        assert!(evs[0].request.messages[0].images().is_empty());
        assert_eq!(evs[1].request.messages[0].images().len(), 1);

        let bodies = seen_bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        assert!(bodies[0]["messages"][0]["content"].is_string());
        assert!(bodies[1]["messages"][0]["content"].is_array());
    }

    #[tokio::test]
    async fn forwards_responses_compat_flags_to_request_body() {
        use crate::ToolSchema;
        use axum::Router;
        use axum::body::Body;
        use axum::response::Response;
        use axum::routing::post;
        use futures::StreamExt;
        use std::sync::Mutex;
        use tokio::net::TcpListener;

        let seen_bodies = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let cap = Arc::clone(&seen_bodies);
        let app_router = Router::new().route(
            "/responses",
            post(move |body: axum::Json<serde_json::Value>| {
                let cap = Arc::clone(&cap);
                async move {
                    cap.lock().unwrap().push(body.0);
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(
                            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_r\"}}\n\n",
                        ))
                        .unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app_router).await;
        });

        let cfg_json = format!(
            r#"{{
              "providers": {{
                "oai-resp": {{
                  "baseUrl": "http://{addr}",
                  "api": "openai-responses",
                  "apiKey": "k-resp",
                  "compat": {{
                    "supportsDeveloperRole": true,
                    "supportsStrictMode": true
                  }},
                  "models": [
                    {{ "id": "m-strict-dev" }},
                    {{
                      "id": "m-no-dev-top",
                      "compat": {{
                        "supportsDeveloperRole": false,
                        "supportsStrictMode": false,
                        "useTopLevelInstructions": true
                      }}
                    }}
                  ]
                }}
              }}
            }}"#
        );
        let cfg: RouterConfig = serde_json::from_str(&cfg_json).unwrap();
        let app = Context::new();
        app.plugin(LlmRuntime::plugin(), ()).await.unwrap();
        app.plugin(LlmRouter::plugin(), cfg).await.unwrap();

        let llm = app.llm().unwrap();
        let tool = ToolSchema {
            name: "ping".into(),
            description: "ping".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "host": { "type": "string" } },
                "required": ["host"]
            }),
        };

        for model_id in ["m-strict-dev", "m-no-dev-top"] {
            let mut s = llm
                .stream(GenerateOptions {
                    provider: Some("oai-resp".into()),
                    model: Some(model_id.into()),
                    messages: vec![Message::system("be crisp"), Message::user("hi")],
                    tools: vec![tool.clone()],
                    ..Default::default()
                })
                .unwrap();
            while let Some(chunk) = s.next().await {
                let _ = chunk.unwrap();
            }
        }

        let bodies = seen_bodies.lock().unwrap();
        assert_eq!(bodies.len(), 2);
        // 模型 1 继承 provider compat：developer 内联 + strict: true
        assert!(bodies[0].get("instructions").is_none());
        assert_eq!(bodies[0]["input"][0]["role"], "developer");
        assert_eq!(bodies[0]["tools"][0]["strict"], true);
        assert_eq!(
            bodies[0]["tools"][0]["parameters"]["additionalProperties"],
            false
        );

        // 模型 2 覆盖 compat：顶层 instructions + strict: false
        assert_eq!(bodies[1]["instructions"], "be crisp");
        assert_eq!(bodies[1]["input"][0]["role"], "user");
        assert_eq!(bodies[1]["tools"][0]["strict"], false);
    }
}
