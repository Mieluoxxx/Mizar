//! LLM 路由覆盖（CLI / 环境变量）、临时路由构造与激活路由校验逻辑。

#[cfg(feature = "router")]
use mizar_llm::LlmError;
use mizar_llm::ReasoningEffort;
#[cfg(feature = "router")]
use mizar_llm_router::{ProtocolKind, ProviderRouteConfig, RouterConfig, parse_protocol_api};
#[cfg(feature = "router")]
use thiserror::Error;

#[cfg(feature = "router")]
use crate::config::LoadedMizarConfig;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CliLlmOverride {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub protocol: Option<String>,
    pub base_url: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub context_window: Option<u64>,
}

pub fn parse_context_window(source: &str, value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<std::num::NonZeroU64>()
        .map(std::num::NonZeroU64::get)
        .map_err(|_| format!("{source} 必须为正整数"))
}

#[cfg(feature = "compaction")]
impl CliLlmOverride {
    pub fn apply_context_window(&self, env: &EnvLlmOverride, spine: &mut crate::AgentSpineConfig) {
        if let Some(cfg) = &mut spine.compaction {
            cfg.context_window_override = self.context_window.or(env.context_window);
        }
    }
}

#[cfg(feature = "router")]
#[derive(Debug, Error)]
pub enum LlmRouteError {
    #[error(
        "no LLM provider or route configured (pass --provider/--protocol/--base-url/--model with credentials, or configure models.json)"
    )]
    NoRouteConfigured,
    #[error(
        "unknown provider \"{0}\" (not in models.json and not a built-in provider; pass --protocol or --base-url to define an explicit route)"
    )]
    UnknownProvider(String),
    #[error("unsupported protocol \"{0}\" (expected completions, responses, or messages)")]
    InvalidProtocol(String),
    #[error("no model configured for provider \"{provider}\" (pass --model or set MODEL)")]
    MissingModel { provider: String },
}

#[cfg(feature = "router")]
fn parse_cli_protocol(raw: &str) -> Result<ProtocolKind, LlmRouteError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(LlmRouteError::InvalidProtocol(raw.to_string()));
    }
    match parse_protocol_api(trimmed) {
        Ok(ProtocolKind::Mock) | Err(_) => Err(LlmRouteError::InvalidProtocol(raw.to_string())),
        Ok(proto) => Ok(proto),
    }
}

#[cfg(feature = "router")]
fn is_builtin_provider(id: &str) -> bool {
    matches!(
        id.trim().to_ascii_lowercase().as_str(),
        "openai" | "openai-responses" | "anthropic" | "deepseek"
    )
}

#[cfg(feature = "router")]
fn apply_cli_overrides_to_route(
    route: &mut ProviderRouteConfig,
    effective_protocol: Option<ProtocolKind>,
    effective_base_url: Option<&str>,
    active_model: &str,
    explicit_api_key: Option<&str>,
) {
    if let Some(proto) = effective_protocol {
        route.protocol = proto;
        route.unsupported_api = None;
    }
    if let Some(url) = effective_base_url {
        route.base_url = Some(url.to_string());
    }
    if let Some(key) = explicit_api_key {
        route.api_key = Some(key.to_string());
    }
    route.model = Some(active_model.to_string());
    if let Some(m) = route.models.iter_mut().find(|m| m.id == active_model) {
        if let Some(proto) = effective_protocol {
            m.api = Some(proto.as_str().to_string());
        }
        if let Some(url) = effective_base_url {
            m.base_url = Some(url.to_string());
        }
    }
}

#[cfg(feature = "router")]
fn build_adhoc_provider_route(
    provider: &str,
    is_builtin: bool,
    effective_protocol: Option<ProtocolKind>,
    env: &EnvLlmOverride,
) -> ProviderRouteConfig {
    let lower = provider.trim().to_ascii_lowercase();
    if is_builtin {
        let mut route = match lower.as_str() {
            "anthropic" => {
                let mut r = ProviderRouteConfig::anthropic_messages("ANTHROPIC_API_KEY");
                r.api_key = env
                    .explicit_api_key
                    .clone()
                    .or_else(|| env.anthropic_api_key.clone());
                r
            }
            "deepseek" => {
                let mut r = ProviderRouteConfig::deepseek_completions("DEEPSEEK_API_KEY");
                r.api_key = env
                    .explicit_api_key
                    .clone()
                    .or_else(|| env.deepseek_api_key.clone());
                r
            }
            "openai-responses" => {
                let mut r = ProviderRouteConfig::openai_responses("OPENAI_API_KEY");
                r.api_key = env
                    .explicit_api_key
                    .clone()
                    .or_else(|| env.openai_api_key.clone());
                r
            }
            _ => {
                let mut r = ProviderRouteConfig::openai_completions("OPENAI_API_KEY");
                r.api_key = env
                    .explicit_api_key
                    .clone()
                    .or_else(|| env.openai_api_key.clone());
                r
            }
        };
        if let Some(proto) = effective_protocol {
            route.protocol = proto;
            if proto == ProtocolKind::AnthropicMessages && route.anthropic_version.is_none() {
                route = ProviderRouteConfig {
                    api_key: route.api_key,
                    api_key_env: route.api_key_env,
                    ..ProviderRouteConfig::anthropic_messages("ANTHROPIC_API_KEY")
                };
            }
        }
        return route;
    }

    let proto = effective_protocol.unwrap_or(ProtocolKind::OpenaiCompletions);
    let mut route = match proto {
        ProtocolKind::AnthropicMessages => ProviderRouteConfig::anthropic_messages("API_KEY"),
        ProtocolKind::OpenaiResponses => ProviderRouteConfig::openai_responses("API_KEY"),
        ProtocolKind::OpenaiCompletions | ProtocolKind::Mock => {
            ProviderRouteConfig::openai_completions("API_KEY")
        }
    };
    route.api_key = env
        .explicit_api_key
        .clone()
        .or_else(|| match proto {
            ProtocolKind::AnthropicMessages => env.anthropic_api_key.clone(),
            ProtocolKind::OpenaiResponses | ProtocolKind::OpenaiCompletions => env
                .openai_api_key
                .clone()
                .or_else(|| env.deepseek_api_key.clone()),
            ProtocolKind::Mock => None,
        })
        .or_else(|| env.provider_api_key.clone());
    route
}

#[cfg(feature = "router")]
fn build_unnamed_route(
    effective_protocol: Option<ProtocolKind>,
    effective_base_url: Option<&str>,
    env: &EnvLlmOverride,
) -> Result<ProviderRouteConfig, LlmRouteError> {
    if let Some(proto) = effective_protocol {
        let mut route = match proto {
            ProtocolKind::AnthropicMessages => {
                ProviderRouteConfig::anthropic_messages("ANTHROPIC_API_KEY")
            }
            ProtocolKind::OpenaiResponses => {
                ProviderRouteConfig::openai_responses("OPENAI_API_KEY")
            }
            ProtocolKind::OpenaiCompletions | ProtocolKind::Mock => {
                if effective_base_url.is_none()
                    && env.explicit_api_key.is_none()
                    && env.provider_hint == Some(EnvProviderKind::Deepseek)
                {
                    ProviderRouteConfig::deepseek_completions("DEEPSEEK_API_KEY")
                } else {
                    ProviderRouteConfig::openai_completions(if effective_base_url.is_some() {
                        "API_KEY"
                    } else {
                        "OPENAI_API_KEY"
                    })
                }
            }
        };
        route.api_key = env.explicit_api_key.clone().or_else(|| match proto {
            ProtocolKind::AnthropicMessages => env.anthropic_api_key.clone(),
            ProtocolKind::OpenaiResponses | ProtocolKind::OpenaiCompletions => {
                env.openai_api_key.clone().or_else(|| {
                    if effective_base_url.is_some()
                        || env.provider_hint == Some(EnvProviderKind::Deepseek)
                    {
                        env.deepseek_api_key.clone()
                    } else {
                        None
                    }
                })
            }
            ProtocolKind::Mock => None,
        });
        return Ok(route);
    }

    if effective_base_url.is_some() {
        let mut route = ProviderRouteConfig::openai_completions("API_KEY");
        route.api_key = env
            .explicit_api_key
            .clone()
            .or_else(|| env.openai_api_key.clone())
            .or_else(|| env.deepseek_api_key.clone())
            .or_else(|| env.provider_api_key.clone());
        return Ok(route);
    }

    if let Some(k) = env.explicit_api_key.clone() {
        let mut route = ProviderRouteConfig::openai_completions("API_KEY");
        route.api_key = Some(k);
        return Ok(route);
    }

    match env.provider_hint {
        Some(EnvProviderKind::Anthropic) => {
            let mut r = ProviderRouteConfig::anthropic_messages("ANTHROPIC_API_KEY");
            r.api_key = env.anthropic_api_key.clone();
            Ok(r)
        }
        Some(EnvProviderKind::Deepseek) => {
            let mut r = ProviderRouteConfig::deepseek_completions("DEEPSEEK_API_KEY");
            r.api_key = env.deepseek_api_key.clone();
            Ok(r)
        }
        Some(EnvProviderKind::Openai) => {
            let mut r = ProviderRouteConfig::openai_completions("OPENAI_API_KEY");
            r.api_key = env.openai_api_key.clone();
            Ok(r)
        }
        None => Err(LlmRouteError::NoRouteConfigured),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvProviderKind {
    Openai,
    Anthropic,
    Deepseek,
}

#[derive(Debug, Clone, Default)]
pub struct EnvLlmOverride {
    pub explicit_api_key: Option<String>,
    pub provider_api_key: Option<String>,
    pub provider_hint: Option<EnvProviderKind>,
    pub protocol: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub context_window: Option<u64>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub deepseek_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
}

impl EnvLlmOverride {
    pub fn from_env(cli: &CliLlmOverride) -> Result<Self, String> {
        Self::from_lookup(|key| match key {
            "MIZAR_CONTEXT_WINDOW" if cli.context_window.is_some() => None,
            "MIZAR_CONTEXT_WINDOW" => {
                std::env::var_os(key).map(|v| v.to_string_lossy().into_owned())
            }
            _ => std::env::var(key).ok(),
        })
    }

    pub fn from_lookup<F>(mut lookup: F) -> Result<Self, String>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let context_window = lookup("MIZAR_CONTEXT_WINDOW")
            .map(|v| parse_context_window("MIZAR_CONTEXT_WINDOW", &v))
            .transpose()?;
        let mut clean = |k: &str| lookup(k).filter(|s| !s.trim().is_empty());
        let openai_api_key = clean("OPENAI_API_KEY");
        let anthropic_api_key = clean("ANTHROPIC_API_KEY");
        let deepseek_api_key = clean("DEEPSEEK_API_KEY");
        let openrouter_api_key = clean("OPENROUTER_API_KEY");
        let (provider_api_key, provider_hint) = if let Some(k) = openai_api_key.clone() {
            (Some(k), Some(EnvProviderKind::Openai))
        } else if let Some(k) = anthropic_api_key.clone() {
            (Some(k), Some(EnvProviderKind::Anthropic))
        } else if let Some(k) = deepseek_api_key.clone() {
            (Some(k), Some(EnvProviderKind::Deepseek))
        } else {
            (None, None)
        };
        Ok(Self {
            explicit_api_key: clean("API_KEY"),
            provider_api_key,
            provider_hint,
            protocol: clean("PROTOCOL"),
            base_url: clean("BASE_URL"),
            model: clean("MODEL"),
            context_window,
            openai_api_key,
            anthropic_api_key,
            deepseek_api_key,
            openrouter_api_key,
        })
    }
}

#[cfg(feature = "router")]
#[derive(Debug, Clone)]
pub struct ResolvedLlmRouter {
    pub router: RouterConfig,
    pub active_provider: String,
    pub active_model: Option<String>,
}

#[cfg(feature = "router")]
impl ResolvedLlmRouter {
    /// 验证激活的 provider / model / 协议与凭据是否完整就绪，返回解析后的原始凭据以供脱敏登记。
    pub fn validate_active_route(&self) -> Result<Option<String>, LlmError> {
        let Some(route) = self.router.providers.get(&self.active_provider) else {
            return Err(LlmError::new(
                mizar_llm::LlmErrorCode::NoAdapter,
                format!("provider \"{}\" not found in router", self.active_provider),
            ));
        };
        if let Some(ref api) = route.unsupported_api {
            return Err(LlmError::invalid_request(format!(
                "unsupported api \"{api}\" on provider \"{}\"",
                self.active_provider
            )));
        }
        if route.protocol == ProtocolKind::Mock {
            return Err(LlmError::invalid_request(format!(
                "mock protocol is disabled in headless mode on provider \"{}\"",
                self.active_provider
            )));
        }
        let Some(model_id) = self
            .active_model
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Err(LlmError::invalid_request(format!(
                "no model selected for provider \"{}\"",
                self.active_provider
            )));
        };
        if let Some(m) = route.models.iter().find(|m| m.id == model_id)
            && let Some(ref m_api) = m.api
        {
            match parse_protocol_api(m_api) {
                Ok(ProtocolKind::Mock) | Err(_) => {
                    return Err(LlmError::invalid_request(format!(
                        "unsupported api \"{m_api}\" on model \"{model_id}\""
                    )));
                }
                Ok(_) => {}
            }
        }
        mizar_llm_router::resolve_credential(
            &self.active_provider,
            route.api_key.as_deref(),
            route.api_key_env.as_deref(),
        )
    }
}

#[cfg(feature = "router")]
impl LoadedMizarConfig {
    /// 按严格优先序（`CLI > Env > LoadedConfig`）解析并校验 LLM 路由，用于无头 `exec` 模式。
    pub fn resolve_llm_router_with_cli(
        &self,
        env: EnvLlmOverride,
        cli: CliLlmOverride,
    ) -> Result<ResolvedLlmRouter, LlmRouteError> {
        let clean = |s: Option<String>| -> Option<String> {
            s.and_then(|v| {
                let t = v.trim();
                (!t.is_empty()).then(|| t.to_string())
            })
        };

        let effective_protocol = match cli.protocol.as_deref() {
            Some(raw) => Some(parse_cli_protocol(raw)?),
            None => match env.protocol.as_deref() {
                Some(raw) => Some(parse_cli_protocol(raw)?),
                None => None,
            },
        };
        let effective_base_url = clean(cli.base_url).or_else(|| clean(env.base_url.clone()));
        let explicit_model = clean(cli.model)
            .or_else(|| clean(env.model.clone()))
            .or_else(|| clean(self.settings.default_model.clone()));

        let mut router = self.router.clone();

        if let Some(raw_provider) = cli.provider {
            let prov = raw_provider.trim().to_string();
            if prov.is_empty() {
                return Err(LlmRouteError::UnknownProvider(raw_provider));
            }
            if router.providers.contains_key(&prov) {
                let route = router.providers.get_mut(&prov).unwrap();
                let active_model = explicit_model
                    .or_else(|| route.default_model_id())
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| LlmRouteError::MissingModel {
                        provider: prov.clone(),
                    })?;
                apply_cli_overrides_to_route(
                    route,
                    effective_protocol,
                    effective_base_url.as_deref(),
                    &active_model,
                    env.explicit_api_key.as_deref(),
                );
                return Ok(ResolvedLlmRouter {
                    router,
                    active_provider: prov,
                    active_model: Some(active_model),
                });
            }

            let is_builtin = is_builtin_provider(&prov);
            if !is_builtin && effective_protocol.is_none() && effective_base_url.is_none() {
                return Err(LlmRouteError::UnknownProvider(prov));
            }

            let mut route = build_adhoc_provider_route(&prov, is_builtin, effective_protocol, &env);
            if let Some(url) = effective_base_url {
                route.base_url = Some(url);
                // 显式指定自定义 endpoint 时，禁止静默退回 gpt-4o / claude 默认模型名
                route.model = None;
            }
            if !is_builtin {
                route.model = None;
            }
            let active_model = explicit_model
                .or_else(|| route.default_model_id())
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| LlmRouteError::MissingModel {
                    provider: prov.clone(),
                })?;
            route.model = Some(active_model.clone());
            router.providers.insert(prov.clone(), route);
            return Ok(ResolvedLlmRouter {
                router,
                active_provider: prov,
                active_model: Some(active_model),
            });
        }

        // 未显式指定 --provider：
        // 若显式给了 CLI/Env 路由或配置为空，尝试构建无 models.json 临时路由；否则走已加载 models.json 挑选提供方。
        let adhoc_route_override = env.explicit_api_key.is_some()
            && (effective_protocol.is_some() || effective_base_url.is_some());
        if router.providers.is_empty() || adhoc_route_override {
            let has_any_signal = env.explicit_api_key.is_some()
                || env.provider_api_key.is_some()
                || effective_protocol.is_some()
                || effective_base_url.is_some();
            if !has_any_signal {
                return Err(LlmRouteError::NoRouteConfigured);
            }

            let mut route =
                build_unnamed_route(effective_protocol, effective_base_url.as_deref(), &env)?;
            if let Some(url) = effective_base_url {
                route.base_url = Some(url);
                route.model = None;
            }
            let active_model = explicit_model
                .or_else(|| route.default_model_id())
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| LlmRouteError::MissingModel {
                    provider: "default".to_string(),
                })?;
            route.model = Some(active_model.clone());
            router.providers.insert("default".to_string(), route);
            return Ok(ResolvedLlmRouter {
                router,
                active_provider: "default".to_string(),
                active_model: Some(active_model),
            });
        }

        let active_provider = self
            .settings
            .default_provider
            .as_ref()
            .filter(|name| router.providers.contains_key(*name))
            .cloned()
            .or_else(|| {
                let target_model = explicit_model.as_deref()?;
                router
                    .providers
                    .iter()
                    .find(|(_, r)| r.models.iter().any(|m| m.id == target_model))
                    .map(|(k, _)| k.clone())
            })
            .or_else(|| {
                router
                    .providers
                    .contains_key("default")
                    .then(|| "default".to_string())
            })
            .or_else(|| router.providers.keys().next().cloned())
            .ok_or(LlmRouteError::NoRouteConfigured)?;

        let route = router.providers.get_mut(&active_provider).unwrap();
        let active_model = explicit_model
            .or_else(|| route.default_model_id())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| LlmRouteError::MissingModel {
                provider: active_provider.clone(),
            })?;
        apply_cli_overrides_to_route(
            route,
            effective_protocol,
            effective_base_url.as_deref(),
            &active_model,
            env.explicit_api_key.as_deref(),
        );
        Ok(ResolvedLlmRouter {
            router,
            active_provider,
            active_model: Some(active_model),
        })
    }

    pub fn resolve_llm_router_with_env(&self, env: EnvLlmOverride) -> Option<ResolvedLlmRouter> {
        let mut router = self.router.clone();
        let fallback_key = env.explicit_api_key.clone().or(env.provider_api_key);

        if env.explicit_api_key.is_some() || (router.providers.is_empty() && fallback_key.is_some())
        {
            let env_hint = if env.explicit_api_key.is_some() {
                None
            } else {
                env.provider_hint
            };
            let mut route = if let Some(protocol) = env.protocol.as_deref() {
                match protocol.to_ascii_lowercase().as_str() {
                    "responses" | "openai-responses" => {
                        ProviderRouteConfig::openai_responses("API_KEY")
                    }
                    "messages" | "anthropic" | "anthropic-messages" => {
                        ProviderRouteConfig::anthropic_messages("API_KEY")
                    }
                    "deepseek" => ProviderRouteConfig::deepseek_completions("API_KEY"),
                    _ => match env_hint {
                        Some(EnvProviderKind::Deepseek) => {
                            ProviderRouteConfig::deepseek_completions("DEEPSEEK_API_KEY")
                        }
                        _ => ProviderRouteConfig::openai_completions("API_KEY"),
                    },
                }
            } else {
                match env_hint {
                    Some(EnvProviderKind::Anthropic) => {
                        ProviderRouteConfig::anthropic_messages("ANTHROPIC_API_KEY")
                    }
                    Some(EnvProviderKind::Deepseek) => {
                        ProviderRouteConfig::deepseek_completions("DEEPSEEK_API_KEY")
                    }
                    Some(EnvProviderKind::Openai) => {
                        ProviderRouteConfig::openai_completions("OPENAI_API_KEY")
                    }
                    None => ProviderRouteConfig::openai_completions("API_KEY"),
                }
            };
            route.api_key = fallback_key;
            if let Some(url) = env.base_url {
                route.base_url = Some(url);
            }
            if let Some(model) = &env.model {
                route.model = Some(model.clone());
            } else if route.model.is_none() {
                route.model = self.settings.default_model.clone();
            }
            let active_model = env.model.or_else(|| route.model.clone());
            router.providers.insert("default".to_string(), route);
            return Some(ResolvedLlmRouter {
                router,
                active_provider: "default".to_string(),
                active_model,
            });
        }

        if router.providers.is_empty() {
            return None;
        }

        let active_provider = self
            .settings
            .default_provider
            .as_ref()
            .filter(|name| router.providers.contains_key(*name))
            .cloned()
            .or_else(|| {
                let target_model = env
                    .model
                    .as_deref()
                    .or(self.settings.default_model.as_deref())?;
                router
                    .providers
                    .iter()
                    .find(|(_, r)| r.models.iter().any(|m| m.id == target_model))
                    .map(|(k, _)| k.clone())
            })
            .or_else(|| {
                router
                    .providers
                    .contains_key("default")
                    .then(|| "default".to_string())
            })
            .or_else(|| router.providers.keys().next().cloned())?;

        let mut active_model = env
            .model
            .clone()
            .or_else(|| self.settings.default_model.clone());
        if let Some(route) = router.providers.get_mut(&active_provider) {
            if let Some(url) = env.base_url {
                route.base_url = Some(url.clone());
                if let Some(target_id) = active_model.as_deref()
                    && let Some(m) = route.models.iter_mut().find(|m| m.id == target_id)
                {
                    m.base_url = Some(url);
                }
            }
            if let Some(model) = &env.model {
                route.model = Some(model.clone());
            }
            if active_model.is_none() {
                active_model = route.default_model_id();
            }
        }

        Some(ResolvedLlmRouter {
            router,
            active_provider,
            active_model,
        })
    }
}
