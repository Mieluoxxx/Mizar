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
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub deepseek_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

impl EnvLlmOverride {
    pub fn from_env() -> Self {
        Self::from_lookup(env_non_empty)
    }

    pub fn from_lookup<F>(mut lookup: F) -> Self
    where
        F: FnMut(&str) -> Option<String>,
    {
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
        Self {
            explicit_api_key: clean("API_KEY"),
            provider_api_key,
            provider_hint,
            protocol: clean("PROTOCOL"),
            base_url: clean("BASE_URL"),
            model: clean("MODEL"),
            openai_api_key,
            anthropic_api_key,
            deepseek_api_key,
            openrouter_api_key,
        }
    }

    /// 收集环境变量中可能出现的所有非空 API Key（供诊断/日志脱敏）。
    pub fn collect_secrets(&self) -> Vec<String> {
        let mut out = Vec::new();
        for k in [
            &self.explicit_api_key,
            &self.provider_api_key,
            &self.openai_api_key,
            &self.anthropic_api_key,
            &self.deepseek_api_key,
            &self.openrouter_api_key,
        ]
        .into_iter()
        .flatten()
        {
            if !k.trim().is_empty() && !out.contains(k) {
                out.push(k.clone());
            }
        }
        out
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
    /// 结合环境变量覆盖与 `models.json` / `settings.json` 解析最终模型路由（TUI 兼容入口）。
    pub fn resolve_llm_router(&self) -> Option<ResolvedLlmRouter> {
        self.resolve_llm_router_with_env(EnvLlmOverride::from_env())
    }

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

#[cfg(all(test, feature = "router"))]
mod tests {
    use super::*;

    #[test]
    fn provider_specific_env_keys_infer_matching_endpoint_and_model() {
        let empty_cfg = LoadedMizarConfig::default();

        // 仅设 DEEPSEEK_API_KEY 时应走 DeepSeek 端点与 deepseek-chat，而非 OpenAI
        let ds_env = EnvLlmOverride::from_lookup(|k| match k {
            "DEEPSEEK_API_KEY" => Some("sk-ds".to_string()),
            _ => None,
        });
        let ds_resolved = empty_cfg
            .resolve_llm_router_with_env(ds_env)
            .expect("DEEPSEEK_API_KEY should resolve");
        let ds_route = &ds_resolved.router.providers["default"];
        assert_eq!(
            ds_route.protocol,
            mizar_llm_router::ProtocolKind::OpenaiCompletions
        );
        assert_eq!(
            ds_route.base_url.as_deref(),
            Some("https://api.deepseek.com/v1")
        );
        assert_eq!(ds_resolved.active_model.as_deref(), Some("deepseek-chat"));
        assert_eq!(ds_route.api_key.as_deref(), Some("sk-ds"));

        // 仅设 ANTHROPIC_API_KEY 时应走 Anthropic Messages 协议与 claude 模型
        let ant_env = EnvLlmOverride::from_lookup(|k| match k {
            "ANTHROPIC_API_KEY" => Some("sk-ant".to_string()),
            _ => None,
        });
        let ant_resolved = empty_cfg
            .resolve_llm_router_with_env(ant_env)
            .expect("ANTHROPIC_API_KEY should resolve");
        let ant_route = &ant_resolved.router.providers["default"];
        assert_eq!(
            ant_route.protocol,
            mizar_llm_router::ProtocolKind::AnthropicMessages
        );
        assert_eq!(
            ant_route.base_url.as_deref(),
            Some("https://api.anthropic.com")
        );
        assert_eq!(
            ant_resolved.active_model.as_deref(),
            Some("claude-sonnet-4-5")
        );

        // 显式 BASE_URL / MODEL 覆盖推断默认值
        let ds_custom = EnvLlmOverride::from_lookup(|k| match k {
            "DEEPSEEK_API_KEY" => Some("sk-ds".to_string()),
            "MODEL" => Some("deepseek-reasoner".to_string()),
            "BASE_URL" => Some("https://proxy.example.com/v1".to_string()),
            _ => None,
        });
        let custom_resolved = empty_cfg.resolve_llm_router_with_env(ds_custom).unwrap();
        let custom_route = &custom_resolved.router.providers["default"];
        assert_eq!(
            custom_route.base_url.as_deref(),
            Some("https://proxy.example.com/v1")
        );
        assert_eq!(
            custom_resolved.active_model.as_deref(),
            Some("deepseek-reasoner")
        );
    }
}

#[cfg(all(test, feature = "router"))]
mod exec_route_tests {
    use super::*;
    #[cfg(feature = "coding")]
    use crate::config::load_mizar_config_isolated;
    #[cfg(feature = "coding")]
    use crate::{AgentSpineConfig, AgentToolsMode};

    #[test]
    fn resolves_cli_precedence_custom_routes_and_rejects_invalid_routes() {
        let loaded = LoadedMizarConfig::default();

        // 空配置且无 CLI/Env 时拒绝收敛
        assert!(matches!(
            loaded
                .resolve_llm_router_with_cli(EnvLlmOverride::default(), CliLlmOverride::default()),
            Err(LlmRouteError::NoRouteConfigured)
        ));

        // 显式 protocol/baseUrl + 仅环境变量 ANTHROPIC_API_KEY（无需 models.json）
        let ant_env = EnvLlmOverride::from_lookup(|k| match k {
            "OPENAI_API_KEY" => Some("sk-oai".to_string()),
            "ANTHROPIC_API_KEY" => Some("sk-ant".to_string()),
            _ => None,
        });
        let r_ant = loaded
            .resolve_llm_router_with_cli(
                ant_env.clone(),
                CliLlmOverride {
                    protocol: Some("messages".to_string()),
                    base_url: Some("http://127.0.0.1:9999".to_string()),
                    model: Some("claude-eval".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        let route_ant = &r_ant.router.providers[&r_ant.active_provider];
        assert_eq!(route_ant.protocol, ProtocolKind::AnthropicMessages);
        assert_eq!(route_ant.base_url.as_deref(), Some("http://127.0.0.1:9999"));
        assert_eq!(route_ant.api_key.as_deref(), Some("sk-ant"));
        assert_eq!(r_ant.active_model.as_deref(), Some("claude-eval"));

        // 自定义 baseUrl 必须显式指定模型，不能静默落回 gpt-4o
        assert!(matches!(
            loaded.resolve_llm_router_with_cli(
                ant_env,
                CliLlmOverride {
                    protocol: Some("completions".to_string()),
                    base_url: Some("http://127.0.0.1:9999/v1".to_string()),
                    ..Default::default()
                },
            ),
            Err(LlmRouteError::MissingModel { .. })
        ));

        // 未知 provider 且无 --protocol/--base-url 必须报 UnknownProvider
        assert!(matches!(
            loaded.resolve_llm_router_with_cli(
                EnvLlmOverride::default(),
                CliLlmOverride {
                    provider: Some("unknown-gw".to_string()),
                    model: Some("m1".to_string()),
                    ..Default::default()
                },
            ),
            Err(LlmRouteError::UnknownProvider(_))
        ));

        // 禁用 mock 协议
        assert!(matches!(
            loaded.resolve_llm_router_with_cli(
                EnvLlmOverride::default(),
                CliLlmOverride {
                    protocol: Some("mock".to_string()),
                    ..Default::default()
                },
            ),
            Err(LlmRouteError::InvalidProtocol(_))
        ));
    }

    #[cfg(feature = "coding")]
    #[test]
    fn cli_protocol_and_base_url_override_models_json_model_entry_and_isolated_skips_global() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tmp =
            std::env::temp_dir().join(format!("mizar_cfg_cli_test_{}_{uniq}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let global = tmp.join("global");
        let ws = tmp.join("ws");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(ws.join(".mizar")).unwrap();

        std::fs::write(
            global.join("models.json"),
            r#"{"providers":{"personal":{"api":"openai-completions","baseUrl":"https://bad.invalid/v1","apiKey":"sk-personal","model":"p-model"}}}"#,
        )
        .unwrap();
        std::fs::write(
            ws.join(".mizar/models.json"),
            r#"{
                "providers": {
                    "gw": {
                        "api": "openai-completions",
                        "baseUrl": "http://127.0.0.1:1111/v1",
                        "apiKey": "sk-gw-secret",
                        "models": [
                            { "id": "m1", "api": "openai-responses", "baseUrl": "http://127.0.0.1:2222/v1" }
                        ]
                    }
                }
            }"#,
        )
        .unwrap();

        let isolated = load_mizar_config_isolated(&ws);
        assert!(isolated.isolated);
        assert!(!isolated.router.providers.contains_key("personal"));
        assert!(isolated.router.providers.contains_key("gw"));

        let mut spine = AgentSpineConfig::coding(&ws);
        isolated.apply_to_spine(&mut spine);
        match &spine.tools_mode {
            AgentToolsMode::Coding {
                mcp,
                skills,
                instructions,
                ..
            } => {
                assert!(mcp.skip_global_config);
                assert!(!skills.include_user_roots);
                assert!(!instructions.include_default_global);
            }
            _ => panic!("expected coding mode"),
        }

        // CLI --protocol 与 --base-url 必须同步覆写 route.models[m1]，保证 RoutedAdapter 生效
        let resolved = isolated
            .resolve_llm_router_with_cli(
                EnvLlmOverride::default(),
                CliLlmOverride {
                    provider: Some("gw".to_string()),
                    model: Some("m1".to_string()),
                    protocol: Some("messages".to_string()),
                    base_url: Some("http://127.0.0.1:3333".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        let gw = &resolved.router.providers["gw"];
        assert_eq!(gw.protocol, ProtocolKind::AnthropicMessages);
        assert_eq!(gw.base_url.as_deref(), Some("http://127.0.0.1:3333"));
        assert_eq!(gw.models[0].api.as_deref(), Some("anthropic-messages"));
        assert_eq!(
            gw.models[0].base_url.as_deref(),
            Some("http://127.0.0.1:3333")
        );
        assert_eq!(
            resolved.validate_active_route().unwrap().as_deref(),
            Some("sk-gw-secret")
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
