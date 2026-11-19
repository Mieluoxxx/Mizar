//! Pi 兼容的 `models.json` 配置数据结构与合并/归一化逻辑。
//!
//! 兼容两种写法：
//! 1. **Pi `models.json` 格式**（`~/.pi/agent/models.json`）：
//!    - 顶层支持 `$schema` 与 `providers`
//!    - provider 支持 `api`、`name`、`baseUrl`、`apiKey`（字面量 / `$VAR` / `${VAR}` / `!command`）、`authHeader`、`headers`、`compat`、`models`、`modelOverrides`
//!    - `models[]` 支持单模型覆盖 `api`、`baseUrl`、`headers`、`compat`、`reasoning`、`thinkingLevelMap`、`input`、`cost`、`contextWindow`、`maxTokens`、`promptCache`、`samplingParams`、`samplingParamsByThinkingLevel`
//! 2. **既有简化 `RouterConfig` 格式**：
//!    - `protocol`、`displayName`、`apiKeyEnv`、`useBearerAuth`、`model`、`reasoningEfforts` 等字段保持向后兼容

use std::collections::BTreeMap;

use mizar_llm::{ModelInfo, ReasoningEffort};
use mizar_llm_completions::{CompletionsCompat, MaxTokensField, ThinkingFormat};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProtocolKind {
    #[default]
    OpenaiCompletions,
    OpenaiResponses,
    AnthropicMessages,
    Mock,
}

impl ProtocolKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenaiCompletions => "openai-completions",
            Self::OpenaiResponses => "openai-responses",
            Self::AnthropicMessages => "anthropic-messages",
            Self::Mock => "mock",
        }
    }
}

pub fn parse_protocol_api(raw: &str) -> Result<ProtocolKind, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "openai-completions" | "completions" => Ok(ProtocolKind::OpenaiCompletions),
        "openai-responses" | "openai-codex-responses" | "azure-openai-responses" | "responses" => {
            Ok(ProtocolKind::OpenaiResponses)
        }
        "anthropic-messages" | "anthropic" | "messages" => Ok(ProtocolKind::AnthropicMessages),
        "mock" => Ok(ProtocolKind::Mock),
        other => Err(other.to_string()),
    }
}

/// Pi 兼容开关：既覆盖 Provider 默认项，也可在单个模型或 `modelOverrides` 上局部覆写。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "camelCase")]
pub struct CompatConfig {
    pub thinking_format: Option<String>,
    pub max_tokens_field: Option<MaxTokensField>,
    #[serde(alias = "includeUsage")]
    pub supports_usage_in_streaming: Option<bool>,
    pub supports_max_output_tokens: Option<bool>,
    pub force_adaptive_thinking: Option<bool>,
    pub send_session_affinity_headers: Option<bool>,
    pub supports_temperature: Option<bool>,
    pub supports_cache_control_on_tools: Option<bool>,
    pub supports_store: Option<bool>,
    pub supports_developer_role: Option<bool>,
    pub supports_reasoning_effort: Option<bool>,
}

impl From<CompletionsCompat> for CompatConfig {
    fn from(c: CompletionsCompat) -> Self {
        let fmt = match c.thinking_format {
            ThinkingFormat::None => "none",
            ThinkingFormat::DeepSeek => "deepseek",
            ThinkingFormat::OpenAI => "openai",
            ThinkingFormat::OpenRouter => "openrouter",
            ThinkingFormat::Qwen => "qwen",
        };
        Self {
            thinking_format: Some(fmt.to_string()),
            max_tokens_field: Some(c.max_tokens_field),
            supports_usage_in_streaming: Some(c.include_usage),
            ..Default::default()
        }
    }
}

impl CompatConfig {
    pub fn merge_overlay(&self, overlay: &Self) -> Self {
        Self {
            thinking_format: overlay
                .thinking_format
                .clone()
                .or_else(|| self.thinking_format.clone()),
            max_tokens_field: overlay.max_tokens_field.or(self.max_tokens_field),
            supports_usage_in_streaming: overlay
                .supports_usage_in_streaming
                .or(self.supports_usage_in_streaming),
            supports_max_output_tokens: overlay
                .supports_max_output_tokens
                .or(self.supports_max_output_tokens),
            force_adaptive_thinking: overlay
                .force_adaptive_thinking
                .or(self.force_adaptive_thinking),
            send_session_affinity_headers: overlay
                .send_session_affinity_headers
                .or(self.send_session_affinity_headers),
            supports_temperature: overlay.supports_temperature.or(self.supports_temperature),
            supports_cache_control_on_tools: overlay
                .supports_cache_control_on_tools
                .or(self.supports_cache_control_on_tools),
            supports_store: overlay.supports_store.or(self.supports_store),
            supports_developer_role: overlay
                .supports_developer_role
                .or(self.supports_developer_role),
            supports_reasoning_effort: overlay
                .supports_reasoning_effort
                .or(self.supports_reasoning_effort),
        }
    }

    pub fn to_completions_compat(&self, mut base: CompletionsCompat) -> CompletionsCompat {
        if let Some(fmt) = self.thinking_format.as_deref() {
            match fmt.trim().to_ascii_lowercase().as_str() {
                "none" | "off" => base.thinking_format = ThinkingFormat::None,
                "deepseek" => base.thinking_format = ThinkingFormat::DeepSeek,
                "openai" => base.thinking_format = ThinkingFormat::OpenAI,
                "openrouter" => base.thinking_format = ThinkingFormat::OpenRouter,
                "qwen" => base.thinking_format = ThinkingFormat::Qwen,
                _ => {}
            }
        }
        if let Some(field) = self.max_tokens_field {
            base.max_tokens_field = field;
        }
        if let Some(inc) = self.supports_usage_in_streaming {
            base.include_usage = inc;
        }
        base
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelCostTierConfig {
    pub input_tokens_above: u64,
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelCostConfig {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    #[serde(default)]
    pub tiers: Vec<ModelCostTierConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "camelCase")]
pub struct PromptCacheConfig {
    pub short: Option<u64>,
    pub long: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct RouterModelConfig {
    pub id: String,
    pub name: Option<String>,
    #[serde(alias = "protocol")]
    pub api: Option<String>,
    pub base_url: Option<String>,
    pub reasoning: Option<bool>,
    #[serde(default)]
    pub thinking_level_map: BTreeMap<String, Option<String>>,
    #[serde(default)]
    pub reasoning_efforts: Vec<ReasoningEffort>,
    #[serde(default)]
    pub input: Vec<String>,
    pub input_limits: Option<Value>,
    pub cost: Option<ModelCostConfig>,
    pub prompt_cache: Option<PromptCacheConfig>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub sampling_params: BTreeMap<String, Value>,
    #[serde(default)]
    pub sampling_params_by_thinking_level: BTreeMap<String, BTreeMap<String, Value>>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    pub compat: Option<CompatConfig>,
}

impl From<ModelInfo> for RouterModelConfig {
    fn from(info: ModelInfo) -> Self {
        Self {
            id: info.id,
            name: Some(info.name),
            reasoning: (!info.reasoning_efforts.is_empty()).then_some(true),
            reasoning_efforts: info.reasoning_efforts,
            context_window: info.context_window,
            max_tokens: info.max_tokens,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct ModelOverrideConfig {
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub thinking_level_map: Option<BTreeMap<String, Option<String>>>,
    pub input: Option<Vec<String>>,
    pub input_limits: Option<Value>,
    pub cost: Option<ModelCostConfig>,
    pub prompt_cache: Option<PromptCacheConfig>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u32>,
    pub sampling_params: Option<BTreeMap<String, Value>>,
    pub sampling_params_by_thinking_level: Option<BTreeMap<String, BTreeMap<String, Value>>>,
    pub headers: Option<BTreeMap<String, String>>,
    pub compat: Option<CompatConfig>,
}

const THINKING_ORDER: [ReasoningEffort; 6] = [
    ReasoningEffort::Minimal,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
    ReasoningEffort::Max,
];

pub fn effort_to_key(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Off => "off",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
}

pub fn parse_effort_key(raw: &str) -> Option<ReasoningEffort> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Some(ReasoningEffort::Off),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

impl RouterModelConfig {
    pub fn effective_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }

    pub fn apply_override(&mut self, ov: &ModelOverrideConfig) {
        if let Some(name) = &ov.name {
            self.name = Some(name.clone());
        }
        if let Some(reasoning) = ov.reasoning {
            self.reasoning = Some(reasoning);
        }
        if let Some(map) = &ov.thinking_level_map {
            for (k, v) in map {
                self.thinking_level_map.insert(k.clone(), v.clone());
            }
        }
        if let Some(input) = &ov.input {
            self.input = input.clone();
        }
        if let Some(limits) = &ov.input_limits {
            self.input_limits = Some(limits.clone());
        }
        if let Some(cost) = &ov.cost {
            self.cost = Some(cost.clone());
        }
        if let Some(cache) = &ov.prompt_cache {
            self.prompt_cache = Some(cache.clone());
        }
        if let Some(cw) = ov.context_window {
            self.context_window = Some(cw);
        }
        if let Some(mt) = ov.max_tokens {
            self.max_tokens = Some(mt);
        }
        if let Some(sp) = &ov.sampling_params {
            for (k, v) in sp {
                self.sampling_params.insert(k.clone(), v.clone());
            }
        }
        if let Some(sp_lvl) = &ov.sampling_params_by_thinking_level {
            for (lvl, map) in sp_lvl {
                let dst = self
                    .sampling_params_by_thinking_level
                    .entry(lvl.clone())
                    .or_default();
                for (k, v) in map {
                    dst.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(headers) = &ov.headers {
            for (k, v) in headers {
                self.headers.insert(k.clone(), v.clone());
            }
        }
        if let Some(compat) = &ov.compat {
            self.compat = Some(match &self.compat {
                Some(prev) => prev.merge_overlay(compat),
                None => compat.clone(),
            });
        }
    }

    pub fn supported_efforts(&self) -> Vec<ReasoningEffort> {
        if !self.reasoning_efforts.is_empty() {
            return self.reasoning_efforts.clone();
        }
        if !self.thinking_level_map.is_empty() {
            return THINKING_ORDER
                .iter()
                .copied()
                .filter(|eff| {
                    let key = effort_to_key(*eff);
                    match self.thinking_level_map.get(key) {
                        Some(Some(_)) => true,
                        Some(None) => false,
                        // Pi 规则：未在 thinkingLevelMap 映射的档位默认到 high 为止可用
                        None => !matches!(eff, ReasoningEffort::XHigh | ReasoningEffort::Max),
                    }
                })
                .collect();
        }
        if self.reasoning == Some(true) {
            return vec![
                ReasoningEffort::Minimal,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
            ];
        }
        Vec::new()
    }

    /// 按 Pi 的 `thinkingLevelMap` 与 `reasoning` 能力裁剪/映射推理档位。
    pub fn resolve_thinking_level(&self, requested: ReasoningEffort) -> Option<ReasoningEffort> {
        if self.reasoning == Some(false) {
            return None;
        }
        if self.thinking_level_map.is_empty() {
            return Some(requested);
        }

        if requested == ReasoningEffort::Off {
            return match self.thinking_level_map.get("off") {
                Some(Some(mapped)) => parse_effort_key(mapped),
                Some(None) => None,
                None => Some(ReasoningEffort::Off),
            };
        }

        let is_supported = |eff: ReasoningEffort| -> Option<ReasoningEffort> {
            let key = effort_to_key(eff);
            match self.thinking_level_map.get(key) {
                Some(Some(mapped)) => match parse_effort_key(mapped) {
                    Some(ReasoningEffort::Off) => None,
                    other => other.or(Some(eff)),
                },
                Some(None) => None,
                None => {
                    if matches!(eff, ReasoningEffort::XHigh | ReasoningEffort::Max) {
                        None
                    } else {
                        Some(eff)
                    }
                }
            }
        };

        if let Some(exact) = is_supported(requested) {
            return Some(exact);
        }

        let pos = THINKING_ORDER
            .iter()
            .position(|&e| e == requested)
            .unwrap_or(THINKING_ORDER.len() - 1);

        // 先向下兼容最接近的可用档位，再向上回退
        for idx in (0..pos).rev() {
            if let Some(found) = is_supported(THINKING_ORDER[idx]) {
                return Some(found);
            }
        }
        for eff in &THINKING_ORDER[pos + 1..] {
            if let Some(found) = is_supported(*eff) {
                return Some(found);
            }
        }
        None
    }

    pub fn effective_temperature(&self, effort: Option<ReasoningEffort>) -> Option<f32> {
        let from_level = effort.and_then(|e| {
            self.sampling_params_by_thinking_level
                .get(effort_to_key(e))
                .and_then(|m| m.get("temperature"))
                .and_then(Value::as_f64)
        });
        from_level
            .or_else(|| {
                self.sampling_params
                    .get("temperature")
                    .and_then(Value::as_f64)
            })
            .map(|v| v as f32)
    }

    pub fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.clone(),
            name: self.effective_name().to_string(),
            context_window: self.context_window,
            max_tokens: self.max_tokens,
            reasoning_efforts: self.supported_efforts(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProviderRouteConfig {
    /// 协议类型（兼容 Pi 的 `"api"` 与既有 `"protocol"`）。
    #[serde(alias = "api")]
    pub protocol: ProtocolKind,
    #[serde(skip)]
    pub unsupported_api: Option<String>,
    #[serde(alias = "name")]
    pub display_name: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub anthropic_version: Option<String>,
    #[serde(alias = "authHeader")]
    pub use_bearer_auth: bool,
    pub thinking_budget_tokens: Option<u32>,
    pub compat: Option<CompatConfig>,
    pub supports_max_output_tokens: Option<bool>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub models: Vec<RouterModelConfig>,
    #[serde(default)]
    pub model_overrides: BTreeMap<String, ModelOverrideConfig>,
}

impl Default for ProviderRouteConfig {
    fn default() -> Self {
        Self {
            protocol: ProtocolKind::OpenaiCompletions,
            unsupported_api: None,
            display_name: None,
            base_url: None,
            api_key: None,
            api_key_env: None,
            model: None,
            max_tokens: None,
            anthropic_version: None,
            use_bearer_auth: false,
            thinking_budget_tokens: None,
            compat: None,
            supports_max_output_tokens: None,
            headers: BTreeMap::new(),
            models: Vec::new(),
            model_overrides: BTreeMap::new(),
        }
    }
}

impl ProviderRouteConfig {
    pub fn openai_completions(api_key_env: impl Into<String>) -> Self {
        Self {
            protocol: ProtocolKind::OpenaiCompletions,
            display_name: Some("OpenAI Completions".to_string()),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key_env: Some(api_key_env.into()),
            model: Some("gpt-4o".to_string()),
            compat: Some(CompatConfig::from(CompletionsCompat::openai())),
            ..Self::default()
        }
    }

    pub fn deepseek_completions(api_key_env: impl Into<String>) -> Self {
        Self {
            protocol: ProtocolKind::OpenaiCompletions,
            display_name: Some("DeepSeek".to_string()),
            base_url: Some("https://api.deepseek.com/v1".to_string()),
            api_key_env: Some(api_key_env.into()),
            model: Some("deepseek-chat".to_string()),
            compat: Some(CompatConfig::from(CompletionsCompat::deepseek())),
            ..Self::default()
        }
    }

    pub fn openai_responses(api_key_env: impl Into<String>) -> Self {
        Self {
            protocol: ProtocolKind::OpenaiResponses,
            display_name: Some("OpenAI Responses".to_string()),
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key_env: Some(api_key_env.into()),
            model: Some("gpt-5".to_string()),
            supports_max_output_tokens: Some(true),
            ..Self::default()
        }
    }

    pub fn anthropic_messages(api_key_env: impl Into<String>) -> Self {
        Self {
            protocol: ProtocolKind::AnthropicMessages,
            display_name: Some("Anthropic Messages".to_string()),
            base_url: Some("https://api.anthropic.com".to_string()),
            api_key_env: Some(api_key_env.into()),
            model: Some("claude-sonnet-4-5".to_string()),
            max_tokens: Some(mizar_llm_messages::DEFAULT_MAX_TOKENS),
            anthropic_version: Some(mizar_llm_messages::DEFAULT_ANTHROPIC_VERSION.to_string()),
            ..Self::default()
        }
    }

    pub fn mock() -> Self {
        Self {
            protocol: ProtocolKind::Mock,
            display_name: Some("Offline Mock".to_string()),
            model: Some("mock-v1".to_string()),
            ..Self::default()
        }
    }

    /// 获取经过 `modelOverrides` 合并后的模型列表。
    pub fn resolved_models(&self) -> Vec<RouterModelConfig> {
        self.models
            .iter()
            .cloned()
            .map(|mut m| {
                if let Some(ov) = self.model_overrides.get(&m.id) {
                    m.apply_override(ov);
                }
                m
            })
            .collect()
    }

    /// 默认模型 ID：显式 `model` 优先，其次 `models` 第一项。
    pub fn default_model_id(&self) -> Option<String> {
        self.model
            .clone()
            .or_else(|| self.models.first().map(|m| m.id.clone()))
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct RawProviderRouteConfig {
    api: Option<String>,
    protocol: Option<String>,
    name: Option<String>,
    display_name: Option<String>,
    base_url: Option<String>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    model: Option<String>,
    max_tokens: Option<u32>,
    anthropic_version: Option<String>,
    #[serde(alias = "authHeader")]
    use_bearer_auth: Option<bool>,
    thinking_budget_tokens: Option<u32>,
    compat: Option<CompatConfig>,
    supports_max_output_tokens: Option<bool>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    models: Vec<RouterModelConfig>,
    #[serde(default)]
    model_overrides: BTreeMap<String, ModelOverrideConfig>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct RawRouterConfig {
    providers: BTreeMap<String, RawProviderRouteConfig>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RouterConfig {
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderRouteConfig>,
}

impl<'de> Deserialize<'de> for RouterConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawRouterConfig::deserialize(deserializer)?;
        let mut providers = BTreeMap::new();
        for (id, raw_route) in raw.providers {
            providers.insert(id.clone(), normalize_raw_provider(&id, raw_route));
        }
        Ok(RouterConfig { providers })
    }
}

fn builtin_protocol_hint(provider_id: &str) -> Option<ProtocolKind> {
    match provider_id.trim().to_ascii_lowercase().as_str() {
        "anthropic" => Some(ProtocolKind::AnthropicMessages),
        "openai" => Some(ProtocolKind::OpenaiCompletions),
        "openai-responses" => Some(ProtocolKind::OpenaiResponses),
        "mock" => Some(ProtocolKind::Mock),
        _ => None,
    }
}

fn builtin_env_hint(provider_id: &str) -> Option<&'static str> {
    match provider_id.trim().to_ascii_lowercase().as_str() {
        "openai" | "openai-responses" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        _ => None,
    }
}

fn normalize_raw_provider(provider_id: &str, raw: RawProviderRouteConfig) -> ProviderRouteConfig {
    let explicit_api = raw.api.or(raw.protocol);
    let mut unsupported_api = None;

    let protocol = if let Some(ref api_str) = explicit_api {
        match parse_protocol_api(api_str) {
            Ok(p) => p,
            Err(unknown) => {
                unsupported_api = Some(unknown);
                ProtocolKind::OpenaiCompletions
            }
        }
    } else if let Some(model_api) = raw.models.iter().find_map(|m| m.api.as_deref()) {
        match parse_protocol_api(model_api) {
            Ok(p) => p,
            Err(unknown) => {
                unsupported_api = Some(unknown);
                ProtocolKind::OpenaiCompletions
            }
        }
    } else {
        builtin_protocol_hint(provider_id).unwrap_or(ProtocolKind::OpenaiCompletions)
    };

    let api_key_env = if raw.api_key.is_none() && raw.api_key_env.is_none() {
        builtin_env_hint(provider_id).map(ToString::to_string)
    } else {
        raw.api_key_env
    };

    let base_url = raw.base_url.or_else(|| match protocol {
        ProtocolKind::AnthropicMessages => Some("https://api.anthropic.com".to_string()),
        ProtocolKind::OpenaiCompletions | ProtocolKind::OpenaiResponses => {
            Some("https://api.openai.com/v1".to_string())
        }
        ProtocolKind::Mock => None,
    });

    ProviderRouteConfig {
        protocol,
        unsupported_api,
        display_name: raw.display_name.or(raw.name),
        base_url,
        api_key: raw.api_key,
        api_key_env,
        model: raw.model,
        max_tokens: raw.max_tokens,
        anthropic_version: raw.anthropic_version,
        use_bearer_auth: raw.use_bearer_auth.unwrap_or(false),
        thinking_budget_tokens: raw.thinking_budget_tokens,
        compat: raw.compat,
        supports_max_output_tokens: raw.supports_max_output_tokens,
        headers: raw.headers,
        models: raw.models,
        model_overrides: raw.model_overrides,
    }
}
