//! 全局与项目级配置加载：统一存放在 `~/.dsh-rs/` 与 `<workspace>/.dsh-rs/`。
//!
//! 支持的配置文件：
//! - `settings.json`：默认模型、推理强度、工具预设/过滤、codemode 与 tool_search 选项
//! - `models.json`：多提供方路由（复用 `dsh_llm_router::RouterConfig` 格式）
//! - `mcp.json`：MCP Server 配置（由 `dsh-mcp` 从同一目录加载）

use std::path::{Path, PathBuf};

use dsh_llm::ReasoningEffort;
#[cfg(feature = "router")]
use dsh_llm_router::{ProviderRouteConfig, RouterConfig};
#[cfg(feature = "coding")]
use dsh_tools_coding::CodingToolsPreset;
use serde::{Deserialize, Serialize};

#[cfg(feature = "coding")]
use crate::AgentToolsMode;
use crate::AgentSpineConfig;

/// 解析 `${VAR}` 环境变量占位符（与 `dsh-mcp` 配置解析保持一致，避免跨 crate 强耦合）。
#[cfg(feature = "router")]
fn expand_env_vars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        if let Some(end) = after.find('}') {
            let var_name = &after[..end];
            if let Ok(val) = std::env::var(var_name) {
                out.push_str(&val);
            }
            rest = &after[end + 1..];
        } else {
            out.push_str("${");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

pub const CONFIG_DIR_NAME: &str = ".dsh-rs";

/// 全局配置目录（支持 `DSH_RS_HOME` 环境变量覆盖，默认 `~/.dsh-rs`）。
pub fn resolve_global_config_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("DSH_RS_HOME") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    dirs::home_dir().map(|home| home.join(CONFIG_DIR_NAME))
}

pub fn resolve_project_config_dir(workspace: &Path) -> PathBuf {
    workspace.join(CONFIG_DIR_NAME)
}

/// 解析默认持久化 SQLite 数据库路径：支持 `DSH_RS_SESSION_DB` 环境变量覆盖
/// （设为 `"none"` / `"off"` / `"memory"` 可禁用自动落盘），默认 `~/.dsh-rs/sessions.sqlite`。
pub fn resolve_default_session_db_path() -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("DSH_RS_SESSION_DB") {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("none")
            || trimmed.eq_ignore_ascii_case("off")
            || trimmed.is_empty()
        {
            return None;
        }
        return Some(PathBuf::from(trimmed));
    }
    resolve_global_config_dir().map(|dir| dir.join("sessions.sqlite"))
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct CodemodeSettings {
    pub auto_enable: Option<bool>,
    pub only_mode: Option<bool>,
    pub inline_budget: Option<usize>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct ToolSearchSettings {
    pub auto_enable: Option<bool>,
    pub default_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct SkillsSettings {
    pub enabled: Option<bool>,
    #[serde(default)]
    pub custom_dirs: Vec<PathBuf>,
    pub bundled_dir: Option<PathBuf>,
    pub catalog_description_max_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct SettingsConfig {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    #[serde(alias = "defaultThinkingLevel")]
    pub reasoning_effort: Option<ReasoningEffort>,
    pub max_steps: Option<usize>,
    pub persona: Option<String>,
    /// `"all"` | `"minimal"` | `"read-only"`
    pub tools_preset: Option<String>,
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_tools: Vec<String>,
    pub restrict_to_workspace: Option<bool>,
    pub session_db: Option<PathBuf>,
    pub codemode: Option<CodemodeSettings>,
    pub tool_search: Option<ToolSearchSettings>,
    pub skills: Option<SkillsSettings>,
}

impl SettingsConfig {
    fn merge_overlay(&mut self, other: SettingsConfig) {
        if other.default_provider.is_some() {
            self.default_provider = other.default_provider;
        }
        if other.default_model.is_some() {
            self.default_model = other.default_model;
        }
        if other.reasoning_effort.is_some() {
            self.reasoning_effort = other.reasoning_effort;
        }
        if other.max_steps.is_some() {
            self.max_steps = other.max_steps;
        }
        if other.persona.is_some() {
            self.persona = other.persona;
        }
        if other.tools_preset.is_some() {
            self.tools_preset = other.tools_preset;
        }
        if other.tools.is_some() {
            self.tools = other.tools;
        }
        for pat in other.exclude_tools {
            if !self.exclude_tools.contains(&pat) {
                self.exclude_tools.push(pat);
            }
        }
        if other.restrict_to_workspace.is_some() {
            self.restrict_to_workspace = other.restrict_to_workspace;
        }
        if other.session_db.is_some() {
            self.session_db = other.session_db;
        }
        if let Some(cm) = other.codemode {
            let dst = self.codemode.get_or_insert_with(Default::default);
            if cm.auto_enable.is_some() {
                dst.auto_enable = cm.auto_enable;
            }
            if cm.only_mode.is_some() {
                dst.only_mode = cm.only_mode;
            }
            if cm.inline_budget.is_some() {
                dst.inline_budget = cm.inline_budget;
            }
            if cm.timeout_ms.is_some() {
                dst.timeout_ms = cm.timeout_ms;
            }
        }
        if let Some(ts) = other.tool_search {
            let dst = self.tool_search.get_or_insert_with(Default::default);
            if ts.auto_enable.is_some() {
                dst.auto_enable = ts.auto_enable;
            }
            if ts.default_limit.is_some() {
                dst.default_limit = ts.default_limit;
            }
        }
        if let Some(sk) = other.skills {
            let dst = self.skills.get_or_insert_with(Default::default);
            if sk.enabled.is_some() {
                dst.enabled = sk.enabled;
            }
            for dir in sk.custom_dirs {
                if !dst.custom_dirs.contains(&dir) {
                    dst.custom_dirs.push(dir);
                }
            }
            if sk.bundled_dir.is_some() {
                dst.bundled_dir = sk.bundled_dir;
            }
            if sk.catalog_description_max_length.is_some() {
                dst.catalog_description_max_length = sk.catalog_description_max_length;
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadedDshConfig {
    pub global_dir: Option<PathBuf>,
    pub project_dir: PathBuf,
    pub settings: SettingsConfig,
    #[cfg(feature = "router")]
    pub router: RouterConfig,
    pub loaded_files: Vec<PathBuf>,
    pub errors: Vec<String>,
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let no_bom = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
    let parsed: T = serde_json::from_str(no_bom).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Some(parsed))
}

#[cfg(feature = "router")]
fn expand_route(mut route: ProviderRouteConfig) -> ProviderRouteConfig {
    if let Some(url) = route.base_url {
        route.base_url = Some(expand_env_vars(&url));
    }
    for m in &mut route.models {
        if let Some(url) = m.base_url.take() {
            m.base_url = Some(expand_env_vars(&url));
        }
    }
    route
}

pub fn load_dsh_config(workspace: &Path) -> LoadedDshConfig {
    load_dsh_config_with_global(workspace, resolve_global_config_dir().as_deref())
}

pub fn load_dsh_config_with_global(
    workspace: &Path,
    global_dir: Option<&Path>,
) -> LoadedDshConfig {
    let mut out = LoadedDshConfig {
        global_dir: global_dir.map(Path::to_path_buf),
        project_dir: resolve_project_config_dir(workspace),
        ..Default::default()
    };

    let mut dirs_in_order = Vec::new();
    if let Some(g) = global_dir {
        dirs_in_order.push(g.to_path_buf());
    }
    dirs_in_order.push(out.project_dir.clone());

    for dir in dirs_in_order {
        let settings_path = dir.join("settings.json");
        match read_json_file::<SettingsConfig>(&settings_path) {
            Ok(Some(cfg)) => {
                out.settings.merge_overlay(cfg);
                out.loaded_files.push(settings_path);
            }
            Ok(None) => {}
            Err(err) => out.errors.push(err),
        }

        #[cfg(feature = "router")]
        {
            let models_path = dir.join("models.json");
            match read_json_file::<RouterConfig>(&models_path) {
                Ok(Some(router)) => {
                    for (name, route) in router.providers {
                        out.router.providers.insert(name, expand_route(route));
                    }
                    out.loaded_files.push(models_path);
                }
                Ok(None) => {}
                Err(err) => out.errors.push(err),
            }
        }
    }

    out
}

impl LoadedDshConfig {
    pub fn apply_to_spine(&self, spine: &mut AgentSpineConfig) {
        if let Some(persona) = &self.settings.persona {
            spine.persona = Some(persona.clone());
        }
        if let Some(provider) = &self.settings.default_provider {
            spine.agent.provider = Some(provider.clone());
        }
        if let Some(model) = &self.settings.default_model {
            spine.agent.model = Some(model.clone());
        }
        if let Some(max_steps) = self.settings.max_steps {
            spine.agent.max_steps = match max_steps {
                0 => None,
                n => Some(n),
            };
        }
        if let Some(effort) = self.settings.reasoning_effort {
            spine.agent.reasoning_effort = Some(effort);
        }
        if let Some(tools) = &self.settings.tools {
            spine.include_tools = Some(tools.clone());
        }
        for pat in &self.settings.exclude_tools {
            if !spine.exclude_tools.contains(pat) {
                spine.exclude_tools.push(pat.clone());
            }
        }
        if let Some(db_path) = &self.settings.session_db {
            spine.session_db_path = Some(db_path.clone());
        }

        #[cfg(feature = "coding")]
        if let AgentToolsMode::Coding {
            coding,
            tool_search,
            codemode,
            mcp,
            skills,
        } = &mut spine.tools_mode
        {
            mcp.global_config_dir = self.global_dir.clone();
            if let Some(sk) = &self.settings.skills {
                if let Some(v) = sk.enabled {
                    skills.enabled = v;
                }
                for dir in &sk.custom_dirs {
                    if !skills.custom_skill_dirs.contains(dir) {
                        skills.custom_skill_dirs.push(dir.clone());
                    }
                }
                if sk.bundled_dir.is_some() {
                    skills.bundled_skill_dir = sk.bundled_dir.clone();
                }
                if let Some(max_len) = sk.catalog_description_max_length {
                    skills.catalog_description_max_length = max_len;
                }
            }
            if let Some(preset_str) = &self.settings.tools_preset {
                coding.preset = match preset_str.trim().to_ascii_lowercase().as_str() {
                    "minimal" | "minimal-coding" => CodingToolsPreset::MinimalCoding,
                    "read-only" | "readonly" => CodingToolsPreset::ReadOnly,
                    _ => CodingToolsPreset::All,
                };
            }
            if let Some(restrict) = self.settings.restrict_to_workspace {
                coding.restrict_to_workspace = restrict;
            }
            if let Some(cm) = &self.settings.codemode {
                if let Some(v) = cm.auto_enable {
                    codemode.auto_enable = v;
                }
                if let Some(v) = cm.only_mode {
                    codemode.only_mode = v;
                }
                if let Some(v) = cm.inline_budget {
                    codemode.inline_budget = v;
                }
                if let Some(v) = cm.timeout_ms {
                    codemode.timeout_ms = v;
                }
            }
            if let Some(ts) = &self.settings.tool_search {
                if let Some(v) = ts.auto_enable {
                    tool_search.auto_enable = v;
                }
                if let Some(v) = ts.default_limit {
                    tool_search.default_limit = v;
                }
            }
        }
    }

    /// 结合环境变量覆盖与 `models.json` / `settings.json` 解析最终模型路由。
    #[cfg(feature = "router")]
    pub fn resolve_llm_router(&self) -> Option<ResolvedLlmRouter> {
        self.resolve_llm_router_with_env(EnvLlmOverride::from_env())
    }

    #[cfg(feature = "router")]
    pub fn resolve_llm_router_with_env(&self, env: EnvLlmOverride) -> Option<ResolvedLlmRouter> {
        let mut router = self.router.clone();
        let fallback_key = env.explicit_api_key.clone().or(env.provider_api_key);

        if env.explicit_api_key.is_some() || (router.providers.is_empty() && fallback_key.is_some())
        {
            let protocol = env
                .protocol
                .as_deref()
                .unwrap_or("completions")
                .to_ascii_lowercase();
            let mut route = match protocol.as_str() {
                "responses" => ProviderRouteConfig::openai_responses("API_KEY"),
                "messages" | "anthropic" => ProviderRouteConfig::anthropic_messages("API_KEY"),
                _ => ProviderRouteConfig::openai_completions("API_KEY"),
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
                let target_model = env.model.as_deref().or(self.settings.default_model.as_deref())?;
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

#[derive(Debug, Clone, Default)]
pub struct EnvLlmOverride {
    pub explicit_api_key: Option<String>,
    pub provider_api_key: Option<String>,
    pub protocol: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty())
}

impl EnvLlmOverride {
    pub fn from_env() -> Self {
        Self {
            explicit_api_key: env_non_empty("API_KEY"),
            provider_api_key: env_non_empty("OPENAI_API_KEY")
                .or_else(|| env_non_empty("ANTHROPIC_API_KEY"))
                .or_else(|| env_non_empty("DEEPSEEK_API_KEY")),
            protocol: env_non_empty("PROTOCOL"),
            base_url: env_non_empty("BASE_URL"),
            model: env_non_empty("MODEL"),
        }
    }
}

#[cfg(feature = "router")]
#[derive(Debug, Clone)]
pub struct ResolvedLlmRouter {
    pub router: RouterConfig,
    pub active_provider: String,
    pub active_model: Option<String>,
}

#[cfg(all(test, feature = "router", feature = "coding"))]
mod tests {
    use super::*;

    #[test]
    fn loads_and_merges_settings_and_models_from_dsh_rs_dirs() {
        let tmp = std::env::temp_dir().join(format!("dsh_rs_cfg_test_{}", std::process::id()));
        let global = tmp.join(".dsh-rs");
        let proj = tmp.join("repo");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".dsh-rs")).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "defaultProvider": "openai",
                "defaultModel": "gpt-4o",
                "maxSteps": 40,
                "excludeTools": ["mcp__fff__multi_grep"]
            }"#,
        )
        .unwrap();
        std::fs::write(
            global.join("models.json"),
            r#"{
                "providers": {
                    "openai": {
                        "protocol": "openai-completions",
                        "baseUrl": "https://api.openai.com/v1",
                        "apiKeyEnv": "OPENAI_API_KEY",
                        "model": "gpt-4o"
                    }
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj.join(".dsh-rs/settings.json"),
            r#"{
                "toolsPreset": "minimal",
                "reasoningEffort": "medium"
            }"#,
        )
        .unwrap();

        let loaded = load_dsh_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(loaded.settings.default_provider.as_deref(), Some("openai"));
        assert_eq!(loaded.settings.default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(loaded.settings.max_steps, Some(40));
        assert_eq!(loaded.settings.reasoning_effort, Some(ReasoningEffort::Medium));
        assert!(loaded.router.providers.contains_key("openai"));

        let resolved = loaded
            .resolve_llm_router_with_env(EnvLlmOverride::default())
            .expect("models.json should resolve");
        assert_eq!(resolved.active_provider, "openai");
        assert_eq!(resolved.active_model.as_deref(), Some("gpt-4o"));

        let env_overridden = loaded
            .resolve_llm_router_with_env(EnvLlmOverride {
                explicit_api_key: Some("sk-override".to_string()),
                protocol: Some("anthropic".to_string()),
                model: Some("claude-sonnet-4-5".to_string()),
                ..Default::default()
            })
            .expect("API_KEY override should resolve");
        assert_eq!(env_overridden.active_provider, "default");
        assert_eq!(
            env_overridden.active_model.as_deref(),
            Some("claude-sonnet-4-5")
        );
        assert!(env_overridden.router.providers.contains_key("openai"));
        assert!(env_overridden.router.providers.contains_key("default"));

        let empty_cfg = LoadedDshConfig::default();
        assert!(empty_cfg
            .resolve_llm_router_with_env(EnvLlmOverride::default())
            .is_none());

        let mut spine = AgentSpineConfig::coding(&proj);
        loaded.apply_to_spine(&mut spine);
        assert_eq!(spine.agent.provider.as_deref(), Some("openai"));
        assert_eq!(spine.agent.model.as_deref(), Some("gpt-4o"));
        assert_eq!(spine.agent.max_steps, Some(40));
        assert_eq!(spine.exclude_tools, vec!["mcp__fff__multi_grep".to_string()]);
        match &spine.tools_mode {
            AgentToolsMode::Coding { coding, mcp, .. } => {
                assert_eq!(coding.preset, CodingToolsPreset::MinimalCoding);
                assert_eq!(mcp.global_config_dir.as_deref(), Some(global.as_path()));
            }
            _ => panic!("expected coding mode"),
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn loads_pi_style_models_and_settings_json() {
        let tmp = std::env::temp_dir().join(format!("dsh_rs_pi_cfg_{}", std::process::id()));
        let global = tmp.join(".dsh-rs");
        let proj = tmp.join("repo");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(&proj).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "defaultThinkingLevel": "max",
                "defaultModel": "mayoru-messages/claude-sonnet-5-5"
            }"#,
        )
        .unwrap();
        std::fs::write(
            global.join("models.json"),
            r#"{
                "$schema": "https://pi.dev/schemas/models.schema.json",
                "providers": {
                    "alpha": {
                        "api": "openai-completions",
                        "apiKey": "k-alpha",
                        "baseUrl": "http://127.0.0.1:4000/v1",
                        "models": [{ "id": "other-model" }]
                    },
                    "magpie": {
                        "name": "magpie",
                        "api": "openai-completions",
                        "apiKey": "$MY_SECRET",
                        "baseUrl": "http://127.0.0.1:3425/v1",
                        "compat": { "sendSessionAffinityHeaders": true },
                        "models": [
                            { "id": "mayoru-completions/gemini-4-hydrogen" },
                            {
                                "id": "mayoru-messages/claude-sonnet-5-5",
                                "api": "anthropic-messages",
                                "baseUrl": "http://127.0.0.1:3425",
                                "compat": { "forceAdaptiveThinking": true }
                            }
                        ]
                    }
                }
            }"#,
        )
        .unwrap();

        let loaded = load_dsh_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(loaded.settings.reasoning_effort, Some(ReasoningEffort::Max));

        // apiKey 中的 "$MY_SECRET" 应保留到请求期再解析
        let magpie = loaded.router.providers.get("magpie").unwrap();
        assert_eq!(magpie.api_key.as_deref(), Some("$MY_SECRET"));
        assert_eq!(magpie.models.len(), 2);

        // 未写 defaultProvider 时，根据 defaultModel 匹配对应 provider "magpie"
        let resolved = loaded
            .resolve_llm_router_with_env(EnvLlmOverride::default())
            .expect("should select magpie by defaultModel");
        assert_eq!(resolved.active_provider, "magpie");
        assert_eq!(
            resolved.active_model.as_deref(),
            Some("mayoru-messages/claude-sonnet-5-5")
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
