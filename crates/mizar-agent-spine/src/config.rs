//! 全局与项目级配置加载：统一存放在 `~/.mizar/` 与 `<workspace>/.mizar/`。
//!
//! 支持的配置文件：
//! - `settings.json`：默认模型、推理强度、工具预设/过滤、codemode 与 tool_search 选项
//! - `models.json`：多提供方路由（复用 `mizar_llm_router::RouterConfig` 格式）
//! - `mcp.json`：MCP Server 配置（由 `mizar-mcp` 从同一目录加载）

pub mod route;
pub mod settings;

pub use route::*;
pub use settings::*;

use std::path::{Path, PathBuf};

#[cfg(feature = "router")]
use mizar_llm_router::{ProviderRouteConfig, RouterConfig};
#[cfg(feature = "coding")]
use mizar_tools_coding::CodingToolsPreset;
use serde::Deserialize;

use crate::AgentSpineConfig;
#[cfg(feature = "coding")]
use crate::AgentToolsMode;

/// 解析 `${VAR}` 环境变量占位符（与 `mizar-mcp` 配置解析保持一致，避免跨 crate 强耦合）。
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

pub const CONFIG_DIR_NAME: &str = ".mizar";

/// 全局配置目录（支持 `MIZAR_HOME` 环境变量覆盖，默认 `~/.mizar`）。
pub fn resolve_global_config_dir() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("MIZAR_HOME") {
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

/// 解析默认持久化 SQLite 数据库路径：支持 `MIZAR_SESSION_DB` 环境变量覆盖
/// （设为 `"none"` / `"off"` / `"memory"` 可禁用自动落盘），默认 `~/.mizar/sessions.sqlite`。
pub fn resolve_default_session_db_path() -> Option<PathBuf> {
    if let Ok(raw) = std::env::var("MIZAR_SESSION_DB") {
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

#[derive(Debug, Clone, Default)]
pub struct LoadedMizarConfig {
    pub global_dir: Option<PathBuf>,
    pub project_dir: PathBuf,
    /// 是否处于 `--isolated` 隔离模式（跳过个人全局配置、个人 skills、全局指令与全局 MCP）。
    pub isolated: bool,
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

pub fn load_mizar_config(workspace: &Path) -> LoadedMizarConfig {
    load_mizar_config_with_global(workspace, resolve_global_config_dir().as_deref())
}

/// 以隔离模式加载配置：完全跳过个人全局目录（`~/.mizar` / `$MIZAR_HOME`），仅加载工作区 `<workspace>/.mizar/`。
pub fn load_mizar_config_isolated(workspace: &Path) -> LoadedMizarConfig {
    let mut cfg = load_mizar_config_with_global(workspace, None);
    cfg.isolated = true;
    cfg
}

pub fn load_mizar_config_with_global(
    workspace: &Path,
    global_dir: Option<&Path>,
) -> LoadedMizarConfig {
    let mut out = LoadedMizarConfig {
        global_dir: global_dir.map(Path::to_path_buf),
        project_dir: resolve_project_config_dir(workspace),
        isolated: false,
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

impl LoadedMizarConfig {
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
        if !self.settings.disabled_plugins.is_empty() {
            let mut merged = std::mem::take(&mut spine.disabled_plugins);
            merged.extend(self.settings.disabled_plugins.iter().cloned());
            spine.disabled_plugins = normalize_disabled_plugins(merged);
        }
        if let Some(db_path) = &self.settings.session_db {
            spine.session_db_path = Some(db_path.clone());
        }

        #[cfg(feature = "compaction")]
        {
            if let Some(comp) = &self.settings.compaction
                && let Some(dst) = spine.compaction.as_mut()
            {
                if let Some(v) = comp.enabled {
                    dst.enabled = v;
                }
                if let Some(v) = comp.auto_compact {
                    dst.auto_compact = v;
                }
                if let Some(v) = comp.max_context_tokens {
                    dst.max_context_tokens = Some(v);
                }
                if let Some(v) = comp.trigger_ratio {
                    dst.trigger_ratio = v;
                }
                if let Some(v) = comp.keep_recent_tokens {
                    dst.keep_recent_tokens = v;
                }
                if let Some(v) = comp.supersede_reads {
                    dst.supersede_reads = v;
                }
                if let Some(v) = comp.drop_useless {
                    dst.drop_useless = v;
                }
            }
            if let Some(rec) = &self.settings.recall
                && let Some(dst) = spine.recall.as_mut()
            {
                if let Some(v) = rec.enabled {
                    dst.enabled = v;
                }
                if let Some(v) = rec.default_search_limit {
                    dst.default_search_limit = v;
                }
                if let Some(v) = rec.read_char_limit {
                    dst.read_char_limit = v;
                }
            }
        }

        #[cfg(feature = "session-title")]
        if let Some(st) = &self.settings.session_title
            && let Some(dst) = spine.session_title.as_mut()
        {
            if let Some(v) = st.enabled {
                dst.enabled = v;
            }
            if let Some(model_raw) = &st.model {
                if let Some((provider, model)) = mizar_session_title::parse_model_ref(model_raw) {
                    dst.provider = provider;
                    dst.model = Some(model);
                } else {
                    dst.provider = None;
                    dst.model = None;
                }
            }
            dst.reasoning_effort = Some(normalize_session_title_thinking_level(
                st.thinking_level.as_deref(),
            ));
        }

        #[cfg(feature = "coding")]
        if let AgentToolsMode::Coding {
            coding,
            tool_search,
            codemode,
            mcp,
            skills,
            instructions,
        } = &mut spine.tools_mode
        {
            mcp.global_config_dir = self.global_dir.clone();
            mcp.skip_global_config = self.isolated;
            skills.include_user_roots = !self.isolated;
            instructions.mizar_home = self.global_dir.clone();
            instructions.include_default_global = !self.isolated;
            if let Some(inst) = &self.settings.instructions {
                if let Some(v) = inst.enabled {
                    instructions.enabled = v;
                }
                if let Some(cands) = &inst.candidates {
                    instructions.candidates = cands.clone();
                }
                if let Some(local) = &inst.local_candidates {
                    instructions.local_candidates = local.clone();
                }
                if let Some(markers) = &inst.project_root_markers {
                    instructions.project_root_markers = markers.clone();
                }
                if let Some(v) = inst.max_bytes {
                    instructions.max_bytes = v;
                }
                if let Some(v) = inst.max_source_bytes {
                    instructions.max_source_bytes = v;
                }
            }
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
}

#[cfg(all(test, feature = "router", feature = "coding"))]
mod tests {
    use super::*;
    use mizar_llm::ReasoningEffort;

    #[test]
    fn loads_and_merges_settings_and_models_from_mizar_dirs() {
        let tmp = std::env::temp_dir().join(format!("mizar_cfg_test_{}", std::process::id()));
        let global = tmp.join(".mizar");
        let proj = tmp.join("repo");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".mizar")).unwrap();

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
            proj.join(".mizar/settings.json"),
            r#"{
                "toolsPreset": "minimal",
                "reasoningEffort": "medium",
                "instructions": {
                    "maxBytes": 32768,
                    "followSubdirs": false
                }
            }"#,
        )
        .unwrap();

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(loaded.settings.default_provider.as_deref(), Some("openai"));
        assert_eq!(loaded.settings.default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(loaded.settings.max_steps, Some(40));
        assert_eq!(
            loaded.settings.reasoning_effort,
            Some(ReasoningEffort::Medium)
        );
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

        let empty_cfg = LoadedMizarConfig::default();
        assert!(
            empty_cfg
                .resolve_llm_router_with_env(EnvLlmOverride::default())
                .is_none()
        );

        let mut spine = AgentSpineConfig::coding(&proj);
        loaded.apply_to_spine(&mut spine);
        assert_eq!(spine.agent.provider.as_deref(), Some("openai"));
        assert_eq!(spine.agent.model.as_deref(), Some("gpt-4o"));
        assert_eq!(spine.agent.max_steps, Some(40));
        assert_eq!(
            spine.exclude_tools,
            vec!["mcp__fff__multi_grep".to_string()]
        );
        match &spine.tools_mode {
            AgentToolsMode::Coding {
                coding,
                mcp,
                instructions,
                ..
            } => {
                assert_eq!(coding.preset, CodingToolsPreset::MinimalCoding);
                assert_eq!(mcp.global_config_dir.as_deref(), Some(global.as_path()));
                assert_eq!(instructions.mizar_home.as_deref(), Some(global.as_path()));
                assert_eq!(instructions.max_bytes, 32768);
            }
            _ => panic!("expected coding mode"),
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn loads_pi_style_models_and_settings_json() {
        let tmp = std::env::temp_dir().join(format!("mizar_pi_cfg_{}", std::process::id()));
        let global = tmp.join(".mizar");
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

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
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
