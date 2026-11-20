//! 全局与项目级配置加载：统一存放在 `~/.mizar/` 与 `<workspace>/.mizar/`。
//!
//! 支持的配置文件：
//! - `settings.json`：默认模型、推理强度、工具预设/过滤、codemode 与 tool_search 选项
//! - `models.json`：多提供方路由（复用 `mizar_llm_router::RouterConfig` 格式）
//! - `mcp.json`：MCP Server 配置（由 `mizar-mcp` 从同一目录加载）

use std::path::{Path, PathBuf};

#[cfg(feature = "router")]
use mizar_llm::LlmError;
use mizar_llm::ReasoningEffort;
#[cfg(feature = "router")]
use mizar_llm_router::{ProtocolKind, ProviderRouteConfig, RouterConfig, parse_protocol_api};
#[cfg(feature = "coding")]
use mizar_tools_coding::CodingToolsPreset;
use serde::{Deserialize, Serialize};
#[cfg(feature = "router")]
use thiserror::Error;

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
pub struct InstructionsSettings {
    pub enabled: Option<bool>,
    pub candidates: Option<Vec<String>>,
    pub local_candidates: Option<Vec<String>>,
    pub project_root_markers: Option<Vec<String>>,
    pub max_bytes: Option<usize>,
    pub max_source_bytes: Option<usize>,
    pub follow_subdirs: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct CompactionSettings {
    pub enabled: Option<bool>,
    pub auto_compact: Option<bool>,
    pub auto_prune: Option<bool>,
    pub soft_window_cap: Option<u64>,
    pub trigger_ratio: Option<f64>,
    pub prune_threshold_chars: Option<usize>,
    pub cold_cache_idle_secs: Option<u64>,
    pub reuse_prefix_cache: Option<bool>,
    pub disable_summary_reasoning: Option<bool>,
    pub enable_read_dedup: Option<bool>,
    pub enable_write_after_compact_guard: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct RecallSettings {
    pub enabled: Option<bool>,
    pub auto_activate_on_compaction: Option<bool>,
    pub default_search_limit: Option<usize>,
    pub read_char_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct SessionTitleSettings {
    pub enabled: Option<bool>,
    pub model: Option<String>,
    pub thinking_level: Option<String>,
}

/// 将 `sessionTitle.thinkingLevel` 字符串规范化为 `ReasoningEffort`（缺省或无法识别时回退为 `Low`）。
pub fn normalize_session_title_thinking_level(raw: Option<&str>) -> ReasoningEffort {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return ReasoningEffort::Low;
    };
    match s.to_ascii_lowercase().as_str() {
        "off" | "none" | "0" => ReasoningEffort::Off,
        "minimal" | "min" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" | "med" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" | "x-high" => ReasoningEffort::XHigh,
        "max" => ReasoningEffort::Max,
        _ => ReasoningEffort::Low,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct UiSettings {
    pub theme: Option<String>,
    pub screen_mode: Option<String>,
    pub show_thinking: Option<bool>,
    pub verbose_expanded: Option<bool>,
    pub side_panel_open: Option<bool>,
    pub mouse_capture: Option<bool>,
    pub approval_mode: Option<String>,
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
    /// 声明式屏蔽的内置可选 Cordis 模块 ID 列表（受保护核心模块不可禁用）。
    #[serde(default, rename = "disabledPlugins")]
    pub disabled_plugins: Vec<String>,
    pub restrict_to_workspace: Option<bool>,
    pub session_db: Option<PathBuf>,
    pub codemode: Option<CodemodeSettings>,
    pub tool_search: Option<ToolSearchSettings>,
    pub skills: Option<SkillsSettings>,
    pub instructions: Option<InstructionsSettings>,
    pub compaction: Option<CompactionSettings>,
    pub recall: Option<RecallSettings>,
    pub session_title: Option<SessionTitleSettings>,
    pub ui: Option<UiSettings>,
}

/// 规范化插件 ID 列表：裁剪空白、过滤空串、排序并去重。
pub fn normalize_disabled_plugins(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = ids
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    out.sort();
    out.dedup();
    out
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
        self.disabled_plugins.extend(other.disabled_plugins);
        self.disabled_plugins =
            normalize_disabled_plugins(std::mem::take(&mut self.disabled_plugins));
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
        if let Some(inst) = other.instructions {
            let dst = self.instructions.get_or_insert_with(Default::default);
            if inst.enabled.is_some() {
                dst.enabled = inst.enabled;
            }
            if inst.candidates.is_some() {
                dst.candidates = inst.candidates;
            }
            if inst.local_candidates.is_some() {
                dst.local_candidates = inst.local_candidates;
            }
            if inst.project_root_markers.is_some() {
                dst.project_root_markers = inst.project_root_markers;
            }
            if inst.max_bytes.is_some() {
                dst.max_bytes = inst.max_bytes;
            }
            if inst.max_source_bytes.is_some() {
                dst.max_source_bytes = inst.max_source_bytes;
            }
            if inst.follow_subdirs.is_some() {
                dst.follow_subdirs = inst.follow_subdirs;
            }
        }
        if let Some(comp) = other.compaction {
            let dst = self.compaction.get_or_insert_with(Default::default);
            if comp.enabled.is_some() {
                dst.enabled = comp.enabled;
            }
            if comp.auto_compact.is_some() {
                dst.auto_compact = comp.auto_compact;
            }
            if comp.auto_prune.is_some() {
                dst.auto_prune = comp.auto_prune;
            }
            if comp.soft_window_cap.is_some() {
                dst.soft_window_cap = comp.soft_window_cap;
            }
            if comp.trigger_ratio.is_some() {
                dst.trigger_ratio = comp.trigger_ratio;
            }
            if comp.prune_threshold_chars.is_some() {
                dst.prune_threshold_chars = comp.prune_threshold_chars;
            }
            if comp.cold_cache_idle_secs.is_some() {
                dst.cold_cache_idle_secs = comp.cold_cache_idle_secs;
            }
            if comp.reuse_prefix_cache.is_some() {
                dst.reuse_prefix_cache = comp.reuse_prefix_cache;
            }
            if comp.disable_summary_reasoning.is_some() {
                dst.disable_summary_reasoning = comp.disable_summary_reasoning;
            }
            if comp.enable_read_dedup.is_some() {
                dst.enable_read_dedup = comp.enable_read_dedup;
            }
            if comp.enable_write_after_compact_guard.is_some() {
                dst.enable_write_after_compact_guard = comp.enable_write_after_compact_guard;
            }
        }
        if let Some(rec) = other.recall {
            let dst = self.recall.get_or_insert_with(Default::default);
            if rec.enabled.is_some() {
                dst.enabled = rec.enabled;
            }
            if rec.auto_activate_on_compaction.is_some() {
                dst.auto_activate_on_compaction = rec.auto_activate_on_compaction;
            }
            if rec.default_search_limit.is_some() {
                dst.default_search_limit = rec.default_search_limit;
            }
            if rec.read_char_limit.is_some() {
                dst.read_char_limit = rec.read_char_limit;
            }
        }
        if let Some(st) = other.session_title {
            let dst = self.session_title.get_or_insert_with(Default::default);
            if st.enabled.is_some() {
                dst.enabled = st.enabled;
            }
            if st.model.is_some() {
                dst.model = st.model;
            }
            if st.thinking_level.is_some() {
                dst.thinking_level = st.thinking_level;
            }
        }
        if let Some(ui) = other.ui {
            let dst = self.ui.get_or_insert_with(Default::default);
            if ui.theme.is_some() {
                dst.theme = ui.theme;
            }
            if ui.screen_mode.is_some() {
                dst.screen_mode = ui.screen_mode;
            }
            if ui.show_thinking.is_some() {
                dst.show_thinking = ui.show_thinking;
            }
            if ui.verbose_expanded.is_some() {
                dst.verbose_expanded = ui.verbose_expanded;
            }
            if ui.side_panel_open.is_some() {
                dst.side_panel_open = ui.side_panel_open;
            }
            if ui.mouse_capture.is_some() {
                dst.mouse_capture = ui.mouse_capture;
            }
            if ui.approval_mode.is_some() {
                dst.approval_mode = ui.approval_mode;
            }
        }
    }
}

/// 解析全局 `settings.json` 路径（默认 `~/.mizar/settings.json`，受 `MIZAR_HOME` 覆盖）。
pub fn resolve_global_settings_path() -> Option<PathBuf> {
    resolve_global_config_dir().map(|dir| dir.join("settings.json"))
}

/// 原子化增量修补 `settings.json`：仅更新调用方修改的键，完好保留其他手写字段。
pub fn patch_settings_file(
    path: &Path,
    patch: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Result<(), String> {
    let mut root_map = if path.exists() {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let no_bom = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
        if no_bom.trim().is_empty() {
            serde_json::Map::new()
        } else {
            let val: serde_json::Value =
                serde_json::from_str(no_bom).map_err(|e| format!("{}: {e}", path.display()))?;
            match val {
                serde_json::Value::Object(m) => m,
                _ => {
                    return Err(format!(
                        "{}: settings.json 顶层必须为 JSON 对象",
                        path.display()
                    ));
                }
            }
        }
    } else {
        serde_json::Map::new()
    };

    patch(&mut root_map);

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut rendered = serde_json::to_string_pretty(&serde_json::Value::Object(root_map))
        .map_err(|e| e.to_string())?;
    rendered.push('\n');

    let tmp_path = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp_path, rendered.as_bytes())
        .map_err(|e| format!("{}: {e}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!("{}: {e}", path.display())
    })?;
    Ok(())
}

/// 原子回写 `settings.json` 中的 `"disabledPlugins"` 字段（自动排序去重并保留其余手写字段）。
pub fn persist_disabled_plugins(path: &Path, disabled: &[String]) -> Result<(), String> {
    let normalized = normalize_disabled_plugins(disabled.iter().cloned());
    patch_settings_file(path, |root| {
        root.insert("disabledPlugins".to_string(), serde_json::json!(normalized));
    })
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
                if let Some(v) = comp.auto_prune {
                    dst.auto_prune = v;
                }
                if let Some(v) = comp.soft_window_cap {
                    dst.soft_window_cap = v;
                }
                if let Some(v) = comp.trigger_ratio {
                    dst.trigger_ratio = v;
                }
                if let Some(v) = comp.prune_threshold_chars {
                    dst.prune_threshold_chars = v;
                }
                if let Some(v) = comp.cold_cache_idle_secs {
                    dst.cold_cache_idle_secs = v;
                }
                if let Some(v) = comp.reuse_prefix_cache {
                    dst.reuse_prefix_cache = v;
                }
                if let Some(v) = comp.disable_summary_reasoning {
                    dst.disable_summary_reasoning = v;
                }
                if let Some(v) = comp.enable_read_dedup {
                    dst.enable_read_dedup = v;
                }
                if let Some(v) = comp.enable_write_after_compact_guard {
                    dst.enable_write_after_compact_guard = v;
                }
            }
            if let Some(rec) = &self.settings.recall
                && let Some(dst) = spine.recall.as_mut()
            {
                if let Some(v) = rec.enabled {
                    dst.enabled = v;
                }
                if let Some(v) = rec.auto_activate_on_compaction {
                    dst.auto_activate_on_compaction = v;
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
                if let Some(v) = inst.follow_subdirs {
                    instructions.follow_subdirs = v;
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

    /// 结合环境变量覆盖与 `models.json` / `settings.json` 解析最终模型路由（TUI 兼容入口）。
    #[cfg(feature = "router")]
    pub fn resolve_llm_router(&self) -> Option<ResolvedLlmRouter> {
        self.resolve_llm_router_with_env(EnvLlmOverride::from_env())
    }

    /// 按严格优先序（`CLI > Env > LoadedConfig`）解析并校验 LLM 路由，用于无头 `exec` 模式。
    #[cfg(feature = "router")]
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

    #[cfg(feature = "router")]
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

#[cfg(all(test, feature = "router", feature = "coding"))]
mod tests {
    use super::*;

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
                assert!(!instructions.follow_subdirs);
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

    #[test]
    fn patch_settings_file_preserves_unrelated_keys_and_merges_ui() {
        let tmp =
            std::env::temp_dir().join(format!("mizar_patch_settings_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let file = tmp.join("settings.json");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(
            &file,
            r#"{
                "persona": "keep me",
                "excludeTools": ["mcp__demo__drop"]
            }"#,
        )
        .unwrap();

        patch_settings_file(&file, |root| {
            root.insert(
                "ui".into(),
                serde_json::json!({ "theme": "sakura-dark", "showThinking": false }),
            );
        })
        .unwrap();

        let loaded = load_mizar_config_with_global(&tmp.join("proj"), Some(&tmp));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(loaded.settings.persona.as_deref(), Some("keep me"));
        assert_eq!(
            loaded.settings.exclude_tools,
            vec!["mcp__demo__drop".to_string()]
        );
        let ui = loaded.settings.ui.as_ref().unwrap();
        assert_eq!(ui.theme.as_deref(), Some("sakura-dark"));
        assert_eq!(ui.show_thinking, Some(false));

        let _ = std::fs::remove_dir_all(&tmp);
    }

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

#[cfg(test)]
mod disabled_plugins_tests {
    use super::*;

    #[test]
    fn disabled_plugins_serializes_merges_dedups_sorts_and_patches_atomically() {
        let tmp =
            std::env::temp_dir().join(format!("mizar_disabled_plugins_cfg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let global = tmp.join("global");
        let proj = tmp.join("repo");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".mizar")).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "persona": "preserve-me",
                "disabledPlugins": ["skills", " mcp ", "", "skills", "codemode"]
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj.join(".mizar/settings.json"),
            r#"{
                "disabledPlugins": ["recall", "mcp", "compaction"]
            }"#,
        )
        .unwrap();

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(
            loaded.settings.disabled_plugins,
            vec![
                "codemode".to_string(),
                "compaction".to_string(),
                "mcp".to_string(),
                "recall".to_string(),
                "skills".to_string(),
            ]
        );

        let mut spine = AgentSpineConfig::new(&proj);
        spine.disabled_plugins = vec!["trajectory".to_string(), "mcp".to_string()];
        loaded.apply_to_spine(&mut spine);
        assert_eq!(
            spine.disabled_plugins,
            vec![
                "codemode".to_string(),
                "compaction".to_string(),
                "mcp".to_string(),
                "recall".to_string(),
                "skills".to_string(),
                "trajectory".to_string(),
            ]
        );

        // 原子回写应保留原有字段，同时排序去重写入 disabledPlugins
        let global_settings = global.join("settings.json");
        persist_disabled_plugins(
            &global_settings,
            &[
                "trajectory".to_string(),
                "mcp".to_string(),
                "trajectory".to_string(),
            ],
        )
        .unwrap();

        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&global_settings).unwrap()).unwrap();
        assert_eq!(raw["persona"], "preserve-me");
        assert_eq!(
            raw["disabledPlugins"],
            serde_json::json!(["mcp", "trajectory"])
        );

        // 空列表回写正常序列化为 []
        persist_disabled_plugins(&global_settings, &[]).unwrap();
        let raw_empty: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&global_settings).unwrap()).unwrap();
        assert_eq!(raw_empty["persona"], "preserve-me");
        assert_eq!(raw_empty["disabledPlugins"], serde_json::json!([]));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn loads_and_normalizes_session_title_settings() {
        assert_eq!(
            normalize_session_title_thinking_level(None),
            ReasoningEffort::Low
        );
        assert_eq!(
            normalize_session_title_thinking_level(Some("unknown")),
            ReasoningEffort::Low
        );
        assert_eq!(
            normalize_session_title_thinking_level(Some("OFF")),
            ReasoningEffort::Off
        );
        assert_eq!(
            normalize_session_title_thinking_level(Some("Medium")),
            ReasoningEffort::Medium
        );

        let tmp =
            std::env::temp_dir().join(format!("mizar_session_title_cfg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let global = tmp.join("global");
        let proj = tmp.join("repo");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".mizar")).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "sessionTitle": {
                    "enabled": false,
                    "model": "magpie/mayoru-responses/deepseek-flash"
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj.join(".mizar/settings.json"),
            r#"{
                "sessionTitle": {
                    "enabled": true,
                    "thinkingLevel": "high"
                }
            }"#,
        )
        .unwrap();

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        let st = loaded.settings.session_title.as_ref().unwrap();
        assert_eq!(st.enabled, Some(true));
        assert_eq!(
            st.model.as_deref(),
            Some("magpie/mayoru-responses/deepseek-flash")
        );
        assert_eq!(st.thinking_level.as_deref(), Some("high"));

        #[cfg(feature = "session-title")]
        {
            let mut spine = AgentSpineConfig::new(&proj);
            loaded.apply_to_spine(&mut spine);
            let cfg = spine.session_title.unwrap();
            assert!(cfg.enabled);
            assert_eq!(cfg.provider.as_deref(), Some("magpie"));
            assert_eq!(
                cfg.model.as_deref(),
                Some("mayoru-responses/deepseek-flash")
            );
            assert_eq!(cfg.reasoning_effort, Some(ReasoningEffort::High));
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(all(test, feature = "router", feature = "coding"))]
mod exec_route_tests {
    use super::*;

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
