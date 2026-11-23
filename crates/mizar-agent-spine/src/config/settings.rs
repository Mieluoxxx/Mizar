//! 全局与项目级 `settings.json` 配置模型、规范化辅助函数与原子增量写回。

use std::path::{Path, PathBuf};

use mizar_llm::{CacheWarmingMode, ReasoningEffort};
use serde::{Deserialize, Serialize};

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
    pub default_context_window: Option<u64>,
    pub max_context_tokens: Option<u64>,
    pub trigger_ratio: Option<f64>,
    pub keep_recent_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct RecallSettings {
    pub enabled: Option<bool>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct SubagentSettings {
    pub max_steps: Option<usize>,
    pub max_steps_cap: Option<usize>,
    pub timeout_secs: Option<u64>,
    pub max_concurrent: Option<usize>,
    pub stall_timeout_secs: Option<u64>,
    pub max_watchdog_alerts: Option<usize>,
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
    pub subagent: Option<SubagentSettings>,
    pub cache_warming: Option<CacheWarmingMode>,
    pub show_cache_miss_notices: Option<bool>,
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
    pub(super) fn merge_overlay(&mut self, other: SettingsConfig) {
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
            if comp.default_context_window.is_some() {
                dst.default_context_window = comp.default_context_window;
            }
            if comp.max_context_tokens.is_some() {
                dst.max_context_tokens = comp.max_context_tokens;
            }
            if comp.trigger_ratio.is_some() {
                dst.trigger_ratio = comp.trigger_ratio;
            }
            if comp.keep_recent_tokens.is_some() {
                dst.keep_recent_tokens = comp.keep_recent_tokens;
            }
        }
        if let Some(rec) = other.recall {
            let dst = self.recall.get_or_insert_with(Default::default);
            if rec.enabled.is_some() {
                dst.enabled = rec.enabled;
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
        if let Some(sub) = other.subagent {
            let dst = self.subagent.get_or_insert_with(Default::default);
            if sub.max_steps.is_some() {
                dst.max_steps = sub.max_steps;
            }
            if sub.max_steps_cap.is_some() {
                dst.max_steps_cap = sub.max_steps_cap;
            }
            if sub.timeout_secs.is_some() {
                dst.timeout_secs = sub.timeout_secs;
            }
            if sub.max_concurrent.is_some() {
                dst.max_concurrent = sub.max_concurrent;
            }
            if sub.stall_timeout_secs.is_some() {
                dst.stall_timeout_secs = sub.stall_timeout_secs;
            }
            if sub.max_watchdog_alerts.is_some() {
                dst.max_watchdog_alerts = sub.max_watchdog_alerts;
            }
        }
        if other.cache_warming.is_some() {
            self.cache_warming = other.cache_warming;
        }
        if other.show_cache_miss_notices.is_some() {
            self.show_cache_miss_notices = other.show_cache_miss_notices;
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
    crate::config::resolve_global_config_dir().map(|dir| dir.join("settings.json"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_mizar_config_with_global;

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
}

#[cfg(test)]
mod disabled_plugins_tests {
    use super::*;
    use crate::AgentSpineConfig;
    use crate::config::load_mizar_config_with_global;
    #[cfg(feature = "session-title")]
    use mizar_llm::{CacheWarmingMode, ReasoningEffort};

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
            assert!(spine.session_title.is_none());
            spine.session_title = Some(crate::SessionTitleConfig::default());
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

    #[test]
    fn loads_and_overlays_cache_warming_settings() {
        let tmp =
            std::env::temp_dir().join(format!("mizar_cache_warming_cfg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let global = tmp.join("global");
        let proj = tmp.join("repo");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".mizar")).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "cacheWarming": "idle",
                "showCacheMissNotices": false
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj.join(".mizar/settings.json"),
            r#"{
                "cacheWarming": "streaming",
                "showCacheMissNotices": true
            }"#,
        )
        .unwrap();

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        assert_eq!(
            loaded.settings.cache_warming,
            Some(CacheWarmingMode::Streaming)
        );
        assert_eq!(loaded.settings.show_cache_miss_notices, Some(true));

        #[cfg(feature = "cache-warming")]
        {
            let mut spine = AgentSpineConfig::new(&proj);
            loaded.apply_to_spine(&mut spine);
            let cw = spine.cache_warming.unwrap();
            assert_eq!(cw.mode, CacheWarmingMode::Streaming);
            assert!(cw.show_cache_miss_notices);
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn loads_and_overlays_subagent_settings() {
        let tmp = std::env::temp_dir().join(format!("mizar_subagent_cfg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let global = tmp.join("global");
        let proj = tmp.join("repo");
        std::fs::create_dir_all(&global).unwrap();
        std::fs::create_dir_all(proj.join(".mizar")).unwrap();

        std::fs::write(
            global.join("settings.json"),
            r#"{
                "subagent": {
                    "maxSteps": 25,
                    "timeoutSecs": 240,
                    "stallTimeoutSecs": 60
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            proj.join(".mizar/settings.json"),
            r#"{
                "subagent": {
                    "maxConcurrent": 6,
                    "stallTimeoutSecs": 30,
                    "maxWatchdogAlerts": 3
                }
            }"#,
        )
        .unwrap();

        let loaded = load_mizar_config_with_global(&proj, Some(&global));
        assert!(loaded.errors.is_empty(), "{:?}", loaded.errors);
        let s = loaded.settings.subagent.as_ref().unwrap();
        assert_eq!(s.max_steps, Some(25));
        assert_eq!(s.timeout_secs, Some(240));
        assert_eq!(s.max_concurrent, Some(6));
        assert_eq!(s.stall_timeout_secs, Some(30));
        assert_eq!(s.max_watchdog_alerts, Some(3));

        #[cfg(feature = "subagent")]
        {
            let mut spine = AgentSpineConfig::new(&proj);
            loaded.apply_to_spine(&mut spine);
            let sub = spine.subagent.unwrap();
            assert_eq!(sub.default_max_steps, 25);
            assert_eq!(sub.timeout_secs, 240);
            assert_eq!(sub.max_concurrent_per_parent, 6);
            assert_eq!(sub.stall_timeout_secs, 30);
            assert_eq!(sub.max_watchdog_alerts, 3);
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
