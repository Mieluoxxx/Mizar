//! 前端无关设置条目构建、运行期配置热切换、原子落盘与内置模块启停控制器。

use std::path::Path;

use cordis::Context;
use mizar_agent_spine::{
    CompactionContextExt, SpinePluginsContextExt, normalize_session_title_thinking_level,
    patch_settings_file, persist_disabled_plugins, resolve_global_settings_path,
};
use mizar_client_core::{ApprovalMode, NoticeLevel, SettingEntry, SettingItemKind};
use mizar_llm::ReasoningEffort;
use mizar_session_title::{SessionTitleContextExt, parse_model_ref};

/// 前端传递给设置列表构建器的纯视图状态快照（零终端/UI 结构依赖）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsViewSnapshot {
    pub show_thinking: bool,
    pub verbose_expanded: bool,
    pub side_panel_open: bool,
    pub mouse_capture_enabled: bool,
    pub approval_mode: ApprovalMode,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub theme_preset_id: String,
    pub screen_mode_id: String,
}

impl Default for SettingsViewSnapshot {
    fn default() -> Self {
        Self {
            show_thinking: true,
            verbose_expanded: false,
            side_panel_open: false,
            mouse_capture_enabled: true,
            approval_mode: ApprovalMode::Auto,
            reasoning_effort: None,
            theme_preset_id: "auto".to_string(),
            screen_mode_id: "fullscreen".to_string(),
        }
    }
}

/// 内置 Cordis 模块热切换的结构化执行结果，供各端（TUI / Merak）同步各自界面与提示。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleToggleOutcome {
    pub ok: bool,
    pub plugin_id: String,
    pub enabled: bool,
    pub label: String,
    pub disabled_ids: Vec<String>,
    pub flash: Option<String>,
    pub notices: Vec<(NoticeLevel, String)>,
}

fn choice_index(options: &[&str], current: &str) -> usize {
    options
        .iter()
        .position(|opt| opt.eq_ignore_ascii_case(current))
        .unwrap_or(0)
}

fn format_max_context_tokens_label(cap: Option<u64>) -> &'static str {
    match cap {
        None | Some(0) => "off",
        Some(1..=300_000) => "200k",
        Some(_) => "400k",
    }
}

fn parse_max_context_tokens_label(label: &str) -> Option<u64> {
    match label.trim().to_ascii_lowercase().as_str() {
        "200k" | "200000" => Some(200_000),
        "400k" | "400000" => Some(400_000),
        _ => None,
    }
}

/// 根据前端视图快照与 Cordis 运行时服务状态构建统一 `/settings` 条目列表。
pub fn build_settings_entries(ctx: &Context, snap: &SettingsViewSnapshot) -> Vec<SettingEntry> {
    let theme_opts = [
        "auto",
        "mist-dark",
        "mist-light",
        "sakura-dark",
        "system",
        "ansi16",
    ];
    let screen_opts = ["fullscreen", "inline"];
    let approval_opts = ["auto", "ask"];
    let effort_opts = [
        "default", "off", "minimal", "low", "medium", "high", "xhigh", "max",
    ];
    let current_effort = snap
        .reasoning_effort
        .map(|e| format!("{e:?}").to_lowercase())
        .unwrap_or_else(|| "default".to_string());

    let mut items = vec![
        SettingEntry {
            key: "ui.theme".into(),
            category: "界面".into(),
            label: "主题配色".into(),
            description: "切换 Oklab / 自适应终端主题，切换后即时刷新".into(),
            kind: SettingItemKind::Choice {
                options: theme_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&theme_opts, &snap.theme_preset_id),
            },
        },
        SettingEntry {
            key: "ui.screenMode".into(),
            category: "界面".into(),
            label: "终端视图模式".into(),
            description: "fullscreen（全屏独立缓冲区）或 inline（内联保留终端滚屏历史）".into(),
            kind: SettingItemKind::Choice {
                options: screen_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&screen_opts, &snap.screen_mode_id),
            },
        },
        SettingEntry {
            key: "ui.showThinking".into(),
            category: "界面".into(),
            label: "显示思考过程".into(),
            description: "在对话流中显示模型的推理思考块（快捷键 Ctrl+O）".into(),
            kind: SettingItemKind::Toggle {
                value: snap.show_thinking,
            },
        },
        SettingEntry {
            key: "ui.verboseExpanded".into(),
            category: "界面".into(),
            label: "展开工具详情".into(),
            description: "默认展开工具参数与折叠 diff 预览（快捷键 Ctrl+O）".into(),
            kind: SettingItemKind::Toggle {
                value: snap.verbose_expanded,
            },
        },
        SettingEntry {
            key: "ui.sidePanelOpen".into(),
            category: "界面".into(),
            label: "右侧信息分栏".into(),
            description: "展开或隐藏右侧上下文占用、文件变更与工具侧栏（快捷键 Ctrl+B）".into(),
            kind: SettingItemKind::Toggle {
                value: snap.side_panel_open,
            },
        },
        SettingEntry {
            key: "ui.mouseCapture".into(),
            category: "界面".into(),
            label: "鼠标追踪与点击".into(),
            description: "开启应用内鼠标滚动/点击；关闭后可直接使用终端原生划词复制（F2）".into(),
            kind: SettingItemKind::Toggle {
                value: snap.mouse_capture_enabled,
            },
        },
        SettingEntry {
            key: "ui.approvalMode".into(),
            category: "运行".into(),
            label: "危险工具审批模式".into(),
            description: "auto 自动放行；ask 在写文件与执行 Shell 前弹窗确认".into(),
            kind: SettingItemKind::Choice {
                options: approval_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&approval_opts, snap.approval_mode.as_str()),
            },
        },
        SettingEntry {
            key: "reasoningEffort".into(),
            category: "运行".into(),
            label: "默认思考强度".into(),
            description: "控制推理模型的思考预算（default 使用模型默认；也可用 Tab 快速循环）"
                .into(),
            kind: SettingItemKind::Choice {
                options: effort_opts.iter().map(|s| s.to_string()).collect(),
                selected: choice_index(&effort_opts, &current_effort),
            },
        },
    ];

    if let Ok(comp) = ctx.compaction() {
        let cfg = comp.config();
        let cap_opts = ["off", "200k", "400k"];
        items.extend([
            SettingEntry {
                key: "compaction.enabled".into(),
                category: "压缩".into(),
                label: "启用上下文压缩".into(),
                description: "总开关：关闭后不会执行自动压缩或 /compact".into(),
                kind: SettingItemKind::Toggle { value: cfg.enabled },
            },
            SettingEntry {
                key: "compaction.autoCompact".into(),
                category: "压缩".into(),
                label: "自动检查点摘要".into(),
                description: "上下文达到有效窗口 80% 时自动生成结构化检查点摘要".into(),
                kind: SettingItemKind::Toggle {
                    value: cfg.auto_compact,
                },
            },
            SettingEntry {
                key: "compaction.maxContextTokens".into(),
                category: "压缩".into(),
                label: "上下文软上限".into(),
                description: "为超大窗口模型设置可选上限（默认 off，使用真实可用窗口）".into(),
                kind: SettingItemKind::Choice {
                    options: cap_opts.iter().map(|s| s.to_string()).collect(),
                    selected: choice_index(
                        &cap_opts,
                        format_max_context_tokens_label(cfg.max_context_tokens),
                    ),
                },
            },
        ]);
    }

    if let Ok(st) = ctx.session_title() {
        let cfg = st.config();
        let thinking_opts = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
        let current_thinking = cfg.reasoning_effort.unwrap_or(ReasoningEffort::Low);
        let current_thinking_str = format!("{current_thinking:?}").to_lowercase();
        items.extend([
            SettingEntry {
                key: "sessionTitle.enabled".into(),
                category: "会话标题".into(),
                label: "自动生成会话标题".into(),
                description: "首轮输入及周期性自动概括会话主题，人工改名后不自动覆盖".into(),
                kind: SettingItemKind::Toggle { value: cfg.enabled },
            },
            SettingEntry {
                key: "sessionTitle.model".into(),
                category: "会话标题".into(),
                label: "标题生成模型".into(),
                description: "指定生成会话标题所用模型（Enter 选择，← 恢复跟随会话模型）".into(),
                kind: SettingItemKind::Model {
                    provider: cfg.provider,
                    model: cfg.model,
                },
            },
            SettingEntry {
                key: "sessionTitle.thinkingLevel".into(),
                category: "会话标题".into(),
                label: "标题生成思考强度".into(),
                description: "控制后台生成标题时的推理强度，默认 low 兼顾速度与成本".into(),
                kind: SettingItemKind::Choice {
                    options: thinking_opts.iter().map(|s| s.to_string()).collect(),
                    selected: choice_index(&thinking_opts, &current_thinking_str),
                },
            },
        ]);
    }

    if let Ok(plugins) = ctx.spine_plugins() {
        for module in plugins.optional_modules() {
            items.push(SettingEntry {
                key: format!("plugin.{}", module.id),
                category: "模块".into(),
                label: format!("{} ({})", module.label, module.id),
                description: module.description,
                kind: SettingItemKind::Toggle {
                    value: module.enabled,
                },
            });
        }
    }

    items
}

/// 运行期热更新内置 `SpinePlugins` 模块的启用状态，并可选原子写回 `disabledPlugins`。
pub async fn toggle_spine_module(
    ctx: &Context,
    settings_path: Option<&Path>,
    plugin_id: &str,
    enabled: bool,
) -> ModuleToggleOutcome {
    let Ok(plugins) = ctx.spine_plugins() else {
        return ModuleToggleOutcome {
            ok: false,
            plugin_id: plugin_id.to_string(),
            enabled,
            label: plugin_id.to_string(),
            disabled_ids: Vec::new(),
            flash: None,
            notices: vec![(
                NoticeLevel::Error,
                "spinePlugins 控制器服务不可用".to_string(),
            )],
        };
    };

    let label = plugins
        .get(plugin_id)
        .map(|m| format!("{} ({})", m.label, m.id))
        .unwrap_or_else(|| plugin_id.to_string());

    match plugins.set_enabled(plugin_id, enabled).await {
        Ok(_) => {
            let disabled = plugins.disabled_ids();
            let mut notices = Vec::new();
            if let Some(path) = settings_path {
                if let Err(err) = persist_disabled_plugins(path, &disabled) {
                    notices.push((
                        NoticeLevel::Warn,
                        format!("模块 `{plugin_id}` 已实时切换，但写入 settings.json 失败：{err}"),
                    ));
                }
            } else {
                notices.push((
                    NoticeLevel::Warn,
                    format!(
                        "模块 `{plugin_id}` 已实时切换，但无法解析 ~/.mizar/settings.json 路径"
                    ),
                ));
            }
            let state_word = if enabled { "启用" } else { "禁用" };
            let msg = format!("已{state_word}模块 {label}");
            let flash = Some(format!("✓ {msg}"));
            notices.push((NoticeLevel::Info, msg));
            ModuleToggleOutcome {
                ok: true,
                plugin_id: plugin_id.to_string(),
                enabled,
                label,
                disabled_ids: disabled,
                flash,
                notices,
            }
        }
        Err(err) => ModuleToggleOutcome {
            ok: false,
            plugin_id: plugin_id.to_string(),
            enabled,
            label,
            disabled_ids: plugins.disabled_ids(),
            flash: Some(format!("切换模块 `{plugin_id}` 失败")),
            notices: vec![(
                NoticeLevel::Error,
                format!("切换模块 `{plugin_id}` 失败：{err}"),
            )],
        },
    }
}

/// 运行期热更新与终端实例无关的模块配置（`compaction.*`、`sessionTitle.*`）。
pub fn apply_runtime_setting_change(ctx: &Context, key: &str, value: &str) {
    if key.starts_with("compaction.")
        && let Ok(comp) = ctx.compaction()
    {
        let flag = value == "true";
        comp.update_config(|cfg| match key {
            "compaction.enabled" => cfg.enabled = flag,
            "compaction.autoCompact" => cfg.auto_compact = flag,
            "compaction.maxContextTokens" => {
                cfg.max_context_tokens = parse_max_context_tokens_label(value)
            }
            _ => {}
        });
    } else if key.starts_with("sessionTitle.")
        && let Ok(st) = ctx.session_title()
    {
        st.update_config(|cfg| match key {
            "sessionTitle.enabled" => cfg.enabled = value == "true",
            "sessionTitle.model" => {
                if let Some((provider, model)) = parse_model_ref(value) {
                    cfg.provider = provider;
                    cfg.model = Some(model);
                } else {
                    cfg.provider = None;
                    cfg.model = None;
                }
            }
            "sessionTitle.thinkingLevel" => {
                cfg.reasoning_effort = Some(normalize_session_title_thinking_level(Some(value)));
            }
            _ => {}
        });
    }
}

fn upsert_nested_json(
    root: &mut serde_json::Map<String, serde_json::Value>,
    section: &str,
    field: &str,
    val: serde_json::Value,
) {
    let entry = root
        .entry(section.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !entry.is_object() {
        *entry = serde_json::Value::Object(serde_json::Map::new());
    }
    if let Some(obj) = entry.as_object_mut() {
        obj.insert(field.to_string(), val);
    }
}

/// 将单条设置原子写入默认的 `~/.mizar/settings.json` 文件。
pub fn persist_setting_change(key: &str, value: &str) -> Result<(), String> {
    let path = resolve_global_settings_path()
        .ok_or_else(|| "无法解析 ~/.mizar/settings.json 路径".to_string())?;
    persist_setting_change_to(&path, key, value)
}

/// 将单条设置原子写入指定的 `settings.json` 文件。
pub fn persist_setting_change_to(path: &Path, key: &str, value: &str) -> Result<(), String> {
    patch_settings_file(path, |root| match key {
        "reasoningEffort" => {
            if value == "default" {
                root.remove("reasoningEffort");
            } else {
                root.insert(
                    "reasoningEffort".to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
        "ui.theme" => upsert_nested_json(
            root,
            "ui",
            "theme",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.screenMode" => upsert_nested_json(
            root,
            "ui",
            "screenMode",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.approvalMode" => upsert_nested_json(
            root,
            "ui",
            "approvalMode",
            serde_json::Value::String(value.to_string()),
        ),
        "ui.showThinking" => upsert_nested_json(
            root,
            "ui",
            "showThinking",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.verboseExpanded" => upsert_nested_json(
            root,
            "ui",
            "verboseExpanded",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.sidePanelOpen" => upsert_nested_json(
            root,
            "ui",
            "sidePanelOpen",
            serde_json::Value::Bool(value == "true"),
        ),
        "ui.mouseCapture" => upsert_nested_json(
            root,
            "ui",
            "mouseCapture",
            serde_json::Value::Bool(value == "true"),
        ),
        "compaction.maxContextTokens" => {
            if let Some(cap) = parse_max_context_tokens_label(value) {
                upsert_nested_json(
                    root,
                    "compaction",
                    "maxContextTokens",
                    serde_json::Value::Number(serde_json::Number::from(cap)),
                );
            } else if let Some(obj) = root.get_mut("compaction").and_then(|v| v.as_object_mut()) {
                obj.remove("maxContextTokens");
            }
        }
        k if let Some(field) = k.strip_prefix("compaction.") => upsert_nested_json(
            root,
            "compaction",
            field,
            serde_json::Value::Bool(value == "true"),
        ),
        "sessionTitle.enabled" => upsert_nested_json(
            root,
            "sessionTitle",
            "enabled",
            serde_json::Value::Bool(value == "true"),
        ),
        "sessionTitle.model" => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                if let Some(obj) = root.get_mut("sessionTitle").and_then(|v| v.as_object_mut()) {
                    obj.remove("model");
                }
            } else {
                upsert_nested_json(
                    root,
                    "sessionTitle",
                    "model",
                    serde_json::Value::String(trimmed.to_string()),
                );
            }
        }
        "sessionTitle.thinkingLevel" => upsert_nested_json(
            root,
            "sessionTitle",
            "thinkingLevel",
            serde_json::Value::String(value.trim().to_ascii_lowercase()),
        ),
        _ => {}
    })
}

#[cfg(test)]
mod tests {
    use mizar_agent_spine::{AgentSpineConfig, AgentSpinePlugin, AgentToolsMode};
    use mizar_session::SessionContextExt;

    use super::*;

    #[tokio::test]
    async fn settings_module_entries_toggle_persists_disabled_plugins_and_refreshes_runtime_state()
    {
        let tmp =
            std::env::temp_dir().join(format!("mizar_client_settings_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let ws = tmp.join("ws");
        std::fs::create_dir_all(ws.join(".git")).unwrap();
        std::fs::create_dir_all(ws.join(".agents/skills/client-demo")).unwrap();
        std::fs::write(ws.join("AGENTS.md"), "# Client Rule\nKeep clean\n").unwrap();
        std::fs::write(
            ws.join(".agents/skills/client-demo/SKILL.md"),
            "---\nname: client-demo\ndescription: Client demo skill\n---\nDEMO\n",
        )
        .unwrap();
        let settings_path = tmp.join(".mizar/settings.json");
        std::fs::create_dir_all(settings_path.parent().unwrap()).unwrap();
        std::fs::write(&settings_path, r#"{"defaultModel":"gpt-4o"}"#).unwrap();

        let ctx = Context::new();
        let mut cfg = AgentSpineConfig::coding(&ws).with_disabled_plugins(["compaction"]);
        if let AgentToolsMode::Coding { ref mut mcp, .. } = cfg.tools_mode {
            mcp.global_config_dir = Some(tmp.join(".mizar"));
        }
        ctx.plugin(AgentSpinePlugin, cfg).await.unwrap();
        crate::events::wait_for_services(&ctx).await.unwrap();

        let session = ctx.sessions().unwrap().create(None);
        let snap = SettingsViewSnapshot::default();
        assert!(
            crate::workspace::collect_user_skill_names(&ctx, &ws)
                .await
                .contains(&"client-demo".to_string())
        );
        assert!(
            !crate::workspace::collect_instructions_summary(&ctx, Some(&session.id)).is_empty()
        );

        // 启动时 compaction 被屏蔽：设置列表中不含 [压缩] 配置项，且 plugin.compaction 为 false
        let initial_entries = build_settings_entries(&ctx, &snap);
        assert!(
            !initial_entries.iter().any(|e| e.category == "压缩"),
            "compaction entries must be omitted while module is disabled"
        );
        let comp_entry = initial_entries
            .iter()
            .find(|e| e.key == "plugin.compaction")
            .unwrap();
        assert_eq!(comp_entry.kind, SettingItemKind::Toggle { value: false });

        // 1. 运行期启用 compaction -> 服务上线 + 列表包含 [压缩] + 写入 disabledPlugins: []
        let out_enable = toggle_spine_module(&ctx, Some(&settings_path), "compaction", true).await;
        assert!(out_enable.ok);
        assert!(out_enable.disabled_ids.is_empty());
        assert!(ctx.compaction().is_ok());

        let after_enable = build_settings_entries(&ctx, &snap);
        let comp_keys: Vec<&str> = after_enable
            .iter()
            .filter(|e| e.category == "压缩")
            .map(|e| e.key.as_str())
            .collect();
        assert_eq!(
            comp_keys,
            vec![
                "compaction.enabled",
                "compaction.autoCompact",
                "compaction.maxContextTokens",
            ]
        );
        apply_runtime_setting_change(&ctx, "compaction.maxContextTokens", "200k");
        assert_eq!(
            ctx.compaction().unwrap().config().max_context_tokens,
            Some(200_000)
        );
        persist_setting_change_to(&settings_path, "compaction.maxContextTokens", "200k").unwrap();
        let persisted_cap: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(persisted_cap["compaction"]["maxContextTokens"], 200_000);
        apply_runtime_setting_change(&ctx, "compaction.maxContextTokens", "off");
        assert_eq!(ctx.compaction().unwrap().config().max_context_tokens, None);
        persist_setting_change_to(&settings_path, "compaction.maxContextTokens", "off").unwrap();
        let persisted_off: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert!(
            persisted_off["compaction"]
                .get("maxContextTokens")
                .is_none()
        );
        let persisted_1: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(persisted_1["defaultModel"], "gpt-4o");
        assert_eq!(persisted_1["disabledPlugins"], serde_json::json!([]));

        // 2. 禁用 skills / instructions / compaction -> skills 与指令采集清空、[压缩] 移除、排序去重落盘
        for id in ["skills", "instructions", "compaction"] {
            let res = toggle_spine_module(&ctx, Some(&settings_path), id, false).await;
            assert!(res.ok);
        }
        assert!(
            crate::workspace::collect_user_skill_names(&ctx, &ws)
                .await
                .is_empty()
        );
        assert!(crate::workspace::collect_instructions_summary(&ctx, Some(&session.id)).is_empty());
        assert!(ctx.compaction().is_err());
        let after_disable = build_settings_entries(&ctx, &snap);
        assert!(!after_disable.iter().any(|e| e.category == "压缩"));
        let persisted_2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&settings_path).unwrap()).unwrap();
        assert_eq!(
            persisted_2["disabledPlugins"],
            serde_json::json!(["compaction", "instructions", "skills"])
        );

        // 3. 尝试禁用受保护核心模块必须被拒绝且不破坏已有配置
        let protected_res = toggle_spine_module(&ctx, Some(&settings_path), "llm", false).await;
        assert!(!protected_res.ok);

        ctx.fiber().dispose().await;
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
