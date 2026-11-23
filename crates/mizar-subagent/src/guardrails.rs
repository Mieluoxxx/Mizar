//! 单层 Subagent 安全护栏、工具集裁剪、世代号守卫与路由解析辅助模块。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use mizar_llm::ProviderInfo;
use mizar_session::SessionStore;
use mizar_tools::{PreExecuteDecision, ToolExecution, ToolRuntime, ToolRuntimeInfo};

use crate::tasks::ChildActivityTracker;
use crate::{SUBAGENT_TOOL_NAME, SubagentError, SubagentMode};

/// 默认判定为只读的工具白名单（兼容未显式填写 `ToolAnnotations.read_only` 的内置/legacy 工具）。
pub const READ_ONLY_TOOL_ALLOWLIST: &[&str] = &[
    "read",
    "grep",
    "find",
    "list_dir",
    "read_file",
    "get_time",
    "recall",
    "skill",
    "list_mcp_resources",
    "list_mcp_resource_templates",
    "read_mcp_resource",
];

pub(crate) type ActiveChildrenMap = HashMap<String, (u64, SubagentMode)>;
pub(crate) type ActivityTrackersMap = HashMap<String, (u64, Arc<ChildActivityTracker>)>;

/// 判断指定工具在只读模式下是否允许被子代理声明与执行。
pub fn is_read_only_tool(name: &str, info: Option<&ToolRuntimeInfo>) -> bool {
    if name == SUBAGENT_TOOL_NAME {
        return false;
    }
    if READ_ONLY_TOOL_ALLOWLIST.contains(&name) {
        return true;
    }
    info.is_some_and(|i| i.annotations.read_only)
}

/// 根据父会话活跃工具与执行模式计算子会话的显式工具白名单（始终剔除 `subagent`）。
pub fn compute_allowed_tools(
    candidate_names: &[String],
    all_infos: &[ToolRuntimeInfo],
    mode: SubagentMode,
) -> Vec<String> {
    let by_name: HashMap<&str, &ToolRuntimeInfo> =
        all_infos.iter().map(|i| (i.name.as_str(), i)).collect();
    let mut seen = HashSet::new();
    let mut allowed = Vec::new();
    for name in candidate_names {
        if name == SUBAGENT_TOOL_NAME || !seen.insert(name.as_str()) {
            continue;
        }
        let info = by_name.get(name.as_str()).copied();
        let keep = match mode {
            SubagentMode::ReadOnly => is_read_only_tool(name, info),
            SubagentMode::Edit => info.is_some(),
        };
        if keep {
            allowed.push(name.clone());
        }
    }
    allowed
}

/// 在 `tools/pre-execute` 管线最外层拦截子代理越权或嵌套调用（含 `codemode` 嵌套沙箱调用）。
pub fn check_pre_execute(
    exec: &ToolExecution,
    active_children: &Mutex<ActiveChildrenMap>,
    tools: &ToolRuntime,
    sessions: &SessionStore,
) -> PreExecuteDecision {
    let Some(sid) = exec.session_id.as_deref() else {
        return PreExecuteDecision::allow();
    };
    let active_mode = active_children.lock().unwrap().get(sid).map(|(_, m)| *m);
    if let Some(mode) = active_mode {
        if exec.name == SUBAGENT_TOOL_NAME {
            return PreExecuteDecision::deny("子代理严禁嵌套调用 subagent（深度严格限制为 1）");
        }
        if mode == SubagentMode::ReadOnly && !READ_ONLY_TOOL_ALLOWLIST.contains(&exec.name.as_str())
        {
            let info = tools.get_tool_info(&exec.name);
            if !is_read_only_tool(&exec.name, info.as_ref()) {
                return PreExecuteDecision::deny(format!(
                    "只读模式子代理禁止调用非只读工具 {}",
                    exec.name
                ));
            }
        }
        return PreExecuteDecision::allow();
    }

    // 兜底：即使子代理历史会话被外部复用并再次发起 `subagent` 调用，也严格拒绝深度 > 1
    if exec.name == SUBAGENT_TOOL_NAME && is_subagent_child_session(sessions, sid) {
        return PreExecuteDecision::deny("子代理会话严禁嵌套调用 subagent（深度严格限制为 1）");
    }

    PreExecuteDecision::allow()
}

/// 校验当前调用会话是否本身就是子代理会话（正在运行或历史子会话），用于阻止深度 > 1 嵌套。
pub fn ensure_not_nested_parent(
    parent_session_id: Option<&str>,
    active_children: &Mutex<ActiveChildrenMap>,
    sessions: &SessionStore,
) -> Result<(), SubagentError> {
    let Some(pid) = parent_session_id else {
        return Ok(());
    };
    if active_children.lock().unwrap().contains_key(pid) || is_subagent_child_session(sessions, pid)
    {
        return Err(SubagentError::NestedSubagentDenied);
    }
    Ok(())
}

/// 判断给定会话是否属于子代理派生会话。
pub fn is_subagent_child_session(sessions: &SessionStore, session_id: &str) -> bool {
    sessions
        .inspect_readonly(session_id)
        .is_some_and(|loaded| loaded.header.is_subagent_child())
}

/// 绑定单调 `generation`（`run_id`）的子会话 RAII 守卫：仅卸载同一代次的护栏与追踪记录，防并发/乱序误删。
pub struct ChildSessionGuard {
    active_children: Arc<Mutex<ActiveChildrenMap>>,
    activity_trackers: Arc<Mutex<ActivityTrackersMap>>,
    tools: Arc<ToolRuntime>,
    child_session_id: String,
    generation: u64,
}

impl ChildSessionGuard {
    pub(crate) fn register(
        active_children: Arc<Mutex<ActiveChildrenMap>>,
        activity_trackers: Arc<Mutex<ActivityTrackersMap>>,
        tools: Arc<ToolRuntime>,
        child_session_id: String,
        generation: u64,
        mode: SubagentMode,
        tracker: Arc<ChildActivityTracker>,
    ) -> Self {
        active_children
            .lock()
            .unwrap()
            .insert(child_session_id.clone(), (generation, mode));
        activity_trackers
            .lock()
            .unwrap()
            .insert(child_session_id.clone(), (generation, tracker));
        Self {
            active_children,
            activity_trackers,
            tools,
            child_session_id,
            generation,
        }
    }
}

impl Drop for ChildSessionGuard {
    fn drop(&mut self) {
        let removed_active = {
            let mut guard = self.active_children.lock().unwrap();
            if guard
                .get(&self.child_session_id)
                .is_some_and(|(gen_id, _)| *gen_id == self.generation)
            {
                guard.remove(&self.child_session_id);
                true
            } else {
                false
            }
        };
        {
            let mut trackers = self.activity_trackers.lock().unwrap();
            if trackers
                .get(&self.child_session_id)
                .is_some_and(|(gen_id, _)| *gen_id == self.generation)
            {
                trackers.remove(&self.child_session_id);
            }
        }
        if removed_active {
            self.tools.reset_session_activations(&self.child_session_id);
        }
    }
}

/// 生成用于子会话标题与快照的单行紧凑摘要（按 UTF-8 字符边界截断）。
pub fn shorten_prompt(prompt: &str, max_chars: usize) -> String {
    let normalized: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    let mut truncated = false;
    for (count, ch) in normalized.chars().enumerate() {
        if count >= max_chars {
            truncated = true;
            break;
        }
        out.push(ch);
    }
    if truncated {
        out.push('…');
    }
    out
}

/// 从 `AgentToolCallEvent` 生成简短可读工具标签（如 `bash: cargo test`），UTF-8 安全截断至 64 字符。
pub(crate) fn format_tool_activity_label(name: &str, arguments: &str) -> String {
    let hint = serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| {
            ["command", "path", "pattern", "query", "action", "name"]
                .iter()
                .find_map(|key| v.get(key).and_then(|val| val.as_str()).map(str::to_string))
        });
    let combined = match hint {
        Some(h) if !h.trim().is_empty() => format!("{name}: {}", shorten_prompt(&h, 40)),
        _ => name.to_string(),
    };
    combined.chars().take(64).collect()
}

/// 解析模型覆盖参数（支持 `"model_id"` 与 `"provider/model_id"`），未指定时回退父会话路由。
pub fn resolve_model_route(
    requested_model: Option<&str>,
    parent_provider: Option<String>,
    parent_model: Option<String>,
    providers: &[ProviderInfo],
) -> (Option<String>, Option<String>) {
    let Some(spec) = requested_model.map(str::trim).filter(|s| !s.is_empty()) else {
        return (parent_provider, parent_model);
    };

    if let Some(ref cur_pid) = parent_provider
        && let Some(prov) = providers.iter().find(|p| p.id == *cur_pid)
        && prov.models.iter().any(|m| m.id == spec)
    {
        return (Some(cur_pid.clone()), Some(spec.to_string()));
    }

    if let Some((prov_prefix, rest)) = spec.split_once('/') {
        let prov_prefix = prov_prefix.trim();
        let rest = rest.trim();
        if !prov_prefix.is_empty() && !rest.is_empty() {
            if providers.iter().any(|p| p.id == prov_prefix) {
                return (Some(prov_prefix.to_string()), Some(rest.to_string()));
            }
            if let Some(prov) = providers
                .iter()
                .find(|p| p.models.iter().any(|m| m.id == spec))
            {
                return (Some(prov.id.clone()), Some(spec.to_string()));
            }
            return (Some(prov_prefix.to_string()), Some(rest.to_string()));
        }
    }

    if let Some(ref cur_pid) = parent_provider
        && providers
            .iter()
            .find(|p| p.id == *cur_pid)
            .is_some_and(|p| p.models.is_empty() || p.models.iter().any(|m| m.id == spec))
    {
        return (Some(cur_pid.clone()), Some(spec.to_string()));
    }

    if let Some(prov) = providers
        .iter()
        .find(|p| p.models.iter().any(|m| m.id == spec))
    {
        return (Some(prov.id.clone()), Some(spec.to_string()));
    }

    (parent_provider, Some(spec.to_string()))
}

#[cfg(test)]
mod tests {
    use mizar_llm::ModelInfo;
    use mizar_tools::{ToolAnnotations, ToolExecutionMode, ToolExposure};
    use serde_json::json;

    use super::*;

    fn mk_info(name: &str, read_only: bool) -> ToolRuntimeInfo {
        ToolRuntimeInfo {
            name: name.to_string(),
            description: String::new(),
            parameters: json!({}),
            output_schema: None,
            exposure: ToolExposure::Direct,
            execution_mode: ToolExecutionMode::Sequential,
            annotations: ToolAnnotations {
                read_only,
                ..Default::default()
            },
            namespace: None,
            default_active: None,
            prompt_snippet: None,
            prompt_guidelines: Vec::new(),
        }
    }

    #[test]
    fn shorten_prompt_collapses_whitespace_and_respects_utf8_boundaries() {
        assert_eq!(
            shorten_prompt("  检查   模块\n依赖  ", 20),
            "检查 模块 依赖"
        );
        assert_eq!(shorten_prompt("一二三四五六", 4), "一二三四…");
    }

    #[test]
    fn compute_allowed_tools_enforces_mode_and_excludes_subagent() {
        let candidates = vec![
            "read".into(),
            "write".into(),
            "bash".into(),
            "codemode".into(),
            "get_time".into(),
            "custom_ro".into(),
            "subagent".into(),
        ];
        let infos = vec![
            mk_info("read", true),
            mk_info("write", false),
            mk_info("bash", false),
            mk_info("codemode", false),
            mk_info("get_time", false),
            mk_info("custom_ro", true),
            mk_info("subagent", false),
        ];

        assert_eq!(
            compute_allowed_tools(&candidates, &infos, SubagentMode::ReadOnly),
            vec!["read", "get_time", "custom_ro"]
        );
        assert_eq!(
            compute_allowed_tools(&candidates, &infos, SubagentMode::Edit),
            vec!["read", "write", "bash", "codemode", "get_time", "custom_ro"]
        );
    }

    #[test]
    fn resolve_model_route_handles_provider_prefix_and_literal_slash_models() {
        let providers = vec![
            ProviderInfo {
                id: "openrouter".into(),
                name: "OpenRouter".into(),
                protocol: "openai-completions".into(),
                models: vec![ModelInfo {
                    id: "anthropic/claude-3.5-sonnet".into(),
                    name: "Claude 3.5".into(),
                    ..Default::default()
                }],
            },
            ProviderInfo {
                id: "deepseek".into(),
                name: "DeepSeek".into(),
                protocol: "openai-completions".into(),
                models: vec![ModelInfo {
                    id: "deepseek-chat".into(),
                    name: "DeepSeek Chat".into(),
                    ..Default::default()
                }],
            },
        ];

        assert_eq!(
            resolve_model_route(
                None,
                Some("openrouter".into()),
                Some("m1".into()),
                &providers
            ),
            (Some("openrouter".into()), Some("m1".into()))
        );
        assert_eq!(
            resolve_model_route(
                Some("anthropic/claude-3.5-sonnet"),
                Some("openrouter".into()),
                Some("m1".into()),
                &providers
            ),
            (
                Some("openrouter".into()),
                Some("anthropic/claude-3.5-sonnet".into())
            )
        );
        assert_eq!(
            resolve_model_route(
                Some("deepseek/deepseek-chat"),
                Some("openrouter".into()),
                Some("m1".into()),
                &providers
            ),
            (Some("deepseek".into()), Some("deepseek-chat".into()))
        );
        assert_eq!(
            resolve_model_route(
                Some("deepseek-chat"),
                Some("openrouter".into()),
                Some("m1".into()),
                &providers
            ),
            (Some("deepseek".into()), Some("deepseek-chat".into()))
        );
    }

    #[tokio::test]
    async fn check_pre_execute_readonly_uses_allowlist_and_single_tool_lookup() {
        use cordis::{BoxError, Context, Service};
        use mizar_session::SessionContextExt;
        use mizar_tools::{ToolDefinition, ToolsContextExt};

        let app = Context::new();
        app.plugin(SessionStore::plugin(), ()).await.unwrap();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        let tools = app.tools().unwrap();
        let _ro = tools
            .register(
                ToolDefinition::sync_tool("custom_ro", "ro", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".into())
                })
                .with_annotations(ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                }),
            )
            .unwrap();
        let _rw = tools
            .register(ToolDefinition::sync_tool(
                "custom_rw",
                "rw",
                json!({}),
                |_, _| Ok::<_, BoxError>("ok".into()),
            ))
            .unwrap();
        let _excl = tools
            .register(
                ToolDefinition::sync_tool("excluded_ro", "ro", json!({}), |_, _| {
                    Ok::<_, BoxError>("ok".into())
                })
                .with_annotations(ToolAnnotations {
                    read_only: true,
                    ..Default::default()
                }),
            )
            .unwrap();
        tools.set_tool_selection(None, vec!["excluded_ro".to_string()]);

        let sessions = app.sessions().unwrap();
        let active = Mutex::new(HashMap::from([(
            "child-ro".to_string(),
            (1u64, SubagentMode::ReadOnly),
        )]));

        assert!(
            check_pre_execute(
                &ToolExecution::new("read", json!({}), Some("child-ro".into())),
                &active,
                &tools,
                &sessions,
            )
            .allow
        );
        assert!(
            check_pre_execute(
                &ToolExecution::new("custom_ro", json!({}), Some("child-ro".into())),
                &active,
                &tools,
                &sessions,
            )
            .allow
        );
        assert!(
            !check_pre_execute(
                &ToolExecution::new("custom_rw", json!({}), Some("child-ro".into())),
                &active,
                &tools,
                &sessions,
            )
            .allow
        );
        assert!(
            !check_pre_execute(
                &ToolExecution::new("excluded_ro", json!({}), Some("child-ro".into())),
                &active,
                &tools,
                &sessions,
            )
            .allow
        );
    }
}
