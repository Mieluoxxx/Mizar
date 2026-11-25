//! `tool_search` 插件：使用 BM25 检索当前会话未激活的 `Codemode`/`Deferred` 工具并动态加载到下一轮声明集。

pub mod ranker;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_tools::{
    EVENT_TOOLS_CHANGE, ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolExposure,
    ToolOutput, ToolRuntime, ToolRuntimeInfo, ToolsContextExt,
};
use serde_json::{Value, json};

pub use ranker::{
    Bm25Ranker, ToolSearchDocument, ToolSearchMatch, create_tool_search_document, tokenize,
};

pub const TOOL_SEARCH_TOOL_NAME: &str = "tool_search";
pub const DEFAULT_TOOL_SEARCH_LIMIT: usize = 8;

#[derive(Debug, Clone)]
pub struct ToolSearchConfig {
    /// 当存在任意外部 MCP `Codemode` 或 `Deferred` 工具时自动激活 `tool_search`（默认 true）。
    pub auto_enable: bool,
    pub default_limit: usize,
}

impl Default for ToolSearchConfig {
    fn default() -> Self {
        Self {
            auto_enable: true,
            default_limit: DEFAULT_TOOL_SEARCH_LIMIT,
        }
    }
}

pub fn is_searchable_exposure(exposure: ToolExposure) -> bool {
    matches!(exposure, ToolExposure::Codemode | ToolExposure::Deferred)
}

pub fn is_external_mcp_tool(info: &ToolRuntimeInfo) -> bool {
    info.name.starts_with("mcp__")
        || info
            .namespace
            .as_ref()
            .is_some_and(|ns| ns.name.starts_with("mcp__"))
}

pub fn is_searchable_tool(info: &ToolRuntimeInfo) -> bool {
    is_searchable_exposure(info.exposure) && is_external_mcp_tool(info)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutoEnableState {
    Inactive,
    Active,
}

fn sync_auto_enable(tools: &ToolRuntime, state: &Mutex<AutoEnableState>) {
    let has_searchable = tools.list_all().into_iter().any(|t| is_searchable_tool(&t));
    let next = if has_searchable {
        AutoEnableState::Active
    } else {
        AutoEnableState::Inactive
    };
    let changed = {
        let mut cur = state.lock().unwrap();
        if *cur == next {
            false
        } else {
            *cur = next;
            true
        }
    };
    if changed {
        tools.set_auto_default_active(TOOL_SEARCH_TOOL_NAME, Some(has_searchable));
    }
}

pub fn search_inactive_tools(
    tools: &ToolRuntime,
    session_id: Option<&str>,
    query: &str,
    limit: usize,
) -> Vec<ToolRuntimeInfo> {
    let active: HashSet<String> = match session_id {
        Some(sid) => tools.get_active_tools(sid).into_iter().collect(),
        None => tools
            .list_declared(None)
            .into_iter()
            .map(|t| t.name)
            .collect(),
    };

    let candidates: Vec<ToolRuntimeInfo> = tools
        .list_all()
        .into_iter()
        .filter(|t| is_searchable_tool(t) && !active.contains(&t.name))
        .collect();

    let docs: Vec<ToolSearchDocument> =
        candidates.iter().map(create_tool_search_document).collect();
    let matches = Bm25Ranker::default().rank(query, &docs, limit);

    let matched_names: Vec<String> = matches.iter().map(|m| m.name.clone()).collect();
    if let Some(sid) = session_id {
        if !matched_names.is_empty() {
            tools.activate_tools(sid, &matched_names);
        }
    } else {
        for name in &matched_names {
            tools.set_global_active_override(name, true);
        }
    }

    matches
        .into_iter()
        .filter_map(|m| candidates.iter().find(|t| t.name == m.name).cloned())
        .collect()
}

#[derive(Default, Clone, Copy)]
pub struct ToolSearchPlugin;

impl Plugin for ToolSearchPlugin {
    type Config = ToolSearchConfig;

    fn name(&self) -> Option<&str> {
        Some("tool-search")
    }

    fn inject(&self) -> Inject {
        Inject::from(["tools"])
    }

    fn apply(
        &self,
        ctx: Context,
        config: Self::Config,
    ) -> BoxFuture<'static, Result<Option<Cleanup>, BoxError>> {
        Box::pin(async move {
            let tools = ctx.tools()?;
            let default_limit = config.default_limit.max(1);
            let reg = ctx.effect("tool-search.register", || {
                Ok::<_, mizar_tools::ToolError>(Some(tools.register(
                    ToolDefinition::async_rich_tool(
                        TOOL_SEARCH_TOOL_NAME,
                        "Search over deferred tool metadata with BM25 and expose matching tools for your next model call. Use this when a deferred (`tool_search`) MCP tool is not yet declared to you; for `codemode` servers, prefer `searchTools()` / `describeTool()` inside `codemode` scripts.",
                        json!({
                            "type": "object",
                            "properties": {
                                "query": { "type": "string", "description": "Search query for deferred or MCP tools" },
                                "limit": { "type": "integer", "minimum": 1, "description": format!("Maximum number of tools to return. Defaults to {default_limit}.") }
                            },
                            "required": ["query"]
                        }),
                        move |args, call_ctx| async move {
                            let query = args
                                .get("query")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .ok_or("query 不能为空")?;
                            let limit = args
                                .get("limit")
                                .and_then(Value::as_u64)
                                .map(|v| usize::try_from(v.max(1)).unwrap_or(default_limit))
                                .unwrap_or(default_limit);

                            let rt = call_ctx.ctx.tools()?;
                            let loaded = search_inactive_tools(
                                &rt,
                                call_ctx.session_id(),
                                query,
                                limit,
                            );
                            let loaded_names: Vec<String> =
                                loaded.iter().map(|t| t.name.clone()).collect();
                            let text = if loaded.is_empty() {
                                "No matching tools found.".to_string()
                            } else {
                                let lines: Vec<String> = loaded
                                    .iter()
                                    .map(|t| {
                                        let first_line = t
                                            .description
                                            .trim()
                                            .lines()
                                            .next()
                                            .unwrap_or("");
                                        format!("- {}: {}", t.name, first_line)
                                    })
                                    .collect();
                                format!(
                                    "Loaded {} tool{}. They are available from your next call:\n{}",
                                    loaded.len(),
                                    if loaded.len() == 1 { "" } else { "s" },
                                    lines.join("\n")
                                )
                            };

                            let meta = json!({ "loaded": loaded_names });
                            Ok(ToolOutput::text(text)
                                .with_structured(meta.clone())
                                .with_details(meta))
                        },
                    )
                    .with_exposure(ToolExposure::ModelOnly)
                    .with_default_active(false)
                    .with_execution_mode(ToolExecutionMode::Sequential)
                    .with_annotations(ToolAnnotations {
                        read_only: true,
                        destructive: false,
                        idempotent: true,
                        open_world: false,
                    })
                    .with_prompt_snippet("Search for unloaded MCP/deferred tools and activate matches"),
                )?))
            })?;

            let mut handles = vec![reg];
            if config.auto_enable {
                let last_state = Arc::new(Mutex::new(AutoEnableState::Inactive));
                sync_auto_enable(&tools, &last_state);
                let tools_sub = tools;
                let last_sub = Arc::clone(&last_state);
                let sub = ctx.on(EVENT_TOOLS_CHANGE, move |_: &()| {
                    sync_auto_enable(&tools_sub, &last_sub);
                })?;
                handles.push(sub);
            }

            Ok(Some(Cleanup::async_fn(move || async move {
                for h in handles {
                    h.dispose().await;
                }
            })))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cordis::{Context, Service};
    use mizar_tools::{ToolExecution, ToolNamespace};

    #[tokio::test]
    async fn bm25_finds_and_activates_deferred_mcp_tools() {
        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(ToolSearchPlugin, ToolSearchConfig::default())
            .await
            .unwrap();

        let tools = app.tools().unwrap();
        assert!(
            tools.list_declared(Some("s1")).is_empty(),
            "tool_search should stay inactive until searchable tools exist"
        );

        // 非外部 MCP 的 Deferred 工具不应触发 tool_search 注入
        let _local_deferred = tools
            .register(
                ToolDefinition::sync_tool(
                    "local_deferred_helper",
                    "Internal deferred helper not from MCP",
                    json!({"type": "object", "properties": {}}),
                    |_, _| Ok::<_, BoxError>("ok".into()),
                )
                .with_exposure(ToolExposure::Deferred),
            )
            .unwrap();
        assert!(
            tools.list_declared(Some("s1")).is_empty(),
            "non-MCP deferred tool must not activate tool_search"
        );

        let _m1 = tools
            .register(
                ToolDefinition::sync_tool(
                    "mcp__fff__find_files",
                    "Fuzzy file search with frecency ranking",
                    json!({"type": "object", "properties": {"query": {"type": "string"}}}),
                    |_, _| Ok::<_, BoxError>("file.rs".into()),
                )
                .with_exposure(ToolExposure::Deferred)
                .with_namespace(ToolNamespace {
                    name: "mcp__fff".into(),
                    aliases: vec!["fff".into()],
                    description: Some("Fast File Finder".into()),
                    instructions: None,
                }),
            )
            .unwrap();

        let _m2 = tools
            .register(
                ToolDefinition::sync_tool(
                    "mcp__fff__grep",
                    "Search file content across repository",
                    json!({"type": "object", "properties": {"pattern": {"type": "string"}}}),
                    |_, _| Ok::<_, BoxError>("match".into()),
                )
                .with_exposure(ToolExposure::Deferred)
                .with_namespace(ToolNamespace::new("mcp__fff")),
            )
            .unwrap();

        let declared = tools.list_declared(Some("s1"));
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].name, "tool_search");

        let res = tools
            .execute(ToolExecution::new(
                "tool_search",
                json!({ "query": "fff grep content" }),
                Some("s1".into()),
            ))
            .await;
        assert!(!res.is_error, "err: {}", res.content);
        assert!(res.content.contains("mcp__fff__grep"));

        let declared_after = tools.list_declared(Some("s1"));
        assert!(declared_after.iter().any(|t| t.name == "mcp__fff__grep"));

        app.fiber().dispose().await;
    }
}
