//! Codemode JS 标识符转换、TypeScript 工具声明渲染与 `prepare_loadout` 预算筛选（对齐 `pi` `codemode/tool.ts`）。

use std::collections::{BTreeMap, HashMap, HashSet};

use mizar_tools::{ToolExposure, ToolLoadout, ToolLoadoutChanges, ToolRuntimeInfo};

use crate::schema_ts::{DEFAULT_INPUT_SCHEMA_MAX_CHARS, describe_output, render_tool_sample};

pub const CODEMODE_TOOL_NAME: &str = "codemode";
pub const DEFAULT_CODEMODE_INLINE_BUDGET: usize = 3000;
const CHARS_PER_TOKEN: usize = 4;

/// 将工具原始名转为合法 JS 标识符（如 `mcp__fff__grep` 原样可用，`-`/`.` 等替换为 `_`）。
pub fn to_codemode_identifier(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() || out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

fn render_tool_section(info: &ToolRuntimeInfo) -> String {
    let id = to_codemode_identifier(&info.name);
    let heading = if id == info.name {
        format!("### `{id}`")
    } else {
        format!("### `{id}` (`{}`)", info.name)
    };
    let sample = render_tool_sample(info, DEFAULT_INPUT_SCHEMA_MAX_CHARS);
    format!("{heading}\n{}", sample.trim())
}

struct CatalogEntry {
    name: String,
    section: String,
    cost: usize,
}

/// 轮询每个命名空间，保证各个命名空间都能按预算公平展示至少一部分工具。
fn select_catalog(
    groups: &[(Option<String>, Vec<CatalogEntry>)],
    budget_tokens: usize,
) -> Vec<String> {
    let mut queues: Vec<Vec<&CatalogEntry>> = groups
        .iter()
        .map(|(_, entries)| {
            let mut sorted: Vec<&CatalogEntry> = entries.iter().collect();
            sorted.sort_by_key(|e| e.cost);
            sorted
        })
        .collect();

    let mut shown = Vec::new();
    let mut remaining = budget_tokens;
    let mut active: Vec<usize> = (0..queues.len())
        .filter(|&i| !queues[i].is_empty())
        .collect();

    while !active.is_empty() {
        let mut next_active = Vec::new();
        for idx in active {
            if let Some(next) = queues[idx].first().copied()
                && next.cost <= remaining
            {
                remaining -= next.cost;
                shown.push(next.name.clone());
                queues[idx].remove(0);
                if !queues[idx].is_empty() {
                    next_active.push(idx);
                }
            }
        }
        active = next_active;
    }
    shown
}

/// 生成供模型读取的 `codemode` 工具说明书：使用说明 + 全局变量 + 预算内的 TypeScript 工具声明目录。
pub fn create_codemode_description(callable: &[ToolRuntimeInfo], inline_budget: usize) -> String {
    let intro = "Run JavaScript that calls other tools. The input is raw JavaScript (not JSON, no code fence), run as an async function body in a QuickJS sandbox: top-level `await` and `return` work. No Node, file system, network, or timers.\n\
- `await tools.<name>({ ...args })` resolves to `structuredContent` when present, otherwise text, and rejects with an Error on failure.\n\
- Optional first line: `// @options: {\"timeoutMs\": 60000}`\n\n\
Globals:\n\
- `text(value)`, `console.log(...)`, and top-level `return` add output; `exit(msg?)` ends the script.\n\
- `store(key, value)` and `load(key)` keep JSON values across codemode calls.\n\
- `ALL_TOOLS`, `await searchTools(query, { limit?, namespace? })`, `await describeTool(name)`, `await describeNamespace(name)`: find unlisted tools, such as MCP tools.";

    let non_deferred: Vec<&ToolRuntimeInfo> = callable
        .iter()
        .filter(|t| {
            !matches!(
                t.exposure,
                ToolExposure::Deferred | ToolExposure::Hidden | ToolExposure::ModelOnly
            ) && t.name != CODEMODE_TOOL_NAME
        })
        .collect();
    if non_deferred.is_empty() {
        return intro.to_string();
    }

    struct NamespaceCatalogGroup {
        name: Option<String>,
        description: Option<String>,
        entries: Vec<CatalogEntry>,
    }

    let mut grouped: BTreeMap<String, NamespaceCatalogGroup> = BTreeMap::new();
    for info in non_deferred {
        let ns_key = info
            .namespace
            .as_ref()
            .map(|n| n.name.clone())
            .unwrap_or_default();
        let ns_desc = info.namespace.as_ref().and_then(|n| n.description.clone());
        let section = render_tool_section(info);
        let cost = section.len().div_ceil(CHARS_PER_TOKEN);
        grouped
            .entry(ns_key.clone())
            .or_insert_with(|| NamespaceCatalogGroup {
                name: (!ns_key.is_empty()).then_some(ns_key),
                description: ns_desc,
                entries: Vec::new(),
            })
            .entries
            .push(CatalogEntry {
                name: info.name.clone(),
                section,
                cost,
            });
    }

    let group_vec: Vec<(Option<String>, Vec<CatalogEntry>)> = grouped
        .values()
        .map(|g| {
            let clones = g
                .entries
                .iter()
                .map(|e| CatalogEntry {
                    name: e.name.clone(),
                    section: e.section.clone(),
                    cost: e.cost,
                })
                .collect();
            (g.name.clone(), clones)
        })
        .collect();

    let shown: HashSet<String> = select_catalog(&group_vec, inline_budget)
        .into_iter()
        .collect();

    let mut out = vec![intro.to_string(), "Nested tools:".to_string()];
    for g in grouped.values() {
        let visible: Vec<&CatalogEntry> = g
            .entries
            .iter()
            .filter(|e| shown.contains(&e.name))
            .collect();
        if let Some(ns) = &g.name {
            let listing = if visible.len() == g.entries.len() {
                ""
            } else if visible.is_empty() {
                " (tools not listed)"
            } else {
                " (some tools not listed)"
            };
            let mut header = format!("## {ns}{listing}");
            if let Some(d) = &g.description
                && !d.trim().is_empty()
            {
                header.push('\n');
                header.push_str(d.trim());
            }
            out.push(header);
        }
        for entry in visible {
            out.push(entry.section.clone());
        }
    }

    out.join("\n\n")
}

fn describe_script_call(tool: &ToolRuntimeInfo) -> String {
    let id = to_codemode_identifier(&tool.name);
    let out = describe_output(&tool.name, tool.output_schema.as_ref());
    format!(
        "{}\n\nCodemode: `tools.{id}(args)` resolves to {out}.",
        tool.description.trim()
    )
}

pub fn prepare_codemode_loadout(
    loadout: &ToolLoadout,
    hide_direct_in_only_mode: bool,
    inline_budget: usize,
) -> ToolLoadoutChanges {
    let callable: Vec<ToolRuntimeInfo> = loadout
        .callable
        .iter()
        .filter(|t| t.name != CODEMODE_TOOL_NAME)
        .cloned()
        .collect();
    let callable_names: HashSet<&str> = callable.iter().map(|t| t.name.as_str()).collect();

    let mut descriptions = HashMap::new();
    if !hide_direct_in_only_mode {
        for tool in &loadout.declared {
            if callable_names.contains(tool.name.as_str()) {
                descriptions.insert(tool.name.clone(), describe_script_call(tool));
            }
        }
    }

    let listed: Vec<ToolRuntimeInfo> = if hide_direct_in_only_mode {
        callable.clone()
    } else {
        callable
            .iter()
            .filter(|t| t.exposure != ToolExposure::Direct)
            .cloned()
            .collect()
    };

    descriptions.insert(
        CODEMODE_TOOL_NAME.to_string(),
        create_codemode_description(&listed, inline_budget),
    );

    let hidden_declarations = if hide_direct_in_only_mode {
        loadout
            .declared
            .iter()
            .filter(|t| t.exposure == ToolExposure::Direct && t.name != CODEMODE_TOOL_NAME)
            .map(|t| t.name.clone())
            .collect()
    } else {
        Vec::new()
    };

    ToolLoadoutChanges {
        descriptions,
        hidden_declarations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_tools::{ToolAnnotations, ToolExecutionMode, ToolNamespace};
    use serde_json::json;

    fn make_tool(name: &str, ns: Option<ToolNamespace>, exposure: ToolExposure) -> ToolRuntimeInfo {
        ToolRuntimeInfo {
            name: name.to_string(),
            description: format!("Tool {name}"),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "Search query" } },
                "required": ["query"]
            }),
            output_schema: None,
            exposure,
            execution_mode: ToolExecutionMode::Parallel,
            annotations: ToolAnnotations::default(),
            namespace: ns,
            default_active: None,
            prompt_snippet: None,
            prompt_guidelines: Vec::new(),
        }
    }

    #[test]
    fn renders_typescript_declarations_and_omits_namespace_instructions_from_catalog() {
        let ns = ToolNamespace {
            name: "mcp__fff".to_string(),
            aliases: vec!["fff".to_string()],
            description: None,
            instructions: Some("VERY LONG INSTRUCTIONS THAT MUST NOT LEAK".to_string()),
        };
        let t1 = make_tool("mcp__fff__grep", Some(ns), ToolExposure::Codemode);
        let desc = create_codemode_description(&[t1], DEFAULT_CODEMODE_INLINE_BUDGET);
        assert!(desc.contains("declare const tools: { mcp__fff__grep(args:"));
        assert!(!desc.contains("VERY LONG INSTRUCTIONS"));
        assert!(!desc.contains("\"properties\""));
    }

    #[test]
    fn prepare_loadout_annotates_declared_callable_tools_when_not_only_mode() {
        let direct = make_tool("read", None, ToolExposure::Direct);
        let cm = make_tool(
            "mcp__demo__ping",
            Some(ToolNamespace::new("mcp__demo")),
            ToolExposure::Codemode,
        );
        let loadout = ToolLoadout {
            declared: vec![direct.clone()],
            callable: vec![direct, cm],
            registered: Vec::new(),
        };
        let changes = prepare_codemode_loadout(&loadout, false, DEFAULT_CODEMODE_INLINE_BUDGET);
        assert!(
            changes.descriptions["read"]
                .contains("Codemode: `tools.read(args)` resolves to a string.")
        );
        assert!(!changes.descriptions["codemode"].contains("### `read`"));
        assert!(changes.descriptions["codemode"].contains("### `mcp__demo__ping`"));
    }
}
