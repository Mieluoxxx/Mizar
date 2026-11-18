//! Codemode JS 标识符转换、工具目录渲染与 `prepare_loadout` 预算筛选（对齐 `pi` `codemode/tool.ts`）。

use std::collections::{BTreeMap, HashMap};

use dsh_tools::{ToolExposure, ToolLoadout, ToolLoadoutChanges, ToolRuntimeInfo};

pub const CODEMODE_TOOL_NAME: &str = "codemode";
pub const DEFAULT_CODEMODE_INLINE_BUDGET: usize = 6000;
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
    let mut desc = info.description.trim().to_string();
    for g in &info.prompt_guidelines {
        let trimmed = g.trim();
        if !trimmed.is_empty() {
            desc.push_str(&format!("\n- {trimmed}"));
        }
    }
    let params = serde_json::to_string(&info.parameters).unwrap_or_else(|_| "{}".into());
    let output_note = match &info.output_schema {
        Some(s) => serde_json::to_string(s).unwrap_or_else(|_| "\"any\"".into()),
        None => "string".to_string(),
    };
    format!("{heading}\n{desc}\nCall: `await tools.{id}(args)`\nInput: `{params}`\nOutput: `{output_note}`")
}

struct CatalogEntry {
    name: String,
    section: String,
    cost: usize,
}

/// 轮询每个命名空间，保证各个命名空间都能按预算公平展示至少一部分工具。
fn select_catalog(groups: &[(Option<String>, Vec<CatalogEntry>)], budget_tokens: usize) -> Vec<String> {
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
    let mut active: Vec<usize> = (0..queues.len()).filter(|&i| !queues[i].is_empty()).collect();

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

pub fn create_codemode_description(callable: &[ToolRuntimeInfo], inline_budget: usize) -> String {
    let intro = "Run JavaScript inside a sandboxed QuickJS runtime to orchestrate one or more tools in a single turn.\n\
Only your script's returned value / `text()` / `console.log()` reaches the model; intermediate nested tool results stay out of context.\n\n\
Globals:\n\
- `tools.<name>(args)`: async call to any callable tool (resolves to `structuredContent` if present, otherwise text; throws on error)\n\
- `ALL_TOOLS`: array of declared callable tool metadata\n\
- `await searchTools(query, { limit?, namespace? })`: find deferred/codemode tools via BM25\n\
- `await describeTool(name)` / `await describeNamespace(name)`: inspect full JSON schema\n\
- `store(key, value)` / `load(key)`: persist JSON state across scripts in this session\n\
- `text(str)` / `console.log(...)` / `return value` / `exit(msg?)`";

    let non_deferred: Vec<&ToolRuntimeInfo> = callable
        .iter()
        .filter(|t| t.exposure != ToolExposure::Deferred && t.name != CODEMODE_TOOL_NAME)
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
        let ns_desc = info
            .namespace
            .as_ref()
            .and_then(|n| n.description.clone());
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

    let shown: std::collections::HashSet<String> =
        select_catalog(&group_vec, inline_budget).into_iter().collect();

    let mut out = vec![intro.to_string(), "Nested tools:".to_string()];
    for g in grouped.values() {
        let visible: Vec<&CatalogEntry> =
            g.entries.iter().filter(|e| shown.contains(&e.name)).collect();
        if let Some(ns) = &g.name {
            let listing = if visible.len() == g.entries.len() {
                ""
            } else if visible.is_empty() {
                " (tools not listed; use searchTools())"
            } else {
                " (some tools not listed; use searchTools())"
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

    let listed: Vec<ToolRuntimeInfo> = if hide_direct_in_only_mode {
        callable.clone()
    } else {
        callable
            .iter()
            .filter(|t| t.exposure != ToolExposure::Direct)
            .cloned()
            .collect()
    };

    let mut descriptions = HashMap::new();
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
