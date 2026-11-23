//! 核心编码工具套件：`read` / `write` / `edit` / `bash` / `grep` / `find`。

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod find;
pub mod grep;
pub mod path_utils;
pub mod read;
pub mod write;

use std::path::PathBuf;
use std::sync::Arc;

use mizar_types::hooks::HookGuard;

use crate::{ToolDefinition, ToolError, ToolRuntime, ToolRuntimeInfo};

pub use bash::create_bash_tool;
pub use edit::create_edit_tool;
pub use find::create_find_tool;
pub use grep::create_grep_tool;
pub use read::create_read_tool;
pub use write::create_write_tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodingToolsPreset {
    /// 全部 6 个内置编码工具：`read`、`write`、`edit`、`bash`、`grep`、`find`。
    #[default]
    All,
    /// `pi` 极简默认 4 件套：`read`、`bash`、`edit`、`write`。
    MinimalCoding,
    /// 纯只读套件：`read`、`grep`、`find`。
    ReadOnly,
}

#[derive(Debug, Clone)]
pub struct CodingToolsConfig {
    pub workspace: PathBuf,
    /// 是否将文件工具（read/write/edit/grep/find）限制在 workspace 目录内部（默认 false，不设审批流）。
    pub restrict_to_workspace: bool,
    pub preset: CodingToolsPreset,
    /// 是否向 `SystemPrompt` 贡献可用工具与工作目录段落（由上层 `mizar-core::spine` 消费）。
    pub inject_system_prompt: bool,
}

impl CodingToolsConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            restrict_to_workspace: false,
            preset: CodingToolsPreset::All,
            inject_system_prompt: true,
        }
    }
}

pub fn build_coding_tools(config: &CodingToolsConfig) -> Vec<ToolDefinition> {
    let cwd = config.workspace.clone();
    let restrict = config.restrict_to_workspace;
    match config.preset {
        CodingToolsPreset::All => vec![
            create_read_tool(cwd.clone(), restrict),
            create_bash_tool(cwd.clone()),
            create_edit_tool(cwd.clone(), restrict),
            create_write_tool(cwd.clone(), restrict),
            create_grep_tool(cwd.clone(), restrict),
            create_find_tool(cwd, restrict),
        ],
        CodingToolsPreset::MinimalCoding => vec![
            create_read_tool(cwd.clone(), restrict),
            create_bash_tool(cwd.clone()),
            create_edit_tool(cwd.clone(), restrict),
            create_write_tool(cwd, restrict),
        ],
        CodingToolsPreset::ReadOnly => vec![
            create_read_tool(cwd.clone(), restrict),
            create_grep_tool(cwd.clone(), restrict),
            create_find_tool(cwd, restrict),
        ],
    }
}

/// 根据当前会话已声明工具构建 `tools_guide` 系统提示词段落（纯英文，供 `mizar-core` 注册）。
pub fn build_tools_guide(declared: &[ToolRuntimeInfo]) -> Option<String> {
    if declared.is_empty() {
        return None;
    }
    let has = |name: &str| declared.iter().any(|info| info.name == name);
    let mut lines = Vec::new();
    let mut snippets = Vec::new();
    let mut extra_guidelines = Vec::new();
    for info in declared {
        if let Some(s) = &info.prompt_snippet {
            snippets.push(format!("- `{}`: {s}", info.name));
        }
        if !matches!(info.name.as_str(), "read" | "edit") {
            for g in &info.prompt_guidelines {
                if !extra_guidelines.contains(g) {
                    extra_guidelines.push(g.clone());
                }
            }
        }
    }
    if !snippets.is_empty() {
        lines.push("Available tools:".into());
        lines.extend(snippets);
    }

    let mut usage = Vec::new();
    if has("bash") {
        let mut dedicated = Vec::new();
        if has("read") {
            dedicated.push("`read` to inspect files");
        }
        if has("grep") {
            dedicated.push("`grep` to search content");
        }
        if has("find") {
            dedicated.push("`find` to locate files");
        }
        if has("edit") {
            dedicated
                .push("`edit` for targeted edits (batch disjoint edits to one file in `edits[]`)");
        }
        if !dedicated.is_empty() {
            usage.push(format!(
                "- Prefer dedicated tools over `bash`: use {}; never run `cat`, `head`, `tail`, `sed`, `grep`, or `find` via `bash`.",
                dedicated.join(", ")
            ));
        }
    } else if has("edit") {
        usage.push(
            "- Use `edit` for targeted changes to existing files (batch disjoint edits to one file in `edits[]`) instead of rewriting whole files."
                .into(),
        );
    }
    for g in extra_guidelines {
        usage.push(format!("- {g}"));
    }
    if has("edit") || has("write") || has("bash") {
        usage.push(
            "- Batch independent tool calls in one turn: issue parallel `read`, `grep`, `find`, cross-file `edit`/`write`, and independent `bash` calls together when they have no data dependency. Run calls sequentially only when a later step depends on an earlier result."
                .into(),
        );
    } else {
        usage.push(
            "- Batch independent read-only tool calls (`read`, `grep`, `find`) in one turn when they have no data dependency. Run calls sequentially only when a later step depends on an earlier result."
                .into(),
        );
    }

    if !usage.is_empty() {
        lines.push("Tool usage:".into());
        lines.extend(usage);
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

pub fn register_coding_tools(
    tools: &Arc<ToolRuntime>,
    config: &CodingToolsConfig,
) -> Result<Vec<HookGuard>, ToolError> {
    let defs = build_coding_tools(config);
    let mut guards = Vec::with_capacity(defs.len());
    for def in defs {
        guards.push(tools.register(def)?);
    }
    Ok(guards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolExecution, ToolRuntime, ToolUpdate};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn exercises_all_six_coding_tools_end_to_end() {
        let root =
            std::env::temp_dir().join(format!("mizar_coding_tools_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.log\n").unwrap();
        std::fs::write(root.join("ignored.log"), "secret_needle\n").unwrap();

        let tools = ToolRuntime::new();
        let _guards = register_coding_tools(&tools, &CodingToolsConfig::new(&root)).unwrap();

        // 1. write
        let write_res = tools
            .execute(ToolExecution::new(
                "write",
                json!({
                    "path": "src/demo.rs",
                    "content": "fn main() {\n    let x = \"needle_one\";\n    println!(\"{}\", x);\n}\n"
                }),
                None,
            ))
            .await;
        assert!(!write_res.is_error, "write err: {}", write_res.content);

        // 2. read (with offset + limit)
        let read_res = tools
            .execute(ToolExecution::new(
                "read",
                json!({ "path": "src/demo.rs", "offset": 2, "limit": 2 }),
                None,
            ))
            .await;
        assert!(!read_res.is_error, "read err: {}", read_res.content);
        assert!(read_res.content.contains("needle_one"));
        assert!(read_res.content.contains("Use offset=4 to continue"));

        // 3. edit (with fuzzy smart quotes + unified diff details)
        let edit_res = tools
            .execute(ToolExecution::new(
                "edit",
                json!({
                    "path": "src/demo.rs",
                    "edits": [{
                        "oldText": "    let x = “needle_one”;",
                        "newText": "    let x = \"needle_two\";"
                    }]
                }),
                None,
            ))
            .await;
        assert!(!edit_res.is_error, "edit err: {}", edit_res.content);
        assert!(
            edit_res
                .details
                .as_ref()
                .and_then(|d| d.get("diff"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("+    let x = \"needle_two\";")
        );

        // 4. grep (using rg, respects .gitignore)
        let grep_res = tools
            .execute(ToolExecution::new(
                "grep",
                json!({ "pattern": "needle_", "literal": true }),
                None,
            ))
            .await;
        assert!(!grep_res.is_error, "grep err: {}", grep_res.content);
        assert!(grep_res.content.contains("src/demo.rs:2:"));
        assert!(!grep_res.content.contains("ignored.log"));

        // 5. find (fd or ignore+globset fallback, respects .gitignore)
        let find_res = tools
            .execute(ToolExecution::new(
                "find",
                json!({ "pattern": "*.rs" }),
                None,
            ))
            .await;
        assert!(!find_res.is_error, "find err: {}", find_res.content);
        assert!(find_res.content.contains("src/demo.rs"));

        // 6. bash (streams update + truncates/spills huge output)
        let updates = Arc::new(Mutex::new(Vec::<String>::new()));
        let updates_cb = Arc::clone(&updates);
        let bash_res = tools
            .execute_with(
                ToolExecution::new(
                    "bash",
                    json!({ "command": "echo hi_bash; seq 1 2100" }),
                    None,
                ),
                CancellationToken::new(),
                Some(Arc::new(move |u: ToolUpdate| {
                    updates_cb.lock().unwrap().push(u.preview);
                })),
            )
            .await;
        assert!(!bash_res.is_error, "bash err: {}", bash_res.content);
        assert!(bash_res.content.contains("Full output saved to:"));
        assert!(bash_res.content.contains("2100"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn build_tools_guide_renders_pure_english_and_respects_session_filtering() {
        let root = std::env::temp_dir().join(format!(
            "mizar_coding_tools_guide_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let tools = ToolRuntime::new();
        let _guards = register_coding_tools(&tools, &CodingToolsConfig::new(&root)).unwrap();
        let built = build_tools_guide(&tools.list_declared_infos(None)).unwrap();
        assert!(built.contains("Available tools:"), "prompt:\n{built}");
        assert!(built.contains("Tool usage:"), "prompt:\n{built}");
        assert!(built.contains("edits[]"), "prompt:\n{built}");
        assert!(
            built.contains("never run `cat`, `head`, `tail`, `sed`, `grep`, or `find` via `bash`"),
            "prompt:\n{built}"
        );
        assert!(
            built.contains("Batch independent tool calls in one turn"),
            "prompt:\n{built}"
        );
        assert!(
            !built
                .chars()
                .any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)),
            "tools_guide must be pure English:\n{built}"
        );

        tools.set_active_tools("s-ro", &["read".into()]);
        let built_ro = build_tools_guide(&tools.list_declared_infos(Some("s-ro"))).unwrap();
        assert!(built_ro.contains("- `read`:"), "s-ro:\n{built_ro}");
        assert!(!built_ro.contains("- `bash`:"), "s-ro:\n{built_ro}");

        let _ = std::fs::remove_dir_all(&root);
    }
}
