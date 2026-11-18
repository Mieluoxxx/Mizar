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

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_system_prompt::{PromptSection, SystemPrompt};
use mizar_tools::{ToolDefinition, ToolsContextExt};

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
    /// 是否向 `SystemPrompt` 贡献可用工具与工作目录段落。
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

#[derive(Default, Clone, Copy)]
pub struct CodingToolsPlugin;

impl Plugin for CodingToolsPlugin {
    type Config = CodingToolsConfig;

    fn name(&self) -> Option<&str> {
        Some("tools-coding")
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
            let defs = build_coding_tools(&config);
            let mut cleanups = Vec::with_capacity(defs.len());
            for def in defs {
                let key = format!("tools-coding.{}", def.name);
                let rt = std::sync::Arc::clone(&tools);
                let handle = ctx.effect(key, move || {
                    Ok::<_, mizar_tools::ToolError>(Some(rt.register(def)?))
                })?;
                cleanups.push(handle);
            }

            if config.inject_system_prompt
                && let Some(sp) = ctx.get::<SystemPrompt>("systemPrompt")
            {
                let tools_ref = ctx.tools()?;
                let cwd_display = config.workspace.to_string_lossy().replace('\\', "/");
                let sec_handle = ctx.effect("tools-coding.system-prompt", move || {
                    let tools = tools_ref;
                    let cwd = cwd_display;
                    Ok::<_, cordis::CordisError>(Some(sp.section(PromptSection::new(
                        "coding_tools",
                        20,
                        move || {
                            let declared = tools.list_declared_infos(None);
                            let mut lines = vec![format!("当前工作目录：{cwd}")];
                            let mut snippets = Vec::new();
                            let mut guidelines = Vec::new();
                            for info in &declared {
                                if let Some(s) = &info.prompt_snippet {
                                    snippets.push(format!("- {}: {}", info.name, s));
                                }
                                for g in &info.prompt_guidelines {
                                    if !guidelines.contains(g) {
                                        guidelines.push(g.clone());
                                    }
                                }
                            }
                            if !snippets.is_empty() {
                                lines.push("核心工具能力：".into());
                                lines.extend(snippets);
                            }
                            if !guidelines.is_empty() {
                                lines.push("操作原则：".into());
                                lines.extend(guidelines.into_iter().map(|g| format!("- {g}")));
                            }
                            Some(lines.join("\n"))
                        },
                    ))))
                })?;
                cleanups.push(sec_handle);
            }

            Ok(Some(Cleanup::async_fn(move || async move {
                for h in cleanups {
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
    use mizar_tools::{ToolExecution, ToolRuntime, ToolUpdate};
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn exercises_all_six_coding_tools_end_to_end() {
        let root = std::env::temp_dir().join(format!("mizar_coding_tools_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.log\n").unwrap();
        std::fs::write(root.join("ignored.log"), "secret_needle\n").unwrap();

        let app = Context::new();
        app.plugin(ToolRuntime::plugin(), ()).await.unwrap();
        app.plugin(CodingToolsPlugin, CodingToolsConfig::new(&root))
            .await
            .unwrap();

        let tools = app.tools().unwrap();

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
        app.fiber().dispose().await;
    }
}
