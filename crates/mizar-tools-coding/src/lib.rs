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
use mizar_tools::{ToolDefinition, ToolRuntimeInfo, ToolsContextExt};

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

/// 根据当前会话已声明工具构建 `tools_guide` 系统提示词段落（纯英文静态模板）。
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

    if !usage.is_empty() {
        lines.push("Tool usage:".into());
        lines.extend(usage);
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
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

            if config.inject_system_prompt {
                let tools_ref = std::sync::Arc::clone(&tools);
                let prompt_fiber = ctx.try_inject(["systemPrompt"], move |c| {
                    let tools = std::sync::Arc::clone(&tools_ref);
                    async move {
                        let sp = c.service::<SystemPrompt>("systemPrompt")?;
                        c.effect("tools-coding.system-prompt", move || {
                            Ok::<_, cordis::CordisError>(Some(sp.section(
                                PromptSection::new_for_session(
                                    "coding_tools",
                                    30,
                                    move |session_id| {
                                        let declared = tools.list_declared_infos(session_id);
                                        build_tools_guide(&declared)
                                    },
                                ),
                            )))
                        })?;
                        Ok::<_, cordis::CordisError>(())
                    }
                })?;
                // 若 systemPrompt 已就绪则等待子 fiber 落稳，保证热启用返回后提示段立即可见；
                // 若尚未挂载（启动并发或无 SystemPrompt 单测）则不阻塞，待其到达后自动补注。
                if ctx.get::<SystemPrompt>("systemPrompt").is_some() {
                    prompt_fiber.wait_settled().await?;
                }
            }

            Ok(Some(Cleanup::async_fn(move || async move {
                for h in cleanups {
                    h.dispose().await;
                }
            })))
        })
    }
}
