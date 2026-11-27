//! 核心编码工具套件：`read` / `write` / `edit` / `bash` / `grep` / `find`。

pub mod bash;
pub mod edit;
pub mod edit_diff;
pub mod find;
pub mod grep;
pub mod managed_bins;
pub mod path_utils;
pub mod read;
pub mod write;

use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;

use cordis::{BoxError, BoxFuture, Cleanup, Context, Inject, Plugin};
use mizar_system_prompt::{PromptSection, SystemPrompt};
use mizar_tools::{ToolDefinition, ToolRuntimeInfo, ToolsContextExt};

pub use bash::create_bash_tool;
pub use edit::create_edit_tool;
pub use find::create_find_tool;
pub use grep::create_grep_tool;
pub use managed_bins::{EnsureBinFn, mizar_bin_dir, resolve_managed_bin};
pub use read::create_read_tool;
pub use write::create_write_tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodingToolsPreset {
    /// 全部 6 个内置编码工具：`read`、`write`、`edit`、`bash`、`grep`、`find`。
    All,
    /// `pi` 极简默认 4 件套：`read`、`bash`、`edit`、`write`。
    #[default]
    MinimalCoding,
    /// 纯只读套件：`read`、`grep`、`find`。
    ReadOnly,
}

#[derive(Clone)]
pub struct CodingToolsConfig {
    pub workspace: PathBuf,
    /// 是否将文件工具（read/write/edit/grep/find）限制在 workspace 目录内部（默认 false，不设审批流）。
    pub restrict_to_workspace: bool,
    pub preset: CodingToolsPreset,
    /// 是否向 `SystemPrompt` 贡献可用工具与工作目录段落。
    pub inject_system_prompt: bool,
    pub ensure_bin: Option<EnsureBinFn>,
}

impl fmt::Debug for CodingToolsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodingToolsConfig")
            .field("workspace", &self.workspace)
            .field("restrict_to_workspace", &self.restrict_to_workspace)
            .field("preset", &self.preset)
            .field("inject_system_prompt", &self.inject_system_prompt)
            .field("ensure_bin", &self.ensure_bin.is_some())
            .finish()
    }
}

impl CodingToolsConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            restrict_to_workspace: false,
            preset: CodingToolsPreset::default(),
            inject_system_prompt: true,
            ensure_bin: None,
        }
    }
}

pub fn build_coding_tools(config: &CodingToolsConfig) -> Vec<ToolDefinition> {
    let cwd = config.workspace.clone();
    let restrict = config.restrict_to_workspace;
    let mut tools = vec![create_read_tool(cwd.clone(), restrict)];
    if config.preset != CodingToolsPreset::ReadOnly {
        tools.extend([
            create_bash_tool(cwd.clone()),
            create_edit_tool(cwd.clone(), restrict),
            create_write_tool(cwd.clone(), restrict),
        ]);
    }
    let search_active = config.preset != CodingToolsPreset::MinimalCoding;
    tools.push(
        create_grep_tool(cwd.clone(), restrict, config.ensure_bin.clone())
            .with_default_active(search_active),
    );
    tools.push(create_find_tool(cwd, restrict).with_default_active(search_active));
    tools
}

/// 根据当前会话已声明工具构建 `tools_guide` 系统提示词段落（纯英文静态模板）。
pub fn build_tools_guide(declared: &[ToolRuntimeInfo]) -> String {
    let has = |name: &str| declared.iter().any(|info| info.name == name);
    let mut lines = Vec::new();
    if !declared.is_empty() {
        lines.push("<tools>".into());
        for info in declared {
            match info
                .prompt_snippet
                .as_deref()
                .and_then(|s| s.lines().next())
            {
                Some(snippet) => lines.push(format!("- `{}`: {snippet}", info.name)),
                None => lines.push(format!("- `{}`", info.name)),
            }
        }
        lines.push("</tools>\n".into());
    }

    lines.push("<rules>".into());
    let mut seen = HashSet::new();
    let mut add_rule = |rule: &str| {
        if seen.insert(rule.to_owned()) {
            lines.push(format!("- {rule}"));
        }
    };
    if has("bash") {
        let searches: Vec<_> = ["grep", "find"]
            .into_iter()
            .filter(|name| has(name))
            .collect();
        if searches.is_empty() {
            add_rule("Use bash for file operations like ls, rg, find.");
        } else {
            add_rule(&format!(
                "You may use {} for standalone searches, or bash for pipelines and compound commands.",
                searches.join("/")
            ));
        }
    }
    for rule in declared.iter().flat_map(|tool| &tool.prompt_guidelines) {
        add_rule(rule);
    }
    add_rule("Maximize parallel tool calls for independent operations.");
    add_rule("Be concise in your responses.");
    add_rule("Show file paths clearly when working with files.");
    lines.push("</rules>".into());
    lines.join("\n")
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
                                        Some(build_tools_guide(&declared))
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
