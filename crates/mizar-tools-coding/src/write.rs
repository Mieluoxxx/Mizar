//! `write` 工具：创建或覆写文件，自动创建父目录，受 `with_file_mutation_queue` 串行写锁保护。

use std::path::PathBuf;

use mizar_tools::mutation_queue::with_file_mutation_queue;
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};

use crate::path_utils::resolve_to_cwd;

pub fn create_write_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "write",
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to write (relative or absolute)" },
                "content": { "type": "string", "description": "Complete content to write to the file" }
            },
            "required": ["path", "content"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let raw_path = args
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("缺少 path 参数")?;
                let content = args
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or("缺少 content 参数")?;

                if call_ctx.cancel.is_cancelled() {
                    return Err("Operation aborted".into());
                }

                let target = resolve_to_cwd(&cwd, raw_path, restrict_to_workspace)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;

                let content_owned = content.to_string();
                let bytes = content_owned.len();
                let lines = if content_owned.is_empty() {
                    0
                } else {
                    content_owned.lines().count()
                };

                with_file_mutation_queue(&target, || async {
                    if let Some(parent) = target.parent() {
                        tokio::fs::create_dir_all(parent).await?;
                    }
                    tokio::fs::write(&target, content_owned.as_bytes()).await
                })
                .await?;

                let meta = json!({
                    "path": raw_path,
                    "bytes": bytes,
                    "lines": lines,
                });
                Ok(ToolOutput::text(format!(
                    "Successfully wrote {bytes} bytes ({lines} lines) to {raw_path}"
                ))
                .with_structured(meta.clone())
                .with_details(meta))
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Sequential)
    .with_annotations(ToolAnnotations {
        read_only: false,
        destructive: true,
        idempotent: true,
        open_world: false,
    })
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "bytes": { "type": "integer" },
            "lines": { "type": "integer" }
        }
    }))
    .with_prompt_snippet("Create or overwrite files")
}
