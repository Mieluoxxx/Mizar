//! `write` 工具：创建或覆写文件，自动创建父目录，并发调用受 `with_file_mutation_queue` 同路径写锁保护。

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
                    .ok_or("Missing required parameter: path")?;
                let content = args
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or("Missing required parameter: content")?;

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
    .with_execution_mode(ToolExecutionMode::Parallel)
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

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_tools::{ToolCallContext, ToolExecution};

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mizar_w13_write_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call_ctx(args: &Value) -> ToolCallContext {
        ToolCallContext {
            exec: ToolExecution::new("write", args.clone(), None),
            cancel: tokio_util::sync::CancellationToken::new(),
            on_update: None,
            ctx: cordis::Context::new(),
        }
    }

    #[tokio::test]
    async fn parallel_mode_creates_parents_and_serializes_same_path() {
        let root = test_dir("parallel");
        let tool = create_write_tool(root.clone(), true);
        assert_eq!(tool.execution_mode, ToolExecutionMode::Parallel);

        let missing_args = json!({ "path": "a.txt" });
        let missing = (tool.execute)(missing_args.clone(), call_ctx(&missing_args))
            .await
            .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Missing required parameter: content")
        );

        let write_args = json!({ "path": "nested/dir/file.txt", "content": "hello\nworld\n" });
        let out = (tool.execute)(write_args.clone(), call_ctx(&write_args))
            .await
            .unwrap();
        assert!(!out.is_error);
        assert_eq!(
            std::fs::read_to_string(root.join("nested/dir/file.txt")).unwrap(),
            "hello\nworld\n"
        );

        let args1 = json!({ "path": "a.txt", "content": "first\n" });
        let args2 = json!({ "path": "b.txt", "content": "second\n" });
        let (r1, r2) = tokio::join!(
            (tool.execute)(args1.clone(), call_ctx(&args1)),
            (tool.execute)(args2.clone(), call_ctx(&args2)),
        );
        assert!(!r1.unwrap().is_error);
        assert!(!r2.unwrap().is_error);
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "first\n"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("b.txt")).unwrap(),
            "second\n"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
