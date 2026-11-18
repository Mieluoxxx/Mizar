//! `read` 工具：按行分页流式读取文件（支持 `offset` 与 `limit`，按 `2000 行 / 50KB` 截断并附带续读提示）。

use std::path::PathBuf;

use dsh_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncatedBy, format_size, truncate_head,
};
use dsh_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::path_utils::resolve_read_path;

pub fn create_read_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "read",
        format!(
            "Read the contents of a text file. Output is truncated to {} lines or {}KB (whichever is reached first). Use offset and limit to paginate large files.",
            DEFAULT_MAX_LINES,
            DEFAULT_MAX_BYTES / 1024
        ),
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
                "offset": { "type": "integer", "minimum": 1, "description": "1-indexed line number to start reading from" },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of lines to read" }
            },
            "required": ["path"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let raw_path = args
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("缺少 path 参数")?;
                let offset = args
                    .get("offset")
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v.max(1)).unwrap_or(usize::MAX))
                    .unwrap_or(1);
                let limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v.max(1)).unwrap_or(usize::MAX));

                if call_ctx.cancel.is_cancelled() {
                    return Err("Operation aborted".into());
                }

                let target = resolve_read_path(&cwd, raw_path, restrict_to_workspace)
                    .await
                    .map_err(|e| -> cordis::BoxError { e.into() })?;

                let meta = tokio::fs::metadata(&target).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        format!("File not found: {}", target.display())
                    } else {
                        e.to_string()
                    }
                })?;
                if meta.is_dir() {
                    return Err(format!("Path is a directory, not a file: {}", target.display()).into());
                }

                let file = tokio::fs::File::open(&target).await?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let mut total_lines = 0usize;
                let mut selected_lines: Vec<String> = Vec::new();
                let mut collected_bytes = 0usize;
                let max_collect_lines = limit.unwrap_or(DEFAULT_MAX_LINES).saturating_add(1);
                let byte_cap = DEFAULT_MAX_BYTES.saturating_add(4096);

                loop {
                    if call_ctx.cancel.is_cancelled() {
                        return Err("Operation aborted".into());
                    }
                    buf.clear();
                    let n = reader.read_until(b'\n', &mut buf).await?;
                    if n == 0 {
                        break;
                    }
                    if total_lines == 0 && buf.iter().take(512).any(|&b| b == 0) {
                        return Err(format!("Cannot read binary file as text: {}", target.display()).into());
                    }

                    total_lines += 1;
                    if total_lines < offset {
                        continue;
                    }
                    if selected_lines.len() >= max_collect_lines && collected_bytes > byte_cap {
                        continue;
                    }
                    if let Some(lim) = limit
                        && selected_lines.len() >= lim
                    {
                        continue;
                    }

                    let line_str = String::from_utf8(buf.clone())
                        .map_err(|_| format!("File is not valid UTF-8 text: {}", target.display()))?;
                    let line_no_lf = line_str.strip_suffix('\n').unwrap_or(&line_str);
                    let clean_line = line_no_lf.strip_suffix('\r').unwrap_or(line_no_lf);
                    if selected_lines.len() < DEFAULT_MAX_LINES.saturating_add(1)
                        && collected_bytes <= byte_cap
                    {
                        collected_bytes += clean_line.len() + 1;
                        selected_lines.push(clean_line.to_string());
                    }
                }

                if total_lines == 0 {
                    return Ok(ToolOutput::text("(empty file)")
                        .with_structured(json!({
                            "path": raw_path,
                            "start_line": 1,
                            "end_line": 0,
                            "total_lines": 0,
                            "truncated": false,
                        }))
                        .with_details(json!({
                            "path": raw_path,
                            "start_line": 1,
                            "end_line": 0,
                            "total_lines": 0,
                            "truncated": false,
                        })));
                }

                if offset > total_lines {
                    return Err(format!(
                        "Offset {offset} is beyond end of file ({total_lines} lines total)"
                    )
                    .into());
                }

                let selected_joined = selected_lines.join("\n");
                let truncation = truncate_head(&selected_joined, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);

                let (output_text, truncated, end_line, next_offset) =
                    if truncation.first_line_exceeds_limit {
                        let first_line_bytes = selected_lines.first().map(|l| l.len()).unwrap_or(0);
                        let msg = format!(
                            "[Line {} is {}, exceeding {} limit. Use bash (e.g. `sed -n '{}p' {} | head -c 4096`) to inspect it.]",
                            offset,
                            format_size(first_line_bytes),
                            format_size(DEFAULT_MAX_BYTES),
                            offset,
                            raw_path
                        );
                        (msg, true, offset, Some(offset + 1))
                    } else if truncation.truncated {
                        let shown = truncation.output_lines;
                        let end_line = offset + shown.saturating_sub(1);
                        let next = end_line + 1;
                        let mut text = truncation.content.clone();
                        match truncation.truncated_by {
                            Some(TruncatedBy::Bytes) => {
                                text.push_str(&format!(
                                    "\n\n[Showing lines {offset}-{end_line} of {total_lines} ({} limit). Use offset={next} to continue.]",
                                    format_size(DEFAULT_MAX_BYTES)
                                ));
                            }
                            _ => {
                                text.push_str(&format!(
                                    "\n\n[Showing lines {offset}-{end_line} of {total_lines}. Use offset={next} to continue.]"
                                ));
                            }
                        }
                        (text, true, end_line, Some(next))
                    } else if let Some(lim) = limit
                        && offset + selected_lines.len().saturating_sub(1) < total_lines
                    {
                        let end_line = offset + selected_lines.len().saturating_sub(1);
                        let next = end_line + 1;
                        let remaining = total_lines - end_line;
                        let mut text = truncation.content.clone();
                        text.push_str(&format!(
                            "\n\n[{remaining} more lines in file (limit={lim}). Use offset={next} to continue.]"
                        ));
                        (text, true, end_line, Some(next))
                    } else {
                        let end_line = offset + truncation.output_lines.saturating_sub(1);
                        (truncation.content.clone(), false, end_line, None)
                    };

                let meta_json = json!({
                    "path": raw_path,
                    "start_line": offset,
                    "end_line": end_line,
                    "total_lines": total_lines,
                    "truncated": truncated,
                    "next_offset": next_offset,
                });

                Ok(ToolOutput::text(output_text)
                    .with_structured(meta_json.clone())
                    .with_details(meta_json))
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
    .with_annotations(ToolAnnotations {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
    })
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "start_line": { "type": "integer" },
            "end_line": { "type": "integer" },
            "total_lines": { "type": "integer" },
            "truncated": { "type": "boolean" },
            "next_offset": { "type": ["integer", "null"] }
        }
    }))
    .with_prompt_snippet("Read file contents with optional line offset and limit")
    .with_prompt_guidelines(vec![
        "Use read to examine files instead of cat, head, or tail via bash.".into(),
    ])
}
