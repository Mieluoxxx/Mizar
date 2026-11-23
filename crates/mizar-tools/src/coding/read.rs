//! `read` 工具：按行分页流式读取文件（支持 `offset` 与 `limit`，输出 `<line>\t<text>` 行号，限制单行字符数与总输出上限）。

use std::path::PathBuf;

use crate::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, READ_MAX_LINE_CHARS, TruncatedBy, format_size,
    truncate_head, truncate_line,
};
use crate::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

use super::path_utils::resolve_read_path;

pub fn create_read_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "read",
        format!(
            "Read the contents of a text file with 1-indexed `<line>\\t<text>` line numbers. Output is truncated to {} lines, {}KB, or {} chars per line (whichever is reached first). Use offset and limit to paginate large files.",
            DEFAULT_MAX_LINES,
            DEFAULT_MAX_BYTES / 1024,
            READ_MAX_LINE_CHARS
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
                    .ok_or("Missing required parameter: path")?;
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
                    .map_err(|e| -> mizar_types::BoxError { e.into() })?;

                let meta = tokio::fs::metadata(&target).await.map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        format!("File not found: {}", target.display())
                    } else {
                        e.to_string()
                    }
                })?;
                if meta.is_dir() {
                    return Err(
                        format!("Path is a directory, not a file: {}", target.display()).into(),
                    );
                }

                let file = tokio::fs::File::open(&target).await?;
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                let mut total_lines = 0usize;
                let mut selected_lines: Vec<String> = Vec::new();
                let mut collected_bytes = 0usize;
                let mut any_line_truncated = false;
                let max_collect_lines = limit
                    .unwrap_or(DEFAULT_MAX_LINES)
                    .min(DEFAULT_MAX_LINES)
                    .saturating_add(1);
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
                        return Err(format!(
                            "Cannot read binary file as text: {}",
                            target.display()
                        )
                        .into());
                    }

                    total_lines += 1;
                    if total_lines < offset {
                        continue;
                    }
                    // 达到行数或字节收集上限后仅继续统计总行数，避免超长行或大文件继续占内存。
                    if selected_lines.len() >= max_collect_lines || collected_bytes > byte_cap {
                        if buf.capacity() > 64 * 1024 {
                            buf.shrink_to(4096);
                        }
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
                    let (short_line, was_line_trunc) =
                        truncate_line(clean_line, READ_MAX_LINE_CHARS);
                    if was_line_trunc {
                        any_line_truncated = true;
                    }

                    let numbered = format!("{total_lines}\t{short_line}");
                    collected_bytes += numbered.len() + 1;
                    selected_lines.push(numbered);
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
                let truncation =
                    truncate_head(&selected_joined, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);

                let (mut output_text, truncated, end_line, next_offset) =
                    if truncation.first_line_exceeds_limit {
                        let first_line_bytes =
                            selected_lines.first().map(|l| l.len()).unwrap_or(0);
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
                            "\n\n[Showing lines {offset}-{end_line} of {total_lines} ({remaining} more lines, limit={lim}). Use offset={next} to continue.]"
                        ));
                        (text, true, end_line, Some(next))
                    } else {
                        let end_line = offset + truncation.output_lines.saturating_sub(1);
                        (truncation.content.clone(), false, end_line, None)
                    };

                if any_line_truncated {
                    output_text.push_str(&format!(
                        "\n[Some lines exceeded {READ_MAX_LINE_CHARS} chars and were truncated.]"
                    ));
                }

                let truncated = truncated || any_line_truncated;
                let meta_json = json!({
                    "path": raw_path,
                    "start_line": offset,
                    "end_line": end_line,
                    "total_lines": total_lines,
                    "truncated": truncated,
                    "line_truncated": any_line_truncated,
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
            "line_truncated": { "type": "boolean" },
            "next_offset": { "type": ["integer", "null"] }
        }
    }))
    .with_prompt_snippet("Read file contents with line numbers, optional offset and limit")
    .with_prompt_guidelines(vec![
        "Use read to examine files instead of cat, head, or tail via bash.".into(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolCallContext, ToolExecution};

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mizar_w13_read_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call_ctx(args: &Value) -> ToolCallContext {
        ToolCallContext {
            exec: ToolExecution::new("read", args.clone(), None),
            cancel: tokio_util::sync::CancellationToken::new(),
            on_update: None,
            tools: crate::ToolRuntime::new(),
        }
    }

    #[tokio::test]
    async fn formats_line_numbers_and_paginates_with_offset_limit() {
        let root = test_dir("paginate");
        std::fs::write(root.join("sample.txt"), "alpha\nbeta\ngamma\ndelta\n").unwrap();

        let tool = create_read_tool(root.clone(), true);
        let args = json!({ "path": "sample.txt", "offset": 2, "limit": 2 });
        let out = (tool.execute)(args.clone(), call_ctx(&args)).await.unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains("2\tbeta\n3\tgamma"),
            "got:\n{}",
            out.content
        );
        assert!(out.content.contains("Use offset=4 to continue."));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn truncates_overlong_single_line_safely_including_utf8() {
        let root = test_dir("long_line");
        let long_utf8 = format!("{}END", "中".repeat(2500));
        std::fs::write(root.join("min.txt"), format!("{long_utf8}\nsecond\n")).unwrap();

        let tool = create_read_tool(root.clone(), true);
        let args = json!({ "path": "min.txt" });
        let out = (tool.execute)(args.clone(), call_ctx(&args)).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.starts_with("1\t"));
        assert!(out.content.contains("... [truncated]"));
        assert!(out.content.contains("2\tsecond"));
        assert!(out.content.contains("exceeded 2000 chars"));
        assert!(!out.content.contains("END"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn stops_collecting_early_past_max_lines_and_reports_total_lines() {
        let root = test_dir("many_lines");
        let content = (1..=2205)
            .map(|i| format!("row-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(root.join("big.txt"), content).unwrap();

        let tool = create_read_tool(root.clone(), true);
        let args = json!({ "path": "big.txt" });
        let out = (tool.execute)(args.clone(), call_ctx(&args)).await.unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1\trow-1"));
        assert!(out.content.contains("2000\trow-2000"));
        assert!(!out.content.contains("2001\trow-2001"));
        assert!(
            out.content
                .contains("Showing lines 1-2000 of 2205. Use offset=2001 to continue.")
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn reports_english_errors_for_missing_param_offset_and_binary() {
        let root = test_dir("errors");
        std::fs::write(root.join("tiny.txt"), "one\n").unwrap();
        std::fs::write(root.join("bin.dat"), b"\x00\x01\x02").unwrap();

        let tool = create_read_tool(root.clone(), true);
        let missing = (tool.execute)(json!({}), call_ctx(&json!({})))
            .await
            .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Missing required parameter: path")
        );

        let beyond_args = json!({ "path": "tiny.txt", "offset": 5 });
        let beyond = (tool.execute)(beyond_args.clone(), call_ctx(&beyond_args))
            .await
            .unwrap_err();
        assert!(beyond.to_string().contains("beyond end of file"));

        let bin_args = json!({ "path": "bin.dat" });
        let binary = (tool.execute)(bin_args.clone(), call_ctx(&bin_args))
            .await
            .unwrap_err();
        assert!(binary.to_string().contains("binary file"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
