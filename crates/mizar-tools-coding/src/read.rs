//! `read` 工具：按行分页流式读取文件（支持 `offset` 与 `limit`，输出 `<line>\t<text>` 行号，限制单行字符数与总输出上限）。

use std::path::PathBuf;

use mizar_image::{MAX_IMAGE_INPUT_BYTES, process_image_bytes, sniff_image_format};
use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, READ_MAX_LINE_CHARS, TruncatedBy, format_size,
    truncate_head, truncate_line,
};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::path_utils::resolve_read_path;

async fn read_image(
    reader: BufReader<tokio::fs::File>,
    path: &str,
    file_size: u64,
) -> Result<ToolOutput, cordis::BoxError> {
    let size_error = || format!("Image exceeds the 32 MiB input limit: {path}");
    if file_size > MAX_IMAGE_INPUT_BYTES as u64 {
        return Err(size_error().into());
    }
    let mut bytes = Vec::new();
    reader
        .take(MAX_IMAGE_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > MAX_IMAGE_INPUT_BYTES {
        return Err(size_error().into());
    }
    let image = tokio::task::spawn_blocking(move || process_image_bytes(&bytes)).await??;
    let mut text = format!(
        "{path}: {}×{} ({})",
        image.original_width, image.original_height, image.attachment.media_type
    );
    if let Some(changes) = image.marker_suffix() {
        text.push_str(&format!(" · {changes}"));
    }
    let metadata = json!({
        "path": path,
        "media_type": image.attachment.media_type,
        "width": image.attachment.width,
        "height": image.attachment.height,
        "original_width": image.original_width,
        "original_height": image.original_height,
    });
    Ok(ToolOutput::text(text)
        .with_structured(metadata)
        .with_images(vec![image.attachment]))
}

pub fn create_read_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "read",
        format!(
            "Read a PNG, JPEG, WebP or GIF image, or a text file with 1-indexed `<line>\\t<text>` line numbers. Text output is truncated to {} lines, {}KB, or {} chars per line (whichever is reached first). Use offset and limit to paginate text files.",
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
                    .map_err(|e| -> cordis::BoxError { e.into() })?;

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
                if sniff_image_format(reader.fill_buf().await?).is_some() {
                    return read_image(reader, raw_path, meta.len()).await;
                }
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
            "media_type": { "type": "string" },
            "width": { "type": "integer" },
            "height": { "type": "integer" },
            "original_width": { "type": "integer" },
            "original_height": { "type": "integer" },
            "start_line": { "type": "integer" },
            "end_line": { "type": "integer" },
            "total_lines": { "type": "integer" },
            "truncated": { "type": "boolean" },
            "line_truncated": { "type": "boolean" },
            "next_offset": { "type": ["integer", "null"] }
        }
    }))
    .with_prompt_snippet("Read images or inspect text files with line numbers, offset and limit")
    .with_prompt_guidelines(vec!["Use read to examine files instead of cat or sed.".into()])
}
