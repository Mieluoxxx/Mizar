//! `grep` 工具：基于本地 `rg` (`ripgrep --json`) 执行正则/字面量搜索，遵守 `.gitignore`，截断长行与输出。

use std::path::{Path, PathBuf};
use std::process::Stdio;

use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, format_size, truncate_head, truncate_line,
};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;

use crate::path_utils::resolve_to_cwd;

const DEFAULT_LIMIT: usize = 100;

fn decode_rg_field(field: Option<&Value>) -> Option<String> {
    let obj = field?.as_object()?;
    if let Some(text) = obj.get("text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if let Some(b64) = obj.get("bytes").and_then(Value::as_str)
        && let Some(bytes) = decode_base64(b64)
    {
        return Some(String::from_utf8_lossy(&bytes).into_owned());
    }
    None
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
    fn val(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    for chunk in clean.as_chunks::<4>().0 {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            let c = val(chunk[2])?;
            out.push((b << 4) | (c >> 2));
            if chunk[3] != b'=' {
                let d = val(chunk[3])?;
                out.push((c << 6) | d);
            }
        }
    }
    Some(out)
}

fn format_rel_path(raw: &str, search_path: &Path, is_dir: bool) -> String {
    if is_dir {
        let joined = search_path.join(raw);
        if let Ok(rel) = joined.strip_prefix(search_path) {
            let s = rel.to_string_lossy().replace('\\', "/");
            if !s.is_empty() && !s.starts_with("..") {
                return s;
            }
        }
    }
    Path::new(raw)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string())
}

pub fn create_grep_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "grep",
        format!(
            "Search file contents for a pattern using ripgrep. Respects .gitignore. Output is truncated to {} matches or {}KB. Long lines are truncated to {} chars.",
            DEFAULT_LIMIT,
            DEFAULT_MAX_BYTES / 1024,
            GREP_MAX_LINE_LENGTH
        ),
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Search pattern (regex by default, or literal when literal=true)" },
                "path": { "type": "string", "description": "Directory or file to search (defaults to workspace root)" },
                "glob": { "type": "string", "description": "Filter files by glob pattern, e.g. '*.rs' or '**/*.ts'" },
                "ignoreCase": { "type": "boolean", "description": "Case-insensitive search (default: false)" },
                "literal": { "type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)" },
                "context": { "type": "integer", "minimum": 0, "description": "Number of lines to show before and after each match" },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of matches to return (default: 100)" }
            },
            "required": ["pattern"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let pattern = args
                    .get("pattern")
                    .and_then(Value::as_str)
                    .ok_or("缺少 pattern 参数")?
                    .to_string();
                let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
                let glob = args.get("glob").and_then(Value::as_str);
                let ignore_case = args
                    .get("ignoreCase")
                    .or_else(|| args.get("ignore_case"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let literal = args
                    .get("literal")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let context_lines = args
                    .get("context")
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v).unwrap_or(0))
                    .unwrap_or(0);
                let effective_limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v.max(1)).unwrap_or(DEFAULT_LIMIT))
                    .unwrap_or(DEFAULT_LIMIT);

                if call_ctx.cancel.is_cancelled() {
                    return Err("Operation aborted".into());
                }

                let search_path = resolve_to_cwd(&cwd, raw_path, restrict_to_workspace)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;
                let meta = tokio::fs::metadata(&search_path)
                    .await
                    .map_err(|_| format!("Path not found: {}", search_path.display()))?;
                let is_dir = meta.is_dir();

                let mut cmd = Command::new("rg");
                cmd.arg("--no-config")
                    .arg("--json")
                    .arg("--line-number")
                    .arg("--color=never")
                    .arg("--hidden")
                    .arg("--glob=!.git");
                if ignore_case {
                    cmd.arg("--ignore-case");
                }
                if literal {
                    cmd.arg("--fixed-strings");
                }
                if let Some(g) = glob {
                    let clean_glob = if let Some(rest) = g.strip_prefix("!") {
                        format!("!{}", rest.trim_start_matches("./"))
                    } else {
                        g.trim_start_matches("./").to_string()
                    };
                    cmd.arg("--glob").arg(clean_glob);
                }
                if context_lines > 0 {
                    cmd.arg("--context").arg(context_lines.to_string());
                }
                cmd.arg("--").arg(&pattern);
                if is_dir {
                    cmd.arg(".").current_dir(&search_path);
                } else {
                    cmd.arg(&search_path);
                }
                cmd.stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                let mut child = cmd.spawn().map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        "ripgrep (rg) 未安装或不在 PATH 中".to_string()
                    } else {
                        format!("Failed to run ripgrep: {e}")
                    }
                })?;

                let stdout = child.stdout.take().expect("stdout piped");
                let mut stderr = child.stderr.take().expect("stderr piped");
                let stderr_task = tokio::spawn(async move {
                    let mut s = String::new();
                    let _ = stderr.read_to_string(&mut s).await;
                    s
                });

                let mut reader = BufReader::new(stdout);
                let mut line_buf = String::new();
                let mut output_lines = Vec::new();
                let mut output_bytes = 0usize;
                let mut total_line_count = 0usize;
                let mut match_count = 0usize;
                let mut match_limit_reached = false;
                let mut lines_truncated = false;
                let mut limit_hit = false;
                let mut last_file = String::new();
                let mut trailing_context_end = 0usize;
                let mut killed_due_to_limit = false;
                let mut aborted = false;

                loop {
                    line_buf.clear();
                    let n = tokio::select! {
                        _ = call_ctx.cancel.cancelled() => {
                            aborted = true;
                            let _ = child.start_kill();
                            break;
                        }
                        res = reader.read_line(&mut line_buf) => res?,
                    };
                    if n == 0 {
                        break;
                    }
                    let trimmed = line_buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let Ok(ev) = serde_json::from_str::<Value>(trimmed) else {
                        continue;
                    };
                    let ev_type = ev.get("type").and_then(Value::as_str).unwrap_or("");
                    let is_match = ev_type == "match";
                    if !is_match && ev_type != "context" {
                        continue;
                    }
                    let Some(data) = ev.get("data") else {
                        continue;
                    };
                    let Some(file_raw) = decode_rg_field(data.get("path")) else {
                        continue;
                    };
                    let Some(line_num) = data.get("line_number").and_then(Value::as_u64) else {
                        continue;
                    };
                    let line_num = line_num as usize;
                    let raw_line_text = decode_rg_field(data.get("lines")).unwrap_or_default();
                    let clean_line = raw_line_text.trim_end_matches(&['\r', '\n'][..]);

                    let record_line = |file_raw: &str,
                                       line_num: usize,
                                       clean_line: &str,
                                       is_match: bool,
                                       last_file: &mut String,
                                       lines_truncated: &mut bool,
                                       output_lines: &mut Vec<String>,
                                       output_bytes: &mut usize,
                                       total_line_count: &mut usize| {
                        *last_file = file_raw.to_string();
                        let rel = format_rel_path(file_raw, &search_path, is_dir);
                        let (short_line, was_trunc) = truncate_line(clean_line, GREP_MAX_LINE_LENGTH);
                        if was_trunc {
                            *lines_truncated = true;
                        }
                        let formatted = if is_match {
                            format!("{rel}:{line_num}: {short_line}")
                        } else {
                            format!("{rel}-{line_num}- {short_line}")
                        };
                        if *output_bytes <= DEFAULT_MAX_BYTES {
                            output_lines.push(formatted.clone());
                        }
                        *output_bytes += formatted.len() + 1;
                        *total_line_count += 1;
                    };

                    if !limit_hit {
                        record_line(
                            &file_raw,
                            line_num,
                            clean_line,
                            is_match,
                            &mut last_file,
                            &mut lines_truncated,
                            &mut output_lines,
                            &mut output_bytes,
                            &mut total_line_count,
                        );
                        if is_match {
                            match_count += 1;
                            if match_count >= effective_limit {
                                limit_hit = true;
                                trailing_context_end = line_num + context_lines;
                            }
                        }
                    } else {
                        let in_window = file_raw == last_file && line_num <= trailing_context_end;
                        if in_window {
                            record_line(
                                &file_raw,
                                line_num,
                                clean_line,
                                false,
                                &mut last_file,
                                &mut lines_truncated,
                                &mut output_lines,
                                &mut output_bytes,
                                &mut total_line_count,
                            );
                        }
                        if is_match {
                            match_limit_reached = true;
                        }
                        if match_limit_reached && !(in_window && line_num < trailing_context_end) {
                            killed_due_to_limit = true;
                            let _ = child.start_kill();
                            break;
                        }
                    }
                }

                let status = child.wait().await?;
                let stderr_text = stderr_task.await.unwrap_or_default();

                if aborted {
                    return Err("Operation aborted".into());
                }
                if !killed_due_to_limit && !matches!(status.code(), Some(0) | Some(1)) {
                    let msg = if stderr_text.trim().is_empty() {
                        format!("ripgrep exited with status {status}")
                    } else {
                        stderr_text.trim().to_string()
                    };
                    return Err(msg.into());
                }

                if match_count == 0 {
                    return Ok(ToolOutput::text("No matches found")
                        .with_structured(json!({ "matches": 0, "output": "" })));
                }

                let raw_joined = output_lines.join("\n");
                let trunc = truncate_head(&raw_joined, usize::MAX, DEFAULT_MAX_BYTES);
                let mut text = trunc.content.clone();
                let mut notices = Vec::new();
                if match_limit_reached {
                    notices.push(format!(
                        "{effective_limit} matches limit reached. Use limit={} for more, or refine pattern",
                        effective_limit * 2
                    ));
                }
                if trunc.truncated || output_bytes > DEFAULT_MAX_BYTES {
                    notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
                }
                if lines_truncated {
                    notices.push(format!(
                        "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
                    ));
                }
                if !notices.is_empty() {
                    text.push_str(&format!("\n\n[{}]", notices.join(". ")));
                }

                Ok(ToolOutput::text(text)
                    .with_structured(json!({
                        "matches": match_count,
                        "limit_reached": match_limit_reached,
                        "truncated": trunc.truncated || output_bytes > DEFAULT_MAX_BYTES,
                        "lines_shown": trunc.output_lines,
                        "total_lines": total_line_count,
                    }))
                    .with_details(json!({
                        "matches": match_count,
                        "limit_reached": match_limit_reached,
                    })))
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
            "matches": { "type": "integer" },
            "limit_reached": { "type": "boolean" },
            "truncated": { "type": "boolean" }
        }
    }))
    .with_prompt_snippet("Search file contents by regex or literal pattern (ripgrep)")
}
