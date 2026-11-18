//! `bash` 工具：执行 shell 命令，支持流式预览、超时/取消整棵子进程树清理、尾部截断与完整日志落盘。

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use dsh_tools::truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size, truncate_tail, write_temp_output,
};
use dsh_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

const BASH_UPDATE_THROTTLE: Duration = Duration::from_millis(120);
const PREVIEW_MAX_BYTES: usize = 8 * 1024;
const PREVIEW_MAX_LINES: usize = 40;

fn resolve_shell() -> String {
    if let Ok(sh) = std::env::var("SHELL")
        && !sh.trim().is_empty()
    {
        return sh;
    }
    if std::path::Path::new("/bin/bash").exists() {
        "/bin/bash".to_string()
    } else {
        "/bin/sh".to_string()
    }
}

fn kill_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // 以独立进程组启动时，向 `-pid` 发 SIGKILL 可一并清理后台子进程
        unsafe {
            libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
        }
    }
    let _ = child.start_kill();
}

pub fn create_bash_tool(cwd: PathBuf) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "bash",
        format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
            DEFAULT_MAX_LINES,
            DEFAULT_MAX_BYTES / 1024
        ),
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Bash command to execute" },
                "timeout": { "type": "number", "description": "Optional timeout in seconds" }
            },
            "required": ["command"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let command = args
                    .get("command")
                    .and_then(Value::as_str)
                    .ok_or("缺少 command 参数")?
                    .to_string();
                let timeout_secs = args.get("timeout").and_then(Value::as_f64);
                if let Some(secs) = timeout_secs
                    && (!secs.is_finite() || secs <= 0.0)
                {
                    return Err("timeout 必须是大于 0 的数字".into());
                }

                if call_ctx.cancel.is_cancelled() {
                    return Err("Command aborted".into());
                }

                let shell = resolve_shell();
                let mut cmd = Command::new(&shell);
                cmd.arg("-c")
                    .arg(&command)
                    .current_dir(&cwd)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                #[cfg(unix)]
                cmd.process_group(0);

                let started = Instant::now();
                let mut child = cmd
                    .spawn()
                    .map_err(|e| format!("Failed to spawn shell `{shell}`: {e}"))?;

                let mut stdout = child.stdout.take();
                let mut stderr = child.stderr.take();

                let mut raw_output = Vec::<u8>::new();
                let mut out_buf = [0u8; 4096];
                let mut err_buf = [0u8; 4096];
                let mut out_eof = stdout.is_none();
                let mut err_eof = stderr.is_none();
                let mut last_update = Instant::now() - BASH_UPDATE_THROTTLE;
                let mut timed_out = false;
                let mut aborted = false;

                let deadline = timeout_secs.map(|s| tokio::time::Instant::now() + Duration::from_secs_f64(s));
                let sleep_fut = async {
                    if let Some(dl) = deadline {
                        tokio::time::sleep_until(dl).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                };
                tokio::pin!(sleep_fut);

                while !(out_eof && err_eof) {
                    tokio::select! {
                        _ = call_ctx.cancel.cancelled() => {
                            aborted = true;
                            kill_process_tree(&mut child);
                            break;
                        }
                        _ = &mut sleep_fut, if deadline.is_some() => {
                            timed_out = true;
                            kill_process_tree(&mut child);
                            break;
                        }
                        res = async {
                            match stdout.as_mut() {
                                Some(s) => s.read(&mut out_buf).await,
                                None => std::future::pending().await,
                            }
                        }, if !out_eof => {
                            match res {
                                Ok(0) | Err(_) => out_eof = true,
                                Ok(n) => {
                                    raw_output.extend_from_slice(&out_buf[..n]);
                                    if last_update.elapsed() >= BASH_UPDATE_THROTTLE {
                                        last_update = Instant::now();
                                        let preview_full = String::from_utf8_lossy(&raw_output);
                                        let trunc = truncate_tail(&preview_full, PREVIEW_MAX_LINES, PREVIEW_MAX_BYTES);
                                        call_ctx.emit_update(trunc.content, None);
                                    }
                                }
                            }
                        }
                        res = async {
                            match stderr.as_mut() {
                                Some(s) => s.read(&mut err_buf).await,
                                None => std::future::pending().await,
                            }
                        }, if !err_eof => {
                            match res {
                                Ok(0) | Err(_) => err_eof = true,
                                Ok(n) => {
                                    raw_output.extend_from_slice(&err_buf[..n]);
                                    if last_update.elapsed() >= BASH_UPDATE_THROTTLE {
                                        last_update = Instant::now();
                                        let preview_full = String::from_utf8_lossy(&raw_output);
                                        let trunc = truncate_tail(&preview_full, PREVIEW_MAX_LINES, PREVIEW_MAX_BYTES);
                                        call_ctx.emit_update(trunc.content, None);
                                    }
                                }
                            }
                        }
                    }
                }

                let exit_code = match child.wait().await {
                    Ok(status) => status.code().unwrap_or(if aborted || timed_out { -1 } else { 1 }),
                    Err(_) => -1,
                };
                let duration_ms = started.elapsed().as_millis() as u64;

                let full_text = String::from_utf8_lossy(&raw_output).into_owned();
                let truncation = truncate_tail(&full_text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);

                let full_output_path = if truncation.truncated {
                    write_temp_output("dsh-bash-", &full_text)
                        .await
                        .ok()
                        .map(|p| p.to_string_lossy().into_owned())
                } else {
                    None
                };

                let mut display = if truncation.content.is_empty() {
                    "(no output)".to_string()
                } else {
                    truncation.content.clone()
                };

                if truncation.truncated {
                    let start_line = truncation
                        .total_lines
                        .saturating_sub(truncation.output_lines)
                        + 1;
                    let end_line = truncation.total_lines;
                    let path_note = full_output_path
                        .as_deref()
                        .map(|p| format!(" Full output saved to: {p}"))
                        .unwrap_or_default();
                    if truncation.last_line_partial {
                        display.push_str(&format!(
                            "\n\n[Showing last {} of line {end_line} (line is {}).{path_note}]",
                            format_size(truncation.output_bytes),
                            format_size(truncation.total_bytes),
                        ));
                    } else {
                        display.push_str(&format!(
                            "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit).{path_note}]",
                            truncation.total_lines,
                            format_size(DEFAULT_MAX_BYTES),
                        ));
                    }
                }

                if aborted {
                    display.push_str("\n\nCommand aborted");
                } else if timed_out {
                    let secs = timeout_secs.unwrap_or(0.0);
                    display.push_str(&format!("\n\nCommand timed out after {secs} seconds"));
                } else if exit_code != 0 {
                    display.push_str(&format!("\n\nCommand exited with code {exit_code}"));
                }

                let is_error = aborted || timed_out || exit_code != 0;
                let structured = json!({
                    "output": truncation.content,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "cancelled": aborted,
                    "full_output_path": full_output_path,
                });
                let details = json!({
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "cancelled": aborted,
                    "truncated": truncation.truncated,
                    "full_output_path": full_output_path,
                    "duration_ms": duration_ms,
                });

                Ok(ToolOutput {
                    content: display,
                    is_error,
                    structured_content: Some(structured),
                    details: Some(details),
                })
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Sequential)
    .with_annotations(ToolAnnotations {
        read_only: false,
        destructive: true,
        idempotent: false,
        open_world: true,
    })
    .with_output_schema(json!({
        "type": "object",
        "properties": {
            "output": { "type": "string" },
            "exit_code": { "type": "integer" },
            "timed_out": { "type": "boolean" },
            "cancelled": { "type": "boolean" },
            "full_output_path": { "type": ["string", "null"] }
        }
    }))
    .with_prompt_snippet("Execute bash commands in the workspace")
}
