//! `bash` 工具：执行 shell 命令，支持并行批次、默认 `120s` 超时（上限 `600s`）、追加 `~/.mizar/bin` 到 `PATH` 末尾、防 OOM 流式落盘、O(1) 实时预览与 `20KB / 400 lines / 768B` 双向省略输出。

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use crate::truncate::{
    BASH_MAX_BYTES, BASH_MAX_LINE_BYTES, BASH_MAX_LINES, TruncatedBy, create_temp_output_file,
    format_size, slice_utf8_head, slice_utf8_tail, truncate_head_tail, truncate_line_bytes,
    truncate_tail, write_temp_output,
};
use crate::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_BASH_TIMEOUT_SECS: f64 = 120.0;
pub const MAX_BASH_TIMEOUT_SECS: f64 = 600.0;

const BASH_UPDATE_THROTTLE: Duration = Duration::from_millis(120);
const PREVIEW_MAX_BYTES: usize = 8 * 1024;
const PREVIEW_MAX_LINES: usize = 40;
const SPILL_THRESHOLD_BYTES: usize = 64 * 1024;
const HEAD_CAPTURE_CAP: usize = 32 * 1024;
const TAIL_CAPTURE_CAP: usize = 32 * 1024;
const POST_EXIT_DRAIN_GRACE: Duration = Duration::from_millis(200);

/// 固定优先使用 POSIX 兼容 shell，避免继承用户交互式 `fish`/`nu`/`csh` 导致语法失败。
pub fn resolve_shell() -> String {
    for candidate in ["/bin/bash", "/usr/bin/bash", "/bin/sh"] {
        if Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }
    if let Ok(sh) = std::env::var("SHELL")
        && !sh.trim().is_empty()
    {
        return sh;
    }
    "/bin/sh".to_string()
}

/// 将 `extra` 目录追加到 `PATH` 末尾（若尚未存在），确保用户原有工具链始终优先。
pub(crate) fn append_path_if_missing(current_path: Option<&OsStr>, extra: &Path) -> OsString {
    let mut dirs: Vec<PathBuf> = current_path
        .map(std::env::split_paths)
        .into_iter()
        .flatten()
        .filter(|p| !p.as_os_str().is_empty())
        .collect();
    if !dirs.iter().any(|p| p == extra) {
        dirs.push(extra.to_path_buf());
    }
    std::env::join_paths(dirs).unwrap_or_else(|_| {
        current_path
            .map(OsStr::to_os_string)
            .unwrap_or_else(|| extra.as_os_str().to_os_string())
    })
}

pub(crate) fn clamp_tool_timeout(timeout_secs: Option<f64>) -> Result<f64, String> {
    match timeout_secs {
        Some(secs) if !secs.is_finite() || secs <= 0.0 => {
            Err("timeout must be a positive number".into())
        }
        Some(secs) => Ok(secs.min(MAX_BASH_TIMEOUT_SECS)),
        None => Ok(DEFAULT_BASH_TIMEOUT_SECS),
    }
}

pub fn kill_process_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // 以独立进程组启动时，向 `-pid` 发 SIGKILL 可一并清理后台子进程
        unsafe {
            libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
        }
    }
    let _ = child.start_kill();
}

/// 有界输出收集器：<= 64KB 留存内存，超出立即转流式落盘防 OOM，同时维护首/尾 32KB 滑动窗口实现 O(1) 预览。
struct OutputCollector {
    in_mem: Vec<u8>,
    head_buf: Vec<u8>,
    tail_buf: Vec<u8>,
    spill: Option<(PathBuf, tokio::fs::File)>,
    total_bytes: usize,
    total_newlines: usize,
    last_byte: Option<u8>,
}

impl OutputCollector {
    fn new() -> Self {
        Self {
            in_mem: Vec::new(),
            head_buf: Vec::new(),
            tail_buf: Vec::new(),
            spill: None,
            total_bytes: 0,
            total_newlines: 0,
            last_byte: None,
        }
    }

    async fn push(&mut self, chunk: &[u8]) {
        if chunk.is_empty() {
            return;
        }
        self.total_bytes = self.total_bytes.saturating_add(chunk.len());
        self.total_newlines = self
            .total_newlines
            .saturating_add(chunk.iter().filter(|&&b| b == b'\n').count());
        self.last_byte = chunk.last().copied();

        if self.head_buf.len() < HEAD_CAPTURE_CAP {
            let take = (HEAD_CAPTURE_CAP - self.head_buf.len()).min(chunk.len());
            self.head_buf.extend_from_slice(&chunk[..take]);
        }

        if chunk.len() >= TAIL_CAPTURE_CAP {
            self.tail_buf.clear();
            self.tail_buf
                .extend_from_slice(&chunk[chunk.len() - TAIL_CAPTURE_CAP..]);
        } else {
            self.tail_buf.extend_from_slice(chunk);
            if self.tail_buf.len() >= TAIL_CAPTURE_CAP * 2 {
                let drop_n = self.tail_buf.len() - TAIL_CAPTURE_CAP;
                self.tail_buf.drain(..drop_n);
            }
        }

        if let Some((_, file)) = &mut self.spill {
            if file.write_all(chunk).await.is_err() {
                self.spill = None;
            }
            return;
        }

        if self.in_mem.len() + chunk.len() <= SPILL_THRESHOLD_BYTES {
            self.in_mem.extend_from_slice(chunk);
            return;
        }

        // 超过阈值时迁移到流式临时文件并释放 in_mem，防止百 MB 输出 OOM。
        if let Ok((path, mut file)) = create_temp_output_file("mizar-bash-").await
            && file.write_all(&self.in_mem).await.is_ok()
            && file.write_all(chunk).await.is_ok()
        {
            self.spill = Some((path, file));
        }
        self.in_mem.clear();
        self.in_mem.shrink_to_fit();
    }

    fn preview_tail(&self) -> String {
        let tail_start = self.tail_buf.len().saturating_sub(TAIL_CAPTURE_CAP);
        let tail_slice = &self.tail_buf[tail_start..];
        let lossy = String::from_utf8_lossy(tail_slice);
        truncate_tail(&lossy, PREVIEW_MAX_LINES, PREVIEW_MAX_BYTES).content
    }

    fn total_lines_estimate(&self) -> usize {
        if self.total_bytes == 0 {
            0
        } else if self.last_byte == Some(b'\n') {
            self.total_newlines.max(1)
        } else {
            self.total_newlines + 1
        }
    }
}

fn format_spilled_head_tail(
    head_buf: &[u8],
    tail_buf: &[u8],
    total_lines: usize,
) -> (String, usize, Option<TruncatedBy>) {
    let head_str = String::from_utf8_lossy(head_buf);
    // head_buf 可能在中间行字节边界被截断，丢弃末段不完整行（除非整段只有这一行）。
    let head_complete = match head_str.rfind('\n') {
        Some(pos) => &head_str[..pos],
        None => head_str.as_ref(),
    };
    let head_lines_raw: Vec<(String, bool)> = head_complete
        .split('\n')
        .map(|l| truncate_line_bytes(l, BASH_MAX_LINE_BYTES))
        .collect();

    let tail_slice = if tail_buf.len() > TAIL_CAPTURE_CAP {
        &tail_buf[tail_buf.len() - TAIL_CAPTURE_CAP..]
    } else {
        tail_buf
    };
    let tail_str = String::from_utf8_lossy(tail_slice);
    let tail_trimmed = tail_str.strip_suffix('\n').unwrap_or(&tail_str);
    // tail_buf 起点可能落在某行中途，丢弃首段残行（除非整段只有一行）。
    let tail_complete = match tail_trimmed.find('\n') {
        Some(pos) => &tail_trimmed[pos + 1..],
        None => tail_trimmed,
    };
    let tail_lines_raw: Vec<(String, bool)> = tail_complete
        .split('\n')
        .map(|l| truncate_line_bytes(l, BASH_MAX_LINE_BYTES))
        .collect();

    let head_line_budget = (BASH_MAX_LINES * 2 / 5).max(1);
    let tail_line_budget = BASH_MAX_LINES.saturating_sub(head_line_budget).max(1);
    const MARKER_RESERVE: usize = 96;
    let usable_bytes = BASH_MAX_BYTES.saturating_sub(MARKER_RESERVE);
    let head_byte_budget = (usable_bytes * 2 / 5).max(1);
    let tail_byte_budget = usable_bytes.saturating_sub(head_byte_budget).max(1);

    let mut head_selected: Vec<String> = Vec::new();
    let mut head_bytes = 0usize;
    let mut stopped_by_bytes = false;
    for (line, _) in &head_lines_raw {
        if head_selected.len() >= head_line_budget {
            break;
        }
        let add = line.len() + usize::from(!head_selected.is_empty());
        if head_bytes + add > head_byte_budget {
            if head_selected.is_empty() {
                head_selected.push(slice_utf8_head(line, head_byte_budget).to_string());
            }
            stopped_by_bytes = true;
            break;
        }
        head_selected.push(line.clone());
        head_bytes += add;
    }

    let mut tail_selected: Vec<String> = Vec::new();
    let mut tail_bytes = 0usize;
    for (line, _) in tail_lines_raw.iter().rev() {
        if tail_selected.len() >= tail_line_budget {
            break;
        }
        let add = line.len() + usize::from(!tail_selected.is_empty());
        if tail_bytes + add > tail_byte_budget {
            if tail_selected.is_empty() {
                tail_selected.push(slice_utf8_tail(line, tail_byte_budget).to_string());
            }
            stopped_by_bytes = true;
            break;
        }
        tail_selected.push(line.clone());
        tail_bytes += add;
    }
    tail_selected.reverse();

    let kept = head_selected.len() + tail_selected.len();
    let omitted = total_lines.saturating_sub(kept);
    let mut parts = Vec::with_capacity(kept + 1);
    parts.extend(head_selected);
    if omitted > 0 {
        parts.push(format!("... [{omitted} lines omitted] ..."));
    }
    parts.extend(tail_selected);
    let joined = parts.join("\n");
    let trunc_by = if stopped_by_bytes && kept < BASH_MAX_LINES {
        Some(TruncatedBy::Bytes)
    } else {
        Some(TruncatedBy::Lines)
    };
    (joined, omitted, trunc_by)
}

/// 本地 shell 命令执行结果。
#[derive(Debug, Clone)]
pub struct LocalBashResult {
    pub display: String,
    pub truncated_output: String,
    pub truncation_notice: Option<String>,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub truncated: bool,
    pub full_output_path: Option<String>,
}

impl LocalBashResult {
    /// 生成供会话事件与上下文推导使用的输出文本（含截断落盘路径提示，不含重复退出码尾注）。
    pub fn session_output(&self) -> String {
        match &self.truncation_notice {
            Some(note) if self.truncated_output.is_empty() => note.clone(),
            Some(note) => format!("{}\n\n{note}", self.truncated_output),
            None => self.truncated_output.clone(),
        }
    }
}

/// 底层异步执行 Shell 进程，统一流式预览、整棵进程组取消、继承 fd 防挂起与大输出流式落盘。
pub async fn execute_bash_process<F>(
    command: &str,
    cwd: &Path,
    timeout_secs: Option<f64>,
    cancel: CancellationToken,
    mut on_preview: F,
) -> Result<LocalBashResult, String>
where
    F: FnMut(&str) + Send + 'static,
{
    if let Some(secs) = timeout_secs
        && (!secs.is_finite() || secs <= 0.0)
    {
        return Err("timeout must be a positive number".into());
    }

    if cancel.is_cancelled() {
        return Ok(LocalBashResult {
            display: "Command aborted".to_string(),
            truncated_output: String::new(),
            truncation_notice: None,
            exit_code: Some(-1),
            cancelled: true,
            timed_out: false,
            duration_ms: 0,
            truncated: false,
            full_output_path: None,
        });
    }

    let shell = resolve_shell();
    let mut cmd = Command::new(&shell);
    cmd.arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(mizar_bin) = super::grep::mizar_bin_dir()
        && mizar_bin.is_dir()
    {
        let merged_path = append_path_if_missing(std::env::var_os("PATH").as_deref(), &mizar_bin);
        cmd.env("PATH", merged_path);
    }

    #[cfg(unix)]
    cmd.process_group(0);

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn shell `{shell}`: {e}"))?;

    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let mut collector = OutputCollector::new();
    let mut out_buf = [0u8; 4096];
    let mut err_buf = [0u8; 4096];
    let mut out_eof = stdout.is_none();
    let mut err_eof = stderr.is_none();
    let mut last_update = Instant::now() - BASH_UPDATE_THROTTLE;
    let mut timed_out = false;
    let mut aborted = false;
    let mut exit_status: Option<std::process::ExitStatus> = None;

    let deadline = timeout_secs.map(|s| tokio::time::Instant::now() + Duration::from_secs_f64(s));
    let timeout_fut = async {
        if let Some(dl) = deadline {
            tokio::time::sleep_until(dl).await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    tokio::pin!(timeout_fut);

    let drain_grace_fut = tokio::time::sleep(Duration::from_secs(86400));
    tokio::pin!(drain_grace_fut);

    while !(out_eof && err_eof && exit_status.is_some()) {
        tokio::select! {
            _ = cancel.cancelled() => {
                aborted = true;
                drop(stdout.take());
                drop(stderr.take());
                kill_process_tree(&mut child);
                break;
            }
            _ = &mut timeout_fut, if deadline.is_some() => {
                timed_out = true;
                drop(stdout.take());
                drop(stderr.take());
                kill_process_tree(&mut child);
                break;
            }
            _ = &mut drain_grace_fut, if exit_status.is_some() => {
                // 主 shell 已退出但孤儿后台进程仍持有 stdout/stderr fd，宽限期满后主动断开避免无限挂起。
                drop(stdout.take());
                drop(stderr.take());
                break;
            }
            wait_res = child.wait(), if exit_status.is_none() => {
                exit_status = Some(wait_res.unwrap_or_default());
                if out_eof && err_eof {
                    break;
                }
                drain_grace_fut
                    .as_mut()
                    .reset(tokio::time::Instant::now() + POST_EXIT_DRAIN_GRACE);
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
                        collector.push(&out_buf[..n]).await;
                        if last_update.elapsed() >= BASH_UPDATE_THROTTLE {
                            last_update = Instant::now();
                            on_preview(&collector.preview_tail());
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
                        collector.push(&err_buf[..n]).await;
                        if last_update.elapsed() >= BASH_UPDATE_THROTTLE {
                            last_update = Instant::now();
                            on_preview(&collector.preview_tail());
                        }
                    }
                }
            }
        }
    }

    if exit_status.is_none() {
        exit_status = tokio::time::timeout(Duration::from_secs(1), child.wait())
            .await
            .ok()
            .and_then(Result::ok);
    }
    let exit_code = exit_status
        .and_then(|s| s.code())
        .or(Some(if aborted || timed_out { -1 } else { 0 }));
    let duration_ms = started.elapsed().as_millis() as u64;

    let (truncated_output, truncated, total_lines, omitted_lines, truncated_by, full_output_path) =
        if let Some((path, mut file)) = collector.spill.take() {
            let _ = file.flush().await;
            drop(file);
            let total_lines = collector.total_lines_estimate();
            let (content, omitted, trunc_by) =
                format_spilled_head_tail(&collector.head_buf, &collector.tail_buf, total_lines);
            (
                content,
                true,
                total_lines,
                omitted,
                trunc_by,
                Some(path.to_string_lossy().into_owned()),
            )
        } else {
            let full_text = String::from_utf8_lossy(&collector.in_mem).into_owned();
            let truncation = truncate_head_tail(
                &full_text,
                BASH_MAX_LINES,
                BASH_MAX_BYTES,
                BASH_MAX_LINE_BYTES,
            );
            let full_path = if truncation.truncated {
                write_temp_output("mizar-bash-", &full_text)
                    .await
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            } else {
                None
            };
            (
                truncation.content,
                truncation.truncated,
                truncation.total_lines,
                truncation.omitted_lines,
                truncation.truncated_by,
                full_path,
            )
        };

    let mut display = if truncated_output.is_empty() {
        "(no output)".to_string()
    } else {
        truncated_output.clone()
    };

    let truncation_notice = if truncated {
        let path_note = full_output_path
            .as_deref()
            .map(|p| format!(" Full output saved to: {p}"))
            .unwrap_or_default();
        let reason = match (omitted_lines > 0, truncated_by) {
            (true, Some(TruncatedBy::Bytes)) => format!(
                "[Output truncated ({} limit, {omitted_lines} of {total_lines} lines omitted; lines capped at {BASH_MAX_LINE_BYTES}B).{path_note}]",
                format_size(BASH_MAX_BYTES),
            ),
            (true, _) => format!(
                "[Output truncated ({BASH_MAX_LINES} lines limit, {omitted_lines} of {total_lines} lines omitted; lines capped at {BASH_MAX_LINE_BYTES}B).{path_note}]",
            ),
            (false, _) => format!(
                "[Long lines capped at {BASH_MAX_LINE_BYTES}B (total output {}).{path_note}]",
                format_size(collector.total_bytes),
            ),
        };
        display.push_str("\n\n");
        display.push_str(&reason);
        Some(reason)
    } else {
        None
    };

    if aborted {
        display.push_str("\n\nCommand aborted");
    } else if timed_out {
        let secs = timeout_secs.unwrap_or(0.0);
        display.push_str(&format!("\n\nCommand timed out after {secs} seconds"));
    } else if let Some(code) = exit_code
        && code != 0
    {
        display.push_str(&format!("\n\nCommand exited with code {code}"));
    }

    Ok(LocalBashResult {
        display,
        truncated_output,
        truncation_notice,
        exit_code,
        cancelled: aborted,
        timed_out,
        duration_ms,
        truncated,
        full_output_path,
    })
}

pub fn create_bash_tool(cwd: PathBuf) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "bash",
        format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Default timeout is {}s (max {}s). Output is capped at {} lines / {}KB (head + tail with middle omitted) and {}B per line; full output is saved to a temp file when truncated.",
            DEFAULT_BASH_TIMEOUT_SECS as u64,
            MAX_BASH_TIMEOUT_SECS as u64,
            BASH_MAX_LINES,
            BASH_MAX_BYTES / 1024,
            BASH_MAX_LINE_BYTES
        ),
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Bash command to execute" },
                "timeout": { "type": "number", "description": "Optional timeout in seconds (default: 120, max: 600)" }
            },
            "required": ["command"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let command = args
                    .get("command")
                    .and_then(Value::as_str)
                    .ok_or("Missing required parameter: command")?
                    .to_string();
                let raw_timeout = args.get("timeout").and_then(Value::as_f64);
                let effective_timeout = clamp_tool_timeout(raw_timeout)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;

                if call_ctx.cancel.is_cancelled() {
                    return Err("Command aborted".into());
                }
                let ctx_clone = call_ctx.clone();
                let res = execute_bash_process(
                    &command,
                    &cwd,
                    Some(effective_timeout),
                    call_ctx.cancel.clone(),
                    move |preview| {
                        ctx_clone.emit_update(preview.to_string(), None);
                    },
                )
                .await?;

                let is_error = res.cancelled || res.timed_out || res.exit_code.unwrap_or(0) != 0;
                let structured = json!({
                    "output": res.truncated_output,
                    "exit_code": res.exit_code,
                    "timed_out": res.timed_out,
                    "cancelled": res.cancelled,
                    "full_output_path": res.full_output_path,
                });
                let details = json!({
                    "exit_code": res.exit_code,
                    "timed_out": res.timed_out,
                    "cancelled": res.cancelled,
                    "truncated": res.truncated,
                    "full_output_path": res.full_output_path,
                    "duration_ms": res.duration_ms,
                });

                Ok(ToolOutput {
                    content: res.display,
                    is_error,
                    structured_content: Some(structured),
                    details: Some(details),
                })
            }
        },
    )
    .with_execution_mode(ToolExecutionMode::Parallel)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolCallContext, ToolExecution};

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mizar_w13_bash_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call_ctx(args: &Value) -> ToolCallContext {
        ToolCallContext {
            exec: ToolExecution::new("bash", args.clone(), None),
            cancel: CancellationToken::new(),
            on_update: None,
            ctx: cordis::Context::new(),
        }
    }

    #[test]
    fn timeout_clamping_and_path_append_helpers() {
        assert_eq!(clamp_tool_timeout(None).unwrap(), 120.0);
        assert_eq!(clamp_tool_timeout(Some(30.0)).unwrap(), 30.0);
        assert_eq!(clamp_tool_timeout(Some(9999.0)).unwrap(), 600.0);
        assert!(clamp_tool_timeout(Some(0.0)).is_err());
        assert!(clamp_tool_timeout(Some(-5.0)).is_err());

        let extra = PathBuf::from("/tmp/mizar_bin_test");
        let base = std::env::join_paths([Path::new("/usr/bin"), Path::new("/bin")]).unwrap();
        let merged = append_path_if_missing(Some(&base), &extra);
        let parts: Vec<PathBuf> = std::env::split_paths(&merged).collect();
        assert_eq!(parts.last().unwrap(), &extra);

        // 幂等：已存在时不重复追加
        let again = append_path_if_missing(Some(&merged), &extra);
        assert_eq!(std::env::split_paths(&again).count(), parts.len());

        let sh = resolve_shell();
        assert!(sh.ends_with("bash") || sh.ends_with("sh"));
    }

    #[tokio::test]
    async fn tool_is_parallel_and_enforces_timeout() {
        let root = test_dir("timeout");
        let tool = create_bash_tool(root.clone());
        assert_eq!(tool.execution_mode, ToolExecutionMode::Parallel);
        assert!(tool.description.contains("120s"));
        assert!(tool.description.contains("600s"));

        let missing = (tool.execute)(json!({}), call_ctx(&json!({})))
            .await
            .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("Missing required parameter: command")
        );

        let res = execute_bash_process(
            "sleep 5",
            &root,
            Some(0.2),
            CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert!(res.timed_out);
        assert!(res.display.contains("Command timed out after 0.2 seconds"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn head_tail_omission_and_streaming_spill_prevent_oom() {
        let root = test_dir("spill");

        // 1. <= 64KB 但超过 400 行：走内存 head+tail 并落盘
        let res_small_spill = execute_bash_process(
            "seq 1 600",
            &root,
            Some(10.0),
            CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert!(res_small_spill.truncated);
        assert!(res_small_spill.display.contains("1\n2\n"));
        assert!(res_small_spill.display.contains("lines omitted] ..."));
        assert!(res_small_spill.display.contains("600"));
        let p1 = PathBuf::from(res_small_spill.full_output_path.unwrap());
        assert!(p1.exists());
        let _ = std::fs::remove_file(p1);

        // 2. > 64KB（~100KB+）：触发流式 spill_file
        let res_big_spill = execute_bash_process(
            "seq 1 18000",
            &root,
            Some(15.0),
            CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert!(res_big_spill.truncated);
        assert!(res_big_spill.display.contains("1\n2\n"));
        assert!(res_big_spill.display.contains("18000"));
        assert!(res_big_spill.display.contains("Full output saved to:"));
        let p2 = PathBuf::from(res_big_spill.full_output_path.unwrap());
        assert!(p2.exists());
        assert!(std::fs::metadata(&p2).unwrap().len() > 64 * 1024);
        let _ = std::fs::remove_file(p2);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn background_process_inheriting_fds_does_not_hang() {
        let root = test_dir("bg_fd");
        let started = Instant::now();
        let res = execute_bash_process(
            "sleep 3 & echo done_bg",
            &root,
            Some(5.0),
            CancellationToken::new(),
            |_| {},
        )
        .await
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "should not wait for background sleep fd, elapsed {:?}",
            started.elapsed()
        );
        assert_eq!(res.exit_code, Some(0));
        assert!(res.display.contains("done_bg"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
