//! `find` 工具：按用户 `PATH` (`fd` → `fdfind`) → `~/.mizar/bin/fd` → 同源 Rust (`ignore` + `globset`) 三级定位；支持可选 `pattern`、修改时间倒序、默认排除构建目录、上限 200 条与并发抽干 stderr。

use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use globset::GlobBuilder;
use ignore::WalkBuilder;
use mizar_tools::truncate::{DEFAULT_MAX_BYTES, format_size, truncate_head};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;

use crate::grep::resolve_managed_tool;
use crate::path_utils::resolve_to_cwd;

const DEFAULT_LIMIT: usize = 200;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const DEFAULT_EXCLUDES: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
];

fn relativize_path(candidate: &str, search_path: &Path) -> String {
    let p = Path::new(candidate);
    if let Ok(rel) = p.strip_prefix(search_path) {
        let s = rel.to_string_lossy().replace('\\', "/");
        return s.trim_start_matches("./").to_string();
    }
    candidate
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
}

fn has_git_ancestor(start: &Path) -> bool {
    let mut cur = Some(start);
    while let Some(dir) = cur {
        if dir.join(".git").exists() {
            return true;
        }
        cur = dir.parent();
    }
    false
}

fn normalize_optional_pattern(raw: Option<&str>) -> Option<String> {
    let pat = raw?.trim();
    if pat.is_empty() || pat == "*" {
        return None;
    }
    Some(pat.trim_start_matches("./").replace('\\', "/"))
}

async fn drain_capped_stderr(mut stderr: tokio::process::ChildStderr) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if buf.len() < MAX_STDERR_BYTES {
                    let take = (MAX_STDERR_BYTES - buf.len()).min(n);
                    buf.extend_from_slice(&chunk[..take]);
                }
            }
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[derive(Debug, Clone)]
struct FoundEntry {
    display: String,
    mtime: SystemTime,
}

async fn run_fd_with_bin(
    fd_bin: &Path,
    search_path: &Path,
    pattern: Option<&str>,
    effective_limit: usize,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<Option<Vec<FoundEntry>>, String> {
    let scan_cap = effective_limit.saturating_mul(20).clamp(2_000, 10_000);
    let mut cmd = Command::new(fd_bin);
    cmd.arg("--color=never").arg("--hidden");
    for excluded in DEFAULT_EXCLUDES {
        cmd.arg("--exclude").arg(excluded);
    }
    cmd.arg("--max-results").arg((scan_cap + 1).to_string());

    if !has_git_ancestor(search_path) {
        cmd.arg("--no-require-git");
    }

    if let Some(clean_pat) = pattern {
        let is_path_pat = clean_pat.contains('/');
        let ignore_case = clean_pat == clean_pat.to_lowercase();
        cmd.arg("--glob");
        if ignore_case {
            cmd.arg("--ignore-case");
        } else {
            cmd.arg("--case-sensitive");
        }
        if is_path_pat {
            cmd.arg("--full-path");
            let abs_pat = if Path::new(clean_pat).is_absolute() {
                clean_pat.to_string()
            } else {
                format!(
                    "{}/{}",
                    search_path.to_string_lossy().trim_end_matches('/'),
                    clean_pat
                )
            };
            cmd.arg("--").arg(abs_pat);
        } else {
            cmd.arg("--").arg(clean_pat);
        }
    } else {
        // fd 位置参数为 `[<pattern>] [<path>...]`；传空正则占位防止把 search_path 误当成 pattern。
        cmd.arg("--").arg("");
    }

    cmd.arg(search_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            return Ok(None);
        }
        Err(e) => return Err(format!("Failed to execute fd: {e}")),
    };

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    // 并发抽干 stderr，防止子进程输出超过 64KB 管道缓冲导致死锁。
    let stderr_task = tokio::spawn(drain_capped_stderr(stderr));

    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut raw_paths = Vec::new();
    let mut aborted = false;

    loop {
        line.clear();
        let n = tokio::select! {
            _ = cancel.cancelled() => {
                aborted = true;
                let _ = child.start_kill();
                break;
            }
            res = reader.read_line(&mut line) => res.map_err(|e| e.to_string())?,
        };
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            raw_paths.push(trimmed.to_string());
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    let stderr_text = stderr_task.await.unwrap_or_default();
    if aborted {
        return Err("Operation aborted".into());
    }
    if !status.success() && raw_paths.is_empty() && !matches!(status.code(), Some(0) | Some(1)) {
        let detail = stderr_text.trim();
        if detail.is_empty() {
            return Err(format!("fd exited with status {status}"));
        }
        return Err(format!("fd exited with status {status}: {detail}"));
    }

    let search_path_owned = search_path.to_path_buf();
    let entries = tokio::task::spawn_blocking(move || {
        raw_paths
            .into_iter()
            .map(|raw| {
                let p = Path::new(&raw);
                let full_p = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    search_path_owned.join(p)
                };
                let mtime = std::fs::metadata(&full_p)
                    .and_then(|m| m.modified())
                    .unwrap_or(UNIX_EPOCH);
                FoundEntry {
                    display: relativize_path(&raw, &search_path_owned),
                    mtime,
                }
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| format!("find stat task failed: {e}"))?;

    Ok(Some(entries))
}

async fn run_ignore_walk(
    search_path: PathBuf,
    pattern: Option<String>,
    effective_limit: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<Vec<FoundEntry>, String> {
    let scan_cap = effective_limit.saturating_mul(20).clamp(2_000, 10_000);
    let matcher = match pattern.as_deref() {
        Some(clean_pat) => {
            let is_path_pat = clean_pat.contains('/');
            let ignore_case = clean_pat == clean_pat.to_lowercase();
            let compiled = GlobBuilder::new(clean_pat)
                .case_insensitive(ignore_case)
                .literal_separator(is_path_pat)
                .build()
                .map_err(|e| format!("Invalid glob pattern `{clean_pat}`: {e}"))?
                .compile_matcher();
            Some((compiled, is_path_pat))
        }
        None => None,
    };

    tokio::task::spawn_blocking(move || {
        let mut results = Vec::new();
        let walker = WalkBuilder::new(&search_path)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .require_git(false)
            .filter_entry(|entry| {
                entry.depth() == 0
                    || !entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| DEFAULT_EXCLUDES.contains(&name))
            })
            .build();

        for dent in walker {
            if cancel.is_cancelled() {
                return Err("Operation aborted".to_string());
            }
            let Ok(entry) = dent else { continue };
            let path = entry.path();
            if path == search_path {
                continue;
            }
            let rel = path
                .strip_prefix(&search_path)
                .map(|r| r.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"));

            let matched = match &matcher {
                Some((m, is_path_pat)) => {
                    if *is_path_pat {
                        m.is_match(&rel)
                    } else {
                        path.file_name().map(|n| m.is_match(n)).unwrap_or(false)
                    }
                }
                None => true,
            };

            if matched {
                let display = if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                    format!("{rel}/")
                } else {
                    rel
                };
                let mtime = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(UNIX_EPOCH);
                results.push(FoundEntry { display, mtime });
                if results.len() > scan_cap {
                    break;
                }
            }
        }
        Ok(results)
    })
    .await
    .map_err(|e| format!("find task failed: {e}"))?
}

pub fn create_find_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "find",
        format!(
            "Find files and directories sorted by modification time (newest first). Respects .gitignore and excludes build/dependency dirs (target, node_modules, .git, dist, build). Pattern is optional. Output is capped at {} results or {}KB.",
            DEFAULT_LIMIT,
            DEFAULT_MAX_BYTES / 1024
        ),
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Optional glob pattern to filter files (e.g. '*.rs', 'src/*.ts'). Omit to list all non-ignored files." },
                "path": { "type": "string", "description": "Directory to search in (defaults to workspace root)" },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of results (default: 200)" }
            }
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let pattern = normalize_optional_pattern(args.get("pattern").and_then(Value::as_str));
                let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
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
                if !tokio::fs::try_exists(&search_path).await.unwrap_or(false) {
                    return Err(format!("Path not found: {}", search_path.display()).into());
                }

                let (mut entries, backend) =
                    if let Some(fd_bin) = resolve_managed_tool(&["fd", "fdfind"], "fd") {
                        match run_fd_with_bin(
                            &fd_bin,
                            &search_path,
                            pattern.as_deref(),
                            effective_limit,
                            &call_ctx.cancel,
                        )
                        .await?
                        {
                            Some(res) => (res, "fd"),
                            None => (
                                run_ignore_walk(
                                    search_path.clone(),
                                    pattern.clone(),
                                    effective_limit,
                                    call_ctx.cancel.clone(),
                                )
                                .await?,
                                "rust",
                            ),
                        }
                    } else {
                        (
                            run_ignore_walk(
                                search_path.clone(),
                                pattern.clone(),
                                effective_limit,
                                call_ctx.cancel.clone(),
                            )
                            .await?,
                            "rust",
                        )
                    };

                if entries.is_empty() {
                    return Ok(ToolOutput::text("No files found matching pattern")
                        .with_structured(json!({
                            "count": 0,
                            "files": [],
                            "backend": backend,
                        })));
                }

                // 按修改时间倒序排列（最新修改优先），时间相同按路径字典序稳定兜底。
                entries.sort_by(|a, b| {
                    Reverse(a.mtime)
                        .cmp(&Reverse(b.mtime))
                        .then_with(|| a.display.to_ascii_lowercase().cmp(&b.display.to_ascii_lowercase()))
                });

                let limit_reached = entries.len() > effective_limit;
                entries.truncate(effective_limit);
                let results: Vec<String> = entries.into_iter().map(|e| e.display).collect();

                let joined = results.join("\n");
                let trunc = truncate_head(&joined, usize::MAX, DEFAULT_MAX_BYTES);
                let mut text = trunc.content.clone();
                let mut notices = Vec::new();
                if limit_reached {
                    notices.push(format!("{effective_limit} results limit reached"));
                }
                if trunc.truncated {
                    notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
                }
                if !notices.is_empty() {
                    text.push_str(&format!("\n\n[{}]", notices.join(". ")));
                }

                Ok(ToolOutput::text(text)
                    .with_structured(json!({
                        "count": results.len(),
                        "limit_reached": limit_reached,
                        "truncated": trunc.truncated,
                        "files": results,
                        "backend": backend,
                    }))
                    .with_details(json!({
                        "count": results.len(),
                        "limit_reached": limit_reached,
                        "backend": backend,
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
            "count": { "type": "integer" },
            "limit_reached": { "type": "boolean" },
            "truncated": { "type": "boolean" },
            "files": { "type": "array", "items": { "type": "string" } },
            "backend": { "type": "string" }
        }
    }))
    .with_prompt_snippet("Find files by optional glob pattern sorted by modification time")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grep::resolve_tool_with;
    use mizar_tools::{ToolCallContext, ToolExecution};
    use std::time::Duration;

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mizar_w13_find_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call_ctx(args: &Value) -> ToolCallContext {
        ToolCallContext {
            exec: ToolExecution::new("find", args.clone(), None),
            cancel: tokio_util::sync::CancellationToken::new(),
            on_update: None,
            ctx: cordis::Context::new(),
        }
    }

    fn make_exec(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(path).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(path, perm).unwrap();
        }
    }

    #[test]
    fn resolves_fd_fdfind_and_managed_dir() {
        let root = test_dir("resolve");
        let path_dir = root.join("bin");
        let mizar_dir = root.join("mizar_bin");
        std::fs::create_dir_all(&path_dir).unwrap();
        std::fs::create_dir_all(&mizar_dir).unwrap();

        let path_env = std::env::join_paths([&path_dir]).unwrap();
        let managed_fd = mizar_dir.join("fd");
        make_exec(&managed_fd, "#!/bin/sh\nexit 0\n");

        // PATH 无 fd/fdfind 时回退到 ~/.mizar/bin/fd
        assert_eq!(
            resolve_tool_with(Some(&path_env), Some(&mizar_dir), &["fd", "fdfind"], "fd"),
            Some(managed_fd)
        );

        // PATH 有 fdfind 时优先于 ~/.mizar/bin/fd
        let sys_fdfind = path_dir.join("fdfind");
        make_exec(&sys_fdfind, "#!/bin/sh\nexit 0\n");
        assert_eq!(
            resolve_tool_with(Some(&path_env), Some(&mizar_dir), &["fd", "fdfind"], "fd"),
            Some(sys_fdfind)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn rust_walk_supports_optional_pattern_excludes_and_limit() {
        let root = test_dir("rust_walk");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("target/debug/out.bin"), "ignored").unwrap();
        std::fs::write(root.join("node_modules/pkg/index.js"), "ignored").unwrap();
        std::fs::write(root.join("src/a.rs"), "a").unwrap();
        std::fs::write(root.join("README.md"), "readme").unwrap();

        let entries = run_ignore_walk(
            root.clone(),
            None,
            DEFAULT_LIMIT,
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
        let names: Vec<String> = entries.into_iter().map(|e| e.display).collect();
        assert!(names.contains(&"src/a.rs".to_string()));
        assert!(names.contains(&"README.md".to_string()));
        assert!(!names.iter().any(|n| n.contains("target")));
        assert!(!names.iter().any(|n| n.contains("node_modules")));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn find_tool_sorts_by_mtime_descending_and_caps_at_200_by_default() {
        let root = test_dir("mtime_and_cap");
        let older = root.join("older.rs");
        let newer = root.join("newer.rs");
        std::fs::write(&older, "old").unwrap();
        std::fs::write(&newer, "new").unwrap();
        let now = SystemTime::now();
        std::fs::File::options()
            .write(true)
            .open(&older)
            .unwrap()
            .set_modified(now - Duration::from_secs(120))
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&newer)
            .unwrap()
            .set_modified(now)
            .unwrap();

        let tool = create_find_tool(root.clone(), true);
        let out = (tool.execute)(json!({}), call_ctx(&json!({})))
            .await
            .unwrap();
        let lines: Vec<&str> = out.content.lines().collect();
        assert_eq!(lines[0], "newer.rs", "expected newest first: {lines:?}");
        assert_eq!(lines[1], "older.rs");

        // 创建 210 个文件，验证默认 200 上限
        for i in 0..210 {
            std::fs::write(root.join(format!("extra_{i:03}.txt")), "x").unwrap();
        }
        let capped = (tool.execute)(json!({}), call_ctx(&json!({})))
            .await
            .unwrap();
        assert!(capped.content.contains("200 results limit reached"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn run_fd_drains_large_stderr_without_64kb_pipe_deadlock() {
        let root = test_dir("stderr_deadlock");
        let fake_fd = root.join("fake_fd");
        // 输出 > 100KB 到 stderr 同时写 stdout，若未并发抽干必然在 64KB 管道死锁
        make_exec(
            &fake_fd,
            "#!/bin/sh\nawk 'BEGIN { for (i=0; i<2000; i++) print \"stderr_warning_padding_012345678901234567890123456789\" > \"/dev/stderr\"; print \"found.rs\" }'\n",
        );

        let cancel = tokio_util::sync::CancellationToken::new();
        let fut = run_fd_with_bin(&fake_fd, &root, None, DEFAULT_LIMIT, &cancel);
        let res = tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .expect("fd must not deadlock on >64KB stderr")
            .unwrap()
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].display, "found.rs");

        let _ = std::fs::remove_dir_all(&root);
    }
}
