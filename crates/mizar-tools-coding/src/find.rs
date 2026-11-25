//! `find` 工具：优先调用本地 `fd`（尊重 `.gitignore` 与隐藏文件），若宿主未安装 `fd` 则自动回退到同源 Rust `ignore` + `globset` 实现。

use std::path::{Path, PathBuf};
use std::process::Stdio;

use globset::GlobBuilder;
use ignore::WalkBuilder;
use mizar_tools::truncate::{DEFAULT_MAX_BYTES, format_size, truncate_head};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::path_utils::resolve_to_cwd;

const DEFAULT_LIMIT: usize = 1000;

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

async fn run_fd(
    search_path: &Path,
    pattern: &str,
    effective_limit: usize,
    cancel: &tokio_util::sync::CancellationToken,
) -> Result<Option<Vec<String>>, String> {
    let clean_pat = pattern.trim_start_matches("./").replace('\\', "/");
    let is_path_pat = clean_pat.contains('/');
    let ignore_case = clean_pat == clean_pat.to_lowercase();

    let mut cmd = Command::new("fd");
    cmd.arg("--color=never")
        .arg("--hidden")
        .arg("--exclude")
        .arg(".git")
        .arg("--glob")
        .arg("--max-results")
        .arg((effective_limit + 1).to_string());

    if !has_git_ancestor(search_path) {
        cmd.arg("--no-require-git");
    }
    if ignore_case {
        cmd.arg("--ignore-case");
    } else {
        cmd.arg("--case-sensitive");
    }
    if is_path_pat {
        cmd.arg("--full-path");
        let abs_pat = if Path::new(&clean_pat).is_absolute() {
            clean_pat.clone()
        } else {
            format!(
                "{}/{}",
                search_path.to_string_lossy().trim_end_matches('/'),
                clean_pat
            )
        };
        cmd.arg(abs_pat);
    } else {
        cmd.arg(&clean_pat);
    }
    cmd.arg(search_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("Failed to execute fd: {e}")),
    };

    let stdout = child.stdout.take().expect("stdout piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut results = Vec::new();
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
            results.push(relativize_path(trimmed, search_path));
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if aborted {
        return Err("Operation aborted".into());
    }
    if !status.success() && results.is_empty() && !matches!(status.code(), Some(0) | Some(1)) {
        return Err(format!("fd exited with status {status}"));
    }

    Ok(Some(results))
}

async fn run_ignore_walk(
    search_path: PathBuf,
    pattern: String,
    effective_limit: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<Vec<String>, String> {
    let clean_pat = pattern.trim_start_matches("./").replace('\\', "/");
    let is_path_pat = clean_pat.contains('/');
    let ignore_case = clean_pat == clean_pat.to_lowercase();

    let matcher = GlobBuilder::new(&clean_pat)
        .case_insensitive(ignore_case)
        .literal_separator(is_path_pat)
        .build()
        .map_err(|e| format!("Invalid glob pattern `{pattern}`: {e}"))?
        .compile_matcher();

    tokio::task::spawn_blocking(move || {
        let mut results = Vec::new();
        let walker = WalkBuilder::new(&search_path)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .require_git(false)
            .filter_entry(|entry| entry.file_name() != ".git")
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

            let matched = if is_path_pat {
                matcher.is_match(&rel)
            } else {
                path.file_name()
                    .map(|n| matcher.is_match(n))
                    .unwrap_or(false)
            };

            if matched {
                let display = if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                    format!("{rel}/")
                } else {
                    rel
                };
                results.push(display);
                if results.len() > effective_limit {
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
            "Search for files by glob pattern using fd (or gitignore-aware Rust fallback). Returns matching file paths relative to search directory. Output is truncated to {} results or {}KB.",
            DEFAULT_LIMIT,
            DEFAULT_MAX_BYTES / 1024
        ),
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern to match files, e.g. '*.rs', '**/*.ts', 'src/*'" },
                "path": { "type": "string", "description": "Directory to search in (defaults to workspace root)" },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of results (default: 1000)" }
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

                let mut results = match run_fd(&search_path, &pattern, effective_limit, &call_ctx.cancel).await {
                    Ok(Some(res)) => res,
                    Ok(None) => run_ignore_walk(
                        search_path.clone(),
                        pattern.clone(),
                        effective_limit,
                        call_ctx.cancel.clone(),
                    )
                    .await
                    .map_err(|e| -> cordis::BoxError { e.into() })?,
                    Err(e) => return Err(e.into()),
                };

                if results.is_empty() {
                    return Ok(ToolOutput::text("No files found matching pattern")
                        .with_structured(json!({ "count": 0, "files": [] })));
                }

                let limit_reached = results.len() > effective_limit;
                results.truncate(effective_limit);
                results.sort_by_key(|s| s.to_ascii_lowercase());

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
                    }))
                    .with_details(json!({
                        "count": results.len(),
                        "limit_reached": limit_reached,
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
            "files": { "type": "array", "items": { "type": "string" } }
        }
    }))
    .with_prompt_snippet("Find files by glob pattern respecting .gitignore (fd / ignore walker)")
}
