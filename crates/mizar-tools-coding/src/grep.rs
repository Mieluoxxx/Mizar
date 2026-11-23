//! `grep` 工具：按用户 `PATH` → `~/.mizar/bin/` → 同源 Rust (`ignore` + `regex`) 三级定位，支持 `content` / `files_with_matches` / `count` 三种输出模式及单文件/总匹配上限。

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use mizar_tools::truncate::{
    DEFAULT_MAX_BYTES, GREP_MAX_LINE_LENGTH, format_size, truncate_head, truncate_line,
};
use mizar_tools::{ToolAnnotations, ToolDefinition, ToolExecutionMode, ToolOutput};
use regex::RegexBuilder;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::path_utils::resolve_to_cwd;

const DEFAULT_LIMIT: usize = 100;
const DEFAULT_MAX_PER_FILE: usize = 20;
const MAX_STDERR_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GrepOutputMode {
    Content,
    FilesWithMatches,
    Count,
}

impl GrepOutputMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::FilesWithMatches => "files_with_matches",
            Self::Count => "count",
        }
    }

    fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("content") => Ok(Self::Content),
            Some("files_with_matches") => Ok(Self::FilesWithMatches),
            Some("count") => Ok(Self::Count),
            Some(other) => Err(format!(
                "Invalid output_mode '{other}'. Expected 'content', 'files_with_matches', or 'count'."
            )),
        }
    }
}

/// 返回私有托管工具目录 `~/.mizar/bin`。
pub(crate) fn mizar_bin_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mizar").join("bin"))
}

pub(crate) fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// 三级定位纯函数：优先用户 `PATH`（按 `path_names` 顺序），次选 `mizar_bin/managed_name`，均缺失返回 `None`。
pub(crate) fn resolve_tool_with(
    path_env: Option<&OsStr>,
    mizar_bin: Option<&Path>,
    path_names: &[&str],
    managed_name: &str,
) -> Option<PathBuf> {
    if let Some(raw_path) = path_env {
        for name in path_names {
            for dir in std::env::split_paths(raw_path) {
                if dir.as_os_str().is_empty() {
                    continue;
                }
                let candidate = dir.join(name);
                if is_executable_file(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }

    if let Some(bin_dir) = mizar_bin {
        let managed = bin_dir.join(managed_name);
        if is_executable_file(&managed) {
            return Some(managed);
        }
    }

    None
}

pub(crate) fn resolve_managed_tool(path_names: &[&str], managed_name: &str) -> Option<PathBuf> {
    resolve_tool_with(
        std::env::var_os("PATH").as_deref(),
        mizar_bin_dir().as_deref(),
        path_names,
        managed_name,
    )
}

#[derive(Debug, Clone)]
pub(crate) struct GrepParams {
    pub pattern: String,
    pub glob: Option<String>,
    pub ignore_case: bool,
    pub literal: bool,
    pub context_lines: usize,
    pub limit: usize,
    pub max_per_file: Option<usize>,
    pub mode: GrepOutputMode,
}

#[derive(Debug, Clone)]
pub(crate) struct GrepOutcome {
    pub output_lines: Vec<String>,
    pub matches: usize,
    pub limit_reached: bool,
    pub per_file_limit_reached: bool,
    pub lines_truncated: bool,
    pub bytes_overflowed: bool,
    pub total_line_count: usize,
    pub backend: &'static str,
}

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
        let raw_path = Path::new(raw);
        let stripped = raw_path
            .strip_prefix(search_path)
            .or_else(|_| raw_path.strip_prefix("."))
            .unwrap_or(raw_path);
        let s = stripped
            .to_string_lossy()
            .replace('\\', "/")
            .trim_start_matches("./")
            .to_string();
        if !s.is_empty() && !s.starts_with("..") {
            return s;
        }
    }
    Path::new(raw)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string())
}

fn normalize_rg_glob(raw: &str) -> String {
    if let Some(rest) = raw.strip_prefix('!') {
        format!("!{}", rest.trim_start_matches("./"))
    } else {
        raw.trim_start_matches("./").to_string()
    }
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

async fn run_rg_grep(
    rg_bin: &Path,
    search_path: &Path,
    is_dir: bool,
    params: &GrepParams,
    cancel: &CancellationToken,
) -> Result<Option<GrepOutcome>, String> {
    let mut cmd = Command::new(rg_bin);
    cmd.arg("--no-config")
        .arg("--color=never")
        .arg("--hidden")
        .arg("--glob=!.git");

    match params.mode {
        GrepOutputMode::Content => {
            cmd.arg("--json").arg("--line-number");
            if let Some(per_file) = params.max_per_file {
                cmd.arg("--max-count")
                    .arg(per_file.saturating_add(1).to_string());
            }
            if params.context_lines > 0 {
                cmd.arg("--context").arg(params.context_lines.to_string());
            }
        }
        GrepOutputMode::FilesWithMatches => {
            cmd.arg("--files-with-matches");
        }
        GrepOutputMode::Count => {
            cmd.arg("--count");
        }
    }

    if params.ignore_case {
        cmd.arg("--ignore-case");
    }
    if params.literal {
        cmd.arg("--fixed-strings");
    }
    if let Some(g) = params.glob.as_deref() {
        cmd.arg("--glob").arg(normalize_rg_glob(g));
    }

    cmd.arg("--").arg(&params.pattern);
    if is_dir {
        cmd.arg(".").current_dir(search_path);
    } else {
        cmd.arg(search_path);
    }
    cmd.stdin(Stdio::null())
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
        Err(e) => return Err(format!("Failed to execute ripgrep: {e}")),
    };

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let stderr_task = tokio::spawn(drain_capped_stderr(stderr));

    let mut reader = BufReader::new(stdout);
    let mut line_buf = String::new();
    let mut output_lines = Vec::new();
    let mut output_bytes = 0usize;
    let mut total_line_count = 0usize;
    let mut match_count = 0usize;
    let mut limit_reached = false;
    let mut per_file_limit_reached = false;
    let mut lines_truncated = false;
    let mut killed_due_to_limit = false;
    let mut aborted = false;

    let mut per_file_counts: HashMap<String, usize> = HashMap::new();
    let mut limit_hit = false;
    let mut last_file = String::new();
    let mut trailing_context_end = 0usize;

    loop {
        line_buf.clear();
        let n = tokio::select! {
            _ = cancel.cancelled() => {
                aborted = true;
                let _ = child.start_kill();
                break;
            }
            res = reader.read_line(&mut line_buf) => res.map_err(|e| e.to_string())?,
        };
        if n == 0 {
            break;
        }

        match params.mode {
            GrepOutputMode::FilesWithMatches => {
                let raw = line_buf.trim_end_matches(&['\r', '\n'][..]);
                if raw.is_empty() {
                    continue;
                }
                if match_count >= params.limit {
                    limit_reached = true;
                    killed_due_to_limit = true;
                    let _ = child.start_kill();
                    break;
                }
                let rel = format_rel_path(raw, search_path, is_dir);
                if output_bytes <= DEFAULT_MAX_BYTES {
                    output_lines.push(rel.clone());
                }
                output_bytes += rel.len() + 1;
                total_line_count += 1;
                match_count += 1;
            }
            GrepOutputMode::Count => {
                let raw = line_buf.trim_end_matches(&['\r', '\n'][..]);
                if raw.is_empty() {
                    continue;
                }
                let Some((file_part, count_part)) = raw.rsplit_once(':') else {
                    continue;
                };
                let Ok(file_matches) = count_part.trim().parse::<usize>() else {
                    continue;
                };
                if file_matches == 0 {
                    continue;
                }
                if match_count >= params.limit {
                    limit_reached = true;
                    killed_due_to_limit = true;
                    let _ = child.start_kill();
                    break;
                }
                let rel = format_rel_path(file_part, search_path, is_dir);
                let formatted = format!("{rel}:{file_matches}");
                if output_bytes <= DEFAULT_MAX_BYTES {
                    output_lines.push(formatted.clone());
                }
                output_bytes += formatted.len() + 1;
                total_line_count += 1;
                match_count += 1;
            }
            GrepOutputMode::Content => {
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

                if is_match && let Some(cap) = params.max_per_file {
                    let count = per_file_counts.entry(file_raw.clone()).or_insert(0);
                    if *count >= cap {
                        per_file_limit_reached = true;
                        continue;
                    }
                    *count += 1;
                }

                let mut push_record = |is_match_line: bool, last_file: &mut String| {
                    *last_file = file_raw.clone();
                    let rel = format_rel_path(&file_raw, search_path, is_dir);
                    let (short_line, was_trunc) = truncate_line(clean_line, GREP_MAX_LINE_LENGTH);
                    if was_trunc {
                        lines_truncated = true;
                    }
                    let formatted = if is_match_line {
                        format!("{rel}:{line_num}: {short_line}")
                    } else {
                        format!("{rel}-{line_num}- {short_line}")
                    };
                    if output_bytes <= DEFAULT_MAX_BYTES {
                        output_lines.push(formatted.clone());
                    }
                    output_bytes += formatted.len() + 1;
                    total_line_count += 1;
                };

                if !limit_hit {
                    push_record(is_match, &mut last_file);
                    if is_match {
                        match_count += 1;
                        if match_count >= params.limit {
                            limit_hit = true;
                            trailing_context_end = line_num + params.context_lines;
                        }
                    }
                } else {
                    let in_window = file_raw == last_file && line_num <= trailing_context_end;
                    if in_window && !is_match {
                        push_record(false, &mut last_file);
                    }
                    if is_match {
                        limit_reached = true;
                    }
                    if limit_reached && !(in_window && line_num < trailing_context_end) {
                        killed_due_to_limit = true;
                        let _ = child.start_kill();
                        break;
                    }
                }
            }
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
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
        return Err(msg);
    }

    Ok(Some(GrepOutcome {
        output_lines,
        matches: match_count,
        limit_reached,
        per_file_limit_reached,
        lines_truncated,
        bytes_overflowed: output_bytes > DEFAULT_MAX_BYTES,
        total_line_count,
        backend: "rg",
    }))
}

struct GlobFilter {
    negated: bool,
    is_path_pat: bool,
    matcher: GlobMatcher,
}

impl GlobFilter {
    fn parse(raw: &str) -> Result<Self, String> {
        let (negated, pat_raw) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw),
        };
        let clean = pat_raw.trim_start_matches("./").replace('\\', "/");
        let is_path_pat = clean.contains('/');
        let matcher = GlobBuilder::new(&clean)
            .literal_separator(is_path_pat)
            .build()
            .map_err(|e| format!("Invalid glob pattern `{raw}`: {e}"))?
            .compile_matcher();
        Ok(Self {
            negated,
            is_path_pat,
            matcher,
        })
    }

    fn matches(&self, path: &Path, rel: &str) -> bool {
        let hit = if self.is_path_pat {
            self.matcher.is_match(rel)
        } else {
            path.file_name()
                .map(|n| self.matcher.is_match(n))
                .unwrap_or_else(|| self.matcher.is_match(rel))
        };
        if self.negated { !hit } else { hit }
    }
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|&b| b == 0)
}

/// 纯 Rust (`ignore` + `regex`) 离线兜底搜索，保证无 `rg` 二进制时行为一致。
pub(crate) async fn run_rust_grep(
    search_path: PathBuf,
    is_dir: bool,
    params: GrepParams,
    cancel: CancellationToken,
) -> Result<GrepOutcome, String> {
    let raw_regex = if params.literal {
        regex::escape(&params.pattern)
    } else {
        params.pattern.clone()
    };
    let re = RegexBuilder::new(&raw_regex)
        .case_insensitive(params.ignore_case)
        .build()
        .map_err(|e| format!("Invalid search pattern: {e}"))?;

    let glob_filter = match params.glob.as_deref() {
        Some(g) => Some(GlobFilter::parse(g)?),
        None => None,
    };

    tokio::task::spawn_blocking(move || {
        let mut files: Vec<(PathBuf, String)> = Vec::new();
        if !is_dir {
            let rel = format_rel_path(&search_path.to_string_lossy(), &search_path, false);
            files.push((search_path.clone(), rel));
        } else {
            let walker = WalkBuilder::new(&search_path)
                .hidden(false)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .require_git(false)
                .filter_entry(|entry| entry.depth() == 0 || entry.file_name() != ".git")
                .build();

            for dent in walker {
                if cancel.is_cancelled() {
                    return Err("Operation aborted".to_string());
                }
                let Ok(entry) = dent else { continue };
                if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                    continue;
                }
                let path = entry.path().to_path_buf();
                let rel = path
                    .strip_prefix(&search_path)
                    .map(|r| r.to_string_lossy().replace('\\', "/"))
                    .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"));
                if let Some(gf) = &glob_filter
                    && !gf.matches(&path, &rel)
                {
                    continue;
                }
                files.push((path, rel));
            }
            files.sort_by(|a, b| a.1.cmp(&b.1));
        }

        let mut output_lines = Vec::new();
        let mut output_bytes = 0usize;
        let mut total_line_count = 0usize;
        let mut match_count = 0usize;
        let mut limit_reached = false;
        let mut per_file_limit_reached = false;
        let mut lines_truncated = false;

        'file_loop: for (file_path, rel) in files {
            if cancel.is_cancelled() {
                return Err("Operation aborted".to_string());
            }

            let Ok(mut f) = File::open(&file_path) else {
                continue;
            };
            let mut raw_bytes = Vec::new();
            if f.read_to_end(&mut raw_bytes).is_err() || is_probably_binary(&raw_bytes) {
                continue;
            }
            let text = String::from_utf8_lossy(&raw_bytes);
            let lines: Vec<&str> = text
                .strip_suffix('\n')
                .unwrap_or(&text)
                .split('\n')
                .map(|l| l.strip_suffix('\r').unwrap_or(l))
                .collect();

            match params.mode {
                GrepOutputMode::FilesWithMatches => {
                    if lines.iter().any(|l| re.is_match(l)) {
                        if match_count >= params.limit {
                            limit_reached = true;
                            break 'file_loop;
                        }
                        if output_bytes <= DEFAULT_MAX_BYTES {
                            output_lines.push(rel.clone());
                        }
                        output_bytes += rel.len() + 1;
                        total_line_count += 1;
                        match_count += 1;
                    }
                }
                GrepOutputMode::Count => {
                    let count = lines.iter().filter(|l| re.is_match(l)).count();
                    if count > 0 {
                        if match_count >= params.limit {
                            limit_reached = true;
                            break 'file_loop;
                        }
                        let formatted = format!("{rel}:{count}");
                        if output_bytes <= DEFAULT_MAX_BYTES {
                            output_lines.push(formatted.clone());
                        }
                        output_bytes += formatted.len() + 1;
                        total_line_count += 1;
                        match_count += 1;
                    }
                }
                GrepOutputMode::Content => {
                    let matching_indices: Vec<usize> = lines
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, l)| re.is_match(l).then_some(idx))
                        .collect();
                    if matching_indices.is_empty() {
                        continue;
                    }

                    let mut accepted_indices = Vec::new();
                    for (file_hits, idx) in matching_indices.into_iter().enumerate() {
                        if let Some(cap) = params.max_per_file
                            && file_hits >= cap
                        {
                            per_file_limit_reached = true;
                            break;
                        }
                        if match_count >= params.limit {
                            limit_reached = true;
                            break;
                        }
                        match_count += 1;
                        accepted_indices.push(idx);
                    }

                    if accepted_indices.is_empty() {
                        if limit_reached {
                            break 'file_loop;
                        }
                        continue;
                    }

                    let mut is_match_line = vec![false; lines.len()];
                    let mut show_line = vec![false; lines.len()];
                    for &idx in &accepted_indices {
                        is_match_line[idx] = true;
                        let start = idx.saturating_sub(params.context_lines);
                        let end = (idx + params.context_lines).min(lines.len().saturating_sub(1));
                        for slot in &mut show_line[start..=end] {
                            *slot = true;
                        }
                    }

                    for (idx, line_text) in lines.iter().enumerate() {
                        if !show_line[idx] {
                            continue;
                        }
                        let (short_line, was_trunc) =
                            truncate_line(line_text, GREP_MAX_LINE_LENGTH);
                        if was_trunc {
                            lines_truncated = true;
                        }
                        let line_num = idx + 1;
                        let formatted = if is_match_line[idx] {
                            format!("{rel}:{line_num}: {short_line}")
                        } else {
                            format!("{rel}-{line_num}- {short_line}")
                        };
                        if output_bytes <= DEFAULT_MAX_BYTES {
                            output_lines.push(formatted.clone());
                        }
                        output_bytes += formatted.len() + 1;
                        total_line_count += 1;
                    }

                    if limit_reached {
                        break 'file_loop;
                    }
                }
            }
        }

        Ok(GrepOutcome {
            output_lines,
            matches: match_count,
            limit_reached,
            per_file_limit_reached,
            lines_truncated,
            bytes_overflowed: output_bytes > DEFAULT_MAX_BYTES,
            total_line_count,
            backend: "rust",
        })
    })
    .await
    .map_err(|e| format!("grep task failed: {e}"))?
}

pub fn create_grep_tool(cwd: PathBuf, restrict_to_workspace: bool) -> ToolDefinition {
    ToolDefinition::async_rich_tool(
        "grep",
        format!(
            "Search file contents by regex or literal pattern (uses ripgrep when available, with built-in Rust fallback). Respects .gitignore. Supports output_mode ('content', 'files_with_matches', 'count'). Output is capped at {} matches (default {} per file in content mode) or {}KB, and lines are capped at {} chars.",
            DEFAULT_LIMIT,
            DEFAULT_MAX_PER_FILE,
            DEFAULT_MAX_BYTES / 1024,
            GREP_MAX_LINE_LENGTH
        ),
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Search pattern (regex by default, or literal when literal=true)" },
                "path": { "type": "string", "description": "Directory or file to search (defaults to workspace root)" },
                "glob": { "type": "string", "description": "Filter files by glob pattern, e.g. '*.rs' or '!*.lock'" },
                "ignoreCase": { "type": "boolean", "description": "Case-insensitive search (default: false)" },
                "literal": { "type": "boolean", "description": "Treat pattern as literal string instead of regex (default: false)" },
                "context": { "type": "integer", "minimum": 0, "description": "Number of context lines before and after each match (content mode only)" },
                "limit": { "type": "integer", "minimum": 1, "description": "Maximum number of matches or file entries to return (default: 100)" },
                "maxPerFile": { "type": "integer", "minimum": 1, "description": "Maximum matches per file in content mode (default: 20 for directory search)" },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output format: 'content' (matching lines), 'files_with_matches' (file paths only), or 'count' (path:count per file)"
                }
            },
            "required": ["pattern"]
        }),
        move |args, call_ctx| {
            let cwd = cwd.clone();
            async move {
                let pattern = args
                    .get("pattern")
                    .and_then(Value::as_str)
                    .ok_or("Missing required parameter: pattern")?
                    .to_string();
                let raw_path = args.get("path").and_then(Value::as_str).unwrap_or(".");
                let glob = args.get("glob").and_then(Value::as_str).map(str::to_string);
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
                let limit = args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v.max(1)).unwrap_or(DEFAULT_LIMIT))
                    .unwrap_or(DEFAULT_LIMIT);
                let mode = GrepOutputMode::parse(
                    args.get("output_mode")
                        .or_else(|| args.get("outputMode"))
                        .and_then(Value::as_str),
                )
                .map_err(|e| -> cordis::BoxError { e.into() })?;

                if call_ctx.cancel.is_cancelled() {
                    return Err("Operation aborted".into());
                }

                let search_path = resolve_to_cwd(&cwd, raw_path, restrict_to_workspace)
                    .map_err(|e| -> cordis::BoxError { e.into() })?;
                let meta = tokio::fs::metadata(&search_path)
                    .await
                    .map_err(|_| format!("Path not found: {}", search_path.display()))?;
                let is_dir = meta.is_dir();

                let explicit_per_file = args
                    .get("maxPerFile")
                    .or_else(|| args.get("max_per_file"))
                    .and_then(Value::as_u64)
                    .map(|v| usize::try_from(v.max(1)).unwrap_or(DEFAULT_MAX_PER_FILE));
                let max_per_file = match mode {
                    GrepOutputMode::Content => {
                        explicit_per_file.or_else(|| is_dir.then_some(DEFAULT_MAX_PER_FILE))
                    }
                    GrepOutputMode::FilesWithMatches | GrepOutputMode::Count => None,
                };

                let params = GrepParams {
                    pattern,
                    glob,
                    ignore_case,
                    literal,
                    context_lines: if mode == GrepOutputMode::Content {
                        context_lines
                    } else {
                        0
                    },
                    limit,
                    max_per_file,
                    mode,
                };

                let outcome = if let Some(rg_bin) = resolve_managed_tool(&["rg"], "rg") {
                    match run_rg_grep(&rg_bin, &search_path, is_dir, &params, &call_ctx.cancel)
                        .await?
                    {
                        Some(res) => res,
                        None => {
                            run_rust_grep(search_path, is_dir, params, call_ctx.cancel.clone())
                                .await?
                        }
                    }
                } else {
                    run_rust_grep(search_path, is_dir, params, call_ctx.cancel.clone()).await?
                };

                if outcome.matches == 0 {
                    return Ok(ToolOutput::text("No matches found").with_structured(json!({
                        "mode": mode.as_str(),
                        "matches": 0,
                        "output": "",
                        "backend": outcome.backend,
                    })));
                }

                let raw_joined = outcome.output_lines.join("\n");
                let trunc = truncate_head(&raw_joined, usize::MAX, DEFAULT_MAX_BYTES);
                let mut text = trunc.content.clone();
                let mut notices = Vec::new();
                if outcome.limit_reached {
                    notices.push(format!(
                        "{limit} results limit reached. Use limit={} for more, or refine pattern",
                        limit * 2
                    ));
                }
                if outcome.per_file_limit_reached
                    && let Some(cap) = max_per_file
                {
                    notices.push(format!(
                        "Per-file limit of {cap} matches reached in some files. Pass maxPerFile={} or search a single file to see more",
                        cap * 2
                    ));
                }
                let truncated = trunc.truncated || outcome.bytes_overflowed;
                if truncated {
                    notices.push(format!("{} limit reached", format_size(DEFAULT_MAX_BYTES)));
                }
                if outcome.lines_truncated {
                    notices.push(format!(
                        "Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
                    ));
                }
                if !notices.is_empty() {
                    text.push_str(&format!("\n\n[{}]", notices.join(". ")));
                }

                Ok(ToolOutput::text(text)
                    .with_structured(json!({
                        "mode": mode.as_str(),
                        "matches": outcome.matches,
                        "limit_reached": outcome.limit_reached,
                        "per_file_limit_reached": outcome.per_file_limit_reached,
                        "truncated": truncated,
                        "lines_shown": trunc.output_lines,
                        "total_lines": outcome.total_line_count,
                        "backend": outcome.backend,
                    }))
                    .with_details(json!({
                        "mode": mode.as_str(),
                        "matches": outcome.matches,
                        "limit_reached": outcome.limit_reached,
                        "per_file_limit_reached": outcome.per_file_limit_reached,
                        "backend": outcome.backend,
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
            "mode": { "type": "string" },
            "matches": { "type": "integer" },
            "limit_reached": { "type": "boolean" },
            "per_file_limit_reached": { "type": "boolean" },
            "truncated": { "type": "boolean" },
            "backend": { "type": "string" }
        }
    }))
    .with_prompt_snippet("Search file contents by regex or literal pattern")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_tools::{ToolCallContext, ToolExecution};

    fn test_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mizar_w13_grep_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn call_ctx(args: &Value) -> ToolCallContext {
        ToolCallContext {
            exec: ToolExecution::new("grep", args.clone(), None),
            cancel: CancellationToken::new(),
            on_update: None,
            ctx: cordis::Context::new(),
        }
    }

    fn make_exec(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perm = std::fs::metadata(path).unwrap().permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(path, perm).unwrap();
        }
    }

    #[test]
    fn resolves_three_tiers_path_then_mizar_bin_then_none() {
        let root = test_dir("resolve");
        let path_dir = root.join("usr_bin");
        let mizar_dir = root.join("mizar_bin");
        std::fs::create_dir_all(&path_dir).unwrap();
        std::fs::create_dir_all(&mizar_dir).unwrap();

        // 1. 都没有 -> None
        let empty_path = std::env::join_paths([&path_dir]).unwrap();
        assert_eq!(
            resolve_tool_with(Some(&empty_path), Some(&mizar_dir), &["rg"], "rg"),
            None
        );

        // 2. 只有 ~/.mizar/bin 有 -> 返回 ~/.mizar/bin/rg
        let managed_rg = mizar_dir.join("rg");
        make_exec(&managed_rg);
        assert_eq!(
            resolve_tool_with(Some(&empty_path), Some(&mizar_dir), &["rg"], "rg"),
            Some(managed_rg.clone())
        );

        // 3. PATH 也有 -> PATH 优先
        let user_rg = path_dir.join("rg");
        make_exec(&user_rg);
        assert_eq!(
            resolve_tool_with(Some(&empty_path), Some(&mizar_dir), &["rg"], "rg"),
            Some(user_rg)
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn rust_fallback_supports_all_modes_gitignore_glob_context_and_caps() {
        let root = test_dir("rust_fallback");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(root.join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(root.join("bin.dat"), b"needle\x00binary").unwrap();
        std::fs::write(
            root.join("src/a.rs"),
            "before\nNEEDLE_one\nafter\nneedle_two\nafter_two\nneedle_three\n",
        )
        .unwrap();
        std::fs::write(root.join("src/b.txt"), "needle_in_txt\n").unwrap();

        // content mode + ignoreCase + context + maxPerFile + glob
        let content_res = run_rust_grep(
            root.clone(),
            true,
            GrepParams {
                pattern: "needle_".into(),
                glob: Some("*.rs".into()),
                ignore_case: true,
                literal: true,
                context_lines: 1,
                limit: 100,
                max_per_file: Some(2),
                mode: GrepOutputMode::Content,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(content_res.backend, "rust");
        assert_eq!(content_res.matches, 2);
        assert!(content_res.per_file_limit_reached);
        let joined = content_res.output_lines.join("\n");
        assert!(joined.contains("src/a.rs:2: NEEDLE_one"));
        assert!(joined.contains("src/a.rs-1- before"));
        assert!(joined.contains("src/a.rs:4: needle_two"));
        assert!(!joined.contains("needle_three"));
        assert!(!joined.contains("ignored.txt"));
        assert!(!joined.contains("b.txt"));
        assert!(!joined.contains("bin.dat"));

        // files_with_matches mode
        let files_res = run_rust_grep(
            root.clone(),
            true,
            GrepParams {
                pattern: "needle_".into(),
                glob: None,
                ignore_case: true,
                literal: false,
                context_lines: 0,
                limit: 100,
                max_per_file: None,
                mode: GrepOutputMode::FilesWithMatches,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(files_res.output_lines, vec!["src/a.rs", "src/b.txt"]);

        // count mode
        let count_res = run_rust_grep(
            root.clone(),
            true,
            GrepParams {
                pattern: "needle_".into(),
                glob: None,
                ignore_case: true,
                literal: false,
                context_lines: 0,
                limit: 100,
                max_per_file: None,
                mode: GrepOutputMode::Count,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(count_res.output_lines, vec!["src/a.rs:3", "src/b.txt:1"]);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn grep_tool_supports_output_modes_and_rejects_invalid_mode() {
        let root = test_dir("tool_modes");
        std::fs::write(root.join("one.rs"), "hit 1\nhit 2\nhit 3\n").unwrap();
        std::fs::write(root.join("two.rs"), "hit only\n").unwrap();

        let tool = create_grep_tool(root.clone(), true);

        let args_files = json!({ "pattern": "hit", "output_mode": "files_with_matches" });
        let files_out = (tool.execute)(args_files.clone(), call_ctx(&args_files))
            .await
            .unwrap();
        assert!(!files_out.is_error);
        assert!(files_out.content.contains("one.rs"));
        assert!(files_out.content.contains("two.rs"));
        assert!(!files_out.content.contains("hit 1"));

        let args_count = json!({ "pattern": "hit", "output_mode": "count" });
        let count_out = (tool.execute)(args_count.clone(), call_ctx(&args_count))
            .await
            .unwrap();
        assert!(!count_out.is_error);
        assert!(count_out.content.contains("one.rs:3"));
        assert!(count_out.content.contains("two.rs:1"));

        let args_per_file = json!({ "pattern": "hit", "maxPerFile": 1 });
        let per_file_out = (tool.execute)(args_per_file.clone(), call_ctx(&args_per_file))
            .await
            .unwrap();
        assert!(
            per_file_out
                .content
                .contains("Per-file limit of 1 matches reached")
        );

        let args_invalid = json!({ "pattern": "hit", "output_mode": "invalid_mode" });
        let invalid = (tool.execute)(args_invalid.clone(), call_ctx(&args_invalid))
            .await
            .unwrap_err();
        assert!(invalid.to_string().contains("Invalid output_mode"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
