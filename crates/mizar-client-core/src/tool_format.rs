//! 客户端纯展示格式化与摘要工具函数（参数/结果摘要、家目录缩写、Diff 统计、Token 格式化等）。
//!
//! 迁移自 `mizar-tui-ui/src/app.rs`，供 TUI 与 Merak 共用。

use serde_json::Value;
use unicode_width::UnicodeWidthChar;

use crate::types::ToolExtraMeta;

pub const MAX_ARGS_SUMMARY_COLS: usize = 512;
pub const MAX_PREVIEW_LINE_COLS: usize = 240;

pub fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

pub fn format_tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}k", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

pub fn truncate_display(s: &str, max_cols: usize) -> String {
    let mut out = String::new();
    let mut width = 0usize;
    for ch in s.chars() {
        let ch = if ch == '\n' || ch == '\r' { ' ' } else { ch };
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > max_cols.saturating_sub(1) {
            out.push('…');
            return out;
        }
        out.push(ch);
        width += w;
    }
    out
}

pub fn single_line_text(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_home_prefix_boundary(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '"' | '\'' | '=' | '(' | ',' | ':' | ';' | '`')
}

fn is_home_suffix_boundary(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '/' | '\\' | '"' | '\'' | ')' | ',' | ':' | ';' | '&' | '|' | '`'
        )
}

/// 将文本中的用户家目录绝对路径缩写为 `~`（带边界检查，避免误替换同前缀用户名路径）。
pub fn abbreviate_home_with(text: &str, home: Option<&str>) -> String {
    let Some(raw_home) = home else {
        return text.to_string();
    };
    let home_trimmed = raw_home.trim_end_matches(['/', '\\']);
    if home_trimmed.is_empty() || home_trimmed == "/" {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut pos = 0usize;
    while pos < text.len() {
        let Some(rel) = text[pos..].find(home_trimmed) else {
            out.push_str(&text[pos..]);
            break;
        };
        let match_start = pos + rel;
        let match_end = match_start + home_trimmed.len();
        out.push_str(&text[pos..match_start]);

        let left_ok = match_start == 0
            || text[..match_start]
                .chars()
                .next_back()
                .is_some_and(is_home_prefix_boundary);
        let right_ok = match_end == text.len()
            || text[match_end..]
                .chars()
                .next()
                .is_some_and(is_home_suffix_boundary);

        if left_ok && right_ok {
            out.push('~');
            pos = match_end;
        } else if let Some(ch) = text[match_start..].chars().next() {
            out.push(ch);
            pos = match_start + ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// 使用当前系统家目录将路径或命令中的 `$HOME` 缩写为 `~`。
pub fn abbreviate_home_path(text: &str) -> String {
    let home = dirs::home_dir();
    let home_str = home.as_ref().and_then(|p| p.to_str());
    abbreviate_home_with(text, home_str)
}

/// 判断是否为只读/检索/发现类工具（`Done` 态在默认折叠模式下单行收口）。
pub fn is_read_only_compact_tool(name: &str) -> bool {
    matches!(
        name,
        "read"
            | "read_file"
            | "list_dir"
            | "ls"
            | "grep"
            | "find"
            | "tool_search"
            | "skill"
            | "list_mcp_resources"
            | "list_mcp_resource_templates"
            | "read_mcp_resource"
            | "get_time"
    )
}

pub fn bash_scope_label(exclude_from_context: bool) -> &'static str {
    if exclude_from_context {
        "仅本地"
    } else {
        "已入上下文"
    }
}

pub fn bash_inline_meta(
    exclude_from_context: bool,
    exit_code: Option<i32>,
    cancelled: bool,
) -> String {
    let scope = bash_scope_label(exclude_from_context);
    if cancelled {
        format!("{scope} · 已取消")
    } else if let Some(code) = exit_code
        && code != 0
    {
        format!("{scope} · 退出码 {code}")
    } else {
        scope.to_string()
    }
}

pub fn format_bash_display(output: &str, exit_code: Option<i32>, cancelled: bool) -> String {
    let mut display = if output.trim().is_empty() {
        "(no output)".to_string()
    } else {
        output.to_string()
    };
    if cancelled {
        display.push_str("\n\nCommand aborted");
    } else if let Some(code) = exit_code
        && code != 0
    {
        display.push_str(&format!("\n\nCommand exited with code {code}"));
    }
    display
}

/// 判断工具的主要参数是否为路径（窄宽度时优先保留首部前缀与尾部文件名）。
pub fn is_path_arg_tool(name: &str) -> bool {
    matches!(
        name,
        "read" | "read_file" | "list_dir" | "ls" | "edit" | "write" | "read_mcp_resource"
    )
}

pub fn count_diff_stats(diff: &str) -> (usize, usize) {
    let mut adds = 0usize;
    let mut dels = 0usize;
    for line in diff.lines() {
        if line.starts_with("+++ ") || line.starts_with("--- ") {
            continue;
        }
        if line.starts_with('+') {
            adds += 1;
        } else if line.starts_with('-') {
            dels += 1;
        }
    }
    (adds, dels)
}

pub fn format_diff_stat(adds: usize, dels: usize) -> Option<String> {
    if adds == 0 && dels == 0 {
        None
    } else {
        Some(format!("+{adds} -{dels}"))
    }
}

fn plural_unit(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("1 {singular}")
    } else {
        format!("{count} {plural}")
    }
}

pub fn clean_bash_output_lines(content: &str, keep_error_footer: bool) -> Vec<&str> {
    content
        .lines()
        .map(str::trim_end)
        .filter(|line| {
            let t = line.trim();
            if t.is_empty()
                || t == "(no output)"
                || t.starts_with("[Showing lines ")
                || t.starts_with("[Showing last ")
            {
                return false;
            }
            if !keep_error_footer
                && (t == "Command aborted"
                    || t.starts_with("Command timed out after ")
                    || t.starts_with("Command exited with code "))
            {
                return false;
            }
            true
        })
        .collect()
}

fn value_inline(val: &Value) -> String {
    match val {
        Value::String(s) => single_line_text(s),
        other => other.to_string(),
    }
}

/// 为展开态（`Ctrl+O`）解析完整参数与附带脚本/写入正文。
pub fn format_expanded_tool_args(
    name: &str,
    args_summary: &str,
    extra: Option<&ToolExtraMeta>,
) -> (String, Vec<String>) {
    let Some(raw) = extra
        .map(|e| e.full_args.trim())
        .filter(|s| !s.is_empty() && *s != "{}")
    else {
        return (args_summary.to_string(), Vec::new());
    };

    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(raw) else {
        return (single_line_text(raw), Vec::new());
    };

    if name == "bash"
        && let Some(cmd) = map.get("command").and_then(Value::as_str)
    {
        if cmd.contains('\n') {
            let mut extra_lines = Vec::new();
            for (i, line) in cmd.lines().enumerate() {
                if i == 0 {
                    extra_lines.push(format!("$ {line}"));
                } else {
                    extra_lines.push(format!("  {line}"));
                }
            }
            let first = single_line_text(cmd.lines().next().unwrap_or(""));
            return (first, extra_lines);
        }
        return (cmd.trim().to_string(), Vec::new());
    }

    if name == "codemode"
        && let Some(code) = map.get("code").and_then(Value::as_str)
    {
        let n = code.lines().count().max(1);
        let extra_lines: Vec<String> = code.lines().map(ToString::to_string).collect();
        return (format!("{n} lines"), extra_lines);
    }

    if name == "write"
        && let Some(path) = map.get("path").and_then(Value::as_str)
    {
        let extra_lines = map
            .get("content")
            .and_then(Value::as_str)
            .map(|c| c.lines().map(ToString::to_string).collect())
            .unwrap_or_default();
        return (path.to_string(), extra_lines);
    }

    if name == "edit"
        && let Some(path) = map.get("path").and_then(Value::as_str)
    {
        return (path.to_string(), Vec::new());
    }

    let mut pairs = Vec::new();
    for (k, v) in &map {
        pairs.push(format!("{k}={}", value_inline(v)));
    }
    if map.len() == 1
        && let Some(val) = map.values().next()
    {
        return (value_inline(val), Vec::new());
    }
    (pairs.join(", "), Vec::new())
}

pub fn summarize_tool_args(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return String::new();
    }
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
        if let Some(code) = map.get("code").and_then(Value::as_str) {
            let n = code.lines().count().max(1);
            let first = single_line_text(code.lines().next().unwrap_or(""));
            let summary = abbreviate_home_path(&format!("{n} lines: {first}"));
            return truncate_display(&summary, MAX_ARGS_SUMMARY_COLS);
        }
        if let (Some(pat), Some(path_val)) = (map.get("pattern"), map.get("path")) {
            let pat_s = value_inline(pat);
            let path_s = abbreviate_home_path(&value_inline(path_val));
            let combined = if path_s.is_empty() || path_s == "." {
                pat_s
            } else {
                format!("{pat_s} in {path_s}")
            };
            return truncate_display(&combined, MAX_ARGS_SUMMARY_COLS);
        }
        for priority_key in [
            "path", "file", "command", "query", "pattern", "uri", "server", "name",
        ] {
            if let Some(val) = map.get(priority_key) {
                let mut text = abbreviate_home_path(&value_inline(val));
                if priority_key == "path"
                    && map.get("edits").is_none()
                    && map.get("content").is_none()
                {
                    let off = map.get("offset").and_then(Value::as_u64);
                    let lim = map.get("limit").and_then(Value::as_u64);
                    text = match (off, lim) {
                        (Some(o), Some(l)) => {
                            format!("{text}:{o}-{}", o.saturating_add(l.saturating_sub(1)))
                        }
                        (Some(o), None) => format!("{text}:{o}+"),
                        (None, Some(l)) => format!("{text}:1-{l}"),
                        (None, None) => text,
                    };
                }
                return truncate_display(&text, MAX_ARGS_SUMMARY_COLS);
            }
        }
        let pairs: Vec<String> = map
            .iter()
            .take(3)
            .map(|(k, v)| format!("{k}={}", abbreviate_home_path(&value_inline(v))))
            .collect();
        return truncate_display(&pairs.join(", "), MAX_ARGS_SUMMARY_COLS);
    }
    let cleaned = abbreviate_home_path(&single_line_text(trimmed));
    truncate_display(&cleaned, MAX_ARGS_SUMMARY_COLS)
}

pub fn summarize_live_preview(preview: &str) -> String {
    let lines: Vec<&str> = preview
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let start = lines.len().saturating_sub(2);
    lines[start..]
        .iter()
        .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn extract_inline_meta(
    name: &str,
    content: &str,
    details: Option<&Value>,
    is_error: bool,
) -> Option<String> {
    if is_error {
        if name == "bash"
            && let Some(d) = details
        {
            if d.get("cancelled").and_then(Value::as_bool) == Some(true) {
                return Some("cancelled".to_string());
            }
            if d.get("timed_out").and_then(Value::as_bool) == Some(true) {
                return Some("timeout".to_string());
            }
            if let Some(code) = d.get("exit_code").and_then(Value::as_i64)
                && code != 0
            {
                return Some(format!("exit {code}"));
            }
        }
        return None;
    }

    match name {
        "read" => {
            if let Some(d) = details
                && let Some(total) = d.get("total_lines").and_then(Value::as_u64)
            {
                if total == 0 {
                    return Some("empty".to_string());
                }
                let start = d.get("start_line").and_then(Value::as_u64).unwrap_or(1);
                let end = d.get("end_line").and_then(Value::as_u64).unwrap_or(total);
                let truncated = d.get("truncated").and_then(Value::as_bool).unwrap_or(false);
                if truncated || start > 1 || end < total {
                    return Some(format!("L{start}-{end}/{total}"));
                }
                return Some(plural_unit(total as usize, "line", "lines"));
            }
            if content.trim().is_empty() || content.trim() == "(empty file)" {
                Some("empty".to_string())
            } else {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            }
        }
        "read_file"
        | "read_mcp_resource"
        | "list_mcp_resources"
        | "list_mcp_resource_templates" => {
            if content.trim().is_empty() {
                Some("empty".to_string())
            } else {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            }
        }
        "list_dir" | "ls" => {
            let trimmed = content.trim();
            if trimmed.is_empty() || trimmed == "（空目录）" {
                Some("empty".to_string())
            } else {
                let n = content.lines().filter(|l| !l.trim().is_empty()).count();
                Some(plural_unit(n, "item", "items"))
            }
        }
        "grep" => {
            if let Some(d) = details
                && let Some(m) = d.get("matches").and_then(Value::as_u64)
            {
                if m == 0 {
                    return Some("0 matches".to_string());
                }
                let limited = d
                    .get("limit_reached")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if limited {
                    return Some(format!("{m}+ matches"));
                }
                return Some(plural_unit(m as usize, "match", "matches"));
            }
            if content.trim_start().starts_with("No matches") {
                Some("0 matches".to_string())
            } else if !content.trim().is_empty() {
                Some(plural_unit(content.lines().count(), "line", "lines"))
            } else {
                None
            }
        }
        "find" => {
            if let Some(d) = details
                && let Some(c) = d.get("count").and_then(Value::as_u64)
            {
                if c == 0 {
                    return Some("0 files".to_string());
                }
                let limited = d
                    .get("limit_reached")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if limited {
                    return Some(format!("{c}+ files"));
                }
                return Some(plural_unit(c as usize, "file", "files"));
            }
            if content.trim_start().starts_with("No files found") {
                Some("0 files".to_string())
            } else if !content.trim().is_empty() {
                let n = content.lines().filter(|l| !l.trim().is_empty()).count();
                Some(plural_unit(n, "file", "files"))
            } else {
                None
            }
        }
        "tool_search" => details
            .and_then(|d| d.get("loaded"))
            .and_then(Value::as_array)
            .map(|arr| format!("{} loaded", arr.len())),
        "skill" => Some("loaded".to_string()),
        "edit" => details
            .and_then(|d| d.get("diff"))
            .and_then(Value::as_str)
            .and_then(|diff| {
                let (adds, dels) = count_diff_stats(diff);
                format_diff_stat(adds, dels)
            }),
        "write" => details
            .and_then(|d| d.get("lines"))
            .and_then(Value::as_u64)
            .map(|n| plural_unit(n as usize, "line", "lines")),
        "bash" => {
            let lines = clean_bash_output_lines(content, false);
            if lines.len() > 2 {
                Some(plural_unit(lines.len(), "line", "lines"))
            } else {
                None
            }
        }
        "codemode" => details
            .and_then(|d| d.get("nested_calls"))
            .and_then(Value::as_array)
            .filter(|arr| !arr.is_empty())
            .map(|arr| plural_unit(arr.len(), "call", "calls")),
        "get_time" => {
            let t = content.trim();
            if !t.is_empty() && !t.contains('\n') {
                Some(t.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn summarize_tool_result(
    name: &str,
    content: &str,
    details: Option<&Value>,
    is_error: bool,
) -> String {
    if is_error {
        if name == "bash" {
            let lines = clean_bash_output_lines(content, true);
            let chosen = if lines.is_empty() {
                content
                    .lines()
                    .map(str::trim_end)
                    .filter(|l| !l.trim().is_empty())
                    .collect::<Vec<_>>()
            } else {
                lines
            };
            let start = chosen.len().saturating_sub(4);
            return chosen[start..]
                .iter()
                .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                .collect::<Vec<_>>()
                .join("\n");
        }
        return content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(4)
            .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
            .collect::<Vec<_>>()
            .join("\n");
    }

    match name {
        "skill" | "write" => String::new(),
        "edit" => {
            if let Some(diff) = details.and_then(|d| d.get("diff")).and_then(Value::as_str) {
                let lines: Vec<String> = diff
                    .lines()
                    .filter(|l| {
                        !l.starts_with("--- ")
                            && !l.starts_with("+++ ")
                            && (l.starts_with('+') || l.starts_with('-'))
                    })
                    .take(8)
                    .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                    .collect();
                if !lines.is_empty() {
                    return lines.join("\n");
                }
            }
            String::new()
        }
        "bash" => {
            let lines = clean_bash_output_lines(content, false);
            let start = lines.len().saturating_sub(2);
            lines[start..]
                .iter()
                .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
                .collect::<Vec<_>>()
                .join("\n")
        }
        "codemode" => {
            let mut out = Vec::new();
            if let Some(calls) = details
                .and_then(|d| d.get("nested_calls"))
                .and_then(Value::as_array)
                && !calls.is_empty()
            {
                let mut parts = Vec::new();
                for c in calls.iter().take(3) {
                    let status = c.get("status").and_then(Value::as_str).unwrap_or("✓");
                    let cname = c.get("name").and_then(Value::as_str).unwrap_or("?");
                    parts.push(format!("{status} {cname}"));
                }
                let mut line = parts.join(", ");
                if calls.len() > 3 {
                    line.push_str(&format!(" … ({} calls)", calls.len()));
                }
                out.push(truncate_display(&line, MAX_PREVIEW_LINE_COLS));
            }
            for l in content.lines().filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with("Nested tool calls:")
                    && !t.starts_with("[Truncated to ")
            }) {
                if out.len() >= 2 {
                    break;
                }
                out.push(truncate_display(l, MAX_PREVIEW_LINE_COLS));
            }
            out.join("\n")
        }
        _ => content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(2)
            .map(|l| truncate_display(l, MAX_PREVIEW_LINE_COLS))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviate_home_with_handles_paths_commands_and_prefix_boundaries() {
        let home = Some("/Users/alice");
        assert_eq!(abbreviate_home_with("/Users/alice", home), "~");
        assert_eq!(
            abbreviate_home_with("/Users/alice/workspace/mizar/src/lib.rs", home),
            "~/workspace/mizar/src/lib.rs"
        );
        assert_eq!(
            abbreviate_home_with(
                "cd /Users/alice/workspace && cat \"/Users/alice/.bashrc\"",
                home
            ),
            "cd ~/workspace && cat \"~/.bashrc\""
        );
        // 不能误替换同前缀用户名或非路径边界子串
        assert_eq!(
            abbreviate_home_with("/Users/alice_other/foo", home),
            "/Users/alice_other/foo"
        );
        assert_eq!(
            abbreviate_home_with("/tmp/Users/alice/foo", home),
            "/tmp/Users/alice/foo"
        );
    }

    #[test]
    fn summarize_tool_args_preserves_long_commands_ranges_and_grep_context() {
        let long_cmd =
            "cd /tmp/very-long-workspace-directory-name && cargo test --workspace --all-targets";
        let cmd_raw = format!(r#"{{"command":"{long_cmd}"}}"#);
        let cmd_summary = summarize_tool_args(&cmd_raw);
        assert_eq!(cmd_summary, long_cmd);
        assert!(cmd_summary.len() > 48);

        assert_eq!(
            summarize_tool_args(
                r#"{"path":"crates/mizar-tui-ui/src/ui.rs","offset":100,"limit":50}"#
            ),
            "crates/mizar-tui-ui/src/ui.rs:100-149"
        );
        assert_eq!(
            summarize_tool_args(r#"{"pattern":"build_tool_lines","path":"crates/mizar-tui-ui"}"#),
            "build_tool_lines in crates/mizar-tui-ui"
        );
        assert_eq!(summarize_tool_args(r#"{"pattern":"*","path":"."}"#), "*");
    }

    #[test]
    fn summarize_tool_result_and_extras_keep_full_args_and_compact_previews() {
        let raw_args =
            r#"{"command":"echo line1 && echo line2 && echo very_long_command_preserved"}"#;
        let extra = ToolExtraMeta {
            full_args: raw_args.into(),
            ..ToolExtraMeta::default()
        };
        let (expanded_head, expanded_lines) = format_expanded_tool_args("bash", "", Some(&extra));
        assert_eq!(
            expanded_head,
            "echo line1 && echo line2 && echo very_long_command_preserved"
        );
        assert!(expanded_lines.is_empty());

        let out = (1..=10)
            .map(|i| format!("out {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let details = serde_json::json!({ "exit_code": 0, "duration_ms": 15 });
        assert_eq!(
            extract_inline_meta("bash", &out, Some(&details), false).as_deref(),
            Some("10 lines")
        );
        assert_eq!(
            summarize_tool_result("bash", &out, Some(&details), false),
            "out 9\nout 10"
        );

        // bash 无输出或报错场景
        assert_eq!(
            summarize_tool_result("bash", "(no output)", None, false),
            ""
        );
        let err_preview = summarize_tool_result(
            "bash",
            "error[E0425]: missing\nCommand exited with code 1",
            Some(&serde_json::json!({ "exit_code": 1 })),
            true,
        );
        assert!(err_preview.contains("error[E0425]: missing"));
        assert_eq!(
            extract_inline_meta(
                "bash",
                "Command exited with code 2",
                Some(&serde_json::json!({ "exit_code": 2 })),
                true
            )
            .as_deref(),
            Some("exit 2")
        );
        assert_eq!(
            extract_inline_meta(
                "edit",
                "ok",
                Some(&serde_json::json!({ "diff": "--- a\n+++ b\n-old\n+new1\n+new2\n" })),
                false
            )
            .as_deref(),
            Some("+2 -1")
        );
    }
}
