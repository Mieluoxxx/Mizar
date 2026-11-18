//! 输出截断与临时文件落盘工具（对齐 `pi` 的 `truncate.ts` 规则：2000 行 / 50KB 阈值，按 UTF-8 与行边界安全截断）。

use std::path::PathBuf;

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const GREP_MAX_LINE_LENGTH: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncateResult {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
}

pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// 在不超过 `max_bytes` 的前提下返回合法 UTF-8 前缀切片。
pub fn slice_utf8_head(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// 在不超过 `max_bytes` 的前提下返回合法 UTF-8 后缀切片（用于超长末行截断）。
pub fn slice_utf8_tail(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut start = s.len().saturating_sub(max_bytes);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// 头部截断：用于 `read` / `grep` / `find`。绝不在多字节 UTF-8 或普通行中间切断。
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_bytes = content.len();
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncateResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
        };
    }

    if let Some(first) = lines.first()
        && first.len() > max_bytes
    {
        return TruncateResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
        };
    }

    let mut out = Vec::new();
    let mut bytes_used = 0usize;
    let mut truncated_by = TruncatedBy::Lines;

    for (idx, line) in lines.iter().enumerate() {
        if idx >= max_lines {
            truncated_by = TruncatedBy::Lines;
            break;
        }
        let add_bytes = line.len() + usize::from(idx > 0);
        if bytes_used + add_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        out.push(*line);
        bytes_used += add_bytes;
    }

    let output_lines = out.len();
    let joined = out.join("\n");
    let output_bytes = joined.len();

    TruncateResult {
        content: joined,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines,
        output_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
    }
}

/// 尾部截断：用于 `bash` 等命令输出，保留最近输出；若最后一行本身超过 `max_bytes` 则保留其合法 UTF-8 尾部。
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TruncateResult {
    let total_bytes = content.len();
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncateResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
        };
    }

    let mut selected: Vec<&str> = Vec::new();
    let mut bytes_used = 0usize;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if selected.len() >= max_lines {
            truncated_by = TruncatedBy::Lines;
            break;
        }
        let add_bytes = line.len() + usize::from(!selected.is_empty());
        if bytes_used + add_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if selected.is_empty() {
                let tail = slice_utf8_tail(line, max_bytes);
                selected.push(tail);
                last_line_partial = true;
            }
            break;
        }
        selected.push(*line);
        bytes_used += add_bytes;
    }

    selected.reverse();
    let output_lines = selected.len();
    let joined = selected.join("\n");
    let output_bytes = joined.len();

    TruncateResult {
        content: joined,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines,
        output_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
    }
}

/// 截断单行过长文本（按字符数），避免 minified 文件炸掉上下文。
pub fn truncate_line(line: &str, max_chars: usize) -> (String, bool) {
    let total_chars = line.chars().count();
    if total_chars <= max_chars {
        return (line.to_string(), false);
    }
    let prefix: String = line.chars().take(max_chars).collect();
    (format!("{prefix}... [truncated]"), true)
}

/// 把完整输出落盘到系统临时目录，供截断提示引用。
pub async fn write_temp_output(prefix: &str, content: &str) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("dsh-rs");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{prefix}{}.log", uuid::Uuid::new_v4()));
    tokio::fs::write(&path, content).await?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_head_preserves_utf8_and_line_boundaries() {
        let text = "第一行你好世界\n第二行abc\n第三行def";
        let res = truncate_head(text, 2, DEFAULT_MAX_BYTES);
        assert!(res.truncated);
        assert_eq!(res.truncated_by, Some(TruncatedBy::Lines));
        assert_eq!(res.output_lines, 2);
        assert_eq!(res.content, "第一行你好世界\n第二行abc");

        let by_bytes = truncate_head(text, 100, "第一行你好世界".len() + 2);
        assert!(by_bytes.truncated);
        assert_eq!(by_bytes.truncated_by, Some(TruncatedBy::Bytes));
        assert_eq!(by_bytes.content, "第一行你好世界");

        let first_too_long = truncate_head(text, 100, 4);
        assert!(first_too_long.first_line_exceeds_limit);
        assert_eq!(first_too_long.output_lines, 0);
    }

    #[test]
    fn truncate_tail_keeps_recent_lines_and_handles_huge_last_line() {
        let text = "line1\nline2\nline3\nline4";
        let res = truncate_tail(text, 2, DEFAULT_MAX_BYTES);
        assert!(res.truncated);
        assert_eq!(res.content, "line3\nline4");

        let huge_utf8 = format!("前缀{}", "中".repeat(20));
        let partial = truncate_tail(&huge_utf8, 10, 10);
        assert!(partial.truncated);
        assert!(partial.last_line_partial);
        assert!(partial.content.len() <= 10);
        assert!(partial.content.chars().all(|c| c == '中'));
    }

    #[test]
    fn truncate_line_caps_chars() {
        let (out, hit) = truncate_line(&"a".repeat(600), GREP_MAX_LINE_LENGTH);
        assert!(hit);
        assert!(out.ends_with("... [truncated]"));
    }
}
