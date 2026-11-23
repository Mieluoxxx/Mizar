//! 输出截断与临时文件落盘工具（对齐 `pi` 的 `truncate.ts` 规则：2000 行 / 50KB 阈值，按 UTF-8 与行边界安全截断）。

use std::path::PathBuf;

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const GREP_MAX_LINE_LENGTH: usize = 500;
pub const BASH_MAX_LINES: usize = 400;
pub const BASH_MAX_BYTES: usize = 20 * 1024;
pub const BASH_MAX_LINE_BYTES: usize = 768;
pub const READ_MAX_LINE_CHARS: usize = 2000;

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

/// 截断单行过长文本（按字节数），保证输出总字节数不超过 `max_bytes` 且落在合法 UTF-8 边界。
pub fn truncate_line_bytes(line: &str, max_bytes: usize) -> (String, bool) {
    if line.len() <= max_bytes {
        return (line.to_string(), false);
    }
    const SUFFIX: &str = "... [truncated]";
    if max_bytes <= SUFFIX.len() {
        return (slice_utf8_head(line, max_bytes).to_string(), true);
    }
    let prefix = slice_utf8_head(line, max_bytes - SUFFIX.len());
    (format!("{prefix}{SUFFIX}"), true)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadTailTruncateResult {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub lines_capped: bool,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub head_lines: usize,
    pub tail_lines: usize,
    pub omitted_lines: usize,
    pub output_bytes: usize,
}

/// 头部 + 尾部双向截断（中间省略）：先按单行最大字节数收紧行宽，再按约 40% 头部 / 60% 尾部预算保留上下文。
pub fn truncate_head_tail(
    content: &str,
    max_lines: usize,
    max_bytes: usize,
    max_line_bytes: usize,
) -> HeadTailTruncateResult {
    let total_bytes = content.len();
    if content.is_empty() {
        return HeadTailTruncateResult {
            content: String::new(),
            truncated: false,
            truncated_by: None,
            lines_capped: false,
            total_lines: 0,
            total_bytes: 0,
            head_lines: 0,
            tail_lines: 0,
            omitted_lines: 0,
            output_bytes: 0,
        };
    }

    // 剥离单个末尾换行符以对齐逻辑行数（例如 `seq 1 2100` 共 2100 行，末行为 "2100"）。
    let logical = content.strip_suffix('\n').unwrap_or(content);
    let raw_lines: Vec<&str> = logical.split('\n').collect();
    let total_lines = raw_lines.len();

    let mut capped_lines = Vec::with_capacity(total_lines);
    let mut lines_capped = false;
    let mut capped_total_bytes = 0usize;
    for (idx, raw) in raw_lines.iter().enumerate() {
        let (capped, was_capped) = truncate_line_bytes(raw, max_line_bytes);
        if was_capped {
            lines_capped = true;
        }
        capped_total_bytes += capped.len() + usize::from(idx > 0);
        capped_lines.push(capped);
    }

    if total_lines <= max_lines && capped_total_bytes <= max_bytes {
        let joined = capped_lines.join("\n");
        let output_bytes = joined.len();
        return HeadTailTruncateResult {
            content: joined,
            truncated: lines_capped,
            truncated_by: lines_capped.then_some(TruncatedBy::Bytes),
            lines_capped,
            total_lines,
            total_bytes,
            head_lines: total_lines,
            tail_lines: 0,
            omitted_lines: 0,
            output_bytes,
        };
    }

    let eff_max_lines = max_lines.max(2);
    let head_line_budget = (eff_max_lines * 2 / 5).max(1);
    let tail_line_budget = eff_max_lines.saturating_sub(head_line_budget).max(1);

    const MARKER_RESERVE: usize = 96;
    let usable_bytes = max_bytes.saturating_sub(MARKER_RESERVE);
    let head_byte_budget = (usable_bytes * 2 / 5).max(1);
    let tail_byte_budget = usable_bytes.saturating_sub(head_byte_budget).max(1);

    let mut head: Vec<&str> = Vec::new();
    let mut head_bytes = 0usize;
    let mut head_stopped_by_bytes = false;
    for line in &capped_lines {
        if head.len() >= head_line_budget {
            break;
        }
        let add = line.len() + usize::from(!head.is_empty());
        if head_bytes + add > head_byte_budget {
            if head.is_empty() {
                head.push(slice_utf8_head(line, head_byte_budget));
            }
            head_stopped_by_bytes = true;
            break;
        }
        head.push(line.as_str());
        head_bytes += add;
    }

    let head_count = head.len();
    let mut tail: Vec<&str> = Vec::new();
    let mut tail_bytes = 0usize;
    let mut tail_stopped_by_bytes = false;
    for line in capped_lines[head_count..].iter().rev() {
        if tail.len() >= tail_line_budget {
            break;
        }
        let add = line.len() + usize::from(!tail.is_empty());
        if tail_bytes + add > tail_byte_budget {
            if tail.is_empty() {
                tail.push(slice_utf8_tail(line, tail_byte_budget));
            }
            tail_stopped_by_bytes = true;
            break;
        }
        tail.push(line.as_str());
        tail_bytes += add;
    }
    tail.reverse();

    let tail_count = tail.len();
    let omitted_lines = total_lines.saturating_sub(head_count + tail_count);

    let mut parts: Vec<String> = Vec::with_capacity(head_count + tail_count + 1);
    parts.extend(head.into_iter().map(str::to_string));
    if omitted_lines > 0 {
        parts.push(format!("... [{omitted_lines} lines omitted] ..."));
    }
    parts.extend(tail.into_iter().map(str::to_string));

    let joined = parts.join("\n");
    let output_bytes = joined.len();
    let truncated_by = if (head_stopped_by_bytes || tail_stopped_by_bytes)
        && (head_count + tail_count < eff_max_lines)
    {
        Some(TruncatedBy::Bytes)
    } else {
        Some(TruncatedBy::Lines)
    };

    HeadTailTruncateResult {
        content: joined,
        truncated: true,
        truncated_by,
        lines_capped,
        total_lines,
        total_bytes,
        head_lines: head_count,
        tail_lines: tail_count,
        omitted_lines,
        output_bytes,
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

/// 创建用于流式落盘的唯一临时输出文件。
pub async fn create_temp_output_file(prefix: &str) -> std::io::Result<(PathBuf, tokio::fs::File)> {
    let dir = std::env::temp_dir().join("mizar");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{prefix}{}.log", uuid::Uuid::new_v4()));
    let file = tokio::fs::File::create(&path).await?;
    Ok((path, file))
}

/// 把完整输出落盘到系统临时目录，供截断提示引用。
pub async fn write_temp_output(prefix: &str, content: &str) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("mizar");
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

    #[test]
    fn truncate_line_bytes_preserves_utf8_and_budget() {
        let (same, hit_same) = truncate_line_bytes("hello", 10);
        assert!(!hit_same);
        assert_eq!(same, "hello");

        let utf8_line = "中".repeat(400);
        let (capped, hit) = truncate_line_bytes(&utf8_line, BASH_MAX_LINE_BYTES);
        assert!(hit);
        assert!(capped.len() <= BASH_MAX_LINE_BYTES);
        assert!(capped.ends_with("... [truncated]"));

        let (tiny, hit_tiny) = truncate_line_bytes(&utf8_line, 5);
        assert!(hit_tiny);
        assert!(tiny.len() <= 5);
    }

    #[test]
    fn truncate_head_tail_omits_middle_on_lines_and_bytes() {
        // 无截断场景
        let small = "a\nb\nc\n";
        let res_small = truncate_head_tail(small, 400, 20 * 1024, 768);
        assert!(!res_small.truncated);
        assert_eq!(res_small.total_lines, 3);
        assert_eq!(res_small.content, "a\nb\nc");

        // 行数超限：保留头部与尾部，中间插入省略标记
        let many_lines = (1..=600)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let res_lines = truncate_head_tail(&many_lines, 400, 20 * 1024, 768);
        assert!(res_lines.truncated);
        assert_eq!(res_lines.truncated_by, Some(TruncatedBy::Lines));
        assert!(res_lines.omitted_lines >= 200);
        assert!(res_lines.content.contains("line-1\n"));
        assert!(res_lines.content.contains("lines omitted] ..."));
        assert!(res_lines.content.ends_with("line-600"));

        // 字节超限 + 行宽超限
        let wide_lines = (1..=50)
            .map(|i| format!("{i:02}:{}", "x".repeat(1200)))
            .collect::<Vec<_>>()
            .join("\n");
        let res_bytes = truncate_head_tail(&wide_lines, 400, 8 * 1024, BASH_MAX_LINE_BYTES);
        assert!(res_bytes.truncated);
        assert!(res_bytes.lines_capped);
        assert_eq!(res_bytes.truncated_by, Some(TruncatedBy::Bytes));
        assert!(res_bytes.output_bytes <= 8 * 1024);
        assert!(res_bytes.content.contains("01:"));
        assert!(res_bytes.content.contains("50:"));
    }
}
