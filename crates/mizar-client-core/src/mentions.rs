//! `@` 文件与目录引用解析、光标上下文识别、模糊评分与替换格式化。
//!
//! 融合 `dsh-TUI`（词界 `@token`、带空格引号 `@"..."`、`#L12-14` 行范围切片与 `strip-first` 解析）
//! 与 `pi`（中文标点与括号/反引号边界、目录连续下钻补全）两套语义。

use std::collections::HashSet;

pub const FILE_MENTIONS_SOURCE: &str = "file-mentions";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionToken {
    /// `@` 在原字符串中的 UTF-8 字节起始偏移。
    pub start: usize,
    /// 整个引用令牌（含引号与 `#L` 后缀）在原字符串中的独占结束字节偏移。
    pub end: usize,
    /// 去掉前导 `@`、引号与 `#L` 行后缀后的路径文本。
    pub path: String,
    /// 仅当解析出 `#L` 行范围后缀时存在：保留后缀的字面路径（用于后缀文件名的回退解析）。
    pub literal: Option<String>,
    /// 1-based 起始行号（含）。
    pub start_line: Option<usize>,
    /// 1-based 结束行号（含；单行形式等于 `start_line`）。
    pub end_line: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionCaret {
    /// `@` 在原字符串中的字节起始偏移。
    pub start: usize,
    /// 整个引用令牌的结束偏移（含闭合引号与 `#L` 后缀）。
    pub end: usize,
    /// 路径部分的结束偏移（位于 `#L` 后缀之前；未闭合引号时等于 `end`）。
    pub path_end: usize,
    /// 光标之前正在编辑的路径查询文本（不含 `@`、开引号及 `#L`）。
    pub query: String,
    /// 是否以 `@"` 引号形式开头。
    pub is_quoted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MentionCandidateKind {
    File,
    Directory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MentionCandidate {
    /// 相对工作区（或带 `~/`、`/` 前缀）的规范化显示路径；目录以 `/` 结尾。
    pub path: String,
    /// 末级文件或目录名。
    pub name: String,
    pub kind: MentionCandidateKind,
}

pub fn is_cjk_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。'
            | '．'
            | '：'
            | '；'
            | '！'
            | '？'
            | '（'
            | '）'
            | '［'
            | '］'
            | '｛'
            | '｝'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '、'
            | '…'
            | '—'
    )
}

pub fn is_opening_wrapper(ch: char) -> bool {
    matches!(ch, '(' | '[' | '{' | '<' | '`' | '"' | '\'')
}

pub fn matching_closer(opener: char) -> Option<char> {
    match opener {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '<' => Some('>'),
        '`' => Some('`'),
        '"' => Some('"'),
        '\'' => Some('\''),
        _ => None,
    }
}

/// 判断 `@` 之前的字符是否构成合法的 `@` 引用触发边界。
pub fn is_mention_boundary_char(ch: char) -> bool {
    ch.is_whitespace() || is_opening_wrapper(ch) || is_cjk_punctuation(ch)
}

fn is_unquoted_token_stop(ch: char) -> bool {
    ch.is_whitespace() || is_cjk_punctuation(ch)
}

/// 剥离末尾合法的 `#L7` 或 `#L7-9` 行范围后缀。
///
/// 对非法范围（如 `#L0`、`#L14-12`、空路径 `#L5`）返回 `None`，由调用方将其视为普通字面路径。
pub fn split_line_suffix(raw: &str) -> Option<(&str, usize, usize)> {
    let hash_pos = raw.rfind("#L")?;
    let path = &raw[..hash_pos];
    if path.is_empty() {
        return None;
    }
    let rest = &raw[hash_pos + 2..];
    if rest.is_empty() {
        return None;
    }
    let (start_s, end_s) = match rest.split_once('-') {
        Some((a, b)) => (a, Some(b)),
        None => (rest, None),
    };
    if !start_s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let start_line: usize = start_s.parse().ok()?;
    let end_line: usize = match end_s {
        Some(b) => {
            if b.is_empty() || !b.bytes().all(|ch| ch.is_ascii_digit()) {
                return None;
            }
            b.parse().ok()?
        }
        None => start_line,
    };
    if start_line < 1 || end_line < start_line {
        return None;
    }
    Some((path, start_line, end_line))
}

fn trim_unmatched_closer(
    text: &str,
    token_start: usize,
    mut end: usize,
    opener: Option<char>,
) -> usize {
    let Some(closer) = opener.and_then(matching_closer) else {
        return end;
    };
    let open_char = opener.unwrap();
    let slice = &text[token_start..end];
    if slice.ends_with(closer) {
        let should_strip = if open_char == closer {
            true
        } else {
            let opens = slice.chars().filter(|&c| c == open_char).count();
            let closes = slice.chars().filter(|&c| c == closer).count();
            closes > opens
        };
        if should_strip {
            end -= closer.len_utf8();
        }
    }
    end
}

fn scan_unquoted_end(text: &str, after_at: usize, opener: Option<char>) -> usize {
    let mut end = after_at;
    for (rel, ch) in text[after_at..].char_indices() {
        if is_unquoted_token_stop(ch) {
            break;
        }
        end = after_at + rel + ch.len_utf8();
    }
    trim_unmatched_closer(text, after_at, end, opener)
}

/// 从输入文本中按出现顺序提取所有去重后的 `@` 引用令牌。
pub fn extract_mentions(text: &str) -> Vec<MentionToken> {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();
    let mut index = 0usize;

    while index < text.len() {
        let Some(rel) = text[index..].find('@') else {
            break;
        };
        let at = index + rel;
        index = at + 1;

        let prev_char = text[..at].chars().next_back();
        if prev_char.is_some_and(|ch| !is_mention_boundary_char(ch)) {
            continue;
        }
        let opener = prev_char.filter(|&ch| is_opening_wrapper(ch));

        // 引号形式：@"my dir/a.ts" 或 @"my dir/a.ts"#L3-5
        if text[at + 1..].starts_with('"') {
            let body_start = at + 2;
            let line_end = text[body_start..]
                .find('\n')
                .map(|p| body_start + p)
                .unwrap_or(text.len());
            let Some(close_rel) = text[body_start..line_end].find('"') else {
                continue;
            };
            let close = body_start + close_rel;
            let quoted = &text[body_start..close];
            let mut end = close + 1;
            let mut line_range: Option<(usize, usize)> = None;
            let mut literal: Option<String> = None;

            if !quoted.is_empty() && text[end..].starts_with('#') {
                let tail_end = scan_unquoted_end(text, end, opener);
                let tail = &text[end..tail_end];
                let combined = format!("{quoted}{tail}");
                if let Some((stripped, s, e)) = split_line_suffix(&combined)
                    && stripped == quoted
                {
                    line_range = Some((s, e));
                    literal = Some(combined);
                    end = tail_end;
                }
            }

            let key = &text[at..end];
            if !quoted.is_empty() && seen.insert(key.to_string()) {
                tokens.push(MentionToken {
                    start: at,
                    end,
                    path: quoted.to_string(),
                    literal,
                    start_line: line_range.map(|r| r.0),
                    end_line: line_range.map(|r| r.1),
                });
            }
            index = end;
            continue;
        }

        // 无引号形式：@src/app.rs 或 @src/app.rs#L12-14
        let end = scan_unquoted_end(text, at + 1, opener);
        let raw = &text[at + 1..end];
        if raw.is_empty() || raw.starts_with('@') {
            continue;
        }
        if !seen.insert(raw.to_string()) {
            index = end;
            continue;
        }
        if let Some((stripped, s, e)) = split_line_suffix(raw) {
            tokens.push(MentionToken {
                start: at,
                end,
                path: stripped.to_string(),
                literal: Some(raw.to_string()),
                start_line: Some(s),
                end_line: Some(e),
            });
        } else {
            tokens.push(MentionToken {
                start: at,
                end,
                path: raw.to_string(),
                literal: None,
                start_line: None,
                end_line: None,
            });
        }
        index = end;
    }

    tokens
}

/// 识别光标当前正在编辑的 `@` 引用令牌（支持句子中间、`@"带空格路径"` 以及 `#L` 后缀保留）。
pub fn mention_at_caret(text: &str, cursor: usize) -> Option<MentionCaret> {
    if cursor > text.len() || !text.is_char_boundary(cursor) {
        return None;
    }
    let line_start = text[..cursor].rfind('\n').map(|p| p + 1).unwrap_or(0);
    let line_end = text[cursor..]
        .find('\n')
        .map(|p| cursor + p)
        .unwrap_or(text.len());

    // 1. 优先识别光标落在 `@"..."` 引号令牌内部的情况
    if let Some(rel_at_quote) = text[line_start..cursor].rfind("@\"") {
        let at = line_start + rel_at_quote;
        let prev_ok = text[..at]
            .chars()
            .next_back()
            .is_none_or(is_mention_boundary_char);
        if prev_ok {
            let body_start = at + 2;
            let close_opt = text[body_start..line_end].find('"').map(|p| body_start + p);
            match close_opt {
                None => {
                    return Some(MentionCaret {
                        start: at,
                        end: line_end,
                        path_end: line_end,
                        query: text[body_start..cursor].to_string(),
                        is_quoted: true,
                    });
                }
                Some(close) if cursor <= close => {
                    let mut end = close + 1;
                    let opener = text[..at]
                        .chars()
                        .next_back()
                        .filter(|&c| is_opening_wrapper(c));
                    if text[end..line_end].starts_with('#') {
                        let tail_end = scan_unquoted_end(text, end, opener);
                        let combined =
                            format!("{}{}", &text[body_start..close], &text[end..tail_end]);
                        if split_line_suffix(&combined).is_some() {
                            end = tail_end;
                        }
                    }
                    return Some(MentionCaret {
                        start: at,
                        end,
                        path_end: close + 1,
                        query: text[body_start..cursor].to_string(),
                        is_quoted: true,
                    });
                }
                Some(_) => {}
            }
        }
    }

    // 2. 无引号 `@token`：向左扫描到空白或中文标点停下，跳过前导包装符号检查是否为 `@`
    let mut token_left = cursor;
    for (idx, ch) in text[line_start..cursor].char_indices().rev() {
        if is_unquoted_token_stop(ch) {
            token_left = line_start + idx + ch.len_utf8();
            break;
        }
        if idx == 0 {
            token_left = line_start;
        }
    }

    let mut at = token_left;
    let mut opener: Option<char> = None;
    for (rel, ch) in text[token_left..cursor].char_indices() {
        if ch == '@' {
            at = token_left + rel;
            break;
        }
        if is_opening_wrapper(ch) {
            opener = Some(ch);
            continue;
        }
        return None;
    }
    if !text[at..].starts_with('@') {
        return None;
    }
    if text[at + 1..].starts_with('"') {
        return None;
    }

    // 如果紧邻 `@` 的字符也是 `@`（如 `@@`）或前面是闭合的另一 token，不触发
    let prev_char = text[..at].chars().next_back();
    if prev_char.is_some_and(|ch| !is_mention_boundary_char(ch)) {
        return None;
    }

    let end = scan_unquoted_end(text, at + 1, opener).min(line_end);
    if cursor > end {
        return None;
    }
    let raw = &text[at + 1..end];
    if raw.starts_with('@') {
        return None;
    }

    let path_end = match split_line_suffix(raw) {
        Some((stripped, _, _)) => at + 1 + stripped.len(),
        None => end,
    };
    if cursor > path_end {
        return None;
    }

    Some(MentionCaret {
        start: at,
        end,
        path_end,
        query: text[at + 1..cursor].to_string(),
        is_quoted: false,
    })
}

pub fn is_path_like_query(query: &str) -> bool {
    let bytes = query.as_bytes();
    let is_windows_drive = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'/' || bytes[2] == b'\\');
    query.starts_with('.')
        || query.starts_with('~')
        || query.starts_with('/')
        || query.contains('/')
        || query.contains('\\')
        || is_windows_drive
}

fn subsequence_score_single(needle: &str, haystack: &str) -> Option<i32> {
    if needle.is_empty() {
        return Some(0);
    }
    let n_chars: Vec<char> = needle.chars().collect();
    let h_chars: Vec<char> = haystack.chars().collect();
    if n_chars.len() > h_chars.len() {
        return None;
    }

    let mut positions = Vec::with_capacity(n_chars.len());
    let mut h_idx = 0usize;
    for &nc in &n_chars {
        let mut found = None;
        while h_idx < h_chars.len() {
            if h_chars[h_idx] == nc {
                found = Some(h_idx);
                h_idx += 1;
                break;
            }
            h_idx += 1;
        }
        positions.push(found?);
    }

    let score_alignment = |pos: &[usize]| -> i32 {
        let mut score = (pos.len() as i32) * 16;
        let mut last: Option<usize> = None;
        let mut streak = 0i32;
        for &p in pos {
            if let Some(prev) = last {
                if p == prev + 1 {
                    streak += 1;
                    score += 14 + streak * 4;
                } else {
                    streak = 0;
                    score -= ((p - prev - 1) as i32) * 2;
                }
            }
            if p == 0 || matches!(h_chars[p - 1], '/' | '\\' | '_' | '-' | '.' | ' ') {
                score += 22;
            }
            last = Some(p);
        }
        score
    };

    let mut best = score_alignment(&positions);
    // 兼顾精确连续子串出现在路径后段（如文件名中）的更高对齐得分
    if let Some(byte_pos) = haystack.find(needle) {
        let char_start = haystack[..byte_pos].chars().count();
        let cont: Vec<usize> = (char_start..char_start + n_chars.len()).collect();
        best = best.max(score_alignment(&cont) + 40);
    }
    if haystack == needle {
        best += 400;
    } else if haystack.starts_with(needle) {
        best += 180;
    }
    Some(best)
}

/// 对候选项（路径 + 末级名）做子序列模糊打分，兼顾前缀、连续命中、词边界、文件名加权与路径深度。
pub fn fuzzy_subsequence_score(
    query: &str,
    candidate_path: &str,
    candidate_name: &str,
    kind: MentionCandidateKind,
) -> Option<i32> {
    let q = query.trim().to_ascii_lowercase();
    let path_lower = candidate_path.to_ascii_lowercase();
    let name_lower = candidate_name.to_ascii_lowercase();
    let depth = candidate_path.trim_end_matches('/').matches('/').count() as i32;

    if q.is_empty() {
        let dir_bias = if kind == MentionCandidateKind::Directory {
            20
        } else {
            0
        };
        return Some(100 + dir_bias - depth * 12 - candidate_path.len() as i32);
    }

    let path_score = subsequence_score_single(&q, &path_lower)?;
    let name_bonus = subsequence_score_single(&q, &name_lower)
        .map(|s| s + 220)
        .unwrap_or(0);
    let dir_bonus = if kind == MentionCandidateKind::Directory {
        15
    } else {
        0
    };
    Some(path_score + name_bonus + dir_bonus - depth * 6 - candidate_path.len() as i32)
}

/// 生成插入编辑器的替换文本及从 `token_start` 起算的新光标字节偏移。
///
/// - 文件：`@path[suffix] `（末尾带空格，自动结束补全）
/// - 目录：`@dir/[suffix]`（末尾不加空格，光标紧贴 `/` 以便连续下钻；引号目录光标置于闭合 `"` 前）
pub fn format_mention_replacement(
    candidate: &MentionCandidate,
    preserve_suffix: &str,
    force_quote: bool,
) -> (String, usize) {
    let needs_quotes = force_quote || candidate.path.chars().any(char::is_whitespace);
    let is_dir = candidate.kind == MentionCandidateKind::Directory;
    let body = if needs_quotes {
        format!("@\"{}\"", candidate.path)
    } else {
        format!("@{}", candidate.path)
    };

    if is_dir {
        let replacement = format!("{body}{preserve_suffix}");
        let cursor_offset = if needs_quotes {
            body.len().saturating_sub(1)
        } else {
            body.len()
        };
        (replacement, cursor_offset)
    } else {
        let replacement = format!("{body}{preserve_suffix} ");
        let cursor_offset = replacement.len();
        (replacement, cursor_offset)
    }
}
