//! `edit` 工具的精确/模糊多 hunk 匹配、BOM 与换行保留、原子重叠检查、紧凑 diff 与近邻预览。

use std::collections::HashSet;

use crate::truncate::{truncate_head, truncate_line};
use similar::{ChangeTag, TextDiff};
use unicode_normalization::UnicodeNormalization;

pub const EDIT_DIFF_MAX_LINES: usize = 80;
pub const EDIT_DIFF_MAX_BYTES: usize = 4 * 1024;
const PREVIEW_MAX_LINES: usize = 15;
const PREVIEW_MAX_LINE_CHARS: usize = 200;
const DIFF_MAX_LINE_CHARS: usize = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOp {
    pub old_text: String,
    pub new_text: String,
    pub replace_all: bool,
}

#[derive(Debug, Clone)]
pub struct TextReplacement {
    pub match_index: usize,
    pub match_length: usize,
    pub new_text: String,
}

#[derive(Debug, Clone)]
pub struct AppliedEditsResult {
    pub base_content: String,
    pub new_content: String,
    pub used_fuzzy_match: bool,
    pub replacements_applied: usize,
}

pub fn strip_bom(content: &str) -> (&'static str, &str) {
    if let Some(rest) = content.strip_prefix('\u{FEFF}') {
        ("\u{FEFF}", rest)
    } else {
        ("", content)
    }
}

pub fn detect_line_ending(content: &str) -> &'static str {
    let crlf_idx = content.find("\r\n");
    let lf_idx = content.find('\n');
    match (crlf_idx, lf_idx) {
        (None, _) => "\n",
        (Some(_), None) => "\r\n",
        (Some(crlf), Some(lf)) => {
            if crlf <= lf {
                "\r\n"
            } else {
                "\n"
            }
        }
    }
}

pub fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

pub fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_string()
    }
}

fn map_fuzzy_char(ch: char) -> char {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        other => other,
    }
}

/// 模糊匹配归一化：NFKC 归一化 + 按行去除行尾空白 + Unicode 弯引号/破折号/特殊空格转 ASCII。
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    let nfkc: String = text.nfkc().map(map_fuzzy_char).collect();
    nfkc.split('\n')
        .map(|line| line.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Copy)]
struct LineSpan {
    start: usize,
    end: usize,
}

fn split_line_spans(content: &str) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    for (idx, _) in content.match_indices('\n') {
        let end = idx + 1;
        spans.push(LineSpan { start, end });
        start = end;
    }
    if start < content.len() {
        spans.push(LineSpan {
            start,
            end: content.len(),
        });
    }
    spans
}

fn get_replacement_line_range(
    lines: &[LineSpan],
    rep: &TextReplacement,
) -> Result<(usize, usize), String> {
    let rep_start = rep.match_index;
    let rep_end = rep.match_index + rep.match_length;

    let Some(start_line) = lines
        .iter()
        .position(|line| rep_start >= line.start && rep_start < line.end)
    else {
        return Err("Replacement range is outside the base content.".into());
    };

    let mut end_line = start_line;
    while end_line < lines.len() && lines[end_line].end < rep_end {
        end_line += 1;
    }
    if end_line >= lines.len() {
        return Err("Replacement range is outside the base content.".into());
    }

    Ok((start_line, end_line + 1))
}

fn apply_replacements(content: &str, replacements: &[TextReplacement], offset: usize) -> String {
    let mut result = content.to_string();
    for rep in replacements.iter().rev() {
        let idx = rep.match_index - offset;
        result.replace_range(idx..idx + rep.match_length, &rep.new_text);
    }
    result
}

fn widen_replacements_to_lines(
    original_content: &str,
    base_content: &str,
    replacements: Vec<TextReplacement>,
) -> Result<Vec<TextReplacement>, String> {
    let original_lines = split_line_spans(original_content);
    let base_lines = split_line_spans(base_content);
    if original_lines.len() != base_lines.len() {
        return Err(
            "Cannot preserve unchanged lines because the base content has a different line count."
                .into(),
        );
    }

    #[derive(Clone)]
    struct Group {
        start_line: usize,
        end_line: usize,
        replacements: Vec<TextReplacement>,
    }

    let mut sorted = replacements;
    sorted.sort_by_key(|r| r.match_index);
    let mut groups: Vec<Group> = Vec::new();
    for rep in sorted {
        let (start_line, end_line) = get_replacement_line_range(&base_lines, &rep)?;
        if let Some(cur) = groups.last_mut()
            && start_line < cur.end_line
        {
            cur.end_line = cur.end_line.max(end_line);
            cur.replacements.push(rep);
            continue;
        }
        groups.push(Group {
            start_line,
            end_line,
            replacements: vec![rep],
        });
    }

    Ok(groups
        .into_iter()
        .map(|g| {
            let base_start = base_lines[g.start_line].start;
            let base_end = base_lines[g.end_line - 1].end;
            let orig_start = original_lines[g.start_line].start;
            let orig_end = original_lines[g.end_line - 1].end;
            let rewritten = apply_replacements(
                &base_content[base_start..base_end],
                &g.replacements,
                base_start,
            );
            TextReplacement {
                match_index: orig_start,
                match_length: orig_end - orig_start,
                new_text: rewritten,
            }
        })
        .collect())
}

fn extract_tokens(text: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            for lc in ch.to_lowercase() {
                current.push(lc);
            }
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.insert(current);
    }
    tokens
}

fn line_number_at(content: &str, byte_idx: usize) -> usize {
    let clamped = byte_idx.min(content.len());
    content[..clamped].bytes().filter(|&b| b == b'\n').count() + 1
}

/// 匹配失败时定位最相近的行区段预览（`<line>\t<text>` 格式），辅助模型快速修正 `oldText`。
pub fn find_closest_snippet(normalized_content: &str, old_text: &str) -> Option<String> {
    let content_lines: Vec<&str> = normalized_content.split('\n').collect();
    if content_lines.is_empty() || (content_lines.len() == 1 && content_lines[0].is_empty()) {
        return None;
    }

    let format_range = |start: usize, end: usize| -> String {
        let clamped_end = end.min(content_lines.len());
        let clamped_start = start.min(clamped_end);
        let body = (clamped_start..clamped_end)
            .map(|idx| {
                let (short, _) = truncate_line(content_lines[idx], PREVIEW_MAX_LINE_CHARS);
                format!("{}\t{}", idx + 1, short)
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Closest matching region (lines {}-{}):\n{body}",
            clamped_start + 1,
            clamped_end
        )
    };

    if content_lines.len() <= PREVIEW_MAX_LINES {
        return Some(format_range(0, content_lines.len()));
    }

    let old_tokens = extract_tokens(old_text);
    let old_trimmed_lines: Vec<String> = old_text
        .split('\n')
        .map(|l| normalize_for_fuzzy_match(l.trim()))
        .filter(|l| !l.is_empty())
        .collect();

    if old_tokens.is_empty() && old_trimmed_lines.is_empty() {
        return None;
    }

    let window = old_text.split('\n').count().clamp(1, 8);
    let max_start = content_lines.len().saturating_sub(window);
    let mut best_score = 0.0f64;
    let mut best_start = 0usize;

    let norm_content_lines: Vec<String> = content_lines
        .iter()
        .map(|l| normalize_for_fuzzy_match(l.trim()))
        .collect();

    for start in 0..=max_start {
        let end = (start + window).min(content_lines.len());
        let slice_text = content_lines[start..end].join("\n");
        let win_tokens = extract_tokens(&slice_text);

        let token_score = if old_tokens.is_empty() {
            0.0
        } else {
            let hits = old_tokens
                .iter()
                .filter(|t| win_tokens.contains(*t))
                .count();
            hits as f64 / old_tokens.len() as f64
        };

        let line_score = if old_trimmed_lines.is_empty() {
            0.0
        } else {
            let line_hits = old_trimmed_lines
                .iter()
                .filter(|ol| {
                    norm_content_lines[start..end].iter().any(|cl| {
                        !cl.is_empty()
                            && (cl == *ol || cl.contains(ol.as_str()) || ol.contains(cl.as_str()))
                    })
                })
                .count();
            line_hits as f64 / old_trimmed_lines.len() as f64
        };

        let score = 0.65 * token_score + 0.35 * line_score;
        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }

    if best_score < 0.15 {
        return None;
    }

    let ctx_start = best_start.saturating_sub(2);
    let ctx_end = (best_start + window + 2)
        .min(content_lines.len())
        .min(ctx_start + PREVIEW_MAX_LINES);
    Some(format_range(ctx_start, ctx_end))
}

fn edit_label(edit_index: usize, total: usize) -> String {
    if total == 1 {
        "the text".to_string()
    } else {
        format!("edits[{edit_index}]")
    }
}

pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[EditOp],
    path: &str,
) -> Result<AppliedEditsResult, String> {
    if edits.is_empty() {
        return Err(format!("No edits provided for {path}."));
    }

    let normalized_edits: Vec<EditOp> = edits
        .iter()
        .map(|e| EditOp {
            old_text: normalize_to_lf(&e.old_text),
            new_text: normalize_to_lf(&e.new_text),
            replace_all: e.replace_all,
        })
        .collect();

    let total = normalized_edits.len();
    for (i, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(if total == 1 {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{i}].oldText must not be empty in {path}.")
            });
        }
    }

    // 若任意段无法精确命中但可经模糊归一化命中，则整体切换至模糊归一化视图，保持多段定位坐标一致。
    let mut need_fuzzy = false;
    for edit in &normalized_edits {
        if !normalized_content.contains(&edit.old_text) {
            let fuzzy_content = normalize_for_fuzzy_match(normalized_content);
            let fuzzy_old = normalize_for_fuzzy_match(&edit.old_text);
            if !fuzzy_old.is_empty() && fuzzy_content.contains(&fuzzy_old) {
                need_fuzzy = true;
                break;
            }
        }
    }

    let fuzzy_view;
    let search_space = if need_fuzzy {
        fuzzy_view = normalize_for_fuzzy_match(normalized_content);
        fuzzy_view.as_str()
    } else {
        normalized_content
    };

    #[derive(Clone)]
    struct Matched {
        edit_index: usize,
        match_index: usize,
        match_length: usize,
        new_text: String,
    }

    let mut matched: Vec<Matched> = Vec::with_capacity(total);
    for (i, edit) in normalized_edits.iter().enumerate() {
        let needle = if need_fuzzy {
            normalize_for_fuzzy_match(&edit.old_text)
        } else {
            edit.old_text.clone()
        };
        if needle.is_empty() {
            return Err(if total == 1 {
                format!("oldText must not be empty in {path}.")
            } else {
                format!("edits[{i}].oldText must not be empty in {path}.")
            });
        }

        let hits: Vec<usize> = search_space
            .match_indices(&needle)
            .map(|(idx, _)| idx)
            .collect();

        if hits.is_empty() {
            let label = edit_label(i, total);
            let mut msg = format!(
                "Could not find {label} in {path}. The oldText must match exactly including whitespace and newlines."
            );
            if let Some(preview) = find_closest_snippet(normalized_content, &edit.old_text) {
                msg.push_str("\n\n");
                msg.push_str(&preview);
            }
            return Err(msg);
        }

        if hits.len() > 1 && !edit.replace_all {
            let line_nums: Vec<String> = hits
                .iter()
                .take(10)
                .map(|&idx| line_number_at(search_space, idx).to_string())
                .collect();
            let suffix = if hits.len() > 10 { ", ..." } else { "" };
            let label = edit_label(i, total);
            return Err(format!(
                "Found {} occurrences of {label} in {path} (at lines {}{suffix}). Provide more surrounding context to make it unique, or set replaceAll=true.",
                hits.len(),
                line_nums.join(", ")
            ));
        }

        for idx in hits {
            matched.push(Matched {
                edit_index: i,
                match_index: idx,
                match_length: needle.len(),
                new_text: edit.new_text.clone(),
            });
        }
    }

    matched.sort_by_key(|m| (m.match_index, m.match_length));
    for pair in matched.windows(2) {
        let prev = &pair[0];
        let curr = &pair[1];
        if prev.match_index + prev.match_length > curr.match_index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                prev.edit_index, curr.edit_index
            ));
        }
    }

    let replacements_applied = matched.len();
    let raw_replacements: Vec<TextReplacement> = matched
        .into_iter()
        .map(|m| TextReplacement {
            match_index: m.match_index,
            match_length: m.match_length,
            new_text: m.new_text,
        })
        .collect();

    let final_replacements = if need_fuzzy {
        widen_replacements_to_lines(normalized_content, search_space, raw_replacements)?
    } else {
        raw_replacements
    };

    let new_content = apply_replacements(normalized_content, &final_replacements, 0);
    if new_content == normalized_content {
        return Err(if total == 1 {
            format!("No changes made to {path}. The replacement produced identical content.")
        } else {
            format!("No changes made to {path}. The replacements produced identical content.")
        });
    }

    Ok(AppliedEditsResult {
        base_content: normalized_content.to_string(),
        new_content,
        used_fuzzy_match: need_fuzzy,
        replacements_applied,
    })
}

pub fn generate_diff(
    path: &str,
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> (String, Option<usize>) {
    let diff = TextDiff::from_lines(old_content, new_content);
    let unified = diff
        .unified_diff()
        .context_radius(context_lines)
        .header(path, path)
        .to_string();

    let mut first_changed_line = None;
    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => {
                first_changed_line = change.new_index().map(|i| i + 1);
                break;
            }
            ChangeTag::Delete => {
                first_changed_line = change.old_index().map(|i| i + 1);
                break;
            }
            ChangeTag::Equal => {}
        }
    }

    (unified, first_changed_line)
}

/// 生成给模型看的紧凑限长 diff 片段（单行封顶 + 总行数/字节封顶），避免大改动回显撑爆上下文。
pub fn format_compact_diff(
    unified_diff: &str,
    max_lines: usize,
    max_bytes: usize,
) -> (String, bool) {
    if unified_diff.trim().is_empty() {
        return (String::new(), false);
    }
    let mut any_line_capped = false;
    let capped_lines: Vec<String> = unified_diff
        .lines()
        .map(|line| {
            let (short, hit) = truncate_line(line, DIFF_MAX_LINE_CHARS);
            if hit {
                any_line_capped = true;
            }
            short
        })
        .collect();
    let total_lines = capped_lines.len();
    let joined = capped_lines.join("\n");
    let trunc = truncate_head(&joined, max_lines, max_bytes);
    if trunc.truncated {
        let mut out = trunc.content;
        out.push_str(&format!(
            "\n... [diff truncated, showing {} of {total_lines} lines]",
            trunc.output_lines
        ));
        (out, true)
    } else {
        (trunc.content, any_line_capped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_multi_hunk_fuzzy_matching_and_bom_crlf() {
        let raw = "\u{FEFF}fn a() {\r\n    let s = “hello”;   \r\n}\r\n\r\nfn b() {\r\n    println!(\"world\");\r\n}\r\n";
        let (bom, without_bom) = strip_bom(raw);
        let ending = detect_line_ending(without_bom);
        assert_eq!(bom, "\u{FEFF}");
        assert_eq!(ending, "\r\n");

        let normalized = normalize_to_lf(without_bom);
        let applied = apply_edits_to_normalized_content(
            &normalized,
            &[
                EditOp {
                    old_text: "    let s = \"hello\";".into(),
                    new_text: "    let s = \"hi\";".into(),
                    replace_all: false,
                },
                EditOp {
                    old_text: "    println!(\"world\");".into(),
                    new_text: "    println!(\"rust\");".into(),
                    replace_all: false,
                },
            ],
            "demo.rs",
        )
        .unwrap();

        assert!(applied.used_fuzzy_match);
        assert_eq!(applied.replacements_applied, 2);
        let final_out = format!(
            "{bom}{}",
            restore_line_endings(&applied.new_content, ending)
        );
        assert!(final_out.starts_with('\u{FEFF}'));
        assert!(final_out.contains("let s = \"hi\";\r\n"));
        assert!(final_out.contains("println!(\"rust\");\r\n"));

        let (diff, first_line) =
            generate_diff("demo.rs", &applied.base_content, &applied.new_content, 2);
        assert!(diff.contains("-    let s"));
        assert_eq!(first_line, Some(2));
    }

    #[test]
    fn rejects_duplicate_and_overlapping_edits() {
        let content = "dup\ndup\n";
        let dup_err = apply_edits_to_normalized_content(
            content,
            &[EditOp {
                old_text: "dup".into(),
                new_text: "x".into(),
                replace_all: false,
            }],
            "f.txt",
        )
        .unwrap_err();
        assert!(dup_err.contains("2 occurrences"));
        assert!(dup_err.contains("lines 1, 2"));
        assert!(dup_err.contains("replaceAll=true"));

        let overlap = apply_edits_to_normalized_content(
            "abcdef",
            &[
                EditOp {
                    old_text: "abcd".into(),
                    new_text: "1".into(),
                    replace_all: false,
                },
                EditOp {
                    old_text: "cdef".into(),
                    new_text: "2".into(),
                    replace_all: false,
                },
            ],
            "f.txt",
        )
        .unwrap_err();
        assert!(overlap.contains("overlap"));
    }

    #[test]
    fn supports_replace_all_and_atomic_original_matching() {
        let content = "let foo = bar;\nlet baz = foo + foo;\n";
        let applied = apply_edits_to_normalized_content(
            content,
            &[
                EditOp {
                    old_text: "foo".into(),
                    new_text: "qux".into(),
                    replace_all: true,
                },
                EditOp {
                    old_text: "bar".into(),
                    new_text: "42".into(),
                    replace_all: false,
                },
            ],
            "vars.rs",
        )
        .unwrap();
        assert_eq!(applied.replacements_applied, 4);
        assert_eq!(applied.new_content, "let qux = 42;\nlet baz = qux + qux;\n");
    }

    #[test]
    fn not_found_error_includes_closest_matching_region_preview() {
        let mut lines = Vec::new();
        for i in 1..=30 {
            if i == 18 {
                lines.push("fn compute_total_price(tax: f64, base: f64) -> f64 {".to_string());
            } else if i == 19 {
                lines.push("    base * (1.0 + tax)".to_string());
            } else if i == 20 {
                lines.push("}".to_string());
            } else {
                lines.push(format!("// unrelated filler line {i}"));
            }
        }
        let content = lines.join("\n");

        let err = apply_edits_to_normalized_content(
            &content,
            &[EditOp {
                old_text: "fn compute_total_price(tax: f64, amount: f64) -> f64 {\n    amount * (1.0 + tax)\n}"
                    .into(),
                new_text: "fn compute_total_price() -> f64 { 0.0 }".into(),
                replace_all: false,
            }],
            "price.rs",
        )
        .unwrap_err();

        assert!(err.contains("Closest matching region"), "err:\n{err}");
        assert!(err.contains("18\tfn compute_total_price"), "err:\n{err}");
    }

    #[test]
    fn format_compact_diff_caps_large_diffs() {
        let old = (1..=150)
            .map(|i| format!("old_line_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let new = (1..=150)
            .map(|i| format!("new_line_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (full_diff, _) = generate_diff("big.txt", &old, &new, 2);
        let (compact, was_trunc) = format_compact_diff(&full_diff, 30, 1024);
        assert!(was_trunc);
        assert!(compact.contains("diff truncated, showing"));
        assert!(compact.len() < full_diff.len());
    }
}
