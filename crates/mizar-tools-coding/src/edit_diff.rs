//! `edit` 工具的精确/模糊匹配、BOM 与换行保留、多 hunk 冲突检查与 unified diff 生成。

use similar::{ChangeTag, TextDiff};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOp {
    pub old_text: String,
    pub new_text: String,
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

struct MatchAttempt {
    index: usize,
    match_length: usize,
    used_fuzzy_match: bool,
}

fn find_text_in(content: &str, old_text: &str) -> Option<MatchAttempt> {
    if let Some(idx) = content.find(old_text) {
        return Some(MatchAttempt {
            index: idx,
            match_length: old_text.len(),
            used_fuzzy_match: false,
        });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    fuzzy_content.find(&fuzzy_old).map(|idx| MatchAttempt {
        index: idx,
        match_length: fuzzy_old.len(),
        used_fuzzy_match: true,
    })
}

fn count_fuzzy_occurrences(content: &str, old_text: &str) -> usize {
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    if fuzzy_old.is_empty() {
        return 0;
    }
    fuzzy_content.matches(&fuzzy_old).count()
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

    let mut need_fuzzy = false;
    for edit in &normalized_edits {
        if let Some(attempt) = find_text_in(normalized_content, &edit.old_text)
            && attempt.used_fuzzy_match
        {
            need_fuzzy = true;
            break;
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
        let Some(attempt) = find_text_in(search_space, &edit.old_text) else {
            return Err(if total == 1 {
                format!(
                    "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
                )
            } else {
                format!(
                    "Could not find edits[{i}] in {path}. The oldText must match exactly including all whitespace and newlines."
                )
            });
        };

        let occurrences = count_fuzzy_occurrences(search_space, &edit.old_text);
        if occurrences > 1 {
            return Err(if total == 1 {
                format!(
                    "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
                )
            } else {
                format!(
                    "Found {occurrences} occurrences of edits[{i}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
                )
            });
        }

        matched.push(Matched {
            edit_index: i,
            match_index: attempt.index,
            match_length: attempt.match_length,
            new_text: edit.new_text.clone(),
        });
    }

    matched.sort_by_key(|m| m.match_index);
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
                },
                EditOp {
                    old_text: "    println!(\"world\");".into(),
                    new_text: "    println!(\"rust\");".into(),
                },
            ],
            "demo.rs",
        )
        .unwrap();

        assert!(applied.used_fuzzy_match);
        let final_out = format!(
            "{bom}{}",
            restore_line_endings(&applied.new_content, ending)
        );
        assert!(final_out.starts_with('\u{FEFF}'));
        assert!(final_out.contains("let s = \"hi\";\r\n"));
        assert!(final_out.contains("println!(\"rust\");\r\n"));

        let (diff, first_line) =
            generate_diff("demo.rs", &applied.base_content, &applied.new_content, 2);
        assert!(diff.contains("-\t") || diff.contains("-    let s"));
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
            }],
            "f.txt",
        )
        .unwrap_err();
        assert!(dup_err.contains("2 occurrences"));

        let overlap = apply_edits_to_normalized_content(
            "abcdef",
            &[
                EditOp {
                    old_text: "abcd".into(),
                    new_text: "1".into(),
                },
                EditOp {
                    old_text: "cdef".into(),
                    new_text: "2".into(),
                },
            ],
            "f.txt",
        )
        .unwrap_err();
        assert!(overlap.contains("overlap"));
    }
}
