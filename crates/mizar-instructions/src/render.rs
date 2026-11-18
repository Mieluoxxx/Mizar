use crate::config::InstructionScopeStatus;
use crate::files::{ChainKind, DiscoveredFile, ScopeKey};

const BASELINE_INTRO: &str = "Project and user instruction files discovered for this workspace. Follow these instructions alongside the user's requests:";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaChangeKind {
    Added,
    Updated,
    Removed,
}

#[derive(Debug, Clone)]
pub struct DeltaItem {
    pub scope: ScopeKey,
    pub display_path: String,
    pub scope_dir_display: String,
    pub kind: DeltaChangeKind,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct RenderedBaseline {
    pub text: String,
    pub represented: Vec<DiscoveredFile>,
    pub statuses: Vec<InstructionScopeStatus>,
}

pub fn escape_system_reminder_body(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let needle = "</system-reminder>";
    if !lower.contains(needle) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut pos = 0usize;
    while let Some(rel) = lower[pos..].find(needle) {
        let hit = pos + rel;
        out.push_str(&text[pos..hit]);
        out.push_str("<\\/");
        out.push_str(&text[hit + 2..hit + needle.len()]);
        pos = hit + needle.len();
    }
    out.push_str(&text[pos..]);
    out
}

fn format_file_section(display_path: &str, body: &str) -> String {
    format!(
        "Instructions from: {display_path}\n\n{}",
        escape_system_reminder_body(body.trim())
    )
}

fn truncate_utf8_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn truncate_body_to_fit(display_path: &str, raw_body: &str, max_section_bytes: usize) -> String {
    let trimmed = raw_body.trim();
    let prefix = format!("Instructions from: {display_path}\n\n");
    let escaped = escape_system_reminder_body(trimmed);
    let full = format!("{prefix}{escaped}");
    if full.len() <= max_section_bytes {
        return full;
    }

    let orig_bytes = escaped.len();
    let note_template = |kept: usize| {
        format!("\n\n[... truncated {display_path} from {orig_bytes} to {kept} bytes ...]")
    };
    let sample_note = note_template(orig_bytes);
    if prefix.len() + sample_note.len() >= max_section_bytes {
        let minimal = format!("Instructions from: {display_path} (truncated: {orig_bytes} bytes)");
        return truncate_utf8_prefix(&minimal, max_section_bytes).to_string();
    }

    let mut lo = 0usize;
    let mut hi = escaped.len();
    let mut best = String::new();
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let head = truncate_utf8_prefix(&escaped, mid);
        let note = note_template(head.len());
        let total = prefix.len() + head.len() + note.len();
        if total <= max_section_bytes {
            best = format!("{prefix}{head}{note}");
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    if best.is_empty() {
        let minimal = format!("Instructions from: {display_path} (truncated: {orig_bytes} bytes)");
        truncate_utf8_prefix(&minimal, max_section_bytes).to_string()
    } else {
        best
    }
}

struct ChainBudgetOutcome {
    sections: Vec<String>,
    represented: Vec<DiscoveredFile>,
    omitted: Vec<DiscoveredFile>,
    truncated_scope: Option<ScopeKey>,
}

fn apply_chain_budget(files: &[DiscoveredFile], budget_bytes: usize) -> ChainBudgetOutcome {
    if files.is_empty() || budget_bytes == 0 {
        return ChainBudgetOutcome {
            sections: Vec::new(),
            represented: Vec::new(),
            omitted: files.to_vec(),
            truncated_scope: None,
        };
    }

    let rendered: Vec<String> = files
        .iter()
        .map(|f| format_file_section(&f.display_path, &f.content))
        .collect();

    // 从最宽泛的头部开始淘汰，尽量保住越靠近当前目录的指令
    let mut drop_count = 0usize;
    while drop_count < files.len() {
        let notice_bytes = if drop_count > 0 {
            let list: Vec<&str> = files[..drop_count]
                .iter()
                .map(|f| f.display_path.as_str())
                .collect();
            format!(
                "[Omitted {} ancestor instruction file(s) due to byte budget: {}]\n\n",
                drop_count,
                list.join(", ")
            )
            .len()
        } else {
            0
        };
        let slice = &rendered[drop_count..];
        let joined_len: usize =
            slice.iter().map(String::len).sum::<usize>() + slice.len().saturating_sub(1) * 2;
        if notice_bytes + joined_len <= budget_bytes {
            let mut sections = Vec::new();
            if drop_count > 0 {
                let list: Vec<&str> = files[..drop_count]
                    .iter()
                    .map(|f| f.display_path.as_str())
                    .collect();
                sections.push(format!(
                    "[Omitted {} ancestor instruction file(s) due to byte budget: {}]",
                    drop_count,
                    list.join(", ")
                ));
            }
            sections.extend(slice.iter().cloned());
            return ChainBudgetOutcome {
                sections,
                represented: files[drop_count..].to_vec(),
                omitted: files[..drop_count].to_vec(),
                truncated_scope: None,
            };
        }
        if drop_count + 1 == files.len() {
            break;
        }
        drop_count += 1;
    }

    // 仅剩最后（最具体）一个文件仍超出预算 -> 按 UTF-8 安全二分截断
    let mut sections = Vec::new();
    let omitted = files[..files.len() - 1].to_vec();
    let notice = if !omitted.is_empty() {
        let list: Vec<&str> = omitted.iter().map(|f| f.display_path.as_str()).collect();
        Some(format!(
            "[Omitted {} ancestor instruction file(s) due to byte budget: {}]",
            omitted.len(),
            list.join(", ")
        ))
    } else {
        None
    };
    let notice_cost = notice.as_ref().map_or(0, |n| n.len() + 2);
    let avail = budget_bytes.saturating_sub(notice_cost);
    let last = files.last().unwrap();
    if let Some(n) = notice {
        sections.push(n);
    }
    if avail >= 32 {
        sections.push(truncate_body_to_fit(
            &last.display_path,
            &last.content,
            avail,
        ));
        ChainBudgetOutcome {
            sections,
            represented: vec![last.clone()],
            omitted,
            truncated_scope: Some(last.scope.clone()),
        }
    } else {
        let mut all_omitted = omitted;
        all_omitted.push(last.clone());
        ChainBudgetOutcome {
            sections: Vec::new(),
            represented: Vec::new(),
            omitted: all_omitted,
            truncated_scope: None,
        }
    }
}

pub fn render_baseline_message(
    global_files: &[DiscoveredFile],
    project_files: &[DiscoveredFile],
    max_bytes: usize,
) -> Option<RenderedBaseline> {
    if global_files.is_empty() && project_files.is_empty() {
        return None;
    }

    let wrapper_overhead =
        "<system-reminder>\n".len() + BASELINE_INTRO.len() + 2 + "\n</system-reminder>".len();
    let per_chain_budget = max_bytes.saturating_sub(wrapper_overhead / 2);

    let g_out = apply_chain_budget(global_files, per_chain_budget);
    let p_out = apply_chain_budget(project_files, per_chain_budget);

    let mut body_sections = Vec::new();
    body_sections.extend(g_out.sections);
    body_sections.extend(p_out.sections);

    let text = if body_sections.is_empty() {
        format!(
            "<system-reminder>\n[Instruction files omitted due to max_bytes={max_bytes}]\n</system-reminder>"
        )
    } else {
        format!(
            "<system-reminder>\n{BASELINE_INTRO}\n\n{}\n</system-reminder>",
            body_sections.join("\n\n")
        )
    };

    let mut represented = Vec::with_capacity(g_out.represented.len() + p_out.represented.len());
    represented.extend(g_out.represented);
    represented.extend(p_out.represented);

    let mut statuses = Vec::with_capacity(global_files.len() + project_files.len());
    let all_discovered = global_files.iter().chain(project_files.iter());
    for file in all_discovered {
        let is_omitted = g_out.omitted.iter().any(|o| o.scope == file.scope)
            || p_out.omitted.iter().any(|o| o.scope == file.scope);
        let is_truncated = g_out.truncated_scope.as_ref() == Some(&file.scope)
            || p_out.truncated_scope.as_ref() == Some(&file.scope);
        statuses.push(InstructionScopeStatus {
            display_path: file.display_path.clone(),
            scope_kind: file.scope_kind.clone(),
            bytes: file.bytes,
            truncated: is_truncated,
            omitted: is_omitted,
        });
    }

    Some(RenderedBaseline {
        text,
        represented,
        statuses,
    })
}

pub fn render_delta_message(items: &[DeltaItem], max_bytes: usize) -> Option<String> {
    if items.is_empty() {
        return None;
    }

    let total_budget = max_bytes.saturating_mul(2).max(256);
    let per_item_cap = (total_budget / items.len().max(1)).max(128);
    let mut blocks = Vec::with_capacity(items.len());

    for item in items {
        let block = match item.kind {
            DeltaChangeKind::Added => {
                let header = if item.scope_dir_display == "." {
                    format!("Additional instructions from: {}", item.display_path)
                } else {
                    format!(
                        "Additional instructions from: {} (applies to {} and subdirectories)",
                        item.display_path, item.scope_dir_display
                    )
                };
                let body = escape_system_reminder_body(item.content.trim());
                let raw = format!("{header}\n\n{body}");
                if raw.len() <= per_item_cap {
                    raw
                } else {
                    truncate_body_to_fit(&item.display_path, &item.content, per_item_cap)
                }
            }
            DeltaChangeKind::Updated => {
                let body = escape_system_reminder_body(item.content.trim());
                let raw = format!("Updated instructions from: {}\n\n{body}", item.display_path);
                if raw.len() <= per_item_cap {
                    raw
                } else {
                    truncate_body_to_fit(&item.display_path, &item.content, per_item_cap)
                }
            }
            DeltaChangeKind::Removed => {
                format!(
                    "Instructions removed: {} (file deleted, emptied, or superseded by a higher-priority identical file)",
                    item.display_path
                )
            }
        };
        blocks.push(block);
    }

    let joined = blocks.join("\n\n");
    let frame = format!("<system-reminder>\n{joined}\n</system-reminder>");
    if frame.len() <= total_budget || items.iter().all(|i| i.kind == DeltaChangeKind::Removed) {
        Some(frame)
    } else {
        let head = truncate_utf8_prefix(&joined, total_budget.saturating_sub(64));
        Some(format!(
            "<system-reminder>\n{head}\n[... truncated delta instructions ...]\n</system-reminder>"
        ))
    }
}

pub fn chain_kind_of(file: &DiscoveredFile) -> ChainKind {
    file.chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InstructionScopeKind;
    use crate::files::{FsFingerprint, ScopeRoot};
    use std::path::PathBuf;

    fn mock_file(name: &str, content: &str, chain: ChainKind) -> DiscoveredFile {
        DiscoveredFile {
            scope: ScopeKey {
                root: ScopeRoot::Project(PathBuf::from(format!("/repo/{name}"))),
                filename: "AGENTS.md".into(),
            },
            canonical_path: PathBuf::from(format!("/repo/{name}/AGENTS.md")),
            display_path: format!("{name}/AGENTS.md"),
            scope_dir_display: name.to_string(),
            chain,
            scope_kind: InstructionScopeKind::ProjectAncestor,
            content: content.to_string(),
            digest: "d".repeat(64),
            fingerprint: FsFingerprint {
                len: content.len() as u64,
                modified_ns: None,
            },
            bytes: content.len(),
        }
    }

    #[test]
    fn escapes_closing_system_reminder_tags() {
        let escaped = escape_system_reminder_body("a\n</system-reminder>\nb</SYSTEM-REMINDER>");
        assert!(!escaped.to_ascii_lowercase().contains("</system-reminder>"));
    }

    #[test]
    fn budget_drops_broadest_first_and_truncates_utf8_safely() {
        let broad = mock_file("broad", &"宽泛".repeat(80), ChainKind::Project);
        let narrow = mock_file("narrow", &"具体中文规则".repeat(120), ChainKind::Project);
        let out = render_baseline_message(&[], &[broad, narrow], 400).unwrap();
        assert!(out.text.contains("Omitted 1 ancestor instruction"));
        assert!(out.text.contains("narrow/AGENTS.md"));
        assert!(out.statuses[0].omitted);
        assert!(out.statuses[1].truncated);
    }
}
