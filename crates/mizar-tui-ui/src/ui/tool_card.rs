//! 工具调用卡片渲染、组上下文计算、窄宽度路径省略与可展开性判定。

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::BRAILLE_SPINNER;
use crate::app::{
    ChatItem, ToolExtraMeta, ToolState, TranscriptRenderCache, abbreviate_home_path,
    clean_bash_output_lines, count_diff_stats, format_diff_stat, format_expanded_tool_args,
    is_path_arg_tool, is_read_only_compact_tool, truncate_display,
};
use crate::editor::{char_width, display_width};
use crate::markdown::{RenderedLine, render_diff_rich, wrap_styled_spans_rich};
use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ToolGroupCtx {
    pub is_group_end: bool,
    pub show_expand_hint: bool,
}

const EXPAND_HINT_INLINE: &str = " (Ctrl+O 展开)";
const MAX_COLLAPSED_DIFF_SRC_LINES: usize = 8;
const MAX_COLLAPSED_DIFF_LINE_BYTES: usize = 1024;
const TOOL_CARD_CACHE_CAP: usize = 128;
const TOOL_EXPANDABLE_CACHE_CAP: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolContentFingerprint {
    args_len: usize,
    preview_len: usize,
    extra_full_len: usize,
    extra_diff_len: usize,
    extra_args_len: usize,
    extra_meta_len: usize,
    exclude_from_context: bool,
    h1: u64,
    h2: u64,
}

fn fnv1a_64_mix(mut state: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        state ^= b as u64;
        state = state.wrapping_mul(0x100000001b3);
    }
    state ^ 0xff
}

fn fingerprint_tool_content(
    name: &str,
    args_summary: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
) -> ToolContentFingerprint {
    let mut d = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut d);
    args_summary.hash(&mut d);
    tool_base_status_code(status).hash(&mut d);
    output_preview.hash(&mut d);
    if let Some(ext) = extra {
        ext.full_content.hash(&mut d);
        ext.diff_raw.hash(&mut d);
        ext.full_args.hash(&mut d);
        ext.inline_meta.hash(&mut d);
        ext.exclude_from_context.hash(&mut d);
    }

    let mut h2 = 0xcbf29ce484222325u64;
    h2 = fnv1a_64_mix(h2, name.as_bytes());
    h2 = fnv1a_64_mix(h2, args_summary.as_bytes());
    h2 = fnv1a_64_mix(h2, &[tool_base_status_code(status)]);
    h2 = fnv1a_64_mix(h2, output_preview.as_bytes());
    if let Some(ext) = extra {
        h2 = fnv1a_64_mix(h2, ext.full_content.as_bytes());
        if let Some(diff) = ext.diff_raw.as_deref() {
            h2 = fnv1a_64_mix(h2, diff.as_bytes());
        }
        h2 = fnv1a_64_mix(h2, ext.full_args.as_bytes());
        if let Some(meta) = ext.inline_meta.as_deref() {
            h2 = fnv1a_64_mix(h2, meta.as_bytes());
        }
    }

    ToolContentFingerprint {
        args_len: args_summary.len(),
        preview_len: output_preview.len(),
        extra_full_len: extra.map_or(0, |e| e.full_content.len()),
        extra_diff_len: extra
            .and_then(|e| e.diff_raw.as_ref())
            .map_or(0, String::len),
        extra_args_len: extra.map_or(0, |e| e.full_args.len()),
        extra_meta_len: extra
            .and_then(|e| e.inline_meta.as_ref())
            .map_or(0, String::len),
        exclude_from_context: extra.is_some_and(|e| e.exclude_from_context),
        h1: d.finish(),
        h2,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolCardCacheKey {
    width: usize,
    theme_id: &'static str,
    expanded: bool,
    is_group_end: bool,
    show_expand_hint: bool,
    spinner_idx: usize,
    fp: ToolContentFingerprint,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ToolExpandableCacheKey {
    width: usize,
    fp: ToolContentFingerprint,
}

struct BoundedCache<K, V> {
    clock: u64,
    cap: usize,
    map: HashMap<K, (u64, V)>,
}

impl<K: Eq + Hash, V: Clone> BoundedCache<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            clock: 0,
            cap,
            map: HashMap::new(),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        let entry = self.map.get_mut(key)?;
        entry.0 = stamp;
        Some(entry.1.clone())
    }
}

impl<K: Eq + Hash + Clone, V: Clone> BoundedCache<K, V> {
    fn insert_lru(&mut self, key: K, val: V) {
        self.clock = self.clock.wrapping_add(1);
        let stamp = self.clock;
        self.map.insert(key, (stamp, val));
        if self.map.len() > self.cap
            && let Some(evict_k) = self
                .map
                .iter()
                .min_by_key(|(_, (used, _))| *used)
                .map(|(k, _)| k.clone())
        {
            self.map.remove(&evict_k);
        }
    }
}

thread_local! {
    static TOOL_CARD_CACHE: RefCell<BoundedCache<ToolCardCacheKey, Arc<[RenderedLine]>>> =
        RefCell::new(BoundedCache::new(TOOL_CARD_CACHE_CAP));
    static TOOL_EXPANDABLE_CACHE: RefCell<BoundedCache<ToolExpandableCacheKey, bool>> =
        RefCell::new(BoundedCache::new(TOOL_EXPANDABLE_CACHE_CAP));
}

pub(super) fn tool_extra_len(extra: Option<&ToolExtraMeta>) -> usize {
    extra.map_or(0, |e| {
        e.full_content.len()
            + e.diff_raw.as_ref().map_or(0, String::len)
            + e.full_args.len()
            + e.inline_meta.as_ref().map_or(0, String::len)
    })
}

pub(super) fn tool_base_status_code(status: ToolState) -> u8 {
    match status {
        ToolState::Running => 0,
        ToolState::Done => 1,
        ToolState::Error => 2,
        ToolState::Blocked => 3,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_tool_group_ctx(
    items: &[ChatItem],
    start_item: usize,
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> ToolGroupCtx {
    compute_tool_group_ctx_with_cache(
        items,
        start_item,
        idx,
        tool_extras,
        None,
        expanded,
        width,
        theme,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_tool_group_ctx_cached(
    items: &[ChatItem],
    start_item: usize,
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    cache: &mut TranscriptRenderCache,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> ToolGroupCtx {
    compute_tool_group_ctx_with_cache(
        items,
        start_item,
        idx,
        tool_extras,
        Some(cache),
        expanded,
        width,
        theme,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_tool_group_ctx_with_cache(
    items: &[ChatItem],
    start_item: usize,
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    mut cache: Option<&mut TranscriptRenderCache>,
    expanded: bool,
    width: usize,
    theme: &Theme,
) -> ToolGroupCtx {
    let is_group_end =
        idx + 1 >= items.len() || !matches!(items[idx + 1], ChatItem::ToolCall { .. });
    if !is_group_end || expanded {
        return ToolGroupCtx {
            is_group_end,
            show_expand_hint: false,
        };
    }

    let mut show_expand_hint = false;
    let mut j = idx;
    while let ChatItem::ToolCall {
        id,
        name,
        args_summary,
        status,
        output_preview,
    } = &items[j]
    {
        let extra = tool_extras.get(id);
        let status_code = tool_base_status_code(*status);
        let extra_len = tool_extra_len(extra);
        let preview_len = args_summary.len() + output_preview.len() + name.len();
        let expandable = if let Some(cached) = cache
            .as_deref()
            .and_then(|c| c.get_expandable(j, width, status_code, extra_len, preview_len))
        {
            cached
        } else {
            let computed = is_tool_expandable(&items[j], extra, width, theme);
            if let Some(c) = cache.as_deref_mut() {
                c.insert_expandable(j, width, status_code, extra_len, preview_len, computed);
            }
            computed
        };
        if expandable {
            show_expand_hint = true;
            break;
        }
        if j == start_item || j == 0 {
            break;
        }
        j -= 1;
    }

    ToolGroupCtx {
        is_group_end,
        show_expand_hint,
    }
}

fn tool_status_glyph(
    status: ToolState,
    spinner_idx: usize,
    theme: &Theme,
) -> (&'static str, Color) {
    match status {
        ToolState::Running => (
            BRAILLE_SPINNER[spinner_idx % BRAILLE_SPINNER.len()],
            theme.warning,
        ),
        ToolState::Done => ("●", theme.success),
        ToolState::Error => ("✗", theme.error),
        ToolState::Blocked => ("⊘", theme.error),
    }
}

fn truncate_tail_display(s: &str, max_cols: usize) -> String {
    if display_width(s) <= max_cols {
        return s.to_string();
    }
    if max_cols == 0 {
        return String::new();
    }
    if max_cols == 1 {
        return "…".to_string();
    }
    let budget = max_cols - 1;
    let mut rev_chars = Vec::new();
    let mut used = 0usize;
    for ch in s.chars().rev() {
        let w = char_width(ch);
        if used + w > budget {
            break;
        }
        used += w;
        rev_chars.push(ch);
    }
    let mut out = String::with_capacity(max_cols);
    out.push('…');
    for ch in rev_chars.into_iter().rev() {
        out.push(ch);
    }
    out
}

/// 对文件路径优先保留根目录/家目录前缀与尾部文件名（如 `~/…/crates/mizar-tui-ui/src/ui.rs`）。
fn truncate_path_middle(path: &str, max_cols: usize) -> String {
    if display_width(path) <= max_cols {
        return path.to_string();
    }
    if max_cols <= 4 {
        return truncate_display(path, max_cols);
    }

    let (prefix, rest) = if let Some(r) = path.strip_prefix("~/") {
        ("~/", r)
    } else if let Some(r) = path.strip_prefix('/') {
        ("/", r)
    } else if let Some(r) = path.strip_prefix("./") {
        ("./", r)
    } else {
        ("", path)
    };

    let segments: Vec<&str> = rest.split('/').filter(|s| !s.is_empty()).collect();
    if segments.len() >= 2 {
        let marker = format!("{prefix}…/");
        let marker_w = display_width(&marker);
        if marker_w >= max_cols {
            return truncate_display(path, max_cols);
        }
        let tail_budget = max_cols - marker_w;
        let last = segments[segments.len() - 1];
        if display_width(last) > tail_budget {
            let prefix_w = display_width(prefix);
            let avail = max_cols.saturating_sub(prefix_w);
            return format!("{prefix}{}", truncate_tail_display(last, avail));
        }

        let mut kept: Vec<&str> = vec![last];
        let mut cur_w = display_width(last);
        for seg in segments[..segments.len() - 1].iter().rev() {
            let seg_w = display_width(seg);
            if cur_w + 1 + seg_w > tail_budget {
                break;
            }
            cur_w += 1 + seg_w;
            kept.push(seg);
        }
        kept.reverse();
        return format!("{marker}{}", kept.join("/"));
    }

    let prefix_w = display_width(prefix);
    if prefix_w + 2 < max_cols {
        return format!("{prefix}{}", truncate_display(rest, max_cols - prefix_w));
    }
    truncate_display(path, max_cols)
}

fn fit_collapsed_tool_args(name: &str, args: &str, max_cols: usize) -> String {
    if max_cols == 0 || args.is_empty() {
        return String::new();
    }
    if display_width(args) <= max_cols {
        return args.to_string();
    }
    let single_path_like = !args.contains(' ') && !args.contains('\t') && args.contains('/');
    if is_path_arg_tool(name) || single_path_like {
        truncate_path_middle(args, max_cols)
    } else {
        truncate_display(args, max_cols)
    }
}

fn resolve_inline_meta(
    name: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
) -> Option<String> {
    if let Some(meta) = extra.and_then(|e| e.inline_meta.as_ref()) {
        return Some(meta.clone());
    }
    if matches!(status, ToolState::Running) {
        return None;
    }
    if name == "edit"
        && !matches!(status, ToolState::Error | ToolState::Blocked)
        && let Some(diff) = extra.and_then(|e| e.diff_raw.as_deref())
    {
        let (adds, dels) = count_diff_stats(diff);
        return format_diff_stat(adds, dels);
    }

    let full_text = extra
        .filter(|e| !e.full_content.is_empty())
        .map_or(output_preview, |e| e.full_content.as_str());

    if matches!(status, ToolState::Done) && is_read_only_compact_tool(name) {
        let trimmed = full_text.trim();
        if trimmed.is_empty() {
            return None;
        }
        return match name {
            "get_time" if !trimmed.contains('\n') => Some(trimmed.to_string()),
            "find" => {
                if trimmed.starts_with("No files found") {
                    Some("0 files".to_string())
                } else {
                    let n = full_text.lines().filter(|l| !l.trim().is_empty()).count();
                    Some(if n == 1 {
                        "1 file".to_string()
                    } else {
                        format!("{n} files")
                    })
                }
            }
            "list_dir" | "ls" => {
                if trimmed == "（空目录）" {
                    Some("empty".to_string())
                } else {
                    let n = full_text.lines().filter(|l| !l.trim().is_empty()).count();
                    Some(if n == 1 {
                        "1 item".to_string()
                    } else {
                        format!("{n} items")
                    })
                }
            }
            "skill" => Some("loaded".to_string()),
            _ => {
                if trimmed == "(empty file)" {
                    Some("empty".to_string())
                } else {
                    let n = full_text.lines().count();
                    Some(if n == 1 {
                        "1 line".to_string()
                    } else {
                        format!("{n} lines")
                    })
                }
            }
        };
    }

    if name == "bash" && matches!(status, ToolState::Done) {
        let lines = clean_bash_output_lines(full_text, false);
        if lines.len() > 2 {
            return Some(format!("{} lines", lines.len()));
        }
    }
    None
}

fn truncate_diff_line_utf8(line: &str, max_bytes: usize) -> (&str, bool) {
    if line.len() <= max_bytes {
        return (line, false);
    }
    let cut = line.floor_char_boundary(max_bytes);
    (&line[..cut], true)
}

fn compact_diff_preview(diff_raw: &str) -> (String, bool) {
    let mut changed = Vec::with_capacity(MAX_COLLAPSED_DIFF_SRC_LINES);
    let mut skipped_non_header = false;
    for line in diff_raw.lines() {
        if line.starts_with("--- ") || line.starts_with("+++ ") {
            continue;
        }
        if line.starts_with('+') || line.starts_with('-') {
            if changed.len() >= MAX_COLLAPSED_DIFF_SRC_LINES {
                skipped_non_header = true;
                break;
            }
            let (clipped, line_truncated) =
                truncate_diff_line_utf8(line, MAX_COLLAPSED_DIFF_LINE_BYTES);
            if line_truncated {
                skipped_non_header = true;
            }
            changed.push(clipped);
        } else if !line.trim().is_empty() {
            skipped_non_header = true;
        }
    }
    if changed.is_empty() {
        let mut fallback = Vec::with_capacity(MAX_COLLAPSED_DIFF_SRC_LINES);
        let mut fallback_truncated = false;
        for line in diff_raw.lines() {
            if fallback.len() >= MAX_COLLAPSED_DIFF_SRC_LINES {
                fallback_truncated = true;
                break;
            }
            let (clipped, line_truncated) =
                truncate_diff_line_utf8(line, MAX_COLLAPSED_DIFF_LINE_BYTES);
            if line_truncated {
                fallback_truncated = true;
            }
            fallback.push(clipped);
        }
        (fallback.join("\n"), fallback_truncated)
    } else {
        (changed.join("\n"), skipped_non_header)
    }
}

fn collapsed_preview_lines(name: &str, status: ToolState, output_preview: &str) -> Vec<String> {
    if matches!(status, ToolState::Error | ToolState::Blocked) {
        let max_lines = if is_read_only_compact_tool(name) {
            3
        } else {
            4
        };
        let lines: Vec<String> = if name == "bash" {
            let cleaned = clean_bash_output_lines(output_preview, true);
            let src = if cleaned.is_empty() {
                output_preview
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .collect()
            } else {
                cleaned
            };
            let start = src.len().saturating_sub(max_lines);
            src[start..].iter().map(|s| (*s).to_string()).collect()
        } else {
            output_preview
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(max_lines)
                .map(ToString::to_string)
                .collect()
        };
        return lines;
    }

    if is_read_only_compact_tool(name) || name == "write" {
        return Vec::new();
    }

    if name == "bash" {
        let cleaned = clean_bash_output_lines(output_preview, false);
        let start = cleaned.len().saturating_sub(2);
        return cleaned[start..].iter().map(|s| (*s).to_string()).collect();
    }

    output_preview
        .lines()
        .filter(|l| !l.trim().is_empty() && l.trim() != "(no output)")
        .take(2)
        .map(ToString::to_string)
        .collect()
}

fn collapsed_header_budget(
    glyph: &str,
    name: &str,
    abbr_args: &str,
    inline_meta: Option<&str>,
    show_expand_hint: bool,
    width: usize,
) -> (bool, bool, usize) {
    let fixed_w = display_width(glyph) + 1 + display_width(name) + 2;
    let hint_w = if show_expand_hint {
        display_width(EXPAND_HINT_INLINE)
    } else {
        0
    };
    let meta_w = inline_meta.map_or(0, |m| 3 + display_width(m));
    let full_args_w = display_width(abbr_args);

    if full_args_w == 0 {
        let rem = width.saturating_sub(fixed_w);
        if meta_w + hint_w <= rem {
            return (inline_meta.is_some(), show_expand_hint, 0);
        }
        if hint_w <= rem {
            return (false, show_expand_hint, 0);
        }
        if meta_w <= rem {
            return (inline_meta.is_some(), false, 0);
        }
        return (false, false, 0);
    }

    if fixed_w + full_args_w + meta_w + hint_w <= width {
        return (inline_meta.is_some(), show_expand_hint, full_args_w);
    }
    if width.saturating_sub(fixed_w + meta_w + hint_w) >= 16 {
        return (
            inline_meta.is_some(),
            show_expand_hint,
            width - fixed_w - meta_w - hint_w,
        );
    }
    if width.saturating_sub(fixed_w + hint_w) >= 16 {
        return (false, show_expand_hint, width - fixed_w - hint_w);
    }
    if width.saturating_sub(fixed_w + meta_w) >= 12 {
        return (inline_meta.is_some(), false, width - fixed_w - meta_w);
    }
    (false, false, width.saturating_sub(fixed_w))
}

pub(super) fn is_tool_expandable(
    item: &ChatItem,
    extra: Option<&ToolExtraMeta>,
    width: usize,
    theme: &Theme,
) -> bool {
    let ChatItem::ToolCall {
        name,
        args_summary,
        status,
        output_preview,
        ..
    } = item
    else {
        return false;
    };
    let cache_key = ToolExpandableCacheKey {
        width,
        fp: fingerprint_tool_content(name, args_summary, *status, output_preview, extra),
    };
    if let Some(hit) = TOOL_EXPANDABLE_CACHE.with(|c| c.borrow_mut().get(&cache_key)) {
        return hit;
    }
    let computed = compute_is_tool_expandable(
        name,
        args_summary,
        *status,
        output_preview,
        extra,
        width,
        theme,
    );
    TOOL_EXPANDABLE_CACHE.with(|c| c.borrow_mut().insert_lru(cache_key, computed));
    computed
}

fn compute_is_tool_expandable(
    name: &str,
    args_summary: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
    width: usize,
    theme: &Theme,
) -> bool {
    let body_w = width.saturating_sub(4).max(4);
    let inline_meta = resolve_inline_meta(name, status, output_preview, extra);
    let abbr_args = abbreviate_home_path(args_summary);
    let (glyph, _) = tool_status_glyph(status, 0, theme);
    let (_, _, args_budget) =
        collapsed_header_budget(glyph, name, &abbr_args, inline_meta.as_deref(), true, width);
    if !abbr_args.is_empty() && display_width(&abbr_args) > args_budget {
        return true;
    }

    let (_, extra_arg_lines) = format_expanded_tool_args(name, args_summary, extra);
    if !extra_arg_lines.is_empty() {
        return true;
    }

    let full_text = extra
        .filter(|e| !e.full_content.is_empty())
        .map_or(output_preview, |e| e.full_content.as_str());

    if !matches!(status, ToolState::Error | ToolState::Blocked)
        && let Some(diff_raw) = extra.and_then(|e| e.diff_raw.as_deref())
        && !diff_raw.trim().is_empty()
    {
        let (compact_src, stripped) = compact_diff_preview(diff_raw);
        if stripped {
            return true;
        }
        return render_diff_rich(&compact_src, body_w, theme).len() > 4;
    }

    if matches!(status, ToolState::Running) {
        return output_preview
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count()
            > 2;
    }

    if matches!(status, ToolState::Error | ToolState::Blocked) {
        let preview_src = if output_preview.trim().is_empty() {
            full_text
        } else {
            output_preview
        };
        let shown = collapsed_preview_lines(name, status, preview_src);
        let full_count = if name == "bash" {
            clean_bash_output_lines(full_text, true).len()
        } else {
            full_text.lines().filter(|l| !l.trim().is_empty()).count()
        };
        return full_count > shown.len() || shown.iter().any(|l| display_width(l) > body_w);
    }

    if is_read_only_compact_tool(name) {
        if name == "get_time" {
            let t = full_text.trim();
            return !t.is_empty() && (t.contains('\n') || display_width(t) > 40);
        }
        return !full_text.trim().is_empty();
    }

    if name == "write" {
        return !full_text.trim().is_empty();
    }

    let shown = collapsed_preview_lines(name, status, output_preview);
    if name == "bash" {
        let full_lines = clean_bash_output_lines(full_text, false);
        let has_trunc_note = full_text
            .lines()
            .any(|l| l.trim_start().starts_with("[Showing "));
        return full_lines.len() > shown.len()
            || has_trunc_note
            || shown.iter().any(|l| display_width(l) > body_w);
    }

    let full_count = full_text
        .lines()
        .filter(|l| !l.trim().is_empty() && l.trim() != "(no output)")
        .count();
    full_count > shown.len() || shown.iter().any(|l| display_width(l) > body_w)
}

struct ToolHeaderSpec<'a> {
    name: &'a str,
    args: &'a str,
    status: ToolState,
    inline_meta: Option<&'a str>,
    spinner_idx: usize,
    width: usize,
    theme: &'a Theme,
}

fn build_tool_header_base_spans(
    glyph: &'static str,
    glyph_color: Color,
    name: &str,
    fitted_args: &str,
    inline_meta: Option<&str>,
    theme: &Theme,
) -> Vec<Span<'static>> {
    let mut spans = vec![
        Span::styled(
            format!("{glyph} "),
            Style::default()
                .fg(glyph_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            name.to_string(),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("({fitted_args})"),
            Style::default().fg(theme.text_muted),
        ),
    ];
    if let Some(meta) = inline_meta {
        spans.push(Span::styled(
            format!(" · {meta}"),
            Style::default().fg(theme.text_dim),
        ));
    }
    spans
}

fn build_collapsed_tool_header(
    spec: &ToolHeaderSpec<'_>,
    show_expand_hint: bool,
) -> Vec<RenderedLine> {
    let (glyph, glyph_color) = tool_status_glyph(spec.status, spec.spinner_idx, spec.theme);
    let abbr_args = abbreviate_home_path(spec.args);
    let (use_meta, use_hint, args_budget) = collapsed_header_budget(
        glyph,
        spec.name,
        &abbr_args,
        spec.inline_meta,
        show_expand_hint,
        spec.width,
    );
    let fitted_args = fit_collapsed_tool_args(spec.name, &abbr_args, args_budget);
    let visible_meta = use_meta.then_some(spec.inline_meta).flatten();
    let mut spans = build_tool_header_base_spans(
        glyph,
        glyph_color,
        spec.name,
        &fitted_args,
        visible_meta,
        spec.theme,
    );
    if use_hint {
        spans.push(Span::styled(
            EXPAND_HINT_INLINE,
            Style::default().fg(spec.theme.text_dim),
        ));
    }

    let line = Line::from(spans);
    if line.width() <= spec.width {
        vec![RenderedLine::hard(line)]
    } else {
        wrap_styled_spans_rich(line.spans, None, None, spec.width)
    }
}

fn build_expanded_tool_header(spec: &ToolHeaderSpec<'_>) -> Vec<RenderedLine> {
    let (glyph, glyph_color) = tool_status_glyph(spec.status, spec.spinner_idx, spec.theme);
    let spans = build_tool_header_base_spans(
        glyph,
        glyph_color,
        spec.name,
        spec.args,
        spec.inline_meta,
        spec.theme,
    );
    wrap_styled_spans_rich(spans, None, None, spec.width)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tool_lines(
    name: &str,
    args_summary: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
    expanded: bool,
    group_ctx: ToolGroupCtx,
    spinner_idx: usize,
    width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let cache_key = ToolCardCacheKey {
        width,
        theme_id: theme.id,
        expanded,
        is_group_end: group_ctx.is_group_end,
        show_expand_hint: group_ctx.show_expand_hint,
        spinner_idx: if matches!(status, ToolState::Running) {
            spinner_idx
        } else {
            0
        },
        fp: fingerprint_tool_content(name, args_summary, status, output_preview, extra),
    };
    if let Some(hit) = TOOL_CARD_CACHE.with(|c| c.borrow_mut().get(&cache_key)) {
        return hit.to_vec();
    }
    let built = build_tool_lines_uncached(
        name,
        args_summary,
        status,
        output_preview,
        extra,
        expanded,
        group_ctx,
        spinner_idx,
        width,
        theme,
    );
    let shared: Arc<[RenderedLine]> = Arc::from(built.as_slice());
    TOOL_CARD_CACHE.with(|c| c.borrow_mut().insert_lru(cache_key, shared));
    built
}

#[allow(clippy::too_many_arguments)]
fn build_tool_lines_uncached(
    name: &str,
    args_summary: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
    expanded: bool,
    group_ctx: ToolGroupCtx,
    spinner_idx: usize,
    width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let inline_meta = resolve_inline_meta(name, status, output_preview, extra);
    let body_w = width.saturating_sub(4).max(4);
    let prefix = Span::styled("  │ ", Style::default().fg(theme.border));

    if !expanded {
        out.extend(build_collapsed_tool_header(
            &ToolHeaderSpec {
                name,
                args: args_summary,
                status,
                inline_meta: inline_meta.as_deref(),
                spinner_idx,
                width,
                theme,
            },
            group_ctx.show_expand_hint,
        ));

        if !matches!(status, ToolState::Error | ToolState::Blocked)
            && let Some(diff_raw) = extra.and_then(|e| e.diff_raw.as_deref())
            && !diff_raw.trim().is_empty()
        {
            let (compact_src, _) = compact_diff_preview(diff_raw);
            let diff_lines = render_diff_rich(&compact_src, body_w, theme);
            for rl in diff_lines.into_iter().take(4) {
                let mut spans = vec![prefix.clone()];
                spans.extend(rl.line.spans);
                out.push(RenderedLine {
                    line: Line::from(spans),
                    soft_wrapped: rl.soft_wrapped,
                });
            }
        } else {
            let preview_src = if matches!(status, ToolState::Error | ToolState::Blocked)
                && output_preview.trim().is_empty()
                && let Some(ext) = extra
                && !ext.full_content.is_empty()
            {
                ext.full_content.as_str()
            } else {
                output_preview
            };
            let body_color = match status {
                ToolState::Error | ToolState::Blocked => theme.error,
                ToolState::Running | ToolState::Done => theme.text_muted,
            };
            for raw in collapsed_preview_lines(name, status, preview_src) {
                let clipped = truncate_display(&raw, body_w);
                out.push(RenderedLine::hard(Line::from(vec![
                    prefix.clone(),
                    Span::styled(clipped, Style::default().fg(body_color)),
                ])));
            }
        }

        if group_ctx.is_group_end {
            out.push(RenderedLine::hard(Line::raw("")));
        }
        return out;
    }

    // 展开态（Ctrl+O）：完整保留参数/脚本、完整 Diff 与全文输出
    let (expanded_args, extra_arg_lines) = format_expanded_tool_args(name, args_summary, extra);
    out.extend(build_expanded_tool_header(&ToolHeaderSpec {
        name,
        args: &expanded_args,
        status,
        inline_meta: inline_meta.as_deref(),
        spinner_idx,
        width,
        theme,
    }));

    if !matches!(status, ToolState::Error | ToolState::Blocked)
        && let Some(diff_raw) = extra.and_then(|e| e.diff_raw.as_deref())
        && !diff_raw.trim().is_empty()
    {
        for rl in render_diff_rich(diff_raw, body_w, theme) {
            let mut spans = vec![prefix.clone()];
            spans.extend(rl.line.spans);
            out.push(RenderedLine {
                line: Line::from(spans),
                soft_wrapped: rl.soft_wrapped,
            });
        }
    } else {
        let mut emitted = 0usize;
        let max_lines = 160usize;
        let mut total_omitted = 0usize;

        if !extra_arg_lines.is_empty() {
            let total_arg_lines = extra_arg_lines.len();
            let take_n = total_arg_lines.min(max_lines);
            total_omitted += total_arg_lines.saturating_sub(take_n);
            for raw in extra_arg_lines.into_iter().take(take_n) {
                out.extend(wrap_styled_spans_rich(
                    vec![Span::styled(raw, Style::default().fg(theme.text_dim))],
                    Some(prefix.clone()),
                    Some(prefix.clone()),
                    width,
                ));
                emitted += 1;
            }
        }

        let body_text = if let Some(ext) = extra
            && !ext.full_content.is_empty()
        {
            ext.full_content.as_str()
        } else {
            output_preview
        };

        if !body_text.trim().is_empty() {
            let body_color = match status {
                ToolState::Error | ToolState::Blocked => theme.error,
                ToolState::Running | ToolState::Done => theme.text_muted,
            };
            let raw_lines: Vec<&str> = body_text.lines().collect();
            let rem = max_lines.saturating_sub(emitted);
            total_omitted += raw_lines.len().saturating_sub(rem);
            for raw in raw_lines.into_iter().take(rem) {
                out.extend(wrap_styled_spans_rich(
                    vec![Span::styled(
                        raw.to_string(),
                        Style::default().fg(body_color),
                    )],
                    Some(prefix.clone()),
                    Some(prefix.clone()),
                    width,
                ));
            }
        }

        if total_omitted > 0 {
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    format!("  │ … 其余 {total_omitted} 行已省略（完整记录请查 Ctrl+T）"),
                    Style::default().fg(theme.text_dim),
                )],
                None,
                None,
                width,
            ));
        }
    }

    if group_ctx.is_group_end {
        out.push(RenderedLine::hard(Line::raw("")));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::app::{abbreviate_home_with, summarize_tool_args};

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
    }

    #[test]
    fn edit_tightens_collapsed_diff_and_errors_stay_visible_without_expand() {
        let theme = Theme::mist_dark();
        let diff = [
            "--- src/ui.rs",
            "+++ src/ui.rs",
            "@@ -10,12 +10,12 @@",
            " ctx1",
            " ctx2",
            "-old1",
            "-old2",
            "-old3",
            "-old4",
            "-old5",
            "-old6",
            "+new1",
            "+new2",
            "+new3",
            "+new4",
            "+new5",
            "+new6",
            " ctx3",
        ]
        .join("\n");
        let extra = ToolExtraMeta {
            full_content: "Successfully replaced 1 edit(s) in src/ui.rs.".into(),
            diff_raw: Some(diff),
            full_args: r#"{"path":"src/ui.rs"}"#.into(),
            inline_meta: Some("+6 -6".into()),
            ..Default::default()
        };

        let collapsed = build_tool_lines(
            "edit",
            "src/ui.rs",
            ToolState::Done,
            "",
            Some(&extra),
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: true,
            },
            0,
            80,
            &theme,
        );
        let collapsed_txt: Vec<String> = collapsed.iter().map(|rl| line_text(&rl.line)).collect();
        // 1 头行 + 最多 4 行 Diff + 1 组尾空行 = 6 行，且第一行 Diff 直接展示真实修改而非 ctx1
        assert_eq!(
            collapsed_txt.len(),
            6,
            "collapsed:\n{}",
            collapsed_txt.join("\n")
        );
        assert!(collapsed_txt[0].contains("+6 -6"));
        assert!(!collapsed_txt.iter().any(|l| l.contains("ctx1")));
        assert!(collapsed_txt.iter().any(|l| l.contains("old1")));

        let expanded = build_tool_lines(
            "edit",
            "src/ui.rs",
            ToolState::Done,
            "",
            Some(&extra),
            true,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: false,
            },
            0,
            80,
            &theme,
        );
        let expanded_joined = expanded
            .iter()
            .map(|rl| line_text(&rl.line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            expanded_joined.contains("@@ -10,12 +10,12 @@"),
            "{expanded_joined}"
        );
        assert!(expanded_joined.contains("new6"), "{expanded_joined}");

        // 错误态无需 Ctrl+O 直接显示关键报错
        let err_lines = build_tool_lines(
            "read",
            "missing.rs",
            ToolState::Error,
            "File not found: missing.rs",
            None,
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: false,
            },
            0,
            80,
            &theme,
        );
        let err_joined = err_lines
            .iter()
            .map(|rl| line_text(&rl.line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(err_joined.contains("✗ read(missing.rs)"), "{err_joined}");
        assert!(
            err_joined.contains("File not found: missing.rs"),
            "{err_joined}"
        );
    }

    #[test]
    fn long_paths_use_middle_elision_and_narrow_width_does_not_panic() {
        let theme = Theme::mist_dark();
        let long_path = abbreviate_home_with(
            "/Users/alice/workspace/dsh-space/very-long-subproject-name/crates/mizar-tui-ui/src/ui.rs",
            Some("/Users/alice"),
        );
        assert!(long_path.starts_with("~/workspace/"));

        let elided = truncate_path_middle(&long_path, 48);
        assert!(elided.starts_with("~/…/"), "elided={elided}");
        assert!(
            elided.ends_with("crates/mizar-tui-ui/src/ui.rs"),
            "elided={elided}"
        );
        assert!(
            display_width(&elided) <= 48,
            "width={}",
            display_width(&elided)
        );

        // 20 列极窄宽度渲染不 panic
        let narrow = build_tool_lines(
            "read",
            &long_path,
            ToolState::Done,
            "line 1\nline 2",
            None,
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: true,
            },
            0,
            20,
            &theme,
        );
        assert!(!narrow.is_empty());

        // summarize_tool_args 对当前真实家目录同样执行 ~/ 缩写
        if let Some(home) = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            && let Some(home_str) = home.to_str()
            && !home_str.is_empty()
            && home_str != "/"
        {
            let raw = format!(r#"{{"path":"{home_str}/workspace/demo.rs"}}"#);
            assert_eq!(summarize_tool_args(&raw), "~/workspace/demo.rs");
        }
    }

    #[test]
    fn large_collapsed_edit_diff_truncates_before_word_diff_and_single_card_cache_isolates() {
        let theme = Theme::mist_dark();
        let long_old = format!("-{}", "alpha beta gamma ".repeat(1500));
        let long_new = format!("+{}", "alpha delta gamma ".repeat(1500));
        let mut lines = vec![
            "--- a.rs".to_string(),
            "+++ a.rs".to_string(),
            long_old,
            long_new,
        ];
        for i in 0..2000 {
            lines.push(format!("-old_{i}"));
            lines.push(format!("+new_{i}"));
        }
        let huge_diff = lines.join("\n");
        let (preview, stripped) = compact_diff_preview(&huge_diff);
        assert!(stripped);
        assert!(preview.lines().count() <= MAX_COLLAPSED_DIFF_SRC_LINES);
        assert!(
            preview
                .lines()
                .all(|l| l.len() <= MAX_COLLAPSED_DIFF_LINE_BYTES)
        );

        let extra_huge = ToolExtraMeta {
            diff_raw: Some(huge_diff),
            ..Default::default()
        };
        let item = ChatItem::ToolCall {
            id: "e1".into(),
            name: "edit".into(),
            args_summary: "a.rs".into(),
            status: ToolState::Done,
            output_preview: String::new(),
        };
        assert!(is_tool_expandable(&item, Some(&extra_huge), 100, &theme));

        // 单卡缓存：同组内只有变更参数的卡重算，未变动的兄弟卡命中缓存
        let sibling_key = ToolCardCacheKey {
            width: 80,
            theme_id: theme.id,
            expanded: false,
            is_group_end: false,
            show_expand_hint: false,
            spinner_idx: 0,
            fp: fingerprint_tool_content("read", "sib.rs", ToolState::Done, "ok", None),
        };
        let _ = build_tool_lines(
            "read",
            "sib.rs",
            ToolState::Done,
            "ok",
            None,
            false,
            ToolGroupCtx {
                is_group_end: false,
                show_expand_hint: false,
            },
            0,
            80,
            &theme,
        );
        let hit_before = TOOL_CARD_CACHE.with(|c| c.borrow_mut().get(&sibling_key));
        assert!(hit_before.is_some());

        // 另一张卡更新内容，不得清空 sibling_key 的单卡缓存
        let _ = build_tool_lines(
            "bash",
            "ls",
            ToolState::Running,
            "step-2",
            None,
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: false,
            },
            1,
            80,
            &theme,
        );
        let hit_after = TOOL_CARD_CACHE.with(|c| c.borrow_mut().get(&sibling_key));
        assert!(hit_after.is_some());
    }
}
