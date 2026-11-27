//! 工具调用卡片渲染、组上下文计算、窄宽度路径省略与可展开性判定。

use std::collections::HashMap;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::app::{
    ChatItem, ToolExtraMeta, ToolState, TranscriptRenderCache, abbreviate_home_path,
    count_diff_stats, extract_inline_meta, format_diff_stat, format_expanded_tool_args,
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
        ToolState::Drafting => 0,
        ToolState::Running => 1,
        ToolState::Done => 2,
        ToolState::Error | ToolState::Blocked => 3,
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

fn tool_status_glyph(status: ToolState, theme: &Theme) -> (&'static str, Color) {
    match status {
        ToolState::Drafting => ("○", theme.text_muted),
        ToolState::Running => ("●", theme.warning),
        ToolState::Done => ("●", theme.success),
        ToolState::Error | ToolState::Blocked => ("●", theme.error),
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

fn tool_body<'a>(output_preview: &'a str, extra: Option<&'a ToolExtraMeta>) -> &'a str {
    extra
        .filter(|e| !e.full_content.is_empty())
        .map_or(output_preview, |e| e.full_content.as_str())
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
    if matches!(status, ToolState::Drafting | ToolState::Running) {
        return None;
    }
    if name == "edit"
        && !matches!(status, ToolState::Error | ToolState::Blocked)
        && let Some(diff) = extra.and_then(|e| e.diff_raw.as_deref())
    {
        let (adds, dels) = count_diff_stats(diff);
        return format_diff_stat(adds, dels);
    }

    let full_text = tool_body(output_preview, extra);

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
        return extract_inline_meta(name, full_text, None, false);
    }
    None
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

    let inline_meta = resolve_inline_meta(name, *status, output_preview, extra);
    let abbr_args = abbreviate_home_path(args_summary);
    let (glyph, _) = tool_status_glyph(*status, theme);
    let (_, _, args_budget) = collapsed_header_budget(
        glyph,
        name,
        &abbr_args,
        inline_meta.as_deref(),
        false,
        width,
    );
    if !abbr_args.is_empty() && display_width(&abbr_args) > args_budget {
        return true;
    }

    if matches!(status, ToolState::Error | ToolState::Blocked)
        || extra
            .and_then(|e| e.diff_raw.as_deref())
            .is_some_and(|diff| !diff.trim().is_empty())
    {
        return true;
    }
    if tool_body(output_preview, extra).lines().any(|line| {
        !matches!(
            line.trim(),
            "" | "(no output)" | "(empty file)" | "（空目录）"
        )
    }) {
        return true;
    }
    if *status == ToolState::Drafting
        && let Some(extra) = extra
    {
        return args_summary.ends_with('…') || extra.full_args.contains('\n');
    }
    let (expanded_args, extra_arg_lines) = format_expanded_tool_args(name, args_summary, extra);
    !extra_arg_lines.is_empty() || abbreviate_home_path(&expanded_args) != abbr_args
}

struct ToolHeaderSpec<'a> {
    name: &'a str,
    args: &'a str,
    status: ToolState,
    inline_meta: Option<&'a str>,
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

fn build_collapsed_tool_header(spec: &ToolHeaderSpec<'_>, show_expand_hint: bool) -> RenderedLine {
    let (glyph, glyph_color) = tool_status_glyph(spec.status, spec.theme);
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

    RenderedLine::hard(Line::from(spans))
}

fn build_expanded_tool_header(spec: &ToolHeaderSpec<'_>) -> Vec<RenderedLine> {
    let (glyph, glyph_color) = tool_status_glyph(spec.status, spec.theme);
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
    width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let inline_meta = resolve_inline_meta(name, status, output_preview, extra);
    let header = ToolHeaderSpec {
        name,
        args: args_summary,
        status,
        inline_meta: inline_meta.as_deref(),
        width,
        theme,
    };
    if !expanded {
        out.push(build_collapsed_tool_header(
            &header,
            group_ctx.show_expand_hint,
        ));
        if group_ctx.is_group_end {
            out.push(RenderedLine::hard(Line::raw("")));
        }
        return out;
    }

    let body_w = width.saturating_sub(4).max(4);
    let prefix = Span::styled("  │ ", Style::default().fg(theme.border));
    let (expanded_args, extra_arg_lines) = format_expanded_tool_args(name, args_summary, extra);
    out.extend(build_expanded_tool_header(&ToolHeaderSpec {
        args: &expanded_args,
        ..header
    }));

    for raw in extra_arg_lines {
        out.extend(wrap_styled_spans_rich(
            vec![Span::styled(raw, Style::default().fg(theme.text_dim))],
            Some(prefix.clone()),
            Some(prefix.clone()),
            width,
        ));
    }

    let mut body = tool_body(output_preview, extra);
    if let Some(diff) = extra
        .and_then(|e| e.diff_raw.as_deref())
        .filter(|diff| !diff.trim().is_empty())
    {
        for rl in render_diff_rich(diff, body_w, theme) {
            let mut spans = vec![prefix.clone()];
            spans.extend(rl.line.spans);
            out.push(RenderedLine {
                line: Line::from(spans),
                soft_wrapped: rl.soft_wrapped,
            });
        }
        // 部分回执把同一份差异附在正文末尾，保留说明即可。
        body = body
            .trim_end()
            .strip_suffix(diff.trim_end())
            .unwrap_or(body);
    }
    if !body.trim().is_empty() {
        let color = match status {
            ToolState::Error | ToolState::Blocked => theme.error,
            _ => theme.text_muted,
        };
        for raw in body.lines() {
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(raw.to_owned(), Style::default().fg(color))],
                Some(prefix.clone()),
                Some(prefix.clone()),
                width,
            ));
        }
    }

    if group_ctx.is_group_end {
        out.push(RenderedLine::hard(Line::raw("")));
    }
    out
}
