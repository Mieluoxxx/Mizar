//! 增强型流式 Markdown、CJK 强强调修正、`pi-emphasis` 行内渲染与词级 Diff 引擎。

use std::borrow::Cow;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

use crate::editor::{char_width, display_width};
use crate::emphasis::{
    EmphasisConfig, decorate_action_lead_in, decorate_label_with_icon, is_mode_pill,
    pill_matches_label, render_pill_spans,
};
use crate::theme::Theme;

/// 带软换行元信息的渲染行，供选区复制时区分硬换行与折行续行。
#[derive(Debug, Clone)]
pub struct RenderedLine {
    pub line: Line<'static>,
    pub soft_wrapped: bool,
}

impl RenderedLine {
    pub fn hard(line: Line<'static>) -> Self {
        Self {
            line,
            soft_wrapped: false,
        }
    }
}

/// 兼容旧接口：使用默认主题与关闭胶囊装饰的简化渲染。
pub fn render_markdown_lines(
    text: &str,
    max_width: usize,
    base_style: Style,
    accent_color: Color,
) -> Vec<Line<'static>> {
    let mut theme = Theme::ansi16();
    theme.accent = accent_color;
    theme.link = accent_color;
    if let Some(fg) = base_style.fg {
        theme.text = fg;
    }
    let cfg = EmphasisConfig {
        enabled: false,
        round_caps: false,
        icons: false,
        guidance: false,
    };
    render_markdown_rich(text, max_width, &theme, &cfg)
        .into_iter()
        .map(|r| r.line)
        .collect()
}

/// 完整富文本 Markdown 渲染（支持表格、语法着色代码块、CJK 加粗修正与 `pi-emphasis` 胶囊/图标）。
pub fn render_markdown_rich(
    text: &str,
    max_width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let width = max_width.max(4);
    let base_style = Style::default().fg(theme.text);
    let dim_style = Style::default().fg(theme.text_dim);
    let mut out = Vec::new();
    let lines: Vec<&str> = text.split('\n').collect();
    let mut idx = 0usize;
    let mut tree_rails: Vec<bool> = Vec::new();
    let mut key_align_end = 0usize;
    let mut key_align_width: Option<usize> = None;

    while idx < lines.len() {
        let raw_line = lines[idx];
        let trimmed = raw_line.trim_start();

        // 1. 围栏代码块（``` 或 ~~~）
        if let Some(fence_ch) = fence_opener(trimmed) {
            tree_rails.clear();
            let fence_len = trimmed.bytes().take_while(|&b| b == fence_ch as u8).count();
            let lang = trimmed[fence_len..].trim();
            let label = if lang.is_empty() {
                "┌─ code".to_string()
            } else {
                format!("┌─ {lang}")
            };
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(label, dim_style)],
                None,
                None,
                width,
            ));
            idx += 1;
            let mut closed = false;
            while idx < lines.len() {
                let code_line = lines[idx];
                let ct = code_line.trim_start();
                let close_len = ct.bytes().take_while(|&b| b == fence_ch as u8).count();
                if close_len >= fence_len && ct[close_len..].trim().is_empty() {
                    closed = true;
                    idx += 1;
                    break;
                }
                let prefix = Span::styled("│ ", dim_style);
                let highlighted = highlight_code_line(code_line, lang, theme);
                out.extend(wrap_styled_spans_rich(
                    highlighted,
                    Some(prefix.clone()),
                    Some(prefix),
                    width,
                ));
                idx += 1;
            }
            if closed {
                out.push(RenderedLine::hard(Line::from(Span::styled(
                    "└─", dim_style,
                ))));
            }
            continue;
        }

        // 2. GFM 表格（表头含 `|` 且下一行匹配分割线）
        if idx + 1 < lines.len() && is_table_pair(lines[idx], lines[idx + 1]) {
            tree_rails.clear();
            let start = idx;
            idx += 2;
            while idx < lines.len()
                && lines[idx].trim().contains('|')
                && !lines[idx].trim().is_empty()
            {
                idx += 1;
            }
            out.extend(render_table_block(
                &lines[start..idx],
                width,
                theme,
                emphasis,
            ));
            continue;
        }

        // 3. 空行（若处于未闭合的树分支作用域中，按前瞻行决定是否续画竖脊）
        if raw_line.trim().is_empty() {
            if !tree_rails.is_empty()
                && let Some(guide) = active_tree_blank_guide(&lines, idx + 1, &tree_rails)
            {
                if guide.is_empty() {
                    out.push(RenderedLine::hard(Line::default()));
                } else {
                    out.push(RenderedLine::hard(Line::from(Span::styled(
                        guide, dim_style,
                    ))));
                }
                idx += 1;
                continue;
            }
            tree_rails.clear();
            out.push(RenderedLine::hard(Line::default()));
            idx += 1;
            continue;
        }

        // 4. 分割线
        if is_horizontal_rule(trimmed) {
            tree_rails.clear();
            let rule = "─".repeat(width.min(48));
            out.push(RenderedLine::hard(Line::from(Span::styled(
                rule, dim_style,
            ))));
            idx += 1;
            continue;
        }

        // 5. 标题
        if let Some((level, heading)) = parse_heading_level(trimmed) {
            tree_rails.clear();
            let unwrapped = heading
                .strip_prefix("**")
                .and_then(|s| s.strip_suffix("**"))
                .filter(|s| !s.is_empty() && !s.contains("**"))
                .unwrap_or(heading);
            let decorated = if level == 2 {
                decorate_label_with_icon(unwrapped, emphasis)
            } else {
                unwrapped.to_string()
            };
            let style = Style::default()
                .fg(if level <= 2 {
                    theme.accent
                } else {
                    theme.warning
                })
                .add_modifier(Modifier::BOLD);
            let spans = parse_inline_rich(&decorated, style, theme, emphasis, false);
            out.extend(wrap_styled_spans_rich(spans, None, None, width));
            idx += 1;
            continue;
        }

        // 5.5 树形行与带 `│` 前缀行（支持 `**├── ...**` 外层粗体剥离、状态栈维护及子列表归位）
        if let Some(tree_line) = parse_tree_line(raw_line) {
            match tree_line.kind {
                TreeKind::Branch { depth, is_open } => {
                    update_tree_rails(&mut tree_rails, depth, &tree_line.ancestor_rails, is_open);
                    let rest = tree_line.rest.trim_end();
                    let pref = Span::styled(tree_line.prefix.clone(), dim_style);
                    let cont = Span::styled(
                        tree_branch_cont_guide(&tree_rails, &tree_line.prefix),
                        dim_style,
                    );
                    let spans = if rest.is_empty() {
                        vec![Span::raw("")]
                    } else {
                        parse_inline_rich(rest, base_style, theme, emphasis, false)
                    };
                    out.extend(wrap_styled_spans_with_prefixes(
                        spans,
                        &[pref],
                        &[cont],
                        width,
                    ));
                    idx += 1;
                    continue;
                }
                TreeKind::RailLine => {
                    let rail_span = Span::styled(tree_line.prefix.clone(), dim_style);
                    let rest = tree_line.rest.trim_start();
                    if let Some(bullet_rest) = strip_unordered_bullet(rest) {
                        if idx >= key_align_end {
                            let (end, align_w) = compute_key_desc_align_run(
                                &lines,
                                idx,
                                &tree_rails,
                                width,
                                base_style,
                                theme,
                                emphasis,
                            );
                            key_align_end = end;
                            key_align_width = align_w;
                        }
                        let target_w = if idx < key_align_end {
                            key_align_width
                        } else {
                            None
                        };
                        out.extend(render_bullet_item(
                            bullet_rest,
                            Some(rail_span),
                            "",
                            target_w,
                            width,
                            base_style,
                            theme,
                            emphasis,
                        ));
                        idx += 1;
                        continue;
                    }
                    if let Some((num_prefix, ord_rest)) = parse_ordered_list(rest) {
                        let first = Span::styled(
                            format!("{num_prefix} "),
                            Style::default().fg(theme.text_muted),
                        );
                        let cont = Span::raw(" ".repeat(display_width(&format!("{num_prefix} "))));
                        let spans = parse_inline_rich(ord_rest, base_style, theme, emphasis, true);
                        out.extend(wrap_styled_spans_with_prefixes(
                            spans,
                            &[rail_span.clone(), first],
                            &[rail_span, cont],
                            width,
                        ));
                        idx += 1;
                        continue;
                    }
                    let spans = if rest.is_empty() {
                        vec![Span::raw("")]
                    } else {
                        parse_inline_rich(rest, base_style, theme, emphasis, true)
                    };
                    out.extend(wrap_styled_spans_with_prefixes(
                        spans,
                        std::slice::from_ref(&rail_span),
                        std::slice::from_ref(&rail_span),
                        width,
                    ));
                    idx += 1;
                    continue;
                }
            }
        }

        // 6. 无序列表（若处于活跃树作用域内，自动补齐外层 `│  ` 竖脊并对齐所属层级；成组 `key — desc` 对齐）
        if let Some(rest) = strip_unordered_bullet(trimmed) {
            let indent_cols = raw_line.len() - trimmed.len();
            let rail_span = if !tree_rails.is_empty() {
                Some(Span::styled(tree_child_guide(&tree_rails), dim_style))
            } else {
                None
            };
            let pad_cols = if !tree_rails.is_empty() {
                if indent_cols >= 4 { 2 } else { 0 }
            } else {
                indent_cols.min(8)
            };
            let pad = " ".repeat(pad_cols);
            if idx >= key_align_end {
                let (end, align_w) = compute_key_desc_align_run(
                    &lines,
                    idx,
                    &tree_rails,
                    width,
                    base_style,
                    theme,
                    emphasis,
                );
                key_align_end = end;
                key_align_width = align_w;
            }
            let target_w = if idx < key_align_end {
                key_align_width
            } else {
                None
            };
            out.extend(render_bullet_item(
                rest, rail_span, &pad, target_w, width, base_style, theme, emphasis,
            ));
            idx += 1;
            continue;
        }

        // 7. 有序列表
        if let Some((num_prefix, rest)) = parse_ordered_list(trimmed) {
            let indent_cols = raw_line.len() - trimmed.len();
            let mut first_prefixes = Vec::new();
            let mut cont_prefixes = Vec::new();
            let pad = if !tree_rails.is_empty() {
                let g = Span::styled(tree_child_guide(&tree_rails), dim_style);
                first_prefixes.push(g.clone());
                cont_prefixes.push(g);
                " ".repeat(if indent_cols >= 4 { 2 } else { 0 })
            } else {
                " ".repeat(indent_cols.min(8))
            };
            let first_str = format!("{pad}{num_prefix} ");
            let cont_str = " ".repeat(display_width(&first_str));
            first_prefixes.push(Span::styled(
                first_str,
                Style::default().fg(theme.text_muted),
            ));
            cont_prefixes.push(Span::raw(cont_str));
            let spans = parse_inline_rich(rest, base_style, theme, emphasis, true);
            out.extend(wrap_styled_spans_with_prefixes(
                spans,
                &first_prefixes,
                &cont_prefixes,
                width,
            ));
            idx += 1;
            continue;
        }

        // 8. 引用行
        if let Some(quoted) = trimmed.strip_prefix("> ") {
            tree_rails.clear();
            let bar = Span::styled("▎ ", dim_style);
            let quote_style = Style::default().fg(theme.text_muted);
            let spans = parse_inline_rich(quoted, quote_style, theme, emphasis, false);
            out.extend(wrap_styled_spans_rich(
                spans,
                Some(bar.clone()),
                Some(bar),
                width,
            ));
            idx += 1;
            continue;
        }

        // 9. 普通段落（树作用域下缩进≥2格的纯文本视为无短横线树子行，否则退出树作用域）
        if !tree_rails.is_empty() {
            let indent_cols = raw_line.len() - trimmed.len();
            if indent_cols >= 2 {
                let guide = Span::styled(tree_child_guide(&tree_rails), dim_style);
                let spans = parse_inline_rich(trimmed, base_style, theme, emphasis, true);
                out.extend(wrap_styled_spans_with_prefixes(
                    spans,
                    std::slice::from_ref(&guide),
                    std::slice::from_ref(&guide),
                    width,
                ));
                idx += 1;
                continue;
            }
            tree_rails.clear();
        }

        if let Some((pill_spans, rest_spans)) =
            split_leading_mode_pill_paragraph(raw_line, base_style, theme, emphasis, width)
        {
            out.extend(wrap_styled_spans_rich(pill_spans, None, None, width));
            if !rest_spans.is_empty() {
                out.extend(wrap_styled_spans_rich(rest_spans, None, None, width));
            }
        } else {
            let spans = parse_inline_rich(raw_line, base_style, theme, emphasis, true);
            out.extend(wrap_styled_spans_rich(spans, None, None, width));
        }
        idx += 1;
    }

    out
}

/// 若顶级段落以工作模式胶囊（如 `[[Investigate]]` 或去重后的 `**Investigate** [[Investigate]]`）开头
/// 且后续正文超出单行宽度，将模式胶囊独立成标题行，避免长段中文/英文正文粘连在胶囊末尾。
fn split_leading_mode_pill_paragraph(
    raw_line: &str,
    base: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    width: usize,
) -> Option<(Vec<Span<'static>>, Vec<Span<'static>>)> {
    let trimmed = raw_line.trim_start();
    let mut scan_pos = raw_line.len() - trimmed.len();
    if let Some((inner, strong_end)) = try_parse_strong(raw_line, scan_pos, "**")
        .or_else(|| try_parse_strong(raw_line, scan_pos, "__"))
    {
        let after_ws = skip_ascii_spaces(raw_line, strong_end);
        if let Some((pill_label, _)) = try_parse_pill(raw_line, after_ws)
            && pill_matches_label(pill_label, &inner)
        {
            scan_pos = after_ws;
        }
    }
    let (pill_label, pill_end) = try_parse_pill(raw_line, scan_pos)?;
    if !is_mode_pill(pill_label) {
        return None;
    }
    let tail = raw_line[pill_end..]
        .trim_start()
        .trim_start_matches([':', '：'])
        .trim_start();
    if tail.is_empty() {
        return None;
    }
    let pill_spans = render_pill_spans(pill_label, theme, emphasis);
    let rest_spans = parse_inline_rich(tail, base, theme, emphasis, true);
    let rest_w = spans_display_width(&rest_spans);
    let combined_w = spans_display_width(&pill_spans) + 1 + rest_w;
    if combined_w <= width && rest_w <= 24 {
        return None;
    }
    Some((pill_spans, rest_spans))
}

fn skip_ascii_spaces(s: &str, mut byte_idx: usize) -> usize {
    let bytes = s.as_bytes();
    while byte_idx < bytes.len() && bytes[byte_idx] == b' ' {
        byte_idx += 1;
    }
    byte_idx
}

fn strip_unordered_bullet(trimmed: &str) -> Option<&str> {
    trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Branch { depth: usize, is_open: bool },
    RailLine,
}

struct ParsedTreeLine {
    prefix: String,
    rest: String,
    kind: TreeKind,
    ancestor_rails: Vec<bool>,
}

fn is_box_border_continuation(ch: char) -> bool {
    matches!(ch, '─' | '┼' | '┬' | '┴' | '┤' | '┐' | '┘' | '═')
}

fn match_tree_branch_token(s: &str) -> Option<(usize, bool)> {
    for (tok, is_open) in [("├──", true), ("└──", false), ("├─", true), ("└─", false)]
    {
        if let Some(after) = s.strip_prefix(tok) {
            if after.chars().next().is_some_and(is_box_border_continuation) {
                return None;
            }
            return Some((tok.len(), is_open));
        }
    }
    None
}

/// 解析行首树线前缀（支持 `├──` / `└──` / `├─` / `└─` 及嵌套 `│  `）。
fn split_tree_prefix_raw(s: &str) -> Option<(usize, Vec<bool>, TreeKind, usize)> {
    let mut pos = 0usize;
    let mut ancestor_rails: Vec<bool> = Vec::new();

    loop {
        let sp_start = pos;
        while pos < s.len() && s.as_bytes()[pos] == b' ' {
            pos += 1;
        }
        let spaces = pos - sp_start;
        let push_closed_gap = |rails: &mut Vec<bool>, sp: usize| {
            let count = if rails.is_empty() && sp >= 2 {
                (sp + 1) / 3
            } else if !rails.is_empty() && sp >= 5 {
                (sp - 2) / 3
            } else {
                0
            };
            rails.extend(std::iter::repeat_n(false, count));
        };

        if let Some(after_bar) = s[pos..].strip_prefix('│') {
            push_closed_gap(&mut ancestor_rails, spaces);
            ancestor_rails.push(true);
            pos += '│'.len_utf8();
            let trailing_start = pos;
            while pos < s.len() && s.as_bytes()[pos] == b' ' {
                pos += 1;
            }
            let after_spaces = pos - trailing_start;
            let rest_preview = after_bar.trim_start_matches(' ');
            if !rest_preview.starts_with('│') && match_tree_branch_token(rest_preview).is_none() {
                return Some((pos, ancestor_rails, TreeKind::RailLine, pos));
            }
            // 若紧跟的是另一层 `│` 或 `├──`/`└──`，把多出的空白组折算为闭合层
            push_closed_gap(&mut ancestor_rails, after_spaces);
            continue;
        }

        if let Some((tok_len, is_open)) = match_tree_branch_token(&s[pos..]) {
            push_closed_gap(&mut ancestor_rails, spaces);
            pos += tok_len;
            while pos < s.len() && s.as_bytes()[pos] == b' ' {
                pos += 1;
            }
            let depth = ancestor_rails.len();
            return Some((
                pos,
                ancestor_rails,
                TreeKind::Branch { depth, is_open },
                pos,
            ));
        }

        return None;
    }
}

/// 识别树形行（兼容模型习惯性把整行写成 `**├── ...**` 或 `**│  ├── ...**` 的情况）。
fn parse_tree_line(raw_line: &str) -> Option<ParsedTreeLine> {
    if let Some((pref_end, rails, kind, rest_start)) = split_tree_prefix_raw(raw_line) {
        return Some(ParsedTreeLine {
            prefix: raw_line[..pref_end].to_string(),
            rest: raw_line[rest_start..].to_string(),
            kind,
            ancestor_rails: rails,
        });
    }

    let leading_ws = raw_line.len() - raw_line.trim_start_matches(' ').len();
    let after_ws = &raw_line[leading_ws..];
    for delim in ["**", "__"] {
        if after_ws.starts_with(delim)
            && let Some((inner, close_end)) = try_parse_strong(after_ws, 0, delim)
            && let Some((pref_end, rails, kind, rest_start)) = split_tree_prefix_raw(&inner)
        {
            let prefix = format!("{}{}", &raw_line[..leading_ws], &inner[..pref_end]);
            let inner_rest = &inner[rest_start..];
            let after_strong = &after_ws[close_end..];
            let rest = if inner_rest.is_empty() {
                after_strong.trim_start().to_string()
            } else {
                format!("{delim}{inner_rest}{delim}{after_strong}")
            };
            return Some(ParsedTreeLine {
                prefix,
                rest,
                kind,
                ancestor_rails: rails,
            });
        }
    }

    None
}

fn update_tree_rails(
    tree_rails: &mut Vec<bool>,
    depth: usize,
    ancestor_rails: &[bool],
    is_open: bool,
) {
    let mut next = Vec::with_capacity(depth + 1);
    for d in 0..depth {
        let open = ancestor_rails.get(d).copied() == Some(true)
            || tree_rails.get(d).copied().unwrap_or(false);
        next.push(open);
    }
    next.push(is_open);
    *tree_rails = next;
}

fn tree_child_guide(rails: &[bool]) -> String {
    let mut s = String::with_capacity(rails.len() * 3);
    for &open in rails {
        s.push_str(if open { "│  " } else { "   " });
    }
    s
}

fn tree_branch_cont_guide(rails: &[bool], prefix: &str) -> String {
    let outer_count = rails.len().saturating_sub(1);
    let mut s = tree_child_guide(&rails[..outer_count]);
    let used = display_width(&s);
    let target = display_width(prefix);
    if target > used {
        s.push_str(&" ".repeat(target - used));
    }
    s
}

fn active_tree_blank_guide(
    lines: &[&str],
    scan_from: usize,
    tree_rails: &[bool],
) -> Option<String> {
    let mut j = scan_from;
    while j < lines.len() && lines[j].trim().is_empty() {
        j += 1;
    }
    let next = lines.get(j)?;
    let next_trimmed = next.trim_start();
    if fence_opener(next_trimmed).is_some()
        || (j + 1 < lines.len() && is_table_pair(next, lines[j + 1]))
        || is_horizontal_rule(next_trimmed)
        || parse_heading_level(next_trimmed).is_some()
        || next_trimmed.starts_with("> ")
    {
        return None;
    }
    if let Some(next_tree) = parse_tree_line(next) {
        let limit = match next_tree.kind {
            TreeKind::Branch { depth, .. } => (depth + 1).min(tree_rails.len()),
            TreeKind::RailLine => tree_rails.len(),
        };
        let guide = tree_child_guide(&tree_rails[..limit]);
        return Some(guide.trim_end().to_string());
    }
    let has_open_rail = tree_rails.iter().any(|&o| o);
    if has_open_rail
        && (strip_unordered_bullet(next_trimmed).is_some()
            || parse_ordered_list(next_trimmed).is_some()
            || next.len() - next_trimmed.len() >= 2)
    {
        return Some(tree_child_guide(tree_rails).trim_end().to_string());
    }
    None
}

fn compute_key_desc_align_run(
    lines: &[&str],
    start: usize,
    tree_rails: &[bool],
    max_width: usize,
    base_style: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> (usize, Option<usize>) {
    let Some(first_raw) = lines.get(start) else {
        return (start, None);
    };
    let first_tree = parse_tree_line(first_raw);
    let first_is_rail = first_tree
        .as_ref()
        .is_some_and(|t| matches!(t.kind, TreeKind::RailLine));
    let first_indent = first_raw.len() - first_raw.trim_start().len();
    let first_rail_prefix = first_tree.as_ref().map(|t| t.prefix.as_str()).unwrap_or("");

    let mut end = start;
    let mut widths: Vec<usize> = Vec::new();
    while end < lines.len() {
        let line = lines[end];
        if line.trim().is_empty() {
            break;
        }
        let tree_opt = parse_tree_line(line);
        let rest_opt = match (&tree_opt, first_is_rail) {
            (Some(t), true)
                if matches!(t.kind, TreeKind::RailLine) && t.prefix == first_rail_prefix =>
            {
                strip_unordered_bullet(t.rest.trim_start())
            }
            (None, false) => {
                let tr = line.trim_start();
                if line.len() - tr.len() == first_indent {
                    strip_unordered_bullet(tr)
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(rest) = rest_opt else {
            break;
        };
        let Some((key_raw, tail_raw)) = rest.split_once(" — ") else {
            widths.clear();
            break;
        };
        if tail_raw.trim().is_empty() {
            widths.clear();
            break;
        }
        let key_spans = parse_inline_rich(key_raw, base_style, theme, emphasis, true);
        let kw = spans_display_width(&key_spans);
        if kw == 0 || kw > 24 {
            widths.clear();
            break;
        }
        widths.push(kw);
        end += 1;
    }

    if widths.len() < 2 {
        return (start + 1, None);
    }
    let min_w = widths.iter().copied().min().unwrap_or(0);
    let max_w = widths.iter().copied().max().unwrap_or(0);
    let guide_w = if first_is_rail {
        display_width(first_rail_prefix)
    } else if !tree_rails.is_empty() {
        tree_rails.len() * 3
    } else {
        first_indent.min(8)
    };
    if max_w - min_w > 20 || guide_w + 2 + max_w + 6 > max_width {
        return (end, None);
    }
    (end, Some(max_w))
}

#[allow(clippy::too_many_arguments)]
fn render_bullet_item(
    rest: &str,
    rail_span: Option<Span<'static>>,
    pad: &str,
    align_key_width: Option<usize>,
    width: usize,
    base_style: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let mut first_prefixes = Vec::new();
    let mut cont_prefixes = Vec::new();
    if let Some(rail) = rail_span {
        first_prefixes.push(rail.clone());
        cont_prefixes.push(rail);
    }
    first_prefixes.push(Span::styled(
        format!("{pad}• "),
        Style::default().fg(theme.text_muted),
    ));
    cont_prefixes.push(Span::raw(format!("{pad}  ")));

    let spans = if let Some(target_w) = align_key_width
        && let Some((key_raw, tail_raw)) = rest.split_once(" — ")
    {
        let mut combined = parse_inline_rich(key_raw, base_style, theme, emphasis, true);
        let kw = spans_display_width(&combined);
        let gap = target_w.saturating_sub(kw) + 1;
        combined.push(Span::styled(format!("{}— ", " ".repeat(gap)), base_style));
        combined.extend(parse_inline_rich(
            tail_raw, base_style, theme, emphasis, false,
        ));
        combined
    } else {
        parse_inline_rich(rest, base_style, theme, emphasis, true)
    };

    wrap_styled_spans_with_prefixes(spans, &first_prefixes, &cont_prefixes, width)
}

fn fence_opener(trimmed: &str) -> Option<char> {
    if trimmed.starts_with("```") {
        Some('`')
    } else if trimmed.starts_with("~~~") {
        Some('~')
    } else {
        None
    }
}

fn is_horizontal_rule(trimmed: &str) -> bool {
    let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() < 3 {
        return false;
    }
    let first = compact.chars().next().unwrap_or(' ');
    matches!(first, '-' | '*' | '_') && compact.chars().all(|c| c == first)
}

fn parse_heading_level(line: &str) -> Option<(usize, &str)> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes) {
        line[hashes..].strip_prefix(' ').map(|rest| (hashes, rest))
    } else {
        None
    }
}

fn parse_ordered_list(trimmed: &str) -> Option<(&str, &str)> {
    let dot = trimmed.find(". ")?;
    if dot == 0 || dot > 4 {
        return None;
    }
    if trimmed[..dot].bytes().all(|b| b.is_ascii_digit()) {
        Some((&trimmed[..dot + 1], &trimmed[dot + 2..]))
    } else {
        None
    }
}

fn is_table_pair(header: &str, delim: &str) -> bool {
    if !header.contains('|') {
        return false;
    }
    let t = delim.trim();
    if t.is_empty() || !t.contains('-') {
        return false;
    }
    let cells = split_table_row(t);
    !cells.is_empty()
        && cells.iter().all(|c| {
            let c = c.trim();
            !c.is_empty() && c.chars().all(|ch| matches!(ch, '-' | ':'))
        })
}

fn split_table_row(line: &str) -> Vec<&str> {
    let mut s = line.trim();
    if let Some(stripped) = s.strip_prefix('|') {
        s = stripped;
    }
    if let Some(stripped) = s.strip_suffix('|') {
        s = stripped;
    }
    s.split('|').map(str::trim).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColAlign {
    Left,
    Center,
    Right,
}

fn parse_col_align(cell: &str) -> ColAlign {
    let t = cell.trim();
    let left = t.starts_with(':');
    let right = t.ends_with(':');
    match (left, right) {
        (true, true) => ColAlign::Center,
        (false, true) => ColAlign::Right,
        _ => ColAlign::Left,
    }
}

fn spans_display_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|s| display_width(&s.content)).sum()
}

/// 按终端显示列宽裁剪 `Span` 列表，返回裁剪后的 spans 与真实显示宽度（供表格对齐与分栏 Diff 共用）。
fn clip_spans_to_width(
    spans: &[Span<'static>],
    max_w: usize,
    add_ellipsis: bool,
) -> (Vec<Span<'static>>, usize) {
    let total = spans_display_width(spans);
    if total <= max_w {
        return (spans.to_vec(), total);
    }
    let limit = if add_ellipsis && max_w >= 2 {
        max_w.saturating_sub(1)
    } else {
        max_w
    };
    let mut used = 0usize;
    let mut out = Vec::new();
    let mut last_style = Style::default();
    'outer: for sp in spans {
        last_style = sp.style;
        let mut part = String::new();
        for ch in sp.content.chars() {
            let cw = char_width(ch);
            if cw > 0 && used + cw > limit {
                if !part.is_empty() {
                    out.push(Span::styled(part, sp.style));
                }
                break 'outer;
            }
            part.push(ch);
            used += cw;
        }
        if !part.is_empty() {
            out.push(Span::styled(part, sp.style));
        }
    }
    if add_ellipsis && max_w >= 2 {
        out.push(Span::styled("…", last_style));
        used += 1;
    }
    (out, used)
}

fn parse_table_row_spans(
    cells: &[&str],
    cols: usize,
    base_style: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<Vec<Span<'static>>> {
    (0..cols)
        .map(|i| {
            let raw = cells.get(i).copied().unwrap_or("");
            let normalized: Cow<'_, str> = if raw.contains('\t') {
                Cow::Owned(raw.replace('\t', " "))
            } else {
                Cow::Borrowed(raw)
            };
            parse_inline_rich(&normalized, base_style, theme, emphasis, false)
        })
        .collect()
}

/// 在 `[mins[c], maxs[c]]` 区间内按 Water-filling（削峰填谷）分配总宽 `target`。
fn water_fill_clamped(mins: &[usize], maxs: &[usize], target: usize) -> Vec<usize> {
    let cols = mins.len();
    if cols == 0 {
        return Vec::new();
    }
    let min_sum: usize = mins.iter().sum();
    if target <= min_sum {
        return mins.to_vec();
    }
    let max_sum: usize = maxs.iter().sum();
    if target >= max_sum {
        return maxs.to_vec();
    }

    let mut lo = 0usize;
    let mut hi = *maxs.iter().max().unwrap_or(&0);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let s: usize = (0..cols).map(|c| mid.clamp(mins[c], maxs[c])).sum();
        if s <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let mut widths: Vec<usize> = (0..cols).map(|c| lo.clamp(mins[c], maxs[c])).collect();
    let rem = target.saturating_sub(widths.iter().sum::<usize>());
    if rem > 0 {
        // 只向 `lo + 1` 时实际可扩张的列派发余数，优先补给自然宽更大的列
        let mut candidates: Vec<usize> = (0..cols)
            .filter(|&c| widths[c] < maxs[c] && lo >= mins[c])
            .collect();
        candidates.sort_by(|&a, &b| maxs[b].cmp(&maxs[a]).then_with(|| a.cmp(&b)));
        for c in candidates.into_iter().take(rem) {
            widths[c] += 1;
        }
    }
    widths
}

/// 按 Water-filling（「保短缩长」）计算表格各列显示宽度，优先保护短列与表头完整。
fn allocate_table_col_widths(
    nat_widths: &[usize],
    header_widths: &[usize],
    avail: usize,
) -> Vec<usize> {
    let cols = nat_widths.len();
    if cols == 0 {
        return Vec::new();
    }
    let desired: Vec<usize> = nat_widths.iter().map(|&w| w.max(1)).collect();
    if desired.iter().map(|&w| w.max(3)).sum::<usize>() <= avail {
        return desired.into_iter().map(|w| w.max(3)).collect();
    }
    if desired.iter().sum::<usize>() <= avail {
        return desired;
    }

    // 短列（<= 12）全量保护；宽列按表头宽度适度保护（4..=10），避免被极端长列误伤
    const SHORT_COL_PROTECT: usize = 12;
    const HEADER_PROTECT_MAX: usize = 10;
    let floors: Vec<usize> = (0..cols)
        .map(|c| {
            if desired[c] <= SHORT_COL_PROTECT {
                desired[c]
            } else {
                header_widths[c]
                    .clamp(4, HEADER_PROTECT_MAX)
                    .min(desired[c])
            }
        })
        .collect();

    if floors.iter().sum::<usize>() <= avail {
        water_fill_clamped(&floors, &desired, avail)
    } else {
        // 极窄终端连保护底线都超出时，从保底宽度起用 water-filling 退让
        let min_bounds = vec![usize::from(avail >= cols); cols];
        water_fill_clamped(&min_bounds, &floors, avail)
    }
}

/// 将单元格富文本折行到指定列宽；极窄列（`< 4`）安全降级为单行裁剪加省略号。
fn wrap_table_cell(cell_spans: &[Span<'static>], col_w: usize) -> Vec<(Vec<Span<'static>>, usize)> {
    if col_w == 0 {
        return vec![(Vec::new(), 0)];
    }
    let nat_w = spans_display_width(cell_spans);
    if nat_w <= col_w {
        return vec![(cell_spans.to_vec(), nat_w)];
    }
    if col_w < 4 {
        return vec![clip_spans_to_width(cell_spans, col_w, true)];
    }

    let wrapped = wrap_styled_spans_with_prefixes(cell_spans.to_vec(), &[], &[], col_w);
    if wrapped.is_empty() {
        return vec![(Vec::new(), 0)];
    }
    wrapped
        .into_iter()
        .map(|rl| {
            let cw = spans_display_width(&rl.line.spans);
            if cw > col_w {
                clip_spans_to_width(&rl.line.spans, col_w, false)
            } else {
                (rl.line.spans, cw)
            }
        })
        .collect()
}

fn render_table_fallback_plain(
    parsed_rows: &[Vec<Vec<Span<'static>>>],
    max_width: usize,
    border_style: Style,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    for (r_idx, row_spans) in parsed_rows.iter().enumerate() {
        let mut joined = Vec::new();
        for (c_idx, cell) in row_spans.iter().enumerate() {
            if c_idx > 0 {
                joined.push(Span::styled(" │ ", border_style));
            }
            joined.extend(cell.iter().cloned());
        }
        if max_width < 2 {
            let (clipped, _) = clip_spans_to_width(&joined, max_width, false);
            out.push(RenderedLine::hard(Line::from(clipped)));
        } else {
            for rl in wrap_styled_spans_with_prefixes(joined, &[], &[], max_width) {
                let cw = spans_display_width(&rl.line.spans);
                if cw > max_width {
                    let (clipped, _) = clip_spans_to_width(&rl.line.spans, max_width, false);
                    out.push(RenderedLine {
                        line: Line::from(clipped),
                        soft_wrapped: rl.soft_wrapped,
                    });
                } else {
                    out.push(rl);
                }
            }
        }
        if r_idx == 0 {
            out.push(RenderedLine::hard(Line::from(Span::styled(
                "─".repeat(max_width),
                border_style,
            ))));
        }
    }
    out
}

fn render_table_block(
    rows: &[&str],
    max_width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let header = split_table_row(rows[0]);
    let delim = split_table_row(rows[1]);
    let cols = header.len().max(delim.len()).max(1);
    let aligns: Vec<ColAlign> = (0..cols)
        .map(|i| {
            delim
                .get(i)
                .map(|s| parse_col_align(s))
                .unwrap_or(ColAlign::Left)
        })
        .collect();

    // 先解析每一格的富文本 Span，再基于真实渲染宽度计算列宽与对齐填充，
    // 防止剥除 `` ` `` / `**` 或展开 `[[Pill]]` 后破坏右侧 `│` 边框对齐。
    let mut parsed_rows: Vec<Vec<Vec<Span<'static>>>> = Vec::with_capacity(rows.len() - 1);
    let header_style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    parsed_rows.push(parse_table_row_spans(
        &header,
        cols,
        header_style,
        theme,
        emphasis,
    ));
    let body_style = Style::default().fg(theme.text);
    for raw in &rows[2..] {
        let cells = split_table_row(raw);
        parsed_rows.push(parse_table_row_spans(
            &cells, cols, body_style, theme, emphasis,
        ));
    }

    let border_style = Style::default().fg(theme.border);
    let padded_chrome = cols * 3 + 1;
    let compact_chrome = cols + 1;
    let pad_x = if max_width >= padded_chrome + cols {
        1usize
    } else if max_width >= compact_chrome + cols {
        0usize
    } else {
        return render_table_fallback_plain(&parsed_rows, max_width, border_style);
    };

    let chrome = cols + 1 + cols * pad_x * 2;
    let avail = max_width.saturating_sub(chrome);

    let header_widths: Vec<usize> = (0..cols)
        .map(|c| spans_display_width(&parsed_rows[0][c]).max(1))
        .collect();
    let mut nat_widths = vec![1usize; cols];
    for r in &parsed_rows {
        for (c, cell_spans) in r.iter().enumerate() {
            nat_widths[c] = nat_widths[c].max(spans_display_width(cell_spans));
        }
    }
    let col_widths = allocate_table_col_widths(&nat_widths, &header_widths, avail);

    let make_border = |left: &str, mid: &str, right: &str| -> RenderedLine {
        let mut s = String::from(left);
        for (i, w) in col_widths.iter().enumerate() {
            if i > 0 {
                s.push_str(mid);
            }
            s.push_str(&"─".repeat(*w + pad_x * 2));
        }
        s.push_str(right);
        RenderedLine::hard(Line::from(Span::styled(s, border_style)))
    };

    let mut out = Vec::new();
    out.push(make_border("┌", "┬", "┐"));

    for (r_idx, row_spans) in parsed_rows.iter().enumerate() {
        let is_header = r_idx == 0;
        let wrapped_cells: Vec<Vec<(Vec<Span<'static>>, usize)>> = (0..cols)
            .map(|c_idx| wrap_table_cell(&row_spans[c_idx], col_widths[c_idx]))
            .collect();
        let row_height = wrapped_cells.iter().map(Vec::len).max().unwrap_or(1).max(1);

        for sub_row in 0..row_height {
            let mut spans = vec![Span::styled("│", border_style)];
            for c_idx in 0..cols {
                let w = col_widths[c_idx];
                if let Some((cell_line_spans, cw)) = wrapped_cells[c_idx].get(sub_row) {
                    let pad = w.saturating_sub(*cw);
                    let (left_pad, right_pad) = match aligns[c_idx] {
                        ColAlign::Left => (0, pad),
                        ColAlign::Right => (pad, 0),
                        ColAlign::Center => (pad / 2, pad - pad / 2),
                    };
                    if pad_x + left_pad > 0 {
                        spans.push(Span::raw(" ".repeat(pad_x + left_pad)));
                    }
                    spans.extend(cell_line_spans.iter().cloned());
                    if right_pad + pad_x > 0 {
                        spans.push(Span::raw(" ".repeat(right_pad + pad_x)));
                    }
                } else {
                    spans.push(Span::raw(" ".repeat(pad_x + w + pad_x)));
                }
                spans.push(Span::styled("│", border_style));
            }
            out.push(RenderedLine::hard(Line::from(spans)));
        }

        if is_header {
            out.push(make_border("├", "┼", "┤"));
        }
    }

    out.push(make_border("└", "┴", "┘"));
    out
}

fn truncate_cell(s: &str, max_w: usize) -> String {
    if display_width(s) <= max_w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    let limit = max_w.saturating_sub(1);
    for ch in s.chars() {
        let cw = char_width(ch);
        if w + cw > limit {
            out.push('…');
            return out;
        }
        out.push(ch);
        w += cw;
    }
    out
}

/// 轻量关键词/字符串/注释着色（覆盖 Rust/TS/JS/Python/Shell/JSON 常见语法）。
fn highlight_code_line(line: &str, lang: &str, theme: &Theme) -> Vec<Span<'static>> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//")
        || (trimmed.starts_with('#')
            && matches!(
                lang.to_ascii_lowercase().as_str(),
                "py" | "python" | "sh" | "bash" | "zsh" | "yaml" | "yml" | "toml" | ""
            ))
    {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(theme.syntax_comment),
        )];
    }

    let mut spans = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let mut plain_start = 0usize;
    let base_style = Style::default().fg(theme.code_fg);

    while i < bytes.len() {
        let b = bytes[i];
        // 行尾注释
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            if i > plain_start {
                spans.push(Span::styled(line[plain_start..i].to_string(), base_style));
            }
            spans.push(Span::styled(
                line[i..].to_string(),
                Style::default().fg(theme.syntax_comment),
            ));
            return spans;
        }
        // 字符串字面量
        if b == b'"' || b == b'\'' {
            if i > plain_start {
                spans.push(Span::styled(line[plain_start..i].to_string(), base_style));
            }
            let quote = b;
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                    continue;
                }
                if bytes[i] == quote {
                    i += 1;
                    break;
                }
                i += 1;
            }
            spans.push(Span::styled(
                line[start..i].to_string(),
                Style::default().fg(theme.syntax_string),
            ));
            plain_start = i;
            continue;
        }
        // 标识符 / 关键词 / 数字
        if b.is_ascii_alphanumeric() || b == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let token = &line[start..i];
            let token_color = if token.bytes().all(|c| c.is_ascii_digit() || c == b'_') {
                Some(theme.syntax_number)
            } else if is_code_keyword(token) {
                Some(theme.syntax_keyword)
            } else if token.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                Some(theme.syntax_type)
            } else {
                None
            };
            if let Some(color) = token_color {
                if start > plain_start {
                    spans.push(Span::styled(
                        line[plain_start..start].to_string(),
                        base_style,
                    ));
                }
                spans.push(Span::styled(token.to_string(), Style::default().fg(color)));
                plain_start = i;
            }
            continue;
        }
        // UTF-8 多字节字符安全步进
        let ch_len = line[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        i += ch_len;
    }

    if plain_start < line.len() {
        spans.push(Span::styled(line[plain_start..].to_string(), base_style));
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base_style));
    }
    spans
}

fn is_code_keyword(token: &str) -> bool {
    matches!(
        token,
        "fn" | "let"
            | "mut"
            | "pub"
            | "struct"
            | "enum"
            | "impl"
            | "trait"
            | "for"
            | "in"
            | "while"
            | "loop"
            | "match"
            | "if"
            | "else"
            | "return"
            | "async"
            | "await"
            | "use"
            | "mod"
            | "crate"
            | "self"
            | "Self"
            | "const"
            | "type"
            | "where"
            | "true"
            | "false"
            | "None"
            | "Some"
            | "Ok"
            | "Err"
            | "function"
            | "import"
            | "export"
            | "from"
            | "def"
            | "class"
    )
}

/// 判断字符是否属于“无空格分隔的东亚文字”（用于 CJK 加粗闭合修正，对齐 `dsh-TUI` `cjk-emphasis.ts`）。
fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    (0x3040..=0x30FF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x4E00..=0x9FFF).contains(&cp)
        || (0xAC00..=0xD7AF).contains(&cp)
        || (0xFF00..=0xFFEF).contains(&cp)
}

fn is_unicode_punct(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '，' | '。'
                | '：'
                | '；'
                | '！'
                | '？'
                | '（'
                | '）'
                | '【'
                | '】'
                | '《'
                | '》'
                | '「'
                | '」'
                | '『'
                | '』'
                | '〔'
                | '〕'
                | '〈'
                | '〉'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '、'
                | '…'
                | '—'
                | '–'
                | '～'
                | '·'
                | '・'
                | '°'
                | '％'
                | '￥'
        )
}

/// 匹配 `**...**` 或 `__...__`，跳过内部平衡反引号代码，并包含针对 `**证据与影响：**源码中` 的 CJK 标点闭合修正。
fn try_parse_strong(line: &str, byte_idx: usize, delim: &str) -> Option<(String, usize)> {
    let rest = line.get(byte_idx..)?.strip_prefix(delim)?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) || rest.starts_with(delim) {
        return None;
    }
    let delim_ch = delim.chars().next()?;
    let mut pos = 0usize;
    while pos < rest.len() {
        let tail = &rest[pos..];
        if tail.starts_with('\n') {
            return None;
        }
        if tail.starts_with('`') {
            if let Some((_, next_rel)) = try_parse_inline_code(rest, pos) {
                pos = next_rel;
                continue;
            }
            pos += 1;
            continue;
        }
        if tail.starts_with(delim) {
            let abs_end = pos;
            let inner = &rest[..abs_end];
            if inner.is_empty() {
                return None;
            }
            let prev_ch = inner.chars().next_back()?;
            let after = &rest[abs_end + delim.len()..];
            let next_ch = after.chars().next();
            if next_ch == Some(delim_ch) && inner.starts_with(delim_ch) {
                pos += 1;
                continue;
            }

            let valid_standard = !prev_ch.is_whitespace()
                && (!is_unicode_punct(prev_ch)
                    || next_ch.is_none_or(|c| c.is_whitespace() || is_unicode_punct(c)));
            let valid_cjk_adjacent = is_unicode_punct(prev_ch) && next_ch.is_some_and(is_cjk_char);

            if valid_standard || valid_cjk_adjacent {
                let next_byte = byte_idx + delim.len() + abs_end + delim.len();
                return Some((inner.to_string(), next_byte));
            }
            pos += delim.len();
            continue;
        }
        let ch_len = tail.chars().next().map(char::len_utf8).unwrap_or(1);
        pos += ch_len;
    }
    None
}

/// 匹配轻量行内斜体 `*...*`（排除 `**`、`crates/*/src` 与 `2 * 3` 等非强调星号）。
fn try_parse_italic(line: &str, byte_idx: usize) -> Option<(String, usize)> {
    let rest = line.get(byte_idx..)?.strip_prefix('*')?;
    if rest.is_empty() || rest.starts_with('*') || rest.starts_with(char::is_whitespace) {
        return None;
    }
    if byte_idx > 0
        && line[..byte_idx]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric())
    {
        return None;
    }
    let mut pos = 0usize;
    while pos < rest.len() {
        let tail = &rest[pos..];
        if tail.starts_with('\n') {
            return None;
        }
        if tail.starts_with('`') {
            if let Some((_, next_rel)) = try_parse_inline_code(rest, pos) {
                pos = next_rel;
                continue;
            }
            pos += 1;
            continue;
        }
        if tail.starts_with("**") {
            pos += 2;
            continue;
        }
        if tail.starts_with('*') {
            let inner = &rest[..pos];
            if inner.is_empty() {
                return None;
            }
            let prev_ch = inner.chars().next_back()?;
            let next_ch = rest[pos + 1..].chars().next();
            if !prev_ch.is_whitespace() && !next_ch.is_some_and(|c| c.is_ascii_alphanumeric()) {
                return Some((inner.to_string(), byte_idx + 1 + pos + 1));
            }
            pos += 1;
            continue;
        }
        let ch_len = tail.chars().next().map(char::len_utf8).unwrap_or(1);
        pos += ch_len;
    }
    None
}

fn try_parse_inline_code(line: &str, byte_idx: usize) -> Option<(&str, usize)> {
    let rest = line.get(byte_idx..)?;
    if !rest.starts_with('`') {
        return None;
    }
    let ticks = rest.bytes().take_while(|&b| b == b'`').count();
    let body = &rest[ticks..];
    let fence = "`".repeat(ticks);
    let mut pos = 0usize;
    while let Some(rel) = body[pos..].find(&fence) {
        let abs = pos + rel;
        let after = &body[abs + ticks..];
        if !after.starts_with('`') {
            let inner = body[..abs].trim();
            return Some((inner, byte_idx + ticks + abs + ticks));
        }
        pos = abs + ticks;
    }
    None
}

fn try_parse_pill(line: &str, byte_idx: usize) -> Option<(&str, usize)> {
    let rest = line.get(byte_idx..)?.strip_prefix("[[")?;
    let rel_end = rest.find("]]")?;
    let inner = &rest[..rel_end];
    if inner.is_empty() || inner.contains('[') || inner.contains(']') || inner.chars().count() > 32
    {
        return None;
    }
    Some((inner, byte_idx + 2 + rel_end + 2))
}

fn try_parse_link(line: &str, byte_idx: usize) -> Option<(&str, &str, usize)> {
    let rest = line.get(byte_idx..)?.strip_prefix('[')?;
    if rest.starts_with('[') {
        return None;
    }
    let close_bracket = rest.find("](")?;
    let label = &rest[..close_bracket];
    if label.is_empty() || label.contains('\n') {
        return None;
    }
    let after_open = &rest[close_bracket + 2..];
    let close_paren = after_open.find(')')?;
    let url = &after_open[..close_paren];
    if url.contains('\n') || url.contains(' ') {
        return None;
    }
    Some((
        label,
        url,
        byte_idx + 1 + close_bracket + 2 + close_paren + 1,
    ))
}

/// 富文本行内解析：支持反引号代码、CJK 修正粗体、`[[Pill]]`、动作引导词图标、`✓/✗/→/├──` 着色与 `[text](url)` 链接。
pub fn parse_inline_rich(
    line: &str,
    base: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    allow_lead_label_icon: bool,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut i = 0usize;
    let mut is_line_lead = allow_lead_label_icon;

    let flush_buf = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };

    while i < chars.len() {
        let (byte_idx, ch) = chars[i];

        // 1. 转义字符（\* \[ 等）
        if ch == '\\'
            && let Some(&(_, next_ch)) = chars.get(i + 1)
            && matches!(next_ch, '*' | '_' | '[' | ']' | '`' | '\\')
        {
            buf.push(next_ch);
            i += 2;
            is_line_lead = false;
            continue;
        }

        // 2. 行内代码
        if let Some((code, next_byte)) = try_parse_inline_code(line, byte_idx) {
            flush_buf(&mut buf, &mut spans);
            let code_style = Style::default().fg(theme.code_fg).bg(theme.code_bg);
            spans.push(Span::styled(code.to_string(), code_style));
            while i < chars.len() && chars[i].0 < next_byte {
                i += 1;
            }
            is_line_lead = false;
            continue;
        }

        // 3. pi-emphasis 状态胶囊 [[Word]]
        if let Some((pill_label, next_byte)) = try_parse_pill(line, byte_idx) {
            flush_buf(&mut buf, &mut spans);
            spans.extend(render_pill_spans(pill_label, theme, emphasis));
            while i < chars.len() && chars[i].0 < next_byte {
                i += 1;
            }
            is_line_lead = false;
            continue;
        }

        // 4. 粗体（含 CJK 邻接修正、同词胶囊去重、内部 `code`/`[[Pill]]`、行首标签与行动引导词图标）
        if let Some((inner, next_byte)) = try_parse_strong(line, byte_idx, "**")
            .or_else(|| try_parse_strong(line, byte_idx, "__"))
        {
            flush_buf(&mut buf, &mut spans);
            // 若紧跟同词 `[[Pill]]`（例如 `**Investigate** [[Investigate]]`），跳过冗余粗体避免双份图标/标签
            if !inner.contains("[[") && !inner.contains('`') {
                let after_ws = skip_ascii_spaces(line, next_byte);
                if let Some((pill_label, _)) = try_parse_pill(line, after_ws)
                    && pill_matches_label(pill_label, &inner)
                {
                    while i < chars.len() && chars[i].0 < after_ws {
                        i += 1;
                    }
                    continue;
                }
            }

            let preserve_base_color = base.add_modifier.contains(Modifier::BOLD);
            let strong_style = base.add_modifier(Modifier::BOLD);
            spans.extend(render_strong_inner_spans(
                &inner,
                strong_style,
                preserve_base_color,
                theme,
                emphasis,
                is_line_lead,
            ));
            while i < chars.len() && chars[i].0 < next_byte {
                i += 1;
            }
            is_line_lead = false;
            continue;
        }

        // 4.5 行内斜体 `*...*`
        if let Some((inner, next_byte)) = try_parse_italic(line, byte_idx) {
            flush_buf(&mut buf, &mut spans);
            let italic_style = base.add_modifier(Modifier::ITALIC);
            let inner_spans = parse_inline_rich(&inner, italic_style, theme, emphasis, false);
            spans.extend(inner_spans);
            while i < chars.len() && chars[i].0 < next_byte {
                i += 1;
            }
            is_line_lead = false;
            continue;
        }

        // 5. Markdown 链接 [label](url)
        if let Some((label, url, next_byte)) = try_parse_link(line, byte_idx) {
            flush_buf(&mut buf, &mut spans);
            spans.push(Span::styled(
                label.to_string(),
                base.fg(theme.link).add_modifier(Modifier::UNDERLINED),
            ));
            if label != url {
                spans.push(Span::styled(
                    format!(" ({url})"),
                    Style::default().fg(theme.text_dim),
                ));
            }
            while i < chars.len() && chars[i].0 < next_byte {
                i += 1;
            }
            is_line_lead = false;
            continue;
        }

        // 6. 树线连接符 ├── / └── / ├─ / └─ 降暗（排除表格框线续接符，必须位于字符索引）
        if emphasis.enabled
            && let Some((tok_len, _)) = match_tree_branch_token(&line[byte_idx..])
        {
            flush_buf(&mut buf, &mut spans);
            let connector = &line[byte_idx..byte_idx + tok_len];
            let char_count = connector.chars().count();
            spans.push(Span::styled(
                connector.to_string(),
                Style::default().fg(theme.text_dim),
            ));
            i += char_count;
            continue;
        }

        // 7. ✓ / ✗ / → 语义着色
        if emphasis.enabled && matches!(ch, '✓' | '✔' | '✗' | '✘' | '→') {
            flush_buf(&mut buf, &mut spans);
            let col = match ch {
                '✓' | '✔' => theme.success,
                '✗' | '✘' => theme.error,
                _ => theme.accent,
            };
            spans.push(Span::styled(ch.to_string(), Style::default().fg(col)));
            i += 1;
            is_line_lead = false;
            continue;
        }

        if !ch.is_whitespace() {
            is_line_lead = false;
        }
        buf.push(ch);
        i += 1;
    }

    flush_buf(&mut buf, &mut spans);
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

fn push_strong_segment(
    out: &mut Vec<Span<'static>>,
    seg: &str,
    strong_style: Style,
    preserve_base_color: bool,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    lead: bool,
) {
    let dim_style = Style::default().fg(theme.text_dim);
    let body = if lead && let Some((pref_end, _, _, rest_start)) = split_tree_prefix_raw(seg) {
        out.push(Span::styled(seg[..pref_end].to_string(), dim_style));
        &seg[rest_start..]
    } else {
        seg
    };
    if body.is_empty() {
        return;
    }
    let decorated = if lead {
        decorate_label_with_icon(body, emphasis)
    } else {
        decorate_action_lead_in(body, emphasis)
    };
    let span_style = if !preserve_base_color && emphasis.enabled && decorated != body {
        strong_style.fg(theme.accent)
    } else {
        strong_style
    };
    out.push(Span::styled(decorated, span_style));
}

/// 渲染粗体内部文本：支持内联代码 `` `code` ``、嵌套胶囊 `**[[Passed]]**`、树线前缀 `**├── ...**` 或混排。
fn render_strong_inner_spans(
    inner: &str,
    strong_style: Style,
    preserve_base_color: bool,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    is_line_lead: bool,
) -> Vec<Span<'static>> {
    if !inner.contains("[[") && !inner.contains('`') {
        let mut out = Vec::new();
        push_strong_segment(
            &mut out,
            inner,
            strong_style,
            preserve_base_color,
            theme,
            emphasis,
            is_line_lead,
        );
        if out.is_empty() {
            out.push(Span::styled(String::new(), strong_style));
        }
        return out;
    }

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut text_start = 0usize;
    let mut lead = is_line_lead;
    let flush_seg = |out: &mut Vec<Span<'static>>, seg: &str, lead_flag: &mut bool| {
        if !seg.is_empty() {
            push_strong_segment(
                out,
                seg,
                strong_style,
                preserve_base_color,
                theme,
                emphasis,
                *lead_flag,
            );
            *lead_flag = false;
        }
    };

    while pos < inner.len() {
        if inner[pos..].starts_with('`')
            && let Some((code, next_byte)) = try_parse_inline_code(inner, pos)
        {
            flush_seg(&mut out, &inner[text_start..pos], &mut lead);
            let code_style = Style::default()
                .fg(theme.code_fg)
                .bg(theme.code_bg)
                .add_modifier(Modifier::BOLD);
            out.push(Span::styled(code.to_string(), code_style));
            lead = false;
            pos = next_byte;
            text_start = pos;
            continue;
        }
        if inner[pos..].starts_with("[[")
            && let Some((pill_label, next_byte)) = try_parse_pill(inner, pos)
        {
            flush_seg(&mut out, &inner[text_start..pos], &mut lead);
            out.extend(render_pill_spans(pill_label, theme, emphasis));
            lead = false;
            pos = next_byte;
            text_start = pos;
            continue;
        }
        let ch_len = inner[pos..].chars().next().map(char::len_utf8).unwrap_or(1);
        pos += ch_len;
    }
    flush_seg(&mut out, &inner[text_start..], &mut lead);

    if out.is_empty() {
        out.push(Span::styled(inner.to_string(), strong_style));
    }
    out
}

/// 兼容旧签名的折行封装。
pub fn wrap_styled_spans(
    spans: Vec<Span<'static>>,
    prefix_first: Option<Span<'static>>,
    prefix_cont: Option<Span<'static>>,
    max_width: usize,
) -> Vec<Line<'static>> {
    wrap_styled_spans_rich(spans, prefix_first, prefix_cont, max_width)
        .into_iter()
        .map(|r| r.line)
        .collect()
}

/// 按终端显示列宽对富文本 `Span` 折行，并准确记录每一行是否为软换行续行。
pub fn wrap_styled_spans_rich(
    spans: Vec<Span<'static>>,
    prefix_first: Option<Span<'static>>,
    prefix_cont: Option<Span<'static>>,
    max_width: usize,
) -> Vec<RenderedLine> {
    wrap_styled_spans_with_prefixes(
        spans,
        prefix_first.as_slice(),
        prefix_cont.as_slice(),
        max_width,
    )
}

fn is_kinsoku_no_line_start(ch: char) -> bool {
    matches!(
        ch,
        '。' | '，'
            | '、'
            | '；'
            | '：'
            | '！'
            | '？'
            | '）'
            | '】'
            | '》'
            | '」'
            | '』'
            | '”'
            | '’'
            | '…'
            | '—'
            | '–'
            | '～'
            | '·'
            | '・'
            | '％'
            | '°'
            | '.'
            | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | ')'
            | ']'
            | '}'
    )
}

fn is_kinsoku_no_line_end(ch: char) -> bool {
    matches!(
        ch,
        '（' | '【' | '《' | '「' | '『' | '“' | '‘' | '(' | '[' | '{'
    )
}

enum RawToken {
    Space {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
    OpenPunct {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
    ClosePunct {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
    Atom {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
}

enum WrapItem {
    BreakSpace {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
    Atom {
        pieces: Vec<(String, Style)>,
        width: usize,
    },
}

fn tokenize_spans_for_wrap(spans: Vec<Span<'static>>, avail_w: usize) -> Vec<RawToken> {
    let mut raw = Vec::new();
    let mut seen_non_space = false;
    let mut idx = 0usize;

    while idx < spans.len() {
        // 1. 保持 Powerline 圆角胶囊（`\u{e0b6}` ... `\u{e0b4}`）跨 Span 原子不断开
        if spans[idx].content.as_ref() == "\u{e0b6}"
            && let Some(end_idx) = (idx + 1..spans.len().min(idx + 5))
                .find(|&j| spans[j].content.as_ref() == "\u{e0b4}")
        {
            let mut pieces = Vec::with_capacity(end_idx - idx + 1);
            let mut w = 0usize;
            for sp in &spans[idx..=end_idx] {
                w += display_width(&sp.content);
                pieces.push((sp.content.to_string(), sp.style));
            }
            seen_non_space = true;
            raw.push(RawToken::Atom { pieces, width: w });
            idx = end_idx + 1;
            continue;
        }

        let span = &spans[idx];
        let style = span.style;
        let text = span.content.as_ref();

        // 2. 短行内代码 / 背景词块尽量保持原子性，避免 `6 000` 或 `ToolRuntime` 被拆散
        if style.bg.is_some_and(|c| c != Color::Reset)
            && !text.starts_with(' ')
            && !text.contains('\t')
        {
            let w = display_width(text);
            if w > 0 && w <= avail_w.min(28) {
                seen_non_space = true;
                raw.push(RawToken::Atom {
                    pieces: vec![(text.to_string(), style)],
                    width: w,
                });
                idx += 1;
                continue;
            }
        }

        let mut chars = text.char_indices().peekable();
        while let Some(&(start_b, ch)) = chars.peek() {
            if ch == ' ' {
                let mut end_b = start_b + ch.len_utf8();
                let mut w = char_width(ch);
                chars.next();
                while let Some(&(nb, nch)) = chars.peek() {
                    if nch != ' ' {
                        break;
                    }
                    end_b = nb + nch.len_utf8();
                    w += char_width(nch);
                    chars.next();
                }
                let piece = (text[start_b..end_b].to_string(), style);
                if seen_non_space {
                    raw.push(RawToken::Space {
                        pieces: vec![piece],
                        width: w,
                    });
                } else {
                    raw.push(RawToken::Atom {
                        pieces: vec![piece],
                        width: w,
                    });
                }
                continue;
            }

            seen_non_space = true;
            if is_kinsoku_no_line_end(ch) {
                chars.next();
                raw.push(RawToken::OpenPunct {
                    pieces: vec![(ch.to_string(), style)],
                    width: char_width(ch),
                });
                continue;
            }
            if is_kinsoku_no_line_start(ch) {
                chars.next();
                raw.push(RawToken::ClosePunct {
                    pieces: vec![(ch.to_string(), style)],
                    width: char_width(ch),
                });
                continue;
            }
            if is_cjk_char(ch) || char_width(ch) >= 2 {
                chars.next();
                raw.push(RawToken::Atom {
                    pieces: vec![(ch.to_string(), style)],
                    width: char_width(ch),
                });
                continue;
            }

            // 连续 ASCII 单词/标识符/路径块：在非 ASCII 字符、全角标点处结束
            let mut end_b = start_b + ch.len_utf8();
            let mut w = char_width(ch);
            chars.next();
            while let Some(&(nb, nch)) = chars.peek() {
                if nch == ' ' || !nch.is_ascii() || is_cjk_char(nch) || char_width(nch) >= 2 {
                    break;
                }
                end_b = nb + nch.len_utf8();
                w += char_width(nch);
                chars.next();
            }
            raw.push(RawToken::Atom {
                pieces: vec![(text[start_b..end_b].to_string(), style)],
                width: w,
            });
        }

        idx += 1;
    }

    raw
}

fn merge_kinsoku_items(raw: Vec<RawToken>, avail_w: usize) -> Vec<WrapItem> {
    let mut items: Vec<WrapItem> = Vec::with_capacity(raw.len());
    let mut pending_open: Vec<(String, Style)> = Vec::new();
    let mut pending_open_w = 0usize;

    let flush_pending_open =
        |open: &mut Vec<(String, Style)>, open_w: &mut usize, items: &mut Vec<WrapItem>| {
            if !open.is_empty() {
                items.push(WrapItem::Atom {
                    pieces: std::mem::take(open),
                    width: std::mem::take(open_w),
                });
            }
        };

    for tok in raw {
        match tok {
            RawToken::Space { pieces, width } => {
                flush_pending_open(&mut pending_open, &mut pending_open_w, &mut items);
                items.push(WrapItem::BreakSpace { pieces, width });
            }
            RawToken::OpenPunct { pieces, width } => {
                pending_open.extend(pieces);
                pending_open_w += width;
            }
            RawToken::ClosePunct { pieces, width } => {
                if !pending_open.is_empty() {
                    pending_open.extend(pieces);
                    pending_open_w += width;
                    flush_pending_open(&mut pending_open, &mut pending_open_w, &mut items);
                } else if let Some(WrapItem::Atom {
                    pieces: prev_p,
                    width: prev_w,
                }) = items.last_mut()
                    && *prev_w + width <= avail_w
                {
                    prev_p.extend(pieces);
                    *prev_w += width;
                } else {
                    items.push(WrapItem::Atom { pieces, width });
                }
            }
            RawToken::Atom { mut pieces, width } => {
                if !pending_open.is_empty() {
                    let mut merged = std::mem::take(&mut pending_open);
                    let merged_w = std::mem::take(&mut pending_open_w) + width;
                    merged.append(&mut pieces);
                    items.push(WrapItem::Atom {
                        pieces: merged,
                        width: merged_w,
                    });
                } else {
                    items.push(WrapItem::Atom { pieces, width });
                }
            }
        }
    }

    flush_pending_open(&mut pending_open, &mut pending_open_w, &mut items);
    items
}

fn push_styled_piece(cur_spans: &mut Vec<Span<'static>>, text: String, style: Style) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = cur_spans.last_mut()
        && last.style == style
    {
        last.content.to_mut().push_str(&text);
    } else {
        cur_spans.push(Span::styled(text, style));
    }
}

fn is_preferred_atom_break(ch: char) -> bool {
    matches!(
        ch,
        '/' | '\\' | '-' | '_' | '.' | ',' | ';' | ':' | ' ' | '(' | '&' | '?' | '='
    )
}

fn split_pieces_at_separators(pieces: Vec<(String, Style)>) -> Vec<(Vec<(String, Style)>, usize)> {
    let mut chunks: Vec<(Vec<(String, Style)>, usize)> = Vec::new();
    let mut cur_pieces: Vec<(String, Style)> = Vec::new();
    let mut cur_w = 0usize;

    for (text, style) in pieces {
        let mut buf = String::new();
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            let cw = char_width(ch);
            buf.push(ch);
            cur_w += cw;
            let next_same_sep = matches!(ch, ':' | '.') && chars.peek() == Some(&ch);
            if is_preferred_atom_break(ch) && !next_same_sep {
                cur_pieces.push((std::mem::take(&mut buf), style));
                chunks.push((std::mem::take(&mut cur_pieces), std::mem::take(&mut cur_w)));
            }
        }
        if !buf.is_empty() {
            cur_pieces.push((buf, style));
        }
    }
    if !cur_pieces.is_empty() {
        chunks.push((cur_pieces, cur_w));
    }
    chunks
}

fn wrap_styled_spans_with_prefixes(
    spans: Vec<Span<'static>>,
    prefix_first: &[Span<'static>],
    prefix_cont: &[Span<'static>],
    max_width: usize,
) -> Vec<RenderedLine> {
    let width = max_width.max(2);
    let cont_prefix_w: usize = prefix_cont.iter().map(|p| display_width(&p.content)).sum();
    let avail_w = width.saturating_sub(cont_prefix_w).max(1);

    let raw = tokenize_spans_for_wrap(spans, avail_w);
    let items = merge_kinsoku_items(raw, avail_w);

    let mut lines: Vec<RenderedLine> = Vec::new();
    let mut cur_spans: Vec<Span<'static>> = Vec::new();
    let mut cur_col = 0usize;
    let mut is_first_visual_line = true;
    let mut line_has_content = false;
    let mut pending_space: Vec<(String, Style)> = Vec::new();
    let mut pending_space_w = 0usize;

    let push_prefixes =
        |target: &mut Vec<Span<'static>>, col: &mut usize, prefs: &[Span<'static>]| {
            for p in prefs {
                let pw: usize = p.content.chars().map(char_width).sum();
                *col += pw;
                target.push(p.clone());
            }
        };

    let flush_line = |lines: &mut Vec<RenderedLine>,
                      cur_spans: &mut Vec<Span<'static>>,
                      cur_col: &mut usize,
                      is_first: &mut bool,
                      has_content: &mut bool| {
        lines.push(RenderedLine {
            line: Line::from(std::mem::take(cur_spans)),
            soft_wrapped: !*is_first,
        });
        *is_first = false;
        *cur_col = 0;
        *has_content = false;
        push_prefixes(cur_spans, cur_col, prefix_cont);
    };

    push_prefixes(&mut cur_spans, &mut cur_col, prefix_first);

    for item in items {
        match item {
            WrapItem::BreakSpace { pieces, width: sw } => {
                pending_space.extend(pieces);
                pending_space_w += sw;
            }
            WrapItem::Atom { pieces, width: aw } => {
                if cur_col + pending_space_w + aw <= width {
                    for (t, st) in pending_space.drain(..) {
                        push_styled_piece(&mut cur_spans, t, st);
                    }
                    cur_col += std::mem::take(&mut pending_space_w);
                    for (t, st) in pieces {
                        push_styled_piece(&mut cur_spans, t, st);
                    }
                    cur_col += aw;
                    line_has_content = true;
                } else if aw <= avail_w && (line_has_content || cur_col > 0) {
                    pending_space.clear();
                    pending_space_w = 0;
                    flush_line(
                        &mut lines,
                        &mut cur_spans,
                        &mut cur_col,
                        &mut is_first_visual_line,
                        &mut line_has_content,
                    );
                    for (t, st) in pieces {
                        push_styled_piece(&mut cur_spans, t, st);
                    }
                    cur_col += aw;
                    line_has_content = true;
                } else {
                    // 单个原子词块本身超过续行可用宽度：若当前行已有内容先整块换行，再按分隔符/字符安全切分
                    if line_has_content {
                        pending_space.clear();
                        pending_space_w = 0;
                        flush_line(
                            &mut lines,
                            &mut cur_spans,
                            &mut cur_col,
                            &mut is_first_visual_line,
                            &mut line_has_content,
                        );
                    } else if cur_col + pending_space_w <= width {
                        for (t, st) in pending_space.drain(..) {
                            push_styled_piece(&mut cur_spans, t, st);
                        }
                        cur_col += std::mem::take(&mut pending_space_w);
                    } else {
                        pending_space.clear();
                        pending_space_w = 0;
                    }

                    for (sub_pieces, sub_w) in split_pieces_at_separators(pieces) {
                        if sub_w <= avail_w && cur_col + sub_w <= width {
                            for (t, st) in sub_pieces {
                                push_styled_piece(&mut cur_spans, t, st);
                            }
                            cur_col += sub_w;
                            line_has_content = true;
                            continue;
                        }
                        if sub_w <= avail_w && line_has_content {
                            flush_line(
                                &mut lines,
                                &mut cur_spans,
                                &mut cur_col,
                                &mut is_first_visual_line,
                                &mut line_has_content,
                            );
                            for (t, st) in sub_pieces {
                                push_styled_piece(&mut cur_spans, t, st);
                            }
                            cur_col += sub_w;
                            line_has_content = true;
                            continue;
                        }
                        for (t, st) in sub_pieces {
                            let mut chunk = String::new();
                            for ch in t.chars() {
                                let cw = char_width(ch);
                                if cw > 0 && cur_col + cw > width && cur_col > 0 {
                                    push_styled_piece(
                                        &mut cur_spans,
                                        std::mem::take(&mut chunk),
                                        st,
                                    );
                                    flush_line(
                                        &mut lines,
                                        &mut cur_spans,
                                        &mut cur_col,
                                        &mut is_first_visual_line,
                                        &mut line_has_content,
                                    );
                                }
                                chunk.push(ch);
                                cur_col += cw;
                                line_has_content = true;
                            }
                            push_styled_piece(&mut cur_spans, chunk, st);
                        }
                    }
                }
            }
        }
    }

    if !pending_space.is_empty() && cur_col + pending_space_w <= width {
        for (t, st) in pending_space {
            push_styled_piece(&mut cur_spans, t, st);
        }
    }

    lines.push(RenderedLine {
        line: Line::from(cur_spans),
        soft_wrapped: !is_first_visual_line,
    });
    lines
}

/// 使用 `similar` 进行行内词级 Diff 渲染（隔离共同前导缩进，宽屏可切分双栏）。
pub fn render_diff_rich(diff_text: &str, max_width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let raw_lines: Vec<&str> = diff_text
        .lines()
        .filter(|l| !l.starts_with("--- ") && !l.starts_with("+++ "))
        .collect();

    // 若宽屏且包含可对齐的 +/- 块，使用双栏 Split Diff；否则使用词级高亮 Unified Diff
    if max_width >= 96 && has_hunk_changes(&raw_lines) {
        return render_split_diff(&raw_lines, max_width, theme);
    }
    render_unified_word_diff(&raw_lines, max_width, theme)
}

fn has_hunk_changes(lines: &[&str]) -> bool {
    lines
        .iter()
        .any(|l| l.starts_with('+') || l.starts_with('-'))
}

fn expand_tabs(s: &str) -> String {
    s.replace('\t', "   ")
}

/// 隔离共同前导缩进后再做词级 Diff（借鉴 `pi` `renderIntraLineDiff`）。
fn intra_line_word_diff_spans(
    old_line: &str,
    new_line: &str,
    theme: &Theme,
) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let old_exp = expand_tabs(old_line);
    let new_exp = expand_tabs(new_line);
    let common_indent_bytes = old_exp
        .char_indices()
        .zip(new_exp.char_indices())
        .take_while(|((_, a), (_, b))| a == b && a.is_whitespace())
        .map(|((idx, ch), _)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);

    let indent = &old_exp[..common_indent_bytes];
    let old_rest = &old_exp[common_indent_bytes..];
    let new_rest = &new_exp[common_indent_bytes..];

    let base_del = Style::default().fg(theme.diff_del_fg).bg(theme.diff_del_bg);
    let base_add = Style::default().fg(theme.diff_add_fg).bg(theme.diff_add_bg);
    let word_del = base_del
        .bg(theme.diff_del_word_bg)
        .add_modifier(Modifier::BOLD);
    let word_add = base_add
        .bg(theme.diff_add_word_bg)
        .add_modifier(Modifier::BOLD);

    let mut del_spans = Vec::new();
    let mut add_spans = Vec::new();
    if !indent.is_empty() {
        del_spans.push(Span::styled(indent.to_string(), base_del));
        add_spans.push(Span::styled(indent.to_string(), base_add));
    }

    let diff = TextDiff::from_words(old_rest, new_rest);
    for change in diff.iter_all_changes() {
        let val = change.value().to_string();
        match change.tag() {
            ChangeTag::Equal => {
                del_spans.push(Span::styled(val.clone(), base_del));
                add_spans.push(Span::styled(val, base_add));
            }
            ChangeTag::Delete => del_spans.push(Span::styled(val, word_del)),
            ChangeTag::Insert => add_spans.push(Span::styled(val, word_add)),
        }
    }

    (del_spans, add_spans)
}

fn render_unified_word_diff(lines: &[&str], max_width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        if let Some(removed_body) = line.strip_prefix('-') {
            let mut removed = vec![removed_body];
            let mut j = i + 1;
            while j < lines.len() {
                if let Some(b) = lines[j].strip_prefix('-') {
                    removed.push(b);
                    j += 1;
                } else {
                    break;
                }
            }
            let mut added = Vec::new();
            while j < lines.len() {
                if let Some(b) = lines[j].strip_prefix('+') {
                    added.push(b);
                    j += 1;
                } else {
                    break;
                }
            }

            if removed.len() == 1 && added.len() == 1 {
                let (del_spans, add_spans) =
                    intra_line_word_diff_spans(removed[0], added[0], theme);
                let del_pref = Span::styled(
                    "- ",
                    Style::default()
                        .fg(theme.diff_del_fg)
                        .bg(theme.diff_del_bg)
                        .add_modifier(Modifier::BOLD),
                );
                let add_pref = Span::styled(
                    "+ ",
                    Style::default()
                        .fg(theme.diff_add_fg)
                        .bg(theme.diff_add_bg)
                        .add_modifier(Modifier::BOLD),
                );
                out.extend(wrap_styled_spans_rich(
                    del_spans,
                    Some(del_pref),
                    Some(Span::raw("  ")),
                    max_width,
                ));
                out.extend(wrap_styled_spans_rich(
                    add_spans,
                    Some(add_pref),
                    Some(Span::raw("  ")),
                    max_width,
                ));
            } else {
                for r in removed {
                    let s = Style::default().fg(theme.diff_del_fg).bg(theme.diff_del_bg);
                    out.extend(wrap_styled_spans_rich(
                        vec![Span::styled(format!("- {}", expand_tabs(r)), s)],
                        None,
                        None,
                        max_width,
                    ));
                }
                for a in added {
                    let s = Style::default().fg(theme.diff_add_fg).bg(theme.diff_add_bg);
                    out.extend(wrap_styled_spans_rich(
                        vec![Span::styled(format!("+ {}", expand_tabs(a)), s)],
                        None,
                        None,
                        max_width,
                    ));
                }
            }
            i = j;
            continue;
        }

        if let Some(added_body) = line.strip_prefix('+') {
            let s = Style::default().fg(theme.diff_add_fg).bg(theme.diff_add_bg);
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(format!("+ {}", expand_tabs(added_body)), s)],
                None,
                None,
                max_width,
            ));
        } else if line.starts_with("@@") {
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    line.to_string(),
                    Style::default().fg(theme.accent),
                )],
                None,
                None,
                max_width,
            ));
        } else {
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    expand_tabs(line),
                    Style::default().fg(theme.text_muted),
                )],
                None,
                None,
                max_width,
            ));
        }
        i += 1;
    }
    out
}

/// 宽屏双栏 Split Diff（对齐 `dsh-TUI` `SplitDiffView.tsx`：纯增删块保留单栏全宽，普通修改左右双栏词级高亮）。
fn render_split_diff(lines: &[&str], max_width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let has_del = lines.iter().any(|l| l.starts_with('-'));
    let has_add = lines.iter().any(|l| l.starts_with('+'));
    if !(has_del && has_add) {
        return render_unified_word_diff(lines, max_width, theme);
    }

    let pane_w = (max_width.saturating_sub(3)) / 2;
    let sep = Span::styled(" │ ", Style::default().fg(theme.border));
    let mut out = Vec::new();
    let mut i = 0usize;

    let pad_spans_to =
        |spans: Vec<Span<'static>>, bg: Color, target_w: usize| -> Vec<Span<'static>> {
            let (mut clipped, used) = clip_spans_to_width(&spans, target_w, false);
            if used < target_w {
                clipped.push(Span::styled(
                    " ".repeat(target_w - used),
                    Style::default().bg(bg),
                ));
            }
            clipped
        };

    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("@@") {
            out.push(RenderedLine::hard(Line::from(Span::styled(
                truncate_cell(line, max_width),
                Style::default().fg(theme.accent),
            ))));
            i += 1;
            continue;
        }
        if line.starts_with('-') || line.starts_with('+') {
            let mut del_lines = Vec::new();
            let mut add_lines = Vec::new();
            while i < lines.len() && lines[i].starts_with('-') {
                del_lines.push(&lines[i][1..]);
                i += 1;
            }
            while i < lines.len() && lines[i].starts_with('+') {
                add_lines.push(&lines[i][1..]);
                i += 1;
            }
            let rows = del_lines.len().max(add_lines.len());
            for r in 0..rows {
                let left_opt = del_lines.get(r).copied();
                let right_opt = add_lines.get(r).copied();
                let (left_spans, left_bg, right_spans, right_bg) = match (left_opt, right_opt) {
                    (Some(l), Some(ri)) => {
                        let (ls, rs) = intra_line_word_diff_spans(l, ri, theme);
                        let mut full_l = vec![Span::styled(
                            "- ",
                            Style::default().fg(theme.diff_del_fg).bg(theme.diff_del_bg),
                        )];
                        full_l.extend(ls);
                        let mut full_r = vec![Span::styled(
                            "+ ",
                            Style::default().fg(theme.diff_add_fg).bg(theme.diff_add_bg),
                        )];
                        full_r.extend(rs);
                        (full_l, theme.diff_del_bg, full_r, theme.diff_add_bg)
                    }
                    (Some(l), None) => (
                        vec![Span::styled(
                            format!("- {}", expand_tabs(l)),
                            Style::default().fg(theme.diff_del_fg).bg(theme.diff_del_bg),
                        )],
                        theme.diff_del_bg,
                        Vec::new(),
                        theme.card_bg,
                    ),
                    (None, Some(ri)) => (
                        Vec::new(),
                        theme.card_bg,
                        vec![Span::styled(
                            format!("+ {}", expand_tabs(ri)),
                            Style::default().fg(theme.diff_add_fg).bg(theme.diff_add_bg),
                        )],
                        theme.diff_add_bg,
                    ),
                    (None, None) => unreachable!(),
                };
                let mut combined = pad_spans_to(left_spans, left_bg, pane_w);
                combined.push(sep.clone());
                combined.extend(pad_spans_to(right_spans, right_bg, pane_w));
                out.push(RenderedLine::hard(Line::from(combined)));
            }
            continue;
        }

        let ctx_text = expand_tabs(line.strip_prefix(' ').unwrap_or(line));
        let left_ctx = vec![Span::styled(
            ctx_text.clone(),
            Style::default().fg(theme.text_muted),
        )];
        let right_ctx = vec![Span::styled(
            ctx_text,
            Style::default().fg(theme.text_muted),
        )];
        let mut combined = pad_spans_to(left_ctx, Color::Reset, pane_w);
        combined.push(sep.clone());
        combined.extend(pad_spans_to(right_ctx, Color::Reset, pane_w));
        out.push(RenderedLine::hard(Line::from(combined)));
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_cjk_adjacent_bold_and_emphasis_pills_and_tables() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let md = "## Finding\n**证据与影响：**源码中发现 [[Done]] 与 [[Needs you]]，**Check:** 运行测试。\n\n| 列A | 列B |\n| --- | ---: |\n| 左 | 123 |";
        let rendered = render_markdown_rich(md, 80, &theme, &cfg);
        let text: String = rendered
            .iter()
            .map(|r| {
                r.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("证据与影响："), "cjk bold broken:\n{text}");
        assert!(!text.contains("**证据"), "raw asterisks leaked:\n{text}");
        assert!(text.contains("Done"));
        assert!(text.contains("Needs\u{00a0}you"));
        assert!(text.contains("┌"));
        assert!(text.contains("123"));
    }

    #[test]
    fn repairs_bold_tree_prefixes_indent_and_vertical_rails() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: false,
            icons: false,
            guidance: true,
        };
        let md = "\
### Toolkit

**├── 核心工具 (直接可用)**
- **read** — 读文件
- **tool_search** — 检索工具

**├── 延迟加载工具 (需先搜索激活)**

**│  ├── FFF 文件查找器**
- **find_files** — 模糊文件名搜索
- **grep** — 内容搜索

**│  └── 外挂 MCP 命名空间**
- **codegraph** — 代码图谱
- **magpie-image** — 文生图

**└── 说明**
- 外部工具默认不挂载上下文
- 想让我用某个具体能力可以直接说

普通总结段落应回到第0列。";

        let rendered = render_markdown_rich(md, 96, &theme, &cfg);
        let texts: Vec<String> = rendered
            .iter()
            .map(|r| {
                r.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        let joined = texts.join("\n");

        // 1. 树线前缀降暗且不带 BOLD，而正文保持 BOLD
        let core_line = &rendered
            .iter()
            .find(|r| r.line.spans.iter().any(|s| s.content.contains("核心工具")))
            .expect("core branch")
            .line;
        assert_eq!(core_line.spans[0].content, "├── ");
        assert_eq!(core_line.spans[0].style.fg, Some(theme.text_dim));
        assert!(
            !core_line.spans[0]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            core_line.spans[1]
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );

        // 2. 子列表保留外层竖脊并缩进对齐，同时成组 key — desc 对齐 `—`
        let read_row = texts.iter().find(|l| l.contains("read")).unwrap();
        let ts_row = texts.iter().find(|l| l.contains("tool_search")).unwrap();
        assert!(read_row.starts_with("│  • read"), "read row: {read_row:?}");
        assert_eq!(
            read_row.find('—'),
            ts_row.find('—'),
            "em-dash should align:\n{read_row}\n{ts_row}"
        );

        let ff_row = texts.iter().find(|l| l.contains("find_files")).unwrap();
        assert!(
            ff_row.starts_with("│  │  • find_files"),
            "ff row: {ff_row:?}"
        );

        let cg_row = texts.iter().find(|l| l.contains("codegraph")).unwrap();
        assert!(
            cg_row.starts_with("│     • codegraph"),
            "cg row: {cg_row:?}"
        );

        let note_row = texts.iter().find(|l| l.contains("外部工具")).unwrap();
        assert!(
            note_row.starts_with("   • 外部工具"),
            "note row: {note_row:?}"
        );

        // 3. 树分支之间的空行保持竖脊不断裂
        assert!(
            texts.iter().any(|l| l == "│") && texts.iter().any(|l| l == "│  │"),
            "missing vertical rail continuity across blanks:\n{joined}"
        );

        // 4. 关闭树之后空行后的普通段落回到第 0 列
        let summary_row = texts.iter().find(|l| l.contains("普通总结段落")).unwrap();
        assert!(
            summary_row.starts_with("普通总结段落"),
            "summary leaked tree indent: {summary_row:?}"
        );
    }

    #[test]
    fn parses_bold_with_inline_code_and_preserves_heading_hierarchy() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };
        let md = "## **Medium - `grep` 硬依赖外部 `rg`**\n### **Sub**\n1. **`cargo fmt --all` 单独提交**，让 CI 变绿。\n2. 普通文本与 *斜体*。";
        let rendered = render_markdown_rich(md, 96, &theme, &cfg);
        let texts: Vec<String> = rendered
            .iter()
            .map(|r| {
                r.line
                    .spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        let joined = texts.join("\n");
        assert!(!joined.contains("**"), "raw asterisks leaked:\n{joined}");
        assert!(joined.contains("cargo fmt --all 单独提交"));

        // H3 内部粗体保持 warning 色，不被 link 覆盖
        let h3_span = rendered[1]
            .line
            .spans
            .iter()
            .find(|s| s.content.contains("Sub"))
            .expect("h3 span");
        assert_eq!(h3_span.style.fg, Some(theme.warning));
        assert!(h3_span.style.add_modifier.contains(Modifier::BOLD));

        // 列表项内普通粗体用 crisp text+BOLD 而非全句变蓝，其中的内联代码保留 code_bg+BOLD
        let item1_code = rendered[2]
            .line
            .spans
            .iter()
            .find(|s| s.content.contains("cargo fmt --all"))
            .expect("bold inline code span");
        assert_eq!(item1_code.style.bg, Some(theme.code_bg));
        assert!(item1_code.style.add_modifier.contains(Modifier::BOLD));

        let item1_bold = rendered[2]
            .line
            .spans
            .iter()
            .find(|s| s.content.contains("单独提交"))
            .expect("bold prose span");
        assert_eq!(item1_bold.style.fg, Some(theme.text));
        assert!(item1_bold.style.add_modifier.contains(Modifier::BOLD));

        let italic_span = rendered[3]
            .line
            .spans
            .iter()
            .find(|s| s.content.contains("斜体"))
            .expect("italic span");
        assert!(italic_span.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn wraps_words_paths_and_cjk_kinsoku_without_splitting_tokens_or_orphaning_punct() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: true,
            icons: true,
            guidance: true,
        };

        // 1. 句末中文句号不能单独落到新行行首
        let md_punct = "要么在启动探测阶段就提示缺依赖。";
        // "要么在启动探测阶段就提示缺依赖" = 30 列，"." = 2 列，width=30 时必须把"赖。"一起推到下一行
        let r_punct = render_markdown_rich(md_punct, 30, &theme, &cfg);
        assert_eq!(r_punct.len(), 2);
        let p_line1: String = r_punct[1]
            .line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(p_line1, "赖。");

        // 2. 英文标识符与行内代码路径不能从中间腰斩
        let md_tokens = "\
建议先在高风险点 (`Session`、`ToolRuntime`、`McpTransport`) 替换。\n\
- 仍引用已删除的 `crates/dsh-webserver` 与根 `src/main.rs`。\n\
- 手写 `poll/read` 可改用树里已有的 `rustix` (`crossterm` 已引入)。\n\
1. 单独提交，让 CI 变绿（[[Done]] 后可解锁合并门禁）。";
        for w in [42usize, 46, 52, 60] {
            let rendered = render_markdown_rich(md_tokens, w, &theme, &cfg);
            let lines_str: Vec<String> = rendered
                .iter()
                .map(|r| r.line.spans.iter().map(|s| s.content.as_ref()).collect())
                .collect();
            for l in &lines_str {
                assert!(display_width(l) <= w, "exceeded width {w}: {l:?}");
                let trimmed_l = l.trim_start();
                assert!(
                    !trimmed_l.starts_with('。')
                        && !trimmed_l.starts_with('、')
                        && !trimmed_l.starts_with('）'),
                    "kinsoku violated at line start (w={w}): {l:?}\nall:\n{}",
                    lines_str.join("\n")
                );
                assert!(
                    !l.ends_with('（'),
                    "kinsoku violated at line end (w={w}): {l:?}"
                );
            }
            let joined = lines_str.join("\n");
            assert!(
                joined.contains("ToolRuntime"),
                "ToolRuntime split across lines (w={w}):\n{joined}"
            );
            assert!(
                joined.contains("src/main.rs"),
                "src/main.rs split across lines (w={w}):\n{joined}"
            );
            assert!(
                joined.contains("rustix"),
                "rustix split across lines (w={w}):\n{joined}"
            );
            // 验证圆角胶囊左右帽永远在同一行
            for l in &lines_str {
                if l.contains('\u{e0b6}') || l.contains('\u{e0b4}') {
                    assert!(
                        l.contains('\u{e0b6}') && l.contains('\u{e0b4}'),
                        "powerline pill split across lines: {l:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn table_preserves_short_col_and_wraps_long_rich_cells_across_widths() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig::default();
        let md = "\
| 类别 | 具体 |
| --- | --- |
| 读代码 | 读文件、全文搜索（`grep`）、按文件名查找、解释某段逻辑、追调用链、对照 `docs/plugins.md` 拓扑 |
| 跑命令 | `cargo check -p <crate>`、`cargo check --workspace`、`cargo test --workspace`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo fmt`、`git` |
| 改代码 | 用精确编辑做局部手术改动，新增 crate / 加工具插件 / 加 LLM 协议适配器；跨 crate 枚举改完自动跑全量 check |
| 守规矩 | 按仓库约定来：不用 `mod.rs`、错误用 `thiserror`、副作用包 `ctx.effect`、工具描述写英文、`inject` 漏写要查、feature 组合要验最小集 |";

        for w in [80usize, 100, 120, 140] {
            let rendered = render_markdown_rich(md, w, &theme, &cfg);
            assert!(
                rendered.len() > 8,
                "expected multi-line wrapped rows at width {w}, got {}",
                rendered.len()
            );

            let first_w = spans_display_width(&rendered[0].line.spans);
            assert!(
                first_w <= w,
                "width {w}: table width {first_w} exceeded {w}"
            );

            let mut row_texts = Vec::with_capacity(rendered.len());
            for (idx, rl) in rendered.iter().enumerate() {
                let rw = spans_display_width(&rl.line.spans);
                assert_eq!(
                    rw, first_w,
                    "width {w} row {idx} mismatch ({rw} != {first_w})"
                );
                assert_eq!(
                    rl.line.width(),
                    rw,
                    "width {w} row {idx} ratatui Line::width mismatch"
                );
                assert!(!rl.soft_wrapped, "table rows must stay hard lines");
                let text: String = rl.line.spans.iter().map(|s| s.content.as_ref()).collect();
                assert!(
                    !text.contains('…'),
                    "unexpected ellipsis at width {w}: {text:?}"
                );
                row_texts.push(text);
            }

            // 边框完整闭合，且数据行之间不插额外横线（仅表头下 1 条 `├──┼──┤`）
            assert!(
                row_texts[0].starts_with('┌') && row_texts[0].ends_with('┐'),
                "top border broken at {w}: {:?}",
                row_texts[0]
            );
            let last = row_texts.last().unwrap();
            assert!(
                last.starts_with('└') && last.ends_with('┘'),
                "bottom border broken at {w}: {last:?}"
            );
            let sep_count = row_texts
                .iter()
                .filter(|r| r.starts_with('├') && r.ends_with('┤'))
                .count();
            assert_eq!(sep_count, 1, "expected single header separator at {w}");
            for row in &row_texts[1..row_texts.len() - 1] {
                if !row.starts_with('├') {
                    assert!(
                        row.starts_with('│') && row.ends_with('│'),
                        "data border broken at {w}: {row:?}"
                    );
                }
            }

            // 第 1 列短类别必须完整保留
            let joined = row_texts.join("\n");
            for cat in ["类别", "读代码", "跑命令", "改代码", "守规矩"] {
                assert!(
                    joined.contains(cat),
                    "short column label {cat:?} truncated at width {w}:\n{joined}"
                );
            }

            // 第 2 列折行后保留完整命令与全部字符（不丢字、不截断）
            let col2_space_joined: String = row_texts
                .iter()
                .filter(|r| r.starts_with('│'))
                .filter_map(|r| r.split('│').nth(2).map(str::trim))
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let col2_compact: String = col2_space_joined.chars().filter(|&c| c != ' ').collect();
            for ascii_cmd in [
                "docs/plugins.md",
                "cargo check -p <crate>",
                "cargo check --workspace",
                "cargo test --workspace",
                "cargo clippy --workspace --all-targets -- -D warnings",
                "cargo fmt",
                "git",
                "ctx.effect",
            ] {
                assert!(
                    col2_space_joined.contains(ascii_cmd),
                    "lost command {ascii_cmd:?} at width {w}:\n{col2_space_joined}\nfull:\n{joined}"
                );
            }
            for raw_sentence in [
                "读文件、全文搜索（grep）、按文件名查找、解释某段逻辑、追调用链、对照docs/plugins.md拓扑",
                "用精确编辑做局部手术改动，新增crate/加工具插件/加LLM协议适配器；跨crate枚举改完自动跑全量check",
                "按仓库约定来：不用mod.rs、错误用thiserror、副作用包ctx.effect、工具描述写英文、inject漏写要查、feature组合要验最小集",
            ] {
                assert!(
                    col2_compact.contains(raw_sentence),
                    "lost text {raw_sentence:?} at width {w}:\n{col2_compact}\nfull:\n{joined}"
                );
            }

            // 确认折行后内联代码背景样式仍在
            let has_code_bg_on_warnings = rendered.iter().any(|rl| {
                rl.line
                    .spans
                    .iter()
                    .any(|s| s.content.contains("warnings") && s.style.bg == Some(theme.code_bg))
            });
            assert!(
                has_code_bg_on_warnings,
                "inline code style lost after wrapping at width {w}"
            );
        }
    }

    #[test]
    fn table_water_filling_protects_multiple_short_cols_and_handles_narrow_terminals() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: false,
            icons: false,
            guidance: true,
        };

        // 1. 多列混合：3 个短列 + 1 个超长列，验证短列完整保护、对齐符号与链接样式保留
        let multi_md = "\
| ID | 类别 | 状态 | 详细说明与链接 |
| ---: | :---: | --- | --- |
| 1 | 读代码 | [[Done]] | 查看 [设计文档](https://example.com/docs) 与 `crates/mizar-tui-ui/src/markdown.rs` 折行逻辑 |
| 2 | 守规矩 | [[Failed]] | 必须保持 `cargo clippy --workspace --all-targets -- -D warnings` 零警告通过 |";

        for w in [52usize, 80, 120] {
            let rendered = render_markdown_rich(multi_md, w, &theme, &cfg);
            let first_w = spans_display_width(&rendered[0].line.spans);
            assert!(first_w <= w, "width {w}: {first_w} > {w}");
            for rl in &rendered {
                assert_eq!(spans_display_width(&rl.line.spans), first_w);
                assert_eq!(rl.line.width(), first_w);
            }
            let joined: String = rendered
                .iter()
                .map(|r| {
                    r.line
                        .spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n");
            for token in [
                "ID",
                "类别",
                "状态",
                "读代码",
                "守规矩",
                "Done",
                "Failed",
                "设计文档",
            ] {
                assert!(
                    joined.contains(token),
                    "missing {token:?} at width {w}:\n{joined}"
                );
            }
            assert!(
                rendered.iter().any(|rl| rl.line.spans.iter().any(|s| {
                    s.content.contains("设计文档") && s.style.fg == Some(theme.link)
                })),
                "link color lost at width {w}"
            );
        }

        // 2. 极窄 20 列终端：2 列宽表仍能保住 6 列短列、折行不 panic 且不越界
        let two_col_md = "\
| 类别 | 具体 |
| --- | --- |
| 读代码 | `cargo clippy --workspace --all-targets -- -D warnings` 与全量测试 |";
        let narrow_20 = render_markdown_rich(two_col_md, 20, &theme, &cfg);
        let w20_first = spans_display_width(&narrow_20[0].line.spans);
        assert!(w20_first <= 20, "20-col table overflowed: {w20_first}");
        let mut narrow_text = String::new();
        for rl in &narrow_20 {
            let rw = spans_display_width(&rl.line.spans);
            assert_eq!(rw, w20_first);
            assert_eq!(rl.line.width(), rw);
            for s in &rl.line.spans {
                narrow_text.push_str(s.content.as_ref());
            }
            narrow_text.push('\n');
        }
        assert!(
            narrow_text.contains("读代码"),
            "short col damaged at 20 cols:\n{narrow_text}"
        );
        assert!(narrow_text.starts_with('┌') && narrow_text.contains('┐'));
        assert!(narrow_text.contains('└') && narrow_text.contains('┘'));

        // 3. 极端列数与极窄宽度（20 / 14 / 8 / 4 列）均不 panic、不越界
        let many_col_md = "\
| A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- |
| alpha | beta | gamma | delta | epsilon | superlongtokenwithoutspaces |";
        for w in [20usize, 14, 8, 4] {
            let rendered = render_markdown_rich(many_col_md, w, &theme, &cfg);
            assert!(!rendered.is_empty());
            for rl in &rendered {
                let rw = spans_display_width(&rl.line.spans);
                assert!(rw <= w, "many-col narrow {w} overflowed: {rw} > {w}");
                assert!(rl.line.width() <= w, "ratatui width overflowed at {w}");
            }
        }
    }
}
