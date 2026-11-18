//! 增强型流式 Markdown、CJK 强强调修正、`pi-emphasis` 行内渲染与词级 Diff 引擎。

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use similar::{ChangeTag, TextDiff};

use crate::editor::{char_width, display_width};
use crate::emphasis::{
    EmphasisConfig, decorate_action_lead_in, decorate_label_with_icon, render_pill_spans,
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
                    "└─",
                    dim_style,
                ))));
            }
            continue;
        }

        // 2. GFM 表格（表头含 `|` 且下一行匹配分割线）
        if idx + 1 < lines.len() && is_table_pair(lines[idx], lines[idx + 1]) {
            tree_rails.clear();
            let start = idx;
            idx += 2;
            while idx < lines.len() && lines[idx].trim().contains('|') && !lines[idx].trim().is_empty()
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
            let decorated = if level == 2 {
                decorate_label_with_icon(heading, emphasis)
            } else {
                heading.to_string()
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
                    let cont = Span::styled(tree_branch_cont_guide(&tree_rails, &tree_line.prefix), dim_style);
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
                            Style::default().fg(theme.accent),
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
                rest,
                rail_span,
                &pad,
                target_w,
                width,
                base_style,
                theme,
                emphasis,
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
            first_prefixes.push(Span::styled(first_str, Style::default().fg(theme.accent)));
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

        let spans = parse_inline_rich(raw_line, base_style, theme, emphasis, true);
        out.extend(wrap_styled_spans_rich(spans, None, None, width));
        idx += 1;
    }

    out
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
    for (tok, is_open) in [
        ("├──", true),
        ("└──", false),
        ("├─", true),
        ("└─", false),
    ] {
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
            if !rest_preview.starts_with('│')
                && match_tree_branch_token(rest_preview).is_none()
            {
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

fn active_tree_blank_guide(lines: &[&str], scan_from: usize, tree_rails: &[bool]) -> Option<String> {
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
            (Some(t), true) if matches!(t.kind, TreeKind::RailLine) && t.prefix == first_rail_prefix => {
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
        Style::default().fg(theme.accent),
    ));
    cont_prefixes.push(Span::raw(format!("{pad}  ")));

    let spans = if let Some(target_w) = align_key_width
        && let Some((key_raw, tail_raw)) = rest.split_once(" — ")
    {
        let mut combined = parse_inline_rich(key_raw, base_style, theme, emphasis, true);
        let kw = spans_display_width(&combined);
        let gap = target_w.saturating_sub(kw) + 1;
        combined.push(Span::styled(format!("{}— ", " ".repeat(gap)), base_style));
        combined.extend(parse_inline_rich(tail_raw, base_style, theme, emphasis, false));
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
        .map(|i| delim.get(i).map(|s| parse_col_align(s)).unwrap_or(ColAlign::Left))
        .collect();

    // 先解析每一格的富文本 Span，再基于真实渲染宽度计算列宽与对齐填充，
    // 防止剥除 `` ` `` / `**` 或展开 `[[Pill]]` 后破坏右侧 `│` 边框对齐。
    let mut parsed_rows: Vec<Vec<Vec<Span<'static>>>> = Vec::with_capacity(rows.len() - 1);
    let header_style = Style::default().fg(theme.accent).add_modifier(Modifier::BOLD);
    parsed_rows.push(
        (0..cols)
            .map(|i| {
                parse_inline_rich(
                    header.get(i).copied().unwrap_or(""),
                    header_style,
                    theme,
                    emphasis,
                    false,
                )
            })
            .collect(),
    );
    let body_style = Style::default().fg(theme.text);
    for raw in &rows[2..] {
        let cells = split_table_row(raw);
        parsed_rows.push(
            (0..cols)
                .map(|i| {
                    parse_inline_rich(
                        cells.get(i).copied().unwrap_or(""),
                        body_style,
                        theme,
                        emphasis,
                        false,
                    )
                })
                .collect(),
        );
    }

    // 边框消耗：cols + 1 个框线 + 每列左右各 1 空格
    let chrome = cols + 1 + cols * 2;
    let avail = max_width.saturating_sub(chrome).max(cols * 2);
    let mut col_widths = vec![3usize; cols];
    for r in &parsed_rows {
        for (c, cell_spans) in r.iter().enumerate() {
            col_widths[c] = col_widths[c].max(spans_display_width(cell_spans));
        }
    }
    let sum: usize = col_widths.iter().sum();
    if sum > avail {
        let scale = avail as f64 / sum as f64;
        for w in &mut col_widths {
            *w = ((*w as f64 * scale).floor() as usize).max(2);
        }
        let mut cur_sum: usize = col_widths.iter().sum();
        while cur_sum > avail {
            if let Some(max_col) = col_widths.iter_mut().max()
                && *max_col > 2
            {
                *max_col -= 1;
                cur_sum -= 1;
            } else {
                break;
            }
        }
    }

    let border_style = Style::default().fg(theme.border);
    let make_border = |left: &str, mid: &str, right: &str| -> RenderedLine {
        let mut s = String::from(left);
        for (i, w) in col_widths.iter().enumerate() {
            if i > 0 {
                s.push_str(mid);
            }
            s.push_str(&"─".repeat(*w + 2));
        }
        s.push_str(right);
        RenderedLine::hard(Line::from(Span::styled(s, border_style)))
    };

    let mut out = Vec::new();
    out.push(make_border("┌", "┬", "┐"));

    for (r_idx, row_spans) in parsed_rows.iter().enumerate() {
        let is_header = r_idx == 0;
        let mut spans = vec![Span::styled("│", border_style)];
        for (c_idx, cell_spans) in row_spans.iter().enumerate() {
            let w = col_widths[c_idx];
            let (clipped_spans, cw) = clip_spans_to_width(cell_spans, w, true);
            let pad = w.saturating_sub(cw);
            let (left_pad, right_pad) = match aligns[c_idx] {
                ColAlign::Left => (0, pad),
                ColAlign::Right => (pad, 0),
                ColAlign::Center => (pad / 2, pad - pad / 2),
            };
            spans.push(Span::raw(format!(" {}", " ".repeat(left_pad))));
            spans.extend(clipped_spans);
            spans.push(Span::raw(format!("{} ", " ".repeat(right_pad))));
            spans.push(Span::styled("│", border_style));
        }
        out.push(RenderedLine::hard(Line::from(spans)));
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
                spans.push(Span::styled(
                    token.to_string(),
                    Style::default().fg(color),
                ));
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
                | '“'
                | '”'
                | '‘'
                | '’'
                | '、'
                | '…'
                | '—'
        )
}

/// 匹配 `**...**` 或 `__...__`，包含针对 `**证据与影响：**源码中` 的 CJK 标点闭合修正。
fn try_parse_strong(line: &str, byte_idx: usize, delim: &str) -> Option<(String, usize)> {
    let rest = line.get(byte_idx..)?.strip_prefix(delim)?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        return None;
    }
    let mut search_from = 0usize;
    while let Some(rel_end) = rest[search_from..].find(delim) {
        let abs_end = search_from + rel_end;
        let inner = &rest[..abs_end];
        if inner.is_empty() || inner.contains('`') || inner.contains('\n') {
            return None;
        }
        let prev_ch = inner.chars().next_back()?;
        let after = &rest[abs_end + delim.len()..];
        let next_ch = after.chars().next();

        let valid_standard = !prev_ch.is_whitespace()
            && (!is_unicode_punct(prev_ch)
                || next_ch.is_none_or(|c| c.is_whitespace() || is_unicode_punct(c)));
        let valid_cjk_adjacent = is_unicode_punct(prev_ch) && next_ch.is_some_and(is_cjk_char);

        if valid_standard || valid_cjk_adjacent {
            let next_byte = byte_idx + delim.len() + abs_end + delim.len();
            return Some((inner.to_string(), next_byte));
        }
        search_from = abs_end + delim.len();
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
    if inner.is_empty() || inner.contains('[') || inner.contains(']') || inner.chars().count() > 32 {
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
    Some((label, url, byte_idx + 1 + close_bracket + 2 + close_paren + 1))
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

        // 4. 粗体（含 CJK 邻接修正、内部嵌套 `[[Pill]]` 解包、行首标签图标与行动引导词图标）
        if let Some((inner, next_byte)) = try_parse_strong(line, byte_idx, "**")
            .or_else(|| try_parse_strong(line, byte_idx, "__"))
        {
            flush_buf(&mut buf, &mut spans);
            let bold_color = if emphasis.enabled {
                theme.link
            } else {
                base.fg.unwrap_or(theme.text)
            };
            let strong_style = base.fg(bold_color).add_modifier(Modifier::BOLD);
            spans.extend(render_strong_inner_spans(
                &inner,
                strong_style,
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
    out.push(Span::styled(decorated, strong_style));
}

/// 渲染粗体内部文本：支持模型习惯性把胶囊写成 `**[[Passed]]**`、树线前缀 `**├── ...**` 或混排 `**[[Done]] 说明**` 的场景。
fn render_strong_inner_spans(
    inner: &str,
    strong_style: Style,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    is_line_lead: bool,
) -> Vec<Span<'static>> {
    if !inner.contains("[[") {
        let mut out = Vec::new();
        push_strong_segment(&mut out, inner, strong_style, theme, emphasis, is_line_lead);
        if out.is_empty() {
            out.push(Span::styled(String::new(), strong_style));
        }
        return out;
    }

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut lead = is_line_lead;
    while pos < inner.len() {
        if let Some((pill_label, next_byte)) = try_parse_pill(inner, pos) {
            out.extend(render_pill_spans(pill_label, theme, emphasis));
            pos = next_byte;
            lead = false;
            continue;
        }
        let next_bracket = inner[pos..]
            .find("[[")
            .map(|rel| pos + rel)
            .filter(|&p| p > pos)
            .unwrap_or(inner.len());
        let seg = &inner[pos..next_bracket];
        push_strong_segment(&mut out, seg, strong_style, theme, emphasis, lead);
        pos = next_bracket;
        lead = false;
    }
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

fn wrap_styled_spans_with_prefixes(
    spans: Vec<Span<'static>>,
    prefix_first: &[Span<'static>],
    prefix_cont: &[Span<'static>],
    max_width: usize,
) -> Vec<RenderedLine> {
    let width = max_width.max(2);
    let mut lines: Vec<RenderedLine> = Vec::new();
    let mut cur_spans: Vec<Span<'static>> = Vec::new();
    let mut cur_col = 0usize;
    let mut is_first_visual_line = true;

    let push_prefixes = |target: &mut Vec<Span<'static>>,
                         col: &mut usize,
                         prefs: &[Span<'static>]| {
        for p in prefs {
            let pw: usize = p.content.chars().map(char_width).sum();
            *col += pw;
            target.push(p.clone());
        }
    };

    push_prefixes(&mut cur_spans, &mut cur_col, prefix_first);

    for span in spans {
        let style = span.style;
        let mut chunk = String::new();
        for ch in span.content.chars() {
            let cw = char_width(ch);
            if cw > 0 && cur_col + cw > width && cur_col > 0 {
                if !chunk.is_empty() {
                    cur_spans.push(Span::styled(std::mem::take(&mut chunk), style));
                }
                lines.push(RenderedLine {
                    line: Line::from(std::mem::take(&mut cur_spans)),
                    soft_wrapped: !is_first_visual_line,
                });
                is_first_visual_line = false;
                cur_col = 0;
                push_prefixes(&mut cur_spans, &mut cur_col, prefix_cont);
            }
            chunk.push(ch);
            cur_col += cw;
        }
        if !chunk.is_empty() {
            cur_spans.push(Span::styled(chunk, style));
        }
    }

    lines.push(RenderedLine {
        line: Line::from(cur_spans),
        soft_wrapped: !is_first_visual_line,
    });
    lines
}

/// 使用 `similar` 进行行内词级 Diff 渲染（隔离共同前导缩进，宽屏可切分双栏）。
pub fn render_diff_rich(
    diff_text: &str,
    max_width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
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

fn render_unified_word_diff(
    lines: &[&str],
    max_width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
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

    let pad_spans_to = |spans: Vec<Span<'static>>, bg: Color, target_w: usize| -> Vec<Span<'static>> {
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
        let right_ctx = vec![Span::styled(ctx_text, Style::default().fg(theme.text_muted))];
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
    fn wraps_markdown_and_code_blocks_within_width() {
        let md = "# 标题\n- 第一条包含 `inline` 和 **粗体**\n```rust\nlet x = 1234567890;\n```";
        let lines = render_markdown_lines(md, 14, Style::default(), Color::Cyan);
        assert!(lines.len() >= 5);
        for line in &lines {
            let w: usize = line.spans.iter().map(|s| display_width(&s.content)).sum();
            assert!(w <= 14, "line exceeded width 14: {line:?} (w={w})");
        }
    }

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
    fn split_diff_and_word_diff_render_cleanly() {
        let theme = Theme::mist_dark();
        let diff = "@@ -1,2 +1,2 @@\n-    let old_val = 1;\n+    let new_val = 2;";
        let narrow = render_diff_rich(diff, 60, &theme);
        assert!(narrow.len() >= 3);
        let wide = render_diff_rich(diff, 100, &theme);
        let wide_str: String = wide[1]
            .line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(wide_str.contains('│'), "expected split diff: {wide_str}");
    }

    #[test]
    fn table_borders_stay_aligned_with_code_bold_and_nested_pills() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig {
            enabled: true,
            round_caps: false,
            icons: false,
            guidance: true,
        };
        let md = "\
| 检查 | 结果 |
| --- | --- |
| `cargo check --all-targets` | **[[Passed]]** 16.6s, 零错误 |
| `cargo test --workspace` | **[[Passed]]** 全部绿，含 SSE 集成 |
| `cargo clippy` | **[[Passed]]** 全仓仅 `1` 条警告 |
| `unsafe` | **[[Keep]]** 仅 2 处真实块 |
| panic/expect | 8 处，对一个含 `TUI` 的项目来说很低 |";

        let rendered = render_markdown_rich(md, 96, &theme, &cfg);
        assert!(rendered.len() >= 8);
        let first_w = spans_display_width(&rendered[0].line.spans);
        for (row_idx, rl) in rendered.iter().enumerate() {
            let row_str: String = rl
                .line
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect();
            let rw = spans_display_width(&rl.line.spans);
            assert_eq!(
                rw, first_w,
                "table row {row_idx} width mismatch ({rw} != {first_w}): {row_str:?}"
            );
            assert!(
                !row_str.contains("[[Passed]]") && !row_str.contains("[[Keep]]"),
                "nested bold pill leaked raw brackets: {row_str:?}"
            );
        }
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
        assert!(!core_line.spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert!(core_line.spans[1].style.add_modifier.contains(Modifier::BOLD));

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
        assert!(ff_row.starts_with("│  │  • find_files"), "ff row: {ff_row:?}");

        let cg_row = texts.iter().find(|l| l.contains("codegraph")).unwrap();
        assert!(cg_row.starts_with("│     • codegraph"), "cg row: {cg_row:?}");

        let note_row = texts.iter().find(|l| l.contains("外部工具")).unwrap();
        assert!(note_row.starts_with("   • 外部工具"), "note row: {note_row:?}");

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
    fn tree_bullet_soft_wrap_preserves_outer_rail_within_width() {
        let theme = Theme::mist_dark();
        let cfg = EmphasisConfig::default();
        let md = "**├── 分支**\n- **codemode** — QuickJS 沙箱执行多个工具调用中间结果不进上下文";
        let rendered = render_markdown_rich(md, 30, &theme, &cfg);
        assert!(rendered.len() >= 3);
        assert!(rendered[2].soft_wrapped);
        let cont_str: String = rendered[2]
            .line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(
            cont_str.starts_with("│    "),
            "soft wrap lost outer rail: {cont_str:?}"
        );
        for rl in &rendered {
            assert!(spans_display_width(&rl.line.spans) <= 30);
        }
    }
}
