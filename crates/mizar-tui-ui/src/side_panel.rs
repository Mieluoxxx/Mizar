//! 可折叠、支持鼠标拖拽调宽的右侧分栏面板（借鉴 `dsh-TUI` `SidePanelLayout` + `ContextBarView`）。
//!
//! - 按 `Ctrl+B` 在宽终端（`>= 96` 列）展开/关闭右侧面板，窄屏自动退回整屏聊天；
//! - 1 列分割线与面板标题分隔线对齐绘制 `├` 丁字接头，支持鼠标拖拽调整宽度；
//! - 内置 4 个页签：Context（分段上下文条 + TPS/TTFT）、Files（会话涉及文件）、Tools（工具与 MCP）、Stats（轨迹与告警）。

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

pub use mizar_types::client::{
    ContextBreakdownSnapshot, FileTouchRow, InstructionSummaryRow, McpSummaryRow, SidePanelData,
    ToolSummaryRow,
};

use crate::editor::display_width;
use crate::selection::{SelectionGeometry, SelectionPane};
use crate::theme::Theme;

pub const SIDE_PANEL_MIN_TOTAL_WIDTH: u16 = 96;
pub const CHAT_MIN_COLS: u16 = 52;
pub const PANEL_MIN_COLS: u16 = 30;
pub const PANEL_DEFAULT_COLS: u16 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidePanelTab {
    Context,
    Files,
    Tools,
    Stats,
}

impl SidePanelTab {
    pub fn all() -> &'static [Self] {
        &[Self::Context, Self::Files, Self::Tools, Self::Stats]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Context => "Context",
            Self::Files => "Files",
            Self::Tools => "Tools",
            Self::Stats => "Stats",
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::Context => "◔",
            Self::Files => "▤",
            Self::Tools => "⚙",
            Self::Stats => "◈",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Context => Self::Files,
            Self::Files => Self::Tools,
            Self::Tools => Self::Stats,
            Self::Stats => Self::Context,
        }
    }
}

#[derive(Debug, Clone)]
struct CachedSidePanelLines {
    tab: SidePanelTab,
    inner_w: usize,
    data: SidePanelData,
    theme: Theme,
    lines: Vec<Line<'static>>,
}

#[derive(Debug, Clone)]
pub struct SidePanelState {
    pub open: bool,
    pub width: u16,
    pub tab: SidePanelTab,
    pub scroll: usize,
    pub dragging_divider: bool,
    pub hovered_divider: bool,
    pub divider_col: Option<u16>,
    pub tab_hit_boxes: Vec<(SidePanelTab, Rect)>,
    cached_lines: Option<CachedSidePanelLines>,
}

impl Default for SidePanelState {
    fn default() -> Self {
        Self {
            open: false,
            width: PANEL_DEFAULT_COLS,
            tab: SidePanelTab::Context,
            scroll: 0,
            dragging_divider: false,
            hovered_divider: false,
            divider_col: None,
            tab_hit_boxes: Vec::new(),
            cached_lines: None,
        }
    }
}

impl SidePanelState {
    /// 计算双栏几何；窄屏时自动隐藏分栏。返回 `(chat_w, divider_x, panel_w)`。
    pub fn split_columns(&self, total_width: u16) -> Option<(u16, u16, u16)> {
        if !self.open || total_width < SIDE_PANEL_MIN_TOTAL_WIDTH {
            return None;
        }
        let max_panel = total_width.saturating_sub(CHAT_MIN_COLS + 1);
        let panel_w = self
            .width
            .clamp(PANEL_MIN_COLS, max_panel.max(PANEL_MIN_COLS));
        let chat_w = total_width.saturating_sub(panel_w + 1);
        Some((chat_w, chat_w, panel_w))
    }

    pub fn set_width_from_divider_col(&mut self, total_width: u16, target_divider_col: u16) {
        let max_panel = total_width.saturating_sub(CHAT_MIN_COLS + 1);
        let desired_panel = total_width.saturating_sub(target_divider_col + 1);
        self.width = desired_panel.clamp(PANEL_MIN_COLS, max_panel.max(PANEL_MIN_COLS));
    }
}

/// 最大余数法把 5 类已用 token 分配到整数列宽中，保证各段之和与总占用严格一致（对齐 `ContextBarView`）。
pub fn allocate_context_bar_columns(
    counts: [u64; 5],
    window: u64,
    bar_width: usize,
) -> ([usize; 5], usize) {
    if bar_width == 0 || window == 0 {
        return ([0; 5], bar_width);
    }
    let total_used: u64 = counts.iter().sum();
    if total_used == 0 {
        return ([0; 5], bar_width);
    }
    let ratio = (total_used as f64 / window as f64).min(1.0);
    let filled_total = (ratio * bar_width as f64).round() as usize;
    let filled_total = filled_total.min(bar_width);
    let free = bar_width.saturating_sub(filled_total);

    if filled_total == 0 {
        return ([0; 5], bar_width);
    }

    let mut cols = [0usize; 5];
    let mut remainders = [(0.0f64, 0usize); 5];
    let mut assigned = 0usize;
    for i in 0..5 {
        let exact = (counts[i] as f64 / total_used as f64) * filled_total as f64;
        let floor_v = exact.floor() as usize;
        cols[i] = floor_v;
        assigned += floor_v;
        remainders[i] = (exact - floor_v as f64, i);
    }
    remainders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, idx) in remainders {
        if assigned >= filled_total {
            break;
        }
        cols[idx] += 1;
        assigned += 1;
    }
    (cols, free)
}

pub fn render_divider_column(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &SidePanelState,
    theme: &Theme,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let color = if state.dragging_divider || state.hovered_divider {
        theme.accent
    } else {
        theme.border
    };
    let style = Style::default().fg(color);
    let bottom_junction = area.height.saturating_sub(2);
    let buf = frame.buffer_mut();
    for row_off in 0..area.height {
        let glyph = if row_off == 1 || row_off == bottom_junction {
            "├"
        } else {
            "│"
        };
        buf.set_string(area.x, area.y + row_off, glyph, style);
    }
}

pub fn render_side_panel(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut SidePanelState,
    data: &SidePanelData,
    theme: &Theme,
    sel_geom: &mut SelectionGeometry,
) {
    if area.width < 12 || area.height < 5 {
        return;
    }
    sel_geom.side_panel_rect = Some(area);
    state.tab_hit_boxes.clear();

    // Row 0: 页签栏
    let mut tab_spans = vec![Span::raw(" ")];
    let mut x_cursor = area.x + 1;
    for tab in SidePanelTab::all() {
        let active = *tab == state.tab;
        let text = if active {
            format!("{} {}", tab.symbol(), tab.label())
        } else {
            tab.symbol().to_string()
        };
        let w = display_width(&text) as u16;
        let rect = Rect::new(x_cursor, area.y, w + 2, 1);
        state.tab_hit_boxes.push((*tab, rect));

        let style = if active {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.text_muted)
        };
        tab_spans.push(Span::styled(format!(" {text} "), style));
        x_cursor = x_cursor.saturating_add(w + 2);
    }
    let tab_line = Line::from(tab_spans);
    frame
        .buffer_mut()
        .set_line(area.x, area.y, &tab_line, area.width);

    // Row 1: 水平连接线（与分割线 row 1 的 `├` 对齐）
    let border_style = Style::default().fg(theme.border);
    {
        let buf = frame.buffer_mut();
        for x in area.x..area.x + area.width {
            buf[(x, area.y + 1)].set_symbol("─").set_style(border_style);
        }
    }

    // 主体内容区域 (row 2 .. height - 2)
    let body_y = area.y + 2;
    let body_h = area.height.saturating_sub(4);
    let body_area = Rect::new(area.x + 1, body_y, area.width.saturating_sub(2), body_h);
    let inner_w = body_area.width as usize;

    let cache_hit = state.cached_lines.as_ref().is_some_and(|c| {
        c.tab == state.tab && c.inner_w == inner_w && &c.data == data && &c.theme == theme
    });
    if !cache_hit {
        let lines = match state.tab {
            SidePanelTab::Context => {
                build_context_lines(&data.context, &data.instructions, inner_w, theme)
            }
            SidePanelTab::Files => build_files_lines(
                &data.files,
                &data.git_changes,
                &data.git_diff_stat,
                inner_w,
                theme,
            ),
            SidePanelTab::Tools => build_tools_lines(&data.tools, &data.mcps, inner_w, theme),
            SidePanelTab::Stats => build_stats_lines(&data.context, &data.findings, inner_w, theme),
        };
        state.cached_lines = Some(CachedSidePanelLines {
            tab: state.tab,
            inner_w,
            data: data.clone(),
            theme: theme.clone(),
            lines,
        });
    }

    let lines = &state.cached_lines.as_ref().unwrap().lines;
    let max_scroll = lines.len().saturating_sub(body_h as usize);
    state.scroll = state.scroll.min(max_scroll);
    let visible_end = (state.scroll + body_h as usize).min(lines.len());
    let buf = frame.buffer_mut();
    for (row_off, line) in lines[state.scroll..visible_end].iter().enumerate() {
        let y = body_area.y + row_off as u16;
        buf.set_line(body_area.x, y, line, body_area.width);
        sel_geom.push_row(
            SelectionPane::SidePanel,
            y,
            body_area.x,
            body_area.x + body_area.width,
            false,
        );
    }

    // 底部提示（与分割线 bottom_junction 对齐）
    if area.height >= 4 {
        let rule_y = area.y + area.height - 2;
        let hint_y = area.y + area.height - 1;
        for x in area.x..area.x + area.width {
            buf[(x, rule_y)].set_symbol("─").set_style(border_style);
        }
        buf.set_stringn(
            area.x,
            hint_y,
            " Ctrl+B 关/焦点 · 点页签切换 · 拖分割线调宽",
            area.width as usize,
            Style::default().fg(theme.text_dim),
        );
    }
}

fn build_context_lines(
    ctx: &ContextBreakdownSnapshot,
    instructions: &[InstructionSummaryRow],
    width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let total_used = ctx.system_tokens
        + ctx.user_tokens
        + ctx.assistant_tokens
        + ctx.reasoning_tokens
        + ctx.tool_tokens;
    let window = ctx.context_window.max(total_used).max(1);
    let pct = ((total_used as f64 / window as f64) * 100.0).min(100.0);

    out.push(Line::from(vec![
        Span::styled(
            "上下文占用 ",
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{total_used} / {window} ({pct:.1}%)"),
            Style::default().fg(if pct >= 90.0 {
                theme.error
            } else if pct >= 75.0 {
                theme.warning
            } else {
                theme.accent
            }),
        ),
    ]));

    // 分段进度条
    let bar_w = width.max(8);
    let (segs, free) = allocate_context_bar_columns(
        [
            ctx.system_tokens,
            ctx.user_tokens,
            ctx.assistant_tokens,
            ctx.reasoning_tokens,
            ctx.tool_tokens,
        ],
        window,
        bar_w,
    );
    let mut bar_spans = Vec::new();
    for (i, &w) in segs.iter().enumerate() {
        if w > 0 {
            bar_spans.push(Span::styled(
                "█".repeat(w),
                Style::default().fg(theme.context_segments[i]),
            ));
        }
    }
    if free > 0 {
        bar_spans.push(Span::styled(
            "░".repeat(free),
            Style::default().fg(theme.context_free),
        ));
    }
    out.push(Line::from(bar_spans));
    out.push(Line::default());

    let labels = [
        ("System", ctx.system_tokens),
        ("User", ctx.user_tokens),
        ("Assistant", ctx.assistant_tokens),
        ("Reasoning", ctx.reasoning_tokens),
        ("Tools", ctx.tool_tokens),
    ];
    for (i, (name, tok)) in labels.iter().enumerate() {
        out.push(Line::from(vec![
            Span::styled("■ ", Style::default().fg(theme.context_segments[i])),
            Span::styled(
                format!("{name:<11} "),
                Style::default().fg(theme.text_muted),
            ),
            Span::styled(format!("{tok:>7} tok"), Style::default().fg(theme.text)),
        ]));
    }

    out.push(Line::default());
    out.push(Line::from(Span::styled(
        "实时生成仪表",
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    )));
    let tps_str = ctx
        .tps
        .map(|v| format!("{v:.1} tok/s"))
        .unwrap_or_else(|| "-".to_string());
    let tps_col = match ctx.tps {
        Some(v) if v >= 50.0 => theme.success,
        Some(v) if v >= 20.0 => theme.warning,
        Some(_) => theme.error,
        None => theme.text_muted,
    };
    out.push(Line::from(vec![
        Span::styled("TPS:  ", Style::default().fg(theme.text_muted)),
        Span::styled(
            tps_str,
            Style::default().fg(tps_col).add_modifier(Modifier::BOLD),
        ),
    ]));
    if !ctx.tps_history.is_empty() {
        let spark = unicode_sparkline(&ctx.tps_history, width.saturating_sub(7).max(6));
        out.push(Line::from(vec![
            Span::styled("      ", Style::default().fg(theme.text_dim)),
            Span::styled(spark, Style::default().fg(tps_col)),
        ]));
    }
    out.push(Line::from(vec![
        Span::styled("TTFT: ", Style::default().fg(theme.text_muted)),
        Span::styled(
            ctx.ttft_ms
                .map(|ms| format!("{ms} ms"))
                .unwrap_or_else(|| "-".to_string()),
            Style::default().fg(theme.info),
        ),
    ]));
    if !ctx.ttft_history.is_empty() {
        let ttft_f: Vec<f64> = ctx.ttft_history.iter().map(|&v| v as f64).collect();
        let spark = unicode_sparkline(&ttft_f, width.saturating_sub(7).max(6));
        out.push(Line::from(vec![
            Span::styled("      ", Style::default().fg(theme.text_dim)),
            Span::styled(spark, Style::default().fg(theme.info)),
        ]));
    }
    let cache_rate = ctx.cache_hit_rate();
    let cache_detail = if ctx.cache_write_tokens > 0 {
        format!(
            "{} 读 / {} 写",
            ctx.cache_read_tokens, ctx.cache_write_tokens
        )
    } else {
        format!("{}", ctx.cache_read_tokens)
    };
    out.push(Line::from(vec![
        Span::styled("缓存命中: ", Style::default().fg(theme.text_muted)),
        Span::styled(
            format!("{cache_rate:.1}% ({cache_detail})"),
            Style::default().fg(theme.success),
        ),
    ]));
    if !instructions.is_empty() {
        out.push(Line::default());
        out.push(Line::from(Span::styled(
            format!("指令文件 ({})", instructions.len()),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )));
        for row in instructions {
            let (badge, col) = if row.omitted {
                ("omitted", theme.error)
            } else if row.truncated {
                ("trunc", theme.warning)
            } else {
                (row.scope.as_str(), theme.success)
            };
            out.push(Line::from(vec![
                Span::styled(format!("  [{badge}] "), Style::default().fg(col)),
                Span::styled(row.path.clone(), Style::default().fg(theme.text)),
                Span::styled(
                    format!(" ({}B)", row.bytes),
                    Style::default().fg(theme.text_dim),
                ),
            ]));
        }
    }
    out
}

pub fn unicode_sparkline(values: &[f64], max_bars: usize) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || max_bars == 0 {
        return String::new();
    }
    let slice = if values.len() > max_bars {
        &values[values.len() - max_bars..]
    } else {
        values
    };
    let min = slice.iter().copied().fold(f64::INFINITY, f64::min);
    let max = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-6);
    slice
        .iter()
        .map(|&v| {
            let norm = if (max - min).abs() < 1e-6 {
                0.5
            } else {
                ((v - min) / span).clamp(0.0, 1.0)
            };
            let idx = (norm * 7.0).round() as usize;
            BARS[idx.min(7)]
        })
        .collect()
}

fn build_files_lines(
    files: &[FileTouchRow],
    git_changes: &[FileTouchRow],
    git_diff_stat: &str,
    _width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut out = vec![Line::from(Span::styled(
        format!("会话访问/变更文件 ({})", files.len()),
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    ))];
    if files.is_empty() {
        out.push(Line::from(Span::styled(
            "  暂无文件读取或编辑记录",
            Style::default().fg(theme.text_dim),
        )));
    } else {
        for f in files {
            let col = match f.op.as_str() {
                "edit" | "write" => theme.warning,
                "read" => theme.info,
                _ => theme.text_muted,
            };
            out.push(Line::from(vec![
                Span::styled(format!("  [{:<5}] ", f.op), Style::default().fg(col)),
                Span::styled(f.path.clone(), Style::default().fg(theme.text)),
            ]));
        }
    }

    out.push(Line::default());
    out.push(Line::from(Span::styled(
        format!("工作区 Git 状态 ({})", git_changes.len()),
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    )));
    if !git_diff_stat.trim().is_empty() {
        out.push(Line::from(Span::styled(
            format!("  {}", git_diff_stat.trim()),
            Style::default().fg(theme.text_muted),
        )));
    }
    if git_changes.is_empty() {
        out.push(Line::from(Span::styled(
            "  工作区干净 / 非 Git 仓库",
            Style::default().fg(theme.text_dim),
        )));
    } else {
        for g in git_changes {
            let col = match g.op.as_str() {
                "M" | "AM" | "MM" => theme.warning,
                "A" | "??" => theme.success,
                "D" => theme.error,
                _ => theme.info,
            };
            out.push(Line::from(vec![
                Span::styled(format!("  {:<3} ", g.op), Style::default().fg(col)),
                Span::styled(g.path.clone(), Style::default().fg(theme.text)),
            ]));
        }
    }
    out
}

fn build_tools_lines(
    tools: &[ToolSummaryRow],
    mcps: &[McpSummaryRow],
    _width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut out = vec![Line::from(Span::styled(
        format!("MCP 服务 ({})", mcps.len()),
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    ))];
    if mcps.is_empty() {
        out.push(Line::from(Span::styled(
            "  未配置 MCP Server",
            Style::default().fg(theme.text_dim),
        )));
    } else {
        for m in mcps {
            out.push(Line::from(vec![
                Span::styled(
                    if m.ok { "● " } else { "○ " },
                    Style::default().fg(if m.ok { theme.success } else { theme.error }),
                ),
                Span::styled(
                    format!("{}: {}", m.name, m.status),
                    Style::default().fg(theme.text),
                ),
            ]));
        }
    }

    out.push(Line::default());
    out.push(Line::from(Span::styled(
        format!("已注册工具 ({})", tools.len()),
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    )));
    for t in tools {
        out.push(Line::from(vec![
            Span::styled(
                if t.declared { "● " } else { "○ " },
                Style::default().fg(if t.declared {
                    theme.success
                } else {
                    theme.text_dim
                }),
            ),
            Span::styled(t.name.clone(), Style::default().fg(theme.text)),
            Span::styled(
                format!(" [{}]", t.exposure),
                Style::default().fg(theme.text_dim),
            ),
        ]));
    }
    out
}

fn build_stats_lines(
    ctx: &ContextBreakdownSnapshot,
    findings: &[String],
    _width: usize,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut out = vec![
        Line::from(Span::styled(
            "累计用量与异常",
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
        Line::from(vec![
            Span::styled("Input:  ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}", ctx.cumulative_prompt_tokens()),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            Span::styled("Output: ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}", ctx.output_tokens),
                Style::default().fg(theme.text),
            ),
        ]),
        Line::from(vec![
            Span::styled("Total:  ", Style::default().fg(theme.text_muted)),
            Span::styled(
                format!("{}", ctx.total_tokens),
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::default(),
        Line::from(Span::styled(
            format!("轨迹告警 ({})", findings.len()),
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )),
    ];
    if findings.is_empty() {
        out.push(Line::from(Span::styled(
            "  暂无异常发现 (Ctrl+T 查看完整轨迹)",
            Style::default().fg(theme.text_dim),
        )));
    } else {
        for f in findings {
            out.push(Line::from(Span::styled(
                format!("⚠ {f}"),
                Style::default().fg(theme.warning),
            )));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemePreset;

    #[test]
    fn context_bar_largest_remainder_matches_total() {
        let (cols, free) = allocate_context_bar_columns([100, 250, 400, 150, 100], 2000, 40);
        let sum: usize = cols.iter().sum::<usize>() + free;
        assert_eq!(sum, 40);
        assert_eq!(cols.iter().sum::<usize>(), 20); // 1000/2000 = 50%

        let spark = unicode_sparkline(&[10.0, 30.0, 60.0], 8);
        assert_eq!(spark.chars().count(), 3);
    }

    fn line_text(line: &Line<'static>) -> String {
        line.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
    }

    #[test]
    fn cache_hit_rate_uses_prompt_total_as_denominator() {
        // input 只是未命中的净额；分母必须把 cache_read / cache_write 算进去，否则会超过 100%
        let ctx = ContextBreakdownSnapshot {
            input_tokens: 1_000,
            cache_read_tokens: 9_000,
            cache_write_tokens: 2_000,
            output_tokens: 500,
            total_tokens: 12_500,
            ..Default::default()
        };
        assert_eq!(ctx.cumulative_prompt_tokens(), 12_000);
        assert!((ctx.cache_hit_rate() - 75.0).abs() < 1e-9);

        // 零输入也不能除零
        assert_eq!(ContextBreakdownSnapshot::default().cache_hit_rate(), 0.0);
    }

    #[test]
    fn context_panel_renders_cache_rate_and_write_volume() {
        let theme = Theme::from_preset(ThemePreset::Auto, None);
        let ctx = ContextBreakdownSnapshot {
            system_tokens: 1_000,
            input_tokens: 1_000,
            cache_read_tokens: 9_000,
            cache_write_tokens: 2_000,
            output_tokens: 500,
            total_tokens: 12_500,
            context_window: 128_000,
            ..Default::default()
        };
        let lines: Vec<String> = build_context_lines(&ctx, &[], 60, &theme)
            .iter()
            .map(line_text)
            .collect();
        let joined = lines.join("\n");
        assert!(joined.contains("75.0%"), "{joined}");
        assert!(joined.contains("9000 读 / 2000 写"), "{joined}");
    }

    #[test]
    fn stats_panel_reports_prompt_total_and_grand_total() {
        let theme = Theme::from_preset(ThemePreset::Auto, None);
        let ctx = ContextBreakdownSnapshot {
            input_tokens: 1_000,
            cache_read_tokens: 9_000,
            cache_write_tokens: 2_000,
            output_tokens: 500,
            total_tokens: 12_500,
            ..Default::default()
        };
        let lines: Vec<String> = build_stats_lines(&ctx, &[], 60, &theme)
            .iter()
            .map(line_text)
            .collect();
        let joined = lines.join("\n");
        // Input 行展示的是 prompt 全量（含缓存），不是净输入 1000
        assert!(joined.contains("Input:  12000"), "{joined}");
        assert!(joined.contains("Output: 500"), "{joined}");
        assert!(joined.contains("Total:  12500"), "{joined}");
        assert!(!joined.contains("Input:  1000"), "{joined}");
    }
}
