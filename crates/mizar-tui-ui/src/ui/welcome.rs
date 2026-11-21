//! Launchpad 欢迎页与 MIZAR 双色调 ASCII Shadow 艺术字构建。

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::{color_to_rgb8, effort_color, format_effort_label};
use crate::app::{AppState, LaunchpadQuickAction, short_id, truncate_display};
use crate::editor::display_width;
use crate::markdown::{RenderedLine, wrap_styled_spans_rich};
use crate::theme::{Theme, mix_oklab};

/// Launchpad 快捷操作芯片在欢迎页行内的水平跨度元数据。
pub(super) struct WelcomeChipSpan {
    pub action: LaunchpadQuickAction,
    pub x_offset: u16,
    pub width: u16,
}

/// 欢迎页序言构行结果：包含行序列以及快捷操作行相对该块的索引与各按钮水平跨度。
pub(super) struct WelcomeSection {
    pub lines: Vec<RenderedLine>,
    pub chips_line_idx: Option<usize>,
    pub chip_spans: Vec<WelcomeChipSpan>,
}

/// 统一构建 Launchpad 欢迎页序言行与快捷芯片相对布局，供全屏对话流与 Inline 原生 scrollback 共用。
pub(super) fn build_welcome_section(app: &AppState, width: usize, height: usize) -> WelcomeSection {
    let theme = &app.theme;
    let model_label = app.active_model_label();
    let effort_str = format_effort_label(app.reasoning_effort);
    let effort_col = effort_color(theme, app.reasoning_effort);

    // 极窄或极矮终端紧凑模式（如 20x8）
    if height < 7 || width < 36 {
        let mut lines = wrap_styled_spans_rich(
            vec![
                Span::styled(
                    "Mizar",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    truncate_display(&model_label, width.saturating_sub(2)),
                    Style::default().fg(theme.text),
                ),
            ],
            None,
            None,
            width.max(1),
        );
        lines.push(RenderedLine::hard(Line::from(Span::styled(
            truncate_display(&model_label, width),
            Style::default().fg(theme.success),
        ))));
        lines.push(RenderedLine::hard(Line::from(Span::styled(
            truncate_display(&app.workspace.display().to_string(), width),
            Style::default().fg(theme.text_muted),
        ))));
        return WelcomeSection {
            lines,
            chips_line_idx: None,
            chip_spans: Vec::new(),
        };
    }

    let ws_str = truncate_display(
        &app.workspace.display().to_string(),
        width.saturating_sub(14),
    );
    let rule_w = width.saturating_sub(4).min(64);
    let sep = "─".repeat(rule_w);

    let mut lines: Vec<RenderedLine> = Vec::new();
    let show_banner = width >= 46 && height >= 16;
    let top_pad = if show_banner {
        if height >= 20 {
            2
        } else if height >= 18 {
            1
        } else {
            0
        }
    } else if height >= 14 {
        2
    } else {
        0
    };
    for _ in 0..top_pad {
        lines.push(RenderedLine::hard(Line::raw("")));
    }

    if show_banner {
        let mut banner = Vec::with_capacity(WELCOME_ASCII_LOGO.len());
        append_welcome_banner(&mut banner, theme);
        lines.extend(banner.into_iter().map(RenderedLine::hard));
    }

    lines.push(RenderedLine::hard(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "◆ Mizar",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  ", Style::default().fg(theme.text_dim)),
        Span::styled(
            format!("theme: {}", theme.id),
            Style::default().fg(theme.text_muted),
        ),
    ])));
    lines.push(RenderedLine::hard(Line::from(Span::styled(
        format!("  {sep}"),
        Style::default().fg(theme.border),
    ))));
    lines.push(RenderedLine::hard(Line::from(vec![
        Span::styled("  工作区  ", Style::default().fg(theme.text_dim)),
        Span::styled(ws_str, Style::default().fg(theme.text)),
    ])));
    push_welcome_wrapped(
        &mut lines,
        Span::styled("  模型    ", Style::default().fg(theme.text_dim)),
        vec![
            Span::styled(
                model_label,
                Style::default()
                    .fg(theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("[effort: {effort_str}]"),
                Style::default().fg(effort_col),
            ),
        ],
        width,
    );
    let mut session_spans = vec![Span::styled(
        short_id(&app.session_id).to_string(),
        Style::default().fg(theme.text_muted),
    )];
    if let Some(title) = app
        .session_title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        session_spans.push(Span::styled(" · ", Style::default().fg(theme.border)));
        session_spans.push(Span::styled(
            truncate_display(title, width.saturating_sub(36).max(16)),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        ));
    }
    session_spans.push(Span::raw("   "));
    session_spans.push(Span::styled(
        format!("approval: {}", app.approval_mode.as_str()),
        Style::default().fg(theme.text_dim),
    ));
    push_welcome_wrapped(
        &mut lines,
        Span::styled("  会话    ", Style::default().fg(theme.text_dim)),
        session_spans,
        width,
    );
    lines.push(RenderedLine::hard(Line::raw("")));
    let mut row1_spans = vec![
        Span::styled(
            "Enter",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 发送/排队  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled(
            "Tab",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 思考  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled(
            "Ctrl+L",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 模型", Style::default().fg(theme.text_muted)),
    ];
    if app.features.side_panel {
        row1_spans.push(Span::styled("  ·  ", Style::default().fg(theme.text_muted)));
        row1_spans.push(Span::styled(
            "Ctrl+B",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ));
        row1_spans.push(Span::styled(" 侧栏", Style::default().fg(theme.text_muted)));
    }
    push_welcome_wrapped(
        &mut lines,
        Span::styled("  操作    ", Style::default().fg(theme.text_dim)),
        row1_spans,
        width,
    );
    let mut row2_spans = vec![
        Span::styled(
            "Esc",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 暂停  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled(
            "/",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 命令  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled(
            "@",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" 文件  ·  ", Style::default().fg(theme.text_muted)),
    ];
    if app.features.trajectory_view {
        row2_spans.push(Span::styled(
            "Ctrl+T",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ));
        row2_spans.push(Span::styled(
            " 轨迹  ·  ",
            Style::default().fg(theme.text_muted),
        ));
    }
    row2_spans.push(Span::styled(
        "?",
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    ));
    row2_spans.push(Span::styled(" 帮助", Style::default().fg(theme.text_muted)));
    push_welcome_wrapped(&mut lines, Span::raw("          "), row2_spans, width);

    let mut chips_line_idx = None;
    let mut chip_spans_meta = Vec::new();
    if height >= 12 && width >= 56 && lines.len() + 1 < height {
        let mut actions = vec![
            (" [模型] ", LaunchpadQuickAction::ModelPicker),
            (" [主题] ", LaunchpadQuickAction::ThemePicker),
            (" [技能] ", LaunchpadQuickAction::SkillPicker),
        ];
        if app.features.trajectory_view {
            actions.push((" [轨迹] ", LaunchpadQuickAction::TrajectoryScene));
        }
        if app.features.side_panel {
            actions.push((" [侧栏] ", LaunchpadQuickAction::ToggleSidePanel));
        }
        lines.push(RenderedLine::hard(Line::raw("")));
        chips_line_idx = Some(lines.len());
        let mut chip_spans = vec![Span::styled(
            "  快捷    ",
            Style::default().fg(theme.text_dim),
        )];
        let mut x_off: u16 = 10;
        for (label, qa) in actions {
            let w = display_width(label) as u16;
            if (x_off as usize) + (w as usize) < width {
                chip_spans_meta.push(WelcomeChipSpan {
                    action: qa,
                    x_offset: x_off,
                    width: w,
                });
                chip_spans.push(Span::styled(
                    label,
                    Style::default()
                        .fg(theme.accent)
                        .bg(theme.card_bg)
                        .add_modifier(Modifier::BOLD),
                ));
                chip_spans.push(Span::raw(" "));
                x_off += w + 1;
            }
        }
        lines.push(RenderedLine::hard(Line::from(chip_spans)));
    }

    WelcomeSection {
        lines,
        chips_line_idx,
        chip_spans: chip_spans_meta,
    }
}

/// 将欢迎页序言构成为可写入 Inline 原生 scrollback 的行序列（末尾附带 1 行空行分隔后续消息）。
pub fn build_welcome_lines_for_scrollback(app: &AppState, width: usize) -> Vec<Line<'static>> {
    let section = build_welcome_section(app, width.max(12), usize::MAX);
    let mut lines: Vec<Line<'static>> = section.lines.into_iter().map(|rl| rl.line).collect();
    lines.push(Line::raw(""));
    lines
}

/// 将欢迎页带标签前导的元数据/快捷键行按视口宽度折行并保持 10 列悬挂缩进，确保 `lines.len()` 与真实行数一致。
fn push_welcome_wrapped(
    lines: &mut Vec<RenderedLine>,
    prefix_first: Span<'static>,
    spans: Vec<Span<'static>>,
    width: usize,
) {
    lines.extend(wrap_styled_spans_rich(
        spans,
        Some(prefix_first),
        Some(Span::raw("          ")),
        width,
    ));
}

/// 欢迎页使用的 MIZAR ANSI Shadow 立体艺术字（固定 6 行、最大显示宽度 42 列）。
const WELCOME_ASCII_LOGO: [&str; 6] = [
    "███╗   ███╗ ██╗ ███████╗  █████╗  ██████╗",
    "████╗ ████║ ██║ ╚══███╔╝ ██╔══██╗ ██╔══██╗",
    "██╔████╔██║ ██║   ███╔╝  ███████║ ██████╔╝",
    "██║╚██╔╝██║ ██║  ███╔╝   ██╔══██║ ██╔══██╗",
    "██║ ╚═╝ ██║ ██║ ███████╗ ██║  ██║ ██║  ██║",
    "╚═╝     ╚═╝ ╚═╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum BannerGlyphKind {
    Face,
    Shadow,
    Space,
}

fn classify_banner_char(ch: char) -> BannerGlyphKind {
    match ch {
        '█' => BannerGlyphKind::Face,
        ' ' => BannerGlyphKind::Space,
        _ => BannerGlyphKind::Shadow,
    }
}

fn welcome_banner_face_color(theme: &Theme, row_idx: usize, total_rows: usize) -> Color {
    if !theme.truecolor {
        return theme.accent;
    }
    match (color_to_rgb8(theme.link), color_to_rgb8(theme.accent)) {
        (Some(top), Some(bottom)) => {
            let denom = total_rows.saturating_sub(1).max(1) as f64;
            let t = row_idx as f64 / denom;
            mix_oklab(top, bottom, t).to_color()
        }
        _ => theme.accent,
    }
}

fn push_banner_run(
    spans: &mut Vec<Span<'static>>,
    kind: BannerGlyphKind,
    text: String,
    face_style: Style,
    shadow_style: Style,
) {
    if text.is_empty() {
        return;
    }
    let span = match kind {
        BannerGlyphKind::Face => Span::styled(text, face_style),
        BannerGlyphKind::Shadow => Span::styled(text, shadow_style),
        BannerGlyphKind::Space => Span::raw(text),
    };
    spans.push(span);
}

/// 将双色调 ANSI Shadow 艺术字逐行写入欢迎页（实心主面在 OKLab 空间从 `link` 渐变至 `accent`，立体阴影复用 `accent_soft`）。
fn append_welcome_banner<'a>(lines: &mut Vec<Line<'a>>, theme: &Theme) {
    let face_base = Style::default().add_modifier(Modifier::BOLD);
    let shadow_style = Style::default().fg(theme.accent_soft);
    let total_rows = WELCOME_ASCII_LOGO.len();

    for (row_idx, raw_row) in WELCOME_ASCII_LOGO.iter().enumerate() {
        let face_style = face_base.fg(welcome_banner_face_color(theme, row_idx, total_rows));
        let mut spans = vec![Span::raw("  ")];
        let mut cur_kind: Option<BannerGlyphKind> = None;
        let mut buf = String::new();

        for ch in raw_row.chars() {
            let kind = classify_banner_char(ch);
            if cur_kind != Some(kind) {
                if let Some(prev_kind) = cur_kind {
                    push_banner_run(
                        &mut spans,
                        prev_kind,
                        std::mem::take(&mut buf),
                        face_style,
                        shadow_style,
                    );
                }
                cur_kind = Some(kind);
            }
            buf.push(ch);
        }
        if let Some(last_kind) = cur_kind {
            push_banner_run(&mut spans, last_kind, buf, face_style, shadow_style);
        }

        lines.push(Line::from(spans));
    }
}
