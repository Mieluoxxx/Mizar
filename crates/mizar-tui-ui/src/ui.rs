//! TUI 布局与渲染引擎：双栏工作台、时间轴导轨、虚拟化流式视图、`pi-emphasis` 语法、词级 Diff 与浮层叠加。

use std::collections::HashMap;

use mizar_llm_types::ReasoningEffort;
use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders};

use crate::app::{
    AppState, ApprovalMode, BashInputMode, ChatItem, ItemRenderKey, LaunchpadQuickAction,
    NoticeLevel, ToolExtraMeta, ToolState, abbreviate_home_path, clean_bash_output_lines,
    count_diff_stats, format_diff_stat, format_expanded_tool_args, format_tokens, is_path_arg_tool,
    is_read_only_compact_tool, short_id, truncate_display,
};
use crate::editor::{char_width, display_width};
use crate::emphasis::EmphasisConfig;
use crate::markdown::{
    RenderedLine, parse_inline_rich, render_diff_rich, render_markdown_rich, wrap_styled_spans_rich,
};
use crate::overlay::render_overlay;
use crate::reveal::RevealSlot;
use crate::selection::{SelectionPane, paint_selection_overlay};
use crate::side_panel::{render_divider_column, render_side_panel};
use crate::terminal::ScreenMode;
use crate::theme::{Rgb8, Theme, mix_oklab};
use crate::timeline_rail::{
    RAIL_MIN_TERMINAL_WIDTH, RAIL_MIN_TURNS, RAIL_WIDTH, TimelineSnapshot, TimelineTurn,
    build_timeline_snapshot, compute_rail_geometry, render_timeline_rail,
};
use crate::trajectory_scene::render_trajectory_scene;

const BRAILLE_SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const IGNITION_DURATION_MS: u128 = 1200;

/// 渲染主入口：每帧同步重建选区几何、点击命中矩形、导轨快照与虚实视图。
pub fn render(frame: &mut Frame<'_>, app: &mut AppState) {
    let area = frame.area();
    if area.width < 4 || area.height < 4 {
        return;
    }

    app.selection_geom.clear();
    app.launchpad_hit_rects.clear();
    app.rail_geo = None;
    app.side_panel.divider_col = None;
    app.timeline_snap = TimelineSnapshot::default();

    // 若主题启用了不透明面板底色，整帧先铺底
    if app.theme.panel_bg != Color::Reset {
        frame
            .buffer_mut()
            .set_style(area, Style::default().bg(app.theme.panel_bg));
    }

    // 1. 全屏轨迹分析场景接管整个视口
    if app.trajectory_scene.open {
        render_trajectory_scene(frame, area, &mut app.trajectory_scene, &app.theme);
        app.last_buffer = Some(frame.buffer_mut().clone());
        return;
    }

    // 2. 双栏 / 单栏工作台分配
    let chat_area =
        if let Some((chat_w, divider_x, panel_w)) = app.side_panel.split_columns(area.width) {
            app.side_panel.divider_col = Some(divider_x);
            let chat_rect = Rect::new(area.x, area.y, chat_w, area.height);
            let divider_rect = Rect::new(area.x + divider_x, area.y, 1, area.height);
            let panel_rect = Rect::new(area.x + divider_x + 1, area.y, panel_w, area.height);

            render_divider_column(frame, divider_rect, &app.side_panel, &app.theme);
            render_side_panel(
                frame,
                panel_rect,
                &mut app.side_panel,
                &app.side_panel_data,
                &app.theme,
                &mut app.selection_geom,
            );
            chat_rect
        } else {
            area
        };

    // 3. 对话区垂直切分：transcript + queued strip + input + status bar
    let input_inner_w = chat_area.width.saturating_sub(2).max(1);
    let input_layout = app.input.layout(input_inner_w);

    let status_h = 1u16;
    let avail_h = chat_area.height.saturating_sub(status_h);
    // 队列条：默认 1 行摘要；选择器打开时逐条展开（最多 3 行），不再因终端矮而直接隐藏。
    let queued_h = if app.queued_prompts.is_empty() {
        0
    } else if app.queue_selected.is_some() {
        (app.queued_prompts.len().min(3) as u16) + 1
    } else {
        1
    }
    .min(avail_h.saturating_sub(3));
    let max_input_h = avail_h.saturating_sub(queued_h + 1).max(3).min(avail_h);
    let input_h = ((input_layout.lines.len() as u16) + 2)
        .clamp(3, 8)
        .min(max_input_h);
    let transcript_h = avail_h.saturating_sub(queued_h + input_h);

    let transcript_area = Rect::new(chat_area.x, chat_area.y, chat_area.width, transcript_h);
    let queued_area = Rect::new(
        chat_area.x,
        chat_area.y + transcript_h,
        chat_area.width,
        queued_h,
    );
    let input_area = Rect::new(
        chat_area.x,
        chat_area.y + transcript_h + queued_h,
        chat_area.width,
        input_h,
    );
    let status_area = Rect::new(
        chat_area.x,
        chat_area.y + transcript_h + queued_h + input_h,
        chat_area.width,
        status_h,
    );

    app.selection_geom.input_rect = Some(input_area);

    // 4. 渲染对话历史（含顶部欢迎页序言）
    if transcript_area.height > 0 && transcript_area.width > 0 {
        render_transcript(frame, transcript_area, app);
    }

    // 5. 渲染排队待发送条
    if queued_area.height > 0 {
        render_queued_strip(frame, queued_area, app);
    }

    // 6. 渲染输入区（含 Effort 点火扫描光效与折叠粘贴块高亮）
    render_input_box(frame, input_area, &input_layout, app);

    // 7. 渲染底部状态栏
    render_status_bar(frame, status_area, app);

    // 8. 渲染浮层栈（补全弹窗 / Model / Effort / Theme / 历史 / 审批 / 帮助）
    if matches!(
        app.overlay.active,
        Some(crate::overlay::OverlayKind::FileMention { .. })
    ) && app.file_cache.consume_generation_update()
    {
        app.sync_input_completion_overlay();
    }
    render_overlay(
        frame,
        area,
        input_area,
        &mut app.overlay,
        &app.user_skill_names,
        &app.theme,
    );

    // 9. 应用级高亮选区叠绘 + 快照缓冲供拖拽/双击取词与 Ctrl+C 复制使用
    paint_selection_overlay(frame.buffer_mut(), &app.selection, &app.theme);
    app.last_buffer = Some(frame.buffer_mut().clone());
}

/// Launchpad 快捷操作芯片在欢迎页行内的水平跨度元数据。
struct WelcomeChipSpan {
    action: LaunchpadQuickAction,
    x_offset: u16,
    width: u16,
}

/// 欢迎页序言构行结果：包含行序列以及快捷操作行相对该块的索引与各按钮水平跨度。
struct WelcomeSection {
    lines: Vec<RenderedLine>,
    chips_line_idx: Option<usize>,
    chip_spans: Vec<WelcomeChipSpan>,
}

/// 统一构建 Launchpad 欢迎页序言行与快捷芯片相对布局，供全屏对话流与 Inline 原生 scrollback 共用。
fn build_welcome_section(app: &AppState, width: usize, height: usize) -> WelcomeSection {
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
    push_welcome_wrapped(
        &mut lines,
        Span::styled("  会话    ", Style::default().fg(theme.text_dim)),
        vec![
            Span::styled(
                short_id(&app.session_id).to_string(),
                Style::default().fg(theme.text_muted),
            ),
            Span::raw("   "),
            Span::styled(
                format!("approval: {}", app.approval_mode.as_str()),
                Style::default().fg(theme.text_dim),
            ),
        ],
        width,
    );
    lines.push(RenderedLine::hard(Line::raw("")));
    push_welcome_wrapped(
        &mut lines,
        Span::styled("  操作    ", Style::default().fg(theme.text_dim)),
        vec![
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
            Span::styled(" 模型  ·  ", Style::default().fg(theme.text_muted)),
            Span::styled(
                "Ctrl+B",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" 侧栏", Style::default().fg(theme.text_muted)),
        ],
        width,
    );
    push_welcome_wrapped(
        &mut lines,
        Span::raw("          "),
        vec![
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
            Span::styled(
                "Ctrl+T",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" 轨迹  ·  ", Style::default().fg(theme.text_muted)),
            Span::styled(
                "?",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" 帮助", Style::default().fg(theme.text_muted)),
        ],
        width,
    );

    let mut chips_line_idx = None;
    let mut chip_spans_meta = Vec::new();
    if height >= 12 && width >= 56 && lines.len() + 1 < height {
        let actions = [
            (" [模型] ", LaunchpadQuickAction::ModelPicker),
            (" [主题] ", LaunchpadQuickAction::ThemePicker),
            (" [技能] ", LaunchpadQuickAction::SkillPicker),
            (" [轨迹] ", LaunchpadQuickAction::TrajectoryScene),
            (" [侧栏] ", LaunchpadQuickAction::ToggleSidePanel),
        ];
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

/// 将单条已稳定完成的对话项渲染为可推送至 Inline 原生 scrollback 的行序列。
pub fn build_item_lines_for_scrollback(
    item: &ChatItem,
    extra: Option<&ToolExtraMeta>,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    show_thinking: bool,
    verbose_expanded: bool,
) -> Vec<Line<'static>> {
    let w = width.max(12);
    let rendered = match item {
        ChatItem::User { text } => build_user_lines(text, w, theme),
        ChatItem::Assistant { reasoning, text } => build_assistant_lines(
            reasoning,
            if show_thinking {
                reasoning.as_str()
            } else {
                ""
            },
            text.as_str(),
            show_thinking,
            w,
            theme,
            emphasis,
        ),
        ChatItem::ToolCall {
            name,
            args_summary,
            status,
            output_preview,
            ..
        } => {
            let show_expand_hint = !verbose_expanded && is_tool_expandable(item, extra, w, theme);
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                verbose_expanded,
                ToolGroupCtx {
                    is_group_end: true,
                    show_expand_hint,
                },
                0,
                w,
                theme,
            )
        }
        ChatItem::Notice { level, text } => build_notice_lines(*level, text, w, theme, emphasis),
    };
    rendered.into_iter().map(|rl| rl.line).collect()
}

/// 在完整条目切片上下文中将第 `idx` 条渲染为 Inline scrollback 行序列，保持连续工具卡片分组去空行一致。
#[allow(clippy::too_many_arguments)]
pub fn build_item_lines_for_scrollback_at(
    items: &[ChatItem],
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
    show_thinking: bool,
    verbose_expanded: bool,
) -> Vec<Line<'static>> {
    let Some(item) = items.get(idx) else {
        return Vec::new();
    };
    let w = width.max(12);
    let rendered = match item {
        ChatItem::User { text } => build_user_lines(text, w, theme),
        ChatItem::Assistant { reasoning, text } => build_assistant_lines(
            reasoning,
            if show_thinking {
                reasoning.as_str()
            } else {
                ""
            },
            text.as_str(),
            show_thinking,
            w,
            theme,
            emphasis,
        ),
        ChatItem::ToolCall {
            id,
            name,
            args_summary,
            status,
            output_preview,
        } => {
            let extra = tool_extras.get(id);
            let group_ctx =
                compute_tool_group_ctx(items, 0, idx, tool_extras, verbose_expanded, w, theme);
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                verbose_expanded,
                group_ctx,
                0,
                w,
                theme,
            )
        }
        ChatItem::Notice { level, text } => build_notice_lines(*level, text, w, theme, emphasis),
    };
    rendered.into_iter().map(|rl| rl.line).collect()
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let total_items = app.items.len();
    let start_item = if app.screen_mode == ScreenMode::Inline {
        app.inline_committed_items.min(total_items)
    } else {
        0
    };
    let user_turn_count = app.items[start_item..]
        .iter()
        .filter(|i| matches!(i, ChatItem::User { .. }))
        .count();
    let reserve_rail = app.screen_mode == ScreenMode::Fullscreen
        && area.width >= RAIL_MIN_TERMINAL_WIDTH
        && area.height >= 3
        && user_turn_count >= RAIL_MIN_TURNS;

    let content_w = if reserve_rail {
        area.width.saturating_sub(RAIL_WIDTH)
    } else {
        area.width
    };
    let content_area = Rect::new(area.x, area.y, content_w, area.height);
    app.selection_geom.chat_rect = Some(content_area);

    let width_usize = content_area.width.max(1) as usize;

    let mut all_lines: Vec<RenderedLine> = Vec::with_capacity((total_items - start_item) * 4);
    let mut turns: Vec<TimelineTurn> = Vec::with_capacity(user_turn_count);

    // 借用拆分：先收集各 item 的只读引用数据并按需更新 reveal 与 item_cache
    let theme = app.theme.clone();
    let emphasis = app.emphasis;
    let is_busy = app.is_busy;
    let show_thinking = app.show_thinking;
    let verbose_expanded = app.verbose_expanded;
    let spinner_tick = app.spinner_tick;

    if app.screen_mode == ScreenMode::Inline && !app.items.is_empty() && start_item >= total_items {
        let hint = Line::from(vec![
            Span::styled(
                " ↑ 历史回复已归档至终端原生滚动区 ",
                Style::default().fg(theme.text_dim),
            ),
            Span::styled("│ F3 切回全屏", Style::default().fg(theme.accent)),
        ]);
        frame.buffer_mut().set_line(
            content_area.x,
            content_area.y + content_area.height.saturating_sub(1),
            &hint,
            content_area.width,
        );
        return;
    }

    // 全屏模式始终在顶部保留欢迎页序言；Inline 模式仅在空会话时于视口内渲染欢迎页
    let mut welcome_chips: Option<(usize, Vec<WelcomeChipSpan>)> = None;
    if app.screen_mode == ScreenMode::Fullscreen || app.items.is_empty() {
        let welcome = build_welcome_section(
            app,
            content_area.width as usize,
            content_area.height as usize,
        );
        all_lines.extend(welcome.lines);
        if let Some(chip_line_idx) = welcome.chips_line_idx {
            welcome_chips = Some((chip_line_idx, welcome.chip_spans));
        }
        if start_item < total_items {
            all_lines.push(RenderedLine::hard(Line::raw("")));
        }
    }

    for idx in start_item..total_items {
        let start_line = all_lines.len();
        let is_last_active = is_busy && idx + 1 == total_items;

        match &app.items[idx] {
            ChatItem::User { text } => {
                let preview = text
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or(text.as_str())
                    .to_string();
                turns.push(TimelineTurn {
                    item_index: idx,
                    top_line: start_line,
                    preview,
                });

                let key = ItemRenderKey {
                    width: width_usize,
                    theme_id: theme.id,
                    emphasis_enabled: emphasis.enabled,
                    round_caps: emphasis.round_caps,
                    expanded: verbose_expanded,
                    status_code: 0,
                    reasoning_len: 0,
                    text_len: text.len(),
                    preview_len: 0,
                };
                let lines = get_or_build_cached(&mut app.item_cache, idx, key, || {
                    build_user_lines(text, width_usize, &theme)
                });
                all_lines.extend(lines);
            }
            ChatItem::Assistant { reasoning, text } => {
                app.reveal.observe(
                    RevealSlot::AssistantReasoning(idx),
                    reasoning,
                    is_last_active,
                );
                app.reveal
                    .observe(RevealSlot::AssistantText(idx), text, is_last_active);

                let vis_reasoning = if show_thinking {
                    app.reveal
                        .visible_slice(RevealSlot::AssistantReasoning(idx), reasoning)
                } else {
                    ""
                };
                let vis_text = app
                    .reveal
                    .visible_slice(RevealSlot::AssistantText(idx), text);

                // 当思考被折叠且确实存在思考内容时，用 status_code=1 区分缓存
                let collapsed_marker = if !show_thinking && !reasoning.is_empty() {
                    1
                } else {
                    0
                };
                let key = ItemRenderKey {
                    width: width_usize,
                    theme_id: theme.id,
                    emphasis_enabled: emphasis.enabled,
                    round_caps: emphasis.round_caps,
                    expanded: verbose_expanded,
                    status_code: collapsed_marker,
                    reasoning_len: vis_reasoning.len(),
                    text_len: vis_text.len(),
                    preview_len: if show_thinking { 0 } else { reasoning.len() },
                };
                let lines = get_or_build_cached(&mut app.item_cache, idx, key, || {
                    build_assistant_lines(
                        reasoning,
                        vis_reasoning,
                        vis_text,
                        show_thinking,
                        width_usize,
                        &theme,
                        &emphasis,
                    )
                });
                all_lines.extend(lines);
            }
            ChatItem::ToolCall {
                id,
                name,
                args_summary,
                status,
                output_preview,
            } => {
                let extra = app.tool_extras.get(id).cloned();
                let expanded = verbose_expanded;
                let group_ctx = compute_tool_group_ctx(
                    &app.items,
                    start_item,
                    idx,
                    &app.tool_extras,
                    expanded,
                    width_usize,
                    &theme,
                );
                let extra_len = extra.as_ref().map_or(0, |e| {
                    e.full_content.len()
                        + e.diff_raw.as_ref().map_or(0, String::len)
                        + e.full_args.len()
                        + e.inline_meta.as_ref().map_or(0, String::len)
                });
                let base_status = match status {
                    ToolState::Running => 0u8,
                    ToolState::Done => 1u8,
                    ToolState::Error => 2u8,
                    ToolState::Blocked => 3u8,
                };
                let status_code = base_status
                    | if group_ctx.is_group_end { 4 } else { 0 }
                    | if group_ctx.show_expand_hint { 8 } else { 0 };
                let spin_key = if matches!(status, ToolState::Running) {
                    spinner_tick % BRAILLE_SPINNER.len()
                } else {
                    0
                };
                let key = ItemRenderKey {
                    width: width_usize,
                    theme_id: theme.id,
                    emphasis_enabled: emphasis.enabled,
                    round_caps: emphasis.round_caps,
                    expanded,
                    status_code,
                    reasoning_len: spin_key,
                    text_len: extra_len,
                    preview_len: args_summary.len() + output_preview.len() + name.len(),
                };
                let lines = get_or_build_cached(&mut app.item_cache, idx, key, || {
                    build_tool_lines(
                        name,
                        args_summary,
                        *status,
                        output_preview,
                        extra.as_ref(),
                        expanded,
                        group_ctx,
                        spin_key,
                        width_usize,
                        &theme,
                    )
                });
                all_lines.extend(lines);
            }
            ChatItem::Notice { level, text } => {
                let level_code = match level {
                    NoticeLevel::Info => 0,
                    NoticeLevel::Warn => 1,
                    NoticeLevel::Error => 2,
                };
                let key = ItemRenderKey {
                    width: width_usize,
                    theme_id: theme.id,
                    emphasis_enabled: emphasis.enabled,
                    round_caps: emphasis.round_caps,
                    expanded: verbose_expanded,
                    status_code: level_code,
                    reasoning_len: 0,
                    text_len: text.len(),
                    preview_len: 0,
                };
                let lines = get_or_build_cached(&mut app.item_cache, idx, key, || {
                    build_notice_lines(*level, text, width_usize, &theme, &emphasis)
                });
                all_lines.extend(lines);
            }
        }
    }

    let total_lines_u16 = all_lines.len().min(u16::MAX as usize) as u16;
    let scroll_top = if app.items.is_empty() && app.auto_scroll {
        app.last_viewport_height = content_area.height;
        app.last_total_lines = total_lines_u16;
        app.scroll_from_top = 0;
        0
    } else {
        app.effective_scroll(content_area.height, total_lines_u16) as usize
    };
    let max_scroll = total_lines_u16.saturating_sub(content_area.height) as usize;
    app.timeline_snap = build_timeline_snapshot(turns, scroll_top, max_scroll);

    // 顶部吸顶轮次条（仅当活动 turn 首行已滚出视口上方且视口足够高时显示）
    let show_sticky = content_area.height >= 6
        && app.timeline_snap.pinned_idx.is_some()
        && all_lines.len() > content_area.height as usize;

    let buf = frame.buffer_mut();
    let sticky_rows: usize = if show_sticky {
        if let Some(pinned_idx) = app.timeline_snap.pinned_idx
            && let Some(turn) = app.timeline_snap.turns.get(pinned_idx)
        {
            let sticky_rect = Rect::new(content_area.x, content_area.y, content_area.width, 1);
            let sticky_bg = if theme.card_bg != Color::Reset {
                theme.card_bg
            } else {
                Color::DarkGray
            };
            buf.set_style(sticky_rect, Style::default().bg(sticky_bg));
            let label = format!(
                " ↳ Turn #{}: {} ",
                pinned_idx + 1,
                truncate_display(
                    &turn.preview,
                    content_area.width.saturating_sub(14) as usize
                )
            );
            buf.set_stringn(
                content_area.x,
                content_area.y,
                label,
                content_area.width as usize,
                Style::default()
                    .fg(theme.accent)
                    .bg(sticky_bg)
                    .add_modifier(Modifier::BOLD),
            );
        }
        1
    } else {
        0
    };

    // 逐行绘入 Buffer 并登记软换行选区行跨度
    let view_h = content_area.height as usize;
    for row_off in sticky_rows..view_h {
        let line_idx = scroll_top + row_off;
        let y = content_area.y + row_off as u16;
        let Some(rl) = all_lines.get(line_idx) else {
            break;
        };
        buf.set_line(content_area.x, y, &rl.line, content_area.width);
        let lw = (rl.line.width() as u16).min(content_area.width);
        let col_end = content_area.x + lw.max(1).min(content_area.width);
        app.selection_geom.push_row(
            SelectionPane::Chat,
            y,
            content_area.x,
            col_end,
            rl.soft_wrapped,
        );
    }

    // 仅当快捷操作行落在可视区间 `[sticky_rows..view_h)` 内时动态投影命中矩形
    if let Some((chip_line_idx, chip_spans)) = welcome_chips
        && chip_line_idx >= scroll_top + sticky_rows
        && chip_line_idx - scroll_top < view_h
    {
        let y = content_area.y + (chip_line_idx - scroll_top) as u16;
        for chip in chip_spans {
            app.launchpad_hit_rects.push((
                chip.action,
                Rect::new(content_area.x + chip.x_offset, y, chip.width, 1),
            ));
        }
    }

    // 2 列时间轴导轨绘制（使用同一份 timeline_snap 快照）
    if reserve_rail {
        let rail_area = Rect::new(
            content_area.x + content_area.width,
            area.y,
            RAIL_WIDTH,
            area.height,
        );
        let at_bottom = app.auto_scroll || scroll_top >= max_scroll;
        if let Some(geo) = compute_rail_geometry(rail_area, &app.timeline_snap, at_bottom) {
            render_timeline_rail(
                frame.buffer_mut(),
                &geo,
                &app.timeline_snap,
                app.hovered_rail_row,
                &app.theme,
            );
            app.rail_geo = Some(geo);
        }
    }
}

fn get_or_build_cached<F>(
    cache: &mut std::collections::HashMap<usize, crate::app::CachedItemLines>,
    idx: usize,
    key: ItemRenderKey,
    builder: F,
) -> Vec<RenderedLine>
where
    F: FnOnce() -> Vec<RenderedLine>,
{
    if let Some(hit) = cache.get(&idx)
        && hit.key == key
    {
        return hit.lines.clone();
    }
    let lines = builder();
    cache.insert(
        idx,
        crate::app::CachedItemLines {
            key,
            lines: lines.clone(),
        },
    );
    lines
}

fn build_user_lines(text: &str, width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let user_style = Style::default().fg(theme.text).bg(theme.user_bg);
    let image_pill_style = Style::default()
        .fg(theme.info)
        .bg(theme.user_bg)
        .add_modifier(Modifier::BOLD);
    let prefix_first = Span::styled(
        "❯ ",
        Style::default()
            .fg(theme.accent)
            .bg(theme.user_bg)
            .add_modifier(Modifier::BOLD),
    );
    let prefix_cont = Span::styled("  ", Style::default().bg(theme.user_bg));

    let raw_lines: Vec<&str> = if text.is_empty() {
        vec![""]
    } else {
        text.split('\n').collect()
    };

    for (i, line) in raw_lines.iter().enumerate() {
        let pf = if i == 0 {
            Some(prefix_first.clone())
        } else {
            Some(prefix_cont.clone())
        };
        let spans = render_inline_pills(line, user_style, image_pill_style, false);
        let wrapped = wrap_styled_spans_rich(spans, pf, Some(prefix_cont.clone()), width);
        out.extend(wrapped);
    }
    out.push(RenderedLine::hard(Line::raw("")));
    out
}

fn build_assistant_lines(
    full_reasoning: &str,
    vis_reasoning: &str,
    vis_text: &str,
    show_thinking: bool,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();

    let mut emitted_reasoning = false;
    if !full_reasoning.is_empty() {
        if show_thinking {
            if !vis_reasoning.is_empty() {
                let quote_w = width.saturating_sub(2).max(2);
                let r_lines = render_markdown_rich(vis_reasoning, quote_w, theme, emphasis);
                for rl in r_lines {
                    let mut spans = vec![Span::styled("┆ ", Style::default().fg(theme.text_dim))];
                    for s in rl.line.spans {
                        spans.push(Span::styled(
                            s.content,
                            s.style.fg(theme.text_dim).add_modifier(Modifier::ITALIC),
                        ));
                    }
                    out.push(RenderedLine {
                        line: Line::from(spans),
                        soft_wrapped: rl.soft_wrapped,
                    });
                    emitted_reasoning = true;
                }
            }
        } else {
            let line_count = full_reasoning.lines().count().max(1);
            let collapsed = format!("┆ 思考过程已折叠 ({line_count} 行，按 Ctrl+O 展开)");
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    collapsed,
                    Style::default()
                        .fg(theme.text_dim)
                        .add_modifier(Modifier::ITALIC),
                )],
                None,
                None,
                width,
            ));
            emitted_reasoning = true;
        }
    }

    if !vis_text.is_empty() {
        if emitted_reasoning {
            out.push(RenderedLine::hard(Line::raw("")));
        }
        let md_lines = render_markdown_rich(vis_text, width, theme, emphasis);
        out.extend(md_lines);
    } else if full_reasoning.is_empty() {
        out.push(RenderedLine::hard(Line::from(Span::styled(
            "● …",
            Style::default().fg(theme.text_dim),
        ))));
    }

    out.push(RenderedLine::hard(Line::raw("")));
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ToolGroupCtx {
    is_group_end: bool,
    show_expand_hint: bool,
}

const EXPAND_HINT_INLINE: &str = " (Ctrl+O 展开)";

fn compute_tool_group_ctx(
    items: &[ChatItem],
    start_item: usize,
    idx: usize,
    tool_extras: &HashMap<String, ToolExtraMeta>,
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
    while let ChatItem::ToolCall { id, .. } = &items[j] {
        if is_tool_expandable(&items[j], tool_extras.get(id), width, theme) {
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

fn compact_diff_preview(diff_raw: &str) -> (String, bool) {
    let mut changed = Vec::new();
    let mut skipped_non_header = false;
    for line in diff_raw.lines() {
        if line.starts_with("--- ") || line.starts_with("+++ ") {
            continue;
        }
        if line.starts_with('+') || line.starts_with('-') {
            changed.push(line);
        } else if !line.trim().is_empty() {
            skipped_non_header = true;
        }
    }
    if changed.is_empty() {
        (diff_raw.to_string(), false)
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

fn is_tool_expandable(
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

    let body_w = width.saturating_sub(4).max(4);
    let inline_meta = resolve_inline_meta(name, *status, output_preview, extra);
    let abbr_args = abbreviate_home_path(args_summary);
    let (glyph, _) = tool_status_glyph(*status, 0, theme);
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
        .map_or(output_preview.as_str(), |e| e.full_content.as_str());

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
            output_preview.as_str()
        };
        let shown = collapsed_preview_lines(name, *status, preview_src);
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

    let shown = collapsed_preview_lines(name, *status, output_preview);
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
fn build_tool_lines(
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

fn build_notice_lines(
    level: NoticeLevel,
    text: &str,
    width: usize,
    theme: &Theme,
    emphasis: &EmphasisConfig,
) -> Vec<RenderedLine> {
    let (prefix, color) = match level {
        NoticeLevel::Info => ("! ", theme.info),
        NoticeLevel::Warn => ("⚠ ", theme.warning),
        NoticeLevel::Error => ("✗ ", theme.error),
    };
    let mut out = Vec::new();
    let prefix_span = Span::styled(
        prefix,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    );
    let cont_span = Span::raw("  ");
    let raw_lines: Vec<&str> = text.split('\n').collect();
    for (idx, line) in raw_lines.iter().enumerate() {
        let spans = parse_inline_rich(line, Style::default().fg(color), theme, emphasis, false);
        let pf = if idx == 0 {
            Some(prefix_span.clone())
        } else {
            Some(cont_span.clone())
        };
        out.extend(wrap_styled_spans_rich(
            spans,
            pf,
            Some(cont_span.clone()),
            width,
        ));
    }
    out.push(RenderedLine::hard(Line::raw("")));
    out
}

fn render_queued_strip(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let theme = &app.theme;
    let count = app.queued_prompts.len();
    let selected = app.queue_selected;
    let one_line = |text: &str| text.replace('\n', " ");

    let mut lines: Vec<Line<'_>> = Vec::with_capacity(area.height as usize);
    if selected.is_none() {
        let hint = truncate_display(&one_line(&app.queued_prompts[0].text), avail_of(area, 16));
        lines.push(Line::from(vec![
            Span::styled(
                format!(" ↳ 队列({count}) "),
                Style::default()
                    .fg(theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{hint}  ↑ 选择 · Alt+↑ 直接撤回"),
                Style::default().fg(theme.text_muted),
            ),
        ]));
    } else {
        let marker = format!(" ↳ 队列({count}) ");
        let help = " ↑↓ 选择 · ⏎ 编辑 · Alt+↑ 直接撤 · Del 丢弃 · Esc 关闭 ";
        let mut header = vec![Span::styled(
            marker.clone(),
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )];
        if area.width as usize > marker.chars().count() + help.chars().count() {
            header.push(Span::styled(help, Style::default().fg(theme.text_dim)));
        }
        lines.push(Line::from(header));

        // 最多展示 3 条；选中项尽量落在可见窗口内。
        let visible = area.height.saturating_sub(1).min(3) as usize;
        let sel_index = app
            .queued_prompts
            .iter()
            .position(|item| Some(item.id) == selected)
            .unwrap_or(0);
        let start = sel_index.saturating_sub(visible.saturating_sub(1));
        let end = (start + visible).min(count);
        for (offset, item) in app.queued_prompts[start..end].iter().enumerate() {
            let index = start + offset;
            let is_current = Some(item.id) == selected;
            let marker = if is_current { " ❯ " } else { "   " };
            let prefix = format!("{marker}{}. ", index + 1);
            let body = truncate_display(
                &one_line(&item.text),
                avail_of(area, prefix.chars().count() + 1),
            );
            let style = if is_current {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.text_muted)
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(body, style),
            ]));
        }
    }

    let buf = frame.buffer_mut();
    for (row, line) in lines.iter().enumerate().take(area.height as usize) {
        buf.set_line(area.x, area.y + row as u16, line, area.width);
    }
}

/// 队列条一行内除标记外可用的显示列数。
fn avail_of(area: Rect, reserved: usize) -> usize {
    (area.width as usize).saturating_sub(reserved)
}

fn render_input_box(
    frame: &mut Frame<'_>,
    area: Rect,
    layout: &crate::editor::InputLayout,
    app: &mut AppState,
) {
    if area.width < 3 || area.height < 3 {
        return;
    }
    let theme = &app.theme;

    let bash_mode = app.bash_mode();
    let effort_col = effort_color(theme, app.reasoning_effort);
    let mut base_border_color = if let Some(mode) = bash_mode {
        match mode {
            BashInputMode::Context => theme.success,
            BashInputMode::NoContext => theme.text_muted,
        }
    } else if app.is_busy || app.is_bash_running {
        theme.border_busy
    } else if app.reasoning_effort.is_some() {
        effort_col
    } else {
        theme.border_focus
    };

    // High/XHigh/Max 推理强度切换点火扫描波（1.2s 内沿边框做 OKLab 色彩脉冲）
    let mut ignition_progress: Option<f64> = None;
    if bash_mode.is_none()
        && let Some(started) = app.effort_ignition_at
    {
        let elapsed_ms = started.elapsed().as_millis();
        if elapsed_ms < IGNITION_DURATION_MS {
            let p = elapsed_ms as f64 / IGNITION_DURATION_MS as f64;
            ignition_progress = Some(p);
            if theme.truecolor {
                let wave = (1.0 - (p * 2.0 - 1.0).abs()).clamp(0.0, 1.0);
                if let (Some(a), Some(b)) = (color_to_rgb8(effort_col), color_to_rgb8(theme.accent))
                {
                    base_border_color = mix_oklab(a, b, wave).to_color();
                }
            }
        } else {
            app.effort_ignition_at = None;
        }
    }

    let (title, title_style) = match bash_mode {
        Some(BashInputMode::Context) => (
            " Bash (! · 载入上下文 · Enter 执行 · Esc 退出) ".to_string(),
            Style::default()
                .fg(theme.success)
                .add_modifier(Modifier::BOLD),
        ),
        Some(BashInputMode::NoContext) => (
            " Bash (!! · 不载入上下文 · Enter 执行 · Esc 退出) ".to_string(),
            Style::default().fg(theme.text_muted),
        ),
        None if app.is_busy => (
            " 输入 (Enter 排队 · Ctrl+Enter 立即发送 · Tab 思考 · Esc 暂停) ".to_string(),
            Style::default().fg(theme.text_muted),
        ),
        None if app.is_bash_running => (
            " 输入 (Bash 命令执行中 · Esc 取消) ".to_string(),
            Style::default().fg(theme.text_muted),
        ),
        None if app.input.has_collapsed_pastes() && app.input.has_images() => (
            format!(
                " 输入 (Enter 发送 · 已附 {} 张图片 · 含折叠粘贴块，Ctrl+E 展开) ",
                app.input.image_count()
            ),
            Style::default().fg(theme.text_muted),
        ),
        None if app.input.has_collapsed_pastes() => (
            " 输入 (Enter / Ctrl+Enter 发送 · Tab 思考 · 含折叠粘贴块，Ctrl+E 展开) ".to_string(),
            Style::default().fg(theme.text_muted),
        ),
        None if app.input.has_images() => (
            format!(
                " 输入 (Enter / Ctrl+Enter 发送 · 已附 {} 张图片 · Ctrl+V 粘贴 · ? 帮助) ",
                app.input.image_count()
            ),
            Style::default().fg(theme.text_muted),
        ),
        None => (
            " 输入 (Enter / Ctrl+Enter 发送 · Tab 思考 · Ctrl+L 模型 · ? 帮助) ".to_string(),
            Style::default().fg(theme.text_muted),
        ),
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(base_border_color))
        .title(Span::styled(title, title_style));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // 若处于点火扫描窗口内且终端支持真彩色，在顶边与底边绘制流动高亮扫描段
    if let Some(p) = ignition_progress
        && theme.truecolor
        && area.width > 6
    {
        let span_len = (area.width / 5).max(3);
        let travel = area.width.saturating_sub(2);
        let head = ((p * (travel as f64 + span_len as f64)) as u16).min(travel);
        let buf = frame.buffer_mut();
        for dx in 0..span_len {
            let pos = head.saturating_sub(dx);
            if pos >= 1 && pos < area.width.saturating_sub(1) {
                let x = area.x + pos;
                buf[(x, area.y)]
                    .set_style(Style::default().fg(theme.text).add_modifier(Modifier::BOLD));
                let by = area.y + area.height - 1;
                let mirror_x = area.x + area.width - 1 - pos;
                buf[(mirror_x, by)]
                    .set_style(Style::default().fg(theme.text).add_modifier(Modifier::BOLD));
            }
        }
    }

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let scroll_y = layout
        .cursor_row
        .saturating_sub(inner.height.saturating_sub(1));
    let buf = frame.buffer_mut();

    if app.input.is_empty() {
        let placeholder = truncate_display("输入消息、/ 命令或 @ 文件引用…", inner.width as usize);
        buf.set_stringn(
            inner.x,
            inner.y,
            placeholder,
            inner.width as usize,
            Style::default().fg(theme.text_dim),
        );
    } else {
        for row_off in 0..inner.height {
            let line_idx = (scroll_y + row_off) as usize;
            let Some(line_str) = layout.lines.get(line_idx) else {
                break;
            };
            let y = inner.y + row_off;
            let line = render_input_line_with_paste_pills(line_str, theme);
            buf.set_line(inner.x, y, &line, inner.width);
            let lw = (display_width(line_str) as u16).min(inner.width);
            let col_end = inner.x + lw.max(1).min(inner.width);
            app.selection_geom
                .push_row(SelectionPane::Input, y, inner.x, col_end, false);
        }
    }

    // 居中模态浮层打开时不把物理光标抢回到输入框
    if !app.overlay.is_modal() {
        let rel_row = layout.cursor_row.saturating_sub(scroll_y);
        if rel_row < inner.height {
            let cursor_x = inner.x + layout.cursor_col.min(inner.width.saturating_sub(1));
            let cursor_y = inner.y + rel_row;
            frame.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

fn render_inline_pills(
    text: &str,
    base_style: Style,
    pill_style: Style,
    highlight_paste: bool,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut rest = text;

    while !rest.is_empty() {
        let next_image = rest.find("[Image #").map(|i| (i, false));
        let next_paste = if highlight_paste {
            rest.find("[paste #").map(|i| (i, true))
        } else {
            None
        };
        let next = match (next_image, next_paste) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let Some((start, is_paste)) = next else {
            spans.push(Span::styled(rest.to_string(), base_style));
            break;
        };
        if start > 0 {
            spans.push(Span::styled(rest[..start].to_string(), base_style));
        }
        let after = &rest[start..];
        let prefix_len = 8; // "[Image #" 与 "[paste #" 同为 8 字节
        if let Some(end_rel) = after.find(']') {
            let token = &after[..=end_rel];
            let valid = if is_paste {
                token.contains("lines") || token.contains("chars")
            } else {
                token[prefix_len..end_rel]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            };
            if valid {
                spans.push(Span::styled(token.to_string(), pill_style));
                rest = &after[end_rel + 1..];
                continue;
            }
        }
        spans.push(Span::styled(after[..prefix_len].to_string(), base_style));
        rest = &after[prefix_len..];
    }

    if spans.is_empty() {
        spans.push(Span::raw(""));
    }
    spans
}

fn render_input_line_with_paste_pills(line_str: &str, theme: &Theme) -> Line<'static> {
    let base_style = Style::default().fg(theme.text);
    let pill_style = Style::default()
        .fg(theme.info)
        .bg(theme.card_bg)
        .add_modifier(Modifier::BOLD);
    Line::from(render_inline_pills(line_str, base_style, pill_style, true))
}

fn render_status_bar(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = app.theme.clone();
    let is_busy = app.is_busy;
    let is_bash = app.is_bash_running;
    let is_compacting = app.is_compacting;
    let spinner = BRAILLE_SPINNER[app.spinner_tick % BRAILLE_SPINNER.len()];
    let status_text = if is_compacting {
        let detail = app.compaction_status.as_deref().unwrap_or("上下文压缩中…");
        format!("{spinner} {detail}")
    } else if is_busy {
        format!("{spinner} 运行中")
    } else if is_bash {
        format!("{spinner} Bash")
    } else {
        "● 就绪".to_string()
    };
    let status_color = if is_compacting {
        theme.info
    } else if is_busy || is_bash {
        theme.warning
    } else {
        theme.success
    };

    let model_label = app.active_model_label();
    let effort_str = format_effort_label(app.reasoning_effort);
    let effort_col = effort_color(&theme, app.reasoning_effort);
    let total_tok = app
        .total_usage
        .total_tokens
        .unwrap_or_else(|| app.total_usage.grand_total());
    let tokens_str = format_tokens(total_tok);
    let auto_scroll = app.auto_scroll;
    let approval_mode = app.approval_mode;
    let ttft_ms = app.last_ttft_ms;
    let last_tps = app.tps.rate();
    let flash = app.active_flash().map(str::to_string);

    let mut spans = vec![Span::styled(
        format!(" {status_text}"),
        Style::default()
            .fg(status_color)
            .add_modifier(Modifier::BOLD),
    )];

    let avail = area.width as usize;
    if avail >= 34 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            truncate_display(&model_label, 24),
            Style::default().fg(theme.accent),
        ));
        spans.push(Span::styled(
            format!(" [{effort_str}]"),
            Style::default().fg(effort_col),
        ));
    }

    if avail >= 54 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            format!("tok {tokens_str}"),
            Style::default().fg(theme.text_muted),
        ));
    }

    if avail >= 74 {
        if let Some(ttft) = ttft_ms {
            spans.push(Span::styled(
                format!(" · {ttft}ms"),
                Style::default().fg(theme.text_dim),
            ));
        }
        if let Some(tps) = last_tps {
            spans.push(Span::styled(
                format!(" · {tps:.1}t/s"),
                Style::default().fg(theme.info),
            ));
        }
    }

    if approval_mode == ApprovalMode::Ask && avail >= 48 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            "[审批:ask]",
            Style::default().fg(theme.warning),
        ));
    }

    if app.screen_mode == ScreenMode::Inline && avail >= 44 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            "[inline · F3全屏]",
            Style::default().fg(theme.info),
        ));
    } else if !app.mouse_capture_enabled && avail >= 48 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            "[原生选区 · F2恢复]",
            Style::default().fg(theme.info),
        ));
    }

    if !auto_scroll && avail >= 44 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            "[回看中 · End 到底]",
            Style::default().fg(theme.info),
        ));
    }

    if let Some(msg) = flash {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled(
            truncate_display(&msg, avail.saturating_sub(18).max(8)),
            Style::default()
                .fg(theme.success)
                .add_modifier(Modifier::BOLD),
        ));
    } else if avail >= 96 {
        let toggle_hint = if app.verbose_expanded {
            "Ctrl+O 收起"
        } else {
            "Ctrl+O 展开"
        };
        spans.push(Span::styled(
            format!(" │ {toggle_hint} · Ctrl+B 侧栏 · Ctrl+T 轨迹 · ? 帮助"),
            Style::default().fg(theme.text_dim),
        ));
    }

    frame
        .buffer_mut()
        .set_line(area.x, area.y, &Line::from(spans), area.width);
}

fn format_effort_label(effort: Option<ReasoningEffort>) -> &'static str {
    match effort {
        None => "auto",
        Some(ReasoningEffort::Off) => "off",
        Some(ReasoningEffort::Minimal) => "min",
        Some(ReasoningEffort::Low) => "low",
        Some(ReasoningEffort::Medium) => "med",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::XHigh) => "xhigh",
        Some(ReasoningEffort::Max) => "max",
    }
}

fn effort_color(theme: &Theme, effort: Option<ReasoningEffort>) -> Color {
    match effort {
        None => theme.text_muted,
        Some(ReasoningEffort::Off) => theme.effort_colors[0],
        Some(ReasoningEffort::Minimal) => theme.effort_colors[1],
        Some(ReasoningEffort::Low) => theme.effort_colors[2],
        Some(ReasoningEffort::Medium) => theme.effort_colors[3],
        Some(ReasoningEffort::High) => theme.effort_colors[4],
        Some(ReasoningEffort::XHigh) => theme.effort_colors[5],
        Some(ReasoningEffort::Max) => theme.effort_colors[6],
    }
}

fn color_to_rgb8(c: Color) -> Option<Rgb8> {
    match c {
        Color::Rgb(r, g, b) => Some(Rgb8::new(r, g, b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::app::{UiEvent, abbreviate_home_with, summarize_tool_args};

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
    }

    fn render_buffer_text(app: &mut AppState, w: u16, h: u16) -> String {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, app)).unwrap();
        let buf = terminal.backend().buffer();
        let mut rows = Vec::new();
        for y in 0..buf.area.height {
            let mut row = String::new();
            let mut x = 0;
            while x < buf.area.width {
                let sym = buf[(x, y)].symbol();
                row.push_str(sym);
                x += display_width(sym).max(1) as u16;
            }
            rows.push(row.trim_end().to_string());
        }
        rows.join("\n")
    }

    #[test]
    fn welcome_screen_renders_ansi_shadow_banner_and_falls_back_on_small_terminals() {
        assert_eq!(WELCOME_ASCII_LOGO.len(), 6);
        for (idx, row) in WELCOME_ASCII_LOGO.iter().enumerate() {
            let w = display_width(row);
            assert!(
                (40..=42).contains(&w),
                "banner row {idx} unexpected width {w}: {row:?}"
            );
        }

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-welcome",
            Vec::new(),
            None,
            None,
            None,
        );

        let wide = render_buffer_text(&mut app, 120, 30);
        for row in WELCOME_ASCII_LOGO {
            assert!(wide.contains(row), "missing banner row {row:?}:\n{wide}");
        }
        assert!(wide.contains("◆ Mizar"), "missing title:\n{wide}");
        assert!(wide.contains("[模型]"), "missing quick chips:\n{wide}");

        let wide_lines: Vec<&str> = wide.lines().collect();
        assert!(!app.launchpad_hit_rects.is_empty());
        for (_, rect) in &app.launchpad_hit_rects {
            assert!(rect.y < 30, "hit rect out of frame: {rect:?}");
            let row_txt = wide_lines.get(rect.y as usize).copied().unwrap_or("");
            assert!(
                row_txt.contains('['),
                "expected quick chip at y={}, got: {row_txt:?}\n{wide}",
                rect.y
            );
        }

        // 标准 80x24 终端（transcript 高度 20）完整展示艺术字与快捷入口
        let std80 = render_buffer_text(&mut app, 80, 24);
        for row in WELCOME_ASCII_LOGO {
            assert!(
                std80.contains(row),
                "80x24 missing banner row {row:?}:\n{std80}"
            );
        }
        assert!(
            std80.contains("[模型]"),
            "80x24 missing quick chips:\n{std80}"
        );

        // 58x28 中窄终端下长模型名与操作栏折行后，快捷入口命中坐标依然与渲染行精确对齐
        app.provider = Some("magpie".into());
        app.model = Some("mayoru-responses/deepseek-flash-long".into());
        let wrapped = render_buffer_text(&mut app, 58, 28);
        for row in WELCOME_ASCII_LOGO {
            assert!(
                wrapped.contains(row),
                "58x28 missing banner row {row:?}:\n{wrapped}"
            );
        }
        let wrapped_lines: Vec<&str> = wrapped.lines().collect();
        assert!(!app.launchpad_hit_rects.is_empty());
        for (_, rect) in &app.launchpad_hit_rects {
            assert!(rect.y < 28, "wrapped hit rect out of frame: {rect:?}");
            let row_txt = wrapped_lines.get(rect.y as usize).copied().unwrap_or("");
            assert!(
                row_txt.contains('['),
                "expected wrapped quick chip at y={}, got: {row_txt:?}\n{wrapped}",
                rect.y
            );
        }

        // 46x20 下界（transcript_area 恰好为 46x16）仍可正常展示 6 行艺术字
        let boundary = render_buffer_text(&mut app, 46, 20);
        for row in WELCOME_ASCII_LOGO {
            assert!(
                boundary.contains(row),
                "46x20 missing banner row {row:?}:\n{boundary}"
            );
        }

        let narrow_w = render_buffer_text(&mut app, 44, 28);
        assert!(
            !narrow_w.contains("███╗   ███╗"),
            "should fall back on narrow width:\n{narrow_w}"
        );
        assert!(narrow_w.contains("◆ Mizar"), "narrow fallback:\n{narrow_w}");

        let short = render_buffer_text(&mut app, 100, 15);
        assert!(
            !short.contains("███╗   ███╗"),
            "should fall back on short height:\n{short}"
        );
        assert!(short.contains("◆ Mizar"), "short fallback:\n{short}");

        let tiny = render_buffer_text(&mut app, 20, 8);
        assert!(tiny.contains("Mizar"), "tiny fallback:\n{tiny}");

        for preset_theme in [
            Theme::mist_dark(),
            Theme::mist_light(),
            Theme::sakura_dark(),
        ] {
            let mut banner_lines = Vec::new();
            append_welcome_banner(&mut banner_lines, &preset_theme);
            assert_eq!(banner_lines.len(), 6);
            let top_fg = welcome_banner_face_color(&preset_theme, 0, 6);
            let bot_fg = welcome_banner_face_color(&preset_theme, 5, 6);
            assert_ne!(
                top_fg, bot_fg,
                "expected vertical OKLab gradient on {}",
                preset_theme.id
            );
            assert!(
                banner_lines[0]
                    .spans
                    .iter()
                    .any(|s| s.style.fg == Some(preset_theme.accent_soft)),
                "expected accent_soft shadow on {}",
                preset_theme.id
            );
        }

        app.theme = Theme::ansi16();
        assert_eq!(
            welcome_banner_face_color(&app.theme, 0, 6),
            app.theme.accent
        );
        let ansi = render_buffer_text(&mut app, 120, 30);
        for row in WELCOME_ASCII_LOGO {
            assert!(ansi.contains(row), "ansi16 missing row {row:?}:\n{ansi}");
        }
    }

    #[test]
    fn consecutive_read_only_tools_render_one_line_each_with_single_trailing_expand_hint() {
        let theme = Theme::mist_dark();
        let emphasis = EmphasisConfig::default();
        let items = vec![
            ChatItem::ToolCall {
                id: "t1".into(),
                name: "read".into(),
                args_summary: "crates/cordis/src/lib.rs".into(),
                status: ToolState::Done,
                output_preview: "line 1\nline 2".into(),
            },
            ChatItem::ToolCall {
                id: "t2".into(),
                name: "read_file".into(),
                args_summary: "Cargo.toml".into(),
                status: ToolState::Done,
                output_preview: "[package]".into(),
            },
            ChatItem::ToolCall {
                id: "t3".into(),
                name: "list_dir".into(),
                args_summary: "crates".into(),
                status: ToolState::Done,
                output_preview: "cordis/\nmizar-tui/".into(),
            },
            ChatItem::ToolCall {
                id: "t4".into(),
                name: "grep".into(),
                args_summary: "build_tool_lines in crates".into(),
                status: ToolState::Done,
                output_preview: "ui.rs:10: fn build_tool_lines".into(),
            },
            ChatItem::ToolCall {
                id: "t5".into(),
                name: "find".into(),
                args_summary: "*.rs".into(),
                status: ToolState::Done,
                output_preview: "src/lib.rs\nsrc/ui.rs".into(),
            },
        ];
        let mut extras = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            if let ChatItem::ToolCall { id, .. } = item {
                extras.insert(
                    id.clone(),
                    ToolExtraMeta {
                        full_content: format!("full body for {i}\nsecond line"),
                        inline_meta: Some(format!("{} lines", (i + 1) * 10)),
                        ..Default::default()
                    },
                );
            }
        }

        let mut all_lines = Vec::new();
        for idx in 0..items.len() {
            all_lines.extend(build_item_lines_for_scrollback_at(
                &items, idx, &extras, 110, &theme, &emphasis, true, false,
            ));
        }

        // 5 条工具 + 组尾 1 个空行 = 6 行；中间无空行、无 │ 预览线
        assert_eq!(all_lines.len(), 6, "lines: {all_lines:?}");
        let texts: Vec<String> = all_lines.iter().map(line_text).collect();
        for (idx, line) in texts.iter().take(5).enumerate() {
            assert!(line.starts_with("● "), "row {idx}: {line}");
            assert!(!line.contains("│"), "unexpected rail in row {idx}: {line}");
            if idx < 4 {
                assert!(
                    !line.contains("(Ctrl+O 展开)"),
                    "mid-group row {idx} should not have expand hint: {line}"
                );
            } else {
                assert!(
                    line.ends_with("(Ctrl+O 展开)"),
                    "last row should have inline expand hint: {line}"
                );
            }
        }
        assert!(
            texts[5].is_empty(),
            "expected breathing blank line after group"
        );
    }

    #[test]
    fn bash_collapses_long_output_and_ctrl_o_expands_full_command_and_body() {
        let theme = Theme::mist_dark();
        let long_cmd = "cd /tmp/workspace/mizar && cargo check --workspace --all-targets --message-format=short";
        let full_out = (1..=25)
            .map(|i| format!("output line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let extra = ToolExtraMeta {
            full_content: full_out,
            diff_raw: None,
            full_args: format!(r#"{{"command":"{long_cmd}"}}"#),
            inline_meta: Some("25 lines".into()),
            ..Default::default()
        };
        let preview = "output line 24\noutput line 25";

        let collapsed = build_tool_lines(
            "bash",
            long_cmd,
            ToolState::Done,
            preview,
            Some(&extra),
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: true,
            },
            0,
            130,
            &theme,
        );
        let collapsed_txt: Vec<String> = collapsed.iter().map(|rl| line_text(&rl.line)).collect();
        // 1 头行 + 2 尾行 + 1 组尾空行 = 4 行
        assert_eq!(
            collapsed_txt.len(),
            4,
            "collapsed:\n{}",
            collapsed_txt.join("\n")
        );
        assert!(collapsed_txt[0].contains(long_cmd));
        assert!(collapsed_txt[0].contains("25 lines"));
        assert!(collapsed_txt[0].ends_with("(Ctrl+O 展开)"));
        assert!(collapsed_txt[1].contains("output line 24"));
        assert!(collapsed_txt[2].contains("output line 25"));

        // 无输出 bash 单行收口且不显示 (Ctrl+O 展开)
        let silent = build_tool_lines(
            "bash",
            "mkdir -p target/tmp",
            ToolState::Done,
            "",
            None,
            false,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: false,
            },
            0,
            100,
            &theme,
        );
        assert_eq!(silent.len(), 2);
        assert!(!line_text(&silent[0].line).contains("Ctrl+O"));

        // 展开态恢复全部 25 行输出
        let expanded = build_tool_lines(
            "bash",
            "short",
            ToolState::Done,
            preview,
            Some(&extra),
            true,
            ToolGroupCtx {
                is_group_end: true,
                show_expand_hint: false,
            },
            0,
            130,
            &theme,
        );
        let expanded_joined = expanded
            .iter()
            .map(|rl| line_text(&rl.line))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(expanded_joined.contains(long_cmd), "{expanded_joined}");
        assert!(
            expanded_joined.contains("output line 1"),
            "{expanded_joined}"
        );
        assert!(
            expanded_joined.contains("output line 25"),
            "{expanded_joined}"
        );
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
        if let Some(home) = dirs::home_dir()
            && let Some(home_str) = home.to_str()
            && !home_str.is_empty()
            && home_str != "/"
        {
            let raw = format!(r#"{{"path":"{home_str}/workspace/demo.rs"}}"#);
            assert_eq!(summarize_tool_args(&raw), "~/workspace/demo.rs");
        }
    }

    #[test]
    fn tool_group_cache_invalidates_when_adjacent_tool_arrives_or_finishes() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-cache",
            Vec::new(),
            None,
            None,
            None,
        );
        app.apply_ui_event(UiEvent::ToolCall(mizar_llm_types::ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: r#"{"path":"a.rs"}"#.into(),
        }));
        let frame1 = render_buffer_text(&mut app, 100, 14);
        assert!(frame1.contains("read(a.rs)"), "{frame1}");

        // 第二个工具到来后，c1 不再是组尾，必须失效缓存并消除两者之间的空行
        app.apply_ui_event(UiEvent::ToolCall(mizar_llm_types::ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: r#"{"path":"b.rs"}"#.into(),
        }));
        let frame2 = render_buffer_text(&mut app, 100, 14);
        let f2_lines: Vec<&str> = frame2.lines().collect();
        let c1_row = f2_lines
            .iter()
            .position(|l| l.contains("read(a.rs)"))
            .expect("c1 row");
        assert!(
            f2_lines[c1_row + 1].contains("read(b.rs)"),
            "expected no blank line between c1 and c2:\n{frame2}"
        );
        assert!(
            !frame2.contains("(Ctrl+O 展开)"),
            "no expandable body yet:\n{frame2}"
        );

        // c1 完成后拥有可展开正文，组尾 c2 的缓存必须自动刷新并带上 (Ctrl+O 展开)
        app.apply_ui_event(UiEvent::ToolResult {
            call_id: "c1".into(),
            name: "read".into(),
            content: "pub fn a() {}\n".repeat(15),
            is_error: false,
            blocked: false,
            details: Some(serde_json::json!({
                "path": "a.rs",
                "start_line": 1,
                "end_line": 15,
                "total_lines": 15,
                "truncated": false
            })),
        });
        let frame3 = render_buffer_text(&mut app, 100, 14);
        let c2_line = frame3
            .lines()
            .find(|l| l.contains("read(b.rs)"))
            .expect("c2 row");
        assert!(
            c2_line.contains("(Ctrl+O 展开)"),
            "c2 should refresh with expand hint when sibling c1 becomes expandable:\n{frame3}"
        );
    }

    #[test]
    fn bash_mode_input_box_visuals_and_antigravity_cards() {
        use crate::app::{LocalBashOutput, UiEvent};
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-bash-render",
            Vec::new(),
            None,
            None,
            None,
        );

        let draw_with_border = |app: &mut AppState| -> (String, Color) {
            let backend = TestBackend::new(110, 32);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal.draw(|f| render(f, app)).unwrap();
            let buf = terminal.backend().buffer();
            let area = buf.area;
            let mut out = String::new();
            let mut input_top_border_fg = Color::Reset;
            for y in 0..area.height {
                let mut line = String::new();
                let mut x = 0u16;
                while x < area.width {
                    let cell = &buf[(x, y)];
                    let sym = cell.symbol();
                    line.push_str(sym);
                    if sym == "┌" && y > 10 {
                        input_top_border_fg = cell.fg;
                    }
                    x += display_width(sym).max(1) as u16;
                }
                out.push_str(line.trim_end());
                out.push('\n');
            }
            (out, input_top_border_fg)
        };

        // 1. 输入框 `!` 与 `!!` 视觉区分
        app.input.set_text_end("! echo hi");
        let (ctx_screen, ctx_border_fg) = draw_with_border(&mut app);
        assert!(
            ctx_screen.contains("Bash (! · 载入上下文 · Enter 执行 · Esc 退出)"),
            "{ctx_screen}"
        );
        assert_eq!(ctx_border_fg, app.theme.success);

        app.input.set_text_end("!! echo hi");
        let (local_screen, local_border_fg) = draw_with_border(&mut app);
        assert!(
            local_screen.contains("Bash (!! · 不载入上下文 · Enter 执行 · Esc 退出)"),
            "{local_screen}"
        );
        assert_eq!(local_border_fg, app.theme.text_muted);
        assert_ne!(ctx_border_fg, local_border_fg);
        app.input.clear();

        // 2. 运行态 + 完成(!) + 失败(!!) + 取消(!!) 卡片渲染
        app.start_local_bash("b-run", "sleep 2", false);
        let (running_screen, _) = draw_with_border(&mut app);
        assert!(
            running_screen.contains("bash(sleep 2) · 已入上下文"),
            "{running_screen}"
        );
        assert!(running_screen.contains("Bash"), "{running_screen}");

        app.apply_ui_event(UiEvent::LocalBashFinished {
            call_id: "b-run".into(),
            command: "sleep 2".into(),
            exclude_from_context: false,
            output: Ok(LocalBashOutput {
                display: "line 1\nline 2\nline 3\nline 4".into(),
                session_output: "line 1\nline 2\nline 3\nline 4".into(),
                exit_code: Some(0),
                cancelled: false,
                timed_out: false,
            }),
        });
        app.start_local_bash("b-err", "false_cmd", true);
        app.apply_ui_event(UiEvent::LocalBashFinished {
            call_id: "b-err".into(),
            command: "false_cmd".into(),
            exclude_from_context: true,
            output: Ok(LocalBashOutput {
                display: "boom detail\n\nCommand exited with code 2".into(),
                session_output: "boom detail".into(),
                exit_code: Some(2),
                cancelled: false,
                timed_out: false,
            }),
        });
        app.start_local_bash("b-cancel", "tail -f foo", true);
        app.apply_ui_event(UiEvent::LocalBashFinished {
            call_id: "b-cancel".into(),
            command: "tail -f foo".into(),
            exclude_from_context: true,
            output: Ok(LocalBashOutput {
                display: "(no output)\n\nCommand aborted".into(),
                session_output: String::new(),
                exit_code: Some(-1),
                cancelled: true,
                timed_out: false,
            }),
        });

        let (cards_screen, _) = draw_with_border(&mut app);
        assert!(
            cards_screen.contains("● bash(sleep 2) · 已入上下文"),
            "{cards_screen}"
        );
        assert!(
            cards_screen.contains("✗ bash(false_cmd) · 仅本地 · 退出码 2"),
            "{cards_screen}"
        );
        assert!(
            cards_screen.contains("⊘ bash(tail -f foo) · 仅本地 · 已取消"),
            "{cards_screen}"
        );
        assert!(cards_screen.contains("boom detail"), "{cards_screen}");
        assert!(cards_screen.contains("line 3"), "{cards_screen}");
        assert!(cards_screen.contains("line 4"), "{cards_screen}");
        assert!(!cards_screen.contains("line 1"), "{cards_screen}");
        assert!(cards_screen.contains("(Ctrl+O 展开)"), "{cards_screen}");

        // 3. Ctrl+O 展开恢复完整命令与输出
        app.verbose_expanded = true;
        app.item_cache.clear();
        let (expanded_screen, _) = draw_with_border(&mut app);
        assert!(expanded_screen.contains("line 1"), "{expanded_screen}");
        assert!(expanded_screen.contains("line 4"), "{expanded_screen}");
    }

    #[test]
    fn welcome_persists_above_notices_and_turns_with_dynamic_chip_projection() {
        use crate::selection::extract_selected_text;
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-append",
            Vec::new(),
            Some("mock".into()),
            Some("mock-v1".into()),
            None,
        );

        // 1. 推送模型切换通知后，欢迎页仍在顶部，通知隔 1 行空行紧随其后，芯片命中矩形精准对齐
        app.push_notice(NoticeLevel::Info, "已切换后续请求模型为：mock/mock-v1");
        let after_notice = render_buffer_text(&mut app, 100, 30);
        assert!(
            after_notice.contains(WELCOME_ASCII_LOGO[0]),
            "welcome banner replaced by notice:\n{after_notice}"
        );
        assert!(after_notice.contains("◆ Mizar"), "{after_notice}");
        assert!(
            after_notice.contains("已切换后续请求模型为：mock/mock-v1"),
            "{after_notice}"
        );
        let notice_lines: Vec<&str> = after_notice.lines().collect();
        let chip_row = notice_lines
            .iter()
            .position(|l| l.contains("[模型]"))
            .expect("chips row");
        let notice_row = notice_lines
            .iter()
            .position(|l| l.contains("已切换后续请求模型为"))
            .expect("notice row");
        assert_eq!(
            notice_row,
            chip_row + 2,
            "expected exactly 1 blank line between welcome and first notice:\n{after_notice}"
        );
        assert!(!app.launchpad_hit_rects.is_empty());
        for (_, rect) in &app.launchpad_hit_rects {
            assert_eq!(rect.y as usize, chip_row, "hit rect misaligned: {rect:?}");
        }

        // 2. 首轮对话直接追加在欢迎页与通知之后
        app.push_user_message("第一轮问题");
        app.items.push(ChatItem::Assistant {
            reasoning: String::new(),
            text: "第一轮答复".into(),
        });
        let after_turn1 = render_buffer_text(&mut app, 100, 34);
        assert!(after_turn1.contains(WELCOME_ASCII_LOGO[0]), "{after_turn1}");
        assert!(after_turn1.contains("第一轮问题"), "{after_turn1}");
        assert!(after_turn1.contains("第一轮答复"), "{after_turn1}");
        let t1_lines: Vec<&str> = after_turn1.lines().collect();
        let welcome_idx = t1_lines.iter().position(|l| l.contains("◆ Mizar")).unwrap();
        let user1_idx = t1_lines
            .iter()
            .position(|l| l.contains("第一轮问题"))
            .unwrap();
        let asst1_idx = t1_lines
            .iter()
            .position(|l| l.contains("第一轮答复"))
            .unwrap();
        assert!(welcome_idx < user1_idx && user1_idx < asst1_idx);

        // 3. 多轮超屏后 auto_scroll 自然顶出欢迎页，快捷芯片滚出视口时不注册误命中
        for i in 2..=6 {
            app.push_user_message(&format!("第 {i} 轮问题"));
            app.items.push(ChatItem::Assistant {
                reasoning: String::new(),
                text: format!("第 {i} 轮正文\n细节 A\n细节 B"),
            });
        }
        let scrolled_bottom = render_buffer_text(&mut app, 100, 26);
        assert!(
            !scrolled_bottom.contains(WELCOME_ASCII_LOGO[0]),
            "welcome should scroll off top when conversation overflows:\n{scrolled_bottom}"
        );
        assert!(scrolled_bottom.contains("第 6 轮正文"), "{scrolled_bottom}");
        assert!(
            app.launchpad_hit_rects.is_empty(),
            "off-screen chips must not register hit rects: {:?}",
            app.launchpad_hit_rects
        );
        assert!(
            app.timeline_snap
                .turns
                .first()
                .is_some_and(|t| t.top_line > 10),
            "timeline turn top_line should include welcome offset: {:?}",
            app.timeline_snap.turns
        );

        // 4. 部分向上滚动时，芯片命中矩形随 scroll_top 同步上移且不与吸顶条重叠
        app.auto_scroll = false;
        app.scroll_from_top = 4;
        let partial = render_buffer_text(&mut app, 100, 26);
        let partial_chip_row = partial
            .lines()
            .position(|l| l.contains("[模型]"))
            .expect("partially scrolled chips row");
        assert!(!app.launchpad_hit_rects.is_empty());
        for (_, rect) in &app.launchpad_hit_rects {
            assert_eq!(rect.y as usize, partial_chip_row);
        }

        // 5. 滚回顶部后，欢迎页完整可见、无吸顶条遮挡、划词复制与点击侧栏芯片均可正常工作
        app.scroll_up(u16::MAX);
        let back_top = render_buffer_text(&mut app, 100, 26);
        assert!(back_top.contains(WELCOME_ASCII_LOGO[0]), "{back_top}");
        assert!(back_top.contains("◆ Mizar"), "{back_top}");
        assert!(!back_top.contains("↳ Turn #"), "{back_top}");
        let top_lines: Vec<&str> = back_top.lines().collect();
        let top_chip_row = top_lines
            .iter()
            .position(|l| l.contains("[模型]"))
            .expect("top chip row");
        assert!(!app.launchpad_hit_rects.is_empty());
        let (_, side_chip) = app
            .launchpad_hit_rects
            .iter()
            .find(|(qa, _)| matches!(qa, LaunchpadQuickAction::ToggleSidePanel))
            .expect("side panel chip at top")
            .clone();
        assert_eq!(side_chip.y as usize, top_chip_row);

        // 验证欢迎页区域登记了 SelectionPane::Chat，支持划词提取
        let mizar_row = top_lines
            .iter()
            .position(|l| l.contains("◆ Mizar"))
            .unwrap() as u16;
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: mizar_row,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 10,
            row: mizar_row,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 10,
            row: mizar_row,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        let selected = extract_selected_text(
            app.last_buffer.as_ref().unwrap(),
            &app.selection,
            &app.selection_geom,
        )
        .expect("welcome selection text");
        assert!(selected.contains("Mizar"), "selected={selected:?}");
        app.selection.clear();

        // 单击侧栏快捷芯片应正常触发 ToggleSidePanel
        let act = app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: side_chip.x + 1,
            row: side_chip.y,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        assert_eq!(act, crate::app::AppAction::RefreshSidePanelData);
        assert!(app.side_panel.open);

        // 6. 激活 TimelineRail（60 列窄边界，content_area.width = 58）时，欢迎页与芯片不得越界侵占导轨列
        app.side_panel.open = false;
        app.scroll_up(u16::MAX);
        let _rail_top = render_buffer_text(&mut app, 60, 26);
        let rail = app.rail_geo.expect("rail active at 60 cols with >=2 turns");
        assert_eq!(rail.area.x, 58);
        for (_, rect) in &app.launchpad_hit_rects {
            assert!(
                rect.x + rect.width <= rail.area.x,
                "welcome chip overlapped timeline rail: {rect:?} vs rail {rail:?}"
            );
        }
    }

    #[test]
    fn inline_mode_renders_welcome_when_empty_and_scrollback_helper_appends_blank_line() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-inline-ui",
            Vec::new(),
            Some("mock".into()),
            Some("mock-v1".into()),
            None,
        );
        app.screen_mode = ScreenMode::Inline;
        assert!(!app.inline_welcome_committed);

        // 1. 空会话 Inline viewport 直接展示欢迎页，不显示“已归档”提示
        let empty_inline = render_buffer_text(&mut app, 80, 12);
        assert!(empty_inline.contains("Mizar"), "{empty_inline}");
        assert!(!empty_inline.contains("历史回复已归档"), "{empty_inline}");

        // 2. 非空且有未提交 item 时，Inline viewport 仅绘制 start_item..total_items，不重复画欢迎页
        app.push_user_message("未提交提问");
        app.items.push(ChatItem::Assistant {
            reasoning: String::new(),
            text: "流式输出中".into(),
        });
        let streaming_inline = render_buffer_text(&mut app, 80, 12);
        assert!(
            streaming_inline.contains("未提交提问"),
            "{streaming_inline}"
        );
        assert!(
            streaming_inline.contains("流式输出中"),
            "{streaming_inline}"
        );
        assert!(
            !streaming_inline.contains("◆ Mizar"),
            "must not duplicate welcome in inline viewport:\n{streaming_inline}"
        );

        // 3. 全部 commit 后显示归档提示
        app.inline_welcome_committed = true;
        app.inline_committed_items = app.items.len();
        let archived_inline = render_buffer_text(&mut app, 80, 12);
        assert!(
            archived_inline.contains("历史回复已归档至终端原生滚动区"),
            "{archived_inline}"
        );

        // 4. build_welcome_lines_for_scrollback 复用统一逻辑并附带 1 行尾空行
        let sb_lines = build_welcome_lines_for_scrollback(&app, 80);
        let sb_text: Vec<String> = sb_lines.iter().map(line_text).collect();
        for row in WELCOME_ASCII_LOGO {
            assert!(
                sb_text.iter().any(|l| l.contains(row)),
                "missing banner row {row:?} in scrollback"
            );
        }
        assert!(sb_text.iter().any(|l| l.contains("◆ Mizar")));
        assert!(
            sb_text.last().is_some_and(|l| l.is_empty()),
            "expected trailing blank line: {sb_text:?}"
        );

        // 5. reset_for_new_session 重置 inline_welcome_committed
        app.reset_for_new_session("sess-inline-next", None);
        assert!(!app.inline_welcome_committed);
        assert_eq!(app.inline_committed_items, 0);
    }

    #[test]
    fn queued_strip_lists_items_with_marker_when_selector_is_open() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-queued",
            Vec::new(),
            None,
            None,
            None,
        );
        app.is_busy = true;
        for text in ["第一条排队消息", "第二条排队消息"] {
            app.input.insert_str(text);
            app.handle_key(crossterm::event::KeyEvent::new(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ));
        }
        // 选择器关闭：单行摘要，且必须给出「↑ 选择」的可发现性提示。
        let closed = render_buffer_text(&mut app, 100, 20);
        assert!(closed.contains("队列(2)"), "screen:\n{closed}");
        assert!(closed.contains("第一条排队消息"), "screen:\n{closed}");
        assert!(!closed.contains('❯'), "screen:\n{closed}");

        // 选择器打开：逐条列出，选中项带 ❯ 标记。
        app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        ));
        let opened = render_buffer_text(&mut app, 100, 20);
        assert!(opened.contains('❯'), "screen:\n{opened}");
        assert!(opened.contains("第二条排队消息"), "screen:\n{opened}");
        assert!(opened.contains("第一条排队消息"), "screen:\n{opened}");
    }

    #[test]
    fn queued_strip_stays_visible_on_short_terminals() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-short",
            Vec::new(),
            None,
            None,
            None,
        );
        app.is_busy = true;
        app.input.insert_str("queued on short terminal");
        app.handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        // 12 行高以前会被 avail_h >= 6 的门槛整行吞掉。
        let screen = render_buffer_text(&mut app, 80, 12);
        assert!(screen.contains("队列(1)"), "screen:\n{screen}");
    }
}
