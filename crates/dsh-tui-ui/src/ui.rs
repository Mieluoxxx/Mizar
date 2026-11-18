//! TUI 布局与渲染引擎：双栏工作台、时间轴导轨、虚拟化流式视图、`pi-emphasis` 语法、词级 Diff 与浮层叠加。

use dsh_llm_types::ReasoningEffort;
use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::{
    AppState, ApprovalMode, ChatItem, ItemRenderKey, LaunchpadQuickAction, NoticeLevel,
    ToolExtraMeta, ToolState, format_tokens, short_id, truncate_display,
};
use crate::editor::display_width;
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
    app.tool_hit_rects.clear();
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
    let chat_area = if let Some((chat_w, divider_x, panel_w)) = app.side_panel.split_columns(area.width)
    {
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
    let queued_h = if !app.queued_prompts.is_empty() && avail_h >= 6 {
        1
    } else {
        0
    };
    let max_input_h = avail_h.saturating_sub(queued_h + 1).max(3).min(avail_h);
    let input_h = ((input_layout.lines.len() as u16) + 2)
        .clamp(3, 8)
        .min(max_input_h);
    let transcript_h = avail_h.saturating_sub(queued_h + input_h);

    let transcript_area = Rect::new(chat_area.x, chat_area.y, chat_area.width, transcript_h);
    let queued_area = Rect::new(chat_area.x, chat_area.y + transcript_h, chat_area.width, queued_h);
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

    // 4. 渲染对话历史或 Launchpad 欢迎页
    if transcript_area.height > 0 && transcript_area.width > 0 {
        if app.items.is_empty() {
            app.selection_geom.chat_rect = Some(transcript_area);
            render_welcome(frame, transcript_area, app);
        } else {
            render_transcript(frame, transcript_area, app);
        }
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

fn render_welcome(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let theme = &app.theme;
    let model_label = app.active_model_label();
    let effort_str = format_effort_label(app.reasoning_effort);
    let effort_col = effort_color(theme, app.reasoning_effort);

    // 极窄终端紧凑模式（如 20x8）
    if area.height < 7 || area.width < 36 {
        let lines = vec![
            Line::from(vec![
                Span::styled(
                    "dsh-rs TUI",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(
                    truncate_display(&model_label, area.width.saturating_sub(2) as usize),
                    Style::default().fg(theme.text),
                ),
            ]),
            Line::from(Span::styled(
                truncate_display(&model_label, area.width as usize),
                Style::default().fg(theme.success),
            )),
            Line::from(Span::styled(
                truncate_display(&app.workspace.display().to_string(), area.width as usize),
                Style::default().fg(theme.text_muted),
            )),
        ];
        frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
        return;
    }

    let ws_str = truncate_display(
        &app.workspace.display().to_string(),
        area.width.saturating_sub(14) as usize,
    );
    let rule_w = (area.width.saturating_sub(4) as usize).min(64);
    let sep = "─".repeat(rule_w);

    let mut lines = Vec::new();
    let top_pad = if area.height >= 14 { 2 } else { 0 };
    for _ in 0..top_pad {
        lines.push(Line::raw(""));
    }

    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "◆ dsh-rs TUI",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  ", Style::default().fg(theme.text_dim)),
        Span::styled(
            format!("theme: {}", theme.id),
            Style::default().fg(theme.text_muted),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        format!("  {sep}"),
        Style::default().fg(theme.border),
    )));
    lines.push(Line::from(vec![
        Span::styled("  工作区  ", Style::default().fg(theme.text_dim)),
        Span::styled(ws_str, Style::default().fg(theme.text)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  模型    ", Style::default().fg(theme.text_dim)),
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
    ]));
    lines.push(Line::from(vec![
        Span::styled("  会话    ", Style::default().fg(theme.text_dim)),
        Span::styled(
            short_id(&app.session_id),
            Style::default().fg(theme.text_muted),
        ),
        Span::raw("   "),
        Span::styled(
            format!("approval: {}", app.approval_mode.as_str()),
            Style::default().fg(theme.text_dim),
        ),
    ]));
    lines.push(Line::raw(""));
    lines.push(Line::from(vec![
        Span::styled("  操作    ", Style::default().fg(theme.text_dim)),
        Span::styled("Enter", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 发送  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("Ctrl+B", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 侧栏  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("Ctrl+T", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 轨迹透视", Style::default().fg(theme.text_muted)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("          ", Style::default().fg(theme.text_dim)),
        Span::styled("/", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 命令  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("@", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 文件  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("Ctrl+G", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 外部编辑器  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("F3", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 视图  ·  ", Style::default().fg(theme.text_muted)),
        Span::styled("?", Style::default().fg(theme.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" 帮助", Style::default().fg(theme.text_muted)),
    ]));

    if area.height >= 12 && area.width >= 56 {
        let chip_y = area.y + (lines.len() as u16) + 1;
        if chip_y < area.y + area.height {
            let actions = [
                (" [模型] ", LaunchpadQuickAction::ModelPicker),
                (" [主题] ", LaunchpadQuickAction::ThemePicker),
                (" [技能] ", LaunchpadQuickAction::SkillPicker),
                (" [轨迹] ", LaunchpadQuickAction::TrajectoryScene),
                (" [侧栏] ", LaunchpadQuickAction::ToggleSidePanel),
            ];
            lines.push(Line::raw(""));
            let mut chip_spans = vec![Span::styled("  快捷    ", Style::default().fg(theme.text_dim))];
            let mut cur_x = area.x + 10;
            for (label, qa) in actions {
                let w = display_width(label) as u16;
                if cur_x + w < area.x + area.width {
                    app.launchpad_hit_rects
                        .push((qa, Rect::new(cur_x, chip_y, w, 1)));
                    chip_spans.push(Span::styled(
                        label,
                        Style::default()
                            .fg(theme.accent)
                            .bg(theme.card_bg)
                            .add_modifier(Modifier::BOLD),
                    ));
                    chip_spans.push(Span::raw(" "));
                    cur_x += w + 1;
                }
            }
            lines.push(Line::from(chip_spans));
        }
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
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
            if show_thinking { reasoning.as_str() } else { "" },
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
            let expanded = extra
                .and_then(|e| e.local_expanded)
                .unwrap_or(verbose_expanded);
            build_tool_lines(
                name,
                args_summary,
                *status,
                output_preview,
                extra,
                expanded,
                0,
                w,
                theme,
            )
        }
        ChatItem::Notice { level, text } => build_notice_lines(*level, text, w, theme, emphasis),
    };
    rendered.into_iter().map(|rl| rl.line).collect()
}

#[derive(Debug, Clone)]
struct ToolLineRange {
    id: String,
    start_line: usize,
    end_line: usize,
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
    let mut tool_ranges: Vec<ToolLineRange> = Vec::new();

    // 借用拆分：先收集各 item 的只读引用数据并按需更新 reveal 与 item_cache
    let theme = app.theme.clone();
    let emphasis = app.emphasis;
    let is_busy = app.is_busy;
    let show_thinking = app.show_thinking;
    let verbose_expanded = app.verbose_expanded;
    let spinner_tick = app.spinner_tick;

    if app.screen_mode == ScreenMode::Inline && start_item >= total_items {
        let hint = Line::from(vec![
            Span::styled(" ↑ 历史回复已归档至终端原生滚动区 ", Style::default().fg(theme.text_dim)),
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
                app.reveal
                    .observe(RevealSlot::AssistantReasoning(idx), reasoning, is_last_active);
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
                let expanded = extra
                    .as_ref()
                    .and_then(|e| e.local_expanded)
                    .unwrap_or(verbose_expanded);
                let extra_len = extra.as_ref().map_or(0, |e| {
                    e.full_content.len() + e.diff_raw.as_ref().map_or(0, String::len)
                });
                let status_code = match status {
                    ToolState::Running => 0,
                    ToolState::Done => 1,
                    ToolState::Error => 2,
                    ToolState::Blocked => 3,
                };
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
                let tool_id = id.clone();
                let lines = get_or_build_cached(&mut app.item_cache, idx, key, || {
                    build_tool_lines(
                        name,
                        args_summary,
                        *status,
                        output_preview,
                        extra.as_ref(),
                        expanded,
                        spin_key,
                        width_usize,
                        &theme,
                    )
                });
                all_lines.extend(lines);
                let end_line = all_lines.len().saturating_sub(1).max(start_line + 1);
                tool_ranges.push(ToolLineRange {
                    id: tool_id,
                    start_line,
                    end_line,
                });
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
    app.last_viewport_height = content_area.height;
    app.last_total_lines = total_lines_u16;

    let scroll_top = app.effective_scroll(content_area.height, total_lines_u16) as usize;
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
                truncate_display(&turn.preview, content_area.width.saturating_sub(14) as usize)
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

    // 计算当前视口内的 ToolCall 点击矩形
    let visible_start = scroll_top + sticky_rows;
    let visible_end = scroll_top + view_h;
    for tr in tool_ranges {
        let vis_s = tr.start_line.max(visible_start);
        let vis_e = tr.end_line.min(visible_end);
        if vis_e > vis_s {
            let y_top = content_area.y + (vis_s - scroll_top) as u16;
            let height = (vis_e - vis_s) as u16;
            app.tool_hit_rects.push((
                tr.id,
                Rect::new(content_area.x, y_top, content_area.width, height),
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
        let wrapped = wrap_styled_spans_rich(
            vec![Span::styled((*line).to_string(), user_style)],
            pf,
            Some(prefix_cont.clone()),
            width,
        );
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
                }
            }
        } else {
            let line_count = full_reasoning.lines().count().max(1);
            let collapsed = format!("┆ 思考过程已折叠 ({line_count} 行，按 t 展开)");
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
        }
    }

    if !vis_text.is_empty() {
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

#[allow(clippy::too_many_arguments)]
fn build_tool_lines(
    name: &str,
    args_summary: &str,
    status: ToolState,
    output_preview: &str,
    extra: Option<&ToolExtraMeta>,
    expanded: bool,
    spinner_idx: usize,
    width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let mut out = Vec::new();
    let (glyph, glyph_color) = match status {
        ToolState::Running => (BRAILLE_SPINNER[spinner_idx % BRAILLE_SPINNER.len()], theme.warning),
        ToolState::Done => ("✓", theme.success),
        ToolState::Error => ("✗", theme.error),
        ToolState::Blocked => ("⊘", theme.error),
    };

    let bg_card = theme.card_bg;
    let mut header_spans = vec![
        Span::styled(
            format!("{glyph} "),
            Style::default()
                .fg(glyph_color)
                .bg(bg_card)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            name.to_string(),
            Style::default()
                .fg(theme.text)
                .bg(bg_card)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("({args_summary})"),
            Style::default().fg(theme.text_muted).bg(bg_card),
        ),
    ];

    let has_more = extra.is_some_and(|e| {
        !e.full_content.is_empty() && e.full_content != output_preview || e.diff_raw.is_some()
    });
    if has_more && width >= 36 {
        let tag = if expanded {
            "  [▾ 收起]"
        } else {
            "  [▸ 展开]"
        };
        header_spans.push(Span::styled(
            tag,
            Style::default().fg(theme.text_dim).bg(bg_card),
        ));
    }

    out.extend(wrap_styled_spans_rich(header_spans, None, None, width));

    let body_w = width.saturating_sub(4).max(4);
    let prefix = Span::styled("  │ ", Style::default().fg(theme.border));

    // 若有结构化 Diff（如 edit 工具），优先渲染词级/双栏 Diff
    if let Some(diff_raw) = extra.and_then(|e| e.diff_raw.as_deref())
        && !diff_raw.trim().is_empty()
    {
        let diff_lines = render_diff_rich(diff_raw, body_w, theme);
        let max_show = if expanded { usize::MAX } else { 14 };
        let total = diff_lines.len();
        for rl in diff_lines.into_iter().take(max_show) {
            let mut spans = vec![prefix.clone()];
            spans.extend(rl.line.spans);
            out.push(RenderedLine {
                line: Line::from(spans),
                soft_wrapped: rl.soft_wrapped,
            });
        }
        if !expanded && total > max_show {
            out.extend(wrap_styled_spans_rich(
                vec![Span::styled(
                    format!("  │ … 共 {total} 行 Diff（点击或按 Ctrl+O 展开）"),
                    Style::default().fg(theme.text_dim),
                )],
                None,
                None,
                width,
            ));
        }
    } else {
        let body_text = if expanded
            && let Some(ext) = extra
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
            let max_lines = if expanded { 160 } else { 5 };
            let truncated = raw_lines.len().saturating_sub(max_lines);
            for raw in raw_lines.into_iter().take(max_lines) {
                let wrapped = wrap_styled_spans_rich(
                    vec![Span::styled(
                        raw.to_string(),
                        Style::default().fg(body_color),
                    )],
                    Some(prefix.clone()),
                    Some(prefix.clone()),
                    width,
                );
                out.extend(wrapped);
            }
            if truncated > 0 {
                let hint = if expanded {
                    format!("  │ … 其余 {truncated} 行已省略（完整记录请查 Ctrl+T）")
                } else {
                    format!("  │ … 还有 {truncated} 行输出（点击或按 Ctrl+O 展开）")
                };
                out.extend(wrap_styled_spans_rich(
                    vec![Span::styled(hint, Style::default().fg(theme.text_dim))],
                    None,
                    None,
                    width,
                ));
            }
        }
    }

    out.push(RenderedLine::hard(Line::raw("")));
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
    let preview = app
        .queued_prompts
        .first()
        .map(|s| s.replace('\n', " "))
        .unwrap_or_default();
    let avail = area.width.saturating_sub(20) as usize;
    let snippet = truncate_display(&preview, avail);
    let spans = vec![
        Span::styled(
            format!(" ↳ 队列({count}): "),
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(snippet, Style::default().fg(theme.text_muted)),
    ];
    frame
        .buffer_mut()
        .set_line(area.x, area.y, &Line::from(spans), area.width);
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

    let effort_col = effort_color(theme, app.reasoning_effort);
    let mut base_border_color = if app.is_busy {
        theme.border_busy
    } else if app.reasoning_effort.is_some() {
        effort_col
    } else {
        theme.border_focus
    };

    // High/XHigh/Max 推理强度切换点火扫描波（1.2s 内沿边框做 OKLab 色彩脉冲）
    let mut ignition_progress: Option<f64> = None;
    if let Some(started) = app.effort_ignition_at {
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

    let title = if app.is_busy {
        " 输入 (Tab 排队 · Ctrl/Alt+Enter 转向 · Ctrl+C 中断) "
    } else if app.input.has_collapsed_pastes() {
        " 输入 (Enter 发送 · 含折叠粘贴块，Ctrl+E 展开) "
    } else {
        " 输入 (Enter 发送 · / 命令 · @ 文件 · ? 帮助) "
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(base_border_color))
        .title(Span::styled(title, Style::default().fg(theme.text_muted)));

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
        let placeholder = truncate_display(
            "输入消息、/ 命令或 @ 文件引用…",
            inner.width as usize,
        );
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

fn render_input_line_with_paste_pills(line_str: &str, theme: &Theme) -> Line<'static> {
    let base_style = Style::default().fg(theme.text);
    let mut spans = Vec::new();
    let mut rest = line_str;

    while let Some(start) = rest.find("[paste #") {
        if start > 0 {
            spans.push(Span::styled(rest[..start].to_string(), base_style));
        }
        let after = &rest[start..];
        if let Some(end_rel) = after.find(']') {
            let token = &after[..=end_rel];
            if token.contains("lines") {
                spans.push(Span::styled(
                    token.to_string(),
                    Style::default()
                        .fg(theme.info)
                        .bg(theme.card_bg)
                        .add_modifier(Modifier::BOLD),
                ));
                rest = &after[end_rel + 1..];
                continue;
            }
        }
        spans.push(Span::styled("[paste #".to_string(), base_style));
        rest = &rest[start + 8..];
    }

    if !rest.is_empty() {
        spans.push(Span::styled(rest.to_string(), base_style));
    }
    if spans.is_empty() {
        Line::raw("")
    } else {
        Line::from(spans)
    }
}

fn render_status_bar(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = app.theme.clone();
    let is_busy = app.is_busy;
    let spinner = BRAILLE_SPINNER[app.spinner_tick % BRAILLE_SPINNER.len()];
    let status_text = if is_busy {
        format!("{spinner} 运行中")
    } else {
        "● 就绪".to_string()
    };
    let status_color = if is_busy { theme.warning } else { theme.success };

    let model_label = app.active_model_label();
    let effort_str = format_effort_label(app.reasoning_effort);
    let effort_col = effort_color(&theme, app.reasoning_effort);
    let total_tok = app
        .total_usage
        .total_tokens
        .unwrap_or(app.total_usage.input_tokens + app.total_usage.output_tokens);
    let tokens_str = format_tokens(total_tok);
    let auto_scroll = app.auto_scroll;
    let approval_mode = app.approval_mode;
    let ttft_ms = app.last_ttft_ms;
    let last_tps = app.last_tps;
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
        spans.push(Span::styled("[inline · F3全屏]", Style::default().fg(theme.info)));
    } else if !app.mouse_capture_enabled && avail >= 48 {
        spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
        spans.push(Span::styled("[原生选区 · F2恢复]", Style::default().fg(theme.info)));
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
        spans.push(Span::styled(
            " │ Ctrl+B 侧栏 · Ctrl+T 轨迹 · ? 帮助",
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
