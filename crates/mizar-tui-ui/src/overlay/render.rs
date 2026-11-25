//! 浮层渲染实现：命令/引用补全弹窗、居中选择器（模型/思考/主题/历史/技能/会话/分支树）、设置中心、审批与帮助面板。

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use unicode_width::UnicodeWidthChar;

use super::{
    OverlayKind, OverlayState, SessionPickerItem, SettingEntry, SettingItemKind, all_efforts,
    filter_session_picker_indices, filter_setting_indices, filter_slash,
};
use crate::editor::display_width;
use crate::theme::{Theme, ThemePreset};

fn truncate_to_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 1 {
        return "…".to_string();
    }
    let target = max_width - 1;
    let mut out = String::new();
    let mut cur = 0usize;
    for ch in text.chars() {
        let w = ch.width().unwrap_or(0);
        if cur + w > target {
            break;
        }
        out.push(ch);
        cur += w;
    }
    out.push('…');
    out
}

fn pad_to_display_width(text: &str, width: usize) -> String {
    let truncated = truncate_to_display_width(text, width);
    let pad = width.saturating_sub(display_width(&truncated));
    if pad == 0 {
        truncated
    } else {
        format!("{truncated}{}", " ".repeat(pad))
    }
}

/// 按终端显示宽度对齐单条会话选择器行，保障包含全角符号（如 `｜`）时跨行列对齐。
pub fn format_session_picker_row(
    item: &SessionPickerItem,
    title_col_w: usize,
    cwd_col_w: usize,
) -> (String, String) {
    let title_col = pad_to_display_width(&item.title, title_col_w);
    let cwd_col = pad_to_display_width(&item.cwd_tail, cwd_col_w);
    let meta_col = format!(
        "{cwd_col} · {:>3} ev · {}",
        item.event_count, item.updated_at
    );
    (title_col, meta_col)
}

#[allow(clippy::too_many_arguments)]
fn render_settings_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    items: &[SettingEntry],
    query: &str,
    selected: usize,
    save_path_hint: &str,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let filtered = filter_setting_indices(items, query);
    let box_w = full_area.width.saturating_sub(4).clamp(44, 86);
    let box_h = full_area.height.saturating_sub(4).clamp(11, 22);
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);

    frame.render_widget(Clear, rect);
    let title = if query.is_empty() {
        " ⚙ 设置中心 (/settings · ↑↓ 移动 · ←/→/Enter 切换 · Esc 关闭) ".to_string()
    } else {
        format!(" ⚙ 设置中心 [过滤: {query}] (Esc 清空过滤) ")
    };
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border_focus))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    if inner.height < 4 || inner.width < 16 {
        return;
    }

    // 底部预留 3 行（分割线 + 当前项说明 + 保存路径提示）
    let list_h = (inner.height as usize).saturating_sub(3).max(1);
    let mut lines: Vec<Line<'static>> = Vec::with_capacity(inner.height as usize);

    if filtered.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (没有匹配的配置项)",
            Style::default().fg(theme.text_dim),
        )));
        while lines.len() < list_h {
            lines.push(Line::default());
        }
    } else {
        let sel = selected.min(filtered.len() - 1);
        let start = sel
            .saturating_sub(list_h / 2)
            .min(filtered.len().saturating_sub(list_h));
        let end = (start + list_h).min(filtered.len());

        for (offset, &item_idx) in filtered[start..end].iter().enumerate() {
            let abs = start + offset;
            let is_sel = abs == sel;
            let item = &items[item_idx];
            let row_rect = Rect::new(inner.x, inner.y + offset as u16, inner.width, 1);
            state.item_rects.push((abs, row_rect));

            let bg = if is_sel {
                theme.card_bg_alt
            } else {
                theme.panel_bg
            };
            let ptr = if is_sel { "▸ " } else { "  " };
            let cat_tag = format!("[{}] ", item.category);
            let val_str = item.display_value();
            let val_style = match &item.kind {
                SettingItemKind::Toggle { value: true } => Style::default()
                    .fg(theme.success)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
                SettingItemKind::Toggle { value: false } => {
                    Style::default().fg(theme.text_dim).bg(bg)
                }
                SettingItemKind::Choice { .. } => Style::default()
                    .fg(theme.accent)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
                SettingItemKind::Model { provider, model } => {
                    let is_default = provider.as_deref().is_none_or(|p| p.trim().is_empty())
                        && model.as_deref().is_none_or(|m| m.trim().is_empty());
                    if is_default {
                        Style::default().fg(theme.text_muted).bg(bg)
                    } else {
                        Style::default()
                            .fg(theme.accent)
                            .bg(bg)
                            .add_modifier(Modifier::BOLD)
                    }
                }
            };

            let left_w = 2 + display_width(&cat_tag) + display_width(&item.label);
            let right_w = display_width(&val_str) + 1;
            let pad = (inner.width as usize)
                .saturating_sub(left_w + right_w)
                .max(2);

            lines.push(Line::from(vec![
                Span::styled(
                    ptr,
                    Style::default()
                        .fg(if is_sel { theme.accent } else { theme.text_dim })
                        .bg(bg),
                ),
                Span::styled(cat_tag, Style::default().fg(theme.info).bg(bg)),
                Span::styled(
                    item.label.clone(),
                    Style::default()
                        .fg(if is_sel { theme.accent } else { theme.text })
                        .bg(bg)
                        .add_modifier(if is_sel {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(" ".repeat(pad), Style::default().bg(bg)),
                Span::styled(val_str, val_style),
                Span::styled(" ", Style::default().bg(bg)),
            ]));
        }
        while lines.len() < list_h {
            lines.push(Line::default());
        }
    }

    lines.push(Line::from(Span::styled(
        "─".repeat(inner.width as usize),
        Style::default().fg(theme.border),
    )));
    let desc = filtered
        .get(selected.min(filtered.len().saturating_sub(1)))
        .map(|&idx| items[idx].description.as_str())
        .unwrap_or("输入关键词快速搜索配置项");
    lines.push(Line::from(Span::styled(
        format!(" {desc}"),
        Style::default().fg(theme.text_muted),
    )));
    lines.push(Line::from(Span::styled(
        format!(" 自动即时生效并保存至: {save_path_hint}"),
        Style::default().fg(theme.text_dim),
    )));

    frame.render_widget(Paragraph::new(lines), inner);
}

pub fn render_overlay(
    frame: &mut Frame<'_>,
    full_area: Rect,
    input_area: Rect,
    state: &mut OverlayState,
    skill_names: &[String],
    theme: &Theme,
) {
    state.last_box = None;
    state.item_rects.clear();
    let Some(active) = &state.active else {
        return;
    };

    match active {
        OverlayKind::SlashComplete {
            query, selected, ..
        } => {
            let entries = filter_slash(skill_names, query);
            let rows: Vec<(String, String)> = entries
                .iter()
                .map(|e| (format!("/{}", e.command), e.description.clone()))
                .collect();
            render_anchored_list(
                frame,
                full_area,
                input_area,
                " 命令补全 (Tab 填入 · Enter 执行) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::FileMention {
            query,
            matches,
            selected,
            ..
        } => {
            let max_w = full_area.width.clamp(28, 104);
            let left_budget = (max_w as usize).saturating_sub(6).max(12);
            let rows: Vec<(String, String)> = matches
                .iter()
                .map(|c| {
                    let display = clip_left_path(&format!("@{}", c.path), left_budget);
                    (display, String::new())
                })
                .collect();
            let title = format!(" 文件/目录引用: @{query} (Tab/Enter 插入 · Esc 隐藏) ");
            render_anchored_list_sized(
                frame, full_area, input_area, &title, &rows, *selected, max_w, 10, theme, state,
            );
        }
        OverlayKind::ModelPicker {
            items,
            query,
            selected,
        } => {
            let filtered: Vec<(String, String)> = items
                .iter()
                .filter(|it| {
                    query.is_empty()
                        || it
                            .label
                            .to_ascii_lowercase()
                            .contains(&query.to_ascii_lowercase())
                })
                .map(|it| (it.label.clone(), it.detail.clone()))
                .collect();
            let title = format!(" 选择模型 [搜索: {query}] ");
            render_centered_list(frame, full_area, &title, &filtered, *selected, theme, state);
        }
        OverlayKind::EffortPicker { selected } => {
            let rows: Vec<(String, String)> = all_efforts()
                .iter()
                .map(|e| {
                    let desc = match e {
                        mizar_llm_types::ReasoningEffort::Off => "关闭显式思考",
                        mizar_llm_types::ReasoningEffort::Minimal => "极少推理 token",
                        mizar_llm_types::ReasoningEffort::Low => "轻量快速推理",
                        mizar_llm_types::ReasoningEffort::Medium => "均衡模式",
                        mizar_llm_types::ReasoningEffort::High => "深度推理",
                        mizar_llm_types::ReasoningEffort::XHigh => "超长推理预算",
                        mizar_llm_types::ReasoningEffort::Max => "最大推理深度",
                    };
                    (format!("{e:?}").to_lowercase(), desc.to_string())
                })
                .collect();
            render_centered_list(
                frame,
                full_area,
                " 选择思考强度 (Reasoning Effort) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::ThemePicker { selected, .. } => {
            let rows: Vec<(String, String)> = ThemePreset::all()
                .iter()
                .map(|p| {
                    let desc = match p {
                        ThemePreset::Auto => "按 OSC 11 自动切换深浅雾蓝",
                        ThemePreset::MistDark => "雾蓝深色 (默认)",
                        ThemePreset::MistLight => "雾蓝浅色",
                        ThemePreset::SakuraDark => "樱粉深色",
                        ThemePreset::System => "OKHSL 终端色板自动拟合",
                        ThemePreset::Ansi16 => "16 色高兼容回退",
                    };
                    (p.name().to_string(), desc.to_string())
                })
                .collect();
            render_centered_list(
                frame,
                full_area,
                " 切换主题 (上下即时预览 · Enter 确认) ",
                &rows,
                *selected,
                theme,
                state,
            );
        }
        OverlayKind::HistorySearch {
            query,
            matches,
            selected,
        } => {
            let rows: Vec<(String, String)> = matches
                .iter()
                .map(|m| (m.replace('\n', " ↵ "), String::new()))
                .collect();
            let title = format!(" 历史模糊搜索 (Ctrl+R): {query} ");
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::SkillPicker {
            items,
            query,
            selected,
        } => {
            let rows: Vec<(String, String)> = items
                .iter()
                .filter(|s| {
                    query.is_empty()
                        || s.name
                            .to_ascii_lowercase()
                            .contains(&query.to_ascii_lowercase())
                        || s.description
                            .to_ascii_lowercase()
                            .contains(&query.to_ascii_lowercase())
                })
                .map(|s| (format!("/{} [{}]", s.name, s.source), s.description.clone()))
                .collect();
            let title = format!(" 交互式技能选择器 [过滤: {query}] (Enter 填入) ");
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::SessionPicker {
            items,
            query,
            selected,
        } => {
            let items_cloned = items.clone();
            let query_cloned = query.clone();
            let sel = *selected;
            render_session_picker(
                frame,
                full_area,
                &items_cloned,
                &query_cloned,
                sel,
                theme,
                state,
            );
        }
        OverlayKind::SessionTree {
            entries,
            selected,
            pan_x,
        } => {
            let rows: Vec<(String, String)> = entries
                .iter()
                .map(|e| {
                    let branch = if e.depth == 0 {
                        format!("#{} {}", e.seq, e.kind_tag)
                    } else {
                        format!("  └─ #{} {}", e.seq, e.kind_tag)
                    };
                    let shifted: String = e.summary.chars().skip(*pan_x).collect();
                    let fork_hint = if e.can_fork { " [f 分叉]" } else { "" };
                    (branch, format!("{shifted}{fork_hint}"))
                })
                .collect();
            let title = format!(
                " 会话时间线与分支树 (/tree · ↑↓选行 · ←→平移 x={} · Enter/f 分叉) ",
                *pan_x
            );
            render_centered_list(frame, full_area, &title, &rows, *selected, theme, state);
        }
        OverlayKind::HelpModal { scroll } => {
            render_help_modal(frame, full_area, *scroll, theme, state);
        }
        OverlayKind::SettingsModal {
            items,
            query,
            selected,
            save_path_hint,
        } => {
            let items_cloned = items.clone();
            let query_cloned = query.clone();
            let sel = *selected;
            let save_hint = save_path_hint.clone();
            render_settings_modal(
                frame,
                full_area,
                &items_cloned,
                &query_cloned,
                sel,
                &save_hint,
                theme,
                state,
            );
        }
        OverlayKind::Approval(ap) => {
            let tool_name = ap.tool_name.clone();
            let args_preview = ap.args_preview.clone();
            let editing_reason = ap.editing_reason;
            let deny_reason = ap.deny_reason.clone();
            render_approval_modal(
                frame,
                full_area,
                &tool_name,
                &args_preview,
                editing_reason,
                &deny_reason,
                theme,
                state,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_anchored_list(
    frame: &mut Frame<'_>,
    full_area: Rect,
    anchor_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.clamp(24, 72);
    render_anchored_list_sized(
        frame,
        full_area,
        anchor_area,
        title,
        rows,
        selected,
        box_w,
        8,
        theme,
        state,
    );
}

#[allow(clippy::too_many_arguments)]
fn render_anchored_list_sized(
    frame: &mut Frame<'_>,
    full_area: Rect,
    anchor_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    box_w: u16,
    max_rows: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_h = (rows.len().clamp(1, max_rows) + 2) as u16;
    let y = anchor_area.y.saturating_sub(box_h);
    let x = anchor_area.x.min(full_area.width.saturating_sub(box_w));
    let rect = Rect::new(x, y, box_w, box_h.min(anchor_area.y.max(3)));
    draw_list_box(frame, rect, title, rows, selected, theme, state);
}

fn clip_left_path(s: &str, max_cols: usize) -> String {
    if display_width(s) <= max_cols {
        return s.to_string();
    }
    let mut kept = Vec::new();
    let mut w = 1usize; // 预留前缀 '…'
    for ch in s.chars().rev() {
        let cw = ch.width().unwrap_or(0);
        if w + cw > max_cols {
            break;
        }
        kept.push(ch);
        w += cw;
    }
    kept.reverse();
    let tail: String = kept.into_iter().collect();
    format!("…{tail}")
}

fn render_session_picker(
    frame: &mut Frame<'_>,
    full_area: Rect,
    items: &[SessionPickerItem],
    query: &str,
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let filtered = filter_session_picker_indices(items, query);
    let box_w = full_area.width.saturating_sub(4).clamp(50, 106);
    let box_h = (filtered.len().clamp(1, 14) + 2) as u16;
    let box_h = box_h.min(full_area.height.saturating_sub(2).max(4));
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);

    let inner_w = box_w.saturating_sub(2) as usize;
    let cwd_col_w = filtered
        .iter()
        .filter_map(|&idx| items.get(idx))
        .map(|it| display_width(&it.cwd_tail))
        .max()
        .unwrap_or(6)
        .clamp(6, 20);
    let title_col_w = inner_w.saturating_sub(4 + cwd_col_w + 22).max(16);

    let rows: Vec<(String, String)> = filtered
        .iter()
        .filter_map(|&idx| items.get(idx))
        .map(|it| format_session_picker_row(it, title_col_w, cwd_col_w))
        .collect();

    let title = if query.is_empty() {
        " 恢复会话 (/resume · ↑↓ 选择 · 输入过滤 · Enter 确认 · Esc 取消) ".to_string()
    } else {
        format!(" 恢复会话 [过滤: {query}] (Enter 确认 · Esc 清空) ")
    };

    draw_list_box(frame, rect, &title, &rows, selected, theme, state);
}

fn render_centered_list(
    frame: &mut Frame<'_>,
    full_area: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(32, 74);
    let box_h = (rows.len().clamp(1, 12) + 2) as u16;
    let box_h = box_h.min(full_area.height.saturating_sub(2).max(4));
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    draw_list_box(
        frame,
        Rect::new(x, y, box_w, box_h),
        title,
        rows,
        selected,
        theme,
        state,
    );
}

fn draw_list_box(
    frame: &mut Frame<'_>,
    rect: Rect,
    title: &str,
    rows: &[(String, String)],
    selected: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border_focus))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    if rows.is_empty() || inner.height == 0 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "  (无匹配项)",
                Style::default().fg(theme.text_dim),
            )),
            inner,
        );
        return;
    }

    let sel = selected.min(rows.len() - 1);
    let cap = inner.height as usize;
    let start = sel
        .saturating_sub(cap / 2)
        .min(rows.len().saturating_sub(cap));
    let end = (start + cap).min(rows.len());

    let mut lines = Vec::new();
    for (offset, (left, right)) in rows[start..end].iter().enumerate() {
        let abs = start + offset;
        let is_sel = abs == sel;
        let row_rect = Rect::new(inner.x, inner.y + offset as u16, inner.width, 1);
        state.item_rects.push((abs, row_rect));

        let bg = if is_sel {
            theme.card_bg_alt
        } else {
            theme.panel_bg
        };
        let ptr = if is_sel { "▸ " } else { "  " };
        let mut spans = vec![
            Span::styled(
                ptr,
                Style::default()
                    .fg(if is_sel { theme.accent } else { theme.text_dim })
                    .bg(bg),
            ),
            Span::styled(
                left.clone(),
                Style::default()
                    .fg(if is_sel { theme.accent } else { theme.text })
                    .bg(bg)
                    .add_modifier(if is_sel {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ];
        if !right.is_empty() {
            let used = 2 + display_width(left) + 2;
            if inner.width as usize > used + 4 {
                spans.push(Span::styled(
                    format!("  {right}"),
                    Style::default().fg(theme.text_muted).bg(bg),
                ));
            }
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

#[allow(clippy::too_many_arguments)]
fn render_approval_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    tool_name: &str,
    args_preview: &str,
    editing_reason: bool,
    deny_reason: &str,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(40, 82);
    let box_h =
        if editing_reason { 11u16 } else { 10u16 }.min(full_area.height.saturating_sub(2).max(6));
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            " ⚠ 工具执行审批 (Approval Required) ",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.warning))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    let mut lines = vec![
        Line::from(vec![
            Span::styled("工具：", Style::default().fg(theme.text_muted)),
            Span::styled(
                tool_name.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            format!("参数：{args_preview}"),
            Style::default().fg(theme.text),
        )),
        Line::default(),
    ];

    if editing_reason {
        lines.push(Line::from(vec![
            Span::styled(
                "拒绝理由（Enter 提交 / Esc 返回）：",
                Style::default()
                    .fg(theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("{deny_reason}▏"), Style::default().fg(theme.text)),
        ]));
    } else {
        let btn0 = " [y / Enter] 允许本次 ";
        let btn1 = " [a] 始终允许 ";
        let btn2 = " [r] 拒绝附理由 ";
        let btn3 = " [n/Esc] 拒绝 ";
        lines.push(Line::from(vec![
            Span::styled(
                btn0,
                Style::default()
                    .fg(theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                btn1,
                Style::default().fg(theme.info).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                btn2,
                Style::default()
                    .fg(theme.warning)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                btn3,
                Style::default()
                    .fg(theme.error)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        if inner.height >= 4 {
            let row = inner.y + 3;
            let mut x_cursor = inner.x;
            for (idx, label) in [btn0, btn1, btn2, btn3].into_iter().enumerate() {
                let w = display_width(label) as u16;
                let max_end = inner.x + inner.width;
                if x_cursor < max_end {
                    let clipped_w = w.min(max_end - x_cursor);
                    if clipped_w > 0 {
                        state
                            .item_rects
                            .push((idx, Rect::new(x_cursor, row, clipped_w, 1)));
                    }
                }
                x_cursor = x_cursor.saturating_add(w + 1);
            }
        }
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_help_modal(
    frame: &mut Frame<'_>,
    full_area: Rect,
    scroll: usize,
    theme: &Theme,
    state: &mut OverlayState,
) {
    let box_w = full_area.width.saturating_sub(6).clamp(40, 80);
    let box_h = full_area.height.saturating_sub(4).clamp(10, 22);
    let x = full_area.x + (full_area.width.saturating_sub(box_w)) / 2;
    let y = full_area.y + (full_area.height.saturating_sub(box_h)) / 2;
    let rect = Rect::new(x, y, box_w, box_h);
    frame.render_widget(Clear, rect);
    let block = Block::default()
        .title(Span::styled(
            " Mizar 快捷键与命令指南 (Esc 关闭) ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border_focus))
        .style(Style::default().bg(theme.panel_bg));
    let inner = block.inner(rect);
    frame.render_widget(block, rect);
    state.last_box = Some(rect);

    let items = [
        (
            "Enter (空闲 / 运行中)",
            "空闲直接发送（草稿空时发队首）；运行中自动加入 Follow-up 队列",
        ),
        (
            "↑ / ⏎ / Del · 队列选择器",
            "草稿空时 ↑ 打开队列：⏎ 撤回编辑 · Ctrl+⏎ 立即发送 · Del 丢弃 · Esc 关闭",
        ),
        (
            "Alt+↑",
            "直接把最后一条排队消息撤回输入框编辑（草稿非空则与之交换）",
        ),
        (
            "!cmd / !!cmd",
            "直接执行本地 Bash 命令（! 载入上下文 / !! 仅本地不入上下文）",
        ),
        ("Ctrl+Enter", "立即发送（运行中打断当前回合并立即插队发送）"),
        (
            "Shift+Enter / Ctrl+J",
            "插入换行；粘贴超长文本自动折叠为原子块",
        ),
        ("Tab", "循环切换思考等级（补全浮层内为补全选中项）"),
        ("Ctrl+L", "打开交互式模型选择器（等同 /model）"),
        ("Esc", "取消选区或浮层 -> 暂停当前生成 -> 清空输入"),
        (
            "Ctrl+C",
            "复制选区 / 关闭弹窗 / 单击清空输入 / 中断或双击退出",
        ),
        (
            "鼠标划词 / 双击 / 三击",
            "应用级分栏选区（自动拼接软换行），松手自动复制；F2 切换鼠标追踪",
        ),
        (
            "Ctrl+B / Ctrl+O",
            "Ctrl+B 开关侧栏；Ctrl+O 展开/折叠思考与完整工具输出",
        ),
        (
            "Ctrl+T / /trace",
            "打开全屏三泳道轨迹分析器（WaveBand + Timeline + Hotspots）",
        ),
        (
            "Ctrl+V",
            "粘贴剪贴板图片、文件路径或文本（SSH 下支持 OSC 5522）",
        ),
        (
            "Ctrl+G / Ctrl+E",
            "Ctrl+G 调用 $EDITOR 编辑草稿；Ctrl+E 展开折叠粘贴块",
        ),
        (
            "Ctrl+R / Ctrl+Z / Ctrl+Y",
            "历史模糊搜索 / 词级撤销 / KillRing 粘回",
        ),
        (
            "/settings /compact",
            "打开交互式设置面板（即时生效并持久化）/ 手动压缩上下文",
        ),
        (
            "/session /stats",
            "查看当前会话完整 ID、存储与事件账目 / 会话运行统计",
        ),
        (
            "/tree /skills /view",
            "会话分支树回溯 / 交互技能选择 / Fullscreen↔Inline 视图热切",
        ),
        (
            "/model /effort /theme",
            "交互切换模型、思考深度与 Oklab 自适应主题",
        ),
        (
            "/approval [auto|ask]",
            "开启或关闭工具调用交互式审批确认弹窗（支持拒绝附理由）",
        ),
    ];
    let cap = inner.height as usize;
    let start = scroll.min(items.len().saturating_sub(cap));
    let end = (start + cap).min(items.len());
    let lines: Vec<Line<'static>> = items[start..end]
        .iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(
                    format!("{k:<24}"),
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled((*v).to_string(), Style::default().fg(theme.text)),
            ])
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}
