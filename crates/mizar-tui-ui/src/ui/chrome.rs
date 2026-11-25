//! 输入框、排队待发送条与底部状态栏渲染。

use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders};

use super::{BRAILLE_SPINNER, color_to_rgb8, effort_color, format_effort_label};
use crate::app::{
    AppState, ApprovalMode, BashInputMode, UiTurnEndReason, format_tokens, truncate_display,
};
use crate::editor::display_width;
use crate::selection::SelectionPane;
use crate::terminal::ScreenMode;
use crate::theme::{Theme, mix_oklab};

const IGNITION_DURATION_MS: u128 = 1200;

pub(super) fn render_queued_strip(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let theme = &app.theme;
    let items = app.queue.items();
    let count = items.len();
    let selected = app.queue.selected_id();
    let one_line = |text: &str| text.replace('\n', " ");

    let mut lines: Vec<Line<'_>> = Vec::with_capacity(area.height as usize);
    if selected.is_none() {
        let hint = truncate_display(&one_line(&items[0].text), avail_of(area, 16));
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
        let sel_index = items
            .iter()
            .position(|item| Some(item.id) == selected)
            .unwrap_or(0);
        let start = sel_index.saturating_sub(visible.saturating_sub(1));
        let end = (start + visible).min(count);
        for (offset, item) in items[start..end].iter().enumerate() {
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

pub(super) fn render_input_box(
    frame: &mut Frame<'_>,
    area: Rect,
    inner_w: u16,
    app: &mut AppState,
) {
    if area.width < 3 || area.height < 3 {
        return;
    }
    let bash_mode = app.bash_mode();
    let effort_col = effort_color(&app.theme, app.reasoning_effort);
    let mut base_border_color = if let Some(mode) = bash_mode {
        match mode {
            BashInputMode::Context => app.theme.success,
            BashInputMode::NoContext => app.theme.text_muted,
        }
    } else if app.is_running_activity() {
        app.theme.border_busy
    } else if app.reasoning_effort.is_some() {
        effort_col
    } else {
        app.theme.border_focus
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
            if app.theme.truecolor {
                let wave = (1.0 - (p * 2.0 - 1.0).abs()).clamp(0.0, 1.0);
                if let (Some(a), Some(b)) =
                    (color_to_rgb8(effort_col), color_to_rgb8(app.theme.accent))
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
                .fg(app.theme.success)
                .add_modifier(Modifier::BOLD),
        ),
        Some(BashInputMode::NoContext) => (
            " Bash (!! · 不载入上下文 · Enter 执行 · Esc 退出) ".to_string(),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.is_busy() => (
            " 输入 (Enter 排队 · Ctrl+Enter 立即发送 · Tab 思考 · Esc 暂停) ".to_string(),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.is_bash_running() => (
            " 输入 (Bash 命令执行中 · Esc 取消) ".to_string(),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.is_compacting() => (
            " 输入 (上下文压缩中 · Esc 取消) ".to_string(),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.input.has_collapsed_pastes() && app.input.has_images() => (
            format!(
                " 输入 (Enter 发送 · 已附 {} 张图片 · 含折叠粘贴块，Ctrl+E 展开) ",
                app.input.image_count()
            ),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.input.has_collapsed_pastes() => (
            " 输入 (Enter / Ctrl+Enter 发送 · Tab 思考 · 含折叠粘贴块，Ctrl+E 展开) ".to_string(),
            Style::default().fg(app.theme.text_muted),
        ),
        None if app.input.has_images() => (
            format!(
                " 输入 (Enter / Ctrl+Enter 发送 · 已附 {} 张图片 · Ctrl+V 粘贴 · ? 帮助) ",
                app.input.image_count()
            ),
            Style::default().fg(app.theme.text_muted),
        ),
        None => (
            " 输入 (Enter / Ctrl+Enter 发送 · Tab 思考 · Ctrl+L 模型 · ? 帮助) ".to_string(),
            Style::default().fg(app.theme.text_muted),
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
        && app.theme.truecolor
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
                buf[(x, area.y)].set_style(
                    Style::default()
                        .fg(app.theme.text)
                        .add_modifier(Modifier::BOLD),
                );
                let by = area.y + area.height - 1;
                let mirror_x = area.x + area.width - 1 - pos;
                buf[(mirror_x, by)].set_style(
                    Style::default()
                        .fg(app.theme.text)
                        .add_modifier(Modifier::BOLD),
                );
            }
        }
    }

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let is_overlay_modal = app.overlay.is_modal();
    let is_empty = app.input.is_empty();
    let theme = &app.theme;
    let selection_geom = &mut app.selection_geom;
    let layout = app.input.layout_cached(inner_w);
    let scroll_y = layout
        .cursor_row
        .saturating_sub(inner.height.saturating_sub(1));
    let buf = frame.buffer_mut();

    if is_empty {
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
            selection_geom.push_row(SelectionPane::Input, y, inner.x, col_end, false);
        }
    }

    // 居中模态浮层打开时不把物理光标抢回到输入框
    if !is_overlay_modal {
        let rel_row = layout.cursor_row.saturating_sub(scroll_y);
        if rel_row < inner.height {
            let cursor_x = inner.x + layout.cursor_col.min(inner.width.saturating_sub(1));
            let cursor_y = inner.y + rel_row;
            frame.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

pub(super) fn render_inline_pills(
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

pub(super) fn render_status_bar(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let theme = app.theme.clone();
    let is_busy = app.is_busy();
    let is_bash = app.is_bash_running();
    let is_compacting = app.is_compacting();
    let spinner = BRAILLE_SPINNER[app.spinner_tick % BRAILLE_SPINNER.len()];
    let (status_text, status_color) = if is_compacting {
        let detail = app.compaction_status().unwrap_or("上下文压缩中…");
        (format!("{spinner} {detail}"), theme.info)
    } else if is_busy {
        (format!("{spinner} 运行中"), theme.warning)
    } else if is_bash {
        (format!("{spinner} Bash"), theme.warning)
    } else {
        match &app.last_turn_end_reason {
            None | Some(UiTurnEndReason::Completed) => ("● 就绪".to_string(), theme.success),
            Some(UiTurnEndReason::Truncated) => ("⚠ 输出截断".to_string(), theme.warning),
            Some(UiTurnEndReason::MaxSteps) => ("⚠ 步数上限".to_string(), theme.warning),
            Some(UiTurnEndReason::Cancelled) => ("● 已中断".to_string(), theme.text_muted),
            Some(UiTurnEndReason::Error { .. }) => ("✗ 请求失败".to_string(), theme.error),
        }
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
    } else {
        if let Some(title) = app
            .session_title
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            && avail >= 64
        {
            spans.push(Span::styled(" │ ", Style::default().fg(theme.border)));
            spans.push(Span::styled(
                truncate_display(title, 28),
                Style::default().fg(theme.text_muted),
            ));
        }
        let hint_min_avail = if app
            .session_title
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
        {
            120
        } else {
            96
        };
        if avail >= hint_min_avail {
            let toggle_hint = if app.verbose_expanded {
                "Ctrl+O 收起"
            } else {
                "Ctrl+O 展开"
            };
            let mut parts = vec![toggle_hint];
            if app.features.side_panel {
                parts.push("Ctrl+B 侧栏");
            }
            if app.features.trajectory_view {
                parts.push("Ctrl+T 轨迹");
            }
            parts.push("? 帮助");
            spans.push(Span::styled(
                format!(" │ {}", parts.join(" · ")),
                Style::default().fg(theme.text_dim),
            ));
        }
    }

    frame
        .buffer_mut()
        .set_line(area.x, area.y, &Line::from(spans), area.width);
}
