//! TUI 布局与渲染引擎主入口：双栏工作台、全屏轨迹切换、浮层与选区叠绘调度。

use mizar_llm_types::ReasoningEffort;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};

use crate::app::AppState;
use crate::overlay::{OverlayKind, render_overlay};
use crate::selection::paint_selection_overlay;
use crate::side_panel::{render_divider_column, render_side_panel};
use crate::theme::{Rgb8, Theme};
use crate::timeline_rail::TimelineSnapshot;
use crate::trajectory_scene::render_trajectory_scene;

mod chrome;
mod tool_card;
mod transcript;
mod welcome;

pub use transcript::{build_item_lines_for_scrollback, build_item_lines_for_scrollback_at};
pub use welcome::build_welcome_lines_for_scrollback;

pub(crate) const BRAILLE_SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// 复用上一帧 `Buffer` 的底层 `Vec<Cell>` 堆内存，仅在区域尺寸变更时重新分配。
fn snapshot_buffer(dst: &mut Option<Buffer>, src: &Buffer) {
    match dst {
        Some(prev) if prev.area == src.area => {
            prev.content.clone_from(&src.content);
        }
        _ => {
            *dst = Some(src.clone());
        }
    }
}

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
    if app.features.trajectory_view && app.trajectory_scene.open {
        render_trajectory_scene(frame, area, &mut app.trajectory_scene, &app.theme);
        snapshot_buffer(&mut app.last_buffer, frame.buffer_mut());
        return;
    }

    // 2. 双栏 / 单栏工作台分配
    let chat_area = if app.features.side_panel
        && let Some((chat_w, divider_x, panel_w)) = app.side_panel.split_columns(area.width)
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
    let input_line_count = app.input.layout_cached(input_inner_w).lines.len();

    let status_h = 1u16;
    let avail_h = chat_area.height.saturating_sub(status_h);
    // 队列条：默认 1 行摘要；选择器打开时逐条展开（最多 3 行），不再因终端矮而直接隐藏。
    let queued_h = if app.queue.is_empty() {
        0
    } else if app.queue.selected_id().is_some() {
        (app.queue.len().min(3) as u16) + 1
    } else {
        1
    }
    .min(avail_h.saturating_sub(3));
    let max_input_h = avail_h.saturating_sub(queued_h + 1).max(3).min(avail_h);
    let input_h = ((input_line_count as u16) + 2).clamp(3, 8).min(max_input_h);
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
        transcript::render_transcript(frame, transcript_area, app);
    }

    // 5. 渲染排队待发送条
    if queued_area.height > 0 {
        chrome::render_queued_strip(frame, queued_area, app);
    }

    // 6. 渲染输入区（含 Effort 点火扫描光效与折叠粘贴块高亮）
    chrome::render_input_box(frame, input_area, input_inner_w, app);

    // 7. 渲染底部状态栏
    chrome::render_status_bar(frame, status_area, app);

    // 8. 渲染浮层栈（补全弹窗 / Model / Effort / Theme / 历史 / 审批 / 帮助）
    if matches!(app.overlay.active, Some(OverlayKind::FileMention { .. }))
        && app.file_cache.consume_generation_update()
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
    if app.features.selection {
        paint_selection_overlay(frame.buffer_mut(), &app.selection, &app.theme);
    }
    snapshot_buffer(&mut app.last_buffer, frame.buffer_mut());
}

pub(super) fn format_effort_label(effort: Option<ReasoningEffort>) -> &'static str {
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

pub(super) fn effort_color(theme: &Theme, effort: Option<ReasoningEffort>) -> Color {
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

pub(super) fn color_to_rgb8(c: Color) -> Option<Rgb8> {
    match c {
        Color::Rgb(r, g, b) => Some(Rgb8::new(r, g, b)),
        _ => None,
    }
}
