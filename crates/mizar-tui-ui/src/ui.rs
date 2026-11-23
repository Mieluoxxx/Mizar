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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::app::{UiEvent, UiStepOutcome, UiTurnEndReason};
    use crate::editor::display_width;

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
    fn status_bar_distinguishes_completed_truncated_max_steps_cancelled_and_error() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-status-bar",
            Vec::new(),
            None,
            None,
            None,
        );

        let initial = render_buffer_text(&mut app, 120, 20);
        assert!(initial.contains("● 就绪"), "initial screen:\n{initial}");

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Completed,
        });
        let completed = render_buffer_text(&mut app, 120, 20);
        assert!(
            completed.contains("● 就绪"),
            "completed screen:\n{completed}"
        );

        app.apply_ui_event(UiEvent::StepEnd {
            step: 1,
            ts_ms: 100,
            outcome: UiStepOutcome::MaxTokens,
            usage: None,
        });
        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Truncated,
        });
        let truncated = render_buffer_text(&mut app, 120, 20);
        assert!(
            truncated.contains("⚠ 输出截断"),
            "truncated screen:\n{truncated}"
        );
        assert!(
            !truncated.contains("● 就绪"),
            "truncated should not show 就绪:\n{truncated}"
        );

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::MaxSteps,
        });
        let max_steps = render_buffer_text(&mut app, 120, 20);
        assert!(
            max_steps.contains("⚠ 步数上限"),
            "max_steps screen:\n{max_steps}"
        );
        assert!(
            !max_steps.contains("● 就绪"),
            "max_steps should not show 就绪:\n{max_steps}"
        );

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Cancelled,
        });
        let cancelled = render_buffer_text(&mut app, 120, 20);
        assert!(
            cancelled.contains("● 已中断"),
            "cancelled screen:\n{cancelled}"
        );
        assert!(
            !cancelled.contains("● 就绪"),
            "cancelled should not show 就绪:\n{cancelled}"
        );

        app.apply_ui_event(UiEvent::TurnEnd {
            reason: UiTurnEndReason::Error {
                message: "boom".into(),
            },
        });
        let errored = render_buffer_text(&mut app, 120, 20);
        assert!(errored.contains("✗ 请求失败"), "error screen:\n{errored}");
        assert!(
            !errored.contains("● 就绪"),
            "error should not show 就绪:\n{errored}"
        );

        app.apply_ui_event(UiEvent::TurnStart { turn: 2 });
        let busy = render_buffer_text(&mut app, 120, 20);
        assert!(busy.contains("运行中"), "busy screen:\n{busy}");
        assert!(
            !busy.contains("✗ 请求失败"),
            "busy should clear prior error:\n{busy}"
        );

        app.apply_ui_event(UiEvent::SubagentState {
            running_count: 2,
            notice: Some("后台子代理 agent-2 已启动".into()),
        });
        let with_sub = render_buffer_text(&mut app, 120, 20);
        assert!(
            with_sub.contains("[subagent:2]"),
            "status bar should show running subagent count:\n{with_sub}"
        );

        app.apply_ui_event(UiEvent::SubagentState {
            running_count: 0,
            notice: None,
        });
        let without_sub = render_buffer_text(&mut app, 120, 20);
        assert!(
            !without_sub.contains("[subagent:"),
            "status bar should hide subagent badge once settled:\n{without_sub}"
        );
    }

    #[test]
    fn transcript_render_cache_caps_lru_entries_and_retains_heights() {
        let mut app = AppState::new(PathBuf::from("."), "sess-lru", Vec::new(), None, None, None);
        for i in 0..140 {
            app.items.push(crate::app::ChatItem::User {
                text: format!("user prompt line #{i}"),
            });
            app.items.push(crate::app::ChatItem::Assistant {
                reasoning: String::new(),
                text: format!("assistant reply paragraph #{i}\n\n- bullet {i}"),
            });
        }

        let _ = render_buffer_text(&mut app, 100, 32);
        assert_eq!(app.item_cache.height_count(), app.items.len());
        assert!(
            app.item_cache.len() <= crate::app::MAX_CACHED_RENDER_ITEMS,
            "cache entries {} should be capped at {}",
            app.item_cache.len(),
            crate::app::MAX_CACHED_RENDER_ITEMS
        );

        // 向上滚回顶部后，只重新生成并置换顶部视口工作集，总量仍受 LRU 上限控制
        app.auto_scroll = false;
        app.scroll_from_top = 0;
        let top_screen = render_buffer_text(&mut app, 100, 32);
        assert!(
            app.item_cache.len() <= crate::app::MAX_CACHED_RENDER_ITEMS,
            "top scroll entries {} exceeded cap",
            app.item_cache.len()
        );
        assert!(top_screen.contains("user prompt line #0"));
    }

    #[test]
    fn resize_defers_offscreen_reflow_and_materializes_when_scrolled_into_view() {
        let mut app = AppState::new(
            PathBuf::from("."),
            "sess-lazy-resize",
            Vec::new(),
            None,
            None,
            None,
        );
        for i in 0..160 {
            app.items.push(crate::app::ChatItem::User {
                text: format!("prompt #{i} with enough words to wrap at 80 cols"),
            });
            app.items.push(crate::app::ChatItem::Assistant {
                reasoning: String::new(),
                text: format!(
                    "assistant paragraph #{i}: long explanation text that differs across widths"
                ),
            });
        }

        // 1. 首次在 120 列全量渲染
        let _ = render_buffer_text(&mut app, 120, 30);

        // 2. 缩窄至 80 列（auto_scroll = true）：离屏第 0 条应保留旧高度占位，不在新宽度下重排
        let _ = render_buffer_text(&mut app, 80, 30);
        let content_w80 = 80usize - crate::timeline_rail::RAIL_WIDTH as usize;
        let key_item0_w80 = crate::app::ItemRenderKey {
            width: content_w80,
            theme_id: app.theme.id,
            emphasis_enabled: app.emphasis.enabled,
            round_caps: app.emphasis.round_caps,
            expanded: app.verbose_expanded,
            status_code: 0,
            reasoning_len: 0,
            text_len: match &app.items[0] {
                crate::app::ChatItem::User { text } => text.len(),
                _ => unreachable!(),
            },
            preview_len: 0,
        };
        assert!(
            app.item_cache.get_height(0, &key_item0_w80).is_none(),
            "offscreen item 0 should not be eagerly reflowed at width 80 while bottom-pinned"
        );

        // 3. 滚回顶部后，第 0 条进入视口并惰性重排准确渲染
        app.auto_scroll = false;
        app.scroll_from_top = 0;
        let top_screen = render_buffer_text(&mut app, 80, 30);
        assert!(top_screen.contains("prompt #0"));
        assert!(app.item_cache.get_height(0, &key_item0_w80).is_some());
    }
}
