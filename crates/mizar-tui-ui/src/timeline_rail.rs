//! 2 列语义化对话时间轴导轨（借鉴 `dsh-TUI` `timeline-rail.ts`）：
//! 以用户回合为单位取代传统比例滚动条，同一份冻结快照同时驱动绘制、顶部吸顶提示与鼠标点击命中。

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};

use crate::theme::Theme;

pub const RAIL_WIDTH: u16 = 2;
pub const RAIL_MIN_TERMINAL_WIDTH: u16 = 60;
pub const RAIL_MIN_TURNS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineTurn {
    pub item_index: usize,
    pub top_line: usize,
    pub preview: String,
}

#[derive(Debug, Clone, Default)]
pub struct TimelineSnapshot {
    pub turns: Vec<TimelineTurn>,
    pub active_idx: Option<usize>,
    pub pinned_idx: Option<usize>,
    pub up_idx: Option<usize>,
    pub down_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RailGeometry {
    pub area: Rect,
    pub window_start: usize,
    pub window_end: usize,
    pub up_row: u16,
    pub tick_top: u16,
    pub down_row: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailHit {
    Up,
    Down,
    Tick(usize),
}

pub fn build_timeline_snapshot(
    turns: Vec<TimelineTurn>,
    scroll_top: usize,
    max_scroll: usize,
) -> TimelineSnapshot {
    if turns.is_empty() {
        return TimelineSnapshot::default();
    }

    let mut active_idx = 0usize;
    for (idx, t) in turns.iter().enumerate() {
        if t.top_line <= scroll_top {
            active_idx = idx;
        } else {
            break;
        }
    }

    // 当激活轮次的首行已经滚出视口上方时，启用顶部吸顶轮次提示
    let pinned_idx = if turns[active_idx].top_line < scroll_top {
        Some(active_idx)
    } else {
        None
    };

    let up_idx = if let Some(prev) = active_idx.checked_sub(1) {
        Some(prev)
    } else if scroll_top > turns[0].top_line {
        Some(0)
    } else {
        None
    };

    let down_idx = turns
        .iter()
        .position(|t| t.top_line > scroll_top && t.top_line <= max_scroll);

    TimelineSnapshot {
        turns,
        active_idx: Some(active_idx),
        pinned_idx,
        up_idx,
        down_idx,
    }
}

pub fn compute_rail_geometry(
    area: Rect,
    snap: &TimelineSnapshot,
    at_bottom: bool,
) -> Option<RailGeometry> {
    let count = snap.turns.len();
    if count < RAIL_MIN_TURNS || area.width < RAIL_WIDTH || area.height < 3 {
        return None;
    }
    let max_ticks = (area.height as usize).saturating_sub(2);
    if max_ticks == 0 {
        return None;
    }

    let mut start = 0usize;
    if count > max_ticks {
        let tail_start = count - max_ticks;
        let anchor = snap.active_idx.unwrap_or(count - 1);
        start = if at_bottom {
            anchor.min(tail_start)
        } else {
            anchor.saturating_sub(max_ticks / 2).min(tail_start)
        };
    }
    let shown = count.min(max_ticks);
    let block_top = area.y + ((area.height as usize - (shown + 2)) / 2) as u16;
    Some(RailGeometry {
        area,
        window_start: start,
        window_end: start + shown,
        up_row: block_top,
        tick_top: block_top + 1,
        down_row: block_top + 1 + shown as u16,
    })
}

pub fn hit_test_rail(geo: &RailGeometry, col: u16, row: u16) -> Option<RailHit> {
    if col < geo.area.x || col >= geo.area.x + geo.area.width {
        return None;
    }
    if row == geo.up_row {
        return Some(RailHit::Up);
    }
    if row == geo.down_row {
        return Some(RailHit::Down);
    }
    if row >= geo.tick_top {
        let rel = (row - geo.tick_top) as usize;
        if rel < geo.window_end - geo.window_start {
            return Some(RailHit::Tick(geo.window_start + rel));
        }
    }
    None
}

pub fn render_timeline_rail(
    buf: &mut Buffer,
    geo: &RailGeometry,
    snap: &TimelineSnapshot,
    hovered_row: Option<u16>,
    theme: &Theme,
) {
    let x = geo.area.x;
    let dim_style = Style::default().fg(theme.text_dim);
    let active_style = Style::default()
        .fg(theme.accent)
        .add_modifier(Modifier::BOLD);
    let hover_style = Style::default()
        .fg(theme.link)
        .add_modifier(Modifier::BOLD);

    let up_style = if snap.up_idx.is_none() {
        dim_style
    } else if hovered_row == Some(geo.up_row) {
        hover_style
    } else {
        Style::default().fg(theme.text_muted)
    };
    buf.set_string(x, geo.up_row, " ▲", up_style);

    for idx in geo.window_start..geo.window_end {
        let y = geo.tick_top + (idx - geo.window_start) as u16;
        let is_active = snap.active_idx == Some(idx);
        let is_hover = hovered_row == Some(y);
        let (glyph, style) = if is_active {
            (" ●", active_style)
        } else if is_hover {
            (" ○", hover_style)
        } else {
            (" ·", dim_style)
        };
        buf.set_string(x, y, glyph, style);
    }

    let down_style = if snap.down_idx.is_none() {
        dim_style
    } else if hovered_row == Some(geo.down_row) {
        hover_style
    } else {
        Style::default().fg(theme.text_muted)
    };
    buf.set_string(x, geo.down_row, " ▼", down_style);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rail_geometry_windows_and_hits_accurately() {
        let turns: Vec<TimelineTurn> = (0..10)
            .map(|i| TimelineTurn {
                item_index: i * 2,
                top_line: i * 20,
                preview: format!("turn {i}"),
            })
            .collect();
        let snap = build_timeline_snapshot(turns, 45, 180);
        assert_eq!(snap.active_idx, Some(2));
        assert_eq!(snap.pinned_idx, Some(2));
        assert_eq!(snap.up_idx, Some(1));
        assert_eq!(snap.down_idx, Some(3));

        let geo = compute_rail_geometry(Rect::new(78, 0, 2, 7), &snap, false).unwrap();
        assert_eq!(hit_test_rail(&geo, 78, geo.up_row), Some(RailHit::Up));
        assert_eq!(hit_test_rail(&geo, 79, geo.down_row), Some(RailHit::Down));
        assert_eq!(
            hit_test_rail(&geo, 78, geo.tick_top),
            Some(RailHit::Tick(geo.window_start))
        );
    }
}
