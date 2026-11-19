//! 全屏三泳道轨迹分析场景（直连 `mizar-trajectory`，借鉴 `dsh-TUI` `TrajectoryScene.tsx`）：
//!
//! - 顶部：WaveBand 全会话波形带（支持鼠标点击列跳转到最近事件）；
//! - 中部：`Timeline` 三泳道事件账本 与 `Hotspots` 工具耗时热点排行双视图；
//! - 底部：固定高度 Inspector，展示单步 TTFT、耗时、Token、参数/结果与异常告警。

use std::collections::BTreeMap;

use crossterm::event::{KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind};
use mizar_trajectory_core::{
    RecordKind, Severity, TimelineMode, TrajectoryRecord, TrajectoryReport,
};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::editor::display_width;
use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryTab {
    Timeline,
    Hotspots,
}

#[derive(Debug, Clone)]
pub struct HotspotEntry {
    pub name: String,
    pub calls: usize,
    pub errors: usize,
    pub total_ms: u64,
    pub first_record_idx: usize,
}

#[derive(Debug, Clone)]
pub struct TrajectorySceneState {
    pub open: bool,
    pub mode: TimelineMode,
    pub tab: TrajectoryTab,
    pub cursor: usize,
    pub hotspot_cursor: usize,
    pub query: String,
    pub search_active: bool,
    pub report: Option<TrajectoryReport>,
    pub wave_area: Option<Rect>,
    pub ledger_area: Option<Rect>,
    pub tab_timeline_area: Option<Rect>,
    pub tab_hotspot_area: Option<Rect>,
    pub mode_chip_area: Option<Rect>,
    pub close_area: Option<Rect>,
    pub ledger_scroll: usize,
}

impl Default for TrajectorySceneState {
    fn default() -> Self {
        Self {
            open: false,
            mode: TimelineMode::Duration,
            tab: TrajectoryTab::Timeline,
            cursor: 0,
            hotspot_cursor: 0,
            query: String::new(),
            search_active: false,
            report: None,
            wave_area: None,
            ledger_area: None,
            tab_timeline_area: None,
            tab_hotspot_area: None,
            mode_chip_area: None,
            close_area: None,
            ledger_scroll: 0,
        }
    }
}

impl TrajectorySceneState {
    pub fn filtered_indices(&self) -> Vec<usize> {
        let Some(report) = &self.report else {
            return Vec::new();
        };
        let q = self.query.trim().to_ascii_lowercase();
        report
            .snapshot
            .records
            .iter()
            .enumerate()
            .filter(|(_, rec)| {
                if q.is_empty() {
                    return true;
                }
                rec.label.to_ascii_lowercase().contains(&q)
                    || rec
                        .tool_name
                        .as_deref()
                        .is_some_and(|n| n.to_ascii_lowercase().contains(&q))
                    || rec
                        .input_detail
                        .as_deref()
                        .is_some_and(|s| s.to_ascii_lowercase().contains(&q))
                    || rec
                        .output_detail
                        .as_deref()
                        .is_some_and(|s| s.to_ascii_lowercase().contains(&q))
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn hotspots(&self) -> Vec<HotspotEntry> {
        let Some(report) = &self.report else {
            return Vec::new();
        };
        let mut map: BTreeMap<String, HotspotEntry> = BTreeMap::new();
        for (idx, rec) in report.snapshot.records.iter().enumerate() {
            if rec.kind != RecordKind::Tool && rec.kind != RecordKind::Assistant {
                continue;
            }
            let key = rec
                .tool_name
                .clone()
                .unwrap_or_else(|| "llm.assistant".to_string());
            let entry = map.entry(key.clone()).or_insert(HotspotEntry {
                name: key,
                calls: 0,
                errors: 0,
                total_ms: 0,
                first_record_idx: idx,
            });
            entry.calls += 1;
            if rec.is_error || rec.blocked {
                entry.errors += 1;
            }
            entry.total_ms = entry.total_ms.saturating_add(rec.duration_ms.unwrap_or(0));
        }
        let mut list: Vec<HotspotEntry> = map.into_values().collect();
        list.sort_by(|a, b| {
            b.total_ms
                .cmp(&a.total_ms)
                .then_with(|| b.calls.cmp(&a.calls))
        });
        list
    }

    pub fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            TimelineMode::Duration => TimelineMode::Sequence,
            TimelineMode::Sequence => TimelineMode::Actual,
            TimelineMode::Actual => TimelineMode::Duration,
        };
    }

    /// 返回 `true` 表示请求从后端刷新 `TrajectoryReport`（例如切换投影模式后）。
    pub fn handle_key(&mut self, key: KeyEvent) -> TrajectoryKeyOutcome {
        if self.search_active {
            match key.code {
                KeyCode::Esc => {
                    self.search_active = false;
                    return TrajectoryKeyOutcome::Consumed;
                }
                KeyCode::Enter => {
                    self.search_active = false;
                    self.cursor = 0;
                    return TrajectoryKeyOutcome::Consumed;
                }
                KeyCode::Backspace => {
                    self.query.pop();
                    self.cursor = 0;
                    return TrajectoryKeyOutcome::Consumed;
                }
                KeyCode::Char(c) => {
                    self.query.push(c);
                    self.cursor = 0;
                    return TrajectoryKeyOutcome::Consumed;
                }
                _ => return TrajectoryKeyOutcome::Consumed,
            }
        }

        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.open = false;
                TrajectoryKeyOutcome::Closed
            }
            KeyCode::Tab => {
                self.tab = match self.tab {
                    TrajectoryTab::Timeline => TrajectoryTab::Hotspots,
                    TrajectoryTab::Hotspots => TrajectoryTab::Timeline,
                };
                TrajectoryKeyOutcome::Consumed
            }
            KeyCode::Char('m') => {
                self.cycle_mode();
                TrajectoryKeyOutcome::ReloadReport
            }
            KeyCode::Char('/') => {
                self.search_active = true;
                TrajectoryKeyOutcome::Consumed
            }
            KeyCode::Up | KeyCode::Char('k') => {
                match self.tab {
                    TrajectoryTab::Timeline => self.cursor = self.cursor.saturating_sub(1),
                    TrajectoryTab::Hotspots => {
                        self.hotspot_cursor = self.hotspot_cursor.saturating_sub(1)
                    }
                }
                TrajectoryKeyOutcome::Consumed
            }
            KeyCode::Down | KeyCode::Char('j') => {
                match self.tab {
                    TrajectoryTab::Timeline => {
                        let max = self.filtered_indices().len().saturating_sub(1);
                        self.cursor = (self.cursor + 1).min(max);
                    }
                    TrajectoryTab::Hotspots => {
                        let max = self.hotspots().len().saturating_sub(1);
                        self.hotspot_cursor = (self.hotspot_cursor + 1).min(max);
                    }
                }
                TrajectoryKeyOutcome::Consumed
            }
            KeyCode::Enter if self.tab == TrajectoryTab::Hotspots => {
                let spots = self.hotspots();
                if let Some(spot) = spots.get(self.hotspot_cursor) {
                    let target = spot.first_record_idx;
                    self.tab = TrajectoryTab::Timeline;
                    self.query.clear();
                    self.cursor = target;
                }
                TrajectoryKeyOutcome::Consumed
            }
            _ => TrajectoryKeyOutcome::Consumed,
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> TrajectoryKeyOutcome {
        let col = mouse.column;
        let row = mouse.row;
        let contains = |r: Option<Rect>| -> bool {
            r.is_some_and(|a| {
                col >= a.x && col < a.x + a.width && row >= a.y && row < a.y + a.height
            })
        };
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if contains(self.close_area) {
                    self.open = false;
                    return TrajectoryKeyOutcome::Closed;
                }
                if contains(self.tab_timeline_area) {
                    self.tab = TrajectoryTab::Timeline;
                    return TrajectoryKeyOutcome::Consumed;
                }
                if contains(self.tab_hotspot_area) {
                    self.tab = TrajectoryTab::Hotspots;
                    return TrajectoryKeyOutcome::Consumed;
                }
                if contains(self.mode_chip_area) {
                    self.cycle_mode();
                    return TrajectoryKeyOutcome::ReloadReport;
                }
                if let Some(wa) = self.wave_area
                    && contains(Some(wa))
                {
                    let filtered = self.filtered_indices();
                    if !filtered.is_empty() && wa.width > 0 {
                        let rel = (col - wa.x) as usize;
                        let target = (rel * filtered.len()) / (wa.width as usize).max(1);
                        self.tab = TrajectoryTab::Timeline;
                        self.cursor = target.min(filtered.len() - 1);
                    }
                    return TrajectoryKeyOutcome::Consumed;
                }
                if let Some(la) = self.ledger_area
                    && contains(Some(la))
                {
                    let clicked_row = (row - la.y) as usize + self.ledger_scroll;
                    match self.tab {
                        TrajectoryTab::Timeline => {
                            let max = self.filtered_indices().len().saturating_sub(1);
                            self.cursor = clicked_row.min(max);
                        }
                        TrajectoryTab::Hotspots => {
                            let max = self.hotspots().len().saturating_sub(1);
                            self.hotspot_cursor = clicked_row.min(max);
                        }
                    }
                    return TrajectoryKeyOutcome::Consumed;
                }
            }
            MouseEventKind::ScrollUp => match self.tab {
                TrajectoryTab::Timeline => self.cursor = self.cursor.saturating_sub(3),
                TrajectoryTab::Hotspots => {
                    self.hotspot_cursor = self.hotspot_cursor.saturating_sub(1)
                }
            },
            MouseEventKind::ScrollDown => match self.tab {
                TrajectoryTab::Timeline => {
                    let max = self.filtered_indices().len().saturating_sub(1);
                    self.cursor = (self.cursor + 3).min(max);
                }
                TrajectoryTab::Hotspots => {
                    let max = self.hotspots().len().saturating_sub(1);
                    self.hotspot_cursor = (self.hotspot_cursor + 1).min(max);
                }
            },
            _ => {}
        }
        TrajectoryKeyOutcome::Consumed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryKeyOutcome {
    Consumed,
    ReloadReport,
    Closed,
}

pub fn render_trajectory_scene(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut TrajectorySceneState,
    theme: &Theme,
) {
    frame.render_widget(Clear, area);
    if area.width < 20 || area.height < 12 {
        frame.render_widget(Paragraph::new("终端尺寸过小，按 Esc 返回"), area);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // 标题与页签栏
            Constraint::Length(4), // WaveBand 三泳道波形总览
            Constraint::Min(4),    // 账本 / 热点主体
            Constraint::Length(7), // 底部固定 Inspector
            Constraint::Length(1), // 快捷键提示行
        ])
        .split(area);

    render_header(frame, chunks[0], state, theme);
    render_waveband(frame, chunks[1], state, theme);
    render_ledger_or_hotspots(frame, chunks[2], state, theme);
    render_inspector(frame, chunks[3], state, theme);
    render_hint_bar(frame, chunks[4], state, theme);
}

fn render_header(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut TrajectorySceneState,
    theme: &Theme,
) {
    let mode_label = match state.mode {
        TimelineMode::Duration => "压缩时长 (Duration)",
        TimelineMode::Sequence => "事件序列 (Sequence)",
        TimelineMode::Actual => "真实墙钟 (Actual)",
    };
    let tl_style = if state.tab == TrajectoryTab::Timeline {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    } else {
        Style::default().fg(theme.text_muted)
    };
    let hs_style = if state.tab == TrajectoryTab::Hotspots {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
    } else {
        Style::default().fg(theme.text_muted)
    };

    let stats_str = state
        .report
        .as_ref()
        .map(|r| {
            let s = &r.snapshot.stats;
            let avg_ttft = s.ttft_ms.checked_div(s.ttft_steps).unwrap_or(0);
            format!(
                "turns:{} steps:{} tools:{} err:{} avgTTFT:{}ms",
                s.turns, s.steps, s.tool_calls, s.tool_errors, avg_ttft
            )
        })
        .unwrap_or_else(|| "暂无轨迹数据".to_string());

    let line1 = Line::from(vec![
        Span::styled(
            " ◈ 会话轨迹分析 (Trajectory) ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("│ ", Style::default().fg(theme.border)),
        Span::styled(" Timeline ", tl_style),
        Span::raw(" "),
        Span::styled(" Hotspots ", hs_style),
        Span::styled(" │ ", Style::default().fg(theme.border)),
        Span::styled(
            format!("[轴: {mode_label}]"),
            Style::default().fg(theme.info),
        ),
        Span::raw("  "),
        Span::styled(stats_str, Style::default().fg(theme.text_muted)),
    ]);

    // 粗略记录可点区域
    state.tab_timeline_area = Some(Rect::new(area.x + 28, area.y, 10, 1));
    state.tab_hotspot_area = Some(Rect::new(area.x + 39, area.y, 10, 1));
    state.mode_chip_area = Some(Rect::new(area.x + 51, area.y, 24, 1));
    state.close_area = Some(Rect::new(
        area.x + area.width.saturating_sub(4),
        area.y,
        4,
        1,
    ));

    let rule = "─".repeat(area.width as usize);
    let line2 = Line::from(Span::styled(rule, Style::default().fg(theme.border)));
    frame.render_widget(Paragraph::new(vec![line1, line2]), area);
}

fn render_waveband(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut TrajectorySceneState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(Span::styled(
            " 三泳道波形总览 (Lane0 输入/告警 · Lane1 模型 · Lane2 工具) ",
            Style::default().fg(theme.text_muted),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    state.wave_area = Some(inner);

    if inner.width == 0 || inner.height < 2 {
        return;
    }

    let Some(report) = &state.report else {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "暂无波形事件",
                Style::default().fg(theme.text_dim),
            )),
            inner,
        );
        return;
    };

    let w = inner.width as usize;
    let filtered = state.filtered_indices();
    let cursor_rec_idx = filtered.get(state.cursor).copied();
    let total_records = report.snapshot.records.len().max(1);
    let cursor_col = cursor_rec_idx.map(|i| (i * w) / total_records);

    let mut lane_rows = [vec![' '; w], vec![' '; w]];
    let mut lane_err = [vec![false; w], vec![false; w]];

    if let Some(tl) = &report.timeline {
        let span_total = tl.end.saturating_sub(tl.start).max(1);
        for sp in &tl.spans {
            let row = if sp.lane == 2 { 1 } else { 0 };
            let c0 = (((sp.start.saturating_sub(tl.start)) as f64 / span_total as f64) * w as f64)
                as usize;
            let c1 = (((sp.end.saturating_sub(tl.start)) as f64 / span_total as f64) * w as f64)
                .ceil() as usize;
            for c in c0.min(w - 1)..=c1.min(w - 1) {
                lane_rows[row][c] = if sp.lane == 1 { '▆' } else { '▃' };
                if sp.is_error {
                    lane_err[row][c] = true;
                }
            }
        }
    }

    let mut lines = Vec::new();
    for r in 0..2 {
        let mut spans = Vec::with_capacity(w);
        for c in 0..w {
            let ch = lane_rows[r][c];
            let is_cur = cursor_col == Some(c);
            let style = if is_cur {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else if lane_err[r][c] {
                Style::default().fg(theme.error)
            } else if ch == '▆' {
                Style::default().fg(theme.info)
            } else if ch == '▃' {
                Style::default().fg(theme.success)
            } else {
                Style::default().fg(theme.text_dim)
            };
            spans.push(Span::styled(ch.to_string(), style));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_ledger_or_hotspots(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &mut TrajectorySceneState,
    theme: &Theme,
) {
    state.ledger_area = Some(area);
    let height = area.height as usize;
    if height == 0 {
        return;
    }

    let Some(report) = &state.report else {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "当前会话尚未记录任何轨迹事件。",
                Style::default().fg(theme.text_muted),
            )),
            area,
        );
        return;
    };

    let mut lines = Vec::new();
    match state.tab {
        TrajectoryTab::Timeline => {
            let filtered = state.filtered_indices();
            if filtered.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  没有匹配的轨迹记录",
                    Style::default().fg(theme.text_muted),
                )));
            } else {
                state.cursor = state.cursor.min(filtered.len() - 1);
                if state.cursor < state.ledger_scroll {
                    state.ledger_scroll = state.cursor;
                } else if state.cursor >= state.ledger_scroll + height {
                    state.ledger_scroll = state.cursor + 1 - height;
                }
                let end = (state.ledger_scroll + height).min(filtered.len());
                for (vis_idx, &rec_idx) in filtered[state.ledger_scroll..end].iter().enumerate() {
                    let abs = state.ledger_scroll + vis_idx;
                    let rec = &report.snapshot.records[rec_idx];
                    lines.push(format_ledger_row(
                        rec,
                        abs == state.cursor,
                        area.width as usize,
                        theme,
                    ));
                }
            }
        }
        TrajectoryTab::Hotspots => {
            let spots = state.hotspots();
            if spots.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  暂无工具或模型耗时数据",
                    Style::default().fg(theme.text_muted),
                )));
            } else {
                state.hotspot_cursor = state.hotspot_cursor.min(spots.len() - 1);
                if state.hotspot_cursor < state.ledger_scroll {
                    state.ledger_scroll = state.hotspot_cursor;
                } else if state.hotspot_cursor >= state.ledger_scroll + height {
                    state.ledger_scroll = state.hotspot_cursor + 1 - height;
                }
                let end = (state.ledger_scroll + height).min(spots.len());
                for (vis_idx, spot) in spots[state.ledger_scroll..end].iter().enumerate() {
                    let abs = state.ledger_scroll + vis_idx;
                    let selected = abs == state.hotspot_cursor;
                    let pointer = if selected { "▸ " } else { "  " };
                    let err_tag = if spot.errors > 0 {
                        format!(" err:{}", spot.errors)
                    } else {
                        String::new()
                    };
                    let row_text = format!(
                        "{pointer}{:<24} {:>5} calls  {:>7}ms{err_tag}",
                        spot.name, spot.calls, spot.total_ms
                    );
                    let style = if selected {
                        Style::default()
                            .fg(theme.accent)
                            .bg(theme.card_bg_alt)
                            .add_modifier(Modifier::BOLD)
                    } else if spot.errors > 0 {
                        Style::default().fg(theme.error)
                    } else {
                        Style::default().fg(theme.text)
                    };
                    lines.push(Line::from(Span::styled(row_text, style)));
                }
            }
        }
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn format_ledger_row(
    rec: &TrajectoryRecord,
    selected: bool,
    max_w: usize,
    theme: &Theme,
) -> Line<'static> {
    let pointer = if selected { "▸ " } else { "  " };
    let (kind_tag, kind_color) = match rec.kind {
        RecordKind::User => ("USER   ", theme.accent),
        RecordKind::Context => ("CTX    ", theme.text_muted),
        RecordKind::Assistant => ("MODEL  ", theme.info),
        RecordKind::Tool => (
            "TOOL   ",
            if rec.is_error {
                theme.error
            } else {
                theme.success
            },
        ),
        RecordKind::Anomaly => ("ALERT  ", theme.warning),
    };
    let turn_str = rec
        .turn
        .map(|t| format!("T{t:<2}"))
        .unwrap_or_else(|| "   ".to_string());
    let dur_str = rec
        .duration_ms
        .map(|d| format!("{d:>5}ms"))
        .unwrap_or_else(|| "      -".to_string());
    let ttft_str = rec
        .ttft_ms
        .map(|t| format!(" ttft={t}ms"))
        .unwrap_or_default();
    let label_budget = max_w.saturating_sub(24 + ttft_str.len());
    let label = clip_text(&rec.label, label_budget);

    let bg_style = if selected {
        Style::default().bg(theme.card_bg_alt)
    } else {
        Style::default()
    };

    Line::from(vec![
        Span::styled(
            pointer,
            bg_style.fg(if selected {
                theme.accent
            } else {
                theme.text_dim
            }),
        ),
        Span::styled(turn_str, bg_style.fg(theme.text_dim)),
        Span::styled(" ", bg_style),
        Span::styled(
            kind_tag,
            bg_style.fg(kind_color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(dur_str, bg_style.fg(theme.text_muted)),
        Span::styled(ttft_str, bg_style.fg(theme.info)),
        Span::styled("  ", bg_style),
        Span::styled(label, bg_style.fg(theme.text)),
    ])
}

fn render_inspector(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TrajectorySceneState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(Span::styled(
            " 检查器 (Inspector) ",
            Style::default().fg(theme.accent),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(report) = &state.report else {
        return;
    };
    let filtered = state.filtered_indices();
    let Some(&rec_idx) = filtered.get(state.cursor) else {
        return;
    };
    let rec = &report.snapshot.records[rec_idx];

    let mut lines = Vec::new();
    let usage_info = rec
        .usage
        .map(|u| format!(" | tokens ↑{} ↓{}", u.input_tokens, u.output_tokens))
        .unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled(
            format!("#{} {} ", rec.index + 1, rec.label),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        ),
        Span::styled(usage_info, Style::default().fg(theme.text_muted)),
    ]));
    if let Some(inp) = &rec.input_detail {
        lines.push(Line::from(Span::styled(
            format!(
                "输入: {}",
                clip_text(inp, inner.width.saturating_sub(6) as usize)
            ),
            Style::default().fg(theme.info),
        )));
    }
    if let Some(out) = &rec.output_detail {
        lines.push(Line::from(Span::styled(
            format!(
                "输出: {}",
                clip_text(out, inner.width.saturating_sub(6) as usize)
            ),
            Style::default().fg(theme.text_muted),
        )));
    }
    if !report.findings.is_empty() {
        let f = &report.findings[0];
        let sev_col = match f.severity {
            Severity::Error => theme.error,
            Severity::Warn => theme.warning,
            Severity::Info => theme.info,
        };
        lines.push(Line::from(Span::styled(
            format!("告警 ({}): {}", report.findings.len(), f.summary),
            Style::default().fg(sev_col),
        )));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_hint_bar(frame: &mut Frame<'_>, area: Rect, state: &TrajectorySceneState, theme: &Theme) {
    let text = if state.search_active {
        format!(" 搜索: {}█  (Enter 确认 / Esc 取消)", state.query)
    } else {
        let q = if state.query.is_empty() {
            String::new()
        } else {
            format!(" | 过滤: `{}`", state.query)
        };
        format!(" ↑↓/jk 移动 · Tab 切换 Timeline/Hotspots · m 切换时间轴 · / 搜索 · Esc/q 返回{q}")
    };
    frame.render_widget(
        Paragraph::new(Span::styled(text, Style::default().fg(theme.text_muted))),
        area,
    );
}

fn clip_text(s: &str, max_w: usize) -> String {
    let single: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if display_width(&single) <= max_w {
        return single;
    }
    let mut out = String::new();
    let mut w = 0usize;
    let limit = max_w.saturating_sub(1);
    for ch in single.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw > limit {
            out.push('…');
            return out;
        }
        out.push(ch);
        w += cw;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mizar_trajectory_core::{TraceEvent, TraceEventKind, derive_timeline, fold_trajectory};

    #[test]
    fn renders_trajectory_scene_and_switches_tabs_and_modes() {
        let events = vec![
            TraceEvent {
                seq: 1,
                ts_ms: 1000,
                session_id: "s1".into(),
                kind: TraceEventKind::TurnStart { turn: 1 },
            },
            TraceEvent {
                seq: 2,
                ts_ms: 1010,
                session_id: "s1".into(),
                kind: TraceEventKind::UserMessage {
                    text: "check files".into(),
                },
            },
            TraceEvent {
                seq: 3,
                ts_ms: 1020,
                session_id: "s1".into(),
                kind: TraceEventKind::ToolCall {
                    turn: 1,
                    step: 1,
                    call_id: "c1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
            },
            TraceEvent {
                seq: 4,
                ts_ms: 1120,
                session_id: "s1".into(),
                kind: TraceEventKind::ToolResult {
                    tool_call_id: "c1".into(),
                    name: "read_file".into(),
                    content: "ok".into(),
                    is_error: false,
                    blocked: false,
                },
            },
        ];
        let snap = fold_trajectory("s1", &events);
        let tl = derive_timeline(&snap.records, TimelineMode::Duration);
        let report = TrajectoryReport {
            snapshot: snap,
            timeline: tl,
            findings: Vec::new(),
        };

        let mut scene = TrajectorySceneState {
            open: true,
            report: Some(report),
            ..Default::default()
        };

        assert_eq!(scene.hotspots().len(), 1);
        assert_eq!(scene.hotspots()[0].name, "read_file");

        let backend = ratatui::backend::TestBackend::new(90, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let theme = Theme::mist_dark();
        terminal
            .draw(|f| render_trajectory_scene(f, f.area(), &mut scene, &theme))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..24 {
            let mut x = 0u16;
            while x < 90 {
                let sym = buf[(x, y)].symbol();
                text.push_str(sym);
                let step = crate::editor::display_width(sym).max(1) as u16;
                x += step;
            }
            text.push('\n');
        }
        assert!(text.contains("会话轨迹分析"));
        assert!(text.contains("read_file"));
    }
}
