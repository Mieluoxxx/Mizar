//! 全屏应用级鼠标文本选区与剪贴板引擎（借鉴 `dsh-TUI` `selection.ts` + `pi` `tui-alt-screen.ts`）：
//!
//! - 解决开启 `EnableMouseCapture` 后终端原生划词复制失效的问题；
//! - 支持单击拖拽按字符选区、双击按词选区（含路径/标识符连接符）、三击选整行；
//! - 方向感知多栏围栏（Direction-Aware Fence）：从聊天区起手拖选绝不越界抓到侧栏/时间轨，
//!   从侧栏起手拖选也只复制侧栏列；
//! - 自动根据渲染阶段登记的 `soft_wrapped` 标志拼接软换行，去除边框装饰并同步写入 OSC 52 + 系统剪贴板。

use std::io::Write;
use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;

use crate::editor::display_width;
use crate::theme::Theme;

const MULTI_CLICK_MS: Duration = Duration::from_millis(450);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionPane {
    Chat,
    SidePanel,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionGranularity {
    #[default]
    Char,
    Word,
    Line,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellPoint {
    pub col: u16,
    pub row: u16,
}

/// 每一渲染行登记给选区系统的几何元信息（行级可选择列范围 + 软换行标志）。
#[derive(Debug, Clone)]
pub struct SelectableRowSpan {
    pub pane: SelectionPane,
    pub row: u16,
    pub col_start: u16,
    pub col_end: u16,
    pub soft_wrapped: bool,
}

/// 每帧重建的屏幕可选择区域索引。
#[derive(Debug, Default, Clone)]
pub struct SelectionGeometry {
    pub chat_rect: Option<Rect>,
    pub side_panel_rect: Option<Rect>,
    pub input_rect: Option<Rect>,
    pub rows: Vec<SelectableRowSpan>,
}

impl SelectionGeometry {
    pub fn clear(&mut self) {
        self.chat_rect = None;
        self.side_panel_rect = None;
        self.input_rect = None;
        self.rows.clear();
    }

    pub fn push_row(
        &mut self,
        pane: SelectionPane,
        row: u16,
        col_start: u16,
        col_end: u16,
        soft_wrapped: bool,
    ) {
        if col_end > col_start {
            self.rows.push(SelectableRowSpan {
                pane,
                row,
                col_start,
                col_end,
                soft_wrapped,
            });
        }
    }

    pub fn pane_at(&self, col: u16, row: u16) -> Option<(SelectionPane, Rect)> {
        if let Some(r) = self.side_panel_rect
            && rect_contains(r, col, row)
        {
            return Some((SelectionPane::SidePanel, r));
        }
        if let Some(r) = self.chat_rect
            && rect_contains(r, col, row)
        {
            return Some((SelectionPane::Chat, r));
        }
        if let Some(r) = self.input_rect
            && rect_contains(r, col, row)
        {
            return Some((SelectionPane::Input, r));
        }
        None
    }

    pub fn row_meta(&self, pane: SelectionPane, row: u16) -> Option<&SelectableRowSpan> {
        self.rows.iter().find(|r| r.pane == pane && r.row == row)
    }
}

fn rect_contains(r: Rect, col: u16, row: u16) -> bool {
    col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
}

#[derive(Debug, Default, Clone)]
pub struct SelectionState {
    pub anchor: Option<CellPoint>,
    pub focus: Option<CellPoint>,
    pub dragging: bool,
    pub moved: bool,
    pub granularity: SelectionGranularity,
    pub pane: Option<SelectionPane>,
    pub fence: Option<Rect>,
    /// 双击选词或三击选行时锁定的原始单元边界 `[unit_start, unit_end]`，用于双向拖拽不丢首单元。
    origin_unit: Option<(CellPoint, CellPoint)>,
    last_click_at: Option<Instant>,
    last_click_pos: Option<CellPoint>,
    last_click_pane: Option<SelectionPane>,
    click_count: u8,
}

impl SelectionState {
    pub fn is_active(&self) -> bool {
        self.normalized_bounds().is_some()
    }

    pub fn clear(&mut self) {
        self.anchor = None;
        self.focus = None;
        self.dragging = false;
        self.moved = false;
        self.granularity = SelectionGranularity::Char;
        self.pane = None;
        self.fence = None;
        self.origin_unit = None;
    }

    pub fn start_press(
        &mut self,
        col: u16,
        row: u16,
        geom: &SelectionGeometry,
        buf: Option<&Buffer>,
    ) {
        let Some((pane, rect)) = geom.pane_at(col, row) else {
            self.clear();
            self.last_click_at = None;
            self.last_click_pos = None;
            self.last_click_pane = None;
            self.click_count = 0;
            return;
        };

        let now = Instant::now();
        let same_spot = self.last_click_pane == Some(pane)
            && self
                .last_click_at
                .is_some_and(|t| now.duration_since(t) <= MULTI_CLICK_MS)
            && self
                .last_click_pos
                .is_some_and(|p| p.row == row && p.col.abs_diff(col) <= 1);
        self.click_count = if same_spot {
            (self.click_count % 3) + 1
        } else {
            1
        };
        self.last_click_at = Some(now);
        self.last_click_pos = Some(CellPoint { col, row });
        self.last_click_pane = Some(pane);

        self.pane = Some(pane);
        self.fence = Some(rect);
        self.dragging = true;
        self.moved = false;
        self.origin_unit = None;

        match self.click_count {
            2 => {
                self.granularity = SelectionGranularity::Word;
                let (start_c, end_c) = if let Some(b) = buf {
                    expand_word_bounds(b, rect, col, row)
                } else {
                    (col, col)
                };
                let u_start = CellPoint { col: start_c, row };
                let u_end = CellPoint { col: end_c, row };
                self.origin_unit = Some((u_start, u_end));
                self.anchor = Some(u_start);
                self.focus = Some(u_end);
            }
            3 => {
                self.granularity = SelectionGranularity::Line;
                let (c0, c1) = geom
                    .row_meta(pane, row)
                    .map(|m| (m.col_start, m.col_end.saturating_sub(1)))
                    .unwrap_or((rect.x, (rect.x + rect.width).saturating_sub(1)));
                let u_start = CellPoint { col: c0, row };
                let u_end = CellPoint { col: c1, row };
                self.origin_unit = Some((u_start, u_end));
                self.anchor = Some(u_start);
                self.focus = Some(u_end);
            }
            _ => {
                self.granularity = SelectionGranularity::Char;
                self.anchor = Some(CellPoint { col, row });
                self.focus = None;
            }
        }
    }

    pub fn update_drag(&mut self, col: u16, row: u16, buf: Option<&Buffer>) {
        if !self.dragging {
            return;
        }
        let Some(fence) = self.fence else {
            return;
        };
        if fence.width == 0 || fence.height == 0 {
            return;
        }
        let clamped_col = col.clamp(fence.x, fence.x + fence.width - 1);
        let clamped_row = row.clamp(fence.y, fence.y + fence.height - 1);
        self.moved = true;

        match self.granularity {
            SelectionGranularity::Char => {
                self.focus = Some(CellPoint {
                    col: clamped_col,
                    row: clamped_row,
                });
            }
            SelectionGranularity::Word => {
                let (w_start, w_end) = if let Some(b) = buf {
                    expand_word_bounds(b, fence, clamped_col, clamped_row)
                } else {
                    (clamped_col, clamped_col)
                };
                let cur_start = CellPoint {
                    col: w_start,
                    row: clamped_row,
                };
                let cur_end = CellPoint {
                    col: w_end,
                    row: clamped_row,
                };
                if let Some((orig_start, orig_end)) = self.origin_unit {
                    if (cur_start.row, cur_start.col) < (orig_start.row, orig_start.col) {
                        self.anchor = Some(orig_end);
                        self.focus = Some(cur_start);
                    } else {
                        self.anchor = Some(orig_start);
                        self.focus = Some(cur_end);
                    }
                } else {
                    self.focus = Some(cur_end);
                }
            }
            SelectionGranularity::Line => {
                let row_start = CellPoint {
                    col: fence.x,
                    row: clamped_row,
                };
                let row_end = CellPoint {
                    col: fence.x + fence.width - 1,
                    row: clamped_row,
                };
                if let Some((orig_start, orig_end)) = self.origin_unit {
                    if clamped_row < orig_start.row {
                        self.anchor = Some(orig_end);
                        self.focus = Some(row_start);
                    } else {
                        self.anchor = Some(orig_start);
                        self.focus = Some(row_end);
                    }
                } else {
                    self.focus = Some(row_end);
                }
            }
        }
    }

    pub fn finish_release(&mut self) -> bool {
        if !self.dragging {
            return false;
        }
        self.dragging = false;
        self.origin_unit = None;
        if self.granularity == SelectionGranularity::Char && !self.moved {
            // 单纯点击（无拖拽）不产生单格幽灵选区
            self.clear();
            return false;
        }
        self.is_active()
    }

    pub fn normalized_bounds(&self) -> Option<(CellPoint, CellPoint)> {
        let a = self.anchor?;
        let f = self.focus?;
        if (a.row, a.col) <= (f.row, f.col) {
            Some((a, f))
        } else {
            Some((f, a))
        }
    }

    pub fn contains_cell(&self, col: u16, row: u16) -> bool {
        let Some((start, end)) = self.normalized_bounds() else {
            return false;
        };
        if let Some(fence) = self.fence
            && !rect_contains(fence, col, row)
        {
            return false;
        }
        if row < start.row || row > end.row {
            return false;
        }
        if start.row == end.row {
            col >= start.col && col <= end.col
        } else if row == start.row {
            col >= start.col
        } else if row == end.row {
            col <= end.col
        } else {
            true
        }
    }
}

fn is_word_char(sym: &str) -> bool {
    let Some(ch) = sym.chars().next() else {
        return false;
    };
    if ch.is_whitespace() {
        return false;
    }
    ch.is_alphanumeric() || matches!(ch, '/' | '-' | '+' | '\\' | '~' | '_' | '.' | ':' | '@')
}

fn expand_word_bounds(buf: &Buffer, fence: Rect, col: u16, row: u16) -> (u16, u16) {
    let min_x = fence.x;
    let max_x = (fence.x + fence.width).saturating_sub(1);
    let col = col.clamp(min_x, max_x);
    let row = row.clamp(fence.y, (fence.y + fence.height).saturating_sub(1));

    if !is_word_char(buf[(col, row)].symbol()) {
        return (col, col);
    }

    let mut lo = col;
    while lo > min_x {
        let prev = lo - 1;
        let sym = buf[(prev, row)].symbol();
        // 宽字符占位格 symbol 为空或空格时需小心，不跨空格
        if sym.is_empty() || is_word_char(sym) {
            lo = prev;
        } else {
            break;
        }
    }
    let mut hi = col;
    while hi < max_x {
        let next = hi + 1;
        let sym = buf[(next, row)].symbol();
        if sym.is_empty() || is_word_char(sym) {
            hi = next;
        } else {
            break;
        }
    }
    (lo, hi)
}

/// 将选区高亮直接叠加绘制到当前 `ratatui::Buffer` 上。
pub fn paint_selection_overlay(buf: &mut Buffer, sel: &SelectionState, theme: &Theme) {
    let Some((start, end)) = sel.normalized_bounds() else {
        return;
    };
    let area = buf.area;
    let row_lo = start.row.max(area.y);
    let row_hi = end.row.min(area.y + area.height.saturating_sub(1));
    let fence = sel.fence.unwrap_or(area);
    let col_lo_fence = fence.x.max(area.x);
    let col_hi_fence = (fence.x + fence.width)
        .min(area.x + area.width)
        .saturating_sub(1);

    for row in row_lo..=row_hi {
        let c_start = if row == start.row {
            start.col.max(col_lo_fence)
        } else {
            col_lo_fence
        };
        let c_end = if row == end.row {
            end.col.min(col_hi_fence)
        } else {
            col_hi_fence
        };
        for col in c_start..=c_end {
            let cell = &mut buf[(col, row)];
            if theme.truecolor {
                cell.set_bg(theme.selection_bg);
                cell.set_fg(theme.selection_fg);
            } else {
                cell.modifier.insert(Modifier::REVERSED);
            }
        }
    }
}

/// 从渲染缓冲提取选区文本（自动按 `SelectableRowSpan` 跳过边框/行轨并拼接软换行）。
pub fn extract_selected_text(
    buf: &Buffer,
    sel: &SelectionState,
    geom: &SelectionGeometry,
) -> Option<String> {
    let (start, end) = sel.normalized_bounds()?;
    let pane = sel.pane?;
    let fence = sel.fence.unwrap_or(buf.area);

    let mut out = String::new();
    let mut first_emitted = true;

    for row in start.row..=end.row {
        let (row_min_c, row_max_c, soft_wrapped) = match geom.row_meta(pane, row) {
            Some(meta) => (
                meta.col_start,
                meta.col_end.saturating_sub(1),
                meta.soft_wrapped,
            ),
            None => (fence.x, (fence.x + fence.width).saturating_sub(1), false),
        };
        let c_start = if row == start.row {
            start.col.max(row_min_c)
        } else {
            row_min_c
        };
        let c_end = if row == end.row {
            end.col.min(row_max_c)
        } else {
            row_max_c
        };
        if c_end < c_start {
            continue;
        }

        let mut line_str = String::new();
        let mut col = c_start;
        while col <= c_end {
            if col >= buf.area.x + buf.area.width || row >= buf.area.y + buf.area.height {
                break;
            }
            let sym = buf[(col, row)].symbol();
            let step = display_width(sym).max(1) as u16;
            // 剥除 pi-emphasis 的 Nerd Font 圆角胶囊帽（复制保持纯文本，对齐 pi-emphasis 约定）
            if sym == "\u{e0b6}" || sym == "\u{e0b4}" {
                col = col.saturating_add(step);
                continue;
            }
            if !sym.is_empty() {
                line_str.push_str(sym);
            }
            col = col.saturating_add(step);
        }
        // 把不可折行空格还原为普通空格并去掉行尾空白
        let clean = line_str.replace('\u{00a0}', " ").trim_end().to_string();

        if !first_emitted && !soft_wrapped {
            out.push('\n');
        }
        out.push_str(&clean);
        first_emitted = false;
    }

    if out.trim().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// 写入剪贴板（OSC 52 终端序列 + 后台调用系统剪贴板工具），绝不阻塞事件循环。
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    if text.is_empty() {
        return Ok(());
    }
    // 1. 发送 OSC 52（支持 Ghostty / Kitty / WezTerm / iTerm2 / Windows Terminal / SSH）
    let b64 = encode_base64(text.as_bytes());
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\x1b]52;c;{b64}\x07");
    let _ = stdout.flush();

    // 2. 尽力写入本机系统剪贴板（不阻塞主渲染线程）
    let owned = text.to_string();
    std::thread::spawn(move || {
        let _ = write_native_clipboard(&owned);
    });
    Ok(())
}

fn write_native_clipboard(text: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    let candidates: &[(&str, &[&str])] = &[("pbcopy", &[])];
    #[cfg(target_os = "linux")]
    let candidates: &[(&str, &[&str])] = &[
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
    ];
    #[cfg(target_os = "windows")]
    let candidates: &[(&str, &[&str])] = &[("clip.exe", &[])];
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let candidates: &[(&str, &[&str])] = &[];

    for (cmd, args) in candidates {
        if let Ok(mut child) = std::process::Command::new(cmd)
            .args(*args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
            return Ok(());
        }
    }
    Ok(())
}

fn encode_base64(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}
