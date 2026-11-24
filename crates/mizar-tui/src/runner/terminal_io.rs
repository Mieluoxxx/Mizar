//! 终端 I/O 与帧渲染节流：同步缓冲输出、双模终端重建、Inline 原生滚屏冲刷与脏帧门控。

use std::sync::{Arc, Mutex};

use crossterm::event::{Event as TermEvent, EventStream, KeyEventKind, MouseEventKind};
use futures::StreamExt;
use mizar_client::events::{LoopClientMsg, RunController};
use mizar_tui_ui::app::{AppAction, AppState, ChatItem, UiEvent};
use mizar_tui_ui::terminal::{BEGIN_SYNCHRONIZED_OUTPUT, END_SYNCHRONIZED_OUTPUT, ScreenMode};
use mizar_tui_ui::ui::{build_item_lines_for_scrollback_at, build_welcome_lines_for_scrollback};
use mizar_types::client::ClientEvent;
use ratatui::backend::CrosstermBackend;
use ratatui::{Terminal, TerminalOptions, Viewport};

/// 把单帧 `BSU + ratatui 差分序列 + ESU` 压进同一个缓冲 Writer 并阻止中间提前 flush，
/// 让终端整帧通过单次底层 `write()` 系统调用原子落盘，减少切帧闪烁与 syscall 开销。
pub(super) struct SyncWriterInner<W: std::io::Write> {
    inner: W,
    defer_flush: bool,
}

impl<W: std::io::Write> SyncWriterInner<W> {
    pub(super) fn new(inner: W) -> Self {
        Self {
            inner,
            defer_flush: false,
        }
    }

    pub(super) fn begin_sync_frame(&mut self) -> std::io::Result<()> {
        self.defer_flush = true;
        self.inner.write_all(BEGIN_SYNCHRONIZED_OUTPUT.as_bytes())
    }

    pub(super) fn end_sync_frame(&mut self) -> std::io::Result<()> {
        self.defer_flush = false;
        self.inner.write_all(END_SYNCHRONIZED_OUTPUT.as_bytes())?;
        self.inner.flush()
    }
}

impl<W: std::io::Write> std::io::Write for SyncWriterInner<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.defer_flush {
            return Ok(());
        }
        self.inner.flush()
    }
}

#[derive(Clone)]
pub(super) struct SharedSyncWriter {
    inner: Arc<Mutex<SyncWriterInner<std::io::BufWriter<std::io::Stdout>>>>,
}

impl std::io::Write for SharedSyncWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.lock().unwrap().flush()
    }
}

pub(super) struct ManagedTerminal {
    pub(super) terminal: Terminal<CrosstermBackend<SharedSyncWriter>>,
    sync: SharedSyncWriter,
}

impl ManagedTerminal {
    pub(super) fn draw_synced<F>(&mut self, render_cb: F) -> std::io::Result<()>
    where
        F: FnOnce(&mut ratatui::Frame<'_>),
    {
        let _ = self.sync.inner.lock().unwrap().begin_sync_frame();
        let draw_res = self.terminal.draw(render_cb);
        let end_res = self.sync.inner.lock().unwrap().end_sync_frame();
        draw_res?;
        end_res
    }
}

pub(super) type BufferedTerminal = ManagedTerminal;

pub(super) async fn poll_term_event(
    stream: Option<&mut EventStream>,
) -> Option<std::io::Result<TermEvent>> {
    match stream {
        Some(s) => s.next().await,
        None => std::future::pending().await,
    }
}

pub(super) fn handle_term_event_dirty(app: &mut AppState, ev: TermEvent) -> (AppAction, bool) {
    match ev {
        TermEvent::Key(key) => {
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                return (AppAction::None, false);
            }
            (app.handle_key(key), true)
        }
        TermEvent::Mouse(mouse) => {
            if matches!(mouse.kind, MouseEventKind::Moved) {
                let prev = (app.side_panel.hovered_divider, app.hovered_rail_row);
                let act = app.handle_mouse(mouse);
                let dirty = act != AppAction::None
                    || app.overlay.is_open()
                    || app.trajectory_scene.open
                    || (app.side_panel.hovered_divider, app.hovered_rail_row) != prev;
                return (act, dirty);
            }
            (app.handle_mouse(mouse), true)
        }
        TermEvent::Paste(data) => {
            app.handle_paste(&data);
            (AppAction::None, true)
        }
        TermEvent::Resize(_, _) => (AppAction::None, true),
        TermEvent::FocusGained => {
            app.terminal_focused = true;
            (AppAction::None, false)
        }
        TermEvent::FocusLost => {
            app.terminal_focused = false;
            (AppAction::None, false)
        }
    }
}

pub(super) fn process_loop_ui_msg_dirty(
    app: &mut AppState,
    task_state: &mut RunController,
    msg: LoopClientMsg,
) -> (AppAction, bool) {
    let is_delta = matches!(
        msg,
        LoopClientMsg::Bus(UiEvent::TextDelta { .. } | UiEvent::ReasoningDelta { .. })
    );
    let had_assistant = matches!(app.items.last(), Some(ChatItem::Assistant { .. }));
    let prev_reveal_gen = app.reveal.generation();

    let action = super::process_loop_ui_msg(app, task_state, msg);
    if action != AppAction::None {
        return (action, true);
    }

    let can_defer_to_reveal = is_delta
        && app.reveal.enabled
        && app.is_busy()
        && had_assistant
        && app.reveal.generation() == prev_reveal_gen;
    (AppAction::None, !can_defer_to_reveal)
}

/// 将 `SessionActor` 广播的同进程零序列化 `ClientEvent` 接入脏帧门控与 UI 归约。
pub(super) fn process_client_event_dirty(
    app: &mut AppState,
    task_state: &mut RunController,
    event: ClientEvent,
) -> (AppAction, bool) {
    process_loop_ui_msg_dirty(app, task_state, LoopClientMsg::Bus(event))
}

pub(super) fn create_terminal(mode: ScreenMode) -> Result<BufferedTerminal, std::io::Error> {
    let sync = SharedSyncWriter {
        inner: Arc::new(Mutex::new(SyncWriterInner::new(
            std::io::BufWriter::with_capacity(64 * 1024, std::io::stdout()),
        ))),
    };
    let backend = CrosstermBackend::new(sync.clone());
    let terminal = match mode {
        ScreenMode::Fullscreen => {
            let mut t = Terminal::new(backend)?;
            t.clear()?;
            t
        }
        ScreenMode::Inline => {
            let (_, rows) = crossterm::terminal::size().unwrap_or((80, 24));
            let viewport_h = rows.clamp(7, 14);
            Terminal::with_options(
                backend,
                TerminalOptions {
                    viewport: Viewport::Inline(viewport_h),
                },
            )?
        }
    };
    Ok(ManagedTerminal { terminal, sync })
}

pub(super) fn insert_scrollback_lines<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    lines: &[ratatui::text::Line<'static>],
    chunk_max: usize,
) {
    for chunk in lines.chunks(chunk_max) {
        let h = chunk.len() as u16;
        let _ = terminal.insert_before(h, |buf| {
            for (row_off, line) in chunk.iter().enumerate() {
                buf.set_line(
                    buf.area.x,
                    buf.area.y + row_off as u16,
                    line,
                    buf.area.width,
                );
            }
        });
    }
}

pub(super) fn flush_inline_scrollback<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
) {
    if app.inline_committed_items >= app.items.len() {
        return;
    }
    let Ok(size) = terminal.size() else {
        return;
    };
    let width = size.width.max(20) as usize;
    let chunk_max = size.height.saturating_sub(2).max(4) as usize;

    // 首次写入原生 scrollback 时先写入欢迎页序言（含 1 行尾空行），切回全屏再返 Inline 不重复追加
    if !app.inline_welcome_committed {
        let welcome_lines = build_welcome_lines_for_scrollback(app, width);
        insert_scrollback_lines(terminal, &welcome_lines, chunk_max);
        app.inline_welcome_committed = true;
    }

    while app.inline_committed_items < app.items.len() {
        let idx = app.inline_committed_items;
        let lines = build_item_lines_for_scrollback_at(
            &app.items,
            idx,
            &app.tool_extras,
            width,
            &app.theme,
            &app.emphasis,
            app.show_thinking,
            app.verbose_expanded,
        );
        insert_scrollback_lines(terminal, &lines, chunk_max);
        app.inline_committed_items += 1;
    }
}
