//! 终端原始模式、同步输出（BSU/ESU）、Kitty 键盘增强、OSC 7501/9;4 状态协议与 OSC 11 探测。

use std::io::{self, Write};
use std::sync::Once;
use std::time::{Duration, Instant};

use crossterm::event::{
    DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste,
    EnableFocusChange, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};

use crate::theme::{Rgb8, TerminalColors};

pub const BEGIN_SYNCHRONIZED_OUTPUT: &str = "\x1b[?2026h";
pub const END_SYNCHRONIZED_OUTPUT: &str = "\x1b[?2026l";
pub const ENABLE_COLOR_SCHEME_NOTIFY: &str = "\x1b[?2031h";
pub const DISABLE_COLOR_SCHEME_NOTIFY: &str = "\x1b[?2031l";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenMode {
    #[default]
    Fullscreen,
    Inline,
}

impl ScreenMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fullscreen => "fullscreen",
            Self::Inline => "inline",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramStatusState {
    Idle,
    Working,
    Blocked,
    Done,
    Error,
    Clear,
}

impl ProgramStatusState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Working => "working",
            Self::Blocked => "blocked",
            Self::Done => "done",
            Self::Error => "error",
            Self::Clear => "clear",
        }
    }
}

/// 生成符合 OSC 7501（Program Status Protocol）的控制序列（对齐 `pi` / `dsh-TUI`）。
pub fn format_osc_7501(
    state: ProgramStatusState,
    kind: Option<&str>,
    message: Option<&str>,
) -> String {
    let mut parts = vec![format!("state={}", state.as_str()), "app=dsh-tui".to_string()];
    if state == ProgramStatusState::Blocked
        && let Some(k) = kind
    {
        parts.push(format!("kind={k}"));
    }
    if let Some(msg) = message {
        let cleaned: String = msg
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let trimmed = cleaned.trim();
        if !trimmed.is_empty() {
            let truncated = truncate_utf8_bytes(trimmed, 1024);
            let b64 = encode_base64_str(truncated.as_bytes());
            parts.push(format!("msg={b64}"));
        }
    }
    format!("\x1b]7501;{}\x1b\\", parts.join(":"))
}

pub fn emit_program_status(
    state: ProgramStatusState,
    kind: Option<&str>,
    message: Option<&str>,
) {
    if matches!(std::env::var("DSH_TUI_PROGRAM_STATUS").as_deref(), Ok("0" | "false")) {
        return;
    }
    let seq = format_osc_7501(state, kind, message);
    let mut out = io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// 生成 OSC 9;4 终端标签页进度指示（`active=true` 为不确定脉冲，`false` 为清除）。
pub fn emit_terminal_progress(active: bool) {
    let seq = if active {
        "\x1b]9;4;3\x07"
    } else {
        "\x1b]9;4;0\x07"
    };
    let mut out = io::stdout();
    let _ = out.write_all(seq.as_bytes());
    let _ = out.flush();
}

/// 包裹同步输出序列（CSI ? 2026 h / l）以消除流式高刷或 resize 闪烁。
pub fn write_begin_sync() {
    let mut out = io::stdout();
    let _ = out.write_all(BEGIN_SYNCHRONIZED_OUTPUT.as_bytes());
}

pub fn write_end_sync() {
    let mut out = io::stdout();
    let _ = out.write_all(END_SYNCHRONIZED_OUTPUT.as_bytes());
    let _ = out.flush();
}

/// 恢复终端到正常状态，保证异常退出或 panic 后终端仍可使用。
pub fn restore_terminal() {
    emit_terminal_progress(false);
    emit_program_status(ProgramStatusState::Clear, None, None);
    let _ = disable_raw_mode();
    let mut out = io::stdout();
    let _ = execute!(
        out,
        PopKeyboardEnhancementFlags,
        DisableFocusChange,
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    let _ = out.write_all(DISABLE_COLOR_SCHEME_NOTIFY.as_bytes());
    let _ = out.flush();
}

/// 安装一次性 panic hook：先恢复终端状态，再输出 panic 信息。
pub fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal();
            prev(info);
        }));
    });
}

/// 在 raw mode 下以最多 `timeout` 毫秒非阻塞探测终端背景色（OSC 11 + DA1 哨兵）。
/// 在不支持或超时的终端中安全回退为 `TerminalColors::default()`，绝不阻塞启动。
#[cfg(unix)]
pub fn probe_terminal_colors(timeout: Duration) -> TerminalColors {
    use std::os::fd::AsRawFd;

    let mut out = io::stdout();
    // 发送 OSC 10 (前景色) + OSC 11 (背景色) + OSC 4 (16 色板) + DA1 (\x1b[c) 作为有序结束哨兵
    let mut query = String::from("\x1b]10;?\x07\x1b]11;?\x07");
    for slot in 0..16u8 {
        query.push_str(&format!("\x1b]4;{slot};?\x07"));
    }
    query.push_str("\x1b[c");
    if out.write_all(query.as_bytes()).is_err() || out.flush().is_err() {
        return TerminalColors::default();
    }

    let stdin = io::stdin();
    let fd = stdin.as_raw_fd();
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 256];

    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    const POLLIN_EVENT: i16 = 0x0001;
    unsafe extern "C" {
        fn poll(fds: *mut PollFd, nfds: std::os::raw::c_uint, timeout: std::os::raw::c_int) -> std::os::raw::c_int;
        fn read(fd: std::os::raw::c_int, buf: *mut std::os::raw::c_void, count: usize) -> isize;
    }

    while Instant::now() < deadline {
        let rem_ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .min(i32::MAX as u128) as i32;
        let mut pfd = PollFd {
            fd,
            events: POLLIN_EVENT,
            revents: 0,
        };
        let ready = unsafe { poll(&mut pfd, 1, rem_ms) };
        if ready <= 0 {
            break;
        }
        let n = unsafe { read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n <= 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n as usize]);
        let s = String::from_utf8_lossy(&buf);
        // 收到 DA1 响应 (`ESC [ ? ... c`) 即说明前面所有 OSC 回复已到达完毕
        if s.contains("\x1b[?") && s.ends_with('c') {
            break;
        }
    }

    parse_osc_colors_response(&String::from_utf8_lossy(&buf))
}

#[cfg(not(unix))]
pub fn probe_terminal_colors(_timeout: Duration) -> TerminalColors {
    TerminalColors::default()
}

pub fn parse_osc_colors_response(raw: &str) -> TerminalColors {
    let mut result = TerminalColors::default();
    if let Some(rgb) = extract_osc_rgb(raw, "10;rgb:") {
        result.foreground = Some(rgb);
    }
    if let Some(rgb) = extract_osc_rgb(raw, "11;rgb:") {
        result.background = Some(rgb);
    }
    let mut pal = [
        Rgb8::new(0, 0, 0),
        Rgb8::new(205, 49, 49),
        Rgb8::new(13, 188, 121),
        Rgb8::new(229, 229, 16),
        Rgb8::new(36, 114, 200),
        Rgb8::new(188, 63, 188),
        Rgb8::new(17, 168, 205),
        Rgb8::new(229, 229, 229),
        Rgb8::new(102, 102, 102),
        Rgb8::new(241, 76, 76),
        Rgb8::new(35, 209, 139),
        Rgb8::new(245, 245, 67),
        Rgb8::new(59, 142, 234),
        Rgb8::new(214, 112, 214),
        Rgb8::new(41, 184, 219),
        Rgb8::new(255, 255, 255),
    ];
    let mut any_pal = false;
    for (idx, dst) in pal.iter_mut().enumerate() {
        let prefix = format!("4;{idx};rgb:");
        if let Some(rgb) = extract_osc_rgb(raw, &prefix) {
            *dst = rgb;
            any_pal = true;
        }
    }
    if any_pal {
        result.palette = Some(pal);
    }
    result
}

fn extract_osc_rgb(raw: &str, prefix: &str) -> Option<Rgb8> {
    let pos = raw.find(prefix)?;
    let rest = &raw[pos + prefix.len()..];
    let end = rest.find(['\x07', '\x1b']).unwrap_or(rest.len());
    let body = &rest[..end];
    let mut comps = body.split('/');
    let r = parse_hex_component(comps.next()?)?;
    let g = parse_hex_component(comps.next()?)?;
    let b = parse_hex_component(comps.next()?)?;
    Some(Rgb8::new(r, g, b))
}

fn parse_hex_component(hex: &str) -> Option<u8> {
    let trimmed = hex.trim();
    if trimmed.is_empty() {
        return None;
    }
    let val = u16::from_str_radix(trimmed, 16).ok()?;
    Some(match trimmed.len() {
        1 => (val * 17) as u8,
        2 => val as u8,
        3 | 4 => (val >> 8) as u8,
        _ => return None,
    })
}

/// RAII 终端守卫：支持 Fullscreen/Inline 双模进入、鼠标追踪开关与外部编辑器挂起恢复。
pub struct TerminalGuard {
    pub probed_colors: TerminalColors,
}

impl TerminalGuard {
    pub fn enter() -> io::Result<Self> {
        Self::enter_with_mode(ScreenMode::Fullscreen, true)
    }

    pub fn enter_with_mode(mode: ScreenMode, mouse_capture: bool) -> io::Result<Self> {
        install_panic_hook();
        enable_raw_mode()?;
        let probed_colors = probe_terminal_colors(Duration::from_millis(85));

        if let Err(err) = apply_terminal_modes(mode, mouse_capture) {
            let _ = disable_raw_mode();
            return Err(err);
        }
        Ok(Self { probed_colors })
    }
}

pub fn apply_terminal_modes(mode: ScreenMode, mouse_capture: bool) -> io::Result<()> {
    let mut out = io::stdout();
    match mode {
        ScreenMode::Fullscreen => {
            execute!(out, EnterAlternateScreen)?;
            if mouse_capture {
                execute!(out, EnableMouseCapture)?;
            } else {
                execute!(out, DisableMouseCapture)?;
            }
        }
        ScreenMode::Inline => {
            execute!(out, LeaveAlternateScreen, DisableMouseCapture)?;
        }
    }
    execute!(out, EnableBracketedPaste, EnableFocusChange)?;
    // Kitty 键盘协议与 CSI 2031 深浅色通知为尽力增强（不支持的终端安全忽略）
    let _ = execute!(
        out,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
        )
    );
    let _ = out.write_all(ENABLE_COLOR_SCHEME_NOTIFY.as_bytes());
    out.flush()
}

pub fn set_mouse_capture_enabled(enabled: bool) -> io::Result<()> {
    let mut out = io::stdout();
    if enabled {
        execute!(out, EnableMouseCapture)?;
    } else {
        execute!(out, DisableMouseCapture)?;
    }
    out.flush()
}

pub fn suspend_for_external_editor() -> io::Result<()> {
    let _ = disable_raw_mode();
    let mut out = io::stdout();
    let _ = execute!(
        out,
        PopKeyboardEnhancementFlags,
        DisableFocusChange,
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    out.flush()
}

pub fn resume_after_external_editor(mode: ScreenMode, mouse_capture: bool) -> io::Result<()> {
    enable_raw_mode()?;
    apply_terminal_modes(mode, mouse_capture)
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

fn truncate_utf8_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn encode_base64_str(input: &[u8]) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_osc10_osc11_and_osc4_hex16_responses() {
        let resp = "\x1b]10;rgb:e8e8/ecec/f2f2\x07\x1b]11;rgb:1616/1a1a/2222\x1b\\\x1b]4;4;rgb:3b3b/8282/f6f6\x07\x1b[?62;c";
        let colors = parse_osc_colors_response(resp);
        assert_eq!(colors.foreground, Some(Rgb8::new(0xe8, 0xec, 0xf2)));
        assert_eq!(colors.background, Some(Rgb8::new(0x16, 0x1a, 0x22)));
        let pal = colors.palette.expect("palette parsed");
        assert_eq!(pal[4], Rgb8::new(0x3b, 0x82, 0xf6));
    }

    #[test]
    fn formats_valid_osc7501_status_sequence() {
        let seq = format_osc_7501(
            ProgramStatusState::Blocked,
            Some("permission"),
            Some("Run\ncargo test"),
        );
        assert!(seq.starts_with("\x1b]7501;state=blocked:app=dsh-tui:kind=permission:msg="));
        assert!(seq.ends_with("\x1b\\"));
    }
}
